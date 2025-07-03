// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "InputAdapter.hpp"

#include <g_lwconfig.h>

#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>
#include <corelib/misc/String.h>

#if LWCFG( GLOBAL_ARCH_TURING )
namespace
{
Knob<bool> k_forceTTUPrimBitsFallback( RT_DSTRING( "rtcore.forceTTUPrimBitsFallback" ),
                                       false,
                                       RT_DSTRING( "Force fallback to 64-bit primBits for TTU" ) );
}
#endif

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

struct PrimBitsInfo
{
  PrimBitsInfo( int numPBits = 0, int pBSizeInBytes = 4, PrimBitsFormat inFormat = PRIMBITS_LEGACY_DIRECT_32 )
      : format( inFormat )
      , numPrimitiveBits( numPBits )
      , primBitsSizeInBytes( pBSizeInBytes )
  {}

  PrimBitsFormat format;
  int numPrimitiveBits;
  int primBitsSizeInBytes;
};

static uint32_t numPrimitiveBitsToMode( uint32_t numPrimitiveBits )
{
  return numPrimitiveBits ? ( numPrimitiveBits - 1 ) / 4 : 0;
}

#if LWCFG( GLOBAL_ARCH_TURING )
static uint32_t modeToNumPrimitiveBits( uint32_t mode )
{
  return mode * 4 + 4;
}
#endif

static PrimBitsInfo computeNumPrimitiveIndexBits( PrimBitsFormat formats, bool triangleBuild, int numGeometries, int numPrimitives )
{
  int numGeometryBits = 0;
  while( ( 1 << numGeometryBits ) < numGeometries )
    numGeometryBits++;

  int numPrimitiveBits = 0;
  while( ( 1 << numPrimitiveBits ) < numPrimitives )
    numPrimitiveBits++;

  if( ( formats & PRIMBITS_DIRECT_64 ) != 0 )
  {
    // Always use 64bit primBits in TTU item (AABB) case
    RT_ASSERT( numGeometryBits <= 31 && numPrimitiveBits <= 32 );  // high bit used for opaque flag
    return PrimBitsInfo(32, 8, PRIMBITS_DIRECT_64 );
  }

#if LWCFG( GLOBAL_ARCH_TURING )
  // Try to encode mode into 32bit primBits only in TTU triangle case.

  if( ( formats & PRIMBITS_DIRECT_32 ) != 0 )
  {
    // PRIMBITS_DIRECT_32 relies on PRIMBITS_INDIRECT_32_TO_64 as a fallback
    RT_ASSERT( ( formats & PRIMBITS_INDIRECT_32_TO_64 ) != 0 );

    // primBits = { int1 opaque, intM geometry, intN primitive, int3 mode }, opaque is used in coalescing
    // mode[M,N] = { 24:4, 20:8, 16:12, 12:16, 8:20, 4:24, 0:28, fallback to 64-bit primBits }
    // to decode numPrimitiveBits = mode * 4 + 4 if mode < 7
    int mode = numPrimitiveBitsToMode( numPrimitiveBits );
    int roundedNumPrimitiveBits = modeToNumPrimitiveBits( mode );
    PrimBitsFormat format;
    int primBitsSizeInBytes;

    if( mode >= 7 || numGeometryBits + roundedNumPrimitiveBits > 28 || k_forceTTUPrimBitsFallback.get()
      )
    {
      numPrimitiveBits    = 32;
      primBitsSizeInBytes = 8;  // fallback
      format = PRIMBITS_INDIRECT_32_TO_64;
    }
    else
    {
      // Enough bits for both geometry index and primitive index.
      // High 4 bits are used for opaque flag and index mode.
      RT_ASSERT( numGeometryBits + roundedNumPrimitiveBits <= 28 );
      numPrimitiveBits    = roundedNumPrimitiveBits;
      primBitsSizeInBytes = 4;
      format = PRIMBITS_DIRECT_32;
    }

    return PrimBitsInfo(numPrimitiveBits, primBitsSizeInBytes, format);
  }
#endif // GLOBAL_ARCH_TURING

  RT_ASSERT( numGeometryBits + numPrimitiveBits <= 31 );

  return PrimBitsInfo(numPrimitiveBits, 4, PRIMBITS_LEGACY_DIRECT_32);
}



void InputAdapter::configure(const Config& cfg)
{
    m_cfg = cfg;

    m_model.clear();

    const InputBuffers& input = *m_cfg.inBuffers;

    // Compute primBits format

    PrimBitsInfo primBitsInfo;
    if (input.inputType == IT_INSTANCE)
    {
        m_coalesced = false;
    }
    else
    {
        primBitsInfo = computeNumPrimitiveIndexBits( m_cfg.allowedPrimBitsFormats, input.inputType == IT_TRI, input.numArrays, input.maxPrimsInSingleGeometry );

        m_numPrimitiveIndexBits = primBitsInfo.numPrimitiveBits;
        m_primBitsFormat = primBitsInfo.format;

        m_primBitsFlags = 0;
        if( input.numArrays == 1 )
        {
            if( input.inputType == IT_TRI || input.inputType == IT_AABB )
            {
                uint64_t opaque = ( input.inputType == IT_TRI && input.trianglesDescArray[0].isOpaque() )
                                  || ( input.inputType == IT_AABB && input.aabbsArray[0].isOpaque() ) ? 1 : 0;
                m_primBitsFlags = opaque << ( (primBitsInfo.primBitsSizeInBytes == 8 ) ? 63 : 31 );

                if( m_primBitsFormat == PRIMBITS_DIRECT_32 && input.inputType == IT_TRI )
                    m_primBitsFlags = numPrimitiveBitsToMode( primBitsInfo.numPrimitiveBits ); // for triangle case encode the mode and ignore the opaque
            }
        }

        // Check coalescing critera

        // The builder can lwrrently only handle a single input with a float3 vertex buffer, with or without an index buffer (uint32).
        // Anything other than that needs to be coalesced.
        m_coalesced = input.inputType == IT_TRI && (
            input.numInitialArrays > 1 ||                               // Single geometry input
            input.trianglesDescArray[0].getVertexByteSize() != 4 ||     // 32-bit float
            input.trianglesDescArray[0].getVertexDim() != 3 ||          // 3-component float
            ( input.trianglesDescArray[0].hasIndexBuffer() &&
              input.trianglesDescArray[0].getIndexByteSize() != 4 ) ||  // either uint32 indices or no index buffer
            input.trianglesPtrArray[0].transform ||                     // No transform
            m_primBitsFormat == PRIMBITS_INDIRECT_32_TO_64
          );

        // Always coalesce AABBs, even if it's just one (need to colwert AABBs to PrimitiveAABBs)
        if( input.inputType == IT_AABB )
            m_coalesced = true;
    }

    // Input coalescing

    InputArrayIndexBuffers inputArrayIndexing;

    // TODO: Should be able to skip this for refit once we can read buffer pointers from AS header
    if( input.inputType != IT_INSTANCE && m_coalesced )
    {
        InputArrayIndexer::Config c;
        c.lwca                  = (m_cfg.useLwda) ? m_cfg.lwdaUtils : NULL;

        c.inBuffers             = &input;

        c.outArrayBaseGlobalIndex  = m_cfg.outArrayBaseGlobalIndex;
        c.outArrayTransitionBits   = m_cfg.outArrayTransitionBits;
        c.outBlockStartArrayIndex  = m_cfg.outBlockStartArrayIndex;
        c.outGeometryIndexArray    = m_cfg.outGeometryIndexArray; 

        m_inputArrayIndexer.configure(c);

        inputArrayIndexing.arrayBaseGlobalIndex = m_cfg.outArrayBaseGlobalIndex;
        inputArrayIndexing.arrayTransitionBits  = m_cfg.outArrayTransitionBits;
        inputArrayIndexing.blockStartArrayIndex = m_cfg.outBlockStartArrayIndex;
        inputArrayIndexing.geometryIndexArray   = m_cfg.outGeometryIndexArray; 
    }

    // ApexPointMapConstructor and MortonSorter take a model. Create one from 
    // input.

    m_model.numPrimitives = input.numPrimitives;
    m_model.inputType = input.inputType;

    if( input.inputType == IT_INSTANCE )
    {
      InstanceDataAdapter::Config c;
      c.useLwda             = m_cfg.useLwda;
      c.lwdaUtils           = m_cfg.lwdaUtils;
      c.numInstances        = input.numPrimitives;
      c.outBvhInstanceData  = m_cfg.outBvhInstanceData;
      c.outWorldSpaceAabbs  = m_cfg.outAabbs;
      c.inInstanceDescs     = input.instanceDescs;

      m_instanceDataGen.configure( c );

      m_model.aabbs     = m_cfg.outAabbs;
    } 
    else if( input.inputType == IT_AABB )
    {
      AabbAdapter::Config c;
      c.useLwda             = m_cfg.useLwda;
      c.lwdaUtils           = m_cfg.lwdaUtils;
      c.inBuffers           = &input;
      c.primitiveIndexBits  = m_numPrimitiveIndexBits;
      c.motionSteps         = m_cfg.motionSteps;
      c.outAabbArray        = m_cfg.outAabbArray;
      c.outPrimAabbs        = m_cfg.outAabbs;
      c.outPrimBits         = m_primBitsFormat == PRIMBITS_INDIRECT_32_TO_64 ? m_cfg.outIndirectPrimBits : m_cfg.outTmpPrimBits;
      c.computePrimBits     = m_cfg.usePrimBits && !m_cfg.refitOnly; // TODO: This can optionally be deferred until GatherPrimBits
      c.primBitsFormat      = m_primBitsFormat;
      c.inArrayIndexing     = inputArrayIndexing;

      m_aabbAdapter.configure( c );

      m_model.aabbs               = m_cfg.outAabbs;
      m_model.inputArrayIndexing  = inputArrayIndexing;
    }
    else if( input.inputType == IT_TRI )
    {
      if (m_coalesced)
      {
          TriangleAdapter::Config c;
          c.useLwda                 = m_cfg.useLwda;
          c.lwdaUtils               = m_cfg.lwdaUtils;
          c.inBuffers               = &input;
          c.primitiveIndexBits      = m_numPrimitiveIndexBits;
          c.outTrianglesDescArray   = m_cfg.outTriangleDescArray;
          c.outTrianglesPtrArray    = m_cfg.outTrianglePtrArray;
          c.outVertices             = m_cfg.outCoalescedVertices;
          c.outPrimBits             = m_primBitsFormat == PRIMBITS_INDIRECT_32_TO_64 ? m_cfg.outIndirectPrimBits : m_cfg.outTmpPrimBits;
          c.computePrimBits         = m_cfg.usePrimBits && !m_cfg.refitOnly; // TODO: This can optionally be deferred until GatherPrimBits
          c.refitOnly               = m_cfg.refitOnly;
          c.inArrayIndexing         = inputArrayIndexing;
          c.primBitsFormat          = m_primBitsFormat;

          m_triangleAdapter.configure( c );

          m_model.vertices          = m_cfg.outCoalescedVertices.reinterpret<float>();
          m_model.indices           = EmptyBuf;

          m_model.indexStride     = 0;
          m_model.vertexStride    = sizeof(float) * 3;

          m_model.inputArrayIndexing  = inputArrayIndexing;
      }
      else
      {
          const InputTrianglesDesc &desc = input.trianglesDescArray[0];
          const InputTrianglesPointers &ptrs = input.trianglesPtrArray[0];
          m_model.indexStride     = desc.getIndexStride();
          m_model.vertexStride    = desc.getVertexStride();

          if (desc.getNumVertices() > 0)
          {
              size_t vertexBytes = (desc.getNumVertices() - 1) * (size_t)desc.getVertexStride() + desc.getVertexByteSize() * desc.getVertexDim();
              m_model.vertices.assignExternal((const float *)ptrs.vertices, vertexBytes / sizeof(float), m_cfg.lwdaUtils ? MemorySpace_LWDA : MemorySpace_Host);
              m_model.vertices.materialize(m_cfg.lwdaUtils);
          }
          if (desc.getIndexByteSize() != 0 && desc.getNumIndices() > 0)
          {
              size_t indexBytes = ((desc.getNumIndices() / 3) - 1) * (size_t)desc.getIndexStride() + desc.getIndexByteSize() * 3;
              m_model.indices.assignExternal((const char *)ptrs.indices, indexBytes, m_cfg.lwdaUtils ? MemorySpace_LWDA : MemorySpace_Host);
              m_model.indices.materialize(m_cfg.lwdaUtils);
          }
      }

    }
    else
    {
      RT_ASSERT( input.inputType == IT_PRIMAABB );
      m_model.aabbs = input.primAabbs;
    }

}

//------------------------------------------------------------------------

void InputAdapter::execute(void)
{
    if (m_coalesced && !m_cfg.refitOnly)
      m_inputArrayIndexer.execute();

    // Create instance data, aabbs and tris
    if( m_cfg.inBuffers->inputType == IT_INSTANCE )
        m_instanceDataGen.execute();
    else if( m_cfg.inBuffers->inputType == IT_AABB )
        m_aabbAdapter.execute();
    else if( m_cfg.inBuffers->inputType == IT_TRI && m_coalesced )
        m_triangleAdapter.execute();

}

//------------------------------------------------------------------------
