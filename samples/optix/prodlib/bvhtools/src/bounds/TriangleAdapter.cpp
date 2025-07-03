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

#include "TriangleAdapter.hpp"
#include "TriangleAdapterKernels.hpp"
#include "ApexPointMapLookup.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/bvhtools/src/misc/InputArrayIndexer.hpp>
#include <prodlib/bvhtools/src/common/Utils.hpp>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void TriangleAdapter::configure(const Config& cfg)
{
    // Set config and resize outputs.

    m_cfg = cfg;
    m_cfg.outVertices.setNumElems(m_cfg.inBuffers->numPrimitives * 3);
    m_cfg.outTrianglesPtrArray.setNumElems(m_cfg.inBuffers->numArrays);
    m_cfg.outTrianglesDescArray.setNumElems(m_cfg.inBuffers->numArrays);
    if (m_cfg.computePrimBits)
    {
        int primBitsSizeInBytes = m_cfg.primBitsFormat == PRIMBITS_DIRECT_32 || m_cfg.primBitsFormat == PRIMBITS_LEGACY_DIRECT_32 ? 4 : 8;
        m_cfg.outPrimBits.reinterpret<char>().setNumElems(m_cfg.inBuffers->numPrimitives * primBitsSizeInBytes);
    }
}

//------------------------------------------------------------------------

void TriangleAdapter::execute(void)
{
    // Write input data array
    {
        MemorySpace memSpace = m_cfg.useLwda ? MemorySpace_LWDA : MemorySpace_Host;

        // Pointers to vertices, indices and transforms are always copied
        memcpyInlineWAR((char *) m_cfg.outTrianglesPtrArray.writeDiscard(memSpace), (char *) m_cfg.inBuffers->trianglesPtrArray.readHost(), sizeof(InputTrianglesPointers) * m_cfg.inBuffers->numArrays, m_cfg.lwdaUtils);

        // Descriptors don't need to be updated for refits since they are not allowed to change
        if (!m_cfg.refitOnly)
            memcpyInlineWAR((char *) m_cfg.outTrianglesDescArray.writeDiscard(memSpace), (char *) m_cfg.inBuffers->trianglesDescArray.readHost(), sizeof(InputTrianglesDesc) * m_cfg.inBuffers->numArrays, m_cfg.lwdaUtils);
    }

    if (m_cfg.useLwda)
    {
        m_cfg.lwdaUtils->beginTimer(getName());
        execDevice();
        m_cfg.lwdaUtils->endTimer();
    }
    else
    {
        execHost();
    }
}

//------------------------------------------------------------------------

void TriangleAdapter::execDevice(void)
{
    // Launch TriangleAdapterExec.
    {
        TriangleAdapterExecParams p = {};

        p.numPrimitives           = m_cfg.inBuffers->numPrimitives;
        p.primitiveIndexBits      = m_cfg.primitiveIndexBits;
        p.computePrimBits         = m_cfg.computePrimBits ? 1 : 0;
        p.primBitsFormat          = m_cfg.primBitsFormat;

        p.trianglesDescArray      = m_cfg.outTrianglesDescArray.readLWDA();
        p.trianglesPtrArray       = m_cfg.outTrianglesPtrArray.readLWDA();

        p.inArrayIndexing         = InputArrayIndexPointers(m_cfg.inArrayIndexing, MemorySpace_LWDA);

        p.outVertices             = m_cfg.outVertices.writeDiscardLWDA();
        p.outPrimBits             = m_cfg.computePrimBits ? m_cfg.outPrimBits.writeDiscardLWDA() : NULL;

        LAUNCH(*m_cfg.lwdaUtils, TriangleAdapterExec, TRIANGLE_ADAPTER_EXEC_WARPS_PER_BLOCK, m_cfg.inBuffers->numPrimitives, p);
    }
}

//------------------------------------------------------------------------

static float3 mult(const float *M, float2 v)
{
  return make_float3(
    M[0 ]*v.x + M[1 ]*v.y + M[3 ],
    M[4 ]*v.x + M[5 ]*v.y + M[7 ],
    M[8 ]*v.x + M[9 ]*v.y + M[11]
  );
}

static float3 mult(const float *M, float3 v)
{
  return make_float3(
    M[0 ]*v.x + M[1 ]*v.y + M[2 ]*v.z + M[3 ],
    M[4 ]*v.x + M[5 ]*v.y + M[6 ]*v.z + M[7 ],
    M[8 ]*v.x + M[9 ]*v.y + M[10]*v.z + M[11]
  );
}

void TriangleAdapter::execHost(void)
{
    Config& p = m_cfg;
    InputArrayIndexPointers inArrayIndexing(m_cfg.inArrayIndexing, MemorySpace_Host);
    const InputTrianglesDesc* inTrianglesDescs = p.outTrianglesDescArray.readHost();
    const InputTrianglesPointers* inTrianglesPtrs = p.outTrianglesPtrArray.readHost();
    p.outVertices.writeDiscardHost();
    int* outPrimBits = NULL;
    if (p.computePrimBits)
      outPrimBits = (int *) p.outPrimBits.writeDiscardHost();

    // Read back data to host side

    int numInputs = p.inBuffers->numArrays;
    MemorySpace memSpace = m_cfg.lwdaUtils ? MemorySpace_LWDA : MemorySpace_Host;
    std::vector<BufferRef<const char>> hostVertices(numInputs), hostIndices(numInputs);
    std::vector<BufferRef<const float>> hostTransform(numInputs);
    for (int i = 0; i < numInputs; i++)
    {
        const InputTrianglesDesc &desc = inTrianglesDescs[i];
        const InputTrianglesPointers &ptrs = inTrianglesPtrs[i];
        hostVertices[i].assignExternal( (const char *) ptrs.vertices, desc.getNumVertices() * desc.getVertexStride(), memSpace );
        hostVertices[i].materialize(m_cfg.lwdaUtils).readHost();
        if ( ptrs.indices && desc.hasIndexBuffer() )
        {
            hostIndices[i].assignExternal( (const char *) ptrs.indices, (desc.getNumIndices() / 3) * desc.getIndexStride(), memSpace ); // TODO: Doesn't work with tri strips
            hostIndices[i].materialize(m_cfg.lwdaUtils).readHost();
        }
        if( ptrs.transform )
        {
            hostTransform[i].assignExternal( ptrs.transform, 12, memSpace );
            hostTransform[i].materialize(m_cfg.lwdaUtils).readHost();
        }
    }

    // No transform - use identity
    static const float4 identityTransform[3] = { 
        make_float4( 1, 0, 0, 0 ),
        make_float4( 0, 1, 0, 0 ),
        make_float4( 0, 0, 1, 0 )
    };

    // Compute flattened buffer with transformed float3 vertices, and compute primBits

    for (int i = 0; i < p.inBuffers->numPrimitives; i++)
    {
        int arrayIndex = inArrayIndexing.getArrayIndex(i);
        int localPrimitiveIndex = inArrayIndexing.getLocalPrimitiveIndex(i, arrayIndex);

        const InputTrianglesDesc &desc = inTrianglesDescs[arrayIndex];
        const char *vertices = &hostVertices[arrayIndex][0];
        const char *indices = NULL;
        if (hostIndices[arrayIndex].isMaterialized())
            indices = &hostIndices[arrayIndex][0];
        const float *transform = NULL;
        if (hostTransform[arrayIndex].isMaterialized())
            transform = &hostTransform[arrayIndex][0];

        // TODO: Handle triangle strips

        uint3 idx;
        if (indices)
        {
            const char* indexPtr = (const char *) indices + localPrimitiveIndex * desc.getIndexStride();
            if (desc.getIndexByteSize() == 2)
            {
                const unsigned short* int16Indices = (const unsigned short*) indexPtr;
                idx = { int16Indices[0], int16Indices[1], int16Indices[2] };
            }
            else // tris.getIndexByteSize() == 4
            {
                const unsigned int* int32Indices = (const unsigned int*) indexPtr;
                idx = { int32Indices[0], int32Indices[1], int32Indices[2] };
            }
        }
        else
            idx = make_uint3( 3*localPrimitiveIndex+0, 3*localPrimitiveIndex+1, 3*localPrimitiveIndex+2 );

        if (!transform)
          transform = (const float *) identityTransform;

        const char* vertexPtr[] = {
            (const char *) vertices + idx.x * desc.getVertexStride(),
            (const char *) vertices + idx.y * desc.getVertexStride(),
            (const char *) vertices + idx.z * desc.getVertexStride()
        };

        InputTrianglesDesc::VertexFormat format = desc.getVertexFormat();

        switch (format)
        {
            case InputTrianglesDesc::VertexFormat_AlignedFloat4:
            case InputTrianglesDesc::VertexFormat_Float3:
                for (int vtx = 0; vtx < 3; vtx++)
                    p.outVertices[i * 3 + vtx] = mult(transform, *(const float3 *) vertexPtr[vtx]);
                break;
            case InputTrianglesDesc::VertexFormat_Float2:
                for (int vtx = 0; vtx < 3; vtx++)
                    p.outVertices[i * 3 + vtx] = mult(transform, *(const float2 *) vertexPtr[vtx]);
                break;
            case InputTrianglesDesc::VertexFormat_Half3:
            case InputTrianglesDesc::VertexFormat_Half2:
                //const half *half_vertex = (const half *) &vertexPtr[vtx];
                // TODO: Colwert half to float
                //p.outVertices[i * 3 + vtx] = mult(transform, float_vertex);
                RT_ASSERT_MSG(false, "Half-to-float colwersion not yet implemented in the CPU-path.");
                break;
            default:
                RT_ASSERT_MSG(false, "Unsupported vertex format");
        }

        if ( p.computePrimBits )
        {
            int geometryIndex = inArrayIndexing.getGeometryIndex(arrayIndex);

            if( p.primBitsFormat == PRIMBITS_INDIRECT_32_TO_64 )
            {
                // Baked 32-bit primBits into triangles with indirection to 64-bit primBits array
                ((uint64_t *) outPrimBits)[i] = (desc.isOpaque() ? ( uint64_t( 1 ) << 63 ) : 0) | ( uint64_t( geometryIndex ) << 32) | localPrimitiveIndex;
            }
            else if( p.primBitsFormat == PRIMBITS_DIRECT_32 )
            {
                // Baked 32-bit primBits into triangles
                int          mode                    = p.primitiveIndexBits ? ( ( p.primitiveIndexBits - 1 ) / 4 ) : 0;
                int          roundedNumPrimitiveBits = mode * 4 + 4;
                unsigned int primBits =
                    ( desc.isOpaque() ? 0x80000000 : 0 ) | ( geometryIndex << ( roundedNumPrimitiveBits + 3 ) ) | ( localPrimitiveIndex << 3 ) | mode;
                ( (uint32_t*) outPrimBits )[i] = primBits;
            }
            else if( p.primBitsFormat == PRIMBITS_DIRECT_64)
            {
                // TODO: This will be used by BVH8
            }
            else if( p.primBitsFormat == PRIMBITS_LEGACY_DIRECT_32 )
            {
                // TODO: This will be removed
                ((uint32_t *) outPrimBits)[i] = (desc.isOpaque() ? 0x80000000 : 0) | (geometryIndex << p.primitiveIndexBits) | localPrimitiveIndex;
            }

        }

    }

}
