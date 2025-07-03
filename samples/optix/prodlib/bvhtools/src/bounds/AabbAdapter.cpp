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

#include "AabbAdapter.hpp"
#include "AabbAdapterKernels.hpp"
#include "ApexPointMapLookup.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/bvhtools/src/misc/InputArrayIndexer.hpp>
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>
#include <prodlib/bvhtools/src/common/Utils.hpp>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void AabbAdapter::configure(const Config& cfg)
{
    RT_ASSERT(cfg.inBuffers->numPrimitives >= 0);
    RT_ASSERT(cfg.motionSteps >= 1 );

    // Set config and resize outputs.

    m_cfg = cfg;
    m_cfg.outAabbArray.setNumElems(m_cfg.inBuffers->numArrays);
    m_cfg.outPrimAabbs.setNumElems(m_cfg.inBuffers->numPrimitives);
    if (m_cfg.computePrimBits)
    {
        int primBitsSizeInBytes = m_cfg.primBitsFormat == PRIMBITS_DIRECT_32 || m_cfg.primBitsFormat == PRIMBITS_LEGACY_DIRECT_32 ? 4 : 8;
        m_cfg.outPrimBits.reinterpret<char>().setNumElems(m_cfg.inBuffers->numPrimitives * primBitsSizeInBytes);
    }
}

//------------------------------------------------------------------------

void AabbAdapter::execute(void)
{
    // Copy input data array to target

    MemorySpace memSpace = m_cfg.useLwda ? MemorySpace_LWDA : MemorySpace_Host;

    memcpyInlineWAR((char *) m_cfg.outAabbArray.writeDiscard(memSpace), (char *) m_cfg.inBuffers->aabbsArray.readHost(), sizeof(InputAABBs) * m_cfg.inBuffers->numArrays, m_cfg.lwdaUtils);

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

void AabbAdapter::execDevice(void)
{
    // Launch AabbAdapterExec.
    {
        AabbAdapterExecParams p = {};

        p.numPrimitives       = m_cfg.inBuffers->numPrimitives;
        p.primitiveIndexBits  = m_cfg.primitiveIndexBits;
        p.computePrimBits     = m_cfg.computePrimBits ? 1 : 0;
        p.primBitsFormat      = m_cfg.primBitsFormat;
        p.motionSteps         = m_cfg.motionSteps;

        p.aabbsArray          = m_cfg.outAabbArray.readLWDA();

        p.inArrayIndexing     = InputArrayIndexPointers(m_cfg.inArrayIndexing, MemorySpace_LWDA);

        p.outPrimAabbs        = m_cfg.outPrimAabbs.writeDiscardLWDA();
        p.outPrimBits         = m_cfg.computePrimBits ? m_cfg.outPrimBits.writeDiscardLWDA() : NULL;

        LAUNCH(*m_cfg.lwdaUtils, AabbAdapterExec, AABB_ADAPTER_EXEC_WARPS_PER_BLOCK,
            m_cfg.inBuffers->numPrimitives, p);
    }
}

//------------------------------------------------------------------------

void AabbAdapter::execHost(void)
{
    Config& p = m_cfg;
    InputArrayIndexPointers inArrayIndexing(m_cfg.inArrayIndexing, MemorySpace_Host);
    const InputAABBs* aabbsArray = p.outAabbArray.readHost();
    p.outPrimAabbs.writeDiscardHost();
    int* outPrimBits = NULL;
    if (p.computePrimBits)
      outPrimBits = (int *) p.outPrimBits.writeDiscardHost();

    // Read back data to host side
    MemorySpace memSpace = m_cfg.lwdaUtils ? MemorySpace_LWDA : MemorySpace_Host;
    std::vector<BufferRef<const char>> hostAabbs(p.inBuffers->numArrays);
    for (int i = 0; i < p.inBuffers->numArrays; i++)
    {
        const InputAABBs &inputAabbs = m_cfg.outAabbArray[i];
        hostAabbs[i].assignExternal( (const char *) inputAabbs.aabbs, p.inBuffers->numPrimsArray[i] * inputAabbs.strideInBytes * p.motionSteps, memSpace );
        hostAabbs[i].materialize(m_cfg.lwdaUtils).readHost();
    }

    // Colwert to PrimitiveAABBs and fill in initial primBits
    for (int i = 0; i < p.inBuffers->numPrimitives; i++)
    {
        int arrayIndex = inArrayIndexing.getArrayIndex(i);
        int localPrimitiveIndex = inArrayIndexing.getLocalPrimitiveIndex(i, arrayIndex);

        const InputAABBs &inputAabbs = aabbsArray[arrayIndex];
        const char *aabbPtr = &hostAabbs[arrayIndex][0];

        const AABB &aabb = *( make_ptr<AABB>( aabbPtr, localPrimitiveIndex * inputAabbs.strideInBytes * p.motionSteps ) );

        p.outPrimAabbs[i].lox = aabb.lo.x;
        p.outPrimAabbs[i].loy = aabb.lo.y;
        p.outPrimAabbs[i].loz = aabb.lo.z;
        p.outPrimAabbs[i].hix = aabb.hi.x;
        p.outPrimAabbs[i].hiy = aabb.hi.y;
        p.outPrimAabbs[i].hiz = aabb.hi.z;
        p.outPrimAabbs[i].primitiveIdx = i;
        p.outPrimAabbs[i].pad = 0;

        if (p.computePrimBits)
        {
            int geometryIndex = inArrayIndexing.getGeometryIndex(arrayIndex);

            if( p.primBitsFormat == PRIMBITS_DIRECT_64 )
            {
                // TODO: This may become the only valid format for AABBs
                ((uint64_t *) outPrimBits)[i] = (inputAabbs.isOpaque() ? ( uint64_t( 1 ) << 63 ) : 0) | ( uint64_t( geometryIndex ) << 32) | localPrimitiveIndex;
            }
            else if( p.primBitsFormat == PRIMBITS_LEGACY_DIRECT_32 )
            {
                // TODO: This will be removed
                ((uint32_t *) outPrimBits)[i] = (inputAabbs.isOpaque() ? 0x80000000 : 0) | (geometryIndex << p.primitiveIndexBits) | localPrimitiveIndex;
            }
            else
            {
                RT_ASSERT_MSG(false, "AabbAdapter primBits format must be either PRIMBITS_DIRECT_64 or PRIMBITS_LEGACY_DIRECT_32");
            }

        }

    }

}
