// Copyright LWPU Corporation 2015
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "MortonSorter.hpp"
#include "MortonSorterKernels.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <algorithm>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void MortonSorter::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.maxPrims >= 0);
    RT_ASSERT(cfg.inPrimOrder.getNumElems() == 0 || cfg.inPrimOrder.getNumElems() >= (size_t)cfg.maxPrims);
    RT_ASSERT(cfg.inModel.isValid());
    RT_ASSERT(cfg.inApexPointMap.getNumBytes() >= sizeof(ApexPointMap));

    if (cfg.bytesPerMortonCode != 4 && cfg.bytesPerMortonCode != 8)
        throw IlwalidValue(RT_EXCEPTION_INFO, "bytesPerMortonCode must be 4 or 8!", cfg.bytesPerMortonCode);

    // Set config and resize outputs.

    m_cfg = cfg;
    m_cfg.outMortonCodes.setNumElems((uint64_t)m_cfg.maxPrims * m_cfg.bytesPerMortonCode);
    m_cfg.outPrimOrder  .setNumElems(m_cfg.maxPrims);
    m_cfg.outPrimRange  .setNumElems(1);

    m_tmpMortonCodes    .assignNew((uint64_t)m_cfg.maxPrims * m_cfg.bytesPerMortonCode);
    m_tmpPrimOrder      .assignNew(m_cfg.maxPrims);
    m_sorterTemp        .assignNew(0);

    // Configure Sorter.
    {
        Sorter::Config c;
        c.lwca          = m_cfg.lwca;
        c.numItems      = m_cfg.maxPrims;
        c.bytesPerKey   = m_cfg.bytesPerMortonCode;
        c.bytesPerValue = sizeof(int);

        c.outKeys       = m_cfg.outMortonCodes;
        c.outValues     = m_cfg.outPrimOrder.reinterpretRaw();
        c.inKeys        = m_tmpMortonCodes;
        c.ilwalues      = m_tmpPrimOrder.reinterpretRaw();
        c.tempBuffer    = m_sorterTemp;

        m_sorter.configure(c);
    }

    m_cfg.tempBuffer
      .aggregate(m_tmpMortonCodes)
      .aggregate(m_tmpPrimOrder)
      .aggregate(m_sorterTemp);
}

//------------------------------------------------------------------------

void MortonSorter::execute(BufferRef<const Range> inPrimRange)
{
    RT_ASSERT(inPrimRange.getNumElems() <= 1);
    m_inPrimRange = inPrimRange;

    if (m_cfg.lwca)
    {
        m_cfg.lwca->beginTimer(getName());
        execDevice();
        m_cfg.lwca->endTimer();
    }
    else
    {
        execHost();
    }

    m_cfg.tempBuffer.markAsUninitialized();
}

//------------------------------------------------------------------------

void MortonSorter::execDevice(void)
{
    // Launch MortonCalc.
    {
        MortonCalcParams p  = {};
        p.outMortonCodes    = m_tmpMortonCodes.writeDiscardLWDA();
        p.outPrimOrder      = m_tmpPrimOrder.writeDiscardLWDA();
        p.outPrimRange      = m_cfg.outPrimRange.writeDiscardLWDA();
        p.inPrimOrder       = m_cfg.inPrimOrder.readLWDA();
        p.inModel           = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
        p.inApexPointMap    = m_cfg.inApexPointMap.readLWDA();
        p.inPrimRange       = m_inPrimRange.readLWDA();
        p.maxPrims          = m_cfg.maxPrims;

        LAUNCH(*m_cfg.lwca, MortonCalc, MORTON_CALC_WARPS_PER_BLOCK,
            m_cfg.maxPrims, p, m_cfg.bytesPerMortonCode);
    }

    // Execute Sorter.

    m_sorter.execute();
}

//------------------------------------------------------------------------

void MortonSorter::execHost(void)
{
    m_cfg.outMortonCodes    .writeDiscardHost();
    m_cfg.outPrimOrder      .writeDiscardHost();
    m_cfg.outPrimRange      .writeDiscardHost();
    m_cfg.inPrimOrder       .readHost();
    ModelPointers inModel   (m_cfg.inModel, MemorySpace_Host);
    m_cfg.inApexPointMap    .readHost();
    m_inPrimRange           .readHost();
    m_tmpMortonCodes        .writeDiscardHost();
    m_tmpPrimOrder          .writeDiscardHost();

    // Fetch primitive range and model AABB.

    Range   primRange       = (m_inPrimRange.getNumElems()) ? *m_inPrimRange : Range(0, m_cfg.maxPrims);
    AABB    modelAABB       = m_cfg.inApexPointMap->getAABB();
    float3  modelCenter     = modelAABB.getCenter();
    float   modelSizeMaxRcp = modelAABB.getSizeMaxRcp();

    // Write outPrimRange.

    *m_cfg.outPrimRange = Range(0, primRange.span());

    // Callwlate Morton code for each primitive.

    for (int outputIdx = 0; outputIdx < primRange.span(); outputIdx++)
    {
        // Determine input primitive index.

        int primIdx = outputIdx + primRange.start;
        if (m_cfg.inPrimOrder.getNumElems())
            primIdx = m_cfg.inPrimOrder[primIdx];

        // Callwlate primitive centroid relative to the model AABB.

        AABB primAABB = inModel.loadPrimitiveAABB(primIdx);
        float3 pos = AABB::transformRelative(primAABB.getCenter(), modelCenter, modelSizeMaxRcp);

        // Form 60-bit Morton code based on the 20 most significant bits.

        unsigned int mortonLo = 0u;
        unsigned int mortonHi = 0u;

        for (int c = 0; c < 3; c++)
        {
            unsigned int v  = std::min((unsigned int)(chooseComponent(pos, c) * (float)(1u << 20)), (1u << 20) - 1u);
            unsigned int lo = v & 0x3FFu;
            unsigned int hi = v >> 10;

            lo += lo << 16, lo &= 0xFF0000FFu, hi += hi << 16, hi &= 0xFF0000FFu;
            lo += lo << 8,  lo &= 0x0F00F00Fu, hi += hi << 8,  hi &= 0x0F00F00Fu;
            lo += lo << 4,  lo &= 0xC30C30C3u, hi += hi << 4,  hi &= 0xC30C30C3u;
            lo += lo << 2,  lo &= 0x49249249u, hi += hi << 2,  hi &= 0x49249249u;

            mortonLo = mortonLo * 2 + lo;
            mortonHi = mortonHi * 2 + hi;
        }

        // Output as 32-bit or 64-bit integer.

        if (m_cfg.bytesPerMortonCode == 4)
            m_tmpMortonCodes.reinterpret<unsigned int>()[outputIdx] = mortonHi;
        else
            m_tmpMortonCodes.reinterpret<unsigned long long>()[outputIdx] = ((unsigned long long)mortonHi << 32) | mortonLo;

        m_tmpPrimOrder[outputIdx] = primIdx;
    }

    // Fill the remaining Morton codes with dummy values.

    for (int outputIdx = primRange.span(); outputIdx < m_cfg.maxPrims; outputIdx++)
    {
        if (m_cfg.bytesPerMortonCode == 4)
            m_tmpMortonCodes.reinterpret<unsigned int>()[outputIdx] = ~0u;
        else
            m_tmpMortonCodes.reinterpret<unsigned long long>()[outputIdx] = ~0ull;

        m_tmpPrimOrder[outputIdx] = -1;
    }

    // Execute Sorter.

    m_sorter.execute();
}

//------------------------------------------------------------------------
