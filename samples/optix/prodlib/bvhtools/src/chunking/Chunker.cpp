// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "Chunker.hpp"
#include "ChunkerKernels.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/bvhtools/src/common/Utils.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
 
using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void Chunker::configure(const Config& cfg)
{
    RT_ASSERT(cfg.numPrims >= 0);
    RT_ASSERT(cfg.maxChunkPrims >= 1);
    RT_ASSERT(cfg.preferredTopTreePrims >= 1);
    RT_ASSERT(cfg.inPrimMorton.getNumElems() >= (size_t)cfg.numPrims);

    m_cfg = cfg;

    // Callwlate maximum number of chunks and top-level primitives.

    int minChunks = (m_cfg.numPrims - 1) / m_cfg.maxChunkPrims + 1;
    int maxChunks = minChunks * (60 + 4); // 60 is the number of Morton bits
    int maxTopTreePrims = max(m_cfg.preferredTopTreePrims, maxChunks * 2); // See the comment in computeLwtoffLevel().

    // Resize output buffers.

    m_cfg.outPrimRanges     .setNumElems(maxChunks);
    m_cfg.outNumChunks      .setNumElems(1);
    m_cfg.outLwtoffLevel    .setNumElems(1);
    m_cfg.allocTrimmedAABBs .setNumElems(maxTopTreePrims);
}

void Chunker::execute(void)
{
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
}

void Chunker::execDevice(void)
{
    // Clear outNumChunks.

    m_cfg.outNumChunks.clearLWDA(0);

    // Launch ChunkerIntervals.
    {
        ChunkerIntervalsParams p = {};
        p.outPrimRanges         = m_cfg.outPrimRanges.writeDiscardLWDA();
        p.outNumChunks          = m_cfg.outNumChunks.writeDiscardLWDA();
        p.outLwtoffLevels       = m_cfg.outLwtoffLevel.writeDiscardLWDA();
        p.inPrimMorton          = m_cfg.inPrimMorton.readLWDA();
        p.numPrims              = m_cfg.numPrims;
        p.maxChunkPrims         = m_cfg.maxChunkPrims;
        p.preferredTopTreePrims = m_cfg.preferredTopTreePrims;

        LAUNCH(*m_cfg.lwca, ChunkerIntervalsA, CHUNKER_INTERVAL_WARPS_PER_BLOCK, m_cfg.numPrims - 1, p);
        LAUNCH(*m_cfg.lwca, ChunkerIntervalsB, 1, 1, p);
    }
}

//------------------------------------------------------------------------

static void computePartitioning(const Chunker::Config& cfg, BufferRef<const unsigned long long> mortonCodes)
{
    int numChunks = 0;
    for(int i = 0; i < cfg.numPrims - 1; i++)
    {
        int s, dir, j;
        computeInterval( i, cfg.numPrims, mortonCodes.getLwrPtr(), s, dir, j );

        int childA = i + s * dir + min(dir, 0);
        int childB = childA + 1;

        // The range [min(i,j), max(i,j)] essentially indicates the range of triangles covered by a (hypothetical) internal node in a (hypothetical) BVH. 
        // Furthermore, [min(i,j), childA] and [childB, max(i,j)] indicate the ranges covered by its children.   
        // Whenever a parent range is >N, but a child range is <=N, the child range corresponds to a valid and maximal group.

        Range parentRange( min(i,j), max(i,j) + 1 );
        Range child0Range( min(i,j), childA   + 1 );
        Range child1Range( childB,   max(i,j) + 1 );

        if ((size_t)parentRange.span() > (size_t)cfg.maxChunkPrims)
        {
            if ((size_t)child0Range.span() <= (size_t)cfg.maxChunkPrims)
                cfg.outPrimRanges[numChunks++] = child0Range;

            if ((size_t)child1Range.span() <= (size_t)cfg.maxChunkPrims)
                cfg.outPrimRanges[numChunks++] = child1Range;
        }
    }

    if (numChunks == 0)
    {
        RT_ASSERT(cfg.numPrims <= cfg.maxChunkPrims);
        cfg.outPrimRanges[numChunks++] = Range(0, cfg.numPrims);
    }

    *cfg.outNumChunks = numChunks;
}

//------------------------------------------------------------------------------

void Chunker::execHost(void)
{
    m_cfg.outPrimRanges     .writeDiscardHost();
    m_cfg.outNumChunks      .writeDiscardHost();
    m_cfg.outLwtoffLevel    .writeDiscardHost();
    m_cfg.inPrimMorton      .readHost();

    computePartitioning(m_cfg, m_cfg.inPrimMorton);
    *m_cfg.outLwtoffLevel = computeLwtoffLevel(m_cfg.preferredTopTreePrims, *m_cfg.outNumChunks);
}

//------------------------------------------------------------------------------
