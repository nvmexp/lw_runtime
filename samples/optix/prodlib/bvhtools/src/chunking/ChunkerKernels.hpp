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

#pragma once
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define CHUNKER_INTERVAL_WARPS_PER_BLOCK    4
#define CHUNKER_INTERVAL_BLOCKS_PER_SM      NUMBLOCKS_KEPLER(16)

//------------------------------------------------------------------------

struct ChunkerIntervalsParams
{
    Range*                      outPrimRanges;    // [maxChunks]
    int*                        outNumChunks;     // [1]  
    int*                        outLwtoffLevels;  // [1]

    const unsigned long long*   inPrimMorton;       // [numPrims]

    int                         numPrims;
    int                         maxChunkPrims;
    int                         preferredTopTreePrims;
};

//------------------------------------------------------------------------

bool launchChunkerIntervalsA    (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const ChunkerIntervalsParams& p);
bool launchChunkerIntervalsB    (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const ChunkerIntervalsParams& p);

//------------------------------------------------------------------------

static INLINE int computeLwtoffLevel(int preferredTopTreePrims, int numChunks)
{
  // For a given cutoff level, TreeTopTrimmer will produce at most 2^(cutoff+1) AABBs per chunk.
  // We want choose the cutoff level so that the total number of AABBs is bounded by preferredTopTreePrims.

  // Because we have to clamp the cutoff to 0, the total number of AABBs can rise higher than preferredTopTreePrims
  // in extreme cases. However, it can never rise higher than max(preferredTopTreePrims, maxChunks * 2).
    
  // The maximum value of 12 was determined empirically to be a point beyond which there were only
  // minimal changes in performance. TODO: Revisit this.

  return clamp(0, 12, findLeadingOne(preferredTopTreePrims / (2 * numChunks)));
}

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
