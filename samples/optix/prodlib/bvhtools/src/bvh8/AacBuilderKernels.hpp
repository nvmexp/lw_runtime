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

#pragma once
#include "WideBvhPlannerKernels.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define AAC_EXEC_WARPS_PER_BLOCK    2
#define AAC_EXEC_BLOCKS_PER_SM      NUMBLOCKS_MAXWELL(20)
#define AAC_EXEC_SIMD_MIN_LANES     8   // Fetch more work when # of lanes doing BVH traversal drops below this.
#define AAC_EXEC_SIMD_TIMEOUT       2   // Max. # of rounds to wait for SIMD partners to merge clusters.

//------------------------------------------------------------------------

struct AacClusterEntry          // 32 bytes
{
    int         childPairIdx;   // outNodePair or ~primRefIdx
    int         subtreePrims;   // Number of primitives in the subtree. Note: Not really needed - could remove.
    AABB        aabb;           // Bounding box.
};

//------------------------------------------------------------------------

struct AacExecParams
{
    PlannerBinaryNode*      outNodes;               // [max(inPrimRange->span() * 2, 2)]
    int*                    clusterIndices;         // [inPrimRange->span()]
    AacClusterEntry*        clusterEntries;         // [inPrimRange->span()]
    int*                    nodePrimRanges;         // [inPrimRange->span()], cleared to -1
    int*                    workCounter;            // [1], cleared to 0

    const int*              inPrimOrder;            // [inPrimRange->end]
    const unsigned char*    inMortonCodes;          // [inPrimRange->end * bytesPerMortonCode]
    ModelPointers           inModel;
    const ApexPointMap*     inApexPointMap;
    const Range*            inPrimRange;            // [1], NULL => (0, maxPrims)

    int                     maxPrims;
    float                   penaltyTermCoef;
};

//------------------------------------------------------------------------

struct AacExecWarpSh
{
    int     inPrimRangeStart;
    int     inPrimRangeEnd;
    int     simdTimer;          // How many rounds have we waited for SIMD partners?
    int     srcLanes[32];       // Compact list of lanes that want to merge clusters.

    int     nodePrimRangeX[32]; // Register spill.
    int     nodePrimRangeY[32]; // Register spill.
    int     mergeSplitIdx[32];  // Register spill.
};

//------------------------------------------------------------------------

bool launchAacExec  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const AacExecParams& p, int bytesPerMortonCode, int maxClusterSize);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
