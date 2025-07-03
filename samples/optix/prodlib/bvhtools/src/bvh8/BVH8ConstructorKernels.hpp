// Copyright LWPU Corporation 2016
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
#include <prodlib/bvhtools/include/BVH8Types.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Auxiliary per-node data produced by BVH8Constructor and consumed
// by BVH8Fitter.

struct BVH8NodeAux                  // 16 bytes
{
    int             numChildNodes;  // Number of child nodes. Temporarily used as a running counter by BVH8Fitter.
    union
    {
        struct
        {
            int     parentNodeIdx;  // Index of the parent node, -1 for the root. Temporarily overwritten with aabbHi by BVH8Fitter.
            int     rootPairIdx;    // First child of the node root in the binary node array. Only accessed by BVH8Constructor.
            int     firstRemapIdx;  // Index of the first remap entry covered by this subtree. Only accessed by BVH8Constructor.
        };
        float3      aabbHi;         // Max corner of node AABB. Only accessed by BVH8Fitter.
    };
};

//------------------------------------------------------------------------

#define BVH8CONSTRUCTOR_EXEC_WARPS_PER_BLOCK    2
#define BVH8CONSTRUCTOR_EXEC_BLOCKS_PER_SM      NUMBLOCKS_MAXWELL(32)
#define BVH8CONSTRUCTOR_EXEC_MAX_DEPTH          5
#define BVH8CONSTRUCTOR_EXEC_MAX_LEAF_SIZE      3

//------------------------------------------------------------------------

struct BVH8ConstructorExecParams
{
    BVH8Node*                   outNodes;               // [maxNodes]
    BVH8NodeAux*                outNodeAux;             // [maxNodes], cleared to -1
    int*                        outRemap;
    int*                        nodeCounter;            // [1], cleared to 0

    const int*                  inNumNodes;             // [1]
    const PlannerBinaryNode*    inBinaryNodes;

    int                         maxBranchingFactor;
};

//------------------------------------------------------------------------

bool launchBVH8ConstructorExec  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const BVH8ConstructorExecParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
