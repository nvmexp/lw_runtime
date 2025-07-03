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
#include <prodlib/bvhtools/src/common/Intrinsics.hpp>
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define TRBVH_RADIXTREE_WARPS_PER_BLOCK 4
#define TRBVH_RADIXTREE_BLOCKS_PER_SM   NUMBLOCKS_KEPLER(16)

#define TRBVH_FIT_WARPS_PER_BLOCK       4
#define TRBVH_FIT_BLOCKS_PER_SM         NUMBLOCKS_KEPLER(16)

#define TRBVH_OPT_WARPS_PER_BLOCK       8
#define TRBVH_OPT_BLOCKS_PER_SM         NUMBLOCKS_KEPLER(8)

#define TRBVH_COLLAPSE_WARPS_PER_BLOCK  4
#define TRBVH_COLLAPSE_BLOCKS_PER_SM    NUMBLOCKS_KEPLER(16)
#define TRBVH_COLLAPSE_MAX_LEAF_SIZE    14  // 2^N-2 for efficient ballot-reduction in restructureTreelet().

//------------------------------------------------------------------------

union TrbvhNode
{
    BvhNode     out;        // View of the node at output.

    struct tmp              // View of the node during intermediate processing stages.
    {
        float   c0lox;      // AABB of the left child.
        float   c0loy;
        float   c0loz;
        float   c0hix;

        float   c0hiy;
        float   c0hiz;
        int     c0idx;      // internalNodeIdx or ~primitiveIdx
        int     c1idx;      // internalNodeIdx or ~primitiveIdx

        float   c1lox;      // AABB of the right child.
        float   c1loy;
        float   c1loz;
        float   c1hix;

        float   c1hiy;
        float   c1hiz;
        float   pad;        // Unused.
        float   area;       // Half of the AABB surface area of this node.
    } tmp;

    struct {
      float4 f4[4];
    };
};

//------------------------------------------------------------------------

struct TrbvhNodeCosts
{
    float   subtreeSize;    // Number of primitives in the subtree.
    float   subtreeCost;    // Unnormalized SAH cost of the subtree. The LSB tells whether the node wants to be a leaf (0) or an internal node (1).
};

//------------------------------------------------------------------------

struct TrbvhOptSh // per-warp
{
    int             numNodes;
    int             round;
    float           gamma;
    unsigned int    activeMask;
};

//------------------------------------------------------------------------

struct TrbvhRadixTreeParams
{
    TrbvhNode*           /*out*/nodes;          // [nodeRangeStart + max(primRange->span() - 1, 1)]
    int*                 /*out*/nodeVisited;    // [nodeVisitedOfs + max(primRange->span() - 1, 1)]. Initialized to -2.
    int*                 /*out*/nodeParents;    // [max(primRange->span() - 1, 1)], complemented for the right child
    Range*                      outNodeRange;   // [1]

    const unsigned long long*   mortonCodes;    // [primRange->span()]
    const Range*                primRange;      // [1]
    const int*                  nodeRangeStart; // [1] Index of the first entry in nodes.
    const int*                  nodeVisitedOfs; // [1] Index of the first entry in nodeVisited.

    int                         maxPrims;
};

//------------------------------------------------------------------------

struct TrbvhFitParams
{
    TrbvhNode*            /*io*/nodes;          // [nodeRange->end]
    int*                  /*io*/nodeVisited;    // [nodeVisitedOfs + nodeRange->span()]
    TrbvhNodeCosts*      /*out*/nodeCosts;      // [nodeRange->span()]
    int*                        outNodeRangeEnd;

    ModelPointers               inModel;
    const int*                  sortOrder;      // [primRange->span()]
    const int*                  nodeParents;    // [nodeRange->span()], complemented for the right child
    const Range*                primRange;      // [1]
    const Range*                nodeRange;      // [1]
    const int*                  nodeVisitedOfs; // [1] Index of the first entry in nodeVisited.
    
    int                         maxPrims;
    float                       sahNodeCost;
    float                       sahPrimCost;
    float                       maxLeafSize;
};

//------------------------------------------------------------------------

struct TrbvhOptParams
{
    TrbvhNode*            /*io*/nodes;          // [nodeRange->end]
    int*                  /*io*/nodeVisited;    // [nodeVisitedOfs + nodeRange->span()] 
    int*                  /*io*/nodeParents;    // [nodeRange->span()], complemented for the right child
    TrbvhNodeCosts*       /*io*/nodeCosts;      // [nodeRange->span()]
    int*                  /*io*/lwrRound;       // [1] Must be zero initially.
    int*                  /*io*/workCounter;    // [1] Must be zero initially.
    
    const Range*                nodeRange;      // [1]
    const int*                  nodeVisitedOfs; // [1] Index of the first entry in nodeVisited.
    
    int                         maxPrims;
    float                       sahNodeCost;
    float                       sahPrimCost;
    float                       maxLeafSize;
    int                         gamma[4];       // One value per round, last value is used for remaining rounds.
};

//------------------------------------------------------------------------

struct TrbvhCollapseParams
{
    TrbvhNode*            /*io*/nodes;          // [nodeRange->end]
    int*                 /*out*/remap;          // [remapSize + max(primRange->span(), 1)]
    int*                  /*io*/remapSize;      // [1] Must be zero initially.

    const int*                  nodeParents;    // [nodeRange->span()], complemented for the right child
    const TrbvhNodeCosts*       nodeCosts;      // [nodeRange->span()]
    const Range*                primRange;      // [1]
    const Range*                nodeRange;      // [1]

    int                         maxPrims;
    float                       maxLeafSize;
    RemapListLenEncoding        listLenEnc;
};

//------------------------------------------------------------------------

bool launchTrbvhRadixTree   (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TrbvhRadixTreeParams& p);
bool launchTrbvhFit         (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TrbvhFitParams& p);
bool launchTrbvhOpt         (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TrbvhOptParams& p, int treeletSize);
bool launchTrbvhCollapse    (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TrbvhCollapseParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
