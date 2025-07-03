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
#include "BVH8ConstructorKernels.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define BVH8FITTER_NODES_WARPS_PER_BLOCK 2
#define BVH8FITTER_NODES_BLOCKS_PER_SM   NUMBLOCKS_MAXWELL(20)
#define BVH8FITTER_NODES_GROUP_SIZE      8  // # of lanes that collaborate on a node
#define BVH8FITTER_NODES_MAX_LEAF_SIZE   3

//------------------------------------------------------------------------

struct BVH8FitterNodesParams
{
    BVH8Node*               ioNodes;            // [maxNodes]
    BVH8NodeAux*            ioNodeAux;          // [maxNodes]
    int*                    ioRemap;
    BVH8Triangle*           outTriangles;       // [maxPrims], non-NULL => peephole optimization to skip BVH8FitterTriangles
    int*                    workCounter;        // [1], cleared to zero

    const int*              inNumNodes;         // [1]
    ModelPointers           inModel;

    int                     translateIndices;   // bool
    int                     removeDuplicates;   // bool
    int                     numReorderRounds;   // 0 if disabled
    int                     supportRefit;       // bool
};

//------------------------------------------------------------------------

#define BVH8FITTER_TRIANGLES_WARPS_PER_BLOCK    1
#define BVH8FITTER_TRIANGLES_BLOCKS_PER_SM      NUMBLOCKS_MAXWELL(32)

//------------------------------------------------------------------------

struct BVH8FitterTrianglesParams
{
    BVH8Triangle*   outTriangles;   // [maxPrims]

    const int*      inRemap;        // [maxPrims]
    ModelPointers   inModel;

    int             maxPrims;
};

//------------------------------------------------------------------------

bool launchBVH8FitterNodes      (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const BVH8FitterNodesParams& p);
bool launchBVH8FitterTriangles  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const BVH8FitterTrianglesParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
