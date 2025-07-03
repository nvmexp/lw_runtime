// Copyright LWPU Corporation 2014
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
#include "../common/Intrinsics.hpp"
#include "../common/TypesInternal.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define TRIM_TOPMOST_NODES_WARPS_PER_BLOCK  4
#define TRIM_TOPMOST_NODES_BLOCKS_PER_SM    NUMBLOCKS_KEPLER(16)

//------------------------------------------------------------------------

struct TrimTopmostNodesParams 
{
    PrimitiveAABB* outTrimmed;     // [max(ioNumTrimmed)] Where to put trimmed leaves and exposed treelet roots.
    Range*         ioTrimmedRange; // [1]
    BvhNode*       ioNodes;        // [numNodes] Uses primRange.

    const int*     lwtoffLevel;    // Trim nodes at or below this level.
    const int*     nodeParents;    // [numNodes], complemented for the right child
    const Range*   nodeRange;      // [1] numNodes = nodeRange->span().
};

//------------------------------------------------------------------------

bool launchTrimTopmostNodes     (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TrimTopmostNodesParams& p);

//------------------------------------------------------------------------

static INLINE bool isHalfNodeValid( const BvhHalfNode& node )
{
  // If a chunk contains exactly one primitive, only the left-hand child of the root
  // is valid. The right-hand child will contain a dummy unhittable AABB that we
  // explicitly need to ignore in TreeTopTrimmer.

  return (node.lox <= node.hix && node.loy <= node.hiy && node.loz <= node.hiz);
}

//------------------------------------------------------------------------
// Encode previous child pointer (internal node) or primitive list (leaf node)
// in the unused pad member.

static INLINE void storeTrimmedHalfNode( PrimitiveAABB* outAabbs, const BvhHalfNode& node, int insertPos )
{
  PrimitiveAABB primAabb;
  primAabb.f4[0] = node.f4[0];  // copy AABB
  primAabb.f4[1] = node.f4[1];
  primAabb.primitiveIdx = insertPos;
  if( node.idx < 0 )   // leaf node
    primAabb.pad = (node.num << RLLEPACK_LEN_SHIFT ) | ~node.idx;  
  else                 // internal node
    primAabb.pad = node.idx;

  outAabbs[insertPos] = primAabb;
}

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
