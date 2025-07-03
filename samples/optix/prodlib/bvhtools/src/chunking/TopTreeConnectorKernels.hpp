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
#include "../common/TypesInternal.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define TOP_TREE_CONNECTOR_WARPS_PER_BLOCK  4
#define TOP_TREE_CONNECTOR_BLOCKS_PER_SM    NUMBLOCKS_KEPLER(16)

//------------------------------------------------------------------------

struct TopTreeConnectorParams 
{
  BvhNode*             outRoot;        // [1]
  BvhNode*             ioNodes;        // [numPrims - 1], Uses primRange 
                                        
  const PrimitiveAABB* inTrimmedAabbs; // [maxPrims] Aabbs used for the top level build
  const int*           inRemap;        // [maxPrims]
  const Range*         nodeRange;      // [1]
};

//------------------------------------------------------------------------

bool launchTopTreeConnector (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TopTreeConnectorParams& p);

//------------------------------------------------------------------------

static INLINE void connectTopTreeLeaf( const PrimitiveAABB* trimmedAabbs, const int* remap, BvhHalfNode& n  )
{
  if( n.idx < 0 ) // leaf? 
  {
    int pos    = ~n.idx;
    int packed = trimmedAabbs[remap[pos]].pad;      
    int num    = (unsigned)packed >> RLLEPACK_LEN_SHIFT;
    int idx    = (unsigned)packed &  RLLEPACK_INDEX_MASK;
    if( num > 0 ) // map back to original leaf
    {
      n.idx = ~idx;
      n.num = num;
    } 
    else          // map back to original internal node
    {
      n.idx = idx;
      n.num = BVHNODE_CHILD_IS_INTERNAL_NODE;
    }
  }
}

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
