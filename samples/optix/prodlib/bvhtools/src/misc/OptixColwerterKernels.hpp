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
#include <prodlib/bvhtools/src/bounds/ApexPointMap.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------------

#define COLWERTER_WARPS_PER_BLOCK 32
#define COLWERTER_BLOCKS_PER_SM   NUMBLOCKS_KEPLER(2)

//------------------------------------------------------------------------------

struct GatherLeafSizesKernelParams
{
    int*                outLeafSize;  // [2*maxNodes]  

    const BvhNode*      nodes;        // [maxNodes]
    const int*          numNodes;     // [1]
};
  
//------------------------------------------------------------------------------

struct OptixColwerterKernelParams
{
    BvhNode*            ioNodes;        // [maxNodes]
    int*                outRemap;       // [maxNodes + 1]
    int*                ioBlockCounter; // [1] Initialized to 0

    const int*          inRemap;        // [maxNodes + 1]
    const int*          numNodes;       // [1] <= maxNodes
    const ApexPointMap* inApexPointMap;
    const int*          inLeafPos;      // [maxNodes * 2]

    int                 maxNodes;
    int                 scale;
    int                 segmentSize;
    int                 segmentCount;
};


//------------------------------------------------------------------------------

bool launchGatherLeafSizesKernel    (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const GatherLeafSizesKernelParams& p);
bool launchOptixColwerterKernel     (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const OptixColwerterKernelParams& p, bool shiftNodes);

//------------------------------------------------------------------------------

size_t exclusiveScanTempSize(unsigned int num_elements);
void exclusiveScan(void* d_temp_storage, size_t temp_storage_bytes, unsigned int * d_in, unsigned int * d_out, unsigned int num_elements, void* stream);

//------------------------------------------------------------------------------
// Returns 0 for non-leaf children

static INLINE void getChildLeafSizes(const BvhNode& node, int* c0num, int* c1num)
{
  *c0num = 0;
  *c1num = 0;
  if (node.c1num != BVHNODE_NOT_IN_USE)
  {
    if (node.c0idx < 0)
      *c0num = node.c0num;
    if (node.c1idx < 0)
      *c1num = node.c1num;
  }
}

//------------------------------------------------------------------------------

static INLINE void colwertHalfNode( int* idx, int* num, int* outRemap, const int* inRemap, int outBegin, int scale, int shift )
{
  if( *idx < 0 )
  {
    int count   = *num;
    int inBegin = ~*idx;  

    // reorder primitive list
    *idx = outBegin;
    *num = outBegin + count;
    
    for( int i=0; i < (int)count; i++ ) 
      outRemap[outBegin+i] = inRemap[inBegin+i];
  }
  else
  {
    *num = scale*(*idx + shift);
    *idx = ~0;
  }
}

//------------------------------------------------------------------------------

static INLINE void colwertNode( BvhNode& node, int* outRemap, const int* inRemap, const int* leafPos, int scale, int shift )
{
  if( node.c1num == BVHNODE_NOT_IN_USE ) 
  {
    node.c0idx = node.c0num = 0;
    node.c1idx = node.c1num = 0;
  }
  else
  {
    colwertHalfNode( &node.c0idx, &node.c0num, outRemap, inRemap, leafPos[0], scale, shift );
    colwertHalfNode( &node.c1idx, &node.c1num, outRemap, inRemap, leafPos[1], scale, shift );
  }
}

//------------------------------------------------------------------------

static INLINE void writeDummyRoot( BvhNode& node, const ApexPointMap* apm, int scale )
{
  AABB aabb = apm->getAABB();
  float3 lo = aabb.lo;
  float3 hi = aabb.hi;

  // Lots of braces to make clang compile without warning about sub objects
  BvhNode root = {{{lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,  ~0,   1*scale,
                       0,    0,    0,    0,    0,    0,   0,   0  }}};
  node = root;
}

//------------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
