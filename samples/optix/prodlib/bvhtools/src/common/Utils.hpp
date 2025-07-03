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

#include <math.h>
#include <stdlib.h>
#include <vector_types.h>
#include <functional>

namespace prodlib
{
namespace bvhtools 
{
  struct BvhNode;
  struct BvhHeader;
  struct ModelPointers;

  class LwdaUtils;

  struct Triangle
  {
    float3 v[3];
  };   
  
//------------------------------------------------------------------------------

// TODO: Temporary workaround!
// We can lwrrently only do an inline-copy of a limited amount of data from host to device (~32kB), and asynchronous copy is not supported.
// This function breaks up medium sized buffers into a series of smaller copy operations.
void memcpyInlineWAR(char *dst, const char *src, size_t size, LwdaUtils *lwca);

//------------------------------------------------------------------------------

void dumpObj( const char* filename, size_t numTriangles, size_t numVertices, const float* vertices, const int* indices=0 );

//------------------------------------------------------------------------------
struct Bvh2Info
{
public:
  const BvhHeader* header = nullptr;
  const BvhNode*   nodes = nullptr;
  const int*       remap = nullptr;

  Bvh2Info() {}
  Bvh2Info(const void* bvh);

  // Returns false if child (0 or 1) is not an interior node
  bool getChildNodeIdx(int nodeIdx, int child, int& childNodeIdx) const;

  // Returns false if child (0 or 1) is not a leaf
  bool getLeafRange(int nodeIdx, int child, int& begin, int& end ) const;
};

// Returns true if all nodes were traversed. Return false in the visitor to end
// traversal. Visitor parameters:
//    idx         Current node index
//    parentIdx   Parent of the current node. -1 for the root. 
//    child       Which child (0 or 1) of the parent the current node is.
//    depth       Depth in the tree (0 for the root)
//    bvh         Access to BVH data.
bool traverse(const Bvh2Info& bvh, 
  std::function<bool (int idx, int parentIdx, int child, int depth, const Bvh2Info& bvh)> visitor,
  int startIdx = 0,
  int maxDepth = 0x7fffffff);

//------------------------------------------------------------------------------
// childScale is the scaling applied to child indices (4 for optix nodes)
bool validateTree( size_t numPrims, const BvhNode* nodesBuffer, size_t nodesBufferSize, const int* remapBuffer, size_t remapBufferSize, int childScale, int maxDepth=0x7fffffff );
bool validateTree( const void* bvh, int maxDepth=0x7fffffff );


//------------------------------------------------------------------------------
// childScale is the scaling applied to child indices (4 for optix nodes)
void printTree( size_t numPrims, const BvhNode* nodesBuffer, size_t nodesBufferSize, const int* remapBuffer, size_t remapBufferSize, int childScale, int maxDepth=0x7fffffff );
void printTree( const void* bvh, int maxDepth=0x7fffffff );

//------------------------------------------------------------------------------

void printNodes( size_t numNodes, const BvhNode* nodesBuffer );
void printNodes( size_t numNodes, const void* bvh );


// TODO: We probably ought to split this file. The above functions are mostly for 
// debugging, while the following are kernels, most of which eventually should be 
// BuildingBlocks.


//------------------------------------------------------------------------------
// TODO: Make this a real building block.
// Expands indexed triangles (indices != nullptr) or copies flat triangles to the
// output buffer. Creates primBits. transform may be nullptr.
void copyTris(int numTris, const void* indices, bool shortIndices, int indexStride, const void* vertices, int numVertices, int vertexStride, int geometryOffset, bool opaque, int primitiveIndexBits, int primBitsSizeInBytes, bool bakeEncodingMode, Triangle* outTris, void* primBits, const void *transform);

//------------------------------------------------------------------------------

void copyAabbs( int numAabbs, const void* aabbs, int aabbStride, int geometryOffset, bool opaque, int primitiveIndexBits, int primBitsSizeInBytes, void* outAabbs, void* primBits);


//------------------------------------------------------------------------------

// Both these functions take an optional nodeParents pointer to override the one in the header.
// This is so we can refit for motion blur only, using a temp buffer for the parent indices.
void updateNodeParentsBvh2( void* bvh2, int numNodes, int* nodeParents = NULL );
void refitBvh2( void* bvh2, int numNodes, int motionStep, const ModelPointers* model, int* nodeParents = NULL );

} // namespace bvhtools
} // namespace prodlib
