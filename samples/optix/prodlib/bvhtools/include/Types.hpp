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

#include <vector_types.h>
#include <driver_types.h>
#include <g_lwconfig.h>

#ifndef __LWDACC__
#include <string.h> // for memset
#include <vector>
#endif

namespace prodlib
{
namespace bvhtools
{
struct ApexPointMap;

enum InputType
{
  IT_TRI,
  IT_PRIMAABB,
  IT_AABB,
  IT_INSTANCE
};

enum InstanceFlags
{
  INSTANCE_FLAG_TRIANGLE_LWLL_DISABLE      = 0x1,
  INSTANCE_FLAG_TRIANGLE_LWLL_FLIP_WINDING = 0x2,
  INSTANCE_FLAG_FORCE_OPAQUE               = 0x4,
  INSTANCE_FLAG_FORCE_NO_OPAQUE            = 0x8,
};

//------------------------------------------------------------------------
// Axis-aligned bounding box of an abstract 3D primitive.
// Output of MortonTriangleSplitter, input of TrbvhBuilder.

struct PrimitiveAABB
{
  union {
    struct {
      float   lox;            // Min corner of the AABB.
      float   loy;
      float   loz;
      float   hix;            // Max corner of the AABB.

      float   hiy;
      float   hiz;
      int     primitiveIdx;   // Index of the primitive that this AABB corresponds to.
      int     pad;
    };
    struct {
      float4 f4[2];
    };
  };
};

//------------------------------------------------------------------------

struct TriangleMesh
{
  unsigned int numTriangles;  // Total number of triangles.
  unsigned int numVertices;   // Total number of vertices.
  const int*   indices;       // Pointer to the first index triplet. NULL for a flat triangle list.
  const float* vertices;      // Pointer to the first vertex position triplet.
  int          indexStride;   // Number of bytes between index triplets.
  int          vertexStride;  // Number of bytes between vertex position triplets.

  const unsigned char* triUserIds;  // User triangle IDs to place in the triangle blocks. NULL => same as primitive indices.
  int                  triUseIdSizeInBytes; // Bytes in triUserIds element.
  int                  triUserIdStride;  // Bytes between conselwtive elements in triUserIds. Must be divisible by 4.
  unsigned char        triAlphaDefaultFlags;  // Triangle alpha flags to place in the triangle blocks if triAlphaFlags array is not set. NULL => all false.
  const unsigned char* triAlphaFlags;       // Triangle alpha flags to place in the triangle blocks. NULL => all false.
  int                  triAlphaFlagStride;  // Bytes between conselwtive elements in triAlphaFlags.
  int                  triAlphaFlagShift;   // ((triAlphaFlags[index] >> triAlphaFlagShift) & 1) ^ triAlphaFlagIlwerse.
  int                  triAlphaFlagIlwerse; // ((triAlphaFlags[index] >> triAlphaFlagShift) & 1) ^ triAlphaFlagIlwerse.

  TriangleMesh( void )
  {
    numTriangles = 0;
    numVertices  = 0;
    indices      = nullptr;
    vertices     = nullptr;
    indexStride  = 0;
    vertexStride = 0;

    triUserIds           = nullptr;
    triUseIdSizeInBytes  = (int)sizeof( int );
    triUserIdStride      = (int)sizeof( int );
    triAlphaDefaultFlags = 0;
    triAlphaFlags        = nullptr;
    triAlphaFlagStride   = (int)sizeof( unsigned char );
    triAlphaFlagShift    = 0;
    triAlphaFlagIlwerse  = 0;
  }
};

//------------------------------------------------------------------------

struct Group
{
    int            numInstances;    // Total number of instances
    int            numUniqueModels; // The number of unique models the complete scene is assembled of.
    PrimitiveAABB* aabbs;           // Per instance global AABB // BL: Does this need to have sequential instance IDs?
    float*         ilwMatrices;     // A matrix per instance to transform the ray during traversal from global to local object space
    int*           modelIds;        // A mapping of instances to models
    int            matrixStride;    // Number of bytes until we find the next matrix entry

    Group(void)
    {
        numInstances    = 0;
        numUniqueModels = 0;
        aabbs           = nullptr;
        ilwMatrices     = nullptr;
        modelIds        = nullptr;
        matrixStride    = 0;
    }
};

//------------------------------------------------------------------------
// Input to build
struct InstanceDesc
{
  float        transform[12];           // Object-to-world transformation.
  unsigned int instanceId;              // User specified
  unsigned int instanceOffsetAndFlags;  // User specified, offset is in 24 low bits, flags are in 8 high bits
  const char*  bvh;                     // Pointer to the BVH containing geometry for this instance.
};

//------------------------------------------------------------------------
// Instance data stored in the BVH (adds the ilwerse transform)
struct BvhInstanceData
{
    float transform[12];            // Object-to-world transformation. 
    float ilwTransform[12];         // World-to-object transformation.
    const char* bvh;                // Pointer to the BVH containing geometry for this instance.
    unsigned int instanceOffset;    // User specified
    unsigned int flags;             // User specified
    unsigned int lwllMask;          // User specified
    unsigned int instanceId;        // User specified
    unsigned int instanceIdx;       // instance index in top level hierarchy
    unsigned int pad[1];
};

//------------------------------------------------------------------------
// Standard BVH node. Each node contains the AABB of its children.
// Output of TrbvhBuilder.

#define BVHNODE_CHILD_IS_INTERNAL_NODE  -1  // Placed in c0num/c1num if the corresponding child is an internal node.
#define BVHNODE_NOT_IN_USE              -2  // Placed in c1num of a node slot that is not in use.

struct BvhNode
{
  union {
    struct {
      float   c0lox;      // Min corner of the left child AABB.
      float   c0loy;
      float   c0loz;
      float   c0hix;      // Max corner of the left child AABB.

      float   c0hiy;
      float   c0hiz;
      int     c0idx;      // internalNodeIdx or ~primitiveRemapBegin
      int     c0num;      // Number of primitives in left child, or BVHNODE_CHILD_IS_INTERNAL_NODE.

      float   c1lox;      // Min corner of the right child AABB.
      float   c1loy;
      float   c1loz;
      float   c1hix;      // Max corner of the right child AABB.

      float   c1hiy;
      float   c1hiz;
      int     c1idx;      // internalNodeIdx or ~primitiveRemapBegin
      int     c1num;      // Number of primitives in right child, or BVHNODE_CHILD_IS_INTERNAL_NODE, or BVHNODE_NOT_IN_USE.
    };  
  
    struct {
      float4 f4[4];
    };
  };
};

//------------------------------------------------------------------------------

enum HeaderFlags
{
  HF_TYPE_NONE = 0x00,
  HF_TYPE_BVH2 = 0x01,
  HF_TYPE_BVH8 = 0x02,
#if LWCFG( GLOBAL_ARCH_TURING )
  HF_TYPE_TTU  = 0x03,
#endif
  HF_TYPE_MASK = 0xFF,

  HF_SINGLE_INSTANCE_IDENTITY_XFORM = 0x100, // Single instance accel with identity transform. Can skip to bottom level traversal.

  HF_OPTIX_NODES = 0x200, // OptiX nodes (OptiX uses a different format)
  
  HF_RLLE_NONE            = 0x0000, 
  HF_RLLE_COMPLEMENT_LAST = 0x1000, 
  HF_RLLE_PACK_IN_FIRST   = 0x2000, 
  HF_RLLE_MASK            = 0xF000,
};

//------------------------------------------------------------------------------

struct BvhHeader
{
    size_t      size                        = 0;  // size of the entire accel buffer data.
    size_t      compactSize                 = 0;
    unsigned    flags                       = 0;
    unsigned    numEntities                 = 0;  // number of entities (tris,AABBs,instances) in the BVH

    size_t      nodesOffset                 = 0;
    size_t      instanceDataOffset          = 0;  // Offset to BvhInstanceData buffer
    size_t      instanceNodeOffset          = 0;  // Offset to TTUInstanceNode buffer
    size_t      remapOffset                 = 0;
    size_t      nodeParentsOffset           = 0;
    size_t      apmOffset                   = 0;  // apex point map
    size_t      trianglesOffset             = 0;  // baked triangles
    size_t      indicesOffset               = 0;  // baked indices
    size_t      nodeAuxOffset               = 0;  // only used when refitting

    size_t      compactNodeAuxOffset        = 0;  // The new offset for the node aux buffer
    size_t      compactTrianglesOffset      = 0;  // The new offset for the triangle buffer
    size_t      compactIndicesOffset        = 0;  // The new offset for the index buffer

    // Use these to recover the sizes of the buffers. TODO: Some of these are likely redundant
    unsigned    numNodes                    = 0;
    unsigned    numRemaps                   = 0;
    unsigned    numTriangles                = 0;  // Number of baked triangles
    unsigned    numAabbs                    = 0;  // Number of AABBs
    unsigned    numTriBlocks                = 0;  // Number of packed triangle blocks (may be fewer than the allocated number of blocks due to compression)
    unsigned    maxTriBlocks                = 0;  // Allocated number of triangle blocks
    unsigned    numInstances                = 0;  // Number of instances
    unsigned    numIndices 	                = 0;  // number of baked indices
    unsigned    numInputGeometries          = 0;
    unsigned    numGeometries               = 0;

    size_t      primBitsOffset              = 0;  // primBits array
    int         numPrimitiveIndexBits       = 0;  // The number of bits used for the primitive index in primBits
    unsigned    pad[5];
};

//------------------------------------------------------------------------------

struct BvhHalfNode
{
  union
  {
    struct
    {
      float   lox;      // Min corner of the child AABB.
      float   loy;
      float   loz;
      float   hix;      // Max corner of the child AABB.

      float   hiy;
      float   hiz;
      int     idx;      // internalNodeIdx or ~primitiveRemapBegin
      int     num;      // Number of primitives in child, BVHNODE_CHILD_IS_INTERNAL_NODE, or BVHNODE_NOT_IN_USE (right half).    
    };
  
    struct
    {
      float4 f4[2];
    };
  };
};

//------------------------------------------------------------------------------

struct MemoryUsage
{
  size_t header;      // bytes used for the header
  size_t nodes;       // bytes used for bvh nodes
  size_t remap;       // bytes used for remap buffer

  size_t output;      // bytes used for output buffer
  size_t temp;        // bytes used for temp buffer
  size_t readback;    // bytes used for the readback buffer

  size_t totalFinal;  // total bytes used after build
  size_t totalMax;    // total bytes used during build
};

//------------------------------------------------------------------------
// Data for Sven Woop's ray-triangle intersection test.
// Output of TriangleWooper.

struct WoopTriangle
{
    float4  t;  // (t.x*X + t.y*Y + t.z*Z = t.w) is the triangle plane. The LSB of t.w is set iff this is the last triangle in a leaf.
    float4  u;  // (u.x*X + u.y*Y + u.z*Z + u.w) gives the barycentric U when (X,Y,Z) lies on the triangle plane.
    float4  v;  // (v.x*X + v.y*Y + v.z*Z + v.w) gives the barycentric V when (X,Y,Z) lies on the triangle plane.
};

//------------------------------------------------------------------------

struct TracerData
{
    bool        inDeviceMem;  // Location of the buffers
    bool        optixNodes;   // BL: Do I want to keep this? It allows tracerNaive to trace optixNodes. I'm not sure that even works still. Could be used for testing optix style notes in test_Prime

    int         numNodes;
    BvhNode*    nodes;

    int         numRemaps;
    int*        remap;

    TracerData(void)
    {
        inDeviceMem = false;
        optixNodes  = false;
        numNodes    = 0;
        nodes       = nullptr;
        numRemaps   = 0;
        remap       = nullptr;
    }

    virtual ~TracerData() {} // make TracerData a polymorphic type
};


//------------------------------------------------------------------------

struct TracerDataMesh : public TracerData
{
    TriangleMesh    mesh;             // All data we need to build a Trbvh
    WoopTriangle*   woopTriangles;    // A pointer to the data for the Woop ray-triangle intersection test
    int             numWoopTriangles; // The number of Woop triangles. Due to splitting in the Trbvh process, this might be different from TriangleMesh.numPrimitives!

    TracerDataMesh(void)
    {
        woopTriangles    = nullptr;
        numWoopTriangles = 0;
    }
};

//------------------------------------------------------------------------

#ifndef __LWDACC__
struct TracerDataGroup : public TracerData
{
  Group group;
  std::vector<TracerDataMesh> meshes;
};
#endif

//------------------------------------------------------------------------

} // namespace bvhtools
} // namespace prodlib
