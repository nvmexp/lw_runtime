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

#include "TypesInternal.hpp"
#include "Utils.hpp"
#include <include/Utils.hpp>
#include "LwdaUtils.hpp"
#include "../bounds/ApexPointMapConstructor.hpp"
#include "../bounds/InstanceDataAdapter.hpp"
#include "../trace/TracerTypes.hpp"
#include "prodlib/exceptions/Assert.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------------

// TODO: Temporary workaround!
void prodlib::bvhtools::memcpyInlineWAR(char *dst, const char *src, size_t size, LwdaUtils *lwca)
{
  if( lwca )
  {
    static const size_t WAR_COPY_SIZE = 32000;
    size_t offset = 0;
    while (offset < size)
    {
      size_t bytesLeft = size - offset;
      size_t s = bytesLeft < WAR_COPY_SIZE ? bytesLeft : WAR_COPY_SIZE;
      lwca->memcpyHtoDAsync( dst + offset, src + offset, s );
      offset += WAR_COPY_SIZE;
    }
  }
  else
  {
    memcpy( dst, src, size );
  }
}


//------------------------------------------------------------------------------
void prodlib::bvhtools::dumpObj( const char* filename, size_t numTriangles, size_t numVertices, const float* vertices, const int* indices )
{
  std::ofstream out(filename);

  for( size_t v=0; v < numVertices; ++v )
    out << "v " << vertices[3*v+0] << " " << vertices[3*v+1] << " " << vertices[3*v+2] << std::endl;

  if( indices )
  {
    for( size_t t=0; t < numTriangles; ++t )
      out << "f " << 1+indices[3*t+0] << " " << 1+indices[3*t+1] << " " << 1+indices[3*t+2] << std::endl;
  }
  else
  {
    for( size_t t=0; t < numTriangles; ++t )
      out << "f " << 1+(3*t+0) << " " << 1+(3*t+1) << " " << 1+(3*t+2) << std::endl;
  }
}

//------------------------------------------------------------------------------
prodlib::bvhtools::Bvh2Info::Bvh2Info(const void* bvh)
{
  const char* bvhPtr = (const char*)bvh;
  header = (const BvhHeader*)bvhPtr;
  nodes = (BvhNode*)(bvhPtr + header->nodesOffset);
  remap = (int*)(bvhPtr + header->remapOffset);
}

//------------------------------------------------------------------------------
bool prodlib::bvhtools::Bvh2Info::getChildNodeIdx(int nodeIdx, int child, int& childNodeIdx) const
{
  const BvhNode& node = nodes[nodeIdx];
  const int idx = (child == 0) ? node.c0idx : node.c1idx;
  const int num = (child == 0) ? node.c0num : node.c1num;
  if( header->flags & HF_OPTIX_NODES )
  {
    if(idx != ~0) // leaf
      return false;
    childNodeIdx = num / 4;
  }
  else
  {
    if(idx < 0) // leaf
      return false;
    childNodeIdx = idx;
  }
  return true;
}

//------------------------------------------------------------------------------
bool prodlib::bvhtools::Bvh2Info::getLeafRange(int nodeIdx, int child, int& begin, int& end) const
{
  const BvhNode& node = nodes[nodeIdx];
  const int idx = (child == 0) ? node.c0idx : node.c1idx;
  const int num = (child == 0) ? node.c0num : node.c1num;
  if( header->flags & HF_OPTIX_NODES )
  {
    if(idx == ~0) // internal node
      return false;
    begin = idx;
    end = num;
  }
  else
  {
    if(idx >= 0) // internal node
      return false;
    begin = -idx;
    end = begin + num;
  }
  return true;
}


//------------------------------------------------------------------------------
struct TraverseParams
{
  const Bvh2Info& bvh;
  std::function<bool(int idx, int parentIdx, int child, int depth, const Bvh2Info& bvh)> visitor;
  int maxDepth;
};

static bool traverseRec( int idx, int parentIdx, int child, int depth, TraverseParams& p )
{
  if( depth > p.maxDepth )
    return true;

  if( idx >= (int)p.bvh.header->numNodes || idx < 0 )
  {
    std::cerr << "Node index out of range: " << idx << std::endl;
    return false;
  }

  if( !p.visitor(idx, parentIdx, child, depth, p.bvh) )
    return false;

  for( int ch = 0; ch < 2; ++ch )
  {
    int childIdx;
    if( p.bvh.getChildNodeIdx(idx, ch, childIdx) )
    {
      if( !traverseRec( childIdx, idx, ch, depth+1, p ) )
        return false;
    }
  }

  return true;
}

//------------------------------------------------------------------------------
bool prodlib::bvhtools::traverse(const Bvh2Info& bvh, std::function<bool(int idx, int parentIdx, int child, int depth, const Bvh2Info& bvh)> visitor, int startIdx/*=0*/, int maxDepth/*=0x7fffffff*/)
{
  TraverseParams p = { bvh, visitor, maxDepth };
  return traverseRec(startIdx, -1, 0, 0, p);
}

//------------------------------------------------------------------------------
static bool validateNode( int idx, int parentIdx, int child, int depth, const Bvh2Info& bvh )
{
  const BvhNode& n = bvh.nodes[idx];
  const AABB* aabb[2] = { (AABB*)&n.c0lox, (AABB*)&n.c1lox };
  if( !aabb[0]->isValid() || !aabb[1]->isValid() )
  {
    std::cerr << "Node:" << idx << " - Invalid AABB\n";
    return false;  
  }
  if( parentIdx >= 0 )
  {
    const BvhNode& pn = bvh.nodes[parentIdx];
    const AABB* parentAabb[2] = { (AABB*)&pn.c0lox, (AABB*)&pn.c1lox };
    if( !parentAabb[child]->contains( *aabb[0] ) || !parentAabb[child]->contains( *aabb[1] ) )
    {
      std::cerr << "Node:" << idx << " - AABB not contained in parent AABB\n";
      return false;
    }
  }

  for( int ch = 0; ch < 2; ++ch )
  {
    int begin, end;
    if( bvh.getLeafRange( idx, ch, begin, end ) )
    {
      if( end < begin || begin >= (int)bvh.header->numRemaps || end >= (int)bvh.header->numRemaps )
      {
        std::cerr << "Node:" << idx << " - Bad primitive list range:" << begin << "," << end << std::endl;
        return false;
      }

      for( int i=begin; i < end; ++i )
      {
        int primIdx = bvh.remap[i];
        if( bvh.header->flags & HF_RLLE_PACK_IN_FIRST) 
        {
          // TODO: Figure out why this isn't working
          //if( i == begin )
          //{
          //  int len = ((unsigned)primIdx) >> RLLEPACK_LEN_SHIFT;
          //  if( len != (end - begin) )
          //  {
          //    std::cerr << "Node:" << idx << " - Packed list len mismatch: " << len << " (should be " << (end - begin) << ")\n";
          //    return false;
          //  }
          //}
          primIdx &= RLLEPACK_INDEX_MASK;
        }
        if( primIdx >= (int)bvh.header->numEntities )
        {
          std::cerr << "Node:" << idx << " - Bad primitive index: " << primIdx << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

//------------------------------------------------------------------------------
bool prodlib::bvhtools::validateTree( size_t numPrims, const BvhNode* nodesBuffer, size_t nodesBufferSize, const int* remapBuffer, size_t remapBufferSize, int childScale, int maxDepth )
{
  BvhHeader header;
  header.flags = HF_TYPE_BVH2;
  if( childScale > 0 )
    header.flags |= HF_OPTIX_NODES;
  header.numEntities = unsigned(numPrims);
  header.numNodes = unsigned(nodesBufferSize / sizeof(BvhNode));
  header.numRemaps = unsigned(remapBufferSize / sizeof(int));

  Bvh2Info bvhInfo;
  bvhInfo.header = &header;
  bvhInfo.nodes = nodesBuffer;
  bvhInfo.remap = remapBuffer;

  return traverse( bvhInfo, validateNode, 0, maxDepth );
}

//------------------------------------------------------------------------------
bool prodlib::bvhtools::validateTree(const void* bvh, int maxDepth/*=0x7fffffff */)
{
  return traverse( bvh, validateNode, 0, maxDepth );
}

//------------------------------------------------------------------------------
static void print( const BvhNode& n, int idx, int depth=0 )
{
  printf(
    "%2d: %*s[%8.2g,%8.2g,%8.2g][%8.2g,%8.2g,%8.2g] %5d %5d || [%8.2g,%8.2g,%8.2g][%8.2g,%8.2g,%8.2g] %5d %5d\n", 
    idx, 2*depth, "", n.c0lox, n.c0loy, n.c0loz, n.c0hix, n.c0hiy, n.c0hiz, n.c0idx, n.c0num, 
    n.c1lox, n.c1loy, n.c1loz, n.c1hix, n.c1hiy, n.c1hiz, n.c1idx, n.c1num
  );
}

//------------------------------------------------------------------------------
void prodlib::bvhtools::printTree( size_t numPrims, const BvhNode* nodesBuffer, size_t nodesBufferSize, const int* remapBuffer, size_t remapBufferSize, int childScale, int maxDepth )
{
  BvhHeader header;
  header.flags = HF_TYPE_BVH2;
  if( childScale > 0 )
    header.flags |= HF_OPTIX_NODES;
  header.numEntities = unsigned(numPrims);
  header.numNodes = unsigned(nodesBufferSize / sizeof(BvhNode));
  header.numRemaps = unsigned(remapBufferSize / sizeof(int));

  Bvh2Info bvhInfo;
  bvhInfo.header = &header;
  bvhInfo.nodes = nodesBuffer;
  bvhInfo.remap = remapBuffer;

  traverse( 
    bvhInfo,
    [](int idx, int parentIdx, int child, int depth, const Bvh2Info& bvh) -> bool {
      print(bvh.nodes[idx], idx, depth);
      return true;
    },
    0,
    maxDepth
  );
}

//------------------------------------------------------------------------------
void prodlib::bvhtools::printTree(const void* bvh, int maxDepth/*=0x7fffffff */)
{
  traverse( 
    bvh,
    [](int idx, int parentIdx, int child, int depth, const Bvh2Info& bvh) -> bool {
      print(bvh.nodes[idx], idx, depth);
      return true;
    },
    0,
    maxDepth);
}

//------------------------------------------------------------------------------
void prodlib::bvhtools::printNodes(size_t numNodes, const BvhNode* nodesBuffer)
{
  for( size_t i = 0; i < numNodes; ++i )
    print( nodesBuffer[i], (int)i );
}

//------------------------------------------------------------------------------
void prodlib::bvhtools::printNodes(size_t numNodes, const void* bvh)
{
  const BvhHeader* bvhHeader = (const BvhHeader*)bvh;
  const BvhNode* nodesBuffer = (const BvhNode*)((char*)bvh + bvhHeader->nodesOffset);
  for( size_t i = 0; i < numNodes; ++i )
    print( nodesBuffer[i], (int)i );  
}

//------------------------------------------------------------------------------

void prodlib::bvhtools::computeInstanceAabbsAndIlwMatrices(
    bool            useLwda,
    int             numInstances,
    int             numModels,
    int             matrixStride,
    int*            instanceIds,
    float4*         transforms,
    ApexPointMap**  hostApexPointMaps,
    PrimitiveAABB*  outWorldSpaceAabbs,
    float4*         outIlwMatrices )
{
  MemorySpace ioMemSpace = (useLwda) ? MemorySpace_LWDA : MemorySpace_Host;
  LwdaUtils* lwdaUtils = (useLwda) ? new LwdaUtils : NULL;
  {
    InstanceDataAdapter instDataAdapter;
    InstanceDataAdapter::Config c;
    
    c.lwdaUtils           = lwdaUtils;
    c.useLwda             = useLwda;
    c.numInstances        = numInstances;
    c.matrixStride        = matrixStride;
    c.outWorldSpaceAabbs  = BufferRef<PrimitiveAABB>(outWorldSpaceAabbs, numInstances, ioMemSpace);
    c.outIlwMatrices      = BufferRef<float4>(outIlwMatrices, numInstances * (matrixStride / 16), ioMemSpace).reinterpret<float>();
    c.inApexPointMaps     = BufferRef<const ApexPointMap* const>(hostApexPointMaps, numModels, MemorySpace_Host);
    c.inTransforms        = BufferRef<const float4>(transforms, numInstances * (matrixStride / 16), ioMemSpace).reinterpret<const float>();
    c.inInstanceIds       = BufferRef<int>(instanceIds, numInstances, ioMemSpace);
    
    c.outWorldSpaceAabbs  .materialize(lwdaUtils);
    c.outIlwMatrices      .materialize(lwdaUtils);
    c.inApexPointMaps     .materialize(lwdaUtils);
    c.inTransforms        .materialize(lwdaUtils);
    c.inInstanceIds       .materialize(lwdaUtils);
    
    instDataAdapter.configure(c);
    instDataAdapter.execute();
  }
  delete lwdaUtils;
}

//------------------------------------------------------------------------------

void prodlib::bvhtools::createAccelInstanceData(
  int                  numInstances,
  const InstanceDesc*  inInstanceDescs,
  BvhInstanceData*     outInstanceData )
{
  LwdaUtils lwdaUtils;
  InstanceDataAdapter instDataAdapter;
  InstanceDataAdapter::Config c;

  c.lwdaUtils           = &lwdaUtils;
  c.useLwda             = true,
  c.numInstances        = numInstances;
  c.computeAabbs        = false;
  c.outWorldSpaceAabbs  = EmptyBuf;
  c.outBvhInstanceData  = BufferRef<BvhInstanceData>(outInstanceData, numInstances, MemorySpace_LWDA);
  c.inInstanceDescs      = BufferRef<const InstanceDesc>(inInstanceDescs, numInstances, MemorySpace_LWDA);

  c.outBvhInstanceData  .materialize(&lwdaUtils);
  c.inInstanceDescs      .materialize(&lwdaUtils);

  instDataAdapter.configure(c);
  instDataAdapter.execute();
}

//------------------------------------------------------------------------------

void prodlib::bvhtools::computeAabb(                                                                
    bool                useLwda,                                                     
    const TriangleMesh& mesh,                                            
    float               result[6] )
{
  ApexPointMap apm;
  LwdaUtils* lwdaUtils = (useLwda) ? new LwdaUtils : NULL;
  {
    ApexPointMapConstructor apmConstructor;

    ApexPointMapConstructor::Config c;
    c.lwca              = lwdaUtils; 
    c.apmResolution     = 1;
    c.outApexPointMap   = BufferRef<ApexPointMap>( &apm, 1, MemorySpace_Host );
    c.inModel           = ModelBuffers(mesh, (useLwda) ? MemorySpace_LWDA : MemorySpace_Host);

    apmConstructor.configure(c);

    c.outApexPointMap.materialize(lwdaUtils);
    c.tempBuffer.materialize(lwdaUtils);
    c.inModel.materialize(lwdaUtils);

    apmConstructor.execute();
  }
  delete lwdaUtils;

  AABB aabb = apm.getAABB();
  result[0] = aabb.lo.x;
  result[1] = aabb.lo.y;
  result[2] = aabb.lo.z;
  result[3] = aabb.hi.x;
  result[4] = aabb.hi.y;
  result[5] = aabb.hi.z;
}

//------------------------------------------------------------------------------

void prodlib::bvhtools::computeApexPointMap( float aabb[6], int resultData[8] )
{
  LwdaUtils* lwdaUtils = nullptr;
  {
    ApexPointMapConstructor apmConstructor;

    ApexPointMapConstructor::Config c;
    c.lwca            = lwdaUtils;
    c.apmResolution   = 1;
    c.outApexPointMap = BufferRef<ApexPointMap>( reinterpret_cast<ApexPointMap*>( resultData ), 1, MemorySpace_Host );
    PrimitiveAABB paabb;
    paabb.lox          = aabb[0];
    paabb.loy          = aabb[1];
    paabb.loz          = aabb[2];
    paabb.hix          = aabb[3];
    paabb.hiy          = aabb[4];
    paabb.hiz          = aabb[5];
    paabb.primitiveIdx = 0;
    c.inModel          = ModelBuffers( 1, &paabb, MemorySpace_Host );

    apmConstructor.configure( c );

    c.outApexPointMap.materialize( lwdaUtils );
    c.tempBuffer.materialize( lwdaUtils );
    c.inModel.materialize( lwdaUtils );

    if( sizeof( int[8] ) < c.outApexPointMap.getNumBytes() )
      throw AssertionFailure( RT_EXCEPTION_INFO, "sizeof(resultData) is too small" );

    apmConstructor.execute();
  }
}
