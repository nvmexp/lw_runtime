/*
 *  Copyright (c) 2012, LWPU Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of LWPU Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <algorithm>

#include "TracerNaive.hpp"
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/Assert.h>
#include "../common/SharedKernelFunctions.hpp"
#include "WatertightOriginal.hpp"

using namespace prodlib::bvhtools;
using namespace prodlib;

//------------------------------------------------------------------------

struct Ray
{
  float   origX, origY, origZ;    // Ray origin.
  float   tmin;                   // Non-negative t-value where the ray begins.
  float   dirX, dirY, dirZ;       // Ray direction.
  float   tmax;                   // Non-negative t-value where the ray ends. To disable a ray, set tmax < tmin.
};

//------------------------------------------------------------------------

struct Hit
{
  float   t;
  int     triangleIdx;            // -1 if no hit.
  float   u;
  float   v;
};

//------------------------------------------------------------------------

#define STACK_SIZE              64              // Size of the traversal stack.
#define EntrypointSentinel      0x76543210      // Bottom-most stack entry, indicating the end of traversal.

inline float min4(float a, float b, float c, float d) { return fminf(fminf(a,b), fminf(c,d)); }
inline float max4(float a, float b, float c, float d) { return fmaxf(fmaxf(a,b), fmaxf(c,d)); }

//------------------------------------------------------------------------

static __forceinline__ float3 operator-( const float3& a, const float3& b )
{
  return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

static __forceinline__ float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

static __forceinline__ float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}

//------------------------------------------------------------------------

TracerNaive::TracerNaive(void)
{
    m_bvhLayout.storeOnGPU        = false;
    m_bvhLayout.arrayAlign        = 64;
    m_bvhLayout.reorderTriangles  = true;
    m_bvhLayout.optixNodes        = false;
}

//------------------------------------------------------------------------

TracerNaive::~TracerNaive(void)
{
}

//------------------------------------------------------------------------------
void fetchRay( const void* rays, int rayIdx, int rayFormat, float3* orig, float3* dir, float* tmin, float* tmax, int* mask )
{
  const float* raysPtr = (float*)rays;
  if( rayFormat == RAY_ORIGIN_TMIN_DIRECTION_TMAX )
  {
    *orig = *(float3*)(raysPtr + 8*rayIdx + 0);
    *tmin =          *(raysPtr + 8*rayIdx + 3);
    *dir  = *(float3*)(raysPtr + 8*rayIdx + 4);
    *tmax =          *(raysPtr + 8*rayIdx + 7);
    *mask = 0;
  }
  else if( rayFormat == RAY_ORIGIN_MASK_DIRECTION_TMAX )
  {
    *orig = *(float3*)(raysPtr + 8*rayIdx + 0);
    *mask =    *(int*)(raysPtr + 8*rayIdx + 3);
    *tmin = 0.0f;
    *dir  = *(float3*)(raysPtr + 8*rayIdx + 4);
    *tmax =          *(raysPtr + 8*rayIdx + 7);
  }
  else
  {
    *orig = *(float3*)(raysPtr + 6*rayIdx + 0);
    *tmin = 0;
    *dir  = *(float3*)(raysPtr + 6*rayIdx + 3);
    *tmax = 1e34f;
    *mask = 0;
  }
}

//------------------------------------------------------------------------------
void fetchRay( const void* rays, int rayIdx, int rayFormat, float3* orig, float3* dir, float* tmin, float* tmax )
{
  int tmpMask = 0;
  fetchRay(rays, rayIdx, rayFormat, orig, dir, tmin, tmax, &tmpMask);
}


//------------------------------------------------------------------------------
static inline
void storeHit( void* hits, int rayIdx, int hitFormat, int triIndex, int instanceIndex, float t, float u, float v )
{
  switch( hitFormat )
  {
  case HIT_T_TRIID_INSTID:
    ((float*)hits)[rayIdx*3+0] = t;
    ((float*)hits)[rayIdx*3+1] = __int_as_float(triIndex);
    ((float*)hits)[rayIdx*3+2] = __int_as_float(instanceIndex);
    break;
  case HIT_T_TRIID_INSTID_U_V:
    ((float*)hits)[rayIdx*5+0] = t;
    ((float*)hits)[rayIdx*5+1] = __int_as_float(triIndex);
    ((float*)hits)[rayIdx*5+2] = __int_as_float(instanceIndex);
    ((float*)hits)[rayIdx*5+3] = u;
    ((float*)hits)[rayIdx*5+4] = v;
    break;
  case HIT_T_TRIID_U_V:
    ((float4*)hits)[rayIdx] = make_float4( t, __int_as_float(triIndex), u, v);
    break;
  case HIT_T_TRIID:
    ((float2*)hits)[rayIdx] = make_float2( t, __int_as_float(triIndex) );
    break;
  case HIT_T:
    ((float*)hits)[rayIdx]  = t;
    break;
  case HIT_BITMASK:
    {
      if( triIndex >= 0 )
      {
        unsigned offset = rayIdx / 32;
        unsigned bit    = (1u << (rayIdx % 32)); 
        ((unsigned*)hits)[offset] |= bit;
      }
      break;
    }
  }
}


//------------------------------------------------------------------------

class TriangleListRemapper
{
public:
  int operator()( const int* remap, const int& hitTriIdx )
  {
    return remap[hitTriIdx] & RLLEPACK_INDEX_MASK;
  }
};

//------------------------------------------------------------------------

class WoopTriangleRemapper
{
public:
  int operator()( const int* remap, const int& hitTriIdx )
  {
    return remap[hitTriIdx];
  }
};

//------------------------------------------------------------------------

template<bool USE_MASKING>
class TriangleListIntersect
{
public:

  struct RayType
  {
    float3  orig;
    float3  dir; 
    float   tmin;
    float   tmax;
    int     mask;
  };

  static float bboxExpansionFactor() 
  {
    return 1.0f;
  }

  static void fetchRay(const void* rays, int rayIdx, int rayFormat, RayType *ray)
  {
    ::fetchRay(rays, rayIdx, rayFormat, &ray->orig, &ray->dir, &ray->tmin, &ray->tmax, &ray->mask);
  }

  static void transformRay(RayType& ray, const float4* im)
  {
    ::transformRay(ray.orig, ray.dir, im);
  }

  void operator()(int& instanceId, int& nodeIdx, const RayType& ray, const TracerDataMesh& bvh, float& hitT, int& hitTriIdx, int& hitInstIdx, float& hitU, float& hitV, bool anyHit)
  {
    const int*    remap    = bvh.remap;   
    const float3* vertices = (float3*)bvh.mesh.vertices; 
    const int3*   indices  = (int3*)bvh.mesh.indices;

    int triIdx = ~nodeIdx;
    int lastPrim = ((unsigned)bvh.remap[triIdx] >> RLLEPACK_LEN_SHIFT) + triIdx;

    do
    {
      int usertriIdx = remap[triIdx] & RLLEPACK_INDEX_MASK;

      // Compute and check intersection t-value.
      float3 p0, p1, p2;
      if( indices )
      {
        if( USE_MASKING )
        {
          int4 v_idx = *(int4*)((char*)indices + ((size_t)usertriIdx * bvh.mesh.indexStride));
          int tri_mask = v_idx.w;
          if( (tri_mask & ray.mask)!=0 ) {
            triIdx++;
            continue;
          }

          p0 = *(float3*)((char*)vertices + (size_t)v_idx.x * bvh.mesh.vertexStride);
          p1 = *(float3*)((char*)vertices + (size_t)v_idx.y * bvh.mesh.vertexStride);
          p2 = *(float3*)((char*)vertices + (size_t)v_idx.z * bvh.mesh.vertexStride);
        }
        else
        {
          int3 v_idx = *(int3*)((char*)indices + ((size_t)usertriIdx * bvh.mesh.indexStride));

          p0 = *(float3*)((char*)vertices + (size_t)v_idx.x * bvh.mesh.vertexStride);
          p1 = *(float3*)((char*)vertices + (size_t)v_idx.y * bvh.mesh.vertexStride);
          p2 = *(float3*)((char*)vertices + (size_t)v_idx.z * bvh.mesh.vertexStride);
        }
      } 
      else 
      {
        int vi = usertriIdx * 3;

        p0 = *(float3*)((char*)vertices + (size_t)(vi+0) * bvh.mesh.vertexStride);
        p1 = *(float3*)((char*)vertices + (size_t)(vi+1) * bvh.mesh.vertexStride);
        p2 = *(float3*)((char*)vertices + (size_t)(vi+2) * bvh.mesh.vertexStride);   
      }  

      const float3 e0 = p1 - p0;
      const float3 e1 = p0 - p2;
      const float3 n  = cross( e1, e0 );

      const float3 e2 = ( 1.0f / dot( n, ray.dir ) ) * ( p0 - ray.orig );
      const float3 i  = cross( ray.dir, e2 );

      float beta  = dot( i, e1 );
      float gamma = dot( i, e0 );
      float t     = dot( n, e2 );

      if( t > ray.tmin && t < hitT )
      {
        if( beta>=0.0f && gamma>=0.0f && beta+gamma<=1.0f )
        {
          // Record intersection.

          hitTriIdx    = triIdx;
          hitInstIdx = instanceId;
          hitT        = t;
          hitU        = 1.0f - beta - gamma;
          hitV        = beta;

          // AnyHit ray => terminate.

          if (anyHit)
          {
            nodeIdx = EntrypointSentinel;
            break;
          }
        }
      }

      triIdx++;
    }
    while(triIdx<lastPrim); // Not the last triangle => keep going.  
  };
};

template<bool USE_MASKING>
class WatertightTriangleListIntersect
{
public:

  struct RayType
  {
    float3  orig;
    float3  dir; 
    float   tmin;
    float   tmax;
    int     mask;

    // Data for watertight intersection
    int     axis;
    float3  translate;
    float3  shear;
  };

  static float bboxExpansionFactor() 
  {
    // Note: this constant is 1 + 3*(ulp) == 1 + 3*2^(-24).
    // It corrects for precision errors in ray-box slab tests.
    // 
    // The following reference suggests 1 + 2*(ulp):
    // Robust BVH Ray Traversal, Thiago Ize, JCGT 2013
    // 
    // Carsten Waechter suggested adding the extra 1*ulp to account for error
    // in the precomputed ray.origOverDir.
     
    return 1.0000003576278687f;
  }

  static void fetchRay(const void* rays, int rayIdx, int rayFormat, RayType *ray)
  {
    ::fetchRay(rays, rayIdx, rayFormat, &ray->orig, &ray->dir, &ray->tmin, &ray->tmax, &ray->mask);

    WatertightOriginal::setupRay(ray->orig, ray->dir,
        ray->axis, ray->translate, ray->shear);
  }

  static void transformRay(RayType& ray, const float4* im)
  {
    ::transformRay(ray.orig, ray.dir, im);

    WatertightOriginal::setupRay(ray.orig, ray.dir,
        ray.axis, ray.translate, ray.shear);
  }

  void operator()(int& instanceId, int& nodeIdx, const RayType& ray, const TracerDataMesh& bvh, float& hitT, int& hitTriIdx, int& hitInstIdx, float& hitU, float& hitV, bool anyHit)
  {
    const int*    remap    = bvh.remap;   
    const float3* vertices = (float3*)bvh.mesh.vertices; 
    const int3*   indices  = (int3*)bvh.mesh.indices;

    int triIdx = ~nodeIdx;
    int lastPrim = ((unsigned)bvh.remap[triIdx] >> RLLEPACK_LEN_SHIFT) + triIdx;

    do
    {
      int usertriIdx = remap[triIdx] & RLLEPACK_INDEX_MASK;

      // Compute and check intersection t-value.
      float3 p0, p1, p2;
      if( indices )
      {
        if( USE_MASKING )
        {
          int4 v_idx = *(int4*)((char*)indices + ((size_t)usertriIdx * bvh.mesh.indexStride));
          int tri_mask = v_idx.w;
          if( (tri_mask & ray.mask)!=0 ) {
            triIdx++;
            continue;
          }

          p0 = *(float3*)((char*)vertices + (size_t)v_idx.x * bvh.mesh.vertexStride);
          p1 = *(float3*)((char*)vertices + (size_t)v_idx.y * bvh.mesh.vertexStride);
          p2 = *(float3*)((char*)vertices + (size_t)v_idx.z * bvh.mesh.vertexStride);
        }
        else
        {
          int3 v_idx = *(int3*)((char*)indices + ((size_t)usertriIdx * bvh.mesh.indexStride));

          p0 = *(float3*)((char*)vertices + (size_t)v_idx.x * bvh.mesh.vertexStride);
          p1 = *(float3*)((char*)vertices + (size_t)v_idx.y * bvh.mesh.vertexStride);
          p2 = *(float3*)((char*)vertices + (size_t)v_idx.z * bvh.mesh.vertexStride);
        }
      } 
      else 
      {
        int vi = usertriIdx * 3;

        p0 = *(float3*)((char*)vertices + (size_t)(vi+0) * bvh.mesh.vertexStride);
        p1 = *(float3*)((char*)vertices + (size_t)(vi+1) * bvh.mesh.vertexStride);
        p2 = *(float3*)((char*)vertices + (size_t)(vi+2) * bvh.mesh.vertexStride);   
      }  

      float t, u, v;
      if ( WatertightOriginal::intersectTriangle(ray.axis, ray.translate, ray.shear, p0, p1, p2, ray.tmin, hitT, &t, &u, &v) ) {

          // Record intersection.
          hitTriIdx    = triIdx;
          hitInstIdx = instanceId;
          hitT        = t;
          hitU        = u;
          hitV        = v;

          // AnyHit ray => terminate.

          if (anyHit) {
            nodeIdx = EntrypointSentinel;
            break;
          }
      }

      triIdx++;
    }
    while(triIdx<lastPrim); // Not the last triangle => keep going.  
  };
};

//------------------------------------------------------------------------

class WoopTriangleIntersect
{
public:

  struct RayType
  {
    float3  orig;
    float3  dir; 
    float   tmin;
    float   tmax;
  };
  
  static float bboxExpansionFactor() 
  {
    return 1.0f;
  }

  static void fetchRay(const void* rays, int rayIdx, int rayFormat, RayType *ray)
  {
    ::fetchRay(rays, rayIdx, rayFormat, &ray->orig, &ray->dir, &ray->tmin, &ray->tmax);
  }

  static void transformRay(RayType& ray, const float4* im)
  {
    ::transformRay(ray.orig, ray.dir, im);
  }

  void operator()(int& instanceId, int& nodeIdx, const RayType& ray, const TracerDataMesh& bvh, float& hitT, int& hitTriIdx, int& hitInstIdx, float& hitU, float& hitV, bool anyHit)
  {
    // Intersect the ray against each triangle using Sven Woop's algorithm.

    int triIdx = ~nodeIdx;
    float4 woopT;
    do
    {
      // Compute and check intersection t-value.

      const WoopTriangle& tri = bvh.woopTriangles[triIdx];
      const float3& orig = ray.orig;
      const float3& dir = ray.dir;
      woopT = tri.t;
      float Oz = woopT.w - orig.x*woopT.x - orig.y*woopT.y - orig.z*woopT.z;
      float ilwDz = 1.0f / (dir.x*woopT.x + dir.y*woopT.y + dir.z*woopT.z);
      float t = Oz * ilwDz;

      if (t > ray.tmin && t < hitT)
      {
        // Compute and check barycentric u.

        float Ox = tri.u.w + orig.x*tri.u.x + orig.y*tri.u.y + orig.z*tri.u.z;
        float Dx = dir.x*tri.u.x + dir.y*tri.u.y + dir.z*tri.u.z;
        float beta = Ox + t*Dx;

        if (beta >= 0.0f && beta <= 1.0f)
        {
          // Compute and check barycentric v.

          float Oy = tri.v.w + orig.x*tri.v.x + orig.y*tri.v.y + orig.z*tri.v.z;
          float Dy = dir.x*tri.v.x + dir.y*tri.v.y + dir.z*tri.v.z;
          float gamma = Oy + t*Dy;

          if (gamma >= 0.0f && beta + gamma <= 1.0f)
          {
            // Record intersection.
            hitTriIdx    = triIdx;
            hitInstIdx = instanceId;
            hitT        = t;
            hitU        = 1.0f - beta - gamma;
            hitV        = beta;

            // AnyHit ray => terminate.

            if (anyHit)
            {
              nodeIdx = EntrypointSentinel;
              break;
            }
          }
        }
      }
      triIdx++;
    }
    while ((*(const int*)&woopT.w & 1) == 0); // Not the last triangle => keep going.
  };
};

//------------------------------------------------------------------------

template<class Intersector, class Remapper>
__forceinline__ void traceSingleRay(int instanceId, const typename Intersector::RayType& ray, float& hitT, const TracerDataMesh& bvh, int& hitTriIdx, int& hitInstIdx, float& hitU, float& hitV, bool anyHit)
{
  int traversalStack[STACK_SIZE];
  traversalStack[0] = EntrypointSentinel;

  int stackPtr = 0;           // Current position in traversal stack.to pop
  int nodeIdx = 0;           // Non-negative: current internal node, negative: second postponed leaf. from the root.

  const BvhNode* nodes      = bvh.nodes;  
  const bool     optixNodes = bvh.optixNodes;

  if( !nodes || bvh.mesh.numTriangles==0 )
    nodeIdx = EntrypointSentinel;

  float origx = ray.orig.x;    // Ray origin.
  float origy = ray.orig.y;
  float origz = ray.orig.z;
  float dirx = ray.dir.x;      // Ray direction.
  float diry = ray.dir.y;
  float dirz = ray.dir.z;

  float ooeps = 1e-20f;       // 1 / dir
  float idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : (dirx>=0 ? ooeps : -ooeps));
  float idiry = 1.0f / (fabsf(diry) > ooeps ? diry : (diry>=0 ? ooeps : -ooeps));
  float idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : (dirz>=0 ? ooeps : -ooeps));
  float oodx = origx * idirx;
  float oody = origy * idiry;
  float oodz = origz * idirz; // orig / dir

  const float bboxExpansionFactor = Intersector::bboxExpansionFactor();

  // traversal loop.

  while (nodeIdx != EntrypointSentinel)
  {
    // Traverse internal nodes until all SIMD lanes have found a leaf.

    while (nodeIdx >= 0 && nodeIdx != EntrypointSentinel)
    {
      // Intersect the ray against the child nodes.

      const BvhNode& node = nodes[nodeIdx];
      float c0lox = node.c0lox * idirx - oodx;
      float c0hix = node.c0hix * idirx - oodx;
      float c0loy = node.c0loy * idiry - oody;
      float c0hiy = node.c0hiy * idiry - oody;
      float c0loz = node.c0loz * idirz - oodz;
      float c0hiz = node.c0hiz * idirz - oodz;
      float c1lox = node.c1lox * idirx - oodx;
      float c1hix = node.c1hix * idirx - oodx;
      float c1loy = node.c1loy * idiry - oody;
      float c1hiy = node.c1hiy * idiry - oody;
      float c1loz = node.c1loz * idirz - oodz;
      float c1hiz = node.c1hiz * idirz - oodz;

      float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), ray.tmin);
      float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
      float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), ray.tmin);
      float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

      bool traverseChild0 = (c0max*bboxExpansionFactor >= c0min);
      bool traverseChild1 = (c1max*bboxExpansionFactor >= c1min);

      // Neither child was intersected => pop stack.

      if (!traverseChild0 && !traverseChild1)
      {
        nodeIdx = traversalStack[stackPtr];
        stackPtr--;
      }

      // Otherwise => fetch child pointers.

      else
      {
        int2 cnodes = make_int2(node.c0idx, node.c1idx);
        if( optixNodes ) // Translate from Optix to lwray nodes
        {
          cnodes = make_int2( 
            node.c0idx != ~0 ? ~node.c0idx : node.c0num, 
            node.c1idx != ~0 ? ~node.c1idx : node.c1num );
        }
        nodeIdx = (traverseChild0) ? cnodes.x : cnodes.y;

        // Both children were intersected => push the farther one.

        if (traverseChild0 && traverseChild1)
        {
          if (c1min < c0min)
            std::swap(nodeIdx, cnodes.y);
          stackPtr++;
          traversalStack[stackPtr] = cnodes.y;
        }
      }
    }   // inner

    // Process leaf nodes.
    while (nodeIdx < 0)
    {
      Intersector intersector;
      intersector( instanceId, nodeIdx, ray, bvh, hitT, hitTriIdx, hitInstIdx, hitU, hitV, anyHit );

      // Pop stack.
      if( nodeIdx < 0 ) {
        nodeIdx = traversalStack[stackPtr];
        stackPtr--;
      }
    } // leaf
  } // traversal
}

//------------------------------------------------------------------------

template<class Intersector, class Remapper>
void trace(const TracerDataMesh& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit)
{
  if( bvh.inDeviceMem )
    throw IlwalidValue( RT_EXCEPTION_INFO, "Can't trace GPU bvh on host" );

  // Loop over rays.

  for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
  {
    // Fetch and initialize ray.

    typename Intersector::RayType ray;
    Intersector::fetchRay( rays, rayIdx, rayFormat, &ray );

    // Hit data

    int     hitTriIdx    = -1;          // Triangle index of the closest intersection, -1 if none.iangle intersected so far.
    int     hitInstIdx = -1;
    float   hitT        = ray.tmax;        // t-value of the closest intersection.
    float   hitU        = 0.0f;
    float   hitV        = 0.0f;

    traceSingleRay<Intersector, Remapper>(-1, ray, hitT, bvh, hitTriIdx, hitInstIdx, hitU, hitV, anyHit);

    // Remap intersected triangle index.
    Remapper remapper;
    if(hitTriIdx != -1) {
      hitTriIdx = remapper( bvh.remap, hitTriIdx );
      hitInstIdx = 0;
    }
    else
      hitT = -1;

    // Store the result.
    storeHit( hits, rayIdx, hitFormat, hitTriIdx, hitInstIdx, hitT, hitU, hitV );
  }
}

template<class Intersector, class Remapper>
void trace2(const TracerDataGroup& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit)
{
  if( bvh.inDeviceMem )
    throw IlwalidValue( RT_EXCEPTION_INFO, "Can't trace GPU bvh on host" );

  // traversal stack
  int traversalStack[STACK_SIZE];
  traversalStack[0] = EntrypointSentinel;

  const float bboxExpansionFactor = Intersector::bboxExpansionFactor();

  // loop over rays.
  for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
  {
    // setup traversal.
    int  stackPtr = 0;

    // fetch and initialize ray data
    typename Intersector::RayType ray;
    Intersector::fetchRay( rays, rayIdx, rayFormat, &ray );

    float origx = ray.orig.x;
    float origy = ray.orig.y;
    float origz = ray.orig.z;
    float dirx = ray.dir.x;
    float diry = ray.dir.y;
    float dirz = ray.dir.z;

    float ooeps = 1e-20f;
    float idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : (dirx>=0 ? ooeps : -ooeps));
    float idiry = 1.0f / (fabsf(diry) > ooeps ? diry : (diry>=0 ? ooeps : -ooeps));
    float idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : (dirz>=0 ? ooeps : -ooeps));
    float oodx = origx * idirx;
    float oody = origy * idiry;
    float oodz = origz * idirz;

    // hit infos
    float   hitT        = ray.tmax;
    float   hitU        = 0.0f;
    float   hitV        = 0.0f;

    int     hitTriIdx  = -1;           // Triangle index of the closest intersection, -1 if none.
    int     hitInstIdx = -1;           // Instance index of the closest intersection, -1 if none

    int     instIdx = -1;

    int      nodeIdx = 0;
    int      state    = 0;
    BvhNode* nodes    = bvh.nodes;

    if( !nodes || bvh.group.numInstances==0 )
      nodeIdx = EntrypointSentinel;

    // outer traversal loop
    do
    {
      // traverse internal nodes until found a leaf
      while((unsigned int)nodeIdx < (unsigned int)EntrypointSentinel)   // functionally equivalent to but faster than: while (nodeIdx >= 0 && nodeIdx != EntrypointSentinel)
      {        
        const BvhNode& node = nodes[nodeIdx];
        float c0lox = node.c0lox * idirx - oodx;
        float c0hix = node.c0hix * idirx - oodx;
        float c0loy = node.c0loy * idiry - oody;
        float c0hiy = node.c0hiy * idiry - oody;
        float c0loz = node.c0loz * idirz - oodz;
        float c0hiz = node.c0hiz * idirz - oodz;
        float c1lox = node.c1lox * idirx - oodx;
        float c1hix = node.c1hix * idirx - oodx;
        float c1loy = node.c1loy * idiry - oody;
        float c1hiy = node.c1hiy * idiry - oody;
        float c1loz = node.c1loz * idirz - oodz;
        float c1hiz = node.c1hiz * idirz - oodz;

        float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), ray.tmin);
        float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
        float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), ray.tmin);
        float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

        bool traverseChild0 = (c0max*bboxExpansionFactor >= c0min);
        bool traverseChild1 = (c1max*bboxExpansionFactor >= c1min);

        if( !traverseChild0 && !traverseChild1 )
        { // neither child was intersected => pop stack.
          nodeIdx = traversalStack[stackPtr];
          stackPtr--;
        }
        else
        { // otherwise => fetch child pointers.
          int2 cnodes = make_int2(node.c0idx, node.c1idx);
          nodeIdx = (traverseChild0) ? cnodes.x : cnodes.y;

          if( traverseChild0 && traverseChild1 )
          { // both children were intersected => push the farther one.
            if( c1min < c0min )
              std::swap( nodeIdx, cnodes.y );

            stackPtr++;
            traversalStack[stackPtr] = cnodes.y;
          }
        }
      }

      if(state==0 && nodeIdx<0) // transform ray and go back to traverse
      {
        int   instAddr = ~(nodeIdx);

        instIdx = bvh.remap[instAddr] & RLLEPACK_INDEX_MASK;

        int    modelId = bvh.group.modelIds[instIdx];

        const float4* ilwTransform = getTransformPtr( bvh.group.ilwMatrices, instIdx, bvh.group.matrixStride );      

        // transform ray
        Intersector::transformRay( ray, ilwTransform );
        origx = ray.orig.x;
        origy = ray.orig.y;
        origz = ray.orig.z;

        dirx = ray.dir.x;
        diry = ray.dir.y;
        dirz = ray.dir.z;

        ooeps = 1e-20f;
        idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : (dirx>=0 ? ooeps : -ooeps));
        idiry = 1.0f / (fabsf(diry) > ooeps ? diry : (diry>=0 ? ooeps : -ooeps));
        idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : (dirz>=0 ? ooeps : -ooeps));
        oodx = origx * idirx;
        oody = origy * idiry;
        oodz = origz * idirz;

        stackPtr++;
        traversalStack[stackPtr] = EntrypointSentinel;

        nodeIdx      = 0;
        nodes         = bvh.meshes[modelId].nodes;

        state++;
      
        if( bvh.meshes[modelId].mesh.numTriangles==0 )
          nodeIdx = EntrypointSentinel;      
      } 
      else if(state==1 && nodeIdx<0) // intersect triangles
      {
        int modelId = bvh.group.modelIds[instIdx];

        Intersector intersector;
        intersector( instIdx, nodeIdx, ray, bvh.meshes[modelId], hitT, hitTriIdx, hitInstIdx, hitU, hitV, anyHit );

        nodeIdx = traversalStack[stackPtr];
        stackPtr--;
      }
      else if(state==1 && nodeIdx==EntrypointSentinel) // after isec, proceed with top level traversal
      {
        state       = 0;
        
        nodeIdx    = traversalStack[stackPtr];
        stackPtr--;

        if(nodeIdx!=EntrypointSentinel)
        {
          Intersector::fetchRay( rays, rayIdx, rayFormat, &ray );

          origx = ray.orig.x;
          origy = ray.orig.y;
          origz = ray.orig.z;

          dirx = ray.dir.x;
          diry = ray.dir.y;
          dirz = ray.dir.z;

          ooeps = 1e-20f;
          idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : (dirx>=0 ? ooeps : -ooeps));
          idiry = 1.0f / (fabsf(diry) > ooeps ? diry : (diry>=0 ? ooeps : -ooeps));
          idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : (dirz>=0 ? ooeps : -ooeps));
          oodx  = origx * idirx;
          oody  = origy * idiry;
          oodz  = origz * idirz;

          nodes = bvh.nodes;              
        }
      }
    }
    while(!((nodeIdx==EntrypointSentinel) && (state==0))); // traversal

    // Remap intersected triangle index.
    Remapper remapper;
    if( hitTriIdx != -1 && hitInstIdx != -1 ) {
      int modelId = bvh.group.modelIds[hitInstIdx];
      int* remap  = bvh.meshes[modelId].remap;
      hitTriIdx = remapper( remap, hitTriIdx );
    } else {
      hitT = -1;
    }

    // Store the result.
    storeHit( hits, rayIdx, hitFormat, hitTriIdx, hitInstIdx, hitT, hitU, hitV );
  } // end ray loop
}

//------------------------------------------------------------------------

void launch1LevelBvh(const TracerDataMesh& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const unused)
{
  if( bvh.woopTriangles )
  {
    trace<WoopTriangleIntersect, WoopTriangleRemapper>( bvh, rays, rayFormat, hits, hitFormat, numRays, anyHit );
  }
  else
  {
    if( rayFormat == RAY_ORIGIN_MASK_DIRECTION_TMAX ) 
    {
      if ( watertight )
      {
        trace<WatertightTriangleListIntersect<true>, TriangleListRemapper>( bvh, rays, rayFormat, hits, hitFormat, numRays, anyHit );
      }
      else 
      {
        trace<TriangleListIntersect<true>, TriangleListRemapper>( bvh, rays, rayFormat, hits, hitFormat, numRays, anyHit );
      }
    } 
    else 
    {
      if ( watertight )
      {
        trace<WatertightTriangleListIntersect<false>, TriangleListRemapper>( bvh, rays, rayFormat, hits, hitFormat, numRays, anyHit );
      }
      else
      {
        trace<TriangleListIntersect<false>, TriangleListRemapper>( bvh, rays, rayFormat, hits, hitFormat, numRays, anyHit );
      }
    }
  }
}

//------------------------------------------------------------------------

void launch2LevelBvh(const TracerDataGroup& tdg, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const unused)
{
  RT_ASSERT( tdg.optixNodes==false );

  if( !tdg.meshes.empty() && tdg.meshes[0].woopTriangles )
  {
    trace2<WoopTriangleIntersect, WoopTriangleRemapper>( tdg, rays, rayFormat, hits, hitFormat, numRays, anyHit );
  }
  else
  {
    if( rayFormat == RAY_ORIGIN_MASK_DIRECTION_TMAX ) 
    {
      if ( watertight ) 
      {
        trace2<WatertightTriangleListIntersect<true>, TriangleListRemapper>( tdg, rays, rayFormat, hits, hitFormat, numRays, anyHit );    
      }
      else
      {
        trace2<TriangleListIntersect<true>, TriangleListRemapper>( tdg, rays, rayFormat, hits, hitFormat, numRays, anyHit );    
      }
    } 
    else 
    {
      if ( watertight )
      {
        trace2<WatertightTriangleListIntersect<false>, TriangleListRemapper>( tdg, rays, rayFormat, hits, hitFormat, numRays, anyHit );    
      }
      else
      {
        trace2<TriangleListIntersect<false>, TriangleListRemapper>( tdg, rays, rayFormat, hits, hitFormat, numRays, anyHit );    
      }
    }
  }
}

//------------------------------------------------------------------------

void TracerNaive::traceFromHostMem(const TracerData& td, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const unused)
{
  if( const TracerDataMesh* tdm = dynamic_cast<const TracerDataMesh*>(&td) )
  {
    launch1LevelBvh( *tdm, rays, rayFormat, hits, hitFormat, numRays, anyHit, watertight, unused );
  } 
  else 
  {
    const TracerDataGroup& tdg = dynamic_cast<const TracerDataGroup&>(td);
    launch2LevelBvh( tdg, rays, rayFormat, hits, hitFormat, numRays, anyHit, watertight, unused );
  } 
}
