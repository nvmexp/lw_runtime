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

#include "TracerTypes.hpp"
#include "../common/RemapListLenEncoding.hpp"

//------------------------------------------------------------------------

// Macros to append "32" or "64" to function names, based on a compiler flag.
// We may build .obj/.h files for both 32-bit addressing and
// 64-bit addressing from the same .ispc source file.  Any non-inlined functions
// must have unique names if the .obj files are linked together, and ispc
// lacks namespaces.

// Passed with the -D flag to ispc
#ifdef ISPC_64_BIT_ADDRESSING
#define ISPC_SUFFIX 64
#else
#define ISPC_SUFFIX 32
#endif

// Note: this indirection is needed to make the C preprocessor expand ISPC_SUFFIX
#define CONCAT(x,y) x ## y
#define EVAL_CONCAT(x, y) CONCAT(x, y)
#define EXPORT_FUNCTION(name) export void EVAL_CONCAT( name, ISPC_SUFFIX )
#define RENAME_FUNCTION(name) EVAL_CONCAT( name, ISPC_SUFFIX )

//------------------------------------------------------------------------

struct TracerParamsMesh
{
  uniform const float<4>* nodes;
  uniform const float<4>* triangles;
  uniform const int*      indices;
  uniform const float*    vertices;
  uniform const int*      remap;
  uniform int             indexStride;
  uniform int             vertexStride;
  uniform int             rootNode;
  uniform int             numEntities;
};

//------------------------------------------------------------------------

struct TracerParamsGroup
{
  uniform const float<4>* nodes;
  uniform const int*      remap;
  uniform const float*    ilwMatrices;
  uniform const int*      modelId;
  uniform int             rootNode;
  uniform int             numEntities;
  uniform int             matrixStride;
};

//------------------------------------------------------------------------

// WARNING: this all needs to be compatible with the KernelParameters struct
struct tracerIspcParams
{
  uniform const float<4>*      rays;
  uniform const float*         hits;
  uniform TracerParamsGroup*   group;
  uniform TracerParamsMesh*    meshes;
  uniform int                  numRays;
  uniform int                  rayFormat;
  uniform int                  hitFormat;
  uniform int                  anyhit;
  uniform int                  watertight;
  uniform int                  isecMode;
  uniform bool                 useMasking;
  uniform int                  endMarker;          // Use &endMarker instead of sizeof() to avoid uploading too much data.
};

 //------------------------------------------------------------------------

#define STACK_SIZE          64          // Size of the traversal stack.
#define EntrypointSentinel  0x76543210  // Bottom-most stack entry, indicating the end of traversal.

//------------------------------------------------------------------------

inline float min4(float a, float b, float c, float d)
{
    return min(min(a, b), min(c, d));
}

inline float max4(float a, float b, float c, float d)
{
    return max(max(a, b), max(c, d));
}

//------------------------------------------------------------------------------

static inline
void storeHit( void* hits, int rayIdx, int hitFormat, int triIndex, int instIndex, float t, float u, float v )
{
  // PLEASE: do not modify this code except you know what you are doing.
  // There are several compiler bugs in ISPC that produce bad code and let the kernels crash.
  // E.g. changing HIT_T_TRIID to use int2/float2 to have a more efficient store does lwrrently crash!
  // Test your changes always with test_prime!!!

  switch( hitFormat )
  {
  case HIT_T_TRIID_INSTID:
  {
      int hit_t = intbits(t);
      int tidx   = triIndex;
      int iidx   = instIndex;

      int* hitPtr = (int*)hits;
      hitPtr[rayIdx*3+0] = hit_t;
      hitPtr[rayIdx*3+1] = tidx;
      hitPtr[rayIdx*3+2] = iidx;
  }
  break;
  case HIT_T_TRIID_INSTID_U_V:
  {
      int hit_t = intbits(t);
      int tidx   = triIndex;
      int iidx   = instIndex;
      int   iu   = intbits(u);
      int   iv   = intbits(v);

      int* hitPtr = (int*)hits;
      hitPtr[rayIdx*5+0] = hit_t;
      hitPtr[rayIdx*5+1] = tidx;
      hitPtr[rayIdx*5+2] = iidx;
      hitPtr[rayIdx*5+3] = iu;
      hitPtr[rayIdx*5+4] = iv;
  }
  break;
  case HIT_T_TRIID_U_V:
    {
        int<4> hit = { intbits(t), triIndex, intbits(u), intbits(v) };

        int<4>* hitPtr = (int<4>*)hits;  
        hitPtr[rayIdx] = hit.xyzw;
    }    
    break;
  case HIT_T_TRIID:
    {
      int hit_t = intbits(t);
      int idx   = triIndex;
      
      int* hitPtr = (int*)hits;
      hitPtr[rayIdx*2+0] = hit_t;
      hitPtr[rayIdx*2+1] = idx;
     }
     break;
  case HIT_T:
      {
        float* hitPtr  = (float*)hits;  
        hitPtr[rayIdx] = t;
      }
      break;
  case HIT_BITMASK:
    {
      if( triIndex >= 0 )
      {
        unsigned int offset = rayIdx / 32;
        unsigned int bit    = (1u << (rayIdx % 32));
        unsigned int* hitPtr = (unsigned int*)hits;
        atomic_or_global( &hitPtr[offset], bit );
      }
      break;
    }
  }
}


inline float<3> cross(const float<3>& a, const float<3>& b)
{
  float<3> v;
  v[0] = a.y*b.z - a.z*b.y;
  v[1] = a.z*b.x - a.x*b.z;
  v[2] = a.x*b.y - a.y*b.x;
  return v;
}

inline float dot(const float<3>& a, const float<3>& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}



/// Regular intersection

struct RayData {
  float<3> dir;
  float<3> orig;
};

inline void setupRayData(const float<4>& orig, const float<4>& dir, RayData& ray)
{
  ray.dir.x = dir.x;
  ray.dir.y = dir.y;
  ray.dir.z = dir.z;

  ray.orig.x = orig.x;
  ray.orig.y = orig.y;
  ray.orig.z = orig.z;
}

inline bool intersectTriangle(void* uniform  rayData,
                              const float<3>& p0, 
                              const float<3>& p1, 
                              const float<3>& p2, 
                              const float tmin,
                              float& hitT,  //inout
                              float& hitU,  //out
                              float& hitV)  //out
{
  const varying RayData* uniform ray = (const varying RayData* uniform)rayData;
  const float<3> dir = ray->dir;
  const float<3> orig = ray->orig;

  const float<3> e0 = p1 - p0;
  const float<3> e1 = p0 - p2;

  const float<3> n  = cross( e1, e0 );

  const float<3> e2 = ( 1.0f / dot( n, dir ) ) * ( p0 - orig );
  const float<3> i  = cross( dir, e2 );

  float beta  = dot( i, e1 );
  float gamma = dot( i, e0 );
  float t     = dot( n, e2 );

  if( t > tmin && t < hitT && !(t != t)) // use for ispc 1.7x
    //            if( t > tmin && t < hitT )
  {
    if( beta>=0.0f && gamma>=0.0f && beta+gamma<=1.0f )
    {
      hitT = t;
      hitU = 1.0f - beta - gamma;
      hitV = beta;
      return true;
    }
  }
  return false;
}


/// Watertight intersection

inline float<3> swizzle(float x, float y, float z, int axis)
{
  float<3> vout = {x, y, z};
  if (axis == 0) {
    vout.x = y; vout.y = z; vout.z = x;
  } else if (axis == 1) {
    vout.x = z; vout.y = x; vout.z = y;
  }
  return vout;
}

struct RayDataWatertight {
  int axis;
  float<3> translate;
  float<3> shear;
};

inline void setupRayDataWatertight(const float<4>& orig, const float<4>& dir,
  RayDataWatertight& ray)
{
    // Choose main axis.

    float ax = abs(dir.x);
    float ay = abs(dir.y);
    float az = abs(dir.z);
    int axis = (ax >= ay && ax >= az) ? 0 : (ay >= az) ? 1 : 2;

    // Swizzle origin and direction.

    float<3> so = swizzle(orig.x, orig.y, orig.z, axis);
    float<3> sd = swizzle(dir.x, dir.y, dir.z, axis);
    float rz = rcp(sd.z);

    // Output ray.

    ray.axis = axis;
    ray.translate.x = so.x;
    ray.translate.y = so.y;
    ray.translate.z = so.z;
    ray.shear.x = rz * sd.x;
    ray.shear.y = rz * sd.y;
    ray.shear.z = rz;
}

inline bool intersectTriangleWatertight(void* uniform rayData,
                                        const float<3>& v0, 
                                        const float<3>& v1, 
                                        const float<3>& v2, 
                                        const float tmin, 
                                        float& hitT, //inout
                                        float& hitU, //out
                                        float& hitV) //out
{

  const varying RayDataWatertight* uniform ray = (const varying RayDataWatertight* uniform)rayData;
  const int axis = ray->axis;
  const float<3> translate = ray->translate;
  const float<3> shear = ray->shear;

  // Swizzle the components of triangle vertices.

  float ax = (axis == 0) ? v0.y : (axis == 1) ? v0.z : v0.x;
  float ay = (axis == 0) ? v0.z : (axis == 1) ? v0.x : v0.y;
  float az = (axis == 0) ? v0.x : (axis == 1) ? v0.y : v0.z;
  float bx = (axis == 0) ? v1.y : (axis == 1) ? v1.z : v1.x;
  float by = (axis == 0) ? v1.z : (axis == 1) ? v1.x : v1.y;
  float bz = (axis == 0) ? v1.x : (axis == 1) ? v1.y : v1.z;
  float cx = (axis == 0) ? v2.y : (axis == 1) ? v2.z : v2.x;
  float cy = (axis == 0) ? v2.z : (axis == 1) ? v2.x : v2.y;
  float cz = (axis == 0) ? v2.x : (axis == 1) ? v2.y : v2.z;

  // Translate, shear, and scale the vertices.

  az -= translate.z;
  bz -= translate.z;
  cz -= translate.z;
  ax += (-shear.x)*az - translate.x;
  ay += (-shear.y)*az - translate.y;
  bx += (-shear.x)*bz - translate.x;
  by += (-shear.y)*bz - translate.y;
  cx += (-shear.x)*cz - translate.x;
  cy += (-shear.y)*cz - translate.y;

  // Compute scaled barycentric coordinates.

  float u = cx*by - cy*bx;
  float v = ax*cy - ay*cx;
  float w = bx*ay - by*ax;

  // Uncertain result => fall back to double precision.

  if ( (u == 0.0f) || (v == 0.0f) || (w == 0.0f) )
  {
    u = (float)((double)cx * (double)by - (double)cy * (double)bx);
    v = (float)((double)ax * (double)cy - (double)ay * (double)cx);
    w = (float)((double)bx * (double)ay - (double)by * (double)ax);
  }

  // Outside the triangle => miss.

  float denom = u + v + w;
  if ((( (u < 0.0f) || (v < 0.0f) || (w < 0.0f) ) && 
       ( (u > 0.0f) || (v > 0.0f) || (w > 0.0f) )) || (denom == 0.0f) )
  {
    return false;
  }

  // Compute intersection t-value.

  az *= shear.z;
  bz *= shear.z;
  cz *= shear.z;
  const float rcpDet = 1.0f / denom;
  float num = u*az + v*bz + w*cz;
  float t = num*rcpDet;

  // Outside the current t-range => miss.

  if ( (t <= tmin) || (t >= hitT) ) {
    return false;
  }

  // Otherwise => hit.
  hitT = t;
  hitU = u*rcpDet;
  hitV = v*rcpDet;
  return true;
}


typedef bool (*IntersectionFunctionType)(void* uniform, const float<3>&, const float<3>&, const float<3>&, const float, float&, float&, float&);


// Note: declaring vertices[] array like this because the pointer can be varying for 2-level traversal.
inline void loadTriangle(uniform const float * const varying vertices, const int vertexStride, const int vx, const int vy, const int vz,
  float<3>& p0, float<3>& p1, float<3>& p2)
{
  int8* v_base = (int8*)vertices;
  float* p0_base = (float*)(v_base + (size_t)vx * vertexStride);
  float p0x = p0_base[0];
  float p0y = p0_base[1];
  float p0z = p0_base[2];

  float* p1_base = (float*)(v_base + (size_t)vy * vertexStride);
  float p1x = p1_base[0];
  float p1y = p1_base[1];
  float p1z = p1_base[2];

  float* p2_base = (float*)(v_base + (size_t)vz * vertexStride);
  float p2x = p2_base[0];
  float p2y = p2_base[1];
  float p2z = p2_base[2];

  p0.x = p0x; p0.y = p0y, p0.z = p0z;
  p1.x = p1x; p1.y = p1y, p1.z = p1z;
  p2.x = p2x; p2.y = p2y, p2.z = p2z;
}


inline void intersectTriangleList(const int instanceIdx, 
                                  int& nodeIdx, 
                                  const uniform int* const varying remap,
                                  const uniform float* const varying vertices,
                                  const int vertexStride,
                                  uniform IntersectionFunctionType intersector,
                                  void* uniform rayData,
                                  float& tmin,
                                  float& hitT,
                                  int& hitTriIndex,
                                  int& hitInstanceIndex,
                                  float& hitU,
                                  float& hitV,
                                  const uniform bool anyHit,
                                  const int traversalStack[],
                                  varying int* const uniform stackPtr)
{
  while (nodeIdx < 0)
  {
    int triIdx = ~nodeIdx;
    int lastPrim = ((unsigned int)remap[triIdx] >> RLLEPACK_LEN_SHIFT) + triIdx;
    do
    {
      int usertriIdx = remap[triIdx] & RLLEPACK_INDEX_MASK;

      // Compute and check intersection t-value.
      int vx = usertriIdx * 3 + 0;
      int vy = usertriIdx * 3 + 1;
      int vz = usertriIdx * 3 + 2;

      float<3> p0, p1, p2;
      loadTriangle(vertices, vertexStride, vx, vy, vz, p0, p1, p2);

      if (intersector(rayData, p0, p1, p2, tmin, hitT, hitU, hitV)) {
        // Record intersection.

        hitTriIndex = triIdx;
        hitInstanceIndex = instanceIdx;

        // AnyHit ray => terminate.

        if (anyHit) {
          nodeIdx = EntrypointSentinel;
          break;
        }                      
      }
      triIdx++;
    }
    while(triIdx<lastPrim); // Not the last triangle => keep going.

    // Pop stack.
  
    if (traversalStack == NULL) return;
    nodeIdx = traversalStack[*stackPtr];
    (*stackPtr)--;
  } // leaf
} 

inline void intersectIndexedTriangleList(const int instanceIdx,
                                         int& nodeIdx,
                                         int ray_mask,
                                         const uniform int* const varying remap, 
                                         const uniform int* const varying indices, 
                                         const uniform float* const varying vertices,
                                         const int indexStride,
                                         const int vertexStride,
                                         uniform IntersectionFunctionType intersector,
                                         void* uniform rayData,
                                         float& tmin,
                                         float& hitT,
                                         int& hitTriIndex,
                                         int& hitInstanceIndex,
                                         float& hitU,
                                         float& hitV,
                                         const uniform bool anyHit,
                                         const uniform bool useMasking,
                                         const int traversalStack[],
                                         varying int* const uniform stackPtr)
{
  while (nodeIdx < 0)
  {
    // Intersect the ray against each triangle.

    int triIdx = ~nodeIdx;
    int lastPrim = ((unsigned int)remap[triIdx] >> RLLEPACK_LEN_SHIFT) + triIdx;
    do
    {
      int usertriIdx = remap[triIdx] & RLLEPACK_INDEX_MASK;

      // Compute and check intersection t-value.
      int8* i_base  = (int8*)indices;
      int*  ii_base = (int*)(i_base + (size_t)usertriIdx * indexStride);

      int vx = ii_base[0];
      int vy = ii_base[1];
      int vz = ii_base[2];

      if( useMasking )
      {
        int vm = ii_base[3];

        if( (vm & ray_mask)!=0 ) {
          triIdx++;
          continue;
        }      
      }

      float<3> p0, p1, p2;
      loadTriangle(vertices, vertexStride, vx, vy, vz, p0, p1, p2);

      if (intersector(rayData, p0, p1, p2, tmin, hitT, hitU, hitV)) {
        // Record intersection.

        hitTriIndex = triIdx;
        hitInstanceIndex = instanceIdx;

        // AnyHit ray => terminate.

        if (anyHit)
        {
          nodeIdx = EntrypointSentinel;
          break;
        }                      
      }
      triIdx++;
    }
    while(triIdx<lastPrim); // Not the last triangle => keep going.

    // Pop stack.

    if (traversalStack == NULL) return;
    nodeIdx = traversalStack[*stackPtr];
    (*stackPtr)--;

  } // leaf
}

inline void intersectWoopTriangles(const int instanceIdx,
                                   int& nodeIdx,
                                   const uniform int* const varying remap,
                                   const uniform float<4>* const varying woopTriangles,
                                   const float dirx, 
                                   const float diry, 
                                   const float dirz,
                                   const float origx,
                                   const float origy,
                                   const float origz,
                                   const float tmin,
                                   float& hitT,
                                   int& hitTriIndex,
                                   int& hitInstanceIndex,
                                   float& hitU,
                                   float& hitV,
                                   const uniform bool anyHit,
                                   const int traversalStack[],
                                   varying int* const uniform stackPtr)
{
    while (nodeIdx < 0)
    {
        // Intersect the ray against each triangle using Sven Woop's algorithm.

        int triIdx = ~nodeIdx;
        int flag;
        do
        {
            // Fetch triangle header.

            const float<4>* trianglePtr = &woopTriangles[triIdx * 3];
            const int<4>* headerPtr = (const int<4>*)trianglePtr;
            int<4> header = *headerPtr;
            flag = header.w;
            float woopTx = floatbits(header.x);
            float woopTy = floatbits(header.y);
            float woopTz = floatbits(header.z);
            float woopTw = floatbits(header.w);

            // Compute and check intersection t-value.
            // Note: The "!(t != t)" conditions below are necessary to ensure that
            // the value is not NaN. Normally, these would not be needed since
            // according to the spec, e.g. "t > tmin" returns false if t is NaN.
            // But ISPC 1.3.0 seems to return true instead.

            float Oz = woopTw - origx*woopTx - origy*woopTy - origz*woopTz;
            float ilwDz = 1.0f / (dirx*woopTx + diry*woopTy + dirz*woopTz);
            float t = Oz * ilwDz;

//            if (t > tmin && t < hitT) // ISPC 1.2.0
                    if (t > tmin && t < hitT && !(t != t)) // ISPC 1.3.0
            {
                // Compute and check barycentric u.

                float<4> woopU = trianglePtr[1];
                float Ox = woopU.w + origx*woopU.x + origy*woopU.y + origz*woopU.z;
                float Dx = dirx*woopU.x + diry*woopU.y + dirz*woopU.z;
                float beta = Ox + t*Dx;

                if (beta >= 0.0f && beta <= 1.0f) // ISPC 1.2.0
//                        if (beta >= 0.0f && beta <= 1.0f && !(u != u)) // ISPC 1.3.0
                {
                    // Compute and check barycentric v.

                    float<4> woopV = trianglePtr[2];
                    float Oy = woopV.w + origx*woopV.x + origy*woopV.y + origz*woopV.z;
                    float Dy = dirx*woopV.x + diry*woopV.y + dirz*woopV.z;
                    float gamma = Oy + t*Dy;

                    if (gamma >= 0.0f && beta + gamma <= 1.0f) // ISPC 1.2.0
//                            if (gamma >= 0.0f && beta + gamma <= 1.0f && !(gamma != gamma)) // ISPC 1.3.0
                    {
                        // Record intersection.

                        hitTriIndex = triIdx;
                        hitInstanceIndex = instanceIdx;
                        hitT = t;
                        hitU = 1.0f - beta - gamma;
                        hitV = beta;

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
        while ((flag & 1) == 0); // Not the last triangle => keep going.

        // Pop stack.
        if (traversalStack == NULL) return;
        nodeIdx = traversalStack[(*stackPtr)];
        (*stackPtr)--;
    } // leaf
}

inline void fetchRay( const uniform float<4> rays[], uniform int rayFormat, int rayIdx, float<4>& o, float<4>& d, int& mask )
{
  switch( rayFormat )
  {
    case RAY_ORIGIN_TMIN_DIRECTION_TMAX:
    {
      o = rays[rayIdx * 2 + 0];
      d = rays[rayIdx * 2 + 1];
      mask = 0;
      break;
    }
    case RAY_ORIGIN_MASK_DIRECTION_TMAX:
    {
      o = rays[rayIdx * 2 + 0];
      mask = intbits(o.w);
      o.w = 0.0f;
      d = rays[rayIdx * 2 + 1];
      break;
    }
    case RAY_ORIGIN_DIRECTION:
    {
      float* raysPtr = (float*)rays;
      float ox = raysPtr[6*rayIdx + 0];
      float oy = raysPtr[6*rayIdx + 1];              
      float oz = raysPtr[6*rayIdx + 2];
      float tmin = 0.0f;
      float dx = raysPtr[6*rayIdx + 3];
      float dy = raysPtr[6*rayIdx + 4];              
      float dz = raysPtr[6*rayIdx + 5];
      float tmax = 1e34f;

      float<4> o4 = { ox, oy, oz, tmin };
      float<4> d4 = { dx, dy, dz, tmax };
            
      o = o4;
      d = d4;         
      mask = 0;
      break;
    }
  }
}

task void RENAME_FUNCTION(traceTaskUnified) (
    uniform const float<4>   rays[],
    uniform float<4>         hits[],
    uniform const float<4>   nodes[],
    uniform const float<4>   woopTriangles[],
    uniform const int        indices[],
    uniform const float      vertices[],
    uniform const int        indexStride,
    uniform const int        vertexStride,
    uniform const int        remap[],
    uniform int              numRays,
    uniform bool             anyHit,
    uniform bool             watertight,
    uniform int              rayFormat,
    uniform int              hitFormat,
    uniform IntersectionType intersectionType,
    uniform bool             useMasking)
{
    // Traversal stack.

    int traversalStack[STACK_SIZE];
    traversalStack[0] = EntrypointSentinel;

    // Loop over rays.

    uniform int numRaysPerTask = ((numRays - 1) / (taskCount * programCount) + 1) * programCount;
    uniform int rayidxBegin    = taskIndex * numRaysPerTask;
    uniform int rayidxEnd      = min(rayidxBegin + numRaysPerTask, numRays);

    uniform float bboxExpansionFactor = watertight ? 1.0000003576278687f : 1.0f;

    for (uniform int rayidxSimd = rayidxBegin; rayidxSimd < rayidxEnd; rayidxSimd += programCount)
    {
        // Fetch and initialize ray.
  
        int rayIdx = min(rayidxSimd + programIndex, numRays - 1);
        float<4> o, d;        
        int ray_mask = 0;
        fetchRay(rays, rayFormat, rayIdx, o, d, ray_mask);

        RayData rayData;
        RayDataWatertight rayDataWatertight;
        if (intersectionType != INTERSECT_WOOP) {
          if (watertight) {
            setupRayDataWatertight(o, d, rayDataWatertight);
          } else {
            setupRayData(o, d, rayData);
          }
        }

        float origx = o.x;              // Ray origin.
        float origy = o.y;
        float origz = o.z;
        float dirx = d.x;               // Ray direction.
        float diry = d.y;
        float dirz = d.z;
        float tmin = o.w;               // t-value from which the ray starts. Usually 0.

        uniform float ooeps = 8.3e-25f; // 1 / dir
        float idirx = 1.0f / (abs(d.x) > ooeps ? d.x : (d.x >= 0 ? ooeps : -ooeps));
        float idiry = 1.0f / (abs(d.y) > ooeps ? d.y : (d.y >= 0 ? ooeps : -ooeps));
        float idirz = 1.0f / (abs(d.z) > ooeps ? d.z : (d.z >= 0 ? ooeps : -ooeps));
        float oodx = origx * idirx;     // orig / dir
        float oody = origy * idiry;
        float oodz = origz * idirz;

        // Setup traversal.

        int     stackPtr = 0;   // Current position in traversal stack.to pop
        int     nodeIdx = 0;   // Non-negative: current internal node, negative: second postponed leaf. from the root.
        int     hitTriIdx = -1;  // Triangle index of the closest intersection, -1 if none.iangle intersected so far.
        float   hitT     = d.w; // t-value of the closest intersection.
        float   hitU     = 0.0f;
        float   hitV     = 0.0f;

        if (!nodes)
            nodeIdx = EntrypointSentinel;

        // Traversal loop.

        while (nodeIdx != EntrypointSentinel)
        {
            // Traverse internal nodes until all SIMD lanes have found a leaf.

            while (nodeIdx >= 0 && nodeIdx != EntrypointSentinel)
            {
                // Fetch AABBs of the two child nodes.

                const float<4>* nodePtr = &nodes[nodeIdx * 4];
                float<4> n0 = nodePtr[0];
                float<4> n1 = nodePtr[1];
                float<4> n2 = nodePtr[2];
                float<4> n3 = nodePtr[3];

                float<3> c0lo = { n0.x, n0.y, n0.z };
                float<3> c0hi = { n0.w, n1.x, n1.y };
                float<3> c1lo = { n2.x, n2.y, n2.z };
                float<3> c1hi = { n2.w, n3.x, n3.y };
                int<2> cnodes = { intbits(n1.z), intbits(n3.z) };

                // Intersect the ray against the child nodes.

                float c0lox = c0lo.x * idirx - oodx;
                float c0hix = c0hi.x * idirx - oodx;
                float c0loy = c0lo.y * idiry - oody;
                float c0hiy = c0hi.y * idiry - oody;
                float c0loz = c0lo.z * idirz - oodz;
                float c0hiz = c0hi.z * idirz - oodz;
                float c1lox = c1lo.x * idirx - oodx;
                float c1hix = c1hi.x * idirx - oodx;
                float c1loy = c1lo.y * idiry - oody;
                float c1hiy = c1hi.y * idiry - oody;
                float c1loz = c1lo.z * idirz - oodz;
                float c1hiz = c1hi.z * idirz - oodz;

                float c0min = max4(min(c0lox, c0hix), min(c0loy, c0hiy), min(c0loz, c0hiz), tmin);
                float c0max = min4(max(c0lox, c0hix), max(c0loy, c0hiy), max(c0loz, c0hiz), hitT);
                float c1min = max4(min(c1lox, c1hix), min(c1loy, c1hiy), min(c1loz, c1hiz), tmin);
                float c1max = min4(max(c1lox, c1hix), max(c1loy, c1hiy), max(c1loz, c1hiz), hitT);

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
                    nodeIdx = (traverseChild0) ? cnodes.x : cnodes.y;

                    // Both children were intersected => push the farther one.

                    if (traverseChild0 && traverseChild1)
                    {
                        if (c1min < c0min)
                        {
                            int t = nodeIdx;
                            nodeIdx = cnodes.y;
                            cnodes.y = t;
                        }
                        stackPtr++;
                        traversalStack[stackPtr] = cnodes.y;
                    }
                }
            }   // inner

            // Process leaf nodes.

            int dummyHitInstanceIndex = 0;
            switch(intersectionType)
            {
              case INTERSECT_WOOP:
              {
              intersectWoopTriangles(
                0,  //instanceIdx
                nodeIdx, 
                remap, 
                woopTriangles, 
                dirx, diry, dirz, 
                origx, origy, origz, 
                tmin, hitT, 
                hitTriIdx, 
                dummyHitInstanceIndex,
                hitU, hitV, 
                anyHit, traversalStack, &stackPtr);
              }
              break;
              case INTERSECT_INDEXED_LIST:
              {
                intersectIndexedTriangleList(
                  0,  //instanceIdx
                  nodeIdx, 
                  ray_mask,
                  remap, 
                  indices, vertices, indexStride, vertexStride,
                  watertight ? intersectTriangleWatertight : intersectTriangle,
                  watertight ? (void*)&rayDataWatertight : (void*)&rayData,
                  tmin, hitT, 
                  hitTriIdx, 
                  dummyHitInstanceIndex,
                  hitU, hitV, 
                  anyHit, useMasking, traversalStack, &stackPtr);
              }
              break;
              case INTERSECT_VERTEX_LIST:
              {
                intersectTriangleList(
                  0,  // instanceIdx
                  nodeIdx, 
                  remap, 
                  vertices, vertexStride,
                  watertight ? intersectTriangleWatertight : intersectTriangle,
                  watertight ? (void*)&rayDataWatertight : (void*)&rayData,
                  tmin, hitT, 
                  hitTriIdx, 
                  dummyHitInstanceIndex,
                  hitU, hitV, 
                  anyHit, traversalStack, &stackPtr);
              }
              break;
            }
        } // traversal

        // Remap intersected triangle index.

        int instIdx = -1;
        if (hitTriIdx != -1) {
            hitTriIdx = remap[hitTriIdx] & RLLEPACK_INDEX_MASK;
            instIdx = 0;
        }
        else
            hitT = -1.0f;

        // Store the result.
        storeHit( hits, rayIdx, hitFormat, hitTriIdx, instIdx, hitT, hitU, hitV );
    }
}

EXPORT_FUNCTION(traceISPCWoopTriangles)
(
    uniform const float<4>   rays[],
    uniform float<4>         hits[],
    uniform const float<4>   nodes[],
    uniform const float<4>   woopTriangles[],
    uniform const int        remap[],
    uniform int              numRays,
    uniform bool             anyHit,
    uniform int              rayFormat,
    uniform int              hitFormat)
{
    launch[128] RENAME_FUNCTION(traceTaskUnified)(rays, hits, nodes, woopTriangles, 0, 0, 0, 0, remap, numRays, anyHit, /*watertight*/ false, rayFormat, hitFormat, INTERSECT_WOOP, 0);
}

//------------------------------------------------------------------------

EXPORT_FUNCTION(traceISPCIndexedTriangleList)
(
    uniform const float<4>   rays[],
    uniform float<4>         hits[],
    uniform const float<4>   nodes[],
    uniform const int        indices[],
    uniform const float      vertices[],
    uniform const int        indexStride,
    uniform const int        vertexStride,
    uniform const int        remap[],
    uniform int              numRays,
    uniform bool             anyHit,
    uniform bool             watertight,
    uniform int              rayFormat,
    uniform int              hitFormat,
    uniform bool             useMasking)
{
    launch[128] RENAME_FUNCTION(traceTaskUnified)(rays, hits, nodes, /*woopTriangles*/ NULL, indices, vertices, indexStride, vertexStride, remap, numRays, anyHit, watertight, rayFormat, hitFormat, INTERSECT_INDEXED_LIST, useMasking);
}

//------------------------------------------------------------------------

EXPORT_FUNCTION(traceISPCTriangleList)
(
    uniform const float<4>   rays[],
    uniform float<4>         hits[],
    uniform const float<4>   nodes[],
    uniform const float      vertices[],
    uniform const int        vertexStride,
    uniform const int        remap[],
    uniform int              numRays,
    uniform bool             anyHit,
    uniform bool             watertight,
    uniform int              rayFormat,
    uniform int              hitFormat)
{
    launch[128] RENAME_FUNCTION(traceTaskUnified)(rays, hits, nodes, /*woopTriangles*/ NULL, 0, vertices, 0, vertexStride, remap, numRays, anyHit, watertight, rayFormat, hitFormat, INTERSECT_VERTEX_LIST, 0);
}

inline float* getTransformPtr( const float* basePtr, int which, int stride )
{
  int* m = &((int*)basePtr)[which * stride];
  return (float*)m;
}

inline void transformRay( float<4>& orig, float<4>& dir, float<4>* im )
{
  float<4> o, d;

  float<4> m0 = im[0];
  o.x = m0.x * orig.x + m0.y * orig.y + m0.z * orig.z + m0.w;
  d.x = m0.x * dir.x  + m0.y * dir.y  + m0.z * dir.z;

  float<4> m1 = im[1];
  o.y = m1.x * orig.x + m1.y * orig.y + m1.z * orig.z + m1.w;
  d.y = m1.x * dir.x  + m1.y * dir.y  + m1.z * dir.z;    

  float<4> m2 = im[2];    
  o.z = m2.x * orig.x + m2.y * orig.y + m2.z * orig.z + m2.w;
  d.z = m2.x * dir.x  + m2.y * dir.y  + m2.z * dir.z;

  o.w = 0.0f;
  d.w = 0.0f;

  orig = o;
  dir  = d;
}

task void RENAME_FUNCTION(traceTask2LevelUnified) ( uniform const tracerIspcParams* uniform params )
{
  // Traversal stack.

  int traversalStack[STACK_SIZE];
  traversalStack[0] = EntrypointSentinel;

  // Loop over rays.

  uniform int numRaysPerTask = ((params->numRays - 1) / (taskCount * programCount) + 1) * programCount;
  uniform int rayidxBegin    = taskIndex * numRaysPerTask;
  uniform int rayidxEnd      = min(rayidxBegin + numRaysPerTask, params->numRays);

  uniform float bboxExpansionFactor = params->watertight ? 1.0000003576278687f : 1.0f;

  for (uniform int rayidxSimd = rayidxBegin; rayidxSimd < rayidxEnd; rayidxSimd += programCount)
  {
    // Fetch and initialize ray.
  
    int rayIdx = min(rayidxSimd + programIndex, params->numRays - 1);
    float<4> o, d;        
    int ray_mask = 0;
    fetchRay(params->rays, params->rayFormat, rayIdx, o, d, ray_mask);

    RayData rayData;
    RayDataWatertight rayDataWatertight;
    if (params->isecMode != INTERSECT_WOOP) {
      if (params->watertight) {
        setupRayDataWatertight(o, d, rayDataWatertight);
      } else {
        setupRayData(o, d, rayData);
      }
    }

    float origx = o.x;              // Ray origin.
    float origy = o.y;
    float origz = o.z;
    float dirx = d.x;               // Ray direction.
    float diry = d.y;
    float dirz = d.z;
    float tmin = o.w;               // t-value from which the ray starts. Usually 0.

    uniform float ooeps = 8.3e-25f; // 1 / dir
    float idirx = 1.0f / (abs(d.x) > ooeps ? d.x : (d.x >= 0 ? ooeps : -ooeps));
    float idiry = 1.0f / (abs(d.y) > ooeps ? d.y : (d.y >= 0 ? ooeps : -ooeps));
    float idirz = 1.0f / (abs(d.z) > ooeps ? d.z : (d.z >= 0 ? ooeps : -ooeps));
    float oodx = origx * idirx;     // orig / dir
    float oody = origy * idiry;
    float oodz = origz * idirz;

    // Setup traversal.

    int     state        = 0;
    int     stackPtr     = 0;   // Current position in traversal stack.to pop
    int     nodeIdx      = 0;   // Non-negative: current internal node, negative: second postponed leaf. from the root.
    int     hitTriIdx    = -1;  // Triangle index of the closest intersection, -1 if none.iangle intersected so far.
    int     hitInstIdx   = -1;  // Instance index of the closest intersection, -1 if none
    float   hitT         = d.w; // t-value of the closest intersection.
    float   hitU         = 0.0f;
    float   hitV         = 0.0f;

    int instIdx = -1;

    uniform float<4>* nodes = (uniform float<4>*)params->group[0].nodes;

    if( !nodes || params->group[0].numEntities==0 )
      nodeIdx = EntrypointSentinel;
    
    // outer traversal loop
    do
    {
      // traverse internal nodes until found a leaf
      while (nodeIdx >= 0 && nodeIdx != EntrypointSentinel)
      {
        // Fetch AABBs of the two child nodes.

        const float<4>* nodePtr = &nodes[nodeIdx * 4];
        float<4> n0 = nodePtr[0];
        float<4> n1 = nodePtr[1];
        float<4> n2 = nodePtr[2];
        float<4> n3 = nodePtr[3];

        float<3> c0lo = { n0.x, n0.y, n0.z };
        float<3> c0hi = { n0.w, n1.x, n1.y };
        float<3> c1lo = { n2.x, n2.y, n2.z };
        float<3> c1hi = { n2.w, n3.x, n3.y };
        int<2> cnodes = { intbits(n1.z), intbits(n3.z) };

        // Intersect the ray against the child nodes.

        float c0lox = c0lo.x * idirx - oodx;
        float c0hix = c0hi.x * idirx - oodx;
        float c0loy = c0lo.y * idiry - oody;
        float c0hiy = c0hi.y * idiry - oody;
        float c0loz = c0lo.z * idirz - oodz;
        float c0hiz = c0hi.z * idirz - oodz;
        float c1lox = c1lo.x * idirx - oodx;
        float c1hix = c1hi.x * idirx - oodx;
        float c1loy = c1lo.y * idiry - oody;
        float c1hiy = c1hi.y * idiry - oody;
        float c1loz = c1lo.z * idirz - oodz;
        float c1hiz = c1hi.z * idirz - oodz;

        float c0min = max4(min(c0lox, c0hix), min(c0loy, c0hiy), min(c0loz, c0hiz), tmin);
        float c0max = min4(max(c0lox, c0hix), max(c0loy, c0hiy), max(c0loz, c0hiz), hitT);
        float c1min = max4(min(c1lox, c1hix), min(c1loy, c1hiy), min(c1loz, c1hiz), tmin);
        float c1max = min4(max(c1lox, c1hix), max(c1loy, c1hiy), max(c1loz, c1hiz), hitT);

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
            nodeIdx = (traverseChild0) ? cnodes.x : cnodes.y;

            // Both children were intersected => push the farther one.

            if (traverseChild0 && traverseChild1)
            {
                if (c1min < c0min)
                {
                    int t = nodeIdx;
                    nodeIdx = cnodes.y;
                    cnodes.y = t;
                }
                stackPtr++;
                traversalStack[stackPtr] = cnodes.y;
            }
        }
      } // end node traversal loop

      if(state==0 && nodeIdx<0) // transform ray and go back to traverse
      {
        int instAddr = ~(nodeIdx);

        instIdx = params->group[0].remap[instAddr] & RLLEPACK_INDEX_MASK;

        int modelId = params->group[0].modelId[instIdx];

        const float* ilwTransform = getTransformPtr( params->group[0].ilwMatrices, instIdx, params->group[0].matrixStride );      

        // transform ray
        transformRay( o, d, (float<4>*)ilwTransform );
        origx = o.x;
        origy = o.y;
        origz = o.z;

        dirx = d.x;
        diry = d.y;
        dirz = d.z;
        
        if (params->isecMode != INTERSECT_WOOP) {
          if (params->watertight) {
            setupRayDataWatertight(o, d, rayDataWatertight);
          } else {
            setupRayData(o, d, rayData);
          }
        }

        ooeps = 1e-20f;
        idirx = 1.0f / (abs(dirx) > ooeps ? dirx : (dirx>=0 ? ooeps : -ooeps));
        idiry = 1.0f / (abs(diry) > ooeps ? diry : (diry>=0 ? ooeps : -ooeps));
        idirz = 1.0f / (abs(dirz) > ooeps ? dirz : (dirz>=0 ? ooeps : -ooeps));
        oodx = origx * idirx;
        oody = origy * idiry;
        oodz = origz * idirz;

        stackPtr++;
        traversalStack[stackPtr] = EntrypointSentinel;

        nodeIdx      = 0;
        nodes         = (uniform float<4>*)params->meshes[modelId].nodes;

        state++;
      
        if( params->meshes[modelId].numEntities==0 )
          nodeIdx = EntrypointSentinel;      
      }
      else if(state==1 && nodeIdx<0) // intersect triangles
      {
        int modelId = params->group[0].modelId[instIdx];
        const uniform TracerParamsMesh* varying mesh = &params->meshes[modelId];

        switch(params->isecMode)
        {
          case INTERSECT_WOOP:
          intersectWoopTriangles(instIdx, nodeIdx, mesh->remap, mesh->triangles, dirx, diry, dirz, origx, origy, origz, tmin, hitT, hitTriIdx, hitInstIdx, hitU, hitV, params->anyhit, /*traversalStack*/ NULL, NULL);
          break;
          case INTERSECT_INDEXED_LIST:
          {
            intersectIndexedTriangleList(instIdx, nodeIdx, ray_mask, 
              mesh->remap, mesh->indices, mesh->vertices, mesh->indexStride, mesh->vertexStride,
              params->watertight ? intersectTriangleWatertight : intersectTriangle, 
              params->watertight ? (void*)&rayDataWatertight : (void*)&rayData, 
              tmin, hitT, hitTriIdx, hitInstIdx, hitU, hitV, params->anyhit, params->useMasking, /*traversalStack*/ NULL, NULL);
            
          }
          break;
          case INTERSECT_VERTEX_LIST:
            intersectTriangleList(instIdx, nodeIdx, 
              mesh->remap, mesh->vertices, mesh->vertexStride, 
              params->watertight ? intersectTriangleWatertight : intersectTriangle, 
              params->watertight ? (void*)&rayDataWatertight : (void*)&rayData, 
              tmin, hitT, hitTriIdx, hitInstIdx, hitU, hitV, params->anyhit, NULL, NULL);
          
          break;
        }

        nodeIdx = traversalStack[stackPtr];
        stackPtr--;
      }
      else if(state == 1 && nodeIdx == EntrypointSentinel) // after isec, proceed with top level traversal
      {
        state      = 0;

        nodeIdx    = traversalStack[stackPtr];
        stackPtr--;

        if(nodeIdx!=EntrypointSentinel)
        {
          fetchRay( (float<4>*)params->rays, params->rayFormat, rayIdx, o, d, ray_mask );

          origx = o.x;
          origy = o.y;
          origz = o.z;

          dirx = d.x;
          diry = d.y;
          dirz = d.z;

          if (params->isecMode != INTERSECT_WOOP) {
            if (params->watertight) {
              setupRayDataWatertight(o, d, rayDataWatertight);
            } else {
              setupRayData(o, d, rayData);
            }
          }

          ooeps = 1e-20f;
          idirx = 1.0f / (abs(dirx) > ooeps ? dirx : (dirx>=0 ? ooeps : -ooeps));
          idiry = 1.0f / (abs(diry) > ooeps ? diry : (diry>=0 ? ooeps : -ooeps));
          idirz = 1.0f / (abs(dirz) > ooeps ? dirz : (dirz>=0 ? ooeps : -ooeps));
          oodx  = origx * idirx;
          oody  = origy * idiry;
          oodz  = origz * idirz;

          nodes = (uniform float<4>*)params->group[0].nodes;              
        }
      }
    }
    while(!((nodeIdx == EntrypointSentinel) && (state == 0))); // one ray terminated 

    if( hitTriIdx != -1 && hitInstIdx != -1 ) {
      int modelId = params->group[0].modelId[hitInstIdx];
      const int* remap  = params->meshes[modelId].remap;
      hitTriIdx = remap[hitTriIdx] & RLLEPACK_INDEX_MASK;
    } else {
      hitT = -1.0f;
    }

    // Store the result.
    storeHit( (void*)params->hits, rayIdx, params->hitFormat, hitTriIdx, hitInstIdx, hitT, hitU, hitV );
  } // end ray loop
}

EXPORT_FUNCTION(traceISPC2LevelBvh)
( uniform const tracerIspcParams* uniform params )
{
    launch[128] RENAME_FUNCTION(traceTask2LevelUnified)(params);
}

//------------------------------------------------------------------------

// Only need one of these
#ifndef ISPC_64_BIT_ADDRESSING
export uniform int getTargetISPC()
{
#if defined(ISPC_TARGET_SSE2)
    return 0;
#elif defined(ISPC_TARGET_SSE4)
    return 1;
#elif defined(ISPC_TARGET_AVX)
    return 2;
#elif defined(ISPC_TARGET_AVX2)
    return 3;
#else
    return -1;
#endif
}
#endif

//------------------------------------------------------------------------
