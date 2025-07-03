#include "WatertightOptimized.lwh"

struct Aabb
{
  float3 lo;
  float3 hi;
};

struct Node
{
  Aabb bbox;
  int  addr;
};

//------------------------------------------------------------------------------

static __device__ __forceinline__ float3 operator*( const float3& a, const float3& b )
{
  return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
}

//------------------------------------------------------------------------

static __device__ __forceinline__ float3 operator-( const float3& a, const float3& b )
{
  return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

//------------------------------------------------------------------------

template<typename T> static __device__ __forceinline__
void swap( T& a, T& b )
{
  const T tmp = a;
  a = b;
  b = tmp;
}

//------------------------------------------------------------------------

static __device__ __forceinline__
int pop( char*& stackPtr )
{
  const int val = *(int*)stackPtr;
  stackPtr -= 4;
  return val;
}

//------------------------------------------------------------------------

static __device__ __forceinline__
void push( char*& stackPtr, const int val )
{
  stackPtr += 4;
  *(int*)stackPtr = val;
}

//------------------------------------------------------------------------

static __device__ __forceinline__ 
void fetchRay( const float4* const __restrict rays, const int rayIdx, float3& orig, float3& dir, float& tmin, float& tmax )
{
  const float4 o = rays[rayIdx * 2 + 0];
  const float4 d = rays[rayIdx * 2 + 1];
  orig = make_float3( o.x, o.y, o.z );
  tmin = o.w;
  dir  = make_float3( d.x, d.y, d.z );
  tmax = d.w;
}

//------------------------------------------------------------------------

static __device__ __forceinline__ 
void fetchRay( const float3* const __restrict rays, const int rayIdx, float3& orig, float3& dir )
{
  loadUncachedFloat6( rays + 2*rayIdx, orig.x, orig.y, orig.z,  dir.x, dir.y, dir.z );
}

//------------------------------------------------------------------------------

static __device__ __forceinline__ 
void fetchRay( const void* const __restrict rays, const int rayIdx, const int rayFormat, float3& orig, float3& dir, float& tmin, float& tmax, int& mask )
{
  if( rayFormat == RAY_ORIGIN_TMIN_DIRECTION_TMAX )
  {
    fetchRay((float4*)rays, rayIdx, orig, dir, tmin, tmax);
    mask = 0;
  }
  else if( rayFormat == RAY_ORIGIN_MASK_DIRECTION_TMAX )
  {
    fetchRay((float4*)rays, rayIdx, orig, dir, tmin, tmax);
    mask = __float_as_int(tmin);
    tmin = 0.0f;
  }
  else
  {
    fetchRay((float3*)rays, rayIdx, orig, dir );
    tmin = 0.0f;
    tmax = 1e34f;
    mask = 0;
  }        
}

static __device__ __forceinline__ 
void fetchRay( const void* const __restrict rays, const int rayIdx, const int rayFormat, float3& orig, float3& dir, float& tmin, float& tmax )
{
  int tmpMask;
  fetchRay(rays, rayIdx, rayFormat, orig, dir, tmin, tmax, tmpMask);
}


//------------------------------------------------------------------------------
static __device__ __forceinline__ 
void* fetchRayPtr( const void* const __restrict rays, const int rayIdx, const int rayFormat )
{
  const float* const __restrict raysPtr = (float*)rays;
  if( rayFormat == RAY_ORIGIN_TMIN_DIRECTION_TMAX )
  {
    return (void*)(raysPtr + 8*rayIdx);
  }
  else if( rayFormat == RAY_ORIGIN_MASK_DIRECTION_TMAX )
  {
    return (void*)(raysPtr + 8*rayIdx);
  }
  else
  {
    return (void*)(raysPtr + 6*rayIdx);
  }
}

//------------------------------------------------------------------------------

static __device__ __forceinline__ 
void fetchRay( const void* const __restrict rays, const int rayIdx, const int rayFormat, float3& orig, float3& dir )
{
  float tmin, tmax;
  if( rayFormat == RAY_ORIGIN_TMIN_DIRECTION_TMAX )
    fetchRay((float4*)rays, rayIdx, orig, dir, tmin, tmax);
  else if( rayFormat == RAY_ORIGIN_MASK_DIRECTION_TMAX )
    fetchRay((float4*)rays, rayIdx, orig, dir, tmin, tmax);
  else
    fetchRay((float3*)rays, rayIdx, orig, dir );
}

//------------------------------------------------------------------------

static __device__ __forceinline__ 
float3 getRcpDir( const float3& dir )
{
  const float ooeps = exp2f(-80.0f); // Avoid div by zero.
  float3 rcpDir;
  rcpDir.x = 1.0f / bitSelect(dir.x, fmaxf(ooeps, fabsf(dir.x)), __uint_as_float(0x80000000));
  rcpDir.y = 1.0f / bitSelect(dir.y, fmaxf(ooeps, fabsf(dir.y)), __uint_as_float(0x80000000));
  rcpDir.z = 1.0f / bitSelect(dir.z, fmaxf(ooeps, fabsf(dir.z)), __uint_as_float(0x80000000));
  return rcpDir;
}

static __device__ __forceinline__ 
void setupRay( const float3& orig, const float3& dir, float3& rcpDir, float3& origOverDir)
{
  rcpDir = getRcpDir(dir);
  origOverDir = orig * rcpDir;
}

//------------------------------------------------------------------------

#ifdef INCLUDE_AABB_TESTS

static __device__ __forceinline__ 
bool intersectAabb( const Aabb& bbox, const float3& rcpDir, const float3& origOverDir, const float tmin, const float tmax, float& out_t )
{
  const float3 tlo = bbox.lo * rcpDir - origOverDir;
  const float3 thi = bbox.hi * rcpDir - origOverDir;
  const float tBegin = spanBegin(tlo.x, thi.x, tlo.y, thi.y, tlo.z, thi.z, tmin);
  const float tEnd   = spanEnd  (tlo.x, thi.x, tlo.y, thi.y, tlo.z, thi.z, tmax);
  out_t = tBegin;
  return tBegin <= tEnd;
}

#endif // INCLUDE_AABB_TESTS

//------------------------------------------------------------------------

static __device__ __forceinline__ float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

static __device__ __forceinline__ float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ __forceinline__ float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}

//------------------------------------------------------------------------

static __device__ __forceinline__
void storeHit( void* const __restrict hits, const int rayIdx, const int hitFormat, const int triIndex, const int instanceIndex, const float t, const float u, const float v )
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
      if( t >= 0.0f ) // TODO: try triIndex. A previous comment said this was faster
      {
        const unsigned offset = rayIdx / 32;
        const unsigned bit    = 1u << (rayIdx % 32);
        atomicOr( (unsigned*)hits + offset, bit );
      }
      break;
    }
  }
}

//

template<typename T>
struct LoadGlobal
{
  __device__ __forceinline__
  static T load(const T* const __restrict ptr, const int offset)
  {
    return ptr[offset];
  }
};

template<typename T>
struct LoadLdg // LDG on SMs that support it, global otherwise
{
  __device__ __forceinline__
  static T load(const T* const __restrict ptr, const int offset)
  {
    return LDG_OR_GLOBAL( &ptr[offset] );
  }
};

#ifdef INCLUDE_TEXTURES

struct LoadTexNodes
{
  __device__ __forceinline__
  static float4 load(const float4* const __restrict ptr, const int offset)
  { 
    // load from t_nodes 
    return tex1Dfetch( t_nodes, offset );
  }
};

struct LoadTexRemap
{
  __device__ __forceinline__
  static int load(const int* const __restrict ptr, const int offset)
  { 
    // load from t_remap 
    return tex1Dfetch( t_remap, offset );
  }
};

struct LoadTexWoopTris
{
  __device__ __forceinline__
  static float4 load(const float4* const __restrict ptr, const int offset)
  {
    return tex1Dfetch( t_triangles, offset );
  }
};

#endif // INCLUDE_TEXTURES

template<typename C> 
static __device__ __forceinline__
bool woopTriangleIntersect( const float4* const __restrict ptr, const int triIdx, const float3& orig, const float3& dir, const float tmin, const float tmax, float& out_t, bool& endOfList )
{
        float4 woopT = C::load(ptr, triIdx * 3 + 0);
  const float4 woopU = C::load(ptr, triIdx * 3 + 1);
  const float4 woopV = C::load(ptr, triIdx * 3 + 2);

  endOfList = ((__float_as_int(woopT.w) & 1) != 0);

  // Prevent the flag bit from affecting callwlations below.
  woopT.w = clearLSB(woopT.w);

  // Compute and check intersection t-value.

  const float Oz = woopT.w - orig.x * woopT.x - orig.y * woopT.y - orig.z * woopT.z;
  const float ilwDz = 1.0f / (dir.x * woopT.x + dir.y * woopT.y + dir.z * woopT.z);
  const float t = Oz * ilwDz;

  bool hit = false;
  if (t > tmin && t < tmax)
  {
    // Compute hit point.

    const float Hx = orig.x + t * dir.x;
    const float Hy = orig.y + t * dir.y;
    const float Hz = orig.z + t * dir.z;

    // Compute barycentrics.

    const float u = woopU.w + Hx * woopU.x + Hy * woopU.y + Hz * woopU.z;
    const float v = woopV.w + Hx * woopV.x + Hy * woopV.y + Hz * woopV.z;

    // Intersection => record.
    // Closest intersection not required => terminate.

    if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f)
      hit = true;    
  }

  out_t = t;
  
  return hit;
}

template<typename WoopLoaderT, typename RemapLoaderT>
struct WoopTriangles
{

  struct RayType
  {
    float3  orig;
    float3  dir; 
    float   tmin;
    float   tmax;

    // Precomputed data for intersection
    float3 rcpDir;
    float3 origOverDir;
  };

  __device__ __forceinline__
  static void fetchRay(const void* const __restrict rays, const int rayIdx, const int rayFormat, RayType& ray)
  {
    ::fetchRay(rays, rayIdx, rayFormat, ray.orig, ray.dir, ray.tmin, ray.tmax);
    ::setupRay(ray.orig, ray.dir, ray.rcpDir, ray.origOverDir);
  }

#ifdef INCLUDE_AABB_TESTS

  __device__ __forceinline__
  static bool intersectAabb( RayType& ray, const Aabb& bbox, const float tmin, const float tmax, float& out_t )
  {
    return ::intersectAabb(bbox, ray.rcpDir, ray.origOverDir, tmin, tmax, out_t);
  }

#endif // INCLUDE_AABB_TESTS

  __device__ __forceinline__
  static void intersectTriangles(const RayType& ray, const TracerParamsMesh& p, const int instIdx, int& leafIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, int& nodeIdx, char** stackPtr, const bool anyhit) 
  {
    while(leafIdx < 0)
    {
        // Intersect the ray against each triangle
        int triIdx = ~leafIdx;

        bool endOfList;
        do
        {
            float t;
            if( woopTriangleIntersect<WoopLoaderT>(p.triangles, triIdx, ray.orig, ray.dir, ray.tmin, hitT, t, endOfList) )
            {
              hitT = t;
              hitTriIdx = triIdx;
              hitInstIdx = instIdx;
              if(anyhit)
              {
                nodeIdx = EntrypointSentinel;
                break;
              }
            }
            triIdx++;
        }
        while (!endOfList); // Not the last triangle => keep going.

        leafIdx = nodeIdx;
        if (nodeIdx < 0)
            nodeIdx = pop( *stackPtr );
    } // leaf  
  }

  __device__ __forceinline__
  static bool intersectTriangles52(const RayType& ray, const TracerParamsMesh& p, const int instIdx, int triIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, const bool anyhit)
  {
    // Intersect the ray against each triangle

    bool foundHit = false;
    bool endOfList;
    do
    {
        float t;
        if (woopTriangleIntersect<WoopLoaderT>(p.triangles, triIdx, ray.orig, ray.dir, ray.tmin, hitT, t, endOfList))
        {
            hitT = t;
            hitTriIdx = triIdx;
            hitInstIdx = instIdx;
            foundHit = true;

            if (anyhit)
                break;
        }
        triIdx++;
    } while (!endOfList); // Not the last triangle => keep going.
    return foundHit;
  }

  __device__ __forceinline__
  static void computeUV(const TracerParamsMesh& p, const int triIndex, const float3& orig, const float3& dir, const float t, float& u, float& v)
  {
    const float4 woopU = WoopLoaderT::load( p.triangles, triIndex*3+1 );
    const float4 woopV = WoopLoaderT::load( p.triangles, triIndex*3+2 );
    
    const float Ox = woopU.w + orig.x*woopU.x + orig.y*woopU.y + orig.z*woopU.z;
    const float Dx =            dir.x*woopU.x +  dir.y*woopU.y +  dir.z*woopU.z;
    const float beta = Ox + t*Dx;

    const float Oy = woopV.w + orig.x*woopV.x + orig.y*woopV.y + orig.z*woopV.z;
    const float Dy =            dir.x*woopV.x +  dir.y*woopV.y +  dir.z*woopV.z;
    const float gamma = Oy + t*Dy;

    u = 1.0f - beta - gamma;
    v = beta;
  }
  
  __device__ __forceinline__
  static int remap( const int* const __restrict ptr, const int i )
  { 
    return RemapLoaderT::load( ptr, i );
  }
};

// woop intersection test for kernels that do (n)ot use (s)pelwlative (t)raversal
template<typename WoopLoaderT, typename RemapLoaderT>
struct WoopTrianglesNST : public WoopTriangles<WoopLoaderT, RemapLoaderT>
{

  typedef WoopTriangles<WoopLoaderT, RemapLoaderT> BaseClassType;
  typedef typename BaseClassType::RayType RayType;

  __device__ __forceinline__
  static void fetchAndTransformRay(const void* const __restrict rays, const int rayIdx, const int rayFormat, RayType& ray, const float4* const __restrict im)
  {
    ::fetchRay(rays, rayIdx, rayFormat, ray.orig, ray.dir, ray.tmin, ray.tmax);
    ::transformRay(ray.orig, ray.dir, im);
    ::setupRay(ray.orig, ray.dir, ray.rcpDir, ray.origOverDir);
  }

  __device__ __forceinline__
  static void transformRay(RayType& ray, const float4* const __restrict im)
  {
    ::transformRay(ray.orig, ray.dir, im);
    ::setupRay(ray.orig, ray.dir, ray.rcpDir, ray.origOverDir);
  }

  __device__ __forceinline__
  static void intersectTriangles(const RayType& ray, const TracerParamsMesh& p, const int instIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, int& nodeIdx, const bool anyhit) 
  {
    // Intersect the ray against each triangle
    int triIdx = ~nodeIdx;

    bool endOfList;
    do
    {
        float t;
        if( woopTriangleIntersect<WoopLoaderT>(p.triangles, triIdx, ray.orig, ray.dir, ray.tmin, hitT, t, endOfList) )
        {
          hitT = t;
          hitTriIdx = triIdx;
          hitInstIdx = instIdx;
          if(anyhit)
          {
            nodeIdx = EntrypointSentinel;
            break;
          }
        }
        triIdx++;
    }
    while (!endOfList); // Not the last triangle => keep going.
  }

};

template<typename T>
static __device__ __forceinline__ 
T makeTriIndices(const int baseIndex)
{
  T t;
  return t;
}

template<>
__device__ __forceinline__ 
int3 makeTriIndices(const int baseIndex)
{
  const int3 i3 = make_int3(baseIndex+0, baseIndex+1, baseIndex+2);
  return i3;
}

template<>
__device__ __forceinline__ 
int4 makeTriIndices(const int baseIndex)
{
  const int4 i4 = make_int4(baseIndex+0, baseIndex+1, baseIndex+2, 0);
  return i4;
}

static __device__ __forceinline__ 
int getMask(const int3& i)
{
  return 0;
}

static __device__ __forceinline__ 
int getMask(const int4& i)
{
  return i.w;
}

struct TriIndexer
{
  template<typename IndexType>
  struct Indexed
  {
    __device__ __forceinline__
    static IndexType getIndices(const IndexType* const __restrict indices, const int triIndex, const int indexStride)
    { 
      return *(IndexType*)((char*)indices + ((size_t)triIndex * indexStride));
    }
  };

  template<typename IndexType>
  struct List
  {
    __device__ __forceinline__    
    static IndexType getIndices(const IndexType* const __restrict indices, const int triIndex, const int indexStride)
    {
      const int baseIndex = triIndex*(int)(sizeof(IndexType)/4);
      return makeTriIndices<IndexType>( baseIndex );
    }
  };

  template<typename IndexType>
  struct Branch
  {
    __device__ __forceinline__    
    static IndexType getIndices(const IndexType* const __restrict indices, const int triIndex, const int indexStride)
    {
      return ( indices ) ? Indexed<IndexType>::getIndices(indices, triIndex, indexStride)
                         :    List<IndexType>::getIndices(indices, triIndex, indexStride);
    }
  };
};

//__device__ __inline__ float3 make_float3( const float3& x ) { return x; } //BL: Is this needed?

template<typename TriIndexerT, typename VertexType, typename IndexType, bool USE_MASKING> 
static __device__ __forceinline__
bool triangleIntersect(const TracerParamsMesh& p, const int triIdx, const float3& orig, const float3& dir, const float tmin, const float tmax, const int ray_mask, float& out_t )
{
    const IndexType vi = TriIndexerT::getIndices((IndexType*)p.indices, triIdx, p.indexStride);
    if( USE_MASKING )
    {
      const int tri_mask = getMask(vi);

      if( (tri_mask&ray_mask)!=0 ) {
        return false;
      }
    }

    const VertexType* const __restrict vertices = reinterpret_cast<const VertexType*>( p.vertices );

    const float3 p0 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.x * p.vertexStride) );
    const float3 p1 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.y * p.vertexStride) );
    const float3 p2 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.z * p.vertexStride) );

    const float3 e0 = p1 - p0;
    const float3 e1 = p0 - p2;
    const float3 n  = cross( e1, e0 );

    const float3 e2 = ( 1.0f / dot( n, dir ) ) * ( p0 - orig );
    const float3 i  = cross( dir, e2 );

    const float beta  = dot( i, e1 );
    const float gamma = dot( i, e0 );
    const float t     = dot( n, e2 );

    out_t = t;

    return ( (t < tmax) & (t > tmin) & (beta >= 0.0f) & (gamma >= 0.0f) & (beta + gamma <= 1.0f) );
}

template<typename TriIndexerT, typename VertexType, typename IndexType, bool USE_MASKING>
struct MeshTriangles
{

  struct RayType
  {
    float3  orig;
    float3  dir; 
    float   tmin;
    float   tmax;
    int     mask;

    // Precomputed data for intersection
    float3 rcpDir;
    float3 origOverDir;
  };

  __device__ __forceinline__
  static void fetchRay(const void* const __restrict rays, const int rayIdx, const int rayFormat, RayType& ray)
  {
    ::fetchRay(rays, rayIdx, rayFormat, ray.orig, ray.dir, ray.tmin, ray.tmax, ray.mask);
    ::setupRay(ray.orig, ray.dir, ray.rcpDir, ray.origOverDir);
  }

#ifdef INCLUDE_AABB_TESTS

  __device__ __forceinline__
  static bool intersectAabb( const RayType& ray, const Aabb& bbox, const float tmin, const float tmax, float& out_t )
  {
    return ::intersectAabb(bbox, ray.rcpDir, ray.origOverDir, tmin, tmax, out_t);
  }

#endif // INCLUDE_AABB_TESTS

  __device__ __forceinline__
  static void intersectTriangles(const RayType &ray, const TracerParamsMesh& p, const int instIdx, int& leafIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, int& nodeIdx, char** stackPtr, const bool anyhit)
  {
      while (leafIdx < 0)
      {
          // Intersect the ray against each triangle
          int triIdx = ~leafIdx;
          const int lastPrim = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx]) >> RLLEPACK_LEN_SHIFT) + triIdx;

          do
          {
              const int remap = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx])) & RLLEPACK_INDEX_MASK;

              float t;
              if (triangleIntersect<TriIndexerT, VertexType, IndexType, USE_MASKING>(p, remap, ray.orig, ray.dir, ray.tmin, hitT, ray.mask, t))
              {
                  hitT = t;
                  hitTriIdx = triIdx;
                  hitInstIdx = instIdx;
                  if (anyhit)
                  {
                      nodeIdx = EntrypointSentinel;
                      break;
                  }
              }
              triIdx++;
          } while (triIdx < lastPrim); // Not the last triangle => keep going.

          leafIdx = nodeIdx;
          if (nodeIdx < 0)
              nodeIdx = pop(*stackPtr);
      } // leaf
  }

  __device__ __forceinline__
  static bool intersectTriangles52(const RayType& ray, const TracerParamsMesh& p, const int instIdx, int triIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, const bool anyhit)
  {
    bool foundHit = false;
    const int lastPrim = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx]) >> RLLEPACK_LEN_SHIFT) + triIdx;

    do
    {
        const int remap = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx])) & RLLEPACK_INDEX_MASK;

        float t;
        if (triangleIntersect<TriIndexerT, VertexType, IndexType, USE_MASKING>(p, remap, ray.orig, ray.dir, ray.tmin, hitT, ray.mask, t))
        {
            hitT = t;
            hitTriIdx = triIdx;
            hitInstIdx = instIdx;
            foundHit = true;

            if (anyhit)
                break;
        }
        triIdx++;
    } while (triIdx < lastPrim); // Not the last triangle => keep going.
    return foundHit;
  }

  __device__ __forceinline__
  static void computeUV(const TracerParamsMesh& p, const int triIndex, float3& orig, float3& dir, const float t, float& u, float& v)
  {
    const int  remapped = remap( p.remap, triIndex );
    const IndexType vi  = TriIndexerT::getIndices((IndexType*)p.indices, remapped, p.indexStride);

    const VertexType* const __restrict vertices = reinterpret_cast<const VertexType*>( p.vertices );

    const float3 p0 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.x * p.vertexStride) );
    const float3 p1 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.y * p.vertexStride) );
    const float3 p2 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.z * p.vertexStride) );

    const float3 e0 = p1 - p0;
    const float3 e1 = p0 - p2;
    const float3 n  = cross( e1, e0 );

    const float3 e2 = ( 1.0f / dot( n, dir ) ) * ( p0 - orig );
    const float3 i  = cross( dir, e2 );

    const float beta  = dot( i, e1 );
    const float gamma = dot( i, e0 );

    u = 1.0f - beta - gamma;
    v = beta;
  }

  __device__ __forceinline__
  static int remap( const int* const __restrict ptr, const int i )
  {
    return LoadLdg<int>::load(ptr, i) & RLLEPACK_INDEX_MASK;
  }
};


template<typename TriIndexerT, typename VertexType, typename IndexType, bool USE_MASKING, bool USE_WATERTIGHT> 
__device__ __forceinline__
bool triangleIntersectWatertight(const TracerParamsMesh& p, const int triIdx,
  const float3& ray_orig, const bool axisx, const bool axisy, const bool axisz, const float3& shear, const float tmin, const float tmax, const int ray_mask, float& out_t )
{
    const IndexType vi = TriIndexerT::getIndices((IndexType*)p.indices, triIdx, p.indexStride);
    if( USE_MASKING )
    {
        const int tri_mask = getMask(vi);

      if( (tri_mask & ray_mask)!=0 ) {
        return false;
      }
    }

    const VertexType* const __restrict vertices = reinterpret_cast<const VertexType*>( p.vertices );

    const float3 p0 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.x * p.vertexStride) );
    const float3 p1 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.y * p.vertexStride) );
    const float3 p2 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.z * p.vertexStride) );

    float2 uv;
    return WatertightOptimized::intersectTriangle<USE_WATERTIGHT>(ray_orig, axisx, axisy, axisz, shear, p0, p1, p2, tmin, tmax, out_t, uv);
}

template<typename TriIndexerT, typename VertexType, typename IndexType, bool USE_MASKING, bool USE_WATERTIGHT>
struct WatertightMeshTriangles
{

  struct RayType
  {
    float3  orig;
    float3  dir; 
    float   tmin;
    float   tmax;
    int     mask;

    float3  ilwDir;

    // Data for watertight intersection
    float3  shear;
    bool    axisx;
    bool    axisy;
    bool    axisz;
  };

  __device__ __forceinline__
  static void fetchRay(const void* const __restrict rays, const int rayIdx, const int rayFormat, RayType& ray)
  {
    ::fetchRay(rays, rayIdx, rayFormat, ray.orig, ray.dir, ray.tmin, ray.tmax, ray.mask);

    WatertightOptimized::setupRay(ray.dir,
      ray.axisx, ray.axisy, ray.axisz, ray.shear, ray.ilwDir);
  }

 
#ifdef INCLUDE_AABB_TESTS

  // References:
  // Robust BVH Ray Traversal, Thiago Ize, JCGT 2013
  // Berger-Perrin, 2004, SSE ray/box intersection test (flipcode), although not used in the lwrrently active variant

  __device__ __forceinline__
  static bool intersectAabb( const RayType& ray, const Aabb& bbox, const float tmin, const float tmax, float& out_t )
  {
#if 1
    // This version does not need the inf/-inf clamp workaround for NaNs (as LWCA min max functions will sort these out (see below)).
    // It will also not suffer from the additional inf/-inf issues mentioned in the TODO below as it does not use min max
    // to "sort" the two intersections per axis, but will rather shuffle the bbox.lo and bbox.hi values accordingly

    /*const float txmin = (slct(bbox.lo.x, bbox.hi.x, float_as_int(ray.ilwDir.x)) - ray.orig.x) * ray.ilwDir.x;
    const float txmax = (slct(bbox.hi.x, bbox.lo.x, float_as_int(ray.ilwDir.x)) - ray.orig.x) * ray.ilwDir.x;
    const float tymin = (slct(bbox.lo.y, bbox.hi.y, float_as_int(ray.ilwDir.y)) - ray.orig.y) * ray.ilwDir.y;
    const float tymax = (slct(bbox.hi.y, bbox.lo.y, float_as_int(ray.ilwDir.y)) - ray.orig.y) * ray.ilwDir.y;
    const float tzmin = (slct(bbox.lo.z, bbox.hi.z, float_as_int(ray.ilwDir.z)) - ray.orig.z) * ray.ilwDir.z;
    const float tzmax = (slct(bbox.hi.z, bbox.lo.z, float_as_int(ray.ilwDir.z)) - ray.orig.z) * ray.ilwDir.z;*/

    const float txmin = (((ray.ilwDir.x < 0.f) ? bbox.hi.x : bbox.lo.x) - ray.orig.x) * ray.ilwDir.x;
    const float txmax = (((ray.ilwDir.x < 0.f) ? bbox.lo.x : bbox.hi.x) - ray.orig.x) * ray.ilwDir.x;
    const float tymin = (((ray.ilwDir.y < 0.f) ? bbox.hi.y : bbox.lo.y) - ray.orig.y) * ray.ilwDir.y;
    const float tymax = (((ray.ilwDir.y < 0.f) ? bbox.lo.y : bbox.hi.y) - ray.orig.y) * ray.ilwDir.y;
    const float tzmin = (((ray.ilwDir.z < 0.f) ? bbox.hi.z : bbox.lo.z) - ray.orig.z) * ray.ilwDir.z;
    const float tzmax = (((ray.ilwDir.z < 0.f) ? bbox.lo.z : bbox.hi.z) - ray.orig.z) * ray.ilwDir.z;

    const float tBegin = fmaxf(fmaxf(tymin, tzmin), fmaxf(txmin, tmin));
          float tEnd   = fminf(fminf(tymax, tzmax), fminf(txmax, tmax));
#elif 0
    // TODO: Smaller code in here disabled for now, as there are corner cases (for example triggered by test_prime) that are
    //       not properly handled by this: Contrary to the comment below, it's not just important to filter NaNs out
    //       but also to handle these cases specifically, as otherwise a wrong outcome can be produced:
    //       For example tBegin=inf or tEnd=-inf if two of the ray dir components = 0 in addition
    //       Thus TBPs inf/-inf workaround below is the correct one for now

    // tlo, thi may contain NaNs if (bbox.lo - ray.orig) == 0 and ray.ilwDir == inf, for some axis.
    // LWCA float min, max are IEEE 754 compliant and suppress NaNs if either one of the arguments is not NaN, see
    // http://docs.lwpu.com/lwca/parallel-thread-exelwtion/index.html#floating-point-instructions-min
    // We assume that the ray span [tmin, tmax] doesn't have NaNs so there are non NaNs left in the final comparison.
    // Effectively dimensions with zero direction are ignored and dimensionality of the intersection test is reduced. 
    // Note that integer VMNMX should not be used as they can propagate NaNs.

    // Previously NaNs were manually replaced here with INFs, similarly to
    // Berger-Perrin, 2004, SSE ray/box intersection test (flipcode).
    // This is not necessary, the Berger-Perrin method is only needed with SSE where float min, max behave differently and
    // return NaN if second argument is NaN.

    const float3 tlo = (bbox.lo - ray.orig) * ray.ilwDir;
    const float3 thi = (bbox.hi - ray.orig) * ray.ilwDir;
  
    const float3 tnear = make_float3(fminf(tlo.x, thi.x), fminf(tlo.y, thi.y), fminf(tlo.z, thi.z));
    const float3 tfar  = make_float3(fmaxf(tlo.x, thi.x), fmaxf(tlo.y, thi.y), fmaxf(tlo.z, thi.z));

    const float tBegin = fmaxf(fmaxf(tnear.x, tmin), fmaxf(tnear.y, tnear.z));
          float tEnd   = fminf(fminf(tfar.x, tmax), fminf(tfar.y, tfar.z));
#else
    // Turn NaNs into +/- inf.  A NaN can occur when (bbox.lo - ray.orig) == 0 and ray.ilwDir == inf, for some axis.
    // see Berger-Perrin.
    
    const float3 tlo = (bbox.lo - ray.orig) * ray.ilwDir;
    const float3 thi = (bbox.hi - ray.orig) * ray.ilwDir;

    const float inf = __int_as_float(0x7F800000);

    const float3 tlo_clamp_neginf = make_float3(fmaxf(tlo.x, -inf), fmaxf(tlo.y, -inf), fmaxf(tlo.z, -inf));
    const float3 thi_clamp_neginf = make_float3(fmaxf(thi.x, -inf), fmaxf(thi.y, -inf), fmaxf(thi.z, -inf));

    const float tBegin = spanBegin(tlo_clamp_neginf.x, thi_clamp_neginf.x, 
                                   tlo_clamp_neginf.y, thi_clamp_neginf.y, 
                                   tlo_clamp_neginf.z, thi_clamp_neginf.z,
                                   tmin);

    const float3 tlo_clamp_inf = make_float3(fminf(tlo.x, inf), fminf(tlo.y, inf), fminf(tlo.z, inf));
    const float3 thi_clamp_inf = make_float3(fminf(thi.x, inf), fminf(thi.y, inf), fminf(thi.z, inf));

          float tEnd   = spanEnd(tlo_clamp_inf.x, thi_clamp_inf.x, 
                                 tlo_clamp_inf.y, thi_clamp_inf.y, 
                                 tlo_clamp_inf.z, thi_clamp_inf.z,
                                 tmax);
#endif

    //  1+3*ulp, suggested by cwaechter (maybe due to rcp.approximate being triggered for ray.ilwDir? -> check)
    tEnd *= 1.0000003576278687f;
    out_t = tBegin;
    return tBegin <= tEnd;
  }

#endif // INCLUDE_AABB_TESTS

  __device__ __forceinline__
  static void intersectTriangles(const RayType &ray, const TracerParamsMesh& p, const int instIdx, int& leafIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, int& nodeIdx, char** stackPtr, const bool anyhit)
  {
    while (leafIdx < 0)
    {
        // Intersect the ray against each triangle
        int triIdx = ~leafIdx;
        const int lastPrim = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx]) >> RLLEPACK_LEN_SHIFT) + triIdx;

        do
        {
            const int remap = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx])) & RLLEPACK_INDEX_MASK;
                  
            float t;
            if( triangleIntersectWatertight<TriIndexerT, VertexType, IndexType, USE_MASKING, USE_WATERTIGHT>(p, remap,
                ray.orig, ray.axisx, ray.axisy, ray.axisz, ray.shear, ray.tmin, hitT, ray.mask, t) )
            {
              hitT = t;
              hitTriIdx = triIdx;
              hitInstIdx = instIdx;
              if(anyhit)
              {
                nodeIdx = EntrypointSentinel;
                break;
              }
            }
            triIdx++;
        }
        while(triIdx<lastPrim); // Not the last triangle => keep going.

        leafIdx = nodeIdx;
        if (nodeIdx < 0)
            nodeIdx = pop( *stackPtr );
    } // leaf
  }

  __device__ __forceinline__
  static bool intersectTriangles52(const RayType& ray, const TracerParamsMesh& p, const int instIdx, int triIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, const bool anyhit)
  {
    bool foundHit = false;
    const int lastPrim = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx]) >> RLLEPACK_LEN_SHIFT) + triIdx;

    do
    {
        const int remap = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx])) & RLLEPACK_INDEX_MASK;

        float t;
        if (triangleIntersectWatertight<TriIndexerT, VertexType, IndexType, USE_MASKING, USE_WATERTIGHT>(p, remap,
            ray.orig, ray.axisx, ray.axisy, ray.axisz, ray.shear, ray.tmin, hitT, ray.mask, t))
        {
            hitT = t;
            hitTriIdx = triIdx;
            hitInstIdx = instIdx;
            foundHit = true;

            if (anyhit)
                break;
        }
        triIdx++;
    } while (triIdx<lastPrim); // Not the last triangle => keep going.
    return foundHit;
  }

  __device__ __forceinline__
  static void computeUV(const TracerParamsMesh& p, const int triIndex, float3& orig, float3& dir, const float t, float& u, float& v)
  {
    const int  remapped = remap( p.remap, triIndex );
    const IndexType vi  = TriIndexerT::getIndices((IndexType*)p.indices, remapped, p.indexStride);

    const VertexType* const __restrict vertices = reinterpret_cast<const VertexType*>( p.vertices );

    const float3 p0 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.x * p.vertexStride) );
    const float3 p1 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.y * p.vertexStride) );
    const float3 p2 = make_float3( *(VertexType*)((char*)vertices + (size_t)vi.z * p.vertexStride) );

    const float3 e0 = p1 - p0;
    const float3 e1 = p0 - p2;
    const float3 n  = cross( e1, e0 );

    const float3 e2 = ( 1.0f / dot( n, dir ) ) * ( p0 - orig );
    const float3 i  = cross( dir, e2 );

    const float beta  = dot( i, e1 );
    const float gamma = dot( i, e0 );

    u = 1.0f - beta - gamma;
    v = beta;
  }

  __device__ __forceinline__
  static int remap( const int* const __restrict ptr, const int i )
  {
    return LoadLdg<int>::load(ptr, i) & RLLEPACK_INDEX_MASK;
  }
};

// user tris intersection test for kernels that do (n)ot use (s)pelwlative (t)raversal
template<typename TriIndexerT, typename VertexType, typename IndexType, bool USE_MASKING>
struct MeshTrianglesNST : public MeshTriangles<TriIndexerT, VertexType, IndexType, USE_MASKING>
{

  typedef MeshTriangles<TriIndexerT, VertexType, IndexType, USE_MASKING> BaseClassType;
  typedef typename BaseClassType::RayType RayType;

  __device__ __forceinline__
  static void fetchAndTransformRay(const void* const __restrict rays, const int rayIdx, const int rayFormat, RayType& ray, const float4* const __restrict im)
  {
    ::fetchRay(rays, rayIdx, rayFormat, ray.orig, ray.dir, ray.tmin, ray.tmax);
    ::transformRay(ray.orig, ray.dir, im);
    ::setupRay(ray.orig, ray.dir, ray.rcpDir, ray.origOverDir);
  }

  __device__ __forceinline__
  static void transformRay(RayType& ray, const float4* const __restrict im)
  {
    ::transformRay(ray.orig, ray.dir, im);
    ::setupRay(ray.orig, ray.dir, ray.rcpDir, ray.origOverDir);
  }

  __device__ __forceinline__
  static void intersectTriangles(const RayType& ray, const TracerParamsMesh& p, const int instIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, int& nodeIdx, const bool anyhit)
  { 
    // Intersect the ray against each triangle
    int triIdx = ~nodeIdx;
    const int lastPrim = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx]) >> RLLEPACK_LEN_SHIFT) + triIdx;

    do
    {
        const int remap = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx])) & RLLEPACK_INDEX_MASK;
                  
        float t;
        if( triangleIntersect<TriIndexerT, VertexType, IndexType, USE_MASKING>(p, remap, ray.orig, ray.dir, ray.tmin, hitT, ray.mask, t) )
        {
          hitT = t;
          hitTriIdx = triIdx;
          hitInstIdx = instIdx;
          if(anyhit)
          {
            nodeIdx = EntrypointSentinel;
            break;
          }
        }
        triIdx++;
    }
    while(triIdx < lastPrim); // Not the last triangle => keep going.
  }
};

// Watertight intersection test for kernels that do (n)ot use (s)pelwlative (t)raversal
template<typename TriIndexerT, typename VertexType, typename IndexType, bool USE_MASKING, bool USE_WATERTIGHT>
struct WatertightMeshTrianglesNST : public WatertightMeshTriangles<TriIndexerT, VertexType, IndexType, USE_MASKING, USE_WATERTIGHT>
{

  typedef WatertightMeshTriangles<TriIndexerT, VertexType, IndexType, USE_MASKING, USE_WATERTIGHT> BaseClassType;
  typedef typename BaseClassType::RayType RayType;

  __device__ __forceinline__
  static void fetchAndTransformRay(const void* const __restrict rays, const int rayIdx, const int rayFormat, RayType& ray, const float4* const __restrict im)
  {
    // Note: calling BaseClass::fetchRay would do the watertight setup.  We delay that until after the transform.
    ::fetchRay(rays, rayIdx, rayFormat, ray.orig, ray.dir, ray.tmin, ray.tmax, ray.mask);
    ::transformRay(ray.orig, ray.dir, im);

    WatertightOptimized::setupRay(ray.dir,
      ray.axisx, ray.axisy, ray.axisz, ray.shear, ray.ilwDir);
    
  }

  __device__ __forceinline__
  static void transformRay(RayType& ray, const float4* const __restrict im)
  {
    ::transformRay(ray.orig, ray.dir, im);

    WatertightOptimized::setupRay(ray.dir,
      ray.axisx, ray.axisy, ray.axisz, ray.shear, ray.ilwDir);
  }

    __device__ __forceinline__
  static void intersectTriangles(const RayType& ray, const TracerParamsMesh& p, const int instIdx, float& hitT, int& hitTriIdx, int& hitInstIdx, int& nodeIdx, const bool anyhit)
  { 
    // Intersect the ray against each triangle
    int triIdx = ~nodeIdx;
    const int lastPrim = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx]) >> RLLEPACK_LEN_SHIFT) + triIdx;

    do
    {
        const int remap = ((unsigned)LDG_OR_GLOBAL(&p.remap[triIdx])) & RLLEPACK_INDEX_MASK;
                  
        float t;
        if( triangleIntersectWatertight<TriIndexerT, VertexType, IndexType, USE_MASKING, USE_WATERTIGHT>(p, remap,
            ray.orig, ray.axisx, ray.axisy, ray.axisz, ray.shear, ray.tmin, hitT, ray.mask, t) )
        {
          hitT = t;
          hitTriIdx = triIdx;
          hitInstIdx = instIdx;
          if(anyhit)
          {
            nodeIdx = EntrypointSentinel;
            break;
          }
        }
        triIdx++;
    }
    while(triIdx < lastPrim); // Not the last triangle => keep going.
  }
};

template<typename NodeLoaderT, typename TrianglesT>
struct Combine : public TrianglesT
{
  __device__ __forceinline__
  static float4 loadNodes(const float4* const __restrict ptr, const int offset)
  { 
    return NodeLoaderT::load(ptr, offset);
  }
};

template<typename C> __device__ __forceinline__
static void fetchChildNodes( const float4* const __restrict nodes, const int nodeIdx, Node& child0, Node& child1 ) 
{ 
  const float4 n0 = C::loadNodes(nodes, nodeIdx*4    );
  const float4 n1 = C::loadNodes(nodes, nodeIdx*4 + 1);
  const float4 n2 = C::loadNodes(nodes, nodeIdx*4 + 2);
  const float4 n3 = C::loadNodes(nodes, nodeIdx*4 + 3);

  child0.bbox.lo = make_float3(n0.x, n0.y, n0.z);
  child0.bbox.hi = make_float3(n0.w, n1.x, n1.y);
  child1.bbox.lo = make_float3(n2.x, n2.y, n2.z);
  child1.bbox.hi = make_float3(n2.w, n3.x, n3.y);
  
  child0.addr    = __float_as_int(n1.z);
  child1.addr    = __float_as_int(n3.z);
  
  //int2 cpad       = make_int2(__float_as_int(n1.w), __float_as_int(n3.w));
}
