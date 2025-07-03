/*
 *  Copyright (c) 2016, LWPU Corporation
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

//------------------------------------------------------------------------
// Naive CPU tracer for 8-wide BVHs.
// For further details, please refer to:
// https://p4viewer.lwpu.com/get///research/research/hylitie/docs/Efficient-RT%202016-07-09.pptx 
//------------------------------------------------------------------------

#include "TracerBVH8Naive.hpp"
#include <prodlib/bvhtools/include/BVH8Types.hpp>
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include "WatertightOriginal.hpp"

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#include <emmintrin.h>
#endif

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

static inline float3 operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }


#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)

//------------------------------------------------------------------------

struct StackEntry
{
    int32_t firstChildIdx;
    uint32_t hits;
};

//------------------------------------------------------------------------

static inline void colwertIL8bitToFloat(__m128i x, const int j, __m128& y0, __m128& y1)
{
    // j==0: XXXX 0000 XXXX 0000 = 4x 4x8bit data blocks (X=needed, 0=not needed)
    // j==1: 0000 XXXX 0000 XXXX = 4x 4x8bit data blocks (X=needed, 0=not needed)

    // shuffle: XXXX XXXX 0000 0000
    x = (j==0) ? _mm_shuffle_epi32(x, _MM_SHUFFLE(0,0,2,0)) : _mm_shuffle_epi32(x, _MM_SHUFFLE(0,0,3,1));

    // blow up each 8bit data to 16bit by padding 0s
    x = _mm_unpacklo_epi8(x, _mm_setzero_si128());

    // blow up each 16bit data to 32bit by padding 0s
    const __m128i xlo = _mm_unpacklo_epi16(x, _mm_setzero_si128());
    const __m128i xhi = _mm_unpackhi_epi16(x, _mm_setzero_si128());

    // colwert to float
    y0 = _mm_cvtepi32_ps(xlo);
    y1 = _mm_cvtepi32_ps(xhi);
}

//------------------------------------------------------------------------

static inline uint32_t getDuplicatedOctant(const float3& dir)
{
    // Use integer sign bit to prevent -0.0 to be treated as 0.0 = positive.
    uint32_t oct = (__float_as_int(dir.x) < 0) ? 1 : 0;
    if (__float_as_int(dir.y) < 0) oct |= 2;
    if (__float_as_int(dir.z) < 0) oct |= 4;
    return oct * 0x01010101u;
}

//------------------------------------------------------------------------

template <class Spec>
static inline void intersectNode(const BVH8Node* const node, const typename Spec::Ray& ray, const uint32_t duplicatedOctant, StackEntry& innerHits, StackEntry& leafHits)
{
    __m128i pos_scale_innermask = _mm_load_si128((const __m128i*)&node->header); // lower 3 floats are used directly as pos below
    const __m128i child_remap_meta = _mm_load_si128(((const __m128i*)&node->header)+1);

    innerHits.firstChildIdx = _mm_cvtsi128_si32(child_remap_meta); // extract firstChildIdx
    leafHits.firstChildIdx = _mm_cvtsi128_si32(_mm_shuffle_epi32(child_remap_meta, _MM_SHUFFLE(0, 0, 0, 1))); // extract firstRemapIdx

    const uint64_t meta = _mm_cvtsi128_si64(_mm_shuffle_epi32(child_remap_meta, _MM_SHUFFLE(0, 0, 3, 2))); // extract meta
    
    // Shift 8-bit scale to floating-point exponent bits.
    __m128i scalei = _mm_shuffle_epi32(pos_scale_innermask, _MM_SHUFFLE(0, 0, 0, 3)); // extract scale (lower 3 uchars in highest uint)
    innerHits.hits = (unsigned int)_mm_cvtsi128_si32(scalei) >> 24; // while at it, also extract innerMask (most upper uchar)
    // continue unpacking the uchars (in the lowest uint now), then shift all separately by 23 (e.g. into FP exponent)
    scalei = _mm_unpacklo_epi8 (scalei, _mm_setzero_si128());
    scalei = _mm_unpacklo_epi16(scalei, _mm_setzero_si128());
    const __m128i maskw = _mm_set_epi32(0, ~0u, ~0u, ~0u);
    pos_scale_innermask = _mm_and_si128(pos_scale_innermask,maskw); // mask out w component to avoid potential NaN/Inf slowdowns
    scalei = _mm_and_si128(scalei,maskw);                           // dto.
    __m128 scale = _mm_castsi128_ps(_mm_slli_epi32(scalei, 23));

    const __m128 rayID = _mm_set_ps(0.f, ray.ilwDir.z, ray.ilwDir.y, ray.ilwDir.x);
    const __m128 rayO  = _mm_set_ps(0.f, ray.orig.z,   ray.orig.y,   ray.orig.x);

    __m128 a;
    if (!Spec::useWatertight())
    {
        scale = _mm_mul_ps(scale,rayID); // Approximate => premultiply tlo and thi with ilwDir; more efficient.
        a = _mm_mul_ps(_mm_sub_ps(_mm_castsi128_ps(pos_scale_innermask),rayO),rayID);
    }
    else
        a = _mm_sub_ps(_mm_castsi128_ps(pos_scale_innermask),rayO);

    const int signRayID = _mm_movemask_ps(rayID);

    uint32_t hits = 0;
    for (int j = 0; j < 2; j++)
    {
        // Leaves insert numtris bits to dynamic hitmask; inner nodes insert 1 and empty nodes 0.
        uint32_t ofs = (j == 0) ? (uint32_t)meta : (uint32_t)(meta >> 32);
        const uint32_t trimask = (ofs >> 5) & 0x07070707; // empty children insert 0 even if hit (shouldn't normally happen).
        uint32_t innermask = (ofs & (ofs << 1)) & 0x10101010; // all inner nodes have bit 5,4 (16,8). Leaves may have one set bot not both.
        innermask = (innermask << 3) & 0x80808080u; // sbyte sign marks inner nodes
        innermask = (innermask << 1) - (innermask >> 7); // extend sign to all bits

        ofs ^= (duplicatedOctant & innermask); // compute traversal priority for inner node children only.
        ofs &= 0x1F1F1F1F; // Low 5 bits contain the offset, mask others out. Not needed with vshl.wrap.

        // Load 8bit bounds and colwert to float

        __m128 lox, loy, loz, hix;
        colwertIL8bitToFloat(_mm_load_si128((const __m128i*)node->lox), j, lox, loy);
        colwertIL8bitToFloat(_mm_load_si128((const __m128i*)node->loz), j, loz, hix);

        // Select tnear, tfar by ray sign for x and y and scale/translate

        const __m128 nearx = (signRayID & 1) == 0 ? lox : hix;
        const __m128 farx  = (signRayID & 1) == 0 ? hix : lox;
        const __m128 ax = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 sx = _mm_shuffle_ps(scale, scale, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 tlox = _mm_add_ps(ax, _mm_mul_ps(sx, nearx));
        __m128 thix = _mm_add_ps(ax, _mm_mul_ps(sx, farx));

        __m128 hiy, hiz;
        colwertIL8bitToFloat(_mm_load_si128((const __m128i*)node->hiy), j, hiy, hiz);

        const __m128 neary = (signRayID & 2) == 0 ? loy : hiy;
        const __m128 fary  = (signRayID & 2) == 0 ? hiy : loy;
        const __m128 ay = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 sy = _mm_shuffle_ps(scale, scale, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 tloy = _mm_add_ps(ay, _mm_mul_ps(sy, neary));
        __m128 thiy = _mm_add_ps(ay, _mm_mul_ps(sy, fary));

        const __m128 nearz = (signRayID & 4) == 0 ? loz : hiz;
        const __m128 farz  = (signRayID & 4) == 0 ? hiz : loz;
        const __m128 az = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 sz = _mm_shuffle_ps(scale, scale, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 tloz = _mm_add_ps(az, _mm_mul_ps(sz, nearz));
        __m128 thiz = _mm_add_ps(az, _mm_mul_ps(sz, farz));

        if (Spec::useWatertight())
        {
            // postmultiply tlo and thi with ilwDir; more precise.
            const __m128 rx = _mm_shuffle_ps(rayID, rayID, _MM_SHUFFLE(0, 0, 0, 0));
            tlox = _mm_mul_ps(tlox, rx);
            thix = _mm_mul_ps(thix, rx);
            const __m128 ry = _mm_shuffle_ps(rayID, rayID, _MM_SHUFFLE(1, 1, 1, 1));
            tloy = _mm_mul_ps(tloy, ry);
            thiy = _mm_mul_ps(thiy, ry);
            const __m128 rz = _mm_shuffle_ps(rayID, rayID, _MM_SHUFFLE(2, 2, 2, 2));
            tloz = _mm_mul_ps(tloz, rz);
            thiz = _mm_mul_ps(thiz, rz);

            // References:
            // Robust BVH Ray Traversal, Thiago Ize, JCGT 2013
            // Berger-Perrin, 2004, SSE ray/box intersection test (flipcode)

            // Turn NaNs into +/- inf.  A NaN can occur when (bbox.lo - ray.orig) == 0 and ray.ilwDir == inf, for some axis.
            // see Berger-Perrin.

            const __m128 ninf = _mm_set1_ps(-LWDART_INF_F);
            tlox = _mm_max_ps(tlox, ninf);
            tloy = _mm_max_ps(tloy, ninf);
            tloz = _mm_max_ps(tloz, ninf);
            const __m128 inf = _mm_set1_ps(LWDART_INF_F);
            thix = _mm_min_ps(thix, inf);
            thiy = _mm_min_ps(thiy, inf);
            thiz = _mm_min_ps(thiz, inf);
        }

        const __m128 tmin = _mm_max_ps(_mm_max_ps(_mm_max_ps(tloz, _mm_set1_ps(ray.tmin)), tlox), tloy);
              __m128 tmax = _mm_min_ps(_mm_min_ps(_mm_min_ps(thiz, _mm_set1_ps(ray.tmax)), thix), thiy);

        if (Spec::useWatertight())
            tmax = _mm_mul_ps(tmax, _mm_set1_ps(1.0000003576278687f)); // 1+3*ulp, suggested by cwaechter

        const int cmp = _mm_movemask_ps(_mm_cmple_ps(tmin, tmax));

        int i = 1;
        for (int i8 = 0; i < 32; i+=i,i8+=8)
        {
            if (cmp & i)
            {
                // If hit, insert bit(s) to correct position in hitmask
                hits |= ((trimask >> i8) & 0xFF) << ((ofs >> i8) & 0xFF);
            }
        }
    }

    // Extract inner, leaf node hits
    leafHits.hits = hits & 0x00FFFFFF;
    innerHits.hits |= (hits & 0xFF000000);
}

//------------------------------------------------------------------------

template <bool USE_EXTERNAL_TRIS, bool USE_WATERTIGHT>
struct TraceSpecialization
{
    struct Ray
    {
        float3          orig;
        float3          dir;
        float           tmin;
        float           tmax;
        unsigned int    mask;

        // Additional data for ray/triangle test and ray/AABB test.

        float3          translate;
        float3          shear;
        float3          ilwDir;
        int             majorAxis;
    };

    struct Model
    {
        const BVH8Node*     nodes;
        const BVH8Triangle* triangles;

        // Additional data for external triangles.

        const char*         indices;
        const char*         vertices;
        const int*          remap;
        int                 indexStride;
        int                 vertexStride;
    };

    static inline void fetchRay(Ray& ray, const void* const rays, const int rayIdx, const int rayFormat)
    {
        //ray.orig = make_float3(0.0f, 0.0f, 0.0f);
        //ray.dir = make_float3(0.0f, 0.0f, 0.0f);
        ray.tmin = 0.0f;
        //ray.tmax = 0.0f;
        ray.mask = 0;
        ray.translate = make_float3(0.0f, 0.0f, 0.0f);
        ray.shear = make_float3(0.0f, 0.0f, 0.0f);
        ray.ilwDir = make_float3(0.0f, 0.0f, 0.0f);
        ray.majorAxis = 0;

        switch (rayFormat)
        {
        default:
        case RAY_ORIGIN_DIRECTION:
            loadUncachedFloat6(&((float3*)rays)[rayIdx * 2], ray.orig.x, ray.orig.y, ray.orig.z, ray.dir.x, ray.dir.y, ray.dir.z);
            ray.tmax = 1.0e34f;
            break;

        case RAY_ORIGIN_TMIN_DIRECTION_TMAX:
            {
                const float4 o = ((float4*)rays)[rayIdx * 2 + 0];
                const float4 d = ((float4*)rays)[rayIdx * 2 + 1];
                ray.orig = make_float3(o.x, o.y, o.z);
                ray.dir  = make_float3(d.x, d.y, d.z);
                ray.tmin = o.w;
                ray.tmax = d.w;
            }
            break;

        case RAY_ORIGIN_MASK_DIRECTION_TMAX:
            {
                const float4 o = ((float4*)rays)[rayIdx * 2 + 0];
                const float4 d = ((float4*)rays)[rayIdx * 2 + 1];
                ray.orig = make_float3(o.x, o.y, o.z);
                ray.dir  = make_float3(d.x, d.y, d.z);
                ray.tmax = d.w;
                ray.mask = __float_as_int(o.w);
            }
            break;
        }        
    }

    static inline void storeHit(void* const hits, const int rayIdx, const int hitFormat, const int triIndex, const int instanceIndex, const float t, const float u, const float v)
    {
        switch( hitFormat )
        {
        case HIT_T_TRIID_INSTID:
            ((float*)hits)[rayIdx * 3 + 0] = t;
            ((float*)hits)[rayIdx * 3 + 1] = __int_as_float(triIndex);
            ((float*)hits)[rayIdx * 3 + 2] = __int_as_float(instanceIndex);
            break;

        case HIT_T_TRIID_INSTID_U_V:
            ((float*)hits)[rayIdx * 5 + 0] = t;
            ((float*)hits)[rayIdx * 5 + 1] = __int_as_float(triIndex);
            ((float*)hits)[rayIdx * 5 + 2] = __int_as_float(instanceIndex);
            ((float*)hits)[rayIdx * 5 + 3] = u;
            ((float*)hits)[rayIdx * 5 + 4] = v;
            break;

        case HIT_T_TRIID_U_V:
            ((float4*)hits)[rayIdx] = make_float4(t, __int_as_float(triIndex), u, v);
            break;

        case HIT_T_TRIID:
            ((float2*)hits)[rayIdx] = make_float2(t, __int_as_float(triIndex));
            break;

        case HIT_T:
            ((float*)hits)[rayIdx]  = t;
            break;

        case HIT_BITMASK:
            if (t >= 0.0f)
                ((uint32_t*)hits)[rayIdx >> 5] |= 1u << (rayIdx & 31);
            break;
        }
    }

    static inline void setupRay(Ray& ray)
    {
        // Clamp ray direction to avoid division by zero.
        // Note: In the watertight mode, we disable the clamping to get as accurate results as possible.

        if (!USE_WATERTIGHT)
        {
            const float eps = exp2f(-80.0f);
            ray.dir.x = (fabsf(ray.dir.x) >= eps) ? ray.dir.x : bitSelect(ray.dir.x, eps, __uint_as_float(0x80000000));
            ray.dir.y = (fabsf(ray.dir.y) >= eps) ? ray.dir.y : bitSelect(ray.dir.y, eps, __uint_as_float(0x80000000));
            ray.dir.z = (fabsf(ray.dir.z) >= eps) ? ray.dir.z : bitSelect(ray.dir.z, eps, __uint_as_float(0x80000000));
        }
        WatertightOriginal::setupRay(ray.orig, ray.dir, ray.majorAxis, ray.translate, ray.shear);
        ray.ilwDir.x = 1.0f / ray.dir.x;
        ray.ilwDir.y = 1.0f / ray.dir.y;
        ray.ilwDir.z = 1.0f / ray.dir.z;
    }

    static inline void initModel(Model& model)
    {
        model.nodes         = NULL;
        model.triangles     = NULL;
        model.indices       = NULL;
        model.vertices      = NULL;
        model.remap         = NULL;
        model.indexStride   = 0;
        model.vertexStride  = 0;
    }

    static inline void initModel(Model& model, const TracerDataMeshBVH8& tdm)
    {
        initModel(model);
        model.nodes = (const BVH8Node*)tdm.bvh8Nodes;

        if (!USE_EXTERNAL_TRIS)
            model.triangles = (const BVH8Triangle*)tdm.bvh8Triangles;
        else
        {
            model.indices       = (const char*)tdm.mesh.indices;
            model.vertices      = (const char*)tdm.mesh.vertices;
            model.remap         = tdm.remap;
            model.indexStride   = tdm.mesh.indexStride;
            model.vertexStride  = tdm.mesh.vertexStride;
        }
    }

    static inline void initModel(Model& model, const TracerDataGroupBVH8& tdg)
    {
        initModel(model);
        model.nodes = tdg.bvh8Nodes;
    }

    static inline void fetchTriangle(BVH8Triangle& tri, const Model& model, const int triIdx, const int rayFormat)
    {
        if (!USE_EXTERNAL_TRIS)
            tri = model.triangles[triIdx];
        else
        {
            // Fetch triangle index from the remap array.

            tri.userTriangleID = model.remap[triIdx];

            // Fetch vertex indices and mask.

            int3 vidx;
            if (!model.indices)
            {
                const int base = tri.userTriangleID * 3;
                vidx = make_int3(base + 0, base + 1, base + 2);
                tri.mask = 0;
            }
            else
            {
                const char* const indexPtr = model.indices + (size_t)tri.userTriangleID * model.indexStride;
                vidx = *(const int3*)indexPtr;
                tri.mask = (rayFormat != RAY_ORIGIN_MASK_DIRECTION_TMAX) ? 0 : ((const int*)indexPtr)[3];
            }

            // Fetch vertex positions.

            const float3 v0 = *(const float3*)(model.vertices + (size_t)vidx.x * model.vertexStride);
            const float3 v1 = *(const float3*)(model.vertices + (size_t)vidx.y * model.vertexStride);
            const float3 v2 = *(const float3*)(model.vertices + (size_t)vidx.z * model.vertexStride);
            tri.v0x = v0.x, tri.v0y = v0.y, tri.v0z = v0.z;
            tri.v1x = v1.x, tri.v1y = v1.y, tri.v1z = v1.z;
            tri.v2x = v2.x, tri.v2y = v2.y, tri.v2z = v2.z;
        }
    }

    static inline bool intersectTriangle(const BVH8Triangle& tri, const Ray& ray, float& outT, float& outU, float& outV)
    {
        if (USE_EXTERNAL_TRIS)
        {
            if ((tri.mask & ray.mask) != 0)
                return false;
        }

        return WatertightOriginal::intersectTriangle(ray.majorAxis, ray.translate, ray.shear,
            make_float3(tri.v0x, tri.v0y, tri.v0z), make_float3(tri.v1x, tri.v1y, tri.v1z), make_float3(tri.v2x, tri.v2y, tri.v2z),
            ray.tmin, ray.tmax, &outT, &outU, &outV);
    }

    static inline bool useWatertight()
    {
        return USE_WATERTIGHT;
    }
};

//------------------------------------------------------------------------

template <class Spec>
static void traceFlat(const int rayIdx, const TracerDataMeshBVH8& tdm, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const bool anyHit)
{
    // Init ray.
    typename Spec::Ray ray;
    Spec::fetchRay(ray, rays, rayIdx, rayFormat);
    Spec::setupRay(ray);
    const uint32_t duplicatedOctant = getDuplicatedOctant(ray.dir);

    // Init hit.
    int     hitTriID = -1;                  // Triangle index of the closest intersection, -1 if none.
    float   hitU = 0.0f;
    float   hitV = 0.0f;

    // Init model.
    typename Spec::Model model;
    Spec::initModel(model, tdm);

    // Init stack.
    StackEntry stack[64];
    StackEntry stackTop;
    stackTop.hits = 1u << 24;
    stackTop.firstChildIdx = 0; // root
    int sstackPtr = 0;

    // Traversal loop.
    for (;;)
    {
        uint32_t primitiveHits;
        uint32_t itemStart;

        if (stackTop.hits > 0x00FFFFFF) // if inner node hits
        {
            int idx = findLeadingOne(stackTop.hits);
            stackTop.hits ^= 1u << idx;
            idx -= 24;
            idx ^= (duplicatedOctant & 0x7); // inner node idx in range [0,7]
            uint32_t validChildren = stackTop.hits; // Valid mask in low byte.
            idx = __popc(validChildren & ~(-1 << idx)); // Compute number if sibling nodes in memory before this node.

            // Issue global loads as early as possible
            const BVH8Node* const node = &model.nodes[stackTop.firstChildIdx + idx];

            // If stacktop still contains hits to test, push it to stack.
            if (stackTop.hits > 0x00FFFFFF)
                stack[sstackPtr++] = stackTop;

            StackEntry leafHits;
            intersectNode<Spec>(node, ray, duplicatedOctant, stackTop, leafHits);
            itemStart = leafHits.firstChildIdx;
            primitiveHits = leafHits.hits;
        }
        else
        {
            primitiveHits = stackTop.hits;
            itemStart = stackTop.firstChildIdx;
            stackTop.hits = 0;
        }

        while (primitiveHits)
        {
            // Select a Triangle
            const int idx = findLeadingOne(primitiveHits);
            // Clear it from list
            primitiveHits ^= 1u << idx;

            BVH8Triangle tri;
            Spec::fetchTriangle(tri, model, itemStart + idx, rayFormat);

            float t, u, v;
            if (Spec::intersectTriangle(tri, ray, t, u, v))
            {
                ray.tmax = t;
                hitTriID = tri.userTriangleID;
                hitU = u;
                hitV = v;
                if (anyHit)
                    break;
            }
        }


        // pop
        if (stackTop.hits <= 0x00FFFFFF)
        {
            if (sstackPtr == 0)
                break;
            stackTop = stack[--sstackPtr];
        }
    } // traversal

    // Store result.
    int hitInstIdx = 0;
    if (hitTriID == -1)
    {
        ray.tmax = -1.0f;
        hitInstIdx = -1;
    }
    Spec::storeHit(hits, rayIdx, hitFormat, hitTriID, hitInstIdx, ray.tmax, hitU, hitV);
}

//------------------------------------------------------------------------

template <class Spec>
static void traceInst(const int rayIdx, const TracerDataGroupBVH8& tdg, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const bool anyHit)
{
    const int EntrypointSentinel = 0x76543210;

    // Init ray.
    typename Spec::Ray ray;
    Spec::fetchRay(ray, rays, rayIdx, rayFormat);
    Spec::setupRay(ray);
    uint32_t duplicatedOctant = getDuplicatedOctant(ray.dir);
    const float3 wsorig = ray.orig;
    const float3 wsdir = ray.dir;

    // Init hit.
    int     hitTriID = -1;      // Triangle index of the closest intersection, -1 if none.
    int     hitInstIdx = -1;    // Instance index of the closest intersection, -1 if none
    int     instIdx = -1;       // Instance index
    float   hitU = 0.0f;
    float   hitV = 0.0f;

    // Init top-level BVH.
    typename Spec::Model model;
    Spec::initModel(model, tdg);

    // Init stack.
    StackEntry stack[64];
    StackEntry stackTop;
    stackTop.hits = 1u << 24;
    stackTop.firstChildIdx = 0; // root
    int sstackPtr = 0;

    // Traversal loop.
    for (;;)
    {
        uint32_t primitiveHits = 0;
        uint32_t itemStart = 0;
        int idx = findLeadingOne(stackTop.hits);

        if (stackTop.hits > 0x00FFFFFF && stackTop.firstChildIdx != EntrypointSentinel) // if inner node hits
        {
            stackTop.hits ^= 1u << idx;
            idx -= 24;
            idx ^= (duplicatedOctant & 0x7); // inner node idx in range [0,7]
            const uint32_t validChildren = stackTop.hits; // Valid mask in low byte.
            idx = __popc(validChildren & ~(-1 << idx)); // Compute number if sibling nodes in memory before this node.


            // Issue global loads as early as possible
            const BVH8Node* const node = &model.nodes[stackTop.firstChildIdx + idx];

            // If stacktop still contains hits to test, push it to stack.
            if (stackTop.hits > 0x00FFFFFF)
                stack[sstackPtr++] = stackTop;

            StackEntry leafHits;
            intersectNode<Spec>(node, ray, duplicatedOctant, stackTop, leafHits);
            itemStart = leafHits.firstChildIdx;
            primitiveHits = leafHits.hits;

            if (primitiveHits)
            {
                // push remaining leaves?
                StackEntry tricluster;
                tricluster.firstChildIdx = itemStart;
                tricluster.hits = primitiveHits;

                stack[sstackPtr++] = tricluster;

                primitiveHits = 0;
            }
        }
        else if (stackTop.firstChildIdx != EntrypointSentinel)
        {
            primitiveHits = stackTop.hits;
            itemStart = stackTop.firstChildIdx;
            stackTop.hits = 0;
        }

        // all leaves come from stack


        //Each stackentry has only 1 type items: inner or leaves, not both!!
        if (primitiveHits)
        {
            if (instIdx < 0)
            {
                // Select a leaf
                idx = findLeadingOne(primitiveHits);
                // Clear it from list
                primitiveHits ^= 1u << idx;
                const int leafIdx = itemStart + idx;


                instIdx = tdg.remap[leafIdx];
                const int modelId = tdg.group.modelIds[instIdx]; // nodes, leaves ptrs could be stored together with xform

                const float4* const ilwTransform = getTransformPtr(tdg.group.ilwMatrices, instIdx, tdg.group.matrixStride);
                ray.orig = wsorig;
                ray.dir = wsdir;
                transformRay(ray.orig, ray.dir, ilwTransform);
                Spec::setupRay(ray);
                duplicatedOctant = getDuplicatedOctant(ray.dir);

                Spec::initModel(model, tdg.bvh8Meshes[modelId]);


                // push old stacktop, remaining leaves
                if (stackTop.hits > 0x00FFFFFF)
                    stack[sstackPtr++] = stackTop;
                stackTop.hits = 0;

                // each leaf here is instance
                if (primitiveHits)
                {
                    // push remaining leaves?
                    StackEntry tricluster;
                    tricluster.firstChildIdx = itemStart;
                    tricluster.hits = primitiveHits;

                    stack[sstackPtr++] = tricluster;
                }

                // set an end mark for model traversal
                StackEntry entry;
                entry.firstChildIdx = EntrypointSentinel;
                entry.hits = 0;
                stack[sstackPtr++] = entry;

                // make stackTop to be new model's root
                stackTop.hits = 1u << 24;
                stackTop.firstChildIdx = 0; // root

                if (tdg.bvh8Meshes[modelId].numRemaps == 0)
                    stackTop.hits = 0;
            }
            else
            {
                while (primitiveHits)
                {
                    // Select a leaf
                    idx = findLeadingOne(primitiveHits);
                    // Clear it from list
                    primitiveHits ^= 1u << idx;

                    BVH8Triangle tri;
                    Spec::fetchTriangle(tri, model, itemStart + idx, rayFormat);

                    float t, u, v;
                    // TODO use test from Iray? less precomputed values.
                    if (Spec::intersectTriangle(tri, ray, t, u, v))
                    {
                        ray.tmax = t;
                        hitTriID = tri.userTriangleID;
                        hitInstIdx = instIdx;
                        hitU = u;
                        hitV = v;
                        if (anyHit)
                            break;
                    }
                }
            }
        }

        // end marker for model traversal
        if (stackTop.firstChildIdx == EntrypointSentinel)
        {
            instIdx = -1; // switch to toplevel
            stackTop.hits = 0; // force mem pop
            primitiveHits = 0;
            Spec::initModel(model, tdg);

            ray.dir = wsdir;
            ray.orig = wsorig;
            Spec::setupRay(ray);
            duplicatedOctant = getDuplicatedOctant(ray.dir);
        }

        // pop
        if (stackTop.hits <= 0x00FFFFFF)
        {
            if (sstackPtr == 0)
                break;
            stackTop = stack[--sstackPtr];
        }
    } // traversal

    // Store result.
    if (hitTriID == -1)
        ray.tmax = -1.0f;
    Spec::storeHit(hits, rayIdx, hitFormat, hitTriID, hitInstIdx, ray.tmax, hitU, hitV);
}

#endif //#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)

//------------------------------------------------------------------------

TracerBVH8Naive::TracerBVH8Naive(void)
{
}

//------------------------------------------------------------------------

TracerBVH8Naive::~TracerBVH8Naive(void)
{
}

//------------------------------------------------------------------------

const char* TracerBVH8Naive::getBackendName(void) const
{
    return "Naive CPU fallback for BVH8";
}

//------------------------------------------------------------------------

Tracer::Layout TracerBVH8Naive::getBVHLayout(void) const // obsolete boilerplate
{
    Layout layout;
    layout.storeOnGPU       = false;
    layout.arrayAlign       = 64;
    layout.reorderTriangles = true;
    layout.optixNodes       = false;
    return layout;
}

//------------------------------------------------------------------------

void TracerBVH8Naive::traceFromHostMem(const TracerData& bvh, const void* rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const unused)
{

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)

    if (bvh.inDeviceMem)
        throw IlwalidValue(RT_EXCEPTION_INFO, "TracerBVH8Naive requires BVH stored in host memory");

    if (dynamic_cast<const TracerDataMeshBVH8*>(&bvh)) // flat
    {
        const TracerDataMeshBVH8* const tdm = dynamic_cast<const TracerDataMeshBVH8*>(&bvh);
        RT_ASSERT(tdm);
        const bool bakedTris = (tdm->numBVH8Triangles != 0);

        if (bakedTris)
            for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
                traceFlat<TraceSpecialization<false, false>>(rayIdx, *tdm, rays, rayFormat, hits, hitFormat, anyHit);

        else if (!watertight)
            for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
                traceFlat<TraceSpecialization<true, false>>(rayIdx, *tdm, rays, rayFormat, hits, hitFormat, anyHit);

        else
            for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
                traceFlat<TraceSpecialization<true, true>>(rayIdx, *tdm, rays, rayFormat, hits, hitFormat, anyHit);
    }
    else // instanced
    {
        const TracerDataGroupBVH8* const tdg = dynamic_cast<const TracerDataGroupBVH8*>(&bvh);
        RT_ASSERT(tdg);

        bool bakedTris = false;
        for (size_t i = 0; i < tdg->bvh8Meshes.size(); i++)
            if (tdg->bvh8Meshes[i].numBVH8Triangles != 0)
                bakedTris = true;

        if (bakedTris)
            for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
                traceInst<TraceSpecialization<false, false>>(rayIdx, *tdg, rays, rayFormat, hits, hitFormat, anyHit);

        else if (!watertight)
            for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
                traceInst<TraceSpecialization<true, false>>(rayIdx, *tdg, rays, rayFormat, hits, hitFormat, anyHit);

        else
            for (int rayIdx = 0; rayIdx < numRays; rayIdx++)
                traceInst<TraceSpecialization<true, true>>(rayIdx, *tdg, rays, rayFormat, hits, hitFormat, anyHit);
    }

#else
    throw prodlib::IlwalidOperation(RT_EXCEPTION_INFO, "TracerBVH8Naive unsupported on this architecture");
#endif

}

//------------------------------------------------------------------------



