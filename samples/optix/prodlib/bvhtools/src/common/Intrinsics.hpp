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
#include <math.h> // float4, make_float4(), etc.
#include <vector_functions.h> // float4, make_float4(), etc.
#include <math_constants.h> // LWDART_MAX_NORMAL_F, etc.
#include <stdint.h>
#include <float.h>

#include <corelib/compiler/WarpLevelIntrinsics.h>

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#include <xmmintrin.h>
#endif

//------------------------------------------------------------------------
// Preprocessor definitions.
//------------------------------------------------------------------------

// Inline function specifier.
#ifdef __LWDACC__
#   define INLINE __device__ __forceinline__
#   define FORCEINLINE __device__ __forceinline__
#else
#   define INLINE inline
 #if defined(__GNUC__) || defined(__clang__)
  #define FORCEINLINE		__inline__ __attribute__((always_inline))
 #elif defined(_WIN32)
  #define FORCEINLINE		__forceinline
 #endif
#endif



// Pointer type in inline PTX.
#if defined(_WIN64) || defined(__LP64__)

    #define PTX_PTRARG "l"
    #define PTX_PTRREG ".b64"
#else
    #define PTX_PTRARG "r"
    #define PTX_PTRREG ".b32"
#endif

// LDG specifier in inline PTX.
#if (__LWDA_ARCH__ >= 350)
    #define PTX_LDG ".nc"
#else
    #define PTX_LDG ""
#endif

//------------------------------------------------------------------------
// Helper macros for defining blocks-per-SM for different GPU architectures.
//
// The name of the macro indicates which GPU architecture the value was
// originally chosen for. The macro is a pass-through when compiling for
// the same arhictecture, and it extrapolates the value semi-reasonably
// when compiling for other architectures.
//------------------------------------------------------------------------

#if !defined(__LWDA_ARCH__) || (__LWDA_ARCH__ >= 500)                       // Compiling for Maxwell (or host)?

#   define NUMBLOCKS_MAXWELL(X) (X)                                         // Maxwell => Maxwell: Pass-through.
#   define NUMBLOCKS_KEPLER(X)  (X)                                         // Kepler  => Maxwell: 1x regs, 2x shmem, 2x CTAs.
#   define NUMBLOCKS_FERMI(X)   ((X) * 2)                                   // Fermi   => Maxwell: 2x regs, 2x shmem, 4x CTAs.

#elif (__LWDA_ARCH__ >= 400)                                                // Compiling for Kepler?

#   define NUMBLOCKS_MAXWELL(X) (((X)/2 < 1) ? 1 : (X)/2)                   // Maxwell => Kepler: 1x regs, 1/2x shmem, 1/2x CTAs.
#   define NUMBLOCKS_KEPLER(X)  (X)                                         // Kepler  => Kepler: Pass-through.
#   define NUMBLOCKS_FERMI(X)   (X)                                         // Fermi   => Kepler: 2x regs, 1x shmem, 2x CTAs.

#else                                                                       // Compiling for Fermi (or Tesla)?

#   define NUMBLOCKS_MAXWELL(X) (((X)/2 < 1) ? 1 : ((X)/2 > 8) ? 8 : (X)/2) // Maxwell => Fermi: 1/2x regs, 1/2x shmem, 1/4x CTAs.
#   define NUMBLOCKS_KEPLER(X)  (((X)/2 < 1) ? 1 : (X)/2)                   // Kepler  => Fermi: 1/2x regs, 1x shmem, 1/2x CTAs.
#   define NUMBLOCKS_FERMI(X)   (X)                                         // Fermi   => Fermi: Pass-through.

#endif

//------------------------------------------------------------------------
// Emulation of standard LWCA intrinsics on the CPU.
//------------------------------------------------------------------------

#ifndef __LWDACC__
#  ifndef BVHTOOLS_NO_LWDA_MATH
static FORCEINLINE int       min             (int a, int b)            { return (a < b) ? a : b; }
static FORCEINLINE int       max             (int a, int b)            { return (a > b) ? a : b; }
static FORCEINLINE uint64_t  min             (uint64_t a, uint64_t b)  { return (a < b) ? a : b; }
static FORCEINLINE uint64_t  max             (uint64_t a, uint64_t b)  { return (a > b) ? a : b; }

// The C99 fminf is more robust against NaNs (both params can be filtered out), but should not be able to be mapped to a single instruction?!
#    if __cplusplus
extern "C" {
#    endif
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
FORCEINLINE float     fminf           (float a, float b)        { return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b))); } // to guarantee NaN filtering for 1st parameter (e.g. if 1st == NaN and 2nd != NaN -> returns 2nd)
FORCEINLINE float     fmaxf           (float a, float b)        { return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b))); } // dto.
#endif
#    if __cplusplus
}
#    endif

#  endif // BVHTOOLS_NO_LWDA_MATH

static FORCEINLINE int           __float_as_int          (float val)         { union { float f; int i; } var; var.f = val; return var.i; }
static FORCEINLINE float         __int_as_float          (int val)           { union { float f; int i; } var; var.i = val; return var.f; }
static FORCEINLINE unsigned int  __float_as_uint         (float val)         { union { float f; unsigned int i; } var; var.f = val; return var.i; }
static FORCEINLINE float         __uint_as_float         (unsigned int val)  { union { float f; unsigned int i; } var; var.i = val; return var.f; }
static FORCEINLINE long long     __double_as_longlong    (double val)        { union { double d; long long ll; } var; var.d = val; return var.ll; }
static FORCEINLINE double        __longlong_as_double    (long long val)     { union { double d; long long ll; } var; var.ll = val; return var.d; }

static INLINE float fstrict(float v) // prevents the compiler from optimizing FP32 math on host side
{
    return __int_as_float(__float_as_int(v));
}

static INLINE float __fsub_rd(float a, float b) // returns a-b rounded down
{
    float x = fstrict(fstrict(a) - fstrict(b));
    if (fstrict(x + fstrict(b)) > fstrict(a) || fstrict(fstrict(a) - x) < fstrict(b))
        if (x > -FLT_MAX && (~__float_as_int(x) & 0x7F800000u) != 0u)
            x = (x == 0.0f) ? -FLT_MIN : __int_as_float(__float_as_int(x) + ((x > 0.0f) ? -1 : +1));
    return x;
}

static INLINE float __fsub_ru(float a, float b) // returns a-b rounded up
{
    float x = fstrict(fstrict(a) - fstrict(b));
    if (fstrict(x + fstrict(b)) < fstrict(a) || fstrict(fstrict(a) - x) > fstrict(b))
        if (x < +FLT_MAX && (~__float_as_int(x) & 0x7F800000u) != 0u)
            x = (x == 0.0f) ? +FLT_MIN : __int_as_float(__float_as_int(x) + ((x > 0.0f) ? +1 : -1));
    return x;
}

static FORCEINLINE int __popc(unsigned int v) // population count
{
    // Adapted from:
    // http://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
    v -= (v >> 1) & 0x55555555u;
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return (((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
}

#endif // __LWDACC__

static FORCEINLINE int clamp( int a, int b, int x )
{
  return max( a, min( b, x ) );
}

// Implement __ldg() as standard memory load on host and on older GPUs.
#if !defined(__LWDACC__) || !defined(__LWDA_ARCH__) || (__LWDA_ARCH__ < 350)
template <class T> static FORCEINLINE T LDG_OR_GLOBAL(const T* ptr) { return *ptr; }
#else
#define LDG_OR_GLOBAL __ldg
#endif

//------------------------------------------------------------------------
// Custom PTX intrinsics, with host-side emulation.
//------------------------------------------------------------------------

// Returns the index of the highest set bit, or -1 if none.
#ifdef __LWDACC__
static FORCEINLINE int   findLeadingOne  (unsigned int v)        { unsigned int r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
#else
static FORCEINLINE int   findLeadingOne  (unsigned int v)        { return (v == 0) ? -1 : ((__double_as_longlong((double)v) >> 52) - 1023); }
#endif

// Returns (c >= 0) ? a : b.
#ifndef __LWDACC__
static FORCEINLINE int   slct      (int a, int b, int c)       { return (c >= 0) ? a : b; }
static FORCEINLINE float slct      (float a, float b, int c)   { return (c >= 0) ? a : b; }
#endif


// Loads/stores that bypass L1.
#ifdef __LWDACC__
static INLINE char      loadCG  (const char* ptr)       { int r; asm("ld.global.cg.s8 %0, [%1];" : "=r"(r) : PTX_PTRARG(ptr)); return (char)r; }
static INLINE int       loadCG  (const int* ptr)        { int r; asm("ld.global.cg.s32 %0, [%1];" : "=r"(r) : PTX_PTRARG(ptr)); return r; }
static INLINE float     loadCG  (const float* ptr)      { float r; asm("ld.global.cg.f32 %0, [%1];" : "=f"(r) : PTX_PTRARG(ptr)); return r; }
static INLINE int2      loadCG  (const int2* ptr)       { int2 r; asm("ld.global.cg.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : PTX_PTRARG(ptr)); return r; }
static INLINE float2    loadCG  (const float2* ptr)     { float2 r; asm("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(r.x), "=f"(r.y) : PTX_PTRARG(ptr)); return r; }
static INLINE float4    loadCG  (const float4* ptr)     { float4 r; asm("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : PTX_PTRARG(ptr)); return r; }
static INLINE void      storeCG (char* ptr, char v)     { asm("st.global.cg.s8 [%0], %1;" :: PTX_PTRARG(ptr), "r"((int)v)); }
static INLINE void      storeCG (float* ptr, float v)   { asm("st.global.cg.f32 [%0], %1;" :: PTX_PTRARG(ptr), "f"(v)); }
static INLINE void      storeCG (float2* ptr, float2 v) { asm("st.global.cg.v2.f32 [%0], {%1, %2};" :: PTX_PTRARG(ptr), "f"(v.x), "f"(v.y)); }
static INLINE void      storeCG (int4* ptr, int4 v)     { asm("st.global.cg.v4.s32 [%0], {%1, %2, %3, %4};" :: PTX_PTRARG(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }
static INLINE void      storeCG (float4* ptr, float4 v) { asm("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};" :: PTX_PTRARG(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)); }

static INLINE char      loadCS  (const char* ptr)       { int r; asm("ld.global.cs.s8 %0, [%1];" : "=r"(r) : PTX_PTRARG(ptr)); return (char)r; }
static INLINE int       loadCS  (const int* ptr)        { int r; asm("ld.global.cs.s32 %0, [%1];" : "=r"(r) : PTX_PTRARG(ptr)); return r; }
static INLINE float     loadCS  (const float* ptr)      { float r; asm("ld.global.cs.f32 %0, [%1];" : "=f"(r) : PTX_PTRARG(ptr)); return r; }
static INLINE int2      loadCS  (const int2* ptr)       { int2 r; asm("ld.global.cs.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : PTX_PTRARG(ptr)); return r; }
static INLINE float2    loadCS  (const float2* ptr)     { float2 r; asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(r.x), "=f"(r.y) : PTX_PTRARG(ptr)); return r; }
static INLINE float4    loadCS  (const float4* ptr)     { float4 r; asm("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : PTX_PTRARG(ptr)); return r; }
static INLINE void      storeCS (char* ptr, char v)     { asm("st.global.cs.s8 [%0], %1;" ::PTX_PTRARG(ptr), "r"((int)v)); }
static INLINE void      storeCS (float* ptr, float v)   { asm("st.global.cs.f32 [%0], %1;" ::PTX_PTRARG(ptr), "f"(v)); }
static INLINE void      storeCS (float2* ptr, float2 v) { asm("st.global.cs.v2.f32 [%0], {%1, %2};" ::PTX_PTRARG(ptr), "f"(v.x), "f"(v.y)); }
static INLINE void      storeCS (int4* ptr, int4 v)     { asm("st.global.cs.v4.s32 [%0], {%1, %2, %3, %4};" ::PTX_PTRARG(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }
static INLINE void      storeCS (float4* ptr, float4 v) { asm("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};" ::PTX_PTRARG(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)); }
#else
template <class T> static FORCEINLINE T      loadCG  (const T* ptr)  { return *ptr; }
template <class T> static FORCEINLINE void   storeCG (T* ptr, T v)   { *ptr = v; }
template <class T> static FORCEINLINE T      loadCS  (const T* ptr)  { return *ptr; }
template <class T> static FORCEINLINE void   storeCS (T* ptr, T v)   { *ptr = v; }
#endif

// Vectorized loads/stores for generic structs.
#ifdef __LWDACC__
template <class T> static INLINE T      loadCachedAlign1        (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs++) *((char*)&v + ofs) = LDG_OR_GLOBAL(((char*)ptr + ofs)); return v; }
template <class T> static INLINE T      loadCachedAlign4        (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs += 4) *(float*)((char*)&v + ofs) = LDG_OR_GLOBAL((float*)((char*)ptr + ofs)); return v; }
template <>               INLINE int    loadCachedAlign4<int>   (const int* ptr)    { return LDG_OR_GLOBAL(ptr); }
template <>               INLINE int3   loadCachedAlign4<int3>  (const int3* ptr)   { return make_int3(LDG_OR_GLOBAL((const int*)ptr), LDG_OR_GLOBAL((const int*)ptr+1), LDG_OR_GLOBAL((const int*)ptr+2)); }
template <>               INLINE float3 loadCachedAlign4<float3>(const float3* ptr) { return make_float3(LDG_OR_GLOBAL((const float*)ptr), LDG_OR_GLOBAL((const float*)ptr+1), LDG_OR_GLOBAL((const float*)ptr+2)); }
template <class T> static INLINE T      loadCachedAlign8        (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs += 8) *(float2*)((char*)&v + ofs) = LDG_OR_GLOBAL((float2*)((char*)ptr + ofs)); return v; }
template <class T> static INLINE T      loadCachedAlign16       (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs += 16) *(float4*)((char*)&v + ofs) = LDG_OR_GLOBAL((float4*)((char*)ptr + ofs)); return v; }
template <>               INLINE float4 loadCachedAlign16<float4>(const float4* ptr){ return LDG_OR_GLOBAL(ptr); }
template <class T> static INLINE T      loadUncachedAlign1      (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs++) *((char*)&v + ofs) = loadCG(((char*)ptr + ofs)); return v; }
template <class T> static INLINE T      loadUncachedAlign4      (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs += 4) *(float*)((char*)&v + ofs) = loadCG((float*)((char*)ptr + ofs)); return v; }
template <class T> static INLINE T      loadUncachedAlign8      (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs += 8) *(float2*)((char*)&v + ofs) = loadCG((float2*)((char*)ptr + ofs)); return v; }
template <class T> static INLINE T      loadUncachedAlign16     (const T* ptr)      { T v; for (int ofs = 0; ofs < sizeof(T); ofs += 16) *(float4*)((char*)&v + ofs) = loadCG((float4*)((char*)ptr + ofs)); return v; }
template <>               INLINE float4 loadUncachedAlign16<float4>(const float4* ptr){ return loadCG(ptr); }
template <class T> static INLINE void   storeCachedAlign1       (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs++) *((char*)ptr + ofs) = *((char*)&v + ofs); }
template <class T> static INLINE void   storeCachedAlign4       (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs += 4) *(float*)((char*)ptr + ofs) = *(float*)((char*)&v + ofs); }
template <class T> static INLINE void   storeCachedAlign8       (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs += 8) *(float2*)((char*)ptr + ofs) = *(float2*)((char*)&v + ofs); }
template <class T> static INLINE void   storeCachedAlign16      (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs += 16) *(float4*)((char*)ptr + ofs) = *(float4*)((char*)&v + ofs); }
template <class T> static INLINE void   storeUncachedAlign1     (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs++) storeCG(((char*)ptr + ofs), *((char*)&v + ofs)); }
template <class T> static INLINE void   storeUncachedAlign4     (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs += 4) storeCG((float*)((char*)ptr + ofs), *(float*)((char*)&v + ofs)); }
template <>               INLINE void   storeUncachedAlign4<float>(float* ptr, float v){ return storeCG(ptr, v); }
template <class T> static INLINE void   storeUncachedAlign8     (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs += 8) storeCG((float2*)((char*)ptr + ofs), *(float2*)((char*)&v + ofs)); }
template <>               INLINE void   storeUncachedAlign8<float2>(float2* ptr, float2 v){ return storeCG(ptr, v); }
template <class T> static INLINE void   storeUncachedAlign16    (T* ptr, T v)       { for (int ofs = 0; ofs < sizeof(T); ofs += 16) storeCG((float4*)((char*)ptr + ofs), *(float4*)((char*)&v + ofs)); }
template <>               INLINE void   storeUncachedAlign16<float4>(float4* ptr, float4 v){ return storeCG(ptr,v); }
#else
template <class T> static FORCEINLINE T      loadCachedAlign1        (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE T      loadCachedAlign4        (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE T      loadCachedAlign8        (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE T      loadCachedAlign16       (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE T      loadUncachedAlign1      (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE T      loadUncachedAlign4      (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE T      loadUncachedAlign8      (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE T      loadUncachedAlign16     (const T* ptr)      { return *ptr; }
template <class T> static FORCEINLINE void   storeCachedAlign1       (T* ptr, T v)       { *ptr = v; }
template <class T> static FORCEINLINE void   storeCachedAlign4       (T* ptr, T v)       { *ptr = v; }
template <class T> static FORCEINLINE void   storeCachedAlign8       (T* ptr, T v)       { *ptr = v; }
template <class T> static FORCEINLINE void   storeCachedAlign16      (T* ptr, T v)       { *ptr = v; }
template <class T> static FORCEINLINE void   storeUncachedAlign1     (T* ptr, T v)       { *ptr = v; }
template <class T> static FORCEINLINE void   storeUncachedAlign4     (T* ptr, T v)       { *ptr = v; }
template <class T> static FORCEINLINE void   storeUncachedAlign8     (T* ptr, T v)       { *ptr = v; }
template <class T> static FORCEINLINE void   storeUncachedAlign16    (T* ptr, T v)       { *ptr = v; }
#endif

//------------------------------------------------------------------------
// Device-side PTX instrinsics.
//------------------------------------------------------------------------

#ifdef __LWDACC__

// Count trailing zero bits.
static INLINE int           countTrailingZeros  (unsigned int v)                            { return __clz(__brev(v)); }

// Warp lane masks.
static INLINE unsigned int  getLaneMaskLt       (void)                                      { unsigned int r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }
static INLINE unsigned int  getLaneMaskLe       (void)                                      { unsigned int r; asm("mov.u32 %0, %lanemask_le;" : "=r"(r)); return r; }
static INLINE unsigned int  getLaneMaskGt       (void)                                      { unsigned int r; asm("mov.u32 %0, %lanemask_gt;" : "=r"(r)); return r; }
static INLINE unsigned int  getLaneMaskGe       (void)                                      { unsigned int r; asm("mov.u32 %0, %lanemask_ge;" : "=r"(r)); return r; }

// Funnel shift.
#if (__LWDA_ARCH__ >= 320)
static INLINE unsigned int  shf_r_clamp         (unsigned int lo, unsigned int hi, int sh)  { unsigned int r; asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(sh)); return r; }
static INLINE unsigned int  shf_r_wrap          (unsigned int lo, unsigned int hi, int sh)  { unsigned int r; asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(sh)); return r; }
#else
static INLINE unsigned int  shf_r_clamp         (unsigned int lo, unsigned int hi, int sh)  { return (sh <= 0) ? lo : (sh >= 32) ? hi : __double2loint(__longlong_as_double(__double_as_longlong(__hiloint2double(hi, lo)) >> sh)); }
static INLINE unsigned int  shf_r_wrap          (unsigned int lo, unsigned int hi, int sh)  { return __double2loint(__longlong_as_double(__double_as_longlong(__hiloint2double(hi, lo)) >> (sh & 31))); }
#endif

#if (__LWDA_ARCH__ >= 320)
static INLINE unsigned int  shf_l_clamp         (unsigned int lo, unsigned int hi, int sh)  { unsigned int r; asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(sh)); return r; }
static INLINE unsigned int  shf_l_wrap          (unsigned int lo, unsigned int hi, int sh)  { unsigned int r; asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(sh)); return r; }
#else
static INLINE unsigned int  shf_l_clamp         (unsigned int lo, unsigned int hi, int sh)  { unsigned n = min(sh, 32); return (hi << n) | (lo >> (32-n)); }
static INLINE unsigned int  shf_l_wrap          (unsigned int lo, unsigned int hi, int sh)  { unsigned n = sh & 0x1f;   return (hi << n) | (lo >> (32-n)); }
#endif

// Bitfield extract/insert. On Volta, should use shifts and ands instead
static INLINE unsigned int  bfe                 (unsigned int val, int pos, int len)                    { unsigned int r; asm("bfe.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(val), "r"(pos), "r"(len)); return r; }
static INLINE unsigned int  bfi                 (unsigned int src, unsigned int dst, int pos, int len)  { unsigned int r; asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(src), "r"(dst), "r"(pos), "r"(len)); return r; }

// Extract byte and shift it left.
static INLINE unsigned int  vshl_clamp_b0       (unsigned int val, unsigned int shift)      { unsigned int r; asm("vshl.u32.u32.u32.clamp %0, %1.b0, %2;" : "=r"(r) : "r"(val), "r"(shift)); return r;}
static INLINE unsigned int  vshl_clamp_b1       (unsigned int val, unsigned int shift)      { unsigned int r; asm("vshl.u32.u32.u32.clamp %0, %1.b1, %2;" : "=r"(r) : "r"(val), "r"(shift)); return r;}
static INLINE unsigned int  vshl_clamp_b2       (unsigned int val, unsigned int shift)      { unsigned int r; asm("vshl.u32.u32.u32.clamp %0, %1.b2, %2;" : "=r"(r) : "r"(val), "r"(shift)); return r;}
static INLINE unsigned int  vshl_clamp_b3       (unsigned int val, unsigned int shift)      { unsigned int r; asm("vshl.u32.u32.u32.clamp %0, %1.b3, %2;" : "=r"(r) : "r"(val), "r"(shift)); return r;}

// Extract byte value, shift it left with another extracted byte and add the result to u32.
static INLINE unsigned int vshl_wrap_add_b0_b0(unsigned int val, unsigned int shift, unsigned int addend)
{
    unsigned int r;
    asm("vshl.u32.u32.u32.wrap.add %0, %1.b0, %2.b0, %3;" : "=r"(r) : "r"(val), "r"(shift), "r"(addend));
    return r;
}
static INLINE unsigned int vshl_wrap_add_b1_b1(unsigned int val, unsigned int shift, unsigned int addend)
{
    unsigned int r;
    asm("vshl.u32.u32.u32.wrap.add %0, %1.b1, %2.b1, %3;" : "=r"(r) : "r"(val), "r"(shift), "r"(addend));
    return r;
}
static INLINE unsigned int vshl_wrap_add_b2_b2(unsigned int val, unsigned int shift, unsigned int addend)
{
    unsigned int r;
    asm("vshl.u32.u32.u32.wrap.add %0, %1.b2, %2.b2, %3;" : "=r"(r) : "r"(val), "r"(shift), "r"(addend));
    return r;
}
static INLINE unsigned int vshl_wrap_add_b3_b3(unsigned int val, unsigned int shift, unsigned int addend)
{
    unsigned int r;
    asm("vshl.u32.u32.u32.wrap.add %0, %1.b3, %2.b3, %3;" : "=r"(r) : "r"(val), "r"(shift), "r"(addend));
    return r;
}

static INLINE unsigned int prmt(unsigned int indices, unsigned int valuesLo, unsigned int valuesHi)
{
    unsigned int r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(valuesLo), "r"(valuesHi), "r"(indices));
    return r;
}

// (c >= 0) ? a : b
static INLINE int   slct                        (int a, int b, int c)                       { int v; asm("slct.s32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
static INLINE float slct                        (float a, float b, int c)                   { float v; asm("slct.f32.s32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "r"(c)); return v; }
static INLINE float slct                        (float a, float b, float c)                 { float v; asm( "slct.f32.f32 %0, %1, %2, %3;" : "=f"( v ) : "f"( a ), "f"( b ), "f"( c ) ); return v; }


// Replicate sign bits of 4 packed signed bytes.
static INLINE unsigned int  vsignExtend4        (unsigned int bytes)                        { unsigned int r; asm("prmt.b32 %0, %1, %1, 8 | (9 << 4) | (10 << 8) | (11 << 12);" : "=r"(r) : "r"(bytes)); return r; }
static INLINE unsigned int  duplicateByte       (unsigned char value)                       { unsigned int r; asm("prmt.b32 %0, %1, %1, 0;" : "=r"(r) : "r"((unsigned int)value)); return r; }
static INLINE unsigned int  duplicateByte       (unsigned int value)                        { unsigned int r; asm("prmt.b32 %0, %1, %1, 0;" : "=r"(r) : "r"(value)); return r; }

// Shared memory operations using 32-bit index to shared memory.
static INLINE void sts4                         (int sharedPtr, int value)                  { asm("st.shared.s32 [%0], %1;" ::"r"(sharedPtr), "r"(value)); }
static INLINE void sts8                         (int sharedPtr, int2 value)                 { asm("st.shared.v2.s32 [%0], {%1, %2};" ::"r"(sharedPtr), "r"(value.x), "r"(value.y)); }
static INLINE int  lds4                         (int sharedPtr)                             { int r; asm("ld.shared.s32 %0, [%1];" : "=r"(r) : "r"(sharedPtr)); return r; }
static INLINE int2 lds8                         (int sharedPtr)                             { int2 r; asm("ld.shared.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "r"(sharedPtr)); return r; }

#endif

//------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------

// Operations on the LSB of a float.
static FORCEINLINE float     setLSB      (float v)   { return __int_as_float(__float_as_int(v) | 1); }
static FORCEINLINE float     clearLSB    (float v)   { return __int_as_float(__float_as_int(v) & ~1); }
static FORCEINLINE int       getLSB      (float v)   { return __float_as_int(v) & 1; }

// Vectors.
static FORCEINLINE float3    make_float3     (const float3& v)                       { return v; }
static FORCEINLINE float3    make_float3     (const float4& v)                       { return make_float3(v.x, v.y, v.z); }
static FORCEINLINE float3    min             (const float3& a, const float3& b)      { return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
static FORCEINLINE float3    max             (const float3& a, const float3& b)      { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
static FORCEINLINE int       chooseComponent (int v0, int v1, int v2, int idx)       { return (idx == 0) ? v0 : (idx == 1) ? v1 : v2; }
static FORCEINLINE float     chooseComponent (float v0, float v1, float v2, int idx) { return (idx == 0) ? v0 : (idx == 1) ? v1 : v2; }
static FORCEINLINE float     chooseComponent (const float3& v, int idx)              { return (idx == 0) ? v.x : (idx == 1) ? v.y : v.z; }
static FORCEINLINE float     aabbHalfArea    (float sizeX, float sizeY, float sizeZ) { return sizeX * sizeY + sizeY * sizeZ + sizeZ * sizeX; }

//------------------------------------------------------------------------
// Warp-shuffle.
//------------------------------------------------------------------------

#ifdef __LWDACC__

#define INIT_SHUFFLE_EMULATION(WARPS_PER_BLOCK)

template <class T> static INLINE T  shfl            (T val, int lane)   { T res; for (int ofs = 0; ofs < sizeof(T); ofs += sizeof(int)) *(int*)((char*)&res + ofs) = __shfl_nosync(*(int*)((char*)&val + ofs), lane); return res; }
template <class T> static INLINE T  shfl_xor        (T val, int lane)   { T res; for (int ofs = 0; ofs < sizeof(T); ofs += sizeof(int)) *(int*)((char*)&res + ofs) = __shfl_xor_nosync(*(int*)((char*)&val + ofs), lane); return res; }
template <class T> static INLINE T  shfl_up         (T val, int lane)   { T res; for (int ofs = 0; ofs < sizeof(T); ofs += sizeof(int)) *(int*)((char*)&res + ofs) = __shfl_up_nosync(*(int*)((char*)&val + ofs), lane); return res; }

static INLINE int                   shfl_up_add     (int val, int lane) { asm("{ .reg .s32 t; .reg .pred p; shfl.up.b32 t|p, %0, %1, 0x2000; @p add.s32 %0, %0, t; }" : "+r"(val) : "r"(lane)); return val; }
static INLINE int                   shfl_up_add16   (int val, int lane) { asm("{ .reg .s32 t; .reg .pred p; shfl.up.b32 t|p, %0, %1, 0x1000; @p add.s32 %0, %0, t; }" : "+r"(val) : "r"(lane)); return val; }
static INLINE int                   shfl_up_min16   (int val, int lane) { asm("{ .reg .s32 t; .reg .pred p; shfl.up.b32 t|p, %0, %1, 0x1000; @p min.s32 %0, %0, t; }" : "+r"(val) : "r"(lane)); return val; }

#endif // __LWDACC__

#if 0
// safe aligned load version, but usually also not faster than a plain 3xint load
#if 1
static FORCEINLINE int3 fastUnalignedLoadInt3(const void* ptr)
{
    int3 r;
    asm("{\n"
        ".reg .pred                       p0;\n"
        ".reg "   PTX_PTRREG "            off;\n"
        "and"     PTX_PTRREG "            off, %3, 7;\n"
        "setp.eq" PTX_PTRREG "            p0, off, 0;\n"
        "@p0  ld.global" PTX_LDG ".v2.b32  {%0,%1},      [%3+0];\n"
        "@p0  ld.global" PTX_LDG ".b32     %2,           [%3+8];\n"
        "@!p0 ld.global" PTX_LDG ".b32     %0,           [%3+0];\n"
        "@!p0 ld.global" PTX_LDG ".v2.b32  {%1,%2},      [%3+4];\n"
        "}"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z) : PTX_PTRARG(ptr));
    return r;
}
#else
// "wrong" aligned load version (must assume one more element at the end if size not matching alignment and maybe even at the beginning if first vertex not properly aligned)
static FORCEINLINE int3 fastUnalignedLoadInt3(const void* ptr)
{
    int3 r;
    asm("{\n"
        ".reg .pred                       p<4>;\n"
        ".reg "   PTX_PTRREG "            off;\n"
        "and"     PTX_PTRREG "            off, %3, 15;\n"
        "setp.eq" PTX_PTRREG "            p0, off, 0;\n"
        "setp.eq" PTX_PTRREG "            p1, off, 4;\n"
        "setp.eq" PTX_PTRREG "            p2, off, 8;\n"
        "setp.eq" PTX_PTRREG "            p3, off, 2;\n"
        "@p0 ld.global" PTX_LDG ".v4.b32  {%0,%1,%2,_}, [%3+0];\n"
        "@p1 ld.global" PTX_LDG ".v4.b32  {_,%0,%1,%2}, [%3+-4];\n"
        "@p2 ld.global" PTX_LDG ".v2.b32  {%0,%1},      [%3+0];\n"
        "@p2 ld.global" PTX_LDG ".b32     %2,           [%3+8];\n"
        "@p3 ld.global" PTX_LDG ".b32     %0,           [%3+0];\n"
        "@p3 ld.global" PTX_LDG ".v2.b32  {%1,%2},      [%3+4];\n"
        "}"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z) : PTX_PTRARG(ptr));
    return r;
}
#endif
static FORCEINLINE float3 fastUnalignedLoadFloat3(const void* ptr)
{
    const int3 r = fastUnalignedLoadInt3(ptr);
    return make_float3(__int_as_float(r.x), __int_as_float(r.y), __int_as_float(r.z));
}
#else
static FORCEINLINE int3   fastUnalignedLoadInt3(const void* ptr)   { return *(const int3*)ptr; }
static FORCEINLINE float3 fastUnalignedLoadFloat3(const void* ptr) { return *(const float3*)ptr; }
#endif

//------------------------------------------------------------------------------

#ifdef __LWDACC__
static FORCEINLINE void loadCachedFloat6( const void* p, float& f0, float& f1, float& f2, float& f3, float& f4, float& f5 )
{
  asm ("{ \n\t"
    ".reg .pred p<4>; \n\t"
    ".reg"    PTX_PTRREG "            off; \n\t"
    "and"     PTX_PTRREG "            off,  %6, 15;\n\t"
    "setp.eq" PTX_PTRREG "            p0, off, 0;\n\t"
    "@p0  ld.global" PTX_LDG ".v4.f32 {%0,%1,%2,%3}, [%6+0];\n\t"
    "@p0  ld.global" PTX_LDG ".v2.f32       {%4,%5}, [%6+16];\n\t"
    "@!p0 ld.global" PTX_LDG ".v2.f32       {%0,%1}, [%6+0];\n\t"
    "@!p0 ld.global" PTX_LDG ".v4.f32 {%2,%3,%4,%5}, [%6+8];\n\t"
    "}"
    :"=f"(f0),"=f"(f1),"=f"(f2),"=f"(f3),"=f"(f4),"=f"(f5) : PTX_PTRARG(p) : );
}
static FORCEINLINE void loadUncachedFloat6(const void* p, float& f0, float& f1, float& f2, float& f3, float& f4, float& f5)
{
    asm("{ \n\t"
        ".reg .pred p<4>; \n\t"
        ".reg"    PTX_PTRREG "            off; \n\t"
        "and"     PTX_PTRREG "            off,  %6, 15;\n\t"
        "setp.eq" PTX_PTRREG "            p0, off, 0;\n\t"
        "@p0  ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%6+0];\n\t"
        "@p0  ld.global.cg.v2.f32       {%4,%5}, [%6+16];\n\t"
        "@!p0 ld.global.cg.v2.f32       {%0,%1}, [%6+0];\n\t"
        "@!p0 ld.global.cg.v4.f32 {%2,%3,%4,%5}, [%6+8];\n\t"
        "}"
        :"=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3), "=f"(f4), "=f"(f5) : PTX_PTRARG(p) : );
}
static FORCEINLINE void loadCSFloat6(const void* p, float& f0, float& f1, float& f2, float& f3, float& f4, float& f5)
{
    asm("{ \n\t"
        ".reg .pred p<4>; \n\t"
        ".reg"    PTX_PTRREG "            off; \n\t"
        "and"     PTX_PTRREG "            off,  %6, 15;\n\t"
        "setp.eq" PTX_PTRREG "            p0, off, 0;\n\t"
        "@p0  ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%6+0];\n\t"
        "@p0  ld.global.cs.v2.f32       {%4,%5}, [%6+16];\n\t"
        "@!p0 ld.global.cs.v2.f32       {%0,%1}, [%6+0];\n\t"
        "@!p0 ld.global.cs.v4.f32 {%2,%3,%4,%5}, [%6+8];\n\t"
        "}"
        :"=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3), "=f"(f4), "=f"(f5) : PTX_PTRARG(p) : );
}
#else
static FORCEINLINE void loadCachedFloat6( const void* p, float& f0, float& f1, float& f2, float& f3, float& f4, float& f5 )
{
  const float* const pFloat = (float*)p;
  f0 = pFloat[0];
  f1 = pFloat[1];
  f2 = pFloat[2];
  f3 = pFloat[3];
  f4 = pFloat[4];
  f5 = pFloat[5];
}
#define loadUncachedFloat6 loadCachedFloat6
#define loadCSFloat6 loadCachedFloat6
#endif

//------------------------------------------------------------------------
// Utilities and PTX intrinsics for sm_50.
//------------------------------------------------------------------------

#if (__LWDA_ARCH__ >= 500)

// According to mask, select set bits from a and unset bits from b.
static INLINE unsigned int  bitSelect       (unsigned int a, unsigned int b, unsigned int mask)     { unsigned int r; asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(r) : "r"(a), "r"(b), "r"(mask)); return r; }
static INLINE float         bitSelect       (float a, float b, float mask)                          { return __uint_as_float(bitSelect(__float_as_uint(a), __float_as_uint(b), __float_as_uint(mask))); }

// For 32 bits in parallel, set result bit if all 3 inputs bits are same (all 0 or all 1).
static INLINE unsigned int  bitTest3        (unsigned int a, unsigned int b, unsigned int c)        { unsigned int r; asm("lop3.b32 %0, %1, %2, %3, 0x81;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }

#else // __LWDA_ARCH__

// According to mask, select set bits from a and unset bits from b.
static FORCEINLINE unsigned int  bitSelect       (unsigned int a, unsigned int b, unsigned int mask)     { return (a & mask) | (b & ~mask); }
static FORCEINLINE float         bitSelect       (float a, float b, float mask)                          { return __uint_as_float(bitSelect(__float_as_uint(a), __float_as_uint(b), __float_as_uint(mask))); }

// For 32 bits in parallel, set result bit if all 3 inputs bits are same (all 0 or all 1).
static FORCEINLINE unsigned int  bitTest3        (unsigned int a, unsigned int b, unsigned int c)        { return (a & b & c) | (~a & ~b & ~c); }

#endif // __LWDA_ARCH__

//------------------------------------------------------------------------
