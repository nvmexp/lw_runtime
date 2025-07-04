//-------------------------------------------------------------------------------------
// DirectXMath.h -- SIMD C++ Math library
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//  
// Copyright (c) Microsoft Corporation. All rights reserved.
//-------------------------------------------------------------------------------------

#ifdef _MSC_VER
#pragma once
#endif

#ifndef __cplusplus
#error DirectX Math requires C++
#endif

#define DIRECTX_MATH_VERSION 308


#if defined(_MSC_VER) && !defined(_M_ARM) && !defined(_M_ARM64) && (!_MANAGED) && (!_M_CEE) && (!defined(_M_IX86_FP) || (_M_IX86_FP > 1)) && !defined(_XM_NO_INTRINSICS_) && !defined(_XM_VECTORCALL_)
#if ((_MSC_FULL_VER >= 170065501) && (_MSC_VER < 1800)) || (_MSC_FULL_VER >= 180020418)
#define _XM_VECTORCALL_ 1
#endif
#endif

#if _XM_VECTORCALL_
#define XM_CALLCOLW __vectorcall
#else
#define XM_CALLCOLW __fastcall
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1800)
#define XM_CTOR_DEFAULT {}
#else
#define XM_CTOR_DEFAULT =default;
#endif



#if !defined(_XM_ARM_NEON_INTRINSICS_) && !defined(_XM_SSE_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)
#if defined(_M_IX86) || defined(_M_X64)
#define _XM_SSE_INTRINSICS_
#elif defined(_M_ARM) || defined(_M_ARM64)
#define _XM_ARM_NEON_INTRINSICS_
#elif !defined(_XM_NO_INTRINSICS_)
#error DirectX Math does not support this target
#endif
#endif // !_XM_ARM_NEON_INTRINSICS_ && !_XM_SSE_INTRINSICS_ && !_XM_NO_INTRINSICS_

#pragma warning(push)
#pragma warning(disable:4514 4820 4985)
#include <math.h>
#include <float.h>
#include <malloc.h>
#pragma warning(pop)


#if defined(_XM_SSE_INTRINSICS_)
#ifndef _XM_NO_INTRINSICS_
#include <xmmintrin.h>
#include <emmintrin.h>
#endif
#elif defined(_XM_ARM_NEON_INTRINSICS_)
#ifndef _XM_NO_INTRINSICS_
#pragma warning(push)
#pragma warning(disable : 4987)
#include <intrin.h>
#pragma warning(pop)
#ifdef _M_ARM64
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#endif
#endif



#include <sal.h>
#include <assert.h>

#ifndef _XM_NO_ROUNDF_
#ifdef _MSC_VER
#include <yvals.h>
#if defined(_CPPLIB_VER) && ( _CPPLIB_VER < 610 )
#define _XM_NO_ROUNDF_
#endif
#endif
#endif

#pragma warning(push)
#pragma warning(disable : 4005 4668)
#include <stdint.h>
#pragma warning(pop)

/****************************************************************************
 *
 * Conditional intrinsics
 *
 ****************************************************************************/


#if defined(_XM_ARM_NEON_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)

#if defined(_MSC_VER) && (_MSC_FULL_VER != 170051221) && (_MSC_FULL_VER < 170065500)
#define XM_VMULQ_N_F32( a, b ) vmulq_f32( (a), vdupq_n_f32( (b) ) )
#define XM_VMLAQ_N_F32( a, b, c ) vmlaq_f32( (a), (b), vdupq_n_f32( (c) ) )
#define XM_VMULQ_LANE_F32( a, b, c ) vmulq_f32( (a), vdupq_lane_f32( (b), (c) ) )
#define XM_VMLAQ_LANE_F32( a, b, c, d ) vmlaq_f32( (a), (b), vdupq_lane_f32( (c), (d) ) )
#else
#define XM_VMULQ_N_F32( a, b ) vmulq_n_f32( (a), (b) )
#define XM_VMLAQ_N_F32( a, b, c ) vmlaq_n_f32( (a), (b), (c) )
#define XM_VMULQ_LANE_F32( a, b, c ) vmulq_lane_f32( (a), (b), (c) )
#define XM_VMLAQ_LANE_F32( a, b, c, d ) vmlaq_lane_f32( (a), (b), (c), (d) )
#endif

#endif // _XM_ARM_NEON_INTRINSICS_ && !_XM_NO_INTRINSICS_

#if defined(_XM_SSE_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)

#if defined(_XM_NO_MOVNT_)
#define XM_STREAM_PS( p, a ) _mm_store_ps( p, a )
#define XM_SFENCE()
#else
#define XM_STREAM_PS( p, a ) _mm_stream_ps( p, a )
#define XM_SFENCE() _mm_sfence()
#endif

#define XM_PERMUTE_PS( v, c ) _mm_shuffle_ps( v, v, c )

#endif // _XM_SSE_INTRINSICS_ && !_XM_NO_INTRINSICS_

namespace DirectX
{

/****************************************************************************
 *
 * Constant definitions
 *
 ****************************************************************************/

#if defined(__XNAMATH_H__) && defined(XM_PI)
#undef XM_PI
#undef XM_2PI
#undef XM_1DIVPI
#undef XM_1DIV2PI
#undef XM_PIDIV2
#undef XM_PIDIV4
#undef XM_SELECT_0
#undef XM_SELECT_1
#undef XM_PERMUTE_0X
#undef XM_PERMUTE_0Y
#undef XM_PERMUTE_0Z
#undef XM_PERMUTE_0W
#undef XM_PERMUTE_1X
#undef XM_PERMUTE_1Y
#undef XM_PERMUTE_1Z
#undef XM_PERMUTE_1W
#undef XM_CRMASK_CR6
#undef XM_CRMASK_CR6TRUE
#undef XM_CRMASK_CR6FALSE
#undef XM_CRMASK_CR6BOUNDS
#undef XM_CACHE_LINE_SIZE
#endif

const float XM_PI           = 3.141592654f;
const float XM_2PI          = 6.283185307f;
const float XM_1DIVPI       = 0.318309886f;
const float XM_1DIV2PI      = 0.159154943f;
const float XM_PIDIV2       = 1.570796327f;
const float XM_PIDIV4       = 0.785398163f;

const uint32_t XM_SELECT_0          = 0x00000000;
const uint32_t XM_SELECT_1          = 0xFFFFFFFF;

const uint32_t XM_PERMUTE_0X        = 0;
const uint32_t XM_PERMUTE_0Y        = 1;
const uint32_t XM_PERMUTE_0Z        = 2;
const uint32_t XM_PERMUTE_0W        = 3;
const uint32_t XM_PERMUTE_1X        = 4;
const uint32_t XM_PERMUTE_1Y        = 5;
const uint32_t XM_PERMUTE_1Z        = 6;
const uint32_t XM_PERMUTE_1W        = 7;

const uint32_t XM_SWIZZLE_X         = 0;
const uint32_t XM_SWIZZLE_Y         = 1;
const uint32_t XM_SWIZZLE_Z         = 2;
const uint32_t XM_SWIZZLE_W         = 3;

const uint32_t XM_CRMASK_CR6        = 0x000000F0;
const uint32_t XM_CRMASK_CR6TRUE    = 0x00000080;
const uint32_t XM_CRMASK_CR6FALSE   = 0x00000020;
const uint32_t XM_CRMASK_CR6BOUNDS  = XM_CRMASK_CR6FALSE;


const size_t XM_CACHE_LINE_SIZE = 64;

/****************************************************************************
 *
 * Macros
 *
 ****************************************************************************/

#if defined(__XNAMATH_H__) && defined(XMComparisonAllTrue)
#undef XMComparisonAllTrue
#undef XMComparisonAnyTrue
#undef XMComparisonAllFalse
#undef XMComparisonAnyFalse
#undef XMComparisonMixed
#undef XMComparisonAllInBounds
#undef XMComparisonAnyOutOfBounds
#endif

// Unit colwersion

inline float XMColwertToRadians(float fDegrees) { return fDegrees * (XM_PI / 180.0f); }
inline float XMColwertToDegrees(float fRadians) { return fRadians * (180.0f / XM_PI); }

// Condition register evaluation proceeding a recording (R) comparison

inline bool XMComparisonAllTrue(uint32_t CR) { return (((CR) & XM_CRMASK_CR6TRUE) == XM_CRMASK_CR6TRUE); }
inline bool XMComparisonAnyTrue(uint32_t CR) { return (((CR) & XM_CRMASK_CR6FALSE) != XM_CRMASK_CR6FALSE); }
inline bool XMComparisonAllFalse(uint32_t CR) { return (((CR) & XM_CRMASK_CR6FALSE) == XM_CRMASK_CR6FALSE); }
inline bool XMComparisonAnyFalse(uint32_t CR) { return (((CR) & XM_CRMASK_CR6TRUE) != XM_CRMASK_CR6TRUE); }
inline bool XMComparisonMixed(uint32_t CR) { return (((CR) & XM_CRMASK_CR6) == 0); }
inline bool XMComparisonAllInBounds(uint32_t CR) { return (((CR) & XM_CRMASK_CR6BOUNDS) == XM_CRMASK_CR6BOUNDS); }
inline bool XMComparisonAnyOutOfBounds(uint32_t CR) { return (((CR) & XM_CRMASK_CR6BOUNDS) != XM_CRMASK_CR6BOUNDS); }


/****************************************************************************
 *
 * Data types
 *
 ****************************************************************************/

#pragma warning(push)
#pragma warning(disable:4068 4201 4365 4324 4820)

#pragma prefast(push)
#pragma prefast(disable : 25000, "FXMVECTOR is 16 bytes")

//------------------------------------------------------------------------------
#if defined(_XM_NO_INTRINSICS_)
// The __vector4 structure is an intrinsic on Xbox but must be separately defined
// for x86/x64
struct __vector4
{
    union
    {
        float       vector4_f32[4];
        uint32_t    vector4_u32[4];
    };
};
#endif // _XM_NO_INTRINSICS_

//------------------------------------------------------------------------------
#ifdef _XM_NO_INTRINSICS_
typedef uint32_t __vector4i[4];
#else
typedef __declspec(align(16)) uint32_t __vector4i[4];
#endif

//------------------------------------------------------------------------------
// Vector intrinsic: Four 32 bit floating point components aligned on a 16 byte 
// boundary and mapped to hardware vector registers
#if defined(_XM_SSE_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)
typedef __m128 XMVECTOR;
#elif defined(_XM_ARM_NEON_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)
typedef float32x4_t XMVECTOR;
#else
typedef __vector4 XMVECTOR;
#endif

// Fix-up for (1st-3rd) XMVECTOR parameters that are pass-in-register for x86, ARM, ARM64, and vector call; by reference otherwise
#if ( defined(_M_IX86) || defined(_M_ARM) || defined(_M_ARM64) || _XM_VECTORCALL_ ) && !defined(_XM_NO_INTRINSICS_)
typedef const XMVECTOR FXMVECTOR;
#else
typedef const XMVECTOR& FXMVECTOR;
#endif

// Fix-up for (4th) XMVECTOR parameter to pass in-register for ARM, ARM64, and x64 vector call; by reference otherwise
#if ( defined(_M_ARM) || defined(_M_ARM64) || (_XM_VECTORCALL_ && !defined(_M_IX86) ) ) && !defined(_XM_NO_INTRINSICS_)
typedef const XMVECTOR GXMVECTOR;
#else
typedef const XMVECTOR& GXMVECTOR;
#endif

// Fix-up for (5th & 6th) XMVECTOR parameter to pass in-register for ARM64 and vector call; by reference otherwise
#if ( defined(_M_ARM64) || _XM_VECTORCALL_ ) && !defined(_XM_NO_INTRINSICS_)
typedef const XMVECTOR HXMVECTOR;
#else
typedef const XMVECTOR& HXMVECTOR;
#endif

// Fix-up for (7th+) XMVECTOR parameters to pass by reference
typedef const XMVECTOR& CXMVECTOR;

//------------------------------------------------------------------------------
// Colwersion types for constants
__declspec(align(16)) struct XMVECTORF32
{
    union
    {
        float f[4];
        XMVECTOR v;
    };

    inline operator XMVECTOR() const { return v; }
    inline operator const float*() const { return f; }
#if !defined(_XM_NO_INTRINSICS_) && defined(_XM_SSE_INTRINSICS_)
    inline operator __m128i() const { return _mm_castps_si128(v); }
    inline operator __m128d() const { return _mm_castps_pd(v); }
#endif
};

__declspec(align(16)) struct XMVECTORI32
{
    union
    {
        int32_t i[4];
        XMVECTOR v;
    };

    inline operator XMVECTOR() const { return v; }
#if !defined(_XM_NO_INTRINSICS_) && defined(_XM_SSE_INTRINSICS_)
    inline operator __m128i() const { return _mm_castps_si128(v); }
    inline operator __m128d() const { return _mm_castps_pd(v); }
#endif
};

__declspec(align(16)) struct XMVECTORU8
{
    union
    {
        uint8_t u[16];
        XMVECTOR v;
    };

    inline operator XMVECTOR() const { return v; }
#if !defined(_XM_NO_INTRINSICS_) && defined(_XM_SSE_INTRINSICS_)
    inline operator __m128i() const { return _mm_castps_si128(v); }
    inline operator __m128d() const { return _mm_castps_pd(v); }
#endif
};

__declspec(align(16)) struct XMVECTORU32
{
    union
    {
        uint32_t u[4];
        XMVECTOR v;
    };

    inline operator XMVECTOR() const { return v; }
#if !defined(_XM_NO_INTRINSICS_) && defined(_XM_SSE_INTRINSICS_)
    inline operator __m128i() const { return _mm_castps_si128(v); }
    inline operator __m128d() const { return _mm_castps_pd(v); }
#endif
};

//------------------------------------------------------------------------------
// Vector operators
XMVECTOR    XM_CALLCOLW     operator+ (FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     operator- (FXMVECTOR V);

XMVECTOR&   XM_CALLCOLW     operator+= (XMVECTOR& V1, FXMVECTOR V2);
XMVECTOR&   XM_CALLCOLW     operator-= (XMVECTOR& V1, FXMVECTOR V2);
XMVECTOR&   XM_CALLCOLW     operator*= (XMVECTOR& V1, FXMVECTOR V2);
XMVECTOR&   XM_CALLCOLW     operator/= (XMVECTOR& V1, FXMVECTOR V2);

XMVECTOR&   operator*= (XMVECTOR& V, float S);
XMVECTOR&   operator/= (XMVECTOR& V, float S);

XMVECTOR    XM_CALLCOLW     operator+ (FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     operator- (FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     operator* (FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     operator/ (FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     operator* (FXMVECTOR V, float S);
XMVECTOR    XM_CALLCOLW     operator* (float S, FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     operator/ (FXMVECTOR V, float S);

//------------------------------------------------------------------------------
// Matrix type: Sixteen 32 bit floating point components aligned on a
// 16 byte boundary and mapped to four hardware vector registers

struct XMMATRIX;

// Fix-up for (1st) XMMATRIX parameter to pass in-register for ARM64 and vector call; by reference otherwise
#if ( defined(_M_ARM64) || _XM_VECTORCALL_ ) && !defined(_XM_NO_INTRINSICS_)
typedef const XMMATRIX FXMMATRIX;
#else
typedef const XMMATRIX& FXMMATRIX;
#endif

// Fix-up for (2nd+) XMMATRIX parameters to pass by reference
typedef const XMMATRIX& CXMMATRIX;

#ifdef _XM_NO_INTRINSICS_
struct XMMATRIX
#else
__declspec(align(16)) struct XMMATRIX
#endif
{
#ifdef _XM_NO_INTRINSICS_
    union
    {
        XMVECTOR r[4];
        struct
        {
            float _11, _12, _13, _14;
            float _21, _22, _23, _24;
            float _31, _32, _33, _34;
            float _41, _42, _43, _44;
        };
        float m[4][4];
    };
#else
    XMVECTOR r[4];
#endif

    XMMATRIX() XM_CTOR_DEFAULT
    XMMATRIX(FXMVECTOR R0, FXMVECTOR R1, FXMVECTOR R2, CXMVECTOR R3) { r[0] = R0; r[1] = R1; r[2] = R2; r[3] = R3; }
    XMMATRIX(float m00, float m01, float m02, float m03,
             float m10, float m11, float m12, float m13,
             float m20, float m21, float m22, float m23,
             float m30, float m31, float m32, float m33);
    explicit XMMATRIX(_In_reads_(16) const float *pArray);

#ifdef _XM_NO_INTRINSICS_
    float       operator() (size_t Row, size_t Column) const { return m[Row][Column]; }
    float&      operator() (size_t Row, size_t Column) { return m[Row][Column]; }
#endif

    XMMATRIX&   operator= (const XMMATRIX& M) { r[0] = M.r[0]; r[1] = M.r[1]; r[2] = M.r[2]; r[3] = M.r[3]; return *this; }

    XMMATRIX    operator+ () const { return *this; }
    XMMATRIX    operator- () const;

    XMMATRIX&   XM_CALLCOLW     operator+= (FXMMATRIX M);
    XMMATRIX&   XM_CALLCOLW     operator-= (FXMMATRIX M);
    XMMATRIX&   XM_CALLCOLW     operator*= (FXMMATRIX M);
    XMMATRIX&   operator*= (float S);
    XMMATRIX&   operator/= (float S);

    XMMATRIX    XM_CALLCOLW     operator+ (FXMMATRIX M) const;
    XMMATRIX    XM_CALLCOLW     operator- (FXMMATRIX M) const;
    XMMATRIX    XM_CALLCOLW     operator* (FXMMATRIX M) const;
    XMMATRIX    operator* (float S) const;
    XMMATRIX    operator/ (float S) const;

    friend XMMATRIX     XM_CALLCOLW     operator* (float S, FXMMATRIX M);
};

//------------------------------------------------------------------------------
// 2D Vector; 32 bit floating point components
struct XMFLOAT2
{
    float x;
    float y;

    XMFLOAT2() XM_CTOR_DEFAULT
    XMFLOAT2(float _x, float _y) : x(_x), y(_y) {}
    explicit XMFLOAT2(_In_reads_(2) const float *pArray) : x(pArray[0]), y(pArray[1]) {}

    XMFLOAT2& operator= (const XMFLOAT2& Float2) { x = Float2.x; y = Float2.y; return *this; }
};

// 2D Vector; 32 bit floating point components aligned on a 16 byte boundary
__declspec(align(16)) struct XMFLOAT2A : public XMFLOAT2
{
    XMFLOAT2A() XM_CTOR_DEFAULT
    XMFLOAT2A(float _x, float _y) : XMFLOAT2(_x, _y) {}
    explicit XMFLOAT2A(_In_reads_(2) const float *pArray) : XMFLOAT2(pArray) {}

    XMFLOAT2A& operator= (const XMFLOAT2A& Float2) { x = Float2.x; y = Float2.y; return *this; }
};

//------------------------------------------------------------------------------
// 2D Vector; 32 bit signed integer components
struct XMINT2
{
    int32_t x;
    int32_t y;

    XMINT2() XM_CTOR_DEFAULT
    XMINT2(int32_t _x, int32_t _y) : x(_x), y(_y) {}
    explicit XMINT2(_In_reads_(2) const int32_t *pArray) : x(pArray[0]), y(pArray[1]) {}

    XMINT2& operator= (const XMINT2& Int2) { x = Int2.x; y = Int2.y; return *this; }
};

// 2D Vector; 32 bit unsigned integer components
struct XMUINT2
{
    uint32_t x;
    uint32_t y;

    XMUINT2() XM_CTOR_DEFAULT
    XMUINT2(uint32_t _x, uint32_t _y) : x(_x), y(_y) {}
    explicit XMUINT2(_In_reads_(2) const uint32_t *pArray) : x(pArray[0]), y(pArray[1]) {}

    XMUINT2& operator= (const XMUINT2& UInt2) { x = UInt2.x; y = UInt2.y; return *this; }
};

//------------------------------------------------------------------------------
// 3D Vector; 32 bit floating point components
struct XMFLOAT3
{
    float x;
    float y;
    float z;

    XMFLOAT3() XM_CTOR_DEFAULT
    XMFLOAT3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    explicit XMFLOAT3(_In_reads_(3) const float *pArray) : x(pArray[0]), y(pArray[1]), z(pArray[2]) {}

    XMFLOAT3& operator= (const XMFLOAT3& Float3) { x = Float3.x; y = Float3.y; z = Float3.z; return *this; }
};

// 3D Vector; 32 bit floating point components aligned on a 16 byte boundary
__declspec(align(16)) struct XMFLOAT3A : public XMFLOAT3
{
    XMFLOAT3A() XM_CTOR_DEFAULT
    XMFLOAT3A(float _x, float _y, float _z) : XMFLOAT3(_x, _y, _z) {}
    explicit XMFLOAT3A(_In_reads_(3) const float *pArray) : XMFLOAT3(pArray) {}

    XMFLOAT3A& operator= (const XMFLOAT3A& Float3) { x = Float3.x; y = Float3.y; z = Float3.z; return *this; }
};

//------------------------------------------------------------------------------
// 3D Vector; 32 bit signed integer components
struct XMINT3
{
    int32_t x;
    int32_t y;
    int32_t z;

    XMINT3() XM_CTOR_DEFAULT
    XMINT3(int32_t _x, int32_t _y, int32_t _z) : x(_x), y(_y), z(_z) {}
    explicit XMINT3(_In_reads_(3) const int32_t *pArray) : x(pArray[0]), y(pArray[1]), z(pArray[2]) {}

    XMINT3& operator= (const XMINT3& i3) { x = i3.x; y = i3.y; z = i3.z; return *this; }
};

// 3D Vector; 32 bit unsigned integer components
struct XMUINT3
{
    uint32_t x;
    uint32_t y;
    uint32_t z;

    XMUINT3() XM_CTOR_DEFAULT
    XMUINT3(uint32_t _x, uint32_t _y, uint32_t _z) : x(_x), y(_y), z(_z) {}
    explicit XMUINT3(_In_reads_(3) const uint32_t *pArray) : x(pArray[0]), y(pArray[1]), z(pArray[2]) {}

    XMUINT3& operator= (const XMUINT3& u3) { x = u3.x; y = u3.y; z = u3.z; return *this; }
};

//------------------------------------------------------------------------------
// 4D Vector; 32 bit floating point components
struct XMFLOAT4
{
    float x;
    float y;
    float z;
    float w;

    XMFLOAT4() XM_CTOR_DEFAULT
    XMFLOAT4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
    explicit XMFLOAT4(_In_reads_(4) const float *pArray) : x(pArray[0]), y(pArray[1]), z(pArray[2]), w(pArray[3]) {}

    XMFLOAT4& operator= (const XMFLOAT4& Float4) { x = Float4.x; y = Float4.y; z = Float4.z; w = Float4.w; return *this; }
};

// 4D Vector; 32 bit floating point components aligned on a 16 byte boundary
__declspec(align(16)) struct XMFLOAT4A : public XMFLOAT4
{
    XMFLOAT4A() XM_CTOR_DEFAULT
    XMFLOAT4A(float _x, float _y, float _z, float _w) : XMFLOAT4(_x, _y, _z, _w) {}
    explicit XMFLOAT4A(_In_reads_(4) const float *pArray) : XMFLOAT4(pArray) {}

    XMFLOAT4A& operator= (const XMFLOAT4A& Float4) { x = Float4.x; y = Float4.y; z = Float4.z; w = Float4.w; return *this; }
};

//------------------------------------------------------------------------------
// 4D Vector; 32 bit signed integer components
struct XMINT4
{
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;

    XMINT4() XM_CTOR_DEFAULT
    XMINT4(int32_t _x, int32_t _y, int32_t _z, int32_t _w) : x(_x), y(_y), z(_z), w(_w) {}
    explicit XMINT4(_In_reads_(4) const int32_t *pArray) : x(pArray[0]), y(pArray[1]), z(pArray[2]), w(pArray[3]) {}

    XMINT4& operator= (const XMINT4& Int4) { x = Int4.x; y = Int4.y; z = Int4.z; w = Int4.w; return *this; }
};

// 4D Vector; 32 bit unsigned integer components
struct XMUINT4
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;

    XMUINT4() XM_CTOR_DEFAULT
    XMUINT4(uint32_t _x, uint32_t _y, uint32_t _z, uint32_t _w) : x(_x), y(_y), z(_z), w(_w) {}
    explicit XMUINT4(_In_reads_(4) const uint32_t *pArray) : x(pArray[0]), y(pArray[1]), z(pArray[2]), w(pArray[3]) {}

    XMUINT4& operator= (const XMUINT4& UInt4) { x = UInt4.x; y = UInt4.y; z = UInt4.z; w = UInt4.w; return *this; }
};

//------------------------------------------------------------------------------
// 3x3 Matrix: 32 bit floating point components
struct XMFLOAT3X3
{
    union
    {
        struct
        {
            float _11, _12, _13;
            float _21, _22, _23;
            float _31, _32, _33;
        };
        float m[3][3];
    };

    XMFLOAT3X3() XM_CTOR_DEFAULT
    XMFLOAT3X3(float m00, float m01, float m02,
                float m10, float m11, float m12,
                float m20, float m21, float m22);
    explicit XMFLOAT3X3(_In_reads_(9) const float *pArray);

    float       operator() (size_t Row, size_t Column) const { return m[Row][Column]; }
    float&      operator() (size_t Row, size_t Column) { return m[Row][Column]; }

    XMFLOAT3X3& operator= (const XMFLOAT3X3& Float3x3);
};

//------------------------------------------------------------------------------
// 4x3 Matrix: 32 bit floating point components
struct XMFLOAT4X3
{
    union
    {
        struct
        {
            float _11, _12, _13;
            float _21, _22, _23;
            float _31, _32, _33;
            float _41, _42, _43;
        };
        float m[4][3];
    };

    XMFLOAT4X3() XM_CTOR_DEFAULT
    XMFLOAT4X3(float m00, float m01, float m02,
                float m10, float m11, float m12,
                float m20, float m21, float m22,
                float m30, float m31, float m32);
    explicit XMFLOAT4X3(_In_reads_(12) const float *pArray);

    float       operator() (size_t Row, size_t Column) const { return m[Row][Column]; }
    float&      operator() (size_t Row, size_t Column) { return m[Row][Column]; }

    XMFLOAT4X3& operator= (const XMFLOAT4X3& Float4x3);

};

// 4x3 Matrix: 32 bit floating point components aligned on a 16 byte boundary
__declspec(align(16)) struct XMFLOAT4X3A : public XMFLOAT4X3
{
    XMFLOAT4X3A() XM_CTOR_DEFAULT
    XMFLOAT4X3A(float m00, float m01, float m02,
                float m10, float m11, float m12,
                float m20, float m21, float m22,
                float m30, float m31, float m32) :
        XMFLOAT4X3(m00,m01,m02,m10,m11,m12,m20,m21,m22,m30,m31,m32) {}
    explicit XMFLOAT4X3A(_In_reads_(12) const float *pArray) : XMFLOAT4X3(pArray) {}

    float       operator() (size_t Row, size_t Column) const { return m[Row][Column]; }
    float&      operator() (size_t Row, size_t Column) { return m[Row][Column]; }

    XMFLOAT4X3A& operator= (const XMFLOAT4X3A& Float4x3);
};

//------------------------------------------------------------------------------
// 4x4 Matrix: 32 bit floating point components
struct XMFLOAT4X4
{
    union
    {
        struct
        {
            float _11, _12, _13, _14;
            float _21, _22, _23, _24;
            float _31, _32, _33, _34;
            float _41, _42, _43, _44;
        };
        float m[4][4];
    };

    XMFLOAT4X4() XM_CTOR_DEFAULT
    XMFLOAT4X4(float m00, float m01, float m02, float m03,
                float m10, float m11, float m12, float m13,
                float m20, float m21, float m22, float m23,
                float m30, float m31, float m32, float m33);
    explicit XMFLOAT4X4(_In_reads_(16) const float *pArray);

    float       operator() (size_t Row, size_t Column) const { return m[Row][Column]; }
    float&      operator() (size_t Row, size_t Column) { return m[Row][Column]; }

    XMFLOAT4X4& operator= (const XMFLOAT4X4& Float4x4);
};

// 4x4 Matrix: 32 bit floating point components aligned on a 16 byte boundary
__declspec(align(16)) struct XMFLOAT4X4A : public XMFLOAT4X4
{
    XMFLOAT4X4A() XM_CTOR_DEFAULT
    XMFLOAT4X4A(float m00, float m01, float m02, float m03,
                float m10, float m11, float m12, float m13,
                float m20, float m21, float m22, float m23,
                float m30, float m31, float m32, float m33)
        : XMFLOAT4X4(m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33) {}
    explicit XMFLOAT4X4A(_In_reads_(16) const float *pArray) : XMFLOAT4X4(pArray) {}

    float       operator() (size_t Row, size_t Column) const { return m[Row][Column]; }
    float&      operator() (size_t Row, size_t Column) { return m[Row][Column]; }

    XMFLOAT4X4A& operator= (const XMFLOAT4X4A& Float4x4);
};

////////////////////////////////////////////////////////////////////////////////

#pragma prefast(pop)
#pragma warning(pop)

/****************************************************************************
 *
 * Data colwersion operations
 *
 ****************************************************************************/

XMVECTOR    XM_CALLCOLW     XMColwertVectorIntToFloat(FXMVECTOR VInt, uint32_t DivExponent);
XMVECTOR    XM_CALLCOLW     XMColwertVectorFloatToInt(FXMVECTOR VFloat, uint32_t MulExponent);
XMVECTOR    XM_CALLCOLW     XMColwertVectorUIntToFloat(FXMVECTOR VUInt, uint32_t DivExponent);
XMVECTOR    XM_CALLCOLW     XMColwertVectorFloatToUInt(FXMVECTOR VFloat, uint32_t MulExponent);

#if defined(__XNAMATH_H__) && defined(XMVectorSetBinaryConstant)
#undef XMVectorSetBinaryConstant
#undef XMVectorSplatConstant
#undef XMVectorSplatConstantInt
#endif

XMVECTOR    XM_CALLCOLW     XMVectorSetBinaryConstant(uint32_t C0, uint32_t C1, uint32_t C2, uint32_t C3);
XMVECTOR    XM_CALLCOLW     XMVectorSplatConstant(int32_t IntConstant, uint32_t DivExponent);
XMVECTOR    XM_CALLCOLW     XMVectorSplatConstantInt(int32_t IntConstant);

/****************************************************************************
 *
 * Load operations
 *
 ****************************************************************************/

XMVECTOR    XM_CALLCOLW     XMLoadInt(_In_ const uint32_t* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadFloat(_In_ const float* pSource);

XMVECTOR    XM_CALLCOLW     XMLoadInt2(_In_reads_(2) const uint32_t* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadInt2A(_In_reads_(2) const uint32_t* PSource);
XMVECTOR    XM_CALLCOLW     XMLoadFloat2(_In_ const XMFLOAT2* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadFloat2A(_In_ const XMFLOAT2A* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadSInt2(_In_ const XMINT2* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadUInt2(_In_ const XMUINT2* pSource);

XMVECTOR    XM_CALLCOLW     XMLoadInt3(_In_reads_(3) const uint32_t* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadInt3A(_In_reads_(3) const uint32_t* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadFloat3(_In_ const XMFLOAT3* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadFloat3A(_In_ const XMFLOAT3A* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadSInt3(_In_ const XMINT3* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadUInt3(_In_ const XMUINT3* pSource);

XMVECTOR    XM_CALLCOLW     XMLoadInt4(_In_reads_(4) const uint32_t* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadInt4A(_In_reads_(4) const uint32_t* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadFloat4(_In_ const XMFLOAT4* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadFloat4A(_In_ const XMFLOAT4A* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadSInt4(_In_ const XMINT4* pSource);
XMVECTOR    XM_CALLCOLW     XMLoadUInt4(_In_ const XMUINT4* pSource);

XMMATRIX    XM_CALLCOLW     XMLoadFloat3x3(_In_ const XMFLOAT3X3* pSource);
XMMATRIX    XM_CALLCOLW     XMLoadFloat4x3(_In_ const XMFLOAT4X3* pSource);
XMMATRIX    XM_CALLCOLW     XMLoadFloat4x3A(_In_ const XMFLOAT4X3A* pSource);
XMMATRIX    XM_CALLCOLW     XMLoadFloat4x4(_In_ const XMFLOAT4X4* pSource);
XMMATRIX    XM_CALLCOLW     XMLoadFloat4x4A(_In_ const XMFLOAT4X4A* pSource);

/****************************************************************************
 *
 * Store operations
 *
 ****************************************************************************/

void        XM_CALLCOLW     XMStoreInt(_Out_ uint32_t* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreFloat(_Out_ float* pDestination, _In_ FXMVECTOR V);

void        XM_CALLCOLW     XMStoreInt2(_Out_writes_(2) uint32_t* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreInt2A(_Out_writes_(2) uint32_t* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreFloat2(_Out_ XMFLOAT2* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreFloat2A(_Out_ XMFLOAT2A* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreSInt2(_Out_ XMINT2* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreUInt2(_Out_ XMUINT2* pDestination, _In_ FXMVECTOR V);

void        XM_CALLCOLW     XMStoreInt3(_Out_writes_(3) uint32_t* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreInt3A(_Out_writes_(3) uint32_t* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreFloat3(_Out_ XMFLOAT3* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreFloat3A(_Out_ XMFLOAT3A* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreSInt3(_Out_ XMINT3* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreUInt3(_Out_ XMUINT3* pDestination, _In_ FXMVECTOR V);

void        XM_CALLCOLW     XMStoreInt4(_Out_writes_(4) uint32_t* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreInt4A(_Out_writes_(4) uint32_t* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreFloat4(_Out_ XMFLOAT4* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreFloat4A(_Out_ XMFLOAT4A* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreSInt4(_Out_ XMINT4* pDestination, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMStoreUInt4(_Out_ XMUINT4* pDestination, _In_ FXMVECTOR V);

void        XM_CALLCOLW     XMStoreFloat3x3(_Out_ XMFLOAT3X3* pDestination, _In_ FXMMATRIX M);
void        XM_CALLCOLW     XMStoreFloat4x3(_Out_ XMFLOAT4X3* pDestination, _In_ FXMMATRIX M);
void        XM_CALLCOLW     XMStoreFloat4x3A(_Out_ XMFLOAT4X3A* pDestination, _In_ FXMMATRIX M);
void        XM_CALLCOLW     XMStoreFloat4x4(_Out_ XMFLOAT4X4* pDestination, _In_ FXMMATRIX M);
void        XM_CALLCOLW     XMStoreFloat4x4A(_Out_ XMFLOAT4X4A* pDestination, _In_ FXMMATRIX M);

/****************************************************************************
 *
 * General vector operations
 *
 ****************************************************************************/

XMVECTOR    XM_CALLCOLW     XMVectorZero();
XMVECTOR    XM_CALLCOLW     XMVectorSet(float x, float y, float z, float w);
XMVECTOR    XM_CALLCOLW     XMVectorSetInt(uint32_t x, uint32_t y, uint32_t z, uint32_t w);
XMVECTOR    XM_CALLCOLW     XMVectorReplicate(float Value);
XMVECTOR    XM_CALLCOLW     XMVectorReplicatePtr(_In_ const float *pValue);
XMVECTOR    XM_CALLCOLW     XMVectorReplicateInt(uint32_t Value);
XMVECTOR    XM_CALLCOLW     XMVectorReplicateIntPtr(_In_ const uint32_t *pValue);
XMVECTOR    XM_CALLCOLW     XMVectorTrueInt();
XMVECTOR    XM_CALLCOLW     XMVectorFalseInt();
XMVECTOR    XM_CALLCOLW     XMVectorSplatX(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSplatY(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSplatZ(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSplatW(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSplatOne();
XMVECTOR    XM_CALLCOLW     XMVectorSplatInfinity();
XMVECTOR    XM_CALLCOLW     XMVectorSplatQNaN();
XMVECTOR    XM_CALLCOLW     XMVectorSplatEpsilon();
XMVECTOR    XM_CALLCOLW     XMVectorSplatSignMask();

float       XM_CALLCOLW     XMVectorGetByIndex(FXMVECTOR V, size_t i);
float       XM_CALLCOLW     XMVectorGetX(FXMVECTOR V);
float       XM_CALLCOLW     XMVectorGetY(FXMVECTOR V);
float       XM_CALLCOLW     XMVectorGetZ(FXMVECTOR V);
float       XM_CALLCOLW     XMVectorGetW(FXMVECTOR V);

void        XM_CALLCOLW     XMVectorGetByIndexPtr(_Out_ float *f, _In_ FXMVECTOR V, _In_ size_t i);
void        XM_CALLCOLW     XMVectorGetXPtr(_Out_ float *x, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMVectorGetYPtr(_Out_ float *y, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMVectorGetZPtr(_Out_ float *z, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMVectorGetWPtr(_Out_ float *w, _In_ FXMVECTOR V);

uint32_t    XM_CALLCOLW     XMVectorGetIntByIndex(FXMVECTOR V, size_t i);
uint32_t    XM_CALLCOLW     XMVectorGetIntX(FXMVECTOR V);
uint32_t    XM_CALLCOLW     XMVectorGetIntY(FXMVECTOR V);
uint32_t    XM_CALLCOLW     XMVectorGetIntZ(FXMVECTOR V);
uint32_t    XM_CALLCOLW     XMVectorGetIntW(FXMVECTOR V);

void        XM_CALLCOLW     XMVectorGetIntByIndexPtr(_Out_ uint32_t *x, _In_ FXMVECTOR V, _In_ size_t i);
void        XM_CALLCOLW     XMVectorGetIntXPtr(_Out_ uint32_t *x, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMVectorGetIntYPtr(_Out_ uint32_t *y, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMVectorGetIntZPtr(_Out_ uint32_t *z, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMVectorGetIntWPtr(_Out_ uint32_t *w, _In_ FXMVECTOR V);

XMVECTOR    XM_CALLCOLW     XMVectorSetByIndex(FXMVECTOR V,float f, size_t i);
XMVECTOR    XM_CALLCOLW     XMVectorSetX(FXMVECTOR V, float x);
XMVECTOR    XM_CALLCOLW     XMVectorSetY(FXMVECTOR V, float y);
XMVECTOR    XM_CALLCOLW     XMVectorSetZ(FXMVECTOR V, float z);
XMVECTOR    XM_CALLCOLW     XMVectorSetW(FXMVECTOR V, float w);

XMVECTOR    XM_CALLCOLW     XMVectorSetByIndexPtr(_In_ FXMVECTOR V, _In_ const float *f, _In_ size_t i);
XMVECTOR    XM_CALLCOLW     XMVectorSetXPtr(_In_ FXMVECTOR V, _In_ const float *x);
XMVECTOR    XM_CALLCOLW     XMVectorSetYPtr(_In_ FXMVECTOR V, _In_ const float *y);
XMVECTOR    XM_CALLCOLW     XMVectorSetZPtr(_In_ FXMVECTOR V, _In_ const float *z);
XMVECTOR    XM_CALLCOLW     XMVectorSetWPtr(_In_ FXMVECTOR V, _In_ const float *w);

XMVECTOR    XM_CALLCOLW     XMVectorSetIntByIndex(FXMVECTOR V, uint32_t x, size_t i);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntX(FXMVECTOR V, uint32_t x);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntY(FXMVECTOR V, uint32_t y);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntZ(FXMVECTOR V, uint32_t z);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntW(FXMVECTOR V, uint32_t w);

XMVECTOR    XM_CALLCOLW     XMVectorSetIntByIndexPtr(_In_ FXMVECTOR V, _In_ const uint32_t *x, _In_ size_t i);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntXPtr(_In_ FXMVECTOR V, _In_ const uint32_t *x);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntYPtr(_In_ FXMVECTOR V, _In_ const uint32_t *y);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntZPtr(_In_ FXMVECTOR V, _In_ const uint32_t *z);
XMVECTOR    XM_CALLCOLW     XMVectorSetIntWPtr(_In_ FXMVECTOR V, _In_ const uint32_t *w);

#if defined(__XNAMATH_H__) && defined(XMVectorSwizzle)
#undef XMVectorSwizzle
#endif

XMVECTOR    XM_CALLCOLW     XMVectorSwizzle(FXMVECTOR V, uint32_t E0, uint32_t E1, uint32_t E2, uint32_t E3);
XMVECTOR    XM_CALLCOLW     XMVectorPermute(FXMVECTOR V1, FXMVECTOR V2, uint32_t PermuteX, uint32_t PermuteY, uint32_t PermuteZ, uint32_t PermuteW);
XMVECTOR    XM_CALLCOLW     XMVectorSelectControl(uint32_t VectorIndex0, uint32_t VectorIndex1, uint32_t VectorIndex2, uint32_t VectorIndex3);
XMVECTOR    XM_CALLCOLW     XMVectorSelect(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR Control);
XMVECTOR    XM_CALLCOLW     XMVectorMergeXY(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorMergeZW(FXMVECTOR V1, FXMVECTOR V2);

#if defined(__XNAMATH_H__) && defined(XMVectorShiftLeft)
#undef XMVectorShiftLeft
#undef XMVectorRotateLeft
#undef XMVectorRotateRight
#undef XMVectorInsert
#endif

XMVECTOR    XM_CALLCOLW     XMVectorShiftLeft(FXMVECTOR V1, FXMVECTOR V2, uint32_t Elements);
XMVECTOR    XM_CALLCOLW     XMVectorRotateLeft(FXMVECTOR V, uint32_t Elements);
XMVECTOR    XM_CALLCOLW     XMVectorRotateRight(FXMVECTOR V, uint32_t Elements);
XMVECTOR    XM_CALLCOLW     XMVectorInsert(FXMVECTOR VD, FXMVECTOR VS, uint32_t VSLeftRotateElements,
                                           uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3);

XMVECTOR    XM_CALLCOLW     XMVectorEqual(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorEqualR(_Out_ uint32_t* pCR, _In_ FXMVECTOR V1, _In_ FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorEqualInt(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorEqualIntR(_Out_ uint32_t* pCR, _In_ FXMVECTOR V, _In_ FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorNearEqual(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR Epsilon);
XMVECTOR    XM_CALLCOLW     XMVectorNotEqual(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorNotEqualInt(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorGreater(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorGreaterR(_Out_ uint32_t* pCR, _In_ FXMVECTOR V1, _In_ FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorGreaterOrEqual(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorGreaterOrEqualR(_Out_ uint32_t* pCR, _In_ FXMVECTOR V1, _In_ FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorLess(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorLessOrEqual(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorInBounds(FXMVECTOR V, FXMVECTOR Bounds);
XMVECTOR    XM_CALLCOLW     XMVectorInBoundsR(_Out_ uint32_t* pCR, _In_ FXMVECTOR V, _In_ FXMVECTOR Bounds);

XMVECTOR    XM_CALLCOLW     XMVectorIsNaN(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorIsInfinite(FXMVECTOR V);

XMVECTOR    XM_CALLCOLW     XMVectorMin(FXMVECTOR V1,FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorMax(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorRound(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorTruncate(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorFloor(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorCeiling(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorClamp(FXMVECTOR V, FXMVECTOR Min, FXMVECTOR Max);
XMVECTOR    XM_CALLCOLW     XMVectorSaturate(FXMVECTOR V);

XMVECTOR    XM_CALLCOLW     XMVectorAndInt(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorAndCInt(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorOrInt(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorNorInt(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorXorInt(FXMVECTOR V1, FXMVECTOR V2);

XMVECTOR    XM_CALLCOLW     XMVectorNegate(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorAdd(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorAddAngles(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorSubtract(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorSubtractAngles(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorMultiply(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorMultiplyAdd(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR V3);
XMVECTOR    XM_CALLCOLW     XMVectorDivide(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorNegativeMultiplySubtract(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR V3);
XMVECTOR    XM_CALLCOLW     XMVectorScale(FXMVECTOR V, float ScaleFactor);
XMVECTOR    XM_CALLCOLW     XMVectorReciprocalEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorReciprocal(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSqrtEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSqrt(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorReciprocalSqrtEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorReciprocalSqrt(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorExp2(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorExpE(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorExp(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorLog2(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorLogE(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorLog(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorPow(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorAbs(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorMod(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVectorModAngles(FXMVECTOR Angles);
XMVECTOR    XM_CALLCOLW     XMVectorSin(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSinEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorCos(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorCosEst(FXMVECTOR V);
void        XM_CALLCOLW     XMVectorSinCos(_Out_ XMVECTOR* pSin, _Out_ XMVECTOR* pCos, _In_ FXMVECTOR V);
void        XM_CALLCOLW     XMVectorSinCosEst(_Out_ XMVECTOR* pSin, _Out_ XMVECTOR* pCos, _In_ FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorTan(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorTanEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorSinH(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorCosH(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorTanH(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorASin(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorASinEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorACos(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorACosEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorATan(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorATanEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVectorATan2(FXMVECTOR Y, FXMVECTOR X);
XMVECTOR    XM_CALLCOLW     XMVectorATan2Est(FXMVECTOR Y, FXMVECTOR X);
XMVECTOR    XM_CALLCOLW     XMVectorLerp(FXMVECTOR V0, FXMVECTOR V1, float t);
XMVECTOR    XM_CALLCOLW     XMVectorLerpV(FXMVECTOR V0, FXMVECTOR V1, FXMVECTOR T);
XMVECTOR    XM_CALLCOLW     XMVectorHermite(FXMVECTOR Position0, FXMVECTOR Tangent0, FXMVECTOR Position1, GXMVECTOR Tangent1, float t);
XMVECTOR    XM_CALLCOLW     XMVectorHermiteV(FXMVECTOR Position0, FXMVECTOR Tangent0, FXMVECTOR Position1, GXMVECTOR Tangent1, HXMVECTOR T);
XMVECTOR    XM_CALLCOLW     XMVectorCatmullRom(FXMVECTOR Position0, FXMVECTOR Position1, FXMVECTOR Position2, GXMVECTOR Position3, float t);
XMVECTOR    XM_CALLCOLW     XMVectorCatmullRomV(FXMVECTOR Position0, FXMVECTOR Position1, FXMVECTOR Position2, GXMVECTOR Position3, HXMVECTOR T);
XMVECTOR    XM_CALLCOLW     XMVectorBaryCentric(FXMVECTOR Position0, FXMVECTOR Position1, FXMVECTOR Position2, float f, float g);
XMVECTOR    XM_CALLCOLW     XMVectorBaryCentricV(FXMVECTOR Position0, FXMVECTOR Position1, FXMVECTOR Position2, GXMVECTOR F, HXMVECTOR G);

/****************************************************************************
 *
 * 2D vector operations
 *
 ****************************************************************************/

bool        XM_CALLCOLW     XMVector2Equal(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector2EqualR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2EqualInt(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector2EqualIntR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2NearEqual(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR Epsilon);
bool        XM_CALLCOLW     XMVector2NotEqual(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2NotEqualInt(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2Greater(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector2GreaterR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2GreaterOrEqual(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector2GreaterOrEqualR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2Less(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2LessOrEqual(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector2InBounds(FXMVECTOR V, FXMVECTOR Bounds);

bool        XM_CALLCOLW     XMVector2IsNaN(FXMVECTOR V);
bool        XM_CALLCOLW     XMVector2IsInfinite(FXMVECTOR V);

XMVECTOR    XM_CALLCOLW     XMVector2Dot(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector2Cross(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector2LengthSq(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2ReciprocalLengthEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2ReciprocalLength(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2LengthEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2Length(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2NormalizeEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2Normalize(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2ClampLength(FXMVECTOR V, float LengthMin, float LengthMax);
XMVECTOR    XM_CALLCOLW     XMVector2ClampLengthV(FXMVECTOR V, FXMVECTOR LengthMin, FXMVECTOR LengthMax);
XMVECTOR    XM_CALLCOLW     XMVector2Reflect(FXMVECTOR Incident, FXMVECTOR Normal);
XMVECTOR    XM_CALLCOLW     XMVector2Refract(FXMVECTOR Incident, FXMVECTOR Normal, float RefractionIndex);
XMVECTOR    XM_CALLCOLW     XMVector2RefractV(FXMVECTOR Incident, FXMVECTOR Normal, FXMVECTOR RefractionIndex);
XMVECTOR    XM_CALLCOLW     XMVector2Orthogonal(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector2AngleBetweenNormalsEst(FXMVECTOR N1, FXMVECTOR N2);
XMVECTOR    XM_CALLCOLW     XMVector2AngleBetweenNormals(FXMVECTOR N1, FXMVECTOR N2);
XMVECTOR    XM_CALLCOLW     XMVector2AngleBetweelwectors(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector2LinePointDistance(FXMVECTOR LinePoint1, FXMVECTOR LinePoint2, FXMVECTOR Point);
XMVECTOR    XM_CALLCOLW     XMVector2IntersectLine(FXMVECTOR Line1Point1, FXMVECTOR Line1Point2, FXMVECTOR Line2Point1, GXMVECTOR Line2Point2);
XMVECTOR    XM_CALLCOLW     XMVector2Transform(FXMVECTOR V, FXMMATRIX M);
XMFLOAT4*   XM_CALLCOLW     XMVector2TransformStream(_Out_writes_bytes_(sizeof(XMFLOAT4)+OutputStride*(VectorCount-1)) XMFLOAT4* pOutputStream,
                                                    _In_ size_t OutputStride,
                                                    _In_reads_bytes_(sizeof(XMFLOAT2)+InputStride*(VectorCount-1)) const XMFLOAT2* pInputStream,
                                                    _In_ size_t InputStride, _In_ size_t VectorCount, _In_ FXMMATRIX M);
XMVECTOR    XM_CALLCOLW     XMVector2TransformCoord(FXMVECTOR V, FXMMATRIX M);
XMFLOAT2*   XM_CALLCOLW     XMVector2TransformCoordStream(_Out_writes_bytes_(sizeof(XMFLOAT2)+OutputStride*(VectorCount-1)) XMFLOAT2* pOutputStream,
                                                          _In_ size_t OutputStride,
                                                          _In_reads_bytes_(sizeof(XMFLOAT2)+InputStride*(VectorCount-1)) const XMFLOAT2* pInputStream,
                                                          _In_ size_t InputStride, _In_ size_t VectorCount, _In_ FXMMATRIX M);
XMVECTOR    XM_CALLCOLW     XMVector2TransformNormal(FXMVECTOR V, FXMMATRIX M);
XMFLOAT2*   XM_CALLCOLW     XMVector2TransformNormalStream(_Out_writes_bytes_(sizeof(XMFLOAT2)+OutputStride*(VectorCount-1)) XMFLOAT2* pOutputStream,
                                                           _In_ size_t OutputStride,
                                                           _In_reads_bytes_(sizeof(XMFLOAT2)+InputStride*(VectorCount-1)) const XMFLOAT2* pInputStream,
                                                           _In_ size_t InputStride, _In_ size_t VectorCount, _In_ FXMMATRIX M);

/****************************************************************************
 *
 * 3D vector operations
 *
 ****************************************************************************/

bool        XM_CALLCOLW     XMVector3Equal(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector3EqualR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3EqualInt(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector3EqualIntR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3NearEqual(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR Epsilon);
bool        XM_CALLCOLW     XMVector3NotEqual(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3NotEqualInt(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3Greater(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector3GreaterR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3GreaterOrEqual(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector3GreaterOrEqualR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3Less(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3LessOrEqual(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector3InBounds(FXMVECTOR V, FXMVECTOR Bounds);

bool        XM_CALLCOLW     XMVector3IsNaN(FXMVECTOR V);
bool        XM_CALLCOLW     XMVector3IsInfinite(FXMVECTOR V);

XMVECTOR    XM_CALLCOLW     XMVector3Dot(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector3Cross(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector3LengthSq(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3ReciprocalLengthEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3ReciprocalLength(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3LengthEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3Length(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3NormalizeEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3Normalize(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3ClampLength(FXMVECTOR V, float LengthMin, float LengthMax);
XMVECTOR    XM_CALLCOLW     XMVector3ClampLengthV(FXMVECTOR V, FXMVECTOR LengthMin, FXMVECTOR LengthMax);
XMVECTOR    XM_CALLCOLW     XMVector3Reflect(FXMVECTOR Incident, FXMVECTOR Normal);
XMVECTOR    XM_CALLCOLW     XMVector3Refract(FXMVECTOR Incident, FXMVECTOR Normal, float RefractionIndex);
XMVECTOR    XM_CALLCOLW     XMVector3RefractV(FXMVECTOR Incident, FXMVECTOR Normal, FXMVECTOR RefractionIndex);
XMVECTOR    XM_CALLCOLW     XMVector3Orthogonal(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector3AngleBetweenNormalsEst(FXMVECTOR N1, FXMVECTOR N2);
XMVECTOR    XM_CALLCOLW     XMVector3AngleBetweenNormals(FXMVECTOR N1, FXMVECTOR N2);
XMVECTOR    XM_CALLCOLW     XMVector3AngleBetweelwectors(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector3LinePointDistance(FXMVECTOR LinePoint1, FXMVECTOR LinePoint2, FXMVECTOR Point);
void        XM_CALLCOLW     XMVector3ComponentsFromNormal(_Out_ XMVECTOR* pParallel, _Out_ XMVECTOR* pPerpendilwlar, _In_ FXMVECTOR V, _In_ FXMVECTOR Normal);
XMVECTOR    XM_CALLCOLW     XMVector3Rotate(FXMVECTOR V, FXMVECTOR RotationQuaternion);
XMVECTOR    XM_CALLCOLW     XMVector3IlwerseRotate(FXMVECTOR V, FXMVECTOR RotationQuaternion);
XMVECTOR    XM_CALLCOLW     XMVector3Transform(FXMVECTOR V, FXMMATRIX M);
XMFLOAT4*   XM_CALLCOLW     XMVector3TransformStream(_Out_writes_bytes_(sizeof(XMFLOAT4)+OutputStride*(VectorCount-1)) XMFLOAT4* pOutputStream,
                                                     _In_ size_t OutputStride,
                                                     _In_reads_bytes_(sizeof(XMFLOAT3)+InputStride*(VectorCount-1)) const XMFLOAT3* pInputStream,
                                                     _In_ size_t InputStride, _In_ size_t VectorCount, _In_ FXMMATRIX M);
XMVECTOR    XM_CALLCOLW     XMVector3TransformCoord(FXMVECTOR V, FXMMATRIX M);
XMFLOAT3*   XM_CALLCOLW     XMVector3TransformCoordStream(_Out_writes_bytes_(sizeof(XMFLOAT3)+OutputStride*(VectorCount-1)) XMFLOAT3* pOutputStream,
                                                          _In_ size_t OutputStride,
                                                          _In_reads_bytes_(sizeof(XMFLOAT3)+InputStride*(VectorCount-1)) const XMFLOAT3* pInputStream,
                                                          _In_ size_t InputStride, _In_ size_t VectorCount, _In_ FXMMATRIX M);
XMVECTOR    XM_CALLCOLW     XMVector3TransformNormal(FXMVECTOR V, FXMMATRIX M);
XMFLOAT3*   XM_CALLCOLW     XMVector3TransformNormalStream(_Out_writes_bytes_(sizeof(XMFLOAT3)+OutputStride*(VectorCount-1)) XMFLOAT3* pOutputStream,
                                                           _In_ size_t OutputStride,
                                                           _In_reads_bytes_(sizeof(XMFLOAT3)+InputStride*(VectorCount-1)) const XMFLOAT3* pInputStream,
                                                           _In_ size_t InputStride, _In_ size_t VectorCount, _In_ FXMMATRIX M);
XMVECTOR    XM_CALLCOLW     XMVector3Project(FXMVECTOR V, float ViewportX, float ViewportY, float ViewportWidth, float ViewportHeight, float ViewportMinZ, float ViewportMaxZ, 
                                             FXMMATRIX Projection, CXMMATRIX View, CXMMATRIX World);
XMFLOAT3*   XM_CALLCOLW     XMVector3ProjectStream(_Out_writes_bytes_(sizeof(XMFLOAT3)+OutputStride*(VectorCount-1)) XMFLOAT3* pOutputStream,
                                                   _In_ size_t OutputStride,
                                                   _In_reads_bytes_(sizeof(XMFLOAT3)+InputStride*(VectorCount-1)) const XMFLOAT3* pInputStream,
                                                   _In_ size_t InputStride, _In_ size_t VectorCount, 
                                                   _In_ float ViewportX, _In_ float ViewportY, _In_ float ViewportWidth, _In_ float ViewportHeight, _In_ float ViewportMinZ, _In_ float ViewportMaxZ, 
                                                   _In_ FXMMATRIX Projection, _In_ CXMMATRIX View, _In_ CXMMATRIX World);
XMVECTOR    XM_CALLCOLW     XMVector3Unproject(FXMVECTOR V, float ViewportX, float ViewportY, float ViewportWidth, float ViewportHeight, float ViewportMinZ, float ViewportMaxZ, 
                                               FXMMATRIX Projection, CXMMATRIX View, CXMMATRIX World);
XMFLOAT3*   XM_CALLCOLW     XMVector3UnprojectStream(_Out_writes_bytes_(sizeof(XMFLOAT3)+OutputStride*(VectorCount-1)) XMFLOAT3* pOutputStream,
                                                     _In_ size_t OutputStride,
                                                     _In_reads_bytes_(sizeof(XMFLOAT3)+InputStride*(VectorCount-1)) const XMFLOAT3* pInputStream,
                                                     _In_ size_t InputStride, _In_ size_t VectorCount, 
                                                     _In_ float ViewportX, _In_ float ViewportY, _In_ float ViewportWidth, _In_ float ViewportHeight, _In_ float ViewportMinZ, _In_ float ViewportMaxZ, 
                                                     _In_ FXMMATRIX Projection, _In_ CXMMATRIX View, _In_ CXMMATRIX World);

/****************************************************************************
 *
 * 4D vector operations
 *
 ****************************************************************************/

bool        XM_CALLCOLW     XMVector4Equal(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector4EqualR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4EqualInt(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector4EqualIntR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4NearEqual(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR Epsilon);
bool        XM_CALLCOLW     XMVector4NotEqual(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4NotEqualInt(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4Greater(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector4GreaterR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4GreaterOrEqual(FXMVECTOR V1, FXMVECTOR V2);
uint32_t    XM_CALLCOLW     XMVector4GreaterOrEqualR(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4Less(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4LessOrEqual(FXMVECTOR V1, FXMVECTOR V2);
bool        XM_CALLCOLW     XMVector4InBounds(FXMVECTOR V, FXMVECTOR Bounds);

bool        XM_CALLCOLW     XMVector4IsNaN(FXMVECTOR V);
bool        XM_CALLCOLW     XMVector4IsInfinite(FXMVECTOR V);

XMVECTOR    XM_CALLCOLW     XMVector4Dot(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector4Cross(FXMVECTOR V1, FXMVECTOR V2, FXMVECTOR V3);
XMVECTOR    XM_CALLCOLW     XMVector4LengthSq(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4ReciprocalLengthEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4ReciprocalLength(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4LengthEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4Length(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4NormalizeEst(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4Normalize(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4ClampLength(FXMVECTOR V, float LengthMin, float LengthMax);
XMVECTOR    XM_CALLCOLW     XMVector4ClampLengthV(FXMVECTOR V, FXMVECTOR LengthMin, FXMVECTOR LengthMax);
XMVECTOR    XM_CALLCOLW     XMVector4Reflect(FXMVECTOR Incident, FXMVECTOR Normal);
XMVECTOR    XM_CALLCOLW     XMVector4Refract(FXMVECTOR Incident, FXMVECTOR Normal, float RefractionIndex);
XMVECTOR    XM_CALLCOLW     XMVector4RefractV(FXMVECTOR Incident, FXMVECTOR Normal, FXMVECTOR RefractionIndex);
XMVECTOR    XM_CALLCOLW     XMVector4Orthogonal(FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMVector4AngleBetweenNormalsEst(FXMVECTOR N1, FXMVECTOR N2);
XMVECTOR    XM_CALLCOLW     XMVector4AngleBetweenNormals(FXMVECTOR N1, FXMVECTOR N2);
XMVECTOR    XM_CALLCOLW     XMVector4AngleBetweelwectors(FXMVECTOR V1, FXMVECTOR V2);
XMVECTOR    XM_CALLCOLW     XMVector4Transform(FXMVECTOR V, FXMMATRIX M);
XMFLOAT4*   XM_CALLCOLW     XMVector4TransformStream(_Out_writes_bytes_(sizeof(XMFLOAT4)+OutputStride*(VectorCount-1)) XMFLOAT4* pOutputStream,
                                                     _In_ size_t OutputStride,
                                                     _In_reads_bytes_(sizeof(XMFLOAT4)+InputStride*(VectorCount-1)) const XMFLOAT4* pInputStream,
                                                     _In_ size_t InputStride, _In_ size_t VectorCount, _In_ FXMMATRIX M);

/****************************************************************************
 *
 * Matrix operations
 *
 ****************************************************************************/

bool        XM_CALLCOLW     XMMatrixIsNaN(FXMMATRIX M);
bool        XM_CALLCOLW     XMMatrixIsInfinite(FXMMATRIX M);
bool        XM_CALLCOLW     XMMatrixIsIdentity(FXMMATRIX M);

XMMATRIX    XM_CALLCOLW     XMMatrixMultiply(FXMMATRIX M1, CXMMATRIX M2);
XMMATRIX    XM_CALLCOLW     XMMatrixMultiplyTranspose(FXMMATRIX M1, CXMMATRIX M2);
XMMATRIX    XM_CALLCOLW     XMMatrixTranspose(FXMMATRIX M);
XMMATRIX    XM_CALLCOLW     XMMatrixIlwerse(_Out_opt_ XMVECTOR* pDeterminant, _In_ FXMMATRIX M);
XMVECTOR    XM_CALLCOLW     XMMatrixDeterminant(FXMMATRIX M);
_Success_(return)
bool        XM_CALLCOLW     XMMatrixDecompose(_Out_ XMVECTOR *outScale, _Out_ XMVECTOR *outRotQuat, _Out_ XMVECTOR *outTrans, _In_ FXMMATRIX M);

XMMATRIX    XM_CALLCOLW     XMMatrixIdentity();
XMMATRIX    XM_CALLCOLW     XMMatrixSet(float m00, float m01, float m02, float m03,
                                        float m10, float m11, float m12, float m13,
                                        float m20, float m21, float m22, float m23,
                                        float m30, float m31, float m32, float m33);
XMMATRIX    XM_CALLCOLW     XMMatrixTranslation(float OffsetX, float OffsetY, float OffsetZ);
XMMATRIX    XM_CALLCOLW     XMMatrixTranslationFromVector(FXMVECTOR Offset);
XMMATRIX    XM_CALLCOLW     XMMatrixScaling(float ScaleX, float ScaleY, float ScaleZ);
XMMATRIX    XM_CALLCOLW     XMMatrixScalingFromVector(FXMVECTOR Scale);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationX(float Angle);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationY(float Angle);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationZ(float Angle);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationRollPitchYaw(float Pitch, float Yaw, float Roll);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationRollPitchYawFromVector(FXMVECTOR Angles);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationNormal(FXMVECTOR NormalAxis, float Angle);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationAxis(FXMVECTOR Axis, float Angle);
XMMATRIX    XM_CALLCOLW     XMMatrixRotationQuaternion(FXMVECTOR Quaternion);
XMMATRIX    XM_CALLCOLW     XMMatrixTransformation2D(FXMVECTOR ScalingOrigin, float ScalingOrientation, FXMVECTOR Scaling, 
                                                     FXMVECTOR RotationOrigin, float Rotation, GXMVECTOR Translation);
XMMATRIX    XM_CALLCOLW     XMMatrixTransformation(FXMVECTOR ScalingOrigin, FXMVECTOR ScalingOrientationQuaternion, FXMVECTOR Scaling, 
                                                   GXMVECTOR RotationOrigin, HXMVECTOR RotationQuaternion, HXMVECTOR Translation);
XMMATRIX    XM_CALLCOLW     XMMatrixAffineTransformation2D(FXMVECTOR Scaling, FXMVECTOR RotationOrigin, float Rotation, FXMVECTOR Translation);
XMMATRIX    XM_CALLCOLW     XMMatrixAffineTransformation(FXMVECTOR Scaling, FXMVECTOR RotationOrigin, FXMVECTOR RotationQuaternion, GXMVECTOR Translation);
XMMATRIX    XM_CALLCOLW     XMMatrixReflect(FXMVECTOR ReflectionPlane);
XMMATRIX    XM_CALLCOLW     XMMatrixShadow(FXMVECTOR ShadowPlane, FXMVECTOR LightPosition);

XMMATRIX    XM_CALLCOLW     XMMatrixLookAtLH(FXMVECTOR EyePosition, FXMVECTOR FolwsPosition, FXMVECTOR UpDirection);
XMMATRIX    XM_CALLCOLW     XMMatrixLookAtRH(FXMVECTOR EyePosition, FXMVECTOR FolwsPosition, FXMVECTOR UpDirection);
XMMATRIX    XM_CALLCOLW     XMMatrixLookToLH(FXMVECTOR EyePosition, FXMVECTOR EyeDirection, FXMVECTOR UpDirection);
XMMATRIX    XM_CALLCOLW     XMMatrixLookToRH(FXMVECTOR EyePosition, FXMVECTOR EyeDirection, FXMVECTOR UpDirection);
XMMATRIX    XM_CALLCOLW     XMMatrixPerspectiveLH(float ViewWidth, float ViewHeight, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixPerspectiveRH(float ViewWidth, float ViewHeight, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixPerspectiveFovLH(float FovAngleY, float AspectHByW, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixPerspectiveFovRH(float FovAngleY, float AspectHByW, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixPerspectiveOffCenterLH(float ViewLeft, float ViewRight, float ViewBottom, float ViewTop, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixPerspectiveOffCenterRH(float ViewLeft, float ViewRight, float ViewBottom, float ViewTop, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixOrthographicLH(float ViewWidth, float ViewHeight, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixOrthographicRH(float ViewWidth, float ViewHeight, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixOrthographicOffCenterLH(float ViewLeft, float ViewRight, float ViewBottom, float ViewTop, float NearZ, float FarZ);
XMMATRIX    XM_CALLCOLW     XMMatrixOrthographicOffCenterRH(float ViewLeft, float ViewRight, float ViewBottom, float ViewTop, float NearZ, float FarZ);


/****************************************************************************
 *
 * Quaternion operations
 *
 ****************************************************************************/

bool        XM_CALLCOLW     XMQuaternionEqual(FXMVECTOR Q1, FXMVECTOR Q2);
bool        XM_CALLCOLW     XMQuaternionNotEqual(FXMVECTOR Q1, FXMVECTOR Q2);

bool        XM_CALLCOLW     XMQuaternionIsNaN(FXMVECTOR Q);
bool        XM_CALLCOLW     XMQuaternionIsInfinite(FXMVECTOR Q);
bool        XM_CALLCOLW     XMQuaternionIsIdentity(FXMVECTOR Q);

XMVECTOR    XM_CALLCOLW     XMQuaternionDot(FXMVECTOR Q1, FXMVECTOR Q2);
XMVECTOR    XM_CALLCOLW     XMQuaternionMultiply(FXMVECTOR Q1, FXMVECTOR Q2);
XMVECTOR    XM_CALLCOLW     XMQuaternionLengthSq(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionReciprocalLength(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionLength(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionNormalizeEst(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionNormalize(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionConjugate(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionIlwerse(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionLn(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionExp(FXMVECTOR Q);
XMVECTOR    XM_CALLCOLW     XMQuaternionSlerp(FXMVECTOR Q0, FXMVECTOR Q1, float t);
XMVECTOR    XM_CALLCOLW     XMQuaternionSlerpV(FXMVECTOR Q0, FXMVECTOR Q1, FXMVECTOR T);
XMVECTOR    XM_CALLCOLW     XMQuaternionSquad(FXMVECTOR Q0, FXMVECTOR Q1, FXMVECTOR Q2, GXMVECTOR Q3, float t);
XMVECTOR    XM_CALLCOLW     XMQuaternionSquadV(FXMVECTOR Q0, FXMVECTOR Q1, FXMVECTOR Q2, GXMVECTOR Q3, HXMVECTOR T);
void        XM_CALLCOLW     XMQuaternionSquadSetup(_Out_ XMVECTOR* pA, _Out_ XMVECTOR* pB, _Out_ XMVECTOR* pC, _In_ FXMVECTOR Q0, _In_ FXMVECTOR Q1, _In_ FXMVECTOR Q2, _In_ GXMVECTOR Q3);
XMVECTOR    XM_CALLCOLW     XMQuaternionBaryCentric(FXMVECTOR Q0, FXMVECTOR Q1, FXMVECTOR Q2, float f, float g);
XMVECTOR    XM_CALLCOLW     XMQuaternionBaryCentricV(FXMVECTOR Q0, FXMVECTOR Q1, FXMVECTOR Q2, GXMVECTOR F, HXMVECTOR G);

XMVECTOR    XM_CALLCOLW     XMQuaternionIdentity();
XMVECTOR    XM_CALLCOLW     XMQuaternionRotationRollPitchYaw(float Pitch, float Yaw, float Roll);
XMVECTOR    XM_CALLCOLW     XMQuaternionRotationRollPitchYawFromVector(FXMVECTOR Angles);
XMVECTOR    XM_CALLCOLW     XMQuaternionRotationNormal(FXMVECTOR NormalAxis, float Angle);
XMVECTOR    XM_CALLCOLW     XMQuaternionRotationAxis(FXMVECTOR Axis, float Angle);
XMVECTOR    XM_CALLCOLW     XMQuaternionRotationMatrix(FXMMATRIX M);

void        XM_CALLCOLW     XMQuaternionToAxisAngle(_Out_ XMVECTOR* pAxis, _Out_ float* pAngle, _In_ FXMVECTOR Q);

/****************************************************************************
 *
 * Plane operations
 *
 ****************************************************************************/

bool        XM_CALLCOLW     XMPlaneEqual(FXMVECTOR P1, FXMVECTOR P2);
bool        XM_CALLCOLW     XMPlaneNearEqual(FXMVECTOR P1, FXMVECTOR P2, FXMVECTOR Epsilon);
bool        XM_CALLCOLW     XMPlaneNotEqual(FXMVECTOR P1, FXMVECTOR P2);

bool        XM_CALLCOLW     XMPlaneIsNaN(FXMVECTOR P);
bool        XM_CALLCOLW     XMPlaneIsInfinite(FXMVECTOR P);

XMVECTOR    XM_CALLCOLW     XMPlaneDot(FXMVECTOR P, FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMPlaneDotCoord(FXMVECTOR P, FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMPlaneDotNormal(FXMVECTOR P, FXMVECTOR V);
XMVECTOR    XM_CALLCOLW     XMPlaneNormalizeEst(FXMVECTOR P);
XMVECTOR    XM_CALLCOLW     XMPlaneNormalize(FXMVECTOR P);
XMVECTOR    XM_CALLCOLW     XMPlaneIntersectLine(FXMVECTOR P, FXMVECTOR LinePoint1, FXMVECTOR LinePoint2);
void        XM_CALLCOLW     XMPlaneIntersectPlane(_Out_ XMVECTOR* pLinePoint1, _Out_ XMVECTOR* pLinePoint2, _In_ FXMVECTOR P1, _In_ FXMVECTOR P2);
XMVECTOR    XM_CALLCOLW     XMPlaneTransform(FXMVECTOR P, FXMMATRIX M);
XMFLOAT4*   XM_CALLCOLW     XMPlaneTransformStream(_Out_writes_bytes_(sizeof(XMFLOAT4)+OutputStride*(PlaneCount-1)) XMFLOAT4* pOutputStream,
                                                   _In_ size_t OutputStride,
                                                   _In_reads_bytes_(sizeof(XMFLOAT4)+InputStride*(PlaneCount-1)) const XMFLOAT4* pInputStream,
                                                   _In_ size_t InputStride, _In_ size_t PlaneCount, _In_ FXMMATRIX M);

XMVECTOR    XM_CALLCOLW     XMPlaneFromPointNormal(FXMVECTOR Point, FXMVECTOR Normal);
XMVECTOR    XM_CALLCOLW     XMPlaneFromPoints(FXMVECTOR Point1, FXMVECTOR Point2, FXMVECTOR Point3);

/****************************************************************************
 *
 * Color operations
 *
 ****************************************************************************/

bool        XM_CALLCOLW     XMColorEqual(FXMVECTOR C1, FXMVECTOR C2);
bool        XM_CALLCOLW     XMColorNotEqual(FXMVECTOR C1, FXMVECTOR C2);
bool        XM_CALLCOLW     XMColorGreater(FXMVECTOR C1, FXMVECTOR C2);
bool        XM_CALLCOLW     XMColorGreaterOrEqual(FXMVECTOR C1, FXMVECTOR C2);
bool        XM_CALLCOLW     XMColorLess(FXMVECTOR C1, FXMVECTOR C2);
bool        XM_CALLCOLW     XMColorLessOrEqual(FXMVECTOR C1, FXMVECTOR C2);

bool        XM_CALLCOLW     XMColorIsNaN(FXMVECTOR C);
bool        XM_CALLCOLW     XMColorIsInfinite(FXMVECTOR C);

XMVECTOR    XM_CALLCOLW     XMColorNegative(FXMVECTOR C);
XMVECTOR    XM_CALLCOLW     XMColorModulate(FXMVECTOR C1, FXMVECTOR C2);
XMVECTOR    XM_CALLCOLW     XMColorAdjustSaturation(FXMVECTOR C, float Saturation);
XMVECTOR    XM_CALLCOLW     XMColorAdjustContrast(FXMVECTOR C, float Contrast);

XMVECTOR    XM_CALLCOLW     XMColorRGBToHSL( FXMVECTOR rgb );
XMVECTOR    XM_CALLCOLW     XMColorHSLToRGB( FXMVECTOR hsl );

XMVECTOR    XM_CALLCOLW     XMColorRGBToHSV( FXMVECTOR rgb );
XMVECTOR    XM_CALLCOLW     XMColorHSVToRGB( FXMVECTOR hsv );

XMVECTOR    XM_CALLCOLW     XMColorRGBToYUV( FXMVECTOR rgb );
XMVECTOR    XM_CALLCOLW     XMColorYUVToRGB( FXMVECTOR yuv );

XMVECTOR    XM_CALLCOLW     XMColorRGBToYUV_HD( FXMVECTOR rgb );
XMVECTOR    XM_CALLCOLW     XMColorYUVToRGB_HD( FXMVECTOR yuv );

XMVECTOR    XM_CALLCOLW     XMColorRGBToXYZ( FXMVECTOR rgb );
XMVECTOR    XM_CALLCOLW     XMColorXYZToRGB( FXMVECTOR xyz );

XMVECTOR    XM_CALLCOLW     XMColorXYZToSRGB( FXMVECTOR xyz );
XMVECTOR    XM_CALLCOLW     XMColorSRGBToXYZ( FXMVECTOR srgb );

XMVECTOR    XM_CALLCOLW     XMColorRGBToSRGB( FXMVECTOR rgb );
XMVECTOR    XM_CALLCOLW     XMColorSRGBToRGB( FXMVECTOR srgb );


/****************************************************************************
 *
 * Miscellaneous operations
 *
 ****************************************************************************/

bool            XMVerifyCPUSupport();

XMVECTOR    XM_CALLCOLW     XMFresnelTerm(FXMVECTOR CosIncidentAngle, FXMVECTOR RefractionIndex);

bool            XMScalarNearEqual(float S1, float S2, float Epsilon);
float           XMScalarModAngle(float Value);

float           XMScalarSin(float Value);
float           XMScalarSinEst(float Value);

float           XMScalarCos(float Value);
float           XMScalarCosEst(float Value);

void            XMScalarSinCos(_Out_ float* pSin, _Out_ float* pCos, float Value);
void            XMScalarSinCosEst(_Out_ float* pSin, _Out_ float* pCos, float Value);

float           XMScalarASin(float Value);
float           XMScalarASinEst(float Value);

float           XMScalarACos(float Value);
float           XMScalarACosEst(float Value);

/****************************************************************************
 *
 * Templates
 *
 ****************************************************************************/

#if defined(__XNAMATH_H__) && defined(XMMin)
#undef XMMin
#undef XMMax
#endif

template<class T> inline T XMMin(T a, T b) { return (a < b) ? a : b; }
template<class T> inline T XMMax(T a, T b) { return (a > b) ? a : b; }

//------------------------------------------------------------------------------

#if defined(_XM_SSE_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)

// PermuteHelper internal template (SSE only)
namespace Internal
{
    // Slow path fallback for permutes that do not map to a single SSE shuffle opcode.
    template<uint32_t Shuffle, bool WhichX, bool WhichY, bool WhichZ, bool WhichW> struct PermuteHelper
    {
        static XMVECTOR     XM_CALLCOLW     Permute(FXMVECTOR v1, FXMVECTOR v2)
        {
            static const XMVECTORU32 selectMask =
            {
                WhichX ? 0xFFFFFFFF : 0,
                WhichY ? 0xFFFFFFFF : 0,
                WhichZ ? 0xFFFFFFFF : 0,
                WhichW ? 0xFFFFFFFF : 0,
            };

            XMVECTOR shuffled1 = XM_PERMUTE_PS(v1, Shuffle);
            XMVECTOR shuffled2 = XM_PERMUTE_PS(v2, Shuffle);

            XMVECTOR masked1 = _mm_andnot_ps(selectMask, shuffled1);
            XMVECTOR masked2 = _mm_and_ps(selectMask, shuffled2);

            return _mm_or_ps(masked1, masked2);
        }
    };

    // Fast path for permutes that only read from the first vector.
    template<uint32_t Shuffle> struct PermuteHelper<Shuffle, false, false, false, false>
    {
        static XMVECTOR     XM_CALLCOLW     Permute(FXMVECTOR v1, FXMVECTOR v2) { (v2); return XM_PERMUTE_PS(v1, Shuffle); }
    };

    // Fast path for permutes that only read from the second vector.
    template<uint32_t Shuffle> struct PermuteHelper<Shuffle, true, true, true, true>
    {
        static XMVECTOR     XM_CALLCOLW     Permute(FXMVECTOR v1, FXMVECTOR v2){ (v1); return XM_PERMUTE_PS(v2, Shuffle); }
    };

    // Fast path for permutes that read XY from the first vector, ZW from the second.
    template<uint32_t Shuffle> struct PermuteHelper<Shuffle, false, false, true, true>
    {
        static XMVECTOR     XM_CALLCOLW     Permute(FXMVECTOR v1, FXMVECTOR v2) { return _mm_shuffle_ps(v1, v2, Shuffle); }
    };

    // Fast path for permutes that read XY from the second vector, ZW from the first.
    template<uint32_t Shuffle> struct PermuteHelper<Shuffle, true, true, false, false>
    {
        static XMVECTOR     XM_CALLCOLW     Permute(FXMVECTOR v1, FXMVECTOR v2) { return _mm_shuffle_ps(v2, v1, Shuffle); }
    };
};

#endif // _XM_SSE_INTRINSICS_ && !_XM_NO_INTRINSICS_

// General permute template
template<uint32_t PermuteX, uint32_t PermuteY, uint32_t PermuteZ, uint32_t PermuteW>
    inline XMVECTOR     XM_CALLCOLW     XMVectorPermute(FXMVECTOR V1, FXMVECTOR V2)
{
    static_assert(PermuteX <= 7, "PermuteX template parameter out of range");
    static_assert(PermuteY <= 7, "PermuteY template parameter out of range");
    static_assert(PermuteZ <= 7, "PermuteZ template parameter out of range");
    static_assert(PermuteW <= 7, "PermuteW template parameter out of range");

#if defined(_XM_SSE_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)
    const uint32_t Shuffle = _MM_SHUFFLE(PermuteW & 3, PermuteZ & 3, PermuteY & 3, PermuteX & 3);

    const bool WhichX = PermuteX > 3;
    const bool WhichY = PermuteY > 3;
    const bool WhichZ = PermuteZ > 3;
    const bool WhichW = PermuteW > 3;

    return Internal::PermuteHelper<Shuffle, WhichX, WhichY, WhichZ, WhichW>::Permute(V1, V2);
#else

    return XMVectorPermute( V1, V2, PermuteX, PermuteY, PermuteZ, PermuteW );

#endif
}

// Special-case permute templates
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,1,2,3>(FXMVECTOR V1, FXMVECTOR V2) { (V2); return V1; }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<4,5,6,7>(FXMVECTOR V1, FXMVECTOR V2) { (V1); return V2; }


#if defined(_XM_ARM_NEON_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)

// If the indices are all in the range 0-3 or 4-7, then use XMVectorSwizzle instead
// The mirror cases are not spelled out here as the programmer can always swap the arguments
// (i.e. prefer permutes where the X element comes from the V1 vector instead of the V2 vector)

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,1,4,5>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vget_low_f32(V1), vget_low_f32(V2) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<1,0,4,5>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_low_f32(V1) ), vget_low_f32(V2) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,1,5,4>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vget_low_f32(V1), vrev64_f32( vget_low_f32(V2) ) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<1,0,5,4>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_low_f32(V1) ), vrev64_f32( vget_low_f32(V2) ) ); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<2,3,6,7>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vget_high_f32(V1), vget_high_f32(V2) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<3,2,6,7>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_high_f32(V1) ), vget_high_f32(V2) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<2,3,7,6>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vget_high_f32(V1), vrev64_f32( vget_high_f32(V2) ) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<3,2,7,6>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_high_f32(V1) ), vrev64_f32( vget_high_f32(V2) ) ); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,1,6,7>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vget_low_f32(V1), vget_high_f32(V2) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<1,0,6,7>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_low_f32(V1) ), vget_high_f32(V2) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,1,7,6>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vget_low_f32(V1), vrev64_f32( vget_high_f32(V2) ) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<1,0,7,6>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_low_f32(V1) ), vrev64_f32( vget_high_f32(V2) ) ); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<3,2,4,5>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_high_f32(V1) ), vget_low_f32(V2) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<2,3,5,4>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vget_high_f32(V1), vrev64_f32( vget_low_f32(V2) ) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<3,2,5,4>(FXMVECTOR V1, FXMVECTOR V2) { return vcombine_f32( vrev64_f32( vget_high_f32(V1) ), vrev64_f32( vget_low_f32(V2) ) ); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,4,2,6>(FXMVECTOR V1, FXMVECTOR V2) { return vtrnq_f32(V1,V2).val[0]; }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<1,5,3,7>(FXMVECTOR V1, FXMVECTOR V2) { return vtrnq_f32(V1,V2).val[1]; }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,4,1,5>(FXMVECTOR V1, FXMVECTOR V2) { return vzipq_f32(V1,V2).val[0]; }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<2,6,3,7>(FXMVECTOR V1, FXMVECTOR V2) { return vzipq_f32(V1,V2).val[1]; }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<0,2,4,6>(FXMVECTOR V1, FXMVECTOR V2) { return vuzpq_f32(V1,V2).val[0]; }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<1,3,5,7>(FXMVECTOR V1, FXMVECTOR V2) { return vuzpq_f32(V1,V2).val[1]; }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<1,2,3,4>(FXMVECTOR V1, FXMVECTOR V2) { return vextq_f32(V1, V2, 1); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<2,3,4,5>(FXMVECTOR V1, FXMVECTOR V2) { return vextq_f32(V1, V2, 2); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorPermute<3,4,5,6>(FXMVECTOR V1, FXMVECTOR V2) { return vextq_f32(V1, V2, 3); }

#endif // _XM_ARM_NEON_INTRINSICS_ && !_XM_NO_INTRINSICS_

//------------------------------------------------------------------------------

// General swizzle template
template<uint32_t SwizzleX, uint32_t SwizzleY, uint32_t SwizzleZ, uint32_t SwizzleW>
    inline XMVECTOR     XM_CALLCOLW     XMVectorSwizzle(FXMVECTOR V)
{
    static_assert(SwizzleX <= 3, "SwizzleX template parameter out of range");
    static_assert(SwizzleY <= 3, "SwizzleY template parameter out of range");
    static_assert(SwizzleZ <= 3, "SwizzleZ template parameter out of range");
    static_assert(SwizzleW <= 3, "SwizzleW template parameter out of range");

#if defined(_XM_SSE_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)
    return XM_PERMUTE_PS( V, _MM_SHUFFLE( SwizzleW, SwizzleZ, SwizzleY, SwizzleX ) );
#else

    return XMVectorSwizzle( V, SwizzleX, SwizzleY, SwizzleZ, SwizzleW );

#endif
}

// Specialized swizzles
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<0,1,2,3>(FXMVECTOR V) { return V; }


#if defined(_XM_ARM_NEON_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<0,0,0,0>(FXMVECTOR V) { return vdupq_lane_f32( vget_low_f32(V), 0); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<1,1,1,1>(FXMVECTOR V) { return vdupq_lane_f32( vget_low_f32(V), 1); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<2,2,2,2>(FXMVECTOR V) { return vdupq_lane_f32( vget_high_f32(V), 0); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<3,3,3,3>(FXMVECTOR V) { return vdupq_lane_f32( vget_high_f32(V), 1); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<1,0,3,2>(FXMVECTOR V) { return vrev64q_f32(V); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<0,1,0,1>(FXMVECTOR V) { float32x2_t vt = vget_low_f32(V); return vcombine_f32( vt, vt ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<2,3,2,3>(FXMVECTOR V) { float32x2_t vt = vget_high_f32(V); return vcombine_f32( vt, vt ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<1,0,1,0>(FXMVECTOR V) { float32x2_t vt = vrev64_f32( vget_low_f32(V) ); return vcombine_f32( vt, vt ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<3,2,3,2>(FXMVECTOR V) { float32x2_t vt = vrev64_f32( vget_high_f32(V) ); return vcombine_f32( vt, vt ); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<0,1,3,2>(FXMVECTOR V) { return vcombine_f32( vget_low_f32(V), vrev64_f32( vget_high_f32(V) ) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<1,0,2,3>(FXMVECTOR V) { return vcombine_f32( vrev64_f32( vget_low_f32(V) ), vget_high_f32(V) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<2,3,1,0>(FXMVECTOR V) { return vcombine_f32( vget_high_f32(V), vrev64_f32( vget_low_f32(V) ) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<3,2,0,1>(FXMVECTOR V) { return vcombine_f32( vrev64_f32( vget_high_f32(V) ), vget_low_f32(V) ); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<3,2,1,0>(FXMVECTOR V) { return vcombine_f32( vrev64_f32( vget_high_f32(V) ), vrev64_f32( vget_low_f32(V) ) ); }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<0,0,2,2>(FXMVECTOR V) { return vtrnq_f32(V,V).val[0]; }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<1,1,3,3>(FXMVECTOR V) { return vtrnq_f32(V,V).val[1]; }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<0,0,1,1>(FXMVECTOR V) { return vzipq_f32(V,V).val[0]; }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<2,2,3,3>(FXMVECTOR V) { return vzipq_f32(V,V).val[1]; }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<0,2,0,2>(FXMVECTOR V) { return vuzpq_f32(V,V).val[0]; }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<1,3,1,3>(FXMVECTOR V) { return vuzpq_f32(V,V).val[1]; }

template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<1,2,3,0>(FXMVECTOR V) { return vextq_f32(V, V, 1); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<2,3,0,1>(FXMVECTOR V) { return vextq_f32(V, V, 2); }
template<> inline XMVECTOR      XM_CALLCOLW     XMVectorSwizzle<3,0,1,2>(FXMVECTOR V) { return vextq_f32(V, V, 3); }

#endif // _XM_ARM_NEON_INTRINSICS_ && !_XM_NO_INTRINSICS_

//------------------------------------------------------------------------------

template<uint32_t Elements>
    inline XMVECTOR     XM_CALLCOLW     XMVectorShiftLeft(FXMVECTOR V1, FXMVECTOR V2)
{
    static_assert( Elements < 4, "Elements template parameter out of range" );
    return XMVectorPermute<Elements, (Elements + 1), (Elements + 2), (Elements + 3)>(V1, V2);
}

template<uint32_t Elements>
    inline XMVECTOR     XM_CALLCOLW     XMVectorRotateLeft(FXMVECTOR V)
{
    static_assert( Elements < 4, "Elements template parameter out of range" );
    return XMVectorSwizzle<Elements & 3, (Elements + 1) & 3, (Elements + 2) & 3, (Elements + 3) & 3>(V);
}

template<uint32_t Elements>
    inline XMVECTOR     XM_CALLCOLW     XMVectorRotateRight(FXMVECTOR V)
{
    static_assert( Elements < 4, "Elements template parameter out of range" );
    return XMVectorSwizzle<(4 - Elements) & 3, (5 - Elements) & 3, (6 - Elements) & 3, (7 - Elements) & 3>(V);
}

template<uint32_t VSLeftRotateElements, uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
    inline XMVECTOR     XM_CALLCOLW     XMVectorInsert(FXMVECTOR VD, FXMVECTOR VS)
{
    XMVECTOR Control = XMVectorSelectControl(Select0&1, Select1&1, Select2&1, Select3&1);
    return XMVectorSelect( VD, XMVectorRotateLeft<VSLeftRotateElements>(VS), Control );
}

/****************************************************************************
 *
 * Globals
 *
 ****************************************************************************/

// The purpose of the following global constants is to prevent redundant 
// reloading of the constants when they are referenced by more than one
// separate inline math routine called within the same function.  Declaring
// a constant locally within a routine is sufficient to prevent redundant
// reloads of that constant when that single routine is called multiple
// times in a function, but if the constant is used (and declared) in a 
// separate math routine it would be reloaded.

#ifndef XMGLOBALCONST
#define XMGLOBALCONST extern const __declspec(selectany)
#endif

XMGLOBALCONST XMVECTORF32 g_XMSinCoefficients0    = {-0.16666667f, +0.0083333310f, -0.00019840874f, +2.7525562e-06f};
XMGLOBALCONST XMVECTORF32 g_XMSinCoefficients1    = {-2.3889859e-08f, -0.16665852f /*Est1*/, +0.0083139502f /*Est2*/, -0.00018524670f /*Est3*/};
XMGLOBALCONST XMVECTORF32 g_XMCosCoefficients0    = {-0.5f, +0.041666638f, -0.0013888378f, +2.4760495e-05f};
XMGLOBALCONST XMVECTORF32 g_XMCosCoefficients1    = {-2.6051615e-07f, -0.49992746f /*Est1*/, +0.041493919f /*Est2*/, -0.0012712436f /*Est3*/};
XMGLOBALCONST XMVECTORF32 g_XMTanCoefficients0    = {1.0f, 0.333333333f, 0.133333333f, 5.396825397e-2f};
XMGLOBALCONST XMVECTORF32 g_XMTanCoefficients1    = {2.186948854e-2f, 8.863235530e-3f, 3.592128167e-3f, 1.455834485e-3f};
XMGLOBALCONST XMVECTORF32 g_XMTanCoefficients2    = {5.900274264e-4f, 2.391290764e-4f, 9.691537707e-5f, 3.927832950e-5f};
XMGLOBALCONST XMVECTORF32 g_XMArcCoefficients0    = {+1.5707963050f, -0.2145988016f, +0.0889789874f, -0.0501743046f};
XMGLOBALCONST XMVECTORF32 g_XMArcCoefficients1    = {+0.0308918810f, -0.0170881256f, +0.0066700901f, -0.0012624911f};
XMGLOBALCONST XMVECTORF32 g_XMATanCoefficients0   = {-0.3333314528f, +0.1999355085f, -0.1420889944f, +0.1065626393f};
XMGLOBALCONST XMVECTORF32 g_XMATanCoefficients1   = {-0.0752896400f, +0.0429096138f, -0.0161657367f, +0.0028662257f};
XMGLOBALCONST XMVECTORF32 g_XMATanEstCoefficients0 = {+0.999866f, +0.999866f, +0.999866f, +0.999866f};
XMGLOBALCONST XMVECTORF32 g_XMATanEstCoefficients1 = {-0.3302995f, +0.180141f, -0.085133f, +0.0208351f};
XMGLOBALCONST XMVECTORF32 g_XMTanEstCoefficients  = {2.484f, -1.954923183e-1f, 2.467401101f, XM_1DIVPI};
XMGLOBALCONST XMVECTORF32 g_XMArcEstCoefficients  = {+1.5707288f,-0.2121144f,+0.0742610f,-0.0187293f};
XMGLOBALCONST XMVECTORF32 g_XMPiConstants0        = {XM_PI, XM_2PI, XM_1DIVPI, XM_1DIV2PI};
XMGLOBALCONST XMVECTORF32 g_XMIdentityR0          = {1.0f, 0.0f, 0.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMIdentityR1          = {0.0f, 1.0f, 0.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMIdentityR2          = {0.0f, 0.0f, 1.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMIdentityR3          = {0.0f, 0.0f, 0.0f, 1.0f};
XMGLOBALCONST XMVECTORF32 g_XMNegIdentityR0       = {-1.0f,0.0f, 0.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMNegIdentityR1       = {0.0f,-1.0f, 0.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMNegIdentityR2       = {0.0f, 0.0f,-1.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMNegIdentityR3       = {0.0f, 0.0f, 0.0f,-1.0f};
XMGLOBALCONST XMVECTORU32 g_XMNegativeZero      = {0x80000000, 0x80000000, 0x80000000, 0x80000000};
XMGLOBALCONST XMVECTORU32 g_XMNegate3           = {0x80000000, 0x80000000, 0x80000000, 0x00000000};
XMGLOBALCONST XMVECTORU32 g_XMMaskXY            = {0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000};
XMGLOBALCONST XMVECTORU32 g_XMMask3             = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000};
XMGLOBALCONST XMVECTORU32 g_XMMaskX             = {0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000};
XMGLOBALCONST XMVECTORU32 g_XMMaskY             = {0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000};
XMGLOBALCONST XMVECTORU32 g_XMMaskZ             = {0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000};
XMGLOBALCONST XMVECTORU32 g_XMMaskW             = {0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF};
XMGLOBALCONST XMVECTORF32 g_XMOne               = { 1.0f, 1.0f, 1.0f, 1.0f};
XMGLOBALCONST XMVECTORF32 g_XMOne3              = { 1.0f, 1.0f, 1.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMZero              = { 0.0f, 0.0f, 0.0f, 0.0f};
XMGLOBALCONST XMVECTORF32 g_XMTwo               = { 2.f, 2.f, 2.f, 2.f };
XMGLOBALCONST XMVECTORF32 g_XMFour              = { 4.f, 4.f, 4.f, 4.f };
XMGLOBALCONST XMVECTORF32 g_XMSix               = { 6.f, 6.f, 6.f, 6.f };
XMGLOBALCONST XMVECTORF32 g_XMNegativeOne       = {-1.0f,-1.0f,-1.0f,-1.0f};
XMGLOBALCONST XMVECTORF32 g_XMOneHalf           = { 0.5f, 0.5f, 0.5f, 0.5f};
XMGLOBALCONST XMVECTORF32 g_XMNegativeOneHalf   = {-0.5f,-0.5f,-0.5f,-0.5f};
XMGLOBALCONST XMVECTORF32 g_XMNegativeTwoPi     = {-XM_2PI, -XM_2PI, -XM_2PI, -XM_2PI};
XMGLOBALCONST XMVECTORF32 g_XMNegativePi        = {-XM_PI, -XM_PI, -XM_PI, -XM_PI};
XMGLOBALCONST XMVECTORF32 g_XMHalfPi            = {XM_PIDIV2, XM_PIDIV2, XM_PIDIV2, XM_PIDIV2};
XMGLOBALCONST XMVECTORF32 g_XMPi                = {XM_PI, XM_PI, XM_PI, XM_PI};
XMGLOBALCONST XMVECTORF32 g_XMReciprocalPi      = {XM_1DIVPI, XM_1DIVPI, XM_1DIVPI, XM_1DIVPI};
XMGLOBALCONST XMVECTORF32 g_XMTwoPi             = {XM_2PI, XM_2PI, XM_2PI, XM_2PI};
XMGLOBALCONST XMVECTORF32 g_XMReciprocalTwoPi   = {XM_1DIV2PI, XM_1DIV2PI, XM_1DIV2PI, XM_1DIV2PI};
XMGLOBALCONST XMVECTORF32 g_XMEpsilon           = {1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f, 1.192092896e-7f};
XMGLOBALCONST XMVECTORI32 g_XMInfinity          = {0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000};
XMGLOBALCONST XMVECTORI32 g_XMQNaN              = {0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000};
XMGLOBALCONST XMVECTORI32 g_XMQNaNTest          = {0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF};
XMGLOBALCONST XMVECTORI32 g_XMAbsMask           = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
XMGLOBALCONST XMVECTORI32 g_XMFltMin            = {0x00800000, 0x00800000, 0x00800000, 0x00800000};
XMGLOBALCONST XMVECTORI32 g_XMFltMax            = {0x7F7FFFFF, 0x7F7FFFFF, 0x7F7FFFFF, 0x7F7FFFFF};
XMGLOBALCONST XMVECTORU32 g_XMNegOneMask        = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
XMGLOBALCONST XMVECTORU32 g_XMMaskA8R8G8B8      = {0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000};
XMGLOBALCONST XMVECTORU32 g_XMFlipA8R8G8B8      = {0x00000000, 0x00000000, 0x00000000, 0x80000000};
XMGLOBALCONST XMVECTORF32 g_XMFixAA8R8G8B8      = {0.0f,0.0f,0.0f,(float)(0x80000000U)};
XMGLOBALCONST XMVECTORF32 g_XMNormalizeA8R8G8B8 = {1.0f/(255.0f*(float)(0x10000)),1.0f/(255.0f*(float)(0x100)),1.0f/255.0f,1.0f/(255.0f*(float)(0x1000000))};
XMGLOBALCONST XMVECTORU32 g_XMMaskA2B10G10R10   = {0x000003FF, 0x000FFC00, 0x3FF00000, 0xC0000000};
XMGLOBALCONST XMVECTORU32 g_XMFlipA2B10G10R10   = {0x00000200, 0x00080000, 0x20000000, 0x80000000};
XMGLOBALCONST XMVECTORF32 g_XMFixAA2B10G10R10   = {-512.0f,-512.0f*(float)(0x400),-512.0f*(float)(0x100000),(float)(0x80000000U)};
XMGLOBALCONST XMVECTORF32 g_XMNormalizeA2B10G10R10 = {1.0f/511.0f,1.0f/(511.0f*(float)(0x400)),1.0f/(511.0f*(float)(0x100000)),1.0f/(3.0f*(float)(0x40000000))};
XMGLOBALCONST XMVECTORU32 g_XMMaskX16Y16        = {0x0000FFFF, 0xFFFF0000, 0x00000000, 0x00000000};
XMGLOBALCONST XMVECTORI32 g_XMFlipX16Y16        = {0x00008000, 0x00000000, 0x00000000, 0x00000000};
XMGLOBALCONST XMVECTORF32 g_XMFixX16Y16         = {-32768.0f,0.0f,0.0f,0.0f};
XMGLOBALCONST XMVECTORF32 g_XMNormalizeX16Y16   = {1.0f/32767.0f,1.0f/(32767.0f*65536.0f),0.0f,0.0f};
XMGLOBALCONST XMVECTORU32 g_XMMaskX16Y16Z16W16  = {0x0000FFFF, 0x0000FFFF, 0xFFFF0000, 0xFFFF0000};
XMGLOBALCONST XMVECTORI32 g_XMFlipX16Y16Z16W16  = {0x00008000, 0x00008000, 0x00000000, 0x00000000};
XMGLOBALCONST XMVECTORF32 g_XMFixX16Y16Z16W16   = {-32768.0f,-32768.0f,0.0f,0.0f};
XMGLOBALCONST XMVECTORF32 g_XMNormalizeX16Y16Z16W16 = {1.0f/32767.0f,1.0f/32767.0f,1.0f/(32767.0f*65536.0f),1.0f/(32767.0f*65536.0f)};
XMGLOBALCONST XMVECTORF32 g_XMNoFraction        = {8388608.0f,8388608.0f,8388608.0f,8388608.0f};
XMGLOBALCONST XMVECTORI32 g_XMMaskByte          = {0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF};
XMGLOBALCONST XMVECTORF32 g_XMNegateX           = {-1.0f, 1.0f, 1.0f, 1.0f};
XMGLOBALCONST XMVECTORF32 g_XMNegateY           = { 1.0f,-1.0f, 1.0f, 1.0f};
XMGLOBALCONST XMVECTORF32 g_XMNegateZ           = { 1.0f, 1.0f,-1.0f, 1.0f};
XMGLOBALCONST XMVECTORF32 g_XMNegateW           = { 1.0f, 1.0f, 1.0f,-1.0f};
XMGLOBALCONST XMVECTORU32 g_XMSelect0101        = {XM_SELECT_0, XM_SELECT_1, XM_SELECT_0, XM_SELECT_1};
XMGLOBALCONST XMVECTORU32 g_XMSelect1010        = {XM_SELECT_1, XM_SELECT_0, XM_SELECT_1, XM_SELECT_0};
XMGLOBALCONST XMVECTORI32 g_XMOneHalfMinusEpsilon = { 0x3EFFFFFD, 0x3EFFFFFD, 0x3EFFFFFD, 0x3EFFFFFD};
XMGLOBALCONST XMVECTORU32 g_XMSelect1000        = {XM_SELECT_1, XM_SELECT_0, XM_SELECT_0, XM_SELECT_0};
XMGLOBALCONST XMVECTORU32 g_XMSelect1100        = {XM_SELECT_1, XM_SELECT_1, XM_SELECT_0, XM_SELECT_0};
XMGLOBALCONST XMVECTORU32 g_XMSelect1110        = {XM_SELECT_1, XM_SELECT_1, XM_SELECT_1, XM_SELECT_0};
XMGLOBALCONST XMVECTORU32 g_XMSelect1011          = { XM_SELECT_1, XM_SELECT_0, XM_SELECT_1, XM_SELECT_1 };
XMGLOBALCONST XMVECTORF32 g_XMFixupY16          = {1.0f,1.0f/65536.0f,0.0f,0.0f};
XMGLOBALCONST XMVECTORF32 g_XMFixupY16W16       = {1.0f,1.0f,1.0f/65536.0f,1.0f/65536.0f};
XMGLOBALCONST XMVECTORU32 g_XMFlipY             = {0,0x80000000,0,0};
XMGLOBALCONST XMVECTORU32 g_XMFlipZ             = {0,0,0x80000000,0};
XMGLOBALCONST XMVECTORU32 g_XMFlipW             = {0,0,0,0x80000000};
XMGLOBALCONST XMVECTORU32 g_XMFlipYZ            = {0,0x80000000,0x80000000,0};
XMGLOBALCONST XMVECTORU32 g_XMFlipZW            = {0,0,0x80000000,0x80000000};
XMGLOBALCONST XMVECTORU32 g_XMFlipYW            = {0,0x80000000,0,0x80000000};
XMGLOBALCONST XMVECTORI32 g_XMMaskDec4          = {0x3FF,0x3FF<<10,0x3FF<<20,0x3<<30};
XMGLOBALCONST XMVECTORI32 g_XMXorDec4           = {0x200,0x200<<10,0x200<<20,0};
XMGLOBALCONST XMVECTORF32 g_XMAddUDec4          = {0,0,0,32768.0f*65536.0f};
XMGLOBALCONST XMVECTORF32 g_XMAddDec4           = {-512.0f,-512.0f*1024.0f,-512.0f*1024.0f*1024.0f,0};
XMGLOBALCONST XMVECTORF32 g_XMMulDec4           = {1.0f,1.0f/1024.0f,1.0f/(1024.0f*1024.0f),1.0f/(1024.0f*1024.0f*1024.0f)};
XMGLOBALCONST XMVECTORU32 g_XMMaskByte4         = {0xFF,0xFF00,0xFF0000,0xFF000000};
XMGLOBALCONST XMVECTORI32 g_XMXorByte4          = {0x80,0x8000,0x800000,0x00000000};
XMGLOBALCONST XMVECTORF32 g_XMAddByte4          = {-128.0f,-128.0f*256.0f,-128.0f*65536.0f,0};
XMGLOBALCONST XMVECTORF32 g_XMFixUnsigned       = {32768.0f*65536.0f,32768.0f*65536.0f,32768.0f*65536.0f,32768.0f*65536.0f};
XMGLOBALCONST XMVECTORF32 g_XMMaxInt            = {65536.0f*32768.0f-128.0f,65536.0f*32768.0f-128.0f,65536.0f*32768.0f-128.0f,65536.0f*32768.0f-128.0f};
XMGLOBALCONST XMVECTORF32 g_XMMaxUInt           = {65536.0f*65536.0f-256.0f,65536.0f*65536.0f-256.0f,65536.0f*65536.0f-256.0f,65536.0f*65536.0f-256.0f};
XMGLOBALCONST XMVECTORF32 g_XMUnsignedFix       = {32768.0f*65536.0f,32768.0f*65536.0f,32768.0f*65536.0f,32768.0f*65536.0f};
XMGLOBALCONST XMVECTORF32 g_XMsrgbScale         = { 12.92f, 12.92f, 12.92f, 1.0f };
XMGLOBALCONST XMVECTORF32 g_XMsrgbA             = { 0.055f, 0.055f, 0.055f, 0.0f };
XMGLOBALCONST XMVECTORF32 g_XMsrgbA1            = { 1.055f, 1.055f, 1.055f, 1.0f };
XMGLOBALCONST XMVECTORI32 g_XMExponentBias      = {127, 127, 127, 127};
XMGLOBALCONST XMVECTORI32 g_XMSubnormalExponent = {-126, -126, -126, -126};
XMGLOBALCONST XMVECTORI32 g_XMNumTrailing       = {23, 23, 23, 23};
XMGLOBALCONST XMVECTORI32 g_XMMinNormal         = {0x00800000, 0x00800000, 0x00800000, 0x00800000};
XMGLOBALCONST XMVECTORU32 g_XMNegInfinity       = {0xFF800000, 0xFF800000, 0xFF800000, 0xFF800000};
XMGLOBALCONST XMVECTORU32 g_XMNegQNaN           = {0xFFC00000, 0xFFC00000, 0xFFC00000, 0xFFC00000};
XMGLOBALCONST XMVECTORI32 g_XMBin128            = {0x43000000, 0x43000000, 0x43000000, 0x43000000};
XMGLOBALCONST XMVECTORU32 g_XMBinNeg150         = {0xC3160000, 0xC3160000, 0xC3160000, 0xC3160000};
XMGLOBALCONST XMVECTORI32 g_XM253               = {253, 253, 253, 253};
XMGLOBALCONST XMVECTORF32 g_XMExpEst1           = {-6.93147182e-1f, -6.93147182e-1f, -6.93147182e-1f, -6.93147182e-1f};
XMGLOBALCONST XMVECTORF32 g_XMExpEst2           = {+2.40226462e-1f, +2.40226462e-1f, +2.40226462e-1f, +2.40226462e-1f};
XMGLOBALCONST XMVECTORF32 g_XMExpEst3           = {-5.55036440e-2f, -5.55036440e-2f, -5.55036440e-2f, -5.55036440e-2f};
XMGLOBALCONST XMVECTORF32 g_XMExpEst4           = {+9.61597636e-3f, +9.61597636e-3f, +9.61597636e-3f, +9.61597636e-3f};
XMGLOBALCONST XMVECTORF32 g_XMExpEst5           = {-1.32823968e-3f, -1.32823968e-3f, -1.32823968e-3f, -1.32823968e-3f};
XMGLOBALCONST XMVECTORF32 g_XMExpEst6           = {+1.47491097e-4f, +1.47491097e-4f, +1.47491097e-4f, +1.47491097e-4f};
XMGLOBALCONST XMVECTORF32 g_XMExpEst7           = {-1.08635004e-5f, -1.08635004e-5f, -1.08635004e-5f, -1.08635004e-5f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst0           = {+1.442693f, +1.442693f, +1.442693f, +1.442693f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst1           = {-0.721242f, -0.721242f, -0.721242f, -0.721242f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst2           = {+0.479384f, +0.479384f, +0.479384f, +0.479384f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst3           = {-0.350295f, -0.350295f, -0.350295f, -0.350295f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst4           = {+0.248590f, +0.248590f, +0.248590f, +0.248590f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst5           = {-0.145700f, -0.145700f, -0.145700f, -0.145700f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst6           = {+0.057148f, +0.057148f, +0.057148f, +0.057148f};
XMGLOBALCONST XMVECTORF32 g_XMLogEst7           = {-0.010578f, -0.010578f, -0.010578f, -0.010578f};
XMGLOBALCONST XMVECTORF32 g_XMLgE               = {+1.442695f, +1.442695f, +1.442695f, +1.442695f};
XMGLOBALCONST XMVECTORF32 g_XMIlwLgE            = {+6.93147182e-1f, +6.93147182e-1f, +6.93147182e-1f, +6.93147182e-1f};
XMGLOBALCONST XMVECTORF32 g_UByteMax            = {255.0f, 255.0f, 255.0f, 255.0f};
XMGLOBALCONST XMVECTORF32 g_ByteMin             = {-127.0f, -127.0f, -127.0f, -127.0f};
XMGLOBALCONST XMVECTORF32 g_ByteMax             = {127.0f, 127.0f, 127.0f, 127.0f};
XMGLOBALCONST XMVECTORF32 g_ShortMin            = {-32767.0f, -32767.0f, -32767.0f, -32767.0f};
XMGLOBALCONST XMVECTORF32 g_ShortMax            = {32767.0f, 32767.0f, 32767.0f, 32767.0f};
XMGLOBALCONST XMVECTORF32 g_UShortMax           = {65535.0f, 65535.0f, 65535.0f, 65535.0f};

/****************************************************************************
 *
 * Implementation
 *
 ****************************************************************************/

#pragma warning(push)
#pragma warning(disable:4068 4214 4204 4365 4616 4640 6001)

#pragma prefast(push)
#pragma prefast(disable : 25000, "FXMVECTOR is 16 bytes")

//------------------------------------------------------------------------------

inline XMVECTOR XM_CALLCOLW XMVectorSetBinaryConstant(uint32_t C0, uint32_t C1, uint32_t C2, uint32_t C3)
{
#if defined(_XM_NO_INTRINSICS_)
    XMVECTORU32 vResult;
    vResult.u[0] = (0-(C0&1)) & 0x3F800000;
    vResult.u[1] = (0-(C1&1)) & 0x3F800000;
    vResult.u[2] = (0-(C2&1)) & 0x3F800000;
    vResult.u[3] = (0-(C3&1)) & 0x3F800000;
    return vResult.v;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
    XMVECTORU32 vResult;
    vResult.u[0] = (0-(C0&1)) & 0x3F800000;
    vResult.u[1] = (0-(C1&1)) & 0x3F800000;
    vResult.u[2] = (0-(C2&1)) & 0x3F800000;
    vResult.u[3] = (0-(C3&1)) & 0x3F800000;
    return vResult.v;
#else // XM_SSE_INTRINSICS_
    static const XMVECTORU32 g_vMask1 = {1,1,1,1};
    // Move the parms to a vector
    __m128i vTemp = _mm_set_epi32(C3,C2,C1,C0);
    // Mask off the low bits
    vTemp = _mm_and_si128(vTemp,g_vMask1);
    // 0xFFFFFFFF on true bits
    vTemp = _mm_cmpeq_epi32(vTemp,g_vMask1);
    // 0xFFFFFFFF -> 1.0f, 0x00000000 -> 0.0f
    vTemp = _mm_and_si128(vTemp,g_XMOne);
    return _mm_castsi128_ps(vTemp);
#endif
}

//------------------------------------------------------------------------------

inline XMVECTOR XM_CALLCOLW XMVectorSplatConstant(int32_t IntConstant, uint32_t DivExponent)
{
    assert( IntConstant >= -16 && IntConstant <= 15 );
    assert( DivExponent < 32 );
#if defined(_XM_NO_INTRINSICS_)

    using DirectX::XMColwertVectorIntToFloat;

    XMVECTORI32 V = { IntConstant, IntConstant, IntConstant, IntConstant };
    return XMColwertVectorIntToFloat( V.v, DivExponent);

#elif defined(_XM_ARM_NEON_INTRINSICS_)
    // Splat the int
    int32x4_t vScale = vdupq_n_s32(IntConstant);
    // Colwert to a float
    XMVECTOR vResult = vcvtq_f32_s32(vScale);
    // Colwert DivExponent into 1.0f/(1<<DivExponent)
    uint32_t uScale = 0x3F800000U - (DivExponent << 23);
    // Splat the scalar value (It's really a float)
    vScale = vdupq_n_s32(uScale);
    // Multiply by the reciprocal (Perform a right shift by DivExponent)
    vResult = vmulq_f32(vResult,reinterpret_cast<const float32x4_t *>(&vScale)[0]);
    return vResult;
#else // XM_SSE_INTRINSICS_
    // Splat the int
    __m128i vScale = _mm_set1_epi32(IntConstant);
    // Colwert to a float
    XMVECTOR vResult = _mm_cvtepi32_ps(vScale);
    // Colwert DivExponent into 1.0f/(1<<DivExponent)
    uint32_t uScale = 0x3F800000U - (DivExponent << 23);
    // Splat the scalar value (It's really a float)
    vScale = _mm_set1_epi32(uScale);
    // Multiply by the reciprocal (Perform a right shift by DivExponent)
    vResult = _mm_mul_ps(vResult,_mm_castsi128_ps(vScale));
    return vResult;
#endif
}

//------------------------------------------------------------------------------

inline XMVECTOR XM_CALLCOLW XMVectorSplatConstantInt(int32_t IntConstant)
{
    assert( IntConstant >= -16 && IntConstant <= 15 );
#if defined(_XM_NO_INTRINSICS_)

    XMVECTORI32 V = { IntConstant, IntConstant, IntConstant, IntConstant };
    return V.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
    int32x4_t V = vdupq_n_s32( IntConstant );
    return reinterpret_cast<float32x4_t *>(&V)[0];
#else // XM_SSE_INTRINSICS_
    __m128i V = _mm_set1_epi32( IntConstant );
    return _mm_castsi128_ps(V);
#endif
}

#include "DirectXMathColwert.inl"
#include "DirectXMathVector.inl"
#include "DirectXMathMatrix.inl"
#include "DirectXMathMisc.inl"

#pragma prefast(pop)
#pragma warning(pop)

}; // namespace DirectX

