/*
 * copyright (c) 2020, lwpu corporation.  all rights reserved.
 *
 * lwpu corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from lwpu corporation is strictly prohibited.
 */

#if !defined(TYPE_COLWERT_H_INCLUDED_)
#define TYPE_COLWERT_H_INCLUDED_

#include <lwda_fp16.hpp>
#include "lwphy_internal.h"
#include "tensor_desc.hpp"

// clang-format off

////////////////////////////////////////////////////////////////////////
// TYPE COLWERSIONS
// Note: No range checking is performed for narrowing colwersions here.

////////////////////////////////////////////////////////////////////////
// Type colwersions from signed char (int8)
template <typename T> T           LWDA_BOTH_INLINE type_colwert(signed char s);
template <>           signed char LWDA_BOTH_INLINE type_colwert(signed char s) { return s; }
template <>           short       LWDA_BOTH_INLINE type_colwert(signed char s) { return static_cast<short>(s); }
template <>           int         LWDA_BOTH_INLINE type_colwert(signed char s) { return static_cast<int>(s); }
template <>           __half      LWDA_BOTH_INLINE type_colwert(signed char s) { return __float2half(static_cast<float>(s)); }
template <>           float       LWDA_BOTH_INLINE type_colwert(signed char s) { return static_cast<float>(s); }
template <>           double      LWDA_BOTH_INLINE type_colwert(signed char s) { return static_cast<double>(s); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from char2 (int8 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(char2 s);
template <>           char2           LWDA_BOTH_INLINE type_colwert(char2 s) { return s; }
template <>           short2          LWDA_BOTH_INLINE type_colwert(char2 s) { short2 r; r.x = static_cast<short>(s.x); r.y = static_cast<short>(s.y); return r; }
template <>           int2            LWDA_BOTH_INLINE type_colwert(char2 s) { int2 r; r.x = static_cast<int>(s.x); r.y = static_cast<int>(s.y); return r; }
template <>           __half2         LWDA_BOTH_INLINE type_colwert(char2 s) { return __floats2half2_rn(static_cast<float>(s.x), static_cast<float>(s.y)); }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(char2 s) { return make_lwComplex(static_cast<float>(s.x), static_cast<float>(s.y)); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(char2 s) { return make_lwDoubleComplex(static_cast<double>(s.x), static_cast<double>(s.y)); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from unsigned char (uint8)
template <typename T> T              LWDA_BOTH_INLINE type_colwert(unsigned char u);
template <>           unsigned char  LWDA_BOTH_INLINE type_colwert(unsigned char u) { return u; }
template <>           unsigned short LWDA_BOTH_INLINE type_colwert(unsigned char u) { return static_cast<unsigned short>(u); }
template <>           unsigned int   LWDA_BOTH_INLINE type_colwert(unsigned char u) { return static_cast<unsigned int>(u); }
template <>           __half         LWDA_BOTH_INLINE type_colwert(unsigned char u) { return __float2half(static_cast<float>(u)); }
template <>           float          LWDA_BOTH_INLINE type_colwert(unsigned char u) { return static_cast<float>(u); }
template <>           double         LWDA_BOTH_INLINE type_colwert(unsigned char u) { return static_cast<double>(u); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from uchar2 (uint8 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(uchar2 u);
template <>           uchar2          LWDA_BOTH_INLINE type_colwert(uchar2 u) { return u; }
template <>           ushort2         LWDA_BOTH_INLINE type_colwert(uchar2 u) { ushort2 r; r.x = static_cast<unsigned short>(u.x); r.y = static_cast<unsigned short>(u.y); return r; }
template <>           uint2           LWDA_BOTH_INLINE type_colwert(uchar2 u) { uint2 r; r.x = static_cast<unsigned int>(u.x); r.y = static_cast<unsigned int>(u.y); return r; }
template <>           __half2         LWDA_BOTH_INLINE type_colwert(uchar2 u) { return __floats2half2_rn(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(uchar2 u) { return make_lwComplex(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(uchar2 u) { return make_lwDoubleComplex(static_cast<double>(u.x), static_cast<double>(u.y)); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from short (int16)
template <typename T> T      LWDA_BOTH_INLINE type_colwert(short s);
template <>           short  LWDA_BOTH_INLINE type_colwert(short s) { return s; }
template <>           int    LWDA_BOTH_INLINE type_colwert(short s) { return static_cast<int>(s); }
template <>           float  LWDA_BOTH_INLINE type_colwert(short s) { return static_cast<float>(s); }
template <>           double LWDA_BOTH_INLINE type_colwert(short s) { return static_cast<double>(s); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from short2 (int16 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(short2 s);
template <>           short2          LWDA_BOTH_INLINE type_colwert(short2 s) { return s; }
template <>           int2            LWDA_BOTH_INLINE type_colwert(short2 s) { int2 r; r.x = static_cast<int>(s.x); r.y = static_cast<int>(s.y); return r; }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(short2 s) { return make_lwComplex(static_cast<float>(s.x), static_cast<float>(s.y)); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(short2 s) { return make_lwDoubleComplex(static_cast<double>(s.x), static_cast<double>(s.y)); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from unsigned short (uint16)
template <typename T> T              LWDA_BOTH_INLINE type_colwert(unsigned short u);
template <>           unsigned short LWDA_BOTH_INLINE type_colwert(unsigned short u) { return u; }
template <>           unsigned int   LWDA_BOTH_INLINE type_colwert(unsigned short u) { return static_cast<unsigned int>(u); }
template <>           float          LWDA_BOTH_INLINE type_colwert(unsigned short u) { return static_cast<float>(u); }
template <>           double         LWDA_BOTH_INLINE type_colwert(unsigned short u) { return static_cast<double>(u); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from ushort2 (uint16 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(ushort2 u);
template <>           ushort2         LWDA_BOTH_INLINE type_colwert(ushort2 u) { return u; }
template <>           uint2           LWDA_BOTH_INLINE type_colwert(ushort2 u) { uint2 r; r.x = static_cast<unsigned int>(u.x); r.y = static_cast<unsigned int>(u.y); return r; }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(ushort2 u) { return make_lwComplex(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(ushort2 u) { return make_lwDoubleComplex(static_cast<double>(u.x), static_cast<double>(u.y)); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from int (int32)
template <typename T> T      LWDA_BOTH_INLINE type_colwert(int i);
template <>           int    LWDA_BOTH_INLINE type_colwert(int i) { return i; }
template <>           double LWDA_BOTH_INLINE type_colwert(int i) { return static_cast<double>(i); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from int2 (int32 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(int2 i);
template <>           int2            LWDA_BOTH_INLINE type_colwert(int2 i) { return i; }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(int2 i) { return make_lwComplex(static_cast<float>(i.x), static_cast<float>(i.y)); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(int2 i) { return make_lwDoubleComplex(static_cast<double>(i.x), static_cast<double>(i.y)); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from uint (uint32)
template <typename T> T      LWDA_BOTH_INLINE type_colwert(unsigned int u);
template <>           uint   LWDA_BOTH_INLINE type_colwert(unsigned int u) { return u; }
template <>           double LWDA_BOTH_INLINE type_colwert(unsigned int u) { return static_cast<double>(u); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from uint2 (uint32 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(uint2 u);
template <>           uint2           LWDA_BOTH_INLINE type_colwert(uint2 u) { return u; }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(uint2 u) { return make_lwComplex(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(uint2 u) { return make_lwDoubleComplex(static_cast<double>(u.x), static_cast<double>(u.y)); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from __half (fp16)
template <typename T> T      LWDA_BOTH_INLINE type_colwert(__half h);
template <>           __half LWDA_BOTH_INLINE type_colwert(__half h) { return h; }
template <>           float  LWDA_BOTH_INLINE type_colwert(__half h) { return __half2float(h); }
template <>           double LWDA_BOTH_INLINE type_colwert(__half h) { return static_cast<double>(__half2float(h)); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from __half2 (fp16 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(__half2 h);
template <>           __half2         LWDA_BOTH_INLINE type_colwert(__half2 h) { return h; }
#if defined(__LWDACC__)
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(__half2 h) { return __half22float2(h); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(__half2 h) { float2 c = __half22float2(h); return lwComplexFloatToDouble(c); }
#else
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(__half2 h) { return make_lwComplex(__half2float(h.x), __half2float(h.y)); }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(__half2 h) { return lwComplexFloatToDouble(type_colwert<lwComplex>(h)); }
#endif // defined(__LWDACC__)

////////////////////////////////////////////////////////////////////////
// Type colwersions from float (fp32)
template <typename T> T      LWDA_BOTH_INLINE type_colwert(float f);
template <>           float  LWDA_BOTH_INLINE type_colwert(float f) { return f; }
template <>           double LWDA_BOTH_INLINE type_colwert(float f) { return static_cast<double>(f); }
template <>           __half LWDA_BOTH_INLINE type_colwert(float f) { return __float2half(f); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from lwComplex (fp32 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(lwComplex c);
template <>           __half2         LWDA_BOTH_INLINE type_colwert(lwComplex c) { return __floats2half2_rn(c.x, c.y); }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(lwComplex c) { return c; }
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(lwComplex c) { return lwComplexFloatToDouble(c); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from double (fp64)
template <typename T> T      LWDA_BOTH_INLINE type_colwert(double d);
template <>           double LWDA_BOTH_INLINE type_colwert(double d) { return d; }
template <>           float  LWDA_BOTH_INLINE type_colwert(double d) { return static_cast<float>(d); }

////////////////////////////////////////////////////////////////////////
// Type colwersions from lwDoubleComplex (fp64 complex)
template <typename T> T               LWDA_BOTH_INLINE type_colwert(lwDoubleComplex d);
template <>           lwDoubleComplex LWDA_BOTH_INLINE type_colwert(lwDoubleComplex d) { return d; }
template <>           lwComplex       LWDA_BOTH_INLINE type_colwert(lwDoubleComplex d) { return lwComplexDoubleToFloat(d); }

// clang-format on

#endif // !defined(TYPE_COLWERT_H_INCLUDED_)
