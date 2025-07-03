/*
 * Copyright 1993-2014 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__TEXTURE_FETCH_FUNCTIONS_HPP__)
#define __TEXTURE_FETCH_FUNCTIONS_HPP__

#if defined(__LWDACC_RTC__)
#define __TEXTURE_FUNCTIONS_DECL__ __device__
#else /* !__LWDACC_RTC__ */
#define __TEXTURE_FUNCTIONS_DECL__ static __forceinline__ __device__
#endif /* !__LWDACC_RTC__ */

#if defined(__cplusplus) && defined(__LWDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "lwda_texture_types.h"
#include "host_defines.h"
#include "texture_types.h"
#include "vector_functions.h"
#include "vector_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex1Dfetch(texture<char, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchi(t, make_int4(x, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex1Dfetch(texture<signed char, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1Dfetch(texture<unsigned char, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex1Dfetch(texture<char1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1Dfetch(texture<uchar1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex1Dfetch(texture<char2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1Dfetch(texture<uchar2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex1Dfetch(texture<char4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1Dfetch(texture<uchar4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex1Dfetch(texture<short, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1Dfetch(texture<unsigned short, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex1Dfetch(texture<short1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1Dfetch(texture<ushort1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex1Dfetch(texture<short2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1Dfetch(texture<ushort2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex1Dfetch(texture<short4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1Dfetch(texture<ushort4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex1Dfetch(texture<int, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1Dfetch(texture<unsigned int, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex1Dfetch(texture<int1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex1Dfetch(texture<uint1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex1Dfetch(texture<int2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex1Dfetch(texture<uint2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex1Dfetch(texture<int4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex1Dfetch(texture<uint4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex1Dfetch(texture<long, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex1Dfetch(texture<unsigned long, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex1Dfetch(texture<long1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex1Dfetch(texture<ulong1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex1Dfetch(texture<long2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex1Dfetch(texture<ulong2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex1Dfetch(texture<long4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  int4 v = __itexfetchi(t, make_int4(x, 0, 0, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex1Dfetch(texture<ulong4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  uint4 v = __utexfetchi(t, make_int4(x, 0, 0, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<float, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  float4 v = __ftexfetchi(t, make_int4(x, 0, 0, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1Dfetch(texture<float1, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  float4 v = __ftexfetchi(t, make_int4(x, 0, 0, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1Dfetch(texture<float2, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  float4 v = __ftexfetchi(t, make_int4(x, 0, 0, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1Dfetch(texture<float4, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
  float4 v = __ftexfetchi(t, make_int4(x, 0, 0, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<signed char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<unsigned char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1Dfetch(texture<char1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1Dfetch(texture<uchar1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1Dfetch(texture<char2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1Dfetch(texture<uchar2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1Dfetch(texture<char4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1Dfetch(texture<uchar4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1Dfetch(texture<unsigned short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1Dfetch(texture<short1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1Dfetch(texture<ushort1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1Dfetch(texture<short2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1Dfetch(texture<ushort2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v  = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1Dfetch(texture<short4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  int4 v   = __itexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1Dfetch(texture<ushort4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x)
{
  uint4 v   = __utexfetchi(t, make_int4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex1D(texture<char, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetch(t, make_float4(x, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex1D(texture<signed char, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1D(texture<unsigned char, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex1D(texture<char1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1D(texture<uchar1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex1D(texture<char2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1D(texture<uchar2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex1D(texture<char4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1D(texture<uchar4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex1D(texture<short, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1D(texture<unsigned short, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex1D(texture<short1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1D(texture<ushort1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex1D(texture<short2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1D(texture<ushort2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex1D(texture<short4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1D(texture<ushort4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex1D(texture<int, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1D(texture<unsigned int, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex1D(texture<int1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex1D(texture<uint1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex1D(texture<int2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex1D(texture<uint2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex1D(texture<int4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex1D(texture<uint4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex1D(texture<long, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex1D(texture<unsigned long, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex1D(texture<long1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex1D(texture<ulong1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex1D(texture<long2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex1D(texture<ulong2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex1D(texture<long4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  int4 v = __itexfetch(t, make_float4(x, 0, 0, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex1D(texture<ulong4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  uint4 v = __utexfetch(t, make_float4(x, 0, 0, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<float, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  float4 v = __ftexfetch(t, make_float4(x, 0, 0, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1D(texture<float1, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  float4 v = __ftexfetch(t, make_float4(x, 0, 0, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1D(texture<float2, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  float4 v = __ftexfetch(t, make_float4(x, 0, 0, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1D(texture<float4, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{
  float4 v = __ftexfetch(t, make_float4(x, 0, 0, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<signed char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<unsigned char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1D(texture<char1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1D(texture<uchar1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1D(texture<char2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1D(texture<uchar2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1D(texture<char4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1D(texture<uchar4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1D(texture<unsigned short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1D(texture<short1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1D(texture<ushort1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1D(texture<short2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1D(texture<ushort2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v  = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1D(texture<short4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  int4 v   = __itexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1D(texture<ushort4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{
  uint4 v   = __utexfetch(t, make_float4(x, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 2D Texture functions                                                         *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex2D(texture<char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetch(t, make_float4(x, y, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex2D(texture<signed char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2D(texture<unsigned char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex2D(texture<char1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2D(texture<uchar1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex2D(texture<char2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2D(texture<uchar2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2D(texture<char4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2D(texture<uchar4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex2D(texture<short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2D(texture<unsigned short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex2D(texture<short1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2D(texture<ushort1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex2D(texture<short2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2D(texture<ushort2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2D(texture<short4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2D(texture<ushort4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex2D(texture<int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2D(texture<unsigned int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex2D(texture<int1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex2D(texture<uint1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex2D(texture<int2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex2D(texture<uint2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2D(texture<int4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2D(texture<uint4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex2D(texture<long, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex2D(texture<unsigned long, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex2D(texture<long1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex2D(texture<ulong1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex2D(texture<long2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex2D(texture<ulong2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex2D(texture<long4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  int4 v = __itexfetch(t, make_float4(x, y, 0, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex2D(texture<ulong4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  uint4 v = __utexfetch(t, make_float4(x, y, 0, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<float, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  float4 v = __ftexfetch(t, make_float4(x, y, 0, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2D(texture<float1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  float4 v = __ftexfetch(t, make_float4(x, y, 0, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2D(texture<float2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  float4 v = __ftexfetch(t, make_float4(x, y, 0, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2D(texture<float4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{
  float4 v = __ftexfetch(t, make_float4(x, y, 0, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<signed char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<unsigned char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2D(texture<char1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2D(texture<uchar1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2D(texture<char2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2D(texture<uchar2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2D(texture<char4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2D(texture<uchar4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2D(texture<unsigned short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2D(texture<short1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2D(texture<ushort1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2D(texture<short2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2D(texture<ushort2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2D(texture<short4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  int4 v   = __itexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2D(texture<ushort4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{
  uint4 v   = __utexfetch(t, make_float4(x, y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 1D Layered Texture functions                                                 *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex1DLayered(texture<char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex1DLayered(texture<signed char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayered(texture<unsigned char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayered(texture<char1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayered(texture<uchar1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayered(texture<char2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayered(texture<uchar2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayered(texture<char4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayered(texture<uchar4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex1DLayered(texture<short, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayered(texture<unsigned short, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayered(texture<short1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayered(texture<ushort1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayered(texture<short2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayered(texture<ushort2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayered(texture<short4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayered(texture<ushort4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex1DLayered(texture<int, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayered(texture<unsigned int, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayered(texture<int1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayered(texture<uint1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayered(texture<int2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayered(texture<uint2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayered(texture<int4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayered(texture<uint4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex1DLayered(texture<long, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex1DLayered(texture<unsigned long, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex1DLayered(texture<long1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex1DLayered(texture<ulong1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex1DLayered(texture<long2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex1DLayered(texture<ulong2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex1DLayered(texture<long4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex1DLayered(texture<ulong4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<float, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayered(texture<float1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayered(texture<float2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayered(texture<float4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, 0, 0, 0), layer);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<signed char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<unsigned char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayered(texture<char1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayered(texture<uchar1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayered(texture<char2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayered(texture<uchar2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayered(texture<char4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayered(texture<uchar4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<short, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayered(texture<unsigned short, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayered(texture<short1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayered(texture<ushort1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayered(texture<short2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayered(texture<ushort2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayered(texture<short4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayered(texture<ushort4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{
  uint4 v   = __utexfetchl(t, make_float4(x, 0, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 2D Layered Texture functions                                                 *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex2DLayered(texture<char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex2DLayered(texture<signed char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayered(texture<unsigned char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayered(texture<char1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayered(texture<uchar1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayered(texture<char2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayered(texture<uchar2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayered(texture<char4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayered(texture<uchar4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex2DLayered(texture<short, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayered(texture<unsigned short, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayered(texture<short1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayered(texture<ushort1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayered(texture<short2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayered(texture<ushort2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayered(texture<short4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayered(texture<ushort4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex2DLayered(texture<int, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayered(texture<unsigned int, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayered(texture<int1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayered(texture<uint1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayered(texture<int2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayered(texture<uint2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayered(texture<int4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayered(texture<uint4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex2DLayered(texture<long, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex2DLayered(texture<unsigned long, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex2DLayered(texture<long1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex2DLayered(texture<ulong1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex2DLayered(texture<long2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex2DLayered(texture<ulong2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex2DLayered(texture<long4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  int4 v = __itexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex2DLayered(texture<ulong4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  uint4 v = __utexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<float, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, y, 0, 0), layer);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayered(texture<float1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayered(texture<float2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayered(texture<float4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{
  float4 v = __ftexfetchl(t, make_float4(x, y, 0, 0), layer);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<signed char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<unsigned char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayered(texture<char1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayered(texture<uchar1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayered(texture<char2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayered(texture<uchar2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayered(texture<char4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayered(texture<uchar4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<short, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayered(texture<unsigned short, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayered(texture<short1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayered(texture<ushort1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayered(texture<short2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayered(texture<ushort2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v  = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayered(texture<short4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  int4 v   = __itexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayered(texture<ushort4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{
  uint4 v   = __utexfetchl(t, make_float4(x, y, 0, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 3D Texture functions                                                         *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex3D(texture<char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetch(t, make_float4(x, y, z, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex3D(texture<signed char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3D(texture<unsigned char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex3D(texture<char1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3D(texture<uchar1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex3D(texture<char2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3D(texture<uchar2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex3D(texture<char4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3D(texture<uchar4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex3D(texture<short, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex3D(texture<unsigned short, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex3D(texture<short1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex3D(texture<ushort1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex3D(texture<short2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex3D(texture<ushort2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex3D(texture<short4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex3D(texture<ushort4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex3D(texture<int, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3D(texture<unsigned int, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex3D(texture<int1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex3D(texture<uint1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex3D(texture<int2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex3D(texture<uint2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex3D(texture<int4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex3D(texture<uint4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex3D(texture<long, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex3D(texture<unsigned long, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex3D(texture<long1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex3D(texture<ulong1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex3D(texture<long2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex3D(texture<ulong2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex3D(texture<long4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetch(t, make_float4(x, y, z, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex3D(texture<ulong4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetch(t, make_float4(x, y, z, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<float, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetch(t, make_float4(x, y, z, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3D(texture<float1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetch(t, make_float4(x, y, z, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3D(texture<float2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetch(t, make_float4(x, y, z, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3D(texture<float4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetch(t, make_float4(x, y, z, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<signed char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<unsigned char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3D(texture<char1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3D(texture<uchar1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3D(texture<char2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3D(texture<uchar2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3D(texture<char4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3D(texture<uchar4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<short, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3D(texture<unsigned short, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3D(texture<short1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3D(texture<ushort1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3D(texture<short2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3D(texture<ushort2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3D(texture<short4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3D(texture<ushort4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v   = __utexfetch(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap Texture functions                                                    *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char texLwbemap(texture<char, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchc(t, make_float4(x, y, z, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char texLwbemap(texture<signed char, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char texLwbemap(texture<unsigned char, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 texLwbemap(texture<char1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 texLwbemap(texture<uchar1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 texLwbemap(texture<char2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 texLwbemap(texture<uchar2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 texLwbemap(texture<char4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 texLwbemap(texture<uchar4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short texLwbemap(texture<short, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short texLwbemap(texture<unsigned short, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 texLwbemap(texture<short1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 texLwbemap(texture<ushort1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 texLwbemap(texture<short2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 texLwbemap(texture<ushort2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 texLwbemap(texture<short4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 texLwbemap(texture<ushort4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int texLwbemap(texture<int, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int texLwbemap(texture<unsigned int, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 texLwbemap(texture<int1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 texLwbemap(texture<uint1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 texLwbemap(texture<int2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 texLwbemap(texture<uint2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 texLwbemap(texture<int4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 texLwbemap(texture<uint4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long texLwbemap(texture<long, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long texLwbemap(texture<unsigned long, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 texLwbemap(texture<long1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 texLwbemap(texture<ulong1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 texLwbemap(texture<long2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 texLwbemap(texture<ulong2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 texLwbemap(texture<long4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  int4 v = __itexfetchc(t, make_float4(x, y, z, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 texLwbemap(texture<ulong4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  uint4 v = __utexfetchc(t, make_float4(x, y, z, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemap(texture<float, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetchc(t, make_float4(x, y, z, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemap(texture<float1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetchc(t, make_float4(x, y, z, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemap(texture<float2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetchc(t, make_float4(x, y, z, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemap(texture<float4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{
  float4 v = __ftexfetchc(t, make_float4(x, y, z, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemap(texture<char, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemap(texture<signed char, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemap(texture<unsigned char, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemap(texture<char1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemap(texture<uchar1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemap(texture<char2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemap(texture<uchar2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemap(texture<char4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemap(texture<uchar4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemap(texture<short, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemap(texture<unsigned short, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemap(texture<short1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemap(texture<ushort1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemap(texture<short2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemap(texture<ushort2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v  = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemap(texture<short4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  int4 v   = __itexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemap(texture<ushort4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{
  uint4 v   = __utexfetchc(t, make_float4(x, y, z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap Layered Texture functions                                            *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char texLwbemapLayered(texture<char, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char texLwbemapLayered(texture<signed char, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char texLwbemapLayered(texture<unsigned char, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 texLwbemapLayered(texture<char1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 texLwbemapLayered(texture<uchar1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 texLwbemapLayered(texture<char2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 texLwbemapLayered(texture<uchar2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 texLwbemapLayered(texture<char4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 texLwbemapLayered(texture<uchar4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short texLwbemapLayered(texture<short, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short texLwbemapLayered(texture<unsigned short, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 texLwbemapLayered(texture<short1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 texLwbemapLayered(texture<ushort1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 texLwbemapLayered(texture<short2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 texLwbemapLayered(texture<ushort2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 texLwbemapLayered(texture<short4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 texLwbemapLayered(texture<ushort4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int texLwbemapLayered(texture<int, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int texLwbemapLayered(texture<unsigned int, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 texLwbemapLayered(texture<int1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 texLwbemapLayered(texture<uint1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 texLwbemapLayered(texture<int2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 texLwbemapLayered(texture<uint2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 texLwbemapLayered(texture<int4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 texLwbemapLayered(texture<uint4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long texLwbemapLayered(texture<long, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long texLwbemapLayered(texture<unsigned long, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 texLwbemapLayered(texture<long1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 texLwbemapLayered(texture<ulong1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 texLwbemapLayered(texture<long2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 texLwbemapLayered(texture<ulong2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 texLwbemapLayered(texture<long4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  int4 v = __itexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 texLwbemapLayered(texture<ulong4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  uint4 v = __utexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayered(texture<float, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  float4 v = __ftexfetchlc(t, make_float4(x, y, z, 0), layer);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayered(texture<float1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  float4 v = __ftexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayered(texture<float2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  float4 v = __ftexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayered(texture<float4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
  float4 v = __ftexfetchlc(t, make_float4(x, y, z, 0), layer);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayered(texture<char, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayered(texture<signed char, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayered(texture<unsigned char, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayered(texture<char1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayered(texture<uchar1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayered(texture<char2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayered(texture<uchar2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayered(texture<char4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayered(texture<uchar4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayered(texture<short, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayered(texture<unsigned short, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayered(texture<short1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayered(texture<ushort1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayered(texture<short2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayered(texture<ushort2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v  = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayered(texture<short4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  int4 v   = __itexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayered(texture<ushort4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
  uint4 v   = __utexfetchlc(t, make_float4(x, y, z, 0), layer);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

#endif /* __cplusplus && __LWDACC__ */

#if defined(__cplusplus) && defined(__LWDACC__)

#if !defined(__LWDA_ARCH__) || __LWDA_ARCH__ >= 200

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/


#define __tex2DgatherUtil(T, f, r, c) \
        { T v = f<c>(t, make_float2(x, y)); return r; }

#define __tex2DgatherUtil1(T, f, r) \
        __tex2DgatherUtil(T, f, r, 0)

#define __tex2DgatherUtil2(T, f, r)                  \
        if (comp == 1) __tex2DgatherUtil(T, f, r, 1) \
        else __tex2DgatherUtil1(T, f, r)

#define __tex2DgatherUtil3(T, f, r)                  \
        if (comp == 2) __tex2DgatherUtil(T, f, r, 2) \
        else __tex2DgatherUtil2(T, f, r)

#define __tex2DgatherUtil4(T, f, r)                  \
        if (comp == 3) __tex2DgatherUtil(T, f, r, 3) \
        else __tex2DgatherUtil3(T, f, r)

__TEXTURE_FUNCTIONS_DECL__ char4 tex2Dgather(texture<char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_char4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2Dgather(texture<signed char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_char4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2Dgather(texture<unsigned char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_uchar4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2Dgather(texture<char1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_char4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2Dgather(texture<uchar1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_uchar4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2Dgather(texture<char2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(int4, __itex2Dgather, make_char4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2Dgather(texture<uchar2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(uint4, __utex2Dgather, make_uchar4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2Dgather(texture<char3, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(int4, __itex2Dgather, make_char4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2Dgather(texture<uchar3, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(uint4, __utex2Dgather, make_uchar4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2Dgather(texture<char4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(int4, __itex2Dgather, make_char4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2Dgather(texture<uchar4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(uint4, __utex2Dgather, make_uchar4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2Dgather(texture<signed short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_short4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2Dgather(texture<unsigned short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_ushort4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2Dgather(texture<short1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_short4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2Dgather(texture<ushort1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_ushort4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2Dgather(texture<short2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(int4, __itex2Dgather, make_short4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2Dgather(texture<ushort2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(uint4, __utex2Dgather, make_ushort4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2Dgather(texture<short3, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(int4, __itex2Dgather, make_short4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2Dgather(texture<ushort3, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(uint4, __utex2Dgather, make_ushort4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2Dgather(texture<short4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(int4, __itex2Dgather, make_short4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2Dgather(texture<ushort4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(uint4, __utex2Dgather, make_ushort4(v.x, v.y, v.z, v.w));
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2Dgather(texture<signed int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2Dgather(texture<unsigned int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2Dgather(texture<int1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2Dgather(texture<uint1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2Dgather(texture<int2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(int4, __itex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2Dgather(texture<uint2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(uint4, __utex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2Dgather(texture<int3, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(int4, __itex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2Dgather(texture<uint3, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(uint4, __utex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2Dgather(texture<int4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(int4, __itex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2Dgather(texture<uint4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(uint4, __utex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<float, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(float4, __ftex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<float1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(float4, __ftex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<float2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(float4, __ftex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<float3, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(float4, __ftex2Dgather, v);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<float4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(float4, __ftex2Dgather, v);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/


__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<signed char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<unsigned char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<char1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<uchar1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<char2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<uchar2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<char3, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<uchar3, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<char4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<uchar4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<signed short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<unsigned short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<short1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<ushort1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil1(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<short2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<ushort2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil2(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<short3, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<ushort3, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil3(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<short4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(int4, __itex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2Dgather(texture<ushort4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp)
{
  __tex2DgatherUtil4(uint4, __utex2Dgather, make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w)));
}

#undef __tex2DgatherUtil
#undef __tex2DgatherUtil1
#undef __tex2DgatherUtil2
#undef __tex2DgatherUtil3
#undef __tex2DgatherUtil4

/*******************************************************************************
*                                                                              *
* 1D Mipmapped Texture functions                                               *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex1DLod(texture<char, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex1DLod(texture<signed char, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLod(texture<unsigned char, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLod(texture<char1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLod(texture<uchar1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLod(texture<char2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLod(texture<uchar2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLod(texture<char4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLod(texture<uchar4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex1DLod(texture<short, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLod(texture<unsigned short, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLod(texture<short1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLod(texture<ushort1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLod(texture<short2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLod(texture<ushort2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLod(texture<short4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLod(texture<ushort4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex1DLod(texture<int, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLod(texture<unsigned int, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLod(texture<int1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLod(texture<uint1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLod(texture<int2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLod(texture<uint2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLod(texture<int4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLod(texture<uint4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_uint4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex1DLod(texture<long, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex1DLod(texture<unsigned long, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex1DLod(texture<long1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex1DLod(texture<ulong1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex1DLod(texture<long2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex1DLod(texture<ulong2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex1DLod(texture<long4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex1DLod(texture<ulong4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<float, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLod(texture<float1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLod(texture<float2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLod(texture<float4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, 0, 0, 0), level);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<signed char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<unsigned char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLod(texture<char1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLod(texture<uchar1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLod(texture<char2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLod(texture<uchar2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLod(texture<char4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLod(texture<uchar4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLod(texture<unsigned short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLod(texture<short1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLod(texture<ushort1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLod(texture<short2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLod(texture<ushort2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLod(texture<short4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLod(texture<ushort4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{
  uint4 v   = __utexfetchlod(t, make_float4(x, 0, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 2D Mipmapped Texture functions                                               *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex2DLod(texture<char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex2DLod(texture<signed char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLod(texture<unsigned char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLod(texture<char1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLod(texture<uchar1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLod(texture<char2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLod(texture<uchar2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLod(texture<char4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLod(texture<uchar4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex2DLod(texture<short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLod(texture<unsigned short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLod(texture<short1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLod(texture<ushort1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLod(texture<short2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLod(texture<ushort2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLod(texture<short4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLod(texture<ushort4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex2DLod(texture<int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLod(texture<unsigned int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLod(texture<int1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLod(texture<uint1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLod(texture<int2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLod(texture<uint2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLod(texture<int4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLod(texture<uint4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex2DLod(texture<long, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex2DLod(texture<unsigned long, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex2DLod(texture<long1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex2DLod(texture<ulong1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex2DLod(texture<long2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex2DLod(texture<ulong2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex2DLod(texture<long4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex2DLod(texture<ulong4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<float, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, 0, 0), level);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLod(texture<float1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLod(texture<float2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLod(texture<float4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, 0, 0), level);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<signed char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<unsigned char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLod(texture<char1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLod(texture<uchar1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLod(texture<char2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLod(texture<uchar2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLod(texture<char4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLod(texture<uchar4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLod(texture<unsigned short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLod(texture<short1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLod(texture<ushort1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLod(texture<short2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLod(texture<ushort2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLod(texture<short4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLod(texture<ushort4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{
  uint4 v   = __utexfetchlod(t, make_float4(x, y, 0, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 1D Layered Mipmapped Texture functions                                       *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex1DLayeredLod(texture<char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex1DLayeredLod(texture<signed char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayeredLod(texture<unsigned char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayeredLod(texture<char1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayeredLod(texture<uchar1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayeredLod(texture<char2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayeredLod(texture<uchar2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayeredLod(texture<char4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayeredLod(texture<uchar4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex1DLayeredLod(texture<short, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayeredLod(texture<unsigned short, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayeredLod(texture<short1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayeredLod(texture<ushort1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayeredLod(texture<short2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayeredLod(texture<ushort2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayeredLod(texture<short4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayeredLod(texture<ushort4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex1DLayeredLod(texture<int, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayeredLod(texture<unsigned int, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayeredLod(texture<int1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayeredLod(texture<uint1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayeredLod(texture<int2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayeredLod(texture<uint2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayeredLod(texture<int4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayeredLod(texture<uint4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex1DLayeredLod(texture<long, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex1DLayeredLod(texture<unsigned long, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex1DLayeredLod(texture<long1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex1DLayeredLod(texture<ulong1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex1DLayeredLod(texture<long2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex1DLayeredLod(texture<ulong2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex1DLayeredLod(texture<long4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex1DLayeredLod(texture<ulong4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<float, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredLod(texture<float1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredLod(texture<float2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredLod(texture<float4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<signed char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<unsigned char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredLod(texture<char1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredLod(texture<uchar1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredLod(texture<char2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredLod(texture<uchar2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredLod(texture<char4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredLod(texture<uchar4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<short, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredLod(texture<unsigned short, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredLod(texture<short1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredLod(texture<ushort1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredLod(texture<short2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredLod(texture<ushort2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredLod(texture<short4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredLod(texture<ushort4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{
  uint4 v   = __utexfetchlodl(t, make_float4(x, 0, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 2D Layered Mipmapped Texture functions                                       *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex2DLayeredLod(texture<char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex2DLayeredLod(texture<signed char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayeredLod(texture<unsigned char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayeredLod(texture<char1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayeredLod(texture<uchar1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayeredLod(texture<char2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayeredLod(texture<uchar2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayeredLod(texture<char4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayeredLod(texture<uchar4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex2DLayeredLod(texture<short, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayeredLod(texture<unsigned short, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayeredLod(texture<short1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayeredLod(texture<ushort1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayeredLod(texture<short2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayeredLod(texture<ushort2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayeredLod(texture<short4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayeredLod(texture<ushort4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex2DLayeredLod(texture<int, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayeredLod(texture<unsigned int, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayeredLod(texture<int1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayeredLod(texture<uint1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayeredLod(texture<int2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayeredLod(texture<uint2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayeredLod(texture<int4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayeredLod(texture<uint4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex2DLayeredLod(texture<long, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex2DLayeredLod(texture<unsigned long, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex2DLayeredLod(texture<long1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex2DLayeredLod(texture<ulong1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex2DLayeredLod(texture<long2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex2DLayeredLod(texture<ulong2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex2DLayeredLod(texture<long4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  int4 v = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex2DLayeredLod(texture<ulong4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  uint4 v = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<float, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredLod(texture<float1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredLod(texture<float2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredLod(texture<float4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{
  float4 v = __ftexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<signed char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<unsigned char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredLod(texture<char1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredLod(texture<uchar1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredLod(texture<char2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredLod(texture<uchar2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredLod(texture<char4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredLod(texture<uchar4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<short, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredLod(texture<unsigned short, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredLod(texture<short1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredLod(texture<ushort1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredLod(texture<short2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredLod(texture<ushort2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v  = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredLod(texture<short4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  int4 v   = __itexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredLod(texture<ushort4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{
  uint4 v   = __utexfetchlodl(t, make_float4(x, y, 0, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 3D Mipmapped Texture functions                                               *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex3DLod(texture<char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlod(t, make_float4(x, y, z, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex3DLod(texture<signed char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3DLod(texture<unsigned char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex3DLod(texture<char1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3DLod(texture<uchar1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex3DLod(texture<char2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3DLod(texture<uchar2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex3DLod(texture<char4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3DLod(texture<uchar4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex3DLod(texture<short, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex3DLod(texture<unsigned short, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex3DLod(texture<short1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex3DLod(texture<ushort1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex3DLod(texture<short2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex3DLod(texture<ushort2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex3DLod(texture<short4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex3DLod(texture<ushort4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex3DLod(texture<int, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3DLod(texture<unsigned int, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex3DLod(texture<int1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex3DLod(texture<uint1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex3DLod(texture<int2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex3DLod(texture<uint2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex3DLod(texture<int4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex3DLod(texture<uint4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex3DLod(texture<long, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex3DLod(texture<unsigned long, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex3DLod(texture<long1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex3DLod(texture<ulong1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex3DLod(texture<long2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex3DLod(texture<ulong2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex3DLod(texture<long4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex3DLod(texture<ulong4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<float, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, z, 0), level);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DLod(texture<float1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DLod(texture<float2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DLod(texture<float4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlod(t, make_float4(x, y, z, 0), level);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<signed char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<unsigned char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DLod(texture<char1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DLod(texture<uchar1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DLod(texture<char2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DLod(texture<uchar2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DLod(texture<char4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DLod(texture<uchar4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<short, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3DLod(texture<unsigned short, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DLod(texture<short1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DLod(texture<ushort1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DLod(texture<short2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DLod(texture<ushort2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DLod(texture<short4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DLod(texture<ushort4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v   = __utexfetchlod(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap Mipmapped Texture functions                                          *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char texLwbemapLod(texture<char, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char texLwbemapLod(texture<signed char, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char texLwbemapLod(texture<unsigned char, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 texLwbemapLod(texture<char1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 texLwbemapLod(texture<uchar1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 texLwbemapLod(texture<char2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 texLwbemapLod(texture<uchar2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 texLwbemapLod(texture<char4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 texLwbemapLod(texture<uchar4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short texLwbemapLod(texture<short, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short texLwbemapLod(texture<unsigned short, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 texLwbemapLod(texture<short1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 texLwbemapLod(texture<ushort1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 texLwbemapLod(texture<short2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 texLwbemapLod(texture<ushort2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 texLwbemapLod(texture<short4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 texLwbemapLod(texture<ushort4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int texLwbemapLod(texture<int, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int texLwbemapLod(texture<unsigned int, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 texLwbemapLod(texture<int1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 texLwbemapLod(texture<uint1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 texLwbemapLod(texture<int2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 texLwbemapLod(texture<uint2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 texLwbemapLod(texture<int4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 texLwbemapLod(texture<uint4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long texLwbemapLod(texture<long, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long texLwbemapLod(texture<unsigned long, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 texLwbemapLod(texture<long1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 texLwbemapLod(texture<ulong1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 texLwbemapLod(texture<long2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 texLwbemapLod(texture<ulong2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 texLwbemapLod(texture<long4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  int4 v = __itexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 texLwbemapLod(texture<ulong4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  uint4 v = __utexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLod(texture<float, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlodc(t, make_float4(x, y, z, 0), level);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLod(texture<float1, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLod(texture<float2, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLod(texture<float4, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{
  float4 v = __ftexfetchlodc(t, make_float4(x, y, z, 0), level);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLod(texture<char, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLod(texture<signed char, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLod(texture<unsigned char, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLod(texture<char1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLod(texture<uchar1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLod(texture<char2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLod(texture<uchar2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLod(texture<char4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLod(texture<uchar4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLod(texture<short, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLod(texture<unsigned short, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLod(texture<short1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLod(texture<ushort1, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLod(texture<short2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLod(texture<ushort2, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v  = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLod(texture<short4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  int4 v   = __itexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLod(texture<ushort4, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{
  uint4 v   = __utexfetchlodc(t, make_float4(x, y, z, 0), level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap Layered Mipmapped Texture functions                                  *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char texLwbemapLayeredLod(texture<char, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char texLwbemapLayeredLod(texture<signed char, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char texLwbemapLayeredLod(texture<unsigned char, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 texLwbemapLayeredLod(texture<char1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 texLwbemapLayeredLod(texture<uchar1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 texLwbemapLayeredLod(texture<char2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 texLwbemapLayeredLod(texture<uchar2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 texLwbemapLayeredLod(texture<char4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 texLwbemapLayeredLod(texture<uchar4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short texLwbemapLayeredLod(texture<short, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short texLwbemapLayeredLod(texture<unsigned short, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 texLwbemapLayeredLod(texture<short1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 texLwbemapLayeredLod(texture<ushort1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 texLwbemapLayeredLod(texture<short2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 texLwbemapLayeredLod(texture<ushort2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 texLwbemapLayeredLod(texture<short4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 texLwbemapLayeredLod(texture<ushort4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int texLwbemapLayeredLod(texture<int, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int texLwbemapLayeredLod(texture<unsigned int, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 texLwbemapLayeredLod(texture<int1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 texLwbemapLayeredLod(texture<uint1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 texLwbemapLayeredLod(texture<int2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 texLwbemapLayeredLod(texture<uint2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 texLwbemapLayeredLod(texture<int4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 texLwbemapLayeredLod(texture<uint4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long texLwbemapLayeredLod(texture<long, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long texLwbemapLayeredLod(texture<unsigned long, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 texLwbemapLayeredLod(texture<long1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 texLwbemapLayeredLod(texture<ulong1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 texLwbemapLayeredLod(texture<long2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 texLwbemapLayeredLod(texture<ulong2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 texLwbemapLayeredLod(texture<long4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  int4 v = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 texLwbemapLayeredLod(texture<ulong4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  uint4 v = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayeredLod(texture<float, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  float4 v = __ftexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayeredLod(texture<float1, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  float4 v = __ftexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayeredLod(texture<float2, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  float4 v = __ftexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayeredLod(texture<float4, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{
  float4 v = __ftexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayeredLod(texture<char, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayeredLod(texture<signed char, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayeredLod(texture<unsigned char, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayeredLod(texture<char1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayeredLod(texture<uchar1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayeredLod(texture<char2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayeredLod(texture<uchar2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayeredLod(texture<char4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayeredLod(texture<uchar4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayeredLod(texture<short, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float texLwbemapLayeredLod(texture<unsigned short, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayeredLod(texture<short1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 texLwbemapLayeredLod(texture<ushort1, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayeredLod(texture<short2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 texLwbemapLayeredLod(texture<ushort2, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v  = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayeredLod(texture<short4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  int4 v   = __itexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 texLwbemapLayeredLod(texture<ushort4, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{
  uint4 v   = __utexfetchlodlc(t, make_float4(x, y, z, 0), layer, level);
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}


/*******************************************************************************
*                                                                              *
* 1D Gradient Texture functions                                                *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex1DGrad(texture<char, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex1DGrad(texture<signed char, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DGrad(texture<unsigned char, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex1DGrad(texture<char1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DGrad(texture<uchar1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex1DGrad(texture<char2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DGrad(texture<uchar2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex1DGrad(texture<char4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DGrad(texture<uchar4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex1DGrad(texture<short, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DGrad(texture<unsigned short, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex1DGrad(texture<short1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DGrad(texture<ushort1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex1DGrad(texture<short2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DGrad(texture<ushort2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex1DGrad(texture<short4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DGrad(texture<ushort4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex1DGrad(texture<int, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DGrad(texture<unsigned int, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex1DGrad(texture<int1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DGrad(texture<uint1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex1DGrad(texture<int2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DGrad(texture<uint2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex1DGrad(texture<int4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DGrad(texture<uint4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex1DGrad(texture<long, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex1DGrad(texture<unsigned long, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex1DGrad(texture<long1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex1DGrad(texture<ulong1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex1DGrad(texture<long2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex1DGrad(texture<ulong2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex1DGrad(texture<long4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex1DGrad(texture<ulong4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<float, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DGrad(texture<float1, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DGrad(texture<float2, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DGrad(texture<float4, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<signed char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<unsigned char, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DGrad(texture<char1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DGrad(texture<uchar1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DGrad(texture<char2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DGrad(texture<uchar2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DGrad(texture<char4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DGrad(texture<uchar4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DGrad(texture<unsigned short, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DGrad(texture<short1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DGrad(texture<ushort1, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DGrad(texture<short2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DGrad(texture<ushort2, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DGrad(texture<short4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DGrad(texture<ushort4, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{
  uint4 v   = __utexfetchgrad(t, make_float4(x, 0, 0, 0), make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 2D Gradient Texture functions                                                *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex2DGrad(texture<char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex2DGrad(texture<signed char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DGrad(texture<unsigned char, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex2DGrad(texture<char1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DGrad(texture<uchar1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex2DGrad(texture<char2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DGrad(texture<uchar2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2DGrad(texture<char4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DGrad(texture<uchar4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex2DGrad(texture<short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DGrad(texture<unsigned short, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex2DGrad(texture<short1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DGrad(texture<ushort1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex2DGrad(texture<short2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DGrad(texture<ushort2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2DGrad(texture<short4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DGrad(texture<ushort4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex2DGrad(texture<int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DGrad(texture<unsigned int, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex2DGrad(texture<int1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DGrad(texture<uint1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex2DGrad(texture<int2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DGrad(texture<uint2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2DGrad(texture<int4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DGrad(texture<uint4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex2DGrad(texture<long, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex2DGrad(texture<unsigned long, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex2DGrad(texture<long1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex2DGrad(texture<ulong1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex2DGrad(texture<long2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex2DGrad(texture<ulong2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex2DGrad(texture<long4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex2DGrad(texture<ulong4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<float, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DGrad(texture<float1, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DGrad(texture<float2, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DGrad(texture<float4, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<signed char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<unsigned char, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DGrad(texture<char1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DGrad(texture<uchar1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DGrad(texture<char2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DGrad(texture<uchar2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DGrad(texture<char4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DGrad(texture<uchar4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DGrad(texture<unsigned short, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DGrad(texture<short1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DGrad(texture<ushort1, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DGrad(texture<short2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DGrad(texture<ushort2, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DGrad(texture<short4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DGrad(texture<ushort4, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 v   = __utexfetchgrad(t, make_float4(x, y, 0, 0), make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 1D Layered Gradient Texture functions                                        *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex1DLayeredGrad(texture<char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex1DLayeredGrad(texture<signed char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex1DLayeredGrad(texture<unsigned char, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex1DLayeredGrad(texture<char1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex1DLayeredGrad(texture<uchar1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex1DLayeredGrad(texture<char2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex1DLayeredGrad(texture<uchar2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex1DLayeredGrad(texture<char4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex1DLayeredGrad(texture<uchar4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex1DLayeredGrad(texture<short, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex1DLayeredGrad(texture<unsigned short, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex1DLayeredGrad(texture<short1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex1DLayeredGrad(texture<ushort1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex1DLayeredGrad(texture<short2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex1DLayeredGrad(texture<ushort2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex1DLayeredGrad(texture<short4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex1DLayeredGrad(texture<ushort4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex1DLayeredGrad(texture<int, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex1DLayeredGrad(texture<unsigned int, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex1DLayeredGrad(texture<int1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex1DLayeredGrad(texture<uint1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex1DLayeredGrad(texture<int2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex1DLayeredGrad(texture<uint2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex1DLayeredGrad(texture<int4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex1DLayeredGrad(texture<uint4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex1DLayeredGrad(texture<long, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex1DLayeredGrad(texture<unsigned long, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex1DLayeredGrad(texture<long1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex1DLayeredGrad(texture<ulong1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex1DLayeredGrad(texture<long2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex1DLayeredGrad(texture<ulong2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex1DLayeredGrad(texture<long4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex1DLayeredGrad(texture<ulong4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<float, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredGrad(texture<float1, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredGrad(texture<float2, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredGrad(texture<float4, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<signed char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<unsigned char, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredGrad(texture<char1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredGrad(texture<uchar1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredGrad(texture<char2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredGrad(texture<uchar2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredGrad(texture<char4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredGrad(texture<uchar4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<short, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex1DLayeredGrad(texture<unsigned short, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredGrad(texture<short1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex1DLayeredGrad(texture<ushort1, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredGrad(texture<short2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex1DLayeredGrad(texture<ushort2, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredGrad(texture<short4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex1DLayeredGrad(texture<ushort4, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{
  uint4 v   = __utexfetchgradl(t, make_float4(x, 0, 0, 0), layer, make_float4(dPdx, 0, 0, 0), make_float4(dPdy, 0, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 2D Layered Gradient Texture functions                                        *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex2DLayeredGrad(texture<char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex2DLayeredGrad(texture<signed char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex2DLayeredGrad(texture<unsigned char, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex2DLayeredGrad(texture<char1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex2DLayeredGrad(texture<uchar1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex2DLayeredGrad(texture<char2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex2DLayeredGrad(texture<uchar2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex2DLayeredGrad(texture<char4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex2DLayeredGrad(texture<uchar4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex2DLayeredGrad(texture<short, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex2DLayeredGrad(texture<unsigned short, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex2DLayeredGrad(texture<short1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex2DLayeredGrad(texture<ushort1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex2DLayeredGrad(texture<short2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex2DLayeredGrad(texture<ushort2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex2DLayeredGrad(texture<short4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex2DLayeredGrad(texture<ushort4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex2DLayeredGrad(texture<int, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex2DLayeredGrad(texture<unsigned int, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex2DLayeredGrad(texture<int1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex2DLayeredGrad(texture<uint1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex2DLayeredGrad(texture<int2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex2DLayeredGrad(texture<uint2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex2DLayeredGrad(texture<int4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex2DLayeredGrad(texture<uint4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex2DLayeredGrad(texture<long, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex2DLayeredGrad(texture<unsigned long, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex2DLayeredGrad(texture<long1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex2DLayeredGrad(texture<ulong1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex2DLayeredGrad(texture<long2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex2DLayeredGrad(texture<ulong2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex2DLayeredGrad(texture<long4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex2DLayeredGrad(texture<ulong4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<float, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredGrad(texture<float1, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredGrad(texture<float2, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredGrad(texture<float4, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 v = __ftexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<signed char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<unsigned char, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredGrad(texture<char1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredGrad(texture<uchar1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredGrad(texture<char2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredGrad(texture<uchar2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredGrad(texture<char4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredGrad(texture<uchar4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<short, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex2DLayeredGrad(texture<unsigned short, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredGrad(texture<short1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex2DLayeredGrad(texture<ushort1, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredGrad(texture<short2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex2DLayeredGrad(texture<ushort2, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v  = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredGrad(texture<short4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 v   = __itexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex2DLayeredGrad(texture<ushort4, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 v   = __utexfetchgradl(t, make_float4(x, y, 0, 0), layer, make_float4(dPdx.x, dPdx.y, 0, 0), make_float4(dPdy.x, dPdy.y, 0, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
* 3D Gradient Texture functions                                                *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ char tex3DGrad(texture<char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v  = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */

  return (char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ signed char tex3DGrad(texture<signed char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (signed char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned char tex3DGrad(texture<unsigned char, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (unsigned char)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ char1 tex3DGrad(texture<char1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_char1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uchar1 tex3DGrad(texture<uchar1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_uchar1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ char2 tex3DGrad(texture<char2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_char2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uchar2 tex3DGrad(texture<uchar2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_uchar2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ char4 tex3DGrad(texture<char4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_char4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uchar4 tex3DGrad(texture<uchar4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_uchar4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ short tex3DGrad(texture<short, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned short tex3DGrad(texture<unsigned short, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (unsigned short)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ short1 tex3DGrad(texture<short1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_short1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ushort1 tex3DGrad(texture<ushort1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_ushort1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ short2 tex3DGrad(texture<short2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_short2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ushort2 tex3DGrad(texture<ushort2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_ushort2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ short4 tex3DGrad(texture<short4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_short4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ushort4 tex3DGrad(texture<ushort4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_ushort4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ int tex3DGrad(texture<int, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned int tex3DGrad(texture<unsigned int, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (unsigned int)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ int1 tex3DGrad(texture<int1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_int1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ uint1 tex3DGrad(texture<uint1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_uint1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ int2 tex3DGrad(texture<int2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_int2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ uint2 tex3DGrad(texture<uint2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_uint2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ int4 tex3DGrad(texture<int4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_int4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ uint4 tex3DGrad(texture<uint4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_uint4(v.x, v.y, v.z, v.w);
}

#if !defined(__LP64__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ long tex3DGrad(texture<long, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ unsigned long tex3DGrad(texture<unsigned long, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return (unsigned long)v.x;
}

__TEXTURE_FUNCTIONS_DECL__ long1 tex3DGrad(texture<long1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_long1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ ulong1 tex3DGrad(texture<ulong1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_ulong1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ long2 tex3DGrad(texture<long2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_long2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ ulong2 tex3DGrad(texture<ulong2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_ulong2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ long4 tex3DGrad(texture<long4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_long4(v.x, v.y, v.z, v.w);
}

__TEXTURE_FUNCTIONS_DECL__ ulong4 tex3DGrad(texture<ulong4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_ulong4(v.x, v.y, v.z, v.w);
}

#endif /* !__LP64__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<float, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return v.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DGrad(texture<float1, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_float1(v.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DGrad(texture<float2, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_float2(v.x, v.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DGrad(texture<float4, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 v = __ftexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));

  return make_float4(v.x, v.y, v.z, v.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<signed char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<unsigned char, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DGrad(texture<char1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DGrad(texture<uchar1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DGrad(texture<char2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DGrad(texture<uchar2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DGrad(texture<char4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DGrad(texture<uchar4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<short, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float tex3DGrad(texture<unsigned short, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return w.x;
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DGrad(texture<short1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float1 tex3DGrad(texture<ushort1, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float1(w.x);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DGrad(texture<short2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float2 tex3DGrad(texture<ushort2, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v  = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float2(w.x, w.y);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DGrad(texture<short4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 v   = __itexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

__TEXTURE_FUNCTIONS_DECL__ float4 tex3DGrad(texture<ushort4, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 v   = __utexfetchgrad(t, make_float4(x, y, z, 0), make_float4(dPdx.x, dPdx.y, dPdx.z, 0), make_float4(dPdy.x, dPdy.y, dPdy.z, 0));
  float4 w = make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));

  return make_float4(w.x, w.y, w.z, w.w);
}

#endif /* !__LWDA_ARCH__ || __LWDA_ARCH__ >= 200 */

#endif /* __cplusplus && __LWDACC__ */

#undef __TEXTURE_FUNCTIONS_DECL__

#endif /* !__TEXTURE_FETCH_FUNCTIONS_HPP__ */


