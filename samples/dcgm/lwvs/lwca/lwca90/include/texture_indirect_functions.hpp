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


#ifndef __TEXTURE_INDIRECT_FUNCTIONS_HPP__
#define __TEXTURE_INDIRECT_FUNCTIONS_HPP__

#if defined(__LWDACC_RTC__)
#define __TEXTURE_INDIRECT_FUNCTIONS_DECL__ __device__
#else /* !__LWDACC_RTC__ */
#define __TEXTURE_INDIRECT_FUNCTIONS_DECL__ static __forceinline__ __device__
#endif /* !__LWDACC_RTC__ */

#if defined(__cplusplus) && defined(__LWDACC__)

#if !defined(__LWDA_ARCH__) || __LWDA_ARCH__ >= 200


#include "builtin_types.h"
#include "host_defines.h"
#include "vector_functions.h"

/*******************************************************************************
*                                                                              *
* 1D Linear Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char *retVal, lwdaTextureObject_t texObject, int x)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(signed char *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char1 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char2 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(char4 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned char *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uchar1 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uchar2 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uchar4 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short1 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short2 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(short4 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned short *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ushort1 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ushort2 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ushort4 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int1 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int2 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(int4 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned int *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uint1 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uint2 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(uint4 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long1 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long2 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(long4 *retVal, lwdaTextureObject_t texObject, int x)
{
  int4 tmp;
  __tex_1d_v4s32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(unsigned long *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ulong1 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ulong2 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(ulong4 *retVal, lwdaTextureObject_t texObject, int x)
{
  uint4 tmp;
  __tex_1d_v4u32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float *retVal, lwdaTextureObject_t texObject, int x)
{
  float4 tmp;
  __tex_1d_v4f32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float1 *retVal, lwdaTextureObject_t texObject, int x)
{
  float4 tmp;
  __tex_1d_v4f32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float2 *retVal, lwdaTextureObject_t texObject, int x)
{
  float4 tmp;
  __tex_1d_v4f32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1Dfetch(float4 *retVal, lwdaTextureObject_t texObject, int x)
{
  float4 tmp;
  __tex_1d_v4f32_s32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 1D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char *retVal, lwdaTextureObject_t texObject, float x)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(signed char *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char1 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char2 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(char4 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned char *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uchar1 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uchar2 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uchar4 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short1 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short2 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(short4 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned short *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ushort1 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ushort2 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ushort4 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int1 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int2 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(int4 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned int *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uint1 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uint2 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(uint4 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long1 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long2 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(long4 *retVal, lwdaTextureObject_t texObject, float x)
{
  int4 tmp;
  __tex_1d_v4s32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(unsigned long *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ulong1 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ulong2 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(ulong4 *retVal, lwdaTextureObject_t texObject, float x)
{
  uint4 tmp;
  __tex_1d_v4u32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float *retVal, lwdaTextureObject_t texObject, float x)
{
  float4 tmp;
  __tex_1d_v4f32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float1 *retVal, lwdaTextureObject_t texObject, float x)
{
  float4 tmp;
  __tex_1d_v4f32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float2 *retVal, lwdaTextureObject_t texObject, float x)
{
  float4 tmp;
  __tex_1d_v4f32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1D(float4 *retVal, lwdaTextureObject_t texObject, float x)
{
  float4 tmp;
  __tex_1d_v4f32_f32(texObject, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 2D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char *retVal, lwdaTextureObject_t texObject, float x, float y)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(signed char *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(char4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(short4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(int4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(long4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  int4 tmp;
  __tex_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  uint4 tmp;
  __tex_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  float4 tmp;
  __tex_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float1 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  float4 tmp;
  __tex_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float2 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  float4 tmp;
  __tex_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2D(float4 *retVal, lwdaTextureObject_t texObject, float x, float y)
{
  float4 tmp;
  __tex_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 3D Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_3d_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_3d_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_3d_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_3d_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_3d_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3D(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_3d_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 1D Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(signed char *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(char4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned char *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uchar1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uchar2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uchar4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(short4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned short *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ushort1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ushort2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ushort4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(int4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned int *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uint1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uint2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(uint4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(long4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  int4 tmp;
  __tex_1d_array_v4s32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(unsigned long *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ulong1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ulong2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(ulong4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  uint4 tmp;
  __tex_1d_array_v4u32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  float4 tmp;
  __tex_1d_array_v4f32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float1 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  float4 tmp;
  __tex_1d_array_v4f32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float2 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  float4 tmp;
  __tex_1d_array_v4f32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayered(float4 *retVal, lwdaTextureObject_t texObject, float x, int layer)
{
  float4 tmp;
  __tex_1d_array_v4f32_f32(texObject, layer, x, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 2D Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  int4 tmp;
  __tex_2d_array_v4s32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  uint4 tmp;
  __tex_2d_array_v4u32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  float4 tmp;
  __tex_2d_array_v4f32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  float4 tmp;
  __tex_2d_array_v4f32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  float4 tmp;
  __tex_2d_array_v4f32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayered(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer)
{
  float4 tmp;
  __tex_2d_array_v4f32_f32(texObject, layer, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(char *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(short *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(int *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(long *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  int4 tmp;
  __tex_lwbe_v4s32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  uint4 tmp;
  __tex_lwbe_v4u32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(float *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_lwbe_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_lwbe_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_lwbe_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemap(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z)
{
  float4 tmp;
  __tex_lwbe_v4f32_f32(texObject, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap Layered Texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  int4 tmp;
  __tex_lwbe_array_v4s32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  uint4 tmp;
  __tex_lwbe_array_v4u32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(float *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  float4 tmp;
  __tex_lwbe_array_v4f32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  float4 tmp;
  __tex_lwbe_array_v4f32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  float4 tmp;
  __tex_lwbe_array_v4f32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayered(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer)
{
  float4 tmp;
  __tex_lwbe_array_v4f32_f32(texObject, layer, x, y, z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 2D Texture indirect gather functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  int4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4s32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  uint4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4u32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  float4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  float4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  float4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2Dgather(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, int comp)
{
  float4 tmp;
  if (comp == 0) {
    __tld4_r_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 1) {
    __tld4_g_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 2) {
    __tld4_b_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  else if (comp == 3) {
    __tld4_a_2d_v4f32_f32(texObject, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  }
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 1D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char *retVal, lwdaTextureObject_t texObject, float x, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(signed char *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(char4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(short4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(int4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uint1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uint2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(uint4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(long4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  int4 tmp;
  __tex_1d_level_v4s32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  uint4 tmp;
  __tex_1d_level_v4u32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  float4 tmp;
  __tex_1d_level_v4f32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float1 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  float4 tmp;
  __tex_1d_level_v4f32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float2 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  float4 tmp;
  __tex_1d_level_v4f32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLod(float4 *retVal, lwdaTextureObject_t texObject, float x, float level)
{
  float4 tmp;
  __tex_1d_level_v4f32_f32(texObject, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 2D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  int4 tmp;
  __tex_2d_level_v4s32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  uint4 tmp;
  __tex_2d_level_v4u32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  float4 tmp;
  __tex_2d_level_v4f32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  float4 tmp;
  __tex_2d_level_v4f32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  float4 tmp;
  __tex_2d_level_v4f32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLod(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float level)
{
  float4 tmp;
  __tex_2d_level_v4f32_f32(texObject, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 3D mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
*retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_3d_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_3d_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_3d_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_3d_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_3d_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DLod(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_3d_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 1D Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(signed char *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(char4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned char *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uchar4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(short4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned short *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ushort4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(int4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned int *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uint1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uint2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(uint4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(long4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  int4 tmp;
  __tex_1d_array_level_v4s32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(unsigned long *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ulong1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ulong2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(ulong4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  uint4 tmp;
  __tex_1d_array_level_v4u32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  float4 tmp;
  __tex_1d_array_level_v4f32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  float4 tmp;
  __tex_1d_array_level_v4f32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  float4 tmp;
  __tex_1d_array_level_v4f32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredLod(float4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float level)
{
  float4 tmp;
  __tex_1d_array_level_v4f32_f32(texObject, layer, x, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 2D Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  int4 tmp;
  __tex_2d_array_level_v4s32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  uint4 tmp;
  __tex_2d_array_level_v4u32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  float4 tmp;
  __tex_2d_array_level_v4f32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  float4 tmp;
  __tex_2d_array_level_v4f32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  float4 tmp;
  __tex_2d_array_level_v4f32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredLod(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float level)
{
  float4 tmp;
  __tex_2d_array_level_v4f32_f32(texObject, layer, x, y, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  int4 tmp;
  __tex_lwbe_level_v4s32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  uint4 tmp;
  __tex_lwbe_level_v4u32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(float *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_lwbe_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_lwbe_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_lwbe_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLod(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float level)
{
  float4 tmp;
  __tex_lwbe_level_v4f32_f32(texObject, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* Lwbemap Layered mipmapped texture indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  int4 tmp;
  __tex_lwbe_array_level_v4s32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  uint4 tmp;
  __tex_lwbe_array_level_v4u32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(float *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  float4 tmp;
  __tex_lwbe_array_level_v4f32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  float4 tmp;
  __tex_lwbe_array_level_v4f32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  float4 tmp;
  __tex_lwbe_array_level_v4f32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void texLwbemapLayeredLod(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
  float4 tmp;
  __tex_lwbe_array_level_v4f32_f32(texObject, layer, x, y, z, level, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 1D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(signed char *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(char4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(short4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(int4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
   uint4 tmp;
   asm volatile ("tex.grad.1d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5}], {%6}, {%7};" : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "l"(texObject), "f"(x), "f"(dPdx), "f"(dPdy));
   *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uint1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uint2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(uint4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(long4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_grad_v4s32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_grad_v4u32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_grad_v4f32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float1 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_grad_v4f32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float2 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_grad_v4f32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DGrad(float4 *retVal, lwdaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_grad_v4f32_f32(texObject, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 2D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_grad_v4s32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_grad_v4u32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_grad_v4f32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_grad_v4f32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_grad_v4f32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DGrad(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_grad_v4f32_f32(texObject, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 3D texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  int4 tmp;
  __tex_3d_grad_v4s32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  uint4 tmp;
  __tex_3d_grad_v4u32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 tmp;
  __tex_3d_grad_v4f32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 tmp;
  __tex_3d_grad_v4f32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 tmp;
  __tex_3d_grad_v4f32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex3DGrad(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
  float4 tmp;
  __tex_3d_grad_v4f32_f32(texObject, x, y, z, dPdx.x, dPdx.y, dPdx.z, dPdy.x, dPdy.y, dPdy.z, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 1D Layered texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(signed char *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(char4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned char *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uchar4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(short4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned short *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ushort4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(int4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned int *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(uint4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(long4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  int4 tmp;
  __tex_1d_array_grad_v4s32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(unsigned long *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ulong1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ulong2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(ulong4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  uint4 tmp;
  __tex_1d_array_grad_v4u32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_array_grad_v4f32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float1 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_array_grad_v4f32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float2 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_array_grad_v4f32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex1DLayeredGrad(float4 *retVal, lwdaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
  float4 tmp;
  __tex_1d_array_grad_v4f32_f32(texObject, layer, x, dPdx, dPdy, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
* 2D Layered texture gradient indirect fetch functions
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  *retVal = (char)tmp.x;
}
__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(signed char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (signed char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(char4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned char *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned char)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uchar1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uchar2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uchar4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(short4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned short *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned short)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ushort1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ushort2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ushort4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(int4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_int4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned int *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned int)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uint1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uint2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(uint4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_uint4(tmp.x, tmp.y, tmp.z, tmp.w);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__LP64__)

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(long4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  int4 tmp;
  __tex_2d_array_grad_v4s32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_long4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(unsigned long *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (unsigned long)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ulong1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ulong2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(ulong4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  uint4 tmp;
  __tex_2d_array_grad_v4u32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_ulong4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif /* !__LP64__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_array_grad_v4f32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = (float)(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float1 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_array_grad_v4f32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float1(tmp.x);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float2 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_array_grad_v4f32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float2(tmp.x, tmp.y);
}

__TEXTURE_INDIRECT_FUNCTIONS_DECL__ void tex2DLayeredGrad(float4 *retVal, lwdaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
  float4 tmp;
  __tex_2d_array_grad_v4f32_f32(texObject, layer, x, y, dPdx.x, dPdx.y, dPdy.x, dPdy.y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
  *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#endif // __LWDA_ARCH__ || __LWDA_ARCH__ >= 200

#endif // __cplusplus && __LWDACC__

#undef __TEXTURE_INDIRECT_FUNCTIONS_DECL__

#endif // __TEXTURE_INDIRECT_FUNCTIONS_HPP__



