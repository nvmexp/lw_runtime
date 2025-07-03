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

#if !defined(__SURFACE_FUNCTIONS_HPP__)
#define __SURFACE_FUNCTIONS_HPP__

#if defined(__LWDACC_RTC__)
#define __SURFACE_FUNCTIONS_DECL__ __device__
#else /* !__LWDACC_RTC__ */
#define __SURFACE_FUNCTIONS_DECL__ static __forceinline__ __device__
#endif /* !__LWDACC_RTC__ */

#if defined(__cplusplus) && defined(__LWDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "lwda_surface_types.h"
#include "host_defines.h"
#include "surface_types.h"
#include "vector_functions.h"
#include "vector_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, mode, type)                                                    \
        ((mode == lwdaBoundaryModeZero)  ? __surf1Dwrite##type(val, surf, x, lwdaBoundaryModeZero ) : \
         (mode == lwdaBoundaryModeClamp) ? __surf1Dwrite##type(val, surf, x, lwdaBoundaryModeClamp) : \
                                           __surf1Dwrite##type(val, surf, x, lwdaBoundaryModeTrap ))

#else /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, mode, type) \
        __surf1Dwrite##type(val, surf, x, lwdaBoundaryModeTrap)

#endif /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(signed char val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned char val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(char4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uchar4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned short val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(short4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ushort4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned int val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(int4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(uint4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long long int val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long long int val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(longlong1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(longlong2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulonglong2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long int val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(unsigned long int val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(long4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(ulong4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float1 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float2 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1Dwrite(float4 val, surface<void, lwdaSurfaceType1D> surf, int x, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, mode, type)                                                    \
        ((mode == lwdaBoundaryModeZero)  ? __surf2Dwrite##type(val, surf, x, y, lwdaBoundaryModeZero ) : \
         (mode == lwdaBoundaryModeClamp) ? __surf2Dwrite##type(val, surf, x, y, lwdaBoundaryModeClamp) : \
                                           __surf2Dwrite##type(val, surf, x, y, lwdaBoundaryModeTrap ))

#else /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, mode, type) \
        __surf2Dwrite##type(val, surf, x, y, lwdaBoundaryModeTrap)

#endif /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(signed char val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned char val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(char4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uchar4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned short val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(short4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ushort4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned int val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(int4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(uint4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long long int val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long long int val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(longlong1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(longlong2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulonglong2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long int val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(unsigned long int val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(long4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(ulong4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float1 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float2 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(float4 val, surface<void, lwdaSurfaceType2D> surf, int x, int y, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, z, mode, type)                                                    \
        ((mode == lwdaBoundaryModeZero)  ? __surf3Dwrite##type(val, surf, x, y, z, lwdaBoundaryModeZero ) : \
         (mode == lwdaBoundaryModeClamp) ? __surf3Dwrite##type(val, surf, x, y, z, lwdaBoundaryModeClamp) : \
                                           __surf3Dwrite##type(val, surf, x, y, z, lwdaBoundaryModeTrap ))

#else /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, z, mode, type) \
        __surf3Dwrite##type(val, surf, x, y, z, lwdaBoundaryModeTrap)

#endif /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(signed char val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned char val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, z, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(char4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, z, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uchar4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned short val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, z, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(short4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, z, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ushort4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned int val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(int4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, z, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(uint4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long long int val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long long int val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(longlong1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(longlong2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, z, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulonglong2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, z, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long int val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(unsigned long int val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(long4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, z, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(ulong4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, z, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float1 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, z, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float2 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, z, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(float4 val, surface<void, lwdaSurfaceType3D> surf, int x, int y, int z, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, z, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, layer, mode, type)                                                    \
        ((mode == lwdaBoundaryModeZero)  ? __surf1DLayeredwrite##type(val, surf, x, layer, lwdaBoundaryModeZero ) : \
         (mode == lwdaBoundaryModeClamp) ? __surf1DLayeredwrite##type(val, surf, x, layer, lwdaBoundaryModeClamp) : \
                                           __surf1DLayeredwrite##type(val, surf, x, layer, lwdaBoundaryModeTrap ))

#else /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, layer, mode, type) \
        __surf1DLayeredwrite##type(val, surf, x, layer, lwdaBoundaryModeTrap)

#endif /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(signed char val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned char val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(char4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uchar4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned short val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(short4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ushort4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned int val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(int4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(uint4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long long int val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long long int val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(longlong2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, layer, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulonglong2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, layer, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long int val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(unsigned long int val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(long4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(ulong4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, layer, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float1 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float2 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf1DLayeredwrite(float4 val, surface<void, lwdaSurfaceType1DLayered> surf, int x, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, layer, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 200

#define __surfModeSwitch(val, surf, x, y, layer, mode, type)                                                    \
        ((mode == lwdaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, layer, lwdaBoundaryModeZero ) : \
         (mode == lwdaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, layer, lwdaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, layer, lwdaBoundaryModeTrap ))

#else /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */

#define __surfModeSwitch(val, surf, x, y, layer, mode, type) \
        __surf2DLayeredwrite##type(val, surf, x, y, layer, lwdaBoundaryModeTrap)

#endif /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(signed char val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned char val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(char4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uchar4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned short val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(short4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ushort4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned int val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(int4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(uint4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long long int val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long long int val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(longlong2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, layer, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulonglong2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layer, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long int val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(unsigned long int val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(long4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layer, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(ulong4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layer, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float1 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, layer, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float2 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, layer, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surf2DLayeredwrite(float4 val, surface<void, lwdaSurfaceType2DLayered> surf, int x, int y, int layer, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, layer, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 200
// Lwbemap and lwbemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, face, mode, type)                                                    \
        ((mode == lwdaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, face, lwdaBoundaryModeZero ) : \
         (mode == lwdaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, face, lwdaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, face, lwdaBoundaryModeTrap ))

#else /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */
// Lwbemap and lwbemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, face, mode, type) \
        __surf2DLayeredwrite##type(val, surf, x, y, face, lwdaBoundaryModeTrap)


#endif /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(char val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(signed char val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(unsigned char val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(char1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(uchar1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(char2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, face, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(uchar2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(char4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, face, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(uchar4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(short val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(unsigned short val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(short1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ushort1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(short2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, face, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ushort2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(short4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, face, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ushort4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(int val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(unsigned int val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(int1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(uint1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(int2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(uint2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(int4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, face, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(uint4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(long long int val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(unsigned long long int val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(longlong1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ulonglong1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(longlong2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, face, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ulonglong2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, face, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(long int val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(unsigned long int val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(long1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ulong1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(long2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ulong2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(long4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, face, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(ulong4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, face, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(float val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(float1 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, face, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(float2 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, face, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapwrite(float4 val, surface<void, lwdaSurfaceTypeLwbemap> surf, int x, int y, int face, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, face, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 200
// Lwbemap and lwbemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, layerFace, mode, type)                                                    \
        ((mode == lwdaBoundaryModeZero)  ? __surf2DLayeredwrite##type(val, surf, x, y, layerFace, lwdaBoundaryModeZero ) : \
         (mode == lwdaBoundaryModeClamp) ? __surf2DLayeredwrite##type(val, surf, x, y, layerFace, lwdaBoundaryModeClamp) : \
                                           __surf2DLayeredwrite##type(val, surf, x, y, layerFace, lwdaBoundaryModeTrap ))

#else /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */
// Lwbemap and lwbemap layered surfaces use 2D Layered instrinsics
#define __surfModeSwitch(val, surf, x, y, layerFace, mode, type) \
       __surf2DLayeredwrite##type(val, surf, x, y, layerFace, lwdaBoundaryModeTrap)


#endif /* __LWDA_ARCH__ && __LWDA_ARCH__ >= 200 */


__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(char val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(signed char val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(unsigned char val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1(val), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(char1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar1((unsigned char)val.x), surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(uchar1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, c1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(char2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar2((unsigned char)val.x, (unsigned char)val.y), surf, x, y, layerFace, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(uchar2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, c2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(char4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uchar4((unsigned char)val.x, (unsigned char)val.y, (unsigned char)val.z, (unsigned char)val.w), surf, x, y, layerFace, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(uchar4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, c4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(short val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val), surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(unsigned short val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1(val), surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(short1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort1((unsigned short)val.x), surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ushort1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, s1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(short2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort2((unsigned short)val.x, (unsigned short)val.y), surf, x, y, layerFace, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ushort2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, s2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(short4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ushort4((unsigned short)val.x, (unsigned short)val.y, (unsigned short)val.z, (unsigned short)val.w), surf, x, y, layerFace, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ushort4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, s4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(int val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(unsigned int val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1(val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(int1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(uint1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(int2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(uint2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(int4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layerFace, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(uint4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(long long int val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val), surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(unsigned long long int val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1(val), surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(longlong1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong1((unsigned long long int)val.x), surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ulonglong1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, l1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(longlong2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_ulonglong2((unsigned long long int)val.x, (unsigned long long int)val.y), surf, x, y, layerFace, mode, l2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ulonglong2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(val, surf, x, y, layerFace, mode, l2);
}

#if !defined(__LP64__)

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(long int val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(unsigned long int val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(long1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ulong1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)val.x), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(long2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ulong2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)val.x, (unsigned int)val.y), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(long4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layerFace, mode, u4);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(ulong4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)val.x, (unsigned int)val.y, (unsigned int)val.z, (unsigned int)val.w), surf, x, y, layerFace, mode, u4);
}

#endif /* !__LP64__ */

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(float val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val)), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(float1 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint1((unsigned int)__float_as_int(val.x)), surf, x, y, layerFace, mode, u1);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(float2 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint2((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y)), surf, x, y, layerFace, mode, u2);
}

__SURFACE_FUNCTIONS_DECL__ void surfLwbemapLayeredwrite(float4 val, surface<void, lwdaSurfaceTypeLwbemapLayered> surf, int x, int y, int layerFace, enum lwdaSurfaceBoundaryMode mode)
{
  __surfModeSwitch(make_uint4((unsigned int)__float_as_int(val.x), (unsigned int)__float_as_int(val.y), (unsigned int)__float_as_int(val.z), (unsigned int)__float_as_int(val.w)), surf, x, y, layerFace, mode, u4);
}

#undef __surfModeSwitch

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#endif /* __cplusplus && __LWDACC__ */

#undef __SURFACE_FUNCTIONS_DECL__

#endif /* !__SURFACE_FUNCTIONS_HPP__ */

