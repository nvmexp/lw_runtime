/*
 * Copyright 2017-2018 LWPU Corporation.  All rights reserved.
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

#if !defined(__LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/sm_70_rt.h is an internal header file and must not be used directly.  Please use lwda_runtime_api.h or lwda_runtime.h instead.")
#else
#warning "crt/sm_70_rt.h is an internal header file and must not be used directly.  Please use lwda_runtime_api.h or lwda_runtime.h instead."
#endif
#define __LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_70_RT_H__
#endif

#if !defined(__SM_70_RT_H__)
#define __SM_70_RT_H__

#if defined(__LWDACC_RTC__)
#define __SM_70_RT_DECL__ __host__ __device__
#else /* !__LWDACC_RTC__ */
#define __SM_70_RT_DECL__ static __device__ __inline__
#endif /* __LWDACC_RTC__ */

#if defined(__cplusplus) && defined(__LWDACC__)

#if !defined(__LWDA_ARCH__) || __LWDA_ARCH__ >= 700

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

#ifndef __LWDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__LWDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __LWDA_ARCH__ */


/******************************************************************************
 *                                   match                                   *
 ******************************************************************************/
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, unsigned value) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, int value) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, unsigned long value) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, long value) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, unsigned long long value) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, long long value) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, float value) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, double value) __DEF_IF_HOST

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, unsigned value, int *pred) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, int value, int *pred) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, unsigned long value, int *pred) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, long value, int *pred) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, unsigned long long value, int *pred) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, long long value, int *pred) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, float value, int *pred) __DEF_IF_HOST
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, double value, int *pred) __DEF_IF_HOST

__SM_70_RT_DECL__ void __nanosleep(unsigned int ns) __DEF_IF_HOST

__SM_70_RT_DECL__ unsigned short int atomicCAS(unsigned short int *address, unsigned short int compare, unsigned short int val) __DEF_IF_HOST

#endif /* !__LWDA_ARCH__ || __LWDA_ARCH__ >= 700 */

#endif /* __cplusplus && __LWDACC__ */

#undef __DEF_IF_HOST
#undef __SM_70_RT_DECL__

#if !defined(__LWDACC_RTC__) && defined(__LWDA_ARCH__)
#include "sm_70_rt.hpp"
#endif /* !__LWDACC_RTC__ && defined(__LWDA_ARCH__) */

#endif /* !__SM_70_RT_H__ */

#if defined(__UNDEF_LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_70_RT_H__)
#undef __LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_70_RT_H__
#endif
