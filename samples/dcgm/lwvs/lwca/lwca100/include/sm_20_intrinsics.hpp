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

#if !defined(__SM_20_INTRINSICS_HPP__)
#define __SM_20_INTRINSICS_HPP__

#if defined(__LWDACC_RTC__)
#define __SM_20_INTRINSICS_DECL__ __device__
#else /* __LWDACC_RTC__ */
#define __SM_20_INTRINSICS_DECL__ static __inline__ __device__
#endif /* __LWDACC_RTC__ */

#if defined(__cplusplus) && defined(__LWDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "lwda_runtime_api.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SM_20_INTRINSICS_DECL__ unsigned int ballot(bool pred)
{
  return __ballot((int)pred);
}

__SM_20_INTRINSICS_DECL__ int syncthreads_count(bool pred)
{
  return __syncthreads_count((int)pred);
}

__SM_20_INTRINSICS_DECL__ bool syncthreads_and(bool pred)
{
  return (bool)__syncthreads_and((int)pred);
}

__SM_20_INTRINSICS_DECL__ bool syncthreads_or(bool pred)
{
  return (bool)__syncthreads_or((int)pred);
}


// This function returns 1 if generic address "ptr" is in global memory space.
// It returns 0 if "ptr" is in shared, local or constant memory space.
__SM_20_INTRINSICS_DECL__ unsigned int __isGlobal(const void *ptr)
{
    unsigned int ret;
    asm volatile ("{ \n\t"
                  "    .reg .pred p; \n\t"
                  "    isspacep.global p, %1; \n\t"
                  "    selp.u32 %0, 1, 0, p;  \n\t"
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__LWDACC_RTC__)
                  "} \n\t" : "=r"(ret) : "l"(ptr));
#else
                  "} \n\t" : "=r"(ret) : "r"(ptr));
#endif

    return ret;
}

#endif /* __cplusplus && __LWDACC__ */

#undef __SM_20_INTRINSICS_DECL__

#endif /* !__SM_20_INTRINSICS_HPP__ */

