/*
 * Copyright 2014 LWPU Corporation.  All rights reserved.
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

#if !defined(LWSOLVER_COMMON_H_)
#define LWSOLVER_COMMON_H_

#include "library_types.h"

#ifndef LWSOLVERAPI
#ifdef _WIN32
#define LWSOLVERAPI __stdcall
#else
#define LWSOLVERAPI 
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

typedef enum{
    LWSOLVER_STATUS_SUCCESS=0,
    LWSOLVER_STATUS_NOT_INITIALIZED=1,
    LWSOLVER_STATUS_ALLOC_FAILED=2,
    LWSOLVER_STATUS_ILWALID_VALUE=3,
    LWSOLVER_STATUS_ARCH_MISMATCH=4,
    LWSOLVER_STATUS_MAPPING_ERROR=5,
    LWSOLVER_STATUS_EXELWTION_FAILED=6,
    LWSOLVER_STATUS_INTERNAL_ERROR=7,
    LWSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    LWSOLVER_STATUS_NOT_SUPPORTED = 9,
    LWSOLVER_STATUS_ZERO_PIVOT=10,
    LWSOLVER_STATUS_ILWALID_LICENSE=11
} lwsolverStatus_t;

typedef enum {
    LWSOLVER_EIG_TYPE_1=1,
    LWSOLVER_EIG_TYPE_2=2,
    LWSOLVER_EIG_TYPE_3=3
} lwsolverEigType_t ;

typedef enum {
    LWSOLVER_EIG_MODE_NOVECTOR=0,
    LWSOLVER_EIG_MODE_VECTOR=1
} lwsolverEigMode_t ;


lwsolverStatus_t LWSOLVERAPI lwsolverGetProperty(libraryPropertyType type, int *value);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif // LWSOLVER_COMMON_H_



