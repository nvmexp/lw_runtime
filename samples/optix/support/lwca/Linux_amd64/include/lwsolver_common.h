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


#if defined(_MSC_VER)
typedef __int64 int64_t;
#else
#include <inttypes.h>
#endif

typedef int lwsolver_int_t;


#define LWSOLVER_VER_MAJOR 11
#define LWSOLVER_VER_MINOR 1
#define LWSOLVER_VER_PATCH 1
#define LWSOLVER_VER_BUILD 0
#define LWSOLVER_VERSION (LWSOLVER_VER_MAJOR * 1000 + \
                        LWSOLVER_VER_MINOR *  100 + \
                        LWSOLVER_VER_PATCH)

/*
 * disable this macro to proceed old API
 */
#define DISABLE_LWSOLVER_DEPRECATED

//------------------------------------------------------------------------------

#if !defined(_MSC_VER)
#   define LWSOLVER_CPP_VERSION __cplusplus
#elif _MSC_FULL_VER >= 190024210 // Visual Studio 2015 Update 3
#   define LWSOLVER_CPP_VERSION _MSVC_LANG
#else
#   define LWSOLVER_CPP_VERSION 0
#endif

//------------------------------------------------------------------------------

#if !defined(DISABLE_LWSOLVER_DEPRECATED)

#   if LWSOLVER_CPP_VERSION >= 201402L

#       define LWSOLVER_DEPRECATED(new_func)                                   \
            [[deprecated("please use " #new_func " instead")]]

#   elif defined(_MSC_VER)

#       define LWSOLVER_DEPRECATED(new_func)                                   \
            __declspec(deprecated("please use " #new_func " instead"))

#   elif defined(__INTEL_COMPILER) || defined(__clang__) ||                    \
         (defined(__GNUC__) &&                                                 \
          (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)))

#       define LWSOLVER_DEPRECATED(new_func)                                   \
            __attribute__((deprecated("please use " #new_func " instead")))

#   elif defined(__GNUC__) || defined(__xlc__)

#       define LWSOLVER_DEPRECATED(new_func)                                   \
            __attribute__((deprecated))

#   else

#       define LWSOLVER_DEPRECATED(new_func)

#   endif // defined(__cplusplus) && __cplusplus >= 201402L
//------------------------------------------------------------------------------

#   if LWSOLVER_CPP_VERSION >= 201703L

#       define LWSOLVER_DEPRECATED_ENUM(new_enum)                              \
            [[deprecated("please use " #new_enum " instead")]]

#   elif defined(__clang__) ||                                                 \
         (defined(__GNUC__) && __GNUC__ >= 6 && !defined(__PGI))

#       define LWSOLVER_DEPRECATED_ENUM(new_enum)                              \
            __attribute__((deprecated("please use " #new_enum " instead")))

#   else

#       define LWSOLVER_DEPRECATED_ENUM(new_enum)

#   endif // defined(__cplusplus) && __cplusplus >= 201402L

#else // defined(DISABLE_LWSOLVER_DEPRECATED)

#   define LWSOLVER_DEPRECATED(new_func)
#   define LWSOLVER_DEPRECATED_ENUM(new_enum)

#endif // !defined(DISABLE_LWSOLVER_DEPRECATED)

#undef LWSOLVER_CPP_VERSION






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
    LWSOLVER_STATUS_ILWALID_LICENSE=11,
    LWSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED=12,
    LWSOLVER_STATUS_IRS_PARAMS_ILWALID=13,
    LWSOLVER_STATUS_IRS_PARAMS_ILWALID_PREC=14,
    LWSOLVER_STATUS_IRS_PARAMS_ILWALID_REFINE=15,
    LWSOLVER_STATUS_IRS_PARAMS_ILWALID_MAXITER=16,
    LWSOLVER_STATUS_IRS_INTERNAL_ERROR=20,
    LWSOLVER_STATUS_IRS_NOT_SUPPORTED=21,
    LWSOLVER_STATUS_IRS_OUT_OF_RANGE=22,
    LWSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES=23,
    LWSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED=25,
    LWSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED=26,
    LWSOLVER_STATUS_IRS_MATRIX_SINGULAR=30,
    LWSOLVER_STATUS_ILWALID_WORKSPACE=31
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


typedef enum {
    LWSOLVER_EIG_RANGE_ALL=1001,
    LWSOLVER_EIG_RANGE_I=1002,
    LWSOLVER_EIG_RANGE_V=1003,
} lwsolverEigRange_t ;



typedef enum {
    LWSOLVER_INF_NORM=104,
    LWSOLVER_MAX_NORM=105,
    LWSOLVER_ONE_NORM=106,
    LWSOLVER_FRO_NORM=107,
} lwsolverNorm_t ;

typedef enum {
    LWSOLVER_IRS_REFINE_NOT_SET          = 1100,
    LWSOLVER_IRS_REFINE_NONE             = 1101,
    LWSOLVER_IRS_REFINE_CLASSICAL        = 1102,
    LWSOLVER_IRS_REFINE_CLASSICAL_GMRES  = 1103,
    LWSOLVER_IRS_REFINE_GMRES            = 1104,
    LWSOLVER_IRS_REFINE_GMRES_GMRES      = 1105,
    LWSOLVER_IRS_REFINE_GMRES_NOPCOND    = 1106,

    LWSOLVER_PREC_DD           = 1150,
    LWSOLVER_PREC_SS           = 1151,
    LWSOLVER_PREC_SHT          = 1152,

} lwsolverIRSRefinement_t;


typedef enum {
    LWSOLVER_R_8I  = 1201,
    LWSOLVER_R_8U  = 1202,
    LWSOLVER_R_64F = 1203,
    LWSOLVER_R_32F = 1204,
    LWSOLVER_R_16F = 1205,
    LWSOLVER_R_16BF  = 1206,
    LWSOLVER_R_TF32  = 1207,
    LWSOLVER_R_AP  = 1208,
    LWSOLVER_C_8I  = 1211,
    LWSOLVER_C_8U  = 1212,
    LWSOLVER_C_64F = 1213,
    LWSOLVER_C_32F = 1214,
    LWSOLVER_C_16F = 1215,
    LWSOLVER_C_16BF  = 1216,
    LWSOLVER_C_TF32  = 1217,
    LWSOLVER_C_AP  = 1218,
} lwsolverPrecType_t ;

typedef enum {
   LWSOLVER_ALG_0 = 0,  /* default algorithm */
   LWSOLVER_ALG_1 = 1
} lwsolverAlgMode_t;


typedef enum {
    LWBLAS_STOREV_COLUMNWISE=0, 
    LWBLAS_STOREV_ROWWISE=1
} lwsolverStorevMode_t; 

typedef enum {
    LWBLAS_DIRECT_FORWARD=0, 
    LWBLAS_DIRECT_BACKWARD=1
} lwsolverDirectMode_t;

lwsolverStatus_t LWSOLVERAPI lwsolverGetProperty(
    libraryPropertyType type, 
    int *value);

lwsolverStatus_t LWSOLVERAPI lwsolverGetVersion(
    int *version);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif // LWSOLVER_COMMON_H_



