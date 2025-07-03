/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */


 /**
 * @file
 * @brief This file defines the types provided by the lwTENSOR library.
 */
#pragma once

#include <stdint.h>

/**
 * \brief This enum captures all unary and binary element-wise operations supported by the lwTENSOR library.
 * \ingroup runtimeDataStructurePLC3
 */
typedef enum 
{
    /* Unary */
    LWTENSOR_OP_IDENTITY = 1,          ///< Identity operator (i.e., elements are not changed)
    LWTENSOR_OP_SQRT = 2,              ///< Square root
    LWTENSOR_OP_RELU = 8,              ///< Rectified linear unit
    LWTENSOR_OP_CONJ = 9,              ///< Complex conjugate
    LWTENSOR_OP_RCP = 10,              ///< Reciprocal
    LWTENSOR_OP_SIGMOID = 11,          ///< y=1/(1+exp(-x))
    LWTENSOR_OP_TANH = 12,             ///< y=tanh(x)
    LWTENSOR_OP_EXP = 22,              ///< Exponentiation.
    LWTENSOR_OP_LOG = 23,              ///< Log (base e).
    LWTENSOR_OP_ABS = 24,              ///< Absolute value.
    LWTENSOR_OP_NEG = 25,              ///< Negation.
    LWTENSOR_OP_SIN = 26,              ///< Sine.
    LWTENSOR_OP_COS = 27,              ///< Cosine.
    LWTENSOR_OP_TAN = 28,              ///< Tangent.
    LWTENSOR_OP_SINH = 29,             ///< Hyperbolic sine.
    LWTENSOR_OP_COSH = 30,             ///< Hyperbolic cosine.
    LWTENSOR_OP_ASIN = 31,             ///< Ilwerse sine.
    LWTENSOR_OP_ACOS = 32,             ///< Ilwerse cosine.
    LWTENSOR_OP_ATAN = 33,             ///< Ilwerse tangent.
    LWTENSOR_OP_ASINH = 34,            ///< Ilwerse hyperbolic sine.
    LWTENSOR_OP_ACOSH = 35,            ///< Ilwerse hyperbolic cosine.
    LWTENSOR_OP_ATANH = 36,            ///< Ilwerse hyperbolic tangent.
    LWTENSOR_OP_CEIL = 37,             ///< Ceiling.
    LWTENSOR_OP_FLOOR = 38,            ///< Floor.
    /* Binary */
    LWTENSOR_OP_ADD = 3,               ///< Addition of two elements
    LWTENSOR_OP_MUL = 5,               ///< Multiplication of two elements
    LWTENSOR_OP_MAX = 6,               ///< Maximum of two elements
    LWTENSOR_OP_MIN = 7,               ///< Minimum of two elements

    LWTENSOR_OP_UNKNOWN = 126, ///< reserved for internal use only

} lwtensorOperator_t;

/**
 * \brief lwTENSOR status type returns
 *
 * \details The type is used for function status returns. All lwTENSOR library functions return their status, which can have the following values.
 * \ingroup runtimeDataStructurePLC3
 */
typedef enum 
{
    /** The operation completed successfully.*/
    LWTENSOR_STATUS_SUCCESS                = 0,
    /** The lwTENSOR library was not initialized.*/
    LWTENSOR_STATUS_NOT_INITIALIZED        = 1,
    /** Resource allocation failed inside the lwTENSOR library.*/
    LWTENSOR_STATUS_ALLOC_FAILED           = 3,
    /** An unsupported value or parameter was passed to the function (indicates an user error).*/
    LWTENSOR_STATUS_ILWALID_VALUE          = 7,
    /** Indicates that the device is either not ready, or the target architecture is not supported.*/
    LWTENSOR_STATUS_ARCH_MISMATCH          = 8,
    /** An access to GPU memory space failed, which is usually caused by a failure to bind a texture.*/
    LWTENSOR_STATUS_MAPPING_ERROR          = 11,
    /** The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.*/
    LWTENSOR_STATUS_EXELWTION_FAILED       = 13,
    /** An internal lwTENSOR error has oclwrred.*/
    LWTENSOR_STATUS_INTERNAL_ERROR         = 14,
    /** The requested operation is not supported.*/
    LWTENSOR_STATUS_NOT_SUPPORTED          = 15,
    /** The functionality requested requires some license and an error was detected when trying to check the current licensing.*/
    LWTENSOR_STATUS_LICENSE_ERROR          = 16,
    /** A call to LWBLAS did not succeed.*/
    LWTENSOR_STATUS_LWBLAS_ERROR           = 17,
    /** Some unknown LWCA error has oclwrred.*/
    LWTENSOR_STATUS_LWDA_ERROR             = 18,
    /** The provided workspace was insufficient.*/
    LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    /** Indicates that the driver version is insufficient.*/
    LWTENSOR_STATUS_INSUFFICIENT_DRIVER    = 20,
    /** Indicates an error related to file I/O.*/
    LWTENSOR_STATUS_IO_ERROR               = 21,
} lwtensorStatus_t;

/**
 * \brief Allows users to specify the algorithm to be used for performing the
 * tensor contraction.
 *
 * \details This enum gives users finer control over which algorithm should be exelwted by
 * lwtensorContraction(); values >= 0 correspond to certain sub-algorithms of GETT.
 */
typedef enum
{
    LWTENSOR_ALGO_LWTE           = -5, ///< Choose the GETT algorithm (via LwTe) // {$lw-internal-release}
    LWTENSOR_ALGO_GETT           = -4, ///< Choose the GETT algorithm
    LWTENSOR_ALGO_TGETT          = -3, ///< Transpose (A or B) + GETT
    LWTENSOR_ALGO_TTGT           = -2, ///< Transpose-Transpose-GEMM-Transpose (requires additional memory)
    LWTENSOR_ALGO_DEFAULT        = -1, ///< Lets the internal heuristic choose
} lwtensorAlgo_t;

/**
 * \brief This enum gives users finer control over the suggested workspace
 *
 * \details This enum gives users finer control over the amount of workspace that is
 * suggested by lwtensorContractionGetWorkspace
 */
typedef enum
{
    LWTENSOR_WORKSPACE_MIN = 1,         ///< At least one algorithm will be available
    LWTENSOR_WORKSPACE_RECOMMENDED = 2, ///< The most suitable algorithm will be available
    LWTENSOR_WORKSPACE_MAX = 3,         ///< All algorithms will be available
} lwtensorWorksizePreference_t;

/**
 * \brief Encodes lwTENSOR's compute type (see "User Guide - Accuracy Guarantees" for details).
 */
typedef enum
{
    LWTENSOR_COMPUTE_16F  = (1U<< 0U),  ///< floating-point: 5-bit exponent and 10-bit mantissa (aka half)
    LWTENSOR_COMPUTE_16BF = (1U<< 10U),  ///< floating-point: 8-bit exponent and 7-bit mantissa (aka bfloat)
    LWTENSOR_COMPUTE_TF32 = (1U<< 12U),  ///< floating-point: 8-bit exponent and 10-bit mantissa (aka tensor-float-32)
    LWTENSOR_COMPUTE_32F  = (1U<< 2U),  ///< floating-point: 8-bit exponent and 23-bit mantissa (aka float)
    LWTENSOR_COMPUTE_64F  = (1U<< 4U),  ///< floating-point: 11-bit exponent and 52-bit mantissa (aka double)
    LWTENSOR_COMPUTE_8U   = (1U<< 6U),  ///< 8-bit unsigned integer
    LWTENSOR_COMPUTE_8I   = (1U<< 8U),  ///< 8-bit signed integer
    LWTENSOR_COMPUTE_32U  = (1U<< 7U),  ///< 32-bit unsigned integer
    LWTENSOR_COMPUTE_32I  = (1U<< 9U),  ///< 32-bit signed integer
   
    /* All compute types below this line will be deprecated in the near future. */
    LWTENSOR_R_MIN_16F  = (1U<< 0U),  ///< DEPRECATED (real as a half), please use LWTENSOR_COMPUTE_16F instead
    LWTENSOR_C_MIN_16F  = (1U<< 1U),  ///< DEPRECATED (complex as a half), please use LWTENSOR_COMPUTE_16F instead
    LWTENSOR_R_MIN_32F  = (1U<< 2U),  ///< DEPRECATED (real as a float), please use LWTENSOR_COMPUTE_32F instead
    LWTENSOR_C_MIN_32F  = (1U<< 3U),  ///< DEPRECATED (complex as a float), please use LWTENSOR_COMPUTE_32F instead
    LWTENSOR_R_MIN_64F  = (1U<< 4U),  ///< DEPRECATED (real as a double), please use LWTENSOR_COMPUTE_64F instead
    LWTENSOR_C_MIN_64F  = (1U<< 5U),  ///< DEPRECATED (complex as a double), please use LWTENSOR_COMPUTE_64F instead
    LWTENSOR_R_MIN_8U   = (1U<< 6U),  ///< DEPRECATED (real as a uint8), please use LWTENSOR_COMPUTE_8U instead
    LWTENSOR_R_MIN_32U  = (1U<< 7U),  ///< DEPRECATED (real as a uint32), please use LWTENSOR_COMPUTE_32U instead
    LWTENSOR_R_MIN_8I   = (1U<< 8U),  ///< DEPRECATED (real as a int8), please use LWTENSOR_COMPUTE_8I instead
    LWTENSOR_R_MIN_32I  = (1U<< 9U),  ///< DEPRECATED (real as a int32), please use LWTENSOR_COMPUTE_32I instead
    LWTENSOR_R_MIN_16BF = (1U<<10U),  ///< DEPRECATED (real as a bfloat16), please use LWTENSOR_COMPUTE_16BF instead
    LWTENSOR_R_MIN_TF32 = (1U<<11U),  ///< DEPRECATED (real as a tensorfloat32), please use LWTENSOR_COMPUTE_TF32 instead
    LWTENSOR_C_MIN_TF32 = (1U<<12U),  ///< DEPRECATED (complex as a tensorfloat32), please use LWTENSOR_COMPUTE_TF32 instead
} lwtensorComputeType_t;

/**
 * This enum lists all attributes of a lwtensorContractionContraction_t that can be modified.
 */
typedef enum
{
    LWTENSOR_CONTRACTION_DESCRIPTOR_TAG ///< uint32_t: enables users to distinguish two identical tensor contractions w.r.t. the sw-managed plan-cache. (default value: 0)
} lwtensorContractionDescriptorAttributes_t;

/**
 * This enum lists all attributes of a lwtensorContractionFind_t that can be modified.
 */
typedef enum
{
    LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE, ///< lwtensorAutotuneMode_t: Determines if the corresponding algrithm/kernel for this plan should be cached.
    LWTENSOR_CONTRACTION_FIND_CACHE_MODE, ///< lwtensorCacheMode_t: Gives fine control over what is considered a cachehit.
    LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT, ///< uint32_t: Only applicable if LWTENSOR_CONTRACTION_FIND_CACHE_MODE is set to LWTENSOR_AUTOTUNE_INCREMENTAL
    LWTENSOR_CONTRACTION_FIND_SPLITK_NUM, ///< int32_t: Number of splits along the contraction dimensions. If != 1, SPLITK_NUM parts of matrix multiplication will be computed in parallel. A value of -1 denotes that this value should be automatically selected by lwTENSOR's heuristic (this is the default). // {$lw-internal-release}
} lwtensorContractionFindAttributes_t;

/**
 * This enum is important w.r.t. lwTENSOR's caching capability of plans.
 */
typedef enum
{
    LWTENSOR_AUTOTUNE_NONE, ///< Indicates no autotuning (default); in this case the cache will help to reduce the plan-creation overhead. In the case of a cachehit: the cached plan will be reused, otherwise the plancache will be neglected.
    LWTENSOR_AUTOTUNE_INCREMENTAL, ///< Indicates an incremental autotuning (i.e., each invocation of corresponding lwtensorInitContractionPlan() will create a plan based on a different algorithm/kernel; the maximum number of kernels that will be tested is defined by the LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT FindAttributes_t). WARNING: If this autotuning mode is selected, then we cannot guarantee bit-wise identical results (since different algorithms could be exelwted).
    LWTENSOR_AUTOTUNE_FULL, ///< This value corresponds to a "full" autotuning run (i.e., measuring all applicable kernels back-to-back); WARNING: The first invocation of lwtensorContraction() will be blocking. // {$lw-internal-release}
} lwtensorAutotuneMode_t;

/**
 * This enum defines what is considered a cache hit.
 */
typedef enum
{
    LWTENSOR_CACHE_MODE_NONE,     ///< Plan will not be cached
    LWTENSOR_CACHE_MODE_PEDANTIC, ///< All parameters of the corresponding descriptor must be identical to the cached plan (default).
    LWTENSOR_CACHE_MODE_STRICT,   ///< All parameters of the corresponding descriptor must be identical to the cached plan; the only exception is the workspace that may be larger (default). // {$lw-internal-release}
    LWTENSOR_CACHE_MODE_RELAXED,  ///< All parameters except for the extents, strides and workspace have to be identical to the cache plan. The provided workspace may be larger. The strides must either be NULL (i.e., automatically inferred) or correspond to a dense tensor with monotonically increasing strides from left to right. // {$lw-internal-release}
} lwtensorCacheMode_t;

/**
 * \brief Opaque structure holding lwTENSOR's library context.
 */
typedef struct { int64_t fields[512]; /*!< Data */ } lwtensorHandle_t;

/**
 * \brief Opaque data structure that represents a cacheline of the software-managed plan cache
 */
typedef struct { int64_t fields[1408]; /*!< Data */ } lwtensorPlanCacheline_t;

/**
 * \brief Opaque data structure that represents the software-managed plan cache. A plan cache must not be used by multiple threads.
 */
typedef struct { int64_t fields[12*1024]; /*!< Data */ } lwtensorPlanCache_t;

/**
 * \brief Opaque structure representing a tensor descriptor.
 */
typedef struct { int64_t fields[72]; /*!< Data */ } lwtensorTensorDescriptor_t;

/**
 * \brief Opaque structure representing a tensor contraction descriptor.
 */
typedef struct { int64_t fields[288]; /*!< Data */ } lwtensorContractionDescriptor_t;

/**
 * \brief Opaque structure representing a plan.
 */
typedef struct { int64_t fields[1408]; /*!< Data */ } lwtensorContractionPlan_t;

/**
 * \brief Opaque structure representing a candidate.
 */
typedef struct { int64_t fields[64]; /*!< Data */ } lwtensorContractionFind_t;

