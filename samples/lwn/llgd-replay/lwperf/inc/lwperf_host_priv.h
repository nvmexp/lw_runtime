#ifndef LWPERF_HOST_PRIV_H
#define LWPERF_HOST_PRIV_H

/*
 * Copyright 2014-2021  LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to LWPU ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and conditions
 * of a form of LWPU software license agreement.
 *
 * LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stddef.h>
#include <stdint.h>
#include "lwperf_common_priv.h"
#include "lwperf_host.h"

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility push(default)
    #if !defined(LWPW_LOCAL)
        #define LWPW_LOCAL __attribute__ ((visibility ("hidden")))
    #endif
#else
    #if !defined(LWPW_LOCAL)
        #define LWPW_LOCAL
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @file   lwperf_host_priv.h
 */

    typedef struct LWPA_RawMetricsConfig LWPA_RawMetricsConfig;

    /// Bitfield of allowable counter providers. TRACE is always enabled.
    typedef enum LWPW_CounterProviderMask
    {
        LWPW_COUNTER_PROVIDER_MASK_ILWALID,
        LWPW_COUNTER_PROVIDER_MASK_HWPM = 1 << 0,
        LWPW_COUNTER_PROVIDER_MASK_SMPC = 1 << 1,
        LWPW_COUNTER_PROVIDER_MASK_SASS = 1 << 2,
        LWPW_COUNTER_PROVIDER_MASK__COUNT
    } LWPW_CounterProviderMask;

    typedef struct LWPW_RawMetricsConfig_SetCounterProviderMask_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// Counter providers to enable.
        LWPW_CounterProviderMask counterProviderMask;
        LWPA_RawMetricsConfig* pRawMetricsConfig;
    } LWPW_RawMetricsConfig_SetCounterProviderMask_Params;
#define LWPW_RawMetricsConfig_SetCounterProviderMask_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_SetCounterProviderMask_Params, pRawMetricsConfig)

    /// Enables a counter provider for scheduling and counter data creation.
    /// This setting applies only to the creation of configuration images and counter data.
    /// Target code assumes all providers referenced in the configuration image are available.
    /// 
    /// Must be called before LWPA_RawMetricsConfig_BeginPassGroup.
    /// 
    /// This API is overridden by provider-specific environment variables.
    LWPW_LOCAL
    LWPA_Status LWPW_RawMetricsConfig_SetCounterProviderMask(LWPW_RawMetricsConfig_SetCounterProviderMask_Params* pParams);

#ifndef LWPERF_LWPA_EXPOSURE_LEVEL_DEFINED
#define LWPERF_LWPA_EXPOSURE_LEVEL_DEFINED

    /// Exposure level for metrics.
    typedef enum LWPA_ExposureLevel
    {
        LWPA_EXPOSURE_LEVEL_ILWALID,
        LWPA_EXPOSURE_LEVEL_PUB,
        LWPA_EXPOSURE_LEVEL_NDA,
        LWPA_EXPOSURE_LEVEL_PRIV,
        LWPA_EXPOSURE_LEVEL_RAW,
        LWPA_EXPOSURE_LEVEL__COUNT
    } LWPA_ExposureLevel;


#endif // LWPERF_LWPA_EXPOSURE_LEVEL_DEFINED


    typedef struct LWPA_RawMetricsPassGroupOptions_Priv
    {
        /// [in]
        size_t structSize;
        LWPA_Bool allowGCM;
    } LWPA_RawMetricsPassGroupOptions_Priv;
#define LWPA_RAW_METRICS_PASS_GROUP_OPTIONS_PRIV_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_RawMetricsPassGroupOptions_Priv, allowGCM)

    typedef struct LWPA_MetricProperties_Priv
    {
        /// [in]
        size_t structSize;
        /// [out]
        LWPA_ExposureLevel exposureLevel;
    } LWPA_MetricProperties_Priv;
#define LWPA_METRIC_PROPERTIES_PRIV_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetricProperties_Priv, exposureLevel)

    typedef struct LWPA_MetricsContext_GetMetricProperties_Begin_PrivParams
    {
        /// [in]
        size_t structSize;
        /// [out]
        LWPA_ExposureLevel exposureLevel;
    } LWPA_MetricsContext_GetMetricProperties_Begin_PrivParams;
#define LWPA_MetricsContext_GetMetricProperties_Begin_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetricsContext_GetMetricProperties_Begin_PrivParams, exposureLevel)

/***************************************************************************//**
 *  @name   PC Sampling
 *  @{
 */

    typedef struct LWPW_PcSampling_IsPcSamplingSupported_PrivParams
    {
        /// [in]
        size_t structSize;
        /// [in]
        LWPA_Bool allowFutureSupportedChips;
    } LWPW_PcSampling_IsPcSamplingSupported_PrivParams;
#define LWPW_PcSampling_IsPcSamplingSupported_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_IsPcSamplingSupported_PrivParams, allowFutureSupportedChips)

    typedef struct LWPW_PcSampling_IsPcSamplingSupported_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const char* pChipName;
        /// [out]
        LWPA_Bool isSupported;
    } LWPW_PcSampling_IsPcSamplingSupported_Params;
#define LWPW_PcSampling_IsPcSamplingSupported_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_IsPcSamplingSupported_Params, isSupported)

    /// Check if the chip supports PC Sampling
    LWPW_LOCAL
    LWPA_Status LWPW_PcSampling_IsPcSamplingSupported(LWPW_PcSampling_IsPcSamplingSupported_Params* pParams);

    typedef struct LWPW_PcSampling_GetNumCounters_PrivParams
    {
        /// [in]
        size_t structSize;
        /// [in]
        LWPA_Bool allowFutureSupportedChips;
    } LWPW_PcSampling_GetNumCounters_PrivParams;
#define LWPW_PcSampling_GetNumCounters_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_GetNumCounters_PrivParams, allowFutureSupportedChips)

    typedef struct LWPW_PcSampling_GetNumCounters_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const char* pChipName;
        /// [out]
        size_t numCounters;
    } LWPW_PcSampling_GetNumCounters_Params;
#define LWPW_PcSampling_GetNumCounters_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_GetNumCounters_Params, numCounters)

    /// Get number of counters from the config
    LWPW_LOCAL
    LWPA_Status LWPW_PcSampling_GetNumCounters(LWPW_PcSampling_GetNumCounters_Params* pParams);

    typedef struct LWPW_PcSampling_GetCounterProperties_PrivParams
    {
        /// [in]
        size_t structSize;
        /// [in]
        LWPA_Bool allowFutureSupportedChips;
    } LWPW_PcSampling_GetCounterProperties_PrivParams;
#define LWPW_PcSampling_GetCounterProperties_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_GetCounterProperties_PrivParams, allowFutureSupportedChips)

    typedef struct LWPW_PcSampling_GetCounterProperties_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const char* pChipName;
        /// [in]
        size_t counterIndex;
        /// [out]
        uint64_t counterId;
        /// [out]
        const char* pCounterName;
        /// [out]
        const char* pCounterDesc;
        /// [out]
        LWPA_Bool hardwareReason;
    } LWPW_PcSampling_GetCounterProperties_Params;
#define LWPW_PcSampling_GetCounterProperties_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_GetCounterProperties_Params, hardwareReason)

    /// Get counter property from the config. Counters are sorted in ascending alphabetic order.
    LWPW_LOCAL
    LWPA_Status LWPW_PcSampling_GetCounterProperties(LWPW_PcSampling_GetCounterProperties_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 


#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_HOST_PRIV_H
