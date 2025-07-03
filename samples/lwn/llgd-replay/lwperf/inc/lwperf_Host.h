#ifndef LWPERF_HOST_H
#define LWPERF_HOST_H

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
#include "lwperf_common.h"

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
 *  @file   lwperf_host.h
 */


// Guard against multiple definition of LwPerf host types
#ifndef LWPERF_HOST_API_DEFINED
#define LWPERF_HOST_API_DEFINED


/***************************************************************************//**
 *  @name   Host Configuration
 *  @{
 */

    typedef struct LWPA_CounterDataImageCopyOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// The CounterDataPrefix generated from e.g.    lwperf2 initdata   or
        /// LWPW_CounterDataBuilder_GetCounterDataPrefix().  Must be align(8).
        const uint8_t* pCounterDataPrefix;
        size_t counterDataPrefixSize;
        /// max number of ranges that can be profiled
        uint32_t maxNumRanges;
        /// max number of RangeTree nodes; must be >= maxNumRanges
        uint32_t maxNumRangeTreeNodes;
        /// max string length of each RangeName, including the trailing NUL character
        uint32_t maxRangeNameLength;
    } LWPA_CounterDataImageCopyOptions;
#define LWPA_COUNTER_DATA_IMAGE_COPY_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_CounterDataImageCopyOptions, maxRangeNameLength)

    /// Load the host library.
    LWPA_Status LWPA_InitializeHost(void);

    typedef struct LWPW_InitializeHost_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
    } LWPW_InitializeHost_Params;
#define LWPW_InitializeHost_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_InitializeHost_Params, pPriv)

    /// Load the host library.
    LWPA_Status LWPW_InitializeHost(LWPW_InitializeHost_Params* pParams);

    LWPA_Status LWPA_CounterData_CallwlateCounterDataImageCopySize(
        const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions,
        const uint8_t* pCounterDataSrc,
        size_t* pCopyDataImageCounterSize);

    typedef struct LWPW_CounterData_CallwlateCounterDataImageCopySize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// The CounterDataPrefix generated from e.g.    lwperf2 initdata   or
        /// LWPW_CounterDataBuilder_GetCounterDataPrefix().  Must be align(8).
        const uint8_t* pCounterDataPrefix;
        size_t counterDataPrefixSize;
        /// max number of ranges that can be profiled
        uint32_t maxNumRanges;
        /// max number of RangeTree nodes; must be >= maxNumRanges
        uint32_t maxNumRangeTreeNodes;
        /// max string length of each RangeName, including the trailing NUL character
        uint32_t maxRangeNameLength;
        const uint8_t* pCounterDataSrc;
        /// [out] required size of the copy buffer
        size_t copyDataImageCounterSize;
    } LWPW_CounterData_CallwlateCounterDataImageCopySize_Params;
#define LWPW_CounterData_CallwlateCounterDataImageCopySize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterData_CallwlateCounterDataImageCopySize_Params, copyDataImageCounterSize)

    LWPA_Status LWPW_CounterData_CallwlateCounterDataImageCopySize(LWPW_CounterData_CallwlateCounterDataImageCopySize_Params* pParams);

    LWPA_Status LWPA_CounterData_InitializeCounterDataImageCopy(
        const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions,
        const uint8_t* pCounterDataSrc,
        uint8_t* pCounterDataDst);

    typedef struct LWPW_CounterData_InitializeCounterDataImageCopy_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// The CounterDataPrefix generated from e.g.    lwperf2 initdata   or
        /// LWPW_CounterDataBuilder_GetCounterDataPrefix().  Must be align(8).
        const uint8_t* pCounterDataPrefix;
        size_t counterDataPrefixSize;
        /// max number of ranges that can be profiled
        uint32_t maxNumRanges;
        /// max number of RangeTree nodes; must be >= maxNumRanges
        uint32_t maxNumRangeTreeNodes;
        /// max string length of each RangeName, including the trailing NUL character
        uint32_t maxRangeNameLength;
        const uint8_t* pCounterDataSrc;
        uint8_t* pCounterDataDst;
    } LWPW_CounterData_InitializeCounterDataImageCopy_Params;
#define LWPW_CounterData_InitializeCounterDataImageCopy_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterData_InitializeCounterDataImageCopy_Params, pCounterDataDst)

    LWPA_Status LWPW_CounterData_InitializeCounterDataImageCopy(LWPW_CounterData_InitializeCounterDataImageCopy_Params* pParams);

    typedef struct LWPA_CounterDataCombiner LWPA_CounterDataCombiner;

    typedef struct LWPA_CounterDataCombinerOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// The destination counter data into which the source datas will be combined
        uint8_t* pCounterDataDst;
    } LWPA_CounterDataCombinerOptions;
#define LWPA_COUNTER_DATA_COMBINER_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_CounterDataCombinerOptions, pCounterDataDst)

    LWPA_Status LWPA_CounterDataCombiner_Create(
        const LWPA_CounterDataCombinerOptions* pCounterDataCombinerOptions,
        LWPA_CounterDataCombiner** ppCounterDataCombiner);

    typedef struct LWPW_CounterDataCombiner_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// The destination counter data into which the source datas will be combined
        uint8_t* pCounterDataDst;
        /// [out] The created counter data combiner
        LWPA_CounterDataCombiner* pCounterDataCombiner;
    } LWPW_CounterDataCombiner_Create_Params;
#define LWPW_CounterDataCombiner_Create_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataCombiner_Create_Params, pCounterDataCombiner)

    LWPA_Status LWPW_CounterDataCombiner_Create(LWPW_CounterDataCombiner_Create_Params* pParams);

    LWPA_Status LWPA_CounterDataCombiner_Destroy(LWPA_CounterDataCombiner* pCounterDataCombiner);

    typedef struct LWPW_CounterDataCombiner_Destroy_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataCombiner* pCounterDataCombiner;
    } LWPW_CounterDataCombiner_Destroy_Params;
#define LWPW_CounterDataCombiner_Destroy_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataCombiner_Destroy_Params, pCounterDataCombiner)

    LWPA_Status LWPW_CounterDataCombiner_Destroy(LWPW_CounterDataCombiner_Destroy_Params* pParams);

    LWPA_Status LWPA_CounterDataCombiner_CreateRange(
        LWPA_CounterDataCombiner* pCounterDataCombiner,
        size_t numDescriptions,
        const char* const* ppDescriptions,
        size_t* pRangeIndexDst);

    typedef struct LWPW_CounterDataCombiner_CreateRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataCombiner* pCounterDataCombiner;
        size_t numDescriptions;
        const char* const* ppDescriptions;
        /// [out]
        size_t rangeIndexDst;
    } LWPW_CounterDataCombiner_CreateRange_Params;
#define LWPW_CounterDataCombiner_CreateRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataCombiner_CreateRange_Params, rangeIndexDst)

    LWPA_Status LWPW_CounterDataCombiner_CreateRange(LWPW_CounterDataCombiner_CreateRange_Params* pParams);

    LWPA_Status LWPA_CounterDataCombiner_AclwmulateIntoRange(
        LWPA_CounterDataCombiner* pCounterDataCombiner,
        size_t rangeIndexDst,
        uint32_t dstMultiplier,
        const uint8_t* pCounterDataSrc,
        size_t rangeIndexSrc,
        uint32_t srcMultiplier);

    typedef struct LWPW_CounterDataCombiner_AclwmulateIntoRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataCombiner* pCounterDataCombiner;
        size_t rangeIndexDst;
        uint32_t dstMultiplier;
        const uint8_t* pCounterDataSrc;
        size_t rangeIndexSrc;
        uint32_t srcMultiplier;
    } LWPW_CounterDataCombiner_AclwmulateIntoRange_Params;
#define LWPW_CounterDataCombiner_AclwmulateIntoRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataCombiner_AclwmulateIntoRange_Params, srcMultiplier)

    LWPA_Status LWPW_CounterDataCombiner_AclwmulateIntoRange(LWPW_CounterDataCombiner_AclwmulateIntoRange_Params* pParams);

    typedef struct LWPW_CounterDataCombiner_SumIntoRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataCombiner* pCounterDataCombiner;
        size_t rangeIndexDst;
        const uint8_t* pCounterDataSrc;
        size_t rangeIndexSrc;
    } LWPW_CounterDataCombiner_SumIntoRange_Params;
#define LWPW_CounterDataCombiner_SumIntoRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataCombiner_SumIntoRange_Params, rangeIndexSrc)

    LWPA_Status LWPW_CounterDataCombiner_SumIntoRange(LWPW_CounterDataCombiner_SumIntoRange_Params* pParams);

    typedef struct LWPW_CounterDataCombiner_WeightedSumIntoRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataCombiner* pCounterDataCombiner;
        size_t rangeIndexDst;
        double dstMultiplier;
        const uint8_t* pCounterDataSrc;
        size_t rangeIndexSrc;
        double srcMultiplier;
    } LWPW_CounterDataCombiner_WeightedSumIntoRange_Params;
#define LWPW_CounterDataCombiner_WeightedSumIntoRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataCombiner_WeightedSumIntoRange_Params, srcMultiplier)

    LWPA_Status LWPW_CounterDataCombiner_WeightedSumIntoRange(LWPW_CounterDataCombiner_WeightedSumIntoRange_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name   Metrics Configuration
 *  @{
 */

    typedef struct LWPA_SupportedChipNames
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out
        const char* const* ppChipNames;
        /// out
        size_t numChipNames;
    } LWPA_SupportedChipNames;
#define LWPA_SUPPORTED_CHIP_NAMES_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_SupportedChipNames, numChipNames)

    typedef struct LWPA_RawMetricsConfig LWPA_RawMetricsConfig;

    typedef struct LWPA_RawMetricsConfigOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_ActivityKind activityKind;
        const char* pChipName;
    } LWPA_RawMetricsConfigOptions;
#define LWPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_RawMetricsConfigOptions, pChipName)

    typedef struct LWPA_RawMetricsPassGroupOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        size_t maxPassCount;
    } LWPA_RawMetricsPassGroupOptions;
#define LWPA_RAW_METRICS_PASS_GROUP_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_RawMetricsPassGroupOptions, maxPassCount)

    typedef struct LWPA_RawMetricProperties
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out
        const char* pMetricName;
        /// out
        LWPA_Bool supportsPipelined;
        /// out
        LWPA_Bool supportsIsolated;
    } LWPA_RawMetricProperties;
#define LWPA_RAW_METRIC_PROPERTIES_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_RawMetricProperties, supportsIsolated)

    typedef struct LWPA_RawMetricRequest
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// in
        const char* pMetricName;
        /// in
        LWPA_Bool isolated;
        /// in; ignored by AddMetric but observed by CounterData initialization
        LWPA_Bool keepInstances;
    } LWPA_RawMetricRequest;
#define LWPA_RAW_METRIC_REQUEST_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_RawMetricRequest, keepInstances)

    LWPA_Status LWPA_GetSupportedChipNames(LWPA_SupportedChipNames* pSupportedChipNames);

    typedef struct LWPW_GetSupportedChipNames_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [out]
        const char* const* ppChipNames;
        /// [out]
        size_t numChipNames;
    } LWPW_GetSupportedChipNames_Params;
#define LWPW_GetSupportedChipNames_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_GetSupportedChipNames_Params, numChipNames)

    LWPA_Status LWPW_GetSupportedChipNames(LWPW_GetSupportedChipNames_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_Create(
        const LWPA_RawMetricsConfigOptions* pMetricsConfigOptions,
        LWPA_RawMetricsConfig** ppRawMetricsConfig);

    LWPA_Status LWPA_RawMetricsConfig_Destroy(LWPA_RawMetricsConfig* pRawMetricsConfig);

    typedef struct LWPW_RawMetricsConfig_Destroy_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_RawMetricsConfig* pRawMetricsConfig;
    } LWPW_RawMetricsConfig_Destroy_Params;
#define LWPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_Destroy_Params, pRawMetricsConfig)

    LWPA_Status LWPW_RawMetricsConfig_Destroy(LWPW_RawMetricsConfig_Destroy_Params* pParams);

    typedef struct LWPW_RawMetricsConfig_SetCounterAvailability_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_RawMetricsConfig* pRawMetricsConfig;
        /// [in] buffer with counter availability image
        const uint8_t* pCounterAvailabilityImage;
    } LWPW_RawMetricsConfig_SetCounterAvailability_Params;
#define LWPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_SetCounterAvailability_Params, pCounterAvailabilityImage)

    LWPA_Status LWPW_RawMetricsConfig_SetCounterAvailability(LWPW_RawMetricsConfig_SetCounterAvailability_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_BeginPassGroup(
        LWPA_RawMetricsConfig* pRawMetricsConfig,
        const LWPA_RawMetricsPassGroupOptions* pRawMetricsPassGroupOptions);

    typedef struct LWPW_RawMetricsConfig_BeginPassGroup_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_RawMetricsConfig* pRawMetricsConfig;
        size_t maxPassCount;
    } LWPW_RawMetricsConfig_BeginPassGroup_Params;
#define LWPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_BeginPassGroup_Params, maxPassCount)

    LWPA_Status LWPW_RawMetricsConfig_BeginPassGroup(LWPW_RawMetricsConfig_BeginPassGroup_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_EndPassGroup(LWPA_RawMetricsConfig* pRawMetricsConfig);

    typedef struct LWPW_RawMetricsConfig_EndPassGroup_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_RawMetricsConfig* pRawMetricsConfig;
    } LWPW_RawMetricsConfig_EndPassGroup_Params;
#define LWPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_EndPassGroup_Params, pRawMetricsConfig)

    LWPA_Status LWPW_RawMetricsConfig_EndPassGroup(LWPW_RawMetricsConfig_EndPassGroup_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_GetNumMetrics(
        const LWPA_RawMetricsConfig* pRawMetricsConfig,
        size_t* pNumMetrics);

    typedef struct LWPW_RawMetricsConfig_GetNumMetrics_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const LWPA_RawMetricsConfig* pRawMetricsConfig;
        /// [out]
        size_t numMetrics;
    } LWPW_RawMetricsConfig_GetNumMetrics_Params;
#define LWPW_RawMetricsConfig_GetNumMetrics_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_GetNumMetrics_Params, numMetrics)

    LWPA_Status LWPW_RawMetricsConfig_GetNumMetrics(LWPW_RawMetricsConfig_GetNumMetrics_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_GetMetricProperties(
        const LWPA_RawMetricsConfig* pRawMetricsConfig,
        size_t metricIndex,
        LWPA_RawMetricProperties* pRawMetricProperties);

    typedef struct LWPW_RawMetricsConfig_GetMetricProperties_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const LWPA_RawMetricsConfig* pRawMetricsConfig;
        size_t metricIndex;
        /// [out]
        const char* pMetricName;
        /// [out]
        LWPA_Bool supportsPipelined;
        /// [out]
        LWPA_Bool supportsIsolated;
    } LWPW_RawMetricsConfig_GetMetricProperties_Params;
#define LWPW_RawMetricsConfig_GetMetricProperties_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_GetMetricProperties_Params, supportsIsolated)

    LWPA_Status LWPW_RawMetricsConfig_GetMetricProperties(LWPW_RawMetricsConfig_GetMetricProperties_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_GetMetricNameFromCounterId(
        const LWPA_RawMetricsConfig* pRawMetricsConfig,
        uint64_t counterId,
        const char** pMetricName);

    typedef struct LWPW_RawMetricsConfig_GetMetricNameFromCounterId_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const LWPA_RawMetricsConfig* pRawMetricsConfig;
        uint64_t counterId;
        /// [out]
        const char* metricName;
    } LWPW_RawMetricsConfig_GetMetricNameFromCounterId_Params;
#define LWPW_RawMetricsConfig_GetMetricNameFromCounterId_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_GetMetricNameFromCounterId_Params, metricName)

    LWPA_Status LWPW_RawMetricsConfig_GetMetricNameFromCounterId(LWPW_RawMetricsConfig_GetMetricNameFromCounterId_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_AddMetrics(
        LWPA_RawMetricsConfig* pRawMetricsConfig,
        const LWPA_RawMetricRequest* pRawMetricRequests,
        size_t numMetricRequests);

    typedef struct LWPW_RawMetricsConfig_AddMetrics_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_RawMetricsConfig* pRawMetricsConfig;
        const LWPA_RawMetricRequest* pRawMetricRequests;
        size_t numMetricRequests;
    } LWPW_RawMetricsConfig_AddMetrics_Params;
#define LWPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_AddMetrics_Params, numMetricRequests)

    LWPA_Status LWPW_RawMetricsConfig_AddMetrics(LWPW_RawMetricsConfig_AddMetrics_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_IsAddMetricsPossible(
        const LWPA_RawMetricsConfig* pRawMetricsConfig,
        const LWPA_RawMetricRequest* pRawMetricRequests,
        size_t numMetricRequests,
        LWPA_Bool* pIsPossible);

    typedef struct LWPW_RawMetricsConfig_IsAddMetricsPossible_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const LWPA_RawMetricsConfig* pRawMetricsConfig;
        const LWPA_RawMetricRequest* pRawMetricRequests;
        size_t numMetricRequests;
        /// [out]
        LWPA_Bool isPossible;
    } LWPW_RawMetricsConfig_IsAddMetricsPossible_Params;
#define LWPW_RawMetricsConfig_IsAddMetricsPossible_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_IsAddMetricsPossible_Params, isPossible)

    LWPA_Status LWPW_RawMetricsConfig_IsAddMetricsPossible(LWPW_RawMetricsConfig_IsAddMetricsPossible_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_GenerateConfigImage(LWPA_RawMetricsConfig* pRawMetricsConfig);

    typedef struct LWPW_RawMetricsConfig_GenerateConfigImage_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_RawMetricsConfig* pRawMetricsConfig;
        /// [in] If true, all existing pass groups may be merged to reduce number of passes.
        /// If merge was successful, distribution of counters in passes may be updated as a side-effect. The effects
        /// will be persistent in pRawMetricsConfig.
        LWPA_Bool mergeAllPassGroups;
    } LWPW_RawMetricsConfig_GenerateConfigImage_Params;
#define LWPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_GenerateConfigImage_Params, mergeAllPassGroups)

    /// This API may fail if called inside a pass group with `mergeAllPassGroups` = true.
    LWPA_Status LWPW_RawMetricsConfig_GenerateConfigImage(LWPW_RawMetricsConfig_GenerateConfigImage_Params* pParams);

    LWPA_Status LWPA_RawMetricsConfig_GetConfigImage(
        const LWPA_RawMetricsConfig* pRawMetricsConfig,
        size_t bufferSize,
        uint8_t* pBuffer,
        size_t* pBufferSize);

    typedef struct LWPW_RawMetricsConfig_GetConfigImage_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const LWPA_RawMetricsConfig* pRawMetricsConfig;
        /// [in] Number of bytes allocated for pBuffer
        size_t bytesAllocated;
        /// [out] [optional] Buffer receiving the config image
        uint8_t* pBuffer;
        /// [out] Count of bytes that would be copied into pBuffer
        size_t bytesCopied;
    } LWPW_RawMetricsConfig_GetConfigImage_Params;
#define LWPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_GetConfigImage_Params, bytesCopied)

    LWPA_Status LWPW_RawMetricsConfig_GetConfigImage(LWPW_RawMetricsConfig_GetConfigImage_Params* pParams);

    /// Total num passes = *pNumPipelinedPasses + *pNumIsolatedPasses * numNestingLevels
    LWPA_Status LWPA_RawMetricsConfig_GetNumPasses(
        const LWPA_RawMetricsConfig* pRawMetricsConfig,
        size_t* pNumPipelinedPasses,
        size_t* pNumIsolatedPasses);

    typedef struct LWPW_RawMetricsConfig_GetNumPasses_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const LWPA_RawMetricsConfig* pRawMetricsConfig;
        /// [out]
        size_t numPipelinedPasses;
        /// [out]
        size_t numIsolatedPasses;
    } LWPW_RawMetricsConfig_GetNumPasses_Params;
#define LWPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_GetNumPasses_Params, numIsolatedPasses)

    /// Total num passes = numPipelinedPasses + numIsolatedPasses * numNestingLevels
    LWPA_Status LWPW_RawMetricsConfig_GetNumPasses(LWPW_RawMetricsConfig_GetNumPasses_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name   CounterData Creation
 *  @{
 */

    typedef struct LWPA_CounterDataBuilder LWPA_CounterDataBuilder;

    typedef struct LWPA_CounterDataBuilderOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const char* pChipName;
    } LWPA_CounterDataBuilderOptions;
#define LWPA_COUNTER_DATA_BUILDER_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_CounterDataBuilderOptions, pChipName)

    LWPA_Status LWPA_CounterDataBuilder_Create(
        const LWPA_CounterDataBuilderOptions* pOptions,
        LWPA_CounterDataBuilder** ppCounterDataBuilder);

    typedef struct LWPW_CounterDataBuilder_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [out]
        LWPA_CounterDataBuilder* pCounterDataBuilder;
        const char* pChipName;
    } LWPW_CounterDataBuilder_Create_Params;
#define LWPW_CounterDataBuilder_Create_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataBuilder_Create_Params, pChipName)

    LWPA_Status LWPW_CounterDataBuilder_Create(LWPW_CounterDataBuilder_Create_Params* pParams);

    LWPA_Status LWPA_CounterDataBuilder_Destroy(LWPA_CounterDataBuilder* pCounterDataBuilder);

    typedef struct LWPW_CounterDataBuilder_Destroy_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataBuilder* pCounterDataBuilder;
    } LWPW_CounterDataBuilder_Destroy_Params;
#define LWPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataBuilder_Destroy_Params, pCounterDataBuilder)

    LWPA_Status LWPW_CounterDataBuilder_Destroy(LWPW_CounterDataBuilder_Destroy_Params* pParams);

    LWPA_Status LWPA_CounterDataBuilder_AddMetrics(
        LWPA_CounterDataBuilder* pCounterDataBuilder,
        const LWPA_RawMetricRequest* pRawMetricRequests,
        size_t numMetricRequests);

    typedef struct LWPW_CounterDataBuilder_AddMetrics_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataBuilder* pCounterDataBuilder;
        const LWPA_RawMetricRequest* pRawMetricRequests;
        size_t numMetricRequests;
    } LWPW_CounterDataBuilder_AddMetrics_Params;
#define LWPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataBuilder_AddMetrics_Params, numMetricRequests)

    LWPA_Status LWPW_CounterDataBuilder_AddMetrics(LWPW_CounterDataBuilder_AddMetrics_Params* pParams);

    LWPA_Status LWPA_CounterDataBuilder_GetCounterDataPrefix(
        LWPA_CounterDataBuilder* pCounterDataBuilder,
        size_t bufferSize,
        uint8_t* pBuffer,
        size_t* pBufferSize);

    typedef struct LWPW_CounterDataBuilder_GetCounterDataPrefix_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_CounterDataBuilder* pCounterDataBuilder;
        /// [in] Number of bytes allocated for pBuffer
        size_t bytesAllocated;
        /// [out] [optional] Buffer receiving the counter data prefix
        uint8_t* pBuffer;
        /// [out] Count of bytes that would be copied to pBuffer
        size_t bytesCopied;
    } LWPW_CounterDataBuilder_GetCounterDataPrefix_Params;
#define LWPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterDataBuilder_GetCounterDataPrefix_Params, bytesCopied)

    LWPA_Status LWPW_CounterDataBuilder_GetCounterDataPrefix(LWPW_CounterDataBuilder_GetCounterDataPrefix_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name   MetricsContext - metric configuration and evaluation
 *  @{
 */

    typedef struct LWPA_MetricsContext LWPA_MetricsContext;

    typedef struct LWPA_MetricsContextOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const char* pChipName;
    } LWPA_MetricsContextOptions;
#define LWPA_METRICS_CONTEXT_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetricsContextOptions, pChipName)

    typedef struct LWPA_MetricsScriptOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// in : if true, upon error, calls PyErr_Print() which causes exceptions to be logged to stderr
        LWPA_Bool printErrors;
        /// in : the script source code
        const char* pSource;
        /// in : the filename reported in stack traces; if NULL, uses an auto-generated name
        const char* pFileName;
    } LWPA_MetricsScriptOptions;
#define LWPA_METRICS_SCRIPT_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetricsScriptOptions, pFileName)

    typedef struct LWPA_MetricsExecOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// in : if true, treats pSource as a statement to be eval'd; otherwise, calls exec.
        LWPA_Bool isStatement;
        /// in : if true, upon error, calls PyErr_Print() which causes exceptions to be logged to stderr
        LWPA_Bool printErrors;
        /// in : the script source code
        const char* pSource;
        /// in : the filename reported in stack traces; if NULL, uses an auto-generated name
        const char* pFileName;
        /// out: if isStatement, points at a string form of the evaluation; if !isStatement, points at
        /// str(locals()['result'])
        const char* pResultStr;
    } LWPA_MetricsExecOptions;
#define LWPA_METRICS_EXEC_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetricsExecOptions, pResultStr)

    typedef struct LWPA_MetricsEnumerationOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out: number of elements in array ppMetricNames
        size_t numMetrics;
        /// out: pointer to array of 'const char* pMetricName'
        const char* const* ppMetricNames;
        /// in : if true, doesn't enumerate \<metric\>.peak_{burst, sustained}
        LWPA_Bool hidePeakSubMetrics;
        /// in : if true, doesn't enumerate \<metric\>.per_{active,elapsed,region,frame}_cycle
        LWPA_Bool hidePerCycleSubMetrics;
        /// in : if true, doesn't enumerate \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
        LWPA_Bool hidePctOfPeakSubMetrics;
        /// in : if false, enumerate \<unit\>__throughput.pct_of_peak_sustained_elapsed even if hidePctOfPeakSubMetrics
        /// is true
        LWPA_Bool hidePctOfPeakSubMetricsOnThroughputs;
    } LWPA_MetricsEnumerationOptions;
#define LWPA_METRICS_ENUMERATION_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetricsEnumerationOptions, hidePctOfPeakSubMetricsOnThroughputs)

    typedef struct LWPA_MetricProperties
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out
        const char* pDescription;
        /// out
        const char* pDimUnits;
        /// out: a NULL-terminated array of pointers to RawMetric names that can be passed to
        /// LWPW_RawMetricsConfig_AddMetrics()
        const char** ppRawMetricDependencies;
        /// out: metric.peak_burst.value.gpu
        double gpuBurstRate;
        /// out: metric.peak_sustained.value.gpu
        double gpuSustainedRate;
    } LWPA_MetricProperties;
#define LWPA_METRIC_PROPERTIES_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetricProperties, gpuSustainedRate)

    typedef struct LWPA_MetrilwserData
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// in: duration in ns of user defined frame
        double frame_duration;
        /// in: duration in ns of user defined region
        double region_duration;
    } LWPA_MetrilwserData;
#define LWPA_METRIC_USER_DATA_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_MetrilwserData, region_duration)

    typedef enum LWPA_MetricDetailLevel
    {
        LWPA_METRIC_DETAIL_LEVEL_ILWALID,
        LWPA_METRIC_DETAIL_LEVEL_GPU,
        LWPA_METRIC_DETAIL_LEVEL_ALL,
        LWPA_METRIC_DETAIL_LEVEL_GPU_AND_LEAF_INSTANCES,
        LWPA_METRIC_DETAIL_LEVEL__COUNT
    } LWPA_MetricDetailLevel;

    LWPA_Status LWPA_MetricsContext_Create(
        const LWPA_MetricsContextOptions* pMetricsContextOptions,
        LWPA_MetricsContext** ppMetricsContext);

    LWPA_Status LWPA_MetricsContext_Destroy(LWPA_MetricsContext* pMetricsContext);

    typedef struct LWPW_MetricsContext_Destroy_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_Destroy_Params;
#define LWPW_MetricsContext_Destroy_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_Destroy_Params, pMetricsContext)

    LWPA_Status LWPW_MetricsContext_Destroy(LWPW_MetricsContext_Destroy_Params* pParams);

    /// Runs code in the metrics module.  Additional metrics can be added through this interface.
    /// If printErrors is true, calls PyErr_Print() which causes exceptions to be logged to stderr.
    /// Equivalent to:
    ///      exec(source, metrics.__dict__, metrics.__dict__)
    LWPA_Status LWPA_MetricsContext_RunScript(
        LWPA_MetricsContext* pMetricsContext,
        const LWPA_MetricsScriptOptions* pOptions);

    typedef struct LWPW_MetricsContext_RunScript_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// in : if true, upon error, calls PyErr_Print() which causes exceptions to be logged to stderr
        LWPA_Bool printErrors;
        /// in : the script source code
        const char* pSource;
        /// in : the filename reported in stack traces; if NULL, uses an auto-generated name
        const char* pFileName;
    } LWPW_MetricsContext_RunScript_Params;
#define LWPW_MetricsContext_RunScript_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_RunScript_Params, pFileName)

    /// Runs code in the metrics module.  Additional metrics can be added through this interface.
    /// If printErrors is true, calls PyErr_Print() which causes exceptions to be logged to stderr.
    /// Equivalent to:
    ///      exec(source, metrics.__dict__, metrics.__dict__)
    LWPA_Status LWPW_MetricsContext_RunScript(LWPW_MetricsContext_RunScript_Params* pParams);

    /// Exelwtes a script in the metrics module, but does not modify its contents (for ordinary queries).
    /// Equivalent to one of:
    ///      eval(source, metrics.__dict__, {})            # isStatement true
    ///      exec(source, metrics.__dict__, {})            # isStatement false
    LWPA_Status LWPA_MetricsContext_ExecScript_Begin(
        LWPA_MetricsContext* pMetricsContext,
        LWPA_MetricsExecOptions* pOptions);

    typedef struct LWPW_MetricsContext_ExecScript_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// in : if true, treats pSource as a statement to be eval'd; otherwise, calls exec.
        LWPA_Bool isStatement;
        /// in : if true, upon error, calls PyErr_Print() which causes exceptions to be logged to stderr
        LWPA_Bool printErrors;
        /// in : the script source code
        const char* pSource;
        /// in : the filename reported in stack traces; if NULL, uses an auto-generated name
        const char* pFileName;
        /// out: if isStatement, points at a string form of the evaluation; if !isStatement, points at
        /// str(locals()['result'])
        const char* pResultStr;
    } LWPW_MetricsContext_ExecScript_Begin_Params;
#define LWPW_MetricsContext_ExecScript_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_ExecScript_Begin_Params, pResultStr)

    /// Exelwtes a script in the metrics module, but does not modify its contents (for ordinary queries).
    /// Equivalent to one of:
    ///      eval(source, metrics.__dict__, {})            # isStatement true
    ///      exec(source, metrics.__dict__, {})            # isStatement false
    LWPA_Status LWPW_MetricsContext_ExecScript_Begin(LWPW_MetricsContext_ExecScript_Begin_Params* pParams);

    /// Cleans up memory internally allocated by LWPA_MetricsContext_ExecScript_Begin.
    LWPA_Status LWPA_MetricsContext_ExecScript_End(LWPA_MetricsContext* pMetricsContext);

    typedef struct LWPW_MetricsContext_ExecScript_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_ExecScript_End_Params;
#define LWPW_MetricsContext_ExecScript_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_ExecScript_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_ExecScript_Begin.
    LWPA_Status LWPW_MetricsContext_ExecScript_End(LWPW_MetricsContext_ExecScript_End_Params* pParams);

    /// Outputs (size, pointer) to an array of "const char* pCounterName".  The lifetime of the array is tied to
    /// MetricsContext.  The names are sorted.
    /// Impl: lazily creates list
    LWPA_Status LWPA_MetricsContext_GetCounterNames_Begin(
        LWPA_MetricsContext* pMetricsContext,
        size_t* pNumCounters,
        const char* const** pppCounterNames);

    typedef struct LWPW_MetricsContext_GetCounterNames_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// [out]
        size_t numCounters;
        /// [out]
        const char* const* ppCounterNames;
    } LWPW_MetricsContext_GetCounterNames_Begin_Params;
#define LWPW_MetricsContext_GetCounterNames_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetCounterNames_Begin_Params, ppCounterNames)

    /// Outputs (size, pointer) to an array of "const char* pCounterName".  The lifetime of the array is tied to
    /// MetricsContext.  The names are sorted.
    /// Impl: lazily creates list
    LWPA_Status LWPW_MetricsContext_GetCounterNames_Begin(LWPW_MetricsContext_GetCounterNames_Begin_Params* pParams);

    /// Cleans up memory internally allocated by LWPA_MetricsContext_GetCounterNames_Begin.
    LWPA_Status LWPA_MetricsContext_GetCounterNames_End(LWPA_MetricsContext* pMetricsContext);

    typedef struct LWPW_MetricsContext_GetCounterNames_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetCounterNames_End_Params;
#define LWPW_MetricsContext_GetCounterNames_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetCounterNames_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetCounterNames_Begin.
    LWPA_Status LWPW_MetricsContext_GetCounterNames_End(LWPW_MetricsContext_GetCounterNames_End_Params* pParams);

    /// Outputs (size, pointer) to an array of "const char* pThroughputName".  The lifetime of the array is tied to
    /// MetricsContext.  The names are sorted.
    /// Impl: lazily creates list
    LWPA_Status LWPA_MetricsContext_GetThroughputNames_Begin(
        LWPA_MetricsContext* pMetricsContext,
        size_t* pNumThroughputs,
        const char* const** pppThroughputName);

    typedef struct LWPW_MetricsContext_GetThroughputNames_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// [out]
        size_t numThroughputs;
        /// [out]
        const char* const* ppThroughputNames;
    } LWPW_MetricsContext_GetThroughputNames_Begin_Params;
#define LWPW_MetricsContext_GetThroughputNames_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetThroughputNames_Begin_Params, ppThroughputNames)

    /// Outputs (size, pointer) to an array of "const char* pThroughputName".  The lifetime of the array is tied to
    /// MetricsContext.  The names are sorted.
    /// Impl: lazily creates list
    LWPA_Status LWPW_MetricsContext_GetThroughputNames_Begin(LWPW_MetricsContext_GetThroughputNames_Begin_Params* pParams);

    /// Cleans up memory internally allocated by LWPA_MetricsContext_GetThroughputNames_Begin.
    LWPA_Status LWPA_MetricsContext_GetThroughputNames_End(LWPA_MetricsContext* pMetricsContext);

    typedef struct LWPW_MetricsContext_GetThroughputNames_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetThroughputNames_End_Params;
#define LWPW_MetricsContext_GetThroughputNames_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetThroughputNames_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetThroughputNames_Begin.
    LWPA_Status LWPW_MetricsContext_GetThroughputNames_End(LWPW_MetricsContext_GetThroughputNames_End_Params* pParams);

    typedef struct LWPW_MetricsContext_GetRatioNames_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// [out]
        size_t numRatios;
        /// [out]
        const char* const* ppRatioNames;
    } LWPW_MetricsContext_GetRatioNames_Begin_Params;
#define LWPW_MetricsContext_GetRatioNames_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetRatioNames_Begin_Params, ppRatioNames)

    /// Outputs (size, pointer) to an array of "const char* pRatioName".  The lifetime of the array is tied to
    /// MetricsContext.  The names are sorted.
    /// Impl: lazily creates list
    LWPA_Status LWPW_MetricsContext_GetRatioNames_Begin(LWPW_MetricsContext_GetRatioNames_Begin_Params* pParams);

    typedef struct LWPW_MetricsContext_GetRatioNames_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetRatioNames_End_Params;
#define LWPW_MetricsContext_GetRatioNames_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetRatioNames_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetCounterNames_Begin.
    LWPA_Status LWPW_MetricsContext_GetRatioNames_End(LWPW_MetricsContext_GetRatioNames_End_Params* pParams);

    /// Outputs (size, pointer) to an array of "const char* pMetricName".  The lifetime of the array is tied to
    /// MetricsContext.  The names are sorted.
    /// Enumerates all metrics at all levels.  Includes:
    ///  *   counter.{sum,avg,min,max}
    ///  *   throughput.{avg,min,max}
    ///  *   \<metric\>.peak_{burst, sustained}
    ///  *   \<metric\>.per_{active,elapsed,region,frame}_cycle
    ///  *   \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
    ///  *   \<metric\>.per.{other, other_pct}
    LWPA_Status LWPA_MetricsContext_GetMetricNames_Begin(
        LWPA_MetricsContext* pMetricsContext,
        LWPA_MetricsEnumerationOptions* pOptions);

    typedef struct LWPW_MetricsContext_GetMetricNames_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// out: number of elements in array ppMetricNames
        size_t numMetrics;
        /// out: pointer to array of 'const char* pMetricName'
        const char* const* ppMetricNames;
        /// in : if true, doesn't enumerate \<metric\>.peak_{burst, sustained}
        LWPA_Bool hidePeakSubMetrics;
        /// in : if true, doesn't enumerate \<metric\>.per_{active,elapsed,region,frame}_cycle
        LWPA_Bool hidePerCycleSubMetrics;
        /// in : if true, doesn't enumerate \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
        LWPA_Bool hidePctOfPeakSubMetrics;
        /// in : if false, enumerate \<unit\>__throughput.pct_of_peak_sustained_elapsed even if hidePctOfPeakSubMetrics
        /// is true
        LWPA_Bool hidePctOfPeakSubMetricsOnThroughputs;
    } LWPW_MetricsContext_GetMetricNames_Begin_Params;
#define LWPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricNames_Begin_Params, hidePctOfPeakSubMetricsOnThroughputs)

    /// Outputs (size, pointer) to an array of "const char* pMetricName".  The lifetime of the array is tied to
    /// MetricsContext.  The names are sorted.
    /// Enumerates all metrics at all levels.  Includes:
    ///  *   counter.{sum,avg,min,max}
    ///  *   throughput.{avg,min,max}
    ///  *   \<metric\>.peak_{burst, sustained}
    ///  *   \<metric\>.per_{active,elapsed,region,frame}_cycle
    ///  *   \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
    ///  *   \<metric\>.per.{other, other_pct}
    LWPA_Status LWPW_MetricsContext_GetMetricNames_Begin(LWPW_MetricsContext_GetMetricNames_Begin_Params* pParams);

    /// Cleans up memory internally allocated by LWPA_MetricsContext_GetMetricNames_Begin.
    LWPA_Status LWPA_MetricsContext_GetMetricNames_End(LWPA_MetricsContext* pMetricsContext);

    typedef struct LWPW_MetricsContext_GetMetricNames_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetMetricNames_End_Params;
#define LWPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricNames_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetMetricNames_Begin.
    LWPA_Status LWPW_MetricsContext_GetMetricNames_End(LWPW_MetricsContext_GetMetricNames_End_Params* pParams);

    /// After this function returns, the lifetimes of strings pointed to by {ppCounterNames, ppSubThroughputNames,
    /// ppSubMetricNames} are guaranteed until LWPA_MetricsContext_GetThroughputBreakdown_End, or until pMetricsContext
    /// is destroyed
    LWPA_Status LWPA_MetricsContext_GetThroughputBreakdown_Begin(
        LWPA_MetricsContext* pMetricsContext,
        const char* pThroughputName,
        const char* const** pppCounterNames,
        const char* const** pppSubThroughputNames);

    typedef struct LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        const char* pThroughputName;
        const char* const* ppCounterNames;
        const char* const* ppSubThroughputNames;
    } LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params;
#define LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params, ppSubThroughputNames)

    /// After this function returns, the lifetimes of strings pointed to by {ppCounterNames, ppSubThroughputNames,
    /// ppSubMetricNames} are guaranteed until LWPW_MetricsContext_GetThroughputBreakdown_End, or until pMetricsContext
    /// is destroyed
    LWPA_Status LWPW_MetricsContext_GetThroughputBreakdown_Begin(LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params* pParams);

    /// Cleans up memory internally allocated by LWPA_MetricsContext_GetThroughputBreakdown_Begin.
    LWPA_Status LWPA_MetricsContext_GetThroughputBreakdown_End(LWPA_MetricsContext* pMetricsContext);

    typedef struct LWPW_MetricsContext_GetThroughputBreakdown_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetThroughputBreakdown_End_Params;
#define LWPW_MetricsContext_GetThroughputBreakdown_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetThroughputBreakdown_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetThroughputBreakdown_Begin.
    LWPA_Status LWPW_MetricsContext_GetThroughputBreakdown_End(LWPW_MetricsContext_GetThroughputBreakdown_End_Params* pParams);

    /// After this function returns, the lifetimes of strings pointed to by pMetricProperties are guaranteed until
    /// LWPA_MetricsContext_GetMetricProperties_End, or until pMetricsContext is destroyed.
    LWPA_Status LWPA_MetricsContext_GetMetricProperties_Begin(
        LWPA_MetricsContext* pMetricsContext,
        const char* pMetricName,
        LWPA_MetricProperties* pMetricProperties);

    typedef struct LWPW_MetricsContext_GetMetricProperties_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        const char* pMetricName;
        /// out
        const char* pDescription;
        /// out
        const char* pDimUnits;
        /// out: a NULL-terminated array of pointers to RawMetric names that can be passed to
        /// LWPW_RawMetricsConfig_AddMetrics()
        const char** ppRawMetricDependencies;
        /// out: metric.peak_burst.value.gpu
        double gpuBurstRate;
        /// out: metric.peak_sustained.value.gpu
        double gpuSustainedRate;
    } LWPW_MetricsContext_GetMetricProperties_Begin_Params;
#define LWPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricProperties_Begin_Params, gpuSustainedRate)

    /// After this function returns, the lifetimes of strings pointed to by pMetricProperties are guaranteed until
    /// LWPW_MetricsContext_GetMetricProperties_End, or until pMetricsContext is destroyed.
    LWPA_Status LWPW_MetricsContext_GetMetricProperties_Begin(LWPW_MetricsContext_GetMetricProperties_Begin_Params* pParams);

    /// Cleans up memory internally allocated by LWPA_MetricsContext_GetMetricProperties_Begin.
    LWPA_Status LWPA_MetricsContext_GetMetricProperties_End(LWPA_MetricsContext* pMetricsContext);

    typedef struct LWPW_MetricsContext_GetMetricProperties_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetMetricProperties_End_Params;
#define LWPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricProperties_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetMetricProperties_Begin.
    LWPA_Status LWPW_MetricsContext_GetMetricProperties_End(LWPW_MetricsContext_GetMetricProperties_End_Params* pParams);

    /// Sets data for subsequent evaluation calls.
    /// Only one (CounterData, range, isolated) set of counters can be active at a time; subsequent calls will overwrite
    /// previous calls' data.
    LWPA_Status LWPA_MetricsContext_SetCounterData(
        LWPA_MetricsContext* pMetricsContext,
        const uint8_t* pCounterDataImage,
        size_t rangeIndex,
        LWPA_Bool isolated);

    typedef struct LWPW_MetricsContext_SetCounterData_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        const uint8_t* pCounterDataImage;
        size_t rangeIndex;
        LWPA_Bool isolated;
    } LWPW_MetricsContext_SetCounterData_Params;
#define LWPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_SetCounterData_Params, isolated)

    /// Sets data for subsequent evaluation calls.
    /// Only one (CounterData, range, isolated) set of counters can be active at a time; subsequent calls will overwrite
    /// previous calls' data.
    LWPA_Status LWPW_MetricsContext_SetCounterData(LWPW_MetricsContext_SetCounterData_Params* pParams);

    /// Sets user data for subsequent evaluation calls.
    LWPA_Status LWPA_MetricsContext_SetUserData(
        LWPA_MetricsContext* pMetricsContext,
        const LWPA_MetrilwserData* pMetrilwserData);

    typedef struct LWPW_MetricsContext_SetUserData_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// duration in ns of user defined frame
        double frameDuration;
        /// duration in ns of user defined region
        double regionDuration;
    } LWPW_MetricsContext_SetUserData_Params;
#define LWPW_MetricsContext_SetUserData_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_SetUserData_Params, regionDuration)

    /// Sets user data for subsequent evaluation calls.
    LWPA_Status LWPW_MetricsContext_SetUserData(LWPW_MetricsContext_SetUserData_Params* pParams);

    /// Evaluate multiple metrics to retrieve their GPU values.
    LWPA_Status LWPA_MetricsContext_EvaluateToGpuValues(
        LWPA_MetricsContext* pMetricsContext,
        size_t numMetrics,
        const char* const* ppMetricNames,
        double* pMetricValues);

    typedef struct LWPW_MetricsContext_EvaluateToGpuValues_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        size_t numMetrics;
        const char* const* ppMetricNames;
        /// [out]
        double* pMetricValues;
    } LWPW_MetricsContext_EvaluateToGpuValues_Params;
#define LWPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_EvaluateToGpuValues_Params, pMetricValues)

    /// Evaluate multiple metrics to retrieve their GPU values.
    LWPA_Status LWPW_MetricsContext_EvaluateToGpuValues(LWPW_MetricsContext_EvaluateToGpuValues_Params* pParams);

    typedef struct LWPW_MetricsContext_GetMetricSuffix_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// in: pointer to the metric name
        const char* pMetricName;
        /// out: number of elements in array ppSuffixes
        size_t numSuffixes;
        /// out: pointer to array of 'const char* pSuffixes'
        const char* const* ppSuffixes;
        /// in : if true, doesn't enumerate \<metric\>.peak_{burst, sustained}
        LWPA_Bool hidePeakSubMetrics;
        /// in : if true, doesn't enumerate \<metric\>.per_{active,elapsed,region,frame}_cycle
        LWPA_Bool hidePerCycleSubMetrics;
        /// in : if true, doesn't enumerate \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
        LWPA_Bool hidePctOfPeakSubMetrics;
        /// in : if false, enumerate \<unit\>__throughput.pct_of_peak_sustained_elapsed even if hidePctOfPeakSubMetrics
        /// is true
        LWPA_Bool hidePctOfPeakSubMetricsOnThroughputs;
    } LWPW_MetricsContext_GetMetricSuffix_Begin_Params;
#define LWPW_MetricsContext_GetMetricSuffix_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricSuffix_Begin_Params, hidePctOfPeakSubMetricsOnThroughputs)

    /// Outputs (size, pointer) to an array of "const char* pSuffixes".  The lifetime of the array is tied to
    /// MetricsContext.
    /// return all the suffixes the metric has.  the possible suffixes include:
    ///  *   counter.{sum,avg,min,max}
    ///  *   throughput.{avg,min,max}
    ///  *   \<metric\>.peak_{burst, sustained}
    ///  *   \<metric\>.per_{active,elapsed,region,frame}_cycle
    ///  *   \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
    ///  *   \<metric\>.per.{other, other_pct}
    LWPA_Status LWPW_MetricsContext_GetMetricSuffix_Begin(LWPW_MetricsContext_GetMetricSuffix_Begin_Params* pParams);

    typedef struct LWPW_MetricsContext_GetMetricSuffix_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetMetricSuffix_End_Params;
#define LWPW_MetricsContext_GetMetricSuffix_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricSuffix_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetMetricSuffix_Begin.
    LWPA_Status LWPW_MetricsContext_GetMetricSuffix_End(LWPW_MetricsContext_GetMetricSuffix_End_Params* pParams);

    typedef struct LWPW_MetricsContext_GetMetricBaseNames_Begin_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        /// out: number of elements in array pMetricsBaseNames
        size_t numMetricBaseNames;
        /// out: pointer to array of 'const char* pMetricsBaseName'
        const char* const* ppMetricBaseNames;
    } LWPW_MetricsContext_GetMetricBaseNames_Begin_Params;
#define LWPW_MetricsContext_GetMetricBaseNames_Begin_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricBaseNames_Begin_Params, ppMetricBaseNames)

    /// Outputs (size, pointer) to an array of "const char* ppMetricBaseNames".  The lifetime of the array is tied to
    /// MetricsContext.
    /// return all the metric base names.
    LWPA_Status LWPW_MetricsContext_GetMetricBaseNames_Begin(LWPW_MetricsContext_GetMetricBaseNames_Begin_Params* pParams);

    typedef struct LWPW_MetricsContext_GetMetricBaseNames_End_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
    } LWPW_MetricsContext_GetMetricBaseNames_End_Params;
#define LWPW_MetricsContext_GetMetricBaseNames_End_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_GetMetricBaseNames_End_Params, pMetricsContext)

    /// Cleans up memory internally allocated by LWPW_MetricsContext_GetMetricBaseNames_Begin.
    LWPA_Status LWPW_MetricsContext_GetMetricBaseNames_End(LWPW_MetricsContext_GetMetricBaseNames_End_Params* pParams);

    /// Evaluate a single metric, retrieving instance values from the requested unit level.
    /// If a metric was scheduled with keepInstances=true, then instance values can be queried at that metric's unit
    /// level and all of its parents.
    LWPA_Status LWPA_MetricsContext_EvaluateToInstanceValues(
        LWPA_MetricsContext* pMetricsContext,
        const char* pMetricName,
        const char* pUnitName,
        size_t numInstances,
        double* pInstanceValues,
        size_t* pNumInstances);

    typedef struct LWPW_MetricsContext_EvaluateToInstanceValues_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_MetricsContext* pMetricsContext;
        const char* pMetricName;
        const char* pUnitName;
        /// number of elements allocated for pInstanceValues
        size_t numInstances;
        double* pInstanceValues;
        /// [out] number of elements copied to pInstanceValues
        size_t instancesCopied;
    } LWPW_MetricsContext_EvaluateToInstanceValues_Params;
#define LWPW_MetricsContext_EvaluateToInstanceValues_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_MetricsContext_EvaluateToInstanceValues_Params, instancesCopied)

    /// Evaluate a single metric, retrieving instance values from the requested unit level.
    /// If a metric was scheduled with keepInstances=true, then instance values can be queried at that metric's unit
    /// level and all of its parents.
    LWPA_Status LWPW_MetricsContext_EvaluateToInstanceValues(LWPW_MetricsContext_EvaluateToInstanceValues_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 

#endif // LWPERF_HOST_API_DEFINED




#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_HOST_H
