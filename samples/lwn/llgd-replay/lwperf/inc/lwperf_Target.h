#ifndef LWPERF_TARGET_H
#define LWPERF_TARGET_H

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
 *  @file   lwperf_target.h
 */

    /// Load the target
    LWPA_Status LWPA_InitializeTarget(void);


    // Device enumeration functions must be preceded by LWPA_<API>_LoadDriver(); any API is fine.


    LWPA_Status LWPA_GetDeviceCount(size_t* pNumDevices);

    LWPA_Status LWPA_Device_GetNames(
        size_t deviceIndex,
        const char** ppDeviceName,
        const char** ppChipName);

    LWPA_Status LWPA_CounterData_GetNumRanges(
        const uint8_t* pCounterDataImage,
        size_t* pNumRanges);

    LWPA_Status LWPA_CounterData_GetRangeDescriptions(
        const uint8_t* pCounterDataImage,
        size_t rangeIndex,
        size_t numDescriptions,
        const char** ppDescriptions,
        size_t* pNumDescriptions);

#ifndef LWPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_DEFINED
#define LWPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_DEFINED
    /// GPU architecture support level
    typedef enum LWPW_GpuArchitectureSupportLevel
    {
        LWPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_UNKNOWN = 0,
        LWPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_UNSUPPORTED,
        LWPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_SUPPORTED
    } LWPW_GpuArchitectureSupportLevel;
#endif //LWPW_GPU_ARCHITECTURE_SUPPORT_LEVEL_DEFINED

#ifndef LWPW_SLI_SUPPORT_LEVEL_DEFINED
#define LWPW_SLI_SUPPORT_LEVEL_DEFINED
    /// SLI configuration support level
    typedef enum LWPW_SliSupportLevel
    {
        LWPW_SLI_SUPPORT_LEVEL_UNKNOWN = 0,
        LWPW_SLI_SUPPORT_LEVEL_UNSUPPORTED,
        /// Only Non-SLI configurations are supported.
        LWPW_SLI_SUPPORT_LEVEL_SUPPORTED_NON_SLI_CONFIGURATION
    } LWPW_SliSupportLevel;
#endif //LWPW_SLI_SUPPORT_LEVEL_DEFINED

#ifndef LWPW_VGPU_SUPPORT_LEVEL_DEFINED
#define LWPW_VGPU_SUPPORT_LEVEL_DEFINED
    /// Virtualized GPU configuration support level
    typedef enum LWPW_VGpuSupportLevel
    {
        LWPW_VGPU_SUPPORT_LEVEL_UNKNOWN = 0,
        LWPW_VGPU_SUPPORT_LEVEL_UNSUPPORTED,
        /// Supported but not allowed by system admin.
        LWPW_VGPU_SUPPORT_LEVEL_SUPPORTED_DISALLOWED,
        LWPW_VGPU_SUPPORT_LEVEL_SUPPORTED_ALLOWED
    } LWPW_VGpuSupportLevel;
#endif //LWPW_VGPU_SUPPORT_LEVEL_DEFINED

#ifndef LWPW_CONF_COMPUTE_SUPPORT_LEVEL_DEFINED
#define LWPW_CONF_COMPUTE_SUPPORT_LEVEL_DEFINED
    /// Confidential Compute mode support level
    typedef enum LWPW_ConfidentialComputeSupportLevel
    {
        LWPW_CONF_COMPUTE_SUPPORT_LEVEL_UNKNOWN = 0,
        LWPW_CONF_COMPUTE_SUPPORT_LEVEL_UNSUPPORTED,
        LWPW_CONF_COMPUTE_SUPPORT_LEVEL_SUPPORTED_NON_CONF_COMPUTE_CONFIGURATION
    } LWPW_ConfidentialComputeSupportLevel;
#endif //LWPW_CONF_COMPUTE_SUPPORT_LEVEL_DEFINED

    typedef struct LWPW_InitializeTarget_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
    } LWPW_InitializeTarget_Params;
#define LWPW_InitializeTarget_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_InitializeTarget_Params, pPriv)

    /// Load the target library.
    LWPA_Status LWPW_InitializeTarget(LWPW_InitializeTarget_Params* pParams);

    typedef struct LWPW_GetDeviceCount_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        size_t numDevices;
    } LWPW_GetDeviceCount_Params;
#define LWPW_GetDeviceCount_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_GetDeviceCount_Params, numDevices)

    LWPA_Status LWPW_GetDeviceCount(LWPW_GetDeviceCount_Params* pParams);

    typedef struct LWPW_Device_GetNames_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        size_t deviceIndex;
        const char* pDeviceName;
        const char* pChipName;
    } LWPW_Device_GetNames_Params;
#define LWPW_Device_GetNames_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Device_GetNames_Params, pChipName)

    LWPA_Status LWPW_Device_GetNames(LWPW_Device_GetNames_Params* pParams);

    typedef struct LWPW_Adapter_GetDeviceIndex_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        struct IDXGIAdapter* pAdapter;
        /// [in]
        size_t sliIndex;
        /// [out]
        size_t deviceIndex;
    } LWPW_Adapter_GetDeviceIndex_Params;
#define LWPW_Adapter_GetDeviceIndex_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Adapter_GetDeviceIndex_Params, deviceIndex)

    LWPA_Status LWPW_Adapter_GetDeviceIndex(LWPW_Adapter_GetDeviceIndex_Params* pParams);

    typedef struct LWPW_CounterData_GetNumRanges_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const uint8_t* pCounterDataImage;
        size_t numRanges;
    } LWPW_CounterData_GetNumRanges_Params;
#define LWPW_CounterData_GetNumRanges_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterData_GetNumRanges_Params, numRanges)

    LWPA_Status LWPW_CounterData_GetNumRanges(LWPW_CounterData_GetNumRanges_Params* pParams);

    typedef struct LWPW_CounterData_GetChipName_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataImageSize;
        /// [out]
        const char* pChipName;
    } LWPW_CounterData_GetChipName_Params;
#define LWPW_CounterData_GetChipName_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterData_GetChipName_Params, pChipName)

    LWPA_Status LWPW_CounterData_GetChipName(LWPW_CounterData_GetChipName_Params* pParams);

    typedef struct LWPW_Config_GetNumPasses_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pConfig;
        /// [out]
        size_t numPipelinedPasses;
        /// [out]
        size_t numIsolatedPasses;
    } LWPW_Config_GetNumPasses_Params;
#define LWPW_Config_GetNumPasses_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Config_GetNumPasses_Params, numIsolatedPasses)

    /// Total num passes = numPipelinedPasses + numIsolatedPasses * numNestingLevels
    LWPA_Status LWPW_Config_GetNumPasses(LWPW_Config_GetNumPasses_Params* pParams);

    typedef struct LWPW_CounterData_GetRangeDescriptions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const uint8_t* pCounterDataImage;
        size_t rangeIndex;
        /// [inout] Number of descriptions allocated in ppDescriptions
        size_t numDescriptions;
        const char** ppDescriptions;
    } LWPW_CounterData_GetRangeDescriptions_Params;
#define LWPW_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_CounterData_GetRangeDescriptions_Params, ppDescriptions)

    LWPA_Status LWPW_CounterData_GetRangeDescriptions(LWPW_CounterData_GetRangeDescriptions_Params* pParams);

    typedef struct LWPW_Profiler_CounterData_GetRangeDescriptions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        const uint8_t* pCounterDataImage;
        size_t rangeIndex;
        /// [inout] Number of descriptions allocated in ppDescriptions
        size_t numDescriptions;
        const char** ppDescriptions;
    } LWPW_Profiler_CounterData_GetRangeDescriptions_Params;
#define LWPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Profiler_CounterData_GetRangeDescriptions_Params, ppDescriptions)

    LWPA_Status LWPW_Profiler_CounterData_GetRangeDescriptions(LWPW_Profiler_CounterData_GetRangeDescriptions_Params* pParams);

    typedef struct LWPW_Profiler_CounterData_UnpackRawMetrics_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t rangeIndex;
        /// [in] : if true, query isolated metric values
        LWPA_Bool isolated;
        /// [in]
        size_t numRawMetrics;
        /// [in]
        const uint64_t* pRawMetricIds;
        /// [out]
        double* pRawMetricValues;
        /// [out]
        uint16_t* pHwUnitCounts;
    } LWPW_Profiler_CounterData_UnpackRawMetrics_Params;
#define LWPW_Profiler_CounterData_UnpackRawMetrics_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Profiler_CounterData_UnpackRawMetrics_Params, pHwUnitCounts)

    LWPA_Status LWPW_Profiler_CounterData_UnpackRawMetrics(LWPW_Profiler_CounterData_UnpackRawMetrics_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_TARGET_H
