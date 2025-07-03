#ifndef LWPERF_HOST_PRIV_H
#define LWPERF_HOST_PRIV_H

/*
 * Copyright 2014-2019  LWPU Corporation.  All rights reserved.
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

/***************************************************************************//**
 *  @name   Common Types
 *  @{
 */

#ifndef LWPERF_LWPA_STATUS_DEFINED
#define LWPERF_LWPA_STATUS_DEFINED

    /// Error codes.
    typedef enum LWPA_Status
    {
        /// Success
        LWPA_STATUS_SUCCESS = 0,
        /// Generic error.
        LWPA_STATUS_ERROR = 1,
        /// Internal error.  Please file a bug!
        LWPA_STATUS_INTERNAL_ERROR = 2,
        /// LWPA_Init() has not been called yet.
        LWPA_STATUS_NOT_INITIALIZED = 3,
        /// The LwPerfAPI DLL/DSO could not be loaded during init.
        LWPA_STATUS_NOT_LOADED = 4,
        /// The function was not found in this version of the LwPerfAPI DLL/DSO.
        LWPA_STATUS_FUNCTION_NOT_FOUND = 5,
        /// The request is intentionally not supported by LwPerfAPI.
        LWPA_STATUS_NOT_SUPPORTED = 6,
        /// The request is not implemented by this version of LwPerfAPI.
        LWPA_STATUS_NOT_IMPLEMENTED = 7,
        /// Invalid argument.
        LWPA_STATUS_ILWALID_ARGUMENT = 8,
        /// A MetricId argument does not belong to the specified LWPA_Activity or LWPA_Config.
        LWPA_STATUS_ILWALID_METRIC_ID = 9,
        /// No driver has been loaded via LWPA_*_LoadDriver().
        LWPA_STATUS_DRIVER_NOT_LOADED = 10,
        /// Failed memory allocation.
        LWPA_STATUS_OUT_OF_MEMORY = 11,
        /// The request could not be fulfilled due to the state of the current thread.
        LWPA_STATUS_ILWALID_THREAD_STATE = 12,
        /// Allocation of context object failed.
        LWPA_STATUS_FAILED_CONTEXT_ALLOC = 13,
        /// The specified GPU is not supported.
        LWPA_STATUS_UNSUPPORTED_GPU = 14,
        /// The installed LWPU driver is too old.
        LWPA_STATUS_INSUFFICIENT_DRIVER_VERSION = 15,
        /// Graphics object has not been registered via LWPA_Register*().
        LWPA_STATUS_OBJECT_NOT_REGISTERED = 16,
        /// The operation failed due to a security check.
        LWPA_STATUS_INSUFFICIENT_PRIVILEGE = 17,
        /// The request could not be fulfilled due to the state of the context.
        LWPA_STATUS_ILWALID_CONTEXT_STATE = 18,
        /// The request could not be fulfilled due to the state of the object.
        LWPA_STATUS_ILWALID_OBJECT_STATE = 19,
        /// The request could not be fulfilled because a system resource is already in use.
        LWPA_STATUS_RESOURCE_UNAVAILABLE = 20,
        /// The LWPA_*_LoadDriver() is called after the context, command queue or device is created.
        LWPA_STATUS_DRIVER_LOADED_TOO_LATE = 21,
        /// The provided buffer is not large enough.
        LWPA_STATUS_INSUFFICIENT_SPACE = 22,
        /// The API object passed to LWPA_[API]_BeginPass/LWPA_[API]_EndPass and
        /// LWPA_[API]_PushRange/LWPA_[API]_PopRange does not match with the LWPA_[API]_BeginSession.
        LWPA_STATUS_OBJECT_MISMATCH = 23,
        LWPA_STATUS__COUNT
    } LWPA_Status;


#endif // LWPERF_LWPA_STATUS_DEFINED


#ifndef LWPERF_LWPA_ACTIVITY_KIND_DEFINED
#define LWPERF_LWPA_ACTIVITY_KIND_DEFINED

    /// The configuration's activity-kind dictates which types of data may be collected.
    typedef enum LWPA_ActivityKind
    {
        /// Invalid value.
        LWPA_ACTIVITY_KIND_ILWALID = 0,
        /// A workload-centric activity for serialized and pipelined collection.
        /// 
        /// Profiler is capable of collecting both serialized and pipelined metrics.  The library introduces any
        /// synchronization required to collect serialized metrics.
        LWPA_ACTIVITY_KIND_PROFILER,
        /// A realtime activity for sampling counters from the CPU or GPU.
        LWPA_ACTIVITY_KIND_REALTIME_SAMPLED,
        /// A realtime activity for profiling counters from the CPU or GPU without CPU/GPU synchronizations.
        LWPA_ACTIVITY_KIND_REALTIME_PROFILER,
        LWPA_ACTIVITY_KIND__COUNT
    } LWPA_ActivityKind;


#endif // LWPERF_LWPA_ACTIVITY_KIND_DEFINED


#ifndef LWPERF_LWPA_BOOL_DEFINED
#define LWPERF_LWPA_BOOL_DEFINED
    /// The type used for boolean values.
    typedef uint8_t LWPA_Bool;
#endif // LWPERF_LWPA_BOOL_DEFINED

#ifndef LWPA_STRUCT_SIZE
#define LWPA_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif // LWPA_STRUCT_SIZE


#ifndef LWPERF_LWPA_GETPROCADDRESS_DEFINED
#define LWPERF_LWPA_GETPROCADDRESS_DEFINED

typedef LWPA_Status (*LWPA_GenericFn)(void);


    /// 
    /// Gets the address of a PerfWorks API function.
    /// 
    /// \return A function pointer to the function, or NULL if the function is not available.
    /// 
    /// \param pFunctionName [in] Name of the function to retrieve.
    LWPA_GenericFn LWPA_GetProcAddress(const char* pFunctionName);

#endif

#ifndef LWPERF_LWPW_SETLIBRARYLOADPATHS_DEFINED
#define LWPERF_LWPW_SETLIBRARYLOADPATHS_DEFINED


    typedef struct LWPW_SetLibraryLoadPaths_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of paths in ppPaths
        size_t numPaths;
        /// [in] array of null-terminated paths
        const char** ppPaths;
    } LWPW_SetLibraryLoadPaths_Params;
#define LWPW_SetLibraryLoadPaths_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_SetLibraryLoadPaths_Params, ppPaths)

    /// Sets library search path for \ref LWPA_InitializeHost() and \ref LWPA_InitializeTarget().
    /// \ref LWPA_InitializeHost() and \ref LWPA_InitializeTarget load the PerfWorks DLL/DSO.  This function sets
    /// ordered paths that will be searched with the LoadLibrary() or dlopen() call.
    /// If load paths are set by this function, the default set of load paths
    /// will not be attempted.
    /// Each path must point at a directory (not a file name).
    /// This function is not thread-safe.
    /// Example Usage:
    /// \code
    ///     const char* paths[] = {
    ///         "path1", "path2", etc
    ///     };
    ///     LWPW_SetLibraryLoadPaths_Params params{LWPW_SetLibraryLoadPaths_Params_STRUCT_SIZE};
    ///     params.numPaths = sizeof(paths)/sizeof(paths[0]);
    ///     params.ppPaths = paths;
    ///     LWPW_SetLibraryLoadPaths(&params);
    ///     LWPA_InitializeHost();
    ///     params.numPaths = 0;
    ///     params.ppPaths = NULL;
    ///     LWPW_SetLibraryLoadPaths(&params);
    /// \endcode
    LWPA_Status LWPW_SetLibraryLoadPaths(LWPW_SetLibraryLoadPaths_Params* pParams);

    typedef struct LWPW_SetLibraryLoadPathsW_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of paths in ppwPaths
        size_t numPaths;
        /// [in] array of null-terminated paths
        const wchar_t** ppwPaths;
    } LWPW_SetLibraryLoadPathsW_Params;
#define LWPW_SetLibraryLoadPathsW_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_SetLibraryLoadPathsW_Params, ppwPaths)

    /// Sets library search path for \ref LWPA_InitializeHost() and \ref LWPA_InitializeTarget().
    /// \ref LWPA_InitializeHost() and \ref LWPA_InitializeTarget load the PerfWorks DLL/DSO.  This function sets
    /// ordered paths that will be searched with the LoadLibrary() or dlopen() call.
    /// If load paths are set by this function, the default set of load paths
    /// will not be attempted.
    /// Each path must point at a directory (not a file name).
    /// This function is not thread-safe.
    /// Example Usage:
    /// \code
    ///     const wchar_t* wpaths[] = {
    ///         L"path1", L"path2", etc
    ///     };
    ///     LWPW_SetLibraryLoadPathsW_Params params{LWPW_SetLibraryLoadPathsW_Params_STRUCT_SIZE};
    ///     params.numPaths = sizeof(wpaths)/sizeof(wpaths[0]);
    ///     params.ppwPaths = wpaths;
    ///     LWPW_SetLibraryLoadPathsW(&params);
    ///     LWPA_InitializeHost();
    ///     params.numPaths = 0;
    ///     params.ppwPaths = NULL;
    ///     LWPW_SetLibraryLoadPathsW(&params);
    /// \endcode
    LWPA_Status LWPW_SetLibraryLoadPathsW(LWPW_SetLibraryLoadPathsW_Params* pParams);

#endif

/**
 *  @}
 ******************************************************************************/
 
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

    typedef struct LWPW_RawMetricsConfig_GetSmpcSchedulerState_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// RawMetricsConfig object for modification.
        LWPA_RawMetricsConfig* pRawMetricsConfig;
        /// [out] Receives a pointer to an Smpc::MicroSchedulerState object.
        void* pSmpcSchedulerState;
    } LWPW_RawMetricsConfig_GetSmpcSchedulerState_Params;
#define LWPW_RawMetricsConfig_GetSmpcSchedulerState_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_RawMetricsConfig_GetSmpcSchedulerState_Params, pSmpcSchedulerState)

    /// Retrieves an internal state pointer to the SMPC microscheduler.
    ///     Intended to be used judiciously for hardware verification purposes.
    /// 
    ///     Must be called, and state modifications finalized, before calling LWPA_RawMetricsConfig_EndPassGroup.
    /// 
    ///     This function returns an error code if SMPC is not available.
    LWPW_LOCAL
    LWPA_Status LWPW_RawMetricsConfig_GetSmpcSchedulerState(LWPW_RawMetricsConfig_GetSmpcSchedulerState_Params* pParams);

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
