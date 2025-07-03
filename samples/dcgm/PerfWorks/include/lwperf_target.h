#ifndef LWPERF_TARGET_H
#define LWPERF_TARGET_H

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
 *  @file   lwperf_target.h
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
 
    /// \deprecated Use of this function is discouraged. Prefer \ref LWPW_InitializeTarget instead.
    LWPA_Status LWPA_InitializeTarget(void);


    // Device enumeration functions must be preceded by LWPA_<API>_LoadDriver(); any API is fine.


    /// \deprecated Use of this function is discouraged. Prefer \ref LWPW_GetDeviceCount instead.
    LWPA_Status LWPA_GetDeviceCount(size_t* pNumDevices);

    /// \deprecated Use of this function is discouraged. Prefer \ref LWPW_Device_GetNames instead.
    LWPA_Status LWPA_Device_GetNames(
        size_t deviceIndex,
        const char** ppDeviceName,
        const char** ppChipName);

    /// \deprecated Use of this function is discouraged. Prefer \ref LWPW_CounterData_GetNumRanges instead.
    LWPA_Status LWPA_CounterData_GetNumRanges(
        const uint8_t* pCounterDataImage,
        size_t* pNumRanges);

    /// \deprecated Use of this function is discouraged. Prefer \ref LWPW_CounterData_GetRangeDescriptions instead.
    LWPA_Status LWPA_CounterData_GetRangeDescriptions(
        const uint8_t* pCounterDataImage,
        size_t rangeIndex,
        size_t numDescriptions,
        const char** ppDescriptions,
        size_t* pNumDescriptions);

    typedef struct LWPW_InitializeTarget_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
    } LWPW_InitializeTarget_Params;
#define LWPW_InitializeTarget_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_InitializeTarget_Params, pPriv)

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

    typedef struct LWPW_PciBusId
    {
        /// The PCI domain on which the device bus resides.
        uint32_t domain;
        ///  The bus on which the device resides.
        uint16_t bus;
        /// device ID.
        uint16_t device;
    } LWPW_PciBusId;
#define LWPW_PciBusId_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PciBusId, device)

    typedef struct LWPW_Device_GetPciBusIds_Params
    {
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in] caller-allocated array of device bus IDs, indexed by LWPW deviceIndex
    LWPW_PciBusId* pBusIds;
    /// size of the pBusIDs array; use result from LWPW_GetDeviceCount
    size_t numDevices;
    } LWPW_Device_GetPciBusIds_Params;
#define LWPW_Device_GetPciBusIds_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Device_GetPciBusIds_Params, numDevices)

    LWPA_Status LWPW_Device_GetPciBusIds(LWPW_Device_GetPciBusIds_Params* pParams);

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

    typedef struct LWPW_PeriodicSampler_CounterData_DelimiterInfo
    {
        const char* pDelimiterName;
        /// defines a half-open interval [rangeIndexStart, rangeIndexEnd)
        uint32_t rangeIndexStart;
        uint32_t rangeIndexEnd;
    } LWPW_PeriodicSampler_CounterData_DelimiterInfo;
#define LWPW_PeriodicSampler_CounterData_DelimiterInfo_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PeriodicSampler_CounterData_DelimiterInfo, rangeIndexEnd)

    typedef struct LWPW_PeriodicSampler_CounterData_GetDelimiters_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t delimiterInfoStructSize;
        /// [inout] if pDelimiters is NULL, then the number of delimiters available is returned in numDelimiters,
        /// otherwise numDelimiters should be set by the user to the number of elements in the pDelimiters array, and on
        /// return the variable is overwritten with the number of elements actually written to pDelimiters
        size_t numDelimiters;
        /// [inout] either NULL or a pointer to an array of LWPW_Sampler_CounterData_DelimiterInfo
        LWPW_PeriodicSampler_CounterData_DelimiterInfo* pDelimiters;
    } LWPW_PeriodicSampler_CounterData_GetDelimiters_Params;
#define LWPW_PeriodicSampler_CounterData_GetDelimiters_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PeriodicSampler_CounterData_GetDelimiters_Params, pDelimiters)

    LWPA_Status LWPW_PeriodicSampler_CounterData_GetDelimiters(LWPW_PeriodicSampler_CounterData_GetDelimiters_Params* pParams);

    typedef struct LWPW_PeriodicSampler_CounterData_GetSampleTime_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t rangeIndex;
        /// [out]
        uint64_t timestampStart;
        /// [out]
        uint64_t timestampEnd;
    } LWPW_PeriodicSampler_CounterData_GetSampleTime_Params;
#define LWPW_PeriodicSampler_CounterData_GetSampleTime_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PeriodicSampler_CounterData_GetSampleTime_Params, timestampEnd)

    LWPA_Status LWPW_PeriodicSampler_CounterData_GetSampleTime(LWPW_PeriodicSampler_CounterData_GetSampleTime_Params* pParams);

    typedef struct LWPW_PeriodicSampler_CounterData_TrimInPlace_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataImageSize;
        /// [out]
        size_t counterDataImageTrimmedSize;
    } LWPW_PeriodicSampler_CounterData_TrimInPlace_Params;
#define LWPW_PeriodicSampler_CounterData_TrimInPlace_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PeriodicSampler_CounterData_TrimInPlace_Params, counterDataImageTrimmedSize)

    LWPA_Status LWPW_PeriodicSampler_CounterData_TrimInPlace(LWPW_PeriodicSampler_CounterData_TrimInPlace_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_TARGET_H
