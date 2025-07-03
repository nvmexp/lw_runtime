#ifndef LWPERF_VULKAN_TARGET_H
#define LWPERF_VULKAN_TARGET_H

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
 *  @file   lwperf_vulkan_target.h
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

/***************************************************************************//**
 *  @name   External Types
 *  @{
 */


    struct VkPhysicalDevice_T;
    typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
    struct VkDevice_T;
    typedef struct VkDevice_T* VkDevice;
    struct VkQueue_T;
    typedef struct VkQueue_T* VkQueue;
    struct VkCommandBuffer_T;
    typedef struct VkCommandBuffer_T* VkCommandBuffer;


/**
 *  @}
 ******************************************************************************/
 
    typedef struct LWPW_VK_Profiler_CounterDataImageOptions
    {
        /// [in]
        size_t structSize;
        /// The CounterDataPrefix generated from e.g.    lwperf2 initdata   or
        /// LWPA_CounterDataBuilder_GetCounterDataPrefix().  Must be align(8).
        const uint8_t* pCounterDataPrefix;
        size_t counterDataPrefixSize;
        /// max number of ranges that can be specified
        uint32_t maxNumRanges;
        /// max number of RangeTree nodes; must be >= maxNumRanges
        uint32_t maxNumRangeTreeNodes;
        /// max string length of each RangeName, including the trailing NUL character
        uint32_t maxRangeNameLength;
    } LWPW_VK_Profiler_CounterDataImageOptions;
#define LWPW_VK_Profiler_CounterDataImageOptions_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImageOptions, maxRangeNameLength)

    typedef struct LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageOptionsSize;
        /// [in]
        const LWPW_VK_Profiler_CounterDataImageOptions* pOptions;
        /// [out]
        size_t counterDataImageSize;
    } LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params;
#define LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params, counterDataImageSize)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateSize(LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params* pParams);

    typedef struct LWPW_VK_Profiler_CounterDataImage_Initialize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageOptionsSize;
        /// [in]
        const LWPW_VK_Profiler_CounterDataImageOptions* pOptions;
        /// [in]
        size_t counterDataImageSize;
        /// [in] The buffer to be written.
        uint8_t* pCounterDataImage;
    } LWPW_VK_Profiler_CounterDataImage_Initialize_Params;
#define LWPW_VK_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_Initialize_Params, pCounterDataImage)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_Initialize(LWPW_VK_Profiler_CounterDataImage_Initialize_Params* pParams);

    typedef struct LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageSize;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [out]
        size_t counterDataScratchBufferSize;
    } LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params;
#define LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params, counterDataScratchBufferSize)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize(LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams);

    typedef struct LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageSize;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataScratchBufferSize;
        /// [in] The scratch buffer to be written.
        uint8_t* pCounterDataScratchBuffer;
    } LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params;
#define LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params, pCounterDataScratchBuffer)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer(LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams);

    typedef struct LWPW_VK_LoadDriver_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
    } LWPW_VK_LoadDriver_Params;
#define LWPW_VK_LoadDriver_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_LoadDriver_Params, pPriv)

    LWPA_Status LWPW_VK_LoadDriver(LWPW_VK_LoadDriver_Params* pParams);

    typedef struct LWPW_VK_Device_GetDeviceIndex_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkPhysicalDevice physicalDevice;
        /// [in]
        size_t sliIndex;
        /// [out]
        size_t deviceIndex;
    } LWPW_VK_Device_GetDeviceIndex_Params;
#define LWPW_VK_Device_GetDeviceIndex_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Device_GetDeviceIndex_Params, deviceIndex)

    LWPA_Status LWPW_VK_Device_GetDeviceIndex(LWPW_VK_Device_GetDeviceIndex_Params* pParams);

    typedef struct LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [out]
        const char* const* ppInstanceExtensionNames;
        /// [out]
        size_t numInstanceExtensionNames;
    } LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params;
#define LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params, numInstanceExtensionNames)

    LWPA_Status LWPW_VK_Profiler_GetRequiredInstanceExtensions(LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params* pParams);

    typedef struct LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [out]
        const char* const* ppDeviceExtensionNames;
        /// [out]
        size_t numDeviceExtensionNames;
    } LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params;
#define LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params, numDeviceExtensionNames)

    LWPA_Status LWPW_VK_Profiler_GetRequiredDeviceExtensions(LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params* pParams);

    typedef struct LWPW_VK_Profiler_CalcTraceBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] Maximum number of Push/Pop pairs that can be recorded in a single pass.
        size_t maxRangesPerPass;
        /// [in] for sizing internal buffers
        size_t avgRangeNameLength;
        /// [out] TraceBuffer size for a single pass.  Pass this to
        /// LWPW_VK_Profiler_BeginSession_Params::traceBufferSize.
        size_t traceBufferSize;
    } LWPW_VK_Profiler_CalcTraceBufferSize_Params;
#define LWPW_VK_Profiler_CalcTraceBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CalcTraceBufferSize_Params, traceBufferSize)

    LWPA_Status LWPW_VK_Profiler_CalcTraceBufferSize(LWPW_VK_Profiler_CalcTraceBufferSize_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_BeginSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkDevice device;
        /// [in] Lwrrently only queues from family index 0 are supported.
        VkQueue queue;
        /// [in] Set to 1 if every pass is synchronized with CPU; for asynchronous collection, increase to
        /// (softwarePipelineDepth + 2).
        size_t numTraceBuffers;
        /// [in] Size of the per-pass TraceBuffer in bytes.  The profiler allocates a numTraceBuffers * traceBufferSize
        /// internally.
        size_t traceBufferSize;
        /// [in] Maximum number of ranges that can be recorded in a single pass.
        size_t maxRangesPerPass;
        /// [in] Maximum number of kernel launches that can be recorded in a single pass.  Must be >= maxRangesPerPass.
        size_t maxLaunchesPerPass;
    } LWPW_VK_Profiler_Queue_BeginSession_Params;
#define LWPW_VK_Profiler_Queue_BeginSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_BeginSession_Params, maxLaunchesPerPass)

    LWPA_Status LWPW_VK_Profiler_Queue_BeginSession(LWPW_VK_Profiler_Queue_BeginSession_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_EndSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in] Maximum number of milliseconds to wait for pending GPU operations. Pass 0xFFFFFFFF to wait forever
        uint32_t timeout;
        /// [out] becomes true if timeout is reached while waiting for pending GPU operations.
        LWPA_Bool timeoutExpired;
    } LWPW_VK_Profiler_Queue_EndSession_Params;
#define LWPW_VK_Profiler_Queue_EndSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_EndSession_Params, timeoutExpired)

    LWPA_Status LWPW_VK_Profiler_Queue_EndSession(LWPW_VK_Profiler_Queue_EndSession_Params* pParams);

    typedef struct LWPW_VK_Queue_ServicePendingGpuOperations_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in] The number of operations to process. Passing `0` will block until EndSession is called
        uint32_t numOperations;
        /// [in] Maximum number of milliseconds to wait for pending GPU operations. Pass 0xFFFFFFFF to wait forever
        uint32_t timeout;
        /// [out] becomes true if timeout is reached while waiting for pending GPU operations.
        LWPA_Bool timeoutExpired;
    } LWPW_VK_Queue_ServicePendingGpuOperations_Params;
#define LWPW_VK_Queue_ServicePendingGpuOperations_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Queue_ServicePendingGpuOperations_Params, timeoutExpired)

    LWPA_Status LWPW_VK_Queue_ServicePendingGpuOperations(LWPW_VK_Queue_ServicePendingGpuOperations_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_SetConfig_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in] Config created by e.g.    lwperf2 configure   or   LWPA_RawMetricsConfig_GetConfigImage().  Must be
        /// align(8).
        const uint8_t* pConfig;
        size_t configSize;
        /// [in] the lowest nesting level to be profiled; must be >= 1
        uint16_t minNestingLevel;
        /// [in] the number of nesting levels to profile; must be >= 1
        uint16_t numNestingLevels;
        /// [in] Set this to zero for in-app replay.  Set this to the output of EndPass() for application replay.
        size_t passIndex;
        /// [in] Set this to minNestingLevel for in-app replay.  Set this to the output of EndPass() for application
        /// replay.
        uint16_t targetNestingLevel;
    } LWPW_VK_Profiler_Queue_SetConfig_Params;
#define LWPW_VK_Profiler_Queue_SetConfig_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_SetConfig_Params, targetNestingLevel)

    LWPA_Status LWPW_VK_Profiler_Queue_SetConfig(LWPW_VK_Profiler_Queue_SetConfig_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_ClearConfig_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
    } LWPW_VK_Profiler_Queue_ClearConfig_Params;
#define LWPW_VK_Profiler_Queue_ClearConfig_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_ClearConfig_Params, queue)

    LWPA_Status LWPW_VK_Profiler_Queue_ClearConfig(LWPW_VK_Profiler_Queue_ClearConfig_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_BeginPass_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
    } LWPW_VK_Profiler_Queue_BeginPass_Params;
#define LWPW_VK_Profiler_Queue_BeginPass_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_BeginPass_Params, queue)

    LWPA_Status LWPW_VK_Profiler_Queue_BeginPass(LWPW_VK_Profiler_Queue_BeginPass_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_EndPass_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [out] The passIndex that will be collected by the *next* BeginPass.
        size_t passIndex;
        /// [out] The targetNestingLevel that will be collected by the *next* BeginPass.
        uint16_t targetNestingLevel;
        /// [out] becomes true when the last pass has been queued to the GPU
        LWPA_Bool allPassesSubmitted;
    } LWPW_VK_Profiler_Queue_EndPass_Params;
#define LWPW_VK_Profiler_Queue_EndPass_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_EndPass_Params, allPassesSubmitted)

    LWPA_Status LWPW_VK_Profiler_Queue_EndPass(LWPW_VK_Profiler_Queue_EndPass_Params* pParams);

    typedef struct LWPW_VK_Profiler_CommandBuffer_PushRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
        /// [in] specifies the range that subsequent launches' counters will be assigned to; must not be NULL
        const char* pRangeName;
        /// [in] assign to strlen(pRangeName) if known; if set to zero, the library will call strlen()
        size_t rangeNameLength;
    } LWPW_VK_Profiler_CommandBuffer_PushRange_Params;
#define LWPW_VK_Profiler_CommandBuffer_PushRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CommandBuffer_PushRange_Params, rangeNameLength)

    LWPA_Status LWPW_VK_Profiler_CommandBuffer_PushRange(LWPW_VK_Profiler_CommandBuffer_PushRange_Params* pParams);

    typedef struct LWPW_VK_Profiler_CommandBuffer_PopRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
    } LWPW_VK_Profiler_CommandBuffer_PopRange_Params;
#define LWPW_VK_Profiler_CommandBuffer_PopRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CommandBuffer_PopRange_Params, commandBuffer)

    LWPA_Status LWPW_VK_Profiler_CommandBuffer_PopRange(LWPW_VK_Profiler_CommandBuffer_PopRange_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_DecodeCounters_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in]
        size_t counterDataImageSize;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataScratchBufferSize;
        /// [in]
        uint8_t* pCounterDataScratchBuffer;
        /// [out] number of ranges whose data was dropped in the processed pass
        size_t numRangesDropped;
        /// [out] number of bytes not written to TraceBuffer due to buffer full
        size_t numTraceBytesDropped;
        /// [out] true if a pass was successfully decoded
        LWPA_Bool onePassCollected;
        /// [out] becomes true when the last pass has been decoded
        LWPA_Bool allPassesCollected;
        /// [out] the Config decoded by this call
        const uint8_t* pConfigDecoded;
        /// [out] the passIndex decoded
        size_t passIndexDecoded;
    } LWPW_VK_Profiler_Queue_DecodeCounters_Params;
#define LWPW_VK_Profiler_Queue_DecodeCounters_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_DecodeCounters_Params, passIndexDecoded)

    LWPA_Status LWPW_VK_Profiler_Queue_DecodeCounters(LWPW_VK_Profiler_Queue_DecodeCounters_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_VULKAN_TARGET_H
