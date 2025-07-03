#ifndef LWPERF_DCGM_TARGET_PRIV_H
#define LWPERF_DCGM_TARGET_PRIV_H

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
 *  @file   lwperf_dcgm_target_priv.h
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
 
/***************************************************************************//**
 *  @name   Priv Common Types
 *  @{
 */

#ifndef LWPERF_PRIV_COMMON_TYPES
#define LWPERF_PRIV_COMMON_TYPES

    typedef enum LWPW_PcSampling_CounterDataGroup
    {
        LWPW_PC_SAMPLING_COUNTER_DATA_GROUP_PC,
        LWPW_PC_SAMPLING_COUNTER_DATA_GROUP_WARP_ID
    } LWPW_PcSampling_CounterDataGroup;

    typedef struct LWPW_LWDA_PcSampling_DecodeStats
    {
        /// number of perfmon bytes decoded
        uint32_t numPerfmonBytesDecoded;
        /// number of bytes dropped, possibly due to back pressure when sampler frequency is too high
        uint32_t droppedBytes;
        /// number of recovery attempts
        uint32_t resyncCount;
        /// one or more samplers have dropped too many bytes and are unrecoverable
        uint8_t overflow;
        /// scratch buffer is full
        uint8_t scratchBufferFull;
        uint16_t rsvd000e;
        /// number of elements in a pPcSampleOffsetData
        uint32_t numPcs;
        uint32_t numCounterValuesPerPc;
    } LWPW_LWDA_PcSampling_DecodeStats;
#define LWPW_LWDA_PcSampling_DecodeStats_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_DecodeStats, numCounterValuesPerPc)

    typedef struct LWPW_LWDA_PcSampling_PcCounterValues
    {
        uint64_t pc;
        /// 2 elements to guarantee 8 bytes aligned
        uint32_t counterValues[2];
    } LWPW_LWDA_PcSampling_PcCounterValues;
#define LWPW_LWDA_PcSampling_PcCounterValues_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_PcCounterValues, counterValues[2])


#if defined(__cplusplus)

    namespace LWPW {

        // Gets LWPW_LWDA_PcSampling_DecodeStats byte offset within varData buffer.
        inline size_t GetDecodeStatsByteOffset()
        {
            return 0;
        }

        // Gets byte offset of the first LWPW_LWDA_PcSampling_PcCounterValues within varData buffer.
        inline size_t GetFirstPcCounterValuesByteOffset(size_t numCounters)
        {
            return sizeof(LWPW_LWDA_PcSampling_DecodeStats);
        }

        // Gets stride in bytes from current to the next LWPW_LWDA_PcSampling_PcCounterValues.
        inline size_t GetPcCounterValuesStride(size_t numCounters)
        {
            size_t align = 8;
            // size of pc and two counter values
            size_t stride = sizeof(LWPW_LWDA_PcSampling_PcCounterValues);
            // size of remaining n-2 counter values, where n is the number of requested counters.
            stride += (numCounters > 2u ? 4u * (numCounters - 2u) : 0u);
            // Power of two align
            const size_t alignedStride = (stride + align - 1) & ~(size_t)(align - 1);
            return alignedStride;
        }

    } // namespace LWPW

#endif // defined(__cplusplus)


#endif


/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name   DCGM Periodic Sampling
 *  @{
 */

    typedef enum LWPW_DCGM_PeriodicSampler_TriggerSource
    {
        LWPW_DCGM_PeriodicSampler_TriggerSource_CPUTrigger
    } LWPW_DCGM_PeriodicSampler_TriggerSource;

    typedef struct LWPW_DCGM_PeriodicSampler_BeginSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
        /// [in]
        LWPW_DCGM_PeriodicSampler_TriggerSource triggerSource;
        /// [in] maximum length of the sample name passed to CPUTrigger_TriggerKeep, excluding the trailing NUL
        /// character.  Must be less than 256.
        size_t maxSampleNameLength;
        /// [in] maximum number of undecoded CPUTrigger_TriggerKeep
        size_t maxCPUTriggers;
    } LWPW_DCGM_PeriodicSampler_BeginSession_Params;
#define LWPW_DCGM_PeriodicSampler_BeginSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_BeginSession_Params, maxCPUTriggers)

    LWPA_Status LWPW_DCGM_PeriodicSampler_BeginSession(LWPW_DCGM_PeriodicSampler_BeginSession_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_EndSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
    } LWPW_DCGM_PeriodicSampler_EndSession_Params;
#define LWPW_DCGM_PeriodicSampler_EndSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_EndSession_Params, deviceIndex)

    LWPA_Status LWPW_DCGM_PeriodicSampler_EndSession(LWPW_DCGM_PeriodicSampler_EndSession_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_SetConfig_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
        /// [in] Config created by e.g.    lwperf2 configure   or   LWPA_RawMetricsConfig_GetConfigImage().  Must be
        /// align(8).
        const uint8_t* pConfig;
        /// [in]
        size_t configSize;
    } LWPW_DCGM_PeriodicSampler_SetConfig_Params;
#define LWPW_DCGM_PeriodicSampler_SetConfig_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_SetConfig_Params, configSize)

    LWPA_Status LWPW_DCGM_PeriodicSampler_SetConfig(LWPW_DCGM_PeriodicSampler_SetConfig_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
    } LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params;
#define LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params, deviceIndex)

    LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling(LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
    } LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params;
#define LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params, deviceIndex)

    LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling(LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
        /// [in]
        const char* pSampleName;
        /// [in] assign to strlen(pSampleName) if known; if set to zero, the library will call strlen().  Must be less
        /// than or equal to maxSampleNameLength.
        size_t sampleNameLength;
    } LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params;
#define LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params, sampleNameLength)

    LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
    } LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params;
#define LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params, deviceIndex)

    LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_CounterDataImageOptions
    {
        /// [in]
        size_t structSize;
        /// The CounterDataPrefix generated from e.g.    lwperf2 initdata   or
        /// LWPA_CounterDataBuilder_GetCounterDataPrefix().  Must be align(8).
        const uint8_t* pCounterDataPrefix;
        size_t counterDataPrefixSize;
        /// maximum length of the sample name passed to CPUTrigger_TriggerKeep, excluding the trailing NUL character.
        /// Must be less than 256.
        size_t maxSampleNameLength;
        /// maximum number of samples
        size_t maxSamples;
    } LWPW_DCGM_PeriodicSampler_CounterDataImageOptions;
#define LWPW_DCGM_PeriodicSampler_CounterDataImageOptions_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CounterDataImageOptions, maxSamples)

    typedef struct LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const LWPW_DCGM_PeriodicSampler_CounterDataImageOptions* pOptions;
        /// [out]
        size_t counterDataImageSize;
    } LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params;
#define LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params, counterDataImageSize)

    LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize(LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const LWPW_DCGM_PeriodicSampler_CounterDataImageOptions* pOptions;
        /// [in] the buffer to be written
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataImageSize;
    } LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params;
#define LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params, counterDataImageSize)

    LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize(LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t rangeIndex;
        /// [in]
        size_t numRawMetrics;
        /// [in]
        const uint64_t* pRawMetricIds;
        /// [out]
        double* pRawMetricValues;
        /// [out]
        uint16_t* pHwUnitCounts;
    } LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params;
#define LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params, pHwUnitCounts)

    LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics(LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params* pParams);

    typedef struct LWPW_DCGM_PeriodicSampler_DecodeCounters_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataImageSize;
        /// [out]
        LWPA_Bool sampleDataBufferOverflow;
        /// [out] number of samples decoded completely
        size_t numSamplesDecoded;
        /// [out] number of samples dropped due to CounterDataImage overflow
        size_t numSamplesDropped;
        /// [out] number of samples merged due to insufficient sample interval
        size_t numSamplesMerged;
    } LWPW_DCGM_PeriodicSampler_DecodeCounters_Params;
#define LWPW_DCGM_PeriodicSampler_DecodeCounters_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_DCGM_PeriodicSampler_DecodeCounters_Params, numSamplesMerged)

    LWPA_Status LWPW_DCGM_PeriodicSampler_DecodeCounters(LWPW_DCGM_PeriodicSampler_DecodeCounters_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 


#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_DCGM_TARGET_PRIV_H
