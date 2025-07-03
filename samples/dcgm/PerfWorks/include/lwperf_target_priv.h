#ifndef LWPERF_TARGET_PRIV_H
#define LWPERF_TARGET_PRIV_H

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
 *  @file   lwperf_target_priv.h
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
 
    typedef struct LWPW_Device_GetSmHierarchy_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
        /// [in]
        size_t sliIndex;
        /// [out]
        uint32_t gpcCount;
        /// [out]
        uint32_t vsmCount;
        /// [out]
        uint32_t maxTpcPerGpcCount;
        /// [out]
        uint32_t maxSmPerTpcCount;
        /// [out]
        uint32_t maxWarpsPerSmCount;
        /// [out] either NULL or a pointer to caller allocated array that has at least \a gpcCount elements
        uint8_t* pTpcsPerGpcCount;
    } LWPW_Device_GetSmHierarchy_Params;
#define LWPW_Device_GetSmHierarchy_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Device_GetSmHierarchy_Params, pTpcsPerGpcCount)

    /// The GPU Hierarchy is
    ///  Device
    ///    GPC
    ///      TPC
    ///        SM
    /// The virtual SMID is a flat ID from 0 - N where N is the total number of SMs in the GPU.
    /// LWPU GPUs can be shipped with TPCs disabled.
    /// \a maxTpcPerGpcCount provides the maximum number of TPCs per GPCs.
    /// \a tpcsPerGpcCount provides the enabled count.
    /// 
    LWPW_LOCAL
    LWPA_Status LWPW_Device_GetSmHierarchy(LWPW_Device_GetSmHierarchy_Params* pParams);

    typedef struct LWPW_Device_VsmMapping
    {
        /// 0 - (vsmCount - 1)
        uint32_t vsmId;
        /// 0 - (gpcCount - 1)
        uint32_t gpcId;
        /// 0 - (tpcPerGpcCount - 1)
        uint32_t tpcInGpcId;
        /// 0 - (maxSmPerTpcCount - 1)
        uint32_t smInTpcId;
    } LWPW_Device_VsmMapping;
#define LWPW_Device_VsmMapping_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Device_VsmMapping, smInTpcId)

    typedef struct LWPW_Device_GetVsmMappings_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
        /// [in]
        size_t sliIndex;
        /// [in] LWPW_Device_VsmMapping_STRUCT_SIZE
        size_t vsmMappingStructSize;
        /// [inout] if pVsmMappings is NULL, then the number of vsm mapping available is returned in numVsmMappings,
        /// otherwise numVsmMappings should be set by the user to the number of elements in the pVsmMappings array, and
        /// on return the variable is overwritten with the number of elements actually written to pVsmMappings
        size_t numVsmMappings;
        /// [inout] either NULL or a pointer to an array of LWPW_Device_VsmMapping
        LWPW_Device_VsmMapping* pVsmMappings;
    } LWPW_Device_GetVsmMappings_Params;
#define LWPW_Device_GetVsmMappings_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Device_GetVsmMappings_Params, pVsmMappings)

    LWPW_LOCAL
    LWPA_Status LWPW_Device_GetVsmMappings(LWPW_Device_GetVsmMappings_Params* pParams);

/***************************************************************************//**
 *  @name   Event Configuration - generic event system that supports variable length data
 *  @{
 */

    typedef enum LWPW_Event_DomainId
    {
        LWPW_EVENT_DOMAIN_ID_ILWALID,
        LWPW_EVENT_DOMAIN_ID_SASS_INSTRUCTION,
        LWPW_EVENT_DOMAIN_ID__COUNT
    } LWPW_Event_DomainId;

    typedef enum LWPW_Event_RecordType
    {
        LWPW_EVENT_RECORD_TYPE_ILWALID,
        LWPW_EVENT_RECORD_TYPE_PC_SAMPLING_STALL_REASONS,
        LWPW_EVENT_RECORD_TYPE_PC_SAMPLING_WARP_SAMPLE_COUNT,
        LWPW_EVENT_RECORD_TYPE_SASS_COUNTERS,
        LWPW_EVENT_RECORD_TYPE_SASS_FUNCTION_EXELWTIONS,
        LWPW_EVENT_RECORD_TYPE__COUNT
    } LWPW_Event_RecordType;


#define LWPW_EVENT_VARDATA_GRANULARITY          32
#define LWPW_EVENT_VARDATA_OFFSET_MASK          (~(LWPW_EVENT_VARDATA_GRANULARITY - 1))
#define LWPW_EVENT_VARDATA_START_OFFSET_ZERO    0x01
#define LWPW_EVENT_TYPE_DOMAIN_OFFSET           20
#define LWPW_EVENT_VARDATA_INFO_OFFSET          31:5
#define LWPW_EVENT_VARDATA_INFO_BEGIN_ZERO      0:0


    typedef struct LWPW_Event_RecordHeader
    {
        /// Type of event record
        /// [31:20] EventDomainId
        /// [19: 0] EventRecordType
        uint32_t type;
        /// varData information
        /// [31: 5] = (vardataOffsetEnd >> 5)
        /// [ 4: 1] = reserved
        /// [ 0: 0] = isVardataOffsetBeginZero = "did wrap-around occur"
        uint32_t varDataInfo;
    } LWPW_Event_RecordHeader;
#define LWPW_Event_RecordHeader_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Event_RecordHeader, varDataInfo)

    typedef struct LWPW_Event_Record
    {
        LWPW_Event_RecordHeader header;
        union
        {
            uint64_t u64[3];
            uint8_t u8[24];
        } inlinePayload;
    } LWPW_Event_Record;
#define LWPW_Event_Record_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Event_Record, inlinePayload)

    typedef enum LWPW_Event_ProducerMode
    {
        LWPW_EVENT_PRODUCER_MODE_DISABLED,
        LWPW_EVENT_PRODUCER_MODE_KEEP_OLDEST,
        LWPW_EVENT_PRODUCER_MODE_KEEP_NEWEST,
        LWPW_EVENT_PRODUCER_MODE__COUNT
    } LWPW_Event_ProducerMode;

    typedef struct LWPW_Event_Stream* LWPW_Event_StreamHandle;

    typedef struct LWPW_Event_CreateEventStream_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] must be at least numRecords * sizeof(LWPW_Event_Record) in size.
        uint8_t* pRecordsBuffer;
        /// [in] size of pRecords in record count, must be at least (3 + max number of events).
        uint32_t numRecords;
        /// [in] must be at least LWPW_EVENT_VARDATA_GRANULARITY size in bytes
        uint8_t* pVarData;
        /// [in] size of pVarData in bytes
        uint32_t varDataBufferSizeInBytes;
        /// [in]
        LWPW_Event_ProducerMode producerMode;
        /// [out]
        LWPW_Event_StreamHandle hEventStream;
    } LWPW_Event_CreateEventStream_Params;
#define LWPW_Event_CreateEventStream_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Event_CreateEventStream_Params, hEventStream)

    /// Creates an event stream.
    LWPW_LOCAL
    LWPA_Status LWPW_Event_CreateEventStream(LWPW_Event_CreateEventStream_Params* pParams);

    typedef struct LWPW_Event_DestroyEventStream_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_Event_StreamHandle hEventStream;
    } LWPW_Event_DestroyEventStream_Params;
#define LWPW_Event_DestroyEventStream_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Event_DestroyEventStream_Params, hEventStream)

    LWPW_LOCAL
    LWPA_Status LWPW_Event_DestroyEventStream(LWPW_Event_DestroyEventStream_Params* pParams);

    typedef struct LWPW_Event_SetProducerMode_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_Event_StreamHandle hEventStream;
        /// [in]
        LWPW_Event_ProducerMode producerMode;
    } LWPW_Event_SetProducerMode_Params;
#define LWPW_Event_SetProducerMode_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Event_SetProducerMode_Params, producerMode)

    /// Sets the producer mode of an event stream.
    LWPW_LOCAL
    LWPA_Status LWPW_Event_SetProducerMode(LWPW_Event_SetProducerMode_Params* pParams);

    typedef struct LWPW_Event_QueryEvents_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_Event_StreamHandle hEventStream;
        /// [out]
        uint32_t recordsPut;
        /// [out]
        uint32_t recordsGet;
        /// [out]
        uint32_t recordsCount;
        /// [out]
        uint32_t recordsDropCount;
        /// [out]
        uint32_t varDataDropCount;
    } LWPW_Event_QueryEvents_Params;
#define LWPW_Event_QueryEvents_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Event_QueryEvents_Params, varDataDropCount)

    LWPW_LOCAL
    LWPA_Status LWPW_Event_QueryEvents(LWPW_Event_QueryEvents_Params* pParams);

    typedef struct LWPW_Event_AcknowledgeEvents_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_Event_StreamHandle hEventStream;
        /// [in] 0 <= recordsGet <  numRecords
        uint32_t recordsGet;
        /// [in] 0 <  varDataGet <= varDataBufferSizeInBytes
        uint32_t varDataGet;
    } LWPW_Event_AcknowledgeEvents_Params;
#define LWPW_Event_AcknowledgeEvents_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Event_AcknowledgeEvents_Params, varDataGet)

    /// Marks records available for discard. Only applicable for KeepOldest producer mode.
    LWPW_LOCAL
    LWPA_Status LWPW_Event_AcknowledgeEvents(LWPW_Event_AcknowledgeEvents_Params* pParams);


#if defined(__cplusplus)

    namespace LWPW {

        inline uint32_t Event_CalcRecordsPrevIdx(const uint32_t numRecords, const uint32_t recordsIdx)
        {
            const uint32_t recordsPrevIdx = (recordsIdx == 0)
                ? numRecords - 1
                : recordsIdx - 1;
            return recordsPrevIdx;
        }

        inline uint32_t Event_CalcRecordsNextIdx(const uint32_t numRecords, const uint32_t recordsIdx)
        {
            const uint32_t recordsNextIdx = (recordsIdx + 1 == numRecords)
                ? 0
                : recordsIdx + 1;
            return recordsNextIdx;
        }

        inline uint32_t Event_CalcRecordsPendingCount(const uint32_t numRecords, const uint32_t recordsPut, const uint32_t recordsGet)
        {
            const uint32_t recordsPendingCount = (recordsGet < recordsPut)
                ? (recordsPut - recordsGet)
                : (recordsPut - recordsGet + numRecords);
            return recordsPendingCount;
        }

    } // namespace LWPW

#endif // defined(__cplusplus)


/**
 *  @}
 ******************************************************************************/
 


#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_TARGET_PRIV_H
