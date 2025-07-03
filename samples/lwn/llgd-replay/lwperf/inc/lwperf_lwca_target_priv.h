#ifndef LWPERF_LWDA_TARGET_PRIV_H
#define LWPERF_LWDA_TARGET_PRIV_H

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
#include "lwperf_target_priv.h"

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
 *  @file   lwperf_lwda_target_priv.h
 */

/***************************************************************************//**
 *  @name   External Types
 *  @{
 */


    struct LWctx_st;
    struct LWstream_st;
    struct LWfunc_st;
    struct LWPW_EventStream;
    typedef int LWdevice;


/**
 *  @}
 ******************************************************************************/
 

    typedef void(*LWPW_LWDA_pfnGetExportTable)(void);

    typedef struct LWPW_LWDA_LoadDriver_PrivParams
    {
        /// [in] handle to driver module
        void* hDriverModule;
        /// [in] function pointer to lwGetExportTable
        LWPW_LWDA_pfnGetExportTable pfnGetExportTable;
    } LWPW_LWDA_LoadDriver_PrivParams;
#define LWPW_LWDA_LoadDriver_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_LoadDriver_PrivParams, pfnGetExportTable)


    typedef void* LWPW_ShaderFunction;

    typedef struct LWPW_Priv_LWDA_FunctionFilterData
    {
        /// LWfunction of the global entry function.
        LWPW_ShaderFunction entryFunctionId;
        /// LWfunction of the exelwted function.
        LWPW_ShaderFunction functionId;
    } LWPW_Priv_LWDA_FunctionFilterData;
#define LWPW_Priv_LWDA_FunctionFilterData_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_Priv_LWDA_FunctionFilterData, functionId)

    typedef struct LWPW_LWDA_Profiler_SetFunctionFilter_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] Size of pFuncFilterData array
        size_t dataCount;
        /// [in] caller-allocated array
        LWPW_Priv_LWDA_FunctionFilterData* pFuncFilterData;
    } LWPW_LWDA_Profiler_SetFunctionFilter_Params;
#define LWPW_LWDA_Profiler_SetFunctionFilter_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_Profiler_SetFunctionFilter_Params, pFuncFilterData)

    /// Sets the list of functions to collect sass counters, before any Passes have begun.
    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_Profiler_SetFunctionFilter(LWPW_LWDA_Profiler_SetFunctionFilter_Params* pParams);

    typedef struct LWPW_LWDA_RedirectToOpenCL_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
    } LWPW_LWDA_RedirectToOpenCL_Params;
#define LWPW_LWDA_RedirectToOpenCL_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_RedirectToOpenCL_Params, pPriv)

    /// Sets target driver to OpenCL for LWCA GAPI, must be called before LWPW_LWDA_LoadDriver.
    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_RedirectToOpenCL(LWPW_LWDA_RedirectToOpenCL_Params* pParams);

    typedef struct LWPW_LWDA_Profiler_BeginSession_PrivParams
    {
        /// [in]
        size_t structSize;
        /// [in] enable/disable SMPC Streamout collection
        LWPA_Bool useSmpcStreamout;
        /// [in] session with HWPM ctxsw enabled/disabled
        LWPA_Bool isHwpmCtxswEnabled;
    } LWPW_LWDA_Profiler_BeginSession_PrivParams;
#define LWPW_LWDA_Profiler_BeginSession_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_Profiler_BeginSession_PrivParams, isHwpmCtxswEnabled)

    typedef struct LWPW_LWDA_Profiler_PerLaunchProfiling_SetLaunchId_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in]
        uint32_t launchId;
    } LWPW_LWDA_Profiler_PerLaunchProfiling_SetLaunchId_Params;
#define LWPW_LWDA_Profiler_PerLaunchProfiling_SetLaunchId_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_Profiler_PerLaunchProfiling_SetLaunchId_Params, launchId)

    /// This API may be used to set `launchId` for per launch profiling in the current session. `launchId` is used to
    /// determine range name for profiled kernel launches. `launchId` is consistent for launches across replay passes.
    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_Profiler_PerLaunchProfiling_SetLaunchId(LWPW_LWDA_Profiler_PerLaunchProfiling_SetLaunchId_Params* pParams);

    typedef struct LWPW_LWDA_Profiler_GetCounterAvailability_PrivParams
    {
        /// [in]
        size_t structSize;
        /// [in] counter availablity with HWPM ctxsw enabled/disabled
        LWPA_Bool isHwpmCtxswEnabled;
    } LWPW_LWDA_Profiler_GetCounterAvailability_PrivParams;
#define LWPW_LWDA_Profiler_GetCounterAvailability_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_Profiler_GetCounterAvailability_PrivParams, isHwpmCtxswEnabled)

/***************************************************************************//**
 *  @name   PC Sampling
 *  @{
 */

    typedef struct LWPW_LWDA_PcSampling_BeginSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] if NULL, the default stream is used
        struct LWstream_st* stream;
        /// [out] pointer to OS-allocated PerfmonBuffer (freed at EndSession)
        uint8_t* pPerfmonBuffer;
        /// [in] size of PerfmonBuffer
        size_t perfmonBufferSizeInBytes;
        /// [in] sample period of 2 ^ (5 + timespan) cycles
        size_t timespan;
        /// [in] list of requested counterIds returned by LWPW_LWDA_PcSampling_GetCounterProperties()
        uint64_t* pCounterIds;
        /// [in] count of pCounterIds
        size_t numCounterIds;
        /// [in] scratch buffer for raw PC counter data downloaded from perfmon buffer
        uint8_t* pPcCountersBuffer;
        /// [in] size of scratch buffer
        size_t pcCountersBufferSizeInBytes;
    } LWPW_LWDA_PcSampling_BeginSession_Params;
#define LWPW_LWDA_PcSampling_BeginSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_BeginSession_Params, pcCountersBufferSizeInBytes)

    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_BeginSession(LWPW_LWDA_PcSampling_BeginSession_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_EndSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] if NULL, the default stream is used
        struct LWstream_st* stream;
    } LWPW_LWDA_PcSampling_EndSession_Params;
#define LWPW_LWDA_PcSampling_EndSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_EndSession_Params, stream)

    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_EndSession(LWPW_LWDA_PcSampling_EndSession_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_GetSampleStats_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] if NULL, the default stream is used
        struct LWstream_st* stream;
        /// [out] number of unacknowledged bytes in perfmon buffer
        size_t numPerfmonBytesUnacknowledged;
        /// [out] true if PerfmonBuffer is full, possibly because buffer is not large enough or buffer was not consumed
        /// frequently enough
        LWPA_Bool perfmonBufferFull;
    } LWPW_LWDA_PcSampling_GetSampleStats_Params;
#define LWPW_LWDA_PcSampling_GetSampleStats_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_GetSampleStats_Params, perfmonBufferFull)

    /// This function should be called on one thread at a time, and not be conlwrrent with BeginSession/EndSession
    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_GetSampleStats(LWPW_LWDA_PcSampling_GetSampleStats_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_GatherData_V2_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] if NULL, the default stream is used
        struct LWstream_st* stream;
        /// [in] number of bytes available in perfmon buffer to decode.
        size_t numPerfmonBytesUnacknowledged;
        /// [in] Event stream's varDataBufferSizeInBytes must be at least: AlignUp(sizeof(LWPW_PcSampling_DecodeStats) +
        /// maxNumPcs * GetPcCounterValuesStride(), LWPW_EVENT_VARDATA_GRANULARITY) + 1
        LWPW_EventStream* pEventStream;
        /// [in] Type by which counter data is grouped and aclwmulated
        LWPW_PcSampling_CounterDataGroup counterDataGroup;
        /// [out] number of bytes decoded
        size_t numPerfmonBytesDecoded;
        /// [out] The event stream is full, data in hEventStream needs to be consumed before next GatherData call
        LWPA_Bool eventStreamFull;
        /// [out]number of bytes dropped, possibly due to back pressure when sampler frequency is too high
        uint32_t droppedBytes;
        /// [out]number of recovery attempts
        uint32_t resyncCount;
        /// [out]one or more samplers have dropped too many bytes and are unrecoverable
        uint8_t overflow;
        /// [out]scratch buffer is full
        uint8_t scratchBufferFull;
    } LWPW_LWDA_PcSampling_GatherData_V2_Params;
#define LWPW_LWDA_PcSampling_GatherData_V2_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_GatherData_V2_Params, scratchBufferFull)

    /// This function should be called on one thread at a time, and not be conlwrrent with BeginSession/EndSession.
    /// Unlike LWPW_LWDA_PcSampling_GatherData, perfmon buffer needs to be acknowledged by
    /// LWPW_LWDA_PcSampling_AcknowledgePerfmonBuffer.
    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_GatherData_V2(LWPW_LWDA_PcSampling_GatherData_V2_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_AcknowledgePerfmonBuffer_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] if NULL, the default stream is used
        struct LWstream_st* stream;
        /// [in] number of bytes in perfmon buffer to acknowledge.
        size_t numPerfmonBytesUnacknowledged;
    } LWPW_LWDA_PcSampling_AcknowledgePerfmonBuffer_Params;
#define LWPW_LWDA_PcSampling_AcknowledgePerfmonBuffer_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_AcknowledgePerfmonBuffer_Params, numPerfmonBytesUnacknowledged)

    /// This function should be called on one thread at a time, and not be conlwrrent with BeginSession/EndSession
    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_AcknowledgePerfmonBuffer(LWPW_LWDA_PcSampling_AcknowledgePerfmonBuffer_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_StartMeasuring_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] if NULL, the default stream is used
        struct LWstream_st* stream;
    } LWPW_LWDA_PcSampling_StartMeasuring_Params;
#define LWPW_LWDA_PcSampling_StartMeasuring_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_StartMeasuring_Params, stream)

    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_StartMeasuring(LWPW_LWDA_PcSampling_StartMeasuring_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_StopMeasuring_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] if NULL, the current LWcontext is used
        struct LWctx_st* ctx;
        /// [in] if NULL, the default stream is used
        struct LWstream_st* stream;
    } LWPW_LWDA_PcSampling_StopMeasuring_Params;
#define LWPW_LWDA_PcSampling_StopMeasuring_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_StopMeasuring_Params, stream)

    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_StopMeasuring(LWPW_LWDA_PcSampling_StopMeasuring_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_GetMinimumScratchBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of requested counters
        size_t numCounterIds;
        /// [out]
        size_t scratchBufferSizeInBytes;
    } LWPW_LWDA_PcSampling_GetMinimumScratchBufferSize_Params;
#define LWPW_LWDA_PcSampling_GetMinimumScratchBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_GetMinimumScratchBufferSize_Params, scratchBufferSizeInBytes)

    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_GetMinimumScratchBufferSize(LWPW_LWDA_PcSampling_GetMinimumScratchBufferSize_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_GetMinimumVarDataBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of requested counters
        size_t numCounterIds;
        /// [out]
        size_t varDataBufferSizeInBytes;
    } LWPW_LWDA_PcSampling_GetMinimumVarDataBufferSize_Params;
#define LWPW_LWDA_PcSampling_GetMinimumVarDataBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_GetMinimumVarDataBufferSize_Params, varDataBufferSizeInBytes)

    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_GetMinimumVarDataBufferSize(LWPW_LWDA_PcSampling_GetMinimumVarDataBufferSize_Params* pParams);

    typedef struct LWPW_LWDA_PcSampling_IsGpuSupported_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWdevice lwDevice;
        /// [out]
        LWPA_Bool isSupported;
        /// [out]
        LWPW_GpuArchitectureSupportLevel gpuArchitectureSupportLevel;
        /// [out]
        LWPW_SliSupportLevel sliSupportLevel;
        /// [out]
        LWPW_VGpuSupportLevel vGpuSupportLevel;
        /// [out]
        LWPW_ConfidentialComputeSupportLevel confidentialComputeSupportLevel;
    } LWPW_LWDA_PcSampling_IsGpuSupported_Params;
#define LWPW_LWDA_PcSampling_IsGpuSupported_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_LWDA_PcSampling_IsGpuSupported_Params, confidentialComputeSupportLevel)

    /// LWPW_LWDA_LoadDriver must be called prior to this API
    LWPW_LOCAL
    LWPA_Status LWPW_LWDA_PcSampling_IsGpuSupported(LWPW_LWDA_PcSampling_IsGpuSupported_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 


#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_LWDA_TARGET_PRIV_H
