#ifndef LWPERF_LWN_H
#define LWPERF_LWN_H

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
#include "lwperf_target.h"

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
 *  @file   lwperf_lwn.h
 */

/***************************************************************************//**
 *  @name   External Types
 *  @{
 */


    struct LWNqueue;
    struct LWNmemoryPool;


/**
 *  @}
 ******************************************************************************/
 

#define LWPA_LWN_MIN_NUM_TRACE_BUFFERS                       2
#define LWPA_LWN_TRACE_BUFFER_PAD_SIZE              0x00010000
#define LWPA_LWN_TRACE_RECORD_SIZE                  0x00000020
#define LWPA_LWN_PERFMON_RECORD_SIZE                0x00000020
#define LWPA_LWN_COMPUTE_BUFFER_PAD_SIZE            0x00000020
#define LWPA_LWN_COMPUTE_RECORD_SIZE                0x00000010

struct LWNqueue;
struct LWNcommandBuffer;


    typedef struct LWPA_LWNC_DecodeCountersOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// in : Config created by e.g.    lwperf2 configure   or   LWPA_RawMetricsConfig_GetConfigImage().  Must be
        /// align(8).
        const uint8_t* pConfig;
        size_t configSize;
        /// in : CounterDataImage, initialized by LWPA_LWNC_InitializeCounterDataImage()
        uint8_t* pCounterDataImage;
        size_t counterDataImageSize;
        /// in :
        uint8_t* pCounterDataScratchBuffer;
        /// in : as returned by LWPA_LWNC_CallwlateCounterDataScratchBufferSize()
        size_t counterDataScratchBufferSize;
    } LWPA_LWNC_DecodeCountersOptions;
#define LWPA_LWNC_DECODE_COUNTERS_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_DecodeCountersOptions, counterDataScratchBufferSize)

    typedef struct LWPA_LWNC_DecodeCountersStats
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out: number of bytes consumed from perfmon buffer
        size_t numPerfmonBytesConsumed;
        /// out: number of bytes not written to trace buffer due to buffer full
        size_t numTraceBytesDropped;
        /// out: number of bytes not written to compute buffer due to buffer full
        size_t numComputeBytesDropped;
        /// out: true if the current call to DecodeCounters was able to decode a pass (Note: DecodeCounters() can return
        /// SUCCESS when no passes are decoded)
        LWPA_Bool onePassDecoded;
        /// out: true if all passes for iteration have been decoded and ready to be evaluated
        LWPA_Bool allPassesDecoded;
    } LWPA_LWNC_DecodeCountersStats;
#define LWPA_LWNC_DECODE_COUNTERS_STATS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_DecodeCountersStats, allPassesDecoded)

    typedef struct LWPA_LWNC_QueueDebugStats
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out: the traceBufferIndex that will be targeted by the next BeginPass()
        size_t traceBufferIndexWrite;
        /// out: the traceBufferIndex that will be consumed by the next DecodeCounters()
        size_t traceBufferIndexRead;
        /// out: number of unread trace buffers (unread or in-flight profiler passes)
        size_t numTraceBuffersUnread;
        /// out: number of bytes read back by DecodeCounters; will be returned to hardware FIFO in BeginPass
        size_t numPerfmonBytesUnacknowledged;
    } LWPA_LWNC_QueueDebugStats;
#define LWPA_LWNC_QUEUE_DEBUG_STATS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_QueueDebugStats, numPerfmonBytesUnacknowledged)

    typedef struct LWPA_LWNC_TraceBufferDebugStats
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out: boolean; zero if pipelined counters collected, 1 if isolated counters collected
        uint32_t isIsolatedPass;
        /// out: passIndex in the configuration
        uint32_t passIndex;
        /// out: targetNestingLevel collected
        uint32_t targetNestingLevel;
        /// out: the hardware "PUT" offset at the end of the pass
        uint32_t perfmonBufferPutOffset;
    } LWPA_LWNC_TraceBufferDebugStats;
#define LWPA_LWNC_TRACE_BUFFER_DEBUG_STATS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_TraceBufferDebugStats, perfmonBufferPutOffset)

    typedef struct LWPA_LWNC_SessionOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        LWPA_ActivityKind activityKind;
        /// in : Config created by e.g.    lwperf2 configure   or   LWPA_RawMetricsConfig_GetConfigImage().  Must be
        /// align(8).
        const uint8_t* pConfig;
        size_t configSize;
        /// in : number of TraceBuffers; must be > 0
        size_t numTraceBuffers;
        /// in : size of TraceBuffer for one pass
        size_t traceBufferSize;
        /// in : traceArenaSize = numTraceBuffers * traceBufferSize;
        uint8_t* pTraceArena;
        /// in : LWNbufferAddress of the TraceArena
        uint64_t traceArenaGpuAddress;
        struct LWNmemoryPool* pTraceArenaMemoryPool;
        /// in : size of ComputeBuffer for one pass
        size_t computeBufferSize;
        /// in : computeArenaSize = numTraceBuffers * computeBufferSize;
        uint8_t* pComputeArena;
        /// in : LWNbufferAddress of the ComputeBuffer
        uint64_t computeArenaGpuAddress;
        struct LWNmemoryPool* pComputeArenaMemoryPool;
        /// in : size of PerfmonBuffer
        size_t perfmonBufferSize;
        /// (NX) in: pointer to page-aligned PerfmonBuffer
        /// (Windows) out: pointer to OS-allocated PerfmonBuffer (freed at EndSession)
        uint8_t* pPerfmonBuffer;
        /// in : if true, EndPass calls lwnQueueFinish automatically
        LWPA_Bool finishOnEndPass;
        /// in : the lowest nesting level to be profiled; must be >= 1
        uint16_t minNestingLevel;
        /// in : the number of nesting levels to profile; must be >= 1
        uint16_t numNestingLevels;
        /// in : if false, lwnCommandBufferPushDebugGroup* & lwnCommandBufferPopDebugGroup* denote profiling boundaries;
        /// if true, they are ignored
        LWPA_Bool disableDebugGroups;
        /// in : if false, LWPA_LWNC_PushRange* and LWPA_LWNC_PopRange denote profiling boundaries; if true, they are
        /// ignored
        LWPA_Bool disableRangeGroups;
    } LWPA_LWNC_SessionOptions;
#define LWPA_LWNC_SESSION_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_SessionOptions, disableRangeGroups)

    typedef struct LWPA_LWNC_CounterDataImageOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// The CounterDataPrefix generated from e.g.    lwperf2 initdata   or
        /// LWPA_CounterDataBuilder_GetCounterDataPrefix().  Must be align(8).
        const uint8_t* pCounterDataPrefix;
        size_t counterDataPrefixSize;
        /// max number of ranges that can be profiled
        uint32_t maxNumRanges;
        /// max number of RangeTree nodes; must be >= maxNumRanges
        uint32_t maxNumRangeTreeNodes;
        /// max string length of each RangeName, including the trailing NUL character
        uint32_t maxRangeNameLength;
    } LWPA_LWNC_CounterDataImageOptions;
#define LWPA_LWNC_COUNTER_DATA_IMAGE_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_CounterDataImageOptions, maxRangeNameLength)

    typedef struct LWPA_LWNC_UnpackRawMetricsOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// in
        const uint8_t* pCounterDataImage;
        /// in
        size_t rangeIndex;
        /// in : if true, query isolated metric values
        LWPA_Bool isolated;
        /// in
        size_t numRawMetrics;
        /// in : sorted array of rawMetricId
        const uint64_t* pRawMetricIds;
        /// out
        double* pRawMetricValues;
        /// out
        uint16_t* pHwUnitCounts;
    } LWPA_LWNC_UnpackRawMetricsOptions;
#define LWPA_LWNC_UNPACK_RAW_METRICS_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_UnpackRawMetricsOptions, pHwUnitCounts)

    LWPA_Status LWPA_LWNC_LoadDriver(void);

    LWPA_Status LWPA_LWNC_BeginSession(
        struct LWNqueue* queue,
        LWPA_LWNC_SessionOptions* pSessionOptions);

    LWPA_Status LWPA_LWNC_EndSession(struct LWNqueue* queue);

    LWPA_Status LWPA_LWNC_BeginPass(struct LWNqueue* queue);

    LWPA_Status LWPA_LWNC_EndPass(
        struct LWNqueue* queue,
        LWPA_Bool* pAllPassesSubmitted);

    LWPA_Status LWPA_LWNC_DecodeCounters(
        const LWPA_LWNC_DecodeCountersOptions* pDecodeOptions,
        LWPA_LWNC_DecodeCountersStats* pDecodeStats);

    LWPA_Status LWPA_LWNC_CallwlateCounterDataImageSize(
        const LWPA_LWNC_CounterDataImageOptions* pCounterDataImageOptions,
        size_t* pCounterDataImageSize);

    LWPA_Status LWPA_LWNC_InitializeCounterDataImage(
        const LWPA_LWNC_CounterDataImageOptions* pCounterDataImageOptions,
        size_t counterDataImageSize,
        uint8_t* pCounterDataImage);

    LWPA_Status LWPA_LWNC_CallwlateCounterDataScratchBufferSize(
        const uint8_t* pCounterDataImage,
        size_t* pCounterDataScratchBufferSize);

    LWPA_Status LWPA_LWNC_InitializeCounterDataScratchBuffer(
        const uint8_t* pCounterDataImage,
        size_t counterDataScratchBufferSize,
        uint8_t* pCounterDataScratchBuffer);

    LWPA_Status LWPA_LWNC_CounterData_UnpackRawMetrics(const LWPA_LWNC_UnpackRawMetricsOptions* pUnpackOptions);

    /// Total num passes = *pNumPipelinedPasses + *pNumIsolatedPasses * numNestingLevels
    LWPA_Status LWPA_LWNC_Config_GetNumPasses(
        const uint8_t* pConfig,
        size_t* pNumPipelinedPasses,
        size_t* pNumIsolatedPasses);

    /// Fills in pDebugStats with the current Queue state.
    LWPA_Status LWPA_LWNC_QueueGetDebugStats(
        struct LWNqueue* queue,
        LWPA_LWNC_QueueDebugStats* pDebugStats);

    /// Fills in pDebugStats for the requested traceBufferIndex.
    LWPA_Status LWPA_LWNC_GetTraceBufferDebugStats(
        size_t traceBufferIndex,
        LWPA_LWNC_TraceBufferDebugStats* pDebugStats);

    /// This function controls whether GPU registers are written from the CPU or GPU in BeginPass and EndPass.
    ///  -   GPU writes are asynchronously performed as GPU commands.
    ///  -   CPU writes cause BeginPass and EndPass to first call lwnQueueFinish().
    /// By default, GPU writes are used (cpuRegisterAccesses = false).
    /// To take effect, this function must be called before LWPA_LWNC_BeginSession().
    LWPA_Status LWPA_LWNC_EnablePerfmonRegisterAccessesFromCpu(LWPA_Bool cpuRegisterAccesses);

    /// Equivalent functionality to lwnCommandBufferPushDebugGroupStatic, but controlled by
    /// LWPA_LWNC_SessionOptions::disableRangeGroups.
    LWPA_Status LWPA_LWNC_CommandBufferPushRangeStatic(
        struct LWNcommandBuffer* cmdBuf,
        uint32_t domainId,
        const char* description);

    /// Equivalent functionality to lwnCommandBufferPushDebugGroupDynamic, but controlled by
    /// LWPA_LWNC_SessionOptions::disableRangeGroups.
    LWPA_Status LWPA_LWNC_CommandBufferPushRangeDynamic(
        struct LWNcommandBuffer* cmdBuf,
        uint32_t domainId,
        const char* description);

    /// Equivalent functionality to lwnCommandBufferPopDebugGroupId, but controlled by
    /// LWPA_LWNC_SessionOptions::disableRangeGroups.
    LWPA_Status LWPA_LWNC_CommandBufferPopRange(
        struct LWNcommandBuffer* cmdBuf,
        uint32_t domainId);

    /// Must be called on an LWNqueue before any Range command is submitted to that queue.
    /// Submitting a Range command to a queue before this function is called will result in undefined behavior.
    /// Range commands are generated by LWPA_LWNC_CommandBuffer*Range*().
    /// This function may not be called between LWPA_LWNC_BeginPass and LWPA_LWNC_EndPass.
    LWPA_Status LWPA_LWNC_QueueInitializeRangeCommands(struct LWNqueue* queue);

    /// Equivalent to LWPA_LWNC_CommandBufferPushRangeStatic, but on a Queue.
    LWPA_Status LWPA_LWNC_QueuePushRangeStatic(
        struct LWNqueue* queue,
        uint32_t domainId,
        const char* description);

    /// Equivalent to LWPA_LWNC_CommandBufferPushRangeDynamic, but on a Queue.
    LWPA_Status LWPA_LWNC_QueuePushRangeDynamic(
        struct LWNqueue* queue,
        uint32_t domainId,
        const char* description);

    /// Equivalent to LWPA_LWNC_CommandBufferPopRange, but on a Queue.
    LWPA_Status LWPA_LWNC_QueuePopRange(
        struct LWNqueue* queue,
        uint32_t domainId);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_LWN_H
