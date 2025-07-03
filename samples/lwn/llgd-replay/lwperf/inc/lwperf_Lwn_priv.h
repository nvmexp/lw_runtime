#ifndef LWPERF_LWN_PRIV_H
#define LWPERF_LWN_PRIV_H

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
 *  @file   lwperf_lwn_priv.h
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
 
    typedef struct LWPA_LWNC_PcSampling_SampleStats
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out: Number of unacknowledged bytes in perfmon buffer.
        size_t numPerfmonBytesUnacknowledged;
        /// out: True if PerfmonBuffer is full, possibly because buffer is not large enough or buffer was not consumed
        /// frequently enough.
        LWPA_Bool perfmonBufferFull;
    } LWPA_LWNC_PcSampling_SampleStats;
#define LWPA_LWNC_PC_SAMPLING_SAMPLE_STATS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_PcSampling_SampleStats, perfmonBufferFull)

    typedef struct LWPA_LWNC_PcSampling_DecodeStats
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out: Number of perfmon bytes decoded.
        size_t numPerfmonBytesDecoded;
        /// out: Number of bytes dropped, possibly due to back pressure when sampler frequency is too high.
        size_t droppedBytes;
        /// out: One or more samplers have dropped too many bytes and are unrecoverable.
        LWPA_Bool overflow;
        /// out: ScratchBuffer is full.
        LWPA_Bool scratchBufferFull;
    } LWPA_LWNC_PcSampling_DecodeStats;
#define LWPA_LWNC_PcSampling_DecodeStats_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_PcSampling_DecodeStats, scratchBufferFull)

    typedef void* LWPA_LWNC_ShaderFunction;

    typedef struct LWPA_LWNC_PcSampleOffsetData
    {
        /// out: PC address
        uint64_t pc;
        /// out: Index into \ref LWPA_LWNC_PcSampleOffsetData ["pCounterValues"] where first of N user requested counter
        /// values begins.
        uint32_t counterValueIndex;
        uint32_t rsvd000c;
    } LWPA_LWNC_PcSampleOffsetData;
#define LWPA_LWNC_PcSampleOffsetData_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_PcSampleOffsetData, rsvd000c)

    typedef struct LWPA_LWNC_PcSampling_ShaderData
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out
        LWPA_LWNC_PcSampling_DecodeStats decodeStats;
        /// out: DTA = DTA_ShaderInstance; LWCA = LWfunction; LWN = NULL
        LWPA_LWNC_ShaderFunction functionId;
        /// out: Number of elements in \a pPcSampleOffsetData.
        size_t numPcs;
        /// out: An array of all sampled PCs. Each element has an index into \a pCounterValues.
        const LWPA_LWNC_PcSampleOffsetData* pPcSampleOffsetData;
        /// out: Number of elements in \a pCounterValues array.
        size_t numCounterValues;
        /// out: Array of counter values, indexed by \ref LWPA_LWNC_PcSampleOffsetData ["counterValueIndex"].
        ///      \ref LWPA_LWNC_PcSampleOffsetData ["counterValueIndex"] points to N conselwtive counter values, where N
        ///      is the number of user requested counters.
        const uint32_t* pCounterValues;
    } LWPA_LWNC_PcSampling_ShaderData;
#define LWPA_LWNC_PC_SAMPLING_SHADER_DATA_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_PcSampling_ShaderData, pCounterValues)

    /// Gets the number of PC sample counters for a given Activity.
    LWPA_Status LWPA_LWNC_PcSampling_GetNumCounters(
        const char* pChipName,
        size_t* pNumCounters);

    typedef struct LWPA_LWNC_PcSampling_CounterProperties
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// out: Counter name.
        const char* pCounterName;
        /// out: Unique counter identifier.
        uint64_t counterId;
        /// out: Counter description.
        const char* pCounterDesc;
        LWPA_Bool hardwareReason;
    } LWPA_LWNC_PcSampling_CounterProperties;
#define LWPA_LWNC_PC_SAMPLING_COUNTER_PROPERTIES_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_PcSampling_CounterProperties, hardwareReason)

    /// Returns the name of given PC sampling counter.
    /// \param pChipName [in]
    /// \param counterIndex [in] 0 <= counterIndex < \ref LWPA_PcSampling_GetNumCounters()
    /// \param pCounterProperties [out]
    LWPA_Status LWPA_LWNC_PcSampling_GetCounterProperties(
        const char* pChipName,
        size_t counterIndex,
        LWPA_LWNC_PcSampling_CounterProperties* pCounterProperties);

    typedef struct LWPA_LWNC_PcSampling_SessionOptions
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// (NX)  in : pointer to page-aligned PerfmonBuffer
        /// (Win) out: pointer to OS-allocated PerfmonBuffer (free'd at EndSession)
        uint8_t* pPerfmonBuffer;
        /// in : Size of PerfmonBuffer
        size_t perfmonBufferSize;
        /// in : Sample period of 2 ^ (5 + timespan) cycles.
        size_t timespan;
        /// in : Max number of PC to decode, minimally 6.
        size_t maxNumPcs;
        /// in : List of requested counterIds returned by \ref LWPA_PcSampling_GetCounterProperties().
        uint64_t* pCounterIds;
        /// in : Count of \ref pCounterIds.
        size_t numCounterIds;
        /// in : Scratch buffer for raw PC counter data downloaded from perfmon buffer.
        uint8_t* pPcCountersBuffer;
        /// in : Size of scratch buffer. Should be at least \a maxNumPcs * AlignUp(\a numCounterIds * 4B + 32B_PADDING,
        /// 8).
        size_t pcCountersBufferSizeInBytes;
        /// in : Scratch buffer for storing LWPA_LWNC_PcSampleOffsetData for each PC.
        uint8_t* pPcSampleOffsetDataBuffer;
        /// in : Size of scratch buffer. Should be at least \a maxNumPcs * sizeof(\ref LWPA_LWNC_PcSampleOffsetData).
        size_t pcSampleOffsetDataBufferSizeInBytes;
        /// in : Scratch buffer for storing counter values for each PC.
        uint8_t* pCounterValuesBuffer;
        /// in : Size of scratch buffer. Should be at least \a maxNumPcs * \a numCounterIds * 4B.
        size_t counterValuesBufferSizeInBytes;
    } LWPA_LWNC_PcSampling_SessionOptions;
#define LWPA_LWNC_PC_SAMPLING_SESSION_OPTIONS_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPA_LWNC_PcSampling_SessionOptions, counterValuesBufferSizeInBytes)

    LWPA_Status LWPA_LWNC_PcSampling_BeginSession(
        struct LWNqueue* queue,
        LWPA_LWNC_PcSampling_SessionOptions* pSessionOptions);

    LWPA_Status LWPA_LWNC_PcSampling_EndSession(struct LWNqueue* queue);

    LWPA_Status LWPA_LWNC_PcSampling_CommandBufferStartSampling(struct LWNcommandBuffer* cmdBuf);

    LWPA_Status LWPA_LWNC_PcSampling_CommandBufferStopSampling(struct LWNcommandBuffer* cmdBuf);

    LWPA_Status LWPA_LWNC_PcSampling_QueueStartSampling(struct LWNqueue* queue);

    LWPA_Status LWPA_LWNC_PcSampling_QueueStopSampling(struct LWNqueue* queue);

    /// Get the number of perfmon bytes that have been been decoded yet, and if the perfmon buffer is full.
    LWPA_Status LWPA_LWNC_PcSampling_QueueGetSampleStats(
        struct LWNqueue* queue,
        LWPA_LWNC_PcSampling_SampleStats* pSampleStats);

    /// Decode the perfmon buffer into \ref LWPA_LWNC_PcSampling_ShaderData, reading at most \ref bytesToDecode perfmon
    /// buffer bytes or at most \ref LWPA_LWNC_PcSampling_SessionOptions ["maxPcSamples"], whichever is first.
    LWPA_Status LWPA_LWNC_PcSampling_QueueDecodeSampleData(
        struct LWNqueue* queue,
        size_t bytesToDecode,
        LWPA_LWNC_PcSampling_ShaderData* pShaderData);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_LWN_PRIV_H
