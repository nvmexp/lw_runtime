#ifndef LWPERF_TARGET_PRIV_H
#define LWPERF_TARGET_PRIV_H

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
 *  @file   lwperf_target_priv.h
 */

    typedef struct LWPW_EnableMemoryStats_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPA_Bool enable;
    } LWPW_EnableMemoryStats_Params;
#define LWPW_EnableMemoryStats_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EnableMemoryStats_Params, enable)

    LWPW_LOCAL
    LWPA_Status LWPW_EnableMemoryStats(LWPW_EnableMemoryStats_Params* pParams);

    typedef struct LWPW_GetMemoryStats_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [out]
        size_t bssSize;
        /// [out] Allocated memory size.
        size_t allocated;
        /// [out] Reserved memory size.
        size_t arena;
        /// [out] Maximum allocated memory size.
        size_t maxAllocated;
        /// [out] Maximum reserved memory size.
        size_t maxArena;
    } LWPW_GetMemoryStats_Params;
#define LWPW_GetMemoryStats_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_GetMemoryStats_Params, maxArena)

    LWPW_LOCAL
    LWPA_Status LWPW_GetMemoryStats(LWPW_GetMemoryStats_Params* pParams);

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


    typedef struct LWPW_EventRecordHeader
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
    } LWPW_EventRecordHeader;
#define LWPW_EventRecordHeader_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventRecordHeader, varDataInfo)

    typedef struct LWPW_EventRecord
    {
        LWPW_EventRecordHeader header;
        union
        {
            uint64_t u64[3];
            uint8_t u8[24];
        } inlinePayload;
    } LWPW_EventRecord;
#define LWPW_EventRecord_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventRecord, inlinePayload)

    typedef enum LWPW_EventStream_BufferingMode
    {
        LWPW_EVENTSTREAM_BUFFERING_MODE_DISABLED,
        LWPW_EVENTSTREAM_BUFFERING_MODE_KEEP_OLDEST,
        LWPW_EVENTSTREAM_BUFFERING_MODE_KEEP_NEWEST,
        LWPW_EVENTSTREAM_BUFFERING_MODE__COUNT
    } LWPW_EventStream_BufferingMode;

    typedef struct LWPW_EventStream LWPW_EventStream;

    typedef struct LWPW_EventStream_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] must be at least numRecords * sizeof(LWPW_Event_Record) in size.
        uint8_t* pRecordsBuffer;
        /// [in] size of pRecords in record count, must be at least (3 + max number of events).
        uint32_t numRecords;
        /// [in] size of the records in bytes
        uint32_t recordStrideInBytes;
        /// [in] must be at least LWPW_EVENT_VARDATA_GRANULARITY size in bytes
        uint8_t* pVarData;
        /// [in] size of pVarData in bytes
        uint32_t varDataBufferSizeInBytes;
        /// [in]
        LWPW_EventStream_BufferingMode bufferingMode;
        /// [out]
        LWPW_EventStream* pEventStream;
    } LWPW_EventStream_Create_Params;
#define LWPW_EventStream_Create_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventStream_Create_Params, pEventStream)

    /// Creates an event stream.
    LWPW_LOCAL
    LWPA_Status LWPW_EventStream_Create(LWPW_EventStream_Create_Params* pParams);

    typedef struct LWPW_EventStream_Destroy_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_EventStream* pEventStream;
    } LWPW_EventStream_Destroy_Params;
#define LWPW_EventStream_Destroy_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventStream_Destroy_Params, pEventStream)

    LWPW_LOCAL
    LWPA_Status LWPW_EventStream_Destroy(LWPW_EventStream_Destroy_Params* pParams);

    typedef struct LWPW_EventStream_SetBufferingMode_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_EventStream* pEventStream;
        /// [in]
        LWPW_EventStream_BufferingMode bufferingMode;
    } LWPW_EventStream_SetBufferingMode_Params;
#define LWPW_EventStream_SetBufferingMode_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventStream_SetBufferingMode_Params, bufferingMode)

    /// Sets the producer mode of an event stream.
    LWPW_LOCAL
    LWPA_Status LWPW_EventStream_SetBufferingMode(LWPW_EventStream_SetBufferingMode_Params* pParams);

    typedef struct LWPW_EventStream_QueryRecords_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_EventStream* pEventStream;
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
    } LWPW_EventStream_QueryRecords_Params;
#define LWPW_EventStream_QueryRecords_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventStream_QueryRecords_Params, varDataDropCount)

    LWPW_LOCAL
    LWPA_Status LWPW_EventStream_QueryRecords(LWPW_EventStream_QueryRecords_Params* pParams);

    typedef struct LWPW_EventStream_AcknowledgeRecords_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPW_EventStream* pEventStream;
        /// [in] 0 <= recordsGet <  numRecords
        uint32_t recordsGet;
        /// [in] 0 <  varDataGet <= varDataBufferSizeInBytes
        uint32_t varDataGet;
    } LWPW_EventStream_AcknowledgeRecords_Params;
#define LWPW_EventStream_AcknowledgeRecords_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventStream_AcknowledgeRecords_Params, varDataGet)

    /// Marks records available for discard. Only applicable for KeepOldest producer mode.
    LWPW_LOCAL
    LWPA_Status LWPW_EventStream_AcknowledgeRecords(LWPW_EventStream_AcknowledgeRecords_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 
/***************************************************************************//**
 *  @name   Pc sampling types
 *  @{
 */

    typedef struct LWPW_EventRecord_PcSampling
    {
        /// Generic event stream header
        LWPW_EventRecord base;
        /// number of LWPW_PcSampling_PcCounterValues in this record
        uint32_t numPcs;
        /// number of counterValues in each LWPW_PcSampling_PcCounterValues
        uint32_t numCounterValuesPerPc;
    } LWPW_EventRecord_PcSampling;
#define LWPW_EventRecord_PcSampling_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_EventRecord_PcSampling, numCounterValuesPerPc)

    typedef struct LWPW_PcSampling_PcCounterValues
    {
        uint64_t pc;
        /// 2 elements to guarantee 8 bytes aligned
        uint32_t counterValues[2];
    } LWPW_PcSampling_PcCounterValues;
#define LWPW_PcSampling_PcCounterValues_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_PcCounterValues, counterValues[2])

    typedef enum LWPW_PcSampling_CounterDataGroup
    {
        LWPW_PC_SAMPLING_COUNTER_DATA_GROUP_PC,
        LWPW_PC_SAMPLING_COUNTER_DATA_GROUP_WARP_ID
    } LWPW_PcSampling_CounterDataGroup;

    typedef struct LWPW_PcSampling_GetNumAvailablePcCounterValuesParams
    {
        /// [in]
        size_t structSize;
        /// [out] number of available PcCounterValues buffers; = capacity() - size()
        size_t numAvailablePcs;
    } LWPW_PcSampling_GetNumAvailablePcCounterValuesParams;
#define LWPW_PcSampling_GetNumAvailablePcCounterValuesParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_GetNumAvailablePcCounterValuesParams, numAvailablePcs)

    typedef void (*LWPW_PcSampling_GetNumAvailablePcCounterValuesFn)(void* pUserData, struct LWPW_PcSampling_GetNumAvailablePcCounterValuesParams* pParams);

    typedef struct LWPW_PcSampling_GetPcCounterValuesParams
    {
        /// [in]
        size_t structSize;
        /// [in]
        uint64_t pc;
        /// [out]
        uint32_t* pPcCounterValues;
    } LWPW_PcSampling_GetPcCounterValuesParams;
#define LWPW_PcSampling_GetPcCounterValuesParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_PcSampling_GetPcCounterValuesParams, pPcCounterValues)

    typedef void (*LWPW_PcSampling_GetPcCounterValuesFn)(void* pUserData, struct LWPW_PcSampling_GetPcCounterValuesParams* pParams);

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
