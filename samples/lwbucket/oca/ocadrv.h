 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: ocadrv.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OCADRV_H
#define _OCADRV_H

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// ISR/DPC Timing Info defines
#define MAX_ISR_DPC_ENTRIES     500
#define ISR_TYPE                0
#define DPC_TYPE                1
#define DID_NOT_COMPLETE        0x7fffffff
#define MAX_ISR_DPC_REENTRANCY  20

// Vblank Info defines
#define MAX_VBLANK_ENTRIES      80

// Error Info defines
#define MAX_ERROR_ENTRIES       128

// Error Warning defines
#define MAX_WARNING_ENTRIES     128

// Paging Log defines
#define MAX_PAGING_ENTRIES      128 // Should this be sufficient?

// Buffers Info defines
#define NUM_BUFFERS_TO_STORE    8

// LW scaling method quanities
// - aligns with LW_CFGEX_GET_FLATPANEL_INFO_SCALER for backwards compatibility
// - used in Vista and up
typedef enum _LWL_SCALING
{
    LWL_SCALING_UNINITIALIZED  = 0,
    LWL_SCALING_IDENTITY       = 1,
    LWL_SCALING_CENTERED       = 2,
    LWL_SCALING_STRETCHED      = 3,
    LWL_SCALING_ASPECTRATIO    = 5,

} LWL_SCALING;

//******************************************************************************
//
// KMD OCA Record Structures
//
//******************************************************************************
// Define the KMD group record types
typedef enum _KMDCD_RECORD_TYPE
{
    KmdGlobalInfo            = 0,               // Kmd Global record
    KmdAdapterInfo           = 1,               // Kmd Adapter record
    KmdEngineIdInfo          = 2,               // IDs for each engine/node
    KmdDeviceInfo            = 3,               // Kmd Device record
    KmdContextInfo           = 4,               // Kmd Context record
    KmdChannelInfo           = 5,               // Kmd Channel record
    KmdAllocationInfo        = 6,               // Kmd Allocation record 
    KmdDmaBufferInfo         = 7,               // Kmd DMA Buffer record   
    KmdRingBufferInfo        = 8,               // Kmd Ring Buffer record
    KmdDisplayTargetInfo     = 9,               // Kmd Display Target record
    KmdHybridControlInfo     = 10,              // Kmd Hybrid Control Info
    KmdMonitorInfo           = 11,              // Kmd Monitor Info
    KmdGlobalInfo_V2         = 12,              // Kmd global data record version 2
    KmdIsrDpcTimingInfo      = 13,              // Kmd ISR/DPC Timing Info
    KmdRingBufferInfo_V2     = 14,              // Kmd OCA data ring buffers (version 2)
    KmdDisplayTargetInfo_V2  = 15,              // Kmd Display Target Info (version 2)
    KmdVblankInfo            = 16,              // Kmd Vblank Info
    KmdErrorInfo             = 17,              // Kmd Error Info
    KmdWarningInfo           = 18,              // Kmd Warning Info
    KmdPagingInfo            = 19,              // Kmd Paging Info
    KmdProcessInfo           = 20,              // Kmd Process Info
    KmdProcessInfo_V2        = 21,              // Kmd Process Info (version 2)
    KmdDeviceInfo_V2         = 22,              // Kmd Device record (version 2)

} KMDCD_RECORD_TYPE;

//******************************************************************************
//
//  class CKmdOcaFlags
//
//******************************************************************************
class CKmdOcaFlags
{
// KMD_OCA_FLAGS Type Helpers
TYPE(kmdOcaFlags)

// KMD_OCA_FLAGS Field Helpers
FIELD(IsHybrid)
FIELD(Value)

// KMD_OCA_FLAGS Members
MEMBER(IsHybrid,        ULONG,  0,  public)
MEMBER(Value,           ULONG,  0,  public)

private:
        const void*     m_pKmdOcaFlags;

public:
                        CKmdOcaFlags(const void* pKmdOcaFlags);
virtual                ~CKmdOcaFlags();

const   void*           kmdOcaFlags() const         { return m_pKmdOcaFlags; }

const   CMemberType&    type() const                { return m_kmdOcaFlagsType; }

}; // CKmdOcaFlags

//******************************************************************************
//
//  class CKmdOcaRecord
//
//******************************************************************************
class CKmdOcaRecord : public CLwcdRecord
{
// KMD_OCA_RECORD Type Helpers
TYPE(kmdOcaRecord)

// KMD_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(AdapterCount)
FIELD(SessionIdentifier)
FIELD(Flags)

// KMD_OCA_RECORD Members
MEMBER(AdapterCount,    ULONG,  0,  public)
MEMBER(Flags,           ULONG,  0,  public)

private:
const   void*           m_pKmdOcaRecord;
const   COcaData*       m_pOcaData;
        CKmdOcaFlagsPtr m_pKmdOcaFlags;
        COcaGuidPtr     m_pSessionIdentifier;

public:
                        CKmdOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CKmdOcaRecord();

const   void*           kmdOcaRecord() const        { return m_pKmdOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }
const   CKmdOcaFlags*   kmdOcaFlags() const         { return m_pKmdOcaFlags; }

        ULONG           IsHybrid() const            { return ((m_pKmdOcaFlags != NULL) ? m_pKmdOcaFlags->IsHybrid() : 0); }
        ULONG           Value() const               { return ((m_pKmdOcaFlags != NULL) ? m_pKmdOcaFlags->Value() : 0); }

const   COcaGuid*       SessionIdentifier() const   { return m_pSessionIdentifier; }

const   CMemberType&    type() const                { return m_kmdOcaRecordType; }

}; // CKmdOcaRecord

//******************************************************************************
//
//  class CHybridControlOcaFlags
//
//******************************************************************************
class CHybridControlOcaFlags
{
// HYBRID_CONTROL_OCA_FLAGS Type Helpers
TYPE(hybridControlOcaFlags)

// HYBRID_CONTROL_OCA_FLAGS Field Helpers
FIELD(HideModeSet)
FIELD(bEnableHybridMode)
FIELD(bEnableGoldOnHybrid)
FIELD(bEnableHybridPerfSLI)
FIELD(bIntelHybrid)
FIELD(bSvcStarted)
FIELD(bM2DSkipFlipUntilSecondBuffer)
FIELD(bDWMOn)
FIELD(Value)

// HYBRID_CONTROL_OCA_FLAGS Members
MEMBER(HideModeSet,                     ULONG,  0,  public)
MEMBER(bEnableHybridMode,               ULONG,  0,  public)
MEMBER(bEnableGoldOnHybrid,             ULONG,  0,  public)
MEMBER(bEnableHybridPerfSLI,            ULONG,  0,  public)
MEMBER(bIntelHybrid,                    ULONG,  0,  public)
MEMBER(bSvcStarted,                     ULONG,  0,  public)
MEMBER(bM2DSkipFlipUntilSecondBuffer,   ULONG,  0,  public)
MEMBER(bDWMOn,                          ULONG,  0,  public)
MEMBER(Value,                           ULONG,  0,  public)

private:
        const void*     m_pHybridControlOcaFlags;

public:
                        CHybridControlOcaFlags(const void* pHybridControlOcaFlags);
virtual                ~CHybridControlOcaFlags();

const   void*           hybridControlOcaFlags() const
                            { return m_pHybridControlOcaFlags; }

const   CMemberType&    type() const                { return m_hybridControlOcaFlagsType; }

}; // CHybridControlOcaFlags

//******************************************************************************
//
//  class CHybridControlOcaRecord
//
//******************************************************************************
class CHybridControlOcaRecord : public CLwcdRecord
{
// HYBRID_CONTROL_OCA_RECORD Type Helpers
TYPE(hybridControlOcaRecord)

// HYBRID_CONTROL_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(HybridState)
FIELD(ulApprovalStatus)
FIELD(ulMGpuId)
FIELD(ulDGpuId)
FIELD(ulSkipM2DRender)
FIELD(ulD2MSkipFlips)
FIELD(Flags)

// HYBRID_CONTROL_OCA_RECORD Members
MEMBER(HybridState,                     ULONG,   0,  public)
MEMBER(ulApprovalStatus,                ULONG,   0,  public)
MEMBER(ulMGpuId,                        ULONG,   0,  public)
MEMBER(ulDGpuId,                        ULONG,   0,  public)
MEMBER(ulSkipM2DRender,                 ULONG,   0,  public)
MEMBER(ulD2MSkipFlips,                  ULONG,   0,  public)
MEMBER(Flags,                           ULONG,   0,  public)

private:
const   void*           m_pHybridControlOcaRecord;
const   COcaData*       m_pOcaData;
        CHybridControlOcaFlagsPtr m_pHybridControlOcaFlags;

public:
                        CHybridControlOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CHybridControlOcaRecord();

const   void*           hybridControlOcaRecord() const
                            { return m_pHybridControlOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CHybridControlOcaFlags* hybridControlOcaFlags() const
                            { return m_pHybridControlOcaFlags; }

        ULONG           HideModeSet() const         { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags->HideModeSet() : 0); }
        ULONG           bEnableHybridMode() const   { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags-> bEnableHybridMode() : 0); }
        ULONG           bEnableGoldOnHybrid() const { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags-> bEnableGoldOnHybrid() : 0); }
        ULONG           bEnableHybridPerfSLI() const{ return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags->bEnableHybridPerfSLI() : 0); }
        ULONG           bIntelHybrid() const        { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags->bIntelHybrid() : 0); }
        ULONG           bSvcStarted() const         { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags->bSvcStarted() : 0); }
        ULONG           bM2DSkipFlipUntilSecondBuffer() const
                            { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags->bM2DSkipFlipUntilSecondBuffer() : 0); }
        ULONG           bDWMOn() const              { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags-> bDWMOn() : 0); }
        ULONG           Value() const               { return ((m_pHybridControlOcaFlags != NULL) ? m_pHybridControlOcaFlags->Value() :0); }

const   CMemberType&    type() const                { return m_hybridControlOcaRecordType; }

}; // CHybridControlOcaRecord

//******************************************************************************
//
//  class CIsrDpcOcaData
//
//******************************************************************************
class CIsrDpcOcaData
{
// ISR_DPC_OCA_DATA Type Helpers
TYPE(isrDpcOcaData)

// ISR_DPC_OCA_DATA Field Helpers
FIELD(Type)
FIELD(duration)
FIELD(timestamp)

// ISR_DPC_OCA_DATA Members
MEMBER(Type,            ULONG,  0,  public)
MEMBER(duration,        ULONG,  0,  public)
MEMBER(timestamp,       ULONG,  0,  public)

private:
        const void*     m_pIsrDpcOcaData;

public:
                        CIsrDpcOcaData(const void* pIsrDpcOcaData);
virtual                ~CIsrDpcOcaData();

const   void*           isrDpcOcaData() const       { return m_pIsrDpcOcaData; }

const   CMemberType&    type() const                { return m_isrDpcOcaDataType; }

}; // CIsrDpcOcaData

//******************************************************************************
//
// class CIsrDpcOcaDatas
//
//******************************************************************************
class CIsrDpcOcaDatas
{
protected:
        ULONG           m_ulIsrDpcDataCount;

const   CIsrDpcOcaRecord* m_pIsrDpcOcaRecord;
mutable CIsrDpcOcaDataArray m_aIsrDpcOcaDatas;

public:
                        CIsrDpcOcaDatas(const CIsrDpcOcaRecord* pIsrDpcOcaRecord, ULONG ulRemaining);
virtual                ~CIsrDpcOcaDatas();

        ULONG           isrDpcDataCount() const     { return m_ulIsrDpcDataCount; }

const   void*           isrDpcOca(ULONG ulIsrDpc) const;

const   CIsrDpcOcaData* isrDpcOcaData(ULONG ulIsrDpc) const;

}; // class CIsrDpcDatas

//******************************************************************************
//
//  class CIsrDpcOcaRecord
//
//******************************************************************************
class CIsrDpcOcaRecord : public CLwcdRecord
{
// ISR_DPC_OCA_RECORD Type Helpers
TYPE(isrDpcOcaRecord)

// ISR_DPC_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(Adapter)
FIELD(frequency)
FIELD(count)
FIELD(data)

// ISR_DPC_OCA_RECORD Members
MEMBER(Adapter,         POINTER,    0,  public)
MEMBER(frequency,       ULONG64,    0,  public)
MEMBER(count,           ULONG,      0,  public)

private:
const   void*           m_pIsrDpcOcaRecord;
const   COcaData*       m_pOcaData;
        CIsrDpcOcaDatasPtr m_pIsrDpcOcaDatas;

public:
                        CIsrDpcOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CIsrDpcOcaRecord();

const   void*           isrDpcOcaRecord() const     { return m_pIsrDpcOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CIsrDpcOcaDatas* isrDpcOcaDatas() const     { return m_pIsrDpcOcaDatas; }
        ULONG           isrDpcDataCount() const     { return ((m_pIsrDpcOcaDatas != NULL) ? m_pIsrDpcOcaDatas->isrDpcDataCount() : 0); }
const   CIsrDpcOcaData* isrDpcOcaData(ULONG ulIsrDpc) const
                            { return ((m_pIsrDpcOcaDatas != NULL) ? m_pIsrDpcOcaDatas->isrDpcOcaData(ulIsrDpc) : NULL); }

const   CMemberType&    type() const                { return m_isrDpcOcaRecordType; }

}; // CIsrDpcOcaRecord

//******************************************************************************
//
//  class CAdapterOcaFlags
//
//******************************************************************************
class CAdapterOcaFlags
{
// ADAPTER_OCA_FLAGS Type Helpers
TYPE(adapterOcaFlags)

// ADAPTER_OCA_FLAGS Field Helpers
FIELD(InTDR)
FIELD(IsSLI)
FIELD(Value)

// ADAPTER_OCA_FLAGS Members
MEMBER(InTDR,           ULONG,  0,  public)
MEMBER(IsSLI,           ULONG,  0,  public)
MEMBER(Value,           ULONG,  0,  public)

private:
        const void*     m_pAdapterOcaFlags;

public:
                        CAdapterOcaFlags(const void* pAdapterOcaFlags);
virtual                ~CAdapterOcaFlags();

const   void*           adapterOcaFlags() const     { return m_pAdapterOcaFlags; }

const   CMemberType&    type() const                { return m_adapterOcaFlagsType; }

}; // CAdapterOcaFlags

//******************************************************************************
//
//  class CAdapterOcaRecord
//
//******************************************************************************
class CAdapterOcaRecord : public CLwcdRecord
{
// ADAPTER_OCA_RECORD Type Helpers
TYPE(adapterOcaRecord)

// ADAPTER_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(Adapter)
FIELD(SemaphoreMemory)
FIELD(hClient)
FIELD(hDevice)
FIELD(DeviceCount)
FIELD(ChannelCount)
FIELD(NumOfIDSRecords)
FIELD(IlwalidSLTOffsetCount)
FIELD(TDRCount)
FIELD(LateBufferCompletionCount)
FIELD(BufferSubmissionErrorCount)
FIELD(EngineExelwtionMask)
FIELD(EnginePreemptionMask)
FIELD(LastPerformanceCounterFrequency)
FIELD(ConnectedDevicesMask)
FIELD(DeviceLoadState)
FIELD(ulArchitecture)
FIELD(ulImplementation)
FIELD(dwRevision)
FIELD(subRevision)
FIELD(SBIOSPowerUpRetry)
FIELD(Flags)

// ADAPTER_OCA_RECORD Members
MEMBER(Adapter,                         POINTER,    NULL,   public)
MEMBER(SemaphoreMemory,                 POINTER,    NULL,   public)
MEMBER(hClient,                         ULONG,      0,      public)
MEMBER(hDevice,                         ULONG,      0,      public)
MEMBER(DeviceCount,                     ULONG,      0,      public)
MEMBER(ChannelCount,                    ULONG,      0,      public)
MEMBER(NumOfIDSRecords,                 ULONG,      0,      public)
MEMBER(IlwalidSLTOffsetCount,           ULONG,      0,      public)
MEMBER(TDRCount,                        ULONG,      0,      public)
MEMBER(LateBufferCompletionCount,       ULONG,      0,      public)
MEMBER(BufferSubmissionErrorCount,      ULONG,      0,      public)
MEMBER(EngineExelwtionMask,             ULONG,      0,      public)
MEMBER(EnginePreemptionMask,            ULONG,      0,      public)
MEMBER(LastPerformanceCounterFrequency, ULONG64,    0,      public)
MEMBER(ConnectedDevicesMask,            ULONG,      0,      public)
MEMBER(DeviceLoadState,                 ULONG,      0,      public)
MEMBER(ulArchitecture,                  ULONG,      0,      public)
MEMBER(ulImplementation,                ULONG,      0,      public)
MEMBER(dwRevision,                      ULONG,      0,      public)
MEMBER(subRevision,                     UCHAR,      0,      public)
MEMBER(SBIOSPowerUpRetry,               ULONG,      0,      public)
MEMBER(Flags,                           ULONG,      0,      public)

private:
const   void*           m_pAdapterOcaRecord;
const   COcaData*       m_pOcaData;

        CAdapterOcaFlagsPtr m_pAdapterOcaFlags;

mutable CDeviceOcaRecordsPtr m_pDeviceOcaRecords;
mutable CAllocationOcaRecordsPtr m_pAllocationOcaRecords;

public:
                        CAdapterOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CAdapterOcaRecord();

const   void*           adapterOcaRecord() const    { return m_pAdapterOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }
const   CAdapterOcaFlags* adapterOcaFlags() const   { return m_pAdapterOcaFlags; }

const   CDeviceOcaRecords* deviceOcaRecords() const;
const   CAllocationOcaRecords* allocationOcaRecords() const;

        ULONG           InTDR() const               { return ((m_pAdapterOcaFlags != NULL) ? m_pAdapterOcaFlags->InTDR() : 0); }
        ULONG           IsSLI() const               { return ((m_pAdapterOcaFlags != NULL) ? m_pAdapterOcaFlags->IsSLI() : 0); }
        ULONG           Value() const               { return ((m_pAdapterOcaFlags != NULL) ? m_pAdapterOcaFlags->Value() : 0); }

const   CMemberType&    type() const                { return m_adapterOcaRecordType; }

}; // CAdapterOcaRecord

//******************************************************************************
//
//  class CBufferInfoRecord
//
//******************************************************************************
class CBufferInfoRecord
{
// BUFFER_INFO_RECORD Type Helpers
TYPE(bufferInfoRecord)

// BUFFER_INFO_RECORD Field Helpers
FIELD(Context)
FIELD(SubmitId)
FIELD(BufferId)
FIELD(FenceId)
FIELD(Size)
FIELD(lwoOffset)
FIELD(IntCount)
FIELD(Type)

// BUFFER_INFO_RECORD Members
MEMBER(Context,         POINTER,    0,  public)
MEMBER(SubmitId,        ULONG,      0,  public)
MEMBER(BufferId,        ULONG,      0,  public)
MEMBER(FenceId,         ULONG,      0,  public)
MEMBER(Size,            ULONG,      0,  public)
MEMBER(lwoOffset,       ULONG64,    0,  public)
MEMBER(IntCount,        ULONG,      0,  public)
MEMBER(Type,            ULONG,      0,  public)

private:
        const void*     m_pBufferInfoRecord;

public:
                        CBufferInfoRecord(const void* pBufferInfoRecord);
virtual                ~CBufferInfoRecord();

const   void*           bufferInfoRecord() const    { return m_pBufferInfoRecord; }

const   CMemberType&    type() const                { return m_bufferInfoRecordType; }

}; // CBufferInfoRecord

//******************************************************************************
//
// class CBufferInfoRecords
//
//******************************************************************************
class CBufferInfoRecords
{
protected:
        ULONG           m_ulBufferInfoCount;

const   CEngineIdOcaRecord* m_pEngineIdOcaRecord;
mutable CBufferInfoRecordArray m_aBufferInfoRecords;

public:
                        CBufferInfoRecords(const CEngineIdOcaRecord* pEngineIdOcaRecord, ULONG ulRemaining);
virtual                ~CBufferInfoRecords();

        ULONG           bufferInfoCount() const     { return m_ulBufferInfoCount; }

const   void*           bufferInfo(ULONG ulBufferInfo) const;

const   CBufferInfoRecord* bufferInfoRecord(ULONG ulBufferInfo) const;

}; // class CBufferInfoRecords

//******************************************************************************
//
//  class CEngineIdOcaRecord
//
//******************************************************************************
class CEngineIdOcaRecord : public CLwcdRecord
{
// ENGINEID_OCA_RECORD Type Helpers
TYPE(engineIdOcaRecord)

// ENGINEID_OCA_RECORD Enum Helpers
ENUM(nodeType)

// ENGINEID_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(BufferId)
FIELD(ISRBufferId)
FIELD(DPCBufferId)
FIELD(FenceId)
FIELD(ISRFenceId)
FIELD(DPCFenceId)
FIELD(Premption)
FIELD(PreemptionId)
FIELD(ISRPreemptionId)
FIELD(SubmitId)
FIELD(LastObservedProgress)
FIELD(LastObservedStall)
FIELD(NodeOrdinal)
FIELD(NodeType)
FIELD(NodeClassType)
FIELD(EngineOrdinal)
FIELD(MaxBufferExelwtionTime)
FIELD(Buffers)

// ENGINEID_OCA_RECORD Members
MEMBER(BufferId,                ULONG,  0,  public)
MEMBER(ISRBufferId,             ULONG,  0,  public)
MEMBER(DPCBufferId,             ULONG,  0,  public)
MEMBER(FenceId,                 ULONG,  0,  public)
MEMBER(ISRFenceId,              ULONG,  0,  public)
MEMBER(DPCFenceId,              ULONG,  0,  public)
MEMBER(Premption,               ULONG,  0,  public)
MEMBER(PreemptionId,            ULONG,  0,  public)
MEMBER(ISRPreemptionId,         ULONG,  0,  public)
MEMBER(SubmitId,                ULONG,  0,  public)
MEMBER(LastObservedProgress,    ULONG,  0,  public)
MEMBER(LastObservedStall,       ULONG,  0,  public)
MEMBER(NodeOrdinal,             ULONG,  0,  public)
MEMBER(NodeType,                ULONG,  0,  public)
MEMBER(NodeClassType,           ULONG,  0,  public)
MEMBER(EngineOrdinal,           ULONG,  0,  public)
MEMBER(MaxBufferExelwtionTime,  ULONG,  0,  public)

private:
const   void*           m_pEngineIdOcaRecord;
const   COcaData*       m_pOcaData;
        CBufferInfoRecordsPtr m_pBufferInfoRecords;

public:
                        CEngineIdOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CEngineIdOcaRecord();

const   void*           engineIdOcaRecord() const   { return m_pEngineIdOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

        CString         typeString() const          { return nodeTypeEnum().valueString(NodeType()); }
        ULONG           typeWidth() const           { return nodeTypeEnum().width(); }
        ULONG           typePrefix() const          { return nodeTypeEnum().prefix(); }

const   CBufferInfoRecord* bufferInfoRecords() const;
        ULONG           bufferInfoCount() const     { return ((m_pBufferInfoRecords != NULL) ? m_pBufferInfoRecords->bufferInfoCount() : 0); }
const   CBufferInfoRecord* bufferInfoRecord(ULONG ulBufferInfo) const
                            { return ((m_pBufferInfoRecords != NULL) ? m_pBufferInfoRecords->bufferInfoRecord(ulBufferInfo) : NULL); }

const   CMemberType&    type() const                { return m_engineIdOcaRecordType; }

}; // CEngineIdOcaRecord

//******************************************************************************
//
//  class CDmaBufferOcaRecord
//
//******************************************************************************
class CDmaBufferOcaRecord : public CLwcdRecord
{
// DMA_BUFFER_OCA_RECORD Type Helpers
TYPE(dmaBufferOcaRecord)

// DMA_BUFFER_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(Context)
FIELD(SubmitId)
FIELD(BufferId)
FIELD(FenceId)
FIELD(Size)
FIELD(lwoOffset)
FIELD(IntCount)
FIELD(Type)

// DMA_BUFFER_OCA_RECORD Members
MEMBER(Context,         POINTER,    NULL,   public)
MEMBER(SubmitId,        ULONG,      0,      public)
MEMBER(BufferId,        ULONG,      0,      public)
MEMBER(FenceId,         ULONG,      0,      public)
MEMBER(Size,            ULONG,      0,      public)
MEMBER(lwoOffset,       ULONG64,    0,      public)
MEMBER(IntCount,        ULONG,      0,      public)
MEMBER(Type,            ULONG,      0,      public)

private:
const   void*           m_pDmaBufferOcaRecord;
const   COcaData*       m_pOcaData;

public:
                        CDmaBufferOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CDmaBufferOcaRecord();

const   void*           dmaBufferOcaRecord() const  { return m_pDmaBufferOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CMemberType&    type() const                { return m_dmaBufferOcaRecordType; }

}; // CDmaBufferOcaRecord

//******************************************************************************
//
//  class CGpuWatchdogEvent
//
//******************************************************************************
class CGpuWatchdogEvent
{
// GPUWATCHDOG_EVENT Type Helpers
TYPE(gpuWatchdogEvent)

// GPUWATCHDOG_EVENT Field Helpers
FIELD(AdapterOrdinal)
FIELD(NodeOrdinal)
FIELD(EngineOrdinal)
FIELD(Exelwting)
FIELD(BufferExelwtionTime)
FIELD(LwrrentTime)
FIELD(FenceId)
FIELD(PBget)
FIELD(C1get)
FIELD(Pixels)
FIELD(Status)

// GPUWATCHDOG_EVENT Members
MEMBER(AdapterOrdinal,      BYTE,       0,  public)
MEMBER(NodeOrdinal,         BYTE,       0,  public)
MEMBER(EngineOrdinal,       BYTE,       0,  public)
MEMBER(Exelwting,           BYTE,       0,  public)
MEMBER(BufferExelwtionTime, ULONG,      0,  public)
MEMBER(LwrrentTime,         ULONG64,    0,  public)
MEMBER(FenceId,             ULONG,      0,  public)
MEMBER(PBget,               ULONG,      0,  public)
MEMBER(C1get,               ULONG,      0,  public)
MEMBER(Pixels,              ULONG,      0,  public)
MEMBER(Status,              ULONG,      0,  public)

private:
        const void*     m_pGpuWatchdogEvent;

public:
                        CGpuWatchdogEvent(const void* pGpuWatchdogEvent);
virtual                ~CGpuWatchdogEvent();

const   void*           gpuWatchdogEvent() const    { return m_pGpuWatchdogEvent; }

const   CMemberType&    type() const                { return m_gpuWatchdogEventType; }

}; // CGpuWatchdogEvent

//******************************************************************************
//
// class CGpuWatchdogEvents
//
//******************************************************************************
class CGpuWatchdogEvents
{
protected:
        ULONG           m_ulGpuWatchdogEventCount;

const   CKmdRingBuffer* m_pKmdRingBuffer;
mutable CGpuWatchdogEventArray m_aGpuWatchdogEvents;

public:
                        CGpuWatchdogEvents(const CKmdRingBuffer* pKmdRingBuffer, ULONG ulRemaining);
virtual                ~CGpuWatchdogEvents();

        ULONG           gpuWatchdogEventCount() const
                            { return m_ulGpuWatchdogEventCount; }

const   void*           watchdogEvent(ULONG ulGpuWatchdogEvent) const;

const   CGpuWatchdogEvent* gpuWatchdogEvent(ULONG ulGpuWatchdogEvent) const;

}; // class CGpuWatchdogEvents

//******************************************************************************
//
//  class CKmdRingBuffer
//
//******************************************************************************
class CKmdRingBuffer
{
// CKmdRingBuffer Type Helpers
TYPE(kmdRingBuffer)

// CKmdRingBuffer Field Helpers
FIELD(ulIndex)
FIELD(ulCount)
FIELD(Elements)

// CKmdRingBuffer Members
MEMBER(ulIndex,     ULONG,       0,  public)
MEMBER(ulCount,     ULONG,       0,  public)

private:
        const void*     m_pKmdRingBuffer;
        CGpuWatchdogEventsPtr m_pGpuWatchdogEvents;

public:
                        CKmdRingBuffer(const void* pKmdRingBuffer, ULONG ulRemaining);
virtual                ~CKmdRingBuffer();

const   void*           kmdRingBuffer() const
                            { return m_pKmdRingBuffer; }

const   CGpuWatchdogEvents* gpuWatchdogEvents() const
                            { return m_pGpuWatchdogEvents; }
        ULONG           gpuWatchdogEventCount() const
                            { return ((m_pGpuWatchdogEvents != NULL) ? m_pGpuWatchdogEvents->gpuWatchdogEventCount() : 0); }
const   CGpuWatchdogEvent* gpuWatchdogEvent(ULONG ulGpuWatchdogEvent) const
                            { return ((m_pGpuWatchdogEvents != NULL) ? m_pGpuWatchdogEvents->gpuWatchdogEvent(ulGpuWatchdogEvent) : NULL); }

const   CMemberType&    type() const                { return m_kmdRingBufferType; }

}; // CKmdRingBuffer

//******************************************************************************
//
//  class CKmdRingBufferOcaRecord
//
//******************************************************************************
class CKmdRingBufferOcaRecord : public CLwcdRecord
{
// KMD_RING_BUFFER_OCA_RECORD Type Helpers
TYPE(kmdRingBufferOcaRecord)

// KMD_RING_BUFFER_OCA_RECORD Field Helpers
FIELD(Header)

private:
const   void*           m_pKmdRingBufferOcaRecord;
const   COcaData*       m_pOcaData;
        CKmdRingBufferPtr m_pKmdRingBuffer;

public:
                        CKmdRingBufferOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CKmdRingBufferOcaRecord();

const   void*           kmdRingBufferOcaRecord() const
                            { return m_pKmdRingBufferOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

        ULONG           ulIndex() const             { return m_pKmdRingBuffer->ulIndex(); }
        ULONG           ulCount() const             { return m_pKmdRingBuffer->ulCount(); }

const   CKmdRingBuffer* kmdRingBuffer() const       { return m_pKmdRingBuffer; }
        ULONG           gpuWatchdogEventCount() const
                            { return ((m_pKmdRingBuffer != NULL) ? m_pKmdRingBuffer->gpuWatchdogEventCount() : 0); }
const   CGpuWatchdogEvent* gpuWatchdogEvent(ULONG ulGpuWatchdogEvent) const
                            { return ((m_pKmdRingBuffer != NULL) ? m_pKmdRingBuffer->gpuWatchdogEvent(ulGpuWatchdogEvent) : NULL); }

const   CMemberType&    type() const                { return m_kmdRingBufferOcaRecordType; }

}; // CKmdRingBufferOcaRecord

//******************************************************************************
//
//  class CAllocationOcaResource
//
//******************************************************************************
class CAllocationOcaResource
{
// ALLOCATION_OCA_RESOURCE Type Helpers
TYPE(allocationOcaResource)

// ALLOCATION_OCA_RESOURCE Field Helpers
FIELD(Type)
FIELD(Format)
FIELD(Width)
FIELD(Height)
FIELD(Depth)
FIELD(MipMapCount)
FIELD(VidPnSourceId)

// ALLOCATION_OCA_RESOURCE Members
MEMBER(Type,            ULONG,      0,  public)
MEMBER(Format,          ULONG,      0,  public)
MEMBER(Width,           ULONG,      0,  public)
MEMBER(Height,          ULONG,      0,  public)
MEMBER(Depth,           ULONG,      0,  public)
MEMBER(MipMapCount,     ULONG,      0,  public)
MEMBER(VidPnSourceId,   ULONG,      0,  public)

private:
        const void*     m_pAllocationOcaResource;

public:
                        CAllocationOcaResource(const void* pAllocationOcaResource);
virtual                ~CAllocationOcaResource();

const   CMemberType&    type() const                { return m_allocationOcaResourceType; }

}; // CAllocationOcaResource

//******************************************************************************
//
//  class CAllocationOcaRecord
//
//******************************************************************************
class CAllocationOcaRecord : public CLwcdRecord
{
// ALLOCATION_OCA_RECORD Type Helpers
TYPE(allocationOcaRecord)

// ALLOCATION_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(Pitch)
FIELD(Height)
FIELD(Bpp)
FIELD(AccessibleSize)
FIELD(TotalSize)
FIELD(AllowedHeaps)
FIELD(PreferredHeap)
FIELD(Segment)
FIELD(Offset)
FIELD(Resource)

// ALLOCATION_OCA_RECORD Members
MEMBER(Pitch,           ULONG,      0,  public)
MEMBER(Height,          ULONG,      0,  public)
MEMBER(Bpp,             ULONG,      0,  public)
MEMBER(AccessibleSize,  ULONG,      0,  public)
MEMBER(TotalSize,       ULONG,      0,  public)
MEMBER(AllowedHeaps,    ULONG,      0,  public)
MEMBER(PreferredHeap,   ULONG,      0,  public)
MEMBER(Segment,         ULONG,      0,  public)
MEMBER(Offset,          ULONG64,    0,  public)

private:
const   void*           m_pAllocationOcaRecord;
const   COcaData*       m_pOcaData;
        CAllocationOcaResourcePtr m_pAllocationOcaResource;

public:
                        CAllocationOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CAllocationOcaRecord();

const   void*           allocationOcaRecord() const { return m_pAllocationOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CAllocationOcaResource* allocationOcaResource() const
                            { return m_pAllocationOcaResource; }

        ULONG           Type() const                { return ((m_pAllocationOcaResource != NULL) ? m_pAllocationOcaResource->Type() : 0); }
        ULONG           Format() const              { return ((m_pAllocationOcaResource != NULL) ? m_pAllocationOcaResource->Format() : 0); }
        ULONG           Width() const               { return ((m_pAllocationOcaResource != NULL) ? m_pAllocationOcaResource->Width() : 0); }
        ULONG           Height() const              { return ((m_pAllocationOcaResource != NULL) ? m_pAllocationOcaResource->Height() : 0); }
        ULONG           Depth() const               { return ((m_pAllocationOcaResource != NULL) ? m_pAllocationOcaResource->Depth() : 0); }
        ULONG           MipMapCount() const         { return ((m_pAllocationOcaResource != NULL) ? m_pAllocationOcaResource->MipMapCount() : 0); }
        ULONG           VidPnSourceId() const       { return ((m_pAllocationOcaResource != NULL) ? m_pAllocationOcaResource->VidPnSourceId() : 0); }

const   CMemberType&    type() const                { return m_allocationOcaRecordType; }

}; // CAllocationOcaRecord

//******************************************************************************
//
//  class CKmdProcessOcaRecord
//
//******************************************************************************
class CKmdProcessOcaRecord : public CLwcdRecord
{
// KMDPROCESS_OCA_RECORD Type Helpers
TYPE(kmdProcessOcaRecord)

// KMDPROCESS_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(hClient)
FIELD(hDevice)
FIELD(KmdProcess)
FIELD(Device)
FIELD(Adapter)
FIELD(Process)
FIELD(ProcessImageName)
FIELD(DeviceCount)

// KMDPROCESS_OCA_RECORD Members
MEMBER(hClient,             ULONG,      0,      public)
MEMBER(hDevice,             ULONG,      0,      public)
MEMBER(KmdProcess,          POINTER,    NULL,   public)
MEMBER(Device,              POINTER,    NULL,   public)
MEMBER(Adapter,             POINTER,    NULL,   public)
MEMBER(Process,             POINTER,    NULL,   public)
MEMBER(ProcessImageName,    BYTE,       0,      public)
MEMBER(DeviceCount,         ULONG,      0,      public)

private:
const   void*           m_pKmdProcessOcaRecord;
const   COcaData*       m_pOcaData;

public:
                        CKmdProcessOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CKmdProcessOcaRecord();

const   void*           kmdProcessOcaRecord() const { return m_pKmdProcessOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

virtual ULONG           size() const;

        CString         processName() const;

const   CMemberType&    type() const                { return m_kmdProcessOcaRecordType; }

}; // CKmdProcessOcaRecord

//******************************************************************************
//
//  class CDeviceOcaRecord
//
//******************************************************************************
class CDeviceOcaRecord : public CLwcdRecord
{
// DEVICE_OCA_RECORD Type Helpers
TYPE(deviceOcaRecord)

// DEVICE_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(hClient)
FIELD(hDevice)
FIELD(KmdProcess)
FIELD(Device)
FIELD(Adapter)
FIELD(Process)
FIELD(ContextCount)
FIELD(ReferenceCount)
FIELD(bLockedPerfMon)
FIELD(dmaBufferSize)

// DEVICE_OCA_RECORD Members
MEMBER(hClient,         ULONG,      0,      public)
MEMBER(hDevice,         ULONG,      0,      public)
MEMBER(KmdProcess,      POINTER,    NULL,   public)
MEMBER(Device,          POINTER,    NULL,   public)
MEMBER(Adapter,         POINTER,    NULL,   public)
MEMBER(Process,         POINTER,    NULL,   public)
MEMBER(ContextCount,    ULONG,      0,      public)
MEMBER(ReferenceCount,  ULONG,      0,      public)
MEMBER(bLockedPerfMon,  ULONG,      0,      public)
MEMBER(dmaBufferSize,   ULONG,      0,      public)

private:
const   void*           m_pDeviceOcaRecord;
const   COcaData*       m_pOcaData;

public:
                        CDeviceOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CDeviceOcaRecord();

const   void*           deviceOcaRecord() const     { return m_pDeviceOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CMemberType&    type() const                { return m_deviceOcaRecordType; }

}; // CDeviceOcaRecord

//******************************************************************************
//
//  class CContextOcaRecord
//
//******************************************************************************
class CContextOcaRecord : public CLwcdRecord
{
// CONTEXT_OCA_RECORD Type Helpers
TYPE(contextOcaRecord)

// CONTEXT_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(Context)
FIELD(Channel)
FIELD(Adapter)
FIELD(Device)
FIELD(NodeOrdinal)
FIELD(EngineOrdinal)
FIELD(BufferId)

// CONTEXT_OCA_RECORD Members
MEMBER(Context,         POINTER,    NULL,   public)
MEMBER(Channel,         POINTER,    NULL,   public)
MEMBER(Adapter,         POINTER,    NULL,   public)
MEMBER(Device,          POINTER,    NULL,   public)
MEMBER(NodeOrdinal,     ULONG,      0,      public)
MEMBER(EngineOrdinal,   ULONG,      0,      public)
MEMBER(BufferId,        ULONG,      0,      public)

private:
const   void*           m_pContextOcaRecord;
const   COcaData*       m_pOcaData;

public:
                        CContextOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CContextOcaRecord();

const   void*           contextOcaRecord() const    { return m_pContextOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CMemberType&    type() const                { return m_contextOcaRecordType; }

}; // CContextOcaRecord

//******************************************************************************
//
//  class CChannelOcaRecord
//
//******************************************************************************
class CChannelOcaRecord : public CLwcdRecord
{
// CHANNEL_OCA_RECORD Type Helpers
TYPE(channelOcaRecord)

// CHANNEL_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(Channel)
FIELD(HwChannelIndex)
FIELD(hClient)
FIELD(hDevice)
FIELD(DmaCount)
FIELD(NodeOrdinal)
FIELD(bShared)
FIELD(bReserved)
FIELD(ContextCount)

// CHANNEL_OCA_RECORD Members
MEMBER(Channel,         POINTER,    NULL,   public)
MEMBER(HwChannelIndex,  ULONG,      0,      public)
MEMBER(hClient,         ULONG,      0,      public)
MEMBER(hDevice,         ULONG,      0,      public)
MEMBER(DmaCount,        ULONG,      0,      public)
MEMBER(NodeOrdinal,     ULONG,      0,      public)
MEMBER(bShared,         ULONG,      0,      public)
MEMBER(bReserved,       ULONG,      0,      public)
MEMBER(ContextCount,    ULONG,      0,      public)

private:
const   void*           m_pChannelOcaRecord;
const   COcaData*       m_pOcaData;

public:
                        CChannelOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CChannelOcaRecord();

const   void*           channelOcaRecord() const    { return m_pChannelOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CMemberType&    type() const                { return m_channelOcaRecordType; }

}; // CChannelOcaRecord

//******************************************************************************
//
//  class CDisplayTargetOcaRecord
//
//******************************************************************************
class CDisplayTargetOcaRecord : public CLwcdRecord
{
// DISPLAY_TARGET_OCA_RECORD Type Helpers
TYPE(displayTargetOcaRecord)

// Display target Enum Helpers
ENUM(lwlScaling)

// DISPLAY_TARGET_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(VidPnTargetId)
FIELD(head)
FIELD(device)
FIELD(connector)
FIELD(srcID)
FIELD(srcImportance)
FIELD(hAllocation)
FIELD(Address)
FIELD(bFlipPending)
FIELD(flipPendingAddress)
FIELD(width)
FIELD(height)
FIELD(depth)
FIELD(refreshRate)
FIELD(colorFormat)
FIELD(rotation)
FIELD(callFromTMM)
FIELD(SelectLwstomTiming)
FIELD(tvFormat)
FIELD(srcPartitionX)
FIELD(srcPartitionY)
FIELD(srcPartitionW)
FIELD(srcPartitionH)
FIELD(viewportInX)
FIELD(viewportInY)
FIELD(viewportInW)
FIELD(viewportInH)
FIELD(scalingMethod)
FIELD(viewportOutX)
FIELD(viewportOutY)
FIELD(viewportOutW)
FIELD(viewportOutH)
FIELD(timingOverride)
FIELD(bVsyncEnabled)
FIELD(HVisible)
FIELD(HBorder)
FIELD(HFrontPorch)
FIELD(HSyncWidth)
FIELD(HTotal)
FIELD(HSyncPol)
FIELD(VVisible)
FIELD(VBorder)
FIELD(VFrontPorch)
FIELD(VSyncWidth)
FIELD(VTotal)
FIELD(VSyncPol)
FIELD(interlaced)
FIELD(pclk)
FIELD(flag)
FIELD(rr)
FIELD(rrx1k)
FIELD(aspect)
FIELD(rep)
FIELD(status)
FIELD(name)

// DISPLAY_TARGET_OCA_RECORD Members
MEMBER(VidPnTargetId,       DWORD,      0,      public)
MEMBER(head,                DWORD,      0,      public)
MEMBER(device,              DWORD,      0,      public)
MEMBER(connector,           DWORD,      0,      public)
MEMBER(srcID,               DWORD,      0,      public)
MEMBER(srcImportance,       DWORD,      0,      public)
MEMBER(hAllocation,         POINTER,    NULL,   public)
MEMBER(Address,             ULONG64,    0,      public)
MEMBER(bFlipPending,        DWORD,      0,      public)
MEMBER(flipPendingAddress,  ULONG64,    0,      public)
MEMBER(width,               DWORD,      0,      public)
MEMBER(height,              DWORD,      0,      public)
MEMBER(depth,               DWORD,      0,      public)
MEMBER(refreshRate,         DWORD,      0,      public)
MEMBER(colorFormat,         DWORD,      0,      public)
MEMBER(rotation,            DWORD,      0,      public)
MEMBER(callFromTMM,         DWORD,      0,      public)
MEMBER(SelectLwstomTiming,  DWORD,      0,      public)
MEMBER(tvFormat,            DWORD,      0,      public)
MEMBER(srcPartitionX,       DWORD,      0,      public)
MEMBER(srcPartitionY,       DWORD,      0,      public)
MEMBER(srcPartitionW,       DWORD,      0,      public)
MEMBER(srcPartitionH,       DWORD,      0,      public)
MEMBER(viewportInX,         DWORD,      0,      public)
MEMBER(viewportInY,         DWORD,      0,      public)
MEMBER(viewportInW,         DWORD,      0,      public)
MEMBER(viewportInH,         DWORD,      0,      public)
MEMBER(scalingMethod,       DWORD,      0,      public)
MEMBER(viewportOutX,        DWORD,      0,      public)
MEMBER(viewportOutY,        DWORD,      0,      public)
MEMBER(viewportOutW,        DWORD,      0,      public)
MEMBER(viewportOutH,        DWORD,      0,      public)
MEMBER(timingOverride,      DWORD,      0,      public)
MEMBER(bVsyncEnabled,       DWORD,      0,      public)
MEMBER(HVisible,            WORD,       0,      public)
MEMBER(HBorder,             WORD,       0,      public)
MEMBER(HFrontPorch,         WORD,       0,      public)
MEMBER(HSyncWidth,          WORD,       0,      public)
MEMBER(HTotal,              WORD,       0,      public)
MEMBER(HSyncPol,            BYTE,       0,      public)
MEMBER(VVisible,            WORD,       0,      public)
MEMBER(VBorder,             WORD,       0,      public)
MEMBER(VFrontPorch,         WORD,       0,      public)
MEMBER(VSyncWidth,          WORD,       0,      public)
MEMBER(VTotal,              WORD,       0,      public)
MEMBER(VSyncPol,            BYTE,       0,      public)
MEMBER(interlaced,          WORD,       0,      public)
MEMBER(pclk,                DWORD,      0,      public)
MEMBER(flag,                DWORD,      0,      public)
MEMBER(rr,                  WORD,       0,      public)
MEMBER(rrx1k,               DWORD,      0,      public)
MEMBER(aspect,              DWORD,      0,      public)
MEMBER(rep,                 WORD,       0,      public)
MEMBER(status,              DWORD,      0,      public)
MEMBER(name,                BYTE,       0,      public)

private:
const   void*           m_pDisplayTargetOcaRecord;
const   COcaData*       m_pOcaData;

public:
                        CDisplayTargetOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CDisplayTargetOcaRecord();

const   void*           displayTargetOcaRecord() const
                            { return m_pDisplayTargetOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CMemberType&    type() const                { return m_displayTargetOcaRecordType; }

}; // CDisplayTargetOcaRecord

//******************************************************************************
//
//  class CMonitorInfoOcaRecord
//
//******************************************************************************
class CMonitorInfoOcaRecord : public CLwcdRecord
{
// MONITOR_INFO_OCA_RECORD Type Helpers
TYPE(monitorInfoOcaRecord)

// MONITOR_INFO_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(dwEDIDSize)
FIELD(EDID)

// MONITOR_INFO_OCA_RECORD Members
MEMBER(dwEDIDSize,  ULONG,  0,  public)
MEMBER(EDID,        BYTE,   0,  public)

private:
const   void*           m_pMonitorInfoOcaRecord;
const   COcaData*       m_pOcaData;

public:
                        CMonitorInfoOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CMonitorInfoOcaRecord();

const   void*           monitorInfoOcaRecord() const{ return m_pMonitorInfoOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CMemberType&    type() const                { return m_monitorInfoOcaRecordType; }

}; // CMonitorInfoOcaRecord

//******************************************************************************
//
//  class CVblankInfoData
//
//******************************************************************************
class CVblankInfoData
{
// VBLANK_INFO_DATA Type Helpers
TYPE(vblankInfoData)

// VBLANK_INFO_DATA Field Helpers
FIELD(offset)
FIELD(head)
FIELD(VidPnTargetId)
FIELD(timestamp)

// VBLANK_INFO_DATA Members
MEMBER(offset,          ULONG,      0,  public)
MEMBER(head,            ULONG,      0,  public)
MEMBER(VidPnTargetId,   ULONG,      0,  public)
MEMBER(timestamp,       ULONG,      0,  public)

private:
        const void*     m_pVblankInfoData;

public:
                        CVblankInfoData(const void* pVblankInfoData);
virtual                ~CVblankInfoData();

const   void*           vblankInfoData() const      { return m_pVblankInfoData; }

const   CMemberType&    type() const                { return m_vblankInfoDataType; }

}; // CVblankInfoData

//******************************************************************************
//
// class CVblankInfoDatas
//
//******************************************************************************
class CVblankInfoDatas
{
protected:
        ULONG           m_ulVblankDataCount;

const   CVblankInfoOcaRecord* m_pVblankInfoOcaRecord;
mutable CVblankInfoDataArray m_aVblankInfoDatas;

public:
                        CVblankInfoDatas(const CVblankInfoOcaRecord* pVblankInfoOcaRecord, ULONG ulRemaining);
virtual                ~CVblankInfoDatas();

        ULONG           vblankDataCount() const     { return m_ulVblankDataCount; }

const   void*           vblankData(ULONG ulVblankInfo) const;

const   CVblankInfoData* vblankInfoData(ULONG ulVblankInfo) const;

}; // class CVblankInfoDatas

//******************************************************************************
//
//  class CVblankInfoOcaRecord
//
//******************************************************************************
class CVblankInfoOcaRecord : public CLwcdRecord
{
// VBLANK_INFO_OCA_RECORD Type Helpers
TYPE(vblankInfoOcaRecord)

// VBLANK_INFO_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(Adapter)
FIELD(frequency)
FIELD(count)
FIELD(data)

// VBLANK_INFO_OCA_RECORD Members
MEMBER(Adapter,     POINTER,    NULL,   public)
MEMBER(frequency,   ULONG64,    0,      public)
MEMBER(count,       ULONG,      0,      public)

private:
const   void*           m_pVblankInfoOcaRecord;
const   COcaData*       m_pOcaData;
        CVblankInfoDatasPtr m_pVblankInfoDatas;

public:
                        CVblankInfoOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CVblankInfoOcaRecord();

const   void*           vblankInfoOcaRecord() const { return m_pVblankInfoOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CVblankInfoDatas* vblankInfoDatas() const   { return m_pVblankInfoDatas; }
        ULONG           vblankDataCount() const     { return ((m_pVblankInfoDatas != NULL) ? m_pVblankInfoDatas->vblankDataCount() : 0); }
const   CVblankInfoData* vblankInfoData(ULONG ulVblankInfo) const
                            { return ((m_pVblankInfoDatas != NULL) ? m_pVblankInfoDatas->vblankInfoData(ulVblankInfo) : NULL); }

const   CMemberType&    type() const                { return m_vblankInfoOcaRecordType; }

}; // CVblankInfoOcaRecord

//******************************************************************************
//
//  class CErrorInfoData
//
//******************************************************************************
class CErrorInfoData
{
// ERROR_INFO_DATA Type Helpers
TYPE(errorInfoData)

// Error Event Enum Helpers
ENUM(errorEvent)
ENUM(dmInterface)
ENUM(lddmInterface)
ENUM(rmInterface)

// ERROR_INFO_DATA Field Helpers
FIELD(Type)
FIELD(subType)
FIELD(status)
FIELD(address)
FIELD(adapter)
FIELD(device)
FIELD(context)
FIELD(channel)
FIELD(allocation)
FIELD(process)
FIELD(thread)
FIELD(data0)
FIELD(data1)
FIELD(data2)
FIELD(data3)
FIELD(data4)
FIELD(timestamp)

// ERROR_INFO_DATA Members
MEMBER(Type,            ULONG,      0,      public)
MEMBER(subType,         ULONG,      0,      public)
MEMBER(status,          ULONG,      0,      public)
MEMBER(address,         POINTER,    NULL,   public)
MEMBER(adapter,         POINTER,    NULL,   public)
MEMBER(device,          POINTER,    NULL,   public)
MEMBER(context,         POINTER,    NULL,   public)
MEMBER(channel,         POINTER,    NULL,   public)
MEMBER(allocation,      POINTER,    NULL,   public)
MEMBER(process,         POINTER,    NULL,   public)
MEMBER(thread,          POINTER,    NULL,   public)
MEMBER(data0,           ULONG64,    0,      public)
MEMBER(data1,           ULONG64,    0,      public)
MEMBER(data2,           ULONG64,    0,      public)
MEMBER(data3,           ULONG64,    0,      public)
MEMBER(data4,           ULONG64,    0,      public)
MEMBER(timestamp,       ULONG64,    0,      public)

private:
        const void*     m_pErrorInfoData;

public:
                        CErrorInfoData(const void* pErrorInfoData);
virtual                ~CErrorInfoData();

const   void*           errorInfoData() const       { return m_pErrorInfoData; }

        CString         statusString() const;
        CString         errorString() const;

        CString         typeName() const;
        CString         subTypeName() const;
        CString         annotation() const;

        CString         format(const char* pFormat) const;

        CString         description() const;

        CString         openString(const char* pOptions = NULL) const;

const   CMemberType&    type() const                { return m_errorInfoDataType; }

}; // CErrorInfoData

//******************************************************************************
//
// class CErrorInfoDatas
//
//******************************************************************************
class CErrorInfoDatas
{
protected:
        ULONG           m_ulErrorDataCount;

const   CErrorInfoOcaRecord* m_pErrorInfoOcaRecord;
mutable CErrorInfoDataArray m_aErrorInfoDatas;

public:
                        CErrorInfoDatas(const CErrorInfoOcaRecord* pErrorInfoOcaRecord, ULONG ulRemaining);
virtual                ~CErrorInfoDatas();

        ULONG           errorDataCount() const      { return m_ulErrorDataCount; }

const   void*           errorData(ULONG ulErrorInfo) const;

const   CErrorInfoData* errorInfoData(ULONG ulErrorInfo) const;

}; // class CErrorInfoDatas

//******************************************************************************
//
//  class CErrorInfoOcaRecord
//
//******************************************************************************
class CErrorInfoOcaRecord : public CLwcdRecord
{
// ERROR_INFO_OCA_RECORD Type Helpers
TYPE(errorInfoOcaRecord)

// ERROR_INFO_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(frequency)
FIELD(count)

// ERROR_INFO_OCA_RECORD Members
MEMBER(frequency,   ULONG64,    0,  public)
MEMBER(count,       ULONG,      0,  public)

private:
const   void*           m_pErrorInfoOcaRecord;
const   COcaData*       m_pOcaData;
        CErrorInfoDatasPtr m_pErrorInfoDatas;

public:
                        CErrorInfoOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CErrorInfoOcaRecord();

const   void*           errorInfoOcaRecord() const  { return m_pErrorInfoOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CErrorInfoDatas* errorInfoDatas() const     { return m_pErrorInfoDatas; }
        ULONG           errorDataCount() const      { return ((m_pErrorInfoDatas != NULL) ? m_pErrorInfoDatas->errorDataCount() : 0); }
const   CErrorInfoData* errorInfoData(ULONG ulErrorInfo) const
                            { return ((m_pErrorInfoDatas != NULL) ? m_pErrorInfoDatas->errorInfoData(ulErrorInfo) : NULL); }

virtual ULONG           size() const;

const   CMemberType&    type() const                { return m_errorInfoOcaRecordType; }

}; // CErrorInfoOcaRecord

//******************************************************************************
//
//  class CWarningInfoData
//
//******************************************************************************
class CWarningInfoData
{
// WARNING_INFO_DATA Type Helpers
TYPE(warningInfoData)

// Warning Event Enum Helpers
ENUM(warningEvent)
ENUM(dmInterface)
ENUM(lddmInterface)
ENUM(rmInterface)

// WARNING_INFO_DATA Field Helpers
FIELD(Type)
FIELD(subType)
FIELD(status)
FIELD(address)
FIELD(adapter)
FIELD(device)
FIELD(context)
FIELD(channel)
FIELD(allocation)
FIELD(process)
FIELD(thread)
FIELD(data0)
FIELD(data1)
FIELD(data2)
FIELD(data3)
FIELD(data4)
FIELD(timestamp)

// WARNING_INFO_DATA Members
MEMBER(Type,            ULONG,      0,      public)
MEMBER(subType,         ULONG,      0,      public)
MEMBER(status,          ULONG,      0,      public)
MEMBER(address,         POINTER,    NULL,   public)
MEMBER(adapter,         POINTER,    NULL,   public)
MEMBER(device,          POINTER,    NULL,   public)
MEMBER(context,         POINTER,    NULL,   public)
MEMBER(channel,         POINTER,    NULL,   public)
MEMBER(allocation,      POINTER,    NULL,   public)
MEMBER(process,         POINTER,    NULL,   public)
MEMBER(thread,          POINTER,    NULL,   public)
MEMBER(data0,           ULONG64,    0,      public)
MEMBER(data1,           ULONG64,    0,      public)
MEMBER(data2,           ULONG64,    0,      public)
MEMBER(data3,           ULONG64,    0,      public)
MEMBER(data4,           ULONG64,    0,      public)
MEMBER(timestamp,       ULONG64,    0,      public)

private:
        const void*     m_pWarningInfoData;

public:
                        CWarningInfoData(const void* pWarningInfoData);
virtual                ~CWarningInfoData();

const   void*           warningInfoData() const     { return m_pWarningInfoData; }

        CString         statusString() const;
        CString         errorString() const;

        CString         typeName() const;
        CString         subTypeName() const;
        CString         annotation() const;

        CString         format(const char* pFormat) const;

        CString         description() const;

        CString         openString(const char* pOptions = NULL) const;

const   CMemberType&    type() const                { return m_warningInfoDataType; }

}; // CWarningInfoData

//******************************************************************************
//
// class CWarningInfoDatas
//
//******************************************************************************
class CWarningInfoDatas
{
protected:
        ULONG           m_ulWarningDataCount;

const   CWarningInfoOcaRecord* m_pWarningInfoOcaRecord;
mutable CWarningInfoDataArray m_aWarningInfoDatas;

public:
                        CWarningInfoDatas(const CWarningInfoOcaRecord* pWarningInfoOcaRecord, ULONG ulRemaining);
virtual                ~CWarningInfoDatas();

        ULONG           warningDataCount() const    { return m_ulWarningDataCount; }

const   void*           warningData(ULONG ulWarningInfo) const;

const   CWarningInfoData* warningInfoData(ULONG ulWarningInfo) const;

}; // class CWarningInfoDatas

//******************************************************************************
//
//  class CWarningInfoOcaRecord
//
//******************************************************************************
class CWarningInfoOcaRecord : public CLwcdRecord
{
// WARNING_INFO_OCA_RECORD Type Helpers
TYPE(warningInfoOcaRecord)

// WARNING_INFO_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(frequency)
FIELD(count)

// WARNING_INFO_OCA_RECORD Members
MEMBER(frequency,   ULONG64,    0,  public)
MEMBER(count,       ULONG,      0,  public)

private:
const   void*           m_pWarningInfoOcaRecord;
const   COcaData*       m_pOcaData;
        CWarningInfoDatasPtr m_pWarningInfoDatas;

public:
                        CWarningInfoOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CWarningInfoOcaRecord();

const   void*           warningInfoOcaRecord() const{ return m_pWarningInfoOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CWarningInfoDatas* warningInfoDatas() const { return m_pWarningInfoDatas; }
        ULONG           warningDataCount() const    { return ((m_pWarningInfoDatas != NULL) ? m_pWarningInfoDatas->warningDataCount() : 0); }
const   CWarningInfoData* warningInfoData(ULONG ulWarningInfo) const
                            { return ((m_pWarningInfoDatas != NULL) ? m_pWarningInfoDatas->warningInfoData(ulWarningInfo) : NULL); }

virtual ULONG           size() const;

const   CMemberType&    type() const                { return m_warningInfoOcaRecordType; }

}; // CWarningInfoOcaRecord

//******************************************************************************
//
//  class CPagingInfoData
//
//******************************************************************************
class CPagingInfoData
{
// PAGING_INFO_DATA Type Helpers
TYPE(pagingInfoData)

// Paging Event Enum Helpers
ENUM(pagingEvent)

// PAGING_INFO_DATA Field Helpers
FIELD(Type)
FIELD(subType)
FIELD(status)
FIELD(address)
FIELD(adapter)
FIELD(device)
FIELD(context)
FIELD(channel)
FIELD(allocation)
FIELD(process)
FIELD(thread)
FIELD(data0)
FIELD(data1)
FIELD(data2)
FIELD(data3)
FIELD(data4)
FIELD(timestamp)

// PAGING_INFO_DATA Members
MEMBER(Type,            ULONG,      0,      public)
MEMBER(subType,         ULONG,      0,      public)
MEMBER(status,          ULONG,      0,      public)
MEMBER(address,         POINTER,    NULL,   public)
MEMBER(adapter,         POINTER,    NULL,   public)
MEMBER(device,          POINTER,    NULL,   public)
MEMBER(context,         POINTER,    NULL,   public)
MEMBER(channel,         POINTER,    NULL,   public)
MEMBER(allocation,      POINTER,    NULL,   public)
MEMBER(process,         POINTER,    NULL,   public)
MEMBER(thread,          POINTER,    NULL,   public)
MEMBER(data0,           ULONG64,    0,      public)
MEMBER(data1,           ULONG64,    0,      public)
MEMBER(data2,           ULONG64,    0,      public)
MEMBER(data3,           ULONG64,    0,      public)
MEMBER(data4,           ULONG64,    0,      public)
MEMBER(timestamp,       ULONG64,    0,      public)

private:
        const void*     m_pPagingInfoData;

public:
                        CPagingInfoData(const void* pPagingInfoData);
virtual                ~CPagingInfoData();

const   void*           pagingInfoData() const     { return m_pPagingInfoData; }

        CString         statusString() const;
        CString         errorString() const;

        CString         typeName() const;
        CString         subTypeName() const;
        CString         annotation() const;

        CString         format(const char* pFormat) const;

        CString         description() const;

        CString         openString(const char* pOptions = NULL) const;

const   CMemberType&    type() const                { return m_pagingInfoDataType; }

}; // CPagingInfoData

//******************************************************************************
//
// class CPagingInfoDatas
//
//******************************************************************************
class CPagingInfoDatas
{
protected:
        ULONG           m_ulPagingDataCount;

const   CPagingInfoOcaRecord* m_pPagingInfoOcaRecord;
mutable CPagingInfoDataArray m_aPagingInfoDatas;

public:
                        CPagingInfoDatas(const CPagingInfoOcaRecord* pPagingInfoOcaRecord, ULONG ulRemaining);
virtual                ~CPagingInfoDatas();

        ULONG           pagingDataCount() const     { return m_ulPagingDataCount; }

const   void*           pagingData(ULONG ulPagingInfo) const;

const   CPagingInfoData* pagingInfoData(ULONG ulPagingInfo) const;

}; // class CPagingInfoDatas

//******************************************************************************
//
//  class CPagingInfoOcaRecord
//
//******************************************************************************
class CPagingInfoOcaRecord : public CLwcdRecord
{
// PAGING_INFO_OCA_RECORD Type Helpers
TYPE(pagingInfoOcaRecord)

// PAGING_INFO_OCA_RECORD Field Helpers
FIELD(Header)
FIELD(frequency)
FIELD(count)

// PAGING_INFO_OCA_RECORD Members
MEMBER(frequency,   ULONG64,    0,  public)
MEMBER(count,       ULONG,      0,  public)

private:
const   void*           m_pPagingInfoOcaRecord;
const   COcaData*       m_pOcaData;
        CPagingInfoDatasPtr m_pPagingInfoDatas;

public:
                        CPagingInfoOcaRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CPagingInfoOcaRecord();

const   void*           pagingInfoOcaRecord() const { return m_pPagingInfoOcaRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

const   CPagingInfoDatas* pagingInfoDatas() const   { return m_pPagingInfoDatas; }
        ULONG           pagingDataCount() const     { return ((m_pPagingInfoDatas != NULL) ? m_pPagingInfoDatas->pagingDataCount() : 0); }
const   CPagingInfoData* pagingInfoData(ULONG ulPagingInfo) const
                            { return ((m_pPagingInfoDatas != NULL) ? m_pPagingInfoDatas->pagingInfoData(ulPagingInfo) : NULL); }

virtual ULONG           size() const;

const   CMemberType&    type() const                { return m_pagingInfoOcaRecordType; }

}; // CPagingInfoOcaRecord

//******************************************************************************
//
//  class CAdapterOcaRecords
//
//******************************************************************************
class CAdapterOcaRecords : public CRefObj
{
private:
const   CDrvOcaRecords* m_pDrvOcaRecords;
        ULONG           m_ulAdapterOcaRecordCount;

mutable DwordArray      m_aDrvOcaIndices;

public:
                        CAdapterOcaRecords(const CDrvOcaRecords* pDrvOcaRecords);
virtual                ~CAdapterOcaRecords();

        ULONG           adapterOcaRecordCount() const
                            { return m_ulAdapterOcaRecordCount; }

const   CAdapterOcaRecord* adapterOcaRecord(ULONG ulAdapterOcaRecord) const;

};  // CAdapterOcaRecords

//******************************************************************************
//
//  class CDeviceOcaRecords
//
//******************************************************************************
class CDeviceOcaRecords : public CRefObj
{
private:
const   CDrvOcaRecords* m_pDrvOcaRecords;
        ULONG           m_ulDeviceOcaRecordCount;

mutable DwordArray      m_aDrvOcaIndices;

public:
                        CDeviceOcaRecords(const CDrvOcaRecords* pDrvOcaRecords);
                        CDeviceOcaRecords(const CAdapterOcaRecord* pAdapterOcaRecord);
virtual                ~CDeviceOcaRecords();

        ULONG           deviceOcaRecordCount() const
                            { return m_ulDeviceOcaRecordCount; }

const   CDeviceOcaRecord* deviceOcaRecord(ULONG ulDeviceOcaRecord) const;

};  // CDeviceOcaRecords

//******************************************************************************
//
//  class CContextOcaRecords
//
//******************************************************************************
class CContextOcaRecords : public CRefObj
{
private:
const   CDrvOcaRecords* m_pDrvOcaRecords;
        ULONG           m_ulContextOcaRecordCount;

mutable DwordArray      m_aDrvOcaIndices;

public:
                        CContextOcaRecords(const CDrvOcaRecords* pDrvOcaRecords);
virtual                ~CContextOcaRecords();

        ULONG           contextOcaRecordCount() const
                            { return m_ulContextOcaRecordCount; }

const   CContextOcaRecord* contextOcaRecord(ULONG ulContextOcaRecord) const;

};  // CContextOcaRecords

//******************************************************************************
//
//  class CChannelOcaRecords
//
//******************************************************************************
class CChannelOcaRecords : public CRefObj
{
private:
const   CDrvOcaRecords* m_pDrvOcaRecords;
        ULONG           m_ulChannelOcaRecordCount;

mutable DwordArray      m_aDrvOcaIndices;

public:
                        CChannelOcaRecords(const CDrvOcaRecords* pDrvOcaRecords);
virtual                ~CChannelOcaRecords();

        ULONG           channelOcaRecordCount() const
                            { return m_ulChannelOcaRecordCount; }

const   CChannelOcaRecord* channelOcaRecord(ULONG ulChannelOcaRecord) const;

};  // CChannelOcaRecords

//******************************************************************************
//
//  class CAllocationOcaRecords
//
//******************************************************************************
class CAllocationOcaRecords : public CRefObj
{
private:
const   CDrvOcaRecords* m_pDrvOcaRecords;
        ULONG           m_ulAllocationOcaRecordCount;

mutable DwordArray      m_aDrvOcaIndices;

public:
                        CAllocationOcaRecords(const CDrvOcaRecords* pDrvOcaRecords);
                        CAllocationOcaRecords(const CAdapterOcaRecord* pAdapterOcaRecord);
virtual                ~CAllocationOcaRecords();

        ULONG           allocationOcaRecordCount() const
                            { return m_ulAllocationOcaRecordCount; }

const   CAllocationOcaRecord* allocationOcaRecord(ULONG ulAllocationOcaRecord) const;

};  // CAllocationOcaRecords

//******************************************************************************
//
//  class CKmdProcessOcaRecords
//
//******************************************************************************
class CKmdProcessOcaRecords : public CRefObj
{
private:
const   CDrvOcaRecords* m_pDrvOcaRecords;
        ULONG           m_ulKmdProcessOcaRecordCount;

mutable DwordArray      m_aDrvOcaIndices;

public:
                        CKmdProcessOcaRecords(const CDrvOcaRecords* pDrvOcaRecords);
virtual                ~CKmdProcessOcaRecords();

        ULONG           kmdProcessOcaRecordCount() const
                            { return m_ulKmdProcessOcaRecordCount; }

const   CKmdProcessOcaRecord* kmdProcessOcaRecord(ULONG ulKmdProcessOcaRecord) const;

};  // CKmdProcessOcaRecords

//******************************************************************************
//
//  class CDmaBufferOcaRecords
//
//******************************************************************************
class CDmaBufferOcaRecords : public CRefObj
{
private:
const   CDrvOcaRecords* m_pDrvOcaRecords;
        ULONG           m_ulDmaBufferOcaRecordCount;

mutable DwordArray      m_aDrvOcaIndices;

public:
                        CDmaBufferOcaRecords(const CDrvOcaRecords* pDrvOcaRecords);
virtual                ~CDmaBufferOcaRecords();

        ULONG           dmaBufferOcaRecordCount() const
                            { return m_ulDmaBufferOcaRecordCount; }

const   CDmaBufferOcaRecord* dmaBufferOcaRecord(ULONG ulDmaBufferOcaRecord) const;

};  // CDmaBufferOcaRecords

//******************************************************************************
//
//  Functions
//
//******************************************************************************
extern  const CErrorInfoOcaRecord*  findOcaErrorRecords();
extern  const CWarningInfoOcaRecord*findOcaWarningRecords();
extern  const CAdapterOcaRecord*    findOcaAdapter(ULONG64 ulAdapter);
extern  const CDeviceOcaRecord*     findOcaDevice(ULONG64 ulDevice);
extern  const CContextOcaRecord*    findOcaContext(ULONG64 ulContext);
extern  const CChannelOcaRecord*    findOcaChannel(ULONG64 ulChannel);
extern  const CKmdProcessOcaRecord* findOcaProcess(ULONG64 ulProcess);
extern  const CKmdProcessOcaRecord* findOcaKmdProcess(ULONG64 ulKmdProcess);
extern  const CEngineIdOcaRecord*   findEngineId(ULONG64 ulAdapter, ULONG ulEngineId);
extern  const CEngineIdOcaRecord*   findEngineId(ULONG64 ulAdapter, ULONG ulNodeOrdinal, ULONG ulEngineOrdinal);

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OCADRV_H
