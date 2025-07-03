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
|*  Module: ocaname.h                                                         *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OCANAME_H
#define _OCANAME_H

//******************************************************************************
//
// oca namespace entries
//
//******************************************************************************
// In ocatypes.h
using oca::COcaData;
using oca::COcaGuid;
using oca::CLwcdHeader;
using oca::CLwcdRecord;
using oca::CLwcdRecords;
using oca::CSysOcaRecords;
using oca::CRmOcaRecords;
using oca::CDrvOcaRecords;
using oca::CHwOcaRecords;
using oca::CInstOcaRecords;

using oca::CRmProtoBufRecord;

using oca::CKmdOcaFlags;
using oca::CKmdOcaRecord;
using oca::CHybridControlOcaFlags;
using oca::CHybridControlOcaRecord;
using oca::CIsrDpcOcaData;
using oca::CIsrDpcOcaDatas;
using oca::CIsrDpcOcaRecord;
using oca::CAdapterOcaFlags;
using oca::CAdapterOcaRecord;
using oca::CBufferInfoRecord;
using oca::CBufferInfoRecords;
using oca::CEngineIdOcaRecord;
using oca::CDmaBufferOcaRecord;
using oca::CGpuWatchdogEvent;
using oca::CGpuWatchdogEvents;
using oca::CKmdRingBuffer;
using oca::CKmdRingBufferOcaRecord;
using oca::CAllocationOcaResource;
using oca::CAllocationOcaRecord;
using oca::CKmdProcessOcaRecord;
using oca::CDeviceOcaRecord;
using oca::CContextOcaRecord;
using oca::CChannelOcaRecord;
using oca::CDisplayTargetOcaRecord;
using oca::CMonitorInfoOcaRecord;
using oca::CVblankInfoData;
using oca::CVblankInfoDatas;
using oca::CVblankInfoOcaRecord;
using oca::CErrorInfoData;
using oca::CErrorInfoDatas;
using oca::CErrorInfoOcaRecord;
using oca::CWarningInfoData;
using oca::CWarningInfoDatas;
using oca::CWarningInfoOcaRecord;
using oca::CPagingInfoData;
using oca::CPagingInfoDatas;
using oca::CPagingInfoOcaRecord;

using oca::CAdapterOcaRecords;
using oca::CDeviceOcaRecords;
using oca::CContextOcaRecords;
using oca::CChannelOcaRecords;
using oca::CAllocationOcaRecords;
using oca::CKmdProcessOcaRecords;
using oca::CDmaBufferOcaRecords;


using oca::COcaDataPtr;
using oca::COcaGuidPtr;
using oca::CLwcdHeaderPtr;
using oca::CLwcdRecordPtr;
using oca::CLwcdRecordsPtr;
using oca::CSysOcaRecordsPtr;
using oca::CRmOcaRecordsPtr;
using oca::CDrvOcaRecordsPtr;
using oca::CHwOcaRecordsPtr;
using oca::CInstOcaRecordsPtr;

using oca::CRmProtoBufRecordPtr;

using oca::CKmdOcaFlagsPtr;
using oca::CKmdOcaRecordPtr;
using oca::CHybridControlOcaFlagsPtr;
using oca::CHybridControlOcaRecordPtr;
using oca::CIsrDpcOcaDataPtr;
using oca::CIsrDpcOcaDatasPtr;
using oca::CIsrDpcOcaRecordPtr;
using oca::CAdapterOcaFlagsPtr;
using oca::CAdapterOcaRecordPtr;
using oca::CBufferInfoRecordPtr;
using oca::CBufferInfoRecordsPtr;
using oca::CEngineIdOcaRecordPtr;
using oca::CDmaBufferOcaRecordPtr;
using oca::CGpuWatchdogEventPtr;
using oca::CGpuWatchdogEventsPtr;
using oca::CKmdRingBufferPtr;
using oca::CKmdRingBufferOcaRecordPtr;
using oca::CAllocationOcaResourcePtr;
using oca::CAllocationOcaRecordPtr;
using oca::CKmdProcessOcaRecordPtr;
using oca::CDeviceOcaRecordPtr;
using oca::CContextOcaRecordPtr;
using oca::CChannelOcaRecordPtr;
using oca::CDisplayTargetOcaRecordPtr;
using oca::CMonitorInfoOcaRecordPtr;
using oca::CVblankInfoDataPtr;
using oca::CVblankInfoDatasPtr;
using oca::CVblankInfoOcaRecordPtr;
using oca::CErrorInfoDataPtr;
using oca::CErrorInfoDatasPtr;
using oca::CErrorInfoOcaRecordPtr;
using oca::CWarningInfoDataPtr;
using oca::CWarningInfoDatasPtr;
using oca::CWarningInfoOcaRecordPtr;
using oca::CPagingInfoDataPtr;
using oca::CPagingInfoDatasPtr;
using oca::CPagingInfoOcaRecordPtr;

using oca::CAdapterOcaRecordsPtr;
using oca::CDeviceOcaRecordsPtr;
using oca::CContextOcaRecordsPtr;
using oca::CChannelOcaRecordsPtr;
using oca::CAllocationOcaRecordsPtr;
using oca::CKmdProcessOcaRecordsPtr;
using oca::CDmaBufferOcaRecordsPtr;

using oca::CLwcdRecordArray;
using oca::CIsrDpcOcaDataArray;
using oca::CBufferInfoRecordArray;
using oca::CGpuWatchdogEventArray;
using oca::CVblankInfoDataArray;
using oca::CErrorInfoDataArray;
using oca::CWarningInfoDataArray;
using oca::CPagingInfoDataArray;

// In oca.h
using oca::ocaData;
using oca::freeOcaData;

using oca::guidKmd;
using oca::guidDxg;
using oca::guidLwcd1;
using oca::guidLwcd2;

// In ocarm.h
using oca::RmGlobalInfo;
using oca::RmLwrrentStateInfo;
using oca::RmFifoErrorInfo;
using oca::RmGraphicsErrorInfo;
using oca::RmLwrrentStateInfo_V2;
using oca::RmFifoErrorInfo_V2;
using oca::RmGraphicsErrorInfo_V2;
using oca::RmGlobalInfo_V2;
using oca::RmBusErrorInfo;
using oca::RmMissedNotifierInfo;
using oca::RmGlobalInfo_V3;
using oca::RmFifoErrorInfo_V3;
using oca::RmGraphicsErrorInfo_V3;
using oca::RmBspErrorInfo;

using oca::RmRC2GlobalCollectionRec;
using oca::RmRC2CrashStateCollectionRec;
using oca::RmRC2RcErrorCollectionRec;
using oca::RmRC2HwExceptionCollectionRec;
using oca::RmRC2SwExceptionCollectionRec;
using oca::RmRC2HwPriTimeoutCollectionRec;
using oca::RmRC2HwBusErrorCollectionRec;
using oca::RmRC2HwTmrErrorCollectionRec;
using oca::RmRC2HwThermalEventCollectionRec;
using oca::RmRC2SaveRestoreFBRetryCollectionRec;

using oca::RmRC2GenericCounter;
using oca::RmRC2SwDbgBreakpoint;
using oca::RmRC2SwRmAssert;
using oca::RmRC2GpuTimeout;
using oca::RmRC2TempReading;
using oca::RmRC2RequestedGpuSignals;

using oca::RmRC2GpuShortInfo;
using oca::RmRC2BiosShortInfo;

using oca::RmRC2AgpShortInfo;
using oca::RmRC2ShortPci30CfgInfo;
using oca::RmRC2PexCapsStateInfo;
using oca::RmRC2PexAERCapsStateInfo;
using oca::RmRC2PexVCCapsStateInfo;

using oca::RmRC2GpuEnginePBUS_HAL;
using oca::RmRC2GpuEnginePFB_HAL;
using oca::RmRC2GpuEnginePFIFO_HAL;
using oca::RmRC2GpuEnginePGRAPH_HAL;
using oca::RmRC2GpuEnginePHOST_HAL;
using oca::RmRC2GpuEnginePMC_HAL;
using oca::RmRC2GpuEnginePTIMER_HAL;

using oca::RmRC2GpuEngineUnitPBUS_DEBUG_HAL;
using oca::RmRC2GpuEngineUnitPBUS_PCI_HAL;
using oca::RmRC2GpuEngineUnitPBUS_THERMALCTRL_HAL;
using oca::RmRC2GpuEngineUnitPBUS_UMA_HAL;
using oca::RmRC2GpuEngineUnitPFB_GART_HAL;
using oca::RmRC2GpuEngineUnitPFB_GTLB_HAL;
using oca::RmRC2GpuEngineUnitPFIFO_C1DUMP_HAL;
using oca::RmRC2GpuEngineUnitPFIFO_CHANNEL_HAL;
using oca::RmRC2GpuEngineUnitPFIFO_DEVICE_HAL;
using oca::RmRC2GpuEngineUnitPGRAPH_FFINTFC_HAL;
using oca::RmRC2GpuEngineUnitPGRAPH_SEMAPHORE_HAL;
using oca::RmRC2GpuEngineUnitPTIMER_PRITO_HAL;
using oca::RmRC2GpuEngineUnitPTIMER_TIME_HAL;
using oca::RmGraphicsErrorInfo_V4;
using oca::RmErrorBlock;

using oca::RmFifoErrorInfo_V4;
using oca::RmGraphicsErrorInfo_V5;
using oca::RmGraphicsErrorInfo_V6;
using oca::RmBusErrorInfo_V2;
using oca::RmGraphicsErrorInfo_V7;
using oca::RmSafeGlobalInfo;
using oca::RmJournalInfo;
using oca::RmRingBuffer;
using oca::RmRC2SmuCommandInfo;
using oca::RmRC2PstateInfo;
using oca::RmVpErrorInfo;
using oca::RmVbiosInfo_V2;
using oca::RmRC2SmuErrorInfo;
using oca::RmRC2SwRmAssert_V2;
using oca::RmRC2GpuTimeout_V2;
using oca::RmRC2SmuCommandInfo_V2;
using oca::RmRC2PstateInfo_V2;
using oca::RmRC2SwDbgBreakpoint_V2;
using oca::RmPrbErrorInfo;
using oca::RmElpgInfo;
using oca::RmBadRead;
using oca::RmJournalInfo_V3;
using oca::RmSurpriseRemoval;
using oca::RmProtoBuf;
using oca::RmProtoBuf_V2;
using oca::RmDclMsg;
using oca::RmJournalEngDump;
using oca::DP_ASSERT_HIT;
using oca::DP_LOG_CALL;
using oca::RmPrbFullDump;

// In ocadrv.h
using oca::findOcaErrorRecords;
using oca::findOcaWarningRecords;
using oca::findOcaAdapter;
using oca::findOcaDevice;
using oca::findOcaContext;
using oca::findOcaChannel;
using oca::findOcaProcess;
using oca::findOcaKmdProcess;
using oca::findEngineId;
using oca::findEngineId;

using oca::KmdGlobalInfo;
using oca::KmdAdapterInfo;
using oca::KmdEngineIdInfo;
using oca::KmdDeviceInfo;
using oca::KmdContextInfo;
using oca::KmdChannelInfo;
using oca::KmdAllocationInfo;
using oca::KmdDmaBufferInfo;
using oca::KmdRingBufferInfo;
using oca::KmdDisplayTargetInfo;
using oca::KmdHybridControlInfo;
using oca::KmdMonitorInfo;
using oca::KmdGlobalInfo_V2;
using oca::KmdIsrDpcTimingInfo;
using oca::KmdRingBufferInfo_V2;
using oca::KmdDisplayTargetInfo_V2;
using oca::KmdVblankInfo;
using oca::KmdErrorInfo;
using oca::KmdWarningInfo;
using oca::KmdPagingInfo;
using oca::KmdProcessInfo;
using oca::KmdProcessInfo_V2;
using oca::KmdDeviceInfo_V2;

// In ocarc.h
using oca::rcType;
using oca::rcLevel;
using oca::rcEngine;

using oca::ocaRcErrorCount;

using oca::findRcErrorRecords;
using oca::displayOcaRcErrors;

// In ocatdr.h
using oca::tdrName;
using oca::tdrReason;
using oca::tdrDescription;
using oca::tdrForced;
using oca::forcedString;

using oca::findTdrVsync;
using oca::findGpuWatchdogEvent;
using oca::findTdrAdapter;
using oca::findTdrContext;
using oca::findTdrBuffer;
using oca::findTdrEngine;

using oca::findRmProtoBuf;
using oca::rmProtoBufData;
using oca::rmProtoBufSize;

using oca::ocaAdapterCount;
using oca::ocaEngineIdCount;
using oca::ocaResetEngineCount;

using oca::findResetEngineProcesses;
using oca::displayOcaResetEngineProcesses;

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OCANAME_H
