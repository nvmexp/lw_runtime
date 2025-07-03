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
|*  Module: ocatypes.h                                                        *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OCATYPES_H
#define _OCATYPES_H

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Fowards
//
//******************************************************************************
class COcaData;
class COcaGuid;
class CLwcdHeader;
class CLwcdRecord;
class CLwcdRecords;
class CSysOcaRecords;
class CRmOcaRecords;
class CDrvOcaRecords;
class CHwOcaRecords;
class CInstOcaRecords;

class CRmProtoBufRecord;

class CKmdOcaFlags;
class CKmdOcaRecord;
class CHybridControlOcaFlags;
class CHybridControlOcaRecord;
class CIsrDpcOcaData;
class CIsrDpcOcaDatas;
class CIsrDpcOcaRecord;
class CAdapterOcaFlags;
class CAdapterOcaRecord;
class CBufferInfoRecord;
class CBufferInfoRecords;
class CEngineIdOcaRecord;
class CDmaBufferOcaRecord;
class CGpuWatchdogEvent;
class CGpuWatchdogEvents;
class CKmdRingBuffer;
class CKmdRingBufferOcaRecord;
class CAllocationOcaResource;
class CAllocationOcaRecord;
class CKmdProcessOcaRecord;
class CDeviceOcaRecord;
class CContextOcaRecord;
class CChannelOcaRecord;
class CDisplayTargetOcaRecord;
class CMonitorInfoOcaRecord;
class CVblankInfoData;
class CVblankInfoDatas;
class CVblankInfoOcaRecord;
class CErrorInfoData;
class CErrorInfoDatas;
class CErrorInfoOcaRecord;
class CWarningInfoData;
class CWarningInfoDatas;
class CWarningInfoOcaRecord;
class CPagingInfoData;
class CPagingInfoDatas;
class CPagingInfoOcaRecord;

class CAdapterOcaRecords;
class CDeviceOcaRecords;
class CContextOcaRecords;
class CChannelOcaRecords;
class CAllocationOcaRecords;
class CKmdProcessOcaRecords;
class CDmaBufferOcaRecords;

typedef CAdapterOcaRecord                   COcaAdapter;
typedef CAllocationOcaRecord                COcaAllocation;
typedef CDeviceOcaRecord                    COcaDevice;
typedef CContextOcaRecord                   COcaContext;
typedef CChannelOcaRecord                   COcaChannel;

//******************************************************************************
//
//  Template type definitions (Mostly smart pointer definitions)
//
//******************************************************************************
typedef CRefPtr<COcaData>                   COcaDataPtr;
typedef CPtr<COcaGuid>                      COcaGuidPtr;
typedef CPtr<CLwcdHeader>                   CLwcdHeaderPtr;
typedef CPtr<CLwcdRecord>                   CLwcdRecordPtr;
typedef CPtr<CLwcdRecords>                  CLwcdRecordsPtr;
typedef CPtr<CSysOcaRecords>                CSysOcaRecordsPtr;
typedef CPtr<CRmOcaRecords>                 CRmOcaRecordsPtr;
typedef CPtr<CDrvOcaRecords>                CDrvOcaRecordsPtr;
typedef CPtr<CHwOcaRecords>                 CHwOcaRecordsPtr;
typedef CPtr<CInstOcaRecords>               CInstOcaRecordsPtr;

typedef CPtr<CRmProtoBufRecord>             CRmProtoBufRecordPtr;

typedef CPtr<CKmdOcaFlags>                  CKmdOcaFlagsPtr;
typedef CPtr<CKmdOcaRecord>                 CKmdOcaRecordPtr;
typedef CPtr<CHybridControlOcaFlags>        CHybridControlOcaFlagsPtr;
typedef CPtr<CHybridControlOcaRecord>       CHybridControlOcaRecordPtr;
typedef CPtr<CIsrDpcOcaData>                CIsrDpcOcaDataPtr;
typedef CPtr<CIsrDpcOcaDatas>               CIsrDpcOcaDatasPtr;
typedef CPtr<CIsrDpcOcaRecord>              CIsrDpcOcaRecordPtr;
typedef CPtr<CAdapterOcaFlags>              CAdapterOcaFlagsPtr;
typedef CPtr<CAdapterOcaRecord>             CAdapterOcaRecordPtr;
typedef CPtr<CBufferInfoRecord>             CBufferInfoRecordPtr;
typedef CPtr<CBufferInfoRecords>            CBufferInfoRecordsPtr;
typedef CPtr<CEngineIdOcaRecord>            CEngineIdOcaRecordPtr;
typedef CPtr<CDmaBufferOcaRecord>           CDmaBufferOcaRecordPtr;
typedef CPtr<CGpuWatchdogEvent>             CGpuWatchdogEventPtr;
typedef CPtr<CGpuWatchdogEvents>            CGpuWatchdogEventsPtr;
typedef CPtr<CKmdRingBuffer>                CKmdRingBufferPtr;
typedef CPtr<CKmdRingBufferOcaRecord>       CKmdRingBufferOcaRecordPtr;
typedef CPtr<CAllocationOcaResource>        CAllocationOcaResourcePtr;
typedef CPtr<CAllocationOcaRecord>          CAllocationOcaRecordPtr;
typedef CPtr<CKmdProcessOcaRecord>          CKmdProcessOcaRecordPtr;
typedef CPtr<CDeviceOcaRecord>              CDeviceOcaRecordPtr;
typedef CPtr<CContextOcaRecord>             CContextOcaRecordPtr;
typedef CPtr<CChannelOcaRecord>             CChannelOcaRecordPtr;
typedef CPtr<CDisplayTargetOcaRecord>       CDisplayTargetOcaRecordPtr;
typedef CPtr<CMonitorInfoOcaRecord>         CMonitorInfoOcaRecordPtr;
typedef CPtr<CVblankInfoData>               CVblankInfoDataPtr;
typedef CPtr<CVblankInfoDatas>              CVblankInfoDatasPtr;
typedef CPtr<CVblankInfoOcaRecord>          CVblankInfoOcaRecordPtr;
typedef CPtr<CErrorInfoData>                CErrorInfoDataPtr;
typedef CPtr<CErrorInfoDatas>               CErrorInfoDatasPtr;
typedef CPtr<CErrorInfoOcaRecord>           CErrorInfoOcaRecordPtr;
typedef CPtr<CWarningInfoData>              CWarningInfoDataPtr;
typedef CPtr<CWarningInfoDatas>             CWarningInfoDatasPtr;
typedef CPtr<CWarningInfoOcaRecord>         CWarningInfoOcaRecordPtr;
typedef CPtr<CPagingInfoData>               CPagingInfoDataPtr;
typedef CPtr<CPagingInfoDatas>              CPagingInfoDatasPtr;
typedef CPtr<CPagingInfoOcaRecord>          CPagingInfoOcaRecordPtr;

typedef CRefPtr<CAdapterOcaRecords>         CAdapterOcaRecordsPtr;
typedef CRefPtr<CDeviceOcaRecords>          CDeviceOcaRecordsPtr;
typedef CRefPtr<CContextOcaRecords>         CContextOcaRecordsPtr;
typedef CRefPtr<CChannelOcaRecords>         CChannelOcaRecordsPtr;
typedef CRefPtr<CAllocationOcaRecords>      CAllocationOcaRecordsPtr;
typedef CRefPtr<CKmdProcessOcaRecords>      CKmdProcessOcaRecordsPtr;
typedef CRefPtr<CDmaBufferOcaRecords>       CDmaBufferOcaRecordsPtr;

typedef CArrayPtr<CLwcdRecordPtr>           CLwcdRecordArray;
typedef CArrayPtr<CIsrDpcOcaDataPtr>        CIsrDpcOcaDataArray;
typedef CArrayPtr<CBufferInfoRecordPtr>     CBufferInfoRecordArray;
typedef CArrayPtr<CGpuWatchdogEventPtr>     CGpuWatchdogEventArray;
typedef CArrayPtr<CVblankInfoDataPtr>       CVblankInfoDataArray;
typedef CArrayPtr<CErrorInfoDataPtr>        CErrorInfoDataArray;
typedef CArrayPtr<CWarningInfoDataPtr>      CWarningInfoDataArray;
typedef CArrayPtr<CPagingInfoDataPtr>       CPagingInfoDataArray;

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OCATYPES_H
