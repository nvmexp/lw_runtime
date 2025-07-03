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
|*  Module: ocadrv.cpp                                                        *|
|*                                                                            *|
 \****************************************************************************/
#include "ocaprecomp.h"

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Locals
//
//******************************************************************************
// KMD_OCA_FLAGS Type Helpers
CMemberType     CKmdOcaFlags::m_kmdOcaFlagsType                     (&kmDriver(), "KMD_OCA_RECORD::<unnamed-type-Flags>");

// KMD_OCA_FLAGS Field Helpers
CMemberField    CKmdOcaFlags::m_IsHybridField                       (&kmdOcaFlagsType(), true, NULL, "IsHybrid");
CMemberField    CKmdOcaFlags::m_ValueField                          (&kmdOcaFlagsType(), true, NULL, "Value");

// KMD_OCA_RECORD Type Helpers
CMemberType     CKmdOcaRecord::m_kmdOcaRecordType                   (&kmDriver(), "KMD_OCA_RECORD");

// KMD_OCA_RECORD Field Helpers
CMemberField    CKmdOcaRecord::m_HeaderField                        (&kmdOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CKmdOcaRecord::m_AdapterCountField                  (&kmdOcaRecordType(), true, NULL, "AdapterCount");
CMemberField    CKmdOcaRecord::m_SessionIdentifierField             (&kmdOcaRecordType(), true, NULL, "SessionIdentifier");
CMemberField    CKmdOcaRecord::m_FlagsField                         (&kmdOcaRecordType(), true, NULL, "Flags");

// HYBRID_CONTROL_OCA_FLAGS Type Helpers
CMemberType     CHybridControlOcaFlags::m_hybridControlOcaFlagsType (&kmDriver(), "_HYBRID_CONTROL_OCA_RECORD::<unnamed-type-Flags>");

// HYBRID_CONTROL_OCA_FLAGS Field Helpers
CMemberField    CHybridControlOcaFlags::m_HideModeSetField          (&hybridControlOcaFlagsType(), true, NULL, "HideModeSet");
CMemberField    CHybridControlOcaFlags::m_bEnableHybridModeField    (&hybridControlOcaFlagsType(), true, NULL, "bEnableHybridMode");
CMemberField    CHybridControlOcaFlags::m_bEnableGoldOnHybridField  (&hybridControlOcaFlagsType(), true, NULL, "bEnableGoldOnHybrid");
CMemberField    CHybridControlOcaFlags::m_bEnableHybridPerfSLIField (&hybridControlOcaFlagsType(), true, NULL, "bEnableHybridPerfSLI");
CMemberField    CHybridControlOcaFlags::m_bIntelHybridField         (&hybridControlOcaFlagsType(), true, NULL, "bIntelHybrid");
CMemberField    CHybridControlOcaFlags::m_bSvcStartedField          (&hybridControlOcaFlagsType(), true, NULL, "bSvcStarted");
CMemberField    CHybridControlOcaFlags::m_bM2DSkipFlipUntilSecondBufferField(&hybridControlOcaFlagsType(), true, NULL, "bM2DSkipFlipUntilSecondBuffer");
CMemberField    CHybridControlOcaFlags::m_bDWMOnField               (&hybridControlOcaFlagsType(), true, NULL, "bDWMOn");
CMemberField    CHybridControlOcaFlags::m_ValueField                (&hybridControlOcaFlagsType(), true, NULL, "Value");

// HYBRID_CONTROL_OCA_RECORD Type Helpers
CMemberType     CHybridControlOcaRecord::m_hybridControlOcaRecordType (&kmDriver(), "HYBRID_CONTROL_OCA_RECORD");

// HYBRID_CONTROL_OCA_RECORD Field Helpers
CMemberField    CHybridControlOcaRecord::m_HeaderField              (&hybridControlOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CHybridControlOcaRecord::m_HybridStateField         (&hybridControlOcaRecordType(), true, NULL, "HybridState");
CMemberField    CHybridControlOcaRecord::m_ulApprovalStatusField    (&hybridControlOcaRecordType(), true, NULL, "ulApprovalStatus");
CMemberField    CHybridControlOcaRecord::m_ulMGpuIdField            (&hybridControlOcaRecordType(), true, NULL, "ulMGpuId");
CMemberField    CHybridControlOcaRecord::m_ulDGpuIdField            (&hybridControlOcaRecordType(), true, NULL, "ulDGpuId");
CMemberField    CHybridControlOcaRecord::m_ulSkipM2DRenderField     (&hybridControlOcaRecordType(), true, NULL, "ulSkipM2DRender");
CMemberField    CHybridControlOcaRecord::m_ulD2MSkipFlipsField      (&hybridControlOcaRecordType(), true, NULL, "ulD2MSkipFlips");
CMemberField    CHybridControlOcaRecord::m_FlagsField               (&hybridControlOcaRecordType(), true, &CHybridControlOcaFlags::hybridControlOcaFlagsType(), "Flags");

// ISR_DPC_OCA_DATA Type Helpers
CMemberType     CIsrDpcOcaData::m_isrDpcOcaDataType                 (&kmDriver(), "_ISR_DPC_OCA_RECORD::<unnamed-type-data>");

// ISR_DPC_OCA_DATA Field Helpers
CMemberField    CIsrDpcOcaData::m_TypeField                         (&isrDpcOcaDataType(), true, NULL, "type");
CMemberField    CIsrDpcOcaData::m_durationField                     (&isrDpcOcaDataType(), true, NULL, "duration");
CMemberField    CIsrDpcOcaData::m_timestampField                    (&isrDpcOcaDataType(), true, NULL, "timestamp");

// ISR_DPC_OCA_RECORD Type Helpers
CMemberType     CIsrDpcOcaRecord::m_isrDpcOcaRecordType             (&kmDriver(), "ISR_DPC_OCA_RECORD");

// ISR_DPC_OCA_RECORD Field Helpers
CMemberField    CIsrDpcOcaRecord::m_HeaderField                     (&isrDpcOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CIsrDpcOcaRecord::m_AdapterField                    (&isrDpcOcaRecordType(), true, NULL, "Adapter");
CMemberField    CIsrDpcOcaRecord::m_frequencyField                  (&isrDpcOcaRecordType(), true, NULL, "frequency");
CMemberField    CIsrDpcOcaRecord::m_countField                      (&isrDpcOcaRecordType(), true, NULL, "count");
CMemberField    CIsrDpcOcaRecord::m_dataField                       (&isrDpcOcaRecordType(), true, NULL, "data");

// ADAPTER_OCA_FLAGS Type Helpers
CMemberType     CAdapterOcaFlags::m_adapterOcaFlagsType             (&kmDriver(), "_ADAPTER_OCA_RECORD::<unnamed-type-Flags>");

// ADAPTER_OCA_FLAGS Field Helpers
CMemberField    CAdapterOcaFlags::m_InTDRField                      (&adapterOcaFlagsType(), true, NULL, "InTDR");
CMemberField    CAdapterOcaFlags::m_IsSLIField                      (&adapterOcaFlagsType(), true, NULL, "IsSLI");
CMemberField    CAdapterOcaFlags::m_ValueField                      (&adapterOcaFlagsType(), true, NULL, "Value");

// ADAPTER_OCA_RECORD Type Helpers
CMemberType     CAdapterOcaRecord::m_adapterOcaRecordType           (&kmDriver(), "ADAPTER_OCA_RECORD");

// ADAPTER_OCA_RECORD Field Helpers
CMemberField    CAdapterOcaRecord::m_HeaderField                    (&adapterOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CAdapterOcaRecord::m_AdapterField                   (&adapterOcaRecordType(), true, NULL, "Adapter");
CMemberField    CAdapterOcaRecord::m_SemaphoreMemoryField           (&adapterOcaRecordType(), true, NULL, "SemaphoreMemory");
CMemberField    CAdapterOcaRecord::m_hClientField                   (&adapterOcaRecordType(), true, NULL, "hClient");
CMemberField    CAdapterOcaRecord::m_hDeviceField                   (&adapterOcaRecordType(), true, NULL, "hDevice");
CMemberField    CAdapterOcaRecord::m_DeviceCountField               (&adapterOcaRecordType(), true, NULL, "DeviceCount");
CMemberField    CAdapterOcaRecord::m_ChannelCountField              (&adapterOcaRecordType(), true, NULL, "ChannelCount");
CMemberField    CAdapterOcaRecord::m_NumOfIDSRecordsField           (&adapterOcaRecordType(), true, NULL, "NumOfIDSRecords");
CMemberField    CAdapterOcaRecord::m_IlwalidSLTOffsetCountField     (&adapterOcaRecordType(), true, NULL, "IlwalidSLTOffsetCount");
CMemberField    CAdapterOcaRecord::m_TDRCountField                  (&adapterOcaRecordType(), true, NULL, "TDRCount");
CMemberField    CAdapterOcaRecord::m_LateBufferCompletionCountField (&adapterOcaRecordType(), true, NULL, "LateBufferCompletionCount");
CMemberField    CAdapterOcaRecord::m_BufferSubmissionErrorCountField(&adapterOcaRecordType(), true, NULL, "BufferSubmissionErrorCount");
CMemberField    CAdapterOcaRecord::m_EngineExelwtionMaskField       (&adapterOcaRecordType(), true, NULL, "EngineExelwtionMask");
CMemberField    CAdapterOcaRecord::m_EnginePreemptionMaskField      (&adapterOcaRecordType(), true, NULL, "EnginePreemptionMask");
CMemberField    CAdapterOcaRecord::m_LastPerformanceCounterFrequencyField(&adapterOcaRecordType(), true, NULL, "LastPerformanceCounterFrequency");
CMemberField    CAdapterOcaRecord::m_ConnectedDevicesMaskField      (&adapterOcaRecordType(), true, NULL, "ConnectedDevicesMask");
CMemberField    CAdapterOcaRecord::m_DeviceLoadStateField           (&adapterOcaRecordType(), true, NULL, "DeviceLoadState");
CMemberField    CAdapterOcaRecord::m_ulArchitectureField            (&adapterOcaRecordType(), true, NULL, "ulArchitecture");
CMemberField    CAdapterOcaRecord::m_ulImplementationField          (&adapterOcaRecordType(), true, NULL, "ulImplementation");
CMemberField    CAdapterOcaRecord::m_dwRevisionField                (&adapterOcaRecordType(), true, NULL, "dwRevision");
CMemberField    CAdapterOcaRecord::m_subRevisionField               (&adapterOcaRecordType(), true, NULL, "subRevision");
CMemberField    CAdapterOcaRecord::m_SBIOSPowerUpRetryField         (&adapterOcaRecordType(), true, NULL, "SBIOSPowerUpRetry");
CMemberField    CAdapterOcaRecord::m_FlagsField                     (&adapterOcaRecordType(), true, NULL, "Flags");

// BUFFER_INFO_RECORD Type Helpers
CMemberType     CBufferInfoRecord::m_bufferInfoRecordType           (&kmDriver(), "BUFFER_INFO_RECORD");

// BUFFER_INFO_RECORD Field Helpers
CMemberField    CBufferInfoRecord::m_ContextField                   (&bufferInfoRecordType(), true, NULL, "Context");
CMemberField    CBufferInfoRecord::m_SubmitIdField                  (&bufferInfoRecordType(), true, NULL, "SubmitId");
CMemberField    CBufferInfoRecord::m_BufferIdField                  (&bufferInfoRecordType(), true, NULL, "BufferId");
CMemberField    CBufferInfoRecord::m_FenceIdField                   (&bufferInfoRecordType(), true, NULL, "FenceId");
CMemberField    CBufferInfoRecord::m_SizeField                      (&bufferInfoRecordType(), true, NULL, "Size");
CMemberField    CBufferInfoRecord::m_lwoOffsetField                 (&bufferInfoRecordType(), true, NULL, "lwoOffset");
CMemberField    CBufferInfoRecord::m_IntCountField                  (&bufferInfoRecordType(), true, NULL, "IntCount");
CMemberField    CBufferInfoRecord::m_TypeField                      (&bufferInfoRecordType(), true, NULL, "Type");

// ENGINEID_OCA_RECORD Type Helpers
CMemberType     CEngineIdOcaRecord::m_engineIdOcaRecordType         (&kmDriver(), "ENGINEID_OCA_RECORD");

// ENGINEID_OCA_RECORD Enum Helpers
CEnum           CEngineIdOcaRecord::m_nodeTypeEnum                  (&kmDriver(), "_NodeType", "NodeType");

// ENGINEID_OCA_RECORD Field Helpers
CMemberField    CEngineIdOcaRecord::m_HeaderField                   (&engineIdOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CEngineIdOcaRecord::m_BufferIdField                 (&engineIdOcaRecordType(), true, NULL, "BufferId");
CMemberField    CEngineIdOcaRecord::m_ISRBufferIdField              (&engineIdOcaRecordType(), true, NULL, "ISRBufferId");
CMemberField    CEngineIdOcaRecord::m_DPCBufferIdField              (&engineIdOcaRecordType(), true, NULL, "DPCBufferId");
CMemberField    CEngineIdOcaRecord::m_FenceIdField                  (&engineIdOcaRecordType(), true, NULL, "FenceId");
CMemberField    CEngineIdOcaRecord::m_ISRFenceIdField               (&engineIdOcaRecordType(), true, NULL, "ISRFenceId");
CMemberField    CEngineIdOcaRecord::m_DPCFenceIdField               (&engineIdOcaRecordType(), true, NULL, "DPCFenceId");
CMemberField    CEngineIdOcaRecord::m_PremptionField                (&engineIdOcaRecordType(), true, NULL, "Premption");
CMemberField    CEngineIdOcaRecord::m_PreemptionIdField             (&engineIdOcaRecordType(), true, NULL, "PreemptionId");
CMemberField    CEngineIdOcaRecord::m_ISRPreemptionIdField          (&engineIdOcaRecordType(), true, NULL, "ISRPreemptionId");
CMemberField    CEngineIdOcaRecord::m_SubmitIdField                 (&engineIdOcaRecordType(), true, NULL, "SubmitId");
CMemberField    CEngineIdOcaRecord::m_LastObservedProgressField     (&engineIdOcaRecordType(), true, NULL, "LastObservedProgress");
CMemberField    CEngineIdOcaRecord::m_LastObservedStallField        (&engineIdOcaRecordType(), true, NULL, "LastObservedStall");
CMemberField    CEngineIdOcaRecord::m_NodeOrdinalField              (&engineIdOcaRecordType(), true, NULL, "NodeOrdinal");
CMemberField    CEngineIdOcaRecord::m_NodeTypeField                 (&engineIdOcaRecordType(), true, NULL, "NodeType");
CMemberField    CEngineIdOcaRecord::m_NodeClassTypeField            (&engineIdOcaRecordType(), true, NULL, "NodeClassType");
CMemberField    CEngineIdOcaRecord::m_EngineOrdinalField            (&engineIdOcaRecordType(), true, NULL, "EngineOrdinal");
CMemberField    CEngineIdOcaRecord::m_MaxBufferExelwtionTimeField   (&engineIdOcaRecordType(), true, NULL, "MaxBufferExelwtionTime");
CMemberField    CEngineIdOcaRecord::m_BuffersField                  (&engineIdOcaRecordType(), true, NULL, "Buffers");

// DMA_BUFFER_OCA_RECORD Type Helpers
CMemberType     CDmaBufferOcaRecord::m_dmaBufferOcaRecordType       (&kmDriver(), "DMA_BUFFER_OCA_RECORD");

// DMA_BUFFER_OCA_RECORD Field Helpers
CMemberField    CDmaBufferOcaRecord::m_HeaderField                  (&dmaBufferOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CDmaBufferOcaRecord::m_ContextField                 (&dmaBufferOcaRecordType(), true, NULL, "Context");
CMemberField    CDmaBufferOcaRecord::m_SubmitIdField                (&dmaBufferOcaRecordType(), true, NULL, "SubmitId");
CMemberField    CDmaBufferOcaRecord::m_BufferIdField                (&dmaBufferOcaRecordType(), true, NULL, "BufferId");
CMemberField    CDmaBufferOcaRecord::m_FenceIdField                 (&dmaBufferOcaRecordType(), true, NULL, "FenceId");
CMemberField    CDmaBufferOcaRecord::m_SizeField                    (&dmaBufferOcaRecordType(), true, NULL, "Size");
CMemberField    CDmaBufferOcaRecord::m_lwoOffsetField               (&dmaBufferOcaRecordType(), true, NULL, "lwoOffset");
CMemberField    CDmaBufferOcaRecord::m_IntCountField                (&dmaBufferOcaRecordType(), true, NULL, "IntCount");
CMemberField    CDmaBufferOcaRecord::m_TypeField                    (&dmaBufferOcaRecordType(), true, NULL, "Type");

// GPUWATCHDOG_EVENT Type Helpers
CMemberType     CGpuWatchdogEvent::m_gpuWatchdogEventType           (&kmDriver(), "GPUWATCHDOG_EVENT");

// GPUWATCHDOG_EVENT Field Helpers
CMemberField    CGpuWatchdogEvent::m_AdapterOrdinalField            (&gpuWatchdogEventType(), true, NULL, "AdapterOrdinal");
CMemberField    CGpuWatchdogEvent::m_NodeOrdinalField               (&gpuWatchdogEventType(), true, NULL, "NodeOrdinal");
CMemberField    CGpuWatchdogEvent::m_EngineOrdinalField             (&gpuWatchdogEventType(), true, NULL, "EngineOrdinal");
CMemberField    CGpuWatchdogEvent::m_ExelwtingField                 (&gpuWatchdogEventType(), true, NULL, "Exelwting");
CMemberField    CGpuWatchdogEvent::m_BufferExelwtionTimeField       (&gpuWatchdogEventType(), true, NULL, "BufferExelwtionTime");
CMemberField    CGpuWatchdogEvent::m_LwrrentTimeField               (&gpuWatchdogEventType(), true, NULL, "LwrrentTime");
CMemberField    CGpuWatchdogEvent::m_FenceIdField                   (&gpuWatchdogEventType(), true, NULL, "FenceId");
CMemberField    CGpuWatchdogEvent::m_PBgetField                     (&gpuWatchdogEventType(), true, NULL, "PBget");
CMemberField    CGpuWatchdogEvent::m_C1getField                     (&gpuWatchdogEventType(), true, NULL, "C1get");
CMemberField    CGpuWatchdogEvent::m_PixelsField                    (&gpuWatchdogEventType(), true, NULL, "Pixels");
CMemberField    CGpuWatchdogEvent::m_StatusField                    (&gpuWatchdogEventType(), true, NULL, "Status");

// CKmdRingBuffer Type Helpers
CMemberType     CKmdRingBuffer::m_kmdRingBufferType                 (&kmDriver(), "CRingBuffer<_GPUWATCHDOG_EVENT,32>");

// CKmdRingBuffer Field Helpers
CMemberField    CKmdRingBuffer::m_ulIndexField                      (&kmdRingBufferType(), false, NULL, "m_ulIndex");
CMemberField    CKmdRingBuffer::m_ulCountField                      (&kmdRingBufferType(), false, NULL, "m_ulCount");
CMemberField    CKmdRingBuffer::m_ElementsField                     (&kmdRingBufferType(), false, NULL, "m_Elements");

// KMD_RING_BUFFER_OCA_RECORD Type Helpers
CMemberType     CKmdRingBufferOcaRecord::m_kmdRingBufferOcaRecordType(&kmDriver(), "LWCD_RECORD");

// ALLOCATION_OCA_RESOURCE Type Helpers
CMemberType     CAllocationOcaResource::m_allocationOcaResourceType (&kmDriver(), "_ALLOCATION_OCA_RECORD::<unnamed-type-Resource>");

// ALLOCATION_OCA_RESOURCE Field Helpers
CMemberField    CAllocationOcaResource::m_TypeField                 (&allocationOcaResourceType(), true, NULL, "Type");
CMemberField    CAllocationOcaResource::m_FormatField               (&allocationOcaResourceType(), true, NULL, "Format");
CMemberField    CAllocationOcaResource::m_WidthField                (&allocationOcaResourceType(), true, NULL, "Width");
CMemberField    CAllocationOcaResource::m_HeightField               (&allocationOcaResourceType(), true, NULL, "Height");
CMemberField    CAllocationOcaResource::m_DepthField                (&allocationOcaResourceType(), true, NULL, "Depth");
CMemberField    CAllocationOcaResource::m_MipMapCountField          (&allocationOcaResourceType(), true, NULL, "MipMapCount");
CMemberField    CAllocationOcaResource::m_VidPnSourceIdField        (&allocationOcaResourceType(), true, NULL, "VidPnSourceId");

// ALLOCATION_OCA_RECORD Type Helpers
CMemberType     CAllocationOcaRecord::m_allocationOcaRecordType     (&kmDriver(), "ALLOCATION_OCA_RECORD");

// ALLOCATION_OCA_RECORD Field Helpers
CMemberField    CAllocationOcaRecord::m_HeaderField                 (&allocationOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CAllocationOcaRecord::m_PitchField                  (&allocationOcaRecordType(), true, NULL, "Pitch");
CMemberField    CAllocationOcaRecord::m_HeightField                 (&allocationOcaRecordType(), true, NULL, "Height");
CMemberField    CAllocationOcaRecord::m_BppField                    (&allocationOcaRecordType(), true, NULL, "Bpp");
CMemberField    CAllocationOcaRecord::m_AccessibleSizeField         (&allocationOcaRecordType(), true, NULL, "AccessibleSize");
CMemberField    CAllocationOcaRecord::m_TotalSizeField              (&allocationOcaRecordType(), true, NULL, "TotalSize");
CMemberField    CAllocationOcaRecord::m_AllowedHeapsField           (&allocationOcaRecordType(), true, NULL, "AllowedHeaps");
CMemberField    CAllocationOcaRecord::m_PreferredHeapField          (&allocationOcaRecordType(), true, NULL, "PreferredHeap");
CMemberField    CAllocationOcaRecord::m_SegmentField                (&allocationOcaRecordType(), true, NULL, "Segment");
CMemberField    CAllocationOcaRecord::m_OffsetField                 (&allocationOcaRecordType(), true, NULL, "Offset");
CMemberField    CAllocationOcaRecord::m_ResourceField               (&allocationOcaRecordType(), true, NULL, "Resource");

// KMDPROCESS_OCA_RECORD Type Helpers
CMemberType     CKmdProcessOcaRecord::m_kmdProcessOcaRecordType     (&kmDriver(), "KMDPROCESS_OCA_RECORD");

// KMDPROCESS_OCA_RECORD Field Helpers
CMemberField    CKmdProcessOcaRecord::m_HeaderField                 (&kmdProcessOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CKmdProcessOcaRecord::m_hClientField                (&kmdProcessOcaRecordType(), true, NULL, "hClient");
CMemberField    CKmdProcessOcaRecord::m_hDeviceField                (&kmdProcessOcaRecordType(), true, NULL, "hDevice");
CMemberField    CKmdProcessOcaRecord::m_KmdProcessField             (&kmdProcessOcaRecordType(), true, NULL, "KmdProcess");
CMemberField    CKmdProcessOcaRecord::m_DeviceField                 (&kmdProcessOcaRecordType(), true, NULL, "Device");
CMemberField    CKmdProcessOcaRecord::m_AdapterField                (&kmdProcessOcaRecordType(), true, NULL, "Adapter");
CMemberField    CKmdProcessOcaRecord::m_ProcessField                (&kmdProcessOcaRecordType(), true, NULL, "Process");
CMemberField    CKmdProcessOcaRecord::m_ProcessImageNameField       (&kmdProcessOcaRecordType(), true, NULL, "ProcessImageName");
CMemberField    CKmdProcessOcaRecord::m_DeviceCountField            (&kmdProcessOcaRecordType(), true, NULL, "DeviceCount");

// DEVICE_OCA_RECORD Type Helpers
CMemberType     CDeviceOcaRecord::m_deviceOcaRecordType             (&kmDriver(), "DEVICE_OCA_RECORD");

// DEVICE_OCA_RECORD Field Helpers
CMemberField    CDeviceOcaRecord::m_HeaderField                     (&deviceOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CDeviceOcaRecord::m_hClientField                    (&deviceOcaRecordType(), true, NULL, "hClient");
CMemberField    CDeviceOcaRecord::m_hDeviceField                    (&deviceOcaRecordType(), true, NULL, "hDevice");
CMemberField    CDeviceOcaRecord::m_KmdProcessField                 (&deviceOcaRecordType(), true, NULL, "KmdProcess");
CMemberField    CDeviceOcaRecord::m_DeviceField                     (&deviceOcaRecordType(), true, NULL, "Device");
CMemberField    CDeviceOcaRecord::m_AdapterField                    (&deviceOcaRecordType(), true, NULL, "Adapter");
CMemberField    CDeviceOcaRecord::m_ProcessField                    (&deviceOcaRecordType(), true, NULL, "Process");
CMemberField    CDeviceOcaRecord::m_ContextCountField               (&deviceOcaRecordType(), true, NULL, "ContextCount");
CMemberField    CDeviceOcaRecord::m_ReferenceCountField             (&deviceOcaRecordType(), true, NULL, "ReferenceCount");
CMemberField    CDeviceOcaRecord::m_bLockedPerfMonField             (&deviceOcaRecordType(), true, NULL, "bLockedPerfMon");
CMemberField    CDeviceOcaRecord::m_dmaBufferSizeField              (&deviceOcaRecordType(), true, NULL, "dmaBufferSize");

// CONTEXT_OCA_RECORD Type Helpers
CMemberType     CContextOcaRecord::m_contextOcaRecordType           (&kmDriver(), "CONTEXT_OCA_RECORD");

// CONTEXT_OCA_RECORD Field Helpers
CMemberField    CContextOcaRecord::m_HeaderField                    (&contextOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CContextOcaRecord::m_ContextField                   (&contextOcaRecordType(), true, NULL, "Context");
CMemberField    CContextOcaRecord::m_ChannelField                   (&contextOcaRecordType(), true, NULL, "Channel");
CMemberField    CContextOcaRecord::m_AdapterField                   (&contextOcaRecordType(), true, NULL, "Adapter");
CMemberField    CContextOcaRecord::m_DeviceField                    (&contextOcaRecordType(), true, NULL, "Device");
CMemberField    CContextOcaRecord::m_NodeOrdinalField               (&contextOcaRecordType(), true, NULL, "NodeOrdinal");
CMemberField    CContextOcaRecord::m_EngineOrdinalField             (&contextOcaRecordType(), true, NULL, "EngineOrdinal");
CMemberField    CContextOcaRecord::m_BufferIdField                  (&contextOcaRecordType(), true, NULL, "BufferId");

// CHANNEL_OCA_RECORD Type Helpers
CMemberType     CChannelOcaRecord::m_channelOcaRecordType           (&kmDriver(), "CHANNEL_OCA_RECORD");

// CHANNEL_OCA_RECORD Field Helpers
CMemberField    CChannelOcaRecord::m_HeaderField                    (&channelOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CChannelOcaRecord::m_ChannelField                   (&channelOcaRecordType(), true, NULL, "Channel");
CMemberField    CChannelOcaRecord::m_HwChannelIndexField            (&channelOcaRecordType(), true, NULL, "HwChannelIndex");
CMemberField    CChannelOcaRecord::m_hClientField                   (&channelOcaRecordType(), true, NULL, "hClient");
CMemberField    CChannelOcaRecord::m_hDeviceField                   (&channelOcaRecordType(), true, NULL, "hDevice");
CMemberField    CChannelOcaRecord::m_DmaCountField                  (&channelOcaRecordType(), true, NULL, "DmaCount");
CMemberField    CChannelOcaRecord::m_NodeOrdinalField               (&channelOcaRecordType(), true, NULL, "NodeOrdinal");
CMemberField    CChannelOcaRecord::m_bSharedField                   (&channelOcaRecordType(), true, NULL, "bShared");
CMemberField    CChannelOcaRecord::m_bReservedField                 (&channelOcaRecordType(), true, NULL, "bReserved");
CMemberField    CChannelOcaRecord::m_ContextCountField              (&channelOcaRecordType(), true, NULL, "ContextCount");

// DISPLAY_TARGET_OCA_RECORD Type Helpers
CMemberType     CDisplayTargetOcaRecord::m_displayTargetOcaRecordType(&kmDriver(), "DISPLAY_TARGET_OCA_RECORD");

// Display target Enum Helpers
CEnum           CDisplayTargetOcaRecord::m_lwlScalingEnum           (&kmDriver(), "LWL_SCALING", "_LWL_SCALING");

// DISPLAY_TARGET_OCA_RECORD Field Helpers
CMemberField    CDisplayTargetOcaRecord::m_HeaderField              (&displayTargetOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CDisplayTargetOcaRecord::m_VidPnTargetIdField       (&displayTargetOcaRecordType(), true, NULL, "VidPnTargetId");
CMemberField    CDisplayTargetOcaRecord::m_headField                (&displayTargetOcaRecordType(), true, NULL, "head");
CMemberField    CDisplayTargetOcaRecord::m_deviceField              (&displayTargetOcaRecordType(), true, NULL, "device");
CMemberField    CDisplayTargetOcaRecord::m_connectorField           (&displayTargetOcaRecordType(), true, NULL, "connector");
CMemberField    CDisplayTargetOcaRecord::m_srcIDField               (&displayTargetOcaRecordType(), true, NULL, "srcID");
CMemberField    CDisplayTargetOcaRecord::m_srcImportanceField       (&displayTargetOcaRecordType(), true, NULL, "srcImportance");
CMemberField    CDisplayTargetOcaRecord::m_hAllocationField         (&displayTargetOcaRecordType(), true, NULL, "hAllocation");
CMemberField    CDisplayTargetOcaRecord::m_AddressField             (&displayTargetOcaRecordType(), true, NULL, "Address");
CMemberField    CDisplayTargetOcaRecord::m_bFlipPendingField        (&displayTargetOcaRecordType(), true, NULL, "bFlipPending");
CMemberField    CDisplayTargetOcaRecord::m_flipPendingAddressField  (&displayTargetOcaRecordType(), true, NULL, "flipPendingAddress");
CMemberField    CDisplayTargetOcaRecord::m_widthField               (&displayTargetOcaRecordType(), true, NULL, "width");
CMemberField    CDisplayTargetOcaRecord::m_heightField              (&displayTargetOcaRecordType(), true, NULL, "height");
CMemberField    CDisplayTargetOcaRecord::m_depthField               (&displayTargetOcaRecordType(), true, NULL, "depth");
CMemberField    CDisplayTargetOcaRecord::m_refreshRateField         (&displayTargetOcaRecordType(), true, NULL, "refreshRate");
CMemberField    CDisplayTargetOcaRecord::m_colorFormatField         (&displayTargetOcaRecordType(), true, NULL, "colorFormat");
CMemberField    CDisplayTargetOcaRecord::m_rotationField            (&displayTargetOcaRecordType(), true, NULL, "rotation");
CMemberField    CDisplayTargetOcaRecord::m_callFromTMMField         (&displayTargetOcaRecordType(), true, NULL, "callFromTMM");
CMemberField    CDisplayTargetOcaRecord::m_SelectLwstomTimingField  (&displayTargetOcaRecordType(), true, NULL, "SelectLwstomTiming");
CMemberField    CDisplayTargetOcaRecord::m_tvFormatField            (&displayTargetOcaRecordType(), true, NULL, "tvFormat");
CMemberField    CDisplayTargetOcaRecord::m_srcPartitionXField       (&displayTargetOcaRecordType(), true, NULL, "srcPartitionX");
CMemberField    CDisplayTargetOcaRecord::m_srcPartitionYField       (&displayTargetOcaRecordType(), true, NULL, "srcPartitionY");
CMemberField    CDisplayTargetOcaRecord::m_srcPartitionWField       (&displayTargetOcaRecordType(), true, NULL, "srcPartitionW");
CMemberField    CDisplayTargetOcaRecord::m_srcPartitionHField       (&displayTargetOcaRecordType(), true, NULL, "srcPartitionH");
CMemberField    CDisplayTargetOcaRecord::m_viewportInXField         (&displayTargetOcaRecordType(), true, NULL, "viewportInX");
CMemberField    CDisplayTargetOcaRecord::m_viewportInYField         (&displayTargetOcaRecordType(), true, NULL, "viewportInY");
CMemberField    CDisplayTargetOcaRecord::m_viewportInWField         (&displayTargetOcaRecordType(), true, NULL, "viewportInW");
CMemberField    CDisplayTargetOcaRecord::m_viewportInHField         (&displayTargetOcaRecordType(), true, NULL, "viewportInH");
CMemberField    CDisplayTargetOcaRecord::m_scalingMethodField       (&displayTargetOcaRecordType(), true, NULL, "scalingMethod");
CMemberField    CDisplayTargetOcaRecord::m_viewportOutXField        (&displayTargetOcaRecordType(), true, NULL, "viewportOutX");
CMemberField    CDisplayTargetOcaRecord::m_viewportOutYField        (&displayTargetOcaRecordType(), true, NULL, "viewportOutY");
CMemberField    CDisplayTargetOcaRecord::m_viewportOutWField        (&displayTargetOcaRecordType(), true, NULL, "viewportOutW");
CMemberField    CDisplayTargetOcaRecord::m_viewportOutHField        (&displayTargetOcaRecordType(), true, NULL, "viewportOutH");
CMemberField    CDisplayTargetOcaRecord::m_timingOverrideField      (&displayTargetOcaRecordType(), true, NULL, "timingOverride");
CMemberField    CDisplayTargetOcaRecord::m_bVsyncEnabledField       (&displayTargetOcaRecordType(), true, NULL, "bVsyncEnabled");
CMemberField    CDisplayTargetOcaRecord::m_HVisibleField            (&displayTargetOcaRecordType(), true, NULL, "HVisible");
CMemberField    CDisplayTargetOcaRecord::m_HBorderField             (&displayTargetOcaRecordType(), true, NULL, "HBorder");
CMemberField    CDisplayTargetOcaRecord::m_HFrontPorchField         (&displayTargetOcaRecordType(), true, NULL, "HFrontPorch");
CMemberField    CDisplayTargetOcaRecord::m_HSyncWidthField          (&displayTargetOcaRecordType(), true, NULL, "HSyncWidth");
CMemberField    CDisplayTargetOcaRecord::m_HTotalField              (&displayTargetOcaRecordType(), true, NULL, "HTotal");
CMemberField    CDisplayTargetOcaRecord::m_HSyncPolField            (&displayTargetOcaRecordType(), true, NULL, "HSyncPol");
CMemberField    CDisplayTargetOcaRecord::m_VVisibleField            (&displayTargetOcaRecordType(), true, NULL, "VVisible");
CMemberField    CDisplayTargetOcaRecord::m_VBorderField             (&displayTargetOcaRecordType(), true, NULL, "VBorder");
CMemberField    CDisplayTargetOcaRecord::m_VFrontPorchField         (&displayTargetOcaRecordType(), true, NULL, "VFrontPorch");
CMemberField    CDisplayTargetOcaRecord::m_VSyncWidthField          (&displayTargetOcaRecordType(), true, NULL, "VSyncWidth");
CMemberField    CDisplayTargetOcaRecord::m_VTotalField              (&displayTargetOcaRecordType(), true, NULL, "VTotal");
CMemberField    CDisplayTargetOcaRecord::m_VSyncPolField            (&displayTargetOcaRecordType(), true, NULL, "VSyncPol");
CMemberField    CDisplayTargetOcaRecord::m_interlacedField          (&displayTargetOcaRecordType(), true, NULL, "interlaced");
CMemberField    CDisplayTargetOcaRecord::m_pclkField                (&displayTargetOcaRecordType(), true, NULL, "pclk");
CMemberField    CDisplayTargetOcaRecord::m_flagField                (&displayTargetOcaRecordType(), true, NULL, "flag");
CMemberField    CDisplayTargetOcaRecord::m_rrField                  (&displayTargetOcaRecordType(), true, NULL, "rr");
CMemberField    CDisplayTargetOcaRecord::m_rrx1kField               (&displayTargetOcaRecordType(), true, NULL, "rrx1k");
CMemberField    CDisplayTargetOcaRecord::m_aspectField              (&displayTargetOcaRecordType(), true, NULL, "aspect");
CMemberField    CDisplayTargetOcaRecord::m_repField                 (&displayTargetOcaRecordType(), true, NULL, "rep");
CMemberField    CDisplayTargetOcaRecord::m_statusField              (&displayTargetOcaRecordType(), true, NULL, "status");
CMemberField    CDisplayTargetOcaRecord::m_nameField                (&displayTargetOcaRecordType(), true, NULL, "name");

// MONITOR_INFO_OCA_RECORD Type Helpers
CMemberType     CMonitorInfoOcaRecord::m_monitorInfoOcaRecordType   (&kmDriver(), "MONITOR_INFO_OCA_RECORD");

// MONITOR_INFO_OCA_RECORD Field Helpers
CMemberField    CMonitorInfoOcaRecord::m_HeaderField                (&monitorInfoOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CMonitorInfoOcaRecord::m_dwEDIDSizeField            (&monitorInfoOcaRecordType(), true, NULL, "dwEDIDSize");
CMemberField    CMonitorInfoOcaRecord::m_EDIDField                  (&monitorInfoOcaRecordType(), true, NULL, "EDID");

// VBLANK_INFO_DATA Type Helpers
CMemberType     CVblankInfoData::m_vblankInfoDataType               (&kmDriver(), "_VBLANK_INFO_OCA_RECORD::<unnamed-type-data>");

// VBLANK_INFO_DATA Field Helpers
CMemberField    CVblankInfoData::m_offsetField                      (&vblankInfoDataType(), true, NULL, "offset");
CMemberField    CVblankInfoData::m_headField                        (&vblankInfoDataType(), true, NULL, "head");
CMemberField    CVblankInfoData::m_VidPnTargetIdField               (&vblankInfoDataType(), true, NULL, "VidPnTargetId");
CMemberField    CVblankInfoData::m_timestampField                   (&vblankInfoDataType(), true, NULL, "timestamp");

// VBLANK_INFO_OCA_RECORD Type Helpers
CMemberType     CVblankInfoOcaRecord::m_vblankInfoOcaRecordType     (&kmDriver(), "VBLANK_INFO_OCA_RECORD");

// VBLANK_INFO_OCA_RECORD Field Helpers
CMemberField    CVblankInfoOcaRecord::m_HeaderField                 (&vblankInfoOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CVblankInfoOcaRecord::m_AdapterField                (&vblankInfoOcaRecordType(), true, NULL, "Adapter");
CMemberField    CVblankInfoOcaRecord::m_frequencyField              (&vblankInfoOcaRecordType(), true, NULL, "frequency");
CMemberField    CVblankInfoOcaRecord::m_countField                  (&vblankInfoOcaRecordType(), true, NULL, "count");
CMemberField    CVblankInfoOcaRecord::m_dataField                   (&vblankInfoOcaRecordType(), true, NULL, "data");

// ERROR_INFO_DATA Type Helpers
CMemberType     CErrorInfoData::m_errorInfoDataType                 (&kmDriver(), "ERROR_INFO_DATA");

// Error Event Enum Helpers
CEnum           CErrorInfoData::m_errorEventEnum                    (&kmDriver(), "LogError");
CEnum           CErrorInfoData::m_dmInterfaceEnum                   (&kmDriver(), "DmInterface");
CEnum           CErrorInfoData::m_lddmInterfaceEnum                 (&kmDriver(), "LddmInterface");
CEnum           CErrorInfoData::m_rmInterfaceEnum                   (&kmDriver(), "RmInterface");

// ERROR_INFO_DATA Field Helpers
CMemberField    CErrorInfoData::m_TypeField                         (&errorInfoDataType(), true, NULL, "type");
CMemberField    CErrorInfoData::m_subTypeField                      (&errorInfoDataType(), true, NULL, "subType");
CMemberField    CErrorInfoData::m_statusField                       (&errorInfoDataType(), true, NULL, "status");
CMemberField    CErrorInfoData::m_addressField                      (&errorInfoDataType(), true, NULL, "address");
CMemberField    CErrorInfoData::m_adapterField                      (&errorInfoDataType(), true, NULL, "adapter");
CMemberField    CErrorInfoData::m_deviceField                       (&errorInfoDataType(), true, NULL, "device");
CMemberField    CErrorInfoData::m_contextField                      (&errorInfoDataType(), true, NULL, "context");
CMemberField    CErrorInfoData::m_channelField                      (&errorInfoDataType(), true, NULL, "channel");
CMemberField    CErrorInfoData::m_allocationField                   (&errorInfoDataType(), true, NULL, "allocation");
CMemberField    CErrorInfoData::m_processField                      (&errorInfoDataType(), true, NULL, "process");
CMemberField    CErrorInfoData::m_threadField                       (&errorInfoDataType(), true, NULL, "thread");
CMemberField    CErrorInfoData::m_data0Field                        (&errorInfoDataType(), true, NULL, "data0");
CMemberField    CErrorInfoData::m_data1Field                        (&errorInfoDataType(), true, NULL, "data1");
CMemberField    CErrorInfoData::m_data2Field                        (&errorInfoDataType(), true, NULL, "data2");
CMemberField    CErrorInfoData::m_data3Field                        (&errorInfoDataType(), true, NULL, "data3");
CMemberField    CErrorInfoData::m_data4Field                        (&errorInfoDataType(), true, NULL, "data4");
CMemberField    CErrorInfoData::m_timestampField                    (&errorInfoDataType(), true, NULL, "timestamp");

// ERROR_INFO_OCA_RECORD Type Helpers
CMemberType     CErrorInfoOcaRecord::m_errorInfoOcaRecordType       (&kmDriver(), "ERROR_INFO_OCA_RECORD");

// ERROR_INFO_OCA_RECORD Field Helpers
CMemberField    CErrorInfoOcaRecord::m_HeaderField                  (&errorInfoOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CErrorInfoOcaRecord::m_frequencyField               (&errorInfoOcaRecordType(), true, NULL, "frequency");
CMemberField    CErrorInfoOcaRecord::m_countField                   (&errorInfoOcaRecordType(), true, NULL, "count");

// WARNING_INFO_DATA Type Helpers
CMemberType     CWarningInfoData::m_warningInfoDataType             (&kmDriver(), "WARNING_INFO_DATA");

// Warning Event Enum Helpers
CEnum           CWarningInfoData::m_warningEventEnum                (&kmDriver(), "LogWarning");
CEnum           CWarningInfoData::m_dmInterfaceEnum                 (&kmDriver(), "DmInterface");
CEnum           CWarningInfoData::m_lddmInterfaceEnum               (&kmDriver(), "LddmInterface");
CEnum           CWarningInfoData::m_rmInterfaceEnum                 (&kmDriver(), "RmInterface");

// WARNING_INFO_DATA Field Helpers
CMemberField    CWarningInfoData::m_TypeField                       (&warningInfoDataType(), true, NULL, "type");
CMemberField    CWarningInfoData::m_subTypeField                    (&warningInfoDataType(), true, NULL, "subType");
CMemberField    CWarningInfoData::m_statusField                     (&warningInfoDataType(), true, NULL, "status");
CMemberField    CWarningInfoData::m_addressField                    (&warningInfoDataType(), true, NULL, "address");
CMemberField    CWarningInfoData::m_adapterField                    (&warningInfoDataType(), true, NULL, "adapter");
CMemberField    CWarningInfoData::m_deviceField                     (&warningInfoDataType(), true, NULL, "device");
CMemberField    CWarningInfoData::m_contextField                    (&warningInfoDataType(), true, NULL, "context");
CMemberField    CWarningInfoData::m_channelField                    (&warningInfoDataType(), true, NULL, "channel");
CMemberField    CWarningInfoData::m_allocationField                 (&warningInfoDataType(), true, NULL, "allocation");
CMemberField    CWarningInfoData::m_processField                    (&warningInfoDataType(), true, NULL, "process");
CMemberField    CWarningInfoData::m_threadField                     (&warningInfoDataType(), true, NULL, "thread");
CMemberField    CWarningInfoData::m_data0Field                      (&warningInfoDataType(), true, NULL, "data0");
CMemberField    CWarningInfoData::m_data1Field                      (&warningInfoDataType(), true, NULL, "data1");
CMemberField    CWarningInfoData::m_data2Field                      (&warningInfoDataType(), true, NULL, "data2");
CMemberField    CWarningInfoData::m_data3Field                      (&warningInfoDataType(), true, NULL, "data3");
CMemberField    CWarningInfoData::m_data4Field                      (&warningInfoDataType(), true, NULL, "data4");
CMemberField    CWarningInfoData::m_timestampField                  (&warningInfoDataType(), true, NULL, "timestamp");

// WARNING_INFO_OCA_RECORD Type Helpers
CMemberType     CWarningInfoOcaRecord::m_warningInfoOcaRecordType   (&kmDriver(), "WARNING_INFO_OCA_RECORD");

// WARNING_INFO_OCA_RECORD Field Helpers
CMemberField    CWarningInfoOcaRecord::m_HeaderField                (&warningInfoOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CWarningInfoOcaRecord::m_frequencyField             (&warningInfoOcaRecordType(), true, NULL, "frequency");
CMemberField    CWarningInfoOcaRecord::m_countField                 (&warningInfoOcaRecordType(), true, NULL, "count");

// PAGING_INFO_DATA Type Helpers
CMemberType     CPagingInfoData::m_pagingInfoDataType               (&kmDriver(), "PAGING_INFO_DATA");

// Paging Event Enum Helpers
CEnum           CPagingInfoData::m_pagingEventEnum                  (&kmDriver(), "LogPaging");

// PAGING_INFO_DATA Field Helpers
CMemberField    CPagingInfoData::m_TypeField                        (&pagingInfoDataType(), true, NULL, "type");
CMemberField    CPagingInfoData::m_subTypeField                     (&pagingInfoDataType(), true, NULL, "subType");
CMemberField    CPagingInfoData::m_statusField                      (&pagingInfoDataType(), true, NULL, "status");
CMemberField    CPagingInfoData::m_addressField                     (&pagingInfoDataType(), true, NULL, "address");
CMemberField    CPagingInfoData::m_adapterField                     (&pagingInfoDataType(), true, NULL, "adapter");
CMemberField    CPagingInfoData::m_deviceField                      (&pagingInfoDataType(), true, NULL, "device");
CMemberField    CPagingInfoData::m_contextField                     (&pagingInfoDataType(), true, NULL, "context");
CMemberField    CPagingInfoData::m_channelField                     (&pagingInfoDataType(), true, NULL, "channel");
CMemberField    CPagingInfoData::m_allocationField                  (&pagingInfoDataType(), true, NULL, "allocation");
CMemberField    CPagingInfoData::m_processField                     (&pagingInfoDataType(), true, NULL, "process");
CMemberField    CPagingInfoData::m_threadField                      (&pagingInfoDataType(), true, NULL, "thread");
CMemberField    CPagingInfoData::m_data0Field                       (&pagingInfoDataType(), true, NULL, "data0");
CMemberField    CPagingInfoData::m_data1Field                       (&pagingInfoDataType(), true, NULL, "data1");
CMemberField    CPagingInfoData::m_data2Field                       (&pagingInfoDataType(), true, NULL, "data2");
CMemberField    CPagingInfoData::m_data3Field                       (&pagingInfoDataType(), true, NULL, "data3");
CMemberField    CPagingInfoData::m_data4Field                       (&pagingInfoDataType(), true, NULL, "data4");
CMemberField    CPagingInfoData::m_timestampField                   (&pagingInfoDataType(), true, NULL, "timestamp");

// PAGING_INFO_OCA_RECORD Type Helpers
CMemberType     CPagingInfoOcaRecord::m_pagingInfoOcaRecordType     (&kmDriver(), "PAGING_INFO_OCA_RECORD");

// PAGING_INFO_OCA_RECORD Field Helpers
CMemberField    CPagingInfoOcaRecord::m_HeaderField                 (&pagingInfoOcaRecordType(), true, &CLwcdRecord::lwcdRecordType(), "Header");
CMemberField    CPagingInfoOcaRecord::m_frequencyField              (&pagingInfoOcaRecordType(), true, NULL, "frequency");
CMemberField    CPagingInfoOcaRecord::m_countField                  (&pagingInfoOcaRecordType(), true, NULL, "count");

//******************************************************************************

CKmdOcaFlags::CKmdOcaFlags
(
    const void         *pKmdOcaFlags
)
:   m_pKmdOcaFlags(pKmdOcaFlags),
    INIT(IsHybrid),
    INIT(Value)
{
    assert(pKmdOcaFlags != NULL);

    // Set the KMD_OCA_FLAGS (from data pointer)
    SET(IsHybrid,   pKmdOcaFlags);
    SET(Value,      pKmdOcaFlags);

} // CKmdOcaFlags

//******************************************************************************

CKmdOcaFlags::~CKmdOcaFlags()
{

} // ~CKmdOcaFlags

//******************************************************************************

CKmdOcaRecord::CKmdOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pKmdOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(AdapterCount),
    INIT(Flags)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the KMD_OCA_RECORD (from data pointer)
    SET(AdapterCount,   m_pKmdOcaRecord);
    SET(Flags,          m_pKmdOcaRecord);

    // Check for partial KMD_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial KMD_OCA_RECORD
        setPartial(true);
    }
    // If Flags field is present create the Flags member
    if (FlagsMember().isPresent())
    {
        // Create the Flags member
        m_pKmdOcaFlags = new CKmdOcaFlags(constcharptr(m_pKmdOcaRecord) + FlagsMember().offset());
    }
    // Check for session identifier member present (Create GUID for session identifier)
    if (SessionIdentifierField().isPresent())
    {
        // Create the GUID for the session identifier
        m_pSessionIdentifier = new COcaGuid(constcharptr(m_pKmdOcaRecord) + SessionIdentifierField().offset());
    }

} // CKmdOcaRecord

//******************************************************************************

CKmdOcaRecord::~CKmdOcaRecord()
{

} // ~CKmdOcaRecord

//******************************************************************************

CHybridControlOcaFlags::CHybridControlOcaFlags
(
    const void         *pHybridControlOcaFlags
)
:   m_pHybridControlOcaFlags(pHybridControlOcaFlags),
    INIT(HideModeSet),
    INIT(bEnableHybridMode),
    INIT(bEnableGoldOnHybrid),
    INIT(bEnableHybridPerfSLI),
    INIT(bIntelHybrid),
    INIT(bSvcStarted),
    INIT(bM2DSkipFlipUntilSecondBuffer),
    INIT(bDWMOn),
    INIT(Value)
{
    assert(pHybridControlOcaFlags != NULL);

    // Set the HYRBID_CONTROL_OCA_FLAGS (from data pointer)
    SET(HideModeSet,                    pHybridControlOcaFlags);
    SET(bEnableHybridMode,              pHybridControlOcaFlags);
    SET(bEnableGoldOnHybrid,            pHybridControlOcaFlags);
    SET(bEnableHybridPerfSLI,           pHybridControlOcaFlags);
    SET(bIntelHybrid,                   pHybridControlOcaFlags);
    SET(bSvcStarted,                    pHybridControlOcaFlags);
    SET(bM2DSkipFlipUntilSecondBuffer,  pHybridControlOcaFlags);
    SET(bDWMOn,                         pHybridControlOcaFlags);
    SET(Value,                          pHybridControlOcaFlags);

} // CHybridControlOcaFlags

//******************************************************************************

CHybridControlOcaFlags::~CHybridControlOcaFlags()
{

} // ~CHybridControlOcaFlags

//******************************************************************************

CHybridControlOcaRecord::CHybridControlOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pHybridControlOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(HybridState),
    INIT(ulApprovalStatus),
    INIT(ulMGpuId),
    INIT(ulDGpuId),
    INIT(ulSkipM2DRender),
    INIT(ulD2MSkipFlips),
    INIT(Flags)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the HYRBID_CONTROL_OCA_RECORD (from data pointer)
    SET(HybridState,        m_pHybridControlOcaRecord);
    SET(ulApprovalStatus,   m_pHybridControlOcaRecord);
    SET(ulMGpuId,           m_pHybridControlOcaRecord);
    SET(ulDGpuId,           m_pHybridControlOcaRecord);
    SET(ulSkipM2DRender,    m_pHybridControlOcaRecord);
    SET(ulD2MSkipFlips,     m_pHybridControlOcaRecord);
    SET(Flags,              m_pHybridControlOcaRecord);

    // Check for partial HYRBID_CONTROL_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial HYRBID_CONTROL_OCA_RECORD
        setPartial(true);
    }
    // If Flags field is present create the Flags member
    if (FlagsMember().isPresent())
    {
        // Create the Flags member
        m_pHybridControlOcaFlags = new CHybridControlOcaFlags(constcharptr(m_pHybridControlOcaRecord) + FlagsMember().offset());
    }

} // CHybridControlOcaRecord

//******************************************************************************

CHybridControlOcaRecord::~CHybridControlOcaRecord()
{

} // ~CHybridControlOcaRecord

//******************************************************************************

CIsrDpcOcaData::CIsrDpcOcaData
(
    const void         *pIsrDpcOcaData
)
:   m_pIsrDpcOcaData(pIsrDpcOcaData),
    INIT(Type),
    INIT(duration),
    INIT(timestamp)
{
    assert(pIsrDpcOcaData != NULL);

    // Set the ISR_DPC_OCA_DATA (from data pointer)
    SET(Type,       pIsrDpcOcaData);
    SET(duration,   pIsrDpcOcaData);
    SET(timestamp,  pIsrDpcOcaData);

} // CKmdOcaData

//******************************************************************************

CIsrDpcOcaData::~CIsrDpcOcaData()
{

} // ~CIsrDpcOcaData

//******************************************************************************

CIsrDpcOcaDatas::CIsrDpcOcaDatas
(
    const CIsrDpcOcaRecord *pIsrDpcOcaRecord,
    ULONG               ulRemaining
)
:   m_pIsrDpcOcaRecord(pIsrDpcOcaRecord),
    m_ulIsrDpcDataCount(0),
    m_aIsrDpcOcaDatas(NULL)
{
    assert(pIsrDpcOcaRecord != NULL);

    // Callwlate the maximum number of ISR/DPC datas (Based on record size and remaining data)
    m_ulIsrDpcDataCount = (min(ulRemaining, pIsrDpcOcaRecord->size()) - pIsrDpcOcaRecord->dataField().offset()) / CIsrDpcOcaData::isrDpcOcaDataType().size();

    // Check for ISR/DPC data count available and valid
    if (pIsrDpcOcaRecord->countMember().isPresent() && (pIsrDpcOcaRecord->count() != 0))
    {
        // Use the number of ISR/DPC OCA datas (If valid)
        m_ulIsrDpcDataCount = min(m_ulIsrDpcDataCount, pIsrDpcOcaRecord->count());
    }
    // Allocate the array of ISR/DPC OCA datas
    m_aIsrDpcOcaDatas = new CIsrDpcOcaDataPtr[m_ulIsrDpcDataCount];

} // CIsrDpcOcaDatas

//******************************************************************************

CIsrDpcOcaDatas::~CIsrDpcOcaDatas()
{

} // ~CIsrDpcOcaDatas

//******************************************************************************

CIsrDpcOcaRecord::CIsrDpcOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pIsrDpcOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(Adapter),
    INIT(frequency),
    INIT(count)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the ISR_DPC_OCA_RECORD (from data pointer)
    SET(Adapter,    m_pIsrDpcOcaRecord);
    SET(frequency,  m_pIsrDpcOcaRecord);
    SET(count,      m_pIsrDpcOcaRecord);

    // Check for partial ISR_DPC_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial ISR_DPC_OCA_RECORD
        setPartial(true);
    }
    // If data field is present create the datas member
    if (dataField().isPresent())
    {
        // Create the datas member
        m_pIsrDpcOcaDatas = new CIsrDpcOcaDatas(this, ulRemaining);
    }

} // CIsrDpcOcaRecord

//******************************************************************************

CIsrDpcOcaRecord::~CIsrDpcOcaRecord()
{

} // ~CIsrDpcOcaRecord

//******************************************************************************

CAdapterOcaFlags::CAdapterOcaFlags
(
    const void         *pAdapterOcaFlags
)
:   m_pAdapterOcaFlags(pAdapterOcaFlags),
    INIT(InTDR),
    INIT(IsSLI),
    INIT(Value)
{
    assert(pAdapterOcaFlags != NULL);

    // Set the ADAPTER_OCA_FLAGS (from data pointer)
    SET(InTDR,  pAdapterOcaFlags);
    SET(IsSLI,  pAdapterOcaFlags);
    SET(Value,  pAdapterOcaFlags);

} // CAdapterOcaFlags

//******************************************************************************

CAdapterOcaFlags::~CAdapterOcaFlags()
{

} // ~CAdapterOcaFlags

//******************************************************************************

CAdapterOcaRecord::CAdapterOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pAdapterOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    m_pAdapterOcaFlags(NULL),
    m_pDeviceOcaRecords(NULL),
    m_pAllocationOcaRecords(NULL),
    INIT(Adapter),
    INIT(SemaphoreMemory),
    INIT(hClient),
    INIT(hDevice),
    INIT(DeviceCount),
    INIT(ChannelCount),
    INIT(NumOfIDSRecords),
    INIT(IlwalidSLTOffsetCount),
    INIT(TDRCount),
    INIT(LateBufferCompletionCount),
    INIT(BufferSubmissionErrorCount),
    INIT(EngineExelwtionMask),
    INIT(EnginePreemptionMask),
    INIT(LastPerformanceCounterFrequency),
    INIT(ConnectedDevicesMask),
    INIT(DeviceLoadState),
    INIT(ulArchitecture),
    INIT(ulImplementation),
    INIT(dwRevision),
    INIT(subRevision),
    INIT(SBIOSPowerUpRetry),
    INIT(Flags)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the ADAPTER_OCA_RECORD (from data pointer)
    SET(Adapter,                            m_pAdapterOcaRecord);
    SET(SemaphoreMemory,                    m_pAdapterOcaRecord);
    SET(hClient,                            m_pAdapterOcaRecord);
    SET(hDevice,                            m_pAdapterOcaRecord);
    SET(DeviceCount,                        m_pAdapterOcaRecord);
    SET(ChannelCount,                       m_pAdapterOcaRecord);
    SET(NumOfIDSRecords,                    m_pAdapterOcaRecord);
    SET(IlwalidSLTOffsetCount,              m_pAdapterOcaRecord);
    SET(TDRCount,                           m_pAdapterOcaRecord);
    SET(LateBufferCompletionCount,          m_pAdapterOcaRecord);
    SET(BufferSubmissionErrorCount,         m_pAdapterOcaRecord);
    SET(EngineExelwtionMask,                m_pAdapterOcaRecord);
    SET(EnginePreemptionMask,               m_pAdapterOcaRecord);
    SET(LastPerformanceCounterFrequency,    m_pAdapterOcaRecord);
    SET(ConnectedDevicesMask,               m_pAdapterOcaRecord);
    SET(DeviceLoadState,                    m_pAdapterOcaRecord);
    SET(ulArchitecture,                     m_pAdapterOcaRecord);
    SET(ulImplementation,                   m_pAdapterOcaRecord);
    SET(dwRevision,                         m_pAdapterOcaRecord);
    SET(subRevision,                        m_pAdapterOcaRecord);
    SET(SBIOSPowerUpRetry,                  m_pAdapterOcaRecord);
    SET(Flags,                              m_pAdapterOcaRecord);

    // Check for partial ADAPTER_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial ADAPTER_OCA_RECORD
        setPartial(true);
    }
    // If Flags field is present create the Flags member
    if (FlagsMember().isPresent())
    {
        // Create the Flags member
        m_pAdapterOcaFlags = new CAdapterOcaFlags(constcharptr(m_pAdapterOcaRecord) + FlagsMember().offset());
    }

} // CAdapterOcaRecord

//******************************************************************************

CAdapterOcaRecord::~CAdapterOcaRecord()
{

} // ~CAdapterOcaRecord

//******************************************************************************

const CDeviceOcaRecords*
CAdapterOcaRecord::deviceOcaRecords() const
{
    // See if the device OCA records not built yet
    if (m_pDeviceOcaRecords == NULL)
    {
        // Try to create the device OCA records (for this adapter)
        m_pDeviceOcaRecords = new CDeviceOcaRecords(this);
    }
    return m_pDeviceOcaRecords;

} // deviceOcaRecords

//******************************************************************************

const CAllocationOcaRecords*
CAdapterOcaRecord::allocationOcaRecords() const
{
    // See if the allocation OCA records not built yet
    if (m_pAllocationOcaRecords == NULL)
    {
        // Try to create the allocation OCA records (for this adapter)
        m_pAllocationOcaRecords = new CAllocationOcaRecords(this);
    }
    return m_pAllocationOcaRecords;

} // allocationOcaRecords

//******************************************************************************

CBufferInfoRecord::CBufferInfoRecord
(
    const void         *pBufferInfoRecord
)
:   m_pBufferInfoRecord(pBufferInfoRecord),
    INIT(Context),
    INIT(SubmitId),
    INIT(BufferId),
    INIT(FenceId),
    INIT(Size),
    INIT(lwoOffset),
    INIT(IntCount),
    INIT(Type)
{
    assert(pBufferInfoRecord != NULL);

    // Set the BUFFER_INFO_RECORD (from data pointer)
    SET(Context,    pBufferInfoRecord);
    SET(SubmitId,   pBufferInfoRecord);
    SET(BufferId,   pBufferInfoRecord);
    SET(FenceId,    pBufferInfoRecord);
    SET(Size,       pBufferInfoRecord);
    SET(lwoOffset,  pBufferInfoRecord);
    SET(IntCount,   pBufferInfoRecord);
    SET(Type,       pBufferInfoRecord);

} // CBufferInfoRecord

//******************************************************************************

CBufferInfoRecord::~CBufferInfoRecord()
{

} // ~CBufferInfoRecord

//******************************************************************************

CBufferInfoRecords::CBufferInfoRecords
(
    const CEngineIdOcaRecord *pEngineIdOcaRecord,
    ULONG               ulRemaining
)
:   m_pEngineIdOcaRecord(pEngineIdOcaRecord),
    m_ulBufferInfoCount(0),
    m_aBufferInfoRecords(NULL)
{
    assert(pEngineIdOcaRecord != NULL);

    // Callwlate the number of buffer info records (Based on record size and remaining data)
    m_ulBufferInfoCount = (min(ulRemaining, pEngineIdOcaRecord->size()) - pEngineIdOcaRecord->BuffersField().offset()) / CBufferInfoRecord::bufferInfoRecordType().size();

    // Allocate the array of buffer info records
    m_aBufferInfoRecords = new CBufferInfoRecordPtr[m_ulBufferInfoCount];

} // CBufferInfoRecords

//******************************************************************************

CBufferInfoRecords::~CBufferInfoRecords()
{

} // ~CBufferInfoRecords

//******************************************************************************

const void*
CBufferInfoRecords::bufferInfo
(
    ULONG               ulBufferInfo
) const
{
    const void         *pBufferInfo = NULL;

    // Check for invalid buffer info record
    if (ulBufferInfo >= bufferInfoCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid buffer info record %d (>= %d)",
                         ulBufferInfo, bufferInfoCount());
    }
    // Compute the requested buffer info record pointer
    pBufferInfo = constcharptr(m_pEngineIdOcaRecord->engineIdOcaRecord()) + m_pEngineIdOcaRecord->BuffersField().offset() + (ulBufferInfo * m_pEngineIdOcaRecord->BuffersField().size());

    return pBufferInfo;

} // bufferInfo

//******************************************************************************

const CBufferInfoRecord*
CBufferInfoRecords::bufferInfoRecord
(
    ULONG               ulBufferInfo
) const
{
    const void         *pBufferInfo;
    const CBufferInfoRecord *pBufferInfoRecord = NULL;

    // Check for invalid buffer info record
    if (ulBufferInfo >= bufferInfoCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid buffer info record %d (>= %d)",
                         ulBufferInfo, bufferInfoCount());
    }
    // Check to see if requested buffer info record needs to be loaded
    if (m_aBufferInfoRecords[ulBufferInfo] == NULL)
    {
        // Get the buffer info record pointer
        pBufferInfo = bufferInfo(ulBufferInfo);

        // Try to create the requested buffer info record
        m_aBufferInfoRecords[ulBufferInfo] = new CBufferInfoRecord(pBufferInfo);
    }
    // Get the requested buffer info record
    pBufferInfoRecord = m_aBufferInfoRecords[ulBufferInfo];

    return pBufferInfoRecord;

} // bufferInfoRecord

//******************************************************************************

CEngineIdOcaRecord::CEngineIdOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pEngineIdOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(BufferId),
    INIT(ISRBufferId),
    INIT(DPCBufferId),
    INIT(FenceId),
    INIT(ISRFenceId),
    INIT(DPCFenceId),
    INIT(Premption),
    INIT(PreemptionId),
    INIT(ISRPreemptionId),
    INIT(SubmitId),
    INIT(LastObservedProgress),
    INIT(LastObservedStall),
    INIT(NodeOrdinal),
    INIT(NodeType),
    INIT(NodeClassType),
    INIT(EngineOrdinal),
    INIT(MaxBufferExelwtionTime)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the ENGINEID_OCA_RECORD (from data pointer)
    SET(BufferId,               m_pEngineIdOcaRecord);
    SET(ISRBufferId,            m_pEngineIdOcaRecord);
    SET(DPCBufferId,            m_pEngineIdOcaRecord);
    SET(FenceId,                m_pEngineIdOcaRecord);
    SET(ISRFenceId,             m_pEngineIdOcaRecord);
    SET(DPCFenceId,             m_pEngineIdOcaRecord);
    SET(Premption,              m_pEngineIdOcaRecord);
    SET(PreemptionId,           m_pEngineIdOcaRecord);
    SET(ISRPreemptionId,        m_pEngineIdOcaRecord);
    SET(SubmitId,               m_pEngineIdOcaRecord);
    SET(LastObservedProgress,   m_pEngineIdOcaRecord);
    SET(LastObservedStall,      m_pEngineIdOcaRecord);
    SET(NodeOrdinal,            m_pEngineIdOcaRecord);
    SET(NodeType,               m_pEngineIdOcaRecord);
    SET(NodeClassType,          m_pEngineIdOcaRecord);
    SET(EngineOrdinal,          m_pEngineIdOcaRecord);
    SET(MaxBufferExelwtionTime, m_pEngineIdOcaRecord);

    // Check for partial ENGINEID_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial ENGINEID_OCA_RECORD
        setPartial(true);
    }
    // If Buffers field is present create the Buffers array
    if (BuffersField().isPresent())
    {
        // Create the Buffers member
        m_pBufferInfoRecords = new CBufferInfoRecords(this, ulRemaining);
    }

} // CEngineIdOcaRecord

//******************************************************************************

CEngineIdOcaRecord::~CEngineIdOcaRecord()
{

} // ~CEngineIdOcaRecord

//******************************************************************************

CDmaBufferOcaRecord::CDmaBufferOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pDmaBufferOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(Context),
    INIT(SubmitId),
    INIT(BufferId),
    INIT(FenceId),
    INIT(Size),
    INIT(lwoOffset),
    INIT(IntCount),
    INIT(Type)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the DMA_BUFFER_OCA_RECORD (from data pointer)
    SET(Context,    m_pDmaBufferOcaRecord);
    SET(SubmitId,   m_pDmaBufferOcaRecord);
    SET(BufferId,   m_pDmaBufferOcaRecord);
    SET(FenceId,    m_pDmaBufferOcaRecord);
    SET(Size,       m_pDmaBufferOcaRecord);
    SET(lwoOffset,  m_pDmaBufferOcaRecord);
    SET(IntCount,   m_pDmaBufferOcaRecord);
    SET(Type,       m_pDmaBufferOcaRecord);

    // Check for partial DMA_BUFFER_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial DMA_BUFFER_OCA_RECORD
        setPartial(true);
    }

} // CDmaBufferOcaRecord

//******************************************************************************

CDmaBufferOcaRecord::~CDmaBufferOcaRecord()
{

} // ~CDmaBufferOcaRecord

//******************************************************************************

CGpuWatchdogEvent::CGpuWatchdogEvent
(
    const void         *pGpuWatchdogEvent
)
:   m_pGpuWatchdogEvent(pGpuWatchdogEvent),
    INIT(AdapterOrdinal),
    INIT(NodeOrdinal),
    INIT(EngineOrdinal),
    INIT(Exelwting),
    INIT(BufferExelwtionTime),
    INIT(LwrrentTime),
    INIT(FenceId),
    INIT(PBget),
    INIT(C1get),
    INIT(Pixels),
    INIT(Status)
{
    assert(pGpuWatchdogEvent != NULL);

    // Set the GPUWATCHDOG_EVENT (from data pointer)
    SET(AdapterOrdinal,         pGpuWatchdogEvent);
    SET(NodeOrdinal,            pGpuWatchdogEvent);
    SET(EngineOrdinal,          pGpuWatchdogEvent);
    SET(Exelwting,              pGpuWatchdogEvent);
    SET(BufferExelwtionTime,    pGpuWatchdogEvent);
    SET(LwrrentTime,            pGpuWatchdogEvent);
    SET(FenceId,                pGpuWatchdogEvent);
    SET(PBget,                  pGpuWatchdogEvent);
    SET(C1get,                  pGpuWatchdogEvent);
    SET(Pixels,                 pGpuWatchdogEvent);
    SET(Status,                 pGpuWatchdogEvent);

} // CGpuWatchdogEvent

//******************************************************************************

CGpuWatchdogEvent::~CGpuWatchdogEvent()
{

} // ~CGpuWatchdogEvent

//******************************************************************************

CGpuWatchdogEvents::CGpuWatchdogEvents
(
    const CKmdRingBuffer *pKmdRingBuffer,
    ULONG               ulRemaining
)
:   m_pKmdRingBuffer(pKmdRingBuffer),
    m_ulGpuWatchdogEventCount(0),
    m_aGpuWatchdogEvents(NULL)
{
    assert(pKmdRingBuffer != NULL);

    // Callwlate the number of GPUWATCHDOG_EVENT events (Based on data type size and remaining data)
    m_ulGpuWatchdogEventCount = (min(ulRemaining, pKmdRingBuffer->kmdRingBufferType().size()) - pKmdRingBuffer->ElementsField().offset()) / CGpuWatchdogEvent::gpuWatchdogEventType().size();

    // Allocate the array of GPUWATCHDOG_EVENT events
    m_aGpuWatchdogEvents = new CGpuWatchdogEventPtr[m_ulGpuWatchdogEventCount];

} // CGpuWatchdogEvents

//******************************************************************************

CGpuWatchdogEvents::~CGpuWatchdogEvents()
{

} // ~CGpuWatchdogEvents

//******************************************************************************

const void*
CGpuWatchdogEvents::watchdogEvent
(
    ULONG               ulGpuWatchdogEvent
) const
{
    const void         *pWatchdogEvent = NULL;

    // Check for invalid GPU watchdog event
    if (ulGpuWatchdogEvent >= gpuWatchdogEventCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid GPU watchdog event %d (>= %d)",
                         ulGpuWatchdogEvent, gpuWatchdogEventCount());
    }
    // Compute the requested watchdog event pointer
    pWatchdogEvent = constcharptr(m_pKmdRingBuffer->kmdRingBuffer()) + m_pKmdRingBuffer->ElementsField().offset() + (ulGpuWatchdogEvent * CGpuWatchdogEvent::gpuWatchdogEventType().size());

    return pWatchdogEvent;

} // watchdogEvent

//******************************************************************************

const CGpuWatchdogEvent*
CGpuWatchdogEvents::gpuWatchdogEvent
(
    ULONG               ulGpuWatchdogEvent
) const
{
    const void         *pWatchdogEvent;
    const CGpuWatchdogEvent *pGpuWatchdogEvent = NULL;

    // Check for invalid GPU watchdog event
    if (ulGpuWatchdogEvent >= gpuWatchdogEventCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid GPU watchdog event %d (>= %d)",
                         ulGpuWatchdogEvent, gpuWatchdogEventCount());
    }
    // Check to see if requested GPU watchdog event needs to be loaded
    if (m_aGpuWatchdogEvents[ulGpuWatchdogEvent] == NULL)
    {
        // Get the GPU watchdog event pointer
        pWatchdogEvent = watchdogEvent(ulGpuWatchdogEvent);

        // Try to create the requested GPU watchdog event
        m_aGpuWatchdogEvents[ulGpuWatchdogEvent] = new CGpuWatchdogEvent(pWatchdogEvent);
    }
    // Get the requested GPU watchdog event
    pGpuWatchdogEvent = m_aGpuWatchdogEvents[ulGpuWatchdogEvent];

    return pGpuWatchdogEvent;

} // gpuWatchdogEvent

//******************************************************************************

CKmdRingBuffer::CKmdRingBuffer
(
    const void         *pKmdRingBuffer,
    ULONG               ulRemaining
)
:   m_pKmdRingBuffer(pKmdRingBuffer),
    INIT(ulIndex),
    INIT(ulCount)
{
    assert(pKmdRingBuffer != NULL);

    // Set the CKmdRingBuffer (from data pointer)
    SET(ulIndex,        pKmdRingBuffer);
    SET(ulCount,        pKmdRingBuffer);

    // If Elements field is present create the GPU watchdog events
    if (ElementsField().isPresent())
    {
        // Create the GPU watchdog events
        m_pGpuWatchdogEvents = new CGpuWatchdogEvents(this, ulRemaining);
    }

} // CKmdRingBuffer

//******************************************************************************

CKmdRingBuffer::~CKmdRingBuffer()
{

} // ~CKmdRingBuffer

//******************************************************************************

CKmdRingBufferOcaRecord::CKmdRingBufferOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pKmdRingBufferOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Check for partial CRingBuffer record
    if (ulRemaining < size())
    {
        // Indicate partial CRingBuffer record
        setPartial(true);
    }
    // Check if record size is large enough to indicate KMD ring buffer info record
    if (size() > CLwcdRecord::type().size())
    {
        // Create the KMD ring buffer member
        m_pKmdRingBuffer = new CKmdRingBuffer(constcharptr(m_pKmdRingBufferOcaRecord) + CLwcdRecord::type().size(), (ulRemaining - CLwcdRecord::type().size()));
    }

} // CKmdRingBufferOcaRecord

//******************************************************************************

CKmdRingBufferOcaRecord::~CKmdRingBufferOcaRecord()
{

} // ~CKmdRingBufferOcaRecord

//******************************************************************************

CAllocationOcaResource::CAllocationOcaResource
(
    const void         *pAllocationOcaResource
)
:   m_pAllocationOcaResource(pAllocationOcaResource),
    INIT(Type),
    INIT(Format),
    INIT(Width),
    INIT(Height),
    INIT(Depth),
    INIT(MipMapCount),
    INIT(VidPnSourceId)
{
    assert(pAllocationOcaResource != NULL);

    // Set the ALLOCATION_OCA_RESOURCE (from data pointer)
    SET(Type,           pAllocationOcaResource);
    SET(Format,         pAllocationOcaResource);
    SET(Width,          pAllocationOcaResource);
    SET(Height,         pAllocationOcaResource);
    SET(Depth,          pAllocationOcaResource);
    SET(MipMapCount,    pAllocationOcaResource);
    SET(VidPnSourceId,  pAllocationOcaResource);

} // CAllocationOcaResource

//******************************************************************************

CAllocationOcaResource::~CAllocationOcaResource()
{

} // ~CAllocationOcaResource

//******************************************************************************

CAllocationOcaRecord::CAllocationOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pAllocationOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(Pitch),
    INIT(Height),
    INIT(Bpp),
    INIT(AccessibleSize),
    INIT(TotalSize),
    INIT(AllowedHeaps),
    INIT(PreferredHeap),
    INIT(Segment),
    INIT(Offset)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the ALLOCATION_OCA_RECORD (from data pointer)
    SET(Pitch,          m_pAllocationOcaRecord);
    SET(Height,         m_pAllocationOcaRecord);
    SET(Bpp,            m_pAllocationOcaRecord);
    SET(AccessibleSize, m_pAllocationOcaRecord);
    SET(TotalSize,      m_pAllocationOcaRecord);
    SET(AllowedHeaps,   m_pAllocationOcaRecord);
    SET(PreferredHeap,  m_pAllocationOcaRecord);
    SET(Segment,        m_pAllocationOcaRecord);
    SET(Offset,         m_pAllocationOcaRecord);

    // Check for partial ALLOCATION_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial ALLOCATION_OCA_RECORD
        setPartial(true);
    }
    // If resource field is present create the resource member
    if (ResourceField().isPresent())
    {
        // Create the resource member
        m_pAllocationOcaResource = new CAllocationOcaResource(constcharptr(m_pAllocationOcaRecord) + ResourceField().offset());
    }

} // CAllocationOcaRecord

//******************************************************************************

CAllocationOcaRecord::~CAllocationOcaRecord()
{

} // ~CAllocationOcaRecord

//******************************************************************************

CKmdProcessOcaRecord::CKmdProcessOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pKmdProcessOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(hClient),
    INIT(hDevice),
    INIT(KmdProcess),
    INIT(Device),
    INIT(Adapter),
    INIT(Process),
    INIT(ProcessImageName),
    INIT(DeviceCount)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the KMDPROCESS_OCA_RECORD (from data pointer)
    SET(hClient,            m_pKmdProcessOcaRecord);
    SET(hDevice,            m_pKmdProcessOcaRecord);
    SET(KmdProcess,         m_pKmdProcessOcaRecord);
    SET(Device,             m_pKmdProcessOcaRecord);
    SET(Adapter,            m_pKmdProcessOcaRecord);
    SET(Process,            m_pKmdProcessOcaRecord);
    SET(ProcessImageName,   m_pKmdProcessOcaRecord);
    SET(DeviceCount,        m_pKmdProcessOcaRecord);

    // Check for partial KMDPROCESS_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial KMDPROCESS_OCA_RECORD
        setPartial(true);
    }

} // CKmdProcessOcaRecord

//******************************************************************************

CKmdProcessOcaRecord::~CKmdProcessOcaRecord()
{

} // ~CKmdProcessOcaRecord

//******************************************************************************

ULONG
CKmdProcessOcaRecord::size() const
{
    ULONG               ulSize;

    // Set size based on actual KMDPROCESS_OCA_RECORD size
    ulSize = type().size();

    return ulSize;

} // size

//******************************************************************************

CString
CKmdProcessOcaRecord::processName() const
{
    ULONG               ulNameLength;
    CString             sProcessName;

    // Check for process name available
    if (ProcessImageNameField().isPresent())
    {
        // Check for an process name available
        if (ProcessImageName() != 0x00)
        {
            // get the process name length (Array size)
            ulNameLength = ProcessImageNameField().dimension(0);

            // Reserve enough string space for the process name
            sProcessName.reserve(ulNameLength);

            // Copy the image filename as the process name
            memcpy(sProcessName.data(), ProcessImageNameMember().getStruct(), ulNameLength);

            // Make sure process name is terminated
            sProcessName[ulNameLength - 1] = '\0';
        }
        else    // Process is not present
        {
            // Reserve enough string space for process address as name
            sProcessName.reserve(2 + pointerWidth());

            // Generate the process name as the process address
            sProcessName.sprintf("0x%0*I64x", PTR(Process()));
        }
    }
    else    // Process name is not available
    {
        // Reserve enough string space for process address as name
        sProcessName.reserve(2 + pointerWidth());

        // Generate the process name as the process address
        sProcessName.sprintf("0x%0*I64x", PTR(Process()));
    }
    return sProcessName;

} // processName

//******************************************************************************

CDeviceOcaRecord::CDeviceOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pDeviceOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(hClient),
    INIT(hDevice),
    INIT(KmdProcess),
    INIT(Device),
    INIT(Adapter),
    INIT(Process),
    INIT(ContextCount),
    INIT(ReferenceCount),
    INIT(bLockedPerfMon),
    INIT(dmaBufferSize)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the DEVICE_OCA_RECORD (from data pointer)
    SET(hClient,        m_pDeviceOcaRecord);
    SET(hDevice,        m_pDeviceOcaRecord);
    SET(KmdProcess,     m_pDeviceOcaRecord);
    SET(Device,         m_pDeviceOcaRecord);
    SET(Adapter,        m_pDeviceOcaRecord);
    SET(Process,        m_pDeviceOcaRecord);
    SET(ContextCount,   m_pDeviceOcaRecord);
    SET(ReferenceCount, m_pDeviceOcaRecord);
    SET(bLockedPerfMon, m_pDeviceOcaRecord);
    SET(dmaBufferSize,  m_pDeviceOcaRecord);

    // Check for partial DEVICE_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial DEVICE_OCA_RECORD
        setPartial(true);
    }

} // CDeviceOcaRecord

//******************************************************************************

CDeviceOcaRecord::~CDeviceOcaRecord()
{

} // ~CDeviceOcaRecord

//******************************************************************************

CContextOcaRecord::CContextOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pContextOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(Context),
    INIT(Channel),
    INIT(Adapter),
    INIT(Device),
    INIT(NodeOrdinal),
    INIT(EngineOrdinal),
    INIT(BufferId)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the CONTEXT_OCA_RECORD (from data pointer)
    SET(Context,        m_pContextOcaRecord);
    SET(Channel,        m_pContextOcaRecord);
    SET(Adapter,        m_pContextOcaRecord);
    SET(Device,         m_pContextOcaRecord);
    SET(NodeOrdinal,    m_pContextOcaRecord);
    SET(EngineOrdinal,  m_pContextOcaRecord);
    SET(BufferId,       m_pContextOcaRecord);

    // Check for partial CONTEXT_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial CONTEXT_OCA_RECORD
        setPartial(true);
    }

} // CContextOcaRecord

//******************************************************************************

CContextOcaRecord::~CContextOcaRecord()
{

} // ~CContextOcaRecord

//******************************************************************************

CChannelOcaRecord::CChannelOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pChannelOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(Channel),
    INIT(HwChannelIndex),
    INIT(hClient),
    INIT(hDevice),
    INIT(DmaCount),
    INIT(NodeOrdinal),
    INIT(bShared),
    INIT(bReserved),
    INIT(ContextCount)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the CHANNEL_OCA_RECORD (from data pointer)
    SET(Channel,        m_pChannelOcaRecord);
    SET(HwChannelIndex, m_pChannelOcaRecord);
    SET(hClient,        m_pChannelOcaRecord);
    SET(hDevice,        m_pChannelOcaRecord);
    SET(DmaCount,       m_pChannelOcaRecord);
    SET(NodeOrdinal,    m_pChannelOcaRecord);
    SET(bShared,        m_pChannelOcaRecord);
    SET(bReserved,      m_pChannelOcaRecord);
    SET(ContextCount,   m_pChannelOcaRecord);

    // Check for partial CHANNEL_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial CHANNEL_OCA_RECORD
        setPartial(true);
    }

} // CChannelOcaRecord

//******************************************************************************

CChannelOcaRecord::~CChannelOcaRecord()
{

} // ~CChannelOcaRecord

//******************************************************************************

CDisplayTargetOcaRecord::CDisplayTargetOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pDisplayTargetOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(VidPnTargetId),
    INIT(head),
    INIT(device),
    INIT(connector),
    INIT(srcID),
    INIT(srcImportance),
    INIT(hAllocation),
    INIT(Address),
    INIT(bFlipPending),
    INIT(flipPendingAddress),
    INIT(width),
    INIT(height),
    INIT(depth),
    INIT(refreshRate),
    INIT(colorFormat),
    INIT(rotation),
    INIT(callFromTMM),
    INIT(SelectLwstomTiming),
    INIT(tvFormat),
    INIT(srcPartitionX),
    INIT(srcPartitionY),
    INIT(srcPartitionW),
    INIT(srcPartitionH),
    INIT(viewportInX),
    INIT(viewportInY),
    INIT(viewportInW),
    INIT(viewportInH),
    INIT(scalingMethod),
    INIT(viewportOutX),
    INIT(viewportOutY),
    INIT(viewportOutW),
    INIT(viewportOutH),
    INIT(timingOverride),
    INIT(bVsyncEnabled),
    INIT(HVisible),
    INIT(HBorder),
    INIT(HFrontPorch),
    INIT(HSyncWidth),
    INIT(HTotal),
    INIT(HSyncPol),
    INIT(VVisible),
    INIT(VBorder),
    INIT(VFrontPorch),
    INIT(VSyncWidth),
    INIT(VTotal),
    INIT(VSyncPol),
    INIT(interlaced),
    INIT(pclk),
    INIT(flag),
    INIT(rr),
    INIT(rrx1k),
    INIT(aspect),
    INIT(rep),
    INIT(status),
    INIT(name)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the DISPLAY_TARGET_OCA_RECORD (from data pointer)
    SET(VidPnTargetId,      m_pDisplayTargetOcaRecord);
    SET(head,               m_pDisplayTargetOcaRecord);
    SET(device,             m_pDisplayTargetOcaRecord);
    SET(connector,          m_pDisplayTargetOcaRecord);
    SET(srcID,              m_pDisplayTargetOcaRecord);
    SET(srcImportance,      m_pDisplayTargetOcaRecord);
    SET(hAllocation,        m_pDisplayTargetOcaRecord);
    SET(Address,            m_pDisplayTargetOcaRecord);
    SET(bFlipPending,       m_pDisplayTargetOcaRecord);
    SET(flipPendingAddress, m_pDisplayTargetOcaRecord);
    SET(width,              m_pDisplayTargetOcaRecord);
    SET(height,             m_pDisplayTargetOcaRecord);
    SET(depth,              m_pDisplayTargetOcaRecord);
    SET(refreshRate,        m_pDisplayTargetOcaRecord);
    SET(colorFormat,        m_pDisplayTargetOcaRecord);
    SET(rotation,           m_pDisplayTargetOcaRecord);
    SET(callFromTMM,        m_pDisplayTargetOcaRecord);
    SET(SelectLwstomTiming, m_pDisplayTargetOcaRecord);
    SET(tvFormat,           m_pDisplayTargetOcaRecord);
    SET(srcPartitionX,      m_pDisplayTargetOcaRecord);
    SET(srcPartitionY,      m_pDisplayTargetOcaRecord);
    SET(srcPartitionW,      m_pDisplayTargetOcaRecord);
    SET(srcPartitionH,      m_pDisplayTargetOcaRecord);
    SET(viewportInX,        m_pDisplayTargetOcaRecord);
    SET(viewportInY,        m_pDisplayTargetOcaRecord);
    SET(viewportInW,        m_pDisplayTargetOcaRecord);
    SET(viewportInH,        m_pDisplayTargetOcaRecord);
    SET(scalingMethod,      m_pDisplayTargetOcaRecord);
    SET(viewportOutX,       m_pDisplayTargetOcaRecord);
    SET(viewportOutY,       m_pDisplayTargetOcaRecord);
    SET(viewportOutW,       m_pDisplayTargetOcaRecord);
    SET(viewportOutH,       m_pDisplayTargetOcaRecord);
    SET(timingOverride,     m_pDisplayTargetOcaRecord);
    SET(bVsyncEnabled,      m_pDisplayTargetOcaRecord);
    SET(HVisible,           m_pDisplayTargetOcaRecord);
    SET(HBorder,            m_pDisplayTargetOcaRecord);
    SET(HFrontPorch,        m_pDisplayTargetOcaRecord);
    SET(HSyncWidth,         m_pDisplayTargetOcaRecord);
    SET(HTotal,             m_pDisplayTargetOcaRecord);
    SET(HSyncPol,           m_pDisplayTargetOcaRecord);
    SET(VVisible,           m_pDisplayTargetOcaRecord);
    SET(VBorder,            m_pDisplayTargetOcaRecord);
    SET(VFrontPorch,        m_pDisplayTargetOcaRecord);
    SET(VSyncWidth,         m_pDisplayTargetOcaRecord);
    SET(VTotal,             m_pDisplayTargetOcaRecord);
    SET(VSyncPol,           m_pDisplayTargetOcaRecord);
    SET(interlaced,         m_pDisplayTargetOcaRecord);
    SET(pclk,               m_pDisplayTargetOcaRecord);
    SET(flag,               m_pDisplayTargetOcaRecord);
    SET(rr,                 m_pDisplayTargetOcaRecord);
    SET(rrx1k,              m_pDisplayTargetOcaRecord);
    SET(aspect,             m_pDisplayTargetOcaRecord);
    SET(rep,                m_pDisplayTargetOcaRecord);
    SET(status,             m_pDisplayTargetOcaRecord);
    SET(name,               m_pDisplayTargetOcaRecord);

    // Check for partial DISPLAY_TARGET_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial DISPLAY_TARGET_OCA_RECORD
        setPartial(true);
    }

} // CDisplayTargetOcaRecord

//******************************************************************************

CDisplayTargetOcaRecord::~CDisplayTargetOcaRecord()
{

} // ~CDisplayTargetOcaRecord

//******************************************************************************

CMonitorInfoOcaRecord::CMonitorInfoOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pMonitorInfoOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(dwEDIDSize),
    INIT(EDID)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the MONITOR_INFO_OCA_RECORD (from data pointer)
    SET(dwEDIDSize, m_pMonitorInfoOcaRecord);
    SET(EDID,       m_pMonitorInfoOcaRecord);

    // Check for partial MONITOR_INFO_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial MONITOR_INFO_OCA_RECORD
        setPartial(true);
    }

} // CMonitorInfoOcaRecord

//******************************************************************************

CMonitorInfoOcaRecord::~CMonitorInfoOcaRecord()
{

} // ~CMonitorInfoOcaRecord

//******************************************************************************

CVblankInfoData::CVblankInfoData
(
    const void         *pVblankInfoData
)
:   m_pVblankInfoData(pVblankInfoData),
    INIT(offset),
    INIT(head),
    INIT(VidPnTargetId),
    INIT(timestamp)
{
    assert(pVblankInfoData != NULL);

    // Set the VBLANK_INFO_DATA (from data pointer)
    SET(offset,         pVblankInfoData);
    SET(head,           pVblankInfoData);
    SET(VidPnTargetId,  pVblankInfoData);
    SET(timestamp,      pVblankInfoData);

} // CVblankInfoData

//******************************************************************************

CVblankInfoData::~CVblankInfoData()
{

} // ~CVblankInfoData

//******************************************************************************

CVblankInfoDatas::CVblankInfoDatas
(
    const CVblankInfoOcaRecord *pVblankInfoOcaRecord,
    ULONG               ulRemaining
)
:   m_pVblankInfoOcaRecord(pVblankInfoOcaRecord),
    m_ulVblankDataCount(0),
    m_aVblankInfoDatas(NULL)
{
    assert(pVblankInfoOcaRecord != NULL);

    // Callwlate the maximum number of vblank info datas (Based on record size and remaining data)
    m_ulVblankDataCount = (min(ulRemaining, pVblankInfoOcaRecord->size()) - pVblankInfoOcaRecord->dataField().offset()) / CVblankInfoData::vblankInfoDataType().size();

    // Check for vblank info count available and valid
    if (pVblankInfoOcaRecord->countMember().isPresent() && (pVblankInfoOcaRecord->count() != 0))
    {
        // Use the number of vblank info datas (If valid)
        m_ulVblankDataCount = min(m_ulVblankDataCount, pVblankInfoOcaRecord->count());
    }
    // Allocate the array of vblank info datas
    m_aVblankInfoDatas = new CVblankInfoDataPtr[m_ulVblankDataCount];

} // CVblankInfoDatas

//******************************************************************************

CVblankInfoDatas::~CVblankInfoDatas()
{

} // ~CVblankInfoDatas

//******************************************************************************

const void*
CVblankInfoDatas::vblankData
(
    ULONG               ulVblankInfo
) const
{
    const void         *pVblankData = NULL;

    // Check for invalid vblank data record
    if (ulVblankInfo >= vblankDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid vblank data record %d (>= %d)",
                         ulVblankInfo, vblankDataCount());
    }
    // Compute the requested vblank info data pointer
    pVblankData = constcharptr(m_pVblankInfoOcaRecord->vblankInfoOcaRecord()) + m_pVblankInfoOcaRecord->dataField().offset() + (ulVblankInfo * m_pVblankInfoOcaRecord->dataField().size());

    return pVblankData;

} // vblankData

//******************************************************************************

const CVblankInfoData*
CVblankInfoDatas::vblankInfoData
(
    ULONG               ulVblankInfo
) const
{
    const void         *pVblankData;
    const CVblankInfoData *pVblankInfoData = NULL;

    // Check for invalid vblank data record
    if (ulVblankInfo >= vblankDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid vblank data record %d (>= %d)",
                         ulVblankInfo, vblankDataCount());
    }
    // Check to see if requested vblank info data needs to be loaded
    if (m_aVblankInfoDatas[ulVblankInfo] == NULL)
    {
        // Get the vblank info data pointer
        pVblankData = vblankData(ulVblankInfo);

        // Try to create the requested vblank info data
        m_aVblankInfoDatas[ulVblankInfo] = new CVblankInfoData(pVblankData);
    }
    // Get the requested vblank info data
    pVblankInfoData = m_aVblankInfoDatas[ulVblankInfo];

    return pVblankInfoData;

} // vblankInfoData

//******************************************************************************

CVblankInfoOcaRecord::CVblankInfoOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pVblankInfoOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(Adapter),
    INIT(frequency),
    INIT(count)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the VBLANK_INFO_OCA_RECORD (from data pointer)
    SET(Adapter,    m_pVblankInfoOcaRecord);
    SET(frequency,  m_pVblankInfoOcaRecord);
    SET(count,      m_pVblankInfoOcaRecord);

    // Check for partial VBLANK_INFO_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial VBLANK_INFO_OCA_RECORD
        setPartial(true);
    }
    // If data field is present create the datas array
    if (dataField().isPresent() && size())
    {
        // Create the datas member
        m_pVblankInfoDatas = new CVblankInfoDatas(this, ulRemaining);
    }
    else
    {
        m_pVblankInfoDatas = NULL;
    }

} // CVblankInfoOcaRecord

//******************************************************************************

CVblankInfoOcaRecord::~CVblankInfoOcaRecord()
{

} // ~CVblankInfoOcaRecord

//******************************************************************************

CErrorInfoData::CErrorInfoData
(
    const void         *pErrorInfoData
)
:   m_pErrorInfoData(pErrorInfoData),
    INIT(Type),
    INIT(subType),
    INIT(status),
    INIT(address),
    INIT(adapter),
    INIT(device),
    INIT(context),
    INIT(channel),
    INIT(allocation),
    INIT(process),
    INIT(thread),
    INIT(data0),
    INIT(data1),
    INIT(data2),
    INIT(data3),
    INIT(data4),
    INIT(timestamp)
{
    assert(pErrorInfoData != NULL);

    // Set the ERROR_INFO_DATA (from data pointer)
    SET(Type,       pErrorInfoData);
    SET(subType,    pErrorInfoData);
    SET(status,     pErrorInfoData);
    SET(address,    pErrorInfoData);
    SET(adapter,    pErrorInfoData);
    SET(device,     pErrorInfoData);
    SET(context,    pErrorInfoData);
    SET(channel,    pErrorInfoData);
    SET(allocation, pErrorInfoData);
    SET(process,    pErrorInfoData);
    SET(thread,     pErrorInfoData);
    SET(data0,      pErrorInfoData);
    SET(data1,      pErrorInfoData);
    SET(data2,      pErrorInfoData);
    SET(data3,      pErrorInfoData);
    SET(data4,      pErrorInfoData);
    SET(timestamp,  pErrorInfoData);

} // CErrorInfoData

//******************************************************************************

CErrorInfoData::~CErrorInfoData()
{

} // ~CErrorInfoData

//******************************************************************************

CString
CErrorInfoData::statusString() const
{
    CString             sStatus(MAX_COMMAND_STRING);

    // Build the status error string
    sStatus.sprintf("0x%08x", status());

    // Check to see if this is an OS or RM status
    if (typeName().find("_RM_API_") == NOT_FOUND)
    {
        // OS status, build command to display error description (if DML enabled)
        sStatus = exec(sStatus, buildModCommand("error", "ext", sStatus));
    }
    return sStatus;

} // statusString

//******************************************************************************

CString
CErrorInfoData::errorString() const
{
    CString             sError(MAX_COMMAND_STRING);

    // Check to see if this is an OS or RM status
    if (typeName().find("_RM_API_") == NOT_FOUND)
    {
        // OS status, get the OS error description
        sError = ::errorString(status());
    }
    else    // RM status
    {
        // RM status, get the RM error description
        sError = rmString(status());
    }
    return sError;

} // errorString

//******************************************************************************

CString
CErrorInfoData::typeName() const
{
    CString             sTypeName(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Check for error event type names present
        if (errorEventEnum().isPresent())
        {
            // Try to get the error event type name
            errorEventEnum().getConstantName(Type(), sTypeName.data(), static_cast<ULONG>(sTypeName.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sTypeName;

} // typeName

//******************************************************************************

CString
CErrorInfoData::subTypeName() const
{
    CString             sTypeName;
    CString             sSubTypeName;

    // Try to get the error type name (Subtype based on type name)
    sTypeName = typeName();
    if (!sTypeName.empty())
    {
        // Check for a DM interface log type
        if (isDmInterfaceType(sTypeName))
        {
            // Get the DM interface subtype name
            sSubTypeName = dmInterfaceName(subType());
        }
        // Check for a LDDM interface log type
        else if (isLddmInterfaceType(sTypeName))
        {
            // Get the DM interface subtype name
            sSubTypeName = lddmInterfaceName(subType());
        }
        // Check for a DL interface log type
        else if (isDlInterfaceType(sTypeName))
        {
            // Get the DL interface subtype name
            sSubTypeName = dlInterfaceName(subType());
        }
        // Check for a CB interface log type
        else if (isCbInterfaceType(sTypeName))
        {
            // Get the CB interface subtype name
            sSubTypeName = cbInterfaceName(subType());
        }
        // Check for a IFACE interface log type
        else if (isIfaceInterfaceType(sTypeName))
        {
            // Get the IFACE interface subtype name
            sSubTypeName = ifaceInterfaceName(subType());
        }
        // Check for a RM interface log type
        else if (isRmInterfaceType(sTypeName))
        {
            // Get the RM interface subtype name
            sSubTypeName = rmInterfaceName(subType());
        }
        // Check for an AGP interface log type
        else if (isAgpInterfaceType(sTypeName))
        {
            // Get the AGP interface subtype name
            sSubTypeName = agpInterfaceName(subType());
        }
        // Check for a TimedOp interface log type
        else if (isTimedOpInterfaceType(sTypeName))
        {
            // Get the TimedOp interface subtype name
            sSubTypeName = timedOpInterfaceName(subType());
        }
        // Check for a Client Arbitration log type
        else if (isClientArbType(sTypeName))
        {
            // Get the Client Arbitration subtype name
            sSubTypeName = clientArbSubtypeName(subType());
        }
        // Check for a GDI Accelerated operation log type
        else if (isGdiAccelOpType(sTypeName))
        {
            // Get the GDI Accelerated operation type name
            sSubTypeName = gdiAccelOpTypeName(subType());
        }
        // Check for a buffer operation log type
        else if (isBufferOperationType(sTypeName))
        {
            // Get the buffer operation subtype name
            sSubTypeName = bufferOperationName(subType());
        }
        // Check for an idle operation log type
        else if (isIdleOperationType(sTypeName))
        {
            // Get the idle operation subtype name
            sSubTypeName = idleOperationName(subType());
        }
        // Check for an ISR notify log type
        else if (isIsrNotifyType(sTypeName))
        {
            // Get the ISR notify subtype name
            sSubTypeName = isrNotifyName(subType());
        }
        // Check for a build paging buffer log type
        else if (isBuildPagingBufferType(sTypeName))
        {
            // Get the build paging buffer subtype name
            sSubTypeName = buildPagingBufferName(subType());
        }
    }
    return sSubTypeName;

} // subTypeName

//******************************************************************************

CString
CErrorInfoData::annotation() const
{
    CString             sAnnotation;

    // Try to get annotation string if address present
    if (addressMember().isPresent())
    {
        // Try to get the annotation string for this address
        sAnnotation = getAnnotation(address());

        // Check for annotation string present
        if (!sAnnotation.empty())
        {
            // Check DML state (Need to DML escape annotation string)
            if (dmlState())
            {
                // DML escape the annotation string
                sAnnotation = dmlEscape(sAnnotation);
            }
            // Try to catch any annotation format errors
            try
            {
                // Format the annotation string
                sAnnotation = format(sAnnotation);
            }
            catch(...)
            {
                // Check DML state (Set annotation string to red, indicating format error)
                sAnnotation = foreground(sAnnotation, RED);
            }
        }
    }
    return sAnnotation;

} // annotation

//******************************************************************************

CString
CErrorInfoData::format
(
    const char         *pFormat
) const
{
    const char         *pLwrrent = pFormat;
    const char         *pLocation;
    const char         *pSpecifier = NULL;
    ULONG               ulStart;
    ULONG               ulEnd;
    ULONG64             ulDefault = 0;
    ULONG64             ulValue = 0;
    bool                bValid;
    bool                bDefault;
    bool                bSize;
    const CDeviceOcaRecord* pDeviceOcaRecord;
    const CContextOcaRecord* pContextOcaRecord;
    CString             sFormat(MAX_DBGPRINTF_STRING);
    CString             sString(MAX_COMMAND_STRING);
    CString             sSpecifier(MAX_COMMAND_STRING);
    CString             sFormatted;

    assert(pFormat != NULL);

    // Check for special log formatting characters in the format string
    pLocation = strchr(pFormat, LOG_FORMAT_ENTRY);
    if (pLocation != NULL)
    {
        // Loop processing all the special format entries
        do
        {
            // Check for characters before special log format entry
            if (pLocation != pLwrrent)
            {
                // Append these characters onto the formatted string
                sFormatted.append(pLwrrent, 0, (pLocation - pLwrrent));
            }
            // Move past the format entry character
            pLocation++;

            // Default to valid, default format, no size log format entry
            bValid   = true;
            bDefault = true;
            bSize    = false;

            // Check for start/end bit position values
            if (*pLocation == LOG_FORMAT_OPEN)
            {
                // Increment past start of bit position values
                pLocation++;

                // Loop trying to get the start bit position
                ulStart = 0;
                while (*pLocation != EOS)
                {
                    // Check next character for end of start value
                    if ((*pLocation < '0') || (*pLocation > '9'))
                    {
                        // Non-decimal character, end of start value
                        break;
                    }
                    // Add this numeric character into the start value
                    ulStart = (ulStart * 10) + (*pLocation - '0');

                    // Increment to next character
                    pLocation++;
                }
                // Check for end value present
                if (*pLocation == LOG_FORMAT_SEPARATOR)
                {
                    // Increment past bit position separator
                    pLocation++;

                    // Loop trying to get the end bit position
                    ulEnd = 0;
                    while (*pLocation != EOS)
                    {
                        // Check next character for end of end value
                        if ((*pLocation < '0') || (*pLocation > '9'))
                        {
                            // Non-decimal character, end of end value
                            break;
                        }
                        // Add this numeric character into the end value
                        ulEnd = (ulEnd * 10) + (*pLocation - '0');

                        // Increment to next character
                        pLocation++;
                    }
                }
                else    // No end bit position value
                {
                    // Set end bit position to start position (Single bit)
                    ulEnd = ulStart;
                }
                // Check for valid bit position end
                if (*pLocation == LOG_FORMAT_CLOSE)
                {
                    // Increment past the bit position values
                    pLocation++;

                    // Check for invalid bit position values
                    if ((ulStart > ulEnd) || (ulEnd > 63))
                    {
                        // Indicate invalid format entry
                        bValid = false;
                    }
                }
                else    // Invalid bit positions
                {
                    // Indicate invalid format entry
                    bValid = false;
                }
            }
            else    // No bit positions specified
            {
                // Set start and end to full 64-bit value
                ulStart = 0;
                ulEnd   = 63;
            }
            // Check the log format entry
            switch(*pLocation)
            {
                case LOG_FORMAT_TYPE:           // Log entry type value

                    // Skip log format character
                    pLocation++;

                    // Set value to type value
                    ulValue = Type();

                    // Setup typename with error color (if DML enabled)
                    sString = foreground(typeName(), RED);

                    // Setup the default value and format specifier
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_SUBTYPE:        // Log entry subtype value

                    // Skip log format character
                    pLocation++;

                    // Set value to subtype value
                    ulValue = subType();

                    // Setup subtypename with error color (if DML enabled)
                    sString = foreground(subTypeName(), BLUE);

                    // Setup the default value and format specifier
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_STATUS:         // Log entry status value

                    // Skip log format character
                    pLocation++;

                    // Set value to status value
                    ulValue = status();

                    // Setup the default value and format specifier
                    sString    = statusString();
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_TIMESTAMP:      // Log entry timestamp value

                    // Skip log format character
                    pLocation++;

                    // Set value to timestamp value
                    ulValue = timestamp();

                    // Setup the default value and format specifier
                    ulDefault  = timestamp();
                    pSpecifier = "0x%016I64x";

                    break;

                case LOG_FORMAT_ADDRESS:        // Log entry address value

                    // Skip log format character
                    pLocation++;

                    // Set value to address value
                    ulValue = address().ptr();

                    // Setup the default value and format specifier
                    sString    = openString();
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_ADAPTER:        // Log entry adapter value

                    // Skip log format character
                    pLocation++;

                    // Set value to adapter address
                    ulValue = adapter().ptr();

                    // Setup the default adapter value and format specifier
                    ulDefault  = adapter().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_DEVICE:         // Log entry device value

                    // Skip log format character
                    pLocation++;

                    // Set value to device address
                    ulValue = device().ptr();

                    // Setup the default device value and format specifier
                    ulDefault  = device().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_CONTEXT:        // Log entry context value

                    // Skip log format character
                    pLocation++;

                    // Set value to context address
                    ulValue = context().ptr();

                    // Setup the default context value and format specifier
                    ulDefault  = context().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_CHANNEL:        // Log entry channel value

                    // Skip log format character
                    pLocation++;

                    // Set value to channel address
                    ulValue = channel().ptr();

                    // Setup the default channel value and format specifier
                    ulDefault  = channel().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_ALLOCATION:     // Log entry allocation value

                    // Skip log format character
                    pLocation++;

                    // Check for allocation member present
                    if (allocationMember().isPresent())
                    {
                        // Setup the default allocation value and format specifier
                        ulDefault  = allocation().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No allocation member
                    {
                        // Setup the default allocation value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_PROCESS:        // Log entry process value

                    // Skip log format character
                    pLocation++;

                    // Check for process member present
                    if (processMember().isPresent())
                    {
                        // Setup the default process value and format specifier
                        ulDefault  = process().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No process member
                    {
                        // Setup the default process value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_THREAD:         // Log entry thread value

                    // Skip log format character
                    pLocation++;

                    // Check for thread member present
                    if (threadMember().isPresent())
                    {
                        // Setup the default thread value and format specifier
                        ulDefault  = thread().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No thread member
                    {
                        // Setup the default thread value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_KMD_PROCESS:    // Log entry KMD process value

                    // Skip log format character
                    pLocation++;

                    // Setup the default KMD process value and format specifier
                    ulDefault  = 0;
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    // Try to find the OCA record for this device
                    pDeviceOcaRecord = findOcaDevice(device().ptr());
                    if (pDeviceOcaRecord == NULL)
                    {
                        // Try to find the OCA record for this context
                        pContextOcaRecord = findOcaContext(context().ptr());
                        if (pContextOcaRecord != NULL)
                        {
                            // Try too find the OCA record for this context device
                            pDeviceOcaRecord = findOcaDevice(pContextOcaRecord->Device().ptr());
                        }
                    }
                    // Check for device OCA record found
                    if (pDeviceOcaRecord != NULL)
                    {
                        // Get the KMD process value for this record
                        ulDefault = pDeviceOcaRecord->KmdProcess().ptr();
                    }
                    break;

                case LOG_FORMAT_DATA_0:         // Log entry data 0 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 0 value
                    ulValue = data0();

                    // Setup default data 0 value and format specifier
                    ulDefault  = data0();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_1:         // Log entry data 1 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 1 value
                    ulValue = data1();

                    // Setup default data 1 value and format specifier
                    ulDefault  = data1();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_2:         // Log entry data 2 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 2 value
                    ulValue = data2();

                    // Setup default data 2 value and format specifier
                    ulDefault  = data2();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_3:         // Log entry data 3 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 3 value
                    ulValue = data3();

                    // Setup default data 3 value and format specifier
                    ulDefault  = data3();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_4:         // Log entry data 4 value

                    // Skip log format character
                    pLocation++;

                    // Check for data 4 member present
                    if (data4Member().isPresent())
                    {
                        // Setup the default data 4 value and format specifier
                        ulDefault  = data4();
                        pSpecifier = "0x%0I64x";
                        bSize      = false;
                    }
                    else    // No data 4 member
                    {
                        // Setup the default data 4 value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                default:                        // Unknown log entry format specifier

                    // Indicate invalid format entry
                    bValid = false;

                    break;
            }
            // Check for user specified format
            if (*pLocation == PERCENT)
            {
                // Indicate no default format
                bDefault = false;

                // Save user format specifier address
                pSpecifier = pLocation;

                // Increment past format specifier
                pLocation++;

                // Search for end of format specifier
                while (*pLocation != EOS)
                {
                    // Almost any alpha character is the format specifier
                    if (isalpha(*pLocation))
                    {
                        // Only check for I64 format modifier for now
                        if (*pLocation != 'I')
                        {
                            // Skip past format specifier and terminate search
                            pLocation++;

                            break;
                        }
                    }
                    // Check for an asterisk format specifier (Pointer size needed)
                    if (*pLocation == ASTERISK)
                    {
                        bSize = true;
                    }
                    // Otherwise, simply skip this format character
                    pLocation++;
                }
                // Setup the user format specifier
                sSpecifier.assign(pSpecifier, 0, (pLocation - pSpecifier));

                // Setup specifier pointer
                pSpecifier = sSpecifier;
            }
            // Check for valid log format entry found
            if (bValid)
            {
                // Check for default or user specified format
                if (bDefault)
                {
                    // Check for pointer size width required
                    if (bSize)
                    {
                        // Format value with default format and size (Assumes pointer)
                        sFormat.sprintf(pSpecifier, pointerWidth(), (ulDefault & pointerMask()));
                    }
                    else    // No pointer size width required
                    {
                        // Format value with default format
                        sFormat.sprintf(pSpecifier, ulDefault);
                    }
                }
                else    // User specified format
                {
                    // Shift and mask the data value
                    ulValue >>= ulStart;
                    ulValue &= ((1ll << (ulEnd - ulStart)) << 1) - 1;

                    // Check for pointer size width required
                    if (bSize)
                    {
                        // Format value with user format and size (Assumes pointer)
                        sFormat.sprintf(pSpecifier, pointerWidth(), (ulValue & pointerMask()));
                    }
                    else    // No pointer size width required
                    {
                        // Format value with user format
                        sFormat.sprintf(pSpecifier, ulValue);
                    }
                }
                // Append this formatted entry to the formatted string
                sFormatted.append(sFormat);
            }
            // Move current location to new location
            pLwrrent = pLocation;

            // Search for another special log format entry
            pLocation = strchr(pLwrrent, LOG_FORMAT_ENTRY);
        }
        while (pLocation != NULL);
    }
    else    // No special log format entries
    {
        // Simply set formatted string to format
        sFormatted = pFormat;
    }
    return sFormatted;

} // format

//******************************************************************************

CString
CErrorInfoData::description() const
{
    CString             sTypeName;
    CString             sSubTypeName;
    CString             sAnnotation;
    CString             sDescription(MAX_DESCRIPTION_STRING);

    // Build description from type/subtype/annotation information (if present)
    if (TypeMember().isPresent() && subTypeMember().isPresent())
    {
        // Get error event type and subtype names (SubType/annotation may be empty)
        sTypeName    = typeName();
        sSubTypeName = subTypeName();
        sAnnotation  = annotation();

        // Check for subtype name present
        if (!sSubTypeName.empty())
        {
            sDescription.sprintf("%s (%s)", DML(foreground(sTypeName, RED)), DML(foreground(sSubTypeName, BLACK)));
        }
        else    // No subtype name
        {
            sDescription.sprintf("%s", DML(foreground(sTypeName, RED)));
        }
        // Append annotation (if present)
        if (!sAnnotation.empty())
        {
            sDescription = sDescription + " [" + sAnnotation + "]";
        }
    }
    return sDescription;

} // description

//******************************************************************************

CString
CErrorInfoData::openString
(
    const char         *pOptions
) const
{
    CString             sString(MAX_COMMAND_STRING);
    CString             sAddress(MAX_COMMAND_STRING);
    CString             sOptions(MAX_COMMAND_STRING);
    CString             sOpen;

    // Build the error address string
    sAddress.sprintf("0x%0*I64x", PTR(address()));

    // Build the options string
    if (pOptions != NULL)
    {
        sOptions.sprintf("-a 0x%0*I64x %s", PTR(address()), pOptions);
    }
    else    // No user options
    {
        sOptions.sprintf("-a 0x%0*I64x", PTR(address()));
    }
    // Build the actual DML open string
    sOpen = exec(sAddress, buildDotCommand("open", sOptions));

    return sOpen;

} // openString

//******************************************************************************

CErrorInfoDatas::CErrorInfoDatas
(
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord,
    ULONG               ulRemaining
)
:   m_pErrorInfoOcaRecord(pErrorInfoOcaRecord),
    m_ulErrorDataCount(0),
    m_aErrorInfoDatas(NULL)
{
    assert(pErrorInfoOcaRecord != NULL);

    // Callwlate the maximum number of error info datas (Based on record size and remaining data)
    m_ulErrorDataCount = (min(ulRemaining, pErrorInfoOcaRecord->size()) - pErrorInfoOcaRecord->type().size()) / CErrorInfoData::errorInfoDataType().size();

    // Get the number of error info datas (If valid)
    m_ulErrorDataCount = min(m_ulErrorDataCount, pErrorInfoOcaRecord->count());

    // Allocate the array of error info datas
    m_aErrorInfoDatas = new CErrorInfoDataPtr[m_ulErrorDataCount];

} // CErrorInfoDatas

//******************************************************************************

CErrorInfoDatas::~CErrorInfoDatas()
{

} // ~CErrorInfoDatas

//******************************************************************************

const void*
CErrorInfoDatas::errorData
(
    ULONG               ulErrorInfo
) const
{
    const void         *pErrorData = NULL;

    // Check for invalid error data record
    if (ulErrorInfo >= errorDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid error data record %d (>= %d)",
                         ulErrorInfo, errorDataCount());
    }
    // Compute the requested error info data pointer
    pErrorData = constcharptr(m_pErrorInfoOcaRecord->errorInfoOcaRecord()) + m_pErrorInfoOcaRecord->errorInfoOcaRecordType().size() + (ulErrorInfo * CErrorInfoData::errorInfoDataType().size());

    return pErrorData;

} // errorData

//******************************************************************************

const CErrorInfoData*
CErrorInfoDatas::errorInfoData
(
    ULONG               ulErrorInfo
) const
{
    const void         *pErrorData;
    const CErrorInfoData *pErrorInfoData = NULL;

    // Check for invalid error data record
    if (ulErrorInfo >= errorDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid error data record %d (>= %d)",
                         ulErrorInfo, errorDataCount());
    }
    // Check to see if requested error info data needs to be loaded
    if (m_aErrorInfoDatas[ulErrorInfo] == NULL)
    {
        // Get the error info data pointer
        pErrorData = errorData(ulErrorInfo);

        // Try to create the requested error info data
        m_aErrorInfoDatas[ulErrorInfo] = new CErrorInfoData(pErrorData);
    }
    // Get the requested error info data
    pErrorInfoData = m_aErrorInfoDatas[ulErrorInfo];

    return pErrorInfoData;

} // errorInfoData

//******************************************************************************

CErrorInfoOcaRecord::CErrorInfoOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pErrorInfoOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(frequency),
    INIT(count)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the ERROR_INFO_OCA_RECORD (from data pointer)
    SET(frequency,  m_pErrorInfoOcaRecord);
    SET(count,      m_pErrorInfoOcaRecord);

    // Check for partial ERROR_INFO_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial ERROR_INFO_OCA_RECORD
        setPartial(true);
    }
    // If count is non-zero create the Datas array
    if ((count() != 0) && (countMember().isValid()))
    {
        // Create the Datas member
        m_pErrorInfoDatas = new CErrorInfoDatas(this, ulRemaining);
    }

} // CErrorInfoOcaRecord

//******************************************************************************

CErrorInfoOcaRecord::~CErrorInfoOcaRecord()
{

} // ~CErrorInfoOcaRecord

//******************************************************************************

ULONG
CErrorInfoOcaRecord::size() const
{
    ULONG               ulSize;

    // Set size based on actual ERROR_INFO_OCA_RECORD size and number of error records
    ulSize = type().size() + (count() * CErrorInfoData::errorInfoDataType().size());

    return ulSize;

} // size

//******************************************************************************

CWarningInfoData::CWarningInfoData
(
    const void         *pWarningInfoData
)
:   m_pWarningInfoData(pWarningInfoData),
    INIT(Type),
    INIT(subType),
    INIT(status),
    INIT(address),
    INIT(adapter),
    INIT(device),
    INIT(context),
    INIT(channel),
    INIT(allocation),
    INIT(process),
    INIT(thread),
    INIT(data0),
    INIT(data1),
    INIT(data2),
    INIT(data3),
    INIT(data4),
    INIT(timestamp)
{
    assert(pWarningInfoData != NULL);

    // Set the WARNING_INFO_DATA (from data pointer)
    SET(Type,       pWarningInfoData);
    SET(subType,    pWarningInfoData);
    SET(status,     pWarningInfoData);
    SET(address,    pWarningInfoData);
    SET(adapter,    pWarningInfoData);
    SET(device,     pWarningInfoData);
    SET(context,    pWarningInfoData);
    SET(channel,    pWarningInfoData);
    SET(allocation, pWarningInfoData);
    SET(process,    pWarningInfoData);
    SET(thread,     pWarningInfoData);
    SET(data0,      pWarningInfoData);
    SET(data1,      pWarningInfoData);
    SET(data2,      pWarningInfoData);
    SET(data3,      pWarningInfoData);
    SET(data4,      pWarningInfoData);
    SET(timestamp,  pWarningInfoData);

} // CWarningInfoData

//******************************************************************************

CWarningInfoData::~CWarningInfoData()
{

} // ~CWarningInfoData

//******************************************************************************

CString
CWarningInfoData::statusString() const
{
    CString             sStatus(MAX_COMMAND_STRING);

    // Build the status error string
    sStatus.sprintf("0x%08x", status());

    // Check to see if this is an OS or RM status
    if (typeName().find("_RM_API_") == NOT_FOUND)
    {
        // OS status, build command to display error description (if DML enabled)
        sStatus = exec(sStatus, buildModCommand("error", "ext", sStatus));
    }
    return sStatus;

} // statusString

//******************************************************************************

CString
CWarningInfoData::errorString() const
{
    CString             sError(MAX_COMMAND_STRING);

    // Check to see if this is an OS or RM status
    if (typeName().find("_RM_API_") == NOT_FOUND)
    {
        // OS status, get the OS error description
        sError = ::errorString(status());
    }
    else    // RM status
    {
        // RM status, get the RM error description
        sError = rmString(status());
    }
    return sError;

} // errorString

//******************************************************************************

CString
CWarningInfoData::typeName() const
{
    CString             sTypeName(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Check for warning event type names present
        if (warningEventEnum().isPresent())
        {
            // Try to get the warning event type name
            warningEventEnum().getConstantName(Type(), sTypeName.data(), static_cast<ULONG>(sTypeName.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sTypeName;

} // typeName

//******************************************************************************

CString
CWarningInfoData::subTypeName() const
{
    CString             sTypeName;
    CString             sSubTypeName;

    // Try to get the warning type name (Subtype based on type name)
    sTypeName = typeName();
    if (!sTypeName.empty())
    {
        // Check for a DM interface log type
        if (isDmInterfaceType(sTypeName))
        {
            // Get the DM interface subtype name
            sSubTypeName = dmInterfaceName(subType());
        }
        // Check for a LDDM interface log type
        else if (isLddmInterfaceType(sTypeName))
        {
            // Get the DM interface subtype name
            sSubTypeName = lddmInterfaceName(subType());
        }
        // Check for a DL interface log type
        else if (isDlInterfaceType(sTypeName))
        {
            // Get the DL interface subtype name
            sSubTypeName = dlInterfaceName(subType());
        }
        // Check for a CB interface log type
        else if (isCbInterfaceType(sTypeName))
        {
            // Get the CB interface subtype name
            sSubTypeName = cbInterfaceName(subType());
        }
        // Check for a IFACE interface log type
        else if (isIfaceInterfaceType(sTypeName))
        {
            // Get the IFACE interface subtype name
            sSubTypeName = ifaceInterfaceName(subType());
        }
        // Check for a RM interface log type
        else if (isRmInterfaceType(sTypeName))
        {
            // Get the RM interface subtype name
            sSubTypeName = rmInterfaceName(subType());
        }
        // Check for an AGP interface log type
        else if (isAgpInterfaceType(sTypeName))
        {
            // Get the AGP interface subtype name
            sSubTypeName = agpInterfaceName(subType());
        }
        // Check for a TimedOp interface log type
        else if (isTimedOpInterfaceType(sTypeName))
        {
            // Get the TimedOp interface subtype name
            sSubTypeName = timedOpInterfaceName(subType());
        }
        // Check for a Client Arbitration log type
        else if (isClientArbType(sTypeName))
        {
            // Get the Client Arbitration subtype name
            sSubTypeName = clientArbSubtypeName(subType());
        }
        // Check for a GDI Accelerated operation log type
        else if (isGdiAccelOpType(sTypeName))
        {
            // Get the GDI Accelerated operation type name
            sSubTypeName = gdiAccelOpTypeName(subType());
        }
        // Check for a buffer operation log type
        else if (isBufferOperationType(sTypeName))
        {
            // Get the buffer operation subtype name
            sSubTypeName = bufferOperationName(subType());
        }
        // Check for an idle operation log type
        else if (isIdleOperationType(sTypeName))
        {
            // Get the idle operation subtype name
            sSubTypeName = idleOperationName(subType());
        }
        // Check for an ISR notify log type
        else if (isIsrNotifyType(sTypeName))
        {
            // Get the ISR notify subtype name
            sSubTypeName = isrNotifyName(subType());
        }
        // Check for a build paging buffer log type
        else if (isBuildPagingBufferType(sTypeName))
        {
            // Get the build paging buffer subtype name
            sSubTypeName = buildPagingBufferName(subType());
        }
    }
    return sSubTypeName;

} // subTypeName

//******************************************************************************

CString
CWarningInfoData::annotation() const
{
    CString             sAnnotation;

    // Try to get annotation string if address present
    if (addressMember().isPresent())
    {
        // Try to get the annotation string for this address
        sAnnotation = getAnnotation(address());

        // Check for annotation string present
        if (!sAnnotation.empty())
        {
            // Check DML state (Need to DML escape annotation string)
            if (dmlState())
            {
                // DML escape the annotation string
                sAnnotation = dmlEscape(sAnnotation);
            }
            // Try to catch any annotation format errors
            try
            {
                // Format the annotation string
                sAnnotation = format(sAnnotation);
            }
            catch(...)
            {
                // Check DML state (Set annotation string to red, indicating format error)
                sAnnotation = foreground(sAnnotation, RED);
            }
        }
    }
    return sAnnotation;

} // annotation

//******************************************************************************

CString
CWarningInfoData::format
(
    const char         *pFormat
) const
{
    const char         *pLwrrent = pFormat;
    const char         *pLocation;
    const char         *pSpecifier = NULL;
    ULONG               ulStart;
    ULONG               ulEnd;
    ULONG64             ulDefault = 0;
    ULONG64             ulValue = 0;
    bool                bValid;
    bool                bDefault;
    bool                bSize;
    const CDeviceOcaRecord* pDeviceOcaRecord;
    const CContextOcaRecord* pContextOcaRecord;
    CString             sFormat(MAX_DBGPRINTF_STRING);
    CString             sString(MAX_COMMAND_STRING);
    CString             sSpecifier(MAX_COMMAND_STRING);
    CString             sFormatted;

    assert(pFormat != NULL);

    // Check for special log formatting characters in the format string
    pLocation = strchr(pFormat, LOG_FORMAT_ENTRY);
    if (pLocation != NULL)
    {
        // Loop processing all the special format entries
        do
        {
            // Check for characters before special log format entry
            if (pLocation != pLwrrent)
            {
                // Append these characters onto the formatted string
                sFormatted.append(pLwrrent, 0, (pLocation - pLwrrent));
            }
            // Move past the format entry character
            pLocation++;

            // Default to valid, default format, no size log format entry
            bValid   = true;
            bDefault = true;
            bSize    = false;

            // Check for start/end bit position values
            if (*pLocation == LOG_FORMAT_OPEN)
            {
                // Increment past start of bit position values
                pLocation++;

                // Loop trying to get the start bit position
                ulStart = 0;
                while (*pLocation != EOS)
                {
                    // Check next character for end of start value
                    if ((*pLocation < '0') || (*pLocation > '9'))
                    {
                        // Non-decimal character, end of start value
                        break;
                    }
                    // Add this numeric character into the start value
                    ulStart = (ulStart * 10) + (*pLocation - '0');

                    // Increment to next character
                    pLocation++;
                }
                // Check for end value present
                if (*pLocation == LOG_FORMAT_SEPARATOR)
                {
                    // Increment past bit position separator
                    pLocation++;

                    // Loop trying to get the end bit position
                    ulEnd = 0;
                    while (*pLocation != EOS)
                    {
                        // Check next character for end of end value
                        if ((*pLocation < '0') || (*pLocation > '9'))
                        {
                            // Non-decimal character, end of end value
                            break;
                        }
                        // Add this numeric character into the end value
                        ulEnd = (ulEnd * 10) + (*pLocation - '0');

                        // Increment to next character
                        pLocation++;
                    }
                }
                else    // No end bit position value
                {
                    // Set end bit position to start position (Single bit)
                    ulEnd = ulStart;
                }
                // Check for valid bit position end
                if (*pLocation == LOG_FORMAT_CLOSE)
                {
                    // Increment past the bit position values
                    pLocation++;

                    // Check for invalid bit position values
                    if ((ulStart > ulEnd) || (ulEnd > 63))
                    {
                        // Indicate invalid format entry
                        bValid = false;
                    }
                }
                else    // Invalid bit positions
                {
                    // Indicate invalid format entry
                    bValid = false;
                }
            }
            else    // No bit positions specified
            {
                // Set start and end to full 64-bit value
                ulStart = 0;
                ulEnd   = 63;
            }
            // Check the log format entry
            switch(*pLocation)
            {
                case LOG_FORMAT_TYPE:           // Log entry type value

                    // Skip log format character
                    pLocation++;

                    // Set value to type value
                    ulValue = Type();

                    // Setup typename with error color (if DML enabled)
                    sString = foreground(typeName(), RED);

                    // Setup the default value and format specifier
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_SUBTYPE:        // Log entry subtype value

                    // Skip log format character
                    pLocation++;

                    // Set value to subtype value
                    ulValue = subType();

                    // Setup subtypename with error color (if DML enabled)
                    sString = foreground(subTypeName(), BLUE);

                    // Setup the default value and format specifier
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_STATUS:         // Log entry status value

                    // Skip log format character
                    pLocation++;

                    // Set value to status value
                    ulValue = status();

                    // Setup the default value and format specifier
                    sString    = statusString();
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_TIMESTAMP:      // Log entry timestamp value

                    // Skip log format character
                    pLocation++;

                    // Set value to timestamp value
                    ulValue = timestamp();

                    // Setup the default value and format specifier
                    ulDefault  = timestamp();
                    pSpecifier = "0x%016I64x";

                    break;

                case LOG_FORMAT_ADDRESS:        // Log entry address value

                    // Skip log format character
                    pLocation++;

                    // Set value to address value
                    ulValue = address().ptr();

                    // Setup the default value and format specifier
                    sString    = openString();
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_ADAPTER:        // Log entry adapter value

                    // Skip log format character
                    pLocation++;

                    // Set value to adapter address
                    ulValue = adapter().ptr();

                    // Setup the default adapter value and format specifier
                    ulDefault  = adapter().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_DEVICE:         // Log entry device value

                    // Skip log format character
                    pLocation++;

                    // Set value to device address
                    ulValue = device().ptr();

                    // Setup the default device value and format specifier
                    ulDefault  = device().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_CONTEXT:        // Log entry context value

                    // Skip log format character
                    pLocation++;

                    // Set value to context address
                    ulValue = context().ptr();

                    // Setup the default context value and format specifier
                    ulDefault  = context().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_CHANNEL:        // Log entry channel value

                    // Skip log format character
                    pLocation++;

                    // Set value to channel address
                    ulValue = channel().ptr();

                    // Setup the default channel value and format specifier
                    ulDefault  = channel().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_ALLOCATION:     // Log entry allocation value

                    // Skip log format character
                    pLocation++;

                    // Check for allocation member present
                    if (allocationMember().isPresent())
                    {
                        // Setup the default allocation value and format specifier
                        ulDefault  = allocation().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No allocation member
                    {
                        // Setup the default allocation value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_PROCESS:        // Log entry process value

                    // Skip log format character
                    pLocation++;

                    // Check for process member present
                    if (processMember().isPresent())
                    {
                        // Setup the default process value and format specifier
                        ulDefault  = process().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No process member
                    {
                        // Setup the default process value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_THREAD:         // Log entry thread value

                    // Skip log format character
                    pLocation++;

                    // Check for thread member present
                    if (threadMember().isPresent())
                    {
                        // Setup the default thread value and format specifier
                        ulDefault  = thread().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No thread member
                    {
                        // Setup the default thread value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_KMD_PROCESS:    // Log entry KMD process value

                    // Skip log format character
                    pLocation++;

                    // Setup the default KMD process value and format specifier
                    ulDefault  = 0;
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    // Try to find the OCA record for this device
                    pDeviceOcaRecord = findOcaDevice(device().ptr());
                    if (pDeviceOcaRecord == NULL)
                    {
                        // Try to find the OCA record for this context
                        pContextOcaRecord = findOcaContext(context().ptr());
                        if (pContextOcaRecord != NULL)
                        {
                            // Try too find the OCA record for this context device
                            pDeviceOcaRecord = findOcaDevice(pContextOcaRecord->Device().ptr());
                        }
                    }
                    // Check for device OCA record found
                    if (pDeviceOcaRecord != NULL)
                    {
                        // Get the KMD process value for this record
                        ulDefault = pDeviceOcaRecord->KmdProcess().ptr();
                    }
                    break;

                case LOG_FORMAT_DATA_0:         // Log entry data 0 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 0 value
                    ulValue = data0();

                    // Setup default data 0 value and format specifier
                    ulDefault  = data0();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_1:         // Log entry data 1 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 1 value
                    ulValue = data1();

                    // Setup default data 1 value and format specifier
                    ulDefault  = data1();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_2:         // Log entry data 2 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 2 value
                    ulValue = data2();

                    // Setup default data 2 value and format specifier
                    ulDefault  = data2();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_3:         // Log entry data 3 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 3 value
                    ulValue = data3();

                    // Setup default data 3 value and format specifier
                    ulDefault  = data3();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_4:         // Log entry data 4 value

                    // Skip log format character
                    pLocation++;

                    // Check for data 4 member present
                    if (data4Member().isPresent())
                    {
                        // Setup the default data 4 value and format specifier
                        ulDefault  = data4();
                        pSpecifier = "0x%0I64x";
                        bSize      = false;
                    }
                    else    // No data 4 member
                    {
                        // Setup the default data 4 value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                default:                        // Unknown log entry format specifier

                    // Indicate invalid format entry
                    bValid = false;

                    break;
            }
            // Check for user specified format
            if (*pLocation == PERCENT)
            {
                // Indicate no default format
                bDefault = false;

                // Save user format specifier address
                pSpecifier = pLocation;

                // Increment past format specifier
                pLocation++;

                // Search for end of format specifier
                while (*pLocation != EOS)
                {
                    // Almost any alpha character is the format specifier
                    if (isalpha(*pLocation))
                    {
                        // Only check for I64 format modifier for now
                        if (*pLocation != 'I')
                        {
                            // Skip past format specifier and terminate search
                            pLocation++;

                            break;
                        }
                    }
                    // Check for an asterisk format specifier (Pointer size needed)
                    if (*pLocation == ASTERISK)
                    {
                        bSize = true;
                    }
                    // Otherwise, simply skip this format character
                    pLocation++;
                }
                // Setup the user format specifier
                sSpecifier.assign(pSpecifier, 0, (pLocation - pSpecifier));

                // Setup specifier pointer
                pSpecifier = sSpecifier;
            }
            // Check for valid log format entry found
            if (bValid)
            {
                // Check for default or user specified format
                if (bDefault)
                {
                    // Check for pointer size width required
                    if (bSize)
                    {
                        // Format value with default format and size (Assumes pointer)
                        sFormat.sprintf(pSpecifier, pointerWidth(), (ulDefault & pointerMask()));
                    }
                    else    // No pointer size width required
                    {
                        // Format value with default format
                        sFormat.sprintf(pSpecifier, ulDefault);
                    }
                }
                else    // User specified format
                {
                    // Shift and mask the data value
                    ulValue >>= ulStart;
                    ulValue &= ((1ll << (ulEnd - ulStart)) << 1) - 1;

                    // Check for pointer size width required
                    if (bSize)
                    {
                        // Format value with user format and size (Assumes pointer)
                        sFormat.sprintf(pSpecifier, pointerWidth(), (ulValue & pointerMask()));
                    }
                    else    // No pointer size width required
                    {
                        // Format value with user format
                        sFormat.sprintf(pSpecifier, ulValue);
                    }
                }
                // Append this formatted entry to the formatted string
                sFormatted.append(sFormat);
            }
            // Move current location to new location
            pLwrrent = pLocation;

            // Search for another special log format entry
            pLocation = strchr(pLwrrent, LOG_FORMAT_ENTRY);
        }
        while (pLocation != NULL);
    }
    else    // No special log format entries
    {
        // Simply set formatted string to format
        sFormatted = pFormat;
    }
    return sFormatted;

} // format

//******************************************************************************

CString
CWarningInfoData::description() const
{
    CString             sTypeName;
    CString             sSubTypeName;
    CString             sAnnotation;
    CString             sDescription(MAX_DESCRIPTION_STRING);

    // Build description from type/subtype/annotation information (if present)
    if (TypeMember().isPresent() && subTypeMember().isPresent())
    {
        // Get warning event type and subtype names (SubType/annotation may be empty)
        sTypeName    = typeName();
        sSubTypeName = subTypeName();
        sAnnotation  = annotation();

        // Check for subtype name present
        if (!sSubTypeName.empty())
        {
            sDescription.sprintf("%s (%s)", DML(foreground(sTypeName, RED)), DML(foreground(sSubTypeName, BLACK)));
        }
        else    // No subtype name
        {
            sDescription.sprintf("%s", DML(foreground(sTypeName, RED)));
        }
        // Append annotation (if present)
        if (!sAnnotation.empty())
        {
            sDescription = sDescription + " [" + sAnnotation + "]";
        }
    }
    return sDescription;

} // description

//******************************************************************************

CString
CWarningInfoData::openString
(
    const char         *pOptions
) const
{
    CString             sString(MAX_COMMAND_STRING);
    CString             sAddress(MAX_COMMAND_STRING);
    CString             sOptions(MAX_COMMAND_STRING);
    CString             sOpen;

    // Build the warning address string
    sAddress.sprintf("0x%0*I64x", PTR(address()));

    // Build the options string
    if (pOptions != NULL)
    {
        sOptions.sprintf("-a 0x%0*I64x %s", PTR(address()), pOptions);
    }
    else    // No user options
    {
        sOptions.sprintf("-a 0x%0*I64x", PTR(address()));
    }
    // Build the actual DML open string
    sOpen = exec(sAddress, buildDotCommand("open", sOptions));

    return sOpen;

} // openString

//******************************************************************************

CWarningInfoDatas::CWarningInfoDatas
(
    const CWarningInfoOcaRecord *pWarningInfoOcaRecord,
    ULONG               ulRemaining
)
:   m_pWarningInfoOcaRecord(pWarningInfoOcaRecord),
    m_ulWarningDataCount(0),
    m_aWarningInfoDatas(NULL)
{
    assert(pWarningInfoOcaRecord != NULL);

    // Callwlate the maximum number of warning info datas (Based on record size and remaining data)
    m_ulWarningDataCount = (min(ulRemaining, pWarningInfoOcaRecord->size()) - pWarningInfoOcaRecord->type().size()) / CWarningInfoData::warningInfoDataType().size();

    // Get the number of warning info datas (If valid)
    m_ulWarningDataCount = min(m_ulWarningDataCount, pWarningInfoOcaRecord->count());

    // Allocate the array of warning info datas
    m_aWarningInfoDatas = new CWarningInfoDataPtr[m_ulWarningDataCount];

} // CWarningInfoDatas

//******************************************************************************

CWarningInfoDatas::~CWarningInfoDatas()
{

} // ~CWarningInfoDatas

//******************************************************************************

const void*
CWarningInfoDatas::warningData
(
    ULONG               ulWarningInfo
) const
{
    const void         *pWarningData = NULL;

    // Check for invalid warning data record
    if (ulWarningInfo >= warningDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid warning data record %d (>= %d)",
                         ulWarningInfo, warningDataCount());
    }
    // Compute the requested warning info data pointer
    pWarningData = constcharptr(m_pWarningInfoOcaRecord->warningInfoOcaRecord()) + m_pWarningInfoOcaRecord->warningInfoOcaRecordType().size() + (ulWarningInfo * CWarningInfoData::warningInfoDataType().size());

    return pWarningData;

} // warningData

//******************************************************************************

const CWarningInfoData*
CWarningInfoDatas::warningInfoData
(
    ULONG               ulWarningInfo
) const
{
    const void         *pWarningData;
    const CWarningInfoData *pWarningInfoData = NULL;

    // Check for invalid warning data record
    if (ulWarningInfo >= warningDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid waarning data record %d (>= %d)",
                         ulWarningInfo, warningDataCount());
    }
    // Check to see if requested warning info data needs to be loaded
    if (m_aWarningInfoDatas[ulWarningInfo] == NULL)
    {
        // Get the warning info data pointer
        pWarningData = warningData(ulWarningInfo);

        // Try to create the requested warning info data
        m_aWarningInfoDatas[ulWarningInfo] = new CWarningInfoData(pWarningData);
    }
    // Get the requested warning info data
    pWarningInfoData = m_aWarningInfoDatas[ulWarningInfo];

    return pWarningInfoData;

} // warningInfoData

//******************************************************************************

CWarningInfoOcaRecord::CWarningInfoOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pWarningInfoOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(frequency),
    INIT(count)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the WARNING_INFO_OCA_RECORD (from data pointer)
    SET(frequency,  m_pWarningInfoOcaRecord);
    SET(count,      m_pWarningInfoOcaRecord);

    // Check for partial WARNING_INFO_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial WARNING_INFO_OCA_RECORD
        setPartial(true);
    }
    // If count is non-zero create the Datas array
    if ((count() != 0) && (countMember().isValid()))
    {
        // Create the Datas member
        m_pWarningInfoDatas = new CWarningInfoDatas(this, ulRemaining);
    }

} // CWarningInfoOcaRecord

//******************************************************************************

CWarningInfoOcaRecord::~CWarningInfoOcaRecord()
{

} // ~CWarningInfoOcaRecord

//******************************************************************************

ULONG
CWarningInfoOcaRecord::size() const
{
    ULONG               ulSize;

    // Set size based on actual WARNING_INFO_OCA_RECORD size and number of warning records
    ulSize = type().size() + (count() * CWarningInfoData::warningInfoDataType().size());

    return ulSize;

} // size

//******************************************************************************

CPagingInfoData::CPagingInfoData
(
    const void         *pPagingInfoData
)
:   m_pPagingInfoData(pPagingInfoData),
    INIT(Type),
    INIT(subType),
    INIT(status),
    INIT(address),
    INIT(adapter),
    INIT(device),
    INIT(context),
    INIT(channel),
    INIT(allocation),
    INIT(process),
    INIT(thread),
    INIT(data0),
    INIT(data1),
    INIT(data2),
    INIT(data3),
    INIT(data4),
    INIT(timestamp)
{
    assert(pPagingInfoData != NULL);

    // Set the PAGING_INFO_DATA (from data pointer)
    SET(Type,       pPagingInfoData);
    SET(subType,    pPagingInfoData);
    SET(status,     pPagingInfoData);
    SET(address,    pPagingInfoData);
    SET(adapter,    pPagingInfoData);
    SET(device,     pPagingInfoData);
    SET(context,    pPagingInfoData);
    SET(channel,    pPagingInfoData);
    SET(allocation, pPagingInfoData);
    SET(process,    pPagingInfoData);
    SET(thread,     pPagingInfoData);
    SET(data0,      pPagingInfoData);
    SET(data1,      pPagingInfoData);
    SET(data2,      pPagingInfoData);
    SET(data3,      pPagingInfoData);
    SET(data4,      pPagingInfoData);
    SET(timestamp,  pPagingInfoData);

} // CPagingInfoData

//******************************************************************************

CPagingInfoData::~CPagingInfoData()
{

} // ~CPagingInfoData

//******************************************************************************

CString
CPagingInfoData::statusString() const
{
    CString             sStatus;

    // Get the OS status string
    sStatus = ::statusString(status());

    return sStatus;

} // statusString

//******************************************************************************

CString
CPagingInfoData::errorString() const
{
    CString             sError;

    // Get the OS error string
    sError = ::errorString(status());

    return sError;

} // errorString

//******************************************************************************

CString
CPagingInfoData::typeName() const
{
    CString             sTypeName(MAX_NAME_STRING);

    // Catch any symbol errors
    try
    {
        // Check for paging event type names present
        if (pagingEventEnum().isPresent())
        {
            // Try to get the paging event type name
            pagingEventEnum().getConstantName(Type(), sTypeName.data(), static_cast<ULONG>(sTypeName.capacity()));
        }
    }
    catch (CSymbolException& exception)
    {
        UNREFERENCED_PARAMETER(exception);
    }
    return sTypeName;

} // typeName

//******************************************************************************

CString
CPagingInfoData::subTypeName() const
{
    CString             sTypeName;
    CString             sSubTypeName;

    // Try to get the paging type name (Subtype based on type name)
    sTypeName = typeName();
    if (!sTypeName.empty())
    {
        // Check for a DM interface log type
        if (isDmInterfaceType(sTypeName))
        {
            // Get the DM interface subtype name
            sSubTypeName = dmInterfaceName(subType());
        }
        // Check for a LDDM interface log type
        else if (isLddmInterfaceType(sTypeName))
        {
            // Get the DM interface subtype name
            sSubTypeName = lddmInterfaceName(subType());
        }
        // Check for a DL interface log type
        else if (isDlInterfaceType(sTypeName))
        {
            // Get the DL interface subtype name
            sSubTypeName = dlInterfaceName(subType());
        }
        // Check for a CB interface log type
        else if (isCbInterfaceType(sTypeName))
        {
            // Get the CB interface subtype name
            sSubTypeName = cbInterfaceName(subType());
        }
        // Check for a IFACE interface log type
        else if (isIfaceInterfaceType(sTypeName))
        {
            // Get the IFACE interface subtype name
            sSubTypeName = ifaceInterfaceName(subType());
        }
        // Check for a RM interface log type
        else if (isRmInterfaceType(sTypeName))
        {
            // Get the RM interface subtype name
            sSubTypeName = rmInterfaceName(subType());
        }
        // Check for an AGP interface log type
        else if (isAgpInterfaceType(sTypeName))
        {
            // Get the AGP interface subtype name
            sSubTypeName = agpInterfaceName(subType());
        }
        // Check for a TimedOp interface log type
        else if (isTimedOpInterfaceType(sTypeName))
        {
            // Get the TimedOp interface subtype name
            sSubTypeName = timedOpInterfaceName(subType());
        }
        // Check for a Client Arbitration log type
        else if (isClientArbType(sTypeName))
        {
            // Get the Client Arbitration subtype name
            sSubTypeName = clientArbSubtypeName(subType());
        }
        // Check for a GDI Accelerated operation log type
        else if (isGdiAccelOpType(sTypeName))
        {
            // Get the GDI Accelerated operation type name
            sSubTypeName = gdiAccelOpTypeName(subType());
        }
        // Check for a buffer operation log type
        else if (isBufferOperationType(sTypeName))
        {
            // Get the buffer operation subtype name
            sSubTypeName = bufferOperationName(subType());
        }
        // Check for an idle operation log type
        else if (isIdleOperationType(sTypeName))
        {
            // Get the idle operation subtype name
            sSubTypeName = idleOperationName(subType());
        }
        // Check for an ISR notify log type
        else if (isIsrNotifyType(sTypeName))
        {
            // Get the ISR notify subtype name
            sSubTypeName = isrNotifyName(subType());
        }
        // Check for a build paging buffer log type
        else if (isBuildPagingBufferType(sTypeName))
        {
            // Get the build paging buffer subtype name
            sSubTypeName = buildPagingBufferName(subType());
        }
    }
    return sSubTypeName;

} // subTypeName

//******************************************************************************

CString
CPagingInfoData::annotation() const
{
    CString             sAnnotation;

    // Try to get annotation string if address present
    if (addressMember().isPresent())
    {
        // Try to get the annotation string for this address
        sAnnotation = getAnnotation(address());

        // Check for annotation string present
        if (!sAnnotation.empty())
        {
            // Check DML state (Need to DML escape annotation string)
            if (dmlState())
            {
                // DML escape the annotation string
                sAnnotation = dmlEscape(sAnnotation);
            }
            // Try to catch any annotation format errors
            try
            {
                // Format the annotation string
                sAnnotation = format(sAnnotation);
            }
            catch(...)
            {
                // Check DML state (Set annotation string to red, indicating format error)
                sAnnotation = foreground(sAnnotation, RED);
            }
        }
    }
    return sAnnotation;

} // annotation

//******************************************************************************

CString
CPagingInfoData::format
(
    const char         *pFormat
) const
{
    const char         *pLwrrent = pFormat;
    const char         *pLocation;
    const char         *pSpecifier = NULL;
    ULONG               ulStart;
    ULONG               ulEnd;
    ULONG64             ulDefault = 0;
    ULONG64             ulValue = 0;
    bool                bValid;
    bool                bDefault;
    bool                bSize;
    const CDeviceOcaRecord* pDeviceOcaRecord;
    const CContextOcaRecord* pContextOcaRecord;
    CString             sFormat(MAX_DBGPRINTF_STRING);
    CString             sString(MAX_COMMAND_STRING);
    CString             sSpecifier(MAX_COMMAND_STRING);
    CString             sFormatted;

    assert(pFormat != NULL);

    // Check for special log formatting characters in the format string
    pLocation = strchr(pFormat, LOG_FORMAT_ENTRY);
    if (pLocation != NULL)
    {
        // Loop processing all the special format entries
        do
        {
            // Check for characters before special log format entry
            if (pLocation != pLwrrent)
            {
                // Append these characters onto the formatted string
                sFormatted.append(pLwrrent, 0, (pLocation - pLwrrent));
            }
            // Move past the format entry character
            pLocation++;

            // Default to valid, default format, no size log format entry
            bValid   = true;
            bDefault = true;
            bSize    = false;

            // Check for start/end bit position values
            if (*pLocation == LOG_FORMAT_OPEN)
            {
                // Increment past start of bit position values
                pLocation++;

                // Loop trying to get the start bit position
                ulStart = 0;
                while (*pLocation != EOS)
                {
                    // Check next character for end of start value
                    if ((*pLocation < '0') || (*pLocation > '9'))
                    {
                        // Non-decimal character, end of start value
                        break;
                    }
                    // Add this numeric character into the start value
                    ulStart = (ulStart * 10) + (*pLocation - '0');

                    // Increment to next character
                    pLocation++;
                }
                // Check for end value present
                if (*pLocation == LOG_FORMAT_SEPARATOR)
                {
                    // Increment past bit position separator
                    pLocation++;

                    // Loop trying to get the end bit position
                    ulEnd = 0;
                    while (*pLocation != EOS)
                    {
                        // Check next character for end of end value
                        if ((*pLocation < '0') || (*pLocation > '9'))
                        {
                            // Non-decimal character, end of end value
                            break;
                        }
                        // Add this numeric character into the end value
                        ulEnd = (ulEnd * 10) + (*pLocation - '0');

                        // Increment to next character
                        pLocation++;
                    }
                }
                else    // No end bit position value
                {
                    // Set end bit position to start position (Single bit)
                    ulEnd = ulStart;
                }
                // Check for valid bit position end
                if (*pLocation == LOG_FORMAT_CLOSE)
                {
                    // Increment past the bit position values
                    pLocation++;

                    // Check for invalid bit position values
                    if ((ulStart > ulEnd) || (ulEnd > 63))
                    {
                        // Indicate invalid format entry
                        bValid = false;
                    }
                }
                else    // Invalid bit positions
                {
                    // Indicate invalid format entry
                    bValid = false;
                }
            }
            else    // No bit positions specified
            {
                // Set start and end to full 64-bit value
                ulStart = 0;
                ulEnd   = 63;
            }
            // Check the log format entry
            switch(*pLocation)
            {
                case LOG_FORMAT_TYPE:           // Log entry type value

                    // Skip log format character
                    pLocation++;

                    // Set value to type value
                    ulValue = Type();

                    // Setup typename with error color (if DML enabled)
                    sString = foreground(typeName(), RED);

                    // Setup the default value and format specifier
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_SUBTYPE:        // Log entry subtype value

                    // Skip log format character
                    pLocation++;

                    // Set value to subtype value
                    ulValue = subType();

                    // Setup subtypename with error color (if DML enabled)
                    sString = foreground(subTypeName(), BLUE);

                    // Setup the default value and format specifier
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_STATUS:         // Log entry status value

                    // Skip log format character
                    pLocation++;

                    // Set value to status value
                    ulValue = status();

                    // Setup the default value and format specifier
                    sString    = statusString();
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_TIMESTAMP:      // Log entry timestamp value

                    // Skip log format character
                    pLocation++;

                    // Set value to timestamp value
                    ulValue = timestamp();

                    // Setup the default value and format specifier
                    ulDefault  = timestamp();
                    pSpecifier = "0x%016I64x";

                    break;

                case LOG_FORMAT_ADDRESS:        // Log entry address value

                    // Skip log format character
                    pLocation++;

                    // Set value to address value
                    ulValue = address().ptr();

                    // Setup the default value and format specifier
                    sString    = openString();
                    ulDefault  = reinterpret_cast<ULONG64>(sString.data());
                    pSpecifier = "%s";

                    break;

                case LOG_FORMAT_ADAPTER:        // Log entry adapter value

                    // Skip log format character
                    pLocation++;

                    // Set value to adapter address
                    ulValue = adapter().ptr();

                    // Setup the default adapter value and format specifier
                    ulDefault  = adapter().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_DEVICE:         // Log entry device value

                    // Skip log format character
                    pLocation++;

                    // Set value to device address
                    ulValue = device().ptr();

                    // Setup the default device value and format specifier
                    ulDefault  = device().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_CONTEXT:        // Log entry context value

                    // Skip log format character
                    pLocation++;

                    // Set value to context address
                    ulValue = context().ptr();

                    // Setup the default context value and format specifier
                    ulDefault  = context().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_CHANNEL:        // Log entry channel value

                    // Skip log format character
                    pLocation++;

                    // Set value to channel address
                    ulValue = channel().ptr();

                    // Setup the default channel value and format specifier
                    ulDefault  = channel().ptr();
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    break;

                case LOG_FORMAT_ALLOCATION:     // Log entry allocation value

                    // Skip log format character
                    pLocation++;

                    // Check for allocation member present
                    if (allocationMember().isPresent())
                    {
                        // Setup the default allocation value and format specifier
                        ulDefault  = allocation().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No allocation member
                    {
                        // Setup the default allocation value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_PROCESS:        // Log entry process value

                    // Skip log format character
                    pLocation++;

                    // Check for process member present
                    if (processMember().isPresent())
                    {
                        // Setup the default process value and format specifier
                        ulDefault  = process().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No process member
                    {
                        // Setup the default process value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_THREAD:         // Log entry thread value

                    // Skip log format character
                    pLocation++;

                    // Check for thread member present
                    if (threadMember().isPresent())
                    {
                        // Setup the default thread value and format specifier
                        ulDefault  = thread().ptr();
                        pSpecifier = "0x%0*I64x";
                        bSize      = true;
                    }
                    else    // No thread member
                    {
                        // Setup the default thread value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                case LOG_FORMAT_KMD_PROCESS:    // Log entry KMD process value

                    // Skip log format character
                    pLocation++;

                    // Setup the default KMD process value and format specifier
                    ulDefault  = 0;
                    pSpecifier = "0x%0*I64x";
                    bSize      = true;

                    // Try to find the OCA record for this device
                    pDeviceOcaRecord = findOcaDevice(device().ptr());
                    if (pDeviceOcaRecord == NULL)
                    {
                        // Try to find the OCA record for this context
                        pContextOcaRecord = findOcaContext(context().ptr());
                        if (pContextOcaRecord != NULL)
                        {
                            // Try too find the OCA record for this context device
                            pDeviceOcaRecord = findOcaDevice(pContextOcaRecord->Device().ptr());
                        }
                    }
                    // Check for device OCA record found
                    if (pDeviceOcaRecord != NULL)
                    {
                        // Get the KMD process value for this record
                        ulDefault = pDeviceOcaRecord->KmdProcess().ptr();
                    }
                    break;

                case LOG_FORMAT_DATA_0:         // Log entry data 0 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 0 value
                    ulValue = data0();

                    // Setup default data 0 value and format specifier
                    ulDefault  = data0();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_1:         // Log entry data 1 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 1 value
                    ulValue = data1();

                    // Setup default data 1 value and format specifier
                    ulDefault  = data1();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_2:         // Log entry data 2 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 2 value
                    ulValue = data2();

                    // Setup default data 2 value and format specifier
                    ulDefault  = data2();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_3:         // Log entry data 3 value

                    // Skip log format character
                    pLocation++;

                    // Set value to data 3 value
                    ulValue = data3();

                    // Setup default data 3 value and format specifier
                    ulDefault  = data3();
                    pSpecifier = "0x%0I64x";
                    bSize      = false;

                    break;

                case LOG_FORMAT_DATA_4:         // Log entry data 4 value

                    // Skip log format character
                    pLocation++;

                    // Check for data 4 member present
                    if (data4Member().isPresent())
                    {
                        // Setup the default data 4 value and format specifier
                        ulDefault  = data4();
                        pSpecifier = "0x%0I64x";
                        bSize      = false;
                    }
                    else    // No data 4 member
                    {
                        // Setup the default data 4 value and format specifier
                        pSpecifier = "N/A";
                        bSize      = false;
                    }
                    break;

                default:                        // Unknown log entry format specifier

                    // Indicate invalid format entry
                    bValid = false;

                    break;
            }
            // Check for user specified format
            if (*pLocation == PERCENT)
            {
                // Indicate no default format
                bDefault = false;

                // Save user format specifier address
                pSpecifier = pLocation;

                // Increment past format specifier
                pLocation++;

                // Search for end of format specifier
                while (*pLocation != EOS)
                {
                    // Almost any alpha character is the format specifier
                    if (isalpha(*pLocation))
                    {
                        // Only check for I64 format modifier for now
                        if (*pLocation != 'I')
                        {
                            // Skip past format specifier and terminate search
                            pLocation++;

                            break;
                        }
                    }
                    // Check for an asterisk format specifier (Pointer size needed)
                    if (*pLocation == ASTERISK)
                    {
                        bSize = true;
                    }
                    // Otherwise, simply skip this format character
                    pLocation++;
                }
                // Setup the user format specifier
                sSpecifier.assign(pSpecifier, 0, (pLocation - pSpecifier));

                // Setup specifier pointer
                pSpecifier = sSpecifier;
            }
            // Check for valid log format entry found
            if (bValid)
            {
                // Check for default or user specified format
                if (bDefault)
                {
                    // Check for pointer size width required
                    if (bSize)
                    {
                        // Format value with default format and size (Assumes pointer)
                        sFormat.sprintf(pSpecifier, pointerWidth(), (ulDefault & pointerMask()));
                    }
                    else    // No pointer size width required
                    {
                        // Format value with default format
                        sFormat.sprintf(pSpecifier, ulDefault);
                    }
                }
                else    // User specified format
                {
                    // Shift and mask the data value
                    ulValue >>= ulStart;
                    ulValue &= ((1ll << (ulEnd - ulStart)) << 1) - 1;

                    // Check for pointer size width required
                    if (bSize)
                    {
                        // Format value with user format and size (Assumes pointer)
                        sFormat.sprintf(pSpecifier, pointerWidth(), (ulValue & pointerMask()));
                    }
                    else    // No pointer size width required
                    {
                        // Format value with user format
                        sFormat.sprintf(pSpecifier, ulValue);
                    }
                }
                // Append this formatted entry to the formatted string
                sFormatted.append(sFormat);
            }
            // Move current location to new location
            pLwrrent = pLocation;

            // Search for another special log format entry
            pLocation = strchr(pLwrrent, LOG_FORMAT_ENTRY);
        }
        while (pLocation != NULL);
    }
    else    // No special log format entries
    {
        // Simply set formatted string to format
        sFormatted = pFormat;
    }
    return sFormatted;

} // format

//******************************************************************************

CString
CPagingInfoData::description() const
{
    CString             sTypeName;
    CString             sSubTypeName;
    CString             sAnnotation;
    CString             sDescription(MAX_DESCRIPTION_STRING);

    // Build description from type/subtype/annotation information (if present)
    if (TypeMember().isPresent() && subTypeMember().isPresent())
    {
        // Get paging event type and subtype names (SubType/annotation may be empty)
        sTypeName    = typeName();
        sSubTypeName = subTypeName();
        sAnnotation  = annotation();

        // Check for subtype name present
        if (!sSubTypeName.empty())
        {
            sDescription.sprintf("%s (%s)", DML(foreground(sTypeName, RED)), DML(foreground(sSubTypeName, BLACK)));
        }
        else    // No subtype name
        {
            sDescription.sprintf("%s", DML(foreground(sTypeName, RED)));
        }
        // Append annotation (if present)
        if (!sAnnotation.empty())
        {
            sDescription = sDescription + " [" + sAnnotation + "]";
        }
    }
    return sDescription;

} // description

//******************************************************************************

CString
CPagingInfoData::openString
(
    const char         *pOptions
) const
{
    CString             sString(MAX_COMMAND_STRING);
    CString             sAddress(MAX_COMMAND_STRING);
    CString             sOptions(MAX_COMMAND_STRING);
    CString             sOpen;

    // Build the paging address string
    sAddress.sprintf("0x%0*I64x", PTR(address()));

    // Build the options string
    if (pOptions != NULL)
    {
        sOptions.sprintf("-a 0x%0*I64x %s", PTR(address()), pOptions);
    }
    else    // No user options
    {
        sOptions.sprintf("-a 0x%0*I64x", PTR(address()));
    }
    // Build the actual DML open string
    sOpen = exec(sAddress, buildDotCommand("open", sOptions));

    return sOpen;

} // openString

//******************************************************************************

CPagingInfoDatas::CPagingInfoDatas
(
    const CPagingInfoOcaRecord *pPagingInfoOcaRecord,
    ULONG               ulRemaining
)
:   m_pPagingInfoOcaRecord(pPagingInfoOcaRecord),
    m_ulPagingDataCount(0),
    m_aPagingInfoDatas(NULL)
{
    assert(pPagingInfoOcaRecord != NULL);

    // Callwlate the maximum number of paging info datas (Based on record size and remaining data)
    m_ulPagingDataCount = (min(ulRemaining, pPagingInfoOcaRecord->size()) - pPagingInfoOcaRecord->type().size()) / CPagingInfoData::pagingInfoDataType().size();

    // Get the number of paging info datas (If valid)
    m_ulPagingDataCount = min(m_ulPagingDataCount, pPagingInfoOcaRecord->count());

    // Allocate the array of paging info datas
    m_aPagingInfoDatas = new CPagingInfoDataPtr[m_ulPagingDataCount];

} // CPagingInfoDatas

//******************************************************************************

CPagingInfoDatas::~CPagingInfoDatas()
{

} // ~CPagingInfoDatas

//******************************************************************************

const void*
CPagingInfoDatas::pagingData
(
    ULONG               ulPagingInfo
) const
{
    const void         *pPagingData = NULL;

    // Check for invalid paging data record
    if (ulPagingInfo >= pagingDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid paging data record %d (>= %d)",
                         ulPagingInfo, pagingDataCount());
    }
    // Compute the requested paging info data pointer
    pPagingData = constcharptr(m_pPagingInfoOcaRecord->pagingInfoOcaRecord()) + m_pPagingInfoOcaRecord->pagingInfoOcaRecordType().size() + (ulPagingInfo * CPagingInfoData::pagingInfoDataType().size());

    return pPagingData;

} // pagingData

//******************************************************************************

const CPagingInfoData*
CPagingInfoDatas::pagingInfoData
(
    ULONG               ulPagingInfo
) const
{
    const void         *pPagingData;
    const CPagingInfoData *pPagingInfoData = NULL;

    // Check for invalid paging data record
    if (ulPagingInfo >= pagingDataCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid paging data record %d (>= %d)",
                         ulPagingInfo, pagingDataCount());
    }
    // Check to see if requested paging info data needs to be loaded
    if (m_aPagingInfoDatas[ulPagingInfo] == NULL)
    {
        // Get the paging info data pointer
        pPagingData = pagingData(ulPagingInfo);

        // Try to create the requested paging info data
        m_aPagingInfoDatas[ulPagingInfo] = new CPagingInfoData(pPagingData);
    }
    // Get the requested paging info data
    pPagingInfoData = m_aPagingInfoDatas[ulPagingInfo];

    return pPagingInfoData;

} // pagingInfoData

//******************************************************************************

CPagingInfoOcaRecord::CPagingInfoOcaRecord
(
    const CLwcdRecord  *pLwcdRecord,
    const COcaData     *pOcaData,
    ULONG               ulRemaining
)
:   CLwcdRecord(pLwcdRecord),
    m_pPagingInfoOcaRecord((pLwcdRecord != NULL) ? pLwcdRecord->lwcdRecord() : NULL),
    m_pOcaData(pOcaData),
    INIT(frequency),
    INIT(count)
{
    assert(pLwcdRecord != NULL);
    assert(pOcaData != NULL);

    // Set the PAGING_INFO_OCA_RECORD (from data pointer)
    SET(frequency,  m_pPagingInfoOcaRecord);
    SET(count,      m_pPagingInfoOcaRecord);

    // Check for partial PAGING_INFO_OCA_RECORD
    if (ulRemaining < size())
    {
        // Indicate partial PAGING_INFO_OCA_RECORD
        setPartial(true);
    }
    // If count is non-zero create the Datas array
    if ((count() != 0) && (countMember().isValid()))
    {
        // Create the Datas member
        m_pPagingInfoDatas = new CPagingInfoDatas(this, ulRemaining);
    }

} // CPagingInfoOcaRecord

//******************************************************************************

CPagingInfoOcaRecord::~CPagingInfoOcaRecord()
{

} // ~CPagingInfoOcaRecord

//******************************************************************************

ULONG
CPagingInfoOcaRecord::size() const
{
    ULONG               ulSize;

    // Set size based on actual PAGING_INFO_OCA_RECORD size and number of paging records
    ulSize = type().size() + (count() * CPagingInfoData::pagingInfoDataType().size());

    return ulSize;

} // size

//******************************************************************************

CAdapterOcaRecords::CAdapterOcaRecords
(
    const CDrvOcaRecords *pDrvOcaRecords
)
:   m_pDrvOcaRecords(pDrvOcaRecords),
    m_ulAdapterOcaRecordCount((pDrvOcaRecords != NULL) ? pDrvOcaRecords->adapterOcaRecordCount() : 0),
    m_aDrvOcaIndices(NULL)
{
    const CLwcdRecord  *pLwcdRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulAdapterOcaRecord = 0;

    assert(pDrvOcaRecords != NULL);

    // Check for adapter OCA records present
    if (m_ulAdapterOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulAdapterOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the adapter OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                pLwcdRecord = pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (pLwcdRecord != NULL)
                {
                    // Check to see if this is an adapter OCA record
                    if (pLwcdRecord->cRecordType() == KmdAdapterInfo)
                    {
                        // Save the adapter OCA record index
                        m_aDrvOcaIndices[ulAdapterOcaRecord++] = ulDrvOcaRecord;
                    }
                }
            }
        }
    }

} // CAdapterOcaRecords

//******************************************************************************

CAdapterOcaRecords::~CAdapterOcaRecords()
{

} // ~CAdapterOcaRecords

//******************************************************************************

const CAdapterOcaRecord*
CAdapterOcaRecords::adapterOcaRecord
(
    ULONG               ulAdapterOcaRecord
) const
{
    // Check for invalid adapter OCA index
    if (ulAdapterOcaRecord >= adapterOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid adapter OCA record index %d (>= %d)",
                         ulAdapterOcaRecord, adapterOcaRecordCount());
    }
    // Return the requested adapter OCA record
    return static_cast<const CAdapterOcaRecord*>(m_pDrvOcaRecords->drvOcaRecord(m_aDrvOcaIndices[ulAdapterOcaRecord]));

} // adapterOcaRecord

//******************************************************************************

CDeviceOcaRecords::CDeviceOcaRecords
(
    const CDrvOcaRecords *pDrvOcaRecords
)
:   m_pDrvOcaRecords(pDrvOcaRecords),
    m_ulDeviceOcaRecordCount((pDrvOcaRecords != NULL) ? pDrvOcaRecords->deviceOcaRecordCount() : 0),
    m_aDrvOcaIndices(NULL)
{
    const CLwcdRecord  *pLwcdRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulDeviceOcaRecord = 0;

    assert(pDrvOcaRecords != NULL);

    // Check for device OCA records present
    if (m_ulDeviceOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulDeviceOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the device OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                pLwcdRecord = pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (pLwcdRecord != NULL)
                {
                    // Check to see if this is a device OCA record
                    if ((pLwcdRecord->cRecordType() == KmdDeviceInfo) || (pLwcdRecord->cRecordType() == KmdDeviceInfo_V2))
                    {
                        // Save the device OCA record index
                        m_aDrvOcaIndices[ulDeviceOcaRecord++] = ulDrvOcaRecord;
                    }
                }
            }
        }
    }

} // CDeviceOcaRecords

//******************************************************************************

CDeviceOcaRecords::CDeviceOcaRecords
(
    const CAdapterOcaRecord *pAdapterOcaRecord
)
:   m_pDrvOcaRecords((pAdapterOcaRecord != NULL) ? pAdapterOcaRecord->ocaData()->drvOcaRecords() : NULL),
    m_ulDeviceOcaRecordCount(0),
    m_aDrvOcaIndices(NULL)
{
    ULONG               ulDrvOcaRecord;
    ULONG               ulDeviceOcaRecord = 0;
    union
    {
        const CLwcdRecord*      pLwcdRecord;
        const CDeviceOcaRecord* pDeviceOcaRecord;
    } drvOcaRecord;

    assert(pAdapterOcaRecord != NULL);

    // Loop counting devices for this adapter
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < m_pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver record to check
        drvOcaRecord.pLwcdRecord = m_pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
        if (drvOcaRecord.pLwcdRecord != NULL)
        {
            // Check to see if this is a driver device OCA record
            if ((drvOcaRecord.pLwcdRecord->cRecordType() == KmdDeviceInfo) || (drvOcaRecord.pLwcdRecord->cRecordType() == KmdDeviceInfo_V2))
            {
                // Check to see if this device is for the given adapter
                if (drvOcaRecord.pDeviceOcaRecord->Adapter() == pAdapterOcaRecord->Adapter())
                {
                    // Increment the adapter device count
                    m_ulDeviceOcaRecordCount++;
                }
            }
        }
    }
    // Check for device OCA records found for this adapter
    if (m_ulDeviceOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulDeviceOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the device OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < m_pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                drvOcaRecord.pLwcdRecord = m_pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (drvOcaRecord.pLwcdRecord != NULL)
                {
                    // Check to see if this is a device OCA record
                    if ((drvOcaRecord.pLwcdRecord->cRecordType() == KmdDeviceInfo) || (drvOcaRecord.pLwcdRecord->cRecordType() == KmdDeviceInfo_V2))
                    {
                        // Check to see if this device is for the given adapter
                        if (drvOcaRecord.pDeviceOcaRecord->Adapter() == pAdapterOcaRecord->Adapter())
                        {
                            // Save the device OCA record index
                            m_aDrvOcaIndices[ulDeviceOcaRecord++] = ulDrvOcaRecord;
                        }
                    }
                }
            }
        }
    }

} // CDeviceOcaRecords

//******************************************************************************

CDeviceOcaRecords::~CDeviceOcaRecords()
{

} // ~CDeviceOcaRecords

//******************************************************************************

const CDeviceOcaRecord*
CDeviceOcaRecords::deviceOcaRecord
(
    ULONG               ulDeviceOcaRecord
) const
{
    // Check for invalid device OCA index
    if (ulDeviceOcaRecord >= deviceOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid device OCA record index %d (>= %d)",
                         ulDeviceOcaRecord, deviceOcaRecordCount());
    }
    // Return the requested device OCA record
    return static_cast<const CDeviceOcaRecord*>(m_pDrvOcaRecords->drvOcaRecord(m_aDrvOcaIndices[ulDeviceOcaRecord]));

} // deviceOcaRecord

//******************************************************************************

CContextOcaRecords::CContextOcaRecords
(
    const CDrvOcaRecords *pDrvOcaRecords
)
:   m_pDrvOcaRecords(pDrvOcaRecords),
    m_ulContextOcaRecordCount((pDrvOcaRecords != NULL) ? pDrvOcaRecords->contextOcaRecordCount() : 0),
    m_aDrvOcaIndices(NULL)
{
    const CLwcdRecord  *pLwcdRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulContextOcaRecord = 0;

    assert(pDrvOcaRecords != NULL);

    // Check for context OCA records present
    if (m_ulContextOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulContextOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the context OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                pLwcdRecord = pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (pLwcdRecord != NULL)
                {
                    // Check to see if this is a context OCA record
                    if (pLwcdRecord->cRecordType() == KmdContextInfo)
                    {
                        // Save the context OCA record index
                        m_aDrvOcaIndices[ulContextOcaRecord++] = ulDrvOcaRecord;
                    }
                }
            }
        }
    }

} // CContextOcaRecords

//******************************************************************************

CContextOcaRecords::~CContextOcaRecords()
{

} // ~CContextOcaRecords

//******************************************************************************

const CContextOcaRecord*
CContextOcaRecords::contextOcaRecord
(
    ULONG               ulContextOcaRecord
) const
{
    // Check for invalid context OCA index
    if (ulContextOcaRecord >= contextOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid context OCA record index %d (>= %d)",
                         ulContextOcaRecord, contextOcaRecordCount());
    }
    // Return the requested context OCA record
    return static_cast<const CContextOcaRecord*>(m_pDrvOcaRecords->drvOcaRecord(m_aDrvOcaIndices[ulContextOcaRecord]));

} // contextOcaRecord

//******************************************************************************

CChannelOcaRecords::CChannelOcaRecords
(
    const CDrvOcaRecords *pDrvOcaRecords
)
:   m_pDrvOcaRecords(pDrvOcaRecords),
    m_ulChannelOcaRecordCount((pDrvOcaRecords != NULL) ? pDrvOcaRecords->channelOcaRecordCount() : 0),
    m_aDrvOcaIndices(NULL)
{
    const CLwcdRecord  *pLwcdRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulChannelOcaRecord = 0;

    assert(pDrvOcaRecords != NULL);

    // Check for channel OCA records present
    if (m_ulChannelOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulChannelOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the channel OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                pLwcdRecord = pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (pLwcdRecord != NULL)
                {
                    // Check to see if this is a channel OCA record
                    if (pLwcdRecord->cRecordType() == KmdChannelInfo)
                    {
                        // Save the channel OCA record index
                        m_aDrvOcaIndices[ulChannelOcaRecord++] = ulDrvOcaRecord;
                    }
                }
            }
        }
    }

} // CChannelOcaRecords

//******************************************************************************

CChannelOcaRecords::~CChannelOcaRecords()
{

} // ~CChannelOcaRecords

//******************************************************************************

const CChannelOcaRecord*
CChannelOcaRecords::channelOcaRecord
(
    ULONG               ulChannelOcaRecord
) const
{
    // Check for invalid channel OCA index
    if (ulChannelOcaRecord >= channelOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid channel OCA record index %d (>= %d)",
                         ulChannelOcaRecord, channelOcaRecordCount());
    }
    // Return the requested channel OCA record
    return static_cast<const CChannelOcaRecord*>(m_pDrvOcaRecords->drvOcaRecord(m_aDrvOcaIndices[ulChannelOcaRecord]));

} // channelOcaRecord

//******************************************************************************

CAllocationOcaRecords::CAllocationOcaRecords
(
    const CDrvOcaRecords *pDrvOcaRecords
)
:   m_pDrvOcaRecords(pDrvOcaRecords),
    m_ulAllocationOcaRecordCount((pDrvOcaRecords != NULL) ? pDrvOcaRecords->allocationOcaRecordCount() : 0),
    m_aDrvOcaIndices(NULL)
{
    const CLwcdRecord  *pLwcdRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulAllocationOcaRecord = 0;

    assert(pDrvOcaRecords != NULL);

    // Check for allocation OCA records present
    if (m_ulAllocationOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulAllocationOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the allocation OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                pLwcdRecord = pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (pLwcdRecord != NULL)
                {
                    // Check to see if this is a allocation OCA record
                    if (pLwcdRecord->cRecordType() == KmdAllocationInfo)
                    {
                        // Save the allocation OCA record index
                        m_aDrvOcaIndices[ulAllocationOcaRecord++] = ulDrvOcaRecord;
                    }
                }
            }
        }
    }

} // CAllocationOcaRecords

//******************************************************************************

CAllocationOcaRecords::CAllocationOcaRecords
(
    const CAdapterOcaRecord *pAdapterOcaRecord
)
:   m_pDrvOcaRecords((pAdapterOcaRecord != NULL) ? pAdapterOcaRecord->ocaData()->drvOcaRecords() : NULL),
    m_ulAllocationOcaRecordCount(0),
    m_aDrvOcaIndices(NULL)
{
    ULONG               ulDrvOcaRecord;
    ULONG               ulAdapterIndex = ILWALID_INDEX;
    ULONG               ulAllocationOcaRecord = 0;
    union
    {
        const CLwcdRecord*          pLwcdRecord;
        const CAdapterOcaRecord*    pAdapterOcaRecord;
        const CAllocationOcaRecord* pAllocationOcaRecord;
    } drvOcaRecord;

    assert(pAdapterOcaRecord != NULL);

    // Loop trying to find the given adapter record index (Allocation records are placed after this record)
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < m_pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver record to check
        drvOcaRecord.pLwcdRecord = m_pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
        if (drvOcaRecord.pLwcdRecord != NULL)
        {
            // Check to see if this is a driver adapter OCA record
            if (drvOcaRecord.pLwcdRecord->cRecordType() == KmdAdapterInfo)
            {
                // Check to see if this adapter is the given adapter
                if (drvOcaRecord.pAdapterOcaRecord->Adapter() == pAdapterOcaRecord->Adapter())
                {
                    // Save the adapter record index and exit the search
                    ulAdapterIndex = ulDrvOcaRecord;
                    break;
                }
            }
        }
    }
    // Check to see if we found the given adapter (should be)
    if (ulAdapterIndex != ILWALID_INDEX)
    {
        // Loop counting allocations for this adapter
        for (ulDrvOcaRecord = (ulAdapterIndex + 1); ulDrvOcaRecord < m_pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
        {
            // Get the next driver record to check
            drvOcaRecord.pLwcdRecord = m_pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
            if (drvOcaRecord.pLwcdRecord != NULL)
            {
                // Check to see if we've hit a new adapter record (No more allocations for the given adapter)
                if (drvOcaRecord.pLwcdRecord->cRecordType() == KmdAdapterInfo)
                {
                    // Exit the search loop
                    break;
                }
                // Check to see if this is a driver allocation OCA record
                if (drvOcaRecord.pLwcdRecord->cRecordType() == KmdAllocationInfo)
                {
                    // Increment the adapter allocation count
                    m_ulAllocationOcaRecordCount++;
                }
            }
        }
        // Check for allocation OCA records found for this adapter
        if (m_ulAllocationOcaRecordCount != 0)
        {
            // Try to allocate an array to hold the driver OCA record indices
            m_aDrvOcaIndices = new DWORD[m_ulAllocationOcaRecordCount];
            if (m_aDrvOcaIndices != NULL)
            {
                // Loop building the allocation OCA record indices
                for (ulDrvOcaRecord = ulAdapterIndex + 1; ulDrvOcaRecord < m_pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
                {
                    // Get the next LWCD record to check
                    drvOcaRecord.pLwcdRecord = m_pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                    if (drvOcaRecord.pLwcdRecord != NULL)
                    {
                        // Check to see if we've hit a new adapter record (No more allocations for the given adapter)
                        if (drvOcaRecord.pLwcdRecord->cRecordType() == KmdAdapterInfo)
                        {
                            // Exit the build loop
                            break;
                        }
                        // Check to see if this is an allocation OCA record
                        if (drvOcaRecord.pLwcdRecord->cRecordType() == KmdAllocationInfo)
                        {
                            // Save the allocation OCA record index
                            m_aDrvOcaIndices[ulAllocationOcaRecord++] = ulDrvOcaRecord;
                        }
                    }
                }
            }
        }
    }

} // CAllocationOcaRecords

//******************************************************************************

CAllocationOcaRecords::~CAllocationOcaRecords()
{

} // ~CAllocationOcaRecords

//******************************************************************************

const CAllocationOcaRecord*
CAllocationOcaRecords::allocationOcaRecord
(
    ULONG               ulAllocationOcaRecord
) const
{
    // Check for invalid allocation OCA index
    if (ulAllocationOcaRecord >= allocationOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid allocation OCA record index %d (>= %d)",
                         ulAllocationOcaRecord, allocationOcaRecordCount());
    }
    // Return the requested allocation OCA record
    return static_cast<const CAllocationOcaRecord*>(m_pDrvOcaRecords->drvOcaRecord(m_aDrvOcaIndices[ulAllocationOcaRecord]));

} // allocationOcaRecord

//******************************************************************************

CKmdProcessOcaRecords::CKmdProcessOcaRecords
(
    const CDrvOcaRecords *pDrvOcaRecords
)
:   m_pDrvOcaRecords(pDrvOcaRecords),
    m_ulKmdProcessOcaRecordCount((pDrvOcaRecords != NULL) ? pDrvOcaRecords->kmdProcessOcaRecordCount() : 0),
    m_aDrvOcaIndices(NULL)
{
    const CLwcdRecord  *pLwcdRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulKmdProcessOcaRecord = 0;

    assert(pDrvOcaRecords != NULL);

    // Check for KMD process OCA records present
    if (m_ulKmdProcessOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulKmdProcessOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the KMD process OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                pLwcdRecord = pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (pLwcdRecord != NULL)
                {
                    // Check to see if this is a device OCA record
                    if ((pLwcdRecord->cRecordType() == KmdProcessInfo) || (pLwcdRecord->cRecordType() == KmdProcessInfo_V2))
                    {
                        // Save the KMD process OCA record index
                        m_aDrvOcaIndices[ulKmdProcessOcaRecord++] = ulDrvOcaRecord;
                    }
                }
            }
        }
    }

} // CKmdProcessOcaRecords

//******************************************************************************

CKmdProcessOcaRecords::~CKmdProcessOcaRecords()
{

} // ~CKmdProcessOcaRecords

//******************************************************************************

const CKmdProcessOcaRecord*
CKmdProcessOcaRecords::kmdProcessOcaRecord
(
    ULONG               ulKmdProcessOcaRecord
) const
{
    // Check for invalid KMD process OCA index
    if (ulKmdProcessOcaRecord >= kmdProcessOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid KMD process OCA record index %d (>= %d)",
                         ulKmdProcessOcaRecord, kmdProcessOcaRecordCount());
    }
    // Return the requested KMD process OCA record
    return static_cast<const CKmdProcessOcaRecord*>(m_pDrvOcaRecords->drvOcaRecord(m_aDrvOcaIndices[ulKmdProcessOcaRecord]));

} // kmdProcessOcaRecord

//******************************************************************************

CDmaBufferOcaRecords::CDmaBufferOcaRecords
(
    const CDrvOcaRecords *pDrvOcaRecords
)
:   m_pDrvOcaRecords(pDrvOcaRecords),
    m_ulDmaBufferOcaRecordCount((pDrvOcaRecords != NULL) ? pDrvOcaRecords->dmaBufferOcaRecordCount() : 0),
    m_aDrvOcaIndices(NULL)
{
    const CLwcdRecord  *pLwcdRecord;
    ULONG               ulDrvOcaRecord;
    ULONG               ulDmaBufferOcaRecord = 0;

    assert(pDrvOcaRecords != NULL);

    // Check for DMA buffer OCA records present
    if (m_ulDmaBufferOcaRecordCount != 0)
    {
        // Try to allocate an array to hold the driver OCA record indices
        m_aDrvOcaIndices = new DWORD[m_ulDmaBufferOcaRecordCount];
        if (m_aDrvOcaIndices != NULL)
        {
            // Loop building the DMA buffer OCA record indices
            for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pDrvOcaRecords->drvOcaRecordCount(); ulDrvOcaRecord++)
            {
                // Get the next LWCD record to check
                pLwcdRecord = pDrvOcaRecords->drvOcaRecord(ulDrvOcaRecord);
                if (pLwcdRecord != NULL)
                {
                    // Check to see if this is a channel OCA record
                    if (pLwcdRecord->cRecordType() == KmdDmaBufferInfo)
                    {
                        // Save the DMA buffer OCA record index
                        m_aDrvOcaIndices[ulDmaBufferOcaRecord++] = ulDrvOcaRecord;
                    }
                }
            }
        }
    }

} // CDmaBufferOcaRecords

//******************************************************************************

CDmaBufferOcaRecords::~CDmaBufferOcaRecords()
{

} // ~CDmaBufferOcaRecords

//******************************************************************************

const CDmaBufferOcaRecord*
CDmaBufferOcaRecords::dmaBufferOcaRecord
(
    ULONG               ulDmaBufferOcaRecord
) const
{
    // Check for invalid DMA buffer OCA index
    if (ulDmaBufferOcaRecord >= dmaBufferOcaRecordCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid DMA buffer OCA record index %d (>= %d)",
                         ulDmaBufferOcaRecord, dmaBufferOcaRecordCount());
    }
    // Return the requested DMA buffer OCA record
    return static_cast<const CDmaBufferOcaRecord*>(m_pDrvOcaRecords->drvOcaRecord(m_aDrvOcaIndices[ulDmaBufferOcaRecord]));

} // dmaBufferOcaRecord

//******************************************************************************

const CErrorInfoOcaRecord*
findOcaErrorRecords()
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;    
    const CLwcdRecord  *pDrvOcaRecord;
    const CErrorInfoOcaRecord *pErrorInfoOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for error records
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the error record type
            if (pDrvOcaRecord->cRecordType() == KmdErrorInfo)
            {
                // Save the error record pointer and stop the search
                pErrorInfoOcaRecord = static_cast<const CErrorInfoOcaRecord *>(pDrvOcaRecord);

                break;
            }
        }
    }
    return pErrorInfoOcaRecord;

} // findOcaErrorRecords

//******************************************************************************

const CWarningInfoOcaRecord*
findOcaWarningRecords()
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;    
    const CLwcdRecord  *pDrvOcaRecord;
    const CWarningInfoOcaRecord *pWarningInfoOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for warning records
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the error record type
            if (pDrvOcaRecord->cRecordType() == KmdWarningInfo)
            {
                // Save the warning record pointer and stop the search
                pWarningInfoOcaRecord = static_cast<const CWarningInfoOcaRecord *>(pDrvOcaRecord);

                break;
            }
        }
    }
    return pWarningInfoOcaRecord;

} // findOcaWarningRecords

//******************************************************************************

const CAdapterOcaRecord*
findOcaAdapter
(
    ULONG64             ulAdapter
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    ULONG               ulOcaAdapter = 0;
    const CLwcdRecord  *pDrvOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for adapter record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA adapter type
            if (pDrvOcaRecord->cRecordType() == KmdAdapterInfo)
            {
                // Get the OCA adapter record and check if it is the requested one
                pAdapterOcaRecord = static_cast<const CAdapterOcaRecord *>(pDrvOcaRecord);
                if (pAdapterOcaRecord != NULL)
                {
                    // Check to see if this is the requested adapter
                    if ((pAdapterOcaRecord->Adapter().ptr() == ulAdapter) || (pAdapterOcaRecord->hClient() == ulAdapter) || (ulOcaAdapter == ulAdapter))
                    {
                        // Found the requested OCA adapter, stop the search
                        break;
                    }
                    else    // Not the requested OCA adapter
                    {
                        // Clear the OCA adapter pointer (in case last adapter record)
                        pAdapterOcaRecord = NULL;
                    }
                    // Increment the OCA adapter index
                    ulOcaAdapter++;
                }
            }
        }
    }
    return pAdapterOcaRecord;

} // findOcaAdapter

//******************************************************************************

const CDeviceOcaRecord*
findOcaDevice
(
    ULONG64             ulDevice
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    const CLwcdRecord  *pDrvOcaRecord;
    const CDeviceOcaRecord *pDeviceOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for device record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA device type
            if ((pDrvOcaRecord->cRecordType() == KmdDeviceInfo) || (pDrvOcaRecord->cRecordType() == KmdDeviceInfo_V2))
            {
                // Get the OCA device record and check if it is the requested one
                pDeviceOcaRecord = static_cast<const CDeviceOcaRecord *>(pDrvOcaRecord);
                if (pDeviceOcaRecord != NULL)
                {
                    // Check to see if this is the requested device
                    if ((pDeviceOcaRecord->Device().ptr() == ulDevice) || (pDeviceOcaRecord->hClient() == ulDevice))
                    {
                        // Found the requested OCA device, stop the search
                        break;
                    }
                    else    // Not the requested OCA device
                    {
                        // Clear the OCA device pointer (in case last device record)
                        pDeviceOcaRecord = NULL;
                    }
                }
            }
        }
    }
    return pDeviceOcaRecord;

} // findOcaDevice

//******************************************************************************

const CContextOcaRecord*
findOcaContext
(
    ULONG64             ulContext
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    const CLwcdRecord  *pDrvOcaRecord;
    const CContextOcaRecord *pContextOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for context record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA context type
            if (pDrvOcaRecord->cRecordType() == KmdContextInfo)
            {
                // Get the OCA context record and check if it is the requested one
                pContextOcaRecord = static_cast<const CContextOcaRecord *>(pDrvOcaRecord);
                if (pContextOcaRecord != NULL)
                {
                    // Check to see if this is the requested context
                    if ((pContextOcaRecord->Context().ptr() == ulContext) || (pContextOcaRecord->Channel().ptr() == ulContext))
                    {
                        // Found the requested OCA context, stop the search
                        break;
                    }
                    else    // Not the requested OCA context
                    {
                        // Clear the OCA context pointer (in case last context record)
                        pContextOcaRecord = NULL;
                    }
                }
            }
        }
    }
    return pContextOcaRecord;

} // findOcaContext

//******************************************************************************

const CChannelOcaRecord*
findOcaChannel
(
    ULONG64             ulChannel
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    const CLwcdRecord  *pDrvOcaRecord;
    const CChannelOcaRecord *pChannelOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for channel record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA channel type
            if (pDrvOcaRecord->cRecordType() == KmdChannelInfo)
            {
                // Get the OCA channel record and check if it is the requested one
                pChannelOcaRecord = static_cast<const CChannelOcaRecord *>(pDrvOcaRecord);
                if (pChannelOcaRecord != NULL)
                {
                    // Check to see if this is the requested channel
                    if ((pChannelOcaRecord->Channel().ptr() == ulChannel) || (pChannelOcaRecord->hClient() == ulChannel))
                    {
                        // Found the requested OCA channel, stop the search
                        break;
                    }
                    else    // Not the requested OCA channel
                    {
                        // Clear the OCA channel pointer (in case last channel record)
                        pChannelOcaRecord = NULL;
                    }
                }
            }
        }
    }
    return pChannelOcaRecord;

} // findOcaChannel

//******************************************************************************

const CKmdProcessOcaRecord*
findOcaProcess
(
    ULONG64             ulProcess
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    const CLwcdRecord  *pDrvOcaRecord;
    const CKmdProcessOcaRecord *pKmdProcessOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for process record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA process type
            if ((pDrvOcaRecord->cRecordType() == KmdProcessInfo) || (pDrvOcaRecord->cRecordType() == KmdProcessInfo_V2))
            {
                // Get the OCA process record and check if it is the requested one
                pKmdProcessOcaRecord = static_cast<const CKmdProcessOcaRecord *>(pDrvOcaRecord);
                if (pKmdProcessOcaRecord != NULL)
                {
                    // Check to see if this is the requested process
                    if ((pKmdProcessOcaRecord->Process().ptr() == ulProcess) || (pKmdProcessOcaRecord->hClient() == ulProcess))
                    {
                        // Found the requested OCA process, stop the search
                        break;
                    }
                    else    // Not the requested OCA process
                    {
                        // Clear the OCA process pointer (in case last process record)
                        pKmdProcessOcaRecord = NULL;
                    }
                }
            }
        }
    }
    return pKmdProcessOcaRecord;

} // findOcaProcess

//******************************************************************************

const CKmdProcessOcaRecord*
findOcaKmdProcess
(
    ULONG64             ulKmdProcess
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    const CLwcdRecord  *pDrvOcaRecord;
    const CKmdProcessOcaRecord *pKmdProcessOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for process record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the KMD process type
            if ((pDrvOcaRecord->cRecordType() == KmdProcessInfo) || (pDrvOcaRecord->cRecordType() == KmdProcessInfo_V2))
            {
                // Get the OCA process record and check if it is the requested one
                pKmdProcessOcaRecord = static_cast<const CKmdProcessOcaRecord *>(pDrvOcaRecord);
                if (pKmdProcessOcaRecord != NULL)
                {
                    // Check to see if this is the requested process
                    if (pKmdProcessOcaRecord->KmdProcess().ptr() == ulKmdProcess)
                    {
                        // Found the requested KMD process, stop the search
                        break;
                    }
                    else    // Not the requested KMD process
                    {
                        // Clear the KMD process pointer (in case last process record)
                        pKmdProcessOcaRecord = NULL;
                    }
                }
            }
        }
    }
    return pKmdProcessOcaRecord;

} // findOcaKmdProcess

//******************************************************************************

const CEngineIdOcaRecord*
findEngineId
(
    ULONG64             ulAdapter,
    ULONG               ulEngineId
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    ULONG               ulNextOcaRecord;
    ULONG               ulOcaAdapter = 0;
    ULONG               ulId = 0;
    const CLwcdRecord  *pDrvOcaRecord;
    const CLwcdRecord  *pNextOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    const CEngineIdOcaRecord *pEngineIdOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for adapter record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA adapter type
            if (pDrvOcaRecord->cRecordType() == KmdAdapterInfo)
            {
                // Get the OCA adapter record and check if it is the requested one
                pAdapterOcaRecord = static_cast<const CAdapterOcaRecord *>(pDrvOcaRecord);
                if (pAdapterOcaRecord != NULL)
                {
                    // Check to see if this is the requested adapter
                    if ((pAdapterOcaRecord->Adapter().ptr() == ulAdapter) || (ulOcaAdapter == ulAdapter))
                    {
                        // Search next records for requested engine ID record
                        for (ulNextOcaRecord = (ulDrvOcaRecord + 1); ulNextOcaRecord < pOcaData->drvOcaRecordCount(); ulNextOcaRecord++)
                        {
                            // Get the next driver OCA record
                            pNextOcaRecord = pOcaData->drvOcaRecord(ulNextOcaRecord);
                            if (pNextOcaRecord != NULL)
                            {
                                // Check the driver record for the OCA engine ID type
                                if (pNextOcaRecord->cRecordType() == KmdEngineIdInfo)
                                {
                                    // Check to see if this is the requested engine ID
                                    if (ulEngineId == ulId)
                                    {
                                        // Get the OCA engine ID record
                                        pEngineIdOcaRecord = static_cast<const CEngineIdOcaRecord *>(pNextOcaRecord);

                                        // Found requested engine ID for this adapter, stop search
                                        break;
                                    }
                                    // Increment the engine ID
                                    ulId++;
                                }
                            }
                        }
                        // Exit since we've already checked the correct adapter
                        break;
                    }
                    // Increment the OCA adapter index
                    ulOcaAdapter++;
                }
            }
        }
    }
    return pEngineIdOcaRecord;

} // findEngineId

const CEngineIdOcaRecord*
findEngineId
(
    ULONG64             ulAdapter,
    ULONG               ulNodeOrdinal,
    ULONG               ulEngineOrdinal
)
{
    const COcaDataPtr   pOcaData = ocaData();
    ULONG               ulDrvOcaRecord;
    ULONG               ulNextOcaRecord;
    ULONG               ulOcaAdapter = 0;
    const CLwcdRecord  *pDrvOcaRecord;
    const CLwcdRecord  *pNextOcaRecord;
    const CAdapterOcaRecord *pAdapterOcaRecord;
    const CEngineIdOcaRecord *pEngineIdOcaRecord = NULL;

    // Loop thru the driver (KMD) OCA records looking for adapter record
    for (ulDrvOcaRecord = 0; ulDrvOcaRecord < pOcaData->drvOcaRecordCount(); ulDrvOcaRecord++)
    {
        // Get the next driver OCA record
        pDrvOcaRecord = pOcaData->drvOcaRecord(ulDrvOcaRecord);
        if (pDrvOcaRecord != NULL)
        {
            // Check the driver record for the OCA adapter type
            if (pDrvOcaRecord->cRecordType() == KmdAdapterInfo)
            {
                // Get the OCA adapter record and check if it is the requested one
                pAdapterOcaRecord = static_cast<const CAdapterOcaRecord *>(pDrvOcaRecord);
                if (pAdapterOcaRecord != NULL)
                {
                    // Check to see if this is the requested adapter
                    if ((pAdapterOcaRecord->Adapter().ptr() == ulAdapter) || (ulOcaAdapter == ulAdapter))
                    {
                        // Search next records for requested engine ID record
                        for (ulNextOcaRecord = (ulDrvOcaRecord + 1); ulNextOcaRecord < pOcaData->drvOcaRecordCount(); ulNextOcaRecord++)
                        {
                            // Get the next driver OCA record
                            pNextOcaRecord = pOcaData->drvOcaRecord(ulNextOcaRecord);
                            if (pNextOcaRecord != NULL)
                            {
                                // Check the driver record for the OCA engine ID type
                                if (pNextOcaRecord->cRecordType() == KmdEngineIdInfo)
                                {
                                    // Get the OCA engine ID record to check
                                    pEngineIdOcaRecord = static_cast<const CEngineIdOcaRecord *>(pNextOcaRecord);
                                    if (pEngineIdOcaRecord != NULL)
                                    {
                                        // Check to see if this is the requested node/engine
                                        if ((pEngineIdOcaRecord->NodeOrdinal() == ulNodeOrdinal) && (pEngineIdOcaRecord->EngineOrdinal() == ulEngineOrdinal))
                                        {
                                            // Found requested engine ID for this adapter, stop search
                                            break;
                                        }
                                        else    // Not the requested engine ID record
                                        {
                                            // Clear the engine ID OCA record
                                            pEngineIdOcaRecord = NULL;
                                        }
                                    }
                                }
                            }
                        }
                        // Exit since we've already checked the correct adapter
                        break;
                    }
                    // Increment the OCA adapter index
                    ulOcaAdapter++;
                }
            }
        }
    }
    return pEngineIdOcaRecord;

} // findEngineId

} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
