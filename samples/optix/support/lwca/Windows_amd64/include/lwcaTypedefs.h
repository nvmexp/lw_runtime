#ifndef LWDATYPEDEFS_H
#define LWDATYPEDEFS_H

#include <lwca.h>

#if defined(LWDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptds
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptsz
#else
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## default_version
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## default_version
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwca.h
 */
#define PFN_lwGetErrorString  PFN_lwGetErrorString_v6000
#define PFN_lwGetErrorName  PFN_lwGetErrorName_v6000
#define PFN_lwInit  PFN_lwInit_v2000
#define PFN_lwDriverGetVersion  PFN_lwDriverGetVersion_v2020
#define PFN_lwDeviceGet  PFN_lwDeviceGet_v2000
#define PFN_lwDeviceGetCount  PFN_lwDeviceGetCount_v2000
#define PFN_lwDeviceGetName  PFN_lwDeviceGetName_v2000
#define PFN_lwDeviceGetUuid  PFN_lwDeviceGetUuid_v9020
#define PFN_lwDeviceGetLuid  PFN_lwDeviceGetLuid_v10000
#define PFN_lwDeviceTotalMem  PFN_lwDeviceTotalMem_v3020
#define PFN_lwDeviceGetTexture1DLinearMaxWidth  PFN_lwDeviceGetTexture1DLinearMaxWidth_v11010
#define PFN_lwDeviceGetAttribute  PFN_lwDeviceGetAttribute_v2000
#define PFN_lwDeviceGetLwSciSyncAttributes  PFN_lwDeviceGetLwSciSyncAttributes_v10020
#define PFN_lwDeviceSetMemPool  PFN_lwDeviceSetMemPool_v11020
#define PFN_lwDeviceGetMemPool  PFN_lwDeviceGetMemPool_v11020
#define PFN_lwDeviceGetDefaultMemPool  PFN_lwDeviceGetDefaultMemPool_v11020
#define PFN_lwDeviceGetProperties  PFN_lwDeviceGetProperties_v2000
#define PFN_lwDeviceComputeCapability  PFN_lwDeviceComputeCapability_v2000
#define PFN_lwDevicePrimaryCtxRetain  PFN_lwDevicePrimaryCtxRetain_v7000
#define PFN_lwDevicePrimaryCtxRelease  PFN_lwDevicePrimaryCtxRelease_v11000
#define PFN_lwDevicePrimaryCtxSetFlags  PFN_lwDevicePrimaryCtxSetFlags_v11000
#define PFN_lwDevicePrimaryCtxGetState  PFN_lwDevicePrimaryCtxGetState_v7000
#define PFN_lwDevicePrimaryCtxReset  PFN_lwDevicePrimaryCtxReset_v11000
#define PFN_lwCtxCreate  PFN_lwCtxCreate_v3020
#define PFN_lwCtxDestroy  PFN_lwCtxDestroy_v4000
#define PFN_lwCtxPushLwrrent  PFN_lwCtxPushLwrrent_v4000
#define PFN_lwCtxPopLwrrent  PFN_lwCtxPopLwrrent_v4000
#define PFN_lwCtxSetLwrrent  PFN_lwCtxSetLwrrent_v4000
#define PFN_lwCtxGetLwrrent  PFN_lwCtxGetLwrrent_v4000
#define PFN_lwCtxGetDevice  PFN_lwCtxGetDevice_v2000
#define PFN_lwCtxGetFlags  PFN_lwCtxGetFlags_v7000
#define PFN_lwCtxSynchronize  PFN_lwCtxSynchronize_v2000
#define PFN_lwCtxSetLimit  PFN_lwCtxSetLimit_v3010
#define PFN_lwCtxGetLimit  PFN_lwCtxGetLimit_v3010
#define PFN_lwCtxGetCacheConfig  PFN_lwCtxGetCacheConfig_v3020
#define PFN_lwCtxSetCacheConfig  PFN_lwCtxSetCacheConfig_v3020
#define PFN_lwCtxGetSharedMemConfig  PFN_lwCtxGetSharedMemConfig_v4020
#define PFN_lwCtxSetSharedMemConfig  PFN_lwCtxSetSharedMemConfig_v4020
#define PFN_lwCtxGetApiVersion  PFN_lwCtxGetApiVersion_v3020
#define PFN_lwCtxGetStreamPriorityRange  PFN_lwCtxGetStreamPriorityRange_v5050
#define PFN_lwCtxResetPersistingL2Cache  PFN_lwCtxResetPersistingL2Cache_v11000
#define PFN_lwCtxAttach  PFN_lwCtxAttach_v2000
#define PFN_lwCtxDetach  PFN_lwCtxDetach_v2000
#define PFN_lwModuleLoad  PFN_lwModuleLoad_v2000
#define PFN_lwModuleLoadData  PFN_lwModuleLoadData_v2000
#define PFN_lwModuleLoadDataEx  PFN_lwModuleLoadDataEx_v2010
#define PFN_lwModuleLoadFatBinary  PFN_lwModuleLoadFatBinary_v2000
#define PFN_lwModuleUnload  PFN_lwModuleUnload_v2000
#define PFN_lwModuleGetFunction  PFN_lwModuleGetFunction_v2000
#define PFN_lwModuleGetGlobal  PFN_lwModuleGetGlobal_v3020
#define PFN_lwModuleGetTexRef  PFN_lwModuleGetTexRef_v2000
#define PFN_lwModuleGetSurfRef  PFN_lwModuleGetSurfRef_v3000
#define PFN_lwLinkCreate  PFN_lwLinkCreate_v6050
#define PFN_lwLinkAddData  PFN_lwLinkAddData_v6050
#define PFN_lwLinkAddFile  PFN_lwLinkAddFile_v6050
#define PFN_lwLinkComplete  PFN_lwLinkComplete_v5050
#define PFN_lwLinkDestroy  PFN_lwLinkDestroy_v5050
#define PFN_lwMemGetInfo  PFN_lwMemGetInfo_v3020
#define PFN_lwMemAlloc  PFN_lwMemAlloc_v3020
#define PFN_lwMemAllocPitch  PFN_lwMemAllocPitch_v3020
#define PFN_lwMemFree  PFN_lwMemFree_v3020
#define PFN_lwMemGetAddressRange  PFN_lwMemGetAddressRange_v3020
#define PFN_lwMemAllocHost  PFN_lwMemAllocHost_v3020
#define PFN_lwMemFreeHost  PFN_lwMemFreeHost_v2000
#define PFN_lwMemHostAlloc  PFN_lwMemHostAlloc_v2020
#define PFN_lwMemHostGetDevicePointer  PFN_lwMemHostGetDevicePointer_v3020
#define PFN_lwMemHostGetFlags  PFN_lwMemHostGetFlags_v2030
#define PFN_lwMemAllocManaged  PFN_lwMemAllocManaged_v6000
#define PFN_lwDeviceGetByPCIBusId  PFN_lwDeviceGetByPCIBusId_v4010
#define PFN_lwDeviceGetPCIBusId  PFN_lwDeviceGetPCIBusId_v4010
#define PFN_lwIpcGetEventHandle  PFN_lwIpcGetEventHandle_v4010
#define PFN_lwIpcOpenEventHandle  PFN_lwIpcOpenEventHandle_v4010
#define PFN_lwIpcGetMemHandle  PFN_lwIpcGetMemHandle_v4010
#define PFN_lwIpcOpenMemHandle  PFN_lwIpcOpenMemHandle_v11000
#define PFN_lwIpcCloseMemHandle  PFN_lwIpcCloseMemHandle_v4010
#define PFN_lwMemHostRegister  PFN_lwMemHostRegister_v6050
#define PFN_lwMemHostUnregister  PFN_lwMemHostUnregister_v4000
#define PFN_lwMemcpy  __API_TYPEDEF_PTDS(PFN_lwMemcpy, 4000, 7000)
#define PFN_lwMemcpyPeer  __API_TYPEDEF_PTDS(PFN_lwMemcpyPeer, 4000, 7000)
#define PFN_lwMemcpyHtoD  __API_TYPEDEF_PTDS(PFN_lwMemcpyHtoD, 3020, 7000)
#define PFN_lwMemcpyDtoH  __API_TYPEDEF_PTDS(PFN_lwMemcpyDtoH, 3020, 7000)
#define PFN_lwMemcpyDtoD  __API_TYPEDEF_PTDS(PFN_lwMemcpyDtoD, 3020, 7000)
#define PFN_lwMemcpyDtoA  __API_TYPEDEF_PTDS(PFN_lwMemcpyDtoA, 3020, 7000)
#define PFN_lwMemcpyAtoD  __API_TYPEDEF_PTDS(PFN_lwMemcpyAtoD, 3020, 7000)
#define PFN_lwMemcpyHtoA  __API_TYPEDEF_PTDS(PFN_lwMemcpyHtoA, 3020, 7000)
#define PFN_lwMemcpyAtoH  __API_TYPEDEF_PTDS(PFN_lwMemcpyAtoH, 3020, 7000)
#define PFN_lwMemcpyAtoA  __API_TYPEDEF_PTDS(PFN_lwMemcpyAtoA, 3020, 7000)
#define PFN_lwMemcpy2D  __API_TYPEDEF_PTDS(PFN_lwMemcpy2D, 3020, 7000)
#define PFN_lwMemcpy2DUnaligned  __API_TYPEDEF_PTDS(PFN_lwMemcpy2DUnaligned, 3020, 7000)
#define PFN_lwMemcpy3D  __API_TYPEDEF_PTDS(PFN_lwMemcpy3D, 3020, 7000)
#define PFN_lwMemcpy3DPeer  __API_TYPEDEF_PTDS(PFN_lwMemcpy3DPeer, 4000, 7000)
#define PFN_lwMemcpyAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpyAsync, 4000, 7000)
#define PFN_lwMemcpyPeerAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpyPeerAsync, 4000, 7000)
#define PFN_lwMemcpyHtoDAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpyHtoDAsync, 3020, 7000)
#define PFN_lwMemcpyDtoHAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpyDtoHAsync, 3020, 7000)
#define PFN_lwMemcpyDtoDAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpyDtoDAsync, 3020, 7000)
#define PFN_lwMemcpyHtoAAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpyHtoAAsync, 3020, 7000)
#define PFN_lwMemcpyAtoHAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpyAtoHAsync, 3020, 7000)
#define PFN_lwMemcpy2DAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpy2DAsync, 3020, 7000)
#define PFN_lwMemcpy3DAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpy3DAsync, 3020, 7000)
#define PFN_lwMemcpy3DPeerAsync  __API_TYPEDEF_PTSZ(PFN_lwMemcpy3DPeerAsync, 4000, 7000)
#define PFN_lwMemsetD8  __API_TYPEDEF_PTDS(PFN_lwMemsetD8, 3020, 7000)
#define PFN_lwMemsetD16  __API_TYPEDEF_PTDS(PFN_lwMemsetD16, 3020, 7000)
#define PFN_lwMemsetD32  __API_TYPEDEF_PTDS(PFN_lwMemsetD32, 3020, 7000)
#define PFN_lwMemsetD2D8  __API_TYPEDEF_PTDS(PFN_lwMemsetD2D8, 3020, 7000)
#define PFN_lwMemsetD2D16  __API_TYPEDEF_PTDS(PFN_lwMemsetD2D16, 3020, 7000)
#define PFN_lwMemsetD2D32  __API_TYPEDEF_PTDS(PFN_lwMemsetD2D32, 3020, 7000)
#define PFN_lwMemsetD8Async  __API_TYPEDEF_PTSZ(PFN_lwMemsetD8Async, 3020, 7000)
#define PFN_lwMemsetD16Async  __API_TYPEDEF_PTSZ(PFN_lwMemsetD16Async, 3020, 7000)
#define PFN_lwMemsetD32Async  __API_TYPEDEF_PTSZ(PFN_lwMemsetD32Async, 3020, 7000)
#define PFN_lwMemsetD2D8Async  __API_TYPEDEF_PTSZ(PFN_lwMemsetD2D8Async, 3020, 7000)
#define PFN_lwMemsetD2D16Async  __API_TYPEDEF_PTSZ(PFN_lwMemsetD2D16Async, 3020, 7000)
#define PFN_lwMemsetD2D32Async  __API_TYPEDEF_PTSZ(PFN_lwMemsetD2D32Async, 3020, 7000)
#define PFN_lwArrayCreate  PFN_lwArrayCreate_v3020
#define PFN_lwArrayGetDescriptor  PFN_lwArrayGetDescriptor_v3020
#define PFN_lwArrayGetSparseProperties  PFN_lwArrayGetSparseProperties_v11010
#define PFN_lwMipmappedArrayGetSparseProperties  PFN_lwMipmappedArrayGetSparseProperties_v11010
#define PFN_lwArrayGetPlane  PFN_lwArrayGetPlane_v11020
#define PFN_lwArrayDestroy  PFN_lwArrayDestroy_v2000
#define PFN_lwArray3DCreate  PFN_lwArray3DCreate_v3020
#define PFN_lwArray3DGetDescriptor  PFN_lwArray3DGetDescriptor_v3020
#define PFN_lwMipmappedArrayCreate  PFN_lwMipmappedArrayCreate_v5000
#define PFN_lwMipmappedArrayGetLevel  PFN_lwMipmappedArrayGetLevel_v5000
#define PFN_lwMipmappedArrayDestroy  PFN_lwMipmappedArrayDestroy_v5000
#define PFN_lwMemAddressReserve  PFN_lwMemAddressReserve_v10020
#define PFN_lwMemAddressFree  PFN_lwMemAddressFree_v10020
#define PFN_lwMemCreate  PFN_lwMemCreate_v10020
#define PFN_lwMemRelease  PFN_lwMemRelease_v10020
#define PFN_lwMemMap  PFN_lwMemMap_v10020
#define PFN_lwMemMapArrayAsync  __API_TYPEDEF_PTSZ(PFN_lwMemMapArrayAsync, 11010, 11010)
#define PFN_lwMemUnmap  PFN_lwMemUnmap_v10020
#define PFN_lwMemSetAccess  PFN_lwMemSetAccess_v10020
#define PFN_lwMemGetAccess  PFN_lwMemGetAccess_v10020
#define PFN_lwMemExportToShareableHandle  PFN_lwMemExportToShareableHandle_v10020
#define PFN_lwMemImportFromShareableHandle  PFN_lwMemImportFromShareableHandle_v10020
#define PFN_lwMemGetAllocationGranularity  PFN_lwMemGetAllocationGranularity_v10020
#define PFN_lwMemGetAllocationPropertiesFromHandle  PFN_lwMemGetAllocationPropertiesFromHandle_v10020
#define PFN_lwMemRetainAllocationHandle  PFN_lwMemRetainAllocationHandle_v11000
#define PFN_lwMemFreeAsync  __API_TYPEDEF_PTSZ(PFN_lwMemFreeAsync, 11020, 11020)
#define PFN_lwMemAllocAsync  __API_TYPEDEF_PTSZ(PFN_lwMemAllocAsync, 11020, 11020)
#define PFN_lwMemPoolTrimTo  PFN_lwMemPoolTrimTo_v11020
#define PFN_lwMemPoolSetAttribute  PFN_lwMemPoolSetAttribute_v11020
#define PFN_lwMemPoolGetAttribute  PFN_lwMemPoolGetAttribute_v11020
#define PFN_lwMemPoolSetAccess  PFN_lwMemPoolSetAccess_v11020
#define PFN_lwMemPoolGetAccess  PFN_lwMemPoolGetAccess_v11020
#define PFN_lwMemPoolCreate  PFN_lwMemPoolCreate_v11020
#define PFN_lwMemPoolDestroy  PFN_lwMemPoolDestroy_v11020
#define PFN_lwMemAllocFromPoolAsync  __API_TYPEDEF_PTSZ(PFN_lwMemAllocFromPoolAsync, 11020, 11020)
#define PFN_lwMemPoolExportToShareableHandle  PFN_lwMemPoolExportToShareableHandle_v11020
#define PFN_lwMemPoolImportFromShareableHandle  PFN_lwMemPoolImportFromShareableHandle_v11020
#define PFN_lwMemPoolExportPointer  PFN_lwMemPoolExportPointer_v11020
#define PFN_lwMemPoolImportPointer  PFN_lwMemPoolImportPointer_v11020
#define PFN_lwPointerGetAttribute  PFN_lwPointerGetAttribute_v4000
#define PFN_lwMemPrefetchAsync  __API_TYPEDEF_PTSZ(PFN_lwMemPrefetchAsync, 8000, 8000)
#define PFN_lwMemAdvise  PFN_lwMemAdvise_v8000
#define PFN_lwMemRangeGetAttribute  PFN_lwMemRangeGetAttribute_v8000
#define PFN_lwMemRangeGetAttributes  PFN_lwMemRangeGetAttributes_v8000
#define PFN_lwPointerSetAttribute  PFN_lwPointerSetAttribute_v6000
#define PFN_lwPointerGetAttributes  PFN_lwPointerGetAttributes_v7000
#define PFN_lwStreamCreate  PFN_lwStreamCreate_v2000
#define PFN_lwStreamCreateWithPriority  PFN_lwStreamCreateWithPriority_v5050
#define PFN_lwStreamGetPriority  __API_TYPEDEF_PTSZ(PFN_lwStreamGetPriority, 5050, 7000)
#define PFN_lwStreamGetFlags  __API_TYPEDEF_PTSZ(PFN_lwStreamGetFlags, 5050, 7000)
#define PFN_lwStreamGetCtx  __API_TYPEDEF_PTSZ(PFN_lwStreamGetCtx, 9020, 9020)
#define PFN_lwStreamWaitEvent  __API_TYPEDEF_PTSZ(PFN_lwStreamWaitEvent, 3020, 7000)
#define PFN_lwStreamAddCallback  __API_TYPEDEF_PTSZ(PFN_lwStreamAddCallback, 5000, 7000)
#define PFN_lwStreamBeginCapture  __API_TYPEDEF_PTSZ(PFN_lwStreamBeginCapture, 10010, 10010)
#define PFN_lwThreadExchangeStreamCaptureMode  PFN_lwThreadExchangeStreamCaptureMode_v10010
#define PFN_lwStreamEndCapture  __API_TYPEDEF_PTSZ(PFN_lwStreamEndCapture, 10000, 10000)
#define PFN_lwStreamIsCapturing  __API_TYPEDEF_PTSZ(PFN_lwStreamIsCapturing, 10000, 10000)
#define PFN_lwStreamGetCaptureInfo  __API_TYPEDEF_PTSZ(PFN_lwStreamGetCaptureInfo, 10010, 10010)
#define PFN_lwStreamGetCaptureInfo_v2  __API_TYPEDEF_PTSZ(PFN_lwStreamGetCaptureInfo, 11030, 11030)
#define PFN_lwStreamUpdateCaptureDependencies  __API_TYPEDEF_PTSZ(PFN_lwStreamUpdateCaptureDependencies, 11030, 11030)
#define PFN_lwStreamAttachMemAsync  __API_TYPEDEF_PTSZ(PFN_lwStreamAttachMemAsync, 6000, 7000)
#define PFN_lwStreamQuery  __API_TYPEDEF_PTSZ(PFN_lwStreamQuery, 2000, 7000)
#define PFN_lwStreamSynchronize  __API_TYPEDEF_PTSZ(PFN_lwStreamSynchronize, 2000, 7000)
#define PFN_lwStreamDestroy  PFN_lwStreamDestroy_v4000
#define PFN_lwStreamCopyAttributes  __API_TYPEDEF_PTSZ(PFN_lwStreamCopyAttributes, 11000, 11000)
#define PFN_lwStreamGetAttribute  __API_TYPEDEF_PTSZ(PFN_lwStreamGetAttribute, 11000, 11000)
#define PFN_lwStreamSetAttribute  __API_TYPEDEF_PTSZ(PFN_lwStreamSetAttribute, 11000, 11000)
#define PFN_lwEventCreate  PFN_lwEventCreate_v2000
#define PFN_lwEventRecord  __API_TYPEDEF_PTSZ(PFN_lwEventRecord, 2000, 7000)
#define PFN_lwEventRecordWithFlags  __API_TYPEDEF_PTSZ(PFN_lwEventRecordWithFlags, 11010, 11010)
#define PFN_lwEventQuery  PFN_lwEventQuery_v2000
#define PFN_lwEventSynchronize  PFN_lwEventSynchronize_v2000
#define PFN_lwEventDestroy  PFN_lwEventDestroy_v4000
#define PFN_lwEventElapsedTime  PFN_lwEventElapsedTime_v2000
#define PFN_lwImportExternalMemory  PFN_lwImportExternalMemory_v10000
#define PFN_lwExternalMemoryGetMappedBuffer  PFN_lwExternalMemoryGetMappedBuffer_v10000
#define PFN_lwExternalMemoryGetMappedMipmappedArray  PFN_lwExternalMemoryGetMappedMipmappedArray_v10000
#define PFN_lwDestroyExternalMemory  PFN_lwDestroyExternalMemory_v10000
#define PFN_lwImportExternalSemaphore  PFN_lwImportExternalSemaphore_v10000
#define PFN_lwSignalExternalSemaphoresAsync  __API_TYPEDEF_PTSZ(PFN_lwSignalExternalSemaphoresAsync, 10000, 10000)
#define PFN_lwWaitExternalSemaphoresAsync  __API_TYPEDEF_PTSZ(PFN_lwWaitExternalSemaphoresAsync, 10000, 10000)
#define PFN_lwDestroyExternalSemaphore  PFN_lwDestroyExternalSemaphore_v10000
#define PFN_lwStreamWaitValue32  __API_TYPEDEF_PTSZ(PFN_lwStreamWaitValue32, 8000, 8000)
#define PFN_lwStreamWaitValue64  __API_TYPEDEF_PTSZ(PFN_lwStreamWaitValue64, 9000, 9000)
#define PFN_lwStreamWriteValue32  __API_TYPEDEF_PTSZ(PFN_lwStreamWriteValue32, 8000, 8000)
#define PFN_lwStreamWriteValue64  __API_TYPEDEF_PTSZ(PFN_lwStreamWriteValue64, 9000, 9000)
#define PFN_lwStreamBatchMemOp  __API_TYPEDEF_PTSZ(PFN_lwStreamBatchMemOp, 8000, 8000)
#define PFN_lwFuncGetAttribute  PFN_lwFuncGetAttribute_v2020
#define PFN_lwFuncSetAttribute  PFN_lwFuncSetAttribute_v9000
#define PFN_lwFuncSetCacheConfig  PFN_lwFuncSetCacheConfig_v3000
#define PFN_lwFuncSetSharedMemConfig  PFN_lwFuncSetSharedMemConfig_v4020
#define PFN_lwLaunchKernel  __API_TYPEDEF_PTSZ(PFN_lwLaunchKernel, 4000, 7000)
#define PFN_lwLaunchCooperativeKernel  __API_TYPEDEF_PTSZ(PFN_lwLaunchCooperativeKernel, 9000, 9000)
#define PFN_lwLaunchCooperativeKernelMultiDevice  PFN_lwLaunchCooperativeKernelMultiDevice_v9000
#define PFN_lwLaunchHostFunc  __API_TYPEDEF_PTSZ(PFN_lwLaunchHostFunc, 10000, 10000)
#define PFN_lwFuncSetBlockShape  PFN_lwFuncSetBlockShape_v2000
#define PFN_lwFuncSetSharedSize  PFN_lwFuncSetSharedSize_v2000
#define PFN_lwParamSetSize  PFN_lwParamSetSize_v2000
#define PFN_lwParamSeti  PFN_lwParamSeti_v2000
#define PFN_lwParamSetf  PFN_lwParamSetf_v2000
#define PFN_lwParamSetv  PFN_lwParamSetv_v2000
#define PFN_lwLaunch  PFN_lwLaunch_v2000
#define PFN_lwLaunchGrid  PFN_lwLaunchGrid_v2000
#define PFN_lwLaunchGridAsync  PFN_lwLaunchGridAsync_v2000
#define PFN_lwParamSetTexRef  PFN_lwParamSetTexRef_v2000
#define PFN_lwGraphCreate  PFN_lwGraphCreate_v10000
#define PFN_lwGraphAddKernelNode  PFN_lwGraphAddKernelNode_v10000
#define PFN_lwGraphKernelNodeGetParams  PFN_lwGraphKernelNodeGetParams_v10000
#define PFN_lwGraphKernelNodeSetParams  PFN_lwGraphKernelNodeSetParams_v10000
#define PFN_lwGraphAddMemcpyNode  PFN_lwGraphAddMemcpyNode_v10000
#define PFN_lwGraphMemcpyNodeGetParams  PFN_lwGraphMemcpyNodeGetParams_v10000
#define PFN_lwGraphMemcpyNodeSetParams  PFN_lwGraphMemcpyNodeSetParams_v10000
#define PFN_lwGraphAddMemsetNode  PFN_lwGraphAddMemsetNode_v10000
#define PFN_lwGraphMemsetNodeGetParams  PFN_lwGraphMemsetNodeGetParams_v10000
#define PFN_lwGraphMemsetNodeSetParams  PFN_lwGraphMemsetNodeSetParams_v10000
#define PFN_lwGraphAddHostNode  PFN_lwGraphAddHostNode_v10000
#define PFN_lwGraphHostNodeGetParams  PFN_lwGraphHostNodeGetParams_v10000
#define PFN_lwGraphHostNodeSetParams  PFN_lwGraphHostNodeSetParams_v10000
#define PFN_lwGraphAddChildGraphNode  PFN_lwGraphAddChildGraphNode_v10000
#define PFN_lwGraphChildGraphNodeGetGraph  PFN_lwGraphChildGraphNodeGetGraph_v10000
#define PFN_lwGraphAddEmptyNode  PFN_lwGraphAddEmptyNode_v10000
#define PFN_lwGraphAddEventRecordNode  PFN_lwGraphAddEventRecordNode_v11010
#define PFN_lwGraphEventRecordNodeGetEvent  PFN_lwGraphEventRecordNodeGetEvent_v11010
#define PFN_lwGraphEventRecordNodeSetEvent  PFN_lwGraphEventRecordNodeSetEvent_v11010
#define PFN_lwGraphAddEventWaitNode  PFN_lwGraphAddEventWaitNode_v11010
#define PFN_lwGraphEventWaitNodeGetEvent  PFN_lwGraphEventWaitNodeGetEvent_v11010
#define PFN_lwGraphEventWaitNodeSetEvent  PFN_lwGraphEventWaitNodeSetEvent_v11010
#define PFN_lwGraphAddExternalSemaphoresSignalNode  PFN_lwGraphAddExternalSemaphoresSignalNode_v11020
#define PFN_lwGraphExternalSemaphoresSignalNodeGetParams  PFN_lwGraphExternalSemaphoresSignalNodeGetParams_v11020
#define PFN_lwGraphExternalSemaphoresSignalNodeSetParams  PFN_lwGraphExternalSemaphoresSignalNodeSetParams_v11020
#define PFN_lwGraphAddExternalSemaphoresWaitNode  PFN_lwGraphAddExternalSemaphoresWaitNode_v11020
#define PFN_lwGraphExternalSemaphoresWaitNodeGetParams  PFN_lwGraphExternalSemaphoresWaitNodeGetParams_v11020
#define PFN_lwGraphExternalSemaphoresWaitNodeSetParams  PFN_lwGraphExternalSemaphoresWaitNodeSetParams_v11020
#define PFN_lwGraphClone  PFN_lwGraphClone_v10000
#define PFN_lwGraphNodeFindInClone  PFN_lwGraphNodeFindInClone_v10000
#define PFN_lwGraphNodeGetType  PFN_lwGraphNodeGetType_v10000
#define PFN_lwGraphGetNodes  PFN_lwGraphGetNodes_v10000
#define PFN_lwGraphGetRootNodes  PFN_lwGraphGetRootNodes_v10000
#define PFN_lwGraphGetEdges  PFN_lwGraphGetEdges_v10000
#define PFN_lwGraphNodeGetDependencies  PFN_lwGraphNodeGetDependencies_v10000
#define PFN_lwGraphNodeGetDependentNodes  PFN_lwGraphNodeGetDependentNodes_v10000
#define PFN_lwGraphAddDependencies  PFN_lwGraphAddDependencies_v10000
#define PFN_lwGraphRemoveDependencies  PFN_lwGraphRemoveDependencies_v10000
#define PFN_lwGraphDestroyNode  PFN_lwGraphDestroyNode_v10000
#define PFN_lwGraphInstantiate  PFN_lwGraphInstantiate_v11000
#define PFN_lwGraphExecKernelNodeSetParams  PFN_lwGraphExecKernelNodeSetParams_v10010
#define PFN_lwGraphExecMemcpyNodeSetParams  PFN_lwGraphExecMemcpyNodeSetParams_v10020
#define PFN_lwGraphExecMemsetNodeSetParams  PFN_lwGraphExecMemsetNodeSetParams_v10020
#define PFN_lwGraphExecHostNodeSetParams  PFN_lwGraphExecHostNodeSetParams_v10020
#define PFN_lwGraphExecChildGraphNodeSetParams  PFN_lwGraphExecChildGraphNodeSetParams_v11010
#define PFN_lwGraphExecEventRecordNodeSetEvent  PFN_lwGraphExecEventRecordNodeSetEvent_v11010
#define PFN_lwGraphExecEventWaitNodeSetEvent  PFN_lwGraphExecEventWaitNodeSetEvent_v11010
#define PFN_lwGraphExecExternalSemaphoresSignalNodeSetParams  PFN_lwGraphExecExternalSemaphoresSignalNodeSetParams_v11020
#define PFN_lwGraphExecExternalSemaphoresWaitNodeSetParams  PFN_lwGraphExecExternalSemaphoresWaitNodeSetParams_v11020
#define PFN_lwGraphUpload  __API_TYPEDEF_PTSZ(PFN_lwGraphUpload, 11010, 11010)
#define PFN_lwGraphLaunch  __API_TYPEDEF_PTSZ(PFN_lwGraphLaunch, 10000, 10000)
#define PFN_lwGraphExecDestroy  PFN_lwGraphExecDestroy_v10000
#define PFN_lwGraphDestroy  PFN_lwGraphDestroy_v10000
#define PFN_lwGraphExelwpdate  PFN_lwGraphExelwpdate_v10020
#define PFN_lwGraphKernelNodeCopyAttributes  PFN_lwGraphKernelNodeCopyAttributes_v11000
#define PFN_lwGraphKernelNodeGetAttribute  PFN_lwGraphKernelNodeGetAttribute_v11000
#define PFN_lwGraphKernelNodeSetAttribute  PFN_lwGraphKernelNodeSetAttribute_v11000
#define PFN_lwGraphDebugDotPrint  PFN_lwGraphDebugDotPrint_v11030
#define PFN_lwOclwpancyMaxActiveBlocksPerMultiprocessor  PFN_lwOclwpancyMaxActiveBlocksPerMultiprocessor_v6050
#define PFN_lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags  PFN_lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000
#define PFN_lwOclwpancyMaxPotentialBlockSize  PFN_lwOclwpancyMaxPotentialBlockSize_v6050
#define PFN_lwOclwpancyMaxPotentialBlockSizeWithFlags  PFN_lwOclwpancyMaxPotentialBlockSizeWithFlags_v7000
#define PFN_lwOclwpancyAvailableDynamicSMemPerBlock  PFN_lwOclwpancyAvailableDynamicSMemPerBlock_v10020
#define PFN_lwTexRefSetArray  PFN_lwTexRefSetArray_v2000
#define PFN_lwTexRefSetMipmappedArray  PFN_lwTexRefSetMipmappedArray_v5000
#define PFN_lwTexRefSetAddress  PFN_lwTexRefSetAddress_v3020
#define PFN_lwTexRefSetAddress2D  PFN_lwTexRefSetAddress2D_v4010
#define PFN_lwTexRefSetFormat  PFN_lwTexRefSetFormat_v2000
#define PFN_lwTexRefSetAddressMode  PFN_lwTexRefSetAddressMode_v2000
#define PFN_lwTexRefSetFilterMode  PFN_lwTexRefSetFilterMode_v2000
#define PFN_lwTexRefSetMipmapFilterMode  PFN_lwTexRefSetMipmapFilterMode_v5000
#define PFN_lwTexRefSetMipmapLevelBias  PFN_lwTexRefSetMipmapLevelBias_v5000
#define PFN_lwTexRefSetMipmapLevelClamp  PFN_lwTexRefSetMipmapLevelClamp_v5000
#define PFN_lwTexRefSetMaxAnisotropy  PFN_lwTexRefSetMaxAnisotropy_v5000
#define PFN_lwTexRefSetBorderColor  PFN_lwTexRefSetBorderColor_v8000
#define PFN_lwTexRefSetFlags  PFN_lwTexRefSetFlags_v2000
#define PFN_lwTexRefGetAddress  PFN_lwTexRefGetAddress_v3020
#define PFN_lwTexRefGetArray  PFN_lwTexRefGetArray_v2000
#define PFN_lwTexRefGetMipmappedArray  PFN_lwTexRefGetMipmappedArray_v5000
#define PFN_lwTexRefGetAddressMode  PFN_lwTexRefGetAddressMode_v2000
#define PFN_lwTexRefGetFilterMode  PFN_lwTexRefGetFilterMode_v2000
#define PFN_lwTexRefGetFormat  PFN_lwTexRefGetFormat_v2000
#define PFN_lwTexRefGetMipmapFilterMode  PFN_lwTexRefGetMipmapFilterMode_v5000
#define PFN_lwTexRefGetMipmapLevelBias  PFN_lwTexRefGetMipmapLevelBias_v5000
#define PFN_lwTexRefGetMipmapLevelClamp  PFN_lwTexRefGetMipmapLevelClamp_v5000
#define PFN_lwTexRefGetMaxAnisotropy  PFN_lwTexRefGetMaxAnisotropy_v5000
#define PFN_lwTexRefGetBorderColor  PFN_lwTexRefGetBorderColor_v8000
#define PFN_lwTexRefGetFlags  PFN_lwTexRefGetFlags_v2000
#define PFN_lwTexRefCreate  PFN_lwTexRefCreate_v2000
#define PFN_lwTexRefDestroy  PFN_lwTexRefDestroy_v2000
#define PFN_lwSurfRefSetArray  PFN_lwSurfRefSetArray_v3000
#define PFN_lwSurfRefGetArray  PFN_lwSurfRefGetArray_v3000
#define PFN_lwTexObjectCreate  PFN_lwTexObjectCreate_v5000
#define PFN_lwTexObjectDestroy  PFN_lwTexObjectDestroy_v5000
#define PFN_lwTexObjectGetResourceDesc  PFN_lwTexObjectGetResourceDesc_v5000
#define PFN_lwTexObjectGetTextureDesc  PFN_lwTexObjectGetTextureDesc_v5000
#define PFN_lwTexObjectGetResourceViewDesc  PFN_lwTexObjectGetResourceViewDesc_v5000
#define PFN_lwSurfObjectCreate  PFN_lwSurfObjectCreate_v5000
#define PFN_lwSurfObjectDestroy  PFN_lwSurfObjectDestroy_v5000
#define PFN_lwSurfObjectGetResourceDesc  PFN_lwSurfObjectGetResourceDesc_v5000
#define PFN_lwDeviceCanAccessPeer  PFN_lwDeviceCanAccessPeer_v4000
#define PFN_lwCtxEnablePeerAccess  PFN_lwCtxEnablePeerAccess_v4000
#define PFN_lwCtxDisablePeerAccess  PFN_lwCtxDisablePeerAccess_v4000
#define PFN_lwDeviceGetP2PAttribute  PFN_lwDeviceGetP2PAttribute_v8000
#define PFN_lwGraphicsUnregisterResource  PFN_lwGraphicsUnregisterResource_v3000
#define PFN_lwGraphicsSubResourceGetMappedArray  PFN_lwGraphicsSubResourceGetMappedArray_v3000
#define PFN_lwGraphicsResourceGetMappedMipmappedArray  PFN_lwGraphicsResourceGetMappedMipmappedArray_v5000
#define PFN_lwGraphicsResourceGetMappedPointer  PFN_lwGraphicsResourceGetMappedPointer_v3020
#define PFN_lwGraphicsResourceSetMapFlags  PFN_lwGraphicsResourceSetMapFlags_v6050
#define PFN_lwGraphicsMapResources  __API_TYPEDEF_PTSZ(PFN_lwGraphicsMapResources, 3000, 7000)
#define PFN_lwGraphicsUnmapResources  __API_TYPEDEF_PTSZ(PFN_lwGraphicsUnmapResources, 3000, 7000)
#define PFN_lwGetExportTable  PFN_lwGetExportTable_v3000
#define PFN_lwFuncGetModule  PFN_lwFuncGetModule_v11000
#define PFN_lwFlushGPUDirectRDMAWrites PFN_lwFlushGPUDirectRDMAWrites_v11030
#define PFN_lwGetProcAddress  PFN_lwGetProcAddress_v11030
#define PFN_lwUserObjectCreate  PFN_lwUserObjectCreate_v11030
#define PFN_lwUserObjectRetain  PFN_lwUserObjectRetain_v11030
#define PFN_lwUserObjectRelease  PFN_lwUserObjectRelease_v11030
#define PFN_lwGraphRetainUserObject  PFN_lwGraphRetainUserObject_v11030
#define PFN_lwGraphReleaseUserObject  PFN_lwGraphReleaseUserObject_v11030


/*
 * Type definitions for functions defined in lwca.h
 */
typedef LWresult (LWDAAPI *PFN_lwGetErrorString_v6000)(LWresult error, const char **pStr);
typedef LWresult (LWDAAPI *PFN_lwGetErrorName_v6000)(LWresult error, const char **pStr);
typedef LWresult (LWDAAPI *PFN_lwInit_v2000)(unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwDriverGetVersion_v2020)(int *driverVersion);
typedef LWresult (LWDAAPI *PFN_lwDeviceGet_v2000)(LWdevice_v1 *device, int ordinal);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetCount_v2000)(int *count);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetName_v2000)(char *name, int len, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetUuid_v9020)(LWuuid *uuid, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetLuid_v10000)(char *luid, unsigned int *deviceNodeMask, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceTotalMem_v3020)(size_t *bytes, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetTexture1DLinearMaxWidth_v11010)(size_t *maxWidthInElements, LWarray_format format, unsigned numChannels, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetAttribute_v2000)(int *pi, LWdevice_attribute attrib, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetLwSciSyncAttributes_v10020)(void *lwSciSyncAttrList, LWdevice_v1 dev, int flags);
typedef LWresult (LWDAAPI *PFN_lwDeviceSetMemPool_v11020)(LWdevice_v1 dev, LWmemoryPool pool);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetMemPool_v11020)(LWmemoryPool *pool, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetDefaultMemPool_v11020)(LWmemoryPool *pool_out, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetProperties_v2000)(LWdevprop_v1 *prop, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDeviceComputeCapability_v2000)(int *major, int *minor, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxRetain_v7000)(LWcontext *pctx, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxRelease_v11000)(LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxSetFlags_v11000)(LWdevice_v1 dev, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxGetState_v7000)(LWdevice_v1 dev, unsigned int *flags, int *active);
typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxReset_v11000)(LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwCtxCreate_v3020)(LWcontext *pctx, unsigned int flags, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwCtxDestroy_v4000)(LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwCtxPushLwrrent_v4000)(LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwCtxPopLwrrent_v4000)(LWcontext *pctx);
typedef LWresult (LWDAAPI *PFN_lwCtxSetLwrrent_v4000)(LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwCtxGetLwrrent_v4000)(LWcontext *pctx);
typedef LWresult (LWDAAPI *PFN_lwCtxGetDevice_v2000)(LWdevice_v1 *device);
typedef LWresult (LWDAAPI *PFN_lwCtxGetFlags_v7000)(unsigned int *flags);
typedef LWresult (LWDAAPI *PFN_lwCtxSynchronize_v2000)(void);
typedef LWresult (LWDAAPI *PFN_lwCtxSetLimit_v3010)(LWlimit limit, size_t value);
typedef LWresult (LWDAAPI *PFN_lwCtxGetLimit_v3010)(size_t *pvalue, LWlimit limit);
typedef LWresult (LWDAAPI *PFN_lwCtxGetCacheConfig_v3020)(LWfunc_cache *pconfig);
typedef LWresult (LWDAAPI *PFN_lwCtxSetCacheConfig_v3020)(LWfunc_cache config);
typedef LWresult (LWDAAPI *PFN_lwCtxGetSharedMemConfig_v4020)(LWsharedconfig *pConfig);
typedef LWresult (LWDAAPI *PFN_lwCtxSetSharedMemConfig_v4020)(LWsharedconfig config);
typedef LWresult (LWDAAPI *PFN_lwCtxGetApiVersion_v3020)(LWcontext ctx, unsigned int *version);
typedef LWresult (LWDAAPI *PFN_lwCtxGetStreamPriorityRange_v5050)(int *leastPriority, int *greatestPriority);
typedef LWresult (LWDAAPI *PFN_lwCtxResetPersistingL2Cache_v11000)(void);
typedef LWresult (LWDAAPI *PFN_lwCtxAttach_v2000)(LWcontext *pctx, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwCtxDetach_v2000)(LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwModuleLoad_v2000)(LWmodule *module, const char *fname);
typedef LWresult (LWDAAPI *PFN_lwModuleLoadData_v2000)(LWmodule *module, const void *image);
typedef LWresult (LWDAAPI *PFN_lwModuleLoadDataEx_v2010)(LWmodule *module, const void *image, unsigned int numOptions, LWjit_option *options, void **optiolwalues);
typedef LWresult (LWDAAPI *PFN_lwModuleLoadFatBinary_v2000)(LWmodule *module, const void *fatLwbin);
typedef LWresult (LWDAAPI *PFN_lwModuleUnload_v2000)(LWmodule hmod);
typedef LWresult (LWDAAPI *PFN_lwModuleGetFunction_v2000)(LWfunction *hfunc, LWmodule hmod, const char *name);
typedef LWresult (LWDAAPI *PFN_lwModuleGetGlobal_v3020)(LWdeviceptr_v2 *dptr, size_t *bytes, LWmodule hmod, const char *name);
typedef LWresult (LWDAAPI *PFN_lwModuleGetTexRef_v2000)(LWtexref *pTexRef, LWmodule hmod, const char *name);
typedef LWresult (LWDAAPI *PFN_lwModuleGetSurfRef_v3000)(LWsurfref *pSurfRef, LWmodule hmod, const char *name);
typedef LWresult (LWDAAPI *PFN_lwLinkCreate_v6050)(unsigned int numOptions, LWjit_option *options, void **optiolwalues, LWlinkState *stateOut);
typedef LWresult (LWDAAPI *PFN_lwLinkAddData_v6050)(LWlinkState state, LWjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, LWjit_option *options, void **optiolwalues);
typedef LWresult (LWDAAPI *PFN_lwLinkAddFile_v6050)(LWlinkState state, LWjitInputType type, const char *path, unsigned int numOptions, LWjit_option *options, void **optiolwalues);
typedef LWresult (LWDAAPI *PFN_lwLinkComplete_v5050)(LWlinkState state, void **lwbinOut, size_t *sizeOut);
typedef LWresult (LWDAAPI *PFN_lwLinkDestroy_v5050)(LWlinkState state);
typedef LWresult (LWDAAPI *PFN_lwMemGetInfo_v3020)(size_t *free, size_t *total);
typedef LWresult (LWDAAPI *PFN_lwMemAlloc_v3020)(LWdeviceptr_v2 *dptr, size_t bytesize);
typedef LWresult (LWDAAPI *PFN_lwMemAllocPitch_v3020)(LWdeviceptr_v2 *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
typedef LWresult (LWDAAPI *PFN_lwMemFree_v3020)(LWdeviceptr_v2 dptr);
typedef LWresult (LWDAAPI *PFN_lwMemGetAddressRange_v3020)(LWdeviceptr_v2 *pbase, size_t *psize, LWdeviceptr_v2 dptr);
typedef LWresult (LWDAAPI *PFN_lwMemAllocHost_v3020)(void **pp, size_t bytesize);
typedef LWresult (LWDAAPI *PFN_lwMemFreeHost_v2000)(void *p);
typedef LWresult (LWDAAPI *PFN_lwMemHostAlloc_v2020)(void **pp, size_t bytesize, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwMemHostGetDevicePointer_v3020)(LWdeviceptr_v2 *pdptr, void *p, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwMemHostGetFlags_v2030)(unsigned int *pFlags, void *p);
typedef LWresult (LWDAAPI *PFN_lwMemAllocManaged_v6000)(LWdeviceptr_v2 *dptr, size_t bytesize, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetByPCIBusId_v4010)(LWdevice_v1 *dev, const char *pciBusId);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetPCIBusId_v4010)(char *pciBusId, int len, LWdevice_v1 dev);
typedef LWresult (LWDAAPI *PFN_lwIpcGetEventHandle_v4010)(LWipcEventHandle_v1 *pHandle, LWevent event);
typedef LWresult (LWDAAPI *PFN_lwIpcOpenEventHandle_v4010)(LWevent *phEvent, LWipcEventHandle_v1 handle);
typedef LWresult (LWDAAPI *PFN_lwIpcGetMemHandle_v4010)(LWipcMemHandle_v1 *pHandle, LWdeviceptr_v2 dptr);
typedef LWresult (LWDAAPI *PFN_lwIpcOpenMemHandle_v11000)(LWdeviceptr_v2 *pdptr, LWipcMemHandle_v1 handle, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwIpcCloseMemHandle_v4010)(LWdeviceptr_v2 dptr);
typedef LWresult (LWDAAPI *PFN_lwMemHostRegister_v6050)(void *p, size_t bytesize, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwMemHostUnregister_v4000)(void *p);
typedef LWresult (LWDAAPI *PFN_lwMemcpy_v7000_ptds)(LWdeviceptr_v2 dst, LWdeviceptr_v2 src, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyPeer_v7000_ptds)(LWdeviceptr_v2 dstDevice, LWcontext dstContext, LWdeviceptr_v2 srcDevice, LWcontext srcContext, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoD_v7000_ptds)(LWdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoH_v7000_ptds)(void *dstHost, LWdeviceptr_v2 srcDevice, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoD_v7000_ptds)(LWdeviceptr_v2 dstDevice, LWdeviceptr_v2 srcDevice, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoA_v7000_ptds)(LWarray dstArray, size_t dstOffset, LWdeviceptr_v2 srcDevice, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoD_v7000_ptds)(LWdeviceptr_v2 dstDevice, LWarray srcArray, size_t srcOffset, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoA_v7000_ptds)(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoH_v7000_ptds)(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoA_v7000_ptds)(LWarray dstArray, size_t dstOffset, LWarray srcArray, size_t srcOffset, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpy2D_v7000_ptds)(const LWDA_MEMCPY2D_v2 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpy2DUnaligned_v7000_ptds)(const LWDA_MEMCPY2D_v2 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3D_v7000_ptds)(const LWDA_MEMCPY3D_v2 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3DPeer_v7000_ptds)(const LWDA_MEMCPY3D_PEER_v1 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAsync_v7000_ptsz)(LWdeviceptr_v2 dst, LWdeviceptr_v2 src, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyPeerAsync_v7000_ptsz)(LWdeviceptr_v2 dstDevice, LWcontext dstContext, LWdeviceptr_v2 srcDevice, LWcontext srcContext, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoDAsync_v7000_ptsz)(LWdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoHAsync_v7000_ptsz)(void *dstHost, LWdeviceptr_v2 srcDevice, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoDAsync_v7000_ptsz)(LWdeviceptr_v2 dstDevice, LWdeviceptr_v2 srcDevice, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoAAsync_v7000_ptsz)(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoHAsync_v7000_ptsz)(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpy2DAsync_v7000_ptsz)(const LWDA_MEMCPY2D_v2 *pCopy, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3DAsync_v7000_ptsz)(const LWDA_MEMCPY3D_v2 *pCopy, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3DPeerAsync_v7000_ptsz)(const LWDA_MEMCPY3D_PEER_v1 *pCopy, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD8_v7000_ptds)(LWdeviceptr_v2 dstDevice, unsigned char uc, size_t N);
typedef LWresult (LWDAAPI *PFN_lwMemsetD16_v7000_ptds)(LWdeviceptr_v2 dstDevice, unsigned short us, size_t N);
typedef LWresult (LWDAAPI *PFN_lwMemsetD32_v7000_ptds)(LWdeviceptr_v2 dstDevice, unsigned int ui, size_t N);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D8_v7000_ptds)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D16_v7000_ptds)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D32_v7000_ptds)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
typedef LWresult (LWDAAPI *PFN_lwMemsetD8Async_v7000_ptsz)(LWdeviceptr_v2 dstDevice, unsigned char uc, size_t N, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD16Async_v7000_ptsz)(LWdeviceptr_v2 dstDevice, unsigned short us, size_t N, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD32Async_v7000_ptsz)(LWdeviceptr_v2 dstDevice, unsigned int ui, size_t N, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D8Async_v7000_ptsz)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D16Async_v7000_ptsz)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D32Async_v7000_ptsz)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwArrayCreate_v3020)(LWarray *pHandle, const LWDA_ARRAY_DESCRIPTOR_v2 *pAllocateArray);
typedef LWresult (LWDAAPI *PFN_lwArrayGetDescriptor_v3020)(LWDA_ARRAY_DESCRIPTOR_v2 *pArrayDescriptor, LWarray hArray);
typedef LWresult (LWDAAPI *PFN_lwArrayGetSparseProperties_v11010)(LWDA_ARRAY_SPARSE_PROPERTIES_v1 *sparseProperties, LWarray array);
typedef LWresult (LWDAAPI *PFN_lwMipmappedArrayGetSparseProperties_v11010)(LWDA_ARRAY_SPARSE_PROPERTIES_v1 *sparseProperties, LWmipmappedArray mipmap);
typedef LWresult (LWDAAPI *PFN_lwArrayGetPlane_v11020)(LWarray *pPlaneArray, LWarray hArray, unsigned int planeIdx);
typedef LWresult (LWDAAPI *PFN_lwArrayDestroy_v2000)(LWarray hArray);
typedef LWresult (LWDAAPI *PFN_lwArray3DCreate_v3020)(LWarray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR_v2 *pAllocateArray);
typedef LWresult (LWDAAPI *PFN_lwArray3DGetDescriptor_v3020)(LWDA_ARRAY3D_DESCRIPTOR_v2 *pArrayDescriptor, LWarray hArray);
typedef LWresult (LWDAAPI *PFN_lwMipmappedArrayCreate_v5000)(LWmipmappedArray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR_v2 *pMipmappedArrayDesc, unsigned int numMipmapLevels);
typedef LWresult (LWDAAPI *PFN_lwMipmappedArrayGetLevel_v5000)(LWarray *pLevelArray, LWmipmappedArray hMipmappedArray, unsigned int level);
typedef LWresult (LWDAAPI *PFN_lwMipmappedArrayDestroy_v5000)(LWmipmappedArray hMipmappedArray);
typedef LWresult (LWDAAPI *PFN_lwMemAddressReserve_v10020)(LWdeviceptr_v2 *ptr, size_t size, size_t alignment, LWdeviceptr_v2 addr, unsigned long long flags);
typedef LWresult (LWDAAPI *PFN_lwMemAddressFree_v10020)(LWdeviceptr_v2 ptr, size_t size);
typedef LWresult (LWDAAPI *PFN_lwMemCreate_v10020)(LWmemGenericAllocationHandle_v1 *handle, size_t size, const LWmemAllocationProp_v1 *prop, unsigned long long flags);
typedef LWresult (LWDAAPI *PFN_lwMemRelease_v10020)(LWmemGenericAllocationHandle_v1 handle);
typedef LWresult (LWDAAPI *PFN_lwMemMap_v10020)(LWdeviceptr_v2 ptr, size_t size, size_t offset, LWmemGenericAllocationHandle_v1 handle, unsigned long long flags);
typedef LWresult (LWDAAPI *PFN_lwMemMapArrayAsync_v11010_ptsz)(LWarrayMapInfo_v1 *mapInfoList, unsigned int count, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemUnmap_v10020)(LWdeviceptr_v2 ptr, size_t size);
typedef LWresult (LWDAAPI *PFN_lwMemSetAccess_v10020)(LWdeviceptr_v2 ptr, size_t size, const LWmemAccessDesc_v1 *desc, size_t count);
typedef LWresult (LWDAAPI *PFN_lwMemGetAccess_v10020)(unsigned long long *flags, const LWmemLocation_v1 *location, LWdeviceptr_v2 ptr);
typedef LWresult (LWDAAPI *PFN_lwMemExportToShareableHandle_v10020)(void *shareableHandle, LWmemGenericAllocationHandle_v1 handle, LWmemAllocationHandleType handleType, unsigned long long flags);
typedef LWresult (LWDAAPI *PFN_lwMemImportFromShareableHandle_v10020)(LWmemGenericAllocationHandle_v1 *handle, void *osHandle, LWmemAllocationHandleType shHandleType);
typedef LWresult (LWDAAPI *PFN_lwMemGetAllocationGranularity_v10020)(size_t *granularity, const LWmemAllocationProp_v1 *prop, LWmemAllocationGranularity_flags option);
typedef LWresult (LWDAAPI *PFN_lwMemGetAllocationPropertiesFromHandle_v10020)(LWmemAllocationProp_v1 *prop, LWmemGenericAllocationHandle_v1 handle);
typedef LWresult (LWDAAPI *PFN_lwMemRetainAllocationHandle_v11000)(LWmemGenericAllocationHandle_v1 *handle, void *addr);
typedef LWresult (LWDAAPI *PFN_lwMemFreeAsync_v11020_ptsz)(LWdeviceptr_v2 dptr, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemAllocAsync_v11020_ptsz)(LWdeviceptr_v2 *dptr, size_t bytesize, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemPoolTrimTo_v11020)(LWmemoryPool pool, size_t minBytesToKeep);
typedef LWresult (LWDAAPI *PFN_lwMemPoolSetAttribute_v11020)(LWmemoryPool pool, LWmemPool_attribute attr, void *value);
typedef LWresult (LWDAAPI *PFN_lwMemPoolGetAttribute_v11020)(LWmemoryPool pool, LWmemPool_attribute attr, void *value);
typedef LWresult (LWDAAPI *PFN_lwMemPoolSetAccess_v11020)(LWmemoryPool pool, const LWmemAccessDesc_v1 *map, size_t count);
typedef LWresult (LWDAAPI *PFN_lwMemPoolGetAccess_v11020)(LWmemAccess_flags *flags, LWmemoryPool memPool, LWmemLocation_v1 *location);
typedef LWresult (LWDAAPI *PFN_lwMemPoolCreate_v11020)(LWmemoryPool *pool, const LWmemPoolProps_v1 *poolProps);
typedef LWresult (LWDAAPI *PFN_lwMemPoolDestroy_v11020)(LWmemoryPool pool);
typedef LWresult (LWDAAPI *PFN_lwMemAllocFromPoolAsync_v11020_ptsz)(LWdeviceptr_v2 *dptr, size_t bytesize, LWmemoryPool pool, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemPoolExportToShareableHandle_v11020)(void *handle_out, LWmemoryPool pool, LWmemAllocationHandleType handleType, unsigned long long flags);
typedef LWresult (LWDAAPI *PFN_lwMemPoolImportFromShareableHandle_v11020)(LWmemoryPool *pool_out, void *handle, LWmemAllocationHandleType handleType, unsigned long long flags);
typedef LWresult (LWDAAPI *PFN_lwMemPoolExportPointer_v11020)(LWmemPoolPtrExportData_v1 *shareData_out, LWdeviceptr_v2 ptr);
typedef LWresult (LWDAAPI *PFN_lwMemPoolImportPointer_v11020)(LWdeviceptr_v2 *ptr_out, LWmemoryPool pool, LWmemPoolPtrExportData_v1 *shareData);
typedef LWresult (LWDAAPI *PFN_lwPointerGetAttribute_v4000)(void *data, LWpointer_attribute attribute, LWdeviceptr_v2 ptr);
typedef LWresult (LWDAAPI *PFN_lwMemPrefetchAsync_v8000_ptsz)(LWdeviceptr_v2 devPtr, size_t count, LWdevice_v1 dstDevice, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemAdvise_v8000)(LWdeviceptr_v2 devPtr, size_t count, LWmem_advise advice, LWdevice_v1 device);
typedef LWresult (LWDAAPI *PFN_lwMemRangeGetAttribute_v8000)(void *data, size_t dataSize, LWmem_range_attribute attribute, LWdeviceptr_v2 devPtr, size_t count);
typedef LWresult (LWDAAPI *PFN_lwMemRangeGetAttributes_v8000)(void **data, size_t *dataSizes, LWmem_range_attribute *attributes, size_t numAttributes, LWdeviceptr_v2 devPtr, size_t count);
typedef LWresult (LWDAAPI *PFN_lwPointerSetAttribute_v6000)(const void *value, LWpointer_attribute attribute, LWdeviceptr_v2 ptr);
typedef LWresult (LWDAAPI *PFN_lwPointerGetAttributes_v7000)(unsigned int numAttributes, LWpointer_attribute *attributes, void **data, LWdeviceptr_v2 ptr);
typedef LWresult (LWDAAPI *PFN_lwStreamCreate_v2000)(LWstream *phStream, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwStreamCreateWithPriority_v5050)(LWstream *phStream, unsigned int flags, int priority);
typedef LWresult (LWDAAPI *PFN_lwStreamGetPriority_v7000_ptsz)(LWstream hStream, int *priority);
typedef LWresult (LWDAAPI *PFN_lwStreamGetFlags_v7000_ptsz)(LWstream hStream, unsigned int *flags);
typedef LWresult (LWDAAPI *PFN_lwStreamGetCtx_v9020_ptsz)(LWstream hStream, LWcontext *pctx);
typedef LWresult (LWDAAPI *PFN_lwStreamWaitEvent_v7000_ptsz)(LWstream hStream, LWevent hEvent, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwStreamAddCallback_v7000_ptsz)(LWstream hStream, LWstreamCallback callback, void *userData, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamBeginCapture_v10010_ptsz)(LWstream hStream, LWstreamCaptureMode mode);
typedef LWresult (LWDAAPI *PFN_lwThreadExchangeStreamCaptureMode_v10010)(LWstreamCaptureMode *mode);
typedef LWresult (LWDAAPI *PFN_lwStreamEndCapture_v10000_ptsz)(LWstream hStream, LWgraph *phGraph);
typedef LWresult (LWDAAPI *PFN_lwStreamIsCapturing_v10000_ptsz)(LWstream hStream, LWstreamCaptureStatus *captureStatus);
typedef LWresult (LWDAAPI *PFN_lwStreamGetCaptureInfo_v10010_ptsz)(LWstream hStream, LWstreamCaptureStatus *captureStatus_out, lwuint64_t *id_out);
typedef LWresult (LWDAAPI *PFN_lwStreamGetCaptureInfo_v11030_ptsz)(LWstream hStream, LWstreamCaptureStatus *captureStatus_out, lwuint64_t *id_out, LWgraph *graph_out, const LWgraphNode **dependencies_out, size_t *numDependencies_out);
typedef LWresult (LWDAAPI *PFN_lwStreamUpdateCaptureDependencies_v11030_ptsz)(LWstream hStream, LWgraphNode *dependencies, size_t numDependencies, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamAttachMemAsync_v7000_ptsz)(LWstream hStream, LWdeviceptr_v2 dptr, size_t length, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamQuery_v7000_ptsz)(LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwStreamSynchronize_v7000_ptsz)(LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwStreamDestroy_v4000)(LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwStreamCopyAttributes_v11000_ptsz)(LWstream dst, LWstream src);
typedef LWresult (LWDAAPI *PFN_lwStreamGetAttribute_v11000_ptsz)(LWstream hStream, LWstreamAttrID attr, LWstreamAttrValue_v1 *value_out);
typedef LWresult (LWDAAPI *PFN_lwStreamSetAttribute_v11000_ptsz)(LWstream hStream, LWstreamAttrID attr, const LWstreamAttrValue_v1 *value);
typedef LWresult (LWDAAPI *PFN_lwEventCreate_v2000)(LWevent *phEvent, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwEventRecord_v7000_ptsz)(LWevent hEvent, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwEventRecordWithFlags_v11010_ptsz)(LWevent hEvent, LWstream hStream, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwEventQuery_v2000)(LWevent hEvent);
typedef LWresult (LWDAAPI *PFN_lwEventSynchronize_v2000)(LWevent hEvent);
typedef LWresult (LWDAAPI *PFN_lwEventDestroy_v4000)(LWevent hEvent);
typedef LWresult (LWDAAPI *PFN_lwEventElapsedTime_v2000)(float *pMilliseconds, LWevent hStart, LWevent hEnd);
typedef LWresult (LWDAAPI *PFN_lwImportExternalMemory_v10000)(LWexternalMemory *extMem_out, const LWDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 *memHandleDesc);
typedef LWresult (LWDAAPI *PFN_lwExternalMemoryGetMappedBuffer_v10000)(LWdeviceptr_v2 *devPtr, LWexternalMemory extMem, const LWDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 *bufferDesc);
typedef LWresult (LWDAAPI *PFN_lwExternalMemoryGetMappedMipmappedArray_v10000)(LWmipmappedArray *mipmap, LWexternalMemory extMem, const LWDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 *mipmapDesc);
typedef LWresult (LWDAAPI *PFN_lwDestroyExternalMemory_v10000)(LWexternalMemory extMem);
typedef LWresult (LWDAAPI *PFN_lwImportExternalSemaphore_v10000)(LWexternalSemaphore *extSem_out, const LWDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 *semHandleDesc);
typedef LWresult (LWDAAPI *PFN_lwSignalExternalSemaphoresAsync_v10000_ptsz)(const LWexternalSemaphore *extSemArray, const LWDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 *paramsArray, unsigned int numExtSems, LWstream stream);
typedef LWresult (LWDAAPI *PFN_lwWaitExternalSemaphoresAsync_v10000_ptsz)(const LWexternalSemaphore *extSemArray, const LWDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 *paramsArray, unsigned int numExtSems, LWstream stream);
typedef LWresult (LWDAAPI *PFN_lwDestroyExternalSemaphore_v10000)(LWexternalSemaphore extSem);
typedef LWresult (LWDAAPI *PFN_lwStreamWaitValue32_v8000_ptsz)(LWstream stream, LWdeviceptr_v2 addr, lwuint32_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamWaitValue64_v9000_ptsz)(LWstream stream, LWdeviceptr_v2 addr, lwuint64_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamWriteValue32_v8000_ptsz)(LWstream stream, LWdeviceptr_v2 addr, lwuint32_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamWriteValue64_v9000_ptsz)(LWstream stream, LWdeviceptr_v2 addr, lwuint64_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamBatchMemOp_v8000_ptsz)(LWstream stream, unsigned int count, LWstreamBatchMemOpParams_v1 *paramArray, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwFuncGetAttribute_v2020)(int *pi, LWfunction_attribute attrib, LWfunction hfunc);
typedef LWresult (LWDAAPI *PFN_lwFuncSetAttribute_v9000)(LWfunction hfunc, LWfunction_attribute attrib, int value);
typedef LWresult (LWDAAPI *PFN_lwFuncSetCacheConfig_v3000)(LWfunction hfunc, LWfunc_cache config);
typedef LWresult (LWDAAPI *PFN_lwFuncSetSharedMemConfig_v4020)(LWfunction hfunc, LWsharedconfig config);
typedef LWresult (LWDAAPI *PFN_lwLaunchKernel_v7000_ptsz)(LWfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, LWstream hStream, void **kernelParams, void **extra);
typedef LWresult (LWDAAPI *PFN_lwLaunchCooperativeKernel_v9000_ptsz)(LWfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, LWstream hStream, void **kernelParams);
typedef LWresult (LWDAAPI *PFN_lwLaunchCooperativeKernelMultiDevice_v9000)(LWDA_LAUNCH_PARAMS_v1 *launchParamsList, unsigned int numDevices, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwLaunchHostFunc_v10000_ptsz)(LWstream hStream, LWhostFn fn, void *userData);
typedef LWresult (LWDAAPI *PFN_lwFuncSetBlockShape_v2000)(LWfunction hfunc, int x, int y, int z);
typedef LWresult (LWDAAPI *PFN_lwFuncSetSharedSize_v2000)(LWfunction hfunc, unsigned int bytes);
typedef LWresult (LWDAAPI *PFN_lwParamSetSize_v2000)(LWfunction hfunc, unsigned int numbytes);
typedef LWresult (LWDAAPI *PFN_lwParamSeti_v2000)(LWfunction hfunc, int offset, unsigned int value);
typedef LWresult (LWDAAPI *PFN_lwParamSetf_v2000)(LWfunction hfunc, int offset, float value);
typedef LWresult (LWDAAPI *PFN_lwParamSetv_v2000)(LWfunction hfunc, int offset, void *ptr, unsigned int numbytes);
typedef LWresult (LWDAAPI *PFN_lwLaunch_v2000)(LWfunction f);
typedef LWresult (LWDAAPI *PFN_lwLaunchGrid_v2000)(LWfunction f, int grid_width, int grid_height);
typedef LWresult (LWDAAPI *PFN_lwLaunchGridAsync_v2000)(LWfunction f, int grid_width, int grid_height, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwParamSetTexRef_v2000)(LWfunction hfunc, int texunit, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwGraphCreate_v10000)(LWgraph *phGraph, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwGraphAddKernelNode_v10000)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, const LWDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphKernelNodeGetParams_v10000)(LWgraphNode hNode, LWDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphKernelNodeSetParams_v10000)(LWgraphNode hNode, const LWDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphAddMemcpyNode_v10000)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, const LWDA_MEMCPY3D_v2 *copyParams, LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwGraphMemcpyNodeGetParams_v10000)(LWgraphNode hNode, LWDA_MEMCPY3D_v2 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphMemcpyNodeSetParams_v10000)(LWgraphNode hNode, const LWDA_MEMCPY3D_v2 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphAddMemsetNode_v10000)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, const LWDA_MEMSET_NODE_PARAMS_v1 *memsetParams, LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwGraphMemsetNodeGetParams_v10000)(LWgraphNode hNode, LWDA_MEMSET_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphMemsetNodeSetParams_v10000)(LWgraphNode hNode, const LWDA_MEMSET_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphAddHostNode_v10000)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, const LWDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphHostNodeGetParams_v10000)(LWgraphNode hNode, LWDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphHostNodeSetParams_v10000)(LWgraphNode hNode, const LWDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphAddChildGraphNode_v10000)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, LWgraph childGraph);
typedef LWresult (LWDAAPI *PFN_lwGraphChildGraphNodeGetGraph_v10000)(LWgraphNode hNode, LWgraph *phGraph);
typedef LWresult (LWDAAPI *PFN_lwGraphAddEmptyNode_v10000)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies);
typedef LWresult (LWDAAPI *PFN_lwGraphAddEventRecordNode_v11010)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, LWevent event);
typedef LWresult (LWDAAPI *PFN_lwGraphEventRecordNodeGetEvent_v11010)(LWgraphNode hNode, LWevent *event_out);
typedef LWresult (LWDAAPI *PFN_lwGraphEventRecordNodeSetEvent_v11010)(LWgraphNode hNode, LWevent event);
typedef LWresult (LWDAAPI *PFN_lwGraphAddEventWaitNode_v11010)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, LWevent event);
typedef LWresult (LWDAAPI *PFN_lwGraphEventWaitNodeGetEvent_v11010)(LWgraphNode hNode, LWevent *event_out);
typedef LWresult (LWDAAPI *PFN_lwGraphEventWaitNodeSetEvent_v11010)(LWgraphNode hNode, LWevent event);
typedef LWresult (LWDAAPI *PFN_lwGraphAddExternalSemaphoresSignalNode_v11020)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, const LWDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphExternalSemaphoresSignalNodeGetParams_v11020)(LWgraphNode hNode, LWDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *params_out);
typedef LWresult (LWDAAPI *PFN_lwGraphExternalSemaphoresSignalNodeSetParams_v11020)(LWgraphNode hNode, const LWDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphAddExternalSemaphoresWaitNode_v11020)(LWgraphNode *phGraphNode, LWgraph hGraph, const LWgraphNode *dependencies, size_t numDependencies, const LWDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphExternalSemaphoresWaitNodeGetParams_v11020)(LWgraphNode hNode, LWDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *params_out);
typedef LWresult (LWDAAPI *PFN_lwGraphExternalSemaphoresWaitNodeSetParams_v11020)(LWgraphNode hNode, const LWDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphClone_v10000)(LWgraph *phGraphClone, LWgraph originalGraph);
typedef LWresult (LWDAAPI *PFN_lwGraphNodeFindInClone_v10000)(LWgraphNode *phNode, LWgraphNode hOriginalNode, LWgraph hClonedGraph);
typedef LWresult (LWDAAPI *PFN_lwGraphNodeGetType_v10000)(LWgraphNode hNode, LWgraphNodeType *type);
typedef LWresult (LWDAAPI *PFN_lwGraphGetNodes_v10000)(LWgraph hGraph, LWgraphNode *nodes, size_t *numNodes);
typedef LWresult (LWDAAPI *PFN_lwGraphGetRootNodes_v10000)(LWgraph hGraph, LWgraphNode *rootNodes, size_t *numRootNodes);
typedef LWresult (LWDAAPI *PFN_lwGraphGetEdges_v10000)(LWgraph hGraph, LWgraphNode *from, LWgraphNode *to, size_t *numEdges);
typedef LWresult (LWDAAPI *PFN_lwGraphNodeGetDependencies_v10000)(LWgraphNode hNode, LWgraphNode *dependencies, size_t *numDependencies);
typedef LWresult (LWDAAPI *PFN_lwGraphNodeGetDependentNodes_v10000)(LWgraphNode hNode, LWgraphNode *dependentNodes, size_t *numDependentNodes);
typedef LWresult (LWDAAPI *PFN_lwGraphAddDependencies_v10000)(LWgraph hGraph, const LWgraphNode *from, const LWgraphNode *to, size_t numDependencies);
typedef LWresult (LWDAAPI *PFN_lwGraphRemoveDependencies_v10000)(LWgraph hGraph, const LWgraphNode *from, const LWgraphNode *to, size_t numDependencies);
typedef LWresult (LWDAAPI *PFN_lwGraphDestroyNode_v10000)(LWgraphNode hNode);
typedef LWresult (LWDAAPI *PFN_lwGraphInstantiate_v11000)(LWgraphExec *phGraphExec, LWgraph hGraph, LWgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
typedef LWresult (LWDAAPI *PFN_lwGraphExecKernelNodeSetParams_v10010)(LWgraphExec hGraphExec, LWgraphNode hNode, const LWDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphExecMemcpyNodeSetParams_v10020)(LWgraphExec hGraphExec, LWgraphNode hNode, const LWDA_MEMCPY3D_v2 *copyParams, LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwGraphExecMemsetNodeSetParams_v10020)(LWgraphExec hGraphExec, LWgraphNode hNode, const LWDA_MEMSET_NODE_PARAMS_v1 *memsetParams, LWcontext ctx);
typedef LWresult (LWDAAPI *PFN_lwGraphExecHostNodeSetParams_v10020)(LWgraphExec hGraphExec, LWgraphNode hNode, const LWDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphExecChildGraphNodeSetParams_v11010)(LWgraphExec hGraphExec, LWgraphNode hNode, LWgraph childGraph);
typedef LWresult (LWDAAPI *PFN_lwGraphExecEventRecordNodeSetEvent_v11010)(LWgraphExec hGraphExec, LWgraphNode hNode, LWevent event);
typedef LWresult (LWDAAPI *PFN_lwGraphExecEventWaitNodeSetEvent_v11010)(LWgraphExec hGraphExec, LWgraphNode hNode, LWevent event);
typedef LWresult (LWDAAPI *PFN_lwGraphExecExternalSemaphoresSignalNodeSetParams_v11020)(LWgraphExec hGraphExec, LWgraphNode hNode, const LWDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphExecExternalSemaphoresWaitNodeSetParams_v11020)(LWgraphExec hGraphExec, LWgraphNode hNode, const LWDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *nodeParams);
typedef LWresult (LWDAAPI *PFN_lwGraphUpload_v11010_ptsz)(LWgraphExec hGraphExec, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGraphLaunch_v10000_ptsz)(LWgraphExec hGraphExec, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGraphExecDestroy_v10000)(LWgraphExec hGraphExec);
typedef LWresult (LWDAAPI *PFN_lwGraphDestroy_v10000)(LWgraph hGraph);
typedef LWresult (LWDAAPI *PFN_lwGraphExelwpdate_v10020)(LWgraphExec hGraphExec, LWgraph hGraph, LWgraphNode *hErrorNode_out, LWgraphExelwpdateResult *updateResult_out);
typedef LWresult (LWDAAPI *PFN_lwGraphKernelNodeCopyAttributes_v11000)(LWgraphNode dst, LWgraphNode src);
typedef LWresult (LWDAAPI *PFN_lwGraphKernelNodeGetAttribute_v11000)(LWgraphNode hNode, LWkernelNodeAttrID attr, LWkernelNodeAttrValue_v1 *value_out);
typedef LWresult (LWDAAPI *PFN_lwGraphKernelNodeSetAttribute_v11000)(LWgraphNode hNode, LWkernelNodeAttrID attr, const LWkernelNodeAttrValue_v1 *value);
typedef LWresult (LWDAAPI *PFN_lwGraphDebugDotPrint_v11030)(LWgraph hGraph, const char *path, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwOclwpancyMaxActiveBlocksPerMultiprocessor_v6050)(int *numBlocks, LWfunction func, int blockSize, size_t dynamicSMemSize);
typedef LWresult (LWDAAPI *PFN_lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000)(int *numBlocks, LWfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwOclwpancyMaxPotentialBlockSize_v6050)(int *minGridSize, int *blockSize, LWfunction func, LWoclwpancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
typedef LWresult (LWDAAPI *PFN_lwOclwpancyMaxPotentialBlockSizeWithFlags_v7000)(int *minGridSize, int *blockSize, LWfunction func, LWoclwpancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwOclwpancyAvailableDynamicSMemPerBlock_v10020)(size_t *dynamicSmemSize, LWfunction func, int numBlocks, int blockSize);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetArray_v2000)(LWtexref hTexRef, LWarray hArray, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetMipmappedArray_v5000)(LWtexref hTexRef, LWmipmappedArray hMipmappedArray, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetAddress_v3020)(size_t *ByteOffset, LWtexref hTexRef, LWdeviceptr_v2 dptr, size_t bytes);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetAddress2D_v4010)(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR_v2 *desc, LWdeviceptr_v2 dptr, size_t Pitch);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetFormat_v2000)(LWtexref hTexRef, LWarray_format fmt, int NumPackedComponents);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetAddressMode_v2000)(LWtexref hTexRef, int dim, LWaddress_mode am);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetFilterMode_v2000)(LWtexref hTexRef, LWfilter_mode fm);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetMipmapFilterMode_v5000)(LWtexref hTexRef, LWfilter_mode fm);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetMipmapLevelBias_v5000)(LWtexref hTexRef, float bias);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetMipmapLevelClamp_v5000)(LWtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetMaxAnisotropy_v5000)(LWtexref hTexRef, unsigned int maxAniso);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetBorderColor_v8000)(LWtexref hTexRef, float *pBorderColor);
typedef LWresult (LWDAAPI *PFN_lwTexRefSetFlags_v2000)(LWtexref hTexRef, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetAddress_v3020)(LWdeviceptr_v2 *pdptr, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetArray_v2000)(LWarray *phArray, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetMipmappedArray_v5000)(LWmipmappedArray *phMipmappedArray, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetAddressMode_v2000)(LWaddress_mode *pam, LWtexref hTexRef, int dim);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetFilterMode_v2000)(LWfilter_mode *pfm, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetFormat_v2000)(LWarray_format *pFormat, int *pNumChannels, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetMipmapFilterMode_v5000)(LWfilter_mode *pfm, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetMipmapLevelBias_v5000)(float *pbias, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetMipmapLevelClamp_v5000)(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetMaxAnisotropy_v5000)(int *pmaxAniso, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetBorderColor_v8000)(float *pBorderColor, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefGetFlags_v2000)(unsigned int *pFlags, LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefCreate_v2000)(LWtexref *pTexRef);
typedef LWresult (LWDAAPI *PFN_lwTexRefDestroy_v2000)(LWtexref hTexRef);
typedef LWresult (LWDAAPI *PFN_lwSurfRefSetArray_v3000)(LWsurfref hSurfRef, LWarray hArray, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwSurfRefGetArray_v3000)(LWarray *phArray, LWsurfref hSurfRef);
typedef LWresult (LWDAAPI *PFN_lwTexObjectCreate_v5000)(LWtexObject_v1 *pTexObject, const LWDA_RESOURCE_DESC_v1 *pResDesc, const LWDA_TEXTURE_DESC_v1 *pTexDesc, const LWDA_RESOURCE_VIEW_DESC_v1 *pResViewDesc);
typedef LWresult (LWDAAPI *PFN_lwTexObjectDestroy_v5000)(LWtexObject_v1 texObject);
typedef LWresult (LWDAAPI *PFN_lwTexObjectGetResourceDesc_v5000)(LWDA_RESOURCE_DESC_v1 *pResDesc, LWtexObject_v1 texObject);
typedef LWresult (LWDAAPI *PFN_lwTexObjectGetTextureDesc_v5000)(LWDA_TEXTURE_DESC_v1 *pTexDesc, LWtexObject_v1 texObject);
typedef LWresult (LWDAAPI *PFN_lwTexObjectGetResourceViewDesc_v5000)(LWDA_RESOURCE_VIEW_DESC_v1 *pResViewDesc, LWtexObject_v1 texObject);
typedef LWresult (LWDAAPI *PFN_lwSurfObjectCreate_v5000)(LWsurfObject_v1 *pSurfObject, const LWDA_RESOURCE_DESC_v1 *pResDesc);
typedef LWresult (LWDAAPI *PFN_lwSurfObjectDestroy_v5000)(LWsurfObject_v1 surfObject);
typedef LWresult (LWDAAPI *PFN_lwSurfObjectGetResourceDesc_v5000)(LWDA_RESOURCE_DESC_v1 *pResDesc, LWsurfObject_v1 surfObject);
typedef LWresult (LWDAAPI *PFN_lwDeviceCanAccessPeer_v4000)(int *canAccessPeer, LWdevice_v1 dev, LWdevice_v1 peerDev);
typedef LWresult (LWDAAPI *PFN_lwCtxEnablePeerAccess_v4000)(LWcontext peerContext, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwCtxDisablePeerAccess_v4000)(LWcontext peerContext);
typedef LWresult (LWDAAPI *PFN_lwDeviceGetP2PAttribute_v8000)(int *value, LWdevice_P2PAttribute attrib, LWdevice_v1 srcDevice, LWdevice_v1 dstDevice);
typedef LWresult (LWDAAPI *PFN_lwGraphicsUnregisterResource_v3000)(LWgraphicsResource resource);
typedef LWresult (LWDAAPI *PFN_lwGraphicsSubResourceGetMappedArray_v3000)(LWarray *pArray, LWgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
typedef LWresult (LWDAAPI *PFN_lwGraphicsResourceGetMappedMipmappedArray_v5000)(LWmipmappedArray *pMipmappedArray, LWgraphicsResource resource);
typedef LWresult (LWDAAPI *PFN_lwGraphicsResourceGetMappedPointer_v3020)(LWdeviceptr_v2 *pDevPtr, size_t *pSize, LWgraphicsResource resource);
typedef LWresult (LWDAAPI *PFN_lwGraphicsResourceSetMapFlags_v6050)(LWgraphicsResource resource, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwGraphicsMapResources_v7000_ptsz)(unsigned int count, LWgraphicsResource *resources, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGraphicsUnmapResources_v7000_ptsz)(unsigned int count, LWgraphicsResource *resources, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGetExportTable_v3000)(const void **ppExportTable, const LWuuid *pExportTableId);
typedef LWresult (LWDAAPI *PFN_lwFuncGetModule_v11000)(LWmodule *hmod, LWfunction hfunc);
typedef LWresult (LWDAAPI *PFN_lwGetProcAddress_v11030)(const char *symbol, void **pfn, int driverVersion, lwuint64_t flags);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoD_v3020)(LWdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoH_v3020)(void *dstHost, LWdeviceptr_v2 srcDevice, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoD_v3020)(LWdeviceptr_v2 dstDevice, LWdeviceptr_v2 srcDevice, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoA_v3020)(LWarray dstArray, size_t dstOffset, LWdeviceptr_v2 srcDevice, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoD_v3020)(LWdeviceptr_v2 dstDevice, LWarray srcArray, size_t srcOffset, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoA_v3020)(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoH_v3020)(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoA_v3020)(LWarray dstArray, size_t dstOffset, LWarray srcArray, size_t srcOffset, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoAAsync_v3020)(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoHAsync_v3020)(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpy2D_v3020)(const LWDA_MEMCPY2D_v2 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpy2DUnaligned_v3020)(const LWDA_MEMCPY2D_v2 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3D_v3020)(const LWDA_MEMCPY3D_v2 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoDAsync_v3020)(LWdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoHAsync_v3020)(void *dstHost, LWdeviceptr_v2 srcDevice, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoDAsync_v3020)(LWdeviceptr_v2 dstDevice, LWdeviceptr_v2 srcDevice, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpy2DAsync_v3020)(const LWDA_MEMCPY2D_v2 *pCopy, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3DAsync_v3020)(const LWDA_MEMCPY3D_v2 *pCopy, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD8_v3020)(LWdeviceptr_v2 dstDevice, unsigned char uc, size_t N);
typedef LWresult (LWDAAPI *PFN_lwMemsetD16_v3020)(LWdeviceptr_v2 dstDevice, unsigned short us, size_t N);
typedef LWresult (LWDAAPI *PFN_lwMemsetD32_v3020)(LWdeviceptr_v2 dstDevice, unsigned int ui, size_t N);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D8_v3020)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D16_v3020)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D32_v3020)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
typedef LWresult (LWDAAPI *PFN_lwMemcpy_v4000)(LWdeviceptr_v2 dst, LWdeviceptr_v2 src, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyAsync_v4000)(LWdeviceptr_v2 dst, LWdeviceptr_v2 src, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpyPeer_v4000)(LWdeviceptr_v2 dstDevice, LWcontext dstContext, LWdeviceptr_v2 srcDevice, LWcontext srcContext, size_t ByteCount);
typedef LWresult (LWDAAPI *PFN_lwMemcpyPeerAsync_v4000)(LWdeviceptr_v2 dstDevice, LWcontext dstContext, LWdeviceptr_v2 srcDevice, LWcontext srcContext, size_t ByteCount, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3DPeer_v4000)(const LWDA_MEMCPY3D_PEER_v1 *pCopy);
typedef LWresult (LWDAAPI *PFN_lwMemcpy3DPeerAsync_v4000)(const LWDA_MEMCPY3D_PEER_v1 *pCopy, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD8Async_v3020)(LWdeviceptr_v2 dstDevice, unsigned char uc, size_t N, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD16Async_v3020)(LWdeviceptr_v2 dstDevice, unsigned short us, size_t N, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD32Async_v3020)(LWdeviceptr_v2 dstDevice, unsigned int ui, size_t N, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D8Async_v3020)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D16Async_v3020)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemsetD2D32Async_v3020)(LWdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwStreamGetPriority_v5050)(LWstream hStream, int *priority);
typedef LWresult (LWDAAPI *PFN_lwStreamGetFlags_v5050)(LWstream hStream, unsigned int *flags);
typedef LWresult (LWDAAPI *PFN_lwStreamGetCtx_v9020)(LWstream hStream, LWcontext *pctx);
typedef LWresult (LWDAAPI *PFN_lwStreamWaitEvent_v3020)(LWstream hStream, LWevent hEvent, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwStreamAddCallback_v5000)(LWstream hStream, LWstreamCallback callback, void *userData, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamAttachMemAsync_v6000)(LWstream hStream, LWdeviceptr_v2 dptr, size_t length, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamQuery_v2000)(LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwStreamSynchronize_v2000)(LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwEventRecord_v2000)(LWevent hEvent, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwEventRecordWithFlags_v11010)(LWevent hEvent, LWstream hStream, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwLaunchKernel_v4000)(LWfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, LWstream hStream, void **kernelParams, void **extra);
typedef LWresult (LWDAAPI *PFN_lwLaunchHostFunc_v10000)(LWstream hStream, LWhostFn fn, void *userData);
typedef LWresult (LWDAAPI *PFN_lwGraphicsMapResources_v3000)(unsigned int count, LWgraphicsResource *resources, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGraphicsUnmapResources_v3000)(unsigned int count, LWgraphicsResource *resources, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwStreamWriteValue32_v8000)(LWstream stream, LWdeviceptr_v2 addr, lwuint32_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamWaitValue32_v8000)(LWstream stream, LWdeviceptr_v2 addr, lwuint32_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamWriteValue64_v9000)(LWstream stream, LWdeviceptr_v2 addr, lwuint64_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamWaitValue64_v9000)(LWstream stream, LWdeviceptr_v2 addr, lwuint64_t value, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwStreamBatchMemOp_v8000)(LWstream stream, unsigned int count, LWstreamBatchMemOpParams *paramArray, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwMemPrefetchAsync_v8000)(LWdeviceptr_v2 devPtr, size_t count, LWdevice_v1 dstDevice, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwLaunchCooperativeKernel_v9000)(LWfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, LWstream hStream, void **kernelParams);
typedef LWresult (LWDAAPI *PFN_lwSignalExternalSemaphoresAsync_v10000)(const LWexternalSemaphore *extSemArray, const LWDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 *paramsArray, unsigned int numExtSems, LWstream stream);
typedef LWresult (LWDAAPI *PFN_lwWaitExternalSemaphoresAsync_v10000)(const LWexternalSemaphore *extSemArray, const LWDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 *paramsArray, unsigned int numExtSems, LWstream stream);
typedef LWresult (LWDAAPI *PFN_lwStreamBeginCapture_v10010)(LWstream hStream, LWstreamCaptureMode mode);
typedef LWresult (LWDAAPI *PFN_lwStreamEndCapture_v10000)(LWstream hStream, LWgraph *phGraph);
typedef LWresult (LWDAAPI *PFN_lwStreamIsCapturing_v10000)(LWstream hStream, LWstreamCaptureStatus *captureStatus);
typedef LWresult (LWDAAPI *PFN_lwStreamGetCaptureInfo_v10010)(LWstream hStream, LWstreamCaptureStatus *captureStatus_out, lwuint64_t *id_out);
typedef LWresult (LWDAAPI *PFN_lwStreamGetCaptureInfo_v11030)(LWstream hStream, LWstreamCaptureStatus *captureStatus_out, lwuint64_t *id_out, LWgraph *graph_out, const LWgraphNode **dependencies_out, size_t *numDependencies_out);
typedef LWresult (LWDAAPI *PFN_lwStreamUpdateCaptureDependencies_v11030)(LWstream hStream, LWgraphNode *dependencies, size_t numDependencies, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwGraphUpload_v11010)(LWgraphExec hGraph, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGraphLaunch_v10000)(LWgraphExec hGraph, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwStreamCopyAttributes_v11000)(LWstream dstStream, LWstream srcStream);
typedef LWresult (LWDAAPI *PFN_lwStreamGetAttribute_v11000)(LWstream hStream, LWstreamAttrID attr, LWstreamAttrValue_v1 *value);
typedef LWresult (LWDAAPI *PFN_lwStreamSetAttribute_v11000)(LWstream hStream, LWstreamAttrID attr, const LWstreamAttrValue_v1 *param);
typedef LWresult (LWDAAPI *PFN_lwMemMapArrayAsync_v11010)(LWarrayMapInfo_v1 *mapInfoList, unsigned int count, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemFreeAsync_v11020)(LWdeviceptr_v2 dptr, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemAllocAsync_v11020)(LWdeviceptr_v2 *dptr, size_t bytesize, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwMemAllocFromPoolAsync_v11020)(LWdeviceptr_v2 *dptr, size_t bytesize, LWmemoryPool pool, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwFlushGPUDirectRDMAWrites_v11030)(LWflushGPUDirectRDMAWritesTarget target, LWflushGPUDirectRDMAWritesScope scope);
typedef LWresult (LWDAAPI *PFN_lwUserObjectCreate_v11030)(LWuserObject *object_out, void *ptr, LWhostFn destroy, unsigned int initialRefcount, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwUserObjectRetain_v11030)(LWuserObject object, unsigned int count);
typedef LWresult (LWDAAPI *PFN_lwUserObjectRelease_v11030)(LWuserObject object, unsigned int count);
typedef LWresult (LWDAAPI *PFN_lwGraphRetainUserObject_v11030)(LWgraph graph, LWuserObject object, unsigned int count, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwGraphReleaseUserObject_v11030)(LWgraph graph, LWuserObject object, unsigned int count);

/*
 * Type definitions for older versioned functions in lwca.h
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
    typedef LWresult (LWDAAPI *PFN_lwMemHostRegister_v4000)(void *p, size_t bytesize, unsigned int Flags);
    typedef LWresult (LWDAAPI *PFN_lwGraphicsResourceSetMapFlags_v3000)(LWgraphicsResource resource, unsigned int flags);
    typedef LWresult (LWDAAPI *PFN_lwLinkCreate_v5050)(unsigned int numOptions, LWjit_option *options, void **optiolwalues, LWlinkState *stateOut);
    typedef LWresult (LWDAAPI *PFN_lwLinkAddData_v5050)(LWlinkState state, LWjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, LWjit_option *options, void **optiolwalues);
    typedef LWresult (LWDAAPI *PFN_lwLinkAddFile_v5050)(LWlinkState state, LWjitInputType type, const char *path, unsigned int numOptions, LWjit_option *options, void **optiolwalues);
    typedef LWresult (LWDAAPI *PFN_lwTexRefSetAddress2D_v3020)(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR_v2 *desc, LWdeviceptr_v2 dptr, size_t Pitch);
    typedef LWresult (LWDAAPI *PFN_lwDeviceTotalMem_v2000)(unsigned int *bytes, LWdevice_v1 dev);
    typedef LWresult (LWDAAPI *PFN_lwCtxCreate_v2000)(LWcontext *pctx, unsigned int flags, LWdevice_v1 dev);
    typedef LWresult (LWDAAPI *PFN_lwModuleGetGlobal_v2000)(LWdeviceptr_v1 *dptr, unsigned int *bytes, LWmodule hmod, const char *name);
    typedef LWresult (LWDAAPI *PFN_lwMemGetInfo_v2000)(unsigned int *free, unsigned int *total);
    typedef LWresult (LWDAAPI *PFN_lwMemAlloc_v2000)(LWdeviceptr_v1 *dptr, unsigned int bytesize);
    typedef LWresult (LWDAAPI *PFN_lwMemAllocPitch_v2000)(LWdeviceptr_v1 *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
    typedef LWresult (LWDAAPI *PFN_lwMemFree_v2000)(LWdeviceptr_v1 dptr);
    typedef LWresult (LWDAAPI *PFN_lwMemGetAddressRange_v2000)(LWdeviceptr_v1 *pbase, unsigned int *psize, LWdeviceptr_v1 dptr);
    typedef LWresult (LWDAAPI *PFN_lwMemAllocHost_v2000)(void **pp, unsigned int bytesize);
    typedef LWresult (LWDAAPI *PFN_lwMemHostGetDevicePointer_v2020)(LWdeviceptr_v1 *pdptr, void *p, unsigned int Flags);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoD_v2000)(LWdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoH_v2000)(void *dstHost, LWdeviceptr_v1 srcDevice, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoD_v2000)(LWdeviceptr_v1 dstDevice, LWdeviceptr_v1 srcDevice, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoA_v2000)(LWarray dstArray, unsigned int dstOffset, LWdeviceptr_v1 srcDevice, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoD_v2000)(LWdeviceptr_v1 dstDevice, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoA_v2000)(LWarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoH_v2000)(void *dstHost, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoA_v2000)(LWarray dstArray, unsigned int dstOffset, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoAAsync_v2000)(LWarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyAtoHAsync_v2000)(void *dstHost, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount, LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwMemcpy2D_v2000)(const LWDA_MEMCPY2D_v1 *pCopy);
    typedef LWresult (LWDAAPI *PFN_lwMemcpy2DUnaligned_v2000)(const LWDA_MEMCPY2D_v1 *pCopy);
    typedef LWresult (LWDAAPI *PFN_lwMemcpy3D_v2000)(const LWDA_MEMCPY3D_v1 *pCopy);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyHtoDAsync_v2000)(LWdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount, LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoHAsync_v2000)(void *dstHost, LWdeviceptr_v1 srcDevice, unsigned int ByteCount, LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwMemcpyDtoDAsync_v3000)(LWdeviceptr_v1 dstDevice, LWdeviceptr_v1 srcDevice, unsigned int ByteCount, LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwMemcpy2DAsync_v2000)(const LWDA_MEMCPY2D_v1 *pCopy, LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwMemcpy3DAsync_v2000)(const LWDA_MEMCPY3D_v1 *pCopy, LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwMemsetD8_v2000)(LWdeviceptr_v1 dstDevice, unsigned char uc, unsigned int N);
    typedef LWresult (LWDAAPI *PFN_lwMemsetD16_v2000)(LWdeviceptr_v1 dstDevice, unsigned short us, unsigned int N);
    typedef LWresult (LWDAAPI *PFN_lwMemsetD32_v2000)(LWdeviceptr_v1 dstDevice, unsigned int ui, unsigned int N);
    typedef LWresult (LWDAAPI *PFN_lwMemsetD2D8_v2000)(LWdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height);
    typedef LWresult (LWDAAPI *PFN_lwMemsetD2D16_v2000)(LWdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height);
    typedef LWresult (LWDAAPI *PFN_lwMemsetD2D32_v2000)(LWdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height);
    typedef LWresult (LWDAAPI *PFN_lwArrayCreate_v2000)(LWarray *pHandle, const LWDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray);
    typedef LWresult (LWDAAPI *PFN_lwArrayGetDescriptor_v2000)(LWDA_ARRAY_DESCRIPTOR_v1 *pArrayDescriptor, LWarray hArray);
    typedef LWresult (LWDAAPI *PFN_lwArray3DCreate_v2000)(LWarray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR_v1 *pAllocateArray);
    typedef LWresult (LWDAAPI *PFN_lwArray3DGetDescriptor_v2000)(LWDA_ARRAY3D_DESCRIPTOR_v1 *pArrayDescriptor, LWarray hArray);
    typedef LWresult (LWDAAPI *PFN_lwTexRefSetAddress_v2000)(unsigned int *ByteOffset, LWtexref hTexRef, LWdeviceptr_v1 dptr, unsigned int bytes);
    typedef LWresult (LWDAAPI *PFN_lwTexRefSetAddress2D_v2020)(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR_v1 *desc, LWdeviceptr_v1 dptr, unsigned int Pitch);
    typedef LWresult (LWDAAPI *PFN_lwTexRefGetAddress_v2000)(LWdeviceptr_v1 *pdptr, LWtexref hTexRef);
    typedef LWresult (LWDAAPI *PFN_lwGraphicsResourceGetMappedPointer_v3000)(LWdeviceptr_v1 *pDevPtr, unsigned int *pSize, LWgraphicsResource resource);
    typedef LWresult (LWDAAPI *PFN_lwCtxDestroy_v2000)(LWcontext ctx);
    typedef LWresult (LWDAAPI *PFN_lwCtxPopLwrrent_v2000)(LWcontext *pctx);
    typedef LWresult (LWDAAPI *PFN_lwCtxPushLwrrent_v2000)(LWcontext ctx);
    typedef LWresult (LWDAAPI *PFN_lwStreamDestroy_v2000)(LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwEventDestroy_v2000)(LWevent hEvent);
    typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxRelease_v7000)(LWdevice_v1 dev);
    typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxReset_v7000)(LWdevice_v1 dev);
    typedef LWresult (LWDAAPI *PFN_lwDevicePrimaryCtxSetFlags_v7000)(LWdevice_v1 dev, unsigned int flags);
    typedef LWresult (LWDAAPI *PFN_lwStreamBeginCapture_v10000)(LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwStreamBeginCapture_v10000_ptsz)(LWstream hStream);
    typedef LWresult (LWDAAPI *PFN_lwIpcOpenMemHandle_v4010)(LWdeviceptr_v2 *pdptr, LWipcMemHandle_v1 handle, unsigned int Flags);
    typedef LWresult (LWDAAPI *PFN_lwGraphInstantiate_v10000)(LWgraphExec *phGraphExec, LWgraph hGraph, LWgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
