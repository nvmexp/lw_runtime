// This file is generated.  Any changes you make will be lost during the next clean build.

// No dependent includes

// LWCA public interface, for type definitions and lw* function prototypes
#include "lwca.h"


// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct lwGetErrorString_params_st {
    LWresult error;
    const char **pStr;
} lwGetErrorString_params;

typedef struct lwGetErrorName_params_st {
    LWresult error;
    const char **pStr;
} lwGetErrorName_params;

typedef struct lwInit_params_st {
    unsigned int Flags;
} lwInit_params;

typedef struct lwDriverGetVersion_params_st {
    int *driverVersion;
} lwDriverGetVersion_params;

typedef struct lwDeviceGet_params_st {
    LWdevice *device;
    int ordinal;
} lwDeviceGet_params;

typedef struct lwDeviceGetCount_params_st {
    int *count;
} lwDeviceGetCount_params;

typedef struct lwDeviceGetName_params_st {
    char *name;
    int len;
    LWdevice dev;
} lwDeviceGetName_params;

typedef struct lwDeviceGetUuid_params_st {
    LWuuid *uuid;
    LWdevice dev;
} lwDeviceGetUuid_params;

typedef struct lwDeviceGetLuid_params_st {
    char *luid;
    unsigned int *deviceNodeMask;
    LWdevice dev;
} lwDeviceGetLuid_params;

typedef struct lwDeviceTotalMem_v2_params_st {
    size_t *bytes;
    LWdevice dev;
} lwDeviceTotalMem_v2_params;

typedef struct lwDeviceGetTexture1DLinearMaxWidth_params_st {
    size_t *maxWidthInElements;
    LWarray_format format;
    unsigned numChannels;
    LWdevice dev;
} lwDeviceGetTexture1DLinearMaxWidth_params;

typedef struct lwDeviceGetAttribute_params_st {
    int *pi;
    LWdevice_attribute attrib;
    LWdevice dev;
} lwDeviceGetAttribute_params;

typedef struct lwDeviceGetLwSciSyncAttributes_params_st {
    void *lwSciSyncAttrList;
    LWdevice dev;
    int flags;
} lwDeviceGetLwSciSyncAttributes_params;

typedef struct lwDeviceGetProperties_params_st {
    LWdevprop *prop;
    LWdevice dev;
} lwDeviceGetProperties_params;

typedef struct lwDeviceComputeCapability_params_st {
    int *major;
    int *minor;
    LWdevice dev;
} lwDeviceComputeCapability_params;

typedef struct lwDevicePrimaryCtxRetain_params_st {
    LWcontext *pctx;
    LWdevice dev;
} lwDevicePrimaryCtxRetain_params;

typedef struct lwDevicePrimaryCtxRelease_v2_params_st {
    LWdevice dev;
} lwDevicePrimaryCtxRelease_v2_params;

typedef struct lwDevicePrimaryCtxSetFlags_v2_params_st {
    LWdevice dev;
    unsigned int flags;
} lwDevicePrimaryCtxSetFlags_v2_params;

typedef struct lwDevicePrimaryCtxGetState_params_st {
    LWdevice dev;
    unsigned int *flags;
    int *active;
} lwDevicePrimaryCtxGetState_params;

typedef struct lwDevicePrimaryCtxReset_v2_params_st {
    LWdevice dev;
} lwDevicePrimaryCtxReset_v2_params;

typedef struct lwCtxCreate_v2_params_st {
    LWcontext *pctx;
    unsigned int flags;
    LWdevice dev;
} lwCtxCreate_v2_params;

typedef struct lwCtxDestroy_v2_params_st {
    LWcontext ctx;
} lwCtxDestroy_v2_params;

typedef struct lwCtxPushLwrrent_v2_params_st {
    LWcontext ctx;
} lwCtxPushLwrrent_v2_params;

typedef struct lwCtxPopLwrrent_v2_params_st {
    LWcontext *pctx;
} lwCtxPopLwrrent_v2_params;

typedef struct lwCtxSetLwrrent_params_st {
    LWcontext ctx;
} lwCtxSetLwrrent_params;

typedef struct lwCtxGetLwrrent_params_st {
    LWcontext *pctx;
} lwCtxGetLwrrent_params;

typedef struct lwCtxGetDevice_params_st {
    LWdevice *device;
} lwCtxGetDevice_params;

typedef struct lwCtxGetFlags_params_st {
    unsigned int *flags;
} lwCtxGetFlags_params;

typedef struct lwCtxSetLimit_params_st {
    LWlimit limit;
    size_t value;
} lwCtxSetLimit_params;

typedef struct lwCtxGetLimit_params_st {
    size_t *pvalue;
    LWlimit limit;
} lwCtxGetLimit_params;

typedef struct lwCtxGetCacheConfig_params_st {
    LWfunc_cache *pconfig;
} lwCtxGetCacheConfig_params;

typedef struct lwCtxSetCacheConfig_params_st {
    LWfunc_cache config;
} lwCtxSetCacheConfig_params;

typedef struct lwCtxGetSharedMemConfig_params_st {
    LWsharedconfig *pConfig;
} lwCtxGetSharedMemConfig_params;

typedef struct lwCtxSetSharedMemConfig_params_st {
    LWsharedconfig config;
} lwCtxSetSharedMemConfig_params;

typedef struct lwCtxGetApiVersion_params_st {
    LWcontext ctx;
    unsigned int *version;
} lwCtxGetApiVersion_params;

typedef struct lwCtxGetStreamPriorityRange_params_st {
    int *leastPriority;
    int *greatestPriority;
} lwCtxGetStreamPriorityRange_params;

typedef struct lwCtxAttach_params_st {
    LWcontext *pctx;
    unsigned int flags;
} lwCtxAttach_params;

typedef struct lwCtxDetach_params_st {
    LWcontext ctx;
} lwCtxDetach_params;

typedef struct lwModuleLoad_params_st {
    LWmodule *module;
    const char *fname;
} lwModuleLoad_params;

typedef struct lwModuleLoadData_params_st {
    LWmodule *module;
    const void *image;
} lwModuleLoadData_params;

typedef struct lwModuleLoadDataEx_params_st {
    LWmodule *module;
    const void *image;
    unsigned int numOptions;
    LWjit_option *options;
    void **optiolwalues;
} lwModuleLoadDataEx_params;

typedef struct lwModuleLoadFatBinary_params_st {
    LWmodule *module;
    const void *fatLwbin;
} lwModuleLoadFatBinary_params;

typedef struct lwModuleUnload_params_st {
    LWmodule hmod;
} lwModuleUnload_params;

typedef struct lwModuleGetFunction_params_st {
    LWfunction *hfunc;
    LWmodule hmod;
    const char *name;
} lwModuleGetFunction_params;

typedef struct lwModuleGetGlobal_v2_params_st {
    LWdeviceptr *dptr;
    size_t *bytes;
    LWmodule hmod;
    const char *name;
} lwModuleGetGlobal_v2_params;

typedef struct lwModuleGetTexRef_params_st {
    LWtexref *pTexRef;
    LWmodule hmod;
    const char *name;
} lwModuleGetTexRef_params;

typedef struct lwModuleGetSurfRef_params_st {
    LWsurfref *pSurfRef;
    LWmodule hmod;
    const char *name;
} lwModuleGetSurfRef_params;

typedef struct lwLinkCreate_v2_params_st {
    unsigned int numOptions;
    LWjit_option *options;
    void **optiolwalues;
    LWlinkState *stateOut;
} lwLinkCreate_v2_params;

typedef struct lwLinkAddData_v2_params_st {
    LWlinkState state;
    LWjitInputType type;
    void *data;
    size_t size;
    const char *name;
    unsigned int numOptions;
    LWjit_option *options;
    void **optiolwalues;
} lwLinkAddData_v2_params;

typedef struct lwLinkAddFile_v2_params_st {
    LWlinkState state;
    LWjitInputType type;
    const char *path;
    unsigned int numOptions;
    LWjit_option *options;
    void **optiolwalues;
} lwLinkAddFile_v2_params;

typedef struct lwLinkComplete_params_st {
    LWlinkState state;
    void **lwbinOut;
    size_t *sizeOut;
} lwLinkComplete_params;

typedef struct lwLinkDestroy_params_st {
    LWlinkState state;
} lwLinkDestroy_params;

typedef struct lwMemGetInfo_v2_params_st {
    size_t *free;
    size_t *total;
} lwMemGetInfo_v2_params;

typedef struct lwMemAlloc_v2_params_st {
    LWdeviceptr *dptr;
    size_t bytesize;
} lwMemAlloc_v2_params;

typedef struct lwMemAllocPitch_v2_params_st {
    LWdeviceptr *dptr;
    size_t *pPitch;
    size_t WidthInBytes;
    size_t Height;
    unsigned int ElementSizeBytes;
} lwMemAllocPitch_v2_params;

typedef struct lwMemFree_v2_params_st {
    LWdeviceptr dptr;
} lwMemFree_v2_params;

typedef struct lwMemGetAddressRange_v2_params_st {
    LWdeviceptr *pbase;
    size_t *psize;
    LWdeviceptr dptr;
} lwMemGetAddressRange_v2_params;

typedef struct lwMemAllocHost_v2_params_st {
    void **pp;
    size_t bytesize;
} lwMemAllocHost_v2_params;

typedef struct lwMemFreeHost_params_st {
    void *p;
} lwMemFreeHost_params;

typedef struct lwMemHostAlloc_params_st {
    void **pp;
    size_t bytesize;
    unsigned int Flags;
} lwMemHostAlloc_params;

typedef struct lwMemHostGetDevicePointer_v2_params_st {
    LWdeviceptr *pdptr;
    void *p;
    unsigned int Flags;
} lwMemHostGetDevicePointer_v2_params;

typedef struct lwMemHostGetFlags_params_st {
    unsigned int *pFlags;
    void *p;
} lwMemHostGetFlags_params;

typedef struct lwMemAllocManaged_params_st {
    LWdeviceptr *dptr;
    size_t bytesize;
    unsigned int flags;
} lwMemAllocManaged_params;

typedef struct lwDeviceGetByPCIBusId_params_st {
    LWdevice *dev;
    const char *pciBusId;
} lwDeviceGetByPCIBusId_params;

typedef struct lwDeviceGetPCIBusId_params_st {
    char *pciBusId;
    int len;
    LWdevice dev;
} lwDeviceGetPCIBusId_params;

typedef struct lwIpcGetEventHandle_params_st {
    LWipcEventHandle *pHandle;
    LWevent event;
} lwIpcGetEventHandle_params;

typedef struct lwIpcOpenEventHandle_params_st {
    LWevent *phEvent;
    LWipcEventHandle handle;
} lwIpcOpenEventHandle_params;

typedef struct lwIpcGetMemHandle_params_st {
    LWipcMemHandle *pHandle;
    LWdeviceptr dptr;
} lwIpcGetMemHandle_params;

typedef struct lwIpcOpenMemHandle_v2_params_st {
    LWdeviceptr *pdptr;
    LWipcMemHandle handle;
    unsigned int Flags;
} lwIpcOpenMemHandle_v2_params;

typedef struct lwIpcCloseMemHandle_params_st {
    LWdeviceptr dptr;
} lwIpcCloseMemHandle_params;

typedef struct lwMemHostRegister_v2_params_st {
    void *p;
    size_t bytesize;
    unsigned int Flags;
} lwMemHostRegister_v2_params;

typedef struct lwMemHostUnregister_params_st {
    void *p;
} lwMemHostUnregister_params;

typedef struct lwMemcpy_ptds_params_st {
    LWdeviceptr dst;
    LWdeviceptr src;
    size_t ByteCount;
} lwMemcpy_ptds_params;

typedef struct lwMemcpyPeer_ptds_params_st {
    LWdeviceptr dstDevice;
    LWcontext dstContext;
    LWdeviceptr srcDevice;
    LWcontext srcContext;
    size_t ByteCount;
} lwMemcpyPeer_ptds_params;

typedef struct lwMemcpyHtoD_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
} lwMemcpyHtoD_v2_ptds_params;

typedef struct lwMemcpyDtoH_v2_ptds_params_st {
    void *dstHost;
    LWdeviceptr srcDevice;
    size_t ByteCount;
} lwMemcpyDtoH_v2_ptds_params;

typedef struct lwMemcpyDtoD_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    LWdeviceptr srcDevice;
    size_t ByteCount;
} lwMemcpyDtoD_v2_ptds_params;

typedef struct lwMemcpyDtoA_v2_ptds_params_st {
    LWarray dstArray;
    size_t dstOffset;
    LWdeviceptr srcDevice;
    size_t ByteCount;
} lwMemcpyDtoA_v2_ptds_params;

typedef struct lwMemcpyAtoD_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} lwMemcpyAtoD_v2_ptds_params;

typedef struct lwMemcpyHtoA_v2_ptds_params_st {
    LWarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
} lwMemcpyHtoA_v2_ptds_params;

typedef struct lwMemcpyAtoH_v2_ptds_params_st {
    void *dstHost;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} lwMemcpyAtoH_v2_ptds_params;

typedef struct lwMemcpyAtoA_v2_ptds_params_st {
    LWarray dstArray;
    size_t dstOffset;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} lwMemcpyAtoA_v2_ptds_params;

typedef struct lwMemcpy2D_v2_ptds_params_st {
    const LWDA_MEMCPY2D *pCopy;
} lwMemcpy2D_v2_ptds_params;

typedef struct lwMemcpy2DUnaligned_v2_ptds_params_st {
    const LWDA_MEMCPY2D *pCopy;
} lwMemcpy2DUnaligned_v2_ptds_params;

typedef struct lwMemcpy3D_v2_ptds_params_st {
    const LWDA_MEMCPY3D *pCopy;
} lwMemcpy3D_v2_ptds_params;

typedef struct lwMemcpy3DPeer_ptds_params_st {
    const LWDA_MEMCPY3D_PEER *pCopy;
} lwMemcpy3DPeer_ptds_params;

typedef struct lwMemcpyAsync_ptsz_params_st {
    LWdeviceptr dst;
    LWdeviceptr src;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyAsync_ptsz_params;

typedef struct lwMemcpyPeerAsync_ptsz_params_st {
    LWdeviceptr dstDevice;
    LWcontext dstContext;
    LWdeviceptr srcDevice;
    LWcontext srcContext;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyPeerAsync_ptsz_params;

typedef struct lwMemcpyHtoDAsync_v2_ptsz_params_st {
    LWdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyHtoDAsync_v2_ptsz_params;

typedef struct lwMemcpyDtoHAsync_v2_ptsz_params_st {
    void *dstHost;
    LWdeviceptr srcDevice;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyDtoHAsync_v2_ptsz_params;

typedef struct lwMemcpyDtoDAsync_v2_ptsz_params_st {
    LWdeviceptr dstDevice;
    LWdeviceptr srcDevice;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyDtoDAsync_v2_ptsz_params;

typedef struct lwMemcpyHtoAAsync_v2_ptsz_params_st {
    LWarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyHtoAAsync_v2_ptsz_params;

typedef struct lwMemcpyAtoHAsync_v2_ptsz_params_st {
    void *dstHost;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyAtoHAsync_v2_ptsz_params;

typedef struct lwMemcpy2DAsync_v2_ptsz_params_st {
    const LWDA_MEMCPY2D *pCopy;
    LWstream hStream;
} lwMemcpy2DAsync_v2_ptsz_params;

typedef struct lwMemcpy3DAsync_v2_ptsz_params_st {
    const LWDA_MEMCPY3D *pCopy;
    LWstream hStream;
} lwMemcpy3DAsync_v2_ptsz_params;

typedef struct lwMemcpy3DPeerAsync_ptsz_params_st {
    const LWDA_MEMCPY3D_PEER *pCopy;
    LWstream hStream;
} lwMemcpy3DPeerAsync_ptsz_params;

typedef struct lwMemsetD8_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
} lwMemsetD8_v2_ptds_params;

typedef struct lwMemsetD16_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    unsigned short us;
    size_t N;
} lwMemsetD16_v2_ptds_params;

typedef struct lwMemsetD32_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
} lwMemsetD32_v2_ptds_params;

typedef struct lwMemsetD2D8_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
} lwMemsetD2D8_v2_ptds_params;

typedef struct lwMemsetD2D16_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
} lwMemsetD2D16_v2_ptds_params;

typedef struct lwMemsetD2D32_v2_ptds_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
} lwMemsetD2D32_v2_ptds_params;

typedef struct lwMemsetD8Async_ptsz_params_st {
    LWdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    LWstream hStream;
} lwMemsetD8Async_ptsz_params;

typedef struct lwMemsetD16Async_ptsz_params_st {
    LWdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    LWstream hStream;
} lwMemsetD16Async_ptsz_params;

typedef struct lwMemsetD32Async_ptsz_params_st {
    LWdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    LWstream hStream;
} lwMemsetD32Async_ptsz_params;

typedef struct lwMemsetD2D8Async_ptsz_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    LWstream hStream;
} lwMemsetD2D8Async_ptsz_params;

typedef struct lwMemsetD2D16Async_ptsz_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    LWstream hStream;
} lwMemsetD2D16Async_ptsz_params;

typedef struct lwMemsetD2D32Async_ptsz_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    LWstream hStream;
} lwMemsetD2D32Async_ptsz_params;

typedef struct lwArrayCreate_v2_params_st {
    LWarray *pHandle;
    const LWDA_ARRAY_DESCRIPTOR *pAllocateArray;
} lwArrayCreate_v2_params;

typedef struct lwArrayGetDescriptor_v2_params_st {
    LWDA_ARRAY_DESCRIPTOR *pArrayDescriptor;
    LWarray hArray;
} lwArrayGetDescriptor_v2_params;

typedef struct lwArrayDestroy_params_st {
    LWarray hArray;
} lwArrayDestroy_params;

typedef struct lwArray3DCreate_v2_params_st {
    LWarray *pHandle;
    const LWDA_ARRAY3D_DESCRIPTOR *pAllocateArray;
} lwArray3DCreate_v2_params;

typedef struct lwArray3DGetDescriptor_v2_params_st {
    LWDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor;
    LWarray hArray;
} lwArray3DGetDescriptor_v2_params;

typedef struct lwMipmappedArrayCreate_params_st {
    LWmipmappedArray *pHandle;
    const LWDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc;
    unsigned int numMipmapLevels;
} lwMipmappedArrayCreate_params;

typedef struct lwMipmappedArrayGetLevel_params_st {
    LWarray *pLevelArray;
    LWmipmappedArray hMipmappedArray;
    unsigned int level;
} lwMipmappedArrayGetLevel_params;

typedef struct lwMipmappedArrayDestroy_params_st {
    LWmipmappedArray hMipmappedArray;
} lwMipmappedArrayDestroy_params;

typedef struct lwMemAddressReserve_params_st {
    LWdeviceptr *ptr;
    size_t size;
    size_t alignment;
    LWdeviceptr addr;
    unsigned long long flags;
} lwMemAddressReserve_params;

typedef struct lwMemAddressFree_params_st {
    LWdeviceptr ptr;
    size_t size;
} lwMemAddressFree_params;

typedef struct lwMemCreate_params_st {
    LWmemGenericAllocationHandle *handle;
    size_t size;
    const LWmemAllocationProp *prop;
    unsigned long long flags;
} lwMemCreate_params;

typedef struct lwMemRelease_params_st {
    LWmemGenericAllocationHandle handle;
} lwMemRelease_params;

typedef struct lwMemMap_params_st {
    LWdeviceptr ptr;
    size_t size;
    size_t offset;
    LWmemGenericAllocationHandle handle;
    unsigned long long flags;
} lwMemMap_params;

typedef struct lwMemUnmap_params_st {
    LWdeviceptr ptr;
    size_t size;
} lwMemUnmap_params;

typedef struct lwMemSetAccess_params_st {
    LWdeviceptr ptr;
    size_t size;
    const LWmemAccessDesc *desc;
    size_t count;
} lwMemSetAccess_params;

typedef struct lwMemGetAccess_params_st {
    unsigned long long *flags;
    const LWmemLocation *location;
    LWdeviceptr ptr;
} lwMemGetAccess_params;

typedef struct lwMemExportToShareableHandle_params_st {
    void *shareableHandle;
    LWmemGenericAllocationHandle handle;
    LWmemAllocationHandleType handleType;
    unsigned long long flags;
} lwMemExportToShareableHandle_params;

typedef struct lwMemImportFromShareableHandle_params_st {
    LWmemGenericAllocationHandle *handle;
    void *osHandle;
    LWmemAllocationHandleType shHandleType;
} lwMemImportFromShareableHandle_params;

typedef struct lwMemGetAllocationGranularity_params_st {
    size_t *granularity;
    const LWmemAllocationProp *prop;
    LWmemAllocationGranularity_flags option;
} lwMemGetAllocationGranularity_params;

typedef struct lwMemGetAllocationPropertiesFromHandle_params_st {
    LWmemAllocationProp *prop;
    LWmemGenericAllocationHandle handle;
} lwMemGetAllocationPropertiesFromHandle_params;

typedef struct lwMemRetainAllocationHandle_params_st {
    LWmemGenericAllocationHandle *handle;
    void *addr;
} lwMemRetainAllocationHandle_params;

typedef struct lwPointerGetAttribute_params_st {
    void *data;
    LWpointer_attribute attribute;
    LWdeviceptr ptr;
} lwPointerGetAttribute_params;

typedef struct lwMemPrefetchAsync_ptsz_params_st {
    LWdeviceptr devPtr;
    size_t count;
    LWdevice dstDevice;
    LWstream hStream;
} lwMemPrefetchAsync_ptsz_params;

typedef struct lwMemAdvise_params_st {
    LWdeviceptr devPtr;
    size_t count;
    LWmem_advise advice;
    LWdevice device;
} lwMemAdvise_params;

typedef struct lwMemRangeGetAttribute_params_st {
    void *data;
    size_t dataSize;
    LWmem_range_attribute attribute;
    LWdeviceptr devPtr;
    size_t count;
} lwMemRangeGetAttribute_params;

typedef struct lwMemRangeGetAttributes_params_st {
    void **data;
    size_t *dataSizes;
    LWmem_range_attribute *attributes;
    size_t numAttributes;
    LWdeviceptr devPtr;
    size_t count;
} lwMemRangeGetAttributes_params;

typedef struct lwPointerSetAttribute_params_st {
    const void *value;
    LWpointer_attribute attribute;
    LWdeviceptr ptr;
} lwPointerSetAttribute_params;

typedef struct lwPointerGetAttributes_params_st {
    unsigned int numAttributes;
    LWpointer_attribute *attributes;
    void **data;
    LWdeviceptr ptr;
} lwPointerGetAttributes_params;

typedef struct lwStreamCreate_params_st {
    LWstream *phStream;
    unsigned int Flags;
} lwStreamCreate_params;

typedef struct lwStreamCreateWithPriority_params_st {
    LWstream *phStream;
    unsigned int flags;
    int priority;
} lwStreamCreateWithPriority_params;

typedef struct lwStreamGetPriority_ptsz_params_st {
    LWstream hStream;
    int *priority;
} lwStreamGetPriority_ptsz_params;

typedef struct lwStreamGetFlags_ptsz_params_st {
    LWstream hStream;
    unsigned int *flags;
} lwStreamGetFlags_ptsz_params;

typedef struct lwStreamGetCtx_ptsz_params_st {
    LWstream hStream;
    LWcontext *pctx;
} lwStreamGetCtx_ptsz_params;

typedef struct lwStreamWaitEvent_ptsz_params_st {
    LWstream hStream;
    LWevent hEvent;
    unsigned int Flags;
} lwStreamWaitEvent_ptsz_params;

typedef struct lwStreamAddCallback_ptsz_params_st {
    LWstream hStream;
    LWstreamCallback callback;
    void *userData;
    unsigned int flags;
} lwStreamAddCallback_ptsz_params;

typedef struct lwStreamBeginCapture_v2_ptsz_params_st {
    LWstream hStream;
    LWstreamCaptureMode mode;
} lwStreamBeginCapture_v2_ptsz_params;

typedef struct lwThreadExchangeStreamCaptureMode_params_st {
    LWstreamCaptureMode *mode;
} lwThreadExchangeStreamCaptureMode_params;

typedef struct lwStreamEndCapture_ptsz_params_st {
    LWstream hStream;
    LWgraph *phGraph;
} lwStreamEndCapture_ptsz_params;

typedef struct lwStreamIsCapturing_ptsz_params_st {
    LWstream hStream;
    LWstreamCaptureStatus *captureStatus;
} lwStreamIsCapturing_ptsz_params;

typedef struct lwStreamGetCaptureInfo_ptsz_params_st {
    LWstream hStream;
    LWstreamCaptureStatus *captureStatus;
    lwuint64_t *id;
} lwStreamGetCaptureInfo_ptsz_params;

typedef struct lwStreamAttachMemAsync_ptsz_params_st {
    LWstream hStream;
    LWdeviceptr dptr;
    size_t length;
    unsigned int flags;
} lwStreamAttachMemAsync_ptsz_params;

typedef struct lwStreamQuery_ptsz_params_st {
    LWstream hStream;
} lwStreamQuery_ptsz_params;

typedef struct lwStreamSynchronize_ptsz_params_st {
    LWstream hStream;
} lwStreamSynchronize_ptsz_params;

typedef struct lwStreamDestroy_v2_params_st {
    LWstream hStream;
} lwStreamDestroy_v2_params;

typedef struct lwStreamCopyAttributes_ptsz_params_st {
    LWstream dst;
    LWstream src;
} lwStreamCopyAttributes_ptsz_params;

typedef struct lwStreamGetAttribute_ptsz_params_st {
    LWstream hStream;
    LWstreamAttrID attr;
    LWstreamAttrValue *value_out;
} lwStreamGetAttribute_ptsz_params;

typedef struct lwStreamSetAttribute_ptsz_params_st {
    LWstream hStream;
    LWstreamAttrID attr;
    const LWstreamAttrValue *value;
} lwStreamSetAttribute_ptsz_params;

typedef struct lwEventCreate_params_st {
    LWevent *phEvent;
    unsigned int Flags;
} lwEventCreate_params;

typedef struct lwEventRecord_ptsz_params_st {
    LWevent hEvent;
    LWstream hStream;
} lwEventRecord_ptsz_params;

typedef struct lwEventQuery_params_st {
    LWevent hEvent;
} lwEventQuery_params;

typedef struct lwEventSynchronize_params_st {
    LWevent hEvent;
} lwEventSynchronize_params;

typedef struct lwEventDestroy_v2_params_st {
    LWevent hEvent;
} lwEventDestroy_v2_params;

typedef struct lwEventElapsedTime_params_st {
    float *pMilliseconds;
    LWevent hStart;
    LWevent hEnd;
} lwEventElapsedTime_params;

typedef struct lwImportExternalMemory_params_st {
    LWexternalMemory *extMem_out;
    const LWDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc;
} lwImportExternalMemory_params;

typedef struct lwExternalMemoryGetMappedBuffer_params_st {
    LWdeviceptr *devPtr;
    LWexternalMemory extMem;
    const LWDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc;
} lwExternalMemoryGetMappedBuffer_params;

typedef struct lwExternalMemoryGetMappedMipmappedArray_params_st {
    LWmipmappedArray *mipmap;
    LWexternalMemory extMem;
    const LWDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc;
} lwExternalMemoryGetMappedMipmappedArray_params;

typedef struct lwDestroyExternalMemory_params_st {
    LWexternalMemory extMem;
} lwDestroyExternalMemory_params;

typedef struct lwImportExternalSemaphore_params_st {
    LWexternalSemaphore *extSem_out;
    const LWDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc;
} lwImportExternalSemaphore_params;

typedef struct lwSignalExternalSemaphoresAsync_ptsz_params_st {
    const LWexternalSemaphore *extSemArray;
    const LWDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray;
    unsigned int numExtSems;
    LWstream stream;
} lwSignalExternalSemaphoresAsync_ptsz_params;

typedef struct lwWaitExternalSemaphoresAsync_ptsz_params_st {
    const LWexternalSemaphore *extSemArray;
    const LWDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray;
    unsigned int numExtSems;
    LWstream stream;
} lwWaitExternalSemaphoresAsync_ptsz_params;

typedef struct lwDestroyExternalSemaphore_params_st {
    LWexternalSemaphore extSem;
} lwDestroyExternalSemaphore_params;

typedef struct lwStreamWaitValue32_ptsz_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint32_t value;
    unsigned int flags;
} lwStreamWaitValue32_ptsz_params;

typedef struct lwStreamWaitValue64_ptsz_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint64_t value;
    unsigned int flags;
} lwStreamWaitValue64_ptsz_params;

typedef struct lwStreamWriteValue32_ptsz_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint32_t value;
    unsigned int flags;
} lwStreamWriteValue32_ptsz_params;

typedef struct lwStreamWriteValue64_ptsz_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint64_t value;
    unsigned int flags;
} lwStreamWriteValue64_ptsz_params;

typedef struct lwStreamBatchMemOp_ptsz_params_st {
    LWstream stream;
    unsigned int count;
    LWstreamBatchMemOpParams *paramArray;
    unsigned int flags;
} lwStreamBatchMemOp_ptsz_params;

typedef struct lwFuncGetAttribute_params_st {
    int *pi;
    LWfunction_attribute attrib;
    LWfunction hfunc;
} lwFuncGetAttribute_params;

typedef struct lwFuncSetAttribute_params_st {
    LWfunction hfunc;
    LWfunction_attribute attrib;
    int value;
} lwFuncSetAttribute_params;

typedef struct lwFuncSetCacheConfig_params_st {
    LWfunction hfunc;
    LWfunc_cache config;
} lwFuncSetCacheConfig_params;

typedef struct lwFuncSetSharedMemConfig_params_st {
    LWfunction hfunc;
    LWsharedconfig config;
} lwFuncSetSharedMemConfig_params;

typedef struct lwLaunchKernel_ptsz_params_st {
    LWfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    LWstream hStream;
    void **kernelParams;
    void **extra;
} lwLaunchKernel_ptsz_params;

typedef struct lwLaunchCooperativeKernel_ptsz_params_st {
    LWfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    LWstream hStream;
    void **kernelParams;
} lwLaunchCooperativeKernel_ptsz_params;

typedef struct lwLaunchCooperativeKernelMultiDevice_params_st {
    LWDA_LAUNCH_PARAMS *launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
} lwLaunchCooperativeKernelMultiDevice_params;

typedef struct lwLaunchHostFunc_ptsz_params_st {
    LWstream hStream;
    LWhostFn fn;
    void *userData;
} lwLaunchHostFunc_ptsz_params;

typedef struct lwFuncSetBlockShape_params_st {
    LWfunction hfunc;
    int x;
    int y;
    int z;
} lwFuncSetBlockShape_params;

typedef struct lwFuncSetSharedSize_params_st {
    LWfunction hfunc;
    unsigned int bytes;
} lwFuncSetSharedSize_params;

typedef struct lwParamSetSize_params_st {
    LWfunction hfunc;
    unsigned int numbytes;
} lwParamSetSize_params;

typedef struct lwParamSeti_params_st {
    LWfunction hfunc;
    int offset;
    unsigned int value;
} lwParamSeti_params;

typedef struct lwParamSetf_params_st {
    LWfunction hfunc;
    int offset;
    float value;
} lwParamSetf_params;

typedef struct lwParamSetv_params_st {
    LWfunction hfunc;
    int offset;
    void *ptr;
    unsigned int numbytes;
} lwParamSetv_params;

typedef struct lwLaunch_params_st {
    LWfunction f;
} lwLaunch_params;

typedef struct lwLaunchGrid_params_st {
    LWfunction f;
    int grid_width;
    int grid_height;
} lwLaunchGrid_params;

typedef struct lwLaunchGridAsync_params_st {
    LWfunction f;
    int grid_width;
    int grid_height;
    LWstream hStream;
} lwLaunchGridAsync_params;

typedef struct lwParamSetTexRef_params_st {
    LWfunction hfunc;
    int texunit;
    LWtexref hTexRef;
} lwParamSetTexRef_params;

typedef struct lwGraphCreate_params_st {
    LWgraph *phGraph;
    unsigned int flags;
} lwGraphCreate_params;

typedef struct lwGraphAddKernelNode_params_st {
    LWgraphNode *phGraphNode;
    LWgraph hGraph;
    const LWgraphNode *dependencies;
    size_t numDependencies;
    const LWDA_KERNEL_NODE_PARAMS *nodeParams;
} lwGraphAddKernelNode_params;

typedef struct lwGraphKernelNodeGetParams_params_st {
    LWgraphNode hNode;
    LWDA_KERNEL_NODE_PARAMS *nodeParams;
} lwGraphKernelNodeGetParams_params;

typedef struct lwGraphKernelNodeSetParams_params_st {
    LWgraphNode hNode;
    const LWDA_KERNEL_NODE_PARAMS *nodeParams;
} lwGraphKernelNodeSetParams_params;

typedef struct lwGraphAddMemcpyNode_params_st {
    LWgraphNode *phGraphNode;
    LWgraph hGraph;
    const LWgraphNode *dependencies;
    size_t numDependencies;
    const LWDA_MEMCPY3D *copyParams;
    LWcontext ctx;
} lwGraphAddMemcpyNode_params;

typedef struct lwGraphMemcpyNodeGetParams_params_st {
    LWgraphNode hNode;
    LWDA_MEMCPY3D *nodeParams;
} lwGraphMemcpyNodeGetParams_params;

typedef struct lwGraphMemcpyNodeSetParams_params_st {
    LWgraphNode hNode;
    const LWDA_MEMCPY3D *nodeParams;
} lwGraphMemcpyNodeSetParams_params;

typedef struct lwGraphAddMemsetNode_params_st {
    LWgraphNode *phGraphNode;
    LWgraph hGraph;
    const LWgraphNode *dependencies;
    size_t numDependencies;
    const LWDA_MEMSET_NODE_PARAMS *memsetParams;
    LWcontext ctx;
} lwGraphAddMemsetNode_params;

typedef struct lwGraphMemsetNodeGetParams_params_st {
    LWgraphNode hNode;
    LWDA_MEMSET_NODE_PARAMS *nodeParams;
} lwGraphMemsetNodeGetParams_params;

typedef struct lwGraphMemsetNodeSetParams_params_st {
    LWgraphNode hNode;
    const LWDA_MEMSET_NODE_PARAMS *nodeParams;
} lwGraphMemsetNodeSetParams_params;

typedef struct lwGraphAddHostNode_params_st {
    LWgraphNode *phGraphNode;
    LWgraph hGraph;
    const LWgraphNode *dependencies;
    size_t numDependencies;
    const LWDA_HOST_NODE_PARAMS *nodeParams;
} lwGraphAddHostNode_params;

typedef struct lwGraphHostNodeGetParams_params_st {
    LWgraphNode hNode;
    LWDA_HOST_NODE_PARAMS *nodeParams;
} lwGraphHostNodeGetParams_params;

typedef struct lwGraphHostNodeSetParams_params_st {
    LWgraphNode hNode;
    const LWDA_HOST_NODE_PARAMS *nodeParams;
} lwGraphHostNodeSetParams_params;

typedef struct lwGraphAddChildGraphNode_params_st {
    LWgraphNode *phGraphNode;
    LWgraph hGraph;
    const LWgraphNode *dependencies;
    size_t numDependencies;
    LWgraph childGraph;
} lwGraphAddChildGraphNode_params;

typedef struct lwGraphChildGraphNodeGetGraph_params_st {
    LWgraphNode hNode;
    LWgraph *phGraph;
} lwGraphChildGraphNodeGetGraph_params;

typedef struct lwGraphAddEmptyNode_params_st {
    LWgraphNode *phGraphNode;
    LWgraph hGraph;
    const LWgraphNode *dependencies;
    size_t numDependencies;
} lwGraphAddEmptyNode_params;

typedef struct lwGraphClone_params_st {
    LWgraph *phGraphClone;
    LWgraph originalGraph;
} lwGraphClone_params;

typedef struct lwGraphNodeFindInClone_params_st {
    LWgraphNode *phNode;
    LWgraphNode hOriginalNode;
    LWgraph hClonedGraph;
} lwGraphNodeFindInClone_params;

typedef struct lwGraphNodeGetType_params_st {
    LWgraphNode hNode;
    LWgraphNodeType *type;
} lwGraphNodeGetType_params;

typedef struct lwGraphGetNodes_params_st {
    LWgraph hGraph;
    LWgraphNode *nodes;
    size_t *numNodes;
} lwGraphGetNodes_params;

typedef struct lwGraphGetRootNodes_params_st {
    LWgraph hGraph;
    LWgraphNode *rootNodes;
    size_t *numRootNodes;
} lwGraphGetRootNodes_params;

typedef struct lwGraphGetEdges_params_st {
    LWgraph hGraph;
    LWgraphNode *from;
    LWgraphNode *to;
    size_t *numEdges;
} lwGraphGetEdges_params;

typedef struct lwGraphNodeGetDependencies_params_st {
    LWgraphNode hNode;
    LWgraphNode *dependencies;
    size_t *numDependencies;
} lwGraphNodeGetDependencies_params;

typedef struct lwGraphNodeGetDependentNodes_params_st {
    LWgraphNode hNode;
    LWgraphNode *dependentNodes;
    size_t *numDependentNodes;
} lwGraphNodeGetDependentNodes_params;

typedef struct lwGraphAddDependencies_params_st {
    LWgraph hGraph;
    const LWgraphNode *from;
    const LWgraphNode *to;
    size_t numDependencies;
} lwGraphAddDependencies_params;

typedef struct lwGraphRemoveDependencies_params_st {
    LWgraph hGraph;
    const LWgraphNode *from;
    const LWgraphNode *to;
    size_t numDependencies;
} lwGraphRemoveDependencies_params;

typedef struct lwGraphDestroyNode_params_st {
    LWgraphNode hNode;
} lwGraphDestroyNode_params;

typedef struct lwGraphInstantiate_v2_params_st {
    LWgraphExec *phGraphExec;
    LWgraph hGraph;
    LWgraphNode *phErrorNode;
    char *logBuffer;
    size_t bufferSize;
} lwGraphInstantiate_v2_params;

typedef struct lwGraphExecKernelNodeSetParams_params_st {
    LWgraphExec hGraphExec;
    LWgraphNode hNode;
    const LWDA_KERNEL_NODE_PARAMS *nodeParams;
} lwGraphExecKernelNodeSetParams_params;

typedef struct lwGraphExecMemcpyNodeSetParams_params_st {
    LWgraphExec hGraphExec;
    LWgraphNode hNode;
    const LWDA_MEMCPY3D *copyParams;
    LWcontext ctx;
} lwGraphExecMemcpyNodeSetParams_params;

typedef struct lwGraphExecMemsetNodeSetParams_params_st {
    LWgraphExec hGraphExec;
    LWgraphNode hNode;
    const LWDA_MEMSET_NODE_PARAMS *memsetParams;
    LWcontext ctx;
} lwGraphExecMemsetNodeSetParams_params;

typedef struct lwGraphExecHostNodeSetParams_params_st {
    LWgraphExec hGraphExec;
    LWgraphNode hNode;
    const LWDA_HOST_NODE_PARAMS *nodeParams;
} lwGraphExecHostNodeSetParams_params;

typedef struct lwGraphUpload_ptsz_params_st {
    LWgraphExec hGraphExec;
    LWstream hStream;
} lwGraphUpload_ptsz_params;

typedef struct lwGraphLaunch_ptsz_params_st {
    LWgraphExec hGraphExec;
    LWstream hStream;
} lwGraphLaunch_ptsz_params;

typedef struct lwGraphExecDestroy_params_st {
    LWgraphExec hGraphExec;
} lwGraphExecDestroy_params;

typedef struct lwGraphDestroy_params_st {
    LWgraph hGraph;
} lwGraphDestroy_params;

typedef struct lwGraphExelwpdate_params_st {
    LWgraphExec hGraphExec;
    LWgraph hGraph;
    LWgraphNode *hErrorNode_out;
    LWgraphExelwpdateResult *updateResult_out;
} lwGraphExelwpdate_params;

typedef struct lwGraphKernelNodeCopyAttributes_params_st {
    LWgraphNode dst;
    LWgraphNode src;
} lwGraphKernelNodeCopyAttributes_params;

typedef struct lwGraphKernelNodeGetAttribute_params_st {
    LWgraphNode hNode;
    LWkernelNodeAttrID attr;
    LWkernelNodeAttrValue *value_out;
} lwGraphKernelNodeGetAttribute_params;

typedef struct lwGraphKernelNodeSetAttribute_params_st {
    LWgraphNode hNode;
    LWkernelNodeAttrID attr;
    const LWkernelNodeAttrValue *value;
} lwGraphKernelNodeSetAttribute_params;

typedef struct lwOclwpancyMaxActiveBlocksPerMultiprocessor_params_st {
    int *numBlocks;
    LWfunction func;
    int blockSize;
    size_t dynamicSMemSize;
} lwOclwpancyMaxActiveBlocksPerMultiprocessor_params;

typedef struct lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags_params_st {
    int *numBlocks;
    LWfunction func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
} lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags_params;

typedef struct lwOclwpancyMaxPotentialBlockSize_params_st {
    int *minGridSize;
    int *blockSize;
    LWfunction func;
    LWoclwpancyB2DSize blockSizeToDynamicSMemSize;
    size_t dynamicSMemSize;
    int blockSizeLimit;
} lwOclwpancyMaxPotentialBlockSize_params;

typedef struct lwOclwpancyMaxPotentialBlockSizeWithFlags_params_st {
    int *minGridSize;
    int *blockSize;
    LWfunction func;
    LWoclwpancyB2DSize blockSizeToDynamicSMemSize;
    size_t dynamicSMemSize;
    int blockSizeLimit;
    unsigned int flags;
} lwOclwpancyMaxPotentialBlockSizeWithFlags_params;

typedef struct lwOclwpancyAvailableDynamicSMemPerBlock_params_st {
    size_t *dynamicSmemSize;
    LWfunction func;
    int numBlocks;
    int blockSize;
} lwOclwpancyAvailableDynamicSMemPerBlock_params;

typedef struct lwTexRefSetArray_params_st {
    LWtexref hTexRef;
    LWarray hArray;
    unsigned int Flags;
} lwTexRefSetArray_params;

typedef struct lwTexRefSetMipmappedArray_params_st {
    LWtexref hTexRef;
    LWmipmappedArray hMipmappedArray;
    unsigned int Flags;
} lwTexRefSetMipmappedArray_params;

typedef struct lwTexRefSetAddress_v2_params_st {
    size_t *ByteOffset;
    LWtexref hTexRef;
    LWdeviceptr dptr;
    size_t bytes;
} lwTexRefSetAddress_v2_params;

typedef struct lwTexRefSetAddress2D_v3_params_st {
    LWtexref hTexRef;
    const LWDA_ARRAY_DESCRIPTOR *desc;
    LWdeviceptr dptr;
    size_t Pitch;
} lwTexRefSetAddress2D_v3_params;

typedef struct lwTexRefSetFormat_params_st {
    LWtexref hTexRef;
    LWarray_format fmt;
    int NumPackedComponents;
} lwTexRefSetFormat_params;

typedef struct lwTexRefSetAddressMode_params_st {
    LWtexref hTexRef;
    int dim;
    LWaddress_mode am;
} lwTexRefSetAddressMode_params;

typedef struct lwTexRefSetFilterMode_params_st {
    LWtexref hTexRef;
    LWfilter_mode fm;
} lwTexRefSetFilterMode_params;

typedef struct lwTexRefSetMipmapFilterMode_params_st {
    LWtexref hTexRef;
    LWfilter_mode fm;
} lwTexRefSetMipmapFilterMode_params;

typedef struct lwTexRefSetMipmapLevelBias_params_st {
    LWtexref hTexRef;
    float bias;
} lwTexRefSetMipmapLevelBias_params;

typedef struct lwTexRefSetMipmapLevelClamp_params_st {
    LWtexref hTexRef;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
} lwTexRefSetMipmapLevelClamp_params;

typedef struct lwTexRefSetMaxAnisotropy_params_st {
    LWtexref hTexRef;
    unsigned int maxAniso;
} lwTexRefSetMaxAnisotropy_params;

typedef struct lwTexRefSetBorderColor_params_st {
    LWtexref hTexRef;
    float *pBorderColor;
} lwTexRefSetBorderColor_params;

typedef struct lwTexRefSetFlags_params_st {
    LWtexref hTexRef;
    unsigned int Flags;
} lwTexRefSetFlags_params;

typedef struct lwTexRefGetAddress_v2_params_st {
    LWdeviceptr *pdptr;
    LWtexref hTexRef;
} lwTexRefGetAddress_v2_params;

typedef struct lwTexRefGetArray_params_st {
    LWarray *phArray;
    LWtexref hTexRef;
} lwTexRefGetArray_params;

typedef struct lwTexRefGetMipmappedArray_params_st {
    LWmipmappedArray *phMipmappedArray;
    LWtexref hTexRef;
} lwTexRefGetMipmappedArray_params;

typedef struct lwTexRefGetAddressMode_params_st {
    LWaddress_mode *pam;
    LWtexref hTexRef;
    int dim;
} lwTexRefGetAddressMode_params;

typedef struct lwTexRefGetFilterMode_params_st {
    LWfilter_mode *pfm;
    LWtexref hTexRef;
} lwTexRefGetFilterMode_params;

typedef struct lwTexRefGetFormat_params_st {
    LWarray_format *pFormat;
    int *pNumChannels;
    LWtexref hTexRef;
} lwTexRefGetFormat_params;

typedef struct lwTexRefGetMipmapFilterMode_params_st {
    LWfilter_mode *pfm;
    LWtexref hTexRef;
} lwTexRefGetMipmapFilterMode_params;

typedef struct lwTexRefGetMipmapLevelBias_params_st {
    float *pbias;
    LWtexref hTexRef;
} lwTexRefGetMipmapLevelBias_params;

typedef struct lwTexRefGetMipmapLevelClamp_params_st {
    float *pminMipmapLevelClamp;
    float *pmaxMipmapLevelClamp;
    LWtexref hTexRef;
} lwTexRefGetMipmapLevelClamp_params;

typedef struct lwTexRefGetMaxAnisotropy_params_st {
    int *pmaxAniso;
    LWtexref hTexRef;
} lwTexRefGetMaxAnisotropy_params;

typedef struct lwTexRefGetBorderColor_params_st {
    float *pBorderColor;
    LWtexref hTexRef;
} lwTexRefGetBorderColor_params;

typedef struct lwTexRefGetFlags_params_st {
    unsigned int *pFlags;
    LWtexref hTexRef;
} lwTexRefGetFlags_params;

typedef struct lwTexRefCreate_params_st {
    LWtexref *pTexRef;
} lwTexRefCreate_params;

typedef struct lwTexRefDestroy_params_st {
    LWtexref hTexRef;
} lwTexRefDestroy_params;

typedef struct lwSurfRefSetArray_params_st {
    LWsurfref hSurfRef;
    LWarray hArray;
    unsigned int Flags;
} lwSurfRefSetArray_params;

typedef struct lwSurfRefGetArray_params_st {
    LWarray *phArray;
    LWsurfref hSurfRef;
} lwSurfRefGetArray_params;

typedef struct lwTexObjectCreate_params_st {
    LWtexObject *pTexObject;
    const LWDA_RESOURCE_DESC *pResDesc;
    const LWDA_TEXTURE_DESC *pTexDesc;
    const LWDA_RESOURCE_VIEW_DESC *pResViewDesc;
} lwTexObjectCreate_params;

typedef struct lwTexObjectDestroy_params_st {
    LWtexObject texObject;
} lwTexObjectDestroy_params;

typedef struct lwTexObjectGetResourceDesc_params_st {
    LWDA_RESOURCE_DESC *pResDesc;
    LWtexObject texObject;
} lwTexObjectGetResourceDesc_params;

typedef struct lwTexObjectGetTextureDesc_params_st {
    LWDA_TEXTURE_DESC *pTexDesc;
    LWtexObject texObject;
} lwTexObjectGetTextureDesc_params;

typedef struct lwTexObjectGetResourceViewDesc_params_st {
    LWDA_RESOURCE_VIEW_DESC *pResViewDesc;
    LWtexObject texObject;
} lwTexObjectGetResourceViewDesc_params;

typedef struct lwSurfObjectCreate_params_st {
    LWsurfObject *pSurfObject;
    const LWDA_RESOURCE_DESC *pResDesc;
} lwSurfObjectCreate_params;

typedef struct lwSurfObjectDestroy_params_st {
    LWsurfObject surfObject;
} lwSurfObjectDestroy_params;

typedef struct lwSurfObjectGetResourceDesc_params_st {
    LWDA_RESOURCE_DESC *pResDesc;
    LWsurfObject surfObject;
} lwSurfObjectGetResourceDesc_params;

typedef struct lwDeviceCanAccessPeer_params_st {
    int *canAccessPeer;
    LWdevice dev;
    LWdevice peerDev;
} lwDeviceCanAccessPeer_params;

typedef struct lwCtxEnablePeerAccess_params_st {
    LWcontext peerContext;
    unsigned int Flags;
} lwCtxEnablePeerAccess_params;

typedef struct lwCtxDisablePeerAccess_params_st {
    LWcontext peerContext;
} lwCtxDisablePeerAccess_params;

typedef struct lwDeviceGetP2PAttribute_params_st {
    int *value;
    LWdevice_P2PAttribute attrib;
    LWdevice srcDevice;
    LWdevice dstDevice;
} lwDeviceGetP2PAttribute_params;

typedef struct lwGraphicsUnregisterResource_params_st {
    LWgraphicsResource resource;
} lwGraphicsUnregisterResource_params;

typedef struct lwGraphicsSubResourceGetMappedArray_params_st {
    LWarray *pArray;
    LWgraphicsResource resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
} lwGraphicsSubResourceGetMappedArray_params;

typedef struct lwGraphicsResourceGetMappedMipmappedArray_params_st {
    LWmipmappedArray *pMipmappedArray;
    LWgraphicsResource resource;
} lwGraphicsResourceGetMappedMipmappedArray_params;

typedef struct lwGraphicsResourceGetMappedPointer_v2_params_st {
    LWdeviceptr *pDevPtr;
    size_t *pSize;
    LWgraphicsResource resource;
} lwGraphicsResourceGetMappedPointer_v2_params;

typedef struct lwGraphicsResourceSetMapFlags_v2_params_st {
    LWgraphicsResource resource;
    unsigned int flags;
} lwGraphicsResourceSetMapFlags_v2_params;

typedef struct lwGraphicsMapResources_ptsz_params_st {
    unsigned int count;
    LWgraphicsResource *resources;
    LWstream hStream;
} lwGraphicsMapResources_ptsz_params;

typedef struct lwGraphicsUnmapResources_ptsz_params_st {
    unsigned int count;
    LWgraphicsResource *resources;
    LWstream hStream;
} lwGraphicsUnmapResources_ptsz_params;

typedef struct lwGetExportTable_params_st {
    const void **ppExportTable;
    const LWuuid *pExportTableId;
} lwGetExportTable_params;

typedef struct lwFuncGetModule_params_st {
    LWmodule *hmod;
    LWfunction hfunc;
} lwFuncGetModule_params;

typedef struct lwMemHostRegister_params_st {
    void *p;
    size_t bytesize;
    unsigned int Flags;
} lwMemHostRegister_params;

typedef struct lwGraphicsResourceSetMapFlags_params_st {
    LWgraphicsResource resource;
    unsigned int flags;
} lwGraphicsResourceSetMapFlags_params;

typedef struct lwLinkCreate_params_st {
    unsigned int numOptions;
    LWjit_option *options;
    void **optiolwalues;
    LWlinkState *stateOut;
} lwLinkCreate_params;

typedef struct lwLinkAddData_params_st {
    LWlinkState state;
    LWjitInputType type;
    void *data;
    size_t size;
    const char *name;
    unsigned int numOptions;
    LWjit_option *options;
    void **optiolwalues;
} lwLinkAddData_params;

typedef struct lwLinkAddFile_params_st {
    LWlinkState state;
    LWjitInputType type;
    const char *path;
    unsigned int numOptions;
    LWjit_option *options;
    void **optiolwalues;
} lwLinkAddFile_params;

typedef struct lwTexRefSetAddress2D_v2_params_st {
    LWtexref hTexRef;
    const LWDA_ARRAY_DESCRIPTOR *desc;
    LWdeviceptr dptr;
    size_t Pitch;
} lwTexRefSetAddress2D_v2_params;

typedef struct lwDeviceTotalMem_params_st {
    unsigned int *bytes;
    LWdevice dev;
} lwDeviceTotalMem_params;

typedef struct lwCtxCreate_params_st {
    LWcontext *pctx;
    unsigned int flags;
    LWdevice dev;
} lwCtxCreate_params;

typedef struct lwModuleGetGlobal_params_st {
    LWdeviceptr_v1 *dptr;
    unsigned int *bytes;
    LWmodule hmod;
    const char *name;
} lwModuleGetGlobal_params;

typedef struct lwMemGetInfo_params_st {
    unsigned int *free;
    unsigned int *total;
} lwMemGetInfo_params;

typedef struct lwMemAlloc_params_st {
    LWdeviceptr_v1 *dptr;
    unsigned int bytesize;
} lwMemAlloc_params;

typedef struct lwMemAllocPitch_params_st {
    LWdeviceptr_v1 *dptr;
    unsigned int *pPitch;
    unsigned int WidthInBytes;
    unsigned int Height;
    unsigned int ElementSizeBytes;
} lwMemAllocPitch_params;

typedef struct lwMemFree_params_st {
    LWdeviceptr_v1 dptr;
} lwMemFree_params;

typedef struct lwMemGetAddressRange_params_st {
    LWdeviceptr_v1 *pbase;
    unsigned int *psize;
    LWdeviceptr_v1 dptr;
} lwMemGetAddressRange_params;

typedef struct lwMemAllocHost_params_st {
    void **pp;
    unsigned int bytesize;
} lwMemAllocHost_params;

typedef struct lwMemHostGetDevicePointer_params_st {
    LWdeviceptr_v1 *pdptr;
    void *p;
    unsigned int Flags;
} lwMemHostGetDevicePointer_params;

typedef struct lwMemcpyHtoD_params_st {
    LWdeviceptr_v1 dstDevice;
    const void *srcHost;
    unsigned int ByteCount;
} lwMemcpyHtoD_params;

typedef struct lwMemcpyDtoH_params_st {
    void *dstHost;
    LWdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
} lwMemcpyDtoH_params;

typedef struct lwMemcpyDtoD_params_st {
    LWdeviceptr_v1 dstDevice;
    LWdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
} lwMemcpyDtoD_params;

typedef struct lwMemcpyDtoA_params_st {
    LWarray dstArray;
    unsigned int dstOffset;
    LWdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
} lwMemcpyDtoA_params;

typedef struct lwMemcpyAtoD_params_st {
    LWdeviceptr_v1 dstDevice;
    LWarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
} lwMemcpyAtoD_params;

typedef struct lwMemcpyHtoA_params_st {
    LWarray dstArray;
    unsigned int dstOffset;
    const void *srcHost;
    unsigned int ByteCount;
} lwMemcpyHtoA_params;

typedef struct lwMemcpyAtoH_params_st {
    void *dstHost;
    LWarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
} lwMemcpyAtoH_params;

typedef struct lwMemcpyAtoA_params_st {
    LWarray dstArray;
    unsigned int dstOffset;
    LWarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
} lwMemcpyAtoA_params;

typedef struct lwMemcpyHtoAAsync_params_st {
    LWarray dstArray;
    unsigned int dstOffset;
    const void *srcHost;
    unsigned int ByteCount;
    LWstream hStream;
} lwMemcpyHtoAAsync_params;

typedef struct lwMemcpyAtoHAsync_params_st {
    void *dstHost;
    LWarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
    LWstream hStream;
} lwMemcpyAtoHAsync_params;

typedef struct lwMemcpy2D_params_st {
    const LWDA_MEMCPY2D_v1 *pCopy;
} lwMemcpy2D_params;

typedef struct lwMemcpy2DUnaligned_params_st {
    const LWDA_MEMCPY2D_v1 *pCopy;
} lwMemcpy2DUnaligned_params;

typedef struct lwMemcpy3D_params_st {
    const LWDA_MEMCPY3D_v1 *pCopy;
} lwMemcpy3D_params;

typedef struct lwMemcpyHtoDAsync_params_st {
    LWdeviceptr_v1 dstDevice;
    const void *srcHost;
    unsigned int ByteCount;
    LWstream hStream;
} lwMemcpyHtoDAsync_params;

typedef struct lwMemcpyDtoHAsync_params_st {
    void *dstHost;
    LWdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
    LWstream hStream;
} lwMemcpyDtoHAsync_params;

typedef struct lwMemcpyDtoDAsync_params_st {
    LWdeviceptr_v1 dstDevice;
    LWdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
    LWstream hStream;
} lwMemcpyDtoDAsync_params;

typedef struct lwMemcpy2DAsync_params_st {
    const LWDA_MEMCPY2D_v1 *pCopy;
    LWstream hStream;
} lwMemcpy2DAsync_params;

typedef struct lwMemcpy3DAsync_params_st {
    const LWDA_MEMCPY3D_v1 *pCopy;
    LWstream hStream;
} lwMemcpy3DAsync_params;

typedef struct lwMemsetD8_params_st {
    LWdeviceptr_v1 dstDevice;
    unsigned char uc;
    unsigned int N;
} lwMemsetD8_params;

typedef struct lwMemsetD16_params_st {
    LWdeviceptr_v1 dstDevice;
    unsigned short us;
    unsigned int N;
} lwMemsetD16_params;

typedef struct lwMemsetD32_params_st {
    LWdeviceptr_v1 dstDevice;
    unsigned int ui;
    unsigned int N;
} lwMemsetD32_params;

typedef struct lwMemsetD2D8_params_st {
    LWdeviceptr_v1 dstDevice;
    unsigned int dstPitch;
    unsigned char uc;
    unsigned int Width;
    unsigned int Height;
} lwMemsetD2D8_params;

typedef struct lwMemsetD2D16_params_st {
    LWdeviceptr_v1 dstDevice;
    unsigned int dstPitch;
    unsigned short us;
    unsigned int Width;
    unsigned int Height;
} lwMemsetD2D16_params;

typedef struct lwMemsetD2D32_params_st {
    LWdeviceptr_v1 dstDevice;
    unsigned int dstPitch;
    unsigned int ui;
    unsigned int Width;
    unsigned int Height;
} lwMemsetD2D32_params;

typedef struct lwArrayCreate_params_st {
    LWarray *pHandle;
    const LWDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray;
} lwArrayCreate_params;

typedef struct lwArrayGetDescriptor_params_st {
    LWDA_ARRAY_DESCRIPTOR_v1 *pArrayDescriptor;
    LWarray hArray;
} lwArrayGetDescriptor_params;

typedef struct lwArray3DCreate_params_st {
    LWarray *pHandle;
    const LWDA_ARRAY3D_DESCRIPTOR_v1 *pAllocateArray;
} lwArray3DCreate_params;

typedef struct lwArray3DGetDescriptor_params_st {
    LWDA_ARRAY3D_DESCRIPTOR_v1 *pArrayDescriptor;
    LWarray hArray;
} lwArray3DGetDescriptor_params;

typedef struct lwTexRefSetAddress_params_st {
    unsigned int *ByteOffset;
    LWtexref hTexRef;
    LWdeviceptr_v1 dptr;
    unsigned int bytes;
} lwTexRefSetAddress_params;

typedef struct lwTexRefSetAddress2D_params_st {
    LWtexref hTexRef;
    const LWDA_ARRAY_DESCRIPTOR_v1 *desc;
    LWdeviceptr_v1 dptr;
    unsigned int Pitch;
} lwTexRefSetAddress2D_params;

typedef struct lwTexRefGetAddress_params_st {
    LWdeviceptr_v1 *pdptr;
    LWtexref hTexRef;
} lwTexRefGetAddress_params;

typedef struct lwGraphicsResourceGetMappedPointer_params_st {
    LWdeviceptr_v1 *pDevPtr;
    unsigned int *pSize;
    LWgraphicsResource resource;
} lwGraphicsResourceGetMappedPointer_params;

typedef struct lwCtxDestroy_params_st {
    LWcontext ctx;
} lwCtxDestroy_params;

typedef struct lwCtxPopLwrrent_params_st {
    LWcontext *pctx;
} lwCtxPopLwrrent_params;

typedef struct lwCtxPushLwrrent_params_st {
    LWcontext ctx;
} lwCtxPushLwrrent_params;

typedef struct lwStreamDestroy_params_st {
    LWstream hStream;
} lwStreamDestroy_params;

typedef struct lwEventDestroy_params_st {
    LWevent hEvent;
} lwEventDestroy_params;

typedef struct lwDevicePrimaryCtxRelease_params_st {
    LWdevice dev;
} lwDevicePrimaryCtxRelease_params;

typedef struct lwDevicePrimaryCtxReset_params_st {
    LWdevice dev;
} lwDevicePrimaryCtxReset_params;

typedef struct lwDevicePrimaryCtxSetFlags_params_st {
    LWdevice dev;
    unsigned int flags;
} lwDevicePrimaryCtxSetFlags_params;

typedef struct lwMemcpyHtoD_v2_params_st {
    LWdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
} lwMemcpyHtoD_v2_params;

typedef struct lwMemcpyDtoH_v2_params_st {
    void *dstHost;
    LWdeviceptr srcDevice;
    size_t ByteCount;
} lwMemcpyDtoH_v2_params;

typedef struct lwMemcpyDtoD_v2_params_st {
    LWdeviceptr dstDevice;
    LWdeviceptr srcDevice;
    size_t ByteCount;
} lwMemcpyDtoD_v2_params;

typedef struct lwMemcpyDtoA_v2_params_st {
    LWarray dstArray;
    size_t dstOffset;
    LWdeviceptr srcDevice;
    size_t ByteCount;
} lwMemcpyDtoA_v2_params;

typedef struct lwMemcpyAtoD_v2_params_st {
    LWdeviceptr dstDevice;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} lwMemcpyAtoD_v2_params;

typedef struct lwMemcpyHtoA_v2_params_st {
    LWarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
} lwMemcpyHtoA_v2_params;

typedef struct lwMemcpyAtoH_v2_params_st {
    void *dstHost;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} lwMemcpyAtoH_v2_params;

typedef struct lwMemcpyAtoA_v2_params_st {
    LWarray dstArray;
    size_t dstOffset;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} lwMemcpyAtoA_v2_params;

typedef struct lwMemcpyHtoAAsync_v2_params_st {
    LWarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyHtoAAsync_v2_params;

typedef struct lwMemcpyAtoHAsync_v2_params_st {
    void *dstHost;
    LWarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyAtoHAsync_v2_params;

typedef struct lwMemcpy2D_v2_params_st {
    const LWDA_MEMCPY2D *pCopy;
} lwMemcpy2D_v2_params;

typedef struct lwMemcpy2DUnaligned_v2_params_st {
    const LWDA_MEMCPY2D *pCopy;
} lwMemcpy2DUnaligned_v2_params;

typedef struct lwMemcpy3D_v2_params_st {
    const LWDA_MEMCPY3D *pCopy;
} lwMemcpy3D_v2_params;

typedef struct lwMemcpyHtoDAsync_v2_params_st {
    LWdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyHtoDAsync_v2_params;

typedef struct lwMemcpyDtoHAsync_v2_params_st {
    void *dstHost;
    LWdeviceptr srcDevice;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyDtoHAsync_v2_params;

typedef struct lwMemcpyDtoDAsync_v2_params_st {
    LWdeviceptr dstDevice;
    LWdeviceptr srcDevice;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyDtoDAsync_v2_params;

typedef struct lwMemcpy2DAsync_v2_params_st {
    const LWDA_MEMCPY2D *pCopy;
    LWstream hStream;
} lwMemcpy2DAsync_v2_params;

typedef struct lwMemcpy3DAsync_v2_params_st {
    const LWDA_MEMCPY3D *pCopy;
    LWstream hStream;
} lwMemcpy3DAsync_v2_params;

typedef struct lwMemsetD8_v2_params_st {
    LWdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
} lwMemsetD8_v2_params;

typedef struct lwMemsetD16_v2_params_st {
    LWdeviceptr dstDevice;
    unsigned short us;
    size_t N;
} lwMemsetD16_v2_params;

typedef struct lwMemsetD32_v2_params_st {
    LWdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
} lwMemsetD32_v2_params;

typedef struct lwMemsetD2D8_v2_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
} lwMemsetD2D8_v2_params;

typedef struct lwMemsetD2D16_v2_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
} lwMemsetD2D16_v2_params;

typedef struct lwMemsetD2D32_v2_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
} lwMemsetD2D32_v2_params;

typedef struct lwMemcpy_params_st {
    LWdeviceptr dst;
    LWdeviceptr src;
    size_t ByteCount;
} lwMemcpy_params;

typedef struct lwMemcpyAsync_params_st {
    LWdeviceptr dst;
    LWdeviceptr src;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyAsync_params;

typedef struct lwMemcpyPeer_params_st {
    LWdeviceptr dstDevice;
    LWcontext dstContext;
    LWdeviceptr srcDevice;
    LWcontext srcContext;
    size_t ByteCount;
} lwMemcpyPeer_params;

typedef struct lwMemcpyPeerAsync_params_st {
    LWdeviceptr dstDevice;
    LWcontext dstContext;
    LWdeviceptr srcDevice;
    LWcontext srcContext;
    size_t ByteCount;
    LWstream hStream;
} lwMemcpyPeerAsync_params;

typedef struct lwMemcpy3DPeer_params_st {
    const LWDA_MEMCPY3D_PEER *pCopy;
} lwMemcpy3DPeer_params;

typedef struct lwMemcpy3DPeerAsync_params_st {
    const LWDA_MEMCPY3D_PEER *pCopy;
    LWstream hStream;
} lwMemcpy3DPeerAsync_params;

typedef struct lwMemsetD8Async_params_st {
    LWdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    LWstream hStream;
} lwMemsetD8Async_params;

typedef struct lwMemsetD16Async_params_st {
    LWdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    LWstream hStream;
} lwMemsetD16Async_params;

typedef struct lwMemsetD32Async_params_st {
    LWdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    LWstream hStream;
} lwMemsetD32Async_params;

typedef struct lwMemsetD2D8Async_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    LWstream hStream;
} lwMemsetD2D8Async_params;

typedef struct lwMemsetD2D16Async_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    LWstream hStream;
} lwMemsetD2D16Async_params;

typedef struct lwMemsetD2D32Async_params_st {
    LWdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    LWstream hStream;
} lwMemsetD2D32Async_params;

typedef struct lwStreamGetPriority_params_st {
    LWstream hStream;
    int *priority;
} lwStreamGetPriority_params;

typedef struct lwStreamGetFlags_params_st {
    LWstream hStream;
    unsigned int *flags;
} lwStreamGetFlags_params;

typedef struct lwStreamGetCtx_params_st {
    LWstream hStream;
    LWcontext *pctx;
} lwStreamGetCtx_params;

typedef struct lwStreamWaitEvent_params_st {
    LWstream hStream;
    LWevent hEvent;
    unsigned int Flags;
} lwStreamWaitEvent_params;

typedef struct lwStreamAddCallback_params_st {
    LWstream hStream;
    LWstreamCallback callback;
    void *userData;
    unsigned int flags;
} lwStreamAddCallback_params;

typedef struct lwStreamAttachMemAsync_params_st {
    LWstream hStream;
    LWdeviceptr dptr;
    size_t length;
    unsigned int flags;
} lwStreamAttachMemAsync_params;

typedef struct lwStreamQuery_params_st {
    LWstream hStream;
} lwStreamQuery_params;

typedef struct lwStreamSynchronize_params_st {
    LWstream hStream;
} lwStreamSynchronize_params;

typedef struct lwEventRecord_params_st {
    LWevent hEvent;
    LWstream hStream;
} lwEventRecord_params;

typedef struct lwLaunchKernel_params_st {
    LWfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    LWstream hStream;
    void **kernelParams;
    void **extra;
} lwLaunchKernel_params;

typedef struct lwLaunchHostFunc_params_st {
    LWstream hStream;
    LWhostFn fn;
    void *userData;
} lwLaunchHostFunc_params;

typedef struct lwGraphicsMapResources_params_st {
    unsigned int count;
    LWgraphicsResource *resources;
    LWstream hStream;
} lwGraphicsMapResources_params;

typedef struct lwGraphicsUnmapResources_params_st {
    unsigned int count;
    LWgraphicsResource *resources;
    LWstream hStream;
} lwGraphicsUnmapResources_params;

typedef struct lwStreamWriteValue32_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint32_t value;
    unsigned int flags;
} lwStreamWriteValue32_params;

typedef struct lwStreamWaitValue32_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint32_t value;
    unsigned int flags;
} lwStreamWaitValue32_params;

typedef struct lwStreamWriteValue64_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint64_t value;
    unsigned int flags;
} lwStreamWriteValue64_params;

typedef struct lwStreamWaitValue64_params_st {
    LWstream stream;
    LWdeviceptr addr;
    lwuint64_t value;
    unsigned int flags;
} lwStreamWaitValue64_params;

typedef struct lwStreamBatchMemOp_params_st {
    LWstream stream;
    unsigned int count;
    LWstreamBatchMemOpParams *paramArray;
    unsigned int flags;
} lwStreamBatchMemOp_params;

typedef struct lwMemPrefetchAsync_params_st {
    LWdeviceptr devPtr;
    size_t count;
    LWdevice dstDevice;
    LWstream hStream;
} lwMemPrefetchAsync_params;

typedef struct lwLaunchCooperativeKernel_params_st {
    LWfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    LWstream hStream;
    void **kernelParams;
} lwLaunchCooperativeKernel_params;

typedef struct lwSignalExternalSemaphoresAsync_params_st {
    const LWexternalSemaphore *extSemArray;
    const LWDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray;
    unsigned int numExtSems;
    LWstream stream;
} lwSignalExternalSemaphoresAsync_params;

typedef struct lwWaitExternalSemaphoresAsync_params_st {
    const LWexternalSemaphore *extSemArray;
    const LWDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray;
    unsigned int numExtSems;
    LWstream stream;
} lwWaitExternalSemaphoresAsync_params;

typedef struct lwStreamBeginCapture_params_st {
    LWstream hStream;
} lwStreamBeginCapture_params;

typedef struct lwStreamBeginCapture_ptsz_params_st {
    LWstream hStream;
} lwStreamBeginCapture_ptsz_params;

typedef struct lwStreamBeginCapture_v2_params_st {
    LWstream hStream;
    LWstreamCaptureMode mode;
} lwStreamBeginCapture_v2_params;

typedef struct lwStreamEndCapture_params_st {
    LWstream hStream;
    LWgraph *phGraph;
} lwStreamEndCapture_params;

typedef struct lwStreamIsCapturing_params_st {
    LWstream hStream;
    LWstreamCaptureStatus *captureStatus;
} lwStreamIsCapturing_params;

typedef struct lwStreamGetCaptureInfo_params_st {
    LWstream hStream;
    LWstreamCaptureStatus *captureStatus;
    lwuint64_t *id;
} lwStreamGetCaptureInfo_params;

typedef struct lwGraphUpload_params_st {
    LWgraphExec hGraph;
    LWstream hStream;
} lwGraphUpload_params;

typedef struct lwGraphLaunch_params_st {
    LWgraphExec hGraph;
    LWstream hStream;
} lwGraphLaunch_params;

typedef struct lwStreamCopyAttributes_params_st {
    LWstream dstStream;
    LWstream srcStream;
} lwStreamCopyAttributes_params;

typedef struct lwStreamGetAttribute_params_st {
    LWstream hStream;
    LWstreamAttrID attr;
    LWstreamAttrValue *value;
} lwStreamGetAttribute_params;

typedef struct lwStreamSetAttribute_params_st {
    LWstream hStream;
    LWstreamAttrID attr;
    const LWstreamAttrValue *param;
} lwStreamSetAttribute_params;

typedef struct lwIpcOpenMemHandle_params_st {
    LWdeviceptr *pdptr;
    LWipcMemHandle handle;
    unsigned int Flags;
} lwIpcOpenMemHandle_params;

typedef struct lwGraphInstantiate_params_st {
    LWgraphExec *phGraphExec;
    LWgraph hGraph;
    LWgraphNode *phErrorNode;
    char *logBuffer;
    size_t bufferSize;
} lwGraphInstantiate_params;
