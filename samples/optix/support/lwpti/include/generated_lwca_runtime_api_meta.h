// This file is generated.  Any changes you make will be lost during the next clean build.

// LWCA public interface, for type definitions and api function prototypes
#include "lwda_runtime_api.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Lwrrently used parameter trace structures 
typedef struct lwdaDeviceSetLimit_v3020_params_st {
    enum lwdaLimit limit;
    size_t value;
} lwdaDeviceSetLimit_v3020_params;

typedef struct lwdaDeviceGetLimit_v3020_params_st {
    size_t *pValue;
    enum lwdaLimit limit;
} lwdaDeviceGetLimit_v3020_params;

typedef struct lwdaDeviceGetTexture1DLinearMaxWidth_v11010_params_st {
    size_t *maxWidthInElements;
    const struct lwdaChannelFormatDesc *fmtDesc;
    int device;
} lwdaDeviceGetTexture1DLinearMaxWidth_v11010_params;

typedef struct lwdaDeviceGetCacheConfig_v3020_params_st {
    enum lwdaFuncCache *pCacheConfig;
} lwdaDeviceGetCacheConfig_v3020_params;

typedef struct lwdaDeviceGetStreamPriorityRange_v5050_params_st {
    int *leastPriority;
    int *greatestPriority;
} lwdaDeviceGetStreamPriorityRange_v5050_params;

typedef struct lwdaDeviceSetCacheConfig_v3020_params_st {
    enum lwdaFuncCache cacheConfig;
} lwdaDeviceSetCacheConfig_v3020_params;

typedef struct lwdaDeviceGetSharedMemConfig_v4020_params_st {
    enum lwdaSharedMemConfig *pConfig;
} lwdaDeviceGetSharedMemConfig_v4020_params;

typedef struct lwdaDeviceSetSharedMemConfig_v4020_params_st {
    enum lwdaSharedMemConfig config;
} lwdaDeviceSetSharedMemConfig_v4020_params;

typedef struct lwdaDeviceGetByPCIBusId_v4010_params_st {
    int *device;
    const char *pciBusId;
} lwdaDeviceGetByPCIBusId_v4010_params;

typedef struct lwdaDeviceGetPCIBusId_v4010_params_st {
    char *pciBusId;
    int len;
    int device;
} lwdaDeviceGetPCIBusId_v4010_params;

typedef struct lwdaIpcGetEventHandle_v4010_params_st {
    lwdaIpcEventHandle_t *handle;
    lwdaEvent_t event;
} lwdaIpcGetEventHandle_v4010_params;

typedef struct lwdaIpcOpenEventHandle_v4010_params_st {
    lwdaEvent_t *event;
    lwdaIpcEventHandle_t handle;
} lwdaIpcOpenEventHandle_v4010_params;

typedef struct lwdaIpcGetMemHandle_v4010_params_st {
    lwdaIpcMemHandle_t *handle;
    void *devPtr;
} lwdaIpcGetMemHandle_v4010_params;

typedef struct lwdaIpcOpenMemHandle_v4010_params_st {
    void **devPtr;
    lwdaIpcMemHandle_t handle;
    unsigned int flags;
} lwdaIpcOpenMemHandle_v4010_params;

typedef struct lwdaIpcCloseMemHandle_v4010_params_st {
    void *devPtr;
} lwdaIpcCloseMemHandle_v4010_params;

typedef struct lwdaGetErrorName_v6050_params_st {
    lwdaError_t error;
} lwdaGetErrorName_v6050_params;

typedef struct lwdaGetErrorString_v3020_params_st {
    lwdaError_t error;
} lwdaGetErrorString_v3020_params;

typedef struct lwdaGetDeviceCount_v3020_params_st {
    int *count;
} lwdaGetDeviceCount_v3020_params;

typedef struct lwdaGetDeviceProperties_v3020_params_st {
    struct lwdaDeviceProp *prop;
    int device;
} lwdaGetDeviceProperties_v3020_params;

typedef struct lwdaDeviceGetAttribute_v5000_params_st {
    int *value;
    enum lwdaDeviceAttr attr;
    int device;
} lwdaDeviceGetAttribute_v5000_params;

typedef struct lwdaDeviceGetLwSciSyncAttributes_v10020_params_st {
    void *lwSciSyncAttrList;
    int device;
    int flags;
} lwdaDeviceGetLwSciSyncAttributes_v10020_params;

typedef struct lwdaDeviceGetP2PAttribute_v8000_params_st {
    int *value;
    enum lwdaDeviceP2PAttr attr;
    int srcDevice;
    int dstDevice;
} lwdaDeviceGetP2PAttribute_v8000_params;

typedef struct lwdaChooseDevice_v3020_params_st {
    int *device;
    const struct lwdaDeviceProp *prop;
} lwdaChooseDevice_v3020_params;

typedef struct lwdaSetDevice_v3020_params_st {
    int device;
} lwdaSetDevice_v3020_params;

typedef struct lwdaGetDevice_v3020_params_st {
    int *device;
} lwdaGetDevice_v3020_params;

typedef struct lwdaSetValidDevices_v3020_params_st {
    int *device_arr;
    int len;
} lwdaSetValidDevices_v3020_params;

typedef struct lwdaSetDeviceFlags_v3020_params_st {
    unsigned int flags;
} lwdaSetDeviceFlags_v3020_params;

typedef struct lwdaGetDeviceFlags_v7000_params_st {
    unsigned int *flags;
} lwdaGetDeviceFlags_v7000_params;

typedef struct lwdaStreamCreate_v3020_params_st {
    lwdaStream_t *pStream;
} lwdaStreamCreate_v3020_params;

typedef struct lwdaStreamCreateWithFlags_v5000_params_st {
    lwdaStream_t *pStream;
    unsigned int flags;
} lwdaStreamCreateWithFlags_v5000_params;

typedef struct lwdaStreamCreateWithPriority_v5050_params_st {
    lwdaStream_t *pStream;
    unsigned int flags;
    int priority;
} lwdaStreamCreateWithPriority_v5050_params;

typedef struct lwdaStreamGetPriority_ptsz_v7000_params_st {
    lwdaStream_t hStream;
    int *priority;
} lwdaStreamGetPriority_ptsz_v7000_params;

typedef struct lwdaStreamGetFlags_ptsz_v7000_params_st {
    lwdaStream_t hStream;
    unsigned int *flags;
} lwdaStreamGetFlags_ptsz_v7000_params;

typedef struct lwdaStreamCopyAttributes_ptsz_v11000_params_st {
    lwdaStream_t dst;
    lwdaStream_t src;
} lwdaStreamCopyAttributes_ptsz_v11000_params;

typedef struct lwdaStreamGetAttribute_ptsz_v11000_params_st {
    lwdaStream_t hStream;
    enum lwdaStreamAttrID attr;
    union lwdaStreamAttrValue *value_out;
} lwdaStreamGetAttribute_ptsz_v11000_params;

typedef struct lwdaStreamSetAttribute_ptsz_v11000_params_st {
    lwdaStream_t hStream;
    enum lwdaStreamAttrID attr;
    const union lwdaStreamAttrValue *value;
} lwdaStreamSetAttribute_ptsz_v11000_params;

typedef struct lwdaStreamDestroy_v5050_params_st {
    lwdaStream_t stream;
} lwdaStreamDestroy_v5050_params;

typedef struct lwdaStreamWaitEvent_ptsz_v7000_params_st {
    lwdaStream_t stream;
    lwdaEvent_t event;
    unsigned int flags;
} lwdaStreamWaitEvent_ptsz_v7000_params;

typedef struct lwdaStreamAddCallback_ptsz_v7000_params_st {
    lwdaStream_t stream;
    lwdaStreamCallback_t callback;
    void *userData;
    unsigned int flags;
} lwdaStreamAddCallback_ptsz_v7000_params;

typedef struct lwdaStreamSynchronize_ptsz_v7000_params_st {
    lwdaStream_t stream;
} lwdaStreamSynchronize_ptsz_v7000_params;

typedef struct lwdaStreamQuery_ptsz_v7000_params_st {
    lwdaStream_t stream;
} lwdaStreamQuery_ptsz_v7000_params;

typedef struct lwdaStreamAttachMemAsync_ptsz_v7000_params_st {
    lwdaStream_t stream;
    void *devPtr;
    size_t length;
    unsigned int flags;
} lwdaStreamAttachMemAsync_ptsz_v7000_params;

typedef struct lwdaStreamBeginCapture_ptsz_v10000_params_st {
    lwdaStream_t stream;
    enum lwdaStreamCaptureMode mode;
} lwdaStreamBeginCapture_ptsz_v10000_params;

typedef struct lwdaThreadExchangeStreamCaptureMode_v10010_params_st {
    enum lwdaStreamCaptureMode *mode;
} lwdaThreadExchangeStreamCaptureMode_v10010_params;

typedef struct lwdaStreamEndCapture_ptsz_v10000_params_st {
    lwdaStream_t stream;
    lwdaGraph_t *pGraph;
} lwdaStreamEndCapture_ptsz_v10000_params;

typedef struct lwdaStreamIsCapturing_ptsz_v10000_params_st {
    lwdaStream_t stream;
    enum lwdaStreamCaptureStatus *pCaptureStatus;
} lwdaStreamIsCapturing_ptsz_v10000_params;

typedef struct lwdaStreamGetCaptureInfo_ptsz_v10010_params_st {
    lwdaStream_t stream;
    enum lwdaStreamCaptureStatus *pCaptureStatus;
    unsigned long long *pId;
} lwdaStreamGetCaptureInfo_ptsz_v10010_params;

typedef struct lwdaEventCreate_v3020_params_st {
    lwdaEvent_t *event;
} lwdaEventCreate_v3020_params;

typedef struct lwdaEventCreateWithFlags_v3020_params_st {
    lwdaEvent_t *event;
    unsigned int flags;
} lwdaEventCreateWithFlags_v3020_params;

typedef struct lwdaEventRecord_ptsz_v7000_params_st {
    lwdaEvent_t event;
    lwdaStream_t stream;
} lwdaEventRecord_ptsz_v7000_params;

typedef struct lwdaEventQuery_v3020_params_st {
    lwdaEvent_t event;
} lwdaEventQuery_v3020_params;

typedef struct lwdaEventSynchronize_v3020_params_st {
    lwdaEvent_t event;
} lwdaEventSynchronize_v3020_params;

typedef struct lwdaEventDestroy_v3020_params_st {
    lwdaEvent_t event;
} lwdaEventDestroy_v3020_params;

typedef struct lwdaEventElapsedTime_v3020_params_st {
    float *ms;
    lwdaEvent_t start;
    lwdaEvent_t end;
} lwdaEventElapsedTime_v3020_params;

typedef struct lwdaImportExternalMemory_v10000_params_st {
    lwdaExternalMemory_t *extMem_out;
    const struct lwdaExternalMemoryHandleDesc *memHandleDesc;
} lwdaImportExternalMemory_v10000_params;

typedef struct lwdaExternalMemoryGetMappedBuffer_v10000_params_st {
    void **devPtr;
    lwdaExternalMemory_t extMem;
    const struct lwdaExternalMemoryBufferDesc *bufferDesc;
} lwdaExternalMemoryGetMappedBuffer_v10000_params;

typedef struct lwdaExternalMemoryGetMappedMipmappedArray_v10000_params_st {
    lwdaMipmappedArray_t *mipmap;
    lwdaExternalMemory_t extMem;
    const struct lwdaExternalMemoryMipmappedArrayDesc *mipmapDesc;
} lwdaExternalMemoryGetMappedMipmappedArray_v10000_params;

typedef struct lwdaDestroyExternalMemory_v10000_params_st {
    lwdaExternalMemory_t extMem;
} lwdaDestroyExternalMemory_v10000_params;

typedef struct lwdaImportExternalSemaphore_v10000_params_st {
    lwdaExternalSemaphore_t *extSem_out;
    const struct lwdaExternalSemaphoreHandleDesc *semHandleDesc;
} lwdaImportExternalSemaphore_v10000_params;

typedef struct lwdaSignalExternalSemaphoresAsync_ptsz_v10000_params_st {
    const lwdaExternalSemaphore_t *extSemArray;
    const struct lwdaExternalSemaphoreSignalParams *paramsArray;
    unsigned int numExtSems;
    lwdaStream_t stream;
} lwdaSignalExternalSemaphoresAsync_ptsz_v10000_params;

typedef struct lwdaWaitExternalSemaphoresAsync_ptsz_v10000_params_st {
    const lwdaExternalSemaphore_t *extSemArray;
    const struct lwdaExternalSemaphoreWaitParams *paramsArray;
    unsigned int numExtSems;
    lwdaStream_t stream;
} lwdaWaitExternalSemaphoresAsync_ptsz_v10000_params;

typedef struct lwdaDestroyExternalSemaphore_v10000_params_st {
    lwdaExternalSemaphore_t extSem;
} lwdaDestroyExternalSemaphore_v10000_params;

typedef struct lwdaLaunchKernel_ptsz_v7000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    lwdaStream_t stream;
} lwdaLaunchKernel_ptsz_v7000_params;

typedef struct lwdaLaunchCooperativeKernel_ptsz_v9000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    lwdaStream_t stream;
} lwdaLaunchCooperativeKernel_ptsz_v9000_params;

typedef struct lwdaLaunchCooperativeKernelMultiDevice_v9000_params_st {
    struct lwdaLaunchParams *launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
} lwdaLaunchCooperativeKernelMultiDevice_v9000_params;

typedef struct lwdaFuncSetCacheConfig_v3020_params_st {
    const void *func;
    enum lwdaFuncCache cacheConfig;
} lwdaFuncSetCacheConfig_v3020_params;

typedef struct lwdaFuncSetSharedMemConfig_v4020_params_st {
    const void *func;
    enum lwdaSharedMemConfig config;
} lwdaFuncSetSharedMemConfig_v4020_params;

typedef struct lwdaFuncGetAttributes_v3020_params_st {
    struct lwdaFuncAttributes *attr;
    const void *func;
} lwdaFuncGetAttributes_v3020_params;

typedef struct lwdaFuncSetAttribute_v9000_params_st {
    const void *func;
    enum lwdaFuncAttribute attr;
    int value;
} lwdaFuncSetAttribute_v9000_params;

typedef struct lwdaLaunchHostFunc_ptsz_v10000_params_st {
    lwdaStream_t stream;
    lwdaHostFn_t fn;
    void *userData;
} lwdaLaunchHostFunc_ptsz_v10000_params;

typedef struct lwdaOclwpancyMaxActiveBlocksPerMultiprocessor_v6050_params_st {
    int *numBlocks;
    const void *func;
    int blockSize;
    size_t dynamicSMemSize;
} lwdaOclwpancyMaxActiveBlocksPerMultiprocessor_v6050_params;

typedef struct lwdaOclwpancyAvailableDynamicSMemPerBlock_v10200_params_st {
    size_t *dynamicSmemSize;
    const void *func;
    int numBlocks;
    int blockSize;
} lwdaOclwpancyAvailableDynamicSMemPerBlock_v10200_params;

typedef struct lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000_params_st {
    int *numBlocks;
    const void *func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
} lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000_params;

typedef struct lwdaMallocManaged_v6000_params_st {
    void **devPtr;
    size_t size;
    unsigned int flags;
} lwdaMallocManaged_v6000_params;

typedef struct lwdaMalloc_v3020_params_st {
    void **devPtr;
    size_t size;
} lwdaMalloc_v3020_params;

typedef struct lwdaMallocHost_v3020_params_st {
    void **ptr;
    size_t size;
} lwdaMallocHost_v3020_params;

typedef struct lwdaMallocPitch_v3020_params_st {
    void **devPtr;
    size_t *pitch;
    size_t width;
    size_t height;
} lwdaMallocPitch_v3020_params;

typedef struct lwdaMallocArray_v3020_params_st {
    lwdaArray_t *array;
    const struct lwdaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    unsigned int flags;
} lwdaMallocArray_v3020_params;

typedef struct lwdaFree_v3020_params_st {
    void *devPtr;
} lwdaFree_v3020_params;

typedef struct lwdaFreeHost_v3020_params_st {
    void *ptr;
} lwdaFreeHost_v3020_params;

typedef struct lwdaFreeArray_v3020_params_st {
    lwdaArray_t array;
} lwdaFreeArray_v3020_params;

typedef struct lwdaFreeMipmappedArray_v5000_params_st {
    lwdaMipmappedArray_t mipmappedArray;
} lwdaFreeMipmappedArray_v5000_params;

typedef struct lwdaHostAlloc_v3020_params_st {
    void **pHost;
    size_t size;
    unsigned int flags;
} lwdaHostAlloc_v3020_params;

typedef struct lwdaHostRegister_v4000_params_st {
    void *ptr;
    size_t size;
    unsigned int flags;
} lwdaHostRegister_v4000_params;

typedef struct lwdaHostUnregister_v4000_params_st {
    void *ptr;
} lwdaHostUnregister_v4000_params;

typedef struct lwdaHostGetDevicePointer_v3020_params_st {
    void **pDevice;
    void *pHost;
    unsigned int flags;
} lwdaHostGetDevicePointer_v3020_params;

typedef struct lwdaHostGetFlags_v3020_params_st {
    unsigned int *pFlags;
    void *pHost;
} lwdaHostGetFlags_v3020_params;

typedef struct lwdaMalloc3D_v3020_params_st {
    struct lwdaPitchedPtr *pitchedDevPtr;
    struct lwdaExtent extent;
} lwdaMalloc3D_v3020_params;

typedef struct lwdaMalloc3DArray_v3020_params_st {
    lwdaArray_t *array;
    const struct lwdaChannelFormatDesc *desc;
    struct lwdaExtent extent;
    unsigned int flags;
} lwdaMalloc3DArray_v3020_params;

typedef struct lwdaMallocMipmappedArray_v5000_params_st {
    lwdaMipmappedArray_t *mipmappedArray;
    const struct lwdaChannelFormatDesc *desc;
    struct lwdaExtent extent;
    unsigned int numLevels;
    unsigned int flags;
} lwdaMallocMipmappedArray_v5000_params;

typedef struct lwdaGetMipmappedArrayLevel_v5000_params_st {
    lwdaArray_t *levelArray;
    lwdaMipmappedArray_const_t mipmappedArray;
    unsigned int level;
} lwdaGetMipmappedArrayLevel_v5000_params;

typedef struct lwdaMemcpy3D_ptds_v7000_params_st {
    const struct lwdaMemcpy3DParms *p;
} lwdaMemcpy3D_ptds_v7000_params;

typedef struct lwdaMemcpy3DPeer_ptds_v7000_params_st {
    const struct lwdaMemcpy3DPeerParms *p;
} lwdaMemcpy3DPeer_ptds_v7000_params;

typedef struct lwdaMemcpy3DAsync_ptsz_v7000_params_st {
    const struct lwdaMemcpy3DParms *p;
    lwdaStream_t stream;
} lwdaMemcpy3DAsync_ptsz_v7000_params;

typedef struct lwdaMemcpy3DPeerAsync_ptsz_v7000_params_st {
    const struct lwdaMemcpy3DPeerParms *p;
    lwdaStream_t stream;
} lwdaMemcpy3DPeerAsync_ptsz_v7000_params;

typedef struct lwdaMemGetInfo_v3020_params_st {
    size_t *free;
    size_t *total;
} lwdaMemGetInfo_v3020_params;

typedef struct lwdaArrayGetInfo_v4010_params_st {
    struct lwdaChannelFormatDesc *desc;
    struct lwdaExtent *extent;
    unsigned int *flags;
    lwdaArray_t array;
} lwdaArrayGetInfo_v4010_params;

typedef struct lwdaMemcpy_ptds_v7000_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy_ptds_v7000_params;

typedef struct lwdaMemcpyPeer_v4000_params_st {
    void *dst;
    int dstDevice;
    const void *src;
    int srcDevice;
    size_t count;
} lwdaMemcpyPeer_v4000_params;

typedef struct lwdaMemcpy2D_ptds_v7000_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2D_ptds_v7000_params;

typedef struct lwdaMemcpy2DToArray_ptds_v7000_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2DToArray_ptds_v7000_params;

typedef struct lwdaMemcpy2DFromArray_ptds_v7000_params_st {
    void *dst;
    size_t dpitch;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2DFromArray_ptds_v7000_params;

typedef struct lwdaMemcpy2DArrayToArray_ptds_v7000_params_st {
    lwdaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    lwdaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2DArrayToArray_ptds_v7000_params;

typedef struct lwdaMemcpyToSymbol_ptds_v7000_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyToSymbol_ptds_v7000_params;

typedef struct lwdaMemcpyFromSymbol_ptds_v7000_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyFromSymbol_ptds_v7000_params;

typedef struct lwdaMemcpyAsync_ptsz_v7000_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyAsync_ptsz_v7000_params;

typedef struct lwdaMemcpyPeerAsync_v4000_params_st {
    void *dst;
    int dstDevice;
    const void *src;
    int srcDevice;
    size_t count;
    lwdaStream_t stream;
} lwdaMemcpyPeerAsync_v4000_params;

typedef struct lwdaMemcpy2DAsync_ptsz_v7000_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpy2DAsync_ptsz_v7000_params;

typedef struct lwdaMemcpy2DToArrayAsync_ptsz_v7000_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpy2DToArrayAsync_ptsz_v7000_params;

typedef struct lwdaMemcpy2DFromArrayAsync_ptsz_v7000_params_st {
    void *dst;
    size_t dpitch;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpy2DFromArrayAsync_ptsz_v7000_params;

typedef struct lwdaMemcpyToSymbolAsync_ptsz_v7000_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyToSymbolAsync_ptsz_v7000_params;

typedef struct lwdaMemcpyFromSymbolAsync_ptsz_v7000_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyFromSymbolAsync_ptsz_v7000_params;

typedef struct lwdaMemset_ptds_v7000_params_st {
    void *devPtr;
    int value;
    size_t count;
} lwdaMemset_ptds_v7000_params;

typedef struct lwdaMemset2D_ptds_v7000_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
} lwdaMemset2D_ptds_v7000_params;

typedef struct lwdaMemset3D_ptds_v7000_params_st {
    struct lwdaPitchedPtr pitchedDevPtr;
    int value;
    struct lwdaExtent extent;
} lwdaMemset3D_ptds_v7000_params;

typedef struct lwdaMemsetAsync_ptsz_v7000_params_st {
    void *devPtr;
    int value;
    size_t count;
    lwdaStream_t stream;
} lwdaMemsetAsync_ptsz_v7000_params;

typedef struct lwdaMemset2DAsync_ptsz_v7000_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    lwdaStream_t stream;
} lwdaMemset2DAsync_ptsz_v7000_params;

typedef struct lwdaMemset3DAsync_ptsz_v7000_params_st {
    struct lwdaPitchedPtr pitchedDevPtr;
    int value;
    struct lwdaExtent extent;
    lwdaStream_t stream;
} lwdaMemset3DAsync_ptsz_v7000_params;

typedef struct lwdaGetSymbolAddress_v3020_params_st {
    void **devPtr;
    const void *symbol;
} lwdaGetSymbolAddress_v3020_params;

typedef struct lwdaGetSymbolSize_v3020_params_st {
    size_t *size;
    const void *symbol;
} lwdaGetSymbolSize_v3020_params;

typedef struct lwdaMemPrefetchAsync_ptsz_v8000_params_st {
    const void *devPtr;
    size_t count;
    int dstDevice;
    lwdaStream_t stream;
} lwdaMemPrefetchAsync_ptsz_v8000_params;

typedef struct lwdaMemAdvise_v8000_params_st {
    const void *devPtr;
    size_t count;
    enum lwdaMemoryAdvise advice;
    int device;
} lwdaMemAdvise_v8000_params;

typedef struct lwdaMemRangeGetAttribute_v8000_params_st {
    void *data;
    size_t dataSize;
    enum lwdaMemRangeAttribute attribute;
    const void *devPtr;
    size_t count;
} lwdaMemRangeGetAttribute_v8000_params;

typedef struct lwdaMemRangeGetAttributes_v8000_params_st {
    void **data;
    size_t *dataSizes;
    enum lwdaMemRangeAttribute *attributes;
    size_t numAttributes;
    const void *devPtr;
    size_t count;
} lwdaMemRangeGetAttributes_v8000_params;

typedef struct lwdaMemcpyToArray_ptds_v7000_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyToArray_ptds_v7000_params;

typedef struct lwdaMemcpyFromArray_ptds_v7000_params_st {
    void *dst;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyFromArray_ptds_v7000_params;

typedef struct lwdaMemcpyArrayToArray_ptds_v7000_params_st {
    lwdaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    lwdaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyArrayToArray_ptds_v7000_params;

typedef struct lwdaMemcpyToArrayAsync_ptsz_v7000_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyToArrayAsync_ptsz_v7000_params;

typedef struct lwdaMemcpyFromArrayAsync_ptsz_v7000_params_st {
    void *dst;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyFromArrayAsync_ptsz_v7000_params;

typedef struct lwdaPointerGetAttributes_v4000_params_st {
    struct lwdaPointerAttributes *attributes;
    const void *ptr;
} lwdaPointerGetAttributes_v4000_params;

typedef struct lwdaDeviceCanAccessPeer_v4000_params_st {
    int *canAccessPeer;
    int device;
    int peerDevice;
} lwdaDeviceCanAccessPeer_v4000_params;

typedef struct lwdaDeviceEnablePeerAccess_v4000_params_st {
    int peerDevice;
    unsigned int flags;
} lwdaDeviceEnablePeerAccess_v4000_params;

typedef struct lwdaDeviceDisablePeerAccess_v4000_params_st {
    int peerDevice;
} lwdaDeviceDisablePeerAccess_v4000_params;

typedef struct lwdaGraphicsUnregisterResource_v3020_params_st {
    lwdaGraphicsResource_t resource;
} lwdaGraphicsUnregisterResource_v3020_params;

typedef struct lwdaGraphicsResourceSetMapFlags_v3020_params_st {
    lwdaGraphicsResource_t resource;
    unsigned int flags;
} lwdaGraphicsResourceSetMapFlags_v3020_params;

typedef struct lwdaGraphicsMapResources_v3020_params_st {
    int count;
    lwdaGraphicsResource_t *resources;
    lwdaStream_t stream;
} lwdaGraphicsMapResources_v3020_params;

typedef struct lwdaGraphicsUnmapResources_v3020_params_st {
    int count;
    lwdaGraphicsResource_t *resources;
    lwdaStream_t stream;
} lwdaGraphicsUnmapResources_v3020_params;

typedef struct lwdaGraphicsResourceGetMappedPointer_v3020_params_st {
    void **devPtr;
    size_t *size;
    lwdaGraphicsResource_t resource;
} lwdaGraphicsResourceGetMappedPointer_v3020_params;

typedef struct lwdaGraphicsSubResourceGetMappedArray_v3020_params_st {
    lwdaArray_t *array;
    lwdaGraphicsResource_t resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
} lwdaGraphicsSubResourceGetMappedArray_v3020_params;

typedef struct lwdaGraphicsResourceGetMappedMipmappedArray_v5000_params_st {
    lwdaMipmappedArray_t *mipmappedArray;
    lwdaGraphicsResource_t resource;
} lwdaGraphicsResourceGetMappedMipmappedArray_v5000_params;

typedef struct lwdaBindTexture_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct lwdaChannelFormatDesc *desc;
    size_t size;
} lwdaBindTexture_v3020_params;

typedef struct lwdaBindTexture2D_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
    const void *devPtr;
    const struct lwdaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    size_t pitch;
} lwdaBindTexture2D_v3020_params;

typedef struct lwdaBindTextureToArray_v3020_params_st {
    const struct textureReference *texref;
    lwdaArray_const_t array;
    const struct lwdaChannelFormatDesc *desc;
} lwdaBindTextureToArray_v3020_params;

typedef struct lwdaBindTextureToMipmappedArray_v5000_params_st {
    const struct textureReference *texref;
    lwdaMipmappedArray_const_t mipmappedArray;
    const struct lwdaChannelFormatDesc *desc;
} lwdaBindTextureToMipmappedArray_v5000_params;

typedef struct lwdaUnbindTexture_v3020_params_st {
    const struct textureReference *texref;
} lwdaUnbindTexture_v3020_params;

typedef struct lwdaGetTextureAlignmentOffset_v3020_params_st {
    size_t *offset;
    const struct textureReference *texref;
} lwdaGetTextureAlignmentOffset_v3020_params;

typedef struct lwdaGetTextureReference_v3020_params_st {
    const struct textureReference **texref;
    const void *symbol;
} lwdaGetTextureReference_v3020_params;

typedef struct lwdaBindSurfaceToArray_v3020_params_st {
    const struct surfaceReference *surfref;
    lwdaArray_const_t array;
    const struct lwdaChannelFormatDesc *desc;
} lwdaBindSurfaceToArray_v3020_params;

typedef struct lwdaGetSurfaceReference_v3020_params_st {
    const struct surfaceReference **surfref;
    const void *symbol;
} lwdaGetSurfaceReference_v3020_params;

typedef struct lwdaGetChannelDesc_v3020_params_st {
    struct lwdaChannelFormatDesc *desc;
    lwdaArray_const_t array;
} lwdaGetChannelDesc_v3020_params;

typedef struct lwdaCreateChannelDesc_v3020_params_st {
    int x;
    int y;
    int z;
    int w;
    enum lwdaChannelFormatKind f;
} lwdaCreateChannelDesc_v3020_params;

typedef struct lwdaCreateTextureObject_v5000_params_st {
    lwdaTextureObject_t *pTexObject;
    const struct lwdaResourceDesc *pResDesc;
    const struct lwdaTextureDesc *pTexDesc;
    const struct lwdaResourceViewDesc *pResViewDesc;
} lwdaCreateTextureObject_v5000_params;

typedef struct lwdaDestroyTextureObject_v5000_params_st {
    lwdaTextureObject_t texObject;
} lwdaDestroyTextureObject_v5000_params;

typedef struct lwdaGetTextureObjectResourceDesc_v5000_params_st {
    struct lwdaResourceDesc *pResDesc;
    lwdaTextureObject_t texObject;
} lwdaGetTextureObjectResourceDesc_v5000_params;

typedef struct lwdaGetTextureObjectTextureDesc_v5000_params_st {
    struct lwdaTextureDesc *pTexDesc;
    lwdaTextureObject_t texObject;
} lwdaGetTextureObjectTextureDesc_v5000_params;

typedef struct lwdaGetTextureObjectResourceViewDesc_v5000_params_st {
    struct lwdaResourceViewDesc *pResViewDesc;
    lwdaTextureObject_t texObject;
} lwdaGetTextureObjectResourceViewDesc_v5000_params;

typedef struct lwdaCreateSurfaceObject_v5000_params_st {
    lwdaSurfaceObject_t *pSurfObject;
    const struct lwdaResourceDesc *pResDesc;
} lwdaCreateSurfaceObject_v5000_params;

typedef struct lwdaDestroySurfaceObject_v5000_params_st {
    lwdaSurfaceObject_t surfObject;
} lwdaDestroySurfaceObject_v5000_params;

typedef struct lwdaGetSurfaceObjectResourceDesc_v5000_params_st {
    struct lwdaResourceDesc *pResDesc;
    lwdaSurfaceObject_t surfObject;
} lwdaGetSurfaceObjectResourceDesc_v5000_params;

typedef struct lwdaDriverGetVersion_v3020_params_st {
    int *driverVersion;
} lwdaDriverGetVersion_v3020_params;

typedef struct lwdaRuntimeGetVersion_v3020_params_st {
    int *runtimeVersion;
} lwdaRuntimeGetVersion_v3020_params;

typedef struct lwdaGraphCreate_v10000_params_st {
    lwdaGraph_t *pGraph;
    unsigned int flags;
} lwdaGraphCreate_v10000_params;

typedef struct lwdaGraphAddKernelNode_v10000_params_st {
    lwdaGraphNode_t *pGraphNode;
    lwdaGraph_t graph;
    const lwdaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct lwdaKernelNodeParams *pNodeParams;
} lwdaGraphAddKernelNode_v10000_params;

typedef struct lwdaGraphKernelNodeGetParams_v10000_params_st {
    lwdaGraphNode_t node;
    struct lwdaKernelNodeParams *pNodeParams;
} lwdaGraphKernelNodeGetParams_v10000_params;

typedef struct lwdaGraphKernelNodeSetParams_v10000_params_st {
    lwdaGraphNode_t node;
    const struct lwdaKernelNodeParams *pNodeParams;
} lwdaGraphKernelNodeSetParams_v10000_params;

typedef struct lwdaGraphKernelNodeCopyAttributes_v11000_params_st {
    lwdaGraphNode_t hSrc;
    lwdaGraphNode_t hDst;
} lwdaGraphKernelNodeCopyAttributes_v11000_params;

typedef struct lwdaGraphKernelNodeGetAttribute_v11000_params_st {
    lwdaGraphNode_t hNode;
    enum lwdaKernelNodeAttrID attr;
    union lwdaKernelNodeAttrValue *value_out;
} lwdaGraphKernelNodeGetAttribute_v11000_params;

typedef struct lwdaGraphKernelNodeSetAttribute_v11000_params_st {
    lwdaGraphNode_t hNode;
    enum lwdaKernelNodeAttrID attr;
    const union lwdaKernelNodeAttrValue *value;
} lwdaGraphKernelNodeSetAttribute_v11000_params;

typedef struct lwdaGraphAddMemcpyNode_v10000_params_st {
    lwdaGraphNode_t *pGraphNode;
    lwdaGraph_t graph;
    const lwdaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct lwdaMemcpy3DParms *pCopyParams;
} lwdaGraphAddMemcpyNode_v10000_params;

typedef struct lwdaGraphMemcpyNodeGetParams_v10000_params_st {
    lwdaGraphNode_t node;
    struct lwdaMemcpy3DParms *pNodeParams;
} lwdaGraphMemcpyNodeGetParams_v10000_params;

typedef struct lwdaGraphMemcpyNodeSetParams_v10000_params_st {
    lwdaGraphNode_t node;
    const struct lwdaMemcpy3DParms *pNodeParams;
} lwdaGraphMemcpyNodeSetParams_v10000_params;

typedef struct lwdaGraphAddMemsetNode_v10000_params_st {
    lwdaGraphNode_t *pGraphNode;
    lwdaGraph_t graph;
    const lwdaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct lwdaMemsetParams *pMemsetParams;
} lwdaGraphAddMemsetNode_v10000_params;

typedef struct lwdaGraphMemsetNodeGetParams_v10000_params_st {
    lwdaGraphNode_t node;
    struct lwdaMemsetParams *pNodeParams;
} lwdaGraphMemsetNodeGetParams_v10000_params;

typedef struct lwdaGraphMemsetNodeSetParams_v10000_params_st {
    lwdaGraphNode_t node;
    const struct lwdaMemsetParams *pNodeParams;
} lwdaGraphMemsetNodeSetParams_v10000_params;

typedef struct lwdaGraphAddHostNode_v10000_params_st {
    lwdaGraphNode_t *pGraphNode;
    lwdaGraph_t graph;
    const lwdaGraphNode_t *pDependencies;
    size_t numDependencies;
    const struct lwdaHostNodeParams *pNodeParams;
} lwdaGraphAddHostNode_v10000_params;

typedef struct lwdaGraphHostNodeGetParams_v10000_params_st {
    lwdaGraphNode_t node;
    struct lwdaHostNodeParams *pNodeParams;
} lwdaGraphHostNodeGetParams_v10000_params;

typedef struct lwdaGraphHostNodeSetParams_v10000_params_st {
    lwdaGraphNode_t node;
    const struct lwdaHostNodeParams *pNodeParams;
} lwdaGraphHostNodeSetParams_v10000_params;

typedef struct lwdaGraphAddChildGraphNode_v10000_params_st {
    lwdaGraphNode_t *pGraphNode;
    lwdaGraph_t graph;
    const lwdaGraphNode_t *pDependencies;
    size_t numDependencies;
    lwdaGraph_t childGraph;
} lwdaGraphAddChildGraphNode_v10000_params;

typedef struct lwdaGraphChildGraphNodeGetGraph_v10000_params_st {
    lwdaGraphNode_t node;
    lwdaGraph_t *pGraph;
} lwdaGraphChildGraphNodeGetGraph_v10000_params;

typedef struct lwdaGraphAddEmptyNode_v10000_params_st {
    lwdaGraphNode_t *pGraphNode;
    lwdaGraph_t graph;
    const lwdaGraphNode_t *pDependencies;
    size_t numDependencies;
} lwdaGraphAddEmptyNode_v10000_params;

typedef struct lwdaGraphClone_v10000_params_st {
    lwdaGraph_t *pGraphClone;
    lwdaGraph_t originalGraph;
} lwdaGraphClone_v10000_params;

typedef struct lwdaGraphNodeFindInClone_v10000_params_st {
    lwdaGraphNode_t *pNode;
    lwdaGraphNode_t originalNode;
    lwdaGraph_t clonedGraph;
} lwdaGraphNodeFindInClone_v10000_params;

typedef struct lwdaGraphNodeGetType_v10000_params_st {
    lwdaGraphNode_t node;
    enum lwdaGraphNodeType *pType;
} lwdaGraphNodeGetType_v10000_params;

typedef struct lwdaGraphGetNodes_v10000_params_st {
    lwdaGraph_t graph;
    lwdaGraphNode_t *nodes;
    size_t *numNodes;
} lwdaGraphGetNodes_v10000_params;

typedef struct lwdaGraphGetRootNodes_v10000_params_st {
    lwdaGraph_t graph;
    lwdaGraphNode_t *pRootNodes;
    size_t *pNumRootNodes;
} lwdaGraphGetRootNodes_v10000_params;

typedef struct lwdaGraphGetEdges_v10000_params_st {
    lwdaGraph_t graph;
    lwdaGraphNode_t *from;
    lwdaGraphNode_t *to;
    size_t *numEdges;
} lwdaGraphGetEdges_v10000_params;

typedef struct lwdaGraphNodeGetDependencies_v10000_params_st {
    lwdaGraphNode_t node;
    lwdaGraphNode_t *pDependencies;
    size_t *pNumDependencies;
} lwdaGraphNodeGetDependencies_v10000_params;

typedef struct lwdaGraphNodeGetDependentNodes_v10000_params_st {
    lwdaGraphNode_t node;
    lwdaGraphNode_t *pDependentNodes;
    size_t *pNumDependentNodes;
} lwdaGraphNodeGetDependentNodes_v10000_params;

typedef struct lwdaGraphAddDependencies_v10000_params_st {
    lwdaGraph_t graph;
    const lwdaGraphNode_t *from;
    const lwdaGraphNode_t *to;
    size_t numDependencies;
} lwdaGraphAddDependencies_v10000_params;

typedef struct lwdaGraphRemoveDependencies_v10000_params_st {
    lwdaGraph_t graph;
    const lwdaGraphNode_t *from;
    const lwdaGraphNode_t *to;
    size_t numDependencies;
} lwdaGraphRemoveDependencies_v10000_params;

typedef struct lwdaGraphDestroyNode_v10000_params_st {
    lwdaGraphNode_t node;
} lwdaGraphDestroyNode_v10000_params;

typedef struct lwdaGraphInstantiate_v10000_params_st {
    lwdaGraphExec_t *pGraphExec;
    lwdaGraph_t graph;
    lwdaGraphNode_t *pErrorNode;
    char *pLogBuffer;
    size_t bufferSize;
} lwdaGraphInstantiate_v10000_params;

typedef struct lwdaGraphExecKernelNodeSetParams_v10010_params_st {
    lwdaGraphExec_t hGraphExec;
    lwdaGraphNode_t node;
    const struct lwdaKernelNodeParams *pNodeParams;
} lwdaGraphExecKernelNodeSetParams_v10010_params;

typedef struct lwdaGraphExecMemcpyNodeSetParams_v10020_params_st {
    lwdaGraphExec_t hGraphExec;
    lwdaGraphNode_t node;
    const struct lwdaMemcpy3DParms *pNodeParams;
} lwdaGraphExecMemcpyNodeSetParams_v10020_params;

typedef struct lwdaGraphExecMemsetNodeSetParams_v10020_params_st {
    lwdaGraphExec_t hGraphExec;
    lwdaGraphNode_t node;
    const struct lwdaMemsetParams *pNodeParams;
} lwdaGraphExecMemsetNodeSetParams_v10020_params;

typedef struct lwdaGraphExecHostNodeSetParams_v10020_params_st {
    lwdaGraphExec_t hGraphExec;
    lwdaGraphNode_t node;
    const struct lwdaHostNodeParams *pNodeParams;
} lwdaGraphExecHostNodeSetParams_v10020_params;

typedef struct lwdaGraphExelwpdate_v10020_params_st {
    lwdaGraphExec_t hGraphExec;
    lwdaGraph_t hGraph;
    lwdaGraphNode_t *hErrorNode_out;
    enum lwdaGraphExelwpdateResult *updateResult_out;
} lwdaGraphExelwpdate_v10020_params;

typedef struct lwdaGraphUpload_ptsz_v10000_params_st {
    lwdaGraphExec_t graphExec;
    lwdaStream_t stream;
} lwdaGraphUpload_ptsz_v10000_params;

typedef struct lwdaGraphLaunch_ptsz_v10000_params_st {
    lwdaGraphExec_t graphExec;
    lwdaStream_t stream;
} lwdaGraphLaunch_ptsz_v10000_params;

typedef struct lwdaGraphExecDestroy_v10000_params_st {
    lwdaGraphExec_t graphExec;
} lwdaGraphExecDestroy_v10000_params;

typedef struct lwdaGraphDestroy_v10000_params_st {
    lwdaGraph_t graph;
} lwdaGraphDestroy_v10000_params;

typedef struct lwdaGetFuncBySymbol_v11000_params_st {
    lwdaFunction_t *functionPtr;
    const void *symbolPtr;
} lwdaGetFuncBySymbol_v11000_params;

typedef struct lwdaMemcpy_v3020_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy_v3020_params;

typedef struct lwdaMemcpyToSymbol_v3020_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyToSymbol_v3020_params;

typedef struct lwdaMemcpyFromSymbol_v3020_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyFromSymbol_v3020_params;

typedef struct lwdaMemcpy2D_v3020_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2D_v3020_params;

typedef struct lwdaMemcpyToArray_v3020_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyToArray_v3020_params;

typedef struct lwdaMemcpy2DToArray_v3020_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2DToArray_v3020_params;

typedef struct lwdaMemcpyFromArray_v3020_params_st {
    void *dst;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyFromArray_v3020_params;

typedef struct lwdaMemcpy2DFromArray_v3020_params_st {
    void *dst;
    size_t dpitch;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2DFromArray_v3020_params;

typedef struct lwdaMemcpyArrayToArray_v3020_params_st {
    lwdaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    lwdaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum lwdaMemcpyKind kind;
} lwdaMemcpyArrayToArray_v3020_params;

typedef struct lwdaMemcpy2DArrayToArray_v3020_params_st {
    lwdaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    lwdaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
} lwdaMemcpy2DArrayToArray_v3020_params;

typedef struct lwdaMemcpy3D_v3020_params_st {
    const struct lwdaMemcpy3DParms *p;
} lwdaMemcpy3D_v3020_params;

typedef struct lwdaMemcpy3DPeer_v4000_params_st {
    const struct lwdaMemcpy3DPeerParms *p;
} lwdaMemcpy3DPeer_v4000_params;

typedef struct lwdaMemset_v3020_params_st {
    void *devPtr;
    int value;
    size_t count;
} lwdaMemset_v3020_params;

typedef struct lwdaMemset2D_v3020_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
} lwdaMemset2D_v3020_params;

typedef struct lwdaMemset3D_v3020_params_st {
    struct lwdaPitchedPtr pitchedDevPtr;
    int value;
    struct lwdaExtent extent;
} lwdaMemset3D_v3020_params;

typedef struct lwdaMemcpyAsync_v3020_params_st {
    void *dst;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyAsync_v3020_params;

typedef struct lwdaMemcpyToSymbolAsync_v3020_params_st {
    const void *symbol;
    const void *src;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyToSymbolAsync_v3020_params;

typedef struct lwdaMemcpyFromSymbolAsync_v3020_params_st {
    void *dst;
    const void *symbol;
    size_t count;
    size_t offset;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyFromSymbolAsync_v3020_params;

typedef struct lwdaMemcpy2DAsync_v3020_params_st {
    void *dst;
    size_t dpitch;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpy2DAsync_v3020_params;

typedef struct lwdaMemcpyToArrayAsync_v3020_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t count;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyToArrayAsync_v3020_params;

typedef struct lwdaMemcpy2DToArrayAsync_v3020_params_st {
    lwdaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void *src;
    size_t spitch;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpy2DToArrayAsync_v3020_params;

typedef struct lwdaMemcpyFromArrayAsync_v3020_params_st {
    void *dst;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpyFromArrayAsync_v3020_params;

typedef struct lwdaMemcpy2DFromArrayAsync_v3020_params_st {
    void *dst;
    size_t dpitch;
    lwdaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum lwdaMemcpyKind kind;
    lwdaStream_t stream;
} lwdaMemcpy2DFromArrayAsync_v3020_params;

typedef struct lwdaMemcpy3DAsync_v3020_params_st {
    const struct lwdaMemcpy3DParms *p;
    lwdaStream_t stream;
} lwdaMemcpy3DAsync_v3020_params;

typedef struct lwdaMemcpy3DPeerAsync_v4000_params_st {
    const struct lwdaMemcpy3DPeerParms *p;
    lwdaStream_t stream;
} lwdaMemcpy3DPeerAsync_v4000_params;

typedef struct lwdaMemsetAsync_v3020_params_st {
    void *devPtr;
    int value;
    size_t count;
    lwdaStream_t stream;
} lwdaMemsetAsync_v3020_params;

typedef struct lwdaMemset2DAsync_v3020_params_st {
    void *devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    lwdaStream_t stream;
} lwdaMemset2DAsync_v3020_params;

typedef struct lwdaMemset3DAsync_v3020_params_st {
    struct lwdaPitchedPtr pitchedDevPtr;
    int value;
    struct lwdaExtent extent;
    lwdaStream_t stream;
} lwdaMemset3DAsync_v3020_params;

typedef struct lwdaStreamQuery_v3020_params_st {
    lwdaStream_t stream;
} lwdaStreamQuery_v3020_params;

typedef struct lwdaStreamGetFlags_v5050_params_st {
    lwdaStream_t hStream;
    unsigned int *flags;
} lwdaStreamGetFlags_v5050_params;

typedef struct lwdaStreamGetPriority_v5050_params_st {
    lwdaStream_t hStream;
    int *priority;
} lwdaStreamGetPriority_v5050_params;

typedef struct lwdaEventRecord_v3020_params_st {
    lwdaEvent_t event;
    lwdaStream_t stream;
} lwdaEventRecord_v3020_params;

typedef struct lwdaStreamWaitEvent_v3020_params_st {
    lwdaStream_t stream;
    lwdaEvent_t event;
    unsigned int flags;
} lwdaStreamWaitEvent_v3020_params;

typedef struct lwdaStreamAddCallback_v5000_params_st {
    lwdaStream_t stream;
    lwdaStreamCallback_t callback;
    void *userData;
    unsigned int flags;
} lwdaStreamAddCallback_v5000_params;

typedef struct lwdaStreamAttachMemAsync_v6000_params_st {
    lwdaStream_t stream;
    void *devPtr;
    size_t length;
    unsigned int flags;
} lwdaStreamAttachMemAsync_v6000_params;

typedef struct lwdaStreamSynchronize_v3020_params_st {
    lwdaStream_t stream;
} lwdaStreamSynchronize_v3020_params;

typedef struct lwdaLaunchKernel_v7000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    lwdaStream_t stream;
} lwdaLaunchKernel_v7000_params;

typedef struct lwdaLaunchCooperativeKernel_v9000_params_st {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    lwdaStream_t stream;
} lwdaLaunchCooperativeKernel_v9000_params;

typedef struct lwdaLaunchHostFunc_v10000_params_st {
    lwdaStream_t stream;
    lwdaHostFn_t fn;
    void *userData;
} lwdaLaunchHostFunc_v10000_params;

typedef struct lwdaMemPrefetchAsync_v8000_params_st {
    const void *devPtr;
    size_t count;
    int dstDevice;
    lwdaStream_t stream;
} lwdaMemPrefetchAsync_v8000_params;

typedef struct lwdaSignalExternalSemaphoresAsync_v10000_params_st {
    const lwdaExternalSemaphore_t *extSemArray;
    const struct lwdaExternalSemaphoreSignalParams *paramsArray;
    unsigned int numExtSems;
    lwdaStream_t stream;
} lwdaSignalExternalSemaphoresAsync_v10000_params;

typedef struct lwdaWaitExternalSemaphoresAsync_v10000_params_st {
    const lwdaExternalSemaphore_t *extSemArray;
    const struct lwdaExternalSemaphoreWaitParams *paramsArray;
    unsigned int numExtSems;
    lwdaStream_t stream;
} lwdaWaitExternalSemaphoresAsync_v10000_params;

typedef struct lwdaGraphUpload_v10000_params_st {
    lwdaGraphExec_t graphExec;
    lwdaStream_t stream;
} lwdaGraphUpload_v10000_params;

typedef struct lwdaGraphLaunch_v10000_params_st {
    lwdaGraphExec_t graphExec;
    lwdaStream_t stream;
} lwdaGraphLaunch_v10000_params;

typedef struct lwdaStreamBeginCapture_v10000_params_st {
    lwdaStream_t stream;
    enum lwdaStreamCaptureMode mode;
} lwdaStreamBeginCapture_v10000_params;

typedef struct lwdaStreamEndCapture_v10000_params_st {
    lwdaStream_t stream;
    lwdaGraph_t *pGraph;
} lwdaStreamEndCapture_v10000_params;

typedef struct lwdaStreamIsCapturing_v10000_params_st {
    lwdaStream_t stream;
    enum lwdaStreamCaptureStatus *pCaptureStatus;
} lwdaStreamIsCapturing_v10000_params;

typedef struct lwdaStreamGetCaptureInfo_v10010_params_st {
    lwdaStream_t hStream;
    enum lwdaStreamCaptureStatus *pCaptureStatus;
    unsigned long long *pId;
} lwdaStreamGetCaptureInfo_v10010_params;

typedef struct lwdaStreamCopyAttributes_v11000_params_st {
    lwdaStream_t dstStream;
    lwdaStream_t srcStream;
} lwdaStreamCopyAttributes_v11000_params;

typedef struct lwdaStreamGetAttribute_v11000_params_st {
    lwdaStream_t stream;
    enum lwdaStreamAttrID attr;
    union lwdaStreamAttrValue *value;
} lwdaStreamGetAttribute_v11000_params;

typedef struct lwdaStreamSetAttribute_v11000_params_st {
    lwdaStream_t stream;
    enum lwdaStreamAttrID attr;
    const union lwdaStreamAttrValue *param;
} lwdaStreamSetAttribute_v11000_params;

// Parameter trace structures for removed functions 


// End of parameter trace structures
