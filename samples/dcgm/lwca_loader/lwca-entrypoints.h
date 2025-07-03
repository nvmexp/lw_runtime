/////////////////////////////////////////////////////////////////////////////////////////
// Re-usable entrypoints
// ------------------------------------------------------------------------------------
// This file contains the entry points for the LWCA library. The macros are defined
// externally. This file may be included multiple times each time with a different 
// definition for the entrypoint macros.

BEGIN_ENTRYPOINTS

LWDA_API_ENTRYPOINT(lwMemAllocHost_v2, lwMemAllocHost_v2,
    (void** pp, size_t bytesize),
    "(%p, %zd)",
    pp, bytesize)

LWDA_API_ENTRYPOINT(lwMemAlloc_v2, lwMemAlloc_v2,
    (LWdeviceptr *dptr, size_t bytesize),
    "(%p, %zd)",
    dptr, bytesize)

LWDA_API_ENTRYPOINT(lwMemFreeHost, lwMemFreeHost,
    (void *p),
    "(%p)",
    p)

LWDA_API_ENTRYPOINT(lwMemFree, lwMemFree,
    (LWdeviceptr_v1 dptr),
    "(%p)",
    dptr)

LWDA_API_ENTRYPOINT(lwMemFree_v2, lwMemFree_v2,
    (LWdeviceptr dptr),
    "(%p)",
    dptr)

LWDA_API_ENTRYPOINT(lwMemcpyHtoDAsync_v2, lwMemcpyHtoDAsync_v2, 
    (LWdeviceptr dstDevice, const void *srcHost, size_t ByteCount, LWstream hStream),
    "(%p, %p, %zd, %p)",
    dstDevice, srcHost, ByteCount, hStream)

LWDA_API_ENTRYPOINT(lwMemcpyDtoHAsync_v2, lwMemcpyDtoHAsync_v2,
    (void *dstHost, LWdeviceptr srcDevice, size_t ByteCount, LWstream hStream),
    "(%p, %p, %zd, %p)",
    dstHost, srcDevice, ByteCount, hStream)

LWDA_API_ENTRYPOINT(lwMemcpyDtoH_v2, lwMemcpyDtoH_v2, 
    (void *dstHost, LWdeviceptr srcDevice, size_t ByteCount),
    "(%p, %p, %zd)",
    dstHost, srcDevice, ByteCount)

LWDA_API_ENTRYPOINT(lwMemcpyHtoD_v2, lwMemcpyHtoD_v2,
    (LWdeviceptr dstDevice, const void *srcHost, size_t ByteCount),
    "(%p, %p, %zd)",
    dstDevice, srcHost, ByteCount)

LWDA_API_ENTRYPOINT(lwMemcpyDtoD_v2, lwMemcpyDtoD_v2,
    (LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t ByteCount),
    "(%p, %p, %zd)",
    dstDevice, srcDevice, ByteCount)

LWDA_API_ENTRYPOINT(lwMemGetInfo_v2, lwMemGetInfo_v2,
    (size_t *free, size_t *total),
    "(%zd, %zd)",
    free, total)

LWDA_API_ENTRYPOINT(lwCtxCreate, lwCtxCreate,
    (LWcontext *pctx, unsigned int flags, LWdevice dev),
    "(%p, %u, %p)",
    pctx, flags, dev)

LWDA_API_ENTRYPOINT(lwCtxCreate_v2, lwCtxCreate_v2,
    (LWcontext *pctx, unsigned int flags, LWdevice dev),
    "(%p, %u, %p)",
    pctx, flags, dev)

LWDA_API_ENTRYPOINT(lwCtxSynchronize, lwCtxSynchronize, (void),
    "()")

LWDA_API_ENTRYPOINT(lwStreamCreate, lwStreamCreate,
    (LWstream *phStream, unsigned int Flags),
    "(%p, %u)",
    phStream, Flags)

LWDA_API_ENTRYPOINT(lwStreamDestroy, lwStreamDestroy,
    (LWstream hStream),
    "(%p)",
    hStream)

LWDA_API_ENTRYPOINT(lwStreamDestroy_v2, lwStreamDestroy_v2,
    (LWstream hStream),
    "(%p)",
    hStream)

LWDA_API_ENTRYPOINT(lwEventCreate, lwEventCreate,
    (LWevent *phEvent, unsigned int Flags),
    "(%p, %u)",
    phEvent, Flags)

LWDA_API_ENTRYPOINT(lwEventRecord, lwEventRecord,
    (LWevent hEvent, LWstream hStream),
    "(%p, %p)",
    hEvent, hStream)

LWDA_API_ENTRYPOINT(lwEventSynchronize, lwEventSynchronize,
    (LWevent hEvent),
    "(%p)",
    hEvent)

LWDA_API_ENTRYPOINT(lwEventElapsedTime, lwEventElapsedTime, 
    (float *pMilliseconds, LWevent hStart, LWevent hEnd),
    "(%f, %p, %p)",
    pMilliseconds, hStart, hEnd)

LWDA_API_ENTRYPOINT(lwEventDestroy, lwEventDestroy, 
    (LWevent hEvent),
    "(%p)",
    hEvent)

LWDA_API_ENTRYPOINT(lwEventDestroy_v2, lwEventDestroy_v2, 
    (LWevent hEvent),
    "(%p)",
    hEvent)

LWDA_API_ENTRYPOINT(lwDeviceGetAttribute, lwDeviceGetAttribute,
    (int *pi, LWdevice_attribute attrib, LWdevice dev),
    "(%p, %p, %p)",
    pi, attrib, dev)

LWDA_API_ENTRYPOINT(lwModuleLoad, lwModuleLoad, 
    (LWmodule *module, const char  *fname),
    "(%p, %p)",
    module, fname)

LWDA_API_ENTRYPOINT(lwModuleLoadData, lwModuleLoadData, 
    (LWmodule *module, const void *image),
    "(%p, %p)",
    module, image)

LWDA_API_ENTRYPOINT(lwModuleGetGlobal_v2, lwModuleGetGlobal_v2,
    (LWdeviceptr *dptr, size_t *bytes, LWmodule hmod, const char *name),
    "(%p, %p, %p, %p)",
    dptr, bytes, hmod, name)

LWDA_API_ENTRYPOINT(lwModuleGetFunction, lwModuleGetFunction,
    (LWfunction *hfunc, LWmodule hmod, const char *name),
    "(%p, %p, %p)",
    hfunc, hmod, name)

LWDA_API_ENTRYPOINT(lwModuleUnload, lwModuleUnload, 
    (LWmodule hmod),
    "(%p)",
    hmod)

LWDA_API_ENTRYPOINT(lwFuncSetBlockShape, lwFuncSetBlockShape,
    (LWfunction hfunc, int x, int y, int z),
    "(%p, %d, %d, %d)",
    hfunc, x, y, z)

LWDA_API_ENTRYPOINT(lwParamSetv, lwParamSetv,
    (LWfunction hfunc, int offset, void *ptr, unsigned int numbytes),
    "(%p, %d, %p, %u)",
    hfunc, offset, ptr, numbytes)

LWDA_API_ENTRYPOINT(lwParamSetSize, lwParamSetSize,
    (LWfunction hfunc, unsigned int numbytes),
    "(%p, %u)",
    hfunc, numbytes)

LWDA_API_ENTRYPOINT(lwLaunchGridAsync, lwLaunchGridAsync, 
    (LWfunction f, int grid_width, int grid_height, LWstream hStream),
    "(%p, %d, %d, %p)",
    f, grid_width, grid_height, hStream)

LWDA_API_ENTRYPOINT(lwLaunchKernel, lwLaunchKernel,
    (LWfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
     unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
     unsigned int  sharedMemBytes, LWstream hStream, void** kernelParams, void** extra ),
     "(%p %u %u %u, %u %u %u %u %p %p %p)",
     f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 
     sharedMemBytes, hStream, kernelParams, extra)

LWDA_API_ENTRYPOINT(lwDeviceGetName, lwDeviceGetName,
    (char *name, int len, LWdevice dev),
    "(%p, %d, %p)",
    name, len, dev)

LWDA_API_ENTRYPOINT(lwDeviceComputeCapability, lwDeviceComputeCapability, 
    (int *major, int *minor, LWdevice dev),
    "(%p, %p, %p)",
    major, minor, dev)

LWDA_API_ENTRYPOINT(lwDeviceTotalMem_v2, lwDeviceTotalMem_v2, 
    (size_t *bytes, LWdevice dev),
    "(%p, %p)",
    bytes, dev)

LWDA_API_ENTRYPOINT(lwCtxDestroy, lwCtxDestroy,
    (LWcontext ctx),
    "(%p)",
    ctx)

LWDA_API_ENTRYPOINT(lwCtxDestroy_v2, lwCtxDestroy_v2,
    (LWcontext ctx),
    "(%p)",
    ctx)

LWDA_API_ENTRYPOINT(lwDevicePrimaryCtxReset, lwDevicePrimaryCtxReset,
    (LWdevice dev),
    "(%p)",
    dev)

LWDA_API_ENTRYPOINT(lwGetExportTable, lwGetExportTable,
    (const void **ppExportTable, const LWuuid *pExportTableId),
    "(%p, %p)",
    ppExportTable, pExportTableId)
    
LWDA_API_ENTRYPOINT(lwDeviceGetByPCIBusId, lwDeviceGetByPCIBusId,
    (LWdevice *dev, const char *pciBusId),
    "(%p, %p)",
    dev, pciBusId)

LWDA_API_ENTRYPOINT(lwDeviceGetCount, lwDeviceGetCount, 
    (int *count),
    "(%p)",
    count)

LWDA_API_ENTRYPOINT(lwInit, lwInit,
    (unsigned int Flags),
    "(%u)",
    Flags)

LWDA_API_ENTRYPOINT(lwDeviceCanAccessPeer, lwDeviceCanAccessPeer, 
    (int *canAccessPeer, LWdevice dev, LWdevice peerDev),
    "(%p, %p, %p)",
    canAccessPeer, dev, peerDev)

LWDA_API_ENTRYPOINT(lwCtxSetLwrrent, lwCtxSetLwrrent, 
    (LWcontext ctx),
    "(%p)",
    ctx)

LWDA_API_ENTRYPOINT(lwMemcpy, lwMemcpy, 
    (LWdeviceptr dst, LWdeviceptr src, size_t ByteCount),
    "(%p, %p, %zd)",
    dst, src, ByteCount)

LWDA_API_ENTRYPOINT(lwMemcpyAsync, lwMemcpyAsync, 
    (LWdeviceptr dst, LWdeviceptr src, size_t ByteCount, LWstream hStream),
    "(%p, %p, %zd, %p)",
    dst, src, ByteCount, hStream)

LWDA_API_ENTRYPOINT(lwMemcpyPeerAsync, lwMemcpyPeerAsync, 
    (LWdeviceptr dst, LWcontext dstContext, LWdeviceptr src, LWcontext srcContext, size_t ByteCount, LWstream hStream),
    "(%p, %p, %p, %p, %zd, %p)",
    dst, dstContext, src, srcContext, ByteCount, hStream)

LWDA_API_ENTRYPOINT(lwCtxEnablePeerAccess, lwCtxEnablePeerAccess, 
    (LWcontext ctx, unsigned int Flags),
    "(%p, %u)",
    ctx, Flags)

LWDA_API_ENTRYPOINT(lwStreamWaitEvent, lwStreamWaitEvent,
    (LWstream hStream, LWevent hEvent, unsigned int Flags),
    "(%p, %p, %u)",
    hStream, hEvent, Flags)

LWDA_API_ENTRYPOINT(lwStreamSynchronize, lwStreamSynchronize,
    (LWstream hStream),
    "(%p)",
    hStream)

LWDA_API_ENTRYPOINT(lwMemHostRegister, lwMemHostRegister, 
    (void *p, size_t byteSize, unsigned int Flags),
    "(%p, %zd, %u)",
    p, byteSize, Flags)

LWDA_API_ENTRYPOINT(lwMemHostUnregister, lwMemHostUnregister, 
    (void *p),
    "(%p)",
    p)
    

LWDA_API_ENTRYPOINT(lwPointerGetAttribute, lwPointerGetAttribute,
    (void *data, LWpointer_attribute attribute, LWdeviceptr ptr),
    "(%p, %d, %p)",
    data, attribute, ptr)

LWDA_API_ENTRYPOINT(lwMemsetD32_v2, lwMemsetD32_v2,
    (LWdeviceptr data, unsigned int ui, size_t N),
    "(%p, %u, %zd)",
    data, ui, N)

LWDA_API_ENTRYPOINT(lwLaunchGrid, lwLaunchGrid,
    (LWfunction f, int width, int height),
    "(%p, %d, %d)",
    f, width, height)

LWDA_API_ENTRYPOINT(lwFuncSetCacheConfig, lwFuncSetCacheConfig,
    (LWfunction f, LWfunc_cache config),
    "(%p, %d)",
    f, config)

LWDA_API_ENTRYPOINT(lwGetErrorString, lwGetErrorString,
    (LWresult lwSt, const char **errorString),
    "(%d, %p)",
    lwSt, errorString)

LWDA_API_ENTRYPOINT(lwCtxGetLwrrent, lwCtxGetLwrrent,
    (LWcontext *pctx),
    "(%p)",
    pctx)

LWDA_API_ENTRYPOINT(lwCtxSetLimit, lwCtxSetLimit,
    (LWlimit limit, size_t value),
    "(%d, %zu)",
    limit, value)

LWDA_API_ENTRYPOINT(lwCtxGetLimit, lwCtxGetLimit,
    (size_t *pvalue, LWlimit limit),
    ("%p, %d"),
    pvalue, limit)

END_ENTRYPOINTS
