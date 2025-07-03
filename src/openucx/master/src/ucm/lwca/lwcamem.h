/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_LWDAMEM_H_
#define UCM_LWDAMEM_H_

#include <ucm/api/ucm.h>
#include <lwda_runtime.h>
#include <lwca.h>


/*lwMemFree */
LWresult ucm_override_lwMemFree(LWdeviceptr dptr);
LWresult ucm_orig_lwMemFree(LWdeviceptr dptr);
LWresult ucm_lwMemFree(LWdeviceptr dptr);

/*lwMemFreeHost */
LWresult ucm_override_lwMemFreeHost(void *p);
LWresult ucm_orig_lwMemFreeHost(void *p);
LWresult ucm_lwMemFreeHost(void *p);

/*lwMemAlloc*/
LWresult ucm_override_lwMemAlloc(LWdeviceptr *dptr, size_t size);
LWresult ucm_orig_lwMemAlloc(LWdeviceptr *dptr, size_t size);
LWresult ucm_lwMemAlloc(LWdeviceptr *dptr, size_t size);

/*lwMemAllocManaged*/
LWresult ucm_override_lwMemAllocManaged(LWdeviceptr *dptr, size_t size,
                                        unsigned int flags);
LWresult ucm_orig_lwMemAllocManaged(LWdeviceptr *dptr, size_t size, unsigned int flags);
LWresult ucm_lwMemAllocManaged(LWdeviceptr *dptr, size_t size, unsigned int flags);

/*lwMemAllocPitch*/
LWresult ucm_override_lwMemAllocPitch(LWdeviceptr *dptr, size_t *pPitch,
                                      size_t WidthInBytes, size_t Height,
                                      unsigned int ElementSizeBytes);
LWresult ucm_orig_lwMemAllocPitch(LWdeviceptr *dptr, size_t *pPitch,
                                  size_t WidthInBytes, size_t Height,
                                  unsigned int ElementSizeBytes);
LWresult ucm_lwMemAllocPitch(LWdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height,
                             unsigned int ElementSizeBytes);

/*lwMemHostGetDevicePointer*/
LWresult ucm_override_lwMemHostGetDevicePointer(LWdeviceptr *pdptr, void *p,
                                                unsigned int Flags);
LWresult ucm_orig_lwMemHostGetDevicePointer(LWdeviceptr *pdptr, void *p,
                                            unsigned int Flags);
LWresult ucm_lwMemHostGetDevicePointer(LWdeviceptr *pdptr, void *p, unsigned int Flags);

/*lwMemHostUnregister */
LWresult ucm_override_lwMemHostUnregister(void *p);
LWresult ucm_orig_lwMemHostUnregister(void *p);
LWresult ucm_lwMemHostUnregister(void *p);

/*lwdaFree*/
lwdaError_t ucm_override_lwdaFree(void *devPtr);
lwdaError_t ucm_orig_lwdaFree(void *devPtr);
lwdaError_t ucm_lwdaFree(void *devPtr);

/*lwdaFreeHost*/
lwdaError_t ucm_override_lwdaFreeHost(void *ptr);
lwdaError_t ucm_orig_lwdaFreeHost(void *ptr);
lwdaError_t ucm_lwdaFreeHost(void *ptr);

/*lwdaMalloc*/
lwdaError_t ucm_override_lwdaMalloc(void **devPtr, size_t size);
lwdaError_t ucm_orig_lwdaMalloc(void **devPtr, size_t size);
lwdaError_t ucm_lwdaMalloc(void **devPtr, size_t size);

/*lwdaMallocManaged*/
lwdaError_t ucm_override_lwdaMallocManaged(void **devPtr, size_t size,
                                           unsigned int flags);
lwdaError_t ucm_orig_lwdaMallocManaged(void **devPtr, size_t size, unsigned int flags);
lwdaError_t ucm_lwdaMallocManaged(void **devPtr, size_t size, unsigned int flags);

/*lwdaMallocPitch*/
lwdaError_t ucm_override_lwdaMallocPitch(void **devPtr, size_t *pitch,
                                         size_t width, size_t height);
lwdaError_t ucm_orig_lwdaMallocPitch(void **devPtr, size_t *pitch,
                                     size_t width, size_t height);
lwdaError_t ucm_lwdaMallocPitch(void **devPtr, size_t *pitch,
                                size_t width, size_t height);

/*lwdaHostGetDevicePointer*/
lwdaError_t ucm_override_lwdaHostGetDevicePointer(void **pDevice, void *pHost,
                                                  unsigned int flags);
lwdaError_t ucm_orig_lwdaHostGetDevicePointer(void **pDevice, void *pHost,
                                              unsigned int flags);
lwdaError_t ucm_lwdaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);


/*lwdaHostUnregister*/
lwdaError_t ucm_override_lwdaHostUnregister(void *ptr);
lwdaError_t ucm_orig_lwdaHostUnregister(void *ptr);
lwdaError_t ucm_lwdaHostUnregister(void *ptr);

#endif
