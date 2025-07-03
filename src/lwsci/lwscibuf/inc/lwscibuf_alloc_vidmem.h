/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_VIDMEM_H
#define INCLUDED_LWSCIBUF_ALLOC_VIDMEM_H

#include "lwscibuf_dev.h"
#include "lwscibuf_internal.h"

typedef struct {
    uint64_t size;
    uint64_t alignment;
    bool coherency;
    bool cpuMapping;
} LwSciBufAllocVidMemVal;

typedef struct {
    LwSciRmGpuId gpuId;
} LwSciBufAllocVidMemAllocContextParam;

LwSciError LwSciBufVidMemOpen(
    LwSciBufDev devHandle,
    void** context);

LwSciError LwSciBufVidMemAlloc(
    const void* context,
    void* allocVal,
    LwSciBufDev devHandle,
    LwSciBufRmHandle* rmHandle);

LwSciError LwSciBufVidMemDealloc(
    void* context,
    LwSciBufRmHandle rmHandle);

LwSciError LwSciBufVidMemDupHandle(
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle *dupRmHandle);

LwSciError LwSciBufVidMemMemMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void** ptr);

LwSciError LwSciBufVidMemMemUnMap(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size);

LwSciError LwSciBufVidMemGetSize(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size);

LwSciError LwSciBufVidMemGetAlignment(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* alignment);

LwSciError LwSciBufVidMemGetHeapType(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* heapType);

LwSciError LwSciBufVidMemCpuCacheFlush(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len);

void LwSciBufVidMemClose(
    void* context);

LwSciError LwSciBufVidMemGetAllocContext(
    const void* allocContextParam,
    void* openContext,
    void** allocContext);

#endif /* INCLUDED_LWSCIBUF_ALLOC_VIDMEM_H */
