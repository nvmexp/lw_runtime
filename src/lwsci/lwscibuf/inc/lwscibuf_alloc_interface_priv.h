/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_INTERFACE_PRIV_H
#define INCLUDED_LWSCIBUF_ALLOC_INTERFACE_PRIV_H

#include "lwscibuf_alloc_interface.h"
#include "lwscibuf_alloc_sysmem.h"
#if (LW_IS_SAFETY == 0)
#include "lwscibuf_alloc_vidmem.h"
#endif

#define LWSCIBUF_ALLOC_IFACE_MAX_STR_SIZE   50

typedef LwSciError (*LwSciBufAllocIfaceOpenFnPtr)(
    LwSciBufDev devHandle,
    void** context);

typedef LwSciError (*LwSciBufAllocIfaceAllocFnPtr)(
    const void* context,
    void* allocVal,
    LwSciBufDev devHandle,
    LwSciBufRmHandle *rmHandle);

typedef LwSciError (*LwSciBufAllocIfaceDeAllocFnPtr)(
    void* context,
    LwSciBufRmHandle rmHandle);

typedef LwSciError (*LwSciBufAllocIfaceDupHandleFnPtr)(
    const void* context,
    LwSciBufAttrValAccessPerm newPerm,
    LwSciBufRmHandle rmHandle,
    LwSciBufRmHandle *dupRmHandle);

typedef LwSciError (*LwSciBufAllocIfaceMemMapFnPtr)(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrValAccessPerm accPerm,
    void** ptr);

typedef LwSciError (*LwSciBufAllocIfaceMemUnMapFnPtr)(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* ptr,
    uint64_t size);

typedef LwSciError (*LwSciBufAllocIfaceGetSizeFnPtr)(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* size);

typedef LwSciError (*LwSciBufAllocIfaceGetAlignmentFnPtr)(
    const void* context,
    LwSciBufRmHandle rmHandle,
    uint64_t* alignment);

typedef LwSciError (*LwSciBufAllocIfaceGetHeapTypeFnPtr)(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* heapType);

typedef LwSciError (*LwSciBufAllocIfaceCpuCacheFlushFnPtr)(
    const void* context,
    LwSciBufRmHandle rmHandle,
    void* cpuPtr,
    uint64_t len);

typedef void (*LwSciBufAllocIfaceCloseFnPtr)(
    void* context);

typedef LwSciError (*LwSciBufAllocIfaceGetAllocContextFnPtr)(
    const void* allocContextParams,
    void* openContext,
    void** allocContext);

typedef LwSciError (*LwSciBufAllocIfaceCreateAllocVal)(
    LwSciBufAllocIfaceVal iFaceAllocVal,
    void** allocVal);

typedef void (*LwSciBufAllocIfaceDestroyAllocVal)(
    void* allocVal);

typedef LwSciError (*LwSciBufAllocIfaceCreateAllocContextParam)(
    LwSciBufAllocIfaceAllocContextParams iFaceAllocContextParams,
    void** allocContextParams);

typedef void (*LwSciBufAllocIfaceDestroyAllocContextParam)(
    void* allocContextParams);

/**
 * @brief Structure specifying pointers to functions corresponding to
 * LwSciBufAllocIfaceType which are called by the unit level interfaces
 * to perform the following operations:
 * - Creation and deletion of allocation interface context
 * - Creation of allocation context
 * - Allocation and deallocation and duplication of the buffers
 * - Mapping and unmapping of the allocated buffers
 * - Flushing of the mapped buffers
 * - Get the properties of the buffers such as size
 */
typedef struct {
    LwSciBufAllocIfaceOpenFnPtr iFaceOpen;
    LwSciBufAllocIfaceAllocFnPtr iFaceAlloc;
    LwSciBufAllocIfaceDeAllocFnPtr iFaceDeAlloc;
    LwSciBufAllocIfaceDupHandleFnPtr iFaceDupHandle;
    LwSciBufAllocIfaceMemMapFnPtr iFaceMemMap;
    LwSciBufAllocIfaceMemUnMapFnPtr iFaceMemUnMap;
    LwSciBufAllocIfaceGetSizeFnPtr iFaceGetSize;
    LwSciBufAllocIfaceGetAlignmentFnPtr iFaceGetAlignment;
    LwSciBufAllocIfaceGetHeapTypeFnPtr iFaceGetHeapType;
    LwSciBufAllocIfaceCpuCacheFlushFnPtr iFaceCpuCacheFlush;
    LwSciBufAllocIfaceCloseFnPtr iFaceClose;
    LwSciBufAllocIfaceGetAllocContextFnPtr iFaceGetAllocContext;
} LwSciBufAllocIfaceFvt;

/**
 * @brief Structure specifying pointers to functions corresponding to
 * LwSciBufAllocIfaceType which are called by the unit level interfaces
 * to covert the members of LwSciBufAllocIfaceVal and
 * LwSciBufAllocIfaceAllocContextParams according to the
 * LwSciBufAllocIfaceType.
 */
typedef struct {
    LwSciBufAllocIfaceCreateAllocVal iFaceCreateAllocVal;
    LwSciBufAllocIfaceDestroyAllocVal iFaceDestroyAllocVal;
    LwSciBufAllocIfaceCreateAllocContextParam iFaceCreateAllocContextParams;
    LwSciBufAllocIfaceDestroyAllocContextParam iFaceDestroyAllocContextParams;
} LwSciBufAllocIfaceHelperFvt;

typedef struct {
    char heapName[LWSCIBUF_ALLOC_IFACE_MAX_STR_SIZE];
    LwSciBufAllocSysMemHeapType allocSysMemHeap;
} LwSciBufAllocIfaceToSysMemHeapMap;

static LwSciError LwSciBufAllocIfaceCreateSysMemAllocVal(
    LwSciBufAllocIfaceVal iFaceAllocVal,
    void** allocVal);

static void LwSciBufAllocIfaceDestroySysMemAllocVal(
    void* allocVal);

static LwSciError LwSciBufAllocIfaceCreateSysMemAllocContextParams(
    LwSciBufAllocIfaceAllocContextParams iFaceAllocContextParams,
    void** allocContextParams);

static void LwSciBufAllocIfaceDestroySysMemAllocContextParams(
    void* allocContextParams);

#if (LW_IS_SAFETY == 0)
static LwSciError LwSciBufAllocIfaceCreateVidMemAllocVal(
    LwSciBufAllocIfaceVal iFaceAllocVal,
    void** allocVal);

static void LwSciBufAllocIfaceDestroyVidMemAllocVal(
    void* allocVal);

static LwSciError LwSciBufAllocIfaceCreateVidMemAllocContextParams(
    LwSciBufAllocIfaceAllocContextParams iFaceAllocContextParams,
    void** allocContextParams);

static void LwSciBufAllocIfaceDestroyVidMemAllocContextParams(
    void* allocContextParams);
#endif /* LW_IS_SAFETY == 0 */

#endif /* INCLUDED_LWSCIBUF_ALLOC_INTERFACE_PRIV_H */
