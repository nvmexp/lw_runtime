/* * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_SYSMEM_X86_PRIV_H
#define INCLUDED_LWSCIBUF_ALLOC_SYSMEM_X86_PRIV_H

#include <string.h>
#include <dlfcn.h>

#include "lwscibuf_alloc_sysmem.h"
#include "lwscibuf_utils_x86.h"
#include "lwRmShim/lwRmShim.h"

typedef LwRmShimError (*allocMemFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimMemoryContext *,
    LwRmShimAllocMemParams *);

typedef LwRmShimError (*openGpuInstanceFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimGpuOpenParams *);

typedef LwRmShimError (*mapMemFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimMemoryContext *,
    LwRmShimMemMapParams *);

typedef LwRmShimError (*dupMemFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimMemoryContext *,
    LwRmShimDupMemContextParams *);

typedef LwRmShimError (*unmapMemFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimMemoryContext *,
    LwRmShimMemUnMapParams *);

typedef LwRmShimError (*freeMemFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimMemoryContext *);

typedef LwRmShimError (*flushCpuCacheFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimMemoryContext *,
    LwRmShimMemFlushParams *);

typedef LwRmShimError (*closeGpuInstanceFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *);

typedef struct {
    allocMemFunc rmAllocMem;
    mapMemFunc rmMemMap;
    dupMemFunc rmDupMem;
    unmapMemFunc rmUnMapMem;
    freeMemFunc rmFreeMem;
    flushCpuCacheFunc rmFlushCpuCache;
    openGpuInstanceFunc rmOpenGpuInstance;
    closeGpuInstanceFunc rmCloseGpuInstance;
} LwSciBufRmShimAllocFVT;

typedef struct LwSciBufAllocSysMemContextRec LwSciBufAllocSysMemContext;

typedef struct {
    LwRmShimDeviceContext* rmDevicePtr;
    LwSciBufAllocSysMemContext* sysMemContextPtr;
} LwSciBufSysMemPerGpuContext;

struct LwSciBufAllocSysMemContextRec {
    /* Resman shim device pointer where memory is allocated */
    LwRmShimDeviceContext* rmDevicePtr;
    /* Resman shim session pointer for memory */
    LwRmShimSessionContext* rmSessionPtr;
    /* PerGpuContext */
    LwSciBufSysMemPerGpuContext* perGpuContext;
    /* Resman shim FVT for allocation related APIs */
    LwSciBufRmShimAllocFVT rmShimAllocFvt;
};

typedef struct {
    LwU64 size;
    LwU64 alignment;
    LwU64 offset;
    LwRmShimMemLocation location;
    LwRmShimCacheCoherency cacheCoherency;
} LwRmSysMemAllocVal;

typedef struct {
    LwRmShimMemMapping redirection;
    LwRmShimMemAccess access;
} LwResmanCpuMappingParam;

typedef enum {
    LwSciBufAllocResmanLocation_SysMem,
    LwSciBufAllocResmanLocation_VideMem,
    LwSciBufAllocResmanLocation_Ilwalid,
} LwSciBufAllocResmanLocation;

typedef enum {
    LwSciBufAllocResmanRedirection_Default,
    LwSciBufAllocResmanRedirection_Direct,
    LwSciBufAllocResmanRedirection_Reflected,
    LwSciBufAllocResmanRedirection_Ilwalid,
} LwSciBufAllocResmanRedirection;

typedef struct {
    LwSciBufAttrValAccessPerm access;
    LwSciBufAllocResmanRedirection redirection;
} LwSciBufResmanCpuMappingParam;

#endif /* INCLUDED_LWSCIBUF_ALLOC_SYSMEM_X86_PRIV_H */
