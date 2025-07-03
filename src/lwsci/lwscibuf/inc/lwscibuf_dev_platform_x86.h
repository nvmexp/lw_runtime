/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_DEV_PLATFORM_X86_H
#define INCLUDED_LWSCIBUF_DEV_PLATFORM_X86_H

#include <string.h>
#include <stdbool.h>


#include "lwscibuf_internal.h"
#include "lwscibuf_internal_x86.h"

#include "lwscilog.h"
#include "lwscicommon_libc.h"
#include "lwRmShim/lwRmShim.h"

#define LWRMSHIM_LIBRARY_NAME "liblwidia-allocator.so.1"

/* forward declaration */
typedef struct LwSciBufDevRec* LwSciBufDev;

LwSciError LwSciBufDevOpen(
    LwSciBufDev* newDev);

void LwSciBufDevClose(
    LwSciBufDev dev);

LwSciError LwSciBufDevGetDeviceHandleCount(
    LwSciBufDev dev,
    LwU32* deviceCount);

LwSciError LwSciBufDevIncreDeviceHandleCount(
    LwSciBufDev dev);

LwSciError LwSciBufDevGetRmHandle(
    LwSciBufDev dev,
    LwU32* hClientHandle);

LwSciError LwSciBufDevGetGpuDeviceInstance(
    LwSciBufDev dev,
    LwU32 gpuID,
    LwU32* gpuDeviceInstance);

LwSciError LwSciBufDevGetRmLibrary(
    LwSciBufDev dev,
    void** rmLib);

LwSciError LwSciBufDevGetRmSessionDevice(
    LwSciBufDev dev,
    LwRmShimSessionContext** rmSession,
    LwRmShimDeviceContext** rmDevice);

LwSciError LwSciBufCheckPlatformVersionCompatibility(
    bool* platformCompatibility);

LwSciError LwSciBufDevValidateUUID(
    LwSciBufDev dev,
    LwSciBufMemDomain memDomain,
    const LwSciRmGpuId* gpuId,
    uint64_t numGpus,
    bool* isValid);

#endif /* INCLUDED_LWSCIBUF_DEV_PLATFORM_X86_H */
