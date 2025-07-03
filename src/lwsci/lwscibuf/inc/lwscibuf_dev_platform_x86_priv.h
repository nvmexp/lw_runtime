/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_DEV_PLATFORM_X86_PRIV_H
#define INCLUDED_LWSCIBUF_DEV_PLATFORM_X86_PRIV_H

#include "lwscibuf_dev_platform_x86.h"
#include "lwscicommon_os.h"
#include "lwRmShim/lwRmShim.h"
#include "lwscibuf_utils_x86.h"

typedef LwRmShimError (*getVersionFunc)(LwRmShimVersion*);
typedef LwRmShimError (*sessionCreateFunc)(LwRmShimSessionContext*);
typedef LwRmShimError (*sessionDestroyFunc)(LwRmShimSessionContext*);
typedef LwRmShimError (*openGpuInstanceFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *,
    LwRmShimGpuOpenParams *);
typedef LwRmShimError (*closeGpuInstanceFunc) (
    LwRmShimSessionContext *,
    LwRmShimDeviceContext *);

typedef LwRmShimError (*validateUUIDFunc) (
    LwRmShimSessionContext*,
    LwRmShimUuidValidationParams*,
    bool*);

typedef struct {
    sessionCreateFunc sessionCreate;
    sessionDestroyFunc sessionDestroy;
    openGpuInstanceFunc rmOpenGpuInstance;
    closeGpuInstanceFunc rmCloseGpuInstance;
    validateUUIDFunc rmValidateUUID;
} LwSciBufRmShimDevFVT;

typedef struct {
    bool probed;
    LwU32 deviceId;
    LwU32 deviceInstance;
    LwU32 subdeviceInstance;
} LwSciBufRmDeviceInfo;

typedef struct LwSciBufDevRec {
    /* library pointer */
    void* lwRmShimLib;
    /* FVT for session and device related APIs */
    LwSciBufRmShimDevFVT rmShimDevFvt;
    /* Resman shim session context */
    LwRmShimSessionContext rmSession;
    /* Resman shim device context */
    LwRmShimDeviceContext* rmDeviceList;
} LwSciBufDevPriv;

#endif /* INCLUDED_LWSCIBUF_DEV_PLATFORM_TEGRA_PRIV_H */
