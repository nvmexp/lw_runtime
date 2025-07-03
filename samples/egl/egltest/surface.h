/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef _EGLTEST_SURFACE_H
#define _EGLTEST_SURFACE_H

#include <stdlib.h>

#if defined(LW_EGL_DESKTOP_COMPATIBLE_HEADERS)
// For mobile base types and lwos/lwutil functionality on desktop builds
#include "mobile_common.h"
#endif

#include "lwrm_surface.h"

#define EGLTEST_MAX_SURFACES 3

typedef struct {
    LwRmSurface rmSurface[EGLTEST_MAX_SURFACES];
    LwU32 numSurfaces;
    LwRmDeviceHandle rmDev;
    void *mapping;
} EglTestSurface;

EglTestSurface *LwEglTestSurfaceCreate(LwU32 type,
                                       LwU32 width,
                                       LwU32 height,
                                       bool vm,
                                       LwU32 vmId);

void LwEglTestSurfaceDestroy(EglTestSurface *surf);

#endif //_EGLTEST_SURFACE_H
