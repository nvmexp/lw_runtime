/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include "testclient.h"

#define EGL_LIBRARY "libEGL.so"

typedef void *(* fnEglGetProcAddress)(const char *);

static bool g_eglInitialized = false;
LwEglApiUtilityFuncs g_eglApiUtilityFuncs;
LwEglApiAccessFuncs g_eglApiAccessFuncs;
static fnEglGetProcAddress g_eglGetProcAddress = NULL;

/*
 * The imports table is filled by the client.
 */
LwError TestClientEglInit(
    LwEglApiClientFuncs *imports,
    const LwEglApiUtilityFuncs *utilities)
{
    g_eglApiUtilityFuncs = *utilities;

    // Todo: Set imports

    return LwSuccess;
}

/*
 * Initialize the test client, The EGL library is loaded.
 * The function called by the test app.
 */
LwError TestClientInit(void)
{
    LwError err = LwSuccess;

    if (g_eglInitialized) {
        LOG_INFO("Test client has already been initialized.\n");
        return LwSuccess;
    }

    void *eglLibrary = dlopen(EGL_LIBRARY, RTLD_NOW);
    if (!eglLibrary) {
        LOG_ERR("Cannot load %s library, (error %s).\n", EGL_LIBRARY, dlerror());
        return LwError_NotSupported;
    }

    g_eglGetProcAddress = (fnEglGetProcAddress)dlsym(eglLibrary, "eglGetProcAddress");
    if (!g_eglGetProcAddress) {
        dlclose(eglLibrary);
        LOG_ERR("Cannot get eglGetProcAddress.\n");
        return LwError_NotSupported;
    }

    // Get the client API registration function.
    LwEglApiGetAccessFnPtr accessApi =
        (LwEglApiGetAccessFnPtr)eglGetProcAddress("LwEglApiGetAccess");
    if (!accessApi) {
        dlclose(eglLibrary);
        LOG_ERR("Cannot find LwEglApiGetAccess.\n");
        return LwError_NotSupported;
    }

    accessApi(&g_eglApiAccessFuncs);

    LOG_INFO("The test client initialized successfully.\n");

    g_eglInitialized = true;
    return LwSuccess;
}

bool TestExpGlsiImageFromLwRmSurface(
    LwGlsiEglImageHandle *image,
    LwEglDisplayHandle display,
    const LwRmSurface *surf,
    LwU32 count)
{
    LOG_FUNC("display %p\n", display);

    LwError err = g_eglApiAccessFuncs.glsi.imageFromLwRmSurface(image, display, surf, count);
    return err == LwSuccess;
};


