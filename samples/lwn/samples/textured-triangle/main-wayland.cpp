/*
 * Copyright (c) 2015-2020, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <lwnTool/lwnTool_GlslcInterface.h>

#include "lwnWin/lwn_win.h"

#define LWN_USE_C_INTERFACE         1
#include "Simple_Triangle.h"

#define LWN_DEBUG_ENABLED false

#define LWN_LOG printf

extern int offscreenWidth, offscreenHeight;

LWNSampleTestConfig testConfig;
LWNSampleTestCInterface testCInterface;

LWNSampleTestCInterface * LWNSampleTestConfig::m_c_interface = &testCInterface;


static void LWNAPIENTRY
lwnSampleDebugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                       LWNdebugCallbackSeverity severity, const char *message, void* userParam)
{
    LWN_LOG("LWN DEBUG ERROR: %s\n", (const char*) message);
}


extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);

int main(int argc, char **argv)
{
    static const int LOOPS_INFINITE = -1;
    int numLoops = LOOPS_INFINITE;

    LWNformat format = LWN_FORMAT_RGBA8;

    for (int i = 1; i < argc; ++i)  {
        if (strcmp(argv[i], "-n") == 0 && (i + 1) < argc) {
            numLoops = atol(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-RGBA8") == 0) {
            format = LWN_FORMAT_RGBA8;
            continue;
        }
        if (strcmp(argv[i], "-RGBX8") == 0) {
            format = LWN_FORMAT_RGBX8;
            continue;
        }
        if (strcmp(argv[i], "-RGBA8_SRGB") == 0) {
            format = LWN_FORMAT_RGBA8_SRGB;
            continue;
        }
        if (strcmp(argv[i], "-RGBX8_SRGB") == 0) {
            format = LWN_FORMAT_RGBX8_SRGB;
            continue;
        }
        if (strcmp(argv[i], "-BGRA8") == 0) {
            format = LWN_FORMAT_BGRA8;
            continue;
        }
        if (strcmp(argv[i], "-BGRA8_SRGB") == 0) {
            format = LWN_FORMAT_BGRA8_SRGB;
            continue;
        }
        printf("Invalid option or option missing value: %s\n", argv[i]);
        exit(1);
    }

    int width = 0;
    int height = 0;
    void *window = NULL;
    LwnWin *instance = LwnWin::GetInstance();
    if (instance == NULL) {
        LWN_LOG("Couldn't get native window instance.\n");
        return -1;
    }

    window = instance->CreateWindow("Textured Triangle", width, height);
    if (window == NULL) {
        LWN_LOG("Couldn't create native window instance.\n");
        return -1;
    }

    // Initialize the LWN function pointer interface.
    PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC) ((*lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
    if (!getProcAddress) {
        LWN_LOG("Couldn't initialize the LWN bootstrap loader (possible version mismatch).\n");
        instance->DestroyWindow(window);
        return -1;
    }
    // Set demo framebuffer to match (same as main-hos.cpp)
    offscreenWidth = width;
    offscreenHeight = height;

    lwnLoadCProcs(NULL, getProcAddress);

    LWNdeviceBuilder deviceBuilder;
    LWNdeviceFlagBits deviceFlags = LWNdeviceFlagBits(0);
    if (LWN_DEBUG_ENABLED) {
        deviceFlags = LWNdeviceFlagBits(LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT |
                                        LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT);
    }
    lwnDeviceBuilderSetDefaults(&deviceBuilder);
    lwnDeviceBuilderSetFlags(&deviceBuilder, deviceFlags);

    LWNdevice *device = new LWNdevice;
    if (!lwnDeviceInitialize(device, &deviceBuilder)) {
        LWN_LOG("Couldn't initialize the LWN device.\n");
        instance->DestroyWindow(window);
        return -1;
    }

    if (LWN_DEBUG_ENABLED) {
        lwnLoadCProcs(device, getProcAddress);
        lwnDeviceInstallDebugCallback(device, lwnSampleDebugCallback, NULL, LWN_TRUE);
    }

    // Check for API version mismatches.  Exit with an error if the major
    // version mismatches (major revisions are backward-incompatible) or if
    // the driver reports a lower minor version.
    int majorVersion, minorVersion;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_API_MAJOR_VERSION, &majorVersion);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_API_MINOR_VERSION, &minorVersion);
    if (majorVersion != LWN_API_MAJOR_VERSION || minorVersion < LWN_API_MINOR_VERSION) {
        LWN_LOG("API version mismatch (application compiled with %d.%d, "
                "driver reports %d.%d).\n",
                LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION,
                majorVersion, minorVersion);
        lwnDeviceFinalize(device);
        delete device;
        instance->DestroyWindow(window);
        return -1;
    }
    LWN_LOG("API version is compatible (application compiled with %d.%d, "
            "driver reports %d.%d).\n",
            LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION,
            majorVersion, minorVersion);


    // Create the "global" queue and command buffer used by the test.
    LWNqueue *queue = lwnDeviceCreateQueue(device);

    // Set up the C and interfaces for the LWN globals.
    testCInterface.device = device;
    testCInterface.queue = queue;

    testConfig.Init((LWNnativeWindow)window, format);

    while (numLoops != 0) {
        testConfig.cDisplay();

        if (numLoops != LOOPS_INFINITE) {
            --numLoops;
        }
    }

    // Teardown
    testConfig.Deinit();

    delete g_lwn.m_texIDPool;

    lwnQueueFinalize(queue);
    delete queue;

    lwnDeviceFinalize(device);
    delete device;
    instance->DestroyWindow(window);

    LWN_LOG("Test finished.");
    return 0;
}
