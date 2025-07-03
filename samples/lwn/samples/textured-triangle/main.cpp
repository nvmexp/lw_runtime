/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
// lwn.cpp : Initialize GLUT, initalize the LWN emulation layer, and test LWN

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define LWN_USE_C_INTERFACE         1
#include "Simple_Triangle.h"

#include "GL/glut.h"

#ifdef _WIN32
#include <windows.h>
// Force the LWpu GPU on Optimus systems
extern "C" {
    _declspec(dllexport) unsigned long LwOptimusEnablement = 0x00000001;
}

#endif

static const int LOOPS_INFINITE = -1;
static int s_numLoops = LOOPS_INFINITE;

#define BOOTSTRAP_FUNC wglGetProcAddress
#define LWN_DEBUG_ENABLED false

#define log_output  printf
LWNSampleTestConfig testConfig;
LWNSampleTestCInterface testCInterface;

LWNSampleTestCInterface * LWNSampleTestConfig::m_c_interface = &testCInterface;

static void LWNAPIENTRY
lwnSampleDebugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                       LWNdebugCallbackSeverity severity, const char *message, void* userParam)
{
    fprintf(stderr, "LWN DEBUG ERROR: %s\n", (const char*) message);
    assert(!"LWN error");
}

void reshape(int w, int h)
{
}

void display(void)
{
    if (s_numLoops != LOOPS_INFINITE) {
        if (s_numLoops == 0) {
            testConfig.Deinit();
            exit(0);
        }
        --s_numLoops;
    }

    testConfig.cDisplay();
    glutSwapBuffers();
}

void idle(void)
{
    glutPostRedisplay();
}

void glutKey( unsigned char key, int x, int y )
{
    switch (key) {
    case '\033':
        exit(0);
    }
    glutPostRedisplay();
}

int main(int argc, char* argv[])
{
    LWNformat format = LWN_FORMAT_RGBA8;

    for (int i = 1; i < argc; ++i)  {
        if (strcmp(argv[i], "-n") == 0 && (i + 1) < argc) {
            s_numLoops = atol(argv[i + 1]);
        }
        if (strcmp(argv[i], "-BGRA8") == 0) {
            format = LWN_FORMAT_BGRA8;
        }
        if (strcmp(argv[i], "-BGRA8_SRGB") == 0) {
            format = LWN_FORMAT_BGRA8_SRGB;
        }
    }

    glutInit(&argc, (char **)argv);

    glutInitDisplayString("double depth rgba stencil");
    glutInitWindowSize (1280, 720);
    glutCreateWindow ((char *)argv[0]);
    glutSetWindowTitle("LWN Test App");

    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(glutKey);

    // Initialize the LWN function pointer interface.
    PFNLWNBOOTSTRAPLOADERPROC bootstrapLoader = NULL;
    PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress = NULL;
    bootstrapLoader = (PFNLWNBOOTSTRAPLOADERPROC) BOOTSTRAP_FUNC("rq34nd2ffz");
    if (bootstrapLoader) {
        getProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC) ((*bootstrapLoader)("lwnDeviceGetProcAddress"));
    }
    if (!bootstrapLoader || !getProcAddress) {
        fprintf(stderr, "Couldn't initialize the LWN bootstrap loader (possible version mismatch).\n");
        return 1;
    }
    lwnLoadCProcs(NULL, getProcAddress);
    if (lwnDeviceInitialize == NULL || lwnDeviceGetInteger == NULL || lwnDeviceFinalize == NULL) {
        return 1;
    }

    LWNdeviceBuilder deviceBuilder;
    lwnDeviceBuilderSetDefaults(&deviceBuilder);
    if (LWN_DEBUG_ENABLED) {
        lwnDeviceBuilderSetFlags(&deviceBuilder, (LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT |
                                                  LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT));
    } else {
        lwnDeviceBuilderSetFlags(&deviceBuilder, 0);
    }
    LWNdevice *device = new LWNdevice;
    if (!lwnDeviceInitialize(device, &deviceBuilder)) {
        fprintf (stderr, "Couldn't initialize the LWN device.\n");
        return 1;
    }
    if (LWN_DEBUG_ENABLED) {
        lwnLoadCProcs(device, getProcAddress);
        lwnDeviceInstallDebugCallback(device, lwnSampleDebugCallback, NULL, LWN_TRUE);
    }

    // Now load the rest of the function pointer interface.
    lwnLoadCProcs(device, getProcAddress);

    // Check for API version mismatches.  Exit with an error if the major
    // version mismatches (major revisions are backward-incompatible) or if
    // the driver reports a lower minor version.
    int majorVersion, minorVersion;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_API_MAJOR_VERSION, &majorVersion);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_API_MINOR_VERSION, &minorVersion);
    if (majorVersion != LWN_API_MAJOR_VERSION || minorVersion < LWN_API_MINOR_VERSION) {
        fprintf(stderr, "API version mismatch (application compiled with %d.%d, "
                "driver reports %d.%d).\n",
                LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION,
                majorVersion, minorVersion);
        return 1;
    }

    // Create the "global" queue used by the test.
    LWNqueue *queue = new LWNqueue;
    LWNqueueBuilder qb;
    lwnQueueBuilderSetDevice(&qb, device);
    lwnQueueBuilderSetDefaults(&qb);
    if (!lwnQueueInitialize(queue, &qb)) {
        delete queue;
        fprintf(stderr, "Could not create global LWN queue.\n");
        return 1;
    }

    // Set up the C and interfaces for the LWN globals.
    testCInterface.device = device;
    testCInterface.queue = queue;

    testConfig.Init(NULL, format);
    glutMainLoop();
    testConfig.Deinit();

    return 0;
}
