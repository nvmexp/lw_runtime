/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "egltest.h"

/*
 * Global state
 */
EglTestState gEglState = {
    (NativeDisplayType)0,   // nativeDisplay
    (NativeWindowType)0,    // nativeWindow
    EGL_NO_DISPLAY,         // display
    EGL_NO_SURFACE,         // surface
    (EGLConfig)0,           // config
    EGL_NO_CONTEXT,         // context
    EGL_NO_STREAM_KHR,      // eglstream
    0,                      // eglStreamState
    640,                    // width
    480,                    // height
    16000,                  // latency
    16000,                  // acquireTimeout
    NULL,                   // streamBuffers
    0,                      // streamBufferCount
    0,                      // streamBufferLwrr
    {0},                    // streamBufferReady
    4,                      // metadataCount
    NULL,                   // iServer
    NULL,                   // iClient
};

EglTestArgs gTestArgs = {
    NULL,                   // testName
    0,                      // testNo
    PRODUCER_CONSUMER,      // procType
    0,                      // fifoLength
    SINGLE_PROCESS,         // processMode
    10,                     // maxFrames
    0,                      // vmId
    "",                     // ipAddr
};

bool gSignalStop = false;
static EGLBoolean eglInitialized = EGL_FALSE;
EGLint devCount = 0;
EGLDeviceEXT* devList = NULL;

/*
 * Declare and initialize EGL function pointers
 */
EXTENSION_LIST(EXTLST_DECL)

typedef void (*extlst_fnptr_t)(void);
static struct {
    extlst_fnptr_t *fnptr;
    char const *name;
    char const *extension;
    int required;
} extensionList[] = { EXTENSION_LIST(EXTLST_ENTRY) };

/*
 * Setup Extensions of EglStream
 */
int EGLSetupExtensions(EGLDisplay display)
{
    const char *exts;
    unsigned int i;

    exts = eglQueryString(display, EGL_EXTENSIONS);
    if (!exts) {
        return 0;
    }

    // Get addresses of EGL extension functions
    for (i = 0; i < (sizeof(extensionList) / sizeof(*extensionList)); i++) {
        if ((*extensionList[i].fnptr == NULL) && extensionList[i].required) {
            if (CheckExtension(exts, extensionList[i].extension)) {
                *extensionList[i].fnptr = eglGetProcAddress(extensionList[i].name);
                if (*extensionList[i].fnptr == NULL) {
                    LOG_ERR("Couldn't get address of %s().\n", extensionList[i].name);
                    return 0;
                }
            } else if (display != EGL_NO_DISPLAY) {
                // Some EGL extensions are only set up for EGLDisplay referred in eglInitialize.
                LOG_ERR("%s not supported.\n", extensionList[i].name);
                return 0;
            }
        }
    }
    return 1;
}

/*
 * Checks for extension name in extension string.  A stricter version of
 * strstr which enforces that the substring is null or space terminated.
 */
GLboolean CheckExtension(const char *extString, const char *extName)
{
    const char *p = extString;
    while ((p = strstr(p, extName))) {
        const char *q = p + strlen(extName);

        if (*q == ' ' || *q == '\0') {
            return GL_TRUE;
        }
        p = q;
    }
    return GL_FALSE;
}

EGLBoolean LwEglTestInit()
{
    const char *exts;
    EGLint n = 0;

    if (eglInitialized) {
        return EGL_TRUE;
    }

    if (!EGLSetupExtensions(EGL_NO_DISPLAY)) {
        LOG_ERR("EGLSetupExtensions failed.\n");
        goto fail;
    }

    // Load device list
    if (!eglQueryDevicesEXT(0, NULL, &n) || !n) {
        LOG_ERR("Failed to query devices.\n");
        goto fail;
    }

    devList = (EGLDeviceEXT*)malloc(n * sizeof(EGLDeviceEXT));
    if (!devList || !eglQueryDevicesEXT(n, devList, &devCount) || !devCount) {
        LOG_ERR("Failed to query devices, n = %d devCount = %d.\n", n, devCount);
        goto fail;
    }

    gEglState.display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
        (NativeDisplayType)devList[0], NULL);
    if (!gEglState.display) {
        LOG_ERR("Failed to get display for nativeDisplay %p.\n", devList[0]);
        goto fail;
    }

    gEglState.nativeDisplay = (NativeDisplayType)devList[0];

    if (!eglInitialize(gEglState.display, 0, 0)) {
        LOG_ERR("Failed to initialize display %p.\n", gEglState.display);
        goto fail;
    }

    if (!EGLSetupExtensions(gEglState.display)) {
        LOG_ERR("EGLSetupExtensions failed.\n");
        goto fail;
    }

    LOG_INFO("EGL initialized successfully.\n");

    eglInitialized = EGL_TRUE;
    return EGL_TRUE;

fail:
    LOG_ERR("Failed to initialize EGL.\n");

    return EGL_FALSE;
}

EGLBoolean LwEglTestTerm()
{
    EGLBoolean ret = EGL_TRUE;

    // Terminate EGL
    if (gEglState.display != EGL_NO_DISPLAY) {
        if (!eglTerminate(gEglState.display)) {
            LOG_ERR("Failed terminating display %p.\n", gEglState.display);
            ret = EGL_FALSE;
        }
        gEglState.display = EGL_NO_DISPLAY;
    }

    // Release EGL thread
    if (!eglReleaseThread()) {
        LOG_ERR("Failed releasing EGL thread.\n");
        ret = EGL_FALSE;
    }

    if (devList) {
        free(devList);
        devList = NULL;
    }

    eglInitialized = EGL_FALSE;
    return ret;
}

EGLBoolean LwEglTestCreateContextSurface()
{
    EGLConfig config;
    EGLint nConfig;
    EGLContext context = EGL_NO_CONTEXT;

    EGLint cfgAttribs[] = {
        EGL_RED_SIZE, 1,
        EGL_GREEN_SIZE, 1,
        EGL_BLUE_SIZE, 1,
        EGL_DEPTH_SIZE, 1,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, //EGL_OPENGL_ES3_BIT_KHR
        EGL_NONE,
    };
    EGLint ctxAttrs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    EGLint bufAttrs[] = {
        EGL_WIDTH, 256,
        EGL_HEIGHT, 256,
        EGL_NONE
    };

    if (!eglChooseConfig(gEglState.display, cfgAttribs, &config, 1, &nConfig)) {
        LOG_ERR("Failed to choose EGLConfig (error 0x%x).\n", eglGetError());
        goto failed;
    }

    gEglState.context = eglCreateContext(gEglState.display, config, context, ctxAttrs);
    if (gEglState.context == EGL_NO_CONTEXT) {
        LOG_ERR("Failed to create context.\n");
        goto failed;
    }

    gEglState.surface = eglCreatePbufferSurface(gEglState.display, config, bufAttrs);
    if (!gEglState.surface) {
        LOG_ERR("Failed to create Pbuffer surface.\n");
        goto failed;
    }

    if (!eglMakeLwrrent(gEglState.display, gEglState.surface, gEglState.surface, gEglState.context)) {
        LOG_ERR("Failed to make context current (error 0x%x)\n",  eglGetError());
        goto failed;
    }

    return EGL_TRUE;

failed:
    LwEglTestClearContextSurface();

    return EGL_FALSE;
}

EGLBoolean LwEglTestClearContextSurface()
{
    EGLBoolean ret = EGL_TRUE;

    // Clear the context/surface
    if (!eglMakeLwrrent(gEglState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)) {
        LOG_ERR("Failed to clear the context/surface.\n");
        ret = EGL_FALSE;
    }

    // Delete the EGL context
    if (gEglState.context != EGL_NO_CONTEXT) {
        if (!eglDestroyContext(gEglState.display, gEglState.context)) {
            LOG_ERR("Failed to destroy context %p.\n", gEglState.context);
            ret = EGL_FALSE;
        }
        gEglState.context = EGL_NO_CONTEXT;
    }

    // Delete the EGL surface
    if (gEglState.surface != EGL_NO_SURFACE) {
        if (!eglDestroySurface(gEglState.display, gEglState.surface)) {
            LOG_ERR("Failed tp destroy surface %p\n", gEglState.surface);
            ret = EGL_FALSE;
        }
        gEglState.surface = EGL_NO_SURFACE;
    }

    return ret;
}

/*
 * Catch SIGINT (Ctrl+C) to gracefully exit from the console
 */
static void sig_handler(int sig) {
    gSignalStop = true;
    LOG_MSG("Signal: %d\n", sig);
}

/*
 * Parse command line arguments
 */
static bool parseArgs(int argc, char **argv)
{
    bool ret = true;
    int c, tmp;

    while ((c = getopt(argc, argv, "ht:n:p:m:l:c:v:i:")) != -1) {
        switch (c) {
        case 't':
            gTestArgs.testName = optarg;
            if (strncmp(optarg, "nonstream", 10) &&
                strncmp(optarg, "stream", 10) &&
                strncmp(optarg, "stream2", 10) &&
                strncmp(optarg, "assert", 10)) {
                LOG_ERR("Invalid test name: [-t nonstream|stream|stream2|assert].\n");
                ret = false;
            }
            break;
        case 'n':
            tmp = atoi(optarg);
            if ((tmp > 0) && (tmp <= MAX_EGLSTREAM_TESTS)) {
                gTestArgs.testNo = tmp;
            } else {
                LOG_ERR("Invalid testNo: -n [1-%d].\n", MAX_EGLSTREAM_TESTS);
                ret = false;
            }
            break;
        case 'p':
            if (!strncmp(optarg, "producer", 8)) {
                if (gTestArgs.procType == CONSUMER) {
                    gTestArgs.procType = PRODUCER_CONSUMER;
                } else {
                    gTestArgs.procType = PRODUCER;
                }
            } else if (!strncmp(optarg, "consumer", 8)) {
                if (gTestArgs.procType == PRODUCER) {
                    gTestArgs.procType = PRODUCER_CONSUMER;
                } else {
                    gTestArgs.procType = CONSUMER;
                }
            } else {
                LOG_ERR("Invalid procType: [-p producer|consumer].\n");
                ret = false;
            }
            break;
        /*
         * Keep this around till we deprecate the '-m ' option and remove it from the
         * GVS script at tests-graphics/opengles/egl/egl/gvs_egltest.sh and
         * update it to use the '-l' option
        */
        case 'm':
            if (!strncmp(optarg, "mailbox", 7)) {
                gTestArgs.fifoLength = 0;
            } else if (!strncmp(optarg, "fifo", 4)) {
                gTestArgs.fifoLength = 1;
            }
            break;
        case 'l':
            tmp = atoi(optarg);
            if ((tmp >= 0) && (tmp <= MAX_EGLSTREAM_FIFO_LEN)) {
                gTestArgs.fifoLength = tmp;
            } else {
                LOG_ERR("Invalid fifo length: -l [0-%d]. 0 for mailbox mode (default)\n", MAX_EGLSTREAM_FIFO_LEN);
                ret = false;
            }
            break;
        case 'c':
            tmp = atoi(optarg);
            if (tmp == 1) {
                gTestArgs.processMode = CROSS_PARTITION;
            } else if (tmp != 0) {
                LOG_ERR("Invalid process mode: -c [0|1] 1: cross-partition 0: not cross-partition\n");
                ret = false;
            }
            break;
        case 'v':
            gTestArgs.vmId = atoi(optarg);
            break;
        case 'i':
            strcpy(gTestArgs.ipAddr, optarg);
            LOG_INFO("ipAddr %s\n", gTestArgs.ipAddr);
            break;
        case 'h':
            ret = false;
            break;
        default:
            LOG_ERR("Invalid command options.\n");
            ret = false;
        }
    }

    if ((gTestArgs.procType != PRODUCER_CONSUMER) &&
        (gTestArgs.processMode != CROSS_PARTITION)) {
        gTestArgs.processMode = CROSS_PROCESS;
    }

    if (ret == false) {
        LOG_MSG("Usage: egltest [-t nonstream|stream|assert] [-p producer|consumer]"
                " [-l 0-%d] [-c 1|0] [-v 0|1] [-i ip]\n", MAX_EGLSTREAM_FIFO_LEN);
        LOG_MSG("\t-l  0-%d: length of stream FIFO. 0 for mailbox mode (default). fifo mode otherwise.\n", MAX_EGLSTREAM_FIFO_LEN);
        LOG_MSG("\t-c  1: cross-partition 0: not cross-partition\n");
        LOG_MSG("\t-v  specify consumer's vmId for producer in cross-partition test\n");
        LOG_MSG("\t-i  specify Consumer's IP address for producer in cross-partition test\n");
    }

    return ret;
}

int main(int argc, char **argv)
{
    int ret = 1;

    // Hook up Ctrl-C handler
    signal(SIGINT, sig_handler);

    if (!parseArgs(argc, argv)) {
        return ret;
    }

    if (LwEglTestInit() != EGL_TRUE) {
        LOG_ERR("LwEglTestInit failed.\n");
        goto done;
    }

    if (TestClientInit() != LwSuccess) {
        LOG_ERR("TestClientInit failed.\n");
        goto done;
    }

    // Test non-stream related interfaces if [-t stream] is not used
    if ((gTestArgs.testName == NULL) || !strcmp(gTestArgs.testName, "nonstream")) {
        // TODO: add tests for utility functions
        //if (LwEglTestUtilityFuncs() != EGL_TRUE) {
        //    goto done;
        //}
    }

    // Test stream related interfaces if [-t nonstream] is not used
    if ((gTestArgs.testName == NULL) ||
        !strcmp(gTestArgs.testName, "stream") ||
        !strcmp(gTestArgs.testName, "stream2") ||
        !strcmp(gTestArgs.testName, "assert")) {
        if (LwEglTestEglStream() != EGL_TRUE) {
            goto done;
        }
    }

    ret = 0;

done:
    if (!LwEglTestTerm()) {
        LOG_ERR("LwEglTestTerm failed.\n");
        ret = 1;
    }

    if (ret || gSignalStop) {
        LOG_MSG("Test failed.\n");
    } else {
        LOG_MSG("Test passed.\n");
    }

    return ret;
}
