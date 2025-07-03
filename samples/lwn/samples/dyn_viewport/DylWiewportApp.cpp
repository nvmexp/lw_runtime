/*
* Copyright (c) 2016 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

///////////////////////////////////////////////////////////////////////////////////////
//
// Sample to illustrate and test lwnWindowSetCrop.
//
// lwnWindowSetCrop allows applications to define a rectangle inside a window texture
// that gets displayed.
// The sample uses a 1920x1080 window texture and renders a quad into this texture
// using different viewports. lwnWindowSetCrop is called to adapt the crop rectangle
// to match the viewport. Since in this case the entire viewport gets displayed,
// the quad will appear in the middle of the screen.
// The calls to lwnWindowSetCrop can be skipped and in this case the quad will
// appear at different positions of the screen.
// If a custom crop rectangle is defined the viewport is set to 0,0,1920,1080 but
// only the portion of the window texture that is inside the crop rectangle gets
// displayed.
//
// -n <num frames> Specifies how many frames should be rendered.
// --topLeft       Sets the window origin mode to UPPER_LEFT.
// --noCropRect    Skip calls to lwnWindowSetCrop.
// -x              X origin in Windows CS of custom crop rectangle.
// -y              Y origin in Windows CS of custom crop rectangle.
// -w              Width of custom crop rectangle.
// -h              Height of custom crop rectangle.
//
///////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lwn/lwn_CppFuncPtrImpl.h>
#include <lwn/lwn_FuncPtrImpl.h>

#include "DynamicViewport.h"
#include "DylwiewportApp.h"

#if defined(LW_HOS)
#include <nn/nn_Log.h>

#define PRINT_LOG NN_LOG
extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);
#elif defined(_WIN32)
#include <windows.h>

#define PRINT_LOG printf
#endif

static const int LOOPS_INFINITE = -1;

// List of different viewports that get tested if no custom crop rectangle is defined
static const lwn::Rectangle TestRect[] = {
    {    0,   0, 1920, 1080 },
    {    0,   0, 1280,  720 },
    {    0,   0,  640,  360 },
    { 1280,   0,  640,  360 },
    { 1280, 720,  640,  360 },
    {    0, 720,  640,  360 },
};


static void LWNAPIENTRY
lwnSampleDebugCallback(lwn::DebugCallbackSource::Enum  source, lwn::DebugCallbackType::Enum  type, int id,
                       lwn::DebugCallbackSeverity::Enum severity, const char *message, void* userParam)
{
    PRINT_LOG("LWN DEBUG ERROR: %s\n", (const char*)message);
}

DylwiewportApp::DylwiewportApp(int argc, char** argv, bool debugEnabled) :
    m_numLoops(LOOPS_INFINITE),
    m_frame(0),
    m_idx(1),
    m_numTests(sizeof(TestRect) / sizeof(lwn::Rectangle)),
    m_useOriginTopLeft(false),
    m_adjustCropRect(true),
    m_useLwstomRect(false),
    m_debugEnabled(debugEnabled),
    m_dv(NULL)
{
    m_lwstomRect.x = 0;
    m_lwstomRect.y = 0;
    m_lwstomRect.width = 0;
    m_lwstomRect.height = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)  {
        if (strcmp(argv[i], "-n") == 0 && (i + 1) < argc) {
            m_numLoops = atol(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-x") == 0 && (i + 1) < argc) {
            m_lwstomRect.x = atol(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-y") == 0 && (i + 1) < argc) {
            m_lwstomRect.y = atol(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-w") == 0 && (i + 1) < argc) {
            m_lwstomRect.width = atol(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-h") == 0 && (i + 1) < argc) {
            m_lwstomRect.height = atol(argv[i + 1]);
        }
        else if (strcmp(argv[i], "--topLeft") == 0)  {
            m_useOriginTopLeft = true;
        }
        else if (strcmp(argv[i], "--noCropRect") == 0)  {
            m_adjustCropRect = false;
        }
    }

    m_useLwstomRect = (m_lwstomRect.width > 0 && m_lwstomRect.height > 0);
}

DylwiewportApp::~DylwiewportApp()
{
    if (m_dv) {
        delete m_dv;
    }

    m_device.Finalize();
}

bool DylwiewportApp::init(LWNnativeWindow *win)
{
    if (!win) {
        return false;
    }

#if defined(_WIN32)
    PFNLWNBOOTSTRAPLOADERPROC lwnBootstrapLoader = NULL;
    lwnBootstrapLoader = (PFNLWNBOOTSTRAPLOADERPROC)wglGetProcAddress("rq34nd2ffz");
#endif

    PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC)((*lwnBootstrapLoader)("lwnDeviceGetProcAddress"));

    if (!getProcAddress) {
        PRINT_LOG("Couldn't initialize the LWN bootstrap loader (possible version mismatch).\n");
        return false;
    }

    lwn::lwnLoadCPPProcs(NULL, reinterpret_cast<lwn::DeviceGetProcAddressFunc>(getProcAddress));

    lwn::DeviceBuilder devBuilder;
    devBuilder.SetDefaults();

    lwn::DeviceFlagBits devFlags = lwn::DeviceFlagBits(0);

    if (m_debugEnabled) {
        devFlags = LWNdeviceFlagBits(LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT |
            LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT);
    }

    devBuilder.SetFlags(devFlags);

    m_device.Initialize(&devBuilder);

    int majorVersion, minorVersion;
    m_device.GetInteger(lwn::DeviceInfo::API_MAJOR_VERSION, &majorVersion);
    m_device.GetInteger(lwn::DeviceInfo::API_MINOR_VERSION, &minorVersion);

    if (majorVersion != LWN_API_MAJOR_VERSION || minorVersion < LWN_API_MINOR_VERSION) {
        PRINT_LOG("API version mismatch (application compiled with %d.%d, driver reports %d.%d).\n",
            LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION, majorVersion, minorVersion);

        m_device.Finalize();
        return false;
    }

    if (m_debugEnabled) {
        lwn::lwnLoadCPPProcs(&m_device, reinterpret_cast<lwn::DeviceGetProcAddressFunc>(getProcAddress));
        m_device.InstallDebugCallback(lwnSampleDebugCallback, NULL, true);
    }

    lwnLoadCProcs(reinterpret_cast<LWNdevice*>(&m_device), getProcAddress);

    if (m_useOriginTopLeft) {
        m_device.SetWindowOriginMode(lwn::WindowOriginMode::UPPER_LEFT);
    }
    else {
        m_device.SetWindowOriginMode(lwn::WindowOriginMode::LOWER_LEFT);
    }

    m_dv = new DynamicViewport(win, &m_device);

    m_dv->init();

    if (m_useLwstomRect) {
        // If a custom crop rectangle was defined we want to display a specific area out of the entire
        // windows texture => adapt the display rectangle but leave the viewport unchanged.
        m_dv->resize(m_lwstomRect.x, m_lwstomRect.y, m_lwstomRect.width, m_lwstomRect.height, false, true);
    } else {
        // Do an initial resize to have the viewport match the first test case and if not
        // specified otherwise adapt the crop rectangle to display the entire viewport.
        m_dv->resize(TestRect[0].x, TestRect[0].y, TestRect[0].width, TestRect[0].height, true, m_adjustCropRect);
    }

    return true;
}

bool DylwiewportApp::display()
{
    m_dv->display();

    if (!m_useLwstomRect && (++m_frame == m_switchFrame)) {
        m_frame = 0;
        // Resize viewport and if adjustCropRect is true, resize the crop rectangle as well. If both are resized
        // to the same value the quad is always displayed in the middle of the screen.
        // If only the viewport is resized the quad will be displayed according toe the viewport in different areas
        // of the screen.
        m_dv->resize(TestRect[m_idx].x, TestRect[m_idx].y, TestRect[m_idx].width, TestRect[m_idx].height, true, m_adjustCropRect);

        PRINT_LOG("Resize x: %d y: %d w: %d h: %d\n", TestRect[m_idx].x, TestRect[m_idx].y, TestRect[m_idx].width, TestRect[m_idx].height);

        lwn::Rectangle crop;
        m_dv->getCrop(crop);
        PRINT_LOG("Crop rectangle x: %d y: %d w: %d h: %d\n", crop.x, crop.y, crop.width, crop.height);

        m_idx = (m_idx + 1) % m_numTests;
    }
    if (m_numLoops != LOOPS_INFINITE) {
        --m_numLoops;
    }

    return (m_numLoops == 0);
}