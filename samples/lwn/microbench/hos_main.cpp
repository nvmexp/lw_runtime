/*
 * Copyright (c) 2015-2019, LWPU Corporation.  All rights reserved.
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
#include <math.h>
#include <nn/os.h>
#include <nn/init.h>
#include <nn/fs.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <lw/lw_MemoryManagement.h>
#include <lwnTool/lwnTool_GlslcInterface.h>

#if !defined(WIN_INTERFACE_LWSTOM)
#define WIN_INTERFACE_LWSTOM
#endif
#include <EGL/egl.h>

#define LWN_USE_C_INTERFACE         1

#include <nn/nn_Assert.h>

#include "lwos.h"
#include "lwwinsys.h"

#include "utils.hpp"
#include "bench.hpp"
#include "bench_ogl.hpp"
#include "bench_ctx.hpp"
#include "options.hpp"
#include "mediansampler.h"

static LwWinSysWindowHandle  s_window;
static LwWinSysDesktopHandle s_desktop;
static EGLNativeDisplayType  s_nativeDisplay;
static EGLDisplay            s_eglDisplay;
static EGLConfig             s_eglConfig;
static EGLContext            s_eglContext;
static EGLSurface            s_eglSurface;

/* attribute list for EGL */
static const EGLint attrList[] = {
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
    EGL_DEPTH_SIZE, 24,
    EGL_BUFFER_SIZE, 32,
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_NONE
};

static const int32_t OFFSCREEN_WIDTH = 1920, OFFSCREEN_HEIGHT = 1200;

LWNdevice*      s_device;

static void initEgl(bool forceOffscreen)
{
    s_nativeDisplay = LwWinSysDesktopGetNativeHandle(s_desktop);
    s_eglDisplay    = eglGetDisplay(s_nativeDisplay);
    eglInitialize(s_eglDisplay, NULL, NULL);
    eglBindAPI(EGL_OPENGL_API);

    /* get an appropriate EGL frame buffer configuration */
    int ncfg = 0;
    int ret = eglChooseConfig(s_eglDisplay, attrList, &s_eglConfig, 1, &ncfg);
    NN_ASSERT(ret == EGL_TRUE);
    NN_ASSERT(ncfg != 0);

    static const EGLint ctxAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE
    };

    s_eglContext = eglCreateContext(s_eglDisplay, s_eglConfig, EGL_NO_CONTEXT, ctxAttribs);
    NN_ASSERT(s_eglContext != EGL_NO_CONTEXT);

    if (!forceOffscreen) {
        s_eglSurface = eglCreateWindowSurface(s_eglDisplay,
                                              s_eglConfig,
                                              (EGLNativeWindowType)LwWinSysWindowGetNativeHandle(s_window),
                                              NULL);
    } else {
        // Force an offscreen pbuffer mode if running in benchmark
        // mode.  This will guarantee a fixed size framebuffer
        // independent of what the device display size happens to be.
        static const EGLint pbufAttribs[] = {
            EGL_WIDTH,  OFFSCREEN_WIDTH,
            EGL_HEIGHT, OFFSCREEN_HEIGHT,
            EGL_NONE
        };
        s_eglSurface = eglCreatePbufferSurface(s_eglDisplay, s_eglConfig, pbufAttribs);
    }

    eglMakeLwrrent(s_eglDisplay, s_eglSurface, s_eglSurface, s_eglContext);

    NN_ASSERT(eglGetError() == EGL_SUCCESS);
}

#if !defined(NN_BUILD_CONFIG_SPEC_NX)
#define NN_BUILD_CONFIG_SPEC_NX
#include <nn/pcv/pcv.h>
#undef NN_BUILD_CONFIG_SPEC_NX
#else
#include <nn/pcv/pcv.h>
#endif

#define KHZ 1000
#define MHZ 1000000

static LwU32 s_Rates[nn::pcv::MaxNumClockRates + 1];

void setRates(uint32_t cpuRateMHz, uint32_t gpuRateMHz, uint32_t emcRateMHz)
{
    nn::pcv::Initialize();

    if (LwOsDrvOpen("/dev/lwhost-ctrl-gpu") < 0) {
        NN_LOG("Failed setRates LwOsDrvOpen\n");
        return;
    }

    for(int i=0;i<3;i++) {
        nn::Result result;
        nn::pcv::ModuleState state;
        nn::pcv::ClockRatesListType type;
        nn::pcv::Module module_id;
        LwU32 SetRateHz, GetRateHz, UserRate;
        const char *domain;
        LwS32 num = nn::pcv::MaxNumClockRates;

        switch (i) {
        case 0:
            module_id = nn::pcv::Module_Cpu;
            UserRate = cpuRateMHz;
            domain = "CPU";
            break;
        case 1:
            module_id = nn::pcv::Module_Gpu;
            UserRate = gpuRateMHz;
            domain = "GPU";
            break;
        case 2:
            module_id = nn::pcv::Module_Emc;
            UserRate = emcRateMHz;
            domain = "EMC";
            break;
        }

        if (!nn::pcv::GetPossibleClockRates(
            &type, s_Rates, &num, module_id, num).IsSuccess()) {
            NN_LOG("Failed %s setRates QueryPossibleRates\n", domain);
            continue;
        }

        // Print available rates and pick the closest one to command line param
        NN_LOG("Available %s rates: ", domain);

        LwU32 maxDiff = 1000 * MHZ;
        UserRate *= MHZ;
        SetRateHz = 0;
        for (int j=0;j<num;j++) {
            LwU32 diff = abs((long)(s_Rates[j] - UserRate));
            if (diff < maxDiff) {
                SetRateHz = s_Rates[j];
                maxDiff = diff;
            }
            NN_LOG("%d ", s_Rates[j]);
        }

        NN_LOG("\nSetting rate: %d\n", SetRateHz);

        nn::pcv::SetClockRate(module_id, SetRateHz);

        // Check against S/w settings
        state.clockFrequency = 0;
        nn::pcv::GetState(&state, module_id);
        GetRateHz = state.clockFrequency;

        if (GetRateHz != SetRateHz) {
            NN_LOG("Failed frequency comparison: ");
        }

        NN_LOG("%s Freq hz set = %u / get = %u\n",
               domain, SetRateHz, GetRateHz);
    }
}

namespace {

    const int FsHeapSize       = 512 * 1024;
    const int MallocHeapSize   = 512 * 1024 * 1024;
    const int GraphicsHeapSize = 256 * 1024 * 1024;
    const int TlsHeapSize      = 1 * 1024 * 1024;
    const int DevtoolsHeapSize =  32 * 1024 * 1024;
    const int GraphicsFirmwareMemorySize = 8 * 1024 * 1024;

    char                        g_MallocHeapBuffer[MallocHeapSize];
    char                        g_FsHeapBuffer[FsHeapSize];
    nn::lmem::HeapHandle        g_FsHeap;
    char                        g_GraphicsHeapBuffer[GraphicsHeapSize];
    nn::mem::StandardAllocator  g_GraphicsAllocator(g_GraphicsHeapBuffer, sizeof(g_GraphicsHeapBuffer));
    char                        g_GraphicsFirmwareMemory[GraphicsFirmwareMemorySize] __attribute__((aligned(4096)));
    char                        g_TlsHeapBuffer[TlsHeapSize];
    nn::util::TypedStorage<nn::mem::StandardAllocator, sizeof(nn::mem::StandardAllocator),
                           NN_ALIGNOF(nn::mem::StandardAllocator)> g_TlsAllocator;
    char                        g_DevtoolsHeapBuffer[DevtoolsHeapSize];
    nn::mem::StandardAllocator  g_DevtoolsAllocator;

    void FsInitHeap()
    {
        g_FsHeap = nn::lmem::CreateExpHeap(g_FsHeapBuffer, FsHeapSize, nn::lmem::CreationOption_DebugFill);
    }

    void* FsAllocate(size_t size)
    {
        return nn::lmem::AllocateFromExpHeap(g_FsHeap, size);
    }

    void FsDeallocate(void* p, size_t size)
    {
        NN_UNUSED(size);
        return nn::lmem::FreeToExpHeap(g_FsHeap, p);
    }

    void* GraphicsAllocate(size_t size, size_t alignment, void *userPtr)
    {
        return g_GraphicsAllocator.Allocate(size, alignment);
    }

    void GraphicsFree(void *addr, void *userPtr)
    {
        g_GraphicsAllocator.Free(addr);
    }

    void *GraphicsReallocate(void* addr, size_t newSize, void *userPtr)
    {
        return g_GraphicsAllocator.Reallocate(addr, newSize);
    }

    void* TlsAlloc(size_t size, size_t alignment)
    {
        return nn::util::Get(g_TlsAllocator).Allocate(size, alignment);
    }

    void TlsDealloc(void* p, size_t size)
    {
        nn::util::Get(g_TlsAllocator).Free(p);
        NN_UNUSED(size);
    }

    void* DevtoolsAllocate(size_t size, size_t alignment, void *userPtr)
    {
       return g_DevtoolsAllocator.Allocate(size, alignment);
    }

    void DevtoolsFree(void *addr, void *userPtr)
    {
        g_DevtoolsAllocator.Free(addr);
    }

    void *DevtoolsReallocate(void* addr, size_t newSize, void *userPtr)
    {
        return g_DevtoolsAllocator.Reallocate(addr, newSize);
    }
}

extern "C" void nninitStartup()
{
    nn::init::InitializeAllocator(g_MallocHeapBuffer, sizeof(g_MallocHeapBuffer));

    // Set file system allocator and deallocator
    FsInitHeap();
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
    nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);
}

extern "C" void nnMain()
{
    lw::SetGraphicsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
    lw::InitializeGraphics(g_GraphicsFirmwareMemory, sizeof(g_GraphicsFirmwareMemory));

    g_DevtoolsAllocator.Initialize(g_DevtoolsHeapBuffer, DevtoolsHeapSize);
    lw::SetGraphicsDevtoolsAllocator(DevtoolsAllocate, DevtoolsFree, DevtoolsReallocate, NULL);

    glslcSetAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);

    // Parse command line options
    g_options.init(nn::os::GetHostArgc(), (const char**)nn::os::GetHostArgv());

#if 0
    //test that timer works
    LwnUtil::TimeofdayTimer* testTimer = LwnUtil::TimeofdayTimer::Instance();
    NN_LOG("start timer. Waiting 10 seconds\n");
    uint64_t testStart = testTimer->getTicks();
    while(1) {
        uint64_t ticks = testTimer->getTicks();
        if (ticks - testStart > 10000000000ull)
            break;
    }
    NN_LOG("stop timer\n");
#endif

    setRates(g_options.cpuRateMHz(), g_options.gpuRateMHz(), g_options.emcRateMHz());

    LwWinSysDesktopOpen(NULL, &s_desktop);

    LwSize sz;
    LwWinSysDesktopGetSize(s_desktop, &sz);

    LwError err;
    LwRect windowSize;
    windowSize.left = 0;
    windowSize.top = 0;
    windowSize.right = sz.width;
    windowSize.bottom = sz.height;
    err = LwWinSysWindowCreate(s_desktop, "LWN", &windowSize, NULL, &s_window);
    NN_ASSERT(!err);

    bool renderOffscreen = !(g_options.flags() & Options::FLIP_BIT);
    initEgl(true); //TODO for some reason creating an egl window surface messes lwnQueueAcquireTexture up. Cirlwmvent by always creating a pbuffer surface

    s_device = LwnUtil::init(g_options.flags() & Options::DEBUG_LAYER_BIT ? LwnUtil::LWN_INIT_DEBUG_LAYER_BIT : 0,
                             nullptr);
    if (!s_device)
        return;

    ResultCollector collector;
    ResultPrinterStdout printer;

    int width = renderOffscreen ? OFFSCREEN_WIDTH : sz.width;
    int height = renderOffscreen ? OFFSCREEN_HEIGHT : sz.height;

    {
        BenchmarkContextLwWinsysLWN ctx(s_device,
                                        (LWNnativeWindow)LwWinSysWindowGetNativeHandle(s_window),
                                        width, height);
        ctx.runAll(&collector);
    }

    if (g_options.flags() & Options::OPENGL_TESTS_BIT) {
        BenchmarkContextLwWinsysOGL::InitParams initParams;
        initParams.eglDisplay = s_eglDisplay;
        initParams.eglSurface = s_eglSurface;

        BenchmarkContextLwWinsysOGL ctx(initParams, width, height);
        ctx.runAll(&collector);
    }
    collector.print(printer);

    if (s_eglSurface != EGL_NO_SURFACE) {
        eglDestroySurface(s_device, s_eglSurface);
    }

    LwnUtil::exit();

    lwnDeviceFinalize(s_device);
    delete s_device;

    // Exit
    LwWinSysDesktopClose(s_desktop);
}
