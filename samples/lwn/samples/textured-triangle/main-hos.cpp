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
#include <nn/os.h>
#include <nn/nn_Log.h>
#include <nn/fs.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <lw/lw_MemoryManagement.h>
#include <lwnTool/lwnTool_GlslcInterface.h>

#define LWN_USE_C_INTERFACE         1
#include "Simple_Triangle.h"

#include <nn/nn_Assert.h>
#include <nn/init.h>

#include <lw/lw_MemoryManagement.h>

#include "lwwinsys.h"
#define LWN_DEBUG_ENABLED false

LWNSampleTestConfig testConfig;
LWNSampleTestCInterface testCInterface;

LWNSampleTestCInterface * LWNSampleTestConfig::m_c_interface = &testCInterface;

static LwWinSysWindowHandle  s_window;
static LwWinSysDesktopHandle s_desktop;

extern int offscreenWidth, offscreenHeight;

static void LWNAPIENTRY
lwnSampleDebugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                       LWNdebugCallbackSeverity severity, const char *message, void* userParam)
{
    NN_LOG("LWN DEBUG ERROR: %s\n", (const char*) message);
}

void display(void)
{
    testConfig.cDisplay();
}

void initWinSys()
{
    LwWinSysDesktopOpen(NULL, &s_desktop);

    LwSize sz;
    LwWinSysDesktopGetSize(s_desktop, &sz);

    LwError err;
    LwRect windowSize;
    windowSize.left   = 0;
    windowSize.top    = 0;
    windowSize.right  = sz.width;
    windowSize.bottom = sz.height;
    err = LwWinSysWindowCreate(s_desktop, "LWN", &windowSize, NULL, &s_window);
    NN_ASSERT(!err);

    // Set demo framebuffer to match.
    offscreenWidth = sz.width;
    offscreenHeight = sz.height;
}

namespace {

    const int FsHeapSize = 512 * 1024;
    const int GraphicsHeapSize = 64 * 1024 * 1024;
    const int TlsHeapSize = 1 * 1024 * 1024;
    const int GraphicsFirmwareMemorySize = 8 * 1024 * 1024;

    char                        g_FsHeapBuffer[FsHeapSize];
    nn::lmem::HeapHandle        g_FsHeap;
    char                        g_GraphicsHeapBuffer[GraphicsHeapSize];
    nn::mem::StandardAllocator  g_GraphicsAllocator(g_GraphicsHeapBuffer, sizeof(g_GraphicsHeapBuffer));
    char                        g_GraphicsFirmwareMemory[GraphicsFirmwareMemorySize] __attribute__((aligned(4096)));
    char                        g_TlsHeapBuffer[TlsHeapSize];
    nn::util::TypedStorage<nn::mem::StandardAllocator, sizeof(nn::mem::StandardAllocator),
                           NN_ALIGNOF(nn::mem::StandardAllocator)> g_TlsAllocator;

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
}

//-----------------------------------------------------------------------------
// nninitStartup() is ilwoked before calling nnMain().
//
extern "C" void nninitStartup()
{
    const size_t MallocMemorySize = 64 * 1024 * 1024;
    nn::Result result = nn::os::SetMemoryHeapSize( 512 * 1024 * 1024);
    uintptr_t address;
    result = nn::os::AllocateMemoryBlock( &address, MallocMemorySize );
    NN_ASSERT( result.IsSuccess() );
    nn::init::InitializeAllocator( reinterpret_cast<void*>(address), MallocMemorySize );

    // Set file system allocator and deallocator
    FsInitHeap();
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
    nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);
}

extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);

extern "C" void nnMain()
{
    static const int LOOPS_INFINITE = -1;
    int numLoops = LOOPS_INFINITE;

    int argc = nn::os::GetHostArgc();
    char** argv = nn::os::GetHostArgv();

    LWNformat format = LWN_FORMAT_RGBA8;

    for (int i = 1; i < argc; ++i)  {
        if (strcmp(argv[i], "-n") == 0 && (i + 1) < argc) {
            numLoops = atol(argv[i + 1]);
        }
        if (strcmp(argv[i], "-BGRA8") == 0) {
            format = LWN_FORMAT_BGRA8;
        }
        if (strcmp(argv[i], "-BGRA8_SRGB") == 0) {
            format = LWN_FORMAT_BGRA8_SRGB;
        }
    }

    lw::SetGraphicsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
    lw::InitializeGraphics(g_GraphicsFirmwareMemory, sizeof(g_GraphicsFirmwareMemory));

#if LWN_DEBUG_ENABLED
    // Use graphics allocator for devtools and debug layer as well.
    lw::SetGraphicsDevtoolsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
#endif

    initWinSys();

    // Initialize the LWN function pointer interface.
    PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC) ((*lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
    if (!getProcAddress) {
        NN_LOG("Couldn't initialize the LWN bootstrap loader (possible version mismatch).\n");
        LwWinSysDesktopClose(s_desktop);
        return;
    }

    if (getProcAddress)
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
        NN_LOG("Couldn't initialize the LWN device.\n");
        LwWinSysDesktopClose(s_desktop);
        return;
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
        NN_LOG("API version mismatch (application compiled with %d.%d, "
                "driver reports %d.%d).\n",
                LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION,
                majorVersion, minorVersion);
        lwnDeviceFinalize(device);
        delete device;
        LwWinSysDesktopClose(s_desktop);
        return;
    }
    NN_LOG("API version is compatible (application compiled with %d.%d, "
            "driver reports %d.%d).\n",
            LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION,
            majorVersion, minorVersion);


    // Create the "global" queue and command buffer used by the test.
    LWNqueue *queue = lwnDeviceCreateQueue(device);

    // Set up the C and interfaces for the LWN globals.
    testCInterface.device = device;
    testCInterface.queue = queue;

    testConfig.Init((LWNnativeWindow)LwWinSysWindowGetNativeHandle(s_window), format);

    while (numLoops != 0) {
        display();

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
    LwWinSysDesktopClose(s_desktop);

    NN_LOG("Test finished.");
}
