/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
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
#include <nn/init.h>
#include <nn/nn_Log.h>
#include <nn/nn_Assert.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <lw/lw_MemoryManagement.h>
#include <lwnTool/lwnTool_GlslcInterface.h>

#include "lwwinsys.h"

#include "DylwiewportApp.h"

#define LWN_DEBUG_ENABLED false

namespace {

    const int MallocHeapSize = 128 * 1024 * 1024;
    const int GraphicsHeapSize = 64 * 1024 * 1024;
    const int GraphicsFirmwareMemorySize = 8 * 1024 * 1024;

    char                        g_MallocHeapBuffer[MallocHeapSize];
    char                        g_GraphicsHeapBuffer[GraphicsHeapSize];
    nn::mem::StandardAllocator  g_GraphicsAllocator(g_GraphicsHeapBuffer, sizeof(g_GraphicsHeapBuffer));
    char                        g_GraphicsFirmwareMemory[GraphicsFirmwareMemorySize] __attribute__((aligned(4096)));

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
}

extern "C" void nninitStartup()
{
    nn::init::InitializeAllocator(g_MallocHeapBuffer, MallocHeapSize);
}

extern "C" void nnMain()
{
    static const int LOOPS_INFINITE = -1;
    int numLoops = LOOPS_INFINITE;
    bool useOriginTopLeft = false;
    bool adjustCropRect = true;
    lwn::Rectangle lwstomRect = { 0 };

    int argc = nn::os::GetHostArgc();
    char** argv = nn::os::GetHostArgv();

    lw::SetGraphicsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
    lw::InitializeGraphics(g_GraphicsFirmwareMemory, sizeof(g_GraphicsFirmwareMemory));

    glslcSetAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);

#if LWN_DEBUG_ENABLED
    // Use graphics allocator for devtools and debug layer as well.
    lw::SetGraphicsDevtoolsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
#endif

    LwWinSysWindowHandle  window;
    LwWinSysDesktopHandle desktop;

    LwWinSysDesktopOpen(NULL, &desktop);

    LwSize sz;
    LwWinSysDesktopGetSize(desktop, &sz);

    LwError err;
    LwRect windowSize;
    windowSize.left = 0;
    windowSize.top = 0;
    windowSize.right = sz.width;
    windowSize.bottom = sz.height;

    err = LwWinSysWindowCreate(desktop, "LWN", &windowSize, NULL, &window);
    NN_ASSERT(!err);

    {
        DylwiewportApp app(argc, argv, LWN_DEBUG_ENABLED);

        if (!app.init(static_cast<LWNnativeWindow*>(LwWinSysWindowGetNativeHandle(window)))) {
            NN_LOG("Couldn't initialize the LWN bootstrap loader (possible version mismatch).\n");
            LwWinSysDesktopClose(desktop);
            return;
        }

        bool done = false;

        while (!done) {
            done = app.display();
        }
    }

    LwWinSysDesktopClose(desktop);
}
