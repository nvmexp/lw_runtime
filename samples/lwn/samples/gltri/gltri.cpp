/*
 * Copyright (c) 2016-2019, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/*
 * gltri
 *
 * Demonstrates OpenGL ES rendering, setting up a window using EGL and nn::vi.
 */

#include "gltri.h"

#include <new>

#include <nn/fs.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <nn/nn_Assert.h>
#include <nn/os.h>
#include <nn/os/os_SdkMemoryAllocatorForThreadLocal.h>
#include <nn/vi.h>
#include <nn/init.h>
#include <lw/lw_MemoryManagement.h>

namespace {

const int MallocHeapSize = 512 * 1024 * 1024;
const int FsHeapSize = 512 * 1024;
const int GraphicsHeapSize = 128 * 1024 * 1024;
const int TlsHeapSize = 1 * 1024 * 1024;
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

extern "C" void nninitStartup()
{
    // Set file system allocator and deallocator
    FsInitHeap();
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
    nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);

    // Initialize default heap for malloc and new
    nn::init::InitializeAllocator(g_MallocHeapBuffer, MallocHeapSize);
}

extern "C" void nnMain()
{
    lw::SetGraphicsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
    lw::InitializeGraphics(g_GraphicsFirmwareMemory, sizeof(g_GraphicsFirmwareMemory));

    WindowMgr windowMgr;
    EglMgr eglMgr(windowMgr.GetNativeWindowHandle());
    GlMgr glMgr;
    for (int i = 0; i < 1000; ++i) {
        glMgr.DrawFrame(i * 0.05);
        eglMgr.SwapBuffers();
    }
}
