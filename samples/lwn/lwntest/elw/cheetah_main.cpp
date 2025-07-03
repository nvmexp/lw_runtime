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

#if !defined(WIN_INTERFACE_LWSTOM)
#define WIN_INTERFACE_LWSTOM
#endif

#define LWN_USE_C_INTERFACE         1

#include "lwntest_c.h"
#include "ogtest.h"
#include "elw.h"
#include "testloop.h"
#if defined(LW_HOS)
#include <nn/os.h>
#include <nn/nn_Assert.h>
#include <nn/nn_SdkLog.h>
#include <nn/init.h>
#include <nn/fs.h>
#include <nn/gll.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <lw/lw_MemoryManagement.h>
#include <lwnTool/lwnTool_GlslcInterface.h>
#include <lwnUtil/lwnUtil_AlignedStorage.h>
#endif
#include <felw.h>
#include "lwn_utils.h"
#include "EGL/egl.h"
#include "cmdline.h"

#include "lwnWin/lwn_win.h"

#if defined(LWNTEST_LWDA_ENABLED)
#include "lwca.h"
#include "lwdaNNAllocator.h"
#endif

#if !defined(EGL_OPENGL_ES3_BIT)
// Apparently some versions of egl.h that we use don't have this yet.
#define EGL_OPENGL_ES3_BIT 0x00000040
#endif

static EGLDisplay            s_eglDisplay = EGL_NO_DISPLAY;

extern "C" void lwogSwapBuffers()
{
    // No need for eglSwapBuffers any more.
}

static void CleanUpEgl()
{
    if (s_eglDisplay != EGL_NO_DISPLAY) {
        eglTerminate(s_eglDisplay);
        s_eglDisplay = EGL_NO_DISPLAY;

        if (!eglReleaseThread()) {
            printf("eglReleaseThread failed.\n");
        }
    }
}

static bool InitEgl()
{
    s_eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!s_eglDisplay) {
        printf("eglGetDisplay failed.\n");
        return false;
    }
    if (!eglInitialize(s_eglDisplay, 0, 0)) {
        printf("eglInitialize failed.\n");
        return false;
    }
    EGLint configAttribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    EGLint numConfigs = 0;
    EGLConfig config;
    if (!eglChooseConfig(s_eglDisplay, configAttribs, &config, 1, &numConfigs) ||
            numConfigs != 1) {
        printf("eglChooseConfig failed.\n");
        CleanUpEgl();
        return false;
    }
    EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 6,
        EGL_NONE
    };
    eglBindAPI(EGL_OPENGL_API);
    EGLContext context = eglCreateContext(s_eglDisplay, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        printf("eglCreateContext failed.\n");
        CleanUpEgl();
        return false;
    }
    if (!eglMakeLwrrent(s_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) {
        printf("eglMakeLwrrent failed.\n");
        CleanUpEgl();
        return false;
    }
    return true;
}

void lwogTerminate(lwogExitStatus exitStatus)
{
#if defined (LW_HOS)
    // Call finalize here since nnMain will never return
    // from c_main
#endif

    CleanUpEgl();

    switch (exitStatus) {
    case EXIT_STATUS_NORMAL:
    case EXIT_STATUS_USER_QUIT:
    case EXIT_STATUS_NORMAL_NO_RESULTS:
        exit(EXIT_SUCCESS);
    default:
        exit(EXIT_FAILURE);
    }
}

static bool InitGraphics()
{
    LwnWin *lwnWin = LwnWin::GetInstance();
    if (lwnWin == NULL) {
        printf("Cannot obtain window interface.\n");
        return false;
    }

    void *nativeWindow = lwnWin->CreateWindow("LWN", lwrrentWindowWidth, lwrrentWindowHeight);

    if (!nativeWindow) {
        printf("Cannot create window.\n");
        return false;
    }

    g_lwnWindowFramebuffer.setNativeWindow(nativeWindow);

    if (useGL) {
        // XXX: lwn_gl_interop requires a GLES context, which requires that the context be made
        // current *before* the first LWN device is created. When GL and EGL are more decoupled
        // from LWN this restriction can be lifted.
        if (!InitEgl()) {
            return false;
        }
#if defined(LW_HOS)
        nngllResult gllResult = nngllInitializeGl();
        if (gllResult != nngllResult_Succeeded) {
            printf("nngllInitializeGl failed: %d\n", gllResult);
            CleanUpEgl();
            return false;
        }
#endif
    }

    return true;
}


static int c_main()
{
    InitNonGraphics();
    if (InitGraphics()) {
        MainLoop();
    }

    return 0;
}

#if defined(LW_HOS)

namespace {

    const size_t FsHeapSize       = 512 * 1024;
    const size_t TlsHeapSize      = 1 * 1024 * 1024;

    char                        g_FsHeapBuffer[FsHeapSize];
    nn::lmem::HeapHandle        g_FsHeap;

    char                        g_TlsHeapBuffer[TlsHeapSize];
    nn::util::TypedStorage<nn::mem::StandardAllocator, sizeof(nn::mem::StandardAllocator),
                           NN_ALIGNOF(nn::mem::StandardAllocator)> g_TlsAllocator;

    char                        *g_MallocHeapBuffer;
    size_t                      g_MallocHeapSize;

    char                        *g_GraphicsHeapBuffer;
    size_t                      g_GraphicsHeapSize;
    nn::mem::StandardAllocator  g_GraphicsAllocator;

    char                        *g_DevtoolsHeapBuffer;
    size_t                      g_DevtoolsHeapSize;
    nn::mem::StandardAllocator  g_DevtoolsAllocator;

    char                        *g_CompilerHeapBuffer;
    size_t                      g_CompilerHeapSize;
    nn::mem::StandardAllocator  g_CompilerAllocator;

    char                        *g_GraphicsFirmwareMemory;
    size_t                      g_GraphicsFirmwareMemorySize;

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

    void* CompilerAllocate(size_t size, size_t alignment, void *userPtr)
    {
        return g_CompilerAllocator.Allocate(size, alignment);
    }

    void CompilerFree(void *addr, void *userPtr)
    {
        g_CompilerAllocator.Free(addr);
    }

    void *CompilerReallocate(void* addr, size_t newSize, void *userPtr)
    {
        return g_CompilerAllocator.Reallocate(addr, newSize);
    }

    void InitAllocatorMemory()
    {
        size_t heapPageSize = 4096;

        g_MallocHeapSize = hosMallocHeapMB * 1024 * 1024;
        NN_ASSERT(0 == (g_MallocHeapSize % heapPageSize));

        g_GraphicsHeapSize = hosGraphicsHeapMB * 1024 * 1024;
        NN_ASSERT(0 == (g_GraphicsHeapSize % heapPageSize));

        g_DevtoolsHeapSize = hosDevtoolsHeapMB * 1024 * 1024;
        NN_ASSERT(0 == (g_DevtoolsHeapSize % heapPageSize));

        g_CompilerHeapSize = hosCompilerHeapMB * 1024 * 1024;
        NN_ASSERT(0 == (g_CompilerHeapSize % heapPageSize));

        g_GraphicsFirmwareMemorySize = hosFirmwareMemMB * 1024 * 1024;
        NN_ASSERT(0 == (g_GraphicsFirmwareMemorySize % heapPageSize));

        size_t totalHeapSize = g_MallocHeapSize + g_GraphicsHeapSize + g_DevtoolsHeapSize + g_CompilerHeapSize + g_GraphicsFirmwareMemorySize;
        totalHeapSize += heapPageSize;  // for page alignment, just in case
        totalHeapSize = AlignSize(totalHeapSize, nn::os::MemoryHeapUnitSize);

        nn::Result result;
        result = nn::os::SetMemoryHeapSize(totalHeapSize);
        NN_ASSERT(result.IsSuccess());

        uintptr_t address;
        result = nn::os::AllocateMemoryBlock(&address, totalHeapSize);
        NN_ASSERT(result.IsSuccess());
        char *mem = AlignPointer((char *) address, heapPageSize);

        // Split up our big memory block into chunks to be used when setting
        // up allocators.
        g_MallocHeapBuffer = mem;
        g_GraphicsHeapBuffer = g_MallocHeapBuffer + g_MallocHeapSize;
        g_DevtoolsHeapBuffer = g_GraphicsHeapBuffer + g_GraphicsHeapSize;
        g_CompilerHeapBuffer = g_DevtoolsHeapBuffer + g_DevtoolsHeapSize;
        g_GraphicsFirmwareMemory = g_CompilerHeapBuffer + g_CompilerHeapSize;
    }
}

//-----------------------------------------------------------------------------
// nninitStartup() is ilwoked before calling nnMain().
//
extern "C" void nninitStartup()
{
    // Set file system allocator and deallocator
    FsInitHeap();
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    // The TLS allocator is used before nnMain runs, and possibly before the malloc
    // allocator is initialized, so it must be initialized early and with its own static
    // storage.
    new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
    nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);

    // Parse the command line options during initial startup because options
    // affect the heap sizes we may use.
    int argc = nn::os::GetHostArgc();
    char** argv = nn::os::GetHostArgv();
    ParseCmdLine(argc, argv);

    // Set up memory for allocators.
    InitAllocatorMemory();

    // Set up our malloc() heap during startup in case we have any static
    // constructors needing to allocate memory before running nnMain().
    nn::init::InitializeAllocator(g_MallocHeapBuffer, g_MallocHeapSize);
}

extern "C" void nnMain()
{
    // Bugs 1674275, 1663555: Set rounding mode.
    fesetround(FE_TONEAREST);

    // Initialize the various memory allocators.
    g_GraphicsAllocator.Initialize(g_GraphicsHeapBuffer, g_GraphicsHeapSize);
    lw::SetGraphicsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);

    lw::InitializeGraphics(g_GraphicsFirmwareMemory, g_GraphicsFirmwareMemorySize);

    g_DevtoolsAllocator.Initialize(g_DevtoolsHeapBuffer, g_DevtoolsHeapSize);
    lw::SetGraphicsDevtoolsAllocator(DevtoolsAllocate, DevtoolsFree, DevtoolsReallocate, NULL);

    g_CompilerAllocator.Initialize(g_CompilerHeapBuffer, g_CompilerHeapSize);
    glslcSetAllocator(CompilerAllocate, CompilerFree, CompilerReallocate, NULL);

#if defined(LWNTEST_LWDA_ENABLED)
    LWresult status = lwNNSetAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);

    if (LWDA_SUCCESS != status) {
        printf("Failed in lwNNSetAllocator\n");
        return;
    }
#endif

    int result = c_main();

    if (result != 0) {
        NN_SDK_LOG("nnMain Failed!\n");
    }
}
#else

#if defined(LW_LINUX)
// On L4T, liblwn includes stubs for several symbols that on HOS would be found
// in other system libraries. The code below is used to validate that those
// symbols are present in liblwn.

namespace lw {
    // Taken from <lw/lw_MemoryManagement.h>
    typedef void* (*AllocateFunction)(size_t size, size_t alignment, void *userPtr);
    typedef void  (*FreeFunction)(void *addr, void *userPtr);
    typedef void* (*ReallocateFunction)(void* addr, size_t newSize, void *userPtr);

    // Prototypes
#define NN_NOEXCEPT noexcept
    void SetGraphicsAllocator(AllocateFunction allocate, FreeFunction free, ReallocateFunction reallocate, void *userPtr) NN_NOEXCEPT;
    void SetGraphicsDevtoolsAllocator(AllocateFunction allocate, FreeFunction free, ReallocateFunction reallocate, void *userPtr) NN_NOEXCEPT;
    void InitializeGraphics(void *memory, size_t size) NN_NOEXCEPT;
    void FinalizeGraphics() NN_NOEXCEPT;
    void GetGraphicsMemoryInfo(struct MemoryInfo *memoryInfo) NN_NOEXCEPT;
    void DumpGraphicsMemoryInfo() NN_NOEXCEPT;
#undef NN_NOEXCEPT
}

namespace {

    void ValidateLwNamespaceStubs()
    {
        // Many of these functions are normally called from nnMain. Since there
        // are no return values or side effects to check (besides segfault), we
        // will leave these in the same file as nnMain rather than artificially
        // constructing a test that will always pass.
        lw::SetGraphicsAllocator(NULL, NULL, NULL, NULL);
        lw::InitializeGraphics(NULL, 0);
        lw::SetGraphicsDevtoolsAllocator(NULL, NULL, NULL, NULL);
        lw::GetGraphicsMemoryInfo(NULL);
        lw::DumpGraphicsMemoryInfo();
        lw::FinalizeGraphics();
    }
}
#endif

int main(int argc, char** argv)
{
#if defined(LW_LINUX)
    // Call stubs to ensure they are present and won't segfault
    ValidateLwNamespaceStubs();
#endif
    ParseCmdLine(argc, argv);
    return c_main();
}
#endif

