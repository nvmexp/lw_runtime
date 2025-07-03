/*
 * Copyright (c) 2015-2019, LWPU Corporation.  All rights reserved.
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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#define LWN_USE_C_INTERFACE         1
#define LWN_USE_CPP_INTERFACE       1
#define LWN_OVERLOAD_CPP_OBJECTS    1
#include "lwnexample.h"

#include "lwn/lwn_FuncPtrImpl.h"     // Code to set up LWN C function pointer interface
#include "lwn/lwn_CppFuncPtrImpl.h"  // Code to set up LWN C++ function pointer interface

#include "lwplatform.h"

#ifdef _WIN32
#include <windows.h>
#elif defined __ANDROID__
#include <EGL/egl.h>
#elif defined __linux__
#include <GL/glx.h>
#endif

#ifdef _WIN32
#define BOOTSTRAP_FUNC      wglGetProcAddress
#elif __ANDROID__
#define BOOTSTRAP_FUNC      eglGetProcAddress
#elif __linux__
#elif __ANDROID__
#define BOOTSTRAP_FUNC      glXGetProcAddressARB
#elif LW_HOS
#include <nn/nn_Common.h>
#include <nn/nn_SdkLog.h>
#include <nn/nn_Assert.h>
#include <nn/os.h>
#include <nn/init.h>
#include <nn/fs.h>
#include <nn/lmem/lmem_ExpHeap.h>
#include <nn/mem/mem_StandardAllocator.h>
#include <felw.h>
#include <lw/lw_MemoryManagement.h>
#include <lwnTool/lwnTool_GlslcInterface.h>
#else
#error Need to figure out an appropriate BOOTSTRAP_FUNC.
#endif

#if defined __ANDROID__
#include <android/log.h>
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "LWNTest", __VA_ARGS__))
#define log_output  LOGI
#else
#define log_output  printf
#endif

#if __ANDROID__ || LW_HOS
extern "C"
{
    PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);
}
#endif

static const int LOOPS_INFINITE = -1;
static int s_numLoops = LOOPS_INFINITE;

LWNSampleTestConfig testConfig;
LWNSampleTestCInterface testCInterface;
LWNSampleTestCPPInterface testCPPInterface;

LWNSampleTestCInterface * LWNSampleTestConfig::m_c_interface = &testCInterface;
LWNSampleTestCPPInterface * LWNSampleTestConfig::m_cpp_interface = &testCPPInterface;

MemoryPoolAllocator *g_texAllocator = NULL;
MemoryPoolAllocator *g_bufferAllocator = NULL;

// Include the implementation of the memory pool allocator utilities.
#include "lwnUtil/lwnUtil_AlignedStorageImpl.h"
#include "lwnUtil/lwnUtil_PoolAllocatorImpl.h"
#include "lwnUtil/lwnUtil_GlslcHelperImpl.h"

// Provide functions/methods to access the C-only code to retrieve the registered IDs
// we saved with textures and samplers.
int lwnGetRegisteredTextureID(lwn::Texture *texture)
{
    return lwnTextureGetRegisteredTextureID(reinterpret_cast<LWNtexture *>(texture));
}

int lwnGetRegisteredSamplerID(lwn::Sampler *sampler)
{
    return lwnSamplerGetRegisteredID(reinterpret_cast<LWNsampler *>(sampler));
}

void deinit(void)
{
    if (testConfig.m_cpp) {
        testConfig.cppDeleteWindow();
    }
    else {
        testConfig.cDeleteWindow();
    }
}

void display(void)
{
    if (s_numLoops != LOOPS_INFINITE) {
        if (s_numLoops == 0) {
            deinit();
            exit(0);
        }
        --s_numLoops;
    }

    if (testConfig.m_cpp) {
        testConfig.cppDisplay();
    }
    else {
        testConfig.cDisplay();
    }
    lwplatform_swapBuffers();
}

void glutKey( unsigned char key, int x, int y )
{
    switch (key) {
    case '\033':
        deinit();
        exit(0);
    case 'b':
        if (!testConfig.m_benchmark) {
            log_output("Sampling in benchmark mode:\n");
            testConfig.m_benchmark = true;
            display();
            testConfig.m_benchmark = false;
            // continue to re-display normal images
        }
        break;
    case 's':
        switch (testConfig.m_submitMode) {
        case LWNSampleTestConfig::QUEUE:
            testConfig.m_submitMode = LWNSampleTestConfig::COMMAND;
            log_output("Entering command buffer mode.\n");
            break;
        case LWNSampleTestConfig::COMMAND:
            testConfig.m_submitMode = LWNSampleTestConfig::COMMAND_TRANSIENT;
            log_output("Entering transient command buffer mode.\n");
            break;
        case LWNSampleTestConfig::COMMAND_TRANSIENT:
            testConfig.m_submitMode = LWNSampleTestConfig::QUEUE;
            log_output("Entering queue command buffer mode.\n");
            break;
        }
        break;
    case 'm':
        testConfig.m_multisample = !testConfig.m_multisample;
        log_output("%s multisample mode.\n", testConfig.m_multisample ? "Entering" : "Leaving");
        break;
    case 'g':
        testConfig.m_geometryShader = !testConfig.m_geometryShader;
        log_output("%s geometry shader mode.\n", testConfig.m_geometryShader ? "Entering" : "Leaving");
        break;
    case 't':
        // Cycle from no tessellation -> tessellation (TES only) -> tessellation (TCS + TES).
        if (testConfig.m_tessControlShader) {
            testConfig.m_tessControlShader = false;
            testConfig.m_tessEvalShader = false;
            log_output("Leaving tessellation mode.\n");
        } else {
            testConfig.m_tessControlShader = testConfig.m_tessEvalShader;
            testConfig.m_tessEvalShader = true;
            log_output("Entering tessellation mode (%s control shaders).\n", testConfig.m_tessControlShader ? "with" : "without");
        }
        break;
    case 'w':
        testConfig.m_wireframe = !testConfig.m_wireframe;
        log_output("%s wireframe mode.\n", testConfig.m_wireframe ? "Entering" : "Leaving");
        break;
    case 'B':
        testConfig.m_bindless = !testConfig.m_bindless;
        log_output("%s bindless texture mode.\n", testConfig.m_bindless ? "Entering" : "Leaving");
        break;
    case 'c':
        testConfig.m_cpp = !testConfig.m_cpp;
        log_output("Switching to using the %s API bindings.\n", testConfig.m_cpp ? "C++" : "C");
        break;
    case '?':
        log_output("ESCAPE:  Quit\n");
        log_output("b:  One-time benchmark on the current state\n");
        log_output("s:  Toggle command buffer submission mode (queue, command buffer, transient)\n");
        log_output("m:  Toggle multisample mode\n");
        log_output("g:  Toggle geometry shader mode\n");
        log_output("t:  Toggle tessellation shader mode\n");
        log_output("w:  Toggle wireframe mode\n");
        log_output("B:  Toggle bindless texture mode\n");
        log_output("c:  Switch between C and C++ bindings\n");
        break;
    }

}

#ifndef LW_HOS
int main(int argc, char* argv[])
#else
int appmain(int argc, char* argv[])
#endif
{
    const int windowWidth  = 500;
    const int windowHeight = 500;

    lwplatform_setupWindow(windowWidth, windowHeight);

    lwplatform_displayFunc(display);
    lwplatform_keyboardFunc(glutKey);

    // Parse command line options.
    for (int i = 1; i < argc; i++) {
        if (0 == strcmp(argv[i], "-benchmark")) {
            testConfig.m_benchmark = true;
        } else if (0 == strcmp(argv[i], "-queue")) {
            testConfig.m_submitMode = LWNSampleTestConfig::QUEUE;
        } else if (0 == strcmp(argv[i], "-cmd")) {
            testConfig.m_submitMode = LWNSampleTestConfig::COMMAND;
        } else if (0 == strcmp(argv[i], "-cmdt")) {
            testConfig.m_submitMode = LWNSampleTestConfig::COMMAND_TRANSIENT;
        } else if (0 == strcmp(argv[i], "-ms")) {
            testConfig.m_multisample = true;
        } else if (0 == strcmp(argv[i], "-usegs")) {
            testConfig.m_geometryShader = true;
        } else if (0 == strcmp(argv[i], "-usetcs")) {
            testConfig.m_tessControlShader = true;
        } else if (0 == strcmp(argv[i], "-usetes")) {
            testConfig.m_tessEvalShader = true;
        } else if (0 == strcmp(argv[i], "-wireframe")) {
            testConfig.m_wireframe = true;
        } else if (0 == strcmp(argv[i], "-bindless")) {
            testConfig.m_bindless = true;
        } else if (0 == strcmp(argv[i], "-debug")) {
            testConfig.m_debug = true;
        } else if (0 == strcmp(argv[i], "-cpp")) {
            testConfig.m_cpp = true;
        } else if (0 == strcmp(argv[i], "-c")) {
            testConfig.m_cpp = false;
        } else if (0 == strcmp(argv[i], "-n") && (i + 1) < argc) {
            s_numLoops = atol(argv[++i]);
        } else {
            fprintf(stderr, "%s:  unknown option (%s)\n", argv[0], argv[i]);
            return 1;
        }
    }
    if (testConfig.m_tessControlShader && !testConfig.m_tessEvalShader) {
        fprintf(stderr, "%s:  can't use '-usetcs' without '-usetes'\n", argv[0]);
        return 1;
    }

#if defined(_WIN32)
    PFNLWNBOOTSTRAPLOADERPROC lwnBootstrapLoader = (PFNLWNBOOTSTRAPLOADERPROC)BOOTSTRAP_FUNC("rq34nd2ffz");
#endif
    PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC)((*lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
    if (!getProcAddress) {
        return 1;
    }

    lwnLoadCProcs(NULL, getProcAddress);

    LWNdeviceBuilder deviceBuilder;
    lwnDeviceBuilderSetDefaults(&deviceBuilder);
    if (testConfig.m_debug) {
        lwnDeviceBuilderSetFlags(&deviceBuilder, (LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT |
                                                  LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT));
    } else {
        lwnDeviceBuilderSetFlags(&deviceBuilder, 0);
    }

    LWNdevice *device = new LWNdevice;
    if (!lwnDeviceInitialize(device, &deviceBuilder)) {
        fprintf(stderr, "Couldn't initialize the LWN device.\n");
        return 1;
    }

    // Now load the rest of the function pointer interface.
    lwnLoadCProcs(device, getProcAddress);
    lwnLoadCPPProcs(reinterpret_cast<lwn::objects::Device *>(device),
                    reinterpret_cast<lwn::objects::DeviceGetProcAddressFunc>(getProcAddress));


    // Check for API version mismatches.  Exit with an error if the major
    // version mismatches (major revisions are backward-incompatible) or if
    // the driver reports a lower minor version.
    int majorVersion, minorVersion;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_API_MAJOR_VERSION, &majorVersion);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_API_MINOR_VERSION, &minorVersion);
    if (majorVersion != LWN_API_MAJOR_VERSION || minorVersion < LWN_API_MINOR_VERSION) {
        fprintf(stderr, "%s:  API version mismatch (application compiled with %d.%d, "
                "driver reports %d.%d).\n",
                argv[0], LWN_API_MAJOR_VERSION, LWN_API_MINOR_VERSION,
                majorVersion, minorVersion);
        return 1;
    }

    g_texAllocator = new MemoryPoolAllocator(device, NULL, 32*1024*1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    g_bufferAllocator = new MemoryPoolAllocator(device, NULL, 32*1024*1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Create the "global" queue and command buffer used by the test.
    LWNqueue *queue = lwnDeviceCreateQueue(device);

    // Initialize a queue completion tracker that can track up to 31
    // outstanding fences.
    CompletionTracker *lwnTracker = initCompletionTracker(device, 32);

    // Initialize a QueueCommandBuffer object to submit commands to the queue.
    // We allocate a bigger-than-default buffer for benchmark mode, where we
    // have a huge number of primitives.
    QueueCommandBuffer *queueCB = new QueueCommandBuffer;
#if 0
    queueCB->setCommandPoolAllocSize(32 * 1024 * 1024);
#endif
    if (!queueCB->init(device, queue, lwnTracker)) {
        fprintf(stderr, "Couldn't initialize queue command buffer.\n");
        return 1;
    }

    // Initialize command buffer memory manager to provide memory for API
    // command buffer usage.  We allocate a bigger-than-default buffer for
    // benchmark mode, where we have a huge number of primitives.  We also set
    // smaller-than-default granularities because benchmark mode uses tiny
    // command sets.
    LWNcommandBufferMemoryManager *commandMem = new LWNcommandBufferMemoryManager;
#if 0
    commandMem->setCoherentPoolSize(32 * 1024 * 1024);
    commandMem->setControlPoolSize(4 * 1024 * 1024);
    commandMem->setCoherentChunkSize(1024);
    commandMem->setControlChunkSize(64);
#endif
    if (!commandMem->init(device, lwnTracker)) {
        fprintf(stderr, "Couldn't initialize command buffer memory manager.\n");
        return 1;
    }

    // Initialize the texture ID pool manager.
    g_lwn.m_texIDPool = new LWNsystemTexIDPool(device, queueCB);

    // Initialize the GLSLC library helper.
    lwnUtil::GLSLCLibraryHelper * glslcLibraryHelper = new lwnUtil::GLSLCLibraryHelper;

    if (!glslcLibraryHelper) {
        fprintf(stderr, "GLSLC library helper allocation failed.\n");
        return 1;
    }

    // Load the symbols passing in "NULL" which means to use a statically-linked
    // stub library or interface to reach the GLSLC library exported functions.
    glslcLibraryHelper->LoadDLL( NULL );

    if (!glslcLibraryHelper->IsLoaded()) {
        fprintf(stderr, "GLSLC library failed to initialize.\n");
        return 1;
    }

    // Set up the C and C++ interfaces for the LWN globals.
    testCInterface.device = device;
    testCInterface.queue = queue;
    testCInterface.queueCB = queueCB;
    testCInterface.cmdMemMgr = commandMem;
    testCInterface.completionTracker = lwnTracker;
    testCInterface.glslcLibraryHelper = glslcLibraryHelper;
    testCPPInterface.device = reinterpret_cast<lwn::Device *>(device);
    testCPPInterface.queue = reinterpret_cast<lwn::Queue *>(queue);
    testCPPInterface.queueCB = reinterpret_cast<lwn::CommandBuffer *>(queueCB);
    testCPPInterface.cmdMemMgr = commandMem;
    testCPPInterface.completionTracker = lwnTracker;
    testCPPInterface.glslcLibraryHelper = glslcLibraryHelper;

    if (testConfig.m_cpp) {
        testCPPInterface.window = testConfig.cppCreateWindow(lwplatform_getWindowHandle(), windowWidth, windowHeight);
        testCInterface.window = reinterpret_cast<LWNBasicWindowC*>(testCPPInterface.window);
    }
    else {
        testCInterface.window = testConfig.cCreateWindow(lwplatform_getWindowHandle(), windowWidth, windowHeight);
        testCPPInterface.window = reinterpret_cast<LWNBasicWindowCPP*>(testCInterface.window);
    }

    if (testCPPInterface.window) {
        lwplatform_mainLoop();
    }

    deinit();

    return 0;
}

#if defined(LW_HOS)
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

extern "C" void nninitStartup()
{
    const size_t MallocMemorySize = 256 * 1024 * 1024;
    nn::Result result = nn::os::SetMemoryHeapSize(512 * 1024 * 1024);
    uintptr_t address;
    result = nn::os::AllocateMemoryBlock(&address, MallocMemorySize);
    NN_ASSERT(result.IsSuccess());
    nn::init::InitializeAllocator(reinterpret_cast<void*>(address), MallocMemorySize);

    // Set file system allocator and deallocator
    FsInitHeap();
    nn::fs::SetAllocator(FsAllocate, FsDeallocate);

    new(&nn::util::Get(g_TlsAllocator)) nn::mem::StandardAllocator(g_TlsHeapBuffer, sizeof(g_TlsHeapBuffer));
    nn::os::SetMemoryAllocatorForThreadLocal(TlsAlloc, TlsDealloc);
}

//===========================================================================
// nnMain() - entry point
//===========================================================================
extern "C" void nnMain()
{

    // Bugs 1674275, 1663555: Set rounding mode.
    fesetround(FE_TONEAREST);

    lw::SetGraphicsAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);
    lw::InitializeGraphics(g_GraphicsFirmwareMemory, sizeof(g_GraphicsFirmwareMemory));

    glslcSetAllocator(GraphicsAllocate, GraphicsFree, GraphicsReallocate, NULL);

    if (appmain(nn::os::GetHostArgc(), nn::os::GetHostArgv())) {
        NN_SDK_LOG("LwAppMain Failed\n");
        return;
    }

}
#endif
