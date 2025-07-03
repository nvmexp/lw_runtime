/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/
////===========================================================================
///  demoGfx.cpp
///
///     This is graphics system code for the DEMO library.
///
////===========================================================================

#define _CRT_SELWRE_NO_WARNINGS         // inhibit MSVC warnings about deprecated sprintf

#include <demo.h>
#include <trace.h>

#if !defined(WIN_INTERFACE_LWSTOM)
#define WIN_INTERFACE_LWSTOM
#endif

#include "lwn/lwn_FuncPtrImpl.h"        // Code to set up LWN C function pointer interface

#ifdef _WIN32
#define snprintf _snprintf              // MSVC doesn't support raw "snprintf"
#endif

#include "lwnUtil/lwnUtil_GlslcHelper.h"

static LWNdevice                    *s_device;
static lwnUtil::GLSLCHelper         *s_glslcHelper = NULL;
static lwnUtil::GLSLCLibraryHelper  *s_glslcLibraryHelper = NULL;

static LWNqueue	            *s_queue;
static QueueCommandBuffer   *s_queueCB;
static LWNcommandBuffer     *s_cmd;
static DEMOGfxRenderTarget  s_renderTarget;
static LWNwindow            *s_window;
static LWNsync              s_textureAvailableSync;

static DEMOGfxContextState  s_defaultContextState;
static DEMOGfxContextState  *s_pContextState;

static LWNtextureBuilder    *s_textureBuilder;
static LWNsamplerBuilder    *s_samplerBuilder;
static LWNbufferBuilder     *s_bufferBuilder;

static CompletionTracker                *s_completionTracker;
static LWNcommandBufferMemoryManager    *s_commandMem;

static MemoryPoolAllocator* s_texturePool;
static MemoryPoolAllocator* s_bufferPool;

static BOOL s_demoGfxIncludeCPUPerf = TRUE;
static BOOL s_demoGfxUseCommandBuffer = TRUE;
static BOOL s_demoGfxCommandBufferTransient = TRUE;

u32 DEMOGfxOffscreenWidth = 1280, DEMOGfxOffscreenHeight = 720;

#ifdef ENABLE_PERF_WIN
static LARGE_INTEGER perfStart, perfEnd;
#endif

#ifdef ENABLE_PERF

static u64 frameStart;

static DEMOGfxGPUTimeStamp s_demoGfxGPUStartTime;
static DEMOGfxGPUTimeStamp s_demoGfxGPUPreSubmitTime;
static DEMOGfxGPUTimeStamp s_demoGfxGPUEndTime;

static double s_demoGfxCreateBufferTimeTotal = 0;
static double s_demoGfxCompileBufferTimeTotal = 0;
static double s_demoGfxSubmitBufferTimeTotal = 0;
static double s_demoGfxResetTimeTotal = 0;
static double s_demoGfxCPUFrameTimeTotal = 0;
static double s_demoGfxGPUSubmitTimeTotal = 0;
static double s_demoGfxGPUFrameTimeTotal = 0;
static double s_demoGfxPerfTimeTotal = 0;

static f32 s_demoGfxFrameTimeTotal = 0;

static u32 s_callTimes = 0;

#ifdef _WIN32

u64 getTime() {
    return (u64)clock()*1000000;
}
#endif

#if defined(LW_LINUX)

static int64_t getTime() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec*1000000000LL + now.tv_nsec;
}
#endif
#endif /* ENABLE_PERF */

#if !defined(_WIN32)
extern "C"
{
    PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);
}
#endif

void DEMOGfxInit(int argc, char **argv, LWNnativeWindow nativeWindow, int numPresentTextures, int presentInterval)
{
    // Initialize the LWN function pointer interface.
    PFNLWNBOOTSTRAPLOADERPROC bootstrapLoader = NULL;
    PFNLWNDEVICEINITIALIZEPROC devInit = NULL;
    PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress = NULL;

#ifdef _WIN32
    bootstrapLoader = (PFNLWNBOOTSTRAPLOADERPROC) wglGetProcAddress("rq34nd2ffz");
#else
    bootstrapLoader = (PFNLWNBOOTSTRAPLOADERPROC)lwnBootstrapLoader;
#endif

    if (bootstrapLoader) {
        devInit = (PFNLWNDEVICEINITIALIZEPROC) ((*bootstrapLoader)("lwnDeviceInitialize"));
        getProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC) ((*bootstrapLoader)("lwnDeviceGetProcAddress"));
    }
    if (!bootstrapLoader || !devInit || !getProcAddress) {
        fprintf(stderr, "Couldn't initialize the LWN bootstrap loader (possible version mismatch).\n");
        return;
    }

    // Now load an initial set of function pointers.
    lwnLoadCProcs(NULL, getProcAddress);

    trace_init();

#ifdef ENABLE_PERF
    s_demoGfxCreateBufferTimeTotal = 0;
    s_demoGfxCompileBufferTimeTotal = 0;
    s_demoGfxSubmitBufferTimeTotal = 0;
    s_demoGfxResetTimeTotal = 0;
    s_demoGfxCPUFrameTimeTotal = 0;
    s_demoGfxPerfTimeTotal = 0;

    s_demoGfxFrameTimeTotal = 0;
    s_demoGfxGPUSubmitTimeTotal = 0;
    s_demoGfxGPUFrameTimeTotal = 0;

    s_callTimes = 0;
#endif

#define SKIP_NON_DIGIT(c) ((c)!=0&&((c)<'0'||(c)>'9'))

    // Analyze arguments
    // Note that all arguments might be in a single string!
    for (int i = 0; i < argc; ++i)
    {
        char* p;
        
        p = strstr(argv[i], "DEMO_INCLUDE_CPU_PERF");
        if (p != 0)
            s_demoGfxIncludeCPUPerf = TRUE;

        p = strstr(argv[i], "DEMO_IGNORE_CPU_PERF");
        if (p != 0)
            s_demoGfxIncludeCPUPerf = FALSE;

        p = strstr(argv[i], "DEMO_USE_COMMAND_BUFFER_TRANSIENT");
        if (p != 0)
        {
            s_demoGfxUseCommandBuffer = TRUE;
            s_demoGfxCommandBufferTransient = TRUE;
        }

        p = strstr(argv[i], "DEMO_USE_COMMAND_BUFFER");
        if (p != 0)
        {
            s_demoGfxUseCommandBuffer = TRUE;
            s_demoGfxCommandBufferTransient = FALSE;
        }

        p = strstr(argv[i], "DEMO_USE_QUEUE");
        if (p != 0)
            s_demoGfxUseCommandBuffer = FALSE;
    }
    
    char tempStr[1000] = "";
    int off = 0;

    if (s_demoGfxIncludeCPUPerf)
        off += sprintf(&tempStr[off], "INCLUDE_CPU_PERF  ");
    else
        off += sprintf(&tempStr[off], "IGNORE_CPU_PERF  ");

    if (s_demoGfxUseCommandBuffer && s_demoGfxCommandBufferTransient)
        off += sprintf(&tempStr[off], "USE_COMMAND_BUFFER_TRANSIENT  ");
    else if (s_demoGfxUseCommandBuffer)
        off += sprintf(&tempStr[off], "USE_COMMAND_BUFFER  ");
    else
        off += sprintf(&tempStr[off], "USE_QUEUE  ");
    
    //DEMOPrintf("DEMO: %s\n", tempStr);

    LWNdeviceBuilder deviceBuilder;
    lwnDeviceBuilderSetDefaults(&deviceBuilder);
    lwnDeviceBuilderSetFlags(&deviceBuilder, (LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT |
                                              LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT));

    s_device = new LWNdevice;
    if (!(*devInit)(s_device, &deviceBuilder)) {
        DEMOPrintf("Couldn't initialize the LWN device.\n");
        return;
    }

    // Now load the rest of the function pointer interface.
    lwnLoadCProcs(s_device, getProcAddress);

    // More initialization
    DEMOGfxCreateDevice(nativeWindow, numPresentTextures, presentInterval);

    DEMOGfxInitContextState(&s_defaultContextState);

#ifdef ENABLE_PERF
    s_demoGfxGPUStartTime.Init();
    s_demoGfxGPUPreSubmitTime.Init();
    s_demoGfxGPUEndTime.Init();
#endif

    // Initialize the GLSLC bits.
    s_glslcLibraryHelper = new lwnUtil::GLSLCLibraryHelper();
    if (!s_glslcLibraryHelper) {
        DEMOPrintf("Could not allocate memory for GLSLCLibraryHelper.\n");
        return;
    }

    // Loads the entry points
    s_glslcLibraryHelper->LoadDLL(NULL);
    if (!s_glslcLibraryHelper->IsLoaded()) {
        DEMOPrintf("GLSLC library is not initialized.  Running with online compiler.\n");
    } else {
        DEMOPrintf("GLSLC library initialized.  Running with offline compiler.\n");
    }

    s_glslcHelper = new lwnUtil::GLSLCHelper(s_device, 0x100000UL, s_glslcLibraryHelper);
}

void DEMOGfxShutdown(void)
{
    DEMOGfxReleaseDevice();

    delete s_glslcHelper;
    delete s_glslcLibraryHelper;

    lwnDeviceFinalize(s_device);
    delete s_device;
}

void DEMOGfxBeforeRender(void)
{
#ifdef ENABLE_PERF
    frameStart = getTime();
    s_demoGfxGPUStartTime.ReportCounter();
#endif

    lwnWindowAcquireTexture(s_window, &s_textureAvailableSync, &s_renderTarget.lwrrentIndex);
    lwnQueueWaitSync(s_queue, &s_textureAvailableSync);

    lwnCommandBufferSetRenderTargets(s_queueCB, 1, &s_renderTarget.colors[s_renderTarget.lwrrentIndex],
                                     NULL, s_renderTarget.depth, NULL);

    s_queueCB->submit();
}

void DEMOGfxPresentRender(void)
{
    if(s_queue)
    {
        lwnQueuePresentTexture(s_queue, s_window, s_renderTarget.lwrrentIndex);
        s_renderTarget.lwrrentIndex = -1;
    }
}

void DEMOGfxSetIncludeCPUPerf(bool value)
{
    s_demoGfxIncludeCPUPerf = value;
}

void DEMOGfxSetUseCommandBuffer(bool value)
{
    s_demoGfxUseCommandBuffer = value;
}

void DEMOGfxSetCommandBufferTransient(bool value)
{
    s_demoGfxCommandBufferTransient = value;
}

void DEMOGfxResetLWNPerformace(void)
{
#ifdef ENABLE_PERF
    s_demoGfxCreateBufferTimeTotal = 0;
    s_demoGfxCompileBufferTimeTotal = 0;
    s_demoGfxSubmitBufferTimeTotal = 0;
    s_demoGfxResetTimeTotal = 0;
    s_demoGfxCPUFrameTimeTotal = 0;
    s_demoGfxPerfTimeTotal = 0;

    s_demoGfxFrameTimeTotal = 0;
    s_demoGfxGPUSubmitTimeTotal = 0;
    s_demoGfxGPUFrameTimeTotal = 0;

    s_callTimes = 0;
#endif
}

void DEMOGfxPrintLWNPerformace(void)
{
#ifdef ENABLE_PERF_WIN
    QueryPerformanceCounter(&perfEnd);

    if(perfStart.QuadPart > 0)
    {
        s_demoGfxPerfTimeTotal += (perfEnd.QuadPart - perfStart.QuadPart);
        lwnPrintPerf(s_demoGfxPerfTimeTotal);
    }

    QueryPerformanceCounter(&perfStart);
#endif

#ifdef ENABLE_PERF
    if(s_callTimes > 0)
    {
        char tempStr[1000] = "";
        char tempStr2[1000] = "";
        int off = 0;
        int off2 = 0;

        if (s_demoGfxIncludeCPUPerf)
        {
            off += snprintf(&tempStr[off], 1000-off, "%8f  %8f  %8f  %8f  %8f  ",
                s_demoGfxCreateBufferTimeTotal/s_callTimes,
                s_demoGfxCompileBufferTimeTotal/s_callTimes,
                s_demoGfxSubmitBufferTimeTotal/s_callTimes,
                s_demoGfxResetTimeTotal/s_callTimes,
                s_demoGfxCPUFrameTimeTotal/s_callTimes);
            off2 += snprintf(&tempStr2[off2], 1000-off2, "Create    Compile   Submit    Reset     CPU Time  ");
        }
        //DEMOPrintf("REPORT : %sGPU Submit  GPU Time  Total Time\n", tempStr2);
        off += snprintf(&tempStr[off], 1000-off, "%8f    %8f  %8f", s_demoGfxGPUSubmitTimeTotal/s_callTimes, s_demoGfxGPUFrameTimeTotal/s_callTimes, s_demoGfxFrameTimeTotal/s_callTimes);
        DEMOPrintf("(msec)   %s\n", tempStr);
    }
#endif
}

void DEMOGfxGPUTimeStamp::Init(void)
{
    LWNbufferBuilder *builder = lwnDeviceCreateBufferBuilder(s_device);
    this->buffer = s_bufferPool->allocBuffer(builder, BUFFER_ALIGN_COUNTER_BIT, 16);
    this->address = lwnBufferGetAddress(this->buffer);
    this->value = ((uint64_t *)lwnBufferMap(buffer)) + 1;
    lwnBufferBuilderFree(builder);
}

void DEMOGfxGPUTimeStamp::ReportCounter(void)
{
#ifdef ENABLE_REPORT_COUNTER
    lwnCommandBufferReportCounter(s_queueCB, LWN_COUNTER_TYPE_TIMESTAMP, this->address);
#endif
}

uint64_t DEMOGfxGPUTimeStamp::GetTimeStamp(void)
{
    return *this->value;
}

void DEMOGfxDoneRender(void)
{
#ifdef ENABLE_PERF
    u64 postCreate, postCompile, postSubmit, frameEnd;
#endif

    if (s_demoGfxUseCommandBuffer)
    {
#ifdef ENABLE_PERF
        postCreate = postCompile = getTime();
        s_demoGfxGPUPreSubmitTime.ReportCounter();
#endif
        LWNcommandHandle handle = lwnCommandBufferEndRecording(s_cmd);
        lwnQueueSubmitCommands(s_queue, 1, &handle);
#ifdef ENABLE_PERF
        postSubmit = getTime();
#endif
        lwnCommandBufferFree(s_cmd);
        s_cmd = lwnDeviceCreateCommandBuffer(s_device);
        s_commandMem->populateCommandBuffer(s_cmd, LWNcommandBufferMemoryManager::Coherent);
        lwnCommandBufferBeginRecording(s_cmd);
#ifdef ENABLE_PERF
        frameEnd = getTime();
#endif
    }
#ifdef ENABLE_PERF
    else
    {
        s_queueCB->submit();

        s_demoGfxGPUPreSubmitTime.ReportCounter();

        postCreate = getTime();
        postCompile = postCreate;
        postSubmit = postCreate;
        frameEnd = postCreate;
    }
    
    if (s_demoGfxIncludeCPUPerf)
    {
        
        s_demoGfxCreateBufferTimeTotal += (f32)(postCreate - frameStart) / 1000000; // colwert to msec
        s_demoGfxCompileBufferTimeTotal += (f32)(postCompile - postCreate) / 1000000; // colwert to msec
        s_demoGfxSubmitBufferTimeTotal += (f32)(postSubmit - postCompile) / 1000000; // colwert to msec
        s_demoGfxResetTimeTotal += (f32)(frameEnd - postSubmit) / 1000000; // colwert to msec
        s_demoGfxCPUFrameTimeTotal += (f32)(frameEnd - frameStart) / 1000000; // colwert to msec
    }

    s_demoGfxGPUEndTime.ReportCounter();
    
#endif
    lwnQueueFinish(s_queue);

#ifdef ENABLE_PERF
    frameEnd = getTime();
    
    s_demoGfxGPUSubmitTimeTotal += (f32)(s_demoGfxGPUEndTime.GetTimeStamp() - s_demoGfxGPUPreSubmitTime.GetTimeStamp()) / 1000000; // colwert to msec
    s_demoGfxGPUFrameTimeTotal += (f32)(s_demoGfxGPUEndTime.GetTimeStamp() - s_demoGfxGPUStartTime.GetTimeStamp()) / 1000000; // colwert to msec
    s_demoGfxFrameTimeTotal += (f32)(frameEnd - frameStart) / 1000000; // colwert to msec
    s_callTimes++;
#endif

    if(s_queue)
    {
        insertCompletionTrackerFence(s_completionTracker, s_queue);
        lwnQueuePresentTexture(s_queue, s_window, s_renderTarget.lwrrentIndex);
        s_renderTarget.lwrrentIndex = -1;
    }
}

void DEMOGfxSetContextState(DEMOGfxContextState *pContextState)
{
    s_pContextState = pContextState;
    
    lwnCommandBufferBindDepthStencilState(s_cmd, &pContextState->depthState);
    lwnCommandBufferBindColorState(s_cmd, &pContextState->colorState);
    lwnCommandBufferBindBlendState(s_cmd, &pContextState->blendState);
    lwnCommandBufferBindPolygonState(s_cmd, &pContextState->polygonState);
    lwnCommandBufferBindChannelMaskState(s_cmd, &pContextState->channelMasks);

    lwnCommandBufferSetStencilRef(s_cmd, LWN_FACE_FRONT, pContextState->stencilCtrl.frontRef);
    lwnCommandBufferSetStencilValueMask(s_cmd, LWN_FACE_FRONT, pContextState->stencilCtrl.frontTestMask);
    lwnCommandBufferSetStencilMask(s_cmd, LWN_FACE_FRONT, pContextState->stencilCtrl.frontWriteMask);
    lwnCommandBufferSetStencilRef(s_cmd, LWN_FACE_BACK, pContextState->stencilCtrl.backRef);
    lwnCommandBufferSetStencilValueMask(s_cmd, LWN_FACE_BACK, pContextState->stencilCtrl.backTestMask);
    lwnCommandBufferSetStencilMask(s_cmd, LWN_FACE_BACK, pContextState->stencilCtrl.backWriteMask);

    if (pContextState->shader.pgm)
    {
        lwnCommandBufferBindProgram(s_cmd, pContextState->shader.pgm, pContextState->shader.bits);
        lwnCommandBufferBindVertexAttribState(s_cmd, pContextState->shader.nAttribs, pContextState->shader.attribs);
        lwnCommandBufferBindVertexStreamState(s_cmd, pContextState->shader.nStreams, pContextState->shader.streams);
    }
}

DEMOGfxContextState* DEMOGfxGetContextState(void)
{
    return s_pContextState;
}

void DEMOGfxInitContextState(DEMOGfxContextState *pContextState)
{
    s_pContextState = pContextState;

    DEMOGfxSetStatesDefault();
}


static void LWNAPIENTRY debugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                          LWNdebugCallbackSeverity severity, const char *message, void* userParam)
{
    DEMOPrintf("%s\n", message);
}


void DEMOGfxCreateDevice(LWNnativeWindow nativeWindow, int numPresentTextures, int presentInterval)
{
    lwnDeviceInstallDebugCallback(s_device, debugCallback, NULL, true);

    s_textureBuilder = lwnDeviceCreateTextureBuilder(s_device);
    s_samplerBuilder = lwnDeviceCreateSamplerBuilder(s_device);
    s_bufferBuilder = lwnDeviceCreateBufferBuilder(s_device);

    s_queue  = lwnDeviceCreateQueue(s_device);

    // Initialize a queue completion tracker that can track up to 31
    // outstanding fences.
    s_completionTracker = initCompletionTracker(s_device, 32);

    // Initialize a QueueCommandBuffer object to submit commands to the queue.
    s_queueCB = new QueueCommandBuffer;
    if (!s_queueCB->init(s_device, s_queue, s_completionTracker)) {
        fprintf(stderr, "Couldn't initialize queue command buffer.\n");
        return;
    }

    // Initialize command buffer memory manager to provide memory for API
    // command buffer usage.
    s_commandMem = new LWNcommandBufferMemoryManager;
    if (!s_commandMem->init(s_device, s_completionTracker)) {
        fprintf(stderr, "Couldn't initialize command buffer memory manager.\n");
        return;
    }

    //lwnQueueSetRenderEnable(s_queue, FALSE);

    if (s_demoGfxUseCommandBuffer) {
        s_cmd = lwnDeviceCreateCommandBuffer(s_device);
        s_commandMem->populateCommandBuffer(s_cmd, LWNcommandBufferMemoryManager::Coherent);
        lwnCommandBufferBeginRecording(s_cmd);
    } else {
        s_cmd = s_queueCB;
    }

    // Initialize the texture ID pool manager.
    g_lwn.m_texIDPool = new LWNsystemTexIDPool(s_device, s_cmd);

    // Setup global memory pools
    s_texturePool = new MemoryPoolAllocator(s_device, NULL, 0, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    s_bufferPool  = new MemoryPoolAllocator(s_device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    if (!s_texturePool || !s_bufferPool) {
        fprintf(stderr, "Couldn't initialize global memory pools.\n");
        return;
    }

    // Create a color buffer and depth buffer
    lwnTextureBuilderSetDefaults(s_textureBuilder);
    lwnTextureBuilderSetFlags(s_textureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT | LWN_TEXTURE_FLAGS_DISPLAY_BIT);
    lwnTextureBuilderSetSize2D(s_textureBuilder, DEMOGfxOffscreenWidth, DEMOGfxOffscreenHeight);
    lwnTextureBuilderSetTarget(s_textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(s_textureBuilder, LWN_FORMAT_RGBA8);

    s_renderTarget.colors = new LWNtexture*[numPresentTextures];
    s_renderTarget.numColorTextures = numPresentTextures;

    for (int i = 0; i < s_renderTarget.numColorTextures; i++) {
        s_renderTarget.colors[i] = s_texturePool->allocTexture(s_textureBuilder);
    }

    lwnTextureBuilderSetFlags(s_textureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    lwnTextureBuilderSetFormat(s_textureBuilder, LWN_FORMAT_DEPTH24_STENCIL8);
    s_renderTarget.depth = s_texturePool->allocTexture(s_textureBuilder);

    lwnCommandBufferSetScissor(s_queueCB, 0, 0, DEMOGfxOffscreenWidth, DEMOGfxOffscreenHeight);
    lwnCommandBufferSetViewport(s_queueCB, 0, 0, DEMOGfxOffscreenWidth, DEMOGfxOffscreenHeight);

    // Create a window from color textures, connect to native window.
    LWNwindowBuilder windowBuilder;
    lwnWindowBuilderSetDevice(&windowBuilder, s_device);
    lwnWindowBuilderSetDefaults(&windowBuilder);
    lwnWindowBuilderSetTextures(&windowBuilder, numPresentTextures, s_renderTarget.colors);
    lwnWindowBuilderSetNativeWindow(&windowBuilder, nativeWindow);
    lwnWindowBuilderSetPresentInterval(&windowBuilder, presentInterval);

    s_window = new LWNwindow;
    lwnWindowInitialize(s_window, &windowBuilder);
    lwnSyncInitialize(&s_textureAvailableSync, s_device);
}

void DEMOGfxReleaseDevice(void)
{
    // Queue render targets
    lwnCommandBufferSetRenderTargets(s_queueCB, 0, NULL, NULL, NULL, NULL);

    lwnWindowFinalize(s_window);
    delete s_window;
    lwnSyncFinalize(&s_textureAvailableSync);

    for (int i = 0; i < s_renderTarget.numColorTextures; i++) {
        s_texturePool->freeTexture(s_renderTarget.colors[i]);
    }
    s_texturePool->freeTexture(s_renderTarget.depth);

    delete [] s_renderTarget.colors;
    s_renderTarget.colors = NULL;
    s_renderTarget.numColorTextures = 0;

    if (s_demoGfxUseCommandBuffer)
        lwnCommandBufferFree(s_cmd);

    s_queueCB->destroy();
    delete s_queueCB;

    s_commandMem->destroy();
    delete s_commandMem;

    lwnQueueFree(s_queue);

    delete s_texturePool;
    delete s_bufferPool;

    delete g_lwn.m_texIDPool;
    g_lwn.m_texIDPool = NULL;

    lwnTextureBuilderFree(s_textureBuilder);
    lwnSamplerBuilderFree(s_samplerBuilder);
    lwnBufferBuilderFree(s_bufferBuilder);
}


// shader

void DEMOGfxCreateShaders(DEMOGfxShader* pShader, const char* vertexShader, const char* pixelShader)
{
    pShader->pgm            = lwnDeviceCreateProgram(s_device);

    // XXX This is a hack because we don't have an IL. I'm just jamming through the strings 
    // as if they were an IL blob
    LWNshaderStage stages[3];
    const char *sources[3] = {vertexShader, pixelShader};
    s32 nSources = 2;
    stages[0] = LWN_SHADER_STAGE_VERTEX;
    stages[1] = LWN_SHADER_STAGE_FRAGMENT;
    if (!s_glslcHelper->CompileAndSetShaders(pShader->pgm, stages, nSources, sources))
    {
        printf("Shader compile error. infoLog =\n%s\n", s_glslcHelper->GetInfoLog());
    }

    pShader->bits = LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT;
    pShader->nAttribs = 0;
    pShader->nStreams = 0;
}

void DEMOGfxBindProgram(DEMOGfxShader* pShader)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();

    pContext->shader = *pShader;

    lwnCommandBufferBindProgram(s_cmd, pShader->pgm, pShader->bits);
}

void DEMOGfxSetShaders(DEMOGfxShader* pShader)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();

    pContext->shader = *pShader;

    lwnCommandBufferBindProgram(s_cmd, pShader->pgm, pShader->bits);
    lwnCommandBufferBindVertexAttribState(s_cmd, pShader->nAttribs, pShader->attribs);
    lwnCommandBufferBindVertexStreamState(s_cmd, pShader->nStreams, pShader->streams);
}

void DEMOGfxSetShaderAttribute(DEMOGfxShader *pShader, const DEMOGfxShaderAttributeData* pData)
{
    LWLwertexAttribState *vastate = pShader->attribs + pData->index;
    LWLwertexStreamState *vsstate = pShader->streams + pData->bindingIndex;
    
    if (pData->index >= pShader->nAttribs) {
        for (u32 i = pShader->nAttribs; i <= pData->index; i++) {
            lwlwertexAttribStateSetDefaults(pShader->attribs + i);
        }
        pShader->nAttribs = pData->index + 1;
    }
    if (pData->bindingIndex >= pShader->nStreams) {
        for (u32 i = pShader->nStreams; i <= pData->index; i++) {
            lwlwertexStreamStateSetDefaults(pShader->streams + i);
        }
        pShader->nStreams= pData->bindingIndex + 1;
    }

    lwlwertexAttribStateSetDefaults(vastate);
    lwlwertexAttribStateSetFormat(vastate, pData->type, pData->relativeOffset);
    lwlwertexAttribStateSetStreamIndex(vastate, pData->bindingIndex);
    lwlwertexStreamStateSetStride(vsstate, pData->bindingStride);
}

void DEMOGfxReleaseShaders(DEMOGfxShader* pShader)
{
    lwnProgramFree(pShader->pgm);
}

void DEMOGfxSetStatesDefault()
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();

	// Set default mini objects
    lwnColorStateSetDefaults(&pContext->colorState);

    lwnBlendStateSetDefaults(&pContext->blendState);

    lwnDepthStencilStateSetDefaults(&pContext->depthState);
    lwnDepthStencilStateSetDepthTestEnable(&pContext->depthState, LWN_TRUE);
    lwnDepthStencilStateSetDepthWriteEnable(&pContext->depthState, LWN_TRUE);
    lwnDepthStencilStateSetDepthFunc(&pContext->depthState, LWN_DEPTH_FUNC_LESS);

    lwnPolygonStateSetDefaults(&pContext->polygonState);

    lwnChannelMaskStateSetDefaults(&pContext->channelMasks);

    pContext->stencilCtrl.frontRef = 0;
    pContext->stencilCtrl.frontTestMask = ~0;
    pContext->stencilCtrl.frontWriteMask = ~0;
    pContext->stencilCtrl.backRef = 0;
    pContext->stencilCtrl.backTestMask = ~0;
    pContext->stencilCtrl.backWriteMask = ~0;

    memset(&pContext->shader, 0, sizeof(pContext->shader));

    DEMOGfxSetContextState(pContext);
}

// color control

void DEMOGfxSetColorControl(u32 multiWrite, u32 specialOp, u32 blendEnable, u32 rop3)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();

    for (int i = 0; i < 8; i++) {
        lwnColorStateSetBlendEnable(&pContext->colorState, i, ((blendEnable >> i) & 1) ? LWN_TRUE : LWN_FALSE);
    }
    LWNlogicOp tblLogicOp[] = {
        LWN_LOGIC_OP_CLEAR,
        LWN_LOGIC_OP_NOR,
        LWN_LOGIC_OP_AND_ILWERTED,
        LWN_LOGIC_OP_COPY_ILWERTED,
        LWN_LOGIC_OP_AND_REVERSE,
        LWN_LOGIC_OP_ILWERT,
        LWN_LOGIC_OP_XOR,
        LWN_LOGIC_OP_NAND,
        LWN_LOGIC_OP_AND,
        LWN_LOGIC_OP_EQUIV,
        LWN_LOGIC_OP_NOOP,
        LWN_LOGIC_OP_OR_ILWERTED,
        LWN_LOGIC_OP_COPY,
        LWN_LOGIC_OP_OR_REVERSE,
        LWN_LOGIC_OP_OR,
        LWN_LOGIC_OP_SET
    };
    lwnColorStateSetLogicOp(&pContext->colorState, tblLogicOp[rop3 & 0xF]);

    // multiWrite and specialOp not used.

    lwnCommandBufferBindColorState(s_cmd, &pContext->colorState);
}

// blend control

void DEMOGfxSetBlendControl(int target,
                            LWNblendFunc colorFuncSrc, LWNblendFunc colorFuncDst, LWNblendEquation colorEquation,
                            LWNblendFunc alphaFuncSrc, LWNblendFunc alphaFuncDst, LWNblendEquation alphaEquation)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();

    lwnBlendStateSetBlendTarget(&pContext->blendState, target);
    lwnBlendStateSetBlendFunc(&pContext->blendState, colorFuncSrc, colorFuncDst, alphaFuncSrc, alphaFuncDst);
    lwnBlendStateSetBlendEquation(&pContext->blendState, colorEquation, alphaEquation);

    lwnCommandBufferBindBlendState(s_cmd, &pContext->blendState);
}

// depth control

void DEMOGfxSetDepthControl(int stencilEnable, int depthEnable, int depthWriteEnable, LWNdepthFunc depthFunc)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();

    lwnDepthStencilStateSetStencilTestEnable(&pContext->depthState, stencilEnable != 0);
    lwnDepthStencilStateSetDepthTestEnable(&pContext->depthState, depthEnable != 0);
    lwnDepthStencilStateSetDepthWriteEnable(&pContext->depthState, depthWriteEnable != 0);
    lwnDepthStencilStateSetDepthFunc(&pContext->depthState, depthFunc);

    lwnCommandBufferBindDepthStencilState(s_cmd, &pContext->depthState);
}

// polygon control

void DEMOGfxSetPolygonControl(int lwllFront, int lwllBack, LWNfrontFace frontFace, int polyMode,
                              LWNpolygonMode frontMode, LWNpolygonMode backMode,
                              int offsetFront, int offsetBack, int offsetPointsLines)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();
    
    LWNface lwllFace;
    if (lwllFront) {
        lwllFace = lwllBack ? LWN_FACE_FRONT_AND_BACK : LWN_FACE_FRONT;
    } else {
        lwllFace = lwllBack ? LWN_FACE_BACK : LWN_FACE_NONE;
    }
    lwnPolygonStateSetLwllFace(&pContext->polygonState, lwllFace);
    lwnPolygonStateSetFrontFace(&pContext->polygonState, frontFace);
    if (polyMode) {
        lwnPolygonStateSetPolygonMode(&pContext->polygonState, frontMode);
        // LWN polygon state doesn't have separate front and back polygon modes.
    }
    int offsetEnables = 0;
    if (offsetFront || offsetBack) {
        offsetEnables |= LWN_POLYGON_OFFSET_ENABLE_FILL_BIT;
    }
    if (offsetPointsLines) {
        offsetEnables |= LWN_POLYGON_OFFSET_ENABLE_LINE_BIT;
        offsetEnables |= LWN_POLYGON_OFFSET_ENABLE_POINT_BIT;
    }
    lwnPolygonStateSetPolygonOffsetEnables(&pContext->polygonState, offsetEnables);

    lwnCommandBufferBindPolygonState(s_cmd, &pContext->polygonState);
}

// stencil control

void DEMOGfxSetStencilControl(int stencilRef, int stencilTestMask, int stencilWriteMask,
                              int backStencilRef, int backStencilTestMask, int backStencilWriteMask)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();
    
	pContext->stencilCtrl.frontRef = stencilRef;
    pContext->stencilCtrl.frontTestMask= stencilTestMask;
    pContext->stencilCtrl.frontWriteMask = stencilWriteMask;
    pContext->stencilCtrl.backRef = backStencilRef;
    pContext->stencilCtrl.backTestMask = backStencilTestMask;
    pContext->stencilCtrl.backWriteMask = backStencilWriteMask;

    lwnCommandBufferSetStencilRef(s_cmd, LWN_FACE_FRONT, stencilRef);
    lwnCommandBufferSetStencilValueMask(s_cmd, LWN_FACE_FRONT, stencilTestMask);
    lwnCommandBufferSetStencilMask(s_cmd, LWN_FACE_FRONT, stencilWriteMask);
    lwnCommandBufferSetStencilRef(s_cmd, LWN_FACE_BACK, backStencilRef);
    lwnCommandBufferSetStencilValueMask(s_cmd, LWN_FACE_BACK, backStencilTestMask);
    lwnCommandBufferSetStencilMask(s_cmd, LWN_FACE_BACK, backStencilWriteMask);
}

// channel mask control

void DEMOGfxSetChannelMasks(int target0All)
{
    DEMOGfxContextState* pContext = DEMOGfxGetContextState();
    
    for (int i = 0; i < 8; i++) {
        lwnChannelMaskStateSetChannelMask(&pContext->channelMasks, i,
                                          LWNboolean((target0All >> (4 * i + 0)) & 1),
                                          LWNboolean((target0All >> (4 * i + 1)) & 1),
                                          LWNboolean((target0All >> (4 * i + 2)) & 1),
                                          LWNboolean((target0All >> (4 * i + 3)) & 1));
    }

    lwnCommandBufferBindChannelMaskState(s_cmd, &pContext->channelMasks);
}

// vertex

void DEMOGfxCreateVertexBuffer(DEMOGfxVertexData* pData, u32 size)
{
    // Create a vertex buffer 
    lwnBufferBuilderSetDefaults(s_bufferBuilder);
    pData->object = s_bufferPool->allocBuffer(s_bufferBuilder, BUFFER_ALIGN_VERTEX_BIT, size);
    pData->address = lwnBufferGetAddress(pData->object);
    
    // Create persistent mapping
    pData->pBuffer = lwnBufferMap(pData->object);
}

void DEMOGfxSetVertexBuffer(DEMOGfxVertexData* pData, const void* pVertexData, u32 offset, u32 size)
{
    memcpy((char*)pData->pBuffer + offset, pVertexData, size);
}

void DEMOGfxBindVertexBuffer(DEMOGfxVertexData* pData, u32 index, u32 offset, u32 size)
{
    lwnCommandBufferBindVertexBuffer(s_cmd, index, pData->address + offset, size);
}

void DEMOGfxReleaseVertexBuffer(DEMOGfxVertexData* pData)
{
    s_bufferPool->freeBuffer(pData->object);
}

// index

void DEMOGfxCreateIndexBuffer(DEMOGfxIndexData* pData, u32 size)
{
    // Create a index buffer 
    lwnBufferBuilderSetDefaults(s_bufferBuilder);
    pData->object = s_bufferPool->allocBuffer(s_bufferBuilder, BUFFER_ALIGN_INDEX_BIT, size);
    pData->address = lwnBufferGetAddress(pData->object);
    
    // Create persistent mapping
    pData->pBuffer = lwnBufferMap(pData->object);
}

void DEMOGfxSetIndexBuffer(DEMOGfxIndexData* pData, const void* pIndexData, u32 offset, u32 size)
{
    memcpy((char*)pData->pBuffer + offset, pIndexData, size);
}

void DEMOGfxReleaseIndexBuffer(DEMOGfxIndexData* pData)
{
    s_bufferPool->freeBuffer(pData->object);
}

// uniform

void DEMOGfxCreateUniformBuffer(DEMOGfxUniformData* pData, u32 size)
{
    // Create a uniform buffer 
    lwnBufferBuilderSetDefaults(s_bufferBuilder);
    pData->object = s_bufferPool->allocBuffer(s_bufferBuilder, BUFFER_ALIGN_UNIFORM_BIT, size);
    pData->address = lwnBufferGetAddress(pData->object);
    
    // Create persistent mapping
    pData->pBuffer = lwnBufferMap(pData->object);
}

void DEMOGfxSetUniformBuffer(DEMOGfxUniformData* pData, const void* pUniformData, u32 offset, u32 size)
{
    memcpy((char*)pData->pBuffer + offset, pUniformData, size);
}

void DEMOGfxBindUniformBuffer(const DEMOGfxUniformData* pData, LWNshaderStage stage, u32 index, u32 offset, u32 size)
{
    lwnCommandBufferBindUniformBuffer(s_cmd, stage, index, pData->address + offset, size);
}

void DEMOGfxReleaseUniformBuffer(DEMOGfxUniformData* pData)
{
    s_bufferPool->freeBuffer(pData->object);
}

// texture

void DEMOGfxCreateTextureBuffer(DEMOGfxTexture* pTexture, LWNtextureTarget target, u32 levels, LWNformat format, u32 width, u32 height, u32 depth, u32 samples, u32 size)
{
    int samplerPoolID, texturePoolID;

    // Create the sampler
    // [TODO] Need function to set-up them
    lwnSamplerBuilderSetDefaults(s_samplerBuilder);
    lwnSamplerBuilderSetMinMagFilter(s_samplerBuilder, LWN_MIN_FILTER_LINEAR, LWN_MAG_FILTER_LINEAR);
    lwnSamplerBuilderSetWrapMode(s_samplerBuilder, LWN_WRAP_MODE_CLAMP, LWN_WRAP_MODE_CLAMP, LWN_WRAP_MODE_CLAMP);
    pTexture->sampler.object = lwnSamplerBuilderCreateSampler(s_samplerBuilder);
    samplerPoolID = lwnSamplerGetRegisteredID(pTexture->sampler.object);
    
    // Create the texture
    lwnTextureBuilderSetDefaults(s_textureBuilder);
    lwnTextureBuilderSetTarget(s_textureBuilder, target);
    lwnTextureBuilderSetLevels(s_textureBuilder, levels);
    lwnTextureBuilderSetFormat(s_textureBuilder, format);
    lwnTextureBuilderSetSize2D(s_textureBuilder, width, height);
    lwnTextureBuilderSetDepth(s_textureBuilder, depth);
    lwnTextureBuilderSetSamples(s_textureBuilder, samples);
    pTexture->object = s_texturePool->allocTexture(s_textureBuilder);
    texturePoolID = lwnTextureGetRegisteredTextureID(pTexture->object);
    
    // Create the data
    lwnBufferBuilderSetDefaults(s_bufferBuilder);
    pTexture->data.object = s_bufferPool->allocBuffer(s_bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, size);
    pTexture->data.address = lwnBufferGetAddress(pTexture->data.object);

    // Create persistent mapping
    pTexture->data.pBuffer = lwnBufferMap(pTexture->data.object);

    pTexture->handle = lwnDeviceGetTextureHandle(s_device, texturePoolID, samplerPoolID);
}

void DEMOGfxSetTextureBuffer(DEMOGfxTexture* pTexture, s32 bufferOffset, const void* pTextureData, u32 level, int xOffset, int yOffset, int zOffset, int width, int height, int depth, u32 size)
{
    memcpy((char*)pTexture->data.pBuffer + bufferOffset, pTextureData, size); 

    // [TODO] XXX missing pixelpack object
    // Download the texture data
    LWNcopyRegion copyRegion = { xOffset, yOffset, zOffset, width, height, depth };
    LWNtextureView texViewC;
    lwnTextureViewSetDefaults(&texViewC);
    lwnTextureViewSetLevels(&texViewC, level, 1);
    lwnCommandBufferCopyBufferToTexture(s_queueCB, pTexture->data.address + bufferOffset, pTexture->object, &texViewC, &copyRegion, LWN_COPY_FLAGS_NONE);
}

void DEMOGfxBindTextureBuffer(const DEMOGfxTexture* pTexture, LWNshaderStage stage, u32 index)
{
    lwnCommandBufferBindTexture(s_cmd, stage, index, pTexture->handle);
}

void DEMOGfxReleaseTextureBuffer(DEMOGfxTexture* pTexture)
{
    s_texturePool->freeTexture(pTexture->object);
    lwnSamplerFree(pTexture->sampler.object);
    s_bufferPool->freeBuffer(pTexture->data.object);
}

// clear

void DEMOGfxClearColor(f32 r, f32 g, f32 b, f32 a)
{
    float clearColor[4] = {r, g, b, a};
    lwnCommandBufferClearColor(s_cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
}

void DEMOGfxClearDepthStencil(f32 depth, u32 stencil)
{
    lwnCommandBufferClearDepthStencil(s_cmd, depth, LWN_TRUE, stencil, 0xFFFFFFFF);
}

// draw

void DEMOGfxDrawArrays(LWNdrawPrimitive mode, u32 first, u32 count)
{
    lwnCommandBufferDrawArrays(s_cmd, mode, first, count);
}

void DEMOGfxDrawElements(DEMOGfxIndexData *pData, LWNdrawPrimitive mode, LWNindexType type, u32 count, u32 offset)
{
    lwnCommandBufferDrawElements(s_cmd, mode, type, count, pData->address + offset);
}


// load image file

static void GenTextureImage(const DEMOGfxTGAInfo* pTGAD, const u8* pTgaImage, u8* pTextureImage, u32 width, u32 height, u32 bpp, u32 offset)
{
    u32 x   = 0;
    u32 y   = 0;
    int i   = 0;
    u32 cls = 4; // r,g,b,a
    
    u32 id;
    u8 c1, c2, c3, c4;

    while(y < height)
    {
        switch (bpp)
        {
        // 24bit
        case 24:
            c1  = pTgaImage[i++];
            c3  = pTgaImage[i++];
            c2  = pTgaImage[i++];
            c4  = 0xff;
            break;
        // 32 bit
        case 32:
        default:
            c4 = pTgaImage[i++];
            c3 = pTgaImage[i++];
            c2 = pTgaImage[i++];
            c1 = pTgaImage[i++];
            break;
        }

        id = ((width + offset)* y + x) * cls;

        // RGBA
        pTextureImage[id]     = c1;
        pTextureImage[id + 1] = c2;
        pTextureImage[id + 2] = c3;
        pTextureImage[id + 3] = c4;

        x ++;
        if(x == width)
        {
            x = 0;
            y++;
        }
    }
}


//
// Image swap function
//

static void SwapFromBottomToTop(u8* pTextureImage, u32 width, u32 height)
{
    u32 x,y,c;
    u32 *temp = (u32 *)pTextureImage;
    
    for(y = 0; y < height/2; y++)
    {
        for(x = 0; x < width; x++)
        {
            u32 id0 = (width * y + x);
            u32 id1 = (width * (height-1 - y) + x);
            c = temp[id0];
            temp[id0] = temp[id1];
            temp[id1] = c;
        }
    }
}

static void SwapFromRightToLeft(u8* pTextureImage, u32 width, u32 height)
{
    u32 x,y,c;
    u32 *temp = (u32 *)pTextureImage;

    for(y = 0; y < height; y++)
    {
        for(x = 0; x < width/2; x++)
        {
            u32 id0 = width * y + x;
            u32 id1 = width * y + (width-1 - x);
            c = temp[id0];
            temp[id0] = temp[id1];
            temp[id1] = c;
        }
    }
}

// Load tga function

static BOOL GetImageFromTGAData(const DEMOGfxTGAInfo* pTGAD, const u8* pTgaImage, u8* pTextureImage, u32 offset)
{
    BOOL isRLE     = (pTGAD->header.imageType & 0x08)  != 0;
    BOOL fromTop   = (pTGAD->header.descriptor & 0x20) == 0;
    //BOOL fromTop   = (pTGAD->header.descriptor & 0x20) != 0;
    BOOL fromLeft  = (pTGAD->header.descriptor & 0x10) == 0;
    u32 dwWidth    = pTGAD->header.dwWidth;
    u32 dwHeight   = pTGAD->header.dwHeight;

    if(isRLE)
    {
        DEMOPrintf("RLE format is not supported.\n");
        return FALSE;
    }
    
    GenTextureImage(pTGAD, pTgaImage, pTextureImage, dwWidth, dwHeight, pTGAD->header.colorDepth, offset);

    // Swap image
    if(!fromTop && fromLeft)
    {
        SwapFromBottomToTop(pTextureImage, dwWidth + offset, dwHeight);
    }
    else if(fromTop && !fromLeft)
    {
        SwapFromRightToLeft(pTextureImage, dwWidth + offset, dwHeight);
    }
    else if(!fromTop && !fromLeft)
    {
        SwapFromBottomToTop(pTextureImage, dwWidth + offset, dwHeight);
        SwapFromRightToLeft(pTextureImage, dwWidth + offset, dwHeight);
    }
    
    return TRUE;
}

void *DEMOGfxLoadTGAImage(u8* pTgaData, u32 fileSize, DEMOGfxTGAInfo *pInfo, u32 *pSize)
{
    u32 imageSize = fileSize - sizeof(DEMOGfxTGAHeader);

    // Read header from the file.
    // status must indicate read count.        
    memcpy(&pInfo->header, pTgaData, sizeof(DEMOGfxTGAHeader));

    u8* pImage = (u8*)DEMOAllocEx(imageSize, DEMO_BUFFER_ALIGN);
    DEMOAssert(pImage && "couldn't alloc memory");

    // Set image data to buffer
    memcpy(pImage, (u8*)pTgaData + sizeof(DEMOGfxTGAHeader), imageSize);

    // load app texture            
    BOOL ret = GetImageFromTGAData(pInfo, pImage, pImage, 0);
    DEMOAssert(ret && "couldn't alloc memory");
    (void) ret;

    *pSize = imageSize;

    return pImage;
}

void DEMOGfxReleaseTGAImage(void *pBuffer)
{
    if(pBuffer)
    {
        DEMOFree(pBuffer);
    }
}
