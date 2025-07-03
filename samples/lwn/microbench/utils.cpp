/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "utils.hpp"
#include "options.hpp"
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>

#include <lwn/lwn_FuncPtrImpl.h>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
  #define GETPROCADDR(x)        wglGetProcAddress(#x)
  #define GETPROCADDR_10(x)     x
#else
  #if !defined(WIN_INTERFACE_LWSTOM)
  #define WIN_INTERFACE_LWSTOM
  #endif
  #include <EGL/egl.h>
  #define GETPROCADDR(x)        eglGetProcAddress(#x)
  #define GETPROCADDR_10(x)     eglGetProcAddress(#x)
#endif

#include "lwnutil.h"
#include "lwnUtil/lwnUtil_GlslcHelper.h"
#include "lwnUtil/lwnUtil_GlslcHelperImpl.h"
#include "lwnUtil/lwnUtil_AlignedStorage.h"
#include "lwnUtil/lwnUtil_AlignedStorageImpl.h"
#include "lwnUtil/lwnUtil_PoolAllocator.h"
#include "lwnUtil/lwnUtil_PoolAllocatorImpl.h"

using namespace LwnUtil;

static const float PI = 3.14159265359f;

namespace LwnUtil {
#define DEFINE_GL_CALLBACK(ucname, lcname) \
    PFNGL##ucname##PROC g_gl##lcname;

FOREACH_GL_PROC(DEFINE_GL_CALLBACK)
FOREACH_GL_10_PROC(DEFINE_GL_CALLBACK)

#undef DEFINE_GL_CALLBACK
};

static lwnUtil::GLSLCLibraryHelper* g_glslcLibraryHelper = NULL;
static lwnUtil::GLSLCHelper*        g_glslcHelper = NULL;

static void LWNAPIENTRY debugCB(LWNdebugCallbackSource source,
                                LWNdebugCallbackType type,
                                int id,
                                LWNdebugCallbackSeverity severity,
                                const char *message,
                                void *userParam)
{
    PRINTF("\nLWN Debug callback (from --debug option):\n");
    PRINTF("  source:       0x%08x\n", source);
    PRINTF("  type:         0x%08x\n", type);
    PRINTF("  id:           0x%08x\n", id);
    PRINTF("  severity:     0x%08x\n", severity);
    PRINTF("  userParam:    0x%08x%08x\n",
           uint32_t(uint64_t(userParam) >> 32), uint32_t(uint64_t(userParam)));
    PRINTF("  message:\n    %s\n", message);
    if (severity == LWN_DEBUG_CALLBACK_SEVERITY_HIGH)
        exit(1);
}

//--------------------------------------------------------------------

#if !defined(_WIN32)
extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);
#endif

LWNdevice* LwnUtil::init(uint32_t initFlags, const char* lwnglslcDllPath)
{
    // Start by looking up a single-argument "bootstrap loader" function, and
    // if found, use to find lwnDeviceGetProcAddress.
    //
    // For HOS/LINUX, directly use the lwn exported function.
    PFNLWNDEVICEGETPROCADDRESSPROC lwnDeviceGetProcAddress = NULL;

#if defined(_WIN32)
    PFNLWNBOOTSTRAPLOADERPROC bootstrapLoader = (PFNLWNBOOTSTRAPLOADERPROC)GETPROCADDR(rq34nd2ffz);
    if (bootstrapLoader) {
        lwnDeviceGetProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC) ((*bootstrapLoader)("lwnDeviceGetProcAddress"));
    }
#else
    lwnDeviceGetProcAddress = (PFNLWNDEVICEGETPROCADDRESSPROC) ((*lwnBootstrapLoader)("lwnDeviceGetProcAddress"));
#endif

    LWNdevice* lwnDevice = NULL;

    if (lwnDeviceGetProcAddress) {
        lwnLoadCProcs(NULL, lwnDeviceGetProcAddress);
        // Check that functions exist to create a device, query the version,
        // and release the device if creation fails.  If we're missing any of
        // these, give up.
#if 0   //TODO Compiler thinks these are always true?
        if (!lwnDeviceInitialize || !lwnDeviceGetInteger || !lwnDeviceFinalize) {
            return NULL;
        }
#endif
        bool lwnDebug = (initFlags & LWN_INIT_DEBUG_LAYER_BIT) != 0;

        // Try to create and initialize a device, and bail if it fails.
        LWNdeviceBuilder deviceBuilder;
        lwnDeviceBuilderSetDefaults(&deviceBuilder);

        uint32_t deviceFlags = lwnDebug ?
            (LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT | LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT) : 0;
        lwnDeviceBuilderSetFlags(&deviceBuilder, deviceFlags);

        lwnDevice = new LWNdevice;
        if (!lwnDevice) {
            return NULL;
        }
        if (!lwnDeviceInitialize(lwnDevice, &deviceBuilder)) {
            delete lwnDevice;
            return NULL;
        }

        // Re-initialize the LWN entry points with the initialized device.
        lwnLoadCProcs(lwnDevice, lwnDeviceGetProcAddress);

        if (lwnDebug) {
            PRINTF("enabling LWN debug layer\n");
            lwnDeviceInstallDebugCallback(lwnDevice, debugCB, NULL, LWN_TRUE);
        }

        // Check the API version.  A major version mismatch between the driver
        // and the test indicates that some API changed in a
        // backwards-incompatible manner.  All tests should be disabled in
        // this case, since they could easily crash otherwise.
        int lwnMajorVersion = -1, lwnMinorVersion = -1;
        lwnDeviceGetInteger(lwnDevice, LWN_DEVICE_INFO_API_MAJOR_VERSION, &lwnMajorVersion);
        lwnDeviceGetInteger(lwnDevice, LWN_DEVICE_INFO_API_MINOR_VERSION, &lwnMinorVersion);
        if (lwnMajorVersion != LWN_API_MAJOR_VERSION) {
            PRINTF("!!! LWN API version mismatch.\n");
            lwnDeviceFinalize(lwnDevice);
            delete lwnDevice;
            return NULL;
        }
    }

    // Load OpenGL functions

#define LOAD_GL_CALLBACK(ucname, lcname)                                \
    g_gl##lcname = (PFNGL##ucname##PROC)GETPROCADDR(gl##lcname);
#define LOAD_GL_10_CALLBACK(ucname, lcname)                             \
    g_gl##lcname = (PFNGL##ucname##PROC)GETPROCADDR_10(gl##lcname);

FOREACH_GL_PROC(LOAD_GL_CALLBACK)
FOREACH_GL_10_PROC(LOAD_GL_10_CALLBACK)

#undef LOAD_GL_CALLBACK

    // Setup online compilation for LWN

    g_glslcLibraryHelper = new lwnUtil::GLSLCLibraryHelper;
    g_glslcLibraryHelper->LoadDLL(lwnglslcDllPath);
    g_glslcHelper = new lwnUtil::GLSLCHelper(lwnDevice, 0x100000UL, g_glslcLibraryHelper);

    return lwnDevice;
}

void LwnUtil::exit()
{
    delete g_glslcHelper;
    delete g_glslcLibraryHelper;
}

void LwnUtil::enableWarpLwlling(bool enable)
{
    g_glslcHelper->EnableWarpLwlling(enable);
}

void LwnUtil::enableCBF(bool enable)
{
    g_glslcHelper->EnableCBF(enable);
}

void LwnUtil::setShaderScratchMemory(LwnUtil::GPUBufferPool* pool, size_t size, LWNcommandBuffer* cb)
{
    // TODO: asserts?
    g_glslcHelper->SetShaderScratchMemory(pool->pool(), 0, size, cb);
}

// For GLSLC Specialization
void LwnUtil::setData(GLSLCspecializationUniform * uniform, const char * name, int numElements,
    ArgTypeEnum type, int numArgs, ...)
{

    assert(numArgs % numElements == 0);

    ArrayUnion * arryUnion = (ArrayUnion *)uniform->values;
    int numComponents = numArgs / numElements;
    int elementSize = 0;

    switch (type) {
    case ARG_TYPE_DOUBLE:
        elementSize = numComponents * sizeof(double);
        break;
    case ARG_TYPE_UINT:
    case ARG_TYPE_FLOAT:
    case ARG_TYPE_INT:
        elementSize = numComponents * sizeof(uint32_t);
        break;
    };

    uniform->uniformName = name;
    uniform->numElements = numElements;
    uniform->elementSize = elementSize;

    va_list arguments;

    memset(arryUnion, 0, sizeof(ArrayUnion));

    va_start(arguments, numArgs);
    for (int i = 0; i < numArgs; ++i) {
        switch (type) {
        case ARG_TYPE_INT:
            arryUnion->i[i] = va_arg(arguments, int32_t);
            break;
        case ARG_TYPE_FLOAT:
            arryUnion->f[i] = (float)va_arg(arguments, double);
            break;
        case ARG_TYPE_DOUBLE:
            arryUnion->d[i] = va_arg(arguments, double);
            break;
        case ARG_TYPE_UINT:
            arryUnion->u[i] = va_arg(arguments, uint32_t);
            break;
        default:
            assert(!"Invalid argument type.");
            break;
        };
    }
    va_end(arguments);

    return;
}

void LwnUtil::addSpecializationUniform(int index, const GLSLCspecializationUniform* uniform)
{
    assert(uniform);
    assert(index >= 0);
    g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
}

void LwnUtil::clearSpecializationUniformArrays(void) {
    g_glslcHelper->ClearSpecializationUniformArrays();
}

bool LwnUtil::compileAndSetShaders(LWNprogram* pgm, const LWNshaderStage* stages, uint32_t count, const char** srcs) {
    return g_glslcHelper->CompileAndSetShaders(pgm, stages, count, srcs) == LWN_TRUE;
}

//--------------------------------------------------------------------

BufferPool::BufferPool(LWNdevice* device, uintptr_t size, uint32_t poolFlags) :
    m_poolTop(0),
    m_poolSize(size),
    m_memory(new uint8_t[LwnUtil::alignSize(size, LWN_MEMORY_POOL_STORAGE_ALIGNMENT)])
{
    if (size) {
        LWNmemoryPoolBuilder builder;

        lwnMemoryPoolBuilderSetDevice(&builder, device);
        lwnMemoryPoolBuilderSetDefaults(&builder);
        lwnMemoryPoolBuilderSetStorage(&builder, LwnUtil::align(m_memory.get(), LWN_MEMORY_POOL_STORAGE_ALIGNMENT), size);
        lwnMemoryPoolBuilderSetFlags(&builder, poolFlags);
        lwnMemoryPoolInitialize(&m_pool, &builder);
    }
}

BufferPool::~BufferPool()
{
    if (m_poolSize) {
        lwnMemoryPoolFinalize(&m_pool);
    }
}

uintptr_t BufferPool::alloc(uintptr_t size, uintptr_t align)
{
    assert(m_poolSize && "Can't call BufferPool::alloc on a BufferPool with zero size");

    uintptr_t alignedOffset = LwnUtil::align(m_poolTop, align);
    m_poolTop = alignedOffset + size;
    assert(m_poolTop <= m_poolSize);
    return alignedOffset;
}

void BufferPool::freeAll()
{
    m_poolTop = 0;
}

//--------------------------------------------------------------------

DescriptorPool::DescriptorPool(LWNdevice* device, DescBufferPool* descBufferPool, int maxIDs)
{
    int textureSize, samplerSize;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SAMPLER_DESCRIPTOR_SIZE, &samplerSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE, &textureSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS, &m_numReservedSamplers);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS, &m_numReservedTextures);
    m_maxIDs = maxIDs;
    int numSamplers = m_numReservedSamplers + maxIDs;
    int numTextures = m_numReservedTextures + maxIDs;

    uintptr_t samplerOffset = descBufferPool->alloc(numSamplers * samplerSize, 16);
    lwnSamplerPoolInitialize(&m_samplerPool, descBufferPool->pool(), samplerOffset, numSamplers);

    uintptr_t textureOffset = descBufferPool->alloc(numTextures * textureSize, 16);
    lwnTexturePoolInitialize(&m_texturePool, descBufferPool->pool(), textureOffset, numTextures);

    m_samplerIDTop = m_numReservedSamplers;
    m_textureIDTop = m_numReservedTextures;
}

void DescriptorPool::setPools(LWNcommandBuffer* cmd)
{
    lwnCommandBufferSetSamplerPool(cmd, &m_samplerPool);
    lwnCommandBufferSetTexturePool(cmd, &m_texturePool);
}

DescriptorPool::~DescriptorPool()
{
}

int DescriptorPool::allocTextureID()
{
    assert(m_textureIDTop < m_numReservedTextures + m_maxIDs);
    int ret = m_textureIDTop;
    m_textureIDTop++;
    return ret;
}

void DescriptorPool::registerTexture(uint32_t texId, LWNtexture* tex)
{
    lwnTexturePoolRegisterTexture(&m_texturePool, texId, tex, NULL);
}

void DescriptorPool::registerImage(uint32_t texId, LWNtexture* image)
{
    lwnTexturePoolRegisterImage(&m_texturePool, texId, image, NULL);
}

void DescriptorPool::registerSampler(uint32_t smpId, LWNsampler* smp)
{
    lwnSamplerPoolRegisterSampler(&m_samplerPool, smpId, smp);
}

int DescriptorPool::allocSamplerID()
{
    assert(m_samplerIDTop < m_numReservedSamplers + m_maxIDs);
    int ret = m_samplerIDTop;
    m_samplerIDTop++;
    return ret;
}

void DescriptorPool::freeAll()
{
    m_samplerIDTop = m_numReservedSamplers;
    m_textureIDTop = m_numReservedTextures;
}


//--------------------------------------------------------------------

Pools::Pools(LWNdevice* device, LwnUtil::DescriptorPool* descriptorPool,
             size_t gpuPoolSize, size_t cohPoolSize, size_t cpuCachedSize) :
    m_descriptor(descriptorPool),
    m_gpu(new LwnUtil::GPUBufferPool(device, gpuPoolSize)),
    m_coherent(new LwnUtil::CoherentBufferPool(device, cohPoolSize)),
    m_cpuCached(new LwnUtil::CPUCachedBufferPool(device, cpuCachedSize))
{
}

void Pools::freeAll()
{
    m_gpu->freeAll();
    m_coherent->freeAll();
    m_cpuCached->freeAll();
}

Pools::~Pools()
{
}

//--------------------------------------------------------------------

#if 0
#define CMDPRINT(A) PRINTF A
#else
#define CMDPRINT(A)
#endif

class LwnUtil::RingBuffer {
public:
    RingBuffer(LWNdevice* device, LWNqueue* queue, int numChunks) {
        m_queue = queue;
        m_numChunks = numChunks;
        m_put = 0;
        m_submit = 0;
        m_get = 0;
        m_fences = new LWNsync[numChunks];
        m_fenceNumChunks = new int[numChunks];
        for(int i=0;i<m_numChunks;i++) {
            lwnSyncInitialize(&m_fences[i], device);
            m_fenceNumChunks[i] = 1;
        }
    }
    ~RingBuffer() {
        for(int i=0;i<m_numChunks;i++) {
            lwnSyncFinalize(&m_fences[i]);
        }
        delete[] m_fences;
        delete[] m_fenceNumChunks;
    }
    void submit() {
        int n = numUnsubmittedEntries();
        if (!n) {
            CMDPRINT((" fence -\n"));
            return;
        }
        //issue a fence for the submitted command buffer chunks
        CMDPRINT((" fence %d+%d\n", m_submit, n));
        m_fenceNumChunks[m_submit] = n; //store the number of chunks guarded by this fence
        LWNsync* fence = &m_fences[m_submit];
        lwnQueueFenceSync(m_queue, fence, LWN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, LWN_SYNC_FLAG_FLUSH_FOR_CPU_BIT);
        lwnQueueFlush(m_queue);
        m_submit = add(m_submit, n);
        printStatus();
    }
    int addChunk() {
        printStatus();

        if (isOutOfSpace()) {
            PRINTF("CmdBuf: no chunks left to allocate. The command buffer needs to be made bigger to accommodate all commands, or commands need to be submitted more frequently\n");
            assert(0);
        }

        //read expired fences
        while (hasBusyEntries()) {
            LWNsyncWaitResult res = lwnSyncWait(&m_fences[m_get], LWN_WAIT_TIMEOUT_NONE);
            CMDPRINT(("  get %d+%d: ", m_get, m_fenceNumChunks[m_get]-1)); printWaitResult(res); CMDPRINT(("\n"));
            if (res == LWN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED)
                break;  //chunk is busy, bail
            m_get = add(m_get, m_fenceNumChunks[m_get]);
        }

        //stall if all chunks are in use
        if (isFull()) {
            CMDPRINT(("  no free chunks, stalling (put = %d submit = %d get = %d)\n", m_put, m_submit, m_get));
            LWNsyncWaitResult res = lwnSyncWait(&m_fences[m_get], LWN_WAIT_TIMEOUT_MAXIMUM);
            CMDPRINT(("  get %d+%d: ", m_get, m_fenceNumChunks[m_get]-1)); printWaitResult(res); CMDPRINT(("\n"));
            assert(res != LWN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED);
            m_get = add(m_get, m_fenceNumChunks[m_get]);
        }

        //advance put pointer and reserve space for the new chunk
        m_put = add(m_put, 1);

        printStatus();
        return m_put;
    }
private:
    void printStatus()
    {
        char status[256];
        assert(m_numChunks < (int)sizeof(status));
        status[m_put] = 'c';
        int c = add(m_put, 1);
        for(;c != m_get;c=add(c, 1))
            status[c] = 'f';
        for(;c != m_submit;c=add(c, 1))
            status[c] = 'b';
        for(;c != m_put;c=add(c, 1))
            status[c] = 's';
        CMDPRINT(("  "));
        for(int i=0;i<m_numChunks;i++) {
            CMDPRINT(("%c ", status[i]));
        }
        CMDPRINT(("\n"));
        (void) status;
    }
    void printWaitResult(LWNsyncWaitResult res)
    {
        switch(res) {
        case LWN_SYNC_WAIT_RESULT_ALREADY_SIGNALED:    CMDPRINT(("done")); break;         //The condition was already satisfied at the time the command was issued.
        case LWN_SYNC_WAIT_RESULT_CONDITION_SATISFIED: CMDPRINT(("waited, done")); break; //The condition was not already satisfied at the time the command was issued, but was satisfied before the timeout.
        case LWN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED:     CMDPRINT(("busy")); break;         //The condition was not satisfied before the timeout expired.
        case LWN_SYNC_WAIT_RESULT_FAILED:              CMDPRINT(("failed")); break;       //An error oclwrred. There is lwrrently only one error condition - if the sync object has not had FenceSync called on it yet.
        default: assert(0); break;
        }
    }
    LWNqueue*           m_queue;
    int                 m_numChunks;
    LWNsync*            m_fences;
    int*                m_fenceNumChunks;
    int                 m_put;          //current chunk being used (free entries: [m_put+1, m_get-1])
    int                 m_submit;       //first entry not yet submitted (entries to be submitted: [m_submit, m_put-1])
    int                 m_get;          //first entry still busy after submits (busy entries: [m_get, m_submit-1])
    /*
        a chunk can be in four states
        -current (c): current chunk being filled
        -free (f)
        -full, to be submitted (s)
        -busy (b): submitted, gpu is processing it

        m_get     m_submit    m_put
        |         |           |
        b b b b b s s s s s s c f f f f
    */
    int                 add(int a, int b) const         { return (a + b) % m_numChunks; }
    bool                isFull() const                  { return add(m_put, 1) == m_get; }        // full:         b b b b b b b b b b s s s s s c
    bool                hasBusyEntries() const          { return m_get != m_submit; }
    int                 numUnsubmittedEntries() const   { int n = m_put - m_submit; if (n < 0) n = m_put + m_numChunks - m_submit; assert(n >= 0 && n <= m_numChunks); return n; }
    bool                isOutOfSpace() const            { return add(m_put, 1) == m_submit; }     // out of space: s s s s s s s s s s s s s s s c
};

//--------------------------------------------------------------------

CmdBuf::CmdBuf(LWNdevice* device, LWNqueue* queue, LwnUtil::CoherentBufferPool* coherentPool, int numChunks, int cmdChunkSize, int ctrlChunkSize)
{
    m_device = device;
    m_queue = queue;

    lwnCommandBufferInitialize(&m_cmd, device);
    lwnCommandBufferSetMemoryCallback(&m_cmd, commandBufferMemoryCallback);
    lwnCommandBufferSetMemoryCallbackData(&m_cmd, this);

    m_cmdChunkSize = cmdChunkSize;
    int cmdPoolSize = numChunks * m_cmdChunkSize;
    m_cmdPoolOffset = coherentPool->alloc(cmdPoolSize, 4);
    m_cmdLWNPool = coherentPool->pool();
    lwnCommandBufferAddCommandMemory(&m_cmd, m_cmdLWNPool, m_cmdPoolOffset, m_cmdChunkSize);
    m_cmdRB = new LwnUtil::RingBuffer(device, queue, numChunks);

    m_ctrlChunkSize = ctrlChunkSize;
    int ctrlPoolSize = numChunks * m_ctrlChunkSize;
    m_ctrlPool = new char[ctrlPoolSize];
    lwnCommandBufferAddControlMemory(&m_cmd, m_ctrlPool, m_ctrlChunkSize);
    m_ctrlRB = new LwnUtil::RingBuffer(device, queue, numChunks);
}

CmdBuf::~CmdBuf()
{
    lwnCommandBufferFinalize(&m_cmd);
    delete[] m_ctrlPool;
    delete m_cmdRB;
    delete m_ctrlRB;
}

void CmdBuf::submit(uint32_t numCommands, const LWNcommandHandle* handles)
{
    lwnQueueSubmitCommands(m_queue, numCommands, handles);

    CMDPRINT(("submit cmd"));
    m_cmdRB->submit();
    CMDPRINT(("submit ctrl"));
    m_ctrlRB->submit();
}

void LWNAPIENTRY CmdBuf::commandBufferMemoryCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, size_t minSize, void *callbackData)
{
    CmdBuf* thisPtr = (CmdBuf*)callbackData;
    switch(event) {
    case LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY:
        thisPtr->addCmdChunk();
        break;
    case LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY:
        thisPtr->addCtrlChunk();
        break;
    default:
        assert(0);
    }
}

void CmdBuf::addCmdChunk()
{
    CMDPRINT(("addCmdChunk\n"));
    int c = m_cmdRB->addChunk();
    lwnCommandBufferAddCommandMemory(&m_cmd, m_cmdLWNPool, m_cmdPoolOffset + c * m_cmdChunkSize, m_cmdChunkSize);
}

void CmdBuf::addCtrlChunk()
{
    CMDPRINT(("addCtrlChunk\n"));
    int c = m_ctrlRB->addChunk();
    lwnCommandBufferAddControlMemory(&m_cmd, m_ctrlPool + c * m_ctrlChunkSize, m_ctrlChunkSize);
}

//--------------------------------------------------------------------
CompiledCmdBuf::CompiledCmdBuf(LWNdevice* device, LwnUtil::BufferPool* coherentPool, size_t cmdSize, size_t ctrlSize) : m_ctrlBuf(nullptr)
{
    lwnCommandBufferInitialize(&m_cmd, device);
    lwnCommandBufferSetMemoryCallback(&m_cmd, commandBufferMemoryCallback);
    lwnCommandBufferSetMemoryCallbackData(&m_cmd, this);

    uintptr_t poolOffset = coherentPool->alloc(cmdSize, 4);
    lwnCommandBufferAddCommandMemory(&m_cmd, coherentPool->pool(), poolOffset, cmdSize);

    // There's a minimum size for control memory, at least in the LWN
    // debug layer.  Rather than hit the debug API error, always clamp
    // the request size to the minimum allowed value.
    int minControlSize = 0;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE, &minControlSize);
    size_t reqControlSize = (int)ctrlSize < minControlSize ? minControlSize : ctrlSize;

    m_ctrlBuf.reset(new uint8_t[reqControlSize]);
    lwnCommandBufferAddControlMemory(&m_cmd, m_ctrlBuf.get(), reqControlSize);
}

void CompiledCmdBuf::begin()
{
    lwnCommandBufferBeginRecording(&m_cmd);
}

void CompiledCmdBuf::end()
{
    m_handle = lwnCommandBufferEndRecording(&m_cmd);
}

void CompiledCmdBuf::submit(LWNqueue* queue)
{
    lwnQueueSubmitCommands(queue, 1, &m_handle);
}

CompiledCmdBuf::~CompiledCmdBuf()
{
    lwnCommandBufferFinalize(&m_cmd);
}

void LWNAPIENTRY CompiledCmdBuf::commandBufferMemoryCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, size_t minRequiredSize, void *callbackData)
{
    assert(0 && !"not supported - specify initial size so that cmds fit!");
}

//--------------------------------------------------------------------

void RenderTarget::setColorDepthMode(LWNcommandBuffer* cmdBuf, uint32_t destWriteMask, bool depthTest)
{
    LWNchannelMaskState  channelMask;
    LWNdepthStencilState depth;

    // Create new state vectors
    lwnDepthStencilStateSetDefaults(&depth);
    lwnDepthStencilStateSetDepthTestEnable(&depth, depthTest ? LWN_TRUE : LWN_FALSE);
    lwnDepthStencilStateSetDepthFunc(&depth, LWN_DEPTH_FUNC_LEQUAL);
    lwnDepthStencilStateSetDepthWriteEnable(&depth, (destWriteMask & DEST_WRITE_DEPTH_BIT) != 0);

    lwnChannelMaskStateSetDefaults(&channelMask);
    bool writeColor = (destWriteMask & DEST_WRITE_COLOR_BIT) != 0;
    lwnChannelMaskStateSetChannelMask(&channelMask, 0, writeColor, writeColor, writeColor, writeColor);

    lwnCommandBufferBindChannelMaskState(cmdBuf, &channelMask);
    lwnCommandBufferBindDepthStencilState(cmdBuf, &depth);
}

RenderTarget::RenderTarget(LWNdevice* device,
                           Pools* pools,
                           int w, int h,
                           uint32_t flags) :
    m_numSamples(flags & MSAA_4X ? 4 : 1),
    m_zlwllBuffer(0),
    m_zlwllBufferSize(0)
{
    int compressibleFlag = LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT;

    LWNtextureBuilder* textureBuilder = new LWNtextureBuilder;
    lwnTextureBuilderSetDevice(textureBuilder, device);
    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetFlags(textureBuilder, compressibleFlag | LWN_TEXTURE_FLAGS_DISPLAY_BIT);
    lwnTextureBuilderSetSize2D(textureBuilder, w, h);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

    uintptr_t texSize = lwnTextureBuilderGetStorageSize(textureBuilder);
    uintptr_t texAlign = lwnTextureBuilderGetStorageAlignment(textureBuilder);

    for (int i = 0; i < 2; i++) {
        lwnTextureBuilderSetStorage(textureBuilder, pools->gpu()->pool(), pools->gpu()->alloc(texSize, texAlign));
        m_rtTex[i] = new LWNtexture;
        lwnTextureInitialize(m_rtTex[i], textureBuilder);

        uint32_t textureId = pools->descriptor()->allocTextureID();
        pools->descriptor()->registerTexture(textureId, m_rtTex[i]);
    }

    lwnTextureBuilderSetFlags(textureBuilder, compressibleFlag);

    if (flags & DEPTH_FORMAT_D24) {
        lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_DEPTH24);
    } else {
        lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_DEPTH24_STENCIL8);
    }

    if (m_numSamples != 1) {
        lwnTextureBuilderSetSamples(textureBuilder, m_numSamples);
        lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D_MULTISAMPLE);
    }

    texSize = lwnTextureBuilderGetStorageSize(textureBuilder);
    texAlign = lwnTextureBuilderGetStorageAlignment(textureBuilder);

    lwnTextureBuilderSetStorage(textureBuilder, pools->gpu()->pool(), pools->gpu()->alloc(texSize, texAlign));
    m_depthTex = new LWNtexture;

    if (flags & ADAPTIVE_ZLWLL) {
        lwnTextureBuilderSetFlags(textureBuilder, compressibleFlag | LWN_TEXTURE_FLAGS_ADAPTIVE_ZLWLL_BIT);
    } else {
        lwnTextureBuilderSetFlags(textureBuilder, compressibleFlag);
    }

    lwnTextureInitialize(m_depthTex, textureBuilder);

    // Allocate ZLwll context save/restore buffer
    int alignment = 0;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_ZLWLL_SAVE_RESTORE_ALIGNMENT, &alignment);
    m_zlwllBufferSize = lwnTextureGetZLwllStorageSize(m_depthTex);
    uint64_t zlwllBufPoolOffset = pools->gpu()->alloc(m_zlwllBufferSize, alignment);
    m_zlwllBuffer = (uint64_t)lwnMemoryPoolGetBufferAddress(pools->gpu()->pool()) + zlwllBufPoolOffset;

    uint32_t texId = pools->descriptor()->allocTextureID();
    pools->descriptor()->registerTexture(texId, m_depthTex);

    delete textureBuilder;

    if (m_numSamples != 1) {
        LWNtextureBuilder tb;
        lwnTextureBuilderSetDevice(&tb, device);
        lwnTextureBuilderSetDefaults(&tb);
        lwnTextureBuilderSetFlags(&tb, compressibleFlag);
        lwnTextureBuilderSetSize2D(&tb, w, h);
        lwnTextureBuilderSetTarget(&tb, LWN_TEXTURE_TARGET_2D_MULTISAMPLE);
        lwnTextureBuilderSetFormat(&tb, LWN_FORMAT_RGBA8);
        lwnTextureBuilderSetSamples(&tb, m_numSamples);

        uintptr_t texSize = lwnTextureBuilderGetStorageSize(&tb);
        uintptr_t texAlign = lwnTextureBuilderGetStorageAlignment(&tb);
        lwnTextureBuilderSetStorage(&tb, pools->gpu()->pool(), pools->gpu()->alloc(texSize, texAlign));

        lwnTextureInitialize(&m_rtTexMSAA, &tb);

        uint32_t textureId = pools->descriptor()->allocTextureID();
        pools->descriptor()->registerTexture(textureId, &m_rtTexMSAA);

    }

    for (int i = 0; i < 2; i++) {
        m_setTargetsCmd[i] = new LwnUtil::CompiledCmdBuf(device, pools->coherent(), 2048, 1024);

        auto cmd = m_setTargetsCmd[i]->cmd();
        m_setTargetsCmd[i]->begin();

        if (m_numSamples == 1) {
            LWNmultisampleState msaa;
            lwnMultisampleStateSetDefaults(&msaa);
            lwnCommandBufferBindMultisampleState(cmd, &msaa);
            lwnCommandBufferSetRenderTargets(cmd, 1, &m_rtTex[i], NULL, m_depthTex, NULL);
        } else {
            LWNmultisampleState msaa;
            lwnMultisampleStateSetDefaults(&msaa);
            lwnMultisampleStateSetSamples(&msaa, m_numSamples);
            lwnMultisampleStateSetMultisampleEnable(&msaa, LWN_TRUE);
            lwnCommandBufferBindMultisampleState(cmd, &msaa);

            const LWNtexture* colors[] = { &m_rtTexMSAA };

            lwnCommandBufferSetRenderTargets(cmd, 1, colors, NULL, m_depthTex, NULL);
        }

        setColorDepthMode(cmd, DEST_WRITE_COLOR_BIT | DEST_WRITE_DEPTH_BIT, false);
        m_setTargetsCmd[i]->end();
    }
}

void RenderTarget::downsample(LWNcommandBuffer* cmd, LWNtexture* dst)
{
    lwnCommandBufferDownsample(cmd, &m_rtTexMSAA, dst);
    lwnCommandBufferDiscardColor(cmd, 0);
    lwnCommandBufferDiscardDepthStencil(cmd);
}

void RenderTarget::setTargets(LWNcommandBuffer* cmd, int cbIdx)
{
    LWNcommandHandle cmdHandle = m_setTargetsCmd[cbIdx]->handle();
    lwnCommandBufferCopyCommands(cmd, 1, &cmdHandle);
}

void RenderTarget::setTargets(LWNqueue* q, int cbIdx)
{
    m_setTargetsCmd[cbIdx]->submit(q);
}

RenderTarget::~RenderTarget()
{
    for (int i = 0; i < 2; i++) {
        lwnTextureFinalize(m_rtTex[i]);
        delete m_rtTex[i];
        delete m_setTargetsCmd[i];
    }
    lwnTextureFinalize(m_depthTex);
    delete m_depthTex;
    if (m_numSamples != 1) {
        lwnTextureFinalize(&m_rtTexMSAA);
    }
}

//--------------------------------------------------------------------

static uintptr_t bufferAlignment(BufferAlignBits alignBits)
{
    //*** from lwndocs
    //* Vertex bufferformat-specific,                               1B to 4B
    //* Uniform buffer                                              256B
    //* Shader Storage buffer                                       32B
    //* Transform feedback data buffer                              4B
    //* Transform feedback control buffer                           4B
    //* Index buffer    index size,                                 1B to 4B
    //* Indirect draw buffer                                        4B
    //* Counter reports                                             16B
    //* Texture (TextureTarget::TEXTURE_BUFFER) format-specific,    1B to 16B
    //***
    // ifs are ordered in such a way that this return safe alignment
    // when multiple access bits are set (which is possible given
    // access is a bitfield)
    if (alignBits & BUFFER_ALIGN_UNIFORM_BIT) {
        return 256;
    } else if (alignBits & BUFFER_ALIGN_IMAGE_BIT) {
        return 256; // use worst case assumption for Kepler (requiring 256B)
    } else if (alignBits & BUFFER_ALIGN_SHADER_STORAGE_BIT) {
        return 32; // use worst case (dvec4) in absence of format information
    } else if (alignBits & BUFFER_ALIGN_ZLWLL_SAVE_BIT) {
        return 32;
    } else if (alignBits & BUFFER_ALIGN_TRANSFORM_FEEDBACK_CONTROL_BIT) {
        return 32;
    } else if(alignBits & BUFFER_ALIGN_TEXTURE_BIT) {
        return 16; // use worst case in absence of format information
    } else if(alignBits & BUFFER_ALIGN_COUNTER_BIT) {
        return 16;
    } else if(alignBits & BUFFER_ALIGN_VERTEX_BIT) {
        return 16;
    } else if (alignBits & BUFFER_ALIGN_INDEX_BIT) {
        return 4; // use worst case in absence of format information
    } else if (alignBits & BUFFER_ALIGN_INDIRECT_BIT) {
        return 4;
    } else if (alignBits & BUFFER_ALIGN_TRANSFORM_FEEDBACK_BIT) {
        return 4;
    } else if (alignBits & BUFFER_ALIGN_COPY_READ_BIT) {
        return 4;
    } else if (alignBits & BUFFER_ALIGN_COPY_WRITE_BIT) {
        return 4;
    }
    return 512; // return GOB alignment to be on the safe side
}

Buffer::Buffer(LWNdevice* dev, BufferPool* pool, const void* data, size_t size, BufferAlignBits alignBits) :
    m_size(size)
{
    LWNbufferBuilder* bufferBuilder = new LWNbufferBuilder;
    lwnBufferBuilderSetDevice(bufferBuilder, dev);
    lwnBufferBuilderSetDefaults(bufferBuilder);
    lwnBufferBuilderSetStorage(bufferBuilder, pool->pool(), pool->alloc(size, bufferAlignment(alignBits)), size);

    m_buf = new LWNbuffer;
    lwnBufferInitialize(m_buf, bufferBuilder);

    // create persistent mapping
    m_ptr = lwnBufferMap(m_buf);

    if (data) {
        assert(m_ptr);   //if m_ptr is NULL, probably tried to map the GPU pool
        memcpy(m_ptr, data, size);
        //lwnBufferFlushMappedRange(m_buf, 0, size);  //Note: not needed for now since we only use the coherent CPU pool type
    }
    delete bufferBuilder;
}

Buffer::~Buffer()
{
    lwnBufferFinalize(m_buf);
    delete m_buf;
}

//--------------------------------------------------------------------

static void fillGridVertices(Vec3f* dst, int gridX, int gridY, const Vec2f& offs, const Vec2f& scale, float z)
{
    for (int y = 0; y < gridY; y++) {
        for (int x = 0; x < gridX; x++) {
            const float u = (float)x / (float)(gridX-1);
            const float v = (float)y / (float)(gridY-1);

            Vec2f loc = Vec2f((u + offs.x) * scale.x, (v + offs.y) * scale.y);
            const int didx = x + gridX*y;
            dst[didx] = Vec3f(loc.x, loc.y, z);
        }
    }
}

static void fillGridIndices(Vec3i* dst, int gridX, int gridY)
{
    int didx = 0;
    for (int y = 0; y < gridY - 1; y++) {
        for (int x = 0; x < gridX - 1; x++) {
            int base = gridX * y;
            Vec3i a(base + x, base + x + 1, base + x + gridX);
            Vec3i b(base + x + 1, base + x + gridX + 1, base + x + gridX);
            dst[didx++] = a;
            dst[didx++] = b;
        }
    }
}

static void fillFullscreelwertIdx(Vec3f* vLoc, Vec2f* vTexCoord, Vec3i* tris, float z)
{
    vLoc[0] = Vec3f(-1.f, -1.f, z);
    vLoc[1] = Vec3f(3.f, -1.f, z);
    vLoc[2] = Vec3f(-1.f, 3.f, z);
    tris[0] = Vec3i(0, 1, 2);

    if (vTexCoord) {
        vTexCoord[0] = Vec2f(0.f, 0.f);
        vTexCoord[1] = Vec2f(2.f, 0.f);
        vTexCoord[2] = Vec2f(0.f, 2.f);
    }
}

static void fillCircleVertIdx(Vec3f* loc, Vec3i* tris, int nSegments, const Vec2f& scale, float z)
{
    for (int i = 0; i < nSegments; i++) {
        float ang = (float)i / nSegments * 2.f * PI;

        float x = -sin(ang) * scale.x;
        float y =  cos(ang) * scale.y;
        loc[i] = Vec3f(x, y, z);
    }

    for (int i = 0; i < nSegments-2; i++) {
        tris[i] = Vec3i(0, i + 1, i + 2);
    }
}

Mesh* Mesh::createGrid(LWNdevice* dev, CoherentBufferPool* coherentPool, int gridX, int gridY, Vec2f offs, Vec2f scale, float z)
{
    Mesh* mesh = new Mesh();

    int numVerts = gridX*gridY;
    int numTris  = (gridX-1)*(gridY-1)*2;

    mesh->m_numTriangles = numTris;
    mesh->m_numVertices  = numVerts;
    mesh->m_vbo = new Buffer(dev, coherentPool, nullptr, numVerts*sizeof(Vec3f), BUFFER_ALIGN_VERTEX_BIT);
    mesh->m_vboAddress = mesh->m_vbo->address();

    Vec3f* gridLoc = (Vec3f*)mesh->m_vbo->ptr();
    fillGridVertices(gridLoc, gridX, gridY, offs, scale, z);

    mesh->m_ibo = new Buffer(dev, coherentPool, nullptr, numTris*sizeof(Vec3i), BUFFER_ALIGN_INDEX_BIT);
    mesh->m_iboAddress = mesh->m_ibo->address();

    Vec3i* tris = (Vec3i*)mesh->m_ibo->ptr();
    fillGridIndices(tris, gridX, gridY);

    return mesh;
}

Mesh* Mesh::createCircle(LWNdevice* dev, CoherentBufferPool* coherentPool, int nSegments, Vec2f scale, float z)
{
    Mesh* mesh = new Mesh();

    int numVerts = nSegments;
    int numTris  = nSegments - 2;

    mesh->m_numTriangles = numTris;
    mesh->m_numVertices  = numVerts;
    mesh->m_vbo = new Buffer(dev, coherentPool, nullptr, numVerts*sizeof(Vec3f), BUFFER_ALIGN_VERTEX_BIT);
    mesh->m_vboAddress = mesh->m_vbo->address();

    Vec3f* loc = (Vec3f*)mesh->m_vbo->ptr();
    assert(loc);

    mesh->m_ibo = new Buffer(dev, coherentPool, nullptr, numTris*sizeof(Vec3i), BUFFER_ALIGN_INDEX_BIT);
    mesh->m_iboAddress = mesh->m_ibo->address();

    Vec3i* tris = (Vec3i*)mesh->m_ibo->ptr();
    fillCircleVertIdx(loc, tris, nSegments, scale, z);

    return mesh;
}

Mesh* Mesh::createFullscreenTriangle(LWNdevice* dev, CoherentBufferPool* coherentPool, float z)
{
    Mesh* mesh = new Mesh();

    int numVerts = 3;
    int numTris  = 1;

    mesh->m_numTriangles = numTris;
    mesh->m_numVertices  = numVerts;

    mesh->m_vbo = new Buffer(dev, coherentPool, nullptr, numVerts*sizeof(Vec3f), BUFFER_ALIGN_VERTEX_BIT);
    mesh->m_vboAddress = mesh->m_vbo->address();
    Vec3f* vLoc = (Vec3f*)mesh->m_vbo->ptr();

    mesh->m_texVbo = new Buffer(dev, coherentPool, nullptr, numVerts*sizeof(Vec2f), BUFFER_ALIGN_VERTEX_BIT);
    mesh->m_texVboAddress = mesh->m_texVbo->address();
    Vec2f* texCoord = (Vec2f*)mesh->m_texVbo->ptr();

    mesh->m_ibo = new Buffer(dev, coherentPool, nullptr, numTris*sizeof(Vec3i), BUFFER_ALIGN_INDEX_BIT);
    mesh->m_iboAddress = mesh->m_ibo->address();

    Vec3i* tris = (Vec3i*)mesh->m_ibo->ptr();
    fillFullscreelwertIdx(vLoc, texCoord, tris, z);

    return mesh;
}

Mesh::~Mesh()
{
    delete m_vbo;
    delete m_texVbo;
    delete m_ibo;
}

OGLMesh::~OGLMesh()
{
    g_glDeleteBuffers(1, &m_ibo);
    g_glDeleteBuffers(1, &m_vbo);
}

void OGLMesh::allocGLBuffers(const Vec3f* vloc, const Vec3i* idx)
{
    g_glGenBuffers(1, &m_ibo);
    g_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    g_glBufferData(GL_ELEMENT_ARRAY_BUFFER, numTriangles() * sizeof(Vec3i),
                   idx, GL_STATIC_DRAW);

    g_glGenBuffers(1, &m_vbo);
    g_glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    g_glBufferData(GL_ARRAY_BUFFER, numVertices() * sizeof(Vec3f),
                   vloc, GL_STATIC_DRAW);

}

OGLMesh* OGLMesh::createGrid(int gridX, int gridY, Vec2f offs, Vec2f scale, float z)
{
    OGLMesh* mesh = new OGLMesh();

    int numVerts = gridX*gridY;
    int numTris  = (gridX-1)*(gridY-1)*2;

    mesh->m_numTriangles = numTris;
    mesh->m_numVertices  = numVerts;

    Vec3f* verts = new Vec3f[numVerts];
    fillGridVertices(verts, gridX, gridY, offs, scale, z);

    Vec3i* tris = new Vec3i[numTris];
    fillGridIndices(tris, gridX, gridY);

    mesh->allocGLBuffers(verts, tris);

    delete[] verts;
    delete[] tris;

    return mesh;
}

OGLMesh* OGLMesh::createFullscreenTriangle(float z)
{
    OGLMesh* mesh = new OGLMesh();

    int numVerts = 3;
    int numTris  = 1;

    mesh->m_numTriangles = numTris;
    mesh->m_numVertices  = numVerts;

    Vec3f* verts = new Vec3f[numVerts];
    Vec3i* tris = new Vec3i[numTris];

    fillFullscreelwertIdx(verts, nullptr, tris, z);

    mesh->allocGLBuffers(verts, tris);

    delete[] verts;
    delete[] tris;
    return mesh;
}

OGLMesh* OGLMesh::createCircle(int nSegments, Vec2f scale, float z)
{
    OGLMesh* mesh = new OGLMesh();

    int numVerts = nSegments;
    int numTris  = nSegments - 2;

    mesh->m_numTriangles = numTris;
    mesh->m_numVertices  = numVerts;

    Vec3f* verts = new Vec3f[numVerts];
    Vec3i* tris = new Vec3i[numTris];

    fillCircleVertIdx(verts, tris, nSegments, scale, z);

    mesh->allocGLBuffers(verts, tris);

    delete[] verts;
    delete[] tris;

    return mesh;
}


void OGLMesh::bindGeometryGL(GLuint vtxAttrLoc)
{
    g_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
    g_glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    g_glEnableVertexAttribArray(vtxAttrLoc);
    g_glVertexAttribPointer(vtxAttrLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
}

void VertexState::setAttribute(int n, LWNformat format, ptrdiff_t offset, int stream)
{
    if (n >= nAttribs) {
        for (int i = nAttribs; i <= n; i++) {
            resetAttribute(i);
        }
        nAttribs = n+1;
    }
    lwlwertexAttribStateSetFormat(&attribs[n], format, offset);
    lwlwertexAttribStateSetStreamIndex(&attribs[n], stream);
}

void VertexState::resetAttribute(int n)
{
    lwlwertexAttribStateSetDefaults(&attribs[n]);
}

void VertexState::setStream(int n, ptrdiff_t stride, int divisor /*= 0*/)
{
    if (n >= nStreams) {
        for (int i = nStreams; i <= n; i++) {
            resetStream(i);
        }
        nStreams = n+1;
    }
    lwlwertexStreamStateSetStride(&streams[n], stride);
    lwlwertexStreamStateSetDivisor(&streams[n], divisor);
}

void VertexState::resetStream(int n)
{
    lwlwertexStreamStateSetDefaults(&streams[n]);
}

void VertexState::bind(LWNcommandBuffer *cmdBuf)
{
    lwnCommandBufferBindVertexAttribState(cmdBuf, nAttribs, attribs);
    lwnCommandBufferBindVertexStreamState(cmdBuf, nStreams, streams);
}

