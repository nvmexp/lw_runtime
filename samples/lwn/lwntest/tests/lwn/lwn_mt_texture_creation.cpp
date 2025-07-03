/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "string.h"


using namespace lwn;
using namespace lwn::dt;

#define LWN_DEBUG_LOG   0
static void log_output(const char *fmt, ...)
{
#if LWN_DEBUG_LOG
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif
}

#define BLOCK_DIV(N, S) (((N) + (S) - 1) / (S))
#define ROUND_UP(N, S) (BLOCK_DIV(N, S) * (S))
#define ROUND_DN(N, S) ((N) - ((N) % (S)))


// Multithreaded texture creation test
class LWN_MT_TextureCreation
{
public:
    Format fmt;
    bool compressible;

    LWN_MT_TextureCreation(Format fmt, bool compressible = false) : fmt(fmt), compressible(compressible) {}
    LWNTEST_CppMethods();
};


int LWN_MT_TextureCreation::isSupported() const
{
    return lwogCheckLWNAPIVersion(52,23);
}

lwString LWN_MT_TextureCreation::getDescription() const
{
    lwStringBuf sb;
    sb <<   "Multithreaded texture creation test. Spawns two additional\n"
            "threads, and then all three threads generate and upload textures\n"
            "simultaneously. The app then uses one thread to check that all\n"
            "threads uploaded the textures correctly before using the same\n"
            "procedure to spawn threads to finalize all of the textures.\n";
    return sb.str();
}


// Use an anonymous namespace to avoid potential namespace collisions.
namespace {

static const int NUM_THREADS = 3;
static const int TEXTURES_PER_THREAD = 1000;
static const int TEXTURES_IN_STAGING = 100;
static const int MAX_TEXTURE_DIM = 64;

struct SharedState
{
    Device *device;
    uint8_t *stageMem;
    MemoryPool *stagePool;
    uint8_t *stagePtr;
    uint8_t *textureMem;
    MemoryPool *texturePool;
    Format fmt;
    bool compressible;
    uint32_t textureStride;
    uint8_t *srcData;
    uint32_t srcDataRowStride;

    SharedState(Device *dev, Format fmt, bool compressible);
    ~SharedState();
};

struct ThreadState
{
    LWOGthread *handle;
    struct SharedState *shared;
    int threadNum;
    int coreNum;
    Queue *queue;
    lwnUtil::CompletionTracker *tracker;
    lwnUtil::QueueCommandBufferBase queueCBBase;
    lwnUtil::QueueCommandBuffer &queueCB;
    int textureSizeData[TEXTURES_PER_THREAD*2];
    Texture tex[TEXTURES_PER_THREAD];

    ThreadState(struct SharedState *shared, int threadNum, int coreNum);
    ~ThreadState();
    void CreateTextures();
    void FreeTextures();
    bool Check();

};

SharedState::SharedState(Device *dev, Format fmt, bool compressible)
    : device(dev), fmt(fmt), compressible(compressible)
{
    uint32_t n;

    // Initialize shared structure
    int fmtElementSize = 0;
    switch (fmt) {
        case Format::RGBA8:
            fmtElementSize = 4 * 1;
            break;
        default:
            fmtElementSize = 0;
            assert(false);
            break;
    }

    TextureBuilder tb;
    tb.SetDevice(device)
      .SetDefaults()
      .SetTarget(TextureTarget::TARGET_2D)
      .SetSize2D(MAX_TEXTURE_DIM, MAX_TEXTURE_DIM)
      .SetFormat(fmt)
      .SetFlags(compressible ? TextureFlags::COMPRESSIBLE : 0);

    textureStride = (uint32_t) ROUND_UP(tb.GetStorageSize(), tb.GetStorageAlignment());

    stageMem = (uint8_t*)AlignedStorageAlloc(textureStride * TEXTURES_IN_STAGING, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    stagePool = device->CreateMemoryPool(stageMem, textureStride * TEXTURES_IN_STAGING, MemoryPoolType::CPU_COHERENT);
    stagePtr = (uint8_t *) stagePool->Map();

    uint32_t texturePoolSize = textureStride * TEXTURES_PER_THREAD * NUM_THREADS;
    textureMem = (uint8_t*)AlignedStorageAlloc(texturePoolSize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    texturePool = device->CreateMemoryPool(textureMem, texturePoolSize, MemoryPoolType::GPU_ONLY);

    srcData = new uint8_t[textureStride];
    for (n=0; n<textureStride; n++) {
        srcData[n] = lwIntRand(0, 255);
    }
    srcDataRowStride = MAX_TEXTURE_DIM * fmtElementSize;
    for (n=0; n<TEXTURES_IN_STAGING; n++) {
        memcpy(stagePtr + (textureStride * n), srcData, textureStride);
    }
}

SharedState::~SharedState()
{
    delete [] srcData;
    texturePool->Free();
    AlignedStorageFree(textureMem);
    stagePool->Free();
    AlignedStorageFree(stageMem);
}

ThreadState::ThreadState(struct SharedState *shared, int threadNum, int coreNum)
    : handle(NULL), shared(shared), threadNum(threadNum), coreNum(coreNum), queueCB(queueCBBase)
{
    queue = shared->device->CreateQueue();
    for (uint32_t r=0; r<__GL_ARRAYSIZE(textureSizeData); r++) {
        textureSizeData[r] = lwIntRand(1, MAX_TEXTURE_DIM);
    }
    tracker = new lwnUtil::CompletionTracker(reinterpret_cast<LWNdevice *>(shared->device), 32);
    queueCBBase.init(reinterpret_cast<LWNdevice *>(shared->device), reinterpret_cast<LWNqueue *>(queue), tracker);
}

ThreadState::~ThreadState()
{
    queueCBBase.destroy();
    delete tracker;
    queue->Free();
}


void ThreadState::CreateTextures()
{
    TextureBuilder tb;
    tb.SetDevice(shared->device)
      .SetDefaults()
      .SetTarget(TextureTarget::TARGET_2D)
      .SetFormat(shared->fmt)
      .SetFlags(shared->compressible ? TextureFlags::COMPRESSIBLE : 0);

    BufferAddress stageAddr = shared->stagePool->GetBufferAddress();
    queueCB.SetCopyRowStride(shared->srcDataRowStride);

    CopyRegion region;
    region.xoffset = 0;
    region.yoffset = 0;
    region.zoffset = 0;
    region.depth = 1;
    uint32_t s = 0;
    uint32_t t;
    for (t=0; t<TEXTURES_PER_THREAD; t++) {
        int offset = ((t * NUM_THREADS) + threadNum) * shared->textureStride;
        int x = textureSizeData[s++];
        int y = textureSizeData[s++];

        tb.SetSize2D(x, y)
          .SetStorage(shared->texturePool, offset);
        tex[t].Initialize(&tb);

        region.width = x;
        region.height = y;
        queueCB.CopyBufferToTexture(stageAddr, &tex[t], NULL, &region, 0);
    }
    queueCB.submit();
    queue->Finish();
}

void ThreadState::FreeTextures()
{
    uint32_t t;
    for (t=0; t<TEXTURES_PER_THREAD; t++) {
        tex[t].Finalize();
    }
}

bool ThreadState::Check()
{
    // Single-threaded (writes to shared state)
    bool result = true;

    BufferAddress stageAddr = shared->stagePool->GetBufferAddress();
    queueCB.SetCopyRowStride(shared->srcDataRowStride);

    CopyRegion region;
    region.xoffset = 0;
    region.yoffset = 0;
    region.zoffset = 0;
    region.depth = 1;
    uint32_t s = 0;
    uint32_t stageIdx = 0;
    for (uint32_t t=0; t<TEXTURES_PER_THREAD; t++) {
        int x = textureSizeData[s++];
        int y = textureSizeData[s++];

        region.width = x;
        region.height = y;
        queueCB.CopyTextureToBuffer(&tex[t], NULL, &region, stageAddr + shared->textureStride * stageIdx, 0);
        stageIdx++;

        if (stageIdx == TEXTURES_IN_STAGING || t == TEXTURES_PER_THREAD-1) {
            queueCB.submit();
            queue->Finish();

            for (uint32_t c=0; c<stageIdx; c++) {
                bool correct = (memcmp(shared->srcData, shared->stagePtr + shared->textureStride * c, shared->textureStride) == 0);
                if (!correct) {
                    log_output("thread %u core %u texture %u failed\n", threadNum, coreNum, t);
                }
                result &= correct;
            }

            stageIdx = 0;
        }
    }

    return result;
}

}; // Close anonymous namespace


static void CreateTextures(void *args)
{
    struct ThreadState *state = (struct ThreadState *)args;
    state->CreateTextures();
}

static void FreeTextures(void *args)
{
    struct ThreadState *state = (struct ThreadState *)args;
    state->FreeTextures();
}


void LWN_MT_TextureCreation::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Initialize shared and per-thread data
    struct SharedState sharedState(device, fmt, compressible);
    struct ThreadState * threadState[NUM_THREADS];

#if defined(LW_HOS)
    uint64_t coreMask = 0;
    coreMask = lwogThreadGetAvailableCoreMask();
#endif

    uint32_t i;
    for (i=0; i<NUM_THREADS; i++) {
        // For HOS, we need to specify a specific core. For other platforms,
        // the core is just ignored.
        //
        // Note that threadState[0] will run on the main lwntest thread. For
        // the purposes of core selection here, we will assume that the main
        // thread is already running on the 0'th core.
        int core = i;
#if defined(LW_HOS)
        core = lwogThreadSelectCoreRoundRobin(i, coreMask);
#endif
        threadState[i] = new struct ThreadState(&sharedState, i, core);
    }

    // initialize textures
    for (i=1; i<NUM_THREADS; i++) {
#if defined(LW_HOS)
        threadState[i]->handle = lwogThreadCreateOnCore(CreateTextures, threadState[i], 0x20000, threadState[i]->coreNum);
#else
        threadState[i]->handle = lwogThreadCreate(CreateTextures, threadState[i], 0);
#endif
    }
    CreateTextures(threadState[0]);

    for (i=1; i<NUM_THREADS; i++) {
        lwogThreadWait(threadState[i]->handle);
    }

    // check the work (from a single thread)
    bool passed = true;
    for (i=0; i<NUM_THREADS; i++) {
        passed &= threadState[i]->Check();
    }

    // finalize textures
    for (i=1; i<NUM_THREADS; i++) {
#if defined(LW_HOS)
        threadState[i]->handle = lwogThreadCreateOnCore(FreeTextures, threadState[i], 0x20000, threadState[i]->coreNum);
#else
        threadState[i]->handle = lwogThreadCreate(FreeTextures, threadState[i], 0);
#endif
    }
    FreeTextures(threadState[0]);

    for (i=1; i<NUM_THREADS; i++) {
        lwogThreadWait(threadState[i]->handle);
    }

    // teardown
    for (i=0; i<NUM_THREADS; i++) {
        delete threadState[i];
    }

    queueCB.ClearColor(0, passed ? 0.0 : 1.0, passed ? 1.0 : 0.0, 0.0);
    queueCB.submit();
    queue->Finish();
}


OGTEST_CppTest(LWN_MT_TextureCreation, lwn_mt_texture_creation_rgba8, (Format::RGBA8));
OGTEST_CppTest(LWN_MT_TextureCreation, lwn_mt_texture_creation_rgba8_comp, (Format::RGBA8, true));

