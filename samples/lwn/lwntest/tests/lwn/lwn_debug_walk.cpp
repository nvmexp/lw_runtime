/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_debug.cpp
//
// Basic testing of the LWN debug API.
//

#include <vector>

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#define DEBUG_MODE 1
#if DEBUG_MODE
    #define DEBUG_PRINT(x) printf x
#else
    #define DEBUG_PRINT(x)
#endif

using namespace lwn;

// ----------------------------------- LWNDebugObjectWalkTest --------------------------------------

class LWNDebugObjectWalkTest;
typedef void* (*InitTestAPObjectFunc)(Device *device, LWNDebugObjectWalkTest *wtest);
typedef void (*DestroyTestAPIObjectFunc)(Device *device, LWNDebugObjectWalkTest *wtest, void* object);

class LWNDebugObjectWalkTest {
    LWNdeviceFlagBits m_deviceFlags;
    MemoryPoolAllocator *m_pool;
    std::vector<bool> m_results;
    std::vector<void*> m_debugObjectsList;

    void TestDebugObjectTypeWalk(Device *device, Queue *queue, QueueCommandBuffer& queueCB,
        DebugObjectType::Enum type, InitTestAPObjectFunc initFunc, DestroyTestAPIObjectFunc destroyFunc);

    static void DebugWalkCallback(void* object, void *userParam);
    bool CheckObjectsExistInDebugLayer(Device *device, DebugObjectType::Enum type,
            void** objectsList, int objectsListSize);

public:
    OGTEST_CppMethods();
    LWNDebugObjectWalkTest();

    MemoryPoolAllocator* GetMemoryPoolAlloc(void)
    {
        return m_pool;
    }
};

// ------------------------- API object creation / destruction functions ---------------------------

struct TestWindowObject {
    Window window; // Must be first member.
    Texture *renderTargetTextures[2];
};

static void* TestCreateWindow(Device *device, LWNDebugObjectWalkTest *wtest)
{
    MemoryPoolAllocator* pool = wtest->GetMemoryPoolAlloc();
    TestWindowObject* window = new TestWindowObject;
    TextureBuilder tb;
    tb.SetDevice(device)
      .SetDefaults()
      .SetFlags(LWN_TEXTURE_FLAGS_DISPLAY_BIT)
      .SetTarget(TextureTarget::TARGET_2D)
      .SetFormat(Format::RGBA8)
      .SetSize2D(4, 4);
    for (int i = 0; i < 2; i++) {
        window->renderTargetTextures[i] = pool->allocTexture(&tb);
    }
    WindowBuilder wb;
        wb.SetDevice(device)
          .SetDefaults()
          .SetTextures(2, (lwn::objects::Texture* const*) window->renderTargetTextures);
    window->window.Initialize(&wb);
    return &window->window;
}

static void TestDestroyWindow(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    MemoryPoolAllocator* pool = wtest->GetMemoryPoolAlloc();
    TestWindowObject* window = (TestWindowObject*) object;
    window->window.Finalize();
    for (int i = 0; i < 2; i++) {
        if (window->renderTargetTextures[i]) {
            pool->freeTexture(window->renderTargetTextures[i]);
        }
    }
    delete window;
}

struct TestQueueObject {
    Queue queue; // Must be first member.
    char *queueMem;
};

static void* TestCreateQueue(Device *device, LWNDebugObjectWalkTest *wtest)
{
    TestQueueObject* queue = new TestQueueObject;
    queue->queueMem = (char *) AlignedStorageAlloc(0x80000, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);

    QueueBuilder qb;
    qb.SetDevice(device)
      .SetDefaults()
      .SetCommandMemorySize(0x20000)
      .SetComputeMemorySize(0)
      .SetQueueMemory(queue->queueMem, 0x80000);

    queue->queue.Initialize(&qb);
    return &queue->queue;
}

static void TestDestroyQueue(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    TestQueueObject* queue = (TestQueueObject*) object;
    queue->queue.Finalize();
    AlignedStorageFree(queue->queueMem);
    delete queue;
}

static void* TestCreateCommandBuffer(Device *device, LWNDebugObjectWalkTest *wtest)
{
    return device->CreateCommandBuffer();
}

static void TestDestroyCommandBuffer(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    CommandBuffer* cmdbuf = (CommandBuffer*) object;
    cmdbuf->Free();
}

static void* TestCreateMemoryPool(Device *device, LWNDebugObjectWalkTest *wtest)
{
    return device->CreateMemoryPoolWithFlags(NULL, 16*1024, MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED);
}

static void TestDestroyMemoryPool(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    MemoryPool* mempool = (MemoryPool*) object;
    mempool->Free();
}

static void* TestCreateSync(Device *device, LWNDebugObjectWalkTest *wtest)
{
    return device->CreateSync();
}

static void TestDestroySync(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    Sync* sync = (Sync*) object;
    sync->Free();
}

static void* TestCreateProgram(Device *device, LWNDebugObjectWalkTest *wtest)
{
    return device->CreateProgram();
}

static void TestDestroyProgram(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    Program* pgm = (Program*) object;
    pgm->Free();
}

struct TestTexturePoolObject {
    TexturePool texPool; // Must be first member.
    MemoryPool *pool;
};

static void* TestCreateTexturePool(Device *device, LWNDebugObjectWalkTest *wtest)
{
    TestTexturePoolObject *obj = new TestTexturePoolObject;
    obj->pool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::CPU_COHERENT);
    obj->texPool.Initialize(obj->pool, 0, 256);
    return &obj->texPool;
}

static void TestDestroyTexturePool(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    TestTexturePoolObject *obj = (TestTexturePoolObject*) object;
    obj->texPool.Finalize();
    obj->pool->Free();
}

struct TestSamplerPoolObject {
    SamplerPool samplerPool; // Must be first member.
    MemoryPool *pool;
};

static void* TestCreateSamplerPool(Device *device, LWNDebugObjectWalkTest *wtest)
{
    TestSamplerPoolObject *obj = new TestSamplerPoolObject;
    obj->pool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::CPU_COHERENT);
    obj->samplerPool.Initialize(obj->pool, 0, 256);
    return &obj->samplerPool;
}

static void TestDestroySamplerPool(Device *device, LWNDebugObjectWalkTest *wtest, void* object)
{
    TestSamplerPoolObject *obj = (TestSamplerPoolObject*) object;
    obj->samplerPool.Finalize();
    obj->pool->Free();
}

// ----------------------------------- LWNDebugObjectWalkTest --------------------------------------

LWNDebugObjectWalkTest::LWNDebugObjectWalkTest(void)
{
    m_deviceFlags = (LWNdeviceFlagBits) (LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT |
                                         LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_1_BIT);
    m_pool = NULL;
}
lwString LWNDebugObjectWalkTest::getDescription()
{
    lwStringBuf sb;
    sb <<
        "Test for debug layer object list walking feature.\n"
        "Sets up dummy objects, and then test that the debug layer walk\n"
        "reports them correctly.\n"
        "Passing test should look like all green, failures all red.\n";
    return sb.str();
}

int LWNDebugObjectWalkTest::isSupported()
{
    return lwogCheckLWNAPIVersion(53, 9) && g_lwnDeviceCaps.supportsDebugLayer;
}

void LWNDebugObjectWalkTest::initGraphics()
{
    lwnDefaultInitGraphics();
    DisableLWNObjectTracking();
}

void LWNDebugObjectWalkTest::DebugWalkCallback(void* object, void *userParam)
{
    LWNDebugObjectWalkTest* wtest = (LWNDebugObjectWalkTest*) userParam;
    wtest->m_debugObjectsList.push_back(object);
}

// Check that a list of objects are reported by the debug layer walk.
//
// We only check positively, ie. objects that we expect to be in the walk are in the walk.
//
// We don't test negatively here because random other objects may be created for other unrelated
// purposes; for example command buffers + backing memory may be created for the debug devices
// to work, completely unrelated and out of control of our tests here.
//
bool LWNDebugObjectWalkTest::CheckObjectsExistInDebugLayer(Device *device,
        DebugObjectType::Enum type, void** objectsList, int objectsListSize)
{
    // Read the debug layer objects.
    m_debugObjectsList.clear();
    device->WalkDebugDatabase(type, (lwn::objects::WalkDebugDatabaseCallbackFunc) DebugWalkCallback, this);

    for (int i = 0; i < objectsListSize; i++) {
        bool found = false;
        if (!objectsList[i]) {
            continue;
        }

        for (int j = 0; j < (int) m_debugObjectsList.size(); j++) {
            if (objectsList[i] == m_debugObjectsList[j]) {
                found = true;
                break;
            }
        }

        if (!found) {
            // We have found a missing object.
            DEBUG_PRINT(("\n---------------------- FAIL --------------------\n"));
            DEBUG_PRINT(("object type %d : %p missing\n", (int) type, objectsList[i]));
            for (int i = 0; i < (int) m_debugObjectsList.size(); i++) {
                DEBUG_PRINT(("  debug walk reported object: %p\n", m_debugObjectsList[i]));
            }
            DEBUG_PRINT(("---------------------------------------------\n\n"));
            return false;
        }
    }

    // Everything was found.
    return true;
}

void LWNDebugObjectWalkTest::TestDebugObjectTypeWalk(
        Device *device, Queue *queue, QueueCommandBuffer& queueCB,
        DebugObjectType::Enum type, InitTestAPObjectFunc initFunc, DestroyTestAPIObjectFunc destroyFunc)
{
    #define EXPECT_TRUE(cond) \
        if ((cond)) { \
            m_results.push_back(true); \
        } else { \
            DEBUG_PRINT(("Unexpected result %s on line %d\n", #cond, __LINE__)); \
            m_results.push_back(false); \
        }
    const int objectListSize = 128;
    void** objects = (void**) malloc(objectListSize * sizeof(void*));
    memset(objects, 0, objectListSize * sizeof(void*));

    int numTestObjects = 18;
    if (type == DebugObjectType::Enum::QUEUE) {
        // Don't create too many queue objects, there's a pretty small driver limit.
        numTestObjects = 3;
    }

    // Make some objects.
    for (int i = 0; i < numTestObjects; i++) {
        objects[i] = initFunc(device, this);
    }
    bool check = CheckObjectsExistInDebugLayer(device, type, objects, objectListSize);
    EXPECT_TRUE(check);

    // Delete all the objects.
    for (int i = 0; i < numTestObjects; i++) {
        destroyFunc(device, this, objects[i]);
        objects[i] = NULL;
    }
    check = CheckObjectsExistInDebugLayer(device, type, objects, objectListSize);
    EXPECT_TRUE(check);

    // Make some more objects.
    for (int i = 0; i < numTestObjects; i++) {
        objects[i] = initFunc(device, this);
    }
    check = CheckObjectsExistInDebugLayer(device, type, objects, objectListSize);
    EXPECT_TRUE(check);

    // Delete only half of the objects.
    for (int i = 0; i < numTestObjects; i+=2) {
        destroyFunc(device, this, objects[i]);
        objects[i] = NULL;
    }
    check = CheckObjectsExistInDebugLayer(device, type, objects, objectListSize);
    EXPECT_TRUE(check);

    // Delete the other half of the objects.
    for (int i = 1; i < numTestObjects; i+=2) {
        destroyFunc(device, this, objects[i]);
        objects[i] = NULL;
    }
    check = CheckObjectsExistInDebugLayer(device, type, objects, objectListSize);
    EXPECT_TRUE(check);

    free(objects);
    queue->Finish();
    #undef EXPECT_TRUE
}

static void LWNAPIENTRY TestDebugCallback(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
        DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam)
{
    DEBUG_PRINT(("lwnDebug: %s\n", (const char*) message));
}

void LWNDebugObjectWalkTest::doGraphics()
{
    DeviceState *testDevice = new DeviceState(m_deviceFlags);
    if (!testDevice || !testDevice->isValid()) {
        delete testDevice;
        DeviceState::SetDefaultActive();
        LWNFailTest();
        return;
    }
    Device *device = testDevice->getDevice();
    device->InstallDebugCallback(TestDebugCallback, NULL, LWN_TRUE);
    QueueCommandBuffer &queueCB = testDevice->getQueueCB();
    Queue *queue = testDevice->getQueue();
    testDevice->SetActive();
    m_results.clear();

    m_pool = new MemoryPoolAllocator(device, NULL, 0x60000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);
    assert(m_pool);

    // Run tests testing that various object type list walks operate correctly.
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::WINDOW, TestCreateWindow, TestDestroyWindow);
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::QUEUE, TestCreateQueue, TestDestroyQueue);
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::COMMAND_BUFFER, TestCreateCommandBuffer, TestDestroyCommandBuffer);
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::MEMORY_POOL, TestCreateMemoryPool, TestDestroyMemoryPool);
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::SYNC, TestCreateSync, TestDestroySync);
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::PROGRAM, TestCreateProgram, TestDestroyProgram);
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::TEXTURE_POOL, TestCreateTexturePool, TestDestroyTexturePool);
    TestDebugObjectTypeWalk(device, queue, queueCB, DebugObjectType::Enum::SAMPLER_POOL, TestCreateSamplerPool, TestDestroySamplerPool);

    // Manually clean up API resources that we created.
    device->InstallDebugCallback(TestDebugCallback, NULL, LWN_FALSE);
    delete m_pool;
    delete testDevice;
    DeviceState::SetDefaultActive();

    // Render the results to screen.
    QueueCommandBuffer &gqueueCB = *g_lwnQueueCB;
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    // Renders all green if everything passed, red if there was at least one
    // failure.
    bool failure = false;
    for (size_t i = 0; i < m_results.size(); i++) {
        if (!m_results[i]) {
            DEBUG_PRINT(("test %d failed!\n", (int) i));
            failure = true;
            break;
        }
    }
    LWNfloat color[] = { 0.0, 0.0, 0.0, 1.0 };
    if (failure) {
        color[0] = 1.0;
    } else {
        color[1] = 1.0;
    }
    gqueueCB.ClearColor(0, color, ClearColorMask::RGBA);

    Queue *gqueue = DeviceState::GetActive()->getQueue();
    gqueueCB.submit();
    gqueue->Finish();
}


void LWNDebugObjectWalkTest::exitGraphics()
{
    lwnDefaultExitGraphics();
    m_results.clear();
}

OGTEST_CppTest(LWNDebugObjectWalkTest, lwn_debug_walk, );
