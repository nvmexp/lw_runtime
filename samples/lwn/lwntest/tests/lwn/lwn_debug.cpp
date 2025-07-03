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

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "lwn/lwn_DeviceConstantsNX.h"

#include <list>
#include <memory>
#include <vector>

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) printf x
#else
    #define DEBUG_PRINT(x)
#endif

#define DEBUG_LOG_FAILURES 1

#ifndef ROUND_UP
    #define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#endif

#define LWN_DEBUG_DISABLE_CMDBUF_MEMPOOL_TRACKING_WAR_BUG_1704195 1

// Disabling this test for now, because there's no good way to create a test window that supports
// acquire / present, and the test just uses the global window, which works if the global device
// is the same as the test device, as is the case when -lwndebug is set.
//
#define LWN_DEBUG_DISABLE_WINDOW_PRESENT_INFLIGHT_MEMORY_TEST 1

// For bug 2041101, lwntest will crash on Windows 10 in the VRAM exhaustion test.
// This should be disabled until that behavior is fixed.
#define BUG_2041101_FIXED 0

using namespace lwn;

// List of every debug layer feature, and at which level they are enabled at.
enum LWNdebugLevelFeaturesTestEnum {
    DEBUG_FEATURE_ENABLED = 0,
    DEBUG_FEATURE_SKIP_BUSY_ENTRIES = 0,
    DEBUG_FEATURE_MINIMAL_HANDWRITTEN = 0,
    DEBUG_FEATURE_OBJECT_VALIDATION = 1,
    DEBUG_FEATURE_HANDWRITTEN = 1,
    DEBUG_FEATURE_OBJECT_FULL_VALIDATION = 2,
    DEBUG_FEATURE_MEMPOOL_OBJECT_TRACKING = 2,
    DEBUG_FEATURE_DRAW_TIME_VALIDATION = 3,
    DEBUG_FEATURE_IN_FLIGHT_CMDBUF_TRACKING = 4,
    DEBUG_FEATURE_VIRTUAL_MEMPOOL_MAPPING_TRACKING = 4,
    DEBUG_FEATURE_SCAN_MEMPOOL_FOR_BUFFERADDRESS = 4
};

// ----------------------------------- LWNDebugAPITest ------------------------------------------

#define MAX_SAVED_DEBUG_MESSAGES 128

class ExpectMessage;

class LWNDebugAPITest {
    friend class ExpectMessage;
    // Class to track and check the results of debug callbacks.  We use this
    // to record whether a particular callback function has been called or
    // not, and to check those results against expected behavior.
    static struct CallbackResults {
        int callback1Count;
        int callback2Count;
        lwString debugMessage1[MAX_SAVED_DEBUG_MESSAGES];
        lwString debugMessage2[MAX_SAVED_DEBUG_MESSAGES];

        enum Comparison {
            Equal = 0,
            GreaterOrEqual
        };  
        void reset()
        {
            callback1Count = 0;
            callback2Count = 0;
        }

        bool check(int expectedCallback1, int expectedCallback2, int line = 0, Comparison comparison = Equal)
        {
            bool result = true;

            const char* cmpStrings[] = {"", ">= "};
            const char* cmpString = cmpStrings[comparison];

            bool cmp1 = false, cmp2 = false;
            switch (comparison) {
            case Equal:
                cmp1 = callback1Count == expectedCallback1;
                cmp2 = callback2Count == expectedCallback2;
                break;
            case GreaterOrEqual:
                cmp1 = callback1Count >= expectedCallback1;
                cmp2 = callback2Count >= expectedCallback2;
                break;
            default:
                assert(!"Unknown comparison mode");
                break;
            }

            if (!cmp1) {
                DEBUG_PRINT(("status1 mismatch: expected %s%d got %d, on line %d\n",
                            cmpString, expectedCallback1, callback1Count, line));
                #if DEBUG_LOG_FAILURES
                    printf("status1 mismatch: expected %s%d got %d, on line %d\n",
                           cmpString, expectedCallback1, callback1Count, line);
                    for (int i = 0; i < callback1Count; i++) {
                        printf("                  message: %s\n", debugMessage1[i].c_str());
                    }
                #endif
                reset();
                return false;
            }
            if (!cmp2) {
                DEBUG_PRINT(("status2 mismatch: expected %s%d got %d, on line %d\n",
                            cmpString, expectedCallback2, callback2Count, line));
                #if DEBUG_LOG_FAILURES
                    printf("status1 mismatch: expected %s%d got %d, on line %d\n",
                           cmpString, expectedCallback2, callback2Count, line);
                    for (int i = 0; i < callback2Count; i++) {
                        printf("                  message: %s\n", debugMessage2[i].c_str());
                    }
                #endif
                reset();
                return false;
            }
            reset();
            return result;
        }
    } m_callbackResults;

    // Class to store the user info so the debug layer callback can restore the driver to a sensible
    // state after the debug layer detects a out-of-memory callback gone wrong.
    // Also used to report back to the caller when the out-of-memory callback happened.
    struct CommandBufferOOMTestRecovery {
        CommandBuffer *cmdbuf;
        MemoryPool* commandMemory;
        int commandMemorySize;
        void* controlMemory;
        int controlMemorySize;
        bool recovered;
        bool error;
    };

    struct TestResult {
        bool    result;
        int     linenum;
        TestResult(bool res, int line) : result(res), linenum(line) {}
    };
    int m_debugLevel;
    LWNdeviceFlagBits m_deviceFlags;
    TextureBuilder m_textureBuilder;
    std::vector<struct TestResult> m_results;

#define ADD_RESULT(val) m_results.push_back(TestResult((val), __LINE__))

    // Check the callbackthat requires a shader branch status after some commands and record the results in result vector.
    // Most of our tests use callback 1, but don't use callback 2, so make these default parameters.
    void expectDebugCallbacks(int linenum, int expectedCallback1, int expectedCallback2 = 0,
        CallbackResults::Comparison comparison = CallbackResults::Comparison::Equal);

    static void LWNAPIENTRY Callback1(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                      DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam);
    static void LWNAPIENTRY Callback2(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                      DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam);
    static void LWNAPIENTRY CallbackCmdbufOOMTest(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
        DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam);
    static void LWNAPIENTRY CallbackExpectMessage(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                                  DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam);

    static void LWNAPIENTRY CallbackCmdbufOOMDummy(objects::CommandBuffer *cmdBuf,
        CommandBufferMemoryEvent::Enum event, size_t minSize, void *callbackData);

    /// Test the "Set tracking infrastructure". Should detect missing calls to SetDefaults, SetDevice, and SetDevice with
    /// invalid parameters (if OBJECT_VALIDATION is enabled)
    template<typename Builder, typename Object>
    void TestSetTracker(Device *device, MemoryPool *mp);

    void TestDebugCallbacksInstall(void);
    void TestDebugCallbacksGenerated(Device *device, Queue *queue, QueueCommandBuffer& queueCB);
    void TestDebugCallbacksHandwritten(Device *device, Queue *queue, QueueCommandBuffer& queueCB);
    void TestDebugCallbacksDrawTimeValidations(Device *device, Queue *queue, QueueCommandBuffer& queueCB);
    void TestDebugCallbacksShaderSubroutines(Device *device, Queue *queue, QueueCommandBuffer& queueCB);
    void TestDebugCallbacksBindProgram(Device *device, Queue *queue, QueueCommandBuffer& queueCB);
    void TestDebugCallbacksThreaded(Device* device, Queue* queue, QueueCommandBuffer& queueCB);
    void TestDebugCallbacksQueueMemory(Device* device);
    void TestDebugCallbackQueueMemoryTestTmpQueue(const QueueBuilder *builder, int expectedCallbacks);
    void TestDebugCallbacksGpfifo(Device* device);
    void TestDebugCallbacksGLASMErrors(Device *device);
    void TestDebugCallbacksMemoryPoolErrors(Device *device);
    void TestDebugCallbacksMemoryPoolOverlaps(Device *device);
    void TestDebugCallbacksQueueOverlaps(Device *device);
    void TestDebugCallbacksQueueMempoolOverlaps(Device *device);

public:
    OGTEST_CppMethods();

    LWNDebugAPITest(int debugLevel);
};

LWNDebugAPITest::CallbackResults LWNDebugAPITest::m_callbackResults;

#if DEBUG_MODE
static void DebugPrintLWNDebugCallback(DebugCallbackSource::Enum& source, DebugCallbackType::Enum& type, LWNuint id,\
                                       LWNstring& message, void *userParam)
{
    DEBUG_PRINT(("lwnDebug: %s\n", (const char*) message));
}
#endif

LWNDebugAPITest::LWNDebugAPITest(int debugLevel)
    : m_debugLevel(debugLevel),
      m_deviceFlags((LWNdeviceFlagBits) LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT)
{
    switch (debugLevel) {
        case 0:
            m_deviceFlags = (LWNdeviceFlagBits) (m_deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_0_BIT);
            break;
        case 1:
            m_deviceFlags = (LWNdeviceFlagBits) (m_deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_1_BIT);
            break;
        case 2:
            m_deviceFlags = (LWNdeviceFlagBits) (m_deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT);
            break;
        case 3:
            m_deviceFlags = (LWNdeviceFlagBits) (m_deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_3_BIT);
            break;
        case 4:
            m_deviceFlags = (LWNdeviceFlagBits) (m_deviceFlags | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT);
            break;
        default:
            assert(!"Invalid debug level.");
            break;
    }
}

// We define two callback functions that simply record results to the
// CallbackResults structure.
void LWNAPIENTRY LWNDebugAPITest::Callback1(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                            DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam)
{
    m_callbackResults.debugMessage1[m_callbackResults.callback1Count] = message;
    m_callbackResults.callback1Count++;
#if DEBUG_MODE
    DebugPrintLWNDebugCallback(source, type, id, message, userParam);
#endif
}

void LWNAPIENTRY LWNDebugAPITest::Callback2(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                            DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam)
{
    m_callbackResults.debugMessage2[m_callbackResults.callback2Count] = message;
    m_callbackResults.callback2Count++;
#if DEBUG_MODE
    DebugPrintLWNDebugCallback(source, type, id, message, userParam);
#endif
}

void LWNAPIENTRY LWNDebugAPITest::CallbackCmdbufOOMTest(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                                        DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam)
{
    CommandBufferOOMTestRecovery* recoveryInfo = (CommandBufferOOMTestRecovery*) userParam;
    if (recoveryInfo->controlMemory) {
        recoveryInfo->cmdbuf->AddControlMemory(recoveryInfo->controlMemory, recoveryInfo->controlMemorySize);
    } else if (recoveryInfo->commandMemory) {
        recoveryInfo->cmdbuf->AddCommandMemory(recoveryInfo->commandMemory, 0, recoveryInfo->commandMemorySize);
    } else {
        assert(!"Cannot recover from command buffer OOM.");
    }
    recoveryInfo->recovered = true;
    Callback1(source, type, id, severity, message, &m_callbackResults.callback1Count);
}

void LWNAPIENTRY LWNDebugAPITest::CallbackCmdbufOOMDummy(objects::CommandBuffer *cmdBuf, CommandBufferMemoryEvent::Enum event,
                                                         size_t minSize, void *callbackData)
{
    DEBUG_PRINT(("lwnDebug: Command buffer OOM callback\n"));
    assert(callbackData);
    CommandBufferOOMTestRecovery* recoveryInfo = (CommandBufferOOMTestRecovery*) callbackData;
    if (recoveryInfo->recovered) {
        // We shouldn't be getting a callback OOM if we've recovered.
        recoveryInfo->error |= true;
    }
    if (event == CommandBufferMemoryEvent::OUT_OF_COMMAND_MEMORY && !recoveryInfo->commandMemory) {
        // Wrong recovery type.
        recoveryInfo->error |= true;
    }
    if (event == CommandBufferMemoryEvent::OUT_OF_CONTROL_MEMORY && !recoveryInfo->controlMemory) {
        // Wrong recovery type.
        recoveryInfo->error |= true;
    }
}

lwString LWNDebugAPITest::getDescription()
{
    lwStringBuf sb;
    sb <<
        "Basic tests exercising the LWN debug API.  Produces various error "
        "conditions and verifies that errors are detected and callbacks are "
        "made.  Displays a collection of cells in red or green depending on "
        "whether expected behavior oclwrs or not. Some tests expect given "
        "string literals and may not be forward-compatible with future "
        "LWN releases.\n";
    return sb.str();
}

int LWNDebugAPITest::isSupported()
{
    return lwogCheckLWNAPIVersion(40, 11) && g_lwnDeviceCaps.supportsDebugLayer;
}

void LWNDebugAPITest::initGraphics()
{
    lwnDefaultInitGraphics();
    DisableLWNObjectTracking();
}

#define EXPECT_DEBUG_CALLBACKS(...)   expectDebugCallbacks(__LINE__, __VA_ARGS__)

void LWNDebugAPITest::expectDebugCallbacks(int lineNum,
                                           int expectedCallback1,
                                           int expectedCallback2 /* = 0 */,
                                           CallbackResults::Comparison comparison /* = Equal */)
{
    const bool result = m_callbackResults.check(expectedCallback1, expectedCallback2, lineNum, comparison);
    m_results.push_back(TestResult(result, lineNum));
}

void LWNDebugAPITest::TestDebugCallbacksInstall()
{
    DeviceBuilder deviceBuilder;
    deviceBuilder.SetDefaults();
    Device tempDevice;
    tempDevice.Initialize(&deviceBuilder);
    // Use non-debug-layer entry points, since we are trying to test an incorrectly set up device.
    // Debug-layer entry points will be reset before "normal" lwn_debug tests are run.
    ReloadLWNEntryPoints(reinterpret_cast<LWNdevice *>(&tempDevice), true);

    // Attempting to install a debug callback on a NULL device
    lwnDeviceInstallDebugCallback(NULL, (PFNLWNDEBUGCALLBACKPROC)Callback1, &m_callbackResults.callback1Count, LWN_TRUE);
    EXPECT_DEBUG_CALLBACKS(1, 0);

    // NOT TESTED: Installing a debug callback when no devices have been initialized

    // Installing a debug callback on a device created without DEBUG_ENABLE
    tempDevice.InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);
    EXPECT_DEBUG_CALLBACKS(1, 0);
    tempDevice.Finalize();
}

void LWNDebugAPITest::TestDebugCallbacksGenerated(Device *device,
                                                  Queue *queue, QueueCommandBuffer& queueCB)
{
    // Create a GLSLC helper instead of using the global version
    // since the device may be a debug device depending on the
    // test variant.
    lwnTest::GLSLCHelper glslcHelper(device, 0x100000, g_glslcLibraryHelper, g_glslcHelperCache);
    MemoryPool *scratchMemPool = device->CreateMemoryPool(NULL, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, MemoryPoolType::GPU_ONLY);
    glslcHelper.SetShaderScratchMemory(scratchMemPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, queueCB);

    // Start by testing the installation/removal of callback filters.  For
    // this phase, we will be triggering errors by sending an invalid value
    // for the min filter of a dummy sampler object.
    SamplerBuilder samplerBuilder;
    samplerBuilder.SetDevice(device).SetDefaults();

    // No callbacks installed == no callbacks made.
    samplerBuilder.SetMinMagFilter(MinFilter::Enum(0x7343129A), MagFilter::LINEAR);
    EXPECT_DEBUG_CALLBACKS(0, 0);

    // Installing the first callback should cause it to trigger.
    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);
    samplerBuilder.SetMinMagFilter(MinFilter::Enum(0x7343129A), MagFilter::LINEAR);
    EXPECT_DEBUG_CALLBACKS(1, 0);

    // Installing the second callback should cause both to trigger.
    device->InstallDebugCallback(Callback2, &m_callbackResults.callback2Count, LWN_TRUE);
    samplerBuilder.SetMinMagFilter(MinFilter::Enum(0x7343129A), MagFilter::LINEAR);
    EXPECT_DEBUG_CALLBACKS(1, 1);

    // Installing the second callback again, should have no real effect.
    device->InstallDebugCallback(Callback2, &m_callbackResults.callback2Count, LWN_TRUE);
    samplerBuilder.SetMinMagFilter(MinFilter::Enum(0x7343129A), MagFilter::LINEAR);
    EXPECT_DEBUG_CALLBACKS(1, 1);

    // Removing the first callback should cause only the second to fire.
    device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
    samplerBuilder.SetMinMagFilter(MinFilter::Enum(0x7343129A), MagFilter::LINEAR);
    EXPECT_DEBUG_CALLBACKS(0, 1);

    // Removing the second callback should leave us with no callbacks firing
    // again.
    device->InstallDebugCallback(Callback2, NULL, LWN_FALSE);
    samplerBuilder.SetMinMagFilter(MinFilter::Enum(0x7343129A), MagFilter::LINEAR);
    EXPECT_DEBUG_CALLBACKS(0, 0);

    // At this point, we're not going to bother with multiple callbacks any
    // more.  Just enable one callback and be happy.
    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);

    // Check for invalid bits in a bitfield type (e.g., AccessBits).
    m_textureBuilder.SetDefaults();
    m_textureBuilder.SetFlags(TextureFlags(0xF0F0F0F0));
    EXPECT_DEBUG_CALLBACKS(1);

    // We should detect and report failed shader compilation.
    Program *pgm = device->CreateProgram();
    const char *dummyVS = "This is a vertex shader that has no prayer of compiling!";


    // Temporarily disable logging in the GLSLCHelper since these shaders are expected to fail
    // compilation and we don't want to print the info log as a debug message.
    GLSLCLogger * logger = glslcHelper.GetLogger();
    bool wasLoggerEnabled = logger->IsEnabled();
    logger->SetEnable(false);
    VertexShader vs(440);
    vs << dummyVS;
    glslcHelper.CompileAndSetShaders(pgm, vs);
    logger->SetEnable(wasLoggerEnabled);

    // No callback for GLSLC yet, so draw green.
    EXPECT_DEBUG_CALLBACKS(0, 0);
    pgm->Free();

    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);

    // We should detect invalid or NULL parameters passed to lwnDeviceRegisterTexturePool /
    // lwnDeviceRegisterSamplerPool.

    int texDescSize = 0, sampDescSize = 0;
    const int testPoolSz = 0x10000;
    device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE, &texDescSize);
    device->GetInteger(DeviceInfo::SAMPLER_DESCRIPTOR_SIZE, &sampDescSize);
    MemoryPool *testPool = device->CreateMemoryPool(NULL, testPoolSz, MemoryPoolType::CPU_COHERENT);

    LWNtexturePool texPool;
    LWNsamplerPool smpPool;
    LWNmemoryPool *cTestPool = reinterpret_cast<LWNmemoryPool *>(testPool);
    if (m_debugLevel >= DEBUG_FEATURE_OBJECT_VALIDATION) {
        lwnTexturePoolInitialize(&texPool, (LWNmemoryPool*) (uintptr_t)0xBEEFDEAD, 0, testPoolSz / texDescSize);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnSamplerPoolInitialize(&smpPool, (LWNmemoryPool*) (uintptr_t)0xBEEFDEAD, 0, testPoolSz / sampDescSize);
        EXPECT_DEBUG_CALLBACKS(1);
    }
    lwnTexturePoolInitialize((LWNtexturePool *) NULL, (LWNmemoryPool *) NULL, 0, testPoolSz / texDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnSamplerPoolInitialize((LWNsamplerPool *) NULL, (LWNmemoryPool *) NULL, 0, testPoolSz / sampDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnTexturePoolInitialize((LWNtexturePool *) NULL, (LWNmemoryPool *) NULL, 0, testPoolSz / texDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnSamplerPoolInitialize((LWNsamplerPool *) NULL, (LWNmemoryPool *) NULL, 0, testPoolSz / sampDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnTexturePoolInitialize(&texPool, (LWNmemoryPool*) NULL, 0, testPoolSz / texDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnSamplerPoolInitialize(&smpPool, (LWNmemoryPool*) NULL, 0, testPoolSz / sampDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnTexturePoolInitialize(&texPool, cTestPool, 1, testPoolSz / texDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnSamplerPoolInitialize(&smpPool, cTestPool, 1, testPoolSz / sampDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnTexturePoolInitialize(&texPool, cTestPool, 0x20000, 0x1000 / texDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnSamplerPoolInitialize(&smpPool, cTestPool, 0x20000, 0x1000 / sampDescSize);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnTexturePoolInitialize(&texPool, cTestPool, 0, 0);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnSamplerPoolInitialize(&smpPool, cTestPool, 0, 0);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnTexturePoolInitialize(&texPool, cTestPool, 0, (testPoolSz / texDescSize) + 10);
    EXPECT_DEBUG_CALLBACKS(1);
    lwnSamplerPoolInitialize(&smpPool, cTestPool, 0, (testPoolSz / sampDescSize) + 10);
    EXPECT_DEBUG_CALLBACKS(1);

    // Attempt to set zero dimension for texture builder should generate an error.
    TextureBuilder texb;
    texb.SetDevice(device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::RGBA8)
        .SetSize2D(640, 480);
    LWNuintptr tsz = texb.GetStorageSize();
    (void) tsz;
    EXPECT_DEBUG_CALLBACKS(0, 0);
    texb.SetSize2D(0, 480);
    EXPECT_DEBUG_CALLBACKS(1);
    texb.SetSize2D(640, 0);
    EXPECT_DEBUG_CALLBACKS(1);
    texb.SetSize1D(0);
    EXPECT_DEBUG_CALLBACKS(1);
    texb.SetSize3D(128, 128, 0);
    EXPECT_DEBUG_CALLBACKS(1);
    texb.SetLevels(0);
    EXPECT_DEBUG_CALLBACKS(1);

    // Test that setting invalid patch size results in error.
    if (m_debugLevel > DEBUG_FEATURE_SKIP_BUSY_ENTRIES) {
        queueCB.SetPatchSize(0);
        EXPECT_DEBUG_CALLBACKS(1);
        queueCB.SetPatchSize(33);
        EXPECT_DEBUG_CALLBACKS(1);
        queueCB.SetPatchSize(32);
    }
    queueCB.SetPatchSize(1);
    EXPECT_DEBUG_CALLBACKS(0);

    // Test that using out-of-range texture and sampler IDs are considered
    // legal by the debug layer. This is important because applications may
    // depend on these values as "unbound" values.

    // For our unbound (out-of-range) texture and sampler IDs, use
    // MAX_TEXTURE_POOL_SIZE-1 and MAX_SAMPLER_POOL_SIZE-1 respectively.
    // (Note that this will not work if the lwrrently bound texture or sampler
    // pool is the maximum possible size.)
    int unboundTextureID = 0, unboundSamplerID = 0;
    device->GetInteger(lwn::DeviceInfo::MAX_TEXTURE_POOL_SIZE, &unboundTextureID);
    device->GetInteger(lwn::DeviceInfo::MAX_SAMPLER_POOL_SIZE, &unboundSamplerID);
    unboundTextureID--;
    unboundSamplerID--;

    device->GetTexelFetchHandle(unboundTextureID);
    EXPECT_DEBUG_CALLBACKS(0, 0);

    TextureHandle unusedTexHandle = device->GetTextureHandle(unboundTextureID, unboundSamplerID);
    TextureHandle unusedTexHandles[] = {
        unusedTexHandle, unusedTexHandle, unusedTexHandle, unusedTexHandle,
        unusedTexHandle, unusedTexHandle, unusedTexHandle, unusedTexHandle,
        unusedTexHandle, unusedTexHandle, unusedTexHandle, unusedTexHandle,
        unusedTexHandle, unusedTexHandle, unusedTexHandle, unusedTexHandle
    };
    EXPECT_DEBUG_CALLBACKS(0, 0);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, unusedTexHandle);
    EXPECT_DEBUG_CALLBACKS(0, 0);
    queueCB.BindTextures(ShaderStage::FRAGMENT, 0, __GL_ARRAYSIZE(unusedTexHandles), unusedTexHandles);
    EXPECT_DEBUG_CALLBACKS(0, 0);

    TextureHandle unusedImgHandle = device->GetImageHandle(unboundTextureID);
    TextureHandle unusedImgHandles[] = {
        unusedImgHandle, unusedImgHandle, unusedImgHandle, unusedImgHandle,
        unusedImgHandle, unusedImgHandle, unusedImgHandle, unusedImgHandle
    };
    EXPECT_DEBUG_CALLBACKS(0, 0);
    queueCB.BindImage(ShaderStage::FRAGMENT, 0, unusedImgHandle);
    EXPECT_DEBUG_CALLBACKS(0, 0);
    queueCB.BindImages(ShaderStage::FRAGMENT, 0, __GL_ARRAYSIZE(unusedImgHandles), unusedImgHandles);
    EXPECT_DEBUG_CALLBACKS(0, 0);

    // Uninstall all callbacks so they don't affect future tests or test runs.
    device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
    device->InstallDebugCallback(Callback2, NULL, LWN_FALSE);

    queueCB.submit();
    queue->Finish();
    testPool->Free();
    scratchMemPool->Free();
}

// ----- SetTracking tests -----------------------------------------------------
typedef enum SetTrackingTest {
    SET_TRACKING_NO_SET_DEVICE = 0,
    SET_TRACKING_SET_ILWALID_DEVICE,
    SET_TRACKING_SET_NULL_DEVICE,
    SET_TRACKING_NO_SET_DEFAULTS,
    SET_TRACKING_HAPPY_PATH,

    SET_TRACKING_Count
} SetTrackingTest;

template<typename Builder>
static void SetTrackingCompleteSetup(MemoryPool *mp, Builder &builder) {
    static_assert(sizeof(Builder) != sizeof(Builder), "SetTrackingCompleteSetup must be specialized");
}

template<>
void SetTrackingCompleteSetup(MemoryPool *mp, BufferBuilder &builder) {
    builder.SetStorage(mp, 0, 1024);
}

template<>
void SetTrackingCompleteSetup(MemoryPool *mp, TextureBuilder &builder) {
    builder.SetFormat(Format::RGBA8);
    builder.SetTarget(TextureTarget::TARGET_2D);
    builder.SetSize2D(4, 4);
    builder.SetStorage(mp, 0);
}

template<>
void SetTrackingCompleteSetup(MemoryPool *mp, SamplerBuilder &builder) {
    // Nothing to do here
}

template<>
void SetTrackingCompleteSetup(MemoryPool *mp, MemoryPoolBuilder &builder) {
    builder.SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED |
                     MemoryPoolFlags::VIRTUAL);
    builder.SetStorage(NULL, 64*1024);
}

template<>
void SetTrackingCompleteSetup(MemoryPool *mp, QueueBuilder &builder) {
    // Nothing to do here
}

template<typename Builder, typename Object>
void LWNDebugAPITest::TestSetTracker(Device *device, MemoryPool *mp)
{
    Builder builder;
    Object obj;

    for (int i = 0; i < SET_TRACKING_Count; ++i) {
        SetTrackingTest test = (SetTrackingTest)i;
        if (test == SET_TRACKING_SET_ILWALID_DEVICE && m_debugLevel < DEBUG_FEATURE_OBJECT_VALIDATION) {
            continue;
        }
        memset(reinterpret_cast<void*>(&builder), (test == SET_TRACKING_SET_ILWALID_DEVICE) ? 0xFF : 0x00, sizeof(builder));
        if (test != SET_TRACKING_NO_SET_DEFAULTS) {
            builder.SetDefaults();
        }
        switch (test) {
        case SET_TRACKING_NO_SET_DEFAULTS:
        case SET_TRACKING_HAPPY_PATH:
            builder.SetDevice(device);
            break;
        case SET_TRACKING_SET_NULL_DEVICE:
            builder.SetDevice(nullptr);
            EXPECT_DEBUG_CALLBACKS(1);
            break;
        case SET_TRACKING_NO_SET_DEVICE:
        case SET_TRACKING_SET_ILWALID_DEVICE:
            // We don't want to overwrite our funky initialization
            break;
        case SET_TRACKING_Count:
            assert(test != SET_TRACKING_Count);
        }
        SetTrackingCompleteSetup(mp, builder);
        LWNboolean initialized = obj.Initialize(&builder);
        if (initialized) {
            obj.Finalize();
        }
        EXPECT_DEBUG_CALLBACKS((test == SET_TRACKING_HAPPY_PATH) ? 0 : 1);
    }
}
// -----------------------------------------------------------------------------

void LWNDebugAPITest::TestDebugCallbacksHandwritten(Device *device, Queue *queue,
                                                    QueueCommandBuffer& queueCB)
{
    LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);

    const int tpoolSz = 0x20000;
    MemoryPool *tpool = device->CreateMemoryPool(NULL, tpoolSz, MemoryPoolType::GPU_ONLY);
    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);

    // Check the physical pool compressible kind.
    {
        MemoryPoolFlags vPoolFlags = (MemoryPoolFlags::CPU_NO_ACCESS |
                                      MemoryPoolFlags::GPU_CACHED    |
                                      MemoryPoolFlags::VIRTUAL);
        MemoryPool *vPool = device->CreateMemoryPoolWithFlags(NULL, 0x20000, vPoolFlags);

        MemoryPoolFlags pPoolFlags = (MemoryPoolFlags::CPU_CACHED    |
                                      MemoryPoolFlags::GPU_NO_ACCESS |
                                      MemoryPoolFlags::PHYSICAL);
        MemoryPool *pPool = device->CreateMemoryPoolWithFlags(NULL, 0x20000, pPoolFlags);

        LWNstorageClass uncompressibleClass = 0;
        LWNstorageClass compressibleClass = 0;

        // Positive test, mapping the non-compressible phys pool with non-compressible texture passes the test.
        TextureBuilder tb;
        tb.SetDevice(device)
          .SetDefaults()
          .SetTarget(TextureTarget::TARGET_2D)
          .SetSize2D(64, 64)
          .SetFormat(Format::RGBA8);

        MappingRequest req = { 0, };
        req.physicalPool = pPool;
        req.virtualOffset = 0;
        req.physicalOffset = 0;
        req.size = 0x10000;
        uncompressibleClass = tb.GetStorageClass();
        req.storageClass = uncompressibleClass;
        vPool->MapVirtual(1, &req);
        EXPECT_DEBUG_CALLBACKS(0);

        // Test that mapping the non-compressible phys pool with compressible texture generates debug message.
        tb.SetFlags(TextureFlags::COMPRESSIBLE);
        compressibleClass = tb.GetStorageClass();
        req.storageClass = compressibleClass;
        // On Windows, there is no support for sparse texture compression on KMD for kepler Lwdqro GPUs.
        bool compressionNotSupported = (compressibleClass == uncompressibleClass);
        vPool->MapVirtual(1, &req);
        EXPECT_DEBUG_CALLBACKS(compressionNotSupported ? 0 : 1);

        vPool->Free(); // Finalizing the virtual pool first
        pPool->Free();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Check the max/min number of textures that can be used in LWNwindow objects,
    {
        LWNint maxTexPerWindow;
        LWNint minTexPerWindow;
        device->GetInteger(DeviceInfo::MAX_TEXTURES_PER_WINDOW, &maxTexPerWindow);
        device->GetInteger(DeviceInfo::MIN_TEXTURES_PER_WINDOW, &minTexPerWindow);
        if (maxTexPerWindow != 4 || minTexPerWindow != 2) {
            ADD_RESULT(false);
        }

        const int bufferNum = maxTexPerWindow + 1;
        Texture **renderTargetTextures = new Texture*[bufferNum];
        MemoryPoolAllocator pool(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

        TextureBuilder tb;
        tb.SetDevice(device)
          .SetDefaults()
          .SetFlags(LWN_TEXTURE_FLAGS_DISPLAY_BIT)
          .SetTarget(TextureTarget::TARGET_2D)
          .SetFormat(Format::RGBA8)
          .SetSize2D(64, 64);

        for (int i = 0; i < bufferNum; i++) {
            renderTargetTextures[i] = pool.allocTexture(&tb);
        }

        WindowBuilder wb;
        wb.SetDevice(device)
          .SetDefaults();

        if (lwogCheckLWNAPIVersion(53, 209)) {
            // Check that querying an invalid texture ID from builder is generating an error.
            wb.GetTexture(1);
            EXPECT_DEBUG_CALLBACKS(1);
        }

        // Positive check.
        wb.SetTextures(maxTexPerWindow, renderTargetTextures);
        wb.SetTextures(minTexPerWindow, renderTargetTextures);
        EXPECT_DEBUG_CALLBACKS(0);

        // Test that invalid buffer number causes error.
        wb.SetTextures(bufferNum, renderTargetTextures);
        EXPECT_DEBUG_CALLBACKS(1);

        wb.SetTextures(minTexPerWindow - 1, renderTargetTextures);
        EXPECT_DEBUG_CALLBACKS(1);

        for (int i = 0; i < bufferNum; i++) {
            if (renderTargetTextures[i]) {
                pool.freeTexture(renderTargetTextures[i]);
                renderTargetTextures[i] = NULL;
            }
        }
        delete[] renderTargetTextures;
    }

    // Test for Bug 1955817, checking for BufferAddress.
    if (m_debugLevel >= DEBUG_FEATURE_HANDWRITTEN) {
        DEBUG_PRINT(("--Test for Bug 1955817 begin--"));

        MemoryPoolAllocator pool(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);
        CopyRegion bufferRegion = { 0, 0, 0, 1, 1, 1 };

        TextureBuilder textureBuilder;
        textureBuilder.SetDevice(device)
                      .SetDefaults()
                      .SetTarget(TextureTarget::TARGET_2D)
                      .SetFormat(Format::RGBA8)
                      .SetSize2D(64, 64);

        Texture *texture = pool.allocTexture(&textureBuilder);

        BufferBuilder bufferBuilder;
        bufferBuilder.SetDevice(device);
        bufferBuilder.SetDefaults();
        Buffer *buffer = pool.allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, 0x20);
        BufferAddress bufferAddr = buffer->GetAddress();

        // make dangling BufferAddress by free-ing memory pool after having the bufferAddress.
        LWNmemoryPool *danglingPool;
        danglingPool = lwnDeviceCreateMemoryPool(cdevice, NULL, 8192, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

        BufferAddress danglingAddr = lwnMemoryPoolGetBufferAddress(danglingPool);
        lwnMemoryPoolFree(danglingPool);

// API with passing one bufferAddress, without size.
/*
 *  Pass the following bufferAddress
 *  bufferAddr                                                          -> no error
 *  bufferAddr + 1024 (valid but not starting from memory pool head)    -> no error
 *  0                                                                   -> error
 *  -16 (address huge enough to be out of range)                        -> error
 *  danglingAddr                                                        -> error
 */
#define LWN_TESTBUFFERADDRESS_DRAWELEMENT(NAME, addr, result) \
    {\
        queueCB.DrawElements(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, 4, LWNbufferAddress(addr));\
        if (this->m_callbackResults.callback1Count != result) {\
            printf("%s: addr = %s\n", NAME, #addr);\
        }\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_DRAWELEMENTBASEVERTEX(NAME, addr, result) \
    {\
        queueCB.DrawElementsBaseVertex(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, 4, LWNbufferAddress(addr), 0);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_DRAWELEMENTSINSTANCED(NAME, addr, result) \
    {\
        queueCB.DrawElementsInstanced(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, 4, LWNbufferAddress(addr), 1, 1, 1);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_DRAWARRAYSINDIRECT(NAME, addr, result) \
    {\
        queueCB.DrawArraysIndirect(DrawPrimitive::TRIANGLE_STRIP, LWNbufferAddress(addr));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_DISPATCHCOMPUTEINDIRECT(NAME, addr, result) \
    {\
        queueCB.DispatchComputeIndirect(LWNbufferAddress(addr));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_COPYBUFFERTOTEXTURE(NAME, addr, result) \
    {\
        queueCB.CopyBufferToTexture(LWNbufferAddress(addr), texture, NULL, &bufferRegion, CopyFlags::NONE);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_COPYTEXTURETOBUFFER(NAME, addr, result) \
    {\
        queueCB.CopyTextureToBuffer(texture, NULL, &bufferRegion, LWNbufferAddress(addr), CopyFlags::NONE);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(TESTNAME) \
    {\
        LWN_TESTBUFFERADDRESS_##TESTNAME(#TESTNAME, bufferAddr, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(#TESTNAME, bufferAddr + 1024, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(#TESTNAME, 0, 1);\
        if (m_debugLevel >= DEBUG_FEATURE_SCAN_MEMPOOL_FOR_BUFFERADDRESS) {\
            LWN_TESTBUFFERADDRESS_##TESTNAME(#TESTNAME, -16, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(#TESTNAME, danglingAddr, 1);\
        }\
        queue->Flush();\
    }

        LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(DRAWELEMENT);
        LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(DRAWELEMENTBASEVERTEX);
        LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(DRAWELEMENTSINSTANCED);
        LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(DRAWARRAYSINDIRECT);
        LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(DISPATCHCOMPUTEINDIRECT);
        LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(COPYBUFFERTOTEXTURE);
        LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE(COPYTEXTURETOBUFFER);


#undef LWN_TESTBUFFERADDRESS_ONEADDR_NOSIZE
#undef LWN_TESTBUFFERADDRESS_COPYTEXTURETOBUFFER
#undef LWN_TESTBUFFERADDRESS_COPYBUFFERTOTEXTURE
#undef LWN_TESTBUFFERADDRESS_DISPATCHCOMPUTEINDIRECT
#undef LWN_TESTBUFFERADDRESS_DRAWARRAYSINDIRECT
#undef LWN_TESTBUFFERADDRESS_DRAWELEMENTSINSTANCED
#undef LWN_TESTBUFFERADDRESS_DRAWELEMENTBASEVERTEX
#undef LWN_TESTBUFFERADDRESS_DRAWELEMENT


// API with passing two bufferAddress, without size.
/*
 *  Pass the following bufferAddress
 *  [bufferAddr, bufferAddr]                                            -> no error
 *  [bufferAddr + 1024, bufferAddr + 2048] (valid but not starting from memory pool head)
 *                                                                      -> no error
 *
 *  [0, 0] (0 for invalid buffer address)                               -> error on first parameter
 *  [bufferAddr, 0]                                                     -> error on second parameter
 *  [0, bufferAddr]                                                     -> error on first parameter
 *
 *  [-4, -4] (-4 for huge enough value to be invalid bufferAddress)     -> error on first parameter
 *  [bufferAddr, -4]                                                    -> error on second parameter
 *  [-4, bufferAddr]                                                    -> error on first parameter
 *
 *  [danglingAddr, danglingAddr]                                        -> error on first parameter
 *  [bufferAddr, danglingAddr]                                          -> error on second parameter
 *  [danglingAddr, bufferAddr]                                          -> error on first parameter
 */
// Two Address, No size
#define LWN_TESTBUFFERADDRESS_DRAWELEMENTSINDIRECT(addr1,addr2,result) \
    {\
        queueCB.DrawElementsIndirect(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT,\
                                     LWNbufferAddress(addr1), LWNbufferAddress(addr2));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_MULTIDRAWARRAYSINDIRECTCOUNT(addr1, addr2, result) \
    {\
        queueCB.MultiDrawArraysIndirectCount(DrawPrimitive::TRIANGLE_STRIP,\
                                             LWNbufferAddress(addr1), LWNbufferAddress(addr2), 1, 1);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_COPYBUFFERTOBUFFER(addr1, addr2, result) \
    {\
        queueCB.CopyBufferToBuffer(LWNbufferAddress(addr1), LWNbufferAddress(addr2), 1, CopyFlags::NONE);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_TWOADDR_NOSIZE(TESTNAME) \
    {\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, bufferAddr, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr + 1024, bufferAddr + 2048, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(0, 0, 1);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, 0, 1);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(0, bufferAddr, 1);\
        if (m_debugLevel >= DEBUG_FEATURE_SCAN_MEMPOOL_FOR_BUFFERADDRESS) {\
            LWN_TESTBUFFERADDRESS_##TESTNAME(-4, -4, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, -4, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(-4, bufferAddr, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(danglingAddr, danglingAddr, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, danglingAddr, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(danglingAddr, bufferAddr, 1);\
        }\
        queue->Flush();\
    }


        LWN_TESTBUFFERADDRESS_TWOADDR_NOSIZE(DRAWELEMENTSINDIRECT);
        LWN_TESTBUFFERADDRESS_TWOADDR_NOSIZE(MULTIDRAWARRAYSINDIRECTCOUNT);
        LWN_TESTBUFFERADDRESS_TWOADDR_NOSIZE(COPYBUFFERTOBUFFER);


#undef LWN_TESTBUFFERADDRESS_TWOADDR_NOSIZE
#undef LWN_TESTBUFFERADDRESS_COPYBUFFERTOBUFFER
#undef LWN_TESTBUFFERADDRESS_MULTIDRAWARRAYSINDIRECTCOUNT
#undef LWN_TESTBUFFERADDRESS_DRAWELEMENTSINDIRECT


// API with passing three bufferAddress, without size.
#define LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(addr1, addr2, addr3, result) \
    {\
        queueCB.MultiDrawElementsIndirectCount(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT,\
                                               LWNbufferAddress(addr1), LWNbufferAddress(addr2), LWNbufferAddress(addr3), 1, 1);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
        {
            LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, bufferAddr, bufferAddr, 0);
            LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr + 1024, bufferAddr +2048, bufferAddr + 3096, 0);

            if (m_debugLevel >= 1) {
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(0, 0, 0, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(0, 0, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(0, bufferAddr, 0, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(0, bufferAddr, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, 0, 0, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, 0, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, bufferAddr, 0, 1);
            }
            if (m_debugLevel >= DEBUG_FEATURE_SCAN_MEMPOOL_FOR_BUFFERADDRESS) {
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(-4, -4, -4, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(-4, -4, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(-4, bufferAddr, -4, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(-4, bufferAddr, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, -4, -4, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, -4, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, bufferAddr, -4, 1);

                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(danglingAddr, danglingAddr, danglingAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(danglingAddr, danglingAddr, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(danglingAddr, bufferAddr, danglingAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(danglingAddr, bufferAddr, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, danglingAddr, danglingAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, danglingAddr, bufferAddr, 1);
                LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT(bufferAddr, bufferAddr, danglingAddr, 1);
            }
            queue->Flush();
        }


#undef LWN_TESTBUFFERADDRESS_MULTIDRAWELEMENTSINDIRECTCOUNT


// API with passing one bufferAddress, with fixed size defined.
/*
 *  Pass the following bufferAddress
 *  bufferAddr                                                          -> no error
 *  bufferAddr + 1024 (valid but not starting from memory pool head)    -> no error
 *  0                                                                   -> error
 *  -fixsize     (cause bufferEnd address to overflow)                  -> error
 *  -(2*fixsize) (huge value which bufferEnd will not overflow)         -> error
 *  danglingAddr                                                        -> error
 */
#define LWN_TESTBUFFERADDRESS_REPORTCOUNTER(addr, result) \
    {\
        queueCB.ReportCounter(CounterType::TIMESTAMP, LWNbufferAddress(addr));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_REPORTVALUE(addr, result) \
    {\
        queueCB.ReportValue(1, LWNbufferAddress(addr));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_SETRENDERENABLECONDITIONAL(addr, result) \
    {\
        queueCB.SetRenderEnableConditional(ConditionalRenderMode::RENDER_IF_NOT_EQUAL, LWNbufferAddress(addr));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_ONEADDR_FIXSIZE(TESTNAME,fixsize) \
    {\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr + 1024, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(0, 1);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(-(fixsize), 1);\
        if (m_debugLevel >= DEBUG_FEATURE_SCAN_MEMPOOL_FOR_BUFFERADDRESS) {\
            LWN_TESTBUFFERADDRESS_##TESTNAME(-(2*(fixsize)), 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(danglingAddr, 1);\
        }\
        queue->Flush();\
    }


        LWN_TESTBUFFERADDRESS_ONEADDR_FIXSIZE(REPORTCOUNTER, 16);
        LWN_TESTBUFFERADDRESS_ONEADDR_FIXSIZE(REPORTVALUE, 16);
        LWN_TESTBUFFERADDRESS_ONEADDR_FIXSIZE(SETRENDERENABLECONDITIONAL, 32);


#undef LWN_TESTBUFFERADDRESS_ONEADDR_FIXSIZE
#undef LWN_TESTBUFFERADDRESS_SETRENDERENABLECONDITIONAL
#undef LWN_TESTBUFFERADDRESS_REPORTVALUE
#undef LWN_TESTBUFFERADDRESS_REPORTCOUNTER


// API with passing one bufferAddress, and one size.
/*
 *  Pass the following [bufferAddress, size] pair.
 *  [0, 0]                                                              -> no error
 *  [bufferAddr, 1024]                                                  -> no error
 *  [bufferAddr + 1024, 1024] (valid address but not starting from memory pool head)
 *                                                                      -> no error
 *  [danglingAddr, 0]                                                   -> no error
 *  [-1024, 0] (bufferAddress out of valid range)                       -> no error
 *
 *  [-1024, 2048] (cause bufferEnd address to overflow)                 -> error
 *
 *  [danglingAddr, 1024] (bad address with size != 0)                   -> error
 *  [-2048, 1024]        (bad address with size != 0, bufferEnd will not oveflow)
 *                                                                      -> error
 *
 *  [0, 1024] (bufferAddr = 0, size != 0, result differs according to API)
 *                                                                      -> depends on API.
 */
#define LWN_TESTBUFFERADDRESS_CLEARBUFFER(addr, size, result) \
    {\
        queueCB.ClearBuffer(LWNbufferAddress(addr), (size), 1);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }
#define LWN_TESTBUFFERADDRESS_SAVEZLWLLDATA(addr, size, result) \
    {\
        queueCB.SaveZLwllData(LWNbufferAddress(addr), (size));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_RESTOREZLWLLDATA(addr, size, result) \
    {\
        queueCB.RestoreZLwllData(LWNbufferAddress(addr), (size));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_BINDVERTEXBUFFER(addr, size, result) \
    {\
        queueCB.BindVertexBuffer(0, LWNbufferAddress(addr), (size));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_BINDUNIFORMBUFFER(addr, size, result) \
    {\
        queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, LWNbufferAddress(addr), (size));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_BINDSTORAGEBUFFER(addr, size, result) \
    {\
        queueCB.BindStorageBuffer(ShaderStage::VERTEX, 0, LWNbufferAddress(addr), (size));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_BINDTRANSFORMFEEDBACKBUFFER(addr, size, result) \
    {\
        queueCB.BindTransformFeedbackBuffer(0, LWNbufferAddress(addr), (size));\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(TESTNAME) \
    {\
        LWN_TESTBUFFERADDRESS_##TESTNAME(0, 0, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, 1024, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr + 1024, 1024, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(danglingAddr, 0, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(-1024, 0, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(-1024, 2048, 1);\
        if (m_debugLevel >= DEBUG_FEATURE_SCAN_MEMPOOL_FOR_BUFFERADDRESS) {\
            LWN_TESTBUFFERADDRESS_##TESTNAME(danglingAddr, 1024, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(-2048, 1024, 1);\
        }\
        queue->Flush();\
    }

        LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(CLEARBUFFER);
        LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(SAVEZLWLLDATA);
        LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(RESTOREZLWLLDATA);
        LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(BINDVERTEXBUFFER);
        LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(BINDUNIFORMBUFFER);
        LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(BINDSTORAGEBUFFER);
        LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE(BINDTRANSFORMFEEDBACKBUFFER);

        if (m_debugLevel >= 1) {
            // Test Exceptional case. If Buffer=0,Size!=0, TransformFeedbackBuffer should not return error, others will.
            LWN_TESTBUFFERADDRESS_CLEARBUFFER(0, 1024, 1);
            LWN_TESTBUFFERADDRESS_SAVEZLWLLDATA(0, 1024, 1);
            LWN_TESTBUFFERADDRESS_RESTOREZLWLLDATA(0, 1024, 1);
            LWN_TESTBUFFERADDRESS_BINDVERTEXBUFFER(0, 1024, 1);
            LWN_TESTBUFFERADDRESS_BINDUNIFORMBUFFER(0, 1024, 1);
            LWN_TESTBUFFERADDRESS_BINDSTORAGEBUFFER(0, 1024, 1);
            LWN_TESTBUFFERADDRESS_BINDTRANSFORMFEEDBACKBUFFER(0, 1024, 0);       // This case is exception
        }


#undef LWN_TESTBUFFERADDRESS_ONEADDR_ONESIZE
#undef LWN_TESTBUFFERADDRESS_BINDTRANSFORMFEEDBACKBUFFER
#undef LWN_TESTBUFFERADDRESS_BINDSTORAGEBUFFER
#undef LWN_TESTBUFFERADDRESS_BINDVERTEXBUFFER
#undef LWN_TESTBUFFERADDRESS_BINDUNIFORMBUFFER
#undef LWN_TESTBUFFERADDRESS_RESTOREZLWLLDATA
#undef LWN_TESTBUFFERADDRESS_SAVEZLWLLDATA
#undef LWN_TESTBUFFERADDRESS_CLEARBUFFER


// API passing (bufferAddress, size) pairs using bufferRange.
/*
 *  Pass the following [bufferAddress, size] pair.
 *  [0, 0]                                                              -> no error
 *  [bufferAddr, 1024]                                                  -> no error
 *  [bufferAddr + 1024, 1024] (valid address but not starting from memory pool head)
 *                                                                      -> no error
 *  [danglingAddr, 0]                                                   -> no error
 *  [-1024, 0] (bufferAddress out of valid range)                       -> no error
 *
 *  [-1024, 2048] (cause bufferEnd address to overflow)                 -> error
 *
 *  [danglingAddr, 1024] (bad address with size != 0)                   -> error
 *  [-2048, 1024]        (bad address with size != 0, bufferEnd will not oveflow)
 *                                                                      -> error
 *
 *  [0, 1024] (bufferAddr = 0, size != 0, result differs according to API)
 *                                                                      -> depends on API.
 * ================
 * Test for number of valid BufferRange to be:
 *  0                                                                   -> no error
 *  -1                                                                  -> error
 */
#define LWN_TESTBUFFERADDRESS_KICK_BINDVERTEXBUFFERS(count, result) \
    {\
        queueCB.BindVertexBuffers(0, (count), bufferRange);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }\

#define LWN_TESTBUFFERADDRESS_BINDVERTEXBUFFERS(addr, bufsize, result) \
    {\
        bufferRange[0].address = LWNbufferAddress(addr);\
        bufferRange[0].size = LWNuint64(bufsize);\
        LWN_TESTBUFFERADDRESS_KICK_BINDVERTEXBUFFERS(1, result);\
    }


#define LWN_TESTBUFFERADDRESS_KICK_BINDUNIFORMBUFFERS(count, result) \
    {\
        queueCB.BindUniformBuffers(ShaderStage::VERTEX, 0, (count), bufferRange);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_BINDUNIFORMBUFFERS(addr, bufsize, result) \
    {\
        bufferRange[0].address = LWNbufferAddress(addr);\
        bufferRange[0].size = LWNuint64(bufsize);\
        LWN_TESTBUFFERADDRESS_KICK_BINDUNIFORMBUFFERS(1, result);\
    }


#define LWN_TESTBUFFERADDRESS_KICK_BINDTRANSFORMFEEDBACKBUFFERS(count, result) \
    {\
        queueCB.BindTransformFeedbackBuffers(0, (count), bufferRange);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_BINDTRANSFORMFEEDBACKBUFFERS(addr, bufsize, result) \
    {\
        bufferRange[0].address = LWNbufferAddress(addr);\
        bufferRange[0].size = LWNuint64(bufsize);\
        LWN_TESTBUFFERADDRESS_KICK_BINDTRANSFORMFEEDBACKBUFFERS(1, result);\
    }


#define LWN_TESTBUFFERADDRESS_KICK_BINDSTORAGEBUFFERS(count, result) \
    {\
        queueCB.BindStorageBuffers(ShaderStage::VERTEX, 0, (count), bufferRange);\
        EXPECT_DEBUG_CALLBACKS(result);\
    }

#define LWN_TESTBUFFERADDRESS_BINDSTORAGEBUFFERS(addr, bufsize, result) \
    {\
        bufferRange[0].address = LWNbufferAddress(addr);\
        bufferRange[0].size = LWNuint64(bufsize);\
        LWN_TESTBUFFERADDRESS_KICK_BINDSTORAGEBUFFERS(1, result);\
    }


#define LWN_TESTBUFFERADDRESS_BUFFERRANGE(TESTNAME) \
    {\
        BufferRange bufferRange[1];\
\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, 4, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(0, 0, 0);\
        LWN_TESTBUFFERADDRESS_##TESTNAME(bufferAddr, 0, 0);\
        if (m_debugLevel >= DEBUG_FEATURE_SCAN_MEMPOOL_FOR_BUFFERADDRESS) {\
            LWN_TESTBUFFERADDRESS_##TESTNAME(danglingAddr, 0, 0);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(danglingAddr, 4, 1);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(-256, 0, 0);\
            LWN_TESTBUFFERADDRESS_##TESTNAME(-256, 4, 1);\
        }\
        queue->Flush();\
    }

        LWN_TESTBUFFERADDRESS_BUFFERRANGE(BINDVERTEXBUFFERS);
        LWN_TESTBUFFERADDRESS_BUFFERRANGE(BINDUNIFORMBUFFERS);
        LWN_TESTBUFFERADDRESS_BUFFERRANGE(BINDTRANSFORMFEEDBACKBUFFERS);
        LWN_TESTBUFFERADDRESS_BUFFERRANGE(BINDSTORAGEBUFFERS);

        if (m_debugLevel >= 1) {
            // Test Exceptional case. If Buffer=0,Size!=0, TransformFeedbackBuffers should not return error. Others need to.
            BufferRange bufferRange[1];

            LWN_TESTBUFFERADDRESS_BINDVERTEXBUFFERS(0, 4, 1);
            LWN_TESTBUFFERADDRESS_BINDUNIFORMBUFFERS(0, 4, 1);
            LWN_TESTBUFFERADDRESS_BINDSTORAGEBUFFERS(0, 4, 1);
            LWN_TESTBUFFERADDRESS_BINDTRANSFORMFEEDBACKBUFFERS(0, 4, 0);
        }
        {
            // Test accepting the count=0 case.
            BufferRange bufferRange[1];

            LWN_TESTBUFFERADDRESS_KICK_BINDUNIFORMBUFFERS(0, 0);
            LWN_TESTBUFFERADDRESS_KICK_BINDVERTEXBUFFERS(0, 0);
            LWN_TESTBUFFERADDRESS_KICK_BINDTRANSFORMFEEDBACKBUFFERS(0, 0);
            LWN_TESTBUFFERADDRESS_KICK_BINDSTORAGEBUFFERS(0, 0);
        }
        if (m_debugLevel >= 1) {
            // Test declining the count<0 case.
            BufferRange bufferRange[1];

            LWN_TESTBUFFERADDRESS_KICK_BINDUNIFORMBUFFERS(-1, 1);
            LWN_TESTBUFFERADDRESS_KICK_BINDVERTEXBUFFERS(-1, 1);
            LWN_TESTBUFFERADDRESS_KICK_BINDTRANSFORMFEEDBACKBUFFERS(-1, 1);
            LWN_TESTBUFFERADDRESS_KICK_BINDSTORAGEBUFFERS(-1, 1);
        }

#undef LWN_TESTBUFFERADDRESS_BUFFERRANGE
#undef LWN_TESTBUFFERADDRESS_BINDVERTEXBUFFERS
#undef LWN_TESTBUFFERADDRESS_BINDUNIFORMBUFFERS
#undef LWN_TESTBUFFERADDRESS_BINDTRANSFORMFEEDBACKBUFFERS
#undef LWN_TESTBUFFERADDRESS_BINDSTORAGEBUFFERS
#undef LWN_TESTBUFFERADDRESS_KICK_BINDSTORAGEBUFFERS
#undef LWN_TESTBUFFERADDRESS_KICK_BINDTRANSFORMFEEDBACKBUFFERS
#undef LWN_TESTBUFFERADDRESS_KICK_BINDUNIFORMBUFFERS
#undef LWN_TESTBUFFERADDRESS_KICK_BINDVERTEXBUFFERS

        pool.freeTexture(texture);
        pool.freeBuffer(buffer);
        queue->Flush();
        DEBUG_PRINT(("--Test for Bug 1955817 end--"));
    }

    // Check the shder pre-fetch size when calling SetShaders.
    {
        LWNint preFetchSize;
        device->GetInteger(DeviceInfo::SHADER_CODE_MEMORY_POOL_PADDING_SIZE, &preFetchSize);
        assert(preFetchSize==1024);

        int poolSize = 4096;// This is the smallest memory pool size.
        lwnTest::GLSLCHelper glslcHelper(device, poolSize, g_glslcLibraryHelper, NULL);

        static const char *vsstring =
                "#version 440 core\n"
                "#extension GL_LW_gpu_shader5:require\n"
                "layout(location = 0) in vec4 position;\n"
                "layout(location = 1) in vec4 tc;\n"
                "layout(binding = 0) uniform Block {\n"
                "    vec4 scale;\n"
                "    uint64_t bindlessTex;\n"
                "};\n"
                "out IO { vec4 ftc; };\n"
                "void main() {\n"
                "  gl_Position = position*scale;\n"
                "  ftc = tc;\n"
                "}\n";
        static const char *fsstring_bindless =
                "#version 440 core\n"
                "#extension GL_LW_gpu_shader5:require\n"
                "layout(binding = 0) uniform Block {\n"
                "    vec4 scale;\n"
                "    uint64_t bindlessTex;\n"
                "};\n"
                "layout(location = 0) out vec4 color;\n"
                "in IO { vec4 ftc; };\n"
                "void main() {\n"
                "  color = texture(sampler2D(bindlessTex), ftc.xy);\n"
                "}\n";
        Program *program = device->CreateProgram();

        ShaderStage stages[2];
        const char *sources[2];
        int nSources = 2;
        sources[0] = vsstring;
        stages[0] = ShaderStage::VERTEX;
        sources[1] = fsstring_bindless;
        stages[1] = ShaderStage::FRAGMENT;

        // We need the shder data size in the debug layer, so compiling the shders.
        if (!glslcHelper.CompileShaders(stages, nSources, sources)) {
            DEBUG_PRINT(("We need a valid shader here!\n"));
        }
        const GLSLCoutput * glslcOutput = glslcHelper.GetCompiledOutput(0);
        ShaderData shaderData[6];
        memset(&shaderData[0], 0, sizeof(ShaderData) * 6);

        // Hit case 1: shader code base + shader code size + pre-fetch size (1K) > pool base + pool size.
        LWNmemoryPoolFlags poolFlags1 = LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                                                           LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT    |
                                                           LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT   |
                                                           LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT);
        LWNmemoryPool *pool1 = lwnDeviceCreateMemoryPool(cdevice, NULL, poolSize, poolFlags1);
        LWNbufferAddress poolBase1 = lwnMemoryPoolGetBufferAddress(pool1);

        int lwrrIndex = 0;
        for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
            if (glslcOutput->headers[i].genericHeader.common.type ==
                    GLSLC_SECTION_TYPE_GPU_CODE) {
                const char * control = NULL;
                GLSLCgpuCodeHeader gpuCodeHeader =
                        (GLSLCgpuCodeHeader)(glslcOutput->headers[i].gpuCodeHeader);
                const char * data = (char *)glslcOutput + gpuCodeHeader.common.dataOffset;
                control = data + gpuCodeHeader.controlOffset;

                // VS and FS will hit this test case both.
                // We assume the each shader data is smaller than 4K, to hit the test case:
                // data size:   <=1K    1K-2K   2K-3K   3K-4K
                // offset:      3K      2K      1K      0
                int offsetInPool = poolSize - ROUND_UP(gpuCodeHeader.dataSize, preFetchSize);

                shaderData[lwrrIndex].data = poolBase1 + offsetInPool;
                shaderData[lwrrIndex].control = control;
                ++lwrrIndex;
            }
        }

        program->SetShaders(lwrrIndex/*=count*/, &(shaderData[0]));
        EXPECT_DEBUG_CALLBACKS(1);

        // Hit case 2: Pool without SHADER_CODE bit set
        LWNmemoryPoolFlags poolFlags2 = LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                                                           LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT    |
                                                           LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT);
        LWNmemoryPool * pool2 = lwnDeviceCreateMemoryPool(cdevice, NULL, poolSize, poolFlags2);
        LWNbufferAddress poolBase2 = lwnMemoryPoolGetBufferAddress(pool2);

        lwrrIndex = 0;
        for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
            if (glslcOutput->headers[i].genericHeader.common.type ==
                    GLSLC_SECTION_TYPE_GPU_CODE) {
                const char * control = NULL;
                GLSLCgpuCodeHeader gpuCodeHeader =
                        (GLSLCgpuCodeHeader)(glslcOutput->headers[i].gpuCodeHeader);
                const char * data = (char *)glslcOutput + gpuCodeHeader.common.dataOffset;
                control = data + gpuCodeHeader.controlOffset;

                // We do not care the data value.
                shaderData[lwrrIndex].data = poolBase2;
                shaderData[lwrrIndex].control = control;
                ++lwrrIndex;
            }
        }

        program->SetShaders(lwrrIndex, &(shaderData[0]));
        EXPECT_DEBUG_CALLBACKS(1);

        // Hit case 3: The BufferAddress is not found in any existing memory pool.
        lwrrIndex = 0;
        for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
            if (glslcOutput->headers[i].genericHeader.common.type ==
                    GLSLC_SECTION_TYPE_GPU_CODE) {
                const char * control = NULL;
                GLSLCgpuCodeHeader gpuCodeHeader =
                        (GLSLCgpuCodeHeader)(glslcOutput->headers[i].gpuCodeHeader);
                const char * data = (char *)glslcOutput + gpuCodeHeader.common.dataOffset;
                control = data + gpuCodeHeader.controlOffset;

                shaderData[lwrrIndex].data = 0;
                shaderData[lwrrIndex].control = control;
                ++lwrrIndex;
            }
        }

        program->SetShaders(lwrrIndex, &(shaderData[0]));
        EXPECT_DEBUG_CALLBACKS(1);

        program->Free();
        lwnMemoryPoolFree(pool1);
        lwnMemoryPoolFree(pool2);
    }

    // Check Read/Write Texels arguments for uncompressed textures.
    {
        MemoryPoolAllocator pool(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

        // Copy just one texel
        unsigned char buffer = 0xFF;

        TextureView textureView;
        TextureBuilder textureBuilder;
        textureBuilder.SetDevice(device);

        // Allocate new test 1D texture.
        textureBuilder.SetDefaults()
                      .SetTarget(TextureTarget::TARGET_1D)
                      .SetFormat(Format::RGBA8)
                      .SetSize3D(640, 1, 1);
        Texture *texture1D = pool.allocTexture(&textureBuilder);

        // Allocate new test 2D texture.
        textureBuilder.SetDefaults()
                      .SetTarget(TextureTarget::TARGET_2D)
                      .SetFormat(Format::RGBA8)
                      .SetSize2D(640, 480);
        Texture *texture2D = pool.allocTexture(&textureBuilder);

        // Allocate new test 2D array texture.
        textureBuilder.SetDefaults()
                      .SetTarget(TextureTarget::TARGET_2D_ARRAY)
                      .SetFormat(Format::RGBA8)
                      .SetSize3D(640, 480, 4);
        Texture *texture2DArray = pool.allocTexture(&textureBuilder);

        // Allocate new test 3D texture.
        textureBuilder.SetDefaults()
                      .SetTarget(TextureTarget::TARGET_3D)
                      .SetFormat(Format::RGBA8)
                      .SetSize3D(640, 480, 4);
        Texture *texture3D = pool.allocTexture(&textureBuilder);

        // Everything is fine.
        CopyRegion region = { 0, 0, 0, 1, 1, 1 };
        textureView.SetDefaults().SetLevels(0, 1).SetLayers(2, 1);
        texture2DArray->WriteTexels(&textureView, &region, &buffer);
        EXPECT_DEBUG_CALLBACKS(0);

        // Setting the number of layers to 0 should not trigger an error.
        textureView.SetDefaults().SetLevels(0, 1).SetLayers(2, 0);
        texture2DArray->WriteTexels(&textureView, &region, &buffer);
        EXPECT_DEBUG_CALLBACKS(0);

        // Hit the 2D cases.
        textureView.SetDefaults();
        // xoffset/yoffset/zoffset < 0
        CopyRegion region0 = { -1, 0, 0, 0, 1, 1 };
        texture2D->WriteTexels(&textureView, &region0, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        // xoffset + width >= texture.width, dumb check, to be improved
        CopyRegion region1 = { 1000, 0, 0, 0, 1, 1 };
        texture2D->WriteTexels(&textureView, &region1, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        // width < 1
        CopyRegion region2 = { 0, 0, 0, 0, 1, 1 };
        texture2D->WriteTexels(&textureView, &region2, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        // depth != 1
        CopyRegion region3 = { 0, 0, 0, 1, 1, 2 };
        texture2D->WriteTexels(&textureView, &region3, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        // Hit the 2D array cases.
        textureView.SetDefaults().SetLevels(0, 1).SetLayers(2, 1);
        // 2D array, baseLayer(2) + zoffset(1) >= baseLayer(2) + numLayers(1)
        CopyRegion region4 = { 0, 0, 1, 1, 1, 1 };
        texture2DArray->WriteTexels(&textureView, &region4, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        // Hit the 3D cases.
        // 3D, depth < 1
        CopyRegion region5 = { 0, 0, 0, 1, 1, -1 };
        textureView.SetDefaults().SetLayers(0, 0);
        texture3D->WriteTexels(&textureView, &region5, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        // 3D, baseLayer(0) + zoffset(4) >= 3DTexture.depth(4)
        CopyRegion region6 = { 0, 0, 4, 1, 1, 1 };
        texture3D->WriteTexels(&textureView, &region6, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        // Hit the 1D case.
        // 1D, height != 1
        CopyRegion region7 = { 0, 0, 0, 1, 2, 1 };
        textureView.SetDefaults().SetLevels(0, 1).SetLayers(0, 1);
        texture1D->WriteTexels(&textureView, &region7, &buffer);
        EXPECT_DEBUG_CALLBACKS(1);

        pool.freeTexture(texture1D);
        pool.freeTexture(texture2D);
        pool.freeTexture(texture2DArray);
        pool.freeTexture(texture3D);
    }

    // Check Read/Write Texels arguments for compressed textures
    {
        MemoryPoolAllocator pool(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);

        // 2x2 blocks, in case we need it
        uint64_t buffer[4] = {
            0xDEADBEEFBABE0000UL,
            0xBEEFCAFEDEADBEEFUL,
            0xAAAADEADBEEFFFFFUL,
            0xCAFEFFFAAFFFBABEUL,
        };

        TextureView textureView;
        textureView.SetDefaults();

        TextureBuilder textureBuilder;
        textureBuilder.SetDevice(device);

        // Allocate new test 2D texture.
        textureBuilder.SetDefaults()
                      .SetTarget(TextureTarget::TARGET_2D)
                      .SetFormat(Format::RGBA_ASTC_4x4)
                      .SetSize2D(640, 480);
        Texture *texture2D = pool.allocTexture(&textureBuilder);

        // Allocate new test 2D array texture.
        textureBuilder.SetDefaults()
                      .SetTarget(TextureTarget::TARGET_2D_ARRAY)
                      .SetFormat(Format::RGBA_ASTC_4x4)
                      .SetSize3D(640, 480, 4);
        Texture *texture2DArray = pool.allocTexture(&textureBuilder);

        // Allocate new test 3D texture.
        textureBuilder.SetDefaults()
                      .SetTarget(TextureTarget::TARGET_3D)
                      .SetFormat(Format::RGBA_ASTC_4x4)
                      .SetSize3D(640, 480, 4);
        Texture *texture3D = pool.allocTexture(&textureBuilder);

        // Regions are { xoffset, yoffset, zoffset, width, height, depth }

        // Test happy paths
        {
            CopyRegion region2D = { 0, 0, 0, 4, 4, 1};
            texture2D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(0);

            texture2DArray->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(0);

            texture3D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(0);
        }

        // Test misalignment in X
        {
            // Make sure we detect bad alignment of the offsets
            CopyRegion region2D = { 1, 0, 0, 4, 4, 1};
            texture2D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture2DArray->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture3D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            // And bad alignment of the end points
            region2D = { 0, 0, 0, 2, 4, 1};
            texture2D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture2DArray->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture3D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);
        }

        // Test misalignment in Y
        {
            // Make sure we detect bad alignment of the offsets
            CopyRegion region2D = { 0, 1, 0, 4, 4, 1};
            texture2D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture2DArray->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture3D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            // And bad alignment of the end points
            region2D = { 0, 0, 0, 4, 2, 1};
            texture2D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture2DArray->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

            texture3D->WriteTexels(&textureView, &region2D, buffer);
            EXPECT_DEBUG_CALLBACKS(1);

        }
    }

    // Check texture builder parameter verification.
    {
        TextureBuilder texb;

        // Set up for the rest of the tests
        texb.SetDefaults().SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(640, 480);

        texb.SetFormat(Format::NONE); // Invalid format.
        LWNuintptr tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetFormat(Format::RGB8); // Vertex-only format.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetFormat(Format::RGB16I); // Vertex-only format.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetFormat(Format::RGBA32_I2F); // Vertex-only format.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetFormat(Format::RGBA8)
            .SetTarget(TextureTarget::TARGET_RECTANGLE)
            .SetLevels(2); // TARGET_RECTANGLE must have levels = 1.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetTarget(TextureTarget::TARGET_BUFFER)
            .SetLevels(2); // TARGET_BUFFER must have levels = 1.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetTarget(TextureTarget::TARGET_2D)
            .SetLevels(1)
            .SetSamples(4); // Samples must be zero for non-multisample targets.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE)
            .SetSamples(3); // Samples must be 2, 4, or 8 for multisample targets.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE)
            .SetSamples(64); // Samples must be 2, 4, or 8 for multisample targets.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY)
            .SetSamples(1); // Samples must be 2, 4, or 8  for multisample targets.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY)
            .SetSamples(0); // Samples must be 2, 4, or 8  for multisample targets.
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        (void) tsz;
    }

    // Check textureBuilder parameter verification for textures with DISPLAY_BIT
    {
        TextureBuilder texb;
        texb.SetDevice(device).SetDefaults()
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(640, 480)
            .SetFlags(TextureFlags::COMPRESSIBLE | TextureFlags::DISPLAY);

        // Invalid format
        texb.SetFormat(Format::STENCIL8);
        LWNuintptr tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        // Invalid levels
        texb.SetFormat(Format::RGBA8)
            .SetLevels(4);
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        // Invalid target
        texb.SetTarget(TextureTarget::TARGET_2D_ARRAY)
            .SetLevels(1);
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        // Invalid flag: LINEAR
        texb.SetTarget(TextureTarget::TARGET_2D)
            .SetFlags(TextureFlags::COMPRESSIBLE | TextureFlags::DISPLAY | TextureFlags::LINEAR);
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        // Invalid flag: LINEAR_RENDER_TARGET
        texb.SetTarget(TextureTarget::TARGET_2D)
            .SetFlags(TextureFlags::COMPRESSIBLE | TextureFlags::DISPLAY | TextureFlags::LINEAR_RENDER_TARGET);
        tsz = texb.GetStorageSize();
        EXPECT_DEBUG_CALLBACKS(1);
        (void)texb.CreateTextureFromPool(tpool, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        (void)tsz;
    }

    // We should detect invalid or NULL parameters passed to lwnDeviceRegisterTexture /
    // lwnDeviceRegisterSampler.
    LWNint numReservedTextures, numReservedSamplers;
    device->GetInteger(DeviceInfo::RESERVED_TEXTURE_DESCRIPTORS, &numReservedTextures);
    device->GetInteger(DeviceInfo::RESERVED_SAMPLER_DESCRIPTORS, &numReservedSamplers);
    {
        LWNmemoryPool *ctpool = reinterpret_cast<LWNmemoryPool *>(tpool);
        LWNtexture validTexture;
        LWNtextureBuilder tbuilder;
        lwnTextureBuilderSetDevice(&tbuilder, cdevice);
        lwnTextureBuilderSetDefaults(&tbuilder);
        lwnTextureBuilderSetTarget(&tbuilder, LWN_TEXTURE_TARGET_2D);
        lwnTextureBuilderSetSize2D(&tbuilder, 16, 16);
        lwnTextureBuilderSetFormat(&tbuilder, LWN_FORMAT_R8);
        lwnTextureBuilderSetStorage(&tbuilder, ctpool, 0);
        lwnTextureInitialize(&validTexture, &tbuilder);
        EXPECT_DEBUG_CALLBACKS(0, 0);

        LWNsampler validSampler;
        LWNsamplerBuilder sbuilder;
        lwnSamplerBuilderSetDevice(&sbuilder, cdevice);
        lwnSamplerBuilderSetDefaults(&sbuilder);
        lwnSamplerInitialize(&validSampler, &sbuilder);
        EXPECT_DEBUG_CALLBACKS(0, 0);

        const TexturePool *cppTexPool = g_lwnTexIDPool->GetTexturePool();
        const LWNtexturePool *texPool = reinterpret_cast<const LWNtexturePool *>(cppTexPool);
        device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);
        lwnTexturePoolRegisterTexture(NULL, numReservedTextures + 1, &validTexture, NULL);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnTexturePoolRegisterTexture(texPool, numReservedTextures + 1, NULL, NULL);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnTexturePoolRegisterTexture(texPool, 0, &validTexture, NULL);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnTexturePoolRegisterTexture(texPool, 0xFFFFFF, &validTexture, NULL);
        EXPECT_DEBUG_CALLBACKS(1);

        const SamplerPool *cppSmpPool = g_lwnTexIDPool->GetSamplerPool();
        const LWNsamplerPool *smpPool = reinterpret_cast<const LWNsamplerPool *>(cppSmpPool);
        device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);
        lwnSamplerPoolRegisterSampler(NULL, numReservedSamplers + 1, &validSampler);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnSamplerPoolRegisterSampler(smpPool, numReservedSamplers + 1, NULL);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnSamplerPoolRegisterSampler(smpPool, 0, &validSampler);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnSamplerPoolRegisterSampler(smpPool, 0xFFFFFF, &validSampler);
        EXPECT_DEBUG_CALLBACKS(1);

        if (lwogCheckLWNAPIVersion(53, 0)) {
            // Perform similar checks for the RegisterSamplerBuilder API.  The
            // debug layer doesn't track sampler builder objects, we can't
            // test NULL sampler builder pointers.
            lwnSamplerPoolRegisterSamplerBuilder(NULL, numReservedSamplers + 1, &sbuilder);
            EXPECT_DEBUG_CALLBACKS(1);
            lwnSamplerPoolRegisterSamplerBuilder(smpPool, 0, &sbuilder);
            EXPECT_DEBUG_CALLBACKS(1);
            lwnSamplerPoolRegisterSamplerBuilder(smpPool, 0xFFFFFF, &sbuilder);
            EXPECT_DEBUG_CALLBACKS(1);
        }

        lwnTextureFinalize(&validTexture);
        lwnSamplerFinalize(&validSampler);
    }

    // Test that trying to reset the timestamp 'counter' results in an error.
    {
        queueCB.ResetCounter(CounterType::TIMESTAMP);
        EXPECT_DEBUG_CALLBACKS(1);
    }

    // Test that trying to call the commands outside a BeginRecording->EndRecording code block.
    {
        // Positive test, queueCB is BeginRecording by default.
        float clearColor[] = { 0, 0, 0, 1 };
        queueCB.ClearColor(0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
        queueCB.EndRecording();
        EXPECT_DEBUG_CALLBACKS(0);

        lwnTest::GLSLCHelper glslcHelper(device, 0x100000, g_glslcLibraryHelper, g_glslcHelperCache);
        VertexShader vs(450);
        FragmentShader fs(450);
        vs <<
            "void main()\n"
            "{\n"
            "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        fs <<
            "out vec4 color;\n"
            "void main()\n"
            "{\n"
            "  color = vec4(1.0, 1.0, 1.0, 1.0);\n"
            "}\n";

        Program *pgm = device->CreateProgram();
        if (!glslcHelper.CompileAndSetShaders(pgm, vs, fs)) {
            ADD_RESULT(false);
            return;
        }

        MemoryPoolAllocator pool(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);
        float theFloat = 0.0;
        CopyRegion bufferRegion = { 0, 0, 0, 1, 1, 1 };

        TextureBuilder textureBuilder;
        textureBuilder.SetDevice(device)
                      .SetDefaults()
                      .SetTarget(TextureTarget::TARGET_2D)
                      .SetFormat(Format::RGBA8)
                      .SetSize2D(64, 64);
        Texture *texture = pool.allocTexture(&textureBuilder);

        BufferBuilder bufferBuilder;
        bufferBuilder.SetDevice(device);
        bufferBuilder.SetDefaults();
        Buffer *buffer = pool.allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, 0x20);
        BufferAddress bufferAddr = buffer->GetAddress();

        BlendState blendState;
        blendState.SetDefaults();
        ChannelMaskState maskNoColor;
        maskNoColor.SetDefaults().SetChannelMask(0, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE);
        ColorState colorState;
        colorState.SetDefaults();
        DepthStencilState dssDefault;
        dssDefault.SetDefaults();
        MultisampleState multisampleState;
        multisampleState.SetDefaults();
        PolygonState polygon;
        polygon.SetDefaults();
        VertexAttribState attribState;
        attribState.SetDefaults();
        VertexStreamState streamState;
        streamState.SetDefaults();
        int clearColori[] = { 0, 0, 0, 1 };
        uint32_t clearColorui[] = { 0, 0, 0, 1 };
        float dRanges[] = {0.0, 1.0};
        int rect[] = {0, 0, 1, 1};
        float ranges[] = {0, 0, 1, 1};
        EXPECT_DEBUG_CALLBACKS(0);

        // queueCB is not lwrrently recording, so each command below should generate one debug callback.
        if (m_debugLevel > DEBUG_FEATURE_SKIP_BUSY_ENTRIES) {
            queueCB.Barrier(LWN_BARRIER_ILWALIDATE_TEXTURE_BIT);
            queueCB.BeginTransformFeedback(bufferAddr);
            queueCB.BindBlendState(&blendState);
            queueCB.BindChannelMaskState(&maskNoColor);
            queueCB.BindColorState(&colorState);
            queueCB.BindDepthStencilState(&dssDefault);
            queueCB.BindCoverageModulationTable(&theFloat);
            queueCB.BindMultisampleState(&multisampleState);
            queueCB.BindPolygonState(&polygon);
            queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB.BindStorageBuffer(ShaderStage::VERTEX, 1, bufferAddr, 1);
            queueCB.BindTransformFeedbackBuffer(1, bufferAddr, 4);
            queueCB.BindVertexBuffer(1, bufferAddr, 1);
            queueCB.BindVertexStreamState(1, &streamState);
            queueCB.ClearColor(0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
            queueCB.ClearColori(0, clearColori, LWN_CLEAR_COLOR_MASK_RGBA);
            queueCB.ClearColorui(0, clearColorui, LWN_CLEAR_COLOR_MASK_RGBA);
            queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
            queueCB.CopyBufferToBuffer(bufferAddr, bufferAddr, 1, CopyFlags::NONE);
            queueCB.CopyBufferToTexture(bufferAddr, texture, NULL, &bufferRegion, CopyFlags::NONE);
            queueCB.CopyTextureToBuffer(texture, NULL, &bufferRegion, bufferAddr, CopyFlags::NONE);
            queueCB.CopyTextureToTexture(texture, NULL, &bufferRegion, texture, NULL, &bufferRegion, CopyFlags::NONE);
            queueCB.DiscardColor(0);
            queueCB.DiscardDepthStencil();
            queueCB.Downsample(texture, texture);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            queueCB.DrawArraysIndirect(DrawPrimitive::TRIANGLE_STRIP, bufferAddr);
            queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 1, 1, 1);
            queueCB.DrawElements(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, 4, bufferAddr);
            queueCB.DrawElementsBaseVertex(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, 4, bufferAddr, 0);
            queueCB.DrawElementsIndirect(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, bufferAddr, bufferAddr);
            queueCB.DrawElementsInstanced(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, 4, bufferAddr, 1, 1, 1);
            queueCB.DrawTransformFeedback(DrawPrimitive::TRIANGLE_STRIP, bufferAddr);
            queueCB.EndTransformFeedback(bufferAddr);
            queueCB.MultiDrawArraysIndirectCount(DrawPrimitive::TRIANGLE_STRIP, bufferAddr, bufferAddr, 1, 1);
            queueCB.MultiDrawElementsIndirectCount(DrawPrimitive::TRIANGLE_STRIP, IndexType::UNSIGNED_INT, bufferAddr, bufferAddr, bufferAddr, 1, 1);
            queueCB.PauseTransformFeedback(bufferAddr);
            queueCB.ReportCounter(CounterType::TIMESTAMP, bufferAddr);
            queueCB.ReportValue(1, bufferAddr);
            queueCB.ResetCounter(CounterType::ZLWLL_STATS);
            queueCB.ResolveDepthBuffer();
            queueCB.RestoreZLwllData(bufferAddr, 1);
            queueCB.ResumeTransformFeedback(bufferAddr);
            queueCB.SaveZLwllData(bufferAddr, 1);
            queueCB.SetRenderEnableConditional(ConditionalRenderMode::RENDER_IF_NOT_EQUAL, bufferAddr);
            queueCB.SetRenderTargets(1, &texture, NULL, NULL, NULL);
            queueCB.SetStencilMask(Face::BACK, 1);
            queueCB.SetStencilRef(Face::BACK, 1);
            queueCB.SetStencilValueMask(Face::BACK, 1);
            queueCB.SetTiledCacheAction(TiledCacheAction::ENABLE);
            queueCB.SetAlphaRef(1.0);
            queueCB.SetBlendColor(&theFloat);
            queueCB.SetConservativeRasterDilate(1.0);
            queueCB.SetConservativeRasterEnable(true);
            queueCB.SetDepthBounds(true, 0.0, 1.0);
            queueCB.SetDepthClamp(true);
            queueCB.SetDepthRange(0.0, 1.0);
            queueCB.SetDepthRanges(0, 2, dRanges);
            queueCB.SetInnerTessellationLevels(&theFloat);
            queueCB.SetLineWidth(1.0);
            queueCB.SetOuterTessellationLevels(&theFloat);
            queueCB.SetPatchSize(1);
            queueCB.SetPointSize(1.0);
            queueCB.SetPolygonOffsetClamp(0.0, 0.5, 1.0);
            queueCB.SetPrimitiveRestart(true, 0);
            queueCB.SetRasterizerDiscard(true);
            queueCB.SetRenderEnable(true);
            queueCB.SetSampleMask(0);
            queueCB.SetScissor(0, 0, 1, 1);
            queueCB.SetScissors(0, 1, rect);
            queueCB.SetSubpixelPrecisionBias(1, 1);
            queueCB.SetTiledCacheTileSize(1, 1);
            queueCB.SetViewport(0, 0, 1, 1);
            queueCB.SetViewports(0, 1, ranges);
            queueCB.TiledDownsample(texture, texture);
            EXPECT_DEBUG_CALLBACKS(75);
        } else {
            queueCB.CopyBufferToBuffer(bufferAddr, bufferAddr, 1, CopyFlags::NONE);
            queueCB.CopyBufferToTexture(bufferAddr, texture, NULL, &bufferRegion, CopyFlags::NONE);
            queueCB.CopyTextureToBuffer(texture, NULL, &bufferRegion, bufferAddr, CopyFlags::NONE);
            queueCB.CopyTextureToTexture(texture, NULL, &bufferRegion, texture, NULL, &bufferRegion, CopyFlags::NONE);
            queueCB.DiscardColor(0);
            queueCB.DiscardDepthStencil();
            queueCB.TiledDownsample(texture, texture);
            EXPECT_DEBUG_CALLBACKS(7);
        }

        pool.freeTexture(texture);
        pool.freeBuffer(buffer);
        pgm->Free();

        // Restore the default recording state.
        queueCB.BeginRecording();
    }

    // cmdBuf Command memory overlap test.
    if (m_debugLevel >= DEBUG_FEATURE_IN_FLIGHT_CMDBUF_TRACKING) {
        Queue *newQueue = device->CreateQueue();

        VertexShader vs(440);
        vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "out vec3 ocolor;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  ocolor = color;\n"
            "}\n";
        FragmentShader fs(440);
        fs <<
            "in vec3 ocolor;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = vec4(ocolor, 1.0);\n"
            "}\n";
        Program *pgm = device->CreateProgram();
        g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

        int commandSize = 0;
        int controlSize = 0;
        int cmdAlignment = 0;
        int controlAlignment = 0;
        device->GetInteger(DeviceInfo::COMMAND_BUFFER_MIN_COMMAND_SIZE, &commandSize);
        device->GetInteger(DeviceInfo::COMMAND_BUFFER_MIN_CONTROL_SIZE, &controlSize);
        device->GetInteger(DeviceInfo::COMMAND_BUFFER_COMMAND_ALIGNMENT, &cmdAlignment);
        device->GetInteger(DeviceInfo::COMMAND_BUFFER_CONTROL_ALIGNMENT, &controlAlignment);
        commandSize *= 10;
        controlSize = 10 * ROUND_UP(controlSize, controlAlignment);

        // Needs a CPU_CACHED pool for CopyCommands.
        MemoryPool *nonCoherentPool = device->CreateMemoryPool(NULL, 2 * commandSize, MemoryPoolType::CPU_NON_COHERENT);
        MemoryPool *pool = device->CreateMemoryPool(NULL, 2 * commandSize, MemoryPoolType::CPU_COHERENT);
        int controlSpaceSize = 5 * controlSize;
        char *controlSpace = new char[controlSpaceSize];
        memset(controlSpace, 0, controlSpaceSize);
        char *controlSpaceAligned = (char*)(((uintptr_t)controlSpace + controlAlignment-1) & (~(controlAlignment-1)));

        struct TestCase {
            int overlapType;  //0: non-overlap, 1, overlap, 2, contain
            bool reverse;
            bool needSync;
            bool needCallCommand;

            int expectedErrors;
        };

        const static TestCase testCasesCMDMemory[] = {
            // Positive test, there is no overlap that passes the test.
            {0,  false,   false,   false,  0},

            // Overlapping by cmdAlignment bytes.
            // New range covers end of previous range generates debug message.
            {1,  false,   false,   false,  1},
            // New range covers end of previous range, plus sync, passes the test.
            {1,  false,    true,   false,  0},
            // New range covers start of previous range generates debug message.
            {1,   true,   false,   false,  1},
            // New range covers start of previous range, plus sync, passes the test.
            {1,   true,    true,   false,  0},

            // One contains another.
            // New range is contained entirely within previous range generates debug message.
            {2,  false,   false,   false,  1},
            // New range is contained entirely within previous range, plus sync, passes the test.
            {2,  false,    true,   false,  0},
            // New range contains previous range entirely generates debug message.
            {2,   true,   false,   false,  1},
            // New range contains previous range entirely, plus sync, passes the test.
            {2,   true,    true,   false,  0},

            // CallCommands
            // New range CallCommands on previous range, without overlap, passes the test.
            {0,  false,   false,    true,  0},
            // New range CallCommands on previous range, overlapping by cmdAlignment bytes, generates debug message.
            {1,  false,   false,    true,  2},
        };
        int numTestCases = __GL_ARRAYSIZE(testCasesCMDMemory);

        for (int k = 0; k < numTestCases; k++){
            ptrdiff_t lastCtrlOffset = 0;
            size_t usedCMDSize = 0;
            int bindProgramNum = 4;
            const int cmdBufNum = 2;
            CommandBuffer *cmdBuf[cmdBufNum];
            CommandHandle handle[cmdBufNum];

            for (int i = 0; i < cmdBufNum; i++) {
                cmdBuf[i] = device->CreateCommandBuffer();

                if (i == 0) {
                    cmdBuf[i]->AddCommandMemory(pool, 0, commandSize);
                } else {
                    if (testCasesCMDMemory[k].overlapType == 0) {
                        // no overlap
                        cmdBuf[i]->AddCommandMemory(pool, usedCMDSize, commandSize);
                    } else if (testCasesCMDMemory[k].overlapType == 1){
                        // Overlapping by cmdAlignment bytes.
                        assert(usedCMDSize > (size_t)cmdAlignment);
                        cmdBuf[i]->AddCommandMemory(pool, usedCMDSize - cmdAlignment, commandSize);
                    } else {
                        // One contains another.
                        cmdBuf[i]->AddCommandMemory(pool, 0 + cmdAlignment, commandSize);
                    }
                }

                cmdBuf[i]->AddControlMemory(controlSpaceAligned + lastCtrlOffset, controlSize);
                lastCtrlOffset += controlSize;

                cmdBuf[i]->BeginRecording();
                for (int j = 0; j < bindProgramNum; j++) {
                    cmdBuf[i]->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
                }
                if (i && testCasesCMDMemory[k].needCallCommand) cmdBuf[i]->CallCommands(1, &handle[0]);
                handle[i] = cmdBuf[i]->EndRecording();

                usedCMDSize = cmdBuf[i]->GetCommandMemoryUsed();
                // The second cmdBuf size is smaller than the first, for the contain case.
                bindProgramNum--;
            }

            for (int i = 0; i < cmdBufNum; i++) {
                int ii = testCasesCMDMemory[k].reverse ? (cmdBufNum-1-i) : i;
                newQueue->SubmitCommands(1, &handle[ii]);

                if (testCasesCMDMemory[k].needSync) newQueue->Finish();
            }

            newQueue->Finish();
            EXPECT_DEBUG_CALLBACKS(testCasesCMDMemory[k].expectedErrors);

            for (int i = 0; i < cmdBufNum; i++) {
                cmdBuf[i]->Free();
            }
        }

        // CopyCommands test.
        // cmdBuf[0] and cmdBuf[2] are from CPU_CACHED pool, there is overlap between them.
        // cmdBuf[1] and cmdBuf[3] are from CPU_UNCACHED pool, there is NO overlap between them.
        // cmdBuf[1] CopyCommands on cmdBuf[0], cmdBuf[3] CopyCommands on cmdBuf[2], submitting cmdBuf[1] and cmdBuf[3].
        // Note: The order is: fills cmdBuf[0], copies from cmdBuf[0], fills cmdBuf[2], copies from cmdBuf[2], otherwise crash.
        ptrdiff_t lastCtrlOffset = 0;
        const int cmdBufNum = 4;
        CommandBuffer *cmdBuf[cmdBufNum];
        CommandHandle handle[cmdBufNum];
        size_t usedCMDSize[cmdBufNum] = { 0 };

        for (int i = 0; i < cmdBufNum; i++) {
            cmdBuf[i] = device->CreateCommandBuffer();

            if (i < 2) {
                cmdBuf[i]->AddCommandMemory((i%2) ? pool : nonCoherentPool, 0, commandSize);
            } else {
                // Only cmdBuf[0] and cmdBuf[3] overlapping by 10*cmdAlignment bytes.
                int overlapSize = (i==3) ? 0 : 10*cmdAlignment;
                cmdBuf[i]->AddCommandMemory((i%2) ? pool : nonCoherentPool, usedCMDSize[i-2] - overlapSize, commandSize);
            }

            cmdBuf[i]->AddControlMemory(controlSpaceAligned + lastCtrlOffset, controlSize);
            lastCtrlOffset += controlSize;

            cmdBuf[i]->BeginRecording();
            for (int j = 0; j < 4; j++) {
                cmdBuf[i]->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            }
            if (i % 2) cmdBuf[i]->CopyCommands(1, &handle[i-1]);
            handle[i] = cmdBuf[i]->EndRecording();

            usedCMDSize[i] = cmdBuf[i]->GetCommandMemoryUsed();
        }

        newQueue->SubmitCommands(1, &handle[1]);
        newQueue->SubmitCommands(1, &handle[3]);
        newQueue->Finish();
        EXPECT_DEBUG_CALLBACKS(0);

        for (int i = 0; i < cmdBufNum; i++) {
            cmdBuf[i]->Free();
        }

        pgm->Free();
        pool->Free();
        nonCoherentPool->Free();
        delete[] controlSpace;
        controlSpace = NULL;
        controlSpaceAligned = NULL;
        newQueue->Free();
    }

    // Test that trying to use GPU_ONLY mempool for CmdBuf Command Memory errors out.
    CommandBuffer *cmd = device->CreateCommandBuffer();
    void *controlMem = malloc(0x1000);
    {
        MemoryPool *testGPUPool = device->CreateMemoryPool(NULL, 0x1000, MemoryPoolType::GPU_ONLY);
        cmd->AddCommandMemory(testGPUPool, 0,  0x1000);
        EXPECT_DEBUG_CALLBACKS(1);

        // Test that trying to use the command buffer with no memory errors out.
        cmd->BeginRecording();
        EXPECT_DEBUG_CALLBACKS(1);
        if (cmd->IsRecording()) {
            cmd->EndRecording();
        }

        cmd->AddControlMemory(controlMem, 0x1000);

        // Test that debug layer complains about mapping a GPU only pool.
        testGPUPool->Map();
        EXPECT_DEBUG_CALLBACKS(1);

        testGPUPool->Free();
    }

    // Test that finalizing a mempool before the objects inside it generates an error,
    // and trying to initialize an object twice also generates an error.
    MemoryPool *testCPUPool = device->CreateMemoryPool(NULL, 0x1000, MemoryPoolType::CPU_NON_COHERENT);
    MemoryPool *testCPUPool1 = device->CreateMemoryPool(NULL, 0x1000, MemoryPoolType::CPU_NON_COHERENT);
    MemoryPool *testCPUPool2 = device->CreateMemoryPool(NULL, 0x1000, MemoryPoolType::CPU_NON_COHERENT);
    if (m_debugLevel >= DEBUG_FEATURE_IN_FLIGHT_CMDBUF_TRACKING) {
        BufferBuilder bb;
        bb.SetDevice(device).SetDefaults();
        Buffer *buffer = bb.CreateBufferFromPool(testCPUPool, 0, 0x1000);
        testCPUPool->Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        buffer->Free();

        TextureBuilder tb;
        tb.SetDevice(device);
        tb.SetDefaults().SetTarget(TextureTarget::TARGET_2D).SetSize2D(6, 6).SetFormat(Format::RGBA8);
        Texture *tex = tb.CreateTextureFromPool(testCPUPool, 0);
        testCPUPool->Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        tex->Free();
    }

    // Command buffer not finalized before mempool is.
    {
#if !LWN_DEBUG_DISABLE_CMDBUF_MEMPOOL_TRACKING_WAR_BUG_1704195
        cmd->AddCommandMemory(testCPUPool, 0,  0x1000);
        testCPUPool.Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        cmd->AddCommandMemory(testCPUPool1, 0,  0x1000);
        testCPUPool.Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        testCPUPool1.Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        cmd->AddCommandMemory(testCPUPool2, 0,  0x1000);
        testCPUPool.Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        testCPUPool1.Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        testCPUPool2.Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
#endif
        EXPECT_DEBUG_CALLBACKS(0, 0);
    }

    // Test that commandBufferIsRecording works (not strictly a debug layer test).
    {
        cmd->AddCommandMemory(testCPUPool, 0,  0x1000);
        EXPECT_DEBUG_CALLBACKS(0, 0);
        ADD_RESULT(!cmd->IsRecording()); // Expect to be not recording.
        cmd->BeginRecording();
        EXPECT_DEBUG_CALLBACKS(0, 0);
        ADD_RESULT(cmd->IsRecording() ? true : false); // Expect to be recording now.
        CommandHandle cmdHandle = cmd->EndRecording();
        (void) cmdHandle;
        EXPECT_DEBUG_CALLBACKS(0, 0);
        ADD_RESULT(!cmd->IsRecording()); // Expect to be not recording again.
    }
    cmd->Free();
    free(controlMem);

    // Test that out of memory callback not giving enough memory or no callback raises debug layer concerns.
    {
        MemoryPool *testCPUPool3 = device->CreateMemoryPool(NULL, 0x1000, MemoryPoolType::CPU_NON_COHERENT);
        void* controlMemInitial = malloc(0x1000);
        cmd = device->CreateCommandBuffer();
        cmd->AddControlMemory(controlMemInitial, 0x100);
        cmd->AddCommandMemory(testCPUPool3, 0,  0x200);

        CommandBufferOOMTestRecovery recoveryInfo;
        controlMem = malloc(0x1000);
        MemoryPool *testCPUPool4 = device->CreateMemoryPool(NULL, 0x1000, MemoryPoolType::CPU_NON_COHERENT);
        recoveryInfo.cmdbuf = cmd;
        recoveryInfo.controlMemory = controlMem;
        recoveryInfo.controlMemorySize = 0x100;
        recoveryInfo.commandMemory = NULL;
        recoveryInfo.recovered = false;
        recoveryInfo.error = false;

        cmd->BeginRecording();
        device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
        device->InstallDebugCallback(CallbackCmdbufOOMTest, &recoveryInfo, LWN_TRUE);

        while (!recoveryInfo.recovered) {
            // Saturate control memory.
            cmd->SetShaderScratchMemory(testCPUPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE);
        }
        EXPECT_DEBUG_CALLBACKS(1, 0);

        recoveryInfo.controlMemory = NULL;
        recoveryInfo.commandMemory = testCPUPool4;
        recoveryInfo.commandMemorySize = 0x200;
        recoveryInfo.recovered = false;
        DepthStencilState dpstate;
        while (!recoveryInfo.recovered) {
            // Saturate command memory.
            dpstate.SetDefaults();
            cmd->BindDepthStencilState(&dpstate);
        }
        EXPECT_DEBUG_CALLBACKS(1, 0);

        void *controlMem2 = malloc(0x1000);
        recoveryInfo.controlMemory = controlMem2;
        recoveryInfo.controlMemorySize = 0x400;
        recoveryInfo.commandMemory = NULL;
        recoveryInfo.recovered = false;
        recoveryInfo.error = false;
        cmd->SetMemoryCallbackData(&recoveryInfo);
        cmd->SetMemoryCallback(CallbackCmdbufOOMDummy);
        while (!recoveryInfo.recovered) {
            // Saturate control memory.
            cmd->SetShaderScratchMemory(testCPUPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE);
        }
        EXPECT_DEBUG_CALLBACKS(1, 0);

        recoveryInfo.controlMemory = NULL;
        recoveryInfo.commandMemory = testCPUPool4;
        recoveryInfo.commandMemorySize = 0x200;
        recoveryInfo.recovered = false;
        recoveryInfo.error = false;
        while (!recoveryInfo.recovered) {
            // Saturate command memory.
            dpstate.SetDefaults();
            cmd->BindDepthStencilState(&dpstate);
        }
        EXPECT_DEBUG_CALLBACKS(1, 0);

        device->InstallDebugCallback(CallbackCmdbufOOMTest, NULL, LWN_FALSE);
        device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);
        CommandHandle cmdHandle = cmd->EndRecording();
        (void) cmdHandle;
        EXPECT_DEBUG_CALLBACKS(0, 0);

        free(controlMemInitial);
        free(controlMem2);
        free(controlMem);

        testCPUPool->Free();
        EXPECT_DEBUG_CALLBACKS(0, 0);
        testCPUPool1->Free();
        EXPECT_DEBUG_CALLBACKS(0, 0);
        testCPUPool2->Free();
        EXPECT_DEBUG_CALLBACKS(0, 0);
        testCPUPool3->Free();
        EXPECT_DEBUG_CALLBACKS(0, 0);
        testCPUPool4->Free();
        EXPECT_DEBUG_CALLBACKS(0, 0);


        cmd->Free();
    }

    // Attempt to create a texture in a coherent pool. It should not get created on a Windows platform.
    {
#if defined(LW_WINDOWS)
        LWNmemoryPool *testCPUCoherentPool = lwnDeviceCreateMemoryPool(cdevice, NULL, 0x1000, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(cdevice);
        lwnTextureBuilderSetDefaults(textureBuilder);
        lwnTextureBuilderSetSize2D(textureBuilder, 4, 4);
        lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
        lwnTextureBuilderSetSamples(textureBuilder, 0);
        lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

        // offset needs to be manually adjusted if you change any of the other allocations from this pool
        // (overlap etc.)
        lwnTextureBuilderCreateTextureFromPool(textureBuilder, testCPUCoherentPool, /*offset*/0);
        EXPECT_DEBUG_CALLBACKS(1);
        lwnTextureBuilderFree(textureBuilder);
        lwnMemoryPoolFree(testCPUCoherentPool);
#endif
    }

    // Test lwnQueuePresentTextureSync with no OpenGL context.
    {
#if defined(LW_WINDOWS)
        // Set up a valid (small) texture for an lwnQueuePresentTextureSync test.
        LWNmemoryPool *presentTexturePool = lwnDeviceCreateMemoryPool(cdevice, NULL, 128 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
        LWNtextureBuilder *presentBuilder = lwnDeviceCreateTextureBuilder(cdevice);
        lwnTextureBuilderSetDefaults(presentBuilder);
        lwnTextureBuilderSetFlags(presentBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT | LWN_TEXTURE_FLAGS_DISPLAY_BIT);
        lwnTextureBuilderSetSize2D(presentBuilder, 4, 4);
        lwnTextureBuilderSetTarget(presentBuilder, LWN_TEXTURE_TARGET_2D);
        lwnTextureBuilderSetSamples(presentBuilder, 0);
        lwnTextureBuilderSetFormat(presentBuilder, LWN_FORMAT_RGBA8);

        // The minimum textures per window is 2
        const int bufferNum = 2;
        LWNtexture *presentTexture[bufferNum];
        Texture *cppPresentTexture[bufferNum];
        for (int i = 0; i < bufferNum; i++) {
            presentTexture[i] = lwnTextureBuilderCreateTextureFromPool(presentBuilder, presentTexturePool, /*offset*/0);
            cppPresentTexture[i] = reinterpret_cast<Texture *>(presentTexture[i]);
        }

        lwnTextureBuilderFree(presentBuilder);

        // Set up a Window for use by the present test.
        WindowBuilder *noGCPresentWB = device->CreateWindowBuilder();
        noGCPresentWB->SetDefaults();
        noGCPresentWB->SetTextures(bufferNum, cppPresentTexture);
        Window *noGCPresentWindow = noGCPresentWB->CreateWin();
        Sync *textureAvailableSync = device->CreateSync();

        // We should still be OK at this point.
        EXPECT_DEBUG_CALLBACKS(0, 0);

        // Attempt to present the texture with no current OpenGL context.
        LWNint noGCPresentTexIndex;
        noGCPresentWindow->AcquireTexture(textureAvailableSync, &noGCPresentTexIndex);
        lwogSetupGLContext(0);
        queue->PresentTexture(noGCPresentWindow, noGCPresentTexIndex);
        lwogSetupGLContext(1);
        EXPECT_DEBUG_CALLBACKS(1);

        noGCPresentWB->Free();
        noGCPresentWindow->Free();
        textureAvailableSync->Free();
        for (int i = 0; i < bufferNum; i++) {
            lwnTextureFree(presentTexture[i]);
        }
        lwnMemoryPoolFree(presentTexturePool);
#endif
    }

    // Test that non-flushed fence waits cause errors.
    {
        Sync *testFence = device->CreateSync();
        Queue *queue2 = device->CreateQueue();

        SyncWaitResult condition = testFence->Wait(0);
        EXPECT_DEBUG_CALLBACKS(0); // Okay because uninitialized sync object.
        ADD_RESULT(condition == SyncWaitResult::FAILED);

        queue->FenceSync(testFence, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        condition = testFence->Wait(0);
        EXPECT_DEBUG_CALLBACKS(1); // Error because the fence is unflushed.
        ADD_RESULT(condition == SyncWaitResult::FAILED);

        queue->Flush();

        condition = testFence->Wait(0);
        EXPECT_DEBUG_CALLBACKS(0); // Okay because it's been flushed.
        ADD_RESULT(condition != SyncWaitResult::FAILED);

        condition = testFence->Wait(0);
        EXPECT_DEBUG_CALLBACKS(0); // Okay because empty, already-flushed fence object.
        ADD_RESULT(condition != SyncWaitResult::FAILED);

        queue->FenceSync(testFence, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        queue->WaitSync(testFence);
        EXPECT_DEBUG_CALLBACKS(0); // Okay even though there isn't a flush, because the SyncWait is
                                     // on the same queue as the fence, so they'd get flushed together.
        queue->Flush();
        EXPECT_DEBUG_CALLBACKS(0);

        queue2->FenceSync(testFence, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        queue->WaitSync(testFence);
        EXPECT_DEBUG_CALLBACKS(1); // Error because cross-queue fence without a flush.

        queue2->Flush();
        queue->WaitSync(testFence);
        EXPECT_DEBUG_CALLBACKS(0); // Okay now, because the queue the fence comes from has now been flushed.
        queue2->Finish();

        queueCB.FenceSync(testFence, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        condition = testFence->Wait(0);
        EXPECT_DEBUG_CALLBACKS(0); // Okay, command buffer is unsubmitted so fence is still empty.
        ADD_RESULT(condition != SyncWaitResult::FAILED);

        queueCB.submit();
        condition = testFence->Wait(0);
        EXPECT_DEBUG_CALLBACKS(1); // Now the command buffer is submitted, waiting on the fence should
                                    // cause an error.
        ADD_RESULT(condition == SyncWaitResult::FAILED);

        queue2->WaitSync(testFence);
        EXPECT_DEBUG_CALLBACKS(1); // Cross-queue fence without a flush.

        queueCB.WaitSync(testFence);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(0); // Okay because queueCB got submitted to same queue as the fence,
                                     // so the Fence and wait gets flushed together.


        testFence->Free();
        queue2->Finish();
        queue2->Free();
    }

    // Test some max texture size bounds.
    {
        LWNint maxTexSize, maxTexSizeBuf, maxTexSize3D, maxTexSizeLwbemap, maxTexSizeRect, maxLayers;
        device->GetInteger(DeviceInfo::MAX_TEXTURE_SIZE, &maxTexSize);
        device->GetInteger(DeviceInfo::MAX_BUFFER_TEXTURE_SIZE, &maxTexSizeBuf);
        device->GetInteger(DeviceInfo::MAX_3D_TEXTURE_SIZE, &maxTexSize3D);
        device->GetInteger(DeviceInfo::MAX_LWBE_MAP_TEXTURE_SIZE, &maxTexSizeLwbemap);
        device->GetInteger(DeviceInfo::MAX_RECTANGLE_TEXTURE_SIZE, &maxTexSizeRect);
        device->GetInteger(DeviceInfo::MAX_TEXTURE_LAYERS, &maxLayers);
        DEBUG_PRINT(("Reported maxTexSize %d 0x%x\n", maxTexSize, maxTexSize));
        DEBUG_PRINT(("Reported maxTexSizeBuf %d 0x%x\n", maxTexSizeBuf, maxTexSizeBuf));
        DEBUG_PRINT(("Reported maxTexSize3D %d 0x%x\n", maxTexSize3D, maxTexSize3D));
        DEBUG_PRINT(("Reported maxTexSizeLwbemap %d 0x%x\n", maxTexSizeLwbemap, maxTexSizeLwbemap));
        DEBUG_PRINT(("Reported maxTexSizeRect %d 0x%x\n", maxTexSizeRect, maxTexSizeRect));
        DEBUG_PRINT(("Reported maxLayers %d 0x%x\n", maxLayers, maxLayers));

        LWNtextureBuilder *texBuilder = lwnDeviceCreateTextureBuilder(cdevice);
        lwnTextureBuilderSetDefaults(texBuilder);
        lwnTextureBuilderSetSamples(texBuilder, 0);
        lwnTextureBuilderSetFormat(texBuilder, LWN_FORMAT_RGBA8);

        struct MaxTextureSizeTest {
            LWNtextureTarget target;
            LWNint maxWidth;
            LWNint maxHeight;
            LWNint maxDepth;
        } texSizeTestCase[] = {
            {LWN_TEXTURE_TARGET_1D, maxTexSize, 1, 1},
            {LWN_TEXTURE_TARGET_2D, maxTexSize, maxTexSize, 1},
            {LWN_TEXTURE_TARGET_3D, maxTexSize3D, maxTexSize3D, maxTexSize3D},
            {LWN_TEXTURE_TARGET_1D_ARRAY, maxTexSize, maxLayers, 1},
            {LWN_TEXTURE_TARGET_2D_ARRAY, maxTexSize, maxTexSize, maxLayers},
            {LWN_TEXTURE_TARGET_2D_MULTISAMPLE, maxTexSize, maxTexSize, 1},
            {LWN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY, maxTexSize, maxTexSize, maxLayers},
            {LWN_TEXTURE_TARGET_RECTANGLE, maxTexSizeRect, maxTexSizeRect, 1},
            {LWN_TEXTURE_TARGET_LWBEMAP, maxTexSizeLwbemap, maxTexSizeLwbemap, 6},
            {LWN_TEXTURE_TARGET_LWBEMAP_ARRAY, maxTexSizeLwbemap, maxTexSizeLwbemap, 6 * (maxLayers / 6)},
            {LWN_TEXTURE_TARGET_BUFFER, maxTexSizeBuf, 1, 1}
        };
        const int texSizeTestCases = 11;

        for (int i = 0; i < texSizeTestCases; i++) {
            // Test that it's okay with the max size.
            lwnTextureBuilderSetTarget(texBuilder, texSizeTestCase[i].target);
            lwnTextureBuilderSetSize3D(texBuilder,
                texSizeTestCase[i].maxWidth,
                texSizeTestCase[i].maxHeight,
                texSizeTestCase[i].maxDepth);
            lwnTextureBuilderSetSamples(texBuilder, (i == LWN_TEXTURE_TARGET_2D_MULTISAMPLE || i == LWN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY) ? 2 : 0);
            lwnTextureBuilderGetStorageSize(texBuilder);
            EXPECT_DEBUG_CALLBACKS(0);

            // Test that debug layer complains with larger than max size in every dimension.

            LWNtextureTarget target = texSizeTestCase[i].target;
            if (target == LWN_TEXTURE_TARGET_LWBEMAP || target == LWN_TEXTURE_TARGET_LWBEMAP_ARRAY) {
                // Only test square sizes.
                lwnTextureBuilderSetSize3D(texBuilder,
                    texSizeTestCase[i].maxWidth + 1,
                    texSizeTestCase[i].maxHeight + 1,
                    texSizeTestCase[i].maxDepth);
                lwnTextureBuilderGetStorageSize(texBuilder);
                EXPECT_DEBUG_CALLBACKS(1);
            } else {
                lwnTextureBuilderSetSize3D(texBuilder,
                    texSizeTestCase[i].maxWidth + 1,
                    texSizeTestCase[i].maxHeight,
                    texSizeTestCase[i].maxDepth);
                lwnTextureBuilderGetStorageSize(texBuilder);
                EXPECT_DEBUG_CALLBACKS(1);

                lwnTextureBuilderSetSize3D(texBuilder,
                    texSizeTestCase[i].maxWidth,
                    texSizeTestCase[i].maxHeight + 1,
                    texSizeTestCase[i].maxDepth);
                lwnTextureBuilderGetStorageSize(texBuilder);
                EXPECT_DEBUG_CALLBACKS(1);
            }

            if (target == LWN_TEXTURE_TARGET_LWBEMAP_ARRAY) {
                lwnTextureBuilderSetSize3D(texBuilder,
                    texSizeTestCase[i].maxWidth,
                    texSizeTestCase[i].maxHeight,
                    ROUND_UP(texSizeTestCase[i].maxDepth, 6) + 6);
                lwnTextureBuilderGetStorageSize(texBuilder);
                EXPECT_DEBUG_CALLBACKS(1);
            } else if (target != LWN_TEXTURE_TARGET_LWBEMAP) {
                lwnTextureBuilderSetSize3D(texBuilder,
                    texSizeTestCase[i].maxWidth,
                    texSizeTestCase[i].maxHeight,
                    texSizeTestCase[i].maxDepth + 1);
                lwnTextureBuilderGetStorageSize(texBuilder);
                EXPECT_DEBUG_CALLBACKS(1);
            }
        }

        // Test that non-square sizes for lwbemap targets causes error.
        lwnTextureBuilderSetTarget(texBuilder, LWN_TEXTURE_TARGET_LWBEMAP);
        lwnTextureBuilderSetSize3D(texBuilder, 512, 418, 6);
        lwnTextureBuilderSetSamples(texBuilder, 0);
        lwnTextureBuilderGetStorageSize(texBuilder);
        EXPECT_DEBUG_CALLBACKS(1);

        // Test that invalid lwbemap depth causes error.
        lwnTextureBuilderSetSize3D(texBuilder, 512, 512, 3);
        lwnTextureBuilderGetStorageSize(texBuilder);
        EXPECT_DEBUG_CALLBACKS(1);

        // Test that non-square sizes for lwbemap targets causes error.
        lwnTextureBuilderSetTarget(texBuilder, LWN_TEXTURE_TARGET_LWBEMAP_ARRAY);
        lwnTextureBuilderSetSize3D(texBuilder, 418, 512, 6);
        lwnTextureBuilderSetSamples(texBuilder, 0);
        lwnTextureBuilderGetStorageSize(texBuilder);
        EXPECT_DEBUG_CALLBACKS(1);

        // Test that invalid lwbemap array depth causes error.
        lwnTextureBuilderSetSize3D(texBuilder, 512, 512, 7);
        lwnTextureBuilderGetStorageSize(texBuilder);
        EXPECT_DEBUG_CALLBACKS(1);

        lwnTextureBuilderFree(texBuilder);
    }

    // Test that zero-sized textures and linear textures with zero or illegal
    // strides cause errors.
    {
        const int nTestZeroTextureTests = 12;
        MemoryPool *zeroSizeTexMemPool = device->CreateMemoryPool(NULL, 128 * 1024, MemoryPoolType::GPU_ONLY);
        TextureBuilder zeroBuilder;
        zeroBuilder.SetDevice(device);
        zeroBuilder.SetDefaults();
        zeroBuilder.SetTarget(TextureTarget::TARGET_3D);
        zeroBuilder.SetFormat(Format::RGBA8);
        zeroBuilder.SetSize3D(4, 4, 4);
        zeroBuilder.SetStorage(zeroSizeTexMemPool, 0);

        for (int i = 0; i < nTestZeroTextureTests; i++) {
            Texture zeroTex;
            bool callbackExpected = true;   // expect errors unless otherwise noted

            switch (i) {
            case 0:
                // We start with a normal 3D texture, with plausible sizes and
                // formats.  The initial texture initialization should work.
                callbackExpected = false;
                break;
            case 1:
                // Now we start setting sizes with dimensions of zero.
                zeroBuilder.SetSize3D(0, 4, 4);
                break;
            case 2:
                zeroBuilder.SetSize3D(4, 0, 4);
                break;
            case 3:
                zeroBuilder.SetSize3D(4, 4, 0);
                break;
            case 4:
                // We now switch to linear textures, and start with a plausible
                // texture with a nicely aligned stride.  This should succeed.
                zeroBuilder.SetFlags(TextureFlags::LINEAR);
                zeroBuilder.SetTarget(TextureTarget::TARGET_2D);
                zeroBuilder.SetSize2D(8, 8);
                zeroBuilder.SetStride(256);
                callbackExpected = false;
                break;
            case 5:
                zeroBuilder.SetStride(0);       // zero stride illegal
                break;
            case 6:
                zeroBuilder.SetStride(8);       // misaligned stride illegal
                break;
            case 7:
                // stride of 32 is legal for texture but not render target
                zeroBuilder.SetStride(32);
                callbackExpected = false;
                break;
            case 8:
                // We now switch to linear render target, and start with a
                // plausible texture with a nicely aligned stride.  This should
                // succeed.
                zeroBuilder.SetFlags(TextureFlags::LINEAR_RENDER_TARGET);
                zeroBuilder.SetTarget(TextureTarget::TARGET_2D);
                zeroBuilder.SetSize2D(8, 8);
                zeroBuilder.SetStride(256);
                callbackExpected = false;
                break;
            case 9:
                zeroBuilder.SetStride(0);       // zero stride illegal
                break;
            case 10:
                zeroBuilder.SetStride(8);       // misaligned stride illegal
                break;
            case 11:
                // stride of 32 is legal for texture but not render target
                zeroBuilder.SetStride(32);
                break;
            default:
                assert(0);
                break;
            }

            LWNboolean status = zeroTex.Initialize(&zeroBuilder);
            EXPECT_DEBUG_CALLBACKS(callbackExpected);
            if (status) {
                zeroTex.Finalize();
            }
        }

        zeroSizeTexMemPool->Free();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Test that compressed 1D textures cause errors
    {
        MemoryPool *texMemPool = device->CreateMemoryPool(NULL, 1024, MemoryPoolType::GPU_ONLY);
        TextureBuilder builder;
        Texture tex;

        builder.SetDevice(device);
        builder.SetDefaults();
        builder.SetFormat(Format::RGBA_ASTC_4x4);
        builder.SetSize1D(4);
        builder.SetStorage(texMemPool, 0);

        builder.SetTarget(TextureTarget::TARGET_1D);
        tex.Initialize(&builder);
        EXPECT_DEBUG_CALLBACKS(1);

        builder.SetTarget(TextureTarget::TARGET_1D_ARRAY);
        tex.Initialize(&builder);
        EXPECT_DEBUG_CALLBACKS(1);

        texMemPool->Free();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Test that finalizing a texture before finalizing a window causes debug layer error.
    {
        MemoryPool *testGPUTexPool = device->CreateMemoryPool(NULL, 0x20000, MemoryPoolType::GPU_ONLY);
        TextureBuilder *windowTexB = device->CreateTextureBuilder();
        windowTexB->SetDefaults()
                   .SetFlags(TextureFlags::COMPRESSIBLE | TextureFlags::DISPLAY)
                   .SetTarget(TextureTarget::TARGET_2D)
                   .SetSize2D(32, 32)
                   .SetFormat(Format::RGBA8);

        // The minimum textures per window is 2
        const int bufferNum = 2;
        Texture *testWindowTex[bufferNum];

        for (int i = 0; i < bufferNum; i++) {
            testWindowTex[i] = windowTexB->CreateTextureFromPool(testGPUTexPool, 0);
        }
        WindowBuilder *wb = device->CreateWindowBuilder();
        wb->SetDefaults();
        wb->SetTextures(bufferNum, testWindowTex);
        Window *testWindow = wb->CreateWin();
        wb->Free();
        windowTexB->Free();

        testWindowTex[0]->Finalize();
        EXPECT_DEBUG_CALLBACKS(1);
        testWindow->Free();
        for (int i = 0; i < bufferNum; i++) {
            if (testWindowTex[i]) {
                testWindowTex[i]->Free();
                testWindowTex[i] = NULL;
            }
        }
        testGPUTexPool->Free();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Test that texture copies with NULL region errors out.
    {
        MemoryPool *testGPUTexPool = device->CreateMemoryPool(NULL, 0x30000, MemoryPoolType::GPU_ONLY);
        TextureBuilder copyTexBuilder;
        copyTexBuilder.SetDevice(device).SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).SetSize2D(32, 32).SetFormat(Format::RGBA8);
        BufferBuilder bufferTexBuilder;
        bufferTexBuilder.SetDevice(device).SetDefaults();

        Texture *copyTestTexture = copyTexBuilder.CreateTextureFromPool(testGPUTexPool, 0);
        Texture *copyTestTexture2 = copyTexBuilder.CreateTextureFromPool(testGPUTexPool, 0x10000);
        Buffer *copyTestBuffer = bufferTexBuilder.CreateBufferFromPool(testGPUTexPool, 0, 0x20000);

        queueCB.CopyBufferToTexture (copyTestBuffer->GetAddress(), copyTestTexture, NULL, NULL /* dstRegion */, CopyFlags::NONE);
        EXPECT_DEBUG_CALLBACKS(1);

        queueCB.CopyTextureToBuffer (copyTestTexture, NULL, NULL /* srcRegion */, copyTestBuffer->GetAddress(), CopyFlags::NONE);
        EXPECT_DEBUG_CALLBACKS(1);

        queueCB.CopyTextureToTexture(
            copyTestTexture,  NULL, NULL /* srcRegion */,
            copyTestTexture2, NULL, NULL /* dstRegion */,
            CopyFlags::NONE);
        EXPECT_DEBUG_CALLBACKS(1);

        copyTestBuffer->Free();
        copyTestTexture->Free();
        copyTestTexture2->Free();
        testGPUTexPool->Free();
    }

    // Test that stride validation is working
    {
        MemoryPool *testPool = device->CreateMemoryPool(NULL, 0x100, MemoryPoolType::CPU_NON_COHERENT);

        TextureBuilder copyTexBuilder;
        TextureView textureView;

        copyTexBuilder
            .SetDevice(device).SetDefaults()
            .SetTarget(TextureTarget::TARGET_2D)
            .SetSize2D(4, 4)
            .SetFormat(Format::RGBA8);
        Texture *texture = copyTexBuilder.CreateTextureFromPool(testPool, 0);

        textureView.SetDefaults();

        CopyRegion region = { 0, 0, 0, 4, 4, 1 };

        // Scratch buffer to read from when testing. Is 4x4 texels * 4 bytes per texel
        char data[4 * 4 * 4] = { 0 };

        // There should be no problems with using the "default" sizes
        texture->WriteTexelsStrided(&textureView, &region, data, 0, 0);
        EXPECT_DEBUG_CALLBACKS(0);

        // If the row stride is too short we should error
        texture->ReadTexelsStrided(&textureView, &region, data, 15, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        // If the image stride is too short we should error
        texture->ReadTexelsStrided(&textureView, &region, data, 0, 63);
        EXPECT_DEBUG_CALLBACKS(1);

        texture->Free();
        testPool->Free();
    }

#if 0
    // pbrown 4/28/16:  Temporarily disable the test -- Test results in MMU
    // errors on SDEV when enabled; no errors when disabled.
    TestDebugCallbacksShaderSubroutines(device, queue, queueCB);
#endif

    // Test that misaligned shader data control pointer causes error.
    {
        Program *program = device->CreateProgram();
        ShaderData shaderData;
        shaderData.control = (void*) 0x5001; // Misaligned address
        shaderData.data = 0x1442; // Some invalid address.
        program->SetShaders(1, &shaderData);
        EXPECT_DEBUG_CALLBACKS(1);
        program->Free();
    }

    // Tests for memory pool builder storage validation for different pool types.
    for (int pooltype = 0; pooltype < 3; pooltype++) {
        int poolFlags;
        size_t poolSize;
        switch (pooltype) {
        default:
            assert(0);
        case 0:
            poolFlags = (LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
                         LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT);
            poolSize = 4*1024;
            break;
        case 1:
            poolFlags = (LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
                         LWN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS_BIT |
                         LWN_MEMORY_POOL_FLAGS_PHYSICAL_BIT);
            poolSize = 64*1024;
            break;
        case 2:
            poolFlags = (LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                         LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                         LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT);
            poolSize = 64*1024;
            break;
        }

        char *allocMem = (char *) PoolStorageAlloc(poolSize);
        char *mem = (poolFlags & LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT) ? NULL : allocMem;
        LWNmemoryPool pool;
        LWNmemoryPoolBuilder builder;
        lwnMemoryPoolBuilderSetDefaults(&builder);
        lwnMemoryPoolBuilderSetDevice(&builder, cdevice);
        lwnMemoryPoolBuilderSetStorage(&builder, mem, poolSize);
        lwnMemoryPoolBuilderSetFlags(&builder, poolFlags);
        lwnMemoryPoolInitialize(&pool, &builder);
        EXPECT_DEBUG_CALLBACKS(0); // Base case
        lwnMemoryPoolFinalize(&pool);

        // Zero size is checked by the builder setter.
        lwnMemoryPoolBuilderSetStorage(&builder, mem, 0);
        EXPECT_DEBUG_CALLBACKS(1); // Zero size

        // Misaligned size is tested lwnMemoryPoolInitialize.
        lwnMemoryPoolBuilderSetStorage(&builder, mem, 2048);
        lwnMemoryPoolInitialize(&pool, &builder);
        EXPECT_DEBUG_CALLBACKS(1); // Misaligned size

        // For physical/virtual pools, make sure that 32K is considered misaligned.
        if (poolFlags & (LWN_MEMORY_POOL_FLAGS_PHYSICAL_BIT | LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT)) {
            lwnMemoryPoolBuilderSetStorage(&builder, mem, 32768);
            lwnMemoryPoolInitialize(&pool, &builder);
            EXPECT_DEBUG_CALLBACKS(1); // Must be a multiple of the 64KB GPU page size
        }

        // For non-virtual pools, test misaligned pointers.
        if (!(poolFlags & LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT)) {
            assert(allocMem != NULL);
            lwnMemoryPoolBuilderSetStorage(&builder, allocMem + 2048, poolSize);
            lwnMemoryPoolInitialize(&pool, &builder);
            EXPECT_DEBUG_CALLBACKS(1); // Misaligned pointer
        }

        // Test pools with invalid storage pointers (NULL for physical pools,
        // non-NULL for virtual pools).
        mem = (poolFlags & LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT) ? allocMem : NULL;
        lwnMemoryPoolBuilderSetStorage(&builder, mem, poolSize);
#if !defined(LW_TEGRA)
        if (!(poolFlags & LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT)) {
            // On Windows, non-virtual pools can be created with a NULL
            // pointer, but the first such use for a device will trigger
            // a callback warning that it won't work on CheetAh X1.  The
            // callback is just a warning (API call not skipped), so the pool
            // needs to be finalized.
            lwnMemoryPoolInitialize(&pool, &builder);
            EXPECT_DEBUG_CALLBACKS(pooltype == 0 ? 1 : 0);  // driver will only warn on one iteration
            lwnMemoryPoolFinalize(&pool);
            lwnMemoryPoolInitialize(&pool, &builder);
            EXPECT_DEBUG_CALLBACKS(0);
            lwnMemoryPoolFinalize(&pool);
        } else
#endif
        {
            lwnMemoryPoolInitialize(&pool, &builder);
            EXPECT_DEBUG_CALLBACKS(1);
            lwnMemoryPoolInitialize(&pool, &builder);
            EXPECT_DEBUG_CALLBACKS(1);
        }

        PoolStorageFree(allocMem);
    }

    // Sanity test ZLwll storage size queries for both textures and texture
    // builders.  Test different formats (including invalid ones), array and
    // non-array targets, and different sample counts.
    if (lwogCheckLWNAPIVersion(52, 3)) {

        int sampleCounts[] = { 0, 2, 4 };
        Format formats[] = {
            Format::DEPTH16,
            Format::DEPTH24,
            Format::DEPTH24_STENCIL8,
            Format::DEPTH32F,
            Format::DEPTH32F_STENCIL8,
            Format::RGBA8,                  // Test an invalid depth format
        };
        MemoryPoolFlags poolFlags = (MemoryPoolFlags::CPU_NO_ACCESS |
                                     MemoryPoolFlags::GPU_CACHED |
                                     MemoryPoolFlags::COMPRESSIBLE);
        MemoryPool *testPool = device->CreateMemoryPoolWithFlags(NULL, 16*1024*1024, poolFlags);

        TextureBuilder tb;
        tb.SetDevice(device).SetDefaults().SetLevels(1).SetStorage(testPool, 0);
        tb.SetFlags(TextureFlags::COMPRESSIBLE);
        tb.SetSize2D(256, 256);
        for (int isArray = 0; isArray < 2; isArray++) {
            for (uint32_t sampleIndex = 0; sampleIndex < __GL_ARRAYSIZE(sampleCounts); sampleIndex++) {
                int samples = sampleCounts[sampleIndex];
                tb.SetSamples(samples);
                if (isArray) {
                    tb.SetTarget(samples ? TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY : TextureTarget::TARGET_2D_ARRAY);
                    tb.SetDepth(4);
                } else {
                    tb.SetTarget(samples ? TextureTarget::TARGET_2D_MULTISAMPLE : TextureTarget::TARGET_2D);
                    tb.SetDepth(1);
                }
                for (uint32_t formatIndex = 0; formatIndex < __GL_ARRAYSIZE(formats); formatIndex++) {
                    Texture tex;
                    Format fmt = formats[formatIndex];
                    bool validFormat = (fmt != Format::RGBA8);
                    tb.SetFormat(fmt);

                    // Query the storage size from the texture builder, which
                    // should only fail if we're using an invalid format.
                    size_t tbQuerySize = tb.GetZLwllStorageSize();
                    EXPECT_DEBUG_CALLBACKS(validFormat ? 0 : 1);

                    // Now set up a texture object and query the storage size
                    // there, too.  Again, this should only fail if we have an
                    // invalid format.
                    tex.Initialize(&tb);
                    EXPECT_DEBUG_CALLBACKS(0);
                    size_t texQuerySize = tex.GetZLwllStorageSize();
                    tex.Finalize();
                    EXPECT_DEBUG_CALLBACKS(validFormat ? 0 : 1);

                    // Verify that the returned sizes match for the texture
                    // and builder.  Also verify that we get a non-zero size
                    // for valid formats and a zero size for invalid ones.
                    ADD_RESULT(tbQuerySize == texQuerySize);
                    ADD_RESULT(validFormat ? (tbQuerySize > 0) : (tbQuerySize == 0));
                }
            }
        }
        testPool->Free();
    }

    // Test that failed initializations don't register objects with the debug layer.
    {
        LWNboolean result;
        MemoryPool mp;
        MemoryPoolBuilder mpb;

        mpb.SetDevice(device).SetDefaults();
        mpb.SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED |
                     MemoryPoolFlags::VIRTUAL);

        // Initialize a memory pool with an illegal size. This should fail, and
        // the debug layer should NOT record the pool as a valid object.
        mpb.SetStorage(NULL, 64*1024 + 1);
        result = mp.Initialize(&mpb);
        EXPECT_DEBUG_CALLBACKS(1);
        ADD_RESULT(result == LWN_FALSE);

        // Querying the buffer address of a non-valid pool should fail.
        if (m_debugLevel >= DEBUG_FEATURE_OBJECT_VALIDATION) {
            (void)mp.GetBufferAddress();
            EXPECT_DEBUG_CALLBACKS(1);
        }

        // Initialize a memory pool with a legal size using the same memory as
        // the previous one. This should succeed, where the previous (failed)
        // use of the same memory shouldn't mess up the call.
        mpb.SetStorage(NULL, 64*1024);
        result = mp.Initialize(&mpb);
        EXPECT_DEBUG_CALLBACKS(0);
        ADD_RESULT(result == LWN_TRUE);

        // Querying the buffer address of the now-valid pool should succeed.
        (void)mp.GetBufferAddress();
        EXPECT_DEBUG_CALLBACKS(0);

        // Creating a buffer that's too big for the pool should fail.
        BufferBuilder bb;
        bb.SetDevice(device).SetDefaults();
        bb.SetStorage(&mp, 0, 128*1024);
        Buffer buf;
        result = buf.Initialize(&bb);
        EXPECT_DEBUG_CALLBACKS(1);
        ADD_RESULT(result == LWN_FALSE);

        // Creating a texture that's too big for the pool should fail.
        TextureBuilder tb;
        tb.SetDevice(device).SetDefaults();
        tb.SetTarget(TextureTarget::TARGET_2D).
            SetFormat(Format::RGBA8).
            SetStorage(&mp, 0).
            SetSize2D(512,512);     // too big!
        Texture tex;
        result = tex.Initialize(&tb);
        EXPECT_DEBUG_CALLBACKS(1);
        ADD_RESULT(result == LWN_FALSE);

        // Finalizing the pool shouldn't fail due to unfinalized buffers and
        // textures because neither was created successfully.
        mp.Finalize();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Test the allocation of textures from different kinds of memory pools
    // (PHYSICAL/VIRTUAL/normal) with different combinations of compressible
    // textures and pools.
    {
        for (int pti = 0; pti < 3; pti++) {
            for (int cpi = 0; cpi < 2; cpi++) {
                MemoryPoolFlags mpf; 
                switch (pti) {
                default:
                    assert(0);
                case 0:
                    mpf = MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED;
                    break;
                case 1:
                    mpf = (MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_NO_ACCESS |
                           MemoryPoolFlags::PHYSICAL);
                    break;
                case 2:
                    mpf = (MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED |
                           MemoryPoolFlags::VIRTUAL);
                    break;
                }
                if (cpi) {
                    mpf |= MemoryPoolFlags::COMPRESSIBLE;
                }

                MemoryPool *pool = device->CreateMemoryPoolWithFlags(NULL, 128*1024, mpf);
                EXPECT_DEBUG_CALLBACKS(0);

                for (int cti = 0; cti < 2; cti++) {

                    bool shouldSucceed = true;
                    if (mpf & MemoryPoolFlags::PHYSICAL) {
                        // Can't create textures in physical pools.
                        shouldSucceed = false;
                    }
                    if (cti && !cpi && !(mpf & MemoryPoolFlags::VIRTUAL)) {
                        // Compressible textures require compressible (or
                        // virtual) pools.
                        shouldSucceed = false;
                    }

                    TextureBuilder tb;
                    tb.SetDevice(device).SetDefaults().
                        SetTarget(TextureTarget::TARGET_2D).
                        SetSize2D(4, 4).
                        SetFormat(Format::RGBA8);
                    if (cti) {
                        tb.SetFlags(TextureFlags::COMPRESSIBLE);
                    }

                    Texture *tex = tb.CreateTextureFromPool(pool, 0);
                    EXPECT_DEBUG_CALLBACKS(shouldSucceed ? 0 : 1);
                    if (tex) {
                        tex->Free();
                    }
                }

                pool->Free();
                EXPECT_DEBUG_CALLBACKS(0);
            }
        }
    }

    // Test for missing calls to SetDefaults, missing calls to SetDevice,
    // and SetDevice called with invalid parameters. Each block of code
    // uses a different object type, and tests four different combinations:
    //
    // - builder initialized to 0x00 with no device set (NULL device)
    // - builder initialized to 0xFF with no device set (invalid device)
    // - builder initialized with a valid device
    // - builder without calling SetDefaults
    {
        // Create a temporary virtual memory pool for buffer and texture
        // testing, since we need some storage.
        MemoryPool mp;
        MemoryPoolBuilder mpb;
        mpb.SetDevice(device).SetDefaults();
        mpb.SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED |
                     MemoryPoolFlags::VIRTUAL);
        mpb.SetStorage(NULL, 64*1024);
        mp.Initialize(&mpb);
        EXPECT_DEBUG_CALLBACKS(0);

        TestSetTracker<BufferBuilder, Buffer>(device, &mp);
        TestSetTracker<TextureBuilder, Texture>(device, &mp);
        TestSetTracker<SamplerBuilder, Sampler>(device, &mp);
        TestSetTracker<MemoryPoolBuilder, MemoryPool>(device, nullptr);
        TestSetTracker<QueueBuilder, Queue>(device, &mp);

        mp.Finalize();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Test ReadTexels/WriteTexels on compressible textures (bug 1765684).
    {
        // Create a memory pool for the texture.
        MemoryPool mp;
        MemoryPoolBuilder mpb;
        void *poolMem = PoolStorageAlloc(64 * 1024);
        mpb.SetDevice(device).SetDefaults();
        mpb.SetFlags(MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::GPU_CACHED |
                     MemoryPoolFlags::COMPRESSIBLE);
        mpb.SetStorage(poolMem, 64*1024);
        mp.Initialize(&mpb);
        EXPECT_DEBUG_CALLBACKS(0);

        for (int compressible = 0; compressible < 2; compressible++) {
            TextureBuilder tb;
            tb.SetDefaults().SetDevice(device);
            tb.SetFormat(Format::RGBA8);
            tb.SetTarget(TextureTarget::TARGET_2D);
            tb.SetSize2D(4, 4);
            if (compressible) {
                tb.SetFlags(TextureFlags::COMPRESSIBLE);
            }
            tb.SetStorage(&mp, 0);

            Texture tex;
            LWNboolean initialized = tex.Initialize(&tb);
            EXPECT_DEBUG_CALLBACKS(0);

            if (initialized) {

                // Test calling ReadTexels and WriteTexels to read/write a
                // single texel of the texture.  This should fail if the
                // texture is compressible, but generate no errors if
                // non-compressible.
                CopyRegion region = { 0, 0, 0, 1, 1, 1 };
                char garbage[16];
                tex.ReadTexels(NULL, &region, garbage);
                EXPECT_DEBUG_CALLBACKS(compressible ? 1 : 0);
                tex.WriteTexels(NULL, &region, garbage);
                EXPECT_DEBUG_CALLBACKS(compressible ? 1 : 0);

                tex.Finalize();
            }
        }

        mp.Finalize();
        EXPECT_DEBUG_CALLBACKS(0);
        PoolStorageFree(poolMem);
    }

    // Verify that we can query version information with a NULL device.
    {
        Device *nullDevice = NULL;
        DeviceInfo versionProperties[] = {
            DeviceInfo::API_MAJOR_VERSION,
            DeviceInfo::API_MINOR_VERSION,
            DeviceInfo::GLSLC_MIN_SUPPORTED_GPU_CODE_MAJOR_VERSION,
            DeviceInfo::GLSLC_MIN_SUPPORTED_GPU_CODE_MINOR_VERSION,
            DeviceInfo::GLSLC_MAX_SUPPORTED_GPU_CODE_MAJOR_VERSION,
            DeviceInfo::GLSLC_MAX_SUPPORTED_GPU_CODE_MINOR_VERSION,
        };
        for (size_t i = 0; i < __GL_ARRAYSIZE(versionProperties); i++) {
            int regularQuery, nullQuery;
            device->GetInteger(versionProperties[i], &regularQuery);
            nullDevice->GetInteger(versionProperties[i], &nullQuery);
            EXPECT_DEBUG_CALLBACKS(0);
            ADD_RESULT(regularQuery == nullQuery);
        }
    }

    // Test ReadTexels/WriteTexels on linear textures generates error.
    {
        MemoryPoolAllocator pool(device, NULL, 0x10000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);
        TextureBuilder textureBuilder;
        textureBuilder.SetDefaults()
                      .SetDevice(device)
                      .SetTarget(TextureTarget::TARGET_2D)
                      .SetFormat(Format::RGBA8)
                      .SetFlags(TextureFlags::LINEAR)
                      .SetStride(16 * 4)
                      .SetSize3D(16, 16, 1);
        Texture *texture = pool.allocTexture(&textureBuilder);
        assert(texture);
        static char dummyTextureData[16 * 16 * 4];
        CopyRegion region = { 0, 0, 0, 16, 16, 1 };
        EXPECT_DEBUG_CALLBACKS(0);
        texture->WriteTexels(NULL,& region, dummyTextureData);
        EXPECT_DEBUG_CALLBACKS(1);
        texture->ReadTexels(NULL,& region, dummyTextureData);
        EXPECT_DEBUG_CALLBACKS(1);
    }

    // Test that MINIMAL_LAYOUT invalid flags generates error.
    {
        MemoryPool *tpool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);

        // Create a texture and copy it.
        TextureBuilder texb;
        texb.SetDefaults()
            .SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(64, 64)
            .SetStride(64 * 4)
            .SetFlags(TextureFlags::MINIMAL_LAYOUT | TextureFlags::LINEAR)
            .SetStorage(tpool, 0);

        Texture texture;
        texture.Initialize(&texb);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetFlags(TextureFlags::MINIMAL_LAYOUT | TextureFlags::SPARSE);
        texture.Initialize(&texb);
        EXPECT_DEBUG_CALLBACKS(1);

        texb.SetFlags(TextureFlags::MINIMAL_LAYOUT | TextureFlags::VIDEO_DECODE);
        texture.Initialize(&texb);
        EXPECT_DEBUG_CALLBACKS(1);

        tpool->Free();
    }

    // Test that copytexture to mis-aligned pitch textures are caught.
    // This would otherwise cause GPU channel errors.
    {
        MemoryPool *tpool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);
        MemoryPool *tpool2 = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);

        TextureBuilder texb;
        texb.SetDefaults()
            .SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(64, 64)
            .SetFlags(TextureFlags::LINEAR)
            .SetStorage(tpool, 0);

        Texture textureSrc;
        texb.SetStride(64 * 4);
        textureSrc.Initialize(&texb);
        EXPECT_DEBUG_CALLBACKS(0);

        Texture texture;
        texb.SetStride(64 * 4 + 32)
            .SetStorage(tpool2, 0);
        texture.Initialize(&texb);
        EXPECT_DEBUG_CALLBACKS(0);

        CopyRegion region = { 0, 0, 0, 64, 64, 1 };
        queueCB.CopyTextureToTexture(&textureSrc, NULL, &region, &texture, NULL, &region, 0);
        EXPECT_DEBUG_CALLBACKS(1);

        textureSrc.Finalize();
        texture.Finalize();
        tpool->Free();
        tpool2->Free();
    }

    // Test various Add{Command, Control}Memory cases.
    {
        CommandBuffer *cmd = device->CreateCommandBuffer();
        MemoryPool *testGPUPool = device->CreateMemoryPool(NULL, 0x1000, MemoryPoolType::CPU_COHERENT);

        // Expect no warning or error for adding too little when no callback set.
        cmd->AddCommandMemory(testGPUPool, 0,  4);
        EXPECT_DEBUG_CALLBACKS(0);

        // Expect warning for adding too little when callback is set.
        cmd->SetMemoryCallback(CallbackCmdbufOOMDummy);
        cmd->AddCommandMemory(testGPUPool, 0,  4);
        EXPECT_DEBUG_CALLBACKS(1);

        cmd->Free();
        testGPUPool->Free();
    }

    // Test that depth/stencil textures must have compressed bit set.
    {
        MemoryPool *tpool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);
        Texture texture;
        TextureBuilder texb;
        texb.SetDefaults()
            .SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetSize2D(64, 64)
            .SetStorage(tpool, 0);

        texture.Initialize(&texb.SetFormat(Format::STENCIL8));
        EXPECT_DEBUG_CALLBACKS(1);
        texture.Initialize(&texb.SetFormat(Format::DEPTH16));
        EXPECT_DEBUG_CALLBACKS(1);
        texture.Initialize(&texb.SetFormat(Format::DEPTH24));
        EXPECT_DEBUG_CALLBACKS(1);
        texture.Initialize(&texb.SetFormat(Format::DEPTH32F));
        EXPECT_DEBUG_CALLBACKS(1);
        texture.Initialize(&texb.SetFormat(Format::DEPTH24_STENCIL8));
        EXPECT_DEBUG_CALLBACKS(1);
        texture.Initialize(&texb.SetFormat(Format::DEPTH32F_STENCIL8));
        EXPECT_DEBUG_CALLBACKS(1);

        tpool->Free();
    }

    // Test if the out of memory in VRAM calls debug callbacks on Windows
#if BUG_2041101_FIXED

#if defined(LW_WINDOWS)
    {
        MemoryPool *pool;
        std::vector<MemoryPool*> pools;

        // Alloc VRAM until fail to allocate due to out of memory.
        while ((pool = device->CreateMemoryPool(NULL, 0x8000000, MemoryPoolType::GPU_ONLY)) != NULL) {
            pools.push_back(pool);
        }
        EXPECT_DEBUG_CALLBACKS(1);

        for (size_t i = 0; i < pools.size(); i++) {
            pools[i]->Free();
        }
        pools.clear();
    }
#endif

#endif // BUG_2041101_FIXED

    device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
    device->InstallDebugCallback(Callback2, NULL, LWN_FALSE);
    queueCB.submit();
    queue->Finish();
    tpool->Free();
}

void LWNDebugAPITest::TestDebugCallbacksDrawTimeValidations(Device *device, Queue *queue,
                                                            QueueCommandBuffer& queueCB)
{
    if (m_debugLevel < DEBUG_FEATURE_DRAW_TIME_VALIDATION) {
        // Not enabled.
        DEBUG_PRINT(("Skipping draw time validation tests.\n"));
        return;
    }
    LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);

    // Make sure any previous commands are flushed.
    queueCB.submit();
    queue->Finish();

    // Install the debug callback.
    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);

    // Test that draw with missing program cause DTV error.
    {
        // This should be stopped by the missing program.
        // HW won't let us actually deactivate the vertex stage.
        queueCB.BindProgram(NULL, (ShaderStageBits::ALL_GRAPHICS_BITS & ~ShaderStageBits::VERTEX));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
        queueCB.submit();

        EXPECT_DEBUG_CALLBACKS(1);
    }

    // Create own own command buffer because the shared queueCB already has a texture / sampler pool,
    // then test that binding a texture with no sampler or texture ID pool bound causes debug error.
    {
        Queue *queue2 = device->CreateQueue();
        lwnUtil::CompletionTracker *tracker2 = new lwnUtil::CompletionTracker(cdevice, 32);
        lwnUtil::QueueCommandBufferBase queueCB2Base;
        queueCB2Base.init(cdevice, reinterpret_cast<LWNqueue *>(queue2), tracker2);

        // Should complain because no texture or sampler pool bound.
        QueueCommandBuffer &queueCB2 = queueCB2Base;
        queueCB2.BindTexture(ShaderStage::FRAGMENT, 0, device->GetTextureHandle(312, 310));
        queueCB2.submit();
        EXPECT_DEBUG_CALLBACKS(1);

        queueCB2Base.destroy();
        queue2->Free();
        delete tracker2;
    }

    // Test that BindSeparateTexture triggers debug layer error when using a SeparateSampler handle.
    {
        SamplerBuilder sbuilder;
        sbuilder.SetDefaults()
                .SetDevice(device);

        Sampler *tempSampler = sbuilder.CreateSampler();

        // Create a texture to DrawTexture to.
        MemoryPool *tpool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);
        TextureBuilder texb;
        texb.SetDefaults()
            .SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(64, 64);
        Texture *tempTexture = texb.CreateTextureFromPool(tpool, 0);

        // Create some separate sampler/texture handles.
        SeparateTextureHandle sepTexHandle = device->getSeparateTextureHandle(tempTexture->GetRegisteredTextureID());
        SeparateSamplerHandle sepSampHandle = device->getSeparateSamplerHandle(tempSampler->GetRegisteredID());

        // Create invalid handles by using values of the other type's handle.
        SeparateTextureHandle ilwalidSepTexHandle = { sepSampHandle.value };
        SeparateSamplerHandle ilwalidSepSampHandle = { sepTexHandle.value };

        // Test that bindSeparateTexture triggers two debug layer errors: one for not having the proper device flags set, and
        // one for the input not being a proper texture handle (we send in a *sampler* handle here).
        queueCB.bindSeparateTexture(ShaderStage::FRAGMENT, 0, ilwalidSepTexHandle);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(2);
        
        // Test that bindSeparateSampler triggers two debug layer errors: one for not having the proper device flags set, and
        // one for the input not being a proper sampler handle (we send in a *texture* handle here).
        queueCB.bindSeparateSampler(ShaderStage::FRAGMENT, 0, ilwalidSepSampHandle);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(2);

        // Tests that the multi-bind functions for separate samplers/texture return 7 errors here: 1) because the device wasn't
        // created with the proper device flags, 2) because the range of input bindings exceeds the maximum range for each shader stage,
        // (128 for separate textures, 32 for separate samplers), and 3-7) each handle in the array is an invalid entry.
        SeparateTextureHandle ilwalidSepTexHandles[5] =
            { ilwalidSepTexHandle, ilwalidSepTexHandle, ilwalidSepTexHandle, ilwalidSepTexHandle, ilwalidSepTexHandle };
        SeparateSamplerHandle ilwalidSepSampHandles[5] =
            { ilwalidSepSampHandle, ilwalidSepSampHandle, ilwalidSepSampHandle, ilwalidSepSampHandle, ilwalidSepSampHandle };

        // Bind out of range, with 5 invalid separate texture handles.
        queueCB.BindSeparateTextures(ShaderStage::FRAGMENT, 125, 5, ilwalidSepTexHandles);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(7);
        // Bind out of range, with 5 invalid separate sampler handles.
        queueCB.BindSeparateSamplers(ShaderStage::FRAGMENT, 31, 5, ilwalidSepSampHandles);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(7);

        tempSampler->Free();
        tempTexture->Free();
        tpool->Free();
    }


    // Test that DrawTexture with invalid sampler parameters error out.
    {
        SamplerBuilder sbuilder;
        sbuilder.SetDefaults()
                .SetDevice(device)
                .SetWrapMode(WrapMode::CLAMP, WrapMode::CLAMP, WrapMode::CLAMP);

        // Nonsense filters should spew an error.
        sbuilder.SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::NEAREST);
        Sampler *ilwalidDrawSampler1 = sbuilder.CreateSampler();
        
        // Unsupported wrap mode.
        sbuilder.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST)
                .SetWrapMode(WrapMode::REPEAT, WrapMode::REPEAT, WrapMode::REPEAT);
        Sampler *ilwalidDrawSampler2 = sbuilder.CreateSampler();

        // Invalid compare mode.
        sbuilder.SetWrapMode(WrapMode::CLAMP, WrapMode::CLAMP, WrapMode::CLAMP)
                .SetCompare(CompareMode::COMPARE_R_TO_TEXTURE, CompareFunc::LESS);
        Sampler *ilwalidDrawSampler3 = sbuilder.CreateSampler();

        sbuilder.SetCompare(CompareMode::NONE, CompareFunc::LESS)
                .SetMaxAnisotropy(8.0f);
        Sampler *ilwalidDrawSampler4 = sbuilder.CreateSampler();

        // Create a texture to DrawTexture to.
        MemoryPool *tpool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);
        TextureBuilder texb;
        texb.SetDefaults()
            .SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(64, 64);
        Texture *tempTexture = texb.CreateTextureFromPool(tpool, 0);

        // Figure out texture handles.
        TextureHandle texHandle0 = device->GetTextureHandle(tempTexture->GetRegisteredTextureID(), 0xFFF);
        TextureHandle texHandle1 = device->GetTextureHandle(tempTexture->GetRegisteredTextureID(), ilwalidDrawSampler1->GetRegisteredID());
        TextureHandle texHandle2 = device->GetTextureHandle(tempTexture->GetRegisteredTextureID(), ilwalidDrawSampler2->GetRegisteredID());
        TextureHandle texHandle3 = device->GetTextureHandle(tempTexture->GetRegisteredTextureID(), ilwalidDrawSampler3->GetRegisteredID());
        TextureHandle texHandle4 = device->GetTextureHandle(tempTexture->GetRegisteredTextureID(), ilwalidDrawSampler4->GetRegisteredID());

        // Submit everything everything until now so ONLY the error-ed CommandHandle gets skipped
        // by draw-time validation.
        queueCB.submit();
        queue->Finish();

        // Attempt to DrawTexture.
        DrawTextureRegion region = { 0.0f, 0.0f, 64.0f, 64.0f };
        queueCB.DrawTexture(texHandle0, &region, &region);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(1);
        queueCB.DrawTexture(texHandle1, &region, &region);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(1);
        queueCB.DrawTexture(texHandle2, &region, &region);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(1);
        queueCB.DrawTexture(texHandle3, &region, &region);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(1);
        queueCB.DrawTexture(texHandle4, &region, &region);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(1);

        ilwalidDrawSampler1->Free();
        ilwalidDrawSampler2->Free();
        ilwalidDrawSampler3->Free();
        ilwalidDrawSampler4->Free();
        tempTexture->Free();
        tpool->Free();
    }

    {
        MemoryPool *tpool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);

        // Create a texture and copy it.
        TextureBuilder texb;
        texb.SetDefaults()
            .SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(64, 64);
        Texture *tempTexture = texb.CreateTextureFromPool(tpool, 0);

        // Attempting to use the texture shouldn't cause debug layer error if they are correctly tracked
        // by sequence number.
        Texture tempTextureCopy;
        memcpy(reinterpret_cast<void*>(&tempTextureCopy), reinterpret_cast<void*>(tempTexture), sizeof(Texture));
        tempTextureCopy.GetWidth();
        tempTextureCopy.GetFormat();
        ADD_RESULT(tempTexture->GetDebugID() == tempTextureCopy.GetDebugID());
        EXPECT_DEBUG_CALLBACKS(0);

        // We destroyed the original texture, so the copy should be invalid as well.
        tempTexture->Free();
        tempTextureCopy.GetWidth();
        EXPECT_DEBUG_CALLBACKS(1);

        // Create a buffer and copy it.
        BufferBuilder bufb;
        bufb.SetDevice(device).SetDefaults();
        Buffer *buffer = bufb.CreateBufferFromPool(tpool, 0, 0x1000);

        // Attempting to use the buffer shouldn't cause debug layer error if they are correctly tracked
        // by sequence number.
        Buffer bufferCopy;
        memcpy(reinterpret_cast<void*>(&bufferCopy), reinterpret_cast<void*>(buffer), sizeof(Buffer));
        bufferCopy.GetAddress();
        ADD_RESULT(buffer->GetDebugID() == bufferCopy.GetDebugID());
        EXPECT_DEBUG_CALLBACKS(0);

        // We destroyed the original buffer, so the copy should be invalid as well.
        buffer->Free();
        bufferCopy.GetAddress();
        EXPECT_DEBUG_CALLBACKS(1);

        tpool->Free();

        // Create a sampler and copy it.
        SamplerBuilder samplerBuilder;
        samplerBuilder.SetDevice(device).SetDefaults();
        Sampler *sampler = samplerBuilder.CreateSampler();

        // Attempting to use the sampler shouldn't cause debug layer error if they are correctly tracked
        // by sequence number.
        Sampler samplerCopy;
        memcpy(reinterpret_cast<void*>(&samplerCopy), reinterpret_cast<void*>(sampler), sizeof(Sampler));
        samplerCopy.GetLodBias();
        ADD_RESULT(sampler->GetDebugID() == samplerCopy.GetDebugID());
        EXPECT_DEBUG_CALLBACKS(0);

        // We destroyed the original sampler, so the copy should be invalid as well.
        sampler->Free();
        samplerCopy.GetLodBias();
        EXPECT_DEBUG_CALLBACKS(1);
    }

    // Test that textures in GPU only pools cannot be used with read/write texels.
    {
        MemoryPool *tpool = device->CreateMemoryPool(NULL, 0x10000, MemoryPoolType::GPU_ONLY);

        // Create a texture and copy it.
        TextureBuilder texb;
        texb.SetDefaults()
            .SetDevice(device)
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetSize2D(64, 64);
        Texture *tex = texb.CreateTextureFromPool(tpool, 0);

        static char dummyTextureData[64*64*4];
        CopyRegion region = { 0, 0, 0, 64, 64, 1 };
        tex->WriteTexels(NULL, &region, dummyTextureData);
        EXPECT_DEBUG_CALLBACKS(1);
        tex->ReadTexels(NULL, &region, dummyTextureData);
        EXPECT_DEBUG_CALLBACKS(1);

        tex->Free();
        tpool->Free();
    }

    // Test zero-sized cmdbufs don't cause problems with in-flight memory checks.
    if (m_debugLevel >= DEBUG_FEATURE_IN_FLIGHT_CMDBUF_TRACKING) {
        const TexturePool *cppTexPool = g_lwnTexIDPool->GetTexturePool();
        const SamplerPool *cppSmpPool = g_lwnTexIDPool->GetSamplerPool();

        queueCB.submit();
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(0);

        queueCB.SetSamplerPool(cppSmpPool);
        queueCB.SetTexturePool(cppTexPool);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(0);

        queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(0);
    } else {
        DEBUG_PRINT(("Skipping in-flight cmdbuf tracking.\n"));
    }

    // Test that deleting physical pools with active virtual mappings give a warning.
    if (m_debugLevel >= DEBUG_FEATURE_VIRTUAL_MEMPOOL_MAPPING_TRACKING) {
        MemoryPoolFlags vPoolFlags = (MemoryPoolFlags::CPU_NO_ACCESS |
                                      MemoryPoolFlags::GPU_CACHED |
                                      MemoryPoolFlags::VIRTUAL);
        MemoryPoolFlags pPoolFlags = (MemoryPoolFlags::CPU_CACHED |
                                      MemoryPoolFlags::GPU_NO_ACCESS |
                                      MemoryPoolFlags::PHYSICAL);
        MemoryPool *vPool1 = NULL, *vPool2 = NULL, *pPool1 = NULL, *pPool2 = NULL;

        TextureBuilder tb;
        tb.SetDevice(device)
            .SetDefaults()
            .SetTarget(TextureTarget::TARGET_2D)
            .SetSize2D(64, 64)
            .SetFormat(Format::RGBA8);

        static const int testRequestSize = 0x20000;

        MappingRequest req[2] = { { 0 }, { 0 }, };
        // Mapping to pPool1
        req[0].virtualOffset = 0;
        req[0].physicalOffset = 0;
        req[0].size = testRequestSize;
        req[0].storageClass = tb.GetStorageClass();

        // Mapping to pPool2
        req[1].virtualOffset = 0;
        req[1].physicalOffset = 0;
        req[1].size = testRequestSize;
        req[1].storageClass = tb.GetStorageClass();

        // 1-1 mapping
        // If finalizes physical pool first, generates a warning.
        vPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        pPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        req[0].physicalPool = pPool1;

        vPool1->MapVirtual(1, &req[0]);

        pPool1->Free();
        vPool1->Free();
        EXPECT_DEBUG_CALLBACKS(2);

        // If finalizes virtual pool first, passes.
        vPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        pPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        req[0].physicalPool = pPool1;

        vPool1->MapVirtual(1, &req[0]);

        vPool1->Free();
        pPool1->Free();
        EXPECT_DEBUG_CALLBACKS(0);

        // 2-1 mapping
        // If finalizes physical pool first, generates a warning.
        vPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        vPool2 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        pPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        req[0].physicalPool = pPool1;

        vPool1->MapVirtual(1, &req[0]);
        vPool2->MapVirtual(1, &req[0]);

        pPool1->Free();
        vPool1->Free();
        vPool2->Free();
        EXPECT_DEBUG_CALLBACKS(3);

        // If finalizes virtual pool first, passes.
        vPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        vPool2 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        pPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        req[0].physicalPool = pPool1;

        vPool1->MapVirtual(1, &req[0]);
        vPool2->MapVirtual(1, &req[0]);

        vPool1->Free();
        vPool2->Free();
        pPool1->Free();
        EXPECT_DEBUG_CALLBACKS(0);

        // 1-2 mapping
        // If finalizes physical pool first, generates a warning.
        vPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        pPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        pPool2 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        req[0].physicalPool = pPool1;
        req[1].physicalPool = pPool2;

        vPool1->MapVirtual(1, &req[0]);
        vPool1->MapVirtual(1, &req[1]); // The second map covered the first one.

        pPool2->Free();
        vPool1->Free();
        pPool1->Free();
        EXPECT_DEBUG_CALLBACKS(2);

        // Unmapping before finalizing, passes.
        vPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        vPool2 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, vPoolFlags);
        pPool1 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        pPool2 = device->CreateMemoryPoolWithFlags(NULL, testRequestSize, pPoolFlags);
        req[0].physicalPool = pPool1;
        req[1].physicalPool = pPool2;

        vPool1->MapVirtual(2, req);
        vPool2->MapVirtual(2, req);

        req[0].physicalPool = NULL; // Unmap all pages
        req[0].storageClass = 0;
        req[1].physicalPool = NULL;
        req[1].storageClass = 0;

        vPool1->MapVirtual(1, &req[0]);
        vPool2->MapVirtual(1, &req[0]);

        pPool2->Free();
        pPool1->Free();
        vPool2->Free();
        vPool1->Free();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Test that in-flight cmdbuf tracking gets updated when presenting a texture.
#if LWN_DEBUG_DISABLE_WINDOW_PRESENT_INFLIGHT_MEMORY_TEST == 0
    if (m_debugLevel >= DEBUG_FEATURE_IN_FLIGHT_CMDBUF_TRACKING) {
        const int controlAndCommandSize = 0x2000;
        Queue *queue2 = device->CreateQueue();
        MemoryPool *pool = device->CreateMemoryPool(NULL, controlAndCommandSize, MemoryPoolType::CPU_COHERENT);
        void *controlMem = malloc(controlAndCommandSize);

        CommandBuffer* cmdbuf = device->CreateCommandBuffer();
        cmdbuf->AddControlMemory(controlMem, controlAndCommandSize);

        // First, fill an initial range with some commands.
        ColorState cstate;
        cstate.SetDefaults();
        {
            cmdbuf->AddCommandMemory(pool, 0, controlAndCommandSize);
            cmdbuf->BeginRecording();
            for (int i = 0; i < 32; i++) {
                cmdbuf->BindColorState(&cstate);
            }
            CommandHandle handle = cmdbuf->EndRecording();
            queue2->SubmitCommands(1, &handle);
            EXPECT_DEBUG_CALLBACKS(0);
        }

        // Fill a conflicting range with commands.
        CommandHandle handle2;
        {
            cmdbuf->AddCommandMemory(pool, 32, controlAndCommandSize / 2);
            cmdbuf->BeginRecording();
            for (int i = 0; i < 256; i++) {
                cmdbuf->BindColorState(&cstate);
            }
            handle2 = cmdbuf->EndRecording();
            queue2->SubmitCommands(1, &handle2);
            EXPECT_DEBUG_CALLBACKS(1);
        }

        // Test that re-acquiring the window texture syncs the queue.
        {
           g_lwnWindowFramebuffer.present((LWNqueue*)queue2);
            for (int i = 0; i <= g_lwnWindowFramebuffer.getNumBuffers(); i++) {
                g_lwnWindowFramebuffer.bind();
                g_lwnWindowFramebuffer.present();
            }
            EXPECT_DEBUG_CALLBACKS(0);
            queue2->SubmitCommands(1, &handle2);
            EXPECT_DEBUG_CALLBACKS(0);
        }

        cmdbuf->Free();
        free(controlMem);
        pool->Free();
        queue2->Free();
    }
#endif // LWN_DEBUG_DISABLE_WINDOW_PRESENT_INFLIGHT_MEMORY_TEST

    // Test render target / multisample state matching
    {
        // Bind a valid program.
        lwnTest::GLSLCHelper glslcHelper(device, 0x100000, g_glslcLibraryHelper, g_glslcHelperCache);
        VertexShader vs(450);
        FragmentShader fs(450);

        vs <<
            "void main()\n"
            "{\n"
            "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "}\n";

        fs <<
            "out vec4 color;\n"
            "void main()\n"
            "{\n"
            "  color = vec4(1.0, 1.0, 1.0, 1.0);\n"
            "}\n";

        Program *pgm = device->CreateProgram();
        if (!glslcHelper.CompileAndSetShaders(pgm, vs, fs)) {
            ADD_RESULT(false);
            return;
        }
        queueCB.BindProgram(pgm, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);

        // Create some resources for testing.
        MemoryPoolAllocator pool(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);
        TextureBuilder textureBuilder;
        textureBuilder.SetDevice(device)
                      .SetDefaults()
                      .SetFormat(Format::RGBA8)
                      .SetSize2D(64, 64)
                      .SetTarget(TextureTarget::TARGET_2D);
        Texture *ssColor = pool.allocTexture(&textureBuilder);
        textureBuilder.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE)
                      .SetSamples(4);
        Texture *msColor = pool.allocTexture(&textureBuilder);
        MemoryPoolAllocator depthPool(device, NULL, 0x200000, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
        textureBuilder.SetFormat(Format::DEPTH24)
                      .SetFlags(TextureFlags::COMPRESSIBLE);
        Texture *msDepth4 = depthPool.allocTexture(&textureBuilder);
        textureBuilder.SetSamples(8);
        Texture *msDepth8 = depthPool.allocTexture(&textureBuilder);
        MultisampleState multisampleState;
        multisampleState.SetDefaults();

        // Simple MSAA
        multisampleState.SetSamples(4);
        queueCB.SetRenderTargets(1, &msColor, NULL, msDepth4, NULL);
        queueCB.BindMultisampleState(&multisampleState);
        queueCB.DrawArrays(DrawPrimitive::POINTS, 0, 0);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(0);

        // Mismatched MSAA state / color target
        multisampleState.SetSamples(2);
        queueCB.BindMultisampleState(&multisampleState);
        queueCB.SetRenderTargets(1, &msColor, NULL, NULL, NULL);
        queueCB.DrawArrays(DrawPrimitive::POINTS, 0, 0);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(1);

        // Single-sample texture, MSAA disabled, non-zero sample count (Bug 1880824 regression test)
        multisampleState.SetMultisampleEnable(LWN_FALSE);
        queueCB.BindMultisampleState(&multisampleState);
        queueCB.SetRenderTargets(1, &ssColor, NULL, NULL, NULL);
        queueCB.DrawArrays(DrawPrimitive::POINTS, 0, 0);
        queueCB.submit();
        EXPECT_DEBUG_CALLBACKS(1);

        if (g_lwnDeviceCaps.supportsTargetIndependentRasterization) {
            // Mismatched MSAA state / depth target
            // Note: mismatched MSAA state / color target tested previously outside
            // this block.
            multisampleState.SetMultisampleEnable(LWN_TRUE);
            multisampleState.SetSamples(4);
            queueCB.BindMultisampleState(&multisampleState);
            queueCB.SetRenderTargets(1, &msColor, NULL, msDepth8, NULL);
            queueCB.DrawArrays(DrawPrimitive::POINTS, 0, 0);
            queueCB.submit();
            EXPECT_DEBUG_CALLBACKS(1);

            // TIR
            multisampleState.SetSamples(4);
            multisampleState.SetRasterSamples(8);
            queueCB.BindMultisampleState(&multisampleState);
            queueCB.DrawArrays(DrawPrimitive::POINTS, 0, 0);
            queueCB.submit();
            EXPECT_DEBUG_CALLBACKS(0);

            // TIR for ClearDepthStencil
            queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0, 0);
            queueCB.submit();
            EXPECT_DEBUG_CALLBACKS(0);

            // TIR mismatch
            multisampleState.SetRasterSamples(16);
            queueCB.BindMultisampleState(&multisampleState);
            queueCB.DrawArrays(DrawPrimitive::POINTS, 0, 0);
            queueCB.submit();
            EXPECT_DEBUG_CALLBACKS(1);

#if 0
            // !!! DISABLE due to DVS sanity problems (bug 2996034).
            // TIR mismatch for ClearDepthStencil
            queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0, 0);
            queueCB.submit();
            EXPECT_DEBUG_CALLBACKS(1);
#endif
        }

        multisampleState.SetDefaults();

        // Regression test for Bug 1968085 (binding MSAA state in called/copied command buffer)
        CommandBuffer *cb = device->CreateCommandBuffer();
        g_lwnCommandMem.populateCommandBuffer(cb, CommandBufferMemoryManager::Coherent);
        cb->BeginRecording();
        cb->BindMultisampleState(&multisampleState);
        CommandHandle ch = cb->EndRecording();
        queueCB.CallCommands(1, &ch);
        EXPECT_DEBUG_CALLBACKS(0);

        queueCB.BindMultisampleState(&multisampleState);
        queueCB.SetRenderTargets(0, NULL, NULL, NULL, NULL);
        queueCB.submit();
        queue->Finish();
        EXPECT_DEBUG_CALLBACKS(0);
    }

    // Uninstall all callbacks so they don't affect future tests or test runs.
    device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
    device->InstallDebugCallback(Callback2, NULL, LWN_FALSE);
}

void LWNDebugAPITest::doGraphics()
{
    m_textureBuilder.SetTarget(TextureTarget::TARGET_2D);
    m_textureBuilder.SetFormat(Format::RGBA8);
    m_textureBuilder.SetSize2D(640, 480);

    m_callbackResults.reset();
    m_results.clear();

    // Run tests associated with installing a debug callback on an invalid device.
    TestDebugCallbacksInstall();

    // Create and activate a temporary device and queue for this test.  We
    // need this because the debug layer tracks the validity of all objects
    // created through the API and uses indices instead of "real pointers".
    // Calling into the debug layer with real pointers makes things really
    // confused.
    DeviceState *testDevice = new DeviceState(m_deviceFlags);

    if (!testDevice || !testDevice->isValid()) {
        delete testDevice;
        DeviceState::SetDefaultActive();
        LWNFailTest();
        return;
    }

    Device *device = testDevice->getDevice();
    QueueCommandBuffer &queueCB = testDevice->getQueueCB();
    Queue *queue = testDevice->getQueue();
    testDevice->SetActive();

    // Regression test for Bug 2028523. Ensure that calling lwnBootstrapLoader() does not
    // reset the global debug level. Subsequent tests will fail if this happens.
    lwnBootstrapLoader("lwnDeviceGetProcAddress");

    // Run tests testing errors that should be caught by the auto-generated layer.
    TestDebugCallbacksGenerated(device, queue, queueCB);

    // Run tests testing errors that are manually implemented.
    TestDebugCallbacksHandwritten(device, queue, queueCB);

    // Run tests testing errors that are detected at drawtime.
    TestDebugCallbacksDrawTimeValidations(device, queue, queueCB);

    // Run tests testing errors that make multi-threaded LWN calls. This is a stability-only test.
    TestDebugCallbacksThreaded(device, queue, queueCB);
    // Run tests testing errors related to queue memory allocations.
    if (lwogCheckLWNAPIVersion(52, 11)) {
        TestDebugCallbacksQueueMemory(device);
    }
    if (lwogCheckLWNAPIVersion(52, 21)) {
        TestDebugCallbacksGpfifo(device);
    }

    if (lwogCheckLWNAPIVersion(53, 9)) {
        TestDebugCallbacksGLASMErrors(device);
    }

    if (lwogCheckLWNAPIVersion(53, 100) && m_debugLevel == 0) {
        TestDebugCallbacksMemoryPoolErrors(device);
    }

    if (m_debugLevel > 0) {
        TestDebugCallbacksMemoryPoolOverlaps(device);
        TestDebugCallbacksQueueOverlaps(device);
        TestDebugCallbacksQueueMempoolOverlaps(device);
    }

    // Manually clean up API resources that we created.
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
        if (!m_results[i].result) {
            DEBUG_PRINT(("test %d failed on line %d!\n", (int) i, m_results[i].linenum));
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

void LWNDebugAPITest::TestDebugCallbacksShaderSubroutines(Device *device, Queue *queue, QueueCommandBuffer& queueCB)
{
    lwnTest::GLSLCHelper glslcHelper(device, 0x100000, g_glslcLibraryHelper, g_glslcHelperCache);

    // Vertex shader which uses subroutines.
    VertexShader vsSubroutines(450);
    // Vertex shader which doesn't use subroutines.
    VertexShader vsNoSubroutines(450);
    // Fragment shader which uses subroutines.
    FragmentShader fsSubroutines(450);
    // Fragment shader which uses subroutines.
    FragmentShader fsNoSubroutines(450);
    // Geometry shader which doesn't contain subroutines.
    GeometryShader gs(450);

    vsSubroutines <<
        "out IO { flat int var; };\n"
        "in int test;\n"
        "subroutine int Sub1(int);\n"
        "layout (index = 0) subroutine (Sub1) int Sub1Func(int ilwal) { return ilwal * 9000; }\n"
        "layout (index = 1) subroutine (Sub1) int Sub2Func(int ilwal) { return ilwal * 8000; }\n"
        "layout (location = 0) subroutine uniform Sub1 Sub1Uniform;\n"
        ""
        "void main () \n"
        "{\n"
        "  gl_Position = vec4(0, 0, 0, 0);\n"
        "  var = Sub1Uniform(test);\n"
        "}\n";

    vsNoSubroutines <<
        "out IO { flat int var; };\n"
        "void main () \n"
        "{\n"
        "  gl_Position = vec4(0, 0, 0, 0);\n"
        "}\n";

    gs <<
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "in IO { flat int var; } vi[];\n"
        "out IO { flat int var; };\n"
        "void main() {\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = gl_in[i].gl_Position;\n"
        "    var = vi[i].var;\n"
        "    EmitVertex();\n"
        "  }\n"
        "}\n";

    fsSubroutines <<
        "in IO { flat int var; };\n"
        "out vec4 ocolor;\n"
        "subroutine int Sub1(int);\n"
        "subroutine int Sub2(int);\n"
        "layout (index = 0) subroutine (Sub1) int Sub1Func(int ilwal) { return ilwal * 7000; }\n"
        "layout (index = 1) subroutine (Sub2) int Sub2Func(int ilwal) { return ilwal * 8000; }\n"
        "layout (index = 2) subroutine (Sub1, Sub2) int Sub12Func(int ilwal) { return ilwal * 9000; }\n"
        "layout (location = 0) subroutine uniform Sub1 Sub1Uniform;\n"
        "layout (location = 1) subroutine uniform Sub2 Sub2Uniform;\n"
        ""
        "void main()\n"
        "{\n"
        "  ocolor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "  ocolor.x = float(Sub1Uniform(var));\n"
        "  ocolor.y = float(Sub2Uniform(var));\n"
        "}\n";

    fsNoSubroutines <<
        "in IO { flat int var; };\n"
        "out vec4 ocolor;\n"
        ""
        "void main()\n"
        "{\n"
        "  ocolor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    // Use subroutines in both stages for the rest.
    if (!glslcHelper.CompileAndSetShaders(pgm, vsSubroutines, gs, fsSubroutines)) {
        DEBUG_PRINT(("We need a valid shader here!\n"));
        return;
    }

    int linkageSizes[2] = { 0 };
    LWNsubroutineLinkageMapPtr linkages[2];
    LWNsubroutineLinkageMapPtr tmpLinkagePtr = NULL;

    // Make a copy of the linkage pointers since the GLSLChelper only stores the last compiled linakge maps, and
    // we will be compiling different sets of programs after this, but we don't want to ilwalidate the correct
    // linkage maps.
    tmpLinkagePtr = glslcHelper.GetSubroutineLinkageMap(ShaderStage::VERTEX, 0, &linkageSizes[0]);
    linkages[0] = (LWNsubroutineLinkageMapPtr *)__LWOG_MALLOC(linkageSizes[0]);
    memcpy(linkages[0], tmpLinkagePtr, linkageSizes[0]);

    tmpLinkagePtr = glslcHelper.GetSubroutineLinkageMap(ShaderStage::FRAGMENT, 0, &linkageSizes[1]);
    linkages[1] = (LWNsubroutineLinkageMapPtr *)__LWOG_MALLOC(linkageSizes[1]);
    memcpy(linkages[1], tmpLinkagePtr, linkageSizes[1]);

    // Initially set the uniforms to valid subroutines (uniform 0 gets set to Sub1, uniform 1 gets set to Sub2).
    int fsUniformValues[2] = { 0, 1};

    // Test setting subroutines on program without a linkage map.
    queueCB.SetProgramSubroutines(pgm, ShaderStage::FRAGMENT, 0, 2, fsUniformValues);
    EXPECT_DEBUG_CALLBACKS(1);

    // Test ProgramSetSubroutineLinkage with working parameters first. No debug messages should come up.
    pgm->SetSubroutineLinkage(2, linkages);
    EXPECT_DEBUG_CALLBACKS(0);

    // Test ProgramSetSubroutineLinkage with a count parameter == 0
    pgm->SetSubroutineLinkage(0, linkages);
    EXPECT_DEBUG_CALLBACKS(1);

    // Test setting subroutines for a stage that doesn't use subroutines.
    queueCB.SetProgramSubroutines(pgm, ShaderStage::GEOMETRY, 0, 2, fsUniformValues);
    EXPECT_DEBUG_CALLBACKS(1);

    // Test to ensure no debug callbacks for setting a subroutine of type Sub1 or Sub2 to a subroutine function
    // of type (Sub1, Sub2), since that is allowed.
    fsUniformValues[0] = 2;
    fsUniformValues[1] = 2;
    queueCB.SetProgramSubroutines(pgm, ShaderStage::FRAGMENT, 0, 2, fsUniformValues);
    EXPECT_DEBUG_CALLBACKS(0);

    // Test setting uniforms to invalid types.  Uniform 0 is not compatible with index 1 (Sub2),
    // and uniform 1 is not compatible with index 0 (Sub1).
    fsUniformValues[0] = 1;
    fsUniformValues[1] = 0;
    queueCB.SetProgramSubroutines(pgm, ShaderStage::FRAGMENT, 0, 2, fsUniformValues);
    EXPECT_DEBUG_CALLBACKS(1);

    // Try to set the linkages on an uninitialized program.
    Program *uninitPgm = device->CreateProgram();
    uninitPgm->SetSubroutineLinkage(2, linkages);
    EXPECT_DEBUG_CALLBACKS(1);

    // Test setting subroutines for an uninitialized program
    queueCB.SetProgramSubroutines(uninitPgm, ShaderStage::FRAGMENT, 0, 1, fsUniformValues);
    EXPECT_DEBUG_CALLBACKS(1);

    // Try to call with an invalid <count> parameter.
    pgm->SetSubroutineLinkage(1, linkages);
    EXPECT_DEBUG_CALLBACKS(1);

    // Try to call with one or more linkage maps as NULL.
    LWNsubroutineLinkageMapPtr ilwalidLinkages[2] = {
        linkages[0],
        NULL
    };
    pgm->SetSubroutineLinkage(2, ilwalidLinkages);
    EXPECT_DEBUG_CALLBACKS(1);

    // Try to use the same linkage map for multiple stages.
    ilwalidLinkages[0] = linkages[0];
    ilwalidLinkages[1] = linkages[0];
    pgm->SetSubroutineLinkage(2, ilwalidLinkages);
    EXPECT_DEBUG_CALLBACKS(1);

    // Test setting a subroutine for the wrong stage (a stage which doesn't use subroutines).
    Program *noVsProg = device->CreateProgram();
    if (!glslcHelper.CompileAndSetShaders(noVsProg, vsNoSubroutines, gs, fsSubroutines)) {
        DEBUG_PRINT(("We need a valid shader here!\n"));
        return;
    }
    noVsProg->SetSubroutineLinkage(1, &linkages[0]);
    EXPECT_DEBUG_CALLBACKS(1);

    // Test ProgramSetSubroutineLinkage on a program which doesn't use subroutines.
    Program *noSubProg = device->CreateProgram();
    if (!glslcHelper.CompileAndSetShaders(noSubProg, vsNoSubroutines, fsNoSubroutines)) {
        DEBUG_PRINT(("We need a valid shader here!\n"));
        return;
    }
    noSubProg->SetSubroutineLinkage(2, linkages);
    EXPECT_DEBUG_CALLBACKS(1);

    // We are about to corrupt linkages, so don't try to use them for tests other than testing corruption
    // past this point.
    //

    // Corrupt the linkage map by over-writting the magic value (which always comes as the first 32 bytes).
    ((int *)(linkages[0]))[0] = 666;
    pgm->SetSubroutineLinkage(2, linkages);
    EXPECT_DEBUG_CALLBACKS(1);

    // Submit our data so subroutines don't mess with future tests.
    queueCB.submit();

    // Free our programs and linkage maps.
    pgm->Free();
    noVsProg->Free();
    noSubProg->Free();
    __LWOG_FREE(linkages[0]);
    __LWOG_FREE(linkages[1]);
}

// Test whether we get debug layer error messages for GLASM parsing errors in the LWN driver.
// We do this by trying to use the LW_fragment_shader_interlock extension.
// GLSLC will generate GM20y code, with "OPTION LW_pixel_interlock_ordered" in the GLASM string.
// During recompilation in lwnProgramSetShaders() on Windows reference implementations running
// on GM1 or Kepler-level cards, an ASM error will be thrown, which should trigger the debug-layer
// error below.  On GM2XX cards or CheetAh X1, lwnProgramSetShaders() should not fail.
void LWNDebugAPITest::TestDebugCallbacksGLASMErrors(Device *device)
{
    // Install the debug callback.
    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);

    lwnTest::GLSLCHelper glslcHelper(device, 0x10000, g_glslcLibraryHelper, g_glslcHelperCache);
    VertexShader vs(450);
    FragmentShader fs(450);

    vs <<
        "void main()\n"
        "{\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n";

    fs.addExtension(lwShaderExtension::LW_fragment_shader_interlock);
    fs <<
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "  beginIlwocationInterlockLW();"
        "  color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "  endIlwocationInterlockLW();"
        "}\n";

    Program *pgm = device->CreateProgram();

    if (!glslcHelper.CompileShaders(vs, fs)) {
        ADD_RESULT(false);
        DEBUG_PRINT(("TestDebugCallbacksGLASMErrors failed to compile.\n"));
    } else {
        EXPECT_DEBUG_CALLBACKS(0);

        // Running on a device without FS interlock support (Kepler, Maxwell first generation) should
        // fail in SetShaders() and the debug layer callback should be thrown indicate GLASM errors.
        // No debug layer error should be reported on devices which support FS interlock.
        glslcHelper.SetShaders(pgm, glslcHelper.GetGlslcOutput());
        EXPECT_DEBUG_CALLBACKS(g_lwnDeviceCaps.supportsFragmentShaderInterlock ? 0 : 1);

        pgm->Free();
    }

    // Uninstall all callbacks so they don't affect future tests or test runs.
    device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
    device->InstallDebugCallback(Callback2, NULL, LWN_FALSE);
}

class ExpectMessage {
    Device     *m_device;
    const char *m_warning;
    bool        m_triggered;
public:
    ExpectMessage(Device *device, const char* warning)
        : m_device(device), m_warning(warning), m_triggered(false)
    {
        m_device->InstallDebugCallback(LWNDebugAPITest::CallbackExpectMessage, this, LWN_TRUE);
    }
    ~ExpectMessage()
    {
        m_device->InstallDebugCallback(LWNDebugAPITest::CallbackExpectMessage, NULL, LWN_FALSE);
    }
    bool isTriggered() const       { return m_triggered; }
    void update(LWNstring message) { m_triggered |= !strcmp(m_warning, message); }
};

void LWNAPIENTRY LWNDebugAPITest::CallbackExpectMessage(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                        DebugCallbackSeverity::Enum severity, LWNstring message, void *userParam)
{
    (static_cast<ExpectMessage*>(userParam))->update(message);
}

template<class T>
class Finalizer {
    T m_cargo;
public:
    ~Finalizer<T>() { m_cargo.Finalize(); }
    T* operator->() { return &m_cargo; }
    T* operator&()  { return &m_cargo; }
};

void LWNDebugAPITest::TestDebugCallbacksMemoryPoolErrors(Device *device)
{
#if defined(LW_TEGRA)
    static const char* mempoolInitCompressibleWarning = {
        "lwnMemoryPoolInitialize:  "
        "Compression resources exhausted, falling back to non-compressible memory."
    };

    static const char* mapCompressibleWarning = {
        "lwnMemoryPoolMapVirtual:  "
        "Compression resources exhausted, falling back to non-compressible memory."
    };

    static const uint32_t page = LWN_DEVICE_INFO_CONSTANT_NX_MEMORY_POOL_PAGE_SIZE;
    bool success = false;

    auto dtor = [](MemoryPool* mp) { mp->Finalize(); delete mp; };
    using PoolPtr = std::unique_ptr<MemoryPool,decltype(dtor)>;

    std::list<PoolPtr> pools;

    MemoryPoolBuilder mpb;
    mpb.SetDevice(device)
       .SetDefaults()
       .SetFlags(MemoryPoolFlags::GPU_UNCACHED |
                 MemoryPoolFlags::CPU_CACHED   |
                 MemoryPoolFlags::COMPRESSIBLE);

    // We use 1 compits per pool.
    // We have a maximum of 1216 compbits available.
    {
        static const uint32_t MAX_POOLS_COUNT = 1216;
        void *pool_buffer[MAX_POOLS_COUNT] = {0};

        ExpectMessage expectMessage(device, mempoolInitCompressibleWarning);

        for (uint32_t i = 0; i < MAX_POOLS_COUNT; ++i) {
            PoolPtr p(nullptr, dtor);
            p.reset(new MemoryPool);

            pool_buffer[i] = aligned_alloc(LWN_MEMORY_POOL_STORAGE_ALIGNMENT, page);
            if (pool_buffer[i] == nullptr) {
                DEBUG_PRINT(("we did not have enough heap after %d allocation.\n", i));
                break;
            }

            mpb.SetStorage(pool_buffer[i], page);
            p->Initialize(&mpb);

            if (expectMessage.isTriggered()) {
                success = true;
                break;
            }

            pools.push_back(std::move(p));
        }

        for (uint32_t i = 0; i < MAX_POOLS_COUNT; ++i) {
            free(pool_buffer[i]);
            pool_buffer[i] = nullptr;
        }
    }

    if (success) {
#define FAIL_IF(cond) if(cond) { success = false; goto append_result; }
        ExpectMessage expectMessage(device, mapCompressibleWarning);

        Finalizer<MemoryPool>  physicalMemoryPoolWrapper;
        Finalizer<MemoryPool>   virtualMemoryPoolWrapper;

        Finalizer<Texture>     tex;
        TextureBuilder         tb;

        uint8_t backingStore2[page] __attribute__(( aligned(LWN_MEMORY_POOL_STORAGE_ALIGNMENT) ));

        mpb.SetFlags(MemoryPoolFlags::CPU_CACHED    |
                     MemoryPoolFlags::PHYSICAL      |
                     MemoryPoolFlags::GPU_NO_ACCESS |
                     MemoryPoolFlags::COMPRESSIBLE)
           .SetStorage(backingStore2, page);
        FAIL_IF(!physicalMemoryPoolWrapper->Initialize(&mpb))

        mpb.SetFlags(MemoryPoolFlags::GPU_CACHED    |
                     MemoryPoolFlags::VIRTUAL       |
                     MemoryPoolFlags::CPU_NO_ACCESS |
                     MemoryPoolFlags::COMPRESSIBLE);
        mpb.SetStorage(nullptr, page);
        FAIL_IF(!virtualMemoryPoolWrapper->Initialize(&mpb))

        tb.SetDevice(device)
          .SetDefaults()
          .SetTarget(TextureTarget::TARGET_2D)
          .SetFormat(Format::RGBA8)
          .SetSize2D(8, 8)
          .SetFlags(TextureFlags::COMPRESSIBLE)
          .SetStorage(&virtualMemoryPoolWrapper, 0);
        FAIL_IF(!tex->Initialize(&tb))

        MappingRequest mr;
        mr.physicalPool = &physicalMemoryPoolWrapper;
        mr.virtualOffset  = 0;
        mr.physicalOffset = 0;
        mr.size = page;
        mr.storageClass = tex->GetStorageClass();
        FAIL_IF(!virtualMemoryPoolWrapper->MapVirtual(1, &mr))

        success &= expectMessage.isTriggered();
#undef FAIL_IF
    }

append_result:
    ADD_RESULT(success);
#else
    (void)device;
#endif
}

// Test overlaps between Queue and MemPool.
#if defined(LW_TEGRA)

enum OverlapCheckType {queue = 0, mempool = 1};
typedef struct __MemBlockInfo {
    OverlapCheckType type;
    char *startp;
    union {
        size_t memPoolSize;
        struct {
            size_t command;
            size_t control;
            size_t compute;
        } queueSize;
    } size;
} MemBlockInfo;

static int OverlapMixedStorageCheck(Device *device,
                                    MemBlockInfo &first, MemBlockInfo &second,
                                    const bool expected_result,
                                    const char *first_layout_msg, const char *second_layout_msg)
{
    // expected_result should be true if it should fail at second pool CreateMemoryPool.
    // false if it should succeed at second pool CreateMemoryPool.
    DEBUG_PRINT(("\n"
                 "Start Testing Memory Storage Overlaps:\n"
                 "chunk overlap pattern:\n"));
    DEBUG_PRINT(("   first: %s\n", first_layout_msg));
    DEBUG_PRINT(("  second: %s\n", second_layout_msg));

    // default size of the memory chunk to apply.
    // No big reason for 1Mbyte, just that the sum of queue command/control/conpute will not exceed.
    const int default_memory_size = 1 * 1024 * 1024;

    int result = 0;

    MemoryPool *p1 = nullptr;
    MemoryPool *p2 = nullptr;
    Queue *q1 = nullptr;
    Queue *q2 = nullptr;

    LWNboolean res1;
    LWNboolean res2;

    switch (first.type) {
    default:
        DEBUG_PRINT(("1st type unknown\n"));
        result = 1;
        goto OverlapMixedStorageCheck_cleanup;

    case OverlapCheckType::queue:
        {
            QueueBuilder qb1;
            qb1.SetDefaults();
            qb1.SetDevice(device).SetQueueMemory(first.startp, default_memory_size);
            qb1.SetCommandMemorySize(first.size.queueSize.command)
               .SetControlMemorySize(first.size.queueSize.control)
               .SetComputeMemorySize(first.size.queueSize.compute);

            q1 = new Queue();
            if (!q1) {
                DEBUG_PRINT(("failed to allocate 1st queue.\n"));
                result = 1;
                goto OverlapMixedStorageCheck_cleanup;
            }

            res1 = q1->Initialize(&qb1);
            if (res1 == false) {
                DEBUG_PRINT(("Failed in preparing the 1st queue.\n"));
                delete q1;
                q1 = nullptr;
                result = 1;
                goto OverlapMixedStorageCheck_cleanup;
            }
        }
        break;
    case OverlapCheckType::mempool:
        {
            p1 = device->CreateMemoryPoolWithFlags(first.startp, first.size.memPoolSize, (MemoryPoolFlags::GPU_NO_ACCESS | MemoryPoolFlags::PHYSICAL | MemoryPoolFlags::CPU_CACHED));
            if (!p1) {
                DEBUG_PRINT(("Failed in preparing the 1st pool.\n"));
                result = 1;
                goto OverlapMixedStorageCheck_cleanup;
            }
        }
        break;
    }

    switch (second.type) {
    default:
        DEBUG_PRINT(("2nd type unknown\n"));
        result = 1;
        goto OverlapMixedStorageCheck_cleanup;
    case OverlapCheckType::queue:
        {
            QueueBuilder qb2;
            qb2.SetDefaults();
            qb2.SetDevice(device).SetQueueMemory(second.startp, default_memory_size);
            qb2.SetCommandMemorySize(second.size.queueSize.command)
               .SetControlMemorySize(second.size.queueSize.control)
               .SetComputeMemorySize(second.size.queueSize.compute);

            q2 = new Queue();
            if (!q2) {
                DEBUG_PRINT(("failed to allocate 2nd queue.\n"));
                result = 1;
                goto OverlapMixedStorageCheck_cleanup;
            }

            // Silence the expected error message,
            // for error may occur on next CreateMemoryPoolWithFlags().
            ReloadLWNEntryPoints(reinterpret_cast<LWNdevice *>(device), true);

            res2 = q2->Initialize(&qb2);

            // Bring back the error message feature.
            ReloadLWNEntryPoints(reinterpret_cast<LWNdevice *>(device), false);
        }
        break;
    case OverlapCheckType::mempool:
        {
            // Silence the expected error message,
            // for error may occur on next CreateMemoryPoolWithFlags().
            ReloadLWNEntryPoints(reinterpret_cast<LWNdevice *>(device), true);

            p2 = device->CreateMemoryPoolWithFlags(second.startp, second.size.memPoolSize, (MemoryPoolFlags::GPU_NO_ACCESS | MemoryPoolFlags::PHYSICAL | MemoryPoolFlags::CPU_CACHED));

            // Bring back the error message feature.
            ReloadLWNEntryPoints(reinterpret_cast<LWNdevice *>(device), false);

            if (!p2) {
                res2 = LWN_FALSE;
            } else {
                res2 = LWN_TRUE;
            }
        }
        break;
    }

    // return 0 if the result was as expected. 1 otherwise
    // expected_result == true means res2 should show LWN_FALSE/fail in initialize.
    if (res2 == expected_result) {
        DEBUG_PRINT(("Testing Memory Storage Overlaps Failed.\n"
                     "chunk overlap pattern:\n"));
        DEBUG_PRINT(("   first: %s\n", first_layout_msg));
        DEBUG_PRINT(("  second: %s\n", second_layout_msg));

        result = 1;
    }

OverlapMixedStorageCheck_cleanup:
    if (q2) {
        if (res2) {
            q2->Finish();
            q2->Flush();
            q2->Finalize();
        }
        delete q2;
        q2 = nullptr;
    }
    if (p2) {
        p2->Free();
        p2 = nullptr;
    }
    if (q1) {
        q1->Finish();
        q1->Flush();
        q1->Finalize();
        delete q1;
        q1 = nullptr;
    }
    if (p1) {
        p1->Free();
        p1 = nullptr;
    }
    return result;
}

static int OverlapTestSwaps(Device *device, MemBlockInfo &fi, MemBlockInfo &si,
                            const bool expected_result,
                            const char *fmsg, const char *smsg)
{
    int result = 0;
    result += OverlapMixedStorageCheck(device, fi, si, expected_result, fmsg, smsg);
    result += OverlapMixedStorageCheck(device, si, fi, expected_result, smsg, fmsg);
    return result;
}
#endif // defined(LW_TEGRA)

#define FAIL_IF(cond) if(cond) { ADD_RESULT(false); return; }

void LWNDebugAPITest::TestDebugCallbacksMemoryPoolOverlaps(Device *device)
{
#if !defined(LW_TEGRA)
    (void)device;
#else // !defined(LW_TEGRA)
    ReloadLWNEntryPoints(reinterpret_cast<LWNdevice *>(device), false);

    int result = 0;
    const size_t preparation_size = 4 * 1024 * 1024;   // 4Mbytes for preparation.
    const size_t second_chunk_offset = LWN_DEVICE_INFO_MEMORY_POOL_PAGE_SIZE * 2;
                                                           // slidings for second chunk. should be 16K
    int memory_pool_page_size;

    device->GetInteger(lwn::DeviceInfo::MEMORY_POOL_PAGE_SIZE , &memory_pool_page_size);

    auto chunk_body = std::make_unique<char[]>(preparation_size);
    DEBUG_PRINT(("Base Chunk to use : [ 0x%p .. 0x%p )\n", chunk_body.get(), chunk_body.get() + preparation_size));
    FAIL_IF(chunk_body == nullptr);

    char *first_pointer;
    char *second_pointer;
    first_pointer = AlignPointer(chunk_body.get(), memory_pool_page_size);
    second_pointer = AlignPointer(first_pointer + second_chunk_offset, memory_pool_page_size);

    size_t first_size;
    size_t second_size;
    first_size  = AlignSize(preparation_size - (first_pointer  - chunk_body.get()), memory_pool_page_size);
    while (first_size > preparation_size - (first_pointer - chunk_body.get())) {
        first_size -= memory_pool_page_size;
    }
    second_size = AlignSize(preparation_size - (second_pointer - chunk_body.get()), memory_pool_page_size);
    while (second_size > preparation_size - (second_pointer - chunk_body.get())) {
        second_size -= memory_pool_page_size;
    }

    DEBUG_PRINT(("All Prepared. Let's start creating memory pool\n"));

    DEBUG_PRINT(("Current Debug Level = %d\n", m_debugLevel ));
    DEBUG_PRINT(("first chunk range   = [ 0x%p : 0x%p ) : size = %zd\n", first_pointer, first_pointer + first_size, first_size));
    DEBUG_PRINT(("second chunk range  = [ 0x%p : 0x%p ) : size = %zd\n", second_pointer, second_pointer + second_size, second_size));

    // Lwrrently, (first_pointer, first_size), and (second_pointer, second_size) have the following relationship.
    //
    // first:        x=========================================o
    // second:                x================================o
    //            x means it's included, o means it's not included.


    MemBlockInfo firstInfo;
    MemBlockInfo secondInfo;

    firstInfo.type = OverlapCheckType::mempool;
    secondInfo.type = OverlapCheckType::mempool;

    // Pattern 1
    firstInfo.startp = first_pointer;
    firstInfo.size.memPoolSize = first_size;
    secondInfo.startp = second_pointer;
    secondInfo.size.memPoolSize = second_size;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "x===============================o",
                               "        x=======================o");

    // Pattern 2
    firstInfo.startp = first_pointer;
    firstInfo.size.memPoolSize = first_size - memory_pool_page_size;
    secondInfo.startp = second_pointer;
    secondInfo.size.memPoolSize = second_size;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "x=======================o        ",
                               "        x=======================o");

    // Pattern 3
    firstInfo.startp = first_pointer;
    firstInfo.size.memPoolSize = first_size;
    secondInfo.startp = second_pointer;
    secondInfo.size.memPoolSize = second_size - memory_pool_page_size;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "x===============================o",
                               "        x===============o        ");

    // Pattern 4
    firstInfo.startp = first_pointer;
    firstInfo.size.memPoolSize = first_size;
    secondInfo.startp = first_pointer;
    secondInfo.size.memPoolSize = first_size - memory_pool_page_size;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "x===============================o",
                               "x=======================o        ");

    // Pattern 5
    firstInfo.startp = first_pointer;
    firstInfo.size.memPoolSize = second_pointer - first_pointer;
    secondInfo.startp = second_pointer;
    secondInfo.size.memPoolSize = second_size;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, false,
                               "x=======o                        ",
                               "        x=======================o");

    // result should be 0 if all succeeded.
    DEBUG_PRINT(("TestDebugCallbacksMemoryPoolOverlaps() resulted in error numbers: %d\n", result));
    FAIL_IF(result);

    ADD_RESULT(true);
#endif // !defined(LW_TEGRA)
}

#define SetQueueDefault(info) \
    do { \
        info.type = OverlapCheckType::queue; \
        info.startp = nullptr; \
        info.size.queueSize.command = commandMemorySize; \
        info.size.queueSize.control = controlMemorySize; \
        info.size.queueSize.compute = computeMemorySize; \
    } while (0)

#define SetMempoolDefault(info) \
    do { \
        info.type = OverlapCheckType::mempool; \
        info.startp = nullptr; \
        info.size.memPoolSize = memPoolMemorySize; \
    } while (0)

void LWNDebugAPITest::TestDebugCallbacksQueueOverlaps(Device *device)
{
#if !defined(LW_TEGRA)
    (void)device;
#else // !defined(LW_TEGRA)
    int result = 0;
    const size_t preparation_size = LWN_MEMORY_POOL_STORAGE_GRANULARITY * 1024;   // should be 4Mbytes for preparation.

    auto chunk_body = std::make_unique<char[]>(preparation_size);
    DEBUG_PRINT(("Base Chunk to use : [ 0x%p .. 0x%p )\n", chunk_body.get(), chunk_body.get() + preparation_size));
    FAIL_IF(chunk_body == nullptr);

    char *top_pointer;
    top_pointer = AlignPointer(chunk_body.get(), LWN_MEMORY_POOL_STORAGE_ALIGNMENT);

    // In current implementation, if application passes memory pool to queue initialization,
    // Command, Control, and Compute memory will be allocated in the following order:
    // <--- command ---><--- control ---><--- compute --->
    // to ease the test, let's assign 128kbytes each for command and control.
    // And (128 + 104)kbytes for compute.
    // Also, for total size, let's use 1Mbytes.

    const size_t commandMemorySize = 128 * 1024;
    const size_t controlMemorySize = 128 * 1024;
    const size_t computeMemorySize = 128 * 1024;
    const size_t computeDefault    = 104 * 1024; // Compute requires extra 104kbytes regardless of the size you apply.

    MemBlockInfo firstInfo;
    MemBlockInfo secondInfo;

    // No overlap
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer + (controlMemorySize + commandMemorySize + computeMemorySize + computeDefault);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, false,
                               "<ctr><com><cmp>",
                               "               <ctr><com><cmp>");

    // All overlap
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "<ctr><com><cmp>");

    // overlap half the size of Control
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer + (controlMemorySize / 2);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "  <ctr><com><cmp>");

    // slide the size of control
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer + controlMemorySize;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "     <ctr><com><cmp>");

    // slide the size of control + half the size of command
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer + controlMemorySize + (commandMemorySize / 2);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "       <ctr><com><cmp>");

    // slide the size of control + size of command
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer + controlMemorySize + commandMemorySize;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "          <ctr><com><cmp>");

    // slide the size of control + command + half the size of compute
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer + controlMemorySize + commandMemorySize + (computeMemorySize + computeDefault)/2;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "            <ctr><com><cmp>");

    // result should be 0 if all succeeded.
    DEBUG_PRINT(("TestDebugCallbacksQueueOverlaps() resulted in error numbers: %d\n", result));
    FAIL_IF(result);

    ADD_RESULT(true);
#endif // !defined(LW_TEGRA)
}


void LWNDebugAPITest::TestDebugCallbacksQueueMempoolOverlaps(Device *device)
{
#if !defined(LW_TEGRA)
    (void)device;
#else // !defined(LW_TEGRA)
    int result = 0;
    const size_t preparation_size = LWN_MEMORY_POOL_STORAGE_GRANULARITY * 1024;   // should be 4Mbytes for preparation.

    auto chunk_body = std::make_unique<char[]>(preparation_size);
    DEBUG_PRINT(("Base Chunk to use : [ 0x%p .. 0x%p )\n", chunk_body.get(), chunk_body.get() + preparation_size));
    FAIL_IF(chunk_body == nullptr);

    char *top_pointer;
    top_pointer = AlignPointer(chunk_body.get(), LWN_MEMORY_POOL_STORAGE_ALIGNMENT);

    // In current implementation, if application passes memory pool to queue initialization,
    // Command, Control, and Compute memory will be allocated in the following order:
    // <--- command ---><--- control ---><--- compute --->
    // to ease the test, let's assign 128kbytes each for command and control.
    // And (128 + 104)kbytes for compute (compute require extra 104kbytes).
    // Also, for total size, let's use 1Mbytes.

    const size_t commandMemorySize = 128 * 1024;
    const size_t controlMemorySize = 128 * 1024;
    const size_t computeMemorySize = 128 * 1024;
    const size_t computeDefault    = 104 * 1024; // Compute requires extra 104kbytes regardless of the size you apply.

    int memory_pool_page_size;
    device->GetInteger(lwn::DeviceInfo::MEMORY_POOL_PAGE_SIZE , &memory_pool_page_size);
    const size_t memPoolMemorySize = AlignSize(commandMemorySize + controlMemorySize + computeMemorySize + computeDefault + memory_pool_page_size -1, memory_pool_page_size);

    MemBlockInfo firstInfo;
    MemBlockInfo secondInfo;

    // No overlap
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetMempoolDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + memPoolMemorySize, memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, false,
                               "<ctr><com><cmp>",
                               "               <-memory pool->");

    SetMempoolDefault(firstInfo);
    firstInfo.startp = AlignPointer(top_pointer + memPoolMemorySize, memory_pool_page_size);
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, false,
                               "<-memory pool->",
                               "               <ctr><com><cmp>");

    // All overlap
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetMempoolDefault(secondInfo);
    secondInfo.startp = top_pointer;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "<-memory pool->");

    SetMempoolDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = top_pointer;
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<-memory pool->",
                               "<ctr><com><cmp>");

    // overlap half the size of Control
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetMempoolDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + (controlMemorySize / 2), memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "  <-memory pool->");

    SetMempoolDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + (controlMemorySize / 2), memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<-memory pool->",
                               "  <ctr><com><cmp>");

    // slide the size of control
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetMempoolDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize, memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "     <-memory pool->");

    SetMempoolDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize, memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<-memory pool->",
                               "     <ctr><com><cmp>");

    // slide the size of control + half the size of command
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetMempoolDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize + (commandMemorySize / 2), memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "       <-memory pool->");

    SetMempoolDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize + (commandMemorySize / 2), memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<-memory pool->",
                               "       <ctr><com><cmp>");

    // slide the size of control + the size of command
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetMempoolDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize + commandMemorySize, memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "          <-memory pool->");

    SetMempoolDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize + commandMemorySize, memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<-memory pool->",
                               "          <ctr><com><cmp>");

    // slide the size of control + command + half the size of compute
    SetQueueDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetMempoolDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize + commandMemorySize + (computeMemorySize + computeDefault)/2, memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<ctr><com><cmp>",
                               "            <-memory pool->");

    SetMempoolDefault(firstInfo);
    firstInfo.startp = top_pointer;
    SetQueueDefault(secondInfo);
    secondInfo.startp = AlignPointer(top_pointer + controlMemorySize + commandMemorySize + (computeMemorySize + computeDefault)/2, memory_pool_page_size);
    result += OverlapTestSwaps(device, firstInfo, secondInfo, true,
                               "<-memory pool->",
                               "            <ctr><com><cmp>");

    // result should be 0 if all succeeded.
    DEBUG_PRINT(("TestDebugCallbacksQueueMempoolOverlaps() resulted in error numbers: %d\n", result));
    FAIL_IF(result);

    ADD_RESULT(true);
#endif // !defined(LW_TEGRA)
}

#undef FAIL_IF

void LWNDebugAPITest::TestDebugCallbacksBindProgram(Device *device, Queue *queue, QueueCommandBuffer& queueCB)
{
    lwnTest::GLSLCHelper glslcHelper(device, 0x100000, g_glslcLibraryHelper, g_glslcHelperCache);
    VertexShader vs(450);
    FragmentShader fs(450);

    vs <<
        "void main()\n"
        "{\n"
        "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n";

    fs <<
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "  color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    if (!glslcHelper.CompileAndSetShaders(pgm, vs, fs)) {
        ADD_RESULT(false);
        return;
    }

    EXPECT_DEBUG_CALLBACKS(0);

    // Allow binds with <stages> values holding the stages present in <pgm> or
    // with extra stages to unbind.
    queueCB.BindProgram(pgm, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
    EXPECT_DEBUG_CALLBACKS(0);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    EXPECT_DEBUG_CALLBACKS(0);

    // Don't allow binds with one or more stages present in <pgm> missing in
    // the mask.
    queueCB.BindProgram(pgm, ShaderStageBits::VERTEX);
    EXPECT_DEBUG_CALLBACKS(1);
    queueCB.BindProgram(pgm, ShaderStageBits::FRAGMENT);
    EXPECT_DEBUG_CALLBACKS(1);
    queueCB.BindProgram(pgm, ShaderStageBits(0));
    EXPECT_DEBUG_CALLBACKS(1);

    // Complain about a NULL bind that does nothing (mask of zero).
    queueCB.BindProgram(NULL, ShaderStageBits(0));
    EXPECT_DEBUG_CALLBACKS(1);

    pgm->Free();
}

struct LWNthreadParams {
    int ID;
    Device* device;
    LWOGthread *thread;
};

static void lwnDebugAPIWorkerThread(void* arg)
{
    LWNthreadParams *threadParams = (LWNthreadParams *) arg;
    assert(arg);
    Device* device = threadParams->device;

    // Make some random LWN calls.
    int texDescSize = 0, sampDescSize = 0;
    for (int i = 0; i < 10; i++) {
        device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE, &texDescSize);
        device->GetInteger(DeviceInfo::SAMPLER_DESCRIPTOR_SIZE, &sampDescSize);
    }

    DEBUG_PRINT(("TestDebugCallbacksThreaded id %d\n", ((LWNthreadParams*) arg)->ID));
}

void LWNDebugAPITest::TestDebugCallbacksThreaded(Device* device, Queue* queue, QueueCommandBuffer& queueCB)
{
    return;

    // Create a number of LWN threads.
    const static int NUM_THREADS = 32;
    LWNthreadParams threadParams[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        threadParams[i].ID = i;
        threadParams[i].device = device;
#if defined(LW_HOS)
        // Select a core for the worker thread, spreading the threads across
        // the available processor cores.
        uint64_t availableCores = lwogThreadGetAvailableCoreMask();
        int idealCore = lwogThreadSelectCoreRoundRobin(i, availableCores);
        threadParams[i].thread = lwogThreadCreateOnCore(lwnDebugAPIWorkerThread, &threadParams[i], 0x20000, idealCore);
#else
        threadParams[i].thread = lwogThreadCreate(lwnDebugAPIWorkerThread, &threadParams[i], 0x20000);
#endif
    }

    // Join threads and clean up.
    for (int i = 0; i < NUM_THREADS; i++) {
        lwogThreadWait(threadParams[i].thread);
    }
}

void LWNDebugAPITest::TestDebugCallbackQueueMemoryTestTmpQueue(const QueueBuilder *builder, int expectedCallbacks)
{
    Queue queue;
    LWNboolean initialized = queue.Initialize(builder);
    if (initialized) {
        queue.Finalize();
    }
    EXPECT_DEBUG_CALLBACKS(expectedCallbacks);
}

void LWNDebugAPITest::TestDebugCallbacksQueueMemory(Device* device)
{
    // Install the debug callback.
    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);

    int commandGranularity, commandMinSize, commandDefaultSize,
        computeGranularity, computeMinSize, computeDefaultSize;
    int minFlushThreshold = 0;

    device->GetInteger(DeviceInfo::QUEUE_COMMAND_MEMORY_GRANULARITY, &commandGranularity);
    device->GetInteger(DeviceInfo::QUEUE_COMMAND_MEMORY_MIN_SIZE, &commandMinSize);
    device->GetInteger(DeviceInfo::QUEUE_COMMAND_MEMORY_DEFAULT_SIZE, &commandDefaultSize);
    device->GetInteger(DeviceInfo::QUEUE_COMPUTE_MEMORY_GRANULARITY, &computeGranularity);
    device->GetInteger(DeviceInfo::QUEUE_COMPUTE_MEMORY_MIN_SIZE, &computeMinSize);
    device->GetInteger(DeviceInfo::QUEUE_COMPUTE_MEMORY_DEFAULT_SIZE, &computeDefaultSize);
    if (lwogCheckLWNAPIVersion(52, 12)) {
        device->GetInteger(DeviceInfo::QUEUE_COMMAND_MEMORY_MIN_FLUSH_THRESHOLD, &minFlushThreshold);
    }
    EXPECT_DEBUG_CALLBACKS(0);

    QueueBuilder qb;
    qb.SetDevice(device);

    // Test creating queues with different command memory sizes.
    qb.SetDefaults();
    qb.SetCommandMemorySize(commandMinSize - commandGranularity);       // too small
    EXPECT_DEBUG_CALLBACKS(1);
    qb.SetCommandMemorySize(commandMinSize + commandGranularity / 2);   // misaligned
    EXPECT_DEBUG_CALLBACKS(1);
    qb.SetCommandMemorySize(commandMinSize + commandGranularity);       // OK
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 0);

    // Test creating queues with different compute memory sizes.
    qb.SetDefaults();
    qb.SetComputeMemorySize(computeMinSize - computeGranularity);       // too small
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 1);                   // size not checked until queue creation
    qb.SetComputeMemorySize(computeMinSize + computeGranularity / 2);   // misaligned
    EXPECT_DEBUG_CALLBACKS(1);
    qb.SetComputeMemorySize(computeMinSize + computeGranularity);       // OK
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 0);
    qb.SetComputeMemorySize(0);                                         // zero size is OK
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 0);

    // Try creating queues with different flush thresholds.
    if (lwogCheckLWNAPIVersion(52, 12)) {
        qb.SetDefaults();
        qb.SetCommandFlushThreshold(minFlushThreshold / 2);                    // too small
        EXPECT_DEBUG_CALLBACKS(1);
        qb.SetCommandFlushThreshold(commandDefaultSize + minFlushThreshold);   // too big
        TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 1);
        qb.SetCommandFlushThreshold(commandDefaultSize);                       // legal but not recommended
        TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 0);
    }

    // Try creating queues with user-provided memory, using the default sizes.
    qb.SetDefaults();
    size_t totalSize = qb.GetQueueMemorySize();
    char *queueMem = (char *) AlignedStorageAlloc(totalSize + LWN_MEMORY_POOL_STORAGE_GRANULARITY,
                                                  LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    qb.SetQueueMemory(queueMem + LWN_MEMORY_POOL_STORAGE_ALIGNMENT / 2, totalSize);     // misaligned pointer
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 1);
    qb.SetQueueMemory(queueMem, totalSize + LWN_MEMORY_POOL_STORAGE_GRANULARITY / 2);   // misaligned size
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 1);
    qb.SetQueueMemory(queueMem, totalSize - LWN_MEMORY_POOL_STORAGE_GRANULARITY);       // too small
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 1);
    qb.SetQueueMemory(queueMem, totalSize);                                             // just right
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 0);

    // Try creating a queue with no compute or interlock memory, and then
    // making sure we throw draw-time validation errors if programs attempt to
    // use these features.
    if (m_debugLevel >= DEBUG_FEATURE_DRAW_TIME_VALIDATION) {
        Queue tmpQueue;
        qb.SetDefaults();
        qb.SetComputeMemorySize(0);
        qb.SetFlags(QueueFlags::NO_FRAGMENT_INTERLOCK);
        size_t tmpQueueSize = qb.GetQueueMemorySize();
        qb.SetQueueMemory(queueMem, tmpQueueSize);
        if (tmpQueue.Initialize(&qb)) {

            // Set up a dummy compute shader and attempt to perform a compute
            // dispatch on the queue without compute memory.
            ComputeShader cs(430);
            cs.setCSGroupSize(1, 1, 1);
            cs << "void main() {}\n";
            Program *computePgm = device->CreateProgram();
            if (g_glslcHelper->CompileAndSetShaders(computePgm, cs)) {
                CommandBuffer cb;
                cb.Initialize(device);
                g_lwnCommandMem.populateCommandBuffer(&cb, CommandBufferMemoryManager::Coherent);
                cb.BeginRecording();
                cb.BindProgram(computePgm, ShaderStageBits::COMPUTE);
                cb.DispatchCompute(1, 1, 1);
                CommandHandle handle = cb.EndRecording();
                EXPECT_DEBUG_CALLBACKS(0);
                tmpQueue.SubmitCommands(1, &handle);
                EXPECT_DEBUG_CALLBACKS(1);
                tmpQueue.Finish();
                cb.Finalize();
                ADD_RESULT(true);
            } else {
                ADD_RESULT(false);
            }
            computePgm->Free();

            // If supported, set up a dummy vertex/fragment shader program using
            // fragment shader interlock and attempt to render on the queue
            // without interlock memory.
            if (g_lwnDeviceCaps.supportsFragmentShaderInterlock) {

                VertexShader vs(430);
                FragmentShader fs(430);
                vs << "void main() {}\n";
                fs <<
                    "#extension GL_LW_fragment_shader_interlock: require\n" 
                    "layout(pixel_interlock_ordered) in;\n"
                    "out vec4 color;\n"
                    "void main() {\n"
                    "  beginIlwocationInterlockLW();\n"
                    "  color = vec4(1.0);\n"
                    "  endIlwocationInterlockLW();\n"
                    "}\n";
                Program *interlockPgm = device->CreateProgram();
                if (g_glslcHelper->CompileAndSetShaders(interlockPgm, vs, fs)) {

                    CommandBuffer cb;
                    cb.Initialize(device);
                    g_lwnCommandMem.populateCommandBuffer(&cb, CommandBufferMemoryManager::Coherent);
                    cb.BeginRecording();

                    // Interlock shaders seem to require scratch memory due to the
                    // use of call/return in our code transformation (bug
                    // 1806472).  Since we are using a temporary queue, the
                    // default scratch memory bound in the lwnUtil::DeviceState
                    // doesn't apply, and is not sufficient to run without
                    // throttling debug layer messages.  Allocate and bind our own
                    // scratch memory just for this test.
                    const GLSLCoutput *glslcOutput = g_glslcHelper->GetGlslcOutput();
                    size_t scratchSize = g_glslcHelper->GetScratchMemoryRecommended(device, glslcOutput);

                    MemoryPool *scratchPool = NULL;

                    if (scratchSize > 0) {
                        scratchPool = device->CreateMemoryPool(NULL, scratchSize, MemoryPoolType::GPU_ONLY);
                        g_glslcHelper->SetShaderScratchMemory(scratchPool, 0, scratchSize, &cb);
                        assert(scratchPool);
                    }

                    cb.BindProgram(interlockPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
                    cb.DrawArrays(DrawPrimitive::POINTS, 0, 1);
                    CommandHandle handle = cb.EndRecording();
                    EXPECT_DEBUG_CALLBACKS(0);
                    tmpQueue.SubmitCommands(1, &handle);
                    EXPECT_DEBUG_CALLBACKS(1);
                    tmpQueue.Finish();

                    cb.Finalize();

                    if (scratchPool) {
                        scratchPool->Free();
                    }

                    ADD_RESULT(true);
                } else {
                    ADD_RESULT(false);
                }
                interlockPgm->Free();
            }
            
            tmpQueue.Finalize();
        } else {
            ADD_RESULT(false);
        }
    }
    EXPECT_DEBUG_CALLBACKS(0);

    AlignedStorageFree(queueMem);

    // Uninstall all callbacks so they don't affect future tests or test runs.
    device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
    device->InstallDebugCallback(Callback2, NULL, LWN_FALSE);
}

void LWNDebugAPITest::TestDebugCallbacksGpfifo(Device* device)
{
    // Install the debug callback.
    device->InstallDebugCallback(Callback1, &m_callbackResults.callback1Count, LWN_TRUE);

    int queueControlGranularity, queueControlMinSize, queueControlDefaultSize;

    device->GetInteger(DeviceInfo::QUEUE_CONTROL_MEMORY_GRANULARITY, &queueControlGranularity);
    device->GetInteger(DeviceInfo::QUEUE_CONTROL_MEMORY_MIN_SIZE, &queueControlMinSize);
    device->GetInteger(DeviceInfo::QUEUE_CONTROL_MEMORY_DEFAULT_SIZE, &queueControlDefaultSize);
    EXPECT_DEBUG_CALLBACKS(0);

    QueueBuilder qb;
    qb.SetDevice(device);

    // Test creating queues with different sizes.
    qb.SetDefaults();
    qb.SetControlMemorySize(queueControlMinSize - queueControlGranularity);       // too small
    EXPECT_DEBUG_CALLBACKS(1);
    qb.SetControlMemorySize(queueControlMinSize + queueControlGranularity / 2);   // misaligned
    EXPECT_DEBUG_CALLBACKS(1);
    qb.SetControlMemorySize(queueControlMinSize + queueControlGranularity);       // OK
    TestDebugCallbackQueueMemoryTestTmpQueue(&qb, 0);

#if defined(LW_TEGRA)
    // Generate a command buffer with enough work to keep the GPU busy while we're
    // flooding more submits.
    CommandBuffer cb;
    cb.Initialize(device);
    g_lwnCommandMem.populateCommandBuffer(&cb, CommandBufferMemoryManager::Coherent);
    const lwn::Texture *rtTex = g_lwnWindowFramebuffer.getAcquiredTexture();
    cb.BeginRecording();
    cb.SetRenderTargets(1, &rtTex, NULL, NULL, NULL);
    LWNfloat clearColor[] = {0.1f, 0.2f, 0.3f, 1.0f};
    for (int i = 0; i < 1024; ++i) {
        cb.SetScissor(0,0,1,1);
        cb.ClearColor(0, clearColor, ClearColorMask::RGBA);
    }
    CommandHandle handle = cb.EndRecording();

    // Set small gather memory size and check that we get stall messages.
    Queue tmpQueue;
    qb.SetDefaults();
    qb.SetControlMemorySize(queueControlMinSize);
    if (tmpQueue.Initialize(&qb)) {
        for (int i = 0; i < 1024; ++i) {
            tmpQueue.SubmitCommands(1, &handle);
        }
        tmpQueue.Finish();
        EXPECT_DEBUG_CALLBACKS(m_debugLevel == 0 ? 1 : 0, 0, CallbackResults::Comparison::GreaterOrEqual);
        tmpQueue.Finalize();
    }

    cb.Finalize();
    EXPECT_DEBUG_CALLBACKS(0);
#endif

    // Uninstall all callbacks so they don't affect future tests or test runs.
    device->InstallDebugCallback(Callback1, NULL, LWN_FALSE);
}

void LWNDebugAPITest::exitGraphics()
{
    lwnDefaultExitGraphics();
    m_results.clear();
}

OGTEST_CppTest(LWNDebugAPITest, lwn_debug_level0, (0));
OGTEST_CppTest(LWNDebugAPITest, lwn_debug_level1, (1));
OGTEST_CppTest(LWNDebugAPITest, lwn_debug_level2, (2));
OGTEST_CppTest(LWNDebugAPITest, lwn_debug_level3, (3));
OGTEST_CppTest(LWNDebugAPITest, lwn_debug_level4, (4));

