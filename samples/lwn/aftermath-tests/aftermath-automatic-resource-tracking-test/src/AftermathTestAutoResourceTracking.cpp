/*
 * Copyright (c) 2020-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <array>
#include <deque>

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtilsLWN.h>

#include "shaders/TestAutoResourceTracking_glslcoutput.h"
#include <lwnTool/lwnTool_GlslcInterface.h>

#include <AftermathApi.h>

namespace AftermathTest {

class AutoResourceTrackingTest
{
public:
    bool Initialize();

    bool Test(const Options& options);

private:

    // Testing
    bool TestPreInitializeDevice();
    bool TestWithAutoDeferredFinalize(const Options& options);

    // LWN object handling
    bool InitializeDevice(bool enableDebugLayer);
    bool FinalizeDevice();
    bool InitializeWindow();
    bool InitializeQueue();
    bool InitializeSync();
    bool InitializeShaders();
    bool InitializeVertexBuffer();
    bool InitializeFixedResources();
    bool InitializeFixedObjects();
    bool InitializePerFrameResources();
    bool FinalizePerFrameResources();
    bool FinalizeAllObjects();
    bool InitializeResources(
        LWN::UniqueUint8PtrWithLwstomDeleter& resourcePoolMemory,
        LWN::MemoryPoolHolder& resourceMemoryPool,
        int numTextures,
        std::deque<LWN::TextureHolder>& textures,
        int numBuffers,
        std::deque<LWN::BufferHolder>& buffers,
        int numSamplerPools,
        std::deque<LWN::SamplerPoolHolder>& samplerPools,
        int numTexturePools,
        std::deque<LWN::TexturePoolHolder>& texturePools);

    // Rendering
    bool RenderFrame();

    // Command buffer memory handling
    static void LWNAPIENTRY CommandBufferMemoryCallback(
        CommandBuffer* cmdBuf,
        CommandBufferMemoryEvent::Enum event,
        size_t minSize,
        void* callbackData);
    bool AddCommandMemory(CommandBuffer* cmdBuf, size_t minSize);
    bool AddControlMemory(CommandBuffer* cmdBuf, size_t minSize);

    bool VerifyTrackedResources(int frameNumber);

    static const int s_numFixedTextures = 100;
    static const int s_numFixedBuffers = 200;
    static const int s_numFixedSamplerPools = 300;
    static const int s_numFixedTexturePools = 500;

    static const int s_numPerFrameTextures = 10;
    static const int s_numPerFrameBuffers = 20;
    static const int s_numPerFrameSamplerPools = 30;
    static const int s_numPerFrameTexturePools = 50;

    static const int s_aftermathDefaultDeferredFinalizationAge = 5;

    // LWN objects
    LWN::WindowHolder m_window;
    LWN::QueueHolder m_queue;
    LWN::SyncHolder m_cmdSync;
    LWN::SyncHolder m_windowSync;
    LWN::ShaderBufferHolder m_vertexShader;
    LWN::ShaderBufferHolder m_fragmentShader;
    LWN::ProgramHolder m_graphicsProgram;
    LWN::UniqueUint8PtrWithLwstomDeleter m_vertexBufferData;
    LWN::MemoryPoolHolder m_vertexBufferPool;
    LWN::BufferHolder m_vertexBuffer;
    LWN::UniqueUint8PtrWithLwstomDeleter m_resourcePoolMemory;
    LWN::MemoryPoolHolder m_resourceMemoryPool;
    std::deque<LWN::TextureHolder> m_textures;
    std::deque<LWN::BufferHolder> m_buffers;
    std::deque<LWN::SamplerPoolHolder> m_samplerPools;
    std::deque<LWN::TexturePoolHolder> m_texturePools;
    LWN::UniqueUint8PtrWithLwstomDeleter m_perFrameResourcePoolMemory;
    LWN::MemoryPoolHolder m_perFrameResourceMemoryPool;
    std::deque<LWN::TextureHolder> m_perFrameTextures;
    std::deque<LWN::BufferHolder> m_perFrameBuffers;
    std::deque<LWN::SamplerPoolHolder> m_perFrameSamplerPools;
    std::deque<LWN::TexturePoolHolder> m_perFrameTexturePools;

    // Tracking of command buffer command memory
    struct CommandMemoryPool
    {
        LWN::UniqueUint8PtrWithLwstomDeleter buffer;
        LWN::MemoryPoolHolder pool;
    };
    std::deque<CommandMemoryPool> m_cmdMemoryPools;

    // Tracking of command buffer control memory
    struct ControlMemoryBuffer
    {
        LWN::UniqueUint8PtrWithLwstomDeleter buffer;
        size_t size;
    };
    std::deque<ControlMemoryBuffer> m_controlMemoryBuffers;
};

bool AutoResourceTrackingTest::TestPreInitializeDevice()
{
    // Expect that GPU crash dumps are enabled in DevMenu!
    bool enabled = false;
    TEST_EQ(aftermathIsEnabled(&enabled), AftermathApiError_None);
    TEST_EQ(enabled, true);

    // Set and verify feature level
    TEST_EQ(aftermathSetFeatureLevel(AftermathApiFeatureLevel_Enhanced), AftermathApiError_None);
    AftermathApiFeatureLevel featureLevel = AftermathApiFeatureLevel_Default;
    TEST_EQ(aftermathGetFeatureLevel(&featureLevel), AftermathApiError_None);
    TEST_EQ(featureLevel, AftermathApiFeatureLevel_Enhanced);

    // Set and verify configuration flags
    TEST_EQ(aftermathSetConfigurationFlags(AftermathApiConfigurationFlags_AutomaticResourceTracking), AftermathApiError_None);
    int flags = 0xdeadbeef;
    TEST_EQ(aftermathGetConfigurationFlags(&flags), AftermathApiError_None);
    TEST_EQ(flags, AftermathApiConfigurationFlags_AutomaticResourceTracking);

    return true;
}

bool AutoResourceTrackingTest::InitializeDevice(bool enableDebugLayer)
{
    DeviceFlagBits flags = 0;
    if (enableDebugLayer) {
        flags |= DeviceFlagBits::DEBUG_ENABLE_LEVEL_4;
    }

    // Initialize the device
    LWN::SetupLWNDevice(flags);

    return true;
}

bool AutoResourceTrackingTest::FinalizeDevice()
{
    LWN::ShutdownLWNDevice();

    return true;
}

void LWNAPIENTRY AutoResourceTrackingTest::CommandBufferMemoryCallback(
    CommandBuffer* cmdBuf,
    CommandBufferMemoryEvent::Enum event,
    size_t minSize,
    void* callbackData)
{
    AutoResourceTrackingTest* test = reinterpret_cast<AutoResourceTrackingTest*>(callbackData);
    switch (event) {
    case CommandBufferMemoryEvent::OUT_OF_COMMAND_MEMORY:
        (void)test->AddCommandMemory(cmdBuf, minSize);
        break;
    case CommandBufferMemoryEvent::OUT_OF_CONTROL_MEMORY:
        (void)test->AddControlMemory(cmdBuf, minSize);
        break;
    default:
        break;
    }
}

bool AutoResourceTrackingTest::AddCommandMemory(CommandBuffer* cmdBuf, size_t minSize)
{
    const size_t cmdMemorySize =
        std::max<size_t>(
            Utils::AlignUp(minSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY),
            LWN_MEMORY_POOL_STORAGE_GRANULARITY * 8);

    // Add a new CommandMemoryPool entry
    m_cmdMemoryPools.emplace_back();
    CommandMemoryPool& cmdMemoryPool = m_cmdMemoryPools.back();

    // Allocate memory
    cmdMemoryPool.buffer = LWN::AlignedAllocPodType<uint8_t>(cmdMemorySize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    TEST_NE(cmdMemoryPool.buffer, nullptr);

    // Create pool
    MemoryPoolBuilder poolBuilder;
    poolBuilder.SetDevice(g_device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(cmdMemoryPool.buffer.get(), cmdMemorySize);
    TEST(cmdMemoryPool.pool.Initialize(&poolBuilder));

    cmdBuf->AddCommandMemory(cmdMemoryPool.pool, 0, cmdMemoryPool.pool->GetSize());

    return true;
}

bool AutoResourceTrackingTest::AddControlMemory(CommandBuffer* cmdBuf, size_t minSize)
{
    const size_t controlMemorySize =
        std::max<size_t>(
            Utils::AlignUp(minSize, LWN_DEVICE_INFO_CONSTANT_NX_COMMAND_BUFFER_CONTROL_ALIGNMENT),
            LWN_DEVICE_INFO_CONSTANT_NX_COMMAND_BUFFER_CONTROL_ALIGNMENT * 1024);

    // Allocate a new buffer
    m_controlMemoryBuffers.emplace_back();
    ControlMemoryBuffer& controlMemoryBuffer = m_controlMemoryBuffers.back();

    // Allocate memory
    controlMemoryBuffer.buffer = LWN::AlignedAllocPodType<uint8_t>(controlMemorySize, LWN_DEVICE_INFO_CONSTANT_NX_COMMAND_BUFFER_CONTROL_ALIGNMENT);
    TEST_NE(controlMemoryBuffer.buffer, nullptr);
    controlMemoryBuffer.size = controlMemorySize;

    cmdBuf->AddControlMemory(controlMemoryBuffer.buffer.get(), controlMemoryBuffer.size);

    return true;
}

bool AutoResourceTrackingTest::InitializeWindow()
{
    m_window.Initialize(g_device);

    return true;
}

bool AutoResourceTrackingTest::InitializeQueue()
{
    m_queue.Initialize(g_device);

    return true;
}

bool AutoResourceTrackingTest::InitializeSync()
{
    m_cmdSync.Initialize((Device*)g_device);
    m_windowSync.Initialize((Device*)g_device);

    return true;
}

bool AutoResourceTrackingTest::InitializeShaders()
{
    struct glslcShaderStage
    {
        const char* data;
        size_t dataSize;
        const char* control;
        size_t controlSize;
    };
    std::array<glslcShaderStage, GLSLC_NUM_SHADER_STAGES> shaders;

    const GLSLCoutput* compiledProgram = reinterpret_cast<const GLSLCoutput*>(testAutoResourceTracking_glslcoutput);
    for (unsigned int i = 0; i < compiledProgram->numSections; ++i)
    {
        if (compiledProgram->headers[i].genericHeader.common.type == GLSLC_SECTION_TYPE_GPU_CODE) {
            const GLSLCgpuCodeHeader& gpuCodeHeader = compiledProgram->headers[i].gpuCodeHeader;

            const ShaderStage::Enum stage = ShaderStage::Enum(gpuCodeHeader.stage);

            const char* base = (char*)compiledProgram + gpuCodeHeader.common.dataOffset;
            shaders[stage].data = base + gpuCodeHeader.dataOffset;
            shaders[stage].dataSize = gpuCodeHeader.dataSize;
            shaders[stage].control = base + gpuCodeHeader.controlOffset;
            shaders[stage].controlSize = gpuCodeHeader.controlSize;
        }
    }

    m_vertexShader.Initialize(g_device, shaders[ShaderStage::VERTEX].data, shaders[ShaderStage::VERTEX].dataSize);
    m_fragmentShader.Initialize(g_device, shaders[ShaderStage::FRAGMENT].data, shaders[ShaderStage::FRAGMENT].dataSize);

    std::array<ShaderData, 2> lwnShaderData;
    lwnShaderData[0].data = m_vertexShader->GetAddress();
    lwnShaderData[0].data = m_vertexShader->GetAddress();
    lwnShaderData[0].control = shaders[ShaderStage::VERTEX].control;
    lwnShaderData[1].data = m_fragmentShader->GetAddress();
    lwnShaderData[1].control = shaders[ShaderStage::FRAGMENT].control;

    // Initialize the program and provide it with the compiled shader
    m_graphicsProgram.Initialize((Device*)g_device);
    m_graphicsProgram->SetShaders(lwnShaderData.size(), lwnShaderData.data());

    return true;
}

bool AutoResourceTrackingTest::InitializeVertexBuffer()
{
    // Allocate memory
    const size_t vertexBufferSize = 3 * 4 * sizeof(float);
    const size_t vertexBufferPoolSize = Utils::AlignUp(vertexBufferSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);
    m_vertexBufferData = LWN::AlignedAllocPodType<uint8_t>(vertexBufferSize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    TEST_NE(m_vertexBufferData, nullptr);

    // Create pool
    MemoryPoolBuilder poolBuilder;
    poolBuilder.SetDevice(g_device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED);
    poolBuilder.SetStorage(m_vertexBufferData.get(), vertexBufferPoolSize);
    TEST(m_vertexBufferPool.Initialize(&poolBuilder));

    // Create buffer
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(g_device)
        .SetDefaults()
        .SetStorage(m_vertexBufferPool, 0, vertexBufferSize);
    TEST(m_vertexBuffer.Initialize(&bufferBuilder));

    float* vertexData = (float*)m_vertexBufferPool->Map();
    vertexData[0] = -0.5f;
    vertexData[1] = -0.5f;
    vertexData[2] = -0.25f;
    vertexData[3] = 1.0f;
    vertexData[4] = 0.0f;
    vertexData[5] = 0.5f;
    vertexData[6] = -0.25f;
    vertexData[7] = 1.0f;
    vertexData[8] = 0.5f;
    vertexData[9] = -0.5f;
    vertexData[10] = -0.25f;
    vertexData[11] = 1.0f;
    m_vertexBufferPool->FlushMappedRange(0, vertexBufferSize);

    return true;
}

bool AutoResourceTrackingTest::InitializeFixedResources()
{
    return InitializeResources(
        m_resourcePoolMemory,
        m_resourceMemoryPool,
        s_numFixedTextures,
        m_textures,
        s_numFixedBuffers,
        m_buffers,
        s_numFixedSamplerPools,
        m_samplerPools,
        s_numFixedTexturePools,
        m_texturePools);
}

bool AutoResourceTrackingTest::InitializeResources(
    LWN::UniqueUint8PtrWithLwstomDeleter& resourcePoolMemory,
    LWN::MemoryPoolHolder& resourceMemoryPool,
    int numTextures,
    std::deque<LWN::TextureHolder>& textures,
    int numBuffers,
    std::deque<LWN::BufferHolder>& buffers,
    int numSamplerPools,
    std::deque<LWN::SamplerPoolHolder>& samplerPools,
    int numTexturePools,
    std::deque<LWN::TexturePoolHolder>& texturePools)
{
    TextureBuilder textureBuilder;
    textureBuilder
        .SetDevice(g_device)
        .SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::RGBA8)
        .SetSize2D(16, 32);

    size_t BufferSize = 256;
    BufferBuilder bufferBuilder;
    bufferBuilder
        .SetDevice(g_device)
        .SetDefaults();

    // Determine required pool size
    size_t poolMemorySize = 0;
    for (int i = 0; i < numTextures; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, textureBuilder.GetStorageAlignment());
        poolMemorySize += textureBuilder.GetStorageSize();
    }
    for (int i = 0; i < numBuffers; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_DEVICE_INFO_CONSTANT_NX_UNIFORM_BUFFER_ALIGNMENT);
        poolMemorySize += BufferSize;
    }
    const size_t numSamplerDescriptors = numTextures + LWN_DEVICE_INFO_CONSTANT_NX_RESERVED_SAMPLER_DESCRIPTORS;
    for (int i = 0; i < numSamplerPools; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_DEVICE_INFO_CONSTANT_NX_SAMPLER_DESCRIPTOR_SIZE);
        poolMemorySize += numSamplerDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_SAMPLER_DESCRIPTOR_SIZE;
    }
    const size_t numTextureDescriptors = numTextures + LWN_DEVICE_INFO_CONSTANT_NX_RESERVED_TEXTURE_DESCRIPTORS;
    for (int i = 0; i < numTexturePools; ++i) {
        poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_DEVICE_INFO_CONSTANT_NX_TEXTURE_DESCRIPTOR_SIZE);
        poolMemorySize += numTextureDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_TEXTURE_DESCRIPTOR_SIZE;
    }

    poolMemorySize = Utils::AlignUp(poolMemorySize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);

    // Allocate memory
    resourcePoolMemory = LWN::AlignedAllocPodType<uint8_t>(poolMemorySize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    TEST_NE(resourcePoolMemory, nullptr);

    // Create pool
    MemoryPoolBuilder poolBuilder;
    poolBuilder.SetDevice(g_device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(resourcePoolMemory.get(), poolMemorySize);
    TEST(resourceMemoryPool.Initialize(&poolBuilder));

    size_t memoryPoolOffset = 0;

    // Create textures
    for (int i = 0; i < numTextures; ++i) {
        memoryPoolOffset = Utils::AlignUp(memoryPoolOffset, textureBuilder.GetStorageAlignment());
        textureBuilder.SetStorage(resourceMemoryPool, memoryPoolOffset);
        memoryPoolOffset += textureBuilder.GetStorageSize();

        textures.emplace_back();
        LWN::TextureHolder& texture = textures.back();
        TEST(texture.Initialize(&textureBuilder));
    }

    // Create buffers
    for (int i = 0; i < numBuffers; ++i) {
        memoryPoolOffset = Utils::AlignUp(memoryPoolOffset, LWN_DEVICE_INFO_CONSTANT_NX_UNIFORM_BUFFER_ALIGNMENT);
        bufferBuilder.SetStorage(resourceMemoryPool, memoryPoolOffset, BufferSize);
        memoryPoolOffset += BufferSize;

        buffers.emplace_back();
        LWN::BufferHolder& buffer = buffers.back();
        TEST(buffer.Initialize(&bufferBuilder));
    }

    // Create sampler pools
    for (int i = 0; i < numSamplerPools; ++i) {
        samplerPools.emplace_back();
        LWN::SamplerPoolHolder& samplerPool = samplerPools.back();
        TEST(samplerPool.Initialize((MemoryPool*)resourceMemoryPool, memoryPoolOffset, numSamplerDescriptors));
        memoryPoolOffset += numSamplerDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_SAMPLER_DESCRIPTOR_SIZE;
    }

    // Create texture pools
    for (int i = 0; i < numTexturePools; ++i) {
        texturePools.emplace_back();
        LWN::TexturePoolHolder& texturePool = texturePools.back();
        TEST(texturePool.Initialize((MemoryPool*)resourceMemoryPool, memoryPoolOffset, numTextureDescriptors));
        memoryPoolOffset += numTextureDescriptors * LWN_DEVICE_INFO_CONSTANT_NX_TEXTURE_DESCRIPTOR_SIZE;
    }

    return true;
}

bool AutoResourceTrackingTest::FinalizeAllObjects()
{
    m_perFrameTextures.clear();
    m_perFrameBuffers.clear();
    m_perFrameSamplerPools.clear();
    m_perFrameTexturePools.clear();
    m_perFrameResourceMemoryPool.Finalize();
    m_perFrameResourcePoolMemory = nullptr;
    m_textures.clear();
    m_buffers.clear();
    m_samplerPools.clear();
    m_texturePools.clear();
    m_resourceMemoryPool.Finalize();
    m_resourcePoolMemory = nullptr;
    m_vertexBuffer.Finalize();
    m_vertexBufferPool.Finalize();
    m_vertexBufferData = nullptr;
    m_graphicsProgram.Finalize();
    m_fragmentShader.Finalize();
    m_vertexShader.Finalize();
    m_windowSync.Finalize();
    m_cmdSync.Finalize();
    m_queue.Finalize();
    m_window.Finalize();

    return true;
}

bool AutoResourceTrackingTest::InitializeFixedObjects()
{
    TEST(InitializeWindow());
    TEST(InitializeQueue());
    TEST(InitializeSync());
    TEST(InitializeShaders());
    TEST(InitializeVertexBuffer());

    TEST(InitializeFixedResources());

    return true;
}

bool AutoResourceTrackingTest::InitializePerFrameResources()
{
    return InitializeResources(
        m_perFrameResourcePoolMemory,
        m_perFrameResourceMemoryPool,
        s_numPerFrameTextures,
        m_perFrameTextures,
        s_numPerFrameBuffers,
        m_perFrameBuffers,
        s_numPerFrameSamplerPools,
        m_perFrameSamplerPools,
        s_numPerFrameTexturePools,
        m_perFrameTexturePools);

    return true;
}

bool AutoResourceTrackingTest::FinalizePerFrameResources()
{
    m_perFrameTextures.clear();
    m_perFrameBuffers.clear();
    m_perFrameSamplerPools.clear();
    m_perFrameTexturePools.clear();
    m_perFrameResourceMemoryPool.Finalize();
    m_perFrameResourcePoolMemory = nullptr;

    m_cmdMemoryPools.clear();
    m_controlMemoryBuffers.clear();

    return true;
}

bool AutoResourceTrackingTest::RenderFrame()
{
    // Get current RT index
    int rtIndex = 0;
    WindowAcquireTextureResult acquireTextureResult = m_window->AcquireTexture(m_windowSync, &rtIndex);
    TEST_EQ(acquireTextureResult, WindowAcquireTextureResult::SUCCESS);

    // Setup rendering commands
    LWN::CommandBufferHolder cmdBuf;
    TEST(cmdBuf.Initialize((Device*)g_device));
    cmdBuf->SetMemoryCallback(CommandBufferMemoryCallback);
    cmdBuf->SetMemoryCallbackData(this);
    cmdBuf->BeginRecording();
    {
        // Set and clear RT
        Texture* rt = m_window.GetColorRt(rtIndex);
        cmdBuf->SetRenderTargets(1, &rt, NULL, NULL, NULL);
        static const float background[4] = { 0.0f, 1.0f, 0.5f, 1.0f };
        cmdBuf->ClearColor(0, background, ClearColorMask::RGBA);
        cmdBuf->BindProgram(m_graphicsProgram, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
        cmdBuf->BindVertexBuffer(0, m_vertexBuffer->GetAddress(), m_vertexBuffer->GetSize());
        cmdBuf->DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
    }
    const LWNcommandHandle commands = cmdBuf->EndRecording();

    // Submit
    m_queue->SubmitCommands(1, &commands);

    // Present
    m_queue->PresentTexture(m_window, rtIndex);

    // Wait for all outstanding GPU work to complete
    m_queue->FenceSync(m_cmdSync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
    m_queue->Finish();
    SyncWaitResult queueFinishResult = m_cmdSync->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);
    TEST_NE(queueFinishResult, SyncWaitResult::TIMEOUT_EXPIRED);
    TEST_NE(queueFinishResult, SyncWaitResult::FAILED);

    QueueGetErrorResult errorResult = m_queue->GetError(nullptr);
    TEST_EQ(errorResult, QueueGetErrorResult::GPU_NO_ERROR);

    return true;
}

bool AutoResourceTrackingTest::VerifyTrackedResources(int frameNumber)
{
    AftermathTestTrackedResourcesCounts resourceCounts = {};
    TEST_EQ(aftermathTestGetNumTrackedResources(&resourceCounts), AftermathApiError_None);

    const int perFrameResourcesPendingDeferredFinalization = frameNumber <= s_aftermathDefaultDeferredFinalizationAge ? frameNumber + 1 : s_aftermathDefaultDeferredFinalizationAge + 1;

    const size_t numTextures =
        2 +// Render targets
        s_numFixedTextures; // fixed texture resources
    const size_t numTexturesPendingDeferredFinalization =
        perFrameResourcesPendingDeferredFinalization * s_numPerFrameTextures; // Per frame textures pending deferred finalization
    const size_t numBuffers =
        3 + // vertex buffer and shader buffers
        s_numFixedBuffers; // fixed buffer resources
    const size_t numBuffersPendingDeferredFinalization =
        perFrameResourcesPendingDeferredFinalization * s_numPerFrameBuffers; // Per frame buffers pending deferred finalization
    const size_t numSamplerPools =
        s_numFixedSamplerPools; // fixed sampler pool resources
    const size_t numSamplerPoolsPendingDeferredFinalization =
        perFrameResourcesPendingDeferredFinalization * s_numPerFrameSamplerPools; // Per frame sampler pools pending deferred finalization
    const size_t numTexturePools =
        s_numFixedTexturePools; // fixed texture pool resources
    const size_t numTexturePoolsPendingDeferredFinalization =
        perFrameResourcesPendingDeferredFinalization * s_numPerFrameTexturePools; // Per frame texture pools pending deferred finalization

    TEST_EQ(resourceCounts.numTextures, numTextures);
    TEST_EQ(resourceCounts.numBuffers, numBuffers);
    TEST_EQ(resourceCounts.numSamplerPools, numSamplerPools);
    TEST_EQ(resourceCounts.numTexturePools, numTexturePools);

    // Since resource tracking data of finalized resources is reused when their finalization age has
    // been reached we cannot know the exact number of finalized resources that are still accessible.
    // However, their number must be always greater than the number of those pending deferred finalization.
    TEST_GE(resourceCounts.numFinalizedTextures, numTexturesPendingDeferredFinalization);
    TEST_GE(resourceCounts.numFinalizedBuffers, numBuffersPendingDeferredFinalization);
    TEST_GE(resourceCounts.numFinalizedSamplerPools, numSamplerPoolsPendingDeferredFinalization);
    TEST_GE(resourceCounts.numFinalizedTexturePools, numTexturePoolsPendingDeferredFinalization);

    return true;
}

bool AutoResourceTrackingTest::TestWithAutoDeferredFinalize(const Options& options)
{
    TEST(InitializeFixedObjects());

    static const int FrameCount = 15;
    for (int i = 0; i < FrameCount; ++i) {
        // Initialize some per-frame resources
        TEST(InitializePerFrameResources());

        TEST(RenderFrame());

        TEST(FinalizePerFrameResources());

        TEST(VerifyTrackedResources(i));
    }

    // Finalize all remaining resources and other LWN objects that were created
    TEST(FinalizeAllObjects());

    // Finalize the device
    TEST(FinalizeDevice());

    return true;
}

bool AutoResourceTrackingTest::Test(const Options& options)
{
    // Initialize graphics
    LWN::SetupLWNGraphics();

    TEST(TestPreInitializeDevice());

    TEST(InitializeDevice(!options.disableDebugLayer));

    TEST(TestWithAutoDeferredFinalize(options));

    return true;
}

// Integration test - requires Aftermath to be enabled by DevMenu setting!
AFTERMATH_DEFINE_TEST(AutoResourceTracking, INTEGRATION,
    LwError Execute(const Options& options)
    {
        AutoResourceTrackingTest test;
            bool success = test.Test(options);
            return success ? LwSuccess : LwError_IlwalidState;
    }
);

} // namespace AftermathTest
