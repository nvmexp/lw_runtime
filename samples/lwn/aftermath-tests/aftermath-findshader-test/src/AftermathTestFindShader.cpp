/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <array>
#include <deque>
#include <mutex>
#include <random>
#include <vector>

#include <nn/os.h>
#include <nn/os/os_Thread.h>

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtilsLWN.h>

#include "shaders/TestFindShaderGfxProgram.h"
#include "shaders/TestFindShaderCompProgram.h"
#include <lwnTool/lwnTool_GlslcInterface.h>
#include <lwndevtools_bootstrap.h>
#include <lwndevtools_aftermath.h>

#include <glslc/lwnglslc_binary_layout.h>

#include <AftermathFileFormat.h>
#include <AftermathApi.h>

namespace AftermathTest {


typedef struct
{
    float Pos[2];
    float TexCoord[2];
} Vertex;


class FindShaderTest {
public:
    bool Initialize(const Options& options);

    bool Test();

private:
    bool InitializeWindow();
    bool InitializeQueue();
    bool InitializeSync();
    bool InitializeTexturesAndSamplers();
    bool InitializeShaders();
    bool InitializeVertexBuffer();
    bool FindShaderPCs();
    void Reset();

    // Command buffer memory handling
    static void LWNAPIENTRY CommandBufferMemoryCallback(
        CommandBuffer* cmdBuf,
        CommandBufferMemoryEvent::Enum event,
        size_t minSize,
        void* callbackData);
    bool AddCommandMemory(CommandBuffer* cmdBuf, size_t minSize);
    bool AddControlMemory(CommandBuffer* cmdBuf, size_t minSize);

    static const size_t REPEAT_COUNT = 10;

    // Command buffer recording threads
    std::relwrsive_mutex m_mutex;

    // Tracking of command buffer command memory
    struct CommandMemoryPool
    {
        LWN::UniqueUint8PtrWithLwstomDeleter buffer;
        LWN::MemoryPoolHolder pool;
    };
    std::deque<CommandMemoryPool> m_cmdMemoryPools;
    std::deque<CommandMemoryPool*> m_availableCmdMemoryPools;

    // Tracking of command buffer control memory
    struct ControlMemoryBuffer
    {
        LWN::UniqueUint8PtrWithLwstomDeleter buffer;
        size_t size;
    };
    std::deque<ControlMemoryBuffer> m_controlMemoryBuffers;
    std::deque<ControlMemoryBuffer*> m_availableControlMemoryBuffers;

    // LWN objects
    LWN::WindowHolder m_window;
    LWN::QueueHolder m_queue;
    LWN::SyncHolder m_cmdSync;
    LWN::SyncHolder m_windowSync;
    LWN::ShaderBufferHolder m_vertexShader;
    LWN::ShaderBufferHolder m_fragmentShader;
    LWN::ShaderBufferHolder m_computeShader;
    LWN::ProgramHolder m_graphicsProgram;
    LWN::ProgramHolder m_computeProgram;
    LWN::UniqueUint8PtrWithLwstomDeleter m_vertexBufferData;
    LWN::MemoryPoolHolder m_vertexBufferPool;
    LWN::BufferHolder m_vertexBuffer;

    LWN::UniqueUint8PtrWithLwstomDeleter m_textureBufferData;
    LWN::MemoryPoolHolder m_textureBufferPool;
    LWN::TexturePoolHolder m_texturePool;
    LWN::TextureHolder m_texture;
    int m_numReservedTextures;
    LWN::UniqueUint8PtrWithLwstomDeleter m_samplerBufferData;
    LWN::MemoryPoolHolder m_samplerBufferPool;
    LWN::SamplerPoolHolder m_samplerPool;
    LWN::SamplerHolder m_sampler;
    int m_numReservedSamplers;

    std::array<LWNdevtoolsSphPrepad, GLSLC_NUM_SHADER_STAGES> m_expectedPrepads;
};

static const uint32_t TEXTURE_WIDTH = 512;
static const uint32_t TEXTURE_HEIGHT = 512;

static float vertexData[] = {-1.0f, -1.0f,
                            1.0f, -1.0f,
                            -1.0f, 1.0f,
                            1.0f, 1.0f};

static float texcoordData[] = {0.0f, 0.0f,
                                1.0f, 0.0f,
                                0.0f, 1.0f,
                                1.0f, 1.0f};

void LWNAPIENTRY FindShaderTest::CommandBufferMemoryCallback(
    CommandBuffer* cmdBuf,
    CommandBufferMemoryEvent::Enum event,
    size_t minSize,
    void* callbackData)
{
    FindShaderTest* test = reinterpret_cast<FindShaderTest*>(callbackData);
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

bool FindShaderTest::AddCommandMemory(CommandBuffer* cmdBuf, size_t minSize)
{
    std::lock_guard<std::relwrsive_mutex> lock(m_mutex);

    const size_t cmdMemorySize =
        std::max<size_t>(
            Utils::AlignUp(minSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY),
            LWN_MEMORY_POOL_STORAGE_GRANULARITY * 8);

    // Try to re-use an existing pool
    auto i = std::find_if(m_availableCmdMemoryPools.begin(), m_availableCmdMemoryPools.end(),
        [cmdMemorySize](const CommandMemoryPool* cmdMemoryPool) { return cmdMemoryPool->pool->GetSize() >= cmdMemorySize; });
    if (i != m_availableCmdMemoryPools.end()) {
        CommandMemoryPool& cmdMemoryPool = **i;
        cmdBuf->AddCommandMemory(cmdMemoryPool.pool, 0, cmdMemoryPool.pool->GetSize());
        m_availableCmdMemoryPools.erase(i);
        return true;
    }

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

bool FindShaderTest::AddControlMemory(CommandBuffer* cmdBuf, size_t minSize)
{
    std::lock_guard<std::relwrsive_mutex> lock(m_mutex);

    const size_t controlMemorySize =
        std::max<size_t>(
            Utils::AlignUp(minSize, LWN_DEVICE_INFO_CONSTANT_NX_COMMAND_BUFFER_CONTROL_ALIGNMENT),
            LWN_DEVICE_INFO_CONSTANT_NX_COMMAND_BUFFER_CONTROL_ALIGNMENT * 1024);

    // Try to re-use an existing buffer
    auto i = std::find_if(m_availableControlMemoryBuffers.begin(), m_availableControlMemoryBuffers.end(),
        [controlMemorySize](const ControlMemoryBuffer* controlMemoryBuffer) { return controlMemoryBuffer->size >= controlMemorySize; });
    if (i != m_availableControlMemoryBuffers.end()) {
        ControlMemoryBuffer& controlMemoryBuffer = **i;
        cmdBuf->AddControlMemory(controlMemoryBuffer.buffer.get(), controlMemoryBuffer.size);
        m_availableControlMemoryBuffers.erase(i);
        return true;
    }

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

bool FindShaderTest::InitializeWindow()
{
    m_window.Initialize(g_device);

    return true;
}

bool FindShaderTest::InitializeQueue()
{
    m_queue.Initialize(g_device);

    return true;
}

bool FindShaderTest::InitializeSync()
{
    m_cmdSync.Initialize((Device*)g_device);
    m_windowSync.Initialize((Device*)g_device);

    return true;
}

bool FindShaderTest::InitializeTexturesAndSamplers()
{
    TextureBuilder textureBuilder;
    textureBuilder
        .SetDevice(g_device)
        .SetDefaults()
        .SetFlags(TextureFlags::IMAGE)
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::RGBA8)
        .SetSize2D(TEXTURE_WIDTH, TEXTURE_HEIGHT);

    int textureSize;
    g_device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE, &textureSize);
    g_device->GetInteger(DeviceInfo::RESERVED_TEXTURE_DESCRIPTORS, &m_numReservedTextures);
    size_t textureBufferSize =
        (textureBuilder.GetStorageSize() + textureBuilder.GetStorageAlignment()) +
        (m_numReservedTextures + 2) * textureSize;
    textureBufferSize = Utils::AlignUp(textureBufferSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);
    m_textureBufferData = LWN::AlignedAllocPodType<uint8_t>(textureBufferSize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    TEST_NE(m_textureBufferData, nullptr);

    MemoryPoolBuilder poolBuilder;
    poolBuilder.SetDevice(g_device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(m_textureBufferData.get(), textureBufferSize);
    TEST(m_textureBufferPool.Initialize(&poolBuilder));

    textureBuilder.SetStorage(m_textureBufferPool, textureBuilder.GetStorageAlignment());
    TEST(m_texture.Initialize(&textureBuilder));

    TEST(m_texturePool.Initialize((MemoryPool*)m_textureBufferPool,
        (textureBuilder.GetStorageSize() + textureBuilder.GetStorageAlignment()),
        m_numReservedTextures + 2));

    m_texturePool->RegisterTexture(m_numReservedTextures + 0, m_texture, nullptr);
    m_texturePool->RegisterImage(m_numReservedTextures + 1, m_texture, nullptr);

    int samplerSize;
    g_device->GetInteger(DeviceInfo::SAMPLER_DESCRIPTOR_SIZE, &samplerSize);
    g_device->GetInteger(DeviceInfo::RESERVED_SAMPLER_DESCRIPTORS, &m_numReservedSamplers);
    size_t samplerBufferSize = (m_numReservedSamplers + 1) * samplerSize;
    samplerBufferSize = Utils::AlignUp(samplerBufferSize, LWN_MEMORY_POOL_STORAGE_GRANULARITY);
    m_samplerBufferData = LWN::AlignedAllocPodType<uint8_t>(samplerBufferSize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    TEST_NE(m_samplerBufferData, nullptr);

    poolBuilder.SetDevice(g_device)
        .SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(m_samplerBufferData.get(), samplerBufferSize);
    TEST(m_samplerBufferPool.Initialize(&poolBuilder));
    TEST(m_samplerPool.Initialize((MemoryPool*)m_samplerBufferPool, 0, m_numReservedSamplers + 1));

    SamplerBuilder samplerBuilder;
    samplerBuilder
        .SetDevice(g_device)
        .SetDefaults();
    TEST(m_sampler.Initialize(&samplerBuilder));
    m_samplerPool->RegisterSampler(m_numReservedSamplers + 0, m_sampler);

    return true;
}

bool FindShaderTest::InitializeShaders()
{
    struct glslcShaderStage
    {
        const char* data;
        size_t dataSize;
        const char* control;
        size_t controlSize;
    };
    std::array<glslcShaderStage, GLSLC_NUM_SHADER_STAGES> shaders;

    const GLSLCoutput* compiledGfxProgram = reinterpret_cast<const GLSLCoutput*>(TestFindShaderGfxProgramData);
    for (unsigned int i = 0; i < compiledGfxProgram->numSections; ++i)
    {
        if (compiledGfxProgram->headers[i].genericHeader.common.type == GLSLC_SECTION_TYPE_GPU_CODE) {
            const GLSLCgpuCodeHeader& gpuCodeHeader = compiledGfxProgram->headers[i].gpuCodeHeader;

            const ShaderStage::Enum stage = ShaderStage::Enum(gpuCodeHeader.stage);

            const char* base = (char*)compiledGfxProgram + gpuCodeHeader.common.dataOffset;
            shaders[stage].data = base + gpuCodeHeader.dataOffset;
            shaders[stage].dataSize = gpuCodeHeader.dataSize;
            shaders[stage].control = base + gpuCodeHeader.controlOffset;
            shaders[stage].controlSize = gpuCodeHeader.controlSize;
        }
    }

    const GLSLCoutput* compiledCompProgram = reinterpret_cast<const GLSLCoutput*>(TestFindShaderCompProgramData);
    for (unsigned int i = 0; i < compiledCompProgram->numSections; ++i)
    {
        if (compiledCompProgram->headers[i].genericHeader.common.type == GLSLC_SECTION_TYPE_GPU_CODE) {
            const GLSLCgpuCodeHeader& gpuCodeHeader = compiledCompProgram->headers[i].gpuCodeHeader;

            const ShaderStage::Enum stage = ShaderStage::Enum(gpuCodeHeader.stage);

            const char* base = (char*)compiledCompProgram + gpuCodeHeader.common.dataOffset;
            shaders[stage].data = base + gpuCodeHeader.dataOffset;
            shaders[stage].dataSize = gpuCodeHeader.dataSize;
            shaders[stage].control = base + gpuCodeHeader.controlOffset;
            shaders[stage].controlSize = gpuCodeHeader.controlSize;
        }
    }

    // Expected shaders
    TEST(shaders[ShaderStage::VERTEX].dataSize != 0);
    TEST(shaders[ShaderStage::FRAGMENT].dataSize != 0);
    TEST(shaders[ShaderStage::COMPUTE].dataSize != 0);

    m_vertexShader.Initialize(g_device, shaders[ShaderStage::VERTEX].data, shaders[ShaderStage::VERTEX].dataSize);
    m_fragmentShader.Initialize(g_device, shaders[ShaderStage::FRAGMENT].data, shaders[ShaderStage::FRAGMENT].dataSize);
    m_computeShader.Initialize(g_device, shaders[ShaderStage::COMPUTE].data, shaders[ShaderStage::COMPUTE].dataSize);

    std::array<ShaderData, 2> lwnShaderData;
    lwnShaderData[0].data = m_vertexShader->GetAddress();
    lwnShaderData[0].control = shaders[ShaderStage::VERTEX].control;
    lwnShaderData[1].data = m_fragmentShader->GetAddress();
    lwnShaderData[1].control = shaders[ShaderStage::FRAGMENT].control;

    // Initialize the program and provide it with the compiled shader
    m_graphicsProgram.Initialize((Device*)g_device);
    m_graphicsProgram->SetShaders(lwnShaderData.size(), lwnShaderData.data());

    // See lwn/glslc/lwnglslc_binary_layout.h for proper offsets
    TEST(*((uint32_t*)(shaders[ShaderStage::VERTEX].control + offsetof(GLSLCBinaryVersion1Data, programGlStage))) == ShaderStage::VERTEX);
    m_expectedPrepads[ShaderStage::VERTEX] = {
        GLSLC_GFX_GPU_CODE_SECTION_DATA_MAGIC_NUMBER,
        *((uint32_t*)(shaders[ShaderStage::VERTEX].control + offsetof(GLSLCBinaryVersion1Data, ucodeSize))),
        { (LWNprogram *)&m_graphicsProgram },
        *((uint64_t*)(shaders[ShaderStage::VERTEX].control + offsetof(GLSLCBinaryVersion1Data, debugHash)))};
    TEST(*((uint32_t*)(shaders[ShaderStage::FRAGMENT].control + offsetof(GLSLCBinaryVersion1Data, programGlStage))) == ShaderStage::FRAGMENT);
    m_expectedPrepads[ShaderStage::FRAGMENT] = {
        GLSLC_GFX_GPU_CODE_SECTION_DATA_MAGIC_NUMBER,
        *((uint32_t*)(shaders[ShaderStage::FRAGMENT].control + offsetof(GLSLCBinaryVersion1Data, ucodeSize))),
        { (LWNprogram *)&m_graphicsProgram },
        *((uint64_t*)(shaders[ShaderStage::FRAGMENT].control + offsetof(GLSLCBinaryVersion1Data, debugHash)))};

    lwnShaderData[0].data = m_computeShader->GetAddress();
    lwnShaderData[0].control = shaders[ShaderStage::COMPUTE].control;

    m_computeProgram.Initialize((Device*)g_device);
    m_computeProgram->SetShaders(1, lwnShaderData.data());
    TEST(*((uint32_t*)(shaders[ShaderStage::COMPUTE].control + offsetof(GLSLCBinaryVersion1Data, programGlStage))) == ShaderStage::COMPUTE);
    m_expectedPrepads[ShaderStage::COMPUTE] = {
        GLSLC_COMP_GPU_CODE_SECTION_DATA_MAGIC_NUMBER,
        *((uint32_t*)(shaders[ShaderStage::COMPUTE].control + offsetof(GLSLCBinaryVersion1Data, ucodeSize))),
        { (LWNprogram *)&m_computeProgram },
        *((uint64_t*)(shaders[ShaderStage::COMPUTE].control + offsetof(GLSLCBinaryVersion1Data, debugHash)))};

    return true;
}

bool FindShaderTest::InitializeVertexBuffer()
{
    const size_t vertexBufferSize = sizeof(vertexData) + sizeof(texcoordData);
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

    void *ptr = (Vertex*)m_vertexBufferPool->Map();
    memcpy(ptr, vertexData, sizeof(vertexData));
    memcpy((char *)ptr + sizeof(vertexData), texcoordData, sizeof(texcoordData));
    m_vertexBufferPool->FlushMappedRange(0, vertexBufferSize);

    return true;
}

bool FindShaderTest::Initialize(const Options& options)
{
    // Setup allcoators, etc.
    LWN::SetupLWNGraphics();

    // Enable Aftermath feature level FULL
    TEST_EQ(aftermathSetFeatureLevel(AftermathApiFeatureLevel_Full), AftermathApiError_None);

    // Initialize the LWN device
    LWN::SetupLWNDevice(!options.disableDebugLayer);

    TEST(InitializeWindow());
    TEST(InitializeQueue());
    TEST(InitializeSync());
    TEST(InitializeTexturesAndSamplers());
    TEST(InitializeShaders());
    TEST(InitializeVertexBuffer());

    return true;
}

void FindShaderTest::Reset()
{
    // Mark all allocated command memory pools and control memory buffers as free
    m_availableCmdMemoryPools.clear();
    for (CommandMemoryPool& cmdMemoryPool : m_cmdMemoryPools) {
        m_availableCmdMemoryPools.push_back(&cmdMemoryPool);
    }
    m_availableControlMemoryBuffers.clear();
    for (ControlMemoryBuffer& controlMemoryBuffer : m_controlMemoryBuffers) {
        m_availableControlMemoryBuffers.push_back(&controlMemoryBuffer);
    }
}

struct AftermathTestFindShaderData
{
    const LWNdevice* device;
    LwU64 gpuva;
    struct LWNdevtoolsAftermathShaderInfo shaderInfo;
    LwBool result;
};

AFTERMATHAPI AftermathApiError aftermathTestFindShader(AftermathTestFindShaderData* data);

bool FindShaderTest::FindShaderPCs()
{
    static const LwU32 SPH_PREPAD_SIZE = 0x30;
    static const LwU32 SPH_SIZE = 0x50;
    static const LwU32 GFX_PREPAD_SIZE = SPH_PREPAD_SIZE + SPH_SIZE;
    static const LwU32 COMPUTE_PREPAD_SIZE = 0x100;
    static const LwU32 INSTR_SIZE = sizeof(LwU64);

    std::random_device randDevice;
    std::mt19937 randGenerator(randDevice());
    // We might fall on an OPEX, but it does not matter for the sake of the test
    std::uniform_int_distribution<int> vertextOffsetDist(0, (m_vertexShader->GetSize() - GFX_PREPAD_SIZE) / INSTR_SIZE);
    std::uniform_int_distribution<int> fragmentOffsetDist(0, (m_fragmentShader->GetSize() - GFX_PREPAD_SIZE) / INSTR_SIZE);
    std::uniform_int_distribution<int> computeOffsetDist(0, (m_computeShader->GetSize() - COMPUTE_PREPAD_SIZE) / INSTR_SIZE);

    AftermathTestFindShaderData shaderData;
    shaderData.device = g_device;

    auto testShaderData = [this, &shaderData](const LwU64 gpuva, const ShaderStage stage, const LwU64 expectedShaderVA) -> bool {
        shaderData.gpuva = gpuva;
        TEST_EQ(aftermathTestFindShader(&shaderData), AftermathApiError_None);
        TEST_EQ(shaderData.result, true);
        TEST_EQ(shaderData.shaderInfo.type, stage);
        TEST_EQ((((uint64_t)shaderData.shaderInfo.debugdataHash.debugHashHi32 << 32) | ((uint64_t)shaderData.shaderInfo.debugdataHash.debugHashLo32)), m_expectedPrepads[stage].debugHash);
        if (stage == ShaderStage::COMPUTE)
        {
            TEST_EQ(shaderData.shaderInfo.baseAddress, expectedShaderVA);
            TEST_EQ(shaderData.shaderInfo.size, m_expectedPrepads[stage].ucodeSize);
        }
        else
        {
            TEST_EQ(shaderData.shaderInfo.baseAddress, expectedShaderVA);
            TEST_EQ(shaderData.shaderInfo.size, m_expectedPrepads[stage].ucodeSize + SPH_PREPAD_SIZE);
        }return true;
    };

    BufferAddress vertexUcodeStart = m_vertexShader->GetAddress() + GFX_PREPAD_SIZE;
     // First Instruction
    testShaderData(vertexUcodeStart,
        ShaderStage::VERTEX,
        m_vertexShader->GetAddress());
    // Last Instruction
    testShaderData(vertexUcodeStart + m_vertexShader->GetSize() - GFX_PREPAD_SIZE - INSTR_SIZE,
        ShaderStage::VERTEX,
        m_vertexShader->GetAddress());
    for (size_t i = 0; i < 10; ++i) {
        testShaderData(vertexUcodeStart + vertextOffsetDist(randGenerator) * INSTR_SIZE,
            ShaderStage::VERTEX,
            m_vertexShader->GetAddress());
    }
    BufferAddress fragmentUcodeStart = m_fragmentShader->GetAddress() + GFX_PREPAD_SIZE;
    // First Instruction
    testShaderData(fragmentUcodeStart,
        ShaderStage::FRAGMENT,
        m_fragmentShader->GetAddress());
    // Last Instruction
    testShaderData(fragmentUcodeStart + m_fragmentShader->GetSize() - GFX_PREPAD_SIZE - INSTR_SIZE,
        ShaderStage::FRAGMENT,
        m_fragmentShader->GetAddress());
    for (size_t i = 0; i < 10; ++i) {
        testShaderData(fragmentUcodeStart + fragmentOffsetDist(randGenerator) * INSTR_SIZE,
            ShaderStage::FRAGMENT,
            m_fragmentShader->GetAddress());
    }
    BufferAddress computeUcodeStart = m_computeShader->GetAddress() + COMPUTE_PREPAD_SIZE;
    // First Instruction
    testShaderData(computeUcodeStart,
        ShaderStage::COMPUTE,
        m_computeShader->GetAddress() + COMPUTE_PREPAD_SIZE);
    // Last Instruction
    testShaderData(computeUcodeStart + m_computeShader->GetSize() - COMPUTE_PREPAD_SIZE - INSTR_SIZE,
        ShaderStage::COMPUTE,
        m_computeShader->GetAddress() + COMPUTE_PREPAD_SIZE);
    for (size_t i = 0; i < 10; ++i) {
        testShaderData(computeUcodeStart + computeOffsetDist(randGenerator) * INSTR_SIZE,
            ShaderStage::COMPUTE,
            m_computeShader->GetAddress() + COMPUTE_PREPAD_SIZE);
    }

    shaderData.device = g_device;
    shaderData.gpuva = 0x00000000;
    TEST_EQ(aftermathTestFindShader(&shaderData), AftermathApiError_None);
    TEST_EQ(shaderData.result, false);

    shaderData.device = g_device;
    shaderData.gpuva = 0xDEADBEEF;
    TEST_EQ(aftermathTestFindShader(&shaderData), AftermathApiError_None);
    TEST_EQ(shaderData.result, false);

    return true;
}

bool FindShaderTest::Test()
{
    int rtIndex = 0;
    for (size_t i = 0; i < REPEAT_COUNT; ++i)
    {
        // get current RT index
        WindowAcquireTextureResult acquireTextureResult = m_window->AcquireTexture(m_windowSync, &rtIndex);
        TEST_EQ(acquireTextureResult, WindowAcquireTextureResult::SUCCESS);

        // Init pass
        {
            LWN::CommandBufferHolder initCmds;
            TEST(initCmds.Initialize((Device*)g_device));
            initCmds->SetMemoryCallback(CommandBufferMemoryCallback);
            initCmds->SetMemoryCallbackData(this);
            initCmds->BeginRecording();
            initCmds->SetViewport(0, 0, m_window.GetWidth(), m_window.GetHeight());
            initCmds->SetScissor(0, 0, m_window.GetWidth(), m_window.GetHeight());
            static const float background[4] = { 0.0f, 1.0f, 0.5f, 1.0f };
            initCmds->ClearColor(0, background, ClearColorMask::RGBA);
            initCmds->SetTexturePool(m_texturePool);
            initCmds->SetSamplerPool(m_samplerPool);
            LWNcommandHandle initPass = initCmds->EndRecording();
            m_queue->SubmitCommands(1, &initPass);
        }

        // Compute Pass
        {
            LWN::CommandBufferHolder computeCmds;
            TEST(computeCmds.Initialize((Device*)g_device));
            computeCmds->SetMemoryCallback(CommandBufferMemoryCallback);
            computeCmds->SetMemoryCallbackData(this);
            computeCmds->BeginRecording();
            computeCmds->BindImage(ShaderStage::COMPUTE, 0, g_device->GetImageHandle(m_numReservedTextures + 1));
            computeCmds->BindProgram(m_computeProgram, ShaderStageBits::COMPUTE);
            computeCmds->DispatchCompute(TEXTURE_WIDTH/32, TEXTURE_HEIGHT/32, 1);
            computeCmds->Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_TEXTURE | BarrierBits::ILWALIDATE_SHADER);
            LWNcommandHandle computePass = computeCmds->EndRecording();
            m_queue->SubmitCommands(1, &computePass);
        }

        // Draw Pass
        {
            LWN::CommandBufferHolder drawCmds;
            TEST(drawCmds.Initialize((Device*)g_device));
            drawCmds->SetMemoryCallback(CommandBufferMemoryCallback);
            drawCmds->SetMemoryCallbackData(this);
            drawCmds->BeginRecording();
            Texture* rt = m_window.GetColorRt(rtIndex);
            drawCmds->SetRenderTargets(1, &rt, NULL, NULL, NULL);
            drawCmds->BindProgram(m_graphicsProgram, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
            drawCmds->BindVertexBuffer(0, m_vertexBuffer->GetAddress(), sizeof(vertexData));
            drawCmds->BindVertexBuffer(1, m_vertexBuffer->GetAddress() + sizeof(vertexData), sizeof(texcoordData));
            drawCmds->BindTexture(ShaderStage::FRAGMENT, 0, g_device->GetTextureHandle(m_numReservedTextures + 0, m_numReservedSamplers + 0));
            drawCmds->DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            LWNcommandHandle drawPass = drawCmds->EndRecording();
            m_queue->SubmitCommands(1, &drawPass);
        }

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

        TEST(FindShaderPCs());

        Reset();
    }

    return true;
}

AFTERMATH_DEFINE_TEST(FindShader, UNIT,
    LwError Execute(const Options& options)
    {
        FindShaderTest test;
        if (!test.Initialize(options)) {
            return LwError_IlwalidState;
        }

        bool success = test.Test();
        return success ? LwSuccess : LwError_IlwalidState;
    }
);

} // namespace AftermathTest
