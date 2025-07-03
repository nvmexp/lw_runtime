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

#include "shaders/TestCheckPoints_glslcoutput.h"
#include <lwnTool/lwnTool_GlslcInterface.h>

#include <AftermathFileFormat.h>
#include <AftermathApi.h>

namespace AftermathTest {

class CheckpointsTest {
public:
    bool Initialize(const Options& options);

    bool Test();

private:
    bool InitializeWindow();
    bool InitializeQueue();
    bool InitializeSync();
    bool InitializeShaders();
    bool InitializeVertexBuffer();
    bool RunRecordingThreads();
    bool RecordCmdBuffers();
    static void RecordCmdBuffer(void* data);
    void Reset();

    // Command buffer memory handling
    static void LWNAPIENTRY CommandBufferMemoryCallback(
        CommandBuffer* cmdBuf,
        CommandBufferMemoryEvent::Enum event,
        size_t minSize,
        void* callbackData);
    bool AddCommandMemory(CommandBuffer* cmdBuf, size_t minSize);
    bool AddControlMemory(CommandBuffer* cmdBuf, size_t minSize);

    static const size_t CONFIG_COUNT = 100;
    static const size_t REPEAT_COUNT = 10;

    // Command buffer recording threads
    static const size_t NUM_THREADS = 8;
    static const size_t THREAD_STACK_SIZE = 16384;
    NN_ALIGNAS(nn::os::ThreadStackAlignment) static uint8_t s_threadStack[NUM_THREADS][THREAD_STACK_SIZE];
    std::array<nn::os::ThreadType, NUM_THREADS> m_threads;
    std::relwrsive_mutex m_mutex;

    // Data buffer backing static markers
    static uint8_t s_staticMarkerData[LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_STATIC_MARKER_DATA_SIZE];

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

    // State for conlwrrent command buffer recording
    struct RecordingState
    {
        LWN::CommandBufferHolder cmdBuf;
        CommandHandle cmdHandle;
        int drawCallCount;
        int commandSetCount;
        int commandSetBaseIndex;
    };
    std::deque<RecordingState> m_recordingStates;
    std::atomic<std::size_t> m_lwrrentRecordingStateIndex;
    std::vector<CommandHandle> m_cmdHandles;

    // The data of the last user checkpoint event marker that will be submitted to the queue.
    // For verification of semapahore releases.
    std::string m_lastUserMarkerData;

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
};

NN_ALIGNAS(nn::os::ThreadStackAlignment) uint8_t CheckpointsTest::s_threadStack[NUM_THREADS][THREAD_STACK_SIZE];
uint8_t CheckpointsTest::s_staticMarkerData[LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_STATIC_MARKER_DATA_SIZE];

void LWNAPIENTRY CheckpointsTest::CommandBufferMemoryCallback(
    CommandBuffer* cmdBuf,
    CommandBufferMemoryEvent::Enum event,
    size_t minSize,
    void* callbackData)
{
    CheckpointsTest* test = reinterpret_cast<CheckpointsTest*>(callbackData);
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

bool CheckpointsTest::AddCommandMemory(CommandBuffer* cmdBuf, size_t minSize)
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

bool CheckpointsTest::AddControlMemory(CommandBuffer* cmdBuf, size_t minSize)
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

bool CheckpointsTest::InitializeWindow()
{
    m_window.Initialize(g_device);

    return true;
}

bool CheckpointsTest::InitializeQueue()
{
    m_queue.Initialize(g_device);

    return true;
}

bool CheckpointsTest::InitializeSync()
{
    m_cmdSync.Initialize((Device*)g_device);
    m_windowSync.Initialize((Device*)g_device);

    return true;
}

bool CheckpointsTest::InitializeShaders()
{
    struct glslcShaderStage
    {
        const char* data;
        size_t dataSize;
        const char* control;
        size_t controlSize;
    };
    std::array<glslcShaderStage, GLSLC_NUM_SHADER_STAGES> shaders;

    const GLSLCoutput* compiledProgram = reinterpret_cast<const GLSLCoutput*>(testCheckPoints_glslcoutput);
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
    lwnShaderData[0].control = shaders[ShaderStage::VERTEX].control;
    lwnShaderData[1].data = m_fragmentShader->GetAddress();
    lwnShaderData[1].control = shaders[ShaderStage::FRAGMENT].control;

    // Initialize the program and provide it with the compiled shader
    m_graphicsProgram.Initialize((Device*)g_device);
    m_graphicsProgram->SetShaders(lwnShaderData.size(), lwnShaderData.data());

    return true;
}

bool CheckpointsTest::InitializeVertexBuffer()
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

bool CheckpointsTest::Initialize(const Options& options)
{
    memset(s_staticMarkerData, '$', sizeof(s_staticMarkerData));

    // Setup allcoators, etc.
    LWN::SetupLWNGraphics();

    // Set Aftermath feature level required for checkpoints
    TEST_EQ(aftermathSetFeatureLevel(AftermathApiFeatureLevel_Enhanced), AftermathApiError_None);

    // Initialize the LWN device
    const DeviceFlagBits flags = options.disableDebugLayer ? 0 : DeviceFlagBits::DEBUG_ENABLE_LEVEL_4;
    LWN::SetupLWNDevice(flags);

    TEST(InitializeWindow());
    TEST(InitializeQueue());
    TEST(InitializeSync());
    TEST(InitializeShaders());
    TEST(InitializeVertexBuffer());

    return true;
}

bool CheckpointsTest::RunRecordingThreads()
{
    m_lwrrentRecordingStateIndex = 0;

    for (size_t i = 0; i < m_threads.size(); ++i) {
        nn::Result result = nn::os::CreateThread(&m_threads[i], RecordCmdBuffer, this, s_threadStack[i], THREAD_STACK_SIZE, nn::os::DefaultThreadPriority);
        TEST(result.IsSuccess());
    }

    for (size_t i = 0; i < m_threads.size(); ++i) {
        nn::os::StartThread(&m_threads[i]);
    }
    for (size_t i = 0; i < m_threads.size(); ++i) {
        nn::os::WaitThread(&m_threads[i]);
        nn::os::DestroyThread(&m_threads[i]);
    }

    return true;
}

bool CheckpointsTest::RecordCmdBuffers()
{
    std::random_device randDevice;
    std::mt19937 randGenerator(randDevice());
    std::uniform_int_distribution<int> maxCallStacksDepthDist(1, LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_CALL_STACK_CAPTURE_DEPTH);
    std::bernoulli_distribution autoCheckpointsEnabledDist(1.0/3.0);
    std::discrete_distribution<int> autoCheckpointsMaskDist({1, 1, 1, 1, 4}); // select one of the mask bits with propability as defined below
    int autoCheckpointsMasks[] = {
        LWN_AUTOMATIC_CHECKPOINTS_MASK_DRAW_BIT,                // probability 1/8
        LWN_AUTOMATIC_CHECKPOINTS_MASK_COMPUTE_DISPATCH_BIT,    // probability 1/8
        LWN_AUTOMATIC_CHECKPOINTS_MASK_COPY_BIT,                // probability 1/8
        LWN_AUTOMATIC_CHECKPOINTS_MASK_DEBUG_GROUP_BIT,         // probability 1/8
        LWN_AUTOMATIC_CHECKPOINTS_MASK_ALL                      // probability 1/2
    };
    std::uniform_int_distribution<int> autoCheckpointsSamplingIntervalDist(1, 20);
    std::bernoulli_distribution callStacksEnabledDist(2.0/3.0);
    std::uniform_int_distribution<int> drawCallCountDist(1, 5000);
    std::uniform_int_distribution<int> commandSetCountDist(1, 20);

    // Push all command buffer configs to the queue
    int commandSetsCount = 0;
    for (size_t i = 0; i < CONFIG_COUNT; ++i) {
        m_recordingStates.emplace_back();
        RecordingState& recordingState = m_recordingStates.back();

        // Initialize command buffer
        TEST(recordingState.cmdBuf.Initialize((Device*)g_device));
        recordingState.cmdBuf->SetMemoryCallback(CommandBufferMemoryCallback);
        recordingState.cmdBuf->SetMemoryCallbackData(this);

        recordingState.drawCallCount = drawCallCountDist(randGenerator);
        recordingState.commandSetCount = std::min(commandSetCountDist(randGenerator), recordingState.drawCallCount);
        recordingState.commandSetBaseIndex = commandSetsCount;

        commandSetsCount += recordingState.commandSetCount;
    }

    m_cmdHandles.resize(commandSetsCount);

    // Create randomized checkpoints config
    const bool autoCheckpointsEnabled = autoCheckpointsEnabledDist(randGenerator);
    const int autoCheckpointsMask = autoCheckpointsMasks[autoCheckpointsMaskDist(randGenerator)];
    const int autoCheckpointsSamplingInterval = autoCheckpointsSamplingIntervalDist(randGenerator);
    const uint32_t autoCheckpointsByDebugGroupDomainId = 0;
    const char* autoCheckpointsByDebugGroupName = nullptr;
    const bool callStacksEnabled = callStacksEnabledDist(randGenerator);
    const int maxCallStacksDepth = maxCallStacksDepthDist(randGenerator);

    // Per-device state setup
    TEST(g_device->SetAutomaticCheckpointsEnable(autoCheckpointsEnabled));
    TEST(g_device->SetAutomaticCheckpointsMask(autoCheckpointsMask));
    TEST(g_device->SetAutomaticCheckpointsSamplingInterval(autoCheckpointsSamplingInterval));
    TEST(g_device->SetAutomaticCheckpointsByDebugGroup(autoCheckpointsByDebugGroupDomainId, autoCheckpointsByDebugGroupName));
    TEST(g_device->SetCheckpointCallStacksEnable(callStacksEnabled));
    TEST(g_device->SetCheckpointMaxCallStacksDepth(maxCallStacksDepth));

    // Record the command buffers
    TEST(RunRecordingThreads());

    return true;
}

void CheckpointsTest::RecordCmdBuffer(void* data)
{
    CheckpointsTest* test = static_cast<CheckpointsTest*>(data);

    std::random_device randDevice;
    std::mt19937 randGenerator(randDevice());
    std::uniform_int_distribution<int> dynamicMarkerSizeDist(1, LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_DYNAMIC_MARKER_DATA_SIZE);
    std::uniform_int_distribution<int> staticMarkerSizeDist(1, LWN_DEVICE_INFO_CONSTANT_NX_CHECKPOINT_MAXIMUM_STATIC_MARKER_DATA_SIZE);
    std::bernoulli_distribution staticMarkerDist(1.0/3.0);

    do {
        const size_t recordingStateIndex = test->m_lwrrentRecordingStateIndex.fetch_add(1);
        if (recordingStateIndex >= test->m_recordingStates.size()) {
            break;
        }

        RecordingState& recordingState = test->m_recordingStates[recordingStateIndex];

        LWN::CommandBufferHolder& cmdBuf = recordingState.cmdBuf;

        const int drawcallsPerCommandSet = recordingState.drawCallCount / recordingState.commandSetCount;

        for (int j = 0; j < recordingState.commandSetCount; ++j) {
            cmdBuf->BeginRecording();
            {
                cmdBuf->PushDebugGroupDynamic(recordingStateIndex, "Render");

                const int drawcallCount = j == 0
                    ? recordingState.drawCallCount - (recordingState.commandSetCount - 1) * drawcallsPerCommandSet
                    : drawcallsPerCommandSet;

                for (int i = 0; i < drawcallCount; ++i) {
                    // Is this the last marker that will be submitted?
                    const bool lastMarker =
                        i == drawcallCount - 1 &&
                        j == recordingState.commandSetCount - 1 &&
                        recordingStateIndex == test->m_recordingStates.size() - 1;

                    if (staticMarkerDist(randGenerator)) {
                        const int staticMarkerSize = staticMarkerSizeDist(randGenerator);
                        cmdBuf->InsertCheckpointStatic(s_staticMarkerData, staticMarkerSize);
                        if (lastMarker) {
                            test->m_lastUserMarkerData.assign(s_staticMarkerData, s_staticMarkerData + staticMarkerSize);
                        }
                    }
                    else {
                        std::string marker(dynamicMarkerSizeDist(randGenerator), '#');
                        cmdBuf->InsertCheckpointDynamic(marker.c_str(), marker.size());
                        if (lastMarker) {
                            test->m_lastUserMarkerData.swap(marker);
                        }
                    }
                    cmdBuf->DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                }

                cmdBuf->PopDebugGroupId(recordingStateIndex);
            }
            test->m_cmdHandles[recordingState.commandSetBaseIndex + j] = cmdBuf->EndRecording();
        }
    } while (true);
}

void CheckpointsTest::Reset()
{
    // Reset Aftermath device state
    g_device->SetAutomaticCheckpointsEnable(false);
    g_device->SetAutomaticCheckpointsMask(LWN_AUTOMATIC_CHECKPOINTS_MASK_ALL);
    g_device->SetAutomaticCheckpointsSamplingInterval(1);
    g_device->SetAutomaticCheckpointsByDebugGroup(0, nullptr);
    g_device->SetCheckpointCallStacksEnable(false);
    g_device->SetCheckpointMaxCallStacksDepth(8);

    // Mark all allocated command memory pools and control memory buffers as free
    m_availableCmdMemoryPools.clear();
    for (CommandMemoryPool& cmdMemoryPool : m_cmdMemoryPools) {
        m_availableCmdMemoryPools.push_back(&cmdMemoryPool);
    }
    m_availableControlMemoryBuffers.clear();
    for (ControlMemoryBuffer& controlMemoryBuffer : m_controlMemoryBuffers) {
        m_availableControlMemoryBuffers.push_back(&controlMemoryBuffer);
    }

    // Clear last recording states
    m_recordingStates.clear();
    m_lastUserMarkerData.clear();
}

bool CheckpointsTest::Test()
{
    int rtIndex = 0;
    for (size_t i = 0; i < REPEAT_COUNT; ++i)
    {
        // get current RT index
        WindowAcquireTextureResult acquireTextureResult = m_window->AcquireTexture(m_windowSync, &rtIndex);
        TEST_EQ(acquireTextureResult, WindowAcquireTextureResult::SUCCESS);

        // Init pass
        LWN::CommandBufferHolder initCmds;
        TEST(initCmds.Initialize((Device*)g_device));
        initCmds->SetMemoryCallback(CommandBufferMemoryCallback);
        initCmds->SetMemoryCallbackData(this);
        initCmds->BeginRecording();
        {
            // Set and clear RT
            Texture* rt = m_window.GetColorRt(rtIndex);
            initCmds->SetRenderTargets(1, &rt, NULL, NULL, NULL);
            static const float background[4] = { 0.0f, 1.0f, 0.5f, 1.0f };
            initCmds->ClearColor(0, background, ClearColorMask::RGBA);
            initCmds->BindProgram(m_graphicsProgram, ShaderStageBits::VERTEX | ShaderStageBits::FRAGMENT);
            initCmds->BindVertexBuffer(0, m_vertexBuffer->GetAddress(), m_vertexBuffer->GetSize());
        }
        LWNcommandHandle initPass = initCmds->EndRecording();
        m_queue->SubmitCommands(1, &initPass);

        // Record the test command buffers
        TEST(RecordCmdBuffers());

        // Submit command buffers in batches of 3
        // This avoids exhausting queue control memory and still
        // allows to test submitting multiple command handles at once.
        size_t commandBuffersToSubmit = m_cmdHandles.size();
        for (size_t i = 0; commandBuffersToSubmit >= 3; i += 3) {
            m_queue->SubmitCommands(3, &m_cmdHandles[i]);
            m_queue->Flush();
            commandBuffersToSubmit -= 3;
        }
        if (commandBuffersToSubmit > 0) {
            m_queue->SubmitCommands(commandBuffersToSubmit, &m_cmdHandles[m_cmdHandles.size() - commandBuffersToSubmit]);
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

        Reset();
    }

    return true;
}

AFTERMATH_DEFINE_TEST(Checkpoints, UNIT,
    LwError Execute(const Options& options)
    {
        CheckpointsTest test;
        if (!test.Initialize(options)) {
            return LwError_IlwalidState;
        }

        bool success = test.Test();
        return success ? LwSuccess : LwError_IlwalidState;
    }
);

} // namespace AftermathTest
