/*
 * Copyright (c) 2018 - 2021, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"

#include "lwn_utils.h"

using namespace lwn;

namespace {

struct OomData {
    MemoryPool* commandPool;
    uint8_t *controlMem;
    size_t commandGranularity;
    size_t controlGranularity;
    uint32_t outOfCommandMemoryCalled;
    uint32_t outOfControlMemoryCalled;
};

// Handle out-of-memory callbacks. The initial allocation uses memory from
// OomData::commandPool and Oom::controlMem with offsets of zero and sizes of
// OomData::commandGranularity and OomData::controlGranularity. We expect at
// most one callback of each type and set up a chunk whose offset and size is
// the initial granularity.
void LWNAPIENTRY OomCallback(objects::CommandBuffer *cmdBuf, CommandBufferMemoryEvent::Enum event,
                             size_t minSize, void *callbackData)
{
    OomData* oomData = (OomData*)callbackData;
    if (event == CommandBufferMemoryEvent::OUT_OF_COMMAND_MEMORY) {
        cmdBuf->AddCommandMemory(oomData->commandPool, oomData->commandGranularity, oomData->commandGranularity);
        oomData->outOfCommandMemoryCalled++;
    } else if (event == CommandBufferMemoryEvent::OUT_OF_CONTROL_MEMORY) {
        cmdBuf->AddControlMemory(oomData->controlMem + oomData->controlGranularity, oomData->controlGranularity);
        oomData->outOfControlMemoryCalled++;
    }
}

} // anonymous namespace

class LWNCmdDisableOomTest {
public:
    LWNTEST_CppMethods();

private:
    bool runTest() const;
};

lwString LWNCmdDisableOomTest::getDescription() const
{
    return "Verify that CommandBuffer::SetCommandMemoryCallbackEnabled(false)\n"
           "and CommandBuffer::SetControlMemoryCallbackEnabled(false)\n"
           "prevents the out-of-memory callback from being called.";
}

int LWNCmdDisableOomTest::isSupported() const
{
    return 1;
}

bool LWNCmdDisableOomTest::runTest() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();

    // Set up a throwaway command buffer object for this test.
    CommandBuffer *commandBuffer = device->CreateCommandBuffer();

    // Allocate 64KB of command and control memory to use for recording.
    const size_t commandMemoryAllocSize = 64 * 1024;
    const size_t controlMemoryAllocSize = 64 * 1024;
    MemoryPool *commandPool = device->CreateMemoryPool(nullptr, commandMemoryAllocSize, MemoryPoolType::CPU_COHERENT);
    std::vector<uint8_t> controlMemory(controlMemoryAllocSize);

    // We will record a fixed number BindDepthStencilState and FenceSync calls,
    // which needs to result in recordings exceeding the minimum command and
    // control memory sizes.
    const uint32_t recordingStepIterations = 50;
    int commandMinSize = device->GetInfo(DeviceInfo::COMMAND_BUFFER_MIN_COMMAND_SIZE);
    int controlMinSize = device->GetInfo(DeviceInfo::COMMAND_BUFFER_MIN_CONTROL_SIZE);

    // Set up dummy sync and depth stencil state objects for this test.
    Sync *sync = device->CreateSync();
    DepthStencilState depthStencilState;
    depthStencilState.SetDefaults();

    // We run three recording passes reusing the same memory. The first uses
    // half the command buffer and records the total command and control memory
    // used. The second and third passes set the chunk size to *exactly* the
    // amount of memory used in the first pass. The second runs with callbacks
    // enabled and the third runs with callbacks disabled.
    size_t commandChunkSize = commandMemoryAllocSize / 2;
    size_t controlChunkSize = controlMemoryAllocSize / 2;

    for (int pass = 0; pass < 3; pass++) {

        // Set up our callback structure for cases where we trigger the callback.
        OomData oomData;
        oomData.commandPool = commandPool;
        oomData.controlMem = controlMemory.data();
        oomData.commandGranularity = commandChunkSize;
        oomData.controlGranularity = controlChunkSize;
        oomData.outOfCommandMemoryCalled = 0;
        oomData.outOfControlMemoryCalled = 0;

        // Set up the out-of-memory callbacks, but disable them on the third pass.
        commandBuffer->SetMemoryCallback(OomCallback);
        commandBuffer->SetMemoryCallbackData(&oomData);
        if (pass == 2) {
            commandBuffer->SetCommandMemoryCallbackEnabled(LWN_FALSE);
            commandBuffer->SetControlMemoryCallbackEnabled(LWN_FALSE);
        }

        // Set up the command buffer to start with chunks of memory at the
        // beginning of our allocations.
        commandBuffer->AddCommandMemory(commandPool, 0, commandChunkSize);
        commandBuffer->AddControlMemory(controlMemory.data(), controlChunkSize);

        // Record some commands. BindDepthStencilState should always use command
        // memory, while FenceSync should always use control memory. Test a long
        // run of "command memory" calls, followed by a long run of "control
        // memory" calls, followed by alternating calls.
        commandBuffer->BeginRecording();
        for (uint32_t i = 0; i < recordingStepIterations; i++) {
            commandBuffer->BindDepthStencilState(&depthStencilState);
        }
        for (uint32_t i = 0; i < recordingStepIterations; i++) {
            commandBuffer->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        }
        for (uint32_t i = 0; i < recordingStepIterations; i++) {
            commandBuffer->BindDepthStencilState(&depthStencilState);
            commandBuffer->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        }
        commandBuffer->EndRecording();
        size_t commandUsed = commandBuffer->GetCommandMemoryUsed();
        size_t controlUsed = commandBuffer->GetControlMemoryUsed();

        switch (pass) {
        case 0:
            // On the first pass, we provide way more memory than we expect to
            // use. Fail if we got any out-of-memory callback or if the actual
            // memory used by the commands ends up being less than the minimum
            // granularity required for command or control memory usage.
            if (oomData.outOfCommandMemoryCalled || oomData.outOfControlMemoryCalled) {
                return false;
            }
            if (commandUsed < (size_t)commandMinSize || controlUsed < (size_t)controlMinSize) {
                return false;
            }

            // Adjust our memory chunk sizes so that the subsequent passes will
            // have exactly enough memory to record the full set of commands.
            commandChunkSize = commandUsed;
            controlChunkSize = controlUsed;
            break;

        case 1:
            // On the second pass where we have barely enough memory to fit, we
            // will trigger both callbacks because (a) BindDepthStencilState
            // asks for more command memory than our state object requires and
            // (b) we reserve control memory to write a jump token in the event
            // we run out.
            if (!oomData.outOfCommandMemoryCalled || !oomData.outOfControlMemoryCalled) {
                return false;
            }
            break;

        case 2:
            // On the third pass, we have callback disabled and should run out
            // of memory because we have exactly enough memory. We should use
            // exactly the same amount of memory as in the first pass.
            if (oomData.outOfCommandMemoryCalled || oomData.outOfControlMemoryCalled) {
                return false;
            }
            if (commandUsed != commandChunkSize || controlUsed != controlChunkSize) {
                return false;
            }
            break;
        }
    }
    return true;
}

void LWNCmdDisableOomTest::doGraphics() const
{
    bool result = runTest();

    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    queueCB.ClearColor(0, result ? 0.0 : 1.0, result ? 1.0 : 0.0, 0.0, 1.0);
    queueCB.submit();
    deviceState->getQueue()->Finish();
}

OGTEST_CppTest(LWNCmdDisableOomTest, lwn_cmdbuf_disable_oom, );
