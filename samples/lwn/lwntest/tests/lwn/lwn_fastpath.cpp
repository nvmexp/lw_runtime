/*
 * Copyright (c) 2020, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#define LWN_NOESCAPE

#include "cmdline.h"
#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <functional>
#include <vector>

namespace {

using namespace lwn;

class FastpathTest
{
public:
    LWNTEST_CppMethods();
};

lwString FastpathTest::getDescription() const
{
    return "Verify that fastpath functions produce results identical to the "
           "default versions. For CommandBuffers, run the functions on a parallel "
           "pair of CommandBuffers, then verify that the command memory for each "
           "contains the same contents. Fastpath functions are not compatible with "
           "the debug layer, so this test is not compatible with the -lwnDebug "
           "command-line parameter.";
}

int FastpathTest::isSupported() const
{
    // Fastpath functions are not compatible with the debug layer
    return !lwnDebugEnabled && lwogCheckLWNAPIVersion(55, 7);
}

void FastpathTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();

    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    CommandBuffer *cmdBufDefault = device->CreateCommandBuffer();
    CommandBuffer *cmdBufFastpath = device->CreateCommandBuffer();

    int commandSize = 65536;
    MemoryPool *pool = device->CreateMemoryPool(nullptr, commandSize * 2, MemoryPoolType::CPU_COHERENT);
    void* cmdMemoryDefault = pool->Map();
    memset(cmdMemoryDefault, 0, commandSize * 2);
    cmdBufDefault->AddCommandMemory(pool, 0, commandSize);
    void* cmdMemoryFastpath = static_cast<char*>(cmdMemoryDefault) + commandSize;
    cmdBufFastpath->AddCommandMemory(pool, commandSize, commandSize);

    int controlSize = 65536;
    std::vector<char> controlMemory(controlSize * 2);
    cmdBufDefault->AddControlMemory(controlMemory.data(), controlSize);
    cmdBufFastpath->AddControlMemory(controlMemory.data() + controlSize, controlSize);

#define API_FUNCTIONS \
    F(CommandBuffer, BeginRecording, void(CommandBuffer*)) \
    F(CommandBuffer, BindUniformBuffer, void(CommandBuffer*, ShaderStage, int, BufferAddress, size_t)) \
    F(CommandBuffer, BindUniformBuffers, void(CommandBuffer*, ShaderStage, int, int, LWN_NOESCAPE BufferRange*)) \
    /*F(CommandBuffer, BindVertexStreamState, void(CommandBuffer*, LWN_NOESCAPE const VertexStreamState*))*/ \
    F(CommandBuffer, BindVertexBuffer, void(CommandBuffer*, int, BufferAddress, size_t)) \
    F(CommandBuffer, DrawElements, void(CommandBuffer*, DrawPrimitive, IndexType, int, BufferAddress)) \
    F(CommandBuffer, UpdateUniformBuffer, void(CommandBuffer*, BufferAddress, size_t, ptrdiff_t, size_t, LWN_NOESCAPE void*))

    typedef struct {
        CommandBuffer* cmdBuf;
#define F(object, name, type) std::function<type> name;
        API_FUNCTIONS
#undef F
    } TestData;

    TestData testData[2] = {
        {
            cmdBufDefault,
#define F(object, name, type) &object::name,
            API_FUNCTIONS
#undef F
        },
        {
            cmdBufFastpath,
#define F(object, name, type) &object::name ## _fastpath,
            API_FUNCTIONS
#undef F
        }
    };

    for (int i = 0; i < 2; ++i) {
        testData[i].BeginRecording(testData[i].cmdBuf);
        testData[i].BindUniformBuffer(testData[i].cmdBuf, ShaderStage::FRAGMENT, 3,
                                      BufferAddress(0x12345678), 0);
        testData[i].BindUniformBuffer(testData[i].cmdBuf, ShaderStage::VERTEX, 7,
                                      BufferAddress(0x87654321), 4096);
        BufferRange ranges[2] = {
            { BufferAddress(0xabadcafe), 0 },
            { BufferAddress(0x1dead0af), 8192 },
        };
        testData[i].BindUniformBuffers(testData[i].cmdBuf, ShaderStage::GEOMETRY, 4, 2, ranges);
        testData[i].BindVertexBuffer(testData[i].cmdBuf, 5, 0x8badf00d, 2001);
        /* TODO: Implementing this is harder than expected.
        VertexStreamState vss;
        vss.SetDefaults().SetStride(3).SetDivisor(7);
        testData[i].BindVertexStreamState(testData[i].cmdBuf, &vss);*/
        testData[i].DrawElements(testData[i].cmdBuf, DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_INT, 1776, 0x8675309);
        // The fastpath implementation of UpdateUniformData will actually encode this differently
        // than the non-fastpath version for sufficiently large (> 508 bytes) updates. This is
        // because the driver version conservatively assumes that the application will
        // incrementally feed it minimum-sized chunks of command memory, whereas the fastpath
        // version will only work if the entire update will fit in the current block of command
        // memory.
        uint8_t updateData[400];
        for (int j = 0; j < (int) sizeof(updateData); ++j) {
            updateData[j] = uint8_t(j * 3);
        }
        testData[i].UpdateUniformBuffer(testData[i].cmdBuf, 0x12345678, 0x3140, 0x2710, sizeof(updateData), updateData);
        testData[i].cmdBuf->EndRecording();
    }

    bool result = memcmp(cmdMemoryDefault, cmdMemoryFastpath, commandSize) == 0;

    if (result) {
        queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
    } else {
        queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
    }

    queueCB.submit();
    queue->Finish();
}

} // namespace <unnamed>

OGTEST_CppTest(FastpathTest, lwn_fastpath, );
