/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>

#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <array>
#include <cassert>

#define METHOD_INDEX_TO_ADDR(methodIndex) (methodIndex * 4)
static const uint32_t ReceiveEventMaxCount = 256u;
static const uint32_t StateDeltaValueMaxCount = 64u;
static const uint32_t IlwalidStateDeltaValue = ~0u;

class SubmitCommandsSegmentedValidator {
public:
    void Initialize();
    bool TestEvents();
    bool TestStateDeltas();

    // Map from method to method values

    // TODO: (https://jirasw.lwpu.com/browse/LLGD-2628) Investigate why unordered_map<uint32_t, vector<uint32_t>> leads to crash on HR24
    using StateDeltas = std::map<uint32_t, std::array<uint32_t, StateDeltaValueMaxCount>>;

private:

    enum Event
    {
        DRAW_ARRAYS,            // Generic GPFIFO event method
        DRAW_ARRAYS_INDIRECT,   // GPFIFO indirect event method
        DISPATCH,               // Token event
        DRAW_TRANSFORM_FEEDBACK,// GPFIFO indirect event method
        COPY_BUFFER_TO_BUFFER,  // Bookended event methods
        UPDATE_UNIFORM_BUFFER,  // Bookended event methods
    };

    using StateChangeFn = std::function<StateDeltas(llgd_lwn::CommandBufferHolder&)>;

    void InitializeLWN();
    void InitializePools();
    bool DoEventTest(int repeatCount, int beginStateChanges, int endStateChanges, int preStateChanges, int postStateChanges, Event *events, int eventCount);
    bool DoStateDeltaTest(Event *events, int eventCount, const StateChangeFn& beginStateChangeFn); // beginStateChangeFn returns expected state deltas should be received by this state change
    static void SubmitCommandHandle(LWNqueue* queue, void* userData);
    static void ProcessEvent(const LlgdEvent& event, void* userData);
    static void PreSubmitEvent(LWNqueue* queue, void* userData);
    static void PostSubmitEvent(LWNqueue* queue, void* userData);
    static void ProcessStateDeltaMethod(uint32_t method, uint32_t numValues, const uint32_t* values, void* userData);
    void DoPreSubmitEvent();
    void DoPostSubmitEvent();
    void DoProcessStateDeltaMethod(uint32_t method, uint32_t numValues, const uint32_t* values);
    void ResetSubmittedEvents();
    bool VerifyEventCounts(int expectedEventCount);
    bool VerifySubmittedEvents(const std::vector<LlgdEvent>& submittedEvents);
    bool VerifyStateDeltas();

    static const size_t            SIZE = 4096 * 64;
    uint8_t              cmd_space[SIZE] __attribute__((aligned(4096)));
    uint8_t             ctrl_space[SIZE] __attribute__((aligned(4096)));
    uint8_t         indirect_space[SIZE] __attribute__((aligned(4096)));

    llgd_lwn::QueueHolder           qh;
    llgd_lwn::MemoryPoolHolder      mph;
    llgd_lwn::MemoryPoolHolder      imph;
    llgd_lwn::CommandBufferHolder   cbh;

    // TODO: (https://jirasw.lwpu.com/browse/LLGD-2628) Investigate why variable length array (e.g. std::vector) leads to crash on HR24
    std::array<LlgdEvent, ReceiveEventMaxCount> submittedEvents;
    int preSubmitEventCount;
    int postSubmitEventCount;

    StateDeltas receivedStateDeltas;
    StateDeltas expectedToReceiveDeltas;
};

void SubmitCommandsSegmentedValidator::InitializeLWN()
{
    qh.Initialize(g_device);

    CHECK(cbh.Initialize((Device*)g_device));
}

void SubmitCommandsSegmentedValidator::InitializePools()
{
    MemoryPoolBuilder pool_builder;

    pool_builder.SetDevice(g_device).SetDefaults()
                .SetFlags(MemoryPoolFlags::CPU_CACHED |
                          MemoryPoolFlags::GPU_CACHED);

    pool_builder.SetStorage(cmd_space, SIZE);
    CHECK(mph.Initialize(&pool_builder));

    pool_builder.SetStorage(indirect_space, SIZE);
    CHECK(imph.Initialize(&pool_builder));
}

void SubmitCommandsSegmentedValidator::Initialize()
{
    InitializeLWN();
    InitializePools();
}

static void NullSubmitCommands(LWNqueue *queue, int numCommands, LWN_NOESCAPE const LWNcommandHandle *handles)
{
    // Does nothing
}

static void NullFinish(LWNqueue *queue)
{
    // Does nothing
}

bool SubmitCommandsSegmentedValidator::DoEventTest(int repeatCount, int beginStateChanges, int endStateChanges, int preStateChanges, int postStateChanges, Event *events, int eventCount)
{
    const int expectedEventCount = repeatCount * eventCount;

    std::vector<LlgdEvent> expectedEvents;

    cbh->AddCommandMemory(mph, 0, SIZE);
    cbh->AddControlMemory(ctrl_space, SIZE);
    cbh->BeginRecording();
    {
        for (int i = 0; i < beginStateChanges; ++i) {
            cbh->SetLineWidth(1.0f);
        }
        for (int i = 0; i < repeatCount; ++i) {
            for (int j = 0; j < preStateChanges; ++j) {
                cbh->SetAlphaRef(1.0f);
            }

            for (int k = 0; k < eventCount; ++k) {
                LlgdEvent expectedEvent;

                switch (events[k]) {
                case DRAW_ARRAYS:
                    expectedEvent.type = LLGD_EVENT_TYPE_DRAW_ARRAYS;
                    expectedEvent.drawArrays.mode = LWN_DRAW_PRIMITIVE_TRIANGLES;
                    expectedEvent.drawArrays.first = 0;
                    expectedEvent.drawArrays.count = 3;
                    cbh->DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                    break;
                case DRAW_ARRAYS_INDIRECT:
                    expectedEvent.type = LLGD_EVENT_TYPE_DRAW_ARRAYS_INDIRECT;
                    expectedEvent.drawArraysIndirect.mode = LWN_DRAW_PRIMITIVE_TRIANGLES;
                    expectedEvent.drawArraysIndirect.indirectBuffer = imph->GetBufferAddress();
                    cbh->DrawArraysIndirect(DrawPrimitive::TRIANGLES, imph->GetBufferAddress());
                    break;
                case DISPATCH:
                    expectedEvent.type = LLGD_EVENT_TYPE_COMPUTE_DISPATCH;
                    expectedEvent.computeDispatch.groupsX = 1;
                    expectedEvent.computeDispatch.groupsY = 2;
                    expectedEvent.computeDispatch.groupsZ = 3;
                    cbh->DispatchCompute(1, 2, 3);
                    break;
                case DRAW_TRANSFORM_FEEDBACK:
                    expectedEvent.type = LLGD_EVENT_TYPE_DRAW_TRANSFORM_FEEDBACK;
                    expectedEvent.drawTransformFeedback.mode = LWN_DRAW_PRIMITIVE_TRIANGLES;
                    expectedEvent.drawTransformFeedback.buffer = imph->GetBufferAddress();
                    cbh->DrawTransformFeedback(DrawPrimitive::TRIANGLES, imph->GetBufferAddress());
                    break;
                case COPY_BUFFER_TO_BUFFER:
                    expectedEvent.type = LLGD_EVENT_TYPE_COPY_BUFFER_TO_BUFFER;
                    expectedEvent.copyBufferToBuffer.src = imph->GetBufferAddress();
                    expectedEvent.copyBufferToBuffer.dst = imph->GetBufferAddress() + 16;
                    expectedEvent.copyBufferToBuffer.size = 16;
                    cbh->CopyBufferToBuffer(imph->GetBufferAddress(), imph->GetBufferAddress() + 16, 16, 0);
                    break;
                case UPDATE_UNIFORM_BUFFER:
                {
                    static const size_t UPDATE_OFFSET = 32;
                    static const size_t UPDATE_SIZE = 4096;
                    char updateData[UPDATE_SIZE];
                    for (char& data : updateData) {
                        data = rand() % 256;
                    }
                    cbh->UpdateUniformBuffer(imph->GetBufferAddress(), imph->GetSize(), UPDATE_OFFSET, UPDATE_SIZE, updateData);

                    expectedEvent.type = LLGD_EVENT_TYPE_UPDATE_UNIFORM_BUFFER;
                    expectedEvent.updateUniformBuffer.buffer = imph->GetBufferAddress();
                    expectedEvent.updateUniformBuffer.alignedBufferSize = imph->GetSize();
                    expectedEvent.updateUniformBuffer.updateOffset = UPDATE_OFFSET;
                    expectedEvent.updateUniformBuffer.updateSize = UPDATE_SIZE;
                    break;
                }
                }

                expectedEvents.push_back(expectedEvent);
            }

            for (int j = 0; j < postStateChanges; ++j) {
                cbh->SetAlphaRef(1.0f);
            }
        }
        for (int i = 0; i < endStateChanges; ++i) {
            cbh->SetPointSize(1.0f);
        }
    }
    CommandHandle handle = cbh->EndRecording();

    // Reset events
    ResetSubmittedEvents();

    // Split the command set into segments, counting the events
    bool signaledEvent = false;
    bool waitingOnEvent = false;
    LlgdCommandSetProgress progress;

    static const uint32_t EVENT_PROCESSOR_SHADOW_STATE_MEM_SIZE = 1 * 1024;
    char eventProcessorShadowStateMem[EVENT_PROCESSOR_SHADOW_STATE_MEM_SIZE];

    llgdLwnSubmitCommandsSegmented(
        qh, // queue
        handle, // handles
        NullSubmitCommands, // submitCommandsFn,
        NullFinish, // finishFn,
        eventProcessorShadowStateMem, // shadowStateMem
        EVENT_PROCESSOR_SHADOW_STATE_MEM_SIZE, // shadowStateMemSize
        ProcessEvent, // processEventFn
        ProcessStateDeltaMethod, // process3dMethodFn
        PreSubmitEvent, // preSubmitEvent,
        PostSubmitEvent, // postSubmitEvent
        nullptr, // isBlockedOnWaitEvent
        nullptr,
        this, // userData
        0, // cpuEventIndex
        progress,
        signaledEvent,
        waitingOnEvent
        );

    // Test the events
    return VerifyEventCounts(expectedEventCount) && VerifySubmittedEvents(expectedEvents);
}

bool SubmitCommandsSegmentedValidator::DoStateDeltaTest(Event *events, int eventCount, const StateChangeFn& beginStateChangeFn)
{
    expectedToReceiveDeltas.clear();

    cbh->AddCommandMemory(mph, 0, SIZE);
    cbh->AddControlMemory(ctrl_space, SIZE);
    cbh->BeginRecording();
    {
        cbh->SetLineWidth(1.0f);

        if (beginStateChangeFn) {
            expectedToReceiveDeltas = beginStateChangeFn(cbh);
        }

        for (int k = 0; k < eventCount; ++k) {
            switch (events[k]) {
            case DRAW_ARRAYS:
                cbh->DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                break;
            case DRAW_ARRAYS_INDIRECT:
                cbh->DrawArraysIndirect(DrawPrimitive::TRIANGLES, imph->GetBufferAddress());
                break;
            case DISPATCH:
                cbh->DispatchCompute(1, 1, 1);
                break;
            case DRAW_TRANSFORM_FEEDBACK:
                cbh->DrawTransformFeedback(DrawPrimitive::TRIANGLES, imph->GetBufferAddress());
                break;
            case COPY_BUFFER_TO_BUFFER:
                cbh->CopyBufferToBuffer(imph->GetBufferAddress(), imph->GetBufferAddress() + 16, 16, 0);
                break;
            case UPDATE_UNIFORM_BUFFER:
            {
                static const size_t UPDATE_OFFSET = 32;
                static const size_t UPDATE_SIZE = 4096;
                char updateData[UPDATE_SIZE];
                for (char& data : updateData) {
                    data = rand() % 256;
                }
                cbh->UpdateUniformBuffer(imph->GetBufferAddress(), imph->GetSize(), UPDATE_OFFSET, UPDATE_SIZE, updateData);
                break;
            }
            }
        }

        cbh->SetPointSize(1.0f);
    }
    CommandHandle handle = cbh->EndRecording();

    // Reset received data
    receivedStateDeltas.clear();
    ResetSubmittedEvents();

    // Split the command set into segments, counting the events
    bool signaledEvent = false;
    bool waitingOnEvent = false;
    LlgdCommandSetProgress progress;

    static const uint32_t EVENT_PROCESSOR_SHADOW_STATE_MEM_SIZE = 1 * 1024;
    char eventProcessorShadowStateMem[EVENT_PROCESSOR_SHADOW_STATE_MEM_SIZE];

    llgdLwnSubmitCommandsSegmented(
        qh, // queue
        handle, // handles
        NullSubmitCommands, // submitCommandsFn,
        NullFinish, // finishFn,
        eventProcessorShadowStateMem, // shadowStateMem
        EVENT_PROCESSOR_SHADOW_STATE_MEM_SIZE, // shadowStateMemSize
        ProcessEvent, // processEventFn
        ProcessStateDeltaMethod, // process3dMethodFn
        PreSubmitEvent, // preSubmitEvent,
        PostSubmitEvent, // postSubmitEvent
        nullptr, // isBlockedOnWaitEvent
        nullptr,
        this, // userData
        0, // cpuEventIndex
        progress,
        signaledEvent,
        waitingOnEvent
    );

    return VerifyStateDeltas();
}

void SubmitCommandsSegmentedValidator::SubmitCommandHandle(LWNqueue* queue, void* userData)
{
    // Do nothing
}

void SubmitCommandsSegmentedValidator::ProcessEvent(const LlgdEvent& event, void* userData)
{
    SubmitCommandsSegmentedValidator* validator = reinterpret_cast<SubmitCommandsSegmentedValidator*>(userData);

    for (uint32_t i=0; i<ReceiveEventMaxCount; ++i) {
        if (int(validator->submittedEvents[i].type) == 0) {
            // Empty event. Insert here.
            validator->submittedEvents[i] = event;
            return;
        }
    }

    assert(false && "Exceeds receivable max num of events");
}

void SubmitCommandsSegmentedValidator::PreSubmitEvent(LWNqueue* queue, void* userData)
{
    ((SubmitCommandsSegmentedValidator*)userData)->DoPreSubmitEvent();
}

void SubmitCommandsSegmentedValidator::PostSubmitEvent(LWNqueue* queue, void* userData)
{
    ((SubmitCommandsSegmentedValidator*)userData)->DoPostSubmitEvent();
}

void SubmitCommandsSegmentedValidator::ProcessStateDeltaMethod(uint32_t method, uint32_t numValues, const uint32_t* values, void* userData)
{
    reinterpret_cast<SubmitCommandsSegmentedValidator*>(userData)->DoProcessStateDeltaMethod(method, numValues, values);
}

void SubmitCommandsSegmentedValidator::DoPreSubmitEvent()
{
    ++preSubmitEventCount;
}

void SubmitCommandsSegmentedValidator::DoPostSubmitEvent()
{
    ++postSubmitEventCount;
}

void SubmitCommandsSegmentedValidator::DoProcessStateDeltaMethod(uint32_t method, uint32_t numValues, const uint32_t* values)
{
    // TODO: (https://jirasw.lwpu.com/browse/LLGD-2628) Investigate why this leads to crash on HR24
    // unordered_map<uint32_t, vector<uint32_t>> receivedStateDeltas;
    // receivedStateDeltas[method] = std::vector<uint32_t>(values, values + numValues);
    assert(numValues <= StateDeltaValueMaxCount);
    for (uint32_t i = 0; i < StateDeltaValueMaxCount; ++i) {
        if (i < numValues) {
            receivedStateDeltas[method][i] = values[i];
        } else {
            receivedStateDeltas[method][i] = IlwalidStateDeltaValue;
        }
    }
}

void SubmitCommandsSegmentedValidator::ResetSubmittedEvents()
{
    preSubmitEventCount = 0;
    postSubmitEventCount = 0;
    LlgdEvent nullEvent;
    memset(&nullEvent, 0, sizeof(LlgdEvent));
    std::fill(submittedEvents.begin(), submittedEvents.end(), nullEvent);
}

bool SubmitCommandsSegmentedValidator::VerifyEventCounts(int expectedEventCount)
{
    return (preSubmitEventCount == expectedEventCount) && (postSubmitEventCount == expectedEventCount);
}

bool SubmitCommandsSegmentedValidator::VerifySubmittedEvents(const std::vector<LlgdEvent>& expectedEvents)
{
    for (size_t i=0; i<expectedEvents.size(); ++i) {
        const auto& expected = expectedEvents[i];
        const auto& submitted = submittedEvents[i];
        if (int(submitted.type) == 0) {
            return false;
        }
        if (expected.type != submitted.type) {
            return false;
        }

        switch (expected.type)
        {
        case LLGD_EVENT_TYPE_DRAW_ARRAYS:
        {
            const auto& realData = submitted.drawArrays;
            const auto& goldenData = expected.drawArrays;

            if (realData.mode != goldenData.mode ||
                realData.first != goldenData.first ||
                realData.count != goldenData.count) {
                    return false;
            }
            break;
        }
        case LLGD_EVENT_TYPE_DRAW_ARRAYS_INDIRECT:
        {
            const auto& realData = submitted.drawArraysIndirect;
            const auto& goldenData = expected.drawArraysIndirect;

            if (realData.mode != goldenData.mode ||
                realData.indirectBuffer != goldenData.indirectBuffer) {
                    return false;
            }
            break;
        }
        case LLGD_EVENT_TYPE_COMPUTE_DISPATCH:
        {
            const auto& realData = submitted.computeDispatch;
            const auto& goldenData = expected.computeDispatch;

            if (realData.groupsX != goldenData.groupsX ||
                realData.groupsY != goldenData.groupsY ||
                realData.groupsZ != goldenData.groupsZ) {
                    return false;
            }
            break;
        }
        case LLGD_EVENT_TYPE_DRAW_TRANSFORM_FEEDBACK:
        {
            const auto& realData = submitted.drawTransformFeedback;
            const auto& goldenData = expected.drawTransformFeedback;

            if (realData.mode != goldenData.mode ||
                realData.buffer != goldenData.buffer) {
                    return false;
            }
            break;
        }
        case LLGD_EVENT_TYPE_COPY_BUFFER_TO_BUFFER:
        {
            const auto& realData = submitted.copyBufferToBuffer;
            const auto& goldenData = expected.copyBufferToBuffer;

            if (realData.src != goldenData.src ||
                realData.dst != goldenData.dst ||
                realData.size != goldenData.size) {
                    return false;
            }
            break;
        }
        case LLGD_EVENT_TYPE_UPDATE_UNIFORM_BUFFER:
        {
            const auto& realData = submitted.updateUniformBuffer;
            const auto& goldenData = expected.updateUniformBuffer;

            if (realData.buffer != goldenData.buffer ||
                realData.alignedBufferSize != goldenData.alignedBufferSize ||
                realData.updateOffset != goldenData.updateOffset ||
                realData.updateSize != goldenData.updateSize) {
                    return false;
            }
            break;
        }
        default:
            assert(false && "Unsupported type in this test");
            return false;
            break;
        }
    }

    return true;
}

bool SubmitCommandsSegmentedValidator::VerifyStateDeltas()
{
    if (expectedToReceiveDeltas.empty())
    {
        return true;
    }

    // Expected deltas exist but no state deltas were received. => Test fails
    if (receivedStateDeltas.empty())
    {
        return false;
    }

    // Confirm if expected deltas are in the received deltas or not
    // Note that EXACT MATCH isn't required
    for (const auto& expectedDelta : expectedToReceiveDeltas)
    {
        const auto expectedMethod = expectedDelta.first;
        const auto& expectedValues = expectedDelta.second;

        const auto findItr = receivedStateDeltas.find(expectedMethod);
        if (findItr == receivedStateDeltas.end())
        {
            // Not found. Fail
            return false;
        }

        if (expectedValues != findItr->second)
        {
            // Values are not equal
            return false;
        }
    }

    return true;
}

bool SubmitCommandsSegmentedValidator::TestEvents()
{
    Event drawArrays[] = {
        DRAW_ARRAYS,
    };
    Event drawArraysIndirect[] = {
        DRAW_ARRAYS,
    };
    Event dispatch[] = {
        DISPATCH,
    };
    Event drawTransformFeedback[] {
        DRAW_TRANSFORM_FEEDBACK,
    };
    Event copyBuffer[] = {
        COPY_BUFFER_TO_BUFFER,
    };
    Event updateUniformBuffer[] = {
        UPDATE_UNIFORM_BUFFER,
    };
    Event indirectMix[] = {
        DRAW_ARRAYS_INDIRECT,
        DRAW_ARRAYS,
    };
    Event computeMix[] = {
        DISPATCH,
        DRAW_ARRAYS,
    };
    Event xfbMix[] = {
        DRAW_TRANSFORM_FEEDBACK,
        DRAW_ARRAYS,
    };
    Event bookendMix[] = {
        COPY_BUFFER_TO_BUFFER,
        DRAW_ARRAYS,
    };
    Event allMix[] = {
        DRAW_ARRAYS,
        DRAW_ARRAYS_INDIRECT,
        DRAW_TRANSFORM_FEEDBACK,
        DISPATCH,
        DRAW_ARRAYS,
        DRAW_ARRAYS_INDIRECT,
        DISPATCH,
        DRAW_TRANSFORM_FEEDBACK,
        COPY_BUFFER_TO_BUFFER,
    };
    Event* tests[] = {
        drawArrays,
        drawArraysIndirect,
        dispatch,
        drawTransformFeedback,
        copyBuffer,
        updateUniformBuffer,
        indirectMix,
        computeMix,
        xfbMix,
        bookendMix,
        allMix,
    };
    #define NUM_EVENTS(_x) int((sizeof(_x)) / sizeof(Event))
    int testCounts[] = {
        NUM_EVENTS(drawArrays),
        NUM_EVENTS(drawArraysIndirect),
        NUM_EVENTS(dispatch),
        NUM_EVENTS(drawTransformFeedback),
        NUM_EVENTS(copyBuffer),
        NUM_EVENTS(updateUniformBuffer),
        NUM_EVENTS(indirectMix),
        NUM_EVENTS(computeMix),
        NUM_EVENTS(xfbMix),
        NUM_EVENTS(bookendMix),
        NUM_EVENTS(allMix),
    };
    int testCount = int(sizeof(testCounts) / sizeof(int));


    for (int repeatCount = 1; repeatCount < 3; ++repeatCount) {
    for (int beginStateChanges = 0; beginStateChanges < 2; ++beginStateChanges) {
    for (int endStateChanges = 0; endStateChanges < 2; ++endStateChanges) {
    for (int preStateChanges = 0; preStateChanges < 2; ++preStateChanges) {
    for (int postStateChanges = 0; postStateChanges < 2; ++postStateChanges) {
    for (int test = 0; test < testCount; ++test) {
        const bool passed = DoEventTest(repeatCount, beginStateChanges, endStateChanges, preStateChanges, postStateChanges, tests[test], testCounts[test]);
        if (!passed) {
            return false;
        }
    }}}}}}

    return true;
}

enum CreateBindDataFlag
{
    FLAG_DEFAULT = 0,
    FLAG_HIGH_ALWAYS_ZERO = 1,
    FLAG_SIZE_NOT_REQUIRED = 2,
};

// Generate random address, its range and binding method data
static auto CreateBindExpectedData(int stage, int index, const uint32_t* lowMethodAddressTable, uint32_t wordsPerIndex, uint32_t createFlag = CreateBindDataFlag::FLAG_DEFAULT)
{
    bool highIsZero = !!(createFlag & FLAG_HIGH_ALWAYS_ZERO);

    uint32_t high = highIsZero ? 0 : uint32_t(rand() % 0x1000000) + 1;
    uint32_t low = uint32_t(rand() % 0x1000000) + 1;
    uint64_t address = (uint64_t(high) << 32) | low;
    uint32_t size = LlgdAlignUp((rand() % 0x1000) + 1, LWN_STORAGE_CLASS_BUFFER);

    const auto lowMethod = lowMethodAddressTable[stage] + index * wordsPerIndex;
    const auto highMethod = lowMethod + 1;
    const auto sizeMethod = lowMethod + 2;

    // TODO: (https://jirasw.lwpu.com/browse/LLGD-2628) Investigate why this leads to crash
    // unordered_map<uint32_t, vector<uint32_t>> expecteds; expecteds[METHOD_INDEX_TO_ADDR(lowMethod)] = { low };
    SubmitCommandsSegmentedValidator::StateDeltas expecteds;
    expecteds[METHOD_INDEX_TO_ADDR(lowMethod)].fill(IlwalidStateDeltaValue);
    expecteds[METHOD_INDEX_TO_ADDR(lowMethod)][0] = low;
    expecteds[METHOD_INDEX_TO_ADDR(highMethod)].fill(IlwalidStateDeltaValue);
    expecteds[METHOD_INDEX_TO_ADDR(highMethod)][0] = high;

    if (!(createFlag & FLAG_SIZE_NOT_REQUIRED))
    {
        expecteds[METHOD_INDEX_TO_ADDR(sizeMethod)].fill(IlwalidStateDeltaValue);
        expecteds[METHOD_INDEX_TO_ADDR(sizeMethod)][0] = size;
    }

    lwn::BufferRange buffer;
    buffer.address = address;
    buffer.size = size;

    return std::make_pair(buffer, expecteds);
}

// Generate random buffer data and its binding method data, then perform binding to command buffer by single bind API
template <typename T_BindFn>
static auto CreateSingleBindBufferStateChange(llgd_lwn::CommandBufferHolder& cbh, const uint32_t* lowMethodAddressTable, const T_BindFn& bindFn)
{
    const uint32_t wordsPerIndex = 3;

    // Bind by single bind API
    SubmitCommandsSegmentedValidator::StateDeltas expecteds;
    for (int stage = 0; stage <= LWN_SHADER_STAGE_COMPUTE; stage++) {
        int index = rand() % 8;
        auto bufferAndExpecteds = CreateBindExpectedData(stage, index, lowMethodAddressTable, wordsPerIndex);
        auto shaderStage = lwn::ShaderStage::Enum(stage);
        bindFn(cbh, shaderStage, index, bufferAndExpecteds.first);

        // unordered_map::merge is available from C++17. Use old style to merge
        for (const auto& newExpectedData : bufferAndExpecteds.second)
        {
            expecteds[newExpectedData.first] = newExpectedData.second;
        }
    }

    return expecteds;
}

// Generate random buffer data and its binding method data, then perform binding to command buffer by multiple bind API
template <typename T_BindFn>
static auto CreateMultipleBindBufferStateChange(llgd_lwn::CommandBufferHolder& cbh, const uint32_t* lowMethodAddressTable, const T_BindFn& bindFn)
{
    const uint32_t wordsPerIndex = 3;

    SubmitCommandsSegmentedValidator::StateDeltas expecteds;
    for (int stage = 0; stage <= LWN_SHADER_STAGE_COMPUTE; stage++) {
        int bindCount = 1 + rand() % 7;
        const int startIndex = 0;
        auto shaderStage = lwn::ShaderStage::Enum(stage);

        std::vector<lwn::BufferRange> buffers;
        for (int bindI = 0; bindI < bindCount; bindI++) {
            int index = startIndex + bindI;

            auto bufferAndExpecteds = CreateBindExpectedData(stage, index, lowMethodAddressTable, wordsPerIndex);
            buffers.push_back(bufferAndExpecteds.first);
            for (const auto& newExpectedData : bufferAndExpecteds.second)
            {
                expecteds[newExpectedData.first] = newExpectedData.second;
            }
        }

        bindFn(cbh, shaderStage, startIndex, buffers);
    }

    return expecteds;
}

// Generate random texture / image / sampler handle and its binding method data
template <typename T_LwnHandleType>
static auto CreateTextureOrImageBindExpectedData(int stage, int index, const uint32_t* lowMethodAddressTable)
{
    const uint32_t graphicsWordsPerIndex = 2;

    const auto data = CreateBindExpectedData(stage, index, lowMethodAddressTable, graphicsWordsPerIndex, CreateBindDataFlag::FLAG_HIGH_ALWAYS_ZERO | CreateBindDataFlag::FLAG_SIZE_NOT_REQUIRED);
    std::pair<T_LwnHandleType, SubmitCommandsSegmentedValidator::StateDeltas> colwertedData;
    memcpy(&colwertedData.first, &data.first.address, sizeof(uint64_t));
    colwertedData.second = data.second;
    return colwertedData;
}

// Generate random texture / image / sampler handle and its binding method data, then perform binding to command buffer by single bind API
template <typename T_LwnHandleType, typename T_BindFn>
static auto CreateSingleBindTextureOrImageStateChange(llgd_lwn::CommandBufferHolder& cbh, const uint32_t* lowMethodAddressTable, const T_BindFn& bindFn)
{
    // Bind by single bind API
    SubmitCommandsSegmentedValidator::StateDeltas expecteds;
    for (int stage = 0; stage <= LWN_SHADER_STAGE_COMPUTE; stage++) {
        int index = rand() % 8;
        auto handleAndExpecteds = CreateTextureOrImageBindExpectedData<T_LwnHandleType>(stage, index, lowMethodAddressTable);
        auto shaderStage = lwn::ShaderStage::Enum(stage);
        bindFn(cbh, shaderStage, index, handleAndExpecteds.first);
        for (const auto& newExpectedData : handleAndExpecteds.second)
        {
            expecteds[newExpectedData.first] = newExpectedData.second;
        }
    }

    return expecteds;
}

// Generate random texture / image / sampler handle and its binding method data, then perform binding to command buffer by multiple bind API
template <typename T_LwnHandleType, typename T_BindFn>
static auto CreateMultipleBindTextureOrImageStateChange(llgd_lwn::CommandBufferHolder& cbh, const uint32_t* lowMethodAddressTable, const T_BindFn& bindFn)
{
    SubmitCommandsSegmentedValidator::StateDeltas expecteds;
    for (int stage = 0; stage <= LWN_SHADER_STAGE_COMPUTE; stage++) {
        int bindCount = 1 + rand() % 7;
        const int startIndex = 0;
        auto shaderStage = lwn::ShaderStage::Enum(stage);

        std::vector<T_LwnHandleType> handles;
        for (int bindI = 0; bindI < bindCount; bindI++) {
            int index = startIndex + bindI;

            auto handleAndExpecteds = CreateTextureOrImageBindExpectedData<T_LwnHandleType>(stage, index, lowMethodAddressTable);
            handles.push_back(handleAndExpecteds.first);
            for (const auto& newExpectedData : handleAndExpecteds.second)
            {
                expecteds[newExpectedData.first] = newExpectedData.second;
            }
        }

        bindFn(cbh, shaderStage, startIndex, handles);
    }

    return expecteds;
}

bool SubmitCommandsSegmentedValidator::TestStateDeltas()
{
    Event allMix[] = {
        DRAW_ARRAYS,
        DRAW_ARRAYS_INDIRECT,
        DRAW_TRANSFORM_FEEDBACK,
        DISPATCH,
        DRAW_ARRAYS,
        DRAW_ARRAYS_INDIRECT,
        DISPATCH,
        DRAW_TRANSFORM_FEEDBACK,
        COPY_BUFFER_TO_BUFFER,
    };
    Event* tests[] = {
        allMix,
    };
    #define NUM_EVENTS(_x) int((sizeof(_x)) / sizeof(Event))
    int testCounts[] = {
        NUM_EVENTS(allMix),
    };
    int testCount = int(sizeof(testCounts) / sizeof(int));

    static const uint32_t ssboAddrLowBaseMethodTableOfNlwShaderStage[] = {
        LLGD_FAKE_METHOD_INDEX_VS_SSBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_FS_SSBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_GS_SSBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_TCS_SSBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_TES_SSBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_CS_SSBO_0_ADDR_LOW,
    };
    const uint32_t uboLowBaseMethodTable[] = {
        LLGD_FAKE_METHOD_INDEX_VS_UBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_FS_UBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_GS_UBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_TCS_UBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_TES_UBO_0_ADDR_LOW,
        LLGD_FAKE_METHOD_INDEX_CS_UBO_0_ADDR_LOW,
    };

    static const uint32_t textureLowBaseMethodTable[] = {
        LLGD_FAKE_METHOD_INDEX_VS_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_FS_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_GS_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TCS_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TES_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_CS_TEXTURE_0_LOW,
    };
    const uint32_t imageLowBaseMethodTable[] = {
        LLGD_FAKE_METHOD_INDEX_VS_IMAGE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_FS_IMAGE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_GS_IMAGE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TCS_IMAGE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TES_IMAGE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_CS_IMAGE_0_LOW,
    };
    const uint32_t separateTextureLowBaseMethodTable[] = {
        LLGD_FAKE_METHOD_INDEX_VS_SEPARATE_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_FS_SEPARATE_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_GS_SEPARATE_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TCS_SEPARATE_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TES_SEPARATE_TEXTURE_0_LOW,
        LLGD_FAKE_METHOD_INDEX_CS_SEPARATE_TEXTURE_0_LOW,
    };
    const uint32_t separateSamplersLowBaseMethodTable[] = {
        LLGD_FAKE_METHOD_INDEX_VS_SEPARATE_SAMPLER_0_LOW,
        LLGD_FAKE_METHOD_INDEX_FS_SEPARATE_SAMPLER_0_LOW,
        LLGD_FAKE_METHOD_INDEX_GS_SEPARATE_SAMPLER_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TCS_SEPARATE_SAMPLER_0_LOW,
        LLGD_FAKE_METHOD_INDEX_TES_SEPARATE_SAMPLER_0_LOW,
        LLGD_FAKE_METHOD_INDEX_CS_SEPARATE_SAMPLER_0_LOW,
    };

    std::vector<StateChangeFn> stateChangers = {
        nullptr,
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind SSBO by single bind API
            return CreateSingleBindBufferStateChange(cbh, ssboAddrLowBaseMethodTableOfNlwShaderStage,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int index, const lwn::BufferRange& buffer) {
                    cbh->BindStorageBuffer(shaderStage, index, buffer.address, buffer.size);
                });
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind SSBO by multiple bind API
            return CreateMultipleBindBufferStateChange(cbh, ssboAddrLowBaseMethodTableOfNlwShaderStage,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int startIndex, const std::vector<lwn::BufferRange>& buffers) {
                    cbh->BindStorageBuffers(shaderStage, startIndex, int(buffers.size()), buffers.data());
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind UBO by single bind API
            return CreateSingleBindBufferStateChange(cbh, uboLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int index, const lwn::BufferRange& buffer) {
                    cbh->BindUniformBuffer(shaderStage, index, buffer.address, buffer.size);
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind SSBO by multiple bind API
            return CreateMultipleBindBufferStateChange(cbh, uboLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int startIndex, const std::vector<lwn::BufferRange>& buffers) {
                    cbh->BindUniformBuffers(shaderStage, startIndex, int(buffers.size()), buffers.data());
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind Texture by single bind API
            return CreateSingleBindTextureOrImageStateChange<lwn::TextureHandle>(cbh, textureLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int index, lwn::TextureHandle handle) {
                    cbh->BindTexture(shaderStage, index, handle);
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind Textures by multiple bind API
            return CreateMultipleBindTextureOrImageStateChange<lwn::TextureHandle>(cbh, textureLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int startIndex, const std::vector<lwn::TextureHandle>& handles) {
                    cbh->BindTextures(shaderStage, startIndex, int(handles.size()), handles.data());
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind Image by single bind API
            return CreateSingleBindTextureOrImageStateChange<lwn::ImageHandle>(cbh, imageLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int index, lwn::ImageHandle handle) {
                    cbh->BindImage(shaderStage, index, handle);
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind Images by multiple bind API
            return CreateMultipleBindTextureOrImageStateChange<lwn::ImageHandle>(cbh, imageLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int startIndex, const std::vector<lwn::ImageHandle>& handles) {
                    cbh->BindImages(shaderStage, startIndex, int(handles.size()), handles.data());
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind SeparateTexture by single bind API
            return CreateSingleBindTextureOrImageStateChange<lwn::SeparateTextureHandle>(cbh, separateTextureLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int index, lwn::SeparateTextureHandle handle) {
                    cbh->BindSeparateTexture(shaderStage, index, handle);
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind SeparateTextures by multiple bind API
            return CreateMultipleBindTextureOrImageStateChange<lwn::SeparateTextureHandle>(cbh, separateTextureLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int startIndex, const std::vector<lwn::SeparateTextureHandle>& handles) {
                    cbh->BindSeparateTextures(shaderStage, startIndex, int(handles.size()), handles.data());
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind SeparateSampler by single bind API
            return CreateSingleBindTextureOrImageStateChange<lwn::SeparateSamplerHandle>(cbh, separateSamplersLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int index, lwn::SeparateSamplerHandle handle) {
                    cbh->BindSeparateSampler(shaderStage, index, handle);
                }
            );
        },
        [&](llgd_lwn::CommandBufferHolder& cbh) {
            // Bind SeparateSamplers by multiple bind API
            return CreateMultipleBindTextureOrImageStateChange<lwn::SeparateSamplerHandle>(cbh, separateSamplersLowBaseMethodTable,
                [](llgd_lwn::CommandBufferHolder& cbh, lwn::ShaderStage::Enum shaderStage, int startIndex, const std::vector<lwn::SeparateSamplerHandle>& handles) {
                    cbh->BindSeparateSamplers(shaderStage, startIndex, int(handles.size()), handles.data());
                }
            );
        },
    };

    for (const auto& stateChanger : stateChangers) {
    for (int test = 0; test < testCount; ++test) {
        const bool passed = DoStateDeltaTest(tests[test], testCounts[test], stateChanger);
        if (!passed) {
            return false;
        }
    }}

    return true;
}

LLGD_DEFINE_TEST(SubmitCommandsSegmentedEvents, UNIT,
LwError Execute()
{
    auto v = std::make_unique<SubmitCommandsSegmentedValidator>();
    v->Initialize();

    if (!v->TestEvents()) { return LwError_IlwalidState; }
    else                  { return LwSuccess;            }
}
);
LLGD_DEFINE_TEST(SubmitCommandsSegmentedStateDeltas, UNIT,
LwError Execute()
{
    auto v = std::make_unique<SubmitCommandsSegmentedValidator>();
    v->Initialize();

    if (!v->TestStateDeltas()) { return LwError_IlwalidState; }
    else                       { return LwSuccess;            }
}
);
