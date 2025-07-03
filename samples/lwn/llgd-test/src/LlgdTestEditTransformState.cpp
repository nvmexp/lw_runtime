/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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
#include <LlgdTestUtilEditState.h>

#include <array>
#include <random>
#include <vector>

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>
#include <lwn_DeviceConstantsNX.h>

#include <nn/os.h>

namespace {
class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool TestEditViewport();
    bool TestEditDepthRange();
    bool TestEditSwizzle();
    bool TestEditScissors();

    bool TestViewport(uint32_t start, uint32_t end, int index, int coord);
    float ToViewport(const GpuState& state, int index, int coord);

private:
    llgd_lwn::QueueHolder qh;
    llgd_lwn::MemoryPoolHolder mph;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> cmd;

    lwn::Event semaphore;

private:
    static const size_t PAGE = LWN_DEVICE_INFO_CONSTANT_NX_MEMORY_POOL_PAGE_SIZE;
    uint8_t pool[PAGE] __attribute__((aligned(4096)));
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    cmd = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(cmd->Initialize());

    {
        MemoryPoolBuilder pool_builder;
        pool_builder.SetDevice(g_device).SetDefaults()
            .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_UNCACHED)
            .SetStorage(pool, PAGE);
        TEST(mph.Initialize(&pool_builder))
    }

    {
        EventBuilder builder;
        builder.SetDefaults()
               .SetStorage(mph, 0);
        TEST(semaphore.Initialize(&builder))
    }

    return true;
}

template <int N>
static constexpr std::array<float, N> Range(int start) noexcept
{
    std::array<float, N> res;
    for (int i = 0; i < N; ++i) {
        res[i] = i + start;
    }
    return res;
}

float Validator::ToViewport(const GpuState& state, int index, int coord)
{
    TransformState transform;
    llgdLwnExtractTransformState(state, transform);

    const auto& val = transform.viewportsAndScissors[index].viewport.value;
    const float xywh[]{ val.x, val.y, val.w, val.h };
    return xywh[coord];
}

// --------------------------------------------------------------------
// TestEditViewport
// --------------------------------------------------------------------
bool Validator::TestViewport(uint32_t start, uint32_t end, int index, int coord)
{
    static const int startOffset = 1; // Never gets shown, always patched
    static const int endOffset = 22200;
    static const int defaultOffset = 330;
    static const int patchOffset = 400;

    static const int COUNT = 5;
    static const auto RECT0 = Range<4 * COUNT>(startOffset);
    static const auto RECT1 = Range<4 * COUNT>(endOffset);
    static const auto RECT2 = Range<4 * COUNT>(defaultOffset);

    // Prepare
    cmd->Run(cmd->MakeHandle([&] (CommandBuffer* cb) {
        cb->SetViewports(0, COUNT, &RECT0[0]);
    }));
    const GpuState startState = llgd_lwn::ExtractGpuState(qh);

    cmd->Run(cmd->MakeHandle([&] (CommandBuffer* cb) {
        cb->SetViewports(0, COUNT, &RECT1[0]);
    }));
    const GpuState endState = llgd_lwn::ExtractGpuState(qh);

    // MakeHandle reuses Pool space, so our handle need to be created
    // after that r-value handle above is Run.
    const auto handle = cmd->MakeHandle([&] (CommandBuffer* cb) {
/* S1 */ cb->SetViewports(0, COUNT, &RECT2[0]);
/*  1 */ cb->Barrier(~0);
/*  2 */ cb->SignalEvent(&semaphore, EventSignalMode::WRITE, EventSignalLocation::BOTTOM, 0, 1);
/*  M */ // Measure here, just before ev 3 + 10.
/*  3 */ cb->WaitEvent(&semaphore, EventWaitMode::EQUAL, 2);
/* S4 */ cb->SetViewports(0, COUNT, &RECT2[0]);
/*  4 */ cb->Barrier(~0);
/*  5 */ cb->SignalEvent(&semaphore, EventSignalMode::WRITE, EventSignalLocation::BOTTOM, 0, 3);
/*  M */ // Measure here, just before ev 6 + 10.
/*  6 */ cb->WaitEvent(&semaphore, EventWaitMode::EQUAL, 4);
/* S7 */ cb->SetViewports(0, COUNT, &RECT2[0]);
/*  7 */ cb->Barrier(~0);
/*  8 */ cb->SignalEvent(&semaphore, EventSignalMode::WRITE, EventSignalLocation::BOTTOM, 0, 5);
    });

    // Edit
    cmd->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetEditTransformStateViewport(
        handle,
        [](uint32_t index, void*) { return 10 + index; },
        start,
        end,
        startState,
        endState,
        index,  patchOffset + 4 * index + 0.0f, patchOffset + 4 * index + 1.0f, patchOffset + 4 * index + 2.0f, patchOffset + 4 * index + 3.0f,
        cmd->WriteControlMemoryForEditing,
        cmd->WriteCommandMemoryForEditing,
        llgd_lwn::EmptyMethodUpdatedFn,
        cmd.get());
    const auto decoded = cmd->MakeCommandHandleRunnable(patched);

    // Measure
    semaphore.Signal(EventSignalMode::WRITE, 0);
    qh->SubmitCommands(1, &decoded);
    qh->Flush();

    const float offset = 4 * index + coord;
    static const uint32_t MEASURE_POINTS = 2;
    std::array<float, MEASURE_POINTS> ports{{ 0.0 }};
    for (uint32_t i = 0; i < MEASURE_POINTS; ++i) {
        while ((i * 2 + 1) != semaphore.GetValue()) {
            nn::os::SleepThread(nn::TimeSpan::FromMilliSeconds(1));
        }

        const auto state = llgd_lwn::ExtractGpuState(qh);
        ports[i] = ToViewport(state, index, coord) - offset;

        semaphore.Signal(EventSignalMode::WRITE, (i * 2 + 2));
    }

    // Check
    static const std::array<uint32_t, 2> measure_points{{ 13, 16 }}; // where we measure
    static const std::array<uint32_t, 2> viewport_points{{ 11, 14 }}; // where we set

    for (uint32_t i = 0; i < MEASURE_POINTS; ++i) {
        const auto point = measure_points[i];
        if (start <= point && point <= end) {
            // So here, we are in the range
            TEST_EQ(ports[i], float{ patchOffset })
        } else {
            // The point where we measure is outside the patch range
            if (end < point && end >= viewport_points[i]) {
                // The end patch is not overridden by a viewport set
                TEST_EQ(ports[i], float{ endOffset })
            } else {
                TEST_EQ(ports[i], float{ defaultOffset })
            }
        }
    }

    qh->Finish();
    return true;
}
bool Validator::TestEditViewport()
{
    static const int START_EV = 11; // First ev is 1 + 10
    static const int END_EV = 19; // Last is 8 + 10
    static const int START = START_EV - 1; // Start testing just before
#define DETERMINISTIC 0
#if DETERMINISTIC
    static const int END = END_EV + 1; // End testing just after
    for (int start = START; start < END_EV; ++start) {
    for (int end = START_EV; end < END; ++end) {
    for (int index = 0; index < 2; ++index) {
    for (int coord = 0; coord < 4; ++coord) {
        if (start > end) continue;
        TEST_FMT(TestViewport(start, end, index, coord), "Failed on start=%d, end=%d, index=%d, coord=%d", start, end, index, coord);
    }}}}
#else
    std::random_device rd;
    std::mt19937 mt;
    std::uniform_int_distribution<uint64_t> dist;
    for (int start = START; start < END_EV; ++start) {
        const int end = std::max(start, START_EV) + dist(mt) % (END_EV - start);
        const int index = dist(mt) % 2;
        const int coord = dist(mt) % 4;
        TEST_FMT(TestViewport(start, end, index, coord), "Failed on start=%d, end=%d, index=%d, coord=%d", start, end, index, coord);
    }

    TEST(TestViewport(12, 13, 1, 1))
    TEST(TestViewport(13, 14, 1, 1))
#endif

    return true;
}

// --------------------------------------------------------------------
// Note: Inserting edge patch test is done enough on TestEditViewport,
//       thus remaining tests check only simple editing.
// --------------------------------------------------------------------
static const float BLUE[4]{ 0, 0, 1, 1 };
static const uint32_t IDX_OFFSET = 10;
static const auto indexMapFn = [](uint32_t idx, void*) { return idx + IDX_OFFSET; };

// --------------------------------------------------------------------
// TestEditDepthRange
// --------------------------------------------------------------------
bool Validator::TestEditDepthRange()
{
    static const float defaultStateNear = 0.54f;
    static const float editedStateNear = 0.325f;
    static const float defaultStateFar = 210.f;
    static const float editedStateFar = 545.6f;

    auto setupDefaultState = [](CommandBuffer* cbh) {
        float ranges[TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS * 2];
        for (uint32_t i = 0; i < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++i) {
            ranges[i * 2] = defaultStateNear;
            ranges[i * 2 + 1] = defaultStateFar;
        }
        cbh->SetDepthRanges(0, TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS, ranges);
    };

    // Test for all viewports
    for (uint32_t viewport = 0; viewport < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++viewport) {
        // Create basecommand
        auto baseCommand = cmd->MakeHandle([&](CommandBuffer* cbh) {
            setupDefaultState(cbh);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        // Extract the baseCommand gpuState (use as start/end state)
        const auto gpuState = cmd->RunAndExtractGpuState(baseCommand);
        // Verify default state
        auto st = cmd->ExtractTransformState(&gpuState);
        TEST_ALMOST_EQ(st.viewportsAndScissors[viewport].depthRange.value.n, defaultStateNear);
        TEST_ALMOST_EQ(st.viewportsAndScissors[viewport].depthRange.value.f, defaultStateFar);

        // Edit
        cmd->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditTransformStateDepthRange(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            viewport, editedStateNear, editedStateFar,
            cmd->WriteControlMemoryForEditing,
            cmd->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            cmd.get()
        );
        // Execute
        cmd->Run(cmd->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = cmd->ExtractTransformState();
        for (uint32_t compareViewport = 0; compareViewport < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++compareViewport) {
            const auto& expectedNear = compareViewport == viewport ? editedStateNear : defaultStateNear;
            const auto& expectedFar = compareViewport == viewport ? editedStateFar : defaultStateFar;
            const auto actualNear = st.viewportsAndScissors[compareViewport].depthRange.value.n;
            const auto actualFar = st.viewportsAndScissors[compareViewport].depthRange.value.f;
            TEST_ALMOST_EQ(actualNear, expectedNear);
            TEST_ALMOST_EQ(actualFar, expectedFar);
        }
    }

    return true;
}

// --------------------------------------------------------------------
// TestEditSwizzle
// --------------------------------------------------------------------
bool Validator::TestEditSwizzle()
{
    using Swizzle = TransformState::ViewportAndScissor::Swizzle;
    static const Swizzle defaultState { LWN_VIEWPORT_SWIZZLE_POSITIVE_X, LWN_VIEWPORT_SWIZZLE_NEGATIVE_Y, LWN_VIEWPORT_SWIZZLE_POSITIVE_Z, LWN_VIEWPORT_SWIZZLE_NEGATIVE_W };
    static const Swizzle editedState { LWN_VIEWPORT_SWIZZLE_NEGATIVE_X, LWN_VIEWPORT_SWIZZLE_POSITIVE_Y, LWN_VIEWPORT_SWIZZLE_NEGATIVE_Z, LWN_VIEWPORT_SWIZZLE_POSITIVE_W };

    auto setupDefaultState = [](CommandBuffer* cbh) {
        ViewportSwizzle swizzles[TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS * 4];
        for (uint32_t i = 0; i < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++i) {
            swizzles[i * 4 + 0] = ViewportSwizzle::Enum(defaultState.x);
            swizzles[i * 4 + 1] = ViewportSwizzle::Enum(defaultState.y);
            swizzles[i * 4 + 2] = ViewportSwizzle::Enum(defaultState.z);
            swizzles[i * 4 + 3] = ViewportSwizzle::Enum(defaultState.w);
        }
        cbh->SetViewportSwizzles(0, TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS, swizzles);
    };

    // Test for all viewports
    for (uint32_t viewport = 0; viewport < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++viewport) {
        // Create basecommand
        auto baseCommand = cmd->MakeHandle([&](CommandBuffer* cbh) {
            setupDefaultState(cbh);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        // Extract the baseCommand gpuState (use as start/end state)
        const auto gpuState = cmd->RunAndExtractGpuState(baseCommand);
        // Verify default state
        auto st = cmd->ExtractTransformState(&gpuState);
        TEST_EQ(st.viewportsAndScissors[viewport].viewportSwizzle.value.x, defaultState.x);
        TEST_EQ(st.viewportsAndScissors[viewport].viewportSwizzle.value.y, defaultState.y);
        TEST_EQ(st.viewportsAndScissors[viewport].viewportSwizzle.value.z, defaultState.z);
        TEST_EQ(st.viewportsAndScissors[viewport].viewportSwizzle.value.w, defaultState.w);

        // Edit
        cmd->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditTransformStateSwizzle(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            viewport, editedState.x, editedState.y, editedState.z, editedState.w,
            cmd->WriteControlMemoryForEditing,
            cmd->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            cmd.get()
        );
        // Execute
        cmd->Run(cmd->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = cmd->ExtractTransformState();
        for (uint32_t compareViewport = 0; compareViewport < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++compareViewport) {
            const auto& expected = compareViewport == viewport ? editedState : defaultState;
            const auto& actual = st.viewportsAndScissors[compareViewport].viewportSwizzle.value;
            TEST_EQ(actual.x, expected.x);
            TEST_EQ(actual.y, expected.y);
            TEST_EQ(actual.z, expected.z);
            TEST_EQ(actual.w, expected.w);
        }
    }

    return true;
}

// --------------------------------------------------------------------
// TestEditScissors
// --------------------------------------------------------------------
bool Validator::TestEditScissors()
{
    using Scissor = TransformState::ViewportAndScissor::Scissor;
    static const Scissor defaultState{ 1, 0, 3, 2 };
    static const Scissor editedState{ 0, 3, 4, 1 };

    auto setupDefaultState = [](CommandBuffer* cbh) {
        int scissors[TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS * 4];
        for (uint32_t i = 0; i < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++i) {
            scissors[i * 4 + 0] = defaultState.x;
            scissors[i * 4 + 1] = defaultState.y;
            scissors[i * 4 + 2] = defaultState.w;
            scissors[i * 4 + 3] = defaultState.h;
        }
        cbh->SetScissors(0, TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS, scissors);
    };

    // Test for all viewports
    for (uint32_t viewport = 0; viewport < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++viewport) {
        // Create basecommand
        auto baseCommand = cmd->MakeHandle([&](CommandBuffer* cbh) {
            setupDefaultState(cbh);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        // Extract the baseCommand gpuState (use as start/end state)
        const auto gpuState = cmd->RunAndExtractGpuState(baseCommand);
        // Verify default state
        auto st = cmd->ExtractTransformState(&gpuState);
        TEST_EQ(st.viewportsAndScissors[viewport].scissor.value.x, defaultState.x);
        TEST_EQ(st.viewportsAndScissors[viewport].scissor.value.y, defaultState.y);
        TEST_EQ(st.viewportsAndScissors[viewport].scissor.value.w, defaultState.w);
        TEST_EQ(st.viewportsAndScissors[viewport].scissor.value.h, defaultState.h);

        // Edit
        cmd->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditTransformStateScissors(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            viewport, editedState.x, editedState.y, editedState.w, editedState.h,
            cmd->WriteControlMemoryForEditing,
            cmd->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            cmd.get()
        );
        // Execute
        cmd->Run(cmd->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = cmd->ExtractTransformState();
        for (uint32_t compareViewport = 0; compareViewport < TransformState::NUM_VIEWPORTS_AND_SCISSOR_RECTS; ++compareViewport) {
            const auto& expected = compareViewport == viewport ? editedState : defaultState;
            const auto& actual = st.viewportsAndScissors[compareViewport].scissor.value;
            TEST_EQ(actual.x, expected.x);
            TEST_EQ(actual.y, expected.y);
            TEST_EQ(actual.w, expected.w);
            TEST_EQ(actual.h, expected.h);
        }
    }

    return true;
}

bool Validator::Test()
{
    TEST(TestEditViewport());
    TEST(TestEditDepthRange());
    TEST(TestEditSwizzle());
    TEST(TestEditScissors());

    return true;
}
}

LLGD_DEFINE_TEST(EditTransformState, UNIT, LwError Execute() {
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
});
