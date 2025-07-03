/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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
#include <vector>

namespace {
static const float GREEN[4]{ 0, 1, 0, 1 };
static const uint32_t COMMAND_BUFFER_EVENT_INDEX_OFFSET = 10;

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool TestPatchCommandBuffer(LWNcommandHandle targetHandle, uint32_t beginIndex, uint32_t endIndex, uint32_t expectedIncreasingMethodCount);
    void TrackMethodInfos(LWNcommandHandle handle, std::vector<LlgdCommandSetMethodTrackerMethodInfo>& methodInfos);

private:
    llgd_lwn::QueueHolder qh;
    llgd_lwn::SyncHolder sync; // Used for FenceSync (create event token)

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;
    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelperForMethodTracking;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    m_spCommandHelperForMethodTracking = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelperForMethodTracking->Initialize());

    return true;
}

bool Validator::TestPatchCommandBuffer(LWNcommandHandle handle, uint32_t beginIndex, uint32_t endIndex, uint32_t expectedIncreasingMethodCount)
{
    // Get initial methods
    const auto methodInfosInit = llgd_lwn::GetMethods(handle);

    static uint32_t s_beginIndex = 0;
    static uint32_t s_endIndex = 0;
    s_beginIndex = beginIndex;
    s_endIndex = endIndex;

    // Patching
    m_spCommandHelper->ResetPointersForEditingCB();
    const auto patched = llgdPatchCommandSetForShaderProfiling(
        handle,
        // Patch position decider
        [](uint32_t gpfifoOrderIndex, LlgdProfilePatchLocation location, void* callbackData) {
            const auto index = gpfifoOrderIndex + COMMAND_BUFFER_EVENT_INDEX_OFFSET;
            if (index == s_beginIndex && location == LlgdProfilePatchLocation::StartEvent) {
                return LlgdProfilePatchCommand{ nullptr, LlgdProfilePatchOperation::Push };
            }
            if (index == s_endIndex && location == LlgdProfilePatchLocation::EndEvent) {
                return LlgdProfilePatchCommand{ nullptr, LlgdProfilePatchOperation::Pop };
            }
            return LlgdProfilePatchCommand{ nullptr, LlgdProfilePatchOperation::Nop };
        },
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        m_spCommandHelper.get());

    const auto methodInfosActual = llgd_lwn::GetMethods(patched);

    TEST_EQ(methodInfosActual.size(), methodInfosInit.size() + expectedIncreasingMethodCount);

    return true;
}

bool Validator::Test()
{
    // Patch target
    const auto handle = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA); // Event1 method
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event2 by token
        cb->Barrier(~0); // Event3 barrier
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA); // Event4 method
    });

    // Patch normally
    TEST(TestPatchCommandBuffer(handle, COMMAND_BUFFER_EVENT_INDEX_OFFSET + 2, COMMAND_BUFFER_EVENT_INDEX_OFFSET + 4, 3 /*Start, Stop, Wait are expected to be inserted*/));

    // Patch out of command handle
    TEST(TestPatchCommandBuffer(handle, COMMAND_BUFFER_EVENT_INDEX_OFFSET + 6, COMMAND_BUFFER_EVENT_INDEX_OFFSET + 7, 0));

    // Start is inside of CB but end is out of CB
    TEST(TestPatchCommandBuffer(handle, COMMAND_BUFFER_EVENT_INDEX_OFFSET + 2, COMMAND_BUFFER_EVENT_INDEX_OFFSET + 7, 1 /*Start is expected to be inserted*/));

    // End is inside of CB but start is out of CB
    TEST(TestPatchCommandBuffer(handle, COMMAND_BUFFER_EVENT_INDEX_OFFSET - 1, COMMAND_BUFFER_EVENT_INDEX_OFFSET + 3, 2 /*Stop and Wait are expected to be inserted*/));

    return true;
}
}

LLGD_DEFINE_TEST(ShaderProfilerCommandPatcher, UNIT, LwError Execute() {
    Validator v{};
    return (v.Initialize() && v.Test()) ? LwSuccess : LwError_IlwalidState;
});
