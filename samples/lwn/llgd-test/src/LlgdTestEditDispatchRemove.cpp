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

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>

#include <nn/os.h>

namespace {
class Validator {
public:
    bool Initialize();
    bool Test();

    bool TestDispatchRemove(uint32_t removeIndex, DispatchEventType eventType, bool expectedToBeRemoved);
private:
    llgd_lwn::QueueHolder qh;
    llgd_lwn::SyncHolder sync; // Used for FenceSync (create event token)

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    return true;
}

bool Validator::TestDispatchRemove(uint32_t removeIndex, DispatchEventType eventType, bool expectedToBeRemoved /* TODO cleanup */)
{
    // Create command buffers, one contains dispatch, one doesn't contain dispatch

    // CommandBuffer without a dispatch method
    m_spCommandHelper->ResetPointersForEditingCB();
    const auto handleNoDispatch = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
        cb->Barrier(~0);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
    });
    uint64_t eventCountWithoutDispatch = llgdLwnGetEventCount(handleNoDispatch);

    // CommandBuffer with a dispatch method
    m_spCommandHelper->ResetPointersForEditingCB();
    const LWNbufferAddress randomIndirectAddress = 0xe38130dc1000;
    const auto handle = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
        cb->Barrier(~0); // Event1 method
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event2 by token
        switch (eventType)
        {
        case DispatchEventType::Dispatch:
            cb->DispatchCompute(2, 4, 6);
            break;
        case DispatchEventType::DispatchIndirect:
            cb->DispatchComputeIndirect(randomIndirectAddress);
            break;
        default:
            break;
        }
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event4 by token
    });
    uint64_t eventCountWithDispatch = llgdLwnGetEventCount(handle);

    // Execute removing
    m_spCommandHelper->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetRemoveDispatch(
        handle,
        [](uint32_t index, void*) { return 10 + index; },
        removeIndex,
        eventType,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        m_spCommandHelper.get());

    uint64_t eventCountAfterRemove = llgdLwnGetEventCount(patched);

    if (expectedToBeRemoved)
    {
        // Dispatch event is represented as one token. Thus, if correctly dispatch was removed, only one token event was removed.
        return eventCountAfterRemove == eventCountWithoutDispatch && eventCountAfterRemove + 1 == eventCountWithDispatch;
    }
    else
    {
        return eventCountAfterRemove == eventCountWithDispatch;
    }
}


bool Validator::Test()
{
    // Remove a regular dispatch
    TEST(TestDispatchRemove(13, DispatchEventType::Dispatch, true /*removeExpected*/));

    // Remove a dispatch indirect
    TEST(TestDispatchRemove(13, DispatchEventType::DispatchIndirect, true /*removeExpected*/));

    return true;
}
}

LLGD_DEFINE_TEST(EditDispatchRemove, UNIT, LwError Execute() {
    Validator v{};
    return (v.Initialize() && v.Test()) ? LwSuccess : LwError_IlwalidState;
});
