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
#include <vector>

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>

namespace {
static const float GREEN[4]{ 0, 1, 0, 1 };

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool TestBarrierEditFlags(uint32_t eventIndex, int flags);

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

bool Validator::TestBarrierEditFlags(uint32_t eventIndex, int flags)
{
    const auto handle = m_spCommandHelper->MakeHandle([&] (CommandBuffer* cb) {
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA); // Event1 method
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event2 by token
        cb->Barrier(~0); // Event3 barrier
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA); // Event4 method
    });

    m_spCommandHelper->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetEditBarrier(
        handle,
        [](uint32_t index, void*) { return 10 + index; },
        eventIndex,
        flags,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        m_spCommandHelper.get());

    const auto decoded = m_spCommandHelper->MakeCommandHandleRunnable(patched);
    m_spCommandHelper->Run(decoded);

    return true;
}

bool Validator::Test()
{
    int newFlags = 0xABCD;
    for (uint32_t e = 10; e <= 15; ++e) {
        TEST(TestBarrierEditFlags(e, newFlags));
    }
    return true;
}

LLGD_DEFINE_TEST(EditBarrierEditFlags, UNIT,
LwError Execute()
{
    Validator v{};
    return (v.Initialize() && v.Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
