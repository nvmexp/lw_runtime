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
    bool TestBarrier(bool tokens, uint32_t index);

private:
    llgd_lwn::QueueHolder qh;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    return true;
}

bool Validator::TestBarrier(bool tokens, uint32_t insert)
{
    Sync sync;
    (void)sync; // Not used when not using token in CB
    const auto condition = SyncCondition::ALL_GPU_COMMANDS_COMPLETE;
    const auto handle = m_spCommandHelper->MakeHandle([&] (CommandBuffer* cb) {
        if (tokens) cb->FenceSync(&sync, condition, 0);
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
        if (tokens) cb->FenceSync(&sync, condition, 0);
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
        if (tokens) cb->FenceSync(&sync, condition, 0);
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
        if (tokens) cb->FenceSync(&sync, condition, 0);
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
        if (tokens) cb->FenceSync(&sync, condition, 0);
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
    });

    m_spCommandHelper->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetInsertBarrier(
        handle,
        [](uint32_t index, void*) { return 10 + index; },
        insert,
        ~0 /* ALL BARRIER BITS! */,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        m_spCommandHelper.get());

    const auto decoded = m_spCommandHelper->MakeCommandHandleRunnable(patched);
    m_spCommandHelper->Run(decoded);

    return true;
}

bool Validator::Test()
{
    for (int t = 0; t < 2; ++t) {
    for (uint32_t i = 9; i < 22; ++i) {
        TEST(TestBarrier(t == 0, i))
    }}
    return true;
}

LLGD_DEFINE_TEST(EditBarrierInsert, UNIT,
LwError Execute()
{
    Validator v{};
    return (v.Initialize() && v.Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
