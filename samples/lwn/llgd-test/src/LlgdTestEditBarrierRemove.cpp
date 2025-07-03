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

namespace {
bool Test()
{
    llgd_lwn::QueueHolder qh;
    llgd_lwn::SyncHolder sync; // Used for FenceSync (create event token)

    qh.Initialize(g_device);

    auto cmd = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(cmd->Initialize());

    static const float GREEN[4]{ 0, 1, 0, 1 };
    const auto barred = cmd->MakeHandle([&] (CommandBuffer* cb) {
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        cb->Barrier(~0);
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
    });

    cmd->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetRemoveBarrier(
        barred,
        [](uint32_t index, void*) { return 10 + index; }, 13,
        cmd->WriteControlMemoryForEditing,
        cmd->WriteCommandMemoryForEditing,
        cmd.get());

    const auto decoded = cmd->MakeCommandHandleRunnable(patched);
    cmd->Run(decoded); // don't crash
    const auto patchedMethods = llgd_lwn::GetMethods(decoded);

    const auto reference = cmd->MakeHandle([&](CommandBuffer* cb) {
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        cb->ClearColor(0, GREEN, ClearColorMask::RGBA);
    });
    const auto referenceMethods = llgd_lwn::GetMethods(reference);

    TEST(llgd_lwn::CompareMethods(patchedMethods, referenceMethods));
    return true;
}
}

LLGD_DEFINE_TEST(EditBarrierRemove, UNIT, LwError Execute() { return Test() ? LwSuccess : LwError_IlwalidState; });
