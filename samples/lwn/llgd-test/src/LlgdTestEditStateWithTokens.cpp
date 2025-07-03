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

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>

namespace {
static const float BLUE[4]{ 0, 0, 1, 1 };
static const float GREEN[4]{ 0, 1, 0, 1 };

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    llgd_lwn::QueueHolder qh;
    llgd_lwn::SyncHolder sync; // Used for FenceSync (create event token)

    GpuState defaultGpuState;
    LWNcommandHandle baseCommandHandle;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;

    static const int IDX_MAP_OFFSET = 10;

    const float defaultState = 0.84f;
    const float editedState = 0.75f;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    // Get defaultGpuState
    defaultGpuState = m_spCommandHelper->RunAndExtractGpuState(m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
        cbh->SetPointSize(defaultState);
    }));

    // Construct base command for editing
    sync.Initialize((Device *)g_device);
    auto bufferConstructor = [&](CommandBuffer *cbh) {
        // clear is event method
        // FenceSync is an event which create a token
        cbh->SetPointSize(defaultState);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        cbh->SetPointSize(defaultState);
        cbh->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event2 by token
        cbh->SetPointSize(defaultState);
        cbh->ClearColor(0, GREEN, ClearColorMask::RGBA); // Event3 method
    };

    baseCommandHandle = m_spCommandHelper->MakeHandle(bufferConstructor);
    TEST_NEQ(baseCommandHandle, 0);

    return true;
}

bool Validator::Test()
{
    // Edit
    m_spCommandHelper->ResetPointersForEditingCB();
    auto editedHandle = llgdCommandSetEditRasterStateRasterizationPointSize(
        baseCommandHandle,
        [](uint32_t gpfifoOrderIdx, void*) { return gpfifoOrderIdx + IDX_MAP_OFFSET; },
        0, 1000,  // Edit everything and don't insert start/end edge patch
        defaultGpuState, defaultGpuState,
        editedState,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        llgd_lwn::EmptyMethodUpdatedFn,
        m_spCommandHelper.get());

    // Run edited command
    m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

    // Extract state value after exelwted
    const auto raster = m_spCommandHelper->ExtractRasterState();
    const auto editedActualValue = raster.rasterization.pointSize.value;
    TEST_EQ(editedActualValue, editedState);

    return true;
}

LLGD_DEFINE_TEST(EditStateWithToken, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
