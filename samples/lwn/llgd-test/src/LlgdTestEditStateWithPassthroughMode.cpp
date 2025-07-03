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

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool InitializeBaseCommand();

private:
    llgd_lwn::QueueHolder qh;

    GpuState defaultGpuState;
    LWNcommandHandle baseCommandHandle;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;

    static const int numberOfSetStencilMask = 2;
    static const int IDX_MAP_OFFSET = 10;
    // StencilMask state
    const uint32_t defaultState = 0x3F;
    const uint32_t editedState = 0xDE;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    return InitializeBaseCommand();
}

bool Validator::InitializeBaseCommand()
{
    // Get defaultGpuState
    defaultGpuState = m_spCommandHelper->RunAndExtractGpuState(m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
        cbh->SetStencilMask(lwn::Face::FRONT, defaultState);
    }));

    // Construct base command for editing
    auto bufferConstructor = [&](CommandBuffer *cbh) {
        // ClearDepthStencil is event method using passthrough mode.
        // In addition, SetStencilMask method is used during its passthrough mode. Those SetStencilMask must not be edited.
        cbh->SetStencilMask(lwn::Face::FRONT, defaultState);
        cbh->ClearDepthStencil(0.1f, true, 0x02, 0x1D);; // Event1 method
        cbh->SetStencilMask(lwn::Face::FRONT, defaultState);
    };
    baseCommandHandle = m_spCommandHelper->MakeHandle(bufferConstructor);
    TEST_NEQ(baseCommandHandle, 0);

    return true;
}

bool Validator::Test()
{
    static int updatedFnCalledCount = 0;
    updatedFnCalledCount = 0;
    auto methodUpdatedFn = [](uint32_t eventIndex, uint32_t method, uint32_t count, const uint32_t* values, void* callback) {
        updatedFnCalledCount++;
    };

    // Edit
    m_spCommandHelper->ResetPointersForEditingCB();
    auto editedHandle = llgdCommandSetEditPixelStateStencilMask(
        baseCommandHandle,
        [](uint32_t gpfifoOrderIdx, void*) { return gpfifoOrderIdx + IDX_MAP_OFFSET; },
        0, 1000,  // Edit everything, and don't insert start/end edge patch
        defaultGpuState, defaultGpuState, LWN_FACE_FRONT, editedState,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        methodUpdatedFn,
        m_spCommandHelper.get()
    );

    // Run edited command
    m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

    // 1. Test state is correctly edited
    const auto ps = m_spCommandHelper->ExtractPixelState();
    const auto actualEditedState = ps.stencil.front.mask.value;
    TEST_EQ(actualEditedState, editedState);

    // 2. Test updates were exelwted only for explicit SetStencilMask, not exelwted for SetStencilMask inside of ClearDepthStencil
    TEST_EQ(updatedFnCalledCount, numberOfSetStencilMask);

    return true;
}

LLGD_DEFINE_TEST(EditStateWithPassthrough, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
