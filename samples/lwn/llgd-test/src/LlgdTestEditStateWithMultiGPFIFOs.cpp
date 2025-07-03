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
    llgd_lwn::MemoryPoolHolder additionalCommandMemory;
    LlgdUniqueUint8PtrWithLwstomDeleter additionalStorage;

    GpuState defaultGpuState;
    LWNcommandHandle baseCommandHandle;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;

    static const int IDX_MAP_OFFSET = 10;
    static const int defaultState = 0xF3;
    static const int editedState = 0xEF73;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    // Create additional memory pool to add command memory during command buffer exelwtion
    static const size_t POOL_SIZE = 65536;
    additionalStorage = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE, 4096);
    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(additionalStorage.get(), POOL_SIZE);
    TEST(additionalCommandMemory.Initialize(&pool_builder));

    // Create defaultGpuState
    defaultGpuState = m_spCommandHelper->RunAndExtractGpuState(m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
        cbh->SetSampleMask(defaultState);
    }));

    // Construct base command for editing
    auto bufferConstructor = [&](CommandBuffer *cbh) {
        // clear is event method
        cbh->SetSampleMask(defaultState);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        cbh->SetSampleMask(defaultState);
        // -- End 1st GPFIFO here --
        cbh->AddCommandMemory(additionalCommandMemory, 0, POOL_SIZE);
        // -- New 2nd GPFIFO starts --
        cbh->ClearColor(0, GREEN, ClearColorMask::RGBA); // Event2 method
    };
    baseCommandHandle = m_spCommandHelper->MakeHandle(bufferConstructor);
    TEST_NEQ(baseCommandHandle, 0);

    return true;
}

bool Validator::Test()
{
    // Edit
    m_spCommandHelper->ResetPointersForEditingCB();
    auto editedHandle = llgdCommandSetEditRasterStateRasterizationSampleMask(
        baseCommandHandle,
        [](uint32_t gpfifoOrderIdx, void*) { return gpfifoOrderIdx + IDX_MAP_OFFSET; },
        0, 1000,  // Edit everything, and don't insert start/end edge patch
        defaultGpuState, defaultGpuState, editedState,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        llgd_lwn::EmptyMethodUpdatedFn,
        m_spCommandHelper.get()
    );

    // Run edited command
    m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

    // Extract state value after exelwted
    const auto raster = m_spCommandHelper->ExtractRasterState();
    const auto actualEditedState = raster.rasterization.sampleMask.value;
    // Then test
    TEST_EQ(actualEditedState, editedState);

    return true;
}

LLGD_DEFINE_TEST(EditStateWithMultiGPFIFOs, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
