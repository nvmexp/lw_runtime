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

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool InitializeBaseCommand();

private:
    llgd_lwn::QueueHolder qh;
    llgd_lwn::TextureHolder tex;  // Used for ClearTexture (bookends event)
    llgd_lwn::MemoryPoolHolder poolForTex;
    LlgdUniqueUint8PtrWithLwstomDeleter texStorage;

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

    return InitializeBaseCommand();
}

bool Validator::InitializeBaseCommand()
{
    // Get defaultGpuState
    defaultGpuState = m_spCommandHelper->RunAndExtractGpuState(m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
        cbh->SetPolygonOffsetClamp(defaultState, 0, 0);
    }));

    // Create pool for texture
    static const size_t POOL_SIZE = 65536;
    texStorage = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE, 4096);
    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::COMPRESSIBLE)
        .SetStorage(texStorage.get(), POOL_SIZE);
    TEST(poolForTex.Initialize(&pool_builder));

    // Build texture
    TextureBuilder tex_builder;
    tex_builder.SetDevice(g_device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_3D)
        .SetSize3D(16, 16, 2)
        .SetFormat(Format::RGBX8)
        .SetStorage(poolForTex, 0);
    TEST(tex.Initialize(&tex_builder));

    // Variables for ClearTexture
    TextureView texView;
    texView.SetDefaults();

    CopyRegion copyRegion{ 0 };
    copyRegion.width = copyRegion.height = 2;
    copyRegion.depth = 1;

    static const float clearColor[4] = { 0.3f, 0.5f, 0.8f, 1.0f };

    // Construct base command for editing
    auto bufferConstructor = [&](CommandBuffer *cbh) {
        // clear is event method
        // FenceSync and WaitSync are events which create tokens
        cbh->SetPolygonOffsetClamp(defaultState, 0.5f, 0.5f);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        cbh->SetPolygonOffsetClamp(defaultState, 0.5f, 0.5f);
        cbh->ClearTexture(tex, &texView, &copyRegion, clearColor, lwn::ClearColorMask::RGBA); // Bookend Event: idx=2
        cbh->SetPolygonOffsetClamp(defaultState, 0.5f, 0.5f);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event3 method
    };
    baseCommandHandle = m_spCommandHelper->MakeHandle(bufferConstructor);
    TEST_NEQ(baseCommandHandle, 0);

    return true;
}

bool Validator::Test()
{
    // Edit
    m_spCommandHelper->ResetPointersForEditingCB();
    auto editedHandle = llgdCommandSetEditRasterStateRasterizationPolygonOffsetFactor(
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

    // Extract state value after exelwted, then test
    const auto raster = m_spCommandHelper->ExtractRasterState();
    const auto actualEditedState = raster.rasterization.polygonOffsetFactor.value;
    TEST_EQ(actualEditedState, editedState);

    return true;
}

LLGD_DEFINE_TEST(EditStateWithBookends, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
