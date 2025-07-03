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

#include <class/cl9097.h>
#include <include/g_gf100lwnmmemethods.h>

static const float GREEN[4]{ 0, 1, 0, 1 };

enum class ClearRTType
{
    Color,
    DepthStencil,
};

enum class ClearNotRTType
{
    Texture,
    Buffer,
};

namespace {
class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool TestClearRTsRemove(ClearRTType clearType);
    bool TestClearNotRTsRemove(ClearNotRTType clearType);

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

bool Validator::TestClearRTsRemove(ClearRTType clearType)
{
    const auto handle = m_spCommandHelper->MakeHandle([&] (CommandBuffer* cb) {
        cb->Barrier(~0); // Event1 method
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event2 by token
        if (clearType == ClearRTType::Color) {
            cb->ClearColor(0, GREEN, ClearColorMask::RGBA); // Event3 clear color
        } else {
            cb->ClearDepthStencil(0.5f, false /*depthMask*/, 1, 0xff); // Event3 clear depth-stencil
        }
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event4 by token
    });

    m_spCommandHelper->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetRemoveClear(
        handle,
        [](uint32_t index, void*) { return 10 + index; }, 13,
        clearType == ClearRTType::Color ? ClearEventType::Color : ClearEventType::DepthStencil,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        m_spCommandHelper.get());

    const auto decoded = m_spCommandHelper->MakeCommandHandleRunnable(patched);
    m_spCommandHelper->Run(decoded); // don't crash

    const auto patchedMethods = llgd_lwn::GetMethods(decoded);
    const uint32_t clearTargetMethod = clearType == ClearRTType::Color ? LW9097_LWN_MME_CLEAR_COLOR : LW9097_LWN_MME_CLEAR_DEPTH_STENCIL;
    TEST(!llgd_lwn::FindMethod(patchedMethods, clearTargetMethod));

    return true;
}

bool Validator::TestClearNotRTsRemove(ClearNotRTType clearType)
{
    // Create memory pool for texture and buffer
    static const size_t POOL_SIZE = 40 << 10;
    auto space_pool = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE, 4096);
    llgd_lwn::MemoryPoolHolder mph;
    {
        MemoryPoolBuilder pool_builder;
        pool_builder.SetDevice(g_device).SetDefaults()
            .SetFlags(MemoryPoolFlags::CPU_CACHED |
                MemoryPoolFlags::GPU_CACHED |
                MemoryPoolFlags::COMPRESSIBLE);
        pool_builder.SetStorage(space_pool.get(), POOL_SIZE);
        TEST(mph.Initialize(&pool_builder));
    }

    // Create texture to clear
    static const int width = 4, height = 4;
    llgd_lwn::TextureHolder texture;
    TextureBuilder tex_builder;
    tex_builder.SetDevice(g_device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetSize2D(width, height)
        .SetFormat(Format::RGBX8)
        .SetStorage(mph, 0);
    TEST(texture.Initialize(&tex_builder));

    // Build buffer for clearing
    static const int BUFFER_SIZE = POOL_SIZE / 4;
    static const int BUFFER_OFFSET = POOL_SIZE / 2;
    Buffer buffer;
    BufferBuilder buffer_builder;
    buffer_builder.SetDevice(g_device).SetDefaults()
        .SetStorage(mph, BUFFER_OFFSET, BUFFER_SIZE);
    TEST(buffer.Initialize(&buffer_builder));

    const auto reference = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
        cb->Barrier(~0);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
    });
    const auto referenceMethods = llgd_lwn::GetMethods(reference);

    const auto handle = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
        cb->Barrier(~0); // Event1 method
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event2 by token
        if (clearType == ClearNotRTType::Texture) {
            lwn::CopyRegion region = {0, 0, 0, width, height, 1};
            cb->ClearTexture((const lwn::Texture *)texture, nullptr, &region, GREEN, 0xff); // Event3 clear texture
        } else {
            cb->ClearBuffer(mph->GetBufferAddress() + BUFFER_OFFSET, BUFFER_SIZE, 1); // Event3 clear buffer
        }
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); // Event4 by token
    });

    m_spCommandHelper->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetRemoveClear(
        handle,
        [](uint32_t index, void*) { return 10 + index; }, 13,
        clearType == ClearNotRTType::Texture ? ClearEventType::Texture : ClearEventType::Buffer,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        m_spCommandHelper.get());

    const auto decoded = m_spCommandHelper->MakeCommandHandleRunnable(patched);
    m_spCommandHelper->Run(decoded); // don't crash
    const auto patchedMethods = llgd_lwn::GetMethods(decoded);

    TEST(llgd_lwn::CompareMethods(patchedMethods, referenceMethods));

    return true;
}

bool Validator::Test()
{
    TEST(TestClearRTsRemove(ClearRTType::Color));
    TEST(TestClearRTsRemove(ClearRTType::DepthStencil));
    TEST(TestClearNotRTsRemove(ClearNotRTType::Texture));
    TEST(TestClearNotRTsRemove(ClearNotRTType::Buffer));

    return true;
}
}

LLGD_DEFINE_TEST(EditClearRemove, UNIT, LwError Execute() {
    Validator v{};
    return (v.Initialize() && v.Test()) ? LwSuccess : LwError_IlwalidState;
});
