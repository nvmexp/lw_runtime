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
#include <string>
#include <functional>

namespace {
class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool TestBlitRemove(uint32_t removeIndex, BlitEventType blitType);
    bool CreateTexture(llgd_lwn::TextureHolder& texture, llgd_lwn::MemoryPoolHolder& memoryPool, uint64_t offsetInPool, uint32_t width, uint32_t height, uint32_t depth, uint32_t samples = 1);
    bool CreateBuffer(Buffer& buffer, llgd_lwn::MemoryPoolHolder& memoryPool, uint64_t offsetInPool, uint32_t bufferSize);

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

bool Validator::CreateTexture(llgd_lwn::TextureHolder& texture, llgd_lwn::MemoryPoolHolder& memoryPool, uint64_t offsetInPool, uint32_t width, uint32_t height, uint32_t depth, uint32_t samples)
{
    TextureBuilder tex_builder;
    tex_builder.SetDevice(g_device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetSize3D(width, height, depth)
        .SetFormat(Format::RGBX8)
        .SetSamples(samples)
        .SetStorage(memoryPool, offsetInPool);
    TEST(texture.Initialize(&tex_builder));

    return true;
}

bool Validator::CreateBuffer(Buffer& buffer, llgd_lwn::MemoryPoolHolder& memoryPool, uint64_t offsetInPool, uint32_t bufferSize)
{
    BufferBuilder buffer_builder;
    buffer_builder.SetDevice(g_device).SetDefaults()
        .SetStorage(memoryPool, offsetInPool, bufferSize);
    TEST(buffer.Initialize(&buffer_builder));

    return true;
}

bool Validator::TestBlitRemove(uint32_t removeIndex, BlitEventType eventType)
{
    // Create memory pool for texture and buffer
    static const size_t POOL_SIZE = 40 << 20;
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

    static const uint32_t BUFFER_SIZE = 5 << 20;
    static const uint32_t TEXTURE1_WIDTH = 100;
    static const uint32_t TEXTURE1_HEIGHT = 50;
    static const uint32_t TEXTURE1_DEPTH = 3;
    static const uint32_t TEXTURE2_WIDTH = 50;
    static const uint32_t TEXTURE2_HEIGHT = 40;
    static const uint32_t TEXTURE2_DEPTH = 4;
    llgd_lwn::TextureHolder textureSrc, textureDst;
    Buffer bufferSrc, bufferDst;
    lwn::CopyRegion copyRegion, copyRegionDst;

    // Create command buffers, one contains blit, one doesn't contain blit

    // CommandBuffer without the draw method
    const auto reference = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
        cb->Barrier(~0);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
    });
    const auto referenceMethods = llgd_lwn::GetMethods(reference);

    // CommandBuffer with the draw method
    bool successMakeHandle = true;
    const auto handle = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
        cb->Barrier(~0);
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        switch (eventType)
        {
        case BlitEventType::CopyBufferToBuffer:
            successMakeHandle &= CreateBuffer(bufferSrc, mph, 0, BUFFER_SIZE);
            successMakeHandle &= CreateBuffer(bufferDst, mph, BUFFER_SIZE, BUFFER_SIZE);
            cb->CopyBufferToBuffer(bufferSrc.GetAddress(), bufferDst.GetAddress(), BUFFER_SIZE, 0 /*flags*/);
            break;
        case BlitEventType::CopyBufferToTexture:
            successMakeHandle &= CreateBuffer(bufferSrc, mph, 0, BUFFER_SIZE);
            successMakeHandle &= CreateTexture(textureDst, mph, BUFFER_SIZE, TEXTURE1_WIDTH, TEXTURE1_HEIGHT, TEXTURE1_DEPTH);
            copyRegion.xoffset = 1;
            copyRegion.yoffset = 2;
            copyRegion.zoffset = 0;
            copyRegion.width = 90;
            copyRegion.height = 40;
            copyRegion.depth = 3;
            cb->CopyBufferToTexture(bufferSrc.GetAddress(), (const lwn::Texture *)textureDst, nullptr, &copyRegion, 0 /*flags*/);
            break;
        case BlitEventType::CopyTextureToBuffer:
            successMakeHandle &= CreateTexture(textureSrc, mph, 0, TEXTURE1_WIDTH, TEXTURE1_HEIGHT, TEXTURE1_DEPTH);
            successMakeHandle &= CreateBuffer(bufferDst, mph, BUFFER_SIZE, BUFFER_SIZE);
            copyRegion.xoffset = 1;
            copyRegion.yoffset = 2;
            copyRegion.zoffset = 0;
            copyRegion.width = 90;
            copyRegion.height = 40;
            copyRegion.depth = 3;
            cb->CopyTextureToBuffer((const lwn::Texture *)textureSrc, nullptr, &copyRegion, bufferDst.GetAddress(), 0 /*flags*/);
            break;
        case BlitEventType::CopyTextureToTexture:
            successMakeHandle &= CreateTexture(textureSrc, mph, 0, TEXTURE1_WIDTH, TEXTURE1_HEIGHT, TEXTURE1_DEPTH);
            successMakeHandle &= CreateTexture(textureDst, mph, BUFFER_SIZE, TEXTURE2_WIDTH, TEXTURE2_HEIGHT, TEXTURE2_DEPTH);
            copyRegion.xoffset = 1;
            copyRegion.yoffset = 2;
            copyRegion.zoffset = 1;
            copyRegionDst.xoffset = 9;
            copyRegionDst.yoffset = 20;
            copyRegionDst.zoffset = 2;
            copyRegion.width = copyRegionDst.width = 40;
            copyRegion.height = copyRegionDst.height = 20;
            copyRegion.depth = copyRegionDst.depth = 2;
            cb->CopyTextureToTexture((const lwn::Texture *)textureSrc, nullptr, &copyRegion, (const lwn::Texture *)textureDst, nullptr, &copyRegionDst, 0);
            break;
        case BlitEventType::Downsample:
            successMakeHandle &= CreateTexture(textureSrc, mph, 0, TEXTURE1_WIDTH, TEXTURE1_HEIGHT, TEXTURE1_DEPTH);
            successMakeHandle &= CreateTexture(textureDst, mph, BUFFER_SIZE, TEXTURE2_WIDTH, TEXTURE2_HEIGHT, TEXTURE2_DEPTH);
            cb->Downsample((const lwn::Texture *)textureSrc, (const lwn::Texture *)textureDst);
            break;
        case BlitEventType::TiledDownsample:
            successMakeHandle &= CreateTexture(textureSrc, mph, 0, TEXTURE1_WIDTH, TEXTURE1_HEIGHT, TEXTURE1_DEPTH, 2);
            successMakeHandle &= CreateTexture(textureDst, mph, BUFFER_SIZE, TEXTURE2_WIDTH, TEXTURE2_HEIGHT, TEXTURE2_DEPTH, 2);
            cb->TiledDownsample((const lwn::Texture *)textureSrc, (const lwn::Texture *)textureDst);
            break;
        default:
            break;
        }
        cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
    });
    TEST(successMakeHandle);
    const uint64_t eventCountWithBlit = llgdLwnGetEventCount(handle);

    // Execute removing!
    m_spCommandHelper->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetRemoveBlit(
        handle,
        [](uint32_t index, void*) { return 10 + index; },
        removeIndex,
        eventType,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        m_spCommandHelper.get());

    const auto decoded = m_spCommandHelper->MakeCommandHandleRunnable(patched);
    const uint64_t eventCountAfterRemove = llgdLwnGetEventCount(decoded);
    const auto patchedMethods = llgd_lwn::GetMethods(decoded);

    TEST(eventCountAfterRemove < eventCountWithBlit);

    TEST(llgd_lwn::CompareMethods(patchedMethods, referenceMethods));

    return true;
}

bool Validator::Test()
{
    // Test remove all the events
    for (uint8_t eventType = 0; eventType <= uint8_t(BlitEventType::TiledDownsample); ++eventType) {
        TEST(TestBlitRemove(13, BlitEventType(eventType)));
    }

    return true;
}
}

LLGD_DEFINE_TEST(EditBlitRemove, UNIT, LwError Execute() {
    Validator v{};
    return (v.Initialize() && v.Test()) ? LwSuccess : LwError_IlwalidState;
});
