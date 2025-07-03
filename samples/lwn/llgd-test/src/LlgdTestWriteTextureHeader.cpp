/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
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

#include <liblwn-llgd.h>

namespace {
TextureState OnlyEditable(const TextureState& state)
{
    TextureState partial;
    memset(&partial, 0, sizeof(partial));
    partial.internal.byte_pitch = state.internal.byte_pitch;
    partial.baseAddress = state.baseAddress;
    partial.format = state.format;
    partial.target = state.target;
    partial.flags = state.flags;
    partial.width = state.width;
    partial.height = state.height;
    partial.depth = state.depth;
    partial.samples = state.samples;
    partial.swizzles[0] = state.swizzles[0];
    partial.swizzles[1] = state.swizzles[1];
    partial.swizzles[2] = state.swizzles[2];
    partial.swizzles[3] = state.swizzles[3];
    partial.depthStencilMode = state.depthStencilMode;
    partial.minLevel = state.minLevel;
    partial.maxLevel = state.maxLevel;
    partial.minLayer = state.minLayer;
    partial.numLayers = state.numLayers;
    return partial;
}
}

class Validator {
public:
    void Initialize();
    bool Test();

private:
    template<typename F>
    bool SubTest(F&& config);
    int tex_head_size;
    int reserved_tex;

    static const size_t POOL_SIZE = 30 << 20;
    LlgdUniqueUint8PtrWithLwstomDeleter tex_storage;
    llgd_lwn::MemoryPoolHolder mph_tex;

    static const size_t TEX_POOL_SIZE = 4096;
    LlgdUniqueUint8PtrWithLwstomDeleter tex_pool_storage;
    llgd_lwn::MemoryPoolHolder mph_tex_pool;
    llgd_lwn::TexturePoolHolder tph;
};

void Validator::Initialize()
{
    g_device->GetInteger(DeviceInfo::RESERVED_TEXTURE_DESCRIPTORS, &reserved_tex);
    g_device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE, &tex_head_size);

    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
                .SetFlags(MemoryPoolFlags::CPU_UNCACHED |
                          MemoryPoolFlags::GPU_CACHED |
                          MemoryPoolFlags::COMPRESSIBLE);

    tex_storage = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE, 4096);
    pool_builder.SetStorage(tex_storage.get(), POOL_SIZE);
    CHECK(mph_tex.Initialize(&pool_builder));

    tex_pool_storage = LlgdAlignedAllocPodType<uint8_t>(TEX_POOL_SIZE, 4096);
    pool_builder.SetStorage(tex_pool_storage.get(), TEX_POOL_SIZE);
    CHECK(mph_tex_pool.Initialize(&pool_builder));
    CHECK(tph.Initialize((const MemoryPool *)mph_tex_pool, 0, TEX_POOL_SIZE / tex_head_size));
}

template<typename F>
bool Validator::SubTest(F&& config)
{
    llgd_lwn::TextureHolder th;
    {
        TextureBuilder tex_builder;
        tex_builder.SetDevice(g_device).SetDefaults()
                   .SetStorage(mph_tex, 0);
        config(tex_builder);
        CHECK(th.Initialize(&tex_builder));
    }

    const auto tex_id = static_cast<uint16_t>(reserved_tex) + 1 ;
    tph->RegisterTexture(tex_id, th, nullptr);

    const auto head = (uint32_t*)(tex_pool_storage.get() + tex_head_size * tex_id);
    TextureState state;
    CHECK(llgdLwnExtractTextureState(head, state));

    // Hand copy editable fields
    TextureState partial = OnlyEditable(state);

    CHECK(llgdTextureStateCallwlateDerived(partial));

    struct {
        uint32_t _inner[8];
    } copy;
    CHECK(sizeof(copy) == tex_head_size);
    CHECK(llgdTextureStateToHwTexture(partial, &copy, sizeof(copy)));

    // Careful, __lwnLlgdWriteTexHeader() may muddle things up.
    CHECK(0 == memcmp(head, &copy, sizeof(copy)));

    return true;
}

bool Validator::Test()
{
    CHECK(SubTest([] (TextureBuilder& tb) {
        tb.SetFormat(Format::RGBA8)
          .SetTarget(TextureTarget::TARGET_3D)
          .SetSize3D(1280, 720, 4);
    }));
    CHECK(SubTest([] (TextureBuilder& tb) {
        tb.SetFormat(Format::DEPTH24_STENCIL8)
          .SetTarget(TextureTarget::TARGET_2D)
          .SetSize2D(1280, 720);
    }));
    CHECK(SubTest([] (TextureBuilder& tb) {
        tb.SetFormat(Format::RGB32F)
          .SetTarget(TextureTarget::TARGET_BUFFER)
          .SetSize1D(1000);
    }));
    return true;
}

LLGD_DEFINE_TEST(WriteTextureHeader, UNIT,
LwError Execute()
{
    Validator v;
    v.Initialize();

    if (!v.Test()) { return LwError_IlwalidState; }
    else           { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
