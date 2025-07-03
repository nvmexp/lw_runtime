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

static bool Verify(const uint32_t* head,
            const int  original_minlayer,
            const bool original_sparse,
            const bool original_compressible,
            const bool original_image)
{
    int  minlayer     = llgdLwnReadTexHeaderMinLayer(head);
    bool sparse       = llgdLwnReadTexHeaderSparseBit(head);
    bool compressible = llgdLwnReadTexHeaderCompressibleBit(head);
    bool image        = llgdLwnReadTexHeaderImageBit(head);

    TEST_EQ(minlayer, original_minlayer);
    TEST_EQ(sparse, original_sparse);
    TEST_EQ(compressible, original_compressible);
    TEST_EQ(image, original_image);

    return true;
}

class TextureHeadValidator {
public:
    void Initialize();
    bool Test();

private:
    void InitializeLWN();
    void InitializePools();
    void InitializeTextures();

    static const size_t POOL_SIZE = 40 << 20;
    static const size_t HEAD_SIZE =  4096 << 10;
    // Note: don't put storage on stack because its size is too huge and driver cannot change the memory attribute storage on stack
    using SmartStoragePtr = std::unique_ptr<uint8_t, std::function<void(uint8_t*)>>;
    SmartStoragePtr     space_pool;
    SmartStoragePtr     space_head;

    int tex_head_size;
    int reserved_tex;
    int num_layers;

    llgd_lwn::QueueHolder qh;
    llgd_lwn::MemoryPoolHolder mph;
    llgd_lwn::MemoryPoolHolder mph_head;
    llgd_lwn::TexturePoolHolder tph;
    llgd_lwn::TextureHolder th_full;
    llgd_lwn::TextureHolder th_sparse;
    llgd_lwn::TextureHolder th_compressible;
};

void TextureHeadValidator::InitializeLWN()
{
    qh.Initialize(g_device);

    int max_texture_layers = 0;
    g_device->GetInteger(DeviceInfo::MAX_TEXTURE_LAYERS,           &max_texture_layers);
    g_device->GetInteger(DeviceInfo::RESERVED_TEXTURE_DESCRIPTORS, &reserved_tex);
    g_device->GetInteger(DeviceInfo::TEXTURE_DESCRIPTOR_SIZE,      &tex_head_size);

    num_layers = std::min(5, max_texture_layers);
}

void TextureHeadValidator::InitializePools()
{
    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
                .SetFlags(MemoryPoolFlags::CPU_CACHED |
                          MemoryPoolFlags::GPU_CACHED |
                          MemoryPoolFlags::COMPRESSIBLE);

    space_pool = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE, 4096);
    pool_builder.SetStorage(space_pool.get(), POOL_SIZE);
    CHECK(mph.Initialize(&pool_builder));

    space_head = LlgdAlignedAllocPodType<uint8_t>(HEAD_SIZE, 4096);
    pool_builder.SetStorage(space_head.get(), HEAD_SIZE);
    CHECK(mph_head.Initialize(&pool_builder));
    CHECK(tph.Initialize((const MemoryPool *)mph_head, 0, HEAD_SIZE / tex_head_size));
}

void TextureHeadValidator::InitializeTextures()
{
    TextureBuilder tex_builder;
    tex_builder.SetDevice(g_device).SetDefaults()
               .SetTarget(TextureTarget::TARGET_3D)
               .SetSize3D(16, 16, num_layers)
               .SetFormat(Format::RGBX8)
               .SetStorage(mph, 0);
    CHECK(th_full.Initialize(&tex_builder));

    tex_builder.SetFlags(TextureFlags::SPARSE);
    CHECK(th_sparse.Initialize(&tex_builder));

    tex_builder.SetFlags(TextureFlags::COMPRESSIBLE);
    CHECK(th_compressible.Initialize(&tex_builder));
}

void TextureHeadValidator::Initialize()
{
    InitializeLWN();
    InitializePools();
    InitializeTextures();
}

#define REGISTER_VERIFY_TEST_CASE(s,s_b,i,i_b)          \
    tph->Register##i(tex_id, th_##s, &tex_view);  \
    if (!Verify(head, l, s_b, false, i_b)) { return false; }

bool TextureHeadValidator::Test()
{
    const uint32_t *head;
    uint16_t tex_id;
    TextureView tex_view;
    tex_view.SetDefaults();

    tex_id = (uint16_t)reserved_tex + 0;

#if defined(LW_HOS)
    uint8_t* base = space_head.get();
#else
    // On a non-HOS system, driver remaps the application provided memory. Optimally
    // we should not rely on another liblwn-llgd call in the test, but llgdLwnGetLlgdMemoryPool
    // appears to be the easiest way (if not the only way) to get the actual CPU address
    // of the pool here.
    LlgdMemoryPool llgdMemoryPool = llgdLwnGetLlgdMemoryPool(mph_head);
    uint8_t* base = (uint8_t*)llgdMemoryPool.cpuAddress;
#endif
    head = (uint32_t*)(base + tex_head_size * tex_id);

    for (int l = 0; l < num_layers; l++) {
        tex_view.SetLayers(l, 1);

        tph->RegisterTexture(tex_id, th_compressible, &tex_view);
        if (!Verify(head, l, false, true, false)) { return false; }

        REGISTER_VERIFY_TEST_CASE(    full, false,   Texture, false  );
        REGISTER_VERIFY_TEST_CASE(  sparse, true,    Texture, false  );
        REGISTER_VERIFY_TEST_CASE(  sparse, true,      Image, true   );
        REGISTER_VERIFY_TEST_CASE(    full, false,     Image, true   );
    }

    return true;
}
#undef REGISTER_VERIFY_TEST_CASE

LLGD_DEFINE_TEST(TextureHeader, UNIT,
LwError Execute()
{
    auto v = std::make_unique<TextureHeadValidator>();
    v->Initialize();
    return v->Test() ? LwSuccess : LwError_IlwalidState;
}
); // LLGD_DEFINE_TEST