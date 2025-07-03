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
#include <LlgdTestUtilEditState.h>

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>
#include "lwnExt/lwnExt_Internal.h"

#pragma clang diagnostic ignored "-Wunused-function"
// Collapser unused
#include <LlgdKinds.h>

#include <vector>

namespace {
#define __LWN_MMU_PTE_KIND_GENERIC_16BX2_TX1 0xfe

#define GET(T,name)                                     \
    const auto name = (T)g_device->GetProcAddress(#name);  \
    TEST(name)

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    static const int PAGE = 65536;
    static const int SIZE = PAGE;

    bool TestGobs();
    uint8_t backing[SIZE] __attribute__(( aligned(PAGE) ));
    uint8_t target[SIZE] __attribute__(( aligned(PAGE) ));

    llgd_lwn::QueueHolder qh;
    llgd_lwn::MemoryPoolHolder mph;
    llgd_lwn::MemoryPoolHolder pph;
    llgd_lwn::MemoryPoolHolder vph;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> commandHelper;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    commandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(commandHelper->Initialize());

    {
        using MPF = MemoryPoolFlags;
        lwn::MemoryPoolBuilder builder;
        builder.SetDefaults();
        builder.SetDevice(g_device);

        builder.SetFlags(MPF::CPU_UNCACHED | MPF::GPU_CACHED | MPF::PHYSICAL);
        builder.SetStorage(backing, SIZE);
        TEST(pph.Initialize(&builder));

        builder.SetFlags(MPF::CPU_UNCACHED | MPF::GPU_CACHED | MPF::COMPRESSIBLE | MPF::VIRTUAL);
        builder.SetStorage(nullptr, SIZE);
        TEST(vph.Initialize(&builder));

        builder.SetFlags(MPF::CPU_UNCACHED | MPF::GPU_UNCACHED);
        builder.SetStorage(target, SIZE);
        TEST(mph.Initialize(&builder));
    }

    {
        lwn::MappingRequest request{
            .physicalPool = pph,
            .physicalOffset = 0,
            .virtualOffset = 0,
            .size = SIZE,
            .storageClass = llgdGetLwnStorageClass(__LWN_MMU_PTE_KIND_GENERIC_16BX2_TX1)
        };
        const auto mapped = vph->MapVirtual(1, &request);
        TEST(mapped)
    }

    return true;
}

bool Validator::Test()
{
    return TestGobs();
}

bool Validator::TestGobs()
{
    const auto devtools = lwnDevtoolsBootstrap();

    for (int i = 0; i < SIZE; ++i) backing[i] = i % 256;

#if defined(MAKE_PARAMETERS)
    {
        lwn::Texture srcTex;
        TextureBuilder builder;
        builder.SetDefaults();
        builder.SetDevice(g_device);
        builder.SetSize2D(128, 128);
        builder.SetTarget(TextureTarget::TARGET_2D);
        builder.SetFormat(Format::RGBA8);

        TEST(builder.GetStorageSize() <= SIZE)
        TEST(builder.GetStorageAlignment() <= PAGE)

        builder.SetStorage(vph, 0);
        TEST(srcTex.Initialize(&builder))

        TextureState state;
        llgdLwnExtractTextureStateFromLwnTexture(*((LWNtexture*)&srcTex), state);

        printf("format %d gob %d %d %d \n",
            (int)state.format,
            (int)state.internal.gobs_per_block_width,
            (int)state.internal.gobs_per_block_height,
            (int)state.internal.gobs_per_block_depth);
    }
#endif

    LWNtexture srcTexture;
    LWNtexture dstTexture;
    {
        int w = 128, h = 128;
         devtools->TextureInitializeCompressedFromBlParams(
             &srcTexture,
             g_device,
             vph->GetBufferAddress(),
             (LWNformat)37,
             &w, &h, 1 /* h */, 0,
             /* log gob per blk x, y, z */ 0, 4, 0);

        int used = 0;
        int texelSize = 0;
        devtools->TextureInitializeForBlCompatImageCopy(&dstTexture, &srcTexture, nullptr,
            SIZE, mph->GetBufferAddress(), &w, &h, &used, &texelSize);
    }

    {
        const auto asserter = [](void* data, uint64_t start, uint64_t size, uint32_t kind) {
            CHECK(kind == 0xfe /* __LWN_MMU_PTE_KIND_GENERIC_16BX2_TX1 */);
        };

        LlgdUtils::KindRange range{ 0 };
        range.send = asserter;
        const auto info = llgdLwnGetLlgdMemoryPool(vph);
        TEST(LlgdUtils::GetMemoryPoolKinds(devtools, g_device, info, asserter, &range))
    }

    {
        GET(PFNLWNCOMMANDBUFFERCOPYTEXTEXCELWX, lwnCommandBufferCopyTextureToTextureWithCopyEngineLWX)
        LWNcopyRegion srcRegion = {0, 0, 0, 128, 128, 1};
        LWNcopyRegion dstRegion = {0, 0, 0, 128, 128, 1};
        const auto handle = commandHelper->MakeHandle([&] (CommandBuffer* cb) {
            lwnCommandBufferCopyTextureToTextureWithCopyEngineLWX(
                (LWNcommandBuffer*)cb,
                &srcTexture, &srcRegion,
                &dstTexture, &dstRegion);
        });
        commandHelper->Run(handle);
    }

    TEST_EQ(backing[15], 15)
    TEST_EQ(backing[16], 16)
    TEST_EQ(target[15], 15)
    TEST_EQ(target[16], 32 /* unswizzled by pitch-in-gob copy */)

    llgdLwnTextureReswizzlePitchInGobInPlace(target, SIZE);

    for (int i = 0; i < SIZE; ++i) TEST_EQ(backing[i], target[i]);

    return true;
}

} // namespace

LLGD_DEFINE_TEST(Reswizzle, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
});
