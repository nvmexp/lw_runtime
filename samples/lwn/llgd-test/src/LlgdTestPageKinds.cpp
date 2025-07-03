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

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>

#include <assert.h>
#include <LlgdKinds.h>

#include <iostream>
#include <vector>
#include <cstdio>

// ----------------------------------------------------------------------------
// Printer
// ----------------------------------------------------------------------------
static uint64_t id = 42;
static uint64_t ev = 32;

struct KindTriplet {
    uint64_t start;
    uint64_t size;
    uint32_t kind;

    KindTriplet(uint64_t _start, uint64_t _size, uint32_t _kind)
        : start(_start), size(_size), kind(_kind) {}

    bool operator ==(const KindTriplet& rhs) const
    {
        return start == rhs.start && size == rhs.size && kind == rhs.kind;
    }

    std::string toString() const
    {
        char buf[1024];
        snprintf(buf, 1024, "start=0x%lx, size=0x%lx, kind=%d", start, size, kind);
        return std::string(buf);
    }
};

static std::vector<KindTriplet> triplets{};

static void Appender(void* data, uint64_t start, uint64_t size, uint32_t kind)
{
#if defined(DEBUG)
    auto& range = *reinterpret_cast<LlgdUtils::KindRange*>(data);
    CHECK(id == range.poolId);
    CHECK(ev == range.eventIndex);
#endif
    triplets.push_back({ start, size, kind });
}

static const LWNdevtoolsBootstrapFunctions* devtools;

static void getKinds(llgd_lwn::MemoryPoolHolder& mph)
{
    LlgdUtils::KindRange range{ 0 };
    range.poolId = id;
    range.eventIndex = ev;
    range.send = &Appender;

    const auto info = llgdLwnGetLlgdMemoryPool(mph);

    triplets.clear();
    if (!LlgdUtils::GetMemoryPoolKinds(devtools, g_device, info, LlgdUtils::Collapser, &range)) { return; }

    if (range.size != 0) {
        Appender(&range, range.start, range.size, range.kind);
    }
};

static const size_t PAGE = 65536;
static const size_t SIZE = 4 * PAGE;
static const size_t ALIGNT = LWN_MEMORY_POOL_STORAGE_ALIGNMENT;

static bool TestCompressiblePageKinds()
{
    llgd_lwn::MemoryPoolHolder mph;

    using Flags = lwn::MemoryPoolFlags;
    static const auto poolFlags = Flags::CPU_UNCACHED | Flags::GPU_CACHED | Flags::COMPRESSIBLE;

    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> storage = LlgdAlignedAllocPodType<uint8_t>(SIZE, ALIGNT);

    MemoryPoolBuilder mpb;
    mpb.SetDevice(g_device).SetDefaults()
        .SetFlags(poolFlags)
        .SetStorage(storage.get(), SIZE);
    TEST(mph.Initialize(&mpb));

    static const int width = 4, height = 4;
    TextureBuilder tb;
    tb.SetDevice(g_device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::DEPTH24_STENCIL8)
        .SetFlags(TextureFlags::COMPRESSIBLE)
        .SetSize2D(width, height);

    getKinds(mph);
    TEST((triplets[0] == KindTriplet{ 0 * PAGE, 4 * PAGE, 219 }));

    llgd_lwn::TextureHolder th0;
    tb.SetStorage(mph, 2 * PAGE);
    TEST(th0.Initialize(&tb));
    getKinds(mph);
    TEST((triplets[0] == KindTriplet{ 0 * PAGE, 2 * PAGE, 219 }));
    TEST((triplets[1] == KindTriplet{ 2 * PAGE, 1 * PAGE, 81 }));
    TEST((triplets[2] == KindTriplet{ 3 * PAGE, 1 * PAGE, 219 }));

    llgd_lwn::TextureHolder th1;
    tb.SetStorage(mph, 0);
    TEST(th1.Initialize(&tb));
    getKinds(mph);
    TEST((triplets[0] == KindTriplet{ 0 * PAGE, 1 * PAGE, 81 }));
    TEST((triplets[1] == KindTriplet{ 1 * PAGE, 1 * PAGE, 219 }));
    TEST((triplets[2] == KindTriplet{ 2 * PAGE, 1 * PAGE, 81 }));
    TEST((triplets[3] == KindTriplet{ 3 * PAGE, 1 * PAGE, 219 }));

    llgd_lwn::TextureHolder th2;
    tb.SetStorage(mph, 3 * PAGE);
    TEST(th2.Initialize(&tb));
    getKinds(mph);
    TEST((triplets[0] == KindTriplet{ 0 * PAGE, 1 * PAGE, 81 }));
    TEST((triplets[1] == KindTriplet{ 1 * PAGE, 1 * PAGE, 219 }));
    TEST((triplets[2] == KindTriplet{ 2 * PAGE, 2 * PAGE, 81 }));

    llgd_lwn::TextureHolder th3;
    tb.SetStorage(mph, 1 * PAGE);
    TEST(th3.Initialize(&tb));
    getKinds(mph);
    TEST((triplets[0] == KindTriplet{ 0 * PAGE, 4 * PAGE, 81 }));

    return true;
}
#if defined(LW_HOS) //WAR
// TODO: (http://lwbugs/3102903) Remove this #if when REMAP is ported to linux
static bool TestVirtualPageKinds()
{
    using Flags = lwn::MemoryPoolFlags;
    static const auto virtualPoolFlags = Flags::CPU_NO_ACCESS | Flags::GPU_CACHED | Flags::VIRTUAL;

    // Initialize virtual pool
    llgd_lwn::MemoryPoolHolder vph;
    MemoryPoolBuilder vpb;
    vpb.SetDevice(g_device).SetDefaults()
        .SetFlags(virtualPoolFlags)
        .SetStorage(nullptr, SIZE);
    TEST(vph.Initialize(&vpb));

    // Create / send a mapping request
    auto mapVirtual = [&](lwn::MemoryPool* physicalPool, TextureBuilder& tb) {
        MappingRequest mapping;
        mapping.physicalPool = physicalPool;
        mapping.physicalOffset = 0;
        mapping.virtualOffset = 0;
        mapping.size = SIZE;
        mapping.storageClass = tb.GetStorageClass();
        return vph->MapVirtual(1, &mapping);
    };

    // Test with no compression buffer
    {
        // Storage for physical pool which is mapped to the virtual pool
        std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> storage = LlgdAlignedAllocPodType<uint8_t>(SIZE, ALIGNT);

        // Create a linear texture (no compression) builder
        static const int width = 4, height = 4;
        TextureBuilder tb;
        tb.SetDevice(g_device).SetDefaults()
            .SetTarget(TextureTarget::TARGET_BUFFER)
            .SetFormat(Format::RGBA8)
            .SetFlags(TextureFlags::LINEAR)
            .SetSize2D(width, height);

        // Map the physical pool to virtual pool
        llgd_lwn::MemoryPoolHolder mph;
        MemoryPoolBuilder mpb;
        mpb.SetDevice(g_device).SetDefaults()
            .SetFlags(Flags::CPU_NO_ACCESS | Flags::GPU_CACHED)
            .SetStorage(storage.get(), SIZE);
        TEST(mph.Initialize(&mpb));
        TEST(mapVirtual(mph, tb));

        // The mapped physical pool is linear buffer, therefore the PTE kind must be "pitch"
        getKinds(vph);
        TEST_EQ(triplets.size(), 1u);
        TEST(llgdLwnIsPteKindPitch(triplets[0].kind));

        // Unmap before finalizing physical pool
        TEST(mapVirtual(nullptr, tb));

        mph.Finalize();
    }

    // Test with compressible physical buffer
    {
        // Storage for physical pool which is mapped to the virtual pool
        std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> storage = LlgdAlignedAllocPodType<uint8_t>(SIZE, ALIGNT);

        // Create a compressible texture (no compression) builder
        static const int width = 4, height = 4;
        TextureBuilder tb;
        tb.SetDevice(g_device).SetDefaults()
            .SetTarget(TextureTarget::TARGET_2D)
            .SetFormat(Format::RGBA8)
            .SetFlags(TextureFlags::COMPRESSIBLE)
            .SetSize2D(width, height);

        // Map the physical pool to virtual pool
        llgd_lwn::MemoryPoolHolder mph;
        MemoryPoolBuilder mpb;
        mpb.SetDevice(g_device).SetDefaults()
            .SetFlags(Flags::CPU_NO_ACCESS | Flags::GPU_CACHED | Flags::COMPRESSIBLE)
            .SetStorage(storage.get(), SIZE);
        TEST(mph.Initialize(&mpb));
        TEST(mapVirtual(mph, tb));

        // The mapped physical pool is compressible buffer, therefore the PTE kind must not be "pitch" or "block linear"
        getKinds(vph);
        TEST_EQ(triplets.size(), 1u);
        TEST(!llgdLwnIsPteKindPitch(triplets[0].kind));
        TEST(!llgdLwnIsPteKindBlockLinear(triplets[0].kind));

        // Unmap before finalizing physical pool
        TEST(mapVirtual(nullptr, tb));

        mph.Finalize();
    }

    return true;
}
#endif
static bool TestPageKinds()
{
    devtools = lwnDevtoolsBootstrap();

    // We need to have one live queue for some of the Getter things in
    // devtools bootstrap land.
    llgd_lwn::QueueHolder qh;
    qh.Initialize(g_device);

    TEST(TestCompressiblePageKinds());
#if defined(LW_HOS) //WAR
    // TODO: (http://lwbugs/3102903) Remove this #if when REMAP is ported to linux
    TEST(TestVirtualPageKinds());
#endif
    return true;
}

LLGD_DEFINE_TEST(PageKinds, UNIT, LwError Execute() { return TestPageKinds() ? LwSuccess : LwError_IlwalidState; });
