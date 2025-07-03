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
#include <LlgdTestUtilPool.h>

#include <lwndevtools_bootstrap.h>

static bool TestReservedVa()
{
    llgd_lwn::QueueHolder qh;
    qh.Initialize(g_device);
    // We need to have one live queue for some of the Getter things in
    // devtools bootstrap land.

    const auto devtools = lwnDevtoolsBootstrap();

    llgd_lwn::PoolUtil poolUtil;

    auto Make = [&] (llgd_lwn::PoolType poolType) {
        LWNdevtoolsReservedVa reserved{ 0 };

        llgd_lwn::MemoryPoolHolder mph;
        poolUtil.InitPool(mph, poolType);

        if (poolType == llgd_lwn::PoolType::CompressedPhysicalHasCompbits) {
            // Compbits are lazy allocated, trigger alloc.

            llgd_lwn::MemoryPoolHolder vph;
            poolUtil.InitPool(vph, llgd_lwn::PoolType::CompressedVirtual);

            const auto compressedStorageClass = [&] {
                TextureBuilder tb;
                tb.SetDevice(g_device)
                  .SetDefaults()
                  .SetFlags(TextureFlags::COMPRESSIBLE)
                  .SetTarget(TextureTarget::TARGET_2D)
                  .SetSize2D(64, 64)
                  .SetFormat(Format::RGBA8);
                return tb.GetStorageClass();
            }();

            MappingRequest mapping;
            mapping.physicalPool = mph;
            mapping.physicalOffset = 0;
            mapping.virtualOffset = 0;
            mapping.size = llgd_lwn::PoolUtil::MinSize(poolType);
            mapping.storageClass = compressedStorageClass;
            vph->MapVirtual(1, &mapping);

            // While things are alive.
            reserved = devtools->MemoryPoolProbeVas(mph);
        } else {
            reserved = devtools->MemoryPoolProbeVas(mph);
        }

        return reserved;
    };

    auto Test = [&] (llgd_lwn::PoolType poolType) {
        const auto vas1 = Make(poolType);
        const auto vas2 = Make(poolType);
        const auto success = devtools->ReservedVasReserve(vas2);
        const auto vas3 = Make(poolType);
        devtools->ReservedVasFree(vas2);
        const auto vas4 = Make(poolType);

        const bool ownsCompbits = poolType == llgd_lwn::PoolType::Compressed ||
            poolType == llgd_lwn::PoolType::CompressedPhysicalHasCompbits;
        const bool ownsGpuVa = !llgd_lwn::PoolUtil::IsPhysical(poolType);
        const bool ownsIoVa = !llgd_lwn::PoolUtil::IsVirtual(poolType);

        TEST(success);

        if (ownsCompbits) {
            TEST_EQ (vas1.comptags.start, vas2.comptags.start);
            TEST_NEQ(vas3.comptags.start, vas2.comptags.start);
            TEST_EQ (vas4.comptags.start, vas2.comptags.start);
        } else {
            TEST_EQ(vas1.comptags.size, 0);
            TEST_EQ(vas2.comptags.size, 0);
            TEST_EQ(vas3.comptags.size, 0);
            TEST_EQ(vas4.comptags.size, 0);
        }

        if (ownsGpuVa) {
            TEST_EQ (vas1.pitchGpuVa.start, vas2.pitchGpuVa.start);
            TEST_NEQ(vas3.pitchGpuVa.start, vas2.pitchGpuVa.start);
            TEST_EQ (vas4.pitchGpuVa.start, vas2.pitchGpuVa.start);
        } else {
            TEST_EQ(vas1.pitchGpuVa.size, 0);
            TEST_EQ(vas2.pitchGpuVa.size, 0);
            TEST_EQ(vas3.pitchGpuVa.size, 0);
            TEST_EQ(vas4.pitchGpuVa.size, 0);
        }

        if (ownsIoVa) {
            TEST_EQ (vas1.iova.start, vas2.iova.start);
            TEST_NEQ(vas3.iova.start, vas2.iova.start);
            TEST_EQ (vas4.iova.start, vas2.iova.start);
        } else {
            TEST_EQ(vas1.iova.size, 0);
            TEST_EQ(vas2.iova.size, 0);
            TEST_EQ(vas3.iova.size, 0);
            TEST_EQ(vas4.iova.size, 0);
        }

        return true;
    };

    for (int type = 0; type < static_cast<int>(llgd_lwn::PoolType::Last); ++type) {
        TEST(Test(static_cast<llgd_lwn::PoolType>(type)));
    }

    return true;
}

LLGD_DEFINE_TEST(ReservedVa, UNIT, LwError Execute() { return TestReservedVa() ? LwSuccess : LwError_IlwalidState; });
