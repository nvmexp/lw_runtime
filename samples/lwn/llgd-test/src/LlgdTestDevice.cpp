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

#include <lwndevtools_bootstrap.h>

#include <iostream>

static bool TestDevice()
{
    llgd_lwn::QueueHolder qh;
    qh.Initialize(g_device);
    // We need to have one live queue for some of the Getter things in
    // devtools bootstrap land.

    const auto devtools = lwnDevtoolsBootstrap();

    // Please don't crash.
    {
        lwn::Device device;
        lwn::DeviceBuilder db;
        db.SetDefaults();
        devtools->ReplayDeviceInitializePartOne(reinterpret_cast<LWNdevice*>(&device), reinterpret_cast<LWNdeviceBuilder*>(&db));
        // No part II !
        device.Finalize();
    }

    auto NextIova = [&] () {
        using Flags = lwn::MemoryPoolFlags;
        const auto poolFlags = Flags::CPU_UNCACHED | Flags::GPU_CACHED;

        const size_t ONE_PAGE = 65536;
        const size_t ALIGNT = LWN_MEMORY_POOL_STORAGE_ALIGNMENT;

        auto storage = LlgdAlignedAllocPodType<uint8_t>(ONE_PAGE, ALIGNT);

        LWNdevtoolsReservedVa reserved{ 0 };
        llgd_lwn::MemoryPoolHolder mph;

        MemoryPoolBuilder mpb;
        mpb.SetDevice(g_device).SetDefaults()
           .SetFlags(poolFlags)
           .SetStorage(storage.get(), ONE_PAGE);
        if (!mph.Initialize(&mpb)) { __builtin_trap(); }

        reserved = devtools->MemoryPoolProbeVas(mph);
        mph.Finalize();

        return reserved.iova.start;
    };

    // Make half & check iova unchanged
    const auto iova0 = NextIova();

    std::unique_ptr<lwn::Device, std::function<void(lwn::Device*)>> device{ new lwn::Device{}, [] (lwn::Device *d) { d->Finalize(); } };

    lwn::DeviceBuilder db;
    db.SetDefaults();

    auto lwnDevice = reinterpret_cast<LWNdevice*>(device.get());
    auto lwnDeviceBuilder = reinterpret_cast<LWNdeviceBuilder*>(&db);

    const auto partOne = devtools->ReplayDeviceInitializePartOne(lwnDevice, lwnDeviceBuilder);
    TEST(partOne);
    auto nextIova = NextIova();
    TEST_EQ(nextIova, iova0);

    // Make full & check changed
    const auto partTwo = devtools->ReplayDeviceInitializePartTwo(lwnDevice, lwnDeviceBuilder, nullptr, 3);
    TEST(partTwo);
    nextIova = NextIova();
    TEST_NEQ(nextIova, iova0);

    return true;
}

LLGD_DEFINE_TEST(Device, UNIT, LwError Execute() { return TestDevice() ? LwSuccess : LwError_IlwalidState; });
