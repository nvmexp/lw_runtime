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

#include "lwnExt/lwnExt_Internal.h"
#include <lwndevtools_bootstrap.h>

extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);

namespace {
#define GET(type_, name_)                                                \
    static const auto name_ = (type_)cppGetProcAddress(nullptr, #name_); \
    TEST(name_)

struct DeviceGuard {
    lwn::Device *device_ = nullptr;
    DeviceGuard(lwn::Device *device) : device_{ device } {}
    ~DeviceGuard() { if (device_) { device_->Finalize(); } }
};

static const float BIAS = 3.14;

bool called = false;
static void LWNAPIENTRY interceptSamplerBuilderSetLodBias(LWNsamplerBuilder *builder, float bias)
{
    CHECK(bias == BIAS)
    called = true;
}

bool called2 = false;
static void LWNAPIENTRY intercept2SamplerBuilderSetLodBias(LWNsamplerBuilder *builder, float bias)
{
    CHECK(bias == BIAS)
    called2 = true;
}

bool TestInterception()
{
    // Lwn interception / API infrastructure.
    static const auto devtools = lwnDevtoolsBootstrap();

    static const auto getProcAddress = lwnBootstrapLoader("lwnDeviceGetProcAddress");
    static const auto cppGetProcAddress = reinterpret_cast<DeviceGetProcAddressFunc>(getProcAddress);
    TEST(getProcAddress)

    GET(PFNLWNINTERCEPTIONINITIALIZELWXPROC, lwnInterceptionInitializeLWX)
    GET(PFNLWNINTERCEPTIONSETPROCADDRESSLWXPROC, lwnInterceptionSetProcAddressLWX)
    GET(PFNLWNINTERCEPTIONSETDISPATCHRESETCALLBACKLWXPROC, lwnInterceptionSetDispatchResetCallbackLWX)
    GET(PFNLWNINTERCEPTIONTRYACQUIRELWXPROC, lwnInterceptionTryAcquireLWX)
    GET(PFNLWNINTERCEPTIONOWNEDLWXPROC, lwnInterceptionOwnedLWX)
    GET(PFNLWNINTERCEPTIONRELEASELWXPROC, lwnInterceptionReleaseLWX)
    GET(PFNLWNINTERCEPTIONFINALIZELWXPROC, lwnInterceptionFinalizeLWX)

    // Test infrastructure.
    static const char* name = "lwnSamplerBuilderSetLodBias";

    auto checkCallSamplerBuilderSetLodBias = [](bool expectCall, bool expectCall2) {
        called = false, called2 = false;
        lwn::SamplerBuilder testSubject;
        testSubject.SetLodBias(BIAS);
        TEST_EQ(called, expectCall)
        TEST_EQ(called2, expectCall2)
        return true;
    };

    static LWNinterception layer; // static because of resetCb.
    LWNinterception layer2; // no resetCb here.

    // Make a device with debug layer.
    lwn::Device device;
    lwn::DeviceBuilder db;
    db.SetDefaults();
    db.SetFlags(DeviceFlagBits::DEBUG_ENABLE_LEVEL_4);
    {
        const auto success = device.Initialize(&db);
        TEST(success);
    }
    DeviceGuard dg{ &device };
    const auto debugLayerSamplerBuilderSetLodBias = device.GetProcAddress(name);

    // Start building the interception.
    lwnInterceptionInitializeLWX(&layer);
    const auto driverLayerSamplerBuilderSetLodBias = device.GetProcAddress(name);
    TEST_NEQ(debugLayerSamplerBuilderSetLodBias, driverLayerSamplerBuilderSetLodBias)

    // Update PFNs.
    lwnLoadCPPProcs(&device, cppGetProcAddress);

    // Build layer.
    void* originalSamplerBuilderSetLodBias = nullptr;
    lwnInterceptionSetProcAddressLWX(&layer, name, (void*)&interceptSamplerBuilderSetLodBias, &originalSamplerBuilderSetLodBias);
    TEST_EQ(originalSamplerBuilderSetLodBias, debugLayerSamplerBuilderSetLodBias)
    TEST(checkCallSamplerBuilderSetLodBias(false, false))

    static auto resetCb = [] (LWNdevice* device) {
        lwnInterceptionSetProcAddressLWX(&layer, name, (void*)&interceptSamplerBuilderSetLodBias, nullptr);
        if (lwnInterceptionOwnedLWX(&layer)) {
            const auto reacquire = lwnInterceptionTryAcquireLWX(&layer);
            CHECK(reacquire);
        }
    };
    lwnInterceptionSetDispatchResetCallbackLWX(&layer, resetCb, nullptr);

    // Activate layer.
    const auto active = lwnInterceptionTryAcquireLWX(&layer);
    TEST(active)
    TEST(checkCallSamplerBuilderSetLodBias(true, false))

    // Second device, very annoying, resets PFNs.
    {
        lwn::Device device2;
        lwn::DeviceBuilder db2;
        db2.SetDefaults();
        {
            const auto success = device2.Initialize(&db2);
            TEST(success);
        }

        DeviceGuard dg2{ &device2 };
        lwnLoadCPPProcs(&device2, cppGetProcAddress);
        TEST(checkCallSamplerBuilderSetLodBias(true, false))
    }

    // Second device just died.
    TEST(checkCallSamplerBuilderSetLodBias(true, false))

    // Start building layer2
    lwnInterceptionInitializeLWX(&layer2);
    lwnInterceptionSetProcAddressLWX(&layer2, name, (void*)&intercept2SamplerBuilderSetLodBias, nullptr);

    // layer1 still alive, can't acquire.
    const auto busyFailed = lwnInterceptionTryAcquireLWX(&layer2);
    TEST_EQ(busyFailed, false)
    TEST(checkCallSamplerBuilderSetLodBias(true, false))

    // Release layer.
    lwnInterceptionReleaseLWX(&layer);
    TEST(checkCallSamplerBuilderSetLodBias(false, false))

    // Acquire layer2.
    const auto active2 = lwnInterceptionTryAcquireLWX(&layer2);
    TEST(active2)
    TEST(checkCallSamplerBuilderSetLodBias(false, true))
    const auto owned2 = lwnInterceptionOwnedLWX(&layer2);
    TEST(owned2);

    // And reacquire.
    const auto reactive2 = lwnInterceptionTryAcquireLWX(&layer2);
    TEST(reactive2)
    TEST(checkCallSamplerBuilderSetLodBias(false, true))

    // We release all layers and check that layer2 keeps
    // the interception layer alive in the driver.
    lwnInterceptionReleaseLWX(&layer2);
    lwnInterceptionFinalizeLWX(&layer);
    const auto driver2LayerSamplerBuilderSetLodBias = device.GetProcAddress(name);
    TEST_EQ(driverLayerSamplerBuilderSetLodBias, driver2LayerSamplerBuilderSetLodBias)

    // Check that we have 3 varieties of pointers when DL is enabled in the build,
    // only 2 varieties when no DL: DL pointer, LWN pointer, interception pointer.
    const auto devtoolsSamplerBuilderSetLodBias = devtools->InterceptionGetProcAddress(nullptr, name);
    TEST_NEQ(driverLayerSamplerBuilderSetLodBias, devtoolsSamplerBuilderSetLodBias)

    int supportsDebugLayer = 1;
    device.GetInteger(DeviceInfo::SUPPORTS_DEBUG_LAYER, &supportsDebugLayer);
    if (supportsDebugLayer) {
        const auto rawLwnSamplerBuilderSetLodBias = devtools->GetNonDebugProcAddress(name);
        TEST_NEQ(driverLayerSamplerBuilderSetLodBias, rawLwnSamplerBuilderSetLodBias)
        TEST_NEQ(devtoolsSamplerBuilderSetLodBias, rawLwnSamplerBuilderSetLodBias)
    }

    // When all layers are finalized, we go back to business as usual,
    // not interception layer active.
    lwnInterceptionFinalizeLWX(&layer2);
    const auto reSamplerBuilderSetLodBias = device.GetProcAddress(name);
    TEST_EQ(debugLayerSamplerBuilderSetLodBias, reSamplerBuilderSetLodBias)

    // Restore PFNs state.
    device.Finalize();
    dg.device_ = nullptr;
    lwnLoadCPPProcs(g_device, cppGetProcAddress);

    return true;
}
}

LLGD_DEFINE_TEST(Interception, UNIT, LwError Execute() { return TestInterception() ? LwSuccess : LwError_IlwalidState; });
