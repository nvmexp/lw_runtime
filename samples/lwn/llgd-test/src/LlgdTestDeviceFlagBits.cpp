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

#define TEST_FLAGS(flags, x)\
    if( !(x) ) { LlgdErr( "DeviceFlags = %X " #x " failed, file: %s line: %d \n", flags, __FILE__, __LINE__); return false; }

static const uint32_t s_testDeviceFlags[] = {
    0,
    0xFFFFFFFF,
    // all debug flags
    LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_0_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_1_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_3_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT | LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT,
    LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_0_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_1_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_3_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT,
    LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT,
    LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT | LWN_DEVICE_FLAG_ENABLE_SEPARATE_SAMPLER_TEXTURE_SUPPORT_BIT,
    LWN_DEVICE_FLAG_ENABLE_SEPARATE_SAMPLER_TEXTURE_SUPPORT_BIT,
    LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_3_BIT | LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT,
};
static const int s_numOfTestDeviceFlags = sizeof(s_testDeviceFlags) / sizeof(uint32_t);

class DeviceFlagBitsValidator {
public:
    void Initialize();
    bool Test();

private:
    bool TestGetterSetter();
    bool TestForceDebugLayerOff();

    llgd_lwn::DeviceHolder dh; // Mucking with a local device instead of the global one make our lives easier
};

void DeviceFlagBitsValidator::Initialize()
{
    dh.Initialize();
}

bool DeviceFlagBitsValidator::Test()
{
    if (!TestGetterSetter()) { return false; }
    if (!TestForceDebugLayerOff()) { return false; }
    return true;
}

bool DeviceFlagBitsValidator::TestGetterSetter()
{
    for (int i = 0; i < s_numOfTestDeviceFlags; ++i)
    {
        LWNdeviceFlagBits flags = static_cast<LWNdeviceFlagBits>(s_testDeviceFlags[i]);
        llgdLwnSetDeviceFlagBits(dh, flags);
        TEST_FLAGS(s_testDeviceFlags[i], llgdLwnGetDeviceFlagBits(dh) == flags);
    }
    return true;
}

bool DeviceFlagBitsValidator::TestForceDebugLayerOff()
{
    for (int i = 0; i < s_numOfTestDeviceFlags; ++i)
    {
        LWNdeviceFlagBits flags = static_cast<LWNdeviceFlagBits>(s_testDeviceFlags[i]);
        llgdLwnSetDeviceFlagBits(dh, flags);

        TEST_FLAGS(s_testDeviceFlags[i], llgdLwnDeviceForceDebugLayerOff(dh) == flags);

        LWNdeviceFlagBits newflags = llgdLwnGetDeviceFlagBits(dh);

        TEST_FLAGS(s_testDeviceFlags[i],
            !(newflags & LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_0_BIT) &&
            !(newflags & LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_1_BIT) &&
            !(newflags & LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT) &&
            !(newflags & LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_3_BIT) &&
            !(newflags & LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT));
    }
    return true;
}

LLGD_DEFINE_TEST(DeviceFlagBits, UNIT,
LwError Execute()
{
    DeviceFlagBitsValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
