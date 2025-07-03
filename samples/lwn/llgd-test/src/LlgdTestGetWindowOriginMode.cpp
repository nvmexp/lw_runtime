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

#if defined(LW_LINUX)
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

class GetWindowOriginModeValidator {
public:
    void Initialize();
    bool Test();

private:
    bool TestWindowOriginMode(WindowOriginMode mode);

    llgd_lwn::DeviceHolder dh; // Mucking with a local device instead of the global one make our lives easier
};

void GetWindowOriginModeValidator::Initialize()
{
    dh.Initialize();
}

bool GetWindowOriginModeValidator::TestWindowOriginMode(WindowOriginMode winOriginMode)
{
    llgd_lwn::QueueHolder qh;

    // Window origin mode is applied to gpu state in lwnQueueInitialize
    dh->SetWindowOriginMode(winOriginMode);
    qh.Initialize(dh); // QueueHolder::Initialize has a Queue::Finish

    LWNwindowOriginMode mode = *reinterpret_cast<LWNwindowOriginMode*>(&winOriginMode);
    TEST_EQ_FMT(llgdLwnGetWindowOriginMode(qh), mode, "WindowOriginMode = %d", mode);
    TEST_EQ_FMT(llgdGetDeviceWindowOriginMode(dh), mode, "WindowOriginMode = %d", mode);

    const auto gpuState = llgd_lwn::ExtractGpuState(qh);
    TEST_EQ_FMT(llgdLwnExtractWindowOriginMode(gpuState), mode, "WindowOriginMode = %d", mode);
    return true;
}

bool GetWindowOriginModeValidator::Test()
{
    if (!TestWindowOriginMode(WindowOriginMode::LOWER_LEFT)) {
        return false;
    }
    if (!TestWindowOriginMode(WindowOriginMode::UPPER_LEFT)) {
        return false;
    }
    return true;
}

LLGD_DEFINE_TEST(GetWindowOriginMode, UNIT,
LwError Execute()
{
    GetWindowOriginModeValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
