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

class GetDepthModeValidator {
public:
    void Initialize();
    bool Test();

private:
    bool TestDepthMode(DepthMode depthMode);

    llgd_lwn::DeviceHolder dh; // Mucking with a local device instead of the global one make our lives easier
};

void GetDepthModeValidator::Initialize()
{
    dh.Initialize();
}

bool GetDepthModeValidator::TestDepthMode(DepthMode depthMode)
{
    llgd_lwn::QueueHolder qh;

    // Depth mode is applied to gpu state in lwnQueueInitialize
    dh->SetDepthMode(depthMode);
    qh.Initialize(dh); // QueueHolder::Initialize has a Queue::Finish

    const auto gpuState = llgd_lwn::ExtractGpuState(qh);

    LWNdepthMode  mode = *reinterpret_cast<LWNdepthMode*>(&depthMode);
    TEST_EQ_FMT(llgdLwnGetDepthMode(qh), mode, "DepthMode = %d", mode);
    TEST_EQ_FMT(llgdGetDeviceDepthMode(dh), mode, "DepthMode = %d", mode);
    TEST_EQ_FMT(llgdLwnExtractDepthMode(gpuState), mode, "DepthMode = %d", mode);
    return true;
}

bool GetDepthModeValidator::Test()
{
    if (!TestDepthMode(DepthMode::NEAR_IS_MINUS_W)) {
        return false;
    }
    if (!TestDepthMode(DepthMode::NEAR_IS_ZERO)) {
        return false;
    }
    return true;
}

LLGD_DEFINE_TEST(GetDepthMode, UNIT,
LwError Execute()
{
    GetDepthModeValidator v;
    v.Initialize();

    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
