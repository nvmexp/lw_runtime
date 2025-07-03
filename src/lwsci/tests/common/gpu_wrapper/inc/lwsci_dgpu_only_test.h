/*
 * lwscibuf_dgpu_only_test.h
 *
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCI_DGPU_ONLY_TEST_H
#define INCLUDED_LWSCI_DGPU_ONLY_TEST_H

#include "lwsci_gpu_test.h"

/* This header should be included in LwSci tests where the tests are
 * dependent on dGPU only.
 */
class LwScidGpuOnlyTest : public LwSciGpuTest
{
public:
    void SetUp() override
    {
        LwSciGpuTest::SetUp();

        // TODO: Colwert this to an RAII-style wrapper similar to IpcWrapper
        ASSERT_EQ(LwSciError_Success, initPlatformGpu(dGpuTstResource))
            << "Platform GPU(S) initialization failed";

        getGpuIdOfType(dGpuTstResource, testdGpuId,
            LwSciTestGpuType::LwSciTestGpuType_dGPU);

        if (isGpuInitialized(testdGpuId) == false) {
            GTEST_SKIP() << "Test requires presence of dGPU in the system";
        }

        ASSERT_EQ(LwSciError_Success, setGpuForTest(dGpuTstResource,
            testdGpuId));
    }

    void TearDown() override
    {
        // TODO: Colwert this to an RAII-style wrapper similar to IpcWrapper
        ASSERT_EQ(deinitPlatformGpu(dGpuTstResource), LwSciError_Success);

        LwSciGpuTest::TearDown();
    }

    LwSciRmGpuId testdGpuId = {};
    GpuTestResourceHandle dGpuTstResource = nullptr;
};

#endif // INCLUDED_LWSCI_DGPU_ONLY_TEST_H
