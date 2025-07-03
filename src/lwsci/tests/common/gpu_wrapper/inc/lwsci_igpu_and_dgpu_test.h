/*
 * lwscibuf_igpu_and_dgpu_test.h
 *
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCI_IGPU_AND_DGPU_TEST_H
#define INCLUDED_LWSCI_IGPU_AND_DGPU_TEST_H

/* This header should be included in LwSci tests where the tests are
 * dependent on both iGPU and dGPU.
 */
class LwSciiGpuAnddGpuTest : public LwSciGpuTest
{
public:
    void SetUp() override
    {
        LwSciGpuTest::SetUp();

        // TODO: Colwert this to an RAII-style wrapper similar to IpcWrapper
        ASSERT_EQ(LwSciError_Success, initPlatformGpu(iGpuTstResource))
            << "Platform GPU(S) initialization failed";
        ASSERT_EQ(LwSciError_Success, initPlatformGpu(dGpuTstResource))
            << "Platform GPU(S) initialization failed";

        getGpuIdOfType(iGpuTstResource, testiGpuId,
            LwSciTestGpuType::LwSciTestGpuType_iGPU);
        getGpuIdOfType(dGpuTstResource, testdGpuId,
            LwSciTestGpuType::LwSciTestGpuType_dGPU);

        if ((isGpuInitialized(testiGpuId) == false) ||
        (isGpuInitialized(testdGpuId) == false)) {
            GTEST_SKIP() << "Test requires presence of iGPU and dGPU in the system";
        }

        ASSERT_EQ(LwSciError_Success, setGpuForTest(iGpuTstResource,
            testiGpuId));
        ASSERT_EQ(LwSciError_Success, setGpuForTest(dGpuTstResource,
            testdGpuId));
    }

    void TearDown() override
    {
        // TODO: Colwert this to an RAII-style wrapper similar to IpcWrapper
        ASSERT_EQ(deinitPlatformGpu(iGpuTstResource), LwSciError_Success);
        ASSERT_EQ(deinitPlatformGpu(dGpuTstResource), LwSciError_Success);

        LwSciGpuTest::TearDown();
    }

    LwSciRmGpuId testiGpuId = {};
    LwSciRmGpuId testdGpuId = {};
    GpuTestResourceHandle iGpuTstResource = nullptr;
    GpuTestResourceHandle dGpuTstResource = nullptr;
};

#endif // INCLUDED_LWSCI_IGPU_AND_DGPU_TEST_H
