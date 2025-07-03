/*
 * lwscibuf_igpu_or_dgpu_test.h
 *
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCI_IGPU_OR_DGPU_TEST_H
#define INCLUDED_LWSCI_IGPU_OR_DGPU_TEST_H

#include "lwsci_gpu_test.h"

/* This header should be included in LwSci tests where the tests are
 * dependent on either of iGPU or dGPU.
 */
class LwSciiGpuOrdGpuTest : public LwSciGpuTest
{
public:
    void SetUp() override
    {
        LwSciRmGpuId testiGpuId {};
        LwSciRmGpuId testdGpuId {};

        LwSciGpuTest::SetUp();

        // TODO: Colwert this to an RAII-style wrapper similar to IpcWrapper
        ASSERT_EQ(LwSciError_Success, initPlatformGpu(tstResource))
            << "Platform GPU(S) initialization failed";

        getGpuIdOfType(tstResource, testiGpuId,
            LwSciTestGpuType::LwSciTestGpuType_iGPU);
        getGpuIdOfType(tstResource, testdGpuId,
            LwSciTestGpuType::LwSciTestGpuType_dGPU);

        if (isGpuInitialized(testiGpuId) == true) {
            testGpuId = testiGpuId;
        } else if (isGpuInitialized(testdGpuId) == true) {
            testGpuId = testdGpuId;
        } else {
            GTEST_SKIP() << "Test requires presence of iGPU or dGPU in the system";
        }

        ASSERT_EQ(LwSciError_Success, setGpuForTest(tstResource, testGpuId));
    }

    void TearDown() override
    {
        // TODO: Colwert this to an RAII-style wrapper similar to IpcWrapper
        ASSERT_EQ(deinitPlatformGpu(tstResource), LwSciError_Success);

        LwSciGpuTest::TearDown();
    }

    LwSciRmGpuId testGpuId = {};
    GpuTestResourceHandle tstResource = nullptr;
};

#endif // INCLUDED_LWSCI_IGPU_OR_DGPU_TEST_H
