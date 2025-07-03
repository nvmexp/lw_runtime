/*
 * lwsci_gpu_test.h
 *
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCI_GPU_TEST_H
#define INCLUDED_LWSCI_GPU_TEST_H

#include "gtest/gtest.h"
#include "lwsci_test_gpu_platform.h"
#include "lwscibuf.h"

class LwSciGpuTest
{
public:
    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
    }

    static inline bool isGpuInitialized(
        LwSciRmGpuId gpuId)
    {
        LwSciRmGpuId tmpGpuId = {};
        if (memcmp(&gpuId, &tmpGpuId, sizeof(gpuId))== 0) {
            return false;
        }

        return true;
    }

    void getGpuIdOfType(
        GpuTestResourceHandle tstResource,
        LwSciRmGpuId& gpuId,
        LwSciTestGpuType gpuType);
};

#endif // INCLUDED_LWSCI_GPU_TEST_H
