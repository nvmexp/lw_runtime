/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwsci_gpu_test.h"

void LwSciGpuTest::getGpuIdOfType(
        GpuTestResourceHandle tstResource,
        LwSciRmGpuId& gpuId,
        LwSciTestGpuType gpuType)
{
    LwSciError err = LwSciError_Success;
    std::vector<LwSciRmGpuId> tstGpuId;
    size_t i = 0U;
    LwSciTestGpuType tstGpuType;

    getAllGpus(tstResource, tstGpuId);

    for (i = 0U; i < tstGpuId.size(); i++) {
        err = getGpuType(tstResource, tstGpuId[i], tstGpuType);
        ASSERT_EQ(err, LwSciError_Success);

        if (tstGpuType == gpuType) {
            gpuId = tstGpuId[i];
            return;
        }
    }

    /* GPU ID for given gpu type not found */
}
