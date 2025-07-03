/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCI_TEST_GPU_PLATFORM_H
#define INCLUDED_LWSCI_TEST_GPU_PLATFORM_H

#include "lwsci_test_gpu_platform_specific.h"
#include "lwscierror.h"
#include "lwscibuf.h"
#include "lwscibuf_internal.h"
#include <memory>
#include <vector>

#define CHECK_LW_API(err, func, cleanup)                                       \
    ({                                                                         \
        LwError lwErr = (func);                                                \
        if (lwErr != LwError_Success) {                                        \
            LWSCI_ERR_STR(#func);                                              \
            err = LwSciError_ResourceError;                                    \
            LWSCI_ERR_HEXUINT(" failed: ", err);                               \
            LWSCI_ERR_HEXUINT("\n", lwErr);                                    \
            cleanup;                                                           \
        }                                                                      \
    })
using GpuTestResourceHandle = std::shared_ptr<GpuTestResourceRec>;

enum class LwSciTestGpuType {
    LwSciTestGpuType_iGPU,
    LwSciTestGpuType_dGPU,
};

LwSciError setGpuForTest(
    GpuTestResourceHandle tstResource,
    LwSciRmGpuId gpuId);

LwSciError initPlatformGpu(
    GpuTestResourceHandle& tstResource);

LwSciError deinitPlatformGpu(
    GpuTestResourceHandle tstResource);

void getAllGpus(
    GpuTestResourceHandle tstResource,
    std::vector<LwSciRmGpuId>& allGpuIds);

LwSciError getGpuType(
    GpuTestResourceHandle tstResource,
    LwSciRmGpuId gpu,
    LwSciTestGpuType& gpuType);

LwSciError isGpuKindCompressible(
    GpuTestResourceHandle tstResource,
    LwSciRmGpuId gpuId,
    bool isBlockLinear,
    bool* isCompressible);

#endif
