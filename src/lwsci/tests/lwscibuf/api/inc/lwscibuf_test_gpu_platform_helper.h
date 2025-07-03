/*
 * Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_TEST_GPU_PLATFORM_HELPER_H
#define INCLUDED_LWSCIBUF_TEST_GPU_PLATFORM_HELPER_H

#include "lwscibuf_internal.h"
#include "lwsci_test_gpu_platform.h"

LwSciError testGpuMapping(
    GpuTestResourceHandle tstResource,
    LwSciBufRmHandle rmHandle,
    uint64_t rawBufSize);

LwSciError testDeviceCpuMapping(
    GpuTestResourceHandle tstResource,
    LwSciBufObj srcBufObj,
    LwSciBufRmHandle srcRmHandle,
    LwSciBufObj dstBufObj,
    LwSciBufRmHandle dstRmHandle,
    uint64_t dstOffset,
    uint64_t dstLen,
    uint64_t memSize);

#endif
