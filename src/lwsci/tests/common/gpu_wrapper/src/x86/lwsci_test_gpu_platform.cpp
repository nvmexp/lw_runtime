/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwsci_test_gpu_platform.h"

#include <dlfcn.h>
#include <string.h>
#include "ctrl/ctrl0000.h"
#include "lwRmApi.h"
#include "lwscibuf_internal.h"
#include "lwscicommon_errcolwersion.h"
#include "lwscicommon_libc.h"
#include "lwscilog.h"
#include "lwstatus.h"

// In scibuf resman test logic, we need to open new resman context
// inside the test due to the static linking issue.
// Since we don't know which gpu is scibuf obj's dgpu mem handle attached to
// we need to attached all gpus here
static LW_STATUS clientAlloc(GpuTestResourceHandle res)
{
    LW_STATUS status = LW_OK;
    uint32_t i = 0;
    LwU32 gpuId = 0U;
    LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS probedParams;

    LwRmShimError errShim = LWRMSHIM_OK;
    res->lib_h = dlopen("liblwidia-allocator.so", RTLD_LAZY);
    if (res->lib_h == NULL) {
        status = 0;
        printf("[ERR] cant open lib\n");
        goto ret;
    }

    LwRmShimError (*sessionCreate)(LwRmShimSessionContext*);
    sessionCreate =  (LwRmShimError (*)(LwRmShimSessionContext*))
                      dlsym(res->lib_h, "LwRmShimSessionCreate");
    if (sessionCreate == NULL) {
        status = LW_ERR_ILWALID_STATE;
        printf("[ERR] sessionCreate null\n");
        goto ret;
    }

    errShim = sessionCreate(&res->session);
    if (errShim != LWRMSHIM_OK) {
        printf("[ERR] sessionCreate failed: errShim = %d\n", errShim);
        status = LW_ERR_ILWALID_STATE;
        goto ret;
    }

    status = LwRmAllocRoot(&res->hClient);
    if (status != LW_OK) {
        printf("%s: Alloc client failed, err - %u : %s\n", __FUNCTION__, status,
            LwStatusToString(status));
        goto ret;
    }

    /* Probe GPUs present on the system */
    memset(&probedParams, 0, sizeof(probedParams));
    status = LwRmControl(res->hClient, res->hClient,
                LW0000_CTRL_CMD_GPU_GET_PROBED_IDS, &probedParams,
                sizeof(probedParams));
    if (status != LW_OK) {
        printf("%s: Probe GPU failed, err - %u : %s\n", __FUNCTION__, status,
            LwStatusToString(status));
        goto ret;
    }

    for (i = 0; i < res->session.numGpus; i++) {
        DeviceInfo tmpInfo;
        tmpInfo.probed = true;
        tmpInfo.deviceId = res->session.gpuId[i];
        res->deviceInfo.push_back(tmpInfo);
    }

    /* Attach GPU */
    LW0000_CTRL_GPU_ATTACH_IDS_PARAMS attachParams;
    memset(&attachParams, 0, sizeof(attachParams));

    for (i = 0;
        i < res->deviceInfo.size(); i++) {
        attachParams.gpuIds[i] = res->deviceInfo[i].deviceId;
    }

    if (i != LW0000_CTRL_GPU_MAX_PROBED_GPUS) {
        attachParams.gpuIds[i] = LW0000_CTRL_GPU_ILWALID_ID;
    }

    status = LwRmControl(res->hClient, res->hClient,
                LW0000_CTRL_CMD_GPU_ATTACH_IDS,
                &attachParams, sizeof(attachParams));
    if (status != LW_OK) {
        printf("%s: Attach GPU failed, err - %u : %s\n", __FUNCTION__, status,
            LwStatusToString(status));
        goto ret;
    }

    /* Get device and subdevice instance */
    LW0000_CTRL_GPU_GET_ID_INFO_PARAMS idInfoParams;
    for (i = 0;
        i < res->deviceInfo.size(); i++) {
        memset(&idInfoParams, 0, sizeof(idInfoParams));
        idInfoParams.gpuId = res->deviceInfo[i].deviceId;
        status = LwRmControl(res->hClient, res->hClient,
                            LW0000_CTRL_CMD_GPU_GET_ID_INFO, &idInfoParams,
                            sizeof(idInfoParams));
        if (status != LW_OK) {
            printf("%s: Failed to get gpu info for number %u, deviceID %u, "
                    "error code = 0x%x - %s\n", __FUNCTION__, i,
                    res->deviceInfo[i].deviceId, status,
                    LwStatusToString(status));
            goto ret;
        }
        res->deviceInfo[i].deviceInstance = idInfoParams.deviceInstance;
        res->deviceInfo[i].subdeviceInstance = idInfoParams.subDeviceInstance;
    }

    res->m_gpuSize = res->session.numGpus;

    /* Get all uuids */
    for (i = 0; i < res->m_gpuSize; ++i) {
        LwSciRmGpuId tmpGpuId;
        LW0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS params = {0};
        params.flags =
                LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_BINARY;
        params.gpuId = res->deviceInfo[i].deviceId;
        status = LwRmControl(res->hClient, res->hClient,
                         LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID,
                         &params, sizeof(params));
        if (status != LW_OK) {
            printf("%s: get gpu uuid failed.\n", __FUNCTION__);
            goto ret;
        }

        memcpy(tmpGpuId.bytes, &res->session.gpuUUID[i].bytes[0], sizeof(LwSciRmGpuId));
        res->uuids.push_back(tmpGpuId);
    }

ret:
    return status;
}

LwSciError initPlatformGpu(
    GpuTestResourceHandle& tstResourceHandle)
{
    LwSciError sciErr = LwSciError_Success;
    LW_STATUS status = LW_OK;

    tstResourceHandle = std::make_shared<GpuTestResource>();

    status = clientAlloc(tstResourceHandle);
    sciErr = LwStatusToLwSciErr(status);
    if (sciErr != LwSciError_Success) {
        printf("clientAlloc failed error code = 0x%x - %s\n", status,
            LwStatusToString(status));
        goto free_test_resource;
    }

    goto ret;

free_test_resource:
    tstResourceHandle.reset();

ret:
    return sciErr;
}

LwSciError deinitPlatformGpu(
    GpuTestResourceHandle tstResource)
{
    LwSciError err = LwSciError_Success;

    tstResource.reset();

    return err;
}

LwSciError setGpuForTest(
    GpuTestResourceHandle tstResource,
    LwSciRmGpuId gpuId)
{
    LwSciError err = LwSciError_Success;

    for (uint32_t i = 0; i < tstResource->m_gpuSize; i++) {
        if (memcmp(&tstResource->uuids[i], &gpuId, sizeof(gpuId)) == 0) {
            tstResource->m_gpuId = i;
            goto ret;
        }
    }

    /* We are here implying that GPU passed as input parameter was not found */
    err = LwSciError_BadParameter;

ret:
    return err;
}

void getAllGpus(
    GpuTestResourceHandle res,
    std::vector<LwSciRmGpuId>& allGpuIds)
{
    allGpuIds = res->uuids;
}

LwSciError getGpuType(
    GpuTestResourceHandle res,
    LwSciRmGpuId gpu,
    LwSciTestGpuType& gpuType)
{
    LwSciError err = LwSciError_Success;
    size_t i = 0U;

    for (i = 0U; i < res->m_gpuSize; i++) {
        if (memcmp(&gpu, &res->uuids[i], sizeof(LwSciRmGpuId)) == 0U) {
            /* On x86 there is no iGPU. Always return type = dGPU */
            gpuType = LwSciTestGpuType::LwSciTestGpuType_dGPU;
            break;
        }
    }

    if (i == res->m_gpuSize) {
        /* This should not happen */
        err = LwSciError_IlwalidState;
    }
}

LwSciError isGpuKindCompressible(
    GpuTestResourceHandle tstResource,
    LwSciRmGpuId gpuId,
    bool isBlockLinear,
    bool* isCompressible)
{
    LwSciError err = LwSciError_Success;
    (void)tstResource;
    (void)gpuId;
    (void)isBlockLinear;

    /* PLC compression is disabled on GA100 so always return false here.
     * We will have to change this once we start supporting other dGPUs having
     * compression enabled
     */
    *isCompressible = false;

    return err;
}
