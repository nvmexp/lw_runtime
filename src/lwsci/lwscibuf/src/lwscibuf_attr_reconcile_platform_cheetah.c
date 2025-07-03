/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_attr_reconcile_platform.h"
#include "lwscibuf_attr_mgmt.h"

LwSciError LwSciBufValidateGpuType(
    LwSciBufAttrList attrList,
    LwSciRmGpuId gpuId,
    LwSciBufGpuType gpuType,
    bool* match
)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule module = NULL;
    LwSciBufDev devHandle = NULL;
    const LwRmGpuDeviceInfo* gpuDeviceInfo = NULL;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListValidate failed.");
        goto ret;
    }

    if ((match == NULL) || (gpuType < LwSciBufGpuType_iGPU) ||
        (gpuType > LwSciBufGpuType_dGPU)) {
        LwSciCommonPanic();
    }
    *match = false;

    err = LwSciBufAttrListGetModule(attrList, &module);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get module\n");
        goto ret;
    }

    err = LwSciBufModuleGetDevHandle(module, &devHandle);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get device handle\n");
        goto ret;
    }

    LwSciBufDevGetGpuDeviceInfo(devHandle, gpuId, &gpuDeviceInfo);
    if (gpuDeviceInfo == NULL) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("GPU ID supplied in the attribute list is not present in the system.");
        goto ret;
    }

    if (((gpuDeviceInfo->gpuType == LwRmGpuType_iGPU) &&
        (gpuType == LwSciBufGpuType_iGPU)) ||
        ((gpuDeviceInfo->gpuType == LwRmGpuType_dGPU) &&
        (gpuType == LwSciBufGpuType_dGPU))) {

        *match = true;
    }

    LWSCI_FNEXIT("");

ret:
    return err;
}
