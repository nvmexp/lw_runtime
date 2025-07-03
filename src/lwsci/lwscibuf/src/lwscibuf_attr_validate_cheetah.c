/*
 * Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscibuf_attr_validate.h"
#include "lwscibuf_attr_mgmt.h"

static LwSciError getDevHandle(
    LwSciBufAttrList attrList,
    LwSciBufDev* devHandle)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule module = NULL;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListGetModule(attrList, &module);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get module\n");
        goto ret;
    }

    err = LwSciBufModuleGetDevHandle(module, devHandle);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get device handle\n");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufValidateGpuId(
    const LwSciBufAttrList attrList,
    const void* val)
{
    LwSciError err = LwSciError_Success;
    LwSciBufDev devHandle = NULL;
    const LwRmGpuDeviceInfo* gpuDeviceInfo = NULL;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LwSciRmGpuId gpuId = *(const LwSciRmGpuId *)val;
    LWSCI_FNENTRY("");

    err = getDevHandle(attrList, &devHandle);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("getDevHandle() failed.");
        goto ret;
    }

    LwSciBufDevGetGpuDeviceInfo(devHandle, gpuId, &gpuDeviceInfo);
    if (gpuDeviceInfo == NULL) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("GPU ID supplied in the attribute list is not present in the system.");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufValidateAttrValGpuCache(
    LwSciBufAttrList attrList,
    const void* val)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrValGpuCache gpuCacheVal = *(const LwSciBufAttrValGpuCache*)val;

    LWSCI_FNENTRY("");

    err = LwSciBufValidateGpuId(attrList, &gpuCacheVal.gpuId);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufValidateGpuId failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT();
    return err;
}

LwSciError LwSciBufValidateAttrValGpuCompressionInternal(
    LwSciBufAttrList attrList,
    const void* val)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrValGpuCompression gpuCompression =
        *(const LwSciBufAttrValGpuCompression*)val;

    LWSCI_FNENTRY("");

    err = LwSciBufValidateGpuId(attrList, &gpuCompression.gpuId);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufValidateGpuId failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufValidateGpuCompressionInternal(attrList,
            &gpuCompression.compressionType);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufValidateGpuCompression failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufValidateAttrValGpuCompressionExternal(
    LwSciBufAttrList attrList,
    const void* val)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrValGpuCompression gpuCompression =
        *(const LwSciBufAttrValGpuCompression*)val;

    LWSCI_FNENTRY("");

    err = LwSciBufValidateGpuId(attrList, &gpuCompression.gpuId);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufValidateGpuId failed.");
        goto ret;
    }

    err = LwSciBufValidateGpuCompressionExternal(attrList,
            &gpuCompression.compressionType);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufValidateGpuCompression failed.");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
