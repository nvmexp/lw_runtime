/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciBuf CheetAh Platform Implementation</b>
 *
 * @b Description: This file implements LwSciBuf CheetAh platform APIs.
 */

#include "lwscibuf_dev_platform_tegra_priv.h"
#include "lwscicommon_os.h"

static LwSciError LwSciBufGpuDevOpen(
    LwSciBufAllGpuContext* allGpuContext)
{
    LwSciError sciErr = LwSciError_Success;
    LwError lwErr = LwError_Success;
    size_t i = 0U;

    LWSCI_FNENTRY("");

    allGpuContext->gpuLib = LwRmGpuLibOpen(NULL);
    if (NULL == allGpuContext->gpuLib) {
        sciErr = LwSciError_ResourceError;
        LWSCI_ERR_STR("LwRmGpuLibOpen failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* obtain list of GPU devices in the system */
    allGpuContext->gpuList = LwRmGpuLibListDevices(allGpuContext->gpuLib,
                            &allGpuContext->gpuListSize);
    if (0U == allGpuContext->gpuListSize) {
        /* It is okay if there are no GPUs in the system. Just provide info
         * that system has no GPUs.
         */
        LWSCI_INFO("No GPU devices found\n");
        allGpuContext->gpuList = NULL;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    allGpuContext->perGpuContext = LwSciCommonCalloc(allGpuContext->gpuListSize,
                                    sizeof(LwSciBufPerGpuContext));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == allGpuContext->perGpuContext) {
        sciErr = LwSciError_InsufficientMemory;
        LWSCI_ERR_ULONG("Could not allocate memory for LwSciBufPerGpuContext struct for GPU devices: \n",
            allGpuContext->gpuListSize);
        LWSCI_ERR_ULONG("lwsciErrCode: \n", (uint64_t)sciErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto cleanup_gpu;
    }

    for (i = 0U; i < allGpuContext->gpuListSize; i++) {
        lwErr = LwRmGpuDeviceOpen(allGpuContext->gpuLib,
                    allGpuContext->gpuList[i].deviceIndex, NULL,
                    &allGpuContext->perGpuContext[i].gpuDevice);

        if (LwError_Success != lwErr) {
            LWSCI_WARN("LwRmGpuDeviceOpen failed for device index: %d. LwError: %"PRId32"\n",
                allGpuContext->gpuList[i].deviceIndex, lwErr);
            allGpuContext->perGpuContext[i].gpuDevice = NULL;
            /* continue to initialize other GPU devices */
            continue;
        }

        allGpuContext->perGpuContext[i].gpuDeviceInfo = LwRmGpuDeviceGetInfo(
                            allGpuContext->perGpuContext[i].gpuDevice);

        if (NULL == allGpuContext->perGpuContext[i].gpuDeviceInfo) {
            LWSCI_WARN("Could not obtain device Info for GPU device index %d\n",
                allGpuContext->gpuList[i].deviceIndex);
            lwErr = LwRmGpuDeviceClose(
                        allGpuContext->perGpuContext[i].gpuDevice);
            if (LwError_Success != lwErr) {
                LWSCI_WARN("Failed to close GPU device index %" PRId32
                    " LwError: %" PRId32 "\n",
                     allGpuContext->gpuList[i].deviceIndex, lwErr);
            }
            allGpuContext->perGpuContext[i].gpuDevice = NULL;
        }
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

cleanup_gpu:
    allGpuContext->gpuListSize = 0U;

    allGpuContext->gpuList = NULL;

    lwErr = LwRmGpuLibClose(allGpuContext->gpuLib);
    if (LwError_Success != lwErr) {
        LWSCI_ERR_INT("Failed to close GpuLib in error path. LwError: \n", (int32_t)lwErr);
    }
    allGpuContext->gpuLib = NULL;

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static void LwSciBufGpuDevClose(
    LwSciBufAllGpuContext* allGpuContext)
{
    LwError lwErr = LwError_Success;
    size_t i = 0U;

    LWSCI_FNENTRY("");

    for (i = 0U; i < allGpuContext->gpuListSize; i++) {
        if (NULL != allGpuContext->perGpuContext[i].gpuDevice) {
            lwErr = LwRmGpuDeviceClose(
                        allGpuContext->perGpuContext[i].gpuDevice);
            if (LwError_Success != lwErr) {
                LWSCI_WARN("Failed to close GPU device index %" PRId32
                    " LwError: %" PRId32 "\n", i, lwErr);
            }
            allGpuContext->perGpuContext[i].gpuDeviceInfo = NULL;
        }
    }

    LwSciCommonFree(allGpuContext->perGpuContext);
    allGpuContext->perGpuContext = NULL;

    allGpuContext->gpuList = NULL;
    allGpuContext->gpuListSize = 0U;

    lwErr = LwRmGpuLibClose(allGpuContext->gpuLib);
    if (LwError_Success != lwErr) {
        LWSCI_WARN("LwRmGpuLibClose failed. LwError: %" PRId32 "\n", lwErr);
    }
    allGpuContext->gpuLib = NULL;

    LWSCI_FNEXIT("");
}

LwSciError LwSciBufDevOpen(
    LwSciBufDev* newDev)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufDev dev = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == newDev) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufDevOpen\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: LwSciBufDev* newDev: %p\n", newDev);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    dev = LwSciCommonCalloc(1, sizeof(LwSciBufDevPriv));
    if (NULL == dev) {
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufDevPriv struct\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufGpuDevOpen(&dev->allGpuContext);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_ULONG("Failed to open GPU device(s), lwsciErr \n",
            (uint64_t)sciErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_dev;
    }

    *newDev = dev;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_dev:
    LwSciCommonFree(dev);

ret:
    /* print output parameters */
    LWSCI_INFO("Output: *newDev: %p\n", newDev ? *newDev : 0);

    LWSCI_FNEXIT("");
    return sciErr;
}

void LwSciBufDevClose(
    LwSciBufDev dev)
{
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == dev) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufDevClose\n");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: dev\n", dev);

    LwSciBufGpuDevClose(&dev->allGpuContext);

    LwSciCommonFree(dev);

    LWSCI_FNEXIT("");
}
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
void LwSciBufDevGetAllGpuContext(
    LwSciBufDev dev,
    const LwSciBufAllGpuContext** allGpuContext)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == dev) || (NULL == allGpuContext)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufDevGetAllGpuContext.");
        LwSciCommonPanic();
    }

    /* print input parameters */
    LWSCI_INFO("Input: dev: %p lwGpuDevHandle: %p\n", dev, allGpuContext);

    *allGpuContext = &dev->allGpuContext;

    /* print output parameters */
    LWSCI_INFO("Output: allGpuContext: %p\n", *allGpuContext);

    LWSCI_FNEXIT("");
}

void LwSciBufDevGetGpuDeviceInfo(
    LwSciBufDev dev,
    LwSciRmGpuId gpuId,
    const LwRmGpuDeviceInfo** gpuDeviceInfo)
{
    size_t devIdx = 0U;
    LwSciBufAllGpuContext allGpuContext = {};
    const LwRmGpuDeviceInfo* tmpGpuDeviceInfo = NULL;

    LWSCI_FNENTRY("");

    if ((dev == NULL) || (gpuDeviceInfo == NULL)) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufDevGetGpuDeviceInfo.");
        LwSciCommonPanic();
    }

    *gpuDeviceInfo = NULL;

    allGpuContext = dev->allGpuContext;

    for (devIdx = 0U; devIdx < allGpuContext.gpuListSize; devIdx++) {
        tmpGpuDeviceInfo = allGpuContext.perGpuContext[devIdx].gpuDeviceInfo;

        if (LwSciCommonMemcmp(&tmpGpuDeviceInfo->deviceId.gid, &gpuId,
            sizeof(LwRmGpuDeviceGID)) == 0) {
            *gpuDeviceInfo = tmpGpuDeviceInfo;
            break;
        }
    }

    LWSCI_FNEXIT("");
}

LwSciError LwSciBufCheckPlatformVersionCompatibility(
    bool* platformCompatibility)
{
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == platformCompatibility) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufCheckPlatformVersionCompatibility\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input parameters */
    LWSCI_INFO("Input: platformCompatibility %p\n",platformCompatibility);

    *platformCompatibility = true;

    /* print output parameters */
    LWSCI_INFO("Output: platformCompatibility: %u\n", *platformCompatibility);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}
