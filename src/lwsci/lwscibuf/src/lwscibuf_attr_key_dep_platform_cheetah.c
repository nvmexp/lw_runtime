/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_attr_key_dep_platform.h"

/* TODO:the first dimension of array is being callwlated as LwRmGpuType_dGPU + 1
 * below. This is not good since if LwGpu adds another GPU types, we will end
 * up changing this code too. Lets ask LwGpu team to add something like,
 * LwRmGpuType_Num as the enum to identify number of GPU types on cheetah and
 * use that enum here. Right now, this enum is not supported by LwGpu team.
 */
static const bool
    defaultGpuCacheValue[LwRmGpuType_dGPU + 1][LwSciBufMemDomain_UpperBound] = {
    [LwRmGpuType_iGPU][LwSciBufMemDomain_Sysmem] = true,
    [LwRmGpuType_dGPU][LwSciBufMemDomain_Sysmem] = false,
#if (LW_IS_SAFETY == 0)
    [LwRmGpuType_dGPU][LwSciBufMemDomain_Vidmem] = true,
#endif //(LW_IS_SAFETY == 0)
};

/* TODO: The attribute key dependency is part of reconciliation.
 * Reconciliation is platform agnostic operation. The heap types that we are
 * setting here are cheetah specific and needed when passing heap type to
 * LwRm layer. We should remove setting heap type altogether from here and
 * move it to 'CheetAh sysmem interface' OR 'CheetAh common interface' unit where
 * these units figure out the heap type the same way it is figured out here.
 * We should eventually remove this platform specific file. We are not moving
 * this immediately to the units mentioned above since that changes unit
 * dependency and thus needs SWAD updates. We should take up this activity in
 * separate task.
 */
LwSciError LwSciBufAttrListPlatformHeapDependency(
    LwSciBufAttrList attrList,
    LwSciBufHeapType* heapType)
{
    LwSciError err = LwSciError_Success;
    LwSciBufInternalAttrKeyValuePair intKeyValPair = {0};
#if (LW_IS_SAFETY == 0)
    LwSciBufMemDomain memDomailwal = LwSciBufMemDomain_UpperBound;
#endif //LW_IS_SAFETY == 0
    bool isIsoEngine = false;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("attrList is invalid.");
        goto ret;
    }

#if (LW_IS_SAFETY == 0)
    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_MemDomainArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &intKeyValPair, 1,
            LwSciBufAttrKeyType_Internal, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get LwSciBufInternalGeneralAttrKey_MemDomainArray key");
        goto ret;
    }

    memDomailwal = *(const LwSciBufMemDomain*)intKeyValPair.value;
#endif //LW_IS_SAFETY == 0

    err = LwSciBufAttrListIsIsoEngine(attrList, &isIsoEngine);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListIsIsoEngine failed");
        goto ret;
    }

#if (LW_IS_SAFETY == 0)
    if (memDomailwal == LwSciBufMemDomain_Cvsram) {
        *heapType = LwSciBufHeapType_CvsRam;
    } else
#endif //LW_IS_SAFETY == 0
    if (isIsoEngine == true) {
#if (LW_IS_SAFETY == 0)
        *heapType = LwSciBufHeapType_ExternalCarveout;
#else //LW_IS_SAFETY == 0
        bool containsVi = false;
        bool containsDisplay = false;
        size_t engineCount = 0UL;
        uint64_t len = 0UL;
        LwSciBufHwEngine engineList[LW_SCI_BUF_HW_ENGINE_MAX_NUMBER] = {0};

        intKeyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;
        err = LwSciBufAttrListCommonGetAttrs(attrList, 0UL, &intKeyValPair,
                            1U, LwSciBufAttrKeyType_Internal, true);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
            LwSciCommonPanic();
        }

        len = (uint64_t)intKeyValPair.len;
        LwSciCommonMemcpyS(engineList,
                        sizeof(LwSciBufHwEngine)*LW_SCI_BUF_HW_ENGINE_MAX_NUMBER,
                        intKeyValPair.value,
                        len);

        engineCount = len / sizeof(LwSciBufHwEngine);

        err = LwSciBufHasEngine(engineList, engineCount, LwSciBufHwEngName_Vi, &containsVi);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to verify if the attribute list constain engine ", LwSciBufHwEngName_Vi);
            LwSciCommonPanic();
        }

        err = LwSciBufHasEngine(engineList, engineCount, LwSciBufHwEngName_Display, &containsDisplay);
        if (LwSciError_Success != err) {
            LWSCI_ERR_UINT("Failed to verify if the attribute list constain engine ", LwSciBufHwEngName_Display);
            LwSciCommonPanic();
        }

        if (containsVi) {
            /* Whenever Vi is provided, the allocation must come from IVC */
            *heapType = LwSciBufHeapType_IVC;
        } else if (containsDisplay) {
            /* Vi was not requested, so Display will be allocated from the
             * external carveout */
            *heapType = LwSciBufHeapType_ExternalCarveout;
        } else {
            LwSciCommonPanic();
        }
#endif //LW_IS_SAFETY == 0
    } else {
        *heapType = LwSciBufHeapType_IOMMU;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListPlatformGpuIdDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule module = NULL;
    LwSciBufDev devHandle = NULL;
    const LwRmGpuDeviceInfo* gpuDeviceInfo = NULL;
    size_t gpuIdx = 0U;
    size_t numGpus = 0U;
    size_t numiGpus = 0U;
    const LwSciRmGpuId* gpuIdPtr = NULL;
    LwSciBufAttrKeyValuePair keyValPair;
#if (LW_IS_SAFETY == 0)
    LwSciBufInternalAttrKeyValuePair intKeyValPair;
    LwSciBufMemDomain memDomain = LwSciBufMemDomain_UpperBound;
#endif //(LW_IS_SAFETY == 0)

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("attrList is invalid.");
        goto ret;
    }

    err = LwSciBufAttrListGetModule(attrList, &module);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get module.");
        goto ret;
    }

    err = LwSciBufModuleGetDevHandle(module, &devHandle);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get device handle.");
        goto ret;
    }

    keyValPair.key = LwSciBufGeneralAttrKey_GpuId;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        goto ret;
    }
    gpuIdPtr = (const LwSciRmGpuId*)keyValPair.value;
    numGpus = keyValPair.len / sizeof(LwSciRmGpuId);

    for (gpuIdx = 0U; gpuIdx < numGpus; gpuIdx++) {
        LwSciBufDevGetGpuDeviceInfo(devHandle, gpuIdPtr[gpuIdx],
            &gpuDeviceInfo);

        if (gpuDeviceInfo == NULL) {
            /* GPU ID from attribute list did not match with GPU ID in the
             * system. This should not happen.
             */
            LwSciCommonPanic();
        }

        /* GPU ID in the attribute list matched with GPU ID in the
         * system. Lets see if the GPU is of type, iGPU.
         */
        if (gpuDeviceInfo->gpuType == LwRmGpuType_iGPU) {
            numiGpus++;
        }
    }

    /* On cheetah, there are no multiple physical iGPUs. However, multiple
     * iGPUs are possible with SMC case. However, SMC case for iGPUs is not
     * supported for UMDs yet and thus, LwSciBuf shall not allow this.
     */
    if (numiGpus > 1U) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("Multiple iGPU case not supported.");
        goto ret;
    }

#if (LW_IS_SAFETY == 0)
    /* Get the memory domain */
    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_MemDomainArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &intKeyValPair, 1,
            LwSciBufAttrKeyType_Internal, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        goto ret;
    }
    memDomain = *(const LwSciBufMemDomain*)intKeyValPair.value;

    /* if iGPUs are specified in LwSciBufGeneralAttrKey_GpuId and the memory
     * domain is vidmem then this combination is not allowed.
     */
    if ((numiGpus > 0U) && (memDomain == LwSciBufMemDomain_Vidmem)) {
        err = LwSciError_ReconciliationFailed;
        LWSCI_ERR_STR("iGPUs cannot access vidmem. Looks like LwSciBufGeneralAttrKey_VidMem_GpuId attribute is specified while iGPU is specified in LwSciBufGeneralAttrKey_GpuId.");
        goto ret;
    }
#endif //(LW_IS_SAFETY == 0)

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListPlatformGetDefaultGpuCacheability(
    LwSciBufAttrList attrList,
    LwSciRmGpuId gpuId,
    bool* defaultCacheability)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyValPair;
    LwSciBufInternalAttrKeyValuePair intKeyValPair;
    LwSciBufMemDomain memDomain = LwSciBufMemDomain_UpperBound;
    LwRmGpuType gpuType = LwRmGpuType_iGPU;
    LwSciBufModule module = NULL;
    LwSciBufDev devHandle = NULL;
    const LwRmGpuDeviceInfo* gpuDeviceInfo = NULL;

    LWSCI_FNENTRY("");

    /* Get memory domain */
    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_MemDomainArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &intKeyValPair, 1,
            LwSciBufAttrKeyType_Internal, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get LwSciBufInternalGeneralAttrKey_MemDomainArray key.");
        goto ret;
    }

    if (intKeyValPair.len == 0U) {
        /* if memdomain is not set in the reconciled attribute list then it
         * implies that the reconciliation of memdomain is not complete. lets
         * get value on our own here. Also, lets not worry about reconciliation
         * dependencies for memdomain here since ultimately, if those conditions
         * are not met, the reconciliation will fail when reconciling the
         * memdomain key. Lets do simple reconciliation here.
         */
        keyValPair.key = LwSciBufGeneralAttrKey_VidMem_GpuId;
        err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Public, true);
        if (err != LwSciError_Success) {
           LWSCI_ERR_STR("Failed to get LwSciBufGeneralAttrKey_VidMem_GpuId key.");
            goto ret;
        }

        if (keyValPair.len != 0U) {
#if (LW_IS_SAFETY == 0)
            /* vidmem GPU ID is set. Thus, memdomain is vidmem */
            memDomain = LwSciBufMemDomain_Vidmem;
#endif //(LW_IS_SAFETY == 0)
        } else {
            /* memdomain key is not set and vidmem GPU ID key is also not set.
             * Lets default to memdomain = sysmem.
             */
            memDomain = LwSciBufMemDomain_Sysmem;
        }
    } else {
        memDomain = *(const LwSciBufMemDomain *)intKeyValPair.value;

#if (LW_IS_SAFETY == 0)
        if (memDomain == LwSciBufMemDomain_Cvsram) {
            err = LwSciError_ReconciliationFailed;
            LWSCI_ERR_STR("GPUs cannot access CVSRAM.");
            goto ret;
        }
#endif //(LW_IS_SAFETY == 0)
    }

    /* Get the GPU type */
    err = LwSciBufAttrListGetModule(attrList, &module);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get module.");
        goto ret;
    }

    err = LwSciBufModuleGetDevHandle(module, &devHandle);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get device handle.");
        goto ret;
    }

    LwSciBufDevGetGpuDeviceInfo(devHandle, gpuId, &gpuDeviceInfo);
    if (gpuDeviceInfo == NULL) {
        /* GPU ID not found in the system. This should not happen */
        LwSciCommonPanic();
    }
    gpuType = gpuDeviceInfo->gpuType;

    *defaultCacheability = defaultGpuCacheValue[gpuType][memDomain];

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListPlatformGpuCompressionDependency(
    LwSciBufAttrList attrList,
    LwSciRmGpuId gpuId,
    bool isBlockLinear,
    bool* isCompressible)
{
    LwSciError err = LwSciError_Success;
    LwSciBufModule module = NULL;
    LwSciBufDev devHandle = NULL;
    const LwSciBufAllGpuContext* allGpuContext = NULL;
    size_t index = 0U;
    LwRmGpuDevice* gpuDevice = NULL;
    LwRmGpuKindAttributes kindAttrs = {};
    LwRmGpuMemKind kind = 0U;
    LwRmGpuKindInfo kindInfo = {};
    LwError lwErr = LwError_Success;

    LWSCI_FNENTRY("");

    if (isCompressible == NULL) {
        LwSciCommonPanic();
    }

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("attrList is invalid.");
        goto ret;
    }

    err = LwSciBufAttrListGetModule(attrList, &module);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get module.");
        goto ret;
    }

    err = LwSciBufModuleGetDevHandle(module, &devHandle);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to get device handle.");
        goto ret;
    }

    LwSciBufDevGetAllGpuContext(devHandle, &allGpuContext);

    for (index = 0U; index < allGpuContext->gpuListSize; index++) {
        const LwRmGpuDeviceInfo* gpuDeviceInfo =
            allGpuContext->perGpuContext[index].gpuDeviceInfo;
        if (gpuDeviceInfo == NULL) {
            continue;
        }

        if (LwSciCommonMemcmp(&gpuDeviceInfo->deviceId.gid, &gpuId,
            sizeof(gpuId)) == 0U) {
            gpuDevice = allGpuContext->perGpuContext[index].gpuDevice;
            break;
        }
    }

    if (index == allGpuContext->gpuListSize) {
        /* gpuId supplied as input parameter did not match with any of the
         * GPUs in the LwSciBufDev. This should not happen.
         */
        LwSciCommonPanic();
    }

    if (isBlockLinear == true) {
        LwRmGpuKindSetGenericBlockLinearAttributes(true, true, &kindAttrs);
    } else {
        LwRmGpuKindSetPitchAttributes(&kindAttrs);
        kindAttrs.allowCompression = true;
        kindAttrs.allowPLC = true;
    }

    lwErr = LwRmGpuDeviceChooseKind(gpuDevice, &kindAttrs, &kind);
    if (lwErr != LwError_Success) {
        LWSCI_ERR_STR("LwRmGpuDeviceChooseKind failed.");
        err = LwSciError_ResourceError;
        goto ret;
    }

    lwErr = LwRmGpuDeviceGetKindInfo(gpuDevice, kind, &kindInfo);
    if (lwErr != LwError_Success) {
        LWSCI_ERR_STR("LwRmGpuDeviceGetKindInfo failed.");
        err = LwSciError_ResourceError;
        goto ret;
    }

    *isCompressible = kindInfo.compressible;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListPlatformVidmemDependency(
    LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;
    LwSciBufInternalAttrKeyValuePair intKeyValPair = {0};
    bool hasPcieEngine = false;
    size_t engineCount = 0UL;
    const LwSciBufHwEngine* engineArray = NULL;

    LWSCI_FNENTRY("");

    /* Get engine array. */
    intKeyValPair.key = LwSciBufInternalGeneralAttrKey_EngineArray;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &intKeyValPair, 1,
            LwSciBufAttrKeyType_Internal, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed to get LwSciBufInternalGeneralAttrKey_EngineArray.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (0UL == intKeyValPair.len) {
        /* No other HW engine is accessing the LwSciBufObj. */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    engineCount = intKeyValPair.len / sizeof(LwSciBufHwEngine);

    /* If Vidmem_Gpu_Id is specified and if there is an engine specified in
     * EngineArray other than PCIe engine then the reconciliation fails
     * because only PCIe engine can access Vidmem.
     */
    if (1UL < engineCount) {
        /* More than one engine present in the EngineArray. Even if one of the
         * engines is PCIe, the other one is not. Fail reconciliation here.
         */
        LWSCI_ERR_STR("Any non-GPU engine other than PCIe cannot access Vidmem. However, both LwSciBufGeneralAttrKey_VidMem_GpuId and engines not having access to vidmem specified in LwSciBufInternalGeneralAttrKey_EngineArray in the unreconciled list");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    engineArray = (const LwSciBufHwEngine *)intKeyValPair.value;
    err = LwSciBufHasEngine(engineArray, engineCount, LwSciBufHwEngName_PCIe,
            &hasPcieEngine);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufHasEngine failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == hasPcieEngine) {
        LWSCI_ERR_STR("Any non-GPU engine other than PCIe cannot access Vidmem. However, both LwSciBufGeneralAttrKey_VidMem_GpuId and engines not having access to vidmem specified in LwSciBufInternalGeneralAttrKey_EngineArray in the unreconciled list");
        err = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
