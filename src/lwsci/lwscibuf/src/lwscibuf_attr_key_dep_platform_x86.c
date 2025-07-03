/*
 * Copyright (c) 2021-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_attr_key_dep_platform.h"

/* Note: As opposed to cheetah, we dont need to create 2D array of gpuType and
 * memory domain for x86 since x86 only supports dGPUs. Thus, an array based
 * on memory domain only suffices.
 */
static const bool
    defaultGpuCacheValue[LwSciBufMemDomain_UpperBound] = {
    [LwSciBufMemDomain_Sysmem] = false,
    [LwSciBufMemDomain_Vidmem] = true,
};

static LwSciError LwSciBufAttrValidateUuidForMIG(
    LwSciBufAttrList attrList)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufModule module = NULL;
    LwSciBufDev devHandle = NULL;
    bool valid = false;
    LwSciBufAttrKeyValuePair publicAttrs = {0};
    LwSciBufInternalAttrKeyValuePair internalAttrs = {0};
    LwSciBufMemDomain memDomain;
    const LwSciRmGpuId* gpuId = NULL;
    uint64_t numGpus = 0U;

    sciErr = LwSciBufAttrListGetModule(attrList, &module);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListGetModule failed.");
        goto ret;
    }

    sciErr = LwSciBufModuleGetDevHandle(module, &devHandle);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciModuleGetDevHandle failed.");
        goto ret;
    }

    publicAttrs.key = LwSciBufGeneralAttrKey_GpuId;
    internalAttrs.key = LwSciBufInternalGeneralAttrKey_MemDomainArray;

    sciErr = LwSciBufAttrListCommonGetAttrs(attrList, 0, &publicAttrs,
        1U, LwSciBufAttrKeyType_Public, true);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        goto ret;
    }

    sciErr = LwSciBufAttrListCommonGetAttrs(attrList, 0, &internalAttrs,
        1U, LwSciBufAttrKeyType_Internal, true);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        goto ret;
    }

    memDomain = *(const LwSciBufMemDomain*)internalAttrs.value;
    gpuId = (const LwSciRmGpuId*)publicAttrs.value;
    numGpus = publicAttrs.len/sizeof(LwSciRmGpuId);

    /* If memory domain is sysmem or cvsram and user has not set GPU ID in
     *  attribute list, LwSciBuf will choose default device (device 0)
     *  during allocation.
     *  Hence, we skip validation for default case since GPU 0 allocation will
     *  always succeed.
     * NOTE :
     *  Device == GPU (in non-MIG case)
     *  Device == MIG instance (in MIG case)
     */
    if (memDomain != LwSciBufMemDomain_Vidmem && numGpus == 0U) {
        goto ret;
    }

    sciErr = LwSciBufDevValidateUUID(devHandle, memDomain, gpuId, numGpus,
                &valid);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrValidateUUIDx86 failed.");
        goto ret;
    }

ret:
    return sciErr;
}

/* TODO: The attribute key dependency is part of reconciliation.
 * Reconciliation is platform agnostic operation. Also, the heap types that we
 * are setting here is dummy heap type and x86 does not really use any heap
 * type. We should ultimately remove this platform dependency. (Check TODO in
 * lwscibuf_attr_key_dep_platform_tegra.c for how to move the heap type
 * dependency to other units). We are keeping this as is for now so as not to
 * open a can of worms. This cleanup should be taken up in separate task.
 */
LwSciError LwSciBufAttrListPlatformHeapDependency(
    LwSciBufAttrList attrList,
    LwSciBufHeapType* heapType)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("attrList is invalid.");
        goto ret;
    }

    *heapType = LwSciBufHeapType_Resman;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListPlatformGpuIdDependency(
 LwSciBufAttrList attrList)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("attrList is invalid.");
        goto ret;
    }

    err = LwSciBufAttrValidateUuidForMIG(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrValidateUuidForMIG() failed.");
        goto ret;
    }

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
    (void)gpuId;

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
            /* vidmem GPU ID is set. Thus, memdomain is vidmem */
            memDomain = LwSciBufMemDomain_Vidmem;
        } else {
            /* memdomain key is not set and vidmem GPU ID key is also not set.
             * Lets default to memdomain = sysmem.
             */
            memDomain = LwSciBufMemDomain_Sysmem;
        }
    } else {
        memDomain = *(const LwSciBufMemDomain *)intKeyValPair.value;
    }

    *defaultCacheability = defaultGpuCacheValue[memDomain];

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

    LWSCI_FNENTRY("");
    (void)gpuId;
    (void)isBlockLinear;

    err = LwSciBufAttrListValidate(attrList);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("attrList is invalid.");
        goto ret;
    }

    if (isCompressible == NULL) {
        LwSciCommonPanic();
    }

    /* Initial GPU compression feature are only supporting PLC compression.
     * Since PLC compression is disabled on GA100 due to HW bug, disallow
     * compression here without querying hardware.
     * In future if we start supporting other dGPUs where PLC is supported OR
     * we start supporting other compression types then we should start
     * querying hardware and check if compression is supported.
     */
    *isCompressible = false;

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciBufAttrListPlatformVidmemDependency(
    LwSciBufAttrList attrList)
{
    (void)attrList;
    /* TODO: Figure out which sub-engines of the dGPU can access vidmem? */
    return LwSciError_Success;
}
