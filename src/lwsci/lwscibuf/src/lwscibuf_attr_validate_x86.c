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

// TODO: Validate it for x86
LwSciError LwSciBufValidateGpuId(
    const LwSciBufAttrList attrList,
    const void *val)
{
    (void)attrList;
    (void)val;

    return LwSciError_Success;
}

// TODO: Validate it for x86
LwSciError LwSciBufValidateAttrValGpuCache(
    LwSciBufAttrList attrList,
    const void* val)
{
    (void)attrList;
    (void)val;

    return LwSciError_Success;
}

LwSciError LwSciBufValidateAttrValGpuCompressionInternal(
    LwSciBufAttrList attrList,
    const void* val)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrValGpuCompression gpuCompression =
        *(const LwSciBufAttrValGpuCompression*)val;

    LWSCI_FNENTRY("");

    /* TODO: Validate GPU ID for x86
     * Validity of GPU ID on X86 is complex because of MIG. Basically, figuring
     * out if GPU ID is valid or not ilwolves knowledge of memdomain for MIG.
     * This validation function will get called when user sets the attributes
     * in LwSciBufAttrList and thus, it is not necessary that we would have
     * memory domain available at that time. (It would be available when we
     * reconcile the unreconciled LwSciBufAttrLists).
     * Maybe, we need to move the GPU ID validation during reconciliation
     * for X86?
     */
    err = LwSciBufValidateGpuCompressionInternal(attrList,
            &gpuCompression.compressionType);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufValidateGpuCompressionInternal failed.");
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

    /* TODO: Validate GPU ID for x86
     * Validity of GPU ID on X86 is complex because of MIG. Basically, figuring
     * out if GPU ID is valid or not ilwolves knowledge of memdomain for MIG.
     * This validation function will get called when user sets the attributes
     * in LwSciBufAttrList and thus, it is not necessary that we would have
     * memory domain available at that time. (It would be available when we
     * reconcile the unreconciled LwSciBufAttrLists).
     * Maybe, we need to move the GPU ID validation during reconciliation
     * for X86?
     */
    err = LwSciBufValidateGpuCompressionExternal(attrList,
            &gpuCompression.compressionType);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufValidateGpuCompressionExternal failed.");
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}
