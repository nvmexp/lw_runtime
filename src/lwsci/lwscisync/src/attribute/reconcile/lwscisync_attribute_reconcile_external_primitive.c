/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscisync_attribute_core_cluster.h"
#include "lwscisync_primitive.h"
#include "lwscisync_attribute_reconcile_private.h"

/*
 * Determine the primitive info for reconciled primitive type
 */
static void GetPrimitiveInfoForReconciledPrimitiveType(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitiveType,
    LwSciSyncPrimitiveInfo** reconciledPrimitiveInfo);

/** Validate ExternalPrimitiveInfo */
static LwSciError ValidateExternalPrimitiveInfo(
    LwSciSyncPrimitiveInfo* primitiveInfo,
    const LwSciSyncCoreAttrs* attrs)
{
    LwSciError error = LwSciError_Success;

    switch (primitiveInfo->simplePrimitiveInfo.primitiveType) {
        case LwSciSyncInternalAttrValPrimitiveType_Syncpoint:
            if (attrs->signalerPrimitiveCount !=
                    primitiveInfo->simplePrimitiveInfo.numIds) {
                LWSCI_ERR_STR("SignalerPrimitiveCount mismatch\n");
                error = LwSciError_ReconciliationFailed;
                goto fail;
            }
            if (primitiveInfo->simplePrimitiveInfo.ids == NULL) {
                LWSCI_ERR_STR("simplePrimitiveInfo ids is NULL\n");
                error = LwSciError_BadParameter;
                goto fail;
            }
            break;
        case LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore:
        case LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore:
        case LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b:
        case LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphorePayload64b:
            if ((primitiveInfo->semaphorePrimitiveInfo.len %
                    LWSCISYNC_CORE_PRIMITIVE_SEMAPHORE_SIZE) != 0U) {
                LWSCI_ERR_STR("Invalid semaphore size\n");
                error = LwSciError_ReconciliationFailed;
                goto fail;
            }
            if (attrs->signalerPrimitiveCount !=
                    (primitiveInfo->semaphorePrimitiveInfo.len /
                    LWSCISYNC_CORE_PRIMITIVE_SEMAPHORE_SIZE)) {
                LWSCI_ERR_STR("SignalerPrimitiveCount mismatch\n");
                error = LwSciError_ReconciliationFailed;
                goto fail;
            }
            break;
        default:
            LWSCI_ERR_STR("Unrecognized primitive type.\n");
            LwSciCommonPanic();
    }

fail:
    return error;
}

LwSciError ReconcileUseExternalPrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList,
    LwSciSyncAttrList newReconciledList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitiveType =
            LwSciSyncInternalAttrValPrimitiveType_UpperBound;
    LwSciSyncCoreAttrList* newCoreAttrList = newObjAttrList->coreAttrList;
    bool isCpuSignaler = false;
    size_t i = 0;
    LwSciSyncPrimitiveInfo* reconciledExternalPrimitiveInfo = NULL;
    LwSciSyncCoreAttrs* attrs = &newObjAttrList->coreAttrList->attrs;
    size_t attrIndex = LwSciSyncCoreKeyToIndex(
                (uint32_t)LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo);
    bool isC2cSignaler = false;

    LWSCI_FNENTRY("");

    /* Get reconciled Primitive Type. We expect that this function is called
     * after the Primitive Type has been reconciled. */
    reconciledPrimitiveType = attrs->signalerPrimitiveInfo[0];

    /* Get whether this is a CPU signaler by traversing the Unreconciled
     * Attribute Lists. */
    for (i = 0; i < objAttrList->numCoreAttrList; ++i) {
        LwSciSyncCoreAttrList coreAttrList = objAttrList->coreAttrList[i];

        if (LwSciSyncCoreAttrListHasSignalerPerm(&coreAttrList)) {
            /* This is the Signaler's attribute list */
            isCpuSignaler = coreAttrList.attrs.needCpuAccess;
            GetPrimitiveInfoForReconciledPrimitiveType(&coreAttrList,
                    reconciledPrimitiveType, &reconciledExternalPrimitiveInfo);
            break;
        }
    }
    if (isCpuSignaler && (reconciledExternalPrimitiveInfo != NULL)) {
        LWSCI_ERR_STR("External primitive info set for CPU Signaler\n");
        attrs->keyState[attrIndex] = LwSciSyncCoreAttrKeyState_Conflict;
        error = LwSciError_ReconciliationFailed;
        goto fail;
    }
    if (reconciledExternalPrimitiveInfo != NULL) {
        error = ValidateExternalPrimitiveInfo(reconciledExternalPrimitiveInfo,
                attrs);
        if (LwSciError_Success != error) {
            attrs->keyState[attrIndex] = LwSciSyncCoreAttrKeyState_Conflict;
            goto fail;
        }
    }

    /* assumes that _EngineArray and _ActualPerm have been reconciled already */
    LwSciSyncCoreAttrListTypeIsC2cSignaler(
        newReconciledList, &isC2cSignaler);

    /* Now set the key on the Reconciled Attribute List */
    newCoreAttrList->signalerUseExternalPrimitive =
        ((!isCpuSignaler) &&
         (!isC2cSignaler) &&
         (!LwSciSyncCoreIpcTableHasC2C(&newObjAttrList->coreAttrList->ipcTable)) &&
         ((reconciledPrimitiveType ==
           LwSciSyncInternalAttrValPrimitiveType_Syncpoint) ||
          (reconciledExternalPrimitiveInfo != NULL)));

    if (reconciledExternalPrimitiveInfo != NULL) {
        error = LwSciSyncCoreCopySignalerExternalPrimitiveInfo(
                    attrs->signalerExternalPrimitiveInfo,
                    &reconciledExternalPrimitiveInfo,
                    1U);
        if (LwSciError_Success != error) {
            goto fail;
        }
        attrs->valSize[attrIndex] = sizeof(LwSciSyncPrimitiveInfo*);
        attrs->keyState[attrIndex] = LwSciSyncCoreAttrKeyState_Reconciled;
    }

fail:
    LWSCI_FNEXIT("");
    return error;
}

static void GetPrimitiveInfoForReconciledPrimitiveType(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitiveType,
    LwSciSyncPrimitiveInfo** reconciledPrimitiveInfo)
{
    LwSciSyncPrimitiveInfo** signalerExternalPrimitiveInfo =
            coreAttrList->attrs.signalerExternalPrimitiveInfo;
    uint64_t valSize =
           coreAttrList->attrs.valSize[LwSciSyncCoreKeyToIndex(
           (int32_t)LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo)];
    size_t i;

    *reconciledPrimitiveInfo = NULL;

    for (i = 0U; i < valSize/sizeof(LwSciSyncPrimitiveInfo*); i++) {
        if (signalerExternalPrimitiveInfo[i]->simplePrimitiveInfo.primitiveType ==
                reconciledPrimitiveType) {
            *reconciledPrimitiveInfo = signalerExternalPrimitiveInfo[i];
            break;
        }
    }
}

static bool ExternalPrimitiveInfoIsEqual(
    const LwSciSyncPrimitiveInfo* primitiveInfo1,
    const LwSciSyncPrimitiveInfo* primitiveInfo2)
{
    bool isEqual = true;

    if (primitiveInfo1 == primitiveInfo2) {
        isEqual = true;
        goto fn_exit;
    }
    if ((primitiveInfo1 == NULL) || (primitiveInfo2 == NULL)) {
        isEqual = false;
        goto fn_exit;
    }
    if (primitiveInfo1->simplePrimitiveInfo.primitiveType !=
            primitiveInfo2->simplePrimitiveInfo.primitiveType) {
        isEqual = false;
        goto fn_exit;
    }
    switch (primitiveInfo1->simplePrimitiveInfo.primitiveType) {
        case LwSciSyncInternalAttrValPrimitiveType_Syncpoint:
            if (primitiveInfo1->simplePrimitiveInfo.numIds !=
                    primitiveInfo2->simplePrimitiveInfo.numIds) {
                isEqual = false;
                goto fn_exit;
            }
            if (LwSciCommonMemcmp((const void*)primitiveInfo1->simplePrimitiveInfo.ids,
                    (const void *)primitiveInfo2->simplePrimitiveInfo.ids,
                    sizeof(uint64_t)*primitiveInfo1->simplePrimitiveInfo.numIds) != 0U) {
                isEqual = false;
                goto fn_exit;
            }
            break;
        case LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore:
        case LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore:
        case LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b:
        case LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphorePayload64b:
            if (LwSciCommonMemcmp((const void*)primitiveInfo1,
                    (const void *)primitiveInfo2,
                    sizeof(LwSciSyncSemaphorePrimitiveInfo)) != 0U) {
                isEqual = false;
                goto fn_exit;
            }
            break;
        default:
            LWSCI_ERR_STR("Unrecognized primitive type.\n");
            LwSciCommonPanic();
    }

fn_exit:
    return isEqual;
}

LwSciError ValidateReconciledExternalPrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* recObjAttrList)
{
    LwSciError error = LwSciError_Success;
    bool isCpuSignaler = false;
    size_t i = 0U;
    LwSciSyncPrimitiveInfo* requestedExternalPrimitiveInfo = NULL;
    LwSciSyncPrimitiveInfo* reconciledExternalPrimitiveInfo = NULL;
    const LwSciSyncCoreAttrs* attrs = &recObjAttrList->coreAttrList->attrs;
    size_t attrIndex = LwSciSyncCoreKeyToIndex(
                (uint32_t)LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo);
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitiveType =
            (LwSciSyncInternalAttrValPrimitiveType) attrs->signalerPrimitiveInfo[0];
    reconciledExternalPrimitiveInfo = (attrs->keyState[attrIndex] ==
            LwSciSyncCoreAttrKeyState_Reconciled) ?
            attrs->signalerExternalPrimitiveInfo[0] : NULL;

    if (NULL == objAttrList) {
        goto fn_exit;
    }

    for (i = 0; i < objAttrList->numCoreAttrList; ++i) {
        LwSciSyncCoreAttrList coreAttrList = objAttrList->coreAttrList[i];

        if (LwSciSyncCoreAttrListHasSignalerPerm(&coreAttrList)) {
            /* This is the Signaler's attribute list */
            isCpuSignaler = coreAttrList.attrs.needCpuAccess;
            GetPrimitiveInfoForReconciledPrimitiveType(&coreAttrList,
                    reconciledPrimitiveType, &requestedExternalPrimitiveInfo);

            if (!ExternalPrimitiveInfoIsEqual(requestedExternalPrimitiveInfo,
                    reconciledExternalPrimitiveInfo)) {
                LWSCI_ERR_STR("Mismatch in external primitive info\n");
                error = LwSciError_BadParameter;
                goto fn_exit;
            }
            if (isCpuSignaler && (requestedExternalPrimitiveInfo != NULL)) {
                LWSCI_ERR_STR("External primitive info set for CPU Signaler\n");
                error = LwSciError_BadParameter;
                goto fn_exit;
            }
            if (reconciledExternalPrimitiveInfo != NULL) {
                error = ValidateExternalPrimitiveInfo(
                        reconciledExternalPrimitiveInfo, attrs);
                if (LwSciError_Success != error) {
                    error = LwSciError_BadParameter;
                    goto fn_exit;
                }
            }
        }
    }

fn_exit:

    return error;
}
