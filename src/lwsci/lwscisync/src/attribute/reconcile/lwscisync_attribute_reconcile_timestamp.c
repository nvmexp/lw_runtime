/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync timestamp reconciliation implementation</b>
 *
 * @b Description: This file implements timestamp attribute reconciliation logic
 */
#include "lwscisync_attribute_reconcile_private.h"

#include "lwscicommon_libc.h"
#include "lwscicommon_utils.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core_cluster.h"
#include "lwscisync_attribute_reconcile_priv.h"

static LwSciError ReconcileTimestampBufAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList
);

static bool TraveledViaC2C(
    const LwSciSyncCoreAttrListObj* objAttrList);

LwSciError ReconcileWaiterRequireTimestamps(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;

    size_t i = 0U;

    LwSciSyncCoreAttrKeyState reconciledKeyState = LwSciSyncCoreAttrKeyState_Empty;
    bool timestampRequired = false;
    bool timestampProvided = false;

    LwSciSyncCoreAttrs* reconciledAttrs = NULL;

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        LwSciSyncCoreAttrList* coreAttrList = &objAttrList->coreAttrList[i];
        const LwSciSyncCoreAttrs* attrs = &coreAttrList->attrs;

        /* Check if timestamp is required. */
        timestampRequired = (timestampRequired || attrs->waiterRequireTimestamps);

        /* Check if timestamp support is provided by a signaler. */
        if (LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList)) {
            LwSciSyncCoreAttrKeyState keyState = LwSciSyncCoreAttrKeyState_Empty;

            /* Timestamp information can be provided via these attribute keys:
             *      - SignalerTimestampInfo
             *      - SignalerTimestampInfoMulti */
            keyState = attrs->keyState[LwSciSyncCoreKeyToIndex(
                    (int32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo)];
            if (keyState != LwSciSyncCoreAttrKeyState_Empty) {
                timestampProvided = true;
            }

            keyState = attrs->keyState[LwSciSyncCoreKeyToIndex(
                    (int32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti)];
            if (keyState != LwSciSyncCoreAttrKeyState_Empty) {
                timestampProvided = true;
            }
        }
    }

    /** Reconciliation logic for LwSciSyncAttrKey_WaiterRequireTimestamps */
    reconciledKeyState = LwSciSyncCoreAttrKeyState_Reconciled;
    if (true == timestampRequired) {
        if (false == timestampProvided) {
            reconciledKeyState = LwSciSyncCoreAttrKeyState_Conflict;
            error = LwSciError_ReconciliationFailed;
            LWSCI_ERR_STR("timestamps are required but not provided");
        } else if (TraveledViaC2C(objAttrList)) {
            reconciledKeyState = LwSciSyncCoreAttrKeyState_Conflict;
            error = LwSciError_ReconciliationFailed;
            LWSCI_ERR_STR("timestamps are not supported over C2C");
        } else {
            /* no conflict detected, so no action */
        }
    }

    reconciledAttrs = &newObjAttrList->coreAttrList->attrs;
    reconciledAttrs->waiterRequireTimestamps = timestampRequired;
    reconciledAttrs->keyState[LwSciSyncCoreKeyToIndex(
            (int32_t)LwSciSyncAttrKey_WaiterRequireTimestamps)] = reconciledKeyState;
    reconciledAttrs->valSize[LwSciSyncCoreKeyToIndex(
            (int32_t)LwSciSyncAttrKey_WaiterRequireTimestamps)] = sizeof(bool);

    return error;
}

static bool TraveledViaC2C(
    const LwSciSyncCoreAttrListObj* objAttrList)
{
    size_t i = 0U;
    bool result = false;

    for (i = 0U; i < objAttrList->numCoreAttrList; ++i) {
        if (LwSciSyncCoreIpcTableHasC2C(
                &objAttrList->coreAttrList[i].ipcTable)) {
            result = true;
            goto end;
        }
    }

end:
    return result;
}

static void SetTimestampInfo(
    const LwSciSyncCoreAttrs* attrs,
    LwSciSyncCoreAttrs* reconciledAttrs,
    size_t primitiveIndex)
{
    const size_t multiIndex = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
    LwSciSyncCoreAttrKeyState keyStateMulti = attrs->keyState[multiIndex];

    const size_t index = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo);
    LwSciSyncCoreAttrKeyState keyState = attrs->keyState[index];

    if (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty) {
        LwSciSyncAttrValTimestampInfo timestampInfo =
            attrs->signalerTimestampInfoMulti[primitiveIndex];

        reconciledAttrs->signalerTimestampInfoMulti[0] = timestampInfo;
        reconciledAttrs->keyState[multiIndex] = LwSciSyncCoreAttrKeyState_Reconciled;
        reconciledAttrs->valSize[multiIndex] = sizeof(timestampInfo);
        goto fn_exit;
    }
    if (keyState != LwSciSyncCoreAttrKeyState_Empty) {
        LwSciSyncAttrValTimestampInfo timestampInfo =
            attrs->signalerTimestampInfo;

        reconciledAttrs->signalerTimestampInfo = timestampInfo;
        reconciledAttrs->keyState[index] = LwSciSyncCoreAttrKeyState_Reconciled;
        reconciledAttrs->valSize[index] = sizeof(timestampInfo);
        goto fn_exit;
    }

fn_exit:
    return;
}

LwSciError ReconcileSignalerTimestampInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;

    bool timestampRequired =
        newObjAttrList->coreAttrList->attrs.waiterRequireTimestamps;

    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitive =
        LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    size_t i = 0U;

    LwSciSyncCoreAttrs* reconciledAttrs = &newObjAttrList->coreAttrList->attrs;
    const LwSciSyncAttrValTimestampInfo* timestampInfo = NULL;

    LWSCI_FNENTRY("");

    if (timestampRequired == false) {
        /* Nothing to do */
        goto fn_exit;
    }

    /* We have the reconciled primitive, so we just need to find the timestamp
     * format that corresponds to the given primitive. */
    reconciledPrimitive = newObjAttrList->coreAttrList->attrs.signalerPrimitiveInfo[0];

    /* Get the index in SignalerPrimitiveInfo that corresponds to the chosen
     * primitive */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        LwSciSyncCoreAttrList* coreAttrList = &objAttrList->coreAttrList[i];

        if (LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList)) {
            size_t primitiveIndex = 0U;

            error = LwSciSyncGetSignalerPrimitiveInfoIndex(coreAttrList,
                reconciledPrimitive, &primitiveIndex);
            if (error != LwSciError_Success) {
                LWSCI_ERR_STR("A primitive was reconciled to a primitive "
                        "that was not provided by any signaler");
                LwSciCommonPanic();
            }
            SetTimestampInfo(&coreAttrList->attrs, reconciledAttrs,
                primitiveIndex);
        }
    }

    /** Reconcile LwSciBuf attrlists for the timestamp buffer only if the
     * reconciled LwSciSyncTimestampFormat requires it. */
    LwSciSyncCoreGetTimestampInfo(reconciledAttrs, &timestampInfo);
    if (timestampInfo == NULL) {
        LwSciCommonPanic();
    }
    if (timestampInfo->format != LwSciSyncTimestampFormat_EmbeddedInPrimitive) {
        error = ReconcileTimestampBufAttrList(objAttrList, newObjAttrList);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}


static LwSciError ReconcileTimestampBufAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList
)
{
    LwSciError error = LwSciError_Success;
    LwSciBufAttrList* inputAttrLists = NULL;
    LwSciBufAttrList* outputAttrList = NULL;
    size_t numInputAttrList = 0U;
    size_t i = 0U;
    uint8_t addStatus = OP_FAIL;

    error = LwSciSyncCoreFillTimestampBufAttrList(objAttrList);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    inputAttrLists = (LwSciBufAttrList*)LwSciCommonCalloc(
            objAttrList->numCoreAttrList, sizeof(LwSciBufAttrList));
    if (inputAttrLists == NULL) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

    numInputAttrList = 0U;
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        if (objAttrList->coreAttrList[i].timestampBufAttrList != NULL) {
            inputAttrLists[numInputAttrList] =
                    objAttrList->coreAttrList[i].timestampBufAttrList;
            u64Add(numInputAttrList, 1U, &numInputAttrList, &addStatus);
            if (addStatus != OP_SUCCESS) {
                LWSCI_ERR_STR("numInputAttrList value is out of range.\n");
                error = LwSciError_Overflow;
                goto fn_exit;
            }
        }
    }

    outputAttrList = &newObjAttrList->coreAttrList->timestampBufAttrList;
    error = LwSciBufAttrListReconcile(inputAttrLists, numInputAttrList,
            outputAttrList, outputAttrList);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

fn_exit:
    LwSciCommonFree(inputAttrLists);
    return error;
}

void LwSciSyncCoreFillSignalerTimestampInfo(
    const LwSciSyncCoreAttrListObj* objAttrList)
{
    LwSciSyncCoreAttrList* coreAttrList = objAttrList->coreAttrList;

    size_t keyIdx = LwSciSyncCoreKeyToIndex(
            (int32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo);
    LwSciSyncCoreAttrKeyState keyState = coreAttrList->attrs.keyState[keyIdx];
    size_t keyIdxMulti = LwSciSyncCoreKeyToIndex(
            (int32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
    LwSciSyncCoreAttrKeyState keyStateMulti = coreAttrList->attrs.keyState[keyIdxMulti];
    bool timestampInfoProvided =
        ((keyState != LwSciSyncCoreAttrKeyState_Empty) ||
         (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty));

    bool isSignaler = LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList);

    if (!timestampInfoProvided && isSignaler && coreAttrList->attrs.needCpuAccess) {
        LwSciSyncAttrValTimestampInfo tinfo = {
            .format = LwSciSyncTimestampFormat_8Byte,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        };

        coreAttrList->attrs.signalerTimestampInfoMulti[0] = tinfo;
        coreAttrList->attrs.keyState[keyIdxMulti] =
                LwSciSyncCoreAttrKeyState_SetLocked;
        coreAttrList->attrs.valSize[keyIdxMulti] = sizeof(tinfo);
    }
}
