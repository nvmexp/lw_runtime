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
 * \brief <b>LwSciSync Attribute Reconciliation Implementation</b>
 *
 * @b Description: This file implements LwSciSync attribute reconcile APIs
 *
 * The code in this file is organised as below:
 * -Core interfaces declaration.
 * -Public interfaces definition.
 * -Core interfaces definition.
 */
#include "lwscisync_attribute_reconcile_private.h"

#include "lwscicommon_utils.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core.h"
#include "lwscisync_attribute_core_cluster.h"
#include "lwscisync_attribute_reconcile_priv.h"
#include "lwscisync_core.h"
#include "lwscisync_module.h"
#include "lwscisync_primitive.h"

/** Sanity check for input permissions */
static inline bool CheckRequiredPermValues(
    LwSciSyncAccessPerm perm)
{
    return ((LwSciSyncAccessPerm_WaitOnly == perm) ||
        (LwSciSyncAccessPerm_SignalOnly == perm) ||
        (LwSciSyncAccessPerm_WaitSignal == perm));
}

/** Choose the common signaler and waiter primitive */
static LwSciError ReconcilePrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

/** Determine if WaiterContextInsensitiveFenceExports flag is needed */
static void ReconcileWaiterContextInsensitiveFenceExports(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

static LwSciError ReconcileEngineArray(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

/** Validate public keys set in attr list */
static LwSciError ValidatePublicKeyValues(
    const LwSciSyncCoreAttrListObj* objAttrList);

/** Validate internal keys set in attr list */
static LwSciError ValidateInternalKeyValues(
    const LwSciSyncCoreAttrListObj* objAttrList);

static LwSciError ValidateReconciledSignalerTimestampInfo(
    const LwSciSyncCoreAttrListObj* unreconciledObjAttrList,
    const LwSciSyncCoreAttrListObj* reconciledObjAttrList);

/** Verify that reconciled primitive type matches with requested one */
static LwSciError ValidateReconciledPrimitiveType(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* recObjAttrList);

/** Verify that reconciled engine array matches with requested one */
static LwSciError ValidateReconciledEngineArray(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* recObjAttrList);

static LwSciError ValidateReconciledSignalerPrimitiveCount(
    const LwSciSyncCoreAttrListObj* unreconciledObjAttrList,
    const LwSciSyncCoreAttrListObj* reconciledObjAttrList);

/** Validate reconciled timestamp buffer LwSciBufAttrList */
static LwSciError ValidateReconciledTimestampBufAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

/** Get size of valid primitives */
static size_t GetValidPrimitiveTypeSize(
    const LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len);

/** Colwert the primitive types to bit-shifted mask */
static size_t CreateMaskFromPrimitiveType(
    const LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len);

/** Colwert the timestamp formats to bit-shifted mask */
static size_t CreateMaskOfTimestampPrimitives(
    const LwSciSyncInternalAttrValPrimitiveType* signalerPrimitiveInfo,
    size_t signalerPrimitiveInfoLen,
    const LwSciSyncAttrValTimestampInfo* timestampInfo);

static LwSciError CreateMaskOfTimestampPrimitivesMulti(
    const LwSciSyncInternalAttrValPrimitiveType* signalerPrimitiveInfo,
    size_t signalerPrimitiveInfoLen,
    const LwSciSyncAttrValTimestampInfo* timestampInfo,
    size_t signalerTimestampInfoLen,
    size_t* mask);

/** Check LwSciSyncAttrListReconcile input arguments */
static LwSciError AttrListReconcileCheckArgs(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    const LwSciSyncAttrList* newReconciledList,
    const LwSciSyncAttrList* newConflictList);

/**
 * @brief Prepare object attr list
 *
 * Note: We expect the caller to provide any locking on the Attribute Lists as
 * needed.
 */
static LwSciError PrepareObjAttrList(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    LwSciSyncAttrList* newUnreconciledAttrList,
    LwSciSyncCoreAttrListObj** objAttrList,
    LwSciSyncAttrList* multiSlotAttrList,
    LwSciSyncCoreAttrListObj** newObjAttrList);

/** Check LwSciSyncAttrListValidateReconciled input arguments */
static LwSciError AttrListValidateReconciledCheckArgs(
    LwSciSyncAttrList reconciledAttrList,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    const bool* isReconciledListValid
);

/** Copy IPC permissions table to new list */
static LwSciError CopyIpcPermTable(
    const LwSciSyncCoreAttrListObj* objAttrList,
    LwSciSyncCoreAttrList* newCoreAttrList);

/** Check that timestamp is required and was provided */
static LwSciError AttrListCheckTimestampRequired(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList,
    bool* timestampRequired
);

/** Ensure two lists are bound to same LwSciSyncModule */
static LwSciError EnsureSameModule(
    const LwSciSyncCoreAttrListObj* unreconciledObjAttrList,
    const LwSciSyncCoreAttrListObj* reconciledObjAttrList
);

/*
 * Determine the reconciled permissions
 */
static LwSciError ReconcilePerms(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

/*
 * Determine the reconciled ReconcileRequireDeterministicFences
 */
static void ReconcileRequireDeterministicFences(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList);

#ifndef LWSCISYNC_EMU_SUPPORT
/** Reconcile the UseExternalPrimitive private key */
static void ReconcileUseExternalPrimitive(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitiveType =
            LwSciSyncInternalAttrValPrimitiveType_UpperBound;
    LwSciSyncCoreAttrList* newCoreAttrList = newObjAttrList->coreAttrList;
    bool isCpuSignaler = false;
    size_t i = 0;

    LWSCI_FNENTRY("");

    /* Get reconciled Primitive Type. We expect that this function is called
     * after the Primitive Type has been reconciled. */
    reconciledPrimitiveType = newCoreAttrList->attrs.signalerPrimitiveInfo[0];

    /* Get whether this is a CPU signaler by traversing the Unreconciled
     * Attribute Lists. */
    for (i = 0; i < objAttrList->numCoreAttrList; ++i) {
        LwSciSyncCoreAttrList coreAttrList = objAttrList->coreAttrList[i];

        if (LwSciSyncCoreAttrListHasSignalerPerm(&coreAttrList)) {
            /* This is the Signaler's attribute list */
            isCpuSignaler = coreAttrList.attrs.needCpuAccess;
            break;
        }
    }

    /* Now set the key on the Reconciled Attribute List */
    newCoreAttrList->signalerUseExternalPrimitive = (!isCpuSignaler &&
            (reconciledPrimitiveType ==
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint));

    LWSCI_FNEXIT("");
}
#endif

/** Reconcile the UseExternalPrimitive private key */
static void ReconcileExportIpcEndpoint(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    size_t i = 0U;

    for (i = 0U; i < objAttrList->numCoreAttrList; ++i) {
        if (LwSciSyncCoreIpcTableHasC2C(
                &objAttrList->coreAttrList[i].ipcTable)) {
            newObjAttrList->coreAttrList->lastExport =
                objAttrList->coreAttrList[i].lastExport;
            break;
        }
    }
}

static bool TraveledViaC2C(
    const LwSciSyncCoreAttrListObj* objAttrList)
{
    size_t i = 0U;
    bool result = false;

    for (i = 0U; i < objAttrList->numCoreAttrList; ++i) {
        if ((LwSciSyncCoreIpcTableHasC2C(
                &objAttrList->coreAttrList[i].ipcTable)) ||
               (objAttrList->coreAttrList[i].lastExport != 0U)) {
            result = true;
            goto fn_exit;
        }
    }

fn_exit:
    return result;
}

/******************************************************
 *            Public interfaces definition
 ******************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListReconcile(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    LwSciSyncAttrList* newReconciledList,
    LwSciSyncAttrList* newConflictList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList newUnreconciledAttrList = NULL;
    LwSciSyncAttrList multiSlotAttrList = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciSyncCoreAttrListObj* newObjAttrList = NULL;
    LwSciSyncCoreAttrList* newCoreAttrList = NULL;
    LwSciSyncAccessPerm reconciledPerm;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = AttrListReconcileCheckArgs(inputArray, inputCount,
            newReconciledList, newConflictList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ilwalid_args;
    }

    error = LwSciSyncCoreAttrListsLock(inputArray, inputCount);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ilwalid_args;
    }

    /** prepare new AttrLists */
    error = PrepareObjAttrList(inputArray, inputCount,
            &newUnreconciledAttrList, &objAttrList, &multiSlotAttrList,
            &newObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

    /* Determine the reconciled permissions */
    error = ReconcilePerms(objAttrList, newObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }
    reconciledPerm = newObjAttrList->coreAttrList->attrs.actualPerm;
    if (LwSciSyncAccessPerm_WaitSignal != reconciledPerm) {
#if (LW_IS_SAFETY == 0)
        LwSciSyncCoreAttrs* attrs = &objAttrList->coreAttrList->attrs;
        attrs->keyState[LwSciSyncCoreKeyToIndex(
                (int32_t)LwSciSyncAttrKey_RequiredPerm)] =
                LwSciSyncCoreAttrKeyState_Conflict;
#endif
        LWSCI_ERR_STR("Invalid signaler/waiter perms.\n");
        error = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

    /* Determine the RequiredDeterministicFences key */
    ReconcileRequireDeterministicFences(objAttrList, newObjAttrList);

    /* Determine the WaiterRequireTimestamps key.
     * There is a dependency here where this must occur prior to performing
     * reconciliation of the primitive, as the timestamp format specifying
     * whether a primitive supports timetamps must be accounted for. */
    error = ReconcileWaiterRequireTimestamps(objAttrList, newObjAttrList);
    if (LwSciError_Success != error) {
        goto fail;
    }

    /* Determine the reconciled primitive */
    error = ReconcilePrimitiveInfo(objAttrList, newObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

    error = ReconcileSemaAttrList(objAttrList,
            newObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

    ReconcileWaiterContextInsensitiveFenceExports(objAttrList, newObjAttrList);

    /* Reconcile the SignalerTimestampInfo/SignalerTimestampInfoMulti keys */
    error = ReconcileSignalerTimestampInfo(objAttrList, newObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

    /** Create IPC perm table */
    newCoreAttrList = newObjAttrList->coreAttrList;
    error = CopyIpcPermTable(objAttrList, newCoreAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

    /* Reconcile LwSciSyncInternalAttrKey_EngineArray, which depends on the
     * IPC table */
    error = ReconcileEngineArray(objAttrList, newObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

#ifdef LWSCISYNC_EMU_SUPPORT
    error = ReconcileUseExternalPrimitiveInfo(
        objAttrList, newObjAttrList, multiSlotAttrList);
    if (error != LwSciError_Success) {
        goto fail;
    }
#else
    /* Determine whether we need to allocate a primitive or not */
    ReconcileUseExternalPrimitive(objAttrList, newObjAttrList);
#endif

    ReconcileExportIpcEndpoint(objAttrList, newObjAttrList);

    newObjAttrList->state = LwSciSyncCoreAttrListState_Reconciled;
    newObjAttrList->writable = false;

    *newReconciledList = multiSlotAttrList;

fail:
    {
        LwSciError err = LwSciError_Success;

        if (LwSciError_ReconciliationFailed == error) {
            // Reconciliation failure branch, should return newConflictList for
            // non-safety builds
            *newReconciledList = NULL;
#if (LW_IS_SAFETY == 0)
            *newConflictList = newUnreconciledAttrList;
            objAttrList->state = LwSciSyncCoreAttrListState_Conflict;
            objAttrList->writable = false;
#else
            *newConflictList = NULL;
            LwSciSyncAttrListFree(newUnreconciledAttrList);
#endif
            LwSciSyncAttrListFree(multiSlotAttrList);
        } else if (LwSciError_Success != error) {
            // For all other failures do clean-up
            *newReconciledList = NULL;
            *newConflictList = NULL;
            LwSciSyncAttrListFree(multiSlotAttrList);
            LwSciSyncAttrListFree(newUnreconciledAttrList);
        } else {
            // LwSciError_Success branch, clean-up unreconciled list
            *newConflictList = NULL;
            LwSciSyncAttrListFree(newUnreconciledAttrList);
        }

        err = LwSciSyncCoreAttrListsUnlock(inputArray, inputCount);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Could not unlock Attribute Lists\n");
            LwSciCommonPanic();
        }

        LWSCI_INFO("*newReconciledList: %p\n", *newReconciledList);
#if (LW_IS_SAFETY == 0)
        LWSCI_INFO("*newConflictList: %p\n", *newConflictList);
#endif
    }

ilwalid_args:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncCoreAttrListValidateReconciledWithLocks(
    LwSciSyncAttrList reconciledAttrList,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool acquireLocks,
    bool* isReconciledListValid)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList newUnreconciledAttrList = NULL;
    LwSciSyncCoreAttrListObj* unreconciledObjAttrList = NULL;
    LwSciSyncCoreAttrListObj* reconciledObjAttrList = NULL;
    LwSciSyncCoreAttrListObj tempObjAttrList = {0};

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = AttrListValidateReconciledCheckArgs(
            reconciledAttrList, inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount, isReconciledListValid);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (0U != inputUnreconciledAttrListCount) {
        error = LwSciSyncCoreAttrListAppendUnreconciledWithLocks(
                inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
                acquireLocks, &newUnreconciledAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        LwSciSyncCoreAttrListGetObjFromRef(newUnreconciledAttrList,
                                                   &unreconciledObjAttrList);
    }

    LwSciSyncCoreAttrListGetObjFromRef(reconciledAttrList,
            &reconciledObjAttrList);

    if (NULL != unreconciledObjAttrList) {
        /** Ensure all attr list belong to same module */
        error = EnsureSameModule(unreconciledObjAttrList,
                reconciledObjAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    LWSCI_INFO("reconciledAttrList: %p\n", reconciledAttrList);
    LWSCI_INFO("inputUnreconciledAttrListArray: %p\n",
            inputUnreconciledAttrListArray);
    LWSCI_INFO("inputUnreconciledAttrListCount: %p\n",
            inputUnreconciledAttrListCount);

    /* Set initially to false */
    *isReconciledListValid = false;

    if (NULL != unreconciledObjAttrList) {
        bool timestampRequired = false;
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        tempObjAttrList.coreAttrList =
                (LwSciSyncCoreAttrList*)LwSciCommonCalloc(1U,
                sizeof(LwSciSyncCoreAttrList));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        if (NULL == tempObjAttrList.coreAttrList) {
            LWSCI_ERR_STR("failed to allocate memory.\n");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        tempObjAttrList.numCoreAttrList = 1U;
        error = ReconcilePerms(unreconciledObjAttrList,
                &tempObjAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        error = AttrListCheckTimestampRequired(unreconciledObjAttrList,
            reconciledObjAttrList, &timestampRequired);
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
        tempObjAttrList.coreAttrList->attrs.waiterRequireTimestamps = timestampRequired;

        ReconcileRequireDeterministicFences(unreconciledObjAttrList,
            &tempObjAttrList);
    }

    /* Ignore any value set in RequiredPerm */

    /* Verify actual perm */
    if (LwSciSyncCorePermLessThan(
            reconciledObjAttrList->coreAttrList->attrs.actualPerm,
            LwSciSyncAccessPerm_WaitOnly)) {
        LWSCI_ERR_STR("Reconciled attri list must have at least wait permissions\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL != unreconciledObjAttrList) {
        if (LwSciSyncCorePermLessThan(
                reconciledObjAttrList->coreAttrList->attrs.actualPerm,
                tempObjAttrList.coreAttrList->attrs.actualPerm)) {
            LWSCI_ERR_STR("Insufficient Reconciled list permissions\n");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        /* Verify CPU access. It is failure if unreconciled attrList has
         * requested for CPU access, but is set to false in reconciled list */
        if (tempObjAttrList.coreAttrList->attrs.needCpuAccess &&
                !reconciledObjAttrList->coreAttrList->attrs.needCpuAccess) {
            LWSCI_ERR_STR("Insufficient Reconciled cpu access permissions\n");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        /* Verify RequireDeterministicFences. It is a failure if the
         * unreconciled LwSciSyncAttrList requested for deterministic
         * primitives, but is set to false in the reconciled list. */
        if (tempObjAttrList.coreAttrList->attrs.requireDeterministicFences &&
            !reconciledObjAttrList->coreAttrList->attrs.requireDeterministicFences) {
            LWSCI_ERR_STR("LwSciSyncAttrKey_RequireDeterministicFences was not satisfied");
            error = LwSciError_BadParameter;
            goto fn_exit;
        }

        error = ValidateReconciledSignalerTimestampInfo(unreconciledObjAttrList,
            reconciledObjAttrList);
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
    }
    error = ValidateReconciledSignalerPrimitiveCount(unreconciledObjAttrList,
            reconciledObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* Verify the reconciled primitive type matches with requested type */
    error = ValidateReconciledPrimitiveType(unreconciledObjAttrList,
            reconciledObjAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

#ifdef LWSCISYNC_EMU_SUPPORT
    error = ValidateReconciledExternalPrimitiveInfo(unreconciledObjAttrList,
            reconciledObjAttrList);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }
#endif

    if (NULL != unreconciledObjAttrList) {
        error = ValidateReconciledSemaAttrList(
                unreconciledObjAttrList,
                reconciledObjAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    if (NULL != unreconciledObjAttrList) {
        /* Verify the timestampBufAttrList */
        error = ValidateReconciledTimestampBufAttrList(unreconciledObjAttrList,
                reconciledObjAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    if (NULL != unreconciledObjAttrList) {
        error = ValidateReconciledEngineArray(unreconciledObjAttrList,
                reconciledObjAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    /* Set to true */
    *isReconciledListValid = true;

    LWSCI_INFO("*isReconciledListValid: %d\n", *isReconciledListValid);

fn_exit:
    if (NULL != tempObjAttrList.coreAttrList) {
        LwSciCommonFree(tempObjAttrList.coreAttrList);
    }
    LwSciSyncAttrListFree(newUnreconciledAttrList);

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListValidateReconciled(
    LwSciSyncAttrList reconciledAttrList,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool* isReconciledListValid)
{
    LwSciError error = LwSciError_Success;

    if ((NULL == inputUnreconciledAttrListArray) ||
            (0U == inputUnreconciledAttrListCount)) {
        LWSCI_ERR_STR("Empty input unreconciled list array\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreAttrListValidateReconciledWithLocks(reconciledAttrList,
            inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
            true, isReconciledListValid);
    if (error != LwSciError_Success){
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LwSciError LwSciSyncAttrListIsReconciled(
    LwSciSyncAttrList attrList,
    bool* isReconciled)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    LWSCI_FNENTRY("");

    /** validate all input args */
    if (NULL == isReconciled) {
        LWSCI_ERR_STR("Invalid argument: isReconciled: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("attrList: %p\n", attrList);
    LWSCI_INFO("isReconciled: %p\n", isReconciled);

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    *isReconciled = (objAttrList->state ==
            LwSciSyncCoreAttrListState_Reconciled);

    LWSCI_INFO("*isReconciled : %d\n", *isReconciled);

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ReconcilePerms(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    uint64_t perm = 0U;

    LWSCI_FNENTRY("");

    newObjAttrList->coreAttrList->attrs.needCpuAccess = false;

    /* Set key value */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        newObjAttrList->coreAttrList->attrs.needCpuAccess =
            (newObjAttrList->coreAttrList->attrs.needCpuAccess ||
            objAttrList->coreAttrList[i].attrs.needCpuAccess);

        perm = (perm |
             (uint64_t)objAttrList->coreAttrList[i].attrs.requiredPerm);
    }

    LwSciCommonMemcpyS(&newObjAttrList->coreAttrList->attrs.actualPerm,
                       sizeof(newObjAttrList->coreAttrList->attrs.actualPerm),
                       &perm, sizeof(perm));

    /* Error on invalid values */
    if ((0U == ((size_t)newObjAttrList->coreAttrList->attrs.actualPerm)) ||
        (!LwSciSyncCorePermLEq(
            newObjAttrList->coreAttrList->attrs.actualPerm,
            LwSciSyncAccessPerm_WaitSignal))) {
        LWSCI_ERR_STR("Invalid permissions\n");
        LWSCI_ERR_ULONG("perms: \n", newObjAttrList->coreAttrList->attrs.actualPerm);
        error = LwSciError_ReconciliationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* Update key states and sizes */
    newObjAttrList->coreAttrList->attrs.keyState[
            LwSciSyncCoreKeyToIndex((uint32_t) LwSciSyncAttrKey_NeedCpuAccess)]
            = LwSciSyncCoreAttrKeyState_Reconciled;
    newObjAttrList->coreAttrList->attrs.valSize[
            LwSciSyncCoreKeyToIndex((uint32_t) LwSciSyncAttrKey_NeedCpuAccess)]
            = sizeof(bool);

    newObjAttrList->coreAttrList->attrs.keyState[
            LwSciSyncCoreKeyToIndex((uint32_t) LwSciSyncAttrKey_ActualPerm)] =
            LwSciSyncCoreAttrKeyState_Reconciled;
    newObjAttrList->coreAttrList->attrs.valSize[
            LwSciSyncCoreKeyToIndex((uint32_t) LwSciSyncAttrKey_ActualPerm)] =
            sizeof(LwSciSyncAccessPerm);
fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static void ReconcileRequireDeterministicFences(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    size_t i = 0U;
    const size_t attrIndex = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncAttrKey_RequireDeterministicFences);

    LWSCI_FNENTRY("");

    newObjAttrList->coreAttrList->attrs.requireDeterministicFences = false;

    /* Set key value */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        newObjAttrList->coreAttrList->attrs.requireDeterministicFences =
            (newObjAttrList->coreAttrList->attrs.requireDeterministicFences ||
            objAttrList->coreAttrList[i].attrs.requireDeterministicFences);
    }

    /* Update key states and sizes */
    newObjAttrList->coreAttrList->attrs.keyState[attrIndex] =
        LwSciSyncCoreAttrKeyState_Reconciled;
    newObjAttrList->coreAttrList->attrs.valSize[attrIndex] = sizeof(bool);

    LWSCI_FNEXIT("");
}

/******************************************************
 *             Core interfaces definition
 ******************************************************/

LwSciError LwSciSyncCoreFillTimestampBufAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t pairCount = 0U;
    bool cpuAccess = false;
    bool timestampBufNeeded = false;
    uint64_t bufSize = 4096U;
    uint64_t bufAlign = 16U;
    uint8_t subStatus = OP_FAIL;
    LwSciSyncCoreAttrList* coreAttrList = NULL;
    LwSciBufModule bufModule = NULL;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_Ilwalid;
    LwSciBufAttrKeyValuePair pairArray[] = {
        {    .key = LwSciBufGeneralAttrKey_Types,
             .value = (const void*) &bufType,
             .len = sizeof(bufType)
        },
        {    .key = LwSciBufGeneralAttrKey_NeedCpuAccess,
             .value = (const void*) &cpuAccess,
             .len = sizeof(cpuAccess)
        },
        {    .key = LwSciBufGeneralAttrKey_RequiredPerm,
             .value = (const void*) &perm,
             .len = sizeof(perm)
        },
        {    .key = LwSciBufRawBufferAttrKey_Align,
             .value = (const void*) &bufAlign,
             .len = sizeof(bufAlign)
        },
        /** Buffer size is last entry as waiters skip setting this attr key */
        {    .key = LwSciBufRawBufferAttrKey_Size,
             .value = (const void*) &bufSize,
             .len = sizeof(bufSize)
        },
    };

    LwSciSyncCoreModuleGetBufModule(objAttrList->module, &bufModule);

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        coreAttrList = &objAttrList->coreAttrList[i];
        /** Skip already created buf attr list */
        if (NULL != coreAttrList->timestampBufAttrList) {
            continue;
        }

        pairCount = sizeof(pairArray)/sizeof(pairArray[0]);

        /** Check if the timestamp buffer is needed and set appropriate perm */
        timestampBufNeeded = coreAttrList->attrs.waiterRequireTimestamps;

        if (LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList)) {
            LwSciSyncCoreAttrKeyState keyState =
                coreAttrList->attrs.keyState[LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_SignalerTimestampInfo)];
            LwSciSyncCoreAttrKeyState keyStateMulti =
                coreAttrList->attrs.keyState[LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti)];

            if ((keyState != LwSciSyncCoreAttrKeyState_Empty) ||
                    (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty)) {
                timestampBufNeeded = true;
                perm = LwSciBufAccessPerm_ReadWrite;
            }
        } else {
            perm = LwSciBufAccessPerm_Readonly;
            /* Waiters don't set the buffer size key */
            u64Sub(pairCount, 1U, &pairCount, &subStatus);
            if (OP_SUCCESS != subStatus) {
                error = LwSciError_Overflow;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }

        if (false == timestampBufNeeded) {
            continue;
        }

        cpuAccess = coreAttrList->attrs.needCpuAccess;

        error = LwSciBufAttrListCreate(bufModule,
                &coreAttrList->timestampBufAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        error = LwSciBufAttrListSetAttrs(coreAttrList->timestampBufAttrList,
                pairArray, pairCount);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    return error;
}

static void GetTimestampInfo(
    const LwSciSyncCoreAttrs* reconciledAttrs,
    const LwSciSyncAttrValTimestampInfo** timestampInfo)
{
    const size_t multiIndex = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
    const size_t index = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo);
    uint64_t keyStateMulti = reconciledAttrs->keyState[multiIndex];
    uint64_t keyState = reconciledAttrs->keyState[index];

    if (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty) {
        *timestampInfo =
            &reconciledAttrs->signalerTimestampInfoMulti[0];
        goto fn_exit;
    }
    if (LwSciSyncCoreAttrKeyState_Empty != keyState) {
        *timestampInfo = &reconciledAttrs->signalerTimestampInfo;
        goto fn_exit;
    }

fn_exit:
    return;
}

static LwSciError ValidateReconciledTimestampBufAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciBufAttrList reconciledAttrList = NULL;
    LwSciBufAttrList* unreconciledAttrLists = NULL;
    bool isReconcileListValid = false;
    size_t numUnreconciledAttrList = 0U;
    size_t i = 0U;
    uint8_t addStatus = OP_FAIL;

    bool timestampRequired =
        newObjAttrList->coreAttrList->attrs.waiterRequireTimestamps;
    const LwSciSyncAttrValTimestampInfo* timestampInfo = NULL;

    /* If timestamps weren't required, then there's no need to deal with
     * timestamps at all. */
    if (false == timestampRequired) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    GetTimestampInfo(&newObjAttrList->coreAttrList->attrs, &timestampInfo);
    if (timestampInfo == NULL) {
        LWSCI_ERR_STR("Reconciled LwSciSyncAttrList does not have a timestamp "
                "format set");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /* If the timestamp format is LwSciSyncTimestampFormat_EmbeddedInPrimitive,
     * then there is no separate timestampBufAttrList. */
    if (timestampInfo->format == LwSciSyncTimestampFormat_EmbeddedInPrimitive) {
        goto fn_exit;
    }

    reconciledAttrList = newObjAttrList->coreAttrList->timestampBufAttrList;
    if (NULL == reconciledAttrList) {
        LWSCI_ERR_STR("reconciled attr list does not have a timestamp buffer.\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreFillTimestampBufAttrList(objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    unreconciledAttrLists = (LwSciBufAttrList*) LwSciCommonCalloc(
            objAttrList->numCoreAttrList, sizeof(LwSciBufAttrList));
    if (NULL == unreconciledAttrLists) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    numUnreconciledAttrList = 0U;
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        if (NULL != objAttrList->coreAttrList[i].timestampBufAttrList) {
            unreconciledAttrLists[numUnreconciledAttrList] =
                    objAttrList->coreAttrList[i].timestampBufAttrList;
            u64Add(numUnreconciledAttrList, 1U,
                   &numUnreconciledAttrList, &addStatus);
            if (OP_SUCCESS != addStatus) {
                LWSCI_ERR_STR("numUnreconciledAttrList value out of range.\n");
                error = LwSciError_Overflow;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }
    }

    error = LwSciBufAttrListValidateReconciled(reconciledAttrList,
            unreconciledAttrLists, numUnreconciledAttrList,
            &isReconcileListValid);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (false == isReconcileListValid) {
        LWSCI_ERR_STR("Invalid reconciled timestamp buffer attr list.\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LwSciCommonFree(unreconciledAttrLists);
    return error;
}

static LwSciError ValidatePublicKeyValues(
    const LwSciSyncCoreAttrListObj* objAttrList)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreAttrList* coreAttrList = NULL;
    size_t i = 0U;
    LwSciSyncAccessPerm perm;

    LWSCI_FNENTRY("");

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        coreAttrList = &objAttrList->coreAttrList[i];
        /** Check Value of LwSciSyncAttrKey_RequiredPerm key */
        perm = coreAttrList->attrs.requiredPerm;
        if (!CheckRequiredPermValues(perm)) {
            LWSCI_ERR_ULONG("Invalid value for LwSciSyncAttrKey_RequiredPerm: \n",
                    (uint64_t)perm);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ValidateInternalKeyValues(
    const LwSciSyncCoreAttrListObj* objAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t keyIdx = 0U;
    size_t size = 0U;
    size_t primitiveCnt = 0U;
    LwSciSyncCoreAttrKeyState keyState;
    const LwSciSyncInternalAttrValPrimitiveType* primitiveInfo = NULL;

    bool hasSignalerTimestampInfo = false;
    bool hasSignalerTimestampInfoMulti = false;

    LWSCI_FNENTRY("");

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        const LwSciSyncCoreAttrList* coreAttrList =
            &objAttrList->coreAttrList[i];
        size_t signalerPrimitiveInfoLen = 0U;
        size_t engineArrayLen = 0U;

        /** Check Value of LwSciSyncInternalAttrKey_SignalerPrimitiveInfo key */
        primitiveInfo = coreAttrList->attrs.signalerPrimitiveInfo;
        keyIdx = LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
        signalerPrimitiveInfoLen = coreAttrList->attrs.valSize[keyIdx] /
                LwSciSyncCoreKeyInfo[keyIdx].elemSize;
        error = LwSciSyncCoreCheckPrimitiveValues(primitiveInfo,
            signalerPrimitiveInfoLen);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Invalid value for "
                    "LwSciSyncInternalAttrKey_SignalerPrimitiveInfo");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        /** Check Value of LwSciSyncInternalAttrKey_WaiterPrimitiveInfo key */
        primitiveInfo = coreAttrList->attrs.waiterPrimitiveInfo;
        keyIdx = LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_WaiterPrimitiveInfo);
        size = coreAttrList->attrs.valSize[keyIdx] /
                 LwSciSyncCoreKeyInfo[keyIdx].elemSize;
        error = LwSciSyncCoreCheckPrimitiveValues(primitiveInfo, size);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Invalid value for "
                    "LwSciSyncInternalAttrKey_WaiterPrimitiveInfo\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        /** Check Value of LwSciSyncInternalAttrKey_SignalerPrimitiveCount key*/
        primitiveCnt = coreAttrList->attrs.signalerPrimitiveCount;
        keyIdx = LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_SignalerPrimitiveCount);
        keyState = coreAttrList->attrs.keyState[keyIdx];
        if ((primitiveCnt > UINT32_MAX) ||
                ((LwSciSyncCoreAttrKeyState_Empty != keyState) &&
                LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList) &&
                (0U == primitiveCnt))) {
            LWSCI_ERR_ULONG("Invalid value for "
                    "LwSciSyncInternalAttrKey_SignalerPrimitiveCount: \n",
                    primitiveCnt);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        if (LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList)) {
            /** Check Value of LwSciSyncInternalAttrKey_SignalerTimestampInfo key.
             *  Only do so when the attrList is a signaler list. */
            keyIdx = LwSciSyncCoreKeyToIndex(
                    (uint32_t) LwSciSyncInternalAttrKey_SignalerTimestampInfo);
            keyState = coreAttrList->attrs.keyState[keyIdx];
            size = coreAttrList->attrs.valSize[keyIdx] /
                 LwSciSyncCoreKeyInfo[keyIdx].elemSize;

            if (keyState != LwSciSyncCoreAttrKeyState_Empty) {
                const LwSciSyncAttrValTimestampInfo* timestampInfo =
                    &coreAttrList->attrs.signalerTimestampInfo;
                error = LwSciSyncValidateTimestampInfo(timestampInfo, size);
                if (LwSciError_Success != error) {
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto fn_exit;
                }

                hasSignalerTimestampInfo = true;
            }
        }

        if (LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList)) {
            size_t signalerTimestampInfoMultiLen = 0U;
            /** Check the value of the
             * LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti attribute
             * key. Only do so when the attrList is a signaler list. */
            keyIdx = LwSciSyncCoreKeyToIndex(
                    (uint32_t) LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
            keyState = coreAttrList->attrs.keyState[keyIdx];
            signalerTimestampInfoMultiLen = coreAttrList->attrs.valSize[keyIdx] /
                 LwSciSyncCoreKeyInfo[keyIdx].elemSize;

            if (keyState != LwSciSyncCoreAttrKeyState_Empty) {
                const LwSciSyncAttrValTimestampInfo* timestampInfo =
                    coreAttrList->attrs.signalerTimestampInfoMulti;

                error = LwSciSyncValidateTimestampInfo(timestampInfo,
                    signalerTimestampInfoMultiLen);
                if (error != LwSciError_Success) {
                    goto fn_exit;
                }

                /* Check if the dependent key has been set.
                 * This key depends on the value of SignalerPrimitiveInfo,
                 * which must contain the same number of items. */
                if (signalerTimestampInfoMultiLen != signalerPrimitiveInfoLen) {
                    LWSCI_ERR_STR("LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti "
                        "and LwSciSyncInternalAttrKey_SignalerPrimitiveInfo "
                        "must have the same number of items.");
                    error = LwSciError_BadParameter;
                    goto fn_exit;
                }

                hasSignalerTimestampInfoMulti = true;
            }
        }

        /** The SignalerTimestampInfo and SignalerTimestampInfoMulti attribute
         * keys are mutually exclusive across _all_ attribute lists ilwolved
         * in reconciliation. */
        if (hasSignalerTimestampInfo && hasSignalerTimestampInfoMulti) {
            LWSCI_ERR_STR("The LwSciSyncInternalAttrKey_SignalerTimestampInfo "
                    "and LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti "
                    "attribute keys are mutually exclusive.");
            error = LwSciError_BadParameter;
            goto fn_exit;
        }

        /** Check Value of LwSciSyncInternalAttrKey_EngineArray key*/
        keyIdx = LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_EngineArray);
        engineArrayLen = coreAttrList->attrs.valSize[keyIdx] /
                LwSciSyncCoreKeyInfo[keyIdx].elemSize;
        error = LwSciSyncCoreCheckHwEngineValues(
                coreAttrList->attrs.engineArray, engineArrayLen);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static bool TimestampScalingIsEqual(
    const LwSciSyncTimestampScaling* left,
    const LwSciSyncTimestampScaling* right)
{
    return
        ((left->scalingFactorNumerator == right->scalingFactorNumerator) &&
         (left->scalingFactorDenominator == right->scalingFactorDenominator) &&
         (left->sourceOffset == right->sourceOffset));
}

static bool TimestampInfoIsEqual(
    const LwSciSyncAttrValTimestampInfo* left,
    const LwSciSyncAttrValTimestampInfo* right)
{
    return
        ((left->format == right->format) &&
         TimestampScalingIsEqual(&left->scaling, &right->scaling));
}

static LwSciError ValidateReconciledSignalerTimestampInfo(
    const LwSciSyncCoreAttrListObj* unreconciledObjAttrList,
    const LwSciSyncCoreAttrListObj* reconciledObjAttrList)
{
    LwSciError error = LwSciError_Success;

    size_t i = 0U;

    const LwSciSyncCoreAttrs* attrs;
    LwSciSyncCoreAttrKeyState keyState = LwSciSyncCoreAttrKeyState_Empty;
    LwSciSyncCoreAttrKeyState keyStateMulti = LwSciSyncCoreAttrKeyState_Empty;

    bool reconciledHasTimestampInfo = false;
    bool reconciledHasTimestampInfoMulti = false;

    bool hasMatchingFormat = false;
    const LwSciSyncAttrValTimestampInfo* timestampInfo = NULL;

    const size_t multiIndex = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
    const size_t index = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo);
    const size_t primitiveInfoIndex = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);

    if (reconciledObjAttrList->coreAttrList->attrs.waiterRequireTimestamps == false) {
        goto fn_exit;
    }

    /* Compare this against the reconciled attribute list */
    attrs = &reconciledObjAttrList->coreAttrList->attrs;
    keyState = attrs->keyState[index];
    keyStateMulti = attrs->keyState[multiIndex];

    reconciledHasTimestampInfo =
        (keyState != LwSciSyncCoreAttrKeyState_Empty);
    reconciledHasTimestampInfoMulti =
        (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty);

    if (reconciledHasTimestampInfo || reconciledHasTimestampInfoMulti) {
        GetTimestampInfo(&reconciledObjAttrList->coreAttrList->attrs, &timestampInfo);

        /* Since LwSciSync assumes that only 1 signaler is ever provided, if an
         * unreconciled list corresponds to a signaler and that signaler has a
         * different attribute key set describing the signaler's timestamp
         * information than that of the reconciled attribute list, then validation
         * must fail. */
        for (i = 0U; i < unreconciledObjAttrList->numCoreAttrList; i++) {
            LwSciSyncCoreAttrList* coreAttrList =
                &unreconciledObjAttrList->coreAttrList[i];
            attrs = &coreAttrList->attrs;

            /* Find signaler unreconciled attribute list */
            if (LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList)) {
                bool hasTimestampInfo = false;
                bool hasTimestampInfoMulti = false;

                /* Check which one of the attributes were set. */
                keyState = attrs->keyState[index];
                keyStateMulti = attrs->keyState[multiIndex];

                hasTimestampInfo =
                    (keyState != LwSciSyncCoreAttrKeyState_Empty);
                hasTimestampInfoMulti =
                    (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty);

                if (hasTimestampInfo && hasTimestampInfoMulti) {
                    LWSCI_ERR_STR("Unreconciled attribute list cannot specify both "
                            "LwSciSyncInternalAttrKey_SignalerTimestampInfo and "
                            "LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti");
                    error = LwSciError_BadParameter;
                    goto fn_exit;
                }

                 /* Whatever key chosen in the reconciled list must also match
                  * the key provided in the unreconciled list */
                if ((hasTimestampInfo && reconciledHasTimestampInfoMulti) ||
                        (hasTimestampInfoMulti && reconciledHasTimestampInfo)) {
                    error = LwSciError_BadParameter;
                    goto fn_exit;
                }

                if (hasTimestampInfoMulti) {
                    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitive =
                        reconciledObjAttrList->coreAttrList->attrs.signalerPrimitiveInfo[0];
                    size_t primitiveIndex = 0U;

                    const LwSciSyncAttrValTimestampInfo* signalerTimestampInfo =
                        coreAttrList->attrs.signalerTimestampInfoMulti;
                    const size_t timestampInfoLen =
                        (coreAttrList->attrs.valSize[multiIndex] /
                         sizeof(coreAttrList->attrs.signalerTimestampInfoMulti[0]));

                    const size_t primitiveLen =
                        (coreAttrList->attrs.valSize[primitiveInfoIndex] /
                         sizeof(coreAttrList->attrs.signalerPrimitiveInfo[0]));

                    /* Ensure that the lengths of the arrays are equal */
                    if (timestampInfoLen != primitiveLen) {
                        error = LwSciError_BadParameter;
                        LWSCI_ERR_STR(
                            "Lengths of LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti "
                            "and LwSciSyncInternalAttrKey_SignalerPrimitiveInfo do not match");
                        goto fn_exit;
                    }

                    error = LwSciSyncGetSignalerPrimitiveInfoIndex(coreAttrList,
                            reconciledPrimitive, &primitiveIndex);
                    if (error != LwSciError_Success) {
                        LWSCI_ERR_STR(
                            "The reconciled primitive does not satisfy the given primitives");
                        goto fn_exit;
                    }

                    /* Check if the timestamp info at that index matches */
                    if (TimestampInfoIsEqual(&signalerTimestampInfo[primitiveIndex], timestampInfo)) {
                        hasMatchingFormat = true;
                        break;
                    }
                }
                if (hasTimestampInfo) {
                    const LwSciSyncAttrValTimestampInfo* unreconciledInfo =
                        &coreAttrList->attrs.signalerTimestampInfo;

                    if (TimestampInfoIsEqual(unreconciledInfo, timestampInfo)) {
                        hasMatchingFormat = true;
                        break;
                    }
                }

                if ((hasTimestampInfo || hasTimestampInfoMulti) && !hasMatchingFormat) {
                    LWSCI_ERR_STR("Timestamp formats are not the same");
                    error = LwSciError_BadParameter;
                    goto fn_exit;
                }
            }
        }
    }

fn_exit:
    return error;
}

static LwSciError ValidateReconciledSignalerPrimitiveCount(
    const LwSciSyncCoreAttrListObj* unreconciledObjAttrList,
    const LwSciSyncCoreAttrListObj* reconciledObjAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;

    uint64_t keyIdx = 0U;
    LwSciSyncCoreAttrKeyState keyState = LwSciSyncCoreAttrKeyState_Conflict;
    size_t unreconciledNum =
            (NULL == unreconciledObjAttrList) ?
            0U : unreconciledObjAttrList->numCoreAttrList;

    keyIdx = LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveCount);
    keyState = reconciledObjAttrList->coreAttrList->attrs.keyState[keyIdx];

    if ((LwSciSyncCoreAttrKeyState_Empty != keyState) &&
            (0U == reconciledObjAttrList->coreAttrList->attrs.signalerPrimitiveCount)) {
        LWSCI_ERR_UINT("Invalid value for "
                "LwSciSyncInternalAttrKey_SignalerPrimitiveCount: \n",
                reconciledObjAttrList->coreAttrList->attrs.signalerPrimitiveCount);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    for (i = 0U; i < unreconciledNum; ++i) {
        const LwSciSyncCoreAttrList* coreAttrList =
                &unreconciledObjAttrList->coreAttrList[i];
        const LwSciSyncCoreAttrList* reconciledCoreAttrList =
                reconciledObjAttrList->coreAttrList;
        uint32_t reconciledSignalerPrimitiveCount =
                reconciledCoreAttrList->attrs.signalerPrimitiveCount;

        if (LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList) &&
                (((coreAttrList->attrs.needCpuAccess &&
                (reconciledSignalerPrimitiveCount != 1U))) ||
                (!coreAttrList->attrs.needCpuAccess &&
                (reconciledSignalerPrimitiveCount !=
                coreAttrList->attrs.signalerPrimitiveCount)))) {
            LWSCI_ERR_UINT("Invalid value for "
                    "LwSciSyncInternalAttrKey_SignalerPrimitiveCount: \n",
                    reconciledObjAttrList->coreAttrList->attrs.signalerPrimitiveCount);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    return error;
}

static LwSciError ValidateReconciledPrimitiveType(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* recObjAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrs attrs;
    size_t i = 0U;
    size_t reconciledPrimitiveMask = 0U;
    size_t signalerPrimitiveMask = 0U;
    size_t waiterPrimitiveMask = 0U;
    size_t signalIndex = LwSciSyncCoreKeyToIndex(
            (uint32_t) LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
    size_t waitIndex = LwSciSyncCoreKeyToIndex(
            (uint32_t) LwSciSyncInternalAttrKey_WaiterPrimitiveInfo);
    const LwSciSyncCoreAttrList* reconciled = recObjAttrList->coreAttrList;
    size_t arrSize = 0U;

    LWSCI_FNENTRY("");

    attrs = reconciled->attrs;
    reconciledPrimitiveMask = CreateMaskFromPrimitiveType(
            attrs.signalerPrimitiveInfo,
            attrs.valSize[signalIndex] / sizeof(attrs.signalerPrimitiveInfo[0]));

    arrSize = attrs.valSize[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_WaiterPrimitiveInfo)];
    if (sizeof(LwSciSyncInternalAttrValPrimitiveType) != arrSize) {
        LWSCI_ERR_STR("Too many primitives in WaiterPrimitiveInfo\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    arrSize = attrs.valSize[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo)];
    if (sizeof(LwSciSyncInternalAttrValPrimitiveType) != arrSize) {
        LWSCI_ERR_STR("Too many primitives in SignalerPrimitiveInfo\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (attrs.signalerPrimitiveInfo[0] != attrs.waiterPrimitiveInfo[0]) {
        LWSCI_ERR_STR("Mismatch in signalerPrimitiveInfo vs waiterPrimitiveInfo\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* If deterministic primitives are requested, then the value of the
     * reconciled primitive should also be deterministic. */
    if (attrs.requireDeterministicFences) {
        LwSciSyncInternalAttrValPrimitiveType reconciledPrimitive =
            attrs.signalerPrimitiveInfo[0];

        LwSciSyncInternalAttrValPrimitiveType deterministicPrimitives[
            MAX_PRIMITIVE_TYPE];
        size_t deterministicPrimitiveMask = 0U;

        /** fill with invalid entries initially */
        for (i = 0U; i < MAX_PRIMITIVE_TYPE; i++) {
            deterministicPrimitives[i] =
                LwSciSyncInternalAttrValPrimitiveType_LowerBound;
        }

        LwSciSyncCoreGetDeterministicPrimitives(deterministicPrimitives,
            sizeof(deterministicPrimitives));
        deterministicPrimitiveMask = CreateMaskFromPrimitiveType(
            deterministicPrimitives,
            sizeof(deterministicPrimitives) / sizeof(deterministicPrimitives[0]));

        deterministicPrimitiveMask &= CreateMaskFromPrimitiveType(
            &reconciledPrimitive, 1U);

        if (deterministicPrimitiveMask == 0U) {
            LWSCI_ERR_STR("Unsupported deterministic primitive");
            error = LwSciError_BadParameter;
            goto fn_exit;
        }
    }

    if (NULL == objAttrList) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** TODO: Cannot pass false always as validation can be done upon receiving
     * reconciled list across chip */
    LwSciSyncCoreFillCpuPrimitiveInfo(objAttrList, TraveledViaC2C(recObjAttrList));

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        size_t timestampMask = 0U;
        attrs = objAttrList->coreAttrList[i].attrs;

        if (LwSciSyncCoreAttrListHasSignalerPerm(&objAttrList->coreAttrList[i])) {
            signalerPrimitiveMask = CreateMaskFromPrimitiveType(
                    attrs.signalerPrimitiveInfo,
                    attrs.valSize[signalIndex] / sizeof(attrs.signalerPrimitiveInfo[0]));

            /* Check if the reconciled primitive is a subset of the signaler mask */
            reconciledPrimitiveMask &= signalerPrimitiveMask;
        }

        if (LwSciSyncCoreAttrListHasWaiterPerm(&objAttrList->coreAttrList[i])) {
            waiterPrimitiveMask = CreateMaskFromPrimitiveType(
                attrs.waiterPrimitiveInfo,
                attrs.valSize[waitIndex] / sizeof(attrs.waiterPrimitiveInfo[0]));

            /* Check if the reconciled primitive is a subset of the waiter mask */
            reconciledPrimitiveMask &= waiterPrimitiveMask;
        }

        /* If timestamps are required, then the value of the reconciled primitive
         * should also be one that has the timestamp info specified. */
        if (reconciled->attrs.waiterRequireTimestamps) {
            const size_t multiIndex = LwSciSyncCoreKeyToIndex(
                (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
            const size_t index = LwSciSyncCoreKeyToIndex(
                (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo);
            const size_t primitiveInfoIndex = LwSciSyncCoreKeyToIndex(
                (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
            LwSciSyncCoreAttrKeyState keyStateMulti = attrs.keyState[multiIndex];
            LwSciSyncCoreAttrKeyState keyState = attrs.keyState[index];

            const LwSciSyncAttrValTimestampInfo* timestampInfo = NULL;
            size_t signalerPrimitiveInfoLen = 0U;
            size_t timestampInfoLen = 0U;

            if (LwSciSyncCoreAttrListHasSignalerPerm(&objAttrList->coreAttrList[i])) {
                if ((keyStateMulti == LwSciSyncCoreAttrKeyState_Empty) &&
                    (keyState == LwSciSyncCoreAttrKeyState_Empty)) {
                    continue;
                }
                if ((keyStateMulti != LwSciSyncCoreAttrKeyState_Empty) &&
                    (keyState != LwSciSyncCoreAttrKeyState_Empty)) {
                    LWSCI_ERR_STR("Mutually exclusive attribute keys used");
                    error = LwSciError_BadParameter;
                    goto fn_exit;
                }

                signalerPrimitiveInfoLen = attrs.valSize[primitiveInfoIndex] / sizeof(attrs.signalerPrimitiveInfo[0]);
                if (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty) {
                    timestampInfo = attrs.signalerTimestampInfoMulti;
                    timestampInfoLen =
                        (attrs.valSize[multiIndex] / sizeof(attrs.signalerTimestampInfoMulti[0]));

                    error = CreateMaskOfTimestampPrimitivesMulti(
                        attrs.signalerPrimitiveInfo, signalerPrimitiveInfoLen,
                        timestampInfo, timestampInfoLen, &timestampMask);
                    if (error != LwSciError_Success) {
                        goto fn_exit;
                    }

                    if ((attrs.valSize[signalIndex] / sizeof(attrs.signalerPrimitiveInfo[0])) !=
                        timestampInfoLen) {
                        LWSCI_ERR_STR("Mismatched lengths of SignalerTimestampInfoMulti and SignalerPrimitiveInfo");
                        error = LwSciError_BadParameter;
                        goto fn_exit;
                    }
                }
                if (keyState != LwSciSyncCoreAttrKeyState_Empty) {
                    timestampInfo = &attrs.signalerTimestampInfo;
                    timestampInfoLen =
                        (attrs.valSize[index] / sizeof(attrs.signalerTimestampInfo));

                    timestampMask = CreateMaskOfTimestampPrimitives(
                        attrs.signalerPrimitiveInfo, signalerPrimitiveInfoLen,
                        timestampInfo);
                }

                /* Check if the reconciled primitive is a subset of the timestamp
                 * mask */
                reconciledPrimitiveMask &= timestampMask;
            }
        }

        if (0U == reconciledPrimitiveMask) {
            LWSCI_ERR_STR("Invalid primitive type in input attr list\n");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ValidateReconciledEngineArray(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* recObjAttrList)
{
    LwSciError error = LwSciError_Success;

    /* Verify the LwSciSyncInternalAttrKey_EngineArray key */
    for (size_t i = 0U; i < objAttrList->numCoreAttrList; i++) {
        /* Find the attribute list corresponding to the current peer, if it
         * was provided. This attribute list will have an empty IPC route. */
        if (LwSciSyncCoreIpcTableRouteIsEmpty(&objAttrList->coreAttrList[i].ipcTable)) {
            const size_t index = LwSciSyncCoreKeyToIndex(
                    (int32_t)LwSciSyncInternalAttrKey_EngineArray);
            const LwSciSyncCoreAttrs* unreconciledAttrs =
                &objAttrList->coreAttrList[i].attrs;
            size_t unreconciledLen = unreconciledAttrs->valSize[index] /
                sizeof(unreconciledAttrs->engineArray[0]);

            const LwSciSyncCoreAttrs* reconciledAttrs =
                &recObjAttrList->coreAttrList->attrs;
            size_t reconciledLen = reconciledAttrs->valSize[index] /
                sizeof(reconciledAttrs->engineArray[0]);

            if (0U != unreconciledAttrs->valSize[index]) {
                /* Each entry of the EngineArray attribute key must be
                 * present in this view of the reconciled LwSciSyncAttrList. */
                for (size_t j = 0; j < unreconciledLen; j++) {
                    bool found = false;
                    for (size_t k = 0; k < reconciledLen; k++) {
                        if (LwSciSyncHwEngineEqual(
                            &unreconciledAttrs->engineArray[j],
                            &reconciledAttrs->engineArray[k])) {
                            found = true;
                            break;
                        }
                    }

                    if (false == found) {
                        /* Validation failed */
                        LWSCI_ERR_STR("Validation of LwSciSyncInternalAttrKey_EngineArray failed");
                        error = LwSciError_BadParameter;
                        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                        goto fn_exit;
                    }
                }
            }
        }
    }

fn_exit:
    return error;
}

static LwSciError ValidateTimestampFormat(
    LwSciSyncTimestampFormat format)
{
    LwSciError err = LwSciError_Success;

    switch (format) {
        case LwSciSyncTimestampFormat_Unsupported:
            /* fall through */
        case LwSciSyncTimestampFormat_8Byte:
            /* fall through */
        case LwSciSyncTimestampFormat_EmbeddedInPrimitive:
        {
            err = LwSciError_Success;
            break;
        }
        default:
        {
            err = LwSciError_BadParameter;
            break;
        }
    }

    return err;
}

LwSciError LwSciSyncValidateTimestampInfo(
    const LwSciSyncAttrValTimestampInfo* timestampInfo,
    size_t timestampInfoLen)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;

    for (i = 0U; i < timestampInfoLen; ++i) {
        LwSciSyncTimestampFormat format = timestampInfo->format;

        error = ValidateTimestampFormat(format);
        if (error != LwSciError_Success) {
            LWSCI_ERR_INT("Invalid value for "
                    "LwSciSyncTimestampFormat: ",
                    format);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        if (timestampInfo->scaling.scalingFactorDenominator == 0U) {
            LWSCI_ERR_STR("Invalid value for "
                    "scalingFactorDenominator: 0");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    return error;
}

static size_t GetValidPrimitiveTypeSize(
    const LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len)
{
    size_t i = 0U;
    size_t validLen = 0U;
    for (i = 0U; i < len; i++) {
        if (primitiveType[i] > LwSciSyncInternalAttrValPrimitiveType_LowerBound) {
            ++validLen;
        }
    }
    return validLen * sizeof(LwSciSyncInternalAttrValPrimitiveType);
}

static size_t CreateMaskFromPrimitiveType(
    const LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len)
{
    size_t i = 0U;
    size_t mask = 0U;
    for (i = 0U; i < len; i++) {
        if (LwSciSyncInternalAttrValPrimitiveType_LowerBound < primitiveType[i]) {
            mask |= (1UL << (size_t)primitiveType[i]);
        }
    }
    return mask;
}

static size_t CreateMaskOfTimestampPrimitives(
    const LwSciSyncInternalAttrValPrimitiveType* signalerPrimitiveInfo,
    size_t signalerPrimitiveInfoLen,
    const LwSciSyncAttrValTimestampInfo* timestampInfo)
{
    size_t mask = 0U;

    if (timestampInfo->format != LwSciSyncTimestampFormat_Unsupported) {
        mask |= CreateMaskFromPrimitiveType(signalerPrimitiveInfo,
                signalerPrimitiveInfoLen);
    }

    return mask;
}

static LwSciError CreateMaskOfTimestampPrimitivesMulti(
    const LwSciSyncInternalAttrValPrimitiveType* signalerPrimitiveInfo,
    size_t signalerPrimitiveInfoLen,
    const LwSciSyncAttrValTimestampInfo* timestampInfo,
    size_t signalerTimestampInfoLen,
    size_t* mask)
{
    LwSciError err = LwSciError_Success;

    size_t primitiveIndex = 0U;

    if (signalerPrimitiveInfoLen != signalerTimestampInfoLen) {
        LWSCI_ERR_STR("Lengths of LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti "
            "and LwSciSyncInternalAttrKey_SignalerPrimitiveInfo do not match");
        err = LwSciError_BadParameter;
        goto ret;
    }

    for (primitiveIndex = 0; primitiveIndex < signalerPrimitiveInfoLen; primitiveIndex++) {
        LwSciSyncInternalAttrValPrimitiveType signalerPrimitive =
            signalerPrimitiveInfo[primitiveIndex];

        /* Check if the corresponding item is valid */
        if (timestampInfo[primitiveIndex].format != LwSciSyncTimestampFormat_Unsupported) {
            *mask |= CreateMaskFromPrimitiveType(&signalerPrimitive, 1U);
        }
    }

ret:
    return err;
}

static void ReconcileWaiterContextInsensitiveFenceExports(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    size_t i = 0U;
    bool result = false;
    LwSciSyncCoreAttrs* attrs = NULL;

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        attrs = &objAttrList->coreAttrList[i].attrs;
        result = result || attrs->waiterContextInsensitiveFenceExports;
    }
    attrs = &newObjAttrList->coreAttrList->attrs;
    attrs->waiterContextInsensitiveFenceExports = result;
    /* Update key state */
    attrs->keyState[LwSciSyncCoreKeyToIndex(
            (uint32_t) LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports)] =
            LwSciSyncCoreAttrKeyState_Reconciled;
    /* Update val size */
    attrs->valSize[LwSciSyncCoreKeyToIndex(
            (uint32_t) LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports)] =
            sizeof(bool);
}

static LwSciError ReconcileEngineArray(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;

    /* We consider values from the peers separately since this
     * key has a separate view for each peer */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        const size_t index = LwSciSyncCoreKeyToIndex(
            (int32_t)LwSciSyncInternalAttrKey_EngineArray);
        const LwSciSyncCoreAttrs* unreconciledAttrs =
            &objAttrList->coreAttrList[i].attrs;
        const size_t unreconciledEngineLen = unreconciledAttrs->valSize[index] /
            sizeof(LwSciSyncHwEngine);
        LwSciSyncCoreAttrs* reconciledAttrs =
            &newObjAttrList->coreAttrList->attrs;
        size_t reconciledEngineLen = reconciledAttrs->valSize[index] /
            sizeof(LwSciSyncHwEngine);
        const size_t maxEngines = sizeof(reconciledAttrs->engineArray) /
            sizeof(reconciledAttrs->engineArray[0]);
        size_t newRemainingSize = 0U;

        /* The peer that is doing the reconciling is the one that wasn't
         * transported via IPC. This attribute list will have an empty IPC
         * route. */
        if ((LwSciSyncCoreIpcTableRouteIsEmpty(
                 &objAttrList->coreAttrList[i].ipcTable)) &&
                (0U != unreconciledEngineLen)) {

            LwSciSyncAppendHwEngineToArrayUnion(
                reconciledAttrs->engineArray, maxEngines,
                unreconciledAttrs->engineArray, unreconciledEngineLen,
                &reconciledEngineLen);

            /* recompute the used space */
            newRemainingSize = sizeof(reconciledAttrs->engineArray[0]) *
                reconciledEngineLen;
            reconciledAttrs->valSize[index] = newRemainingSize;
            reconciledAttrs->keyState[index] =
                LwSciSyncCoreAttrKeyState_Reconciled;
        }
    }

    return error;
}

static LwSciError ReconcilePrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t numSignaler = 0U;
    size_t reconciledPrimitiveMask = 0U;
    uint32_t reconciledPrimitiveCount = 0U;
    LwSciSyncInternalAttrValPrimitiveType signalerPrimitiveInfo[
            MAX_PRIMITIVE_TYPE];
    LwSciSyncInternalAttrValPrimitiveType supportedPrimitives[
            MAX_PRIMITIVE_TYPE];
    LwSciSyncInternalAttrValPrimitiveType deterministicPrimitives[
            MAX_PRIMITIVE_TYPE];
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitiveInfo =
            LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    LwSciSyncCoreAttrs* attrs = NULL;
    LwSciSyncCoreAttrKeyState keyState;
    size_t index = 0U;
    uint8_t addStatus = OP_FAIL;
    bool isCpuSignaler = false;

    LWSCI_FNENTRY("");

    /** fill with invalid entries initially */
    for (i = 0U; i < MAX_PRIMITIVE_TYPE; i++) {
        signalerPrimitiveInfo[i] =
            LwSciSyncInternalAttrValPrimitiveType_LowerBound;
        supportedPrimitives[i] =
            LwSciSyncInternalAttrValPrimitiveType_LowerBound;
        deterministicPrimitives[i] =
            LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    }

    /** set supported primitives info in reconciledPrimitiveMask */
    if (newObjAttrList->coreAttrList->attrs.needCpuAccess) {
        LwSciSyncCoreCopyCpuPrimitives(supportedPrimitives,
                sizeof(supportedPrimitives));
    } else {
        LwSciSyncCoreGetSupportedPrimitives(supportedPrimitives,
                sizeof(supportedPrimitives));
    }
    reconciledPrimitiveMask = CreateMaskFromPrimitiveType(
            supportedPrimitives,
            sizeof(supportedPrimitives) / sizeof(supportedPrimitives[0]));

    /* go over all input primitive info lists and produce a bitmask sum */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        attrs = &objAttrList->coreAttrList[i].attrs;
        if (LwSciSyncCoreAttrListHasSignalerPerm(
                &objAttrList->coreAttrList[i])) {
            u64Add(numSignaler, 1U, &numSignaler, &addStatus);
            if (OP_SUCCESS != addStatus) {
                LWSCI_ERR_STR("numSignaler value is out of range.\n");
                error = LwSciError_Overflow;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
            /* Colwert primitive info to masked variable and update final mask */
            index = LwSciSyncCoreKeyToIndex(
                    (uint32_t) LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
            reconciledPrimitiveMask &= CreateMaskFromPrimitiveType(
                    attrs->signalerPrimitiveInfo,
                    attrs->valSize[index] / sizeof(attrs->signalerPrimitiveInfo[0]));

            /* Account for the timestamp formats if necessary */
            if (newObjAttrList->coreAttrList->attrs.waiterRequireTimestamps) {
                size_t multiIndex = LwSciSyncCoreKeyToIndex(
                    (uint32_t) LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);

                /* We've already validated that exactly one of the
                 * SignalerTimestampInfo and SignalerTimestampInfoMulti keys
                 * was provided in the PrepareObjAttrList helper. In addition,
                 * PrepareObjAttrList has validated that the length of
                 * SignalerTimestampInfoMulti is equal to the length of the
                 * SignalerPrimitiveInfo key. */
                if (attrs->keyState[multiIndex] != LwSciSyncCoreAttrKeyState_Empty) {
                    /* If SignalerTimestampInfoMulti is set, use it. */
                    size_t timestampMask = 0U;
                    error = CreateMaskOfTimestampPrimitivesMulti(
                        attrs->signalerPrimitiveInfo,
                        attrs->valSize[index] / sizeof(attrs->signalerPrimitiveInfo[0]),
                        attrs->signalerTimestampInfoMulti,
                        attrs->valSize[multiIndex] / sizeof(attrs->signalerTimestampInfoMulti[0]),
                        &timestampMask);
                    if (error != LwSciError_Success) {
                        goto fn_exit;
                    }

                    reconciledPrimitiveMask &= timestampMask;
                } else {
                    /* Otherwise, this implies that SignalerTimestampInfo
                     * should be used. */
                    reconciledPrimitiveMask &= CreateMaskOfTimestampPrimitives(
                        attrs->signalerPrimitiveInfo,
                        attrs->valSize[index] / sizeof(attrs->signalerPrimitiveInfo[0]),
                        &attrs->signalerTimestampInfo);
                }
            }

            /* Save the signaler primitive info to retrieve final type */
            LwSciCommonMemcpyS(signalerPrimitiveInfo,
                    sizeof(signalerPrimitiveInfo),
                    attrs->signalerPrimitiveInfo,
                    sizeof(attrs->signalerPrimitiveInfo));
            /* Reconciled primitive count is 1 if CPU signaler else it's the
             * Signaler primitive count */
            isCpuSignaler = objAttrList->coreAttrList[i].attrs.needCpuAccess;
            reconciledPrimitiveCount = isCpuSignaler ? 1U :
                    attrs->signalerPrimitiveCount;
        }
        if (LwSciSyncCoreAttrListHasWaiterPerm(
                &objAttrList->coreAttrList[i])) {
            /* Colwert primitive info to masked variable and update final mask */
            index = LwSciSyncCoreKeyToIndex(
                    (uint32_t) LwSciSyncInternalAttrKey_WaiterPrimitiveInfo);
            reconciledPrimitiveMask &= CreateMaskFromPrimitiveType(
                    attrs->waiterPrimitiveInfo,
                    attrs->valSize[index] / sizeof(attrs->waiterPrimitiveInfo[0]));
        }
    }

    /* Account for deterministic primitives if necessary */
    if (newObjAttrList->coreAttrList->attrs.requireDeterministicFences) {
        LwSciSyncCoreGetDeterministicPrimitives(deterministicPrimitives,
                sizeof(deterministicPrimitives));
        reconciledPrimitiveMask &= CreateMaskFromPrimitiveType(
            deterministicPrimitives,
            sizeof(deterministicPrimitives) / sizeof(deterministicPrimitives[0]));
    }

    /* Multi-signaler & no signaler case is not supported */
    if (1U != numSignaler) {
        LWSCI_ERR_ULONG("Invalid number of signalers:\n", numSignaler);
        error = LwSciError_UnsupportedConfig;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /* Fail in case of no common primitive */
    if (0UL == reconciledPrimitiveMask) {
        LWSCI_ERR_STR("Unsupported configuration\n");
        error = LwSciError_UnsupportedConfig;
#if (LW_IS_SAFETY == 0)
        keyState = LwSciSyncCoreAttrKeyState_Conflict;
        attrs = &objAttrList->coreAttrList->attrs;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto set_state;
#else
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
#endif
    } else {
        keyState = LwSciSyncCoreAttrKeyState_Reconciled;
    }
    /* Final type is the first type proposed by signaler which everyone aligns with */
    for (i = 0U; (signalerPrimitiveInfo[i] !=
            LwSciSyncInternalAttrValPrimitiveType_LowerBound); i++) {
        if (0U !=(reconciledPrimitiveMask &
                (1ULL << (size_t)signalerPrimitiveInfo[i]))) {
            reconciledPrimitiveInfo = signalerPrimitiveInfo[i];
            break;
        }
    }

    /* Update the internal keys */
    attrs = &newObjAttrList->coreAttrList->attrs;
    attrs->signalerPrimitiveInfo[0] = reconciledPrimitiveInfo;
    attrs->waiterPrimitiveInfo[0] = reconciledPrimitiveInfo;
    attrs->signalerPrimitiveCount = reconciledPrimitiveCount;
    attrs->valSize[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo)] =
        sizeof(reconciledPrimitiveInfo);
    attrs->valSize[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_WaiterPrimitiveInfo)] =
        sizeof(reconciledPrimitiveInfo);
    attrs->valSize[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveCount)] =
        sizeof(reconciledPrimitiveCount);

#if (LW_IS_SAFETY == 0)
set_state:
#endif
    /* Update the internal keys status */
    attrs->keyState[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo)] = keyState;
    attrs->keyState[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_WaiterPrimitiveInfo)] = keyState;
    attrs->keyState[LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveCount)] = keyState;

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError AttrListReconcileCheckArgs(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    const LwSciSyncAttrList* newReconciledList,
    const LwSciSyncAttrList* newConflictList)
{
    LwSciError error = LwSciError_Success;

    if (NULL == newReconciledList) {
        LWSCI_ERR_STR("Invalid argument: newReconciledList: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ilwalid_args;
    }
#if (LW_IS_SAFETY == 0)
    if (NULL == newConflictList) {
        LWSCI_ERR_STR("Invalid argument: newConflictList: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ilwalid_args;
    }
#else
    (void)newConflictList;
#endif

    error = LwSciSyncCoreValidateAttrListArray(inputArray, inputCount, false);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Invalid argument: inputArray: NULL pointer\n");
        LWSCI_ERR_ULONG("Invalid argument: inputCount: \n",
            inputCount);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ilwalid_args;
    }

ilwalid_args:

    return error;
}

static LwSciError CheckSingleEngineArray(
    LwSciSyncCoreAttrListObj* objAttrList,
    size_t slotIndex,
    bool* containsEngineArray)
{
    LwSciError error = LwSciError_Success;

    const LwSciSyncCoreAttrs* unreconciledAttrs =
        &objAttrList->coreAttrList[slotIndex].attrs;
    size_t index = LwSciSyncCoreKeyToIndex(
            (uint32_t)LwSciSyncInternalAttrKey_EngineArray);
    size_t numEngines = unreconciledAttrs->valSize[index] /
        sizeof(unreconciledAttrs->engineArray[0]);

    if (LwSciSyncCoreAttrKeyState_Empty != unreconciledAttrs->keyState[index]) {
        for (size_t i = 0U; i < numEngines; i++) {
            LwSciSyncHwEngName eng = LwSciSyncHwEngName_LowerBound;
            error = LwSciSyncHwEngGetNameFromId(
                unreconciledAttrs->engineArray[i].rmModuleID,
                &eng);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }

            if (LwSciSyncHwEngName_PCIe == eng) {
                if (*containsEngineArray) {
                    LwSciSyncCoreAttrs* attrs = &objAttrList->coreAttrList->attrs;
                    attrs->keyState[LwSciSyncCoreKeyToIndex(
                            (int32_t)LwSciSyncInternalAttrKey_EngineArray)] =
                            LwSciSyncCoreAttrKeyState_Conflict;

                    LWSCI_ERR_STR("C2C only supports a single LwSciSyncAttrList requesting engine access");
                    error = LwSciError_ReconciliationFailed;
                    goto fn_exit;
                }
                if (1U != numEngines) {
                    LwSciSyncCoreAttrs* attrs = &objAttrList->coreAttrList->attrs;
                    attrs->keyState[LwSciSyncCoreKeyToIndex(
                            (int32_t)LwSciSyncInternalAttrKey_EngineArray)] =
                            LwSciSyncCoreAttrKeyState_Conflict;

                    LWSCI_ERR_STR("C2C only supports a single engine in the LwSciSyncAttrList requesting engine access");
                    error = LwSciError_ReconciliationFailed;
                    goto fn_exit;
                }

                *containsEngineArray = true;
                break;
            }
        }
    }

fn_exit:
    return error;
}

static inline LwSciSyncInternalAttrValPrimitiveType getExpectedPrimitives(
    void)
{
#if (defined(__x86_64__))
    return LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
#else
    return LwSciSyncInternalAttrValPrimitiveType_Syncpoint;
#endif
}

static LwSciError ValidateC2LWseCases(
    LwSciSyncCoreAttrListObj* objAttrList)
{
    LwSciError error = LwSciError_Success;

    if (TraveledViaC2C(objAttrList)) {
        /* This is potentially a C2C case. */
        bool containsEngineArray = false;

        /* only 2 unreconciled LwSciSyncAttrList(s) are supported.
         * We assume that a slot is 1:1 with an LwSciSyncAttrList that existed
         * at some point (even if the peer appended lists together). */
        if (2U != objAttrList->numCoreAttrList) {
            LWSCI_ERR_STR("C2C use cases only involve 2 LwSciSyncAttrLists");
            error = LwSciError_ReconciliationFailed;
            goto fn_exit;
        }

        for (size_t i = 0U; i < objAttrList->numCoreAttrList; i++) {
            const LwSciSyncCoreAttrs* unreconciledAttrs =
                &objAttrList->coreAttrList[i].attrs;
            LwSciSyncCoreIpcTable* ipcTable = &objAttrList->coreAttrList[i].ipcTable;
            LwSciSyncInternalAttrValPrimitiveType expectedPrimitives[] = {
                getExpectedPrimitives()
            };
            const size_t initialMask = CreateMaskFromPrimitiveType(
                expectedPrimitives, sizeof(expectedPrimitives) / sizeof(expectedPrimitives[0]));
            size_t mask = initialMask;

            if (LwSciSyncCoreAttrListHasSignalerPerm(
                    &objAttrList->coreAttrList[i])) {
                /* SignalerPrimitiveInfo must contain syncpoint on CheetAh
                 * or semaphore on x86 */
                size_t index = LwSciSyncCoreKeyToIndex(
                        (uint32_t) LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
                size_t numPrimitives = unreconciledAttrs->valSize[index] /
                    sizeof(unreconciledAttrs->signalerPrimitiveInfo[0]);

                size_t signalerMask = CreateMaskFromPrimitiveType(
                    unreconciledAttrs->signalerPrimitiveInfo, numPrimitives);

                mask &= signalerMask;
                if (mask != initialMask) {
                    LWSCI_ERR_STR("SignalerPrimitiveInfo should contain syncpoint on CheetAh"
                            " semaphore on x86");
                    error = LwSciError_ReconciliationFailed;
                    goto fn_exit;
                }

                /* The signaler's IPC route can contain 1 entry, which is a
                 * C2C channel */
                if (!(LwSciSyncCoreIpcTableHasC2C(ipcTable) &&
                            (ipcTable->ipcRouteEntries == 1U))) {
                    LWSCI_ERR_STR("Signaler traveled via an unexpected IPC route");
                    error = LwSciError_ReconciliationFailed;
                    goto fn_exit;
                }
            }
            if (LwSciSyncCoreAttrListHasWaiterPerm(
                    &objAttrList->coreAttrList[i])) {
                size_t index = LwSciSyncCoreKeyToIndex(
                        (uint32_t) LwSciSyncInternalAttrKey_WaiterPrimitiveInfo);
                size_t numPrimitives = unreconciledAttrs->valSize[index] /
                    sizeof(unreconciledAttrs->waiterPrimitiveInfo[0]);

                size_t waiterMask = CreateMaskFromPrimitiveType(
                    unreconciledAttrs->waiterPrimitiveInfo, numPrimitives);

                mask &= waiterMask;
                if (mask != initialMask) {
                    LWSCI_ERR_STR("C2C WaiterPrimitiveInfo should contain syncpoint on CheetAh"
                            " semaphore on x86");
                    error = LwSciError_ReconciliationFailed;
                    goto fn_exit;
                }

                /* The waiter must not travel over IPC */
                if (false == LwSciSyncCoreIpcTableRouteIsEmpty(ipcTable)) {
                    LWSCI_ERR_STR("C2C Waiter travelled over IPC");
                    error = LwSciError_ReconciliationFailed;
                    goto fn_exit;
                }
            }
            /* Ensure only 1 LwSciSyncAttrList contains the
             * LwSciSyncHwEngName_PCIe engine in
             * LwSciSyncInternalAttrKey_EngineArray */
            error = CheckSingleEngineArray(objAttrList, i, &containsEngineArray);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }

        if (false == containsEngineArray) {
            LWSCI_ERR_STR("LwSciSyncHwEngName_PCIe missing in LwSciSyncInternalAttrKey_EngineArray");
            error = LwSciError_ReconciliationFailed;
            goto fn_exit;
        }
    } else {
        /* If no unreconciled LwSciSyncAttrList(s) travelled over a C2C IPC
         * boundary, then the only case we support is when the signaler is the
         * one performing the reconciliation (was not transferred over IPC). */
        for (size_t i = 0U; i < objAttrList->numCoreAttrList; i++) {
            /* Find the signaler attribute list */
            if (LwSciSyncCoreAttrListHasSignalerPerm(
                    &objAttrList->coreAttrList[i])) {
                /* We expect that the signaler is doing the reconciliation.
                 * Lwrrently, we only support cases where the signaler's
                 * LwSciSyncAttrList wasn't transferred over IPC. */
                if (!LwSciSyncCoreIpcTableRouteIsEmpty(
                        &objAttrList->coreAttrList[i].ipcTable)) {
                    LWSCI_ERR_STR("LwSciSyncAttrList with signaling permissions traveled over IPC");
                    error = LwSciError_ReconciliationFailed;
                    goto fn_exit;
                }
            }
        }
    }

fn_exit:
    return error;
}

static LwSciError PrepareObjAttrList(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    LwSciSyncAttrList* newUnreconciledAttrList,
    LwSciSyncCoreAttrListObj** objAttrList,
    LwSciSyncAttrList* multiSlotAttrList,
    LwSciSyncCoreAttrListObj** newObjAttrList)
{
    LwSciError error = LwSciError_Success;
    error = LwSciSyncCoreAttrListAppendUnreconciledWithLocks(inputArray,
            inputCount, false, newUnreconciledAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreAttrListGetObjFromRef(*newUnreconciledAttrList,
            objAttrList);

    error = LwSciSyncCoreAttrListCreateMultiSlot((*objAttrList)->module, 1U,
            true, multiSlotAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreAttrListGetObjFromRef(*multiSlotAttrList, newObjAttrList);

    /* Sanity check for attr list values */
    error = ValidatePublicKeyValues(*objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = ValidateInternalKeyValues(*objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* Update Signaler Timestamp Info */
    LwSciSyncCoreFillSignalerTimestampInfo(*objAttrList);
    /* Update local CPU primitive info */
    LwSciSyncCoreFillCpuPrimitiveInfo(*objAttrList, false);

    error = ValidateC2LWseCases(*objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    return error;
}

static LwSciError AttrListValidateReconciledCheckArgs(
    LwSciSyncAttrList reconciledAttrList,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    const bool* isReconciledListValid
)
{
    LwSciError error = LwSciError_Success;
    bool isReconciled = false;

    error = LwSciSyncCoreValidateAttrListArray(inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount, true);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == isReconciledListValid) {
        LWSCI_ERR_STR("Invalid argument: isReconciledListValid: NULL pointer");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncAttrListIsReconciled(reconciledAttrList, &isReconciled);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Invalid argument: reconciledAttrList");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (false == isReconciled) {
        LWSCI_ERR_STR("Input attr list is not reconciled");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

static LwSciError CopyIpcPermTable(
    const LwSciSyncCoreAttrListObj* objAttrList,
    LwSciSyncCoreAttrList* newCoreAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0;
    size_t permIndex = 0U;
    size_t branchNum = 0U;
    uint8_t addStatus = OP_FAIL;

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        if (LwSciSyncCoreIpcTableRouteIsEmpty(
                &objAttrList->coreAttrList[i].ipcTable) == false) {
            sizeAdd(branchNum, 1U, &branchNum, &addStatus);
            if (OP_SUCCESS != addStatus) {
                LWSCI_ERR_STR("Arithmetic overflow\n");
                LwSciCommonPanic();
            }
        }
    }

    error = LwSciSyncCoreIpcTableTreeAlloc(
            &newCoreAttrList->ipcTable, branchNum);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }


    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        /** Copy Ipc route entry from core attr list to Ipc Perm table */
        if (LwSciSyncCoreIpcTableRouteIsEmpty(
                &objAttrList->coreAttrList[i].ipcTable) == false) {
            error = LwSciSyncCoreIpcTableAddBranch(
                &newCoreAttrList->ipcTable, permIndex,
                &objAttrList->coreAttrList[i].ipcTable,
                objAttrList->coreAttrList[i].attrs.needCpuAccess,
                objAttrList->coreAttrList[i].attrs.waiterRequireTimestamps,
                objAttrList->coreAttrList[i].attrs.requiredPerm,
                objAttrList->coreAttrList[i].attrs.engineArray,
                objAttrList->coreAttrList[i].attrs.valSize[
                    LwSciSyncCoreKeyToIndex(LwSciSyncInternalAttrKey_EngineArray)] /
                    sizeof(objAttrList->coreAttrList[i].attrs.engineArray[0]));
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
            sizeAdd(permIndex, 1U, &permIndex, &addStatus);
            if (OP_SUCCESS != addStatus) {
                LWSCI_ERR_STR("Arithmetic overflow\n");
                LwSciCommonPanic();
            }
        }
    }

fn_exit:
    return error;
}

static LwSciError AttrListCheckTimestampRequired(
    const LwSciSyncCoreAttrListObj* objAttrList,
    const LwSciSyncCoreAttrListObj* newObjAttrList,
    bool* timestampRequired
)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreAttrs* attrs = NULL;
    LwSciSyncCoreAttrKeyState keyState;
    LwSciSyncCoreAttrKeyState keyStateMulti;

    bool timestampProvided = false;
    bool hasTimestampInfo = false;
    bool hasTimestampInfoMulti = false;

    const size_t multiIndex = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
    const size_t index = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo);

    size_t i = 0U;

    *timestampRequired = false;

    if (NULL == objAttrList) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* Check if timestamp support was provided */
    attrs = &newObjAttrList->coreAttrList->attrs;
    keyState = attrs->keyState[index];
    keyStateMulti = attrs->keyState[multiIndex];

    hasTimestampInfo = (keyState != LwSciSyncCoreAttrKeyState_Empty);
    hasTimestampInfoMulti = (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty);
    timestampProvided = (hasTimestampInfo || hasTimestampInfoMulti);

    /** Check timestamp requirement. */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        /* Check if timestamp is required. */
        attrs = &objAttrList->coreAttrList[i].attrs;
        *timestampRequired = *timestampRequired ||
                attrs->waiterRequireTimestamps;
    }

    if ((true == *timestampRequired) && (false == timestampProvided)) {
        LWSCI_ERR_STR("Timestamp support not provided.");
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

    /* Verify WaiterRequireTimestamps. It is a failure if the unreconciled
     * LwSciSyncAttrList requested for timestamps, but is set to false
     * in the reconciled list. */
    if (*timestampRequired &&
        !newObjAttrList->coreAttrList->attrs.waiterRequireTimestamps) {
        LWSCI_ERR_STR("LwSciSyncAttrKey_WaiterRequireTimestamps was not satisfied");
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

fn_exit:
    return error;
}

static LwSciError EnsureSameModule(
    const LwSciSyncCoreAttrListObj* unreconciledObjAttrList,
    const LwSciSyncCoreAttrListObj* reconciledObjAttrList
)
{
    LwSciError error = LwSciError_Success;
    bool isModuleDup = false;

    LwSciSyncCoreModuleIsDup(reconciledObjAttrList->module,
            unreconciledObjAttrList->module, &isModuleDup);

    if (false == isModuleDup) {
        LWSCI_ERR_STR("Incompatible modules\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    return error;
}

void LwSciSyncCoreFillCpuPrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    bool hasC2C)
{
    const LwSciSyncCoreAttrs* attrs = NULL;
    size_t i = 0U;
    size_t keyIdx = 0U;

    LWSCI_FNENTRY("");

    if (NULL == objAttrList) {
        LWSCI_ERR_STR("Invalid objAttrList: NULL pointer\n");
        LwSciCommonPanic();
    }

    /** Update the primitive info for CPU entity */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        LwSciSyncCoreAttrList* coreAttrList = &objAttrList->coreAttrList[i];

        const bool isCpuSignaler = (coreAttrList->attrs.needCpuAccess &&
                LwSciSyncCoreAttrListHasSignalerPerm(coreAttrList));
        const bool isCpuWaiter = (coreAttrList->attrs.needCpuAccess &&
                LwSciSyncCoreAttrListHasWaiterPerm(coreAttrList));

        attrs = &coreAttrList->attrs;
        keyIdx = LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
        if (isCpuSignaler && (0U == attrs->valSize[keyIdx])) {
            LwSciSyncInternalAttrValPrimitiveType* primitiveType =
                coreAttrList->attrs.signalerPrimitiveInfo;
            const size_t arrLen =
                sizeof(coreAttrList->attrs.signalerPrimitiveInfo) /
                sizeof(coreAttrList->attrs.signalerPrimitiveInfo[0]);

            if (hasC2C) {
                LwSciSyncCoreCopyC2cCpuPrimitives(primitiveType,
                        sizeof(coreAttrList->attrs.signalerPrimitiveInfo));
            } else {
                LwSciSyncCoreCopyCpuPrimitives(primitiveType,
                        sizeof(coreAttrList->attrs.signalerPrimitiveInfo));
            }

            coreAttrList->attrs.valSize[keyIdx] = GetValidPrimitiveTypeSize(
                    primitiveType, arrLen);
            coreAttrList->attrs.signalerPrimitiveCount = 1U;
            keyIdx = LwSciSyncCoreKeyToIndex(
                    (uint32_t) LwSciSyncInternalAttrKey_SignalerPrimitiveCount);
            coreAttrList->attrs.valSize[keyIdx] =
                    LwSciSyncCoreKeyInfo[keyIdx].elemSize;
        }
        keyIdx = LwSciSyncCoreKeyToIndex(
                (uint32_t) LwSciSyncInternalAttrKey_WaiterPrimitiveInfo);
        if (isCpuWaiter && (0U == attrs->valSize[keyIdx])) {
            LwSciSyncInternalAttrValPrimitiveType* primitiveType =
                coreAttrList->attrs.waiterPrimitiveInfo;
            const size_t arrLen =
                sizeof(coreAttrList->attrs.waiterPrimitiveInfo) /
                sizeof(coreAttrList->attrs.waiterPrimitiveInfo[0]);

            if (hasC2C) {
                LwSciSyncCoreCopyC2cCpuPrimitives(primitiveType,
                        sizeof(coreAttrList->attrs.waiterPrimitiveInfo));
            } else {
                LwSciSyncCoreCopyCpuPrimitives(primitiveType,
                        sizeof(coreAttrList->attrs.waiterPrimitiveInfo));
            }

            coreAttrList->attrs.valSize[keyIdx] = GetValidPrimitiveTypeSize(
                    primitiveType, arrLen);
        }
    }

    LWSCI_FNEXIT("");
}

LwSciError LwSciSyncGetSignalerPrimitiveInfoIndex(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitive,
    size_t* primitiveIndex)
{
    LwSciError err = LwSciError_Success;

    size_t i = 0U;

    const size_t index = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
    size_t len = 0U;
    const LwSciSyncInternalAttrValPrimitiveType* signalerPrimitiveInfo = NULL;

    if ((coreAttrList == NULL) || (primitiveIndex == NULL)) {
        LwSciCommonPanic();
    }

    len = (coreAttrList->attrs.valSize[index] /
            sizeof(coreAttrList->attrs.signalerPrimitiveInfo[0]));
    signalerPrimitiveInfo = coreAttrList->attrs.signalerPrimitiveInfo;

    for (i = 0U; i < len; i++) {
        if (signalerPrimitiveInfo[i] == reconciledPrimitive) {
            *primitiveIndex = i;
            goto ret;
        }
    }

    err = LwSciError_BadParameter;
ret:
    return err;
}
