/*
 * Copyright (c) 2019-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <string.h>
#include <errno.h>

#include "lwscibuf_attr_mgmt.h"
#include "lwscibuf_attr_reconcile.h"
#include "lwscibuf_fsm.h"
#include "lwscibuf_transport_priv.h"
#include "lwscibuf_utils.h"
#include "lwscicommon_transportutils.h"
/* TODO: This is a new dependency on object core unit. */
#include "lwscibuf_obj_mgmt.h"

/*********************************************
 *  FSM forward declarations
 *********************************************/
/* FSM output function forward declarations */
static LwSciError OutputFnError(
    void* context,
    const void* data);

static LwSciError OutputFnSlotCount(
    void* context,
    const void* data);

static LwSciError OutputFnReconciliationFlag(
    void* context,
    const void* data);

static LwSciError OutputFnSlotIndex(
    void* context,
    const void* data);

static LwSciError OutputFnAttributeKeys(
    void* context,
    const void* data);

static LwSciError OutputFnObjDesc(
    void* context,
    const void* data);

/* FSM Guard function forward declarations */
static bool SlotCountGuard(
    void* context,
    const void* data);

static bool ReconciliationFlagGuard(
    void* context,
    const void* data);

static bool SlotIndexGuard(
    void* context,
    const void* data);

static bool AttributeKeysGuard(
    void* context,
    const void* data);

static bool NewerAttributeKeysGuard(
    void* context,
    const void* data);

static bool ObjDescGuard(
    void* context,
    const void* data);

/*********************************************
 *  FSM state declarations
 *********************************************/

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_9), "LwSciBuf-ADV-MISRAC2012-011")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 20_7), "LwSciBuf-REQ-MISRAC2012-005")
/* Error state is common across all FSMs */
FSM_DEFINE_STATE(stateError, OutputFnError);

/* When only importing Attribute List descriptor */
FSM_DEFINE_STATE(attrFsmStateStart, NULL);
FSM_DEFINE_STATE(attrFsmStateSlotCount, OutputFnSlotCount);
FSM_DEFINE_STATE(attrFsmStateReconciliationFlag, OutputFnReconciliationFlag);
FSM_DEFINE_STATE(attrFsmStateSlotIndex, OutputFnSlotIndex);
/* We introduce a no-op state here to handle the case when importing a
 * reconciled LwSciBufAttrList which contains newer Attribute Keys that this
 * version is not aware of. */
FSM_DEFINE_STATE(attrFsmStateNewerAttrKeys, NULL);
FSM_DEFINE_STATE(attrFsmStateAttrKeys, OutputFnAttributeKeys);

/* When importing a LwSciBufObj descriptor */
FSM_DEFINE_STATE(objFsmStateStart, NULL);
FSM_DEFINE_STATE(objFsmStateObjDesc, OutputFnObjDesc);

/* When importing a combined descriptor */
FSM_DEFINE_STATE(combinedFsmStateStart, NULL);
FSM_DEFINE_STATE(combinedFsmStateSlotCount, OutputFnSlotCount);
FSM_DEFINE_STATE(combinedFsmStateReconciliationFlag,
    OutputFnReconciliationFlag);
FSM_DEFINE_STATE(combinedFsmStateSlotIndex, OutputFnSlotIndex);
/* We introduce a no-op state here to handle the case when importing a
 * reconciled LwSciBufAttrList which contains newer Attribute Keys that this
 * version is not aware of. */
FSM_DEFINE_STATE(combinedFsmStateNewerAttrKeys, NULL);
FSM_DEFINE_STATE(combinedFsmStateAttrKeys, OutputFnAttributeKeys);
FSM_DEFINE_STATE(combinedFsmStateObjDesc, OutputFnObjDesc);

LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 20_7))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_9))

/*********************************************
 *  FSM transition tables
 *********************************************/
/* LwSciBufAttrList FSM */
FSM_DEFINE_STATE_TRANSITION_TABLE(attrFsmStateStart) {
    FSM_ADD_TRANSITION_STATE(attrFsmStateSlotCount, SlotCountGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(attrFsmStateSlotCount) {
    FSM_ADD_TRANSITION_STATE(attrFsmStateReconciliationFlag,
        ReconciliationFlagGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(attrFsmStateReconciliationFlag) {
    FSM_ADD_TRANSITION_STATE(attrFsmStateSlotIndex, SlotIndexGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(attrFsmStateSlotIndex) {
    FSM_ADD_TRANSITION_STATE(attrFsmStateNewerAttrKeys,
        NewerAttributeKeysGuard);
    FSM_ADD_TRANSITION_STATE(attrFsmStateAttrKeys, AttributeKeysGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(attrFsmStateNewerAttrKeys) {
    FSM_ADD_TRANSITION_STATE(attrFsmStateSlotIndex, SlotIndexGuard);
    FSM_ADD_TRANSITION_STATE(attrFsmStateNewerAttrKeys,
        NewerAttributeKeysGuard);
    FSM_ADD_TRANSITION_STATE(attrFsmStateAttrKeys, AttributeKeysGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(attrFsmStateAttrKeys) {
    FSM_ADD_TRANSITION_STATE(attrFsmStateSlotIndex, SlotIndexGuard);
    FSM_ADD_TRANSITION_STATE(attrFsmStateNewerAttrKeys,
        NewerAttributeKeysGuard);
    FSM_ADD_TRANSITION_STATE(attrFsmStateAttrKeys, AttributeKeysGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

/* LwSciBufObj FSM */
FSM_DEFINE_STATE_TRANSITION_TABLE(objFsmStateStart) {
    FSM_ADD_TRANSITION_STATE(objFsmStateObjDesc, ObjDescGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
FSM_DEFINE_STATE_TRANSITION_TABLE(objFsmStateObjDesc) {
    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

/* LwSciBufAttrList + LwSciBufObj FSM */
FSM_DEFINE_STATE_TRANSITION_TABLE(combinedFsmStateStart) {
    FSM_ADD_TRANSITION_STATE(combinedFsmStateSlotCount, SlotCountGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(combinedFsmStateSlotCount) {
    FSM_ADD_TRANSITION_STATE(combinedFsmStateReconciliationFlag,
        ReconciliationFlagGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(combinedFsmStateReconciliationFlag) {
    FSM_ADD_TRANSITION_STATE(combinedFsmStateSlotIndex, SlotIndexGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(combinedFsmStateSlotIndex) {
    FSM_ADD_TRANSITION_STATE(combinedFsmStateNewerAttrKeys,
        NewerAttributeKeysGuard);
    FSM_ADD_TRANSITION_STATE(combinedFsmStateAttrKeys, AttributeKeysGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(combinedFsmStateNewerAttrKeys) {
    FSM_ADD_TRANSITION_STATE(combinedFsmStateObjDesc, ObjDescGuard);
    FSM_ADD_TRANSITION_STATE(combinedFsmStateNewerAttrKeys,
        NewerAttributeKeysGuard);
    FSM_ADD_TRANSITION_STATE(combinedFsmStateAttrKeys, AttributeKeysGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

FSM_DEFINE_STATE_TRANSITION_TABLE(combinedFsmStateAttrKeys) {
    FSM_ADD_TRANSITION_STATE(combinedFsmStateObjDesc, ObjDescGuard);
    FSM_ADD_TRANSITION_STATE(combinedFsmStateNewerAttrKeys,
        NewerAttributeKeysGuard);
    FSM_ADD_TRANSITION_STATE(combinedFsmStateAttrKeys, AttributeKeysGuard);

    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
FSM_DEFINE_STATE_TRANSITION_TABLE(combinedFsmStateObjDesc) {
    FSM_ADD_DEFAULT_TRANSITION(stateError);
    FSM_DEFINE_STATE_TRANSITION_TABLE_END;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
FSM_DEFINE_STATE_TRANSITION_TABLE(stateError) {
    /* This is a terminal state. There are no transitions in or out. */
    (void)fsm;
    (void)data;
    (void)transitioned;
}

static LwSciError LwSciBufGetIpcRouteExportDesc(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm exportAPIperms,
    void** exportDesc,
    uint64_t* len);

static LwSciError LwSciBufAttrListIpcRouteImport(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    const void* exportDesc,
    uint64_t len);

static LwSciError LwSciBufGetIpcTableExportDesc(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm exportAPIperms,
    void** exportDesc,
    uint64_t* len);

static LwSciError LwSciBufAttrListIpcTableImport(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    const void* exportDesc,
    uint64_t len);

static LwSciError LwSciBufValidateImportAttrKey(
    LwSciBufAttrList* attrList,
    const uint64_t* slotIdx,
    uint32_t key,
    bool importingReconciledAttr,
    bool *skippedValidation);

static LwSciBufAttrValAccessPerm LwSciBufComputeImportObjPerms(
    LwSciBufAttrValAccessPerm importAPIPerm,
    LwSciBufAttrValAccessPerm exportObjPerm);

/*
 * As of now, special handling of export/import is required only for
 * general keys and not for other datatypes.
 * Hence, defining the descriptor table only for general keys.
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 9_3), "LwSciBuf-REQ-MISRAC2012-002")
static const LwSciBufAttrKeyTransportDesc
    genAttrKeyTransportDescTable[LwSciBufAttrKeyType_MaxValid]
                                [LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE] =
{
    [LwSciBufAttrKeyType_Private] = {
        [LW_SCI_BUF_DECODE_ATTRKEY(LwSciBufPrivateAttrKey_SciIpcRoute)] = {
            LwSciBufGetIpcRouteExportDesc,
            LwSciBufAttrListIpcRouteImport,
        },
        [LW_SCI_BUF_DECODE_ATTRKEY(LwSciBufPrivateAttrKey_IPCTable)] = {
            LwSciBufGetIpcTableExportDesc,
            LwSciBufAttrListIpcTableImport,
        },
        { NULL, NULL },
    },
    {{ NULL, NULL }},
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 9_3))

static const LwSciBufTransportKeyDesc
    LwSciBufTransportKeysDescTable[LW_SCI_BUF_NUM_TRANSPORT_KEYS] = {
    [LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotCount)] =
                        {
                            sizeof(uint64_t),
                        },

    [LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListReconciledFlag)] =
                        {
                            sizeof(uint8_t),
                        },

    [LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotIndex)] =
                        {
                            sizeof(uint64_t),
                        },

    [LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_ObjDesc)] =
                        {
                            sizeof(LwSciBufObjExportDescPriv),
                        },
};

/* Output function for the SlotCount State */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError OutputFnSlotCount(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* transportContext =
        (const LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LwSciBufModule module = transportContext->module;
    LwSciBufAttrList* outputAttrList = transportContext->outputAttrList;

    const void* value = pair.value;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    uint64_t slotCount = *(const uint64_t *)value;

    LWSCI_FNENTRY("");

    /* Allocate LwSciBufAttrList */
    err = LwSciBufAttrListCreateMultiSlot(module, slotCount, outputAttrList,
            false);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

/* Output function for the Slot Index State */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError OutputFnSlotIndex(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    /* Import the Slot Index */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LwSciBufTransportFsmContext* transportContext =
        (LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;

    const size_t* value = (const size_t*)pair.value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LWSCI_FNENTRY("");

    transportContext->slotIndex = *value;

    LWSCI_FNEXIT("");
    return err;
}

static LwSciBufAttrListState LwSciBufDeserializeReconcileState(
    uint8_t value)
{
    LwSciBufAttrListState state = LwSciBufAttrListState_UpperBound;
    if (1U == value) {
        state = LwSciBufAttrListState_Reconciled;
    } else {
        state = LwSciBufAttrListState_Appended;
    }
    return state;
}

/* Output function for the Reconciliation Flag State */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError OutputFnReconciliationFlag(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    /* Import the Reconciliation State flag */
    const LwSciBufTransportFsmContext* transportContext =
        (LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;

    const LwSciBufAttrList* outputAttrList = transportContext->outputAttrList;
    uint8_t isReconciled = *(const uint8_t *)pair.value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LWSCI_FNENTRY("");

    /* When multiple unreconciled LwSciBufAttrLists are imported, they are
     * imported as appended LwSciBufAttrLists. If single unreconciled
     * LwSciBufAttrList is imported, we can generalise that, single unreconciled
     * LwSciBufAttrList is appended with itself and thus for any number of
     * unreconciled LwSciBufAttrList(s) imported, we set the status to
     * LwSciBufAttrListState_Appended
     */
    err = LwSciBufAttrListSetState(*outputAttrList,
                    LwSciBufDeserializeReconcileState(isReconciled));
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Unable to Mark Reconcile flag while importing.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Outputs - isReconciled: %s \n",
        (isReconciled == 1U)?"True":"False");
ret:
    LWSCI_FNEXIT("");
    return err;
}

/* Output function for the Attribute Keys State */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError OutputFnAttributeKeys(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    /* Import the Attribute Key */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* transportContext =
        (LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    uint32_t key = pair.key;
    const void* value = pair.value;
    size_t length = pair.length;

    uint32_t decKey = 0U;
    uint32_t tmpDataType = 0U;
    uint32_t tmpKeyType = 0U;

    LwSciBufType dataType = LwSciBufType_MaxValid;
    LwSciBufAttrKeyType decKeyType = LwSciBufAttrKeyType_MaxValid;

    LwSciBufAttrKeyImportCb importFn;

    const LwSciBufAttrList* attrList = transportContext->outputAttrList;
    LwSciIpcEndpoint ipcEndpoint = transportContext->ipcEndpoint;
    size_t slotIdx = transportContext->slotIndex;
    bool importingReconciledAttr = transportContext->importingReconciledAttr;

    LWSCI_FNENTRY("");

    /* Check whether the given key is within bounds*/
    err = LwSciBufAttrKeyDecode(key, &decKey, &tmpDataType, &tmpKeyType);
    if (LwSciError_Success != err) {
        /* We just validated the key in LwSciBufValidateImportAttrKey. */
        LwSciCommonPanic();
    }

    LwSciCommonMemcpyS(&decKeyType, sizeof(decKeyType),
            &tmpKeyType, sizeof(tmpKeyType));
    LwSciCommonMemcpyS(&dataType, sizeof(dataType),
            &tmpDataType, sizeof(tmpDataType));

    importFn = (dataType == LwSciBufType_General)?
       genAttrKeyTransportDescTable[(uint32_t)decKeyType][decKey].importCallback:
       NULL;
    if (NULL != importFn) {
        err = importFn(*attrList, slotIdx, ipcEndpoint, value,
                           length);
    } else {
        LwSciBufAttrKeyType localKeyType;
        uint32_t KeyType = LW_SCI_BUF_DECODE_KEYTYPE(key);
        /* LwSciBuf controls values of any attributes set in reconciled
         * LwSciBufAttrList.
         * LwSciBuf also controls the values of private attributes set in
         * unreconciled LwSciBufAttrLists. Thus, set the override flag for
         * setting the attributes to true when any of the above mentioned
         * conditions are encountered, otherwise set it to false.
         */
        bool overrideAttrSet = (importingReconciledAttr ? true : false);
        bool skipValidation = false;
        bool isSocBoundary = false;

        err = LwSciBufIsSocBoundary(ipcEndpoint, &isSocBoundary);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Could not determine SoC boundary from ipcEndpoint.");
            goto ret;
        }

        /* If we encounter any of the input only OR input/output GPU keys and
         * ipcEndpoint has crossed SoC boundary and we are importing
         * unreconciled list then skip the validation. This is because when
         * importing unreconciled list the importing peer cannot determine if
         * GPU IDs in these keys are valid or not since it is on different SoC.
         * Note that validation is _not_ skipped for reconciled list because
         * when importing reconciled list the exporting peer would callwlate
         * the values of these keys for the importing peer only and thus
         * importing peer can validate them.
         * Also note that, we are skipping
         * LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency key from the condition
         * below because it is output only key so it wont be present in the
         * unreconciled list.
         */
        skipValidation = (((LwSciBufGeneralAttrKey_GpuId == key) ||
                            (LwSciBufGeneralAttrKey_VidMem_GpuId == key) ||
                            (LwSciBufGeneralAttrKey_EnableGpuCache == key) ||
                            (LwSciBufGeneralAttrKey_EnableGpuCompression == key))
                            && (true == isSocBoundary) &&
                            (false == importingReconciledAttr));

        LwSciCommonMemcpyS(&localKeyType, sizeof(localKeyType),
                                   &KeyType, sizeof(KeyType));
        if (LwSciBufAttrKeyType_Public == localKeyType) {
            LwSciBufAttrKeyValuePair keyValPair = {0};

            LwSciCommonMemcpyS(&keyValPair.key, sizeof(keyValPair.key),
                &key, sizeof(key));
            keyValPair.len = length;
            keyValPair.value = value;

           err = LwSciBufAttrListCommonSetAttrs(*attrList, slotIdx,
                    &keyValPair, 1, localKeyType, overrideAttrSet,
                    skipValidation);
        } else if (
            (LwSciBufAttrKeyType_Internal == localKeyType) ||
            (LwSciBufAttrKeyType_UMDPrivate == localKeyType)) {
            LwSciBufInternalAttrKeyValuePair keyValPair ={0};

            LwSciCommonMemcpyS(&keyValPair.key, sizeof(keyValPair.key),
                &key, sizeof(key));
            keyValPair.len = length;
            keyValPair.value = value;

            err = LwSciBufAttrListCommonSetAttrs(*attrList, slotIdx,
                        &keyValPair, 1, localKeyType, overrideAttrSet,
                        skipValidation);
        } else if (
            LwSciBufAttrKeyType_Private == localKeyType) {
            LwSciBufPrivateAttrKeyValuePair keyValPair = {0};
            /* LwSciBuf controls values of any attributes set in reconciled
             * LwSciBufAttrList.
             * LwSciBuf also controls the values of private attributes set in
             * unreconciled LwSciBufAttrLists. Thus, set the override flag for
             * setting the attributes to true when any of the above mentioned
             * conditions are encountered, otherwise set it to false.
             */
            overrideAttrSet = true;

            LwSciCommonMemcpyS(&keyValPair.key, sizeof(keyValPair.key),
                &key, sizeof(key));

            keyValPair.len = length;
            keyValPair.value = value;

            err = LwSciBufAttrListCommonSetAttrs(*attrList, slotIdx,
                        &keyValPair, 1, localKeyType, overrideAttrSet,
                        skipValidation);
        } else {
            /* This should never happen */
            err = LwSciError_BadParameter;
        }
    }

    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Unable to set key value for attrList.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError ValidateReconciledPerm(
    LwSciBufAttrValAccessPerm perm)
{
    LwSciError error = LwSciError_Success;

    switch (perm) {
        /* The exporter should only export Read{only,Write} permissions. Auto
         * is not a valid value. */
        case LwSciBufAccessPerm_Readonly:
            /* fall through */
        case LwSciBufAccessPerm_ReadWrite:
        {
            error = LwSciError_Success;
            break;
        }
        default:
        {
            error = LwSciError_BadParameter;
            break;
        }
    }

    return error;
}

static LwSciError DeserializeLwSciBufObjExportDescPriv(
    LwSciBufObjExportDescPriv* objDesc,
    const void* value)
{
    LwSciError error = LwSciError_Success;

    /* Deserialize LwSciBufObjExportDescPriv */
    LwSciCommonMemcpyS(objDesc, sizeof(*objDesc), value, sizeof(*objDesc));

    error = ValidateReconciledPerm(objDesc->perms);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fn_exit;
    }

fn_exit:
    return error;
}

/* Output function for the Object Descriptor State */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError OutputFnObjDesc(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* transportContext =
        (LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    const void* value = pair.value;
    bool isSocBoundary = false;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    bool isRemoteObject = false;
#endif

    LwSciBufAttrList attrList = *transportContext->outputAttrList;
    LwSciIpcEndpoint ipcEndpoint = transportContext->ipcEndpoint;
    LwSciBufAttrValAccessPerm minPerms = transportContext->perms;
    LwSciBufObj* bufObj = transportContext->bufObj;

    LwSciBufObjExportDescPriv objDesc;
    LwSciBufAttrValAccessPerm finalPerms;
    LwSciBufAttrValAccessPerm actualPerms;
    LwSciBufAttrList clonedAttrList = NULL;
    LwSciBufAttrKeyValuePair keyValPair = {0};
    LwSciBufRmHandle hmem;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciC2cCopyFuncs copyFuncs = {};
    LwSciC2cInterfaceTargetHandle c2cTargetHandle = {};
#endif

    LWSCI_FNENTRY("");

    (void)memset(&objDesc, 0, sizeof(objDesc));
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    (void)memset(&copyFuncs, 0, sizeof(copyFuncs));
    (void)memset(&c2cTargetHandle, 0, sizeof(c2cTargetHandle));
#endif

    LWSCI_INFO("Value: %p AttrList: %p ipcEndpoint: %"PRIu64" MinPerms: %"
               PRIu32" BufferObj: %p \n", value, attrList,
               ipcEndpoint, minPerms, bufObj);

    err = LwSciBufAttrListCompareReconcileStatus(attrList, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Invalid attribute list to import the object.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = DeserializeLwSciBufObjExportDescPriv(&objDesc, value);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    finalPerms = LwSciBufComputeImportObjPerms(minPerms, objDesc.perms);
    if (LwSciBufAccessPerm_Ilwalid == finalPerms) {
        LWSCI_ERR_STR("LwSciBufObj Import failure due to lack of Permissions.\n");
        err = LwSciError_AccessDenied;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    keyValPair.key = LwSciBufGeneralAttrKey_ActualPerm;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &keyValPair, 1,
            LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to get ActualPerm from reconciled attrList.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    actualPerms = *(const LwSciBufAttrValAccessPerm *)keyValPair.value;
    if (actualPerms == finalPerms) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto import_obj;
    }

    err = LwSciBufAttrListClone(attrList, &clonedAttrList);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to clone the reconciled attrlist.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    keyValPair.key = LwSciBufGeneralAttrKey_ActualPerm;
    keyValPair.value = &finalPerms;
    keyValPair.len = sizeof(finalPerms);

    err = LwSciBufAttrListCommonSetAttrs(clonedAttrList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Public, true, false);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to set ActualPerm on reconciled attrList.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_attr_list;
    }
    attrList = clonedAttrList;

import_obj:
    err = LwSciBufIsSocBoundary(ipcEndpoint, &isSocBoundary);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Could not determine if ipcEndpoint belongs to Soc boundary.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_attr_list;
    }

    if (true == isSocBoundary) {
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        int c2cErrorCode = -1;

        isRemoteObject = true;

        err = LwSciIpcGetC2cCopyFuncSet(ipcEndpoint, &copyFuncs);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("LwSciIpcGetC2cCopyFuncSet failed when importing LwSciBufObj.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto free_attr_list;
        }

        c2cErrorCode = copyFuncs.bufGetHandleFromAuthToken(
                        objDesc.c2cToken.pcieAuthToken, ipcEndpoint,
                        &c2cTargetHandle.pcieTargetHandle);
        if (0 != c2cErrorCode) {
            if (EAGAIN == -c2cErrorCode) {
                LWSCI_WARN("Could not get C2c target handle from auth token this time. User should retry");
                err = LwSciError_TryItAgain;
            } else {
                LWSCI_ERR_STR("Could not get C2c target handle from auth token.");
                err = LwSciError_ResourceError;
            }
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto free_attr_list;
        }
#else
        err = LwSciError_NotSupported;
        goto free_attr_list;
#endif
    } else {
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        isRemoteObject = false;
#endif

        err = LwSciBufTransportGetMemHandle(objDesc.platformDesc,
                        ipcEndpoint, finalPerms, &hmem);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Failed to get memHandle from platform descriptor.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_attr_list;
        }
    }

    /* FIXME: Need to move offset, len to attrlist private keys. */
    err = LwSciBufTransportCreateObjFromMemHandle(hmem, objDesc.offset,
            objDesc.bufLen, attrList,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
            isRemoteObject, copyFuncs, c2cTargetHandle,
#endif
            bufObj);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to create Object from memory handle.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_attr_list;
    }

free_attr_list:
    if (NULL != clonedAttrList) {
        LwSciBufAttrListFree(clonedAttrList);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

/* Output function for the Error State */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError OutputFnError(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_BadParameter;

    LWSCI_FNENTRY("");

    (void)context;
    (void)data;

    LWSCI_FNEXIT("");
    return err;
}

static bool CheckTransportKey(
    uint32_t key,
    uint32_t expectedKey,
    size_t length)
{
    bool accept = true;
    uint32_t index;

    LWSCI_FNENTRY("");

    if (key != expectedKey) {
        accept = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if (false == LW_SCI_BUF_VALID_TRANSPORT_KEYS(expectedKey)) {
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    index = LW_SCI_BUF_TRANSKEY_IDX(expectedKey);
    if (LwSciBufTransportKeysDescTable[index].keysize != length) {
        accept = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return accept;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static bool SlotCountGuard(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    bool accept = true;

    /* Validate if the data contains the appropriate values */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* fsmContext =
        (const LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    bool importingReconciledAttr = fsmContext->importingReconciledAttr;
    uint32_t key = pair.key;
    const void* value = pair.value;
    size_t length = pair.length;

    uint64_t slotCount = 0U;

    LWSCI_FNENTRY("");

    (void)context;

    accept = CheckTransportKey(key, (uint32_t)LwSciBufTransportAttrKey_AttrListSlotCount,
        length);
    if (false == accept) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    /* Since we've validated the length, it's now safe to dereference this */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    slotCount = *(const uint64_t*)value;

    if (importingReconciledAttr) {
        /* When importing a reconciled LwSciBufAttrList, we expect that the
         * slot count is equal to 1. */
        accept = (slotCount == 1U);
    } else {
        /* When importing an unreconciled LwSciBufAttrList, we expect that the
         * slot count is at least 1. */
        accept = (slotCount >= 1U);
    }

ret:
    LWSCI_FNEXIT("");
    return accept;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static bool ReconciliationFlagGuard(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    bool accept = true;

    /* Validate if the data contains the appropriate values */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* fsmContext =
        (const LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    bool importingReconciledAttr = fsmContext->importingReconciledAttr;
    uint32_t key = pair.key;
    const void* value = pair.value;
    size_t length = pair.length;

    uint8_t reconciliationState = 0U;
    LwSciBufAttrListState expectedVal = LwSciBufAttrListState_UpperBound;

    LWSCI_FNENTRY("");

    accept = CheckTransportKey(key,
        (uint32_t)LwSciBufTransportAttrKey_AttrListReconciledFlag, length);
    if (false == accept) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    /* Since we've validated the length, it's now safe to dereference this */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    reconciliationState = *(const uint8_t*)value;

    if (importingReconciledAttr) {
        expectedVal = LwSciBufAttrListState_Reconciled;
    } else {
        expectedVal = LwSciBufAttrListState_Appended;
    }
    accept =
        (expectedVal == LwSciBufDeserializeReconcileState(reconciliationState));

ret:
    LWSCI_FNEXIT("");
    return accept;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static bool SlotIndexGuard(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    bool accept = true;

    /* Validate if the data contains the appropriate values */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* transportContext =
        (const LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    const LwSciBufAttrList* outputAttrList = transportContext->outputAttrList;

    uint32_t key = pair.key;
    const void* value = pair.value;
    size_t length = pair.length;

    size_t slotIndex = 0U;

    LWSCI_FNENTRY("");

    accept = CheckTransportKey(key,
        (uint32_t)LwSciBufTransportAttrKey_AttrListSlotIndex, length);
    if (false == accept) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Since we've validated the length, it's now safe to dereference this */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    slotIndex = *(const size_t*)value;

    /* Ensure slot index is less than the total slot count of the attribute
     * list */
    if (LwSciBufAttrListGetSlotCount(*outputAttrList) <= slotIndex) {
        accept = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* TODO: Ensure slot is monotonically increasing? */
ret:
    LWSCI_FNEXIT("");
    return accept;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static bool NewerAttributeKeysGuard(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    bool accept = true;
    LwSciError err = LwSciError_Success;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* transportContext =
        (const LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LwSciBufAttrList* attrList = transportContext->outputAttrList;
    uint64_t slotIndex = transportContext->slotIndex;
    bool importingReconciledAttr = transportContext->importingReconciledAttr;

    uint32_t key = pair.key;

    bool validationSkipped = false;

    LWSCI_FNENTRY("");

    err = LwSciBufValidateImportAttrKey(attrList, &slotIndex, key,
        importingReconciledAttr, &validationSkipped);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate imported attr list key.");
        accept = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* If the validation was skipped, this means that the current Attribute Key
     * is newer, in which case we accept and perform a no-op to consume this
     * key. */
    accept = validationSkipped;

ret:
    LWSCI_FNEXIT("");
    return accept;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static bool AttributeKeysGuard(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    bool accept = true;
    LwSciError err = LwSciError_Success;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    const LwSciBufTransportFsmContext* transportContext =
        (const LwSciBufTransportFsmContext*)context;
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LwSciBufAttrList* attrList = transportContext->outputAttrList;
    uint64_t slotIndex = transportContext->slotIndex;
    bool importingReconciledAttr = transportContext->importingReconciledAttr;

    uint32_t key = pair.key;

    bool validationSkipped = false;

    LWSCI_FNENTRY("");

    err = LwSciBufValidateImportAttrKey(attrList, &slotIndex, key,
        importingReconciledAttr, &validationSkipped);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate imported attr list key.");
        accept = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == validationSkipped) {
        /* If we skipped validation, this means that the key is newer, and so
         * this state cannot accept, since the output function only handles
         * keys that this version knows how to import. */
        accept = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* For attribute keys, we lwrrently don't split setting the attribute key
     * from validation. As such, we lwrrently don't validate this here, and
     * push this into the output function. */

ret:
    LWSCI_FNEXIT("");
    return accept;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static bool ObjDescGuard(
    void* context,
    const void* data)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    bool accept = true;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LwSciBufSerializedKeyValPair pair =
        *(const LwSciBufSerializedKeyValPair*)data;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    uint32_t key = pair.key;
    size_t length = pair.length;

    LWSCI_FNENTRY("");

    (void)context;

    accept = CheckTransportKey(key, (uint32_t)LwSciBufTransportAttrKey_ObjDesc, length);
    if (false == accept) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return accept;
}

/*********************************************
 *  Definitions of all static functions
 *********************************************/

static inline LwSciError LwSciBufValidateIpcEndpoint(
    const LwSciIpcEndpoint handle)
{
    LwSciError sciErr = LwSciError_Success;
    struct LwSciIpcEndpointInfo info;
    LWSCI_FNENTRY("");

    sciErr = LwSciIpcGetEndpointInfo(handle, &info);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to validate LwSciIpcEndpoint.\n");
        sciErr = LwSciError_BadParameter;
    }

    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufNumKeysSerialized(
    size_t numKeysHeader,
    size_t numKeyValPair,
    size_t slotCount,
    size_t index,
    uint32_t *keyCount)
{
    LwSciError sciErr = LwSciError_Success;
    size_t tmpAdd1 = 0U;
    size_t tmpAdd2 = 0U;
    size_t tmpAdd3 = 0U;

    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;
    uint8_t addStatus3 = OP_FAIL;
    LWSCI_FNENTRY("");

    u64Add(slotCount, index, &tmpAdd1, &addStatus1);
    u64Add(tmpAdd1, numKeyValPair, &tmpAdd2, &addStatus2);
    u64Add(tmpAdd2, numKeysHeader, &tmpAdd3, &addStatus3);

    if (OP_SUCCESS != (addStatus1 & addStatus2 & addStatus3)) {
        LWSCI_ERR_STR("Buffer overflow\n");
        sciErr = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (tmpAdd3 > UINT32_MAX) {
        LWSCI_ERR_STR("Casting can cause loss of data\n");
        sciErr = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *keyCount = (uint32_t)tmpAdd3;

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufTotalKeyValSize(
    size_t keySize1,
    size_t keySize2,
    size_t attrListValSize,
    size_t keySize3,
    size_t slotCount,
    size_t *totalSize)
{
    LwSciError sciErr = LwSciError_Success;
    size_t tmpMul = 0U;
    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;
    uint8_t addStatus3 = OP_FAIL;
    uint8_t addStatus4 = OP_FAIL;
    uint8_t mulStatus = OP_FAIL;
    LWSCI_FNENTRY("");

    // Header: Add LwSciBufTransportAttrKey_AttrListSlotCount value size
    u64Add(*totalSize, keySize1, totalSize, &addStatus1);

    // Header: Add LwSciBufTransportAttrKey_AttrListReconciledFlag value size
    u64Add(*totalSize, keySize2, totalSize, &addStatus2);

    // Keys for each key-value pair
    u64Add(*totalSize, attrListValSize, totalSize, &addStatus3);

    u64Mul(slotCount, keySize3, &tmpMul, &mulStatus);
    u64Add(*totalSize, tmpMul, totalSize, &addStatus4);

    if (OP_SUCCESS != (addStatus1 & addStatus2 & addStatus3 & addStatus4 & mulStatus)) {
        LWSCI_ERR_STR("Buffer overflow\n");
        sciErr = LwSciError_Success;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciBufAttrValAccessPerm LwSciBufComputeImportObjPerms(
    LwSciBufAttrValAccessPerm importAPIPerm,
    LwSciBufAttrValAccessPerm exportObjPerm)
{
    LwSciBufAttrValAccessPerm finalPerms = exportObjPerm;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs - IMP.API.PERM: %"PRIu32" EXP.OBJ.PERM: %"PRIu32"\n",
               importAPIPerm, exportObjPerm);

    if (LwSciBufAccessPerm_Auto == importAPIPerm) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto out;
    }

    if (exportObjPerm < importAPIPerm) {
        finalPerms = LwSciBufAccessPerm_Ilwalid;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto out;
    }

out:
    LWSCI_INFO("Outputs - Computed Import permissions %"PRIu32"\n", finalPerms);
    LWSCI_FNEXIT("");
    return finalPerms;
}

static LwSciError LwSciBufComputeAttrListExportPerm(
    LwSciBufAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm perms,
    LwSciBufAttrValAccessPerm* finalPerms)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrList clonedList = NULL;
    LwSciBufAttrKeyValuePair keyValPair = {};

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr)  {
        LWSCI_ERR_STR("Invalid attribute list for computing permissions.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - AttrList %p ipcEndpoint %"PRIu32
               " FinalPermptr %p Perms %"PRIu32"\n",
               reconciledAttrList, ipcEndpoint, finalPerms, perms);

    sciErr = LwSciBufAttrListClone(reconciledAttrList, &clonedList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListClone failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufAttrListReconcileFromIpcTable(clonedList,
                LwSciBufGeneralAttrKey_RequiredPerm, ipcEndpoint, false, false,
                LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to Read permissions from reconciled attrList.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    keyValPair.key = LwSciBufGeneralAttrKey_RequiredPerm;
    sciErr = LwSciBufAttrListCommonGetAttrs(clonedList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Public, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    if (0U != keyValPair.len) {
        LwSciCommonMemcpyS(finalPerms, sizeof(*finalPerms), keyValPair.value,
            keyValPair.len);
    } else {
        /* This should not happen since LwSciBufAttrListReconcileFromIpcTable()
         * would assign default value to LwSciBufGeneralAttrKey_RequiredPerm key
         * if no values are found in IPC table.
         */
        LwSciCommonPanic();
    }

    if (*finalPerms > perms) {
        LWSCI_ERR_STR("Cannot export the object with lesser permissions.\n");
        sciErr = LwSciError_IlwalidOperation;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_clonedList;
    }

    if (LwSciBufAccessPerm_Auto != perms) {
        *finalPerms = perms;
    }

    LWSCI_INFO("Outputs- Computed permissions %"PRIu32"\n", *finalPerms);

free_clonedList:
    LwSciBufAttrListFree(clonedList);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufGetIpcRouteExportDesc(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm exportAPIperms,
    void** exportDesc,
    uint64_t* len)
{
    LwSciError sciErr = LwSciError_Success;
    bool isReconciled = false;
    const LwSciBufIpcRoute* ipcRoute = NULL;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair = {};

    (void)exportAPIperms;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs - AttrList %p exportDesc %p lenptr %p \n",
              attrList, exportDesc, len);

    if (NULL != exportDesc) {
        *exportDesc = NULL;
    }
    *len = 0;

    sciErr = LwSciBufAttrListIsReconciled(attrList, &isReconciled);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to Prepare for transport.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == isReconciled) {
        LWSCI_INFO("Ipc Route is not exported for Reconciled attrlist.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto output;
    }

    pvtKeyValPair.key = LwSciBufPrivateAttrKey_SciIpcRoute;
    sciErr = LwSciBufAttrListCommonGetAttrsWithLock(attrList, slotIdx, &pvtKeyValPair,
                1, LwSciBufAttrKeyType_Private, true, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get Ipc Route from Attrlist to export.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* NULL ipcroute is exported so that during import, IPC Endpoints can be
     * appended. Hence, ipcRoute NULL is skipped.
     */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),
        "LwSciBuf-ADV-MISRAC2012-014")
    ipcRoute = *(const LwSciBufIpcRoute* const *)pvtKeyValPair.value;

    if (NULL == exportDesc) {
        *len = LwSciBufIpcRouteExportSize(ipcRoute, ipcEndpoint);
    } else {
        sciErr = LwSciBufIpcRouteExport(ipcRoute, exportDesc, len, ipcEndpoint);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to get export descriptor for Ipc Route. \n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

output:
    LWSCI_INFO("Outputs - exportDesc%p Length %"PRIu64"\n",
                exportDesc, *len);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufGetIpcTableExportDesc(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm exportAPIperms,
    void** exportDesc,
    uint64_t* len)
{
    LwSciError sciErr = LwSciError_Success;
    bool isReconciled = false;
    const LwSciBufIpcTable* ipcTable = NULL;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair = {};

    (void)slotIdx;
    (void)exportAPIperms;

    LWSCI_FNENTRY("");

    LWSCI_INFO("Inputs - AttrList %p ipcEndpoint %"PRIu64
              " exportDesc %p lenptr %p \n",
              attrList, ipcEndpoint, exportDesc, len);

    if (NULL != exportDesc) {
        *exportDesc = NULL;
    }
    *len = 0;

    sciErr = LwSciBufAttrListIsReconciled(attrList, &isReconciled);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to Prepare for transport.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == isReconciled) {
        LWSCI_INFO("IPC Table is not exported for Unreconciled list.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto output;
    }

    pvtKeyValPair.key = LwSciBufPrivateAttrKey_IPCTable;
    sciErr = LwSciBufAttrListCommonGetAttrs(attrList, slotIdx, &pvtKeyValPair,
                1, LwSciBufAttrKeyType_Private, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get Ipc Table from Attrlist to export.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),
        "LwSciBuf-ADV-MISRAC2012-014")
    ipcTable = *(LwSciBufIpcTable* const *)pvtKeyValPair.value;
    if (NULL == ipcTable) {
        LWSCI_INFO("Skipping NULL ipcTable export.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto output;
    }

    if (NULL == exportDesc) {
        *len = LwSciBufIpcTableExportSize(ipcTable, ipcEndpoint);
    } else {
        sciErr = LwSciBufIpcTableExport(ipcTable, ipcEndpoint, exportDesc, len);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to get export descriptor for Ipc Route. \n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

output:
    LWSCI_INFO("Outputs - exportDesc %p Length %"PRIu64"\n",
                exportDesc, *len);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufAttrListIpcRouteImport(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    const void* exportDesc,
    uint64_t len)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcRoute* ipcRoute = NULL;
    LwSciBufPrivateAttrKeyValuePair keyValPair = {0};

    LWSCI_FNENTRY("");

    if (0U == len) {
        LWSCI_ERR_STR("Invalid inputs to import IPCRoute.\n");
        LWSCI_ERR_ULONG("len \n", len);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - AttrList %p ipcEndpoint %"PRIu64
              " exportDesc %p len %"PRIu64"\n",
              attrList, ipcEndpoint, exportDesc, len);

    sciErr = LwSciBufAttrListCompareReconcileStatus(attrList, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to Prepare for transport.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufIpcRouteImport(ipcEndpoint, exportDesc, len, &ipcRoute);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to import IPCRoute while import.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    keyValPair.key = LwSciBufPrivateAttrKey_SciIpcRoute;
    keyValPair.value = &ipcRoute;
    keyValPair.len = sizeof(ipcRoute);

    sciErr = LwSciBufAttrListCommonSetAttrs(attrList, slotIdx, &keyValPair, 1,
                LwSciBufAttrKeyType_Private, true, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to set imported IPCRoute to attrList.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_ret;
    }

    LWSCI_INFO("Outputs - Imported IpcRoute %p\n", ipcRoute);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_ret:
    LwSciBufFreeIpcRoute(&ipcRoute);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufAttrListIpcTableImport(
    LwSciBufAttrList attrList,
    uint64_t slotIdx,
    LwSciIpcEndpoint ipcEndpoint,
    const void* exportDesc,
    uint64_t len)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufIpcTable* ipcTable = NULL;
    LwSciBufPrivateAttrKeyValuePair keyValPair = {0};

    (void)slotIdx;

    LWSCI_FNENTRY("");

    if (0U == len) {
        LWSCI_ERR_STR("Invalid inputs to import IPCTable.\n");
        LWSCI_ERR_ULONG("len \n", len);
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs - AttrList %p exportDesc %p len %"PRIu64"\n",
              attrList, exportDesc, len);

    sciErr = LwSciBufAttrListCompareReconcileStatus(attrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to Prepare for transport.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufIpcTableImport(exportDesc, len, &ipcTable, ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to import IPCTable while import.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    keyValPair.key = LwSciBufPrivateAttrKey_IPCTable;
    keyValPair.value = &ipcTable;
    keyValPair.len = sizeof(LwSciBufIpcTable*);

    sciErr = LwSciBufAttrListCommonSetAttrs(attrList, 0, &keyValPair, 1,
                LwSciBufAttrKeyType_Private, true, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to set imported IPCTable to attrList.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_ret;
    }


    LWSCI_INFO("Outputs - Imported IpcTable \n");
#if (LW_IS_SAFETY == 0)
    LwSciBufPrintIpcTable(ipcTable);
#endif
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_ret:
    LwSciBufFreeIpcTable(&ipcTable);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);

}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufValidateImportAttrKey(
    LwSciBufAttrList* attrList,
    const uint64_t* slotIdx,
    uint32_t key,
    bool importingReconciledAttr,
    bool *skippedValidation)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError sciErr = LwSciError_Success;
    uint32_t decKey = 0U;
    uint32_t tmpKeyType = 0U;
    uint32_t tmpDataType = 0U;
    void* tmpBaseAddr = NULL;
    LwSciBufAttrStatus* tmpStatus = NULL;
    uint64_t* tmpSetLen = NULL;
    LwSciBufInternalAttrKeyValuePair internalKeyValPair = {0};

    LWSCI_FNENTRY("");

    *skippedValidation = false;

    if ((NULL == attrList) || (NULL == slotIdx)) {
        LWSCI_ERR_STR("Bad descriptor. Failed to import.");
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Decode keys. Since this key may potentially be newer, we cannot actually
     * validate against the LwSciBufAttrKeyDescPriv. */
    if (LW_SCI_BUF_DECODE_KEYTYPE(key) != (uint32_t)LwSciBufAttrKeyType_UMDPrivate) {
        sciErr = LwSciBufAttrKeyDecode(key, &decKey, &tmpDataType, &tmpKeyType);
        if (LwSciError_Success != sciErr) {
            /* If importing reconciled attributes it is possible that the
             * reconciler may have populated valid keys that the receiver is
             * not aware of. */
            if (true == importingReconciledAttr) {
                *skippedValidation = true;
                sciErr = LwSciError_Success;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            } else {
                /* If this is an unreconciled attribute list, this is not a
                 * supported use case. This is because if we were to re-export
                 * this attribute list, then we would lose data. */
                LWSCI_ERR_STR("Failed to find attrkey descriptor entry.");
                sciErr = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }

        /* If we reached here, then we understand the given Attribute Key and
         * thus it is safe to fetch the data from the LwSciBufAttrKeyDescPriv. */
        LwSciBufAttrGetKeyDetail(*attrList, *slotIdx, key, &tmpBaseAddr,
            &tmpStatus, &tmpSetLen);
        if ((LwSciBufAttrStatus_SetLocked == *tmpStatus) || (0U != *tmpSetLen)) {
            LWSCI_ERR_STR("Duplicate key found. Failed to import.");
            sciErr = LwSciError_NotSupported;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    } else {
        /* We handle UMD private keys differently for validation.
         * For private UMD keys, LwSciBufAttrKeyDescPriv is NULL for
         * all the keys except the first private key for the UMD.
         * For example,
         * only LwSciBufInternalAttrKey_LwMediaPrivateFirst will
         * have LwSciBufAttrKeyDescPriv entry while all other
         * private UMD keys of LwMedia wont have the entry.
         */
        LwSciCommonMemcpyS(&internalKeyValPair.key, sizeof(internalKeyValPair.key),
                                                        &key, sizeof(key));
        sciErr = LwSciBufAttrListCommonGetAttrs(*attrList, *slotIdx,
                    &internalKeyValPair, 1,
                    LwSciBufAttrKeyType_Internal, true);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to get UMD private key value.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        if (NULL != internalKeyValPair.value) {
            LWSCI_ERR_STR("Duplicate key found. Failed to import.\n");
            sciErr = LwSciError_NotSupported;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufDeserializeDesc(
    LwSciBufFsm* fsm,
    const LwSciBufFsmState* initialState,
    LwSciBufTransportFsmContext* context,
    LwSciCommonTransportBuf* const * rxBuf)
{
    LwSciError err = LwSciError_Success;

    bool lastKey = false;

    LWSCI_FNENTRY("");

    LwSciBufFsmInit(fsm, initialState, context);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    while (false == lastKey) {
        uint32_t key = 0U;
        size_t keySize = 0UL;
        const void* value = NULL;

        LwSciError fsmOutputErr = LwSciError_Success;
        LwSciBufSerializedKeyValPair data = { 0 };
        bool transitioned = false;

        err = LwSciCommonTransportGetNextKeyValuePair(*rxBuf, &key,
                    &keySize, &value, &lastKey);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Unable to read from rx descriptor.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        data.key = key;
        data.value = value;
        data.length = keySize;

        transitioned = LwSciBufFsmEventProcess(fsm, &data, &fsmOutputErr);
        if (false == transitioned) {
            /* This FSM is set up in a way such that there are default
             * transitions for every state. As such, everything should either
             * transition or we return an error. */
            LwSciCommonPanic();
        }
        if (LwSciError_Success != fsmOutputErr) {
            err = fsmOutputErr;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufImportAttrListDesc(
    LwSciCommonTransportBuf* rxBuf,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrList* attrList,
    LwSciBufModule module,
    bool importingReconciledAttr)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    LwSciBufFsm fsm = { 0 };
    LwSciBufTransportFsmContext context = {
        .module = module,
        .outputAttrList = attrList,
        .ipcEndpoint = ipcEndpoint,
        .importingReconciledAttr = importingReconciledAttr,
        .slotIndex = 0U,
        .perms = LwSciBufAccessPerm_Ilwalid,
        .bufObj = NULL,
    };

    LWSCI_FNENTRY("");

    err = LwSciBufDeserializeDesc(&fsm, &attrFsmStateStart, &context, &rxBuf);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((&attrFsmStateAttrKeys != fsm.lwrrState) &&
        (&attrFsmStateNewerAttrKeys != fsm.lwrrState)) {
        /* If we finish parsing from the buffer, but aren't in the final state,
         * then this is considered an error.
         *
         * When importing an LwSciBufAttrList descriptor, this should be the
         * attrFsmStateAttrKeys state. Alternatively, this can be the
         * attrFsmStateNewerAttrKeys state, since we do not define the order
         * in which newer attribute keys are serialized. */
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufImportBufObjDesc(
    LwSciCommonTransportBuf* rxBuf,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrList* attrList,
    LwSciBufAttrValAccessPerm perms,
    LwSciBufObj* bufObj)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    LwSciBufFsm fsm = { 0 };
    LwSciBufTransportFsmContext context = {
        .module = NULL,
        .outputAttrList = attrList,
        .ipcEndpoint = ipcEndpoint,
        .importingReconciledAttr = false,
        .slotIndex = 0U,
        .perms = perms,
        .bufObj = bufObj,
    };

    LWSCI_FNENTRY("");

    err = LwSciBufDeserializeDesc(&fsm, &objFsmStateStart, &context, &rxBuf);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (&objFsmStateObjDesc != fsm.lwrrState) {
        /* If we finish parsing from the buffer, but aren't in the final state,
         * then this is considered an error.
         *
         * When importing an LwSciBufObj descriptor, this should be the
         * objFsmStateObjDesc state. */
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciBuf-ADV-MISRAC2012-012")
static LwSciError LwSciBufImportCombinedDesc(
    LwSciCommonTransportBuf* rxBuf,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrList* attrList,
    LwSciBufModule module,
    LwSciBufAttrValAccessPerm perms,
    LwSciBufObj* bufObj)
{
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
    LwSciError err = LwSciError_Success;

    LwSciBufFsm fsm = { 0 };
    LwSciBufTransportFsmContext context = {
        .module = module,
        .outputAttrList = attrList,
        .ipcEndpoint = ipcEndpoint,
        .importingReconciledAttr = true,
        .slotIndex = 0U,
        .perms = perms,
        .bufObj = bufObj,
    };

    LWSCI_FNENTRY("");

    err = LwSciBufDeserializeDesc(&fsm, &combinedFsmStateStart, &context,
        &rxBuf);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (&combinedFsmStateObjDesc != fsm.lwrrState) {
        /* If we finish parsing from the buffer, but aren't in the final state,
         * then this is considered an error.
         *
         * When importing a combined descriptor, this should be the
         * combinedFsmStateObjDesc state. */
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufIterateAttrList(
    LwSciBufAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    uint64_t slotIdx,
    LwSciBufAttrValAccessPerm exportPerms,
    LwSciCommonTransportBuf* buf,
    bool appendKeyValue,
    uint32_t* countkeys,
    size_t* attrListSize,
    bool isUnreconciled)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrKeyIterator iter;
    uint32_t attrKey = 0U;
    uint32_t decKey = 0U;
    LwSciBufType dataType;
    LwSciBufAttrKeyType  keyType;
    uint64_t keySize = 0U;
    void* exportDesc = NULL;
    const void* value = NULL;
    bool dataTypeEnd = false;
    bool keyTypeEnd = false;
    bool lastKey = false;
    LwSciBufAttrKeyExportCb exportFn = NULL;
    uint32_t tmpDataType = 0U;
    uint32_t tmpKeyType = 0U;
    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;
    LwSciBufAttrList clonedAttrList = NULL;
    LwSciBufAttrList tmpAttrList = NULL;

    LWSCI_FNENTRY("");

    if (false == isUnreconciled) {
        /* When exporting the reconciled list, keys having valid IPC route
         * affinity need to be recallwlated from IPC table. Thus, clone the
         * reconciled list and store the recallwlated values of keys in the
         * cloned reconciled list.
         */
        sciErr = LwSciBufAttrListClone(attrList, &clonedAttrList);
        if (LwSciError_Success != sciErr) {
            LwSciCommonPanic();
        }

        /* TODO: Fix cloning */
        tmpAttrList = clonedAttrList;
    } else {
        tmpAttrList = attrList;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_General, 1U, &iter);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    while(true) {
        uint32_t decodedKeyType = 0U;
        uint32_t decodedDataType = 0U;
        uint32_t decodedKey = 0U;
        bool acquireLock = false;

        LwSciBufAttrKeyIterNext(&iter,
            &keyTypeEnd, &dataTypeEnd, &lastKey, &attrKey);

        // terminate after complete attrKey List is traversed
        if (true == keyTypeEnd) {
            break;
        }

        // continue if either row or column end is reached in list
        if ((true == dataTypeEnd) || (true == lastKey)) {
            continue;
        }

        sciErr = LwSciBufAttrKeyDecode(attrKey, &decKey,
                                       &tmpDataType, &tmpKeyType);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Failed to decode attrkey while transport.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_clonedList;
        }

        LwSciCommonMemcpyS(&keyType, sizeof(keyType),
            &tmpKeyType, sizeof(tmpKeyType));
        LwSciCommonMemcpyS(&dataType, sizeof(dataType),
            &tmpDataType, sizeof(tmpDataType));

        exportFn = (dataType == LwSciBufType_General)?
            genAttrKeyTransportDescTable[(uint32_t)keyType][decKey].exportCallback:
            NULL;
        if (NULL != exportFn) {
            sciErr = exportFn(tmpAttrList, slotIdx, ipcEndpoint, exportPerms,
                        &exportDesc, &keySize);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_HEXUINT("Export function failed for key: ", decKey);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_exportdesc;
            }

            value = exportDesc;
        } else {
            LwSciBufIpcRouteAffinity routeAffinity = LwSciBufIpcRoute_Max;

            LwSciBufAttrKeyGetIpcRouteAffinity(attrKey, false, &routeAffinity);
            if ((LwSciBufIpcRoute_AffinityNone != routeAffinity) &&
                (0U != ipcEndpoint) && (false == isUnreconciled)) {
                /* Retrieve recallwlated values from IPC table if ipcEndpoint
                 * is non-zero, list is reconciled and the attribute has valid
                 * IPC route affinity.
                 */
                sciErr = LwSciBufAttrListReconcileFromIpcTable(tmpAttrList,
                            attrKey, ipcEndpoint, false, false,
                             LwSciBufIpcRoute_AffinityNone);
                if (LwSciError_Success != sciErr) {
                    LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed.");
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto free_clonedList;
                }
            }

            if (true == isUnreconciled) {
                /* For unreconciled list export, the locks are taken via
                 * call to LwSciBufAttrListsLock() so there is no need
                 * to take locks again.
                 */
                acquireLock = false;
            } else {
                acquireLock = true;
            }

            sciErr = LwSciBufAttrKeyDecode(attrKey, &decodedKey,
                        &decodedDataType, &decodedKeyType);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Key decoding failed.");
                LwSciCommonPanic();
            }

            if (LwSciBufAttrKeyType_Public == decodedKeyType) {
                LwSciBufAttrKeyValuePair keyValPair = {};
                keyValPair.key = attrKey;
                sciErr = LwSciBufAttrListCommonGetAttrsWithLock(tmpAttrList,
                            slotIdx, &keyValPair, 1, decodedKeyType, true,
                            acquireLock);
                if (LwSciError_Success != sciErr) {
                    LWSCI_ERR_HEXUINT("LwSciBufAttrListCommonGetAttrsWithLock failed for key: ",
                        attrKey);
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto free_clonedList;
                }

                keySize = keyValPair.len;
                value = keyValPair.value;
            } else if ((LwSciBufAttrKeyType_Internal == decodedKeyType) ||
                    (LwSciBufAttrKeyType_UMDPrivate == decodedKeyType)) {
                LwSciBufInternalAttrKeyValuePair keyValPair = {};
                keyValPair.key = attrKey;
                sciErr = LwSciBufAttrListCommonGetAttrsWithLock(tmpAttrList,
                            slotIdx, &keyValPair, 1, decodedKeyType, true,
                            acquireLock);
                if (LwSciError_Success != sciErr) {
                    LWSCI_ERR_HEXUINT("LwSciBufAttrListCommonGetAttrsWithLock failed for key: ",
                        attrKey);
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto free_clonedList;
                }

                keySize = keyValPair.len;
                value = keyValPair.value;
            } else if (LwSciBufAttrKeyType_Private == decodedKeyType) {
                LwSciBufPrivateAttrKeyValuePair keyValPair = {};
                keyValPair.key = attrKey;
                sciErr = LwSciBufAttrListCommonGetAttrsWithLock(tmpAttrList,
                            slotIdx, &keyValPair, 1, decodedKeyType, true,
                            acquireLock);
                if (LwSciError_Success != sciErr) {
                    LWSCI_ERR_HEXUINT("LwSciBufAttrListCommonGetAttrsWithLock failed for key: ",
                        attrKey);
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                    goto free_clonedList;
                }

                keySize = keyValPair.len;
                value = keyValPair.value;
            } else {
                LWSCI_ERR_HEXUINT("Invalid keytype decoded: ",
                    decodedKeyType);
                LwSciCommonPanic();
            }
        }

        if (0U == keySize) {
            continue;
        }

        if (true == appendKeyValue) {
            sciErr = LwSciCommonTransportAppendKeyValuePair(buf, attrKey, keySize, value);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Unable to append keys to transport buffer.\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_exportdesc;
            }
        } else {
             // (*countkeys)++
             u32Add(*countkeys, 1U, countkeys, &addStatus1);

             //*attrListSize += keySize
             u64Add(*attrListSize, keySize, attrListSize, &addStatus2);

             if (OP_SUCCESS != (addStatus1 & addStatus2)) {
                 LWSCI_ERR_STR("Buffer overflow\n");
                 sciErr = LwSciError_Overflow;
                 LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                 goto free_exportdesc;
             }
        }

        if (NULL != exportFn) {
            LwSciCommonFree(exportDesc);
            exportDesc = NULL;
        }
    }

free_exportdesc:
    if (NULL != exportDesc) {
        LwSciCommonFree(exportDesc);
        exportDesc = NULL;
    }

free_clonedList:
    if (false == isUnreconciled) {
        LwSciBufAttrListFree(clonedAttrList);
    }

    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufIterateUmdAttrList(
    LwSciBufAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    uint64_t slotIdx,
    LwSciCommonTransportBuf* buf,
    bool appendKeyValue,
    uint32_t* countkeys,
    size_t* attrListSize,
    bool isUnreconciled)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufUmdAttrKeyIterator umdKeyIter;
    bool lastKey = false;

    LWSCI_FNENTRY("");

    (void)ipcEndpoint;

    sciErr = LwSciBufUmdAttrKeyIterInit(attrList, slotIdx, &umdKeyIter);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to initialize UMD key iterator.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    lastKey = false;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    while(false == lastKey) {
        uint32_t attrKey = 0U;
        uint64_t keySize = 0U;
        const void* value = NULL;
        LwSciBufInternalAttrKeyValuePair keyValPair = {};
        bool acquireLock = false;

        LwSciBufUmdAttrKeyIterNext(&umdKeyIter, &lastKey, &attrKey);

        if (true == lastKey) {
            continue;
        }

        if (true == isUnreconciled) {
            /* For unreconciled list export, the locks are taken via
             * call to LwSciBufAttrListsLock() so there is no need
             * to take locks again.
             */
            acquireLock = false;
        } else {
            acquireLock = true;
        }

        keyValPair.key = attrKey;
        sciErr = LwSciBufAttrListCommonGetAttrsWithLock(attrList, slotIdx,
                    &keyValPair, 1, LwSciBufAttrKeyType_UMDPrivate, true,
                    acquireLock);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_HEXUINT("Unable to get Value for AttrKey .\n", attrKey);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
        keySize = keyValPair.len;
        value = keyValPair.value;

        if (0U == keySize) {
            continue;
        }

        if (appendKeyValue) {
            sciErr = LwSciCommonTransportAppendKeyValuePair(buf, attrKey, keySize, value);
            if (LwSciError_Success != sciErr) {
                LWSCI_ERR_STR("Unable to append keys to transport buffer.\n");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        } else {
            uint8_t addStatus1 = OP_FAIL;
            uint8_t addStatus2 = OP_FAIL;

            //(*countkeys)++
            u32Add(*countkeys, 1U, countkeys, &addStatus1);

            //*attrListSize += keySize
            u64Add(*attrListSize, keySize, attrListSize, &addStatus2);

            if (OP_SUCCESS != (addStatus1 & addStatus2)) {
                LWSCI_ERR_STR("Buffer overflow\n");
                sciErr = LwSciError_Overflow;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto ret;
            }
        }
    }
ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

/**
 * \brief Callwlate the number of key-value pairs on an Attribute List, the
 * number of slots, and the size of the values stored in key-value pairs.
 *
 * \param[in] attrList the Attribute List to process
 * \param[in] ipcEndpoint LwSciIpcEndpoint to identify the peer process.
 * \param[out] keyCount the total amount of key-value pairs that the Attribute
 *      List contains (ie. # of keys * \a slotCount)
 * \param[out] slotCount the number of slots this Attribute List contains
 * \param[out] valueSize the total size of all the values stored in key-value
 *      pairs in this Attribute List (ie. not including the size of the keys)
 */
static LwSciError LwSciBufGetAttrListSlotKeyCountAndSize(
    LwSciBufAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    uint32_t* keyCount,
    size_t* slotCount,
    size_t* valueSize,
    bool isUnreconciled)
{
    LwSciError sciErr = LwSciError_Success;
    size_t slotIdx = 0;
    uint32_t totalKeyValPairs = 0;
    size_t attrListValsSize = 0;

    LWSCI_FNENTRY("");
    sciErr = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid input arguments.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("Inputs: Attrlist: %p Keycountptr: %p Valuesizeptr: %p\n",
                        attrList, keyCount, valueSize);

    *slotCount = LwSciBufAttrListGetSlotCount(attrList);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (slotIdx = 0; slotIdx < *slotCount; slotIdx++) {
        LwSciIpcEndpoint tmpIpcEndpoint = 0U;
        sciErr =  LwSciBufIterateAttrList(attrList, ipcEndpoint, slotIdx,
                    LwSciBufAccessPerm_Ilwalid, NULL, false,
                    &totalKeyValPairs, &attrListValsSize, isUnreconciled);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to append keys to tranport buffer.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        /* If this function is called from unreconciled export then set
         * ipcEndpoint to 0 here to be passed to LwSciBufIterateUmdAttrList.
         */
        if (true == isUnreconciled) {
            tmpIpcEndpoint = 0U;
        } else {
            tmpIpcEndpoint = ipcEndpoint;
        }

        sciErr = LwSciBufIterateUmdAttrList(attrList, tmpIpcEndpoint, slotIdx,
                    NULL, false, &totalKeyValPairs, &attrListValsSize,
                    isUnreconciled);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to Umd key to transport buffer.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    *keyCount = totalKeyValPairs;
    *valueSize = attrListValsSize;
    LWSCI_INFO("Outputs: Keycount: %p Valuesize: %p\n", *keyCount, *valueSize);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufAppendListHeader(
    LwSciBufAttrList attrList,
    uint64_t totalSlotCount,
    LwSciCommonTransportBuf* buf,
    bool appendListHeader)
{
    LwSciError sciErr = LwSciError_Success;
    bool isReconciled = false;
    uint8_t reconciledFlag = 0;

    LWSCI_FNENTRY("");

    if (true == appendListHeader) {
        sciErr = LwSciCommonTransportAppendKeyValuePair(buf,
                    (uint32_t)LwSciBufTransportAttrKey_AttrListSlotCount,
                    sizeof(totalSlotCount), &totalSlotCount);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to append slot-count to transport buffer.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        sciErr = LwSciBufAttrListIsReconciled(attrList, &isReconciled);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Invalid attribute list to Prepare for transport.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        reconciledFlag = (isReconciled == true) ? 1U : 0U;
        sciErr = LwSciCommonTransportAppendKeyValuePair(buf,
                    (uint32_t)LwSciBufTransportAttrKey_AttrListReconciledFlag,
                    sizeof(reconciledFlag), &reconciledFlag);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to append reconciled flag to transport buffer.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufPrepareAttrlistForTransport(
    LwSciBufAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    uint64_t totalSlotCount,
    uint64_t firstSlotIndex,
    LwSciBufAttrValAccessPerm exportPerms,
    LwSciCommonTransportBuf* buf,
    bool appendListHeader,
    bool isUnreconciled)
{
    LwSciError sciErr = LwSciError_Success;
    uint64_t slotCount = 0;
    uint64_t slotIdx = 0;
    uint64_t slotIdxarg = 0;

    LWSCI_FNENTRY("");
    sciErr = LwSciBufAttrListValidate(attrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid input arguments.\n");
        LWSCI_ERR_INT("sciErr: \n", (int32_t)sciErr);
        LWSCI_ERR_ULONG("ipcEndpoint: \n", ipcEndpoint);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufAppendListHeader(attrList, totalSlotCount, buf, appendListHeader);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to append attribute to transport buffer.\n");
    }

    slotCount = LwSciBufAttrListGetSlotCount(attrList);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (slotIdx = 0; slotIdx < slotCount; slotIdx ++) {
        LwSciIpcEndpoint tmpIpcEndpoint = 0U;
        slotIdxarg = firstSlotIndex + slotIdx;
        sciErr = LwSciCommonTransportAppendKeyValuePair(buf,
                   (uint32_t)LwSciBufTransportAttrKey_AttrListSlotIndex,
                   sizeof(slotIdx), &slotIdxarg);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to append slot-no to transport buffer.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        sciErr =  LwSciBufIterateAttrList(attrList, ipcEndpoint, slotIdx,
                    exportPerms, buf, true, NULL, NULL, isUnreconciled);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to append keys to tranport buffer.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

      /* If this function is called from unreconciled export then set
       * ipcEndpoint to 0 here to be passed to LwSciBufIterateUmdAttrList.
       */
       if (true == isUnreconciled) {
            tmpIpcEndpoint = 0U;
        } else {
            tmpIpcEndpoint = ipcEndpoint;
        }

        sciErr = LwSciBufIterateUmdAttrList(attrList, tmpIpcEndpoint, slotIdx,
                    buf, true, NULL, NULL, isUnreconciled);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to Umd key to transport buffer.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    LWSCI_INFO("Successfully Appended All Keys of Attrlist\n");

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
static void colwertPermsToC2cPerms(
    LwSciBufAttrValAccessPerm bufPerms,
    LwSciC2cPciePermissions* c2cPerms)
{
    LWSCI_FNENTRY("");

    switch (bufPerms) {
        case LwSciBufAccessPerm_Readonly:
        {
            *c2cPerms = LWSCIC2C_PCIE_PERM_READONLY;
            break;
        }

        case LwSciBufAccessPerm_ReadWrite:
        {
            *c2cPerms = LWSCIC2C_PCIE_PERM_READWRITE;
            break;
        }

        default:
        {
            /* unsupported case. */
            LwSciCommonPanic();
        }
    }

    LWSCI_FNEXIT("");
}
#endif

static LwSciError LwSciBufPrepareObjForTransport(
    LwSciBufObj bufObj,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufAttrValAccessPerm perms,
    LwSciCommonTransportBuf* txBuf)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufRmHandle hmem;
    LwSciBufObjExportDescPriv objDesc;
    uint64_t offset = 0U;
    uint64_t len = 0U;
    LwSciBufAttrValAccessPerm finalPerms = LwSciBufAccessPerm_Ilwalid;
    LwSciBufAttrList reconciledAttrList = NULL;
    bool isSocBoundary = false;

    LWSCI_FNENTRY("");

    (void)memset(&objDesc, 0x0, sizeof(objDesc));

    sciErr = LwSciBufObjGetAttrList(bufObj, &reconciledAttrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to get attrlist from object. \n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufComputeAttrListExportPerm(reconciledAttrList,
                            ipcEndpoint, perms, &finalPerms);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to compute export permissions.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufObjGetMemHandle(bufObj, &hmem, &offset, &len);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to Get Memory handle.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufIsSocBoundary(ipcEndpoint, &isSocBoundary);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufIsSocBoundary() failed when preparing LwSciBufObj for transport.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (true == isSocBoundary) {
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        LwSciC2cCopyFuncs copyFuncs = {};
        /* TODO: If LwSciC2c provides interface agnostic handle then use that
         * otherwise we need to figure out the backend type and use either
         * PCIe or NPM handle. For 6.0.2.0, only PCIe is supported.
         */
        LwSciC2cInterfaceTargetHandle c2cTargetHandle = {};
        LwSciC2cPcieBufRmHandle pcieRmHandle = {};
        LwSciC2cPciePermissions c2cPerms = LWSCIC2C_PCIE_PERM_ILWALID;
        int c2cErrorCode = -1;

        sciErr = LwSciBufTransportSetC2cRmHandle(hmem, &pcieRmHandle);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciBufTransportSetC2cRmHandle failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        colwertPermsToC2cPerms(finalPerms, &c2cPerms);

        sciErr = LwSciIpcGetC2cCopyFuncSet(ipcEndpoint, &copyFuncs);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciIpcGetC2cCopyFuncSet() failed when preparing LwSciBufObj for transport.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        sciErr = LwSciBufObjSetC2cCopyFunctions(bufObj, copyFuncs);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciBufObjSetC2cCopyFunctions() failed when preparing LwSciBufObj for transport.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        c2cErrorCode = copyFuncs.bufMapTargetMemHandle(&pcieRmHandle, c2cPerms,
                        ipcEndpoint, &c2cTargetHandle.pcieTargetHandle);
        if (0 != c2cErrorCode) {
            LWSCI_ERR_STR("Could not map bufObj to C2c target handle.");
            sciErr = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        sciErr = LwSciBufObjSetC2cTargetHandle(bufObj, c2cTargetHandle);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("LwSciBufObjSetTargetHandle when preparing bufObj for transport.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        c2cErrorCode = copyFuncs.bufGetAuthTokenFromHandle(
                        c2cTargetHandle.pcieTargetHandle, ipcEndpoint,
                        &objDesc.c2cToken.pcieAuthToken);
        if (0 != c2cErrorCode) {
            if (EAGAIN == -c2cErrorCode) {
                LWSCI_WARN("Could not get auth token from C2c target handle this time. User should retry.");
                sciErr = LwSciError_TryItAgain;
            } else {
                LWSCI_ERR_STR("Could not get auth token from C2c target handle.");
                sciErr = LwSciError_ResourceError;
            }
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
                "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
#else
        sciErr = LwSciError_NotSupported;
        goto ret;
#endif
    } else {
        sciErr = LwSciBufTransportGetPlatformDesc(hmem,
                        ipcEndpoint, finalPerms, &objDesc.platformDesc);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to get platformDesc from LwSciBufRmHandle.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    /* FIXME: Check if offset and len can be moved to private Keys
     * and transport via reconciled-attrlist.
     */
    objDesc.offset = offset;
    objDesc.bufLen = len;
    objDesc.perms = finalPerms;

    sciErr = LwSciCommonTransportAppendKeyValuePair(txBuf,
                                  (uint32_t)LwSciBufTransportAttrKey_ObjDesc,
                                  sizeof(objDesc), &objDesc);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to append objDesc to transport buffer.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufImportBuffer(
    const void* descBuf,
    size_t desclen,
    LwSciCommonTransportBuf** rxBuf)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciCommonTransportParams params = { 0U };

    LWSCI_FNENTRY("");

    sciErr = LwSciCommonTransportGetRxBufferAndParams(descBuf, desclen,
                                                      rxBuf, &params);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to read rx descriptor.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (LW_SCI_BUF_TRANSPORT_MAGIC != params.msgMagic) {
        LWSCI_ERR_STR("Import Descriptor doesn't belong to LwSciBuf.");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    if (LW_SCI_BUF_VERSION_MAJOR(params.msgVersion) != LwSciBufMajorVersion) {
        LWSCI_ERR_STR("Cannot import Descriptor from incompatible Version.\n");
        sciErr = LwSciError_NotSupported;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

static LwSciError LwSciBufAttrListReconciledAllSetCheckHelper(
    LwSciBufAttrKeyIterator* iter,
    LwSciBufAttrList reconciledList)
{
    LwSciError sciErr = LwSciError_Success;

    bool datatypeEnd = false;
    bool keytypeEnd = false;
    uint32_t key = 0U;
    bool keyEnd = false;
    void* tmpBaseAddr = NULL;
    LwSciBufAttrStatus* tmpStatus = NULL;
    uint64_t* tmpSetLen = NULL;
    LwSciBufKeyAccess keyAccess = LwSciBufKeyAccess_InOut;

    LWSCI_FNENTRY("");

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    while (true) {
        bool needsCheck = false;

        LwSciBufAttrKeyIterNext(iter, &keytypeEnd, &datatypeEnd,
            &keyEnd, &key);

        if (true == keyEnd) {
            break;
        }

        LwSciBufAttrGetKeyAccessDetails(key, &keyAccess);

        sciErr = LwSciBufImportCheckingNeeded(reconciledList, key, &needsCheck);
        if (LwSciError_Success != sciErr) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }

        if (false == needsCheck) {
            continue;
        }

        LwSciBufAttrGetKeyDetail(reconciledList, 0, key, &tmpBaseAddr, &tmpStatus,
                    &tmpSetLen);

        if ((LwSciBufAttrStatus_SetLocked != *tmpStatus) || (0U == *tmpSetLen)) {
            LWSCI_ERR_STR("key value not set.\n");
            sciErr = LwSciError_NotSupported;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufAttrListReconciledAllSetCheck(
    LwSciBufAttrList reconciledList)
{
    LwSciError sciErr = LwSciError_Success;

    LwSciBufAttrKeyIterator iter;
    const LwSciBufType* bufType = NULL;
    size_t numDataTypes = 0;
    uint32_t index1 = 0U;

    LWSCI_FNENTRY("");

    /*Iterate through the general keys*/
    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)LwSciBufType_General, 1U, &iter);

    sciErr = LwSciBufAttrListReconciledAllSetCheckHelper(&iter, reconciledList);

    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Internal,
        (uint32_t)LwSciBufType_General, 1U, &iter);

    sciErr = LwSciBufAttrListReconciledAllSetCheckHelper(&iter, reconciledList);

    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufAttrListGetDataTypes(reconciledList, &bufType, &numDataTypes);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufAttrListGetDataTypes failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (index1 = 0U; index1 < numDataTypes; index1++) {

        /*Iterate through the data type specific keys*/
        LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
            (uint32_t)bufType[index1], 1U, &iter);

        sciErr = LwSciBufAttrListReconciledAllSetCheckHelper(&iter, reconciledList);

        if (LwSciError_Success != sciErr) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static void CompareReconciledAttrListsHelper(
    LwSciBufAttrKeyIterator* iter,
    LwSciBufAttrList clonedReconciledList,
    LwSciBufAttrList importedReconciledList,
    bool* areEqual)
{
    LWSCI_FNENTRY("");
    *areEqual = false;

    while (true) {
        uint32_t key = 0U;
        bool keyEnd = false;
        bool datatypeEnd = false;
        bool keytypeEnd = false;
        void* clonedAttrBaseAddr = NULL;
        void* importedAttrBaseAddr = NULL;
        LwSciBufAttrStatus* clonedAttrStatus = NULL;
        LwSciBufAttrStatus* importedAttrStatus = NULL;
        uint64_t* clonedAttrSize = NULL;
        uint64_t* importedAttrSize = NULL;
        int cmpResult = 0;

        LwSciBufAttrKeyIterNext(iter, &keytypeEnd, &datatypeEnd, &keyEnd, &key);
        if (keyEnd == true) {
            break;
        }

        LwSciBufAttrGetKeyDetail(clonedReconciledList, 0, key,
            &clonedAttrBaseAddr, &clonedAttrStatus, &clonedAttrSize);
        LwSciBufAttrGetKeyDetail(importedReconciledList, 0, key,
            &importedAttrBaseAddr, &importedAttrStatus, &importedAttrSize);

        if (((clonedAttrSize == NULL) && (importedAttrSize != NULL)) ||
            ((clonedAttrSize != NULL) && (importedAttrSize == NULL))) {
            /* If any of the attrSize is NULL, it would mean that there is no
             * memory allocated for this key (See LwSciBufPerSlotAttrList).
             * (This can only happen for datatype keys).
             * If memory is allocated for key in one attribute list but not
             * for other attribute list then it means that they don't match.
             */
            *areEqual = false;
            goto ret;
        }

        if (*clonedAttrSize != *importedAttrSize) {
            *areEqual = false;
            goto ret;
        }

        if (key == (uint32_t)LwSciBufGeneralAttrKey_ActualPerm) {
            /* LwSciBufGeneralAttrKey_ActualPerm is special because when it is
             * exported, the reconciled value callwlated from IPC table may
             * get replaced by permissions requested by the API provided API
             * requested permissions are greater than the ActualPerm callwlated
             * during export. When we clone the imported reconciled attribute
             * list and reconcile it again, the ActualPerm in the cloned
             * reconciled list will get callwlated to value of RequestedPerm in
             * the cloned reconciled list and thus, the ActualPerm value in
             * imported reconciled list and cloned reconciled list may not match
             * . In that case, as long as ActualPerm value in cloned reconciled
             * list is smaller or equal to the value in imported reconciled list
             *, it would suffice. Note that, we have to add this check here in
             * 5.2 because at some point in attribute constraint unit, we copy
             * the value from RequiredPerm to ActualPerm during reconciliation.
             * When we move to new reconciliation framework, this check won't
             * be needed.
             */
            LwSciBufAttrValAccessPerm clonedListActualPerm =
                *(LwSciBufAttrValAccessPerm *)clonedAttrBaseAddr;
            LwSciBufAttrValAccessPerm importedListActualPerm =
                *(LwSciBufAttrValAccessPerm *)importedAttrBaseAddr;

            if (clonedListActualPerm > importedListActualPerm) {
                *areEqual = false;
                goto ret;
            }
        }

        if (key == (uint32_t)LwSciBufPrivateAttrKey_IPCTable) {
            /* comparing IPC table value is tricky because ipc table unit
             * exposes LwSciBufIpcTable* opaque handle which is stored as part
             * of LwSciBufPrivateAttrKey_IPCTable attribute. When the imported
             * reconciled list is cloned and reconciled again, the
             * LwSciBufIpcTable value in cloned list and imported list won't
             * match (since its a pointer). We could compare the underlying
             * LwSciBufIpcTableRec struct itself but is not exposed outside
             * IPC table unit. However, since LwSciBufPrivateAttrKey_IPCTable
             * is not dependent on other output attributes in the reconciled
             * list, the contents of the IPC table in the cloned list and
             * imported list will always match and thus, we can safely skip the
             * check.
             */
            continue;
        }

        cmpResult = LwSciCommonMemcmp(clonedAttrBaseAddr, importedAttrBaseAddr,
                        *clonedAttrSize);
        if (cmpResult != 0) {
            *areEqual = false;
            goto ret;
        }
    }

    *areEqual = true;

ret:
    LWSCI_FNEXIT("");
}

static void CompareReconciledAttrListsPerDatatype(
    LwSciBufType bufType,
    LwSciBufAttrList clonedReconciledList,
    LwSciBufAttrList importedReconciledList,
    bool* areEqual)
{
    LwSciBufAttrKeyIterator iter = {0};
    uint32_t keyOffset = 0U;

    LWSCI_FNENTRY("");

    *areEqual = false;

    if (bufType == LwSciBufType_General) {
        /* for LwSciBufType_General, start from keyoffset 1 */
        keyOffset = 1U;
    } else {
        /* for other LwSciBufTypes, start from keyoffset 0 */
        keyOffset = 0U;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Public,
        (uint32_t)bufType, keyOffset, &iter);
    CompareReconciledAttrListsHelper(&iter, clonedReconciledList,
        importedReconciledList, areEqual);
    if (*areEqual == false) {
        goto ret;
    }

    LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Internal,
        (uint32_t)bufType, keyOffset, &iter);
    CompareReconciledAttrListsHelper(&iter, clonedReconciledList,
        importedReconciledList, areEqual);
    if (*areEqual == false) {
        goto ret;
    }

    if (bufType == LwSciBufType_General) {
        LwSciBufAttrKeyIterInit((uint32_t)LwSciBufAttrKeyType_Private,
            (uint32_t)bufType, keyOffset, &iter);
        CompareReconciledAttrListsHelper(&iter, clonedReconciledList,
            importedReconciledList, areEqual);
        if (*areEqual == false) {
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
}

static LwSciError CompareReconciledAttrLists(
    LwSciBufAttrList clonedReconciledList,
    LwSciBufAttrList importedReconciledList,
    bool* areEqual)
{
    LwSciError err = LwSciError_Success;
    LwSciBufAttrKeyValuePair keyVal = {0};
    const LwSciBufType* bufTypes = NULL;
    size_t bufLen = 0U;
    size_t index = 0U;

    LWSCI_FNENTRY("");

    *areEqual = false;

    CompareReconciledAttrListsPerDatatype(LwSciBufType_General,
        clonedReconciledList, importedReconciledList, areEqual);
    if (*areEqual == false) {
        goto ret;
    }

    keyVal.key = LwSciBufGeneralAttrKey_Types;
    err = LwSciBufAttrListCommonGetAttrs(clonedReconciledList, 0, &keyVal, 1,
            LwSciBufAttrKeyType_Public, true);
    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListCommonGetAttrs() failed.");
        goto ret;
    }

    if (keyVal.len == 0U) {
        goto ret;
    }

    bufLen = keyVal.len / sizeof(LwSciBufType);
    bufTypes = (const LwSciBufType *)keyVal.value;

    for (index = 0U; index < bufLen; index++) {
        CompareReconciledAttrListsPerDatatype(bufTypes[index],
            clonedReconciledList, importedReconciledList, areEqual);
        if (*areEqual == false) {
            goto ret;
        }
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError CheckImportedReconciledListConsistency(
    LwSciBufAttrList importedReconciledAttrList)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrList clonedReconciledAttrList = NULL;
    LwSciBufAttrList conflictList = NULL;
    bool areReconciledListsEqual = false;

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListReconciledAllSetCheck(importedReconciledAttrList);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("Reconciled attribute import error.\n");
        goto ret;
    }

    sciErr = LwSciBufAttrListClone(importedReconciledAttrList,
            &clonedReconciledAttrList);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListClone() failed.");
        goto ret;
    }

    /* We check the consistency of output attributes in the imported reconciled
     * attribute list by doing the following:
     * 1. Imported reconciled attribute list is cloned.
     * 2. Cloned reconciled list is passed to the
     *    LwSciBufAttrListReconcileInternal() API for reconciliation.
     * 3. The step 2) above will recallwlate the output attributes which are
     *    dependent on other output attributes in the reconciled list.
     *    For ex: LwSciBufImageAttrKey_Size is dependent on reconciled values
     *    of LwSciBufImageAttrKey_Width and LwSciBufImageAttrKey_Height.
     *    Thus, we recallwlate the LwSciBufImageAttrKey_Size in the cloned
     *    reconciled list.
     * 4. Compare the output attributes from imported reconciled list and cloned
     *    reconciled list. If all the output attributes match, it would mean
     *    that the imported reconciled list is consistent with the reconciled
     *    attribute list that LwSciBuf would have callwlated.
     */
    sciErr = LwSciBufAttrListReconcileInternal(NULL, 0,
                clonedReconciledAttrList, &conflictList, true);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileInternal() failed.");

        if (sciErr == LwSciError_ReconciliationFailed) {
            sciErr = LwSciError_BadParameter;
        }

        goto free_conflictlist;
    }

    sciErr = CompareReconciledAttrLists(clonedReconciledAttrList,
        importedReconciledAttrList, &areReconciledListsEqual);
    if (sciErr != LwSciError_Success) {
        LWSCI_ERR_STR("CompareReconciledAttrLists() failed.");
        goto free_attrlist;
    }

    if (areReconciledListsEqual == false) {
        sciErr = LwSciError_BadParameter;
        goto free_attrlist;
    }

free_attrlist:
    LwSciBufAttrListFree(clonedReconciledAttrList);

free_conflictlist:
    if (conflictList != NULL) {
        LwSciBufAttrListFree(conflictList);
    }

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static LwSciError LwSciBufTransportCheckIpcPath(
    LwSciBufAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    bool* ipcPathExists)
{
    LwSciError err = LwSciError_Success;
    LwSciBufIpcTable* const * ipcTablePtr = NULL;
    size_t ipcTableLen = 0U;
    LwSciBufIpcTableIter* ipcTableIter = NULL;
    LwSciBufPrivateAttrKeyValuePair pvtKeyValPair = {};

    LWSCI_FNENTRY("");

    *ipcPathExists = false;

    pvtKeyValPair.key = LwSciBufPrivateAttrKey_IPCTable;
    err = LwSciBufAttrListCommonGetAttrs(attrList, 0, &pvtKeyValPair, 1,
            LwSciBufAttrKeyType_Private, true);
    if (LwSciError_Success != err) {
        /* This should not happen */
        LwSciCommonPanic();
    }

    ipcTablePtr = (LwSciBufIpcTable* const *)pvtKeyValPair.value;
    ipcTableLen = pvtKeyValPair.len;
    if ((0U == ipcTableLen) || (NULL == ipcTablePtr)) {
        /* IPC path does not exist since IPC table is not present in the
         * attribute list at all.
         */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Pass affinity = LwSciBufIpcRoute_RouteAffinity here since we are trying
     * to search if the ipcEndpoint exists in any of the IPC routes in the
     * IPC table.
     */
    err = LwSciBufInitIpcTableIter(*ipcTablePtr, &ipcTableIter);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufInitIpcTableIter failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    while (LwSciBufIpcTableIterNext(ipcTableIter)) {
        const LwSciBufIpcRoute* ipcRoute = NULL;
        LwSciBufIpcGetRouteFromIter(ipcTableIter, &ipcRoute);
        LwSciBufIpcRouteMatchAffinity(ipcRoute, LwSciBufIpcRoute_RouteAffinity,
            ipcEndpoint, false, ipcPathExists);
        if (true == *ipcPathExists) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ipc_itr;
        }
    }

    /* If we are here, it implies that IPC path does not exist */

free_ipc_itr:
    LwSciBufFreeIpcIter(ipcTableIter);

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError RecallwlatePlatformAttributesForReconciledImport(
    LwSciBufAttrList reconciledList,
    LwSciIpcEndpoint ipcEndpoint)
{
    /* The attributes callwlated below are dependent on the platform
     * for their callwlation. As such, in Inter-Soc case, the exporting
     * peer might not be capable of callwlating these keys during
     * export. For ex: in cheetah <-> X86 case, if cheetah Soc is exporter
     * then peer in cheetah cannot query underlying GPUs in the system.
     * As such, we callwlate these keys during import.
     */
    LwSciError err = LwSciError_Success;
    bool isSocBoundary = false;

    LWSCI_FNENTRY("");

    err = LwSciBufIsSocBoundary(ipcEndpoint, &isSocBoundary);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufIsSocBoundary failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == isSocBoundary) {
        /* skip recallwlation */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* Note that we are callwlating the attributes here in sequence of their
     * dependency. For ex: LwSciBufGeneralAttrKey_GpuId must be callwlated
     * before we can callwlate LwSciBufGeneralAttrKey_EnableGpuCache since
     * LwSciBufGeneralAttrKey_EnableGpuCache depends on
     * LwSciBufGeneralAttrKey_GpuId.
     */
    err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
            LwSciBufGeneralAttrKey_VidMem_GpuId, 0U, true, false,
            LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed when recallwlating LwSciBufGeneralAttrKey_VidMem_GpuId during reconciled list import.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
            LwSciBufInternalGeneralAttrKey_MemDomainArray, 0U, true, false,
            LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed when recallwlating LwSciBufInternalGeneralAttrKey_MemDomainArray during reconciled list import.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
            LwSciBufGeneralAttrKey_GpuId, 0U, true, false,
            LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed when recallwlating LwSciBufGeneralAttrKey_GpuId during reconciled list import.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
            LwSciBufGeneralAttrKey_EnableGpuCache, 0U, true, false,
            LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed when recallwlating LwSciBufGeneralAttrKey_EnableGpuCache during reconciled list import.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
            LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, 0U, true, false,
            LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed when recallwlating LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency during reconciled list import.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
            LwSciBufGeneralAttrKey_EnableGpuCompression, 0U, true, false,
            LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed when recallwlating LwSciBufGeneralAttrKey_EnableGpuCompression during reconciled list import.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufAttrListReconcileFromIpcTable(reconciledList,
            LwSciBufPrivateAttrKey_HeapType, 0U, true, false,
            LwSciBufIpcRoute_AffinityNone);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("LwSciBufAttrListReconcileFromIpcTable() failed when recallwlating LwSciBufPrivateAttrKey_HeapType during reconciled list import.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1),
            "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

static LwSciError LwSciBufTransportValidateAccessPerm(
    LwSciBufAttrValAccessPerm perm)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    switch (perm) {
        case LwSciBufAccessPerm_Readonly:
        case LwSciBufAccessPerm_ReadWrite:
        case LwSciBufAccessPerm_Auto:
        {
            error = LwSciError_Success;
            break;
        }
        default:
        {
            error = LwSciError_BadParameter;
            break;
        }
    }

    LWSCI_FNEXIT("");
    return error;
}

/****************************************************
 * Public Functions - Export functionality
 ****************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufIpcExportAttrListAndObj(
    LwSciBufObj bufObj,
    LwSciBufAttrValAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** attrListAndObjDesc,
    size_t* attrListAndObjDescSize)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufAttrList reconciledAttrList = NULL;
    LwSciCommonTransportBuf* txBuf = NULL;
    LwSciCommonTransportParams bufparams;

    LwSciBufTransportKeyDesc keyDescriptor;
    LwSciBufTransportKeyDesc keyDescriptor2;
    LwSciBufTransportKeyDesc keyDescriptor3;

    uint32_t totalKeyValPairs = 0U;
    size_t slotCount = 0U;
    size_t attrListValsSize = 0U;

    size_t totalValsSize = 0U;
    uint8_t status = OP_FAIL;
    bool ipcPathExists = false;

    LWSCI_FNENTRY("");
    if ((NULL == bufObj) || (0U == ipcEndpoint) ||
        (attrListAndObjDesc == NULL) || (attrListAndObjDescSize == NULL)) {
        LWSCI_ERR_STR("Invalid input arguments to export reconciled list.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufTransportValidateAccessPerm(permissions);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("wrong parameter LwSciBufAttrValAccessPerm supplied.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufObjGetAttrList(bufObj, &reconciledAttrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to get attribute list for lwscibuf obj.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufTransportCheckIpcPath(reconciledAttrList, ipcEndpoint,
                &ipcPathExists);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufTransportCheckIpcPath failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == ipcPathExists) {
        LWSCI_ERR_STR("When IPC is ilwolved, reconciled LwSciBufAttrLists and objects must flow in the reverse IPC path (ie. originate from the allocator application) of unreconciled LwSciBufAttrLists.");
        sciErr = LwSciError_NotPermitted;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufGetAttrListSlotKeyCountAndSize(reconciledAttrList,
            ipcEndpoint, &totalKeyValPairs, &slotCount, &attrListValsSize,
            false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to get key count and size for attribute list.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    bufparams.msgVersion = LW_SCI_BUF_VERSION;
    bufparams.msgMagic = LW_SCI_BUF_TRANSPORT_MAGIC;
    /**
     * We are exporting keys, which come from:
     *
     * 1. Header
     * 2. Keys for each key-value pair
     * 3. AttrListSlotIndex for each Slot Attribute List
     * 4. LwSciBufTransportAttrKey_ObjDesc
     */
    sciErr = LwSciBufNumKeysSerialized(LW_SCI_BUF_NUM_ATTRLIST_HEADER_TRANSPORT_KEYS, totalKeyValPairs, slotCount,
                                       1U, &bufparams.keyCount);

    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    // Header: Add LwSciBufTransportAttrKey_AttrListSlotCount value size
    keyDescriptor = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotCount)];

    // Header: Add LwSciBufTransportAttrKey_AttrListReconciledFlag value size
    keyDescriptor2 = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListReconciledFlag)];

    // Add the size of the exported AttrListSlotIndex key values
    keyDescriptor3 = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotIndex)];
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    sciErr = LwSciBufTotalKeyValSize(keyDescriptor.keysize,
                                     keyDescriptor2.keysize,
                                     attrListValsSize,
                                     keyDescriptor3.keysize,
                                     slotCount,
                                     &totalValsSize);

    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    // Add size of LwSciBufTransportAttrKey_ObjDesc value
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    keyDescriptor = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_ObjDesc)];
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    u64Add(totalValsSize, keyDescriptor.keysize, &totalValsSize, &status);
    if (OP_SUCCESS != status) {
        LWSCI_ERR_STR("Buffer overflow\n");
        sciErr = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalValsSize,
                                                     &txBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to allocate tx transport descriptor.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufPrepareAttrlistForTransport(reconciledAttrList,
                            ipcEndpoint,
                            LwSciBufAttrListGetSlotCount(reconciledAttrList),
                            0U, permissions, txBuf, true, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to prepare attrList in the array.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_txBuf;
    }

    sciErr = LwSciBufPrepareObjForTransport(bufObj,
                            ipcEndpoint, permissions, txBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to prepare descriptor for lwscibufobj.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_txBuf;
    }

    LwSciCommonTransportPrepareBufferForTx(txBuf, attrListAndObjDesc,
                            attrListAndObjDescSize);

    LWSCI_INFO("Outputs: descBuf: %p descLen: %p\n",
        *attrListAndObjDesc, *attrListAndObjDescSize);

free_txBuf:
    LwSciCommonTransportBufferFree(txBuf);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjIpcExport(
    LwSciBufObj bufObj,
    LwSciBufAttrValAccessPerm accPerm,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciBufObjIpcExportDescriptor* exportData)
{
    LwSciCommonTransportBuf* txBuf = NULL;
    LwSciCommonTransportParams bufparams;
    LwSciError sciErr = LwSciError_Success;
    size_t transportBufSize = 0;
    void* descBuf = NULL;
    size_t descLen = 0;
    bool ipcPathExists = false;
    LwSciBufAttrList attrList = NULL;

    LWSCI_FNENTRY("");
    if ((NULL == bufObj) || (0U == ipcEndpoint) || (NULL == exportData)) {
        LWSCI_ERR_STR("Invalid input arguments to export LwSciBufObj.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufTransportValidateAccessPerm(accPerm);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("wrong parameter LwSciBufAttrValAccessPerm supplied.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufObjGetAttrList(bufObj, &attrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to get reconciled attribute list from LwSciBufObj");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufTransportCheckIpcPath(attrList, ipcEndpoint,
                &ipcPathExists);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufTransportCheckIpcPath failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == ipcPathExists) {
        LWSCI_ERR_STR("When IPC is ilwolved, objects must flow in the reverse IPC path (ie. originate from the allocator application) of unreconciled LwSciBufAttrLists.");
        sciErr = LwSciError_NotPermitted;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    (void)memset(exportData, 0, sizeof(*exportData));
    bufparams.msgVersion = LW_SCI_BUF_VERSION;
    bufparams.msgMagic = LW_SCI_BUF_TRANSPORT_MAGIC;
    bufparams.keyCount = 1;
    transportBufSize = sizeof(LwSciBufObjExportDescPriv);
    sciErr = LwSciCommonTransportAllocTxBufferForKeys(bufparams, transportBufSize,
                                                     &txBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to allocate Object transport descriptor.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufPrepareObjForTransport(bufObj,
                            ipcEndpoint, accPerm, txBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to prepare lwscibuf object desc.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_txBuf;
    }

    LwSciCommonTransportPrepareBufferForTx(txBuf, &descBuf, &descLen);

    LwSciCommonMemcpyS(exportData, sizeof(*exportData), descBuf, descLen);

    LWSCI_INFO("Outputs: descBuf: %p descLen: %lu\n", descBuf, descLen);

    LwSciCommonFree(descBuf);

free_txBuf:
    LwSciCommonTransportBufferFree(txBuf);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListIpcExportUnreconciled(
    const LwSciBufAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciCommonTransportBuf* txBuf = NULL;
    LwSciCommonTransportParams bufparams;
    LwSciBufTransportKeyDesc keyDescriptor;
    LwSciBufTransportKeyDesc keyDescriptor2;
    LwSciBufTransportKeyDesc keyDescriptor3;
    size_t listno = 0UL;
    size_t slotIndex = 0UL;

    uint32_t totalKeyValPairs = 0U;
    size_t totalSlotCount = 0UL;
    size_t totalValsSize = 0UL;

    uint32_t numKeyValPairs = 0U;
    size_t slotCount = 0UL;
    size_t attrListValsSize = 0UL;
    size_t unreconciledAttrListsValsSize = 0UL;

    bool appendListHeader = true;

    LWSCI_FNENTRY("");
    if ((NULL == unreconciledAttrListArray) || (0U == unreconciledAttrListCount) ||
        (0U == ipcEndpoint) || (NULL == descBuf) || (NULL == descLen)) {
        LWSCI_ERR_STR("Invalid input arguments to export unreconciled list.");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWSCI_INFO("UnrenconciledListArr %p, Count %"PRIu64" IpcEndpoint %"PRIu64
               " descBuf: %p descLen: %p", unreconciledAttrListArray,
               unreconciledAttrListCount, ipcEndpoint, descBuf, descLen);

    sciErr = LwSciBufValidateAttrListArray(unreconciledAttrListArray,
            unreconciledAttrListCount);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to validate attribute list array");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* UpdateBeforeExport aquires implicit lock to attr list when Getting and Setting
     *  attributes. Hence, keeping this interaction outside lock.
     */
    for (listno = 0UL; listno < unreconciledAttrListCount; listno++) {
        sciErr = LwSciBufAttrListUpdateBeforeExport(
                            unreconciledAttrListArray[listno], ipcEndpoint);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_ULONG("LwSciBufAttrListUpdateBeforeExport failed for attribute list:", listno);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto ret;
        }
    }

    sciErr = LwSciBufAttrListsLock(unreconciledAttrListArray,
            unreconciledAttrListCount);
    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciBuf-ADV-MISRAC2012-016")
    for (listno = 0UL; listno < unreconciledAttrListCount; listno++) {
        sciErr = LwSciBufAttrListCompareReconcileStatus(
                   unreconciledAttrListArray[listno], false);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_ULONG("Attribute list is not valid: ", listno);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto unlock_attr_lists;
        }

        sciErr = LwSciBufGetAttrListSlotKeyCountAndSize(
                unreconciledAttrListArray[listno], ipcEndpoint,
                &numKeyValPairs, &slotCount, &attrListValsSize, true);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_STR("Unable to get key count and size for attribute list.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto unlock_attr_lists;
        }

        totalKeyValPairs += numKeyValPairs;
        totalSlotCount += slotCount;
        unreconciledAttrListsValsSize += attrListValsSize;
    }

    bufparams.msgVersion = LW_SCI_BUF_VERSION;
    bufparams.msgMagic = LW_SCI_BUF_TRANSPORT_MAGIC;
    /**
     * We are exporting keys, which come from:
     *
     * 1. Header
     * 2. Keys for each key-value pair
     * 3. AttrListSlotIndex for each Slot Attribute List
     */

    sciErr = LwSciBufNumKeysSerialized(LW_SCI_BUF_NUM_ATTRLIST_HEADER_TRANSPORT_KEYS, totalKeyValPairs, totalSlotCount,
                                       0U, &bufparams.keyCount);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attr_lists;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    // Header: Add LwSciBufTransportAttrKey_AttrListSlotCount value size
    keyDescriptor = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotCount)];

    // Header: Add LwSciBufTransportAttrKey_AttrListReconciledFlag value size
    keyDescriptor2 = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListReconciledFlag)];

    // Add the size of the exported AttrListSlotIndex key values
    keyDescriptor3 = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotIndex)];
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    sciErr = LwSciBufTotalKeyValSize(keyDescriptor.keysize,
                                     keyDescriptor2.keysize,
                                     unreconciledAttrListsValsSize,
                                     keyDescriptor3.keysize,
                                     totalSlotCount,
                                     &totalValsSize);

    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attr_lists;
    }
    sciErr = LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalValsSize,
                                                     &txBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to allocate tx transport descriptor.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto unlock_attr_lists;
    }

    slotIndex = 0;
    for (listno = 0; listno < unreconciledAttrListCount; listno++) {
        sciErr = LwSciBufPrepareAttrlistForTransport(
                unreconciledAttrListArray[listno], ipcEndpoint,
                (uint64_t)totalSlotCount, (uint64_t)slotIndex,
                LwSciBufAccessPerm_Auto, txBuf, appendListHeader, true);
        if (LwSciError_Success != sciErr) {
            LWSCI_ERR_ULONG("Failed to prepare attrList in the array:", listno);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_txBuf;
        }
        appendListHeader = false;

        slotIndex += LwSciBufAttrListGetSlotCount(
                            unreconciledAttrListArray[listno]);
    }

    LwSciCommonTransportPrepareBufferForTx(txBuf, descBuf, descLen);

    LWSCI_INFO("Outputs: descBuf: %p descLen: %"PRIu64"", *descBuf, *descLen);

free_txBuf:
    LwSciCommonTransportBufferFree(txBuf);

unlock_attr_lists:
    {
        LwSciError error = LwSciError_Success;

        error = LwSciBufAttrListsUnlock(unreconciledAttrListArray,
                unreconciledAttrListCount);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Could not unlock Attribute Lists");
            LwSciCommonPanic();
        }
    }

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListIpcExportReconciled(
    LwSciBufAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciCommonTransportBuf* txBuf = NULL;
    LwSciCommonTransportParams bufparams;
    LwSciBufTransportKeyDesc keyDescriptor;
    LwSciBufTransportKeyDesc keyDescriptor2;
    LwSciBufTransportKeyDesc keyDescriptor3;
    size_t totalValsSize = 0U;

    uint32_t totalKeyValPairs = 0U;
    size_t totalSlotCount = 0U;
    size_t attrListValsSize = 0U;
    bool ipcPathExists = false;

    LWSCI_FNENTRY("");

    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Reconciled AttrList is not valid.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((0U == ipcEndpoint) || (NULL == descBuf) || (NULL == descLen)) {
        LWSCI_ERR_STR("Invalid input arguments to export reconciled list.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufTransportCheckIpcPath(reconciledAttrList, ipcEndpoint,
                &ipcPathExists);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufTransportCheckIpcPath failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (false == ipcPathExists) {
        LWSCI_ERR_STR("When IPC is ilwolved, reconciled LwSciBufAttrLists must flow in the reverse IPC path (ie. originate from the allocator application) of unreconciled LwSciBufAttrLists.");
        sciErr = LwSciError_NotPermitted;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufGetAttrListSlotKeyCountAndSize(reconciledAttrList,
            ipcEndpoint, &totalKeyValPairs, &totalSlotCount, &attrListValsSize,
            false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to get key count and size for attribute list.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    bufparams.msgVersion = LW_SCI_BUF_VERSION;
    bufparams.msgMagic = LW_SCI_BUF_TRANSPORT_MAGIC;
    /**
     * We are exporting keys, which come from:
     *
     * 1. Header
     * 2. Keys for each key-value pair
     * 3. AttrListSlotIndex for each Slot Attribute List
     */

    sciErr = LwSciBufNumKeysSerialized(LW_SCI_BUF_NUM_ATTRLIST_HEADER_TRANSPORT_KEYS, totalKeyValPairs, totalSlotCount,
                                       0U, &bufparams.keyCount);

    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciBuf-ADV-MISRAC2012-008")
    // Header: Add LwSciBufTransportAttrKey_AttrListSlotCount value size
    keyDescriptor = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotCount)];

    // Header: Add LwSciBufTransportAttrKey_AttrListReconciledFlag value size
    keyDescriptor2 = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListReconciledFlag)];

    // Add the size of the exported AttrListSlotIndex key values
    keyDescriptor3 = LwSciBufTransportKeysDescTable[
            LW_SCI_BUF_TRANSKEY_IDX(LwSciBufTransportAttrKey_AttrListSlotIndex)];
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    sciErr = LwSciBufTotalKeyValSize(keyDescriptor.keysize,
                                     keyDescriptor2.keysize,
                                     attrListValsSize,
                                     keyDescriptor3.keysize,
                                     totalSlotCount,
                                     &totalValsSize);

    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Buffer overflow\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalValsSize,
                                                     &txBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to allocate tx transport descriptor.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufPrepareAttrlistForTransport(reconciledAttrList,
                    ipcEndpoint,
                    LwSciBufAttrListGetSlotCount(reconciledAttrList),
                    0, LwSciBufAccessPerm_Auto, txBuf, true, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to prepare the attrList\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_txBuf;
    }

    LwSciCommonTransportPrepareBufferForTx(txBuf, descBuf, descLen);

    LWSCI_INFO("Outputs: descBuf: %p descLen: %p", *descBuf, *descLen);

free_txBuf:
    LwSciCommonTransportBufferFree(txBuf);

ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

/****************************************************
 * Public Functions - Import functionality
 ****************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufObjIpcImport(
    LwSciIpcEndpoint ipcEndpoint,
    const LwSciBufObjIpcExportDescriptor* desc,
    LwSciBufAttrList reconciledAttrList,
    LwSciBufAttrValAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciBufObj* bufObj)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciCommonTransportBuf* rxBuf;

    (void)timeoutUs;

    LWSCI_FNENTRY("");
    sciErr = LwSciBufAttrListCompareReconcileStatus(reconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to import the object.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if ((0U == ipcEndpoint) || (NULL == desc) || (NULL == bufObj)) {
        LWSCI_ERR_STR("Invalid input arguments to import object.");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufTransportValidateAccessPerm(minPermissions);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("wrong parameter LwSciBufAttrValAccessPerm supplied.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufImportBuffer(desc, sizeof(LwSciBufObjIpcExportDescriptor),
                         &rxBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to import the Received descriptor.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufImportBufObjDesc(rxBuf, ipcEndpoint, &reconciledAttrList,
        minPermissions, bufObj);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to import LwSciBuf object.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

buffer_free:
    LwSciCommonTransportBufferFree(rxBuf);
ret:
    LWSCI_FNEXIT("");
    return (sciErr);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListIpcImportReconciled(
    LwSciBufModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const LwSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciBufAttrList* importedReconciledAttrList)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciCommonTransportBuf* rxBuf;
    bool isListValid = false;
    bool requireValidation = (inputUnreconciledAttrListArray != NULL);

    LWSCI_FNENTRY("");

    if ((0U == ipcEndpoint) || (NULL == descBuf) || (0U == descLen) ||
        (NULL == module) || (NULL == importedReconciledAttrList)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Invalid input arguments to import reconciled attrList.\n");
        LWSCI_ERR_ULONG("ipcEndpoing \n", ipcEndpoint);
        LWSCI_ERR_ULONG("descLen \n", descLen);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *importedReconciledAttrList = NULL;

    sciErr = LwSciBufImportBuffer(descBuf, descLen, &rxBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to lock all Attribute Lists\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufImportAttrListDesc(rxBuf, ipcEndpoint,
        importedReconciledAttrList, module, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to import LwSciBuf attribute.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

    sciErr = LwSciBufAttrListCompareReconcileStatus(*importedReconciledAttrList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to import the reconciled.\n");
        LwSciBufAttrListFree(*importedReconciledAttrList);
        *importedReconciledAttrList = NULL;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

    sciErr = RecallwlatePlatformAttributesForReconciledImport(
                *importedReconciledAttrList, ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("RecallwlatePlatformAttributesForReconciledImport() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

    sciErr = CheckImportedReconciledListConsistency(
                *importedReconciledAttrList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("CheckImportedReconciledListConsistency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

    if (false == requireValidation) {
        LWSCI_INFO("Validation skipped due to NULL unreconciled list.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

    sciErr = LwSciBufAttrListValidateReconciled(
                 *importedReconciledAttrList, inputUnreconciledAttrListArray,
                 inputUnreconciledAttrListCount, &isListValid);

    if ((LwSciError_Success != sciErr) || (false == isListValid)) {
        LWSCI_ERR_STR("Reconciled list doesn't match input unreconciled lists\n.");
        LwSciBufAttrListFree(*importedReconciledAttrList);
        *importedReconciledAttrList = NULL;
        sciErr = LwSciError_AttrListValidationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

buffer_free:
    LwSciCommonTransportBufferFree(rxBuf);

ret:
    if ((LwSciError_Success != sciErr) &&
        (NULL != importedReconciledAttrList)) {
        LwSciBufAttrListFree(*importedReconciledAttrList);
    }
    LWSCI_FNEXIT("");
    return (sciErr);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufAttrListIpcImportUnreconciled(
    LwSciBufModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    LwSciBufAttrList* importedUnreconciledAttrList)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciCommonTransportBuf* rxBuf;

    LWSCI_FNENTRY("");
    if ((0U == ipcEndpoint) || (NULL == descBuf) || (0U == descLen) ||
        (NULL == module) || (NULL == importedUnreconciledAttrList)) {
        LWSCI_ERR_STR("Invalid input arguments to import reconciled attrList.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *importedUnreconciledAttrList = NULL;

    sciErr = LwSciBufImportBuffer(descBuf, descLen, &rxBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to import the Received descriptor.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufImportAttrListDesc(rxBuf, ipcEndpoint,
        importedUnreconciledAttrList, module, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to import unreconciled LwSciBufAttrList.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

    sciErr = LwSciBufAttrListCompareReconcileStatus(*importedUnreconciledAttrList, false);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to import the unreconciled.");
        LwSciBufAttrListFree(*importedUnreconciledAttrList);
        *importedUnreconciledAttrList = NULL;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto buffer_free;
    }

buffer_free:
    LwSciCommonTransportBufferFree(rxBuf);
ret:
   if ((LwSciError_Success != sciErr) && (NULL != importedUnreconciledAttrList)) {
        if (NULL != *importedUnreconciledAttrList) {
            LwSciBufAttrListFree(*importedUnreconciledAttrList);
        }
    }
    LWSCI_FNEXIT("");
    return (sciErr);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufIpcImportAttrListAndObj(
    LwSciBufModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* attrListAndObjDesc,
    size_t attrListAndObjDescSize,
    const LwSciBufAttrList attrList[],
    size_t count,
    LwSciBufAttrValAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciBufObj* bufObj)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciCommonTransportBuf* rxBuf = NULL;
    LwSciBufAttrList importedReconciledList = NULL;
    bool isListValid = false;
    bool requireValidation = (attrList != NULL);

    (void)timeoutUs;

    LWSCI_FNENTRY("");
    if ((0U == ipcEndpoint) || (NULL == attrListAndObjDesc) ||
        (0U == attrListAndObjDescSize) || (NULL == module) || (NULL == bufObj)) {
        LWSCI_ERR_STR("Invalid input arguments to import reconciled attrList.\n");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufTransportValidateAccessPerm(minPermissions);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("wrong parameter LwSciBufAttrValAccessPerm supplied.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("IpcEndpoint validation failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *bufObj = NULL;

    sciErr = LwSciBufImportBuffer(attrListAndObjDesc, attrListAndObjDescSize,
                                  &rxBuf);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to import the Received descriptor.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufImportCombinedDesc(rxBuf, ipcEndpoint,
        &importedReconciledList, module, minPermissions, bufObj);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Unable to import reconciled LwSciBufAttrList and LwSciBufObj.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fail_free;
    }

    sciErr = LwSciBufAttrListCompareReconcileStatus(importedReconciledList, true);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Invalid attribute list to import the unreconciled.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fail_free;
    }

    sciErr = RecallwlatePlatformAttributesForReconciledImport(
                importedReconciledList, ipcEndpoint);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("RecallwlatePlatformAttributesForReconciledImport() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fail_free;
    }

    sciErr = CheckImportedReconciledListConsistency(importedReconciledList);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("CheckImportedReconciledListConsistency() failed.");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fail_free;
    }

    if (false == requireValidation) {
        LWSCI_INFO("Skip validation as user passed list is NULL.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufAttrListValidateReconciled(
            importedReconciledList, attrList, count,  &isListValid);

    if ((LwSciError_Success != sciErr) || (false == isListValid)) {
        LWSCI_ERR_STR("Reconciled list doesn't match input unreconciled lists\n.");
        sciErr = LwSciError_AttrListValidationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto fail_free;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

fail_free:
    if (NULL != *bufObj) {
        LwSciBufObjFree(*bufObj);
    }
    *bufObj = NULL;

ret:
    /* We have cloned the Attribute List in LwSciBufTransportAttrKey_ObjDesc's
     * callback, so we can free our local reference. */
    LwSciBufAttrListFree(importedReconciledList);
    LwSciCommonTransportBufferFree(rxBuf);

    LWSCI_FNEXIT("");
    return (sciErr);
}


/****************************************************
 * Public Functions - Free functions
 ****************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
void LwSciBufAttrListAndObjFreeDesc(
    void* attrListAndObjDescBuf)
{
    LWSCI_FNENTRY("");
    LwSciCommonFree(attrListAndObjDescBuf);
    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
void LwSciBufAttrListFreeDesc(
    void* descBuf)
{
    LWSCI_FNENTRY("");
    LwSciCommonFree(descBuf);
    LWSCI_FNEXIT("");
}

/****************************************************
 * Public Functions - Debug function
 ****************************************************/

#if (LW_IS_SAFETY == 0)
LwSciError LwSciBufAttrListDebugDump(LwSciBufAttrList attrList, void** buf,
    size_t* len)
{
    LwSciError err = LwSciError_Success;
    LWSCI_FNENTRY("");
    err = LwSciBufAttrListIpcExportUnreconciled(&attrList, 1U, 0U, buf, len);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to create debug dump\n");
    }

    LWSCI_FNEXIT("");
    return err;
}
#endif
