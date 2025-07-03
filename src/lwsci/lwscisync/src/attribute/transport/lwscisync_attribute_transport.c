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
 * \brief <b>LwSciSync Attribute Export/Import Implementation</b>
 *
 * @b Description: This file implements LwSciSync attribute list transport APIs
 *
 */
#include "lwscisync_attribute_transport.h"

#include "lwscicommon_covanalysis.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_objref.h"
#include "lwscicommon_os.h"
#include "lwscicommon_transportutils.h"
#include "lwscicommon_utils.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core.h"
#include "lwscisync_attribute_core_cluster.h"
#include "lwscisync_attribute_reconcile_priv.h"
#include "lwscisync_attribute_transport_semaphore.h"
#include "lwscisync_core.h"
#include "lwscisync_module.h"

/**
 * Magic ID used to validate export descriptor received over IPC
 */
#define LW_SCI_SYNC_TRANSPORT_ATTR_MAGIC  (0xF00ABD57U)

/**
 * \brief Types of LwSciSyncCoreAttrList Keys for Exporting
 */
typedef enum {
    /** For LwSciSync internal use only */
    LwSciSyncCoreAttrKey_LowerBound = (1 << 20),
    /** (size_t) */
    LwSciSyncCoreAttrKey_NumCoreAttrList,
    /** (uint64_t) */
    LwSciSyncCoreAttrKey_MaxPrimitiveType,
    /** (LwSciSyncCoreAttrList) */
    LwSciSyncCoreAttrKey_CoreAttrList,
    /** (LwSciSyncCoreAttrListState) */
    LwSciSyncCoreAttrKey_AttrListState,
    /** (LwSciSyncCoreIpcTable) */
    LwSciSyncCoreAttrKey_IpcTable,
    /** (size_t) */
    LwSciSyncCoreAttrKey_AttrListTypeMask,
    /** (LwSciBufAttrList for semaphore) */
    LwSciSyncCoreAttrKey_SemaAttrList,
    /** (LwSciBufAttrList for timestamps) */
    LwSciSyncCoreAttrKey_TimestampsAttrList,
    /** ipcEndpoint identifier for fences -
     *  the ipcEndpoint unreconciled list was exported through;
     *  only exported if C2C */
    LwSciSyncCoreAttrKey_ExportIpcEndpoint,
    /** For LwSciSync internal use only */
    LwSciSyncCoreAttrKey_UpperBound,
} LwSciSyncCoreAttrKey;

#define CORE_ATTR_KEY_TO_IDX(key) \
    ((uint64_t)(key) -                                      \
     (uint64_t)LwSciSyncCoreAttrKey_LowerBound - 1U)

#define IDX_TO_CORE_ATTR_KEY(idx) \
    ((idx) + (uint64_t)LwSciSyncCoreAttrKey_LowerBound + 1U)

/** Callwlate key count and value size */
static LwSciError GetKeyCntAndValSize(
    const LwSciSyncCoreAttrs* attrs,
    uint32_t* keyCnt,
    uint64_t* valSize);

/** Callwlate private key count and value size */
static LwSciError GetPrivateKeyCntAndValSize(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    uint32_t* keyCnt,
    uint64_t* valSize);

/** Serialize attributes */
static LwSciError ExportAttributes(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciCommonTransportBuf* txbuf);

/** Serialize private keys */
static LwSciError ExportPrivateKeys(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciCommonTransportBuf* txbuf);

/** Serialize obj attr list */
static LwSciError ExportObjAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList,
    size_t attrListValueSize,
    void** coreAttrListTxbufPtr,
    const size_t* coreAttrListBufSize,
    void** attrListTxbufPtr,
    size_t* attrListBufSize);

/** Serialize core attr list */
static LwSciError ExportCoreAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    void** coreAttrListTxbufPtr,
    size_t* coreAttrListBufSize);

/** Serialize attr list to desc */
static LwSciError ExportAttrList(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/** Deserialize attribute */
static LwSciError ImportAttribute(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncInternalAttrValPrimitiveType primitiveUpperBound,
    uint32_t inputKey,
    const void* value,
    size_t length);

/** Deserialize private keys */
static LwSciError ImportPrivateKeys(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    uint32_t inputKey,
    const void* value,
    size_t length,
    bool importReconciled);

/** Deserialize core attr list */
static LwSciError ImportCoreAttrList(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    LwSciSyncInternalAttrValPrimitiveType primitiveUpperBound,
    const void* inputValue,
    size_t length,
    bool importReconciled);

/** Deserialize obj attr list */
static LwSciError ImportObjAttrList(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    const void* attrListRxbufPtr,
    size_t attrListBufSize,
    bool importReconciled,
    LwSciSyncAttrList* attrList);

/** Deserialize attr list desc */
static LwSciError ImportAttrList(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    const void* descBuf,
    size_t descLen,
    bool importReconciled,
    LwSciSyncAttrList* attrList);

/** Skip incompatible primitives */
static LwSciError ImportPrimitiveType(
    LwSciSyncInternalAttrValPrimitiveType primitiveUpperBound,
    const void* value,
    size_t length,
    LwSciSyncInternalAttrValPrimitiveType* outputPrimitiveType);

/** Export the timestmaps buf attr list */
static LwSciError ExportTimestampsAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    void** txbufPtr,
    size_t* txbufSize);

/** Import the timestamp buf attr list */
static LwSciError ImportTimestampsAttrList(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    bool importReconciled,
    const void* inputValue,
    size_t length);

/** Check input args for LwSciSyncAttrListIpcExportReconciled */
static LwSciError AttrListIpcExportReconciledCheckArgs(
    const LwSciSyncAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    const size_t* descLen);

/** Check input args for LwSciSyncAttrListIpcImportReconciled and
 *  LwSciSyncAttrListIpcImportUnreconciled
 */
static LwSciError AttrListIpcImportCheckArgs(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const LwSciSyncAttrList* importedAttrList);

/** Export private key */
static LwSciError ExportKey(
    LwSciCommonTransportBuf* txbuf,
    uint32_t key,
    const void* value,
    size_t len);

/** Prepares a blob for tx from the provided subblob */
static LwSciError TransportCreateBufferWithSingleKey(
    LwSciCommonTransportParams bufParams,
    uint32_t key,
    const void* inputPtr,
    size_t inputSize,
    void** descBufPtr,
    size_t* descBufSize);

/******************************************************
 *            Public interfaces definition
 ******************************************************/

#if (LW_IS_SAFETY == 0)
LwSciError LwSciSyncAttrListDebugDump(
    LwSciSyncAttrList attrList,
    void** buf,
    size_t* len)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (NULL == buf) {
        LWSCI_ERR_STR("Invalid argument: buf: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == len) {
        LWSCI_ERR_STR("Invalid argument: len: NULL pointer\n");
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
    LWSCI_INFO("buf: %p\n", buf);
    LWSCI_INFO("len: %p\n", len);

    /* Serialize attr list in desc blob */
    error = ExportAttrList(attrList, 0U, buf, len);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("*buf: %p\n", *buf);
    LWSCI_INFO("*len: %zu\n", *len);

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}
#endif

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListIpcExportUnreconciled(
    const LwSciSyncAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList newUnreconciledAttrList = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    bool hasC2C = false;

    LWSCI_FNENTRY("");

    if (NULL == descBuf) {
        LWSCI_ERR_STR("Invalid argument: descBuf: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == descLen) {
        LWSCI_ERR_STR("Invalid argument: descLen: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != error) {
        LWSCI_ERR_ULONG("Invalid LwSciIpcEndpoint", ipcEndpoint);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("descBuf: %p\n", descBuf);
    LWSCI_INFO("descLen: %p\n", descLen);
    LWSCI_INFO("ipcEndpoint: %" PRIu64 "\n", ipcEndpoint);

    *descBuf = NULL;
    *descLen = 0U;

    error = LwSciSyncAttrListAppendUnreconciled(
            unreconciledAttrListArray, unreconciledAttrListCount,
            &newUnreconciledAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncCoreAttrListGetObjFromRef(newUnreconciledAttrList,
            &objAttrList);

    error = LwSciSyncCoreIsIpcEndpointC2c(
            ipcEndpoint,
            &hasC2C);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreFillCpuPrimitiveInfo(objAttrList, hasC2C);

    error = LwSciSyncCoreFillSemaAttrList(objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_attr_list;
    }

    error = LwSciSyncCoreFillTimestampBufAttrList(objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_attr_list;
    }

    error = ExportAttrList(newUnreconciledAttrList, ipcEndpoint, descBuf,
            descLen);
    if (LwSciError_Success != error) {
        /* Internal overflow is considered ResourceError to the client */
        if (LwSciError_Overflow == error) {
            error = LwSciError_ResourceError;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_attr_list;
    }

    LWSCI_INFO("*descBuf: %p\n", *descBuf);
    LWSCI_INFO("*descLen: %zu\n", *descLen);

free_attr_list:
    LwSciSyncAttrListFree(newUnreconciledAttrList);

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListIpcExportReconciled(
    const LwSciSyncAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList clonedReconciledAttrList = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciSyncCoreAttrList* coreAttrList = NULL;

    size_t engineArrayLen = 0U;

    LWSCI_FNENTRY("");

    error = AttrListIpcExportReconciledCheckArgs(reconciledAttrList, ipcEndpoint,
        descBuf, descLen);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Invalid arguments\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("reconciledAttrList: %p\n", reconciledAttrList);
    LWSCI_INFO("descBuf: %p\n", descBuf);
    LWSCI_INFO("descLen: %p\n", descLen);
    LWSCI_INFO("ipcEndpoint: %" PRIu64 "\n", ipcEndpoint);

    *descBuf = NULL;
    *descLen = 0U;

    error = LwSciSyncAttrListClone(reconciledAttrList,
            &clonedReconciledAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreAttrListGetObjFromRef(clonedReconciledAttrList,
            &objAttrList);

    /* Fetch the attr list type from the Ipc perm table */
    coreAttrList = objAttrList->coreAttrList;
    LwSciSyncCoreIpcTableLwtSubTree(&coreAttrList->ipcTable, ipcEndpoint,
            &coreAttrList->attrs.needCpuAccess,
            &coreAttrList->attrs.waiterRequireTimestamps,
            &coreAttrList->attrs.actualPerm,
            coreAttrList->attrs.engineArray,
            sizeof(coreAttrList->attrs.engineArray) / sizeof(coreAttrList->attrs.engineArray[0]),
            &engineArrayLen);

    /* Update attribute list being exported to reflect changes to EngineArray
     * key based on the state of the IPC table. We need to update sizes here
     * to ensure that the attribute key is exported/not-exported as required. */
    coreAttrList->attrs.valSize[LwSciSyncCoreKeyToIndex(LwSciSyncInternalAttrKey_EngineArray)] =
        engineArrayLen * sizeof(coreAttrList->attrs.engineArray[0]);

    error = ExportAttrList(clonedReconciledAttrList, ipcEndpoint, descBuf,
            descLen);
    if (LwSciError_Success != error) {
        /* Internal overflow is considered ResourceError to the client */
        if (LwSciError_Overflow == error) {
            error = LwSciError_ResourceError;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("*descBuf: %p\n", *descBuf);
    LWSCI_INFO("*descLen: %zu\n", *descLen);

fn_exit:
    LwSciSyncAttrListFree(clonedReconciledAttrList);

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListIpcImportUnreconciled(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    LwSciSyncAttrList* importedUnreconciledAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciSyncCoreAttrList* coreAttrList = NULL;
    size_t i = 0U;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = AttrListIpcImportCheckArgs(module, ipcEndpoint,
        descBuf, descLen, importedUnreconciledAttrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Invalid arguments");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("module: %p", module);
    LWSCI_INFO("importedUnreconciledAttrList: %p", importedUnreconciledAttrList);
    LWSCI_INFO("descBuf: %p", descBuf);
    LWSCI_INFO("descLen: %zu", descLen);
    LWSCI_INFO("ipcEndpoint: %" PRIu64, ipcEndpoint);

    error = ImportAttrList(module, ipcEndpoint, NULL, descBuf, descLen,
            false, importedUnreconciledAttrList);
    if (LwSciError_Success != error) {
        /* Internal overflow is considered ResourceError to the client */
        if (LwSciError_Overflow == error) {
            error = LwSciError_ResourceError;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncCoreAttrListGetObjFromRef(*importedUnreconciledAttrList,
        &objAttrList);

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        coreAttrList = &objAttrList->coreAttrList[i];
        error = LwSciSyncCoreIpcTableAppend(&coreAttrList->ipcTable,
                ipcEndpoint);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_imported_list;
        }
    }
    objAttrList->writable = false;
    LWSCI_INFO("*importedUnreconciledAttrList: %p", *importedUnreconciledAttrList);

free_imported_list:
    if (LwSciError_Success != error) {
        LwSciSyncAttrListFree(*importedUnreconciledAttrList);
        *importedUnreconciledAttrList = NULL;
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListIpcImportReconciled(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncAttrList* importedReconciledAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* newObjAttrList = NULL;
    bool isReconciledListValid = false;
    LwSciSyncAttrList newUnreconciledAttrList = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    bool hasC2C = false;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = AttrListIpcImportCheckArgs(module, ipcEndpoint,
        descBuf, descLen, importedReconciledAttrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Invalid arguments\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("module: %p\n", module);
    LWSCI_INFO("importedReconciledAttrList: %p\n", importedReconciledAttrList);
    LWSCI_INFO("descBuf: %p\n", descBuf);
    LWSCI_INFO("descLen: %zu\n", descLen);
    LWSCI_INFO("ipcEndpoint: %" PRIu64 "\n", ipcEndpoint);

    /* We need to lock all the Attribute Lists here in order to operate on the
     * Attribute Lists atomically. We can't lock/unlock and then lock/unlock
     * again when needed, since the Attribute Lists could have been modified
     * between calls to LwSciSyncAttrListAppend and
     * LwSciSyncAttrListValidateReconciled.
     */
    error = LwSciSyncCoreAttrListsLock(inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (0U != inputUnreconciledAttrListCount) {
        error = LwSciSyncCoreAttrListAppendUnreconciledWithLocks(
                inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
                false, &newUnreconciledAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto unlock_attr_lists;
        }

        LwSciSyncCoreAttrListGetObjFromRef(newUnreconciledAttrList,
                &objAttrList);

        error = LwSciSyncCoreIsIpcEndpointC2c(
                ipcEndpoint,
                &hasC2C);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        LwSciSyncCoreFillCpuPrimitiveInfo(objAttrList, hasC2C);

        /* Update sema attrs in the input attr lists */
        error = LwSciSyncCoreFillSemaAttrList(objAttrList);
        if (error !=LwSciError_Success) {
            goto free_attr_list;
        }

        /*Update timestamp buf attrs in the input attr lists */
        error = LwSciSyncCoreFillTimestampBufAttrList(objAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_attr_list;
        }
    }

    error = ImportAttrList(module, ipcEndpoint, newUnreconciledAttrList,
            descBuf, descLen, true, importedReconciledAttrList);
    if (LwSciError_Success != error) {
        /* Internal overflow is considered ResourceError to the client */
        if (LwSciError_Overflow == error) {
            error = LwSciError_ResourceError;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_attr_list;
    }
    LwSciSyncCoreAttrListGetObjFromRef(*importedReconciledAttrList,
            &newObjAttrList);

    /* Verify the reconciled list against input unreconciled attr lists */
    error = LwSciSyncCoreAttrListValidateReconciledWithLocks(
            *importedReconciledAttrList, inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount, false, &isReconciledListValid);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_new_attr_list;
    }
    if (false == isReconciledListValid) {
        error = LwSciError_AttrListValidationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_new_attr_list;
    }

    /* Set state and endpoint info */
    if (newObjAttrList->state != LwSciSyncCoreAttrListState_Reconciled) {
        LWSCI_ERR_STR("Invalid Reconciled attr list desc\n");
        error = LwSciError_AttrListValidationFailed;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_new_attr_list;
    }
    newObjAttrList->writable = false;

    LWSCI_INFO("*importedReconciledAttrList: %p\n",
            *importedReconciledAttrList);

free_new_attr_list:
    if (LwSciError_Success != error) {
        LwSciSyncAttrListFree(*importedReconciledAttrList);
    }

free_attr_list:
    LwSciSyncAttrListFree(newUnreconciledAttrList);

unlock_attr_lists:
    {
        LwSciError err = LwSciError_Success;

        err = LwSciSyncCoreAttrListsUnlock(inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Could not unlock Attribute Lists\n");
            LwSciCommonPanic();
        }
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncAttrListFreeDesc(void *descBuf)
{
    LWSCI_FNENTRY("");

    if (NULL == descBuf) {
        LWSCI_ERR_STR("Invalid argument: descBuf: NULL pointer\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonFree(descBuf);

fn_exit:

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCoreAttrListGetIpcExportPerm(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAccessPerm* actualPerm)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    error = LwSciSyncCoreIpcTableGetPermAtSubTree(
            &objAttrList->coreAttrList->ipcTable,
            ipcEndpoint, actualPerm);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

void LwSciSyncCoreAttrListGetIpcExportRequireTimestamps(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    bool* waiterRequireTimestamps)
{
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    LwSciSyncCoreIpcTableGetRequireTimestampsAtSubTree(
            &objAttrList->coreAttrList->ipcTable,
            ipcEndpoint, waiterRequireTimestamps);
}

static LwSciError ExportAttributes(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciCommonTransportBuf* txbuf)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t len = 0U;
    const void* value = NULL;
    uint32_t key = 0U;

    LWSCI_FNENTRY("");

    for (i = 0U; i < KEYS_COUNT; i++) {
        const LwSciSyncCoreAttrs* attrs = &coreAttrList->attrs;
        if (0U != attrs->valSize[i]) {
            len = attrs->valSize[i];
            value = LwSciSyncCoreAttrListGetConstValForKey(coreAttrList, i);
            key = LwSciSyncCoreIndexToKey(i);
            error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len, value);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }
    }
fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ExportTimestampsAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    void** txbufPtr,
    size_t* txbufSize)
{
    LwSciError error = LwSciError_Success;

    /** Return if nothing to export */
    if (NULL == coreAttrList->timestampBufAttrList) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** WAR to ilwoke debug dump function */
    if (0U == ipcEndpoint) {
#if (LW_IS_SAFETY == 0)
        error = LwSciBufAttrListDebugDump(coreAttrList->timestampBufAttrList,
                txbufPtr, txbufSize);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Debug Dump Failed.\n");
        }
#endif
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Ilwoke appropriate export function based on timestamp attr list state */
    if (LwSciSyncCoreAttrListState_Unreconciled == state) {
        error = LwSciBufAttrListIpcExportUnreconciled(
                &coreAttrList->timestampBufAttrList, 1U, ipcEndpoint, txbufPtr,
                txbufSize);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if (LwSciSyncCoreAttrListState_Reconciled == state) {
        bool waiterRequireTimestamps = false;
        LwSciSyncCoreIpcTableGetRequireTimestampsSum(
                &coreAttrList->ipcTable, &waiterRequireTimestamps);
        if (true == waiterRequireTimestamps) {
            error = LwSciBufAttrListIpcExportReconciled(
                    coreAttrList->timestampBufAttrList,
                    ipcEndpoint, txbufPtr, txbufSize);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }
    } else {
        LWSCI_ERR_STR("Exporting invalid buf attr list\n");
    }

fn_exit:
    return error;
}

static LwSciError GetKeyCntAndValSize(
    const LwSciSyncCoreAttrs* attrs,
    uint32_t* keyCnt,
    uint64_t* valSize)
{
    LwSciError error = LwSciError_Success;

    size_t i = 0U;
    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;

    *valSize = 0U;
    *keyCnt = 0U;

    for (i = 0U; i < KEYS_COUNT; i++) {
        if (0U != attrs->valSize[i]) {
            u32Add((*keyCnt), 1U, keyCnt, &addStatus1);
            u64Add((*valSize), (attrs->valSize[i]), valSize, &addStatus2);
            if (OP_SUCCESS != (addStatus1 & addStatus2)) {
                error = LwSciError_Overflow;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }
    }

fn_exit:
    return error;
}

static LwSciError GetPrivateKeyCntAndValSize(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    uint32_t* keyCnt,
    uint64_t* valSize)
{
    LwSciError error = LwSciError_Success;
    void* ipcTableBufPtr = NULL;
    size_t ipcTableBufSize = 0U;
    uint8_t addStatus = OP_FAIL;

    void* tempAttrListDesc = NULL;
    size_t tempAttrListDescSize = 0U;
#if (LW_IS_SAFETY == 0)
    LwSciSyncIpcTopoId syncTopoId = {0};
#endif

#if !defined(__x86_64__)
    (void)state;
    (void)ipcEndpoint;
#endif

    /** Start empty */
    *valSize = 0U;
    *keyCnt = 0U;

    /* additional private keys only exported sometimes */
#if (LW_IS_SAFETY == 0)
    error = LwSciSyncCoreGetSyncTopoId(ipcEndpoint, &syncTopoId);
    if (LwSciError_Success != error) {
        LWSCI_ERR_INT("Something went wrong with LwSciIpc ", error);
        error = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (LwSciSyncCoreIsTopoIdC2c(syncTopoId.topoId)) {
        *keyCnt += 1U;
        *valSize += sizeof(ipcEndpoint);
    }
#endif

    /** Append IPC table key and value size info */
    error = LwSciSyncCoreExportIpcTable(&coreAttrList->ipcTable,
            &ipcTableBufPtr, &ipcTableBufSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonFree(ipcTableBufPtr);
    if (0U < ipcTableBufSize) {
        *keyCnt += 1U;
        u64Add((*valSize), ipcTableBufSize, valSize, &addStatus);
        if (OP_SUCCESS != addStatus) {
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    /** Append sema attr list key and value size info */
    error = ExportSemaAttrList(coreAttrList, state, ipcEndpoint,
            &tempAttrListDesc,
            &tempAttrListDescSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonFree(tempAttrListDesc);
    if (0U < tempAttrListDescSize) {
        *keyCnt += 1U;
        *valSize += tempAttrListDescSize;
    }
    tempAttrListDescSize = 0U;
    tempAttrListDesc = NULL;

    /** Append timestamp buf attr list key and value size info */
    error = ExportTimestampsAttrList(coreAttrList,
            state,
            ipcEndpoint, &tempAttrListDesc,
            &tempAttrListDescSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (0U < tempAttrListDescSize) {
        LwSciCommonFree(tempAttrListDesc);
        *keyCnt += 1U;
        u64Add((*valSize), tempAttrListDescSize, valSize, &addStatus);
        if (OP_SUCCESS != addStatus) {
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    return error;
}

static LwSciError ExportPrivateKeys(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciCommonTransportBuf* txbuf)
{
    LwSciError error = LwSciError_Success;
    void* ipcTableBufPtr = NULL;
    size_t ipcTableBufSize = 0U;
    void* semaAttrListDesc = NULL;
    size_t semaAttrListDescSize = 0U;
    void* timestampsAttrListDesc = NULL;
    size_t timestampsAttrListDescSize = 0U;
#if (LW_IS_SAFETY == 0)
    LwSciSyncIpcTopoId syncTopoId = {0};
#endif

#if !defined(__x86_64__)
    (void)state;
    (void)ipcEndpoint;
#endif

    LWSCI_FNENTRY("");

    /** Export Ipc Table */
    error = LwSciSyncCoreExportIpcTable(&coreAttrList->ipcTable,
            &ipcTableBufPtr, &ipcTableBufSize);
    if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
    }
    error = ExportKey(txbuf,
        (uint32_t)LwSciSyncCoreAttrKey_IpcTable, (const void*)ipcTableBufPtr,
        ipcTableBufSize);
    if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
    }

    /** Export sema attr list */
    error = ExportSemaAttrList(coreAttrList, state, ipcEndpoint,
            &semaAttrListDesc,
            &semaAttrListDescSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = ExportKey(txbuf,
        (uint32_t)LwSciSyncCoreAttrKey_SemaAttrList, (const void*)semaAttrListDesc,
        semaAttrListDescSize);
    if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
    }

    /** Export timestamps buf attr list */
    error = ExportTimestampsAttrList(coreAttrList,
            state,
            ipcEndpoint, &timestampsAttrListDesc,
            &timestampsAttrListDescSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = ExportKey(txbuf,
        (uint32_t)LwSciSyncCoreAttrKey_TimestampsAttrList,
        (const void*)timestampsAttrListDesc,
        timestampsAttrListDescSize);
    if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
    }

#if (LW_IS_SAFETY == 0)
    error = LwSciSyncCoreGetSyncTopoId(ipcEndpoint, &syncTopoId);
    if (LwSciError_Success != error) {
        LWSCI_ERR_INT("Something went wrong with LwSciIpc ", error);
        error = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (LwSciSyncCoreIsTopoIdC2c(syncTopoId.topoId)) {
        error = ExportKey(
            txbuf,
            (uint32_t)LwSciSyncCoreAttrKey_ExportIpcEndpoint,
            &ipcEndpoint,
            sizeof(ipcEndpoint));
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
#endif
fn_exit:
    LwSciCommonFree(semaAttrListDesc);
    LwSciCommonFree(timestampsAttrListDesc);
    LwSciCommonFree(ipcTableBufPtr);

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError ExportObjAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList,
    size_t attrListValueSize,
    void** coreAttrListTxbufPtr,
    const size_t* coreAttrListBufSize,
    void** attrListTxbufPtr,
    size_t* attrListBufSize)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t len = 0U;
    size_t totalSize = 0U;
    const void* value = NULL;
    uint32_t key;
    uint64_t tmpSize = 0UL;
    uint8_t addStatus = OP_FAIL;
    LwSciCommonTransportParams bufparams = {0};
    LwSciCommonTransportBuf* attrListTxbuf = NULL;
    LwSciSyncInternalAttrValPrimitiveType maxPrimitiveType =
            LwSciSyncInternalAttrValPrimitiveType_UpperBound;

    LWSCI_FNENTRY("");

    if (UINT32_MAX < objAttrList->numCoreAttrList) {
      LWSCI_ERR_STR("Number of core Attr list is too big to be exported\n");
      error = LwSciError_Overflow;
      LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
      goto fn_exit;
    }

    /* Callwlate key count and value size for attr list desc */
    bufparams.keyCount = (uint32_t)objAttrList->numCoreAttrList + 3U;
    tmpSize = sizeof(objAttrList->numCoreAttrList) +
               sizeof(LwSciSyncInternalAttrValPrimitiveType) +
               sizeof(LwSciSyncCoreAttrListState);
    u64Add(attrListValueSize, tmpSize, &totalSize, &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("totalSize value is out of range.\n");
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /* Create attr list desc */
    error = LwSciCommonTransportAllocTxBufferForKeys(bufparams,
            totalSize, &attrListTxbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    len = sizeof(objAttrList->numCoreAttrList);
    value = (const void*)&objAttrList->numCoreAttrList;
    key = (uint32_t)LwSciSyncCoreAttrKey_NumCoreAttrList;
    error = LwSciCommonTransportAppendKeyValuePair(attrListTxbuf, key, len,
            value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Export attr list state */
    len = sizeof(objAttrList->state);
    value = (const void*)&objAttrList->state;
    key = (uint32_t)LwSciSyncCoreAttrKey_AttrListState;
    error = LwSciCommonTransportAppendKeyValuePair(attrListTxbuf, key, len, value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    len = sizeof(LwSciSyncInternalAttrValPrimitiveType);
    value = (const void*)&maxPrimitiveType;
    key = (uint32_t)LwSciSyncCoreAttrKey_MaxPrimitiveType;
    error = LwSciCommonTransportAppendKeyValuePair(attrListTxbuf, key, len,
            value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        len = coreAttrListBufSize[i];
        value = (const void*)coreAttrListTxbufPtr[i];
        key = (uint32_t)LwSciSyncCoreAttrKey_CoreAttrList;
        error = LwSciCommonTransportAppendKeyValuePair(attrListTxbuf, key, len,
                value);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    LwSciCommonTransportPrepareBufferForTx(attrListTxbuf,
            attrListTxbufPtr, attrListBufSize);

fn_exit:

    LWSCI_FNEXIT("");

    LwSciCommonTransportBufferFree(attrListTxbuf);
    return error;
}

static LwSciError ExportCoreAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    void** coreAttrListTxbufPtr,
    size_t* coreAttrListBufSize)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportParams bufparams = {0};
    LwSciCommonTransportBuf* coreAttrListTxbuf = NULL;
    uint32_t keyCnt = 0U;
    uint64_t valSize = 0U;
    size_t totalValueSize = 0U;
    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;

    LWSCI_FNENTRY("");

    /* Callwlate key count and value size for all keys */
    error = GetKeyCntAndValSize(&coreAttrList->attrs, &keyCnt, &valSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    bufparams.keyCount = keyCnt;
    totalValueSize = valSize;

    error = GetPrivateKeyCntAndValSize(coreAttrList, state,
            ipcEndpoint, &keyCnt,
            &valSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    u32Add((bufparams.keyCount), keyCnt, &(bufparams.keyCount), &addStatus1);
    u64Add(totalValueSize, valSize, &totalValueSize, &addStatus2);
    if (OP_SUCCESS != (addStatus1 & addStatus2)) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* Create buffer and serialize all keys */
    error = LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalValueSize,
            &coreAttrListTxbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = ExportPrivateKeys(coreAttrList, state,
            ipcEndpoint, coreAttrListTxbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_transportbuf;
    }

    error = ExportAttributes(coreAttrList, coreAttrListTxbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_transportbuf;
    }

    LwSciCommonTransportPrepareBufferForTx(coreAttrListTxbuf,
            coreAttrListTxbufPtr, coreAttrListBufSize);

free_transportbuf:
    LwSciCommonTransportBufferFree(coreAttrListTxbuf);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

static LwSciError ExportAttrList(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen)
{
    LwSciError error = LwSciError_Success;
    uint8_t addStatus = OP_FAIL;

    LwSciCommonTransportParams bufParams = {0};
    size_t attrListValueSize = 0U;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    void** coreAttrListTxBufPtr = NULL;
    size_t* coreAttrListBufSize = NULL;
    void* attrListTxBufPtr = NULL;
    size_t attrListTxBufSize = 0U;
    size_t i = 0U;

    LWSCI_FNENTRY("");

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    /* Add library info in the desc */
    bufParams.msgVersion = LwSciSyncCoreGetLibVersion();
    bufParams.msgMagic = LW_SCI_SYNC_TRANSPORT_ATTR_MAGIC;

    /* Alloc memory for core attr list desc */
    coreAttrListTxBufPtr = (void**)LwSciCommonCalloc(
            objAttrList->numCoreAttrList, sizeof(void*));
    if (NULL == coreAttrListTxBufPtr) {
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    coreAttrListBufSize = (size_t*)LwSciCommonCalloc(
            objAttrList->numCoreAttrList, sizeof(size_t));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == coreAttrListBufSize) {
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_core_attr_list;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        error = ExportCoreAttrList(&objAttrList->coreAttrList[i],
                objAttrList->state, ipcEndpoint, &coreAttrListTxBufPtr[i],
                &coreAttrListBufSize[i]);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_core_attr_list_tx_buf;
        }
        u64Add(attrListValueSize, coreAttrListBufSize[i], &attrListValueSize,
                &addStatus);
        if (OP_SUCCESS != addStatus) {
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_core_attr_list_tx_buf;
        }
    }
    /* Create attr list desc */
    error = ExportObjAttrList(objAttrList, attrListValueSize,
            coreAttrListTxBufPtr, coreAttrListBufSize, &attrListTxBufPtr,
            &attrListTxBufSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_attr_list_tx_buf;
    }
    /* Final desc */
    bufParams.keyCount = 1U;
    error = TransportCreateBufferWithSingleKey(
            bufParams, (uint32_t) LwSciSyncCoreDescKey_AttrList,
            attrListTxBufPtr, attrListTxBufSize, descBuf, descLen);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_attr_list_tx_buf;
    }

free_attr_list_tx_buf:
    /** Free Attribute List TX buffer */
    LwSciCommonFree(attrListTxBufPtr);

free_core_attr_list_tx_buf:
    {
        /* Free Core Attribute List TX buffers that were successfully
         * allocated */
        size_t numProcessed = i;
        if (OP_FAIL == addStatus) {
            /* If we overflowed, we allocated a buffer for that Core Attribute
             * List TX Buffer, which still needs to be freed. */
            numProcessed += 1U;
        }
        for (i = 0U; i < numProcessed; i++) {
            LwSciCommonFree(coreAttrListTxBufPtr[i]);
        }
    }

    LwSciCommonFree(coreAttrListBufSize);

free_core_attr_list:
    LwSciCommonFree(coreAttrListTxBufPtr);

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

/*
 * Assumes that outputPrimitiveType has size MAX_PRIMITIVE_TYPE
 * Assumes length / sizeof(primitiveType) < MAX_PRIMITIVE_TYPE
 */
static LwSciError ImportPrimitiveType(
    LwSciSyncInternalAttrValPrimitiveType primitiveUpperBound,
    const void* value,
    size_t length,
    LwSciSyncInternalAttrValPrimitiveType* outputPrimitiveType)
{
    size_t i = 0U;
    LwSciError error = LwSciError_Success;
    const LwSciSyncInternalAttrValPrimitiveType* importedTypes = NULL;
    size_t numImported = length / sizeof(LwSciSyncInternalAttrValPrimitiveType);

    /* in unreconciled import we should only receive known types from
     * a version <= the local
     * in reconciled import we should get a known, reconciled type
     * so we error out here on unrecognized type */

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    importedTypes = (const LwSciSyncInternalAttrValPrimitiveType*)value;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    for (i = 0U; i < numImported; i++) {
        if (primitiveUpperBound <= importedTypes[i]) {
            LWSCI_ERR_UINT("Encountered primitve type ", (uint32_t)importedTypes[i]);
            LWSCI_ERR_UINT("bigger than promised \n", (uint32_t)primitiveUpperBound);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        if (importedTypes[i] >=
                LwSciSyncInternalAttrValPrimitiveType_UpperBound) {
            LWSCI_ERR_UINT("Unrecognized primitive \n",
                    (uint32_t)importedTypes[i]);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        outputPrimitiveType[i] = importedTypes[i];
    }

    /* explicitly fill the rest of the outputPrimitiveBuffer
     * with no-type tokens */
    for (i = numImported; i < MAX_PRIMITIVE_TYPE; i++) {
        outputPrimitiveType[i] =
                LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    }

fn_exit:
    return error;
}

static LwSciError ImportAttribute(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncInternalAttrValPrimitiveType primitiveUpperBound,
    uint32_t inputKey,
    const void* value,
    size_t length)
{
    size_t keyIdx;
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* check that this is a public/internal attribute by a key range check */
    if (inputKey >= (uint32_t) LwSciSyncCoreAttrKey_LowerBound) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* based on checks before, we know inputKey to be a valid
     * public or internal key */
    keyIdx = LwSciSyncCoreKeyToIndex(inputKey);

    /* check length sanity */
    if (0U == length) {
        LWSCI_ERR_UINT("length of key is invalid: \n", inputKey);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (0U != (length % LwSciSyncCoreKeyInfo[keyIdx].elemSize)) {
        LWSCI_ERR_UINT("length of key is invalid: \n", inputKey);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if ((length / LwSciSyncCoreKeyInfo[keyIdx].elemSize) >
            LwSciSyncCoreKeyInfo[keyIdx].maxElements) {
        LWSCI_ERR_UINT("length of key is too big: \n", inputKey);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if ((uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo == inputKey) {
        error = ImportPrimitiveType(primitiveUpperBound, value, length,
                coreAttrList->attrs.signalerPrimitiveInfo);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if ((uint32_t)LwSciSyncInternalAttrKey_WaiterPrimitiveInfo ==
               inputKey) {
        error = ImportPrimitiveType(primitiveUpperBound, value, length,
                coreAttrList->attrs.waiterPrimitiveInfo);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if ((uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveCount ==
            inputKey) {
        LwSciCommonMemcpyS(&coreAttrList->attrs.signalerPrimitiveCount,
                sizeof(coreAttrList->attrs.signalerPrimitiveCount),
                value, length);
    } else if ((uint32_t)LwSciSyncInternalAttrKey_GpuId == inputKey) {
        /* This value is validated at usage in Rm */
        LwSciCommonMemcpyS(&coreAttrList->attrs.gpuId,
                sizeof(coreAttrList->attrs.gpuId),
                value, length);
    } else if ((uint32_t)LwSciSyncAttrKey_NeedCpuAccess == inputKey) {
        /* We don't range validate bools */
        LwSciCommonMemcpyS(&coreAttrList->attrs.needCpuAccess,
                sizeof(coreAttrList->attrs.needCpuAccess),
                value, length);
    } else if ((uint32_t)LwSciSyncAttrKey_RequiredPerm == inputKey) {
        LwSciCommonMemcpyS(&coreAttrList->attrs.requiredPerm,
                sizeof(coreAttrList->attrs.requiredPerm),
                value, length);
        if (!LwSciSyncCorePermLessThan(
                coreAttrList->attrs.requiredPerm,
                LwSciSyncAccessPerm_WaitSignal)) {
            LWSCI_ERR_HEXUINT("Unrecognized requested permissions \n",
                    (uint32_t)coreAttrList->attrs.requiredPerm);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if ((uint32_t)LwSciSyncAttrKey_ActualPerm == inputKey) {
        LwSciCommonMemcpyS(&coreAttrList->attrs.actualPerm,
                sizeof(coreAttrList->attrs.actualPerm),
                value, length);
        if (!LwSciSyncCorePermValid(coreAttrList->attrs.actualPerm)) {
            LWSCI_ERR_HEXUINT("Unrecognized granted permissions \n",
                    (uint32_t)coreAttrList->attrs.actualPerm);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if (inputKey ==
            (uint32_t)LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports) {
        LwSciCommonMemcpyS(
                &coreAttrList->attrs.waiterContextInsensitiveFenceExports,
                sizeof(coreAttrList->attrs.waiterContextInsensitiveFenceExports),
                value, length);
    } else if ((uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo ==
            inputKey) {
        LwSciCommonMemcpyS(&coreAttrList->attrs.signalerTimestampInfo,
                sizeof(coreAttrList->attrs.signalerTimestampInfo),
                value, length);

        error = LwSciSyncValidateTimestampInfo(
            &coreAttrList->attrs.signalerTimestampInfo, 1U);
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
    } else if (inputKey ==
            (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti) {
        LwSciCommonMemcpyS(&coreAttrList->attrs.signalerTimestampInfoMulti,
                sizeof(coreAttrList->attrs.signalerTimestampInfoMulti),
                value, length);

        error = LwSciSyncValidateTimestampInfo(
            coreAttrList->attrs.signalerTimestampInfoMulti,
            length / sizeof(coreAttrList->attrs.signalerTimestampInfoMulti[0]));
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
    } else if (inputKey == (uint32_t)LwSciSyncInternalAttrKey_EngineArray) {
        LwSciCommonMemcpyS(&coreAttrList->attrs.engineArray,
                sizeof(coreAttrList->attrs.engineArray),
                value, length);
        error = LwSciSyncCoreCheckHwEngineValues(
            coreAttrList->attrs.engineArray,
            length / sizeof(coreAttrList->attrs.engineArray[0]));
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Invalid engine array");
            goto fn_exit;
        }
    } else if ((uint32_t)LwSciSyncAttrKey_WaiterRequireTimestamps == inputKey) {
        /* We don't range validate bools */
        LwSciCommonMemcpyS(&coreAttrList->attrs.waiterRequireTimestamps,
                sizeof(coreAttrList->attrs.waiterRequireTimestamps),
                value, length);
    } else if (inputKey == (uint32_t)LwSciSyncAttrKey_RequireDeterministicFences) {
        /* We don't range validate bools */
        LwSciCommonMemcpyS(&coreAttrList->attrs.requireDeterministicFences,
                sizeof(coreAttrList->attrs.requireDeterministicFences),
                value, length);
    } else {
        LWSCI_ERR_UINT("Encountered an unrecognized key despite all the checks \n",
                  inputKey);
        LwSciCommonPanic();
    }

    coreAttrList->attrs.valSize[keyIdx] = length;
    coreAttrList->attrs.keyState[keyIdx] =
            LwSciSyncCoreAttrKeyState_SetLocked;

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ImportTimestampsAttrList(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    bool importReconciled,
    const void* inputValue,
    size_t length)
{
    LwSciError error = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciBufAttrList* timestampsAttrListArray = NULL;
    size_t timestampsAttrListCount = 0U;
    size_t i = 0U;
    uint8_t addStatus = OP_FAIL;

    LwSciSyncCoreModuleGetBufModule(module, &bufModule);

    if (!importReconciled) {
        error = LwSciBufAttrListIpcImportUnreconciled(bufModule, ipcEndpoint,
                inputValue, length, &coreAttrList->timestampBufAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else {
        if (NULL != inputAttrList) {
            LwSciSyncCoreAttrListGetObjFromRef(inputAttrList, &objAttrList);
            timestampsAttrListArray = (LwSciBufAttrList*) LwSciCommonCalloc(
                    objAttrList->numCoreAttrList,
                    sizeof(LwSciBufAttrList));
            if (NULL == timestampsAttrListArray) {
                LWSCI_ERR_STR("failed to allocate memory.\n");
                error = LwSciError_InsufficientMemory;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
            for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
                if (objAttrList->coreAttrList[i].timestampBufAttrList != NULL) {
                        timestampsAttrListArray[timestampsAttrListCount] =
                            objAttrList->coreAttrList[i].timestampBufAttrList;
                    u64Add(timestampsAttrListCount, 1U,
                           &timestampsAttrListCount, &addStatus);
                    if (OP_SUCCESS != addStatus) {
                        error = LwSciError_Overflow;
                        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                        goto fn_exit;
                    }
                }
            }
            if (0U == timestampsAttrListCount) {
                LwSciCommonFree(timestampsAttrListArray);
                timestampsAttrListArray = NULL;
            }
        }
        error = LwSciBufAttrListIpcImportReconciled(bufModule, ipcEndpoint,
                inputValue, length, timestampsAttrListArray,
                timestampsAttrListCount,
                &coreAttrList->timestampBufAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
fn_exit:
    if (NULL != timestampsAttrListArray) {
        LwSciCommonFree(timestampsAttrListArray);
    }
    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError ImportPrivateKeys(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    uint32_t inputKey,
    const void* value,
    size_t length,
    bool importReconciled)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrKey key;
    LwSciCommonMemcpyS(&key, sizeof(key), &inputKey, sizeof(inputKey));

#if !defined(__x86_64__)
    (void)module;
    (void)ipcEndpoint;
    (void)inputAttrList;
#endif

    LWSCI_FNENTRY("");

    /* only private keys handled here */
    if (inputKey < (uint32_t)LwSciSyncCoreAttrKey_LowerBound) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (LwSciSyncCoreAttrKey_IpcTable == key) {
        error = LwSciSyncCoreImportIpcTable(&coreAttrList->ipcTable,
                value, length, importReconciled);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if (LwSciSyncCoreAttrKey_SemaAttrList == key) {
        error = ImportSemaAttrList(coreAttrList, module,
                ipcEndpoint, inputAttrList, importReconciled,
                value, length);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("call ImportSemaAttrList failed.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if (LwSciSyncCoreAttrKey_TimestampsAttrList == key) {
        error = ImportTimestampsAttrList(coreAttrList, module, ipcEndpoint,
                inputAttrList, importReconciled, value, length);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("call ImportTimestampsAttrList failed.\n");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    } else if (LwSciSyncCoreAttrKey_ExportIpcEndpoint == key) {
        if (sizeof(LwSciIpcEndpoint) != length) {
            error = LwSciError_BadParameter;
            LWSCI_ERR_STR("wrong size of _ExportIpcEndpoint value");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        coreAttrList->lastExport = *(const LwSciIpcEndpoint*)value;
    } else {
        if (importReconciled) {
            LWSCI_INFO("Unrecognized private key %u\n", inputKey);
        } else {
            LWSCI_ERR_UINT("Unrecognized private key in unreconciled import: \n",
                    inputKey);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

struct CoreAttrListTagInfo {
    uint32_t tag;
    bool alreadyHandled;
};

static struct CoreAttrListTagInfo* FindTagInfoOfCoreAttrListImportKey(
    uint32_t key,
    struct CoreAttrListTagInfo* tagInfo,
    size_t numTags)
{
    size_t i = 0U;
    struct CoreAttrListTagInfo* info = NULL;

    for (i = 0U; i < numTags; ++i) {
        if (key == tagInfo[i].tag) {
            info = &tagInfo[i];
            break;
        }
    }

    return info;
}

static LwSciError ImportCoreAttrList(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    LwSciSyncInternalAttrValPrimitiveType primitiveUpperBound,
    const void* inputValue,
    size_t length,
    bool importReconciled)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportBuf* coreAttrListRxbuf = NULL;
    bool doneReading = true;
    LwSciCommonTransportParams params = {0};
    struct CoreAttrListTagInfo tagInfo[] = {
        {(uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveInfo, false},
        {(uint32_t)LwSciSyncInternalAttrKey_WaiterPrimitiveInfo, false},
        {(uint32_t)LwSciSyncInternalAttrKey_SignalerPrimitiveCount, false},
        {(uint32_t)LwSciSyncInternalAttrKey_GpuId, false},
        {(uint32_t)LwSciSyncAttrKey_NeedCpuAccess, false},
        {(uint32_t)LwSciSyncAttrKey_RequiredPerm, false},
        {(uint32_t)LwSciSyncAttrKey_ActualPerm, false},
        {(uint32_t)LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
         false},
        {(uint32_t)LwSciSyncAttrKey_RequireDeterministicFences, false},
        {(uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo, false},
        {(uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti, false},
        {(uint32_t)LwSciSyncInternalAttrKey_EngineArray, false},
        {(uint32_t)LwSciSyncAttrKey_WaiterRequireTimestamps, false},
        {(uint32_t)LwSciSyncCoreAttrKey_TimestampsAttrList, false},
        {(uint32_t)LwSciSyncCoreAttrKey_SemaAttrList, false},
        {(uint32_t)LwSciSyncCoreAttrKey_IpcTable, false},
        {(uint32_t)LwSciSyncCoreAttrKey_ExportIpcEndpoint, false},
    };
    size_t numTags = sizeof(tagInfo) / sizeof(struct CoreAttrListTagInfo);
    struct CoreAttrListTagInfo* info = NULL;

    LWSCI_FNENTRY("");

    error = LwSciCommonTransportGetRxBufferAndParams(inputValue, length,
            &coreAttrListRxbuf, &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    do {
        uint32_t key = 0U;
        const void* value = NULL;

        error = LwSciCommonTransportGetNextKeyValuePair(coreAttrListRxbuf,
                &key, &length, &value, &doneReading);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }

        /* recognize the key */
        info = FindTagInfoOfCoreAttrListImportKey(key, tagInfo, numTags);
        if (NULL == info) {
            if (importReconciled) {
                /* don't check for repeated in unrecognized case */
                LWSCI_INFO("Unrecognized key %u\n", key);
                continue;
            } else {
                LWSCI_ERR_UINT("Unrecognized key in import unreconciled: \n", key);
                error = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_rx_buf;
            }
        }

        if (info->alreadyHandled) {
            LWSCI_ERR_UINT("Unexpected repetition of: \n", key);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }

        error = ImportAttribute(coreAttrList, primitiveUpperBound, key, value,
                length);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }
        error = ImportPrivateKeys(coreAttrList, module, ipcEndpoint,
                inputAttrList, key, value, length, importReconciled);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }

        info->alreadyHandled = true;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    } while (doneReading == false);

free_rx_buf:
    LwSciCommonTransportBufferFree(coreAttrListRxbuf);

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError ImportNumCoreAttrList(
    LwSciSyncCoreAttrListObj** objAttrList,
    LwSciSyncModule module,
    LwSciSyncAttrList inputAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncInternalAttrValPrimitiveType* primitiveUpperBound,
    size_t* slotIndex,
    const void* value,
    size_t length,
    bool importReconciled,
    LwSciSyncAttrList* attrList)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;

    size_t slotCnt = 0U;

    (void)inputAttrList;
    (void)ipcEndpoint;
    (void)primitiveUpperBound;
    (void)slotIndex;
    (void)importReconciled;

    if (sizeof(size_t) != length) {
        LWSCI_ERR_STR("Invalid length of slot count tag\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    slotCnt = *(const size_t*)value;
    if (0U == slotCnt) {
        LWSCI_ERR_STR("Imported attribute list must have at least 1 slot\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreAttrListCreateMultiSlot(module, slotCnt,
            true, attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreAttrListGetObjFromRef(*attrList, objAttrList);

fn_exit:
    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError ImportAttrListState(
    LwSciSyncCoreAttrListObj** objAttrList,
    LwSciSyncModule module,
    LwSciSyncAttrList inputAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncInternalAttrValPrimitiveType* primitiveUpperBound,
    size_t* slotIndex,
    const void* value,
    size_t length,
    bool importReconciled,
    LwSciSyncAttrList* attrList)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;

    (void)module;
    (void)inputAttrList;
    (void)ipcEndpoint;
    (void)primitiveUpperBound;
    (void)slotIndex;
    (void)attrList;

    if (sizeof(LwSciSyncCoreAttrListState) != length) {
        LWSCI_ERR_STR("Invalid length of attribute list state tag\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    (*objAttrList)->state = *(const LwSciSyncCoreAttrListState*)value;

    if (importReconciled &&
        ((*objAttrList)->state != LwSciSyncCoreAttrListState_Reconciled)) {
        LWSCI_ERR_ULONG("Expected reconciled state but got \n",
                  (uint64_t) (*objAttrList)->state);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (!importReconciled &&
        ((*objAttrList)->state != LwSciSyncCoreAttrListState_Unreconciled)) {
        LWSCI_ERR_ULONG("Expected unreconciled state but got \n",
                  (uint64_t) (*objAttrList)->state);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError ImportKeyMaxPrimitiveType(
    LwSciSyncCoreAttrListObj** objAttrList,
    LwSciSyncModule module,
    LwSciSyncAttrList inputAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncInternalAttrValPrimitiveType* primitiveUpperBound,
    size_t* slotIndex,
    const void* value,
    size_t length,
    bool importReconciled,
    LwSciSyncAttrList* attrList)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;

    (void)objAttrList;
    (void)module;
    (void)inputAttrList;
    (void)ipcEndpoint;
    (void)slotIndex;
    (void)attrList;

    if (sizeof(LwSciSyncInternalAttrValPrimitiveType) != length) {
        LWSCI_ERR_STR("Invalid length of max primitive type tag\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    *primitiveUpperBound =
            *(const LwSciSyncInternalAttrValPrimitiveType*)value;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if (((uint64_t)*primitiveUpperBound >
             (uint64_t) LwSciSyncInternalAttrValPrimitiveType_UpperBound) &&
        !importReconciled) {
        LWSCI_ERR_ULONG("Unexpected primitives in import unreconciled: \n",
                (uint64_t)*primitiveUpperBound);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (((uint64_t)*primitiveUpperBound <
             (uint64_t) LwSciSyncInternalAttrValPrimitiveType_UpperBound) &&
        importReconciled) {
        LWSCI_ERR_ULONG("Too few primitives recognized upstream: \n",
                (uint64_t)*primitiveUpperBound);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError ImportKeyCoreAttrList(
    LwSciSyncCoreAttrListObj** objAttrList,
    LwSciSyncModule module,
    LwSciSyncAttrList inputAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncInternalAttrValPrimitiveType* primitiveUpperBound,
    size_t* slotIndex,
    const void* value,
    size_t length,
    bool importReconciled,
    LwSciSyncAttrList* attrList)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrList* coreAttrList = NULL;
    uint8_t addStatus = OP_FAIL;

    (void)attrList;

    if ((NULL == *objAttrList) ||
        (*primitiveUpperBound ==
         LwSciSyncInternalAttrValPrimitiveType_LowerBound)) {
        LWSCI_ERR_STR("Invalid sequence of keys found in desc\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    coreAttrList = &(*objAttrList)->coreAttrList[*slotIndex];
    u64Add((*slotIndex), 1U, slotIndex, &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("slotIndex value is out of range.\n");
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = ImportCoreAttrList(coreAttrList, module, ipcEndpoint,
            inputAttrList, *primitiveUpperBound, value, length,
            importReconciled);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("ImportCoreAttrList func call failed.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

typedef
LwSciError(* ObjAttrListImportFunc)(
    LwSciSyncCoreAttrListObj** objAttrList,
    LwSciSyncModule module,
    LwSciSyncAttrList inputAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncInternalAttrValPrimitiveType* primitiveUpperBound,
    size_t* slotIndex,
    const void* value,
    size_t length,
    bool importReconciled,
    LwSciSyncAttrList* attrList);

struct ImportObjAttrListTagInfo {
    ObjAttrListImportFunc importFunc;
    size_t expectedTags;
};

static LwSciError ImportObjAttrList(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    const void* attrListRxbufPtr,
    size_t attrListBufSize,
    bool importReconciled,
    LwSciSyncAttrList* attrList)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportBuf* attrListRxbuf = NULL;
    uint32_t inputKey = 0;
    size_t length = 0U;
    const void* value = NULL;
    bool doneReading = true;
    size_t slotIndex = 0U;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciSyncInternalAttrValPrimitiveType primitiveUpperBound =
            LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    LwSciCommonTransportParams params = {0};
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciSync-ADV-MISRAC2012-007")
    struct ImportObjAttrListTagInfo tagInfo
        [CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_UpperBound)] = {
            [CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_NumCoreAttrList)] =
                {ImportNumCoreAttrList, 1},
            [CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_AttrListState)] =
                {ImportAttrListState, 1},
            [CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_MaxPrimitiveType)] =
                {ImportKeyMaxPrimitiveType, 1},
            [CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_CoreAttrList)] =
                {ImportKeyCoreAttrList, 0},
    };
    size_t numTags
        [CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_UpperBound)] = {0};
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))
    uint64_t tagIndex = 0U;
    uint64_t i = 0U;

    LWSCI_FNENTRY("");

    error = LwSciCommonTransportGetRxBufferAndParams(attrListRxbufPtr,
            attrListBufSize, &attrListRxbuf, &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* _NumCoreAttrList must be first */
    error = LwSciCommonTransportGetNextKeyValuePair(attrListRxbuf,
            &inputKey, &length, &value, &doneReading);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }
    if ((size_t)LwSciSyncCoreAttrKey_NumCoreAttrList != inputKey) {
        LWSCI_ERR_UINT("Unexpected tag instead of _NumCoreAttrList: \n",
                inputKey);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciSync-ADV-MISRAC2012-007")
    tagIndex = CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_NumCoreAttrList);
    if (NULL == tagInfo[tagIndex].importFunc) {
        LwSciCommonPanic();
    }

    error = tagInfo[tagIndex].importFunc(&objAttrList, module,
            inputAttrList, ipcEndpoint,
            &primitiveUpperBound, &slotIndex,
            value, length, importReconciled, attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }
    numTags[tagIndex] = 1U;
    /* number of _CoreAttrList must match _NumCoreAttrList */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciSync-ADV-MISRAC2012-007")
    tagInfo[CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_CoreAttrList)]
            .expectedTags = objAttrList->numCoreAttrList;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

    /* rest of the tags */
    do {
        error = LwSciCommonTransportGetNextKeyValuePair(attrListRxbuf,
                &inputKey, &length, &value, &doneReading);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_attr_list;
        }
        if ((inputKey <= (uint32_t) LwSciSyncCoreAttrKey_LowerBound) ||
                (inputKey >= (uint32_t) LwSciSyncCoreAttrKey_UpperBound)) {
            if (importReconciled) {
                LWSCI_INFO("Unrecognized key %u. Ignoring...\n", inputKey);
                continue;
            } else {
                LWSCI_ERR_UINT("Unrecognized key in unreconciled import: \n",
                        inputKey);
                error = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_attr_list;
            }
        }

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciSync-ADV-MISRAC2012-007")
        tagIndex = CORE_ATTR_KEY_TO_IDX(inputKey);
        if (numTags[tagIndex] >= tagInfo[tagIndex].expectedTags) {
            LWSCI_ERR_UINT("too many tags: \n", inputKey);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_attr_list;
        }

        if (NULL != tagInfo[tagIndex].importFunc) {
            error = tagInfo[tagIndex].importFunc(&objAttrList, module,
                    inputAttrList, ipcEndpoint,
                    &primitiveUpperBound, &slotIndex,
                    value, length, importReconciled, attrList);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_attr_list;
            }
            numTags[tagIndex]++;
        } else if (importReconciled) {
            LWSCI_INFO("Key %u unexpected in this context\n", inputKey);
            continue;
        } else {
            LWSCI_ERR_UINT("Key should not appear in this context: \n", inputKey);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_attr_list;
        }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    } while (doneReading == false);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciSync-ADV-MISRAC2012-007")
    for (i = 0U; i < CORE_ATTR_KEY_TO_IDX(LwSciSyncCoreAttrKey_UpperBound);
             i++) {
        if (numTags[i] != tagInfo[i].expectedTags) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_9), "LwSciSync-ADV-MISRAC2012-007")
            LWSCI_ERR_ULONG("Incorrect number of ", IDX_TO_CORE_ATTR_KEY(i));
            LWSCI_ERR_UINT("tags: ", numTags[i]);
            LWSCI_ERR_UINT(" but expected \n", tagInfo[i].expectedTags);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_attr_list;
        }
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
    goto free_rx_buf;

free_attr_list:
    LwSciSyncAttrListFree(*attrList);

free_rx_buf:
    LwSciCommonTransportBufferFree(attrListRxbuf);

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ImportAttrList(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    const void* descBuf,
    size_t descLen,
    bool importReconciled,
    LwSciSyncAttrList* attrList)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportParams params = {0};
    LwSciCommonTransportBuf* rxbuf = NULL;
    bool doneReading = true;
    uint64_t libVersion;
    bool tagAttrListFound = false;

    LWSCI_FNENTRY("");

    /* Read the attr list key */
    error = LwSciCommonTransportGetRxBufferAndParams(descBuf, descLen, &rxbuf,
            &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    libVersion = LwSciSyncCoreGetLibVersion();
    if ((libVersion >> 32U) != (params.msgVersion >> 32U)) {
        LWSCI_ERR_STR("Incompatible Library major version\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }
    if (LW_SCI_SYNC_TRANSPORT_ATTR_MAGIC != params.msgMagic) {
        LWSCI_ERR_STR("Export descriptor's magic id is incorrect");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }

    do {
        uint32_t key = 0U;
        size_t attrListBufSize = 0U;
        const void* attrListRxbufPtr = NULL;

        error = LwSciCommonTransportGetNextKeyValuePair(rxbuf, &key,
                &attrListBufSize, &attrListRxbufPtr, &doneReading);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }
        if ((uint32_t)LwSciSyncCoreDescKey_AttrList == key) {
            if (tagAttrListFound) {
                LWSCI_ERR_STR("LwSciSyncCoreDescKey_AttrList more than once");
                error = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_rx_buf;
            }
            /* Read the attr list */
            error = ImportObjAttrList(module, ipcEndpoint, inputAttrList,
                    attrListRxbufPtr, attrListBufSize, importReconciled,
                    attrList);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_rx_buf;
            }
            tagAttrListFound = true;
        } else if (importReconciled) {
            LWSCI_INFO("Unrecognized tag in Reconciled attr list import %u\n",
                    key);
        } else {
            LWSCI_ERR_UINT("Unrecognized tag in Unreconciled attr list import: \n",
                    key);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    } while (doneReading == false);

    if (!tagAttrListFound) {
        LWSCI_ERR_STR("missing LwSciSyncCoreDescKey_AttrList");
        error = LwSciError_BadParameter;
    }

free_rx_buf:
    LwSciCommonTransportBufferFree(rxbuf);

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError AttrListIpcExportReconciledCheckArgs(
    const LwSciSyncAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    const size_t* descLen)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;
    bool isReconciled = false;

    if (NULL == descBuf) {
        LWSCI_ERR_STR("Invalid argument: descBuf: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == descLen) {
        LWSCI_ERR_STR("Invalid argument: descLen: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (error != LwSciError_Success) {
        LWSCI_ERR_ULONG("Invalid LwSciIpcEndpoint", ipcEndpoint);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncAttrListIsReconciled(reconciledAttrList, &isReconciled);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (false == isReconciled) {
        LWSCI_ERR_STR("Input attr list is not reconciled\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }


fn_exit:
    return error;
}

static LwSciError AttrListIpcImportCheckArgs(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const LwSciSyncAttrList* importedAttrList)
{
    LwSciError error = LwSciError_Success;

    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != error) {
        LWSCI_ERR_ULONG("Invalid LwSciIpcEndpoint", ipcEndpoint);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == descBuf) {
        LWSCI_ERR_STR("Invalid argument: descBuf: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (0U == descLen) {
        LWSCI_ERR_ULONG("Invalid argument: descLen", descLen);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == importedAttrList) {
        LWSCI_ERR_STR("Invalid argument: importedAttrList: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

static LwSciError ExportKey(
    LwSciCommonTransportBuf* txbuf,
    uint32_t key,
    const void* value,
    size_t len)
{
    LwSciError error = LwSciError_Success;
    if (0U < len) {
        error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len, value);
    }
    return error;
}

static LwSciError TransportCreateBufferWithSingleKey(
    LwSciCommonTransportParams bufParams,
    uint32_t key,
    const void* inputPtr,
    size_t inputSize,
    void** descBufPtr,
    size_t* descBufSize)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportBuf* txbuf = NULL;

    error = LwSciCommonTransportAllocTxBufferForKeys(bufParams, inputSize,
            &txbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }
    error = LwSciCommonTransportAppendKeyValuePair(txbuf,
            key, inputSize, inputPtr);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }
    LwSciCommonTransportPrepareBufferForTx(txbuf,
            descBufPtr, descBufSize);
ret:
    LwSciCommonTransportBufferFree(txbuf);
    return error;
}
