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
 * \brief <b>LwSciSync Object external Implementation</b>
 *
 * @b Description: This file implements LwSciSync object external APIs
 *
 * The code in this file is organised as below:
 * -Core structures declaration.
 * -Public interfaces definition.
 */

#include "lwscisync_object_external.h"

#include <string.h>
#include "lwscicommon_libc.h"
#include "lwscicommon_objref.h"
#include "lwscicommon_utils.h"
#include "lwscicommon_transportutils.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core.h"
#include "lwscisync_attribute_transport.h"
#include "lwscisync_core.h"
#include "lwscisync_module.h"
#include "lwscisync_object_core.h"
#include "lwscisync_object_core_cluster.h"

/**
 * Magic ID used to validate export descriptor received over IPC
 */
#define LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC  (0xF00ABD56U)

/******************************************************
 *            Core structures declaration
 ******************************************************/

/**
 * \brief Types of LwSciSyncCoreObj Keys for Exporting
 */
typedef enum {
    /** (LwSciSyncAccessPerm) */
    LwSciSyncCoreObjKey_AccessPerm,
    /** (uint64_t) */
    LwSciSyncCoreObjKey_ModuleCnt,
    /** (LwSciIpcEndpoint) */
    LwSciSyncCoreObjKey_IpcEndpoint,
    /** (LwSciSyncCorePrimitive) */
    LwSciSyncCoreObjKey_CorePrimitive,
    /** (LwSciSyncCoreTimestamps) */
    LwSciSyncCoreObjKey_CoreTimestamps,
} LwSciSyncCoreObjKey;

/******************************************************
 *             Core interfaces declaration
 ******************************************************/

/** Initializes core object structure */
static LwSciError CoreObjInit(
    LwSciSyncAttrList reconciledList,
    LwSciSyncCoreObj* coreObj);

/** Serialize core obj data */
static LwSciError ExportCoreObj(
    const LwSciSyncCoreObj* coreObj,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** descPtr,
    size_t* descSize);

/** Deserialize core obj data */
static LwSciError ImportCoreObj(
    LwSciSyncAccessPerm permissions,
    LwSciSyncAttrList inputAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    const void* objRxBufPtr,
    size_t objRxBufSize,
    int64_t timeoutUs,
    LwSciSyncCoreObj* coreObj);

/** Checks if primitive allocation is required */
static inline bool NeedsPrimitiveAllocation(
    LwSciSyncAttrList attrList)
{
    bool useExternalPrimitive = true;

    LwSciSyncCoreGetSignalerUseExternalPrimitive(attrList,
            &useExternalPrimitive);
    /**
     * If we're using an externally allocated primitive, then we do not need
     * to allocate. In other words, one is true iff the other is false.
     */
    return !useExternalPrimitive;

}

/** Sanity check for export/import permissions */
static inline bool CheckExportPermValues(
    LwSciSyncAccessPerm perm)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTC(INT31_C), "LwSciSync-REQ-CERTC-001")
        return ((LwSciSyncCorePermValid(perm)) ||
        (perm == LwSciSyncAccessPerm_Auto));
    LWCOV_ALLOWLIST_END(LWCOV_CERTC(INT31_C))
}

static inline LwSciSyncObj CastRefToSciSyncObj(LwSciRef* arg) {
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),"LwSciSync-ADV-MISRAC2012-013")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4),"LwSciSync-ADV-MISRAC2012-016")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciSync-ADV-MISRAC2012-001")
    return (LwSciSyncObj)(void*)((char*)(void*)arg
        - LW_OFFSETOF(struct LwSciSyncObjRec, refObj));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
}

/******************************************************
 *            Public interfaces definition
 ******************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjAlloc(
    LwSciSyncAttrList reconciledList,
    LwSciSyncObj *syncObj)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciRef* syncObjParam = NULL;
    bool isReconciled = false;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == syncObj) {
        LWSCI_ERR_STR("Invalid arguments: syncObj: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *syncObj = NULL;

    error = LwSciSyncAttrListIsReconciled(reconciledList, &isReconciled);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (false == isReconciled) {
        LWSCI_ERR_STR("Attr list not reconciled\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);

    /** Allocate memory for new object */
    error = LwSciCommonAllocObjWithRef(sizeof(LwSciSyncCoreObj),
            sizeof(struct LwSciSyncObjRec), &coreObjParam,
            &syncObjParam);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to create sync object\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);
    *syncObj = CastRefToSciSyncObj(syncObjParam);

    LWSCI_INFO("*syncObj: %p\n", *syncObj);

    error = CoreObjInit(reconciledList, coreObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    if (LwSciError_Success != error) {
        if (NULL != coreObj) {
            LwSciSyncObjFreeObjAndRef(*syncObj);
        }
    }

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncObjFree(
    LwSciSyncObj syncObj)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);

    LwSciSyncObjFreeObjAndRef(syncObj);

fn_exit:
    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjDup(
    LwSciSyncObj syncObj,
    LwSciSyncObj* dupObj)
{
    LwSciRef* dupObjParam = NULL;
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == dupObj) {
        LWSCI_ERR_STR("Invalid arguments: dupObj: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("dupObj: %p\n", dupObj);

    error = LwSciCommonDuplicateRef(&syncObj->refObj, &dupObjParam);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *dupObj = CastRefToSciSyncObj(dupObjParam);

fn_exit:

    LWSCI_FNEXIT("");
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjIpcExport(
    LwSciSyncObj syncObj,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncObjIpcExportDescriptor* desc)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciCommonTransportBuf* txBuf = NULL;
    LwSciCommonTransportParams bufparams = { 0 };
    void* coreObjTxBufPtr = NULL;
    void* txBufPtr = NULL;
    size_t coreObjTxBufSize = 0U;
    size_t len = 0U;
    uint32_t key = 0U;
    const void* value = NULL;
    size_t descSize = 0U;

    LWSCI_FNENTRY("");

    if (NULL == desc) {
        LWSCI_ERR_STR("Invalid argument: desc: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (!CheckExportPermValues(permissions)) {
        LWSCI_ERR_ULONG("Invalid argument: permissions: \n", (uint64_t)permissions);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != error) {
        LWSCI_ERR_ULONG("Invalid LwSciIpcEndpoint: \n", ipcEndpoint);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("permissions: %d\n", permissions);
    LWSCI_INFO("ipcEndpoint: %" PRIu64 "\n", ipcEndpoint);
    LWSCI_INFO("desc: %p\n", desc);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    /* Create buffer and serialize all keys */
    error = ExportCoreObj(coreObj, permissions, ipcEndpoint, &coreObjTxBufPtr,
            &coreObjTxBufSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /* Add library info in the desc */
    bufparams.msgVersion = LwSciSyncCoreGetLibVersion();
    bufparams.msgMagic = LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC;
    bufparams.keyCount = 1U;

    /* create top level desc buffer */
    error = LwSciCommonTransportAllocTxBufferForKeys(bufparams, coreObjTxBufSize,
            &txBuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_coreObjTxBufPtr;
    }
    len = coreObjTxBufSize;
    value = (const void*)coreObjTxBufPtr;
    key = (uint32_t)LwSciSyncCoreDescKey_SyncObj;
    error = LwSciCommonTransportAppendKeyValuePair(txBuf, key, len, value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txBuf;
    }
    LwSciCommonTransportPrepareBufferForTx(txBuf, &txBufPtr, &descSize);

    LwSciCommonMemcpyS(desc, sizeof(LwSciSyncObjIpcExportDescriptor),
            txBufPtr, descSize);

    LWSCI_INFO("desc: %p\n", desc);

    LwSciCommonFree(txBufPtr);
free_txBuf:
    LwSciCommonTransportBufferFree(txBuf);
fn_coreObjTxBufPtr:
    LwSciCommonFree(coreObjTxBufPtr);
fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjIpcImport(
    LwSciIpcEndpoint ipcEndpoint,
    const LwSciSyncObjIpcExportDescriptor* desc,
    LwSciSyncAttrList inputAttrList,
    LwSciSyncAccessPerm permissions,
    int64_t timeoutUs,
    LwSciSyncObj* syncObj)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciRef* syncObjParam = NULL;
    LwSciCommonTransportBuf* rxBuf = NULL;
    LwSciCommonTransportParams params = {0};
    const void* objRxBufPtr = NULL;
    size_t objRxBufSize = 0U;
    uint64_t libVersion;
    uint32_t key = 0U;
    void* tempDescBuf = NULL;
    bool doneReading;
    bool isReconciled;
    size_t descLen = sizeof(LwSciSyncObjIpcExportDescriptor);

    LWSCI_FNENTRY("");

    /** validate all input args */
    if (NULL == syncObj) {
        LWSCI_ERR_STR("Invalid argument: syncObj: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == desc) {
        LWSCI_ERR_STR("Invalid argument: desc: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (!CheckExportPermValues(permissions)) {
        LWSCI_ERR_ULONG("Invalid argument: permissions: \n", (uint64_t)permissions);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != error) {
        LWSCI_ERR_ULONG("Invalid LwSciIpcEndpoint: \n", ipcEndpoint);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncAttrListIsReconciled(inputAttrList, &isReconciled);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (false == isReconciled) {
        LWSCI_ERR_STR("Attr list not reconciled\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("inputAttrList: %p\n", inputAttrList);
    LWSCI_INFO("permissions: %d\n", permissions);
    LWSCI_INFO("ipcEndpoint: %" PRIu64 "\n", ipcEndpoint);
    LWSCI_INFO("timeoutUs: %" PRId64 "\n", timeoutUs);
    LWSCI_INFO("desc: %p\n", desc);

    /* Copy the input desc to local desc */
    tempDescBuf = (void*)LwSciCommonCalloc(1U, descLen);
    if (NULL == tempDescBuf) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonMemcpyS(tempDescBuf, descLen, desc, descLen);

    /** Allocate memory for new object */
    error = LwSciCommonAllocObjWithRef(sizeof(LwSciSyncCoreObj),
            sizeof(struct LwSciSyncObjRec), &coreObjParam,
            &syncObjParam);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to create sync object\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_desc_buf;
    }
    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);
    *syncObj = CastRefToSciSyncObj(syncObjParam);

    /** Set module header for future validation */
    coreObj->header = LwSciSyncCoreGenerateObjHeader(coreObj);
    /** Clone the attr list */
    error = LwSciSyncAttrListClone(inputAttrList, &coreObj->attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_obj_and_ref;
    }

    /* Read the sync object key */
    error = LwSciCommonTransportGetRxBufferAndParams(tempDescBuf, descLen,
            &rxBuf, &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_obj_and_ref;
    }

    libVersion = LwSciSyncCoreGetLibVersion();
    if ((libVersion >> 32U) != (params.msgVersion >> 32U)) {
        LWSCI_ERR_STR("Incompatible Library major version\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }
    if (LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC != params.msgMagic) {
        LWSCI_ERR_STR("Export descriptor's magic id is incorrect");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }

    error = LwSciCommonTransportGetNextKeyValuePair(rxBuf, &key,
            &objRxBufSize, &objRxBufPtr, &doneReading);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }
    if ((uint32_t)LwSciSyncCoreDescKey_SyncObj == key) {
        /* Read the core sync object */
        error = ImportCoreObj(permissions, inputAttrList, ipcEndpoint,
                objRxBufPtr, objRxBufSize, timeoutUs, coreObj);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }
    } else {
        LWSCI_ERR_UINT("Invalid key in place of expected _SyncObj: \n", key);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rx_buf;
    }

    LWSCI_INFO("*syncObj: %p\n", *syncObj);

free_rx_buf:
    LwSciCommonTransportBufferFree(rxBuf);

free_obj_and_ref:
    if (LwSciError_Success != error) {
        LwSciSyncObjFreeObjAndRef(*syncObj);
    }

free_desc_buf:
    LwSciCommonFree(tempDescBuf);

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjGenerateFence(
    LwSciSyncObj syncObj,
    LwSciSyncFence *syncFence)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    uint64_t fenceSnapshot = 0U;
    uint64_t fenceId = 0U;
    LwSciSyncAttrList attrList = NULL;
    bool isCpuSignaler = false;
    bool isC2cSignaler = false;
    bool signalerUseExternalPrimitive = false;
    uint32_t timestampSlot = TIMESTAMPS_ILWALID_SLOT;

    LWSCI_FNENTRY("");

    if (NULL == syncFence) {
        LWSCI_ERR_STR("Invalid argument: syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** This API validates the passed syncObj and retrieves attrList */
    error = LwSciSyncObjGetAttrList(syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("syncFence: %p\n", syncFence);

    /* This API must be called only from cpu signaler or C2c signaler */
    LwSciSyncCoreAttrListTypeIsC2cSignaler(attrList, &isC2cSignaler);
    LwSciSyncCoreAttrListTypeIsCpuSignaler(attrList, &isCpuSignaler);

    if (isCpuSignaler) {
        LwSciSyncCoreGetSignalerUseExternalPrimitive(
            attrList,
            &signalerUseExternalPrimitive);

        if (signalerUseExternalPrimitive) {
            LWSCI_ERR_STR("Invalid operation\n");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
        }

    } else if (isC2cSignaler) {
        /* no additional checks here */
    } else {
        LWSCI_ERR_STR("Invalid operation\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    LwSciCommonObjLock(&syncObj->refObj);

    fenceSnapshot = LwSciSyncCorePrimitiveGetNewFence(coreObj->primitive);
    fenceId = LwSciSyncCorePrimitiveGetId(coreObj->primitive);

    if (NULL != coreObj->timestamps) {
        error = LwSciSyncCoreTimestampsGetNextSlot(coreObj->timestamps,
                &timestampSlot);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    LwSciCommonObjUnlock(&syncObj->refObj);

    error = LwSciSyncFenceUpdateFenceWithTimestamp(syncObj, fenceId,
            fenceSnapshot, timestampSlot, syncFence);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to generate fence");
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjSignal(
    LwSciSyncObj syncObj)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciSyncAttrList attrList = NULL;
    bool isCpuSignaler = false;
    bool signalerUseExternalPrimitive = false;

    LWSCI_FNENTRY("");

    /** This API validates the passed syncObj and retrieves attrList */
    error = LwSciSyncObjGetAttrList(syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);

    /** This API must be called only from cpu signaler */
    LwSciSyncCoreAttrListTypeIsCpuSignaler(attrList, &isCpuSignaler);

    if (!isCpuSignaler) {
        LWSCI_ERR_STR("Invalid operation\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreGetSignalerUseExternalPrimitive(attrList,
            &signalerUseExternalPrimitive);

    if (signalerUseExternalPrimitive) {
        LWSCI_ERR_STR("Invalid operation\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    LwSciCommonObjLock(&syncObj->refObj);

    if (NULL != coreObj->timestamps) {
        error = LwSciSyncCoreTimestampsWriteTime(coreObj->timestamps,
                coreObj->primitive);
        if (LwSciError_Success != error) {
            LwSciCommonObjUnlock(&syncObj->refObj);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    /** Signal the primitive */
    error = LwSciSyncCoreSignalPrimitive(coreObj->primitive);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto unlock_syncobj_refobj;
    }

unlock_syncobj_refobj:
    /* Unlocking shouldn't fail if locking was successful. */
    LwSciCommonObjUnlock(&syncObj->refObj);

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListReconcileAndObjAlloc(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    LwSciSyncObj* syncObj,
    LwSciSyncAttrList* newConflictList)
{
    LwSciError error;
    LwSciSyncAttrList newReconciledList = NULL;

    LWSCI_FNENTRY("");

    /** Reconcile attr lists */
    error = LwSciSyncAttrListReconcile(inputArray, inputCount,
            &newReconciledList, newConflictList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /** Allocate sync object */
    error = LwSciSyncObjAlloc((const LwSciSyncAttrList)newReconciledList,
            syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    if (NULL != newReconciledList) {
        LwSciSyncAttrListFree(newReconciledList);
        newReconciledList = NULL;
    }
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncIpcExportAttrListAndObj(
    LwSciSyncObj syncObj,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** attrListAndObjDesc,
    size_t* attrListAndObjDescSize)
{
    LwSciError error;
    LwSciSyncAttrList attrList = NULL;
    void* attListDescBuf;
    size_t attListDescLen;
    size_t attListDescLenTemp = 0U;
    uint8_t addStatus = OP_FAIL;
    LwSciSyncObjIpcExportDescriptor objDescBuf = {0};

    LWSCI_FNENTRY("");

    if (NULL == attrListAndObjDesc) {
        LWSCI_ERR_STR("Invalid descriptor: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == attrListAndObjDescSize) {
        LWSCI_ERR_STR("Invalid descriptor size pointer: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *attrListAndObjDesc = NULL;
    *attrListAndObjDescSize = 0U;
    /** Retrieve attr list from sync object */
    error = LwSciSyncObjGetAttrList(syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /** Export attr list */
    error = LwSciSyncAttrListIpcExportReconciled(attrList, ipcEndpoint,
            &attListDescBuf, &attListDescLen);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /** Export sync object */
    error = LwSciSyncObjIpcExport(syncObj, permissions, ipcEndpoint,
            &objDescBuf);
    if (LwSciError_Success != error) {
        LwSciSyncAttrListFreeDesc(attListDescBuf);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /** Allocate memory for attr list + sync object descriptor */
    sizeAdd(sizeof(objDescBuf), attListDescLen, &attListDescLenTemp, &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("Arithmetic overflow\n");
        LwSciCommonPanic();
    }
    *attrListAndObjDesc = LwSciCommonCalloc(1U, attListDescLenTemp);
    if (NULL == *attrListAndObjDesc) {
        LwSciSyncAttrListFreeDesc(attListDescBuf);
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /** Copy attr list and sync object descriptors */
    sizeAdd(sizeof(objDescBuf), attListDescLen, &attListDescLenTemp, &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("Arithmetic overflow\n");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(*attrListAndObjDesc, attListDescLenTemp,
            attListDescBuf, attListDescLen);
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),"LwSciSync-ADV-MISRAC2012-013")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4),"LwSciSync-ADV-MISRAC2012-016")
    LwSciCommonMemcpyS(((uint8_t *)*attrListAndObjDesc + attListDescLen),
            sizeof(objDescBuf),
            &objDescBuf, sizeof(objDescBuf));
    sizeAdd(sizeof(objDescBuf), attListDescLen, attrListAndObjDescSize, &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("Arithmetic overflow\n");
        LwSciCommonPanic();
    }
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LwSciSyncAttrListFreeDesc(attListDescBuf);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncAttrListAndObjFreeDesc(
    void* attrListAndObjDescBuf)
{
    if (NULL != attrListAndObjDescBuf) {
        LwSciCommonFree(attrListAndObjDescBuf);
    }
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncIpcImportAttrListAndObj(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* attrListAndObjDesc,
    size_t attrListAndObjDescSize,
    LwSciSyncAttrList const attrList[],
    size_t attrListCount,
    LwSciSyncAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciSyncObj* syncObj)
{
    LwSciError error;
    LwSciSyncAttrList importedAttrList = NULL;
    size_t attrListDescLen;
    const LwSciSyncObjIpcExportDescriptor* objDesc = NULL;

    LWSCI_FNENTRY("");

    if (NULL == attrListAndObjDesc) {
        LWSCI_ERR_STR("Invalid descriptor: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == syncObj) {
        LWSCI_ERR_STR("Invalid sync object: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (attrListAndObjDescSize < sizeof(LwSciSyncObjIpcExportDescriptor)) {
        LWSCI_ERR_UINT("Invalid descriptor size", attrListAndObjDescSize);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    attrListDescLen = (attrListAndObjDescSize -
            sizeof(LwSciSyncObjIpcExportDescriptor));

    /** Import attr list */
    error = LwSciSyncAttrListIpcImportReconciled(module, ipcEndpoint,
            attrListAndObjDesc, attrListDescLen, attrList, attrListCount,
            &importedAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /** Import sync object */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),"LwSciSync-ADV-MISRAC2012-013")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4),"LwSciSync-ADV-MISRAC2012-016")
    objDesc = (const LwSciSyncObjIpcExportDescriptor*)((const void*)
            ((const uint8_t*)attrListAndObjDesc + attrListDescLen));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    error = LwSciSyncObjIpcImport(ipcEndpoint, objDesc, importedAttrList,
            minPermissions, timeoutUs, syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    if (NULL != importedAttrList) {
        LwSciSyncAttrListFree(importedAttrList);
        importedAttrList = NULL;
    }
    return error;
}

/******************************************************
 *           Internal interfaces definition
 ******************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjGetNumPrimitives(
    LwSciSyncObj syncObj,
    uint32_t* numPrimitives)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList attrList = NULL;
    LwSciSyncInternalAttrKey key =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
    const void* value;
    size_t len;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == numPrimitives) {
        LWSCI_ERR_STR("Invalid number of primitives: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncObjGetAttrList(syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncAttrListGetSingleInternalAttr(attrList, key, &value, &len);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    *numPrimitives = *(const uint32_t*) value;

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncObjGetSemaphoreInfo(
    const LwSciSyncObj syncObj,
    uint32_t index,
    LwSciSyncSemaphoreInfo* semaphoreInfo)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    uint32_t numPrimitives;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == semaphoreInfo) {
        LWSCI_ERR_STR("Invalid output param: semaphoreInfo: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncObjGetNumPrimitives(syncObj, &numPrimitives);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (index >= numPrimitives) {
        LWSCI_ERR_UINT("Invalid index: \n", index);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    /** Clear LwSciSyncSemaphoreInfo */
    (void)memset(semaphoreInfo, 0, sizeof(LwSciSyncSemaphoreInfo));
    /** Set semaphore id needed for determining offset */
    semaphoreInfo->id = index;
    /** Get semaphore info */
    error = LwSciSyncCorePrimitiveGetSpecificData(coreObj->primitive,
            (void**) &semaphoreInfo);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncObjGetNextTimestampSlot(
    const LwSciSyncObj syncObj,
    uint32_t* slotIndex)
{
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == slotIndex) {
        LWSCI_ERR_STR("Invalid slotIndex: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("slotIndex: %p\n", slotIndex);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    if (NULL == coreObj->timestamps) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreTimestampsGetNextSlot(coreObj->timestamps, slotIndex);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncObjGetTimestampBufferInfo(
    LwSciSyncObj syncObj,
    LwSciSyncTimestampBufferInfo* bufferInfo)
{
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == bufferInfo) {
        LWSCI_ERR_STR("Invalid bufferInfo: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("bufferInfo: %p\n", bufferInfo);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    if (NULL == coreObj->timestamps) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncCoreTimestampsGetBufferInfo(coreObj->timestamps, bufferInfo);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncObjGetPrimitiveType(
    LwSciSyncObj syncObj,
    LwSciSyncInternalAttrValPrimitiveType* primitiveType)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList attrList = NULL;
    LwSciSyncInternalAttrKey key =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
    const void* value;
    size_t len;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == primitiveType) {
        LWSCI_ERR_STR("Invalid primitive type: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncObjGetAttrList(syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncAttrListGetSingleInternalAttr(attrList, key, &value, &len);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonMemcpyS(primitiveType,
            sizeof(LwSciSyncInternalAttrValPrimitiveType),
            value, len);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
LwSciError LwSciSyncObjGetC2cSyncHandle(
    LwSciSyncObj syncObj,
    LwSciC2cInterfaceSyncHandle* syncHandle)
{
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciError error = LwSciError_Success;
    LwSciC2cPcieSyncHandle pcieSyncHandle = NULL;

    LWSCI_FNENTRY("");

    if (NULL == syncHandle) {
        LWSCI_ERR_STR("Invalid syncHandle: NULL pointer");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);
    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    error = LwSciSyncCorePrimitiveGetC2cSyncHandle(
        coreObj->primitive, &pcieSyncHandle);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    syncHandle->pcieSyncHandle = pcieSyncHandle;

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncCoreObjGetC2cRmHandle(
    LwSciSyncObj syncObj,
    LwSciC2cPcieSyncRmHandle* syncRmHandle)
{
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (NULL == syncRmHandle) {
        LWSCI_ERR_STR("Invalid syncRmHandle: NULL pointer");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);
    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    error = LwSciSyncCorePrimitiveGetC2cRmHandle(
        coreObj->primitive, syncRmHandle);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}
#endif

static LwSciError CoreObjInit(
    LwSciSyncAttrList reconciledList,
    LwSciSyncCoreObj* coreObj)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList dupReconciledList = NULL;
    LwSciSyncModule module = NULL;
    bool needsAllocation = false;
    LwSciSyncInternalAttrValPrimitiveType primitiveType =
        LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    const void *value = NULL;
    size_t len = 0U;

    LWSCI_FNENTRY("");

    /** Set module header for future validation */
    coreObj->header = LwSciSyncCoreGenerateObjHeader(coreObj);

    /** Set reference to attr list */
    error = LwSciSyncCoreAttrListDup(reconciledList,
            &dupReconciledList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    coreObj->attrList = dupReconciledList;

    /** Get sync object Id */
    LwSciSyncCoreAttrListGetModule(reconciledList, &module);

    error = LwSciSyncCoreModuleCntrGetNextValue(module,
            &coreObj->objId.moduleCntr);
    if (LwSciError_Success != error) {
        /* this call returns _Overflow but we promise _ResourceError in the spec */
        error = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreGetLastExport(dupReconciledList, &coreObj->objId.ipcEndpoint);

    /** Initialize the backend primitive */
    error = LwSciSyncAttrListGetSingleInternalAttr(reconciledList,
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo, &value, &len);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonMemcpyS(&primitiveType, sizeof(primitiveType),
            value, len);
    /** No allocation needed if primitive is Syncpoint and primitive
     *  provided by the UMD */
    needsAllocation = NeedsPrimitiveAllocation(reconciledList);

    error = LwSciSyncCoreInitPrimitive(primitiveType, reconciledList,
            &coreObj->primitive, needsAllocation);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreTimestampsInit(reconciledList, &coreObj->primitive,
        &coreObj->timestamps);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

typedef struct {
    LwSciCommonTransportBuf* coreObjTxbuf;
    LwSciCommonTransportParams bufparams;
    size_t totalValueSize;
    size_t primitiveExportSize;
    void* primitiveExportBuf;
    size_t timestampsExportSize;
    void* timestampsExportBuf;
} LwSciSyncExportCoreObjContext;

static LwSciError CoreObjTimestampsExport(
    const LwSciSyncCoreObj* coreObj,
    LwSciIpcEndpoint ipcEndpoint,
    void** timestampsExportBuf,
    size_t* timestampsExportSize)
{
    LwSciError error = LwSciError_Success;
    bool waiterRequireTimestamps = false;

    LwSciSyncCoreAttrListGetIpcExportRequireTimestamps(coreObj->attrList,
            ipcEndpoint, &waiterRequireTimestamps);

    if (true == waiterRequireTimestamps) {
        error = LwSciSyncCoreTimestampsExport(coreObj->timestamps, ipcEndpoint,
                timestampsExportBuf, timestampsExportSize);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    return error;
}

static LwSciError ExportCoreObjCreateTxDesc(
    const LwSciSyncCoreObj* coreObj,
    LwSciSyncAccessPerm finalPerm,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncExportCoreObjContext* exportCoreObjContext)
{
    LwSciCommonTransportBuf* coreObjTxbuf = NULL;
    LwSciCommonTransportParams bufparams = exportCoreObjContext->bufparams;

    size_t totalValueSize = 0U;
    size_t primitiveExportSize = 0U;
    void* primitiveExportBuf = NULL;
    size_t timestampsExportSize = 0U;
    void* timestampsExportBuf = NULL;

    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncCorePrimitiveExport(coreObj->primitive,
            finalPerm, ipcEndpoint,
            &primitiveExportBuf, &primitiveExportSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = CoreObjTimestampsExport(coreObj, ipcEndpoint,
            &timestampsExportBuf, &timestampsExportSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    bufparams.keyCount = 3U;
    totalValueSize += sizeof(finalPerm);
    totalValueSize += sizeof(coreObj->objId.moduleCntr);
    totalValueSize += sizeof(ipcEndpoint);
    if (0U < primitiveExportSize) {
        bufparams.keyCount += 1U;
        if ((SIZE_MAX - primitiveExportSize) < totalValueSize) {
            /* it is not possible as everything is set up to fit into
             * sizeof(LwSciSyncObjIpcExportDescriptor) */
            LWSCI_ERR_STR("Sizes arithmetic exceeding "
                          "SIZE_MAX should not be possible here");
            LwSciCommonPanic();
        }
        totalValueSize += primitiveExportSize;
    }
    if (timestampsExportSize > 0U) {
        bufparams.keyCount += 1U;
        if ((SIZE_MAX - timestampsExportSize) < totalValueSize) {
            /* it is not possible as everything is set up to fit into
             * sizeof(LwSciSyncObjIpcExportDescriptor) */
            LWSCI_ERR_STR("Sizes arithmetic exceeding "
                          "SIZE_MAX should not be possible here");
            LwSciCommonPanic();
        }
        totalValueSize += timestampsExportSize;
    }

    error = LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalValueSize,
            &coreObjTxbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    exportCoreObjContext->coreObjTxbuf = coreObjTxbuf;
    exportCoreObjContext->bufparams = bufparams;
    exportCoreObjContext->totalValueSize = totalValueSize;
    exportCoreObjContext->primitiveExportSize = primitiveExportSize;
    exportCoreObjContext->primitiveExportBuf = primitiveExportBuf;
    exportCoreObjContext->timestampsExportSize = timestampsExportSize;
    exportCoreObjContext->timestampsExportBuf = timestampsExportBuf;

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

typedef struct {
    size_t len;
    uint32_t key;
    const void* value;
    bool applicable;
} TransportAppendKeyValuePairParam;

static LwSciError ExportCoreObj(
    const LwSciSyncCoreObj* coreObj,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** descPtr,
    size_t* descSize)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncExportCoreObjContext ecoCtx = { 0 };
    LwSciSyncAccessPerm actualPerm;
    LwSciSyncAccessPerm finalPerm;
    uint32_t param_no;

    TransportAppendKeyValuePairParam params[5] = {
        { sizeof(LwSciSyncAccessPerm), (uint32_t)LwSciSyncCoreObjKey_AccessPerm,
          (const void*)&finalPerm, true },
        { sizeof(coreObj->objId.moduleCntr), (uint32_t)LwSciSyncCoreObjKey_ModuleCnt,
          (const void*)&coreObj->objId.moduleCntr, true },
        { sizeof(ipcEndpoint), (uint32_t)LwSciSyncCoreObjKey_IpcEndpoint,
          (coreObj->objId.ipcEndpoint != 0U)
            ? ((const void*)&coreObj->objId.ipcEndpoint)
            : ((const void*)&ipcEndpoint), true },
        // This entry will be updated below after we fetch the primitive size
        { 0U, (uint32_t)LwSciSyncCoreObjKey_CorePrimitive, NULL, false },
        // This entry will be updated below after we fetch the timestamp size
        { 0U, (uint32_t)LwSciSyncCoreObjKey_CoreTimestamps, NULL, false }
    };

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreAttrListGetIpcExportPerm(coreObj->attrList,
            ipcEndpoint, &actualPerm);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(INT31_C), "LwSciSync-REQ-CERTC-001")
    if (LwSciSyncAccessPerm_Auto == permissions) {
        finalPerm = actualPerm;
    } else {
        finalPerm = permissions;
        if (LwSciSyncCorePermLessThan((LwSciSyncAccessPerm)actualPerm,
                permissions)) {
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    /* Get total size and create a Tx desc */
    error = ExportCoreObjCreateTxDesc(coreObj, finalPerm, ipcEndpoint, &ecoCtx);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    params[3].len = ecoCtx.primitiveExportSize;
    params[3].value = (const void*)ecoCtx.primitiveExportBuf;
    params[3].applicable = (ecoCtx.primitiveExportSize > 0U);

    params[4].len = ecoCtx.timestampsExportSize;
    params[4].value = (const void*)ecoCtx.timestampsExportBuf;
    params[4].applicable = (ecoCtx.timestampsExportSize > 0U);

    for (param_no = 0; param_no < (sizeof(params) / sizeof(params[0])); ++param_no) {
        if (params[param_no].applicable) {
            error = LwSciCommonTransportAppendKeyValuePair(ecoCtx.coreObjTxbuf,
                params[param_no].key, params[param_no].len, params[param_no].value);

            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_coreObjTxbuf;
            }
        }
    }

    LwSciCommonTransportPrepareBufferForTx(ecoCtx.coreObjTxbuf,
            descPtr, descSize);

free_coreObjTxbuf:
    LwSciCommonTransportBufferFree(ecoCtx.coreObjTxbuf);
fn_exit:
    LwSciCommonFree(ecoCtx.timestampsExportBuf);
    LwSciCommonFree(ecoCtx.primitiveExportBuf);

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ImportCoreObjSetActualPerm(
    const LwSciSyncCoreObj* coreObj,
    LwSciSyncAttrList inputAttrList,
    LwSciSyncAccessPerm permissions,
    LwSciSyncAccessPerm grantedPerm)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAccessPerm finalPerm;
    const void* attrVal = NULL;

    LWSCI_FNENTRY("");

    if (!LwSciSyncCorePermValid(grantedPerm)) {
        LWSCI_ERR_ULONG("Invalid desc permissions: \n", (uint64_t)grantedPerm);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(INT31_C), "LwSciSync-REQ-CERTC-001")
    if (LwSciSyncAccessPerm_Auto == permissions) {
        size_t len = 0U;
        LwSciSyncAccessPerm actualPerm;

        error = LwSciSyncAttrListGetAttr(inputAttrList,
                LwSciSyncAttrKey_ActualPerm, &attrVal, &len);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        actualPerm = *(const LwSciSyncAccessPerm*)attrVal;

        finalPerm = actualPerm;
    } else {
        finalPerm = permissions;
    }
    if (!LwSciSyncCorePermLEq(finalPerm, grantedPerm)) {
        LWSCI_ERR_ULONG("Requested perm:  ", (uint64_t)(finalPerm));
        LWSCI_ERR_ULONG("higher than granted perm: \n", (uint64_t)grantedPerm);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    finalPerm = grantedPerm;
    LwSciSyncCoreAttrListSetActualPerm(coreObj->attrList,
            finalPerm);

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

struct CoreObjTagInfo {
    uint32_t tag;
    uint32_t expectedNum;
    uint32_t handledNum;
    bool optional;
};

static struct CoreObjTagInfo* FindCoreObjInfo(
    uint32_t key,
    struct CoreObjTagInfo* tagInfo,
    size_t tagsNum)
{
    size_t i = 0U;
    struct CoreObjTagInfo* result = NULL;

    for (i = 0U; i < tagsNum; ++i) {
        if (key == tagInfo[i].tag) {
            result = &tagInfo[i];
            break;
        }
    }

    return result;
}

static LwSciError ImportCoreObj(
    LwSciSyncAccessPerm permissions,
    LwSciSyncAttrList inputAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    const void* objRxBufPtr,
    size_t objRxBufSize,
    int64_t timeoutUs,
    LwSciSyncCoreObj* coreObj)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportBuf* coreObjRxbuf = NULL;
    LwSciCommonTransportParams params = { 0 };
    LwSciSyncCoreObjKey key;
    bool doneReading = false;
    struct CoreObjTagInfo* info = NULL;
    struct CoreObjTagInfo tagInfo[] = {
        {(uint32_t)LwSciSyncCoreObjKey_AccessPerm, 1U, 0U, false},
        {(uint32_t)LwSciSyncCoreObjKey_ModuleCnt, 1U, 0U, false},
        {(uint32_t)LwSciSyncCoreObjKey_IpcEndpoint, 1U, 0U, false},
        {(uint32_t)LwSciSyncCoreObjKey_CorePrimitive, 1U, 0U, false},
        {(uint32_t)LwSciSyncCoreObjKey_CoreTimestamps, 1U, 0U, true},
    };
    size_t numTags = sizeof(tagInfo) / sizeof(struct CoreObjTagInfo);
    size_t i = 0U;

    (void)timeoutUs;

    LWSCI_FNENTRY("");

    error = LwSciCommonTransportGetRxBufferAndParams(objRxBufPtr,
            objRxBufSize, &coreObjRxbuf, &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    do {
        uint32_t inputKey = 0U;
        size_t length = 0U;
        const void* value = NULL;

        error = LwSciCommonTransportGetNextKeyValuePair(coreObjRxbuf, &inputKey,
                &length, &value, &doneReading);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        info = FindCoreObjInfo(inputKey, tagInfo, numTags);
        if (NULL == info) {
            LWSCI_INFO("Unrecognized tag %u in object descriptor\n", inputKey);
            continue;
        }
        if (info->handledNum >= info->expectedNum) {
            LWSCI_ERR_UINT("Tag is not allowed here in object: \n", inputKey);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        LwSciCommonMemcpyS(&key, sizeof(key), &inputKey, sizeof(inputKey));
        if (LwSciSyncCoreObjKey_AccessPerm == key) {
            if (sizeof(LwSciSyncAccessPerm) != length) {
                LWSCI_ERR_STR("Invalid length for AccessPerm in obj descriptor\n");
                error = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
            error = ImportCoreObjSetActualPerm(coreObj, inputAttrList,
                    permissions, *(const LwSciSyncAccessPerm*)value);
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        } else if (LwSciSyncCoreObjKey_ModuleCnt == key) {
            if (sizeof(coreObj->objId.moduleCntr) != length) {
                LWSCI_ERR_STR("Invalid length of moduleCntr in obj descriptor\n");
                error = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
            coreObj->objId.moduleCntr = *(const uint64_t*)value;
        } else if (LwSciSyncCoreObjKey_IpcEndpoint == key) {
            if (sizeof(coreObj->objId.ipcEndpoint) != length) {
                LWSCI_ERR_STR("Invalid length of ipcEndpoint in obj descriptor\n");
                error = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
            coreObj->objId.ipcEndpoint = *(const LwSciIpcEndpoint*)value;
        } else if (LwSciSyncCoreObjKey_CorePrimitive == key) {
            error = LwSciSyncCorePrimitiveImport(ipcEndpoint, inputAttrList,
                    value, length, &coreObj->primitive);
        } else if (LwSciSyncCoreObjKey_CoreTimestamps == key) {
            error = LwSciSyncCoreTimestampsImport(ipcEndpoint, inputAttrList,
                    value, length, &coreObj->timestamps);
        } else {
            LWSCI_ERR_UINT("Unrecognized key despite performing a check before: \n",
                    inputKey);
            LwSciCommonPanic();
        }
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        info->handledNum++;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    } while (false == doneReading);

    for (i = 0U; i < numTags; ++i) {
        if (!tagInfo[i].optional &&
                (tagInfo[i].expectedNum != tagInfo[i].handledNum)) {
            LWSCI_ERR_UINT("Missing tag in object descriptor: \n", tagInfo[i].tag);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    LwSciCommonTransportBufferFree(coreObjRxbuf);

    LWSCI_FNEXIT("");

    return error;
}
