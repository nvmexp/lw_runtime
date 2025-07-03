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
 * \brief <b>LwSciSync Core Attribute Management Implementation</b>
 *
 * @b Description: This file implements Ipc Table interface
 */
#include "lwscisync_ipc_table.h"

#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscicommon_utils.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscisync_core.h"

/**
 * \brief Tags for export descriptor
 */
typedef enum {
    /** (LwSciIpcEndpoint[]) */
    LwSciSyncCoreIpcTableKey_IpcEndpoints,
    /** (size_t) */
    LwSciSyncCoreIpcTableKey_NumIpcEndpoint,
    /** (LwSciSyncCoreAttrIpcPerm) */
    LwSciSyncCoreIpcTableKey_IpcPermEntry,
    /** (size_t) */
    LwSciSyncCoreIpcTableKey_NumIpcPerm,
    /** (bool) */
    LwSciSyncCoreIpcTableKey_NeedCpuAccess,
    /** (LwSciSyncAccessPerm) */
    LwSciSyncCoreIpcTableKey_RequiredPerm,
    /** (bool) */
    LwSciSyncCoreIpcTableKey_WaiterRequireTimestamps,
    /** (LwSciSyncIpcTopoId[]) */
    LwSciSyncCoreIpcTableKey_TopoIds,
} LwSciSyncCoreIpcTableKey;

struct IpcTableTagInfo {
    uint32_t tag;
    uint32_t expectedNum;
    uint32_t handledNum;
};

/* Validates LwSciIpcEndpoint and TopoId information
 * that should be accessible to the current peer */
static LwSciError ValidateLocalIpcEndpoint(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncIpcTopoId allegedTopoId);

/* Returns allocation info for ipc route and topoIds */
static LwSciError GetIpcRouteKeyCntAndValSize(
    const LwSciSyncCoreIpcTable* ipcTable,
    uint32_t* keyCnt,
    uint64_t* valSize);

/* export the ipc route and the topoids */
static LwSciError ExportIpcRoute(
    const LwSciIpcEndpoint* ipcRoute,
    const LwSciSyncIpcTopoId* topoIds,
    size_t ipcRouteEntries,
    LwSciCommonTransportBuf* txbuf);

/* export a single ipc tree branch */
static LwSciError ExportIpcPermEntry(
    const LwSciSyncCoreAttrIpcPerm* ipcPerm,
    void** txbufPtr,
    size_t* txbufSize);

/* import a single ipc tree branch */
static LwSciError ImportIpcPermEntry(
    LwSciSyncCoreAttrIpcPerm* ipcPerm,
    const void* inputValue,
    size_t length);

#if (LW_IS_SAFETY == 0)
/* fills default values for absent tags */
static void FillDefaultImportedFields(
    LwSciSyncCoreIpcTable* ipcTable,
    struct IpcTableTagInfo tagInfo);
#endif

/* export ipc Perm part of ipc table */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_5), "LwSciSync-ADV-MISRAC2012-017")
static LwSciError ExportIpcPerm(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciCommonTransportParams* bufparams,
    uint64_t* totalBufSize,
    size_t* ipcPermEntriesPtr,
    void*** txbufPtr,
    size_t** bufSize);
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_5))

/* wrapper for appending all the key to txbuf */
static LwSciError AppendKeys(
    LwSciCommonTransportBuf* txbuf,
    const LwSciSyncCoreIpcTable* ipcTable,
    size_t ipcPermEntries,
    void* const* ipcPermTxbufPtr,
    const size_t* ipcPermBufSize);

/* import the size of ipcPermEntries */
static LwSciError NumIpcPermImport(
    LwSciSyncCoreIpcTable* ipcTable,
    struct IpcTableTagInfo* tagInfo,
    size_t tagsNum,
    const void* value,
    size_t length);

/* import the size of ipcRouteEntries */
static LwSciError NumIpcEndpointImport(
    LwSciSyncCoreIpcTable* ipcTable,
    struct IpcTableTagInfo* tagInfo,
    size_t tagsNum,
    const void* value,
    size_t length);

/* import ipcRoute */
static LwSciError IpcRouteImport(
    const LwSciSyncCoreIpcTable* ipcTable,
    const void* value,
    size_t length);

/* import topoIds */
static LwSciError TopoIdsImport(
    const LwSciSyncCoreIpcTable* ipcTable,
    const void* value,
    size_t length);

/* clear dest and copy src onto dest. */
static void MoveIpcPermEntry(
    LwSciSyncCoreAttrIpcPerm* dest,
    const LwSciSyncCoreAttrIpcPerm* src);

bool LwSciSyncCoreIpcTableRouteIsEmpty(
    const LwSciSyncCoreIpcTable* ipcTable)
{
    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    return (ipcTable->ipcRouteEntries == 0U);
}

LwSciError LwSciSyncCoreIpcTableTreeAlloc(
    LwSciSyncCoreIpcTable* ipcTable,
    size_t size)
{
    LwSciError error = LwSciError_Success;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }

    if ((NULL != ipcTable->ipcRoute) || (0U != ipcTable->ipcPermEntries) ||
            (NULL != ipcTable->ipcPerm) || (0U != ipcTable->ipcRouteEntries) ||
            (NULL != ipcTable->topoIds)) {
        LWSCI_ERR_STR("invalid ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }

    if (0U == size) {
        /* nothing to do */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    ipcTable->ipcPermEntries = size;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    ipcTable->ipcPerm = (LwSciSyncCoreAttrIpcPerm*)LwSciCommonCalloc(
            size, sizeof(LwSciSyncCoreAttrIpcPerm));
    if (NULL == ipcTable->ipcPerm) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LwSciError LwSciSyncCoreIpcTableAddBranch(
    const LwSciSyncCoreIpcTable* ipcTable,
    size_t slot,
    const LwSciSyncCoreIpcTable* ipcTableWithRoute,
    bool needCpuAccess,
    bool waiterRequireTimestamps,
    LwSciSyncAccessPerm requiredPerm,
    LwSciSyncHwEngine* engineArray,
    size_t engineArrayLen)
{
    LwSciError error = LwSciError_Success;
    const LwSciIpcEndpoint* route = NULL;
    LwSciSyncIpcTopoId* topoIds = NULL;
    size_t routeLen = 0U;
    size_t size = 0U;
    uint8_t mulStatus = OP_FAIL;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == ipcTableWithRoute) {
        LWSCI_ERR_STR("Null ipcTableWithRoute. Panicking!!\n");
        LwSciCommonPanic();
    }
    /* assert that the slot is a valid index of an empty branch */
    if (slot >= ipcTable->ipcPermEntries) {
        LwSciCommonPanic();
    }

    if (NULL != ipcTable->ipcPerm[slot].ipcRoute) {
        LwSciCommonPanic();
    }

    if (NULL != ipcTable->ipcPerm[slot].topoIds) {
        LwSciCommonPanic();
    }

    if (!LwSciSyncCorePermLEq(requiredPerm, LwSciSyncAccessPerm_WaitSignal)) {
        LwSciCommonPanic();
    }

    if (NULL == engineArray) {
        LwSciCommonPanic();
    }

    route = ipcTableWithRoute->ipcRoute;
    topoIds = ipcTableWithRoute->topoIds;
    routeLen = ipcTableWithRoute->ipcRouteEntries;

    ipcTable->ipcPerm[slot].ipcRouteEntries = routeLen;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    ipcTable->ipcPerm[slot].ipcRoute =
        (LwSciIpcEndpoint*)LwSciCommonCalloc(
            ipcTable->ipcPerm[slot].ipcRouteEntries,
            sizeof(LwSciIpcEndpoint));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == ipcTable->ipcPerm[slot].ipcRoute) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    ipcTable->ipcPerm[slot].topoIds =
        (LwSciSyncIpcTopoId*)LwSciCommonCalloc(
            ipcTable->ipcPerm[slot].ipcRouteEntries,
            sizeof(LwSciSyncIpcTopoId));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == ipcTable->ipcPerm[slot].topoIds) {
        LWSCI_ERR_STR("failed to allocate memory.");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ipcroute;
    }

    u64Mul(routeLen, sizeof(LwSciIpcEndpoint), &size, &mulStatus);
    if (OP_SUCCESS != mulStatus) {
        LWSCI_ERR_STR("Unexpected overflow of size value already used for allocation");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(ipcTable->ipcPerm[slot].ipcRoute, size,
            route, size);

    u64Mul(routeLen, sizeof(LwSciSyncIpcTopoId), &size, &mulStatus);
    if (OP_SUCCESS != mulStatus) {
        LWSCI_ERR_STR("Unexpected overflow of size value already used for allocation");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(ipcTable->ipcPerm[slot].topoIds, size,
            topoIds, size);

    ipcTable->ipcPerm[slot].needCpuAccess = needCpuAccess;
    ipcTable->ipcPerm[slot].waiterRequireTimestamps = waiterRequireTimestamps;
    ipcTable->ipcPerm[slot].requiredPerm = requiredPerm;

    LwSciCommonMemcpyS(ipcTable->ipcPerm[slot].engineArray,
            sizeof(ipcTable->ipcPerm[slot].engineArray),
            engineArray, engineArrayLen * sizeof(*engineArray));
    ipcTable->ipcPerm[slot].engineArrayLen = engineArrayLen;

free_ipcroute:
    if (LwSciError_Success != error) {
        LwSciCommonFree(ipcTable->ipcPerm[slot].ipcRoute);
    }
fn_exit:
    return error;
}

static struct IpcTableTagInfo* FindIpcTableTagInfo(
    uint32_t key,
    struct IpcTableTagInfo* tagInfo,
    size_t tagsNum)
{
    size_t i = 0U;
    struct IpcTableTagInfo* result = NULL;

    for (i = 0U; i < tagsNum; ++i) {
        if (key == tagInfo[i].tag) {
            result = &tagInfo[i];
            break;
        }
    }

    return result;
}


static LwSciError NumIpcEndpointImport(
    LwSciSyncCoreIpcTable* ipcTable,
    struct IpcTableTagInfo* tagInfo,
    size_t tagsNum,
    const void* value,
    size_t length)
{
    LwSciError error = LwSciError_Success;
    uint8_t opStatus = OP_FAIL;
    struct IpcTableTagInfo* tmpInfo = NULL;

    if (sizeof(size_t) != length) {
        LWSCI_ERR_STR("Invalid length of _NumIpcEndpoints\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /* last element doesn't come from import, is filled locally later */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    u64Add((*(const size_t*)value), 1U,
            &(ipcTable->ipcRouteEntries), &opStatus);
    if (OP_SUCCESS != opStatus) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    ipcTable->ipcRoute = (LwSciIpcEndpoint*)LwSciCommonCalloc(
            ipcTable->ipcRouteEntries, sizeof(LwSciIpcEndpoint));
    if (NULL == ipcTable->ipcRoute) {
        LWSCI_ERR_STR("Failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    ipcTable->topoIds = (LwSciSyncIpcTopoId*)LwSciCommonCalloc(
            ipcTable->ipcRouteEntries, sizeof(LwSciSyncIpcTopoId));
    if (NULL == ipcTable->topoIds) {
        LWSCI_ERR_STR("Failed to allocate memory.");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ipcroute;
    }

    /* allow handling of IpcEndpoints and TopoIds now */
    tmpInfo = FindIpcTableTagInfo(
            (uint32_t)LwSciSyncCoreIpcTableKey_IpcEndpoints,
            tagInfo, tagsNum);
    if (NULL == tmpInfo) {
        LWSCI_ERR_STR("_IpcEndpoints tag not found\n");
        LwSciCommonPanic();
    }
    tmpInfo->expectedNum = 1U;
    tmpInfo = FindIpcTableTagInfo(
            (uint32_t)LwSciSyncCoreIpcTableKey_TopoIds,
            tagInfo, tagsNum);
    if (NULL == tmpInfo) {
        LWSCI_ERR_STR("_TopoIds tag not found");
        LwSciCommonPanic();
    }
    tmpInfo->expectedNum = 1U;

free_ipcroute:
    if (LwSciError_Success != error) {
        LwSciCommonFree(ipcTable->ipcRoute);
    }
fn_exit:
    return error;
}

static LwSciError NumIpcPermImport(
    LwSciSyncCoreIpcTable* ipcTable,
    struct IpcTableTagInfo* tagInfo,
    size_t tagsNum,
    const void* value,
    size_t length)
{
    LwSciError error = LwSciError_Success;
    struct IpcTableTagInfo* tmpInfo = NULL;

    if (sizeof(size_t) != length) {
        LWSCI_ERR_STR("Invalid length of _NumIpcPerm\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    ipcTable->ipcPermEntries = *(const size_t*)value;
    if (0U == ipcTable->ipcPermEntries) {
        LWSCI_ERR_STR("ipc perm entries must not be 0\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    ipcTable->ipcPerm = (LwSciSyncCoreAttrIpcPerm*)LwSciCommonCalloc(
            ipcTable->ipcPermEntries, sizeof(LwSciSyncCoreAttrIpcPerm));
    if (NULL == ipcTable->ipcPerm) {
        LWSCI_ERR_STR("Failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* allow handling of IpcPermEntries now */
    tmpInfo = FindIpcTableTagInfo(
            (uint32_t)LwSciSyncCoreIpcTableKey_IpcPermEntry,
            tagInfo, tagsNum);
    if (NULL == tmpInfo) {
        LWSCI_ERR_STR("_IpcPermEntry tag not found\n");
        LwSciCommonPanic();
    }
    tmpInfo->expectedNum = (uint32_t)ipcTable->ipcPermEntries;

fn_exit:
    return error;
}

static LwSciError IpcRouteImport(
    const LwSciSyncCoreIpcTable* ipcTable,
    const void* value,
    size_t length)
{
    LwSciError error = LwSciError_Success;
    size_t tmpSize = 0U;
    size_t routeSize = 0U;
    uint8_t opStatus = OP_FAIL;

    u64Mul(sizeof(LwSciIpcEndpoint), ipcTable->ipcRouteEntries,
            &routeSize, &opStatus);
    if (OP_SUCCESS != opStatus) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /* -1 for what we added to ipcRouteEntries above */
    u64Sub(routeSize, sizeof(LwSciIpcEndpoint),
            &tmpSize, &opStatus);
    if (OP_SUCCESS != opStatus) {
        LWSCI_ERR_STR("Impossible arithmetic problem\n");
        LwSciCommonPanic();
    }
    if (tmpSize != length) {
        LWSCI_ERR_UINT("Size of ipcRoute ", length);
        LWSCI_ERR_ULONG("different from promised \n",tmpSize);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonMemcpyS(ipcTable->ipcRoute, routeSize,
            value, length);

fn_exit:
    return error;
}

static LwSciError TopoIdsImport(
    const LwSciSyncCoreIpcTable* ipcTable,
    const void* value,
    size_t length)
{
    LwSciError error = LwSciError_Success;
    size_t tmpSize = 0U;
    size_t routeSize = 0U;
    uint8_t opStatus = OP_FAIL;

    u64Mul(sizeof(LwSciSyncIpcTopoId), ipcTable->ipcRouteEntries,
            &routeSize, &opStatus);
    if (OP_SUCCESS != opStatus) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /* -1 for what we added to ipcRouteEntries above */
    u64Sub(routeSize, sizeof(LwSciSyncIpcTopoId),
            &tmpSize, &opStatus);
    if (OP_SUCCESS != opStatus) {
        LWSCI_ERR_STR("Impossible arithmetic problem");
        LwSciCommonPanic();
    }
    if (tmpSize != length) {
        LWSCI_ERR_UINT("Size of topoIds ", length);
        LWSCI_ERR_ULONG("different from promised ",tmpSize);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonMemcpyS(ipcTable->topoIds, routeSize,
            value, length);

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreImportIpcTableParamCheck(
    const LwSciSyncCoreIpcTable* ipcTable,
    const void* desc,
    size_t size)
{
    LwSciError error = LwSciError_Success;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == desc) {
        LWSCI_ERR_STR("Null desc. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (0U == size) {
        error = LwSciError_BadParameter;
    }

    return error;
}

#if (LW_IS_SAFETY == 0)
static void FillDefaultImportedFields(
    LwSciSyncCoreIpcTable* ipcTable,
    struct IpcTableTagInfo tagInfo)
{
    size_t i = 0U;

    // If TopoIds are missing then assume that it comes from
    // the older library and the scheme works fully inside a single VM
    if (tagInfo.tag == LwSciSyncCoreIpcTableKey_TopoIds &&
        tagInfo.expectedNum == 1U &&
        tagInfo.handledNum == 0U) {
        for (i = 0U; i < ipcTable->ipcRouteEntries; i++) {
            ipcTable->topoIds[i].topoId.SocId = LWSCIIPC_SELF_SOCID;
        }
        tagInfo.handledNum = 1U;
    }
}
#endif

LwSciError LwSciSyncCoreImportIpcTable(
    LwSciSyncCoreIpcTable* ipcTable,
    const void* desc,
    size_t size,
    bool importReconciled)
{
    LwSciError error = LwSciError_Success;
    bool doneReading = false;
    LwSciCommonTransportBuf* rxBuf = NULL;
    LwSciCommonTransportParams params;
    LwSciSyncCoreIpcTableKey key;
    struct IpcTableTagInfo* info = NULL;
    struct IpcTableTagInfo tagInfo[] = {
        {(uint32_t)LwSciSyncCoreIpcTableKey_NumIpcEndpoint,
         (importReconciled ? 0U : 1U), 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_IpcEndpoints,
         0U, 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_TopoIds,
         0U, 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_NumIpcPerm,
         (importReconciled ? 1U : 0U), 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_IpcPermEntry,
         0U, 0U},
    };
    size_t tagsNum = sizeof(tagInfo) / sizeof(struct IpcTableTagInfo);
    size_t i = 0U;

    error = LwSciSyncCoreImportIpcTableParamCheck(ipcTable, desc, size);
    if (error != LwSciError_Success) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciCommonTransportGetRxBufferAndParams(
            desc, size, &rxBuf, &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    do {
        uint32_t inputKey = 0U;
        size_t length = 0U;
        const void* value = NULL;

        error = LwSciCommonTransportGetNextKeyValuePair(
                rxBuf, &inputKey, &length, &value, &doneReading);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }
        LwSciCommonMemcpyS(&key, sizeof(key), &inputKey, sizeof(inputKey));
        info = FindIpcTableTagInfo((uint32_t)key, tagInfo, tagsNum);
        if (NULL == info) {
            if (importReconciled) {
                LWSCI_INFO("Unrecognized tag in IpcTable %u\n", key);
                continue;
            } else {
                LWSCI_ERR_UINT("Unrecognized tag in IpcTable \n", key);
                error = LwSciError_BadParameter;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_rx_buf;
            }
        }
        if (info->handledNum >= info->expectedNum) {
            LWSCI_ERR_UINT("Tag is not allowed here in IpcTable: \n", key);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }

        switch (key) {
            case LwSciSyncCoreIpcTableKey_NumIpcEndpoint:
            {
                error = NumIpcEndpointImport(ipcTable, tagInfo, tagsNum,
                        value, length);
                if (LwSciError_Success != error) {
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_rx_buf;
                }
                break;
            }
            case LwSciSyncCoreIpcTableKey_IpcEndpoints:
            {
                error = IpcRouteImport(ipcTable, value, length);
                if (LwSciError_Success != error) {
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_rx_buf;
                }
                break;
            }
            case LwSciSyncCoreIpcTableKey_TopoIds:
            {
                error = TopoIdsImport(ipcTable, value, length);
                if (LwSciError_Success != error) {
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_rx_buf;
                }
                break;
            }
            case LwSciSyncCoreIpcTableKey_NumIpcPerm:
            {
                error = NumIpcPermImport(ipcTable, tagInfo, tagsNum, value, length);
                if (LwSciError_Success != error) {
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_rx_buf;
                }
                break;
            }
            case LwSciSyncCoreIpcTableKey_IpcPermEntry:
            {
                error = ImportIpcPermEntry(&ipcTable->ipcPerm[info->handledNum],
                        value, length);
                if (LwSciError_Success != error) {
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_rx_buf;
                }
                break;
            }
            default:
            {
                LWSCI_ERR_UINT("Unrecognized tag despite previous check: \n", key);
                LwSciCommonPanic();
                break;
            }
        }
        info->handledNum++;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    } while (false == doneReading);

    /* validate expected contents */
    for (i = 0U; i < tagsNum; ++i) {
#if (LW_IS_SAFETY == 0)
        FillDefaultImportedFields(ipcTable, tagInfo[i]);
#endif
        if (tagInfo[i].expectedNum != tagInfo[i].handledNum) {
            LWSCI_ERR_UINT("IpcTable descriptor is missing tag \n",
                      tagInfo[i].tag);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_rx_buf;
        }
    }

free_rx_buf:
    LwSciCommonTransportBufferFree(rxBuf);

fn_exit:
    return error;
}

void LwSciSyncCoreIpcTableFree(
    LwSciSyncCoreIpcTable* ipcTable)
{
    size_t j = 0U;

    if (NULL == ipcTable) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciCommonFree(ipcTable->ipcRoute);
    LwSciCommonFree(ipcTable->topoIds);
    if (NULL != ipcTable->ipcPerm) {
        for (j = 0U; j < ipcTable->ipcPermEntries; j++) {
            LwSciCommonFree(ipcTable->ipcPerm[j].ipcRoute);
            LwSciCommonFree(ipcTable->ipcPerm[j].topoIds);
        }
    }
    LwSciCommonFree(ipcTable->ipcPerm);
    ipcTable->ipcRoute = NULL;
    ipcTable->topoIds = NULL;
    ipcTable->ipcPerm = NULL;
    ipcTable->ipcRouteEntries = 0U;
    ipcTable->ipcPermEntries = 0U;

fn_exit:
    return;
}

LwSciError LwSciSyncCoreCopyIpcTable(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciSyncCoreIpcTable* newIpcTable)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t arrSize = 0U;
    uint8_t opStatus = OP_FAIL;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == newIpcTable) {
        LWSCI_ERR_STR("Null newIpcTable. Panicking!!\n");
        LwSciCommonPanic();
    }

    if ((0U != newIpcTable->ipcRouteEntries) ||
            (NULL != newIpcTable->ipcRoute) ||
            (NULL != newIpcTable->topoIds) ||
            (0U != newIpcTable->ipcPermEntries) ||
            (NULL != newIpcTable->ipcPerm)) {
        LWSCI_ERR_STR("Non 0/Null initialized newIpcTable. Panicking!!\n");
        LwSciCommonPanic();
    }

    /* copy ipcRoute and topoIds */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    newIpcTable->ipcRouteEntries = ipcTable->ipcRouteEntries;
    if (0U != newIpcTable->ipcRouteEntries) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        newIpcTable->ipcRoute = (LwSciIpcEndpoint*)LwSciCommonCalloc(
                newIpcTable->ipcRouteEntries, sizeof(LwSciIpcEndpoint));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        if (NULL == newIpcTable->ipcRoute) {
            LWSCI_ERR_STR("failed to allocate memory.\n");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        u64Mul(newIpcTable->ipcRouteEntries, sizeof(LwSciIpcEndpoint),
                &arrSize, &opStatus);
        if (OP_SUCCESS != opStatus) {
            LWSCI_ERR_STR("Unexpected overflow of size used for allocation");
            LwSciCommonPanic();
        }
        LwSciCommonMemcpyS(newIpcTable->ipcRoute, arrSize,
                ipcTable->ipcRoute, arrSize);

        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        newIpcTable->topoIds = (LwSciSyncIpcTopoId*)LwSciCommonCalloc(
                newIpcTable->ipcRouteEntries, sizeof(LwSciSyncIpcTopoId));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        if (NULL == newIpcTable->topoIds) {
            LWSCI_ERR_STR("failed to allocate memory.");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_ipcroute;
        }
        u64Mul(newIpcTable->ipcRouteEntries, sizeof(LwSciSyncIpcTopoId),
                &arrSize, &opStatus);
        if (OP_SUCCESS != opStatus) {
            LWSCI_ERR_STR("Unexpected overflow of size used for allocation");
            LwSciCommonPanic();
        }
        LwSciCommonMemcpyS(newIpcTable->topoIds, arrSize,
                ipcTable->topoIds, arrSize);
    }

    /* Copy perm tree */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    newIpcTable->ipcPermEntries = ipcTable->ipcPermEntries;
    if (0U != newIpcTable->ipcPermEntries) {
        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        newIpcTable->ipcPerm = (LwSciSyncCoreAttrIpcPerm*)LwSciCommonCalloc(
                newIpcTable->ipcPermEntries,
                sizeof(LwSciSyncCoreAttrIpcPerm));
        LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
        if (NULL == newIpcTable->ipcPerm) {
            LWSCI_ERR_STR("failed to allocate memory.\n");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_topoids;
        }
    }
    for (i = 0U; i < newIpcTable->ipcPermEntries; i++) {
        newIpcTable->ipcPerm[i].ipcRouteEntries =
                ipcTable->ipcPerm[i].ipcRouteEntries;
        newIpcTable->ipcPerm[i].needCpuAccess =
                ipcTable->ipcPerm[i].needCpuAccess;
        newIpcTable->ipcPerm[i].waiterRequireTimestamps =
                ipcTable->ipcPerm[i].waiterRequireTimestamps;
        newIpcTable->ipcPerm[i].requiredPerm =
                ipcTable->ipcPerm[i].requiredPerm;
        LwSciCommonMemcpyS(newIpcTable->ipcPerm[i].engineArray,
                sizeof(newIpcTable->ipcPerm[i].engineArray),
                ipcTable->ipcPerm[i].engineArray,
                ipcTable->ipcPerm[i].engineArrayLen * sizeof(ipcTable->ipcPerm[i].engineArray[0]));
        newIpcTable->ipcPerm[i].engineArrayLen =
                ipcTable->ipcPerm[i].engineArrayLen;
        if (0U != newIpcTable->ipcPerm[i].ipcRouteEntries) {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
            newIpcTable->ipcPerm[i].ipcRoute =
                    (LwSciIpcEndpoint*)LwSciCommonCalloc(
                    newIpcTable->ipcPerm[i].ipcRouteEntries,
                    sizeof(LwSciIpcEndpoint));
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            if (NULL == newIpcTable->ipcPerm[i].ipcRoute) {
                LWSCI_ERR_STR("failed to allocate memory.\n");
                error = LwSciError_InsufficientMemory;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_ipcperm;
            }
            u64Mul(newIpcTable->ipcPerm[i].ipcRouteEntries,
                   sizeof(LwSciIpcEndpoint),
                   &arrSize, &opStatus);
            if (OP_SUCCESS != opStatus) {
                LWSCI_ERR_STR("Unexpected overflow of size used for allocation");
                LwSciCommonPanic();
            }
            LwSciCommonMemcpyS(newIpcTable->ipcPerm[i].ipcRoute, arrSize,
                    ipcTable->ipcPerm[i].ipcRoute, arrSize);

                        LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
            newIpcTable->ipcPerm[i].topoIds =
                    (LwSciSyncIpcTopoId*)LwSciCommonCalloc(
                    newIpcTable->ipcPerm[i].ipcRouteEntries,
                    sizeof(LwSciSyncIpcTopoId));
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            if (NULL == newIpcTable->ipcPerm[i].topoIds) {
                LWSCI_ERR_STR("failed to allocate memory.");
                error = LwSciError_InsufficientMemory;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_ipcperm;
            }
            u64Mul(newIpcTable->ipcPerm[i].ipcRouteEntries,
                   sizeof(LwSciSyncIpcTopoId),
                   &arrSize, &opStatus);
            if (OP_SUCCESS != opStatus) {
                LWSCI_ERR_STR("Unexpected overflow of size used for allocation");
                LwSciCommonPanic();
            }
            LwSciCommonMemcpyS(newIpcTable->ipcPerm[i].topoIds, arrSize,
                    ipcTable->ipcPerm[i].topoIds, arrSize);
        }
    }
free_ipcperm:
    if (LwSciError_Success != error) {
        for (i = 0U; i < newIpcTable->ipcPermEntries; i++) {
            LwSciCommonFree(newIpcTable->ipcPerm[i].ipcRoute);
        }
        LwSciCommonFree(newIpcTable->ipcPerm);
    }
free_topoids:
    if (LwSciError_Success != error) {
        LwSciCommonFree(newIpcTable->topoIds);
    }
free_ipcroute:
    if (LwSciError_Success != error) {
        LwSciCommonFree(newIpcTable->ipcRoute);
    }
fn_exit:
    return error;
}

void LwSciSyncCoreIpcTableLwtSubTree(
    LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    bool* needCpuAccess,
    bool* waiterRequireTimestamps,
    LwSciSyncAccessPerm* requiredPerm,
    LwSciSyncHwEngine* engineArray,
    size_t bufLen,
    size_t* engineArrayLen)
{
    size_t i;
    size_t lastEntry;
    size_t tmp1 = 0U;
    size_t tmp2 = 0U;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!");
        LwSciCommonPanic();
    }
    if (NULL == needCpuAccess) {
        LWSCI_ERR_STR("Null needCpuAccess. Panicking!!");
        LwSciCommonPanic();
    }
    if (NULL == waiterRequireTimestamps) {
        LWSCI_ERR_STR("Null waiterRequireTimestamps. Panicking!!");
        LwSciCommonPanic();
    }
    if (NULL == requiredPerm) {
        LWSCI_ERR_STR("Null requiredPerm. Panicking!!");
        LwSciCommonPanic();
    }
    if (NULL == engineArray) {
        LWSCI_ERR_STR("Null engineArray. Panicking!!");
        LwSciCommonPanic();
    }
    if (NULL == engineArrayLen) {
        LWSCI_ERR_STR("Null engineArrayLen. Panicking!!");
        LwSciCommonPanic();
    }

    *needCpuAccess = false;
    *waiterRequireTimestamps = false;
    *requiredPerm = 0U;
    *engineArrayLen = 0U;

    i = 0U;
    while (i < ipcTable->ipcPermEntries) {
        if (0U == ipcTable->ipcPerm[i].ipcRouteEntries) {
            LWSCI_ERR_STR("Unexpected empty ipc perm branch");
            LwSciCommonPanic();
        }

        lastEntry = ipcTable->ipcPerm[i].ipcRouteEntries - 1U;
        if (ipcEndpoint == ipcTable->ipcPerm[i].ipcRoute[lastEntry]) {
            /* Remove the last Ipc route entry */
            ipcTable->ipcPerm[i].ipcRouteEntries -= 1U;
            /* Exported list contains cumulative perms of all unreconciled attr
             * list imported through this ipcEndpoint */
            *needCpuAccess = (*needCpuAccess) ||
                             (ipcTable->ipcPerm[i].needCpuAccess);
            if (0U == lastEntry) {
                *waiterRequireTimestamps = (*waiterRequireTimestamps) ||
                        (ipcTable->ipcPerm[i].waiterRequireTimestamps);

                /* It's possible for multiple LwSciSyncAttrList's belonging to
                 * a peer to be present. As such, we need to combine all such
                 * lists for the EngineArray key. */
                LwSciSyncAppendHwEngineToArrayUnion(
                    engineArray, bufLen,
                    ipcTable->ipcPerm[i].engineArray,
                    ipcTable->ipcPerm[i].engineArrayLen,
                    engineArrayLen);
            }

            tmp1 = (size_t)(ipcTable->ipcPerm[i].requiredPerm);
            tmp2 = (size_t)(*requiredPerm);
            tmp1 |= tmp2;
            LwSciCommonMemcpyS(requiredPerm, sizeof(*requiredPerm),
                                      &tmp1, sizeof(tmp1));
            i++;
        } else {
            /* Removes unrelated paths */
            ipcTable->ipcPermEntries--;
            MoveIpcPermEntry(&ipcTable->ipcPerm[i],
                    &ipcTable->ipcPerm[ipcTable->ipcPermEntries]);
        }
    }
}

LwSciError LwSciSyncCoreIpcTableGetPermAtSubTree(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAccessPerm* perm)
{
    LwSciError error = LwSciError_Success;
    size_t lastEntry;
    size_t i;
    size_t tmp1 = 0U;
    size_t tmp2 = 0U;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == perm) {
        LWSCI_ERR_STR("Null perm. Panicking!!\n");
        LwSciCommonPanic();
    }

    /* Set default to value zero */
    LwSciCommonMemcpyS(perm, sizeof(*perm), &tmp1, sizeof(tmp1));
    for (i = 0U; i < ipcTable->ipcPermEntries; i++) {
        if (0U == ipcTable->ipcPerm[i].ipcRouteEntries) {
            LWSCI_ERR_STR("Unexpected empty ipc perm branch");
            LwSciCommonPanic();
        }

        lastEntry = ipcTable->ipcPerm[i].ipcRouteEntries - 1U;
        if (ipcEndpoint == ipcTable->ipcPerm[i].ipcRoute[lastEntry]) {
            /* Collect the cumulative perms of all unreconciled attr
             * list imported through this ipcEndpoint */
            tmp1 = (size_t)(ipcTable->ipcPerm[i].requiredPerm);
            tmp2 = (size_t)(*perm);
            tmp1 |= tmp2;
            LwSciCommonMemcpyS(perm, sizeof(*perm), &tmp1, sizeof(tmp1));
        }
    }
    if (0U == (uint64_t)*perm) {
        error = LwSciError_BadParameter;
    }

    return error;
}

LwSciError LwSciSyncCoreIpcTableGetEngineArrayAtSubTree(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncHwEngine* engineArray,
    size_t bufLen,
    size_t* engineArrayLen)
{
    LwSciError error = LwSciError_Success;
    size_t i;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!");
        LwSciCommonPanic();
    }
    if (NULL == engineArray) {
        LWSCI_ERR_STR("Null engineArray. Panicking!!");
        LwSciCommonPanic();
    }
    if (NULL == engineArrayLen) {
        LWSCI_ERR_STR("Null engineArrayLen. Panicking!!");
        LwSciCommonPanic();
    }

    (void)memset(engineArray, 0, sizeof(*engineArray) * bufLen);
    *engineArrayLen = 0U;

    for (i = 0U; i < ipcTable->ipcPermEntries; i++) {
        size_t lastEntry;

        if (0U == ipcTable->ipcPerm[i].ipcRouteEntries) {
            LWSCI_ERR_STR("Unexpected empty ipc perm branch");
            LwSciCommonPanic();
        }

        lastEntry = ipcTable->ipcPerm[i].ipcRouteEntries - 1U;
        if (ipcEndpoint == ipcTable->ipcPerm[i].ipcRoute[lastEntry]) {
            LwSciSyncAppendHwEngineToArrayUnion(
                engineArray, bufLen,
                ipcTable->ipcPerm[i].engineArray,
                ipcTable->ipcPerm[i].engineArrayLen,
                engineArrayLen);
        }
    }

    return error;
}

void LwSciSyncCoreIpcTableGetRequireTimestampsAtSubTree(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    bool* waiterRequireTimestamps)
{
    size_t lastEntry;
    size_t i;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == waiterRequireTimestamps) {
        LWSCI_ERR_STR("Null waiterRequireTimestamps. Panicking!!\n");
        LwSciCommonPanic();
    }

    *waiterRequireTimestamps = false;
    for (i = 0U; i < ipcTable->ipcPermEntries; i++) {
        lastEntry = ipcTable->ipcPerm[i].ipcRouteEntries - 1U;
        if (ipcEndpoint == ipcTable->ipcPerm[i].ipcRoute[lastEntry]) {
            *waiterRequireTimestamps = (*waiterRequireTimestamps) ||
                    (ipcTable->ipcPerm[i].waiterRequireTimestamps);
        }
    }
}

void LwSciSyncCoreIpcTableGetRequireTimestampsSum(
    const LwSciSyncCoreIpcTable* ipcTable,
    bool* waiterRequireTimestamps)
{
    size_t i;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == waiterRequireTimestamps) {
        LWSCI_ERR_STR("Null waiterRequireTimestamps. Panicking!!\n");
        LwSciCommonPanic();
    }

    *waiterRequireTimestamps = false;
    for (i = 0U; i < ipcTable->ipcPermEntries; i++) {
        *waiterRequireTimestamps = (*waiterRequireTimestamps) ||
                (ipcTable->ipcPerm[i].waiterRequireTimestamps);
    }
}

LwSciError LwSciSyncCoreIpcTableAppend(
    LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint)
{
    LwSciError error = LwSciError_Success;
    uint8_t subStatus = OP_FAIL;
    size_t index = 0U;
    LwSciSyncIpcTopoId topoId = {0};

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }

    error = LwSciSyncCoreGetSyncTopoId(ipcEndpoint, &topoId);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("getting topoid for an LwSciIpcEndpoint failed.");
        goto fn_exit;
    }

    /** Set the last entry to current IpcEndpoint */
    if (0U == ipcTable->ipcRouteEntries) {
        ipcTable->ipcRouteEntries = 1U;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        ipcTable->ipcRoute = (LwSciIpcEndpoint*)LwSciCommonCalloc(
                ipcTable->ipcRouteEntries, sizeof(LwSciIpcEndpoint));
        if (NULL == ipcTable->ipcRoute) {
            LWSCI_ERR_STR("failed to allocate memory.\n");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        ipcTable->topoIds = (LwSciSyncIpcTopoId*)LwSciCommonCalloc(
                ipcTable->ipcRouteEntries, sizeof(LwSciSyncIpcTopoId));
        if (NULL == ipcTable->topoIds) {
            LWSCI_ERR_STR("failed to allocate memory.\n");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_ipcroute;
        }
    }

    /* The subtraction won't fail since ipcRouteEntries won't be less than one */
    u64Sub((ipcTable->ipcRouteEntries), 1U, &index, &subStatus);
    if (OP_FAIL == subStatus) {
        LWSCI_ERR_STR("Subtraction underflow. Panicking!!\n");
        LwSciCommonPanic();
    }

    ipcTable->ipcRoute[index] = ipcEndpoint;
    ipcTable->topoIds[index] = topoId;

free_ipcroute:
    if (LwSciError_Success != error) {
        LwSciCommonFree(ipcTable->ipcRoute);
    }
fn_exit:
    return error;
}

LwSciError LwSciSyncCoreExportIpcTable(
    const LwSciSyncCoreIpcTable* ipcTable,
    void** txbufPtr,
    size_t* txbufSize)
{
    LwSciError error = LwSciError_Success;
    uint64_t valSize = 0U;
    size_t i = 0U;
    LwSciCommonTransportParams bufparams = {0};
    LwSciCommonTransportBuf* txbuf = NULL;
    size_t ipcPermEntries = 0U;
    void** ipcPermTxbufPtr = NULL;
    size_t* ipcPermBufSize = NULL;
    uint32_t ipcRouteExportKeys = 0U;
    uint64_t ipcRouteExportSize = 0U;
    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;

    if (NULL == ipcTable) {
        LWSCI_ERR_STR("Null ipcTable. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == txbufPtr) {
        LWSCI_ERR_STR("Null txbufPtr. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == txbufSize) {
        LWSCI_ERR_STR("Null txbufSize. Panicking!!\n");
        LwSciCommonPanic();
    }

    error = ExportIpcPerm(ipcTable, &bufparams, &valSize,
            &ipcPermEntries, &ipcPermTxbufPtr, &ipcPermBufSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Add ipcRoute info */
    error = GetIpcRouteKeyCntAndValSize(ipcTable,
                    &ipcRouteExportKeys,
                    &ipcRouteExportSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ipcPermTxBuf;
    }
    u32Add(bufparams.keyCount, ipcRouteExportKeys,
           &(bufparams.keyCount), &addStatus1);
    u64Add(valSize, ipcRouteExportSize,
           &valSize, &addStatus2);
    if ((OP_SUCCESS != addStatus1) || (OP_SUCCESS != addStatus2)) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ipcPermTxBuf;
    }

    /* if nothing to export, just return */
    if (0U == valSize) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ipcPermTxBuf;
    }

    /** Create attr list desc */
    error = LwSciCommonTransportAllocTxBufferForKeys(bufparams, valSize,
            &txbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ipcPermTxBuf;
    }

    error = ExportIpcRoute(ipcTable->ipcRoute, ipcTable->topoIds,
            ipcTable->ipcRouteEntries, txbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }

    error = AppendKeys(txbuf, ipcTable, ipcPermEntries,
                    ipcPermTxbufPtr, ipcPermBufSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }

    LwSciCommonTransportPrepareBufferForTx(txbuf,
            txbufPtr, txbufSize);

free_txbuf:
    LwSciCommonTransportBufferFree(txbuf);
free_ipcPermTxBuf:
    for (i = 0U; i < ipcPermEntries; i++) {
        LwSciCommonFree(ipcPermTxbufPtr[i]);
    }
    LwSciCommonFree(ipcPermTxbufPtr);
    LwSciCommonFree(ipcPermBufSize);
fn_exit:
    return error;
}

bool LwSciSyncCoreIpcTableHasC2C(
    const LwSciSyncCoreIpcTable* ipcTable)
{
    size_t i = 0U;
    size_t j = 0U;
    bool result = false;
    LwSciSyncCoreAttrIpcPerm* ipcPerm = NULL;

    for (i = 0U; i < ipcTable->ipcRouteEntries; i++) {
        if (LwSciSyncCoreIsTopoIdC2c(ipcTable->topoIds[i].topoId)) {
            result = true;
            goto end;
        }
    }

    for (i = 0U; i < ipcTable->ipcPermEntries; ++i) {
        ipcPerm = &ipcTable->ipcPerm[i];
        for (j = 0U; j < ipcPerm->ipcRouteEntries; ++j) {
            if (LwSciSyncCoreIsTopoIdC2c(ipcPerm->topoIds[j].topoId)) {
                result = true;
                goto end;
            }
        }
    }

end:
    return result;
}

static LwSciError ExportIpcRoute(
    const LwSciIpcEndpoint* ipcRoute,
    const LwSciSyncIpcTopoId* topoIds,
    size_t ipcRouteEntries,
    LwSciCommonTransportBuf* txbuf)
{
    LwSciError error = LwSciError_Success;
    size_t len = 0U;
    const void* value = NULL;
    uint32_t key = 0U;

    if (ipcRouteEntries > 0U) {
        /** Export num Ipc route entries */
        len = sizeof(ipcRouteEntries);
        value = (const void*)&ipcRouteEntries;
        key = (uint32_t)LwSciSyncCoreIpcTableKey_NumIpcEndpoint;
        error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len, value);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        /** Export Ipc route info */
        len = sizeof(LwSciIpcEndpoint) * ipcRouteEntries;
        value = (const void*)ipcRoute;
        key = (uint32_t)LwSciSyncCoreIpcTableKey_IpcEndpoints;
        error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len, value);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        /** Export topoIds */
        len = sizeof(LwSciSyncIpcTopoId) * ipcRouteEntries;
        value = (const void*)topoIds;
        key = (uint32_t)LwSciSyncCoreIpcTableKey_TopoIds;
        error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len, value);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
fn_exit:
    return error;
}

static LwSciError ExportIpcPermEntry(
    const LwSciSyncCoreAttrIpcPerm* ipcPerm,
    void** txbufPtr,
    size_t* txbufSize)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportParams bufparams = {0};
    LwSciCommonTransportBuf* txbuf = NULL;
    uint32_t keyCnt = 0U;
    uint64_t valSize = 0U;
    const void* value = NULL;
    size_t len = 0U;
    uint32_t key = 0U;
    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;
    uint8_t addStatus3 = OP_FAIL;

    /** Append IpcEndpoint key and value size info */
    keyCnt += 1U;
    valSize += ipcPerm->ipcRouteEntries * sizeof(LwSciIpcEndpoint);
    /** Append TopoIds key and value size info */
    keyCnt += 1U;
    valSize += ipcPerm->ipcRouteEntries * sizeof(LwSciSyncIpcTopoId);
    /** Append num IpcEndpoint key and value size info */
    keyCnt += 1U;
    u64Add(valSize, sizeof(ipcPerm->ipcRouteEntries), &valSize, &addStatus1);
    /** Append attr list type key and value size info */
    keyCnt += 1U;
    u64Add(valSize, sizeof(ipcPerm->needCpuAccess), &valSize, &addStatus2);
    keyCnt += 1U;
    u64Add(valSize, sizeof(ipcPerm->waiterRequireTimestamps), &valSize, &addStatus2);
    keyCnt += 1U;
    u64Add(valSize, sizeof(ipcPerm->requiredPerm), &valSize, &addStatus3);
    if (OP_SUCCESS != (addStatus1 & addStatus2 & addStatus3)) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Create attr list desc */
    bufparams.keyCount = keyCnt;
    error = LwSciCommonTransportAllocTxBufferForKeys(bufparams, valSize,
            &txbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    /** Export attr list type info */
    len = sizeof(ipcPerm->needCpuAccess);
    value = (const void*)&ipcPerm->needCpuAccess;
    key = (uint32_t)LwSciSyncCoreIpcTableKey_NeedCpuAccess;
    error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len,
            value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }
    /** Export attr list type info */
    len = sizeof(ipcPerm->waiterRequireTimestamps);
    value = (const void*)&ipcPerm->waiterRequireTimestamps;
    key = (uint32_t)LwSciSyncCoreIpcTableKey_WaiterRequireTimestamps;
    error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len,
            value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }
    /** Export attr list type info */
    len = sizeof(ipcPerm->requiredPerm);
    value = (const void*)&ipcPerm->requiredPerm;
    key = (uint32_t)LwSciSyncCoreIpcTableKey_RequiredPerm;
    error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len,
            value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }
    /** Export Ipc route info */
    error = ExportIpcRoute(ipcPerm->ipcRoute,
            ipcPerm->topoIds,
            ipcPerm->ipcRouteEntries, txbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }

    LwSciCommonTransportPrepareBufferForTx(txbuf,
            txbufPtr, txbufSize);


free_txbuf:
    LwSciCommonTransportBufferFree(txbuf);
fn_exit:
    return error;

}

static LwSciError GetIpcRouteKeyCntAndValSize(
    const LwSciSyncCoreIpcTable* ipcTable,
    uint32_t* keyCnt,
    uint64_t* valSize)
{
    LwSciError error = LwSciError_Success;
    uint8_t addStatus = OP_FAIL;
    uint8_t mulStatus = OP_FAIL;
    uint64_t topoIdSize = 0U;

    /** Start empty */
    *valSize = 0U;
    *keyCnt = 0U;

    if (ipcTable->ipcRouteEntries > 0U) {
        /** Append IpcEndpoint key and value size info */
        *keyCnt = 1U;
        *valSize = ipcTable->ipcRouteEntries * sizeof(LwSciIpcEndpoint);
        /** Append TopoIds key and value size info */
        *keyCnt += 1U;
        u64Mul(ipcTable->ipcRouteEntries, sizeof(LwSciSyncIpcTopoId),
               &topoIdSize, &mulStatus);
        if (OP_SUCCESS != mulStatus) {
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        u64Add((*valSize), topoIdSize,
               valSize, &addStatus);
        if (OP_SUCCESS != addStatus) {
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        /** Append num IpcEndpoint key and value size info */
        *keyCnt += 1U;
        u64Add((*valSize), sizeof(ipcTable->ipcRouteEntries),
               valSize, &addStatus);
        if (OP_SUCCESS != addStatus) {
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    return error;
}

struct IpcPermTagInfo {
    uint32_t tag;
    uint32_t expectedNum;
    uint32_t handledNum;
};

static struct IpcPermTagInfo* FindIpcPermTagInfo(
    uint32_t key,
    struct IpcPermTagInfo* tagInfo,
    size_t tagsNum)
{
    size_t i = 0U;
    struct IpcPermTagInfo* result = NULL;

    for (i = 0U; i < tagsNum; ++i) {
        if (key == tagInfo[i].tag) {
            result = &tagInfo[i];
            break;
        }
    }

    return result;
}

static LwSciError ImportIpcPermEntry(
    LwSciSyncCoreAttrIpcPerm* ipcPerm,
    const void* inputValue,
    size_t length)
{
    LwSciError error = LwSciError_Success;
    LwSciCommonTransportBuf* ipcPermRxbuf = NULL;
    LwSciSyncCoreIpcTableKey ipcTableKey;
    bool doneReading = true;
    LwSciCommonTransportParams params = {0};
    size_t expectedSize = 0U;
    uint8_t arithStatus = OP_FAIL;
    struct IpcPermTagInfo* info = NULL;
    struct IpcPermTagInfo* tmpInfo = NULL;
    struct IpcPermTagInfo tagInfo[] = {
        {(uint32_t)LwSciSyncCoreIpcTableKey_NeedCpuAccess, 1U, 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_WaiterRequireTimestamps, 1U, 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_RequiredPerm, 1U, 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_NumIpcEndpoint, 1U, 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_IpcEndpoints, 0U, 0U},
        {(uint32_t)LwSciSyncCoreIpcTableKey_TopoIds, 0U, 0U},
    };
    size_t numTags = sizeof(tagInfo) / sizeof(struct IpcPermTagInfo);
    size_t lastIndex = 0U;
    size_t i = 0U;

    LWSCI_FNENTRY("");

    error = LwSciCommonTransportGetRxBufferAndParams(inputValue, length,
            &ipcPermRxbuf, &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    do {
        uint32_t key = 0U;
        const void* value = NULL;

        error = LwSciCommonTransportGetNextKeyValuePair(ipcPermRxbuf,
                &key, &length, &value, &doneReading);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_ipcroute;
        }
        info = FindIpcPermTagInfo(key, tagInfo, numTags);
        if (NULL == info) {
            LWSCI_INFO("Unrecognized key under IpcPermEntry %u\n", key);
            continue;
        }
        if (info->handledNum >= info->expectedNum) {
            LWSCI_ERR_UINT("Tag is not allowed here in IpcTable: \n", key);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_ipcroute;
        }

        LwSciCommonMemcpyS(&ipcTableKey, sizeof(ipcTableKey), &key, sizeof(key));
        switch (ipcTableKey) {
            case LwSciSyncCoreIpcTableKey_NeedCpuAccess:
            {
                if (sizeof(bool) != length) {
                    LWSCI_ERR_UINT("Invalid size for _NeedCpuAccess key in IpcPerm: \n",
                              length);
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
                LwSciCommonMemcpyS(&ipcPerm->needCpuAccess, sizeof(bool),
                        value, length);
                break;
            }
            case LwSciSyncCoreIpcTableKey_WaiterRequireTimestamps:
            {
                if (sizeof(bool) != length) {
                    LWSCI_ERR_UINT("Invalid size for _WaiterRequireTimestamps key in IpcPerm: \n",
                              length);
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                ipcPerm->waiterRequireTimestamps = *(const bool*)value;
                break;
            }
            case LwSciSyncCoreIpcTableKey_RequiredPerm:
            {
                if (sizeof(LwSciSyncAccessPerm) != length) {
                    LWSCI_ERR_UINT("Invalid size for _RequiredPerm key in IpcPerm: \n",
                              length);
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
                LwSciCommonMemcpyS(&ipcPerm->requiredPerm, sizeof(LwSciSyncAccessPerm),
                        value, length);
                if (!LwSciSyncCorePermValid(ipcPerm->requiredPerm)) {
                    LWSCI_ERR_UINT("Invalid value for _RequiredPerm key in IpcPerm: ",
                              ipcPerm->requiredPerm);
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                break;
            }
            case LwSciSyncCoreIpcTableKey_NumIpcEndpoint:
            {
                if (sizeof(size_t) != length) {
                    LWSCI_ERR_UINT("Invalid size of _NumIpcEndpoint key in IpcPerm: \n",
                            length);
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
                ipcPerm->ipcRouteEntries = *(const size_t*)value;
                if (0U == ipcPerm->ipcRouteEntries) {
                    LWSCI_ERR_STR("invalid value of _NumIpcEndpoint - it cannot be 0");
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }

                ipcPerm->ipcRoute = (LwSciIpcEndpoint*)LwSciCommonCalloc(
                        ipcPerm->ipcRouteEntries,
                        sizeof(LwSciIpcEndpoint));
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
                if (NULL == ipcPerm->ipcRoute) {
                    LWSCI_ERR_STR("failed to allocate memory.\n");
                    error = LwSciError_InsufficientMemory;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                tmpInfo = FindIpcPermTagInfo(
                        (uint32_t)LwSciSyncCoreIpcTableKey_IpcEndpoints,
                        tagInfo, numTags);
                if (NULL == tmpInfo) {
                    LWSCI_ERR_STR("Couldn't find a tag known to exist\n");
                    LwSciCommonPanic();
                }
                tmpInfo->expectedNum = 1U;
                ipcPerm->topoIds = (LwSciSyncIpcTopoId*)LwSciCommonCalloc(
                        ipcPerm->ipcRouteEntries,
                        sizeof(LwSciSyncIpcTopoId));
                LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
                if (NULL == ipcPerm->topoIds) {
                    LWSCI_ERR_STR("failed to allocate memory.");
                    error = LwSciError_InsufficientMemory;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                tmpInfo = FindIpcPermTagInfo(
                        (uint32_t)LwSciSyncCoreIpcTableKey_TopoIds,
                        tagInfo, numTags);
                if (NULL == tmpInfo) {
                    LWSCI_ERR_STR("Couldn't find a tag known to exist");
                    LwSciCommonPanic();
                }
                tmpInfo->expectedNum = 1U;
                break;
            }
            case LwSciSyncCoreIpcTableKey_IpcEndpoints:
            {
                u64Mul((ipcPerm->ipcRouteEntries), sizeof(LwSciIpcEndpoint),
                       &expectedSize, &arithStatus);
                if (OP_SUCCESS != arithStatus) {
                    LWSCI_ERR_STR("expected size value is out of range.\n");
                    LwSciCommonPanic();
                }
                if (length != expectedSize) {
                    LWSCI_ERR_STR("_IpcEndpoints size different than promised\n");
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                LwSciCommonMemcpyS(ipcPerm->ipcRoute, expectedSize,
                        value, length);
                break;
            }
            case LwSciSyncCoreIpcTableKey_TopoIds:
            {
                u64Mul((ipcPerm->ipcRouteEntries), sizeof(LwSciSyncIpcTopoId),
                       &expectedSize, &arithStatus);
                if (OP_SUCCESS != arithStatus) {
                    LWSCI_ERR_STR("expected size value is out of range.");
                    LwSciCommonPanic();
                }
                if (length != expectedSize) {
                    LWSCI_ERR_STR("_TopoIds size different than promised");
                    error = LwSciError_BadParameter;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto free_ipcroute;
                }
                LwSciCommonMemcpyS(ipcPerm->topoIds, expectedSize,
                        value, length);
                break;
            }
            default:
            {
                LWSCI_ERR_STR("Impossible to reach an unknown tag\n");
                LwSciCommonPanic();
                break;
            }
        }
        info->handledNum++;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    } while (false == doneReading);

    for (i = 0U; i < numTags; ++i) {
        if (tagInfo[i].expectedNum != tagInfo[i].handledNum) {
            LWSCI_ERR_UINT("IpcTable descriptor is missing tag ",
                      tagInfo[i].tag);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_ipcroute;
        }
    }

    /* Last IpcEndpoint is local, so it can be validated */
    u64Sub((ipcPerm->ipcRouteEntries), 1U, &lastIndex, &arithStatus);
    if (OP_FAIL == arithStatus) {
        LWSCI_ERR_STR("Subtraction underflow. Panicking!!\n");
        LwSciCommonPanic();
    }
    error = ValidateLocalIpcEndpoint(ipcPerm->ipcRoute[lastIndex],
                                     ipcPerm->topoIds[lastIndex]);

free_ipcroute:
    if (LwSciError_Success != error) {
        LwSciCommonFree(ipcPerm->topoIds);
        LwSciCommonFree(ipcPerm->ipcRoute);
    }

    LwSciCommonTransportBufferFree(ipcPermRxbuf);
fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_5), "LwSciSync-ADV-MISRAC2012-017")
static LwSciError ExportIpcPerm(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciCommonTransportParams* bufparams,
    uint64_t* totalBufSize,
    size_t* ipcPermEntriesPtr,
    void*** txbufPtr,
    size_t** bufSize)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_5))
{
    LwSciError error = LwSciError_Success;
    size_t ipcPermEntries = 0U;
    size_t ipcPermIdx = 0U;
    size_t i = 0U;
    void** ipcPermTxbufPtr = NULL;
    size_t* ipcPermBufSize = NULL;
    uint64_t valSize = 0U;
    uint8_t addStatus1 = OP_FAIL;
    uint8_t addStatus2 = OP_FAIL;
    uint8_t addStatus3 = OP_FAIL;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    for (i = 0U; i < ipcTable->ipcPermEntries; i++) {
        if (ipcTable->ipcPerm[i].ipcRouteEntries > 0U) {
            /* The loop iterates at most SIZE_MAX times, won't overflow
             * ipcPermEntries */
            u64Add(ipcPermEntries, 1U, &ipcPermEntries, &addStatus1);
            if (OP_SUCCESS != addStatus1) {
                LWSCI_ERR_STR("Arithmetic overflow!\n");
                LwSciCommonPanic();
            }
        }
    }

    if (ipcPermEntries > 0U) {
        ipcPermTxbufPtr = (void**)LwSciCommonCalloc(
            ipcPermEntries, sizeof(void*));
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        ipcPermBufSize = (size_t*)LwSciCommonCalloc(
            ipcPermEntries, sizeof(size_t));
        if ((NULL == ipcPermTxbufPtr) ||
                (NULL == ipcPermBufSize)) {
            LWSCI_ERR_STR("Failed to allocate memory\n");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
        for (i = 0U; i < ipcTable->ipcPermEntries; i++) {
            if (ipcTable->ipcPerm[i].ipcRouteEntries > 0U) {
                error = ExportIpcPermEntry(&ipcTable->ipcPerm[i],
                        &ipcPermTxbufPtr[ipcPermIdx],
                        &ipcPermBufSize[ipcPermIdx]);
                if (LwSciError_Success != error) {
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto fn_exit;
                }
                u32Add(bufparams->keyCount, 1U,
                       &(bufparams->keyCount), &addStatus1);
                u64Add(valSize, ipcPermBufSize[ipcPermIdx],
                       &valSize, &addStatus2);
                u64Add(ipcPermIdx, 1U, &ipcPermIdx, &addStatus3);
                if (OP_SUCCESS != (addStatus1 & addStatus2)) {
                    error = LwSciError_Overflow;
                    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                    goto fn_exit;
                }
                /* ipcPermIdx should not overflow since it is of size_t type
                 * and the loop iterates at most SIZE_MAX times. */
                if (OP_SUCCESS != addStatus3) {
                    LWSCI_ERR_STR("Arithmetic overflow!\n");
                    LwSciCommonPanic();
                }
            }
        }

        /** Add key-value for num IPC perm entries */
        u32Add(bufparams->keyCount, 1U,
               &(bufparams->keyCount), &addStatus1);
        u64Add(valSize, sizeof(ipcTable->ipcPermEntries),
               &valSize, &addStatus2);
        if (OP_SUCCESS != (addStatus1 & addStatus2)) {
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
    *txbufPtr = ipcPermTxbufPtr;
    *bufSize = ipcPermBufSize;
    u64Add((*totalBufSize), valSize, totalBufSize, &addStatus1);
    if (OP_SUCCESS != addStatus1) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *ipcPermEntriesPtr = ipcPermEntries;

fn_exit:
    if (LwSciError_Success != error) {
        LwSciCommonFree(ipcPermBufSize);
        LwSciCommonFree(ipcPermTxbufPtr);
    }
    return error;
}

static LwSciError AppendKeys(
    LwSciCommonTransportBuf* txbuf,
    const LwSciSyncCoreIpcTable* ipcTable,
    size_t ipcPermEntries,
    void* const* ipcPermTxbufPtr,
    const size_t* ipcPermBufSize)
{
    LwSciError error = LwSciError_Success;
    uint32_t key = 0U;
    const void* value = NULL;
    size_t len = 0U;
    size_t i = 0U;

    if (ipcTable->ipcPermEntries > 0U) {
        len = sizeof(size_t);
        value = (const void*)&ipcPermEntries;
        key = (uint32_t)LwSciSyncCoreIpcTableKey_NumIpcPerm;
        error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len, value);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        for (i = 0U; i < ipcPermEntries; i++) {
            len = ipcPermBufSize[i];
            value = (const void*)ipcPermTxbufPtr[i];
            key = (uint32_t)LwSciSyncCoreIpcTableKey_IpcPermEntry;
            error = LwSciCommonTransportAppendKeyValuePair(txbuf, key,
                    len, value);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }
    }
fn_exit:
    return error;
}

static void MoveIpcPermEntry(
    LwSciSyncCoreAttrIpcPerm* dest,
    const LwSciSyncCoreAttrIpcPerm* src)
{
    LwSciCommonFree(dest->ipcRoute);
    LwSciCommonFree(dest->topoIds);

    if (dest != src) {
        LwSciCommonMemcpyS(dest, sizeof(LwSciSyncCoreAttrIpcPerm),
                src, sizeof(LwSciSyncCoreAttrIpcPerm));
    }
}

static LwSciError ValidateLocalIpcEndpoint(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncIpcTopoId allegedTopoId)
{
    LwSciSyncIpcTopoId actualTopoId = {0};
    LwSciError error = LwSciError_Success;

    error = LwSciSyncCoreGetSyncTopoId(ipcEndpoint, &actualTopoId);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

    if (allegedTopoId.topoId.SocId != actualTopoId.topoId.SocId ||
            allegedTopoId.topoId.VmId != actualTopoId.topoId.VmId ||
            allegedTopoId.vuId != actualTopoId.vuId) {
        LWSCI_ERR_STR("topoId information of a local LwSciIpcEndpoint does not match"
                      " the imported topoId information");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fail;
    }

fail:
    return error;
}
