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
 * \brief <b>LwSciSync Syncpoint related Implementation</b>
 *
 * @b Description: Implements the syncpoint related APIs
 */
#include <errno.h>

#ifdef LW_TEGRA_MIRROR_INCLUDES
//cheetah build from perforce tree - use mobile_common.h
#include "mobile_common.h"
#else
//cheetah build from git tree - use lwrm_host1x_safe.h
#include "lwrm_host1x_safe.h"
#endif
#include "lwscicommon_libc.h"
#include "lwscicommon_utils.h"
#include "lwscicommon_os.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core.h"
#ifdef LWSCISYNC_EMU_SUPPORT
#include "lwscisync_attribute_core_cluster.h"
#endif
#include "lwscisync_backend_tegra.h"
#include "lwscisync_cpu_wait_context.h"
#include "lwscisync_module.h"
#include "lwscisync_primitive_core.h"
#include "lwscisync_syncpoint_core.h"


#define LWSCISYNC_ILWALID_SYNCPOINT_ID LWRM_ILWALID_SYNCPOINT_ID

/**
 * \brief Tags for export descriptor
 */
typedef enum {
    /** for initializing */
    LwSciSyncCoreSyncpointKey_Ilwalid,
    /** (uint64_t[]) */
    LwSciSyncCoreSyncpointKey_Ids,
    /** (whatever the token's type is) */
    LwSciSyncCoreSyncpointKey_C2CAuthToken,
} LwSciSyncCoreSyncpointKey;

static LwSciError LwSciSyncCoreSyncpointExport(
    LwSciSyncCorePrimitive primitive,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** data,
    size_t* length)
{
    LwSciError error = LwSciError_Success;
    size_t totalSize = 0U;
#if (LW_IS_SAFETY == 0)
    const void* value = NULL;
    size_t len = 0U;
    uint32_t key = 0U;
    size_t size = 0U;
    uint8_t arithStatus = OP_FAIL;
    LwSciSyncCoreSyncpointInfo* info =
        (LwSciSyncCoreSyncpointInfo*) primitive->specificData;
    LwSciSyncIpcTopoId syncTopoId = {0};
    LwSciCommonTransportParams bufparams = {0};
    LwSciCommonTransportBuf* txbuf = NULL;

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciC2cPcieAuthToken authToken = 0U;
    LwSciC2cPcieSyncRmHandle syncRmHandle = {info->syncpt};
    int err = 0;
#endif

    bufparams.keyCount += 1U;
    u64Mul(info->numIds, sizeof(uint64_t), &size, &arithStatus);
    if (OP_SUCCESS != arithStatus) {
        LWSCI_ERR_STR("arithmetic overflow during export");
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (LwSciSyncCorePermLEq(LwSciSyncAccessPerm_SignalOnly, permissions)) {
        error = LwSciSyncCoreGetSyncTopoId(ipcEndpoint, &syncTopoId);
        if (LwSciError_Success != error) {
            LWSCI_ERR_INT("Something went wrong with LwSciIpc ", error);
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        if (!LwSciSyncCoreIsTopoIdC2c(syncTopoId.topoId)) {
            LWSCI_ERR_STR("exporting syncpoints with signaling is only allowed"
                          " via C2C");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        if (NULL == info->syncpt) {
            LWSCI_ERR_STR("trying to export signaling without owning a syncpoint");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014");
            goto fn_exit;
        }

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        err = LwSciIpcGetC2cCopyFuncSet(ipcEndpoint, &info->c2cCopyFuncs);
        if (0 != err) {
            error = LwSciError_ResourceError;
            LWSCI_ERR_STR("LwSciIpcGetC2cCopyFuncSet failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto fn_exit;
        }

        err = info->c2cCopyFuncs.syncMapRemoteMemHandle(
            &syncRmHandle, ipcEndpoint, &info->syncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("mapping remote mem handle failed.");
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014");
            goto fn_exit;
        }
        err = info->c2cCopyFuncs.syncGetAuthTokenFromHandle(
            info->syncHandle, ipcEndpoint, &authToken);
        if (0 != err) {
            if (EAGAIN == -err) {
                LWSCI_WARN("getting auth token from handle failed. User should retry the operation.");
                error = LwSciError_TryItAgain;
            } else {
                LWSCI_ERR_STR("getting auth token from handle failed");
                error = LwSciError_ResourceError;
            }
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014");
            goto free_syncHandle;
        }
#else /* L4T */
        error = LwSciError_NotSupported;
        LWSCI_ERR_STR("Exporting signaling over C2C not supported on C2C");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014");
        goto fn_exit;
#endif
    }

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    bufparams.keyCount += 1U;
    u64Add(sizeof(authToken), size, &totalSize, &arithStatus);
    if (OP_SUCCESS != arithStatus) {
        LWSCI_ERR_STR("arithmetic overflow during export");
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_syncHandle;
    }
#else
    totalSize = size;
#endif

    error = LwSciCommonTransportAllocTxBufferForKeys(
        bufparams, totalSize, &txbuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_syncHandle;
    }

    len = size;
    value = (const void*)info->ids;
    key = (uint32_t)LwSciSyncCoreSyncpointKey_Ids;
    error = LwSciCommonTransportAppendKeyValuePair(txbuf, key, len,
            value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    len = sizeof(authToken);
    value = &authToken;
    key = (uint32_t)LwSciSyncCoreSyncpointKey_C2CAuthToken;
    error = LwSciCommonTransportAppendKeyValuePair(
        txbuf, key, len, value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_txbuf;
    }
#endif

    LwSciCommonTransportPrepareBufferForTx(txbuf,
            data, length);

    LwSciCommonTransportBufferFree(txbuf);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
    goto fn_exit;

free_txbuf:
    LwSciCommonTransportBufferFree(txbuf);
free_syncHandle:
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    if (NULL != info->syncHandle) {
        err = info->c2cCopyFuncs.syncFreeHandle(info->syncHandle);
        if (0 != err) {
            LWSCI_ERR_INT("error in error path when freeing sync handle: ", err);
        }
    }
#endif
fn_exit:
#else /* SAFETY */
    (void)permissions;
    (void)ipcEndpoint;
    (void)primitive;

    /* Nothing to export which is syncpoint specific */
    *data = NULL;
    *length = totalSize;
#endif /* end of SAFETY */
    return error;
}

static LwSciError LwSciSyncCoreSyncpointImport(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList reconciledList,
    const void* data,
    size_t len,
    LwSciSyncCorePrimitive primitive)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncModule module = NULL;
    LwSciSyncCoreRmBackEnd backEnd = NULL;
    LwRmHost1xHandle host1x = NULL;
    LwSciSyncCoreSyncpointInfo* info = NULL;
#if (LW_IS_SAFETY == 0)
    LwSciSyncInternalAttrKey attrKey =
        LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo;
    const void* attrValue = NULL;
    size_t attrLength = 0U;
    uint32_t key = LwSciSyncCoreSyncpointKey_Ilwalid;
    const void* value = NULL;
    size_t length = 0U;
    LwSciCommonTransportBuf* ipcPermRxbuf = NULL;
    LwSciCommonTransportParams params = {0};
    bool doneReading = true;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciSyncIpcTopoId syncTopoId = {0};
    LwSciSyncAttrKeyValuePair publicAttrs[] = {
        {
            .attrKey = LwSciSyncAttrKey_ActualPerm,
        },
        {
            .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
        },
    };
    LwSciC2cPcieSyncHandle syncHandle = NULL;
    int err = 0;
#endif
#endif

    (void)ipcEndpoint;
    (void)data;
    (void)len;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    info = LwSciCommonCalloc(1U, sizeof(LwSciSyncCoreSyncpointInfo));
    if (NULL == info) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncCoreAttrListGetModule(reconciledList, &module);

    LwSciSyncCoreModuleGetRmBackEnd(module, &backEnd);

    host1x = LwSciSyncCoreRmGetHost1xHandle(backEnd);

    info->host1x = host1x;
    info->syncpt = NULL;
    primitive->specificData = (void*) info;

#if (LW_IS_SAFETY == 0)
    error = LwSciCommonTransportGetRxBufferAndParams(data, len,
            &ipcPermRxbuf, &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_info;
    }

    error = LwSciCommonTransportGetNextKeyValuePair(ipcPermRxbuf,
            &key, &length, &value, &doneReading);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rxbuf;
    }
    if ((uint32_t) LwSciSyncCoreSyncpointKey_Ids != key) {
        LWSCI_ERR_INT("expected LwSciSyncCoreSyncpointKey_Ids but encountered ", key);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rxbuf;
    }

    info->numIds = length;
    if (0 != info->numIds % sizeof(uint64_t)) {
        LWSCI_ERR_ULONG("value of LwSciSyncCoreSyncpointKey_Ids should be"
                        " divisible by sizeof(uint64_t) ", info->numIds);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_rxbuf;
    }

    info->numIds /= sizeof(uint64_t);
    info->ids = LwSciCommonCalloc(info->numIds, sizeof(uint64_t));
    if (NULL == info->ids) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        goto free_rxbuf;
    }
    LwSciCommonMemcpyS(info->ids, length, value, length);

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    if (doneReading) {
        LWSCI_ERR_STR("too few tags");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ids;
    }

    error = LwSciCommonTransportGetNextKeyValuePair(ipcPermRxbuf,
            &key, &length, &value, &doneReading);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ids;
    }
    if ((uint32_t) LwSciSyncCoreSyncpointKey_C2CAuthToken != key) {
        LWSCI_ERR_INT("expected LwSciSyncCoreSyncpointKey_C2CAuthToken"
                      " but encountered ", key);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ids;
    }

    if (sizeof(LwSciC2cPcieAuthToken) != length) {
        LWSCI_ERR_ULONG("invalid length of LwSciSyncCoreSyncpointKey_C2CAuthToken ",
                        length);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_ids;
    }
#endif /* end of (LW_IS_SAFETY == 0) && (LW_L4T == 0) */

    /* ignore any additional tags */

    error = LwSciSyncAttrListGetSingleInternalAttr(reconciledList, attrKey, &attrValue,
            &attrLength);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("This call isn't expected to fail\n");
        LwSciCommonPanic();
    }
    primitive->hasExternalPrimitiveInfo = (NULL != attrValue);

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    error = LwSciSyncAttrListGetAttrs(
        reconciledList, publicAttrs,
        sizeof(publicAttrs)/sizeof(publicAttrs[0]));
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("This call isn't expected to fail\n");
        LwSciCommonPanic();
    }

    if (LwSciSyncCorePermLEq(
            LwSciSyncAccessPerm_SignalOnly,
            *(const LwSciSyncAccessPerm*)publicAttrs[0].value)) {
        error = LwSciSyncCoreGetSyncTopoId(ipcEndpoint, &syncTopoId);
        if (LwSciError_Success != error) {
            goto free_ids;
        }
        if (!LwSciSyncCoreIsTopoIdC2c(syncTopoId.topoId)) {
            LWSCI_ERR_STR("signaling rights for a syncpoint can only"
                          " be transferred via C2C");
            error = LwSciError_BadParameter;
            goto free_ids;
        }

        err = LwSciIpcGetC2cCopyFuncSet(ipcEndpoint, &info->c2cCopyFuncs);
        if (0 != err) {
            error = LwSciError_ResourceError;
            LWSCI_ERR_STR("LwSciIpcGetC2cCopyFuncSet failed.");
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
            goto free_ids;
        }

        err = info->c2cCopyFuncs.syncGetHandleFromAuthToken(
            *(const LwSciC2cPcieAuthToken*) value, ipcEndpoint,
            &syncHandle);
        if (0 != err) {
            if (EAGAIN == -err) {
                error = LwSciError_TryItAgain;
                LWSCI_ERR_STR("getting sync handle from auth token failed. User should retry the operation.");
            } else {
                error = LwSciError_ResourceError;
                LWSCI_ERR_STR("getting sync handle from auth token failed");
            }
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014");
            goto free_ids;
        }
        info->syncHandle = syncHandle;

        if (*(const bool*)publicAttrs[1].value) {
            err = info->c2cCopyFuncs.syncCreateCpuMapping(
                syncHandle, (void**)&info->memShim);
            if (0 != err) {
                error = LwSciError_ResourceError;
                LWSCI_ERR_STR("creating cpu mapping failed");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014");
                goto free_syncHandle;
            }
        }
    }

free_syncHandle:
    if (NULL != info->syncHandle &&
        LwSciError_Success != error) {
        err = info->c2cCopyFuncs.syncFreeHandle(info->syncHandle);
        if (0 != err) {
            LWSCI_ERR_INT("error in error path when freeing sync handle: ", err);
        }
    }

free_ids:
#endif /* end of (LW_IS_SAFETY == 0) && (LW_L4T == 0) */
    if (LwSciError_Success != error) {
        LwSciCommonFree(info->ids);
    }

free_rxbuf:
    LwSciCommonTransportBufferFree(ipcPermRxbuf);

free_info:
#endif /* end of LW_IS_SAFETY == 0 */
    if (LwSciError_Success != error) {
        LwSciCommonFree(info);
    }

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreSyncpointInit(
    LwSciSyncAttrList reconciledList,
    LwSciSyncCorePrimitive primitive)
{
    LwSciError error = LwSciError_Success;
    LwError lwErr;
    uint32_t syncptId = 0U;
    uint32_t syncptValue = 0U;
    LwSciSyncModule module = NULL;
    LwSciSyncCoreRmBackEnd backEnd = NULL;
    LwRmHost1xSyncpointAllocateAttrs syncptAllocAttrs;
    LwSciSyncCoreSyncpointInfo* info = NULL;
#ifdef LWSCISYNC_EMU_SUPPORT
    const void* value = NULL;
    size_t len = 0U;
    const LwSciSyncPrimitiveInfo* externalPrimitiveInfo = NULL;

    info = LwSciCommonCalloc(1U, sizeof(LwSciSyncCoreSyncpointInfo));
    if (info == NULL) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

    error = LwSciSyncAttrListGetSingleInternalAttr(reconciledList,
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount, &value,
            &len);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }
    info->numIds = *(const size_t*) value;
#endif

    if (primitive->ownsPrimitive) {
#ifndef LWSCISYNC_EMU_SUPPORT
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
        info = LwSciCommonCalloc(1U, sizeof(LwSciSyncCoreSyncpointInfo));
        if (NULL == info) {
            LWSCI_ERR_STR("Failed to allocate memory\n");
            error = LwSciError_InsufficientMemory;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
#endif

        /* Get the host1x handle*/
        LwSciSyncCoreAttrListGetModule(reconciledList, &module);

        LwSciSyncCoreModuleGetRmBackEnd(module, &backEnd);

        info->host1x = LwSciSyncCoreRmGetHost1xHandle(backEnd);

        /* Reserve Syncpoint */
        syncptAllocAttrs = LwRmHost1xGetDefaultSyncpointAllocateAttrs();
        lwErr = LwRmHost1xSyncpointAllocate(&info->syncpt, info->host1x,
                syncptAllocAttrs);
        if (LwError_Success != lwErr) {
            error = LwSciError_ResourceError;
            LWSCI_ERR_INT("Failed to reserve Syncpoint. LwError: \n", lwErr);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        syncptId = LwRmHost1xSyncpointGetId(info->syncpt);
        lwErr = LwRmHost1xSyncpointRead(info->host1x, syncptId, &syncptValue);
        if (LwError_Success != lwErr) {
            error = LwSciError_ResourceError;
            LWSCI_ERR_INT("Failed to read Syncpoint value. LwError: \n", lwErr);
            LwRmHost1xSyncpointFree(info->syncpt);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        primitive->id = syncptId;
        primitive->lastFence = (uint64_t) syncptValue;
#ifdef LWSCISYNC_EMU_SUPPORT
        info->ids = LwSciCommonCalloc(info->numIds, sizeof(uint64_t));
        if (info->ids == NULL) {
            LWSCI_ERR_STR("Failed to allocate memory\n");
            error = LwSciError_InsufficientMemory;
            goto fn_exit;
        }
        info->ids[0] = syncptId;
    } else {
        error = LwSciSyncAttrListGetSingleInternalAttr(reconciledList,
                LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo, &value,
                &len);
        if (error != LwSciError_Success) {
            LWSCI_ERR_STR("This call isn't expected to fail\n");
            LwSciCommonPanic();
        }
        if (value != NULL) {
            info->ids = LwSciCommonCalloc(info->numIds, sizeof(uint64_t));
            if (info->ids == NULL) {
                LWSCI_ERR_STR("Failed to allocate memory\n");
                error = LwSciError_InsufficientMemory;
                goto fn_exit;
            }
            externalPrimitiveInfo = (const LwSciSyncPrimitiveInfo*)
                    (*(const uintptr_t*)value);
            LwSciCommonMemcpyS(info->ids, (sizeof(uint64_t) * info->numIds),
                    externalPrimitiveInfo->simplePrimitiveInfo.ids,
                    (sizeof(uint64_t) * info->numIds));
        } else {
            /* in this case we don't know the ids, so only allocate
               a token buffer */
            info->numIds = 1U;
            info->ids = LwSciCommonCalloc(info->numIds, sizeof(uint64_t));
            if (info->ids == NULL) {
                LWSCI_ERR_STR("Failed to allocate memory\n");
                error = LwSciError_InsufficientMemory;
                goto fn_exit;
            }
            info->ids[0] = LWSCISYNC_ILWALID_SYNCPOINT_ID;
        }
        primitive->hasExternalPrimitiveInfo = (value != NULL);
#endif
    }
    primitive->specificData = (void*) info;

fn_exit:
    if (LwSciError_Success != error) {
        LwSciCommonFree(info);
    }
    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static void LwSciSyncCoreSyncpointDeinit(
    LwSciSyncCorePrimitive primitive)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwRmHost1xSyncpointHandle syncpt;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),"LwSciSync-ADV-MISRAC2012-013")
    LwSciSyncCoreSyncpointInfo* info =
        (LwSciSyncCoreSyncpointInfo*) primitive->specificData;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if (NULL != info) {
        syncpt = info->syncpt;
        /* Free reserved Syncpoint */
        if (primitive->ownsPrimitive) {
            LwRmHost1xSyncpointFree(syncpt);
        }
#if (LW_IS_SAFETY == 0)
        LwSciCommonFree(info->ids);
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
        if (NULL != info->syncHandle) {
            int err = 0;
            if (NULL != info->memShim) {
                err = info->c2cCopyFuncs.syncDeleteCpuMapping(
                    info->syncHandle, info->memShim);
                if (0 != err) {
                    LWSCI_ERR_STR("C2C error on deleting cpu mapping");
                    LwSciCommonPanic();
                }
            }

            err = info->c2cCopyFuncs.syncFreeHandle(info->syncHandle);
            if (0 != err) {
                LWSCI_ERR_STR("C2C error on freeing syncHandle");
                LwSciCommonPanic();
            }
        }
#endif /* LW_IS_SAFETY == 0 && LW_L4T == 0 */
#endif
        LwSciCommonFree(info);
    }
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static LwSciError LwSciSyncCoreSyncpointSignal(
    LwSciSyncCorePrimitive primitive)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LwSciError error = LwSciError_Success;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    const LwSciSyncCoreSyncpointInfo* info =
        (const LwSciSyncCoreSyncpointInfo*)primitive->specificData;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    if (NULL != info->syncpt) {
        LwError lwErr = LwSuccess;
        lwErr = LwRmHost1xSyncpointIncrement(info->syncpt, 1);
        if (lwErr != LwSuccess) {
            LWSCI_ERR_INT("Failed to signal syncpoint because "
                          "LwRmHost1xSyncpointIncrement failed. LwError: ",
                          lwErr);
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    } else if (NULL != info->memShim) {
        const uint32_t shimIncrement = 1U;
        *info->memShim = shimIncrement;
#endif
    } else {
        LWSCI_ERR_STR("This primitive cannot be signaled");
        LwSciCommonPanic();
    }

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreSyncpointCheckIdValue(
    LwSciSyncCorePrimitive primitive,
    uint64_t id,
    uint64_t value)
{
    LwSciError error = LwSciError_Success;
#ifndef LWSCISYNC_EMU_SUPPORT
    (void)primitive;
#else
    size_t i = 0U;
    LwSciSyncCoreSyncpointInfo* info =
        (LwSciSyncCoreSyncpointInfo*) primitive->specificData;

    if (primitive->hasExternalPrimitiveInfo) {
        for (i = 0U; i < info->numIds; i++) {
            if (id == info->ids[i]) {
                break;
            }
        }
        if (i == info->numIds) {
            LWSCI_ERR_STR("Invalid id\n");
            error = LwSciError_Overflow;
            goto fn_exit;
        }
    } else {
#endif
        if (id >= LWSCISYNC_ILWALID_SYNCPOINT_ID) {
            LWSCI_ERR_STR("Invalid id\n");
            error = LwSciError_Overflow;
            goto fn_exit;
        }
#ifdef LWSCISYNC_EMU_SUPPORT
    }
#endif

    if (value > UINT32_MAX) {
        LWSCI_ERR_STR("Invalid value\n");
        error = LwSciError_Overflow;
        goto fn_exit;
    }

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreSyncpointWaitOn(
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
    LwSciSyncCorePrimitive primitive,
    LwSciSyncCpuWaitContext waitContext,
    uint64_t id,
    uint64_t value,
    int64_t timeout_us)
{
    LwSciError error = LwSciError_Success;
    LwError lwErr;
    uint64_t timeout_us_u64 = 0ULL;
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd = NULL;
    LwRmHost1xWaiterHandle waiterHandle = NULL;
    uint32_t u32Value = 0U;
    uint32_t u32Id = 0U;

    waitContextBackEnd = LwSciSyncCoreCpuWaitContextGetBackEnd(waitContext);
    waiterHandle = LwSciSyncCoreRmWaitCtxGetWaiterHandle(
            waitContextBackEnd);

    timeout_us_u64 = (timeout_us == -1) ?
            LWRMHOST1X_MAX_WAIT : (uint64_t)timeout_us;

    error = LwSciSyncCoreSyncpointCheckIdValue(primitive, id, value);
    if (error != LwSciError_Success) {
         goto fn_exit;
    }

    u32Id = (uint32_t)id;
    u32Value = (uint32_t)value;

    lwErr = LwRmHost1xSyncpointWait(waiterHandle, u32Id,
            u32Value, timeout_us_u64, NULL);
    if (LwError_Timeout == lwErr) {
        error = LwSciError_Timeout;
        LWSCI_ERR_INT("LwRmFenceWait timed out. LwError: \n", lwErr);
    } else if (LwError_Success != lwErr) {
        error = LwSciError_ResourceError;
        LWSCI_ERR_INT("LwRmFenceWait failed. LwError: \n", lwErr);
    } else {
        error = LwSciError_Success;
    }

fn_exit:
    return error;
}

static uint64_t LwSciSyncCoreSyncpointGetNewFence(
    LwSciSyncCorePrimitive primitive)
{
    uint8_t addStatus = OP_FAIL;
    u64Add(primitive->lastFence, 1U, &(primitive->lastFence), &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("primitive->lastFence value is out of range.\n");
        LwSciCommonPanic();
    }
    return (primitive->lastFence) & UINT32_MAX;
}

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
static LwSciError LwSciSyncCoreSyncpointGetC2cSyncHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncHandle* syncHandle)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSyncpointInfo* info =
        (LwSciSyncCoreSyncpointInfo*) primitive->specificData;

    if (NULL == info->syncHandle) {
        /* no print as this error could be used for informative purposes */
        error = LwSciError_NotInitialized;
    } else {
        *syncHandle = info->syncHandle;
    }

    return error;
}

static LwSciError LwSciSyncCoreSyncpointGetC2cRmHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncRmHandle* syncRmHandle)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSyncpointInfo* info =
        (LwSciSyncCoreSyncpointInfo*) primitive->specificData;

    if (NULL == info->syncpt) {
        /* no print as this error could be used for informative purposes */
        error = LwSciError_NotInitialized;
    } else {
        syncRmHandle->syncPoint = info->syncpt;
    }

    return error;
}
#endif

static LwSciError LwSciSyncCoreSyncpointImportThreshold(
        LwSciSyncCorePrimitive primitive,
        uint64_t* threshold)
{
    LwSciError error = LwSciError_Success;
    /* guaranteed to be in uint32_t range for syncpoint */
    uint64_t base = primitive->lastFence;

    if (base > UINT32_MAX) {
        LWSCI_ERR_ULONG("Impossible value stored in primitive for syncpoint ", base);
        LwSciCommonPanic();
    }

    if (*threshold > UINT32_MAX) {
        LWSCI_ERR_ULONG("Fence value invalid: ", *threshold);
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

    *threshold = ((*threshold) + base) & UINT32_MAX;

fn_exit:
    return error;
}

const LwSciSyncPrimitiveOps LwSciSyncBackEndSyncpoint =
{
    .Init = LwSciSyncCoreSyncpointInit,
    .Deinit = LwSciSyncCoreSyncpointDeinit,
    .Export = LwSciSyncCoreSyncpointExport,
    .Import = LwSciSyncCoreSyncpointImport,
    .Signal = LwSciSyncCoreSyncpointSignal,
    .WaitOn = LwSciSyncCoreSyncpointWaitOn,
    .GetNewFence = LwSciSyncCoreSyncpointGetNewFence,
    .GetSpecificData = NULL,
    .CheckIdValue = LwSciSyncCoreSyncpointCheckIdValue,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    .GetC2cSyncHandle = LwSciSyncCoreSyncpointGetC2cSyncHandle,
    .GetC2cRmHandle = LwSciSyncCoreSyncpointGetC2cRmHandle,
#endif
    .ImportThreshold = LwSciSyncCoreSyncpointImportThreshold,
};
