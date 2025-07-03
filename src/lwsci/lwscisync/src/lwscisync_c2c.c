/*
* Copyright (c) 2021-2022, LWPU CORPORATION.  All rights reserved.
*
* LWPU CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU CORPORATION is strictly prohibited.
*/

#include "lwscibuf_c2c_internal.h"

#include "lwscicommon_covanalysis.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"

#include "lwscilog.h"

#include "lwscisync_core.h"
#include "lwscisync_object_external.h"
#include "lwscisync_c2c_priv.h"

#include "lwsciipc_internal.h"

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)

#define LWSCISYNC_C2C_SYNC_HANDLE_MAGIC (0xBBFF0385U)

static inline LwSciSyncHwEngNamespace getEngNamespace(
    void)
{
#if (defined(__x86_64__))
    return LwSciSyncHwEngine_ResmanNamespaceId;
#else
    return LwSciSyncHwEngine_TegraNamespaceId;
#endif
}

static LwSciSyncInternalAttrValPrimitiveType getWaiterPrimitiveType(void) {
#if (defined(__x86_64__))
    return LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
#else
    return LwSciSyncInternalAttrValPrimitiveType_Syncpoint;
#endif
}

struct LwSciC2cSyncHandleRec {
    uint64_t magic;
    LwSciC2cHandle channelHandle;
    LwSciC2cInterfaceSyncHandle syncHandle;
    LwSciSyncObj syncObj;
    LwSciSyncAccessPerm permissions;
};

static LwSciError ValidateSyncRegistration(
    LwSciC2cHandle channelHandle,
    LwSciSyncObj syncObj,
    const LwSciC2cSyncHandle* syncHandle,
    LwSciSyncAccessPerm expectedPerm,
    LwSciSyncHwEngName expectedEngineName)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncAttrList attrList = NULL;
    LwSciSyncAccessPerm* permissions = NULL;
    size_t len = 0U;
    LwSciSyncHwEngine* engines = NULL;
    LwSciSyncHwEngName engineName = LwSciSyncHwEngName_LowerBound;

    if ((NULL == channelHandle) || (NULL == syncObj) ||
            (NULL == syncHandle)) {
        LWSCI_ERR_STR("NULL parameters supplied to registration");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (LWSCIBUF_C2C_CHANNEL_MAGIC != channelHandle->magic) {
        LWSCI_ERR_STR("Invalid channelHandle.");
        LwSciCommonPanic();
    }

    error = LwSciSyncObjGetAttrList(syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncAttrListGetAttr(
        attrList, LwSciSyncAttrKey_ActualPerm,
        (const void**)&permissions, &len);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (0U == ((uint64_t)expectedPerm & (uint64_t)(*permissions))) {
        LWSCI_ERR_INT("Input syncObj has wrong permissions ", *permissions);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncAttrListGetSingleInternalAttr(
        attrList, LwSciSyncInternalAttrKey_EngineArray,
        (const void**)&engines, &len);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if ((NULL == engines) || (0U == len)) {
        LWSCI_ERR_STR("No engines associated with this syncObj");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (sizeof(LwSciSyncHwEngine) != len) {
        LWSCI_ERR_STR("Too many engines associated with this syncObj");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncHwEngGetNameFromId(engines[0].rmModuleID, &engineName);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (engineName != expectedEngineName) {
        LWSCI_ERR_INT("syncObj is associated with unrecognized engine ",
                      engineName);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LwSciError LwSciSyncFillAttrsIndirectChannelC2c(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList unrecAttrList,
    LwSciSyncAccessPerm permissions)
{
    LwSciError error = LwSciError_Success;

    LwSciSyncInternalAttrValPrimitiveType signalerPrimitiveInfo[] =
        { LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
          LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};
    LwSciSyncInternalAttrValPrimitiveType waiterPrimitiveInfo[] =
        { getWaiterPrimitiveType() };
    uint32_t signalerPrimitiveCount = 1U;
    LwSciSyncHwEngine engines[] =
        {
            {
                .engNamespace = getEngNamespace(),
                /* .rmModuleId is initialized dynamically below */
            }
        };
    LwSciSyncInternalAttrKeyValuePair internalKeyValuePairs[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
             .value = (void*) signalerPrimitiveInfo,
             .len = sizeof(signalerPrimitiveInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
             .value = (void*)&signalerPrimitiveCount,
             .len = sizeof(signalerPrimitiveCount),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
             .value = (void*) waiterPrimitiveInfo,
             .len = sizeof(waiterPrimitiveInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_EngineArray,
             .value = (void*) engines,
             .len = sizeof(engines),
        },
    };
    LwSciSyncAttrKeyValuePair publicKeyValuePairs[] = {
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &permissions,
             .len = sizeof(permissions),
        },
    };

    /* ipcEndpoint should be used in the future to determine
       which engine to set in the EngineArray */
    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error =  LwSciSyncAttrListSetAttrs(unrecAttrList, publicKeyValuePairs,
        sizeof(publicKeyValuePairs)/sizeof(LwSciSyncAttrKeyValuePair));
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncHwEngCreateIdWithoutInstance(
        LwSciSyncHwEngName_PCIe, &engines[0].rmModuleID);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncAttrListSetInternalAttrs(
        unrecAttrList, internalKeyValuePairs,
        sizeof(internalKeyValuePairs) /
            sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LwSciError LwSciSyncRegisterWaitObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciSyncObj syncObj,
    LwSciC2cSyncHandle* syncHandle)
{
    LwSciError error = LwSciError_Success;
    LwSciC2cSyncHandle tmpSyncHandle = NULL;
    int err = 0;
    LwSciC2cInterfaceSyncHandle tmpC2cSyncHandle = {0};

    LWSCI_FNENTRY("");

    error = ValidateSyncRegistration(
        channelHandle, syncObj, syncHandle,
        LwSciSyncAccessPerm_WaitOnly, LwSciSyncHwEngName_PCIe);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* actual operations */

    tmpSyncHandle = LwSciCommonCalloc(1U, sizeof(*tmpSyncHandle));
    if (NULL == tmpSyncHandle) {
        LWSCI_ERR_STR("Insufficient memory");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncObjGetC2cSyncHandle(
        syncObj, &tmpC2cSyncHandle);
    if (LwSciError_NotInitialized == error) { /* engineWritesDoneObj - local */
        err = channelHandle->copyFuncs.syncCreateLocalHandle(
            channelHandle->interfaceHandle.pcieStreamHandle,
            &tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("Failure in creating a local syncHandle");
            error = LwSciError_ResourceError;
            goto free_syncHandleMem;
        }

        err = channelHandle->copyFuncs.syncRegisterLocalHandle(
            channelHandle->interfaceHandle.pcieStreamHandle,
            tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("failed to register local handle");
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_syncHandle;
        }
    } else if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_syncHandleMem;
    } else { /* consReadsDoneProdObj - remote */
        err = channelHandle->copyFuncs.syncDupRemoteHandle(
            tmpC2cSyncHandle.pcieSyncHandle,
            &tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("failed to duplicate remote handle");
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_syncHandleMem;
        }

        err = channelHandle->copyFuncs.syncRegisterRemoteHandle(
            channelHandle->interfaceHandle.pcieStreamHandle,
            tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("failed to register remote handle");
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_syncHandle;
        }
    }

    error = LwSciSyncObjRef(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_syncHandle;
    }

    tmpSyncHandle->syncObj = syncObj;
    tmpSyncHandle->channelHandle = channelHandle;
    tmpSyncHandle->magic = LWSCISYNC_C2C_SYNC_HANDLE_MAGIC;
    tmpSyncHandle->permissions = LwSciSyncAccessPerm_WaitOnly;

    *syncHandle = tmpSyncHandle;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
    goto fn_exit;

free_syncHandle:
    err = channelHandle->copyFuncs.syncFreeHandle(
        tmpSyncHandle->syncHandle.pcieSyncHandle);
    if (0 != err) {
        LWSCI_ERR_STR("another error during freeing syncHandle");
    }
free_syncHandleMem:
    LwSciCommonFree(tmpSyncHandle);
fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncRegisterSignalObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciSyncObj syncObj,
    LwSciC2cSyncHandle* syncHandle)
{
    LwSciError error = LwSciError_Success;
    LwSciC2cSyncHandle tmpSyncHandle = NULL;
    int err = 0;
    LwSciC2cPcieSyncRmHandle syncRmHandle;
    LwSciC2cInterfaceSyncHandle tmpC2cSyncHandle = {0};
    memset(&syncRmHandle, 0U, sizeof(LwSciC2cPcieSyncRmHandle));

    LWSCI_FNENTRY("");

    error = ValidateSyncRegistration(
        channelHandle, syncObj, syncHandle,
        LwSciSyncAccessPerm_SignalOnly, LwSciSyncHwEngName_PCIe);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /* actual operations */

    tmpSyncHandle = LwSciCommonCalloc(1U, sizeof(*tmpSyncHandle));
    if (NULL == tmpSyncHandle) {
        LWSCI_ERR_STR("Insufficient memory");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncObjGetC2cSyncHandle(
        syncObj, &tmpC2cSyncHandle);
    if (LwSciError_NotInitialized == error) { /* copyDoneProdObj - local */
        error = LwSciSyncCoreObjGetC2cRmHandle(
            syncObj, &syncRmHandle);
        if (LwSciError_Success != error) {
            goto free_syncHandleMem;
        }
        err = channelHandle->copyFuncs.syncMapLocalMemHandle(
            &syncRmHandle,
            channelHandle->interfaceHandle.pcieStreamHandle,
            &tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("Failure in mapping local signaler syncHandle");
            error = LwSciError_ResourceError;
            goto free_syncHandleMem;
        }

        err = channelHandle->copyFuncs.syncRegisterLocalHandle(
            channelHandle->interfaceHandle.pcieStreamHandle,
            tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("failed to register local handle");
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_syncHandle;
        }
    } else if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_syncHandleMem;
    } else { /* copyDoneConsObj - remote */
        err = channelHandle->copyFuncs.syncDupRemoteHandle(
            tmpC2cSyncHandle.pcieSyncHandle,
            &tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("failed to duplicate remote handle");
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_syncHandleMem;
        }

        err = channelHandle->copyFuncs.syncRegisterRemoteHandle(
            channelHandle->interfaceHandle.pcieStreamHandle,
            tmpSyncHandle->syncHandle.pcieSyncHandle);
        if (0 != err) {
            LWSCI_ERR_STR("failed to register remote handle");
            error = LwSciError_ResourceError;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_syncHandle;
        }
    }

    error = LwSciSyncObjRef(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_syncHandle;
    }

    tmpSyncHandle->syncObj = syncObj;
    tmpSyncHandle->channelHandle = channelHandle;
    tmpSyncHandle->magic = LWSCISYNC_C2C_SYNC_HANDLE_MAGIC;
    tmpSyncHandle->permissions = LwSciSyncAccessPerm_SignalOnly;

    *syncHandle = tmpSyncHandle;

     LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
     goto fn_exit;

free_syncHandle:
    err = channelHandle->copyFuncs.syncFreeHandle(
        tmpSyncHandle->syncHandle.pcieSyncHandle);
    if (0 != err) {
        LWSCI_ERR_STR("another error during freeing syncHandle");
    }
free_syncHandleMem:
    LwSciCommonFree(tmpSyncHandle);
fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

static LwSciError ValidateSyncPush(
    LwSciC2cHandle channelHandle,
    LwSciC2cSyncHandle syncHandle,
    const LwSciSyncFence* fence,
    LwSciSyncAccessPerm expectedPerm)
{
    LwSciError error = LwSciError_Success;

    if ((NULL == channelHandle) || (NULL == syncHandle) ||
            (NULL == fence)) {
        LWSCI_ERR_STR("NULL parameters supplied to push");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (LWSCIBUF_C2C_CHANNEL_MAGIC != channelHandle->magic) {
        LWSCI_ERR_STR("Invalid channelHandle.");
        LwSciCommonPanic();
    }

    if (LWSCISYNC_C2C_SYNC_HANDLE_MAGIC != syncHandle->magic) {
        LWSCI_ERR_STR("Invalid syncHandle.");
        LwSciCommonPanic();
    }

    if (syncHandle->channelHandle != channelHandle) {
        LWSCI_ERR_STR("This syncHandle is not associated with this"
                      " channelHandle");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (expectedPerm != syncHandle->permissions) {
        LWSCI_ERR_STR("This syncHandle was registered for"
                      " a different operation");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

LwSciError LwSciBufPushWaitIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cSyncHandle syncHandle,
    const LwSciSyncFence* preFence)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncObj syncObj;
    uint64_t id = 0U;
    uint64_t value = 0U;
    int err = 0;

    LWSCI_FNENTRY("");

    error = ValidateSyncPush(
        channelHandle, syncHandle, preFence, LwSciSyncAccessPerm_WaitOnly);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncFenceGetSyncObj(preFence, &syncObj);
    if (LwSciError_ClearedFence == error) {
        /* cleared fences do not block */
        error = LwSciError_Success;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    } else if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    } else {
        /* success case continued after this "if else if" */
    }

    if (syncObj != syncHandle->syncObj) {
        LWSCI_ERR_STR("The input preFence mismatches the input syncHandle");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncFenceExtractFence(preFence, &id, &value);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    err = channelHandle->copyFuncs.pushWaitIndirectChannel(
        channelHandle->interfaceHandle.pcieStreamHandle,
        syncHandle->syncHandle.pcieSyncHandle,
        id, value);
    if (0 != err) {
        LWSCI_ERR_STR("pushing wait command failed");
        error = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciBufPushSignalIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cSyncHandle syncHandle,
    LwSciSyncFence* postFence)
{
    LwSciError error = LwSciError_Success;
    int err = 0;

    LWSCI_FNENTRY("");

    error = ValidateSyncPush(
        channelHandle, syncHandle, postFence, LwSciSyncAccessPerm_SignalOnly);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncObjGenerateFence(
        syncHandle->syncObj, postFence);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    err = channelHandle->copyFuncs.pushSignalIndirectChannel(
        channelHandle->interfaceHandle.pcieStreamHandle,
        syncHandle->syncHandle.pcieSyncHandle);
    if (0 != err) {
        LWSCI_ERR_STR("pushing signal command failed");
        error = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncFreeObjIndirectChannelC2c(
    LwSciC2cSyncHandle syncHandle)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncObj syncObj = NULL;
    int err = 0;

    LWSCI_FNENTRY("");

    if (NULL == syncHandle) {
        LWSCI_ERR_STR("NULL syncHandle.");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (LWSCISYNC_C2C_SYNC_HANDLE_MAGIC != syncHandle->magic) {
        LWSCI_ERR_STR("Invalid syncHandle.");
        LwSciCommonPanic();
    }

    syncObj = syncHandle->syncObj;

    err = syncHandle->channelHandle->copyFuncs.syncFreeHandle(
        syncHandle->syncHandle.pcieSyncHandle);
    if (0 != err) {
        LWSCI_ERR_STR("error during freeing pcie syncHandle");
        error = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncObjFree(syncObj);

    LwSciCommonFree(syncHandle);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}
#else

LwSciError LwSciSyncFillAttrsIndirectChannelC2c(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList unrecAttrList,
    LwSciSyncAccessPerm permissions)
{
    (void) ipcEndpoint;
    (void) unrecAttrList;
    (void)  permissions;
    return LwSciError_NotSupported;
}

LwSciError LwSciSyncRegisterWaitObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciSyncObj syncObj,
    LwSciC2cSyncHandle* syncHandle)
{
    (void) channelHandle;
    (void) syncObj;
    (void) syncHandle;
    return LwSciError_NotSupported;
}

LwSciError LwSciSyncRegisterSignalObjIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciSyncObj syncObj,
    LwSciC2cSyncHandle* syncHandle)
{
    (void) channelHandle;
    (void) syncObj;
    (void) syncHandle;
    return LwSciError_NotSupported;
}

LwSciError LwSciBufPushWaitIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cSyncHandle syncHandle,
    const LwSciSyncFence* preFence)
{
    (void) channelHandle;
    (void) syncHandle;
    (void) preFence;
    return LwSciError_NotSupported;
}

LwSciError LwSciBufPushSignalIndirectChannelC2c(
    LwSciC2cHandle channelHandle,
    LwSciC2cSyncHandle syncHandle,
    LwSciSyncFence* postFence)
{
    (void) channelHandle;
    (void) syncHandle;
    (void) postFence;
    return LwSciError_NotSupported;
}

LwSciError LwSciSyncFreeObjIndirectChannelC2c(
    LwSciC2cSyncHandle syncHandle)
{
    (void) syncHandle;
    return LwSciError_NotSupported;
}
#endif
