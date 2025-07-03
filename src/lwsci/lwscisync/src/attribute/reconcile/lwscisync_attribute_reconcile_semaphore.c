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
 * \brief <b>LwSciSync semaphore reconciliation implementation</b>
 *
 * @b Description: This file implements semaphore attribute reconciliation logic
 */
#include "lwscisync_attribute_reconcile_priv.h"

#include "lwscibuf.h"
#include "lwscibuf_internal.h"
#include "lwscicommon_libc.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core_cluster.h"
#include "lwscisync_attribute_reconcile_private.h"
#include "lwscisync_module.h"

/** Check if buf Attr list is needed for a given core sync attr list */
static bool IsSemaAttrListNeeded(
    LwSciSyncCoreAttrList* coreAttrList);

/** Check if sysmem semaphore is one of the primitives */
static bool CheckForSemaphorePrimitive(
    const LwSciSyncInternalAttrValPrimitiveType* primitiveInfo,
    size_t size);

static LwSciError SetGpuId(
    LwSciSyncCoreAttrList* coreAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciBufAttrKeyValuePair pairArray[] = {
        {    .key = LwSciBufGeneralAttrKey_GpuId,
        }
    };
    size_t pairCount = sizeof(pairArray)/sizeof(pairArray[0]);
    size_t keyIdx = LwSciSyncCoreKeyToIndex(LwSciSyncInternalAttrKey_GpuId);
    if ((coreAttrList->attrs.keyState[keyIdx] !=
            LwSciSyncCoreAttrKeyState_Empty) &&
            (coreAttrList->attrs.keyState[keyIdx] !=
            LwSciSyncCoreAttrKeyState_Conflict)) {
        pairArray[0].value = (const void*)&coreAttrList->attrs.gpuId;
        pairArray[0].len = sizeof(coreAttrList->attrs.gpuId);
        error = LwSciBufAttrListSetAttrs(coreAttrList->semaAttrList, pairArray,
                pairCount);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    }

fn_exit:

    return error;
}

LwSciError LwSciSyncCoreFillSemaAttrList(
    LwSciSyncCoreAttrListObj* objAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    LwSciBufModule bufModule = NULL;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    bool cpuAccess = false;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_Ilwalid;
    bool cpuCache = false;
    uint64_t bufSize = 0U;
    uint64_t bufAlign = 16U;
    LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;
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
        {    .key = LwSciBufGeneralAttrKey_EnableCpuCache,
             .value = (const void*) &cpuCache,
             .len = sizeof(cpuCache)
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

    LwSciBufInternalAttrKeyValuePair internalAttrs[] = {
        {
            LwSciBufInternalGeneralAttrKey_MemDomainArray,
            &memDomain,
            sizeof(memDomain),
        },
    };

    LwSciSyncCoreAttrList* coreAttrList = NULL;

    LwSciSyncCoreModuleGetBufModule(objAttrList->module, &bufModule);

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        bool isBufAttrsNeeded = false;
        size_t pairCount = sizeof(pairArray)/sizeof(pairArray[0]);
        coreAttrList = &objAttrList->coreAttrList[i];
        /** Skip already created buf attr list */
        if (coreAttrList->semaAttrList != NULL) {
            continue;
        }

        /** Check if LwSciBuf sema attr list creation is needed */
        isBufAttrsNeeded = IsSemaAttrListNeeded(coreAttrList);
        if (isBufAttrsNeeded == false) {
            continue;
        }

        if (coreAttrList->attrs.requiredPerm ==
                LwSciSyncAccessPerm_WaitOnly) {
            perm = LwSciBufAccessPerm_Readonly;
            /** Skip setting buffer size for waiters */
            pairCount -= 1U;
        } else {
            perm = LwSciBufAccessPerm_ReadWrite;
            bufSize =
                    16ULL * coreAttrList->attrs.signalerPrimitiveCount;
        }

        cpuAccess = coreAttrList->attrs.needCpuAccess;

        error = LwSciBufAttrListCreate(bufModule, &coreAttrList->semaAttrList);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }

        error = LwSciBufAttrListSetAttrs(coreAttrList->semaAttrList, pairArray,
                pairCount);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }

        error = SetGpuId(coreAttrList);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }

        error = LwSciBufAttrListSetInternalAttrs(coreAttrList->semaAttrList,
                internalAttrs,
                sizeof(internalAttrs)/sizeof(internalAttrs[0]));
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    }

fn_exit:

    return error;
}

LwSciError ReconcileSemaAttrList(
    LwSciSyncCoreAttrListObj* objAttrList,
    LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    /* This holds the reconciled attr list if LwSciBufAttrListReconcile()
     * succeeds, else holds the conflict attr list */
    LwSciBufAttrList* semaAttrList = NULL;
    LwSciBufAttrList* inputArray = NULL;
    const LwSciSyncCoreAttrs* attrs = &newObjAttrList->coreAttrList->attrs;

    /* TODO: Extend this in future for Vidmem semaphore */
    if (!CheckForSemaphorePrimitive(attrs->signalerPrimitiveInfo, 1U)) {
        /* nothing to do if semaphore primitive was not chosen */
        goto fn_exit;
    }

    error = LwSciSyncCoreFillSemaAttrList(objAttrList);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    semaAttrList = &newObjAttrList->coreAttrList->semaAttrList;
    inputArray = (LwSciBufAttrList*)LwSciCommonCalloc(
            objAttrList->numCoreAttrList, sizeof(LwSciBufAttrList));
    if (inputArray == NULL) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        inputArray[i] = objAttrList->coreAttrList[i].semaAttrList;
    }
    error = LwSciBufAttrListReconcile(inputArray, objAttrList->numCoreAttrList,
            semaAttrList, semaAttrList);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

fn_exit:
    LwSciCommonFree(inputArray);

    return error;
}

LwSciError ValidateReconciledSemaAttrList(
    LwSciSyncCoreAttrListObj* objAttrList,
    LwSciSyncCoreAttrListObj* newObjAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    bool isReconcileListValid = false;
    LwSciBufAttrList reconciledSemaAttrList =
            newObjAttrList->coreAttrList->semaAttrList;
    LwSciBufAttrList* inputArray = NULL;
    const LwSciSyncCoreAttrs* attrs = &newObjAttrList->coreAttrList->attrs;

    if (!CheckForSemaphorePrimitive(attrs->signalerPrimitiveInfo, 1U)) {
        /* nothing to do if semaphore primitive was not chosen */
        goto fn_exit;
    }

    error = LwSciSyncCoreFillSemaAttrList(objAttrList);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }
    inputArray = (LwSciBufAttrList*)LwSciCommonCalloc(
            objAttrList->numCoreAttrList, sizeof(LwSciBufAttrList));
    if (inputArray == NULL) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        inputArray[i] = objAttrList->coreAttrList[i].semaAttrList;
    }

    error = LwSciBufAttrListValidateReconciled(reconciledSemaAttrList,
            inputArray, objAttrList->numCoreAttrList, &isReconcileListValid);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    if (isReconcileListValid == false) {
        LWSCI_ERR_STR("Invalid reconciled sema attr list.\n");
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

fn_exit:
    LwSciCommonFree(inputArray);

    return error;
}

static bool IsSemaAttrListNeeded(
    LwSciSyncCoreAttrList* coreAttrList)
{
    bool signalerSemaphore = false;
    bool waiterSemaphore = false;
    LwSciSyncInternalAttrValPrimitiveType* primitiveInfo = NULL;
    LwSciSyncAccessPerm requiredPerm = coreAttrList->attrs.requiredPerm;
    size_t keyIdx = 0U;
    size_t size = 0U;
    if ((requiredPerm == LwSciSyncAccessPerm_SignalOnly) ||
            (requiredPerm ==  LwSciSyncAccessPerm_WaitSignal)) {
        primitiveInfo = coreAttrList->attrs.signalerPrimitiveInfo;
        keyIdx = LwSciSyncCoreKeyToIndex(
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo);
        size = coreAttrList->attrs.valSize[keyIdx] /
                LwSciSyncCoreKeyInfo[keyIdx].elemSize;
        /** Buf attr list is needed only if one of the primitive is semaphore */
        signalerSemaphore = CheckForSemaphorePrimitive(primitiveInfo, size);
    }
    if ((requiredPerm == LwSciSyncAccessPerm_WaitOnly) ||
            (requiredPerm ==  LwSciSyncAccessPerm_WaitSignal)) {
        primitiveInfo = coreAttrList->attrs.waiterPrimitiveInfo;
        keyIdx = LwSciSyncCoreKeyToIndex(
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo);
        size = coreAttrList->attrs.valSize[keyIdx] /
                 LwSciSyncCoreKeyInfo[keyIdx].elemSize;
        /** Buf attr list is needed only if one of the primitive is semaphore */
        waiterSemaphore = CheckForSemaphorePrimitive(primitiveInfo, size);
    }
    return signalerSemaphore || waiterSemaphore;
}

static bool CheckForSemaphorePrimitive(
    const LwSciSyncInternalAttrValPrimitiveType* primitiveInfo,
    size_t size)
{
    bool hasSemaphore = false;
    size_t i;
    for (i = 0U; i < size; i++) {
        if ((primitiveInfo[i] ==
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore) ||
               (primitiveInfo[i] ==
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b)) {
            hasSemaphore = true;
            break;
        }
    }
    return hasSemaphore;
}
