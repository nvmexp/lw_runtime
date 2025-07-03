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
 * \brief <b>LwSciSync semaphore related Implementation</b>
 *
 * @b Description: Implements the semaphore related APIs
 */
#include <time.h>
#include <unistd.h>
#include "lwscicommon_arch.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core.h"
#ifdef LWSCISYNC_EMU_SUPPORT
#include "lwscisync_attribute_core_cluster.h"
#endif
#include "lwscisync_internal.h"
#include "lwscisync_primitive_core.h"
#include "lwscisync_primitive.h"

/**
 * \brief Represents LwSciSync core primitive
 */
typedef struct {
    /** LwSciBuf object for semaphore buffer */
    LwSciBufObj bufObj;
    /** A read-write pointer to the CPU mapping of the LwSciBufObj */
    void* cpuReadWritePtr;
    /** A read-only pointer to the CPU mapping of the LwSciBufObj */
    const void* cpuReadOnlyPtr;
    /** The bufObj holds maxId number of semaphores.
     * Valid semaphore Id is in [0 - (maxId-1)] */
    uint32_t maxId;
    /** Number of bytes per semaphore */
    uint32_t semaphoreSize;
    /** Payload max size */
    uint64_t payloadMaxVal;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    /** C2C handle */
    LwSciC2cPcieSyncHandle syncHandle;
#endif
} LwSciSyncCoreSemaphoreInfo;

static inline uint64_t GetPayloadMaxVal(
    LwSciSyncCorePrimitive primitive)
{
    uint64_t payloadMaxVal = 0U;
    if (primitive->type ==
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore) {
        payloadMaxVal = UINT32_MAX;
    } else if (primitive->type ==
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b) {
        payloadMaxVal = UINT64_MAX;
    } else {
        LWSCI_ERR_STR("Unsupported semaphore type\n");
        LwSciCommonPanic();
    }
    return payloadMaxVal;
}

static LwSciError LwSciSyncCoreSemaphoreExport(
    LwSciSyncCorePrimitive primitive,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** data,
    size_t* length)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSemaphoreInfo* info = NULL;
    LwSciBufObjIpcExportDescriptor exportDesc;
    info = (LwSciSyncCoreSemaphoreInfo*) primitive->specificData;

    if (LwSciSyncCorePermLEq(LwSciSyncAccessPerm_SignalOnly, permissions)) {
        LWSCI_ERR_STR("semaphores cannot be exported with signaling permissions");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciBufObjIpcExport(info->bufObj, LwSciBufAccessPerm_Auto,
            ipcEndpoint, &exportDesc);
    if (error != LwSciError_Success) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *data = LwSciCommonCalloc(1U, sizeof(LwSciBufObjIpcExportDescriptor));
    if (*data == NULL) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *length = sizeof(LwSciBufObjIpcExportDescriptor);
    LwSciCommonMemcpyS(*data, sizeof(LwSciBufObjIpcExportDescriptor),
            &exportDesc, *length);

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreSemaphoreImport(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList reconciledList,
    const void* data,
    size_t len,
    LwSciSyncCorePrimitive primitive)
{
    LwSciError error = LwSciError_Success;
    LwSciBufAttrList bufAttrList = NULL;
    LwSciSyncCoreSemaphoreInfo* info = NULL;
    bool cpuAccess = false;
    LwSciBufAttrKeyValuePair bufKeyValuePair = {
        .key = LwSciBufGeneralAttrKey_NeedCpuAccess,
        .value = NULL,
        .len = 0U
    };
    LwSciSyncAttrKeyValuePair syncKeyValuePair[] = {
        {
            .attrKey = LwSciSyncAttrKey_ActualPerm,
            .value = NULL,
            .len = 0U
        }
    };
    LwSciSyncInternalAttrKey key =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
    const void* value = NULL;
    size_t length = 0U;

    (void)len;

    /* Dummy import should not happen for semaphore */
    if (ipcEndpoint == 0U) {
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

    error = LwSciSyncAttrListGetAttrs(reconciledList, syncKeyValuePair, 1U);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("getting atrributes should not fail");
        LwSciCommonPanic();
    }

    if (LwSciSyncCorePermLEq(
            LwSciSyncAccessPerm_SignalOnly,
            *(const uint64_t*)syncKeyValuePair[0].value)) {
        LWSCI_ERR_STR("signaling permissions cannot be transferred"
                      " in case of semaphores");
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

    info = LwSciCommonCalloc(1U, sizeof(LwSciSyncCoreSemaphoreInfo));
    if (info == NULL) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

    LwSciSyncCoreAttrListGetSemaAttrList(reconciledList,
            &bufAttrList);

    error = LwSciBufObjIpcImport(ipcEndpoint,
            (const LwSciBufObjIpcExportDescriptor*) data, bufAttrList,
            LwSciBufAccessPerm_Auto, -1, &info->bufObj);
    if (error != LwSciError_Success) {
        goto free_sema;
    }

    error = LwSciBufAttrListGetAttrs(bufAttrList, &bufKeyValuePair, 1U);
    if (error != LwSciError_Success) {
        goto free_bufObj;
    }
    LwSciCommonMemcpyS(&cpuAccess, sizeof(bool),
            bufKeyValuePair.value, bufKeyValuePair.len);

    if (cpuAccess == true) {
        error = LwSciBufObjGetConstCpuPtr(info->bufObj,
                (const void**) &info->cpuReadOnlyPtr);
        if (error != LwSciError_Success) {
            goto free_bufObj;
        }
    }

    error = LwSciSyncAttrListGetSingleInternalAttr(reconciledList, key, &value,
            &length);
    if (error != LwSciError_Success) {
        goto free_bufObj;
    }
    info->maxId = *(const uint32_t*) value;
    info->semaphoreSize = LWSCISYNC_CORE_PRIMITIVE_SEMAPHORE_SIZE;
    info->payloadMaxVal = GetPayloadMaxVal(primitive);

    primitive->id = 0U;
    primitive->lastFence = 0U;
    primitive->specificData = (void*) info;

free_bufObj:
    if (error != LwSciError_Success) {
        LwSciBufObjFree(info->bufObj);
    }

free_sema:
    if (error != LwSciError_Success) {
        LwSciCommonFree(info);
    }

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreSemaphoreInit(
    LwSciSyncAttrList reconciledList,
    LwSciSyncCorePrimitive primitive)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSemaphoreInfo* info = NULL;
    LwSciBufAttrList bufAttrList = NULL;
    bool cpuAccess = false;
    LwSciBufAttrKeyValuePair keyValuePair = {
        .key = LwSciBufGeneralAttrKey_NeedCpuAccess,
        .value = NULL,
        .len = 0U
    };
    LwSciSyncInternalAttrKeyValuePair pairArray[] = {
        {   .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
            .value = NULL,
            .len = 0U
        },
#ifdef LWSCISYNC_EMU_SUPPORT
        {   .attrKey = LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo,
            .value = NULL,
            .len = 0U
        },
#endif
    };
#ifdef LWSCISYNC_EMU_SUPPORT
    const LwSciSyncPrimitiveInfo* externalPrimitiveInfo = NULL;
#endif

    error = LwSciSyncAttrListGetInternalAttrs(reconciledList, &pairArray[0],
            sizeof(pairArray)/sizeof(pairArray[0]));
    if (error != LwSciError_Success) {
        goto fn_exit;
    }
#ifdef LWSCISYNC_EMU_SUPPORT
    if (pairArray[1].value != NULL) {
        externalPrimitiveInfo = (const LwSciSyncPrimitiveInfo*)
                (*(const uintptr_t*)pairArray[1].value);
    }
#endif

    LwSciSyncCoreAttrListGetSemaAttrList(reconciledList,
            &bufAttrList);

    info = LwSciCommonCalloc(1U, sizeof(LwSciSyncCoreSemaphoreInfo));
    if (info == NULL) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

#ifdef LWSCISYNC_EMU_SUPPORT
    if (externalPrimitiveInfo == NULL) {
#endif
        error = LwSciBufObjAlloc(bufAttrList, &info->bufObj);
        if (error != LwSciError_Success) {
            goto free_sema;
        }
#ifdef LWSCISYNC_EMU_SUPPORT
    } else {
        error = LwSciBufObjCreateFromMemHandle(
                externalPrimitiveInfo->semaphorePrimitiveInfo.memHandle,
                externalPrimitiveInfo->semaphorePrimitiveInfo.offset,
                externalPrimitiveInfo->semaphorePrimitiveInfo.len,
                bufAttrList,
                &info->bufObj);
        if (error != LwSciError_Success) {
            goto free_sema;
        }
    }
#endif

    error = LwSciBufAttrListGetAttrs(bufAttrList, &keyValuePair, 1U);
    if (error != LwSciError_Success) {
        goto free_bufObj;
    }
    LwSciCommonMemcpyS(&cpuAccess, sizeof(bool),
            keyValuePair.value, keyValuePair.len);

    if (cpuAccess == true) {
        error = LwSciBufObjGetCpuPtr(info->bufObj,
                (void**) &info->cpuReadWritePtr);
        if (error != LwSciError_Success) {
            goto free_bufObj;
        }
        error = LwSciBufObjGetConstCpuPtr(info->bufObj,
                (const void**) &info->cpuReadOnlyPtr);
        if (error != LwSciError_Success) {
            goto free_bufObj;
        }
    }

    info->maxId = *(const uint32_t*) pairArray[0].value;
    info->semaphoreSize = LWSCISYNC_CORE_PRIMITIVE_SEMAPHORE_SIZE;
    info->payloadMaxVal = GetPayloadMaxVal(primitive);

    primitive->id = 0U;
    primitive->lastFence = 0U;
    primitive->specificData = (void*) info;

free_bufObj:
    if (error != LwSciError_Success) {
        LwSciBufObjFree(info->bufObj);
    }

free_sema:
    if (error != LwSciError_Success) {
        LwSciCommonFree(info);
    }

fn_exit:
    return error;
}

static void LwSciSyncCoreSemaphoreDeinit(
    LwSciSyncCorePrimitive primitive)
{
    LwSciSyncCoreSemaphoreInfo* info = NULL;

    info = (LwSciSyncCoreSemaphoreInfo*) primitive->specificData;

    if (info != NULL) {
        LwSciBufObjFree(info->bufObj);
    }
    LwSciCommonFree(info);
}

static LwSciError LwSciSyncCoreSemaphoreSignal(
    LwSciSyncCorePrimitive primitive)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSemaphoreInfo* info = NULL;
    info = (LwSciSyncCoreSemaphoreInfo*) primitive->specificData;
    if (*(volatile uint64_t*)info->cpuReadWritePtr == info->payloadMaxVal) {
        *(volatile uint64_t*)info->cpuReadWritePtr = 0U;
    } else {
        ++(*(volatile uint64_t*)info->cpuReadWritePtr);
    }
    /* Ensures value is updated in the memory */
    LwSciCommonDMB();
    return error;
}

static inline bool fenceIsExpired(
    uint64_t lwrrentVal,
    uint64_t threshold,
    uint64_t payloadMaxVal)
{
    bool fenceIsExpired = false;
    if (payloadMaxVal == UINT32_MAX) {
        fenceIsExpired = (int32_t)((uint32_t)lwrrentVal - (uint32_t)threshold) >= 0;
    } else if (payloadMaxVal == UINT64_MAX) {
        fenceIsExpired = (int64_t)(lwrrentVal - threshold) >= 0;
    }
    return fenceIsExpired;
}

#define timercmp(tvp, uvp, cmp) \
    (((tvp).tv_sec cmp (uvp).tv_sec) || \
    ((tvp).tv_sec == (uvp).tv_sec && (tvp).tv_nsec cmp (uvp).tv_nsec))

static LwSciError LwSciSyncCoreSemaphoreWaitOn(
    LwSciSyncCorePrimitive primitive,
    LwSciSyncCpuWaitContext waitContext,
    uint64_t id,
    uint64_t value,
    int64_t timeout_us)
{
    LwSciError error = LwSciError_Success;
    const volatile uint64_t* cpuReadOnlyPtr = NULL;
    LwSciSyncCoreSemaphoreInfo* info =
        (LwSciSyncCoreSemaphoreInfo*) primitive->specificData;
    uint64_t skipCount = 1;
    uint64_t count = 1;
    struct timespec sleepTimeSpec = {0};
    uint64_t timeout_us_u64 = 0ULL;
    uint32_t offset = 0U;
    struct timespec timespecExpected = { 0 };
    struct timespec timespecActual = { 0 };
    uint64_t seconds = 0U;
    uint64_t nanoseconds = 0U;

    timeout_us_u64 = (timeout_us == -1) ?
            (uint64_t)LwSciSyncFenceMaxTimeout : (uint64_t)timeout_us;

    (void)waitContext;

    if (id >= info->maxId) {
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

    /** adjust cpuReadOnlyPtr accordingly based on syncFence id */
    offset = id * info->semaphoreSize;
    cpuReadOnlyPtr = (const volatile uint64_t*)
            ((const uint8_t*)info->cpuReadOnlyPtr + offset);

    /* Ensures updated value is read from the memory */
    LwSciCommonDMB();
    /* no wait for expired fence */
    if (fenceIsExpired(*cpuReadOnlyPtr, value, info->payloadMaxVal)) {
        goto fn_exit;
    }
    /* Return Timeout error as fence is not expired and timeout is 0 */
    if (timeout_us_u64 == 0) {
        error = LwSciError_Timeout;
        goto fn_exit;
    }

    clock_gettime(CLOCK_REALTIME, &timespecExpected);
    timespecActual = timespecExpected;
    seconds = timeout_us_u64/1000000;
    nanoseconds = (timeout_us_u64%1000000)*1000;
    timespecExpected.tv_sec += seconds;
    timespecExpected.tv_nsec += nanoseconds;
    if (timespecExpected.tv_nsec >= 1000000000) {
        timespecExpected.tv_sec += (timespecExpected.tv_nsec/1000000000);
        timespecExpected.tv_nsec %= 1000000000;
    }

    /* poll skipCount number of times before sleeping for sleepTimeSpec.tv_nsec */
    if (timeout_us_u64 < 100) {
        skipCount = 999999999;
        sleepTimeSpec.tv_nsec = 100;
    } else if (timeout_us_u64 < 1000) {
        skipCount = 50;
        sleepTimeSpec.tv_nsec = 100;
    } else if (timeout_us_u64 < 10000) {
        skipCount = 10;
        sleepTimeSpec.tv_nsec = 200;
    } else {
        skipCount = 10;
        sleepTimeSpec.tv_nsec = 500;
    }


    while (true) {
        /* Ensures updated value is read from the memory */
        LwSciCommonDMB();
        if (fenceIsExpired(*cpuReadOnlyPtr, value, info->payloadMaxVal)) {
            break;
        }
        clock_gettime(CLOCK_REALTIME, &timespecActual);
        if ((timercmp(timespecActual, timespecExpected, >))) {
            error = LwSciError_Timeout;
            break;
        }

        if (count++ % skipCount == 0) {
            nanosleep(&sleepTimeSpec, NULL);
            count = 1;
        }
    }

fn_exit:
    return error;
}

static uint64_t LwSciSyncCoreSemaphoreGetNewFence(
    LwSciSyncCorePrimitive primitive)
{
    LwSciSyncCoreSemaphoreInfo* info = NULL;
    info = (LwSciSyncCoreSemaphoreInfo*) primitive->specificData;
    if (primitive->lastFence == info->payloadMaxVal) {
        primitive->lastFence = 0U;
    } else {
        ++primitive->lastFence;
    }
    return primitive->lastFence;
}

static void LwSciSyncCoreSemaphoreGetBufObj(
    LwSciSyncCorePrimitive primitive,
    void** data)
{
    LwSciSyncSemaphoreInfo* semaphoreInfo = (LwSciSyncSemaphoreInfo*) *data;
    LwSciSyncCoreSemaphoreInfo* info = (LwSciSyncCoreSemaphoreInfo*)
            primitive->specificData;
    semaphoreInfo->bufObj = info->bufObj;
    semaphoreInfo->semaphoreSize = info->semaphoreSize;
    semaphoreInfo->offset = semaphoreInfo->id * semaphoreInfo->semaphoreSize;
    semaphoreInfo->gpuCacheable = false;
}

static LwSciError LwSciSyncCoreSemaphoreCheckIdValue(
    LwSciSyncCorePrimitive primitive,
    uint64_t id,
    uint64_t value)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSemaphoreInfo* info = (LwSciSyncCoreSemaphoreInfo*)
            primitive->specificData;

    if (id >= info->maxId) {
        LWSCI_ERR_STR("Invalid id\n");
        error = LwSciError_Overflow;
        goto fn_exit;
    }

    if (value > info->payloadMaxVal) {
        LWSCI_ERR_STR("Invalid value\n");
        error = LwSciError_Overflow;
        goto fn_exit;
    }

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreSemaphoreImportThreshold32b(
        LwSciSyncCorePrimitive primitive,
        uint64_t* threshold)
{
    LwSciError error = LwSciError_Success;

    /* for semaphores, the base is always 0 */
    (void) primitive;

    if (*threshold > UINT32_MAX) {
        LWSCI_ERR_ULONG("Fence value invalid: ", *threshold);
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

fn_exit:
    return error;
}

static LwSciError LwSciSyncCoreSemaphoreImportThreshold64b(
        LwSciSyncCorePrimitive primitive,
        uint64_t* threshold)
{
    (void) primitive;
    (void) threshold;
    /* for semaphores, the base is always 0 */
    return LwSciError_Success;
}

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
static LwSciError LwSciSyncCoreSemaphoreGetC2cSyncHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncHandle* syncHandle)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSemaphoreInfo* info = (LwSciSyncCoreSemaphoreInfo*)
            primitive->specificData;

    if (NULL == info->syncHandle) {
        /* no print as this error could be used for informative purposes */
        error = LwSciError_NotInitialized;
    } else {
        *syncHandle = info->syncHandle;
    }

    return error;
}

static LwSciError LwSciSyncCoreSemaphoreGetC2cRmHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncRmHandle* syncRmHandle)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreSemaphoreInfo* info = (LwSciSyncCoreSemaphoreInfo*)
            primitive->specificData;

    if (NULL == info->bufObj) {
        /* no print as this error could be used for informative purposes */
        error = LwSciError_NotInitialized;
    } else {
        /* TODO: Fill in rm shim handles */
        memset(syncRmHandle, 0U, sizeof(LwSciC2cPcieSyncRmHandle));
    }

    return error;
}
#endif

const LwSciSyncPrimitiveOps LwSciSyncBackEndSysmemSema =
{
    .Init = LwSciSyncCoreSemaphoreInit,
    .Deinit = LwSciSyncCoreSemaphoreDeinit,
    .Export = LwSciSyncCoreSemaphoreExport,
    .Import = LwSciSyncCoreSemaphoreImport,
    .Signal = LwSciSyncCoreSemaphoreSignal,
    .WaitOn = LwSciSyncCoreSemaphoreWaitOn,
    .GetNewFence = LwSciSyncCoreSemaphoreGetNewFence,
    .GetSpecificData = LwSciSyncCoreSemaphoreGetBufObj,
    .CheckIdValue = LwSciSyncCoreSemaphoreCheckIdValue,
#if (LW_IS_SAFETY == 0)&& (LW_L4T == 0)
    .GetC2cSyncHandle = LwSciSyncCoreSemaphoreGetC2cSyncHandle,
    .GetC2cRmHandle = LwSciSyncCoreSemaphoreGetC2cRmHandle,
#endif
    .ImportThreshold = LwSciSyncCoreSemaphoreImportThreshold32b,
};

const LwSciSyncPrimitiveOps LwSciSyncBackEndSysmemSemaPayload64b =
{
    .Init = LwSciSyncCoreSemaphoreInit,
    .Deinit = LwSciSyncCoreSemaphoreDeinit,
    .Export = LwSciSyncCoreSemaphoreExport,
    .Import = LwSciSyncCoreSemaphoreImport,
    .Signal = LwSciSyncCoreSemaphoreSignal,
    .WaitOn = LwSciSyncCoreSemaphoreWaitOn,
    .GetNewFence = LwSciSyncCoreSemaphoreGetNewFence,
    .GetSpecificData = LwSciSyncCoreSemaphoreGetBufObj,
    .CheckIdValue = LwSciSyncCoreSemaphoreCheckIdValue,
#if (LW_IS_SAFETY == 0)&& (LW_L4T == 0)
    .GetC2cSyncHandle = NULL,
    .GetC2cRmHandle = NULL,
#endif
    .ImportThreshold = LwSciSyncCoreSemaphoreImportThreshold64b,
};
