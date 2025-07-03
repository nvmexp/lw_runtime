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
 * \brief <b>LwSciSync Timestamps Buffer Implementation</b>
 *
 * @b Description: This file implements LwSciSync timestamps buffer APIs
 *
 */
#include "lwscisync_timestamps.h"

#include "lwscibuf_internal.h"
#include "lwscicommon_arch.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscicommon_transportutils.h"
#include "lwscicommon_utils.h"
#include "lwscilog.h"
#include "lwscisync_primitive.h"
#include "lwscisync_primitive_core.h"
#include "lwscisync_attribute_core.h"

struct LwSciSyncCoreTimestampsRec {
    /* immutable after creation */
    /* buffer info */
    LwSciBufObj bufObj;
    size_t size;
    /* base address for easy CPU access */
    uint64_t* buffer;
    const uint64_t* constBuffer;

    /* timestamp format */
    LwSciSyncAttrValTimestampInfo formatInfo;

    /* mutable */
    /* slot for the next fence */
    uint32_t lwrrentFenceSlot;
    /* slot to be used by the next cpu Signal() */
    uint32_t lwrrentWriteSlot;
    /* slot iterators' protection */
    LwSciCommonMutex slotMutex;
};

/**
 * \brief Types of LwSciSyncCorePrimitive Keys for Exporting
 */
typedef enum {
    /** (LwSciBufObjIpcExportDescriptor) */
    LwSciSyncCoreTimestampsKey_BufObj,
} LwSciSyncCoreTimestampsKey;

static uint64_t slotsNumber(
    LwSciSyncCoreTimestamps timestamps);

static void slotAddress(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t slotIndex,
    uint64_t primitiveId,
    uint32_t primitiveSize,
    uint64_t** address);

static void constSlotAddress(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t slotIndex,
    uint64_t primitiveId,
    uint32_t primitiveSize,
    const uint64_t** address);


static LwSciError CoreTimestampsInit(
    LwSciBufObj bufObj,
    size_t size,
    bool cpuAccess,
    LwSciSyncAttrValTimestampInfo formatInfo,
    LwSciSyncCoreTimestamps* timestamps);

static LwSciError GetTimestampInfo(
    LwSciSyncAttrList reconciledList,
    LwSciSyncAttrValTimestampInfo* timestampInfo)
{
    LwSciError error = LwSciError_Success;

    size_t len = 0U;
    size_t lenMulti = 0U;
    const void *value = NULL;
    const void *valueMulti = NULL;

    /* Either one of the attribute keys should be set. */
    error = LwSciSyncAttrListGetSingleInternalAttr(reconciledList,
        LwSciSyncInternalAttrKey_SignalerTimestampInfo, &value, &len);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    error = LwSciSyncAttrListGetSingleInternalAttr(reconciledList,
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti, &valueMulti,
        &lenMulti);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    if (len != 0U) {
        LwSciCommonMemcpyS(timestampInfo, sizeof(LwSciSyncAttrValTimestampInfo),
            value, len);
    } else if (lenMulti != 0U) {
        LwSciCommonMemcpyS(timestampInfo, sizeof(LwSciSyncAttrValTimestampInfo),
            valueMulti, lenMulti);
    } else {
        error = LwSciError_BadParameter;
    }

fn_exit:
    return error;
}

LwSciError LwSciSyncCoreTimestampsInit(
    LwSciSyncAttrList reconciledList,
    LwSciSyncCorePrimitive* primitive,
    LwSciSyncCoreTimestamps* timestamps)
{
    LwSciError error = LwSciError_Success;
    uint64_t size = 0U;
    bool cpuAccess = false;
    const void *value = NULL;
    size_t len = 0U;
    bool supportsTimestamps = false;
    LwSciBufAttrList timestampBufAttrList = NULL;
    LwSciBufObj bufObj = NULL;
    LwSciSyncAttrValTimestampInfo formatInfo;
    LwSciBufAttrKeyValuePair keyValuePair[] = {
        {
            .key = LwSciBufRawBufferAttrKey_Size,
            .value = NULL,
            .len = 0U
        },
        {
            .key = LwSciBufGeneralAttrKey_NeedCpuAccess,
            .value = NULL,
            .len = 0U
        }
    };

    LWSCI_FNENTRY("");

    /* Check if init is needed */
    error = LwSciSyncAttrListGetAttr(reconciledList,
            LwSciSyncAttrKey_WaiterRequireTimestamps, &value, &len);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    if (value != NULL) {
        LwSciCommonMemcpyS(&supportsTimestamps, sizeof(bool),
                value, len);
    }

    if (supportsTimestamps == false) {
        goto fn_exit;
    }

    error = GetTimestampInfo(reconciledList, &formatInfo);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }
    if (formatInfo.format == LwSciSyncTimestampFormat_EmbeddedInPrimitive) {
        /* This uses the same buffer as the Semaphore buffer. */
        LwSciSyncSemaphoreInfo semaphoreInfo = {};
        LwSciSyncSemaphoreInfo* semaphoreInfoPtr = &semaphoreInfo;

        error = LwSciSyncCorePrimitiveGetSpecificData(*primitive,
            (void**) &semaphoreInfoPtr);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }

        /* We don't Clone the Semaphore attribute list here since cloning would
         * create a new LwSciBufAttrList, which wouldn't be associated with the
         * Semaphore LwSciBufObj (since it wasn't used to allocate that memory).
         *
         * As long as the reference to the underlying memory object is not
         * freed, then things are fine. */
        error = LwSciBufObjDup(semaphoreInfo.bufObj, &bufObj);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
        LwSciSyncCoreAttrListGetSemaAttrList(reconciledList,
            &timestampBufAttrList);
    } else {
        /* Otherwise, a separate timestamp buffer is allocated via LwSciBuf. */
        LwSciSyncCoreAttrListGetTimestampBufAttrList(reconciledList,
            &timestampBufAttrList);

        error = LwSciBufObjAlloc(timestampBufAttrList, &bufObj);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    }

    /* Get timestamp buf attrs */
    error = LwSciBufAttrListGetAttrs(timestampBufAttrList, keyValuePair, 2U);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }
    size = *(const size_t*) keyValuePair[0].value;
    cpuAccess = *(const bool*) keyValuePair[1].value;

    error = CoreTimestampsInit(bufObj, size, cpuAccess, formatInfo, timestamps);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

fn_exit:
    if (error != LwSciError_Success) {
        if (bufObj != NULL) {
            LwSciBufObjFree(bufObj);
        }
    }

    LWSCI_FNEXIT("");

    return error;
}

void LwSciSyncCoreTimestampsDeinit(
    LwSciSyncCoreTimestamps timestamps)
{
    LWSCI_FNENTRY("");

    if (timestamps == NULL) {
        goto fn_exit;
    }

    LwSciBufObjFree(timestamps->bufObj);
    LwSciCommonMutexDestroy(&timestamps->slotMutex);
    LwSciCommonFree(timestamps);

fn_exit:
    LWSCI_FNEXIT("");
}

LwSciError LwSciSyncCoreTimestampsImport(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList reconciledList,
    const void* data,
    size_t len,
    LwSciSyncCoreTimestamps* timestamps)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreTimestamps result = NULL;
    LwSciCommonTransportBuf* corePrimitiveRxbuf = NULL;
    LwSciCommonTransportParams params;
    bool doneReading = false;
    LwSciSyncCoreTimestampsKey key;
    LwSciBufAttrList bufAttrList = NULL;
    LwSciBufAttrKeyValuePair keyValuePair[] = {
        {
            .key = LwSciBufRawBufferAttrKey_Size,
            .value = NULL,
            .len = 0U,
        },
        {
            .key = LwSciBufGeneralAttrKey_NeedCpuAccess,
            .value = NULL,
            .len = 0U,
        },
    };
    LwSciSyncInternalAttrKeyValuePair syncKeyValPair[] = {
        {
            .attrKey = LwSciSyncInternalAttrKey_SignalerTimestampInfo,
            .value = NULL,
            .len = 0U,
        },
        {
            .attrKey = LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
            .value = NULL,
            .len = 0U,
        },
    };
    bool cpuAccess = false;

    LWSCI_FNENTRY("");

    error = LwSciSyncAttrListGetInternalAttrs(reconciledList, syncKeyValPair,
            sizeof(syncKeyValPair)/sizeof(syncKeyValPair[0]));
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    LwSciSyncCoreAttrListGetTimestampBufAttrList(
            reconciledList, &bufAttrList);

    result = (LwSciSyncCoreTimestamps) LwSciCommonCalloc(
            1U, sizeof(struct LwSciSyncCoreTimestampsRec));
    if (result == NULL) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

    error = LwSciCommonTransportGetRxBufferAndParams(
            data, len, &corePrimitiveRxbuf,
            &params);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    do {
        uint32_t inputKey = 0U;
        size_t length = 0U;
        const void* value = NULL;

        error = LwSciCommonTransportGetNextKeyValuePair(
                corePrimitiveRxbuf, &inputKey,
                &length, &value, &doneReading);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
        key = (LwSciSyncCoreTimestampsKey) inputKey;
        if (key == LwSciSyncCoreTimestampsKey_BufObj) {
            error = LwSciBufObjIpcImport(ipcEndpoint,
                   (const LwSciBufObjIpcExportDescriptor*) value, bufAttrList,
                   LwSciBufAccessPerm_Readonly, -1, &result->bufObj);
            if (error != LwSciError_Success) {
                goto fn_exit;
            }
        } else {
            LWSCI_INFO("Ignoring unknown key %zu\n", (size_t) key);
        }
    } while (doneReading == false);

    error = LwSciBufAttrListGetAttrs(bufAttrList, keyValuePair,
            sizeof(keyValuePair)/sizeof(keyValuePair[0]));
    if (error != LwSciError_Success) {
        goto fn_exit;
    }
    result->size = *(const size_t*) keyValuePair[0].value;
    cpuAccess = *(const bool*) keyValuePair[1].value;
    if (syncKeyValPair[0].len != 0U) {
        LwSciCommonMemcpyS(&result->formatInfo,
            sizeof(LwSciSyncAttrValTimestampInfo),
            syncKeyValPair[0].value,
            syncKeyValPair[0].len);
    } else {
        LwSciCommonMemcpyS(&result->formatInfo,
            sizeof(LwSciSyncAttrValTimestampInfo),
            syncKeyValPair[1].value,
            syncKeyValPair[1].len);
    }

    /* initialize mutables etc. */
    error = LwSciCommonMutexCreate(&result->slotMutex);
    if (LwSciError_Success != error) {
        goto fn_exit;
    }

    if (cpuAccess == true) {
        error = LwSciBufObjGetConstCpuPtr(result->bufObj,
                (const void**) &result->constBuffer);
        if (error != LwSciError_Success) {
            goto destroy_mutex;
        }
    }

    *timestamps = result;
    goto fn_exit;

destroy_mutex:
    LwSciCommonMutexDestroy(&result->slotMutex);
fn_exit:
    LwSciCommonTransportBufferFree(corePrimitiveRxbuf);
    if (LwSciError_Success != error) {
        LwSciCommonFree(result);
    }
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncCoreTimestampsExport(
    LwSciSyncCoreTimestamps timestamps,
    LwSciIpcEndpoint ipcEndpoint,
    void** data,
    size_t* length)
{
    LwSciError error = LwSciError_Success;
    size_t len = 0U;
    uint32_t key = 0U;
    const void* value = NULL;
    LwSciCommonTransportBuf* corePrimitiveTxbuf = NULL;
    LwSciCommonTransportParams bufparams = {0};
    size_t totalValueSize = 0U;
    LwSciBufObjIpcExportDescriptor bufExportDesc;

    LWSCI_FNENTRY("");

    if (timestamps == NULL) {
        goto fn_exit;
    }

    bufparams.keyCount = 1U;
    totalValueSize = sizeof(LwSciBufObjIpcExportDescriptor);

    error = LwSciCommonTransportAllocTxBufferForKeys(bufparams, totalValueSize,
            &corePrimitiveTxbuf);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    error = LwSciBufObjIpcExport(timestamps->bufObj,
            LwSciBufAccessPerm_Readonly,
            ipcEndpoint, &bufExportDesc);
    if (error != LwSciError_Success) {
        goto free_corePrimitiveTxBuf;
    }

    len = sizeof(LwSciBufObjIpcExportDescriptor);
    value = (const void*) &bufExportDesc;
    key = (uint32_t)LwSciSyncCoreTimestampsKey_BufObj;
    error = LwSciCommonTransportAppendKeyValuePair(
            corePrimitiveTxbuf, key, len, value);
    if (error != LwSciError_Success) {
        goto free_corePrimitiveTxBuf;
    }

    LwSciCommonTransportPrepareBufferForTx(corePrimitiveTxbuf,
            data, length);

free_corePrimitiveTxBuf:
    LwSciCommonTransportBufferFree(corePrimitiveTxbuf);
fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LwSciError LwSciSyncCoreTimestampsGetNextSlot(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t* slotIndex)
{
    uint64_t slots = slotsNumber(timestamps);
    uint32_t lwrSlot = 0U;
    uint32_t tmpSlot = 0U;
    LwSciError error = LwSciError_Success;
    uint8_t addStatus = OP_FAIL;
    uint32_t u32Slots = 0U;

    LWSCI_FNENTRY("");

    LwSciCommonMutexLock(&timestamps->slotMutex);

    lwrSlot = timestamps->lwrrentFenceSlot;
    u32Add(lwrSlot, 1U, &tmpSlot, &addStatus);
    if ((slots > UINT32_MAX) && (addStatus != OP_SUCCESS)) {
        LwSciCommonMutexUnlock(&timestamps->slotMutex);
        error = LwSciError_Overflow;
        goto fn_exit;
    } else {
        u32Slots = (uint32_t)slots;
    }
    timestamps->lwrrentFenceSlot = tmpSlot % u32Slots;

    LwSciCommonMutexUnlock(&timestamps->slotMutex);

    *slotIndex = lwrSlot;

    LWSCI_FNEXIT("");

fn_exit:
    return error;
}

void LwSciSyncCoreTimestampsGetBufferInfo(
    LwSciSyncCoreTimestamps timestamp,
    LwSciSyncTimestampBufferInfo* bufferInfo)
{
    bufferInfo->bufObj = timestamp->bufObj;
    bufferInfo->size = timestamp->size;
}

LwSciError LwSciSyncCoreTimestampsGetTimestamp(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t slotIndex,
    LwSciSyncCorePrimitive primitive,
    uint64_t fenceId,
    uint64_t* stamp)
{
    LwSciError error = LwSciError_Success;
    const uint64_t* address = NULL;
    uint64_t rawTs = 0U;
    uint64_t num = timestamps->formatInfo.scaling.scalingFactorNumerator;
    uint64_t den = timestamps->formatInfo.scaling.scalingFactorDenominator;
    uint64_t off = timestamps->formatInfo.scaling.sourceOffset;
    uint8_t addStatus = OP_FAIL;
    uint8_t mulStatus = OP_FAIL;
    uint64_t tmp = 0U;

    uint32_t primitiveSize = 0U;

    LWSCI_FNENTRY("");

    if (LwSciSyncCoreTimestampsIsSlotValid(timestamps, slotIndex) == false) {
        error = LwSciError_BadParameter;
        LWSCI_ERR_UINT("invalid slotIndex \n", slotIndex);
        goto fn_exit;
    }

    if (primitive->type != LwSciSyncInternalAttrValPrimitiveType_Syncpoint) {
        LwSciSyncSemaphoreInfo info = { 0 };
        LwSciSyncSemaphoreInfo* infoPtr = &info;

        error = LwSciSyncCorePrimitiveGetSpecificData(primitive,
            (void**)&infoPtr);
        if (error != LwSciError_Success) {
            LwSciCommonPanic();
        }
        primitiveSize = info.semaphoreSize;
    }

    constSlotAddress(timestamps, slotIndex, fenceId, primitiveSize, &address);
    if (address == NULL) {
        LWSCI_ERR_UINT("Could not find the address for slot \n",
                slotIndex);
        error = LwSciError_NotSupported;
        goto fn_exit;
    }

    rawTs = *address;
    u64Mul(num, rawTs, &tmp, &mulStatus);
    tmp = tmp / den;
    u64Add(off, tmp, stamp, &addStatus);
    if ((mulStatus & addStatus) != OP_SUCCESS) {
        error = LwSciError_Overflow;
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

bool LwSciSyncCoreTimestampsIsSlotValid(
    LwSciSyncCoreTimestamps timestamps,
    uint64_t slotIndex)
{
    return slotIndex < slotsNumber(timestamps);
}

LwSciError LwSciSyncCoreTimestampsWriteTime(
    LwSciSyncCoreTimestamps timestamps,
    LwSciSyncCorePrimitive primitive)
{
    LwSciError error = LwSciError_Success;
    uint64_t slots = slotsNumber(timestamps);
    uint32_t lwrSlot = 0U;
    uint64_t* address = NULL;
    uint32_t tmpSlot = 0U;
    uint8_t addStatus = OP_FAIL;
    uint32_t u32Slots = 0U;

    uint32_t primitiveSize = 0U;

    LWSCI_FNENTRY("");

    LwSciCommonMutexLock(&timestamps->slotMutex);

    lwrSlot = timestamps->lwrrentWriteSlot;
    u32Add(lwrSlot, 1U, &tmpSlot, &addStatus);
    if ((slots > UINT32_MAX) && (addStatus != OP_SUCCESS)) {
        LwSciCommonMutexUnlock(&timestamps->slotMutex);
        error = LwSciError_Overflow;
        goto fn_exit;
    } else {
        u32Slots = (uint32_t)slots;
    }
    timestamps->lwrrentWriteSlot = tmpSlot % u32Slots;

    LwSciCommonMutexUnlock(&timestamps->slotMutex);

    if (primitive->type != LwSciSyncInternalAttrValPrimitiveType_Syncpoint) {
        LwSciSyncSemaphoreInfo info = { 0 };
        LwSciSyncSemaphoreInfo* infoPtr = &info;

        error = LwSciSyncCorePrimitiveGetSpecificData(primitive,
            (void**)&infoPtr);
        if (error != LwSciError_Success) {
            LwSciCommonPanic();
        }
        primitiveSize = info.semaphoreSize;
    }

    slotAddress(timestamps, lwrSlot, primitive->id, primitiveSize, &address);
    if (address == NULL) {
        LWSCI_ERR_STR("Could not find the address to write timestamp");
        error = LwSciError_NotSupported;
        goto fn_exit;
    }

    *address = LwSciCommonGetTimeUS();

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

static uint64_t slotsNumber(
    LwSciSyncCoreTimestamps timestamps)
{
    uint32_t numSlots = 0U;

    switch (timestamps->formatInfo.format) {
        case LwSciSyncTimestampFormat_8Byte:
        {
            uint32_t slotSize = 8U;
            numSlots = (timestamps->size / slotSize);
            break;
        }
        case LwSciSyncTimestampFormat_EmbeddedInPrimitive:
        {
            numSlots = 1U;
            break;
        }
        default:
        {
            uint32_t slotSize = 16U;
            numSlots = (timestamps->size / slotSize);
            break;
        }
    }

    return numSlots;
}

static void slotAddress(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t slotIndex,
    uint64_t primitiveId,
    uint32_t primitiveSize,
    uint64_t** address)
{
    uint64_t* result = NULL;
    uint64_t* buffer = NULL;

    switch (timestamps->formatInfo.format) {
        case LwSciSyncTimestampFormat_8Byte:
        {
            buffer = timestamps->buffer;
            result = &buffer[slotIndex];
            break;
        }
        case LwSciSyncTimestampFormat_EmbeddedInPrimitive:
        {
            /* Compute the offset for each semaphore ID */
            uint32_t offset = primitiveId * primitiveSize;
            buffer = (uint64_t*)((uint8_t*)timestamps->buffer + offset);

            /* - 4 bytes for 32-bit semaphore payload
             * - 4 bytes reserved
             * - 8 bytes for timestamp value.
             *
             * Thus, we increment 8 bytes past the start of the buffer. */
            result = &buffer[slotIndex + 1];
            break;
        }
        default:
        {
            result = NULL;
            break;
        }
    }

    *address = result;
}

static void constSlotAddress(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t slotIndex,
    uint64_t primitiveId,
    uint32_t primitiveSize,
    const uint64_t** address)
{
    const uint64_t* result = NULL;

    switch (timestamps->formatInfo.format) {
        case LwSciSyncTimestampFormat_8Byte:
        {
            result = &timestamps->constBuffer[slotIndex];
            break;
        }
        case LwSciSyncTimestampFormat_EmbeddedInPrimitive:
        {
            /* Compute the offset for each semaphore ID */
            uint32_t offset = primitiveId * primitiveSize;
            const uint64_t* buffer =
                (const uint64_t*)((const uint8_t*)timestamps->constBuffer + offset);

            /* - 4 bytes for 32-bit semaphore payload
             * - 4 bytes reserved
             * - 8 bytes for timestamp value.
             *
             * Thus, we increment 8 bytes past the start of the buffer. */
            result = (const uint64_t*)&buffer[slotIndex + 1];
            break;
        }
        default:
        {
            result = NULL;
            break;
        }
    }

    *address = result;
}

static LwSciError CoreTimestampsInit(
    LwSciBufObj bufObj,
    size_t size,
    bool cpuAccess,
    LwSciSyncAttrValTimestampInfo formatInfo,
    LwSciSyncCoreTimestamps* timestamps)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreTimestamps result = NULL;

    LWSCI_FNENTRY("");

    /* Allocate core timestamp structure and set fields */
    result = (LwSciSyncCoreTimestamps) LwSciCommonCalloc(
            1U, sizeof(struct LwSciSyncCoreTimestampsRec));
    if (result == NULL) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        goto fn_exit;
    }

    result->bufObj = bufObj;
    result->size = size;
    result->formatInfo = formatInfo;

    error = LwSciCommonMutexCreate(&result->slotMutex);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    if (cpuAccess == true) {
        error = LwSciBufObjGetCpuPtr(result->bufObj, (void**) &result->buffer);
        if (error != LwSciError_Success) {
            goto destroy_mutex;
        }
        error = LwSciBufObjGetConstCpuPtr(bufObj,
                (const void**) &result->constBuffer);
        if (error != LwSciError_Success) {
            goto destroy_mutex;
        }
    }

    *timestamps = result;
    goto fn_exit;

destroy_mutex:
    LwSciCommonMutexDestroy(&result->slotMutex);
fn_exit:
    if (error != LwSciError_Success) {
        LwSciCommonFree(result);
    }

    LWSCI_FNEXIT("");

    return error;
}
