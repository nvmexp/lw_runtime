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
 * \brief <b>LwSciSync generic primitive Implementation</b>
 *
 * @b Description: This file implements LwSciSync generic primitive APIs
 */

#include "lwscisync_primitive.h"

#include "lwscilog.h"
#include "lwscicommon_os.h"
#include "lwscicommon_transportutils.h"
#include "lwscicommon_utils.h"
#include "lwscicommon_covanalysis.h"
#include "lwscisync_attribute_core.h"
#include "lwscisync_core.h"
#include "lwscisync_primitive_core.h"

/******************************************************
 *            Core structures declaration
 ******************************************************/

/**
 * \brief Types of LwSciSyncCorePrimitive Keys for Exporting
 */
typedef enum {
    /** (LwSciSyncInternalAttrValPrimitiveType) */
    LwSciSyncCorePrimitiveKey_Type,
    /** (uint64_t) */
    LwSciSyncCorePrimitiveKey_Id,
    /** (void *) Data specific to actual primitive */
    LwSciSyncCorePrimitiveKey_Specific,
} LwSciSyncCorePrimitiveKey;

/******************************************************
 *             Core interfaces definition
 ******************************************************/

static const LwSciSyncPrimitiveOps LwSciSyncBackEndIlwalid = {0};

static const LwSciSyncPrimitiveOps LwSciSyncBackEndVidmemSema = {0};

static const LwSciSyncPrimitiveOps LwSciSyncBackEndVidmemSemaPayload64b = {0};

static const LwSciSyncPrimitiveOps*
LwSciSyncPrimitiveOpsArray[LwSciSyncInternalAttrValPrimitiveType_UpperBound] = {
    &LwSciSyncBackEndIlwalid,
    &LwSciSyncBackEndSyncpoint,
    &LwSciSyncBackEndSysmemSema,
    &LwSciSyncBackEndVidmemSema,
    &LwSciSyncBackEndSysmemSemaPayload64b,
    &LwSciSyncBackEndVidmemSemaPayload64b,
};

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCoreInitPrimitive(
    LwSciSyncInternalAttrValPrimitiveType primitiveType,
    LwSciSyncAttrList reconciledList,
    LwSciSyncCorePrimitive* primitive,
    bool needsAllocation)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCorePrimitive resultPrimitive = NULL;
    uint32_t index = 0U;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreCheckPrimitiveValues(&primitiveType, 1U);
    if (LwSciError_Success != error) {
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreAttrListValidate(reconciledList);
    if (LwSciError_Success != error) {
        LwSciCommonPanic();
    }
    if (NULL == primitive) {
        LwSciCommonPanic();
    }

    *primitive = NULL;

    /** create primitive struct */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    resultPrimitive = LwSciCommonCalloc(1,
            sizeof(struct LwSciSyncCorePrimitiveRec));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == resultPrimitive) {
        LWSCI_ERR_STR("Failed to allocate memory");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    resultPrimitive->ownsPrimitive = needsAllocation;
    resultPrimitive->type = primitiveType;
    index = (uint32_t)primitiveType;
    if ((sizeof(LwSciSyncPrimitiveOpsArray) /
                  sizeof(const LwSciSyncPrimitiveOps*)) <= index) {
        LWSCI_ERR_HEXUINT("Unsupported primitive type", primitiveType);
        LwSciCommonPanic();
    }
    resultPrimitive->ops = LwSciSyncPrimitiveOpsArray[index];
    if ((NULL == resultPrimitive->ops) || (NULL == resultPrimitive->ops->Init)) {
        LWSCI_ERR_HEXUINT("Unsupported primitive type", primitiveType);
        LwSciCommonPanic();
    }

    error = resultPrimitive->ops->Init(reconciledList, resultPrimitive);
    if (LwSciError_Success != error) {
        LwSciCommonFree(resultPrimitive);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    *primitive = resultPrimitive;

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreDeinitPrimitive(
    LwSciSyncCorePrimitive primitive)
{
    LWSCI_FNENTRY("");

    if (NULL != primitive) {
        if ((NULL == primitive->ops) || (NULL == primitive->ops->Deinit)) {
            LwSciCommonPanic();
        }
        primitive->ops->Deinit(primitive);
        LwSciCommonFree(primitive);
    }

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCorePrimitiveExport(
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
    LwSciSyncCorePrimitive primitive,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** data,
    size_t* length)
{
    LwSciError error = LwSciError_Success;

    LwSciCommonTransportBuf* corePrimitiveTxBuf = NULL;
    LwSciCommonTransportParams bufParams = { 0 };
    size_t totalValueSize = 0U;
    void* primitiveSpecificData = NULL;
    size_t primitiveSpecificDataSize = 0U;
    size_t len = 0U;
    uint32_t key = 0U;
    const void* value = NULL;

#if (LW_IS_SAFETY != 0)
    (void)ipcEndpoint;
#endif

    LWSCI_FNENTRY("");

    if (NULL == primitive) {
        LWSCI_ERR_STR("Null primitive. Panicking!!\n");
        LwSciCommonPanic();
    }

    if (NULL == data) {
        LWSCI_ERR_STR("Null data. Panicking!!\n");
        LwSciCommonPanic();
    }

    if (NULL == length) {
        LWSCI_ERR_STR("Null length. Panicking!!\n");
        LwSciCommonPanic();
    }

    bufParams.keyCount += 1U;
    totalValueSize += sizeof(primitive->type);

    bufParams.keyCount += 1U;
    totalValueSize += sizeof(primitive->id);

    if ((NULL == primitive->ops) || (NULL == primitive->ops->Export)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!\n");
        LwSciCommonPanic();
    }

    error = primitive->ops->Export(primitive, permissions,
            ipcEndpoint, &primitiveSpecificData,
            &primitiveSpecificDataSize);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (primitiveSpecificDataSize > 0U) {
        uint8_t addStatus = OP_SUCCESS;

        sizeAdd(totalValueSize, primitiveSpecificDataSize, &totalValueSize, &addStatus);
        if (OP_SUCCESS != addStatus) {
            LWSCI_ERR_SLONG("Primitive specific data size too large",
                      primitiveSpecificDataSize);
            LwSciCommonPanic();
        }

        bufParams.keyCount += 1U;
    }

    error = LwSciCommonTransportAllocTxBufferForKeys(bufParams, totalValueSize,
            &corePrimitiveTxBuf);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_primitive_data;
    }

    /* Serialize data into the desc */
    len = sizeof(primitive->type);
    value = (const void*)&primitive->type;
    key = (uint32_t)LwSciSyncCorePrimitiveKey_Type;
    error = LwSciCommonTransportAppendKeyValuePair(
            corePrimitiveTxBuf, key, len, value);
    if (LwSciError_Success != error) {
        /* corePrimitiveTxBuf is allocated using values provided by this
         * function. This means we computed something wrong when passing
         * bufParams to LwSciCommon. */
        LwSciCommonPanic();
    }

    len = sizeof(primitive->id);
    value = (const void*)&primitive->id;
    key = (uint32_t)LwSciSyncCorePrimitiveKey_Id;
    error = LwSciCommonTransportAppendKeyValuePair(
            corePrimitiveTxBuf, key, len, value);
    if (LwSciError_Success != error) {
        /* corePrimitiveTxBuf is allocated using values provided by this
         * function. This means we computed something wrong when passing
         * bufParams to LwSciCommon. */
        LwSciCommonPanic();
    }

    len = primitiveSpecificDataSize;
    value = (const void*)primitiveSpecificData;
    key = (uint32_t)LwSciSyncCorePrimitiveKey_Specific;
    if (len > 0U) {
        error = LwSciCommonTransportAppendKeyValuePair(
                corePrimitiveTxBuf, key, len, value);
        if (LwSciError_Success != error) {
            /* corePrimitiveTxBuf is allocated using values provided by this
             * function. This means we computed something wrong when passing
             * bufParams to LwSciCommon. */
            LwSciCommonPanic();
        }
    }

    LwSciCommonTransportPrepareBufferForTx(corePrimitiveTxBuf,
            data, length);

    LwSciCommonTransportBufferFree(corePrimitiveTxBuf);

free_primitive_data:
    LwSciCommonFree(primitiveSpecificData);

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

struct PrimitiveTagInfo {
    uint32_t tag;
    uint32_t expectedNum;
    uint32_t handledNum;
    bool optional;
};

static struct PrimitiveTagInfo* FindPrimitiveTagInfo(
    uint32_t key,
    struct PrimitiveTagInfo* tagInfo,
    size_t tagsNum)
{
    size_t i = 0U;
    struct PrimitiveTagInfo* result = NULL;

    for (i = 0U; i < tagsNum; ++i) {
        if (key == tagInfo[i].tag) {
            result = &tagInfo[i];
            break;
        }
    }

    return result;
}

static void LwSciSyncCorePrimitiveImportParamCheck(
    const void* data,
    size_t len,
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
    LwSciSyncCorePrimitive* primitive)
{
    if ((NULL == data) || (0U == len) || (NULL == primitive)) {
        LwSciCommonPanic();
    }

}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCorePrimitiveImport(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList reconciledList,
    const void* data,
    size_t len,
    LwSciSyncCorePrimitive* primitive)
{
    LwSciError error = LwSciError_Success;

    LwSciCommonTransportBuf* corePrimitiveRxbuf = NULL;
    LwSciCommonTransportParams params = { 0 };
    bool doneReading = false;
    LwSciSyncCorePrimitiveKey key;
    LwSciSyncCorePrimitive resultPrimitive = NULL;
    struct PrimitiveTagInfo* tmpInfo = NULL;
    struct PrimitiveTagInfo* info = NULL;
    struct PrimitiveTagInfo tagInfo[] = {
        {(uint32_t)LwSciSyncCorePrimitiveKey_Type, 1U, 0U, false},
        {(uint32_t)LwSciSyncCorePrimitiveKey_Id, 1U, 0U, false},
        {(uint32_t)LwSciSyncCorePrimitiveKey_Specific, 0U, 0U, true},
    };
    size_t numTags = sizeof(tagInfo) / sizeof(struct PrimitiveTagInfo);
    size_t i = 0U;

#if (LW_IS_SAFETY != 0)
    (void)ipcEndpoint;
#endif

    LwSciSyncCorePrimitiveImportParamCheck(data, len, primitive);

    LWSCI_FNENTRY("");

    /** create primitive struct */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    resultPrimitive = LwSciCommonCalloc(1,
            sizeof(struct LwSciSyncCorePrimitiveRec));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == resultPrimitive) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    resultPrimitive->ownsPrimitive = false;

    error = LwSciCommonTransportGetRxBufferAndParams(
            data, len, &corePrimitiveRxbuf,
            &params);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto free_prim;
    }
    do {
        uint32_t inputKey = 0U;
        size_t length = 0U;
        const void* value = NULL;

        error = LwSciCommonTransportGetNextKeyValuePair(
                corePrimitiveRxbuf, &inputKey,
                &length, &value, &doneReading);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_buf;
        }
        info = FindPrimitiveTagInfo(inputKey, tagInfo, numTags);
        if (NULL == info) {
            LWSCI_INFO("Unrecognized tag %u in primitive\n", inputKey);
            continue;
        }
        if (info->handledNum >= info->expectedNum) {
            LWSCI_ERR_UINT("Tag is not allowed here in primitive: \n", inputKey);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_buf;
        }
        LwSciCommonMemcpyS(&key, sizeof(key), &inputKey, sizeof(inputKey));
        if (LwSciSyncCorePrimitiveKey_Type == key) {
            LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
            resultPrimitive->type =
                    *(const LwSciSyncInternalAttrValPrimitiveType*)value;
            LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
            resultPrimitive->ops =
                    LwSciSyncPrimitiveOpsArray[(uint32_t)(resultPrimitive->type)];
            tmpInfo = FindPrimitiveTagInfo(
                    (uint32_t)LwSciSyncCorePrimitiveKey_Specific,
                    tagInfo, numTags);
            if (NULL == tmpInfo) {
                LwSciCommonPanic();
            }
            tmpInfo->expectedNum = 1U;
        } else if (LwSciSyncCorePrimitiveKey_Id == key) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
            resultPrimitive->id = *(const uint64_t*)value;
        } else if (LwSciSyncCorePrimitiveKey_Specific == key) {
            if ((NULL == resultPrimitive->ops) ||
                    (NULL == resultPrimitive->ops->Import)) {
                LWSCI_ERR_HEXUINT("Unsupported primitive type", resultPrimitive->type);
                LwSciCommonPanic();
            }
            error = resultPrimitive->ops->Import(ipcEndpoint, reconciledList,
                    value, length, resultPrimitive);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto free_buf;
            }
        } else {
            LWSCI_ERR_UINT("Unrecognized key despite performing a check before: \n",
                    inputKey);
            LwSciCommonPanic();
        }
        info->handledNum++;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    } while (doneReading == false);

    for (i = 0U; i < numTags; ++i) {
        if (!tagInfo[i].optional &&
                (tagInfo[i].expectedNum != tagInfo[i].handledNum)) {
            LWSCI_ERR_UINT("Missing tag in primitive: \n", tagInfo[i].tag);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_buf;
        }
    }

    /* TODO: remove in the future. Keeping for compatibility */
    info = FindPrimitiveTagInfo((uint32_t)LwSciSyncCorePrimitiveKey_Specific,
            tagInfo, numTags);
    if (NULL == info) {
        LwSciCommonPanic();
    }
    if (0U == info->handledNum) {
        error = resultPrimitive->ops->Import(0U, reconciledList,
                NULL, 0U, resultPrimitive);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto free_buf;
        }
    }

    *primitive = resultPrimitive;

free_buf:
    LwSciCommonTransportBufferFree(corePrimitiveRxbuf);

free_prim:
    if (LwSciError_Success != error) {
        LwSciCommonFree(resultPrimitive);
    }
fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCoreSignalPrimitive(
    LwSciSyncCorePrimitive primitive)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((NULL == primitive) || (NULL == primitive->ops) ||
            (NULL == primitive->ops->Signal)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!\n");
        LwSciCommonPanic();
    }

    error = primitive->ops->Signal(primitive);

    LWSCI_FNEXIT("");
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCoreWaitOnPrimitive(
    LwSciSyncCorePrimitive primitive,
    LwSciSyncCpuWaitContext waitContext,
    uint64_t id,
    uint64_t value,
    int64_t timeout_us)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((NULL == primitive) || (NULL == primitive->ops) ||
            (NULL == primitive->ops->WaitOn)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!\n");
        LwSciCommonPanic();
    }

    if ((timeout_us < -1) || (timeout_us > LwSciSyncFenceMaxTimeout)) {
        LWSCI_ERR_STR("Invalid timeout value");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = primitive->ops->WaitOn(
        primitive, waitContext, id, value, timeout_us);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
uint64_t LwSciSyncCorePrimitiveGetNewFence(
    LwSciSyncCorePrimitive primitive)
{
    if ((NULL == primitive) || (NULL == primitive->ops) ||
            (NULL == primitive->ops->GetNewFence)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!\n");
        LwSciCommonPanic();
    }
    return primitive->ops->GetNewFence(primitive);
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
uint64_t LwSciSyncCorePrimitiveGetId(
    LwSciSyncCorePrimitive primitive)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    if (NULL == primitive) {
        LWSCI_ERR_STR("Null primitive. Panicking!!\n");
        LwSciCommonPanic();
    }
    return primitive->id;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreCopyCpuPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len)
{
    if (NULL == primitiveType) {
        LWSCI_ERR_STR("Null primitiveType. Panicking!!\n");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(primitiveType, len,
            LwSciSyncCoreSupportedPrimitives,
            sizeof(LwSciSyncCoreSupportedPrimitives));
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreCopyC2cCpuPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len)
{
    if (NULL == primitiveType) {
        LWSCI_ERR_STR("Null primitiveType. Panicking!!\n");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(primitiveType, len,
            LwSciSyncCoreSupportedC2cCpuPrimitives,
            sizeof(LwSciSyncCoreSupportedC2cCpuPrimitives));
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreGetSupportedPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len)
{
    if (NULL == primitiveType) {
        LWSCI_ERR_STR("Null primitiveType. Panicking!!\n");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(primitiveType, len,
            LwSciSyncCoreSupportedPrimitives,
            sizeof(LwSciSyncCoreSupportedPrimitives));
}

void LwSciSyncCoreGetDeterministicPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len)
{
    if (primitiveType == NULL) {
        LWSCI_ERR_STR("Null primitiveType. Panicking!!");
        LwSciCommonPanic();
    }
    LwSciCommonMemcpyS(primitiveType, len,
            LwSciSyncCoreDeterministicPrimitives,
            sizeof(LwSciSyncCoreDeterministicPrimitives));
}

LwSciError LwSciSyncCorePrimitiveGetSpecificData(
    LwSciSyncCorePrimitive primitive,
    void** data)
{
    LwSciError error = LwSciError_Success;

    if ((NULL == primitive) || (NULL == primitive->ops)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == primitive->ops->GetSpecificData) {
        LWSCI_ERR_STR("Unsupported primitive operation.\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    primitive->ops->GetSpecificData(primitive, data);

fn_exit:
    return error;
}

LwSciError LwSciSyncCoreValidatePrimitiveIdValue(
    LwSciSyncCorePrimitive primitive,
    uint64_t id,
    uint64_t value)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if ((primitive == NULL) || (primitive->ops == NULL)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!\n");
        LwSciCommonPanic();
    }

    error = primitive->ops->CheckIdValue(primitive, id, value);
    if (LwSciError_Success != error) {
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
LwSciError LwSciSyncCorePrimitiveGetC2cSyncHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncHandle* syncHandle)
{
    LwSciError error = LwSciError_Success;

    if ((NULL == primitive) || (NULL == primitive->ops)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!");
        LwSciCommonPanic();
    }

    if (NULL == syncHandle) {
        LWSCI_ERR_STR("NULL syncHandle!");
        LwSciCommonPanic();
    }

    if (NULL == primitive->ops->GetC2cSyncHandle) {
        LWSCI_ERR_STR("C2C handles are not supported for this primitive");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = primitive->ops->GetC2cSyncHandle(primitive, syncHandle);

fn_exit:
    return error;
}

LwSciError LwSciSyncCorePrimitiveGetC2cRmHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncRmHandle* syncRmHandle)
{
    LwSciError error = LwSciError_Success;

    if ((NULL == primitive) || (NULL == primitive->ops)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!");
        LwSciCommonPanic();
    }

    if (NULL == syncRmHandle) {
        LWSCI_ERR_STR("NULL syncRmHandle!");
        LwSciCommonPanic();
    }

    if (NULL == primitive->ops->GetC2cRmHandle) {
        LWSCI_ERR_STR("This API is not supported for this primitive");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = primitive->ops->GetC2cRmHandle(primitive, syncRmHandle);

fn_exit:
    return error;
}
#endif

LwSciError LwSciSyncCorePrimitiveImportThreshold(
    LwSciSyncCorePrimitive primitive,
    uint64_t* threshold)
{
    if ((NULL == primitive) || (NULL == primitive->ops)) {
        LWSCI_ERR_STR("Null primitive ops. Panicking!!");
        LwSciCommonPanic();
    }

    if (NULL == threshold) {
        LWSCI_ERR_STR("NULL threshold!");
        LwSciCommonPanic();
    }

    return primitive->ops->ImportThreshold(primitive, threshold);
}
