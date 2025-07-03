/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscisync_attribute_core_cluster.h"
#include "lwscicommon_libc.h"


LwSciError LwSciSyncCoreSignalerExternalPrimitiveAttrAlloc(
    size_t valueCount,
    LwSciSyncCoreAttrList* coreAttrList)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t j = 0U;

    /** Allocate memory for external primitive info */
    for (i = 0U; i < valueCount; i++) {
        for (j = 0U; j < MAX_PRIMITIVE_TYPE; j++) {
            coreAttrList[i].attrs.signalerExternalPrimitiveInfo[j] =
                    (LwSciSyncPrimitiveInfo*)LwSciCommonCalloc(1U,
                    sizeof(LwSciSyncPrimitiveInfo));
            if (NULL == coreAttrList[i].attrs.signalerExternalPrimitiveInfo[j]) {
                LWSCI_ERR_STR("memory alloc failed for external primitive info.\n");
                error = LwSciError_InsufficientMemory;
                goto fn_exit;
            }
        }
    }
fn_exit:
    return error;
}

void LwSciSyncCoreSignalerExternalPrimitiveAttrFree(
    LwSciSyncCoreAttrList* coreAttrList)
{
    size_t j = 0U;

    /** Free external primitive info memory */
    for (j = 0U; j < MAX_PRIMITIVE_TYPE; j++) {
        if (coreAttrList->attrs.signalerExternalPrimitiveInfo[j]->
                simplePrimitiveInfo.primitiveType ==
                LwSciSyncInternalAttrValPrimitiveType_Syncpoint) {
            if (coreAttrList->attrs.signalerExternalPrimitiveInfo[j]->
                    simplePrimitiveInfo.ids) {
                LwSciCommonFree(coreAttrList->attrs.
                        signalerExternalPrimitiveInfo[j]->simplePrimitiveInfo.ids);
            }
        }
        LwSciCommonFree(coreAttrList->attrs.signalerExternalPrimitiveInfo[j]);
    }
}


LwSciError LwSciSyncCoreCopyAttrVal(
    void* val,
    CoreAttribute* attribute,
    size_t maxSize)
{
    LwSciError error = LwSciError_Success;
    /* Handle setting of external primitive info */
    if (LwSciSyncCoreIndexToKey(attribute->index) ==
            LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo) {
        error = LwSciSyncCoreCopySignalerExternalPrimitiveInfo(val, attribute->value,
                    (attribute->len/sizeof(void*)));
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
    } else {
        LwSciCommonMemcpyS(val, maxSize, attribute->value, attribute->len);
    }

fn_exit:
    return error;
}

LwSciError LwSciSyncCoreCopySignalerExternalPrimitiveInfo(
    LwSciSyncPrimitiveInfo** dest,
    LwSciSyncPrimitiveInfo* const* src,
    size_t cnt)
{
    LwSciError error = LwSciError_Success;
    size_t i = 0U;
    size_t size = 0U;
    LwSciSyncInternalAttrValPrimitiveType primitiveType =
            LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    const LwSciSyncSimplePrimitiveInfo* simplePrimitiveInfo = NULL;

    LWSCI_FNENTRY("");

    for (i = 0U; i < cnt; i++) {
        const LwSciSyncPrimitiveInfo* primitiveInfoPtr = *(src + i);
        primitiveType = *(const LwSciSyncInternalAttrValPrimitiveType*)
                primitiveInfoPtr;
        switch (primitiveType) {
            case LwSciSyncInternalAttrValPrimitiveType_Syncpoint:
                size = sizeof(LwSciSyncSimplePrimitiveInfo);
                simplePrimitiveInfo = (const LwSciSyncSimplePrimitiveInfo*)primitiveInfoPtr;
                dest[i]->simplePrimitiveInfo.primitiveType = primitiveType;
                dest[i]->simplePrimitiveInfo.ids =
                        LwSciCommonCalloc(simplePrimitiveInfo->numIds, sizeof(uint64_t));
                if (dest[i]->simplePrimitiveInfo.ids == NULL) {
                    LWSCI_ERR_STR("Failed to allocate memory\n");
                    error = LwSciError_InsufficientMemory;
                    goto fn_exit;
                }
                LwSciCommonMemcpyS(dest[i]->simplePrimitiveInfo.ids,
                        (sizeof(uint64_t) * simplePrimitiveInfo->numIds),
                        (void*)simplePrimitiveInfo->ids,
                        (sizeof(uint64_t) * simplePrimitiveInfo->numIds));
                dest[i]->simplePrimitiveInfo.numIds = simplePrimitiveInfo->numIds;
                break;
            case LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore:
            case LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore:
            case LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b:
            case LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphorePayload64b:
                size = sizeof(LwSciSyncSemaphorePrimitiveInfo);
                LwSciCommonMemcpyS(dest[i], sizeof(LwSciSyncPrimitiveInfo),
                        (const void*)primitiveInfoPtr, size);
                break;
            default:
                LWSCI_ERR_STR("Unrecognized primitive type.\n");
                LwSciCommonPanic();
        }
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}
