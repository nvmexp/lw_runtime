/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

/*
 * This file illustrates the sample UMD APIs using LwSciSync internal APIs.
 */

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <umd.h>

LwSciError umdAddLwSciSyncAttr(LwSciSyncAccessPerm perm,
                               LwSciSyncAttrList attrList)
{
    LwSciError err;
    uint32_t count = 0;
    LwSciSyncAttrKeyValuePair publicKeyValue = {
        .attrKey = LwSciSyncAttrKey_RequiredPerm,
        .value = (const void*) &perm,
        .len = sizeof(perm)
    };
    LwSciSyncInternalAttrKeyValuePair internalKeyValue[2];
    uint32_t primitiveCnt = 2U;
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[2] =
        {LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
         LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};

    /* Choose attributes based on requested access permissions */
    if ((perm == LwSciSyncAccessPerm_SignalOnly) ||
        (perm == LwSciSyncAccessPerm_WaitSignal)) {
        internalKeyValue[count].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
        internalKeyValue[count].value = (const void*) &primitiveInfo[0];
        internalKeyValue[count].len = sizeof(primitiveInfo);
        count++;
        internalKeyValue[count].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
        internalKeyValue[count].value = (const void*) &primitiveCnt;
        internalKeyValue[count].len = sizeof(primitiveCnt);
        count++;
    } else if ((perm == LwSciSyncAccessPerm_WaitOnly) ||
               (perm == LwSciSyncAccessPerm_WaitSignal)) {
        internalKeyValue[count].attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
        internalKeyValue[count].value =
            (const void*) &primitiveInfo[0];
        internalKeyValue[count].len = sizeof(primitiveInfo);
        count++;
    }

    /* Set values in attribute list */
    err = LwSciSyncAttrListSetAttrs(attrList, &publicKeyValue, 1);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = LwSciSyncAttrListSetInternalAttrs(attrList,
                                            internalKeyValue, count);
    if (err != LwSciError_Success) {
        goto fail;
    }

    return LwSciError_Success;

fail:
    return err;
}
