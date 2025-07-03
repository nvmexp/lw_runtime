/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <lwscisync.h>
#include <lwscisync_test_signaler.h>
#include <lwscisync_test_waiter.h>
#include <umd.h>
#include "lwscisync_peer.h"

LwSciError LwSciSyncTest_FillCpuAttrListAutoPerm(LwSciSyncAttrList list)
{
    LwSciSyncAttrKeyValuePair keyValue[2];
    bool cpuSignaler = true;
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuSignaler;
    keyValue[0].len = sizeof(cpuSignaler);
    LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_Auto;
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*) &cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    return LwSciSyncAttrListSetAttrs(list, keyValue, 2);
}

LwSciError LwSciSyncTest_FillCpuSignalerAttrList(LwSciSyncAttrList list)
{
    LwSciSyncAttrKeyValuePair keyValue[2];
    memset(keyValue, 0, sizeof(keyValue));
    bool cpuSignaler = true;
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuSignaler;
    keyValue[0].len = sizeof(cpuSignaler);
    LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_SignalOnly;
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*) &cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    return LwSciSyncAttrListSetAttrs(list, keyValue, 2);
}

LwSciError LwSciSyncTest_FillUmdSignalerAttrList(LwSciSyncAttrList list)
{
    return umdAddLwSciSyncAttr(LwSciSyncAccessPerm_SignalOnly, list);
}

#ifdef LWSCISYNC_EMU_SUPPORT
LwSciError LwSciSyncTest_FillUmdExternalPrimitiveInfo(
    LwSciSyncAttrList list,
    TestResources res)
{
    return umdAddExternalPrimitiveInfo(list, res);
}
#endif

LwSciError LwSciSyncTest_FillCpuSignalerSysmemSemaAttrList(LwSciSyncAttrList list)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue[2];
    memset(internalKeyValue, 0, sizeof(internalKeyValue));
    uint32_t primitiveCnt = 1U;
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo =
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
    internalKeyValue[0U].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
    internalKeyValue[0U].value = (const void*) &primitiveInfo;
    internalKeyValue[0U].len = sizeof(primitiveInfo);
    internalKeyValue[1U].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
    internalKeyValue[1U].value = (const void*) &primitiveCnt;
    internalKeyValue[1U].len = sizeof(primitiveCnt);

    err = LwSciSyncTest_FillCpuSignalerAttrList(list);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(list, internalKeyValue, 2U);
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

LwSciError LwSciSyncTest_FillCpuSignalerSysmemSemaPayload64bAttrList(LwSciSyncAttrList list)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue[2];
    memset(internalKeyValue, 0, sizeof(internalKeyValue));
    uint32_t primitiveCnt = 1U;
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo =
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b;
    internalKeyValue[0U].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
    internalKeyValue[0U].value = (const void*) &primitiveInfo;
    internalKeyValue[0U].len = sizeof(primitiveInfo);
    internalKeyValue[1U].attrKey =
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
    internalKeyValue[1U].value = (const void*) &primitiveCnt;
    internalKeyValue[1U].len = sizeof(primitiveCnt);

    err = LwSciSyncTest_FillCpuSignalerAttrList(list);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(list, internalKeyValue, 2U);
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

LwSciError LwSciSyncTest_FillTimestampsSignalerAttrList(LwSciSyncAttrList list)
{
    LwSciError err = LwSciError_Success;
    bool cpuSignaler = true;
    LwSciSyncAccessPerm accessPerm = LwSciSyncAccessPerm_SignalOnly;
    bool requireTimestamps = true;

    LwSciSyncAttrValTimestampInfo tinfo = {
        .format = LwSciSyncTimestampFormat_8Byte,
        .scaling = {
            .scalingFactorNumerator = 1U,
            .scalingFactorDenominator = 1U,
            .sourceOffset = 0U,
        },
    };
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncPeer::attrs.defaultPlatformPrimitive };

    LwSciSyncAttrKeyValuePair keyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuSignaler,
             .len = sizeof(cpuSignaler),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &accessPerm,
             .len = sizeof(accessPerm),
        },
    };

    uint32_t signalerPrimitiveCount = 1U;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerTimestampInfo,
             .value = (void*) &tinfo,
             .len = sizeof(LwSciSyncAttrValTimestampInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
             .value = (void*) &primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
             .value = (void*)&signalerPrimitiveCount,
             .len = sizeof(signalerPrimitiveCount),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(list, keyValue,
            sizeof(keyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(list, internalKeyValue,
            sizeof(internalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

LwSciError LwSciSyncTest_FillTimestampsSignalerImplicitAttrList(LwSciSyncAttrList list)
{
    LwSciError err = LwSciError_Success;
    bool cpuSignaler = true;
    LwSciSyncAccessPerm accessPerm = LwSciSyncAccessPerm_SignalOnly;
    bool requireTimestamps = true;

    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncPeer::attrs.defaultPlatformPrimitive };

    LwSciSyncAttrKeyValuePair keyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuSignaler,
             .len = sizeof(cpuSignaler),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &accessPerm,
             .len = sizeof(accessPerm),
        },
    };

    uint32_t signalerPrimitiveCount = 1U;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
             .value = (void*) &primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
        {
            .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
            .value = (void*)&signalerPrimitiveCount,
            .len = sizeof(signalerPrimitiveCount),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(list, keyValue,
            sizeof(keyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(list, internalKeyValue,
            sizeof(internalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

LwSciError LwSciSyncTest_FillCpuWaiterAttrList(LwSciSyncAttrList list)
{
    LwSciSyncAttrKeyValuePair keyValue[2];
    memset(keyValue, 0, sizeof(keyValue));
    bool cpuWaiter = true;
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*) &cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    return LwSciSyncAttrListSetAttrs(list, keyValue, 2);
}

LwSciError LwSciSyncTest_FillUmdWaiterAttrList(LwSciSyncAttrList list)
{
    return umdAddLwSciSyncAttr(LwSciSyncAccessPerm_WaitOnly, list);
}

LwSciError LwSciSyncTest_FillCpuWaiterSysmemSemaAttrList(LwSciSyncAttrList list)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue;
    memset(&internalKeyValue, 0, sizeof(internalKeyValue));
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo =
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
    internalKeyValue.attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
    internalKeyValue.value = (const void*) &primitiveInfo;
    internalKeyValue.len = sizeof(primitiveInfo);

    err = LwSciSyncTest_FillCpuWaiterAttrList(list);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(list, &internalKeyValue, 1U);
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

LwSciError LwSciSyncTest_FillCpuWaiterSysmemSemaPayload64bAttrList(LwSciSyncAttrList list)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrKeyValuePair internalKeyValue;
    memset(&internalKeyValue, 0, sizeof(internalKeyValue));
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo =
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b;
    internalKeyValue.attrKey =
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo;
    internalKeyValue.value = (const void*) &primitiveInfo;
    internalKeyValue.len = sizeof(primitiveInfo);

    err = LwSciSyncTest_FillCpuWaiterAttrList(list);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(list, &internalKeyValue, 1U);
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

LwSciError LwSciSyncTest_FillTimestampsWaiterAttrList(LwSciSyncAttrList list)
{
    LwSciError err = LwSciError_Success;
    bool cpuWaiter = true;
    LwSciSyncAccessPerm accessPerm = LwSciSyncAccessPerm_WaitOnly;
    bool requireTimestamps = true;

    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncPeer::attrs.defaultPlatformPrimitive };

    LwSciSyncAttrKeyValuePair keyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuWaiter,
             .len = sizeof(cpuWaiter),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &accessPerm,
             .len = sizeof(accessPerm),
        },
        {    .attrKey = LwSciSyncAttrKey_WaiterRequireTimestamps,
             .value = (void*) &requireTimestamps,
             .len = sizeof(bool),
        },
    };
    LwSciSyncInternalAttrKeyValuePair internalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
             .value = (void*) &primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(list, keyValue,
            sizeof(keyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(list, internalKeyValue,
            sizeof(internalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}
