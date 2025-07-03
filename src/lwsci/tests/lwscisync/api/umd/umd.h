/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_UMD_H
#define INCLUDED_UMD_H

#include <lwscisync_internal.h>
#include <lwscisync_test_common.h>

#ifdef __cplusplus
extern "C" {
#endif

LwSciError umdGetPostLwSciSyncFence(
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFence* syncFence);

LwSciError umdWaitOnPreLwSciSyncFence(
    TestResources resource,
    LwSciSyncFence* syncFence);

LwSciError LwRmGpu_TestSetup(
    TestResources* resources);

LwSciError LwRmGpu_TestMapSemaphore(
    TestResources res,
    LwSciSyncObj syncObj);

void LwRmGpu_TestTeardown(
    TestResources res);

LwSciError MockUmdSignalStreamFrame(
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFence* syncFence,
    uint32_t slotIndex);

LwSciError umdAddLwSciSyncAttr(
    LwSciSyncAccessPerm perm,
    LwSciSyncAttrList attrList);

#ifdef LWSCISYNC_EMU_SUPPORT
LwSciError umdAddExternalPrimitiveInfo(
    LwSciSyncAttrList list,
    TestResources res);
#endif

#ifdef __cplusplus
}
#endif

#endif
