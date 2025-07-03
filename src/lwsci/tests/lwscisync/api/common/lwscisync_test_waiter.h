/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_LWSCISYNC_TEST_WAITER_H
#define INCLUDED_LWSCISYNC_TEST_WAITER_H

#include <lwscisync.h>
#include <lwscisync_test_common.h>

#ifdef __cplusplus
extern "C" {
#endif

LwSciError LwSciSyncTest_FillCpuWaiterAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillUmdWaiterAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillCpuWaiterSysmemSemaAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillCpuWaiterSysmemSemaPayload64bAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillTimestampsWaiterAttrList(LwSciSyncAttrList list);
LwSciError cpuWaitStream(
    struct ThreadConf* conf,
    struct StreamResources* resources);
LwSciSyncTestStatus standardWaiter(void* args);

#ifdef __cplusplus
}
#endif

#endif
