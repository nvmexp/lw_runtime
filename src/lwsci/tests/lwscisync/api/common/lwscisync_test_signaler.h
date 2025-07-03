/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_LWSCISYNC_TEST_SIGNALER_H
#define INCLUDED_LWSCISYNC_TEST_SIGNALER_H

#include <lwscisync.h>
#include <lwscisync_test_common.h>

#ifdef __cplusplus
extern "C" {
#endif

LwSciError LwSciSyncTest_FillCpuAttrListAutoPerm(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillCpuSignalerAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillTimestampsSignalerImplicitAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillUmdSignalerAttrList(LwSciSyncAttrList list);
#ifdef LWSCISYNC_EMU_SUPPORT
LwSciError LwSciSyncTest_FillUmdExternalPrimitiveInfo(LwSciSyncAttrList list, TestResources res);
#endif
LwSciError LwSciSyncTest_FillCpuSignalerSysmemSemaAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillCpuSignalerSysmemSemaPayload64bAttrList(LwSciSyncAttrList list);
LwSciError LwSciSyncTest_FillTimestampsSignalerAttrList(LwSciSyncAttrList list);
LwSciError cpuSignalStream(
    struct ThreadConf* conf,
    struct StreamResources* resources);
LwSciSyncTestStatus standardSignaler(void* args);

#ifdef __cplusplus
}
#endif

#endif
