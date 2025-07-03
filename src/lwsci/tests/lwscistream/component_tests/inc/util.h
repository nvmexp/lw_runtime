//! \file
//! \brief LwSciStream APIs unit testing utils.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef UTIL_H
#define UTIL_H

#include "gtest/gtest.h"
#include "lwscistream.h"

#define CHECK_ERR_OR_PANIC(err, func) {                                       \
    ASSERT_EQ(err, func);                                                     \
}

#define CHNAME_MAX_LEN    14

//! \brief Define end point info for ipc channel
typedef struct {
    //! \brief channel name
    char chname[CHNAME_MAX_LEN+1];
    //! \brief LwIPC handle
    LwSciIpcEndpoint endpoint = 0U;
    //! \brief channel info
    LwSciIpcEndpointInfo info;
#ifdef __linux__
    //! \brief LwIPC event handle
    // Change LwSciIpcEventNotifier type to int32_t  Refer bug 200532359
    int32_t eventHandle = 0;
#endif
} Endpoint;


// LwSciBuf util functions

static void makeRawBufferAttrList(
    LwSciBufModule bufModule,
    LwSciBufAttrList &attrList)
{
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t rawsize = (128 * 1024);
    uint64_t align = (4 * 1024);
    bool cpuaccess_flag = true;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    LwSciBufAttrKeyValuePair rawbuffattrs[] = {
        { LwSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { LwSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
        { LwSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
        { LwSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
            sizeof(cpuaccess_flag) },
        { LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) }
    };

    ASSERT_EQ(LwSciError_Success,
        LwSciBufAttrListCreate(bufModule, &attrList));

    ASSERT_EQ(LwSciError_Success,
        LwSciBufAttrListSetAttrs(attrList, rawbuffattrs,
            sizeof(rawbuffattrs) / sizeof(LwSciBufAttrKeyValuePair)));
}

static void makeRawBuffer(
    const LwSciBufAttrList& attrList,
    LwSciBufObj &bufObj)
{
    LwSciBufAttrList unreconciledAttrList[1] = { attrList };
    LwSciBufAttrList reconciledAttrList = nullptr;
    LwSciBufAttrList conflictAttrList = nullptr;

    ASSERT_EQ(LwSciError_Success,
        LwSciBufAttrListReconcile(
            unreconciledAttrList,
            1U,
            &reconciledAttrList,
            &conflictAttrList));

    ASSERT_EQ(LwSciError_Success,
        LwSciBufObjAlloc(reconciledAttrList, &bufObj));

    if (reconciledAttrList != nullptr) {
        LwSciBufAttrListFree(reconciledAttrList);
    }
    if (conflictAttrList != nullptr) {
        LwSciBufAttrListFree(conflictAttrList);
    }
}

// LwSciSync util functions

static void cpuSignalerAttrList(
    LwSciSyncModule syncModule,
    LwSciSyncAttrList& list)
{
    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListCreate(syncModule, &list));

    LwSciSyncAttrKeyValuePair keyValue[2];
    bool cpuSignaler = true;
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*)&cpuSignaler;
    keyValue[0].len = sizeof(cpuSignaler);
    LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_SignalOnly;
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListSetAttrs(list, keyValue, 2));
}

static void cpuWaiterAttrList(
    LwSciSyncModule syncModule,
    LwSciSyncAttrList& list)
{
    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListCreate(syncModule, &list));

    LwSciSyncAttrKeyValuePair keyValue[2];
    bool cpuWaiter = true;
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    LwSciSyncAccessPerm cpuPerm = LwSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListSetAttrs(list, keyValue, 2));
}

static void getSyncObj(
    LwSciSyncModule syncModule,
    LwSciSyncObj& syncObj)
{
    LwSciSyncAttrList unreconciledList[2];
    LwSciSyncAttrList reconciledList;
    LwSciSyncAttrList conflictList;
    cpuSignalerAttrList(syncModule, unreconciledList[0]);
    cpuWaiterAttrList(syncModule, unreconciledList[1]);
    EXPECT_EQ(LwSciError_Success,
        LwSciSyncAttrListReconcile(unreconciledList, 2,
            &reconciledList, &conflictList));

    EXPECT_EQ(LwSciError_Success,
        LwSciSyncObjAlloc(reconciledList, &syncObj));

    LwSciSyncAttrListFree(unreconciledList[0]);
    LwSciSyncAttrListFree(unreconciledList[1]);
    LwSciSyncAttrListFree(reconciledList);
}

#endif // !UTIL_H
