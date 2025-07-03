/*
 * Copyright (c) 2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#include <lwscisync.h>
#include <lwscisync_test_common.h>
#include <lwscisync_test_signaler.h>
#include <lwscisync_test_waiter.h>
#include <lwscisync_internal.h>


/** @jama{13561103} Failure on trying to set Attribute on appended list
 *
 *   This test creates two empty lists, appends them, and tries to set attribute
 *   on appended attribute list.
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, SetAttrOnAppendedList, 111111)
{
    LwSciError err = LwSciError_Success;

    LwSciSyncModule module = NULL;
    LwSciSyncAttrList appendedList = NULL;
    LwSciSyncAttrList unreconciledList[2] = { NULL };

    err = LwSciSyncModuleOpen(&module);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledList[0]);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledList[1]);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListAppendUnreconciled(unreconciledList, 2,
            &appendedList);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    NegativeTestPrint();
    err = LwSciSyncTest_FillUmdWaiterAttrList(appendedList);
    EXPECT_EQ(err, LwSciError_BadParameter);


fail:
    /* Free Attribute list objects */
    if (appendedList != NULL) {
        LwSciSyncAttrListFree(appendedList);
    }
    if (unreconciledList[0] != NULL) {
        LwSciSyncAttrListFree(unreconciledList[0]);
    }
    if (unreconciledList[1] != NULL) {
        LwSciSyncAttrListFree(unreconciledList[1]);
    }

    LwSciSyncModuleClose(module);
}
