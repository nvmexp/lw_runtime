/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <lwscisync_test_common.h>
#include <lwscisync_test_signaler.h>
#include <lwscisync_test_waiter.h>
#include <lwscisync.h>
#include <umd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

/** @jama{9318380} Multiple exports - V1
 * -multiple_exports
 * single thread test case with a longer export path
 * causes an expected error: [ERROR: LwSciSyncCoreSyncpointWaitOn] LwRmFenceWait failed
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, MultipleExports, 9318380)
{
    LwSciError err;
    /* below endpoint simulate topology:
     * signaler <-> in <-> waiter
     */
    LwSciIpcEndpoint sigEndpoint;
    LwSciIpcEndpoint waitEndpoint;
    LwSciIpcEndpoint inEndpoint_sig;
    LwSciIpcEndpoint inEndpoint_wait;
    LwSciSyncModule module = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    LwSciSyncAttrList waiterAttrList = NULL;
    LwSciSyncAttrList inAttrList = NULL;
    LwSciSyncAttrList signalerAttrList = NULL;
    size_t waiterAttrListSize = 0U;
    void* waiterListDesc;
    LwSciSyncAttrList inUnreconciledLists[2] = {0};
    LwSciSyncAttrList inAppendedList = NULL;
    size_t inAttrListSize = 0U;
    void* inListDesc;
    LwSciSyncAttrList importedUnreconciledList = NULL;
    LwSciSyncAttrList unreconciledLists[3] = {NULL};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncObj syncObj = NULL;
    void* objAndListDesc = NULL;
    size_t objAndListSize = 0U;
    LwSciSyncObj inSyncObj = NULL;
    void* inObjAndListDesc = NULL;
    size_t inObjAndListSize = 0U;
    LwSciSyncObj waitSyncObj = NULL;

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciIpcOpenEndpoint("lwscisync_a_0", &sigEndpoint);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciIpcOpenEndpoint("lwscisync_a_1", &inEndpoint_sig);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciIpcOpenEndpoint("lwscisync_b_1", &waitEndpoint);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciIpcOpenEndpoint("lwscisync_b_0", &inEndpoint_wait);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncModuleOpen(&module);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncCpuWaitContextAlloc(module, &waitContext);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = LwSciSyncTest_FillCpuWaiterAttrList(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &inAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = LwSciSyncTest_FillCpuWaiterAttrList(inAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = LwSciSyncTest_FillCpuSignalerAttrList(signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* communicate unreconciled lists */

    err = LwSciSyncAttrListIpcExportUnreconciled(
        &waiterAttrList, 1,
        waitEndpoint, &waiterListDesc, &waiterAttrListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListIpcImportUnreconciled(
        module, inEndpoint_wait,
        waiterListDesc, waiterAttrListSize,
        &inUnreconciledLists[0]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    inUnreconciledLists[1] = inAttrList;

    err = LwSciSyncAttrListAppendUnreconciled(
        inUnreconciledLists, 2,
        &inAppendedList);

    err = LwSciSyncAttrListIpcExportUnreconciled(
        &inAppendedList, 1,
        inEndpoint_sig, &inListDesc, &inAttrListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListIpcImportUnreconciled(
        module, sigEndpoint,
        inListDesc, inAttrListSize,
        &importedUnreconciledList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* reconcile */

    unreconciledLists[0] = signalerAttrList;
    unreconciledLists[1] = importedUnreconciledList;

    err = LwSciSyncAttrListReconcile(unreconciledLists, 2, &reconciledList,
            &newConflictList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncObjAlloc(reconciledList, &syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* export reconciled and object */

    err = LwSciSyncIpcExportAttrListAndObj(
        syncObj,
        LwSciSyncAccessPerm_WaitOnly, sigEndpoint,
        &objAndListDesc, &objAndListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncIpcImportAttrListAndObj(
        module, inEndpoint_sig,
        objAndListDesc, objAndListSize,
        inUnreconciledLists, 2,
        LwSciSyncAccessPerm_WaitOnly, 10000, &inSyncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncIpcExportAttrListAndObj(
        inSyncObj,
        LwSciSyncAccessPerm_WaitOnly, inEndpoint_wait,
        &inObjAndListDesc, &inObjAndListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* additionally verify that empty array can be passed */
    err = LwSciSyncIpcImportAttrListAndObj(
        module, waitEndpoint,
        inObjAndListDesc, inObjAndListSize,
        NULL, 0U,
        LwSciSyncAccessPerm_WaitOnly, 10000, &waitSyncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }
    LwSciSyncObjFree(waitSyncObj);

    err = LwSciSyncIpcImportAttrListAndObj(
        module, waitEndpoint,
        inObjAndListDesc, inObjAndListSize,
        &waiterAttrList, 1,
        LwSciSyncAccessPerm_WaitOnly, 10000, &waitSyncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* streaming */

    {
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        LwSciSyncFenceIpcExportDescriptor fenceDesc;

        /* generate a fence */
        err = LwSciSyncObjGenerateFence(syncObj, &syncFence);
        if (err != LwSciError_Success) {
            goto fail;
        }

        /* communicate the fence */

        err = LwSciSyncIpcExportFence(&syncFence, sigEndpoint,
                                      &fenceDesc);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }
        LwSciSyncFenceClear(&syncFence);

        err = LwSciSyncIpcImportFence(inSyncObj,
                                      &fenceDesc,
                                      &syncFence);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }

        err = LwSciSyncIpcExportFence(&syncFence, inEndpoint_wait,
                                      &fenceDesc);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }
        LwSciSyncFenceClear(&syncFence);

        err = LwSciSyncIpcImportFence(waitSyncObj,
                                      &fenceDesc,
                                      &syncFence);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }

        /* wait and signal */

        NegativeTestPrint();
        err = LwSciSyncFenceWait(&syncFence,
                waitContext, 100);
        if (err != LwSciError_Timeout) {
            printf("err is x%x\n", err);
            goto streaming_fail;
        }

        err = LwSciSyncObjSignal(syncObj);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }

        err = LwSciSyncFenceWait(&syncFence,
                waitContext, -1);
        if (err != LwSciError_Success) {
            printf("err is x%x\n", err);
            goto streaming_fail;
        }

    streaming_fail:

        LwSciSyncFenceClear(&syncFence);
    }

fail:
    LwSciSyncObjFree(waitSyncObj);
    LwSciSyncAttrListAndObjFreeDesc(inObjAndListDesc);
    LwSciSyncObjFree(inSyncObj);
    LwSciSyncAttrListAndObjFreeDesc(objAndListDesc);

    LwSciSyncObjFree(syncObj);
    LwSciSyncAttrListFree(reconciledList);

    LwSciSyncAttrListFree(importedUnreconciledList);
    LwSciSyncAttrListFreeDesc(inListDesc);
    LwSciSyncAttrListFree(inUnreconciledLists[0]);
    LwSciSyncAttrListFreeDesc(waiterListDesc);

    LwSciSyncAttrListFree(inAppendedList);
    LwSciSyncAttrListFree(waiterAttrList);
    LwSciSyncAttrListFree(inAttrList);
    LwSciSyncAttrListFree(signalerAttrList);

    LwSciSyncCpuWaitContextFree(waitContext);
    LwSciSyncModuleClose(module);

    LwSciIpcCloseEndpoint(inEndpoint_wait);
    LwSciIpcCloseEndpoint(waitEndpoint);
    LwSciIpcCloseEndpoint(inEndpoint_sig);
    LwSciIpcCloseEndpoint(sigEndpoint);
    LwSciIpcDeinit();

    ASSERT_EQ(err, LwSciError_Success);
}
