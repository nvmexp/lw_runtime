/*
 * Copyright (c) 2019-2022 LWPU Corporation.  All Rights Reserved.
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
#include <cinttypes>

static LwSciSyncTestStatus
auto_perms_export(TestInfo* info)
{
    LwSciSyncTestStatus status = LwSciSyncTestStatus::Success;
    struct ThreadConf conf = {0};
    pid_t peers[5] = {0};

    conf.info = info;
    conf.objExportPerm = LwSciSyncAccessPerm_Auto;
    conf.objImportPerm = LwSciSyncAccessPerm_Auto;

    if ((peers[0] = fork()) == 0) {
        const char* upstream[] = {
            "lwscisync_a_0",
            "lwscisync_b_0",
        };
        conf.downstream = NULL;
        conf.upstream = upstream;
        conf.upstreamSize = sizeof(upstream) / sizeof(char*);
        conf.fillAttrList = LwSciSyncTest_FillCpuSignalerAttrList;
        conf.stream = cpuSignalStream;

        status = standardSignaler(&conf);
        if (status == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        } else {
            exit(EXIT_FAILURE);
        }
    } else if ((peers[1] = fork()) == 0) {
        const char* downstream = "lwscisync_a_1";
        conf.downstream = downstream;
        conf.upstream = NULL;
        conf.upstreamSize = 0U;
        conf.fillAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
        conf.stream = cpuWaitStream;

        status = standardWaiter(&conf);
        if (status == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        } else {
            exit(EXIT_FAILURE);
        }
    } else if ((peers[2] = fork()) == 0) {
        const char* upstream[] = {
            "lwscisync_c_0",
            "lwscisync_d_0",
        };
        const char* downstream = "lwscisync_b_1";
        conf.downstream = downstream;
        conf.upstream = upstream;
        conf.upstreamSize = 2U;
        conf.fillAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
        conf.stream = cpuWaitStream;

        status = standardWaiter(&conf);
        if (status == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        } else {
            exit(EXIT_FAILURE);
        }
    } else if ((peers[3] = fork()) == 0) {
        const char* downstream = "lwscisync_c_1";
        conf.downstream = downstream;
        conf.upstream = NULL;
        conf.upstreamSize = 0U;
        conf.fillAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
        conf.stream = cpuWaitStream;

        status = standardWaiter(&conf);
        if (status == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        } else {
            exit(EXIT_FAILURE);
        }
    } else if ((peers[4] = fork()) == 0) {
        const char* downstream = "lwscisync_d_1";
        conf.downstream = downstream;
        conf.upstream = NULL;
        conf.upstreamSize = 0U;
        conf.fillAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
        conf.stream = cpuWaitStream;

        status = standardWaiter(&conf);
        if (status == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        } else {
            exit(EXIT_FAILURE);
        }
    } else {
        int procExit = 0;
        int i = 0;

        for (i = 0; i < 5; ++i) {
            waitpid(peers[i], &procExit, 0);
            if (!WIFEXITED(procExit)) {
                printf("a peer did not exit\n");
                status = LwSciSyncTestStatus::Failure;
            }
            status = WEXITSTATUS(procExit) == EXIT_SUCCESS ?
                    status : LwSciSyncTestStatus::Failure;
        }
    }
    return status;
}

LwSciSyncTestStatus
export_import_with_unsupported_perm(TestInfo* info)
{
    LwSciError err;
    /* below endpoint simulate topology:
     * signaler <-> waiter
     */
    LwSciIpcEndpoint sigEndpoint;
    LwSciIpcEndpoint waitEndpoint;
    LwSciSyncModule module = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    LwSciSyncAttrList waiterAttrList = NULL;
    LwSciSyncAttrList signalerAttrList = NULL;
    LwSciSyncAttrList tempAttrList = NULL;
    size_t waiterAttrListSize = 0U;
    void* waiterListDesc;
    LwSciSyncAttrList importedUnreconciledList = NULL;
    LwSciSyncAttrList unreconciledLists[3] = {NULL};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncObj syncObj = NULL;
    void* objAndListDesc = NULL;
    size_t objAndListSize = 0U;
    LwSciSyncObj waitSyncObj = NULL;

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciIpcOpenEndpoint("lwscisync_a_0", &sigEndpoint);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciIpcOpenEndpoint("lwscisync_a_1", &waitEndpoint);
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

    err = LwSciSyncAttrListCreate(module, &signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = LwSciSyncTest_FillCpuSignalerAttrList(signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &tempAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = LwSciSyncTest_FillCpuAttrListAutoPerm(tempAttrList);
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
        module, sigEndpoint,
        waiterListDesc, waiterAttrListSize,
        &importedUnreconciledList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* reconcile */

    unreconciledLists[0] = signalerAttrList;
    unreconciledLists[1] = importedUnreconciledList;
    unreconciledLists[2] = tempAttrList;

    NegativeTestPrint();
    err = LwSciSyncAttrListReconcile(unreconciledLists, 3, &reconciledList,
            &newConflictList);
    if (err == LwSciError_Success) {
        goto fail;
    }

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

    NegativeTestPrint();
    err = LwSciSyncIpcExportAttrListAndObj(
        syncObj,
        LwSciSyncAccessPerm_SignalOnly, sigEndpoint,
        &objAndListDesc, &objAndListSize);
    if (err == LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncIpcExportAttrListAndObj(
        syncObj,
        LwSciSyncAccessPerm_WaitOnly, sigEndpoint,
        &objAndListDesc, &objAndListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    {
        NEGATIVE_TEST();
        err = LwSciSyncIpcImportAttrListAndObj(
            module, waitEndpoint, objAndListDesc, objAndListSize,
            &waiterAttrList, 1, LwSciSyncAccessPerm_SignalOnly, 10000,
            &waitSyncObj);
        if (err == LwSciError_Success) {
            goto fail;
        }
    }

    err = LwSciSyncIpcImportAttrListAndObj(
        module, waitEndpoint,
        objAndListDesc, objAndListSize,
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

        err = LwSciSyncIpcImportFence(waitSyncObj,
                                      &fenceDesc,
                                      &syncFence);
        if (err != LwSciError_Success) {
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
    LwSciSyncAttrListAndObjFreeDesc(objAndListDesc);

    LwSciSyncObjFree(syncObj);
    LwSciSyncAttrListFree(reconciledList);

    LwSciSyncAttrListFree(importedUnreconciledList);
    LwSciSyncAttrListFreeDesc(waiterListDesc);

    LwSciSyncAttrListFree(waiterAttrList);
    LwSciSyncAttrListFree(signalerAttrList);
    LwSciSyncAttrListFree(tempAttrList);

    LwSciSyncCpuWaitContextFree(waitContext);
    LwSciSyncModuleClose(module);

    LwSciIpcCloseEndpoint(waitEndpoint);
    LwSciIpcCloseEndpoint(sigEndpoint);
    LwSciIpcDeinit();

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return LwSciSyncTestStatus::Failure;
    }

    return LwSciSyncTestStatus::Success;
}

/** @jama{10392630} Automatic permission test
 * -Object export/import is using LwSciSyncAccessPerm_Auto
 * -Object export/import is using unsupported perms
 * -Setting LwSciSyncAccessPerm_Auto in attrList
 */
LWSCISYNC_DECLARE_TEST(TestAutomaticPermission, AutomaticPermissions, 10392630)
{
    LwSciSyncTestStatus status = LwSciSyncTestStatus::Success;
    status = auto_perms_export(info);
    if (status != LwSciSyncTestStatus::Success) {
        goto fn_exit;
    }
    status = export_import_with_unsupported_perm(info);
    if (status != LwSciSyncTestStatus::Success) {
        goto fn_exit;
    }

fn_exit:
    ASSERT_EQ(status, LwSciSyncTestStatus::Success);
}
