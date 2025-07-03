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

/** @jama{10142213} Small process tree performing LwSciSync cpu communication
 * -small_process_tree
 *   inter process use case with a small tree topology:
 *   S - W0
 *   |
 *   W1 - W2
 *   |
 *   W3
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, SmallProcessTree, 10142213)
{
    LwSciSyncTestStatus status = LwSciSyncTestStatus::Success;
    struct ThreadConf conf = {0};
    pid_t peers[5] = {0};

    conf.info = info;
    conf.objExportPerm = LwSciSyncAccessPerm_WaitOnly;
    conf.objImportPerm = LwSciSyncAccessPerm_WaitOnly;

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
    ASSERT_EQ(status, LwSciSyncTestStatus::Success);
}
