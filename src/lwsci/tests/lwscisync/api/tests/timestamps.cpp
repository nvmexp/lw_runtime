/*
 * Copyright (c) 2019-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <fcntl.h>
#include <lwscibuf.h>
#include <lwscicommon_arch.h>
#include <lwscisync.h>
#include <lwscisync_ipc_peer_old.h>
#include <lwscisync_test_common.h>
#include <lwscisync_test_signaler.h>
#include <lwscisync_test_waiter.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <umd.h>
#include <unistd.h>

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_TIMESTAMP_TEST(testSuite, JamaID)                     \
    class testSuite : public LwSciSyncTransportTest<JamaID>                    \
    {                                                                          \
    };

/* Declare additional test case for a test */
#define LWSCISYNC_TIMESTAMP_TEST_CASE(testSuite, testName)              \
    TEST_F(testSuite, testName)

#define TEST_TIMESTAMP_SUPPORT_NUM_FENCES 16U

/** @jama{9873872} Single thread basic timestamp usage
 * -basic_timestamps
 *   single thread test case with timestamps
 */
LWSCISYNC_DECLARE_TEST(TestTimestampSupport, BasicTimestamps, 9873872)
{
    LwSciError err;
    LwSciSyncModule module = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    LwSciSyncAttrList unreconciledAttrLists[2] = {0};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncObj syncObj = NULL;
    bool verbose = info->verbose;
    uint64_t i = 0U;

    err = LwSciSyncModuleOpen(&module);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncCpuWaitContextAlloc(module, &waitContext);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledAttrLists[0]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncTest_FillTimestampsSignalerAttrList(unreconciledAttrLists[0]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledAttrLists[1]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncTest_FillTimestampsWaiterAttrList(unreconciledAttrLists[1]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListReconcile(unreconciledAttrLists, 2, &reconciledList,
            &newConflictList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncObjAlloc(reconciledList, &syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* streaming */
    for (i = 0U; i < TEST_TIMESTAMP_SUPPORT_NUM_FENCES; ++i) {
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        uint64_t timestamp = 0U;
        uint64_t timestampBeforeSignal = 0U;
        uint64_t timestampAfterSignal = 0U;

        /* generate a fence */
        err = LwSciSyncObjGenerateFence(syncObj, &syncFence);
        if (err != LwSciError_Success) {
            goto fail;
        }

        timestampBeforeSignal = LwSciCommonGetTimeUS();
        err = LwSciSyncObjSignal(syncObj);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }
        timestampAfterSignal = LwSciCommonGetTimeUS();

        err = LwSciSyncFenceWait(&syncFence,
                waitContext, -1);
        if (err != LwSciError_Success) {
            printf("err is x%x\n", err);
            goto streaming_fail;
        }

        err = LwSciSyncFenceGetTimestamp(&syncFence, &timestamp);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }
        EXPECT_GE(timestamp, timestampBeforeSignal);
        EXPECT_LE(timestamp, timestampAfterSignal);

        if (verbose) {
            printf("timestamp on the fence %" PRIu64 " is %" PRIu64 "\n",
                    i, timestamp);
        }

    streaming_fail:

        LwSciSyncFenceClear(&syncFence);
    }

fail:
    LwSciSyncObjFree(syncObj);
    LwSciSyncAttrListFree(reconciledList);
    LwSciSyncAttrListFree(unreconciledAttrLists[0]);
    LwSciSyncAttrListFree(unreconciledAttrLists[1]);
    LwSciSyncCpuWaitContextFree(waitContext);
    LwSciSyncModuleClose(module);

    ASSERT_EQ(err, LwSciError_Success);
}

/** @jama{13243093} Single thread basic timestamp usage
 * BasicTimestampsManyFencesActive verifies that when multiple (but not too
 * many) signals has passed, the timestamps are still retrievable properly.
 */
LWSCISYNC_DECLARE_TEST(TestTimestampSupport, BasicTimestampsManyFencesActive, 13243093)
{
    LwSciError err;
    LwSciSyncModule module = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    LwSciSyncAttrList unreconciledAttrLists[2] = {0};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncObj syncObj = NULL;
    bool verbose = info->verbose;
    uint64_t i = 0U;
    LwSciSyncFence syncFences[TEST_TIMESTAMP_SUPPORT_NUM_FENCES] = {
        LwSciSyncFenceInitializer
    };
    uint64_t timestampBeforeSignal[TEST_TIMESTAMP_SUPPORT_NUM_FENCES] = { 0 };
    uint64_t timestampAfterSignal[TEST_TIMESTAMP_SUPPORT_NUM_FENCES] = { 0 };

    err = LwSciSyncModuleOpen(&module);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncCpuWaitContextAlloc(module, &waitContext);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledAttrLists[0]);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncTest_FillTimestampsSignalerAttrList(unreconciledAttrLists[0]);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledAttrLists[1]);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncTest_FillTimestampsWaiterAttrList(unreconciledAttrLists[1]);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListReconcile(unreconciledAttrLists, 2, &reconciledList,
            &newConflictList);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncObjAlloc(reconciledList, &syncObj);
    EXPECT_EQ(err, LwSciError_Success);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Generate a bunch of fences */
    for (i = 0; i < TEST_TIMESTAMP_SUPPORT_NUM_FENCES; ++i) {
        err = LwSciSyncObjGenerateFence(syncObj, &syncFences[i]);
        EXPECT_EQ(err, LwSciError_Success);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    /* streaming */
    for (i = 0U; i < TEST_TIMESTAMP_SUPPORT_NUM_FENCES; ++i) {
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;

        timestampBeforeSignal[i] = LwSciCommonGetTimeUS();
        err = LwSciSyncObjSignal(syncObj);
        EXPECT_EQ(err, LwSciError_Success);
        timestampAfterSignal[i] = LwSciCommonGetTimeUS();

        err = LwSciSyncFenceWait(&syncFences[i], waitContext, -1);
        EXPECT_EQ(err, LwSciError_Success);
    }

    /* Now read back all the timestamps */
    for (i = 0U; i < TEST_TIMESTAMP_SUPPORT_NUM_FENCES; ++i) {
        uint64_t timestamp = 0U;
        err = LwSciSyncFenceGetTimestamp(&syncFences[i], &timestamp);
        EXPECT_EQ(err, LwSciError_Success);
        EXPECT_GE(timestamp, timestampBeforeSignal[i]);
        EXPECT_LE(timestamp, timestampAfterSignal[i]);

        if (verbose) {
            printf("timestamp on the fence %" PRIu64 " is %" PRIu64 "\n",
                    i, timestamp);
        }
    streaming_fail:
        LwSciSyncFenceClear(&syncFences[i]);
    }

fail:
    LwSciSyncObjFree(syncObj);
    LwSciSyncAttrListFree(reconciledList);
    LwSciSyncAttrListFree(unreconciledAttrLists[0]);
    LwSciSyncAttrListFree(unreconciledAttrLists[1]);
    LwSciSyncCpuWaitContextFree(waitContext);
    LwSciSyncModuleClose(module);

    ASSERT_EQ(err, LwSciError_Success);
}

/** @jama{9937973} Single thread timestamp usage by mocked umd
 * -mock_umd_timestamps
 *   single thread test case with timestamps
 *   where the application simulates a umd
 */
LWSCISYNC_DECLARE_TEST(TestTimestampSupport, MockUmdTimestamps, 9937973)
{
    LwSciError err;
    TestResources resource = NULL;
    LwSciSyncModule module = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    LwSciSyncAttrList unreconciledAttrLists[2] = {0};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncObj syncObj = NULL;
    bool verbose = info->verbose;
    uint64_t i = 0U;
    LwSciSyncTimestampBufferInfo timestampBufferInfo = {0};
    uint64_t* timestampBase = NULL;

    err = LwSciSyncModuleOpen(&module);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncCpuWaitContextAlloc(module, &waitContext);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledAttrLists[0]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncTest_FillTimestampsSignalerAttrList(unreconciledAttrLists[0]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &unreconciledAttrLists[1]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncTest_FillTimestampsWaiterAttrList(unreconciledAttrLists[1]);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListReconcile(unreconciledAttrLists, 2, &reconciledList,
            &newConflictList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncObjAlloc(reconciledList, &syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncObjGetTimestampBufferInfo(syncObj, &timestampBufferInfo);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciBufObjGetCpuPtr(timestampBufferInfo.bufObj,
            (void**) &timestampBase);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwRmGpu_TestSetup(&resource);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwRmGpu_TestMapSemaphore(resource, syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* streaming */
    for (i = 0U; i < TEST_TIMESTAMP_SUPPORT_NUM_FENCES; ++i) {
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        uint64_t timestamp = 0U;
        uint32_t slotIndex = 0U;
        uint64_t* timestampAddress = NULL;
        uint64_t magicValue = 0xf00ba4;

        err = LwSciSyncObjGetNextTimestampSlot(syncObj, &slotIndex);
        if (err != LwSciError_Success) {
            goto fail;
        }

        /* 1 slot is 8 byte */
        timestampAddress = timestampBase + slotIndex;

        *timestampAddress = magicValue;
        err = MockUmdSignalStreamFrame(syncObj, resource, &syncFence,
                slotIndex);
        if (err != LwSciError_Success) {
            goto fail;
        }

        err = LwSciSyncFenceWait(&syncFence,
                waitContext, -1);
        if (err != LwSciError_Success) {
            printf("err is x%x\n", err);
            goto streaming_fail;
        }

        err = LwSciSyncFenceGetTimestamp(&syncFence, &timestamp);
        if (err != LwSciError_Success) {
            goto streaming_fail;
        }

        EXPECT_EQ(timestamp, magicValue);

        if (verbose) {
            printf("timestamp on the fence %" PRIu64 " is %" PRIu64 "\n",
                    i, timestamp);
        }

    streaming_fail:

        LwSciSyncFenceClear(&syncFence);
    }

fail:
    LwRmGpu_TestTeardown(resource);

    LwSciSyncObjFree(syncObj);
    LwSciSyncAttrListFree(reconciledList);
    LwSciSyncAttrListFree(unreconciledAttrLists[0]);
    LwSciSyncAttrListFree(unreconciledAttrLists[1]);
    LwSciSyncCpuWaitContextFree(waitContext);
    LwSciSyncModuleClose(module);

    ASSERT_EQ(err, LwSciError_Success);
}

static void LwSciSyncFenceCleanup(
    LwSciSyncFence* syncFence) {
    LwSciSyncFenceClear(syncFence);
    delete syncFence;
}

/**
 * @jama{0} -
 * @brief Test verifies that export-import is successful when CPU Waiter
 * attrList is relayed through peer process, without additional attrList
 * from the peer process.
 *
 * 1. Process A creates CPU signaler attr list (L1), CPU waiter attr list (L2),
 *    CPU waiter attr list with timestamps (L3).
 * 2. Process A exports L2 to Process B.
 * 3. Process B imports and re-exports L2 to Process A.
 * 4. Process A reconciles, allocates syncObj, generates fence and successfully
 *    exports them to process B.
 * 5. Process B successfully imports the reconciled attrList, syncObj and fence.
 */
LWSCISYNC_TIMESTAMP_TEST(LwSciSyncTimestampTest, 0)
LWSCISYNC_TIMESTAMP_TEST_CASE(LwSciSyncTimestampTest, Transport1)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        // Process A
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsSignalerAttrList(signalerAttrList.get()));

        // Create attribute lists with CPU waiter timestamp attributes
        auto waiterTimestampAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsWaiterAttrList(waiterTimestampAttrList.get()));

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Export waiterAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Import waiterAttrList
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList = peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), importedAttrList.get(),
             waiterTimestampAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList.get(), nullptr);

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        ASSERT_NE(syncObj, nullptr);
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        // Export reconiled list and sync obj
        auto attrListAndObjDesc = peer.exportAttrListAndObj(
            syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(attrListAndObjDesc), LwSciError_Success);
        // Generate, signal and export fence
        auto syncFence = peer.generateFence(syncObj, &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);

        auto fenceDesc = peer.exportFence(syncFence.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(peer.sendExportDesc(fenceDesc), LwSciError_Success);
        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(syncFence.get(),
                &timestampUS), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        // Process B
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Import waiter attr lists
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Re-export the importedAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Import reconiled list and sync obj
        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedSyncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), importedSyncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} -
 * @brief Test verifies that export-import is successful when CPU Waiter
 * timestamp attrList is relayed through peer process, without additional
 * attrList from the peer process.
 *
 * 1. Process A creates CPU signaler attr list (L1), CPU waiter attr list (L2),
 *    CPU waiter attr list with timestamps (L3).
 * 2. Process A exports L3 to Process B.
 * 3. Process B imports and re-exports L3 to Process A.
 * 4. Process A reconciles, allocates syncObj, generates fence and successfully
 *    exports them to process B.
 * 5. Process B successfully imports the reconciled attrList, syncObj and fence.
 */
LWSCISYNC_TIMESTAMP_TEST_CASE(LwSciSyncTimestampTest, Transport2)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        // Process A
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsSignalerAttrList(signalerAttrList.get()));

        // Create attribute lists with CPU waiter timestamp attributes
        auto waiterTimestampAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsWaiterAttrList(waiterTimestampAttrList.get()));

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Export waiterTimestampAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({waiterTimestampAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Import waiterTimestampAttrList
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList = peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), importedAttrList.get(),
             waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList.get(), nullptr);

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        ASSERT_NE(syncObj, nullptr);
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        auto attrListAndObjDesc = peer.exportAttrListAndObj(
            syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(attrListAndObjDesc), LwSciError_Success);

        // Generate, signal and export fence
        auto syncFence = peer.generateFence(syncObj, &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);

        auto fenceDesc = peer.exportFence(syncFence.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(peer.sendExportDesc(fenceDesc), LwSciError_Success);
        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(syncFence.get(),
                &timestampUS), LwSciError_Success);

        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        // Process B
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Import waiter timestamp attr list
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Re-export the importedAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto importedSyncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), importedSyncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} -
 * @brief Test verifies that peer process if requested for timestamp get access,
 * and otherPeer connected through the peer process does not get access to
 * timestamp if not requested.
 *
 * 1. Process A creates CPU signaler attr list (L1).
 * 2. Process B creates CPU waiter attr list with timestamps (L2) and exports
 *    to Process A.
 * 3. Process C creates CPU waiter attr list (L3) and exports to Process B.
 * 4. Process B imports L3 and re-exports to Process A.
 * 5. Process A imports L2 and L3, reconciles, allocates syncObj and exports
 *    attrList and syncObj to Process B.
 * 6. Process B imports attrList and syncObj and exports to Process C.
 * 7. Process A generates fence and signals syncObj and exports fence to
 *    process B and successfully calls LwSciSyncFenceGetTimestamp().
 * 8. Process B imports fence and exports it to Process C. Process B
 *    successfully calls LwSciSyncFenceGetTimestamp().
 * 9. Process C imports fence and call to LwSciSyncFenceGetTimestamp() fails.
 */
LWSCISYNC_TIMESTAMP_TEST_CASE(LwSciSyncTimestampTest, Transport3)
{
    LwSciError error = LwSciError_Success;
    pid_t pids[2] = { 0 };

    if ((pids[0] = fork()) == 0) {
        // Process A
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsSignalerAttrList(signalerAttrList.get()));

        // Import waiterTimestampAttrList
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterTimestampAttrList = peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(waiterTimestampAttrList.get(), nullptr);

        // Import waiterAttrList
        auto importedBuf1 = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList = peer.importUnreconciledList(importedBuf1, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(waiterAttrList.get(), nullptr);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterTimestampAttrList.get(),
            waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList.get(), nullptr);

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        ASSERT_NE(syncObj, nullptr);
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        auto attrListAndObjDesc = peer.exportAttrListAndObj(
            syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(attrListAndObjDesc), LwSciError_Success);

        // Generate, signal and export fence
        auto syncFence = peer.generateFence(syncObj, &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);
        auto fenceDesc = peer.exportFence(syncFence.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(peer.sendExportDesc(fenceDesc), LwSciError_Success);
        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(syncFence.get(),
                &timestampUS), LwSciError_Success);


        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        // Process B
        initIpc();
        peer.SetUp("lwscisync_a_1");
        otherPeer.SetUp("lwscisync_b_0", peer);

        // Create attribute lists with CPU waiter timestamp attributes
        auto waiterTimestampAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsWaiterAttrList(waiterTimestampAttrList.get()));

        // Export waiterTimestampAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({waiterTimestampAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);
        // Import waiter attr list
        auto importedBuf = otherPeer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            otherPeer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Export the importedAttrList to signaler
        auto listDescBuf1 =
            peer.exportUnreconciledList({importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf1), LwSciError_Success);
        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {waiterTimestampAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto attrListAndObjDesc1 = otherPeer.exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(otherPeer.sendBuf(attrListAndObjDesc1), LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export fence
        auto fenceDesc = otherPeer.exportFence(importedFencePtr.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(otherPeer.sendExportDesc(fenceDesc), LwSciError_Success);

        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(importedFencePtr.get(),
                &timestampUS), LwSciError_Success);

        // wait for other peer to exit
        ASSERT_EQ(otherPeer.waitComplete(), LwSciError_Success);
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
    } else {
        // Process C
        pid = 1;
        initIpc();
        peer.SetUp("lwscisync_b_1");
        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Export waiterAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {waiterAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // This process does not support timestamps
            uint64_t timestampUS;

            NEGATIVE_TEST();
            ASSERT_EQ(LwSciSyncFenceGetTimestamp(importedFencePtr.get(),
                                                 &timestampUS),
                      LwSciError_BadParameter);
        }

        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        // wait for other peer to exit
        int status = EXIT_SUCCESS;
        status |= wait_for_child_fork(pids[0]);
        status |= wait_for_child_fork(pids[1]);
        ASSERT_EQ(EXIT_SUCCESS, status);
    }
}

/**
 * @jama{0} -
 * @brief Test verifies that peer process if did not request for timestamp does
 * not gets access, even if otherPeer connected through the peer process did
 * request for timestamps.
 *
 * 1. Process A creates CPU signaler attr list (L1).
 * 2. Process B creates CPU waiter attr list (L2) and exports to Process A.
 * 3. Process C creates CPU waiter attr list with timestamps (L3) and exports
 *    to Process B.
 * 4. Process B imports L3 and re-exports to Process A.
 * 5. Process A imports L2 and L3, reconciles, allocates syncObj and exports
 *    attrList and syncObj to Process B.
 * 6. Process B imports attrList and syncObj and exports to Process C.
 * 7. Process A generates fence and signals syncObj and exports fence to
 *    process B and successfully calls LwSciSyncFenceGetTimestamp().
 * 8. Process B imports fence and exports it to Process C. Process B
 *    fails to call LwSciSyncFenceGetTimestamp().
 * 9. Process C imports fence and successfully calls LwSciSyncFenceGetTimestamp().
 */
LWSCISYNC_TIMESTAMP_TEST_CASE(LwSciSyncTimestampTest, Transport4)
{
    LwSciError error = LwSciError_Success;
    pid_t pids[2] = { 0 };

    if ((pids[0] = fork()) == 0) {
        // Process A
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsSignalerAttrList(signalerAttrList.get()));

        // Import waiterTimestampAttrList
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterTimestampAttrList = peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(waiterTimestampAttrList.get(), nullptr);

        // Import waiterAttrList
        auto importedBuf1 = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList = peer.importUnreconciledList(importedBuf1, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(waiterAttrList.get(), nullptr);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterTimestampAttrList.get(),
             waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList.get(), nullptr);

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        ASSERT_NE(syncObj, nullptr);
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        auto attrListAndObjDesc = peer.exportAttrListAndObj(
            syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(attrListAndObjDesc), LwSciError_Success);

        // Generate, signal and export fence
        auto syncFence = peer.generateFence(syncObj, &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);
        auto fenceDesc = peer.exportFence(syncFence.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(peer.sendExportDesc(fenceDesc), LwSciError_Success);
        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(syncFence.get(),
                &timestampUS), LwSciError_Success);

        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        // Process B
        initIpc();
        peer.SetUp("lwscisync_a_1");
        otherPeer.SetUp("lwscisync_b_0", peer);

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Import waiter attr list
        auto importedBuf = otherPeer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            otherPeer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Export the importedAttrList to signaler
        auto listDescBuf =
            peer.exportUnreconciledList({importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Export waiterAttrList
        auto listDescBuf1 =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf1), LwSciError_Success);

        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {waiterAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto attrListAndObjDesc1 = otherPeer.exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(otherPeer.sendBuf(attrListAndObjDesc1), LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export fence
        auto fenceDesc = otherPeer.exportFence(importedFencePtr.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(otherPeer.sendExportDesc(fenceDesc), LwSciError_Success);

        {
            // This process does not support timestamps
            uint64_t timestampUS;

            NegativeTestPrint();
            ASSERT_EQ(LwSciSyncFenceGetTimestamp(importedFencePtr.get(),
                                                 &timestampUS),
                      LwSciError_BadParameter);
        }

        // wait for other peer to exit
        ASSERT_EQ(otherPeer.waitComplete(), LwSciError_Success);
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
    } else {
        // Process C
        pid = 1;
        initIpc();
        peer.SetUp("lwscisync_b_1");

        // Create attribute lists with CPU waiter timestamp attributes
        auto waiterTimestampAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsWaiterAttrList(waiterTimestampAttrList.get()));

        // Export waiterTimestampAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({waiterTimestampAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {waiterTimestampAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(importedFencePtr.get(),
                &timestampUS), LwSciError_Success);

        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);

        // wait for other peer to exit
        int status = EXIT_SUCCESS;
        status |= wait_for_child_fork(pids[0]);
        status |= wait_for_child_fork(pids[1]);
        ASSERT_EQ(EXIT_SUCCESS, status);
    }
}

/**
 * @jama{0} -
 * @brief Test verifies that peer process if did not request for timestamp does
 * not get access, and the otherPeer connected to the same process gets access if
 * requested for timestamp.
 *
 * 1. Process A creates CPU signaler attr list (L1).
 * 2. Process B creates CPU waiter attr list (L2) and exports to Process A.
 * 3. Process C creates CPU waiter attr list with timestamps (L3) and exports
 *    to Process A.
 * 4. Process A imports L2 and L3, reconciles, allocates syncObj and exports
 *    attrList and syncObj to Process B and Process C.
 * 5. Process B imports attrList and syncObj.
 * 6. Process C imports attrList and syncObj.
 * 7. Process A generates fence and signals syncObj and exports fence to
 *    process B and Process C and successfully calls LwSciSyncFenceGetTimestamp().
 * 8. Process B imports fence and call to LwSciSyncFenceGetTimestamp() fails.
 * 9. Process C imports fence and successfully calls LwSciSyncFenceGetTimestamp().
 */
LWSCISYNC_TIMESTAMP_TEST_CASE(LwSciSyncTimestampTest, Transport5)
{
    LwSciError error = LwSciError_Success;
    pid_t pids[2] = { 0 };

    if ((pids[0] = fork()) == 0) {
        // Process A
        initIpc();
        peer.SetUp("lwscisync_a_0");
        otherPeer.SetUp("lwscisync_b_0", peer);

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsSignalerAttrList(signalerAttrList.get()));

        // Import waiterTimestampAttrList
        auto importedBuf = otherPeer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterTimestampAttrList = otherPeer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(waiterTimestampAttrList.get(), nullptr);

        // Import waiterAttrList
        importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList = peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(waiterAttrList.get(), nullptr);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterTimestampAttrList.get(),
             waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList.get(), nullptr);

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        ASSERT_NE(syncObj, nullptr);
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        auto attrListAndObjDesc = peer.exportAttrListAndObj(
            syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(attrListAndObjDesc), LwSciError_Success);

        auto attrListAndObjDesc1 = otherPeer.exportAttrListAndObj(
            syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(otherPeer.sendBuf(attrListAndObjDesc1), LwSciError_Success);

        // Generate, signal and export fence
        auto syncFence = peer.generateFence(syncObj, &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);
        auto fenceDesc1 = peer.exportFence(syncFence.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(peer.sendExportDesc(fenceDesc1), LwSciError_Success);
        auto fenceDesc = otherPeer.exportFence(syncFence.get(), &error);
        ASSERT_EQ(LwSciError_Success, error);
        ASSERT_EQ(otherPeer.sendExportDesc(fenceDesc), LwSciError_Success);
        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(syncFence.get(),
                &timestampUS), LwSciError_Success);

        ASSERT_EQ(otherPeer.waitComplete(), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        // Process B
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Export waiterAttrList
        auto listDescBuf1 =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf1), LwSciError_Success);

        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {waiterAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // This process does not support timestamps
            uint64_t timestampUS;

            NegativeTestPrint();
            ASSERT_EQ(LwSciSyncFenceGetTimestamp(importedFencePtr.get(),
                                                 &timestampUS),
                      LwSciError_BadParameter);
        }

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
    } else {
        // Process C
        pid = 1;
        initIpc();
        peer.SetUp("lwscisync_b_1");

        // Create attribute lists with CPU waiter timestamp attributes
        auto waiterTimestampAttrList = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
            LwSciSyncTest_FillTimestampsWaiterAttrList(waiterTimestampAttrList.get()));

        // Export waiterTimestampAttrList
        auto listDescBuf =
            peer.exportUnreconciledList({waiterTimestampAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.importAttrListAndObj(
            attrListAndObjDesc, {waiterTimestampAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import fence
        auto desc =
            peer.recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedFencePtr =
            peer.importFence(desc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // This process does support timestamps
        uint64_t timestampUS;
        ASSERT_EQ(LwSciSyncFenceGetTimestamp(importedFencePtr.get(),
                &timestampUS), LwSciError_Success);

        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);

        // wait for other peer to exit
        int status = EXIT_SUCCESS;
        status |= wait_for_child_fork(pids[0]);
        status |= wait_for_child_fork(pids[1]);
        ASSERT_EQ(EXIT_SUCCESS, status);
    }
}
