/*
 * Copyright (c) 2019-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <lwscisync_internal.h>
#include <lwscisync_test_signaler.h>
#include <lwscisync_test_waiter.h>
#include <stdio.h>
#include <cinttypes>
#include "lwscisync_interprocess_test.h"

static bool isFenceEmpty(LwSciSyncFence* fence)
{
    size_t i;
    size_t size = sizeof(fence->payload) / sizeof(fence->payload[0]);

    for (i = 0U; i < size; ++i) {
        if (fence->payload[i] != 0U)
            return false;
    }
    return true;
}

/** @jama{10060442} Empty fences behavior
 * -empty_fences
 *   Verifies that LwSciSync APIs correctly handle empty fences
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, EmptyFences, 10060442)
{
    LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
    bool empty = false;
    LwSciSyncTestStatus status = LwSciSyncTestStatus::Success;
    LwSciError err = LwSciError_Success;
    LwSciSyncModule module = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    /* magic value that should not change */
    const uint64_t magic = 19U;
    uint64_t value = magic;
    LwSciSyncObj syncObj = NULL;

    empty = isFenceEmpty(&syncFence);
    if (!empty) {
        status = LwSciSyncTestStatus::Failure;
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

    err = LwSciSyncFenceWait(&syncFence, waitContext, 0);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* retrieving info attempts should result in LwSciError_ClearedFence */
    err = LwSciSyncFenceGetTimestamp(&syncFence, &value);
    if (err != LwSciError_ClearedFence) {
        printf("unexpected %d in LwSciSyncFenceGetTimestamp\n", err);
        status = LwSciSyncTestStatus::Failure;
        goto fail;
    }
    if (value != magic) {
        printf("LwSciSyncFenceGetTimestamp changed the out value to %" PRIu64
               "\n", value);
        status = LwSciSyncTestStatus::Failure;
        goto fail;
    }

    err = LwSciSyncFenceExtractFence(&syncFence, &value, &value);
    if (err != LwSciError_ClearedFence) {
        printf("unexpected %d in LwSciSyncFenceExtractFence\n", err);
        status = LwSciSyncTestStatus::Failure;
        goto fail;
    }
    if (value != magic) {
        printf("LwSciSyncFenceExtractFence changed the out value to %" PRIu64
                "\n", value);
        status = LwSciSyncTestStatus::Failure;
        goto fail;
    }

    err = LwSciSyncFenceGetSyncObj(&syncFence, &syncObj);
    if (err != LwSciError_ClearedFence) {
        printf("unexpected %d in LwSciSyncFenceGetSyncObj\n", err);
        status = LwSciSyncTestStatus::Failure;
        goto fail;
    }
    if (syncObj != NULL) {
        printf("LwSciSyncFenceGetSyncObj changed the out value to %p\n",
                syncObj);
        status = LwSciSyncTestStatus::Failure;
        goto fail;
    }

    /* set the err to Success here to avoid test failure */
    err = LwSciError_Success;

fail:
    LwSciSyncCpuWaitContextFree(waitContext);
    LwSciSyncModuleClose(module);

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        status = LwSciSyncTestStatus::Failure;
    }

    ASSERT_EQ(status, LwSciSyncTestStatus::Success);
}

/** @jama{10060484} Clearing fences by various means
 * -clearing_fences
 *   Tests that fences are correctly cleared in various situations
 */
class ClearingFences : public LwSciSyncInterProcessTest
{
};

TEST_F(ClearingFences, BasicSupport)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = LwSciSyncTest_FillCpuSignalerAttrList(signalerAttrList.get());
        ASSERT_EQ(error, LwSciError_Success);

        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto attrListAndObjDesc = peer->exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(attrListAndObjDesc), LwSciError_Success);

        auto emptyFence = LwSciSyncPeer::generateFence(syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_FALSE(isFenceEmpty(emptyFence.get())) << "fence cleared after generating";

        LwSciSyncFenceClear(emptyFence.get());
        ASSERT_TRUE(isFenceEmpty(emptyFence.get())) << "fence not cleared after clearing";

        /* repeat to test double fence clearing */
        LwSciSyncFenceClear(emptyFence.get());
        ASSERT_TRUE(isFenceEmpty(emptyFence.get())) << "fence not cleared after double clearing";

        /* generate fence to have a non empty start */
        auto syncFence = LwSciSyncPeer::generateFence(syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_FALSE(isFenceEmpty(syncFence.get())) << "fence cleared after generating";

        /* transport non-empty fence */
        auto fenceDesc = peer->exportFence(syncFence.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(fenceDesc), LwSciError_Success);

        /* transport empty fence */
        auto emptyFenceDesc = peer->exportFence(emptyFence.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(emptyFenceDesc), LwSciError_Success);

        /* generate fence to have a non empty start */
        auto dupSyncFence = LwSciSyncPeer::generateFence(syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_FALSE(isFenceEmpty(dupSyncFence.get())) << "fence cleared after generating";

        /* duplicate empty fence */
        error = LwSciSyncFenceDup(emptyFence.get(), dupSyncFence.get());
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_TRUE(isFenceEmpty(dupSyncFence.get())) << "fence not cleared after duplicating empty fence";

        /* Wait before freeing LwSciBufObj for primitive buffers. This is
         * needed for Desktop. */
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        error = LwSciSyncTest_FillCpuWaiterAttrList(waiterAttrList.get());
        ASSERT_EQ(error, LwSciError_Success);

        auto listDescBuf =
            peer->exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer->importAttrListAndObj(
            attrListAndObjDesc, {waiterAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* import non-empty fence to have a non empty start */
        auto fenceDesc =
            peer->recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(
                &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncFence = peer->importFence(fenceDesc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_FALSE(isFenceEmpty(syncFence.get())) << "fence cleared after importing";

        /* import empty fence */
        auto emptyFenceDesc =
            peer->recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(
                &error);
        ASSERT_EQ(error, LwSciError_Success);

        syncFence = peer->importFence(fenceDesc.get(), syncObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_FALSE(isFenceEmpty(syncFence.get())) << "fence not cleared after importing";

        /* Signal that the LwSciBufObj used for the primitive buffer can be
         * freed since on Desktop the memory handle is duplicated on import not
         * export. Otherwise on Desktop, there can be a race where the
         * allocator may free before this process performs the import,
         * resulting in an invalid handle being used when duplicating on
         * import. */
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
