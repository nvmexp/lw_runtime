/*
 * Copyright (c) 2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#include "lwsci_igpu_only_test.h"
#include "lwscisync_interprocess_test.h"

class StmTest : public LwSciSyncInterProcessTest, public LwSciiGpuOnlyTest
{
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciSyncInterProcessTest::SetUp();
        LwSciiGpuOnlyTest::SetUp();
    }

    void TearDown() override
    {
        LwSciiGpuOnlyTest::TearDown();
        LwSciSyncInterProcessTest::TearDown();
    }
};

TEST_F(StmTest, Normal)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        {
            SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                     LwSciSyncAccessPerm_SignalOnly);
            SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess,
                     true);

            LwSciSyncInternalAttrValPrimitiveType primitives[] = {
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};
            SET_INTERNAL_ATTR(signalerAttrList.get(),
                              LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                              primitives);
            SET_INTERNAL_ATTR(signalerAttrList.get(),
                              LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                              (uint32_t)1);
        }

        // Import Unreconciled Waiter Attribute List
        auto importedListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer->importUnreconciledList(importedListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile
        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
            {signalerAttrList.get(), importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export
        auto attrListAndObjDesc = peer->exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(attrListAndObjDesc), LwSciError_Success);

        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peerA = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peerA);
        peerA->SetUp("lwscisync_a_1");

        auto peerB = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peerB);
        peerB->SetUp("lwscisync_b_0", *peerA);

        // Import Unreconciled Waiter Attribute List
        auto importedListDescBuf = peerB->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peerB->importUnreconciledList(importedListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // This emulates the STM use-case, where there's an empty hop
        auto listDescBuf =
            peerA->exportUnreconciledList({importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerA->sendBuf(listDescBuf), LwSciError_Success);

        //
        auto importedAttrListAndObjDesc = peerA->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto syncObj = peerA->importAttrListAndObj(
            importedAttrListAndObjDesc, {importedAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Re-export
        auto attrListAndObjDesc = peerB->exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerB->sendBuf(attrListAndObjDesc), LwSciError_Success);

        ASSERT_EQ(peerB->waitComplete(), LwSciError_Success);
        ASSERT_EQ(peerA->signalComplete(), LwSciError_Success);
    } else if ((pids[2] = fork()) == 0) {
        pid = 3;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_b_1");

        auto attrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            SET_ATTR(attrList.get(), LwSciSyncAttrKey_RequiredPerm,
                     LwSciSyncAccessPerm_WaitOnly);
            SET_INTERNAL_ATTR(attrList.get(), LwSciSyncInternalAttrKey_GpuId,
                              testiGpuId);

            LwSciSyncInternalAttrValPrimitiveType primitives[] = {
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};
            SET_INTERNAL_ATTR(attrList.get(),
                              LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                              primitives);
        }

        auto listDescBuf =
            peer->exportUnreconciledList({attrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto syncObj =
            peer->importAttrListAndObj(attrListAndObjDesc, {attrList.get()},
                                       LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

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
