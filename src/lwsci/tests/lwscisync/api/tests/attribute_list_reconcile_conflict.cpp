/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include "lwscisync_test_attribute_list.h"
#include "lwscisync_interprocess_test.h"

/**
 * @jama{14686193} Reconciliation conflict - Permissions conflict
 *
 * @brief This test checks LwSciSyncAttrListReconcile() fails if RequiredPerm
 * attribute of input attribute lists has illegal configuration:
 *  1. none of the attribute value being set to signaler’s permission,
 *  2. more than one case of the attribute value being set to signaler’s
 * permission,
 *  3. none of the attribute value being set to waiter’s permission.
 *
 * @verifies @jama{13013246} Permissions conflict
 * @verifies @jama{12978442} Reconciliation conflict
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListReconciliationPermissionsConflict,
                              14686193);

LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(
    AttributeListReconciliationPermissionsConflict, Base)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();
    auto listC = peer.createAttrList();

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listB.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listC.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    {
        // Should fail - more than one case with Signaler permissions
        NegativeTestPrint();
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {listA.get(), listB.get(), listC.get()}, &error);

        ASSERT_EQ(newReconciledList.get(), nullptr);
        ASSERT_EQ(error, LwSciError_UnsupportedConfig);
    }

    {
        // Should fail - no case with Waiter permissions, two with Signaler
        NegativeTestPrint();
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);

        ASSERT_EQ(newReconciledList.get(), nullptr);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }

    {
        // Should fail - no case with Waiter permissions
        NegativeTestPrint();
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists({listA.get()}, &error);

        ASSERT_EQ(newReconciledList.get(), nullptr);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }

    {
        // Should fail - no case with Signaler permissions
        NegativeTestPrint();
        auto newReconciledList =
            LwSciSyncPeer::reconcileLists({listC.get()}, &error);

        ASSERT_EQ(newReconciledList.get(), nullptr);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

/**
 * @jama{14686027} Reconciliation conflict - PrimitiveInfo conflict
 *
 * @brief This test checks that LwSciSyncAttrListReconcile() fails if any of the
 * following conditions are met:
 * The intersection of all of the following is empty:
 *    - Signaler's Primitives,
 *    - Waiter's Primitives of all the waiters' input attribute lists,
 *
 * @verifies @jama{13013592} PrimitiveInfo conflict
 * @verifies @jama{12978442} Reconciliation conflict
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(AttributeListReconciliationPrimitiveInfoConflict,
                              14686027)

/**
 * @jama{14686027} Reconciliation conflict - PrimitiveInfoConfilct conflict
 * Test Case 1: Intersection of signaler and waiter primitive types is empty
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(
    AttributeListReconciliationPrimitiveInfoConflict,
    PrimitiveIntersectionEmpty)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList(); // Signaler, SysmemSemaphore
    auto listB = peer.createAttrList(); // Waiter,   SysmemSemaphore
    auto listC = peer.createAttrList(); // Waiter,   Syncpoint

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  listA.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                  listA.get(),
                  LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                  LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listB.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                  listB.get(), LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                  LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listC.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                  listC.get(), LwSciSyncPeer::attrs.waiterSyncpointAttrs.data(),
                  LwSciSyncPeer::attrs.waiterSyncpointAttrs.size()),
              LwSciError_Success);

    NegativeTestPrint();
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {listA.get(), listB.get(), listC.get()}, &error);
    ASSERT_EQ(newReconciledList.get(), nullptr);
    ASSERT_EQ(error, LwSciError_UnsupportedConfig);
}

/**
 * @jama{13561103} Reconciliation failure when SignalerPrimitiveCount is
 * invalid
 *
 * @brief This test checks that we have a reconciliation failure when an
 * attribute list is provided with an invalid SignalerPrimitiveCount key. The
 * key is considered invalid when it is equal to 0.
 *
 * @verifies @jama{13563697} SignalerPrimitiveCount conflict
 * @verifies @jama{12978442} Reconciliation conflict
 */
LWSCISYNC_ATTRIBUTE_LIST_TEST(
    AttributeListReconciliationSignalerPrimitiveCountIlwalid, 13561103);

LWSCISYNC_ATTRIBUTE_LIST_TEST_CASE(
    AttributeListReconciliationSignalerPrimitiveCountIlwalid, Base)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList();
    auto listB = peer.createAttrList();

    auto signalerAttrs = LwSciSyncPeer::attrs.cpuSignaler;
    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listA.get(), signalerAttrs.data(),
                                        signalerAttrs.size()),
              LwSciError_Success);

    auto signalerInternalAttrs = LwSciSyncPeer::attrs.ilwalidPrimitiveCount;
    ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(listA.get(),
                                                signalerInternalAttrs.data(),
                                                signalerInternalAttrs.size()),
              LwSciError_Success);

    auto waiterAttrs = LwSciSyncPeer::attrs.cpuWaiter;
    ASSERT_EQ(LwSciSyncAttrListSetAttrs(listB.get(), waiterAttrs.data(),
                                        waiterAttrs.size()),
              LwSciError_Success);

    NegativeTestPrint();
    auto newReconciledList =
        LwSciSyncPeer::reconcileLists({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_BadParameter);
    ASSERT_EQ(newReconciledList.get(), nullptr);
}

class LwSciSyncReconcileConflictTimestamps : public LwSciSyncInterProcessTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey>>
{
};

TEST_P(LwSciSyncReconcileConflictTimestamps, NoSupportedPrimitive)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey signalerTimestampInfoKey = std::get<0>(params);

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

        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_SignalOnly);

        LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
        };
        uint32_t primitiveCount = 1U;
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                          primitiveInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                          primitiveCount);
        // Set SignalerTimestampInfo/SignalerTimestampInfoMulti such that only
        // SysmemSemaphore specifies a timestamp format.
        LwSciSyncAttrValTimestampInfo timestampInfo[] = {
            {
                .format = LwSciSyncTimestampFormat_Unsupported,
                .scaling = {
                    .scalingFactorNumerator = 1U,
                    .scalingFactorDenominator = 1U,
                    .sourceOffset = 0U,
                },
            },
        };
        SET_INTERNAL_ATTR(signalerAttrList.get(), signalerTimestampInfoKey,
                          timestampInfo);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            NegativeTestPrint();
            // There was no primitive that specified a supported timestamp format.
            auto reconciledList = LwSciSyncPeer::attrListReconcile(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_UnsupportedConfig);
        }

        {
            NegativeTestPrint();
            // There was no primitive that specified a supported timestamp format.
            auto bufObj = LwSciSyncPeer::reconcileAndAllocate(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_UnsupportedConfig);
        }
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);

        LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
        };
        SET_INTERNAL_ATTR(waiterAttrList.get(),
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                          primitiveInfo);

        // Export unreconciled waiter list to Peer A
        auto listDescBuf = peer->exportUnreconciledList(
                {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
INSTANTIATE_TEST_CASE_P(
    LwSciSyncReconcileConflictTimestamps, LwSciSyncReconcileConflictTimestamps,
    ::testing::Values(
        /**
         * @jama{} Reconciliation failure when there is no supported primitive
         * that supports timestamps.
         *
         * @brief This test checks that we have a reconciliation failure when an
         * attribute list is provided with an unsatisfiable set of constraints
         * via the SignalerTimestampInfo key.
         */
        std::make_tuple(LwSciSyncInternalAttrKey_SignalerTimestampInfo),
        std::make_tuple(LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti)));

class TestLwSciSyncReconcileConflictEngineArray
    : public LwSciSyncInterProcessTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>>
{
};
TEST_P(TestLwSciSyncReconcileConflictEngineArray, Reconciliation)
{
    // Maybe pass std::functions instead? Or figure out generators + GTest
    auto params = GetParam();
    bool setEngineSignaler = std::get<0>(params);
    bool setEngineWaiter = std::get<1>(params);

    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    LwSciSyncHwEngine engine{};
    LwSciError err = LwSciError_Success;
#if !defined(__x86_64__)
    engine.engNamespace = LwSciSyncHwEngine_TegraNamespaceId;
#else
    engine.engNamespace = LwSciSyncHwEngine_ResmanNamespaceId;
#endif
    err = LwSciSyncHwEngCreateIdWithoutInstance(LwSciSyncHwEngName_PCIe,
                                                &engine.rmModuleID);
    ASSERT_EQ(err, LwSciError_Success);

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_SignalOnly);

        if (setEngineSignaler) {
            SET_INTERNAL_ATTR(signalerAttrList.get(),
                              LwSciSyncInternalAttrKey_EngineArray, engine);
        }

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile
        NegativeTestPrint();
        auto reconciledList = LwSciSyncPeer::attrListReconcile(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);

        if (setEngineWaiter) {
            SET_INTERNAL_ATTR(waiterAttrList.get(),
                              LwSciSyncInternalAttrKey_EngineArray, engine);
        }

        auto listDescBuf =
            peer->exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
INSTANTIATE_TEST_CASE_P(TestLwSciSyncReconcileConflictEngineArray,
                        TestLwSciSyncReconcileConflictEngineArray,
                        ::testing::Values(
                            // signaler: PCIe, waiter: none
                            std::make_tuple(true, false),
                            // signaler: none, waiter: PCIe
                            std::make_tuple(false, true)));

class LwSciSyncReconcileWaiter : public LwSciSyncInterProcessTest
{
};

TEST_F(LwSciSyncReconcileWaiter, WaiterReconciliation)
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

        ASSERT_EQ(
            LwSciSyncAttrListSetAttrs(signalerAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                                      LwSciSyncPeer::attrs.cpuSignaler.size()),
            LwSciError_Success);

        // Export unreconciled waiter list to Peer A
        auto listDescBuf =
            peer->exportUnreconciledList({signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()),
            LwSciError_Success);

        // Import Unreconciled Waiter Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            NegativeTestPrint();
            // Reconciliation must occur on the signaler
            auto reconciledList = LwSciSyncPeer::attrListReconcile(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_ReconciliationFailed);
        }
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
