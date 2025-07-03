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

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_TRANSPORT_TEST(testSuite, JamaID)                            \
    class testSuite : public LwSciSyncTransportTest<JamaID>                    \
    {                                                                          \
    };

/* Declare additional test case for a test */
#define LWSCISYNC_TRANSPORT_TEST_CASE(testSuite, testName)                     \
    TEST_F(testSuite, testName)

/**
 * @jama{0} - Unreconciled attribute list transport
 * @verify{@jama{13019712}} - Attribute list export
 * @verify{@jama{13019718}} - Unreconciled attribute list import
 */
LWSCISYNC_TRANSPORT_TEST(UnreconciledListTransport, 0)

/**
 * @jama{0} - Unreconciled attribute list transport
 * Test Case 1. Verify correct SignalerPrimitiveInfo and SignalerPrimitiveCount
 * for unreconciled attribute list transport with Signaler permissions and CPU
 * access.
 *
 * @verify{@jama{13529719}} - Adding CPU primitives in SignalerPrimitiveInfo on
 * unreconciled attribute list export
 */
LWSCISYNC_TRANSPORT_TEST_CASE(UnreconciledListTransport, CpuSignaler)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();

        ASSERT_EQ(
            LwSciError_Success,
            LwSciSyncAttrListSetAttrs(signalerAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                                      LwSciSyncPeer::attrs.cpuSignaler.size()));

        // Export unreconciled attr lists
        auto listDescBuf =
            peer.exportUnreconciledList({signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Import unreconciled attr lists
        auto listDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer.importUnreconciledList(listDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Verify attributes
        peer.verifyAttr(importedAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess,
                        true);
        peer.verifyAttr(importedAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                        LwSciSyncAccessPerm_SignalOnly);

        LwSciSyncPeer::verifyInternalAttr(
            importedAttrList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo, ALL_CPU_PRIMITIVES);

        LwSciSyncPeer::verifyInternalAttr(
            importedAttrList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 1);

        // Verify slot count
        ASSERT_EQ(LwSciSyncAttrListGetSlotCount(importedAttrList.get()), 1);

        // Check unreconciled
        LwSciSyncPeer::checkAttrListIsReconciled(importedAttrList.get(), false);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Unreconciled attribute list transport
 * Test Case 2. Verify correctWaiterPrimitiveInfo for
 * unreconciled attribute list transport with Waiter permissions and CPU
 * access.
 *
 * @verify{@jama{13529725}} - Adding CPU primitives in WaiterPrimitiveInfo on
 * unreconciled attribute list export
 */
LWSCISYNC_TRANSPORT_TEST_CASE(UnreconciledListTransport, CpuWaiter)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();

        ASSERT_EQ(
            LwSciError_Success,
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Export unreconciled attr lists
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Import unreconciled attr lists
        auto listDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer.importUnreconciledList(listDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Verify attributes
        LwSciSyncPeer::verifyAttr(importedAttrList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(importedAttrList.get(),
                                  LwSciSyncAttrKey_RequiredPerm,
                                  LwSciSyncAccessPerm_WaitOnly);

        LwSciSyncPeer::verifyInternalAttr(
            importedAttrList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo, ALL_CPU_PRIMITIVES);

        // Verify slot count
        ASSERT_EQ(LwSciSyncAttrListGetSlotCount(importedAttrList.get()), 1);

        // Check unreconciled
        LwSciSyncPeer::checkAttrListIsReconciled(importedAttrList.get(), false);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Unreconciled attribute list transport
 * Test Case 3. Verify we cannot export WaiterSignaler list.
 */
LWSCISYNC_TRANSPORT_TEST_CASE(UnreconciledListTransport, CpuWaiterSignaler)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.cpuWaiterSignaler.data(),
                      LwSciSyncPeer::attrs.cpuWaiterSignaler.size()));

        // Export unreconciled attr lists
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Import unreconciled attr lists
        auto listDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            NEGATIVE_TEST();
            auto importedAttrList =
                peer.importUnreconciledList(listDescBuf, &error);

            ASSERT_EQ(error, LwSciError_BadParameter);
        }

        // This check is disabled due to a bug in code - output attrList in case
        // of invalid list is not NULL
        // ASSERT_EQ(importedAttrList.get(), nullptr);
        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Unreconciled attribute list transport
 * Test Case 4. Verify imported attribute list is bound to correct
 * LwSciSyncModule
 * @verify{@jama{13019718}} - Unreconciled attribute list import
 */
LWSCISYNC_TRANSPORT_TEST_CASE(UnreconciledListTransport, IsBoundToModule)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");
        otherPeer.SetUp("lwscisync_b_0");

        // Create attribute lists with CPU waiter attributes
        auto waiter = peer.createAttrList();

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      waiter.get(), LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Create attribute lists with CPU Signaler attributes
        auto signaler = otherPeer.createAttrList();

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      signaler.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()));

        // Export unreconciled attr lists
        auto listDescBuf = peer.exportUnreconciledList({waiter.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        listDescBuf =
            otherPeer.exportUnreconciledList({signaler.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(otherPeer.sendBuf(listDescBuf), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Import unreconciled attr lists
        auto listDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedWaiter = peer.importUnreconciledList(listDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedWaiter.get(), nullptr);

        // Import signaler with another LwSciSyncModule
        otherPeer.SetUp("lwscisync_b_1");
        listDescBuf = otherPeer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedSignaler =
            otherPeer.importUnreconciledList(listDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedSignaler.get(), nullptr);

        {
            LwSciError error = LwSciError_Success;

            NegativeTestPrint();
            LwSciSyncPeer::reconcileLists(
                {importedWaiter.get(), importedSignaler.get()}, &error);

            ASSERT_EQ(error, LwSciError_BadParameter);
        }

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Unreconciled attribute list transport
 * Test Case 4. Verify correct number of slots in imported attribute list.
 * @verify{@jama{13019718}} - Unreconciled attribute list import
 */
LWSCISYNC_TRANSPORT_TEST_CASE(UnreconciledListTransport, Multislot)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");
        // Create attribute lists with CPU waiter attributes
        auto waiterA = peer.createAttrList();

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      waiterA.get(), LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Create second attribute lists with CPU waiter attributes
        auto waiterB = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      waiterB.get(), LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()));
        // Export unreconciled attr lists
        auto listDescBuf =
            peer.exportUnreconciledList({waiterA.get(), waiterB.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");
        auto listDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer.importUnreconciledList(listDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);
        // Verify slot count
        ASSERT_EQ(LwSciSyncAttrListGetSlotCount(importedAttrList.get()), 2);

        // Check unreconciled
        LwSciSyncPeer::checkAttrListIsReconciled(importedAttrList.get(), false);

        // Create attribute lists with CPU Signaler attributes
        // We need a signaler to perform the reconciliation
        auto signaler = peer.createAttrList();
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      signaler.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()));

        // Reconcile attr lists
        LwSciError error = LwSciError_Success;
        LwSciSyncAttrList reconciledList = nullptr;
        auto reconciledListPtr = LwSciSyncPeer::reconcileLists(
            {importedAttrList.get(), signaler.get()}, &error);
        reconciledList = reconciledListPtr.get();
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList, nullptr);

        LwSciSyncPeer::verifyAttr(reconciledList,
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(reconciledList, LwSciSyncAttrKey_ActualPerm,
                                  LwSciSyncAccessPerm_WaitSignal);
        LwSciSyncPeer::verifyInternalAttr(
            reconciledList, LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            DEFAULT_RECONCILED_PRIMITIVE);
        LwSciSyncPeer::verifyInternalAttr(
            reconciledList, LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            DEFAULT_RECONCILED_PRIMITIVE);
        LwSciSyncPeer::verifyInternalAttr(
            reconciledList, LwSciSyncInternalAttrKey_SignalerPrimitiveCount, 1);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Reconciled attribute list transport
 * @verify{@jama{13052185}} - Reconciled attribute list import
 * @verify{@jama{13545689}} - Permissions of the imported reconciled list
 */
LWSCISYNC_TRANSPORT_TEST(ReconciledListTransport, 0)

/**
 * @jama{0} - Reconciled attribute list transport
 * Test Case 1. Verify that imported reconciled  attribute list has correct
 * attributes and primitive info.
 * @verify{@jama{13052185}} - Reconciled attribute list import
 * @verify{@jama{13545689}} - Permissions of the imported reconciled list
 */
LWSCISYNC_TRANSPORT_TEST_CASE(ReconciledListTransport, WaiterSuccess)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();

        ASSERT_EQ(
            LwSciError_Success,
            LwSciSyncAttrListSetAttrs(signalerAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                                      LwSciSyncPeer::attrs.cpuSignaler.size()));

        // Import waiter attr lists
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList.get(), nullptr);

        LwSciSyncPeer::verifyAttr(reconciledList.get(),
                                  LwSciSyncAttrKey_ActualPerm,
                                  LwSciSyncAccessPerm_WaitSignal);

        // Export reconciled attr list
        auto listDescBuf = peer.exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();

        ASSERT_EQ(
            LwSciError_Success,
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        // Export unreconciled attr lists
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled attr list
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList = peer.importReconciledList(
            importedBuf, {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Verify attributes
        LwSciSyncPeer::verifyAttr(importedAttrList.get(),
                                  LwSciSyncAttrKey_NeedCpuAccess, true);
        LwSciSyncPeer::verifyAttr(importedAttrList.get(),
                                  LwSciSyncAttrKey_ActualPerm,
                                  LwSciSyncAccessPerm_WaitOnly);

        LwSciSyncPeer::verifyInternalAttr(
            importedAttrList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo, DEFAULT_RECONCILED_PRIMITIVE);

        // Verify slot count
        ASSERT_EQ(LwSciSyncAttrListGetSlotCount(importedAttrList.get()), 1);

        // Check unreconciled
        LwSciSyncPeer::checkAttrListIsReconciled(importedAttrList.get(), true);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Reconciled attribute list transport
 * Verify that import and validation against unsatisfied WaiterPrimitiveInfo
 * fails.
 * @verify{@jama{13544513}} - Validation on import reconciled list
 */
LWSCISYNC_TRANSPORT_TEST_CASE(ReconciledListTransport, ValidationFailure)
{
    LwSciError error = LwSciError_Success;
    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signalerAttrList = peer.createAttrList();

        ASSERT_EQ(
            LwSciError_Success,
            LwSciSyncAttrListSetAttrs(signalerAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuSignaler.data(),
                                      LwSciSyncPeer::attrs.cpuSignaler.size()));

        // Import waiter attr lists
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList =
            peer.importUnreconciledList(importedBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedAttrList.get(), nullptr);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), importedAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(reconciledList.get(), nullptr);

        // Export reconciled attr list
        auto listDescBuf = peer.exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        initIpc();
        peer.SetUp("lwscisync_a_1");

        // Create attribute lists with CPU waiter attributes
        auto waiterAttrList = peer.createAttrList();

        ASSERT_EQ(
            LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                      LwSciSyncPeer::attrs.cpuWaiter.data(),
                                      LwSciSyncPeer::attrs.cpuWaiter.size()),
            LwSciError_Success);

        // Export unreconciled attr lists
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Create unreconciled attr list for reconciled list import
        auto unreconciledList = peer.createAttrList();
        LwSciSyncPeer::setAttr(unreconciledList.get(),
                               LwSciSyncAttrKey_RequiredPerm,
                               LwSciSyncAccessPerm_WaitOnly);
        LwSciSyncPeer::setAttr(unreconciledList.get(),
                               LwSciSyncAttrKey_NeedCpuAccess, false);
        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      unreconciledList.get(),
#if (__x86_64__)
                      LwSciSyncPeer::attrs.waiterSyncpointAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSyncpointAttrs.size()),
#else
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
#endif
                  LwSciError_Success);

        // Import reconciled attr list
        // It should fail as the WaiterPrimitiveInfo is unsatisfied
        NegativeTestPrint();
        auto importedBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList = peer.importReconciledList(
            importedBuf, {unreconciledList.get()}, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);
        ASSERT_EQ(importedAttrList.get(), nullptr);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

class LwSciSyncValidateRequireDeterministicFences : public LwSciSyncInterProcessTest
{
};

TEST_F(LwSciSyncValidateRequireDeterministicFences, Transport)
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

        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  signalerAttrList.get(),
                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                  LwSciSyncPeer::attrs.cpuSignaler.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      signalerAttrList.get(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.signalerSemaphoreAttrs.size()),
                  LwSciError_Success);
        SET_ATTR(signalerAttrList.get(),
                 LwSciSyncAttrKey_RequireDeterministicFences, true);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Assert on LwSciSyncAttrListReconcile
        {
            auto newReconciledList = LwSciSyncPeer::reconcileLists(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_NE(newReconciledList.get(), nullptr);

            // Assert on result values
            LwSciSyncPeer::verifyAttr(newReconciledList.get(),
                                      LwSciSyncAttrKey_RequireDeterministicFences, true);
            LwSciSyncPeer::verifyInternalAttr(
                newReconciledList.get(),
                LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
            LwSciSyncPeer::verifyInternalAttr(
                newReconciledList.get(),
                LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);

            auto reconciledListDesc =
                peer->exportReconciledList(newReconciledList.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(peer->sendBuf(reconciledListDesc),
                  LwSciError_Success);
        }
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetAttrs(
                  waiterAttrList.get(),
                  LwSciSyncPeer::attrs.cpuWaiter.data(),
                  LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

        ASSERT_EQ(LwSciSyncAttrListSetInternalAttrs(
                      waiterAttrList.get(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.data(),
                      LwSciSyncPeer::attrs.waiterSemaphoreAttrs.size()),
                  LwSciError_Success);

        // Export unreconciled waiter list to Peer A
        auto listDesc = peer->exportUnreconciledList(
                {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDesc), LwSciError_Success);

        {
            auto reconciledListDesc = peer->recvBuf(&error);
            ASSERT_EQ(error, LwSciError_Success);
            auto reconciledList = peer->importReconciledList(
                reconciledListDesc, {waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);

            ASSERT_TRUE(
                    LwSciSyncPeer::verifyAttrNew(
                        reconciledList.get(),
                        LwSciSyncAttrKey_RequireDeterministicFences, true));
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
