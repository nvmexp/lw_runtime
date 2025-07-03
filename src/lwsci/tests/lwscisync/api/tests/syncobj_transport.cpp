/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include "lwscisync_ipc_peer_old.h"

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_OBJECT_TRANSPORT_TEST(testSuite, JamaID)                     \
    class testSuite : public LwSciSyncTransportTest<JamaID>                    \
    {                                                                          \
    };

/* Declare additional test case for a test */
#define LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(testSuite, testName)              \
    TEST_F(testSuite, testName)

/**
 * @jama{0} - LwSciSync object allocation
 *
 * @brief Verify LwSciSyncObjAlloc returns LwSciSyncObject with expected
 * permissions and primitive type.
 *
 * 1. Process A creates CPU signaler attr list.
 * 2. Process B creates CPU waiter attr list, exports to A.
 * 3. Process A reconciles attr list, exports to B.
 * 4. Process B allocates sync object. Expect input reconciled list to be equal
 * to SyncObject's list.
 * 5. A allocates sync object. Expect input reconciled list to be equal to
 * SyncObject's list.
 *
 * @verify{@jama{13053547}} - LwSciSync object allocation
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST(LwSciSyncObjAllocTest, 0)
LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(LwSciSyncObjAllocTest, Success)
{
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

        LwSciError error = LwSciError_Success;
        auto unreconciledListDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer.importUnreconciledList(unreconciledListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList = peer.reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledListDescBuf =
            peer.exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(reconciledListDescBuf), LwSciError_Success);

        auto syncObj = peer.allocateSyncObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get attr list from the sync obj
        LwSciSyncAttrList syncObjList = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjGetAttrList(syncObj.get(), &syncObjList));
        ASSERT_NE(syncObjList, nullptr);

        // Verify attributes
        LwSciSyncPeer::checkAttrListsEqual(reconciledList.get(), syncObjList);

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
        LwSciError error = LwSciError_Success;
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        auto reconciledListDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer.importReconciledList(
            reconciledListDescBuf, {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.allocateSyncObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get attr list from the sync obj
        LwSciSyncAttrList syncObjList = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjGetAttrList(syncObj.get(), &syncObjList));
        ASSERT_NE(syncObjList, nullptr);

        // Verify attributes
        LwSciSyncPeer::checkAttrListsEqual(reconciledList.get(), syncObjList);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Reconciliation and object allocation
 *
 * @brief When requested via the LwSciSync API ICD, Rev 1.0, the LwSciSync shall
 * perform reconciliation, followed by LwSciSync Object allocation with the
 * inputs provided, and return a new LwSciSync Object handle bound to a new
 * LwSciSync Object bound to the resulting reconciled attribute list if the
 * reconciliation did not fail.
 *
 * 1. Process A creates CPU waiter attr list, exports to B.
 * 2. Process B creates CPU signaler attr list
 * 3. Process B calls LwSciSyncAttrListReconcileAndObjAlloc()
 * 4. Verify object's ActualPerm should be WaitSignal, and NeedCpuAccess TRUE.
 *
 * @verify{@jama{13141957}} - Reconciliation and object allocation
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST(LwSciSyncObjReconcileAndAllocTest, 0)

LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(LwSciSyncObjReconcileAndAllocTest, Success)
{
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

        LwSciError error = LwSciError_Success;
        auto unreconciledListDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer.importUnreconciledList(unreconciledListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get attr list from the sync obj
        LwSciSyncAttrList syncObjList = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjGetAttrList(syncObj.get(), &syncObjList));
        ASSERT_NE(syncObjList, nullptr);

        // Verify attributes
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_NeedCpuAccess, true);
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_ActualPerm,
                        LwSciSyncAccessPerm_WaitSignal);
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
        LwSciError error = LwSciError_Success;
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - LwSciSyncAttrList and Object transport
 *
 * @brief Verify LwSciSyncIpcExportAttrListAndObj() /
 * LwSciSyncIpcImportAttrListAndObj() pair produces LwSciSyncObject with
 * expected permissions and primitive type.
 *
 * 1. Process A creates CPU signaler attr list.
 * 2. Process B creates CPU waiter attr list, exports to A.
 * 3. Process A reconciles attr list, allocate sync object.
 * 4. Verify object's ActualPerm should be WaitSignal, and NeedCpuAccess TRUE.
 * 5. A exports reconciled list and sync obect to B.
 * 6. B imports object with Waiter permissions, object's ActualPerm should be
 * WaitOnly, and NeedCpuAccess TRUE.
 *
 * @verify{@jama{13053871}} - Transporting attribute list and object
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST(LwSciSyncAttrListAndObjTransportTest, 0);

LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(LwSciSyncAttrListAndObjTransportTest,
                                     Success)
{
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

        LwSciError error = LwSciError_Success;
        auto unreconciledListDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer.importUnreconciledList(unreconciledListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList = peer.reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer.allocateSyncObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get attr list from the sync obj
        LwSciSyncAttrList syncObjList = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjGetAttrList(syncObj.get(), &syncObjList));
        ASSERT_NE(syncObjList, nullptr);

        // Verify attributes
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_NeedCpuAccess, true);
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_ActualPerm,
                        LwSciSyncAccessPerm_WaitSignal);

        auto attrListAndObjDesc = peer.exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(attrListAndObjDesc), LwSciError_Success);
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
        LwSciError error = LwSciError_Success;
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

        // Get attr list from the sync obj
        LwSciSyncAttrList syncObjList = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjGetAttrList(syncObj.get(), &syncObjList));
        ASSERT_NE(syncObjList, nullptr);

        // Verify attributes
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_NeedCpuAccess, true);
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_ActualPerm,
                        LwSciSyncAccessPerm_WaitOnly);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{0} - Invalid permissions granted
 *
 * @verify{@jama{13545317}} - Invalid permissions granted
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST(LwSciSyncObjTransportIlwalidPermissions, 0);

/**
 * @jama{0} - Invalid permissions granted
 * @brief Test verifies that peer cannot import or export sync object with
 * inappropriate permissions
 *
 * 1. Process A creates CPU signaler attr list, exports to B.
 * 2. Process B creates CPU waiter attr list, exports to A.
 * 3. Process C creates CPU waiter attr list, exports to B, B exports to A.
 * 3. Process A reconciles and allocates object, exports to B.
 * 4. Process B try to import with bigger permissions, fails
 * 5. Process B imports with correct permissions, success
 * 5. Process B exports object with bigger permissions and smaller permissions -
 * fails.
 *
 * @verify{@jama{13545317}} - Invalid permissions granted
 * @verify{@jama{13545393}} - Granted signaling/waiting permissions are too small
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(LwSciSyncObjTransportIlwalidPermissions,
                                     Basic)
{
    LwSciError error = LwSciError_Success;

    pid_t peers[3] = {0};

    if ((pid = peers[0] = fork()) == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_0");

        // Create attribute lists with CPU Signaler attributes
        auto signaler = peer.createAttrList();

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      signaler.get(), LwSciSyncPeer::attrs.cpuSignaler.data(),
                      LwSciSyncPeer::attrs.cpuSignaler.size()));

        // Import waiter A attr lists
        auto waiterADescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterA = peer.importUnreconciledList(waiterADescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import waiter B attr lists
        auto waiterBDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterB = peer.importUnreconciledList(waiterBDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile and allocate
        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
            {signaler.get(), waiterA.get(), waiterB.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObjDesc = peer.exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(syncObjDesc), LwSciError_Success);

        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else if ((pid = peers[1] = fork()) == 0) {
        initIpc();
        peer.SetUp("lwscisync_a_1");
        otherPeer.SetUp("lwscisync_b_0", peer);

        // Create attribute lists with CPU waiter attributes
        auto waiter = peer.createAttrList();

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      waiter.get(), LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        auto waiterDescBuf =
            peer.exportUnreconciledList({waiter.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(waiterDescBuf), LwSciError_Success);

        auto importedWaiterBDescBuf = otherPeer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto waiterB = otherPeer.importUnreconciledList(importedWaiterBDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto waiterBDescBuf = peer.exportUnreconciledList({waiterB.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(peer.sendBuf(waiterBDescBuf), LwSciError_Success);

        auto syncObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // Import sync object with bigger permissions
            NEGATIVE_TEST();
            peer.importAttrListAndObj(syncObjDesc, {waiter.get()},
                                      LwSciSyncAccessPerm_WaitSignal, &error);

            ASSERT_EQ(error, LwSciError_BadParameter);
        }

        // Import sync object with correct permissions
        auto syncObj = peer.importAttrListAndObj(
            syncObjDesc, {waiter.get()}, LwSciSyncAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObj.get(), nullptr);

        {
            // Input permissions bigger than expected
            NegativeTestPrint();
            otherPeer.exportAttrListAndObj(
                syncObj.get(), LwSciSyncAccessPerm_WaitSignal, &error);

            ASSERT_EQ(error, LwSciError_BadParameter);
        }

        {
            // Input permissions smaller than expected
            NegativeTestPrint();
            otherPeer.exportAttrListAndObj(syncObj.get(),
                                           (LwSciSyncAccessPerm)0, &error);

            ASSERT_EQ(error, LwSciError_BadParameter);
        }

        ASSERT_EQ(otherPeer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
    } else if ((pid = peers[2] = fork()) == 0) {
        initIpc();
        peer.SetUp("lwscisync_b_1");

        // Create attribute lists with CPU waiter attributes
        auto waiter = peer.createAttrList();

        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncAttrListSetAttrs(
                      waiter.get(), LwSciSyncPeer::attrs.cpuWaiter.data(),
                      LwSciSyncPeer::attrs.cpuWaiter.size()));

        auto waiterDescBuf =
            peer.exportUnreconciledList({waiter.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(waiterDescBuf), LwSciError_Success);

        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peer : peers) {
            TEST_COUT << "Wait for pid " << peer;
            status |= wait_for_child_fork(peer);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

/**
 * @jama{0} - LwSciSyncObj transport
 *
 * @verify{@jama{13053733}} - Exporting LwSciSync object
 * @verify{@jama{13053745}} - Importing LwSciSync object
 * @verify{@jama{13053397}} - Object's ActualPerm import
 * @verify{@jama}{13545353} - Granting signaling/waiting permissions on object
 * export
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST(LwSciSyncObjTransportTest, 0);

LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(LwSciSyncObjTransportTest, Success)
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

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        ASSERT_NE(syncObj, nullptr);
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        // Export sync obj
        auto syncObjDesc =
            peer.exportSyncObj(syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendExportDesc(syncObjDesc), LwSciError_Success);
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

        // Import sync obj
        auto syncObjDesc =
            peer.recvExportDesc<LwSciSyncObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObjDesc.get(), nullptr);
        auto importedSyncObj =
            peer.importSyncObj(syncObjDesc.get(), importedAttrList.get(),
                               LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedSyncObj.get(), nullptr);

        // Get attr list from the sync obj
        LwSciSyncAttrList syncObjList = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjGetAttrList(importedSyncObj.get(), &syncObjList));
        ASSERT_NE(syncObjList, nullptr);

        // Verify attributes
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_NeedCpuAccess, true);
        peer.verifyAttr(syncObjList, LwSciSyncAttrKey_ActualPerm,
                        LwSciSyncAccessPerm_WaitOnly);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

/**
 * @jama{15247278} - Update Requests to Sync Points
 * @jama{15247304} - Integrity of Sync Points
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(LwSciSyncObjTransportTest,
                                     SyncObjStateIntegrity)
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

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        ASSERT_NE(syncObj, nullptr);
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        // Export sync obj
        auto syncObjDesc =
            peer.exportSyncObj(syncObj, LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendExportDesc(syncObjDesc), LwSciError_Success);

        // Signaler process can update syncObj state
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        ASSERT_EQ(LwSciSyncObjGenerateFence(syncObj, &syncFence),
                  LwSciError_Success);
        LwSciSyncFenceClear(&syncFence);
        ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);
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

        // Import sync obj
        auto syncObjDesc =
            peer.recvExportDesc<LwSciSyncObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObjDesc.get(), nullptr);
        auto importedSyncObj =
            peer.importSyncObj(syncObjDesc.get(), importedAttrList.get(),
                               LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedSyncObj.get(), nullptr);

        // Waiter process cannot update syncObj state
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncObjGenerateFence(importedSyncObj.get(), &syncFence),
                  LwSciError_BadParameter);
        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncObjSignal(importedSyncObj.get()),
                  LwSciError_BadParameter);

        // wait for other peer to exit
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);
        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}

static LwSciError fillSignalerListForValidateExportDescriptors(
    LwSciSyncAttrList signalerAttrList)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncPeer::attrs.defaultPlatformPrimitive };

    bool cpuSignaler = false;
    LwSciSyncAccessPerm signalerAccessPerm = LwSciSyncAccessPerm_SignalOnly;
    LwSciSyncAttrKeyValuePair signalerKeyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuSignaler,
             .len = sizeof(cpuSignaler),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &signalerAccessPerm,
             .len = sizeof(signalerAccessPerm),
        },
    };

    uint32_t signalerPrimitiveCount = 1U;
    LwSciSyncInternalAttrKeyValuePair signalerInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
             .value = (void*)&signalerPrimitiveCount,
             .len = sizeof(signalerPrimitiveCount),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(signalerAttrList, signalerKeyValue,
            sizeof(signalerKeyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(signalerAttrList, signalerInternalKeyValue,
            sizeof(signalerInternalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiterListForValidateExportDescriptors(
    LwSciSyncAttrList waiterAttrList)
{
    LwSciError err = LwSciError_Success;
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncPeer::attrs.defaultPlatformPrimitive };
    bool cpuWaiter = true;
    LwSciSyncAccessPerm waiterAccessPerm = LwSciSyncAccessPerm_WaitOnly;
    bool contextInsensitiveFenceExports = true;
    LwSciSyncAttrKeyValuePair waiterKeyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuWaiter,
             .len = sizeof(cpuWaiter),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &waiterAccessPerm,
             .len = sizeof(waiterAccessPerm),
        },
        {    .attrKey = LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
             .value = (void*)&contextInsensitiveFenceExports,
             .len = sizeof(contextInsensitiveFenceExports),
        },
    };

    LwSciSyncInternalAttrKeyValuePair waiterInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(waiterAttrList, waiterKeyValue,
            sizeof(waiterKeyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(waiterAttrList, waiterInternalKeyValue,
            sizeof(waiterInternalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

/**
 * @jama{22820614} - Validate export descriptor on import
 * Verify that that when importing valid unreconciled attribute list,
 * reconciled attribute list, LwSciSync Object or attribute list and
 * object export descriptors, the LwSciSync reports success.
 *
 * @verify{@jama{18844038}} - Validate export descriptor on import
 */
LWSCISYNC_OBJECT_TRANSPORT_TEST(LwSciSyncValidateDescriptorTest, 22820614)
LWSCISYNC_OBJECT_TRANSPORT_TEST_CASE(LwSciSyncValidateDescriptorTest, Success)
{
    LwSciError error = LwSciError_Success;

    pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        // -- Setup
        initIpc();
        peer.SetUp("lwscisync_a_0");

        auto signalerAttrList = peer.createAttrList();
        error = fillSignalerListForValidateExportDescriptors(
            signalerAttrList.get());
        ASSERT_EQ(LwSciError_Success, error);

        // -- Testing

        // Import waiter attr list
        auto unreconciledListDescBuf = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedWaiterAttrList =
            peer.importUnreconciledList(unreconciledListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile attr lists
        auto reconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), importedWaiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export reconciled attr list
        auto listDescBuf = peer.exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Allocate sync obj
        LwSciSyncObj syncObj = nullptr;
        ASSERT_EQ(LwSciError_Success,
                  LwSciSyncObjAlloc(reconciledList.get(), &syncObj));
        auto syncObjPtr =
            std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

        // Export sync obj
        auto syncObjDesc =
            peer.exportSyncObj(syncObj, LwSciSyncAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendExportDesc(syncObjDesc), LwSciError_Success);

        // Export list and obj
        auto syncObjDesc2 =
            peer.exportAttrListAndObj(syncObj, LwSciSyncAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(syncObjDesc2), LwSciError_Success);

        /* needed for Desktop */
        ASSERT_EQ(peer.waitComplete(), LwSciError_Success);
    } else {
        // -- Setup
        initIpc();
        peer.SetUp("lwscisync_a_1");

        auto waiterAttrList = peer.createAttrList();
        error = fillWaiterListForValidateExportDescriptors(
            waiterAttrList.get());
        ASSERT_EQ(LwSciError_Success, error);

        // -- Testing
        // Export unreconciled attr list
        auto listDescBuf =
            peer.exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer.sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled attr list
        auto reconciledDescriptor = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedAttrList = peer.importReconciledList(
            reconciledDescriptor, {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import sync obj
        auto syncObjDesc =
            peer.recvExportDesc<LwSciSyncObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObjDesc.get(), nullptr);
        auto importedSyncObj =
            peer.importSyncObj(syncObjDesc.get(), importedAttrList.get(),
                               LwSciSyncAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedSyncObj.get(), nullptr);

        // Import list and obj
        auto attrListAndObjDesc = peer.recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto importedSyncObj2 = peer.importAttrListAndObj(
            attrListAndObjDesc, {},
            LwSciSyncAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* needed for Desktop */
        ASSERT_EQ(peer.signalComplete(), LwSciError_Success);

        ASSERT_EQ(0, wait_for_child_fork(pid));
    }
}
