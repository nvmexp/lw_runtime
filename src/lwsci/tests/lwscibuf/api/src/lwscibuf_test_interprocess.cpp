/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscibuf_interprocess_test.h"
#include "lwsci_igpu_only_test.h"

class TestLwSciBufInterprocess : public LwSciBufInterProcessTest
{
};

class TestLwSciBufInterprocessGpu : public LwSciBufInterProcessTest,
    public LwSciiGpuOnlyTest
{
    void SetUp() override
    {
        LwSciBufInterProcessTest::SetUp();
        LwSciiGpuOnlyTest::SetUp();
    }

    void TearDown() override
    {
        LwSciiGpuOnlyTest::TearDown();
        LwSciBufInterProcessTest::TearDown();
    }
};

TEST_F(TestLwSciBufInterprocessGpu, EmptyHops)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peerAToB = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peerAToB);
        peerAToB->SetUp("ipc_test_a_0");

        auto list = peerAToB->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Create Attribute List A
        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            LwSciBufAttrValAccessPerm requiredPerm =
                LwSciBufAccessPerm_Readonly;
            bool needCpuAccess = false;
            uint64_t alignment = 1U;

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     needCpuAccess);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     requiredPerm);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, alignment);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_GpuId, testiGpuId);
        }

        auto listDescBuf =
            peerAToB->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerAToB->sendBuf(listDescBuf), LwSciError_Success);

        // Import Reconciled List from B
        auto reconciledListDescBuf = peerAToB->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peerAToB->importReconciledList(
            reconciledListDescBuf, {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(peerAToB->signalComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peerBToA = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peerBToA);
        peerBToA->SetUp("ipc_test_a_1");

        auto peerBToC = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peerBToC);
        peerBToC->SetUp("ipc_test_b_0");

        // Import A's Unreconciled Attribute List
        auto upstreamListDescBuf = peerBToA->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peerBToA->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Re-export A's Attribute List to C
        auto listDescBuf =
            peerBToC->exportUnreconciledList({upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerBToC->sendBuf(listDescBuf), LwSciError_Success);

        // Import Reconciled List from C
        auto reconciledListDescBuf = peerBToC->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peerBToC->importReconciledList(
            reconciledListDescBuf, {upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export the Reconciled List to A
        auto reExportReconciledListDescBuf =
            peerBToA->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerBToA->sendBuf(reExportReconciledListDescBuf),
                  LwSciError_Success);

        // Wait for peers to exit before releasing allocated object
        ASSERT_EQ(peerBToA->waitComplete(), LwSciError_Success);
        ASSERT_EQ(peerBToC->signalComplete(), LwSciError_Success);
    } else if ((pids[2] = fork()) == 0) {
        pid = 3;
        auto peerCToB = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peerCToB);
        peerCToB->SetUp("ipc_test_b_1");

        auto list = peerCToB->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Create Attribute List C
        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            LwSciBufAttrValAccessPerm requiredPerm =
                LwSciBufAccessPerm_ReadWrite;
            bool needCpuAccess = true;
            uint64_t alignment = 1U;
            uint64_t bufSize = 4096U;

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     needCpuAccess);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     requiredPerm);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, alignment);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, bufSize);
        }

        // Import A's Unreconciled Attribute List, exported through B
        auto upstreamListDescBuf = peerCToB->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peerCToB->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile A + B's attribute lists
        auto reconciledList = peerCToB->attrListReconcile(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate Buffer
        auto bufObj = peerCToB->allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export the Reconciled List to B
        auto reconciledListDescBuf =
            peerCToB->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerCToB->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Wait for peers to exit before releasing allocated object
        ASSERT_EQ(peerCToB->waitComplete(), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

/**
 * This test verifies that LwSciBufObjAlloc() does not modify the reconciled
 * LwSciBufAttrList associated with an existing LwSciBufObj when the provided
 * reconciled LwSciBufAttrList requires an update to the permissions specified
 * in LwSciBufGeneralAttrKey_ActualPerm.
 * */
TEST_F(TestLwSciBufInterprocess, LwSciBufObjAlloc_AttributeListModification)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_a_0");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            LwSciBufAttrValAccessPerm requiredPerm =
                LwSciBufAccessPerm_Readonly;
            bool needCpuAccess = true;
            uint64_t alignment = 1U;
            uint64_t size = 1024U;

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     needCpuAccess);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     requiredPerm);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, alignment);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, alignment);
        }

        auto listDescBuf =
            peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import Reconciled List
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(
            reconciledListDescBuf, {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import buffer object
        auto objDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto bufObj = peer->importBufObj(
            objDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            LwSciBufAttrList bufObjAttrList = nullptr;
            error = LwSciBufObjGetAttrList(bufObj.get(), &bufObjAttrList);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_TRUE(LwSciBufPeer::verifyAttr(
                bufObjAttrList, LwSciBufGeneralAttrKey_ActualPerm,
                LwSciBufAccessPerm_Readonly));
        }

        // Now try allocating its own LwSciBufObj
        // This should not modify the provided reconciled LwSciBufAttrList
        auto bufObj2 = peer->allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(
            reconciledList.get(), LwSciBufGeneralAttrKey_ActualPerm,
            LwSciBufAccessPerm_Readonly));

        // The LwSciBufAttrList associated with the original LwSciBufObj should
        // not get modified here. The permission on that LwSciBufObj should
        // still be LwSciBufAccessPerm_Readonly
        {
            LwSciBufAttrList bufObjAttrList = nullptr;
            error = LwSciBufObjGetAttrList(bufObj.get(), &bufObjAttrList);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_TRUE(LwSciBufPeer::verifyAttr(
                bufObjAttrList, LwSciBufGeneralAttrKey_ActualPerm,
                LwSciBufAccessPerm_Readonly));
        }

        // However, the LwSciBufAttrList associated with the _new_ LwSciBufObj
        // should have LwSciBufAccessPerm_ReadWrite
        {
            LwSciBufAttrList bufObjAttrList = nullptr;
            error = LwSciBufObjGetAttrList(bufObj2.get(), &bufObjAttrList);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_TRUE(LwSciBufPeer::verifyAttr(
                bufObjAttrList, LwSciBufGeneralAttrKey_ActualPerm,
                LwSciBufAccessPerm_ReadWrite));
        }

        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_a_1");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            LwSciBufAttrValAccessPerm requiredPerm =
                LwSciBufAccessPerm_ReadWrite;
            bool needCpuAccess = true;
            uint64_t alignment = 1U;
            uint64_t size = 1024U;

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     needCpuAccess);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     requiredPerm);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, alignment);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, alignment);
        }

        // Import A's Unreconciled Attribute List
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile A + B's attribute lists
        auto reconciledList = peer->attrListReconcile(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate Buffer
        auto bufObj = peer->allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export the Reconciled List to A
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export LwSciBufObj to A
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

#if !defined(__x86_64__)
// This technically is verifying CheetAh-specific behavior since we're exercising
// the fact that during export LwRmMemGetSciIpcId() does some reference
// counting of its own when exporting a memory handle.
//
// On Desktop, the platform-specific export just takes the memory handle and
// does the duplication on import, so the allocator cannot free the handle
// prior to the other peer completing its import.
TEST_F(TestLwSciBufInterprocess, LwMapReferenceCounting)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Create Attribute List A
        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            LwSciBufAttrValAccessPerm requiredPerm =
                LwSciBufAccessPerm_ReadWrite;
            bool needCpuAccess = true;
            uint64_t alignment = 1U;

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     needCpuAccess);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     requiredPerm);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        }

        // Import Unreconciled Attribute List
        auto unreconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(unreconciledListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile
        auto reconciledList = peer->attrListReconcile(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto bufObj = peer->allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export the Reconciled List to B
        auto attrListAndObjDesc =
            peer->exportAttrListAndObj(bufObj.get(), LwSciBufAccessPerm_Readonly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(attrListAndObjDesc), LwSciError_Success);

        // Explicitly call the destructor to force the deinit APIs to be called
        bufObj.reset();
        // Signal to start import
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);

        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Create Attribute List B
        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            LwSciBufAttrValAccessPerm requiredPerm =
                LwSciBufAccessPerm_Readonly;
            bool needCpuAccess = true;
            uint64_t alignment = 1U;
            uint64_t size = 4096U;

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     needCpuAccess);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     requiredPerm);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, alignment);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        }

        // Export Unreconciled list
        auto listDescBuf =
            peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        // Wait to ensure that the deinit APIs were called in the other peer
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
        auto bufObj = peer->importAttrListAndObj(
            attrListAndObjDesc, {list.get()},
            LwSciBufAccessPerm_Readonly, &error);
        // This import should be successful since LwMap should properly keep
        // track of underlying references
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
#endif
