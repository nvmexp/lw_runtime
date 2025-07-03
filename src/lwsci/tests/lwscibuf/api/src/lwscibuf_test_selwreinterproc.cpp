/*
 * lwscibuf_test_selwreinterproc.cpp
 *
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscibuf_interprocess_test.h"
#include <memory>

class LwSciBufTamperableIpcPeer : public LwSciBufIpcPeer
{
public:
    std::shared_ptr<LwSciBufObjIpcExportDescriptor>
    exportBufObjWithTamperedPerms(LwSciBufObj bufObj,
                                  LwSciBufAttrValAccessPerm permissions,
                                  LwSciBufAttrValAccessPerm tamperedPermissions,
                                  LwSciError* error)
    {
        auto bufObjDesc = exportBufObj(bufObj, permissions, error);

        if (*error != LwSciError_Success) {
            return std::shared_ptr<LwSciBufObjIpcExportDescriptor>(nullptr);
        }

        LwSciBufAttrValAccessPerm originalPerm = (LwSciBufAttrValAccessPerm)0U;
        memcpy(&originalPerm, (((uint8_t*)bufObjDesc.get()) + 59),
               sizeof(tamperedPermissions));

        /* Negative test. Tamper permissions */
        memcpy((((uint8_t*)bufObjDesc.get()) + 59), &tamperedPermissions,
               sizeof(tamperedPermissions));

        return bufObjDesc;
    }

    std::shared_ptr<LwSciBufObjRefRec> importBufObjTamperedPerms(
        LwSciBufObjIpcExportDescriptor* bufObjDesc,
        LwSciBufAttrList inputAttrList, LwSciBufAttrValAccessPerm permissions,
        LwSciBufAttrValAccessPerm origVal, LwSciError* error)
    {
        /* Fix the tampered permissions to the actual permission */
        memcpy((((uint8_t*)bufObjDesc) + 59), (void*)&origVal, sizeof(origVal));

        /* Then actually import the buffer to decrement refcount */
        return importBufObj(bufObjDesc, inputAttrList, permissions, error);
    }
};

class TestLwSciBufSelwreBuffer : public LwSciBufInterProcessTest
{
public:
    void testIncorrectEndpoint(std::shared_ptr<LwSciBufIpcPeer> downstreamPeer,
                               std::shared_ptr<LwSciBufIpcPeer> incorrectPeer,
                               std::shared_ptr<LwSciBufAttrListRec> attrList,
                               LwSciBufAttrValAccessPerm permissions,
                               std::shared_ptr<LwSciBufObjRefRec>& bufferObj)
    {
        LwSciBufObj rawObj = nullptr;
        LwSciBufObjIpcExportDescriptor objDesc;

        LwSciError error = LwSciError_Success;
        // Get descriptor
        auto bufObjDescBuf =
            downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* importing object */
        {
            NEGATIVE_TEST();

            auto bufObj = incorrectPeer->importBufObj(
                bufObjDescBuf.get(), attrList.get(), permissions, &error);
            ASSERT_NE(error, LwSciError_Success);
        }

        bufferObj = downstreamPeer->importBufObj(
            bufObjDescBuf.get(), attrList.get(), permissions, &error);
        ASSERT_EQ(error, LwSciError_Success);
    }

    void setupAttrList1(std::shared_ptr<LwSciBufAttrListRec> list)
    {
        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size,
                 (uint64_t)(128 * 1024U));
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align,
                 (uint64_t)(4U * 1024U));
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                 LwSciBufAccessPerm_ReadWrite);

        SET_INTERNAL_ATTR(list.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
    }

    void setupAttrList2(std::shared_ptr<LwSciBufAttrListRec> list)
    {
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size,
                 (uint64_t)(128 * 1024U));
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align,
                 (uint64_t)(8U * 1024U));
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                 LwSciBufAccessPerm_Readonly);
    }
};

TEST_F(TestLwSciBufSelwreBuffer, SelwresharingLessPermission)
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

        setupAttrList1(list);

        // Import from Peer B and reconcile
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto bufObj =
            LwSciBufPeer::allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        {
            NEGATIVE_TEST();
            auto objDescBuf = peer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_IlwalidOperation);
        }

        // Wait for Peer A to exit before releasing allocated object
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
                     LwSciBufType_RawBuffer);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size,
                     (uint64_t)(128 * 1024U));
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align,
                     (uint64_t)(8U * 1024U));
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
            // LwSciBufAccessPerm_ReadWrite instead of
            // LwSciBufAccessPerm_Readonly
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_ReadWrite);
        }

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
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

TEST_F(TestLwSciBufSelwreBuffer, Selwresharing)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    const uint8_t sampleData[] = {
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0,
        0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x11, 0x21, 0x31, 0x41, 0x51,
        0x61, 0x71, 0x81, 0x91, 0xA1, 0xB1, 0xC1, 0xD1, 0xE1, 0xF1,

    };
    const uint8_t sampleData1[] = {
        0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x60, 0x70, 0x80, 0x90, 0xA0,
        0x10, 0x20, 0x30, 0x40, 0x50, 0xB1, 0xC1, 0xD1, 0xE1, 0xF1,
        0x61, 0x71, 0x81, 0x91, 0xA1, 0x11, 0x21, 0x31, 0x41, 0x51,
    };

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList1(list);

        // Import from Peer B and reconcile
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            peer->attrListReconcile({list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate object
        auto bufObj = peer->allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object to Peer B
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        // Wait until the peer has imported before starting
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
        LwSciBufInterProcessTest::testAccessPermissions(
            peer, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite, nullptr, 0,
            sampleData, sizeof(sampleData));

        // Wait for Peer B to exit before releasing allocated object
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_B_A");

        auto ilwalidPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(ilwalidPeer);
        ilwalidPeer->SetUp("lwscibuf_ipc_D_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        std::shared_ptr<LwSciBufObjRefRec> bufObj;
        testIncorrectEndpoint(peer, ilwalidPeer, reconciledList,
                              LwSciBufAccessPerm_ReadWrite, bufObj);
        // Signal that import has completed
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);

        LwSciBufInterProcessTest::testAccessPermissions(
            nullptr, peer, bufObj, LwSciBufAccessPerm_ReadWrite, sampleData,
            sizeof(sampleData), sampleData1, sizeof(sampleData1));

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

TEST_F(TestLwSciBufSelwreBuffer, NegativeAccessPermission)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    const uint8_t sampleData[] = {
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0,
        0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x11, 0x21, 0x31, 0x41, 0x51,
        0x61, 0x71, 0x81, 0x91, 0xA1, 0xB1, 0xC1, 0xD1, 0xE1, 0xF1,
    };

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciBufTamperableIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList1(list);

        // Import from Peer B and reconcile
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            peer->attrListReconcile({list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate object
        auto bufObj = peer->allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // wewlad
        auto objDescBuf = peer->exportBufObjWithTamperedPerms(
            bufObj.get(), LwSciBufAccessPerm_Readonly,
            LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        LwSciBufInterProcessTest::testAccessPermissions(
            peer, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite, nullptr, 0,
            sampleData, sizeof(sampleData));

        // Wait until import finishes
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufTamperableIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object allocated by Peer A
        auto bufObjDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Attempt importing the tampered buffer with ReadWrite
        {
            NEGATIVE_TEST();
            auto bufObj =
                peer->importBufObj(bufObjDescBuf.get(), reconciledList.get(),
                                   LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_NE(error, LwSciError_Success);
        }

        auto bufObj = peer->importBufObjTamperedPerms(
            bufObjDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_Readonly, (LwSciBufAttrValAccessPerm)0x100,
            &error);
        ASSERT_NE(bufObj.get(), nullptr);
        EXPECT_EQ(error, LwSciError_Success);

        // Since we don't call testAccessPermissions() we need to clear the IPC
        // queue manually
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);

        // Signal to move onto next phase
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

// This subtest tests following scenario
// Consider peer A, B and C.
// IPC channel is setup between A <-> B
// IPC channel is setup between A <-> C
// Peer B exports unreconciled list to peer A.
// Peer C exports unreconciled list to peer A.
// Peer A reconciles the list, allocates object, exports reconciled list and
// object back to peer B & C.
// Verifies that A, the reconciler/allocator process is able to export the
// reconciled list/object to B & C and they are able to import it.
TEST_F(TestLwSciBufSelwreBuffer, ValidExport1)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;

    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        //Peer A
        pid = 1;
        auto peerAtoB = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peerAtoB);

        //setup IPC channel between Peer A & B
        peerAtoB->SetUp("lwscibuf_ipc_A_B");

        auto list = peerAtoB->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList1(list);

        auto peerAtoC = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peerAtoC);

        //setup IPC channel between Peer A & C
        peerAtoC->SetUp("lwscibuf_ipc_B_C", *peerAtoB);

        // Import unreconciled attribute list from Peer B
        auto peerBListDesc = peerAtoB->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto peerBlist =
            peerAtoB->importUnreconciledList(peerBListDesc, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import unreconciled attribute list from Peer C
        auto peerCListDesc = peerAtoC->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto peerClist =
            peerAtoC->importUnreconciledList(peerCListDesc, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile Peer A, B & C attribute lists
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), peerBlist.get(),
                peerClist.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate object shareable between A, B & C
        auto bufObj = LwSciBufPeer::allocateBufObj(reconciledList.get(),
                        &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peerAtoB->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerAtoB->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object to Peer B
        auto objDescBufAtoB = peerAtoB->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerAtoB->sendExportDesc(objDescBufAtoB), LwSciError_Success);

        // Export attr list back to Peer C
        // Note: We dont need to explicitly create export descriptor for
        // reconciled list again. Reuse the one from PeerAtoB.
        ASSERT_EQ(peerAtoC->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object to Peer C
        // Note: Need to create export descriptor for object explicitly for
        // Peer C since LwMap does book-keeping of exports for processes and
        // wont allow the import of the buffer in process C if we are not
        // explicitly exporting it.
        auto objDescBufAtoC = peerAtoC->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peerAtoC->sendExportDesc(objDescBufAtoC), LwSciError_Success);

        // Wait until the peer B has imported buffer
        ASSERT_EQ(peerAtoB->waitComplete(), LwSciError_Success);

        // Wait until the peer C has imported buffer
        ASSERT_EQ(peerAtoC->waitComplete(), LwSciError_Success);

    } else if ((pids[1] = fork()) == 0) {
        //Peer B
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get descriptor
        auto objDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object from Peer A
        auto bufObj = peer->importBufObj(
            objDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Signal that object import has completed
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);

    } else if ((pids[2] = fork()) == 0) {
        //Peer C
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & C
       peer->SetUp("lwscibuf_ipc_C_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get descriptor
        auto objDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object from Peer A
        auto bufObj = peer->importBufObj(
            objDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Signal that object import has completed
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

// This subtest tests following scenario
// Consider peer A, B and C.
// IPC channel is setup between A <-> B
// IPC channel is setup between A <-> C
// Peer B exports unreconciled list to peer A.
// Peer A reconciles the list, allocates object, exports reconciled list and
// object back to peer B. This should be successful since A & B are ilwolved in
// reconciliation and reconciled list and object are flowing in reverse path of
// unreconciled list.
// Now, peer A tries to export reconciled list and object to peer C. this should
// fail since peer C is not ilwolved in reconciliation.
TEST_F(TestLwSciBufSelwreBuffer, IlwalidExport1)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;

    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        //Peer A
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_A_B");

        // Setup IPC channel between Peer A & C
        // Note: Channel name is bit confusing. We are just using available
        // channel names in lwsciipc.cfg
        auto peer1 = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer1);
        peer1->SetUp("lwscibuf_ipc_B_C", *peer);

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList1(list);

        // Import unreconciled attribute list from Peer B
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile Peer A & B attribute lists
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()},
            &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate object shareable between A & B
        auto bufObj = LwSciBufPeer::allocateBufObj(reconciledList.get(),
                        &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object to Peer B
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        // Wait until the peer B has imported buffer
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);

        {
            NEGATIVE_TEST();

            // Export reconciled attribute list from Peer A & B back to Peer C.
            // We shouldnt be able to do this since C was not ilwolved in
            // reconciliation. (It was not ilwolved in IPC path which sent
            // unreconciled lists to A for reconciliation).
            auto reconciledListDescBuf1 =
                peer1->exportReconciledList(reconciledList.get(), &error);
            ASSERT_EQ(error, LwSciError_NotPermitted);

            // Export object from Peer A & B back to Peer C.
            // We shouldnt be able to do this since C was not ilwolved in
            // reconciliation. (It was not ilwolved in IPC path which sent
            // unreconciled lists to A for reconciliation).
            auto objDescBuf1 = peer1->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_EQ(error, LwSciError_NotPermitted);
        }

        // Signal Peer C that we're finished such that LwSciIpc does not race
        // when attempting to deliver a pulse on QNX
        ASSERT_EQ(peer1->signalComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        //Peer B
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get descriptor
        auto objDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object from Peer A
        auto bufObj = peer->importBufObj(
            objDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Signal that object import has completed
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);

    } else if ((pids[2] = fork()) == 0) {
        //Peer C
        pid = 3;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        // Setup IPC channel between Peer A & C and check if Peer A is able to
        // export reconciled attribute list/object over this channel.
        // Note: Channel name is bit confusing. We are just using available
        // channel names in lwsciipc.cfg
        peer->SetUp("lwscibuf_ipc_C_B");

        // Wait here for Peer A to finish such that LwSciIpc does not race when
        // attempting to deliver a pulse on QNX
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

// This subtest tests following scenario
// Consider peer A & B.
// IPC channel is setup between A <-> B
// Peer B exports unreconciled list to peer A.
// Peer A reconciles the list, allocates object, exports reconciled list and
// object back to peer B. This should be successful since A & B are ilwolved in
// reconciliation and reconciled list and object are flowing in reverse path of
// unreconciled list.
// Now, peer B tries to export reconciled list and object back to peer A.
// this should fail since reconciled list and object are NOT flowing in the
// reverse direction of the unreconciled list IPC path.
TEST_F(TestLwSciBufSelwreBuffer, DISABLED_IlwalidExport2)
{
    /* With C2c use-case, we are going to allow flow of reconciled lists/objects
     * in a direction that does not follow reverse IPC path in which
     * unreconciled lists flow. Thus temporarily disabling this test. Ultimately
     * this test will be repurposed such that the test case checks that the
     * reconciled lists/objects cannot be shared with the peers which are not
     * part of "sharing group".
     */
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;

    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        //Peer A
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList1(list);

        // Import unreconciled attribute list from Peer B
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile Peer A & B attribute lists
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()},
            &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate object shareable between A & B
        auto bufObj = LwSciBufPeer::allocateBufObj(reconciledList.get(),
                        &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object to Peer B
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        // Wait until the peer B has imported buffer
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);

    } else if ((pids[1] = fork()) == 0) {
        //Peer B
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get descriptor
        auto objDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object from Peer A
        auto bufObj = peer->importBufObj(
            objDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Signal that object import has completed
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);

        {
            NEGATIVE_TEST();
            // Now, try to export the reconciled list back to Peer A. This
            // should fail since we are NOT exporting object in reverse path of
            // the unreconciled lists.
            auto reconciledListDescBuf1 =
                peer->exportReconciledList(reconciledList.get(), &error);
            ASSERT_EQ(error, LwSciError_NotPermitted);

            // Now, try to export the LwSciBufObj back to Peer A. This should
            // fail since we are NOT exporting object in reverse path of the
            // unreconciled lists.
            auto objDescBuf1 = peer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_EQ(error, LwSciError_NotPermitted);
        }

    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

// This subtest tests following scenario
// Consider peer A, B and C.
// IPC channel is setup between A <-> B
// IPC channel is setup between B <-> C
// Peer C creates and exports unreconciled list to peer B.
// Peer B creates its own unreconciled list, imports unreconciled list from C
// and exports unreconciled lists from B & C to peer A.
// Peer A reconciles the lists, allocates object, exports reconciled list and
// object back to peer B. This should be successful since A & B are ilwolved in
// reconciliation and reconciled list and object are flowing in reverse path of
// unreconciled lists.
// Now, peer B tries to export reconciled list and object back to peer A.
// this should fail since reconciled list and object are NOT flowing in the
// reverse direction of the unreconciled list IPC path.
TEST_F(TestLwSciBufSelwreBuffer, DISABLED_IlwalidExport3)
{
    /* With C2c use-case, we are going to allow flow of reconciled lists/objects
     * in a direction that does not follow reverse IPC path in which
     * unreconciled lists flow. Thus temporarily disabling this test. Ultimately
     * this test will be repurposed such that the test case checks that the
     * reconciled lists/objects cannot be shared with the peers which are not
     * part of "sharing group".
     */
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;

    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        //Peer A
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList1(list);

        // Import unreconciled attribute list from Peer B
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile Peer A & B attribute lists
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()},
            &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate object shareable between A & B
        auto bufObj = LwSciBufPeer::allocateBufObj(reconciledList.get(),
                        &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object to Peer B
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        // Wait until the peer B has imported buffer
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);

    } else if ((pids[1] = fork()) == 0) {
        // Peer B
        pid = 2;
        auto downstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        auto upstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(downstreamPeer);
        peers.push_back(upstreamPeer);

        // setup IPC channel between Peer A & B
        downstreamPeer->SetUp("lwscibuf_ipc_B_A");

        // setup IPC channel between Peer B & C
        upstreamPeer->SetUp("lwscibuf_ipc_B_C", *downstreamPeer);

        auto list = downstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Import unreconciled attribute list from Peer C
        auto upstreamListDescBuf = upstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            upstreamPeer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export unreconciled lists to Peer A
        auto listDescBuf = downstreamPeer->exportUnreconciledList({list.get(),
                            upstreamList.get()},
                            &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(downstreamPeer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = downstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList =
            downstreamPeer->importReconciledList(reconciledListDescBuf,
                                                    {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get descriptor
        auto objDescBuf =
            downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object from Peer A
        auto bufObj = downstreamPeer->importBufObj(
            objDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Signal that object import has completed
        ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

        {
            NEGATIVE_TEST();
            // Now, try to export the reconciled list back to Peer A. This
            // should fail since we are NOT exporting object in reverse path of
            // the unreconciled lists.
            auto reconciledListDescBuf1 =
                downstreamPeer->exportReconciledList(reconciledList.get(),
                    &error);
            ASSERT_EQ(error, LwSciError_NotPermitted);

            // Now, try to export the LwSciBufObj back to Peer A. This should
            // fail since we are NOT exporting object in reverse path of the
            // unreconciled lists.
            auto objDescBuf1 = downstreamPeer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_EQ(error, LwSciError_NotPermitted);
        }

    } else if ((pids[2] = fork()) == 0) {
        //Peer C
        pid = 2;
        auto downstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(downstreamPeer);

        //setup IPC channel between Peer B & C
        downstreamPeer->SetUp("lwscibuf_ipc_C_B");

        auto list = downstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer B
        auto listDescBuf = downstreamPeer->exportUnreconciledList({list.get()},
                            &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(downstreamPeer->sendBuf(listDescBuf), LwSciError_Success);

        // Note: We are not covering importing reconciled list and object from
        // peer B in this test since this scenario is covered in other LwSciBuf
        // tests

    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

// This subtest tests following scenario
// Consider peer A, B and C.
// IPC channel is setup between A <-> B
// IPC channel is setup between A <-> C
// Peer B exports unreconciled list to peer A.
// Peer A reconciles the list, allocates object, exports reconciled list and
// object back to peer B. This should be successful since A & B are ilwolved in
// reconciliation and reconciled list and object are flowing in reverse path of
// unreconciled list.
// Now, peer A exports reconciled list and object obtained for Peer B over to
// Peer C's channel. This should be successful since export descriptors are
// valid.
// Now, during import, ideally, Peer C shall not allow import of reconciled list
// and object since export descriptors were not meant for that peer to be
// imported. However, lwrrently, reconciled list import passes because
// LwSciBuf driver does not support import check for valid descriptor. The test
// case should be fixed once the support is added. (TODO)
// Import of object descriptor fails because LwMap does not allow importing the
// object on a channel for which the object was not exported. Note that LwSciBuf
// does not need to add a special check in the driver for object import since
// LwMap takes care of it.
TEST_F(TestLwSciBufSelwreBuffer, IlwalidImport1)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;

    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        //Peer A
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList1(list);

        // Import unreconciled attribute list from Peer B
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Reconcile Peer A & B attribute lists
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()},
            &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate object shareable between A & B
        auto bufObj = LwSciBufPeer::allocateBufObj(reconciledList.get(),
                        &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object to Peer B
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        // Wait until the peer B has imported buffer
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);

        // Setup IPC channel between Peer A & C
        // Note: Channel name is bit confusing. We are just using available
        // channel names in lwsciipc.cfg
        auto peer1 = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer1);
        peer1->SetUp("lwscibuf_ipc_B_C", *peer);

        // Export reconciled list export descriptor obtained from
        // reconciliation of A & B to Peer C. The descriptor is valid so we
        // should be able to send it over IPC.
        ASSERT_EQ(peer1->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export export descriptor for object allocated from reconciled
        // attribute list of A & B to Peer C. The descriptor is valid so we
        // should be able to send it over IPC.
        ASSERT_EQ(peer1->sendExportDesc(objDescBuf), LwSciError_Success);

    } else if ((pids[1] = fork()) == 0) {
        //Peer B
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        //setup IPC channel between Peer A & B
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        setupAttrList2(list);

        // Export unreconciled list to Peer A
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer A
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get descriptor
        auto objDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object from Peer A
        auto bufObj = peer->importBufObj(
            objDescBuf.get(), reconciledList.get(),
            LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Signal that object import has completed
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);

    } else if ((pids[2] = fork()) == 0) {
        //Peer C
        pid = 3;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);

        // Setup IPC channel between Peer A & C and check if Peer A is able to
        // export reconciled attribute list/object over this channel.
        // Note: Channel name is bit confusing. We are just using available
        // channel names in lwsciipc.cfg
        peer->SetUp("lwscibuf_ipc_C_B");

        {
            NEGATIVE_TEST();
            // Import reconciled list from Peer A
            auto reconciledListDescBuf = peer->recvBuf(&error);
            ASSERT_EQ(error, LwSciError_Success);
            auto reconciledList = peer->importReconciledList(
                                    reconciledListDescBuf,{}, &error);
            // TODO: Ideally, we should fail here since Peer C was not ilwolved
            // in reconciliation and thus import should have failed. Current
            // driver code does not support this. The assert should be checked
            // for LwSciError_NotPermitted when the driver code is fixed.
            // Keeping this under NEGATIVE_TEST() deliberately.
            ASSERT_EQ(error, LwSciError_Success);

            // Get descriptor
            auto objDescBuf =
                peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
            ASSERT_EQ(error, LwSciError_Success);

            // Import object from Peer A
            auto bufObj = peer->importBufObj(
                objDescBuf.get(), reconciledList.get(),
                LwSciBufAccessPerm_ReadWrite, &error);
            // This should fail because Peer A exported the buffer meant for
            // Peer B to Peer C. LwMap blocks this import. LwMap only allows
            // import if the export was done for that particular Peer.
            ASSERT_EQ(error, LwSciError_ResourceError);
        }

    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
