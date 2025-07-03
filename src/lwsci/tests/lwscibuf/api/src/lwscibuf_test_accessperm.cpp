/*
 * lwscibuf_test_accessperm.cpp
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

class TestLwSciBufAccessPermission : public LwSciBufInterProcessTest
{
public:
    void setupAttrListAlign(std::shared_ptr<LwSciBufAttrListRec> list)
    {
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align,
                 (uint64_t)(4U * 1024U));
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);

        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};
        SET_INTERNAL_ATTR(list.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
    }

    void setupAttrListSizeAndAlign(std::shared_ptr<LwSciBufAttrListRec> list)
    {
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size,
                 (uint64_t)(128 * 1024U));
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align,
                 (uint64_t)(4U * 1024U));
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
    }
};

/* Test topology
procA
|
|
procB
|
|
procC
*/
TEST_F(TestLwSciBufAccessPermission, AutomaticPermissions)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;
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
        auto upstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(upstreamPeer);
        upstreamPeer->SetUp("lwscibuf_ipc_A_B");

        auto list = upstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            setupAttrListAlign(list);

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_ReadWrite);
        }

        /* Import unreconciled list from Peer 2 */
        auto upstreamListDescBuf = upstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            upstreamPeer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Reconcile */
        auto bufObj = LwSciBufPeer::reconcileAndAllocate(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Export Reconciled Attribute List to Peer 2 */
        auto attrListAndObjDesc = upstreamPeer->exportAttrListAndObj(
            bufObj.get(), LwSciBufAccessPerm_Auto, &error);
        ASSERT_EQ(upstreamPeer->sendBuf(attrListAndObjDesc),
                  LwSciError_Success);

        /* Test Access Permissions */
        LwSciBufInterProcessTest::testAccessPermissions(
            upstreamPeer, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite,
            nullptr, 0, sampleData1, sizeof(sampleData1));

        ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto downstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(downstreamPeer);
        downstreamPeer->SetUp("lwscibuf_ipc_B_A");

        auto upstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(upstreamPeer);
        upstreamPeer->SetUp("lwscibuf_ipc_B_C",
                            *downstreamPeer); // Share the same module instance

        auto list = downstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
                     LwSciBufType_RawBuffer);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size,
                     (uint64_t)(128 * 1024U));
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align,
                     (uint64_t)(8U * 1024U)); // 8U instead of 4U
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_Readonly);
        }

        /* Import unreconciled list from Peer 3 */
        auto upstreamListDescBuf = upstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            upstreamPeer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Export unreconciled list to Peer 1 */
        auto listDescBuf = downstreamPeer->exportUnreconciledList(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(downstreamPeer->sendBuf(listDescBuf), LwSciError_Success);

        /* Import Reconciled Attribute List from Peer 1 */
        auto downstreamAttrListAndObjDescBuf = downstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto bufObj = downstreamPeer->importAttrListAndObj(
            downstreamAttrListAndObjDescBuf, {list.get()},
            LwSciBufAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Export Reconciled Attribute List to Peer 3 */
        auto upstreamAttrListAndObjDescBuf = upstreamPeer->exportAttrListAndObj(
            bufObj.get(), LwSciBufAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(upstreamPeer->sendBuf(upstreamAttrListAndObjDescBuf),
                  LwSciError_Success);

        /* Test Access Permissions */
        LwSciBufInterProcessTest::testAccessPermissions(
            upstreamPeer, downstreamPeer, bufObj, LwSciBufAccessPerm_ReadWrite,
            sampleData1, sizeof(sampleData1), sampleData, sizeof(sampleData));

        ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
        ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
    } else if ((pids[2] = fork()) == 0) {
        pid = 3;
        auto downstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(downstreamPeer);
        downstreamPeer->SetUp("lwscibuf_ipc_C_B");

        auto list = downstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            setupAttrListSizeAndAlign(list);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_ReadWrite);

            LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};
            SET_INTERNAL_ATTR(list.get(),
                              LwSciBufInternalGeneralAttrKey_MemDomainArray,
                              memDomain);
        }

        /* Export Unreconciled Attribute List to Peer 1 */
        auto listDescBuf =
            downstreamPeer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(downstreamPeer->sendBuf(listDescBuf), LwSciError_Success);

        /* Import from Peer 1 */
        auto downstreamAttrListAndObjDescBuf = downstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto bufObj = downstreamPeer->importAttrListAndObj(
            downstreamAttrListAndObjDescBuf, {list.get()},
            LwSciBufAccessPerm_Auto, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Test Access Permissions */
        LwSciBufInterProcessTest::testAccessPermissions(
            nullptr, downstreamPeer, bufObj, LwSciBufAccessPerm_ReadWrite,
            sampleData, sizeof(sampleData), sampleData1, sizeof(sampleData1));

        ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

TEST_F(TestLwSciBufAccessPermission, NonAutomaticPermissions)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 3;
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
        auto upstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(upstreamPeer);
        upstreamPeer->SetUp("lwscibuf_ipc_A_B");

        auto list = upstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            setupAttrListAlign(list);

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_ReadWrite);
        }

        auto upstreamListDescBuf = upstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            upstreamPeer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Reconcile and allocate LwSciBufObj */
        auto bufObj = LwSciBufPeer::reconcileAndAllocate(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        LwSciBufAttrList reconciledList = nullptr;
        ASSERT_EQ(LwSciBufObjGetAttrList(bufObj.get(), &reconciledList),
                  LwSciError_Success);

        /* Export Reconciled Attribute List to Peer 2 */
        auto reconciledListDescBuf =
            upstreamPeer->exportReconciledList(reconciledList, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(upstreamPeer->sendBuf(reconciledListDescBuf),
                  LwSciError_Success);

        {
            /*****************NONAUTO TEST EXPORT1**********************/
            /* Non-auto export test1: EXP.RLIST.PERM: RW, EXP.API.PERM: RW
             * Note: EXP.RLIST.PERM is auto computed by LwSciBuf.
             */
            /* Export LwSciBufObj to Peer 2 */
            auto bufObjDescBuf = upstreamPeer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_EQ(upstreamPeer->sendExportDesc(bufObjDescBuf),
                      LwSciError_Success);

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);

            LwSciBufInterProcessTest::testAccessPermissions(
                upstreamPeer, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite,
                nullptr, 0, sampleData, sizeof(sampleData));
            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
        }

        {
            /*****************NONAUTO TEST EXPORT2**********************/
            /* Test1 complete. Re-export with other permissions
             * Non-auto export test2: EXP.RLIST.PERM: RW, EXP.API.PERM: RW
             */
            /* Export LwSciBufObj to Peer 3 */
            auto bufObjDescBuf = upstreamPeer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_EQ(upstreamPeer->sendExportDesc(bufObjDescBuf),
                      LwSciError_Success);

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);

            LwSciBufInterProcessTest::testAccessPermissions(
                upstreamPeer, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite,
                nullptr, 0, sampleData, sizeof(sampleData));
            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
        }

        {
            /*****************NONAUTO TEST EXPORT3**********************/
            /* Test2 complete. Re-test for other combinations
             * Non-auto export test2: EXP.RLIST.PERM: RW, EXP.API.PERM: RW
             */
            /* Export LwSciBufObj to Peer 3 */
            auto bufObjDescBuf = upstreamPeer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_EQ(upstreamPeer->sendExportDesc(bufObjDescBuf),
                      LwSciError_Success);

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);

            LwSciBufInterProcessTest::testAccessPermissions(
                upstreamPeer, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite,
                nullptr, 0, sampleData, sizeof(sampleData));
            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
        }
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto downstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(downstreamPeer);
        downstreamPeer->SetUp("lwscibuf_ipc_B_A");

        auto upstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(upstreamPeer);
        upstreamPeer->SetUp("lwscibuf_ipc_B_C",
                            *downstreamPeer); // Share the same module instance

        auto list = upstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
                     LwSciBufType_RawBuffer);
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size,
                     (uint64_t)(128 * 1024U));
            SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align,
                     (uint64_t)(8U * 1024U)); // 8U instead of 4U
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_ReadWrite);
        }

        /* Import Unreconciled Attribute List from Peer 3 */
        auto upstreamListDescBuf = upstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            upstreamPeer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Export Unreconciled Attribute List to Peer 1 */
        auto listDescBuf = downstreamPeer->exportUnreconciledList(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(downstreamPeer->sendBuf(listDescBuf), LwSciError_Success);

        /* Import Reconciled Attribute List from Peer 1 */
        auto reconciledListDescBuf = downstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = downstreamPeer->importReconciledList(
            reconciledListDescBuf, {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Export Reconciled Attribute List to Peer 3 */
        {
            auto reconciledListDescBuf = upstreamPeer->exportReconciledList(
                reconciledList.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(upstreamPeer->sendBuf(reconciledListDescBuf),
                      LwSciError_Success);
        }

        {
            /*****************NONAUTO TEST EXPORT1**********************/
            /* Non-auto Import test1: EXP.OBJ.PERM: RW, IMP.API.PERM: RO */
            /* Import LwSciBufObj from Peer 1 */
            auto bufObjDescBuf =
                downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                    &error);
            ASSERT_EQ(error, LwSciError_Success);
            auto bufObj = downstreamPeer->importBufObj(
                bufObjDescBuf.get(), {reconciledList.get()},
                LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            /* Non-auto Export test2: EXP.RLIST.PERM: RO, EXP.API.PERM: RO */
            /* Export LwSciBufObj to Peer 3 */
            auto bufObjDescBufReadonly = upstreamPeer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(upstreamPeer->sendExportDesc(bufObjDescBufReadonly),
                      LwSciError_Success);

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

            /* Test Access Permissions */
            LwSciBufInterProcessTest::testAccessPermissions(
                upstreamPeer, downstreamPeer, bufObj,
                LwSciBufAccessPerm_ReadWrite, sampleData, sizeof(sampleData),
                sampleData1, sizeof(sampleData1));

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
        }

        {
            /*****************NONAUTO TEST EXPORT2**********************/
            /* Re-export and test other combinations */
            /* Non-auto Import test2: EXP.OBJ.PERM: RW, IMP.API.PERM: RW */
            /* Import LwSciBufObj from Peer 1 */
            auto bufObjDescBuf =
                downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                    &error);
            ASSERT_EQ(error, LwSciError_Success);
            auto bufObj = downstreamPeer->importBufObj(
                bufObjDescBuf.get(), {reconciledList.get()},
                LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            /* Non-auto Export Retest: EXP.RLIST.PERM: RO, EXP.API.PERM: RO */
            /* Export LwSciBufObj to Peer 3 */
            auto bufObjDescBufReadonly = upstreamPeer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(upstreamPeer->sendExportDesc(bufObjDescBufReadonly),
                      LwSciError_Success);

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

            /* Test Access Permissions */
            LwSciBufInterProcessTest::testAccessPermissions(
                upstreamPeer, downstreamPeer, bufObj,
                LwSciBufAccessPerm_ReadWrite, sampleData, sizeof(sampleData),
                sampleData1, sizeof(sampleData1));

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
        }

        {
            /*****************NONAUTO TEST EXPORT3**********************/
            /* Re-export and test other combinations */
            /* Import LwSciBufObj from Peer 1 */
            /* Non-auto Import test2: EXP.OBJ.PERM: RW, IMP.API.PERM: RW */
            auto bufObjDescBuf =
                downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                    &error);
            ASSERT_EQ(error, LwSciError_Success);
            auto bufObj = downstreamPeer->importBufObj(
                bufObjDescBuf.get(), {reconciledList.get()},
                LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            /* Non-auto Export test2: EXP.RLIST.PERM: RO, EXP.API.PERM: RW */
            /* Export LwSciBufObj to Peer 3 */
            auto bufObjDescBufReadWrite = upstreamPeer->exportBufObj(
                bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(upstreamPeer->sendExportDesc(bufObjDescBufReadWrite),
                      LwSciError_Success);

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

            /* Test Access Permissions */
            LwSciBufInterProcessTest::testAccessPermissions(
                upstreamPeer, downstreamPeer, bufObj,
                LwSciBufAccessPerm_ReadWrite, sampleData, sizeof(sampleData),
                sampleData1, sizeof(sampleData1));

            ASSERT_EQ(upstreamPeer->waitComplete(), LwSciError_Success);
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
        }
    } else if ((pids[2] = fork()) == 0) {
        pid = 3;

        auto downstreamPeer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(downstreamPeer);
        downstreamPeer->SetUp("lwscibuf_ipc_C_B");

        auto list = downstreamPeer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            setupAttrListSizeAndAlign(list);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_Readonly);

            LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};
            SET_INTERNAL_ATTR(list.get(),
                              LwSciBufInternalGeneralAttrKey_MemDomainArray,
                              memDomain);
        }

        /* Export Unreconciled Attribute List to Peer 2 */
        auto listDescBuf =
            downstreamPeer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(downstreamPeer->sendBuf(listDescBuf), LwSciError_Success);

        /* Import Reconciled Attribute List from Peer 2 */
        auto reconciledListDescBuf = downstreamPeer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = downstreamPeer->importReconciledList(
            reconciledListDescBuf, {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            auto bufObjDescBuf =
                downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                    &error);
            ASSERT_EQ(error, LwSciError_Success);

            /*****************NONAUTO TEST EXPORT1**********************/
            /* Non-auto Import test1: EXP.OBJ.PERM: RO, IMP.API.PERM: RW */
            {
                NEGATIVE_TEST();
                auto bufObj = downstreamPeer->importBufObj(
                    bufObjDescBuf.get(), reconciledList.get(),
                    LwSciBufAccessPerm_ReadWrite, &error);
                ASSERT_EQ(error, LwSciError_AccessDenied);
            }
            /* Then actually import the buffer to decrement refcount */
            auto bufObj = downstreamPeer->importBufObj(
                bufObjDescBuf.get(), reconciledList.get(),
                LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            /* Signal Peer that we are done */
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

            /* Extra waits/signals since we aren't doing a test */
            ASSERT_EQ(downstreamPeer->waitComplete(), LwSciError_Success);
            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
        }

        {
            auto bufObjDescBuf =
                downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                    &error);
            ASSERT_EQ(error, LwSciError_Success);

            /*****************NONAUTO TEST EXPORT2**********************/
            /* Non-auto Import test2: EXP.OBJ.PERM: RO, IMP.API.PERM: RO */
            auto bufObj = downstreamPeer->importBufObj(
                bufObjDescBuf.get(), reconciledList.get(),
                LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

            /* Test Access Permissions */
            LwSciBufInterProcessTest::testAccessPermissions(
                nullptr, downstreamPeer, bufObj, LwSciBufAccessPerm_Readonly,
                sampleData1, sizeof(sampleData1), nullptr, 0);

            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
        }

        {
            auto bufObjDescBuf =
                downstreamPeer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(
                    &error);
            ASSERT_EQ(error, LwSciError_Success);

            /*****************NONAUTO TEST EXPORT3**********************/
            /* Non-auto Import test3: EXP.OBJ.PERM: RW, IMP.API.PERM: RO */
            auto bufObj = downstreamPeer->importBufObj(
                bufObjDescBuf.get(), reconciledList.get(),
                LwSciBufAccessPerm_Readonly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);

            /* Test Access Permissions */
            LwSciBufInterProcessTest::testAccessPermissions(
                nullptr, downstreamPeer, bufObj, LwSciBufAccessPerm_ReadWrite,
                sampleData1, sizeof(sampleData1), nullptr, 0);

            ASSERT_EQ(downstreamPeer->signalComplete(), LwSciError_Success);
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

TEST_F(TestLwSciBufAccessPermission, ReducedPermissions)
{
    const uint8_t sampleData[] = {
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0,
        0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0x11, 0x21, 0x31, 0x41, 0x51,
        0x61, 0x71, 0x81, 0x91, 0xA1, 0xB1, 0xC1, 0xD1, 0xE1, 0xF1,
    };

    LwSciError error = LwSciError_Success;

    auto peer = std::make_shared<LwSciBufPeer>();
    peer->SetUp();

    auto list = peer->createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        setupAttrListSizeAndAlign(list);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                 LwSciBufAccessPerm_ReadWrite);

        /* Set-up internal attr list */
        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};
        SET_INTERNAL_ATTR(list.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
    }

    auto bufObj = LwSciBufPeer::reconcileAndAllocate({list.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    /* Test Access Permissions */
    LwSciBufInterProcessTest::testAccessPermissions(
        nullptr, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite, nullptr, 0,
        sampleData, sizeof(sampleData));

    auto reducedObj = LwSciBufPeer::duplicateBufObjWithReducedPerm(
        bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
    ASSERT_EQ(error, LwSciError_Success);

    /* Test Access Permissions */
    LwSciBufInterProcessTest::testAccessPermissions(
        nullptr, nullptr, bufObj, LwSciBufAccessPerm_ReadWrite, sampleData,
        sizeof(sampleData), nullptr, 0);
}
