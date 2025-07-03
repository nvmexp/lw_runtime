/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"
#include "lwscibuf_interprocess_test.h"

class AttributeListValidate : public LwSciBufBasicTest
{
protected:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        listA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listA.get(), nullptr);

        listB = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listB.get(), nullptr);

        listC = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listC.get(), nullptr);
    };

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        listA.reset();
        listB.reset();
        listC.reset();
    };

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;
    std::shared_ptr<LwSciBufAttrListRec> listC;
};

// TODO: Lwrrently the only key that uses ArrayIntersectionPolicy is the
// LwSciBufInternalGeneralAttrKey_MemDomainArray key. Once other keys use this
// key, we can exercise more branches via the element-level APIs.

#if (LW_IS_SAFETY == 0) && !defined(__x86_64__)

TEST_F(AttributeListValidate, ArrayIntersectionPolicy_Negative)
{
    LwSciError error = LwSciError_Success;
    uint64_t size = 1024U * 4U;
    bool isReconciledListValid = false;

    LwSciBufMemDomain memdomain[] = {LwSciBufMemDomain_Sysmem};

    // List A wants Sysmem
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_INTERNAL_ATTR(listA.get(),
                      LwSciBufInternalGeneralAttrKey_MemDomainArray, memdomain);

    // List B also wants Sysmem
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_INTERNAL_ATTR(listB.get(),
                      LwSciBufInternalGeneralAttrKey_MemDomainArray, memdomain);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // Since List A and B has the same preferences, we choose the same value
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(
        reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray,
        memdomain);

    {
        NEGATIVE_TEST();

        // List C wants CVSRAM, which conflicts with the Sysmem
        LwSciBufMemDomain ilwalidMemdomain[] = {LwSciBufMemDomain_Cvsram};

        SET_ATTR(listC.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_RawBuffer);
        SET_ATTR(listC.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_INTERNAL_ATTR(listC.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          ilwalidMemdomain);

        // List C does not validate
        ASSERT_EQ(LwSciBufPeer::validateReconciled({listC.get()},
                                                   reconciledList.get(),
                                                   &isReconciledListValid),
                  LwSciError_ReconciliationFailed);
        ASSERT_FALSE(isReconciledListValid);

        // List C + A does not validate
        ASSERT_EQ(LwSciBufPeer::validateReconciled({listC.get(), listA.get()},
                                                   reconciledList.get(),
                                                   &isReconciledListValid),
                  LwSciError_ReconciliationFailed);
        ASSERT_FALSE(isReconciledListValid);

        // List C + B does not validate
        ASSERT_EQ(LwSciBufPeer::validateReconciled({listC.get(), listB.get()},
                                                   reconciledList.get(),
                                                   &isReconciledListValid),
                  LwSciError_ReconciliationFailed);
        ASSERT_FALSE(isReconciledListValid);
    }
}
#endif

#if (LW_IS_SAFETY == 0)
/** Empty arrays are ignored during validation. */
TEST_F(AttributeListValidate, ArrayIntersectionPolicy_Empty_Ignored)
{
    LwSciError error = LwSciError_Success;
    uint64_t size = 1024U * 4U;
    bool isReconciledListValid = false;

    LwSciBufMemDomain memdomain[] = {LwSciBufMemDomain_Sysmem};

    // List A wants Sysmem
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_INTERNAL_ATTR(listA.get(),
                      LwSciBufInternalGeneralAttrKey_MemDomainArray, memdomain);

    // List B has no preferencealso wants Sysmem
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // Since List B has no preference, we choose the value from List A
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(
        reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray,
        memdomain);

    // Verify that validating against empty arrays are ignored
    ASSERT_EQ(LwSciBufPeer::validateReconciled(
                  {listB.get()}, reconciledList.get(), &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);
}
#endif

class AttributeListValidateInterProc : public LwSciBufInterProcessTest
{
};

TEST_F(AttributeListValidateInterProc, NeedCpuAccessValidation)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        uint64_t size = 1024U * 4U;
        bool cpuAccess = true;

        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_0");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
            LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
            cpuAccess);

        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        uint64_t size = 1024U * 4U;
        bool cpuAccess = false;
        bool isReconciledListValid = false;

        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_1");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
            LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
            cpuAccess);

        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()},
                &error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
            LwSciBufGeneralAttrKey_NeedCpuAccess, false));

        ASSERT_EQ(LwSciBufPeer::validateReconciled(
            {list.get(), upstreamList.get()}, reconciledList.get(),
            &isReconciledListValid),
            LwSciError_Success);
        ASSERT_TRUE(isReconciledListValid);

        {
            NEGATIVE_TEST();

            auto ilwalidList = peer->createAttrList(&error);
            bool cpuAccess = true;

            SET_ATTR(ilwalidList.get(), LwSciBufGeneralAttrKey_Types,
                LwSciBufType_RawBuffer);
            SET_ATTR(ilwalidList.get(), LwSciBufRawBufferAttrKey_Size,
                size);
            SET_ATTR(ilwalidList.get(),
                LwSciBufGeneralAttrKey_NeedCpuAccess, cpuAccess);

            ASSERT_EQ(LwSciBufPeer::validateReconciled(
                {list.get(), ilwalidList.get()}, reconciledList.get(),
                &isReconciledListValid),
                LwSciError_ReconciliationFailed);
            ASSERT_FALSE(isReconciledListValid);
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

TEST_F(AttributeListValidateInterProc, ActualPermValidation)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        uint64_t size = 1024U * 4U;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_0");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
            LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
            perm);

        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        uint64_t size = 1024U * 4U;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        bool isReconciledListValid = false;

        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_1");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types,
            LwSciBufType_RawBuffer);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
            perm);

        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()},
                &error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
            LwSciBufGeneralAttrKey_ActualPerm, LwSciBufAccessPerm_ReadWrite));

        ASSERT_EQ(LwSciBufPeer::validateReconciled(
            {list.get(), upstreamList.get()}, reconciledList.get(),
            &isReconciledListValid),
            LwSciError_Success);
        ASSERT_TRUE(isReconciledListValid);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
