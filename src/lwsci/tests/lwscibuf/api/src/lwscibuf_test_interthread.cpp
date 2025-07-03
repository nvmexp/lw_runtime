/*
 * lwscibuf_test_interthread.cpp
 *
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <thread>

#include "lwscibuf_basic_test.h"
#include "lwscibuf_ipc_peer.h"

#define UMDSTRING "LwSciBuf"

void runThreadA()
{
    auto peer = std::make_shared<LwSciBufIpcPeer>();
    peer->SetUp("itc_test_0");

    LwSciError error = LwSciError_Success;

    auto list = peer->createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t rawSize = 1024U;
        uint64_t align = 1U;
        bool cpuAccessFlag = true;

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, rawSize);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, align);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);
    }

    // Export unreconciled list to Peer B
    auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

    // Import reconciled list from Peer B
    auto reconciledListDescBuf = peer->recvBuf(&error);
    ASSERT_EQ(error, LwSciError_Success);
    auto reconciledList =
        peer->importReconciledList(reconciledListDescBuf, {list.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Import object allocated by Peer B
    auto bufObjDescBuf =
        peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
    ASSERT_EQ(error, LwSciError_Success);
    auto bufObj = peer->importBufObj(bufObjDescBuf.get(), reconciledList.get(),
                                     LwSciBufAccessPerm_ReadWrite, &error);
    ASSERT_EQ(error, LwSciError_Success);

    peer->testObject(bufObj.get(), GetPageSize());

    ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
}

void runThreadB()
{
    auto peer = std::make_shared<LwSciBufIpcPeer>();
    peer->SetUp("itc_test_1");

    LwSciError error = LwSciError_Success;

    auto list = peer->createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t align = 1U;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        bool cpuAccessFlag = true;

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, align);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);

        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};
        std::vector<LwSciBufInternalAttrKeyValuePair> rawBufIntAttrs = {
            {LwSciBufInternalGeneralAttrKey_MemDomainArray, memDomain,
             sizeof(LwSciBufMemDomain)},
            {LwSciBufInternalAttrKey_LwMediaPrivateFirst, UMDSTRING,
             strlen(UMDSTRING) + 1},
        };
        ASSERT_EQ(LwSciBufAttrListSetInternalAttrs(
                      list.get(), rawBufIntAttrs.data(), rawBufIntAttrs.size()),
                  LwSciError_Success);
    }

    // Import from Peer A and reconcile
    auto upstreamListDescBuf = peer->recvBuf(&error);
    ASSERT_EQ(error, LwSciError_Success);
    auto upstreamList =
        peer->importUnreconciledList(upstreamListDescBuf, &error);
    ASSERT_EQ(error, LwSciError_Success);

    auto reconciledList =
        peer->attrListReconcile({list.get(), upstreamList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Export attr list back to Peer A
    auto reconciledListDescBuf =
        peer->exportReconciledList(reconciledList.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

    // Allocate object
    auto bufObj = peer->allocateBufObj(reconciledList.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);

    peer->testObject(bufObj.get(), GetPageSize());

    // Export object to Peer A
    auto objDescBuf =
        peer->exportBufObj(bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

    ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
}

class TestLwSciBufInterThread : public LwSciBufBasicTest
{
};

TEST_F(TestLwSciBufInterThread, InterThread)
{
    std::thread threadA(runThreadA);
    std::thread threadB(runThreadB);

    threadA.join();
    threadB.join();
}
