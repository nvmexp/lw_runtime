/*
 * lwscibuf_test_interprocess_tensor.cpp
 *
 * Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "lwscibuf_interprocess_test.h"

class TestLwSciBufInterProcessTensor : public LwSciBufInterProcessTest
{
};

TEST_F(TestLwSciBufInterProcessTensor, InterProcess)
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

        uint32_t dimcount = 5;
        //NCxHWx
        uint64_t sizes[] = { 2, 8, 64, 32, 1}; // [N, C, H, W, X]
        uint32_t alignment[] = { 1, 1, 32, 1, 1}; // align H for 32
        uint64_t baseAddrAlign = 512;
        // sentinel value for LwMedia private key
        uint32_t lwMediaPrivKeyVal = 10;

        LwSciBufInternalAttrKey lwMediaPrivKey;
        ASSERT_EQ(
            LwSciBufGetUMDPrivateKeyWithOffset(
                LwSciBufInternalAttrKey_LwMediaPrivateFirst, 1, &lwMediaPrivKey),
            LwSciError_Success);

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                 LwSciBufAccessPerm_ReadWrite);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_DataType,
                 LwSciDataType_Int16);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_BaseAddrAlign,
                 baseAddrAlign);

        SET_INTERNAL_ATTR(list.get(), lwMediaPrivKey, lwMediaPrivKeyVal);

        // Export unreconciled list to Peer B
        LwSciError error = LwSciError_Success;
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer B
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Import object allocated by Peer B
        auto bufObjDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto bufObj =
            peer->importBufObj(bufObjDescBuf.get(), reconciledList.get(),
                               LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Test access and alignment
        peer->testObject(bufObj.get(), GetPageSize());

        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_a_1");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        uint32_t dimcount = 5;
        //NCxHWx
        uint64_t sizes[] = { 2, 8, 64, 32, 1}; // [N, C, H, W, X]
        uint32_t alignment[] = { 1, 1, 32, 1, 1}; // align H for 32
        uint64_t baseAddrAlign = 512;
        // sentinel value for LwMedia private key
        uint32_t lwMediaPrivKeyVal = 10;
        LwSciBufInternalAttrKey lwMediaPrivKey;
        ASSERT_EQ(
            LwSciBufGetUMDPrivateKeyWithOffset(
                LwSciBufInternalAttrKey_LwMediaPrivateFirst, 1, &lwMediaPrivKey),
            LwSciError_Success);

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                 LwSciBufAccessPerm_ReadWrite);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_DataType,
                 LwSciDataType_Int16);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_BaseAddrAlign,
                 baseAddrAlign);

        SET_INTERNAL_ATTR(list.get(), lwMediaPrivKey, lwMediaPrivKeyVal);

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

        // Test access and alignment
        peer->testObject(bufObj.get(), GetPageSize());

        // Export object to Peer A
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        // Wait for Peer A to exit before releasing allocated object
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
