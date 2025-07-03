/*
 * lwscibuf_test_interprocess_image.cpp
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

class TestLwSciBufInterprocessImage : public LwSciBufInterProcessTest
{
};

TEST_F(TestLwSciBufInterprocessImage, InterProcess)
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

        // Set attributes
        {
            LwSciBufType bufType = LwSciBufType_Image;
            LwSciBufAttrValImageLayoutType layout =
                LwSciBufImage_PitchLinearType;
            uint64_t lrpad = 0, tbpad = 0;
            bool cpuaccess_flag = true;
            bool vpr = false;
            int32_t planecount = 1;

            LwSciBufAttrValColorFmt planecolorfmts[] = {LwSciColor_A8B8G8R8};
            LwSciBufAttrValColorStd planecolorstds[] = {LwSciColorStd_SRGB};
            LwSciBufAttrValImageScanType planescantype[] = {
                LwSciBufScan_ProgressiveType};

            int32_t plane_widths[] = {1920};
            int32_t plane_heights[] = {1080};

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     cpuaccess_flag);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_Layout, layout);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_TopPadding, tbpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_BottomPadding, tbpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_LeftPadding, lrpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_RightPadding, lrpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_VprFlag, vpr);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneCount, planecount);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                     planecolorfmts);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneColorStd,
                     planecolorstds);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneWidth, plane_widths);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneHeight,
                     plane_heights);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_ScanType, planescantype);
        }

        LwSciError error = LwSciError_Success;
        // Export unreconciled list to Peer B
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

        // Test object
        LwSciBufPeer::testObject(bufObj.get(), GetPageSize());

        // Signal test complete
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_a_1");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            LwSciBufType bufType = LwSciBufType_Image;
            LwSciBufAttrValImageLayoutType layout =
                LwSciBufImage_PitchLinearType;
            uint64_t lrpad = 0, tbpad = 0;
            bool cpuaccess_flag = true;
            bool vpr = false;
            int32_t planecount = 1;

            LwSciBufAttrValColorFmt planecolorfmts[] = {LwSciColor_A8B8G8R8};
            LwSciBufAttrValColorStd planecolorstds[] = {LwSciColorStd_SRGB};
            LwSciBufAttrValImageScanType planescantype[] = {
                LwSciBufScan_ProgressiveType};

            int32_t plane_widths[] = {1920};
            int32_t plane_heights[] = {1080};

            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     cpuaccess_flag);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_Layout, layout);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_TopPadding, tbpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_BottomPadding, tbpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_LeftPadding, lrpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_RightPadding, lrpad);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_VprFlag, vpr);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneCount, planecount);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                     planecolorfmts);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneColorStd,
                     planecolorstds);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneWidth, plane_widths);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_PlaneHeight,
                     plane_heights);
            SET_ATTR(list.get(), LwSciBufImageAttrKey_ScanType, planescantype);
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

        // Test access and alignment
        LwSciBufPeer::testObject(bufObj.get(), GetPageSize());

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
