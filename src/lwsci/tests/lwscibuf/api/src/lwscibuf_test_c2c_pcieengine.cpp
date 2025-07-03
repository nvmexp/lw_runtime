/*
 * lwscibuf_test_interprocess_image.cpp
 *
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_interprocess_test.h"

class TestLwSciBufPCIeEngine : public LwSciBufInterProcessTest
{
protected :
    void SetPublicAttrs(std::shared_ptr<LwSciBufAttrListRec> list)
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
};

TEST_F(TestLwSciBufPCIeEngine, UnreconciledExportWithoutPCIeNonC2C)
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

        SetPublicAttrs(list);

        // Set internal attributes
        {
            LwSciBufHwEngine engine = {};
#if !defined(__x86_64__)
            engine.engNamespace = LwSciBufHwEngine_TegraNamespaceId;
#else
            engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
#endif
            LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA, &engine.rmModuleID);

            LwSciBufHwEngine engineArray[] = {engine};
            SET_INTERNAL_ATTR(list.get(), LwSciBufInternalGeneralAttrKey_EngineArray, engineArray);
        }

        LwSciError error = LwSciError_Success;
        // Export unreconciled list to Peer B
        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        //Test for PCIe engine after export
        {
            LwSciBufInternalAttrKeyValuePair intKeyValPair = {LwSciBufInternalGeneralAttrKey_EngineArray, nullptr, 0};
            error = LwSciBufAttrListGetInternalAttrs(list.get(), &intKeyValPair, 1);
            ASSERT_EQ(error, LwSciError_Success);
            const LwSciBufHwEngine *engineList = (const LwSciBufHwEngine*)intKeyValPair.value;
            uint64_t engineCount = intKeyValPair.len / sizeof(LwSciBufHwEngine);
            ASSERT_EQ(engineCount, 1);
            LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;
            error = LwSciBufHwEngGetNameFromId(engineList[0].rmModuleID, &engineName);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(engineName, LwSciBufHwEngName_DLA);
        }

        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        // Import reconciled list from Peer B
        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        //Test for PCIe engine after import in reconciled list
        {
            LwSciBufInternalAttrKeyValuePair intKeyValPair = {LwSciBufInternalGeneralAttrKey_EngineArray, nullptr, 0};
            error = LwSciBufAttrListGetInternalAttrs(reconciledList.get(), &intKeyValPair, 1);
            ASSERT_EQ(error, LwSciError_Success);
            const LwSciBufHwEngine *engineList = (const LwSciBufHwEngine*)intKeyValPair.value;
            uint64_t engineCount = intKeyValPair.len / sizeof(LwSciBufHwEngine);
            ASSERT_EQ(engineCount, 1);
            LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;
            error = LwSciBufHwEngGetNameFromId(engineList[0].rmModuleID, &engineName);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(engineName, LwSciBufHwEngName_DLA);
        }

        // Signal test complete
        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_a_1");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SetPublicAttrs(list);

        // Import from Peer A and reconcile
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            peer->attrListReconcile({list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        //Test for PCIe engine after import of peer unreconciled list
        {
            LwSciBufInternalAttrKeyValuePair intKeyValPair = {LwSciBufInternalGeneralAttrKey_EngineArray, nullptr, 0};
            error = LwSciBufAttrListGetInternalAttrs(upstreamList.get(), &intKeyValPair, 1);
            ASSERT_EQ(error, LwSciError_Success);
            const LwSciBufHwEngine *engineList = (const LwSciBufHwEngine*)intKeyValPair.value;
            uint64_t engineCount = intKeyValPair.len / sizeof(LwSciBufHwEngine);
            ASSERT_EQ(engineCount, 1);
            LwSciBufHwEngName engineName = LwSciBufHwEngName_Num;
            error = LwSciBufHwEngGetNameFromId(engineList[0].rmModuleID, &engineName);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(engineName, LwSciBufHwEngName_DLA);
        }

        // Export attr list back to Peer A
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

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
