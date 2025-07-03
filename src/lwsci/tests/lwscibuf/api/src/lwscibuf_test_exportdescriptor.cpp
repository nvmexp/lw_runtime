/*
 * lwscibuf_test_exportdescriptor.cpp
 *
 * Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <string.h>
#include <iostream>

#include "lwscibuf_interprocess_test.h"

struct LwSciBufRawBufAttrs {
    bool setSize;
    bool cpuFlag;
    bool setMemDomain;
    bool enableCpuCache;
    bool cacheCoherency;
    uint64_t size;
    uint64_t alignment;
    LwSciBufType bufType;
    LwSciBufMemDomain memDomain;
    LwSciBufAttrValAccessPerm perm;
    LwSciBufAttrValAccessPerm actualPerm;
};

struct LwSciBufImageBufAttrs {
    LwSciBufType bufType;
    LwSciBufAttrValImageLayoutType layout;
    uint64_t lrpad;
    uint64_t tbpad;
    bool cpuAccess;
    bool vpr;
    uint32_t planecount;
    uint64_t imageCount;
    LwSciBufAttrValColorFmt planecolorfmts[1];
    LwSciBufAttrValColorStd planecolorstds[1];
    LwSciBufAttrValImageScanType planescantype[1];
    int32_t plane_widths[1];
    int32_t plane_heights[1];
    uint64_t imageAlignment;
    LwSciBufAttrValAccessPerm perm;
};

// We use these instead of the verifyAttr() helpers from Peer since they don't
// allow us to easily check length.
template<typename T>
static void validateAttrAccess(LwSciBufAttrList reconciledList,
                        LwSciBufAttrKey key, T expected, size_t len, bool accessible)
{
    LwSciBufAttrKeyValuePair pair = { key, nullptr, 0 };

    ASSERT_EQ(LwSciError_Success,
            LwSciBufAttrListGetAttrs(reconciledList, &pair, 1));

    if (accessible) {
        ASSERT_EQ(expected, *((const T*)pair.value));
        ASSERT_EQ(len, pair.len);
    } else {
        ASSERT_EQ(NULL, *((const T*)pair.value));
        ASSERT_EQ(0, pair.len);
    }
}

template<typename T>
static void validateInternalAttrAccess(LwSciBufAttrList reconciledList,
                        LwSciBufInternalAttrKey key, T expected, size_t len, bool accessible)
{
    LwSciBufInternalAttrKeyValuePair pair = { key, nullptr, 0 };
    ASSERT_EQ(LwSciError_Success,
            LwSciBufAttrListGetInternalAttrs(reconciledList, &pair, 1));

    if (accessible) {
        ASSERT_EQ(expected, *((const T*)pair.value));
        ASSERT_EQ(len, pair.len);
    } else {
        ASSERT_EQ(NULL, *((const T*)pair.value));
        ASSERT_EQ(0, pair.len);
    }
}

template<typename T>
static void validateArrayAttrAccess(LwSciBufAttrList reconciledList,
        LwSciBufAttrKey key, T expected[], size_t len, uint32_t elements)
{
    LwSciBufAttrKeyValuePair pair = { key, nullptr, 0 };

    ASSERT_EQ(LwSciError_Success,
            LwSciBufAttrListGetAttrs(reconciledList, &pair, 1));

    for (uint32_t i = 0; i < elements; ++i) {
        ASSERT_EQ(expected[i], ((const T*)pair.value)[i]);
    }
    ASSERT_EQ(len, pair.len);
}



class TestLwSciBufExportDesc : public LwSciBufInterProcessTest
{
public:
    void SetUpImageAttrList(std::shared_ptr<LwSciBufAttrListRec> attrList, LwSciBufImageBufAttrs attrs)
    {
        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, attrs.bufType);
        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, attrs.cpuAccess);
        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, attrs.perm);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_Layout, attrs.layout);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_TopPadding, attrs.tbpad);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_BottomPadding, attrs.tbpad);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_LeftPadding, attrs.lrpad);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_RightPadding, attrs.lrpad);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_VprFlag, attrs.vpr);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneCount, attrs.planecount);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_ImageCount, attrs.imageCount);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneColorFormat, attrs.planecolorfmts);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneColorStd, attrs.planecolorstds);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneWidth, attrs.plane_widths);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneHeight, attrs.plane_heights);
        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_ScanType, attrs.planescantype);
    }
    void validateGeneralAttributeAccess(LwSciBufAttrList reconciledList,
                        LwSciBufRawBufAttrs requested)
    {
        // Validate accessible general attributes
        validateAttrAccess(reconciledList, LwSciBufGeneralAttrKey_Types, requested.bufType,
                sizeof(requested.bufType), true);
        validateInternalAttrAccess(reconciledList, LwSciBufInternalGeneralAttrKey_MemDomainArray,
                requested.memDomain, sizeof(requested.memDomain), true);
        validateAttrAccess(reconciledList, LwSciBufGeneralAttrKey_NeedCpuAccess, requested.cpuFlag,
                sizeof(requested.cpuFlag), true);
        validateAttrAccess(reconciledList, LwSciBufGeneralAttrKey_EnableCpuCache,
                requested.enableCpuCache, sizeof(requested.enableCpuCache), false);
        validateAttrAccess(reconciledList, LwSciBufGeneralAttrKey_CpuNeedSwCacheCoherency,
                requested.cacheCoherency, sizeof(requested.cacheCoherency), true);
        validateAttrAccess(reconciledList, LwSciBufGeneralAttrKey_ActualPerm,
                requested.actualPerm, sizeof(requested.actualPerm), true);

        // GpuId is not accessible
        LwSciRmGpuId uuIdVidMem = {.bytes = { 0xab, 0xcd}}; // dummy GpuId
        validateAttrAccess(reconciledList, LwSciBufGeneralAttrKey_GpuId, &uuIdVidMem,
                sizeof(uuIdVidMem), false);

        // EngineArray is not accessible
        LwSciBufHwEngine engine; // dummy engine array
        LwSciBufHwEngine engineArray[] = { engine };
        validateInternalAttrAccess(reconciledList,LwSciBufInternalGeneralAttrKey_EngineArray,
                engineArray, sizeof(engineArray), false);
    }

    void validateRawBufferAttributeAccess(LwSciBufAttrList reconciledList,
        LwSciBufRawBufAttrs requested)
    {
        // Validate accessible raw buffer attributes
        validateAttrAccess(reconciledList, LwSciBufRawBufferAttrKey_Size,
                requested.size, sizeof(requested.size), true);
        validateAttrAccess(reconciledList, LwSciBufRawBufferAttrKey_Align, requested.alignment,
                sizeof(requested.alignment), true);
    }

    void validateImageAttributeAccess(LwSciBufAttrList reconciledList,
        LwSciBufImageBufAttrs required)
    {
        // Non array type image attrs which are expected to be set but accessible
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_ImageCount,
                required.imageCount , sizeof(required.imageCount), true);
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_TopPadding,
                required.tbpad , sizeof(required.tbpad), true);
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_BottomPadding,
                required.tbpad , sizeof(required.tbpad), true);
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_LeftPadding,
                required.lrpad , sizeof(required.lrpad), true);
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_RightPadding,
                required.lrpad , sizeof(required.lrpad), true);
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_VprFlag,
                required.vpr , sizeof(required.vpr), true);
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_Layout,
                required.layout , sizeof(required.layout), true);
        validateAttrAccess(reconciledList, LwSciBufImageAttrKey_PlaneCount,
                required.planecount , sizeof(required.planecount), true);

        // array type image attrs which are expected to be set and accessible
        validateArrayAttrAccess(reconciledList, LwSciBufImageAttrKey_ScanType,
                required.planescantype , sizeof(required.planescantype),
                sizeof(required.planescantype)/sizeof(required.planescantype[0]));
        validateArrayAttrAccess(reconciledList, LwSciBufImageAttrKey_PlaneColorFormat,
                required.planecolorfmts, sizeof(required.planecolorfmts),
                sizeof(required.planecolorfmts)/sizeof(required.planecolorfmts[0]));
        validateArrayAttrAccess(reconciledList, LwSciBufImageAttrKey_PlaneColorStd,
                required.planecolorstds , sizeof(required.planecolorstds),
                sizeof(required.planecolorstds)/sizeof(required.planecolorstds[0]));
        validateArrayAttrAccess(reconciledList, LwSciBufImageAttrKey_PlaneWidth,
                required.plane_widths , sizeof(required.plane_widths),
                sizeof(required.plane_widths)/sizeof(required.plane_widths[0]));
        validateArrayAttrAccess(reconciledList, LwSciBufImageAttrKey_PlaneHeight,
                required.plane_heights , sizeof(required.plane_heights),
                sizeof(required.plane_heights)/sizeof(required.plane_heights[0]));
    }
};

TEST_F(TestLwSciBufExportDesc, ValidateGeneralAndRawBufAttr)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    LwSciBufRawBufAttrs attrs;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        attrs.setSize = false;
        attrs.cpuFlag = true;
        attrs.setMemDomain = true;
        attrs.alignment = (4U * 1024U);
        attrs.perm = LwSciBufAccessPerm_ReadWrite;
        attrs.size = (128 * 1024U);
        attrs.memDomain = LwSciBufMemDomain_Sysmem;
        attrs.bufType = LwSciBufType_RawBuffer;
        attrs.enableCpuCache = false;
        attrs.cacheCoherency = false;

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, attrs.bufType);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, attrs.alignment);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, attrs.cpuFlag);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm, attrs.perm);

        SET_INTERNAL_ATTR(list.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, attrs.memDomain);

        // Import Unreconciled Attribute List
        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            peer->attrListReconcile({list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Allocate Object
        auto bufObj = peer->allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Export attr list back to Peer B
        auto reconciledListDescBuf =
            peer->exportReconciledList(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDescBuf), LwSciError_Success);

        // Export object back to Peer B
        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        // Wait for Peer B to exit before releasing allocated object
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        attrs.setSize = true;
        attrs.cpuFlag = true;
        attrs.setMemDomain = false;
        attrs.alignment = (8U * 1024U);
        attrs.perm = LwSciBufAccessPerm_Readonly;
        attrs.size = (128 * 1024U);
        attrs.bufType = LwSciBufType_RawBuffer;
        attrs.memDomain = LwSciBufMemDomain_Sysmem;
        attrs.actualPerm = LwSciBufAccessPerm_Readonly;
        attrs.enableCpuCache = false;
        attrs.cacheCoherency = false;

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, attrs.bufType);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Size, attrs.size);
        SET_ATTR(list.get(), LwSciBufRawBufferAttrKey_Align, attrs.alignment);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, attrs.cpuFlag);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm, attrs.perm);

        // Export unreconciled list to Peer A
        LwSciError error = LwSciError_Success;
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
        auto bufObj =
            peer->importBufObj(bufObjDescBuf.get(), reconciledList.get(),
                               LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        validateGeneralAttributeAccess(reconciledList.get(), attrs);
        validateRawBufferAttributeAccess(reconciledList.get(), attrs);

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

TEST_F(TestLwSciBufExportDesc, ValidateImageAttr)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    LwSciBufImageBufAttrs attrs;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_A_B");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        attrs.bufType = LwSciBufType_Image;
        attrs.layout = LwSciBufImage_PitchLinearType;
        attrs.lrpad = 0;
        attrs.tbpad = 0;
        attrs.perm = LwSciBufAccessPerm_Readonly;
        attrs.cpuAccess = true;
        attrs.vpr = false;
        attrs.planecount = 1;
        attrs.imageCount = 1;
        attrs.planecolorfmts[0] = { LwSciColor_A8B8G8R8 };
        attrs.planecolorstds[0] = { LwSciColorStd_SRGB };
        attrs.planescantype[0] = { LwSciBufScan_ProgressiveType};
        attrs.plane_widths[0] = { 1920 };
        attrs.plane_heights[0] = { 1080 };

        SetUpImageAttrList(list, attrs);

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

        // Wait for Peer B to exit before releasing allocated object
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_B_A");

        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);


        attrs.bufType = LwSciBufType_Image;
        attrs.perm = LwSciBufAccessPerm_ReadWrite;
        attrs.layout = LwSciBufImage_PitchLinearType;
        attrs.lrpad = 0;
        attrs.tbpad = 0;
        attrs.cpuAccess = true;
        attrs.vpr = false;
        attrs.planecount = 1;
        attrs.imageCount = 1;
        attrs.planecolorfmts[0] = { LwSciColor_A8B8G8R8 };
        attrs.planecolorstds[0] = { LwSciColorStd_SRGB };
        attrs.planescantype[0] = { LwSciBufScan_ProgressiveType};
        attrs.plane_widths[0] = { 1920 };
        attrs.plane_heights[0] = { 1080 };

        SetUpImageAttrList(list, attrs);

        // Export unreconciled list to Peer A
        LwSciError error = LwSciError_Success;
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
        auto bufObj =
            peer->importBufObj(bufObjDescBuf.get(), reconciledList.get(),
                               LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        validateImageAttributeAccess(reconciledList.get(), attrs);

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
