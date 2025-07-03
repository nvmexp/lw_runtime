/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"

class TestLwSciBufImagePitch : public LwSciBufBasicTest
{
public:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        umd2AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd2AttrList.get(), nullptr);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        umd1AttrList.reset();
        umd2AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
    std::shared_ptr<LwSciBufAttrListRec> umd2AttrList;
};

TEST_F(TestLwSciBufImagePitch, IntrathreadImagePLProg)
{
    {
        LwSciBufType bufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
        uint64_t lrPad = 0U, tbPad = 100U;
        bool cpuAccessFlag = true;
        bool vpr = false;
        int32_t planeCount = 2U;

        LwSciBufAttrValColorFmt planeColorFmts[] = {LwSciColor_Y16,
                                                    LwSciColor_U8V8};
        LwSciBufAttrValColorStd planeColorStds[] = {LwSciColorStd_YcCbcCrc_SR,
                                                    LwSciColorStd_YcCbcCrc_SR};
        LwSciBufAttrValImageScanType planeScanType[] = {
            LwSciBufScan_ProgressiveType};

        int32_t planeWidths[] = {640U, 320U};
        int32_t planeHeights[] = {480U, 240U};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout, layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_TopPadding, tbPad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_BottomPadding, tbPad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_LeftPadding, lrPad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_RightPadding, lrPad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_VprFlag, vpr);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneCount,
                 planeCount);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                 planeColorFmts);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorStd,
                 planeColorStds);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
                 planeWidths);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
                 planeHeights);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
                 planeScanType);
    }

    {
        LwSciBufType bufType = LwSciBufType_Image;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        int32_t planeCount = 2U;
        int32_t planeWidths[] = {640U, 320U};
        int32_t planeHeights[] = {480U, 240U};
        bool cpuAccessFlag = true;

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
                 planeWidths);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
                 planeHeights);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    }

    LwSciError error = LwSciError_Success;
    auto bufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U;
    uint64_t len = 0U;
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    void* vaPtr = NULL;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), &vaPtr), LwSciError_Success)
        << "Failed to get cpu ptr";

    /* Verify CPU access */
    *(uint64_t *)vaPtr = (uint64_t)0xC0DEC0DEC0DEC0DEU;
    uint64_t testval = *(uint64_t*)vaPtr;
    ASSERT_EQ(testval, *(uint64_t *)vaPtr) << "CPU access failed";

    uint64_t size = GetMemorySize(rmHandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "Allocated size is not same as callwlated size."
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());
}
