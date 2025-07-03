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
#include "gtest/gtest.h"

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class ImageAttributes : public LwSciBufBasicTest
{
protected:
    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;

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
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        listA.reset();
        listB.reset();
    }
};

/**
* Test case: Test to verify Image Attributes Comparison with EqualValue Policy
*            and TrueValue Policy - Set any of the slots of the provided
*            unreconciled attribute lists equivalent of true
*/
TEST_F(ImageAttributes, EqualValueAndTrueValuePolicy)
{
    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;

    uint64_t lPad = 0U, tPad = 100U, bPad = 50U, rPad = 25U;
    uint32_t planeCount = 2U;
    uint64_t imageCount = 1U;

    bool vprFlag = true;
    bool needCpuAccess = true;

    LwSciBufAttrValColorFmt planeColorFmts[2] = { LwSciColor_Y16,
                                                 LwSciColor_U8V8 };
    LwSciBufAttrValColorStd planeColorStds[2] = { LwSciColorStd_YcCbcCrc_SR,
                                                 LwSciColorStd_YcCbcCrc_SR };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_ProgressiveType };
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
    LwSciBufType bufType = LwSciBufType_Image;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    uint32_t planeWidths[2] = { 640U, 320U };
    uint32_t planeHeights[2] = { 480U, 240U };

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, needCpuAccess);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, needCpuAccess);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_Layout, layout);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_ImageCount, imageCount);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_TopPadding, tPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_BottomPadding, bPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_LeftPadding, lPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_RightPadding, rPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_ScanType, planeScanType);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneColorFormat,
             planeColorFmts);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneColorStd, planeColorStds);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_VprFlag, vprFlag);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);
    ASSERT_EQ(error, LwSciError_Success);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_Types, bufType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_NeedCpuAccess,
                                         needCpuAccess));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufImageAttrKey_Layout, layout));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_ImageCount, imageCount));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_TopPadding, tPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_BottomPadding, bPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_LeftPadding, lPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_RightPadding, rPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_PlaneCount, planeCount));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_ScanType, planeScanType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufImageAttrKey_PlaneColorFormat,
                                         planeColorFmts));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufImageAttrKey_PlaneColorStd,
                                         planeColorStds));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_VprFlag, vprFlag));
}

/**
* Image Attributes Comparision - TrueValue Policy
* Test case : None of the slots of the provided unreconciled attribute lists has
*             the attribute value set to equivalent of true
*/
TEST_F(ImageAttributes, TrueValuePolicy)
{
    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;

    uint64_t lPad = 0U, tPad = 100U, bPad = 50U, rPad = 25U;
    uint32_t planeCount = 2U;
    uint64_t imageCount = 1U;
    LwSciBufAttrValColorFmt planeColorFmts[2] = { LwSciColor_Y16,
                                                 LwSciColor_U8V8 };
    LwSciBufAttrValColorStd planeColorStds[2] = { LwSciColorStd_YcCbcCrc_SR,
                                                 LwSciColorStd_YcCbcCrc_SR };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_ProgressiveType };
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
    LwSciBufType bufType = LwSciBufType_Image;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    uint32_t planeWidths[2] = { 640U, 320U };
    uint32_t planeHeights[2] = { 480U, 240U };

    bool vprFlag = false;
    bool needCpuAccess = true;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, needCpuAccess);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, needCpuAccess);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_Layout, layout);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_ImageCount, imageCount);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_TopPadding, tPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_BottomPadding, bPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_LeftPadding, lPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_RightPadding, rPad);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_ScanType, planeScanType);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneColorFormat,
             planeColorFmts);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneColorStd, planeColorStds);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
    SET_ATTR(listB.get(), LwSciBufImageAttrKey_VprFlag, vprFlag);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);
    ASSERT_EQ(error, LwSciError_Success);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_Types, bufType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_NeedCpuAccess,
                                         needCpuAccess));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufImageAttrKey_Layout, layout));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_ImageCount, imageCount));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_TopPadding, tPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_BottomPadding, bPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_LeftPadding, lPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_RightPadding, rPad));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_PlaneCount, planeCount));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_ScanType, planeScanType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufImageAttrKey_PlaneColorFormat,
                                         planeColorFmts));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufImageAttrKey_PlaneColorStd,
                                         planeColorStds));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufImageAttrKey_VprFlag, vprFlag));
}
