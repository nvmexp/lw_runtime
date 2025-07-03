/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 *(LwSciBufAttrValDataType)1 distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"
#include "lwcolor.h"

class TestLwSciBufRGBIRColorFormats
    : public LwSciBufBasicTest,
      public ::testing::WithParamInterface<std::tuple<
                                                LwSciBufAttrValColorFmt,
                                                uint32_t,
                                                LwSciBufAttrValDataType,
                                                uint8_t,
                                                LwColorFormat>>
{
public:
    void SetUpPeerAAttrList()
    {
        LwSciBufType bufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
        uint64_t lrPad = 0U, tbPad = 100U;
        bool cpuAccessFlag = true;
        bool vpr = false;
        int32_t planeWidths[] = {640U, 320U};
        int32_t planeHeights[] = {480U, 240U};
        uint32_t planeCount = 1U;
        LwSciBufAttrValImageScanType planeScanType[] = {LwSciBufScan_ProgressiveType};

        SET_ATTR(listPeerA.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listPeerA.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, cpuAccessFlag);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_Layout, layout);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_TopPadding, tbPad);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_BottomPadding, tbPad);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_LeftPadding, lrPad);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_RightPadding, lrPad);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_VprFlag, vpr);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
        SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_ScanType, planeScanType);
    }

    void SetUpPeerBAttrList()
    {
        LwSciBufType bufType = LwSciBufType_Image;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        int32_t planeWidths[] = {640U, 320U};
        int32_t planeHeights[] = {480U, 240U};
        bool cpuAccessFlag = true;

        SET_ATTR(listPeerB.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listPeerB.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
        SET_ATTR(listPeerB.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
        SET_ATTR(listPeerB.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, cpuAccessFlag);
        SET_ATTR(listPeerB.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    }

    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;
        LwSciBufBasicTest::SetUp();

        listPeerA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listPeerA.get(), nullptr);

        listPeerB = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listPeerB.get(), nullptr);
    }

    virtual void TearDown() override
    {
        listPeerA.reset();
        listPeerB.reset();
        LwSciBufBasicTest::TearDown();
    }

    std::shared_ptr<LwSciBufAttrListRec> listPeerA, listPeerB;
};

TEST_P(TestLwSciBufRGBIRColorFormats, OutputAttrsTest)
{
    auto params = GetParam();
    LwSciBufAttrValColorFmt colorFmt = std::get<0>(params);
    uint32_t bitsPerPixel = std::get<1>(params);
    LwSciBufAttrValDataType dataType = std::get<2>(params);
    uint8_t channelCount = std::get<3>(params);

    SetUpPeerAAttrList();
    SetUpPeerBAttrList();

    SET_ATTR(listPeerA.get(), LwSciBufImageAttrKey_PlaneColorFormat, colorFmt);

    LwSciError error = LwSciError_Success;
    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {listPeerA.get(), listPeerB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(reconciledList, nullptr);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufImageAttrKey_PlaneColorFormat, colorFmt, 0), true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufImageAttrKey_PlaneBitsPerPixel, bitsPerPixel, 0), true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufImageAttrKey_PlaneDatatype, dataType, 0), true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufImageAttrKey_PlaneChannelCount, channelCount, 0), true);

}

TEST_P(TestLwSciBufRGBIRColorFormats, ColorColwersionTest)
{
    auto params = GetParam();
    LwSciBufAttrValColorFmt sciColorFmt = std::get<0>(params);
    LwSciBufAttrValColorFmt colwertedSciColorFmt = LwSciColor_LowerBound;
    LwColorFormat lwColorFmt = std::get<4>(params);
    LwColorFormat colwertedLwColorFmt = LwColorFormat_Unspecified;

    LwSciError error = LwSciError_Success;
    error = LwSciColorToLwColor(sciColorFmt, &colwertedLwColorFmt);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_EQ(colwertedLwColorFmt, lwColorFmt);

    error = LwColorToLwSciColor(lwColorFmt, &colwertedSciColorFmt);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_EQ(colwertedSciColorFmt, sciColorFmt);
}

INSTANTIATE_TEST_CASE_P(
    TestLwSciBufRGBIRColorFormats, TestLwSciBufRGBIRColorFormats,
    testing::Values(
        std::make_tuple(LwSciColor_X6Bayer10BGGI_RGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10BGGI_RGGI),
        std::make_tuple(LwSciColor_X6Bayer10GBIG_GRIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10GBIG_GRIG),
        std::make_tuple(LwSciColor_X6Bayer10GIBG_GIRG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10GIBG_GIRG),
        std::make_tuple(LwSciColor_X6Bayer10IGGB_IGGR, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10IGGB_IGGR),
        std::make_tuple(LwSciColor_X6Bayer10RGGI_BGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10RGGI_BGGI),
        std::make_tuple(LwSciColor_X6Bayer10GRIG_GBIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10GRIG_GBIG),
        std::make_tuple(LwSciColor_X6Bayer10GIRG_GIBG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10GIRG_GIBG),
        std::make_tuple(LwSciColor_X6Bayer10IGGR_IGGB, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X6Bayer10IGGR_IGGB),
        std::make_tuple(LwSciColor_X4Bayer12BGGI_RGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12BGGI_RGGI),
        std::make_tuple(LwSciColor_X4Bayer12GBIG_GRIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12GBIG_GRIG),
        std::make_tuple(LwSciColor_X4Bayer12GIBG_GIRG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12GIBG_GIRG),
        std::make_tuple(LwSciColor_X4Bayer12IGGB_IGGR, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12IGGB_IGGR),
        std::make_tuple(LwSciColor_X4Bayer12RGGI_BGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12RGGI_BGGI),
        std::make_tuple(LwSciColor_X4Bayer12GRIG_GBIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12GRIG_GBIG),
        std::make_tuple(LwSciColor_X4Bayer12GIRG_GIBG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12GIRG_GIBG),
        std::make_tuple(LwSciColor_X4Bayer12IGGR_IGGB, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X4Bayer12IGGR_IGGB),
        std::make_tuple(LwSciColor_X2Bayer14BGGI_RGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14BGGI_RGGI),
        std::make_tuple(LwSciColor_X2Bayer14GBIG_GRIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14GBIG_GRIG),
        std::make_tuple(LwSciColor_X2Bayer14GIBG_GIRG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14GIBG_GIRG),
        std::make_tuple(LwSciColor_X2Bayer14IGGB_IGGR, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14IGGB_IGGR),
        std::make_tuple(LwSciColor_X2Bayer14RGGI_BGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14RGGI_BGGI),
        std::make_tuple(LwSciColor_X2Bayer14GRIG_GBIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14GRIG_GBIG),
        std::make_tuple(LwSciColor_X2Bayer14GIRG_GIBG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14GIRG_GIBG),
        std::make_tuple(LwSciColor_X2Bayer14IGGR_IGGB, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_X2Bayer14IGGR_IGGB),
        std::make_tuple(LwSciColor_Bayer16BGGI_RGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16BGGI_RGGI),
        std::make_tuple(LwSciColor_Bayer16GBIG_GRIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16GBIG_GRIG),
        std::make_tuple(LwSciColor_Bayer16GIBG_GIRG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16GIBG_GIRG),
        std::make_tuple(LwSciColor_Bayer16IGGB_IGGR, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16IGGB_IGGR),
        std::make_tuple(LwSciColor_Bayer16RGGI_BGGI, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16RGGI_BGGI),
        std::make_tuple(LwSciColor_Bayer16GRIG_GBIG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16GRIG_GBIG),
        std::make_tuple(LwSciColor_Bayer16GIRG_GIBG, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16GIRG_GIBG),
        std::make_tuple(LwSciColor_Bayer16IGGR_IGGB, (uint32_t)16, LwSciDataType_Uint16, (uint8_t)1U, LwColorFormat_Bayer16IGGR_IGGB)));
