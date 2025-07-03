/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"

typedef struct LwSciBufTestColorSpecRec {
    uint32_t planeCount;
    LwSciBufAttrValColorFmt colorFmt[LW_SCI_BUF_IMAGE_MAX_PLANES];
    LwSciBufAttrValDataType dataType[LW_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t bitsPerPixel[LW_SCI_BUF_IMAGE_MAX_PLANES];
} LwSciBufTestColorSpec;

class TestLwSciBufISPColorFormats : public LwSciBufBasicTest
{
public:
    void IntraThreadBayerISP(LwSciBufTestColorSpec* colorSpec,
                             uint64_t subTestNum)
    {
        // TODO: Each of these should probably be split into their own tests.
        // Deferring for now to avoid needing to update JAMA.
        for (uint32_t i = 0; i < subTestNum; i++) {
            LwSciError error = LwSciError_Success;

            auto umd1AttrList = peer.createAttrList(&error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_NE(umd1AttrList.get(), nullptr);

            {
                LwSciBufType bufType = LwSciBufType_Image;
                LwSciBufAttrValImageLayoutType layout =
                    LwSciBufImage_PitchLinearType;
                uint64_t lrPad = 0U, tbPad = 100U;
                bool cpuAccessFlag = true;
                bool vpr = false;

                LwSciBufAttrValImageScanType planeScanType[] = {
                    LwSciBufScan_ProgressiveType};

                int32_t planeWidths[] = {640U, 320U};
                int32_t planeHeights[] = {480U, 240U};

                SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types,
                         bufType);
                SET_ATTR(umd1AttrList.get(),
                         LwSciBufGeneralAttrKey_NeedCpuAccess, cpuAccessFlag);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout,
                         layout);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_TopPadding,
                         tbPad);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_BottomPadding,
                         tbPad);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_LeftPadding,
                         lrPad);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_RightPadding,
                         lrPad);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_VprFlag, vpr);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneCount,
                         colorSpec[i].planeCount);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
                         planeWidths);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
                         planeHeights);
                SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
                         planeScanType);

                LwSciBufAttrKeyValuePair imgBufAttrs[] = {
                    {LwSciBufImageAttrKey_PlaneColorFormat,
                     &colorSpec[i].colorFmt[0],
                     sizeof(colorSpec[i].colorFmt[0]) *
                         colorSpec[i].planeCount}};

                ASSERT_EQ(
                    LwSciBufAttrListSetAttrs(
                        umd1AttrList.get(), imgBufAttrs,
                        sizeof(imgBufAttrs) / sizeof(LwSciBufAttrKeyValuePair)),
                    LwSciError_Success);
            }

            auto umd2AttrList = peer.createAttrList(&error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_NE(umd2AttrList.get(), nullptr);

            {
                LwSciBufType bufType = LwSciBufType_Image;
                LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
                int32_t planeWidths[] = {640U, 320U};
                int32_t planeHeights[] = {480U, 240U};
                bool cpuAccessFlag = true;

                SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types,
                         bufType);
                SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
                         planeWidths);
                SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
                         planeHeights);
                SET_ATTR(umd2AttrList.get(),
                         LwSciBufGeneralAttrKey_NeedCpuAccess, cpuAccessFlag);
                SET_ATTR(umd2AttrList.get(),
                         LwSciBufGeneralAttrKey_RequiredPerm, perm);
            }

            auto reconciledList = LwSciBufPeer::attrListReconcile(
                {umd1AttrList.get(), umd2AttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_NE(reconciledList, nullptr);

            auto bufObj =
                LwSciBufPeer::allocateBufObj(reconciledList.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_NE(reconciledList, nullptr);

            ASSERT_EQ(VerifyReconciledList(reconciledList.get(), colorSpec[i]),
                      LwSciError_Success);

            LwSciBufRmHandle rmHandle = {0};
            uint64_t offset = 0U;
            uint64_t len = 0U;
            ASSERT_EQ(
                LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
                LwSciError_Success);

            void* vaPtr = NULL;
            ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), &vaPtr),
                      LwSciError_Success);
            ;

            /* Verify CPU access */
            *(uint64_t*)vaPtr = (uint64_t)0xC0DEC0DEC0DEC0DEU;
            uint64_t testval = *(uint64_t*)vaPtr;
            ASSERT_EQ(testval, *(uint64_t*)vaPtr);

            uint64_t size = GetMemorySize(rmHandle);
            ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()));
        }
    }

    LwSciError VerifyReconciledList(LwSciBufAttrList reconciledList,
                                    LwSciBufTestColorSpec colorSpec)
    {
        LwSciError err = LwSciError_Success;
        const void* value = NULL;
        size_t size = 0U;

        LwSciBufAttrKeyValuePair imgBufAttrs[] = {
            {LwSciBufImageAttrKey_PlaneCount, NULL, 0},
            {LwSciBufImageAttrKey_PlaneColorFormat, NULL, 0},
            {LwSciBufImageAttrKey_PlaneDatatype, NULL, 0},
            {LwSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0},
        };

        err = LwSciBufAttrListGetAttrs(reconciledList, imgBufAttrs,
                                       sizeof(imgBufAttrs) /
                                           sizeof(LwSciBufAttrKeyValuePair));
        TESTERR_CHECK(err, "Failed to set UMD2 attribute list", err);

        if (colorSpec.planeCount != *(const uint32_t*)imgBufAttrs[0].value) {
            err = LwSciError_Unknown;
            TESTERR_CHECK(err, "Plane count didnt match", err);
        }

        for (uint32_t i = 0; i < colorSpec.planeCount; i++) {
            if (colorSpec.colorFmt[i] !=
                ((const LwSciBufAttrValColorFmt*)imgBufAttrs[1].value)[i]) {
                err = LwSciError_IlwalidState;
                TESTERR_CHECK(err, "Plane color didnt match", err);
            }

            if (colorSpec.dataType[i] !=
                ((const LwSciBufAttrValDataType*)imgBufAttrs[2].value)[i]) {
                err = LwSciError_IlwalidState;
                TESTERR_CHECK(err, "Plane data type didnt match", err);
            }

            if (colorSpec.bitsPerPixel[i] !=
                ((const LwSciBufAttrValDataType*)imgBufAttrs[3].value)[i]) {
                err = LwSciError_IlwalidState;
                TESTERR_CHECK(err, "Plane BPP didnt match", err);
            }
        }

        return err;
}
};

TEST_F(TestLwSciBufISPColorFormats, IntraThreadBayer12)
{
    LwSciBufTestColorSpec colorSpec [] = {
    {1U, {LwSciColor_X4Bayer12RGGB}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12BGGR}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12GRBG}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12GBRG}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12RCCB}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12BCCR}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12CRBC}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12CBRC}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12RCCC}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12CCCR}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12CRCC}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12CCRC}, {LwSciDataType_Uint16}, {16U}},
    {1U, {LwSciColor_X4Bayer12CCCC}, {LwSciDataType_Uint16}, {16U}},
    };

    uint32_t subTestNum = sizeof(colorSpec)/sizeof(colorSpec[0]);
    IntraThreadBayerISP(colorSpec, subTestNum);
}

TEST_F(TestLwSciBufISPColorFormats, IntraThreadBayer16)
{
    LwSciBufTestColorSpec colorSpec[] = {
        {1U, {LwSciColor_Bayer16RGGB}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16BGGR}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16GRBG}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16GBRG}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16RCCB}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16BCCR}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16CRBC}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16CBRC}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16RCCC}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16CCCR}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16CRCC}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16CCRC}, {LwSciDataType_Uint16}, {16U}},
        {1U, {LwSciColor_Bayer16CCCC}, {LwSciDataType_Uint16}, {16U}},
    };

    uint32_t subTestNum = sizeof(colorSpec)/sizeof(colorSpec[0]);

    IntraThreadBayerISP(colorSpec, subTestNum);
}

TEST_F(TestLwSciBufISPColorFormats, IntraThreadBayer20)
{
    LwSciBufTestColorSpec colorSpec[] = {
        {1U, {LwSciColor_X12Bayer20RGGB}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20BGGR}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20GRBG}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20GBRG}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20RCCB}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20BCCR}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20CRBC}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20CBRC}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20RCCC}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20CCCR}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20CRCC}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20CCRC}, {LwSciDataType_Uint32}, {32U}},
        {1U, {LwSciColor_X12Bayer20CCCC}, {LwSciDataType_Uint32}, {32U}},
    };

    uint32_t subTestNum = sizeof(colorSpec)/sizeof(colorSpec[0]);

    IntraThreadBayerISP(colorSpec, subTestNum);
}

TEST_F(TestLwSciBufISPColorFormats, IntraThreadBayerFloatISP)
{
    LwSciBufTestColorSpec colorSpec[] = {
        {1U,
         {LwSciColor_FloatISP_Bayer16RGGB},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16BGGR},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16GRBG},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16GBRG},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16RCCB},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16BCCR},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16CRBC},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16CBRC},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16RCCC},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16CCCR},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16CRCC},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16CCRC},
         {LwSciDataType_FloatISP},
         {16U}},
        {1U,
         {LwSciColor_FloatISP_Bayer16CCCC},
         {LwSciDataType_FloatISP},
         {16U}},
    };

    uint32_t subTestNum = sizeof(colorSpec)/sizeof(colorSpec[0]);

    IntraThreadBayerISP(colorSpec, subTestNum);
}

TEST_F(TestLwSciBufISPColorFormats, IntraThreadISPRGBA)
{
    LwSciBufTestColorSpec colorSpec[] = {
        {1U, {LwSciColor_A8B8G8R8}, {LwSciDataType_Uint8}, {32U}},
        {1U, {LwSciColor_A16B16G16R16}, {LwSciDataType_Uint16}, {64U}},
    };

    uint32_t subTestNum = sizeof(colorSpec)/sizeof(colorSpec[0]);

    IntraThreadBayerISP(colorSpec, subTestNum);
}

TEST_F(TestLwSciBufISPColorFormats, IntraThreadISPYUVX)
{
    LwSciBufTestColorSpec colorSpec[] = {
        {1U, {LwSciColor_A8Y8U8V8}, {LwSciDataType_Uint8}, {32U}},
        {1U, {LwSciColor_A16Y16U16V16}, {LwSciDataType_Uint16}, {64U}},
    };

    uint32_t subTestNum = sizeof(colorSpec)/sizeof(colorSpec[0]);

    IntraThreadBayerISP(colorSpec, subTestNum);
}

TEST_F(TestLwSciBufISPColorFormats, IntraThreadISPYUV)
{
    LwSciBufTestColorSpec colorSpec[] = {
        {2U,
         {LwSciColor_Y8, LwSciColor_U8_V8},
         {LwSciDataType_Uint8, LwSciDataType_Uint8},
         {8U, 16U}},
        {2U,
         {LwSciColor_Y8, LwSciColor_V8U8},
         {LwSciDataType_Uint8, LwSciDataType_Uint8},
         {8U, 16U}},
    };

    uint32_t subTestNum = sizeof(colorSpec)/sizeof(colorSpec[0]);

    IntraThreadBayerISP(colorSpec, subTestNum);
}
