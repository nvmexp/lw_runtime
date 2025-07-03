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

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class TestLwSciBufImagePyramid : public LwSciBufBasicTest
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

TEST_F(TestLwSciBufImagePyramid, IntraThreadImagePyramid)
{
    {
        LwSciBufType bufType = LwSciBufType_Pyramid;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
        uint64_t lrpad = 0, tbpad = 100;
        bool cpuaccess_flag = true;
        bool vpr = false;
        int32_t planecount = 2;
        uint32_t levelcount = 4;
        float scale = 0.5;

        LwSciBufAttrValColorFmt planecolorfmts[] = {LwSciColor_Y16,
                                                    LwSciColor_U8V8};
        LwSciBufAttrValColorStd planecolorstds[] = {LwSciColorStd_YcCbcCrc_SR,
                                                    LwSciColorStd_YcCbcCrc_SR};
        LwSciBufAttrValImageScanType planescantype[] = {
            LwSciBufScan_ProgressiveType};

        uint32_t plane_widths[] = {1920, 960};
        uint32_t plane_heights[] = {1080, 540};
        uint32_t planeAlignment[] = {1024U, 1024U};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuaccess_flag);
        SET_ATTR(umd1AttrList.get(), LwSciBufPyramidAttrKey_NumLevels,
                 levelcount);
        SET_ATTR(umd1AttrList.get(), LwSciBufPyramidAttrKey_Scale, scale);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout, layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_TopPadding, tbpad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_BottomPadding, tbpad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_LeftPadding, lrpad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_RightPadding, lrpad);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_VprFlag, vpr);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneCount,
                 planecount);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                 planecolorfmts);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorStd,
                 planecolorstds);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
                 plane_widths);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
                 plane_heights);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
                 planescantype);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneBaseAddrAlign,
                 planeAlignment);
    }

    {
        int32_t planecount = 2;
        int32_t plane_widths[] = {1920, 960};
        int32_t plane_heights[] = {1080, 540};
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufMemDomain memdomain = LwSciBufMemDomain_Sysmem;
        LwSciBufType bufType = LwSciBufType_Pyramid;

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
                 plane_widths);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
                 plane_heights);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memdomain);
    }

    {
        LwSciError error = LwSciError_Success;
#if !defined(__x86_64__)
        /* The Image pyramid size is different on cheetah and x86 because the
         * default Image constraints applied are different in both cases.
         */
        uint64_t precomputedImagePyramidSize = 0x00000000720000;
#else
        uint64_t precomputedImagePyramidSize = 0x00000000691010;
#endif
        uint64_t precomputedImagePyramidAlignment = 0x00000000000400;

        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get(), umd2AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Verify that reconciled size callwlated by LwSciBuf matches with the
         * precomputed size. Note that, the values of input attributes set in
         * this test-case should not be changed. The size value must be
         * precomputed if at all there is a need to change the input attribute
         * values.
         */
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
            LwSciBufImageAttrKey_Size, precomputedImagePyramidSize));

        /* Verify that reconciled alignment callwlated by LwSciBuf matches with
         * the precomputed alignment. Note that, the values of input attributes
         * set in this test-case should not be changed. The alignment value must
         * be precomputed if at all there is a need to change the input
         * attribute values.
         */
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
            LwSciBufImageAttrKey_Alignment, precomputedImagePyramidAlignment));

        auto bufObj = LwSciBufPeer::allocateBufObj(reconciledList.get(),
                        &error);
        ASSERT_EQ(error, LwSciError_Success);

        /* Allocation size check */
        LwSciBufRmHandle rmHandle = {0};
        uint64_t offset = 0U;
        uint64_t len = 0U;
        ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset,
                &len), LwSciError_Success)
            << "Failed to Get Lwrm Memhandle for the object";

        /* Get size from RM */
        uint64_t size = GetMemorySize(rmHandle);
        ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
            << "size verification failed"
            << " Expected " << size << " Got " <<
            CEIL_TO_LEVEL(len, GetPageSize());
    }
}

class TestLwSciBufImagePyramidMandatoryOptionalAttrs :
            public TestLwSciBufImagePyramid,
            public ::testing::WithParamInterface<
                std::tuple<LwSciBufAttrKey, LwSciError>>
{
};

/* This test verifies that if mandatory attributes are not set in the
 * unreconciled list then the reconciliation fails.
 */
TEST_P(TestLwSciBufImagePyramidMandatoryOptionalAttrs, MandatoryOptionalAttrs)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Pyramid;
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
    uint64_t top = 0U, bottom = 0U, left = 0U, right = 0U, imageCount = 1U;
    bool vpr = false;
    uint32_t planeCount = 1U, planeBaseAddr = 4096U, planeWidth = 1920U,
        planeHeight = 1080U, levelCount = 2U;
    LwSciBufAttrValColorFmt colorFmt = LwSciColor_A8B8G8R8;
    LwSciBufAttrValColorStd colorStd = LwSciColorStd_SRGB;
    LwSciBufAttrValImageScanType scanType = LwSciBufScan_ProgressiveType;
    float scale = 0.5;

    auto param = GetParam();
    auto key = std::get<0>(param);
    auto expectedError = std::get<1>(param);

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);

    if (key != LwSciBufImageAttrKey_Layout) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout, layout);
    }

    if (key != LwSciBufImageAttrKey_TopPadding) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_TopPadding, top);
    }

    if (key != LwSciBufImageAttrKey_BottomPadding) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_BottomPadding,
            bottom);
    }

    if (key != LwSciBufImageAttrKey_LeftPadding) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_LeftPadding, left);
    }

    if (key != LwSciBufImageAttrKey_RightPadding) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_RightPadding, right);
    }

    if (key != LwSciBufImageAttrKey_VprFlag) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_VprFlag, vpr);
    }

    if (key != LwSciBufImageAttrKey_PlaneCount) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneCount,
            planeCount);
    }

    if (key != LwSciBufImageAttrKey_PlaneColorFormat) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorFormat,
            colorFmt);
    }

    if (key != LwSciBufImageAttrKey_PlaneColorStd) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorStd,
            colorStd);
    }

    if (key != LwSciBufImageAttrKey_PlaneBaseAddrAlign) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneBaseAddrAlign,
            planeBaseAddr);
    }

    if (key != LwSciBufImageAttrKey_PlaneWidth) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
            planeWidth);
    }

    if (key != LwSciBufImageAttrKey_PlaneHeight) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
            planeHeight);
    }

    if (key != LwSciBufImageAttrKey_ScanType) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    if (key != LwSciBufImageAttrKey_ImageCount) {
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ImageCount,
            imageCount);
    }

    if (key != LwSciBufPyramidAttrKey_NumLevels) {
        SET_ATTR(umd1AttrList.get(), LwSciBufPyramidAttrKey_NumLevels,
            levelCount);
    }

    if (key != LwSciBufPyramidAttrKey_Scale) {
        SET_ATTR(umd1AttrList.get(), LwSciBufPyramidAttrKey_Scale,
            scale);
    }

    if (expectedError != LwSciError_Success) {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    } else {
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    }
    ASSERT_EQ(error, expectedError);
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufImagePyramidMandatoryOptionalAttrs,
    TestLwSciBufImagePyramidMandatoryOptionalAttrs,
    ::testing::Values(
        /* First value in tuple represents an attribute to be skipped from
         * setting in the unreconciled list.
         * Second value in tuple represents an expected error code during
         * reconciliation based on whether the attribute skipped from being
         * set in unreconciled list is mandatory or optional.
         */
        std::make_tuple(LwSciBufImageAttrKey_Layout,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufImageAttrKey_TopPadding,
                LwSciError_Success),
        std::make_tuple(LwSciBufImageAttrKey_BottomPadding,
                LwSciError_Success),
        std::make_tuple(LwSciBufImageAttrKey_LeftPadding,
                LwSciError_Success),
        std::make_tuple(LwSciBufImageAttrKey_RightPadding,
                LwSciError_Success),
        std::make_tuple(LwSciBufImageAttrKey_VprFlag,
                LwSciError_Success),
        std::make_tuple(LwSciBufImageAttrKey_PlaneCount,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufImageAttrKey_PlaneColorFormat,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufImageAttrKey_PlaneColorStd,
                LwSciError_Success),
        std::make_tuple(LwSciBufImageAttrKey_PlaneBaseAddrAlign,
                LwSciError_Success),
        std::make_tuple(LwSciBufImageAttrKey_PlaneWidth,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufImageAttrKey_PlaneHeight,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufImageAttrKey_ScanType,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufImageAttrKey_ImageCount,
                LwSciError_Success),
        std::make_tuple(LwSciBufPyramidAttrKey_NumLevels,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufPyramidAttrKey_Scale,
                LwSciError_ReconciliationFailed),
        std::make_tuple(LwSciBufAttrKey_LowerBound,
                LwSciError_Success)
));

/*
 * This test verifies the input/output accessibility of the pyramid attributes.
 * If the attribute is input attribute (input only or input/output) then
 * attribute can be set in unreconciled list. It can also be read from
 * unreconciled list.
 * If the attribute is output attribute (output only or input/output) then
 * attribute can be read from reconciled list.
 */
TEST_F(TestLwSciBufImagePyramid, InOutAttrs)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Pyramid;
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
    uint64_t top = 0U, bottom = 0U, left = 0U, right = 0U, imageCount = 1U,
        imageSize = 4096U, imageAlignment = 4096U, planeOffset = 0U,
        planeSecondFieldOffset = 0U, planeAlignedSize = 4096U,
        levelOffset = 1024U, levelSize = 1024U, levelAlign = 1024U;
    bool vpr = false;
    uint32_t planeCount = 1U, planeBaseAddr = 4096U, planeWidth = 1920U,
        planeHeight = 1080U, levelCount = 2U, planeBPP = 4U, planePitch = 32U,
        planeAlignedHeight = 1024U, GobSize = 1024U, GobX = 1024U, GobY = 1024U,
        GobZ = 1024U;
    LwSciBufAttrValColorFmt colorFmt = LwSciColor_A8B8G8R8;
    LwSciBufAttrValColorStd colorStd = LwSciColorStd_SRGB;
    LwSciBufAttrValImageScanType scanType = LwSciBufScan_ProgressiveType;
    float scale = 0.5;
    LwSciBufAttrValDataType planeDatatype = LwSciDataType_Uint8;
    uint8_t planeChannelCount = 4U;

    std::vector<LwSciBufAttrKeyValuePair> pyramidPublicAttrKeySet = {
        {
            .key = LwSciBufImageAttrKey_Layout,
            .value = &layout,
            .len = sizeof(layout)
        },

        {
            .key = LwSciBufImageAttrKey_TopPadding,
            .value = &top,
            .len = sizeof(top)
        },

        {
            .key = LwSciBufImageAttrKey_BottomPadding,
            .value = &bottom,
            .len = sizeof(bottom)
        },

        {
            .key = LwSciBufImageAttrKey_LeftPadding,
            .value = &left,
            .len = sizeof(left)
        },

        {
            .key = LwSciBufImageAttrKey_RightPadding,
            .value = &right,
            .len = sizeof(right)
        },

        {
            .key = LwSciBufImageAttrKey_VprFlag,
            .value = &vpr,
            .len = sizeof(vpr)
        },

        {
            .key = LwSciBufImageAttrKey_Size,
            .value = &imageSize,
            .len = sizeof(imageSize)
        },

        {
            .key = LwSciBufImageAttrKey_Alignment,
            .value = &imageAlignment,
            .len = sizeof(imageAlignment)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneCount,
            .value = &planeCount,
            .len = sizeof(planeCount)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneColorFormat,
            .value = &colorFmt,
            .len = sizeof(colorFmt)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneColorStd,
            .value = &colorStd,
            .len = sizeof(colorStd)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneBaseAddrAlign,
            .value = &planeBaseAddr,
            .len = sizeof(planeBaseAddr)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneWidth,
            .value = &planeWidth,
            .len = sizeof(planeWidth)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneHeight,
            .value = &planeHeight,
            .len = sizeof(planeHeight)
        },

        {
            .key = LwSciBufImageAttrKey_ScanType,
            .value = &scanType,
            .len = sizeof(scanType)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneBitsPerPixel,
            .value = &planeBPP,
            .len = sizeof(planeBPP)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneOffset,
            .value = &planeOffset,
            .len = sizeof(planeOffset)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneDatatype,
            .value = &planeDatatype,
            .len = sizeof(planeDatatype)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneChannelCount,
            .value = &planeChannelCount,
            .len = sizeof(planeChannelCount)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneSecondFieldOffset,
            .value = &planeSecondFieldOffset,
            .len = sizeof(planeSecondFieldOffset)
        },

        {
            .key = LwSciBufImageAttrKey_PlanePitch,
            .value = &planePitch,
            .len = sizeof(planePitch)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneAlignedHeight,
            .value = &planeAlignedHeight,
            .len = sizeof(planeAlignedHeight)
        },

        {
            .key = LwSciBufImageAttrKey_PlaneAlignedSize,
            .value = &planeAlignedSize,
            .len = sizeof(planeAlignedSize)
        },

        {
            .key = LwSciBufImageAttrKey_ImageCount,
            .value = &imageCount,
            .len = sizeof(imageCount)
        },

        {
            .key = LwSciBufPyramidAttrKey_NumLevels,
            .value = &levelCount,
            .len = sizeof(levelCount)
        },

        {
            .key = LwSciBufPyramidAttrKey_Scale,
            .value = &scale,
            .len = sizeof(scale)
        },

        {
            .key = LwSciBufPyramidAttrKey_LevelOffset,
            .value = &levelOffset,
            .len = sizeof(levelOffset)
        },

        {
            .key = LwSciBufPyramidAttrKey_LevelSize,
            .value = &levelSize,
            .len = sizeof(levelSize)
        },

        {
            .key = LwSciBufPyramidAttrKey_Alignment,
            .value = &levelAlign,
            .len = sizeof(levelAlign)
        },
    };

    std::vector<LwSciBufInternalAttrKeyValuePair> pyramidInternalAttrKeySet = {
        {
            .key = LwSciBufInternalImageAttrKey_PlaneGobSize,
            .value = &GobSize,
            .len = sizeof(GobSize)
        },

        {
            .key = LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockX,
            .value = &GobX,
            .len = sizeof(GobX)},

        {
            .key = LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockY,
            .value = &GobY,
            .len = sizeof(GobY)
        },

        {
            .key = LwSciBufInternalImageAttrKey_PlaneLog2GobsPerBlockZ,
            .value = &GobZ,
            .len = sizeof(GobZ)
        },
    };

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);

    for (auto pyramidPublicAttrKey : pyramidPublicAttrKeySet) {
        auto it1 = std::find(LwSciBufPeer::outputAttrKeys.begin(),
                    LwSciBufPeer::outputAttrKeys.end(),
                    pyramidPublicAttrKey.key);
        if (it1 == LwSciBufPeer::outputAttrKeys.end()) {
            /* attribute is input only or input/output attribute. We should be
             * able to set it in the unreconciled list. We should also be able
             * to get it from unreconciled list.
             */
            error = LwSciBufAttrListSetAttrs(umd1AttrList.get(),
                        &pyramidPublicAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);

            error = LwSciBufAttrListGetAttrs(umd1AttrList.get(),
                        &pyramidPublicAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);
        } else {
            /* attribute is output only. Trying to set it in unreconciled list
             * should throw an error. Similarly, trying to get it from
             * unreconciled list should throw an error.
             */
            NEGATIVE_TEST();
            error = LwSciBufAttrListSetAttrs(umd1AttrList.get(),
                        &pyramidPublicAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);

            error = LwSciBufAttrListGetAttrs(umd1AttrList.get(),
                        &pyramidPublicAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);
        }
    }

    for (auto pyramidInternalAttrKey : pyramidInternalAttrKeySet) {
        auto it1 = std::find(LwSciBufPeer::internalAttrKeys.begin(),
                    LwSciBufPeer::internalAttrKeys.end(),
                    pyramidInternalAttrKey.key);
        if (it1 != LwSciBufPeer::internalAttrKeys.end()) {
            /* attribute is input only or input/output attribute. We should be
             * able to set it in the unreconciled list. We should also be able
             * to get it from unreconciled list.
             */
            error = LwSciBufAttrListSetInternalAttrs(umd1AttrList.get(),
                        &pyramidInternalAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);

            error = LwSciBufAttrListGetInternalAttrs(umd1AttrList.get(),
                        &pyramidInternalAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);
        } else {
            /* attribute is output only. Trying to set it in unreconciled list
             * should throw an error. Similarly, trying to get it from
             * unreconciled list should throw an error.
             */
            NEGATIVE_TEST();
            error = LwSciBufAttrListSetInternalAttrs(umd1AttrList.get(),
                        &pyramidInternalAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);

            error = LwSciBufAttrListGetInternalAttrs(umd1AttrList.get(),
                        &pyramidInternalAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);
        }
    }

    /* Now, reconcile the list. */
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    for (auto pyramidPublicAttrKey : pyramidPublicAttrKeySet) {
        auto it1 = std::find(LwSciBufPeer::inputAttrKeys.begin(),
                    LwSciBufPeer::inputAttrKeys.end(),
                    pyramidPublicAttrKey.key);
        if (it1 == LwSciBufPeer::inputAttrKeys.end()) {
            /* Attribute is output only or input/output. We should be able to
             * read it from reconciled list.
             */
            error = LwSciBufAttrListGetAttrs(reconciledList.get(),
                        &pyramidPublicAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);
        } else {
            /* Attribute is input only attribute. Trying to read it from
             * reconciled list should throw an error.
             */
            NEGATIVE_TEST();
            error = LwSciBufAttrListGetAttrs(reconciledList.get(),
                        &pyramidPublicAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);
        }
    }

    for (auto pyramidInternalAttrKey : pyramidInternalAttrKeySet) {
        auto it1 = std::find(LwSciBufPeer::internalAttrKeys.begin(),
                    LwSciBufPeer::internalAttrKeys.end(),
                    pyramidInternalAttrKey.key);
        auto it2 = std::find(LwSciBufPeer::outputInternalAttrKeys.begin(),
                    LwSciBufPeer::outputInternalAttrKeys.end(),
                    pyramidInternalAttrKey.key);
        if ((it1 != LwSciBufPeer::internalAttrKeys.end()) ||
            (it2 != LwSciBufPeer::outputInternalAttrKeys.end())) {
            /* Attribute is output only or input/output. We should be able to
             * read it from reconciled list.
             */
            error = LwSciBufAttrListGetInternalAttrs(reconciledList.get(),
                        &pyramidInternalAttrKey, 1);
            ASSERT_EQ(error, LwSciError_Success);
        } else {
            /* Attribute is input only attribute. Trying to read it from
             * reconciled list should throw an error.
             */
            NEGATIVE_TEST();
            error = LwSciBufAttrListGetInternalAttrs(reconciledList.get(),
                        &pyramidInternalAttrKey, 1);
            ASSERT_EQ(error, LwSciError_BadParameter);
        }
    }
}

/*
 * This test verifies the reconciliation validation functionality for array
 * attributes.
 */
TEST_F(TestLwSciBufImagePyramid, Reconciliatiolwalidation)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Pyramid;
    LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
    uint64_t top = 0U, bottom = 0U, left = 0U, right = 0U, imageCount = 1U;
    bool vpr = false;
    uint32_t planeCount = 1U, planeBaseAddr = 4096U, planeWidth = 1920U,
        planeHeight = 1080U, levelCount = 2U;
    LwSciBufAttrValColorFmt colorFmt = LwSciColor_A8B8G8R8;
    LwSciBufAttrValColorStd colorStd = LwSciColorStd_SRGB;
    LwSciBufAttrValImageScanType scanType = LwSciBufScan_ProgressiveType;
    float scale = 0.5;

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout, layout);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_TopPadding, top);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_BottomPadding, bottom);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_LeftPadding, left);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_RightPadding, right);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_VprFlag, vpr);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorFormat,
        colorFmt);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneColorStd, colorStd);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneBaseAddrAlign,
        planeBaseAddr);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidth);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeight);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType, scanType);
    SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ImageCount, imageCount);
    SET_ATTR(umd1AttrList.get(), LwSciBufPyramidAttrKey_NumLevels, levelCount);
    SET_ATTR(umd1AttrList.get(), LwSciBufPyramidAttrKey_Scale, scale);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        /* Setup another unreconciled list to be verified against reconciled
         * list.
         */
        bool isReconciledListValid = false;

        /* LwSciBufImageAttrKey_Layout uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        LwSciBufAttrValImageLayoutType validateLayout =
            LwSciBufImage_BlockLinearType;
        /* LwSciBufImageAttrKey_TopPadding uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        uint64_t validateTop = 1U;
        /* LwSciBufImageAttrKey_BottomPadding uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        uint64_t validateBottom = 1U;
        /* LwSciBufImageAttrKey_LeftPadding uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        uint64_t validateLeft = 1U;
        /* LwSciBufImageAttrKey_RightPadding uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        uint64_t validateRight = 1U;
        /* LwSciBufImageAttrKey_VprFlag uses true value validation policy
         * Thus set the value in unreconciled list which is true since the value
         * set in umd1AttrList is false and verify that validation fails.
         */
        bool validateVpr = true;
        /* LwSciBufImageAttrKey_PlaneCount uses equal value validation policy
         * Thus set the value in unreconciled list which is not equal to that
         * set in umd1AttrList and verify that validation fails.
         */
        uint32_t validatePlaneCount = 2U;
        /* LwSciBufImageAttrKey_PlaneColorFormat uses equal value validation
         * policy. Thus set the value in unreconciled list which is not equal to
         * that set in umd1AttrList and verify that validation fails.
         */
        LwSciBufAttrValColorFmt validateColorFmt = LwSciColor_A2R10G10B10;
        /* LwSciBufImageAttrKey_PlaneColorStd uses equal value validation
         * policy. Thus set the value in unreconciled list which is not equal to
         * that set in umd1AttrList and verify that validation fails.
         */
        LwSciBufAttrValColorStd validateColorStd = LwSciColorStd_REC601_ER;
        /* LwSciBufImageAttrKey_PlaneBaseAddrAlign uses greater value validation
         * policy. Thus set the value in unreconciled list which is greater than
         * that set in umd1AttrList and verify that validation fails.
         */
        uint32_t validatePlaneBaseAddr = 0x40000000U;
        /* LwSciBufImageAttrKey_PlaneWidth uses equal value validation
         * policy. Thus set the value in unreconciled list which is not equal to
         * that set in umd1AttrList and verify that validation fails.
         */
        uint32_t validatePlaneWidth = 3840U;
        /* LwSciBufImageAttrKey_PlaneHeight uses equal value validation
         * policy. Thus set the value in unreconciled list which is not equal to
         * that set in umd1AttrList and verify that validation fails.
         */
        uint32_t validatePlaneHeight = 2160U;
        /* LwSciBufImageAttrKey_PlaneScanType uses equal value validation
         * policy. Thus set the value in unreconciled list which is not equal to
         * that set in umd1AttrList and verify that validation fails.
         */
        LwSciBufAttrValImageScanType validateScanType =
            LwSciBufScan_InterlaceType;
        /* LwSciBufImageAttrKey_ImageCount uses equal value validation
         * policy. For Image count, LwSciBuf driver only allows value 1 to be
         * set and thus, we cannot generate invalid unreconciled attribute list
         * with invalid image count value. Thus, skip setting this value.
         */
        //uint64_t validateImageCount = 2U;
        /* LwSciBufPyramidAttrKey_NumLevels uses equal value validation
         * policy. Thus set the value in unreconciled list which is not equal to
         * that set in umd1AttrList and verify that validation fails.
         */
        uint32_t validateLevelCount = 3U;
        /* LwSciBufPyramidAttrKey_Scale uses equal value validation
         * policy. Thus set the value in unreconciled list which is not equal to
         * that set in umd1AttrList and verify that validation fails.
         */
        float validateScale = 0.7;

        std::vector<LwSciBufAttrKeyValuePair> keyValPair = {
            {
                .key = LwSciBufImageAttrKey_Layout,
                .value = &validateLayout,
                .len = sizeof(validateLayout),
            },

            {
                .key = LwSciBufImageAttrKey_TopPadding,
                .value = &validateTop,
                .len = sizeof(validateTop),
            },

            {
                .key = LwSciBufImageAttrKey_BottomPadding,
                .value = &validateBottom,
                .len = sizeof(validateBottom),
            },

            {
                .key = LwSciBufImageAttrKey_LeftPadding,
                .value = &validateLeft,
                .len = sizeof(validateLeft),
            },

            {
                .key = LwSciBufImageAttrKey_RightPadding,
                .value = &validateRight,
                .len = sizeof(validateRight),
            },

            {
                .key = LwSciBufImageAttrKey_VprFlag,
                .value = &validateVpr,
                .len = sizeof(validateVpr),
            },

            {
                .key = LwSciBufImageAttrKey_PlaneCount,
                .value = &validatePlaneCount,
                .len = sizeof(validatePlaneCount),
            },

            {
                .key = LwSciBufImageAttrKey_PlaneColorFormat,
                .value = &validateColorFmt,
                .len = sizeof(validateColorFmt),
            },

            {
                .key = LwSciBufImageAttrKey_PlaneColorStd,
                .value = &validateColorStd,
                .len = sizeof(validateColorStd),
            },

            {
                .key = LwSciBufImageAttrKey_PlaneBaseAddrAlign,
                .value = &validatePlaneBaseAddr,
                .len = sizeof(validatePlaneBaseAddr),
            },

            {
                .key = LwSciBufImageAttrKey_PlaneWidth,
                .value = &validatePlaneWidth,
                .len = sizeof(validatePlaneWidth),
            },

            {
                .key = LwSciBufImageAttrKey_PlaneHeight,
                .value = &validatePlaneHeight,
                .len = sizeof(validatePlaneHeight),
            },

            {
                .key = LwSciBufImageAttrKey_ScanType,
                .value = &validateScanType,
                .len = sizeof(validateScanType),
            },

            {
                .key = LwSciBufPyramidAttrKey_NumLevels,
                .value = &validateLevelCount,
                .len = sizeof(validateLevelCount),
            },

            {
                .key = LwSciBufPyramidAttrKey_Scale,
                .value = &validateScale,
                .len = sizeof(validateScale),
            },
        };

        for (auto keyVal : keyValPair) {
            auto validateList = peer.createAttrList(&error);
            ASSERT_EQ(error, LwSciError_Success);

            SET_ATTR(validateList.get(), LwSciBufGeneralAttrKey_Types, bufType);
            error = LwSciBufAttrListSetAttrs(validateList.get(), &keyVal, 1);
            ASSERT_EQ(error, LwSciError_Success);

            NEGATIVE_TEST();

            ASSERT_EQ(LwSciBufPeer::validateReconciled({validateList.get()},
                reconciledList.get(), &isReconciledListValid),
                LwSciError_ReconciliationFailed);

            ASSERT_FALSE(isReconciledListValid);
        }
    }
}
