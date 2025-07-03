/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "gtest/gtest.h"

#include <memory>
#include "lwscibuf_basic_test.h"
#include "lwsci_igpu_or_dgpu_test.h"

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class AttributeValidationCommon
{
public:
    void SetUp(LwSciBufPeer& peer)
    {
        LwSciError error = LwSciError_Success;

        attrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(attrList.get(), nullptr);
    }

    void TearDown()
    {
        attrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> attrList;
};

class AttributeKeyValueValidation : public LwSciBufBasicTest,
                                    public AttributeValidationCommon
{
public:
    void SetUp() override
    {
        LwSciBufBasicTest::SetUp();
        AttributeValidationCommon::SetUp(peer);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
        AttributeValidationCommon::TearDown();
    }
};

/**
 * Set a valid value for LwSciBufImageAttrKey_ImageCount.
 */
TEST_F(AttributeKeyValueValidation, ImageCount_ValidValue)
{
    uint64_t imageCount = 1U;

    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);

    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);

    SET_ATTR(attrList.get(), LwSciBufImageAttrKey_ImageCount, imageCount);
}

/**
 * Set a invalid value for LwSciBufImageAttrKey_ImageCount.
 */
TEST_F(AttributeKeyValueValidation, ImageCount_IlwalidValue)
{
    uint64_t imageCount = 2U;

    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);

    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);

    {
        NEGATIVE_TEST();

        ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                                        LwSciBufImageAttrKey_ImageCount,
                                        imageCount),
                  LwSciError_BadParameter);
    }
}

/**
 * Set a valid value for LwSciBufImageAttrKey_PlaneCount.
 */
TEST_F(AttributeKeyValueValidation, ImagePlaneCount_ValidValue)
{
    uint32_t planeCount = 0U;
    LwSciError error = LwSciError_Success;

    for (planeCount = 1U; planeCount <= (uint32_t)LW_SCI_BUF_IMAGE_MAX_PLANES;
        planeCount++) {
        attrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(attrList.get(), nullptr);

        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_Image);

        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);

        SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneCount, planeCount);
    }
}

/**
 * Set an invalid value for LwSciBufImageAttrKey_PlaneCount.
 */
TEST_F(AttributeKeyValueValidation, ImagePlaneCount_IlwalidValue)
{
    std::vector<uint32_t> planeCount = {0, LW_SCI_BUF_IMAGE_MAX_PLANES + 1};
    LwSciError error = LwSciError_Success;

    for (uint32_t ilwalidPlaneCount : planeCount) {
        attrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(attrList.get(), nullptr);

        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_Image);

        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);

        {
            NEGATIVE_TEST();

            ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                LwSciBufImageAttrKey_PlaneCount,
                ilwalidPlaneCount),
                LwSciError_BadParameter);
        }
    }
}

/**
 * Set an invalid value for an Attribute Key that supports multiple values
 * being set (arrays) using LwSciBufGeneralAttrKey_Types.
 */
TEST_F(AttributeKeyValueValidation, AttributeKeyArray_IlwalidValue)
{
    // LwSciBufType_General is not a valid LwSciBufType
    LwSciBufType ilwalidBufTypes[2] = {
        LwSciBufType_RawBuffer,
        LwSciBufType_General
    };

    {
        NEGATIVE_TEST();

        ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                                        LwSciBufGeneralAttrKey_Types,
                                        ilwalidBufTypes),
                  LwSciError_BadParameter);
    }
}

/**
 * Set a valid value for an Attribute Key that supports multiple values
 * being set (arrays) using LwSciBufGeneralAttrKey_Types.
 */
TEST_F(AttributeKeyValueValidation, AttributeKeyArray_ValidValue)
{
    LwSciBufType validBufTypes[2] = {
        LwSciBufType_RawBuffer,
        LwSciBufType_Image,
    };

    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, validBufTypes);

    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
}

/**
 * Set a valid value for an Attribute Key that requires separate logic when
 * importing.
 */
TEST_F(AttributeKeyValueValidation, TensorKeyArray_ValidValue)
{
    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);

    SET_ATTR(attrList.get(), LwSciBufTensorAttrKey_PixelFormat,
             LwSciColor_Bayer8RGGB);
}

/**
 * Set an invalid value for an Attribute Key that requires separate logic when
 * importing.
 */
TEST_F(AttributeKeyValueValidation, TensorKeyArray_IlwalidValue)
{
    LwSciBufType validBufTypes[2] = {
        LwSciBufType_Tensor,
        LwSciBufType_Image,
    };

    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, validBufTypes);

    {
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                                        LwSciBufTensorAttrKey_PixelFormat,
                                        LwSciColor_LowerBound),
                  LwSciError_BadParameter);
    }
}

/**
 * Set an invalid value for an Attribute Key that requires separate logic when
 * importing.
 */
TEST_F(AttributeKeyValueValidation, TensorKeyArray_IlwalidValue2)
{
    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);

    {
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                                        LwSciBufTensorAttrKey_PixelFormat,
                                        LwSciColor_LowerBound),
                  LwSciError_BadParameter);
    }
}

/**
 * Set an invalid value for LwSciBufImageAttrKey_PlaneBaseAddrAlign
 */
TEST_F(AttributeKeyValueValidation, PlaneBaseAddrAlign_Ilwalid)
{
    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);
    {
        NEGATIVE_TEST();
        // This is not a power of 2
        uint32_t ilwalidPlaneBaseAddr = 3U;

        ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                                        LwSciBufImageAttrKey_PlaneBaseAddrAlign,
                                        ilwalidPlaneBaseAddr),
                  LwSciError_BadParameter);
    }
}

/**
 * Set an invalid value for LwSciBufRawBufferAttrKey_Size
 */
TEST_F(AttributeKeyValueValidation, RawBufferSize_Ilwalid)
{
    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types,
        LwSciBufType_RawBuffer);
    {
        NEGATIVE_TEST();
        // 0 size is not acceptable
        uint64_t size = 0U;

        ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                                        LwSciBufRawBufferAttrKey_Size,
                                        size),
                  LwSciError_BadParameter);
    }
}

class AttributeKeyValueValidationGpu :  public LwSciiGpuOrdGpuTest,
                                        public AttributeValidationCommon,
                                        public LwSciBufBasicTest
{
public:
    void SetUp() override
    {
        LwSciBufBasicTest::SetUp();
        LwSciiGpuOrdGpuTest::SetUp();
        AttributeValidationCommon::SetUp(peer);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
        LwSciiGpuOrdGpuTest::TearDown();
        AttributeValidationCommon::TearDown();
    }
};

/**
 * Set an invalid value for LwSciBufGeneralAttrKey_EnableGpuCompression
 * attribute.
 */
TEST_F(AttributeKeyValueValidationGpu, GpuCompressionAttr_IlwalidValue1)
{
    SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);

    {
        LwSciBufAttrValGpuCompression gpuCompression = {
            .gpuId = testGpuId,
            .compressionType = LwSciBufCompressionType_None
        };

        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::setAttr(attrList.get(),
                                    LwSciBufGeneralAttrKey_EnableGpuCompression,
                                    gpuCompression),
                  LwSciError_BadParameter);
    }
}
