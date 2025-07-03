/*
* Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include <iostream>
#include <string.h>

#include "lwscibuf_basic_test.h"
#include "lwscibuf_interprocess_test.h"

class AttributeListReconcile : public LwSciBufBasicTest
{
protected:
    void SetUp() override {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        listA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listA.get(), nullptr);

        listB = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listB.get(), nullptr);
    };

    void TearDown() override {
        LwSciBufBasicTest::TearDown();

        listA.reset();
        listB.reset();
    };

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;

    void setTensorAttributes(
        LwSciBufAttrList attrList)
    {
        SET_ATTR(attrList, LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
        SET_ATTR(attrList, LwSciBufTensorAttrKey_DataType, LwSciDataType_Uint8);
        uint32_t dimcount = 1U;
        SET_ATTR(attrList, LwSciBufTensorAttrKey_NumDims, dimcount);
        uint64_t sizes[] = { 2, 2, 64, 32, 1}; // [N, C, H, W, X]
        SET_ATTR(attrList, LwSciBufTensorAttrKey_SizePerDim, sizes);
        uint32_t alignment[] = { 1, 1, 32, 1, 1}; // align H for 32
        SET_ATTR(attrList, LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
        uint64_t baseAddrAlign = 512;
        SET_ATTR(attrList, LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);
    }
};

/**
* Test case 1: Set same attribute value of attributes of equal value policy
* in unreconciled list
*
*/
TEST_F(AttributeListReconcile, EqualValue1)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufGeneralAttrKey_Types, bufType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufRawBufferAttrKey_Size, size));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufRawBufferAttrKey_Align, align));
}

/**
* - Default value assignment for LwSciBufRawBufferAttrKey_Align
* Test case 1:  LwSciBufRawBufferAttrKey_Align is optional attribute for which
* LwSciBuf assigns default value if it is not specified by any of the
* unreconciled LwSciBufAttrLists ilwolved in the reconciliation. This test case
* verifies that the default value gets assigned to
* LwSciBufRawBufferAttrKey_Align in this scenario.
*/
TEST_F(AttributeListReconcile, DefaultRawBufferAlignmentValue)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024U;
    uint64_t alignment = 1U;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    /* Verify that default alignment of 1 is assigned to the attribute
     * if it is not specified.
     */
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufRawBufferAttrKey_Align, alignment));
}

/**
* Test case 1: Set an attribute value of attribute of equal value policy
* in only one list list
*
*/
TEST_F(AttributeListReconcile, EqualValue2)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufGeneralAttrKey_Types, bufType);
    LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufRawBufferAttrKey_Size, size);
    LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufRawBufferAttrKey_Align, align);
}

/**
 * Attempt reconciling the LwSciBufAttrList when only 1 Mandatory Attribute Key
 * is not set.
 */
TEST_F(AttributeListReconcile, MandatoryQualifiersImageMissingPlaneWidth)
{
    LwSciError error = LwSciError_Success;

    LwSciBufType bufType = LwSciBufType_Image;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    uint32_t planeWidths[2] = { 640U, 320U };
    uint32_t planeHeights[2] = { 480U, 240U };
    LwSciBufAttrValColorFmt planeColorFmts[1] = { LwSciColor_Bayer8RGGB };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_ProgressiveType };

    // Missing LwSciBufImageAttrKey_PlaneWidth
    {
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_TopPadding, align);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_Layout,
                 LwSciBufImage_BlockLinearType);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneCount, (uint32_t)1);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                 planeColorFmts);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_ScanType,
                 planeScanType);
    }

    {
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    }

    // LwSciBufImageAttrKey_PlaneWidth is always mandatory
    {
        NEGATIVE_TEST();

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(LwSciError_ReconciliationFailed, error);
    }
}

/**
 * Attempt reconciling the LwSciBufAttrList when only 1 Mandatory Attribute Key
 * is not set.
 */
TEST_F(AttributeListReconcile, MandatoryQualifiersImageMissingLayout)
{
    LwSciError error = LwSciError_Success;

    LwSciBufType bufType = LwSciBufType_Image;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    uint32_t planeWidths[2] = { 640U, 320U };
    uint32_t planeHeights[2] = { 480U, 240U };
    LwSciBufAttrValColorFmt planeColorFmts[1] = { LwSciColor_Bayer8RGGB };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_ProgressiveType };

    // Missing LwSciBufImageAttrKey_Layout
    {
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_TopPadding, align);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneCount, (uint32_t)1);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                 planeColorFmts);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_ScanType,
                 planeScanType);
    }

    {
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    }

    // LwSciBufImageAttrKey_Layout is always mandatory
    {
        NEGATIVE_TEST();

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(LwSciError_ReconciliationFailed, error);
    }
}

/**
* - Default value assignment for LwSciBufTensorAttrKey_BaseAddrAlign
* Test case 1:  LwSciBufTensorAttrKey_BaseAddrAlign is optional attribute for
* which LwSciBuf assigns default value if it is not specified by any of the
* unreconciled LwSciBufAttrLists ilwolved in the reconciliation and
* LwSciBufInternalGeneralAttrKey_EngineArray is not specified by any of the
* unreconciled LwSciBufAttrLists. This test case verifies that the default value
* gets assigned to LwSciBufTensorAttrKey_BaseAddrAlign in this scenario.
*/
TEST_F(AttributeListReconcile, DefaultTensorBaseAddrAlignmentValue)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_Tensor;
    LwSciBufAttrValDataType dataype = LwSciDataType_Uint8;
    uint32_t numDims =4U;
    uint64_t sizePerDim[4] = {1, 1080, 1920, 4};
    uint32_t alignment[4] = {1, 1, 1, 1};
    uint64_t baseAddrAlign = 1U;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataype);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, numDims);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizePerDim);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign));
}

/*
 * Attempt reconciling the LwSciBufAttrList when only 1 Conditional Attribute
 * Key is not set and it is required.
 */
TEST_F(AttributeListReconcile, ConditionalQualifierTensorPixelFormatRequired)
{
    LwSciError error = LwSciError_Success;

    uint32_t planeWidths[1] = { 1024U };
    uint32_t planeHeights[1] = { 1024U };
    LwSciBufAttrValColorFmt planeColorFmts[1] = { LwSciColor_A8B8G8R8 };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_InterlaceType };

    {
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Image);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_Layout,
                 LwSciBufImage_PitchLinearType);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneCount, (uint32_t)1);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                 planeColorFmts);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_ScanType, planeScanType);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_ImageCount, (uint64_t)1);

        // Set Internal attribute keys
        LwSciBufHwEngine dlaEngine;
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                             &dlaEngine.rmModuleID);
        LwSciBufHwEngine engineArray[] = { dlaEngine };
        LwSciBufPeer::setInternalAttr(listA.get(), LwSciBufInternalGeneralAttrKey_EngineArray,
                                engineArray);
    }

    {
        uint64_t numImages = 1U;
        uint64_t height = 1024U;
        uint64_t width = 1024U;
        uint64_t channels = 4U;

        uint64_t nhwcTuple[4] = {numImages, height, width, channels};

        // Missing LwSciBufTensorAttrKey_PixelFormat
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_Tensor);
        SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType,
                 LwSciDataType_Uint8);
        SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, (uint32_t)4U);
        SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, nhwcTuple);
    }

    // LwSciBufTensorAttrKey_PixelFormat is mandatory in Image/Tensor.
    {
        NEGATIVE_TEST();

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(LwSciError_ReconciliationFailed, error);
    }
}

/**
 * Attempt reconciling the LwSciBufAttrList when only 1 Conditional Attribute
 * Key is not set and it is not required.
 */
TEST_F(AttributeListReconcile, ConditionalQualifierTensorPixelFormatOptional)
{
    LwSciError error = LwSciError_Success;

    {
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_Tensor);
    }

    {
        uint64_t numImages = 1U;
        uint64_t height = 1024U;
        uint64_t width = 1024U;
        uint64_t channels = 4U;

        uint64_t nhwcTuple[4] = {numImages, height, width, channels};

        uint32_t alignment[4] = { 1024U, 1024U, 1024U, 1024U };

        // Missing LwSciBufTensorAttrKey_PixelFormat
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_Tensor);
        SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType,
                 LwSciDataType_Uint8);
        SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, (uint32_t)4U);
        SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, nhwcTuple);
        SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    }

    // LwSciBufTensorAttrKey_PixelFormat is optional when not Image/Tensor.
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);
}

/**
 * Reconciliation should be successful when listA contains all the Attribute
 * Keys.
 */
TEST_F(AttributeListReconcile, QualifierCommutativityLeft)
{
    LwSciError error = LwSciError_Success;

    LwSciBufType bufType = LwSciBufType_Image;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    uint32_t planeWidths[2] = { 640U, 320U };
    uint32_t planeHeights[2] = { 480U, 240U };
    LwSciBufAttrValColorFmt planeColorFmts[1] = { LwSciColor_Bayer8RGGB };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_ProgressiveType };

    // Attempt reconciling when listA contains all the Attribute Keys
    {
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_TopPadding, align);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_Layout,
                 LwSciBufImage_BlockLinearType);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneCount, (uint32_t)1);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                 planeColorFmts);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
        SET_ATTR(listA.get(), LwSciBufImageAttrKey_ScanType,
                 planeScanType);
    }

    {
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    }

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);
}

/**
 * Reconciliation should be successful when listB contains all the Attribute
 * Keys.
 */
TEST_F(AttributeListReconcile, QualifierCommutativityRight)
{
    LwSciError error = LwSciError_Success;

    LwSciBufType bufType = LwSciBufType_Image;
    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    uint32_t planeWidths[2] = { 640U, 320U };
    uint32_t planeHeights[2] = { 480U, 240U };
    LwSciBufAttrValColorFmt planeColorFmts[1] = { LwSciColor_Bayer8RGGB };
    LwSciBufAttrValImageScanType planeScanType[1] = { LwSciBufScan_ProgressiveType };

    // Attempt reconciling when listB contains all the Attribute Keys
    {
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    }

    {
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listB.get(), LwSciBufImageAttrKey_TopPadding, align);
        SET_ATTR(listB.get(), LwSciBufImageAttrKey_Layout,
                 LwSciBufImage_BlockLinearType);
        SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneCount, (uint32_t)1);
        SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                 planeColorFmts);
        SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneWidth, planeWidths);
        SET_ATTR(listB.get(), LwSciBufImageAttrKey_PlaneHeight, planeHeights);
        SET_ATTR(listB.get(), LwSciBufImageAttrKey_ScanType,
                 planeScanType);
    }

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);
}

TEST_F(AttributeListReconcile, AlignmentValue)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = 128 * 1024;
    uint64_t align1 = 4 * 1024;
    uint64_t align2 = 8 * 1024;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align1);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align2);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufRawBufferAttrKey_Align, align2);
}

TEST_F(AttributeListReconcile, ValueSet)
{
    LwSciError error = LwSciError_Success;
    uint64_t size = 1024U * 4U;
    bool isReconciledListValid = false;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    LwSciBufAttrValAccessPerm perm1 = LwSciBufAccessPerm_ReadWrite;
    LwSciBufAttrValAccessPerm perm2 = LwSciBufAccessPerm_Readonly;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm1);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm2);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufGeneralAttrKey_ActualPerm, perm1);
}

TEST_F(AttributeListReconcile, TrueValue)
{
    LwSciError error = LwSciError_Success;
    uint64_t size = 1024U * 4U;
    bool isReconciledListValid = false;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    bool cpuaccess_true = true;
    bool cpuaccess_false = false;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, cpuaccess_true);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
             cpuaccess_false);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyAttr(reconciledList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, cpuaccess_true);
}

TEST_F(AttributeListReconcile, MemDomain_NoPreference_Default)
{
    LwSciError error = LwSciError_Success;
    uint64_t size = 1024U * 4U;
    bool isReconciledListValid = false;

    // List A has no preferences
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);

    // List B has no preferences
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // Since List A and List B have no preferences, we choose SysMem by default
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, LwSciBufMemDomain_Sysmem);
}

TEST_F(AttributeListReconcile, MemDomain_NoPreference_Sysmem)
{
    LwSciError error = LwSciError_Success;
    uint64_t size = 1024U * 4U;
    bool isReconciledListValid = false;

    LwSciBufMemDomain memdomain[] = { LwSciBufMemDomain_Sysmem };

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_INTERNAL_ATTR(listA.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, memdomain);

    // List B has no preferences
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // Since List B has no preferences, we choose List A's Memory Domain
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, LwSciBufMemDomain_Sysmem);
}

#if (LW_IS_SAFETY == 0) && !defined(__x86_64__)
TEST_F(AttributeListReconcile, MemDomain_NoPreference_CVSRam)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    // List A has no preference
    setTensorAttributes(listA.get());

    // List B wants CVSRam
    setTensorAttributes(listB.get());
    LwSciBufMemDomain listBMemdomain[] = { LwSciBufMemDomain_Cvsram };
    SET_INTERNAL_ATTR(listB.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listBMemdomain);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // List B wants CVSRam
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, LwSciBufMemDomain_Cvsram);
}
#endif

#if (LW_IS_SAFETY == 0) && !defined(__x86_64__)
TEST_F(AttributeListReconcile, MemDomain_TensorCVSRam)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    // List A wants SysMem
    LwSciBufMemDomain listAMemdomain[] = { LwSciBufMemDomain_Cvsram };
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
    SET_INTERNAL_ATTR(listA.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listAMemdomain);

    // List B wants CVSRam
    setTensorAttributes(listB.get());
    LwSciBufMemDomain listBMemdomain[] = { LwSciBufMemDomain_Cvsram };
    SET_INTERNAL_ATTR(listB.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listBMemdomain);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    // List A and List B both want CVSRam
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, LwSciBufMemDomain_Cvsram);
}
#endif

#if (LW_IS_SAFETY == 0) && !defined(__x86_64__)
TEST_F(AttributeListReconcile, MemDomain_TensorCVSRam_Conflict)
{
    LwSciError error = LwSciError_Success;

    // List A wants SysMem
    LwSciBufMemDomain listAMemdomain[] = { LwSciBufMemDomain_Sysmem };
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
    SET_INTERNAL_ATTR(listA.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listAMemdomain);

    // List B wants CVSRam
    setTensorAttributes(listB.get());
    LwSciBufMemDomain listBMemdomain[] = { LwSciBufMemDomain_Cvsram };
    SET_INTERNAL_ATTR(listB.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listBMemdomain);

    {
        NEGATIVE_TEST();
        // This should fail since List A and List B disagree on the
        // LwSciBufInternalGeneralAttrKey_MemDomainArray key
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(LwSciError_ReconciliationFailed, error);
    }
}
#endif

#if (LW_IS_SAFETY == 0) && !defined(__x86_64__)
TEST_F(AttributeListReconcile, MemDomain_Cvsram_Multiple)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    // List A is okay with SysMem or CVSRam
    LwSciBufMemDomain listAMemdomain[] = { LwSciBufMemDomain_Sysmem, LwSciBufMemDomain_Cvsram };
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
    SET_INTERNAL_ATTR(listA.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listAMemdomain);

    // List B wants CVSRam
    setTensorAttributes(listB.get());
    LwSciBufMemDomain listBMemdomain[] = { LwSciBufMemDomain_Cvsram };
    SET_INTERNAL_ATTR(listB.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listBMemdomain);

    // List A and List B both want CVSRam
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, LwSciBufMemDomain_Cvsram);
}
#endif

#if (LW_IS_SAFETY == 0) && !defined(__x86_64__)
TEST_F(AttributeListReconcile, MemDomain_Priority_Cvsram)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    // List A is okay with SysMem or CVSRam
    LwSciBufMemDomain listAMemdomain[] = { LwSciBufMemDomain_Sysmem, LwSciBufMemDomain_Cvsram };
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
    SET_INTERNAL_ATTR(listA.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listAMemdomain);

    // List B is also okay with SysMem and CVSRam
    setTensorAttributes(listB.get());
    LwSciBufMemDomain listBMemdomain[] = { LwSciBufMemDomain_Cvsram , LwSciBufMemDomain_Sysmem };
    SET_INTERNAL_ATTR(listB.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listBMemdomain);

    // List A and List B both are okay with both, however we take the one with
    // higher priority (CVSRam)
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(LwSciError_Success, error);
    ASSERT_NE(reconciledList, nullptr);

    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    LwSciBufPeer::verifyInternalAttr(reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, LwSciBufMemDomain_Cvsram);
}
#endif

#if (LW_IS_SAFETY == 0)
TEST_F(AttributeListReconcile, MemDomain_Missing_GpuId)
{
    LwSciError error = LwSciError_Success;

    // List A wants Vidmem
    LwSciBufMemDomain listAMemdomain[] = { LwSciBufMemDomain_Vidmem };
    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
    SET_INTERNAL_ATTR(listA.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray, listAMemdomain);

    // List B is okay with anything
    setTensorAttributes(listB.get());

    {
        NEGATIVE_TEST();

        // We didn't specify the GPU ID to use to allocate from Vidmem.
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_NE(LwSciError_Success, error);
        ASSERT_EQ(reconciledList, nullptr);
    }
}
#endif

class AttributeListReconcileInterProc : public LwSciBufInterProcessTest
{
};

/**
* Test case 1: Value is equivalent of true, if any of the slots of all
* the attribute lists that are owned by the reconciler in the provided
* unreconciled attribute lists is set to a value equivalent of true.
*/
TEST_F(AttributeListReconcileInterProc, TrueValueForOwner1)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        uint64_t size = 1024U * 4U;

        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_0");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
        peer->setAttr(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);

        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_1");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, false);

        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Verify value of NeedCpuAccess is the same as the reconciler
        peer->verifyAttr(reconciledList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, false);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

/**
* Test case 2: Value is equivalent of false, if none of the slots of all
* the attribute lists that are owned by the reconciler in the provided
* unreconciled attribute lists is set to a value equivalent of true.
*/
TEST_F(AttributeListReconcileInterProc, TrueValueForOwner2)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        uint64_t size = 1024U * 4U;

        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_0");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
        peer->setAttr(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, false);

        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_1");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_RawBuffer);
        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);

        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Verify value of NeedCpuAccess is the same as the reconciler
        peer->verifyAttr(reconciledList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

/**
* Test case to check reconciliation for LwSciBufGeneralAttrKey_ActualPerm
* ActualPerm is reconciled by the reconciler by taking RequiredPerm requests
* from all the peers and providing max value based on that.
*/
TEST_F(AttributeListReconcileInterProc, ActualPermReconciliation)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        uint64_t size = 1024U * 4U;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_0");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_Types,
            LwSciBufType_RawBuffer);
        peer->setAttr(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
            perm);

        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_Readonly;
        uint64_t size = 1024U * 4U;

        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("ipc_test_1");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_Types,
            LwSciBufType_RawBuffer);
        peer->setAttr(list.get(), LwSciBufRawBufferAttrKey_Size, size);
        peer->setAttr(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
            perm);

        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto reconciledList =
            LwSciBufPeer::attrListReconcile({list.get(), upstreamList.get()},
                &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Verify value of ActualPerm is callwlated by considering requests
        // from all peers. In this case, value should be
        // LwSciBufAccessPerm_ReadWrite.
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
            LwSciBufGeneralAttrKey_ActualPerm, LwSciBufAccessPerm_ReadWrite));
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciBufInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
