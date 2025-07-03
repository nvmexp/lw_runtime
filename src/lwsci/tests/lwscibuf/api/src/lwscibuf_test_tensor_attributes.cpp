/*
 * Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"
#include "lwscibuf_interprocess_test.h"

#include <array>

class LwSciBufTestTensorAttributes : public LwSciBufBasicTest
{
public:
    void SetUp() override
    {
        LwSciBufBasicTest::SetUp();
        LwSciError error = LwSciError_Success;
        listA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listA.get(), nullptr);
        listB = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listB.get(), nullptr);
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_Tensor);
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types,
                 LwSciBufType_Tensor);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
        listA.reset();
        listB.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;
};

TEST_F(LwSciBufTestTensorAttributes, TensorAttrsNotSet)
{
    LwSciError error = LwSciError_Success;
    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, SetTensorAttrToWrongBufferType)
{
    LwSciError error = LwSciError_Success;
    auto list = peer.createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(list.get(), nullptr);

    {
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::setAttr(list.get(),
                                        LwSciBufTensorAttrKey_DataType,
                                        LwSciDataType_Int4),
                  LwSciError_BadParameter);
    }
}

TEST_F(LwSciBufTestTensorAttributes, DataType)
{
    LwSciError error = LwSciError_Success;

    std::array<LwSciBufAttrValDataType, 10> validDataTypes = {
        LwSciDataType_Int4,    LwSciDataType_Uint4,    LwSciDataType_Int8,
        LwSciDataType_Uint8,   LwSciDataType_Int16,    LwSciDataType_Uint16,
        LwSciDataType_Int32,   LwSciDataType_Uint32,   LwSciDataType_Float16,
        LwSciDataType_Float32
    };

    std::array<LwSciBufAttrValDataType, 3> ilwalidDataTypes = {
        LwSciDataType_FloatISP, LwSciDataType_Bool,
        LwSciDataType_UpperBound
    };

    for (auto const& dataType : validDataTypes) {
        auto list = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(list.get(), nullptr);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_DataType, dataType);
    }

    for (auto const& dataType : ilwalidDataTypes) {
        auto list = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(list.get(), nullptr);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::setAttr(
                    list.get(), LwSciBufTensorAttrKey_DataType, dataType),
                  LwSciError_BadParameter) << "DataType: " << dataType;
    }
}

TEST_F(LwSciBufTestTensorAttributes, NumDimsOutsideRange)
{
    NEGATIVE_TEST();
    uint64_t dimcount = 0;
    ASSERT_EQ(LwSciBufPeer::setAttr(listA.get(),
                                    LwSciBufTensorAttrKey_NumDims,
                                    dimcount),
              LwSciError_BadParameter);

    dimcount = 9;
    ASSERT_EQ(LwSciBufPeer::setAttr(listA.get(),
                                    LwSciBufTensorAttrKey_NumDims,
                                    dimcount),
              LwSciError_BadParameter);
}

TEST_F(LwSciBufTestTensorAttributes, NumDimsNotEqual)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes1[] = {2, 8, 64, 32};
    uint64_t sizes2[] = {2, 8, 64, 32, 1};
    uint32_t alignment1[] = {1, 1, 32, 1};
    uint32_t alignment2[] = {1, 1, 32, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount1 = 4;
    uint32_t dimcount2 = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount1);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes1);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment1);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount2);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes2);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment2);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, SizePerDimNotEqual)
{
    LwSciError error = LwSciError_Success;
    uint32_t dimcount = 5;
    uint64_t sizes1[] = {2, 8, 64, 32, 1};
    uint64_t sizes2[] = {4, 16, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes1);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes2);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, AlignmentPerDimReconciliation)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment1[] = {1, 1, 32, 32, 32};
    uint32_t alignment2[] = {32, 32, 64, 1, 1};
    uint32_t reconciledAlignment[] = {32, 32, 64, 32, 32};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment1);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment2);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_AlignmentPerDim,
                                       reconciledAlignment),
              true);
}

TEST_F(LwSciBufTestTensorAttributes, AlignmentPerDimNotPowerOfTwo)
{
    LwSciError error = LwSciError_Success;
    uint32_t dimcount = 5;
    uint32_t alignment[] = {1, 1, 32, 32, 32 + 1};

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);

    {
        NEGATIVE_TEST();
        ASSERT_EQ(LwSciBufPeer::setAttr(listA.get(),
                                        LwSciBufTensorAttrKey_AlignmentPerDim,
                                        alignment),
                  LwSciError_BadParameter);
    }
}

TEST_F(LwSciBufTestTensorAttributes, BaseAddrAlignReconciliation)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 32, 32};
    uint64_t baseAddrAlign1 = 512;
    uint64_t baseAddrAlign2 = 1024;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign1);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign2);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_BaseAddrAlign,
                                       baseAddrAlign2),
              true);
}

TEST_F(LwSciBufTestTensorAttributes, BaseAddrAlignNotPowerOfTwo)
{
    NEGATIVE_TEST();
    uint64_t baseAddrAlign = 512 + 1;
    ASSERT_EQ(LwSciBufPeer::setAttr(listA.get(),
                                    LwSciBufTensorAttrKey_BaseAddrAlign,
                                    baseAddrAlign),
              LwSciError_BadParameter);
}

TEST_F(LwSciBufTestTensorAttributes, DataTypeNotEqual)
{
    LwSciError error = LwSciError_Success;
    uint32_t dimcount = 5;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dataType1 = LwSciDataType_Int16;
    uint32_t dataType2 = LwSciDataType_Int32;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType1);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType2);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, NumDimsSetInOneList)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_NumDims, dimcount),
              true);
}

TEST_F(LwSciBufTestTensorAttributes, SizePerDimSetInOneList)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_SizePerDim, sizes),
              true);
}

TEST_F(LwSciBufTestTensorAttributes, DateTypeSetInOneList)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_DataType,
                                       LwSciDataType_Int16),
            true);
}

TEST_F(LwSciBufTestTensorAttributes, AlignmentPerDimSetInOneList)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_AlignmentPerDim,
                                       alignment),
              true);
}

TEST_F(LwSciBufTestTensorAttributes, NormalOperation)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t strides[] = {32768, 4096, 64, 2, 2};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;
    uint64_t size = 65536;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufGeneralAttrKey_Types,
                                       LwSciBufType_Tensor),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_DataType,
                                       LwSciDataType_Int16),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_NumDims, dimcount),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_SizePerDim, sizes),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_AlignmentPerDim,
                                       alignment),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_BaseAddrAlign,
                                       baseAddrAlign),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_StridesPerDim,
                                       strides),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_Size, size),
              true);
}

TEST_F(LwSciBufTestTensorAttributes, AlignmentPerDimNotSet)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, SizePerDimNotSet)
{
    LwSciError error = LwSciError_Success;
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, NumDimsNotSet)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, DataTypeNotSet)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, SizePerDimLargerThanNumDims)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 1};
    uint32_t alignment[] = {1, 1, 32, 1};
    uint64_t strides[] = {32768, 4096, 64, 2};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 4;
    uint32_t dataType = LwSciDataType_Int16;
    uint64_t size = 65536;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_DataType,
                                       LwSciDataType_Int16),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_NumDims, dimcount),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_SizePerDim, sizes),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_AlignmentPerDim,
                                       alignment),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_BaseAddrAlign,
                                       baseAddrAlign),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_StridesPerDim,
                                       strides),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_Size, size),
              true);
}

/**
 * Length of LwSciBufTensorAttrKey_SizePerDim is smaller than
 * LwSciBufTensorAttrKey_NumDims and LwSciBufTensorAttrKey_AlignmentPerDim
 */
TEST_F(LwSciBufTestTensorAttributes, SizePerDimSmallerThanNumDims)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

TEST_F(LwSciBufTestTensorAttributes, AlignmentPerDimLargerThanNumDims)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32};
    uint32_t alignment[] = {1, 1, 32, 1, 1};
    uint64_t strides[] = {32768, 4096, 64, 2};
    uint64_t baseAddrAlign = 512;
    uint32_t dimcount = 4;
    uint32_t dataType = LwSciDataType_Int16;
    uint64_t size = 65536;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddrAlign);

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_DataType,
                                       LwSciDataType_Int16),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_NumDims, dimcount),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_SizePerDim, sizes),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_AlignmentPerDim,
                                       alignment),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_BaseAddrAlign,
                                       baseAddrAlign),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_StridesPerDim,
                                       strides),
              true);
    ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufTensorAttrKey_Size, size),
              true);
}

/**
 * Length of LwSciBufTensorAttrKey_AlignmentPerDim is smaller than
 * LwSciBufTensorAttrKey_NumDims and LwSciBufTensorAttrKey_AlignmentPerDim
 */
TEST_F(LwSciBufTestTensorAttributes, AlignmentPerDimSmallerThanNumDims)
{
    LwSciError error = LwSciError_Success;
    uint64_t sizes[] = {2, 8, 64, 32, 32};
    uint32_t alignment[] = {1, 1, 32, 1};
    uint32_t dimcount = 5;
    uint32_t dataType = LwSciDataType_Int16;

    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listA.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);

    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_DataType, dataType);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
    SET_ATTR(listB.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

class LwSciBufTestTensorReconciliation : public LwSciBufInterProcessTest
{
};

TEST_F(LwSciBufTestTensorReconciliation, Validate)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_A_B");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        uint64_t sizes[] = {2, 8, 64, 32, 1};
        uint32_t alignment[] = {1, 1, 32, 1, 1};
        uint32_t dimcount = 5;
        uint32_t dataType = LwSciDataType_Int16;
        uint64_t baseAddressAlign = 512;

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_DataType, dataType);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddressAlign);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                 LwSciBufAccessPerm_ReadWrite);

        auto upstreamListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto upstreamList =
            peer->importUnreconciledList(upstreamListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto bufObj = peer->reconcileAndAllocate(
            {list.get(), upstreamList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        LwSciBufAttrList objList;
        ASSERT_EQ(LwSciBufObjGetAttrList(bufObj.get(), &objList),
                  LwSciError_Success);

        auto objListDescBuf = peer->exportReconciledList(objList, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(objListDescBuf), LwSciError_Success);

        auto objDescBuf = peer->exportBufObj(
            bufObj.get(), LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(objDescBuf), LwSciError_Success);

        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);

    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciBufIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscibuf_ipc_B_A");
        auto list = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        uint64_t sizes[] = {2, 8, 64, 32, 1};
        uint32_t alignment[] = {1, 1, 32, 1, 1};
        uint32_t dimcount = 5;
        uint32_t dataType = LwSciDataType_Int16;
        uint64_t baseAddressAlign = 512;
        uint64_t strides[] = {32768, 4096, 64, 2, 2};
        uint64_t size = 65536;

        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_Types, LwSciBufType_Tensor);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_DataType, dataType);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_AlignmentPerDim, alignment);
        SET_ATTR(list.get(), LwSciBufTensorAttrKey_BaseAddrAlign, baseAddressAlign);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_NeedCpuAccess, true);
        SET_ATTR(list.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                 LwSciBufAccessPerm_ReadWrite);

        auto listDescBuf = peer->exportUnreconciledList({list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        auto reconciledListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(reconciledListDescBuf,
                                                         {list.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto bufObjDescBuf =
            peer->recvExportDesc<LwSciBufObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        peer->importBufObj(bufObjDescBuf.get(), reconciledList.get(),
                           LwSciBufAccessPerm_ReadWrite, &error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                       LwSciBufGeneralAttrKey_Types,
                                       LwSciBufType_Tensor),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                           LwSciBufTensorAttrKey_DataType,
                                           LwSciDataType_Int16),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                           LwSciBufTensorAttrKey_NumDims,
                                           dimcount),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                           LwSciBufTensorAttrKey_SizePerDim,
                                           sizes),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(
                      reconciledList.get(),
                      LwSciBufTensorAttrKey_AlignmentPerDim, alignment),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                           LwSciBufTensorAttrKey_BaseAddrAlign,
                                           baseAddressAlign),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                           LwSciBufTensorAttrKey_StridesPerDim,
                                           strides),
                  true);
        ASSERT_EQ(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                           LwSciBufTensorAttrKey_Size, size),
                  true);

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
