/*
 * lwscibuf_test_tensor.cpp
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"

class TestLwSciBufTensor : public LwSciBufBasicTest
{
public:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        umdAttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umdAttrList.get(), nullptr);

        {
            // Set common attribute keys
            SET_ATTR(umdAttrList.get(), LwSciBufGeneralAttrKey_Types,
                     LwSciBufType_Tensor);

            SET_ATTR(umdAttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     true);
            SET_ATTR(umdAttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     LwSciBufAccessPerm_ReadWrite);

            SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_BaseAddrAlign,
                     (uint64_t)512U);
        }
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        umdAttrList.reset();
    }

    void testTensor(uint64_t bufSize)
    {
        LwSciError error = LwSciError_Success;

        auto bufObj =
            LwSciBufPeer::reconcileAndAllocate({umdAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        LwSciBufRmHandle rmHandle;
        void* va_ptr;
        uint64_t offset, len;
        ASSERT_EQ(
            LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
            LwSciError_Success)
            << "Failed to Get Lwrm Memhandle for the object";

        ASSERT_EQ(len, bufSize) << "Incorrect buffer size allocated";

        ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), &va_ptr),
                  LwSciError_Success)
            << "Failed to get cpu ptr";

        /* Verify CPU access */
        *(uint64_t *)va_ptr = (uint64_t)0xC0DEC0DEC0DEC0DE;
        uint64_t testval = *(uint64_t*)va_ptr;
        ASSERT_EQ(testval, *(uint64_t *)va_ptr) << "CPU access failed";

        uint64_t size = GetMemorySize(rmHandle);
        ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
            << "Allocated size is not same as callwlated size."
            << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());
    }

    std::shared_ptr<LwSciBufAttrListRec> umdAttrList;
};

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg1)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 5;
        // NCxHWx
        uint64_t sizes[] = {2, 8, 64, 32, 1};    // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(65536);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg2)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 5;
        // NCxHWx
        uint64_t sizes[] = {2, 2, 64, 32, 1};    // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(16384);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg3)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 5;
        // NCxHWx
        uint64_t sizes[] = {2, 8, 32, 32, 1};    // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Signed_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(32768);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg4)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 5;
        // NCxHWx
        uint64_t sizes[] = {2, 2, 32, 32, 1};    // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Signed_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(8192);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg5)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 4;
        // NCHW
        uint64_t sizes[] = {2, 8, 1, 1024};   // [N, C, H, W]
        uint32_t alignment[] = {1, 1, 32, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Signed_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(32768);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg6)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 4;
        // NCHW
        uint64_t sizes[] = {2, 2, 1, 1024};   // [N, C, H, W]
        uint32_t alignment[] = {1, 1, 32, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Signed_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(8192);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg7)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // NCxHWx
        uint64_t sizes[] = {2, 64, 2, 4, 16};    // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Float_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(32768);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg8)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 4;
        // NCHW
        uint64_t sizes[] = {2, 5, 30, 60};    // [N, C, H, W]
        uint32_t alignment[] = {1, 1, 32, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Float_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(38400);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg9)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 4;
        // NCHW
        uint64_t sizes[] = {2, 20, 30, 60};   // [N, C, H, W]
        uint32_t alignment[] = {1, 1, 32, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Float_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(153600);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg10)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 4;
        // NCHW
        uint64_t sizes[] = {2, 20, 1, 1024};  // [N, C, H, W]
        uint32_t alignment[] = {1, 1, 32, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Signed_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(81920);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg11)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Int16;
        uint32_t dimcount = 4;
        // NCHW
        uint64_t sizes[] = {2, 5, 1, 1024};   // [N, C, H, W]
        uint32_t alignment[] = {1, 1, 32, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Signed_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(20480);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg12)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L1 MNIST - 7
        uint64_t sizes[] = {1, 1, 28, 28, 32};   // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_Signed_A16;

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(50176);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg13)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L1 MNIST - 7
        uint64_t sizes[] = {1, 1, 12, 12, 32};   // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(9216);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg14)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L1 MNIST - 7
        uint64_t sizes[] = {1, 1, 1, 1, 32};     // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(64);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg15)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L1 MNIST - 7
        uint64_t sizes[] = {1, 2, 8, 8, 32};     // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(8192);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg16)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L1 MNIST - 7
        uint64_t sizes[] = {1, 2, 4, 4, 32};     // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(2048);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg17)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L1 MNIST - 7
        uint64_t sizes[] = {1, 16, 1, 1, 32};    // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(1024);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg18)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L1 MNIST - 7
        uint64_t sizes[] = {1, 1, 24, 24, 32};   // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(36864);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg19)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 4;
        // linux_dla_L0
        uint64_t sizes[] = {1, 480, 960, 4};  // [N, H, W, C]
        uint32_t alignment[] = {1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(3686400);
}

TEST_F(TestLwSciBufTensor, IntrathreadTensorProg20)
{
    {
        LwSciBufAttrValDataType datatype = LwSciDataType_Float16;
        uint32_t dimcount = 5;
        // linux_dla_L0
        uint64_t sizes[] = {1, 1, 30, 60, 32};   // [N, C, H, W, X]
        uint32_t alignment[] = {1, 1, 32, 1, 1}; // align H for 32

        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_NumDims, dimcount);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_SizePerDim, sizes);
        SET_ATTR(umdAttrList.get(), LwSciBufTensorAttrKey_AlignmentPerDim,
                 alignment);
    }

    testTensor(115200);
}
