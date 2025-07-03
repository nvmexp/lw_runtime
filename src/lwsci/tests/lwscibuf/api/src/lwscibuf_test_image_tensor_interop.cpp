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

class TestLwSciBufImageTensor : public LwSciBufBasicTest
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

TEST_F(TestLwSciBufImageTensor, IntrathreadImagePLTensorNHWCProg)
{
    LwSciError error = LwSciError_Success;
    bool isReconciledListValid = false;

    {
        LwSciBufType bufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_PitchLinearType;
        uint64_t lrpad = 0, tbpad = 0, imageCount = 1;
        bool cpuaccess_flag = true;
        bool vpr = false;
        int32_t planecount = 1;

        LwSciBufAttrValColorFmt planecolorfmts[] = {LwSciColor_A8B8G8R8};
        LwSciBufAttrValColorStd planecolorstds[] = {LwSciColorStd_SRGB};
        LwSciBufAttrValImageScanType planescantype[] = {
            LwSciBufScan_ProgressiveType};

        int32_t plane_widths[] = {1920};
        int32_t plane_heights[] = {1080};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuaccess_flag);
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
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ImageCount,
                 imageCount);
    }

    {
        // public general attributes
        LwSciBufType bufType = LwSciBufType_Tensor;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        bool cpuaccess_flag = true;

        // public tensor attributes
        LwSciBufAttrValDataType datatype = LwSciDataType_Uint8;
        int32_t numDims = 4;
        uint64_t sizePerDim[4] = {1 /*N*/, 1080 /*H*/, 1920 /*W*/, 4 /*C*/};
        LwSciBufAttrValColorFmt colorFmt = LwSciColor_A8B8G8R8;

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuaccess_flag);
        SET_ATTR(umd2AttrList.get(), LwSciBufTensorAttrKey_DataType, datatype);
        SET_ATTR(umd2AttrList.get(), LwSciBufTensorAttrKey_NumDims, numDims);
        SET_ATTR(umd2AttrList.get(), LwSciBufTensorAttrKey_SizePerDim,
                 sizePerDim);
        SET_ATTR(umd2AttrList.get(), LwSciBufTensorAttrKey_PixelFormat,
                 colorFmt);

        // internal general attributes
        LwSciBufMemDomain memdomain = LwSciBufMemDomain_Sysmem;
        LwSciBufHwEngine engine[2] = {};
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                             &engine[0].rmModuleID);
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_PVA,
                                             &engine[1].rmModuleID);

        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memdomain);
    }

    auto bufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(bufObj, nullptr);

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
    *(uint64_t *)vaPtr = (uint64_t)0xC0DEC0DEC0DEC0DE;
    uint64_t testVal = *(uint64_t*)vaPtr;
    ASSERT_EQ(testVal, *(uint64_t *)vaPtr) << "CPU access failed";

    uint64_t size = GetMemorySize(rmHandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "Allocated size is not same as callwlated size."
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());
}
