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

class TestLwSciBufImageBlock : public LwSciBufBasicTest
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

TEST_F(TestLwSciBufImageBlock, IntraThreadImageBLIlace)
{
    {
        LwSciBufType bufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        uint64_t lrPad = 0U, tbPad = 100U;
        bool cpuAccessFlag = true;
        bool vpr = false;
        int32_t planeCount = 2U;

        LwSciBufAttrValColorFmt planeColorFmts[] = {LwSciColor_Y16,
                                                    LwSciColor_U8V8};
        LwSciBufAttrValColorStd planeColorStds[] = {LwSciColorStd_YcCbcCrc_SR,
                                                    LwSciColorStd_YcCbcCrc_SR};
        LwSciBufAttrValImageScanType planeScanType[] = {
            LwSciBufScan_InterlaceType};
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
        int32_t planeCount = 2U;
        int32_t planeWidths[] = {640U, 320U};
        int32_t planeHeights[] = {480U, 240U};
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufHwEngine engine{};
#if !defined(__x86_64__)
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                             &engine.rmModuleID);
#else
        // the following field should be queried first by UMD
        engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
        engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
        engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;
#endif

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneWidth,
                 planeWidths);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_PlaneHeight,
                 planeHeights);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);
        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
    }

    LwSciError error = LwSciError_Success;
    auto lwscibufobj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufRmHandle rmhandle = {0};
    uint64_t offset = 0U;
    uint64_t len = 0U;
    uint64_t testval = 0U;
    ASSERT_EQ(
        LwSciBufObjGetMemHandle(lwscibufobj.get(), &rmhandle, &offset, &len),
        LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    void* vaPtr = NULL;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(lwscibufobj.get(), &vaPtr),
              LwSciError_Success)
        << "Failed to get ptr";

    /* Verify CPU access */
    *(uint64_t *)vaPtr = (uint64_t)0xC0DEC0DEC0DEC0DEU;
    testval = *(uint64_t *)vaPtr;
    ASSERT_EQ(testval, *(uint64_t *)vaPtr) << "CPU access failed";

    /* Allocation size check */
    uint64_t size = GetMemorySize(rmhandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "Allocated size is not same as callwlated size."
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());
}
