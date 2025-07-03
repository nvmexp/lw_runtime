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
#include "lwsci_dgpu_only_test.h"
#include "lwsci_test_gpu_platform.h"
#include "lwscibuf_test_gpu_platform_helper.h"

class TestLwSciBufdGPUVidMem : public LwScidGpuOnlyTest,
                               public LwSciBufBasicTest,
                               public ::testing::WithParamInterface<
                                std::tuple<LwSciBufHwEngName, LwSciError>>
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();
        LwScidGpuOnlyTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        umd2AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd2AttrList.get(), nullptr);
    }

    virtual void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
        LwScidGpuOnlyTest::TearDown();

        umd1AttrList.reset();
        umd2AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
    std::shared_ptr<LwSciBufAttrListRec> umd2AttrList;
};

TEST_F(TestLwSciBufdGPUVidMem, IntraProcessdGPUvidmem)
{
    uint64_t rawBufSize = (128U * 1024U);

    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t alignment = (4U * 1024U);
        bool cpuAccessFlag = false;

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Size, rawBufSize);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testdGpuId);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
            testdGpuId);

        printf("test set attr \n");
        for (auto const& byte : testdGpuId.bytes) {
            printf("%x ", byte);
        }
        printf("\n");
    }

    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t alignment = (8U * 1024U);
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    }

    LwSciError error = LwSciError_Success;
    auto bufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    /* Allocation size check */
    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U;
    uint64_t len = 0U;
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    ASSERT_EQ(LwSciError_Success, testGpuMapping(dGpuTstResource, rmHandle,
        rawBufSize)) << "dGPU mapping test failed";
}

TEST_F(TestLwSciBufdGPUVidMem, dGPUvidmemSizeLessThanSmallPageSize)
{
    /*
     * This test-case covers issue raised in bug 3131794
     * Basically, when size less than small page size (=4k) is requested, the
     * vidmem allocation fails. We should roundup the size to small page size
     * and then perform allocation.
     * This test-case verifies that requesting the raw buffer size of less than
     * small page size results in size getting rounded up to small page size
     * and vidmem allocation succeeds.
     */

    /*
     * TODO: This test-case is cheetah specific since bug mentioned above is on
     * cheetah. Need to check if such round up of size is required for x86. If so,
     * the test-case should be extended for x86 too.
     */
    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t bufSize = (1 * 1024U);
        uint64_t alignment = (4U * 1024U);
        bool cpuAccessFlag = false;

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Size, bufSize);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testdGpuId);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
            testdGpuId);

        printf("test set attr \n");
        for (auto const& byte : testdGpuId.bytes) {
            printf("%x ", byte);
        }
        printf("\n");
    }

    LwSciError error = LwSciError_Success;
    auto bufObj =
        LwSciBufPeer::reconcileAndAllocate({umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
}

/**
 * This test verifies that if LwSciBufGeneralAttrKey_VidMem_GpuId is specified
 * along with LwSciBufInternalGeneralAttrKey_EngineArray then the reconciliation
 * passes only if the engineArray contains only PCIe engine. If any engine
 * other than PCIe is specified in LwSciBufInternalGeneralAttrKey_EngineArray
 * then reconciliation fails.
 */
TEST_P(TestLwSciBufdGPUVidMem, VidmemWithEngineArray)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t rawBufSize = (128U * 1024U);
    uint64_t alignment = (4U * 1024U);
    LwSciBufHwEngine engine{};

    auto params = GetParam();
    LwSciBufHwEngName engName = std::get<0>(params);
    LwSciError expectedError = std::get<1>(params);

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Size, rawBufSize);
    SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testdGpuId);
    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
        testdGpuId);
#if !defined(__x86_64__)
        error =
            LwSciBufHwEngCreateIdWithoutInstance(engName, &engine.rmModuleID);
        ASSERT_EQ(error, LwSciError_Success);
#else
        // the following field should be queried first by UMD
        engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
        engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
        engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;
#endif

        SET_INTERNAL_ATTR(umd1AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_EngineArray, engine);

    if (expectedError != LwSciError_Success) {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
        ASSERT_EQ(error, expectedError);
    } else {
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
        ASSERT_EQ(error, expectedError);

        // Assert that VidMem is reconciled
        LwSciBufPeer::verifyInternalAttr(
            reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray,
            LwSciBufMemDomain_Vidmem);
    }
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufdGPUVidMem, TestLwSciBufdGPUVidMem,
#if !defined(__x86_64__)
    ::testing::Values(std::make_tuple(LwSciBufHwEngName_Display,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_Isp,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_Vi,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_Csi,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_Vic,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_Gpu,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_MSENC,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_LWDEC,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_LWJPG,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_PVA,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_DLA,
                            LwSciError_ReconciliationFailed),
                      std::make_tuple(LwSciBufHwEngName_PCIe,
                            LwSciError_Success),
                      std::make_tuple(LwSciBufHwEngName_OFA,
                            LwSciError_ReconciliationFailed))
#else
    /*
     * On X86, LwSciBufHwEngName is not used. As such, pass dummy value as of
     * now and expect reconciliation to pass.
     */
    ::testing::Values(std::make_tuple(LwSciBufHwEngName_Display,
                            LwSciError_Success))
#endif
);

/**
 * This test verifies that if CPU access is requested and at the same time
 * allocation is requested from vidmem then the reconciliation fails.
 */
TEST_F(TestLwSciBufdGPUVidMem, VidmemWithCpuAccess)
{
    LwSciError error = LwSciError_Success;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t rawBufSize = (128U * 1024U);
    uint64_t alignment = (4U * 1024U);
    bool cpuAccessFlag = true;

    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Size, rawBufSize);
    SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testdGpuId);
    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
        testdGpuId);
    SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);

    {
        NEGATIVE_TEST();
        auto reconciledList =
            LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
