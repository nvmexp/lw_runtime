/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"
#include "lwsci_igpu_or_dgpu_test.h"
#include "lwsci_igpu_and_dgpu_test.h"

class LwSciBufGpuCompression : public LwSciBufBasicTest
{
public:
    static inline void setupAttrList(
        std::shared_ptr<LwSciBufAttrListRec> attrList,
        LwSciBufType bufType) {
        if (bufType == LwSciBufType_RawBuffer) {
            uint64_t rawBufSize = (128U * 1024U);
            uint64_t alignment = (4U * 1024U);

            SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(attrList.get(), LwSciBufRawBufferAttrKey_Size, rawBufSize);
            SET_ATTR(attrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        }

        if (bufType == LwSciBufType_Image) {
            uint32_t planeCount = 1U;
            LwSciBufAttrValColorFmt colorFmt = LwSciColor_A8B8G8R8;
            uint32_t width = 1080U;
            uint32_t height = 1920U;

            SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneCount,
                planeCount);
            SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneColorFormat,
                colorFmt);
            SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneWidth, width);
            SET_ATTR(attrList.get(), LwSciBufImageAttrKey_PlaneHeight, height);
        }
    }

    virtual void SetUp() override
    {
        LwSciBufBasicTest::SetUp();
    }

    virtual void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
    }
};

class TestLwSciBufGpuCompression : public LwSciiGpuOrdGpuTest,
                                       public LwSciBufGpuCompression
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufGpuCompression::SetUp();
        LwSciiGpuOrdGpuTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        umd2AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd2AttrList.get(), nullptr);

    }

    virtual void TearDown() override
    {
        LwSciBufGpuCompression::TearDown();
        LwSciiGpuOrdGpuTest::TearDown();

        umd1AttrList.reset();
        umd2AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
    std::shared_ptr<LwSciBufAttrListRec> umd2AttrList;
};

/**
 * Positive test
 * Set GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request GPU compression for GPU via
 * LwSciBufGeneralAttrKey_EnableGpuCompression.
 * Reconciliation should succeed. The final reconciled value will depend on
 * whether underlying GPU HW grants compression.
 */
TEST_F(TestLwSciBufGpuCompression, GpuCompressionPositiveTest1)
{
    LwSciError error = LwSciError_Success;
    bool isCompressible;
    LwSciBufAttrValGpuCompression expectedVal = {};

    {
        LwSciRmGpuId gpuIds[] = {testGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd1AttrList, LwSciBufType_Image);

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        LwSciRmGpuId gpuIds[] = {testGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd2AttrList, LwSciBufType_Image);

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    ASSERT_EQ(isGpuKindCompressible(tstResource, testGpuId, true,
        &isCompressible), LwSciError_Success);

    expectedVal.gpuId = testGpuId;
    if (isCompressible == true) {
        /* TODO: We will need to consider more conditions once LwSciBuf
         * starts supporting more compression types
         */
        expectedVal.compressionType =
            LwSciBufCompressionType_GenericCompressible;
    } else {
        expectedVal.compressionType = LwSciBufCompressionType_None;
    }

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCompression, expectedVal));
}

/**
 * Positive test
 * Set GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request GPU compression for GPU via
 * LwSciBufGeneralAttrKey_EnableGpuCompression.
 * Reqeust CPU access via LwSciBufGeneralAttrKey_NeedCpuAccess.
 * Reconciliation should succeed and compression should not be allowed
 */
TEST_F(TestLwSciBufGpuCompression, GpuCompressionPositiveTest2)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };
        bool needCpuAccess = true;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd1AttrList, LwSciBufType_Image);

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
            needCpuAccess);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        LwSciRmGpuId gpuIds[] = {testGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd2AttrList, LwSciBufType_Image);

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCompression expectedVal[] = {
        [0] = {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_None
              },
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCompression, expectedVal));
}

/**
 * Positive test
 * Set GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request GPU compression for GPU via
 * LwSciBufGeneralAttrKey_EnableGpuCompression.
 * Reqeust engine access via LwSciBufInternalGeneralAttrKey_EngineArray.
 * Reconciliation should succeed and compression should not be allowed
 */
TEST_F(TestLwSciBufGpuCompression, GpuCompressionPositiveTest3)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciBufHwEngine engine{};
        LwSciRmGpuId gpuIds[] = {testGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };

        setupAttrList(umd1AttrList, LwSciBufType_Image);
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

#if !defined(__x86_64__)
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
            &engine.rmModuleID);
#else
        //TODO: Figure out a way to query this info dynamically on X86.
        engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
        engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
        engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;
#endif
        SET_INTERNAL_ATTR(umd1AttrList.get(),
            LwSciBufInternalGeneralAttrKey_EngineArray, engine);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        LwSciRmGpuId gpuIds[] = {testGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd2AttrList, LwSciBufType_Image);

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCompression expectedVal[] = {
        [0] = {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_None
              },
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCompression, expectedVal));
}

/**
 * Negative test
 * Don't set GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request GPU compression for GPU via
 * LwSciBufGeneralAttrKey_EnableGpuCompression.
 * Reconciliation should fail since we are requesting GPU compression for a
 * GPU without setting that GPU ID in LwSciBufGeneralAttrKey_GpuId.
 */
TEST_F(TestLwSciBufGpuCompression, GpuCompressionNegativeTest1)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };

        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd1AttrList, LwSciBufType_Image);

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testGpuId,
                .compressionType = LwSciBufCompressionType_GenericCompressible
            };

        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd2AttrList, LwSciBufType_Image);

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        NEGATIVE_TEST();

        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get(), umd2AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}

class TestLwSciBufGpuCompressionMultGpu : public LwSciiGpuAnddGpuTest,
                                       public LwSciBufGpuCompression
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufGpuCompression::SetUp();
        LwSciiGpuAnddGpuTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        umd2AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd2AttrList.get(), nullptr);

    }

    virtual void TearDown() override
    {
        LwSciBufGpuCompression::TearDown();
        LwSciiGpuAnddGpuTest::TearDown();

        umd1AttrList.reset();
        umd2AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
    std::shared_ptr<LwSciBufAttrListRec> umd2AttrList;
};

/**
 * Positive test
 * Set iGPU and dGPU GPU IDs in LwSciBufGeneralAttrKey_GpuId.
 * Request GPU compression for both GPU via
 * LwSciBufGeneralAttrKey_EnableGpuCompression.
 * Reconciliation should succeed but compression should not be allowed since
 * we don't allow it for multiple GPUs.
 */
TEST_F(TestLwSciBufGpuCompressionMultGpu, GpuCompressionMultGpuPositiveTest1)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCompression gpuCompression[] =
            {
                {
                    .gpuId = testiGpuId,
                    .compressionType =
                        LwSciBufCompressionType_GenericCompressible
                },
                {
                    .gpuId = testdGpuId,
                    .compressionType =
                        LwSciBufCompressionType_GenericCompressible
                }
            };

        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd1AttrList, LwSciBufType_Image);

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCompression gpuCompression[] =
            {
                {
                    .gpuId = testiGpuId,
                    .compressionType =
                        LwSciBufCompressionType_GenericCompressible
                },
                {
                    .gpuId = testdGpuId,
                    .compressionType =
                        LwSciBufCompressionType_GenericCompressible
                }
            };

        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd2AttrList, LwSciBufType_Image);

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    {
        LwSciBufAttrValGpuCompression expectedVal[] =
            {
                {
                    .gpuId = testiGpuId,
                    .compressionType =
                        LwSciBufCompressionType_None
                },
                {
                    .gpuId = testdGpuId,
                    .compressionType =
                        LwSciBufCompressionType_None
                }
            };

        ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
            LwSciBufGeneralAttrKey_EnableGpuCompression, expectedVal));
    }
}

/**
 * Negative test
 * Set iGPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request GPU compression for dGPU via
 * LwSciBufGeneralAttrKey_EnableGpuCompression.
 * Reconciliation should fail since we are requesting GPU compression for the
 * GPU which is not specified in the LwSciBufGeneralAttrKey_GpuId.
 */
TEST_F(TestLwSciBufGpuCompressionMultGpu, GpuCompressionMultGpuNegativeTest1)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testdGpuId,
                .compressionType =
                    LwSciBufCompressionType_GenericCompressible
            };

        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd1AttrList, LwSciBufType_Image);

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd1AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId};
        LwSciBufAttrValGpuCompression gpuCompression =
            {
                .gpuId = testdGpuId,
                .compressionType =
                    LwSciBufCompressionType_GenericCompressible
            };

        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_InterlaceType;

        setupAttrList(umd2AttrList, LwSciBufType_Image);

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCompression
                , gpuCompression);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_Layout,
            layout);
        SET_ATTR(umd2AttrList.get(), LwSciBufImageAttrKey_ScanType,
            scanType);
    }

    {
        NEGATIVE_TEST();

        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get(), umd2AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
