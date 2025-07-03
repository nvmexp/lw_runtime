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
#include "lwsci_igpu_only_test.h"
#include "lwsci_dgpu_only_test.h"
#include "lwsci_igpu_or_dgpu_test.h"
#include "lwsci_igpu_and_dgpu_test.h"

class LwSciBufGpuCache : public LwSciBufBasicTest
{
public:
    static inline void setupAttrList(
        std::shared_ptr<LwSciBufAttrListRec> attrList) {
        uint64_t rawBufSize = (128U * 1024U);
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t alignment = (4U * 1024U);

        SET_ATTR(attrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(attrList.get(), LwSciBufRawBufferAttrKey_Size, rawBufSize);
        SET_ATTR(attrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
    }

    void SetUp() override
    {
        LwSciBufBasicTest::SetUp();
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();
    }
};

class TestLwSciBufGpuCacheNoGpu: public LwSciBufGpuCache
{
public:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufGpuCache::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        setupAttrList(umd1AttrList);
    }

    void TearDown() override
    {
        LwSciBufGpuCache::TearDown();
        umd1AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
};

/**
 * Positive test
 * Set no GPU IDs in LwSciBufGeneralAttrKey_GpuId.
 * Set no vidmem GPU ID in LwSciBufGeneralAttrKey_VidMem_GpuId.
 * Set sysmem in LwSciBufInternalGeneralAttrKey_MemDomainArray.
 * Reconciliation should pass.
 */
TEST_F(TestLwSciBufGpuCacheNoGpu, GpuIdAttrPositiveTest1)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};

        SET_INTERNAL_ATTR(umd1AttrList.get(),
            LwSciBufInternalGeneralAttrKey_MemDomainArray, memDomain);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
}

/**
 * Positive test
 * Set no GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Dont request cacheability control via LwSciBufGeneralAttrKey_EnableGpuCache.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should not be populated.
 */
TEST_F(TestLwSciBufGpuCacheNoGpu, EnableGpuCacheAttrPositiveTest3)
{
    LwSciError error = LwSciError_Success;

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciRmGpuId expectedGpuId = {};
    LwSciBufAttrValGpuCache expectedVal[] = {};

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCache, expectedVal));
}

class TestLwSciBufGpuCacheiGpudGpu : public LwSciiGpuAnddGpuTest, public LwSciBufGpuCache
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufGpuCache::SetUp();
        LwSciiGpuAnddGpuTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        setupAttrList(umd1AttrList);

        umd2AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd2AttrList.get(), nullptr);

        setupAttrList(umd2AttrList);
    }

    virtual void TearDown() override
    {
        LwSciBufGpuCache::TearDown();
        LwSciiGpuAnddGpuTest::TearDown();

        umd1AttrList.reset();
        umd2AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
    std::shared_ptr<LwSciBufAttrListRec> umd2AttrList;
};

/**
 * Positive test
 * Set iGPU + dGPU GPU IDs in LwSciBufGeneralAttrKey_GpuId.
 * Set sysmem in LwSciBufInternalGeneralAttrKey_MemDomainArray.
 * Reconciliation should pass since iGPU + dGPU can access sysmem.
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, GpuIdAttrPositiveTest2)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_INTERNAL_ATTR(umd1AttrList.get(),
            LwSciBufInternalGeneralAttrKey_MemDomainArray, memDomain);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
}

/**
 * Run the test on non-safety only since we are using vidmem memory domain or
 * CVSRAM memory domain which are only supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Negative test
 * Set iGPU + dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * set vidmem GPU ID in LwSciBufGeneralAttrKey_VidMem_GpuId.
 * Reconciliation should fail since we don't support peer mem case (peer mem case
 * is multiple GPUs accessing vidmem).
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, GpuIdAttrNegativeTest3)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
            testdGpuId);
    }

    {
        NEGATIVE_TEST();
        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using vidmem memory domain or
 * CVSRAM memory domain which are only supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Negative test
 * Set iGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * set vidmem GPU ID in LwSciBufGeneralAttrKey_VidMem_GpuId.
 * Reconciliation should fail since we don't support peer mem case (peer mem case
 * is multiple GPUs accessing vidmem OR one GPU is accessing vidmem of other
 * GPU).
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, GpuIdAttrNegativeTest4)
{
    LwSciError error = LwSciError_Success;

    {
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testiGpuId);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
            testdGpuId);
    }

    {
        NEGATIVE_TEST();
        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set iGPU + dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Dont request cacheability control via LwSciBufGeneralAttrKey_EnableGpuCache.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should be populated with default
 * cacheability values.
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, EnableGpuCacheAttrPositiveTest1)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = true},
        [1] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCache, expectedVal));
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set iGPU + dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request true cacheability control for iGPU in first unreconciled list and
 * request true cacheability for iGPU in second unreconciled list.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should be populated with reconciled
 * value for iGPU and default value for dGPU
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, EnableGpuCacheAttrPositiveTest4)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = true};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = true};

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = true},
        [1] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCache, expectedVal));
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set iGPU + dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request true cacheability control for iGPU in first unreconciled list and
 * request false cacheability for iGPU in second unreconciled list.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should be populated with reconciled
 * value for iGPU and default value for dGPU
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, EnableGpuCacheAttrPositiveTest5)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = true};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = false};

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = false},
        [1] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCache, expectedVal));
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set iGPU + dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request true cacheability control for iGPU in first unreconciled list and
 * don't request cacheability for iGPU in second unreconciled list.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should be populated with reconciled
 * value for iGPU and default value for dGPU
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, EnableGpuCacheAttrPositiveTest6)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = true};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = true},
        [1] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCache, expectedVal));
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set iGPU + dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request false cacheability control for iGPU in first unreconciled list and
 * don't request cacheability for iGPU in second unreconciled list.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should be populated with reconciled
 * value for iGPU and default value for dGPU
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, EnableGpuCacheAttrPositiveTest7)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = false};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = false},
        [1] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCache, expectedVal));
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Negative test
 * Set iGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request false cacheability control for dGPU in first unreconciled list.
 * Reconciliation should fail.
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, EnableGpuCacheAttrNegativeTest1)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testdGpuId, .cacheability = false};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testiGpuId);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    {
        NEGATIVE_TEST();
        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Negative test
 * Request cacheability control for iGPU + dGPU in first unreconciled list.
 * The reconciliation should fail since LwSciBuf only allows requesting
 * cacheability for single iGPU as of now.
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, EnableGpuCacheAttrNegativeTest2)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciBufAttrValGpuCache gpuCache[] = {
            [0] = {.gpuId = testiGpuId, .cacheability = true},
            [1] = {.gpuId = testdGpuId, .cacheability = true},
        };

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testiGpuId);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    {
        NEGATIVE_TEST();
        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set iGPU + dGPU GPU IDs in LwSciBufGeneralAttrKey_GpuId.
 * Request false cacheability control for iGPU via
 * LwSciBufGeneralAttrKey_EnableGpuCache.
 * Reconciliation should succeed and LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency
 * should be set for both GPUs
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, GpuCacheCoherencyAttrPositiveTest1)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = false};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = false},
        [1] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, expectedVal));

}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set iGPU + dGPU GPU IDs in LwSciBufGeneralAttrKey_GpuId.
 * Request true cacheability control for iGPU via
 * LwSciBufGeneralAttrKey_EnableGpuCache.
 * Reconciliation should succeed and LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency
 * should be set for both GPUs
 */
TEST_F(TestLwSciBufGpuCacheiGpudGpu, GpuCacheCoherencyAttrPositiveTest2)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testiGpuId, testdGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = true};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = true},
        [1] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, expectedVal));

}
#endif //(LW_IS_SAFETY == 0)

class TestLwSciBufGpuCachedGpu : public LwScidGpuOnlyTest, public LwSciBufGpuCache
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufGpuCache::SetUp();
        LwScidGpuOnlyTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        setupAttrList(umd1AttrList);
    }

    virtual void TearDown() override
    {
        LwSciBufGpuCache::TearDown();
        LwScidGpuOnlyTest::TearDown();

        umd1AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
};

/**
 * Run the test on non-safety only since we are using vidmem memory domain or
 * CVSRAM memory domain which are only supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Negative test
 * Set no GPU IDs in LwSciBufGeneralAttrKey_GpuId.
 * Set vidmem GPU ID in LwSciBufGeneralAttrKey_VidMem_GpuId.
 * Reconciliation should fail since no GPUs are set in LwSciBufGeneralAttrKey_GpuId
 * to access the vidmem.
 */
TEST_F(TestLwSciBufGpuCachedGpu, GpuIdAttrNegativeTest1)
{
    LwSciError error = LwSciError_Success;

    {
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
            testdGpuId);
    }

    {
        NEGATIVE_TEST();
        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Positive test
 * Set dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Dont request cacheability control via LwSciBufGeneralAttrKey_EnableGpuCache.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should be populated with default
 * cacheability values.
 */
TEST_F(TestLwSciBufGpuCachedGpu, EnableGpuCacheAttrPositiveTest2)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testdGpuId};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testdGpuId, .cacheability = false},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_EnableGpuCache, expectedVal));
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Run the test on non-safety only since we are using dGPU which is only
 * supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
class TestLwSciBufGpuSwNeedCacheCoherency
    : public TestLwSciBufGpuCachedGpu,
      public ::testing::WithParamInterface<std::tuple<LwSciBufHwEngName>>
{
};
/**
 * Positive test
 * Set dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Dont request cacheability control via LwSciBufGeneralAttrKey_EnableGpuCache.
 * Provide non-VidMem coherent engines.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency should be populated with true
 * cacheability values.
 */
TEST_P(TestLwSciBufGpuSwNeedCacheCoherency, NonCoherentVidMemEngines)
{
    LwSciError error = LwSciError_Success;

    auto params = GetParam();
    LwSciBufHwEngName engName = std::get<0>(params);

    {
        LwSciRmGpuId gpuIds[] = {testdGpuId};
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
                 gpuIds);

        // Request Vidmem
        SET_INTERNAL_ATTR(umd1AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          LwSciBufMemDomain_Vidmem);

        LwSciBufHwEngine engine{};
#if !defined(__x86_64__) //TODO: separate this in platform file.
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
    }

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Assert that VidMem is reconciled
    LwSciBufPeer::verifyInternalAttr(
        reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray,
        LwSciBufMemDomain_Vidmem);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testdGpuId, .cacheability = true},
    };
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_EnableGpuCache,
                                         expectedVal));

    // When the reconciled memory domain is VidMem, the result of the
    // GpuSwNeedCacheCoherency key depends on whether all the engines
    // requesting access are cache-coherent.
    //
    // As such, we need to indicate to the software that a flush is necessary.
    LwSciBufAttrValGpuCache expectedCoherencyVal[] = {
        [0] = {.gpuId = testdGpuId, .cacheability = true},
    };
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency,
        expectedCoherencyVal));
}
INSTANTIATE_TEST_CASE_P(
    TestLwSciBufGpuSwNeedCacheCoherency, TestLwSciBufGpuSwNeedCacheCoherency,
    ::testing::Values(std::make_tuple(LwSciBufHwEngName_PCIe)));

/**
 * Positive test
 * Set dGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Dont request cacheability control via LwSciBufGeneralAttrKey_EnableGpuCache.
 * Don't provide any non-VidMem coherent engines.
 * Reconciliation should pass.
 * LwSciBufGeneralAttrKey_EnableGpuCache should be populated with true
 * cacheability values.
 */
TEST_F(TestLwSciBufGpuCachedGpu, CoherentEngine)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciRmGpuId gpuIds[] = {testdGpuId};
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_VidMem_GpuId,
                 gpuIds);

        // Request Vidmem
        SET_INTERNAL_ATTR(umd1AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          LwSciBufMemDomain_Vidmem);
    }

    auto reconciledList =
        LwSciBufPeer::attrListReconcile({umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Assert that VidMem is reconciled
    LwSciBufPeer::verifyInternalAttr(
        reconciledList.get(), LwSciBufInternalGeneralAttrKey_MemDomainArray,
        LwSciBufMemDomain_Vidmem);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testdGpuId, .cacheability = true},
    };
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufGeneralAttrKey_EnableGpuCache,
                                         expectedVal));

    // When the reconciled memory domain is VidMem, the result of the
    // GpuSwNeedCacheCoherency key depends on whether all the engines
    // requesting access are cache-coherent.

    // No HW Engines are specified which are non-coherent with VidMem
    LwSciBufAttrValGpuCache expectedCoherencyVal[] = {
        [0] = {.gpuId = testdGpuId, .cacheability = false},
    };
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency,
        expectedCoherencyVal));
}
#endif //(LW_IS_SAFETY == 0)

class TestLwSciBufGpuCacheiGpu : public LwSciiGpuOnlyTest, public LwSciBufGpuCache
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufGpuCache::SetUp();
        LwSciiGpuOnlyTest::SetUp();

        umd1AttrList = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(umd1AttrList.get(), nullptr);

        setupAttrList(umd1AttrList);
    }

    virtual void TearDown() override
    {
        LwSciBufGpuCache::TearDown();
        LwSciiGpuOnlyTest::TearDown();

        umd1AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
};

/**
 * Run the test on non-safety only since we are using vidmem memory domain or
 * CVSRAM memory domain which are only supported in non-safety.
 */
#if (LW_IS_SAFETY == 0)
/**
 * Negative test
 * Set iGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Set CVSRAM in LwSciBufInternalGeneralAttrKey_MemDomainArray.
 * Reconciliation should fail since GPUs can't access CVSRAM.
 */
TEST_F(TestLwSciBufGpuCacheiGpu, GpuIdAttrNegativeTest2)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Cvsram};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testiGpuId);
        SET_INTERNAL_ATTR(umd1AttrList.get(),
            LwSciBufInternalGeneralAttrKey_MemDomainArray, memDomain);
    }

    {
        NEGATIVE_TEST();
        auto reconciledList = LwSciBufPeer::attrListReconcile(
            {umd1AttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_ReconciliationFailed);
    }
}
#endif //(LW_IS_SAFETY == 0)

/**
 * Positive test
 * Set iGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request true cacheability control for iGPU via
 * LwSciBufGeneralAttrKey_EnableGpuCache.
 * set LwSciBufGeneralAttrKey_NeedCpuAccess to true.
 * Reconciliation should succeed and LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency
 * should be set for the iGPU.
 */
TEST_F(TestLwSciBufGpuCacheiGpu, GpuCacheCoherencyAttrPositiveTest3)
{
    LwSciError error = LwSciError_Success;

    {
        bool needCpuAccess = true;
        LwSciRmGpuId gpuIds[] = {testiGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = true};

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
            needCpuAccess);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = true},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, expectedVal));

}

/**
 * Positive test
 * Set iGPU GPU ID in LwSciBufGeneralAttrKey_GpuId.
 * Request true cacheability control for iGPU via
 * LwSciBufGeneralAttrKey_EnableGpuCache.
 * set LwSciBufInternalGeneralAttrKey_EngineArray with at least one engine.
 * Reconciliation should succeed and LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency
 * should be set for the iGPU.
 */
TEST_F(TestLwSciBufGpuCacheiGpu, GpuCacheCoherencyAttrPositiveTest4)
{
    LwSciError error = LwSciError_Success;

    {
        LwSciBufHwEngine engine {};
        LwSciRmGpuId gpuIds[] = {testiGpuId};
        LwSciBufAttrValGpuCache gpuCache =
            {.gpuId = testiGpuId, .cacheability = true};

#if !defined(__x86_64__) //TODO: separate this in platform file.
        LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vi,
                                             &engine.rmModuleID);
#else
        // the following field should be queried first by UMD
        engine.engNamespace = LwSciBufHwEngine_ResmanNamespaceId;
        engine.subEngineID = LW2080_ENGINE_TYPE_GRAPHICS;
        engine.rev.gpu.arch = LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100;
#endif

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, gpuIds);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_EnableGpuCache,
            gpuCache);
        SET_INTERNAL_ATTR(umd1AttrList.get(),
            LwSciBufInternalGeneralAttrKey_EngineArray, engine);
    }

    auto reconciledList = LwSciBufPeer::attrListReconcile(
        {umd1AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufAttrValGpuCache expectedVal[] = {
        [0] = {.gpuId = testiGpuId, .cacheability = true},
    };

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
        LwSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, expectedVal));
}
