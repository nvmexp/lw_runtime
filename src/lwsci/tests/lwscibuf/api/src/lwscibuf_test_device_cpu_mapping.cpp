/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_basic_test.h"
#include "lwsci_igpu_or_dgpu_test.h"
#include "lwscibuf_test_gpu_platform_helper.h"

static const uint64_t rawSize = (128U * 1024U);

class TestLwSciBufDeviceCpuMapping : public LwSciiGpuOrdGpuTest,
                                        public LwSciBufBasicTest
{
public:
    virtual void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();
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
        LwSciBufBasicTest::TearDown();
        LwSciiGpuOrdGpuTest::TearDown();

        umd1AttrList.reset();
        umd2AttrList.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> umd1AttrList;
    std::shared_ptr<LwSciBufAttrListRec> umd2AttrList;
};

TEST_F(TestLwSciBufDeviceCpuMapping, DeviceCpuMemMapping)
{
    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t alignment = (4U * 1024U);
        bool cpuAccessFlag = true;

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Size, rawSize);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testGpuId);
    }

    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t alignment = (8U * 1024U);
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

        LwSciBufMemDomain memDomain[] = {LwSciBufMemDomain_Sysmem};

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
    }

    /*Allocate source buffer objects*/
    LwSciError error = LwSciError_Success;
    auto srcBufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    /* Allocation size check */
    LwSciBufRmHandle srcRmHandle = {0};
    uint64_t srcOffset = 0U;
    uint64_t srcLen = 0U;
    ASSERT_EQ(LwSciBufObjGetMemHandle(srcBufObj.get(), &srcRmHandle, &srcOffset,
                                      &srcLen),
              LwSciError_Success);

    /* Get size from RM */
    uint64_t srcSize = GetMemorySize(srcRmHandle);
    ASSERT_EQ(srcSize, CEIL_TO_LEVEL(srcLen, GetPageSize()));

    /*Allocate destination buffer objects*/
    auto dstBufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    /* Allocation size check */
    LwSciBufRmHandle dstRmHandle = {0};
    uint64_t dstOffset = 0U;
    uint64_t dstLen = 0U;
    ASSERT_EQ(LwSciBufObjGetMemHandle(dstBufObj.get(), &dstRmHandle, &dstOffset,
                                      &dstLen),
              LwSciError_Success);

    /* Get size from RM */
    uint64_t dstSize = GetMemorySize(dstRmHandle);
    ASSERT_EQ(dstSize, CEIL_TO_LEVEL(dstLen, GetPageSize()));

    ASSERT_EQ(testDeviceCpuMapping(tstResource, srcBufObj.get(), srcRmHandle,
                                   dstBufObj.get(), dstRmHandle, dstOffset,
                                   dstLen, rawSize),
              LwSciError_Success);
}
