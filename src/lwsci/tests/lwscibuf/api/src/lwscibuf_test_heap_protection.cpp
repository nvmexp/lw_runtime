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
#include "lwsci_igpu_or_dgpu_test.h"
#include "lwscibuf_test_gpu_platform_helper.h"

static const uint64_t rawSize = (128U * 1024U);

class TestLwSciBufHeapProtection : public LwSciiGpuOrdGpuTest,
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

        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            uint64_t alignment = (4U * 1024U);
            bool cpuAccessFlag = false;

            SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Size,
                     rawSize);
            SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Align,
                     alignment);
            SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                     cpuAccessFlag);
        }

        {
            LwSciBufType bufType = LwSciBufType_RawBuffer;
            uint64_t alignment = (8U * 1024U);
            LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
            LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

            SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
            SET_ATTR(umd2AttrList.get(), LwSciBufRawBufferAttrKey_Align,
                     alignment);
            SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm,
                     perm);
        }
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

TEST_F(TestLwSciBufHeapProtection, IntraThreadHeapProtection)
{
    {
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_GpuId, testGpuId);
    }

    LwSciError error = LwSciError_Success;
    auto bufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U;
    uint64_t len = 0U;
    /* Allocation size check */
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    /* Get size from RM */
    uint64_t size = GetMemorySize(rmHandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "size verification failed"
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());

    ASSERT_EQ(LwSciError_Success, testGpuMapping(tstResource, rmHandle,
        rawSize)) << "GPU mapping test failed";
}

TEST_F(TestLwSciBufHeapProtection, NegativeIntraThreadHeapProtection)
{
    LwSciError error = LwSciError_Success;
    auto bufObj = LwSciBufPeer::reconcileAndAllocate(
        {umd1AttrList.get(), umd2AttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    LwSciBufRmHandle rmHandle = {0};
    uint64_t offset = 0U;
    uint64_t len = 0U;
    /* Allocation size check */
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    /* Get size from RM */
    uint64_t size = GetMemorySize(rmHandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "size verification failed"
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());

    ASSERT_EQ(LwSciError_IlwalidState, testGpuMapping(tstResource, rmHandle,
        rawSize)) << "dGPU mapping test failed";
}
