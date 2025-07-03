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
#include "lwscibuf_test_integration.h"

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class TestLwSciBufRawBuffer : public LwSciBufBasicTest
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

TEST_F(TestLwSciBufRawBuffer, IntraThreadRawBuffer)
{
    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t rawSize = (128U * 1024U);
        uint64_t alignment = (4U * 1024U);
        bool cpuAccessFlag = false;

        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Size, rawSize);
        SET_ATTR(umd1AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(umd1AttrList.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);
    }

    {
        LwSciBufType bufType = LwSciBufType_RawBuffer;
        uint64_t alignment = (8U * 1024U);
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(umd2AttrList.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(umd2AttrList.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

        SET_INTERNAL_ATTR(umd2AttrList.get(),
                          LwSciBufInternalGeneralAttrKey_MemDomainArray,
                          memDomain);
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

    /* Get size from RM */
    uint64_t size = GetMemorySize(rmHandle);
    ASSERT_EQ(size, CEIL_TO_LEVEL(len, GetPageSize()))
        << "size verification failed"
        << " Expected " << size << " Got " << CEIL_TO_LEVEL(len, GetPageSize());
}
