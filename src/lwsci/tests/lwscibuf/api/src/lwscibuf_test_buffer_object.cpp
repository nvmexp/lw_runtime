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
#include <string.h>

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class BufferObject : public LwSciBufBasicTest
{
public:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        listA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listA.get(), nullptr);

        listB = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listB.get(), nullptr);

        uint64_t size = (128U * 1024U);
        uint64_t alignment = (4U * 1024U);
        uint64_t offset = 0U;
        uint64_t len = 0U;
        bool cpuAccessFlag = true;

        LwSciBufType bufType = LwSciBufType_RawBuffer;
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

        // Setup list A
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_NeedCpuAccess,
                 cpuAccessFlag);

        // Setup list B
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

        // Reconcile Lists
        reconciledList =
            LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
        ASSERT_NE(reconciledList, nullptr);

        // Validate Reconciled List
        bool isReconciledListValid = false;
        ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                                   reconciledList.get(),
                                                   &isReconciledListValid),
                  LwSciError_Success);
        ASSERT_TRUE(isReconciledListValid);

        // Create Buffer Object
        bufObj = LwSciBufPeer::allocateBufObj(reconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        listA.reset();
        listB.reset();

        reconciledList.reset();
        bufObj.reset();
    }

    bool VerifyActualPermission(LwSciBufObj bufObj,
                                LwSciBufAttrValAccessPerm permission)
    {
        bool status = false;
#if !defined(__x86_64__)
        LwSciBufRmHandle rmHandle;
        uint64_t offset, len;
        LwRmMemHandleParams memHandleParams;
        uint32_t lwrmPerms;
#endif

        LwSciBufAttrList reconciledAttrList;
        LwSciBufAttrValAccessPerm accessPerms;
        LwSciBufAttrKeyValuePair keyValuePair = { LwSciBufGeneralAttrKey_ActualPerm,
                                                };

        if (LwSciBufObjGetAttrList(bufObj, &reconciledAttrList) !=
                  LwSciError_Success) {
            goto ret;
        }

        if (LwSciBufAttrListGetAttrs(reconciledAttrList, &keyValuePair, 1) !=
               LwSciError_Success) {
            goto ret;
        }

        accessPerms = *(const LwSciBufAttrValAccessPerm*)keyValuePair.value;
        if (accessPerms != permission)  {
            goto ret;
        }

#if !defined(__x86_64__)
        /* Check that access permissions associated with LwSciBufObj are same
         * as that associated with LwRmMemHandle. This is cheetah specific check
         * because X86 does not support LwSciBufObjDupWithReducePerm()
         * functionality yet (it just ignores the reduced permissions).
         */
        if (LwSciBufObjGetMemHandle(bufObj, &rmHandle, &offset, &len) !=
            LwSciError_Success) {
            goto ret;
        }

        if (LwRmMemQueryHandleParams(rmHandle.memHandle, rmHandle.memHandle,
                &memHandleParams,
            sizeof(memHandleParams)) != LwError_Success) {
            goto ret;
        }

        lwrmPerms = GetLwRmAccessFlags(accessPerms);
        if (lwrmPerms == LWOS_MEM_NONE) {
            goto ret;
        }

        if (lwrmPerms != memHandleParams.AccessFlags) {
            goto ret;
        }
#endif

        status = true;

    ret:
        return status;
    }

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;

    std::shared_ptr<LwSciBufAttrListRec> reconciledList;
    std::shared_ptr<LwSciBufObjRefRec> bufObj;

    const uint64_t testPattern = 0xC0DEC0DEC0DEC0DEU;
};

/**
* Test case: Test the cache coherency ff NeedSWCacheCoherency attribute value in
* the attribute list is set. Validate that the region of CPU cache described by
* the CPU VA, offset & length is flushed.
*/
TEST_F(BufferObject, DISABLED_CpuCacheFlush)
{
    // TODO How to validate that the CPU cache reagion is flushed?
}

/**
* Test case: A buffer handle can be retrieved with lwmap handle, offset, length
* and a reconciled attribute list
*/
TEST_F(BufferObject, CreateObjectFromRmHandle)
{
    uint64_t offset = 0U, len = 0U, offset2 = 0U, len2 = 0U;
    LwSciBufRmHandle rmHandle = {0}, rmHandle2 = {0};

    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    // createFromMemHandle(rmHandle, offset, len, &bufObj2);
    LwSciError error = LwSciError_Success;
    auto bufObj2 = LwSciBufPeer::createFromMemHandle(
        rmHandle, reconciledList.get(), offset, len, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Get RM handle of retrived object
    ASSERT_EQ(
        LwSciBufObjGetMemHandle(bufObj2.get(), &rmHandle2, &offset2, &len2),
        LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    ASSERT_EQ(offset, offset2);
    ASSERT_EQ(len, len2);
}

/**
* Test case: When an allocated buffer with one handle is freed, the RM handle
* associated with the buffer is also freed
*/
TEST_F(BufferObject, DeallocateBuffer)
{
    uint64_t offset = 0U;
    uint64_t len = 0U;

    LwSciBufRmHandle rmHandle = {0};

    // Retrieve RM handle associated with buf handle
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    // Free buf handle
    bufObj.reset();
    bufObj = nullptr;

    /* Verify that RM handle is also freed
    *  Note: There is no direct way or API to verify that the RM handle is indeed
    *  free. This test is doing an indirect verification by doing operations on
    *  freed RM handle and check if resulted in error.
    */
    ASSERT_TRUE(isRMHandleFree(rmHandle)) << "RM handle is not free";
}

/**
* Test case: Test the ActualPerm attribute value of reconciled attribute list
* bound to the duplicate buffer handle
*/
TEST_F(BufferObject, DuplicateHandleBinding)
{
    uint64_t size = (128U * 1024U);
    uint64_t alignment = (4U * 1024U);
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    //** SubTest 1 **//
    {
        // Duplicate the LwSciBuf object handle
        LwSciError error = LwSciError_Success;
        auto dupBufObj = LwSciBufPeer::duplicateBufObj(bufObj.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Get reconciled list
        // Note: We don't need to free this Attribute List since it is managed
        // by LwSciBuf
        LwSciBufAttrList retrievedAttrList = nullptr;
        ASSERT_EQ(LwSciBufObjGetAttrList(bufObj.get(), &retrievedAttrList),
                  LwSciError_Success)
            << "Failed to retrieve the attribute list";

        // Check reconciled list
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(
            retrievedAttrList, LwSciBufGeneralAttrKey_Types, bufType));
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(
            retrievedAttrList, LwSciBufRawBufferAttrKey_Size, size));
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(
            retrievedAttrList, LwSciBufRawBufferAttrKey_Align, alignment));
        ASSERT_TRUE(LwSciBufPeer::verifyAttr(
            retrievedAttrList, LwSciBufGeneralAttrKey_ActualPerm, perm));

        // Check actualPerm permissions
        ASSERT_TRUE(VerifyActualPermission(dupBufObj.get(),
                                           LwSciBufAccessPerm_ReadWrite))
            << "ActualPerm verification failed";
    }

    /* Subtest 2 */
    {
        // Duplicate the LwSciBuf object handle with reduced permissions
        LwSciError error = LwSciError_Success;
        auto dupBufObjReducedPerm =
            LwSciBufPeer::duplicateBufObjWithReducedPerm(
                bufObj.get(), LwSciBufAccessPerm_Readonly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        // Check actualPerm permissions
        ASSERT_TRUE(VerifyActualPermission(dupBufObjReducedPerm.get(),
                                           LwSciBufAccessPerm_Readonly))
            << "ActualPerm verification failed";
    }
}

/**
* Test case: Test the GetCpuPtr API with read/write permission
* permission on after duplicating handle.
*/

TEST_F(BufferObject, DuplicateHandleRWCpuPtr)
{
    void* vaPtr = NULL;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), &vaPtr), LwSciError_Success)
        << "Failed to get cpu ptr";

    // Duplicate the LwSciBuf object handle
    LwSciError error = LwSciError_Success;
    auto dupBufObj = LwSciBufPeer::duplicateBufObj(bufObj.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);

    void* vaPtrDup = NULL;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(dupBufObj.get(), &vaPtrDup), LwSciError_Success)
        << "Failed to get cpu ptr of duplicated object";

    const void* constVaPtrDup = NULL;
    ASSERT_EQ(LwSciBufObjGetConstCpuPtr(dupBufObj.get(), &constVaPtrDup), LwSciError_Success)
        << "Failed to get const cpu ptr of duplicated object";

    /* Verify CPU access */
    *(uint64_t *)vaPtrDup = testPattern;
    uint64_t testval = *(uint64_t*)constVaPtrDup;
    ASSERT_EQ(testval, testPattern)
        << "Failed to verify test pattern in memory when written from duplicated object RW ptr and read from duplicated object RO ptr";

    testval = 0x00;
    testval = *(uint64_t*)vaPtr;
    ASSERT_EQ(testval, testPattern)
        << "Failed to verify test pattern in memory when written from duplicated object RW ptr and read from orignal object RW ptr";
}

/**
* Test case: Test the GetCpuPtr API with read-only permission
* after duplicating handle with reduced permission.
*/
TEST_F(BufferObject, DuplicateHandleROCpuPtr)
{
    void* vaPtr = NULL;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), &vaPtr), LwSciError_Success)
        << "Failed to get cpu ptr";

    // Duplicate the LwSciBuf object handle with reduced permissions
    LwSciError error = LwSciError_Success;
    auto dupBufObjReducedPerm =
        LwSciBufPeer::duplicateBufObjWithReducedPerm(
            bufObj.get(), LwSciBufAccessPerm_Readonly, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Check actualPerm permissions
    ASSERT_TRUE(VerifyActualPermission(dupBufObjReducedPerm.get(),
                                       LwSciBufAccessPerm_Readonly))
        << "ActualPerm verification failed";

    const void* constVaPtrDup = NULL;
    ASSERT_EQ(LwSciBufObjGetConstCpuPtr(dupBufObjReducedPerm.get(),
        &constVaPtrDup), LwSciError_Success)
        << "Failed to get const cpu ptr";

    /* Verify CPU access */
    *(uint64_t *)vaPtr = testPattern;
    uint64_t testval = *(uint64_t*)constVaPtrDup;
    ASSERT_EQ(testval, testPattern)
        << "Failed to verify test pattern in memory when written from original object RW ptr and read from duplicated object RO ptr";
}

/**
* Test case: Negative test to verify GetCpuPtr API fails
* with read-only permission handle after duplicating handle
*  with reduced permission.
*/
TEST_F(BufferObject, NegativeROCpuPtr)
{
    void* vaPtr = NULL;
    ASSERT_EQ(LwSciBufObjGetCpuPtr(bufObj.get(), &vaPtr), LwSciError_Success)
        << "Failed to get cpu ptr";

    // Duplicate the LwSciBuf object handle with reduced permissions
    LwSciError error = LwSciError_Success;
    auto dupBufObjReducedPerm =
        LwSciBufPeer::duplicateBufObjWithReducedPerm(
            bufObj.get(), LwSciBufAccessPerm_Readonly, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Check actualPerm permissions
    ASSERT_TRUE(VerifyActualPermission(dupBufObjReducedPerm.get(),
                                       LwSciBufAccessPerm_Readonly))
        << "ActualPerm verification failed";

    {
        NEGATIVE_TEST();
        void* vaPtrDup = NULL;
        ASSERT_EQ(LwSciBufObjGetCpuPtr(dupBufObjReducedPerm.get(), &vaPtrDup),
            LwSciError_BadParameter)
            << "LwSciBuf failed to verify access permissions when getting CPU ptr";
    }
}
/**
* Test case: Test the access permissions granted to the associated RM handles of
* original & duplicate reference are same
*/
TEST_F(BufferObject, DuplicateHandlePermissions)
{
    uint64_t offset = 0U;
    uint64_t len = 0U;

    LwSciBufRmHandle rmHandle = {0};
    LwSciBufRmHandle rmHandleOfDupObj = {0};

    /* Allocation size check */
    ASSERT_EQ(LwSciBufObjGetMemHandle(bufObj.get(), &rmHandle, &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    /* Dup buf handle */
    LwSciError error = LwSciError_Success;
    auto dupBufObj = LwSciBufPeer::duplicateBufObj(bufObj.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);

    /* Get RM handle of duped buf object */
    ASSERT_EQ(LwSciBufObjGetMemHandle(dupBufObj.get(), &rmHandleOfDupObj,
                                      &offset, &len),
              LwSciError_Success)
        << "Failed to Get Lwrm Memhandle for the object";

    /* Compare permissions of original & duped RM handles */
    ASSERT_TRUE(CompareRmHandlesAccessPermissions(rmHandle, rmHandleOfDupObj))
        << "Access Permissions mismatch between original and duped RM Handles"
        << std::endl;
}

/**
* Test case: The attribute list bound to the buffer reference is retrieved and
* the reconciliation status of the list is reconciled.
*/

TEST_F(BufferObject, GetReconciledList)
{
    // Get reconciled list and check if list is reconciled
    // Note: We don't need to free this Attribute List since it is managed by
    // LwSciBuf
    LwSciBufAttrList retrievedAttrList = nullptr;
    ASSERT_EQ(LwSciBufObjGetAttrList(bufObj.get(), &retrievedAttrList),
              LwSciError_Success)
        << "Failed to retrieve the attribute list";

    bool isReconciled = false;
    ASSERT_EQ(LwSciBufAttrListIsReconciled(retrievedAttrList, &isReconciled),
              LwSciError_Success)
        << "Failed to check the reconiliation status";
    ASSERT_TRUE(isReconciled)
        << "List associated with duplicate handle is not reconciled";
}

class BufferObjectDupe
    : public LwSciBufBasicTest,
      public ::testing::WithParamInterface<
          std::tuple<LwSciBufAttrValAccessPerm, LwSciError>> {
    public:
    void SetUp() override
    {
        LwSciError error = LwSciError_Success;

        LwSciBufBasicTest::SetUp();

        listA = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listA.get(), nullptr);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        listA.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> listA;
};

TEST_P(BufferObjectDupe, Validation)
{
    auto params = GetParam();
    LwSciBufAttrValAccessPerm dupePerm = std::get<0>(params);
    LwSciError expectedError = std::get<1>(params);

    LwSciError error = LwSciError_Success;

    LwSciBufType bufType = LwSciBufType_RawBuffer;
    uint64_t size = (128U * 1024U);
    uint64_t alignment = 1024U;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    {
        // Setup list A
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
        SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
        SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, alignment);
        SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);
    }

    auto bufObj = LwSciBufPeer::reconcileAndAllocate({listA.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    if (expectedError != LwSciError_Success) {
        NEGATIVE_TEST();
        auto dupeObj = LwSciBufPeer::duplicateBufObjWithReducedPerm(
            bufObj.get(), dupePerm, &error);
        ASSERT_EQ(error, expectedError);
    } else {
        auto dupeObj = LwSciBufPeer::duplicateBufObjWithReducedPerm(
            bufObj.get(), dupePerm, &error);
        ASSERT_EQ(error, expectedError);
        ASSERT_NE(dupeObj, nullptr);
    }
}
INSTANTIATE_TEST_CASE_P(BufferObjectDupe,
    BufferObjectDupe,
    testing::Values(std::make_tuple(static_cast<LwSciBufAttrValAccessPerm>(0),
                        LwSciError_BadParameter),
        std::make_tuple(LwSciBufAccessPerm_Readonly, LwSciError_Success),
        std::make_tuple(static_cast<LwSciBufAttrValAccessPerm>(2),
            LwSciError_BadParameter),
        std::make_tuple(LwSciBufAccessPerm_ReadWrite, LwSciError_Success),
        std::make_tuple(LwSciBufAccessPerm_Auto, LwSciError_BadParameter),
        std::make_tuple(LwSciBufAccessPerm_Ilwalid, LwSciError_BadParameter),
        std::make_tuple(
            static_cast<LwSciBufAttrValAccessPerm>(std::numeric_limits<
                std::underlying_type<LwSciBufAttrValAccessPerm>::type>::max()),
            LwSciError_BadParameter)));
