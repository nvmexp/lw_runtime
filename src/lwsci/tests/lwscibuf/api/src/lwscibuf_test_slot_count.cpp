/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>

#include "lwscibuf_basic_test.h"
#include "gtest/gtest.h"

//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

class SlotCount : public LwSciBufBasicTest
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

        listC = peer.createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(listC.get(), nullptr);
    }

    void TearDown() override
    {
        LwSciBufBasicTest::TearDown();

        listA.reset();
        listB.reset();
        listC.reset();
    }

    std::shared_ptr<LwSciBufAttrListRec> listA;
    std::shared_ptr<LwSciBufAttrListRec> listB;
    std::shared_ptr<LwSciBufAttrListRec> listC;
};

/**
* Test case : Retrive the slotcount of an attribute list
*/
TEST_F(SlotCount, GetSlotCountReconciled)
{
    LwSciError error = LwSciError_Success;

    bool isReconciledListValid = false;

    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

    // Reconcile listA and listB
    auto reconciledList =
        LwSciBufPeer::attrListReconcile({listA.get(), listB.get()}, &error);
    ASSERT_NE(reconciledList, nullptr);

    // Validate Reconciled
    ASSERT_EQ(LwSciBufPeer::validateReconciled({listA.get(), listB.get()},
                                               reconciledList.get(),
                                               &isReconciledListValid),
              LwSciError_Success);
    ASSERT_TRUE(isReconciledListValid);

    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufGeneralAttrKey_Types, bufType));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(reconciledList.get(),
                                         LwSciBufRawBufferAttrKey_Size, size));
    ASSERT_TRUE(LwSciBufPeer::verifyAttr(
        reconciledList.get(), LwSciBufRawBufferAttrKey_Align, align));

    ASSERT_EQ(LwSciBufAttrListGetSlotCount(listA.get()), 1);
    ASSERT_EQ(LwSciBufAttrListGetSlotCount(listB.get()), 1);
    ASSERT_EQ(LwSciBufAttrListGetSlotCount(reconciledList.get()), 1);
}

/**
* Test case : Retrive the slotcount of an attribute list
*/
TEST_F(SlotCount, GetSlotCountUnreconciled)
{
    LwSciError error = LwSciError_Success;

    uint64_t size = 128 * 1024;
    uint64_t align = 4 * 1024;
    LwSciBufType bufType = LwSciBufType_RawBuffer;
    LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;

    SET_ATTR(listA.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listA.get(), LwSciBufRawBufferAttrKey_Align, align);

    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listB.get(), LwSciBufRawBufferAttrKey_Align, align);
    SET_ATTR(listB.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

    SET_ATTR(listC.get(), LwSciBufGeneralAttrKey_Types, bufType);
    SET_ATTR(listC.get(), LwSciBufRawBufferAttrKey_Size, size);
    SET_ATTR(listC.get(), LwSciBufRawBufferAttrKey_Align, align);
    SET_ATTR(listC.get(), LwSciBufGeneralAttrKey_RequiredPerm, perm);

    auto unreconciledList = LwSciBufPeer::attrListAppend(
        {listA.get(), listB.get(), listC.get()}, &error);
    ASSERT_NE(unreconciledList, nullptr);

    // Validate reconciliation status
    bool isReconciled = false;
    ASSERT_EQ(
        LwSciBufAttrListIsReconciled(unreconciledList.get(), &isReconciled),
        LwSciError_Success);
    ASSERT_FALSE(isReconciled);

    // Verify slot count of all lists
    ASSERT_EQ(LwSciBufAttrListGetSlotCount(listA.get()), 1);
    ASSERT_EQ(LwSciBufAttrListGetSlotCount(listB.get()), 1);
    ASSERT_EQ(LwSciBufAttrListGetSlotCount(listC.get()), 1);
    ASSERT_EQ(LwSciBufAttrListGetSlotCount(unreconciledList.get()), 3);
}
