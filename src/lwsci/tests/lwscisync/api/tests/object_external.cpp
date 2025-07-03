/*
 * Copyright (c) 2020-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <array>
#include <cinttypes>
#include <memory>
#include <stdio.h>
#include <string.h>

#include "lwscisync_internal.h"
#include "lwscisync_peer.h"
#include "lwscisync_test_signaler.h"
#include "lwscisync_test_waiter.h"

template <int64_t JamaID>
class LwSciSyncObjectExternalTest : public LwSciSyncBaseTest<JamaID>
{
public:
    LwSciSyncPeer peer;

    void SetUp() override
    {
        peer.SetUp();
        LwSciSyncBaseTest<JamaID>::SetUp();
    }

    void TearDown() override
    {
        peer.TearDown();
        LwSciSyncBaseTest<JamaID>::TearDown();
    }
};

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_OBJECT_EXTERNAL_TEST(testSuite, JamaID)                      \
    class testSuite : public LwSciSyncObjectExternalTest<JamaID>               \
    {                                                                          \
    };

/* Declare additional test case for a test */
#define LWSCISYNC_OBJECT_EXTERNAL_TEST_CASE(testSuite, testName)               \
    TEST_F(testSuite, testName)

/**
 * @jama{0} LwSciSyncObject duplicate
 * 1. Create CPU Signaler and CPU Waiter attr lists.
 * 2. Reconcile and allocate a LwSciSyncObject.
 * 3. Duplicate object.
 * 4. Compare reconciled attribute lists of original and
 * duplicated objects. Verify number of primitives ( == 1)
 * and primitive type ( == Syncpoint (Semaphore for x86)).
 * 5. Generate fence from the duplicate object.
 * 6. Signal on the original object, fence should unlock.
 *
 * @verify{@jama{13561799}} - LWSTRMS-REQPLCL123-296 Object duplicating
 */
LWSCISYNC_OBJECT_EXTERNAL_TEST(LwSciSyncObjectDuplicate, 1)

LWSCISYNC_OBJECT_EXTERNAL_TEST_CASE(LwSciSyncObjectDuplicate, Success)
{
    LwSciError error = LwSciError_Success;
    auto listA = peer.createAttrList(); // CPU Signaler
    auto listB = peer.createAttrList(); // CPU Waiter

    auto attrs = LwSciSyncPeer::attrs.cpuSignaler;
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(listA.get(), attrs.data(), attrs.size()),
        LwSciError_Success);
    attrs = LwSciSyncPeer::attrs.cpuWaiter;
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(listB.get(), attrs.data(), attrs.size()),
        LwSciError_Success);

    // Reconcile and Allocate
    auto newObj =
        LwSciSyncPeer::reconcileAndAllocate({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newObj.get(), nullptr);

    LwSciSyncAttrList objectAttrList = nullptr;
    LwSciSyncObjGetAttrList(newObj.get(), &objectAttrList);
    ASSERT_NE(objectAttrList, nullptr);
    LwSciSyncPeer::checkAttrListIsReconciled(objectAttrList, true);

    auto newObjDup = LwSciSyncPeer::duplicateSyncObj(newObj.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newObjDup.get(), nullptr);
    LwSciSyncAttrList dupObjectAttrList = nullptr;
    LwSciSyncObjGetAttrList(newObjDup.get(), &dupObjectAttrList);
    ASSERT_NE(objectAttrList, nullptr);
    LwSciSyncPeer::checkAttrListIsReconciled(dupObjectAttrList, true);

    LwSciSyncPeer::checkAttrListsEqual(objectAttrList, dupObjectAttrList);

    uint32_t numPrimitives = 0;
    ASSERT_EQ(LwSciSyncObjGetNumPrimitives(newObj.get(), &numPrimitives),
              LwSciError_Success);
    ASSERT_EQ(numPrimitives, 1);

    LwSciSyncInternalAttrValPrimitiveType primitiveType =
        LwSciSyncInternalAttrValPrimitiveType_LowerBound;
    ASSERT_EQ(LwSciSyncObjGetPrimitiveType(newObj.get(), &primitiveType),
              LwSciError_Success);
#if defined(__x86_64__)
    ASSERT_EQ(primitiveType,
              LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
#else
    ASSERT_EQ(primitiveType, LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
#endif

    auto fence = LwSciSyncPeer::generateFence(newObjDup.get(), &error);
    ASSERT_EQ(error, LwSciError_Success);
    auto waitContext = peer.allocateCpuWaitContext(&error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_EQ(LwSciSyncObjSignal(newObj.get()), LwSciError_Success);

    ASSERT_EQ(LwSciSyncFenceWait(fence.get(), waitContext.get(), 1),
              LwSciError_Success);
}
