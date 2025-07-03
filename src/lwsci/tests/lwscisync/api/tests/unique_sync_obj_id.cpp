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

#include "ipc_wrapper.h"
#include "lwscisync_ipc_peer_old.h"

class LwSciSyncObjUniqueId : public LwSciSyncTransportTest<18851676>
{
public:
    LwSciSyncPeer peer;
    LwSciSyncPeer otherPeer;

    void SetUp() override
    {
        peer.SetUp();
        otherPeer.SetUp();
        LwSciSyncBaseTest::SetUp();
    }

    void TearDown() override
    {
        peer.TearDown();
        otherPeer.TearDown();
        LwSciSyncBaseTest::TearDown();
    }
};

/**
* @jama{18851676} - Unique Session IDs for Sync Points
*
* @brief Verify that LwSciSync shall assign a unique session identifier
* for different synchronization object belonging to same LwSciSyncModule
* within same process.
*
* 1. Create 2 separate attribute list and reconcile and allocate
*    LwSciSyncObj A and LwSciSyncObj B.
* 2. Produce fences for each object.
* 3. Extract fence id from both fences and check that they are different.
* 4. Signal only LwSciSyncObj A.
* 5. Start waiting on both fences.
* 6. Verify that only waiting on LwSciSyncObj A's fence completes successfully.
*
* @verify{@jama{18844017}} - Unique Session IDs for Sync Points
*/
TEST_F(LwSciSyncObjUniqueId, SyncObjUniqueId1)
{
    LwSciError error = LwSciError_Success;
    uint64_t fenceAId = 0;
    uint64_t fenceAValue = 0;
    uint64_t fenceBId = 0;
    uint64_t fenceBValue = 0;
    LwSciSyncCpuWaitContext waitContext = nullptr;
    auto listA = peer.createAttrList(); // CPU Signaler
    auto listB = peer.createAttrList(); // CPU Waiter
    ASSERT_TRUE(listA);
    ASSERT_TRUE(listB);

    ASSERT_EQ(
        LwSciSyncCpuWaitContextAlloc(peer.module(), &waitContext),
        LwSciError_Success);

    auto attrs = LwSciSyncPeer::attrs.cpuSignaler;
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(listA.get(), attrs.data(), attrs.size()),
        LwSciError_Success);
    attrs = LwSciSyncPeer::attrs.cpuWaiter;
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(listB.get(), attrs.data(), attrs.size()),
        LwSciError_Success);

    // Reconcile and Allocate
    auto newObjA =
        LwSciSyncPeer::reconcileAndAllocate({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newObjA.get(), nullptr);
    auto newObjB =
        LwSciSyncPeer::reconcileAndAllocate({listA.get(), listB.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newObjB.get(), nullptr);

    // Generate fence from each syncObj
    LwSciSyncFence syncFenceA = LwSciSyncFenceInitializer;
    ASSERT_EQ(LwSciSyncObjGenerateFence(newObjA.get(), &syncFenceA),
              LwSciError_Success);
    LwSciSyncFence syncFenceB = LwSciSyncFenceInitializer;
    ASSERT_EQ(LwSciSyncObjGenerateFence(newObjB.get(), &syncFenceB),
              LwSciError_Success);

    // Extract fence id from both fences and check that they are different.
    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFenceA, &fenceAId, &fenceAValue),
              LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFenceB, &fenceBId, &fenceBValue),
              LwSciError_Success);
    ASSERT_NE(fenceAId, fenceBId);

    // Signal one of the objects
    ASSERT_EQ(LwSciSyncObjSignal(newObjA.get()), LwSciError_Success);

    // Verify that only its fence expired
    ASSERT_EQ(
        LwSciSyncFenceWait(&syncFenceA, waitContext, 10),
        LwSciError_Success);
    NegativeTestPrint();
    ASSERT_EQ(
        LwSciSyncFenceWait(&syncFenceB, waitContext, 10),
        LwSciError_Timeout);

    // Clear both fences to decrement refcount
    LwSciSyncFenceClear(&syncFenceA);
    LwSciSyncFenceClear(&syncFenceB);

    LwSciSyncCpuWaitContextFree(waitContext);
}

/**
* @jama{18851676} - Unique Session IDs for Sync Points
*
* @brief Verify that LwSciSync shall assign a unique session identifier
* for different synchronization object belonging to different LwSciSyncModule
* within same process.
*
* 1. Create 2 separate attribute list belonging to different module and
*    reconcile and allocate LwSciSyncObj A and LwSciSyncObj B.
* 2. Produce fences for each object.
* 3. Extract fence id from both fences and check that they are different.
* 4. Signal only LwSciSyncObj A.
* 5. Start waiting on both fences.
* 6. Verify that only waiting on LwSciSyncObj A's fence completes successfully.
*
* @verify{@jama{18844017}} - Unique Session IDs for Sync Points
*/
TEST_F(LwSciSyncObjUniqueId, SyncObjUniqueId2)
{
    LwSciError error = LwSciError_Success;
    uint64_t fenceAId = 0;
    uint64_t fenceAValue = 0;
    uint64_t fenceBId = 0;
    uint64_t fenceBValue = 0;
    LwSciSyncCpuWaitContext waitContextA = nullptr;
    LwSciSyncCpuWaitContext waitContextB = nullptr;
    auto list1A = peer.createAttrList(); // CPU Signaler
    auto list1B = peer.createAttrList(); // CPU Waiter
    auto list2A = otherPeer.createAttrList(); // CPU Signaler
    auto list2B = otherPeer.createAttrList(); // CPU Waiter
    ASSERT_TRUE(list1A);
    ASSERT_TRUE(list1B);
    ASSERT_TRUE(list2A);
    ASSERT_TRUE(list2B);

    ASSERT_EQ(
        LwSciSyncCpuWaitContextAlloc(peer.module(), &waitContextA),
        LwSciError_Success);
    ASSERT_EQ(
        LwSciSyncCpuWaitContextAlloc(otherPeer.module(), &waitContextB),
        LwSciError_Success);

    auto attrs = LwSciSyncPeer::attrs.cpuSignaler;
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(list1A.get(), attrs.data(), attrs.size()),
        LwSciError_Success);
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(list2A.get(), attrs.data(), attrs.size()),
        LwSciError_Success);

    attrs = LwSciSyncPeer::attrs.cpuWaiter;
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(list1B.get(), attrs.data(), attrs.size()),
        LwSciError_Success);
    attrs = LwSciSyncPeer::attrs.cpuWaiter;
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(list2B.get(), attrs.data(), attrs.size()),
        LwSciError_Success);

    // Reconcile and Allocate
    auto newObjA =
        LwSciSyncPeer::reconcileAndAllocate({list1A.get(), list1B.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newObjA.get(), nullptr);
    auto newObjB =
        LwSciSyncPeer::reconcileAndAllocate({list2A.get(), list2B.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newObjB.get(), nullptr);

    // Generate fence from each syncObj
    LwSciSyncFence syncFenceA = LwSciSyncFenceInitializer;
    ASSERT_EQ(LwSciSyncObjGenerateFence(newObjA.get(), &syncFenceA),
              LwSciError_Success);
    LwSciSyncFence syncFenceB = LwSciSyncFenceInitializer;
    ASSERT_EQ(LwSciSyncObjGenerateFence(newObjB.get(), &syncFenceB),
              LwSciError_Success);

    // Extract fence id from both fences and check that they are different.
    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFenceA, &fenceAId, &fenceAValue),
              LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFenceB, &fenceBId, &fenceBValue),
              LwSciError_Success);
    ASSERT_NE(fenceAId, fenceBId);

    // Signal one of the objects
    ASSERT_EQ(LwSciSyncObjSignal(newObjA.get()), LwSciError_Success);

    // Verify that only its fence expired
    ASSERT_EQ(
        LwSciSyncFenceWait(&syncFenceA, waitContextA, 10),
        LwSciError_Success);
    NegativeTestPrint();
    ASSERT_EQ(
        LwSciSyncFenceWait(&syncFenceB, waitContextB, 10),
        LwSciError_Timeout);

    // Clear both fences to decrement refcount
    LwSciSyncFenceClear(&syncFenceA);
    LwSciSyncFenceClear(&syncFenceB);

    LwSciSyncCpuWaitContextFree(waitContextA);
    LwSciSyncCpuWaitContextFree(waitContextB);
}
