/*
 * Copyright (c) 2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#include <memory>

#include "lwscisync_ipc_peer_old.h"
#include "lwscisync_test_attribute_list.h"
#include "lwscisync_test_common.h"

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_CPU_WAIT_CONTEXT_TEST(testSuite, testName, JamaID)           \
    class _##testSuite##JamaID : public LwSciSyncAttrListTest<JamaID>          \
    {                                                                          \
    };                                                                         \
    TEST_F(_##testSuite##JamaID, testName)

/* Declare additional test case for a test */
#define LWSCISYNC_CPU_WAIT_CONTEXT_TEST_CASE(testSuite, testName, JamaID)      \
    TEST_F(_##testSuite##JamaID, testName)

/**
 * @jama{15887730} Allocate CPU Wait Context
 *
 * @verifies @jama{13561849}
 *
 * This test checks that we successfully allocate a CPU Wait Context.
 */

LWSCISYNC_CPU_WAIT_CONTEXT_TEST(LwSciSyncCpuWaitContextTest, Allocation,
                                15887730)
{
    LwSciError error = LwSciError_Success;
    auto signalerAttrList = peer.createAttrList();
    auto waiterAttrList = peer.createAttrList();

    // Set up Signaler Attribute List
    ASSERT_EQ(
        LwSciSyncAttrListSetAttrs(signalerAttrList.get(),
                                  LwSciSyncPeer::attrs.cpuSignaler.data(),
                                  LwSciSyncPeer::attrs.cpuSignaler.size()),
        LwSciError_Success);

    // Set up Waiter Attribute List
    ASSERT_EQ(LwSciSyncAttrListSetAttrs(waiterAttrList.get(),
                                        LwSciSyncPeer::attrs.cpuWaiter.data(),
                                        LwSciSyncPeer::attrs.cpuWaiter.size()),
              LwSciError_Success);

    // Signaler: Reconcile Attribute Lists
    auto newReconciledList = LwSciSyncPeer::reconcileLists(
        {signalerAttrList.get(), waiterAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(newReconciledList.get(), nullptr);

    // Signaler: Allocate a sync obj
    LwSciSyncObj syncObj = nullptr;
    ASSERT_EQ(LwSciSyncObjAlloc(newReconciledList.get(), &syncObj),
              LwSciError_Success);
    auto syncObjPtr = std::shared_ptr<LwSciSyncObjRec>(syncObj, LwSciSyncObjFree);

    // Signaler: Generate a fence
    LwSciError err = LwSciError_Success;
    auto syncFencePtr = LwSciSyncPeer::generateFence(syncObj, &err);
    ASSERT_EQ(err, LwSciError_Success);

    // Waiter: Allocate LwSciSyncCpuWaitContext
    LwSciSyncCpuWaitContext waitContext = nullptr;
    ASSERT_EQ(LwSciSyncCpuWaitContextAlloc(peer.module(), &waitContext),
              LwSciError_Success);
    auto cpuWaitContextPtr = std::shared_ptr<LwSciSyncCpuWaitContextRec>(
        waitContext, LwSciSyncCpuWaitContextFree);

    // Signaler: Signal on the Sync Object
    ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);

    // Waiter: Wait on the fence
    ASSERT_EQ(LwSciSyncFenceWait(syncFencePtr.get(), waitContext, 1),
              LwSciError_Success);
}
