/*
 * Copyright (c) 2020-2022 LWPU Corporation.  All Rights Reserved.
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
#ifdef LW_QNX
#include <sys/neutrino.h>
#include <sys/syspage.h>
#endif
#include "lwscicommon_arch.h"
#include "lwscisync_internal.h"
#include "lwscisync_interprocess_test.h"
#include "lwscisync_peer.h"
#include "lwscisync_test_signaler.h"
#include "lwscisync_test_waiter.h"
#include <lwscibuf_internal.h>

constexpr static int FENCE_GENERATE_ITERATIONS = 10;

template <int64_t JamaID>
class TestFenceOperations : public LwSciSyncBaseTest<JamaID>
{
public:
    void SetUp() override
    {
        peer.SetUp();
        peer.createAttrList(&(unreconciledLists[0]));
        peer.createAttrList(&(unreconciledLists[1]));
        LwSciSyncBaseTest<JamaID>::SetUp();
        syncFence = LwSciSyncFenceInitializer;
        waitContext = nullptr;

        ASSERT_EQ(LwSciSyncTest_FillCpuSignalerAttrList(unreconciledLists[0]),
                  LwSciError_Success);
        ASSERT_EQ(LwSciSyncTest_FillCpuWaiterAttrList(unreconciledLists[1]),
                  LwSciError_Success);
        ASSERT_EQ(LwSciSyncAttrListReconcile(unreconciledLists.data(),
                                             unreconciledLists.size(),
                                             &reconciledList, &newConflictList),
                  LwSciError_Success);
        ASSERT_EQ(LwSciSyncObjAlloc(reconciledList, &syncObj),
                  LwSciError_Success);
        ASSERT_EQ(LwSciSyncObjGenerateFence(syncObj, &syncFence),
                  LwSciError_Success);
        ASSERT_EQ(LwSciSyncCpuWaitContextAlloc(peer.module(), &waitContext),
                  LwSciError_Success);
    }

    void TearDown() override
    {
        for (auto& list : unreconciledLists) {
            LwSciSyncAttrListFree(list);
            list = nullptr;
        }

        LwSciSyncFenceClear(&syncFence);

        if (syncObj) {
            LwSciSyncObjFree(syncObj);
        }

        if (reconciledList) {
            LwSciSyncAttrListFree(reconciledList);
            reconciledList = nullptr;
        }

        if (newConflictList) {
            LwSciSyncAttrListFree(newConflictList);
            newConflictList = nullptr;
        }

        if (waitContext) {
            LwSciSyncCpuWaitContextFree(waitContext);
            waitContext = nullptr;
        }

        peer.TearDown();
        LwSciSyncBaseTest<JamaID>::TearDown();
    }

    LwSciSyncPeer peer;
    std::array<LwSciSyncAttrList, 2> unreconciledLists;
    LwSciSyncAttrList reconciledList;
    LwSciSyncAttrList newConflictList;
    LwSciSyncObj syncObj;
    LwSciSyncFence syncFence;
    LwSciSyncCpuWaitContext waitContext;
};

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_FENCE_OPERATIONS_TEST(testSuite, testName, JamaID)           \
    class _##testSuite##JamaID : public TestFenceOperations<JamaID>            \
    {                                                                          \
    };                                                                         \
    TEST_F(_##testSuite##JamaID, testName)

static bool isFenceEmpty(LwSciSyncFence* fence)
{
    size_t i;
    size_t size = sizeof(fence->payload) / sizeof(fence->payload[0]);

    for (i = 0U; i < size; ++i) {
        if (fence->payload[i] != 0U)
            return false;
    }
    return true;
}

/** @jama{10507587} fence duplication behavior
 * -fence_duplicating
     Tests that fence duplicating works properly in various conditions.
     causes an expected error: [ERROR: LwSciSyncFenceDup]: src fence the same as
 dst fence: x%x
*/
LWSCISYNC_FENCE_OPERATIONS_TEST(TestBaseSupportFenceOperations,
                                Duplication,
                                10507587)
{
    uint64_t fenceId = 0;
    uint64_t fenceValue = 0;
    uint64_t newFenceId = 0;
    uint64_t newFenceValue = 0;

    LwSciSyncObj syncObjDup = nullptr;
    LwSciSyncFence syncFenceDup = LwSciSyncFenceInitializer;

    auto syncFenceDupPtr =
        std::shared_ptr<LwSciSyncFence>(&syncFenceDup, LwSciSyncFenceClear);

    /* Test FenceDup with BadParameters */
    NegativeTestPrint();
    ASSERT_EQ(LwSciSyncFenceDup(&syncFence, &syncFence), LwSciError_BadParameter);

    NegativeTestPrint();
    ASSERT_EQ(LwSciSyncFenceDup(nullptr, &syncFenceDup), LwSciError_BadParameter);

    NegativeTestPrint();
    ASSERT_EQ(LwSciSyncFenceDup(&syncFence, nullptr), LwSciError_BadParameter);

    /* Test FenceDup & FenceExtract with correct parameters */
    ASSERT_EQ(LwSciSyncFenceDup(&syncFence, &syncFenceDup), LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFence, &fenceId, &fenceValue),
              LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFenceDup, &newFenceId,
                                                        &newFenceValue),
              LwSciError_Success);
    ASSERT_EQ(fenceId, newFenceId);
    ASSERT_EQ(fenceValue, newFenceValue);
    ASSERT_EQ(memcmp(&syncFence, &syncFenceDup, sizeof(syncFence)), 0);

    /* Test waiting on fence until timeout */
    NegativeTestPrint();
    ASSERT_EQ(LwSciSyncFenceWait(&syncFence, waitContext, 1),
              LwSciError_Timeout);
    NegativeTestPrint();
    ASSERT_EQ(LwSciSyncFenceWait(&syncFenceDup, waitContext, 1),
              LwSciError_Timeout);

    /* Test waiting on fence after signaling */
    ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceWait(&syncFence, waitContext, 1),
              LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceWait(&syncFenceDup, waitContext, 1),
              LwSciError_Success);

    /* Test to wait on orignal fence & duplicated fence after signaling
     * syncObject extracted from duplicated fence */
    ASSERT_EQ(LwSciSyncFenceGetSyncObj(&syncFenceDup, &syncObjDup),
              LwSciError_Success);
    ASSERT_EQ(LwSciSyncObjSignal(syncObjDup), LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceWait(&syncFence, waitContext, 1),
              LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceWait(&syncFenceDup, waitContext, 1),
              LwSciError_Success);

    /* Test FenceClear and check if fence payload is cleared */
    LwSciSyncFenceClear(&syncFence);
    ASSERT_EQ(isFenceEmpty(&syncFence), true);
    ASSERT_EQ(LwSciSyncFenceDup(&syncFence, &syncFenceDup), LwSciError_Success);
    ASSERT_EQ(isFenceEmpty(&syncFenceDup), true);
}

/** @jama{14686253} Fence Extract and Update
 * This test case verifies @jama{13561783} "Fence updating" and
 * @jama{13561785} "Fence extracting"
 */
LWSCISYNC_FENCE_OPERATIONS_TEST(TestBaseSupportFenceOperations,
                                ExtractAndUpdate,
                                14686253)
{
    uint64_t fenceId = 0;
    uint64_t fenceValue = 0;
    uint64_t newFenceId = 0;
    uint64_t newFenceValue = 0;
    LwSciSyncFence newFence = LwSciSyncFenceInitializer;
    auto newFencePtr =
        std::shared_ptr<LwSciSyncFence>(&newFence, LwSciSyncFenceClear);

    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFence, &fenceId, &fenceValue),
              LwSciError_Success);
    ASSERT_EQ(
        LwSciSyncFenceUpdateFence(syncObj, fenceId, fenceValue, &newFence),
        LwSciError_Success);
    ASSERT_EQ(isFenceEmpty(&newFence), false);
    ASSERT_EQ(
        LwSciSyncFenceExtractFence(&newFence, &newFenceId, &newFenceValue),
        LwSciError_Success);
    ASSERT_EQ(fenceId, newFenceId);
    ASSERT_EQ(fenceValue, newFenceValue);
}


static void LwSciSyncFenceCleanup(
    LwSciSyncFence* syncFence) {
    LwSciSyncFenceClear(syncFence);
    delete syncFence;
}

/** @jama{14686333} Fence Generating behaviour
 * This test case verifies @jama{13561797}.
 * LwSciSyncFence is generated and underlying LwSciSync object's
 * max expected state shall be advanced.
 */
LWSCISYNC_FENCE_OPERATIONS_TEST(TestBaseSupportFenceOperations,
                                FenceGenerating,
                                14686333)
{
    uint64_t fenceId = 0;
    uint64_t fenceValue = 0;
    uint64_t prevFenceValue = 0;

    ASSERT_EQ(LwSciSyncFenceExtractFence(&syncFence, &fenceId, &prevFenceValue),
              LwSciError_Success);

    for (int i = 0; i < FENCE_GENERATE_ITERATIONS; i++) {
        std::shared_ptr<LwSciSyncFence> newFence =
            std::shared_ptr<LwSciSyncFence>(new LwSciSyncFence,
                                            LwSciSyncFenceCleanup);
        *(newFence.get()) = LwSciSyncFenceInitializer;
        ASSERT_EQ(LwSciSyncObjGenerateFence(syncObj, newFence.get()),
                  LwSciError_Success);
        ASSERT_EQ(
            LwSciSyncFenceExtractFence(newFence.get(), &fenceId, &fenceValue),
            LwSciError_Success);
        ASSERT_EQ(fenceValue, (prevFenceValue + 1));
        prevFenceValue = fenceValue;
    }
}

static inline uint64_t GetTimeNow()
{
    uint64_t timeus = 0;
    struct timespec ts;

#ifdef LW_QNX
    uint64_t freq = 0;
    asm volatile("mrs %0, CNTFRQ_EL0" : "=r"(freq) :: );
    asm volatile("mrs %0, CNTVCT_EL0" : "=r"(timeus) :: );
    // colwert CPU ticks to micro seconds
    timeus = timeus*1000000/freq;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
    // Colwert time to micro seconds
    timeus = ts.tv_sec*1000000 + ts.tv_nsec/1000;
#endif
    return timeus;
}

/** @jama{15423310} Fence Timeout
 * This test case verifies @jama{13561905}, @jama{13561907}.
*/
LWSCISYNC_FENCE_OPERATIONS_TEST(TestBaseSupportFenceOperations,
                                Timeout,
                                15423310)
{
    {
        /* Test waiting on fence until timeout of 10-60 millisec */
        int64_t randomTimeout = 0;

        unsigned int seed = time(NULL);
        TEST_COUT << "Using random seed: " << seed;
        srand(seed);
        randomTimeout = rand() % 50000 + 10000;

        uint64_t startTime = GetTimeNow();
        uint64_t elapsedTime = 0;

        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncFenceWait(&syncFence, waitContext, randomTimeout),
                  LwSciError_Timeout);

        elapsedTime = GetTimeNow() - startTime;
        ASSERT_GE(elapsedTime, randomTimeout);
        ASSERT_LT(elapsedTime, 2 * randomTimeout);
    }

    {
        /* Test to wait on already non-expired fence */
        NegativeTestPrint();
        ASSERT_EQ(LwSciSyncFenceWait(&syncFence, waitContext, 0),
                  LwSciError_Timeout);
    }

    /* Test waiting on fence after signaling */
    ASSERT_EQ(LwSciSyncObjSignal(syncObj), LwSciError_Success);
    ASSERT_EQ(LwSciSyncFenceWait(&syncFence, waitContext, 1),
              LwSciError_Success);

    /* Test to wait on already expired fence */
    ASSERT_EQ(LwSciSyncFenceWait(&syncFence, waitContext, 0),
              LwSciError_Success);
}

class TestLwSciSyncFenceOperationsTimestamp : public LwSciSyncInterProcessTest,
    public ::testing::WithParamInterface<std::tuple<LwSciSyncInternalAttrKey, LwSciSyncTimestampFormat> >

{};

TEST_P(TestLwSciSyncFenceOperationsTimestamp, EmbeddedInPrimitive)
{
    auto params = GetParam();
    LwSciSyncInternalAttrKey signalerTimestampInfoKey =
        std::get<0>(params);
    LwSciSyncTimestampFormat timestampFormat =
        std::get<1>(params);

    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    const LwSciSyncAttrValTimestampInfo timestampInfo[] = {
        {
            .format = timestampFormat,
            .scaling = {
                .scalingFactorNumerator = 1U,
                .scalingFactorDenominator = 1U,
                .sourceOffset = 0U,
            },
        },
    };

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_SignalOnly);

        LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
        };
        uint32_t primitiveCount = 1U;
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                          primitiveInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                          primitiveCount);
        SET_INTERNAL_ATTR(signalerAttrList.get(), signalerTimestampInfoKey,
                          timestampInfo);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            auto reconciledList = LwSciSyncPeer::attrListReconcile(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);

            bool isReconciled = true;
            ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                          {signalerAttrList.get(), waiterAttrList.get()},
                          reconciledList.get(), &isReconciled),
                      LwSciError_Success);
            ASSERT_EQ(isReconciled, true);

            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(reconciledList.get(),
                        signalerTimestampInfoKey, timestampInfo));

            auto reconciledListDesc = peer->exportReconciledList(reconciledList.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(peer->sendBuf(reconciledListDesc), LwSciError_Success);

            ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
        }

        {
            auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);

            LwSciSyncAttrList reconciledList = nullptr;
            ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &reconciledList), LwSciError_Success);

            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(reconciledList,
                        signalerTimestampInfoKey, timestampInfo));

            auto attrListAndObjDesc = peer->exportAttrListAndObj(
                syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(peer->sendBuf(attrListAndObjDesc), LwSciError_Success);

            auto fence = LwSciSyncPeer::generateFence(syncObj.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);

            // EmbeddedInPrimitive only has 1 slot, slot 0
            if (timestampFormat == LwSciSyncTimestampFormat_EmbeddedInPrimitive) {
                uint32_t slotIndex = 0U;
                ASSERT_EQ(LwSciSyncObjGetNextTimestampSlot(syncObj.get(), &slotIndex),
                        LwSciError_Success);
                ASSERT_EQ(slotIndex, (uint32_t)0U);
            }

            uint64_t timestamp = 0U;
            uint64_t timestampBeforeSignal = LwSciCommonGetTimeUS();
            ASSERT_EQ(LwSciSyncObjSignal(syncObj.get()), LwSciError_Success);

            // Assert that we can obtain the LwSciBufObj
            LwSciSyncTimestampBufferInfo bufferInfo{};
            ASSERT_EQ(LwSciSyncObjGetTimestampBufferInfo(syncObj.get(),
                        &bufferInfo), LwSciError_Success);
            void* timestampBase = nullptr;
            ASSERT_EQ(LwSciBufObjGetCpuPtr(bufferInfo.bufObj,
                    (void**) &timestampBase), LwSciError_Success);

            ASSERT_EQ(LwSciSyncFenceGetTimestamp(fence.get(), &timestamp),
                    LwSciError_Success);
            uint64_t timestampAfterSignal = LwSciCommonGetTimeUS();
            EXPECT_GE(timestamp, timestampBeforeSignal);
            EXPECT_LE(timestamp, timestampAfterSignal);

            auto fenceDesc = peer->exportFence(fence.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(peer->sendExportDesc(fenceDesc), LwSciError_Success);

            // EmbeddedInPrimitive only has 1 slot, slot 0
            if (timestampFormat == LwSciSyncTimestampFormat_EmbeddedInPrimitive) {
                uint32_t slotIndex = 0U;
                ASSERT_EQ(LwSciSyncObjGetNextTimestampSlot(syncObj.get(), &slotIndex),
                        LwSciError_Success);
                ASSERT_EQ(slotIndex, (uint32_t)0U);
            }

            // try setting invalid index
            if (timestampFormat == LwSciSyncTimestampFormat_EmbeddedInPrimitive) {
                uint64_t id = 0U;
                uint64_t value = 0U;
                ASSERT_EQ(LwSciSyncFenceExtractFence(fence.get(), &id, &value),
                        LwSciError_Success);

                NegativeTestPrint();
                ASSERT_EQ(LwSciSyncFenceUpdateFenceWithTimestamp(syncObj.get(),
                            id, value, 1, fence.get()), LwSciError_BadParameter);
            }

            // Done
            ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
        }
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);

        LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
        };
        SET_INTERNAL_ATTR(waiterAttrList.get(),
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                          primitiveInfo);

        // Export unreconciled waiter list to Peer A
        auto listDescBuf = peer->exportUnreconciledList(
                {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        {
            auto reconciledListDescBuf = peer->recvBuf(&error);
            ASSERT_EQ(error, LwSciError_Success);
            auto reconciledList = peer->importReconciledList(
                reconciledListDescBuf, {waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);

            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(reconciledList.get(),
                        signalerTimestampInfoKey, timestampInfo));

            bool isReconciled = true;
            ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                          {waiterAttrList.get()},
                          reconciledList.get(), &isReconciled),
                      LwSciError_Success);
            ASSERT_EQ(isReconciled, true);

            ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
        }

        {
            auto attrListAndObjDesc = peer->recvBuf(&error);
            ASSERT_EQ(error, LwSciError_Success);

            auto syncObj = peer->importAttrListAndObj(
                attrListAndObjDesc, {waiterAttrList.get()},
                LwSciSyncAccessPerm_WaitOnly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            LwSciSyncAttrList reconciledList = nullptr;
            ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &reconciledList), LwSciError_Success);

            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(reconciledList,
                        signalerTimestampInfoKey, timestampInfo));

            auto fenceDesc =
                peer->recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
            ASSERT_EQ(error, LwSciError_Success);

            auto fence = peer->importFence(fenceDesc.get(), syncObj.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);

            uint64_t id = 0U;
            uint64_t value = 0U;
            ASSERT_EQ(
                    LwSciSyncFenceExtractFence(fence.get(), &value, &value),
                    LwSciError_Success);

            auto waitContext = peer->allocateCpuWaitContext(&error);
            ASSERT_EQ(
                    LwSciSyncFenceWait(fence.get(), waitContext.get(), -1),
                    LwSciError_Success);

            uint64_t timestampUS = 0U;
            ASSERT_EQ(LwSciSyncFenceGetTimestamp(fence.get(), &timestampUS),
                      LwSciError_Success);

            ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
        }
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
// These two attribute keys should have the identical behaviour
INSTANTIATE_TEST_CASE_P(
    TestLwSciSyncFenceOperationsTimestamp,
    TestLwSciSyncFenceOperationsTimestamp,
    ::testing::Values(
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo,
        LwSciSyncTimestampFormat_8Byte),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
        LwSciSyncTimestampFormat_8Byte),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfo,
        LwSciSyncTimestampFormat_EmbeddedInPrimitive),
    std::make_tuple(
        LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
        LwSciSyncTimestampFormat_EmbeddedInPrimitive)
    ));

// TODO: Abstract out handle duplication operations like LwSciBuf allocation
// does for system memory between platforms. This test leverages
// platform-specific APIs for mapping/unmapping memory, which haven't been
// abstracted out in a platform-agnostic API.
//
// Lwrrently this is sort of supported by umd_{resman,cheetah}.cpp, but they
// weren't really written for reuse.
#if !defined(__x86_64__)
class TestLwSciSyncFenceOperationsTimestampMultipleSemaphores
    : public LwSciSyncInterProcessTest
{
};

TEST_F(TestLwSciSyncFenceOperationsTimestampMultipleSemaphores, Success)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    constexpr uint64_t magicValue = 0x0FF1CE;
    constexpr uint64_t timestampValueBase = 0x4B1D;

    const LwSciSyncAttrValTimestampInfo timestampInfo[] = {
        {
            .format = LwSciSyncTimestampFormat_EmbeddedInPrimitive,
            .scaling =
                {
                    .scalingFactorNumerator = 1U,
                    .scalingFactorDenominator = 1U,
                    .sourceOffset = 0U,
                },
        },
    };
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};
    uint32_t primitiveCount = 16U;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Set Signaler's CPU access to false to allow for multiple primitives
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, false);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_SignalOnly);

        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                          primitiveInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                          primitiveCount);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
                          timestampInfo);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_Success);

            LwSciSyncAttrList reconciledList = nullptr;
            ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &reconciledList),
                      LwSciError_Success);

            bool isReconciled = true;
            ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                          {signalerAttrList.get(), waiterAttrList.get()},
                          reconciledList, &isReconciled),
                      LwSciError_Success);
            ASSERT_EQ(isReconciled, true);

            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(
                reconciledList,
                LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
                timestampInfo));
            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(
                reconciledList, LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                primitiveCount));

            auto attrListAndObjDesc = peer->exportAttrListAndObj(
                syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(peer->sendBuf(attrListAndObjDesc), LwSciError_Success);

            // Get timestamp buffer
            LwSciSyncTimestampBufferInfo timestampBufferInfo{};
            ASSERT_EQ(LwSciSyncObjGetTimestampBufferInfo(syncObj.get(),
                                                         &timestampBufferInfo),
                      LwSciError_Success);

            //
            LwSciBufAttrList bufAttrList = nullptr;
            ASSERT_EQ(LwSciBufObjGetAttrList(timestampBufferInfo.bufObj,
                                             &bufAttrList),
                      LwSciError_Success);
            LwSciBufAttrKeyValuePair pair = {
                .key = LwSciBufGeneralAttrKey_NeedCpuAccess,
                .value = nullptr,
                .len = 0U,
            };
            ASSERT_EQ(LwSciBufAttrListGetAttrs(bufAttrList, &pair, 1U),
                      LwSciError_Success);
            ASSERT_EQ(pair.len, sizeof(bool));
            ASSERT_FALSE(*(bool*)pair.value);

            for (size_t id = 0; id < primitiveCount; id++) {
                auto fence = LwSciSyncPeer::initFence();
                ASSERT_EQ(LwSciSyncFenceUpdateFenceWithTimestamp(
                              syncObj.get(), id, magicValue, 0U, fence.get()),
                          LwSciError_Success);

                auto fenceDesc = peer->exportFence(fence.get(), &error);
                ASSERT_EQ(error, LwSciError_Success);
                ASSERT_EQ(peer->sendExportDesc(fenceDesc), LwSciError_Success);

                // Duplicate the handle with Read/Write permissions.
                // This is necessary since we need to write to the semaphore
                // in order to signal.
                LwSciSyncSemaphoreInfo semaphoreInfo{};
                ASSERT_EQ(LwSciSyncObjGetSemaphoreInfo(syncObj.get(), id,
                                                       &semaphoreInfo),
                          LwSciError_Success);

                LwSciBufRmHandle memHandle{};
                uint64_t offset = 0U;
                uint64_t len = 0U;
                ASSERT_EQ(LwSciBufObjGetMemHandle(semaphoreInfo.bufObj,
                                                  &memHandle, &offset, &len),
                          LwSciError_Success);
                LwRmMemHandle dupHandle{};
                ASSERT_EQ(LwRmMemHandleDuplicate(memHandle.memHandle,
                                                 LWOS_MEM_READ_WRITE,
                                                 &dupHandle),
                          LwError_Success);

                void* ptr = nullptr;
                ASSERT_EQ(LwRmMemMap(dupHandle, semaphoreInfo.offset,
                                     semaphoreInfo.semaphoreSize,
                                     LWOS_MEM_READ_WRITE, &ptr),
                          LwError_Success);

                // Write to the semaphore buffer to signal
                uint64_t* semaphore = (uint64_t*)ptr;
                semaphore[0] = magicValue;
                semaphore[1] = timestampValueBase + id;

                ASSERT_EQ(
                    LwRmMemUnmap(dupHandle, ptr, semaphoreInfo.semaphoreSize),
                    LwError_Success);

                LwRmMemHandleFree(dupHandle);
            }

            // Done
            ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
        }
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, true);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
        SET_INTERNAL_ATTR(waiterAttrList.get(),
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                          primitiveInfo);

        // Export unreconciled waiter list to Peer A
        auto listDescBuf =
            peer->exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        {
            auto attrListAndObjDesc = peer->recvBuf(&error);
            ASSERT_EQ(error, LwSciError_Success);

            auto syncObj = peer->importAttrListAndObj(
                attrListAndObjDesc, {waiterAttrList.get()},
                LwSciSyncAccessPerm_WaitOnly, &error);
            ASSERT_EQ(error, LwSciError_Success);

            LwSciSyncAttrList reconciledList = nullptr;
            ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &reconciledList),
                      LwSciError_Success);

            ASSERT_TRUE(LwSciSyncPeer::verifyInternalAttrNew(
                reconciledList,
                LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
                timestampInfo));

            bool isReconciled = true;
            ASSERT_EQ(LwSciSyncPeer::validateReconciled({waiterAttrList.get()},
                                                        reconciledList,
                                                        &isReconciled),
                      LwSciError_Success);
            ASSERT_EQ(isReconciled, true);

            // Get timestamp buffer
            LwSciSyncTimestampBufferInfo timestampBufferInfo{};
            ASSERT_EQ(LwSciSyncObjGetTimestampBufferInfo(syncObj.get(),
                                                         &timestampBufferInfo),
                      LwSciError_Success);

            const void* timestampBase = nullptr;
            ASSERT_EQ(LwSciBufObjGetConstCpuPtr(timestampBufferInfo.bufObj,
                                                (const void**)&timestampBase),
                      LwSciError_Success);
            uint64_t* timestampAddrBase = (uint64_t*)timestampBase;

            for (size_t id = 0U; id < primitiveCount; id++) {
                auto fenceDesc =
                    peer->recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(
                        &error);
                ASSERT_EQ(error, LwSciError_Success);

                auto fence =
                    peer->importFence(fenceDesc.get(), syncObj.get(), &error);
                ASSERT_EQ(error, LwSciError_Success);

                uint64_t fenceId = 0U;
                uint64_t value = 0U;
                ASSERT_EQ(
                    LwSciSyncFenceExtractFence(fence.get(), &fenceId, &value),
                    LwSciError_Success);
                ASSERT_EQ(fenceId, id);
                ASSERT_EQ(value, magicValue);

                auto waitContext = peer->allocateCpuWaitContext(&error);
                ASSERT_EQ(error, LwSciError_Success);
                ASSERT_EQ(
                    LwSciSyncFenceWait(fence.get(), waitContext.get(), -1),
                    LwSciError_Success);
                // Assert on timestamp
                uint64_t* timestampAddr = &timestampAddrBase[id * 2];
                uint64_t timestamp = 0U;
                ASSERT_EQ(
                        LwSciSyncFenceGetTimestamp(fence.get(), &timestamp),
                        LwSciError_Success);
                ASSERT_EQ(timestamp, timestampValueBase + id);
                ASSERT_EQ(timestampAddr[1], timestamp);
            }

            ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
        }
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
#endif

class LwSciSyncFenceGetTimestampTest : public LwSciSyncInterProcessTest
{
};

TEST_F(LwSciSyncFenceGetTimestampTest, NoCpuAccess)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];
    pid = 0;

    const LwSciSyncAttrValTimestampInfo timestampInfo[] = {
        {
            .format = LwSciSyncTimestampFormat_EmbeddedInPrimitive,
            .scaling =
                {
                    .scalingFactorNumerator = 1U,
                    .scalingFactorDenominator = 1U,
                    .sourceOffset = 0U,
                },
        },
    };
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] = {
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};
    uint32_t primitiveCount = 16U;
    constexpr uint64_t magicValue = 0xADAB;

    if ((pids[0] = fork()) == 0) {
        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        // Set Signaler's CPU access to false to allow for multiple primitives
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, false);
        SET_ATTR(signalerAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitSignal);

        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
                          primitiveInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
                          primitiveCount);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
                          timestampInfo);
        SET_INTERNAL_ATTR(signalerAttrList.get(),
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                          primitiveInfo);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = LwSciSyncPeer::reconcileAndAllocate(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        LwSciSyncAttrList reconciledList = nullptr;
        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &reconciledList),
                  LwSciError_Success);

        bool isReconciled = true;
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {signalerAttrList.get(), waiterAttrList.get()},
                      reconciledList, &isReconciled),
                  LwSciError_Success);
        ASSERT_EQ(isReconciled, true);

        auto attrListAndObjDesc = peer->exportAttrListAndObj(
            syncObj.get(), LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(attrListAndObjDesc), LwSciError_Success);

        for (size_t id = 0U; id < primitiveCount; id++) {
            auto fence = LwSciSyncPeer::initFence();
            uint32_t slotIndex = 0U;

            error = LwSciSyncFenceUpdateFenceWithTimestamp(
                syncObj.get(), id, magicValue, slotIndex, fence.get());
            ASSERT_EQ(error, LwSciError_Success);

            auto fenceDesc = peer->exportFence(fence.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);
            ASSERT_EQ(peer->sendExportDesc(fenceDesc), LwSciError_Success);

            {
                // CPU access is required
                NEGATIVE_TEST();
                uint64_t timestamp = 0U;
                error = LwSciSyncFenceGetTimestamp(fence.get(), &timestamp);
                ASSERT_EQ(error, LwSciError_BadParameter);
            }
        }

        // Done
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) {
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);

        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_NeedCpuAccess, false);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_RequiredPerm,
                 LwSciSyncAccessPerm_WaitOnly);
        SET_ATTR(waiterAttrList.get(), LwSciSyncAttrKey_WaiterRequireTimestamps,
                 true);
        SET_INTERNAL_ATTR(waiterAttrList.get(),
                          LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
                          primitiveInfo);

        // Export unreconciled waiter list to Peer A
        auto listDescBuf =
            peer->exportUnreconciledList({waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(listDescBuf), LwSciError_Success);

        auto attrListAndObjDesc = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObj = peer->importAttrListAndObj(
            attrListAndObjDesc, {waiterAttrList.get()},
            LwSciSyncAccessPerm_WaitOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);

        LwSciSyncAttrList reconciledList = nullptr;
        ASSERT_EQ(LwSciSyncObjGetAttrList(syncObj.get(), &reconciledList),
                  LwSciError_Success);

        bool isReconciled = true;
        ASSERT_EQ(LwSciSyncPeer::validateReconciled(
                      {waiterAttrList.get()}, reconciledList, &isReconciled),
                  LwSciError_Success);
        ASSERT_EQ(isReconciled, true);

        for (size_t id = 0U; id < primitiveCount; id++) {
            auto fenceDesc =
                peer->recvExportDesc<LwSciSyncFenceIpcExportDescriptor>(&error);
            ASSERT_EQ(error, LwSciError_Success);

            auto fence =
                peer->importFence(fenceDesc.get(), syncObj.get(), &error);
            ASSERT_EQ(error, LwSciError_Success);

            uint64_t fenceId = 0U;
            uint64_t value = 0U;
            ASSERT_EQ(LwSciSyncFenceExtractFence(fence.get(), &fenceId, &value),
                      LwSciError_Success);
            ASSERT_EQ(fenceId, id);
            ASSERT_EQ(value, magicValue);

            {
                // CPU access is required
                NEGATIVE_TEST();
                uint64_t timestamp = 0U;
                error = LwSciSyncFenceGetTimestamp(fence.get(), &timestamp);
                EXPECT_EQ(error, LwSciError_BadParameter);
            }
        }

        ASSERT_EQ(peer->signalComplete(), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}
