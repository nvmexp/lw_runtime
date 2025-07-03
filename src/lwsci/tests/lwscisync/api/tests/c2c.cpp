/*
 * Copyright (c) 2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include "lwscisync_test_attribute_list.h"
#include "lwscisync_interprocess_test.h"

class LwSciSyncC2C : public LwSciSyncInterProcessTest
{
};

static LwSciError fillSignaler_copyDoneConsPublic(
    LwSciSyncAttrList signalerAttrList)
{
    LwSciError err = LwSciError_Success;

    bool cpuSignaler = false;
    LwSciSyncAccessPerm signalerAccessPerm = LwSciSyncAccessPerm_SignalOnly;
    LwSciSyncAttrKeyValuePair signalerKeyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuSignaler,
             .len = sizeof(cpuSignaler),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &signalerAccessPerm,
             .len = sizeof(signalerAccessPerm),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(signalerAttrList, signalerKeyValue,
        sizeof(signalerKeyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillSignaler_copyDoneConsInternal(
    LwSciSyncAttrList signalerAttrList,
    enum LwSciSyncInternalAttrValPrimitiveTypeRec primitive)
{
    LwSciError err = LwSciError_Success;

    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { primitive };
    uint32_t signalerPrimitiveCount = 1U;
    LwSciSyncHwEngine engines[] =
        {
            {
                .engNamespace = LwSciSyncHwEngine_TegraNamespaceId,
                // .rmModuleId has to be initialized dynamically later
            }
        };
    LwSciSyncInternalAttrKeyValuePair signalerInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
             .value = (void*)&signalerPrimitiveCount,
             .len = sizeof(signalerPrimitiveCount),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_EngineArray,
             .value = (void*) engines,
             .len = sizeof(engines),
        },
    };

    err = LwSciSyncHwEngCreateIdWithoutInstance(
        LwSciSyncHwEngName_PCIe, &engines[0].rmModuleID);
    if (LwSciError_Success != err) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(signalerAttrList, signalerInternalKeyValue,
        sizeof(signalerInternalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillSignaler_copyDoneCons(
    LwSciSyncAttrList signalerAttrList,
    enum LwSciSyncInternalAttrValPrimitiveTypeRec primitive)
{
    LwSciError err = LwSciError_Success;

    err = fillSignaler_copyDoneConsPublic(signalerAttrList);
    if (LwSciError_Success != err) {
        goto fail;
    }

    err = fillSignaler_copyDoneConsInternal(signalerAttrList, primitive);
    if (LwSciError_Success != err) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillSignaler_copyDoneConsWrongPrimitive(
    LwSciSyncAttrList signalerAttrList)
{
    LwSciError err = LwSciError_Success;

    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore };
    uint32_t signalerPrimitiveCount = 1U;
    LwSciSyncHwEngine engines[] =
        {
            {
                .engNamespace = LwSciSyncHwEngine_TegraNamespaceId,
                // .rmModuleId has to be initialized dynamically later
            }
        };
    LwSciSyncInternalAttrKeyValuePair signalerInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
             .value = (void*)&signalerPrimitiveCount,
             .len = sizeof(signalerPrimitiveCount),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_EngineArray,
             .value = (void*) engines,
             .len = sizeof(engines),
        },
    };

    err = fillSignaler_copyDoneConsPublic(signalerAttrList);
    if (LwSciError_Success != err) {
        goto fail;
    }

    err = LwSciSyncHwEngCreateIdWithoutInstance(
        LwSciSyncHwEngName_PCIe, &engines[0].rmModuleID);
    if (LwSciError_Success != err) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(signalerAttrList, signalerInternalKeyValue,
        sizeof(signalerInternalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneConsPublic(
    LwSciSyncAttrList waiterAttrList)
{
    LwSciError err = LwSciError_Success;

    bool cpuWaiter = true;
    LwSciSyncAccessPerm waiterAccessPerm = LwSciSyncAccessPerm_WaitOnly;

    LwSciSyncAttrKeyValuePair waiterKeyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuWaiter,
             .len = sizeof(cpuWaiter),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &waiterAccessPerm,
             .len = sizeof(waiterAccessPerm),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(waiterAttrList, waiterKeyValue,
        sizeof(waiterKeyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneConsInternal(
    LwSciSyncAttrList waiterAttrList,
    enum LwSciSyncInternalAttrValPrimitiveTypeRec primitive)
{
    LwSciError err = LwSciError_Success;

    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { primitive };
    LwSciSyncInternalAttrKeyValuePair waiterInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
    };

    err = LwSciSyncAttrListSetInternalAttrs(waiterAttrList, waiterInternalKeyValue,
        sizeof(waiterInternalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneCons(
    LwSciSyncAttrList waiterAttrList,
    enum LwSciSyncInternalAttrValPrimitiveTypeRec primitive)
{
    LwSciError err = LwSciError_Success;

    err = fillWaiter_copyDoneConsPublic(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = fillWaiter_copyDoneConsInternal(waiterAttrList, primitive);
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneConsMultipleEngineArray(
    LwSciSyncAttrList waiterAttrList)
{
    LwSciError err = LwSciError_Success;

    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncInternalAttrValPrimitiveType_Syncpoint };
    LwSciSyncHwEngine engines[] =
        {
            {
                .engNamespace = LwSciSyncHwEngine_TegraNamespaceId,
                // .rmModuleId has to be initialized dynamically later
            }
        };
    LwSciSyncInternalAttrKeyValuePair waiterInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
        {    .attrKey = LwSciSyncInternalAttrKey_EngineArray,
             .value = (void*) engines,
             .len = sizeof(engines),
        },
    };

    err = fillWaiter_copyDoneConsPublic(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncHwEngCreateIdWithoutInstance(
        LwSciSyncHwEngName_PCIe, &engines[0].rmModuleID);
    if (LwSciError_Success != err) {
        goto fail;
    }
    err = LwSciSyncAttrListSetInternalAttrs(waiterAttrList, waiterInternalKeyValue,
        sizeof(waiterInternalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneConsWrongPrimitive(
    LwSciSyncAttrList waiterAttrList)
{
    LwSciError err = LwSciError_Success;

    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore };
    LwSciSyncInternalAttrKeyValuePair waiterInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
    };

    err = fillWaiter_copyDoneConsPublic(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListSetInternalAttrs(
        waiterAttrList,
        waiterInternalKeyValue,
        sizeof(waiterInternalKeyValue)/
          sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

/** @jama{TBD} Basic C2C test
 * In this test, the waiter reconciles and gives signaling permissions
 * to the signaler.
 */
TEST_F(LwSciSyncC2C, DISABLED_BasicC2C)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled signaler list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
                {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);

        auto reconciledListDesc = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(
            reconciledListDesc, {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        ASSERT_TRUE(LwSciSyncPeer::verifyAttrNew(
            reconciledList.get(),
            LwSciSyncAttrKey_ActualPerm,
            LwSciSyncAccessPerm_SignalOnly));

        // needed for Desktop
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) { //waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneCons(
            waiterAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Signaler Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        auto reconciledListDesc =
            peer->exportReconciledList(newReconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDesc),
            LwSciError_Success);

        // needed for Desktop
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

TEST_F(LwSciSyncC2C, DISABLED_C2CMultipleEngineArray)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled signaler list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
            {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) { //waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneConsMultipleEngineArray(
            waiterAttrList.get());
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Signaler Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // Only 1 attribute list can specify EngineArray
            NegativeTestPrint();
            auto newReconciledList = LwSciSyncPeer::reconcileLists(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_ReconciliationFailed);
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

TEST_F(LwSciSyncC2C, DISABLED_C2CWaiterTraveled)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Waiter Attribute List
        auto waiterListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto waiterAttrList =
            peer->importUnreconciledList(waiterListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // The waiter can't travel over IPC
            NegativeTestPrint();
            auto newReconciledList = LwSciSyncPeer::reconcileLists(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_ReconciliationFailed);
        }
    } else if ((pids[1] = fork()) == 0) { //waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneCons(
            waiterAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled waiter list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
            {waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);
    } else {
        int status = EXIT_SUCCESS;
        for (auto const& peerPid : pids) {
            TEST_COUT << "Wait for PID " << peerPid << " to exit";
            status |= LwSciSyncInterProcessTest::wait_for_child_fork(peerPid);
        }
        ASSERT_EQ(status, EXIT_SUCCESS);
    }
}

/** @jama{TBD}
 * In this test, multiple attribute lists are provided in a C2C use case. An
 * error message is logged about the number of allowed LwSciSyncAttrLists.
 */
TEST_F(LwSciSyncC2C, DISABLED_C2CMultipleLists)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrListA = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrListA.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        auto signalerAttrListB = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrListB.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled signaler list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
            {signalerAttrListA.get(), signalerAttrListB.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) { //waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneCons(
            waiterAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Signaler Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // Only 2 attribute lists are supported in C2C
            NegativeTestPrint();
            auto newReconciledList = LwSciSyncPeer::reconcileLists(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_ReconciliationFailed);
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

/** @jama{TBD}
 * In this test, a signaler is provided that is missing a required primitive.
 * An error message is logged about the required signaler primitives.
 */
TEST_F(LwSciSyncC2C, DISABLED_C2CSignalerMissingSyncpoint)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneConsWrongPrimitive(
            signalerAttrList.get());
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled signaler list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
            {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) { //waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneCons(
            waiterAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Signaler Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // The signaler list must provide syncpoint primitive
            NegativeTestPrint();
            auto newReconciledList = LwSciSyncPeer::reconcileLists(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_ReconciliationFailed);
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

/** @jama{TBD}
 * In this test, a waiter is provided that is missing a required primitive.
 * An error message is logged about the required waiter primitives.
 */
TEST_F(LwSciSyncC2C, DISABLED_C2CWaiterMissingSyncpoint)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled signaler list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
            {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) { //waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneConsWrongPrimitive(
            waiterAttrList.get());
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Signaler Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        {
            // The signaler list must provide syncpoint primitive
            NegativeTestPrint();
            auto newReconciledList = LwSciSyncPeer::reconcileLists(
                {signalerAttrList.get(), waiterAttrList.get()}, &error);
            ASSERT_EQ(error, LwSciError_ReconciliationFailed);
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

/** @jama{TBD} copyDoneConsObj
 * In this test, we create copyDoneConsObj and signal.
 */
TEST_F(LwSciSyncC2C, DISABLED_copyDoneConsObj)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //producer/signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled signaler list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
                {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);

        auto reconciledListDesc = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(
            reconciledListDesc, {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        error = LwSciSyncAttrListGetAttr(
            reconciledList.get(), LwSciSyncAttrKey_ActualPerm,
            &val, &len);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(len, sizeof(LwSciSyncAccessPerm));
        ASSERT_EQ(*(LwSciSyncAccessPerm*)val, LwSciSyncAccessPerm_SignalOnly);

        auto syncObjDesc =
            peer->recvExportDesc<LwSciSyncObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObjDesc.get(), nullptr);
        auto importedSyncObj =
            peer->importSyncObj(syncObjDesc.get(), reconciledList.get(),
                               LwSciSyncAccessPerm_SignalOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(importedSyncObj.get(), nullptr);

        // needed for Desktop
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) { //consumer/waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneCons(
            waiterAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Signaler Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        auto reconciledListDesc =
            peer->exportReconciledList(newReconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDesc),
            LwSciError_Success);

        auto syncObj = LwSciSyncPeer::allocateSyncObj(
            newReconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObjDesc =
            peer->exportSyncObj(
                syncObj.get(), LwSciSyncAccessPerm_SignalOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(syncObjDesc), LwSciError_Success);

        // needed for Desktop
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

/** @jama{TBD} copyDoneConsObj
 * In this test, we create copyDoneConsObj and signal.
 */
TEST_F(LwSciSyncC2C, DISABLED_copyDoneConsObjSemaphore)
{
    LwSciError error = LwSciError_Success;
    int peerNumber = 2;
    pid_t pids[peerNumber];

    if ((pids[0] = fork()) == 0) { //producer/signaler
        const void* val = NULL;
        size_t len = 0U;

        pid = 1;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_0");

        auto signalerAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillSignaler_copyDoneCons(
            signalerAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        ASSERT_EQ(LwSciError_Success, error);

        // Export unreconciled signaler list to the waiter
        auto unreconciledListDesc = peer->exportUnreconciledList(
                {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(unreconciledListDesc), LwSciError_Success);

        auto reconciledListDesc = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto reconciledList = peer->importReconciledList(
            reconciledListDesc, {signalerAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);

        error = LwSciSyncAttrListGetAttr(
            reconciledList.get(), LwSciSyncAttrKey_ActualPerm,
            &val, &len);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(len, sizeof(LwSciSyncAccessPerm));
        ASSERT_EQ(*(LwSciSyncAccessPerm*)val, LwSciSyncAccessPerm_SignalOnly);

        auto syncObjDesc =
            peer->recvExportDesc<LwSciSyncObjIpcExportDescriptor>(&error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(syncObjDesc.get(), nullptr);
        auto importedSyncObj =
            peer->importSyncObj(syncObjDesc.get(), reconciledList.get(),
                               LwSciSyncAccessPerm_SignalOnly, &error);
        ASSERT_EQ(error, LwSciError_BadParameter);

        // needed for Desktop
        ASSERT_EQ(peer->waitComplete(), LwSciError_Success);
    } else if ((pids[1] = fork()) == 0) { //consumer/waiter
        pid = 2;
        auto peer = std::make_shared<LwSciSyncIpcPeer>();
        peers.push_back(peer);
        peer->SetUp("lwscisync_a_1");

        auto waiterAttrList = peer->createAttrList(&error);
        ASSERT_EQ(error, LwSciError_Success);
        error = fillWaiter_copyDoneCons(
            waiterAttrList.get(),
            LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
        ASSERT_EQ(LwSciError_Success, error);

        // Import Unreconciled Signaler Attribute List
        auto signalerListDescBuf = peer->recvBuf(&error);
        ASSERT_EQ(error, LwSciError_Success);
        auto signalerAttrList =
            peer->importUnreconciledList(signalerListDescBuf, &error);
        ASSERT_EQ(error, LwSciError_Success);
        auto newReconciledList = LwSciSyncPeer::reconcileLists(
            {signalerAttrList.get(), waiterAttrList.get()}, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_NE(newReconciledList.get(), nullptr);

        auto reconciledListDesc =
            peer->exportReconciledList(newReconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendBuf(reconciledListDesc),
            LwSciError_Success);

        auto syncObj = LwSciSyncPeer::allocateSyncObj(
            newReconciledList.get(), &error);
        ASSERT_EQ(error, LwSciError_Success);

        auto syncObjDesc =
            peer->exportSyncObj(syncObj.get(),
                LwSciSyncAccessPerm_SignalOnly, &error);
        ASSERT_EQ(error, LwSciError_Success);
        ASSERT_EQ(peer->sendExportDesc(syncObjDesc), LwSciError_Success);

        // needed for Desktop
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
