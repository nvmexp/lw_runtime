//! \file
//! \brief LwSciStream kpi perf test.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef TEST_SYNC_SETUP_H
#define TEST_SYNC_SETUP_H

#include "lwscistream.h"
#include "test.h"
#include "test_buf_setup.h"

// Sync Requirement
class SyncAttrProd : virtual public ReconciledElementsProd
{
public:
    SyncAttrProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~SyncAttrProd(void) override = default;

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_WaiterAttrExport, true));
    };

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }
        setupSync();
        if (!syncRet) {
            return;
        }

        // Send sync event
        senderThread();
    }

protected:
    void setupBuf(void) override
    {
        ReconciledElementsProd::setupBuf();
        if (!bufRet) {
            return;
        }
        bufRet = false;

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            pool, LwSciStreamSetup_ElementExport, true));

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            producer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_Elements) {
            printf("event is not LwSciStreamEventType_Elements\n");
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_ElementImport, true));

        bufRet = true;
    };

    virtual void setupSync(void)
    {
        setupAttrLists();

        CHECK_LWSCIERR(LwSciStreamBlockElementWaiterAttrSet(
            producer, 0U, waiterList));

        syncRet = true;
    };

    bool syncRet{ false };
};

class SyncAttrCons : virtual public ReconciledElementsCons
{
public:
    SyncAttrCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~SyncAttrCons(void) override
    {
        if (recvSyncList != nullptr) {
            LwSciSyncAttrListFree(recvSyncList);
        }
    };

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        // Receive sync requirement
        receiver = consumer;
        receiverThread();

        if (event != LwSciStreamEventType_WaiterAttr) {
            printf("event is not LwSciStreamEventType_WaiterAttr\n");
            return;
        }
    }

protected:
    void setupBuf(void) override
    {
        ReconciledElementsCons::setupBuf();
        if (!bufRet) {
            return;
        }
        bufRet = false;

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            consumer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_Elements) {
            printf("event is not LwSciStreamEventType_Elements\n");
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_ElementImport, true));

        bufRet = true;
    };

    LwSciSyncAttrList recvSyncList{ nullptr };
};

// Sync Obj
class SyncObjProd : public virtual SyncAttrProd
{
public:
    SyncObjProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~SyncObjProd(void) override
    {
        if (recvSyncList != nullptr) {
            LwSciSyncAttrListFree(recvSyncList);
        }
        if (reconciledSyncList != nullptr) {
            LwSciSyncAttrListFree(reconciledSyncList);
        }
        if (conflictSyncList != nullptr) {
            LwSciSyncAttrListFree(conflictSyncList);
        }
        if (syncObj != nullptr) {
            LwSciSyncObjFree(syncObj);
        }
    };

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_SignalObjExport, true));
    };

protected:
    void setupSync(void) override
    {
        SyncAttrProd::setupSync();
        if (!syncRet) {
            return;
        }
        syncRet = false;

        // Finish exporting waiter info
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_WaiterAttrExport, true));

        // Receive waiter info event
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
                        producer,
                        QUERY_TIMEOUT,
                        &event));
        if (event != LwSciStreamEventType_WaiterAttr) {
            printf("LwSciStreamEventType_WaiterAttr event expected\n");
            return;
        }

        // Query attribute list
        CHECK_LWSCIERR(LwSciStreamBlockElementWaiterAttrGet(
             producer, 0U, &recvSyncList));

        // Finish importing waiter info
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_WaiterAttrImport, true));

        // Reconcile sync attr list and allocate sync obj
        LwSciSyncAttrList unreconciledList[2] = { signalerList, recvSyncList };
        CHECK_LWSCIERR(LwSciSyncAttrListReconcile(
                        unreconciledList,
                        2U,
                        &reconciledSyncList,
                        &conflictSyncList));

        // Allocate sync object
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledSyncList, &syncObj));

        // Provide sync object
        CHECK_LWSCIERR(LwSciStreamBlockElementSignalObjSet(producer, 0,
                                                           syncObj));

        syncRet = true;
    };

    LwSciSyncAttrList recvSyncList{ nullptr };
    LwSciSyncAttrList reconciledSyncList{ nullptr };
    LwSciSyncAttrList conflictSyncList{ nullptr };
    LwSciSyncObj syncObj{ nullptr };
};

class SyncObjCons : public virtual SyncAttrCons
{
public:
    SyncObjCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }
        setupSync();
        if (!syncRet) {
            return;
        }

        // Receive sync requirement from other end
        receiver = consumer;
        receiverThread();

        if (event != LwSciStreamEventType_SignalObj) {
            printf("LwSciStreamEventType_SignalObj event expected\n");
            return;
        }
    }

protected:
    virtual void setupSync(void)
    {
        setupAttrLists();

        // Set waiter attributes
        CHECK_LWSCIERR(LwSciStreamBlockElementWaiterAttrSet(
            consumer, 0U, waiterList));

        // Send waiter info
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_WaiterAttrExport, true));

        // Receive waiter info
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
                        consumer,
                        QUERY_TIMEOUT,
                        &event));
        if (event != LwSciStreamEventType_WaiterAttr) {
            printf("LwSciStreamEventType_WaiterAttr event expected\n");
            return;
        }

        // Query attribute list
        CHECK_LWSCIERR(LwSciStreamBlockElementWaiterAttrGet(
             consumer, 0U, &recvSyncList));

        // Finish importing waiter info
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_WaiterAttrImport, true));

        syncRet = true;
    };

    bool syncRet{ false };
};

#endif // TEST_SYNC_SETUP_H
