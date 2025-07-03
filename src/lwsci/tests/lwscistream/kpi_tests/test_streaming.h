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

#ifndef TEST_STREAMING_H
#define TEST_STREAMING_H

#include "lwscistream.h"
#include "test.h"
#include "test_sync_setup.h"
#include "test_buf_setup.h"

#define LW_WAIT_INFINITE 0xFFFFFFFF

class ProducerGetProd :
    public virtual PacketStatusProd,
    public virtual SyncObjProd
{
public:
    ProducerGetProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~ProducerGetProd(void) override
    {
        if (waitContext != nullptr) {
            LwSciSyncCpuWaitContextFree(waitContext);
        }
    };

    void action(void) override {};

    void run(void) override
    {
        setup();
        if (!ret) {
            return;
        }

        LwSciStreamCookie cookie;

        KPIStart(&timer);
        LwSciError err = LwSciStreamProducerPacketGet(
                            producer,
                            &cookie);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }

protected:
    virtual void setup(void)
    {
        init();
        createStreams();

        setupBuf();
        if (!bufRet) {
            ret = false;
            return;
        }

        setupSync();
        if (!syncRet) {
            ret = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            producer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_SetupComplete) {
            printf("event is not LwSciStreamEventType_SetupComplete\n");
            ret = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            producer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_PacketReady) {
            printf("event is not LwSciStreamEventType_PacketReady\n");
            ret = false;
            return;
        }

        ret = true;
    };

    void setupBuf(void) override
    {
        PacketStatusProd::setupBuf();
        if (!bufRet) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            pool, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_PacketStatus) {
            printf("event is not LwSciStreamEventType_PacketStatus\n");
            bufRet = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            pool, LwSciStreamSetup_PacketExport, true));
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            pool, LwSciStreamSetup_PacketImport, true));


        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            producer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_PacketsComplete) {
            printf("event is not LwSciStreamEventType_PacketsComplete\n");
            bufRet = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_PacketImport, true));

        bufRet = true;
    };

    void setupSync(void) override
    {
        // TODO: switch to umd sync obj
        CHECK_LWSCIERR(LwSciSyncCpuWaitContextAlloc(syncModule, &waitContext));

        SyncObjProd::setupSync();
        if (!syncRet) {
            return;
        }
        syncRet = false;

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_SignalObjExport, true));

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
                        producer,
                        QUERY_TIMEOUT,
                        &event));
        if (event != LwSciStreamEventType_SignalObj) {
            printf("LwSciStreamEventType_SignalObj event expected\n");
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_SignalObjImport, true));

        syncRet = true;
    };

    bool ret{ false };
    LwSciSyncFence fence;

    // TODO: switch to umd sync obj
    LwSciSyncCpuWaitContext waitContext{ nullptr };
};

class ProducerGetCons :
    public virtual PacketStatusCons,
    public virtual SyncObjCons
{
public:
    ProducerGetCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~ProducerGetCons(void) override
    {
        if (reconciledSyncList != nullptr) {
            LwSciSyncAttrListFree(reconciledSyncList);
        }
        if (conflictSyncList != nullptr) {
            LwSciSyncAttrListFree(conflictSyncList);
        }
        if (syncObj != nullptr) {
            LwSciSyncObjFree(syncObj);
        }
        if (waitContext != nullptr) {
            LwSciSyncCpuWaitContextFree(waitContext);
        }
    };

    void action(void) override {};

    void run(void) override
    {
        setup();
    };

protected:
    virtual void setup(void)
    {
        init();
        createStreams();

        setupBuf();
        if (!bufRet) {
            ret = false;
            return;
        }

        setupSync();
        if (!syncRet) {
            ret = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            consumer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_SetupComplete) {
            printf("event is not LwSciStreamEventType_SetupComplete\n");
            ret = false;
            return;
        }

        ret = true;
    };

    void setupBuf(void) override
    {
        PacketStatusCons::setupBuf();
        if (!bufRet) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockPacketStatusSet(
            consumer, packetHandle, consumerCookie, LwSciError_Success));

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            consumer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_PacketsComplete) {
            printf("event is not LwSciStreamEventType_PacketsComplete\n");
            bufRet = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_PacketImport, true));

        bufRet = true;
    };

    void setupSync(void) override
    {
        // TODO: switch to umd sync obj
        CHECK_LWSCIERR(LwSciSyncCpuWaitContextAlloc(syncModule, &waitContext));

        SyncObjCons::setupSync();
        if (!syncRet) {
            return;
        }
        syncRet = false;

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
        CHECK_LWSCIERR(LwSciStreamBlockElementSignalObjSet(consumer, 0,
                                                           syncObj));

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_SignalObjExport, true));

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
                        consumer,
                        QUERY_TIMEOUT,
                        &event));
        if (event != LwSciStreamEventType_SignalObj) {
            printf("LwSciStreamEventType_SignalObj event expected\n");
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_SignalObjImport, true));

        syncRet = true;
    };

    bool ret{ false };
    LwSciSyncAttrList reconciledSyncList{ nullptr };
    LwSciSyncAttrList conflictSyncList{ nullptr };
    LwSciSyncObj syncObj{ nullptr };

    // TODO: switch to umd sync obj
    LwSciSyncCpuWaitContext waitContext{ nullptr };
};

class ProducerPresentProd : public virtual ProducerGetProd
{
public:
    ProducerPresentProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~ProducerPresentProd(void) override = default;

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamProducerPacketPresent(
                        producer,
                        packetHandle));
    };

    void run(void) override
    {
        setup();
        if (!ret) {
            return;
        }

        // Producer present a packet
        senderThread();

        CHECK_LWSCIERR(LwSciSyncObjSignal(syncObj));
        LwSciSyncFenceClear(&fence);
    }

protected:
    void setup(void) override
    {
        ProducerGetProd::setup();
        if (!ret) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamProducerPacketGet(
                        producer,
                        &producerCookie));

        CHECK_LWSCIERR(LwSciStreamBlockPacketFenceGet(
                       producer,
                       packetHandle,
                       0U, 0U, &fence));

        CHECK_LWSCIERR(LwSciSyncFenceWait(
                        &fence,
                        waitContext,
                        LW_WAIT_INFINITE));
        LwSciSyncFenceClear(&fence);

        CHECK_LWSCIERR(LwSciSyncObjGenerateFence(syncObj, &fence));
        CHECK_LWSCIERR(LwSciStreamBlockPacketFenceSet(producer,
                                                      packetHandle,
                                                      0U, &fence));

        ret = true;
    };
};

class ProducerPresentCons :
    public virtual ProducerGetCons
{
public:
    ProducerPresentCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~ProducerPresentCons(void) override = default;

    void action(void) override {};

    void run(void) override
    {
        ProducerGetCons::setup();
        if (!ret) {
            return;
        }

        // Receive a packet from producer
        receiver = consumer;
        receiverThread();

        if (event != LwSciStreamEventType_PacketReady) {
            printf("event is not LwSciStreamEventType_PacketReady\n");
            return;
        }
    };
};

class ConsumerAcquireProd : public virtual ProducerPresentProd
{
public:
    ConsumerAcquireProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~ConsumerAcquireProd(void) override = default;

    void action(void) override {};

    void run(void) override
    {
        setup();
        if (!ret) {
            return;
        }
    }

protected:
    void setup(void) override
    {
        ProducerPresentProd::setup();
        if (!ret) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamProducerPacketPresent(
                        producer,
                        packetHandle));

        CHECK_LWSCIERR(LwSciSyncObjSignal(syncObj));
        LwSciSyncFenceClear(&fence);

        ret = true;
    };
};

class ConsumerAcquireCons : public virtual ProducerPresentCons
{
public:
    ConsumerAcquireCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~ConsumerAcquireCons(void) override = default;

    void action(void) override {};

    void run(void) override
    {
        setup();
        if (!ret) {
            return;
        }

        KPIStart(&timer);
        LwSciError err = LwSciStreamConsumerPacketAcquire(
                            consumer,
                            &consumerCookie);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };

protected:
    void setup(void) override
    {
        ProducerGetCons::setup();
        if (!ret) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
                        consumer,
                        QUERY_TIMEOUT,
                        &event));
        if (event != LwSciStreamEventType_PacketReady) {
            printf("event is not LwSciStreamEventType_PacketReady\n");
            ret = false;
            return;
        }

        ret = true;
    };

    LwSciSyncFence fence;
};

class ConsumerReleaseProd : public virtual ConsumerAcquireProd
{
public:
    ConsumerReleaseProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~ConsumerReleaseProd(void) override = default;

    void action(void) override {};

    void run(void) override
    {
        ConsumerAcquireProd::setup();
        if (!ret) {
            return;
        }

        // Receive a packet from producer
        receiver = producer;
        receiverThread();

        if (event != LwSciStreamEventType_PacketReady) {
            printf("event is not LwSciStreamEventType_PacketReady\n");
            return;
        }
    };
};

class ConsumerReleaseCons : public virtual ConsumerAcquireCons
{
public:
    ConsumerReleaseCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~ConsumerReleaseCons(void) override = default;

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamConsumerPacketRelease(
                        consumer,
                        packetHandle));
    };

    void run(void) override
    {
        setup();
        if (!ret) {
            return;
        }

        // Consumer releases a packet
        senderThread();

        CHECK_LWSCIERR(LwSciSyncObjSignal(syncObj));
        LwSciSyncFenceClear(&fence);
    };

protected:
    void setup(void) override
    {
        ConsumerAcquireCons::setup();
        if (!ret) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamConsumerPacketAcquire(
                        consumer,
                        &consumerCookie));

        CHECK_LWSCIERR(LwSciStreamBlockPacketFenceGet(
                       consumer,
                       packetHandle,
                       0U, 0U, &fence));

        CHECK_LWSCIERR(LwSciSyncFenceWait(
                        &fence,
                        waitContext,
                        LW_WAIT_INFINITE));
        LwSciSyncFenceClear(&fence);

        CHECK_LWSCIERR(LwSciSyncObjGenerateFence(syncObj, &fence));
        CHECK_LWSCIERR(LwSciStreamBlockPacketFenceSet(consumer,
                                                      packetHandle,
                                                      0U, &fence));

        ret = true;
    };
};

#endif // TEST_STREAMING_H
