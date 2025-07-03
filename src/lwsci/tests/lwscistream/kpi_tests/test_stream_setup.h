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

#ifndef TEST_STREAM_SETUP_H
#define TEST_STREAM_SETUP_H

#include "lwscistream.h"
#include "test.h"

class ProducerCreate : public virtual StreamTestProd
{
public:
    ProducerCreate(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2"){};
    ~ProducerCreate(void) override = default;

    void run(void) override
    {
        CHECK_LWSCIERR(LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));

        KPIStart(&timer);
        LwSciError err = LwSciStreamProducerCreate(pool, &producer);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class PoolCreate : public virtual StreamTestProd
{
public:
    PoolCreate(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};
    ~PoolCreate(void) override = default;

    void run(void) override
    {
        KPIStart(&timer);
        LwSciError err = LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class IpcSrcCreateProd : public virtual StreamTestProd
{
public:
    IpcSrcCreateProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};
    ~IpcSrcCreateProd(void) override = default;

    void run(void) override
    {
        init();

        KPIStart(&timer);
        LwSciError err = LwSciStreamIpcSrcCreate(
                            ipcEndpoint_0,
                            syncModule,
                            bufModule,
                            &ipcsrc);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class IpcSrcCreateCons : public virtual StreamTestCons
{
public:
    IpcSrcCreateCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};
    ~IpcSrcCreateCons(void) override = default;

    void run(void) override
    {
        init();
        CHECK_LWSCIERR(LwSciStreamIpcDstCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcdst));
    }
};

class ConsumerCreate : public virtual StreamTestCons
{
public:
    ConsumerCreate(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};
    ~ConsumerCreate(void) override = default;

    void run(void) override
    {
        CHECK_LWSCIERR(LwSciStreamFifoQueueCreate(&queue));

        KPIStart(&timer);
        LwSciError err = LwSciStreamConsumerCreate(queue, &consumer);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class FifoQueueCreate : public virtual StreamTestCons
{
public:
    FifoQueueCreate(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};
    ~FifoQueueCreate(void) override = default;

    void run(void) override
    {
        KPIStart(&timer);
        LwSciError err = LwSciStreamFifoQueueCreate(&queue);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class IpcDstCreateProd : public virtual StreamTestProd
{
public:
    IpcDstCreateProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};
    ~IpcDstCreateProd(void) override = default;

    void run(void) override
    {
        init();
        CHECK_LWSCIERR(LwSciStreamIpcSrcCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcsrc));
    }
};


class IpcDstCreateCons : public virtual StreamTestCons
{
public:
    IpcDstCreateCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};
    ~IpcDstCreateCons(void) override = default;

    void run(void) override
    {
        init();

        KPIStart(&timer);
        LwSciError err = LwSciStreamIpcDstCreate(
                            ipcEndpoint_0,
                            syncModule,
                            bufModule,
                            &ipcdst);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class ConnectProd2IpcSrcProd : public virtual StreamTestProd
{
public:
    ConnectProd2IpcSrcProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};
    ~ConnectProd2IpcSrcProd(void) override = default;

    void run(void) override
    {
        init();
        CHECK_LWSCIERR(LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));
        CHECK_LWSCIERR(LwSciStreamProducerCreate(pool, &producer));
        CHECK_LWSCIERR(LwSciStreamIpcSrcCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcsrc));

        KPIStart(&timer);
        LwSciError err = LwSciStreamBlockConnect(producer, ipcsrc);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class ConnectProd2IpcSrcCons : public virtual StreamTestCons
{
public:
    ConnectProd2IpcSrcCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};
    ~ConnectProd2IpcSrcCons(void) override = default;

    void run(void) override
    {
        init();
        CHECK_LWSCIERR(LwSciStreamIpcDstCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcdst));
    }
};

class ConnectIpcDst2ConsProd : public virtual StreamTestProd
{
public:
    ConnectIpcDst2ConsProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};
    ~ConnectIpcDst2ConsProd(void) override = default;

    void run(void) override
    {
        init();
        CHECK_LWSCIERR(LwSciStreamIpcSrcCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcsrc));
    }
};

class ConnectIpcDst2ConsCons : public virtual StreamTestCons
{
public:
    ConnectIpcDst2ConsCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};
    ~ConnectIpcDst2ConsCons(void) override = default;

    void run(void) override
    {
        init();
        CHECK_LWSCIERR(LwSciStreamFifoQueueCreate(&queue));
        CHECK_LWSCIERR(LwSciStreamConsumerCreate(queue, &consumer));
        CHECK_LWSCIERR(LwSciStreamIpcDstCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcdst));

        KPIStart(&timer);
        LwSciError err = LwSciStreamBlockConnect(ipcdst, consumer);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

#endif // TEST_STREAM_SETUP_H
