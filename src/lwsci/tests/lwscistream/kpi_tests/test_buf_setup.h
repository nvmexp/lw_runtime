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

#ifndef TEST_BUF_SETUP_H
#define TEST_BUF_SETUP_H

#include "lwscistream.h"
#include "test.h"

constexpr LwSciStreamCookie COOKIE_BASE = 100U;

//
// TODO: Eliminate LWSTRMS52-REQ-1264 or redefine and create new test
//

// TODO: Replace/redefine LWSTRMS52-REQ-1264 and LWSTRMS52-REQ-1267
class ElementsProd : public virtual StreamTestProd
{
public:
    ElementsProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~ElementsProd(void) override = default;

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        // Receive elements
        receiver = pool;
        receiverThread();

        if (event != LwSciStreamEventType_Elements) {
            printf("event is not LwSciStreamEventType_Elements\n");
            return;
        }
    }

protected:
    virtual void setupBuf(void)
    {
        setUpIspBufAttr();
        setUpIspInternalAttr();

        // Producer sends elements
        CHECK_LWSCIERR(LwSciStreamBlockElementAttrSet(
            producer, 0, ispRawAttrList));
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            producer, LwSciStreamSetup_ElementExport, true));

        bufRet = true;
    };

    bool bufRet{ false };
};

class ElementsCons : public virtual StreamTestCons
{
public:
    ElementsCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~ElementsCons(void) override = default;

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_ElementExport, true));
    }

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        senderThread();
    }

protected:
    virtual void setupBuf(void)
    {
        setUpDisplayBufAttr();
        setUpDisplayInternalAttr();

        CHECK_LWSCIERR(LwSciStreamBlockElementAttrSet(
            consumer, 0U, displayRawAttrList));

        bufRet = true;
    };

    bool bufRet{ false };
};

// TODO: Replace/redefine LWSTRMS52-REQ-1264 and LWSTRMS52-REQ-1268
class ReconciledElementsProd : public virtual ElementsProd
{
public:
    ReconciledElementsProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~ReconciledElementsProd(void) override
    {
        if (reconciledBufList != nullptr) {
            LwSciBufAttrListFree(reconciledBufList);
        }
        if (conflictBufList != nullptr) {
            LwSciBufAttrListFree(conflictBufList);
        }
        if (recvdIspRawAttrList != nullptr) {
            LwSciBufAttrListFree(recvdIspRawAttrList);
        }
        if (recvdDisplayRawAttrList != nullptr) {
            LwSciBufAttrListFree(recvdDisplayRawAttrList);
        }
    };

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            pool, LwSciStreamSetup_ElementExport, true));
    };

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        // Send packet element count
        senderThread();
    }

protected:
    void setupBuf(void) override
    {
        ElementsProd::setupBuf();
        if (!bufRet) {
            return;
        }
        bufRet = false;

        // Pool receives elements
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(pool, QUERY_TIMEOUT,
                                                  &event));
        assert(event == LwSciStreamEventType_Elements);

        CHECK_LWSCIERR(LwSciStreamBlockElementAttrGet(
            pool, LwSciStreamBlockType_Producer, 0,
            nullptr, &recvdIspRawAttrList));
        CHECK_LWSCIERR(LwSciStreamBlockElementAttrGet(
            pool, LwSciStreamBlockType_Consumer, 0,
            nullptr, &recvdDisplayRawAttrList));

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            pool, LwSciStreamSetup_ElementImport, true));

        // Reconcile attributes
        const LwSciBufAttrList unreconciledList[2] = {
            recvdIspRawAttrList, recvdDisplayRawAttrList };
        CHECK_LWSCIERR(LwSciBufAttrListReconcile(
            unreconciledList, 2U, &reconciledBufList, &conflictBufList));

        // Pool specifies elements
        CHECK_LWSCIERR(LwSciStreamBlockElementAttrSet(
            pool, 0, reconciledBufList));

        bufRet = true;
    };

    LwSciBufAttrList recvdIspRawAttrList{ nullptr };
    LwSciBufAttrList recvdDisplayRawAttrList{ nullptr };
    LwSciBufAttrList reconciledBufList{ nullptr };
    LwSciBufAttrList conflictBufList{ nullptr };
};

class ReconciledElementsCons : public virtual ElementsCons
{
public:
    ReconciledElementsCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~ReconciledElementsCons(void) override = default;

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        // Receive packet element count
        receiver = consumer;
        receiverThread();

        if (event != LwSciStreamEventType_Elements) {
            printf("event is not LwSciStreamEventType_Elements\n");
            return;
        }
    };

protected:
    void setupBuf(void) override
    {
        ElementsCons::setupBuf();
        if (!bufRet) {
            return;
        }
        bufRet = false;

        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_ElementExport, true));

        bufRet = true;
    };
};

// TODO: Replace/combine requirements
//https://lwpu.jamacloud.com/perspective.req#/items/20053089?projectId=22182
//https://lwpu.jamacloud.com/perspective.req#/items/20053077?projectId=22182
class PacketCreateProd : public virtual ReconciledElementsProd
{
public:
    PacketCreateProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~PacketCreateProd(void) override
    {
        if (bufObj != nullptr) {
            LwSciBufObjFree(bufObj);
        }
    }

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamPoolPacketComplete(pool, packetHandle));
    }

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

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

        CHECK_LWSCIERR(LwSciStreamPoolPacketCreate(
            pool, poolCookie, &packetHandle));

        CHECK_LWSCIERR(LwSciBufObjAlloc(reconciledBufList, &bufObj));

        CHECK_LWSCIERR(LwSciStreamPoolPacketInsertBuffer(
            pool, packetHandle, 0, bufObj));

        bufRet = true;
    };

    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie =
        static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciBufObj bufObj{ nullptr };
};

class PacketCreateCons : public virtual ReconciledElementsCons
{
public:
    PacketCreateCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~PacketCreateCons(void) override = default;

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        // Receive packet create
        receiver = consumer;
        receiverThread();

        if (event != LwSciStreamEventType_PacketCreate) {
            printf("event is not LwSciStreamEventType_PacketCreate\n");
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
};

// TODO: Replace/combine requirements:
//https://lwpu.jamacloud.com/perspective.req#/items/20053116?projectId=22182
//https://lwpu.jamacloud.com/perspective.req#/items/20053128?projectId=22182
class PacketStatusProd : public virtual PacketCreateProd
{
public:
    PacketStatusProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2") {};

    ~PacketStatusProd(void) override = default;

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        // Receive packet acceptance status
        receiver = pool;
        receiverThread();

        if (event != LwSciStreamEventType_PacketStatus) {
            printf("event is not LwSciStreamEventType_PacketStatus\n");
            return;
        }
    }

protected:
    void setupBuf(void) override
    {
        PacketCreateProd::setupBuf();
        if (!bufRet) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamPoolPacketComplete(pool, packetHandle));

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            producer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_PacketCreate) {
            printf("event is not LwSciStreamEventType_PacketCreate\n");
            bufRet = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockPacketNewHandleGet(
            producer, &producerHandle));

        CHECK_LWSCIERR(LwSciStreamBlockPacketStatusSet(
            producer, producerHandle, producerCookie, LwSciError_Success));

        bufRet = true;
    };

    LwSciStreamPacket producerHandle;
    LwSciStreamCookie producerCookie =
        static_cast<LwSciStreamCookie>(COOKIE_BASE);
};

class PacketStatusCons : public virtual PacketCreateCons
{
public:
    PacketStatusCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3") {};

    ~PacketStatusCons(void) override = default;

    void action(void) override
    {
        CHECK_LWSCIERR(LwSciStreamBlockPacketStatusSet(
            consumer, packetHandle, consumerCookie, LwSciError_Success));
    }

    void run(void) override
    {
        init();
        createStreams();
        setupBuf();
        if (!bufRet) {
            return;
        }

        senderThread();
    }

protected:
    void setupBuf(void) override
    {
        PacketCreateCons::setupBuf();
        if (!bufRet) {
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(
            consumer, QUERY_TIMEOUT, &event));
        if (event != LwSciStreamEventType_PacketCreate) {
            printf("event is not LwSciStreamEventType_PacketCreate\n");
            bufRet = false;
            return;
        }

        CHECK_LWSCIERR(LwSciStreamBlockPacketNewHandleGet(
            consumer, &packetHandle));

        bufRet = true;
    };

    LwSciStreamPacket packetHandle;
    LwSciStreamCookie consumerCookie =
        static_cast<LwSciStreamCookie>(COOKIE_BASE);
};

#endif // TEST_BUF_SETUP_H
