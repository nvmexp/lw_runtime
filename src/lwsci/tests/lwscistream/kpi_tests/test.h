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

#ifndef TEST_H
#define TEST_H

#include <thread>
#include <condition_variable>
#include <unistd.h>
#include <cassert>

#include "lwscibuf.h"
#include "lwscibuf_internal.h"
#include "lwscistream.h"

#include "ipc_wrapper.h"
#include "kpitimer.h"
#include "sync_attr.h"
#include "util.h"
#include "constant.h"

class StreamTest
{
public:
    virtual ~StreamTest(void)
    {
        if (syncModule != nullptr) {
            LwSciSyncModuleClose(syncModule);
        }
        if (signalerList != nullptr) {
            LwSciSyncAttrListFree(signalerList);
        }
        if (waiterList != nullptr) {
            LwSciSyncAttrListFree(waiterList);
        }

        if (bufModule != nullptr) {
            LwSciBufModuleClose(bufModule);
        }
        if (ispRawAttrList != nullptr) {
            LwSciBufAttrListFree(ispRawAttrList);
        }
        if (displayRawAttrList != nullptr) {
            LwSciBufAttrListFree(displayRawAttrList);
        }
    };

    virtual void run(void) = 0;

protected:
    void init(void)
    {
        CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));
        CHECK_LWSCIERR(LwSciBufModuleOpen(&bufModule));
    };

    void setupAttrLists(void)
    {
        VolvoSyncAttr attr;

        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &signalerList));
        CHECK_LWSCIERR(LwSciSyncAttrListSetAttrs(
                        signalerList,
                        attr.signalerKeyVal.data(),
                        attr.signalerKeyVal.size()));
        CHECK_LWSCIERR(LwSciSyncAttrListSetInternalAttrs(
                        signalerList,
                        attr.signalerIntKeyVal.data(),
                        attr.signalerIntKeyVal.size()));

        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &waiterList));
        CHECK_LWSCIERR(LwSciSyncAttrListSetAttrs(
                        waiterList,
                        attr.waiterKeyVal.data(),
                        attr.waiterKeyVal.size()));
        CHECK_LWSCIERR(LwSciSyncAttrListSetInternalAttrs(
                        waiterList,
                        attr.waiterIntKeyVal.data(),
                        attr.waiterIntKeyVal.size()));
    };

    void setUpIspBufAttr(void)
    {
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &ispRawAttrList));
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufType rawBufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        uint32_t planeCount = 2U;
        uint64_t topPadding[2] = { 0U };
        uint64_t bottomPadding[2] = { 0U };
        uint64_t leftPadding[2] = { 0U };
        uint64_t rightPadding[2] = { 0U };
        LwSciBufAttrValColorFmt planeColorFmt[2] = { LwSciColor_Y8, LwSciColor_U8_V8 };
        LwSciBufAttrValColorStd planeColorStd[2] = { LwSciColorStd_REC709_ER,
            LwSciColorStd_REC709_ER };
        uint32_t baseAddrAlign[2] = { 256U, 256U };
        uint32_t planeWidths[2] = { 1936U, 968U };
        uint32_t planeHeights[2] = { 1220U, 610U };
        bool vprFlag = false;
        LwSciBufAttrValImageScanType scanType = LwSciBufScan_ProgressiveType;

        LwSciBufAttrKeyValuePair rawBufAttrs[15] = {
            {
                LwSciBufGeneralAttrKey_RequiredPerm,
                &perm,
                sizeof(perm)
            },
            {
                LwSciBufGeneralAttrKey_Types,
                &rawBufType,
                sizeof(rawBufType)
            },
            {
                LwSciBufImageAttrKey_Layout,
                &layout,
                sizeof(layout)
            },
            {
                LwSciBufImageAttrKey_PlaneCount,
                &planeCount,
                sizeof(planeCount)
            },
            {
                LwSciBufImageAttrKey_TopPadding,
                topPadding,
                sizeof(topPadding)
            },
            {
                LwSciBufImageAttrKey_BottomPadding,
                bottomPadding,
                sizeof(bottomPadding)
            },
            {
                LwSciBufImageAttrKey_LeftPadding,
                leftPadding,
                sizeof(leftPadding)
            },
            {
                LwSciBufImageAttrKey_RightPadding,
                rightPadding,
                sizeof(rightPadding)
            },
            {
                LwSciBufImageAttrKey_PlaneColorFormat,
                planeColorFmt,
                sizeof(planeColorFmt)
            },
            {
                LwSciBufImageAttrKey_PlaneColorStd,
                planeColorStd,
                sizeof(planeColorStd)
            },
            {
                LwSciBufImageAttrKey_PlaneBaseAddrAlign,
                baseAddrAlign,
                sizeof(baseAddrAlign)
            },
            {
                LwSciBufImageAttrKey_PlaneWidth,
                planeWidths,
                sizeof(planeWidths)
            },
            {
                LwSciBufImageAttrKey_PlaneHeight,
                planeHeights,
                sizeof(planeHeights)
            },
            {
                LwSciBufImageAttrKey_VprFlag,
                &vprFlag,
                sizeof(vprFlag)
            },
            {
                LwSciBufImageAttrKey_ScanType,
                &scanType,
                sizeof(scanType)
            },
        };

        CHECK_LWSCIERR(LwSciBufAttrListSetAttrs(ispRawAttrList,
                        rawBufAttrs,
                        sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair)));
    };

    void setUpDisplayBufAttr(void)
    {
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &displayRawAttrList));
        LwSciBufAttrValAccessPerm perm = LwSciBufAccessPerm_ReadWrite;
        LwSciBufType rawBufType = LwSciBufType_Image;
        LwSciBufAttrValImageLayoutType layout = LwSciBufImage_BlockLinearType;
        uint32_t planeCount = 2U;
        uint64_t topPadding[2] = { 0U };
        uint64_t bottomPadding[2] = { 0U };
        uint64_t leftPadding[2] = { 0U };
        uint64_t rightPadding[2] = { 0U };
        LwSciBufAttrValColorFmt planeColorFmt[2] = { LwSciColor_Y8, LwSciColor_U8_V8 };
        LwSciBufAttrValColorStd planeColorStd[2] = { LwSciColorStd_REC709_ER,
            LwSciColorStd_REC709_ER };
        uint32_t baseAddrAlign[2] = { 256U, 256U };
        uint32_t planeWidths[2] = { 1936U, 968U };
        uint32_t planeHeights[2] = { 1220U, 610U };
        bool vprFlag = false;
        LwSciBufAttrValImageScanType scanType= LwSciBufScan_ProgressiveType;

        LwSciBufAttrKeyValuePair rawBufAttrs[15] = {
            {
                LwSciBufGeneralAttrKey_RequiredPerm,
                &perm,
                sizeof(perm)
            },
            {
                LwSciBufGeneralAttrKey_Types,
                &rawBufType,
                sizeof(rawBufType)
            },
            {
                LwSciBufImageAttrKey_Layout,
                &layout,
                sizeof(layout)
            },
            {
                LwSciBufImageAttrKey_PlaneCount,
                &planeCount,
                sizeof(planeCount)
            },
            {
                LwSciBufImageAttrKey_TopPadding,
                topPadding,
                sizeof(topPadding)
            },
            {
                LwSciBufImageAttrKey_BottomPadding,
                bottomPadding,
                sizeof(bottomPadding)
            },
            {
                LwSciBufImageAttrKey_LeftPadding,
                leftPadding,
                sizeof(leftPadding)
            },
            {
                LwSciBufImageAttrKey_RightPadding,
                rightPadding,
                sizeof(rightPadding)
            },
            {
                LwSciBufImageAttrKey_PlaneColorFormat,
                planeColorFmt,
                sizeof(planeColorFmt)
            },
            {
                LwSciBufImageAttrKey_PlaneColorStd,
                planeColorStd,
                sizeof(planeColorStd)
            },
            {
                LwSciBufImageAttrKey_PlaneBaseAddrAlign,
                baseAddrAlign,
                sizeof(baseAddrAlign)
            },
            {
                LwSciBufImageAttrKey_PlaneWidth,
                planeWidths,
                sizeof(planeWidths)
            },
            {
                LwSciBufImageAttrKey_PlaneHeight,
                planeHeights,
                sizeof(planeHeights)
            },
            {
                LwSciBufImageAttrKey_VprFlag,
                &vprFlag,
                sizeof(vprFlag)
            },
            {
                LwSciBufImageAttrKey_ScanType,
                &scanType,
                sizeof(scanType)
            },
        };

        CHECK_LWSCIERR(LwSciBufAttrListSetAttrs(displayRawAttrList,
                        rawBufAttrs,
                        sizeof(rawBufAttrs)/sizeof(LwSciBufAttrKeyValuePair)));
    };

    void setUpIspInternalAttr(void)
    {
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;

        LwSciBufHwEngine engine1;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Vic,
                                                 &engine1.rmModuleID));
        engine1.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine2;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Isp,
                                                 &engine2.rmModuleID));
        engine2.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine3;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Csi,
                                                 &engine3.rmModuleID));
        engine3.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine4;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                                 &engine4.rmModuleID));
        engine4.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine5;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_DLA,
                                                 &engine5.rmModuleID));
        engine5.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine6;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_MSENC,
                                                 &engine6.rmModuleID));
        engine6.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engine7;
        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_MSENC,
                                                 &engine7.rmModuleID));
        engine7.engNamespace = LwSciBufHwEngine_TegraNamespaceId;

        LwSciBufHwEngine engineArray[] = { engine1, engine2, engine3, engine4,
            engine5, engine6, engine7 };

        LwSciBufInternalAttrKeyValuePair bufIntAttrs[2] = {
            {
                LwSciBufInternalGeneralAttrKey_EngineArray,
                engineArray,
                sizeof(engineArray),
            },
            {
                LwSciBufInternalGeneralAttrKey_MemDomainArray,
                &memDomain,
                sizeof(memDomain),
            },
        };

        CHECK_LWSCIERR(LwSciBufAttrListSetInternalAttrs(ispRawAttrList, bufIntAttrs,
            sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair)));

    };

    void setUpDisplayInternalAttr(void)
    {
        LwSciBufMemDomain memDomain = LwSciBufMemDomain_Sysmem;
        LwSciBufHwEngine engine;

        CHECK_LWSCIERR(LwSciBufHwEngCreateIdWithoutInstance(LwSciBufHwEngName_Display,
                                                 &engine.rmModuleID));
        engine.engNamespace = LwSciBufHwEngine_TegraNamespaceId;


        LwSciBufInternalAttrKeyValuePair bufIntAttrs[2] = {
            {
                LwSciBufInternalGeneralAttrKey_EngineArray,
                &engine,
                sizeof(engine),
            },
            {
                LwSciBufInternalGeneralAttrKey_MemDomainArray,
                &memDomain,
                sizeof(memDomain),
            },
        };

        CHECK_LWSCIERR(LwSciBufAttrListSetInternalAttrs(displayRawAttrList, bufIntAttrs,
            sizeof(bufIntAttrs)/sizeof(LwSciBufInternalAttrKeyValuePair)));
    };

    StreamTest(void) = default;

    LwSciSyncModule syncModule{ nullptr };
    LwSciSyncAttrList signalerList{ nullptr };
    LwSciSyncAttrList waiterList{ nullptr };
    LwSciBufAttrList ispRawAttrList{ nullptr };
    LwSciBufAttrList displayRawAttrList{ nullptr };

    LwSciBufModule bufModule{ nullptr };

    KPItimer timer;
};

class StreamProcTest : public virtual StreamTest
{
public:
    ~StreamProcTest(void) override
    {
        if (ipcWrapper != nullptr) {
            ipcWrapper.reset();
        }
    };

    virtual void action(void)
    {
        printf("No action\n");
    };

protected:
    explicit StreamProcTest(const char* streamEP, const char* msgEP)
    {
        // Ipc channel used to coordinate stream operations and
        // exchange timing info
        ipcWrapper = IpcWrapper::open(msgEP);

        // Ipc channel used by lwscistream
        CHECK_LWSCIERR(LwSciIpcOpenEndpoint(streamEP, &ipcEndpoint_0));
        LwSciIpcResetEndpoint(ipcEndpoint_0);
    };

    void senderThread(void)
    {
        // Wait for an IPC message from other side that
        // it is ready to receive an event.
        int recv_handshake = 0;
        CHECK_LWSCIERR(ipcWrapper->recvFill(
                                    &recv_handshake,
                                    sizeof(recv_handshake),
                                    QUERY_TIMEOUT));
        assert(recv_handshake == handshake);

        // Record the send time.
        KPIStart(&timer);

        // Perform LwSciStream action.
        action();

        // Sleep 1s to ensure nothing this thread does interferes with
        // the transmission.
        sleep(1);

        // Send IPC message to the other side containing the time at which
        // the action was performed.
        CHECK_LWSCIERR(ipcWrapper->send(
                                    &timer.s_Start,
                                    sizeof(timer.s_Start),
                                    QUERY_TIMEOUT));
    };

    void receiverThread(void)
    {
        dispatchThread =
            std::thread(&StreamProcTest::dispatchThreadFunc, this);

        // Wait for signal from dispatch thread that
        // it is ready to receive an event.
        std::unique_lock<std::mutex> lck(mtx);
        while (!ready) {
            cond.wait(lck);
        }
        lck.unlock();

        // Sleep for 1s to ensure the dispatch thread is actually ready.
        sleep(1);

        // Send IPC message to other side indicating readiness.
        uint32_t send_handshake = handshake;
        CHECK_LWSCIERR(ipcWrapper->send(
                                    &send_handshake,
                                    sizeof(send_handshake),
                                    QUERY_TIMEOUT));

        // Wait for signal from dispatch thread that the event was received.
        lck.lock();
        while (!timer.isSet) {
            cond.wait(lck);
        }
        lck.unlock();

        // Receive IPC message from other side containing the time at which
        // the action was performed.
        CHECK_LWSCIERR(ipcWrapper->recvFill(
                                    &timer.s_Start,
                                    sizeof(timer.s_Start),
                                    QUERY_TIMEOUT));

        // Callwlate the exelwtion time.
        KPIDiffTime(&timer);

        if (dispatchThread.joinable()) {
            dispatchThread.join();
        }
    }

    // Spawned by receiver thread
    void dispatchThreadFunc(void)
    {
        // Signal receiver thread that it is starting event query.
        std::unique_lock<std::mutex> lck(mtx);
        ready = true;
        lck.unlock();
        cond.notify_one();

        // Call EventQuery with infinite timeout.
        LwSciError err = LwSciStreamBlockEventQuery(
                            receiver,
                            QUERY_TIMEOUT,
                            &event);
        // When the call returns, record the current time.
        KPIEnd(&timer, false);

        CHECK_LWSCIERR(err);

        // Signal receiver thread that the event is received.
        cond.notify_one();
    };


    LwSciIpcEndpoint ipcEndpoint_0{ 0U };
    std::unique_ptr<IpcWrapper> ipcWrapper{ nullptr };

    LwSciStreamEventType event;
    LwSciStreamBlock receiver{ 0U };

    std::thread dispatchThread;

    bool ready{ false };
    std::condition_variable cond;
    std::mutex mtx;

    constexpr static uint32_t handshake{ 12345U };
};

class StreamTestProd: public virtual StreamProcTest
{
public:
    StreamTestProd(void) :
        StreamProcTest("lwscistream_0", "lwscistream_2")
    {};

    ~StreamTestProd(void) override
    {
        if (producer != 0U) {
            LwSciStreamBlockDelete(producer);
        }
        if (pool != 0U) {
            LwSciStreamBlockDelete(pool);
        }
        if (ipcsrc != 0U) {
            LwSciStreamBlockDelete(ipcsrc);
        }
    };

protected:
    void createStreams(void)
    {
        CHECK_LWSCIERR(LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));
        CHECK_LWSCIERR(LwSciStreamProducerCreate(pool, &producer));
        CHECK_LWSCIERR(LwSciStreamIpcSrcCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcsrc));

        CHECK_LWSCIERR(LwSciStreamBlockConnect(producer, ipcsrc));

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(producer, QUERY_TIMEOUT, &event));
        assert(event == LwSciStreamEventType_Connected);

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(pool, QUERY_TIMEOUT, &event));
        assert(event == LwSciStreamEventType_Connected);

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(ipcsrc, QUERY_TIMEOUT, &event));
        assert(event == LwSciStreamEventType_Connected);
    };

    LwSciStreamBlock producer{ 0U };
    LwSciStreamBlock pool{ 0U };
    LwSciStreamBlock ipcsrc{ 0U };
};

class StreamTestCons: public virtual StreamProcTest
{
public:
    StreamTestCons(void) :
        StreamProcTest("lwscistream_1", "lwscistream_3")
    {};

    ~StreamTestCons(void) override
    {
        if (consumer != 0U) {
            LwSciStreamBlockDelete(consumer);
        }
        if (queue != 0U) {
            LwSciStreamBlockDelete(queue);
        }
        if (ipcdst != 0U) {
            LwSciStreamBlockDelete(ipcdst);
        }
    };

protected:
    void createStreams(void)
    {
        CHECK_LWSCIERR(LwSciStreamFifoQueueCreate(&queue));
        CHECK_LWSCIERR(LwSciStreamConsumerCreate(queue, &consumer));
        CHECK_LWSCIERR(LwSciStreamIpcDstCreate(
                        ipcEndpoint_0,
                        syncModule,
                        bufModule,
                        &ipcdst));

        CHECK_LWSCIERR(LwSciStreamBlockConnect(ipcdst, consumer));

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(queue, QUERY_TIMEOUT, &event));
        assert(event == LwSciStreamEventType_Connected);

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(consumer, QUERY_TIMEOUT, &event));
        assert(event == LwSciStreamEventType_Connected);

        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(ipcdst, QUERY_TIMEOUT, &event));
        assert(event == LwSciStreamEventType_Connected);
    };

    void consSendPacketAttr()
    {
        CHECK_LWSCIERR(LwSciStreamBlockElementAttrSet(
            consumer, 0, displayRawAttrList));
        CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
            consumer, LwSciStreamSetup_ElementExport, true));

    };

    LwSciStreamBlock queue{ 0U };
    LwSciStreamBlock consumer{ 0U };
    LwSciStreamBlock ipcdst{ 0U };
};

#endif // TEST_H
