//! \file
//! \brief LwSciStream test class Consumer client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "perfconsumer.h"
#include <mutex>

extern TestArg testArg;

std::mutex          presentListMutex;
std::mutex          releaseListMutex;
extern std::vector<double> presentIntervals;
extern std::vector<double> presentIntervals2;
extern std::vector<double> releaseIntervals;

#ifdef __QNX__
uint64_t PerfTimer::cyclesPerSec = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
#endif

PerfConsumer::PerfConsumer(LwSciBufModule buf,
                           LwSciSyncModule sync):
    PerfClient(buf, sync)
{
    if (testArg.isC2c) {
        allocatedPackets.resize(NUM_PACKETS);
    }
    for (uint32_t i{ 0U }; i < allocatedPackets.size(); i++) {
        allocatedPackets[i].buffers.fill(nullptr);
    }
}

PerfConsumer::~PerfConsumer(void)
{
    for (uint32_t i{ 0U }; i < allocatedPackets.size(); i++) {
        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            if (allocatedPackets[i].buffers[j] != nullptr) {
                LwSciBufObjFree(allocatedPackets[i].buffers[j]);
            }
        }
    }

    // Delete blocks
    if (ipcDst != 0U) {
        LwSciStreamBlockDelete(ipcDst);
        ipcDst = 0U;
    }

    if (c2cPool != 0U) {
        LwSciStreamBlockDelete(c2cPool);
        c2cPool = 0U;
    }
    if (queue != 0U) {
        LwSciStreamBlockDelete(queue);
        queue = 0U;
    }

    if (consumer != 0U) {
        LwSciStreamBlockDelete(consumer);
        consumer = 0U;
    }
}

void PerfConsumer::createStream(
    LwSciIpcEndpoint const ipcEndpoint,
    LwSciStreamBlock const upstreamBlock)
{
    // Create blocks
    CHECK_LWSCIERR(LwSciStreamFifoQueueCreate(&queue));
    CHECK_LWSCIERR(LwSciStreamConsumerCreate(queue, &consumer));

    if (testArg.testType == CrossProcCons) {
        if (testArg.isC2c) {
            CHECK_LWSCIERR(LwSciStreamStaticPoolCreate(NUM_PACKETS, &c2cPool));
        }

        CHECK_LWSCIERR(LwSciStreamIpcDstCreate2(ipcEndpoint,
                                                syncModule,
                                                bufModule,
                                                c2cPool,
                                                &ipcDst));
    }

    // Connect blocks
    if (testArg.testType == CrossProcCons) {
        CHECK_LWSCIERR(LwSciStreamBlockConnect(ipcDst, consumer));
    } else {
        CHECK_LWSCIERR(LwSciStreamBlockConnect(upstreamBlock, consumer));
    }

    endpointHandle = consumer;
}

void PerfConsumer::recvConnectEvent(void)
{
    LwSciStreamEventType event;

    if (testArg.testType == CrossProcCons) {
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(ipcDst,
                                                  QUERY_TIMEOUT,
                                                  &event));
        if (event != LwSciStreamEventType_Connected) {
            return;
        }
    }

    if (testArg.isC2c) {
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(c2cPool,
                                                  QUERY_TIMEOUT,
                                                  &event));
        if (event != LwSciStreamEventType_Connected) {
            return;
        }
    }

    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(queue,
                                              QUERY_TIMEOUT,
                                              &event));
    if (event != LwSciStreamEventType_Connected) {
        return;
    }

    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(consumer,
                                              QUERY_TIMEOUT,
                                              &event));
    if (event != LwSciStreamEventType_Connected) {
        return;
    }
}

void PerfConsumer::run(void)
{
    // Setup phase
    recvConnectEvent();
    sendEndpointElements();
    if (testArg.isC2c) {
        createPacket();
    }

    bool syncDone{ false };
    bool bufDone{ false };
    uint32_t numPackets{ 0U };
    while (!syncDone || !bufDone) {
        LwSciStreamEventType event;
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(consumer,
                                                  QUERY_TIMEOUT,
                                                  &event));
        switch (event) {
        case LwSciStreamEventType_Elements:
            recvAllocatedElements();
            setupSync();
            break;
        case LwSciStreamEventType_PacketCreate:
            recvPacket();
            if (NUM_PACKETS == ++numPackets) {
                bufDone = true;
            }
            break;
        case LwSciStreamEventType_WaiterAttr:
            recvWaiterAttr();
            break;
        case LwSciStreamEventType_SignalObj:
            recvSignalObj();
            syncDone = true;
            break;
        default:
            assert(0);
            break;
        }
    }

    if (testArg.isC2c) {
        finalizePacket();
    }
    recvPacketComplete();

    // Streaming phase
    recvSetupComplete();
    streaming();
}

void PerfConsumer::createPacket(void)
{
    std::array<LwSciBufAttrList, NUM_ELEMENTS> allocatedAttrs;

    // Receive Elements event
    LwSciStreamEventType event;
    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(c2cPool, QUERY_TIMEOUT, &event));
    assert(event == LwSciStreamEventType_Elements);

    // Query element data
    uint32_t numAttr{ 0U };
    CHECK_LWSCIERR(LwSciStreamBlockElementCountGet(
        c2cPool, LwSciStreamBlockType_Producer, &numAttr));
    assert(NUM_ELEMENTS == numAttr);

    for (uint32_t i{ 0U }; i < numAttr; i++) {
        uint32_t elemType;
        CHECK_LWSCIERR(LwSciStreamBlockElementAttrGet(
            c2cPool, LwSciStreamBlockType_Producer, i,
            &elemType, &allocatedAttrs[i]));
        assert(elemType == i);
    }

    // Indicate element import is done
    CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
        c2cPool, LwSciStreamSetup_ElementImport, true));

    // Send the packet and buffer objects to both producer and consumer
    for (uint32_t i{ 0U }; i < NUM_PACKETS; i++) {
        Packet *packet = &allocatedPackets[i];
        packet->cookie = static_cast<LwSciStreamCookie>(i + 1U);

        CHECK_LWSCIERR(LwSciStreamPoolPacketCreate(c2cPool,
                                                   packet->cookie,
                                                   &packet->handle));

        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            CHECK_LWSCIERR(LwSciBufObjAlloc(allocatedAttrs[j],
                                            &packet->buffers[j]));

            CHECK_LWSCIERR(LwSciStreamPoolPacketInsertBuffer(c2cPool,
                                                             packet->handle,
                                                             j,
                                                             packet->buffers[j]));
        }

        CHECK_LWSCIERR(LwSciStreamPoolPacketComplete(c2cPool, packet->handle));
    }

    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        LwSciBufAttrListFree(allocatedAttrs[i]);
    }
}

void PerfConsumer::finalizePacket(void)
{
    uint32_t numPacketStatus{ 0U };

    // Receive packet status from endpoints
    while (numPacketStatus < NUM_PACKETS) {
        LwSciStreamEventType event;
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(c2cPool,
                                                  QUERY_TIMEOUT,
                                                  &event));
        assert(LwSciStreamEventType_PacketStatus == event);
        numPacketStatus++;
    }

    // Signal packet export and status import is complete
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(c2cPool,
                                       LwSciStreamSetup_PacketExport,
                                       true));
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(c2cPool,
                                       LwSciStreamSetup_PacketImport,
                                       true));
}

void PerfConsumer::streaming(void)
{
    uint32_t numPacketsRecvd{ 0U };

    while (true) {
        // Receive PacketReady event
        LwSciStreamEventType event;
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(endpointHandle,
                                                  QUERY_TIMEOUT,
                                                  &event));
        if (event != LwSciStreamEventType_PacketReady) {
            break;
        }

        // Acquire a packet
        LwSciStreamCookie cookie{ 0U };
        CHECK_LWSCIERR(LwSciStreamConsumerPacketAcquire(endpointHandle,
                                                        &cookie));

        // Stop the present timer when the consumer receives the packet.
        PerfTimer presentTimer;
        if (testArg.latency) {
            presentTimer.setStop();
        }

        // Wait for prefences
        uint32_t const id{ static_cast<uint32_t>(cookie) - 1U };
        assert(id < NUM_PACKETS);
        Packet *packet = &packets[id];

        for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
            LwSciSyncFence fence;
            CHECK_LWSCIERR(LwSciStreamBlockPacketFenceGet(endpointHandle,
                                                          packet->handle,
                                                          0U,
                                                          i,
                                                          &fence));
            CHECK_LWSCIERR(LwSciSyncFenceWait(&fence,
                                              waitContext,
                                              LW_WAIT_INFINITE));
            LwSciSyncFenceClear(&fence);
        }

        // Stop the present timer when the consumer can read the buffer data.
        PerfTimer presentTimer2;
        if (testArg.latency) {
            presentTimer2.setStop();
        }

        ++numPacketsRecvd;

        // Read buffer
        if (testArg.latency) {
            // Get CPU VA of the first element
            void const* vaPtr{ nullptr };
            CHECK_LWSCIERR(LwSciBufObjGetConstCpuPtr(packet->buffers[0], &vaPtr));
#ifdef __QNX__
            uint64_t const *timerPtr{ static_cast<uint64_t const*>(vaPtr) };
#else
            struct timespec const *timerPtr{
                static_cast<struct timespec const*>(vaPtr) };
#endif
            // Read the present start time and the release stop time from
            // the first element
            presentTimer.setStart(timerPtr[0]);
            presentTimer2.setStart(timerPtr[0]);
            presentListMutex.lock();
            presentIntervals.push_back(presentTimer.checkInterval());
            presentIntervals2.push_back(presentTimer2.checkInterval());
            presentListMutex.unlock();

            if (releaseTimer[id].isSet()) {
                releaseTimer[id].setStop(timerPtr[1]);
                releaseListMutex.lock();
                releaseIntervals.push_back(releaseTimer[id].checkInterval());
                releaseListMutex.unlock();
            }
        }

        // Generate postfences
        std::vector<LwSciSyncFence> postfences(NUM_ELEMENTS,
                                               LwSciSyncFenceInitializer);
        for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
            if (syncs[i] != nullptr) {
                CHECK_LWSCIERR(LwSciSyncObjGenerateFence(syncs[i],
                                                         &postfences[i]));
                CHECK_LWSCIERR(LwSciStreamBlockPacketFenceSet(
                    endpointHandle, packet->handle, i, &postfences[i]));
            }
        }

        // Start the release timer
        if (testArg.latency){
            releaseTimer[id].setStart();
        }

        // Release the packet back to the producer
        CHECK_LWSCIERR(LwSciStreamConsumerPacketRelease(endpointHandle,
                                                        packet->handle));
        // Signal postfences
        for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
            if (syncs[i] != nullptr) {
                CHECK_LWSCIERR(LwSciSyncObjSignal(syncs[i]));
                LwSciSyncFenceClear(&postfences[i]);
            }
        }
    }
    printf("Consumer received %d packet(s)\n", numPacketsRecvd);
}

void PerfConsumer::setEndpointBufAttr(LwSciBufAttrList attrList)
{
    LwSciBufType bufType{ LwSciBufType_RawBuffer };
    LwSciBufAttrValAccessPerm perm{ LwSciBufAccessPerm_Readonly };
    bool cpuaccess_flag{ true };

    LwSciBufAttrKeyValuePair bufAttrs[] = {
        { LwSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { LwSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
            sizeof(cpuaccess_flag) },
    };

    CHECK_LWSCIERR(LwSciBufAttrListSetAttrs(attrList,
                    bufAttrs,
                    sizeof(bufAttrs) / sizeof(LwSciBufAttrKeyValuePair)));
}
