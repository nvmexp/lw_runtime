//! \file
//! \brief LwSciStream test Producer client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifdef __QNX__
#include <sys/neutrino.h>
#else
#include <time.h>
#endif

#include "perfproducer.h"
#include <unistd.h>

extern TestArg testArg;

PerfProducer::PerfProducer(LwSciBufModule buf,
                           LwSciSyncModule sync) :
    PerfClient(buf, sync)
{
    if (testArg.testType == CrossProcProd) {
        c2cQueue.resize(testArg.numConsumers, 0U);
        ipcSrc.resize(testArg.numConsumers, 0U);
    }

    for (uint32_t i{ 0U }; i < NUM_PACKETS; i++) {
        allocatedPackets[i].buffers.fill(nullptr);
    }
}

PerfProducer::~PerfProducer(void)
{
    for (uint32_t i{ 0U }; i < NUM_PACKETS; i++) {
        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            if (allocatedPackets[i].buffers[j] != nullptr) {
                LwSciBufObjFree(allocatedPackets[i].buffers[j]);
            }
        }
    }

    // Delete blocks
    if (producer != 0U) {
        LwSciStreamBlockDelete(producer);
        producer = 0U;
    }
    if (pool != 0U) {
        LwSciStreamBlockDelete(pool);
        pool = 0U;
    }
    if (multicast != 0U) {
        LwSciStreamBlockDelete(multicast);
        multicast = 0U;
    }
    for (uint32_t i{ 0U }; i < c2cQueue.size(); i++) {
        if (c2cQueue[i] != 0U) {
            LwSciStreamBlockDelete(c2cQueue[i]);
            c2cQueue[i] = 0U;
        }
    }
    for (uint32_t i{ 0U }; i < ipcSrc.size(); i++) {
        if (ipcSrc[i] != 0U) {
            LwSciStreamBlockDelete(ipcSrc[i]);
            ipcSrc[i] = 0U;
        }
    }
}

LwSciStreamBlock PerfProducer::createStream(
    std::vector<LwSciIpcEndpoint>* ipcEndpoint)
{
    // Create blocks
    CHECK_LWSCIERR(LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));
    CHECK_LWSCIERR(LwSciStreamProducerCreate(pool, &producer));

    if (testArg.numConsumers > 1U) {
        CHECK_LWSCIERR(LwSciStreamMulticastCreate(testArg.numConsumers,
                                                  &multicast));
    }

    if (testArg.testType == CrossProcProd) {
        for (uint32_t i{ 0U }; i < testArg.numConsumers; i++) {
            if (testArg.isC2c) {
                CHECK_LWSCIERR(LwSciStreamFifoQueueCreate(&c2cQueue[i]));
            }

            CHECK_LWSCIERR(LwSciStreamIpcSrcCreate2((*ipcEndpoint)[i],
                                                    syncModule,
                                                    bufModule,
                                                    c2cQueue[i],
                                                    &ipcSrc[i]));
        }
    }

    // Connect blocks
    LwSciStreamBlock connBlock{ 0U };
    if (testArg.testType == CrossProcProd) {
        if (testArg.numConsumers > 1U) {
            CHECK_LWSCIERR(LwSciStreamBlockConnect(producer, multicast));
            for (uint32_t i{ 0U }; i < testArg.numConsumers; i++) {
                CHECK_LWSCIERR(LwSciStreamBlockConnect(multicast, ipcSrc[i]));
            }
        } else {
            CHECK_LWSCIERR(LwSciStreamBlockConnect(producer, ipcSrc[0]));
        }
    } else {
        if (testArg.numConsumers > 1U) {
            CHECK_LWSCIERR(LwSciStreamBlockConnect(producer, multicast));
            connBlock = multicast;
        } else {
            connBlock = producer;
        }
    }

    endpointHandle = producer;

    return connBlock;
}

void PerfProducer::recvConnectEvent(void)
{
    LwSciStreamEventType event;

    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(producer,
                                              QUERY_TIMEOUT,
                                              &event));
    if (event != LwSciStreamEventType_Connected) {
        return;
    }

    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(pool,
                                              QUERY_TIMEOUT,
                                              &event));
    if (event != LwSciStreamEventType_Connected) {
        return;
    }

    if (testArg.numConsumers > 1U) {
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(multicast,
                                                  QUERY_TIMEOUT,
                                                  &event));
        if (event != LwSciStreamEventType_Connected) {
            return;
        }
    }

    if (testArg.isC2c) {
        for (uint32_t i{ 0U }; i < c2cQueue.size(); i++) {
            CHECK_LWSCIERR(LwSciStreamBlockEventQuery(c2cQueue[i],
                                                      QUERY_TIMEOUT,
                                                      &event));
            if (event != LwSciStreamEventType_Connected) {
                return;
            }
        }
    }

    for (uint32_t i{ 0U }; i < ipcSrc.size(); i++) {
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(ipcSrc[i],
                                                  QUERY_TIMEOUT,
                                                  &event));
        if (event != LwSciStreamEventType_Connected) {
            return;
        }
    }
}

void PerfProducer::run(void)
{
    // Setup phase
    recvConnectEvent();
    sendEndpointElements();
    createPacket();

    bool syncDone{ false };
    bool bufDone{ false };
    uint32_t numPackets{ 0U };
    while (!syncDone || !bufDone) {
        LwSciStreamEventType event;
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(producer,
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

    finalizePacket();
    recvPacketComplete();

    // Streaming phase
    recvSetupComplete();
    streaming();
}


void PerfProducer::createPacket(void)
{
    uint32_t numProdAttr{ 0U };
    uint32_t numConsAttr{ 0U };

    std::array<LwSciBufAttrList, NUM_ELEMENTS> prodAttrs;
    std::array<LwSciBufAttrList, NUM_ELEMENTS> consAttrs;
    std::array<LwSciBufAttrList, NUM_ELEMENTS> allocatedAttrs;

    // Receive Elements event
    LwSciStreamEventType event;
    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(pool, QUERY_TIMEOUT, &event));
    assert(event == LwSciStreamEventType_Elements);

    // Query element data
    CHECK_LWSCIERR(LwSciStreamBlockElementCountGet(
        pool, LwSciStreamBlockType_Producer, &numProdAttr));
    assert(NUM_ELEMENTS == numProdAttr);
    CHECK_LWSCIERR(LwSciStreamBlockElementCountGet(
        pool, LwSciStreamBlockType_Consumer, &numConsAttr));
    assert(NUM_ELEMENTS == numConsAttr);
    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        uint32_t elemType;
        CHECK_LWSCIERR(LwSciStreamBlockElementAttrGet(
            pool, LwSciStreamBlockType_Producer, i, &elemType, &prodAttrs[i]));
        assert(elemType == i);
        CHECK_LWSCIERR(LwSciStreamBlockElementAttrGet(
            pool, LwSciStreamBlockType_Consumer, i, &elemType, &consAttrs[i]));
        assert(elemType == i);
    }

    // Indicate element import is done
    CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
        pool, LwSciStreamSetup_ElementImport, true));

    // Reconcile buffer attrs from producer and consumer
    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        LwSciBufAttrList reconciled{ nullptr };
        LwSciBufAttrList oldBufAttr[2] = { prodAttrs[i], consAttrs[i] };
        LwSciBufAttrList conflictlist = nullptr;
        CHECK_LWSCIERR(LwSciBufAttrListReconcile(oldBufAttr,
                                                 2U,
                                                 &allocatedAttrs[i],
                                                 &conflictlist));

        assert(conflictlist == nullptr);

        CHECK_LWSCIERR(LwSciStreamBlockElementAttrSet(pool, i,
                                                      allocatedAttrs[i]));

        LwSciBufAttrListFree(prodAttrs[i]);
        LwSciBufAttrListFree(consAttrs[i]);
    }

    // Indicate element specification is done
    CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
        pool, LwSciStreamSetup_ElementExport, true));

    // Send the packet and buffer objects to both producer and consumer
    for (uint32_t i{ 0U }; i < NUM_PACKETS; i++) {
        Packet *packet = &allocatedPackets[i];
        packet->cookie = static_cast<LwSciStreamCookie>(i + 1U);

        CHECK_LWSCIERR(LwSciStreamPoolPacketCreate(pool,
                                                   packet->cookie,
                                                   &packet->handle));

        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            CHECK_LWSCIERR(LwSciBufObjAlloc(allocatedAttrs[j],
                                            &packet->buffers[j]));

            CHECK_LWSCIERR(LwSciStreamPoolPacketInsertBuffer(pool,
                                                             packet->handle,
                                                             j,
                                                             packet->buffers[j]));
        }

        CHECK_LWSCIERR(LwSciStreamPoolPacketComplete(pool, packet->handle));
    }

    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        LwSciBufAttrListFree(allocatedAttrs[i]);
    }
}

void PerfProducer::finalizePacket(void)
{
    uint32_t numPacketStatus{ 0U };

    // Receive packet status from endpoints
    while (numPacketStatus < NUM_PACKETS) {
        LwSciStreamEventType event;
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(pool,
                                                  QUERY_TIMEOUT,
                                                  &event));
        assert(LwSciStreamEventType_PacketStatus == event);
        numPacketStatus++;
    }

    // Signal packet export and status import is complete
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(pool,
                                       LwSciStreamSetup_PacketExport,
                                       true));
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(pool,
                                       LwSciStreamSetup_PacketImport,
                                       true));
}

void PerfProducer::streaming(void)
{
    uint32_t numPacketsPresented{ 0U };

    for (uint32_t i{ 0U }; i < testArg.numFrames; i++) {
        // Receive PacketReady event
        LwSciStreamEventType event;
        CHECK_LWSCIERR(LwSciStreamBlockEventQuery(endpointHandle,
                                                  QUERY_TIMEOUT,
                                                  &event));
        assert(event == LwSciStreamEventType_PacketReady);

        // Get a packet
        LwSciStreamCookie cookie{ 0U };
        CHECK_LWSCIERR(LwSciStreamProducerPacketGet(endpointHandle,
                                                    &cookie));

        // Stop the release timer
#ifdef __QNX__
        uint64_t releaseStop{ 0U };
        if (testArg.latency) {
            releaseStop = ClockCycles();
        }
#else
        struct timespec releaseStop;
        if (testArg.latency) {
            clock_gettime(CLOCK_REALTIME, &releaseStop);
        }
#endif

        // Wait for prefences
        uint32_t const id{ static_cast<uint32_t>(cookie) - 1U };
        assert(id < NUM_PACKETS);
        Packet *packet = &packets[id];

        for (uint32_t j{ 0U }; j <testArg.numSyncs; j++) {
            LwSciSyncFence fence;
            CHECK_LWSCIERR(LwSciStreamBlockPacketFenceGet(endpointHandle,
                                                          packet->handle,
                                                          0U,
                                                          j,
                                                          &fence));
            CHECK_LWSCIERR(LwSciSyncFenceWait(&fence,
                                              waitContext,
                                              LW_WAIT_INFINITE));
            LwSciSyncFenceClear(&fence);
        }

        // Write to buffer
        if (testArg.latency) {
            // Get CPU VA of the first element
            void* vaPtr{ nullptr };
            CHECK_LWSCIERR(LwSciBufObjGetCpuPtr(packet->buffers[0], &vaPtr));

            // Start the present timer
#ifdef __QNX__
            uint64_t *timerPtr{ static_cast<uint64_t *>(vaPtr) };
            uint64_t presentStart{ ClockCycles() };
#else
            struct timespec *timerPtr{ static_cast<struct timespec *>(vaPtr) };
            struct timespec presentStart;
            clock_gettime(CLOCK_REALTIME, &presentStart);
#endif
            // Write the present start time and the release stop time
            // to the first element
            timerPtr[0] = presentStart;
            timerPtr[1] = releaseStop;
        }

        // Generate postfences
        std::vector<LwSciSyncFence> postfences(NUM_ELEMENTS,
                                               LwSciSyncFenceInitializer);
        for (uint32_t j{ 0U }; j < testArg.numSyncs; j++) {
            if (syncs[j] != nullptr) {
                CHECK_LWSCIERR(LwSciSyncObjGenerateFence(syncs[j],
                                                         &postfences[j]));
                CHECK_LWSCIERR(LwSciStreamBlockPacketFenceSet(
                    endpointHandle, packet->handle, j, &postfences[j]));
            }
        }

        // Simulate producer producer present rate
        usleep(testArg.sleepUs);

        // Present the packet to the consumer
        CHECK_LWSCIERR(LwSciStreamProducerPacketPresent(endpointHandle,
                                                        packet->handle));
        // Signal postfences
        for (uint32_t j{ 0U }; j < testArg.numSyncs; j++) {
            if (syncs[j] != nullptr) {
                CHECK_LWSCIERR(LwSciSyncObjSignal(syncs[j]));
                LwSciSyncFenceClear(&postfences[j]);
            }
        }

        ++numPacketsPresented;
    }

    printf("Producer presented %d packet(s)\n", numPacketsPresented);
}

void PerfProducer::setEndpointBufAttr(LwSciBufAttrList attrList)
{
    LwSciBufType bufType{ LwSciBufType_RawBuffer };
    uint64_t rawsize{ testArg.bufSize };
    uint64_t align{ 4 * 1024 };
    LwSciBufAttrValAccessPerm perm{ LwSciBufAccessPerm_ReadWrite };
    bool cpuaccess_flag{ true };

    LwSciBufAttrKeyValuePair rawbuffattrs[] = {
        { LwSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { LwSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
        { LwSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
        { LwSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { LwSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
            sizeof(cpuaccess_flag) },
    };

    CHECK_LWSCIERR(LwSciBufAttrListSetAttrs(attrList,
                    rawbuffattrs,
                    sizeof(rawbuffattrs) / sizeof(LwSciBufAttrKeyValuePair)));
}
