//! \file
//! \brief LwSciStream APIs unit testing - Packet Streaming, event-based.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include <thread>
#include <array>
#include <algorithm>
#include <atomic>
#include "lwscistreamtest.h"

//==============================================
// Define PacketEventServiceStreaming test suite
//==============================================

class PacketEventServiceStreaming :
    public LwSciStreamTest
{
protected:
    static constexpr uint16_t POOL_INDEX = 0;
    static constexpr uint16_t PRODUCER_INDEX = 1;
    static constexpr uint16_t CONSUMER_INDEX = 2;

    // [POOL_INDEX] for pool; [PRODUCER_INDEX] for producer;
    // [CONSUMER_INDEX..MAX_CONSUMERS] for consumers;
    using NotifierArray = std::array<LwSciEventNotifier*,CONSUMER_INDEX+MAX_CONSUMERS>;
    using MarkerArray = std::array<bool,CONSUMER_INDEX+MAX_CONSUMERS>;
    NotifierArray notifiers;

protected:
    virtual void SetUp()
    {
        ASSERT_EQ(LwSciError_Success,
            LwSciEventLoopServiceCreate(1U, &eventLoopService));

        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);

        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        prodSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, prodSyncAttrList);

        consSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, consSyncAttrList);

        notifiers.fill(nullptr);
    };

    virtual void TearDown()
    {
        for (auto& elm: notifiers) {
            if (elm != nullptr) {
                elm->Delete(elm);
            }
        }
    }

    inline virtual void createBlocks(
        QueueType type,
        uint32_t numConsumersParam = 1U,
        uint32_t numPacketsParam = NUM_PACKETS)
    {
        numConsumers = numConsumersParam;
        numPackets = numPacketsParam;

        // Create blocks
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamStaticPoolCreate(numPackets, &pool));
        ASSERT_NE(0, pool);

        // Configure the pool to use EventService.
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventServiceSetup(pool,
                &eventLoopService->EventService, &notifiers[POOL_INDEX]));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerCreate(pool, &producer));
        ASSERT_NE(0, producer);

        for (uint32_t n = 0U; n < numConsumers; n++) {
            switch (type) {
            case QueueType::Mailbox:
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamMailboxQueueCreate(&queue[n]));
                break;
            case QueueType::Fifo:
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamFifoQueueCreate(&queue[n]));
                break;
            default:
                ASSERT_TRUE(false) << "Invalid queue type\n";
                break;
            }
            ASSERT_NE(0, queue[n]);

            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerCreate(queue[n], &consumer[n]));
            ASSERT_NE(0, consumer[n]);
        }

        int32_t maxDstConnections = 0U;
        ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxMulticastOutputs, &maxDstConnections));
        ASSERT_LE(numConsumers, maxDstConnections);

        if (numConsumers > 1U) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamMulticastCreate(numConsumers, &multicast));
            ASSERT_NE(0, multicast);

        }

        // Configure the producer to use EventService.
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventServiceSetup(producer,
                &eventLoopService->EventService, &notifiers[PRODUCER_INDEX]));

        // Configure the consumers to use EventService.
        for (uint32_t n = 0; n < numConsumers; n++) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventServiceSetup(consumer[n],
                    &eventLoopService->EventService, &notifiers[CONSUMER_INDEX+n]));
        }

    }

    inline virtual void createSync(uint32_t prodSyncCountParam = NUM_SYNCOBJS,
                                   uint32_t consSyncCountParam = NUM_SYNCOBJS)
    {
        LwSciStreamEventType event;

        ASSERT_TRUE(prodSyncCountParam <= MAX_SYNC_COUNT);
        ASSERT_TRUE(consSyncCountParam <= MAX_SYNC_COUNT);

        // Clamp number of sync objects to number of elements.
        //   Passing in a sync count less than the element count would allow
        //   testing of a case where some elements were asynchronous and
        //   some were synchronous.
        if (prodSyncCountParam > elementCount) {
            prodSyncCountParam = elementCount;
        }
        if (consSyncCountParam > elementCount) {
            consSyncCountParam = elementCount;
        }

        // Initialize prodSyncCount and consSyncCount
        //   If synchronous only flag is set for either, the count is zero
        uint32_t perConsSync = 0U;
        if (!prodSynchronousOnly) {
            prodSyncCount = prodSyncCountParam;
        }
        if (!consSynchronousOnly) {
            for (uint32_t n = 0U; n < numConsumers; n++) {
                consSyncCount[n] = consSyncCountParam;
            }
            perConsSync = consSyncCountParam;
            totalConsSync = consSyncCountParam * numConsumers;
        }

        // Producer sends its sync waiter requirements to the consumer.
        for (uint32_t i = 0; i < prodSyncCount; i++) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementWaiterAttrSet(
                    producer, i, prodSyncAttrList));
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_WaiterAttrExport, true));

        // Consumer sends its sync object requirement to the producer
        for (uint32_t n = 0U; n < numConsumers; n++) {
            for (uint32_t i = 0; i < consSyncCount[n]; i++) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementWaiterAttrSet(
                        consumer[n], i, consSyncAttrList));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_WaiterAttrExport, true));
        }

        // Query max number of sync objects
        int32_t maxNumSyncObjs = 0;
        ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

        // Wake up for a new event on producer
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[PRODUCER_INDEX], 0L));

        // Producer receives consumer's sync object requirement
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_WaiterAttr, event);
        for (uint32_t i = 0; i < elementCount; i++) {
            LwSciSyncAttrList eventSyncAttrList;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementWaiterAttrGet(
                    producer, i, &eventSyncAttrList));
            if (i < perConsSync) {
                ASSERT_NE(nullptr, eventSyncAttrList);
                LwSciSyncAttrListFree(eventSyncAttrList);
            } else {
                ASSERT_EQ(nullptr, eventSyncAttrList);
            }
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_WaiterAttrImport, true));

        // Producer creates sync objects based on consumer's requirement and
        //   sends the sync object to the consumer.
        // TODO: This doesn't take into account whether the consumer is
        //   synchronous for a given element
        for (uint32_t i = 0; i < prodSyncCount; i++) {
            getSyncObj(syncModule, prodSyncObjs[i]);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementSignalObjSet(
                    producer, i, prodSyncObjs[i]));
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_SignalObjExport, true));

        // Wake up for a new event on consumer
        for (uint32_t n = 0U; n < numConsumers; n++) {
            ASSERT_EQ(LwSciError_Success,
                eventLoopService->WaitForEvent(notifiers[CONSUMER_INDEX+n], 0L));
        }

        for (uint32_t n = 0U; n < numConsumers; ++n) {
            // Consumer receives producer's sync object requirement
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_WaiterAttr, event);
            for (uint32_t i = 0; i < elementCount; i++) {
                LwSciSyncAttrList eventSyncAttrList;
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementWaiterAttrGet(
                        consumer[n], i, &eventSyncAttrList));
                if (i < prodSyncCount) {
                    ASSERT_NE(nullptr, eventSyncAttrList);
                    LwSciSyncAttrListFree(eventSyncAttrList);
                } else {
                    ASSERT_EQ(nullptr, eventSyncAttrList);
                }
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_WaiterAttrImport, true));

            // Consumer creates sync objects based on producer's requirement
            //   and sends the sync object to the producer.
            // TODO: This doesn't take into account whether the producer is
            //   synchronous
            for (uint32_t i = 0U; i <  consSyncCount[n]; i++) {
                getSyncObj(syncModule, consSyncObjs[n][i]);
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementSignalObjSet(
                        consumer[n], i, consSyncObjs[n][i]));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_SignalObjExport, true));
        }

        // Wake up for a new event on producer
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[PRODUCER_INDEX], 0L));

        // Producer receives consumer's sync objects
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_SignalObj, event);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_SignalObjImport, true));

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer receives producer's sync objects
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_SignalObj, event);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_SignalObjImport, true));
        }
    }

    inline static bool allTrue(MarkerArray const &markers)
    {
        return std::all_of(markers.begin(), markers.end(), [](bool val) {
            return (val == true);
        });
    }
    inline static bool allFalse(MarkerArray const &markers)
    {
        return std::all_of(markers.begin(), markers.end(), [](bool val) {
            return (val == false);
        });
    }
    inline static void copyTrue(MarkerArray const &src, MarkerArray &dst)
    {
        for (uint32_t i = 0; i < src.size(); ++i) {
            if (src[i]) {
                dst[i] = true;
            }
        }
    }

    inline virtual void connectStream()
    {
        // Connect blocks to create a complete stream.
        if (numConsumers == 1U) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockConnect(producer, consumer[0]));
        } else {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockConnect(producer, multicast));
            for (uint32_t n = 0U; n < numConsumers; n++) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockConnect(multicast, consumer[n]));
            }
        }

        LwSciStreamEventType event;

        // Check pool
        ASSERT_EQ(LwSciError_Success,
                eventLoopService->WaitForEvent(notifiers[POOL_INDEX], 0L));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(pool, 0L, &event));
        ASSERT_EQ(LwSciStreamEventType_Connected, event);


        // Check on producer and consumer
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[PRODUCER_INDEX], 0L));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, 0L, &event));
        ASSERT_EQ(LwSciStreamEventType_Connected, event);

        for (uint32_t n = 0; n < numConsumers; n++) {
            ASSERT_EQ(LwSciError_Success,
                eventLoopService->WaitForEvent(notifiers[CONSUMER_INDEX+n], 0L));

            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], 0L, &event));
            ASSERT_EQ(LwSciStreamEventType_Connected, event);

            // Check queue
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(queue[n], 0L, &event));
            ASSERT_EQ(LwSciStreamEventType_Connected, event);
        }

        // Check multicast
        if (numConsumers > 1U) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(multicast, 0L, &event));
            ASSERT_EQ(LwSciStreamEventType_Connected, event);
        }
    }

    inline virtual void packetAttrSetup(uint32_t numElements = NUM_PACKET_ELEMENTS)
    {
        LwSciStreamEventType event;

        ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
        elementCount = numElements;

        // Query maximum num of packet elements allowed
        int32_t value;
        ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
        ASSERT_TRUE(elementCount <= value);

        // Set producer packet requirements
        uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
        LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
        for (uint32_t i = 0U; i < elementCount; i++) {
            // Use index so that type is unique for each element.
            producerElementType[i] = i;
            producerElementAttr[i] = rawBufAttrList;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementAttrSet(producer,
                                               producerElementType[i],
                                               producerElementAttr[i]));
        }

        // Complete producer element setup
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      producer, LwSciStreamSetup_ElementExport, true));

        // Set consumer packet requirements
        uint32_t consumerElementType[MAX_ELEMENT_PER_PACKET];
        LwSciBufAttrList consumerElementAttr[MAX_ELEMENT_PER_PACKET];
        for (uint32_t i = 0U; i < elementCount; i++) {
            // Use index so that type is unique for each element.
            consumerElementType[i] = i;
            consumerElementAttr[i] = rawBufAttrList;
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            for (uint32_t i = 0U; i < elementCount; i++) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementAttrSet(consumer[n],
                                                   consumerElementType[i],
                                                   consumerElementAttr[i]));
            }

            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockSetupStatusSet(
                          consumer[n], LwSciStreamSetup_ElementExport, true));
        }

        // Wake up for a new event on pool
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[POOL_INDEX], 0L));

        // Pool receives Elements event
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_Elements, event);

        // Complete pool element import
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementImport, true));

        // Pool record packet attr
        for (uint32_t i = 0U; i < elementCount; i++) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementAttrSet(pool,
                                               consumerElementType[i],
                                               consumerElementAttr[i]));
        }

        // Complete pool element setup
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementExport, true));

        // Wake up for a new event on producer
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[PRODUCER_INDEX], 0L));

        // Producer receives Elements event
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_Elements, event);

        // Complete producer element import
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      producer, LwSciStreamSetup_ElementImport, true));

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // TODO: Missing an event service wait here?

            // Consumer receives Elements event
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_Elements, event);

            // Complete consumer element import
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockSetupStatusSet(
                          consumer[n], LwSciStreamSetup_ElementImport, true));

        }
    }

    inline virtual void createPacket()
    {
        LwSciStreamEventType event;
        LwSciStreamPacket eventHandle;

        for (uint32_t i = 0U; i < numPackets; ++i) {
            // Choose pool's cookie and for new packet
            LwSciStreamPacket packetHandle;
            LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));

            // Save the cookie-to-handle mapping
            poolCPMap.emplace(poolCookie, packetHandle);

            // Register buffer to packet handle
            LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
            for (uint32_t k = 0; k < elementCount; ++k) {
                makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamPoolPacketInsertBuffer(pool,
                                                      packetHandle, k,
                                                      poolElementBuf[k]));
            }

            // Indicate packet definition is complete
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketComplete(pool, packetHandle));

            // Wake up for a new event on producer
            ASSERT_EQ(LwSciError_Success,
                eventLoopService->WaitForEvent(notifiers[PRODUCER_INDEX], 0));

            // Producer receives PacketCreate event
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketCreate, event);

            // Retrieve handle for packet
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketNewHandleGet(producer, &eventHandle));

            // Assign cookie to producer packet handle
            LwSciStreamPacket producerPacket = eventHandle;
            LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
            LwSciError producerError = LwSciError_Success;

            // Producer accepts a packet provided by the pool
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketStatusSet(
                producer, producerPacket, producerCookie, producerError));

            // Save the cookie-to-handle mapping
            prodCPMap.emplace(producerCookie, producerPacket);

            // Receive the packet at the consumers
            for (uint32_t n = 0U; n < numConsumers; n++) {

                // TODO: Missing wait for event?

                // Consumer receives PacketCreate event
                ASSERT_EQ(LwSciError_Success,
                          LwSciStreamBlockEventQuery(consumer[n],
                                                     eventQueryTimeout,
                                                     &event));
                ASSERT_EQ(LwSciStreamEventType_PacketCreate, event);

                // Retrieve handle for packet
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockPacketNewHandleGet(consumer[n],
                                                       &eventHandle));

                // Assign cookie to consumer packet handle
                LwSciStreamPacket consumerPacket = eventHandle;
                LwSciStreamCookie consumerCookie
                    = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
                LwSciError consumerError = LwSciError_Success;

                // Consumer accepts packet provided by the pool
                ASSERT_EQ(LwSciError_Success,
                          LwSciStreamBlockPacketStatusSet(consumer[n],
                                                          consumerPacket,
                                                          consumerCookie,
                                                          consumerError));

                // Save the cookie-to-handle mapping
                consCPMap[n].emplace(consumerCookie, consumerPacket);
            }

            // Free the buffer objs in the pool
            for (uint32_t k = 0; k < elementCount; ++k) {
                LwSciBufObjFree(poolElementBuf[k]);
            }
        }
    }

    inline virtual void checkPacketStatus()
    {
        LwSciStreamEventType event;
        uint32_t numPacketComplete = 0U;

        // Pool must receive all PacketStatus events
        while (numPacketComplete < numPackets) {

            // Pool receives packet
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketStatus, event);

            numPacketComplete++;
        }

        // Pool indicates packet list completed
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
               pool, LwSciStreamSetup_PacketExport, true));
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
               pool, LwSciStreamSetup_PacketImport, true));

        // Producer receives PacketsComplete event
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketsComplete, event);

        // Producer indicates packet setup completed
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
               producer, LwSciStreamSetup_PacketImport, true));

        // Complete setup on consumers
        for (uint32_t n = 0U; n < numConsumers; n++) {

            // Consumer receives PacketsComplete event
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout,
                                           &event));
            ASSERT_EQ(LwSciStreamEventType_PacketsComplete, event);

            // Consumer indicates packet setup completed
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSetupStatusSet(
                    consumer[n], LwSciStreamSetup_PacketImport, true));
        }

        // Wake up for a new event on pool
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[POOL_INDEX], 0L));
    }

    inline void setupComplete()
    {
        LwSciStreamEventType event;
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[PRODUCER_INDEX], 0));
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(producer,
                                             eventQueryTimeout,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(pool,
                                             eventQueryTimeout,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        if (1U < numConsumers) {
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockEventQuery(multicast,
                                                 eventQueryTimeout,
                                                 &event));
            EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        }
        for (uint32_t n = 0U; n < numConsumers; n++) {
            ASSERT_EQ(LwSciError_Success,
                eventLoopService->WaitForEvent(notifiers[CONSUMER_INDEX+n], 0L));
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockEventQuery(queue[n],
                                                 eventQueryTimeout,
                                                 &event));
            EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockEventQuery(consumer[n],
                                                 eventQueryTimeout,
                                                 &event));
            EXPECT_EQ(LwSciStreamEventType_SetupComplete, event);
        }
    };

    inline virtual void streaming(uint32_t numFrames = NUM_FRAMES)
    {
        uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

        bool firstFrame = true;
        int64_t timeout = EVENT_QUERY_TIMEOUT;
        uint32_t consumedFrame = 0U;
        uint32_t presentedFrame = 0U;
        MarkerArray marker;
        marker.fill(false);

        // Loop logic:
        // First, do timed wait on the event notififers. The blocks whose
        // marker are set could have zero (spurious wakeup) or multiple events
        // pending. In each iteration of the loop we query one event from each
        // of the notified blocks. If there is no more event in the block, we
        // clear its marker. Then in the subsequent iterations, we use
        // non-blocking wait to check the blocks if there is new incoming
        // signaling, while keeping querying one event from each of the
        // notified blocks (and clear the markers of blocks with zero pending
        // event).
        // After all markers are cleared, we repeat the above process by doing
        // timed wait again.

        // In single-thread mode no frame should be missed by consumer in
        // mailbox mode. So simply use (numFrames*numConsumers) as the stoppage.
        while (consumedFrame < (numFrames*numConsumers)) {
            MarkerArray tmpMarker;
            LwSciError err;
            LwSciStreamEventType event;
            LwSciStreamPacket handle;
            LwSciStreamCookie cookie;

            err = eventLoopService->WaitForMultipleEvents(
                    notifiers.data(),
                    CONSUMER_INDEX+numConsumers,
                    timeout,
                    tmpMarker.data());
            ASSERT_TRUE(
                err == LwSciError_Success || err == LwSciError_Timeout);

            // Element-wise OR query result into the marker
            copyTrue(tmpMarker, marker);

            // Process consumer
            for (uint32_t n = 0; n < numConsumers; n++) {
                if (marker[CONSUMER_INDEX+n]) {
                    err = LwSciStreamBlockEventQuery(consumer[n], 0L, &event);
                    if (err == LwSciError_Success) {
                        //  Consumer gets a packet from the queue
                        ASSERT_EQ(LwSciStreamEventType_PacketReady,
                            event);
                        ASSERT_EQ(LwSciError_Success,
                            LwSciStreamConsumerPacketAcquire(consumer[n],
                                                             &cookie));
                        handle = (consCPMap[n])[cookie];

                        // Consumer returns a data packet to the stream
                        ASSERT_EQ(LwSciError_Success,
                            LwSciStreamConsumerPacketRelease(consumer[n],
                                                             handle));
                        consumedFrame++;

                    } else if (err == LwSciError_Timeout) {
                        // No more event. Clear the marker
                        marker[CONSUMER_INDEX+n] = false;
                    } else {
                        ASSERT_TRUE(false) << "Producer block received an unexpected event\n";
                    }
                }
            }
            // Process producer
            //   There may not be an additional LwSciEvent signal between
            //   finishing setup events and the PacketReady event
            if (marker[PRODUCER_INDEX] || firstFrame) {
                err = LwSciStreamBlockEventQuery(producer, 0L, &event);
                if (err == LwSciError_Success) {
                    firstFrame = false;
                    ASSERT_EQ(LwSciStreamEventType_PacketReady, event);
                    if( presentedFrame < numFrames )
                    {
                        ASSERT_EQ(LwSciError_Success,
                            LwSciStreamProducerPacketGet(producer, &cookie));

                        handle = prodCPMap[cookie];

                        // Producer inserts a data packet into the stream
                        ASSERT_EQ(LwSciError_Success,
                            LwSciStreamProducerPacketPresent(producer, handle));
                        presentedFrame++;
                    }
                } else if (err == LwSciError_Timeout) {
                    // No more event. Clear the marker
                    marker[PRODUCER_INDEX] = false;
                } else {
                    ASSERT_TRUE(false) << "Consumer block received an unexpected event\n";
                }
            }
            timeout = allFalse(marker) ? EVENT_QUERY_TIMEOUT : 0L;
        }
    }

    inline void disConnectStreaming()
    {
        LwSciStreamEventType event;

        // Delete producer block
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(producer));

        // Wake up for a new event on pool
        ASSERT_EQ(LwSciError_Success,
            eventLoopService->WaitForEvent(notifiers[POOL_INDEX], 0L));

        // Pool receives LwSciStreamEventType_Disconnected event
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_Disconnected, event);

        // Delete pool block
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(pool));

        for (uint32_t n = 0; n < numConsumers; n++) {
            // Queue receives LwSciStreamEventType_Disconnected event
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete queue block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(queue[n]));

            // Wake up for a new event on consumer
            ASSERT_EQ(LwSciError_Success,
                eventLoopService->WaitForEvent(notifiers[CONSUMER_INDEX+n], 0L));

            // Consumer receives LwSciStreamEventType_Disconnected event
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete consumer block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(consumer[n]));
        }

        // Check multicast
        if (numConsumers > 1U) {
            // Multicast receives LwSciStreamEventType_Disconnected event
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                multicast, eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete multicast block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(multicast));
        }
      }
};

//==============================================================================
// Test Case 1 : (send and receive frame, fifo mode)
// Producer beings sending payloads to the consumer, and the consumer processes
// them and sends back their packets for reuse. In this simple example, one
// payload being processed at a time.
//==============================================================================

TEST_F(PacketEventServiceStreaming, SendAndReceiveFrame_Fifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();
}

//==========================================================================
// Test Case 2 : (send and receive frame, mailbox mode)
// Producer beings sending payloads to the consumer, and the consumer
// processes them and sends back their packets for reuse. In this
// simple example, one payload being processed at a time.
//==========================================================================

TEST_F(PacketEventServiceStreaming, SendAndReceiveFrame_Mailbox)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();
}

//==============================================================================
// Test Case 3 : (send and receive frame, fifo mode, multiple consumers)
// Producer beings sending payloads to the consumers, and the consumers process
// them and sends back their packets for reuse. In this simple example, one
// payload being processed at a time.
//==============================================================================

TEST_F(PacketEventServiceStreaming, SendAndReceiveFrame_Fifo_Multicast)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();
}

//==========================================================================
// Test Case 4 : (send and receive frame, mailbox mode, multiple consumers)
// Producer beings sending payloads to the consumer, and the consumer
// processes them and sends back their packets for reuse. In this
// simple example, one payload being processed at a time.
//==========================================================================

TEST_F(PacketEventServiceStreaming, SendAndReceiveFrame_Mailbox_Multicast)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();
}

//==============================================================================
// Test Case 5 : Bind Block to EventNotifier for Fifo stream
//==============================================================================

TEST_F(PacketEventServiceStreaming, BindBlockToEventNotifier_Fifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();

    disConnectStreaming();
}

//==========================================================================
// Test Case 6 : Bind Block to EventNotifier for Mailbox stream
//==========================================================================

TEST_F(PacketEventServiceStreaming, BindBlockToEventNotifier_Mailbox)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();

    disConnectStreaming();
}

//==============================================================================
// Test Case 7 : Bind Block to EventNotifier Multicast Fifo stream
//==============================================================================

TEST_F(PacketEventServiceStreaming, BindBlockToEventNotifier_MulticastFifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();

    disConnectStreaming();
}

//==========================================================================
// Test Case 8 : Bind Block to EventNotifier Multicast MailBox stream
//==========================================================================

TEST_F(PacketEventServiceStreaming, BindBlockToEventNotifier_MulticastMailBox)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();

    disConnectStreaming();
}
