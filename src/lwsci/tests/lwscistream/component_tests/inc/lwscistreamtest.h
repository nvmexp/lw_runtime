//! \file
//! \brief LwSciStream unit testing.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef LWSCISTREAMTEST_H
#define LWSCISTREAMTEST_H

#include <unordered_map>
#include "util.h"

using std::unordered_map;
using CookiePacketMap = unordered_map<LwSciStreamCookie, LwSciStreamPacket>;

constexpr int64_t EVENT_QUERY_TIMEOUT = 50;
constexpr uint32_t NUM_PACKETS = 5U;
constexpr uint32_t MAX_SYNC_COUNT = 4U;
constexpr uint32_t NUM_SYNCOBJS = 2U;
constexpr uint32_t MAX_ELEMENT_PER_PACKET = 24U;
constexpr uint32_t NUM_PACKET_ELEMENTS = 2U;
constexpr uint32_t NUM_FRAMES = 10U;
constexpr uint32_t MAX_CONSUMERS = 4U;
constexpr uint32_t NUM_CONSUMERS = 2U;
constexpr LwSciStreamCookie COOKIE_BASE = 100U;
constexpr uint32_t MAX_WAIT_ITERS = 100U;
constexpr uint32_t MAX_DST_CONNECTIONS = 4U;
constexpr uint32_t ALLOWED_MAX_ELEMENTS = 17U;

typedef enum {
    Mailbox,
    Fifo
} QueueType;

typedef enum {
    EventService,
    Internal
} SignalMode;

//==========================================================================
// Define LwSciStream test suite.
//==========================================================================
class LwSciStreamTest :
    public ::testing::Test
{
protected:
    LwSciStreamBlock producer = 0U;
    LwSciStreamBlock pool = 0U;
    LwSciStreamBlock consumer[MAX_CONSUMERS] = {};
    LwSciStreamBlock queue[MAX_CONSUMERS] = {};
    LwSciStreamBlock multicast = 0U;
    LwSciStreamBlock presentsync = 0U;
    LwSciStreamBlock returnsync = 0U;

    // Consumer count
    uint32_t numConsumers = 0U;

    // LwSciSync member data
    uint32_t prodSyncCount = 0U;
    uint32_t consSyncCount[MAX_CONSUMERS] = {};
    uint32_t totalConsSync = 0U;

    LwSciSyncModule syncModule = nullptr;

    bool prodSynchronousOnly;
    LwSciSyncAttrList prodSyncAttrList;
    // All consumers use the same sync attributes.
    bool consSynchronousOnly;
    LwSciSyncAttrList consSyncAttrList;

    LwSciSyncObj prodSyncObjs[MAX_SYNC_COUNT];
    LwSciSyncObj consSyncObjs[MAX_CONSUMERS][MAX_SYNC_COUNT];

    // LwSciBuf member data
    uint32_t elementCount = 0U;
    LwSciBufModule bufModule = nullptr;
    LwSciBufAttrList rawBufAttrList = nullptr;

    // Packet member data
    uint32_t numPackets = 0U;

    CookiePacketMap poolCPMap;
    CookiePacketMap prodCPMap;
    CookiePacketMap consCPMap[MAX_CONSUMERS];

    // Wait time of LwSciStreamBlockEventQuery
    // The default value is zero, which is used
    // in single-thread test cases.
    int64_t eventQueryTimeout = 0L;

    // EventService object
    LwSciEventLoopService* eventLoopService = nullptr;

    LwSciStreamTest()
    {
        prodSyncAttrList = nullptr;
        consSyncAttrList = nullptr;

        for (uint32_t i = 0U; i < MAX_SYNC_COUNT; i++) {
            prodSyncObjs[i] = nullptr;
            for (uint32_t n = 0U; n < MAX_CONSUMERS; n++) {
                consSyncObjs[n][i] = nullptr;
            }
        }
    };

    ~LwSciStreamTest() override
    {
        if (producer != 0U) {
            LwSciStreamBlockDelete(producer);
        }
        if (pool != 0U) {
            LwSciStreamBlockDelete(pool);
        }
        for (uint32_t n = 0U; n < MAX_CONSUMERS; n++) {
            if (queue[n] != 0U) {
                LwSciStreamBlockDelete(queue[n]);
            }
            if (consumer[n] != 0U) {
                LwSciStreamBlockDelete(consumer[n]);
            }
        }
        if (multicast != 0U) {
            LwSciStreamBlockDelete(multicast);
        }

        if (presentsync != 0U) {
            LwSciStreamBlockDelete(presentsync);
        }

        if (returnsync != 0U) {
            LwSciStreamBlockDelete(returnsync);
        }

        // Free buffer resources
        if (rawBufAttrList != nullptr) {
            LwSciBufAttrListFree(rawBufAttrList);
        }
        if (bufModule != nullptr) {
            LwSciBufModuleClose(bufModule);
        }

        // Free sync resources
        if (prodSyncAttrList != nullptr) {
            LwSciSyncAttrListFree(prodSyncAttrList);
        }
        if (consSyncAttrList != nullptr) {
            LwSciSyncAttrListFree(consSyncAttrList);
        }

        for (uint32_t i = 0U; i < MAX_SYNC_COUNT; i++) {
            if (prodSyncObjs[i] != nullptr) {
                LwSciSyncObjFree(prodSyncObjs[i]);
            }
            for (uint32_t n = 0U; n < MAX_CONSUMERS; n++) {
                if (consSyncObjs[n][i] != nullptr) {
                    LwSciSyncObjFree(consSyncObjs[n][i]);
                }
            }
        }

        if (syncModule != nullptr) {
            LwSciSyncModuleClose(syncModule);
        }

        // There is no order imposed for the deletion of an LwSciEventService
        // object and the LwSciStreamBlock blocks configured to use it.
        // It is okay to delete the blocks before deleting the LwSciEventService
        // object.
        if (eventLoopService != nullptr) {
            eventLoopService->EventService.Delete(
                &eventLoopService->EventService);
        }
    };

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
    };

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

        // Check Connect* events
        LwSciStreamEventType event;

        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_Connected, event);

        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_Connected, event);

        for (uint32_t n = 0U; n < numConsumers; n++) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);

            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(queue[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);
        }

        // Check multicast
        if (numConsumers > 1U) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(multicast, eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);
        }

        // Make sure producer and pool see correct number of consumers
        uint32_t queryVal;
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockConsumerCountGet(producer, &queryVal));
        EXPECT_EQ(queryVal, numConsumers);
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockConsumerCountGet(pool, &queryVal));
        EXPECT_EQ(queryVal, numConsumers);
    };

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
        // sends the sync object to the consumer.
        // TODO: This doesn't take into account whether the consumer is
        //   synchronous
        for (uint32_t i = 0; i < prodSyncCount; i++) {
            getSyncObj(syncModule, prodSyncObjs[i]);
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementSignalObjSet(
                    producer, i, prodSyncObjs[i]));
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSetupStatusSet(
                producer, LwSciStreamSetup_SignalObjExport, true));

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
            // TODO: This doesn't take into account whether the producer
            //   is synchronous for the element
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
    };

    inline virtual void packetAttrSetup(uint32_t numElements = NUM_PACKET_ELEMENTS)
    {
        LwSciStreamEventType event;
        uint32_t receivedCount;

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
                      LwSciStreamBlockElementAttrSet(producer, i,
                                                     rawBufAttrList));
        }

        // Indicate producer element specification is done
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
                          LwSciStreamBlockElementAttrSet(consumer[n], i,
                                                         rawBufAttrList));
            }

            // Indicate consumer element specification is done
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockSetupStatusSet(
                          consumer[n], LwSciStreamSetup_ElementExport, true));
        }

        // Pool receives Elements event
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_Elements, event);

        // Check element count from producer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      pool, LwSciStreamBlockType_Producer, &receivedCount));
        EXPECT_EQ(elementCount, receivedCount);

        // Check elements from producer
        for (uint32_t i = 0; i < elementCount; i++) {
            uint32_t elemType;
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrGet(
                          pool, LwSciStreamBlockType_Producer, i,
                          &elemType, nullptr));
            EXPECT_EQ(producerElementType[i], elemType);
            // TODO: Check attrs
        }

        // Check element count from consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      pool, LwSciStreamBlockType_Consumer, &receivedCount));
        EXPECT_EQ(elementCount, receivedCount);

        // Check elements from consumer
        for (uint32_t i = 0; i < elementCount; i++) {
            uint32_t elemType;
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrGet(
                          pool, LwSciStreamBlockType_Consumer, i,
                          &elemType, nullptr));
            EXPECT_EQ(consumerElementType[i], elemType);
            // TODO: Check attrs
        }

        // Indicate pool element import is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementImport, true));

        // Pool record packet attr
        for (uint32_t i = 0U; i < elementCount; i++) {
            // Use index so that type is unique for each element.
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrSet(pool, i, rawBufAttrList));
        }

        // Indicate pool element specification is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementExport, true));

        // Producer receives element specification from pool
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        EXPECT_EQ(LwSciStreamEventType_Elements, event);

        // Check element count at producer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      producer, LwSciStreamBlockType_Pool, &receivedCount));
        EXPECT_LE(elementCount, receivedCount);

        // Check elements at producer
        for (uint32_t i = 0; i < receivedCount; i++) {
            uint32_t elemType;
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementAttrGet(
                          producer, LwSciStreamBlockType_Pool, i,
                          &elemType, nullptr));
            EXPECT_EQ(i, elemType);
            // TODO: Check attrs
        }

        // Indicate producer element import is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      producer, LwSciStreamSetup_ElementImport, true));

        // Check all consumers
        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer receives element specification from pool
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Elements, event);

            // Check element count at consumer
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockElementCountGet(
                          consumer[n], LwSciStreamBlockType_Pool,
                          &receivedCount));
            EXPECT_LE(elementCount, receivedCount);

            // Check elements at consumer and indicate used
            for (uint32_t i = 0; i < receivedCount; i++) {
                uint32_t elemType;
                EXPECT_EQ(LwSciError_Success,
                          LwSciStreamBlockElementAttrGet(
                              consumer[n], LwSciStreamBlockType_Pool, i,
                              &elemType, nullptr));
                EXPECT_EQ(i, elemType);
                // TODO: Check attrs
                EXPECT_EQ(LwSciError_Success,
                          LwSciStreamBlockElementUsageSet(
                              consumer[n], i, true));
            }

            // Indicate consumer element import is done
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockSetupStatusSet(
                          consumer[n], LwSciStreamSetup_ElementImport, true));
        }
    };

    inline virtual void createPacket()
    {
        LwSciStreamEventType event;

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

            // Indicate packet complete
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketComplete(pool, packetHandle));

            // Producer receives PacketCreate event
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketCreate, event);

            // Retrieve packet handle
            LwSciStreamPacket producerPacket;
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketNewHandleGet(producer,
                                                   &producerPacket));

            // Assign cookie to producer packet handle
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

                // Consumer receives PacketCreate event
                EXPECT_EQ(LwSciError_Success,
                          LwSciStreamBlockEventQuery(consumer[n],
                                                     eventQueryTimeout,
                                                     &event));
                EXPECT_EQ(LwSciStreamEventType_PacketCreate, event);

                // Retrieve packet handle
                LwSciStreamPacket consumerPacket;
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockPacketNewHandleGet(consumer[n],
                                                       &consumerPacket));

                // Assign cookie to consumer packet handle
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
    };

    inline virtual void checkPacketStatus()
    {
        LwSciStreamEventType event;
        uint32_t numPacketComplete = 0U;

        // Pool must receive all PacketStatus events
        while (numPacketComplete < numPackets) {

            if (LwSciError_Success !=
                LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event)) {
                ASSERT_TRUE(false);
                break;
            }
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
    };

    inline void setupComplete()
    {
        LwSciStreamEventType event;
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
        LwSciStreamEventType event;
        LwSciSyncFence fence;

        for (uint32_t i = 0; i < numFrames; ++i) {
            LwSciStreamCookie cookie;

            // Pool sends packet ready event to producer
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

            // Producer get a packet from the pool
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie));
            LwSciStreamPacket handle = prodCPMap[cookie];
            for (uint32_t n = 0U; n < numConsumers; n++) {
                for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                    ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockPacketFenceGet(
                            producer, handle, n, j, &fence));
                    LwSciSyncFenceClear(&fence);
                }
            }

            // Producer inserts a data packet into the stream
            fence = LwSciSyncFenceInitializer;
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockPacketFenceSet(
                        producer, handle, j, &fence));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketPresent(producer, handle));

            for (uint32_t n = 0U; n < numConsumers; n++) {
                // Pool sends packet ready event to consumer
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(
                        consumer[n], eventQueryTimeout, &event));
                EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketAcquire(consumer[n], &cookie));
                handle = (consCPMap[n])[cookie];
                for (uint32_t j = 0U; j < prodSyncCount; j++) {
                    ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockPacketFenceGet(
                            consumer[n], handle, 0U, j, &fence));
                    LwSciSyncFenceClear(&fence);
                }

                // Consumer returns a data packet to the stream
                fence = LwSciSyncFenceInitializer;
                for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                    ASSERT_EQ(LwSciError_Success,
                        LwSciStreamBlockPacketFenceSet(
                            consumer[n], handle, j, &fence));
                }
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketRelease(consumer[n], handle));
            }
        } // End of sending frames
    };

    inline virtual void disconnectStream()
    {
        LwSciStreamEventType event;

        // Delete producer block
        if (producer != 0U) {
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(producer));
            producer = 0U;
        }

        if (pool != 0U) {
            // Pool receives both DisconnectUpstream and DisconnectDownstream events.
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(pool, eventQueryTimeout, &event));
            EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete pool block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(pool));
            pool = 0U;
        }


        if (numConsumers > 1U) {
            if (multicast != 0U) {
                // Multicast receives DisconnectUpstream and DisconnectDownstream events.
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(multicast, eventQueryTimeout, &event));
                EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

                // Delete queue block
                ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(multicast));
                multicast = 0U;
            }
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            if (queue[n] != 0U) {
                // Queue receives DisconnectUpstream and DisconnectDownstream events.
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(queue[n], eventQueryTimeout, &event));
                EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

                // Delete queue block
                ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(queue[n]));
                queue[n] = 0U;
            }

            if (consumer[n] != 0U) {
                // Consumer receives DisconnectUpstream event
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout, &event));
                EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

                // Delete consumer block
                ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(consumer[n]));
                consumer[n] = 0U;
            }
        }
    };
};

#endif // !LWSCISTREAMTEST_H
