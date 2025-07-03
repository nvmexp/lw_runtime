//! \file
//! \brief LwSciStream APIs unit testing - Packet Delete.
//!
//! \copyright
//! Copyright (c) 2020-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"

//==============================================
// Define PacketDelete test suite
//==============================================
class PacketDelete :
    public LwSciStreamTest
{
protected:
    virtual void SetUp()
    {
        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);

        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        prodSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, prodSyncAttrList);

        consSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, consSyncAttrList);
    }

    inline void deletePoolPackets()
    {
        // Iterate over pool cookie map to get packet handles, and delete all packets
        for (auto iter = poolCPMap.begin(); iter != poolCPMap.end(); ++iter) {
            LwSciStreamPacket packet = iter->second;

            ASSERT_EQ(LwSciError_Success, LwSciStreamPoolPacketDelete(pool, packet));
        }
    };

    inline void readDeleteEvents()
    {
        LwSciStreamEventType event;

        // Check the packet delete events on producer side
        for (uint32_t i = 0; i < numPackets; ++i) {
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);
        }

        // Check the packet delete events on consumer sides
        for (uint32_t i = 0; i < numConsumers; ++i) {
            for (uint32_t j = 0; j < numPackets; ++j) {
                ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    consumer[i], EVENT_QUERY_TIMEOUT, &event));
                ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);
            }
        }
    };

   inline virtual void streamingWithPacketDeleteAfterGet()
   {
        LwSciStreamEventType event;
        uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

        // Pool sends packet ready event to producer
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

        LwSciStreamCookie producerCookie;

        // Producer get a packet from the pool
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &producerCookie));

        LwSciStreamPacket handle = poolCPMap[producerCookie];

        // Delete pool packet
        ASSERT_EQ(LwSciError_Success, LwSciStreamPoolPacketDelete(pool, handle));

        handle = prodCPMap[producerCookie];

        // Producer inserts a data packet into the stream
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle));

        LwSciStreamCookie consumerCookie[MAX_CONSUMERS];
        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer is expected to receive packet-ready event
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

            // Consumer acquires a packet
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[n], &consumerCookie[n]));

            handle = (consCPMap[n])[consumerCookie[n]];

            // Delete pool packet
            ASSERT_EQ(LwSciError_StreamBadPacket,
                      LwSciStreamPoolPacketDelete(pool, handle));

            // Consumer returns a data packet to the stream
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketRelease(consumer[n], handle));
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Check the packet delete events are received by the consumer
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], EVENT_QUERY_TIMEOUT, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);

            // Check consumer cookie is valid.
            LwSciStreamCookie eventCookie;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketOldCookieGet(consumer[n], &eventCookie));
            ASSERT_EQ(consumerCookie[n], eventCookie);
        }

        // Check the packet delete events are received by the producer
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);

        // Check producer cookie is valid.
        LwSciStreamCookie eventCookie;
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketOldCookieGet(producer, &eventCookie));
        ASSERT_EQ(producerCookie, eventCookie);
   }

   inline virtual void streamingWithPacketDeleteAfterPresent()
   {
        LwSciStreamEventType event;
        uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

        // Pool sends packet ready event to producer
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

        LwSciStreamCookie producerCookie;

        // Producer get a packet from the pool
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &producerCookie));

        LwSciStreamPacket handle = prodCPMap[producerCookie];

        // Producer inserts a data packet into the stream
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle));

        handle = poolCPMap[producerCookie];

        // Delete pool packet
        ASSERT_EQ(LwSciError_Success, LwSciStreamPoolPacketDelete(pool, handle));

        LwSciStreamCookie consumerCookie[MAX_CONSUMERS];
        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer is expected to receive packet-ready event
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

            // Consumer acquires a packet
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[n], &consumerCookie[n]));

            handle = (consCPMap[n])[consumerCookie[n]];

            // Delete pool packet
            ASSERT_EQ(LwSciError_StreamBadPacket,
                      LwSciStreamPoolPacketDelete(pool, handle));

            // Consumer returns a data packet to the stream
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketRelease(consumer[n], handle));
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Check the packet delete events are received by the consumer
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], EVENT_QUERY_TIMEOUT, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);

            // Check consumer cookie is valid.
            LwSciStreamCookie eventCookie;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketOldCookieGet(consumer[n], &eventCookie));
            ASSERT_EQ(consumerCookie[n], eventCookie);
        }

        // Check the packet delete events are received by the producer
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);

        // Check producer cookie is valid.
        LwSciStreamCookie eventCookie;
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketOldCookieGet(producer, &eventCookie));
        ASSERT_EQ(producerCookie, eventCookie);
   }

   inline virtual void streamingWithPacketDeleteAfterAcquire()
   {
        LwSciStreamEventType event;
        uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

        // Pool sends packet ready event to producer
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

        LwSciStreamCookie producerCookie;
        LwSciStreamCookie consumerCookie[MAX_CONSUMERS];

        // Producer get a packet from the pool
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &producerCookie));

        LwSciStreamPacket handle = prodCPMap[producerCookie];

        // Producer inserts a data packet into the stream
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle));

        // Consumer is expected to receive packet-ready event
        ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[0], eventQueryTimeout, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

        // Consumer acquires a packet
        ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[0], &consumerCookie[0]));

        handle = poolCPMap[consumerCookie[0]];

        // Delete pool packet
        ASSERT_EQ(LwSciError_Success, LwSciStreamPoolPacketDelete(pool, handle));

        // Consumer returns a data packet to the stream
        ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketRelease(consumer[0], handle));

        for (uint32_t n = 1U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

            // Consumer acquires a packet
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[n], &consumerCookie[n]));

            handle = (consCPMap[n])[consumerCookie[n]];

            // Delete pool packet
            ASSERT_EQ(LwSciError_StreamBadPacket,
                      LwSciStreamPoolPacketDelete(pool, handle));

            // Consumer returns a data packet to the stream
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketRelease(consumer[n], handle));
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Check the packet delete events are received by the consumer
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[n], EVENT_QUERY_TIMEOUT, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);

            // Check consumer cookie is valid.
            LwSciStreamCookie eventCookie;
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketOldCookieGet(consumer[n], &eventCookie));
            ASSERT_EQ(consumerCookie[n], eventCookie);
        }

        // Check the packet delete events are received by the producer
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        ASSERT_EQ(LwSciStreamEventType_PacketDelete, event);

        // Check producer cookie is valid.
        LwSciStreamCookie eventCookie;
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketOldCookieGet(producer, &eventCookie));
        ASSERT_EQ(producerCookie, eventCookie);
   }

    inline void postDeleteStreaming()
    {
        LwSciStreamEventType event;

        LwSciStreamCookie cookie;

        // Producer try to get a packet from pool
        ASSERT_EQ(LwSciError_NoStreamPacket, LwSciStreamProducerPacketGet(
                    producer, &cookie));

        // Consumer try to get a packet from pool
        for (uint32_t i = 0; i < numConsumers; ++i) {
            ASSERT_EQ(LwSciError_NoStreamPacket, LwSciStreamConsumerPacketAcquire(
                       consumer[i], &cookie));
        }
    }
};

//==============================================================================
// Test Case 1 : (PacketDelete, mailbox queue)
//==============================================================================

TEST_F(PacketDelete, PacketDelete_Mailbox)
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

    // Delete pool packets
    deletePoolPackets();

    // PacketDelete events
    readDeleteEvents();

    // Try to get packets from pool
    postDeleteStreaming();

    // Cleanup
    disconnectStream();
}

//==============================================================================
// Test Case 2 : (PacketDelete, fifo queue)
//==============================================================================

TEST_F(PacketDelete, PacketDelete_Fifo)
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

    // Delete pool packets
    deletePoolPackets();

    // PacketDelete events
    readDeleteEvents();

    // Try to get packets from pool
    postDeleteStreaming();

    // Cleanup
    disconnectStream();
}

//==============================================================================
// Test Case 3 : (Packet delete during streaming scenario 1)
//==============================================================================

TEST_F(PacketDelete, Scenario1Mailbox)
{
    // Create a mailbox stream with 2 packets
    createBlocks(QueueType::Mailbox, 1, 2);
    connectStream();

    // Setup packet attributes for two packets
    packetAttrSetup();

    // Create packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterGet();
}

//==============================================================================
// Test Case 4 : (Packet delete during streaming scenario 1)
//==============================================================================

TEST_F(PacketDelete, Scenario1Fifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo, 1, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterGet();
}

//=================================================================================
// Test Case 5 : (Packet delete during streaming scenario 1)
//=================================================================================

TEST_F(PacketDelete, Scenario1MulticastMailbox)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, 2, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterGet();
}

//==============================================================================
// Test Case 6 : (Packet delete during streaming scenario 1)
//==============================================================================

TEST_F(PacketDelete, Scenario1MulticastFifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo, 2, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterGet();
}

//==============================================================================
// Test Case 7 : (Packet delete during streaming scenario 2)
//==============================================================================

TEST_F(PacketDelete, Scenario2Mailbox)
{
    // Create a mailbox stream with 2 packets
    createBlocks(QueueType::Mailbox, 1, 2);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterPresent();
}

//==============================================================================
// Test Case 8 : (Packet delete during streaming scenario 2)
//==============================================================================

TEST_F(PacketDelete, Scenario2Fifo)
{
    // Create a mailbox stream with 2 packets
    createBlocks(QueueType::Fifo, 1);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterPresent();
}

//=================================================================================
// Test Case 9 : (Packet delete during streaming scenario 2)
//=================================================================================

TEST_F(PacketDelete, Scenario2MulticastMailbox)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, 2, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterPresent();
}

//==============================================================================
// Test Case 10 : (Packet delete during streaming scenario 2)
//==============================================================================

TEST_F(PacketDelete, Scenario2MulticastFifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo, 2, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterPresent();
}

//==============================================================================
// Test Case 11 : (Packet delete during streaming scenario 3)
//==============================================================================

TEST_F(PacketDelete, Scenario3Mailbox)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, 1, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterAcquire();
}

//==============================================================================
// Test Case 12 : (Packet delete during streaming scenario 3)
//==============================================================================

TEST_F(PacketDelete, Scenario3Fifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo, 1, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange two sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterAcquire();
}

//==================================================================================
// Test Case 13 : (Packet delete during streaming scenario 3)
//==================================================================================

TEST_F(PacketDelete, Scenario3MulticastMailbox)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, 2, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterAcquire();
}

//==============================================================================
// Test Case 14 : (Packet delete during streaming scenario 3)
//==============================================================================

TEST_F(PacketDelete, Scenario3MulticastFifo)
{
    // Create a mailbox stream
    createBlocks(QueueType::Fifo, 2, 5);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    streamingWithPacketDeleteAfterAcquire();
}
