//! \file
//! \brief LwSciStream APIs unit testing - Stream Packet.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"

constexpr uint32_t PAYLOAD_COUNT = 6U;

//==============================================
// Define StreamPacket test suite
//==============================================

class StreamPacket :
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

    inline virtual void streaming(uint32_t numFrames = NUM_FRAMES)
    {
        LwSciStreamEventType event;
        uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

        LwSciSyncFence *prefences =
            static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

        for (uint32_t i = 0; i < numFrames; ++i) {
            LwSciStreamCookie cookie;

            // Pool sends packet ready event to producer
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, eventQueryTimeout, &event));
            ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

            // Producer get a packet from the pool
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie));
            LwSciStreamPacket handle = prodCPMap[cookie];

            // Producer try to get a packet again from the pool
            ASSERT_EQ(LwSciError_NoStreamPacket,
                LwSciStreamProducerPacketGet(producer, &cookie));

            // Producer inserts a data packet into the stream
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                prefences[j] = LwSciSyncFenceInitializer;
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockPacketFenceSet(
                        producer, handle, j, &prefences[j]));
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketPresent(producer, handle));

            // Producer try to insert data packet into the stream again
            ASSERT_EQ(LwSciError_StreamPacketInaccessible,
                LwSciStreamProducerPacketPresent(producer, handle));

            for (uint32_t n = 0U; n < numConsumers; n++) {
                // Pool sends packet ready event to consumer
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(
                        consumer[n], eventQueryTimeout, &event));
                ASSERT_EQ(LwSciStreamEventType_PacketReady, event);

                LwSciStreamCookie consumerCookie;

                // Consumer gets a packet from the queue
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketAcquire(consumer[n],
                                                     &consumerCookie));

                // Check consumer cookie is valid.
                ASSERT_EQ(consumerCookie, cookie);
                handle = (consCPMap[n])[cookie];

                // Check consumer fences are valid
                for (uint32_t j = 0U; j < prodSyncCount; j++) {
                    for (uint32_t i = 0U; i < PAYLOAD_COUNT; i++) {
                        LwSciSyncFence fence;
                        ASSERT_EQ(LwSciError_Success,
                            LwSciStreamBlockPacketFenceGet(
                                consumer[n], handle, 0U, j, &fence));
                        ASSERT_EQ(prefences[j].payload[i], fence.payload[i]);
                    }
                }

                // Consumer returns a data packet to the stream
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketRelease(consumer[n], handle));

                // Consumer returns a data packet to the stream again
                ASSERT_EQ(LwSciError_StreamPacketInaccessible,
                    LwSciStreamConsumerPacketRelease(consumer[n], handle));

                // Consumer tries to acquire a packet
                ASSERT_EQ(LwSciError_NoStreamPacket,
                    LwSciStreamConsumerPacketAcquire(consumer[n], &cookie));
            }

        } // End of sending frames
    free(prefences);
   };

    inline void prodSendPacket(void)
    {
        LwSciStreamEventType event;
        LwSciStreamCookie cookie;

        // Producer get a packet from the pool
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &cookie));

        // Producer inserts a data packet into the stream
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, prodCPMap[cookie]));
    };
};

//==============================================================================
// Test Case 1 : Create a packet for a mailbox stream with one consumer, when
// trying to insert a data packet or when trying to release and acquire packet
// again to the stream should result in failure.
//==============================================================================

TEST_F(StreamPacket, StreamingOperations)
{
    // Create a mailbox stream with one consumer and a pool having capacity of one packet
    createBlocks(QueueType::Mailbox, 1, 1);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();

    // Receive status and end packet phase
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();
}

//==============================================================================
// Test Case 2 : Create a packet for a mailbox stream with two consumers, when
// trying to insert a data packet or when trying to release and acquire packet
// again to the stream should result in failure.
//==============================================================================

TEST_F(StreamPacket, StreamingOperationsMulticast)
{
    // Create a mailbox stream with two consumers and a pool having capacity of one packet
    createBlocks(QueueType::Mailbox, 2, 1);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();

    // Receive status and end packet phase
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();
    streaming();
}

//==============================================================================
// Test Case 3 : Creates a FIFO stream with 2 packets and 3 consumers. One
// consumer gets disconnected from the stream without releasing the packets but
// streaming continues with other consumers without any problem.
//==============================================================================

TEST_F(StreamPacket, FIFOMulticastDisconnectStreaming)
{
    LwSciStreamEventType event;
    LwSciStreamCookie packet1cookie[3];
    LwSciStreamCookie packet2cookie[3];

    // Create a FIFO stream with 3 consumers and 2 packets
    createBlocks(QueueType::Fifo, 3, 2);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();

    // Receive status and end packet phase
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    // Streaming

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    // Producer sends the first packet.
    prodSendPacket();

    // Consumers acquire the first packet.
    for (uint32_t i = 0U; i < numConsumers; i++) {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerPacketAcquire(consumer[i], &packet1cookie[i]));
    }

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    // Producer sends the second packet.
    prodSendPacket();

    // Consumers acquire the second packet.
    for (uint32_t i = 0U; i < numConsumers; i++) {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerPacketAcquire(consumer[i], &packet2cookie[i]));
    }

    // Producer waits for packet, since consumers not released the packet,
    // timeout error should occur.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Consumer 0 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[0], (consCPMap[0])[packet1cookie[0]]));

    // Since packet is not released by consumer1 and consumer2, producer should still
    // get timeout error.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Consumer 1 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[1], (consCPMap[1])[packet1cookie[1]]));

    // Since packet is not released by consumer2, producer should still
    // get timeout error.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Consumer2 calls disconnect without releasing the packets
    // Delete consumer2 block
    ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(consumer[2]));
    consumer[2] = 0U;

    // Consumer-2 disconnected from the stream, all the packets held by that
    // consumer will be automatically released. Hence, producer receives the
    // packet as other consumers(consumer0 and consumer1) already released the packet.
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    // Consumer 0 releases the second packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[0], (consCPMap[0])[packet2cookie[0]]));

    // Since packet is not yet released by consumer1, producer should still
    // get timeout error.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Consumer 1 releases the second packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[1], (consCPMap[1])[packet2cookie[1]]));

    // packet released by all consumers, producer should get the packet for reuse.
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    // Checking the stream behavior post consumer disconnect
    // Producer sends the first packet.
    prodSendPacket();

    // Consumers acquire the first packet.
    for (uint32_t i = 0U; i < numConsumers; i++) {
        if (consumer[i] != 0U) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[i], &packet1cookie[i]));
        }
    }

    // Consumer 0 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[0], (consCPMap[0])[packet1cookie[0]]));

    // Since packet is not yet released by consumer1, producer should still
    // get timeout error.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Consumer 1 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[1], (consCPMap[1])[packet1cookie[1]]));

    // packet released by all consumers, producer should get the packet for reuse.
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);
}

//==============================================================================
// Test Case 4 : Creates a mailbox stream with 1 packet and 2 consumers. One
// consumer gets disconnected from the stream without releasing the packets but
// streaming continues with other consumers without any problem.
//==============================================================================
TEST_F(StreamPacket, MailboxMulticastDisconnectStreaming)
{
    LwSciStreamEventType event;
    LwSciStreamCookie packet1cookie[3];

    // Create a mailbox stream with 2 consumers and 1 packets
    createBlocks(QueueType::Mailbox, 2, 1);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();

    // Receive status and end packet phase
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    setupComplete();

    // Streaming

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    // Producer sends the first packet.
    prodSendPacket();

    // Consumers acquire the first packet.
    for (uint32_t i = 0U; i < numConsumers; i++) {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerPacketAcquire(consumer[i], &packet1cookie[i]));
    }

    // Producer waits for packet, since consumers not released the packet,
    // timeout error should occur.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Consumer 0 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[0], (consCPMap[0])[packet1cookie[0]]));

    // Since packet is not released by consumer1, producer should still
    // get timeout error.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Consumer1 calls disconnect without releasing the packets
    // Delete consumer1 block
    ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(consumer[1]));
    consumer[1] = 0U;

    // Consumer-1 disconnected from the stream, all the packets held by that
    // consumer will be automatically released. Hence, producer receives the
    // packet as other consumers(consumer0) already released the packet.
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    // Checking the stream behavior post consumer disconnect
    // Producer sends the first packet.
    prodSendPacket();

    // Consumers acquire the first packet.
    for (uint32_t i = 0U; i < numConsumers; i++) {
        if (consumer[i] != 0U) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[i], &packet1cookie[i]));
        }
    }

    // Consumer 0 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[0], (consCPMap[0])[packet1cookie[0]]));

    // packet released by all consumers, producer should get the packet for reuse.
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);
}
