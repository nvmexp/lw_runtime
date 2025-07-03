//! \file
//! \brief LwSciStream APIs unit testing - Packet Streaming.
//!
//! \copyright
//! Copyright (c) 2020 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"

//==============================================
// Define PacketLimiterStreaming test suite
//==============================================

class PacketLimiterStreaming :
    public LwSciStreamTest
{
protected:
    LwSciStreamBlock limiter = 0U;

    virtual ~PacketLimiterStreaming() {
        if (limiter != 0U) {
            LwSciStreamBlockDelete(limiter);
        }
    };

    // Stream setup
    void streamSetup(
        QueueType queueType,
        uint32_t numPackets,
        uint32_t limitedNumPackets,
        uint32_t numCons,
        uint32_t numSyncs)
    {
        // Create a FIFO stream
        createBlocks(queueType, numCons, numPackets);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamLimiterCreate(limitedNumPackets, &limiter));
        ASSERT_NE(0, limiter);

        // Connect stream
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockConnect(producer, multicast));

        // Consumer 0 doesn't connect with limiter block.
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockConnect(multicast, consumer[0]));

        // Consumer 1 connects with limiter block.
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockConnect(multicast, limiter));
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockConnect(limiter, consumer[1]));


        // Check Connect* events
        LwSciStreamEventType event;

        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Connected, event);

        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Connected, event);

        for (uint32_t i = 0U; i < numConsumers; i++) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);

            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(queue[i], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);
        }

        if (numConsumers > 1U) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(multicast, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Connected, event);
        }

        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(limiter, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Connected, event);

        // Setup packet attr
        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);
        packetAttrSetup();

        // Create packets
        createPacket();
        checkPacketStatus();

        // Create and exchange sync objects
        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
        if (numSyncs == 0U) {
            prodSynchronousOnly = true;
            prodSyncAttrList = nullptr;
            consSynchronousOnly = true;
            consSyncAttrList = nullptr;
        } else {
            prodSynchronousOnly = false;
            cpuWaiterAttrList(syncModule, prodSyncAttrList);
            consSynchronousOnly = false;
            cpuWaiterAttrList(syncModule, consSyncAttrList);
        }
        createSync(numSyncs);

        setupComplete();
    };

    inline void prodSendPacket(void)
    {
        LwSciStreamEventType event;
        LwSciStreamCookie cookie;

        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        // Producer get a packet from the pool
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &cookie));

        // Producer inserts a data packet into the stream
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, prodCPMap[cookie]));
    };
};


TEST_F(PacketLimiterStreaming, Fifo_Multicast_Limiter)
{
    uint32_t numPackets = 3U;
    uint32_t limitedNumPackets = 1U;
    uint32_t numCons = 2U;
    uint32_t numSyncs = 0U;

    streamSetup(QueueType::Fifo,
                numPackets,
                limitedNumPackets,
                numCons,
                numSyncs);

    // Streaming
    LwSciStreamEventType event;
    LwSciStreamCookie cookie[2];

    // Producer sends the first packet.
    prodSendPacket();

    // Consumers acquire the first packet.
    for (uint32_t i = 0U; i < numCons; i++) {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerPacketAcquire(consumer[i], &cookie[i]));
    }

    // Producer sends the second packet.
    prodSendPacket();

    // Consumer 0 acquires the second packet.
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie[0]));

    // Consumer 1 could not see the new packet as the number of
    // downstream packets reaches the limit.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(consumer[1], EVENT_QUERY_TIMEOUT, &event));

    // Consumer 1 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[1], (consCPMap[1])[cookie[1]]));

    // Producer sends the third packet.
    prodSendPacket();

    // Consumers acquire the third packet.
    for (uint32_t i = 0U; i < numCons; i++) {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerPacketAcquire(consumer[i], &cookie[i]));
    }
}

TEST_F(PacketLimiterStreaming, Mailbox_Multicast_Limiter)
{
    uint32_t numPackets = 3U;
    uint32_t limitedNumPackets = 1U;
    uint32_t numCons = 2U;
    uint32_t numSyncs = 0U;

    streamSetup(QueueType::Mailbox,
                numPackets,
                limitedNumPackets,
                numCons,
                numSyncs);

    // Streaming
    LwSciStreamEventType event;
    LwSciStreamCookie cookie[2];

    // Producer sends the first packet.
    prodSendPacket();

    // Consumers acquire the first packet.
    for (uint32_t i = 0U; i < numCons; i++) {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerPacketAcquire(consumer[i], &cookie[i]));
    }

    // Producer sends the second packet.
    prodSendPacket();

    // Consumer 0 acquires the second packet.
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie[0]));

    // Consumer 1 could not see the new packet as the number of
    // downstream packets reaches the limit.
    EXPECT_EQ(LwSciError_Timeout,
        LwSciStreamBlockEventQuery(consumer[1], EVENT_QUERY_TIMEOUT, &event));

    // Consumer 1 releases the first packet.
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketRelease(
            consumer[1], (consCPMap[1])[cookie[1]]));

    // Producer sends the third packet.
    prodSendPacket();

    // Consumers acquire the third packet.
    for (uint32_t i = 0U; i < numCons; i++) {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(consumer[i], EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event);

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamConsumerPacketAcquire(consumer[i], &cookie[i]));
    }
}
