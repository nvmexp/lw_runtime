//! \file
//! \brief LwSciStream APIs unit testing - Buffer Setup.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"
#include <vector>
#include <future>

//==============================================
// Define BufferSetup test suite
//==============================================
class BufferSetup :
    public LwSciStreamTest
{
protected:
    uint32_t prodElementCount = NUM_PACKET_ELEMENTS;
    uint32_t consElementCount = NUM_PACKET_ELEMENTS;
    // Fake number of reconciled packet elements
    uint32_t reconciledElementCount = NUM_PACKET_ELEMENTS;

    virtual void SetUp()
    {
        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);
    };

    inline void queryMaxNumElements()
    {
        int32_t value;
        ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
        ASSERT_TRUE(prodElementCount <= value);
        ASSERT_TRUE(consElementCount <= value);
    };

    inline void prodSendElements()
    {
        // Set producer packet requirements
        for (uint32_t i = 0U; i < prodElementCount; i++) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementAttrSet(producer, i, rawBufAttrList));
        }

        // Indicate producer element specification is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      producer, LwSciStreamSetup_ElementExport, true));
    };

    inline void consSendElements()
    {
        // Set consumer packet requirements
        for (uint32_t i = 0U; i < consElementCount; i++) {
            for (uint32_t n = 0U; n < numConsumers; n++) {
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamBlockElementAttrSet(consumer[n], i,
                                                   rawBufAttrList));
            }
        }

        // Indicate consumer element specification is done
        for (uint32_t n = 0U; n < numConsumers; n++) {
            ASSERT_EQ(LwSciError_Success,
                      LwSciStreamBlockSetupStatusSet(
                          consumer[n], LwSciStreamSetup_ElementExport, true));
        }
    };

    inline void poolRecvElements()
    {
        // Pool receives event
        LwSciStreamEventType event;
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT,
                                             &event));
        ASSERT_EQ(LwSciStreamEventType_Elements, event);

        // Verify counts
        uint32_t inCount;
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      pool, LwSciStreamBlockType_Producer, &inCount));
        ASSERT_EQ(prodElementCount, inCount);
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementCountGet(
                      pool, LwSciStreamBlockType_Consumer, &inCount));
        ASSERT_EQ(consElementCount, inCount);

        // Indicate pool element import is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementImport, true));

    };

    // Pool sends reconciled attributes to producer and consumer
    inline void poolSendReconciledElements()
    {
        // Pool sets packet requirements
        for (uint32_t i = 0U; i < reconciledElementCount; i++) {
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockElementAttrSet(pool, i, rawBufAttrList));
        }

        // Indicate pool element specification is done
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockSetupStatusSet(
                      pool, LwSciStreamSetup_ElementExport, true));

    };
};

//==========================================================================
// Test Case 1 (Packet Requirement):
// Set up producer's and consumer's packet attr object for a mailbox stream
//==========================================================================

TEST_F(BufferSetup, CreatePacketAttr_Mailbox)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();
}

//==========================================================================
// Test Case 1 with multicast:
//==========================================================================
TEST_F(BufferSetup, CreatePacketAttr_Mailbox_multicast)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();
}

//==========================================================================
// Test Case 2 (Packet Requirement):
// Set up producer's and consumer's packet attr object for a fifo stream
//==========================================================================

TEST_F(BufferSetup, CreatePacketAttr_Fifo)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();
}

//==========================================================================
// Test Case 2 with multicast:
//==========================================================================

TEST_F(BufferSetup, CreatePacketAttr_Fifo_multicast)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();
}

//==========================================================================
// Test Case 3 (Packet Requirement failure):
// Producer fails to send packet attr object before a complete downstream
// connection
//==========================================================================

TEST_F(BufferSetup, CreateProducerPacketAttr_Failure1)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Failure
    ASSERT_EQ(LwSciError_StreamNotConnected,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));

    connectStream();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));
}

//==========================================================================
// Test Case 3 with multicast:
//==========================================================================

TEST_F(BufferSetup, CreateProducerPacketAttr_multicast_Failure1)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Failure
    ASSERT_EQ(LwSciError_StreamNotConnected,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));

    connectStream();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));
}

//==========================================================================
// Test Case 4 (Packet Requirement failure):
// Producer fails to send packet attr object before a complete downstream
// connection
//==========================================================================

TEST_F(BufferSetup, CreateProducerPacketAttr_Failure2)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo);

    // Failure
    ASSERT_EQ(LwSciError_StreamNotConnected,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));

    connectStream();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));
}

//==========================================================================
// Test Case 4 with mulitcast:
//==========================================================================

TEST_F(BufferSetup, CreateProducerPacketAttr_multicast_Failure2)
{
    // Create a fifo stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);

    // Failure
    ASSERT_EQ(LwSciError_StreamNotConnected,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));

    connectStream();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(producer, 0, rawBufAttrList));
}

//==========================================================================
// Test Case 5 (Packet Requirement failure):
// Consumer fails to send packet attr object before a complete upstream
// connection
//==========================================================================

TEST_F(BufferSetup, CreateConsumerPacketAttr_Failure1)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Failure
    ASSERT_EQ(LwSciError_StreamNotConnected,
              LwSciStreamBlockElementAttrSet(consumer[0], 0, rawBufAttrList));

    connectStream();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(consumer[0], 0, rawBufAttrList));
}

//==========================================================================
// Test Case 5 with multicast:
//==========================================================================

TEST_F(BufferSetup, CreateConsumerPacketAttr_Multicast_Failure1)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Failure
    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_StreamNotConnected,
                  LwSciStreamBlockElementAttrSet(consumer[n], 0,
                                                 rawBufAttrList));
    }

    connectStream();

    // Success
    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementAttrSet(consumer[n], 0,
                                                 rawBufAttrList));
    }
}

//==========================================================================
// Test Case 6 (Packet Requirement failure):
// Consumer fails to send packet attr object before a complete upstream
// connection
//==========================================================================

TEST_F(BufferSetup, CreateConsumerPacketAttr_Failure2)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);

    // Failure
    ASSERT_EQ(LwSciError_StreamNotConnected,
              LwSciStreamBlockElementAttrSet(consumer[0], 0, rawBufAttrList));

    connectStream();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(consumer[0], 0, rawBufAttrList));
}

//==========================================================================
// Test Case 6 with multicast:
//==========================================================================

TEST_F(BufferSetup, CreateConsumerPacketAttr_Multicast_Failure2)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);

    // Failure
    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_StreamNotConnected,
                  LwSciStreamBlockElementAttrSet(consumer[n], 0,
                                                 rawBufAttrList));
    }

    connectStream();

    // Success
    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
                  LwSciStreamBlockElementAttrSet(consumer[n], 0,
                                                 rawBufAttrList));
    }
}

//==========================================================================
// Test Case 7 (Packet Determination):
// Pool sends producer and consumer packet requirements to each other
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Mailbox)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();
    poolSendReconciledElements();
}

//==========================================================================
// Test Case 7 with multicast:
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Mailbox_Multicast)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();
    poolSendReconciledElements();
}

//==========================================================================
// Test Case 8 (Packet Determination):
// Pool sends producer and consumer packet requirements to each other
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Fifo)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();
    poolSendReconciledElements();
}

//==========================================================================
// Test Case 8 with multicast:
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Fifo_Multicast)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();
    poolSendReconciledElements();
}

//==========================================================================
// Test Case 9 (Packet Determination failure):
// Failure before PACKET_ATTR_* and exchange of packet requirements
// Queue type is irrelevant so do for mailbox.
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Failure1)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    // Failure
    ASSERT_EQ(LwSciError_NotYetAvailable,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));
}

//==========================================================================
// Test Case 9 with multicast:
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Multicast_Failure1)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();

    // Failure
    ASSERT_EQ(LwSciError_NotYetAvailable,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));
}

//==========================================================================
// Test Case 10 (Packet Determination failure):
// Failure before PACKET_ATTR_* and exchange of packet requirements
// Queue type is irrelevant so do for mailbox.
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Failure2)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Failure
    ASSERT_EQ(LwSciError_NotYetAvailable,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));
}


//==========================================================================
// Test Case 10 with multicast:
//==========================================================================

TEST_F(BufferSetup, ExchangePacketRequiement_Multicast_Failure2)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    // Failure
    ASSERT_EQ(LwSciError_NotYetAvailable,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();

    // Success
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList));
}

//==========================================================================
// Test Case 11 (Packet Preparation):
// Producer and consumer receive the layout make sure it is suitable. They
// receive PACKET_ATTR event at their ends.
//==========================================================================

TEST_F(BufferSetup, SetupPacketLayout)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();
}

//==========================================================================
// Test Case 11 with multicast:
//==========================================================================

TEST_F(BufferSetup, SetupPacketLayout_Multicast)
{
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();
}

//==========================================================================
// Test Case 12 (Buffer Registration):
// Packets are added to the pool, and sent to the producer and consumer,
// which map them for use, and report back whether they were successful.
//==========================================================================

TEST_F(BufferSetup, PacketCreateAndAccept)
{
    LwSciStreamEventType event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    createPacket();
    checkPacketStatus();
}

//==========================================================================
// Test Case 12 with multicast:
//==========================================================================

TEST_F(BufferSetup, PacketCreateAndAccept_Multicast)
{
    LwSciStreamEventType event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);
    connectStream();

    packetAttrSetup();

    createPacket();
    checkPacketStatus();
}

//==========================================================================
// Ensure conlwrrency of LwSciStreamBlockPacketAttr()
//==========================================================================
TEST_F(BufferSetup, ConlwrrentLwSciStreamBlockPacketAttr)
{
    int successfullOps = 0;
    const int maxIterations = 10;

    std::vector<std::future<LwSciError>> tasks;

    // Create a FIFO stream.
    createBlocks(QueueType::Fifo, NUM_CONSUMERS);
    connectStream();

    queryMaxNumElements();

    prodSendElements();
    consSendElements();

    poolRecvElements();

    for (int i = 0; i < maxIterations; i++) {
        auto task = std::async(
            std::launch::async,
            [=]() -> LwSciError {
                return LwSciStreamBlockElementAttrSet(pool, 0, rawBufAttrList);
            });
        tasks.push_back(std::move(task));
    }

    for (auto& task : tasks) {
        LwSciError err = task.get();
        if (err == LwSciError_Success) {
            successfullOps++;
        } else {
            // Make sure returned error is correct
            ASSERT_TRUE((LwSciError_Busy == err) ||
                        (LwSciError_AlreadyInUse == err));
        }
    }

    ASSERT_EQ(1, successfullOps);
}
