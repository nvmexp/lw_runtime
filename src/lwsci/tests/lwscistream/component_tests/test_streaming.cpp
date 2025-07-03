//! \file
//! \brief LwSciStream APIs unit testing - Packet Streaming.
//!
//! \copyright
//! Copyright (c) 2019-2020 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"

//==============================================
// Define PacketStreaming test suite
//==============================================

class PacketStreaming :
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
    };
};

//==============================================================================
// Test Case 1 : (send and receive frame, fifo mode)
// Producer beings sending payloads to the consumer, and the consumer processes
// them and sends back their packets for reuse. In this simple example, one
// payload being processed at a time.
//==============================================================================

TEST_F(PacketStreaming, SendAndReceiveFrame_Fifo)
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

TEST_F(PacketStreaming, SendAndReceiveFrame_Mailbox)
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

TEST_F(PacketStreaming, SendAndReceiveFrame_Fifo_Multicast)
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

TEST_F(PacketStreaming, SendAndReceiveFrame_Mailbox_Multicast)
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
