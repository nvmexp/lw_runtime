//! \file
//! \brief LwSciStream APIs unit testing - Stream Shutdown.
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
// Define Shutdown test suite
//==============================================
class Shutdown :
    public LwSciStreamTest
{
protected:
    virtual void SetUp()
    {
        //TODO: use "threadsafe" setting. "fast" setting spews warning
        // when running death tests.
        // (Lwrrently, with threadsafe setting, on safety QNX gtest fails
        //  to intercept the abort() call, as a result the process will abort
        //  in death tests).
        ::testing::FLAGS_gtest_death_test_style = "fast";

        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);

        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        prodSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, prodSyncAttrList);

        consSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, consSyncAttrList);
    };


    inline void disconnectStream2()
    {
        LwSciStreamEventType event;

        // Delete producer block
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(producer));

        // Pool receives Disconnected event.
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

        // Delete pool block
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(pool));

        if (numConsumers > 1U) {
            // Multicast receives Disconnected event.
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(multicast, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete multicast block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(multicast));
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Queue receives Disconnected event.
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(queue[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete queue block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(queue[n]));

            // Consumer receives Disconnected event
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Disconnected, event);

            // Delete consumer block
            ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(consumer[n]));
        }
    }

    // Plain stream reconnect without event verification
    inline void reconnectWithDeletedBlocks()
    {
        // Since blocks are deleted BlockConnect will return BadParameter
        if (numConsumers == 1U) {
            CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
                 LwSciStreamBlockConnect(producer, consumer[0]));
        } else {
            CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
                 LwSciStreamBlockConnect(producer, multicast));
            for (uint32_t n = 0U; n < numConsumers; n++) {
                CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
                    LwSciStreamBlockConnect(multicast, consumer[n]));
            }
        }
    }

    inline void ilwalidateHandles()
    {
        producer = 0U;
        pool = 0U;

        if (numConsumers > 1U) {
            multicast = 0U;
        }

        for (uint32_t n = 0U; n < numConsumers; n++) {
            queue[n] = 0U;
            consumer[n] = 0U;
        }
    }
};

//==============================================================================
// Test Case 1 : (shutdown, mailbox queue)
// After all payloads have been processed, the stream must be torn down. When a
// block is destroyed, messages will be sent up and down stream to inform the
// other blocks of the disconnect, and no further messages will be sent to or
// received from that block.
//==============================================================================

TEST_F(Shutdown, Shutdown_Mailbox)
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

    // Exchange frames
    setupComplete();
    streaming();

    disconnectStream();
}

//==============================================================================
// Test Case 2 : (shutdown, fifo queue)
// After all payloads have been processed, the stream must be torn down. When a
// block is destroyed, messages will be sent up and down stream to inform the
// other blocks of the disconnect, and no further messages will be sent to or
// received from that block.
//==============================================================================

TEST_F(Shutdown, Shutdown_Fifo)
{
    // Create a fifo stream
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

    // Exchange frames
    setupComplete();
    streaming();

    disconnectStream();
}

//==============================================================================
// Test Case 3 : (shutdown, mailbox queue, multiple consumers)
// After all payloads have been processed, the stream must be torn down. When a
// block is destroyed, messages will be sent up and down stream to inform the
// other blocks of the disconnect, and no further messages will be sent to or
// received from that block.
//==============================================================================

TEST_F(Shutdown, Shutdown_Mailbox_Multicast)
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

    // Exchange frames
    setupComplete();
    streaming();

    disconnectStream();
}

//==============================================================================
// Test Case 4 : (shutdown, fifo queue, multiple consumers)
// After all payloads have been processed, the stream must be torn down. When a
// block is destroyed, messages will be sent up and down stream to inform the
// other blocks of the disconnect, and no further messages will be sent to or
// received from that block.
//==============================================================================

TEST_F(Shutdown, Shutdown_Fifo_Multicast)
{
    // Create a fifo stream
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

    // Exchange frames
    setupComplete();
    streaming();

    disconnectStream();
}

//==============================================================================
// Test Case 5 : (delete block)
// After block delete, the block's handle may no longer be used for any function
// calls.
//==============================================================================

TEST_F(Shutdown, BlockDelete)
{
    // Create a fifo stream
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

    // Exchange frames
    setupComplete();
    streaming();

    // Cleanup stream but don't ilwalidate handles
    disconnectStream2();

    // Try to reuse the invalid handles by creating a new stream
    reconnectWithDeletedBlocks();

    // Ilwalidate handles to avoid double free in global destructor
    // Since blocks are already deleted, just ilwlidate the handles.
    ilwalidateHandles();
}
