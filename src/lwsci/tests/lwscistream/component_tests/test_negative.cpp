//! \file
//! \brief LwSciStream APIs unit testing - Negative.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"

static const uintptr_t ilwalidBlock = UINTPTR_MAX;
static const uintptr_t nullBlock = 0U;
static const uintptr_t ilwalidQueryAttrib = 100;

//==========================================================================
// Define Negative test suite.
//==========================================================================
class Negative :
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
    };
};

//==============================================================================
// Test Case 1: Block create and block delete with incorrect parameters
//==============================================================================
TEST_F(Negative, CreateAndDelete)
{
    ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
    // Call block create APIs with nullptr
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamProducerCreate(0U, nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamConsumerCreate(0U, nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamStaticPoolCreate(1, nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamMailboxQueueCreate(nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamFifoQueueCreate(nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamMulticastCreate(2U, nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamPresentSyncCreate(syncModule, nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamReturnSyncCreate(syncModule, nullptr));

    // Call block delete with nullptr
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockDelete(ilwalidBlock));
}

//==============================================================================
// Test Case 2: Invalid combinations of block connect
//==============================================================================
TEST_F(Negative, BlockConnect)
{
    // Create blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // {null, invalid} block <--> {null, invalid} block
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(ilwalidBlock, ilwalidBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(nullBlock, nullBlock));

    // {null, invalid} <--> producer
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(ilwalidBlock, producer));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(nullBlock, producer));

    // consumer <---> {null, invalid} block
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(consumer[0], ilwalidBlock));

    // Connect same blocks
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockConnect(producer, producer));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockConnect(consumer[0], consumer[0]));
    // TODO  can one pool block be connected to another pool block?
    // ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockConnect(pool, pool);
    // TODO  can one queue block be connected to another queue block?
    // ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockConnect(queue, queue);


    // One block is {invalid, null}
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(producer, nullBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(producer, ilwalidBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(nullBlock, pool));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(ilwalidBlock, pool));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(pool, nullBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(pool, ilwalidBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(nullBlock, queue[0]));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(ilwalidBlock, queue[0]));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(queue[0], nullBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(queue[0], ilwalidBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(nullBlock, consumer[0]));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(ilwalidBlock, consumer[0]));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(multicast, nullBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(multicast, ilwalidBlock));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(nullBlock, multicast));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockConnect(ilwalidBlock, multicast));
}

//==============================================================================
// Test Case 3: Call sync requirements with incorrect parameters
//==============================================================================
TEST_F(Negative, SyncSetup_WithoutBlocks)
{
    LwSciSyncAttrList dummyAttr;

    // Call sync waiter attr set/get with bad blocks
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockElementWaiterAttrSet(nullBlock, 0, nullptr));
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockElementWaiterAttrSet(ilwalidBlock, 0, nullptr));
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockElementWaiterAttrGet(nullBlock, 0, &dummyAttr));
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockElementWaiterAttrGet(ilwalidBlock, 0, &dummyAttr));
}

//==============================================================================
// Test Case 4: Call sync requirements on unconnected stream
//==============================================================================
TEST_F(Negative, SyncSetup_UnconnectedStream)
{
    LwSciSyncAttrList dummyAttr;

    // Create blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Call on unconnected stream
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementWaiterAttrSet(producer, 0, nullptr));
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementWaiterAttrSet(consumer[0], 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrSet(pool, 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrSet(queue[0], 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrSet(multicast, 0, nullptr));

    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementWaiterAttrGet(producer, 0, &dummyAttr));
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementWaiterAttrGet(consumer[0], 0, &dummyAttr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrGet(pool, 0, &dummyAttr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrGet(queue[0], 0, &dummyAttr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrGet(multicast, 0, &dummyAttr));
}

//==============================================================================
// Test Case 5: Call sync requirements before element setup is done
//==============================================================================
TEST_F(Negative, SyncSetup_NormalStream)
{
    LwSciSyncAttrList dummyAttr;

    // Create blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Create stream
    connectStream();

    // Call sync requirements on all blocks with null sync attr
    ASSERT_EQ(LwSciError_NotYetAvailable,
        LwSciStreamBlockElementWaiterAttrSet(producer, 0, nullptr));
    ASSERT_EQ(LwSciError_NotYetAvailable,
        LwSciStreamBlockElementWaiterAttrSet(consumer[0], 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrSet(pool, 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrSet(queue[0], 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrSet(multicast, 0, nullptr));

    ASSERT_EQ(LwSciError_NotYetAvailable,
        LwSciStreamBlockElementWaiterAttrGet(producer, 0, &dummyAttr));
    ASSERT_EQ(LwSciError_NotYetAvailable,
        LwSciStreamBlockElementWaiterAttrGet(consumer[0], 0, &dummyAttr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrGet(pool, 0, &dummyAttr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrGet(queue[0], 0, &dummyAttr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementWaiterAttrGet(multicast, 0, &dummyAttr));
}

//==============================================================================
// Test Case: Call sync requirements with invalid index
//==============================================================================
TEST_F(Negative, SyncSetup_ElementRange)
{
    // Create blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Create stream
    connectStream();

    // Open buf and sync modules. Modules will be closed in destructor
    ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
    ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

    // Set up elements
    makeRawBufferAttrList(bufModule, rawBufAttrList);
    packetAttrSetup();

    // Create sync attrs with synchronousOnly flag set
    prodSynchronousOnly = true;
    consSynchronousOnly = true;
    cpuWaiterAttrList(syncModule, prodSyncAttrList);
    cpuWaiterAttrList(syncModule, consSyncAttrList);

    // Call sync requirements with index equal to element count
    ASSERT_EQ(LwSciError_IndexOutOfRange,
            LwSciStreamBlockElementWaiterAttrSet(producer, elementCount,
                                                 prodSyncAttrList));
    for (uint32_t i = 0U; i < numConsumers; i++) {
        ASSERT_EQ(LwSciError_IndexOutOfRange,
                LwSciStreamBlockElementWaiterAttrSet(consumer[i], elementCount,
                                                     consSyncAttrList));
    }
}

// Disabled as these aren't relevant with the new attribute specification
//   APIs, but we can colwert these to functions that make sure the
//   per-element sync objects are consistent with the per-element attributes
//   when the sync object changes go in.
#if 0

//==============================================================================
// Test Case: Call sync object count with non-zero count on a synchronous stream
//==============================================================================
TEST_F(Negative, SyncSetup_SynchronousStream2)
{
    // Create blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Create stream
    connectStream();

    // Open sync module. Module will be closed in destructor
    ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

    prodSynchronousOnly = true;
    consSynchronousOnly = true;
    prodSyncAttrList = nullptr;
    consSyncAttrList = nullptr;

    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(producer,
                                             true,
                                             prodSyncAttrList));
    for (uint32_t i = 0U; i < numConsumers; i++) {
        ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSyncRequirements(consumer[i],
                                                 true,
                                                 consSyncAttrList));
    }

    // Sync object count with non-call sync count
    EXPECT_EQ(LwSciError_IlwalidOperation,
            LwSciStreamBlockSyncObjCount(producer, 1U));
    EXPECT_EQ(LwSciError_IlwalidOperation,
            LwSciStreamBlockSyncObjCount(consumer[0], 1U));
}

//==============================================================================
// Test Case: Send sync objects without explicitly setting sync counts to zero
// on a synchronous stream
//==============================================================================
TEST_F(Negative, SyncSetup_SynchronousStream3)
{
    uint32_t syncCount = 1U;

    // Create blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Create stream
    connectStream();

    // Open sync module. Module will be closed in destructor
    ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

    prodSynchronousOnly = true;
    consSynchronousOnly = true;
    prodSyncAttrList = nullptr;
    consSyncAttrList = nullptr;

    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(producer, true, nullptr));
    for (uint32_t i = 0U; i < numConsumers; i++) {
        ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSyncRequirements(consumer[i], true, nullptr));
    }

    // Don't send sync counts

    // Send only 1 sync object
    for (uint32_t i = 0; i < syncCount; i++) {
        getSyncObj(syncModule, prodSyncObjs[i]);
        ASSERT_EQ(LwSciError_IlwalidOperation,
                  LwSciStreamBlockSyncObject(producer, i, prodSyncObjs[i]));
    }

    for (uint32_t i = 0; i < numConsumers; ++i) {
        // Send only 1 sync object
        for (uint32_t j = 0; j < syncCount; j++) {
            getSyncObj(syncModule, consSyncObjs[i][j]);
            ASSERT_EQ(LwSciError_IlwalidOperation,
                    LwSciStreamBlockSyncObject(consumer[i],
                                               j, consSyncObjs[i][j]));
        }
    }
}

#endif // Temporarily disabled. See comment above

//==============================================================================
// Test Case 6: Call paket setup related APIs with incorrect parameter
// query attribute, packet element count, packet attr, packet
// create, inset buffer, packet accept with incorrect parameter
//==============================================================================
TEST_F(Negative, PacketSetup)
{
    // Call query attribute with incorrect parameters
    int32_t value;
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamAttributeQuery(
        static_cast<LwSciStreamQueryableAttrib>(ilwalidQueryAttrib), nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamAttributeQuery(
        static_cast<LwSciStreamQueryableAttrib>(ilwalidQueryAttrib), &value));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamAttributeQuery(
        LwSciStreamQueryableAttrib_MaxElements, nullptr));
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamAttributeQuery(
        LwSciStreamQueryableAttrib_MaxSyncObj, nullptr));

    // Call packet attr with incorrect parameters
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockElementAttrSet(nullBlock, 0, nullptr));
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockElementAttrSet(ilwalidBlock, 0, nullptr));

    // Call packet create with incorrect parameters
    CHECK_ERR_OR_PANIC(LwSciError_BadAddress,
        LwSciStreamPoolPacketCreate(nullBlock, 0U, nullptr));
    CHECK_ERR_OR_PANIC(LwSciError_BadAddress,
        LwSciStreamPoolPacketCreate(ilwalidBlock, 0U, nullptr));

    // Call insert buffer incorrect parameters
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamPoolPacketInsertBuffer(nullBlock, 0, 0, nullptr));
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamPoolPacketInsertBuffer(ilwalidBlock, 0, 0, nullptr));

    // Call packet accept with incorrect parameters
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockPacketStatusSet(
            nullBlock, 0U, 0U, LwSciError_BadParameter));
    CHECK_ERR_OR_PANIC(LwSciError_StreamBadBlock,
        LwSciStreamBlockPacketStatusSet(
            ilwalidBlock, 0U, 0U, LwSciError_BadParameter));
}

//==============================================================================
// Test Case 7: Call paket setup related APIs on unconnected stream
//==============================================================================
TEST_F(Negative, PacketSetup_UnconnctedStream)
{
    LwSciStreamPacket dummyHandle;

    // Create blocks
    createBlocks(QueueType::Mailbox, NUM_CONSUMERS);

    // Call block packet attr on each block type
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementAttrSet(producer, 0, nullptr));
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementAttrSet(consumer[0], 0, nullptr));
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockElementAttrSet(pool, 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockElementAttrSet(queue[0], 0, nullptr));

    // Call pool packet create on each block type
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamPoolPacketCreate(producer, 0U, &dummyHandle));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamPoolPacketCreate(consumer[0], 0U, &dummyHandle));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamPoolPacketCreate(queue[0], 0U, &dummyHandle));
    // TODO: To test for BadParameter error on the pool, need
    //       to do with a fully connected stream, so desired
    //       error isn't masked by others.
    ASSERT_EQ(LwSciError_BadAddress,
        LwSciStreamPoolPacketCreate(pool, 0U, nullptr));

    // Call packet insert buffer on each block type
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamPoolPacketInsertBuffer(producer, 0, 0, nullptr));
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamPoolPacketInsertBuffer(pool, 0, 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamPoolPacketInsertBuffer(consumer[0], 0, 0, nullptr));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamPoolPacketInsertBuffer(queue[0], 0, 0, nullptr));

    // Call packet accept on each block type
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockPacketStatusSet(
            producer, 0U, 0U, LwSciError_BadParameter));
    ASSERT_EQ(LwSciError_StreamNotConnected,
        LwSciStreamBlockPacketStatusSet(
            consumer[0], 0U, 0U, LwSciError_BadParameter));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockPacketStatusSet(
            pool, 0U, 0U, LwSciError_BadParameter));
    ASSERT_EQ(LwSciError_NotSupported,
        LwSciStreamBlockPacketStatusSet(
            queue[0], 0U, 0U, LwSciError_BadParameter));
}

//==============================================================================
// Test Case 8: Call paket setup related APIs on normal connected stream
//==============================================================================
TEST_F(Negative, PacketSetup_NormalStream)
{
    LwSciStreamPacket dummyHandle = 0U;
    LwSciStreamCookie nullCookie =  static_cast<LwSciStreamCookie>(0U);
    LwSciStreamCookie validCookie =  static_cast<LwSciStreamCookie>(1U);

    // Create blocks
    createBlocks(QueueType::Mailbox);

    // Create stream
    connectStream();

    ASSERT_EQ(LwSciError_BadParameter,
        LwSciStreamBlockElementAttrSet(producer, 0, nullptr));
    ASSERT_EQ(LwSciError_BadParameter,
        LwSciStreamBlockElementAttrSet(consumer[0], 0, nullptr));

    ASSERT_EQ(LwSciError_BadParameter,
        LwSciStreamBlockElementAttrSet(producer, 0, nullptr));
    ASSERT_EQ(LwSciError_BadParameter,
        LwSciStreamBlockElementAttrSet(consumer[0], 0, nullptr));

    CHECK_ERR_OR_PANIC(LwSciError_StreamBadPacket,
        LwSciStreamBlockPacketStatusSet(
            producer, dummyHandle, nullCookie, LwSciError_BadParameter));
    ASSERT_EQ(LwSciError_StreamBadCookie,
        LwSciStreamBlockPacketStatusSet(
            producer, dummyHandle, nullCookie, LwSciError_Success));
    ASSERT_EQ(LwSciError_StreamBadPacket,
        LwSciStreamBlockPacketStatusSet(
            producer, dummyHandle, validCookie, LwSciError_BadParameter));

    CHECK_ERR_OR_PANIC(LwSciError_StreamBadPacket,
        LwSciStreamBlockPacketStatusSet(
            consumer[0], dummyHandle, nullCookie, LwSciError_BadParameter));
    ASSERT_EQ(LwSciError_StreamBadCookie,
        LwSciStreamBlockPacketStatusSet(
            consumer[0], dummyHandle, nullCookie, LwSciError_Success));
    ASSERT_EQ(LwSciError_StreamBadPacket,
        LwSciStreamBlockPacketStatusSet(
            consumer[0], dummyHandle, validCookie, LwSciError_BadParameter));

    // To test the pool, we first have to send support from the producer
    //   and consumer and receive it at the pool. It isn't legal to set
    //   the pool's packet attributes until the supported attributes are
    //   known, so the functions would fail before validating the parameters.
    //   Then before creating the packet, we have to complete the pool's
    //   element import and export.
    LwSciStreamEventType event;
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSetupStatusSet(producer,
                                       LwSciStreamSetup_ElementExport, true));
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSetupStatusSet(consumer[0],
                                       LwSciStreamSetup_ElementExport, true));
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(pool, 0, &event));

    ASSERT_EQ(LwSciError_BadParameter,
        LwSciStreamBlockElementAttrSet(pool, 0, nullptr));
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSetupStatusSet(pool,
                                       LwSciStreamSetup_ElementImport, true));
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSetupStatusSet(pool,
                                       LwSciStreamSetup_ElementExport, true));
    ASSERT_EQ(LwSciError_StreamBadCookie,
        LwSciStreamPoolPacketCreate(pool, nullCookie, &dummyHandle));
    ASSERT_EQ(LwSciError_BadAddress,
        LwSciStreamPoolPacketCreate(pool, validCookie, nullptr));

    ASSERT_EQ(LwSciError_BadParameter,
        LwSciStreamPoolPacketInsertBuffer(pool, 0, 0, nullptr));

}

//==============================================================================
// Test Case 9: Call block query with incorrect parameters
//==============================================================================
TEST_F(Negative, EventHandling)
{
    // Call event query with incorrect parameters
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockEventQuery(nullBlock, 0, nullptr));
    CHECK_ERR_OR_PANIC(LwSciError_BadParameter,
        LwSciStreamBlockEventQuery(ilwalidBlock, 0, nullptr));

    // Create blocks
    createBlocks(QueueType::Mailbox);

    // Call event query on all blocks on unconnected stream
    LwSciStreamEventType event;
    ASSERT_EQ(LwSciError_BadParameter,
    LwSciStreamBlockEventQuery(producer, 0, nullptr));
    ASSERT_EQ(LwSciError_BadParameter,
    LwSciStreamBlockEventQuery(consumer[0], 0, nullptr));
    ASSERT_EQ(LwSciError_BadParameter,
    LwSciStreamBlockEventQuery(pool, 0, nullptr));
    ASSERT_EQ(LwSciError_BadParameter,
    LwSciStreamBlockEventQuery(queue[0], 0, nullptr));

    // Query for valid event on every block.
    // There should be no events in producer and consumer.
    // Producer and queue should receive UpstreamConnection and
    // DownstreamConnection event, respectively.
    ASSERT_EQ(LwSciError_Timeout,
    LwSciStreamBlockEventQuery(producer, 10, &event));
    ASSERT_EQ(LwSciError_Timeout,
    LwSciStreamBlockEventQuery(consumer[0], 10, &event));
    ASSERT_EQ(LwSciError_Timeout,
    LwSciStreamBlockEventQuery(pool, 10, &event));
    ASSERT_EQ(LwSciError_Timeout,
    LwSciStreamBlockEventQuery(queue[0], 10, &event));
}

//==========================================================================
// Test Case 10: Invalid argument checks for LwSciStreamBlockErrorGet
//==========================================================================

TEST_F(Negative, LwSciStreamBlockErrorGet_IlwalidArguments)
{
    LwSciStreamBlock pool = 0xABCD;
    LwSciError status = LwSciError_AccessDenied;

    /* Passing invalid block */
    ASSERT_EQ(LwSciError_StreamBadBlock,
        LwSciStreamBlockErrorGet(pool, &status));

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamStaticPoolCreate(NUM_PACKETS, &pool));
    ASSERT_NE(0, pool);

    /* Passing NULL argument */
    ASSERT_EQ(LwSciError_BadAddress,
        LwSciStreamBlockErrorGet(pool, nullptr));

}