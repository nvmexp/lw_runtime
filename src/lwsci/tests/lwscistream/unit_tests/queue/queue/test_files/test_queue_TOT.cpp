//
// Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
/// @file

#include <iostream>
#include "queue.h"
#include <limits>
#include <functional>
#include "sciwrap.h"
#include "lwscistream_common.h"
#include "producer.h"
#include "pool.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include "lwscistream_panic.h"
#include "lwscistream_panic_mock.h"

class queue_unit_test: public LwSciStreamTest {
public:
   queue_unit_test( ) {
       // initialization code here
   }

   void SetUp( ) {
       // code here will execute just before the test ensues
   }

   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~queue_unit_test( )  {
       // cleanup any pending stuff, but no exceptions and no gtest
       // ASSERT* allowed.
   }

   // put in any custom data members that you need
};

//==============================================
// Define queue_unit_sync_setup_test suite
//==============================================
class queue_unit_sync_setup_test: public LwSciStreamTest {
public:
    queue_unit_sync_setup_test( ) {
       // initialization code here
    }

    void SetUp( ) {
       // code here will execute just before the test ensues
       EXPECT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
    }

    // Producer sends its sync object requirement to the consumer.
    inline void prodSendSyncAttr()
    {
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,
                                             prodSyncAttrList));
    };

    // Consumer sends its sync object requirement to the producer.
    inline void consSendSyncAttr()
    {
        for (uint32_t n = 0U; n < numConsumers; n++) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockSyncRequirements(consumer[n],
                                                 consSynchronousOnly,
                                                 consSyncAttrList));
        }
    };

    // Producer receives consumer's sync object requirement.
    inline void prodRecvSyncAttr()
    {
        LwSciStreamEvent event;
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
        EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
        LwSciSyncAttrListFree(event.syncAttrList);
    };

    // Consumer receives producer's sync object requirement.
    inline void consRecvSyncAttr()
    {
        for (uint32_t n = 0U; n < numConsumers; n++) {
            LwSciStreamEvent event;
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
            EXPECT_EQ(prodSynchronousOnly, event.synchronousOnly);
            LwSciSyncAttrListFree(event.syncAttrList);
        }
    };

    inline void prodSendSyncObj(uint32_t syncCount = NUM_SYNCOBJS)
    {
        EXPECT_TRUE(syncCount <= MAX_SYNC_COUNT);
        prodSyncCount = syncCount;

        // Producer sends its sync count to the consumer.
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

        // Producer creates sync objects based on consumer's requirement and
        // sends the sync object to the consumer.
        for (uint32_t i = 0U; i < prodSyncCount; ++i) {
            getSyncObj(syncModule, prodSyncObjs[i]);
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockSyncObject(producer, i, prodSyncObjs[i]));
        }
    };

    inline void consSendSyncObj(uint32_t syncCount = NUM_SYNCOBJS)
    {
        EXPECT_TRUE((syncCount*numConsumers) <= MAX_SYNC_COUNT);
        for (uint32_t n = 0U; n < numConsumers; ++n) {
            consSyncCount[n] = syncCount;
        }
        totalConsSync = syncCount * numConsumers;

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer sends its sync count to the producer.
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockSyncObjCount(consumer[n], consSyncCount[n]));

            // Consumer creates sync objects based on producer's requirement and
            // sends the sync object to the producer.
            for (uint32_t i = 0U; i < consSyncCount[n]; ++i) {
                getSyncObj(syncModule, consSyncObjs[n][i]);
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockSyncObject(consumer[n],
                                               i, consSyncObjs[n][i]));
            }
        }
    };

    inline void prodRecvSyncObj()
    {
        LwSciStreamEvent event;

        // Producer receives consumer's sync count.
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_SyncCount, event.type);
        EXPECT_EQ(totalConsSync, event.count);

        // Producer receives consumer's sync objects.
        for (uint32_t i = 0U; i < totalConsSync; ++i) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_SyncDesc, event.type);
            EXPECT_LT(event.index, totalConsSync);

            LwSciSyncObjFree(event.syncObj);
        }
    };

    inline void consRecvSyncObj()
    {
        LwSciStreamEvent event;

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Consumer receives producer's sync count.
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_SyncCount, event.type);
            EXPECT_EQ(prodSyncCount, event.count);

            // Consumer receives producer's sync objects.
            for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
                EXPECT_EQ(LwSciStreamEventType_SyncDesc, event.type);
                EXPECT_LT(event.index, prodSyncCount);

                LwSciSyncObjFree(event.syncObj);
            }
        }
    };

   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~queue_unit_sync_setup_test( )  {
       // cleanup any pending stuff, but no exceptions allowed
   }

   // put in any custom data members that you need
};

//==============================================
// Define queue_unit_buf_setup_test suite
//==============================================
class queue_unit_buf_setup_test: public LwSciStreamTest {
public:
    queue_unit_buf_setup_test( ) {
        // initialization code here
    }

    uint32_t prodElementCount = NUM_PACKET_ELEMENTS;
    uint32_t consElementCount = NUM_PACKET_ELEMENTS;
    // Fake number of reconciled packet elements
    uint32_t reconciledElementCount = NUM_PACKET_ELEMENTS;

    void SetUp( ) {
       // code here will execute just before the test ensues
       EXPECT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
    }

    inline void queryMaxNumElements()
    {
        int32_t value;
        EXPECT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
        EXPECT_TRUE(prodElementCount <= value);
        EXPECT_TRUE(consElementCount <= value);
    };

    inline void prodSendPacketAttr()
    {
        // Set the number of elements in a packet
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, prodElementCount));

        // Set producer packet requirements
        for (uint32_t i = 0U; i < prodElementCount; i++) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketAttr(producer, i, i,
                                   LwSciStreamElementMode_Asynchronous,
                                   rawBufAttrList));
        }
    };

    inline void consSendPacketAttr()
    {
        // Set the number of elements in a packet
        for (uint32_t n = 0U; n < numConsumers; n++) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketElementCount(consumer[n], consElementCount));
        }

        // Set consumer packet requirements
        for (uint32_t i = 0U; i < consElementCount; i++) {
            for (uint32_t n = 0U; n < numConsumers; n++) {
                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockPacketAttr(consumer[n], i, i,
                                       LwSciStreamElementMode_Asynchronous,
                                       rawBufAttrList));
            }
        }
    };

    inline void poolRecvPacketAttr()
    {
        LwSciStreamEvent event;
        uint32_t prodCountEvents { 0U };
        uint32_t consCountEvents { 0U };
        uint32_t prodAttrEvents { 0U };
        uint32_t consAttrEvents { 0U };
        uint32_t expectedEvents { 2U + prodElementCount + consElementCount };

        // Loop over all expected events
        for (uint32_t i = 0; i < expectedEvents; i++) {

            // Pool receives event
            EXPECT_EQ(LwSciError_Success,
                      LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT,
                                                 &event));

            switch (event.type) {

            // Producer count only arrives once, and matches expected
            case LwSciStreamEventType_PacketElementCountProducer:
                EXPECT_EQ(prodCountEvents, 0U);
                EXPECT_EQ(prodElementCount, event.count);
                prodCountEvents++;
                break;

            // Consumer count only arrives once, and matches expected
            case LwSciStreamEventType_PacketElementCountConsumer:
                EXPECT_EQ(consCountEvents, 0U);
                EXPECT_EQ(consElementCount, event.count);
                consCountEvents++;
                break;

            // Producer attrs should only come after count and not exceed max
            case LwSciStreamEventType_PacketAttrProducer:
                EXPECT_EQ(prodCountEvents, 1U);
                EXPECT_LT(prodAttrEvents, prodElementCount);
                LwSciBufAttrListFree(event.bufAttrList);
                prodAttrEvents++;
                break;

            // Consumer attrs should only come after count and not exceed max
            case LwSciStreamEventType_PacketAttrConsumer:
                EXPECT_EQ(consCountEvents, 1U);
                EXPECT_LT(consAttrEvents, consElementCount);
                LwSciBufAttrListFree(event.bufAttrList);
                consAttrEvents++;
                break;

            default:
                EXPECT_TRUE(false);
                break;
            }
        }

        EXPECT_EQ(prodCountEvents, 1U);
        EXPECT_EQ(consCountEvents, 1U);
        EXPECT_EQ(prodAttrEvents, prodElementCount);
        EXPECT_EQ(consAttrEvents, consElementCount);
    };

    // Pool sends reconciled attributes to producer and consumer
    inline void poolSendReconciledPacketAttr()
    {
        // Pool sets the number of elements in a packet
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(pool, reconciledElementCount));

        // Pool sets packet requirements
        for (uint32_t i = 0U; i < reconciledElementCount; i++) {
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockPacketAttr(pool, i, i,
                                   LwSciStreamElementMode_Asynchronous,
                                   rawBufAttrList));
        }
    };

   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~queue_unit_buf_setup_test( )  {
       // cleanup any pending stuff, but no exceptions allowed
   }

   // put in any custom data members that you need
};



//==============================================
// Define queue_unit_packet_stream_test suite
//==============================================

class queue_unit_packet_stream_test :
    public LwSciStreamTest
{
protected:
    virtual void SetUp()
    {
        EXPECT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);

        EXPECT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        prodSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, prodSyncAttrList);

        consSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, consSyncAttrList);
    };
};


namespace LwSciStream {

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncAttr_Success1}
 * @testcase{21188043}
 * @verify{19471320}
 * @testpurpose{Test positive scenario of Queue::srcSendSyncAttr(),
 * where the synchronousOnly is equal to False.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncAttr() API from queue object,
 * with valid sync attributes (synchronousOnly flag of false and LwSciWrap::SyncAttr wrapping a
 * valid LwSciSyncAttrList) and srcIndex of Block::singleConn,
 * should call srcSendSyncAttr() interface of consumer block to send the sync attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendSyncAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncAttr()}
 */
TEST_F (queue_unit_sync_setup_test, srcSendSyncAttr_Success1) {

    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    consSynchronousOnly = false;

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendSyncAttr(_, consSynchronousOnly, Ref(syncAttr)))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendSyncAttr(Block::singleConn_access, consSynchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncAttr_Success2}
 * @testcase{21188046}
 * @verify{19471320}
 * @testpurpose{Test positive scenario of Queue::srcSendSyncAttr(),
 * where the synchronousOnly is equal to True.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncAttr() API from queue object,
 * with valid sync attributes (synchronousOnly flag of false and LwSciWrap::SyncAttr wrapping a
 * valid LwSciSyncAttrList) and srcIndex of Block::singleConn,
 * should call srcSendSyncAttr() interface of consumer block to send the sync attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendSyncAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncAttr()}
 */
TEST_F (queue_unit_sync_setup_test, srcSendSyncAttr_Success2) {

    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;


    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    /*test code*/
    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendSyncAttr(_, true, Ref(syncAttr)))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendSyncAttr(Block::singleConn_access, true, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

 /**
 * @testname{queue_unit_sync_setup_test.srcSendSyncAttr_StreamNotConnected}
 * @testcase{21188049}
 * @verify{19471320}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncAttr(), where
 *   srcSendSyncAttr is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncAttr() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncAttr()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncAttr_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendSyncAttr(Block::singleConn_access, consSynchronousOnly, syncAttr);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncAttr_StreamBadSrcIndex}
 * @testcase{21188052}
 * @verify{19471320}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncAttr(),
 * where srcSendSyncAttr is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *    The call of Queue::srcSendSyncAttr() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncAttr()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncAttr_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
     // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendSyncAttr(ILWALID_CONN_IDX, consSynchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncCount_Success1}
 * @testcase{21188055}
 * @verify{19471323}
 * @testpurpose{Test positive scenario of Queue::srcSendSyncCount(),
 * where the count is equal to 1.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncCount() API from queue object,
 * with valid sync count of 1 and srcIndex of Block::singleConn,
 * should call srcSendSyncCount() interface of consumer block to send the sync count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement.}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendSyncCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncCount()}
 */
TEST_F (queue_unit_sync_setup_test, srcSendSyncCount_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;


    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    elementCount=1;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendSyncCount(_, elementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendSyncCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncCount_Success2}
 * @testcase{21188058}
 * @verify{19471323}
 * @testpurpose{Test positive scenario of Queue::srcSendSyncCount(),
 * where the count is equal to 0.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncCount() API from queue object,
 * with valid sync count of 0 and srcIndex of Block::singleConn,
 * should call srcSendSyncCount() interface of consumer block to send the sync count.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendSyncCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncCount()}
 */
TEST_F (queue_unit_sync_setup_test, srcSendSyncCount_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    elementCount=0;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendSyncCount(_, elementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendSyncCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncCount_Success3}
 * @testcase{21188061}
 * @verify{19471323}
 * @testpurpose{Test positive scenario of Queue::srcSendSyncCount(),
 * where the count is equal to MAX_SYNC_OBJECTS.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncCount() API from queue object,
 * with valid sync count of MAX_SYNC_OBJECTS and srcIndex of Block::singleConn,
 * should call srcSendSyncCount() interface of consumer block to send the sync count.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendSyncCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncCount()}
 */
TEST_F (queue_unit_sync_setup_test, srcSendSyncCount_Success3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a fifo stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendSyncCount(_, MAX_SYNC_OBJECTS))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendSyncCount(Block::singleConn_access, MAX_SYNC_OBJECTS);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncCount_IlwalidState}
 * @testcase{21188064}
 * @verify{19471323}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncCount(), where
 *   srcSendSyncCount is ilwoked more that once.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. consumer receives the sync attribute from the producer by querying
 *      LwSciStreamEventType_SyncAttr event.
 *   3. Queue::srcSendSyncCount() API is called with valid parameters.
 *
 *   The call of Queue::srcSendSyncCount() API from queue object again,
 * with valid sync count of 1 and srcIndex of Block::singleConn,
 * should cause LwSciError_IlwalidState event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncCount_IlwalidState)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;


    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    prodSynchronousOnly = false;

    prodSendSyncAttr();
    consRecvSyncAttr();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    queuePtr->srcSendSyncCount(Block::singleConn_access, 1);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendSyncCount(Block::singleConn_access, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncCount_BadParameter}
 * @testcase{21188067}
 * @verify{19471323}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncCount(), where
 *   srcSendSyncCount is ilwoked with sync count greater than MAX_SYNC_OBJECTS.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. consumer receives the sync attribute from the producer by querying
 *      LwSciStreamEventType_SyncAttr event.
 *
 *   The call of Queue::srcSendSyncCount() API from queue object,
 * with sync count MAX_SYNC_OBJECTS + 1 and srcIndex of Block::singleConn,
 * should cause LwSciError_BadParameter event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncCount_BadParameter)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    prodSendSyncAttr();
    consRecvSyncAttr();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    queuePtr->srcSendSyncCount(Block::singleConn_access, MAX_SYNC_OBJECTS + 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

 /**
 * @testname{queue_unit_sync_setup_test.srcSendSyncCount_StreamNotConnected}
 * @testcase{21188070}
 * @verify{19471323}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncCount(), where
 *   srcSendSyncCount is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncCount() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncCount_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    queuePtr->srcSendSyncCount(Block::singleConn_access, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncCount_StreamBadSrcIndex}
 * @testcase{21188073}
 * @verify{19471323}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncCount(),
 * where srcSendSyncCount is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncCount() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncCount_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendSyncCount(ILWALID_CONN_IDX, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendPacketElementCount_Success1}
 * @testcase{21188076}
 * @verify{19471332}
 * @testpurpose{Test positive scenario of Queue::srcSendPacketElementCount(),
 * where the element count is equal to 2(which is less than  MAX_PACKET_ELEMENTS).}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketElementCount() API from queue object,
 * with valid element count of 2 and srcIndex of Block::singleConn,
 * should call srcSendPacketElementCount() interface of consumer block to send the element count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement.}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketElementCount()}
 */
TEST_F (queue_unit_buf_setup_test, srcSendPacketElementCount_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;


    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendPacketElementCount(_, reconciledElementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacketElementCount(Block::singleConn_access, reconciledElementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendPacketElementCount_Success2}
 * @testcase{21188079}
 * @verify{19471332}
 * @testpurpose{Test positive scenario of Queue::srcSendPacketElementCount(),
 * where the element count is equal to 0.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketElementCount() API from queue object,
 * with valid element count of 0 and srcIndex of Block::singleConn,
 * should call srcSendPacketElementCount() interface of consumer block to send the element count.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketElementCount()}
 */
TEST_F (queue_unit_buf_setup_test, srcSendPacketElementCount_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    reconciledElementCount=0;

    queryMaxNumElements();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendPacketElementCount(_, reconciledElementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacketElementCount(Block::singleConn_access, reconciledElementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendPacketElementCount_Success3}
 * @testcase{21188082}
 * @verify{19471332}
 * @testpurpose{Test positive scenario of Queue::srcSendPacketElementCount(),
 * where the element count is equal to MAX_PACKET_ELEMENTS.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketElementCount() API from queue object,
 * with valid element count of MAX_PACKET_ELEMENTS and srcIndex of Block::singleConn,
 * should call srcSendPacketElementCount() interface of consumer block to send the element count.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketElementCount()}
 */
TEST_F (queue_unit_buf_setup_test, srcSendPacketElementCount_Success3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendPacketElementCount(_, MAX_PACKET_ELEMENTS))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacketElementCount(Block::singleConn_access, MAX_PACKET_ELEMENTS);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendPacketElementCount_IlwalidState}
 * @testcase{21188085}
 * @verify{19471332}
 * @testpurpose{Test negative scenario of Queue::srcSendPacketElementCount(), where
 *   srcSendPacketElementCount is ilwoked more that once.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) call Queue::srcSendPacketElementCount() API from queue object with valid element count
 *
 *   The call of Queue::srcSendPacketElementCount() API from queue object,
 * with valid element count of 1 and srcIndex of Block::singleConn,
 * should cause LwSciError_IlwalidState event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketElementCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendPacketElementCount_IlwalidState)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;


    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    prodSynchronousOnly = false;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    queuePtr->srcSendPacketElementCount(Block::singleConn_access, 1);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendPacketElementCount(Block::singleConn_access, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

}

/**
 * @testname{queue_unit_sync_setup_test.srcSendPacketElementCount_BadParameter}
 * @testcase{21188088}
 * @verify{19471332}
 * @testpurpose{Test negative scenario of Queue::srcSendPacketElementCount(), where
 *   srcSendPacketElementCount is ilwoked with element count greater than MAX_PACKET_ELEMENTS.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketElementCount() API from queue object,
 * with element count MAX_PACKET_ELEMENTS + 1 and srcIndex of Block::singleConn,
 * should cause LwSciError_BadParameter event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketElementCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendPacketElementCount_BadParameter)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    prodSynchronousOnly = false;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendPacketElementCount(Block::singleConn_access, MAX_PACKET_ELEMENTS + 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

 /**
 * @testname{queue_unit_sync_setup_test.srcSendPacketElementCount_StreamNotConnected}
 * @testcase{21188091}
 * @verify{19471332}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendPacketElementCount(), where
 *   srcSendPacketElementCount is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketElementCount() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketElementCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendPacketElementCount_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendPacketElementCount(Block::singleConn_access, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

}

/**
 * @testname{queue_unit_sync_setup_test.srcSendPacketElementCount_StreamBadSrcIndex}
 * @testcase{21188097}
 * @verify{19471332}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendPacketElementCount(),
 * where srcSendPacketElementCount is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketElementCount() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketElementCount()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendPacketElementCount_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendPacketElementCount(ILWALID_CONN_IDX, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);

}

/**
 * @testname{queue_unit_sync_setup_test.srcCreatePacket_Success}
 * @testcase{21188100}
 * @verify{19471338}
 * @testpurpose{Test positive scenario of Queue::srcCreatePacket()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcCreatePacket() API from queue object,with valid
 * srcIndex of Block::singleConn, should ilwoke srcCreatePacket() interface
 * of the consumer block.
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcCreatePacket()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcCreatePacket()}
 */
TEST_F (queue_unit_buf_setup_test, srcCreatePacket_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcCreatePacket(_, packetHandle))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcCreatePacket(Block::singleConn_access, packetHandle);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_buf_setup_test.srcCreatePacket_InsufficientMemory}
 * @testcase{21188103}
 * @verify{19471338}
 * @testpurpose{Test negative scenario of Queue::srcCreatePacket(), where
 *   fails to create a new Packet.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Inject fault in Block::pktCreate() to return LwSciError_InsufficientMemory.
 *
 *   The call of Queue::srcCreatePacket() API from queue object,
 * should cause LwSciError_InsufficientMemory event, when Block::pktCreate() fails due to
 * LwSciError_InsufficientMemory.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.
 *      - Block::pktCreate() is mocked to return LwSciError_InsufficientMemory
 * }
 * @verifyFunction{Queue::srcCreatePacket()}
 */
TEST_F (queue_unit_buf_setup_test, srcCreatePacket_InsufficientMemory)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Create a mailbox stream

    createBlocks(QueueType::Mailbox);
    connectStream();

    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(LwSciStreamCookie_Ilwalid);
    LwSciStreamPacket packetHandle {~poolCookie};

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    test_block.pktCreate_fail = true;
    queuePtr->srcCreatePacket(Block::singleConn_access, packetHandle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_InsufficientMemory, event.error);
}

 /**
 * @testname{queue_unit_sync_setup_test.srcCreatePacket_StreamNotConnected}
 * @testcase{21188106}
 * @verify{19471338}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcCreatePacket(), where
 *   srcCreatePacket is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1)Creates the producer, pool, queue(mailbox type) and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcCreatePacket() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcCreatePacket()}
 */
TEST_F(queue_unit_sync_setup_test, srcCreatePacket_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcCreatePacket(Block::singleConn_access, packetHandle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcCreatePacket_StreamBadSrcIndex}
 * @testcase{21188109}
 * @verify{19471338}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcCreatePacket(),
 * where srcCreatePacket is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcCreatePacket() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcCreatePacket()}
 */
TEST_F(queue_unit_sync_setup_test, srcCreatePacket_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcCreatePacket(ILWALID_CONN_IDX, packetHandle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.srcCreatePacket_StreamInternalError}
 * @testcase{21188115}
 * @verify{19471338}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcCreatePacket(),
 * where invalid LwSciStreamPacket handle.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool through LwSciStreamBlockPacketElementCount() API sends reconciled
 *      PacketElementCount and PacketAttr back to producer and consumer.
 *   4) Call to Queue::srcCreatePacket() with packetHandle
 *
 *   The call of Queue::srcCreatePacket() API with same packetHandle,
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcCreatePacket()}
 */
TEST_F (queue_unit_buf_setup_test, srcCreatePacket_StreamInternalError)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Create a mailbox stream

    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    queuePtr->srcCreatePacket(Block::singleConn_access, packetHandle);

    //test code

    queuePtr->srcCreatePacket(Block::singleConn_access, packetHandle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.srcInsertBuffer_Success}
 * @testcase{21188118}
 * @verify{19471341}
 * @testpurpose{Test positive scenario of Queue::srcInsertBuffer()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4) Queue::srcCreatePacket() is ilwoked with packetHandle and producer accepts
 *      the packet by calling LwSciStreamBlockPacketAccept().
 *
 *   The call of Queue::srcInsertBuffer() API from queue object,
 * with valid packetHandle of LwSciStreamPacket, elemIndex of 0, elemBufObj wrapping a LwSciBufObj
 * and srcIndex of Block::singleConn,
 * should call srcInsertBuffer() interface of consumer block to send the LwSciBufObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcInsertBuffer()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcInsertBuffer()}
 */
TEST_F (queue_unit_buf_setup_test, srcInsertBuffer_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    LwSciBufObj poolElementBuf;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    queuePtr->srcCreatePacket(Block::singleConn_access, packetHandle);

    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcInsertBuffer(_, packetHandle, 0, Ref(wrapBufObj)))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcInsertBuffer(Block::singleConn_access, packetHandle, 0, wrapBufObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

 /**
 * @testname{queue_unit_buf_setup_test.srcInsertBuffer_StreamNotConnected}
 * @testcase{21188121}
 * @verify{19471341}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcInsertBuffer(), where
 *   srcInsertBuffer is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1) Creates the producer, pool, queue(mailbox type) and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcInsertBuffer() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcInsertBuffer()}
 */
TEST_F(queue_unit_buf_setup_test, srcInsertBuffer_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciBufObj poolElementBuf;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcInsertBuffer(Block::singleConn_access, packetHandle, 0, wrapBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.srcInsertBuffer_StreamBadSrcIndex}
 * @testcase{21188124}
 * @verify{19471341}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcInsertBuffer(),
 * where srcInsertBuffer is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4) Queue::srcCreatePacket() is ilwoked packetHandle and producer accepts
 *      the packet by calling LwSciStreamBlockPacketAccept.
 *
 *   The call of Queue::srcInsertBuffer() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcInsertBuffer()}
 */
TEST_F(queue_unit_buf_setup_test, srcInsertBuffer_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    LwSciBufObj poolElementBuf;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    queuePtr->srcCreatePacket(Block::singleConn_access, packetHandle);

    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcInsertBuffer(ILWALID_CONN_IDX, packetHandle, 0, wrapBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}



/**
 * @testname{queue_unit_packet_stream_test.srcDeletePacket_Success}
 * @testcase{21188145}
 * @verify{19471344}
 * @testpurpose{Test positive scenario of Queue::srcDeletePacket()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up buffer resources.
 *
 *   The call of Queue::srcDeletePacket() API from producer object with LwSciStreamPacket
 * (received earlier through a LwSciStreamEventType_PacketCreate event during
 * buffer resource setup), should call srcDeletePacket() interface of consumer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcDeletePacket()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDeletePacket()}
 */
TEST_F(queue_unit_packet_stream_test, srcDeletePacket_Success)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    EXPECT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));

    // Save the cookie-to-handle mapping
    poolCPMap.emplace(poolCookie, packetHandle);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t k = 0; k < elementCount; ++k) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
        EXPECT_EQ(LwSciError_Success,
                    LwSciStreamPoolPacketInsertBuffer(pool,
                                                      packetHandle, k,
                                                      poolElementBuf[k]));
    }

    // Producer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Assign cookie to producer packet handle
    LwSciStreamPacket producerPacket = event.packetHandle;
    LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError producerError = LwSciError_Success;

    // Producer accepts a packet provided by the pool
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
              producer, producerPacket, producerCookie, producerError));

    //delete the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcDeletePacket(_, producerPacket))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcDeletePacket(Block::singleConn_access, producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_packet_stream_test.srcDeletePacket_StreamInternalError}
 * @testcase{21188148}
 * @verify{19471344}
 * @testpurpose{Test negative scenario of Queue::srcDeletePacket(), where
 * location of the packet instance is not Location::Upstream.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Setup buffer resources with packet count of 1.
 *   3) Use mock Block::pktFindByHandle() to get the packet object from the
 *      packetHandle (received earlier through a LwSciStreamEventType_PacketCreate event during
 *      buffer resource setup)and ilwoke Packet::locationUpdate()
 *      to change the location of packet handle to Packet::Location::Downstream.
 *
 *   The call of Queue::srcDeletePacket() API from queue object,
 * with valid packetHandle(received earlier through a LwSciStreamEventType_PacketCreate event during
 * buffer resource setup)and srcIndex of Block::singleConn,
 * should cause LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::pktFindByHandle() access is changed to public.
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDeletePacket()}
 */
TEST_F(queue_unit_packet_stream_test, srcDeletePacket_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;


    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    EXPECT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));

    // Save the cookie-to-handle mapping
    poolCPMap.emplace(poolCookie, packetHandle);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t k = 0; k < elementCount; ++k) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
        EXPECT_EQ(LwSciError_Success,
                    LwSciStreamPoolPacketInsertBuffer(pool,
                                                      packetHandle, k,
                                                      poolElementBuf[k]));
    }

    // Producer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Assign cookie to producer packet handle
    LwSciStreamPacket producerPacket = event.packetHandle;
    LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError producerError = LwSciError_Success;

    // Producer accepts a packet provided by the pool
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
              producer, producerPacket, producerCookie, producerError));

    for (uint32_t k = 0; k < elementCount; ++k) {
        EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);
    }

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    PacketPtr const pkt { queuePtr->pktFindByHandle_access(producerPacket) };

    pkt->locationUpdate(Packet::Location::Upstream, Packet::Location::Downstream);

    queuePtr->srcDeletePacket(Block::singleConn_access, producerPacket);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{queue_unit_packet_stream_test.srcDeletePacket_StreamInternalError2}
 * @testcase{21188151}
 * @verify{19471344}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcDeletePacket(), where
 *   producerPacket is invalid.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcDeletePacket() API from queue object,
 * with invalid LwSciStreamPacket handle,
 * should cause LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDeletePacket()}
 */
TEST_F(queue_unit_packet_stream_test, srcDeletePacket_StreamInternalError2)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    queuePtr->srcDeletePacket(Block::singleConn_access, ILWALID_PACKET_HANDLE);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcDeletePacket(Block::singleConn_access, ILWALID_PACKET_HANDLE);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

 /**
 * @testname{queue_unit_packet_stream_test.srcDeletePacket_StreamNotConnected}
 * @testcase{21188154}
 * @verify{19471344}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcDeletePacket(), where
 *   srcDeletePacket is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1) Creates the producer, pool, queue(mailbox type) and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcDeletePacket() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDeletePacket()}
 */
TEST_F(queue_unit_packet_stream_test, srcDeletePacket_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcDeletePacket(Block::singleConn_access, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_packet_stream_test.srcDeletePacket_StreamBadSrcIndex}
 * @testcase{21188157}
 * @verify{19471344}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcDeletePacket(),
 * where srcDeletePacket is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   3) Set up synchronization and buffer resources.
 *
 *   The call of Queue::srcDeletePacket() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDeletePacket()}
 */
TEST_F(queue_unit_packet_stream_test, srcDeletePacket_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcDeletePacket(ILWALID_CONN_IDX, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{queue_unit_test.srcDisconnect_Success1}
 * @testcase{21188160}
 * @verify{19471314}
 * @testpurpose{Test positive scenario of Queue::srcDisconnect(), when the
 * payloadQ is empty.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcDisconnect() API from queue object,
 * with valid srcIndex of Block::singleConn,
 * should ilwoke srcDisconnect() interface of the consumer block. }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcDisconnect()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDisconnect()}
 */
TEST_F (queue_unit_test, srcDisconnect_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcDisconnect(Block::singleConn_access))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcDisconnect(Block::singleConn_access);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));

}

/**
 * @testname{queue_unit_test.srcDisconnect_Success2}
 * @testcase{22571442}
 * @verify{19471314}
 * @testpurpose{Test positive scenario of Queue::srcDisconnect(),
 * when payloadQueue is not empty.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Complete synchronization and buffer setup.
 *   3) Producer presents the packet to the consumer.
 *
 *   The call of Queue::srcDisconnect() API from queue object,
 * with valid srcIndex of Block::singleConn, should not trigger
 * LwSciStreamEventType_Disconnected event until drained the queue by calling
 * the Queue::dstAcquirePacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDisconnect()}
 */

TEST_F (queue_unit_test, srcDisconnect_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };


    ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketPresent(producer, handle, fences));

      //test code

    queuePtr->srcDisconnect(Block::singleConn_access);

    EXPECT_EQ(LwSciError_Timeout, LwSciStreamBlockEventQuery(
              queue[0], EVENT_QUERY_TIMEOUT, &event));

    Payload acquiredPayload;

    EXPECT_EQ(true, queuePtr->dstAcquirePacket(Block::singleConn_access, acquiredPayload));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr));


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
              queue[0], EVENT_QUERY_TIMEOUT, &event));

    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);
}

 /**
 * @testname{queue_unit_test.srcDisconnect_Disconnected}
 * @testcase{22420983}
 * @verify{19471314}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcDisconnect(), where
 *   srcDisconnect is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1) Creates the producer, pool, queue(mailbox type) and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcDisconnect() API from queue object,
 * should cause LwSciStreamEventType_Disconnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDisconnect()}
 */
TEST_F(queue_unit_test, srcDisconnect_Disconnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr, connComplete())
               .WillRepeatedly(Return(false));

    queuePtr->srcDisconnect(Block::singleConn_access);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);

}

/**
 * @testname{queue_unit_test.srcDisconnect_StreamBadSrcIndex}
 * @testcase{21188166}
 * @verify{19471314}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcDisconnect(),
 * where srcDisconnect is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *     1) Create and connect the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcDisconnect() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcDisconnect()}
 */
TEST_F(queue_unit_test, srcDisconnect_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcDisconnect(ILWALID_CONN_IDX);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);

}

/**
 * @testname{queue_unit_test.dstSendSyncAttr_Success}
 * @testcase{21188169}
 * @verify{19471347}
 * @testpurpose{Test positive scenario of Queue::dstSendSyncAttr()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *      The call of Queue::dstSendSyncAttr() API from queue object,
 * with valid sync attributes (synchronousOnly flag and LwSciWrap::SyncAttr) and dstIndex,
 * should call dstSendSyncAttr() interface of pool block to send the sync attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendSyncAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncAttr()}
 */
TEST_F (queue_unit_test, dstSendSyncAttr_Success) {

    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    using ::testing::Ref;

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendSyncAttr(_, consSynchronousOnly, Ref(syncAttr)))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstSendSyncAttr(Block::singleConn_access, consSynchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_test.dstSendSyncAttr_StreamNotConnected}
 * @testcase{21188172}
 * @verify{19471347}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendSyncAttr(), where
 *   dstSendSyncAttr is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1) Creates the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendSyncAttr() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncAttr()}
 */
TEST_F(queue_unit_test, dstSendSyncAttr_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendSyncAttr(Block::singleConn_access, consSynchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_test.dstSendSyncAttr_StreamBadDstIndex}
 * @testcase{21188175}
 * @verify{19471347}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendSyncAttr(),
 * where srcDisconnect is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1) Create and connect the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendSyncAttr() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncAttr()}
 */
TEST_F(queue_unit_test, dstSendSyncAttr_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendSyncAttr(ILWALID_CONN_IDX, consSynchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);

}

/**
 * @testname{queue_unit_test.dstSendSyncCount_Success1}
 * @testcase{21188178}
 * @verify{19471350}
 * @testpurpose{Test positive scenario of Queue::dstSendSyncCount(),
 * where the element count is equal to 0.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(fifo type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendSyncCount() API from queue object,
 * with valid sync count of 0 and dstIndex of Block::singleConn,
 * should call dstSendSyncCount() interface of pool block to send the SyncCount.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement.}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendSyncCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncCount()}
 */
TEST_F (queue_unit_test, dstSendSyncCount_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a fifo stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendSyncCount(_, elementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstSendSyncCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_test.dstSendSyncCount_Success2}
 * @testcase{21188181}
 * @verify{19471350}
 * @testpurpose{Test positive scenario of Queue::dstSendSyncCount(),
 * where the element count is equal to 1.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(fifo type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendSyncCount() API from queue object,
 * with valid sync count of 1 and dstIndex of Block::singleConn,
 * should call dstSendSyncCount() interface of pool block to send the SyncCount.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendSyncCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncCount()}
 */
TEST_F (queue_unit_test, dstSendSyncCount_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    elementCount=1;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendSyncCount(_, elementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstSendSyncCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_test.dstSendSyncCount_Success3}
 * @testcase{21188184}
 * @verify{19471350}
 * @testpurpose{Test positive scenario of Queue::dstSendSyncCount(),
 * where the element count is equal to MAX_SYNC_OBJECTS.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(fifo type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *
 *   The call of Queue::dstSendSyncCount() API from queue object,
 * with valid sync count of MAX_SYNC_OBJECTS and dstIndex of Block::singleConn,
 * should call dstSendSyncCount() interface of pool block to send the SyncCount.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendSyncCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncCount()}
 */
TEST_F (queue_unit_test, dstSendSyncCount_Success3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendSyncCount(_, MAX_SYNC_OBJECTS))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstSendSyncCount(Block::singleConn_access, MAX_SYNC_OBJECTS);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}



/**
 * @testname{queue_unit_test.dstSendSyncCount_StreamNotConnected}
 * @testcase{21188190}
 * @verify{19471350}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendSyncCount(), where
 *   dstSendSyncCount is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendSyncCount() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncCount()}
 */
TEST_F(queue_unit_test, dstSendSyncCount_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendSyncCount(Block::singleConn_access, 1);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_test.dstSendSyncCountr_StreamBadDstIndex}
 * @testcase{21188193}
 * @verify{19471350}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendSyncCount(),
 * where dstSendSyncCount is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendSyncCount() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncCount()}
 */
TEST_F(queue_unit_test, dstSendSyncCount_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendSyncCount(ILWALID_CONN_IDX, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);


}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketElementCount_Success1}
 * @testcase{21188196}
 * @verify{19471356}
 * @testpurpose{Test positive scenario of Queue::dstSendPacketElementCount(),
 * where the element count is equal to 0.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *  The call of Queue::dstSendPacketElementCount() API from queue object,
 * with valid element count of 0 and dstIndex of Block::singleConn,
 * should call dstSendPacketElementCount() interface of pool block to send
 * the allocatedElemCount.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketElementCount()}
 */
TEST_F (queue_unit_buf_setup_test, dstSendPacketElementCount_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(_, elementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstSendPacketElementCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketElementCount_Success2}
 * @testcase{21188199}
 * @verify{19471356}
 * @testpurpose{Test positive scenario of Queue::dstSendPacketElementCount(),
 * where the element count is equal to 1.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *    The call of Queue::dstSendPacketElementCount() API from queue object,
 * with valid element count of 1 and dstIndex of Block::singleConn,
 * should call dstSendPacketElementCount() interface of pool block to send
 * the allocatedElemCount.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketElementCount()}
 */
TEST_F (queue_unit_buf_setup_test, dstSendPacketElementCount_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    elementCount=1;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(_, elementCount))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstSendPacketElementCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketElementCount_Success3}
 * @testcase{21188202}
 * @verify{19471356}
 * @testpurpose{Test positive scenario of Queue::dstSendPacketElementCount(),
 * where the element count is equal to MAX_PACKET_ELEMENTS.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketElementCount() API from queue object,
 * with valid element count of MAX_PACKET_ELEMENTS and dstIndex of Block::singleConn,
 * should call dstSendPacketElementCount() interface of pool block to
 * send the allocatedElemCount.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketElementCount()}
 */
TEST_F (queue_unit_buf_setup_test, dstSendPacketElementCount_Success3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(_, MAX_PACKET_ELEMENTS))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstSendPacketElementCount(Block::singleConn_access, MAX_PACKET_ELEMENTS);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketElementCount_StreamNotConnected}
 * @testcase{21188208}
 * @verify{19471356}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketElementCount(), where
 *   dstSendPacketElementCount is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue(mailbox type) and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketElementCount() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketElementCount()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketElementCount_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendPacketElementCount(Block::singleConn_access, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketElementCount_StreamBadDstIndex}
 * @testcase{21188214}
 * @verify{19471356}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketElementCount(),
 * where dstSendPacketElementCount is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketElementCount() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketElementCount()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketElementCount_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendPacketElementCount(ILWALID_CONN_IDX, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncDesc_Success}
 * @testcase{21188220}
 * @verify{19471326}
 * @testpurpose{Test positive scenario of Queue::srcSendSyncDesc()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncDesc() API from queue object,
 * with valid srcIndex of Block::singleConn, valid LwSciWrap::SyncObj and syncIndex of 0,
 * should call srcSendSyncDesc() interface of consumer block to send the sync objects.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendSyncDesc()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncDesc()}
 */
TEST_F (queue_unit_sync_setup_test, srcSendSyncDesc_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    EXPECT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

    getSyncObj(syncModule, consSyncObjs[0][0]);
    LwSciWrap::SyncObj wrapSyncObj { consSyncObjs[0][0] };

    /*test code*/
    EXPECT_CALL(*consumerPtr[0], srcSendSyncDesc(_, 0, Ref(wrapSyncObj)))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendSyncDesc(Block::singleConn_access, 0, wrapSyncObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncDesc_StreamNotConnected}
 * @testcase{21188232}
 * @verify{19471326}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncDesc(), where
 *   srcSendSyncDesc is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncDesc() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncDesc()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncDesc_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::SyncObj wrapSyncObj { consSyncObjs[0][0] };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendSyncDesc(Block::singleConn_access, 0, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.srcSendSyncDesc_StreamBadSrcIndex}
 * @testcase{21188238}
 * @verify{19471326}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendSyncDesc(),
 * where srcSendSyncDesc is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendSyncDesc() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendSyncDesc()}
 */
TEST_F(queue_unit_sync_setup_test, srcSendSyncDesc_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::SyncObj wrapSyncObj { consSyncObjs[0][0] };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendSyncDesc(ILWALID_CONN_IDX, 0, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.dstSendSyncDesc_Success}
 * @testcase{21188244}
 * @verify{19471353}
 * @testpurpose{Test positive scenario of Queue::dstSendSyncDesc()}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. producer receives the sync attribute from the consumer by querying
 *      LwSciStreamEventType_SyncAttr event.
 *
 *   The call of Queue::dstSendSyncDesc() API from queue object,
 * with valid dstIndex of Block::singleConn, SyncObj of LwSciWrap::SyncObj and
 * LwSciSyncObj count of 0, should call dstSendSyncDesc() interface
 * of pool block to send the sync objects.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendSyncDesc()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncDesc()}
 */
TEST_F (queue_unit_sync_setup_test, dstSendSyncDesc_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    consSynchronousOnly = false;
    consSendSyncAttr();

    prodRecvSyncAttr();


   LwSciStreamBlockSyncObject(producer, 0, prodSyncObjs[0]);


    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        getSyncObj(syncModule, prodSyncObjs[i]);

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
        LwSciWrap::SyncObj wrapSyncObj { prodSyncObjs[i] };

        /*test code*/
         EXPECT_CALL(*poolPtr, dstSendSyncDesc(_, 0, Ref(wrapSyncObj)))
              .Times(1)
              .WillRepeatedly(Return());

        queuePtr->dstSendSyncDesc(Block::singleConn_access, 0, wrapSyncObj);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
    }
}


/**
 * @testname{queue_unit_sync_setup_test.dstSendSyncDesc_StreamNotConnected}
 * @testcase{21188250}
 * @verify{19471353}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendSyncDesc(), where
 *   dstSendSyncDesc is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendSyncDesc() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncDesc()}
 */
TEST_F(queue_unit_sync_setup_test, dstSendSyncDesc_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::SyncObj wrapSyncObj { prodSyncObjs[0] };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendSyncDesc(Block::singleConn_access, 0, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_sync_setup_test.dstSendSyncDesc_StreamBadDstIndex}
 * @testcase{21188253}
 * @verify{19471353}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendSyncDesc(),
 * where dstSendSyncDesc is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *
 *   The call of Queue::dstSendSyncDesc() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in  LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendSyncDesc()}
 */
TEST_F(queue_unit_sync_setup_test, dstSendSyncDesc_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::SyncObj wrapSyncObj { prodSyncObjs[0] };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendSyncDesc(ILWALID_CONN_IDX, 0, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ( LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.srcSendPacketAttr_Success1}
 * @testcase{21188256}
 * @verify{19471335}
 * @testpurpose{Test positive scenario of Queue::srcSendPacketAttr(),
 * where the elemSyncMode is equal to LwSciStreamElementMode_Asynchronous.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool through LwSciStreamBlockPacketElementCount() API sets the element count.
 *
 *   The call of Queue::srcSendPacketAttr() API from queue object,
 * with valid srcIndex of Block::singleConn, elemIndex of NUM_PACKET_ELEMENTS, elemType of
 * NUM_PACKET_ELEMENTS, elemSyncMode of LwSciStreamElementMode_Asynchronous
 * and LwSciWrap::BufAttr, should call srcSendPacketAttr() interface of consumer block to
 * send the producer packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, srcSendPacketAttr_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    prodSendPacketAttr();
    consSendPacketAttr();

    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    LwSciStreamBlockPacketElementCount(pool, reconciledElementCount);

    // Pool sets packet requirements
    for (uint32_t i = 0U; i < reconciledElementCount; i++) {

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
        LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

        /*test code*/
        EXPECT_CALL(*consumerPtr[0], srcSendPacketAttr(_, i, i, LwSciStreamElementMode_Asynchronous,
              Ref(wrapBufAttr)))
             .Times(1)
             .WillRepeatedly(Return());

        queuePtr->srcSendPacketAttr(Block::singleConn_access, i, i,
                                   LwSciStreamElementMode_Asynchronous,
                                   wrapBufAttr);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
    }
}

/**
 * @testname{queue_unit_buf_setup_test.srcSendPacketAttr_Success2}
 * @testcase{21188259}
 * @verify{19471335}
 * @testpurpose{Test positive scenario of Queue::srcSendPacketAttr(),
 * where the elemSyncMode is equal to LwSciStreamElementMode_Immediate.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool through LwSciStreamBlockPacketElementCount() API sets the element count.
 *
 *   The call of Queue::srcSendPacketAttr() API from queue object,
 * with valid srcIndex of Block::singleConn, elemIndex less than count set using
 * LwSciStreamBlockPacketElementCount(), elemType of
 * NUM_PACKET_ELEMENTS, elemSyncMode of LwSciStreamElementMode_Immediate
 * and LwSciWrap::BufAttr, should call srcSendPacketAttr() interface of consumer block to
 * send the producer packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - consumer::srcSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, srcSendPacketAttr_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    prodSendPacketAttr();
    consSendPacketAttr();

    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    LwSciStreamBlockPacketElementCount(pool, reconciledElementCount);

    // Pool sets packet requirements
    for (uint32_t i = 0U; i < reconciledElementCount; i++) {

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
        LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

        /*test code*/
        EXPECT_CALL(*consumerPtr[0], srcSendPacketAttr(_, i, i, LwSciStreamElementMode_Immediate,
              Ref(wrapBufAttr)))
             .Times(1)
             .WillRepeatedly(Return());

        queuePtr->srcSendPacketAttr(Block::singleConn_access, i, i,
                                   LwSciStreamElementMode_Immediate,
                                   wrapBufAttr);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
    }
}


/**
 * @testname{queue_unit_buf_setup_test.srcSendPacketAttr_StreamNotConnected}
 * @testcase{21188268}
 * @verify{19471335}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendPacketAttr(), where
 *   srcSendPacketAttr is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketAttr() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, srcSendPacketAttr_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendPacketAttr(Block::singleConn_access, 1, 1, LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

}

/**
 * @testname{queue_unit_buf_setup_test.srcSendPacketAttr_StreamBadSrcIndex}
 * @testcase{21188271}
 * @verify{19471335}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::srcSendPacketAttr(),
 * where srcSendPacketAttr is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::srcSendPacketAttr() API with invalid srcIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::srcSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, srcSendPacketAttr_StreamBadSrcIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->srcSendPacketAttr(ILWALID_CONN_IDX, 1, 1, LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketAttr_Success1}
 * @testcase{21188274}
 * @verify{19471359}
 * @testpurpose{Test positive scenario of Queue::dstSendPacketAttr(),
 * where the elemSyncMode is equal to LwSciStreamElementMode_Asynchronous.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketAttr() API from queue object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Asynchronous and LwSciWrap::BufAttr,
 * should call dstSendPacketAttr() interface of pool block to send the consumer
 * packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketAttr_Success1)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    // Pool sets packet requirements
    for (uint32_t i = 0U; i < elementCount; i++) {

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
        LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

       /*test code*/
        EXPECT_CALL(*poolPtr, dstSendPacketAttr(_, i, i, LwSciStreamElementMode_Asynchronous,
              Ref(wrapBufAttr)))
             .Times(1)
             .WillRepeatedly(Return());

        queuePtr->dstSendPacketAttr(Block::singleConn_access, i, i,
                                   LwSciStreamElementMode_Asynchronous,
                                   wrapBufAttr);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
    }
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketAttr_Success2}
 * @testcase{21188277}
 * @verify{19471359}
 * @testpurpose{Test positive scenario of Queue::dstSendPacketAttr(),
 * where the elemSyncMode is equal to LwSciStreamElementMode_Immediate.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketAttr() API from queue object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Immediate and LwSciWrap::BufAttr,
 * should call dstSendPacketAttr() interface of pool block to send the consumer
 * packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketAttr_Success2)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    using ::testing::Ref;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    queryMaxNumElements();

    // Pool sets packet requirements
    for (uint32_t i = 0U; i < elementCount; i++) {

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
        LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

       /*test code*/
        EXPECT_CALL(*poolPtr, dstSendPacketAttr(_, i, i, LwSciStreamElementMode_Immediate,
              Ref(wrapBufAttr)))
             .Times(1)
             .WillRepeatedly(Return());

        queuePtr->dstSendPacketAttr(Block::singleConn_access, i, i,
                                   LwSciStreamElementMode_Immediate,
                                   wrapBufAttr);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
    }
}


/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketAttr_StreamNotConnected}
 * @testcase{21188283}
 * @verify{19471359}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketAttr(), where
 *   dstSendPacketAttr is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketAttr() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketAttr_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendPacketAttr(Block::singleConn_access, 1, 1, LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketAttr_StreamBadDstIndex}
 * @testcase{21188322}
 * @verify{19471359}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketAttr(),
 * where dstSendPacketAttr is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketAttr() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketAttr()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketAttr_LwSciError_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendPacketAttr(ILWALID_CONN_IDX, 1, 1, LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketStatus_Success}
 * @testcase{21188325}
 * @verify{19471362}
 * @testpurpose{Test positive scenario of Queue::dstSendPacketStatus()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4) Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5) consumer queries the LwSciStreamEventType_PacketCreate event through
 *      LwSciStreamBlockEventQuery and gets packetHandle.
 *
 *   The call of Queue::dstSendPacketStatus() API from queue object,
 * with valid dstIndex of Block::singleConn, handle of LwSciStreamPacket and
 * packetStatus of LwSciError, should call dstSendPacketStatus() interface of pool block
 * to send the consumer's packet acceptance status.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendPacketStatus()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstsendPacketStatus_Success)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    // Choose pool's cookie and for new packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    EXPECT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));

    // Save the cookie-to-handle mapping
    poolCPMap.emplace(poolCookie, packetHandle);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t k = 0; k < elementCount; ++k) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
        LwSciStreamPoolPacketInsertBuffer(pool,
                                          packetHandle, k,
                                          poolElementBuf[k]);
     }

     // Producer receives PacketCreate event
     LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event);
     EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

     // Assign cookie to producer packet handle
     LwSciStreamPacket producerPacket = event.packetHandle;
     LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
     LwSciError producerError = LwSciError_Success;

     BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstSendPacketStatus(_, producerPacket, producerError))
               .Times(1)
               .WillRepeatedly(Return());

     queuePtr->dstSendPacketStatus(Block::singleConn_access, producerPacket, producerError);

     EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketStatus_StreamInternalError}
 * @testcase{21188328}
 * @verify{19471362}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketStatus(), where
 *   LwSciStreamPacket is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketStatus() API from queue object,
 * with invalid LwSciStreamPacket handle,
 * should cause LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketStatus_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    LwSciError producerError = LwSciError_Success;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

     BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

     queuePtr->dstSendPacketStatus(Block::singleConn_access, ILWALID_PACKET_HANDLE,
                                                 producerError);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketStatus_StreamNotConnected}
 * @testcase{21188331}
 * @verify{19471362}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketStatus(), where
 *   dstSendPacketStatus is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketAttr() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketStatus_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciStreamPacket producerPacket = event.packetHandle;
    LwSciError producerError = LwSciError_Success;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendPacketStatus(Block::singleConn_access, producerPacket, producerError);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

}

/**
 * @testname{queue_unit_buf_setup_test.dstSendPacketStatus_StreamBadDstIndex}
 * @testcase{21188334}
 * @verify{19471362}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketAttr(),
 * where dstSendPacketAttr is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendPacketAttr() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendPacketStatus_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    LwSciStreamEvent event;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    LwSciStreamPacket producerPacket = event.packetHandle;
    LwSciError producerError = LwSciError_Success;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendPacketStatus(ILWALID_CONN_IDX, producerPacket, producerError);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{queue_unit_packet_stream_test.dstDisconnect_Success}
 * @testcase{21188337}
 * @verify{19471317}
 * @testpurpose{Test positive scenario of Queue::dstDisconnect()}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstDisconnect() API from queue object,
 * with valid dstIndex of Block::singleConn, should ilwoke the
 * dstDisconnect() interface of the pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstDisconnect()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstDisconnect()}
 */
TEST_F(queue_unit_packet_stream_test, dstDisconnect_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_CALL(*poolPtr, dstDisconnect(Block::singleConn_access))
               .Times(1)
               .WillRepeatedly(Return());


    queuePtr->dstDisconnect(Block::singleConn_access);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{queue_unit_packet_stream_test.dstDisconnect_Failure}
 * @testcase{21188346}
 * @verify{19471317}
 * @testpurpose{Test negative scenario of Queue::dstDisconnect(), where
 * the block was never connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstDisconnect() API from queue object,
 * should returns false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstDisconnect()}
 */
TEST_F(queue_unit_packet_stream_test, dstDisconnect_Failure)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    /* Initial setup */
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr, connComplete())
               .WillRepeatedly(Return(false));

    queuePtr->dstDisconnect(Block::singleConn_access);

}

/**
 * @testname{queue_unit_packet_stream_test.dstDisconnect_StreamBadDstIndex}
 * @testcase{21188349}
 * @verify{19471317}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstDisconnect(),
 * where dstDisconnect is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstDisconnect() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstDisconnect()}
 */
TEST_F(queue_unit_packet_stream_test, dstDisconnect_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstDisconnect(ILWALID_CONN_IDX);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{queue_unit_packet_stream_test.disconnect_Success}
 * @testcase{21188352}
 * @verify{19471311}
 * @testpurpose{Test positive scenario of Queue::disconnect()}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::disconnect() API from queue object,
 * Disconnects the Source and Destination Blocks by calling the Block::disconnectSrc()
 * and Block::disconnectDst() interfaces.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectSrc()
 *      - Block::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::disconnect()}
 */
TEST_F(queue_unit_packet_stream_test, disconnect_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    EXPECT_EQ(LwSciError_Success,
                    queuePtr->disconnect());

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);

}

/**
 * @testname{queue_unit_test.getOutputConnectPoint_LwSciError_AccessDenied}
 * @testcase{21188361}
 * @verify{19471305}
 * @testpurpose{Test positive scenario of Queue::getOutputConnectPoint()}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::getOutputConnectPoint() API from queue object,
 * with valid of BlockPtr, should return LwSciError_AccessDenied.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   None.}
 * @verifyFunction{Queue::getOutputConnectPoint()}
 */
TEST_F (queue_unit_test, getOutputConnectPoint_LwSciError_AccessDenied)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr poolObj2;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    EXPECT_EQ (LwSciError_AccessDenied, queuePtr->getOutputConnectPoint(poolObj2));
}

/**
 * @testname{queue_unit_test.getInputConnectPoint_LwSciError_AccessDenied}
 * @testcase{21188367}
 * @verify{19471308}
 * @testpurpose{Test positive scenario of Queue::getOutputConnectPoint()}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::getInputConnectPoint() API from queue object,
 * with valid of BlockPtr, should return LwSciError_AccessDenied.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   None.}
 * @verifyFunction{Queue::getInputConnectPoint()}
 */
TEST_F (queue_unit_test, getInputConnectPoint_LwSciError_AccessDenied)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr poolObj2;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    EXPECT_EQ (LwSciError_AccessDenied, queuePtr->getInputConnectPoint(poolObj2));
}

/**
 * @testname{queue_unit_packet_stream_test.dstReusePacket_Success}
 * @testcase{21188373}
 * @verify{19471374}
 * @testpurpose{Test positive scenario of Queue::dstReusePacket()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Consumer acquires packet.
 *
 *   The call of Queue::dstReusePacket() API from queue object,
 * with valid dstIndex of Block::singleConn, LwSciStreamPacket (received during
 * LwSciStreamEventType_PacketCreate event), and FenceArray, should call dstReusePacket()
 * interface of pool block to send the consumer's LwSciStreamPacket and associated FenceArray.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstReusePacket()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstReusePacket()}
 */
TEST_F(queue_unit_packet_stream_test, dstReusePacket_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > consSyncCount[0])
                         ? totalConsSync : consSyncCount[0];
    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));
    LwSciStreamCookie cookie;

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer get a packet from the pool
    for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }

    EXPECT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));
    LwSciStreamPacket handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t j = 0U; j < consSyncCount[0]; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }
    EXPECT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketPresent(producer, handle, fences));

                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
                EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketAcquire(consumer[0],
                                                     &cookie, fences));
                handle = consCPMap[0][cookie];
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    /*test code*/
    EXPECT_CALL(*poolPtr, dstReusePacket(_, handle, Ref(wrapFences)))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->dstReusePacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_packet_stream_test.dstReusePacket_StreamInternalError1}
 * @testcase{21188376}
 * @verify{19471374}
 * @testpurpose{Test negative scenario of Queue::dstReusePacket(), where
 * Location update for the packet instance is failed.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks,
 *      where pool is created with num packets of 1.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and consumer exchange sync attributes and sync count.
 *   3) Use mock Block::pktFindByHandle() to get the packet object from the packet handle
 *   and ilwoke Packet::locationUpdate() to change the location of packet handle to
 *   Packet::Location::Downstream.
 *
 *   The call of Queue::dstReusePacket() API from queue object,
 * should cause LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectEvent()
 *      - Block::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstReusePacket()}
 */
TEST_F(queue_unit_packet_stream_test, dstReusePacket_StreamInternalError1)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;


    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > consSyncCount[0])
                         ? totalConsSync : consSyncCount[0];
    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));
    LwSciStreamCookie cookie;

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer get a packet from the pool
    for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }

    EXPECT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));
    LwSciStreamPacket handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t j = 0U; j < consSyncCount[0]; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }
    EXPECT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketPresent(producer, handle, fences));

                EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
                EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);
                ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketAcquire(consumer[0],
                                                     &cookie, fences));
                handle = consCPMap[0][cookie];
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    ///////////////////////
    //     Test code     //
    ///////////////////////
    PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle) };

    pkt->locationUpdate(Packet::Location::Downstream, Packet::Location::Upstream);

    queuePtr->dstReusePacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{queue_unit_packet_stream_test.dstReusePacket_Failure}
 * @testcase{21188382}
 * @verify{19471374}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstReusePacket(), where
 *   queue block is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstReusePacket() API from queue object,
 * should returns false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstReusePacket()}
 */
TEST_F(queue_unit_packet_stream_test, dstReusePacket_Failure)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);

    // Connect stream
    //connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    FenceArray wrapFences { };
    LwSciStreamPacket handle;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr, connComplete())
               .WillRepeatedly(Return(false));

    queuePtr->dstReusePacket(Block::singleConn_access, handle, wrapFences);
}

/**
 * @testname{queue_unit_packet_stream_test.dstReusePacket_StreamBadDstIndex}
 * @testcase{21216894}
 * @verify{19471374}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstReusePacket(),
 * where dstReusePacket is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstReusePacket() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstReusePacket()}
 */
TEST_F(queue_unit_packet_stream_test, dstReusePacket_StreamBadDstIndex)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    /* Initial setup */
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);

    // Connect stream
    connectStream();

    FenceArray wrapFences { };
    LwSciStreamPacket handle;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    queuePtr->dstReusePacket(ILWALID_CONN_IDX, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);

}

/**
 * @testname{queue_unit_packet_stream_test.dstReusePacket_StreamInternalError2}
 * @testcase{21216897}
 * @verify{19471374}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstReusePacket(), where
 *   LwSciStreamPacket is invalid.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *
 *   The call of Queue::dstReusePacket() API from queue object,
 * with invalid LwSciStreamPacket handle,
 * should cause LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstReusePacket()}
 */
TEST_F(queue_unit_packet_stream_test, dstReusePacket_StreamInternalError2)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    /* Initial setup */
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer get a packet from the pool
    for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }

    EXPECT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));

    LwSciStreamPacket handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t j = 0U; j < prodSyncCount; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }
    EXPECT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketPresent(producer, handle, fences));

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < consSyncCount[0]; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstReusePacket(Block::singleConn_access, ILWALID_PACKET_HANDLE, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    free(fences);
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendElementStatus_Success1}
 * @testcase{21188391}
 * @verify{19471365}
 * @testpurpose{Test positive scenario of Queue::dstSendElementStatus()
 * where the elemIndex is equal to 2.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4) Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5) consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   6) consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *
 *   The call of Queue::dstSendElementStatus() API from queue object,
 * with valid dstIndex of Block::singleConn, handle of LwSciStreamPacket(received during
 * LwSciStreamEventType_PacketCreate event), elemIndex(received during
 * LwSciStreamEventType_PacketElement) and elemStatus of LwSciError_Success,
 * should call dstSendElementStatus() interface of pool block to send
 * the consumer's element acceptance status.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement and boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendElementStatus()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendElementStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstsendElementStatus_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    // Choose pool's cookie and for new packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    EXPECT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));

    // Save the cookie-to-handle mapping
    poolCPMap.emplace(poolCookie, packetHandle);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t k = 0; k < elementCount; ++k) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
        EXPECT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketInsertBuffer(pool,
                                          packetHandle, k,
                                          poolElementBuf[k]));
     }

     // Producer receives PacketCreate event
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

     // Assign cookie to producer packet handle
     LwSciStreamPacket producerPacket = event.packetHandle;
     LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
     LwSciError producerError = LwSciError_Success;

     BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     // Producer accepts a packet provided by the pool
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
          producer, producerPacket, producerCookie, producerError));

     // Save the cookie-to-handle mapping
     prodCPMap.emplace(producerCookie, producerPacket);

     for (uint32_t k = 0; k < elementCount; ++k) {
          BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
          EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
          EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

          producerPacket = prodCPMap[event.packetCookie];

          /*test code*/
          EXPECT_CALL(*poolPtr, dstSendElementStatus(_, producerPacket, event.index, producerError))
                     .Times(1)
                     .WillRepeatedly(Return());

          queuePtr->dstSendElementStatus(Block::singleConn_access,
                        producerPacket, event.index, producerError);

          LwSciBufObjFree(event.bufObj);

          EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
     }
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendElementStatus_Success2}
 * @testcase{21216903}
 * @verify{19471365}
 * @testpurpose{Test positive scenario of Queue::dstSendElementStatus()
 * where the elemIndex is equal to elementcount-1.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3) Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4) Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5) consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   6) consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *
 *   The call of Queue::dstSendElementStatus() API from queue object,
 * with valid dstIndex of Block::singleConn, handle of LwSciStreamPacket(received during
 * LwSciStreamEventType_PacketCreate event), elemIndex(received during
 * LwSciStreamEventType_PacketElement) and elemStatus of LwSciError_Success,
 * should call dstSendElementStatus() interface of pool block to send
 * the consumer's element acceptance status.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - pool::dstSendElementStatus()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendElementStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstsendElementStatus_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    packetAttrSetup();

    // Choose pool's cookie and for new packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    EXPECT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));

    // Save the cookie-to-handle mapping
    poolCPMap.emplace(poolCookie, packetHandle);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t k = 0; k < elementCount; ++k) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
        EXPECT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketInsertBuffer(pool,
                                          packetHandle, k,
                                          poolElementBuf[k]));
     }

     // Producer receives PacketCreate event
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

     // Assign cookie to producer packet handle
     LwSciStreamPacket producerPacket = event.packetHandle;
     LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
     LwSciError producerError = LwSciError_Success;

     BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     // Producer accepts a packet provided by the pool
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
          producer, producerPacket, producerCookie, producerError));

     // Save the cookie-to-handle mapping
     prodCPMap.emplace(producerCookie, producerPacket);

     for (uint32_t k = 0; k < elementCount; ++k) {
          BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
          EXPECT_EQ(LwSciError_Success,
                    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
          EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

          producerPacket = prodCPMap[event.packetCookie];

          /*test code*/
          EXPECT_CALL(*poolPtr, dstSendElementStatus(_, producerPacket, MAX_ELEMENT_PER_PACKET - 1,
                      producerError))
                     .Times(1)
                     .WillRepeatedly(Return());

          queuePtr->dstSendElementStatus(Block::singleConn_access,
                        producerPacket, MAX_ELEMENT_PER_PACKET - 1, producerError);

          LwSciBufObjFree(event.bufObj);

          EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
     }
}


/**
 * @testname{queue_unit_buf_setup_test.dstSendElementStatus_StreamNotConnected}
 * @testcase{21188397}
 * @verify{19471365}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendElementStatus(), where
 *   dstSendElementStatus is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendElementStatus() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendElementStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendElementStatus_StreamNotConnected)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    /* Initial setup */
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    // Assign cookie to producer packet handle
    LwSciStreamPacket producerPacket;
    LwSciError producerError = LwSciError_Success;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendElementStatus(Block::singleConn_access,
                    producerPacket, 0, producerError);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{queue_unit_buf_setup_test.dstSendElementStatus_StreamBadDstIndex}
 * @testcase{21188400}
 * @verify{19471365}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendElementStatus(),
 * where dstSendElementStatus is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstSendElementStatus() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendElementStatus()}
 */
TEST_F(queue_unit_buf_setup_test, dstSendElementStatus_StreamBadDstIndex)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    /* Initial setup */
    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Assign cookie to producer packet handle
    LwSciStreamPacket producerPacket;
    LwSciError producerError = LwSciError_Success;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstSendElementStatus(ILWALID_CONN_IDX,
                    producerPacket, 0, producerError);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{queue_unit_test.dstAcquirePacket_false}
 * @testcase{21188403}
 * @verify{19471368}
 * @testpurpose{Test negative scenario of Queue::dstAcquirePacket(), where
 * no packet is available for reading.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstAcquirePacket() API from queue object,
 * with valid dstIndex of Block::singleConn, should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectEvent()
 *      - Block::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F (queue_unit_packet_stream_test, dstAcquirePacket_false) {

    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;


    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    Payload acquiredPayload;


    EXPECT_EQ(false, queuePtr->dstAcquirePacket(Block::singleConn_access, acquiredPayload));

}


/**
 * @testname{queue_unit_packet_stream_test.dstAcquirePacket_true1}
 * @testcase{22571443}
 * @verify{19471368}
 * @testpurpose{Test positive scenario of Queue::dstAcquirePacket()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(fifo type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Consumer receives LwSciStreamEventType_PacketReady event.
 *
 *   The call of Queue::dstAcquirePacket() API from queue object,
 * with valid dstIndex of Block::singleConn and LwSciStreamPacket handle (received during
 * LwSciStreamEventType_PacketCreate event) along with the associated FenceArray,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectEvent()
 *      - Block::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F (queue_unit_packet_stream_test, dstAcquirePacket_true1) {

    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;


    // Create a fifo stream
    createBlocks(QueueType::Fifo);//add for fifo


    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                     ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < 1; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &cookie, fences));

        LwSciStreamPacket handle = prodCPMap[cookie];

        // Producer inserts a data packet into the stream
        for (uint32_t j = 0U; j < prodSyncCount; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle, fences));

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

        Payload acquiredPayload;

        for (uint32_t n = 0U; n < 1; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            EXPECT_EQ(true,
                 queuePtr->dstAcquirePacket(Block::singleConn_access, acquiredPayload));

        }

    } // End of sending frames

     free(fences);
}


/**
 * @testname{queue_unit_packet_stream_test.dstAcquirePacket_true2}
 * @testcase{21188415}
 * @verify{19471368}
 * @testpurpose{Test positive scenario of Queue::dstAcquirePacket()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue(mailbox type) and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Consumer receives LwSciStreamEventType_PacketReady event.
 *
 *   The call of Queue::dstAcquirePacket() API from queue object,
 * with valid dstIndex of Block::singleConn and LwSciStreamPacket handle (received during
 * LwSciStreamEventType_PacketCreate event) along with the associated FenceArray,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectEvent()
 *      - Block::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F (queue_unit_packet_stream_test, dstAcquirePacket_true2) {

    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                     ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < 1; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &cookie, fences));

        LwSciStreamPacket handle = prodCPMap[cookie];

        // Producer inserts a data packet into the stream
        for (uint32_t j = 0U; j < prodSyncCount; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle, fences));

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

        Payload acquiredPayload;

        for (uint32_t n = 0U; n < 1; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            EXPECT_EQ(true,
                 queuePtr->dstAcquirePacket(Block::singleConn_access, acquiredPayload));

        }

    } // End of sending frames

    free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.dstAcquirePacket_StreamInternalError1}
 * @testcase{21796013}
 * @verify{19471368}
 * @testpurpose{Test negative scenario of Queue::dstAcquirePacket(), where
 * Location update for the packet instance is failed.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources such that number of packets is 1.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Consumer gets the available packet handle through LwSciStreamEventType_PacketReady event.
 *   6) Use mock Block::pktFindByHandle() to get the packet object from the packet handle(received
 *   during LwSciStreamEventType_PacketCreate event) and ilwoke Packet::locationUpdate()
 *   to change the location of packet handle to Packet::Location::Upstream.
 *
 *   The call of Queue::dstAcquirePacket() API from queue object,
 * with valid dstIndex of Block::singleConn, and acquiredPayload of Payload,
 * should cause LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::pktFindByHandle() access changed to public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F (queue_unit_packet_stream_test, dstAcquirePacket_StreamInternalError1)
{

    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                     ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < 1; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &cookie, fences));

        LwSciStreamPacket handle = prodCPMap[cookie];

        // Producer inserts a data packet into the stream
        for (uint32_t j = 0U; j < prodSyncCount; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle, fences));

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

        Payload acquiredPayload;

        for (uint32_t n = 0U; n < 1; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

          PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle) };

          pkt->locationUpdate(Packet::Location::Queued, Packet::Location::Upstream);

            EXPECT_EQ(false, queuePtr->dstAcquirePacket(Block::singleConn_access,
                                                     acquiredPayload));
            EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            queue[0], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Error, event.type);
            EXPECT_EQ(LwSciError_StreamInternalError, event.error);
        }

    } // End of sending frames

     free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.dstAcquirePacket_StreamInternalError2}
 * @testcase{21796014}
 * @verify{19471368}
 * @testpurpose{Test negative scenario of Queue::dstAcquirePacket(), where
 * the no FenceArray(payload) associated with the next available packet.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources such that number of packets is 1.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Consumer receives LwSciStreamEventType_PacketReady event.
 *
 *   The call of Queue::dstAcquirePacket() API from queue object,
 * with valid dstIndex of Block::singleConn, and acquiredPayload of Payload with no FenceArray,
 * should cause LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::pktFindByHandle() access changed to public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F (queue_unit_packet_stream_test, dstAcquirePacket_StreamInternalError2)
{

    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                     ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < 1; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &cookie, fences));

        LwSciStreamPacket handle = prodCPMap[cookie];

        // Producer inserts a data packet into the stream
        for (uint32_t j = 0U; j < prodSyncCount; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle, fences));

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

        Payload acquiredPayload;

        for (uint32_t n = 0U; n < 1; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }



         PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle) };

         acquiredPayload.handle = pkt->handleGet();

         pkt->payloadGet(0, acquiredPayload);

         EXPECT_EQ(false, queuePtr->dstAcquirePacket(Block::singleConn_access,
                                                     acquiredPayload));
             EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            queue[0], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_Error, event.type);
            EXPECT_EQ(LwSciError_StreamInternalError, event.error);

        }

    } // End of sending frames

     free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.dstAcquirePacket_StreamBadDstIndex}
 * @testcase{21217221}
 * @verify{19471368}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstAcquirePacket(),
 * where dstAcquirePacket is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1) Create and connect the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstAcquirePacket() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F(queue_unit_packet_stream_test, dstAcquirePacket_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    Payload acquiredPayload;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    queuePtr->dstAcquirePacket(ILWALID_CONN_IDX, acquiredPayload);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{queue_unit_test.dstAcquirePacket_Failure}
 * @testcase{21217224}
 * @verify{19471368}
 * @testpurpose{Test negative scenario of Queue::dstAcquirePacket(), where
 *   dstAcquirePacket silently returns false.}
 * @testbehavior{
 * Setup:
 *   1) Creates the producer, pool, queue and consumer blocks,
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Queue::dstAcquirePacket() API from queue object,
 * should cause returns false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F(queue_unit_test, dstAcquirePacket_Failure)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;


    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    //connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    Payload acquiredPayload;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr, connComplete())
               .WillRepeatedly(Return(false));

    queuePtr->dstAcquirePacket(Block::singleConn_access, acquiredPayload);

}

/**
 * @testname{queue_unit_packet_stream_test.dstAcquirePacket}
 * @testcase{21611740}
 * @verify{19471368}
 * @testpurpose{Test positive scenario of Queue::dstAcquirePacket()}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Consumer receives LwSciStreamEventType_PacketReady event.
 *   6) Deleted the producer block.
 *
 *   The call of Queue::dstAcquirePacket() API from queue object,
 * with valid dstIndex of Block::singleConn and LwSciStreamPacket handle (received during
 * LwSciStreamEventType_PacketCreate event) along with the associated FenceArray and
 * deleted the producer block should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectEvent()
 *      - Block::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstAcquirePacket()}
 */
TEST_F (queue_unit_packet_stream_test, dstAcquirePacket) {

    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                     ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < 1; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketGet(producer, &cookie, fences));

        LwSciStreamPacket handle = prodCPMap[cookie];

        // Producer inserts a data packet into the stream
        for (uint32_t j = 0U; j < prodSyncCount; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamProducerPacketPresent(producer, handle, fences));

        BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

        Payload acquiredPayload;

        for (uint32_t n = 0U; n < 1; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(producer));
        producer = 0U;

        EXPECT_EQ(true,
            queuePtr->dstAcquirePacket(Block::singleConn_access, acquiredPayload));

        }

    } // End of sending frames

     free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_Success1}
 * @testcase{21611741}
 * @verify{19471404}
 * @testpurpose{Test success scenario of Mailbox::srcSendPacket() when there
 * is no packet in the queue.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, mailbox queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets and presents packet.
 *
 *   The call of Mailbox::srcSendPacket() API from queue object, receives the LwSciStreamPacket
 * from producer and informs the consumer about the new LwSciStreamPacket for acquisition
 * if the old one is not present.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    EXPECT_CALL(*consumerPtr[0], srcSendPacket(_, nullPacketHandle, _))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Timeout, LwSciStreamBlockEventQuery(
                consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}


/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_Success2}
 * @testcase{22060075}
 * @verify{19471404}
 * @testpurpose{Test success scenario of Mailbox::srcSendPacket() when
 * srcSendPacket() called multiple times with different packets to check
 * if old packet getting replaced with new one.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, mailbox queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 2U.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets and presents packet.
 *   4. Call Mailbox::srcSendPacket() API from queue object with packet handle.
 *   5. Get the next available packet.
 *
 *   Call of Mailbox::srcSendPacket() API again, from queue object with new
 *   packet should return the old packet to producer by calling dstReusePacket()
 *   with packet handle in step4 and should queue the new packet. Calling
 *   LwSciStreamConsumerPacketAcquire() should return the new packet.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, 1U, 3U);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle1 = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    queuePtr->srcSendPacket(Block::singleConn_access, handle1, wrapFences);

    ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

    LwSciStreamPacket handle2 = prodCPMap[cookie];

    EXPECT_CALL(*poolPtr, dstReusePacket(_, handle1, _))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacket(Block::singleConn_access, handle2, wrapFences);

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[0], EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     LwSciStreamCookie resultCookie;
     LwSciSyncFence fences1;

     EXPECT_EQ(LwSciError_Success,
     LwSciStreamConsumerPacketAcquire(consumer[0], &resultCookie, &fences1));

     EXPECT_EQ(consCPMap[0][resultCookie], handle2);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_Success3}
 * @testcase{22571444}
 * @verify{19471404}
 * @testpurpose{Test success scenario of Mailbox::srcSendPacket() when there
 * is a packet already in the queue.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, mailbox queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets and presents packet.
 *   4. Call Mailbox::srcSendPacket() API from queue object.
 *   4. Get the next available packet.
 *
 *   Call of Mailbox::srcSendPacket() API again, from queue object should
 *   ilwoke dstReusePacket() interface of the pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_Success3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

    handle = prodCPMap[cookie];

    EXPECT_CALL(*poolPtr, dstReusePacket(_, _, _))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}
/**
 * @testname{queue_unit_packet_stream_test.fifo_srcSendPacket_Success1}
 * @testcase{22571445}
 * @verify{19471404}
 * @testpurpose{Test success scenario of Fifo::srcSendPacket()}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, fifo queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets and presents packet.
 *
 *   The call of Fifo::srcSendPacket() API from queue object, receives the LwSciStreamPacket
 * from producer and informs the consumer about the new LwSciStreamPacket for acquisition.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, fifo_srcSendPacket_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a fifo stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }

     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    handle = prodCPMap[cookie];

    EXPECT_CALL(*consumerPtr[0], srcSendPacket(_, _, _))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

}

/**
 * @testname{queue_unit_packet_stream_test.fifo_srcSendPacket_Success3}
 * @testcase{22060078}
 * @verify{19471395}
 * @testpurpose{Test success scenario of Fifo::srcSendPacket() when multiple
 * packets are queued.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, fifo queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets a packet.
 *
 *   The call of Fifo::srcSendPacket() with first packet handle and call of
 *   Fifo::srcSendPacket() with second packet handle should queue the packets.
 *   Calling LwSciStreamConsumerPacketAcquire() multiple times should retrieve
 *   the packets in the order they are queued.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Fifo::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, fifo_srcSendPacket_Success3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a fifo stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle1 = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }

     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    queuePtr->srcSendPacket(Block::singleConn_access, handle1, wrapFences);

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle2 = prodCPMap[cookie];

     FenceArray wrapFences1 { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences1[j] = LwSciWrap::SyncFence(fences[j]);
     }

    queuePtr->srcSendPacket(Block::singleConn_access, handle2, wrapFences1);

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                consumer[0], EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     LwSciStreamCookie resultCookie;
     LwSciSyncFence fences1;

     EXPECT_EQ(LwSciError_Success,
     LwSciStreamConsumerPacketAcquire(consumer[0], &resultCookie, &fences1));

     EXPECT_EQ(consCPMap[0][resultCookie], handle1);

     EXPECT_EQ(LwSciError_Success,
     LwSciStreamConsumerPacketAcquire(consumer[0], &resultCookie, &fences1));

     EXPECT_EQ(consCPMap[0][resultCookie], handle2);

}

/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_StreamBadSrcIndex}
 * @testcase{21611742}
 * @verify{19471404}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Mailbox::srcSendPacket(),
 * where srcSendPacket() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Mailbox::srcSendPacket() API with invalid srcIndex of value
 * not equal to Block::singleConn which is 0 should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_StreamBadSrcIndex)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    FenceArray wrapFences { };
    LwSciStreamPacket handle;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    /*test code*/

    queuePtr->srcSendPacket(ILWALID_CONN_IDX, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);

}

/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_StreamNotConnected}
 * @testcase{21611743}
 * @verify{19471404}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Mailbox::srcSendPacket(),where
 *   srcSendPacket is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Creates the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Mailbox::srcSendPacket() API from queue object,
 * should cause LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_StreamNotConnected)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    /* Initial setup */
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Since this test operates on unconnected stream so a call to
    // connectStream() is omitted.

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }

    LwSciStreamPacket handle;

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < prodSyncCount; j++) {
       wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }
    ///////////////////////
    //     Test code     //
    ///////////////////////
    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

    free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_StreamInternalError1}
 * @testcase{21611745}
 * @verify{19471404}
 * @testpurpose{Test negative scenario of Mailbox::srcSendPacket(),
 * where invalid LwSciStreamPacket handle is passed as an input.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets and presents packet
 *
 *   The call of Mailbox::srcSendPacket() API from queue object,
 * with invalid LwSciStreamPacket handle and srcIndex value of Block::singleConn.
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_StreamInternalError1)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    /*test code*/

    queuePtr->srcSendPacket(Block::singleConn_access, ILWALID_PACKET_HANDLE, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_StreamInternalError2}
 * @testcase{21611746}
 * @verify{19471404}
 * @testpurpose{Test negative scenario of Mailbox::srcSendPacket(),where
 * Location update for the packet instance is failed in dstreusepacket().}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources such that number of packets is 1.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Use mock Block::pktFindByHandle() to get the packet object from the packet handle(received
 *   during LwSciStreamEventType_PacketCreate event) and ilwoke Packet::locationUpdate()
 *   to change the location of packet handle to Packet::Location::Downstream.
 *
 *   The call of Mailbox::srcSendPacket() API from queue object,
 * with valid LwSciStreamPacket handle,FenceArray and srcIndex value of Block::singleConn
 * when packet location is not Upstream(Packet::Location::Upstream)
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_StreamInternalError2)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;


    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > consSyncCount[0])
                         ? totalConsSync : consSyncCount[0];
    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));
    LwSciStreamCookie cookie;

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer get a packet from the pool
    for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }

    LwSciStreamPacket handle[1];

    for (uint32_t j = 0U; j < totalConsSync; j++) {
    EXPECT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));
     handle[j] = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t j = 0U; j < consSyncCount[0]; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }
    EXPECT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketPresent(producer, handle[j], fences));

    handle[j] = consCPMap[0][cookie];

    }

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    ///////////////////////
    //     Test code     //
    ///////////////////////
    PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle[0]) };
    PacketPtr const oldpkt { queuePtr->pktFindByHandle_access(handle[1]) };

    pkt->locationUpdate(Packet::Location::Queued, Packet::Location::Downstream);

    /*test code*/

    queuePtr->srcSendPacket(Block::singleConn_access, handle[0], wrapFences);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}


/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_IlwalidState}
 * @testcase{21611747}
 * @verify{19471404}
 * @testpurpose{Test negative scenario of Mailbox::srcSendPacket(),where
 * packet instance already in the queue.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources such that number of packets is 1.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Call Mailbox::srcSendPacket() API to associate the FenceArray with the
 *     LwSciStreamPacket.
 *   6) Use mock Block::pktFindByHandle() to get the packet instance and ilwoke
 *      Packet::locationUpdate() to change the location of packet handle to
 *      Packet::Location::Upstream.
 *
 *   The call of Mailbox::srcSendPacket() API from queue object,
 * with same valid LwSciStreamPacket handle,FenceArray and srcIndex value of
 * Block::singleConn when packet instance is already associated with a FenceArray
 * should result in LwSciError_IlwalidState event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::pktFindByHandle() access is changed to public.
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_IlwalidState)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle) };

    pkt->locationUpdate(Packet::Location::Queued, Packet::Location::Upstream);

    /*test code*/

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

    free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.mailbox_srcSendPacket_StreamInternalError3}
 * @testcase{21611746}
 * @verify{19471404}
 * @testpurpose{Test negative scenario of Mailbox::srcSendPacket(),where
 * Location update for the packet instance is failed.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources such that number of packets is 1.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Use mock Block::pktFindByHandle() to get the packet object from the packet handle(received
 *   during LwSciStreamEventType_PacketCreate event) and ilwoke Packet::locationUpdate()
 *   to change the location of packet handle to Packet::Location::Upstream.
 *
 *   The call of Mailbox::srcSendPacket() API from queue object,
 * with valid LwSciStreamPacket handle,FenceArray and srcIndex value of Block::singleConn
 * when packet location is not Downstream(Packet::Location::Downstream)
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Mailbox::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, mailbox_srcSendPacket_StreamInternalError3)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;


    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > consSyncCount[0])
                         ? totalConsSync : consSyncCount[0];
    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));
    LwSciStreamCookie cookie;

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer get a packet from the pool
    for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }


    LwSciStreamPacket handle[2];

    for (uint32_t j = 0U; j < totalConsSync; j++) {
    EXPECT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));
     handle[j] = prodCPMap[cookie];

    }

    // Producer inserts a data packet into the stream
    for (uint32_t j = 0U; j < consSyncCount[0]; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }
    EXPECT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketPresent(producer, handle[0], fences));


    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    ///////////////////////
    //     Test code     //
    ///////////////////////
    PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle[0]) };
    PacketPtr const oldpkt { queuePtr->pktFindByHandle_access(handle[1]) };

    pkt->locationUpdate(Packet::Location::Queued, Packet::Location::Upstream);

    /*test code*/

    queuePtr->srcSendPacket(Block::singleConn_access, handle[1], wrapFences);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.fifo_srcSendPacket_Success2}
 * @testcase{21611749}
 * @verify{19471395}
 * @testpurpose{Test success scenario of Fifo::srcSendPacket()}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets and presents packet.
 *
 *   The call of Fifo::srcSendPacket() API from queue object, receives the LwSciStreamPacket
 * from producer and informs the consumer about the new LwSciStreamPacket for acquisition.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Fifo::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, fifo_srcSendPacket_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    // Create a Fifo stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    EXPECT_CALL(*consumerPtr[0], srcSendPacket(_, nullPacketHandle, _))
               .Times(1)
               .WillRepeatedly(Return());

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Timeout, LwSciStreamBlockEventQuery(
                consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{queue_unit_packet_stream_test.fifo_srcSendPacket_StreamBadSrcIndex}
 * @testcase{21611750}
 * @verify{19471395}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Fifo::srcSendPacket(),
 * where srcSendPacket() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *
 *   The call of Fifo::srcSendPacket() API with invalid srcIndex of value
 * not equal to Block::singleConn which is 0 should result in LwSciError_StreamBadSrcIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Fifo::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, fifo_srcSendPacket_StreamBadSrcIndex)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    FenceArray wrapFences { };
    LwSciStreamPacket handle;
    LwSciStreamEvent event;

    // Create a Fifo stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

    /*test code*/
    queuePtr->srcSendPacket(ILWALID_CONN_IDX, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}



/**
 * @testname{queue_unit_packet_stream_test.fifo_srcSendPacket_StreamInternalError4}
 * @testcase{21611752}
 * @verify{19471395}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Fifo::srcSendPacket(),
 * where invalid LwSciStreamPacket handle is passed as input.}
 * @testbehavior{
 * Setup:
 *   1. Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2. Setup buffer and sync resources.
 *   3. Producer gets and presents packet
 *
 *   The call of Fifo::srcSendPacket() API from queue object,
 * with invalid LwSciStreamPacket handle and srcIndex value of Block::singleConn.
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Fifo::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, fifo_srcSendPacket_StreamInternalError4)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a Fifo stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }

     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    /*test code*/

    queuePtr->srcSendPacket(Block::singleConn_access, ILWALID_PACKET_HANDLE, wrapFences);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.fifo_srcSendPacket_StreamInternalError}
 * @testcase{21611753}
 * @verify{19471395}
 * @testpurpose{Test negative scenario of Fifo::srcSendPacket(),where
 * Location update for the packet instance is failed.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources such that number of packets is 1.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Use mock Block::pktFindByHandle() to get the packet object from the packet handle(received
 *   during LwSciStreamEventType_PacketCreate event) and ilwoke Packet::locationUpdate()
 *   to change the location of packet handle to Packet::Location::Upstream.
 *
 *   The call of Fifo::srcSendPacket() API from queue object,
 * with valid LwSciStreamPacket handle,FenceArray and srcIndex value of Block::singleConn
 * when packet location is not Upstream(Packet::Location::Upstream)
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Fifo::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, fifo_srcSendPacket_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;


    // Create a Fifo stream
    createBlocks(QueueType::Fifo);
    // Connect stream
    connectStream();
    // Create and exchange sync objects
    createSync();
    // Setup packet attr
    packetAttrSetup();
    // Create packets
    createPacket();
    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > consSyncCount[0])
                         ? totalConsSync : consSyncCount[0];
    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));
    LwSciStreamCookie cookie;

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer get a packet from the pool
    for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }

    EXPECT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));
    LwSciStreamPacket handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t j = 0U; j < consSyncCount[0]; j++) {
         fences[j] = LwSciSyncFenceInitializer;
    }
    EXPECT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketPresent(producer, handle, fences));

    EXPECT_EQ(LwSciError_Success,
         LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);
    ASSERT_EQ(LwSciError_Success,
         LwSciStreamConsumerPacketAcquire(consumer[0],
                                          &cookie, fences));
    handle = consCPMap[0][cookie];

    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };
    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    ///////////////////////
    //     Test code     //
    ///////////////////////
    PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle) };

    pkt->locationUpdate(Packet::Location::Downstream, Packet::Location::Queued);

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{queue_unit_packet_stream_test.fifo_srcSendPacket_IlwalidState}
 * @testcase{21611754}
 * @verify{19471395}
 * @testpurpose{Test negative scenario of Fifo::srcSendPacket(),where
 * packet instance already in the queue.}
 * @testbehavior{
 * Setup:
 *   1) Creates and connects the producer, pool, queue and consumer blocks.
 *   (Note that Pool block is configured with number of packets as 5.)
 *   2) set up synchronization and buffer resources such that number of packets is 1.
 *   3) Producer gets packet.
 *   4) Producer presents packet.
 *   5) Call Fifo::srcSendPacket() API to associate the FenceArray with the
 *      LwSciStreamPacket.
 *   6) Use mock Block::pktFindByHandle() to get the packet instance and ilwoke
 *      Packet::locationUpdate() to change the location of packet handle to
 *      Packet::Location::Upstream.
 *
 *   The call of Fifo::srcSendPacket() API from queue object,
 * with same valid LwSciStreamPacket handle,FenceArray and srcIndex value of
 * Block::singleConn when packet instance is already associated with a FenceArray
 * should result in LwSciError_IlwalidState event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::pktFindByHandle() access is changed to public.
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Fifo::srcSendPacket()}
 */
TEST_F(queue_unit_packet_stream_test, fifo_srcSendPacket_IlwalidState)
{
    /*Initial setup*/

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create a Fifo stream
    createBlocks(QueueType::Fifo);

    // Connect stream
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup packet attr
    packetAttrSetup();

    // Create packets
    createPacket();

    //check packet status
    checkPacketStatus();

    LwSciStreamEvent event;
    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences =
        static_cast<LwSciSyncFence*>(
            malloc(sizeof(LwSciSyncFence) * maxSync));

    LwSciStreamCookie cookie;

     // Pool sends packet ready event to producer
     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

     // Producer get a packet from the pool
     for (uint32_t j = 0U; j < totalConsSync; j++) {
         fences[j] = LwSciSyncFenceInitializer;
     }

     ASSERT_EQ(LwSciError_Success,
         LwSciStreamProducerPacketGet(producer, &cookie, fences));

     LwSciStreamPacket handle = prodCPMap[cookie];

     // Producer inserts a data packet into the stream
     for (uint32_t j = 0U; j < prodSyncCount; j++) {
        fences[j] = LwSciSyncFenceInitializer;
     }


     // send the packet
    BlockPtr queuePtr { Block::getRegisteredBlock(queue[0]) };

     FenceArray wrapFences { };
     for (uint32_t j { 0U }; j < prodSyncCount; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
     }

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    PacketPtr const pkt { queuePtr->pktFindByHandle_access(handle) };

    pkt->locationUpdate(Packet::Location::Queued, Packet::Location::Upstream);

    /*test code*/

    queuePtr->srcSendPacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                queue[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

    free(fences);
}

} // namespace LwSciStream

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
