//
// Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
/// @file

#include <limits>
#include <functional>
#include "sciwrap.h"
#include "lwscistream_common.h"
#include "block.h"
#include "pool.h"
#include "producer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include "lwscistream_panic_mock.h"

class block_unit_test: public LwSciStreamTest {
public:
   block_unit_test( ) {
       // initialization code here
   }
   void SetUp( ) {
       // code here will execute just before the test ensues
   }
   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }
   ~block_unit_test( )  {
       // cleanup any pending stuff, but no exceptions and no gtest
       // ASSERT* allowed.
   }
   // put in any custom data members that you need
};

namespace LwSciStream {

/**
 * @testname{block_unit_test.connSrcInitiate_Success}
 * @testcase{21796718}
 * @verify{19388994}
 * @testpurpose{Test success scenario of LwSciStream::Block::connSrcInitiate().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Get producer BlockPtr using getRegisteredBlock().
 *
 *   The call of LwSciStream::Block::connSrcInitiate() API from pool object
 * with producer BlockPtr as argument should return IndexRet, containing error
 * of LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::connSrcInitiate}
 */
TEST_F (block_unit_test, connSrcInitiate_Success) {
    /*Initial setup*/

    IndexRet ret;
    uint32_t numPackets {1};

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    BlockPtr poolObj {std::make_shared<Pool>(numPackets)};
    BlockPtr ProducerPtr { Block::getRegisteredBlock(producer) };
    /*Test code*/
    ret = poolObj->connSrcInitiate(ProducerPtr);
    EXPECT_EQ(ret.error, LwSciError_Success);
}

/**
 * @testname{block_unit_test.connSrcComplete_Connected}
 * @testcase{21796719}
 * @verify{19388994}
 * @verify{19500579}
 * @testpurpose{Test success scenario of LwSciStream::Block::connSrcComplete().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Get upstream and downstream LwSciStream::BlockPtr, using
 *      getRegisteredBlock(), with producer and consumer as
 *      arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint() and
 *      getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      , using connDstInitiate() and connSrcInitiate() respectively.
 *
 * The call of LwSciStream::Block::connDstComplete() followed by
 * LwSciStream::Block::connSrcComplete() API should result in
 * LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 *  directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::connSrcComplete}
 */
TEST_F (block_unit_test, connSrcComplete_Connected) {
    /*Initial setup */

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    /*Test code*/
    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.connSrcCancel_Connected}
 * @testcase{21796720}
 * @verify{18793752}
 * @testpurpose{Test success scenario of LwSciStream::Block::connSrcCancel().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      ,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream and Downstream connections by calling
 *      connDstComplete() and connSrcComplete().
 *   6) Query for LwSciStreamEventType_Connected event using
 *      LwSciStreamBlockEventQuery.
 *
 * The call of LwSciStream::Block::connSrcCancel() followed by
 * LwSciStream::Block::connSrcComplete() API should result in
 * LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::connSrcCancel}
 */
TEST_F (block_unit_test, connSrcCancel_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);

    /*Test code*/
    downstreamPtr->connSrcCancel(srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.connDstInitiate_Success}
 * @testcase{21796721}
 * @verify{19388994}
 * @verify{19500579}
 * @testpurpose{Test success scenario of LwSciStream::Block::connDstInitiate().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create pool BlockPtr type with number of packets of 1.
 *   3) Get the producer BlockPtr using getRegisteredBlock().
 *
 *   The call of LwSciStream::Block::connDstInitiate() API from pool object
 * with producer object as argument should return IndexRet, containing
 * error of LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::connDstInitiate}
 */
TEST_F (block_unit_test, connDstInitiate_Success) {
    /*Initial setup*/
    IndexRet ret;
    uint32_t numPackets {1};

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    BlockPtr poolObj {std::make_shared<Pool>(numPackets)};
    BlockPtr ProducerPtr { Block::getRegisteredBlock(producer) };

    /*Test code*/
    ret = poolObj->connDstInitiate(ProducerPtr);
    EXPECT_EQ(ret.error, LwSciError_Success);
}

/**
 * @testname{block_unit_test.connDstComplete_Connected}
 * @testcase{21796723}
 * @verify{19500579}
 * @testpurpose{Test success scenario of LwSciStream::Block::connDstComplete().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *
 * The call of LwSciStream::Block::connDstComplete() followed by
 * LwSciStream::Block::connSrcComplete() API should result in
 * LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::connDstComplete}
 */
TEST_F (block_unit_test, connDstComplete_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    /*Test code*/
    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.connDstCancel_Connected}
 * @testcase{21796724}
 * @verify{19500579}
 * @testpurpose{Test success scenario of LwSciStream::Block::connDstCancel().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream and Downstream connections by calling
 *      connDstComplete() and connSrcComplete().
 *   6) Query for LwSciStreamEventType_Connected event using
 *      LwSciStreamBlockEventQuery.
 *
 * The call of LwSciStream::Block::connDstCancel() followed by
 * LwSciStream::Block::connDstComplete() API should result in
 * LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::connDstCancel}
 */
TEST_F (block_unit_test, connDstCancel_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);

    /*Test code*/
    upstreamPtr->connDstCancel(dstReserved.index);
    downstreamPtr->connDstComplete(srcReserved.index, dstReserved.index);
    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.getOutputConnectPoint_Success}
 * @testcase{21796726}
 * @verify{18793800}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::getOutputConnectPoint(), where getOutputConnectPoint()
 * is called with BlockPtr.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Get consumer BlockPtr using getRegisteredBlock().
 *
 *   The call of LwSciStream::Block::getOutputConnectPoint() API from consumer
 * object with BlockPtr as argument returns LwSciError_Success and the retrieved
 * block instance should be equal to consumer BlockPtr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getOutputConnectPoint}
 */
TEST_F (block_unit_test, getOutputConnectPoint_Success) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    BlockPtr retrievedBlockPtr;
    BlockPtr consumerPtr { Block::getRegisteredBlock(consumer[0]) };

    /*Test Code*/
    EXPECT_EQ(LwSciError_Success,
            consumerPtr->getOutputConnectPoint(retrievedBlockPtr));

    // getOutputConnectPoint() returns its own block instance
    EXPECT_EQ(retrievedBlockPtr, consumerPtr);

}

/**
 * @testname{block_unit_test.getInputConnectPoint_Success}
 * @testcase{21796727}
 * @verify{18793815}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::getInputConnectPoint().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Get producer BlockPtr using getRegisteredBlock().
 *
 * The call of LwSciStream::Block::getInputConnectPoint() API from producer
 * BlockPtr should returns LwSciError_Success and the retrieved block
 * instance should be equal to producer BlockPtr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getInputConnectPoint}
 */
TEST_F (block_unit_test, getInputConnectPoint_Success) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    BlockPtr retrievedBlockPtr;
    BlockPtr ProducerPtr { Block::getRegisteredBlock(producer) };

    /*Test code*/
    EXPECT_EQ(LwSciError_Success,
            ProducerPtr->getInputConnectPoint(retrievedBlockPtr));

    // getInputConnectPoint() returns its own block instance
    EXPECT_EQ(retrievedBlockPtr, ProducerPtr);
}

/**
 * @testname{block_unit_test.dstNotifyConnection_Connected}
 * @testcase{21796728}
 * @verify{18794379}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::dstNotifyConnection().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream and Downstream connections by calling
 *      connDstComplete() and connSrcComplete().
 *
 * The call of LwSciStream::Block::dstNotifyConnection() API should result in
 * LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery() for valid and zero source indexes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::dstNotifyConnection}
 */
TEST_F (block_unit_test, dstNotifyConnection_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    /*Test code*/
    downstreamPtr->dstNotifyConnection(srcReserved.index);
    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);

    downstreamPtr->dstNotifyConnection(0U);
    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);

    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.srcNotifyConnection_Connected}
 * @testcase{21796729}
 * @verify{18794562}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcNotifyConnection(),
 * where srcNotifyConnection() is called with valid source and destination index.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively, using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream and Downstream connections by calling
 *      connDstComplete() and connSrcComplete().
 *
 * The call of LwSciStream::Block::srcNotifyConnection() API should result in
 * LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::srcNotifyConnection}
 */
TEST_F (block_unit_test, srcNotifyConnection_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    /*Test code*/
    upstreamPtr->srcNotifyConnection(dstReserved.index);
    LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event);

    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.getBlockType_Success1}
 * @testcase{21796731}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from pool object
 * should return BlockType::POOL type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success1) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_EQ(BlockType::POOL, poolPtr->getBlockType());
}

/**
 * @testname{block_unit_test.getBlockType_Success2}
 * @testcase{21796732}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from producer object
 * should return BlockType::PRODUCER type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success2) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    BlockPtr ProducerPtr { Block::getRegisteredBlock(producer) };
    EXPECT_EQ(BlockType::PRODUCER, ProducerPtr->getBlockType());
}

/**
 * @testname{block_unit_test.getBlockType_Success3}
 * @testcase{22060029}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from consumer object
 * should return BlockType::CONSUMER type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success3) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    BlockPtr ConsumerPtr { Block::getRegisteredBlock(consumer[0]) };
    EXPECT_EQ(BlockType::CONSUMER, ConsumerPtr->getBlockType());
}

/**
 * @testname{block_unit_test.getBlockType_Success4}
 * @testcase{22060032}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from queue object
 * should return BlockType::QUEUE type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success4) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    BlockPtr QueuePtr { Block::getRegisteredBlock(queue[0]) };
    EXPECT_EQ(BlockType::QUEUE, QueuePtr->getBlockType());
}

/**
 * @testname{block_unit_test.getBlockType_Success5}
 * @testcase{22060035}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from multicast object
 * should return BlockType::MULTICAST type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success5) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox, 2U);

    /*Test code*/
    BlockPtr MulticastPtr { Block::getRegisteredBlock(multicast) };
    EXPECT_EQ(BlockType::MULTICAST, MulticastPtr->getBlockType());
}

/**
 * @testname{block_unit_test.getBlockType_Success6}
 * @testcase{22060036}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from ipcsrc object
 * should return BlockType::IPCSRC type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success6) {

    initIpcChannel();

    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    BlockPtr IpcSrcPtr { Block::getRegisteredBlock(ipcsrc) };
    EXPECT_EQ(BlockType::IPCSRC, IpcSrcPtr->getBlockType());
}

/**
 * @testname{block_unit_test.getBlockType_Success7}
 * @testcase{22060039}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from ipcdst object
 * should return BlockType::IPCDST type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success7) {

    initIpcChannel();

    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    BlockPtr IpcDstPtr { Block::getRegisteredBlock(ipcdst) };
    EXPECT_EQ(BlockType::IPCDST, IpcDstPtr->getBlockType());
}

/**
 * @testname{block_unit_test.getBlockType_Success8}
 * @testcase{22060042}
 * @verify{18794640}
 * @testpurpose{Test success scenario of LwSciStream::Block::getBlockType().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::getBlockType() API from limiter object
 * should return BlockType::LIMITER type.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getBlockType}
 */
TEST_F (block_unit_test, getBlockType_Success8) {

    setLimiterEnable();

    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    BlockPtr LimiterPtr { Block::getRegisteredBlock(limiter[0]) };
    EXPECT_EQ(BlockType::LIMITER, LimiterPtr->getBlockType());
}

/**
 * @testname{block_unit_test.isInitSuccess_Success}
 * @testcase{21796733}
 * @verify{18794643}
 * @testpurpose{Test success scenario of LwSciStream::Block::isInitSuccess().}
 * @testbehavior{
 * Setup:
 *   Create pool BlockPtr type with number of packets of 1.
 *
 *   The call of LwSciStream::Block::isInitSuccess() API from pool object
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::isInitSuccess_public() API to internally
 *        call LwSciStream::Block::isInitSuccess() API.}
 * @verifyFunction{LwSciStream::Block::isInitSuccess}
 */
TEST_F (block_unit_test, isInitSuccess_Success) {
    /*Initial setup*/
    uint32_t numPackets {1};
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    BlockPtr poolObj {std::make_shared<Pool>(numPackets)};
    /*Test code*/
    EXPECT_EQ(true, poolPtr->isInitSuccess_public());
}

/**
 * @testname{block_unit_test.getHandle_Success}
 * @testcase{21796735}
 * @verify{18794646}
 * @testpurpose{Test success scenario of LwSciStream::Block::getHandle().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::getHandle() API from pool object
 * should return valid handle, which is non NULL.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getHandle}
 */
TEST_F (block_unit_test, getHandle_Success) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_EQ(true, (poolPtr->getHandle() != NULL));
}

/**
 * @testname{block_unit_test.blkMutexLock_Success}
 * @testcase{21796736}
 * @verify{18794673}
 * @testpurpose{Test success scenario of LwSciStream::Block::blkMutexLock().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::blkMutexLock() API from pool object with
 * with boolean argument of false should not cause lwscistreamPanic() to be called.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::blkMutexLock_public() API to internally
 *        call LwSciStream::Block::blkMutexLock() API.
 *   Following LwSciStream API calls are replaced with mocks:
 *      - lwscistreamPanic()}
 * @verifyFunction{LwSciStream::Block::blkMutexLock}
 */
TEST_F (block_unit_test, blkMutexLock_Success) {
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    lwscistreamPanicMock npm;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_CALL(npm, lwscistreamPanic_m())
               .Times(0)
               .WillRepeatedly(Return());
    poolPtr->blkMutexLock_public(false);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&npm));
}

/**
 * @testname{block_unit_test.dstXmitNotifyConnection_Connected}
 * @testcase{21796738}
 * @verify{18794676}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstXmitNotifyConnection().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream and Downstream connections by calling
 *      connDstComplete() and connSrcComplete().
 *
 * The call of LwSciStream::Block::dstXmitNotifyConnection() API call with
 * valid dstindex should result in LwSciStreamEventType_Connected event to
 * be queried through LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::dstXmitNotifyConnection_public() API to internally
 *       call LwSciStream::Block::dstXmitNotifyConnection() API.}
 * @verifyFunction{LwSciStream::Block::dstXmitNotifyConnection}
 */
TEST_F (block_unit_test, dstXmitNotifyConnection_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    LwSciStream::BlockPtr upstreamPtr { LwSciStream::Block::getRegisteredBlock(producer) };
    LwSciStream::BlockPtr downstreamPtr { LwSciStream::Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    LwSciStream::IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    LwSciStream::IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);

    /*Test code*/
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);
    poolPtr->dstXmitNotifyConnection_public(srcReserved.index);
    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);

    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.srcXmitNotifyConnection_Connected}
 * @testcase{21796739}
 * @verify{18794703}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcXmitNotifyConnection(),
 * where srcXmitNotifyConnection() is called with valid source and destination index.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream connection by calling connDstComplete().
 *
 * The call of LwSciStream::Block::connSrcComplete() followed by
 * srcXmitNotifyConnection() API  should result in
 * LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::SrcXmitNotifyConnection_public() API to
 *       internally call LwSciStream::Block::SrcXmitNotifyConnection() API.}
 * @verifyFunction{LwSciStream::Block::SrcXmitNotifyConnection}
 */
TEST_F (block_unit_test, srcXmitNotifyConnection_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    LwSciStream::BlockPtr upstreamPtr { LwSciStream::Block::getRegisteredBlock(producer) };
    LwSciStream::BlockPtr downstreamPtr { LwSciStream::Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    LwSciStream::IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    LwSciStream::IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);

    /*Test code*/
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);
    poolPtr->srcXmitNotifyConnection_public(dstReserved.index);
    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);

    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.setInitFail_Success}
 * @testcase{21796740}
 * @verify{18794712}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::setInitFail().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 * LwSciStream::Block::isInitSuccess() should return false after
 * LwSciStream::Block::setInitFail() API is called.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::setInitFail_public() API to internally
 *       call LwSciStream::Block::setInitFail() API.}
 * @verifyFunction{LwSciStream::Block::setInitFail}
 */
TEST_F (block_unit_test, setInitFail_Success) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    poolPtr->setInitFail_public();
    EXPECT_EQ(false, poolPtr->isInitSuccess());
}

/**
 * @testname{block_unit_test.eventPost_Connected}
 * @testcase{21796742}
 * @verify{18794715}
 * @testpurpose{Test success scenario of LwSciStream::Block::eventPost(), where
 * eventPost() is called with valid source and destination index.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream connection by calling connDstComplete().
 *
 * The call of LwSciStream::Block::connSrcComplete() followed by
 * LwSciStream::Block::eventPost() API call should result
 * in LwSciStreamEventType_Connected event to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::eventPost_public() API to internally
 *       call LwSciStream::Block::eventPost() API.}
 * @verifyFunction{LwSciStream::Block::eventPost}
 */
TEST_F (block_unit_test, eventPost_Connected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    LwSciStream::BlockPtr upstreamPtr { LwSciStream::Block::getRegisteredBlock(producer) };
    LwSciStream::BlockPtr downstreamPtr { LwSciStream::Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    LwSciStream::IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    LwSciStream::IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);

    /*Test code*/
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);
    poolPtr->eventPost_public(0U);
    LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event);

    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{block_unit_test.setErrorEvent}
 * @testcase{21796743}
 * @verify{18794733}
 * @testpurpose{Test success scenario of LwSciStream::Block::setErrorEvent()
 * when error event is not already set.}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::setErrorEvent() API with err set to
 * LwSciError_BadParameter from pool object
 * should return event of LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::setErrorEvent_public() API to internally
 *        call LwSciStream::Block::setErrorEvent() API
 * @verifyFunction{LwSciStream::setErrorEvent}
 */
TEST_F (block_unit_test, setErrorEvent) {
    /*Initial setup*/

    lwscistreamPanicMock npm;
    LwSciStreamEvent event;
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    /* Test code*/
    poolPtr->setErrorEvent_public(LwSciError_BadParameter, false);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{block_unit_test.validateWithError_StreamNotConnected}
 * @testcase{21796746}
 * @verify{18794850}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::validateWithError() API where
 * LwSciError_StreamNotConnected returned when stream is not connected.}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::validateWithError() API expected to
 * return LwSciStreamEvent of type LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::validateWithError_StreamNotConnected()
 *       API to internally call LwSciStream::Block::validateWithError() API.}
 * @verifyFunction{LwSciStream::Block::validateWithError}
 */
TEST_F (block_unit_test, validateWithError_StreamNotConnected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);
    /*Test code*/
    EXPECT_EQ(LwSciError_StreamNotConnected,
            poolPtr->validateWithError_StreamNotConnected(0U));
}

/**
 * @testname{block_unit_test.validateWithEvent_true}
 * @testcase{21796747}
 * @verify{18794928}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::validateWithEvent(), where validateWithEvent()
 * return true.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Connect stream by using connectStream() API.
 *   3) ValidateBits is configured with ValidateCtrl::SetupPhase and
 *      ValidateCtrl::CompleteQueried.
 *
 *   The call of LwSciStream::Block::validateWithEvent() API with pool object
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::validateWithEvent_public() API to internally
 *       call LwSciStream::Block::validateWithEvent() API.}
 * @verifyFunction{LwSciStream::validateWithEvent}
 */
TEST_F (block_unit_test, validateWithEvent_true) {
    /*Initial setup*/
    /* Create a mailbox stream */
    createBlocks(QueueType::Mailbox);
    connectStream();

    /*Test code*/
    EXPECT_EQ(true, poolPtr->validateWithEvent_public(0U));
}

/**
 * @testname{block_unit_test.disconnectSrc_Success}
 * @testcase{21796749}
 * @verify{18794937}
 * @verify{19471317}
 * @verify{19471314}
 * @verify{19471311}
 * @verify{21175203}
 * @verify{21175233}
 * @verify{21175260}
 * @verify{19780875}
 * @verify{19780902}
 * @verify{19780935}
 * @verify{19500630}
 * @verify{19500672}
 * @verify{19514187}
 * @verify{19514193}
 * @verify{19514199}
 * @testpurpose{Test success scenario of LwSciStream::Block::disconnectSrc().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Connect the stream using connectStream().
 *
 *   Call of LwSciStream::Block::disconnectSrc() API from pool object with
 * destination index as argument. Call of
 * LwSciStream::Block::connComplete API should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::disconnectSrc_public() API to internally
 *       call LwSciStream::Block::disconnectSrc() API.}
 * @verifyFunction{LwSciStream::Block::disconnectSrc}
 */
TEST_F (block_unit_test, disconnectSrc_Success) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    /*Test code*/
    poolPtr->disconnectSrc_public(0U);
    EXPECT_EQ(false, poolPtr->connComplete_public());
}

/**
 * @testname{block_unit_test.disconnectDst_Success}
 * @testcase{21796750}
 * @verify{18794940}
 * @verify{19471317}
 * @verify{19471314}
 * @verify{19471311}
 * @verify{19471368}
 * @verify{21175203}
 * @verify{21175233}
 * @verify{21175260}
 * @verify{19780875}
 * @verify{19780902}
 * @verify{19780935}
 * @verify{19389027}
 * @verify{19514187}
 * @verify{19514193}
 * @verify{19514199}
 * @testpurpose{Test success scenario of LwSciStream::Block::disconnectDst().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::disconnectDst() API from pool
 * object with destination index as argument should disconnect the stream then
 * calling Block::connComplete() should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::disconnectDst_public() API to internally
 *       call LwSciStream::Block::disconnectDst() API.}
 * @verifyFunction{LwSciStream::Block::disconnectDst}
 */
TEST_F (block_unit_test, disconnectDst_Success) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    poolPtr->disconnectDst_public(0U);
    EXPECT_EQ(false, poolPtr->connComplete_public());
}

/**
 * @testname{block_unit_test.disconnectEvent_Disconnected}
 * @testcase{21796751}
 * @verify{18794943}
 * @verify{19471368}
 * @verify{21175233}
 * @verify{19780935}
 * @verify{19389057}
 * @verify{19500672}
 * @testpurpose{Test success scenario of LwSciStream::Block::disconnectEvent()
 * API.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Declare a variable of type LwSciStreamEvent.
 *
 *   The call of LwSciStream::Block::disconnectEvent() API from pool object
 * is expected to raise LwSciStreamEvent of type
 * LwSciStreamEventType_Disconnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::disconnectEvent_public() API to internally
 *       call LwSciStream::Block::disconnectEvent() API.}
 * @verifyFunction{LwSciStream::Block::disconnectEvent}
 */
TEST_F (block_unit_test, disconnectEvent_Disconnected) {
    /*Initial setup*/

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    poolPtr->disconnectEvent_public();
    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);
}

/**
 * @testname{block_unit_test.getSrc_NoStreamPacket}
 * @testcase{21796753}
 * @verify{18794946}
 * @testpurpose{Test success scenario of LwSciStream::Block::getSrc() API.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Connect the stream with connectStream().
 *
 *   The call of LwSciStream::Block::dstAcquirePacket() API with object returns
 * from LwSciStream::Block::getSrc() API is expected to return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::getSrc_dstAcquirePacket() API to internally
 *       call LwSciStream::Block::dstAcquirePacket() API with reference returned
 *       from LwSciStream::Block::getSrc() API.}
 * @verifyFunction{LwSciStream::getSrc}
 */
TEST_F (block_unit_test, getSrc_NoStreamPacket) {
    /*Initial Setup*/
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    Payload acquiredPayload { };

    /*Test code*/
    EXPECT_EQ(false,
            poolPtr->getSrc_dstAcquirePacket(acquiredPayload));
}

/**
 * @testname{block_unit_test.connComplete_failure}
 * @testcase{21796754}
 * @verify{19471374}
 * @verify{19389051}
 * @verify{19500624}
 * @testpurpose{Test negative scenario of LwSciStream::Block::connComplete()
 * API.}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::connComplete() API from pool object
 * should return false, when the stream is not connected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::connComplete_public() API to internally
 *     call LwSciStream::Block::connComplete() API.}
 * @verifyFunction{LwSciStream::connComplete}
 */
TEST_F (block_unit_test, connComplete_Success) {
    /*Initial setup*/
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_EQ(false, poolPtr->connComplete_public());
}

/**
 * @testname{block_unit_test.pktFindByHandle_Success}
 * @testcase{21796755}
 * @verify{18796281}
 * @verify{19471362}
 * @verify{19471344}
 * @verify{19471374}
 * @verify{19780899}
 * @verify{19780896}
 * @verify{19780893}
 * @verify{19780923}
 * @verify{19780926}
 * @verify{19780929}
 * @verify{19780932}
 * @verify{19389015}
 * @verify{19389018}
 * @verify{19389021}
 * @verify{19389024}
 * @verify{19500606}
 * @verify{19500621}
 * @verify{19500624}
 * @verify{19506993}
 * @verify{19507020}
 * @verify{19513842}
 * @verify{19513845}
 * @verify{19513896}
 * @verify{19513908}
 * @verify{19513914}
 * @verify{19514184}
 * @testpurpose{Test success scenario of LwSciStream::Block::pktFindByHandle().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Call Block::pktCreate() with packet handle.
 *
 *   The call of Block::pktFindByHandle() API from pool BlockPtr
 * should return a non-NULL PacketPtr and retrieved packet handle should be equal
 * to handle passed in step-2.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktCreate_public() API to internally
 *   - call LwSciStream::Pool::pktCreate() API.
 *   - Configure LwSciStream::Pool::pktFindByHandle_public() API to internally
 *     call LwSciStream::Block::pktFindByHandle() API.}
 * @verifyFunction{LwSciStream::pktFindByHandle}
 */
TEST_F (block_unit_test, pktFindByHandle_Success) {
    /*Initial setup*/
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->pktCreate_public(packetHandle);
    PacketPtr pkt { poolPtr->pktFindByHandle_public(packetHandle, false) };
    EXPECT_NE(nullptr, pkt);
    EXPECT_EQ(packetHandle, pkt->handleGet());


}

/**
 * @testname{block_unit_test.pktRemove_Success}
 * @testcase{21796757}
 * @verify{18796278}
 * @verify{19780929}
 * @testpurpose{Test success scenario of LwSciStream::Block::pktRemove().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create a packet with pktCreate() API .
 *
 * Remove the LwSciStreamPacket using LwSciStream::Block::pktRemove() API.
 * Call of LwSciStream::Block::pktCreate() API with the same LwSciStreamPacket,
 * should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktCreate_public() API to internally
 *   - call LwSciStream::Pool::pktCreate() API.
 *   - Configure LwSciStream::Pool::pktRemove_public() API to internally
 *     call LwSciStream::Block::pktRemove() API.}
 * @verifyFunction{LwSciStream::pktRemove}
 */
TEST_F (block_unit_test, pktRemove_Success) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
                    = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    poolPtr->pktCreate_public(packetHandle);

    /*Test code*/
    poolPtr->pktRemove_public(packetHandle);
    EXPECT_EQ(LwSciError_Success, poolPtr->pktCreate_public(packetHandle));
}

/**
 * @testname{block_unit_test.pktPendingEvent_true}
 * @testcase{21796758}
 * @verify{18796287}
 * @testpurpose{Test success scenario of LwSciStream::Block::pktPendingEvent(),
 * when there is a packet-related LwSciStreamEvent is found .}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create and initialise Cookie, packetHandle respectively.
 *   3) Declare event of LwSciStreamEvent type.
 *
 *   The call of LwSciStream::Block::pktCreate() followed by
 * LwSciStream::Block::pktPendingEvent() API from producer object
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktCreate_public() API to internally
 *   - call LwSciStream::Block::pktCreate() API.
 *   - Configure LwSciStream::Pool::pktPendingEvent_public() API to internally
 *     call LwSciStream::Block::pktPendingEvent() API.}
 * @verifyFunction{LwSciStream::pktPendingEvent}
 */
TEST_F (block_unit_test, pktPendingEvent_false1) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie Cookie
                = static_cast<LwSciStreamCookie>(100);
    LwSciStreamPacket packetHandle {~Cookie};
    LwSciStreamEvent event;

    /*Test code*/
    producerPtr->pktCreate_public(packetHandle, Cookie);
    EXPECT_EQ(true,
            producerPtr->pktPendingEvent_public(event, 1U));
}

/**
 * @testname{block_unit_test.pktCreate_Success}
 * @testcase{21796761}
 * @verify{18796275}
 * @verify{19471338}
 * @verify{19506984}
 * @testpurpose{Test success scenario of LwSciStream::Block::pktCreate().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Declare and initialize producerCookie and packetHandle.
 *
 *   The call to LwSciStream::Block::pktCreate() API from producer object
 * should return LwSciError_Success and querying for pending event
 * should return LwSciStreamEventType_PacketCreate.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Producer::pktCreate_public() API to internally
 *     call LwSciStream::Block::pktCreate() API.}
 * @verifyFunction{LwSciStream::pktCreate}
 */
TEST_F (block_unit_test, pktCreate_Success) {
    /*Initial setup*/

    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    connectStream();

    LwSciStreamCookie prodCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~prodCookie};
    /*Test code*/

    EXPECT_EQ(LwSciError_Success, producerPtr->pktCreate_public(packetHandle, prodCookie));

    PacketPtr pkt { producerPtr->pktFindByHandle_public(packetHandle, false) };

    EXPECT_NE(nullptr, pkt);
    EXPECT_EQ(true, pkt->pendingEvent(event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

}

/**
 * @testname{block_unit_test.pktFindByCookie_Success}
 * @testcase{21796762}
 * @verify{18796284}
 * @testpurpose{Test success scenario of LwSciStream::Block::pktFindByCookie().}
 * @testbehavior{
 * Setup:
 *   Creates the pool block and call to connect stream.
 *
 *   The call of LwSciStream::Block::pktCreate() API followed by
 * LwSciStream::Block::pktFindByCookie() API from pool object should
 * return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktCreate_public() API to internally
 *   - call LwSciStream::Pool::pktCreate() API.
 *   - Configure LwSciStream::Pool::pktFindByCookie_public() API to internally
 *     call LwSciStream::Block::pktFindByCookie() API.}
 * @verifyFunction{LwSciStream::pktFindByCookie}
 */
TEST_F (block_unit_test, pktFindByCookie_Success) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);
    connectStream();
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(100);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->pktCreate_public(packetHandle, poolCookie);
    EXPECT_EQ(true,
              (poolPtr->pktFindByCookie_public(poolCookie) != NULL));
}

/**
 * @testname{block_unit_test.sendSyncAttr_NotImplemented}
 * @testcase{21796764}
 * @verify{18793824}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendSyncAttr().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Creates syncAttr of LwSciWrap::SyncAttr type
 *
 *   The call to LwSciStream::Block::sendSyncAttr() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::sendSyncAttr}
 */
TEST_F (block_unit_test, sendSyncAttr_NotImplemented) {
    /*Initial setup*/
    consSynchronousOnly = false;
    cpuWaiterAttrList(syncModule, consSyncAttrList);

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
    poolPtr->sendSyncAttr(consSynchronousOnly, syncAttr));

}

/**
 * @testname{block_unit_test.sendSyncCount_NotImplemented}
 * @testcase{21796766}
 * @verify{18793845}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendSyncCount().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::sendSyncCount() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::sendSyncCount}
 */
TEST_F (block_unit_test, sendSyncCount_NotImplemented) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
    poolPtr->sendSyncCount(1U, true));
}

/**
 * @testname{block_unit_test.sendSyncDesc_NotImplemented}
 * @testcase{21796767}
 * @verify{18793983}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendSyncDesc().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Creates SyncObj of LwSciWrap::SyncObj type.
 *
 *   The call to LwSciStream::Block::sendSyncDesc() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::sendSyncDesc}
 */
TEST_F (block_unit_test, sendSyncDesc_NotImplemented) {
    /*Initial setup*/
    consSynchronousOnly = false;
    cpuWaiterAttrList(syncModule, consSyncAttrList);

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncObj SyncObj { consSyncObjs[0][0] };

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->sendSyncDesc(1U, SyncObj));
}

/**
 * @testname{block_unit_test.sendPacketElementCount_NotImplemented}
 * @testcase{21796769}
 * @verify{18794001}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendPacketElementCount().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::sendPacketElementCount() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::sendPacketElementCount_public() API to internally
 *     call LwSciStream::Block::sendPacketElementCount() API.}
 * @verifyFunction{LwSciStream::Block::sendPacketElementCount}
 */
TEST_F (block_unit_test, sendPacketElementCount_NotImplemented) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->sendPacketElementCount_public(0U, 0U));
}

/**
 * @testname{block_unit_test.sendPacketAttr_NotImplemented}
 * @testcase{21796770}
 * @verify{18794007}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendPacketAttr().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::sendPacketAttr() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::sendPacketAttr_public() API to internally
 *     call LwSciStream::Block::sendPacketAttr() API.}
 * @verifyFunction{LwSciStream::Block::sendPacketAttr}
 */
TEST_F (block_unit_test, sendPacketAttr_NotImplemented) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::BufAttr wrapBufAttr{};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->sendPacketAttr_public(0U, 0U,
                    LwSciStreamElementMode_Asynchronous, wrapBufAttr));
}

/**
 * @testname{block_unit_test.sendPacketStatus_NotImplemented}
 * @testcase{21796771}
 * @verify{18794190}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendPacketStatus().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::sendPacketStatus() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::sendPacketStatus_public() API to internally
 *     call LwSciStream::Block::sendPacketStatus() API.}
 * @verifyFunction{LwSciStream::Block::sendPacketStatus}
 */
TEST_F (block_unit_test, sendPacketStatus_NotImplemented) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->sendPacketStatus_public(packetHandle, poolCookie,
                    LwSciError_NotImplemented));
}

/**
 * @testname{block_unit_test.sendElementStatus_NotImplemented}
 * @testcase{21796774}
 * @verify{18794247}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendElementStatus().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::sendElementStatus() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::sendElementStatus_public() API to internally
 *     call LwSciStream::Block::sendElementStatus() API.}
 * @verifyFunction{LwSciStream::Block::sendElementStatus}
 */
TEST_F (block_unit_test, sendElementStatus_NotImplemented) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->sendElementStatus_public(packetHandle, 0U,
                    LwSciError_NotImplemented));
}

/**
 * @testname{block_unit_test.registerPacketBuffer_NotImplemented}
 * @testcase{21796775}
 * @verify{18794286}
 * @testpurpose{Test default scenario of LwSciStream::Block::registerPacketBuffer().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::registerPacketBuffer() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::registerPacketBuffer_public() API to internally
 *     call LwSciStream::Block::registerPacketBuffer() API.}
 * @verifyFunction{LwSciStream::Block::registerPacketBuffer}
 */
TEST_F (block_unit_test, registerPacketBuffer_NotImplemented) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufObj wrapBufObj{};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->registerPacketBuffer_public(packetHandle, 0U, wrapBufObj));
}

/**
 * @testname{block_unit_test.getEvent_Success}
 * @testcase{21796776}
 * @verify{18794355}
 * @testpurpose{Test success scenario of LwSciStream::Block::getEvent() API
 * where getEvent() is called with event of type LwSciStreamEvent and time
 * in usec.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream and Downstream connections by calling
 *      connDstComplete() and connSrcComplete().
 *
 *   The call to LwSciStream::Block::getEvent() API is expected to return
 * LwSciError_Success and retrieved event should be equal to
 * LwSciStreamEventType_Connected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getEvent}
 */
TEST_F (block_unit_test, getEvent_Success) {
    /*Initial setup*/

    uint32_t numPackets {1};

    // Create a mailbox stream
    LwSciStreamEvent event;
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    /*Test code*/
    BlockPtr poolObj {std::make_shared<Pool>(numPackets)};
    EXPECT_EQ(LwSciError_Success, poolPtr->getEvent(200U, event));

    EXPECT_EQ(event.type, LwSciStreamEventType_Connected);

}

/**
 * @testname{block_unit_test.getDst_NoStreamPacket}
 * @testcase{21796778}
 * @verify{18794949}
 * @testpurpose{Test success scenario of LwSciStream::Block::getDst().}
 * LwSciError_NoStreamPacket.
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Connect the stream with connectStream().
 *
 *   The call of LwSciStream::Block::srcGetPacket() API with object returned
 * from LwSciStream::Block::getDst() API is expected to return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::getDst_srcGetPacket() API to internally
 *     call LwSciStream::Block::srcGetPacket() API with object returns from
 *     LwSciStream::Pool::getDst() API.}
 * @verifyFunction{LwSciStream::Block::getDst}
 */
TEST_F (block_unit_test, getDst_NoStreamPacket) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);
    connectStream();
    Payload reusablePayload { };

    /*Test code*/
    EXPECT_EQ(false,
            poolPtr->getDst_srcGetPacket(reusablePayload));
}

/**
 * @testname{block_unit_test.disconnect_NotImplemented}
 * @testcase{21796780}
 * @verify{18794349}
 * @testpurpose{Test success scenario of LwSciStream::Block::disconnect().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call to LwSciStream::Block::disconnect() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::disconnect_public() API to internally
 *   - call LwSciStream::Block::disconnect() API.}
 * @verifyFunction{LwSciStream::Block::disconnect}
 */
TEST_F (block_unit_test, disconnect_NotImplemented) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented, poolPtr->disconnect_public());
}

/**
 * @testname{block_unit_test.getPacket_NotImplemented}
 * @testcase{21796782}
 * @verify{18794316}
 * @testpurpose{Test default scenario of LwSciStream::Block::getPacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Allocate dynamic memory for fences of type LwSciSyncFence.
 *   3) Creates dummy cookie of type LwSciStreamCookie.
 *
 *   The call to LwSciStream::Block::getPacket() API from pool object should
 * return LwSciError is of type LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::getPacket_public() API to internally
 *   - call LwSciStream::Block::getPacket() API.}
 * @verifyFunction{LwSciStream::Block::getPacket}
 */
TEST_F (block_unit_test, getPacket_NotImplemented) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    LwSciSyncFence *fences =
            static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence)));
    LwSciStreamCookie cookie;

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->getPacket_public(cookie, fences));
}

/**
 * @testname{block_unit_test.acquirePacket_NotImplemented}
 * @testcase{21796783}
 * @verify{18794337}
 * @testpurpose{Test default scenario of LwSciStream::Block::acquirePacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Allocate dynamic memory for fences of type LwSciSyncFence.
 *   3) Creates poolCookie of LwSciStreamCookie type.
 *   4) Creates packetHandle of type LwSciStreamPacket.
 *
 *   The call to LwSciStream::Block::acquirePacket() API from pool object should
 * return LwSciError is of type LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::acquirePacket_public() API to internally
 *   - call LwSciStream::Block::acquirePacket() API.}
 * @verifyFunction{LwSciStream::Block::acquirePacket}
 */
TEST_F (block_unit_test, acquirePacket_NotImplemented) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    LwSciSyncFence *fences =
            static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence)));
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->acquirePacket_public(packetHandle, fences));
}

/**
 * @testname{block_unit_test.sendPacket_NotImplemented}
 * @testcase{21796785}
 * @verify{18794301}
 * @testpurpose{Test default scenario of LwSciStream::Block::sendPacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Allocate dynamic memory for fences of type LwSciSyncFence.
 *   3) Creates poolCookie of LwSciStreamCookie type.
 *
 *   The call to LwSciStream::Block::sendPacket() API from pool object should
 * return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::sendPacket_public() API to internally
 *   - call LwSciStream::Block::sendPacket() API.}
 * @verifyFunction{LwSciStream::Block::sendPacket}
 */
TEST_F (block_unit_test, sendPacket_NotImplemented) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);

    LwSciSyncFence *fences =
            static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence)));
    LwSciStreamCookie cookie;

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->sendPacket_public(cookie, fences));
}

/**
 * @testname{block_unit_test.reusePacket_NotImplemented}
 * @testcase{21796786}
 * @verify{18794310}
 * @testpurpose{Test default scenario of LwSciStream::Block::reusePacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Allocate dynamic memory for fences of type LwSciSyncFence.
 *   3) Creates poolCookie of LwSciStreamCookie type.
 *
 * The call to LwSciStream::Block::reusePacket() API, via
 * pool object should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::reusePacket_public() API to internally
 *     call LwSciStream::Block::reusePacket() API.}
 * @verifyFunction{LwSciStream::Block::reusePacket}
 */
TEST_F (block_unit_test, reusePacket_NotImplemented) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    LwSciSyncFence *fences =
            static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence)));
    LwSciStreamCookie cookie;
    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
            poolPtr->reusePacket_public(cookie, fences));
}

/**
 * @testname{block_unit_test.createPacket_NotImplemented}
 * @testcase{21796787}
 * @verify{18794271}
 * @testpurpose{Test default scenario of LwSciStream::Block::createPacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Creates packet handle with poolCookie of type LwSciStreamPacket.
 *
 *   The call of LwSciStream::Block::createPacket() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::createPacket_public() API to internally
 *     call LwSciStream::Block::createPacket() API.}
 * @verifyFunction{LwSciStream::Block::createPacket}
 */
TEST_F (block_unit_test, createPacket_NotImplemented) {
    /*Initial setup*/
    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
    poolPtr->createPacket_public(poolCookie, packetHandle));
}

/**
 * @testname{block_unit_test.deletePacket_NotImplemented}
 * @testcase{21796788}
 * @verify{18794295}
 * @testpurpose{Test success scenario of LwSciStream::Block::deletePacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Creates packet handle with poolCookie of type LwSciStreamPacket.
 *
 *   The call of LwSciStream::Block::deletePacket() API from pool object
 * should return LwSciError_NotImplemented.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::deletePacket_public() API to internally
 *     call LwSciStream::Block::deletePacket() API.}
 * @verifyFunction{LwSciStream::Block::deletePacket}
 */
TEST_F (block_unit_test, deletePacket_NotImplemented) {
    /*Initial setup*/
    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    EXPECT_EQ(LwSciError_NotImplemented,
    poolPtr->deletePacket_public(packetHandle));
}

/**
 * @testname{block_unit_test.pktCreate_StreamInternalError}
 * @testcase{21796789}
 * @verify{18796275}
 * @testpurpose{Test negative scenario of LwSciStream::Block::pktCreate(),
 * where pktCreate is called again.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Call pktCreate() with LwSciStreamPacket handle.
 *
 *   The call of LwSciStream::Block::pktCreate API with same LwSciStreamPacket
 * handle, should return LwSciError is of type LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktCreate_public()
 *     API to internally call LwSciStream::Block::pktCreate() API.}
 * @verifyFunction{LwSciStream::pktCreate}
 */
TEST_F (block_unit_test, pktCreate_StreamInternalError) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
        LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->pktCreate_public(packetHandle);
    EXPECT_EQ(LwSciError_StreamInternalError,
            poolPtr->pktCreate_public(packetHandle, LwSciStreamCookie_Ilwalid));
}

/**
 * @testname{block_unit_test.pktCreate_BadParameter}
 * @testcase{21796790}
 * @verify{18796275}
 * @testpurpose{Test negative scenario of LwSciStream::Block::pktCreate() when
 * cookie is already used for another packet. }
 * @testbehavior{
 * Setup:
 *   1. Create producer, pool, queue and consumer blocks.
 *   2. Create a packet with desired cookie.
 *
 *   The call to LwSciStream::Block::pktCreate API from pool object again
 * with same LwSciStreamCookie as in step-2, should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktCreate_public()
 *     API to internally call LwSciStream::Block::pktCreate() API.}
 * @verifyFunction{LwSciStream::pktCreate}
 */
TEST_F (block_unit_test, pktCreate_BadParameter) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
        LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->pktCreate_public(packetHandle, poolCookie);
    EXPECT_EQ(LwSciError_BadParameter,
            poolPtr->pktCreate_public(packetHandle, poolCookie));
}

/**
 * @testname{block_unit_test.eventWait_false}
 * @testcase{21796791}
 * @verify{18794724}
 * @testpurpose{Test negative scenario of LwSciStream::Block::eventWait() API
 * when there is no event available before timeout expires.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Declare and initialize ClockTime and Lock variables.
 *
 *   The call to LwSciStream::Block::eventWait() API from pool object
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::eventWait_public() API to internally
 *     call LwSciStream::Block::eventWait() API.}
 * @verifyFunction{LwSciStream::eventWait}
 */
TEST_F (block_unit_test, eventWait_false) {
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    using Lock = std::unique_lock<std::mutex>;
    using ClockTime = std::chrono::steady_clock::time_point;
    lwscistreamPanicMock npm;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    ClockTime  timeout  { std::chrono::steady_clock::now() };
    ClockTime* pTimeout { nullptr };
    timeout += std::chrono::microseconds(200U);
    pTimeout = &timeout;
    Lock blockLock { poolPtr->blkMutexLock_public(false) };

    /*Test case*/
    EXPECT_EQ(false, poolPtr->eventWait_public(blockLock, pTimeout));
}

/**
 * @testname{block_unit_test.connSrcInitiate_InsufficientResource}
 * @testcase{21796793}
 * @verify{18793548}
 * @testpurpose{Test negative scenario of LwSciStream::Block::connSrcInitiate(),
 * where connSrcInitiate() is called for blocks whose source connections are
 * already initialized.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Get producer BlockPtr using getRegisteredBlock().
 *
 *   The call to LwSciStream::Block::connSrcInitiate() API from pool
 * object should return IndexRet, containing error of
 * LwSciError_InsufficientResource.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::connSrcInitiate}
 */
TEST_F (block_unit_test, connSrcInitiate_InsufficientResource) {
    /*Initial setup*/
    LwSciStream::IndexRet ret;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    BlockPtr ProducerPtr { Block::getRegisteredBlock(producer) };

    /*Test code*/
    ret = poolPtr->connSrcInitiate(ProducerPtr);
    EXPECT_EQ(ret.error, LwSciError_InsufficientResource);
}

/**
 * @testname{block_unit_test.connSrcInitiate_NotSupported}
 * @testcase{21796795}
 * @verify{18793548}
 * @testpurpose{Test negative scenario of
 * LwSciStream::Block::connSrcInitiate() API, is called with
 * blocks that does not support source connection.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Get producer BlockPtr using getRegisteredBlock().
 *
 *   The call to LwSciStream::Block::connSrcInitiate() API from producer
 * BlockPtr should return IndexRet, containing error of
 * LwSciError_NotSupported.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::connSrcInitiate}
 */
TEST_F (block_unit_test, connSrcInitiate_NotSupported) {
    /*Initial setup*/
    LwSciStream::IndexRet ret;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    BlockPtr ProducerPtr { Block::getRegisteredBlock(producer) };

    /*Test code*/
    ret = ProducerPtr->connSrcInitiate(poolPtr);
    EXPECT_EQ(ret.error, LwSciError_NotSupported);
}

/**
 * @testname{block_unit_test.connDstInitiate_InsufficientResource}
 * @testcase{21796797}
 * @verify{18793761}
 * @testpurpose{Test negative scenario of
 * LwSciStream::Block::connDstInitiate(), where connDstInitiate() is called for
 * blocks whose source connections are already initialized.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Creates dummy producer object pointer of type LwSciStream::BlockPtr.
 *
 *   The call of LwSciStream::Block::connDstInitiate_false() API from pool
 * object should return IndexRet, containing error of LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::connDstInitiate}
 */
TEST_F (block_unit_test, connDstInitiate_InsufficientResource) {
    /*Initial setup*/
     LwSciStream::IndexRet ret;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    BlockPtr ProducerPtr {};

    /*Test code*/
    ret = poolPtr->connDstInitiate(ProducerPtr);
    EXPECT_EQ(ret.error, LwSciError_InsufficientResource);
}

/**
 * @testname{block_unit_test.connDstInitiate_NotSupported}
 * @testcase{21796799}
 * @verify{18793761}
 * @testpurpose{Test negative scenario of
 * LwSciStream::Block::connDstInitiate(), where connDstInitiate() API,
 * is called with blocks that does not support source connection.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Get consumer BlockPtr using getRegisteredBlock().
 *
 *   The call of LwSciStream::Block::connDstInitiate() API from consumer BlockPtr
 * return IndexRet, containing error of LwSciError_NotSupported.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::connDstInitiate}
 */
TEST_F (block_unit_test, connDstInitiate_NotSupported) {
    /*Initial setup*/
     LwSciStream::IndexRet ret;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    BlockPtr consumerPtr { Block::getRegisteredBlock(consumer[0]) };

    /*Test code*/
    ret = consumerPtr->connDstInitiate(poolPtr);
    EXPECT_EQ(ret.error, LwSciError_NotSupported);
}

/**
 * @testname{block_unit_test.dstAcquirePacket_default}
 * @testcase{21796800}
 * @verify{18794544}
 * @testpurpose{Test default scenario of LwSciStream::Block::dstAcquirePacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Connect the stream.
 *   3) Creates dummy variable of type Payload.
 *
 *   The call to LwSciStream::Block::dstAcquirePacket() with pool object
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::dstAcquirePacket}
 */
TEST_F (block_unit_test, dstAcquirePacket_false) {
    /*Initial setup*/

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    Payload acquiredPayload;

    /*Test code*/
    EXPECT_EQ(false, poolPtr->dstAcquirePacket(1U, acquiredPayload));
}

/**
 * @testname{block_unit_test.srcGetPacket_default}
 * @testcase{21796801}
 * @verify{18794625}
 * @testpurpose{Test default scenario of LwSciStream::Block::srcGetPacket().}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Connect the stream.
 *   3) Creates dummy variable of type Payload.
 *
 *   The call to LwSciStream::Block::srcGetPacket() through pool BlockPtr
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::srcGetPacket}
 */
TEST_F (block_unit_test, srcGetPacket_false) {
    /*Initial setup*/
    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    Payload acquiredPayload;

    /*Test code*/
    EXPECT_EQ(false, poolPtr->srcGetPacket(0U, acquiredPayload));
}

/**
 * @testname{block_unit_test.getEvent_Timeout}
 * @testcase{21796802}
 * @verify{18794355}
 * @testpurpose{Test failed scenario of LwSciStream::Block::getEvent(),
 * where getEvent is called with timeout and no event is available.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Initialize event of type LwSciStreamEvent.
 *   3) Create pool BlockPtr type with number of packets of 1.
 *
 *   The call of LwSciStream::Block::getEvent() API from pool object, with timeout
 * should return LwSciError_Timeout.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getEvent}
 */
TEST_F (block_unit_test, getEvent_Timeout) {
    /*Initial setup*/

    uint32_t numPackets {1};
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamEvent event;
    BlockPtr poolObj {std::make_shared<Pool>(numPackets)};

    /*Test code*/
    EXPECT_EQ(LwSciError_Timeout, poolPtr->getEvent(200U, event));
}

/**
 * @testname{block_unit_test.pendingEvent_false}
 * @testcase{21796803}
 * @verify{18794727}
 * @testpurpose{Test negative scenario of LwSciStream::Block::pendingEvent(),
 * where pendingEvent() is called when no events are available.}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::pendingEvent() API from pool object
 * with should return false value.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pendingEvent_public() API to internally
 *     call LwSciStream::Block::pendingEvent() API.}
 * @verifyFunction{LwSciStream::pendingEvent}
 */
TEST_F (block_unit_test, pendingEvent_false) {
    /*Initial setup*/
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamEvent msgEvent;

    /*Test code*/
    EXPECT_EQ(false, poolPtr->pendingEvent_public(msgEvent));
}

/**
 * @testname{block_unit_test.pktFindByCookie_failure}
 * @testcase{21796804}
 * @verify{18796284}
 * @testpurpose{Test negative scenario of LwSciStream::Block::pktFindByCookie()
 * when packet instance with matching cookie not found.}
 * @testbehavior{
 * Setup:
 *   1. Create producer, pool, queue and consumer blocks.
 *   2. Create a packet with desired cookie.
 *
 *   The call of LwSciStream::Block::pktFindByCookie() API with cookie value
 * different from cookie used in step-2, should return NULL.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktFindByCookie_public() wrapper
 *     call LwSciStream::Block::pktFindByCookie() API.}
 * @verifyFunction{LwSciStream::pktFindByCookie}
 */
TEST_F (block_unit_test, pktFindByCookie_false) {
    /*Initial setup*/
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->pktCreate_public(packetHandle);

    /*Test code*/
    EXPECT_EQ(nullptr,
            (poolPtr->pktFindByCookie_public(poolCookie+1U)));
}

/**
 * @testname{block_unit_test.pktPendingEvent_false2}
 * @testcase{21796806}
 * @verify{18796287}
 * @testpurpose{Test negative scenario of LwSciStream::Block::pktPendingEvent(),
 * where pktPendingEvent() is called when no events are pending.}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::pktPendingEvent() API from pool object
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktPendingEvent_public() API to internally
 *     call LwSciStream::Block::pktPendingEvent() API.}
 * @verifyFunction{LwSciStream::pktPendingEvent}
 */
TEST_F (block_unit_test, pktPendingEvent_false2) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciStreamEvent msgEvent;

    /*Test code*/
    EXPECT_EQ(false, poolPtr->pktPendingEvent_public(msgEvent, 1U));
}

/**
 * @testname{block_unit_test.validateWithError_StreamBadSrcIndex}
 * @testcase{21796807}
 * @verify{18794850}
 * @testpurpose{Test negative scenario of
 * LwSciStream::Block::validateWithError(), when invalid connection index to
 * a source block was passed to this block instance.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *      and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream and downstream connections by calling connDstComplete()
 *     connSrcComplete() respectively.
 *
 * Call of LwSciStream::Block::validateWithError() API, through pool object with
 * invalid SrcIndex and with ValidateBits set to ValidateCtrl::SrcIndex, should
 * return LwSciError_StreamBadSrcIndex.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 *  directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::validateWithError_StreamBadSrcIndex()
 *     API to internally call LwSciStream::Block::validateWithError() API.}
 * @verifyFunction{LwSciStream::Block::validateWithError}
 */
TEST_F (block_unit_test, validateWithError_StreamBadSrcIndex) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);

    /*Test code*/
    EXPECT_EQ(LwSciError_StreamBadSrcIndex,
            poolPtr->validateWithError_StreamBadSrcIndex(1U));
}

/**
 * @testname{block_unit_test.validateWithError_StreamBadDstIndex}
 * @testcase{21796808}
 * @verify{18794850}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::validateWithError() API, when invalid connection index
 * to a destination block was passed to this block instance.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create upstream and downstream variables of type LwSciStream::BlockPtr
 *      with producer and consumer as arguments, using getRegisteredBlock().
 *   3) Get output and input connection points with upstream and downstream
 *      BlockPtr's as arguments, using getOutputConnectPoint()
 *     and getInputConnectPoint().
 *   4) Initiate respective destination and source with upstream and
 *      downstream BlockPtr's as arguments and get destination and source index
 *      respectively,using connDstInitiate() and connSrcInitiate() respectively.
 *   5) Complete the Upstream connection by calling connDstComplete().
 *
 * The call of LwSciStream::Block::connSrcComplete() followed by
 * LwSciStream::Block::validateWithError() API should return
 * LwSciError of type validateWithError_StreamBadDstIndex.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 *  directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::validateWithError_StreamBadDstIndex()
 *     API to internally call LwSciStream::Block::validateWithError() API.}
 * @verifyFunction{LwSciStream::Block::validateWithError}
 */
TEST_F (block_unit_test, validateWithError_StreamBadDstIndex) {
    /*Initial setup*/

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    BlockPtr upstreamPtr { Block::getRegisteredBlock(producer) };
    BlockPtr downstreamPtr { Block::getRegisteredBlock(consumer[0]) };

    upstreamPtr->getOutputConnectPoint(upstreamPtr);

    downstreamPtr->getInputConnectPoint(downstreamPtr);

    IndexRet const
        dstReserved { upstreamPtr->connDstInitiate(downstreamPtr) };

    IndexRet const
        srcReserved { downstreamPtr->connSrcInitiate(upstreamPtr) };

    upstreamPtr->connDstComplete(dstReserved.index, srcReserved.index);

    /*Test code*/
    downstreamPtr->connSrcComplete(srcReserved.index, dstReserved.index);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, poolPtr->validateWithError_StreamBadDstIndex(1U));
}

/**
 * @testname{block_unit_test.validateWithEvent_false}
 * @testcase{21796809}
 * @verify{18794928}
 * @testpurpose{Test negative scenario of
 * LwSciStream::Block::validateWithEvent() API.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) ValidateBits is configured with ValidateCtrl::SetupPhase and
 *      ValidateCtrl::CompleteQueried.
 *
 *   The call of LwSciStream::Block::validateWithEvent() API with pool object
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::validateWithEvent_public() API to internally
 *     call LwSciStream::Block::validateWithEvent() API.}
 * @verifyFunction{LwSciStream::validateWithEvent}
 */
TEST_F (block_unit_test, validateWithEvent_false) {
    /*Initial setup*/

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    EXPECT_EQ(false, poolPtr->validateWithEvent_public(0U));
}

/**
 * @testname{block_unit_test.srcGetPacket_default}
 * @testcase{21796810}
 * @verify{18794625}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::srcGetPacket() API.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Declare dummy variable of Payload.
 *
 * The call of LwSciStream::Block::srcGetPacket() API should return
 * false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 *  directly.}
 * @testinput{No input external to the test application is required.
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::srcGetPacket_default() API to internally
 *     call LwSciStream::Block::srcGetPacket() API.}
 * @verifyFunction{LwSciStream::Block::srcGetPacket}
 */
TEST_F (block_unit_test, srcGetPacket_default) {
    /*Initial setup*/
    createBlocks(QueueType::Mailbox);
    Payload acquiredPayload { };

    /*Test code*/
    EXPECT_EQ(false, poolPtr->srcGetPacket_default(0U, acquiredPayload));
}

/**
 * @testname{block_unit_test.srcDisconnect_default}
 * @testcase{21796811}
 * @verify{18794637}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcDisconnect().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcDisconnect() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::srcDisconnect_default() API to internally
 *     call LwSciStream::Block::srcDisconnect() API.
 *   Following LwSciStream API calls are replaced with mocks:
 *      - lwscistreamPanic()}
 * @verifyFunction{LwSciStream::srcDisconnect}
 */
TEST_F (block_unit_test, srcDisconnect_default) {
    /*Initial setup*/

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamEvent event;

    /*Test code*/
    poolPtr->srcDisconnect_default(0U);
    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);

    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);

}

/**
 * @testname{block_unit_test.srcSendPacket_default}
 * @testcase{21796812}
 * @verify{18794619}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcSendPacket().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendPacket() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcSendPacket_default() API to internally
 *        call LwSciStream::Block::srcSendPacket() API.
 * @verifyFunction{LwSciStream::srcSendPacket}
 */
TEST_F (block_unit_test, srcSendPacket_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    FenceArray wrapFences { };

    /*Test code*/
    poolPtr->srcSendPacket_default(0U, packetHandle, wrapFences);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcDeletePacket_default}
 * @testcase{21796813}
 * @verify{18794616}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcDeletePacket().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcDeletePacket() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcDeletePacket_default() API to internally
 *        call LwSciStream::Block::srcDeletePacket() API.
 * @verifyFunction{LwSciStream::srcDeletePacket}
 */
TEST_F (block_unit_test, srcDeletePacket_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->srcDeletePacket_default(0U, packetHandle);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcInsertBuffer_default}
 * @testcase{21796814}
 * @verify{18794610}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcInsertBuffer().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcInsertBuffer() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcInsertBuffer_default() API to internally
 *        call LwSciStream::Block::srcInsertBuffer() API.
 * @verifyFunction{LwSciStream::srcInsertBuffer}
 */
TEST_F (block_unit_test, srcInsertBuffer_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufObj wrapBufObj{};

    /*Test code*/
    poolPtr->srcInsertBuffer_default(0U, packetHandle, 0U, wrapBufObj);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcCreatePacket_default}
 * @testcase{21796815}
 * @verify{18794607}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcCreatePacket().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcCreatePacket() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcCreatePacket_default() API to internally
 *        call LwSciStream::Block::srcCreatePacket() API.
 * @verifyFunction{LwSciStream::srcCreatePacket}
 */
TEST_F (block_unit_test, srcCreatePacket_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->srcCreatePacket_default(0U, packetHandle);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcSendElementStatus_default}
 * @testcase{21796816}
 * @verify{18794598}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::srcSendElementStatus().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendElementStatus() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcSendElementStatus_default() API to internally
 *        call LwSciStream::Block::srcSendElementStatus() API.
 * @verifyFunction{LwSciStream::srcSendElementStatus}
 */
TEST_F (block_unit_test, srcSendElementStatus_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->srcSendElementStatus_default(0U, packetHandle, 0U,
            LwSciError_NotImplemented);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcSendPacketStatus_default}
 * @testcase{21796817}
 * @verify{18794589}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcSendPacketStatus().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendPacketStatus() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcSendPacketStatus_default() API to internally
 *        call LwSciStream::Block::srcSendPacketStatus() API.
 * @verifyFunction{LwSciStream::srcSendPacketStatus}
 */
TEST_F (block_unit_test, srcSendPacketStatus_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    poolPtr->srcSendPacketStatus_default(0U, packetHandle,
            LwSciError_NotImplemented);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcSendPacketAttr_default}
 * @testcase{21796818}
 * @verify{18794586}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcSendPacketAttr().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendPacketAttr() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *       - Configure LwSciStream::Pool::srcSendPacketAttr_default() API to internally
 *         call LwSciStream::Block::srcSendPacketAttr() API.
 * @verifyFunction{LwSciStream::srcSendPacketAttr}
 */
TEST_F (block_unit_test, srcSendPacketAttr_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufAttr wrapBufAttr{};

    /*Test code*/
    poolPtr->srcSendPacketAttr_default(0U, 0U, 0U, packetHandle,
            wrapBufAttr);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcSendPacketElementCount_default}
 * @testcase{21796819}
 * @verify{18794583}
 * @testpurpose{Test success scenario of
 * LwSciStream::Block::srcSendPacketElementCount().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendPacketElementCount() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcSendPacketElementCount_default() API to internally
 *        call LwSciStream::Block::srcSendPacketElementCount() API.
 * @verifyFunction{LwSciStream::srcSendPacketElementCount}
 */
TEST_F (block_unit_test, srcSendPacketElementCount_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    /*Test code*/
   poolPtr->srcSendPacketElementCount_default(0U, 0U);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcSendSyncDesc_default}
 * @testcase{21796820}
 * @verify{18794577}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcSendSyncDesc().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendSyncDesc() API from pool
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcSendSyncDesc_default() API to internally
 *        call LwSciStream::Block::srcSendSyncDesc() API.
 * @verifyFunction{LwSciStream::srcSendSyncDesc}
 */
TEST_F (block_unit_test, srcSendSyncDesc_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncObj SyncObj { };

    /*Test code*/
    poolPtr->srcSendSyncDesc_default(0U, 0U, SyncObj);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcSendSyncCount_default}
 * @testcase{21796822}
 * @verify{18794568}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcSendSyncCount().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendSyncCount() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcSendSyncCount_default() API to internally
 *         call LwSciStream::Block::srcSendSyncCount() API.
 * @verifyFunction{LwSciStream::srcSendSyncCount}
 */
TEST_F (block_unit_test, srcSendSyncCount_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncObj SyncObj { };

    /*Test code*/
    poolPtr->srcSendSyncCount_default(0U, 0U);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.srcSendSyncAttr_default}
 * @testcase{21796823}
 * @verify{18794565}
 * @testpurpose{Test success scenario of LwSciStream::Block::srcSendSyncAttr().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::srcSendSyncAttr() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::srcSendSyncAttr_default() API to internally
 *        call LwSciStream::Block::srcSendSyncAttr() API.
 * @verifyFunction{LwSciStream::srcSendSyncAttr}
 */
TEST_F (block_unit_test, srcSendSyncAttr_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncAttr syncAttr{};

    /*Test code*/
    poolPtr->srcSendSyncAttr_default(0U, 0U, syncAttr);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstDisconnect_default}
 * @testcase{21796824}
 * @verify{18794559}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstDisconnect().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstDisconnect() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::dstDisconnect_default() API to internally
 *       call LwSciStream::Block::dstDisconnect() API.
 * @verifyFunction{LwSciStream::dstDisconnect}
 */
TEST_F (block_unit_test, dstDisconnect_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncAttr syncAttr{};

    /*Test code*/
    poolPtr->dstDisconnect_default(0U);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstReusePacket_default}
 * @testcase{21796825}
 * @verify{18794556}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstReusePacket().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstReusePacket() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *     - Configure LwSciStream::Pool::dstReusePacket_default() API to internally
 *       call LwSciStream::Block::dstReusePacket() API.
 * @verifyFunction{LwSciStream::dstReusePacket}
 */
TEST_F (block_unit_test, dstReusePacket_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    FenceArray wrapFences { };

    /*Test code*/
    poolPtr->dstReusePacket_default(0U, packetHandle, wrapFences);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

 /**
 * @testname{block_unit_test.dstInsertBuffer_default}
 * @testcase{21796826}
 * @verify{18794535}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstInsertBuffer().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstInsertBuffer() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::dstInsertBuffer_public() API to internally
 *     call LwSciStream::Block::dstInsertBuffer() API.
 * @verifyFunction{LwSciStream::dstInsertBuffer}
 */
TEST_F (block_unit_test, dstInsertBuffer_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufObj wrapBufObj{};

    /*Test code*/
    poolPtr->dstInsertBuffer_public(0U, packetHandle, 0U, wrapBufObj);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstDeletePacket_default}
 * @testcase{21796829}
 * @verify{18794538}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstDeletePacket().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstDeletePacket() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstDeletePacket_public() API to internally
 *        call LwSciStream::Block::dstDeletePacket() API.
 * @verifyFunction{LwSciStream::dstDeletePacket}
 */
TEST_F (block_unit_test, dstDeletePacket_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufObj wrapBufObj{};

    /*Test code*/
    poolPtr->dstDeletePacket_public(0U, packetHandle);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstCreatePacket_default}
 * @testcase{21796830}
 * @verify{18794484}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstCreatePacket().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstCreatePacket() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::dstCreatePacket}
 */
TEST_F (block_unit_test, dstCreatePacket_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufObj wrapBufObj{};

    /*Test code*/
    poolPtr->dstCreatePacket(0U, packetHandle);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstSendElementStatus_default}
 * @testcase{21796831}
 * @verify{18794448}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstSendElementStatus().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstSendElementStatus() API from pool
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstSendElementStatus_public() API to internally
 *        call LwSciStream::Block::dstSendElementStatus() API.
 * @verifyFunction{LwSciStream::dstSendElementStatus}
 */
TEST_F (block_unit_test, dstSendElementStatus_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufObj wrapBufObj{};

    /*Test code*/
    poolPtr->dstSendElementStatus_public(0U, packetHandle, 0U,
            LwSciError_NotImplemented);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstSendPacketStatus_default}
 * @testcase{21796832}
 * @verify{18794445}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstSendPacketStatus().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstSendPacketStatus() API from pool
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstSendPacketAttr_public() API to internally
 *        call LwSciStream::Block::dstSendPacketAttr() API.
 * @verifyFunction{LwSciStream::dstSendPacketStatus}
 */
TEST_F (block_unit_test, dstSendPacketStatus_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufObj wrapBufObj{};

    /*Test code*/
    poolPtr->dstSendPacketStatus_public(0U, packetHandle, LwSciError_NotImplemented);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstSendPacketAttr_default}
 * @testcase{21796847}
 * @verify{18794427}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstSendPacketAttr().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstSendPacketAttr() API from pool
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstSendPacketAttr_public() API to internally
 *        call LwSciStream::Block::dstSendPacketAttr() API.
 * @verifyFunction{LwSciStream::dstSendPacketAttr}
 */
TEST_F (block_unit_test, dstSendPacketAttr_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};
    LwSciWrap::BufAttr wrapBufObj{};

    /*Test code*/
    poolPtr->dstSendPacketAttr_public(0U, 0U, 0U, LwSciStreamElementMode_Asynchronous,
            wrapBufObj);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstSendPacketElementCount_default}
 * @testcase{21796848}
 * @verify{18794418}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstSendPacketElementCount().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstSendPacketElementCount() API from pool
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstSendPacketElementCount_public() API to internally
 *        call LwSciStream::Block::dstSendPacketElementCount() API.
 * @verifyFunction{LwSciStream::dstSendPacketElementCount}
 */
TEST_F (block_unit_test, dstSendPacketElementCount_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    /*Test code*/
    poolPtr->dstSendPacketElementCount_public(0U, 0U);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstSendSyncDesc_default}
 * @testcase{21796898}
 * @verify{18794415}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstSendSyncDesc(),
 * where dstSendSyncDesc() is called.}
 * @testbehavior{
 * Setup:
 *   1) Create producer, pool, queue and consumer blocks.
 *   2) Create dummy variable of LwSciWrap::SyncObj.
 *
 *   The call of LwSciStream::Block::dstSendSyncDesc() API from pool
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstSendSyncDesc_public() API to internally
 *        call LwSciStream::Block::dstSendSyncDesc_public() API.
 * @verifyFunction{LwSciStream::dstSendSyncDesc}
 */
TEST_F (block_unit_test, dstSendSyncDesc_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncObj SyncObj {  };

    /*Test code*/
    poolPtr->dstSendSyncDesc_public(0U, 0U, SyncObj);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstSendSyncCount_default}
 * @testcase{21796899}
 * @verify{18794409}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstSendSyncCount().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstSendSyncCount() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstSendSyncCount_public() API to internally
 *        call LwSciStream::Block::dstSendSyncCount() API.
 * @verifyFunction{LwSciStream::dstSendSyncCount}
 */
TEST_F (block_unit_test, dstSendSyncCount_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncObj SyncObj {  };

    /*Test code*/
    poolPtr->dstSendSyncCount_public(0U, 0U);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.dstSendSyncAttr_default}
 * @testcase{21796900}
 * @verify{18794397}
 * @testpurpose{Test success scenario of LwSciStream::Block::dstSendSyncAttr().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::dstSendSyncAttr() API from pool object
 * should result in LwSciError_NotImplemented event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::dstSendSyncAttr_public() API to internally
 *        call LwSciStream::Block::dstSendSyncAttr() API.
 * @verifyFunction{LwSciStream::dstSendSyncAttr}
 */
TEST_F (block_unit_test, dstSendSyncAttr_default) {
    /*Initial setup*/
    LwSciStreamEvent event;

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    LwSciWrap::SyncAttr SyncObj {  };

    /*Test code*/
    poolPtr->dstSendSyncAttr_public(0U, 0U, SyncObj);

    LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event);
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_NotImplemented, event.error);
}

/**
 * @testname{block_unit_test.pktDescSet_Success}
 * @testcase{21796904}
 * @verify{20698875}
 * @testpurpose{Test success scenario of LwSciStream::Block::pktDescSet().}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::pktDescSet() API from pool object
 * is of void return.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   Following LwSciStream API public wrappers are added:
 *      - Configure LwSciStream::Pool::pktDescSet_Success() API to internally
 *        call LwSciStream::Block::pktDescSet() API.}
 * @verifyFunction{LwSciStream::Block::pktDescSet}
 */
TEST_F (block_unit_test, pktDescSet_Success) {
    //Initial setup

    IndexRet ret;
    uint32_t numPackets {1};

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Set up packet description
    Packet::Desc desc { };

    //Test code
    poolPtr->pktDescSet_public(std::move(desc));
}

/**
 * @testname{block_unit_test.pktFindByHandle_Failure}
 * @testcase{21796905}
 * @verify{18796281}
 * @testpurpose{Test negative scenario of LwSciStream::Block::pktFindByHandle()
 * when packet instance not found.}
 * @testbehavior{
 * Setup:
 *   Create producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStream::Block::pktFindByHandle() API from pool BlockPtr
 * should return NULL.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::pktFindByHandle_public() API to internally
 *     call LwSciStream::Block::pktFindByHandle() API.}
 * @verifyFunction{LwSciStream::pktFindByHandle}
 */
TEST_F (block_unit_test, pktFindByHandle_Failure) {

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciStreamPacket packetHandle {~poolCookie};

    /*Test code*/
    EXPECT_EQ(nullptr, poolPtr->pktFindByHandle_public(packetHandle, false));
}

/**
 * @testname{block_unit_test.eventNotifierSetup_IlwalidState1}
 * @testcase{21796906}
 * @verify{21698764}
 * @testpurpose{Test negative scenario of LwSciStream::Block::eventNotifierSetup()
 * when default event signaling mode already configured on the block.}
 * @testbehavior{
 * Setup:
 *   1) Create producer and pool block.
 *   2) Create event service loop with count 1.
 *
 *   The call of LwSciStream::Block::eventNotifierSetup() API from pool BlockPtr
 * should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::eventNotifierSetup_public() API to internally
 *     call LwSciStream::Block::eventNotifierSetup() API.}
 * @verifyFunction{LwSciStream::eventNotifierSetup}
 */
TEST_F (block_unit_test, eventNotifierSetup_IlwalidState1) {

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    EventSetupRet ret;
    LwSciEventLoopService *newEventLoopService;
    std::shared_ptr<LwSciStream::Pool> poolPtr1;
    std::shared_ptr<LwSciStream::Producer> producerPtr1;

    // Create Pool block
    poolPtr1 = std::make_shared<LwSciStream::Pool>(3U);
    ASSERT_EQ(true, LwSciStream::Block::registerBlock(poolPtr1));

    // Create Producer block
    producerPtr1 = std::make_shared<LwSciStream::Producer>();
    ASSERT_EQ(true, LwSciStream::Block::registerBlock(producerPtr1));
    ASSERT_EQ(LwSciError_Success, producerPtr1->BindPool(poolPtr1));
    poolPtr1->eventDefaultSetup();

    LwSciEventLoopServiceCreate(1U, &newEventLoopService);
    EXPECT_EQ(LwSciError_IlwalidState,
    (poolPtr1->eventNotifierSetup_public(newEventLoopService->EventService)).err);

    LwSciStreamBlockDelete(producerPtr1->getHandle());
    LwSciStreamBlockDelete(poolPtr1->getHandle());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr1));
}

/**
 * @testname{block_unit_test.eventNotifierSetup_IlwalidState2}
 * @testcase{22060060}
 * @verify{21698764}
 * @testpurpose{Test negative scenario of LwSciStream::Block::eventNotifierSetup()
 * when event notifer already configured on the block.}
 * @testbehavior{
 * Setup:
 *   1) Create pool block.
 *   2) Create event service loop with count 1.
 *   3) Call Block::eventNotifierSetup() to configure event notifier for pool block.
 *
 *   The call of LwSciStream::Block::eventNotifierSetup() API from pool BlockPtr
 * should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::eventNotifierSetup_public() API to internally
 *     call LwSciStream::Block::eventNotifierSetup() API.}
 * @verifyFunction{LwSciStream::eventNotifierSetup}
 */
TEST_F (block_unit_test, eventNotifierSetup_IlwalidState2) {

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    EventSetupRet ret;
    LwSciEventLoopService *newEventLoopService;
    std::shared_ptr<LwSciStream::Pool> poolPtr1;

    // Create Pool block
    poolPtr1 = std::make_shared<LwSciStream::Pool>(3U);
    ASSERT_EQ(true, LwSciStream::Block::registerBlock(poolPtr1));

    LwSciEventLoopServiceCreate(1U, &newEventLoopService);
    EXPECT_EQ(LwSciError_Success,
    (poolPtr1->eventNotifierSetup_public(newEventLoopService->EventService)).err);
    EXPECT_EQ(LwSciError_IlwalidState,
    (poolPtr1->eventNotifierSetup_public(newEventLoopService->EventService)).err);

    LwSciStreamBlockDelete(poolPtr1->getHandle());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr1));

}

/**
 * @testname{block_unit_test.eventNotifierSetup_Success}
 * @testcase{22060045}
 * @verify{21698764}
 * @testpurpose{Test positive scenario of LwSciStream::Block::eventNotifierSetup()
 * when setting up the event notifier is successful.}
 * @testbehavior{
 * Setup:
 *   1) Create pool block.
 *   2) Create event service loop with count 1.
 *
 *   The call of LwSciStream::Block::eventNotifierSetup() API from pool BlockPtr
 * should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::eventNotifierSetup_public() API to internally
 *     call LwSciStream::Block::eventNotifierSetup() API.}
 * @verifyFunction{LwSciStream::eventNotifierSetup}
 */
TEST_F (block_unit_test, eventNotifierSetup_Success) {

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;
    EventSetupRet ret;
    LwSciEventLoopService *newEventLoopService;

    std::shared_ptr<LwSciStream::Pool> poolPtr1;

    // Create Pool block
    poolPtr1 = std::make_shared<LwSciStream::Pool>(3U);
    ASSERT_EQ(true, LwSciStream::Block::registerBlock(poolPtr1));

    LwSciEventLoopServiceCreate(1U, &newEventLoopService);

    ret = poolPtr1->eventNotifierSetup_public(newEventLoopService->EventService);
    EXPECT_EQ(LwSciError_Success, ret.err);

    LwSciStreamBlockDelete(poolPtr1->getHandle());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr1));

}

/**
 * @testname{block_unit_test.eventNotifierSetup_BadParameter}
 * @testcase{22060048}
 * @verify{21698764}
 * @testpurpose{Test negative scenario of LwSciStream::Block::eventNotifierSetup()
 * when LwSciEventService::CreateLocalEvent() failed.}
 * @testbehavior{
 * Setup:
 *   1) Create pool block.
 *   2) Create event service loop with count 1.
 *   3) Inject fault in LwSciEventService::CreateLocalEvent() to return LwSciError_BadParameter.
 *
 *   The call of LwSciStream::Block::eventNotifierSetup() API from pool BlockPtr
 * should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 * Following LwSciStream API public wrappers are added:
 *   - Configure LwSciStream::Pool::eventNotifierSetup_public() API to internally
 *     call LwSciStream::Block::eventNotifierSetup() API.}
 * @verifyFunction{LwSciStream::eventNotifierSetup}
 */
TEST_F (block_unit_test, eventNotifierSetup_BadParameter) {

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    EventSetupRet ret;
    LwSciEventLoopService *newEventLoopService;
    std::shared_ptr<LwSciStream::Pool> poolPtr1;

    // Create Pool block
    poolPtr1 = std::make_shared<LwSciStream::Pool>(3U);
    ASSERT_EQ(true, LwSciStream::Block::registerBlock(poolPtr1));

    test_block.LwSciEventService_CreateLocalEvent_fail = true;
    LwSciEventLoopServiceCreate(1U, &newEventLoopService);
    EXPECT_EQ(LwSciError_BadParameter,
    (poolPtr1->eventNotifierSetup_public(newEventLoopService->EventService)).err);

    LwSciStreamBlockDelete(poolPtr1->getHandle());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr1));
}

/**
 * @testname{block_unit_test.registerBlock_true}
 * @testcase{22060051}
 * @verify{18794649}
 * @testpurpose{Test positive scenario of LwSciStream::Block::registerBlock()
 * when block registration is successful.}
 * @testbehavior{
 * Setup:
 *   Create producer block.
 *
 *   The call of LwSciStream::Block::registerBlock() API with producer BlockPtr
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::registerBlock}
 */
TEST_F (block_unit_test, registerBlock_true) {

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create Producer block
    producerPtr = std::make_shared<LwSciStream::Producer>();
    EXPECT_EQ(true, LwSciStream::Block::registerBlock(producerPtr));

    LwSciStreamBlockDelete(producerPtr->getHandle());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));
}

/**
 * @testname{block_unit_test.getRegisteredBlock_Success}
 * @testcase{22060054}
 * @verify{18794658}
 * @testpurpose{Test positive scenario of LwSciStream::Block::getRegisteredBlock()
 * when retrieving a block is successful.}
 * @testbehavior{
 * Setup:
 *   1. Create producer block.
 *   2. Registers the producer block using Block::registerBlock() and get
 *      the block handle by calling Block::getHandle().
 *
 *   The call of Block::getRegisteredBlock() API with producer block handle
 * should return producer BlockPtr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getRegisteredBlock}
 */
TEST_F (block_unit_test, getRegisteredBlock_Success) {

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create Producer block
    producerPtr = std::make_shared<LwSciStream::Producer>();
    EXPECT_EQ(true, LwSciStream::Block::registerBlock(producerPtr));

    EXPECT_EQ(producerPtr, LwSciStream::Block::getRegisteredBlock(producerPtr->getHandle()));

    LwSciStreamBlockDelete(producerPtr->getHandle());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));

}

/**
 * @testname{block_unit_test.getRegisteredBlock_Failure}
 * @testcase{22060057}
 * @verify{18794658}
 * @testpurpose{Test negative scenario of LwSciStream::Block::getRegisteredBlock()
 * when retrieving a block failed due to invalid block handle.}
 * @testbehavior{
 * Setup:
 *   1. Create producer block.
 *   2. Registers the producer block using Block::registerBlock().
 *
 *   The call of Block::getRegisteredBlock() API with invalid block handle
 * should return NULL.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable
 * directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::Block::getRegisteredBlock}
 */
TEST_F (block_unit_test, getRegisteredBlock_Failure) {

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Create Producer block
    producerPtr = std::make_shared<LwSciStream::Producer>();
    EXPECT_EQ(true, LwSciStream::Block::registerBlock(producerPtr));

    EXPECT_EQ(nullptr, LwSciStream::Block::getRegisteredBlock(2U));

    LwSciStreamBlockDelete(producerPtr->getHandle());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));

}


}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
