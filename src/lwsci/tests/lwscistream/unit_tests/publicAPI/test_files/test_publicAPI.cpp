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

#include <limits>
#include <functional>
#include "lwscistream.h"
#include "lwscistream_common.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include "lwscistream_panic_mock.h"

class publicAPI_unit_test : public LwSciStreamTest
{
public:
    publicAPI_unit_test()
    {
        // initialization code here
    }

    void SetUp()
    {
        // Producer sync obj count
        prodSyncCount = NUM_SYNCOBJS;
        // Consumer sync obj count
        consSyncCount[0] = NUM_SYNCOBJS;
        // Producer synchronousonly flag
        prodSynchronousOnly = false;
        // Consumer synchronousonly flag
        consSynchronousOnly = false;
    }

    void TearDown()
    {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~publicAPI_unit_test()
    {
        // cleanup any pending stuff, but no exceptions allowed
    }
    // put in any custom data members that you need
};

namespace LwSciStream {
/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockConnect_Success}
 * @testcase{22058950}
 * @verify{19789602}
 * @testpurpose{Test positive scenario of LwSciStreamBlockConnect(), when
 * connection between two blocks was made successfully.}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockConnect() API , should return LwSciError_Success,
 * when called with parameters upstream set to producer and downstream
 * set to consumer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockConnect()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockConnect_Success)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Test case
    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockConnect(producer, consumer[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockConnect_InsufficientResource1}
 * @testcase{22058953}
 * @verify{19789602}
 * @testpurpose{Test negative scenario of LwSciStreamBlockConnect(), when
 * upstream has no available outputs.}
 * @testbehavior{
 * Setup:
 *   1.Create the producer, pool, queue and 2 consumer blocks.
 *   2.Connects producer with consumer1 by calling LwSciStreamBlockConnect() API
 * with producer and consumer1.
 *
 *   The call of LwSciStreamBlockConnect() API , should return LwSciError_InsufficientResource,
 * when called with parameters upstream set to producer and downstream
 * set to consumer2.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockConnect()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockConnect_InsufficientResource1)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox, 2);

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockConnect(producer, consumer[0]));

    // Test case
    ASSERT_EQ(LwSciError_InsufficientResource,LwSciStreamBlockConnect(producer, consumer[1]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockConnect_InsufficientResource2}
 * @testcase{22058956}
 * @verify{19789602}
 * @testpurpose{Test negative scenario of LwSciStreamBlockConnect(), when
 * downstream has no available inputs.}
 * @testbehavior{
 * Setup:
 *   1.Create the producer, pool, queue and consumer blocks.
 *   2.Connects producer1 with consumer by calling LwSciStreamBlockConnect() API
 * with producer1 and consumer.
 *
 *   The call of LwSciStreamBlockConnect() API , should return LwSciError_BadParameter,
 * when called with parameters upstream set to producer2 and downstream
 * set to consumer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockConnect()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockConnect_InsufficientResource2)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // initial set up
    uint32_t numPackets = NUM_PACKETS;
    LwSciStreamBlock producer1 = 0U;
    LwSciStreamBlock pool1 = 0U;

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool1));

    ASSERT_EQ(LwSciError_Success, LwSciStreamProducerCreate(pool1, &producer1));

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockConnect(producer, consumer[0]));

    // Test case
    ASSERT_EQ(LwSciError_InsufficientResource,LwSciStreamBlockConnect(producer1, consumer[0]));

    LwSciStreamBlockDelete(producer1);
    LwSciStreamBlockDelete(pool1);
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockConnect_AccessDenied1}
 * @testcase{22058958}
 * @verify{19789602}
 * @testpurpose{Test negative scenario of LwSciStreamBlockConnect(), where
 * upstream does not allow explicit connection.}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockConnect() API , should return LwSciError_AccessDenied,
 * when called with parameters upstream set to queue and downstream
 * set to consumer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockConnect()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockConnect_AccessDenied1)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Test case
    ASSERT_EQ(LwSciError_AccessDenied,LwSciStreamBlockConnect(queue[0], consumer[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockConnect_AccessDenied2}
 * @testcase{22058960}
 * @verify{19789602}
 * @testpurpose{Test negative scenario of LwSciStreamBlockConnect(), where
 * downstream does not allow explicit connection.}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockConnect() API , should return LwSciError_AccessDenied,
 * when called with parameters upstream set to producer and downstream
 * set to pool.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockConnect()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockConnect_AccessDenied2)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Test case
    ASSERT_EQ(LwSciError_AccessDenied,LwSciStreamBlockConnect(producer, pool));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_Success}
 * @testcase{22058965}
 * @verify{19789605}
 * @testpurpose{Test positive scenario of LwSciStreamProducerCreate(), when
 * a new producer block was set up successfully.}
 * @testbehavior{
 * Setup:
 *   Create static pool block through LwSciStreamStaticPoolCreate() API.
 *
 *   The call of LwSciStreamProducerCreate() API , should return LwSciError_Success,
 * when called with parameters pool set to pool block and reference of producer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_Success)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool));

    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamProducerCreate(pool, &producer));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_BadParameter1}
 * @testcase{22058968}
 * @verify{19789605}
 * @testpurpose{Test negative scenario of LwSciStreamProducerCreate(), when
 * pool parameter does not reference a pool block.}
 * @testbehavior{
 * Setup:
 *   Create queue block through LwSciStreamMailboxQueueCreate() API.
 *
 *   The call of LwSciStreamProducerCreate() API , should return LwSciError_BadParameter,
 * when called with parameters pool set to queue block and reference of producer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_BadParameter1)
{
    ASSERT_EQ(LwSciError_Success, LwSciStreamMailboxQueueCreate(&queue[0]));

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamProducerCreate(queue[0], &producer));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_BadParameter2}
 * @testcase{22058971}
 * @verify{19789605}
 * @testpurpose{Test negative scenario of LwSciStreamProducerCreate(), when
 * 'producer' is a null pointer.}
 * @testbehavior{
 * Setup:
 *   Create static pool block through LwSciStreamStaticPoolCreate() API.
 *
 *   The call of LwSciStreamProducerCreate() API , should return LwSciError_BadParameter,
 * when called with parameters pool set to pool block and output parameter producer set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_BadParameter2)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool));

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamProducerCreate(pool, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_InsufficientResource}
 * @testcase{22058974}
 * @verify{19789605}
 * @testpurpose{Test negative scenario of LwSciStreamProducerCreate(), when
 * pool block is already associated with another producer.}
 * @testbehavior{
 * Setup:
 *   1.Create static pool block through LwSciStreamStaticPoolCreate() API.
 *   2.Create producer block by calling LwSciStreamProducerCreate() with pool and reference of producer block
 *
 *   The call of LwSciStreamProducerCreate() API , should return LwSciError_InsufficientResource,
 * when called again with parameters pool set to pool block and reference of producer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_InsufficientResource)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool));

    ASSERT_EQ(LwSciError_Success, LwSciStreamProducerCreate(pool, &producer));

    // Test case
    ASSERT_EQ(LwSciError_InsufficientResource, LwSciStreamProducerCreate(pool, &producer));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_StreamInternalError}
 * @testcase{22058977}
 * @verify{19789605}
 * @testpurpose{Test negative scenario of LwSciStreamProducerCreate(), when
 * producer block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *   Create static pool block through LwSciStreamStaticPoolCreate() API.
 *
 *   The call of LwSciStreamProducerCreate() API , should return LwSciError_StreamInternalError,
 * when called with parameters pool set to pool block and reference of producer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_StreamInternalError)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool));

    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamProducerCreate(pool, &producer));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerCreate_Success}
 * @testcase{22059325}
 * @verify{19789608}
 * @testpurpose{Test positive scenario of LwSciStreamConsumerCreate(), when
 * a new consumer block was set up successfully.}
 * @testbehavior{
 * Setup:
 *   Create maibox queue block through LwSciStreamMailboxQueueCreate() API.
 *
 *   The call of LwSciStreamConsumerCreate() API , should return LwSciError_Success,
 * when called with parameters queue set to queue block and reference of consumer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerCreate_Success)
{
    ASSERT_EQ(LwSciError_Success, LwSciStreamMailboxQueueCreate(&queue[0]));

    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamConsumerCreate(queue[0], &consumer[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerCreate_BadParameter1}
 * @testcase{22058980}
 * @verify{19789608}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerCreate(), when
 * queue parameter does not reference a queue block.}
 * @testbehavior{
 * Setup:
 *   Create static pool block through LwSciStreamStaticPoolCreate() API.
 *
 *   The call of LwSciStreamConsumerCreate() API , should return LwSciError_BadParameter,
 * when called with parameters queue set to pool block and reference of consumer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerCreate_BadParameter1)
{
    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool));

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamConsumerCreate(pool, &consumer[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerCreate_BadParameter2}
 * @testcase{22058984}
 * @verify{19789608}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerCreate(), when
 * 'consumer' is a null pointer.}
 * @testbehavior{
 * Setup:
 *   Create maibox queue block through LwSciStreamMailboxQueueCreate() API.
 *
 *   The call of LwSciStreamConsumerCreate() API , should return LwSciError_BadParameter,
 * when called with parameters queue set to queue block and output parameter consumer set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerCreate_BadParameter2)
{
    ASSERT_EQ(LwSciError_Success, LwSciStreamMailboxQueueCreate(&queue[0]));

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamConsumerCreate(queue[0], nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerCreate_InsufficientResource}
 * @testcase{22058986}
 * @verify{19789608}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerCreate(), when
 * queue block is already bound to another consumer.}
 * @testbehavior{
 * Setup:
 *   1.Create maibox queue block through LwSciStreamMailboxQueueCreate() API.
 *   2.Create consumer1 block by calling LwSciStreamConsumerCreate() with queue
 * and reference of consumer1 block
 *
 *   The call of LwSciStreamConsumerCreate() API , should return LwSciError_InsufficientResource,
 * when called with parameters queue set to queue block and reference of consumer2 block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerCreate_InsufficientResource)
{
    ASSERT_EQ(LwSciError_Success, LwSciStreamMailboxQueueCreate(&queue[0]));

    ASSERT_EQ(LwSciError_Success, LwSciStreamConsumerCreate(queue[0], &consumer[0]));

    // Test case
    ASSERT_EQ(LwSciError_InsufficientResource, LwSciStreamConsumerCreate(queue[0], &consumer[1]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerCreate_StreamInternalError}
 * @testcase{22058989}
 * @verify{19789608}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerCreate(), when
 * consumer block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *   Create maibox queue block through LwSciStreamMailboxQueueCreate() API.
 *
 *   The call of LwSciStreamConsumerCreate() API , should return LwSciError_StreamInternalError,
 * when called parameters queue set to queue block and reference of consumer block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamConsumerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerCreate_StreamInternalError)
{
    ASSERT_EQ(LwSciError_Success, LwSciStreamMailboxQueueCreate(&queue[0]));

    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamConsumerCreate(queue[0], &consumer[0]));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamStaticPoolCreate_Success}
 * @testcase{22058992}
 * @verify{19789611}
 * @testpurpose{Test positive scenario of LwSciStreamStaticPoolCreate(), when
 * a new pool block was set up successfully.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamStaticPoolCreate() API , should return LwSciError_Success,
 * when called with numPackets set to NUM_PACKETS and reference of new pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamStaticPoolCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamStaticPoolCreate_Success)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;

    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamStaticPoolCreate_BadParameter}
 * @testcase{22058995}
 * @verify{19789611}
 * @testpurpose{Test negative scenario of LwSciStreamStaticPoolCreate(), when
 * 'pool' is a null pointer.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamStaticPoolCreate() API , should return LwSciError_BadParameter,
 * when called with numPackets set to NUM_PACKETS and output parameter pool set to nullptr}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamStaticPoolCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamStaticPoolCreate_BadParameter)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamStaticPoolCreate(numPackets, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamStaticPoolCreate_StreamInternalError}
 * @testcase{22058998}
 * @verify{19789611}
 * @testpurpose{Test negative scenario of LwSciStreamStaticPoolCreate(), when
 * pool block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamStaticPoolCreate() API , should return LwSciError_StreamInternalError,
 * when called with numPackets set to NUM_PACKETS and reference of new pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamStaticPoolCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamStaticPoolCreate_StreamInternalError)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;
    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamStaticPoolCreate(numPackets, &pool));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMailboxQueueCreate_Success}
 * @testcase{22059002}
 * @verify{19789614}
 * @testpurpose{Test positive scenario of LwSciStreamMailboxQueueCreate(), when
 * a new mailbox queue block was set up successfully.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMailboxQueueCreate() API , should return
 * LwSciError_Success, when called with reference of new mailbox queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamMailboxQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMailboxQueueCreate_Success)
{
    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamMailboxQueueCreate(&queue[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMailboxQueueCreate_BadParameter}
 * @testcase{22059004}
 * @verify{19789614}
 * @testpurpose{Test negative scenario of LwSciStreamMailboxQueueCreate(), when
 * 'queue' is a null pointer.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMailboxQueueCreate() API , should return
 * LwSciError_BadParameter, when called with parameter queue set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamMailboxQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMailboxQueueCreate_BadParameter)
{
    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamMailboxQueueCreate(nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMailboxQueueCreate_StreamInternalError}
 * @testcase{22059007}
 * @verify{19789614}
 * @testpurpose{Test negative scenario of LwSciStreamMailboxQueueCreate(), when
 * mailbox queue block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMailboxQueueCreate() API , should return
 * LwSciError_StreamInternalError, when called with reference of new mailbox queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamMailboxQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMailboxQueueCreate_StreamInternalError)
{
    // initial set up
    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamMailboxQueueCreate(&queue[0]));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamFifoQueueCreate_Success}
 * @testcase{22059010}
 * @verify{19789617}
 * @testpurpose{Test positive scenario of LwSciStreamFifoQueueCreate(), when
 *  a new  FIFO queue block was set up successfully.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamFifoQueueCreate() API , should return
 * LwSciError_Success, when called with reference of new FIFO queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamFifoQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamFifoQueueCreate_Success)
{
    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamFifoQueueCreate(&queue[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamFifoQueueCreate_BadParameter}
 * @testcase{22059013}
 * @verify{19789617}
 * @testpurpose{Test negative scenario of LwSciStreamFifoQueueCreate(), when
 * 'queue' is a null pointer.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamFifoQueueCreate() API , should return
 * LwSciError_BadParameter, when called with 'queue' set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamFifoQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamFifoQueueCreate_BadParameter)
{
    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamFifoQueueCreate(nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamFifoQueueCreate_StreamInternalError}
 * @testcase{22059016}
 * @verify{19789617}
 * @testpurpose{Test negative scenario of LwSciStreamFifoQueueCreate(), when
 * FIFO queue block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamFifoQueueCreate() API , should return
 * LwSciError_StreamInternalError, when called with reference of new FIFO queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamFifoQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamFifoQueueCreate_StreamInternalError)
{
    // initial set up
    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamFifoQueueCreate(&queue[0]));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMulticastCreate_Success}
 * @testcase{22059019}
 * @verify{19789620}
 * @testpurpose{Test positive scenario of LwSciStreamMulticastCreate(), when
 * a new multicast block was set up successfully.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMulticastCreate() API , should return
 * LwSciError_Success, when called with outputCount (number of connected ouput blocks.)
 * equals to 2 and reference of new multicast block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamMulticastCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMulticastCreate_Success)
{
    // initial set up
    uint32_t numConsumers = 2U;

    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamMulticastCreate(numConsumers, &multicast));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMulticastCreate_BadParameter1}
 * @testcase{22059022}
 * @verify{19789620}
 * @testpurpose{Test negative scenario of LwSciStreamMulticastCreate(), when
 * 'multicast' is a null pointer.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMulticastCreate() API , should return
 * LwSciError_BadParameter, when called with outputCount (number of connected output blocks.)
 * equals to 2 and output parameter 'multicast' set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamMulticastCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMulticastCreate_BadParameter1)
{
    // initial set up
    uint32_t numConsumers = 2U;

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamMulticastCreate(numConsumers, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMulticastCreate_BadParameter2}
 * @testcase{22059025}
 * @verify{19789620}
 * @testpurpose{Test negative scenario of LwSciStreamMulticastCreate(), when
 * outputCount is larger than the number allowed.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMulticastCreate() API , should return
 * LwSciError_BadParameter, when called with outputCount (number of connected ouput blocks)
 * equals to a number which is larger than the allowed output count and reference of new multicast block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamMulticastCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMulticastCreate_BadParameter2)
{
    // initial set up
    int32_t maxDstConnections = 0U;
    uint32_t outputCount;

    LwSciStreamAttributeQuery(LwSciStreamQueryableAttrib_MaxMulticastOutputs,&maxDstConnections);

    outputCount = maxDstConnections+1;

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamMulticastCreate(outputCount, &multicast));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMulticastCreate_StreamInternalError}
 * @testcase{22059026}
 * @verify{19789620}
 * @testpurpose{Test negative scenario of LwSciStreamMulticastCreate(), when
 * multicast block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMulticastCreate() API , should return
 * LwSciError_StreamInternalError, when called with outputCount (number of connected output blocks)
 * equals to 2 and reference of new multicast block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamMulticastCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMulticastCreate_StreamInternalError)
{
    // initial set up
    uint32_t numConsumers = 2U;
    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamMulticastCreate(numConsumers, &multicast));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventQuery_Success}
 * @testcase{22059029}
 * @verify{19789629}
 * @testpurpose{Test positive scenario of LwSciStreamBlockEventQuery(), when
 * event is filled with the queried event data.}
 * @testbehavior{
 * Setup:
 *   1. Create the producer, pool, queue and consumer blocks.
 *   2. Connect the blocks (producer and consumer)through LwSciStreamBlockConnect()
 *
 *   The call of LwSciStreamBlockEventQuery() API , should return LwSciError_Success,
 * when called with parameters block set to producer, EVENT_QUERY_TIMEOUT
 * and reference of LwSciStreamEvent}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventQuery_Success)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockConnect(producer, consumer[0]));

    // Test case
    EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Connected, event.type);
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventQuery_BadParameter}
 * @testcase{22059032}
 * @verify{19789629}
 * @testpurpose{Test negative scenario of LwSciStreamBlockEventQuery(), when
 * event is nullptr.}
 * @testbehavior{
 * Setup:
 *   1. Create the producer, pool, queue and consumer blocks.
 *   2. Connect the blocks (producer and consumer)through LwSciStreamBlockConnect()
 *
 *   The call of LwSciStreamBlockEventQuery() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, EVENT_QUERY_TIMEOUT
 * and event set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventQuery_BadParameter)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockConnect(producer, consumer[0]));

    // Test case
    EXPECT_EQ(LwSciError_BadParameter,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventQuery_Timeout}
 * @testcase{22059035}
 * @verify{19789629}
 * @testpurpose{Test negative scenario of LwSciStreamBlockEventQuery(), when
 * timeoutUsec period was reached before an LwSciStreamEvent became available.}
 * @testbehavior{
 * Setup:
 *   1. Create the producer, pool, queue and consumer blocks.
 *   2. Connect the blocks (producer and consumer)through LwSciStreamBlockConnect()
 *   3. Query connect event on producer through LwSciStreamBlockEventQuery()
 *
 *   The call of LwSciStreamBlockEventQuery() API , should return LwSciError_Timeout,
 * when called with parameters block set to producer, timeoutUsec set to 10
 * and reference of LwSciStreamEvent.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventQuery_Timeout)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockConnect(producer, consumer[0]));

    EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    // Test case
    EXPECT_EQ(LwSciError_Timeout,
            LwSciStreamBlockEventQuery(producer, 10, &event));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncRequirements_Success}
 * @testcase{22059038}
 * @verify{19789632}
 * @testpurpose{Test positive scenario of LwSciStreamBlockSyncRequirements(), when
 * LwSciSyncObj waiter requirements are sent successfully.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockSyncRequirements() API , should return LwSciError_Success,
 * when called with parameters block set to producer, synchronousOnly set to true
 * and LwSciSyncAttrList set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncRequirements()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncRequirements_Success)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncRequirements_BadParameter}
 * @testcase{22059041}
 * @verify{19789632}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncRequirements(), when
 * 'waitSyncAttrList' is null pointer and 'synchronousOnly' is false.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockSyncRequirements() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, synchronousOnly set to false
 * and LwSciSyncAttrList set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncRequirements()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncRequirements_BadParameter)
{
    // initial set up
    bool prodSynchronousOnly = false;
    LwSciSyncAttrList prodSyncAttrList = nullptr;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    ASSERT_EQ(LwSciError_BadParameter,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncRequirements_NotImplemented}
 * @testcase{22059044}
 * @verify{19789632}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncRequirements(), when
 * 'block' is not producer/consumer.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockSyncRequirements() API , should return LwSciError_NotImplemented,
 * when called with parameters block set to pool, synchronousOnly set to true
 * and LwSciSyncAttrList set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncRequirements()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncRequirements_NotImplemented)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    ASSERT_EQ(LwSciError_NotImplemented,LwSciStreamBlockSyncRequirements(pool,
                                             prodSynchronousOnly,prodSyncAttrList));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncRequirements_IlwalidOperation}
 * @testcase{22059049}
 * @verify{19789632}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncRequirements(), when
 * synchronousOnly flag is true but waitSyncAttrList is not NULL.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Create sync attrs through cpuWaiterAttrList().
 *
 *   The call of LwSciStreamBlockSyncRequirements() API , should return LwSciError_IlwalidOperation,
 * when called with parameters block set to producer, synchronousOnly set to true
 * and LwSciSyncAttrList set to not nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncRequirements()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncRequirements_IlwalidOperation)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    cpuWaiterAttrList(syncModule, prodSyncAttrList);
    // Test case
    ASSERT_EQ(LwSciError_IlwalidOperation,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncRequirements_IlwalidState}
 * @testcase{22059050}
 * @verify{19789632}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncRequirements(), when
 * LwSciSyncObj waiter requirements are already sent to other endpoint.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Call LwSciStreamBlockSyncRequirements() with LwSciStreamBlock set to producer, synchronousOnly set to true
 *      and LwSciSyncAttrList set to nullptr.
 *
 *   The call of LwSciStreamBlockSyncRequirements() API , should return LwSciError_IlwalidState,
 * when called again with parameters block set to producer, synchronousOnly set to true
 * and LwSciSyncAttrList set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncRequirements()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncRequirements_IlwalidState)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    // Test case
    ASSERT_EQ(LwSciError_IlwalidState,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObjCount_Success}
 * @testcase{22059055}
 * @verify{19789635}
 * @testpurpose{Test positive scenario of LwSciStreamBlockSyncObjCount(), when
 * LwSciSyncObj count is sent successfully.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *   5. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by
 *      calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockSyncObjCount() API , should return LwSciError_Success,
 * when called with parameters block set to producer and number of syncobj count of producer set to 2.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObjCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObjCount_Success)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[n],
                                         consSynchronousOnly,consSyncAttrList));
    }

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
    LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
    LwSciSyncAttrListFree(event.syncAttrList);

    // Producer sends its sync count to the consumer
    ASSERT_TRUE(prodSyncCount <= maxNumSyncObjs);

    // Test case
    ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSyncObjCount(producer, prodSyncCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObjCount_IlwalidOperation1}
 * @testcase{22059058}
 * @verify{19789635}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObjCount(), when
 * LwSciStreamEventType_SyncAttr event from the block is not yet queried.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockSyncObjCount() API , should return LwSciError_IlwalidOperation,
 * when called with parameters block set to producer and number of syncobj count of producer set to 2.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObjCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObjCount_IlwalidOperation1)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    uint32_t prodSyncCount = NUM_SYNCOBJS;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[n],
                                         consSynchronousOnly,consSyncAttrList));
    }

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
    LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    // Test case
    ASSERT_EQ(LwSciError_IlwalidOperation,
                LwSciStreamBlockSyncObjCount(producer, prodSyncCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObjCount_IlwalidOperation2}
 * @testcase{22059061}
 * @verify{19789635}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObjCount(), when
 * stream is in synchronous mode and count is greater than 0.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement with synchronous mode set to true
 *      to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by
 *       calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockSyncObjCount() API , should return LwSciError_IlwalidOperation,
 * when called with parameters block set to producer and number of syncobj count of producer set to 1.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObjCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObjCount_IlwalidOperation2)
{
    // initial set up
    bool prodSynchronousOnly = true;
    bool consSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    LwSciSyncAttrList consSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[n],
                                         consSynchronousOnly,consSyncAttrList));
    }

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
    LwSciSyncAttrListFree(event.syncAttrList);

    // Test case
    ASSERT_EQ(LwSciError_IlwalidOperation,
                LwSciStreamBlockSyncObjCount(producer, 1U));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObjCount_BadParameter2}
 * @testcase{22059062}
 * @verify{19789635}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObjCount(), when
 * count exceeds the maximum allowed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *   5. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by calling
 *      LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockSyncObjCount() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer and number of syncobj count of producer larger than MaxSyncObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObjCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObjCount_BadParameter2)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[n],
                                         consSynchronousOnly,consSyncAttrList));
    }

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
    LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
    LwSciSyncAttrListFree(event.syncAttrList);

    prodSyncCount = maxNumSyncObjs+1;

    // Test case
    ASSERT_EQ(LwSciError_BadParameter,
                LwSciStreamBlockSyncObjCount(producer, prodSyncCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObjCount_NotImplemented}
 * @testcase{22059066}
 * @verify{19789635}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObjCount(), when
 * 'block' is not producer/consumer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *   5. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by calling
 *      LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockSyncObjCount() API , should return LwSciError_NotImplemented,
 * when called with parameters block set to pool and number of syncobj count of producer set to 2.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObjCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObjCount_NotImplemented)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[n],
                                         consSynchronousOnly,consSyncAttrList));
    }

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
    LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
    LwSciSyncAttrListFree(event.syncAttrList);

    // Producer sends its sync count to the consumer
    ASSERT_TRUE(prodSyncCount <= maxNumSyncObjs);

    // Test case
    ASSERT_EQ(LwSciError_NotImplemented,
                LwSciStreamBlockSyncObjCount(pool, prodSyncCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObjCount_IlwalidState}
 * @testcase{22059069}
 * @verify{19789635}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObjCount(), when
 * LwSciSyncObj count is already sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *   5. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by calling
 *      LwSciStreamBlockEventQuery().
 *   6. Call LwSciStreamBlockSyncObjCount() with LwSciStreamBlock set to producer and number of
 *      syncobj count of producer set to 2.
 *
 *   The call of LwSciStreamBlockSyncObjCount() API , should return LwSciError_IlwalidState,
 * when called again with parameters block set to producer and number of syncobj count of producer set to 2.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObjCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObjCount_IlwalidState)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[n],
                                         consSynchronousOnly,consSyncAttrList));
    }

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
    LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
    LwSciSyncAttrListFree(event.syncAttrList);

    // Producer sends its sync count to the consumer
    ASSERT_TRUE(prodSyncCount <= maxNumSyncObjs);

    ASSERT_EQ(LwSciError_Success,
                LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

    // Test case
    ASSERT_EQ(LwSciError_IlwalidState,
                LwSciStreamBlockSyncObjCount(producer, prodSyncCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObject_Success}
 * @testcase{22059072}
 * @verify{19789638}
 * @testpurpose{Test positive scenario of LwSciStreamBlockSyncObject(), when
 * LwSciSyncObj is sent to the other endpoint(s) successfully.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *   5. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by calling
 *      LwSciStreamBlockEventQuery().
 *   6. Producer sends its sync count to the consumer by calling LwSciStreamBlockSyncObjCount().
 *
 *   The call of LwSciStreamBlockSyncObject() API , should return LwSciError_Success,
 * when called with parameters block set to producer, index set to 0 and LwSciSyncObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObject()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObject_Success)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;
    LwSciSyncModule syncModule = nullptr;
    LwSciSyncObj prodSyncObjs[4];

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();


    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,consSyncAttrList));

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
        LwSciSyncAttrListFree(event.syncAttrList);

    // Producer sends its sync count to the consumer
    ASSERT_TRUE(prodSyncCount <= maxNumSyncObjs);
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

        getSyncObj(syncModule, prodSyncObjs[0]);

    // Test case
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockSyncObject(
                producer, 0, prodSyncObjs[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObject_BadParameter}
 * @testcase{22059075}
 * @verify{19789638}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObject(), when
 * index exceeds the LwSciSyncObj count.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *   5. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by calling
 *      LwSciStreamBlockEventQuery().
 *   6. Producer sends its sync count as 2 to the consumer by calling LwSciStreamBlockSyncObjCount().
 *
 *   The call of LwSciStreamBlockSyncObject() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, index set to 5 and LwSciSyncObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObject()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObject_BadParameter)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;
    LwSciSyncModule syncModule = nullptr;
    LwSciSyncObj prodSyncObjs[4];

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();


    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

    for (uint32_t n = 0U; n < numConsumers; n++) {
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[n],
                                         consSynchronousOnly,consSyncAttrList));
    }

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
        LwSciSyncAttrListFree(event.syncAttrList);

    // Producer sends its sync count to the consumer
    ASSERT_TRUE(prodSyncCount <= maxNumSyncObjs);
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

        getSyncObj(syncModule, prodSyncObjs[0]);

    // Test case
        ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockSyncObject(
                producer, 5, prodSyncObjs[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObject_IlwalidOperation1}
 * @testcase{22059078}
 * @verify{19789638}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObject(), when
 * LwSciStreamEventType_SyncAttr event from the block is not yet queried.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockSyncObject() API , should return LwSciError_IlwalidOperation,
 * when called with parameters block set to producer, index set to 0 and LwSciSyncObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObject()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObject_IlwalidOperation1)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    bool consSynchronousOnly = true;
    LwSciSyncAttrList consSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    uint32_t prodSyncCount = NUM_SYNCOBJS;
    LwSciSyncModule syncModule = nullptr;
    LwSciSyncObj prodSyncObjs[4];

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();


    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,consSyncAttrList));

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

        getSyncObj(syncModule, prodSyncObjs[0]);

    // Test case
        ASSERT_EQ(LwSciError_IlwalidOperation, LwSciStreamBlockSyncObject(
                producer,0, prodSyncObjs[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObject_IlwalidOperation2}
 * @testcase{22059081}
 * @verify{19789638}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObject(), when
 * stream is in synchronous mode.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement with synchronous mode set to true to the producer.
 *      through LwSciStreamBlockSyncRequirements().
 *   4. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by calling
 *      LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockSyncObject() API , should return LwSciError_IlwalidOperation,
 * when called with parameters block set to producer, index set to 0 and LwSciSyncObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObject()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObject_IlwalidOperation2)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    bool consSynchronousOnly = true;
    LwSciSyncAttrList consSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;
    LwSciSyncModule syncModule = nullptr;
    LwSciSyncObj prodSyncObjs[4];

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();


    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,consSyncAttrList));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
        LwSciSyncAttrListFree(event.syncAttrList);

        getSyncObj(syncModule, prodSyncObjs[0]);

    // Test case
        ASSERT_EQ(LwSciError_IlwalidOperation, LwSciStreamBlockSyncObject(
                producer, 0, prodSyncObjs[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObject_NotImplemented}
 * @testcase{22059085}
 * @verify{19789638}
 * @testpurpose{Test negative scenario of LwSciStreamBlockSyncObject(), when
 * 'block' is not producer/consumer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer sends its sync object requirement to the consumer through LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends its sync object requirement to the producer through LwSciStreamBlockSyncRequirements().
 *   4. Query max number of sync objects using LwSciStreamAttributeQuery().
 *   5. Producer receives 'LwSciStreamEventType_SyncAttr' event from consumer by calling
 *      LwSciStreamBlockEventQuery().
 *   6. Producer sends its sync count to the consumer by calling LwSciStreamBlockSyncObjCount().
 *
 *   The call of LwSciStreamBlockSyncObject() API , should return LwSciError_NotImplemented,
 * when called with parameters block set to pool, index set to 0 and LwSciSyncObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObject()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObject_NotImplemented)
{
    // initial set up
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;
    int32_t maxNumSyncObjs = 0;
    LwSciStreamEvent event;
    uint32_t prodSyncCount = NUM_SYNCOBJS;
    LwSciSyncModule syncModule = nullptr;
    LwSciSyncObj prodSyncObjs[4];

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();


    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockSyncRequirements(producer,
                                             prodSynchronousOnly,prodSyncAttrList));

        ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,consSyncAttrList));

    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &maxNumSyncObjs));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);
        LwSciSyncAttrListFree(event.syncAttrList);

    // Producer sends its sync count to the consumer
    ASSERT_TRUE(prodSyncCount <= maxNumSyncObjs);
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

        getSyncObj(syncModule, prodSyncObjs[0]);

    // Test case
        ASSERT_EQ(LwSciError_NotImplemented, LwSciStreamBlockSyncObject(
                pool, 0, prodSyncObjs[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketElementCount_Success}
 * @testcase{22059087}
 * @verify{19789641}
 * @testpurpose{Test positive scenario of LwSciStreamBlockPacketElementCount(), when
 * packet element count is sent successfully.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketElementCount() API , should return LwSciError_Success,
 * when called with parameters block set to producer and elementCount set to NUM_PACKET_ELEMENTS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketElementCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketElementCount_Success)
{
    // initial set up
    uint32_t numElements = 2U;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
        elementCount = numElements;

    // Query maximum num of packet elements allowed
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Test case
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketElementCount_BadParameter1}
 * @testcase{22059091}
 * @verify{19789641}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketElementCount(), when
 * count exceeds the maximum allowed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketElementCount() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer and elementCount set to
 * more than MAX_ELEMENT_PER_PACKET.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketElementCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketElementCount_BadParameter1)
{
    // initial set up
    uint32_t numElements = 2U;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
        elementCount = numElements;

    // Query maximum num of packet elements allowed
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Test case
    ASSERT_EQ(LwSciError_BadParameter,
            LwSciStreamBlockPacketElementCount(producer, 25));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketElementCount_NotImplemented}
 * @testcase{22059092}
 * @verify{19789641}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketElementCount(), when
 * 'block' is not producer/consumer/pool.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketElementCount() API , should return LwSciError_NotImplemented,
 * when called with parameters block set to queue and elementCount set to NUM_PACKET_ELEMENTS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketElementCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketElementCount_NotImplemented)
{
    // initial set up
    uint32_t numElements = 2U;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
        elementCount = numElements;

    // Query maximum num of packet elements allowed
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Test case
    ASSERT_EQ(LwSciError_NotImplemented,
            LwSciStreamBlockPacketElementCount(queue[0], elementCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketElementCount_IlwalidState1}
 * @testcase{22059096}
 * @verify{19789641}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketElementCount(), when
 * block references a pool block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketElementCount() API , should return LwSciError_IlwalidState,
 * when called with parameters block set to pool and elementCount set to NUM_PACKET_ELEMENTS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketElementCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketElementCount_IlwalidState1)
{
    // initial set up
    uint32_t numElements = 2U;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
        elementCount = numElements;

    // Query maximum num of packet elements allowed
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Test case
    ASSERT_EQ(LwSciError_IlwalidState,
            LwSciStreamBlockPacketElementCount(pool, elementCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketElementCount_IlwalidState2}
 * @testcase{22059100}
 * @verify{19789641}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketElementCount(), when
 * the count is already sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *   3. Call LwSciStreamBlockPacketElementCount() with LwSciStreamBlock set to producer
 *      and elementCount set to NUM_PACKET_ELEMENTS.
 *
 *   The call of LwSciStreamBlockPacketElementCount() API , should return LwSciError_IlwalidState,
 * when called again with parameters block set to producer and elementCount set
 * to NUM_PACKET_ELEMENTS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketElementCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketElementCount_IlwalidState2)
{
    // initial set up
    uint32_t numElements = 2U;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
        elementCount = numElements;

    // Query maximum num of packet elements allowed
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));

    // Test case
    ASSERT_EQ(LwSciError_IlwalidState,
            LwSciStreamBlockPacketElementCount(producer, elementCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAttr_Success}
 * @testcase{22059103}
 * @verify{19789644}
 * @testpurpose{Test positive scenario of LwSciStreamBlockPacketAttr(), when
 * packet element information is sent successfully.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketAttr() API , should return LwSciError_Success,
 * when called with parameters block set to producer, Index of packet element, type of element ,
 * LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous and
 * LwSciBufAttrList set to rawBufAttrList.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAttr()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAttr_Success)
{
    // initial set up
    uint32_t numElements = 1U;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
    elementCount = numElements;

    // Query maximum num of packet elements allowed
    int32_t value;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Set the number of elements in producer's packet
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));

    // Set producer packet requirements
    uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
    LwSciStreamElementMode producerElementMode[MAX_ELEMENT_PER_PACKET];
    LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0U; i < elementCount; i++) {
        // Use index so that type is unique for each element.
        producerElementType[i] = i;
        producerElementMode[i] = LwSciStreamElementMode_Asynchronous;
        producerElementAttr[i] = rawBufAttrList;

        // Test case
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAttr(producer, i, i,
                                       LwSciStreamElementMode_Asynchronous, rawBufAttrList));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAttr_BadParameter1}
 * @testcase{22059106}
 * @verify{19789644}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAttr(), when
 * bufAttrList is null pointer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketAttr() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, Index of packet element, type of element ,
 * LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous and
 * LwSciBufAttrList set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAttr()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAttr_BadParameter1)
{
    // initial set up
    uint32_t numElements = 1U;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
    elementCount = numElements;

    // Query maximum num of packet elements allowed
    int32_t value;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Set the number of elements in producer's packet
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));

    // Set producer packet requirements
    uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
    LwSciStreamElementMode producerElementMode[MAX_ELEMENT_PER_PACKET];
    LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0U; i < elementCount; i++) {
        // Use index so that type is unique for each element.
        producerElementType[i] = i;
        producerElementMode[i] = LwSciStreamElementMode_Asynchronous;
        producerElementAttr[i] = rawBufAttrList;

        // Test case
        ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketAttr(producer, i, i,
                                       LwSciStreamElementMode_Asynchronous, nullptr));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAttr_BadParameter2}
 * @testcase{22059109}
 * @verify{19789644}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAttr(), when
 * index exceeds the maximum allowed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketAttr() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, Index set to 1, type of element ,
 * LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous and
 * LwSciBufAttrList set to rawBufAttrList.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAttr()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAttr_BadParameter2)
{
    // initial set up
    uint32_t numElements = 1U;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
    elementCount = numElements;

    // Query maximum num of packet elements allowed
    int32_t value;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));

    ASSERT_TRUE(elementCount <= value);

    // Set the number of elements in producer's packet
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));

    elementCount = value+1;

    // Set producer packet requirements
    uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
    LwSciStreamElementMode producerElementMode[MAX_ELEMENT_PER_PACKET];
    LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 1U; i < elementCount; i++) {
        // Use index so that type is unique for each element.
        producerElementType[i] = i;
        producerElementMode[i] = LwSciStreamElementMode_Asynchronous;
        producerElementAttr[i] = rawBufAttrList;

        // Test case
        ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketAttr(producer, i, i,
                                       LwSciStreamElementMode_Asynchronous, rawBufAttrList));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAttr_NotImplemented}
 * @testcase{22059112}
 * @verify{19789644}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAttr(), when
 * 'block' is not producer/consumer/pool.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketAttr() API , should return LwSciError_NotImplemented,
 * when called with parameters block set to queue, Index of packet element, type of element ,
 * LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous and
 * LwSciBufAttrList set to rawBufAttrList.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAttr()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAttr_NotImplemented)
{
    // initial set up
    uint32_t numElements = 1U;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
    elementCount = numElements;

    // Query maximum num of packet elements allowed
    int32_t value;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Set the number of elements in producer's packet
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));

    // Set producer packet requirements
    uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
    LwSciStreamElementMode producerElementMode[MAX_ELEMENT_PER_PACKET];
    LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0U; i < elementCount; i++) {
        // Use index so that type is unique for each element.
        producerElementType[i] = i;
        producerElementMode[i] = LwSciStreamElementMode_Asynchronous;
        producerElementAttr[i] = rawBufAttrList;

        // Test case
        ASSERT_EQ(LwSciError_NotImplemented, LwSciStreamBlockPacketAttr(queue[0], i, i,
                                       LwSciStreamElementMode_Asynchronous, rawBufAttrList));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAttr_IlwalidState1}
 * @testcase{22059115}
 * @verify{19789644}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAttr(), when
 * 'block' is pool block but LwSciStreamEventType_PacketAttrProducer and
 * LwSciStreamEventType_PacketAttrConsumer events are not yet queried from pool block}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *
 *   The call of LwSciStreamBlockPacketAttr() API , should return LwSciError_IlwalidState,
 * when called with parameters block set to pool, Index of packet element, type of element ,
 * LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous and
 * LwSciBufAttrList set to rawBufAttrList.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAttr()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAttr_IlwalidState1)
{
    // initial set up
    uint32_t numElements = 1U;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
    elementCount = numElements;

    // Query maximum num of packet elements allowed
    int32_t value;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Set the number of elements in producer's packet
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));

    // Set producer packet requirements
    uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
    LwSciStreamElementMode producerElementMode[MAX_ELEMENT_PER_PACKET];
    LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0U; i < elementCount; i++) {
        // Use index so that type is unique for each element.
        producerElementType[i] = i;
        producerElementMode[i] = LwSciStreamElementMode_Asynchronous;
        producerElementAttr[i] = rawBufAttrList;

        // Test case
        ASSERT_EQ(LwSciError_IlwalidState, LwSciStreamBlockPacketAttr(pool, i, i,
                                       LwSciStreamElementMode_Asynchronous, rawBufAttrList));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAttr_IlwalidState2}
 * @testcase{22059118}
 * @verify{19789644}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAttr(), when
 * packet element information for the index is already sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Query maximum num of packet elements allowed using LwSciStreamAttributeQuery().
 *   3. Call LwSciStreamBlockPacketAttr() with LwSciStreamBlock set to producer,
 *      index of packet element, type of element ,
 *      LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous and
 *      LwSciBufAttrList set to rawBufAttrList.
 *
 *   The call of LwSciStreamBlockPacketAttr() API , should return LwSciError_IlwalidState,
 * when called again with parameters block set to producer, same packet element index as that in the
 * previous step, type of element , LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous
 * and LwSciBufAttrList set to rawBufAttrList.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAttr()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAttr_IlwalidState2)
{
    // initial set up
    uint32_t numElements = 1U;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_TRUE(numElements <= MAX_ELEMENT_PER_PACKET);
    elementCount = numElements;

    // Query maximum num of packet elements allowed
    int32_t value;
    ASSERT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
    ASSERT_TRUE(elementCount <= value);

    // Set the number of elements in producer's packet
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(producer, elementCount));

    // Set producer packet requirements
    uint32_t producerElementType[MAX_ELEMENT_PER_PACKET];
    LwSciStreamElementMode producerElementMode[MAX_ELEMENT_PER_PACKET];
    LwSciBufAttrList producerElementAttr[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0U; i < elementCount; i++) {
        // Use index so that type is unique for each element.
        producerElementType[i] = i;
        producerElementMode[i] = LwSciStreamElementMode_Asynchronous;
        producerElementAttr[i] = rawBufAttrList;

        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAttr(producer, i, i,
                                       LwSciStreamElementMode_Asynchronous, rawBufAttrList));

        // Test case
        ASSERT_EQ(LwSciError_IlwalidState, LwSciStreamBlockPacketAttr(producer, i, i,
                                       LwSciStreamElementMode_Asynchronous, rawBufAttrList));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketCreate_Success}
 * @testcase{22059121}
 * @verify{19789647}
 * @testpurpose{Test positive scenario of LwSciStreamPoolPacketCreate(), when
 * packet successfully created for pool block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamPoolPacketCreate() API , should return LwSciError_Success,
 * when called with parameters pool set to pool block, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and reference of LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketCreate_Success)
{
    // initial set up
    uint32_t i;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    for (uint32_t i = 0U; i < numPackets; ++i) {
        // Choose pool's cookie and for new packet
        LwSciStreamPacket packetHandle;
        LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);

        // Test case
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketCreate_BadParameter1}
 * @testcase{22059123}
 * @verify{19789647}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketCreate(), when
 * 'handle' is null pointer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamPoolPacketCreate() API , should return LwSciError_BadParameter,
 * when called with parameters pool set to pool block, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and 'handle' set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketCreate_BadParameter1)
{
    // initial set up
    uint32_t i;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    for (uint32_t i = 0U; i < numPackets; ++i) {
        // Choose pool's cookie and for new packet
        LwSciStreamPacket packetHandle;
        LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);

        // Test case
            ASSERT_EQ(LwSciError_BadParameter,
                LwSciStreamPoolPacketCreate(pool, poolCookie, nullptr));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketCreate_BadParameter2}
 * @testcase{22059126}
 * @verify{19789647}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketCreate(), when
 * 'cookie' is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamPoolPacketCreate() API , should return LwSciError_BadParameter,
 * when called with parameters pool set to pool block, LwSciStreamCookie set to
 * LwSciStreamCookie_Ilwalid and reference of LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketCreate_BadParameter2)
{
    // initial set up
    uint32_t i;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    for (uint32_t i = 0U; i < numPackets; ++i) {
        // Choose pool's cookie and for new packet
        LwSciStreamPacket packetHandle;

        // Test case
            ASSERT_EQ(LwSciError_BadParameter,
                LwSciStreamPoolPacketCreate(pool, LwSciStreamCookie_Ilwalid, &packetHandle));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketCreate_NotImplemented}
 * @testcase{22059129}
 * @verify{19789647}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketCreate(), when
 * pool is valid but it does not reference a pool block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamPoolPacketCreate() API , should return LwSciError_NotImplemented,
 * when called with parameters pool set to producer block, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and reference of LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketCreate_NotImplemented)
{
    // initial set up
    uint32_t i;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    for (uint32_t i = 0U; i < numPackets; ++i) {
        // Choose pool's cookie and for new packet
        LwSciStreamPacket packetHandle;
        LwSciStreamCookie poolCookie
            = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);

        // Test case
            ASSERT_EQ(LwSciError_NotImplemented,
                LwSciStreamPoolPacketCreate(producer, poolCookie, &packetHandle));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketCreate_InsufficientMemory}
 * @testcase{22059133}
 * @verify{19789647}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketCreate(), when
 * unable to create a new packet instance.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamPoolPacketCreate() API , should return LwSciError_InsufficientMemory,
 * when called with parameters pool set to pool block, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and reference of LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::createPacket()}
 * @verifyFunction{LwSciStreamPoolPacketCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketCreate_InsufficientMemory)
{
    // initial set up
    using ::testing::_;
    using ::testing::Return;
    using ::testing::NiceMock;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    // Choose pool's cookie and for new packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie;

     EXPECT_CALL(*poolPtr, createPacket(_,_))
               .Times(1)
               .WillOnce(Return(LwSciError_InsufficientMemory));

    // Test case
    ASSERT_EQ(LwSciError_InsufficientMemory,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketCreate_IlwalidOperation}
 * @testcase{22059135}
 * @verify{19789647}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketCreate(), when
 * pool has reached its limit with the maximum number of packets that can be created from it.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool with 1 packet, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Call LwSciStreamPoolPacketCreate() with LwSciStreamBlock set to pool, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(0 + COOKIE_BASE) and reference of LwSciStreamPacket.
 *
 *   The call of LwSciStreamPoolPacketCreate() API , should return LwSciError_IlwalidOperation,
 * when called again with parameters pool set to pool block, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(1 + COOKIE_BASE) and reference of LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketCreate_IlwalidOperation)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);
    connectStream();
    packetAttrSetup();

        LwSciStreamPacket packetHandle;
        // Choose pool's cookie and for new packet
        LwSciStreamCookie poolCookie1
            = static_cast<LwSciStreamCookie>(0 + COOKIE_BASE);

        LwSciStreamCookie poolCookie2
            = static_cast<LwSciStreamCookie>(1 + COOKIE_BASE);

            ASSERT_EQ(LwSciError_Success,
                LwSciStreamPoolPacketCreate(pool, poolCookie1, &packetHandle));

        // Test case
            ASSERT_EQ(LwSciError_IlwalidOperation,
                LwSciStreamPoolPacketCreate(pool, poolCookie2, &packetHandle));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketInsertBuffer_Success}
 * @testcase{22059139}
 * @verify{19789650}
 * @testpurpose{Test positive scenario of LwSciStreamPoolPacketInsertBuffer(), when
 * bufObj to the packet element has been successfully registered}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *
 *   The call of LwSciStreamPoolPacketInsertBuffer() API , should return LwSciError_Success,
 * when called with parameters pool set to pool block, LwSciStreamPacket,
 * index of element and reference of LwSciBufObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketInsertBuffer()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketInsertBuffer_Success)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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

            // Test case
            ASSERT_EQ(LwSciError_Success,
               LwSciStreamPoolPacketInsertBuffer(pool,
                                                 packetHandle, k,
                                                 poolElementBuf[k]));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketInsertBuffer_BadParameter1}
 * @testcase{22059142}
 * @verify{19789650}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketInsertBuffer(), when
 * bufObj is null pointer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *
 *   The call of LwSciStreamPoolPacketInsertBuffer() API , should return LwSciError_BadParameter,
 * when called with parameters pool set to pool block, LwSciStreamPacket,
 * index of element 'bufObj' is nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketInsertBuffer()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketInsertBuffer_BadParameter1)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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

            // Test case
            ASSERT_EQ(LwSciError_BadParameter,
               LwSciStreamPoolPacketInsertBuffer(pool,
                                                 packetHandle, k,
                                                 nullptr));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketInsertBuffer_BadParameter2}
 * @testcase{22059144}
 * @verify{19789650}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketInsertBuffer(), when
 * index exceeds the maximum allowed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *
 *   The call of LwSciStreamPoolPacketInsertBuffer() API , should return LwSciError_BadParameter,
 * when called with parameters pool set to pool block, LwSciStreamPacket,
 * 'index' set to 2 (invalid) and reference of LwSciBufObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketInsertBuffer()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketInsertBuffer_BadParameter2)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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

            // Test case
            ASSERT_EQ(LwSciError_BadParameter,
               LwSciStreamPoolPacketInsertBuffer(pool,
                                                 packetHandle, 2,
                                                 poolElementBuf[k]));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketInsertBuffer_NotImplemented}
 * @testcase{22059147}
 * @verify{19789650}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketInsertBuffer(), when
 * argument pool is valid but it does not reference a pool block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *
 *   The call of LwSciStreamPoolPacketInsertBuffer() API , should return LwSciError_NotImplemented,
 * when called with parameters pool set to producer block, LwSciStreamPacket,
 * index of element and reference of LwSciBufObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketInsertBuffer()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketInsertBuffer_NotImplemented)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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

            // Test case
            ASSERT_EQ(LwSciError_NotImplemented,
               LwSciStreamPoolPacketInsertBuffer(producer,
                                                 packetHandle, k,
                                                 poolElementBuf[k]));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketInsertBuffer_IlwalidState}
 * @testcase{22059150}
 * @verify{19789650}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketInsertBuffer(), when
 * bufObj for the same index is already registered.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Call LwSciStreamPoolPacketInsertBuffer() with LwSciStreamBlock set to pool, LwSciStreamPacket,
 * index of element and reference of LwSciBufObj.
 *
 *   The call of LwSciStreamPoolPacketInsertBuffer() API , should return LwSciError_IlwalidState,
 * when called again with parameters pool set to pool block, LwSciStreamPacket,
 * same element index as that in the previous step and reference of LwSciBufObj.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketInsertBuffer()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketInsertBuffer_IlwalidState)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    for (uint32_t i = 0U; i < 1; ++i) {
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

            // Test case
            ASSERT_EQ(LwSciError_IlwalidState,
               LwSciStreamPoolPacketInsertBuffer(pool,
                                                 packetHandle, k,
                                                 poolElementBuf[k]));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketDelete_Success}
 * @testcase{22059152}
 * @verify{19789653}
 * @testpurpose{Test positive scenario of LwSciStreamPoolPacketDelete(), when
 * packet successfully deleted from the pool block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create packets
 *
 *   The call of LwSciStreamPoolPacketDelete() API , should return LwSciError_Success,
 * when called with parameters pool set to pool block and LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketDelete()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketDelete_Success)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();
    createPacket();

    for (auto iter = poolCPMap.begin(); iter != poolCPMap.end(); ++iter) {
        LwSciStreamPacket packet = iter->second;

        // Test case
        ASSERT_EQ(LwSciError_Success, LwSciStreamPoolPacketDelete(pool, packet));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketDelete_BadParameter1}
 * @testcase{22059156}
 * @verify{19789653}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketDelete(), when
 * the Packet is already marked for removal.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create packets
 *   4. Delete packets
 *
 *   The call of LwSciStreamPoolPacketDelete() API , should return LwSciError_BadParameter,
 * when called with parameters pool set to pool block and LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketDelete()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketDelete_BadParameter1)
{
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketDelete_NotImplemented}
 * @testcase{22059158}
 * @verify{19789653}
 * @testpurpose{Test negative scenario of LwSciStreamPoolPacketDelete(), when
 * argument pool is valid but it does not reference a pool block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create packets
 *
 *   The call of LwSciStreamPoolPacketDelete() API , should return LwSciError_NotImplemented,
 * when called with parameters pool set to producer block and LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketDelete()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketDelete_NotImplemented)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();
    createPacket();

    for (auto iter = poolCPMap.begin(); iter != poolCPMap.end(); ++iter) {
        LwSciStreamPacket packet = iter->second;

        // Test case
        ASSERT_EQ(LwSciError_NotImplemented, LwSciStreamPoolPacketDelete(producer, packet));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAccept_Success}
 * @testcase{22059162}
 * @verify{19789656}
 * @testpurpose{Test positive scenario of LwSciStreamBlockPacketAccept(), when
 * packet cookie/status is successfully reported.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 * LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockPacketAccept() API , should return LwSciError_Success,
 * when called with parameters block set to producer, LwSciStreamPacket, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAccept_Success)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
                LwSciStreamPoolPacketInsertBuffer(pool, packetHandle, k,
                                                  poolElementBuf[k]));
        }

        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
                    = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
                    producer, producerPacket, producerCookie, producerError));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAccept_BadParameter1}
 * @testcase{22059165}
 * @verify{19789656}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAccept(), when
 * cookie is LwSciStreamCookie_Ilwalid but the err is LwSciError_Success.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 * LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockPacketAccept() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, LwSciStreamPacket, LwSciStreamCookie set to
 * 0U and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAccept_BadParameter1)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
                LwSciStreamPoolPacketInsertBuffer(pool, packetHandle, k,
                                                  poolElementBuf[k]));
        }

        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie nullCookie =  static_cast<LwSciStreamCookie>(0U);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketAccept(
                    producer, producerPacket, nullCookie, producerError));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAccept_BadParameter2}
 * @testcase{22059169}
 * @verify{19789656}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAccept(), when
 * cookie is valid but the err is not LwSciError_Success.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 * LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockPacketAccept() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, LwSciStreamPacket, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and LwSciError set to 'LwSciError_BadParameter'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAccept_BadParameter2)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
                LwSciStreamPoolPacketInsertBuffer(pool, packetHandle, k,
                                                  poolElementBuf[k]));
        }

        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
                    = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_BadParameter;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketAccept(
                    producer, producerPacket, producerCookie, producerError));
    }
}


/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAccept_NotImplemented}
 * @testcase{22059172}
 * @verify{19789656}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAccept(), when
 * 'block' is not a producer/consumer block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 * LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockPacketAccept() API , should return LwSciError_NotImplemented,
 * when called with parameters block set to pool, LwSciStreamPacket, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAccept_NotImplemented)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
                LwSciStreamPoolPacketInsertBuffer(pool, packetHandle, k,
                                                  poolElementBuf[k]));
        }

        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
                    = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_NotImplemented, LwSciStreamBlockPacketAccept(
                    pool, producerPacket, producerCookie, producerError));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAccept_BadParameter3}
 * @testcase{22059174}
 * @verify{19789656}
 * @testpurpose{Test negative scenario of LwSciStreamBlockPacketAccept(), when
 * status for the packet is sent already.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using
 *      LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate'
 *      by calling LwSciStreamBlockEventQuery().
 *   5. Call LwSciStreamBlockPacketAccept() with LwSciStreamBlock set to
 * producer, LwSciStreamPacket, LwSciStreamCookie set to
 * static_cast(LwSciStreamCookie)(i + COOKIE_BASE) and LwSciError set to
 * 'LwSciError_Success'
 *
 *   The call of LwSciStreamBlockPacketAccept() API , should return
 * LwSciError_BadParameter, when called again with parameters block set to
 * producer, same LwSciStreamPacket as that in the previous step,
 * LwSciStreamCookie set to static_cast(LwSciStreamCookie)(i + COOKIE_BASE)
 * and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAccept_BadParameter3)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
                LwSciStreamPoolPacketInsertBuffer(pool, packetHandle, k,
                                                  poolElementBuf[k]));
        }

        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
                    = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
                    producer, producerPacket, producerCookie, producerError));

        ASSERT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketAccept(
                    producer, producerPacket, producerCookie, producerError));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockElementAccept_Success}
 * @testcase{22059176}
 * @verify{19789659}
 * @testpurpose{Test positive scenario of LwSciStreamBlockElementAccept(), when
 * packet element status is successfully reported.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 * LwSciStreamBlockEventQuery().
 *   5. Producer accepts a packet provided by the pool by calling LwSciStreamBlockPacketAccept().
 *   6. Producer receives event 'LwSciStreamEventType_PacketElement' by calling
 * LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockElementAccept() API , should return LwSciError_Success,
 * when called with parameters block set to producer, LwSciStreamPacket, index of the element
 * and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockElementAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockElementAccept_Success)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
            = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
            producer, producerPacket, producerCookie, producerError));

        // Save the cookie-to-handle mapping
        prodCPMap.emplace(producerCookie, producerPacket);

        for (uint32_t k = 0; k < elementCount; ++k) {
            EXPECT_EQ(LwSciError_Success,
                   LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

            producerPacket = prodCPMap[event.packetCookie];

    // Test case
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockElementAccept(producer,
                    producerPacket, event.index, producerError));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockElementAccept_BadParameter1}
 * @testcase{22059178}
 * @verify{19789659}
 * @testpurpose{Test negative scenario of LwSciStreamBlockElementAccept(), when
 * 'index' is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 *      LwSciStreamBlockEventQuery().
 *   5. Producer accepts a packet provided by the pool by calling LwSciStreamBlockPacketAccept().
 *   6. Producer receives event 'LwSciStreamEventType_PacketElement' by calling
 *      LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockElementAccept() API , should return LwSciError_BadParameter,
 * when called with parameters block set to producer, LwSciStreamPacket, 'index' set to invalid
 * and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockElementAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockElementAccept_BadParameter1)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
        for (uint32_t k = 0; k < 1; ++k) {
            makeRawBuffer(rawBufAttrList, poolElementBuf[k]);
            ASSERT_EQ(LwSciError_Success,
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
            = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
            producer, producerPacket, producerCookie, producerError));

        // Save the cookie-to-handle mapping
        prodCPMap.emplace(producerCookie, producerPacket);

        for (uint32_t k = 0; k < 1; ++k) {
            EXPECT_EQ(LwSciError_Success,
                   LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

            producerPacket = prodCPMap[event.packetCookie];

    // Test case
            EXPECT_EQ(LwSciError_BadParameter,
                LwSciStreamBlockElementAccept(producer,
                    producerPacket, 2, producerError));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockElementAccept_NotImplemented}
 * @testcase{22059180}
 * @verify{19789659}
 * @testpurpose{Test negative scenario of LwSciStreamBlockElementAccept(), when
 * argument block is not producer/consumer block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 * LwSciStreamBlockEventQuery().
 *   5. Producer accepts a packet provided by the pool by calling LwSciStreamBlockPacketAccept().
 *   6. Producer receives event 'LwSciStreamEventType_PacketElement' by calling
 * LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamBlockElementAccept() API , should return LwSciError_NotImplemented,
 * when called with parameters block set to pool, LwSciStreamPacket, index of the element
 * and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockElementAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockElementAccept_NotImplemented)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
            = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
            producer, producerPacket, producerCookie, producerError));

        // Save the cookie-to-handle mapping
        prodCPMap.emplace(producerCookie, producerPacket);

        for (uint32_t k = 0; k < elementCount; ++k) {
            EXPECT_EQ(LwSciError_Success,
                   LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

            producerPacket = prodCPMap[event.packetCookie];

    //Test case
            EXPECT_EQ(LwSciError_NotImplemented,
                LwSciStreamBlockElementAccept(pool,
                    producerPacket, event.index, producerError));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockElementAccept_IlwalidState}
 * @testcase{22059181}
 * @verify{19789659}
 * @testpurpose{Test negative scenario of LwSciStreamBlockElementAccept(), when
 *  status for the same packet element is sent already.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create a new packet and adds it to the pool block using LwSciStreamPoolPacketCreate().
 *   4. Producer receives PacketCreate event 'LwSciStreamEventType_PacketCreate' by calling
 * LwSciStreamBlockEventQuery().
 *   5. Producer accepts a packet provided by the pool by calling LwSciStreamBlockPacketAccept().
 *   6. Producer receives event 'LwSciStreamEventType_PacketElement' by calling
 * LwSciStreamBlockEventQuery().
 *   7.Call LwSciStreamBlockElementAccept() with LwSciStreamBlock set to producer, LwSciStreamPacket,
 * index of the element and LwSciError set to 'LwSciError_Success
 *
 *   The call of LwSciStreamBlockElementAccept() API , should return LwSciError_IlwalidState,
 * when called again with parameters block set to producer, LwSciStreamPacket, same element index
 * as that in the previous step and LwSciError set to 'LwSciError_Success'.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockElementAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockElementAccept_IlwalidState)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

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
        // Producer receives PacketCreate event
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
            = static_cast<LwSciStreamCookie>(i + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
        ASSERT_EQ(LwSciError_Success, LwSciStreamBlockPacketAccept(
            producer, producerPacket, producerCookie, producerError));

        // Save the cookie-to-handle mapping
        prodCPMap.emplace(producerCookie, producerPacket);

        for (uint32_t k = 0; k < elementCount; ++k) {
            EXPECT_EQ(LwSciError_Success,
                   LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

            producerPacket = prodCPMap[event.packetCookie];

            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockElementAccept(producer,
                    producerPacket, event.index, producerError));

    //Test case
            EXPECT_EQ(LwSciError_IlwalidState,
                LwSciStreamBlockElementAccept(producer,
                    producerPacket, event.index, producerError));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketGet_Success}
 * @testcase{22059184}
 * @verify{19789662}
 * @testpurpose{Test positive scenario of LwSciStreamProducerPacketGet(), when
 * packet successfully retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. The producer block receives event 'LwSciStreamEventType_PacketReady' from pool
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamProducerPacketGet() API , should return LwSciError_Success,
 * when called with parameters producer set to producer block, reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketGet()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketGet_Success)
{
    // initial set up
    uint32_t numFrames =1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }

        // Test case
        ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketGet_BadParameter1}
 * @testcase{22059186}
 * @verify{19789662}
 * @testpurpose{Test negative scenario of LwSciStreamProducerPacketGet(), when
 * 'cookie' is null pointer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. The producer block receives event 'LwSciStreamEventType_PacketReady' from pool
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamProducerPacketGet() API , should return LwSciError_BadParameter,
 * when called with parameters producer set to producer block, 'cookie' set to nullptr
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketGet()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketGet_BadParameter1)
{
    // initial set up
    uint32_t numFrames =1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }

        // Test case
        ASSERT_EQ(LwSciError_BadParameter,
                LwSciStreamProducerPacketGet(producer, nullptr, fences));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketGet_BadParameter2}
 * @testcase{22059188}
 * @verify{19789662}
 * @testpurpose{Test negative scenario of LwSciStreamProducerPacketGet(), when
 * prefences is null pointer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. The producer block receives event 'LwSciStreamEventType_PacketReady' from pool
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamProducerPacketGet() API , should return LwSciError_BadParameter,
 * when called with parameters producer set to producer block, reference of LwSciStreamCookie
 * and 'prefences' set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketGet()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketGet_BadParameter2)
{
    // initial set up
    uint32_t numFrames =1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }

        // Test case
        ASSERT_EQ(LwSciError_BadParameter,
                LwSciStreamProducerPacketGet(producer, &cookie, nullptr));
    }
}


/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketGet_NotImplemented}
 * @testcase{22059190}
 * @verify{19789662}
 * @testpurpose{Test negative scenario of LwSciStreamProducerPacketGet(), when
 * argument producer is valid but it does not reference a producer block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. The producer block receives event 'LwSciStreamEventType_PacketReady' from pool
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamProducerPacketGet() API , should return LwSciError_NotImplemented,
 * when called with parameters producer set to pool block, reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketGet()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketGet_NotImplemented)
{
    // initial set up
    uint32_t numFrames =1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
        LwSciStreamCookie cookie;

        // Pool sends packet ready event to producer
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            producer, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

        // Producer get a packet from the pool
        for (uint32_t j = 0U; j < totalConsSync; j++) {
            fences[j] = LwSciSyncFenceInitializer;
        }

        // Test case
        ASSERT_EQ(LwSciError_NotImplemented,
                LwSciStreamProducerPacketGet(pool, &cookie, fences));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketGet_NoStreamPacket}
 * @testcase{22059192}
 * @verify{19789662}
 * @testpurpose{Test negative scenario of LwSciStreamProducerPacketGet(), when
 * No packet available with the pool.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. The producer block receives event 'LwSciStreamEventType_PacketReady' from pool
 * by calling LwSciStreamBlockEventQuery().
 *   4. Get a packet from the pool by calling LwSciStreamProducerPacketGet().
 *
 *   The call of LwSciStreamProducerPacketGet() API , should return LwSciError_NoStreamPacket,
 * when called again with parameters producer set to producer block, reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketGet()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketGet_NoStreamPacket)
{
    // initial set up
    uint32_t numFrames =1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        ASSERT_EQ(LwSciError_NoStreamPacket,
                LwSciStreamProducerPacketGet(producer, &cookie, fences));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketPresent_Success}
 * @testcase{22059194}
 * @verify{19789665}
 * @testpurpose{Test positive scenario of LwSciStreamProducerPacketPresent(), when
 * packet successfully presented.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets packet by calling LwSciStreamProducerPacketGet().
 *
 *   The call of LwSciStreamProducerPacketPresent() API , should return LwSciError_Success,
 * when called with parameters producer set to producer block, LwSciStreamPacket
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketPresent()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketPresent_Success)
{
    // initial set up
    uint32_t numFrames=1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        // Test case
        ASSERT_EQ(LwSciError_Success,
                LwSciStreamProducerPacketPresent(producer, handle, fences));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketPresent_BadParameter1}
 * @testcase{22059196}
 * @verify{19789665}
 * @testpurpose{Test negative scenario of LwSciStreamProducerPacketPresent(), when
 * 'postfences' is null pointer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets packet by calling LwSciStreamProducerPacketGet().
 *
 *   The call of LwSciStreamProducerPacketPresent() API , should return LwSciError_BadParameter,
 * when called with parameters producer set to producer block, LwSciStreamPacket
 * and 'postfences' set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketPresent()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketPresent_BadParameter1)
{
    // initial set up
    uint32_t numFrames=1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        // Test case
        ASSERT_EQ(LwSciError_BadParameter,
                LwSciStreamProducerPacketPresent(producer, handle, nullptr));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketPresent_NotImplemented}
 * @testcase{22059197}
 * @verify{19789665}
 * @testpurpose{Test negative scenario of LwSciStreamProducerPacketPresent(), when
 * argument producer is valid but it does not reference a producer block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets packet by calling LwSciStreamProducerPacketGet().
 *
 *   The call of LwSciStreamProducerPacketPresent() API , should return LwSciError_NotImplemented,
 * when called with parameters producer set to pool block, LwSciStreamPacket
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketPresent()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketPresent_NotImplemented)
{
    // initial set up
    uint32_t numFrames=1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        // Test case
        ASSERT_EQ(LwSciError_NotImplemented,
                LwSciStreamProducerPacketPresent(pool, handle, fences));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketPresent_IlwalidOperation}
 * @testcase{22059199}
 * @verify{19789665}
 * @testpurpose{Test negative scenario of LwSciStreamProducerPacketPresent(), when
 * packet is not lwrrently held by the application.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets packet by calling LwSciStreamProducerPacketGet().
 *   4. Call LwSciStreamProducerPacketPresent() with LwSciStreamBlock set to producer,
 * LwSciStreamPacket and Pointer to an array of LwSciSyncFence.
 *
 *   The call of LwSciStreamProducerPacketPresent() API , should return LwSciError_IlwalidOperation,
 * when called again with parameters producer set to producer block, same LwSciStreamPacket as that
 * in the previous step and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketPresent()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketPresent_IlwalidOperation)
{
    // initial set up
    uint32_t numFrames=1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        // Test case
        ASSERT_EQ(LwSciError_IlwalidOperation,
                LwSciStreamProducerPacketPresent(producer, handle, fences));
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketAcquire_Success}
 * @testcase{22059202}
 * @verify{19789668}
 * @testpurpose{Test positive scenario of LwSciStreamConsumerPacketAcquire(), when
 * packet was successfully acquired.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. The consumer block receives event 'LwSciStreamEventType_PacketReady' from queue
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamConsumerPacketAcquire() API , should return LwSciError_Success,
 * when called with parameters consumer set to consumer block, reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketAcquire()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketAcquire_Success)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            // Test case
            ASSERT_EQ(LwSciError_Success, LwSciStreamConsumerPacketAcquire
                                     (consumer[n], &cookie, fences));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketAcquire_BadParameter1}
 * @testcase{22059204}
 * @verify{19789668}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerPacketAcquire(), when
 * 'cookie' is null pointer.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. The consumer block receives event 'LwSciStreamEventType_PacketReady' from queue
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamConsumerPacketAcquire() API , should return LwSciError_BadParameter,
 * when called with parameters consumer set to consumer block, 'cookie' set to nullptr
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketAcquire()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketAcquire_BadParameter1)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            // Test case
            ASSERT_EQ(LwSciError_BadParameter, LwSciStreamConsumerPacketAcquire
                                     (consumer[n], nullptr, fences));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketAcquire_BadParameter2}
 * @testcase{22059206}
 * @verify{19789668}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerPacketAcquire(), when
 * 'prefences' is NULL.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. The consumer block receives event 'LwSciStreamEventType_PacketReady' from queue
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamConsumerPacketAcquire() API , should return LwSciError_BadParameter,
 * when called with parameters consumer set to consumer block, reference of LwSciStreamCookie
 * and 'prefences' set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketAcquire()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketAcquire_BadParameter2)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            // Test case
            ASSERT_EQ(LwSciError_BadParameter, LwSciStreamConsumerPacketAcquire
                                     (consumer[n], &cookie, nullptr));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketAcquire_NotImplemented}
 * @testcase{22059208}
 * @verify{19789668}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerPacketAcquire(), when
 * argument consumer is valid but it does not reference a consumer block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. The consumer block receives event 'LwSciStreamEventType_PacketReady' from queue
 * by calling LwSciStreamBlockEventQuery().
 *
 *   The call of LwSciStreamConsumerPacketAcquire() API , should return LwSciError_NotImplemented,
 * when called with parameters consumer set to pool block, reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketAcquire()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketAcquire_NotImplemented)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            // Test case
            ASSERT_EQ(LwSciError_NotImplemented, LwSciStreamConsumerPacketAcquire
                                     (pool, &cookie, fences));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketAcquire_NoStreamPacket}
 * @testcase{22059209}
 * @verify{19789668}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerPacketAcquire(), when
 * No packet is available with the queue.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. Consumer acquires a packet by calling LwSciStreamConsumerPacketAcquire().
 *
 *   The call of LwSciStreamConsumerPacketAcquire() API , should return LwSciError_NoStreamPacket,
 * when called again with parameters consumer set to consumer block, reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketAcquire()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketAcquire_NoStreamPacket)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            ASSERT_EQ(LwSciError_Success, LwSciStreamConsumerPacketAcquire
                                     (consumer[n], &cookie, fences));

            // Test case
            ASSERT_EQ(LwSciError_NoStreamPacket, LwSciStreamConsumerPacketAcquire
                                     (consumer[n], &cookie, fences));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketRelease_Success}
 * @testcase{22059212}
 * @verify{19789671}
 * @testpurpose{Test positive scenario of LwSciStreamConsumerPacketRelease(), when
 * packet was successfully released.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. Consumer gets packet by calling LwSciStreamConsumerPacketAcquire().
 *
 *   The call of LwSciStreamConsumerPacketRelease() API , should return LwSciError_Success,
 * when called with parameters consumer set to consumer block, LwSciStreamPacket
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketRelease()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketRelease_Success)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

     LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

     for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[n],
                                                 &cookie, fences));

            handle = (consCPMap[n])[cookie];

            // Consumer returns a data packet to the stream
            for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            // Test case
            ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketRelease(consumer[n],
                                                     handle, fences));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketRelease_BadParameter1}
 * @testcase{22059214}
 * @verify{19789671}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerPacketRelease(), when
 * postfences is NULL.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. Consumer gets packet by calling LwSciStreamConsumerPacketAcquire().
 *
 *   The call of LwSciStreamConsumerPacketRelease() API , should return LwSciError_BadParameter,
 * when called with parameters consumer set to consumer block, LwSciStreamPacket
 * and 'postfences' is NULL.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketRelease()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketRelease_BadParameter1)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

     LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

     for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < 1; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[n],
                                                 &cookie, fences));

            handle = (consCPMap[n])[cookie];

            // Consumer returns a data packet to the stream
            for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            // Test case
            ASSERT_EQ(LwSciError_BadParameter,
                    LwSciStreamConsumerPacketRelease(consumer[n],
                                                     handle, nullptr));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketRelease_NotImplemented}
 * @testcase{22059216}
 * @verify{19789671}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerPacketRelease(), when
 * argument consumer is valid but it does not reference a consumer block.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. Consumer gets packet by calling LwSciStreamConsumerPacketAcquire().
 *
 *   The call of LwSciStreamConsumerPacketRelease() API , should return LwSciError_NotImplemented,
 * when called with parameters consumer set to pool block, LwSciStreamPacket
 * and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketRelease()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketRelease_NotImplemented)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

     LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

     for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[n],
                                                 &cookie, fences));

            handle = (consCPMap[n])[cookie];

            // Consumer returns a data packet to the stream
            for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            // Test case
            ASSERT_EQ(LwSciError_NotImplemented,
                    LwSciStreamConsumerPacketRelease(pool,
                                                     handle, fences));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketRelease_IlwalidOperation}
 * @testcase{22059217}
 * @verify{19789671}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerPacketRelease(), when
 * packet is not lwrrently held by the application.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *   3. Producer gets a packet by calling LwSciStreamProducerPacketGet().
 *   4. Producer presents a packet by calling LwSciStreamProducerPacketPresent().
 *   5. Consumer gets packet by calling LwSciStreamConsumerPacketAcquire().
 *   6. Call LwSciStreamConsumerPacketRelease() with LwSciStreamBlock set to consumer,
 * LwSciStreamPacket and Pointer to an array of LwSciSyncFence
 *
 *   The call of LwSciStreamConsumerPacketRelease() API , should return LwSciError_IlwalidOperation,
 * when called again with parameters consumer set to consumer block, same LwSciStreamPacket
 * as that in the previous step and Pointer to an array of LwSciSyncFence.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketRelease()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketRelease_IlwalidOperation)
{
    // initial set up
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

     LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

     for (uint32_t i = 0; i < numFrames; ++i) {
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

        for (uint32_t n = 0U; n < numConsumers; n++) {
            // Pool sends packet ready event to consumer
            EXPECT_EQ(LwSciError_Success,
                LwSciStreamBlockEventQuery(consumer[n], EVENT_QUERY_TIMEOUT, &event));
            EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

            //  Consumer gets a packet from the queue
            for (uint32_t j = 0U; j < prodSyncCount; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }
            ASSERT_EQ(LwSciError_Success,
                LwSciStreamConsumerPacketAcquire(consumer[n],
                                                 &cookie, fences));

            handle = (consCPMap[n])[cookie];

            // Consumer returns a data packet to the stream
            for (uint32_t j = 0U; j < consSyncCount[n]; j++) {
                fences[j] = LwSciSyncFenceInitializer;
            }

            ASSERT_EQ(LwSciError_Success,
                    LwSciStreamConsumerPacketRelease(consumer[n],
                                                     handle, fences));

            // Test case
            ASSERT_EQ(LwSciError_IlwalidOperation,
                    LwSciStreamConsumerPacketRelease(consumer[n],
                                                     handle, fences));
        }
    }
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockDelete_Success}
 * @testcase{22059219}
 * @verify{19789674}
 * @testpurpose{Test positive scenario of LwSciStreamBlockDelete(), when
 * block successfully destroyed.}
 * @testbehavior{
 * Setup:
 *    Create a producer block by calling createBlocks().
 *
 *   The call of LwSciStreamBlockDelete() API , should return LwSciError_Success,
 * when called with parameter block set to producer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockDelete()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockDelete_Success)
{
    // initial set up
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamBlockDelete(producer));
    producer = 0U;

    EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockEventQuery(pool, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamAttributeQuery_Success1}
 * @testcase{22059221}
 * @verify{19789677}
 * @testpurpose{Test positive scenario of LwSciStreamAttributeQuery(), when
 * querying attribute LwSciStreamQueryableAttrib_MaxElements is successful.}
 * @testbehavior{
 * Setup:
 *    Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamAttributeQuery() API , should return LwSciError_Success,
 * when called with LwSciStreamQueryableAttrib set to 'LwSciStreamQueryableAttrib_MaxElements'
 * and the maximum number of packet elements returned should be equal to
 * LwSciStream::MAX_PACKET_ELEMENTS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamAttributeQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamAttributeQuery_Success1)
{
    // initial set up
    uint32_t numElements = NUM_PACKET_ELEMENTS;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    EXPECT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));

    EXPECT_EQ(LwSciStream::MAX_PACKET_ELEMENTS, value);
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamAttributeQuery_Success2}
 * @testcase{22060068}
 * @verify{19789677}
 * @testpurpose{Test positive scenario of LwSciStreamAttributeQuery(), when
 * querying attribute LwSciStreamQueryableAttrib_MaxSyncObj is successful.}
 * @testbehavior{
 * Setup:
 *    Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamAttributeQuery() API , should return LwSciError_Success,
 * when called with LwSciStreamQueryableAttrib set to 'LwSciStreamQueryableAttrib_MaxSyncObj'
 * and the maximum number of packet elements returned should be equal to
 * LwSciStream::MAX_SYNC_OBJECTS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamAttributeQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamAttributeQuery_Success2)
{
    // initial set up
    uint32_t numElements = NUM_PACKET_ELEMENTS;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    EXPECT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxSyncObj, &value));

    EXPECT_EQ(LwSciStream::MAX_SYNC_OBJECTS, value);
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamAttributeQuery_Success3}
 * @testcase{22060071}
 * @verify{19789677}
 * @testpurpose{Test positive scenario of LwSciStreamAttributeQuery(), when
 * querying attribute LwSciStreamQueryableAttrib_MaxMulticastOutputs is successful.}
 * @testbehavior{
 * Setup:
 *    Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamAttributeQuery() API , should return LwSciError_Success,
 * when called with LwSciStreamQueryableAttrib set to
 * 'LwSciStreamQueryableAttrib_MaxMulticastOutputs' and the maximum number of
 * packet elements returned should be equal to LwSciStream::MAX_DST_CONNECTIONS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamAttributeQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamAttributeQuery_Success3)
{
    // initial set up
    uint32_t numElements = NUM_PACKET_ELEMENTS;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    EXPECT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxMulticastOutputs, &value));

    EXPECT_EQ(LwSciStream::MAX_DST_CONNECTIONS, value);
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamAttributeQuery_BadParameter1}
 * @testcase{22059223}
 * @verify{19789677}
 * @testpurpose{Test negative scenario of LwSciStreamAttributeQuery(), when
 * 'attr' is invalid.}
 * @testbehavior{
 * Setup:
 *    Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamAttributeQuery() API , should return LwSciError_BadParameter,
 * when called with LwSciStreamQueryableAttrib set to
 * static_cast(LwSciStreamQueryableAttrib)(ilwalidQueryAttrib) and reference of value.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamAttributeQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamAttributeQuery_BadParameter1)
{
    // initial set up
    uintptr_t ilwalidQueryAttrib = 100;
    int32_t value;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamAttributeQuery(
            static_cast<LwSciStreamQueryableAttrib>(ilwalidQueryAttrib), &value));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamAttributeQuery_BadParameter2}
 * @testcase{22059225}
 * @verify{19789677}
 * @testpurpose{Test negative scenario of LwSciStreamAttributeQuery(), when
 * 'value' is null pointer.}
 * @testbehavior{
 * Setup:
 *    Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamAttributeQuery() API , should return LwSciError_BadParameter,
 * when called with LwSciStreamQueryableAttrib set to 'LwSciStreamQueryableAttrib_MaxElements'
 * and 'value' is nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamAttributeQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamAttributeQuery_BadParameter2)
{
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamLimiterCreate_Success}
 * @testcase{22059227}
 * @verify{21206739}
 * @testpurpose{Test positive scenario of LwSciStreamLimiterCreate(), when
 * a new Limiter block was set up successfully.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamLimiterCreate() API , should return LwSciError_Success,
 * when called with number of packets set to 1 and reference of LwSciStreamBlock.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamLimiterCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamLimiterCreate_Success)
{
    // initial set up
    uint32_t limitedNumPackets =1U;

    // Test case
    ASSERT_EQ(LwSciError_Success,
            LwSciStreamLimiterCreate(limitedNumPackets, &limiter[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamLimiterCreate_BadParameter}
 * @testcase{22059229}
 * @verify{21206739}
 * @testpurpose{Test negative scenario of LwSciStreamLimiterCreate(), when
 * 'limiter' is null pointer.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamLimiterCreate() API , should return LwSciError_BadParameter,
 * when called with number of packets set to 1 and 'limiter' set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamLimiterCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamLimiterCreate_BadParameter)
{
    // initial set up
    uint32_t limitedNumPackets =1U;

    // Test case
    ASSERT_EQ(LwSciError_BadParameter,
            LwSciStreamLimiterCreate(limitedNumPackets, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamLimiterCreate_StreamInternalError}
 * @testcase{22059231}
 * @verify{21206739}
 * @testpurpose{Test negative scenario of LwSciStreamLimiterCreate(), when
 * the limiter block can't be initialized properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamLimiterCreate() API , should return LwSciError_StreamInternalError,
 * when called with number of packets set to 1 and reference of LwSciStreamBlock.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamLimiterCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamLimiterCreate_StreamInternalError)
{
    // initial set up
    uint32_t limitedNumPackets =1U;
    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError,
            LwSciStreamLimiterCreate(limitedNumPackets, &limiter[0]));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcSrcCreate_Success}
 * @testcase{22059232}
 * @verify{19789623}
 * @testpurpose{Test positive scenario of LwSciStreamIpcSrcCreate(), when
 * a new IpcSrc block was set up successfully.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcSrcCreate() API , should return LwSciError_Success,
 * when called with parameters ipcEndpoint set to "ipcSrc.endpoint", syncModule, bufModule
 * and reference of ipcsrc}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamIpcSrcCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcSrcCreate_Success)
{
    // initial set up
    initIpcChannel();

    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamIpcSrcCreate(
                    ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcSrcCreate_StreamInternalError1}
 * @testcase{22059234}
 * @verify{19789623}
 * @testpurpose{Test negative scenario of LwSciStreamIpcSrcCreate(), when
 * IpcSrc block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcSrcCreate() API , should return LwSciError_StreamInternalError,
 * when called with parameters ipcEndpoint set to "ipcSrc.endpoint", syncModule, bufModule
 * and reference of ipcsrc}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamIpcSrcCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcSrcCreate_StreamInternalError1)
{
    // initial set up
    initIpcChannel();
    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamIpcSrcCreate(
                    ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcSrcCreate_StreamInternalError2}
 * @testcase{22059237}
 * @verify{19789623}
 * @testpurpose{Test negative scenario of LwSciStreamIpcSrcCreate(), when
 * connection establishment through ipcEndpoint fails.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcSrcCreate() API , should return LwSciError_StreamInternalError,
 * when called with parameters ipcEndpoint set to 0, syncModule, bufModule
 * and reference of ipcsrc}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - IpcComm::isInitSuccess()}
 * @verifyFunction{LwSciStreamIpcSrcCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcSrcCreate_StreamInternalError2)
{
    // initial set up
    initIpcChannel();

    // IpcComm::isInitSuccess returns false
    test_comm.isInitSuccess_fail= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamIpcSrcCreate(
                    0U, syncModule, bufModule, &ipcsrc));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcSrcCreate_BadParameter}
 * @testcase{22059239}
 * @verify{19789623}
 * @testpurpose{Test negative scenario of LwSciStreamIpcSrcCreate(), when
 * output parameter ipc is a null pointer.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcSrcCreate() API , should return LwSciError_BadParameter,
 * when called with parameters ipcEndpoint set to "ipcSrc.endpoint", syncModule, bufModule
 * and ipc set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamIpcSrcCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcSrcCreate_BadParameter)
{
    // initial set up
    initIpcChannel();

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamIpcSrcCreate(
                    ipcSrc.endpoint, syncModule, bufModule, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcDstCreate_Success}
 * @testcase{22059241}
 * @verify{19789626}
 * @testpurpose{Test positive scenario of LwSciStreamIpcDstCreate(), when
 * a new IpcDst block was set up successfully.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcDstCreate() API , should return LwSciError_Success,
 * when called with parameters ipcEndpoint set to "ipcDst.endpoint", syncModule, bufModule
 * and reference of ipcdst}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamIpcDstCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcDstCreate_Success)
{
    // initial set up
    initIpcChannel();

    // Test case
    ASSERT_EQ(LwSciError_Success, LwSciStreamIpcDstCreate(
                    ipcDst.endpoint, syncModule, bufModule, &ipcdst));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcDstCreate_StreamInternalError1}
 * @testcase{22059242}
 * @verify{19789626}
 * @testpurpose{Test negative scenario of LwSciStreamIpcDstCreate(), when
 * IpcDst block cannot be initialized properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcDstCreate() API , should return LwSciError_StreamInternalError,
 * when called with parameters ipcEndpoint set to "ipcDst.endpoint", syncModule, bufModule
 * and reference of ipcdst}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamIpcDstCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcDstCreate_StreamInternalError1)
{
    // initial set up
    initIpcChannel();
    LwSciStream::initfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamIpcDstCreate(
                    ipcDst.endpoint, syncModule, bufModule, &ipcdst));
    LwSciStream::initfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcDstCreate_StreamInternalError2}
 * @testcase{22059245}
 * @verify{19789626}
 * @testpurpose{Test negative scenario of LwSciStreamIpcDstCreate(), when
 * connection establishment through ipcEndpoint fails.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcDstCreate() API , should return LwSciError_StreamInternalError,
 * when called with parameters ipcEndpoint set to 0, syncModule, bufModule
 * and reference of ipcdst}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - IpcComm::isInitSuccess()}
 * @verifyFunction{LwSciStreamIpcDstCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcDstCreate_StreamInternalError2)
{
    // initial set up
    initIpcChannel();

    // IpcComm::isInitSuccess returns false
    test_comm.isInitSuccess_fail= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamIpcDstCreate(
                    0U, syncModule, bufModule, &ipcdst));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcDstCreate_BadParameter}
 * @testcase{22059247}
 * @verify{19789626}
 * @testpurpose{Test negative   scenario of LwSciStreamIpcDstCreate(), when
 * output parameter ipc is a null pointer.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcDstCreate() API , should return LwSciError_BadParameter,
 * when called with parameters ipcEndpoint set to "ipcDst.endpoint", syncModule, bufModule
 * and ipc set to nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamIpcDstCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcDstCreate_BadParameter)
{
    // initial set up
    initIpcChannel();

    // Test case
    ASSERT_EQ(LwSciError_BadParameter, LwSciStreamIpcDstCreate(
                    ipcDst.endpoint, syncModule, bufModule, nullptr));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockConnect_BadParameter}
 * @testcase{22059249}
 * @verify{19789602}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of LwSciStreamBlockConnect(), when
 * upstream LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockConnect() API,when called with parameters
 * upstream set to invalid block and downstream set to consumer should
 * return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockConnect()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockConnect_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamBlock IlwalidBlock_handle;

    ASSERT_EQ(LwSciError_BadParameter,LwSciStreamBlockConnect(
            IlwalidBlock_handle, consumer[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockConnect_BadParameter2}
 * @testcase{22059251}
 * @verify{19789602}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of LwSciStreamBlockConnect(), when
 * downstream LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockConnect() API, when called with parameters
 * upstream set to producer and downstream set to invalid block
 * should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockConnect()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockConnect_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamBlock IlwalidBlock_handle;

    ASSERT_EQ(LwSciError_BadParameter,LwSciStreamBlockConnect(
            producer, IlwalidBlock_handle));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_StreamInternalError1}
 * @testcase{22059253}
 * @verify{19789605}
 * @testpurpose{Test nagative scenario of LwSciStreamProducerCreate(), when
 * producer block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *   1.Create static pool block through LwSciStreamStaticPoolCreate() API.
 *
 *   The call of LwSciStreamProducerCreate() API, should return LwSciError_StreamInternalError,
 * when called with parameters pool set to pool block and reference of producer block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::registerBlock()}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_StreamInternalError1)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;

    ASSERT_EQ(LwSciError_Success, LwSciStreamStaticPoolCreate(numPackets, &pool));

    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamProducerCreate(pool, &producer));

    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_BadParameter}
 * @testcase{22059255}
 * @verify{19789605}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamProducerCreate(), when
 * pool block is invalid.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamBlockConnect() API, when called with
 * parameters upstream set to producer and downstream set to invalid block
 * should return of LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    LwSciStreamEvent event;

    uint32_t numPackets = NUM_PACKETS;
    LwSciStreamBlock IlwalidBlock_handle;

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockEventQuery(
                pool, EVENT_QUERY_TIMEOUT, &event));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerCreate_StreamInternalError1}
 * @testcase{22059257}
 * @verify{19789608}
 * @testpurpose{Test negative scenario of LwSciStreamConsumerCreate(), when
 * consumer block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *   Create maibox queue block through LwSciStreamMailboxQueueCreate() API.
 *
 *   The call of LwSciStreamConsumerCreate() API , should return LwSciError_StreamInternalError,
 * when called parameters queue set to queue block and reference of consumer block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::registerBlock()}
 * @verifyFunction{LwSciStreamConsumerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerCreate_StreamInternalError1)
{
    ASSERT_EQ(LwSciError_Success, LwSciStreamMailboxQueueCreate(&queue[0]));

    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamConsumerCreate(queue[0], &consumer[0]));

    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerCreate_BadParameter}
 * @testcase{22059259}
 * @verify{19789608}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamConsumerCreate(), when
 * queue block is invalid.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamConsumerCreate() API, when called with parameters
 * queue set to invalid block and reference of consumer block should
 * return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerCreate_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamConsumerCreate(
                IlwalidBlock_handle, &consumer[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamStaticPoolCreate_StreamInternalError1}
 * @testcase{22059261}
 * @verify{19789611}
 * @testpurpose{Test negative scenario of LwSciStreamStaticPoolCreate(), when
 * pool block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamStaticPoolCreate() API , should return LwSciError_StreamInternalError,
 * when called with numPackets set to NUM_PACKETS and reference of new pool block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::isInitSuccess()}
 * @verifyFunction{LwSciStreamStaticPoolCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamStaticPoolCreate_StreamInternalError1)
{
    // initial set up
    uint32_t numPackets = NUM_PACKETS;
    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamStaticPoolCreate(numPackets, &pool));
    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMailboxQueueCreate_StreamInternalError1}
 * @testcase{22059262}
 * @verify{19789614}
 * @testpurpose{Test negative scenario of LwSciStreamMailboxQueueCreate(), when
 * mailbox queue block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMailboxQueueCreate() API , should return
 * LwSciError_StreamInternalError, when called with reference of new mailbox queue block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::registerBlock()}
 * @verifyFunction{LwSciStreamMailboxQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMailboxQueueCreate_StreamInternalError1)
{
    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamMailboxQueueCreate(&queue[0]));
    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamFifoQueueCreate_StreamInternalError1}
 * @testcase{22059264}
 * @verify{19789617}
 * @testpurpose{Test negative scenario of LwSciStreamFifoQueueCreate(), when
 * FIFO queue block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamFifoQueueCreate() API , should return
 * LwSciError_StreamInternalError, when called with reference of new FIFO queue block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::registerBlock()}
 * @verifyFunction{LwSciStreamFifoQueueCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamFifoQueueCreate_StreamInternalError1)
{
    // initial set up
    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamFifoQueueCreate(&queue[0]));
    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamMulticastCreate_StreamInternalError1}
 * @testcase{22059267}
 * @verify{19789620}
 * @testpurpose{Test negative scenario of LwSciStreamMulticastCreate(), when
 * multicast block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamMulticastCreate() API , should return
 * LwSciError_StreamInternalError, when called with outputCount (number of connected output blocks)
 * equals to 2 and reference of new multicast block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::registerBlock()}
 * @verifyFunction{LwSciStreamMulticastCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamMulticastCreate_StreamInternalError1)
{
    // initial set up
    uint32_t numConsumers = 2U;
    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamMulticastCreate(numConsumers, &multicast));
    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventQuery_BadParameter2}
 * @testcase{22059269}
 * @verify{19789629}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockEventQuery(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockEventQuery() API, when called with parameters block set to
 * invalid block, EVENT_QUERY_TIMEOUT and reference of LwSciStreamEvent
 * should return LwSciError_BadParameter}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventQuery()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventQuery_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockEventQuery(
            IlwalidBlock_handle, EVENT_QUERY_TIMEOUT, &event));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncRequirements_BadParameter2}
 * @testcase{22059271}
 * @verify{19789632}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockSyncRequirements(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockSyncRequirements() API,when called with
 * parameters block set to invalid block, prodSynchronousOnly set to true
 * and LwSciSyncAttrList should return LwSciError_BadParameter}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncRequirements()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncRequirements_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    bool prodSynchronousOnly = true;
    LwSciSyncAttrList prodSyncAttrList = nullptr;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockSyncRequirements(
            IlwalidBlock_handle, prodSynchronousOnly, prodSyncAttrList));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObjCount_BadParameter}
 * @testcase{22059273}
 * @verify{19789635}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockSyncObjCount(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockSyncObjCount() API,when called with parameters
 * block set to invalid block and number of syncobj count of producer set to 2
 * should return LwSciError_BadParameter}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObjCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObjCount_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    uint32_t prodSyncCount = NUM_SYNCOBJS;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    LwSciStreamBlockSyncObjCount(IlwalidBlock_handle, prodSyncCount);

    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockSyncObjCount(
            IlwalidBlock_handle, prodSyncCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockSyncObject_BadParameter2}
 * @testcase{22059275}
 * @verify{19789638}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockSyncObject(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockSyncObject() API panics, when called with
 * parameters block set to invalid block, index set to 0 and LwSciSyncObj
 * should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockSyncObject()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockSyncObject_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketElementCount(
            IlwalidBlock_handle, elementCount));
}


/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketElementCount_BadParameter2}
 * @testcase{22059277}
 * @verify{19789641}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockPacketElementCount(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockPacketElementCount() API, when called with
 * parameters block set to invalid block and elementCount set to NUM_PACKET_ELEMENTS
 * should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketElementCount()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketElementCount_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketElementCount(
            IlwalidBlock_handle, elementCount));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAttr_BadParameter}
 * @testcase{22059279}
 * @verify{19789644}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockPacketAttr(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockPacketAttr() API, when called with parameters
 * block set to invalid block, valid Index of packet element, type of element,
 * LwSciStreamElementMode set to LwSciStreamElementMode_Asynchronous and LwSciBufAttrList
 * set to rawBufAttrList should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAttr()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAttr_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    uint32_t numElements = 1U;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

        // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketAttr(
            IlwalidBlock_handle, 0, 0, LwSciStreamElementMode_Asynchronous, rawBufAttrList));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketCreate_BadParameter}
 * @testcase{22059280}
 * @verify{19789647}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamPoolPacketCreate(), when
 * pool block is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamPoolPacketCreate() API,when called with parameters pool set
 * to invalid block, valid LwSciStreamCookie and reference of LwSciStreamPacket should
 * return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketCreate_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    uint32_t i;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    // Choose pool's cookie and for new packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
        = static_cast<LwSciStreamCookie>(0+ COOKIE_BASE);

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamPoolPacketCreate(
            IlwalidBlock_handle, poolCookie, &packetHandle));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketInsertBuffer_BadParameter}
 * @testcase{22059281}
 * @verify{19789650}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamPoolPacketInsertBuffer(), when
 * pool block is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamPoolPacketInsertBuffer() API, with parameters pool set
 * to invalid block handle, valid LwSciStreamPacket, valid index of element and valid
 * reference of LwSciBufObj should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketInsertBuffer()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketInsertBuffer_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

    // Choose pool's cookie and for new packet
    LwSciStreamPacket packetHandle;

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

        // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamPoolPacketInsertBuffer(
            IlwalidBlock_handle, packetHandle, 0, poolElementBuf[0]));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamPoolPacketDelete_BadParameter2}
 * @testcase{22059283}
 * @verify{19789653}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamPoolPacketDelete(), when
 * pool block is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *   3. Create packets
 *
 *   The call of LwSciStreamPoolPacketDelete() API,when called with parameters pool set to
 * invalid block and valid LwSciStreamPacket should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamPoolPacketDelete()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamPoolPacketDelete_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();
    createPacket();

    LwSciStreamPacket packet;

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamPoolPacketDelete(
            IlwalidBlock_handle, packet));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockPacketAccept_BadParameter}
 * @testcase{22059285}
 * @verify{19789656}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockPacketAccept(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamBlockPacketAccept() API,when called with parameters
 * block set to invalid block, valid LwSciStreamPacket, valid LwSciStreamCookie
 * should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockPacketAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockPacketAccept_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = event.packetHandle;
        LwSciStreamCookie producerCookie
                    = static_cast<LwSciStreamCookie>(0 + COOKIE_BASE);
        LwSciError producerError = LwSciError_Success;

        // Producer accepts a packet provided by the pool
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockPacketAccept(
            IlwalidBlock_handle,producerPacket, producerCookie, producerError));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockElementAccept_BadParameter2}
 * @testcase{22059287}
 * @verify{19789659}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockElementAccept(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up packet attributes.
 *
 *   The call of LwSciStreamBlockElementAccept() API,when called with parameters block set
 * to invalid block, valid LwSciStreamPacket, valid index of the element and LwSciError set to
 * 'LwSciError_Success' should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockElementAccept()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockElementAccept_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    packetAttrSetup();

        // Assign cookie to producer packet handle
        LwSciStreamPacket producerPacket = prodCPMap[event.packetCookie];
        LwSciError producerError = LwSciError_Success;

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockElementAccept(
            IlwalidBlock_handle, producerPacket, event.index, producerError));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketGet_BadParameter}
 * @testcase{22059290}
 * @verify{19789662}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamProducerPacketGet(), when
 * producer block is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *
 *   The call of LwSciStreamProducerPacketGet() API, when called with
 * parameters producer set to invalid block, valid reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketGet()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketGet_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    uint32_t numFrames =1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

        LwSciStreamCookie cookie;

        // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamProducerPacketGet(
            IlwalidBlock_handle, &cookie, fences));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerPacketPresent_BadParameter}
 * @testcase{22059292}
 * @verify{19789665}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamProducerPacketPresent(), when
 * producer block is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *
 *   The call of LwSciStreamProducerPacketPresent() API,when called
 * with parameters producer set to invalid block, valid LwSciStreamPacket and
 * Pointer to an array of LwSciSyncFence should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerPacketPresent()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerPacketPresent_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    uint32_t numFrames=1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

        LwSciStreamCookie cookie;

        LwSciStreamPacket handle = prodCPMap[cookie];

        // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamProducerPacketPresent(
            IlwalidBlock_handle, handle, fences));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketAcquire_BadParameter}
 * @testcase{22059294}
 * @verify{19789668}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamConsumerPacketAcquire(), when
 * consumer block is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *
 *   The call of LwSciStreamConsumerPacketAcquire() API,when called with
 * parameters consumer set to invalid block, valid reference of LwSciStreamCookie
 * and Pointer to an array of LwSciSyncFence, should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketAcquire()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketAcquire_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));
        LwSciStreamCookie cookie;

            // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamConsumerPacketAcquire(
            IlwalidBlock_handle, &cookie, fences));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamConsumerPacketRelease_BadParameter2}
 * @testcase{22059296}
 * @verify{19789671}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamConsumerPacketRelease(), when
 * consumer block is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Sets up synchronization and buffer resources.
 *
 *   The call of LwSciStreamConsumerPacketRelease() API,when called with parameters consumer
 * set to invalid block, valid LwSciStreamPacket and Pointer to an array of LwSciSyncFence
 * should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamConsumerPacketRelease()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamConsumerPacketRelease_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    uint32_t numFrames = 1;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();
    createSync();
    packetAttrSetup();
    createPacket();
    checkPacketStatus();

    uint32_t maxSync = (totalConsSync > prodSyncCount)
                         ? totalConsSync : prodSyncCount;

     LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

        LwSciStreamCookie cookie;

        LwSciStreamPacket handle = prodCPMap[cookie];

            // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamConsumerPacketRelease(
            IlwalidBlock_handle, handle, fences));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockDelete_BadParameter}
 * @testcase{22059298}
 * @verify{19789674}
 * @verify{18700800}
 * @testpurpose{Test panic scenario of LwSciStreamBlockDelete(), when
 * LwSciStreamBlock is invalid.}
 * @testbehavior{
 * Setup:
 *   Create a producer block by calling createBlocks().
 *
 *   The call of LwSciStreamBlockDelete() API,when called with parameter block set to
 * invalid block should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockDelete()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockDelete_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamBlock IlwalidBlock_handle;
    LwSciStreamEvent event;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Test case
    EXPECT_EQ(LwSciError_BadParameter, LwSciStreamBlockDelete(
            IlwalidBlock_handle));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcSrcCreate_StreamInternalError3}
 * @testcase{22059302}
 * @verify{19789623}
 * @testpurpose{Test negative scenario of LwSciStreamIpcSrcCreate(), when
 * IpcSrc block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcSrcCreate() API , should return LwSciError_StreamInternalError,
 * when called with parameters ipcEndpoint set to "ipcSrc.endpoint", syncModule, bufModule
 * and reference of ipcsrc}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::registerBlock()}
 * @verifyFunction{LwSciStreamIpcSrcCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcSrcCreate_StreamInternalError3)
{
    // initial set up
    initIpcChannel();
    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamIpcSrcCreate(
                    ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));
    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamIpcDstCreate_StreamInternalError3}
 * @testcase{22059304}
 * @verify{19789626}
 * @testpurpose{Test negative scenario of LwSciStreamIpcDstCreate(), when
 * IpcDst block cannot be registered properly.}
 * @testbehavior{
 * Setup:
 *
 *   The call of LwSciStreamIpcDstCreate() API , should return LwSciError_StreamInternalError,
 * when called with parameters ipcEndpoint set to "ipcDst.endpoint", syncModule, bufModule
 * and reference of ipcdst}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::registerBlock()}
 * @verifyFunction{LwSciStreamIpcDstCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamIpcDstCreate_StreamInternalError3)
{
    // initial set up
    initIpcChannel();
    LwSciStream::registerblockfail_flag= true;

    // Test case
    ASSERT_EQ(LwSciError_StreamInternalError, LwSciStreamIpcDstCreate(
                    ipcDst.endpoint, syncModule, bufModule, &ipcdst));
    LwSciStream::registerblockfail_flag= false;
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamProducerCreate_BadParameter3}
 * @testcase{22059305}
 * @verify{19789605}
 * @testpurpose{Test negative scenario of LwSciStreamProducerCreate().}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamProducerCreate() API, when called with parameter
 * pool set to invalid block should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamProducerCreate()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamProducerCreate_BadParameter3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    LwSciStreamBlock IlwalidBlock_handle;

    ASSERT_EQ(LwSciError_BadParameter,LwSciStreamProducerCreate(
            IlwalidBlock_handle, producer));
}

 /**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventServiceSetup_Success}
 * @testcase{22059310}
 * @verify{21697059}
 * @testpurpose{Test positive scenario of LwSciStreamBlockEventServiceSetup().}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockEventServiceSetup() API , should return
 * LwSciError_Success, when called with parameter block set to producer block
 * and valid LwSciEventService.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventServiceSetup()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventServiceSetup_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    // EventService object
    LwSciEventLoopService* eventLoopService = nullptr;
    LwSciEventNotifier* eventNotifier;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    ASSERT_EQ(LwSciError_Success,
            LwSciEventLoopServiceCreate(1U, &eventLoopService));

    // Test case
    ASSERT_EQ(LwSciError_Success,LwSciStreamBlockEventServiceSetup(producer,
            &eventLoopService->EventService, &eventNotifier));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventServiceSetup_BadParameter}
 * @testcase{22059313}
 * @verify{21697059}
 * @testpurpose{Test negative scenario of LwSciStreamBlockEventServiceSetup().}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockEventServiceSetup() API , should return
 * LwSciError_BadParameter, when called with parameter block set to invalid block}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventServiceSetup()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventServiceSetup_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    // EventService object
    LwSciEventLoopService* eventLoopService = nullptr;
    LwSciEventNotifier* eventNotifier;
    LwSciStreamBlock IlwalidBlock_handle;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    ASSERT_EQ(LwSciError_Success,
            LwSciEventLoopServiceCreate(1U, &eventLoopService));

    // Test case
    ASSERT_EQ(LwSciError_BadParameter,LwSciStreamBlockEventServiceSetup(IlwalidBlock_handle,
            &eventLoopService->EventService, &eventNotifier));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventServiceSetup_BadParameter1}
 * @testcase{22059316}
 * @verify{21697059}
 * @testpurpose{Test negative scenario of LwSciStreamBlockEventServiceSetup().}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockEventServiceSetup() API , should return
 * LwSciError_BadParameter, when called with parameter block set to producer
 * block and LwSciEventService set to nullptr}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventServiceSetup()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventServiceSetup_BadParameter1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    // EventService object
    LwSciEventLoopService* eventLoopService = nullptr;
    LwSciEventNotifier* eventNotifier;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    ASSERT_EQ(LwSciError_Success,
            LwSciEventLoopServiceCreate(1U, &eventLoopService));

    // Test case
    ASSERT_EQ(LwSciError_BadParameter,LwSciStreamBlockEventServiceSetup(producer,
            nullptr, &eventNotifier));
}

/**
 * @testname{publicAPI_unit_test.LwSciStreamBlockEventServiceSetup_BadParameter2}
 * @testcase{22059318}
 * @verify{21697059}
 * @testpurpose{Test negative scenario of LwSciStreamBlockEventServiceSetup().}
 * @testbehavior{
 * Setup:
 *   Create the producer, pool, queue and consumer blocks.
 *
 *   The call of LwSciStreamBlockEventServiceSetup() API , should return
 * LwSciError_BadParameter, when called with parameter block set to producer
 * block and LwSciEventNotifier set to nullptr}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStreamBlockEventServiceSetup()}
 */
TEST_F(publicAPI_unit_test, LwSciStreamBlockEventServiceSetup_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    // EventService object
    LwSciEventLoopService* eventLoopService = nullptr;
    LwSciEventNotifier* eventNotifier;

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    ASSERT_EQ(LwSciError_Success,
            LwSciEventLoopServiceCreate(1U, &eventLoopService));

    // Test case
    ASSERT_EQ(LwSciError_BadParameter,LwSciStreamBlockEventServiceSetup(producer,
            &eventLoopService->EventService, nullptr));
}

}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
