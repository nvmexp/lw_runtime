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


#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test_common.h"
#include "lwscistream_panic_mock.h"

//==============================================
// Define ipcsrc_unit_test suite
//==============================================
class ipcsrc_unit_test: public LwSciStreamTest {
public:
    ipcsrc_unit_test( ) {
        // initialization code here
    }

    void SetUp( ) {
        // code here will execute just before the test ensues
        EXPECT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
        EXPECT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
    }

    void TearDown( ) {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~ipcsrc_unit_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

//==============================================
// Define ipcsrc_unit_sync_setup_test suite
//==============================================
class ipcsrc_unit_sync_setup_test: public LwSciStreamTest {
public:
    ipcsrc_unit_sync_setup_test( ) {
        // initialization code here
    }

    void SetUp( ) {
        // code here will execute just before the test ensues
        EXPECT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
        EXPECT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));

        // Producer sync obj count
        prodSyncCount = NUM_SYNCOBJS;
        // Consumer sync obj count
        consSyncCount[0] = NUM_SYNCOBJS;

        // Producer synchronousonly flag
        prodSynchronousOnly = false;
        // Consumer synchronousonly flag
        consSynchronousOnly = false;
    }

    void TearDown( ) {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~ipcsrc_unit_sync_setup_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

//==============================================
// Define ipcsrc_unit_buf_setup_test suite
//==============================================
class ipcsrc_unit_buf_setup_test: public LwSciStreamTest {
public:
    ipcsrc_unit_buf_setup_test( ) {
        // initialization code here
    }

    // Packet element count set by Producer
    uint32_t prodElementCount = NUM_PACKET_ELEMENTS;

    // Packet element count set by Consumer
    uint32_t consElementCount = NUM_PACKET_ELEMENTS;

    // Consolidated packet element count set by Pool
    uint32_t consolidatedElementCount = NUM_PACKET_ELEMENTS;

    void SetUp( ) {
        // code here will execute just before the test ensues
        EXPECT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
        EXPECT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
    }

    // Query maximum number of packet elements
    inline void queryMaxNumElements()
    {
        int32_t value;
        EXPECT_EQ(LwSciError_Success, LwSciStreamAttributeQuery(
            LwSciStreamQueryableAttrib_MaxElements, &value));
        EXPECT_TRUE(prodElementCount <= value);
        EXPECT_TRUE(consElementCount <= value);
    };

    // Producer sends its supported packet attributes to pool
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

    // Consumer sends its supported packet attributes to pool
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

    // Pool receives producer's and consumer's supported packet attributes
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

    // Pool sends reconciled packet attributes to producer and consumer
    inline void poolSendReconciledPacketAttr()
    {
        // Pool sets the number of elements in a packet
        EXPECT_EQ(LwSciError_Success,
            LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

        // Pool sets packet requirements
        for (uint32_t i = 0U; i < consolidatedElementCount; i++) {
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

    ~ipcsrc_unit_buf_setup_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

//==============================================
// Define ipcsrc_packet_stream_test suite
//==============================================
class ipcsrc_packet_stream_test: public LwSciStreamTest {
public:
    ipcsrc_packet_stream_test( ) {
        // initialization code here
    }

    void SetUp( ) {
        // code here will execute just before the test ensues
        EXPECT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));
        EXPECT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));

        // Producer sync obj count
        prodSyncCount = NUM_SYNCOBJS;
        // Consumer sync obj count
        consSyncCount[0] = NUM_SYNCOBJS;
        // Producer synchronousonly flag
        prodSynchronousOnly = false;
        // Consumer synchronousonly flag
        consSynchronousOnly = false;
    }

    void TearDown( ) {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~ipcsrc_packet_stream_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};


namespace LwSciStream {

/**
 * @testname{ipcsrc_unit_test.disconnect_Success}
 * @testcase{22059351}
 * @verify{19675881}
 * @testpurpose{Test positive scenario of IpcSrc::disconnect().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::disconnect() API from ipcsrc object,
 * should return LwSciError_Success and posts LwScistreamEventType_Disconnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::disconnect()}
 */
TEST_F(ipcsrc_unit_test, disconnect_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstDisconnect(_)).Times(1);

    ASSERT_EQ(LwSciError_Success, ipcsrcPtr->disconnect());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_test.disconnect_StreamInternalError}
 * @testcase{22059367}
 * @verify{19675881}
 * @testpurpose{Test negative scenario of IpcSrc::disconnect() when
 * IpcComm::signalDisconnect() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalDisconnect() to return
 *     LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::disconnect() API from ipcsrc object,
 * should return LwSciError_Success and trigger error event set to
 *  LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::disconnect()}
 */
TEST_F(ipcsrc_unit_test, disconnect_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // IpcComm::signalDisconnect returns LwSciError_StreamInternalError
    test_comm.signalDisconnect_fail = true;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->disconnect();

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}


/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncAttr_Success}
 * @testcase{22059370}
 * @verify{19675884}
 * @testpurpose{Test positive scenario of IpcSrc::srcSendSyncAttr().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendSyncAttr() API from ipcsrc object,
 * with valid sync attributes (synchronousOnly flag as false and LwSciWrap::SyncAttr wraps a valid
 * LwSciSyncAttrList) and srcIndex as Block::singleConn, should call srcSendSyncAttr() interface
 * of Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcSendSyncAttr()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncAttr_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = prodSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcSendSyncAttr(_, _, _)).Times(1);

    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.processWriteMsg_StreamInternalError2}
 * @testcase{22059371}
 * @verify{19977678}
 * @testpurpose{Test negative scenario of IpcSrc::processWriteMsg() when pending
 * event type is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcSrc::pendingSendEvent() to return invalid event type.
 *
 *   The call of IpcSrc::srcSendSyncCount() with valid parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processWriteMsg()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, processWriteMsg_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = prodSyncCount;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_trackcount.pending_event_fail = true;
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.processReadMsg_IlwalidOperation}
 * @testcase{22059373}
 * @verify{19839639}
 * @testpurpose{Test negative scenario of IpcSrc::processReadMsg() when
 * message type is not valid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to corrupt the event type.
 *
 *  IpcSrc::processReadMsg() call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid parameters, should trigger error event set to
 *  LwSciError_IlwalidOperation in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processReadMsg()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, processReadMsg_IlwalidOperation)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_ipcrecvbuffer.unpackIlwalidEvent = true;
    // To call IpcSrc::recvSyncAttr()
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidOperation, event.error);

}

/**
 * @testname{ipcsrc_unit_sync_setup_test.dstXmitNotifyConnection_StreamBadDstIndex}
 * @testcase{22059376}
 * @verify{19675926}
 * @testpurpose{Test negative scenario of IpcSrc::dstXmitNotifyConnection(), where
 * dstXmitNotifyConnection() is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::dstXmitNotifyConnection() API with invalid dstIndex, should
 * result in LwSciError_StreamBadDstIndex error which is to be queried through
 * LwSciStreamBlockEventQuery().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::dstXmitNotifyConnection()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, dstXmitNotifyConnection_StreamBadDstIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    //connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->dstXmitNotifyConnection(ILWALID_CONN_IDX);

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_Error, event.type);
     EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.processWriteMsg_BadParameter}
 * @testcase{22059379}
 * @verify{19977678}
 * @testpurpose{Test negative scenario of IpcSrc::processWriteMsg() when cloning
 * LwSciSyncAttrList failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in LwSciSyncAttrListClone() to return LwSciError_BadParameter.
 *
 *   The call of IpcSrc::srcSendSyncAttr() with required parameters, should trigger
 *   error event set to LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processWriteMsg()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, processWriteMsg_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = prodSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_lwscisync.LwSciSyncAttrListClone_fail = true;
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);

}

/**
 * @testname{ipcsrc_unit_sync_setup_test.packSyncAttr_ResourceError}
 * @testcase{22059521}
 * @verify{20050536}
 * @testpurpose{Test negative scenario of IpcSrc::packSyncAttr() when exporting
 * LwSciSyncObj waiter requirements failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in LwSciSyncAttrListIpcExportUnreconciled() to return
 *      LwSciError_ResourceError.
 *
 *   The call of IpcSrc::srcSendSyncAttr() API from ipcsrc object,
 * with valid sync attributes (synchronousOnly flag as false and
 * LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and srcIndex as
 * Block::singleConn, triggers error event set to  LwSciError_ResourceError. }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, packSyncAttr_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = prodSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_lwscisync.LwSciSyncAttrListIpcExportUnreconciled_fail = true;
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
}


/**
 * @testname{ipcsrc_unit_sync_setup_test.packSyncAttr_StreamInternalError}
 * @testcase{22059524}
 * @verify{20050536}
 * @testpurpose{Test negative scenario of IpcSrc::packSyncAttr() when packing
 *  LwSciSyncObj waiter requirements failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcSendBuffer::packValAndBlob() to return false.
 *
 *   The call of IpcSrc::srcSendSyncAttr() API from ipcsrc object,
 * with valid synchronousOnly flag as true, LwSciWrap::SyncAttr wraps a null
 * attribute list and srcIndex as Block::singleConn, triggers error event set
 * to  LwSciError_StreamInternalError. }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, packSyncAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{nullptr};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.packValAndBlob_fail = true;
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncAttr_IlwalidState}
 * @testcase{22059528}
 * @verify{19675884}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncAttr() when
 *  LwSciSyncObj waiter requirements has already been scheduled to be sent. }
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Call IpcSrc::srcSendSyncAttr() to send the LwSciSyncObj waiter
 *   requirements.
 *
 *   The call of IpcSrc::srcSendSyncAttr() API from ipcsrc object,
 * with valid synchronousOnly flag as false, LwSciWrap::SyncAttr wraps a valid
 * attribute list and srcIndex as Block::singleConn, triggers error event set
 * to  LwSciError_IlwalidState. }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncAttr_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = prodSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}


/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncAttr_StreamInternalError}
 * @testcase{22059530}
 * @verify{19675884}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncAttr() when
 *  IpcComm::signalWrite() failed. }
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcSendSyncAttr() API from ipcsrc object,
 * with valid synchronousOnly flag as false, LwSciWrap::SyncAttr wraps a valid
 * attribute list and srcIndex as Block::singleConn, triggers error event set
 * to  LwSciError_StreamInternalError. }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = prodSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };


    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_comm.signalWrite_fail = true;
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.packCount_StreamInternalError1}
 * @testcase{22059533}
 * @verify{20050533}
 * @testpurpose{Test negative scenario of IpcSrc::packCount() when
 * packing sync count failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcSrc::srcSendSyncCount() API from ipcsrc object,
 * with valid a sync count of 2 and srcIndex of Block::singleConn,
 * should  trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, packCount_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = prodSyncCount;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.packVal_fail = true;
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncDesc_Success}
 * @testcase{22059536}
 * @verify{19675890}
 * @testpurpose{Test positive scenario of IpcSrc::srcSendSyncDesc().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends its sync object requirement to producer
 *      through LwSciStreamBlockSyncRequirements().
 *   5. Producer receives consumer's sync object requirement by querying
 *      through LwSciStreamBlockEventQuery().
 *   6. Producer sends it sync object count to consumer through LwSciStreamBlockSyncObjCount().
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object,
 * with srcIndex of Block::singleConn, syncIndex value less than producer's sync object count set
 * earlier through LwSciStreamBlockSyncObjCount() and syncObj containing a non-NULL LwSciSyncObj,
 * should call srcSendSyncDesc() interface of the Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcSendSyncDesc()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncDesc_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its sync object requirement to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,
                                         consSyncAttrList));

    // Producer receives consumer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcSendSyncDesc(_, _, _))
        .Times(prodSyncCount);

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        getSyncObj(syncModule, prodSyncObjs[i]);
        syncIndex = i;
        wrapSyncObj = { prodSyncObjs[i] };

        ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.packSyncDesc_ResourceError}
 * @testcase{22059540}
 * @verify{20050539}
 * @testpurpose{Test negative scenario of IpcSrc::packSyncDesc() when exporting
 * LwSciSyncObj failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends sync object requirement to producer
 *      through LwSciStreamBlockSyncRequirements().
 *   5. Producer receives consumer's sync object requirement by querying
 *      through LwSciStreamBlockEventQuery().
 *   6. Producer sends sync object count to consumer through LwSciStreamBlockSyncObjCount().
 *   7. Inject fault in LwSciSyncIpcExportAttrListAndObj() to return
 *      LwSciError_ResourceError.
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object,
 * with srcIndex of Block::singleConn, syncIndex value less than producer's sync object count set
 * earlier through LwSciStreamBlockSyncObjCount() and syncObj containing a non-NULL LwSciSyncObj,
 * should trigger an error event set to LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, packSyncDesc_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its sync object requirement to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,
                                         consSyncAttrList));

    // Producer receives consumer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    wrapSyncObj = { prodSyncObjs[0] };

    test_lwscisync.LwSciSyncIpcExportAttrListAndObj_fail = true;
    ipcsrcPtr->srcSendSyncDesc(srcIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

}


/**
 * @testname{ipcsrc_unit_sync_setup_test.packSyncDesc_StreamInternalError}
 * @testcase{22059543}
 * @verify{20050539}
 * @testpurpose{Test negative scenario of IpcSrc::packSyncDesc() when packing
 * LwSciSyncObj failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends sync object requirement to producer
 *      through LwSciStreamBlockSyncRequirements().
 *   5. Producer receives consumer's sync object requirement by querying
 *      through LwSciStreamBlockEventQuery().
 *   6. Producer sends sync object count to consumer through LwSciStreamBlockSyncObjCount().
 *   7. Inject fault in IpcSendBuffer::packValAndBlob() to return false.
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object,
 * with srcIndex of Block::singleConn, syncIndex value less than producer's sync object count set
 * earlier through LwSciStreamBlockSyncObjCount() and syncObj containing a non-NULL LwSciSyncObj,
 * should trigger an error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, packSyncDesc_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its sync object requirement to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,
                                         consSyncAttrList));

    // Producer receives consumer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    wrapSyncObj = { prodSyncObjs[0] };

    test_ipcsendbuffer.packValAndBlob_fail = true;
    ipcsrcPtr->srcSendSyncDesc(srcIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketElementCount_IlwalidState}
 * @testcase{22059546}
 * @verify{19675893}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketElementCount()
 * when packet element count has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Call IpcSrc::srcSendPacketElementCount() to send the packet element count.
 *
 *   The call of IpcSrc::srcSendPacketElementCount() API from ipcsrc object,
 * with srcIndex of Block::singleConn, count of 2, should trigger the error
 * event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketElementCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketElementCount_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = consolidatedElementCount;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);
    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketElementCount_StreamInternalError}
 * @testcase{22059549}
 * @verify{19675893}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketElementCount()
 * when IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcSendPacketElementCount() API from ipcsrc object,
 * with srcIndex of Block::singleConn, count of 2, should trigger the error
 * event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketElementCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketElementCount_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = consolidatedElementCount;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    test_comm.signalWrite_fail = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketAttr_IlwalidState1}
 * @testcase{22059552}
 * @verify{19675896}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketAttr() when
 * reconciled packet element count is not yet sent by the pool.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex set as zero, valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should trigger the error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketAttr_IlwalidState1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, elemType,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketAttr_StreamInternalError}
 * @testcase{22059555}
 * @verify{19675896}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketAttr() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex set as zero, valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should trigger the error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    test_comm.signalWrite_fail = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, elemType,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketAttr_IlwalidState2}
 * @testcase{22059558}
 * @verify{19675896}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketAttr() when
 * reconciled packet element information for the elemIndex has already been
 * scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Call IpcSrc::srcSendPacketAttr() to send the packet element information for
 *      the element index set as zero.
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex set as zero, valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should trigger the error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketAttr_IlwalidState2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, elemType,
                                    elemSyncMode, wrapElemBufAttr);
    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, elemType,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketAttr_BadParameter}
 * @testcase{22059560}
 * @verify{19675896}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketAttr() when
 * elemIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex(greater than value set using
 * LwSciStreamBlockPacketElementCount), valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should trigger the error event set to LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketAttr_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacketAttr(srcIndex, consolidatedElementCount+1U, elemType,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketAttr_Success1}
 * @testcase{22059562}
 * @verify{19675896}
 * @testpurpose{Test positive scenario of IpcSrc::srcSendPacketAttr().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Pool sends the consolidated packet element count to producer and consumer
 *      through LwSciStreamBlockPacketElementCount().
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex value less than pool's consolidated packet element
 * count set earlier through LwSciStreamBlockPacketElementCount(), valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should call srcSendPacketAttr() interface of the Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcSendPacketAttr()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketAttr_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcSendPacketAttr(_, _, _, _, _))
        .Times(consolidatedElementCount);

    for (uint32_t i = 0U; i < consolidatedElementCount; ++i) {
        wrapElemBufAttr = rawBufAttrList;
        elemIndex = i;
        elemType = i;

        ipcsrcPtr->srcSendPacketAttr(srcIndex, elemIndex, elemType,
                                        elemSyncMode, wrapElemBufAttr);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.packElemAttr_Success}
 * @testcase{22059565}
 * @verify{20050542}
 * @testpurpose{Test positive scenario of IpcSrc::packElemAttr() when packing
 * reconciled packet element attributes is successful.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Pool sends the consolidated packet element count to producer and consumer
 *      through LwSciStreamBlockPacketElementCount().
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex value less than pool's consolidated packet element
 * count set earlier through LwSciStreamBlockPacketElementCount(), valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should call srcSendPacketAttr() interface of the Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcSendPacketAttr()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, packElemAttr_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcSendPacketAttr(_, _, _, _, _))
        .Times(consolidatedElementCount);

    test_lwscibuf.LwSciBufAttrListIpcExportReconciled_blobData_null = true;
    for (uint32_t i = 0U; i < consolidatedElementCount; ++i) {
        wrapElemBufAttr = rawBufAttrList;
        elemIndex = i;
        elemType = i;

        ipcsrcPtr->srcSendPacketAttr(srcIndex, elemIndex, elemType,
                                        elemSyncMode, wrapElemBufAttr);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.packElemAttr_ResourceError}
 * @testcase{22059569}
 * @verify{20050542}
 * @testpurpose{Test negative scenario of IpcSrc::packElemAttr() when exporting
 * reconciled packet element attributes failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Pool sends the consolidated packet element count to producer and consumer
 *      through LwSciStreamBlockPacketElementCount().
 *   8. Inject fault in LwSciBufAttrListIpcExportReconciled() to return
 *      LwSciError_ResourceError.
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex value less than pool's consolidated packet element
 * count set earlier through LwSciStreamBlockPacketElementCount(), valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should trigger error event set to LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, packElemAttr_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    wrapElemBufAttr = rawBufAttrList;

    test_lwscibuf.LwSciBufAttrListIpcExportReconciled_fail = true;
    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, 0U,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.packElemAttr_StreamInternalError}
 * @testcase{22059572}
 * @verify{20050542}
 * @testpurpose{Test negative scenario of IpcSrc::packElemAttr() when packing
 * reconciled packet element attributes failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   7. Pool sends the consolidated packet element count to producer and consumer
 *      through LwSciStreamBlockPacketElementCount().
 *   8. Inject fault in IpcSendBuffer::packValAndBlob() to return false.
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object,
 * with srcIndex of Block::singleConn, elemIndex value less than pool's consolidated packet element
 * count set earlier through LwSciStreamBlockPacketElementCount(), valid elemType, elemSyncMode as
 * LwSciStreamElementMode_Asynchronous and elemBufAttr containing a non-NULL LwSciBufAttrList,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
 TEST_F(ipcsrc_unit_buf_setup_test, packElemAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    // Pool sets the number of elements in a packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(pool, consolidatedElementCount));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            consumer[0], EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    wrapElemBufAttr = rawBufAttrList;

    test_ipcsendbuffer.packVal_fail = true;
    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, 0U,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcCreatePacket_Success}
 * @testcase{22059575}
 * @verify{19675899}
 * @testpurpose{Test positive scenario of IpcSrc::srcCreatePacket().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *
 *   The call of IpcSrc::srcCreatePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and handle(as reference to the newly created packet),
 * should call srcCreatePacket() of the Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcCreatePacket()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcCreatePacket()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcCreatePacket_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;

    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    handle = ~poolCookie;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcCreatePacket(_, _)).Times(1);

    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcCreatePacket_StreamInternalError1}
 * @testcase{22059578}
 * @verify{19675899}
 * @testpurpose{Test negative scenario of IpcSrc::srcCreatePacket() when
 * packet instance already exists in the PacketMap.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Call IpcSrc::srcCreatePacket() to create a packet instance.
 *
 *   The call of IpcSrc::srcCreatePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and same handle(refers the newly
 * created packet), should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcCreatePacket()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcCreatePacket_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    handle = ~poolCookie;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcCreatePacket_InsufficientMemory}
 * @testcase{22059582}
 * @verify{19675899}
 * @testpurpose{Test negative scenario of IpcSrc::srcCreatePacket() when
 * packet instance creation failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Inject fault in Block::pktCreate() to return LwSciError_InsufficientMemory.
 *
 *   The call of IpcSrc::srcCreatePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and handle(refers the newly
 * created packet), should trigger error event set to LwSciError_InsufficientMemory.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcCreatePacket()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcCreatePacket_InsufficientMemory)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    handle = ~poolCookie;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_block.pktCreate_fail = true;
    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_InsufficientMemory, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcCreatePacket_StreamInternalError2}
 * @testcase{22059585}
 * @verify{19675899}
 * @testpurpose{Test negative scenario of IpcSrc::srcCreatePacket() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcCreatePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and handle(refers the newly
 * created packet), should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcCreatePacket()}
 */
 TEST_F(ipcsrc_unit_buf_setup_test, srcCreatePacket_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    handle = ~poolCookie;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_comm.signalWrite_fail = true;
    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.packHandle_StreamInternalError}
 * @testcase{22059588}
 * @verify{20050545}
 * @testpurpose{Test negative scenario of IpcSrc::srcCreatePacket() when
 * packing handle failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcSrc::srcCreatePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and handle, should trigger error event
 * set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcCreatePacket()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, packHandle_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    handle = ~poolCookie;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.packVal_fail = true;
    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_Success}
 * @testcase{22059590}
 * @verify{19675905}
 * @testpurpose{Test positive scenario of IpcSrc::srcInsertBuffer().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the newly created packet),
 * elemIndex value less than pool's consolidated packet element count set earlier through
 * LwSciStreamBlockPacketElementCount() and elemBufObj containing a non-NULL LwSciBufObj,
 * should call srcInsertBuffer() interface of the Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcInsertBuffer()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;

    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcInsertBuffer(_, _, _, _))
        .Times(consolidatedElementCount);

    for (uint32_t i = 0U; i < consolidatedElementCount; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        ipcsrcPtr->srcInsertBuffer(srcIndex, handle,
                                    elemIndex, wrapElemBufObj);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.packBuffer_ResourceError}
 * @testcase{22059592}
 * @verify{20050548}
 * @testpurpose{Test negative scenario of IpcSrc::packBuffer() when exporting
 *  LwSciBufObj failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Inject fault in LwSciBufObjIpcExport() to return LwSciError_ResourceError.
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the newly created packet),
 * elemIndex value less than pool's consolidated packet element count set earlier through
 * LwSciStreamBlockPacketElementCount() and elemBufObj containing a non-NULL LwSciBufObj,
 * should trigger error event set to LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, packBuffer_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    makeRawBuffer(rawBufAttrList, poolElementBuf[0]);
    wrapElemBufObj = poolElementBuf[0];

    test_lwscibuf.LwSciBufObjIpcExport_fail = true;
    ipcsrcPtr->srcInsertBuffer(srcIndex, handle,
                                0U, wrapElemBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.packBuffer_StreamInternalError1}
 * @testcase{22059595}
 * @verify{20050548}
 * @testpurpose{Test negative scenario of IpcSrc::packBuffer() when packing
 *  LwSciBufObj failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the newly created packet),
 * elemIndex value less than pool's consolidated packet element count set earlier through
 * LwSciStreamBlockPacketElementCount() and elemBufObj containing a non-NULL LwSciBufObj,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, packBuffer_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            consumer[0], EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    makeRawBuffer(rawBufAttrList, poolElementBuf[0]);
    wrapElemBufObj = poolElementBuf[0];

    test_ipcsendbuffer.counter = 0U;
    test_ipcsendbuffer.packVal_fail = true;
    ipcsrcPtr->srcInsertBuffer(srcIndex, handle,
                                0U, wrapElemBufObj);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_BadParameter1}
 * @testcase{22059598}
 * @verify{19675905}
 * @testpurpose{Test negative scenario of IpcSrc::srcInsertBuffer() when
 *  LwSciBufObj wrapped by elemBufObj is NULL.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the newly created packet),
 * elemIndex value less than pool's consolidated packet element count set earlier through
 * LwSciStreamBlockPacketElementCount() and elemBufObj containing a NULL LwSciBufObj,
 * should trigger error event set to LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_BadParameter1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj{nullptr};
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcInsertBuffer(srcIndex, handle,
                                0U, wrapElemBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_BadParameter2}
 * @testcase{22059601}
 * @verify{19675905}
 * @testpurpose{Test negative scenario of IpcSrc::srcInsertBuffer() when
 *  element index is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the newly created packet),
 * invalid elemIndex value and elemBufObj containing a non-NULL LwSciBufObj,
 * should trigger error event set to LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    ipcsrcPtr->srcInsertBuffer(srcIndex, handle,
                                MAX_ELEMENT_PER_PACKET + 1, wrapBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_IlwalidState}
 * @testcase{22059604}
 * @verify{19675905}
 * @testpurpose{Test negative scenario of IpcSrc::srcInsertBuffer() when
 * LwSciBufObj for the elemIndex has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Call IpcSrc::srcInsertBuffer() to send the packet LwSciBufObj.
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the newly created packet),
 * valid elemIndex value and elemBufObj containing a non-NULL LwSciBufObj,
 * should trigger error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};
    ipcsrcPtr->srcInsertBuffer(srcIndex, handle, 0U, wrapBufObj);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcInsertBuffer(srcIndex, handle, 0U, wrapBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_StreamInternalError2}
 * @testcase{22059607}
 * @verify{19675905}
 * @testpurpose{Test negative scenario of IpcSrc::srcInsertBuffer() when
 * packet handle is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, invalid packet handle,
 * valid elemIndex value and elemBufObj containing a non-NULL LwSciBufObj,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    ipcsrcPtr->srcInsertBuffer(srcIndex, ILWALID_PACKET_HANDLE, 0U, wrapBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_StreamInternalError3}
 * @testcase{22059610}
 * @verify{19675905}
 * @testpurpose{Test negative scenario of IpcSrc::srcInsertBuffer() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object,
 * with srcIndex of Block::singleConn, packet handle,
 * valid elemIndex value and elemBufObj containing a non-NULL LwSciBufObj,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_StreamInternalError3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};
    test_comm.signalWrite_fail = true;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcInsertBuffer(srcIndex, handle, 0U, wrapBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_test.srcDeletePacket_Success}
 * @testcase{22059613}
 * @verify{19675908}
 * @testpurpose{Test positive scenario of IpcSrc::srcDeletePacket().}
 * @testbehavior{
 * Setup:
 *   1. Initialise Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *
 *   The call of IpcSrc::srcDeletePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and handle(as reference to the newly created packet),
 * should call srcDeletePacket() of the Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcDeletePacket()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcDeletePacket()}
 */
TEST_F(ipcsrc_unit_test, srcDeletePacket_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;

    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcDeletePacket(_, _)).Times(1);

    ipcsrcPtr->srcDeletePacket(srcIndex, handle);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_test.srcDeletePacket_StreamInternalError1}
 * @testcase{22059616}
 * @verify{19675908}
 * @testpurpose{Test negative scenario of IpcSrc::srcDeletePacket() when packet
 * handle is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcDeletePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and invalid handle, should trigger error
 * event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcDeletePacket()}
 */
TEST_F(ipcsrc_unit_test, srcDeletePacket_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamEvent event;
    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcDeletePacket(srcIndex, ILWALID_PACKET_HANDLE);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_test.srcDeletePacket_StreamInternalError2}
 * @testcase{22059619}
 * @verify{19675908}
 * @testpurpose{Test negative scenario of IpcSrc::srcDeletePacket() when packet
 * location is not Location::Upstream.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   5. Inject a fault by changing the packet location to other than
 *    Location::Upstream.
 *
 *   The call of IpcSrc::srcDeletePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and valid handle, should trigger error
 * event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcDeletePacket()}
 */
TEST_F(ipcsrc_unit_test, srcDeletePacket_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    PacketPtr const pkt { ipcsrcPtr->pktFindByHandle_access(handle) };
    if (true == pkt->locationCheck(Packet::Location::Upstream))
    {
        pkt->locationUpdate(Packet::Location::Upstream, Packet::Location::Downstream);
    }

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcDeletePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_test.srcDeletePacket_StreamInternalError3}
 * @testcase{22059622}
 * @verify{19675908}
 * @testpurpose{Test negative scenario of IpcSrc::srcDeletePacket() when packet
 * already scheduled for deletion.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   5. call IpcSrc::srcDeletePacket() to schedule a packet for deletion.
 *
 *   The call of IpcSrc::srcDeletePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and same valid handle, should trigger error
 * event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcDeletePacket()}
 */
TEST_F(ipcsrc_unit_test, srcDeletePacket_StreamInternalError3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    PacketPtr const pkt { ipcsrcPtr->pktFindByHandle_access(handle) };

    ipcsrcPtr->srcDeletePacket(srcIndex, handle);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcDeletePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcsrc_unit_test.srcDeletePacket_StreamInternalError4}
 * @testcase{22059625}
 * @verify{19675908}
 * @testpurpose{Test negative scenario of IpcSrc::srcDeletePacket() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   5. Inject a fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcDeletePacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn and valid handle, should trigger error
 * event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcDeletePacket()}
 */
TEST_F(ipcsrc_unit_test, srcDeletePacket_StreamInternalError4)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    PacketPtr const pkt { ipcsrcPtr->pktFindByHandle_access(handle) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_comm.signalWrite_fail = true;
    ipcsrcPtr->srcDeletePacket(srcIndex, handle);
    test_comm.signalWrite_fail = false;

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_packet_stream_test.packPayload_StreamInternalError1}
 * @testcase{22059630}
 * @verify{20050551}
 * @testpurpose{Test negative scenario of IpcSrc::packPayload() when packing
 * handle for the payload failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the producer presented packet) and
 * postfences containing a non-NULL FenceArray, should trigger error event set
 * to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, packPayload_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;
    std::shared_ptr<LwSciStream::Mailbox> mailboxPtr;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.packVal_fail = true;
    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.packPayload_StreamInternalError2}
 * @testcase{22059631}
 * @verify{20050551}
 * @testpurpose{Test negative scenario of IpcSrc::packPayload() when packing
 * fences for the payload failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the producer presented packet) and
 * postfences containing a non-NULL FenceArray, should trigger error event set
 * to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, packPayload_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;
    std::shared_ptr<LwSciStream::Mailbox> mailboxPtr;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.srcSendPacket_packVal_fail = true;
    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.packPayload_ResourceError}
 * @testcase{22059634}
 * @verify{20050551}
 * @testpurpose{Test negative scenario of IpcSrc::packPayload() when exporting
 * fence failed. }
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Inject fault in LwSciSyncIpcExportFence() to return LwSciError_ResourceError.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the producer presented packet) and
 * postfences containing a non-NULL FenceArray, should trigger error event set
 * to LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
 TEST_F(ipcsrc_packet_stream_test, packPayload_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;
    std::shared_ptr<LwSciStream::Mailbox> mailboxPtr;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_lwscisync.LwSciSyncIpcExportFence_fail = true;
    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

    free(fences);
}


/**
 * @testname{ipcsrc_packet_stream_test.srcSendPacket_StreamBadSrcIndex}
 * @testcase{22059637}
 * @verify{19675914}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacket() when srcIndex
 * is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with invalid srcIndex, handle(as reference to the producer presented packet) and
 * postfences containing a non-NULL FenceArray, should trigger error event set
 * to LwSciError_StreamBadSrcIndex.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, srcSendPacket_StreamBadSrcIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 2U;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.srcSendPacket_StreamNotConnected}
 * @testcase{22059641}
 * @verify{19675914}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacket() when stream
 * is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with srcIndex, handle, and postfences containing a non-NULL FenceArray,
 * should trigger error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, srcSendPacket_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.srcSendPacket_StreamInternalError1}
 * @testcase{22059644}
 * @verify{19675914}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacket() when packet
 * handle is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with valid srcIndex, invalid packet handle and postfences containing a non-NULL
 * FenceArray, should trigger error event set
 * to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, srcSendPacket_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacket(srcIndex, ILWALID_PACKET_HANDLE, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.srcSendPacket_StreamInternalError2}
 * @testcase{22059647}
 * @verify{19675914}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacket() when packet
 * location is not Location::Upstream.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Inject fault by changing the packet location to other than Location::Upstream.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with valid srcIndex, packet handle and postfences containing a non-NULL
 * FenceArray, should trigger error event set
 * to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
 TEST_F(ipcsrc_packet_stream_test, srcSendPacket_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;
    LwSciStreamPacket handle;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    PacketPtr const pkt { ipcsrcPtr->pktFindByHandle_access(handle) };
    if (true == pkt->locationCheck(Packet::Location::Upstream))
    {
        pkt->locationUpdate(Packet::Location::Upstream, Packet::Location::Downstream);
    }

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.srcSendPacket_IlwalidState}
 * @testcase{22059650}
 * @verify{19675914}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacket() when payload
 * for the packet is already set.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Call IpcSrc::srcSendPacket() to send the packet downstream.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with valid srcIndex, same packet handle and postfences containing a non-NULL
 * FenceArray, should trigger error event set
 * to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, srcSendPacket_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;
    LwSciStreamPacket handle;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    PacketPtr const pkt { ipcsrcPtr->pktFindByHandle_access(handle) };
    if (true == pkt->locationCheck(Packet::Location::Queued))
    {
        pkt->locationUpdate(Packet::Location::Queued, Packet::Location::Upstream);
    }

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.srcSendPacket_StreamInternalError3}
 * @testcase{22059653}
 * @verify{19675914}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacket() when
 *  IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with valid srcIndex, same packet handle and postfences containing a non-NULL
 * FenceArray, should trigger error event set
 * to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, srcSendPacket_StreamInternalError3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;
    LwSciStreamPacket handle;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_comm.signalWrite_fail = true;

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_unit_test.srcDisconnect_StreamInternalError}
 * @testcase{22059657}
 * @verify{19675917}
 * @testpurpose{Test negative scenario of IpcSrc::srcDisconnect() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject a fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcDisconnect() API from ipcsrc object,
 * with srcIndex of Block::singleConn, should trigger error event set to
 * LwSciError_StreamInternalError. }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcDisconnect()}
 */
TEST_F(ipcsrc_unit_test, srcDisconnect_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_comm.signalWrite_fail = true;
    ipcsrcPtr->srcDisconnect(srcIndex);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncAttr_Success1}
 * @testcase{22059660}
 * @verify{20050557}
 * @testpurpose{Test positive scenario of IpcSrc::recvSyncAttr().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *  IpcSrc::recvSyncAttr() API call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag as false and
 *  LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and dstIndex as
 *  Block::singleConn, should call dstSendSyncAttr() interface of Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendSyncAttr()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncAttr_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = consSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendSyncAttr(_, _, _)).Times(1);

    // To call IpcSrc::recvSyncAttr()
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncAttr_Success2}
 * @testcase{22059663}
 * @verify{20050557}
 * @testpurpose{Test positive scenario of IpcSrc::recvSyncAttr() when endpoint
 * does not support LwSciSyncObj(s).}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *  IpcSrc::recvSyncAttr() API call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag as true and
 *  LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and dstIndex as
 *  Block::singleConn, should call dstSendSyncAttr() interface of Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendSyncAttr()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncAttr_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendSyncAttr(_, _, _)).Times(1);

    // To call IpcSrc::recvSyncAttr()
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcsrc_packet_stream_test.IpcSrc_Ipccomm_StreamInternalError}
 * @testcase{22059666}
 * @verify{19675875}
 * @testpurpose{Test negative scenario of LwSciStream::IpcSrc::IpcSrc
 * when IpcComm::isInitSuccess() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Inject fault in IpcComm::isInitSuccess() to return false.
 *
 *   The call of LwSciStream::IpcSrc::IpcSrc constructor through LwSciStreamIpcSrcCreate()
 * should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::IpcSrc::IpcSrc}
 */
TEST_F(ipcsrc_packet_stream_test, IpcSrc_Ipccomm_StreamInternalError)
{
    /*Initial setup*/

    // Initialise Ipc channel
    initIpcChannel();

    //IpcComm::isInitSuccess returns false
    test_comm.isInitSuccess_fail = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_StreamInternalError,
    LwSciStreamIpcSrcCreate(ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.processReadMsg_StreamInternalError3}
 * @testcase{22059669}
 * @verify{19839639}
 * @testpurpose{Test negative scenario of IpcSrc::processReadMsg() when
 * IpcComm::readFrame() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Stub the return values for IpcComm::waitForEvent () and
 *    IpcComm::waitForConnection().
 *   3. Inject fault in IpcComm::readFrame() to return LwSciError_StreamInternalError.
 *
 *   The call of LwSciStreamIpcSrcCreate() should trigger error event set
 *  to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processReadMsg()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, processReadMsg_StreamInternalError3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    test_comm.readFrame_fail = true;
    test_comm.waitForReadEvent_flag = true;
    test_comm.waitForConnection_pass = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_Success,
    LwSciStreamIpcSrcCreate(ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    test_comm.waitForConnection_pass = false;
    test_comm.waitForReadEvent_flag = false;
    test_comm.readFrame_fail = false;

}


/**
 * @testname{ipcsrc_unit_sync_setup_test.processReadMsg_StreamInternalError1}
 * @testcase{22059672}
 * @verify{19839639}
 * @testpurpose{Test negative scenario of IpcSrc::processReadMsg() when
 * IpcRecvBuffer::unpackBegin() failed during unpacking waiter requirements.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackBegin() to return false.
 *
 *  IpcSrc::processReadMsg() API call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag as true and
 *  LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and dstIndex as
 *  Block::singleConn, should trigger error event set to
 *  LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processReadMsg()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, processReadMsg_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_ipcrecvbuffer.unpackBegin_fail = true;
    // To call IpcSrc::recvSyncAttr()
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_sync_setup_test.processReadMsg_StreamInternalError2}
 * @testcase{22059674}
 * @verify{19839639}
 * @testpurpose{Test negative scenario of IpcSrc::processReadMsg() when
 * IpcRecvBuffer::unpackVal() failed during unpacking waiter requirements.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *  IpcSrc::processReadMsg() API call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag as true and
 *  LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and dstIndex as
 *  Block::singleConn, should trigger error event set to
 *  LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processReadMsg()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, processReadMsg_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_ipcrecvbuffer.processMsg_unpack_fail = true;
    // To call IpcSrc::recvSyncAttr()
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}


/**
 * @testname{ipcsrc_packet_stream_test.processReadMsg_disconnect}
 * @testcase{22059676}
 * @verify{19839639}
 * @testpurpose{Test positive scenario of IpcSrc::processReadMsg() when
 * disconnect requested.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *  IpcSrc::processReadMsg() API call triggered, when IpcDst::dstDisconnect()
 *  API is called with valid dstIndex, should call Block::disconnectSrc() interface.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectSrc().}
 * @verifyFunction{IpcSrc::processReadMsg()}
 */
 TEST_F(ipcsrc_unit_sync_setup_test, processReadMsg_disconnect)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    uint32_t dstIndex = Block::singleConn_access;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*ipcdstPtr, disconnectSrc_imp(_))
           .WillRepeatedly(Return());

    ipcdstPtr->dstDisconnect(dstIndex);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&ipcdstPtr));

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketElementCount_Success}
 * @testcase{22059682}
 * @verify{19675893}
 * @testpurpose{Test positive scenario of IpcSrc::srcSendPacketElementCount().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   5. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   6. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *
 *   The call of IpcSrc::srcSendPacketElementCount() API from ipcsrc object,
 * with srcIndex of Block::singleConn, count of 2,
 * should call srcSendPacketElementCount() interface of Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcSendPacketElementCount()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketElementCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketElementCount_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = consolidatedElementCount;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // Consumer sends its supported packet attributes to pool
    consSendPacketAttr();

    // Pool receives producer's and consumer's supported packet attributes
    poolRecvPacketAttr();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcSendPacketElementCount(_, _)).Times(1);

    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}


/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncCount_StreamInternalError}
 * @testcase{22059684}
 * @verify{20050560}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncCount() when unpacking
 * sync count failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *  IpcSrc::recvSyncCount() API call triggered, when IpcDst::dstSendSyncCount()
 *  API is called with valid a sync count of 2 and dstIndex of Block::singleConn,
 *  should trigger error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncCount_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = consSyncCount[0];
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcrecvbuffer.unpackVal_fail = true;
    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncCount_IlwalidState}
 * @testcase{22059688}
 * @verify{20050560}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncCount() when count
 * value was already received.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Call IpcDst::dstSendSyncCount() with valid sync count of 2U and
 *      dstIndex of Block::singleConn.
 *
 *  IpcSrc::recvSyncCount() API call triggered, when IpcDst::dstSendSyncCount()
 *  API is called with valid a sync count of 2 and dstIndex of Block::singleConn,
 *  should trigger error event set to LwSciError_IlwalidState in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncCount_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = consSyncCount[0];
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcdstPtr->dstSendSyncCount(dstIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncCount, event.type);

    test_trackcount.set_fail_IlwalidState = true;

    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncCount_BadParameter}
 * @testcase{22059691}
 * @verify{20050560}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncCount() when count
 * value is invalid(greater than MAX_SYNC_OBJECTS).}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *  IpcSrc::recvSyncCount() API call triggered, when IpcDst::dstSendSyncCount()
 *  API is called with valid a sync count of 5U and dstIndex of Block::singleConn,
 *  should trigger error event set to LwSciError_BadParameter in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncCount_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = 5U;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_trackcount.set_fail_BadParameter = true;
    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncDesc_Success}
 * @testcase{22059694}
 * @verify{20050563}
 * @testpurpose{Test positive scenario of IpcSrc::recvSyncDesc().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its sync object requirement to consumer
 *      through LwSciStreamBlockSyncRequirements().
 *   5. Consumer receives producer's sync object requirement by querying
 *      through LwSciStreamBlockEventQuery().
 *   6. Consumer sends it sync object count to producer through LwSciStreamBlockSyncObjCount().
 *
 *  IpcSrc::recvSyncDesc() API call triggered, when IpcDst::dstSendSyncDesc()
 *  API is called with dstIndex of Block::singleConn, syncIndex value less than
 *  consumer's sync object count set earlier through LwSciStreamBlockSyncObjCount()
 *  and syncObj containing a non-NULL LwSciSyncObj, should call dstSendSyncDesc()
 *  interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendSyncDesc()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncDesc_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Producer sends its sync object requirement to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // Consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(prodSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], consSyncCount[0]));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendSyncDesc(_, _, _))
        .Times(consSyncCount[0]);

    for (uint32_t i = 0U; i < consSyncCount[0]; ++i) {
        getSyncObj(syncModule, consSyncObjs[0][i]);
        syncIndex = i;
        wrapSyncObj = { consSyncObjs[0][i] };

        // To call IpcSrc::recvSyncDesc()
        ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncDesc_StreamInternalError}
 * @testcase{22059696}
 * @verify{20050563}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncDesc() when unpacking
 * failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its sync object requirement to consumer
 *      through LwSciStreamBlockSyncRequirements().
 *   5. Consumer receives producer's sync object requirement by querying
 *      through LwSciStreamBlockEventQuery().
 *   6. Consumer sends it sync object count to producer through LwSciStreamBlockSyncObjCount().
 *   7. Inject fault in IpcRecvBuffer::unpackValAndBlob() to return false.
 *
 *  IpcSrc::recvSyncDesc() API call triggered, when IpcDst::dstSendSyncDesc()
 *  API is called with dstIndex of Block::singleConn, syncIndex value less than
 *  consumer's sync object count set earlier through LwSciStreamBlockSyncObjCount()
 *  and syncObj containing a non-NULL LwSciSyncObj, should trigger error event set
 *  to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncDesc_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Producer sends its sync object requirement to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // Consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(prodSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], consSyncCount[0]));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    for (uint32_t i = 0U; i < 1U; ++i) {
        getSyncObj(syncModule, consSyncObjs[0][i]);
        syncIndex = i;
        wrapSyncObj = { consSyncObjs[0][i] };

        test_ipcrecvbuffer.unpackValAndBlob_fail = true;
        // To call IpcSrc::recvSyncDesc()
        ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcsrc, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Error, event.type);
        EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    }
}


/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncDesc_ResourceError}
 * @testcase{22059699}
 * @verify{20050563}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncDesc() when
 * LwSciSyncIpcImportAttrListAndObj() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its sync object requirement to consumer
 *      through LwSciStreamBlockSyncRequirements().
 *   5. Consumer receives producer's sync object requirement by querying
 *      through LwSciStreamBlockEventQuery().
 *   6. Consumer sends it sync object count to producer through LwSciStreamBlockSyncObjCount().
 *   7. Inject fault in LwSciSyncIpcImportAttrListAndObj() to return
 *    LwSciError_ResourceError.
 *
 *  IpcSrc::recvSyncDesc() API call triggered, when IpcDst::dstSendSyncDesc()
 *  API is called with dstIndex of Block::singleConn, syncIndex value less than
 *  consumer's sync object count set earlier through LwSciStreamBlockSyncObjCount()
 *  and syncObj containing a non-NULL LwSciSyncObj, should trigger error event set
 *  to LwSciError_ResourceError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncDesc_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Producer sends its sync object requirement to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // Consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(prodSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], consSyncCount[0]));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    getSyncObj(syncModule, consSyncObjs[0][0]);
    wrapSyncObj = { consSyncObjs[0][0] };

    test_lwscisync.LwSciSyncIpcImportAttrListAndObj_fail = true;
    ipcdstPtr->dstSendSyncDesc(dstIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncDesc_IlwalidState}
 * @testcase{22059704}
 * @verify{20050563}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncDesc() when
 * LwSciSyncObj for the SyncIndex already forwarded upstream.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Producer sends its sync object requirement to consumer
 *      through LwSciStreamBlockSyncRequirements().
 *   5. Consumer receives producer's sync object requirement by querying
 *      through LwSciStreamBlockEventQuery().
 *   6. Consumer sends it sync object count to producer through LwSciStreamBlockSyncObjCount().
 *   7. Call IpcDst::dstSendSyncDesc() with dstIndex of Block::singleConn,
 *      syncIndex value less than consumer's sync object count set earlier
 *      through LwSciStreamBlockSyncObjCount() and syncObj containing a
 *      non-NULL LwSciSyncObj to send LwSciSyncObj for the given syncIndex.
 *
 *  IpcSrc::recvSyncDesc() API call triggered, when IpcDst::dstSendSyncDesc()
 *  API is called with dstIndex of Block::singleConn, same syncIndex as in step-7
 *  and syncObj containing a non-NULL LwSciSyncObj, should trigger error event set
 *  to LwSciError_IlwalidState in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncDesc_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Producer sends its sync object requirement to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // Consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(prodSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], consSyncCount[0]));
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncCount, event.type);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    getSyncObj(syncModule, consSyncObjs[0][0]);
    wrapSyncObj = { consSyncObjs[0][0] };


    ipcdstPtr->dstSendSyncDesc(dstIndex, 0U, wrapSyncObj);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncDesc, event.type);

    test_trackArray.performAction_fail = true;
    ipcdstPtr->dstSendSyncDesc(dstIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemCount_Success}
 * @testcase{22059707}
 * @verify{20050566}
 * @testpurpose{Test positive scenario of IpcSrc::recvElemCount().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *  IpcSrc::recvElemCount() API call triggered, when IpcDst::dstSendPacketElementCount()
 *  API is called with dstIndex of Block::singleConn, count of 2, should call
 *  dstSendPacketElementCount() interface of Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendPacketElementCount()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvElemCount_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = consElementCount;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(_, _)).Times(1);

    // To call IpcSrc::recvElemCount()
    ipcsdstPtr->dstSendPacketElementCount(dstIndex, count);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemCount_StreamInternalError}
 * @testcase{22059710}
 * @verify{20050566}
 * @testpurpose{Test negative scenario of IpcSrc::recvElemCount() when unpacking
 *  count failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *  IpcSrc::recvElemCount() API call triggered, when IpcDst::dstSendPacketElementCount()
 *  API is called with dstIndex of Block::singleConn, count of 2, should trigger
 *  error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvElemCount_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = consElementCount;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcrecvbuffer.unpackVal_fail = true;
    ipcsdstPtr->dstSendPacketElementCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemCount_IlwalidState}
 * @testcase{22059713}
 * @verify{20050566}
 * @testpurpose{Test negative scenario of IpcSrc::recvElemCount() when count
 *  was already received.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Call IpcDst::dstSendPacketElementCount() with dstIndex of Block::singleConn
 *      and count of 2U to send the packet element count.
 *
 *  IpcSrc::recvElemCount() API call triggered, when IpcDst::dstSendPacketElementCount()
 *  API is called with dstIndex of Block::singleConn, count of 2, should trigger
 *  error event set to LwSciError_IlwalidState in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemCount()}
 */
 TEST_F(ipcsrc_unit_buf_setup_test, recvElemCount_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = consElementCount;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsdstPtr->dstSendPacketElementCount(dstIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketElementCountConsumer, event.type);

    test_trackcount.set_fail_IlwalidState = true;

    ipcsdstPtr->dstSendPacketElementCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemCount_BadParameter}
 * @testcase{22059714}
 * @verify{20050566}
 * @testpurpose{Test negative scenario of IpcSrc::recvElemCount() when count
 *  value is invalid(greater than MAX_PACKET_ELEMENTS).}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *  IpcSrc::recvElemCount() API call triggered, when IpcDst::dstSendPacketElementCount()
 *  API is called with dstIndex of Block::singleConn, count greater than
 *  MAX_PACKET_ELEMENTS, should trigger error event set to
 *  LwSciError_BadParameter in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemCount()}
 */
 TEST_F(ipcsrc_unit_buf_setup_test, recvElemCount_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = MAX_PACKET_ELEMENTS+1U;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_trackcount.set_fail_BadParameter = true;
    ipcsdstPtr->dstSendPacketElementCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemAttr_Success}
 * @testcase{22059719}
 * @verify{20050569}
 * @testpurpose{Test positive scenario of IpcSrc::recvElemAttr().}
 * @testbehavior{
 * Setup:
 *   1. Initialise Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends its packet element count to pool through
 *      LwSciStreamBlockPacketElementCount().
 *
 *  IpcSrc::recvElemAttr() API call triggered, when IpcDst::dstSendPacketAttr()
 *  API is called with dstIndex of Block::singleConn, elemIndex value less than
 *  consumer's packet element count set earlier through LwSciStreamBlockPacketElementCount(),
 *  valid elemType, elemSyncMode as LwSciStreamElementMode_Asynchronous and
 *  elemBufAttr containing a non-NULL LwSciBufAttrList, should call dstSendPacketAttr()
 *  interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendPacketAttr()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvElemAttr_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendPacketAttr(_, _, _, _, _))
        .Times(consElementCount);

    for (uint32_t i = 0U; i < consElementCount; ++i) {
        wrapElemBufAttr = rawBufAttrList;
        elemIndex = i;
        elemType = i;

        // To call IpcSrc::recvElemAttr()
        ipcsdstPtr->dstSendPacketAttr(dstIndex, elemIndex, elemType,
                                        elemSyncMode, wrapElemBufAttr);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemAttr_StreamInternalError}
 * @testcase{22059721}
 * @verify{20050569}
 * @testpurpose{Test negative scenario of IpcSrc::recvElemAttr() when unpacking
 * element attributes failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends its packet element count to pool through
 *      LwSciStreamBlockPacketElementCount().
 *   5. Inject fault in IpcRecvBuffer::unpackMsgElemAttr() to return false.
 *
 *  IpcSrc::recvElemAttr() API call triggered, when IpcDst::dstSendPacketAttr()
 *  API is called with dstIndex of Block::singleConn, elemIndex value less than
 *  consumer's packet element count set earlier through LwSciStreamBlockPacketElementCount(),
 *  valid elemType, elemSyncMode as LwSciStreamElementMode_Asynchronous and
 *  elemBufAttr containing a non-NULL LwSciBufAttrList, should trigger error event
 *  set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvElemAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    wrapElemBufAttr = rawBufAttrList;

    test_ipcrecvbuffer.unpackMsgElemAttr_fail = true;
    ipcsdstPtr->dstSendPacketAttr(dstIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemAttr_ResourceError}
 * @testcase{22059725}
 * @verify{20050569}
 * @testpurpose{Test negative scenario of IpcSrc::recvElemAttr() when
 * LwSciBufAttrListIpcImportUnreconciled() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends its packet element count to pool through
 *      LwSciStreamBlockPacketElementCount().
 *   5. Inject fault in LwSciBufAttrListIpcImportUnreconciled() to return
 *   LwSciError_ResourceError.
 *
 *  IpcSrc::recvElemAttr() API call triggered, when IpcDst::dstSendPacketAttr()
 *  API is called with dstIndex of Block::singleConn, elemIndex value less than
 *  consumer's packet element count set earlier through LwSciStreamBlockPacketElementCount(),
 *  valid elemType, elemSyncMode as LwSciStreamElementMode_Asynchronous and
 *  elemBufAttr containing a non-NULL LwSciBufAttrList, should trigger error event
 *  set to LwSciError_ResourceError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvElemAttr_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    wrapElemBufAttr = rawBufAttrList;

    test_lwscibuf.LwSciBufAttrListIpcImportUnreconciled_fail = true;

    ipcsdstPtr->dstSendPacketAttr(dstIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvElemAttr_IlwalidState}
 * @testcase{22059728}
 * @verify{20050569}
 * @testpurpose{Test negative scenario of IpcSrc::recvElemAttr() when
 * element information for the elemIndex already forwarded upstream.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends its packet element count to pool through
 *      LwSciStreamBlockPacketElementCount().
 *   5. Call IpcDst::dstSendPacketAttr() with dstIndex of Block::singleConn,
 *      elemIndex value less than consumer's packet element count set earlier
 *      through LwSciStreamBlockPacketElementCount(), valid elemType,
 *      elemSyncMode as LwSciStreamElementMode_Asynchronous and elemBufAttr
 *      containing a non-NULL LwSciBufAttrList to send element information.
 *
 *  IpcSrc::recvElemAttr() API call triggered, when IpcDst::dstSendPacketAttr()
 *  API is called with dstIndex of Block::singleConn, same elemIndex as in step-5,
 *  valid elemType, elemSyncMode as LwSciStreamElementMode_Asynchronous and
 *  elemBufAttr containing a non-NULL LwSciBufAttrList, should trigger error event
 *  set to LwSciError_IlwalidState in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvElemAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvElemAttr_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketElementCountConsumer, event.type);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    wrapElemBufAttr = rawBufAttrList;

    ipcsdstPtr->dstSendPacketAttr(dstIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketAttrConsumer, event.type);

    test_trackArray.performAction_fail = true;

    ipcsdstPtr->dstSendPacketAttr(dstIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvPacketStatus_Success1}
 * @testcase{22059732}
 * @verify{20050572}
 * @testpurpose{Test positive scenario of IpcSrc::recvPacketStatus().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Consumer gets the packet create event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *
 *  IpcSrc::recvPacketStatus() API call triggered, when IpcDst::dstSendPacketStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet) and packetStatus is LwSciError_Success, should call
 *  dstSendPacketStatus() interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendPacketStatus()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPacketStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvPacketStatus_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendPacketStatus(_, _, _)).Times(1);

    // To call IpcSrc::recvPacketStatus()
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvPacketStatus_Success2}
 * @testcase{22059735}
 * @verify{20050572}
 * @testpurpose{Test positive scenario of IpcSrc::recvPacketStatus().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Consumer gets the packet create event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *
 *  IpcSrc::recvPacketStatus() API call triggered, when IpcDst::dstSendPacketStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet) and packetStatus is LwSciError_BadParameter, should call
 *  dstSendPacketStatus() interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendPacketStatus()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPacketStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvPacketStatus_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_BadParameter;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendPacketStatus(_, _, _)).Times(1);

    // To call IpcSrc::recvPacketStatus()
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvPacketStatus_StreamInternalError1}
 * @testcase{22059738}
 * @verify{20050572}
 * @testpurpose{Test negative scenario of IpcSrc::recvPacketStatus() when unpacking
 * packet status failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Consumer gets the packet create event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   7. Inject fault in IpcRecvBuffer::unpackMsgStatus() to return false.
 *
 *  IpcSrc::recvPacketStatus() API call triggered, when IpcDst::dstSendPacketStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet) and packetStatus is LwSciError_Success, should trigger
 *  error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPacketStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvPacketStatus_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcrecvbuffer.unpackMsgStatus_fail = true;
    // To call IpcSrc::recvPacketStatus()
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.recvPacketStatus_StreamInternalError2}
 * @testcase{22059741}
 * @verify{20050572}
 * @testpurpose{Test negative scenario of IpcSrc::recvPacketStatus() when packet
 * handle received is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Consumer gets the packet create event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   7. Inject fault in Block::pktFindByHandle() to return NULL.
 *
 *  IpcSrc::recvPacketStatus() API call triggered, when IpcDst::dstSendPacketStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet) and packetStatus is LwSciError_Success, should trigger
 *  error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPacketStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvPacketStatus_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    // To call IpcSrc::recvPacketStatus()
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);
    test_block.pktFindByHandle_fail = true;

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvPacketStatus_IlwalidState}
 * @testcase{22059744}
 * @verify{20050572}
 * @testpurpose{Test negative scenario of IpcSrc::recvPacketStatus() when status
 * for the packet is already received.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Consumer gets the packet create event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   7. Call IpcDst::dstSendPacketStatus() with dstIndex of Block::singleConn,
 *      handle(as reference to the newly created packet) and packetStatus
 *      is LwSciError_Success.
 *
 *  IpcSrc::recvPacketStatus() API call triggered, when IpcDst::dstSendPacketStatus()
 *  API is called with dstIndex of Block::singleConn, same packet handle as in
 *  step-7 and packetStatus is LwSciError_Success, should trigger
 *  error event set to LwSciError_IlwalidState in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPacketStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvPacketStatus_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    // To call IpcSrc::recvPacketStatus()
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketStatusConsumer, event.type);

    test_packet.StatusAction_fail_IlwalidState = true;
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvPacketStatus_BadParameter}
 * @testcase{22059747}
 * @verify{20050572}
 * @testpurpose{Test negative scenario of IpcSrc::recvPacketStatus() when
 * sending packet status upstream failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate().
 *   6. Consumer gets the packet create event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   7. Inject fault in Packet::packetStatusAction() to return LwSciError_BadParameter.
 *
 *  IpcSrc::recvPacketStatus() API call triggered, when IpcDst::dstSendPacketStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet) and packetStatus is LwSciError_Success, should trigger
 *  error event set to LwSciError_BadParameter in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPacketStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvPacketStatus_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    // To call IpcSrc::recvPacketStatus()
    test_packet.StatusAction_fail_BadParameter = true;
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.recvBufferStatus_StreamInternalError1}
 * @testcase{22059750}
 * @verify{20050575}
 * @testpurpose{Test negative scenario of IpcSrc::recvBufferStatus() when
 * unpacking buffer status failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate() and registers buffers
 *      through LwSciStreamPoolPacketInsertBuffer().
 *   6. Consumer accepts the packet from pool through LwSciStreamBlockPacketAccept().
 *   7. Consumer gets the packet element event by querying through LwSciStreamBlockEventQuery() for
 *      all the buffers.
 *   8. Inject fault in IpcRecvBuffer::unpackMsgStatus() to return false.
 *
 *  IpcSrc::recvBufferStatus() API call triggered, when IpcDst::dstSendElementStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet), elemIndex value less than consumer's packet element
 *  count set earlier through LwSciStreamBlockPacketElementCount() and packetStatus
 *  is LwSciError_Success, should trigger error event set to
 *  LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvBufferStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvBufferStatus_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;
    LwSciStreamPacket consumerPacket;
    LwSciStreamCookie consumerCookie;
    LwSciError consumerError;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0; i < consolidatedElementCount; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketInsertBuffer(pool,
                                              handle, i,
                                              poolElementBuf[i]));
    }

    // Assign cookie to consumer packet handle
    consumerPacket = event.packetHandle;
    consumerCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));

    EXPECT_EQ(LwSciError_Success,
          LwSciStreamBlockEventQuery(pool,
                                     EVENT_QUERY_TIMEOUT,
                                     &event));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    for (uint32_t i = 0U; i < 1; ++i) {
        // Receive all the buffers at the consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(consumer[0],
                                             EVENT_QUERY_TIMEOUT,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

        elemIndex = event.index;

        test_ipcrecvbuffer.unpackMsgStatus_fail = true;
        // To call IpcSrc::recvBufferStatus()
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);

        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Error, event.type);
        EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    }
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvBufferStatus_IlwalidState}
 * @testcase{22059442}
 * @verify{20050575}
 * @testpurpose{Test negative scenario of IpcSrc::recvBufferStatus() when
 * status for the element is already received.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate() and registers buffers
 *      through LwSciStreamPoolPacketInsertBuffer().
 *   6. Consumer accepts the packet from pool through LwSciStreamBlockPacketAccept().
 *   7. Consumer gets the packet element event by querying through LwSciStreamBlockEventQuery() for
 *      all the buffers.
 *   8. Call IpcDst::dstSendElementStatus() with dstIndex of Block::singleConn,
 *      handle(as reference to the newly created packet), elemIndex value less
 *     than consumer's packet element count set earlier through
 *     LwSciStreamBlockPacketElementCount() and packetStatus is LwSciError_Success.
 *
 *  IpcSrc::recvBufferStatus() API call triggered, when IpcDst::dstSendElementStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet), same elemIndex as in step-8 and packetStatus
 *  is LwSciError_Success, should trigger error event set to
 *  LwSciError_IlwalidState in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvBufferStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvBufferStatus_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;
    LwSciStreamPacket consumerPacket;
    LwSciStreamCookie consumerCookie;
    LwSciError consumerError;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0; i < consolidatedElementCount; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketInsertBuffer(pool,
                                              handle, i,
                                              poolElementBuf[i]));
    }

    // Assign cookie to consumer packet handle
    consumerPacket = event.packetHandle;
    consumerCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));

    EXPECT_EQ(LwSciError_Success,
          LwSciStreamBlockEventQuery(pool,
                                     EVENT_QUERY_TIMEOUT,
                                     &event));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    for (uint32_t i = 0U; i < 1; ++i) {
        // Receive all the buffers at the consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(consumer[0],
                                             EVENT_QUERY_TIMEOUT,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

        elemIndex = event.index;

        // To call IpcSrc::recvBufferStatus()
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);
        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_ElementStatusConsumer, event.type);

        test_trackArray.performAction_fail = true;
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);

        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Error, event.type);
        EXPECT_EQ(LwSciError_IlwalidState, event.error);
    }
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.recvBufferStatus_BadParameter1}
 * @testcase{22059448}
 * @verify{20050575}
 * @testpurpose{Test negative scenario of IpcSrc::recvBufferStatus() when
 * element index unpacked is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate() and registers buffers
 *      through LwSciStreamPoolPacketInsertBuffer().
 *   6. Consumer accepts the packet from pool through LwSciStreamBlockPacketAccept().
 *   7. Consumer gets the packet element event by querying through LwSciStreamBlockEventQuery() for
 *      all the buffers.
 *
 *  IpcSrc::recvBufferStatus() API call triggered, when IpcDst::dstSendElementStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet), elemIndex greate than MAX_PACKET_ELEMENTS and packetStatus
 *  is LwSciError_Success, should trigger error event set to
 *  LwSciError_BadParameter in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvBufferStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvBufferStatus_BadParameter1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;
    LwSciStreamPacket consumerPacket;
    LwSciStreamCookie consumerCookie;
    LwSciError consumerError;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0; i < consolidatedElementCount; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketInsertBuffer(pool,
                                              handle, i,
                                              poolElementBuf[i]));
    }

    // Assign cookie to consumer packet handle
    consumerPacket = event.packetHandle;
    consumerCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));

    EXPECT_EQ(LwSciError_Success,
          LwSciStreamBlockEventQuery(pool,
                                     EVENT_QUERY_TIMEOUT,
                                     &event));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    for (uint32_t i = 0U; i < 1; ++i) {
        // Receive all the buffers at the consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(consumer[0],
                                             EVENT_QUERY_TIMEOUT,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

        elemIndex = MAX_PACKET_ELEMENTS+1U;

        test_packet.StatusAction_fail_BadParameter = true;
        // To call IpcSrc::recvBufferStatus()
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);

        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Error, event.type);
        EXPECT_EQ(LwSciError_BadParameter, event.error);
    }
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvBufferStatus_BadParameter2}
 * @testcase{22059453}
 * @verify{20050575}
 * @testpurpose{Test negative scenario of IpcSrc::recvBufferStatus() when
 * sending buffer status upstream failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate() and registers buffers
 *      through LwSciStreamPoolPacketInsertBuffer().
 *   6. Consumer accepts the packet from pool through LwSciStreamBlockPacketAccept().
 *   7. Consumer gets the packet element event by querying through LwSciStreamBlockEventQuery() for
 *      all the buffers.
 *   8. Inject fault in Packet::bufferStatusAction() to return LwSciError_BadParameter.
 *
 *  IpcSrc::recvBufferStatus() API call triggered, when IpcDst::dstSendElementStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet), elemIndex value less than consumer's packet element
 *  count set earlier through LwSciStreamBlockPacketElementCount() and packetStatus
 *  is LwSciError_Success, should trigger error event set to
 *  LwSciError_BadParameter in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvBufferStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvBufferStatus_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;
    LwSciStreamPacket consumerPacket;
    LwSciStreamCookie consumerCookie;
    LwSciError consumerError;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0; i < consolidatedElementCount; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketInsertBuffer(pool,
                                              handle, i,
                                              poolElementBuf[i]));
    }

    // Assign cookie to consumer packet handle
    consumerPacket = event.packetHandle;
    consumerCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));

    EXPECT_EQ(LwSciError_Success,
          LwSciStreamBlockEventQuery(pool,
                                     EVENT_QUERY_TIMEOUT,
                                     &event));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    for (uint32_t i = 0U; i < 1; ++i) {
        // Receive all the buffers at the consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(consumer[0],
                                             EVENT_QUERY_TIMEOUT,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

        elemIndex = event.index;

        test_packet.BufferStatusAction_fail_BadParameter = true;
        // To call IpcSrc::recvBufferStatus()
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);

        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Error, event.type);
        EXPECT_EQ(LwSciError_BadParameter, event.error);
    }
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.recvBufferStatus_StreamInternalError2}
 * @testcase{22059456}
 * @verify{20050575}
 * @testpurpose{Test negative scenario of IpcSrc::recvBufferStatus() when
 * packet handle received is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate() and registers buffers
 *      through LwSciStreamPoolPacketInsertBuffer().
 *   6. Consumer accepts the packet from pool through LwSciStreamBlockPacketAccept().
 *   7. Consumer gets the packet element event by querying through LwSciStreamBlockEventQuery() for
 *      all the buffers.
 *   8. Inject fault in Block::pktFindByHandle() to return NULL.
 *
 *  IpcSrc::recvBufferStatus() API call triggered, when IpcDst::dstSendElementStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet), elemIndex value less than consumer's packet element
 *  count set earlier through LwSciStreamBlockPacketElementCount() and packetStatus
 *  is LwSciError_Success, should trigger error event set to
 *  LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvBufferStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvBufferStatus_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;
    LwSciStreamPacket consumerPacket;
    LwSciStreamCookie consumerCookie;
    LwSciError consumerError;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0; i < 1; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketInsertBuffer(pool,
                                              handle, i,
                                              poolElementBuf[i]));
    }

    // Assign cookie to consumer packet handle
    consumerPacket = event.packetHandle;
    consumerCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    consumerError = LwSciError_Success;

       // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));

    EXPECT_EQ(LwSciError_Success,
          LwSciStreamBlockEventQuery(pool,
                                     EVENT_QUERY_TIMEOUT,
                                     &event));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    for (uint32_t i = 0U; i < 1; ++i) {
        // Receive all the buffers at the consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(consumer[0],
                                             EVENT_QUERY_TIMEOUT,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

        elemIndex = event.index;

        // To call IpcSrc::recvBufferStatus()
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);
        test_block.pktFindByHandle_fail = true;

        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Error, event.type);
        EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    }
}


/**
 * @testname{ipcsrc_packet_stream_test.srcSendPacket_Success}
 * @testcase{22059462}
 * @verify{19675914}
 * @testpurpose{Test positive scenario of IpcSrc::srcSendPacket().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *
 *   The call of IpcSrc::srcSendPacket() API from ipcsrc object,
 * with srcIndex of Block::singleConn, handle(as reference to the producer presented packet) and
 * postfences containing a non-NULL FenceArray, should call srcSendPacket() interface of the Queue
 * block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Queue::srcSendPacket()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacket()}
 */
TEST_F(ipcsrc_packet_stream_test, srcSendPacket_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;
    std::shared_ptr<LwSciStream::Mailbox> mailboxPtr;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
     }

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    mailboxPtr = std::static_pointer_cast<LwSciStream::Mailbox>(queuePtr[0]);
    EXPECT_CALL(*mailboxPtr, srcSendPacket(_, _, _)).Times(1);

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&mailboxPtr));
    free(fences);
}


/**
 * @testname{ipcsrc_unit_test.srcDisconnect_Success}
 * @testcase{22059469}
 * @verify{19675917}
 * @testpurpose{Test positive scenario of IpcSrc::srcDisconnect().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcDisconnect() API from ipcsrc object,
 * with srcIndex of Block::singleConn, should call srcDisconnect() interface of the Queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcDisconnect()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcDisconnect()}
 */
TEST_F(ipcsrc_unit_test, srcDisconnect_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcDisconnect(_)).Times(1);

    ipcsrcPtr->srcDisconnect(srcIndex);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncCount_Success}
 * @testcase{22059475}
 * @verify{20050560}
 * @testpurpose{Test positive scenario of IpcSrc::recvSyncCount().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *  IpcSrc::recvSyncCount() API call triggered, when IpcDst::dstSendSyncCount()
 *  API is called with valid a sync count of 2 and dstIndex of Block::singleConn,
 *  should call dstSendSyncCount() interface of Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendSyncCount()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncCount_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = consSyncCount[0];

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendSyncCount(_, _)).Times(1);

    // To call IpcSrc::recvSyncCount()
    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{ipcsrc_unit_buf_setup_test.recvBufferStatus_Success}
 * @testcase{22059512}
 * @verify{20050575}
 * @testpurpose{Test positive scenario of IpcSrc::recvBufferStatus().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the packet attributes for streaming.
 *   5. Pool creates a new packet through LwSciStreamPoolPacketCreate() and registers buffers
 *      through LwSciStreamPoolPacketInsertBuffer().
 *   6. Consumer accepts the packet from pool through LwSciStreamBlockPacketAccept().
 *   7. Consumer gets the packet element event by querying through LwSciStreamBlockEventQuery() for
 *      all the buffers.
 *
 *  IpcSrc::recvBufferStatus() API call triggered, when IpcDst::dstSendElementStatus()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  newly created packet), elemIndex value less than consumer's packet element
 *  count set earlier through LwSciStreamBlockPacketElementCount() and packetStatus
 *  is LwSciError_Success, should call dstSendElementStatus() interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Pool::dstSendElementStatus()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvBufferStatus()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, recvBufferStatus_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;

    LwSciStreamCookie poolCookie;
    LwSciStreamEvent event;
    LwSciStreamPacket consumerPacket;
    LwSciStreamCookie consumerCookie;
    LwSciError consumerError;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    poolCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    // Register buffer to packet handle
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];
    for (uint32_t i = 0; i < consolidatedElementCount; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        ASSERT_EQ(LwSciError_Success,
            LwSciStreamPoolPacketInsertBuffer(pool,
                                              handle, i,
                                              poolElementBuf[i]));
    }

    // Assign cookie to consumer packet handle
    consumerPacket = event.packetHandle;
    consumerCookie = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendElementStatus(_, _, _, _))
        .Times(consolidatedElementCount);

    for (uint32_t i = 0U; i < consolidatedElementCount; ++i) {
        // Receive all the buffers at the consumer
        EXPECT_EQ(LwSciError_Success,
                  LwSciStreamBlockEventQuery(consumer[0],
                                             EVENT_QUERY_TIMEOUT,
                                             &event));
        EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

        elemIndex = event.index;

        // To call IpcSrc::recvBufferStatus()
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{ipcsrc_packet_stream_test.recvPayload_Success}
 * @testcase{22059518}
 * @verify{20050578}
 * @testpurpose{Test positive scenario of IpcSrc::recvPayload().}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Consumer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *  10. Consumer acquires the packet through LwSciStreamConsumerPacketAcquire()
 *      and retrieves data from the packet.
 *
 *  IpcSrc::recvPayload() API call triggered, when IpcDst::dstReusePacket()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  producer presented packet) and postfences containing a non-NULL FenceArray,
 *  should call dstReusePacket() interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Pool::dstReusePacket()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPayload()}
 */
TEST_F(ipcsrc_packet_stream_test, recvPayload_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketPresent(producer, handle, fences));

    // Pool sends packet ready event to consumer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Consumer acquires packet from the queue
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie, fences));
    handle = (consCPMap[0])[cookie];

    // Consumer returns a data packet to the stream
    for (uint32_t i = 0U; i < consSyncCount[0]; i++) {
        fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < consSyncCount[0]; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
    }

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstReusePacket(_, _, _)).Times(1);

    // To call IpcSrc::dstReusePacket()
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.recvPayload_StreamInternalError1}
 * @testcase{22059382}
 * @verify{20050578}
 * @testpurpose{Test negative scenario of IpcSrc::recvPayload() when unpacking
 * payload failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Consumer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *  10. Consumer acquires the packet through LwSciStreamConsumerPacketAcquire()
 *      and retrieves data from the packet.
 *  11. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *  IpcSrc::recvPayload() API call triggered, when IpcDst::dstReusePacket()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  producer presented packet) and postfences containing a non-NULL FenceArray,
 *  should trigger error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPayload()}
 */
TEST_F(ipcsrc_packet_stream_test, recvPayload_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketPresent(producer, handle, fences));

    // Pool sends packet ready event to consumer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Consumer acquires packet from the queue
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie, fences));
    handle = (consCPMap[0])[cookie];

    // Consumer returns a data packet to the stream
    for (uint32_t i = 0U; i < consSyncCount[0]; i++) {
        fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < consSyncCount[0]; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
    }

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_ipcrecvbuffer.unpackVal_fail = true;

    // To call IpcSrc::dstReusePacket()
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.recvPayload_StreamInternalError2}
 * @testcase{22059384}
 * @verify{20050578}
 * @testpurpose{Test negative scenario of IpcSrc::recvPayload() when packet
 * handle received is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Consumer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *  10. Consumer acquires the packet through LwSciStreamConsumerPacketAcquire()
 *      and retrieves data from the packet.
 *  11. Inject fault in Block::pktFindByHandle() to return NULL.
 *
 *  IpcSrc::recvPayload() API call triggered, when IpcDst::dstReusePacket()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  producer presented packet) and postfences containing a non-NULL FenceArray,
 *  should trigger error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPayload()}
 */
TEST_F(ipcsrc_packet_stream_test, recvPayload_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketPresent(producer, handle, fences));

    // Pool sends packet ready event to consumer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Consumer acquires packet from the queue
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie, fences));
    handle = (consCPMap[0])[cookie];

    // Consumer returns a data packet to the stream
    for (uint32_t i = 0U; i < consSyncCount[0]; i++) {
        fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < consSyncCount[0]; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
    }

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    // To call IpcSrc::dstReusePacket()
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    test_block.pktFindByHandle_fail = true;

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcsrc_packet_stream_test.recvPayload_StreamInternalError3}
 * @testcase{22059387}
 * @verify{20050578}
 * @testpurpose{Test negative scenario of IpcSrc::recvPayload() when packet
 * location update failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Consumer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *  10. Consumer acquires the packet through LwSciStreamConsumerPacketAcquire()
 *      and retrieves data from the packet.
 *  11. Inject fault in Packet::locationUpdate() to return false.
 *
 *  IpcSrc::recvPayload() API call triggered, when IpcDst::dstReusePacket()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  producer presented packet) and postfences containing a non-NULL FenceArray,
 *  should trigger error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPayload()}
 */
TEST_F(ipcsrc_packet_stream_test, recvPayload_StreamInternalError3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketPresent(producer, handle, fences));

    // Pool sends packet ready event to consumer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Consumer acquires packet from the queue
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie, fences));
    handle = (consCPMap[0])[cookie];

    // Consumer returns a data packet to the stream
    for (uint32_t i = 0U; i < consSyncCount[0]; i++) {
        fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < consSyncCount[0]; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
    }

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_packet.locationUpdate_fail = true;

    // To call IpcSrc::dstReusePacket()
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}


/**
 * @testname{ipcsrc_packet_stream_test.recvPayload_StreamInternalError4}
 * @testcase{22059390}
 * @verify{20050578}
 * @testpurpose{Test negative scenario of IpcSrc::recvPayload() when
 * IpcRecvBuffer::unpackFenceExport() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Consumer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *  10. Consumer acquires the packet through LwSciStreamConsumerPacketAcquire()
 *      and retrieves data from the packet.
 *  11. Inject fault in IpcRecvBuffer::unpackFenceExport() to return false.
 *
 *  IpcSrc::recvPayload() API call triggered, when IpcDst::dstReusePacket()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  producer presented packet) and postfences containing a non-NULL FenceArray,
 *  should trigger error event set to LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPayload()}
 */
TEST_F(ipcsrc_packet_stream_test, recvPayload_StreamInternalError4)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketPresent(producer, handle, fences));

    // Pool sends packet ready event to consumer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Consumer acquires packet from the queue
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie, fences));
    handle = (consCPMap[0])[cookie];

    // Consumer returns a data packet to the stream
    for (uint32_t i = 0U; i < consSyncCount[0]; i++) {
        fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < consSyncCount[0]; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
    }

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_ipcrecvbuffer.unpackFenceExport_fail = true;

    // To call IpcSrc::dstReusePacket()
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}


/**
 * @testname{ipcsrc_packet_stream_test.recvPayload_ResourceError}
 * @testcase{22059393}
 * @verify{20050578}
 * @testpurpose{Test negative scenario of IpcSrc::recvPayload() when
 * LwSciSyncIpcImportFence() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Set up the synchronization resources and between producer and consumer.
 *   5. Set up the packet attributes for streaming.
 *   6. Pool creates a packet, registers buffers and checks the packet status.
 *   7. Producer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *   8. Producer gets the packet through LwSciStreamProducerPacketGet()
 *      and inserts data into the packet.
 *   9. Consumer gets the packet ready event from pool by querying
 *      through LwSciStreamBlockEventQuery().
 *  10. Consumer acquires the packet through LwSciStreamConsumerPacketAcquire()
 *      and retrieves data from the packet.
 *  11. Inject fault in LwSciSyncIpcImportFence() to return LwSciError_ResourceError.
 *
 *  IpcSrc::recvPayload() API call triggered, when IpcDst::dstReusePacket()
 *  API is called with dstIndex of Block::singleConn, handle(as reference to the
 *  producer presented packet) and postfences containing a non-NULL FenceArray,
 *  should trigger error event set to LwSciError_ResourceError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvPayload()}
 */
TEST_F(ipcsrc_packet_stream_test, recvPayload_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    FenceArray wrapPostfences;

    uint32_t maxSync;
    LwSciSyncFence *fences;
    LwSciStreamCookie cookie;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Create and exchange sync objects
    createSync();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Create packets
    createPacket();

    // Check packet status
    checkPacketStatus();

    maxSync = (totalConsSync > prodSyncCount)
                ? totalConsSync : prodSyncCount;

    fences = static_cast<LwSciSyncFence*>(
                malloc(sizeof(LwSciSyncFence) * maxSync));

    // Pool sends packet ready event to producer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Producer gets a packet from the pool
    for (uint32_t i = 0U; i < totalConsSync; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketGet(producer, &cookie, fences));
    handle = prodCPMap[cookie];

    // Producer inserts a data packet into the stream
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamProducerPacketPresent(producer, handle, fences));

    // Pool sends packet ready event to consumer
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketReady, event.type);

    // Consumer acquires packet from the queue
    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
                fences[i] = LwSciSyncFenceInitializer;
    }
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamConsumerPacketAcquire(consumer[0], &cookie, fences));
    handle = (consCPMap[0])[cookie];

    // Consumer returns a data packet to the stream
    for (uint32_t i = 0U; i < consSyncCount[0]; i++) {
        fences[i] = LwSciSyncFenceInitializer;
    }

    for (uint32_t i = 0U; i < consSyncCount[0]; ++i) {
        wrapPostfences[i] = LwSciWrap::SyncFence(fences[i]);
    }

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_lwscisync.LwSciSyncIpcImportFence_fail = true;

    // To call IpcSrc::dstReusePacket()
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

    free(fences);
}


/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncAttr_StreamBadSrcIndex}
 * @testcase{22059396}
 * @verify{19675884}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncAttr(), where
 * srcSendSyncAttr() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendSyncAttr() API should result in
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncAttr_StreamBadSrcIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    bool synchronousOnly = prodSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncAttr_StreamNotConnected}
 * @testcase{22059400}
 * @verify{19675884}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncAttr(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcSendSyncAttr() API should trigger error event
 * set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncAttr_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    bool synchronousOnly = prodSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncCount_IlwalidState}
 * @testcase{22059403}
 * @verify{19675887}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncCount() when
 *  LwSciSyncObj count has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Call IpcSrc::srcSendSyncCount() to send the count.
 *
 *   The call of IpcSrc::srcSendSyncCount() API from ipcsrc object,
 * with valid a sync count of 2 and srcIndex of Block::singleConn,
 * should trigger error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncCount_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = prodSyncCount;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncCount_StreamInternalError1}
 * @testcase{22059406}
 * @verify{19675887}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncCount() when
 *  IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcSendSyncCount() API from ipcsrc object,
 * with valid a sync count of 2 and srcIndex of Block::singleConn,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncCount_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = prodSyncCount;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    test_comm.signalWrite_fail = true;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncCount_StreamBadSrcIndex}
 * @testcase{22059408}
 * @verify{19675887}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncCount(), where
 * srcSendSynCount() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendSyncCount() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcSendSyncCount()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncCount_StreamBadSrcIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    uint32_t count = prodSyncCount;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncCount_StreamInternalError2}
 * @testcase{22059411}
 * @verify{19675887}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncCount(), where
 * srcSendSyncCount() is called with count greater than MAX_SYNC_OBJECTS.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendSyncCount() API from ipcsrc object should result
 * LwSciError_StreamInternalError error which is to be queried through
 * LwSciStreamBlockEventQuery() with count value which is greater than MAX_SYNC_OBJECTS.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncCount_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = 5U;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncCount_StreamNotConnected}
 * @testcase{22059414}
 * @verify{19675887}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncCount(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcSendSyncCount() API from ipcsrc object, should
 * trigger error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncCount()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncCount_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = 2U;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncDesc_StreamBadSrcIndex}
 * @testcase{22059417}
 * @verify{19675890}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncDesc(), where
 * srcSendSyncDesc() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncDesc_StreamBadSrcIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    uint32_t syncIndex = 0U;
    LwSciWrap::SyncObj wrapSyncObj = { prodSyncObjs[0U] };
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncDesc_StreamNotConnected}
 * @testcase{22059422}
 * @verify{19675890}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncDesc(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object should
 *  trigger error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncDesc_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t syncIndex = 0U;
    LwSciWrap::SyncObj wrapSyncObj = { prodSyncObjs[0U] };
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncDesc_IlwalidState1}
 * @testcase{22059425}
 * @verify{19675890}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncDesc(), when
 * IpcSrc::srcSendSyncCount() is not yet called.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object with valid
 *   parameters should trigger the error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncDesc_IlwalidState1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t syncIndex = 0U;
    LwSciWrap::SyncObj wrapSyncObj = { prodSyncObjs[0U] };
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncDesc_IlwalidState2}
 * @testcase{22059428}
 * @verify{19675890}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncDesc(), when
 * LwSciSyncObj for the same syncIndex has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends the waiter requirements to producer by calling
 *      LwSciStreamBlockSyncRequirements() interface.
 *   5. Producer send the sync objects count to consumer by calling
 *      LwSciStreamBlockSyncObjCount().
 *   6. Call IpcSrc::srcSendSyncDesc() to send the LwSciSyncObj for syncIndex=0.
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object for the same
 *   syncIndex with valid other parameters, should trigger the error event
 *   set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncDesc_IlwalidState2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t syncIndex = 0U;
    LwSciWrap::SyncObj wrapSyncObj = { prodSyncObjs[0U] };
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its sync object requirement to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,
                                         consSyncAttrList));

    // Producer receives consumer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncDesc_BadParameter}
 * @testcase{22059429}
 * @verify{19675890}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncDesc(), when
 * syncIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Consumer sends the waiter requirements to producer by calling
 *      LwSciStreamBlockSyncRequirements() interface.
 *   5. Producer send the sync objects count to consumer by calling
 *      LwSciStreamBlockSyncObjCount().
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object for the invalid
 *   syncIndex(greater than the count set earlier using LwSciStreamBlockSyncObjCount)
 *   with valid other parameters, should trigger the error event
 *   set to LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncDesc_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t syncIndex = 5U;
    LwSciWrap::SyncObj wrapSyncObj = { prodSyncObjs[0U] };
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its sync object requirement to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,
                                         consSyncAttrList));

    // Producer receives consumer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };


    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

 /**
 * @testname{ipcsrc_unit_sync_setup_test.srcSendSyncDesc_StreamInternalError}
 * @testcase{22059432}
 * @verify{19675890}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendSyncDesc(), when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcSrc::srcSendSyncDesc() API from ipcsrc object with valid
 *   parameters should trigger the error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendSyncDesc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, srcSendSyncDesc_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t syncIndex = 0U;
    LwSciWrap::SyncObj wrapSyncObj = { prodSyncObjs[0U] };
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its sync object requirement to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(consumer[0],
                                         consSynchronousOnly,
                                         consSyncAttrList));

    // Producer receives consumer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);
    EXPECT_EQ(consSynchronousOnly, event.synchronousOnly);

    // Producer sends its sync count to the consumer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(producer, prodSyncCount));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    test_comm.signalWrite_fail = true;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketElementCount_StreamBadSrcIndex}
 * @testcase{22059437}
 * @verify{19675893}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketElementCount(), where
 * srcSendPacketElementCount() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendPacketElementCount() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendPacketElementCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketElementCount_StreamBadSrcIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    uint32_t count = consolidatedElementCount;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketElementCount_BadParameter}
 * @testcase{22059440}
 * @verify{19675893}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketElementCount(),
 * when count is greater than MAX_PACKET_ELEMENTS.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendPacketElementCount() API from ipcsrc object with
 *  count value as MAX_PACKET_ELEMENTS+1, should trigger error event set to
 * LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendPacketElementCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketElementCount_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t count = MAX_PACKET_ELEMENTS+1U;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketElementCount_StreamNotConnected}
 * @testcase{22059446}
 * @verify{19675893}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketElementCount(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcSendPacketElementCount() API from ipcsrc object should
 * trigger error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcSendPacketElementCount()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketElementCount_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t count = consolidatedElementCount;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketAttr_StreamBadSrcIndex}
 * @testcase{22059450}
 * @verify{19675896}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketAttr(), where
 * srcSendPacketAttr() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketAttr_StreamBadSrcIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    uint32_t elemIndex = 0U;
    uint32_t elemType = 0U;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr = rawBufAttrList;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendPacketAttr(srcIndex, elemIndex, elemType,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcSendPacketAttr_StreamNotConnected}
 * @testcase{22059461}
 * @verify{19675896}
 * @testpurpose{Test negative scenario of IpcSrc::srcSendPacketAttr(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcSendPacketAttr() API from ipcsrc object should
 * trigger error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcSendPacketAttr()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcSendPacketAttr_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    uint32_t elemIndex = 0U;
    uint32_t elemType = 0U;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;
    LwSciWrap::BufAttr wrapElemBufAttr = rawBufAttrList;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcSendPacketAttr(srcIndex, elemIndex, elemType,
                                    elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}
/**
 * @testname{ipcsrc_unit_buf_setup_test.srcCreatePacket_StreamBadSrcIndex}
 * @testcase{22059465}
 * @verify{19675899}
 * @testpurpose{Test negative scenario of IpcSrc::srcCreatePacket(), where
 * srcCreatePacket() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcCreatePacket() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcCreatePacket()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcCreatePacket()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcCreatePacket_StreamBadSrcIndex)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    LwSciStreamPacket handle;

    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcCreatePacket_StreamNotConnected}
 * @testcase{22059471}
 * @verify{19675899}
 * @testpurpose{Test negative scenario of IpcSrc::srcCreatePacket(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcCreatePacket() API from ipcsrc object, should
 * trigger error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Queue::srcCreatePacket()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::srcCreatePacket()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcCreatePacket_StreamNotConnected)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    LwSciStreamPacket handle;

    LwSciStreamEvent event;
    LwSciStreamCookie poolCookie;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_StreamBadSrcIndex}
 * @testcase{22059477}
 * @verify{19675905}
 * @testpurpose{Test negative scenario of IpcSrc::srcInsertBuffer(), where
 * srcInsertBuffer() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_StreamBadSrcIndex)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0U;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    makeRawBuffer(rawBufAttrList, poolElementBuf[0U]);

    ipcsrcPtr->srcInsertBuffer(srcIndex, handle,
                                elemIndex, wrapElemBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_buf_setup_test.srcInsertBuffer_StreamNotConnected}
 * @testcase{22059480}
 * @verify{19675905}
 * @testpurpose{Test negative scenario of IpcSrc::srcInsertBuffer(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcInsertBuffer() API from ipcsrc object should trigger
 *  error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcInsertBuffer()}
 */
TEST_F(ipcsrc_unit_buf_setup_test, srcInsertBuffer_StreamNotConnected)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0U;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciStreamCookie poolCookie;
    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    makeRawBuffer(rawBufAttrList, poolElementBuf[0U]);

    ipcsrcPtr->srcInsertBuffer(srcIndex, handle,
                                elemIndex, wrapElemBufObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcsrc_unit_test.srcDeletePacket_StreamBadSrcIndex}
 * @testcase{22059484}
 * @verify{19675908}
 * @testpurpose{Test negative scenario of IpcSrc::srcDeletePacket(), where
 * srcDeletePacket() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcDeletePacket() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcDeletePacket()}
 */
TEST_F(ipcsrc_unit_test, srcDeletePacket_StreamBadSrcIndex)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;

    LwSciStreamCookie poolCookie;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };
    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcDeletePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_test.srcDeletePacket_StreamNotConnected}
 * @testcase{22059486}
 * @verify{19675908}
 * @testpurpose{Test negative scenario of IpcSrc::srcDeletePacket(), when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *
 *   The call of IpcSrc::srcDeletePacket() API from ipcsrc object, should trigger
 *  error event set to LwSciError_StreamNotConnected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcDeletePacket()}
 */
TEST_F(ipcsrc_unit_test, srcDeletePacket_StreamNotConnected)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 0U;
    LwSciStreamPacket handle;
    LwSciStreamEvent event;

    LwSciStreamCookie poolCookie;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };
    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcDeletePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcsrc_unit_test.srcDisconnect_StreamBadSrcIndex}
 * @testcase{22059488}
 * @verify{19675917}
 * @testpurpose{Test negative scenario of IpcSrc::srcDisconnect(), where
 * srcDisconnect() is called with invalid srcIndex.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   The call of IpcSrc::srcDisconnect() API from ipcsrc object should result
 * LwSciError_StreamBadSrcIndex error which is to be queried through
 * LwSciStreamBlockEventQuery() with invalid SrcIndex of value 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::srcDisconnect()}
 */
TEST_F(ipcsrc_unit_test, srcDisconnect_StreamBadSrcIndex)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = 5U;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsrcPtr->srcDisconnect(srcIndex);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncAttr_StreamInternalError}
 * @testcase{22059492}
 * @verify{20050557}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncAttr()
 * when unpacking MsgSyncAttr failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject failure in IpcRecvBuffer::unpackMsgSyncAttr() to return false.
 *
 *  IpcSrc::recvSyncAttr() API call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag as false and
 *  LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and dstIndex as
 *  Block::singleConn, should trigger error event set to
 *  LwSciError_StreamInternalError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = consSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_ipcrecvbuffer.unpackMsgSyncAttr_fail = true;
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncAttr_IlwalidState}
 * @testcase{22059496}
 * @verify{20050557}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncAttr()
 * when waiter requirements already forwarded to upstream.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Call IpcDst::dstSendSyncAttr() with valid sync attributes
 *      (synchronousOnly flag as false and LwSciWrap::SyncAttr wraps a
 *      valid LwSciSyncAttrList) and dstIndex as Block::singleConn.
 *
 *  IpcSrc::recvSyncAttr() API call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag as false and
 *  LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and dstIndex as
 *  Block::singleConn, should trigger error event set to
 *  LwSciError_IlwalidState in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncAttr()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, recvSyncAttr_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = consSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    test_trackArray.performAction_fail = true;
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.recvSyncAttr_ResourceError}
 * @testcase{22059498}
 * @verify{20050557}
 * @testpurpose{Test negative scenario of IpcSrc::recvSyncAttr()
 * when LwSciSyncAttrListIpcImportUnreconciled() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject failure in LwSciSyncAttrListIpcImportUnreconciled() to return
 *      LwSciError_ResourceError.
 *
 *  IpcSrc::recvSyncAttr() API call triggered, when IpcDst::dstSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag as false and
 *  LwSciWrap::SyncAttr wraps a valid LwSciSyncAttrList) and dstIndex as
 *  Block::singleConn, should trigger error event set to
 *  LwSciError_ResourceError in IpcSrc.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcSrc::recvSyncAttr()}
 */
 TEST_F(ipcsrc_unit_sync_setup_test, recvSyncAttr_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = consSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();
    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_lwscisync.LwSciSyncAttrListIpcImportUnreconciled_fail = true;
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
}

/**
 * @testname{ipcsrc_unit_test.dispatchThreadFunc_StreamInternalError1}
 * @testcase{22059500}
 * @verify{19838805}
 * @testpurpose{Test negative scenario of IpcSrc::dispatchThreadFunc() when
*   IpcComm::waitForEvent() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Stub the return value of IpcComm::waitForConnection() to complete
 *      the connection successfully.
 *   3. Create a process write event by stubbing the implementation of
 *     IpcComm::waitForEvent().
 *   4. Upon first invocation of IpcComm::waitForEvent() should return write
 *     event with error flag set to LwSciError_Success.
 *   5. In the second invocation, IpcComm::waitForEvent() should return write event
 *     with error flag set to LwSciError_StreamInternalError.
 *
 *   The call of LwSciStreamIpcSrcCreate() with required parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processWriteMsg()}
 */
TEST_F(ipcsrc_unit_test, dispatchThreadFunc_StreamInternalError1)
{
 /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    test_comm.waitForConnection_pass = true;
    test_comm.waitForEvent_flag = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_Success,
    LwSciStreamIpcSrcCreate(ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_test.dispatchThreadFunc_StreamInternalError2}
 * @testcase{22059504}
 * @verify{19838805}
 * @testpurpose{Test negative scenario of IpcSrc::dispatchThreadFunc() when
 *   IpcComm::flushWriteSignals() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Stub the return value of IpcComm::waitForConnection() to complete
 *      the connection successfully.
 *
 *   The call of LwSciStreamIpcSrcCreate() with required parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::dispatchThreadFunc()}
 */
TEST_F(ipcsrc_unit_test, dispatchThreadFunc_StreamInternalError2)
{
 /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    test_comm.waitForConnection_pass = true;
    test_comm.flushWriteSignals_fail = true;
    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_Success,
    LwSciStreamIpcSrcCreate(ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.processWriteMsg_StreamInternalError1}
 * @testcase{22059507}
 * @verify{19977678}
 * @testpurpose{Test negative scenario of IpcSrc::processWriteMsg() when packing
 * event type failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcSrc::srcSendSyncCount() with valid parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::processWriteMsg()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, processWriteMsg_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamEvent event;
    uint32_t count = prodSyncCount;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.processMsg_pack_fail = true;
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcsrc_unit_sync_setup_test.IpcSrc_SendBuffer_StreamInternalError}
 * @testcase{22059509}
 * @verify{19675875}
 * @testpurpose{Test negative scenario of LwSciStream::IpcSrc::IpcSrc
 * when IpcComm::isInitSuccess() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Inject fault in IpcSendBuffer::isInitSuccess() to return false.
 *
 *   The call of LwSciStream::IpcSrc::IpcSrc constructor through LwSciStreamIpcSrcCreate()
 * should return LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::IpcSrc::IpcSrc}
 */
TEST_F(ipcsrc_packet_stream_test, IpcSrc_SendBuffer_StreamInternalError)
{
    /*Initial setup*/

    // Initialise Ipc channel
    initIpcChannel();

    //IpcSendBuffer::isInitSuccess returns false
    test_ipcsendbuffer.isInitSuccess_fail = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_StreamInternalError,
    LwSciStreamIpcSrcCreate(ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));
}

/**
 * @testname{ipcsrc_unit_sync_setup_test.dispatchThreadFunc_Unknown}
 * @testcase{22059516}
 * @verify{19838805}
 * @testpurpose{Test negative scenario of LwSciStream::IpcSrc::dispatchThreadFunc
 * when IpcComm::waitForConnection() is failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Inject failure in IpcComm::waitForConnection() to return LwSciError_Unknown.
 *
 *   The call of LwSciStream::IpcSrc::dispatchThreadFunc() through LwSciStreamIpcSrcCreate()
 * should trigger error event set to LwSciError_Unknown.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSrc::dispatchThreadFunc()}
 */
TEST_F(ipcsrc_unit_sync_setup_test, dispatchThreadFunc_Unknown)
{
    //Initial setup
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //IpcComm::isInitSuccess returns LwSciError_Unknown
    test_comm.waitForConnection_fail = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_Success,
    LwSciStreamIpcSrcCreate(ipcSrc.endpoint, syncModule, bufModule, &ipcsrc));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcsrc, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_Unknown, event.error);

}

} // namespace LwSciStream


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
