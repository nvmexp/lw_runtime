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
// Define ipcdst_unit_test suite
//==============================================
class ipcdst_unit_test: public LwSciStreamTest {
public:
    ipcdst_unit_test( ) {
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

    ~ipcdst_unit_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

//==============================================
// Define ipcdst_unit_sync_setup_test suite
//==============================================
class ipcdst_unit_sync_setup_test: public LwSciStreamTest {
public:
    ipcdst_unit_sync_setup_test( ) {
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

    ~ipcdst_unit_sync_setup_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

//==============================================
// Define ipcdst_unit_buf_setup_test suite
//==============================================
class ipcdst_unit_buf_setup_test: public LwSciStreamTest {
public:
    ipcdst_unit_buf_setup_test( ) {
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

    ~ipcdst_unit_buf_setup_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

//==============================================
// Define ipcdst_packet_stream_test suite
//==============================================
class ipcdst_packet_stream_test: public LwSciStreamTest {
public:
    ipcdst_packet_stream_test( ) {
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

    ~ipcdst_packet_stream_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};



namespace LwSciStream {

/**
 * @testname{ipcdst_unit_test.dstSendSyncCount_Success}
 * @testcase{21808610}
 * @verify{19791585}
 * @testpurpose{Test positive scenario of IpcDst::dstSendSyncCount(), where
 * dstSendSyncCount() is called with sync count less than MAX_SYNC_OBJECTS.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendSyncCount() API from IpcDst object,
 * with a valid sync count of 1 and dstIndex of Block::singleConn,
 * should in-turn ilwoke dstSendSyncCount() interface of Pool block, to send the sync count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - POOL::dstSendSyncCount()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncCount()}
 */
TEST_F(ipcdst_unit_test, dstSendSyncCount_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = 1U;

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
    EXPECT_CALL(*poolPtr, dstSendSyncCount(_, _)).Times(1);

    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcdst_unit_test.disconnect_Success}
 * @testcase{21808611}
 * @verify{19791579}
 * @testpurpose{Test positive scenario of IpcDst::disconnect()}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::disconnect() API from IpcDst object,
 * should return LwSciError_Success and cause LwSciStreamEventType_Disconnected event
 * to be queried from the IpcDst block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Block::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::disconnect()}
 */
TEST_F(ipcdst_unit_test, disconnect_Success)
{
    //Initial setup
    using ::testing::Return;
    using ::testing::NiceMock;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, ipcdstPtr->disconnect());

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);
}


/**
 * @testname{ipcdst_unit_test.dstSendSyncDesc_Success}
 * @testcase{21808612}
 * @verify{19791588}
 * @testpurpose{Test positive scenario of IpcDst::dstSendSyncDesc()}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendSyncDesc() API from ipcdst object,
 * with dstIndex of Block::singleConn, sends the syncIndex and
 * sync descriptor(wrapped as LwSciWrap::SyncObj) upstream by calling
 * dstSendSyncDesc() interface of the Pool block, through IpcSrc block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - POOL::dstSendSyncDesc()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_test, dstSendSyncDesc_Success)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Ref;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex;
    LwSciWrap::SyncObj wrapSyncObj;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    for (uint32_t i = 0U; i < totalConsSync; ++i) {
        //Consumer creates sync objects based on producer's requirement
        getSyncObj(syncModule, consSyncObjs[0][i]);

        syncIndex = i;
        wrapSyncObj = { consSyncObjs[0][i] };

        ///////////////////////
        //     Test code     //
        ///////////////////////
        EXPECT_CALL(*poolPtr, dstSendSyncDesc(_, 1U,  Ref(wrapSyncObj))).Times(1);

        ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&ipcdstPtr));
    }
}


/**
 * @testname{ipcdst_unit_sync_setup_test.packSyncDesc_ResourceError}
 * @testcase{22839239}
 * @verify{20050605}
 * @testpurpose{Test negative scenario of IpcDst::packSyncDesc() when
 *  LwSciSyncIpcExportAttrListAndObj() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends the waiter requirements to consumer using
 *      LwSciStreamBlockSyncRequirements().
 *   3. Consumer retrieves the waiter requirements by querying them using
 *      LwSciStreamBlockEventQuery().
 *   4. Consumer sends the LwSciSyncObj count to producer using
 *      LwSciStreamBlockSyncObjCount().
 *   5. Producer queries the consumer sync count using
 *      LwSciStreamBlockEventQuery().
 *   6. Inject fault in LwSciSyncIpcExportAttrListAndObj() to return
 *      LwSciError_ResourceError.
 *
 *   The call of IpcDst::dstSendSyncDesc() API from ipcdst object,
 * with dstIndex of Block::singleConn, valid syncIndex and
 * sync descriptor(wrapped as LwSciWrap::SyncObj), should trigger the error event
 * set to LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, packSyncDesc_ResourceError)
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

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));

    // Consumer sends its sync count to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], prodSyncCount));

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    getSyncObj(syncModule, consSyncObjs[0][0]);

    wrapSyncObj = { consSyncObjs[0][0] };

    test_lwscisync.LwSciSyncIpcExportAttrListAndObj_fail = true;
    ipcdstPtr->dstSendSyncDesc(dstIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

}


/**
 * @testname{ipcdst_unit_sync_setup_test.packSyncDesc_StreamInternalError}
 * @testcase{22839288}
 * @verify{20050605}
 * @testpurpose{Test negative scenario of IpcDst::packSyncDesc() when
 *  IpcSendBuffer::packValAndBlob() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends the waiter requirements to consumer using
 *      LwSciStreamBlockSyncRequirements().
 *   3. Consumer retrieves the waiter requirements by querying them using
 *      LwSciStreamBlockEventQuery().
 *   4. Consumer sends the LwSciSyncObj count to producer using
 *      LwSciStreamBlockSyncObjCount().
 *   5. Producer queries the consumer sync count using
 *      LwSciStreamBlockEventQuery().
 *   6. Inject fault in IpcSendBuffer::packValAndBlob() to return
 *      false.
 *
 *   The call of IpcDst::dstSendSyncDesc() API from ipcdst object,
 * with dstIndex of Block::singleConn, valid syncIndex and
 * sync descriptor(wrapped as LwSciWrap::SyncObj), should trigger the error event
 * set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, packSyncDesc_StreamInternalError)
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

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));

    // Consumer sends its sync count to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], prodSyncCount));

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    getSyncObj(syncModule, consSyncObjs[0][0]);

    wrapSyncObj = { consSyncObjs[0][0] };

    test_ipcsendbuffer.packValAndBlob_fail = true;
    ipcdstPtr->dstSendSyncDesc(dstIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}


/**
 * @testname{ipcdst_unit_test.dstSendPacketElementCount_Success1}
 * @testcase{21808613}
 * @verify{19791591}
 * @testpurpose{Test positive scenario of IpcDst::dstSendPacketElementCount(),
 * where the element count is equal to 0.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API from ipcdst object,
 * with valid element count of 0 and dstIndex of Block::singleConn,
 * should call dstSendPacketElementCount() interface of pool block, through
 * IpcSrc block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketElementCount()}
 */
TEST_F (ipcdst_unit_test, dstSendPacketElementCount_Success1)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(_, elementCount)).Times(1);
    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketElementCount_Success2}
 * @testcase{21808614}
 * @verify{19791591}
 * @testpurpose{Test positive scenario of IpcDst::dstSendPacketElementCount(),
 * where the element count is equal to 1.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API from ipcdst object,
 * with valid element count of 1 and dstIndex of Block::singleConn,
 * should call dstSendPacketElementCount() interface of pool block, through
 * IpcSrc block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{Queue::dstSendPacketElementCount()}
 */
TEST_F (ipcdst_unit_test, dstSendPacketElementCount_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    elementCount=1;

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(_, elementCount)).Times(1);
    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, elementCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketElementCount_Success3}
 * @testcase{21808615}
 * @verify{19791591}
 * @testpurpose{Test positive scenario of IpcDst::dstSendPacketElementCount(),
 * where the element count is equal to MAX_PACKET_ELEMENTS.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API from ipcdst object,
 * with valid element count of MAX_PACKET_ELEMENTS and dstIndex of Block::singleConn,
 * should call dstSendPacketElementCount() interface of pool block, through
 * IpcSrc block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketElementCount()}
 */
TEST_F (ipcdst_unit_test, dstSendPacketElementCount_Success3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(_, MAX_PACKET_ELEMENTS)).Times(1);
    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, MAX_PACKET_ELEMENTS);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcdst_unit_test.dstDisconnect_Success1}
 * @testcase{21808636}
 * @verify{19791606}
 * @testpurpose{Test positive scenario of IpcDst::dstDisconnect()}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstDisconnect() API from IpcDst object,
 * with valid dstIndex of Block::singleConn, disconnects the downstream
 * consumer block and upstream producer block by triggering the
 * LwSciStreamEventType_Disconnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstDisconnect()}
 */
TEST_F(ipcdst_unit_test, dstDisconnect_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcdstPtr->dstDisconnect(Block::singleConn_access);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                producer, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);
}


/**
 * @testname{ipcdst_unit_test.dstSendSyncAttr_Success}
 * @testcase{21808617}
 * @verify{19791582}
 * @testpurpose{Test positive scenario of IpcDst::dstSendSyncAttr()}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendSyncAttr() API from ipcdst object,
 * with valid sync attributes (synchronousOnly flag and LwSciWrap::SyncAttr)
 * and dstIndex, should call dstSendSyncAttr() interface of pool block,
 * through IpcSrc block, to send the sync attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendSyncAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncAttr()}
 */
TEST_F(ipcdst_unit_test, dstSendSyncAttr_Success)
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
    EXPECT_CALL(*ipcdstPtr, setErrorEvent_imp(_, _)).Times(0);

    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&ipcdstPtr));
}

/**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncAttr_IlwalidState}
 * @testcase{22839290}
 * @verify{19791582}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncAttr() when
 * LwSciSyncObj waiter requirements have already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Call IpcDst::dstSendSyncAttr() to send the waiter requirements.
 *
 *   The call of IpcDst::dstSendSyncAttr() API from ipcdst object,
 * with valid sync attributes (synchronousOnly flag and LwSciWrap::SyncAttr)
 * and dstIndex, should trigger the error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncAttr_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = consSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);;

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}


/**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncAttr_StreamBadDstIndex}
 * @testcase{22839292}
 * @verify{19791582}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncAttr() when
 * dstIndex is invalid.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendSyncAttr() API from ipcdst object,
 * with valid sync attributes (synchronousOnly flag and LwSciWrap::SyncAttr)
 * and invalid dstIndex, should trigger the error event set to LwSciError_StreamBadDstIndex.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncAttr_StreamBadDstIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t dstIndex = ILWALID_CONN_IDX;
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

    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);;

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendPacketAttr_Success1}
 * @testcase{21808618}
 * @verify{19791594}
 * @testpurpose{Test positive scenario of IpcDst::dstSendPacketAttr(),
 * where the elemSyncMode is equal to LwSciStreamElementMode_Asynchronous.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Consumer sends element count by ilwoking LwSciStreamBlockPacketElementCount().
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Asynchronous and LwSciWrap::BufAttr,
 * should call dstSendPacketAttr() interface of pool block, through IpcSrc
 * block, to send the consumer packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketAttr_Success1)
{
    // Initial setup
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    using ::testing::Ref;

    LwSciStreamEvent event;
    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    EXPECT_EQ(LwSciError_Success,
     LwSciStreamBlockPacketElementCount(consumer[0], 2U));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));

    // Pool sets packet requirements
    for (uint32_t i = 0U; i < elementCount; i++) {

        BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
        LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

        ///////////////////////
        //     Test code     //
        ///////////////////////
        EXPECT_CALL(*poolPtr, dstSendPacketAttr(_, i, i, LwSciStreamElementMode_Asynchronous,
              Ref(wrapBufAttr)))
             .Times(1)
             .WillRepeatedly(Return());

        ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, i, i,
                                   LwSciStreamElementMode_Asynchronous,
                                   wrapBufAttr);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
    }
}

/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendPacketAttr_Success2}
 * @testcase{21808619}
 * @verify{19791594}
 * @testpurpose{Test positive scenario of IpcDst::dstSendPacketAttr(),
 * where the elemSyncMode is equal to LwSciStreamElementMode_Immediate.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Consumer sends element count by ilwoking LwSciStreamBlockPacketElementCount().
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Immediate and LwSciWrap::BufAttr,
 * should call dstSendPacketAttr() interface of pool block, through IpcSrc
 * block, to send the consumer packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketAttr_Success2)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    using ::testing::Ref;

    LwSciStreamEvent event;
    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    EXPECT_EQ(LwSciError_Success,
     LwSciStreamBlockPacketElementCount(consumer[0], 2U));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));

    // Pool sets packet requirements
    for (uint32_t i = 0U; i < elementCount; i++) {

        BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
        LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

        ///////////////////////
        //     Test code     //
        ///////////////////////
        EXPECT_CALL(*poolPtr, dstSendPacketAttr(_, i, i, LwSciStreamElementMode_Immediate,
              Ref(wrapBufAttr)))
             .Times(0)
             .WillRepeatedly(Return());

        ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, i, i,
                                   LwSciStreamElementMode_Immediate,
                                   wrapBufAttr);

        EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
    }
}


/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendPacketAttr_Success3}
 * @testcase{22839294}
 * @verify{19791594}
 * @testpurpose{Test positive scenario of IpcDst::dstSendPacketAttr(),
 * where the elemSyncMode is equal to LwSciStreamElementMode_Asynchronous.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends the supported packet element count and element information
 *      to pool.
 *   3. Consumer sends the supported packet element count to pool and pool queries
 *      the same using LwSciStreamBlockEventQuery().
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Asynchronous and LwSciWrap::BufAttr,
 * should call dstSendPacketAttr() interface of pool block, through IpcSrc
 * block, to send the consumer packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketAttr_Success3)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    using ::testing::Ref;

    uint32_t elemIndex;
    uint32_t elemType;
    LwSciStreamEvent event;
    LwSciStreamElementMode elemSyncMode = LwSciStreamElementMode_Asynchronous;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Producer sends its supported packet attributes to pool
    prodSendPacketAttr();

    // consumer sets the number of elements in a packet
    EXPECT_EQ(LwSciError_Success,
     LwSciStreamBlockPacketElementCount(consumer[0], 2U));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_CALL(*poolPtr, dstSendPacketAttr(_, _, _, _, _))
        .Times(1);

    test_lwscibuf.LwSciBufAttrListIpcExportunreconciled_blobData_null = true;
    for (uint32_t i = 0U; i < 1; ++i) {
        elemIndex = i;
        elemType = i;

        ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, elemIndex, elemType,
                                        elemSyncMode, wrapBufAttr);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{ipcdst_unit_test.dstSendPacketStatus_Success}
 * @testcase{21808620}
 * @verify{19791597}
 * @testpurpose{Test positive scenario of IpcDst::dstSendPacketStatus()}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Producer and Consumer send the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3. Pool sends reconciled PacketElementCount and PacketAttr back to producer and consumer
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr() respectively.
 *   4. Pool block creates the packet using LwSciStreamPoolPacketCreate() and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer().
 *   5. Consumer queries the LwSciStreamEventType_PacketCreate event through
 *      LwSciStreamBlockEventQuery and gets packetHandle.
 *
 *   The call of IpcDst::dstSendPacketStatus() API from IpcDst object,
 * with dstIndex as Block::singleConn, handle as reference to newly created packet
 * and packetStatus is LwSciError_Success, notifies the producer of the packet ready event
 * by calling dstSendPacketStatus() interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Following LwSciStream API calls are replaced with mocks:
 *   - POOL::dstReusePacket()}
 * @verifyFunction{IpcDst::dstSendPacketStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketStatus_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendPacketStatus_IlwalidState}
 * @testcase{22839295}
 * @verify{19791597}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketStatus() when
 * acceptance status for the same LwSciStreamPacket has already been scheduled
 * to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up the packet attributes for streaming.
 *   3. Create a packet using LwSciStreamPoolPacketCreate() and make sure
 *      consumer queries for the LwSciStreamEventType_PacketCreate event.
 *   4. Call IpcDst::dstSendPacketStatus() with valid handle to send the packet status.
 *
 *   The call of IpcDst::dstSendPacketStatus() API from IpcDst object,
 * with dstIndex as Block::singleConn, same handle used in step:4 and packetStatus is
 * LwSciError_Success, should trigger error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Following LwSciStream API calls are replaced with mocks:
 * @verifyFunction{IpcDst::dstSendPacketStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketStatus_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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

    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);


}


/**
 * @testname{ipcdst_unit_buf_setup_test.packStatus_StreamInternalError}
 * @testcase{22839296}
 * @verify{20050611}
 * @testpurpose{Test negative scenario of IpcDst::packStatus() when
 * packing status failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up the packet attributes for streaming.
 *   3. Create a packet using LwSciStreamPoolPacketCreate() and make sure
 *      consumer queries for the LwSciStreamEventType_PacketCreate event.
 *   4. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcDst::dstSendPacketStatus() API from IpcDst object,
 * with dstIndex as Block::singleConn, handle and packetStatus is
 * LwSciError_Success, should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Following LwSciStream API calls are replaced with mocks:
 * @verifyFunction{IpcDst::dstSendPacketStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, packStatus_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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

    test_ipcsendbuffer.packVal_fail = true;
    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);


}

/**
 * @testname{IpcDst_unit_test.dstReusePacket_Success1}
 * @testcase{21808621}
 * @verify{19791603}
 * @testpurpose{Test positive scenario of IpcDst::dstReusePacket(), where
 * the queried packet in the pool is available for reuse.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, queue and consumer blocks.
 *   2. Set up synchronization and buffer resources, with packet count greater than 0.
 *   3. Producer gets the next available packet from Pool using LwSciStreamProducerPacketGet().
 *   4. Producer gets packet and inserts the data into the stream
 *      through LwSciStreamProducerPacketPresent().
 *   5. Pool sends the packet downstream to consumer and triggers packet ready event to consumer.
 *   6. Consumer queries for packet ready event by calling LwSciStreamBlockEventQuery().
 *   7. Consumer acquires the packet through LwSciStreamConsumerPacketAcquire().
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with dstIndex as Block::singleConn, handle as consumer released packet handle
 * and postfences is non-NULL FenceArray. Notifies the producer of the packet availability for reuse
 * by calling the dstReusePacket() interface of the Pool block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Following LwSciStream API calls are replaced with mocks:
 *   - POOL::dstReusePacket()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_packet_stream_test, dstReusePacket_Success1)
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
    EXPECT_CALL(*poolPtr, dstReusePacket(_, _, _)).Times(1);
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
    free(fences);
}

/**
 * @testname{ipcdst_unit_test.disconnect_Success1}
 * @testcase{21808622}
 * @verify{19791579}
 * @testpurpose{Test positive scenario of IpcDst::disconnect()}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::disconnect() API from IpcDst object, should return
 * LwSciError_Success and expected to call POOL::dstdisconnect() API from  POOL object.
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstdisconnect()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::disconnect()}
 */
TEST_F(ipcdst_unit_test, disconnect_Success1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    // Initialise Ipc channel
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
 * @testname{ipcdst_unit_test.recvSyncAttr_Success}
 * @testcase{21808623}
 * @verify{20050617}
 * @testpurpose{Test positive scenario of IpcDst::recvSyncAttr()}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *  IpcDst::recvSyncAttr() API call triggered, when IpcSrc::srcSendSyncAttr()
 *  API is called with valid sync attributes (synchronousOnly flag and LwSciWrap::SyncAttr)
 *  and srcIndex, should call srcSendSyncAttr() interface of queue block to
 *  send the waiter requirements.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcSendSyncAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncAttr_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

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
    EXPECT_CALL(*queuePtr[0], srcSendSyncAttr(_, _, _)).Times(1);

    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcdst_unit_test.recvSyncCount_Success}
 * @testcase{21808626}
 * @verify{20050620}
 * @testpurpose{Test positive scenario of IpcDst::recvSyncCount(), where
 * dstSendSyncCount() is called with sync count.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *  IpcDst::recvSyncCount() API call triggered, when IpcSrc::srcSendSyncCount()
 *  API is called with valid sync count and srcIndex of Block::singleConn,
 *  should call srcSendSyncCount() interface of queue block to
 *  send count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - QUEUE::srcSendSyncCount()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncCount()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncCount_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
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
    EXPECT_CALL(*queuePtr[0], srcSendSyncCount(_, _)).Times(1);

    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncCount_StreamInternalError}
 * @testcase{22839298}
 * @verify{20050620}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncCount(), when
 * unpacking LwSciSyncObj count failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *  IpcDst::recvSyncCount() API call triggered, when IpcSrc::srcSendSyncCount()
 *  API is called with valid sync count and srcIndex of Block::singleConn,
 *  should trigger error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncCount()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncCount_StreamInternalError)
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

    test_ipcrecvbuffer.unpackVal_fail = true;
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncCount_IlwalidState}
 * @testcase{22839300}
 * @verify{20050620}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncCount(), when
 * count was already received.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Call IpcSrc::srcSendSyncCount() with valid sync count and srcIndex of
 *      Block::singleConn.
 *
 *  IpcDst::recvSyncCount() API call triggered, when IpcSrc::srcSendSyncCount()
 *  API is called with valid sync count and srcIndex of Block::singleConn,
 *  should trigger error event set to LwSciError_IlwalidState in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncCount()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncCount_IlwalidState)
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
    ipcsrcPtr->srcSendSyncCount(srcIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncCount, event.type);

    test_trackcount.set_fail_IlwalidState = true;

    ipcsrcPtr->srcSendSyncCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcdst_unit_test.recvSyncDesc_Success}
 * @testcase{21808628}
 * @verify{20050623}
 * @testpurpose{Test positive scenario of IpcDst::recvSyncDesc()}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *  IpcDst::recvSyncDesc() API call triggered, when IpcSrc::srcSendSyncDesc()
 *  API is called with valid srcIndex of Block::singleConn, syncIndex and
 *  sync descriptor(wrapped as LwSciWrap::SyncObj), should call srcSendSyncDesc()
 *  interface of queue block to send LwSciSyncObj(s).}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - QUEUE::srcSendSyncDesc()
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncDesc_Success)
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
    EXPECT_CALL(*queuePtr[0], srcSendSyncDesc(_, _, _)).Times(prodSyncCount);

    for (uint32_t i = 0U; i < prodSyncCount; ++i) {
        getSyncObj(syncModule, prodSyncObjs[i]);
        syncIndex = i;
        wrapSyncObj = { prodSyncObjs[i] };

        ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncDesc_StreamInternalError}
 * @testcase{22839302}
 * @verify{20050623}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncDesc() when
 * IpcRecvBuffer::unpackValAndBlob() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Consumer sends the waiter requirements by ilwoking LwSciStreamBlockSyncRequirements()
 *   3. Query the consumer waiter requirements by using LwSciStreamBlockEventQuery().
 *   4. Producer send the supported sync objects count using LwSciStreamBlockSyncObjCount().
 *   5. Inject fault in IpcRecvBuffer::unpackValAndBlob() to return false.
 *
 *  IpcDst::recvSyncDesc() API call triggered, when IpcSrc::srcSendSyncDesc()
 *  API is called with valid srcIndex of Block::singleConn, syncIndex and
 *  sync descriptor(wrapped as LwSciWrap::SyncObj), should trigger error event
 *  set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncDesc_StreamInternalError)
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
    for (uint32_t i = 0U; i < 1U; ++i) {
        getSyncObj(syncModule, prodSyncObjs[i]);
        syncIndex = i;
        wrapSyncObj = { prodSyncObjs[i] };

        test_ipcrecvbuffer.unpackValAndBlob_fail = true;
        // To call IpcSrc::recvSyncDesc()
        ipcsrcPtr->srcSendSyncDesc(srcIndex, syncIndex, wrapSyncObj);

        EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcdst, EVENT_QUERY_TIMEOUT, &event));
        EXPECT_EQ(LwSciStreamEventType_Error, event.type);
        EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    }
}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncDesc_ResourceError}
 * @testcase{22839303}
 * @verify{20050623}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncDesc() when
 * LwSciSyncIpcImportAttrListAndObj() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Consumer sends the waiter requirements by ilwoking LwSciStreamBlockSyncRequirements()
 *   3. Query the consumer waiter requirements by using LwSciStreamBlockEventQuery().
 *   4. Producer send the supported sync objects count using LwSciStreamBlockSyncObjCount().
 *   5. Inject fault in LwSciSyncIpcImportAttrListAndObj() to return LwSciError_ResourceError.
 *
 *  IpcDst::recvSyncDesc() API call triggered, when IpcSrc::srcSendSyncDesc()
 *  API is called with valid srcIndex of Block::singleConn, syncIndex and
 *  sync descriptor(wrapped as LwSciWrap::SyncObj), should trigger error event
 *  set to LwSciError_ResourceError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncDesc_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
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
    getSyncObj(syncModule, prodSyncObjs[0]);
    wrapSyncObj = { prodSyncObjs[0] };

    test_lwscisync.LwSciSyncIpcImportAttrListAndObj_fail = true;
    ipcsrcPtr->srcSendSyncDesc(srcIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncDesc_IlwalidState}
 * @testcase{22839304}
 * @verify{20050623}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncDesc() when
 * LwSciSyncObj for the SyncIndex already forwarded downstream.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Consumer sends the waiter requirements by ilwoking LwSciStreamBlockSyncRequirements()
 *   3. Query the consumer waiter requirements by using LwSciStreamBlockEventQuery().
 *   4. Producer send the supported sync objects count using LwSciStreamBlockSyncObjCount().
 *   5. Call IpcSrc::srcSendSyncDesc() with valid srcIndex of Block::singleConn, syncIndex and
 *      sync descriptor(wrapped as LwSciWrap::SyncObj).
 *
 *  IpcDst::recvSyncDesc() API call triggered, when IpcSrc::srcSendSyncDesc()
 *  API is called with valid srcIndex of Block::singleConn, same syncIndex as in
 *  step-5 and sync descriptor(wrapped as LwSciWrap::SyncObj), should trigger error
 *  event set to LwSciError_IlwalidState in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncDesc_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
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

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncCount, event.type);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    getSyncObj(syncModule, prodSyncObjs[0]);
    wrapSyncObj = { prodSyncObjs[0] };

    ipcsrcPtr->srcSendSyncDesc(srcIndex, 0U, wrapSyncObj);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncDesc, event.type);

    test_trackArray.performAction_fail = true;
    ipcsrcPtr->srcSendSyncDesc(srcIndex, 0U, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
        ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

}

/**
 * @testname{ipcdst_unit_test.recvElemCount_Success}
 * @testcase{21808629}
 * @verify{20050626}
 * @testpurpose{Test positive scenario of IpcDst::recvElemCount(),
 * with the valid element count.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *  IpcDst::recvElemCount() API call triggered, when IpcSrc::srcSendPacketElementCount()
 *  API is called with valid element count and srcIndex of Block::singleConn,
 *  should call srcSendPacketElementCount() interface of queue block to send
 *  packet element count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcSendPacketElementCount()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvElemCount()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemCount_Success)
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

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcSendPacketElementCount(_, _)).Times(1);

    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvElemCount_StreamInternalError2}
 * @testcase{22839306}
 * @verify{20050626}
 * @testpurpose{Test negative scenario of IpcDst::recvElemCount() when unpacking
 * consolidated packet element count failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   3. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   4. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   5. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *  IpcDst::recvElemCount() API call triggered, when IpcSrc::srcSendPacketElementCount()
 *  API is called with valid element count and srcIndex of Block::singleConn,
 *  should trigger error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvElemCount()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemCount_StreamInternalError2)
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

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcrecvbuffer.unpackVal_fail = true;
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvElemCount_IlwalidState}
 * @testcase{22839307}
 * @verify{20050626}
 * @testpurpose{Test negative scenario of IpcDst::recvElemCount() when
 * consolidated packet element count was already received.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   3. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   4. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   5. Call IpcSrc::srcSendPacketElementCount() with valid element count and
 *      srcIndex of Block::singleConn.
 *
 *  IpcDst::recvElemCount() API call triggered, when IpcSrc::srcSendPacketElementCount()
 *  API is called with valid element count and srcIndex of Block::singleConn,
 *  should trigger error event set to LwSciError_IlwalidState in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvElemCount()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemCount_IlwalidState)
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
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketElementCount, event.type);
    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_trackcount.set_fail_IlwalidState = true;
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}


/**
 * @testname{ipcdst_unit_test.recvElemAttr_Success}
 * @testcase{21808630}
 * @verify{20050629}
 * @testpurpose{Test positive scenario of IpcDst::recvElemAttr().}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   3. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   4. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   5. Pool sends the consolidated packet element count using LwSciStreamBlockPacketElementCount().
 *
 *  IpcDst::recvElemAttr() API call triggered, when IpcSrc::srcSendPacketAttr()
 *  API is called with valid srcIndex of Block::singleConn, elemIndex,  elemType,
 *  elemSyncMode of LwSciStreamElementMode_Asynchronous and LwSciWrap::BufAttr,
 *  should call srcSendPacketAttr() interface of queue block to send the packet attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcSendPacketAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvElemAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemAttr_Success)
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
 * @testname{ipcdst_unit_buf_setup_test.recvElemAttr_StreamInternalError}
 * @testcase{22839311}
 * @verify{20050629}
 * @testpurpose{Test negative scenario of IpcDst::recvElemAttr() when
 * IpcRecvBuffer::unpackMsgElemAttr() failed. }
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   3. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   4. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   5. Pool sends the consolidated packet element count using LwSciStreamBlockPacketElementCount().
 *   6. inject fault in IpcRecvBuffer::unpackMsgElemAttr() to return false.
 *
 *  IpcDst::recvElemAttr() API call triggered, when IpcSrc::srcSendPacketAttr()
 *  API is called with valid srcIndex of Block::singleConn, elemIndex,  elemType,
 *  elemSyncMode of LwSciStreamElementMode_Asynchronous and LwSciWrap::BufAttr,
 *  should trigger the error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvElemAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemAttr_StreamInternalError)
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

    test_ipcrecvbuffer.unpackMsgElemAttr_fail = true;
    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvElemAttr_ResourceError}
 * @testcase{22839312}
 * @verify{20050629}
 * @testpurpose{Test negative scenario of IpcDst::recvElemAttr() when
 * LwSciBufAttrListIpcImportReconciled() failed. }
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   3. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   4. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   5. Pool sends the consolidated packet element count using LwSciStreamBlockPacketElementCount().
 *   6. inject fault in LwSciBufAttrListIpcImportReconciled() to return
 *      LwSciError_ResourceError.
 *
 *  IpcDst::recvElemAttr() API call triggered, when IpcSrc::srcSendPacketAttr()
 *  API is called with valid srcIndex of Block::singleConn, elemIndex,  elemType,
 *  elemSyncMode of LwSciStreamElementMode_Asynchronous and LwSciWrap::BufAttr,
 *  should trigger the error event set to LwSciError_ResourceError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvElemAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemAttr_ResourceError)
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

    test_lwscibuf.LwSciBufAttrListIpcImportReconciled_fail = true;

    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvElemAttr_IlwalidState}
 * @testcase{22839314}
 * @verify{20050629}
 * @testpurpose{Test negative scenario of IpcDst::recvElemAttr() when
 * element information for the elemIndex already forwarded downstream. }
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   3. Consumer sends its packet element count and packet element information to pool
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr().
 *   4. Pool receives both producer's and consumer's packet element count and packet element
 *      information by querying through LwSciStreamBlockEventQuery().
 *   5. Pool sends the consolidated packet element count using LwSciStreamBlockPacketElementCount().
 *   6. Call IpcSrc::srcSendPacketAttr() with valid srcIndex of Block::singleConn,
 *      elemIndex, elemType, elemSyncMode of LwSciStreamElementMode_Asynchronous
 *      and LwSciWrap::BufAttr.
 *
 *  IpcDst::recvElemAttr() API call triggered, when IpcSrc::srcSendPacketAttr()
 *  API is called with valid srcIndex of Block::singleConn, same elemIndex as in
 *  step-6, elemType, elemSyncMode of LwSciStreamElementMode_Asynchronous and
 *  LwSciWrap::BufAttr, should trigger the error event set to
 *  LwSciError_IlwalidState in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvElemAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemAttr_IlwalidState)
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
    EXPECT_EQ(LwSciStreamEventType_PacketElementCount, event.type);

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    wrapElemBufAttr = rawBufAttrList;

    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_PacketAttr, event.type);

    test_trackArray.performAction_fail = true;

    ipcsrcPtr->srcSendPacketAttr(srcIndex, 0U, 0U,
                                        elemSyncMode, wrapElemBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketCreate_Success}
 * @testcase{21808631}
 * @verify{20050632}
 * @testpurpose{Test positive scenario of IpcDst::recvPacketCreate().}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *  IpcDst::recvPacketCreate() API call triggered, when IpcSrc::srcCreatePacket()
 *  API is called with valid srcIndex and LwSciStreamPacket handle,
 *  should call srcCreatePacket() interface of queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcCreatePacket()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketCreate()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketCreate_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
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
 * @testname{ipcdst_unit_buf_setup_test.recvPacketBuffer_Success}
 * @testcase{21808632}
 * @verify{20050635}
 * @testpurpose{Test positive scenario of IpcDst::recvPacketBuffer().}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *  IpcDst::recvPacketBuffer() API call triggered, when IpcSrc::srcInsertBuffer()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, elemIndex and
 *  LwSciWrap::BufObj, should call srcInsertBuffer() interface of queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcInsertBuffer()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketBuffer()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketBuffer_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;

    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*queuePtr[0], srcInsertBuffer(_, _, _, _)).Times(consolidatedElementCount);

    for (uint32_t i = 0U; i < consolidatedElementCount; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        ipcsrcPtr->srcInsertBuffer(srcIndex, handle, elemIndex, wrapElemBufObj);
    }

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketBuffer_StreamInternalError1}
 * @testcase{22839316}
 * @verify{20050635}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketBuffer() when
 * IpcRecvBuffer::unpackMsgPacketBuffer() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *   4. Inject fault in IpcRecvBuffer::unpackMsgPacketBuffer() to return false.
 *
 *  IpcDst::recvPacketBuffer() API call triggered, when IpcSrc::srcInsertBuffer()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, elemIndex and
 *  LwSciWrap::BufObj, should trigger error event set to
 *  LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketBuffer()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketBuffer_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    for (uint32_t i = 0U; i < 1U; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        test_ipcrecvbuffer.unpackMsgPacketBuffer_fail = true;
        ipcsrcPtr->srcInsertBuffer(srcIndex, handle, elemIndex, wrapElemBufObj);
    }

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketBuffer_StreamInternalError2}
 * @testcase{22839319}
 * @verify{20050635}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketBuffer() when
 * packet handle is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *   4. Inject fault in Block::pktFindByHandle() to return NULL.
 *
 *  IpcDst::recvPacketBuffer() API call triggered, when IpcSrc::srcInsertBuffer()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, elemIndex and
 *  LwSciWrap::BufObj, should trigger error event set to
 *  LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketBuffer()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketBuffer_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    for (uint32_t i = 0U; i < 1U; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        ipcsrcPtr->srcInsertBuffer(srcIndex, handle, elemIndex, wrapElemBufObj);
        test_block.pktFindByHandle_fail = true;
    }

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);


}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketBuffer_ResourceError}
 * @testcase{22839321}
 * @verify{20050635}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketBuffer() when
 * LwSciBufObjIpcImport() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *   4. Inject fault in LwSciBufObjIpcImport() to return LwSciError_ResourceError.
 *
 *  IpcDst::recvPacketBuffer() API call triggered, when IpcSrc::srcInsertBuffer()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, elemIndex and
 *  LwSciWrap::BufObj, should trigger error event set to
 *  LwSciError_ResourceError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketBuffer()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketBuffer_ResourceError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    for (uint32_t i = 0U; i < 1U; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        test_lwscibuf.LwSciBufObjIpcImport_fail = true;
        ipcsrcPtr->srcInsertBuffer(srcIndex, handle, elemIndex, wrapElemBufObj);
    }

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketBuffer_IlwalidState}
 * @testcase{22839331}
 * @verify{20050635}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketBuffer() when
 * LwSciBufObj for the packet element is already received.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *   4. Inject fault in Packet::bufferDefineAction() to return LwSciError_IlwalidState.
 *
 *  IpcDst::recvPacketBuffer() API call triggered, when IpcSrc::srcInsertBuffer()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, elemIndex and
 *  LwSciWrap::BufObj, should trigger error event set to
 *  LwSciError_IlwalidState in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketBuffer()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketBuffer_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    for (uint32_t i = 0U; i < 1U; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        test_packet.BufferAction_fail_IlwalidState = true;
        ipcsrcPtr->srcInsertBuffer(srcIndex, handle, elemIndex, wrapElemBufObj);
    }

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);

}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketBuffer_BadParameter1}
 * @testcase{22839333}
 * @verify{20050635}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketBuffer() when
 * sending packet element information downstream failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *   4. Inject fault in Packet::bufferDefineAction() to return LwSciError_BadParameter.
 *
 *  IpcDst::recvPacketBuffer() API call triggered, when IpcSrc::srcInsertBuffer()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, elemIndex and
 *  LwSciWrap::BufObj, should trigger error event set to
 *  LwSciError_BadParameter in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketBuffer()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketBuffer_BadParameter1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    for (uint32_t i = 0U; i < 1U; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        test_packet.BufferAction_fail_BadParameter = true;
        ipcsrcPtr->srcInsertBuffer(srcIndex, handle, elemIndex, wrapElemBufObj);
    }

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);

}


/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketBuffer_BadParameter2}
 * @testcase{22839342}
 * @verify{20050635}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketBuffer() when
 * unpacked element index is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *   4. Inject fault in Packet::bufferDefineAction() to return LwSciError_BadParameter.
 *
 *  IpcDst::recvPacketBuffer() API call triggered, when IpcSrc::srcInsertBuffer()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, elemIndex and
 *  LwSciWrap::BufObj, should trigger error event set to
 *  LwSciError_BadParameter in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketBuffer()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketBuffer_BadParameter2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex = 0;
    LwSciWrap::BufObj wrapElemBufObj;
    LwSciStreamEvent event;

    LwSciBufObj poolElementBuf[MAX_ELEMENT_PER_PACKET];

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    for (uint32_t i = 0U; i < 1U; ++i) {
        makeRawBuffer(rawBufAttrList, poolElementBuf[i]);
        wrapElemBufObj = poolElementBuf[i];
        elemIndex = i;

        test_packet.BufferAction_fail_BadParameter = true;
        ipcsrcPtr->srcInsertBuffer(srcIndex, handle, elemIndex, wrapElemBufObj);
    }

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);

}
/**
 * @testname{ipcdst_packet_stream_test.recvPacketDelete_Success}
 * @testcase{21808634}
 * @verify{20050638}
 * @testpurpose{Test positive scenario of IpcDst::recvPacketDelete().}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *
 *  IpcDst::recvPacketDelete() API call triggered, when IpcSrc::srcDeletePacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, should call
 *  srcDeletePacket() interface of queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcDeletePacket()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketDelete()}
 */
TEST_F(ipcdst_packet_stream_test, recvPacketDelete_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
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

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
 * @testname{ipcdst_packet_stream_test.recvPacketDelete_StreamInternalError}
 * @testcase{22839370}
 * @verify{20050638}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketDelete() when unpacking
 * handle failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup stream packet attributes.
 *   3. Create a packet using LwSciStreamPoolPacketCreate().
 *
 *  IpcDst::recvPacketDelete() API call triggered, when IpcSrc::srcDeletePacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle, should trigger
 *  error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPacketDelete()}
 */
TEST_F(ipcdst_packet_stream_test, recvPacketDelete_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
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

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            consumer[0], EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_ipcrecvbuffer.unpackVal_fail = true;
    ipcsrcPtr->srcDeletePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_unit_sync_setup_test.processReadMsg_StreamInternalError2}
 * @testcase{22839371}
 * @verify{19840539}
 * @testpurpose{Test negative scenario of IpcDst::processReadMsg() when
 * IpcRecvBuffer::unpackBegin() failed during unpacking waiter requirements.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackBegin() to return false.
 *
 *   The call of IpcSrc::srcSendSyncAttr() with parameters(Sync flag and attributes),
 *  should trigger error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processReadMsg()}
 */
TEST_F(ipcdst_unit_sync_setup_test, processReadMsg_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = true;
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
    test_ipcrecvbuffer.unpackBegin_fail = true;
    // To call IpcSrc::recvSyncAttr()
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_unit_sync_setup_test.processReadMsg_StreamInternalError3}
 * @testcase{22839372}
 * @verify{19840539}
 * @testpurpose{Test negative scenario of IpcDst::processReadMsg() when
 * IpcRecvBuffer::unpackVal() failed during unpacking waiter requirements.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *   The call of IpcSrc::srcSendSyncAttr() with parameters(Sync flag and attributes),
 *  should trigger error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processReadMsg()}
 */
TEST_F(ipcdst_unit_sync_setup_test, processReadMsg_StreamInternalError3)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = true;
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
    test_ipcrecvbuffer.processMsg_unpack_fail = true;
    // To call IpcSrc::recvSyncAttr()
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_packet_stream_test.recvPayload_Success}
 * @testcase{21808635}
 * @verify{20050641}
 * @testpurpose{Test positive scenario of IpcDst::recvPayload().}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Complete the Sync and buffer setup.
 *   3. Producer gets the available packer for reuse after querying the
 *      LwSciStreamEventType_PacketReady event.
 *
 *  IpcDst::recvPayload() API call triggered, when IpcSrc::srcSendPacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle and FenceArray,
 *  should call srcSendPacket() interface of queue block.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcSendPacket()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPayload()}
 */
TEST_F(ipcdst_packet_stream_test, recvPayload_Success)
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
    mailboxPtr = std::static_pointer_cast<LwSciStream::Mailbox>(queuePtr[0]);
    EXPECT_CALL(*mailboxPtr, srcSendPacket(_, _, _)).Times(1);

    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&mailboxPtr));
    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.recvPayload_StreamInternalError1}
 * @testcase{22839374}
 * @verify{20050641}
 * @testpurpose{Test negative scenario of IpcDst::recvPayload() when unpacking
 * failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Complete the Sync and buffer setup.
 *   3. Producer gets the available packer for reuse after querying the
 *      LwSciStreamEventType_PacketReady event.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *  IpcDst::recvPayload() API call triggered, when IpcSrc::srcSendPacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle and FenceArray,
 *  should trigger the error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPayload()}
 */
TEST_F(ipcdst_packet_stream_test, recvPayload_StreamInternalError1)
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
    test_ipcrecvbuffer.unpackVal_fail = true;

    // To call IpcSrc::dstReusePacket()
    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.recvPayload_StreamInternalError2}
 * @testcase{22839375}
 * @verify{20050641}
 * @testpurpose{Test negative scenario of IpcDst::recvPayload() when packet
 * handle is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Complete the Sync and buffer setup.
 *   3. Producer gets the available packer for reuse after querying the
 *      LwSciStreamEventType_PacketReady event.
 *   4. Inject fault in Block::pktFindByHandle() to return NULL.
 *
 *  IpcDst::recvPayload() API call triggered, when IpcSrc::srcSendPacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle and FenceArray,
 *  should trigger the error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPayload()}
 */
TEST_F(ipcdst_packet_stream_test, recvPayload_StreamInternalError2)
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

    test_block.pktFindByHandle_fail = true;

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.recvPayload_StreamInternalError3}
 * @testcase{22839376}
 * @verify{20050641}
 * @testpurpose{Test negative scenario of IpcDst::recvPayload() when
 * Packet::locationUpdate() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Complete the Sync and buffer setup.
 *   3. Producer gets the available packer for reuse after querying the
 *      LwSciStreamEventType_PacketReady event.
 *   4. Inject fault in Packet::locationUpdate() to return false.
 *
 *  IpcDst::recvPayload() API call triggered, when IpcSrc::srcSendPacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle and FenceArray,
 *  should trigger the error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPayload()}
 */
TEST_F(ipcdst_packet_stream_test, recvPayload_StreamInternalError3)
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

    test_packet.locationUpdate_fail = true;
    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.recvPayload_StreamInternalError4}
 * @testcase{22839377}
 * @verify{20050641}
 * @testpurpose{Test negative scenario of IpcDst::recvPayload() when
 * IpcRecvBuffer::unpackFenceExport() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Complete the Sync and buffer setup.
 *   3. Producer gets the available packer for reuse after querying the
 *      LwSciStreamEventType_PacketReady event.
 *   4. Inject fault in IpcRecvBuffer::unpackFenceExport() to return false.
 *
 *  IpcDst::recvPayload() API call triggered, when IpcSrc::srcSendPacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle and FenceArray,
 *  should trigger the error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPayload()}
 */
TEST_F(ipcdst_packet_stream_test, recvPayload_StreamInternalError4)
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

    test_ipcrecvbuffer.unpackFenceExport_fail = true;
    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.recvPayload_ResourceError}
 * @testcase{22839378}
 * @verify{20050641}
 * @testpurpose{Test negative scenario of IpcDst::recvPayload() when
 * LwSciSyncIpcImportFence() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Complete the Sync and buffer setup.
 *   3. Producer gets the available packer for reuse after querying the
 *      LwSciStreamEventType_PacketReady event.
 *   4. Inject fault in LwSciSyncIpcImportFence() to return LwSciError_ResourceError.
 *
 *  IpcDst::recvPayload() API call triggered, when IpcSrc::srcSendPacket()
 *  API is called with valid srcIndex, LwSciStreamPacket handle and FenceArray,
 *  should trigger the error event set to LwSciError_ResourceError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvPayload()}
 */
TEST_F(ipcdst_packet_stream_test, recvPayload_ResourceError)
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

    test_lwscisync.LwSciSyncIpcImportFence_fail = true;
    ipcsrcPtr->srcSendPacket(srcIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);

    free(fences);
}

/**
 * @testname{ipcdst_unit_test.dstSendSyncAttr_StreamNotConnected}
 * @testcase{21808637}
 * @verify{19791582}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncAttr(), where
 * dstSendSyncAttr() is called when the stream is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Create the producer, pool, ipcSrc, queue and consumer blocks.
 *   2. IpcDst is created and not connected to any of the blocks.
 *
 *   The call of IpcDst::dstSendSyncAttr() API from ipcdst object,
 * should result in LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncAttr()}
 */
TEST_F(ipcdst_unit_test, dstSendSyncAttr_StreamNotConnected)
{
    // Initial setup
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = consSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendSyncDesc_StreamNotConnected}
 * @testcase{21808639}
 * @verify{19791588}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncDesc(), where
 * dstSendSyncDesc() is called when the stream is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Create the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. ipcdst block is created and not connected to any other blocks.
 *
 *   The call of IpcDst::dstSendSyncDesc() API from IpcDst object,
 * should result in LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_test, dstSendSyncDesc_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex = 0U;
    LwSciWrap::SyncObj wrapSyncObj;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    //Create LwSciStream blocks
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    //Consumer creates sync object for the first index based on producer's requirement
    getSyncObj(syncModule, consSyncObjs[0][0]);

    wrapSyncObj = { consSyncObjs[0][0] };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{pcdst_unit_test.dstSendPacketElementCount_BadParameter}
 * @testcase{21808640}
 * @verify{19791591}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketElementCount(), where
 *   dstSendPacketElementCount is ilwoked with element count greater than MAX_PACKET_ELEMENTS.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API from ipcdst object,
 * with element count MAX_PACKET_ELEMENTS + 1 and dstIndex of Block::singleConn,
 * should result in LwSciError_BadParameter event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketElementCount()}
 */
TEST_F (ipcdst_unit_test, dstSendPacketElementCount_BadParameter)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, MAX_PACKET_ELEMENTS+1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketElementCount_StreamInternalError}
 * @testcase{22839379}
 * @verify{19791591}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketElementCount(),
 * when IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API from ipcdst object,
 * with valid element count of 2U and dstIndex of Block::singleConn,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketElementCount()}
 */
TEST_F (ipcdst_unit_test, dstSendPacketElementCount_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_comm.signalWrite_fail = true;
    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, 2U);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketElementCount_IlwalidState}
 * @testcase{22839381}
 * @verify{19791591}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketElementCount(),
 * when packet element count has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Call IpcDst::dstSendPacketElementCount() to send the element count.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API from ipcdst object,
 * with valid element count of 2U and dstIndex of Block::singleConn,
 * should trigger error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketElementCount()}
 */
TEST_F (ipcdst_unit_test, dstSendPacketElementCount_IlwalidState)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, 2U);
    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, 2U);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketElementCount_StreamNotConnected}
 * @testcase{21808641}
 * @verify{19791591}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of Queue::dstSendPacketElementCount(), where
 *   dstSendPacketElementCount is ilwoked when queue is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Create the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Note that blocks are not connected.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API from ipcdst object,
 * should result in LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketElementCount()}
 */
TEST_F(ipcdst_unit_test, dstSendPacketElementCount_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketElementCount(Block::singleConn_access, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketElementCount_StreamBadDstIndex}
 * @testcase{21808642}
 * @verify{19791591}
 * @verify{18700800}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketElementCount(),
 * where dstSendPacketElementCount is called with invalid dstIndex.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketElementCount() API with invalid dstIndex of value not equal to
 * Block::singleConn, should result in LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketElementCount()}
 */
TEST_F(ipcdst_unit_test, dstSendPacketElementCount_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketElementCount(ILWALID_CONN_IDX, 1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstDisconnect_Success2}
 * @testcase{21808644}
 * @verify{19791606}
 * @verify{18700800}
 * @testpurpose{Test positive scenario of ipcDst::dstDisconnect(), when
 *   stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Note that blocks are not connected.
 *
 *   The call of IpcDst::dstDisconnect() API from IpcDst object
 * cause connComplete() API of IpcDst should cause LwSciStreamEventType_Disconnected
 * event queried through LwSciStreamBlockEventQuery}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *      IpcDst::connComplete() is configured to return false.
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstDisconnect()}
 */
TEST_F(ipcdst_unit_test, dstDisconnect_Success2)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    /* Initial setup */
    lwscistreamPanicMock npm;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstDisconnect(Block::singleConn_access);

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);
}

/**
 * @testname{ipcdst_unit_test.dstReusePacket_Success2}
 * @testcase{21808645}
 * @verify{19791603}
 * @testpurpose{Test positive scenario of IpcDst::dstReusePacket(), where
 *   ipcdst block is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Note that blocks are not connected.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * should return without triggering any error event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_unit_test, dstReusePacket_Success2)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    lwscistreamPanicMock npm;
    // Enable Ipc Streaming
    initIpcChannel();
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    FenceArray wrapFences { };
    LwSciStreamPacket handle;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(*ipcdstPtr, setErrorEvent_imp(_, _))
               .Times(0);

    ipcdstPtr->dstReusePacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&ipcdstPtr));

}

/**
 * @testname{queue_unit_test.dstSendPacketStatus_StreamInternalError}
 * @testcase{21808646}
 * @verify{19791597}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketStatus(), where
 *   LwSciStreamPacket is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Setup buffer attributes.
 *   3. Pool creates a packet by ilwoking LwSciStreamPoolPacketCreate() and registers
 *     a packet element using LwSciStreamPoolPacketInsertBuffer().
 *
 *   The call of IpcDst::dstSendPacketStatus() API from IpcDst object,
 * with invalid LwSciStreamPacket handle,
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketStatus()}
 */
TEST_F(ipcdst_unit_test, dstSendPacketStatus_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

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
                consumer[0], EVENT_QUERY_TIMEOUT, &event);
     EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

     // Assign cookie to producer packet handle
     LwSciStreamPacket producerPacket = event.packetHandle;
     LwSciStreamCookie producerCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
     LwSciError producerError = LwSciError_Success;

     BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

     ipcdstPtr->dstSendPacketStatus(Block::singleConn_access, ILWALID_PACKET_HANDLE, producerError);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcdst_unit_test.dstSendPacketAttr_BadParameter}
 * @testcase{21808648}
 * @verify{19791594}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketAttr(), where
 *   elemIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Consumer sends element count by ilwoking LwSciStreamBlockPacketElementCount().
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with invalid elemIndex of MAX_ELEMENT_PER_PACKET + 1,
 * should result in LwSciError_BadParameter event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement and Boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_test, dstSendPacketAttr_BadParameter)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    EXPECT_EQ(LwSciError_Success,
     LwSciStreamBlockPacketElementCount(consumer[0], 2U));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            pool, EVENT_QUERY_TIMEOUT, &event));

    // Pool sets packet requirements

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, 3U, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}


/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendPacketAttr_StreamInternalError}
 * @testcase{22839382}
 * @verify{19791594}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketAttr(),
 * when IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends the supported packet element count and element information
 *      to pool.
 *   3. Consumer sends the supported packet element count to pool and pool queries
 *      the same using LwSciStreamBlockEventQuery().
 *   4. Inject fault in  IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Immediate and LwSciWrap::BufAttr,
 * should trigger error event set to LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketAttr_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    pool, EVENT_QUERY_TIMEOUT, &event));

    // Pool sets packet requirements

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_comm.signalWrite_fail = true;
    ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, 0U, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.packElemAttr_ResourceError}
 * @testcase{22839383}
 * @verify{20050608}
 * @testpurpose{Test negative scenario of IpcDst::packElemAttr(),
 * when LwSciBufAttrListIpcExportUnReconciled() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends the supported packet element count and element information
 *      to pool.
 *   3. Consumer sends the supported packet element count to pool and pool queries
 *      the same using LwSciStreamBlockEventQuery().
 *   4. Inject fault in  LwSciBufAttrListIpcExportUnReconciled() to return
 *      LwSciError_ResourceError.
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Immediate and LwSciWrap::BufAttr,
 * should trigger error event set to LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, packElemAttr_ResourceError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    pool, EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_lwscibuf.LwSciBufAttrListIpcExportUnreconciled_fail = true;
    ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, 0, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.packElemAttr_StreamInternalError}
 * @testcase{22839384}
 * @verify{20050608}
 * @testpurpose{Test negative scenario of IpcDst::packElemAttr(),
 * when IpcSendBuffer::packValAndBlob() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer sends the supported packet element count and element information
 *      to pool.
 *   3. Consumer sends the supported packet element count to pool and pool queries
 *      the same using LwSciStreamBlockEventQuery().
 *   4. Inject fault in  IpcSendBuffer::packValAndBlob() to return false.
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid dstIndex of Block::singleConn, elemIndex of 0,  elemType of 0,
 * elemSyncMode of LwSciStreamElementMode_Immediate and LwSciWrap::BufAttr,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, packElemAttr_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    pool, EVENT_QUERY_TIMEOUT, &event));

    // Pool sets packet requirements

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.packVal_fail = true;
    ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, 0, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendSyncCount_StreamInternalError1}
 * @testcase{21808649}
 * @verify{19791585}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncCount(), where
 *   dstSendSyncCount is ilwoked with sync count greater than MAX_SYNC_OBJECTS.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer receives the sync attribute from the consumer by querying
 *      LwSciStreamEventType_SyncAttr event.
 *
 *   The call of IpcDst::dstSendSyncCount() API from IpcDst object,
 * with sync count MAX_SYNC_OBJECTS + 1 and dstIndex of Block::singleConn,
 * should result in LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncCount()}
 */
TEST_F (ipcdst_unit_test, dstSendSyncCount_StreamInternalError1)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncCount(Block::singleConn_access, MAX_SYNC_OBJECTS+1);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcdst_unit_sync_setup_test.packCount_StreamInternalError}
 * @testcase{22839385}
 * @verify{20050599}
 * @testpurpose{Test negative scenario of IpcDst::packCount(), when packing
 *  count failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcDst::dstSendSyncCount() API from IpcDst object,
 * with sync count of 2U and dstIndex of Block::singleConn,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::packCount()}
 */
TEST_F (ipcdst_unit_sync_setup_test, packCount_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;
    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = 2U;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Fifo);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.packVal_fail = true;
    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.packSyncAttr_ResourceError}
 * @testcase{22839386}
 * @verify{20050602}
 * @testpurpose{Test negative scenario of IpcDst::packSyncAttr(), when
 * LwSciSyncAttrListIpcExportUnreconciled() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Inject fault in LwSciSyncAttrListIpcExportUnreconciled() to return
 *    LwSciError_ResourceError.
 *
 *   The call of IpcDst::dstSendSyncAttr() API from IpcDst object,
 * with valid sync attributes and dstIndex of Block::singleConn,
 * should trigger error event set to LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::packSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, packSyncAttr_ResourceError)
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

    test_lwscisync.LwSciSyncAttrListIpcExportUnreconciled_fail = true;
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.packSyncAttr_StreamInternalError}
 * @testcase{22839387}
 * @verify{20050602}
 * @testpurpose{Test negative scenario of IpcDst::packSyncAttr(), when
 * packing waiter requirements failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Inject fault in IpcSendBuffer::packValAndBlob() to return false.
 *
 *   The call of IpcDst::dstSendSyncAttr() API from IpcDst object,
 * with valid sync attributes, synchronousOnly=true and dstIndex of Block::singleConn,
 * should trigger error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::packSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, packSyncAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{nullptr};
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

    test_ipcsendbuffer.packValAndBlob_fail = true;
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendElementStatus_BadParameter}
 * @testcase{21808650}
 * @verify{19791600}
 * @testpurpose{Test negative scenario of IpcDst::dstSendElementStatus(), where
 * elemindex is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3. Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4. Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5. consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   6. consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with invalid value elemIndex of MAX_ELEMENT_PER_PACKET
 * should result in LwSciError_BadParameter event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::disconnectEvent()
 *      - POOL::disconnectDst()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
    LwSciStreamPacket consumerPacket = event.packetHandle;
    LwSciStreamCookie consumerCookie
        = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError consumerError = LwSciError_Success;

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

    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

    ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                      MAX_ELEMENT_PER_PACKET, elemStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendElementStatus_StreamInternalError}
 * @testcase{22839389}
 * @verify{19791600}
 * @testpurpose{Test negative scenario of IpcDst::dstSendElementStatus(), when
 * packet handle is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3. Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4. Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5. consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   6. consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *   7. Inject fault in Block::pktFindByHandle() to return NULL.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with valid elemIndex and other parameters should trigger error event set
 * to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
    LwSciStreamPacket consumerPacket = event.packetHandle;
    LwSciStreamCookie consumerCookie
        = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError consumerError = LwSciError_Success;

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

    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

    test_block.pktFindByHandle_fail = true;
    ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                      0U, elemStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendElementStatus_IlwalidState1}
 * @testcase{22839390}
 * @verify{19791600}
 * @testpurpose{Test negative scenario of IpcDst::dstSendElementStatus(), when
 * acceptance status for the same packet element has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3. Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4. Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5. consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   6. consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *   7. Call IpcDst::dstSendElementStatus() to send the element acceptance status
 *      for the given elemIndex.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with same elemIndex and other valid parameters should trigger error event set
 * to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_IlwalidState1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
    LwSciStreamPacket consumerPacket = event.packetHandle;
    LwSciStreamCookie consumerCookie
        = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));
    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

    ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                      0U, elemStatus);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                      0U, elemStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}


/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendElementStatus_IlwalidState2}
 * @testcase{22839391}
 * @verify{19791600}
 * @testpurpose{Test negative scenario of IpcDst::dstSendElementStatus(), when
 * packet element count is not set by calling IpcDst::dstSendPacketElementCount().}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3. Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4. Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5. consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   6. consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *   7. Stub the implementation of Packet::bufferStatusEventPrepare() to return
 *      LwSciError_IlwalidState.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with same elemIndex and other valid parameters should trigger error event set
 * to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_IlwalidState2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
    LwSciStreamPacket consumerPacket = event.packetHandle;
    LwSciStreamCookie consumerCookie
        = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));
    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketElement, event.type);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_trackArray.prepareEvent_fail = true;
    ipcdstPtr->dstSendElementStatus(dstIndex, handle,
                                      0U, elemStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.dstSendElementStatus_Success}
 * @testcase{22838550}
 * @verify{19791600}
 * @testpurpose{Test success scenario of IpcDst::dstSendElementStatus().}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   3. Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   4. Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   5. consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   6. consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with valid elemIndex should call dstSendElementStatus() API of PoolPtr object.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - POOL::dstSendElementStatus()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_Success)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
    LwSciStreamPacket consumerPacket = event.packetHandle;
    LwSciStreamCookie consumerCookie
        = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError consumerError = LwSciError_Success;

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
        ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                            elemIndex, elemStatus);
    }
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{ipcdst_unit_test.disconnect_StreamInternalError}
 * @testcase{22839392}
 * @verify{19791579}
 * @testpurpose{Test negative scenario of IpcDst::disconnect() when
 * IpcComm::signalDisconnect() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialise Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalDisconnect() to return LwSciError_StreamInternalError.
 *
 * The call of IpcDst::disconnect() API from IpcDst object,
 * should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::disconnect()}
 */
TEST_F(ipcdst_unit_test, disconnect_StreamInternalError)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    // IpcComm::signalDisconnect returns returns LwSciError_StreamInternalError
    test_comm.signalDisconnect_fail = true;

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->disconnect();

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_test.recvSyncAttr_StreamInternalError}
 * @testcase{22839393}
 * @verify{20050617}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncAttr() when unpacking
 * waiter requirements failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackMsgSyncAttr() to return false.
 *
 *   IpcDst::recvSyncAttr() API call triggered, when IpcSrc::srcSendSyncAttr() API
 * is called with valid parameters, should trigger error event set to
 * LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::recvSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncAttr_StreamInternalError)
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

    //IpcRecvBuffer::unpackMsgSyncAttr returns false
    test_ipcrecvbuffer.unpackMsgSyncAttr_fail = true;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncAttr_Success2}
 * @testcase{22839396}
 * @verify{20050617}
 * @testpurpose{Test positive scenario of IpcDst::recvSyncAttr() when endpoint
 * does not support LwSciSyncObj(s).}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   IpcDst::recvSyncAttr() API call triggered, when IpcSrc::srcSendSyncAttr() API
 * is called with valid sync attributes (synchronousOnly=true and LwSciWrap::SyncAttr)
 * and srcIndex, should call srcSendSyncAttr() interface of queue block to send the
 * sync attributes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - QUEUE::srcSendSyncAttr()
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncAttr_Success2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = true;
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
    EXPECT_CALL(*queuePtr[0], srcSendSyncAttr(_, _, _)).Times(1);

    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr[0]));
}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncAttr_ResourceError}
 * @testcase{22839399}
 * @verify{20050617}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncAttr()
 * when LwSciSyncAttrListIpcImportUnreconciled() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Inject fault in LwSciSyncAttrListIpcImportUnreconciled() to return
 *     LwSciError_ResourceError.
 *
 *   IpcDst::recvSyncAttr() API call triggered, when IpcSrc::srcSendSyncAttr() API
 * is called with valid sync attributes (synchronousOnly flag and LwSciWrap::SyncAttr)
 * and srcIndex, should trigger error event set to LwSciError_ResourceError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncAttr_ResourceError)
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
    test_lwscisync.LwSciSyncAttrListIpcImportUnreconciled_fail = true;
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.recvSyncAttr_IlwalidState}
 * @testcase{22839400}
 * @verify{20050617}
 * @testpurpose{Test negative scenario of IpcDst::recvSyncAttr()
 * when waiter requirements already forwarded to downstream.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Call IpcSrc::srcSendSyncAttr() with valid sync attributes
 *     (synchronousOnly flag and LwSciWrap::SyncAttr) and srcIndex.
 *
 *   IpcDst::recvSyncAttr() API call triggered, when IpcSrc::srcSendSyncAttr() API
 * is called with valid sync attributes (synchronousOnly flag and LwSciWrap::SyncAttr)
 * and srcIndex, should trigger error event set to LwSciError_IlwalidState in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::recvSyncAttr()}
 */
TEST_F(ipcdst_unit_sync_setup_test, recvSyncAttr_IlwalidState)
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
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    consumer[0], EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_SyncAttr, event.type);

    test_trackArray.performAction_fail = true;
    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.processReadMsg_IlwalidOperation}
 * @testcase{22839401}
 * @verify{19840539}
 * @testpurpose{Test negative scenario of IpcDst::processReadMsg() when
 * message type is not valid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to corrupt the event type.
 *
 *   IpcDst::processReadMsg() API call triggered, when IpcSrc::srcSendSyncAttr() API
 * is called with valid parameters, should trigger error event set to
 * LwSciError_IlwalidOperation in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processReadMsg()}
 */
TEST_F(ipcdst_unit_sync_setup_test, processReadMsg_IlwalidOperation)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = true;
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
    test_ipcrecvbuffer.unpackIlwalidEvent = true;

    ipcsrcPtr->srcSendSyncAttr(srcIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidOperation, event.error);

}

/**
 * @testname{ipcdst_unit_sync_setup_test.processReadMsg_StreamInternalError1}
 * @testcase{22839402}
 * @verify{19840539}
 * @testpurpose{Test negative scenario of IpcDst::processReadMsg() when
 * IpcComm::readFrame() failed with LwSciError_StreamInternalError error.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Stub the return values for IpcComm::waitForEvent() to return
 *      LwSciError_Success during the first call and LwSciError_StreamInternalError
 *      during the next call.
 *   3. Stub the return value of IpcComm::waitForConnection() to always return
 *      LwSciError_Success.
 *   4. Inject fault in IpcComm::readFrame() to return LwSciError_StreamInternalError.
 *
 *   The call of LwSciStreamIpcDstCreate() should trigger error event set
 *  to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processReadMsg()}
 */
TEST_F(ipcdst_unit_sync_setup_test, processReadMsg_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    test_comm.readFrame_fail = true;
    test_comm.IpcDstwaitForReadEvent_flag = true;
    test_comm.waitForConnection_pass = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_Success,
    LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

    test_comm.waitForConnection_pass = false;
    test_comm.IpcDstwaitForReadEvent_flag = false;
    test_comm.readFrame_fail = false;

}

/**
 * @testname{ipcdst_unit_sync_setup_test.processReadMsg_InsufficientMemory}
 * @testcase{22839403}
 * @verify{19840539}
 * @testpurpose{Test negative scenario of IpcDst::processReadMsg() when
 * IpcComm::readFrame() failed with LwSciError_InsufficientMemory error.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Stub the return values for IpcComm::waitForEvent() to return
 *      LwSciError_Success during the first call and LwSciError_BadParameter
 *      during the next call.
 *   3. Stub the return value of IpcComm::waitForConnection() to always return
 *      LwSciError_Success.
 *   4. Inject fault in IpcComm::readFrame() to return LwSciError_InsufficientMemory.
 *
 *   The call of LwSciStreamIpcDstCreate() should trigger error event set
 *  to LwSciError_InsufficientMemory.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processReadMsg()}
 */
TEST_F(ipcdst_unit_sync_setup_test, processReadMsg_InsufficientMemory)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    bool synchronousOnly = true;
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    test_comm.IpcDstreadFrame_fail = true;
    test_comm.IpcDstwaitForReadEvent_flag = true;
    test_comm.waitForConnection_pass = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_Success,
    LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_InsufficientMemory, event.error);

    test_comm.waitForConnection_pass = false;
    test_comm.IpcDstwaitForReadEvent_flag = false;
    test_comm.IpcDstreadFrame_fail = false;

}

/**
 * @testname{ipcdst_unit_test.recvElemCount_BadParameter}
 * @testcase{22839404}
 * @verify{20050626}
 * @testpurpose{Test negative scenario of IpcDst::recvElemCount() when count
 *  value is invalid(greater than MAX_PACKET_ELEMENTS).}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *
 *   IpcDst::recvElemCount() API call triggered, when IpcSrc::srcSendPacketElementCount() API
 * is called with valid srcIndex and count greater than MAX_PACKET_ELEMENTS,
 * should trigger error event set to LwSciError_BadParameter in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::recvElemCount()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvElemCount_BadParameter)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t srcIndex = Block::singleConn_access;
    uint32_t count = MAX_PACKET_ELEMENTS+1U;;

    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Query maximum number of packet elements
    queryMaxNumElements();

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    test_trackcount.set_fail_BadParameter = true;
    ipcsrcPtr->srcSendPacketElementCount(srcIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketCreate_StreamInternalError}
 * @testcase{22839405}
 * @verify{20050632}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketCreate() when unpacking
 * failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcRecvBuffer::unpackVal() to return false.
 *
 *   IpcDst::recvPacketCreate() API call triggered, when IpcSrc::srcCreatePacket() API
 * is called with valid parameters, should trigger error event set to
 * LwSciError_StreamInternalError in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::recvPacketCreate()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketCreate_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    handle = ~poolCookie;

    //IpcRecvBuffer::unpackVal returns false
    test_ipcrecvbuffer.unpackVal_fail = true;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcCreatePacket(srcIndex, handle);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_buf_setup_test.recvPacketCreate_InsufficientMemory}
 * @testcase{22839406}
 * @verify{20050632}
 * @testpurpose{Test negative scenario of IpcDst::recvPacketCreate() when packet
 * creation failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Complete the packet attributes setup.
 *   5. Inject fault in Block::pktCreate() to return LwSciError_InsufficientMemory.
 *
 *   IpcDst::recvPacketCreate() API call triggered, when IpcSrc::srcCreatePacket() API
 * is called with valid parameters, should trigger error event set to
 * LwSciError_InsufficientMemory in IpcDst.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::recvPacketCreate()}
 */
TEST_F(ipcdst_unit_buf_setup_test, recvPacketCreate_InsufficientMemory)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t srcIndex = Block::singleConn_access;
    LwSciStreamPacket handle;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    handle = ~poolCookie;

    BlockPtr ipcsrcPtr { Block::getRegisteredBlock(ipcsrc) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsrcPtr->srcCreatePacket(srcIndex, handle);
    test_block.pktCreate_fail = true;

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_InsufficientMemory, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendSyncAttr_StreamInternalError}
 * @testcase{22839407}
 * @verify{19791582}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncAttr() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstSendSyncAttr() API from ipcdst object,
 * with valid parameters, should call IpcDst block to post LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendSyncAttr()}
 */
TEST_F(ipcdst_unit_test, dstSendSyncAttr_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    LwSciStreamEvent event;

    uint32_t dstIndex = Block::singleConn_access;
    bool synchronousOnly = consSynchronousOnly;
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    //IpcComm::signalWrite returns LwSciError_StreamInternalError
    test_comm.signalWrite_fail = true;

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncCount_IlwalidState}
 * @testcase{22839412}
 * @verify{19791585}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncCount(), when
 *  LwSciSyncObj count has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Call IpcDst::dstSendSyncCount() to send the sync count.
 *
 *   The call of IpcDst::dstSendSyncCount() API from IpcDst object,
 * with valid sync count and dstIndex of Block::singleConn,
 * should trigger error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncCount()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncCount_IlwalidState)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = 1U;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncCount_StreamBadDstIndex}
 * @testcase{22839416}
 * @verify{19791585}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncCount(), when
 *  dstIndex is invalid.}
 * @testbehavior{
 * Setup:
 *   Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendSyncCount() API from IpcDst object,
 * with valid sync count and invalid dstIndex,
 * should trigger error event set to LwSciError_StreamBadDstIndex.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncCount()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncCount_StreamBadDstIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    LwSciStreamEvent event;
    uint32_t dstIndex = ILWALID_CONN_IDX;
    uint32_t count = 1U;

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
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendSyncCount_StreamInternalError2}
 * @testcase{22839417}
 * @verify{19791585}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncCount() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 *Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstSendSyncCount() API from IpcDst object,
 * with a valid parameters, should call IpcDst block to post LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendSyncCount()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncCount_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;


    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = 1U;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    //IpcComm::signalWrite returns LwSciError_StreamInternalError
    test_comm.signalWrite_fail = true;

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendSyncCount_StreamNotConnected}
 * @testcase{22839418}
 * @verify{19791585}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncCount() when stream
 * is not in connected state.}
 * @testbehavior{
 *Setup:
 *   1. Initialize Ipc channel.
 *   2. Create the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendSyncCount() API from IpcDst object,
 * with a invalid dstIndex parameter, should call the IpcDst block to
 * post the LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendSyncCount()}
 */
TEST_F(ipcdst_unit_test, dstSendSyncCount_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = 5U;
    uint32_t count = 1U;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendSyncDesc_StreamInternalError}
 * @testcase{22839419}
 * @verify{19791588}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncDesc() when
 *  IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstSendSyncDesc() API from ipcdst object,
 * with a valid parameters, should call the IpcDst block to post the
 * LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncDesc_StreamInternalError)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Ref;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex = 0;
    LwSciWrap::SyncObj wrapSyncObj;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));

    // Consumer sends its sync count to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], prodSyncCount));

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));

    //IpcComm::signalWrite returns LwSciError_StreamInternalError
    test_comm.signalWrite_fail = true;

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}


/**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncDesc_IlwalidState1}
 * @testcase{22839420}
 * @verify{19791588}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncDesc(), when
 *  LwSciSyncObj for the same syncIndex has already been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer send the waiter requirements to consumer by ilwoking
 *      LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends the sync count to producer by ilwoking LwSciStreamBlockSyncObjCount().
 *   4. Call IpcDst::dstSendSyncDesc() to send the sync object for the syncIndex.
 *
 *   The call of IpcDst::dstSendSyncDesc() API for the same syncIndex
 * should trigger error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncDesc_IlwalidState)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Ref;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex = 0;
    LwSciWrap::SyncObj wrapSyncObj;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));

    // Consumer sends its sync count to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], prodSyncCount));

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));


    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncDesc_IlwalidState2}
 * @testcase{22839421}
 * @verify{19791588}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncDesc(), when
 *   IpcDst::dstSendSyncCount() is not yet called.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer send the waiter requirements to consumer by ilwoking
 *      LwSciStreamBlockSyncRequirements().
 *
 *   The call of IpcDst::dstSendSyncDesc() API with valid parameters,
 * should trigger error event set to LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncDesc_IlwalidState2)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Ref;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex = 0;
    LwSciWrap::SyncObj wrapSyncObj;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

 /**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncDesc_BadParameter}
 * @testcase{22839423}
 * @verify{19791588}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncDesc(), when
 *   syncIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer send the waiter requirements to consumer by ilwoking
 *      LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends the sync count to producer by ilwoking LwSciStreamBlockSyncObjCount().
 *
 *   The call of IpcDst::dstSendSyncDesc() API with invalid syncIndex,
 * should trigger error event set to LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncDesc_BadParameter)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Ref;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t syncIndex = 5U;
    LwSciWrap::SyncObj wrapSyncObj;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));

    // Consumer sends its sync count to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], prodSyncCount));

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));


    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

 /**
 * @testname{ipcdst_unit_sync_setup_test.dstSendSyncDesc_StreamBadDstIndex}
 * @testcase{22839425}
 * @verify{19791588}
 * @testpurpose{Test negative scenario of IpcDst::dstSendSyncDesc(), when
 *   dstIndex is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   2. Producer send the waiter requirements to consumer by ilwoking
 *      LwSciStreamBlockSyncRequirements().
 *   3. Consumer sends the sync count to producer by ilwoking LwSciStreamBlockSyncObjCount().
 *
 *   The call of IpcDst::dstSendSyncDesc() API with invalid dstIndex,
 * should trigger error event set to LwSciError_StreamBadDstIndex.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - Access qualifier of Block::singleConn is made public.}
 * @verifyFunction{IpcDst::dstSendSyncDesc()}
 */
TEST_F(ipcdst_unit_sync_setup_test, dstSendSyncDesc_StreamBadDstIndex)
{
    //Initial setup
    using ::testing::_;
    using ::testing::Ref;
    using ::testing::Mock;

    uint32_t dstIndex = ILWALID_CONN_IDX;
    uint32_t syncIndex = 0U;
    LwSciWrap::SyncObj wrapSyncObj;
    LwSciStreamEvent event;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncRequirements(producer,
                                         prodSynchronousOnly,
                                         prodSyncAttrList));

    // consumer receives producer's sync object requirement
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(consumer[0], EVENT_QUERY_TIMEOUT, &event));

    // Consumer sends its sync count to the producer
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamBlockSyncObjCount(consumer[0], prodSyncCount));

    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockEventQuery(producer, EVENT_QUERY_TIMEOUT, &event));


    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendSyncDesc(dstIndex, syncIndex, wrapSyncObj);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

 /**
 * @testname{ipcdst_unit_buf_setup_test.dstSendPacketAttr_IlwalidState1}
 * @testcase{22839426}
 * @verify{19791594}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketAttr() when
 *  IpcDst::dstSendPacketElementCount() is not called.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid parameters, should trigger error event set to
 * LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketAttr_IlwalidState1)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Pool sets packet requirements
    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, 0U, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

 /**
 * @testname{ipcdst_unit_buf_setup_test.dstSendPacketAttr_IlwalidState2}
 * @testcase{22839427}
 * @verify{19791594}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketAttr() when
 *  consumer's supported packet element information for the elemIndex has already
 *  been scheduled to be sent.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Consumer sends the packet element count using LwSciStreamBlockPacketElementCount().
 *   4. Call IpcDst::dstSendPacketAttr() to send the element attributes for the
 *     given elemIndex.
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with the same elemIndex and other valid parameters, should trigger error event set to
 * LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketAttr_IlwalidState2)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Consumer sends its packet element count to pool
    EXPECT_EQ(LwSciError_Success,
        LwSciStreamBlockPacketElementCount(consumer[0], consElementCount));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    pool, EVENT_QUERY_TIMEOUT, &event));

    // Pool sets packet requirements

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, 0U, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketAttr(Block::singleConn_access, 0U, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketAttr_StreamBadDstIndex}
 * @testcase{22839428}
 * @verify{19791594}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketAttr() when
 * dstIndex is Invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with a invalid elemIndex parameter, should call the IpcDst block to post
 * the LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_test, dstSendPacketAttr_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    uint32_t dstIndex = 5U;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketAttr(dstIndex, MAX_ELEMENT_PER_PACKET + 1, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);

}

/**
 * @testname{ipcdst_unit_test.dstSendPacketAttr_StreamNotConnected}
 * @testcase{22839431}
 * @verify{19791594}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketAttr() when
 * stream is not connected.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketAttr() API from IpcDst object,
 * with valid parameters, should call the IpcDst block to post
 * the LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendPacketAttr()}
 */
TEST_F(ipcdst_unit_test, dstSendPacketAttr_StreamNotConnected)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    uint32_t dstIndex = 0U;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstSendPacketAttr(dstIndex, 0U, 0,
                                LwSciStreamElementMode_Immediate,
                                wrapBufAttr);


    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                    ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

}

/**
 * @testname{ipcdst_unit_test.dstSendPacketStatus_StreamInternalError1}
 * @testcase{22839439}
 * @verify{19791597}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketStatus() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *   3. Producer and Consumer send the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   4. Pool sends reconciled PacketElementCount and PacketAttr back to producer and consumer
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr() respectively.
 *   5. Pool block creates the packet using LwSciStreamPoolPacketCreate() and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer().
 *   6. Consumer queries the LwSciStreamEventType_PacketCreate event through
 *      LwSciStreamBlockEventQuery and gets packetHandle.
 *   7. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstSendPacketStatus() API from IpcDst object,
 * with a valid parameters, should call IpcDst block to post LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendPacketStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketStatus_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    // Pool creates the new packet
    ASSERT_EQ(LwSciError_Success,
        LwSciStreamPoolPacketCreate(pool, poolCookie, &handle));

    // Consumer receives PacketCreate event
    EXPECT_EQ(LwSciError_Success,
              LwSciStreamBlockEventQuery(consumer[0],
                                         EVENT_QUERY_TIMEOUT,
                                         &event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    //IpcComm::signalWrite returns LwSciError_StreamInternalError
    test_comm.signalWrite_fail = true;

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketStatus_StreamBadDstIndex}
 * @testcase{22839447}
 * @verify{19791597}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketStatus() when
 * dstIndex is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketStatus() API from IpcDst object,
 * with a invalid parameters, should call the IpcDst block to post the
 * LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendPacketStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketStatus_StreamBadDstIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = 1U;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendPacketStatus_StreamNotConnected}
 * @testcase{22839459}
 * @verify{19791597}
 * @testpurpose{Test negative scenario of IpcDst::dstSendPacketStatus() when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create the producer, pool, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendPacketStatus() API from IpcDst object,
 * with valid parameters, should call the IpcDst block to post the
 * LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendPacketStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendPacketStatus_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = 0U;
    LwSciStreamPacket handle;
    LwSciError packetStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsdstPtr->dstSendPacketStatus(dstIndex, handle, packetStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstSendElementStatus_StreamInternalError1}
 * @testcase{22839465}
 * @verify{19791600}
 * @testpurpose{Test negative scenario of IpcDst::dstSendElementStatus() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Producer and Consumer sends the PacketElementCount and PacketAttr,
 *      through LwSciStreamBlockPacketElementCount() and LwSciStreamBlockPacketAttr()
 *      to Pool.
 *   4. Pool sends reconciled PacketElementCount and PacketAttr back
 *      to producer and consumer.
 *   5. Pool block creates the packet using LwSciStreamPoolPacketCreate and inserts buffer
 *      using LwSciStreamPoolPacketInsertBuffer.
 *   6. consumer accepts the packet by calling LwSciStreamBlockPacketAccept() after
 *      querying for LwSciStreamEventType_PacketCreate through
 *      LwSciStreamBlockEventQuery() and receiving packetHandle.
 *   7. consumer receives the LwSciStreamEventType_PacketElement by querying through
 *      LwSciStreamBlockEventQuery() and receiving element index.
 *   8. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with a valid parameters, should call IpcDst block to post LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    // Setup the stream packet attributes
    packetAttrSetup();

    // Choose pool's cookie and handle for new packet
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

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
    LwSciStreamPacket consumerPacket = event.packetHandle;
    LwSciStreamCookie consumerCookie
        = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    LwSciError consumerError = LwSciError_Success;

    // Consumer accepts packet provided by the pool
    ASSERT_EQ(LwSciError_Success,
              LwSciStreamBlockPacketAccept(consumer[0],
                                           consumerPacket,
                                           consumerCookie,
                                           consumerError));
    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    //IpcComm::signalWrite returns LwSciError_StreamInternalError
    test_comm.signalWrite_fail = true;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                        elemIndex, elemStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_unit_test.dstSendElementStatus_StreamBadDstIndex}
 * @testcase{22839473}
 * @verify{19791600}
 * @testpurpose{Test negative scenario of IpcDst::dstSendElementStatus() when
 * dstIndex is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with a invalid dstIndex parameter, should call the IpcDst block to post
 * the LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_StreamBadDstIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = 1U;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                        elemIndex, elemStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);

}

/**
 * @testname{ipcdst_unit_test.dstSendElementStatus_StreamNotConnected}
 * @testcase{22839489}
 * @verify{19791600}
 * @testpurpose{Test negative scenario of IpcDst::dstSendElementStatus() when
 * stream is not in connected state.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstSendElementStatus() API from IpcDst object,
 * with valid parameters, should call the IpcDst block to post
 * the LwSciError_StreamNotConnected event.}
 * @testmethod{Requirements Based}
 * @casederiv{boundary values}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstSendElementStatus()}
 */
TEST_F(ipcdst_unit_buf_setup_test, dstSendElementStatus_StreamNotConnected)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    uint32_t dstIndex = 0U;
    LwSciStreamPacket handle;
    uint32_t elemIndex;
    LwSciError elemStatus = LwSciError_Success;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);

    BlockPtr ipcsdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcsdstPtr->dstSendElementStatus(dstIndex, handle,
                                        elemIndex, elemStatus);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamNotConnected, event.error);

}
/**
 * @testname{ipcdst_unit_test.dstReusePacket_StreamInternalError}
 * @testcase{22839500}
 * @verify{19791603}
 * @testpurpose{Test negative scenario of IpcDst::dstReusePacket() when packet
 *  handle is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object with
 * invalid packet handle, should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_unit_test, dstReusePacket_StreamInternalError)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();
    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);

    // Connect stream
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    FenceArray wrapFences { };
    LwSciStreamPacket handle;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstReusePacket(Block::singleConn_access, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_unit_test.dstReusePacket_StreamBadDstIndex}
 * @testcase{22839513}
 * @verify{19791603}
 * @testpurpose{Test negative scenario of IpcDst::dstReusePacket() when dstIndex
 * is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with a invalid dstIndex parameter, should call the IpcDst block to
 * post the LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_unit_test, dstReusePacket_StreamBadDstIndex)
{
    /* Initial setup */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    uint32_t dstIndex = 1U;
    LwSciStreamEvent event;

    // Enable Ipc Streaming
    initIpcChannel();

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox,1,1);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    FenceArray wrapFences { };
    LwSciStreamPacket handle;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapFences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);

}

/**
 * @testname{IpcDst_unit_test.dstReusePacket_StreamInternalError1}
 * @testcase{22839525}
 * @verify{19791603}
 * @testpurpose{Test negative scenario of IpcDst::dstReusePacket() when
 * IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *   3. Set up synchronization and buffer resources, with packet count greater than 0.
 *   4. Producer gets the next available packet from Pool using LwSciStreamProducerPacketGet().
 *   5. Producer gets packet and inserts the data into the stream
 *      through LwSciStreamProducerPacketPresent().
 *   6. Pool sends the packet downstream to consumer and triggers packet ready event to consumer.
 *   7. Consumer queries for packet ready event by calling LwSciStreamBlockEventQuery().
 *   8. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with a valid parameters, should call IpcDst block to post LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_packet_stream_test, dstReusePacket_StreamInternalError1)
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

    //IpcComm::signalWrite returns LwSciError_StreamInternalError
    test_comm.signalWrite_fail = true;

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////


    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.dstReusePacket_StreamInternalError2}
 * @testcase{22839545}
 * @verify{19791603}
 * @testpurpose{Test negative scenario of IpcDst::dstReusePacket() when
 *  packet's current location is not Location::Downstream.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *   3. Set up synchronization and buffer resources, with packet count greater than 0.
 *   4. Producer gets the next available packet from Pool using LwSciStreamProducerPacketGet().
 *   5. Producer gets packet and inserts the data into the stream
 *      through LwSciStreamProducerPacketPresent().
 *   6. Pool sends the packet downstream to consumer and triggers packet ready event to consumer.
 *   7. Consumer queries for packet ready event by calling LwSciStreamBlockEventQuery().
 *   8. Change the packet location to other than Location::Downstream by using
 *      Packet::locationUpdate().
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with a valid parameters, should trigger error event set to
 * LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_packet_stream_test, dstReusePacket_StreamInternalError2)
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

    PacketPtr const pkt { ipcdstPtr->pktFindByHandle_access(handle) };

    if (true == pkt->locationCheck(Packet::Location::Downstream))
    {
        pkt->locationUpdate(Packet::Location::Downstream, Packet::Location::Queued);
    }
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.dstReusePacket_IlwalidState}
 * @testcase{22839552}
 * @verify{19791603}
 * @testpurpose{Test negative scenario of IpcDst::dstReusePacket() when
 *  payload for the packet is already set.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *   3. Set up synchronization and buffer resources, with packet count greater than 0.
 *   4. Producer gets the next available packet from Pool using LwSciStreamProducerPacketGet().
 *   5. Producer gets packet and inserts the data into the stream
 *      through LwSciStreamProducerPacketPresent().
 *   6. Pool sends the packet downstream to consumer and triggers packet ready event to consumer.
 *   7. Consumer queries for packet ready event by calling LwSciStreamBlockEventQuery().
 *   8. Call IpcDst::dstReusePacket() to release the packet for reuse.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with same packet handle and other valid parameters, should trigger error event set to
 * LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_packet_stream_test, dstReusePacket_IlwalidState)
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

    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    PacketPtr const pkt { ipcdstPtr->pktFindByHandle_access(handle) };

    if (true == pkt->locationCheck(Packet::Location::Queued))
    {
        pkt->locationUpdate(Packet::Location::Queued, Packet::Location::Downstream);
    }

    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_IlwalidState, event.error);
    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.packPayload_StreamInternalError1}
 * @testcase{22839558}
 * @verify{20050614}
 * @testpurpose{Test negative scenario of IpcDst::packPayload() when packing
 * handle failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *   3. Set up synchronization and buffer resources, with packet count greater than 0.
 *   4. Producer gets the next available packet from Pool using LwSciStreamProducerPacketGet().
 *   5. Producer gets packet and inserts the data into the stream
 *      through LwSciStreamProducerPacketPresent().
 *   6. Pool sends the packet downstream to consumer and triggers packet ready event to consumer.
 *   7. Consumer queries for packet ready event by calling LwSciStreamBlockEventQuery().
 *   8. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with a valid parameters, should trigger error event set to
 * LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_packet_stream_test, packPayload_StreamInternalError1)
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

    test_ipcsendbuffer.packVal_fail = true;
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    free(fences);
}



/**
 * @testname{ipcdst_packet_stream_test.packPayload_StreamInternalError2}
 * @testcase{22839563}
 * @verify{20050614}
 * @testpurpose{Test negative scenario of IpcDst::packPayload() when packing
 * fences failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *   3. Set up synchronization and buffer resources, with packet count greater than 0.
 *   4. Producer gets the next available packet from Pool using LwSciStreamProducerPacketGet().
 *   5. Producer gets packet and inserts the data into the stream
 *      through LwSciStreamProducerPacketPresent().
 *   6. Pool sends the packet downstream to consumer and triggers packet ready event to consumer.
 *   7. Consumer queries for packet ready event by calling LwSciStreamBlockEventQuery().
 *   8. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with a valid parameters, should trigger error event set to
 * LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_packet_stream_test, packPayload_StreamInternalError2)
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

    test_ipcsendbuffer.dstReusePacket_packVal_fail = true;
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
    free(fences);
}

/**
 * @testname{ipcdst_packet_stream_test.packPayload_ResourceError}
 * @testcase{22839566}
 * @verify{20050614}
 * @testpurpose{Test negative scenario of IpcDst::packPayload() when
 * LwSciSyncIpcExportFence() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, queue and consumer blocks.
 *   3. Set up synchronization and buffer resources, with packet count greater than 0.
 *   4. Producer gets the next available packet from Pool using LwSciStreamProducerPacketGet().
 *   5. Producer gets packet and inserts the data into the stream
 *      through LwSciStreamProducerPacketPresent().
 *   6. Pool sends the packet downstream to consumer and triggers packet ready event to consumer.
 *   7. Consumer queries for packet ready event by calling LwSciStreamBlockEventQuery().
 *   8. Inject fault in LwSciSyncIpcExportFence() to return LwSciError_ResourceError.
 *
 *   The call of IpcDst::dstReusePacket() API from IpcDst object,
 * with a valid parameters, should trigger error event set to
 * LwSciError_ResourceError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstReusePacket()}
 */
TEST_F(ipcdst_packet_stream_test, packPayload_ResourceError)
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

    test_lwscisync.LwSciSyncIpcExportFence_fail = true;
    ipcdstPtr->dstReusePacket(dstIndex, handle, wrapPostfences);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_ResourceError, event.error);
    free(fences);
}


/**
 * @testname{ipcdst_unit_sync_setup_test.processWriteMsg_BadParameter}
 * @testcase{22839568}
 * @verify{19839633}
 * @testpurpose{Test negative scenario of IpcDst::processWriteMsg() when cloning
 * LwSciSyncAttrList failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in LwSciSyncAttrListClone() to return LwSciError_BadParameter.
 *
 *   The call of IpcDst::dstSendSyncAttr() with required parameters, should trigger
 *   error event set to LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processWriteMsg()}
 */
 TEST_F(ipcdst_unit_sync_setup_test, processWriteMsg_BadParameter)
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
    test_lwscisync.LwSciSyncAttrListClone_fail = true;
    ipcdstPtr->dstSendSyncAttr(dstIndex, synchronousOnly, syncAttr);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_BadParameter, event.error);

}

/**
 * @testname{ipcdst_unit_sync_setup_test.processWriteMsg_StreamInternalError1}
 * @testcase{22839569}
 * @verify{19839633}
 * @testpurpose{Test negative scenario of IpcDst::processWriteMsg() when pending
 * event type is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcDst::pendingSendEvent() to return invalid event type.
 *
 *   The call of IpcDst::dstSendSyncCount() with valid parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processWriteMsg()}
 */
TEST_F(ipcdst_unit_sync_setup_test, processWriteMsg_StreamInternalError1)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    uint32_t count = 2U;
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
    test_trackcount.pending_event_fail = true;
    ipcdstPtr->dstSendSyncCount(dstIndex, count);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_packet_stream_test.dispatchThreadFunc_StreamInternalError1}
 * @testcase{22839573}
 * @verify{19865832}
 * @testpurpose{Test negative scenario of IpcDst::dispatchThreadFunc() when
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
 *   The call of LwSciStreamIpcDstCreate() with required parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dispatchThreadFunc()}
 */
TEST_F(ipcdst_packet_stream_test, dispatchThreadFunc_StreamInternalError1)
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
    LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}


/**
 * @testname{ipcdst_packet_stream_test.dispatchThreadFunc_StreamInternalError2}
 * @testcase{22839577}
 * @verify{19865832}
 * @testpurpose{Test negative scenario of IpcDst::dispatchThreadFunc() when
 *   IpcComm::flushWriteSignals() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Stub the return value of IpcComm::waitForConnection() to complete
 *      the connection successfully.
 *
 *   The call of LwSciStreamIpcDstCreate() with required parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dispatchThreadFunc()}
 */
TEST_F(ipcdst_packet_stream_test, dispatchThreadFunc_StreamInternalError2)
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
    LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_unit_sync_setup_test.processWriteMsg_StreamInternalError2}
 * @testcase{22839579}
 * @verify{19839633}
 * @testpurpose{Test negative scenario of IpcDst::processWriteMsg() when packing
 * event type failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create producer, pool, ipcsrc, ipcdst, queue and consumer blocks.
 *   3. Connect the blocks to complete the streaming pipeline.
 *   4. Inject fault in IpcSendBuffer::packVal() to return false.
 *
 *   The call of IpcDst::dstSendSyncCount() with valid parameters, should trigger
 *   error event set to LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::processWriteMsg()}
 */
TEST_F(ipcdst_unit_sync_setup_test, processWriteMsg_StreamInternalError2)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;

    uint32_t dstIndex = Block::singleConn_access;
    LwSciStreamEvent event;
    uint32_t count = 2U;

    // Initialise Ipc channel
    initIpcChannel();

    //Create a mailbox stream.
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    test_ipcsendbuffer.processMsg_pack_fail = true;
    ipcdstPtr->dstSendSyncCount(dstIndex, count);
    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);

}

/**
 * @testname{ipcdst_unit_test.dstDisconnect_StreamInternalError}
 * @testcase{22839581}
 * @verify{19791606}
 * @testpurpose{Test negative scenario of IpcDst::dstDisconnect() when
 *  IpcComm::signalWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *   3. Inject fault in IpcComm::signalWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcDst::dstDisconnect() API from IpcDst object,
 * with a valid parameters, should post LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstDisconnect()}
 */
TEST_F(ipcdst_unit_test, dstDisconnect_StreamInternalError)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);

    // Connect stream
    connectStream();

    //IpcComm::signalWrite returns LwSciError_StreamInternalError
    test_comm.signalWrite_fail = true;

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstDisconnect(Block::singleConn_access);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamInternalError, event.error);
}

/**
 * @testname{ipcdst_unit_test.dstDisconnect_StreamBadDstIndex}
 * @testcase{22839582}
 * @verify{19791606}
 * @testpurpose{Test negative scenario of IpcDst::dstDisconnect() when dstIndex
 * is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of IpcDst::dstDisconnect() API from IpcDst object,
 * with a invalid dstIndex parameter, should call the IpcDst block to
 * post the LwSciError_StreamBadDstIndex event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dstDisconnect()}
 */
TEST_F(ipcdst_unit_test, dstDisconnect_StreamBadDstIndex)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Mock;
    LwSciStreamEvent event;
    uint32_t dstIndex = 1U;

    // Initialize Ipc channel
    initIpcChannel();

    // Create a mailbox stream
    createBlocks(QueueType::Mailbox);
    connectStream();

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    ipcdstPtr->dstDisconnect(dstIndex);

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_StreamBadDstIndex, event.error);
}

/**
 * @testname{ipcdst_unit_sync_setup_test.IpcDst_Ipccomm_StreamInternalError}
 * @testcase{22839589}
 * @verify{19791573}
 * @testpurpose{Test negative scenario of LwSciStream::IpcDst::IpcDst
 * when IpcComm::isInitSuccess() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Inject fault in IpcComm::isInitSuccess() to return false.
 *
 *   The call of LwSciStream::IpcDst::IpcDst constructor through LwSciStreamIpcDstCreate method
 * should return LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::IpcDst::IpcDst}
 */
TEST_F(ipcdst_packet_stream_test, IpcDst_Ipccomm_StreamInternalError)
{
    /*Initial setup*/

    // Initialize Ipc channel
    initIpcChannel();

    //IpcComm::isInitSuccess returns false
    test_comm.isInitSuccess_fail = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_StreamInternalError,
    LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));
}

/**
 * @testname{ipcdst_unit_sync_setup_test.IpcDst_SendBuffer_StreamInternalError}
 * @testcase{22839593}
 * @verify{19791573}
 * @testpurpose{Test negative scenario of LwSciStream::IpcDst::IpcDst
 * when IpcComm::isInitSuccess() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Inject fault in IpcSendBuffer::isInitSuccess() to return false.
 *
 *   The call of LwSciStream::IpcDst::IpcDst constructor through LwSciStreamIpcDstCreate method
 * should return LwSciError_StreamInternalError event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciStream::IpcDst::IpcDst}
 */
TEST_F(ipcdst_packet_stream_test, IpcDst_SendBuffer_StreamInternalError)
{
    /*Initial setup*/

    // Initialize Ipc channel
    initIpcChannel();

    //IpcSendBuffer::isInitSuccess returns false
    test_ipcsendbuffer.isInitSuccess_fail = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_StreamInternalError,
    LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));
}

/**
 * @testname{ipcdst_packet_stream_test.dispatchThreadFunc_Unknown}
 * @testcase{22839595}
 * @verify{19865832}
 * @testpurpose{Test negative scenario of IpcDst::dispatchThreadFunc()
 * when IpcComm::waitForConnection() failed.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Inject failure in IpcComm::waitForConnection() to return LwSciError_Unknown.
 *
 *   The call of LwSciStreamIpcDstCreate() API with valid parameters, should
 *  trigger error event set to LwSciError_Unknown.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::dispatchThreadFunc()}
 */
TEST_F(ipcdst_packet_stream_test, dispatchThreadFunc_Unknown)
{
    //Initial setup
    LwSciStreamEvent event;

    // Initialize Ipc channel
    initIpcChannel();


    test_comm.waitForConnection_fail = true;

    //Create a mailbox stream.
    ASSERT_EQ(LwSciError_Success,
    LwSciStreamIpcDstCreate(ipcDst.endpoint, syncModule, bufModule, &ipcdst));

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Error, event.type);
    EXPECT_EQ(LwSciError_Unknown, event.error);

}

/**
 * @testname{ipcdst_unit_sync_setup_test.srcXmitNotifyConnection_StreamBadSrcIndex}
 * @testcase{22839600}
 * @verify{19791615}
 * @testpurpose{Test negative scenario of IpcDst::srcXmitNotifyConnection()
 * when srcIndex is invalid.}
 * @testbehavior{
 * Setup:
 *   1. Initialize Ipc channel.
 *   2. Create and connect the producer, pool, ipcSrc,ipcDst, queue and consumer blocks.
 *
 *   The call of srcXmitNotifyConnection() API with invalid srcIndex, should
 *  trigger error event set to LwSciError_StreamBadSrcIndex.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcDst::srcXmitNotifyConnection()}
 */
TEST_F(ipcdst_unit_sync_setup_test, srcXmitNotifyConnection_StreamBadSrcIndex)
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

    BlockPtr ipcdstPtr { Block::getRegisteredBlock(ipcdst) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    ipcdstPtr->srcXmitNotifyConnection(ILWALID_CONN_IDX);

     EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
            ipcdst, EVENT_QUERY_TIMEOUT, &event));
     EXPECT_EQ(LwSciStreamEventType_Error, event.type);
     EXPECT_EQ(LwSciError_StreamBadSrcIndex, event.error);
}


} // namespace LwSciStream


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
