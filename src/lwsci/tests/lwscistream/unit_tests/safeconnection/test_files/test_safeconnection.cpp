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

#include "block.h"
#include "lwscistream_common.h"
#include "apiblockinterface.h"
#include "srcblockinterface.h"
#include "dstblockinterface.h"
#include "safeconnection.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include <vector>
#include "lwscistream_LwSciCommonPanic_mock.h"

class safeconnection_unit_test: public ::LwSciStreamTest {
public:
   safeconnection_unit_test( ) {
       // initialization code here
   }

   void SetUp( ) {
       // code here will execute just before the test ensues
   }

   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~safeconnection_unit_test( )  {
       // cleanup any pending stuff, but no exceptions allowed
   }

   // put in any custom data members that you need
};
namespace LwSciStream {
/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendSyncAttr_Success}
 * @testcase{21382749}
 * @verify{18723858}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendSyncAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *  The call of BlockWrap-DstBlockInterface::srcSendSyncAttr with 'synchronousOnly flag'
 * and 'LwSciWrap::SyncAttr', will internally call Pool::srcSendSyncAttr() API
 * with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncAttr()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendSyncAttr}
 */
TEST_F (safeconnection_unit_test, srcSendSyncAttr_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    EXPECT_CALL(*poolPtr, srcSendSyncAttr(connIdx, prodSynchronousOnly, Ref(syncAttr)))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncAttr(prodSynchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendSyncAttr_failure}
 * @testcase{22034106}
 * @verify{18723858}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendSyncAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *  The call of BlockWrap-DstBlockInterface::srcSendSyncAttr with 'synchronousOnly flag'
 * and 'LwSciWrap::SyncAttr', after disconnect will fail to call Pool::srcSendSyncAttr().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncAttr()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendSyncAttr}
 */
TEST_F (safeconnection_unit_test, srcSendSyncAttr_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    EXPECT_CALL(*poolPtr, srcSendSyncAttr(connIdx, prodSynchronousOnly, Ref(syncAttr)))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncAttr(prodSynchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendSyncAttr_Success}
 * @testcase{21382752}
 * @verify{18723738}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstSendSyncAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendSyncAttr with 'synchronousOnly flag'
 * and 'LwSciWrap::SyncAttr', will internally call Pool::dstSendSyncAttr() API
 * with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncAttr()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendSyncAttr}
 */
TEST_F (safeconnection_unit_test, dstSendSyncAttr_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    EXPECT_CALL(*poolPtr, dstSendSyncAttr(connIdx, prodSynchronousOnly, Ref(syncAttr)))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncAttr(prodSynchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendSyncAttr_failure}
 * @testcase{22034110}
 * @verify{18723738}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstSendSyncAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendSyncAttr with 'synchronousOnly flag'
 * and 'LwSciWrap::SyncAttr', after disconnect will fail to call Pool::dstSendSyncAttr().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncAttr()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendSyncAttr}
 */
TEST_F (safeconnection_unit_test, dstSendSyncAttr_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    EXPECT_CALL(*poolPtr, dstSendSyncAttr(connIdx, prodSynchronousOnly, Ref(syncAttr)))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncAttr(prodSynchronousOnly, syncAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendSyncCount_Success}
 * @testcase{21382755}
 * @verify{18723861}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendSyncCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendSyncCount with sync objects count,
 * will internally call Pool::srcSendSyncCount() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncCount()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendSyncCount}
 */
TEST_F (safeconnection_unit_test, srcSendSyncCount_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendSyncCount(connIdx, syncCount))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendSyncCount_failure}
 * @testcase{22034113}
 * @verify{18723861}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendSyncCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendSyncCount with sync objects count,
 * after disconnect will fail to call Pool::srcSendSyncCount().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncCount()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendSyncCount}
 */
TEST_F (safeconnection_unit_test, srcSendSyncCount_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcSendSyncCount(connIdx, syncCount))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendSyncCount_Success}
 * @testcase{21382758}
 * @verify{18723741}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstSendSyncCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendSyncCount with sync objects count,
 * will internally call Pool::dstSendSyncCount() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncCount()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendSyncCount}
 */
TEST_F (safeconnection_unit_test, dstSendSyncCount_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstSendSyncCount(connIdx, syncCount))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendSyncCount_failure}
 * @testcase{22034116}
 * @verify{18723741}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstSendSyncCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendSyncCount with sync objects count,
 * after disconnect will fail to call Pool::dstSendSyncCount().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncCount()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendSyncCount}
 */
TEST_F (safeconnection_unit_test, dstSendSyncCount_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstSendSyncCount(connIdx, syncCount))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendSyncDesc_Success}
 * @testcase{21382764}
 * @verify{18723864}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendSyncDesc}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendSyncDesc with 'syncIndex' and
 * 'LwSciWrap::syncObj', will internally call Pool::srcSendSyncDesc() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncDesc()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendSyncDesc}
 */
TEST_F (safeconnection_unit_test, srcSendSyncDesc_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncIdx = 0;

    LwSciWrap::SyncObj wrapSyncObj {nullptr};

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendSyncDesc(connIdx, syncIdx, Ref(wrapSyncObj)))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncDesc(syncIdx, wrapSyncObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendSyncDesc_failure}
 * @testcase{22034119}
 * @verify{18723864}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendSyncDesc}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendSyncDesc with 'syncIndex' and
 * 'LwSciWrap::syncObj', after disconnect will fail to call Pool::srcSendSyncDesc().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncDesc()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendSyncDesc}
 */
TEST_F (safeconnection_unit_test, srcSendSyncDesc_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncIdx = 0;

    LwSciWrap::SyncObj wrapSyncObj {nullptr};

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcSendSyncDesc(connIdx, syncIdx, Ref(wrapSyncObj)))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncDesc(syncIdx, wrapSyncObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendSyncDesc_Success}
 * @testcase{21382767}
 * @verify{18723744}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstSendSyncDesc}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendSyncDesc with 'syncIndex' and
 * 'LwSciWrap::syncObj', will internally call Pool::dstSendSyncDesc() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncDesc()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendSyncDesc}
 */
TEST_F (safeconnection_unit_test, dstSendSyncDesc_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncIdx = 0;

    LwSciWrap::SyncObj wrapSyncObj {nullptr};

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstSendSyncDesc(connIdx, syncIdx, Ref(wrapSyncObj)))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncDesc(syncIdx, wrapSyncObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendSyncDesc_failure}
 * @testcase{22034123}
 * @verify{18723744}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstSendSyncDesc}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendSyncDesc with 'syncIndex' and
 * 'LwSciWrap::syncObj', after disconnect will fail to call Pool::dstSendSyncDesc().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncDesc()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendSyncDesc}
 */
TEST_F (safeconnection_unit_test, dstSendSyncDesc_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncIdx = 0;

    LwSciWrap::SyncObj wrapSyncObj {nullptr};

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstSendSyncDesc(connIdx, syncIdx, Ref(wrapSyncObj)))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncDesc(syncIdx, wrapSyncObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacketElementCount_success}
 * @testcase{21382770}
 * @verify{18723867}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendPacketElementCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacketElementCount with element count
 * parameter, will internally call Pool::srcSendPacketElementCount() API with the same parameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketElementCount()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacketElementCount}
 */
TEST_F (safeconnection_unit_test, srcSendPacketElementCount_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemCount = 2;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendPacketElementCount(connIdx, elemCount))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacketElementCount(elemCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacketElementCount_failure}
 * @testcase{22034127}
 * @verify{18723867}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendPacketElementCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacketElementCount with element count
 * parameter, after disconnect will fail to call Pool::srcSendPacketElementCount().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketElementCount()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacketElementCount}
 */
TEST_F (safeconnection_unit_test, srcSendPacketElementCount_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemCount = 2;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcSendPacketElementCount(connIdx, elemCount))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacketElementCount(elemCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendPacketElementCount_success}
 * @testcase{21382773}
 * @verify{18723747}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstSendPacketElementCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendPacketElementCount with element count
 * parameter, will internally call Pool::dstSendPacketElementCount() API with the same parameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendPacketElementCount()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendPacketElementCount}
 */
TEST_F (safeconnection_unit_test, dstSendPacketElementCount_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemCount = 2;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(connIdx, elemCount))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendPacketElementCount(elemCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendPacketElementCount_failure}
 * @testcase{22034131}
 * @verify{18723747}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstSendPacketElementCount}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendPacketElementCount with element count
 * parameter, after disconnect will fail to call Pool::dstSendPacketElementCount().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendPacketElementCount()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendPacketElementCount}
 */
TEST_F (safeconnection_unit_test, dstSendPacketElementCount_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemCount = 2;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstSendPacketElementCount(connIdx, elemCount))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendPacketElementCount(elemCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacketAttr_success}
 * @testcase{21382776}
 * @verify{18723870}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendPacketAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacketAttr with 'elemIndex', 'elemType',
 * 'LwSciStreamElementMode' and 'LwSciWrap::BufAttr', will internally call Pool::srcSendPacketAttr()
 * API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketAttr()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacketAttr}
 */
TEST_F (safeconnection_unit_test, srcSendPacketAttr_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendPacketAttr(connIdx, elemIdx, elemIdx,
                                            LwSciStreamElementMode_Asynchronous,
                                            Ref(wrapBufAttr)))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacketAttr(elemIdx, elemIdx,
                                          LwSciStreamElementMode_Asynchronous,
                                          wrapBufAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacketAttr_failure}
 * @testcase{22034135}
 * @verify{18723870}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendPacketAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacketAttr with 'elemIndex', 'elemType',
 * 'LwSciStreamElementMode' and 'LwSciWrap::BufAttr', after disconnect will fail to call Pool::srcSendPacketAttr().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketAttr()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacketAttr}
 */
TEST_F (safeconnection_unit_test, srcSendPacketAttr_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcSendPacketAttr(connIdx, elemIdx, elemIdx,
                                            LwSciStreamElementMode_Asynchronous,
                                            Ref(wrapBufAttr)))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacketAttr(elemIdx, elemIdx,
                                          LwSciStreamElementMode_Asynchronous,
                                          wrapBufAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendPacketAttr_success}
 * @testcase{21382779}
 * @verify{18723750}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstSendPacketAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendPacketAttr with 'elemIndex', 'elemType',
 * 'LwSciStreamElementMode' and 'LwSciWrap::BufAttr', will internally call Pool::dstSendPacketAttr()
 * API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendPacketAttr()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendPacketAttr}
 */
TEST_F (safeconnection_unit_test, dstSendPacketAttr_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstSendPacketAttr(connIdx, elemIdx, elemIdx,
                                            LwSciStreamElementMode_Asynchronous,
                                            Ref(wrapBufAttr)))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendPacketAttr(elemIdx, elemIdx,
                                          LwSciStreamElementMode_Asynchronous,
                                          wrapBufAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendPacketAttr_failure}
 * @testcase{22058913}
 * @verify{18723750}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstSendPacketAttr}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendPacketAttr with 'elemIndex', 'elemType',
 * 'LwSciStreamElementMode' and 'LwSciWrap::BufAttr', after disconnect will fail to call Pool::dstSendPacketAttr().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendPacketAttr()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendPacketAttr}
 */
TEST_F (safeconnection_unit_test, dstSendPacketAttr_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufAttr wrapBufAttr{rawBufAttrList};

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstSendPacketAttr(connIdx, elemIdx, elemIdx,
                                            LwSciStreamElementMode_Asynchronous,
                                            Ref(wrapBufAttr)))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendPacketAttr(elemIdx, elemIdx,
                                          LwSciStreamElementMode_Asynchronous,
                                          wrapBufAttr);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacketStatus_success}
 * @testcase{21382782}
 * @verify{18723873}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendPacketStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacketStatus with 'LwSciStreamPacket' and
 * 'LwSciError', will internally call Pool::srcSendPacketStatus() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketStatus()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacketStatus}
 */
TEST_F (safeconnection_unit_test, srcSendPacketStatus_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendPacketStatus(connIdx, producerPacket,
                                              producerError))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacketStatus(producerPacket,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacketStatus_failure}
 * @testcase{22034137}
 * @verify{18723873}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendPacketStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacketStatus with 'LwSciStreamPacket' and
 * 'LwSciError', after disconnect will fail to call Pool::srcSendPacketStatus().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketStatus()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacketStatus}
 */
TEST_F (safeconnection_unit_test, srcSendPacketStatus_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcSendPacketStatus(connIdx, producerPacket,
                                              producerError))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacketStatus(producerPacket,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendPacketStatus_success}
 * @testcase{21382788}
 * @verify{18723753}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstSendPacketStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendPacketStatus with 'LwSciStreamPacket' and
 * 'LwSciError', will internally call Pool::dstSendPacketStatus() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketStatus()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendPacketStatus}
 */
TEST_F (safeconnection_unit_test, dstSendPacketStatus_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstSendPacketStatus(connIdx, producerPacket,
                                              producerError))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendPacketStatus(producerPacket,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendPacketStatus_failure}
 * @testcase{22034140}
 * @verify{18723753}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstSendPacketStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendPacketStatus with 'LwSciStreamPacket' and
 * 'LwSciError', after disconnect will fail to call Pool::dstSendPacketStatus().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacketStatus()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendPacketStatus}
 */
TEST_F (safeconnection_unit_test, dstSendPacketStatus_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstSendPacketStatus(connIdx, producerPacket,
                                              producerError))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendPacketStatus(producerPacket,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendElementStatus_success}
 * @testcase{21382791}
 * @verify{18723876}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendElementStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendElementStatus with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciError', will internally call Pool::srcSendElementStatus() API with
 * the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendElementStatus()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendElementStatus}
 */
TEST_F (safeconnection_unit_test, srcSendElementStatus_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;
    uint32_t elemIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendElementStatus(connIdx, producerPacket, elemIdx,
                                              producerError))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendElementStatus(producerPacket, elemIdx,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendElementStatus_failure}
 * @testcase{22034143}
 * @verify{18723876}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendElementStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendElementStatus with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciError', after disconnect will fail to call Pool::srcSendElementStatus().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendElementStatus()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendElementStatus}
 */
TEST_F (safeconnection_unit_test, srcSendElementStatus_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;
    uint32_t elemIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcSendElementStatus(connIdx, producerPacket, elemIdx,
                                              producerError))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendElementStatus(producerPacket, elemIdx,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendElementStatus_success}
 * @testcase{21382794}
 * @verify{18723756}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstSendElementStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendElementStatus with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciError', will internally call Pool::dstSendElementStatus() API with
 * the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendElementStatus()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendElementStatus}
 */
TEST_F (safeconnection_unit_test, dstSendElementStatus_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;
    uint32_t elemIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstSendElementStatus(connIdx, producerPacket, elemIdx,
                                              producerError))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendElementStatus(producerPacket, elemIdx,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstSendElementStatus_failure}
 * @testcase{22034148}
 * @verify{18723756}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstSendElementStatus}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *
 *   The call of BlockWrap-SrcBlockInterface::dstSendElementStatus with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciError', after disconnect will fail to call Pool::dstSendElementStatus().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendElementStatus()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstSendElementStatus}
 */
TEST_F (safeconnection_unit_test, dstSendElementStatus_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    LwSciError producerError = LwSciError_Success;
    uint32_t connIdx = 0;
    uint32_t elemIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstSendElementStatus(connIdx, producerPacket, elemIdx,
                                              producerError))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendElementStatus(producerPacket, elemIdx,
                                            producerError);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacket_success}
 * @testcase{21382797}
 * @verify{18723888}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcSendPacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacket with 'LwSciStreamPacket' and
 * 'FenceArray', will internally call Pool::srcSendPacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacket}
 */
TEST_F (safeconnection_unit_test, srcSendPacket_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    FenceArray emptyFences { };

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendPacket(connIdx, producerPacket, Ref(emptyFences)))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacket(producerPacket,
                                            emptyFences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcSendPacket_failure}
 * @testcase{22034151}
 * @verify{18723888}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcSendPacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcSendPacket with 'LwSciStreamPacket' and
 * 'FenceArray', after disconnect will fail to call Pool::srcSendPacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendPacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcSendPacket}
 */
TEST_F (safeconnection_unit_test, srcSendPacket_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    FenceArray emptyFences { };

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcSendPacket(connIdx, producerPacket, Ref(emptyFences)))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendPacket(producerPacket,
                                            emptyFences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcGetPacket_true}
 * @testcase{21382800}
 * @verify{18723891}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcGetPacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcGetPacket with 'Payload', will return true
 * and internally call Pool::srcGetPacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcGetPacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcGetPacket}
 */
TEST_F (safeconnection_unit_test, srcGetPacket_true) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    Payload emptyPayload {};

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcGetPacket(connIdx, Ref(emptyPayload)))
                       .Times(1)
                       .WillOnce(Return(true));

    EXPECT_EQ(true, connDst.getAccess().srcGetPacket(emptyPayload));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcGetPacket_false}
 * @testcase{21382806}
 * @verify{18723891}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcGetPacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcGetPacket with 'Payload',
 * will return false and internally call Pool::srcGetPacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcGetPacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcGetPacket}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcGetPacket_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    Payload emptyPayload {};

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcGetPacket(connIdx, Ref(emptyPayload)))
                       .Times(1)
                       .WillOnce(Return(false));

    EXPECT_EQ(false, connDst.getAccess().srcGetPacket(emptyPayload));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcGetPacket_failure}
 * @testcase{22034152}
 * @verify{18723891}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcGetPacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcGetPacket with 'Payload',
 * after disconnect will fail to call Pool::srcGetPacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcGetPacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcGetPacket}
 */
TEST_F (safeconnection_unit_test, srcGetPacket_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    Payload emptyPayload {};

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcGetPacket(connIdx, Ref(emptyPayload)))
                       .Times(0)
                       .WillOnce(Return(true));

    connDst.getAccess().srcGetPacket(emptyPayload);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstCreatePacket_success}
 * @testcase{21382812}
 * @verify{18723759}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstCreatePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to producer block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Producer object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstCreatePacket with 'LwSciStreamPacket',
 * will internally call Producer::dstCreatePacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Producer::dstCreatePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstCreatePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstCreatePacket_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(producerPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*producerPtr, dstCreatePacket(connIdx, producerPacket))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstCreatePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));
}


/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstCreatePacket_failure}
 * @testcase{22034157}
 * @verify{18723759}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstCreatePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to producer block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Producer object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstCreatePacket with 'LwSciStreamPacket',
 * after disconnect will fail to call Producer::dstCreatePacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Producer::dstCreatePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstCreatePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstCreatePacket_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(producerPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*producerPtr, dstCreatePacket(connIdx, producerPacket))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstCreatePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcCreatePacket_success}
 * @testcase{21382815}
 * @verify{18723879}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcCreatePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to consumer block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Consumer object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcCreatePacket with 'LwSciStreamPacket',
 * will internally call Consumer::srcCreatePacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Consumer::srcCreatePacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcCreatePacket}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcCreatePacket_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(consumerPtr[0]));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*consumerPtr[0], srcCreatePacket(connIdx, producerPacket))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcCreatePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}


 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcCreatePacket_failure}
 * @testcase{22034158}
 * @verify{18723879}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcCreatePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to consumer block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Consumer object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcCreatePacket with 'LwSciStreamPacket',
 * after disconnect will fail to call Consumer::srcCreatePacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Consumer::srcCreatePacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcCreatePacket}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcCreatePacket_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(consumerPtr[0]));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*consumerPtr[0], srcCreatePacket(connIdx, producerPacket))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcCreatePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstInsertBuffer_success}
 * @testcase{21382818}
 * @verify{18723762}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstInsertBuffer}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to producer block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Producer object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstInsertBuffer with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciWrap::BufObj', will internally call Producer::dstInsertBuffer() API
 * with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Producer::dstInsertBuffer()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstInsertBuffer}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstInsertBuffer_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciBufObj ElementBuf;
    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufObj wrapBufObj{ElementBuf};

    EXPECT_EQ(true, connSrc.connInitiate(producerPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*producerPtr, dstInsertBuffer(connIdx, producerPacket, elemIdx,
                                            Ref(wrapBufObj)))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstInsertBuffer(producerPacket, elemIdx,
                                          wrapBufObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));
}


 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstInsertBuffer_failure}
 * @testcase{22034161}
 * @verify{18723762}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstInsertBuffer}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to producer block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Producer object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstInsertBuffer with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciWrap::BufObj', after disconnect will fail to call Producer::dstInsertBuffer().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Producer::dstInsertBuffer()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstInsertBuffer}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstInsertBuffer_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciBufObj ElementBuf;
    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufObj wrapBufObj{ElementBuf};

    EXPECT_EQ(true, connSrc.connInitiate(producerPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*producerPtr, dstInsertBuffer(connIdx, producerPacket, elemIdx,
                                            Ref(wrapBufObj)))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstInsertBuffer(producerPacket, elemIdx,
                                          wrapBufObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcInsertBuffer_success}
 * @testcase{21382821}
 * @verify{18723882}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcInsertBuffer}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to consumer block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Consumer object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcInsertBuffer with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciWrap::BufObj', will internally call Consumer::srcInsertBuffer() API
 * with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Consumer::srcInsertBuffer()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcInsertBuffer}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcInsertBuffer_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciBufObj ElementBuf;
    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufObj wrapBufObj{ElementBuf};

    EXPECT_EQ(true, connDst.connInitiate(consumerPtr[0]));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*consumerPtr[0], srcInsertBuffer(connIdx, producerPacket, elemIdx,
                                            Ref(wrapBufObj)))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcInsertBuffer(producerPacket, elemIdx,
                                          wrapBufObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}


 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcInsertBuffer_failure}
 * @testcase{22034163}
 * @verify{18723882}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcInsertBuffer}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to consumer block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Consumer object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcInsertBuffer with 'LwSciStreamPacket',
 * 'elemIndex' and 'LwSciWrap::BufObj', after disconnect will fail to call Consumer::srcInsertBuffer().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Consumer::srcInsertBuffer()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcInsertBuffer}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcInsertBuffer_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciBufObj ElementBuf;
    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;
    uint32_t elemIdx = 2;
    LwSciWrap::BufObj wrapBufObj{ElementBuf};

    EXPECT_EQ(true, connDst.connInitiate(consumerPtr[0]));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*consumerPtr[0], srcInsertBuffer(connIdx, producerPacket, elemIdx,
                                            Ref(wrapBufObj)))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcInsertBuffer(producerPacket, elemIdx,
                                          wrapBufObj);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstDeletePacket_success}
 * @testcase{21382830}
 * @verify{18723765}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstDeletePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to producer block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Producer object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstDeletePacket with 'LwSciStreamPacket',
 * will internally call Producer::dstDeletePacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Producer::dstDeletePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstDeletePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstDeletePacket_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(producerPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*producerPtr, dstDeletePacket(connIdx, producerPacket))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstDeletePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));
}


 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstDeletePacket_failure}
 * @testcase{22034166}
 * @verify{18723765}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstDeletePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to producer block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Producer object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstDeletePacket with 'LwSciStreamPacket',
 * after disconnect will fail to call Producer::dstDeletePacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Producer::dstDeletePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstDeletePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstDeletePacket_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(producerPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*producerPtr, dstDeletePacket(connIdx, producerPacket))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstDeletePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&producerPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcDeletePacket_success}
 * @testcase{21382833}
 * @verify{18723885}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcDeletePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to consumer block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Consumer object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::srcDeletePacket with 'LwSciStreamPacket',
 * will internally call Consumer::srcDeletePacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Consumer::srcDeletePacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcDeletePacket}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcDeletePacket_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(consumerPtr[0]));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*consumerPtr[0], srcDeletePacket(connIdx, producerPacket))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcDeletePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}


 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcDeletePacket_failure}
 * @testcase{22034169}
 * @verify{18723885}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcDeletePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to consumer block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Consumer object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::srcDeletePacket with 'LwSciStreamPacket',
 * after disconnect will fail to call Consumer::srcDeletePacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Consumer::srcDeletePacket()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcDeletePacket}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcDeletePacket_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(consumerPtr[0]));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*consumerPtr[0], srcDeletePacket(connIdx, producerPacket))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcDeletePacket(producerPacket);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&consumerPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstReusePacket_success}
 * @testcase{21382836}
 * @verify{18723771}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstReusePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstReusePacket with 'LwSciStreamPacket' and 'FenceArray',
 * will internally call Pool::dstReusePacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstReusePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstReusePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstReusePacket_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    FenceArray emptyFences { };

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstReusePacket(connIdx, producerPacket, Ref(emptyFences)))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstReusePacket(producerPacket,
                                            emptyFences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstReusePacket_failure}
 * @testcase{22034172}
 * @verify{18723771}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstReusePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstReusePacket with 'LwSciStreamPacket' and 'FenceArray',
 * after disconnect will fail to call Pool::dstReusePacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstReusePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstReusePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstReusePacket_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamPacket producerPacket {};
    uint32_t connIdx = 0;

    FenceArray emptyFences { };

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstReusePacket(connIdx, producerPacket, Ref(emptyFences)))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstReusePacket(producerPacket,
                                            emptyFences);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcNotifyConnection_Success}
 * @testcase{21382839}
 * @verify{18723855}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::srcNotifyConnection}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *  The call of BlockWrap-DstBlockInterface::srcNotifyConnection, will internally call
 * Pool::srcNotifyConnection() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcNotifyConnection()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcNotifyConnection}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcNotifyConnection_Success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcNotifyConnection(connIdx))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcNotifyConnection();

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_srcNotifyConnection_failure}
 * @testcase{22034175}
 * @verify{18723855}
 * @testpurpose{Test negative scenario of BlockWrap-DstBlockInterface::srcNotifyConnection}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *  The call of BlockWrap-DstBlockInterface::srcNotifyConnection, after disconnect will
 * fail to call Pool::srcNotifyConnection() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcNotifyConnection()}
 * @verifyFunction{BlockWrap-DstBlockInterface::srcNotifyConnection}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_srcNotifyConnection_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_CALL(*poolPtr, srcNotifyConnection(connIdx))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcNotifyConnection();

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstNotifyConnection_Success}
 * @testcase{21382842}
 * @verify{18723735}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstNotifyConnection}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *  The call of BlockWrap-SrcBlockInterface::dstNotifyConnection, will internally call
 * Pool::dstNotifyConnection() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstNotifyConnection()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstNotifyConnection}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstNotifyConnection_Success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstNotifyConnection(connIdx))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstNotifyConnection();

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}


 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstNotifyConnection_failure}
 * @testcase{22034178}
 * @verify{18723735}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstNotifyConnection}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *  The call of BlockWrap-SrcBlockInterface::dstNotifyConnection, after disconnect will
 * fail to call Pool::dstNotifyConnection() API.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstNotifyConnection()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstNotifyConnection}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstNotifyConnection_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*poolPtr, dstNotifyConnection(connIdx))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstNotifyConnection();

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstAcquirePacket_true}
 * @testcase{21382845}
 * @verify{18723768}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::dstAcquirePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to queue block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Queue object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstAcquirePacket with 'Payload',
 * will return true and internally call Queue::dstAcquirePacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Queue::dstAcquirePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstAcquirePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstAcquirePacket_true)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    Payload emptyPayload {};

    EXPECT_EQ(true, connSrc.connInitiate(queuePtr[0]));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*queuePtr[0], dstAcquirePacket(connIdx, Ref(emptyPayload)))
                       .Times(1)
                       .WillOnce(Return(true));

    EXPECT_EQ(true, connSrc.getAccess().dstAcquirePacket(emptyPayload));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstAcquirePacket_false}
 * @testcase{21382848}
 * @verify{18723768}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstAcquirePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to queue block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Queue object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::dstAcquirePacket with 'Payload',
 * will return false and internally call Queue::dstAcquirePacket() API with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Queue::dstAcquirePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstAcquirePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstAcquirePacket_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    Payload emptyPayload {};

    EXPECT_EQ(true, connSrc.connInitiate(queuePtr[0]));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*queuePtr[0], dstAcquirePacket(connIdx, Ref(emptyPayload)))
                       .Times(1)
                       .WillOnce(Return(false));

    EXPECT_EQ(false, connSrc.getAccess().dstAcquirePacket(emptyPayload));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr));
}


 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_dstAcquirePacket_failure}
 * @testcase{22034181}
 * @verify{18723768}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::dstAcquirePacket}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to queue block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Queue object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::dstAcquirePacket with 'Payload',
 * after disconnect will fail to call Queue::dstAcquirePacket().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Queue::dstAcquirePacket()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::dstAcquirePacket}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_dstAcquirePacket_failure)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    Payload emptyPayload {};

    EXPECT_EQ(true, connSrc.connInitiate(queuePtr[0]));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_CALL(*queuePtr[0], dstAcquirePacket(connIdx, Ref(emptyPayload)))
                       .Times(0)
                       .WillOnce(Return(true));

    connSrc.getAccess().dstAcquirePacket(emptyPayload);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr));
}
 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_connInitiate_true}
 * @testcase{21382851}
 * @verify{18723636}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::connInitiate }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *
 * SafeConnection-SrcBlockInterface::connInitiate returns true, when connection to pool block
 * is initiated by passing valid Pool object pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::connInitiate }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_connInitiate_true)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_connInitiate_true}
 * @testcase{21382854}
 * @verify{18723636}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::connInitiate }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *
 * SafeConnection-DstBlockInterface::connInitiate returns true, when connection to pool block
 * is initiated by passing valid Pool object pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::connInitiate }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_connInitiate_true)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_EQ(true, connSrc.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_connInitiate_false}
 * @testcase{21382857}
 * @verify{18723636}
 * @testpurpose{Test negative scenario of SafeConnection-SrcBlockInterface::connInitiate }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *
 * SafeConnection-SrcBlockInterface::connInitiate returns false, when connection initiation
 * is attempted again for the same SrcConnection object.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::connInitiate }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_connInitiate_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    EXPECT_EQ(false, connSrc.connInitiate(poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_connInitiate_false}
 * @testcase{21382860}
 * @verify{18723636}
 * @testpurpose{Test negative scenario of SafeConnection-DstBlockInterface::connInitiate }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *
 * SafeConnection-DstBlockInterface::connInitiate returns false, when connection initiation
 * is attempted with paramBlockPtr set to NULL.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::connInitiate }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_connInitiate_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(false, connDst.connInitiate(NULL));
}

  /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_connComplete_success}
 * @testcase{21382863}
 * @verify{18724446}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::connComplete }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *
 *  Completes connection by calling connComplete() API of SrcConnection,
 *  with argument paramConnIndex as 0.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::connComplete }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_connComplete_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_EQ(true, connSrc.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_connComplete_success}
 * @testcase{21382869}
 * @verify{18724446}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::connComplete }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *
 *  Completes connection by calling connComplete() API of DstConnection,
 *  with argument paramConnIndex as 0.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::connComplete }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_connComplete_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_EQ(true, connDst.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_connCancel_success}
 * @testcase{21382872}
 * @verify{18724452}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::connCancel }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *
 *  Cancels connection by calling connCancel() API of SrcConnection. The call of
 * SafeConnection-SrcBlockInterface::connInitiate returns true, when connection to pool block
 * is initiated again by passing valid Pool object pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::connCancel }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_connCancel_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connCancel();

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_connCancel_success}
 * @testcase{21382875}
 * @verify{18724452}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::connCancel }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *
 *  Cancels connection by calling connCancel() API of DstConnection. The call of
 * SafeConnection-DstBlockInterface::connInitiate returns true, when connection to pool block
 * is initiated again by passing valid Pool object pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::connCancel }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_connCancel_success)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connCancel();

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_disconnect_success}
 * @testcase{21382878}
 * @verify{18724455}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::disconnect(). }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *  Call of SafeConnection-DstBlockInterface::disconnect() API disconnects the
 * connection and calling SafeConnection-DstBlockInterface::isConnected()  return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{BlockWrap::disconnectBlock()}
 * @verifyFunction{SafeConnection-DstBlockInterface::disconnect() }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_disconnect_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    connSrc.disconnect();

    EXPECT_EQ(false, connSrc.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_disconnect_success}
 * @testcase{21382881}
 * @verify{18724455}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::disconnect(). }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *  Call of SafeConnection-SrcBlockInterface::disconnect() API disconnects the
 * connection and calling of SafeConnection-SrcBlockInterface::isConnected()
 * should return false. }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{BlockWrap::disconnectBlock()}
 * @verifyFunction{SafeConnection-SrcBlockInterface::disconnect() }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_disconnect_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    connDst.disconnect();

    EXPECT_EQ(false, connDst.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_reserve_true}
 * @testcase{21382884}
 * @verify{18724461}
 * @verify{18723648}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::reserve }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 * SafeConnection-SrcBlockInterface::reserve returns true, when SafeConnection is
 * reserved successfully.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::reserve }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_reserve_true) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_EQ(true, connSrc.reserve());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_reserve_true}
 * @testcase{21382887}
 * @verify{18724461}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::reserve}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 * SafeConnection-DstBlockInterface::reserve returns true, when SafeConnection is
 * reserved successfully.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::reserve }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_reserve_true) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_EQ(true, connDst.reserve());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_reserve_false}
 * @testcase{21382890}
 * @verify{18724461}
 * @testpurpose{Test negative scenario of SafeConnection-SrcBlockInterface::reserve}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 * SafeConnection-SrcBlockInterface::reserve returns false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::reserve }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_reserve_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.disconnect();

    EXPECT_EQ(false, connSrc.reserve());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_reserve_false}
 * @testcase{21382893}
 * @verify{18724461}
 * @testpurpose{Test negative scenario of SafeConnection-DstBlockInterface::reserve}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 * SafeConnection-DstBlockInterface::reserve returns false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::reserve }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_reserve_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.disconnect();

    EXPECT_EQ(false, connDst.reserve());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_isConnected_true}
 * @testcase{21382896}
 * @verify{18724458}
 * @verify{18723648}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::isConnected }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 * SafeConnection-SrcBlockInterface::isConnected returns true, when SafeConnection is connected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::isConnected }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_isConnected_true) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_EQ(true, connSrc.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_isConnected_true}
 * @testcase{21382899}
 * @verify{18724458}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::isConnected }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 * SafeConnection-DstBlockInterface::isConnected returns true, when SafeConnection is connected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::isConnected }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_isConnected_true) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_EQ(true, connDst.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

  /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_isConnected_false}
 * @testcase{21382905}
 * @verify{18724458}
 * @testpurpose{Test negative scenario of SafeConnection-SrcBlockInterface::isConnected when
 * SafeConnection is not connected.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Note that the SrcConnection is not initialized.
 *
 * SafeConnection-SrcBlockInterface::isConnected returns false, when SafeConnection is
 * not connected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::isConnected }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_isConnected_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    EXPECT_EQ(false, connSrc.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

  /**
 * @testname{safeconnection_unit_test.DstBlockInterface_isConnected_false}
 * @testcase{21382911}
 * @verify{18724458}
 * @testpurpose{Test negative scenario of SafeConnection-DstBlockInterface::isConnected when
 * SafeConnection is not connected.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Note that the SrcConnection is not initialized.
 *
 * SafeConnection-DstBlockInterface::isConnected returns false, when SafeConnection is
 * not connected.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::isConnected }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_isConnected_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    EXPECT_EQ(false, connDst.isConnected());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_release_success}
 * @testcase{21382914}
 * @verify{18724464}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::release }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0.
 *   5.Reserve connection by calling reserve() API of DstConnection.
 *   6.Call the SafeConnection disconnect() API.
 *
 * Releases connection by calling release() API of SrcConnection, which ilwokes disconnectBlock()
 * interface of the BlockWrap.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *     - BlockWrap::disconnectBlock().}
 * @verifyFunction{SafeConnection-DstBlockInterface::release }
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_release_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    LwSciStreamEvent event;
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_EQ(true, connDst.reserve());

    connDst.disconnect();

    connDst.release();

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                           pool, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_release_success}
 * @testcase{21382917}
 * @verify{18724464}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::release }
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0.
 *   5.Reserve connection by calling reserve() API of SrcConnection.
 *   6.Call the SafeConnection disconnect() API.
 *
 * Releases connection by calling release() API of SrcConnection, which ilwokes disconnectBlock()
 * interface of the BlockWrap.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *     - BlockWrap::disconnectBlock().}
 * @verifyFunction{SafeConnection-SrcBlockInterface::release }
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_release_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    LwSciStreamEvent event;
    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_EQ(true, connSrc.reserve());

    connSrc.disconnect();

    connSrc.release();

    EXPECT_EQ(LwSciError_Success, LwSciStreamBlockEventQuery(
                           pool, EVENT_QUERY_TIMEOUT, &event));
    EXPECT_EQ(LwSciStreamEventType_Disconnected, event.type);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_disconnectBlock_success}
 * @testcase{21382920}
 * @verify{18723774}
 * @testpurpose{Test success scenario of BlockWrap-SrcBlockInterface::disconnectBlock}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-SrcBlockInterface::disconnectBlock with rawPtr of pool blkPtr.get() and connIdx,
 * notifies the pool block about disconnect by calling Pool::dstDisconnect() API
 * with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstDisconnect()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::disconnectBlock}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_disconnectBlock_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstDisconnect(connIdx))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().disconnectBlock(poolPtr, connIdx);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_disconnectBlock_failure}
 * @testcase{22058907}
 * @verify{18723774}
 * @testpurpose{Test negative scenario of BlockWrap-SrcBlockInterface::disconnectBlock}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of SrcConnection
 *
 *   The call of BlockWrap-SrcBlockInterface::disconnectBlock with nullptr and connIdx,
 * fail to call Pool::dstDisconnect() API.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstDisconnect()}
 * @verifyFunction{BlockWrap-SrcBlockInterface::disconnectBlock}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_disconnectBlock_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstDisconnect(connIdx))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().disconnectBlock(nullptr, connIdx);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_disconnectBlock_success}
 * @testcase{21382923}
 * @verify{18723894}
 * @testpurpose{Test success scenario of BlockWrap-DstBlockInterface::disconnectBlock}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to queue block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Queue object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *
 *   The call of BlockWrap-DstBlockInterface::disconnectBlock with rawPtr of queue blkPtr.get() and connIdx,
 * notifies the queue block about disconnect by calling Queue::srcDisconnect() API
 * with the same parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Queue::srcDisconnect()}
 * @verifyFunction{BlockWrap-DstBlockInterface::disconnectBlock}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_disconnectBlock_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(queuePtr[0]));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*queuePtr[0], srcDisconnect(connIdx))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().disconnectBlock(queuePtr[0], connIdx);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_disconnectBlock_failure}
 * @testcase{22058909}
 * @verify{18723894}
 * @testpurpose{Test success negative of BlockWrap-DstBlockInterface::disconnectBlock}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to queue block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Queue object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0
 *   5.Disconnects connection by calling disconnect() API of DstConnection
 *
 *   The call of BlockWrap-DstBlockInterface::disconnectBlock with nullptr and connIdx,
 * fail to call Queue::srcDisconnect() API.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Queue::srcDisconnect()}
 * @verifyFunction{BlockWrap-DstBlockInterface::disconnectBlock}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_disconnectBlock_failure) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(queuePtr[0]));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*queuePtr[0], srcDisconnect(connIdx))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().disconnectBlock(nullptr, connIdx);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&queuePtr));
}

  /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_pathCompleteSet_success}
 * @testcase{21382929}
 * @verify{18724476}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::pathCompleteSet}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0.
 *
 * Marks connection path to the respective endpoint (producer) is established by calling
 * pathCompleteSet() API of SrcConnection.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::pathCompleteSet}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_pathCompleteSet_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.pathCompleteSet();

    EXPECT_EQ(true, connSrc.pathCompleteGet());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_pathCompleteSet_success}
 * @testcase{21382932}
 * @verify{18724476}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::pathCompleteSet}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0.
 *
 * Marks connection path to the respective endpoint (consumer) is established by calling
 * pathCompleteSet() API of DstConnection.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::pathCompleteSet}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_pathCompleteSet_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.pathCompleteSet();

    EXPECT_EQ(true, connDst.pathCompleteGet());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

/**
 * @testname{safeconnection_unit_test.DstBlockInterface_pathCompleteGet_true}
 * @testcase{21382938}
 * @verify{18724479}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::pathCompleteGet,
 * when path is completed}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0.
 *   5.Marks connection path to the respective endpoint (consumer) is established by calling
 *   pathCompleteSet() API of DstConnection.
 *
 *  SafeConnection-DstBlockInterface::pathCompleteGet returns true}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::pathCompleteGet}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_pathCompleteGet_true) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.pathCompleteSet();

    EXPECT_EQ(true, connDst.pathCompleteGet());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_pathCompleteGet_true}
 * @testcase{21382941}
 * @verify{18724479}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::pathCompleteGet,
 * when path is completed}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0.
 *   5.Marks connection path to the respective endpoint (producer) is established by calling
 *   pathCompleteSet() API of SrcConnection.
 *
 *  SafeConnection-SrcBlockInterface::pathCompleteGet returns true}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::pathCompleteGet}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_pathCompleteGet_true) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.pathCompleteSet();

    EXPECT_EQ(true, connSrc.pathCompleteGet());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_pathCompleteGet_false}
 * @testcase{21382944}
 * @verify{18724479}
 * @testpurpose{Test negative scenario of SafeConnection-DstBlockInterface::pathCompleteGet,
 * when pathCompleteGet() is called when pathCompleteSet() is not yet called.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0.
 *
 *  SafeConnection-DstBlockInterface::pathCompleteGet returns false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::pathCompleteGet}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_pathCompleteGet_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_EQ(false, connDst.pathCompleteGet());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_pathCompleteGet_false}
 * @testcase{21382947}
 * @verify{18724479}
 * @testpurpose{Test negative scenario of SafeConnection-SrcBlockInterface::pathCompleteGet,
 * when pathCompleteGet() is called when pathCompleteSet() is not yet called.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0.
 *
 *  SafeConnection-SrcBlockInterface::pathCompleteGet returns false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::pathCompleteGet}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_pathCompleteGet_false) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);

    uint32_t connIdx = 0;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_EQ(false, connSrc.pathCompleteGet());

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_getAccess_success}
 * @testcase{21382953}
 * @verify{18723639}
 * @testpurpose{Test success scenario of SafeConnection-SrcBlockInterface::getAccess,
 * when path is completed}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0.
 *
 *  SafeConnection-SrcBlockInterface::getAccess returns new SafeConnection::Access
 * object. The call of dstSendSyncCount API from the SafeConnection::Access object, with sync
 * objects count parameter will internally call Pool::dstSendSyncCount() API with the same
 * parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncCount()}
 * @verifyFunction{SafeConnection-SrcBlockInterface::getAccess}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_getAccess_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, dstSendSyncCount(connIdx, syncCount))
                       .Times(1)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_getAccess_failure1}
 * @testcase{21709147}
 * @verify{18723639}
 * @testpurpose{Test negative scenario of SafeConnection-SrcBlockInterface::getAccess,
 * when connection is cancelled.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0.
 *   5.Cancels connection by calling Cancel() API of SrcConnection.
 *
 *  SafeConnection-SrcBlockInterface::getAccess fails to return new SafeConnection::Access
 * object. The call of dstSendSyncCount API from the SafeConnection::Access object, with sync
 * objects count parameter, will fail to call Pool::dstSendSyncCount() after cancels the connection.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncCount()}
 * @verifyFunction{SafeConnection-SrcBlockInterface::getAccess}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_getAccess_failure1) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    connSrc.connComplete(connIdx);

    connSrc.connCancel();

    EXPECT_CALL(*poolPtr, dstSendSyncCount(connIdx, syncCount))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_getAccess_failure2}
 * @testcase{21709148}
 * @verify{18723639}
 * @testpurpose{Test negative scenario of SafeConnection-SrcBlockInterface::getAccess,
 * when connection not completed.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *
 *  SafeConnection-SrcBlockInterface::getAccess fails to return new SafeConnection::Access
 * object. The call of dstSendSyncCount API from the SafeConnection::Access object, with sync
 * objects count parameter, will fail to call Pool::dstSendSyncCount() when safeconnection is not completed.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::dstSendSyncCount()}
 * @verifyFunction{SafeConnection-SrcBlockInterface::getAccess}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_getAccess_failure2) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    SrcConnection connSrc;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connSrc.connInitiate(poolPtr));

    EXPECT_CALL(*poolPtr, dstSendSyncCount(connIdx, syncCount))
                       .Times(0)
                       .WillOnce(Return());

    connSrc.getAccess().dstSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_getAccess_success}
 * @testcase{21382959}
 * @verify{18723639}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::getAccess,
 * when path is completed}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0.
 *
 *  SafeConnection-DstBlockInterface::getAccess returns new SafeConnection::Access
 * object. The call of srcSendSyncCount API from the SafeConnection::Access object, with sync
 * objects count parameter will internally call Pool::srcSendSyncCount() API with the same
 * parameters.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncCount()}
 * @verifyFunction{SafeConnection-DstBlockInterface::getAccess}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_getAccess_success) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    EXPECT_CALL(*poolPtr, srcSendSyncCount(connIdx, syncCount))
                       .Times(1)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_getAccess_failure1}
 * @testcase{21709150}
 * @verify{18723639}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::getAccess,
 * when connection is cancelled.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0.
 *   5.Cancels connection by calling Cancel() API of DstConnection.
 *
 *  SafeConnection-DstBlockInterface::getAccess fail to return new SafeConnection::Access
 * object. The call of srcSendSyncCount API from the SafeConnection::Access object, with sync
 * objects count parameter, will fail to call Pool::srcSendSyncCount() after cancels the connection.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncCount()}
 * @verifyFunction{SafeConnection-DstBlockInterface::getAccess}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_getAccess_failure1) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    connDst.connComplete(connIdx);

    connDst.connCancel();

    EXPECT_CALL(*poolPtr, srcSendSyncCount(connIdx, syncCount))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_getAccess_failure2}
 * @testcase{21709151}
 * @verify{18723639}
 * @testpurpose{Test success scenario of SafeConnection-DstBlockInterface::getAccess,
 * when connection not completed.}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *
 *  SafeConnection-DstBlockInterface::getAccess fail to return new SafeConnection::Access
 * object. The call of srcSendSyncCount API from the SafeConnection::Access object, with sync
 * objects count parameter, will fail to call Pool::srcSendSyncCount()  when safeconnection is not completed.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - Pool::srcSendSyncCount()}
 * @verifyFunction{SafeConnection-DstBlockInterface::getAccess}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_getAccess_failure2) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;
    using ::testing::Ref;

    DstConnection connDst;
    createBlocks(QueueType::Mailbox);
    uint32_t connIdx = 0;
    uint32_t syncCount = 2;

    EXPECT_EQ(true, connDst.connInitiate(poolPtr));

    EXPECT_CALL(*poolPtr, srcSendSyncCount(connIdx, syncCount))
                       .Times(0)
                       .WillOnce(Return());

    connDst.getAccess().srcSendSyncCount(syncCount);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&poolPtr));
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_Access_success}
 * @testcase{21709152}
 * @verify{18723651}
 * @verify{18723654}
 * @testpurpose{Test positive scenario of Move constructor SafeConnection-DstBlockInterface::Access::Access}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'DstConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of DstConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of DstConnection,
 *   with argument paramConnIndex as 0.
 *
 *  SafeConnection-DstBlockInterface::Access::Access moves the parent SafeConnection instance
 * and the content of BlockWrap instance from input Access object to the new Access object.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::Access::Access}}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_Access_success) {
   using ::testing::_;
   using ::testing::Return;
   using ::testing::Mock;
   using ::testing::Ref;

   DstConnection connDst;
   createBlocks(QueueType::Mailbox);

   uint32_t connIdx = 0;
   uint32_t syncCount = 2;

   EXPECT_EQ(true, connDst.connInitiate(poolPtr));
   connDst.connComplete(connIdx);

   std::vector <SafeConnection<DstBlockInterface>::Access> vec;
   vec.push_back(connDst.getAccess());
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_Access_success}
 * @testcase{21709153}
 * @verify{18723651}
 * @testpurpose{Test positive scenario of Move constructor SafeConnection-SrcBlockInterface::Access::Access}
 * @testbehavior{
 * Setup:
 *   1.Creates safe connection object of type 'SrcConnection'
 *   2.Create the producer, pool, queue and consumer blocks.
 *   3.Connection to pool block is instantiated through connInitiate() API of SrcConnection,
 *   by passing valid Pool object pointer.
 *   4.Completes connection by calling connComplete() API of SrcConnection,
 *   with argument paramConnIndex as 0.
 *
 *  SafeConnection-SrcBlockInterface::Access::Access moves the parent SafeConnection instance
 * and the content of BlockWrap instance from input Access object to the new Access object.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::Access::Access}}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_Access_success) {
   using ::testing::_;
   using ::testing::Return;
   using ::testing::Mock;
   using ::testing::Ref;

   SrcConnection connSrc;
   createBlocks(QueueType::Mailbox);

   uint32_t connIdx = 0;
   uint32_t syncCount = 2;

   EXPECT_EQ(true, connSrc.connInitiate(poolPtr));
   connSrc.connComplete(connIdx);

   std::vector <SafeConnection<SrcBlockInterface>::Access> vec;
   vec.push_back(connSrc.getAccess());
}

 /**
 * @testname{safeconnection_unit_test.DstBlockInterface_Access_failure}
 * @testcase{21709154}
 * @verify{18723648}
 * @verify{18723654}
 * @testpurpose{Test positive scenario of Parameterized constructor
 * SafeConnection-DstBlockInterface::Access::Access}
 * @testbehavior{
 *
 *  SafeConnection-DstBlockInterface::Access performs no operation when called with parameters
 * paramConn set to NULL,paramPtr set to NULL and paramConnIndex set to 0.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-DstBlockInterface::Access::Access}}
 */
TEST_F (safeconnection_unit_test, DstBlockInterface_Access_failure) {
   SafeConnection<DstBlockInterface>::Access access{NULL, NULL, 0};
}

 /**
 * @testname{safeconnection_unit_test.SrcBlockInterface_Access_failure}
 * @testcase{21709155}
 * @verify{18723648}
 * @testpurpose{Test negative scenario of Parameterized constructor
 * SafeConnection-SrcBlockInterface::Access::Access}
 * @testbehavior{
 *
 *  SafeConnection-SrcBlockInterface::Access performs no operation when called with parameters
 * paramConn set to NULL,paramPtr set to NULL and paramConnIndex set to 0.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly in target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{SafeConnection-SrcBlockInterface::Access::Access}}
 */
TEST_F (safeconnection_unit_test, SrcBlockInterface_Access_failure) {
   SafeConnection<SrcBlockInterface>::Access access{NULL, NULL, 0};
}

}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

