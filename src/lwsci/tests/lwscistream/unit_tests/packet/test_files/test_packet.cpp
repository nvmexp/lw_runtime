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

#include "packet.h"
#include "lwscistream_common.h"
#include "sciwrap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include "lwscistream_LwSciCommonPanic_mock.h"

class packet_unit_test: public LwSciStreamTest {
public:
   packet_unit_test( ) {
       // initialization code here
   }

   void SetUp( ) {
       // code here will execute just before the test ensues
   }

   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~packet_unit_test( )  {
       // cleanup any pending stuff, but no exceptions and no gtest
       // ASSERT* allowed.
   }

   // put in any custom data members that you need
};

namespace LwSciStream {

/**
 * @testname{packet_unit_test.handleGet_Success}
 * @testcase{21608147}
 * @verify{20040225}
 * @testpurpose{Test success scenario of Packet::handleGet(), when
 *  LwSciStreamPacket associated with the packet instance is retrieved.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::handleGet() API from packet object should
 * return a LwSciStreamPacket.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::handleGet()}
 */
TEST_F (packet_unit_test, handleGet_Success)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(packetObj->handleGet(), packetHandle);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.cookieGet_Success}
 * @testcase{21608148}
 * @verify{20040228}
 * @testpurpose{Test success scenario of Packet::cookieGet(), when
 *  LwSciStreamCookie associated with the packet instance is retrieved.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::cookieGet() API from packet object should
 * return a LwSciStreamCookie.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::cookieGet()}
 */
TEST_F (packet_unit_test, cookieGet_Success)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(packetObj->cookieGet(), poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineAction_StreamInternalError1}
 * @testcase{21608149}
 * @verify{20040243}
 * @testpurpose{Test negative scenario of Packet::bufferDefineAction() when
 *  the packet instance has no tracker of LwSciBufObj for the packet element
 *  (Packet::Desc::defineActionCount is given as 0).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineActionCount set to 0 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineAction() API from packet object with elemIndex value as 0 and
 *  function should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineAction()}
 */
TEST_F (packet_unit_test, bufferDefineAction_StreamInternalError1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->bufferDefineAction(0, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineAction_StreamInternalError2}
 * @testcase{21608150}
 * @verify{20040243}
 * @testpurpose{Test negative scenario of Packet::bufferDefineAction() when
 *  the packet instance has more than one tracker of LwSciBufObj for the packet element
 *  (Packet::Desc::defineActionCount is given greater than 1).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineActionCount set to 2 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineAction() API from packet object with elemIndex value as 0 and
 *  function should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement, Equivalence classes and Boundary values.}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineAction()}
 */
TEST_F (packet_unit_test, bufferDefineAction_StreamInternalError2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineActionCount = 2U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->bufferDefineAction(0, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineAction_BadParameter}
 * @testcase{21608151}
 * @verify{20040243}
 * @testpurpose{Test negative scenario of Packet::bufferDefineAction() when
 *  elemIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineActionCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineAction() API from packet object with elemIndex value as 2 and
 *  function should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineAction()}
 */
TEST_F (packet_unit_test, bufferDefineAction_BadParameter)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_BadParameter, packetObj->bufferDefineAction(2, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineAction_IlwalidState}
 * @testcase{21608152}
 * @verify{20040243}
 * @testpurpose{Test negative scenario of Packet::bufferDefineAction() when
 *  LwSciBufObj for the given elemIndex is already sent.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineActionCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Packet::bufferDefineAction() is ilwoked with elemIndex value as 0 and function.
 *
 *   The call of Packet::bufferDefineAction() API from packet object again when elemIndex value as
 * 0 and function should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineAction()}
 */
TEST_F (packet_unit_test, bufferDefineAction_IlwalidState)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->bufferDefineAction(0, action));

    EXPECT_TRUE(actionPerformed);

    /* test code */
    EXPECT_EQ(LwSciError_IlwalidState, packetObj->bufferDefineAction(0, action));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineAction_Success}
 * @testcase{21608153}
 * @verify{20040243}
 * @testpurpose{Test success scenario of Packet::bufferDefineAction() when
 *  the given function is ilwoked successfully.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineActionCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineAction() API from packet object with elemIndex value as 0 and
 *  function should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineAction()}
 */
TEST_F (packet_unit_test, bufferDefineAction_Success)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->defineActionCount = 1U;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->bufferDefineAction(0, action));

    EXPECT_TRUE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineEventPrepare_StreamInternalError1}
 * @testcase{21608154}
 * @verify{20040246}
 * @testpurpose{Test negative scenario of Packet::bufferDefineEventPrepare() when
 *  packet element count tracked by elemCount object is 0(Packet::Desc::defineEventCount
 *  is given as 0).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineEventCount set to 0 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineEventPrepare() API from packet object with elemIndex value
 *   as 0 and LwSciWrap::BufObj should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineEventPrepare()}
 */
TEST_F (packet_unit_test, bufferDefineEventPrepare_StreamInternalError1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciBufObj poolElementBuf;
    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->bufferDefineEventPrepare(0, wrapBufObj));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineEventPrepare_StreamInternalError2}
 * @testcase{21608156}
 * @verify{20040246}
 * @testpurpose{Test negative scenario of Packet::bufferDefineEventPrepare() when
 *  packet element count tracked by elemCount object is 2
 *  (Packet::Desc::defineEventCount is given as greater than 1).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineEventCount set to 2 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineEventPrepare() API from packet object with elemIndex value
 *  as 0 and LwSciWrap::BufObj should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineEventPrepare()}
 */
TEST_F (packet_unit_test, bufferDefineEventPrepare_StreamInternalError2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 2U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciBufObj poolElementBuf;
    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->bufferDefineEventPrepare(0, wrapBufObj));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineEventPrepare_BadParameter}
 * @testcase{21608157}
 * @verify{20040246}
 * @testpurpose{Test negative scenario of Packet::bufferDefineEventPrepare() when
 *  elemIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineEventPrepare() API from packet object when elemIndex value is
 *  equal to packet element count tracked by elemCount object should return
 *  LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineEventPrepare()}
 */
TEST_F (packet_unit_test, bufferDefineEventPrepare_BadParameter)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciBufObj poolElementBuf;
    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    /* test code */
    EXPECT_EQ(LwSciError_BadParameter, packetObj->bufferDefineEventPrepare(1, wrapBufObj));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineEventPrepare_IlwalidState}
 * @testcase{21608158}
 * @verify{20040246}
 * @testpurpose{Test negative scenario of Packet::bufferDefineEventPrepare() when
 *  LwSciBufObj for the elemIndex is already received.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Packet::bufferDefineEventPrepare() is ilwoked with elemIndex value as 0 and
 *      LwSciWrap::BufObj.
 *
 *   The call of Packet::bufferDefineAction() API from packet object again with elemIndex value
 *  as 0 and LwSciWrap::BufObj should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineEventPrepare()}
 */
TEST_F (packet_unit_test, bufferDefineEventPrepare_IlwalidState)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciBufObj poolElementBuf;
    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    EXPECT_EQ(LwSciError_Success, packetObj->bufferDefineEventPrepare(0, wrapBufObj));

    /* test code */
    EXPECT_EQ(LwSciError_IlwalidState, packetObj->bufferDefineEventPrepare(0, wrapBufObj));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferDefineEventPrepare_Success}
 * @testcase{21608159}
 * @verify{20040246}
 * @testpurpose{Test success scenario of Packet::bufferDefineEventPrepare() when
 *  the LwSciBufObj for the element is set successfully.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferDefineEventPrepare() API from packet object with elemIndex
 *  value as 0 and LwSciWrap::BufObj should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferDefineEventPrepare()}
 */
TEST_F (packet_unit_test, bufferDefineEventPrepare_Success)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciBufObj poolElementBuf;
    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->bufferDefineEventPrepare(0, wrapBufObj));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusAction_StreamInternalError1}
 * @testcase{21608160}
 * @verify{20040249}
 * @testpurpose{Test negative scenario of Packet::packetStatusAction() when
 *  the packet instance doesn't support tracking of packet acceptance status
 *  (Packet::Desc::statusActionCount is given as 0).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 0 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::packetStatusAction() API from packet object with the given
 *  LwSciStreamCookie and function should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusAction()}
 */
TEST_F (packet_unit_test, packetStatusAction_StreamInternalError1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->packetStatusAction(poolCookie, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusAction_StreamInternalError2}
 * @testcase{21608161}
 * @verify{20040249}
 * @testpurpose{Test negative scenario of Packet::packetStatusAction() when
 *  the packet instance has more than one tracking of packet acceptance status
 *  (Packet::Desc::statusActionCount is given as greater than 1).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 2 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::packetStatusAction() API from packet object with the given
 *  LwSciStreamCookie and function should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusAction()}
 */
TEST_F (packet_unit_test, packetStatusAction_StreamInternalError2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusActionCount = 2U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->packetStatusAction(poolCookie, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusAction_IlwalidState}
 * @testcase{21608162}
 * @verify{20040249}
 * @testpurpose{Test negative scenario of Packet::packetStatusAction() when
 *  the packet acceptance status is already sent.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusActionCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Packet::packetStatusAction() is ilwoked with given LwSciStreamCookie and function.
 *
 *   The call of Packet::packetStatusAction() API from packet object again with the given
 *  LwSciStreamCookie and function should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusAction()}
 */
TEST_F (packet_unit_test, packetStatusAction_IlwalidState)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(poolCookie, action));

    /* test code */
    EXPECT_EQ(LwSciError_IlwalidState, packetObj->packetStatusAction(poolCookie, action));

    EXPECT_TRUE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusAction_Success1}
 * @testcase{22060013}
 * @verify{20040249}
 * @testpurpose{Test success scenario of Packet::packetStatusAction() when
 *  the packet acceptance status (accepted) is successfully sent.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::packetStatusAction() API from packet object with the given
 *  valid LwSciStreamCookie and function should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusAction()}
 */
TEST_F (packet_unit_test, packetStatusAction_Success1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    pktDesc->statusActionCount = 1U;
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(poolCookie, action));

    EXPECT_TRUE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusAction_Success2}
 * @testcase{21664569}
 * @verify{20040249}
 * @testpurpose{Test success scenario of Packet::packetStatusAction() when
 *  the packet acceptance status (rejected) is successfully sent.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::packetStatusAction() API from packet object with the given
 *  invalid LwSciStreamCookie and function should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusAction()}
 */
TEST_F (packet_unit_test, packetStatusAction_Success2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = LwSciStreamCookie_Ilwalid;

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    pktDesc->statusActionCount = 1U;
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(poolCookie, action));

    EXPECT_TRUE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusAction_StreamInternalError1}
 * @testcase{21608165}
 * @verify{20040252}
 * @testpurpose{Test negative scenario of Packet::bufferStatusAction() when
 *  the packet instance doesn't support tracking of packet element acceptance status
 *  (Packet::Desc::statusActionCount was given as 0).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 0 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferStatusAction() API from packet object with elemIndex value as 0
 *  and function should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusAction()}
 */
TEST_F (packet_unit_test, bufferStatusAction_StreamInternalError1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->bufferStatusAction(0, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusAction_StreamInternalError2}
 * @testcase{21608166}
 * @verify{20040252}
 * @testpurpose{Test negative scenario of Packet::bufferStatusAction() when
 *  packet instance has more than one tracking of packet element acceptance status
 *  (Packet::Desc::statusActionCount is greater than 1).}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 2 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferStatusAction() API from packet object with elemIndex value as 0
 *  and function should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusAction()}
 */
TEST_F (packet_unit_test, bufferStatusAction_StreamInternalError2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusActionCount = 2U;


    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->bufferStatusAction(0, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusAction_BadParameter}
 * @testcase{21608167}
 * @verify{20040252}
 * @testpurpose{Test negative scenario of Packet::bufferStatusAction() when
 *  elemIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferStatusAction() API from packet object with elemIndex value equal
 *  or greater than packet element count tracked by elemCount object and function
 *  should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusAction()}
 */
TEST_F (packet_unit_test, bufferStatusAction_BadParameter)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_BadParameter, packetObj->bufferStatusAction(1, action));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusAction_IlwalidState}
 * @testcase{21608168}
 * @verify{20040252}
 * @testpurpose{Test negative scenario of Packet::bufferStatusAction() when
 * Packet::bufferStatusAction() is ilwoked more than once.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusActionCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Packet::bufferStatusAction() is ilwoked with elemIndex value as 0 and function.
 *
 *   The call of Packet::bufferStatusAction() API from packet object again with the same elemIndex
 *  and function should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusAction()}
 */
TEST_F (packet_unit_test, bufferStatusAction_IlwalidState)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusAction(0, action));

    /* test code */
    EXPECT_EQ(LwSciError_IlwalidState, packetObj->bufferStatusAction(0, action));

    EXPECT_TRUE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusAction_Success}
 * @testcase{21608169}
 * @verify{20040252}
 * @testpurpose{Test success scenario of Packet::bufferStatusAction() when
 *  the given function is ilwoked successfully.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferStatusAction() API from packet object with elemIndex value as 0 and
 *  function should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusAction()}
 */
TEST_F (packet_unit_test, bufferStatusAction_Success)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    pktDesc->statusActionCount = 1U;
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusAction(0, action));

    EXPECT_TRUE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusEventPrepare_StreamInternalError}
 * @testcase{21608170}
 * @verify{20040255}
 * @testpurpose{Test negative scenario of Packet::packetStatusEventPrepare(), where
 *  endIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::packetStatusEventPrepare() API from packet object
 *  when endIndex is equal or greater than Packet::Desc::statusEventCount
 *  and status as LwSciError_Success should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusEventPrepare()}
 */
TEST_F (packet_unit_test, packetStatusEventPrepare_StreamInternalError)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Success;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->packetStatusEventPrepare(1, status));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusEventPrepare_IlwalidState}
 * @testcase{21608171}
 * @verify{20040255}
 * @testpurpose{Test negative scenario of Packet::packetStatusEventPrepare(), where
 *  when the packet acceptance status is already received.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Packet::packetStatusEventPrepare() is ilwoked with endIndex value as 0
 *      and status as LwSciError_Success.
 *
 *   The call of Packet::packetStatusEventPrepare() API from packet object again with the same
 * endIndex and status should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusEventPrepare()}
 */
TEST_F (packet_unit_test, packetStatusEventPrepare_IlwalidState)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Success;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, status));

    /* test code */
    EXPECT_EQ(LwSciError_IlwalidState, packetObj->packetStatusEventPrepare(0, status));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusEventPrepare_Success1}
 * @testcase{21608172}
 * @verify{20040255}
 * @testpurpose{Test success scenario of Packet::packetStatusEventPrepare() when
 *  producer packet acceptance status is updated successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *
 *   The call of Packet::packetStatusEventPrepare() API from packet object with endIndex value as 0
 *  and status as LwSciError_Success, should return LwSciError_Success. Calling
 *  Packet::pendingEvent() should return event set to LwSciStreamEventType_PacketStatusProducer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusEventPrepare()}
 */
TEST_F (packet_unit_test, packetStatusEventPrepare_Success1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    LwSciStreamEvent event;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Success;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, status));

    EXPECT_EQ(true, packetObj->pendingEvent(event));
    EXPECT_EQ(LwSciStreamEventType_PacketStatusProducer, event.type);
    EXPECT_EQ(packetHandle, event.packetHandle);
    EXPECT_EQ(poolCookie, event.packetCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusEventPrepare_Success2}
 * @testcase{22060015}
 * @verify{20040255}
 * @testpurpose{Test success scenario of Packet::packetStatusEventPrepare() when
 *  consumer packet acceptance status is updated successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusConsumer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusConsumer.
 *
 *   The call of Packet::packetStatusEventPrepare() API from packet object with endIndex value as 0U
 *  and status as LwSciError_Success, should return LwSciError_Success. Calling
 * Packet::pendingEvent() should return event set to LwSciStreamEventType_PacketStatusConsumer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusEventPrepare()}
 */
TEST_F (packet_unit_test, packetStatusEventPrepare_Success2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    LwSciStreamEvent event;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Success;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusConsumer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusConsumer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0U, status));
    EXPECT_EQ(true, packetObj->pendingEvent(event));
    EXPECT_EQ(LwSciStreamEventType_PacketStatusConsumer, event.type);
    EXPECT_EQ(packetHandle, event.packetHandle);
    EXPECT_EQ(poolCookie, event.packetCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventPrepare_StreamInternalError}
 * @testcase{21608173}
 * @verify{20040258}
 * @testpurpose{Test negative scenario of Packet::bufferStatusEventPrepare() when
 * endIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::bufferStatusEventPrepare() API from packet object with elemIndex as 0,
 *  status as LwSciError_Success and endIndex equal or greater than Packet::Desc::statusEventCount
 *  should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventPrepare()}
 */
TEST_F (packet_unit_test, bufferStatusEventPrepare_StreamInternalError)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Success;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_StreamInternalError, packetObj->bufferStatusEventPrepare(1, 0, status));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventPrepare_BadParameter}
 * @testcase{21608174}
 * @verify{20040258}
 * @testpurpose{Test negative scenario of Packet::bufferStatusEventPrepare() when
 * elemIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *
 *   The call of Packet::bufferStatusEventPrepare() API from packet object with endIndex as 0,
 *  status as LwSciError_Success and elemIndex value equal or greater than packet element count
 *  tracked by elemCount object should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventPrepare()}
 */
TEST_F (packet_unit_test, bufferStatusEventPrepare_BadParameter)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Success;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_BadParameter, packetObj->bufferStatusEventPrepare(0, 1, status));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventPrepare_IlwalidState}
 * @testcase{21608175}
 * @verify{20040258}
 * @testpurpose{Test negative scenario of Packet::bufferStatusEventPrepare() when
 *  the element acceptance status for the elemIndex element is already received.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Packet::bufferStatusEventPrepare() is ilwoked with elemIndex and endIndex as 0,
 *      and status as LwSciError_Success.
 *
 *   The call of Packet::bufferStatusEventPrepare() API from packet object again
 *  with the same endIndex, status and elemIndex should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventPrepare()}
 */
TEST_F (packet_unit_test, bufferStatusEventPrepare_IlwalidState)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Success;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusEventPrepare(0, 0, status));

    /* test code */
    EXPECT_EQ(LwSciError_IlwalidState, packetObj->bufferStatusEventPrepare(0, 0, status));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventPrepare_Success}
 * @testcase{21608176}
 * @verify{20040258}
 * @testpurpose{Test success scenario of Packet::bufferStatusEventPrepare() when
 *  element acceptance status is updated successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *
 *   The call of Packet::bufferStatusEventPrepare() API from packet object with endIndex as 0,
 *  elemIndex as 0 and status as LwSciError_Unknown should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventPrepare()}
 */
TEST_F (packet_unit_test, bufferStatusEventPrepare_Success)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciError status = LwSciError_Unknown;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusEventPrepare(0, 0, status));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.statusEventRejectedCheck_False}
 * @testcase{21608177}
 * @verify{20040261}
 * @testpurpose{Test positive scenario of Packet::statusEventRejectedCheck(), when
 *  neither the packet instance nor any of the packet elements were rejected.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *
 *   The call of Packet::statusEventRejectedCheck() API from packet object should
 * return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::statusEventRejectedCheck()}
 */
TEST_F (packet_unit_test, statusEventRejectedCheck_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_FALSE(packetObj->statusEventRejectedCheck());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.statusEventRejectedCheck_True1}
 * @testcase{21608178}
 * @verify{20040261}
 * @testpurpose{Test positive scenario of Packet::statusEventRejectedCheck(), when
 *  there is a packet instance rejection.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Packet instance receives error code (other than LwSciError_Success) as packet status
 *      by the call of packetStatusEventPrepare().
 *
 *   The call of Packet::statusEventRejectedCheck() API from packet object should
 * return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::statusEventRejectedCheck()}
 */
TEST_F (packet_unit_test, statusEventRejectedCheck_True1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Unknown));

    /* test code */
    EXPECT_TRUE(packetObj->statusEventRejectedCheck());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.statusEventRejectedCheck_True2}
 * @testcase{22059910}
 * @verify{20040261}
 * @testpurpose{Test positive scenario of Packet::statusEventRejectedCheck(), when
 *  there is a packet elements rejection.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Packet elements receives error code (other than LwSciError_Success) as packet status
 *      by the call of bufferStatusEventPrepare().
 *
 *   The call of Packet::statusEventRejectedCheck() API from packet object should
 * return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::statusEventRejectedCheck()}
 */
TEST_F (packet_unit_test, statusEventRejectedCheck_True2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    uint32_t const endIndex = 0U;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusEventPrepare(endIndex, 0, LwSciError_Unknown));

    /* test code */
    EXPECT_TRUE(packetObj->statusEventRejectedCheck());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.allStatusEventsUsed_False1}
 * @testcase{21608179}
 * @verify{20040264}
 * @testpurpose{Test negative scenario of Packet::allStatusEventsUsed(), when
 *  not all packet acceptance-status LwSciStreamEvent(s) have been received.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *
 *   The call of Packet::allStatusEventsUsed() API from packet object should
 *  return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::allStatusEventsUsed()}
 */
TEST_F (packet_unit_test, allStatusEventsUsed_False1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_FALSE(packetObj->allStatusEventsUsed());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.allStatusEventsUsed_False2}
 * @testcase{21608180}
 * @verify{20040264}
 * @testpurpose{Test negative scenario of Packet::allStatusEventsUsed(), when
 *  when the packet is accepted but not all element acceptance-status
 *  LwSciStreamEvent(s) have been received.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Calls packetStatusEventPrepare() to store the packet-acceptance status.
 *
 *   The call of Packet::allStatusEventsUsed() API from packet object should
 *  return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::allStatusEventsUsed()}
 */
TEST_F (packet_unit_test, allStatusEventsUsed_False2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    packetObj->packetStatusEventPrepare(0, LwSciError_Success);

    /* test code */
    EXPECT_FALSE(packetObj->allStatusEventsUsed());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.allStatusEventsUsed_True}
 * @testcase{21608181}
 * @verify{20040264}
 * @testpurpose{Test positive scenario of Packet::allStatusEventsUsed(), when
 *  the packet is accepted and element acceptance-status LwSciStreamEvents have been received.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Calls packetStatusEventPrepare() and bufferStatusEventPrepare() to store the
 *      packet-acceptance status and element-acceptance status.
 *
 *   The call of Packet::allStatusEventsUsed() API from packet object should
 * return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::allStatusEventsUsed()}
 */
TEST_F (packet_unit_test, allStatusEventsUsed_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    packetObj->packetStatusEventPrepare(0, LwSciError_Success);
    packetObj->bufferStatusEventPrepare(0, 0, LwSciError_Success);

    /* test code */
    EXPECT_TRUE(packetObj->allStatusEventsUsed());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusEventsDone_False}
 * @testcase{21608182}
 * @verify{20040267}
 * @testpurpose{Test negative scenario of Packet::packetStatusEventsDone(), when
 *  packet acceptance-status LwSciStreamEvent is not retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *
 *   The call of Packet::packetStatusEventsDone() API from packet object should
 * return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusEventsDone()}
 */
TEST_F (packet_unit_test, packetStatusEventsDone_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_FALSE(packetObj->packetStatusEventsDone());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusEventsDone_True}
 * @testcase{21608183}
 * @verify{20040267}
 * @testpurpose{Test positive scenario of Packet::packetStatusEventsDone(), when
 *  packet acceptance-status LwSciStreamEvent has been retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Calls Packet::packetStatusEventPrepare() to store the packet status.
 *   5. Calls Packet::pendingEvent() to retrieve the packet status.
 *
 *   The call of Packet::packetStatusEventsDone() API from packet object should
 * return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusEventsDone()}
 */
TEST_F (packet_unit_test, packetStatusEventsDone_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));

    EXPECT_TRUE(packetObj->pendingEvent(event));

    /* test code */
    EXPECT_TRUE(packetObj->packetStatusEventsDone());

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusProducer);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventsDone_False1}
 * @testcase{21608184}
 * @verify{20040270}
 * @testpurpose{Test negative scenario of Packet::bufferStatusEventsDone(), when
 *  element acceptance-status LwSciStreamEvents are not retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 2.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Calls Packet::packetStatusEventPrepare() to store packet acceptance status.
 *
 *   The call of Packet::bufferStatusEventsDone() API from packet object with elemIndex value as 1
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventsDone()}
 */
TEST_F (packet_unit_test, bufferStatusEventsDone_False1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 2, maxCount = 3;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));

    /* test code */
    EXPECT_FALSE(packetObj->bufferStatusEventsDone(1));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventsDone_False2}
 * @testcase{21608185}
 * @verify{20040270}
 * @testpurpose{Test negative scenario of Packet::bufferStatusEventsDone(), when
 *  the packet acceptance-status LwSciStreamEvents are retrieved but not all
 *  element acceptance-status LwSciStreamEvents are retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 2.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Calls Packet::packetStatusEventPrepare() to store packet acceptance status.
 *   5. Calls Packet::pendingEvent() to retrieve the packet acceptance status.
 *   6. Calls Packet::bufferStatusEventPrepare() to store the element acceptance status.
 *
 *   The call of Packet::bufferStatusEventsDone() API from packet object with elemIndex value as 1
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventsDone()}
 */
TEST_F (packet_unit_test, bufferStatusEventsDone_False2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 2, maxCount = 3;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));
    EXPECT_TRUE(packetObj->pendingEvent(event));
    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusProducer);
    EXPECT_TRUE(packetObj->packetStatusEventsDone());

    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusEventPrepare(0, 1, LwSciError_Success));

    /* test code */
    EXPECT_FALSE(packetObj->bufferStatusEventsDone(1));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventsDone_False3}
 * @testcase{21664570}
 * @verify{20040270}
 * @testpurpose{Test negative scenario of Packet::bufferStatusEventsDone(), when
 * elemIndex is overflow.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 2.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Calls Packet::packetStatusEventPrepare() to store packet acceptance status.
 *   5. Calls Packet::pendingEvent() to retrieve the packet acceptance status.
 *   The call of Packet::bufferStatusEventsDone() API from packet object with elemIndex value as 65
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventsDone()}
 */
TEST_F (packet_unit_test, bufferStatusEventsDone_False3)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    LwSciStreamEvent event;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 2, maxCount = 3;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));
    EXPECT_TRUE(packetObj->pendingEvent(event));
    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusProducer);
    EXPECT_TRUE(packetObj->packetStatusEventsDone());

    /* test code */
    EXPECT_FALSE(packetObj->bufferStatusEventsDone(65));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.bufferStatusEventsDone_True}
 * @testcase{21608186}
 * @verify{20040270}
 * @testpurpose{Test positive scenario of Packet::bufferStatusEventsDone(), when
 *  all expected acceptance-status LwSciStreamEvents have been retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 2.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Calls Packet::packetStatusEventPrepare() to store packet acceptance status.
 *   5. Calls Packet::pendingEvent() to retrieve the packet acceptance status.
 *   6. Calls Packet::bufferStatusEventPrepare() to store the element acceptance status.
 *   7. Calls Packet::pendingEvent() to retrieve the element acceptance status.
 *
 *   The call of Packet::bufferStatusEventsDone() API from packet object with elemIndex value as 1
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::bufferStatusEventsDone()}
 */
TEST_F (packet_unit_test, bufferStatusEventsDone_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 2, maxCount = 3;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));
    EXPECT_TRUE(packetObj->pendingEvent(event));
    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusProducer);
    EXPECT_TRUE(packetObj->packetStatusEventsDone());

    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusEventPrepare(0, 1, LwSciError_Success));
    EXPECT_TRUE(packetObj->pendingEvent(event));

    /* test code */
    EXPECT_TRUE(packetObj->bufferStatusEventsDone(1));

    EXPECT_EQ(event.type, LwSciStreamEventType_ElementStatusProducer);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.locationCheck_False}
 * @testcase{21608187}
 * @verify{20040231}
 * @testpurpose{Test negative scenario of Packet::locationCheck(), when
 *  current packet Location of the packet instance does not match expected packet Location.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with the default packet location Packet::Location::Unknown and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::locationCheck() API with expectLocation other than
 *  Packet::Location::Unknown from packet object should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::locationCheck()}
 */
TEST_F (packet_unit_test, locationCheck_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    Packet::Location expectLocation = Packet::Location::Application;

    /* test code */
    EXPECT_FALSE(packetObj->locationCheck(expectLocation));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.locationCheck_True}
 * @testcase{21608188}
 * @verify{20040231}
 * @testpurpose{Test positive scenario of Packet::locationCheck(), when
 *  current packet Location of the packet instance matches expected packet Location.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with the default packet location Packet::Location::Unknown and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::locationCheck() API with expectLocation as Packet::Location::Unknown
 *  from packet object should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::locationCheck()}
 */
TEST_F (packet_unit_test, locationCheck_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    Packet::Location expectLocation = Packet::Location::Unknown;

    /* test code */
    EXPECT_TRUE(packetObj->locationCheck(expectLocation));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.locationUpdate_False}
 * @testcase{21608189}
 * @verify{20040234}
 * @testpurpose{Test negative scenario of Packet::locationUpdate(), when
 *  the current packet Location doesn't match the input oldLocation.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with the default packet location Packet::Location::Unknown and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::locationUpdate() API with parameter oldLocation not set to
 *  Packet::Location::Unknown from packet object should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::locationUpdate()}
 */
TEST_F (packet_unit_test, locationUpdate_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->initialLocation = Packet::Location::Unknown;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    Packet::Location oldLocation = Packet::Location::Downstream;
    Packet::Location newLocation = Packet::Location::Upstream;

    EXPECT_FALSE(packetObj->locationUpdate(oldLocation, newLocation));

    /* test code */
    EXPECT_FALSE(packetObj->locationCheck(newLocation));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.locationUpdate_True}
 * @testcase{21608190}
 * @verify{20040234}
 * @testpurpose{Test positive scenario of Packet::locationUpdate(), when
 *  current packet Location is replaced with the new packet Location.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::initialLocation as Packet::Location::Application and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::locationUpdate() API from packet object with parameters
 *  (oldLocation as Packet::Location::Application and newLocation as Packet::Location::Upstream)
 *  should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::locationUpdate()}
 */
TEST_F (packet_unit_test, locationUpdate_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->initialLocation = Packet::Location::Application;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    Packet::Location oldLocation = Packet::Location::Application;
    Packet::Location newLocation = Packet::Location::Upstream;

    EXPECT_TRUE(packetObj->locationUpdate(oldLocation, newLocation));

    /* test code */
    EXPECT_TRUE(packetObj->locationCheck(newLocation));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.deleteSet_False}
 * @testcase{21608191}
 * @verify{20040237}
 * @testpurpose{Test negative scenario of Packet::deleteSet(), when
 *  the packet instance was previously marked as deleted.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::elemCount pointing
 *   to the TrackCount object with count set to 1.
 *
 *   The call of Packet::deleteSet() API from packet object again should
 * return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::deleteSet()}
 */
TEST_F (packet_unit_test, deleteSet_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_TRUE(packetObj->deleteSet());

    /* test code */
    EXPECT_FALSE(packetObj->deleteSet());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.deleteSet_True}
 * @testcase{21608192}
 * @verify{20040237}
 * @testpurpose{Test positive scenario of Packet::deleteSet(), when
 *  the packet instance was not previously marked as deleted.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Query the pending LwSciStreamEventType_PacketCreate event by calling
 *    Packet::pendingEvent().
 *
 *   The call of Packet::deleteSet() API from packet object should
 * return true and calling of Packet::pendingEvent() should return
 * LwSciStreamEventType_PacketDelete event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::deleteSet()}
 */
TEST_F (packet_unit_test, deleteSet_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    LwSciStreamEvent event;
    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;
    pktDesc->endpointUse = false;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_EQ(true, packetObj->pendingEvent(event));
    EXPECT_EQ(LwSciStreamEventType_PacketCreate, event.type);

    /* test code */
    EXPECT_TRUE(packetObj->deleteSet());

    EXPECT_EQ(true, packetObj->pendingEvent(event));
    EXPECT_EQ(LwSciStreamEventType_PacketDelete, event.type);
    EXPECT_EQ(packetHandle, event.packetHandle);
    EXPECT_EQ(poolCookie, event.packetCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.deleteGet_False}
 * @testcase{21608193}
 * @verify{20040240}
 * @testpurpose{Test positive scenario of Packet::deleteGet(), when
 *  the packet instance is not marked for deletion.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::elemCount pointing
 *   to the TrackCount object with count set to 1.
 *
 *   The call of Packet::deleteGet() API from packet object should
 * return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::deleteGet()}
 */
TEST_F (packet_unit_test, deleteGet_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_FALSE(packetObj-> deleteGet());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.deleteGet_True}
 * @testcase{21608194}
 * @verify{20040240}
 * @testpurpose{Test positive scenario of Packet::deleteGet(), when
 *  the packet instance is marked for deletion.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::elemCount pointing
 *      to the TrackCount object with count set to 1.
 *   2. Calls Packet::deleteSet() to mark the packet for deletion.
 *
 *   The call of Packet::deleteGet() API from packet object should
 * return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::deleteGet()}
 */
TEST_F (packet_unit_test, deleteGet_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    EXPECT_TRUE(packetObj->deleteSet());

    /* test code */
    EXPECT_TRUE(packetObj-> deleteGet());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_False1}
 * @testcase{21608195}
 * @verify{20040273}
 * @testpurpose{Test negative scenario of Packet::pendingEvent(), when
 *  no LwSciStreamEvent oclwred.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineEventCount set to 2 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_False1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    /* test code */
    EXPECT_FALSE(packetObj->pendingEvent(event));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_False2}
 * @testcase{21664578}
 * @verify{20040273}
 * @testpurpose{Test negative scenario of Packet::pendingEvent(), when
 *  PacketCreate event is already retrieved and no other pending event
 *  oclwrs on the packet at endpoints.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount set to 1,
 *      Packet::Desc::endpointUse set to true and Packet::Desc::elemCount
 *      pointing to the TrackCount object with count set to 1.
 *   2. Call Packet::pendingEvent() to query the PacketCreate event.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return false as the PacketCreate event is already retrieved and
 *  no other pending event is ready for retrieval.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_False2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;
    pktDesc->endpointUse = true;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketCreate);
    EXPECT_EQ(event.packetHandle , packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    /* test code */
    EXPECT_FALSE(packetObj->pendingEvent(event));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_False3}
 * @testcase{21664579}
 * @verify{20040273}
 * @testpurpose{Test negative scenario of Packet::pendingEvent(), when
 *  neither PacketStatus nor ElementStatus event is ready.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return false as no pending PacketStatus or ElementStatus event is for retrieval.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_False3)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    LwSciStreamEvent event;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_FALSE(packetObj->pendingEvent(event));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_False4}
 * @testcase{21664580}
 * @verify{20040273}
 * @testpurpose{Test negative scenario of Packet::pendingEvent(), when
 *  LwSciStreamPacket is accepted and no pending ElementStatus event is retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount and
 *      Packet::Desc::statusActionCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Call Packet::pendingEvent() to retrieve LwSciStreamEventType_PacketCreate event.
 *   3. Call Packet::packetStatusAction() with valid cookie to accept the packet.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return false as no pending PacketElement event oclwrs.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_False4)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketCreate);
    EXPECT_EQ(event.packetHandle , packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(poolCookie, action));
    EXPECT_TRUE(actionPerformed);

    /* test code */
    EXPECT_FALSE(packetObj->pendingEvent(event));
    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_False5}
 * @testcase{21664581}
 * @verify{20040273}
 * @testpurpose{Test negative scenario of Packet::pendingEvent(), when
 *  LwSciStreamPacket is not accepted and clears pending PacketElement events
 *  for the rejected packets, if any exist.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount and
 *      Packet::Desc::statusActionCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Call Packet::pendingEvent() to retrieve LwSciStreamEventType_PacketCreate event.
 *   3. Call Packet::packetStatusAction() with invalid cookie so that packet is not accepted.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return false as no pending PacketElement event oclwrs.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_False5)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketCreate);
    EXPECT_EQ(event.packetHandle , packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(LwSciStreamCookie_Ilwalid, action));
    EXPECT_TRUE(actionPerformed);

    /* test code */
    EXPECT_FALSE(packetObj->pendingEvent(event));
    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_false6}
 * @testcase{21664582}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  LwSciStreamPacket is not accepted and pending buffer events are not dequeued.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount and
 *      Packet::Desc::statusActionCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Call Packet::pendingEvent() to retrieve LwSciStreamEventType_PacketCreate event.
 *   3. Call Packet::bufferDefineEventPrepare() with elemIndex and LwSciWrap::BufObj.
 *   4. Call Packet::packetStatusAction() with invalid cookie so that packet is not accepted.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return false and should not post any LwSciStreamEvent when packet is not accepted.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_false6)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketCreate);
    EXPECT_EQ(event.packetHandle , packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    LwSciBufObj poolElementBuf;
    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    EXPECT_EQ(LwSciError_Success, packetObj->bufferDefineEventPrepare(0, wrapBufObj));

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(LwSciStreamCookie_Ilwalid, action));
    EXPECT_TRUE(actionPerformed);

    /* test code */
    EXPECT_FALSE(packetObj->pendingEvent(event));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True1}
 * @testcase{21608196}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  LwSciStreamEventType_PacketCreate event is retrieved successfully.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::defineEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post LwSciStreamEventType_PacketCreate event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketCreate);
    EXPECT_EQ(event.packetHandle , packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True2}
 * @testcase{21664583}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  LwSciStreamPacket is accepted so that LwSciStreamEventType_PacketElement
 *  event is retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount and
 *   Packet::Desc::statusActionCount set to 1 and Packet::Desc::elemCount pointing
 *   to the TrackCount object with count set to 1.
 *   2. Call Packet::pendingEvent() to retrieve LwSciStreamEventType_PacketCreate event.
 *   3. Call Packet::packetStatusAction() with valid cookie and action.
 *   4. Call Packet::bufferDefineEventPrepare() with elemIndex and LwSciWrap::BufObj.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post LwSciStreamEventType_PacketElement
 *  event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketCreate);
    EXPECT_EQ(event.packetHandle , packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(poolCookie, action));
    EXPECT_TRUE(actionPerformed);

    LwSciBufObj poolElementBuf;
    makeRawBuffer(rawBufAttrList, poolElementBuf);
    LwSciWrap::BufObj wrapBufObj{poolElementBuf};

    EXPECT_EQ(LwSciError_Success, packetObj->bufferDefineEventPrepare(0, wrapBufObj));

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketElement);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True3}
 * @testcase{21664584}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  LwSciStreamPacket is accepted and then deleted so that LwSciStreamEventType_PacketDelete
 *  event is retrieved successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::defineEventCount and
 *   Packet::Desc::statusActionCount set to 1 and Packet::Desc::elemCount pointing
 *   to the TrackCount object with count set to 1.
 *   2. Call Packet::pendingEvent() to retrieve LwSciStreamEventType_PacketCreate event.
 *   3. Call Packet::packetStatusAction() with valid cookie and action.
 *   4. Call Packet::deleteSet() to mark packet instance for deletion.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post LwSciStreamEventType_PacketDelete
 *  event.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True3)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->defineEventCount = 1U;
    pktDesc->statusActionCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketCreate);
    EXPECT_EQ(event.packetHandle , packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusAction(poolCookie, action));
    EXPECT_TRUE(actionPerformed);

    EXPECT_TRUE(packetObj->deleteSet());

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketDelete);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True4}
 * @testcase{21664585}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  producer packet acceptance status LwSciStreamEvent is retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
  *  2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Call Packet::packetStatusEventPrepare() with endIndex as 0 and status as LwSciError_Success.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post event.type as LwSciStreamEventType_PacketStatusProducer
 *  and event.error as LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True4)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusProducer);
    EXPECT_EQ(event.error, LwSciError_Success);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True5}
 * @testcase{22060017}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  producer packet acceptance status is not LwSciError_Success.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
  *  2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Call Packet::packetStatusEventPrepare() with endIndex as 0 and status
 *    as LwSciError_BadParameter.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post event.type as LwSciStreamEventType_PacketStatusProducer
 *  and event.error as LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True5)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_BadParameter));

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusProducer);
    EXPECT_EQ(event.error, LwSciError_BadParameter);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True6}
 * @testcase{22060020}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  consumer packet acceptance status LwSciStreamEvent is retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
  *  2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusConsumer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusConsumer.
 *   4. Call Packet::packetStatusEventPrepare() with endIndex as 0 and status as LwSciError_Success.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post event.type as LwSciStreamEventType_PacketStatusConsumer
 *  and event.error as LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True6)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusConsumer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusConsumer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0U, LwSciError_Success));

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusConsumer);
    EXPECT_EQ(event.error, LwSciError_Success);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True7}
 * @testcase{22060022}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  consumer packet acceptance status is not LwSciError_Success.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
  *  2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusConsumer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusConsumer.
 *   4. Call Packet::packetStatusEventPrepare() with endIndex as 0 and status as
 *    LwSciError_BadParameter.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post event.type as LwSciStreamEventType_PacketStatusConsumer
 *  and event.error as LwSciError_BadParameter.}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True7)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusConsumer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusConsumer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_BadParameter));

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusConsumer);
    EXPECT_EQ(event.error, LwSciError_BadParameter);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True8}
 * @testcase{21664586}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  producer element acceptance status LwSciStreamEvent is retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusProducer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusProducer.
 *   4. Call Packet::packetStatusEventPrepare() with endIndex as 0 and status as LwSciError_Success.
 *   5. Call Packet::pendingEvent() to retrieve LwSciStreamEventType_PacketStatusProducer event.
 *   6. Call Packet::bufferStatusEventPrepare() with endIndex as 0, elemIndex as 0
 *      and status as LwSciError_Success.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post event.type as LwSciStreamEventType_ElementStatusProducer
 *  and event.error as LwSciError_Success.}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True8)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusProducer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusProducer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));
    EXPECT_TRUE(packetObj->pendingEvent(event));
    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusProducer);

    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusEventPrepare(0, 0, LwSciError_Success));

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_ElementStatusProducer);
    EXPECT_EQ(event.error, LwSciError_Success);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.pendingEvent_True9}
 * @testcase{22060025}
 * @verify{20040273}
 * @testpurpose{Test positive scenario of Packet::pendingEvent(), when
 *  consumer element acceptance status LwSciStreamEvent is retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::statusEventCount set to 1 and
 *      Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *   2. Set packetStatus LwSciStreamEvent to LwSciStreamEventType_PacketStatusConsumer.
 *   3. Set bufferStatus LwSciStreamEvent to LwSciStreamEventType_ElementStatusConsumer.
 *   4. Call Packet::packetStatusEventPrepare() with endIndex as 0 and status as LwSciError_Success.
 *   5. Call Packet::pendingEvent() to retrieve LwSciStreamEventType_PacketStatusProducer event.
 *   6. Call Packet::bufferStatusEventPrepare() with endIndex as 0, elemIndex as 0
 *      and status as LwSciError_Success.
 *
 *   The call of Packet::pendingEvent() API from packet object should
 *  return true and should post event.type as LwSciStreamEventType_ElementStatusConsumer
 *  and event.error as LwSciError_Success.}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::pendingEvent()}
 */
TEST_F (packet_unit_test, pendingEvent_True9)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);
    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->statusEventCount = 1U;
    pktDesc->packetStatusEventType.push_back(LwSciStreamEventType_PacketStatusConsumer);
    pktDesc->bufferStatusEventType.push_back(LwSciStreamEventType_ElementStatusConsumer);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciStreamEvent event;

    EXPECT_EQ(LwSciError_Success, packetObj->packetStatusEventPrepare(0, LwSciError_Success));
    EXPECT_TRUE(packetObj->pendingEvent(event));
    EXPECT_EQ(event.type, LwSciStreamEventType_PacketStatusConsumer);

    EXPECT_EQ(LwSciError_Success, packetObj->bufferStatusEventPrepare(0, 0, LwSciError_Success));

    /* test code */
    EXPECT_TRUE(packetObj->pendingEvent(event));

    EXPECT_EQ(event.type, LwSciStreamEventType_ElementStatusConsumer);
    EXPECT_EQ(event.error, LwSciError_Success);
    EXPECT_EQ(event.packetHandle, packetHandle);
    EXPECT_EQ(event.packetCookie, poolCookie);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.payloadSet_BadParameter}
 * @testcase{21608197}
 * @verify{20040285}
 * @testpurpose{Test negative scenario of Packet::payloadSet(), when
 *  endIndex is out of range.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::payloadCount set to 1 and Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::payloadSet() API with parameter endIndex as 1 from packet object
 *  should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::payloadSet()}
 */
TEST_F (packet_unit_test, payloadSet_BadParameter)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    pktDesc->payloadCount = 1U;
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                                 malloc(sizeof(LwSciSyncFence) * MAX_SYNC_OBJECTS));

    for (uint32_t j = 0U; j < MAX_SYNC_OBJECTS; j++) {
        fences[j] = LwSciSyncFenceInitializer;
    }

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < MAX_SYNC_OBJECTS; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    /* test code */
    EXPECT_EQ(LwSciError_BadParameter, packetObj->payloadSet(1 ,1, 1, wrapFences));

    free(fences);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.payloadSet_IlwalidState}
 * @testcase{21608198}
 * @verify{20040285}
 * @testpurpose{Test negative scenario of Packet::payloadSet(), when
 *  the LwSciSyncFences from this endpoint are already set.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::payloadCount set to 1 and Packet::Desc::elemCount
 *      pointing to the TrackCount object with count set to 1.
 *   2. Packet::payloadSet() is ilwoked with endIndex as 0, fenceCount as 1, fenceOffset as 1.
 *      and fences parameter.
 *
 *   The call of Packet::payloadSet() API with the same endIndex, fenceCount and fenceOffset
 *  from packet object again should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::payloadSet()}
 */
TEST_F (packet_unit_test, payloadSet_IlwalidState)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    pktDesc->payloadCount = 1U;
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                                 malloc(sizeof(LwSciSyncFence) * MAX_SYNC_OBJECTS));

    for (uint32_t j = 0U; j < MAX_SYNC_OBJECTS; j++) {
        fences[j] = LwSciSyncFenceInitializer;
    }

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < MAX_SYNC_OBJECTS; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    EXPECT_EQ(LwSciError_Success, packetObj->payloadSet(0 ,1, 1, wrapFences));

    /* test code */
    EXPECT_EQ(LwSciError_IlwalidState, packetObj->payloadSet(0 ,1, 1, wrapFences));

    free(fences);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.payloadSet_Success}
 * @testcase{21608199}
 * @verify{20040285}
 * @testpurpose{Test success scenario of Packet::payloadSet(), when
 *  packet instance's FenceArray is set successfully.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::payloadCount set to 1 and Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::payloadSet() API from packet object should
 *  return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::payloadSet()}
 */
TEST_F (packet_unit_test, payloadSet_Success)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    pktDesc->payloadCount = 1U;
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                                 malloc(sizeof(LwSciSyncFence) * MAX_SYNC_OBJECTS));

    for (uint32_t j = 0U; j < MAX_SYNC_OBJECTS; j++) {
        fences[j] = LwSciSyncFenceInitializer;
    }

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < MAX_SYNC_OBJECTS; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    /* test code */
    EXPECT_EQ(LwSciError_Success, packetObj->payloadSet(0 ,1, 1, wrapFences));

    free(fences);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.payloadGet_False}
 * @testcase{21608200}
 * @verify{20040288}
 * @testpurpose{Test negative scenario of Packet::payloadGet(), when
 *  not all LwSciSyncFence(s) are ready to be retrieved.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::payloadCount set to 1 and Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::payloadGet() API from packet object should
 *  return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::payloadGet()}
 */
TEST_F (packet_unit_test, payloadGet_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->payloadCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                                 malloc(sizeof(LwSciSyncFence) * MAX_SYNC_OBJECTS));

    for (uint32_t j = 0U; j < MAX_SYNC_OBJECTS; j++) {
        fences[j] = LwSciSyncFenceInitializer;
    }

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < MAX_SYNC_OBJECTS; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    Payload extractedPayload;
    extractedPayload.handle = packetHandle;

    /* test code */
    EXPECT_FALSE(packetObj->payloadGet(1 ,extractedPayload));

    EXPECT_EQ(extractedPayload.handle, packetHandle);

    free(fences);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.payloadGet_True}
 * @testcase{21608201}
 * @verify{20040288}
 * @testpurpose{Test positive scenario of Packet::payloadGet(), when
 *  the fences are retrieved successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::payloadCount set to 1 and Packet::Desc::elemCount
 *      pointing to the TrackCount object with count set to 1.
 *   2. Calls Packet::payloadSet() to set the FenceArray from all endpoints.
 *
 *   The call of Packet::payloadGet() API from packet object should
 *  return true and verify that the extracted payload should be matching with
*   what is set by Packet::payloadSet() in step-2 .}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::payloadGet()}
 */
TEST_F (packet_unit_test, payloadGet_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->payloadCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                                 malloc(sizeof(LwSciSyncFence) * MAX_SYNC_OBJECTS));

    for (uint32_t j = 0U; j < MAX_SYNC_OBJECTS; j++) {
        fences[j] = LwSciSyncFenceInitializer;
    }

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < MAX_SYNC_OBJECTS; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    EXPECT_EQ(LwSciError_Success, packetObj->payloadSet(0 ,1, 1, wrapFences));

    Payload extractedPayload;

    /* test code */
    EXPECT_TRUE(packetObj->payloadGet(1 ,extractedPayload));

    EXPECT_EQ(extractedPayload.handle, packetHandle);

    LwSciSyncFence temp1 = extractedPayload.fences[0].viewVal();
    LwSciSyncFence temp2 = wrapFences[0].viewVal();

    EXPECT_EQ(temp1.payload[0], temp2.payload[0]);

    free(fences);

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.payloadSkip_False}
 * @testcase{21608202}
 * @verify{20040291}
 * @testpurpose{Test negative scenario of Packet::payloadSkip(), when
 *  not all LwSciSyncFence(s) are ready to be retrieved.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::payloadCount set to 1 and Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::payloadSkip() API from packet object should
 *  return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::payloadSkip()}
 */
TEST_F (packet_unit_test, payloadSkip_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->payloadCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                                 malloc(sizeof(LwSciSyncFence) * MAX_SYNC_OBJECTS));

    for (uint32_t j = 0U; j < MAX_SYNC_OBJECTS; j++) {
        fences[j] = LwSciSyncFenceInitializer;
    }

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < MAX_SYNC_OBJECTS; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    /* test code */
    EXPECT_FALSE(packetObj->payloadSkip());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.payloadSkip_True}
 * @testcase{21608203}
 * @verify{20040291}
 * @testpurpose{Test positive scenario of Packet::payloadSkip(), when
 *  FenceArray of the packet instance is cleared successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a packet object with Packet::Desc::payloadCount set to 1 and Packet::Desc::elemCount
 *      pointing to the TrackCount object with count set to 1.
 *   2. Calls payloadSet() to set the FenceArray from all endpoints.
 *
 *   The call of Packet::payloadSkip() API from packet object should
 *  return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::payloadSkip()}
 */
TEST_F (packet_unit_test, payloadSkip_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);
    pktDesc->payloadCount = 1U;

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    LwSciSyncFence *fences = static_cast<LwSciSyncFence*>(
                                 malloc(sizeof(LwSciSyncFence) * MAX_SYNC_OBJECTS));

    for (uint32_t j = 0U; j < MAX_SYNC_OBJECTS; j++) {
        fences[j] = LwSciSyncFenceInitializer;
    }

    FenceArray wrapFences { };
    for (uint32_t j { 0U }; j < MAX_SYNC_OBJECTS; j++) {
        wrapFences[j] = LwSciWrap::SyncFence(fences[j]);
    }

    EXPECT_EQ(LwSciError_Success, packetObj->payloadSet(0 ,1, 1, wrapFences));

    /* test code */
    EXPECT_TRUE(packetObj->payloadSkip());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_enqueue_Success1}
 * @testcase{21608204}
 * @verify{20108010}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::enqueue(), when
 *   packet instance is enqueued to an empty queue.}
 * @testbehavior{
 * Setup:
 *   Creates a Packet::PayloadQ and packet object with Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::PayloadQ::enqueue() API from packet object should result in valid pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::enqueue()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_enqueue_Success1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::PayloadQ payloadQ;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> newPacket
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    payloadQ.enqueue(newPacket);

    EXPECT_TRUE(payloadQ.extract(*newPacket));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_enqueue_Success2}
 * @testcase{21664587}
 * @verify{20108010}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::enqueue(), when
 *   packet instance is enqueued to a non-empty queue.}
 * @testbehavior{
 * Setup:
 *   Creates a Packet::PayloadQ and packet object with Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::PayloadQ::enqueue() API from packet object should result in valid pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::enqueue()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_enqueue_Success2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::PayloadQ payloadQ;

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> newPacket
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    payloadQ.enqueue(newPacket);

    // Choose Consumer cookie for packet
    LwSciStreamPacket packetHandle1{ 2U };
    LwSciStreamCookie consumerCookie { 3U };
    std::shared_ptr<Packet> newPacket1
        {std::make_shared<Packet>(*pktDesc, packetHandle1, consumerCookie)};

    /* test code */
    payloadQ.enqueue(newPacket1);

    EXPECT_TRUE(payloadQ.extract(*newPacket1));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_dequeue_Null}
 * @testcase{21608205}
 * @verify{20108016}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::dequeue(), when
 *   packet instance is removed from the head of an empty PayloadQ.}
 * @testbehavior{
 * Setup:
 *   Creates a Packet::PayloadQ object.
 *
 *   The call of Packet::PayloadQ::dequeue() API from packet object should result in nullptr.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::dequeue()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_dequeue_Null)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    Packet::PayloadQ payloadQ;

    /* test code */
    EXPECT_EQ(payloadQ.dequeue(), nullptr);
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_dequeue1}
 * @testcase{21664588}
 * @verify{20108016}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::dequeue(), when
 *   packet instance is removed from a non-empty PayloadQ.}
 * @testbehavior{
 * Setup:
 *   1. Creates a Packet::PayloadQ object.
 *   2. Call Packet::enqueue() to enqueue newly added packet.
 *
 *   The call of Packet::PayloadQ::dequeue() API from packet object should result in valid pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::dequeue()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_dequeue1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    Packet::PayloadQ payloadQ;

    // Choose consumer cookie for packet
    LwSciStreamPacket packetHandle{ 2U };
    LwSciStreamCookie consumerCookie { 3U };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> newPacket
        {std::make_shared<Packet>(*pktDesc, packetHandle, consumerCookie)};

    payloadQ.enqueue(newPacket);

    /* test code */
    EXPECT_NE(payloadQ.dequeue(), nullptr);
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_dequeue2}
 * @testcase{21664589}
 * @verify{20108016}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::dequeue(), when
 *   two packets are enqueued and first packet is dequeued.}
 * @testbehavior{
 * Setup:
 *   1. Creates a Packet::PayloadQ object.
 *   2. Call Packet::enqueue() to enqueue two newly added packets.
 *
 *   The call of Packet::PayloadQ::dequeue() API from packet object should result in valid pointer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::dequeue()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_dequeue2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    Packet::PayloadQ payloadQ;

    // Choose consumer cookie for packet
    LwSciStreamPacket packetHandle{ 2U };
    LwSciStreamCookie consumerCookie { 3U };

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> newPacket
        {std::make_shared<Packet>(*pktDesc, packetHandle, consumerCookie)};

    payloadQ.enqueue(newPacket);

    // Choose producer cookie for packet
    LwSciStreamPacket packetHandle1 { 4U };
    LwSciStreamCookie producerCookie { 6U };

    std::shared_ptr<Packet> newPacket1
        {std::make_shared<Packet>(*pktDesc, packetHandle1, producerCookie)};

    payloadQ.enqueue(newPacket1);

    /* test code */
    EXPECT_NE(payloadQ.dequeue(), nullptr);
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_extract_False}
 * @testcase{21608206}
 * @verify{20108019}
 * @testpurpose{Test negative scenario of Packet::PayloadQ::extract(), when
 *  the packet was not queued into Packet::PayloadQ.}
 * @testbehavior{
 * Setup:
 *   Creates a Packet::PayloadQ and packet object with Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::PayloadQ::extract() API from packet object should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::extract()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_extract_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    Packet::PayloadQ payloadQ;

    std::shared_ptr<Packet> oldPacket {std::make_shared<Packet>
                    (*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_FALSE(payloadQ.extract(*oldPacket));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_extract_True1}
 * @testcase{21608207}
 * @verify{20108019}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::extract(), when
 *  the packet is enqueued and is extracted successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a Packet::PayloadQ and packet object with Packet::Desc::elemCount
 *      pointing to the TrackCount object with count set to 1.
 *   2. Calls Packet::PayloadQ::enqueue() to enqueue the packet.
 *
 *   The call of Packet::PayloadQ::extract() API from packet object should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::extract()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_extract_True1)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    Packet::PayloadQ payloadQ;

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> oldPacket {std::make_shared<Packet>
                    (*pktDesc, packetHandle, poolCookie)};

    payloadQ.enqueue(oldPacket);

    /* test code */
    EXPECT_TRUE(payloadQ.extract(*oldPacket));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_extract_True2}
 * @testcase{21664590}
 * @verify{20108019}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::extract(), when
 *  three packets are enqueued and the second packet is extracted successfully.}
 * @testbehavior{
 * Setup:
 *   1. Creates a Packet::PayloadQ and packet object with Packet::Desc::elemCount
 *      pointing to the TrackCount object with count set to 1.
 *   2. Calls Packet::PayloadQ::enqueue() to enqueue three packets.
 *
 *   The call of Packet::PayloadQ::extract() API from the second packet object should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::extract()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_extract_True2)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    Packet::PayloadQ payloadQ;

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> oldPacket {std::make_shared<Packet>
                    (*pktDesc, packetHandle, poolCookie)};

    payloadQ.enqueue(oldPacket);

    // Choose consumer cookie for packet
    LwSciStreamPacket packetHandle1{ 3U };
    LwSciStreamCookie consumerCookie{ 5U };

    std::shared_ptr<Packet> oldPacket1 {std::make_shared<Packet>
                    (*pktDesc, packetHandle1, consumerCookie)};

    payloadQ.enqueue(oldPacket1);

    // Choose producer cookie for packet
    LwSciStreamPacket packetHandle2{ 8U };
    LwSciStreamCookie producerCookie{ 9U };

    std::shared_ptr<Packet> oldPacket2 {std::make_shared<Packet>
                    (*pktDesc, packetHandle2, producerCookie)};

    payloadQ.enqueue(oldPacket2);

    /* test code */
    EXPECT_TRUE(payloadQ.extract(*oldPacket1));

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_empty_False}
 * @testcase{21608208}
 * @verify{20108025}
 * @testpurpose{Test negative scenario of Packet::PayloadQ::empty(), when
 *  the Packet::PayloadQ is not empty.}
 * @testbehavior{
 * Setup:
 *   1. Creates a Packet::PayloadQ and packet object with Packet::Desc::elemCount
 *   pointing to the TrackCount object with count set to 1.
 *   2. Calls Packet::PayloadQ::enqueue() to mark new packet instance enqueued.
 *
 *   The call of Packet::PayloadQ::empty() API from packet object should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::empty()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_empty_False)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    Packet::PayloadQ payloadQ;

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> newPacket {std::make_shared<Packet>
                    (*pktDesc, packetHandle, poolCookie)};

    payloadQ.enqueue(newPacket);

    /* test code */
    EXPECT_FALSE(payloadQ.empty());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.Packet_PayloadQ_empty_True}
 * @testcase{21608210}
 * @verify{20108025}
 * @testpurpose{Test positive scenario of Packet::PayloadQ::empty(), when
 *  the Packet::PayloadQ is empty.}
 * @testbehavior{
 * Setup:
 *   1. Creates a Packet::PayloadQ and packet object with Packet::Desc::elemCount
 *      pointing to the TrackCount object with count set to 1.
 *   2. Calls Packet::PayloadQ::enqueue() to mark new packet instance enqueued.
 *   3. Calls Packet::PayloadQ::dequeue() to dequeue packet instance which is enqueued.
 *
 *   The call of Packet::PayloadQ::empty() API from packet object should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::PayloadQ::empty()}
 */
TEST_F (packet_unit_test, Packet_PayloadQ_empty_True)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    Packet::Desc* pktDesc = new Packet::Desc();
    Packet::PayloadQ payloadQ;

    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> newPacket {std::make_shared<Packet>
                    (*pktDesc, packetHandle, poolCookie)};

    payloadQ.enqueue(newPacket);

    payloadQ.dequeue();

    /* test code */
    EXPECT_TRUE(payloadQ.empty());

    delete pktDesc->elemCount;
    delete pktDesc;
}

/**
 * @testname{packet_unit_test.packetStatusAction_BadParameter}
 * @testcase{22059915}
 * @verify{20040249}
 * @testpurpose{Test negative scenario of Packet::packetStatusAction() when
 *  the action callback is NULL.}
 * @testbehavior{
 * Setup:
 *   Creates a packet object with Packet::Desc::statusActionCount set to 1 and
 *   Packet::Desc::elemCount pointing to the TrackCount object with count set to 1.
 *
 *   The call of Packet::packetStatusAction() API from packet object with the given
 *  valid LwSciStreamCookie and function callback is NULL should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{Packet::packetStatusAction()}
 */
TEST_F (packet_unit_test, LwSciError_BadParameter)
{
    using ::testing::Return;
    using ::testing::NiceMock;

    // Choose pool's cookie for packet
    LwSciStreamPacket packetHandle{ 1U };
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_BASE);

    bool actionPerformed = false;
    std::function<void(void)> const action {
        [&actionPerformed](void) noexcept -> void {
            actionPerformed = true;
        }
    };

    Packet::Desc* pktDesc = new Packet::Desc();
    pktDesc->statusActionCount = 1U;
    uint32_t count = 1, maxCount = 2;
    pktDesc->elemCount = new TrackCount(maxCount);
    pktDesc->elemCount->set(count);

    std::shared_ptr<Packet> packetObj
        {std::make_shared<Packet>(*pktDesc, packetHandle, poolCookie)};

    /* test code */
    EXPECT_EQ(LwSciError_BadParameter, packetObj->packetStatusAction(poolCookie, NULL));

    EXPECT_FALSE(actionPerformed);

    delete pktDesc->elemCount;
    delete pktDesc;
}

} // namespace LwSciStream

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
