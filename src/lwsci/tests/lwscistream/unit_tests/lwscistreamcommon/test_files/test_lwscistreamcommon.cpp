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
#include "sciwrap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include "lwscistream_types.h"
#include "lwscicommon_os.h"
#include "queue.h"
#include "lwscistream_panic.h"
#include "lwscistream_LwSciCommonPanic_mock.h"

class lwscistreamcommon_unit_test : public LwSciStreamTest
{
public:
    lwscistreamcommon_unit_test()
    {
        // initialization code here
    }

    void SetUp()
    {
        // code here will execute just before the test ensues
    }

    void TearDown()
    {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~lwscistreamcommon_unit_test()
    {
        // cleanup any pending stuff, but no exceptions allowed
    }

    // put in any custom data members that you need
};

/**
 * @testname{lwscistreamcommon_unit_test.uintAdd_True}
 * @testcase{21534025}
 * @verify{19721901}
 * @testpurpose{Test positive scenario of LwSciStream::uintAdd()}
 * @testbehavior{
 * Setup:
 *   None.
 *
 *   The call of LwSciStream::uintAdd() API with const unsigned integer of 1 and 2,
 * should return result of 3 and status of true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::uintAdd()}
 */
TEST_F(lwscistreamcommon_unit_test, uintAdd_True)
{
    uint32_t const var1 = 1;
    uint32_t const var2 = 2;

    LwSciStream::ArithRet<uint32_t> sum
                 { LwSciStream::uintAdd<uint32_t>(var1, var2) };

    EXPECT_TRUE(sum.status);
}

/**
 * @testname{lwscistreamcommon_unit_test.uintAdd_False}
 * @testcase{21534026}
 * @verify{19721901}
 * @testpurpose{Test negative scenario of LwSciStream::uintAdd()}
 * @testbehavior{
 * Setup:
 *   None.
 *
 *   The call of LwSciStream::uintAdd() API with const signed integer of 2 and -1,
 * should return status of false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::uintAdd()}
 */
TEST_F(lwscistreamcommon_unit_test, uintAdd_False)
{
    int32_t const var1 = 2;
    int32_t const var2 = -1;

    LwSciStream::ArithRet<int32_t> sum
                 { LwSciStream::uintAdd<int32_t>(var1, var2) };

    EXPECT_FALSE(sum.status);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_copy_LwSciError_to_LwSciStreamEvent1}
 * @testcase{21534027}
 * @verify{19721928}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   None.
 *
 *   The call of LwSciStream::moveToEvent() API with valid LwSciError and
 * LwSciStreamEvent, should copy the given LwSciError set as LwSciError_Success to LwSciStreamEvent.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_copy_LwSciError_to_LwSciStreamEvent1)
{
    LwSciStreamEvent event;

    LwSciStream::moveToEvent(LwSciError_Success, event);

    EXPECT_EQ(LwSciError_Success, event.error);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_copy_LwSciError_to_LwSciStreamEvent2}
 * @testcase{}
 * @verify{19721928}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   None.
 *
 *   The call of LwSciStream::moveToEvent() API with valid LwSciError and
 * LwSciStreamEvent, should copy the given LwSciError set as LwSciError_BadParameter to LwSciStreamEvent.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_copy_LwSciError_to_LwSciStreamEvent2)
{
    LwSciStreamEvent event;

    LwSciStream::moveToEvent(LwSciError_BadParameter, event);

    EXPECT_EQ(LwSciError_BadParameter, event.error);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_SyncInfo1}
 * @testcase{21534028}
 * @verify{19721940}
 * @verify{19722045}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1) set up the SyncInfo with LwSciWrap::SyncAttr wrapping a LwSciSyncAttrList and
 * synchronous flag.
 *
 *   Calls LwSciStream::moveToEvent() API with valid LwSciStream::SyncInfo and LwSciStreamEvent.
 * LwSciStreamEvent members syncAttrList set as NULL and synchronousOnly set as true,
 * should be evaluated to be equal to LwSciSyncAttrList and synchronous flag set
 * during SyncInfo creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_SyncInfo1)
{
    LwSciWrap::SyncAttr syncAttr{consSyncAttrList};

    LwSciStream::SyncInfo val {true , NULL};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(true, event.synchronousOnly);

    EXPECT_EQ(NULL, event.syncAttrList);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_SyncInfo2}
 * @testcase{}
 * @verify{19721940}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1) set up the SyncInfo with LwSciWrap::SyncAttr wrapping a LwSciSyncAttrList and
 * synchronous flag.
 *
 *   Calls LwSciStream::moveToEvent() API with valid LwSciStream::SyncInfo and LwSciStreamEvent.
 * LwSciStreamEvent members syncAttrList set as prodSyncAttrList and synchronousOnly set as false,
 * should be evaluated to be equal to LwSciSyncAttrList and synchronous flag set
 * during SyncInfo creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_SyncInfo2)
{
    LwSciWrap::SyncAttr syncAttr{prodSyncAttrList};

    LwSciStream::SyncInfo val {false , syncAttr.take()};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(false, event.synchronousOnly);

    EXPECT_EQ(consSyncAttrList, event.syncAttrList);
}


/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_LwSciSyncObj1}
 * @testcase{21534029}
 * @verify{19721946}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1)  set up the LwSciWrap::SyncObj with LwSciSyncObj.
 *
 *   Calls LwSciStream::moveToEvent() API with valid LwSciWrap::SyncObj set to and LwSciStreamEvent.
 * LwSciStreamEvent member syncObj set with 0xABCDEF, should be evaluated to be equal to
 * LwSciSyncObj set during LwSciWrap::SyncObj creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_LwSciSyncObj1)
{
    LwSciSyncObj syncObj {0xABCDEF};
    LwSciWrap::SyncObj val{syncObj};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(syncObj, event.syncObj);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_LwSciSyncObj2}
 * @testcase{}
 * @verify{19721946}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1)  set up the LwSciWrap::SyncObj with LwSciSyncObj.
 *
 *   Calls LwSciStream::moveToEvent() API with valid LwSciWrap::SyncObj and LwSciStreamEvent.
 * LwSciStreamEvent member syncObj set with 0x12AC34, should be evaluated to be equal to
 * LwSciSyncObj set during LwSciWrap::SyncObj creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_LwSciSyncObj2)
{
    LwSciSyncObj syncObj {0x12AC34};
    LwSciWrap::SyncObj val{syncObj};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(syncObj, event.syncObj);
}
/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_ElemInfo1}
 * @testcase{21534030}
 * @verify{19721949}
 * @verify{19722048}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1) set up the ElemInfo with elemType, LwSciStreamElementMode and LwSciWrap::BufAttr
 * wrapping a LwSciBufAttrList.
 *
 *   Calls LwSciStream::moveToEvent() API with valid LwSciStream::ElemInfo and LwSciStreamEvent.
 * LwSciStreamEvent members userData, syncMode and bufAttrList, should be evaluated to be equal to
 * elemType, LwSciStreamElementMode as LwSciStreamElementMode_Asynchronous and LwSciBufAttrList
 * set during ElemInfo creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_ElemInfo1)
{
    LwSciWrap::BufAttr bufAttr{rawBufAttrList};

    LwSciStream::ElemInfo val{1, LwSciStreamElementMode_Asynchronous, bufAttr.take()};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(1, event.userData);

    EXPECT_EQ(LwSciStreamElementMode_Asynchronous, event.syncMode);

    EXPECT_EQ(rawBufAttrList, event.bufAttrList);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_ElemInfo2}
 * @testcase{}
 * @verify{19721949}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1) set up the ElemInfo with elemType, LwSciStreamElementMode as LwSciStreamElementMode_Immediate
 * and LwSciWrap::BufAttr wrapping a LwSciBufAttrList.
 *
 *   Calls LwSciStream::moveToEvent() API with valid LwSciStream::ElemInfo and LwSciStreamEvent.
 * LwSciStreamEvent members userData, syncMode and bufAttrList, should be evaluated to be equal to
 * elemType, LwSciStreamElementMode as LwSciStreamElementMode_Immediate and LwSciBufAttrList
 * set during ElemInfo creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_ElemInfo2)
{
    LwSciWrap::BufAttr bufAttr{rawBufAttrList};

    LwSciStream::ElemInfo val{1, LwSciStreamElementMode_Immediate, bufAttr.take()};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(1, event.userData);

    EXPECT_EQ(LwSciStreamElementMode_Immediate, event.syncMode);

    EXPECT_EQ(rawBufAttrList, event.bufAttrList);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_LwSciBufObj1}
 * @testcase{21534031}
 * @verify{19721955}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1) set up the LwSciWrap::bufObj with LwSciBufObj.
 *
 *   Calls LwSciStream::moveToEvent() API with LwSciWrap::BufObj and LwSciStreamEvent.
 * LwSciStreamEvent member bufObj set with 0xABCDEF, should be evaluated to be equal to
 * LwSciBufObj set during LwSciWrap::BufObj creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_LwSciBufObj1)
{
    LwSciBufObj bufObj{0xABCDEF};
    LwSciWrap::BufObj val{bufObj};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(bufObj, event.bufObj);
}

/**
 * @testname{lwscistreamcommon_unit_test.moveToEvent_update_LwSciStreamEvent_with_LwSciBufObj2}
 * @testcase{}
 * @verify{19721955}
 * @testpurpose{Test positive scenario of LwSciStream::moveToEvent()}
 * @testbehavior{
 * Setup:
 *   1) set up the LwSciWrap::bufObj with LwSciBufObj.
 *
 *   Calls LwSciStream::moveToEvent() API with LwSciWrap::BufObj and LwSciStreamEvent.
 * LwSciStreamEvent member bufObj set with 0xAB23EF, should be evaluated to be equal to
 * LwSciBufObj set during LwSciWrap::BufObj creation.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::moveToEvent()}
 */
TEST_F(lwscistreamcommon_unit_test, moveToEvent_update_LwSciStreamEvent_with_LwSciBufObj2)
{
    LwSciBufObj bufObj{0xAB23EF};
    LwSciWrap::BufObj val{bufObj};

    LwSciStreamEvent event;

    LwSciStream::moveToEvent(val, event);

    EXPECT_EQ(bufObj, event.bufObj);
}
/**
 * @testname{lwscistreamcommon_unit_test.lwscistreamPanic}
 * @testcase{21534032}
 * @verify{20546970}
 * @testpurpose{Test positive scenario of LwSciStream::lwscistreamPanic()}
 * @testbehavior{
 * Setup: None.
 *
 *   The call of LwSciStream::lwscistreamPanic() API calls
 * LwSciCommonPanic() interface to terminate the program exelwtion.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::lwscistreamPanic()}
 */
TEST_F(lwscistreamcommon_unit_test,lwscistreamPanic)
{
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    lwscicommonPanicMock npm;

    EXPECT_CALL(npm, lwscicommonPanic_m())
               .Times(1)
               .WillRepeatedly(Return());

    LwSciStream::lwscistreamPanic();

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&npm));
}

/**
 * @testname{lwscistreamcommon_unit_test.Payload}
 * @testcase{}
 * @verify{19722036}
 * @testpurpose{Test positive scenario of LwSciStream::Payload::Payload construtor}
 * @testbehavior{
 * Setup: None.
 *
 *   An instance LwSciStream::Payload class is successfully created.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::Payload::Payload()}
 */
TEST_F(lwscistreamcommon_unit_test,Payload)
{
   LwSciStream::Payload Payload;
}

/**
 * @testname{lwscistreamcommon_unit_test.isStatusEvent_true1}
 * @testcase{}
 * @verify{22035187}
 * @testpurpose{Test positive scenario of LwSciStream::isStatusEvent.}
 * @testbehavior{
 * Setup: None.
 *
 *   The call of LwSciStream::isStatusEvent API with LwSciStreamEventType_PacketStatusProducer
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::isStatusEvent()}
 */
TEST_F(lwscistreamcommon_unit_test,isStatusEvent_true1)
{
    EXPECT_TRUE(LwSciStream::isStatusEvent(LwSciStreamEventType_PacketStatusProducer));
}

/**
 * @testname{lwscistreamcommon_unit_test.isStatusEvent_true2}
 * @testcase{}
 * @verify{22035187}
 * @testpurpose{Test positive scenario of LwSciStream::isStatusEvent.}
 * @testbehavior{
 * Setup: None.
 *
 *   The call of LwSciStream::isStatusEvent API with LwSciStreamEventType_PacketStatusConsumer
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::isStatusEvent()}
 */
TEST_F(lwscistreamcommon_unit_test,isStatusEvent_true2)
{
    EXPECT_TRUE(LwSciStream::isStatusEvent(LwSciStreamEventType_PacketStatusConsumer));
}

/**
 * @testname{lwscistreamcommon_unit_test.isStatusEvent_true3}
 * @testcase{}
 * @verify{22035187}
 * @testpurpose{Test positive scenario of LwSciStream::isStatusEvent.}
 * @testbehavior{
 * Setup: None.
 *
 *   The call of LwSciStream::isStatusEvent API with LwSciStreamEventType_ElementStatusConsumer
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::isStatusEvent()}
 */
TEST_F(lwscistreamcommon_unit_test,isStatusEvent_true3)
{

    EXPECT_TRUE(LwSciStream::isStatusEvent(LwSciStreamEventType_ElementStatusConsumer));
}

/**
 * @testname{lwscistreamcommon_unit_test.isStatusEvent_true4}
 * @testcase{}
 * @verify{22035187}
 * @testpurpose{Test positive scenario of LwSciStream::isStatusEvent.}
 * @testbehavior{
 * Setup: None.
 *
 *   The call of LwSciStream::isStatusEvent API with LwSciStreamEventType_ElementStatusProducer
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::isStatusEvent()}
 */
TEST_F(lwscistreamcommon_unit_test,isStatusEvent_true4)
{
    EXPECT_TRUE(LwSciStream::isStatusEvent(LwSciStreamEventType_ElementStatusProducer));
}

/**
 * @testname{lwscistreamcommon_unit_test.isStatusEvent_false5}
 * @testcase{}
 * @verify{22035187}
 * @testpurpose{Test positive scenario of LwSciStream::isStatusEvent.}
 * @testbehavior{
 * Setup: None.
 *
 *   The call of LwSciStream::isStatusEvent API with LwSciStreamEventType_PacketReady
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 *  Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - None.}
 * @verifyFunction{LwSciStream::isStatusEvent()}
 */
TEST_F(lwscistreamcommon_unit_test,isStatusEvent_false5)
{
    EXPECT_FALSE(LwSciStream::isStatusEvent(LwSciStreamEventType_PacketReady));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


