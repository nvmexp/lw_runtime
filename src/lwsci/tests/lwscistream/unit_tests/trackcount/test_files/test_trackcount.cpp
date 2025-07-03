//
// Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
/// @file

#include <unistd.h>
#include <thread>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "lwscistream.h"
#include "lwscistream_panic_mock.h"
#include "trackcount.h"

class trackcount_unit_test: public ::testing::Test {
public:
   trackcount_unit_test( ) {
       // Initialization code here
   }

   void SetUp( ) {
       // Code here will execute just before the test ensues
   }

   void TearDown( ) {
       // Code here will be called just after the test completes
       // OK to through exceptions from here if need be
   }

   ~trackcount_unit_test( )  {
       // Cleanup any pending stuff, but no exceptions and no gtest
       // ASSERT* allowed.
   }

   // Put in any custom data members that you need
};


class trackcount_action_unit_test: public ::testing::Test {
public:
   trackcount_action_unit_test( ) {
       // Initialization code here
   }

   void SetUp( ) {
       // Code here will execute just before the test ensues
   }

   void TearDown( ) {
       // Code here will be called just after the test completes
       // OK to through exceptions from here if need be
   }

   ~trackcount_action_unit_test( )  {
       // Cleanup any pending stuff, but no exceptions and no gtest
       // ASSERT* allowed.
   }

   // Put in any custom data members that you need
};


class trackcount_event_unit_test: public ::testing::Test {
public:
   trackcount_event_unit_test( ) {
       // Initialization code here
   }

   void SetUp( ) {
       // Code here will execute just before the test ensues
   }

   void TearDown( ) {
       // Code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~trackcount_event_unit_test( )  {
       // Cleanup any pending stuff, but no exceptions and no gtest
       // ASSERT* allowed.
   }

   // Put in any custom data members that you need
};


namespace std {

void action(LwSciStream::TrackCount const& count)
{
    //This function is ilwoked by TrackCountAction::pendingAction()
    //Time delay of 1 ms is introduced here
    usleep(10000);
}

}


namespace LwSciStream {

/**
 * @testname{trackcount_unit_test.set_Success1}
 * @testcase{21546724}
 * @verify{19525179}
 * @verify{19471323}
 * @verify{19471332}
 * @verify{19389003}
 * @verify{19389009}
 * @verify{19500588}
 * @verify{19500594}
 * @verify{19459773}
 * @verify{19460124}
 * @verify{19469874}
 * @testpurpose{Test positive scenario of TrackCount::set(), where
 * set() is called with value less than maximum allowed count.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 10 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::set() API from Count object,
 * should return LwSciError_Success, when called with value equal to 5.
 * Sets the count and maskBits values to 5 and 31 respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::set()}
 */
TEST_F(trackcount_unit_test, set_Success1)
{
    /*Initial setup*/
    uint32_t value = 5U;

    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, Count.set(value));

    EXPECT_EQ(5, Count.get());
    EXPECT_EQ(31, Count.mask());
}

/**
 * @testname{trackcount_unit_test.set_Success2}
 * @testcase{21546725}
 * @verify{19525179}
 * @verify{19471323}
 * @verify{19471332}
 * @verify{19389003}
 * @verify{19389009}
 * @verify{19500588}
 * @verify{19500594}
 * @verify{19459773}
 * @verify{19460124}
 * @testpurpose{Test positive scenario of TrackCount::set(), where
 * set() is called with value equal to maximum allowed count.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 10 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::set() API from Count object,
 * should return LwSciError_Success, when called with value equal to 10.
 * Sets the count and maskBits values to 10 and 1023 respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::set()}
 */
TEST_F(trackcount_unit_test, set_Success2)
{
    /*Initial setup*/
    uint32_t value = 10U;

    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, Count.set(value));

    EXPECT_EQ(10, Count.get());
    EXPECT_EQ(1023, Count.mask());
}

/**
 * @testname{trackcount_unit_test.set_Success3}
 * @testcase{21680563}
 * @verify{19525179}
 * @verify{19471323}
 * @verify{19471332}
 * @verify{19389003}
 * @verify{19389009}
 * @verify{19500588}
 * @verify{19500594}
 * @verify{19459773}
 * @verify{19460124}
 * @testpurpose{Test positive scenario of TrackCount::set(), where
 * set() is called with value and maximum allowed count equal to 64.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 64 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::set() API from Count object,
 * should return LwSciError_Success, when called with value equal to 64.
 * Sets the count and maskBits values to 64 and 18446744073709551615 respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::set()}
 */
TEST_F(trackcount_unit_test, set_Success3)
{
    /*Initial setup*/
    uint32_t value = 64U;

    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{64U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, Count.set(value));

    EXPECT_EQ(64, Count.get());
    EXPECT_EQ(18446744073709551615, Count.mask());
}

/**
 * @testname{trackcount_unit_test.set_Success4}
 * @testcase{22059365}
 * @verify{19525179}
 * @verify{19471323}
 * @verify{19471332}
 * @verify{19389003}
 * @verify{19389009}
 * @verify{19500588}
 * @verify{19500594}
 * @verify{19459773}
 * @verify{19460124}
 * @testpurpose{Test positive scenario of TrackCount::set(), where
 * set() is called with value equal to 0.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 64 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::set() API from Count object,
 * should return LwSciError_Success, when called with value equal to 0.
 * Sets the count and maskBits values to 0 and 0 respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::set()}
 */
TEST_F(trackcount_unit_test, set_Success4)
{
    /*Initial setup*/
    uint32_t value = 0U;

    //Create a TrackCount instance with maximum allowed count of 0.
    LwSciStream::TrackCount Count{0U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, Count.set(value));

    EXPECT_EQ(0, Count.get());
    EXPECT_EQ(0, Count.mask());
}

/**
 * @testname{trackcount_unit_test.set_BadParameter}
 * @testcase{21546726}
 * @verify{19525179}
 * @testpurpose{Test negative scenario of TrackCount::set(), where
 * set() is called with value greater than maximum allowed count.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 10 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::set() API from Count object,
 * should return LwSciError_BadParameter, when called with value equal to 15.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::set()}
 */
TEST_F(trackcount_unit_test, set_BadParameter)
{
    /*Initial setup*/
    uint32_t value = 15U;

    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_BadParameter, Count.set(value));
}

/**
 * @testname{trackcount_unit_test.set_IlwalidState}
 * @testcase{21546727}
 * @verify{19525179}
 * @testpurpose{Test negative scenario of TrackCount::set(), where
 * set() is called when the count value is already set.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCount instance with maximum allowed count of 10 by calling
 *      the TrackCount::TrackCount() constructor.
 *   2. Call TrackCount::set() with value equal to 5.
 *
 *   The call of TrackCount::set() API from Count object again with value equal to 7,
 * should return LwSciError_IlwalidState.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::set()}
 */
TEST_F(trackcount_unit_test, set_IlwalidState)
{
    /*Initial setup*/
    uint32_t value = 7U;

    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    //Set the count value to 5
    Count.set(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_IlwalidState, Count.set(value));
}


/**
 * @testname{trackcount_unit_test.get_Success}
 * @testcase{21546728}
 * @verify{19525182}
 * @verify{19389021}
 * @verify{19389024}
 * @testpurpose{Test positive scenario of TrackCount::get(), where
 * get() is called after setting the count value.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCount instance with maximum allowed count of 10 by calling
 *      the TrackCount::TrackCount() constructor.
 *   2. Set the count value to 5 by calling TrackCount::set().
 *
 *   The call of TrackCount::get() API from Count object,
 * should return 5, which is the value of count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::get()}
 */
TEST_F(trackcount_unit_test, get_Success)
{
    /*Initial setup*/
    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    //set the count value to 5.
    Count.set(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(5U, Count.get());
}

/**
 * @testname{trackcount_unit_test.get_Ilwalid}
 * @testcase{21546729}
 * @verify{19525182}
 * @testpurpose{Test negative scenario of TrackCount::get(), where
 * get() is called without setting the count value.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 10 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::get() API from Count object,
 * should return 65, which is an invalid count value.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::get()}
 */
TEST_F(trackcount_unit_test, get_Ilwalid)
{
    /*Initial setup*/
    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(65U, Count.get());
}


/**
 * @testname{trackcount_unit_test.checkSet_true}
 * @testcase{21546730}
 * @verify{19525185}
 * @testpurpose{Test positive scenario of TrackCount::checkSet(), where
 * checkSet() is called after setting the count value.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCount instance with maximum allowed count of 10 by calling
 *      the TrackCount::TrackCount() constructor.
 *   2. Set the count value to 5 by calling TrackCount::set().
 *
 *   The call of TrackCount::checkSet() API from Count object,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::checkSet()}
 */
TEST_F(trackcount_unit_test, checkSet_true)
{
    /*Initial setup*/
    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    //set the count value to 5.
    Count.set(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, Count.checkSet());
}

/**
 * @testname{trackcount_unit_test.checkSet_false}
 * @testcase{21546731}
 * @verify{19525185}
 * @testpurpose{Test negative scenario of TrackCount::checkSet(), where
 * checkSet() is called without setting the count value.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 10 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::checkSet() API from Count object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::checkSet()}
 */
TEST_F(trackcount_unit_test, checkSet_false)
{
    /*Initial setup*/
    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, Count.checkSet());
}


/**
 * @testname{trackcount_unit_test.mask_Success}
 * @testcase{21546732}
 * @verify{19525188}
 * @testpurpose{Test positive scenario of TrackCount::mask(), where
 * mask() is called after setting the count value.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCount instance with maximum allowed count of 10 by calling
 *      the TrackCount::TrackCount() constructor.
 *   2. Set the count value to 5 by calling TrackCount::set().
 *
 *   The call of TrackCount::mask() API from Count object,
 * should return 31, which is the maskBits value associated with count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::mask()}
 */
TEST_F(trackcount_unit_test, mask_Success)
{
    /*Initial setup*/
    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    //set the count value to 5.
    Count.set(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(31, Count.mask());
}

/**
 * @testname{trackcount_unit_test.mask_Ilwalid}
 * @testcase{21546733}
 * @verify{19525188}
 * @testpurpose{Test negative scenario of TrackCount::mask(), where
 * mask() is called without setting the count value.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCount instance with maximum allowed count of 10 by calling
 *   the TrackCount::TrackCount() constructor.
 *
 *   The call of TrackCount::mask() API from Count object,
 * should return 0, which is the maskBits value associated with an invalid count.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCount::mask()}
 */
TEST_F(trackcount_unit_test, mask_Ilwalid)
{
    /*Initial setup*/
    //Create a TrackCount instance with maximum allowed count of 10.
    LwSciStream::TrackCount Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(0, Count.mask());
}


/**
 * @testname{trackcount_action_unit_test.setDefault_Success1}
 * @testcase{21546734}
 * @verify{19525203}
 * @verify{19389003}
 * @verify{19389009}
 * @verify{19500588}
 * @verify{19500594}
 * @verify{19469874}
 * @testpurpose{Test positive scenario of TrackCountAction::setDefault(), where
 * setDefault() is called with value less than maximum allowed count.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCountAction instance with maximum allowed count of 10 by calling
 *   the TrackCountAction::TrackCountAction() constructor.
 *
 *   The call of TrackCountAction::setDefault() API from Count object,
 * should return LwSciError_Success, when called with value equal to 5.
 * Sets the default count value to 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountAction::setDefault()}
 */
TEST_F(trackcount_action_unit_test, setDefault_Success1)
{
    /*Initial setup*/
    uint32_t value = 5U;

    //Create a TrackCountAction instance with maximum allowed count of 10.
    LwSciStream::TrackCountAction Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, Count.setDefault(value));

    EXPECT_EQ(5, Count.get());
}

/**
 * @testname{trackcount_action_unit_test.setDefault_Success2}
 * @testcase{21546735}
 * @verify{19525203}
 * @testpurpose{Test positive scenario of TrackCountAction::setDefault(), where
 * setDefault() is called with value equal to maximum allowed count.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCountAction instance with maximum allowed count of 10 by calling
 *   the TrackCountAction::TrackCountAction() constructor.
 *
 *   The call of TrackCountAction::setDefault() API from Count object,
 * should return LwSciError_Success, when called with value equal to 10.
 * Sets the default count value to 10.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountAction::setDefault()}
 */
TEST_F(trackcount_action_unit_test, setDefault_Success2)
{
    /*Initial setup*/
    uint32_t value = 10U;

    //Create a TrackCountAction instance with maximum allowed count of 10.
    LwSciStream::TrackCountAction Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, Count.setDefault(value));

    EXPECT_EQ(10, Count.get());
}

/**
 * @testname{trackcount_action_unit_test.setDefault_Success3}
 * @testcase{21546736}
 * @verify{19525203}
 * @testpurpose{Test positive scenario of TrackCountAction::setDefault(), where
 * setDefault() is called when the default count value is already set.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCountAction instance with maximum allowed count of 10 by calling
 *      the TrackCountAction::TrackCountAction() constructor.
 *   2. Call TrackCountAction::setDefault() with value equal to 5.
 *
 *   The call of TrackCountAction::setDefault() API from Count object again,
 * should return LwSciError_Success, when called with value equal to 8.
 * The default count value is not changed and remains equal to 5.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountAction::setDefault()}
 */
TEST_F(trackcount_action_unit_test, setDefault_Success3)
{
    /*Initial setup*/
    uint32_t value = 8U;

    //Create a TrackCountAction instance with maximum allowed count of 10.
    LwSciStream::TrackCountAction Count{10U};

    //Set default count value to 5
    Count.setDefault(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, Count.setDefault(value));

    EXPECT_EQ(5, Count.get());
}

/**
 * @testname{trackcount_action_unit_test.setDefault_BadParameter}
 * @testcase{21546737}
 * @verify{19525203}
 * @testpurpose{Test negative scenario of TrackCountAction::setDefault(), where
 * setDefault() is called with value greater than maximum allowed count.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCountAction instance with maximum allowed count of 10 by calling
 *   the TrackCountAction::TrackCountAction() constructor.
 *
 *   The call of TrackCountAction::setDefault() API from Count object,
 * should return LwSciError_BadParameter, when called with value equal to 15.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountAction::setDefault()}
 */
TEST_F(trackcount_action_unit_test, setDefault_BadParameter)
{
    /*Initial setup*/
    uint32_t value = 15U;

    //Create a TrackCountAction instance with maximum allowed count of 10.
    LwSciStream::TrackCountAction Count{10U};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_BadParameter, Count.setDefault(value));
}


/**
 * @testname{trackcount_action_unit_test.pendingAction_Success1}
 * @testcase{21546740}
 * @verify{19525209}
 * @verify{19780917}
 * @verify{19389003}
 * @verify{19389009}
 * @verify{19500588}
 * @verify{19469874}
 * @testpurpose{Test positive scenario of TrackCountAction::pendingAction(), where
 * pendingAction() is called and the given function(action) is ilwoked.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCountAction instance with maximum allowed count of 10 by calling
 *      the TrackCountAction::TrackCountAction() constructor.
 *   2. Set the count value to 5 by calling TrackCountAction::setDefault().
 *
 *   The call of TrackCountAction::pendingAction() API from Count object,
 * with action set to the address of the given function std::action(), ilwokes it.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountAction::pendingAction()}
 */
TEST_F(trackcount_action_unit_test, pendingAction_Success1)
{
    /*Initial setup*/
    std::function<void(TrackCount const&)> action = std::action;

    //Create a TrackCountAction instance with maximum allowed count of 10.
    LwSciStream::TrackCountAction Count{10U};

    //set the count value to 5.
    Count.setDefault(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    Count.pendingAction(action);
}

/**
 * @testname{trackcount_action_unit_test.pendingAction_Success2}
 * @testcase{21546741}
 * @verify{19525209}
 * @testpurpose{Test positive scenario of TrackCountAction::pendingAction(), where
 * pendingAction() is called when the given function(action) is already ilwoked by another thread.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCountAction instance with maximum allowed count of 10 by calling
 *      the TrackCountAction::TrackCountAction() constructor.
 *   2. Set the count value to 5 by calling TrackCountAction::setDefault().
 *   3. A thread ilwokes the given function(action) by calling TrackCountAction::pendingAction().
 *
 *   The call of TrackCountAction::pendingAction() API from Count object by another thread,
 * should result in wait, until the first thread completes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountAction::pendingAction()}
 */
TEST_F(trackcount_action_unit_test, pendingAction_Success2)
{
    /*Initial setup*/
    std::function<void(TrackCount const&)> action = std::action;

    //Create a TrackCountAction instance with maximum allowed count of 10.
    LwSciStream::TrackCountAction Count{10U};

    //set the count value to 5.
    Count.setDefault(5U);

    //Call of TrackCount::pendingAction() by thread t1
    std::thread t1(&TrackCountAction::pendingAction, &Count, action);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    //Call of TrackCount::pendingAction() by thread t2
    std::thread t2(&TrackCountAction::pendingAction, &Count, action);


    //main thread waits for threads t1 & t2 to finish
    t1.join();
    t2.join();
}


/**
 * @testname{trackcount_action_unit_test.pendingAction_Panic}
 * @testcase{21680616}
 * @verify{19525209}
 * @testpurpose{Test negative scenario of TrackCountAction::pendingAction(), where
 * pendingAction() is called and the given function(action) is null.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCountAction instance with maximum allowed count of 10 by calling
 *      the TrackCountAction::TrackCountAction() constructor.
 *   2. Set the count value to 5 by calling TrackCountAction::setDefault().
 *   3. The given function(action) points to NULL.
 *
 *   The call of TrackCountAction::pendingAction() API from Count object,
 * with action set to NULL, should result in lwscistreamPanic() to be called.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciStream API calls are replaced with mocks:
 *   - lwscistreamPanic()}
 * @verifyFunction{TrackCountAction::pendingAction()}
 */
TEST_F(trackcount_action_unit_test, pendingAction_Panic)
{
    /*Initial setup*/
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Mock;

    /*Initial setup*/
    std::function<void(TrackCount const&)> action = nullptr;

    lwscistreamPanicMock npm;

    //Create a TrackCountAction instance with maximum allowed count of 10.
    LwSciStream::TrackCountAction Count{10U};

    //set the count value to 5.
    Count.setDefault(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_CALL(npm, lwscistreamPanic_m())
               .Times(1)
               .WillOnce(Return());

    Count.pendingAction(action);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&npm));
}

/**
 * @testname{trackcount_event_unit_test.checkDone_true}
 * @testcase{21546742}
 * @verify{19525221}
 * @testpurpose{Test positive scenario of TrackCountEvent::checkDone(), where
 * checkDone() is called when an event on count is generated.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCountEvent instance with maximum allowed count of 10
 *      and event type set to LwSciStreamEventType_SyncCount, by calling
 *      the TrackCountEvent::TrackCountEvent() constructor.
 *   2. Set the count value to 5 by calling TrackCountEvent::set().
 *   3. Check for any pending event on count and retrieve it by calling
 *      TrackCountEvent::pendingEvent().
 *
 *   The call of TrackCountEvent::checkDone() API from Count object,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountEvent::checkDone()}
 */
TEST_F(trackcount_event_unit_test, checkDone_true)
{
    /*Initial setup*/
    LwSciStreamEvent event;

    //Create a TrackCountEvent instance with maximum allowed count of 10,
    //for 'LwSciStreamEventType_SyncCount' event type.
    LwSciStream::TrackCountEvent Count{10U, LwSciStreamEventType_SyncCount};

    //set the count value to 5.
    Count.set(5U);

    //Check for pending event.
    Count.pendingEvent(event);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, Count.checkDone());
}

/**
 * @testname{trackcount_event_unit_test.checkDone_false}
 * @testcase{21546743}
 * @verify{19525221}
 * @testpurpose{Test negative scenario of TrackCountEvent::checkDone(), where
 * checkDone() is called when no event on count is generated.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCountEvent instance with maximum allowed count of 10
 *   and event type set to LwSciStreamEventType_SyncCount, by calling
 *   the TrackCountEvent::TrackCountEvent() constructor.
 *
 *   The call of TrackCountEvent::checkDone() API from Count object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountEvent::checkDone()}
 */
TEST_F(trackcount_event_unit_test, checkDone_false)
{
    /*Initial setup*/
    LwSciStreamEvent event;

    //Create a TrackCountEvent instance with maximum allowed count of 10,
    //for 'LwSciStreamEventType_SyncCount' event type.
    LwSciStream::TrackCountEvent Count{10U, LwSciStreamEventType_SyncCount};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, Count.checkDone());
}


/**
 * @testname{trackcount_event_unit_test.pendingEvent_true}
 * @testcase{21546744}
 * @verify{19525224}
 * @testpurpose{Test positive scenario of TrackCountEvent::pendingEvent(), where
 * pendingEvent() is called and the event on count is retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCountEvent instance with maximum allowed count of 10
 *      and event type set to LwSciStreamEventType_SyncCount, by calling
 *      the TrackCountEvent::TrackCountEvent() constructor.
 *   2. Set the count value to 5 by calling TrackCountEvent::set().
 *
 *   The call of TrackCountEvent::pendingEvent() API from Count object,
 * should return true. Fills in LwSciStreamEvent structure and marks it done.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountEvent::pendingEvent()}
 */
TEST_F(trackcount_event_unit_test, pendingEvent_true)
{
    /*Initial setup*/
    LwSciStreamEvent event;

    //Create a TrackCountEvent instance with maximum allowed count of 10,
    //for 'LwSciStreamEventType_SyncCount' event type.
    LwSciStream::TrackCountEvent Count{10U, LwSciStreamEventType_SyncCount};

    //set the count value to 5.
    Count.set(5U);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, Count.pendingEvent(event));

    EXPECT_EQ(LwSciStreamEventType_SyncCount, event.type);
    EXPECT_EQ(5, event.count);
    EXPECT_EQ(true, Count.checkDone());
}

/**
 * @testname{trackcount_event_unit_test.pendingEvent_false1}
 * @testcase{21546745}
 * @verify{19525224}
 * @testpurpose{Test negative scenario of TrackCountEvent::pendingEvent(), where
 * pendingEvent() is called before setting the count value.}
 * @testbehavior{
 * Setup:
 *   Create a TrackCountEvent instance with maximum allowed count of 10
 *   and event type set to LwSciStreamEventType_SyncCount, by calling
 *   the TrackCountEvent::TrackCountEvent() constructor.
 *
 *   The call of TrackCountEvent::pendingEvent() API from Count object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountEvent::pendingEvent()}
 */
TEST_F(trackcount_event_unit_test, pendingEvent_false1)
{
    /*Initial setup*/
    LwSciStreamEvent event;

    //Create a TrackCountEvent instance with maximum allowed count of 10,
    //for 'LwSciStreamEventType_SyncCount' event type.
    LwSciStream::TrackCountEvent Count{10U, LwSciStreamEventType_SyncCount};

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, Count.pendingEvent(event));
}

/**
 * @testname{trackcount_event_unit_test.pendingEvent_false2}
 * @testcase{21680641}
 * @verify{19525224}
 * @testpurpose{Test negative scenario of TrackCountEvent::pendingEvent(), where
 * pendingEvent() is called after the event on count is already retrieved.}
 * @testbehavior{
 * Setup:
 *   1. Create a TrackCountEvent instance with maximum allowed count of 10
 *      and event type set to LwSciStreamEventType_SyncCount, by calling
 *      the TrackCountEvent::TrackCountEvent() constructor.
 *   2. Set the count value to 5 by calling TrackCountEvent::set().
 *   3. Call TrackCountEvent::pendingEvent() with a valid LwSciStreamEvent object.
 *
 *   The call of TrackCountEvent::pendingEvent() API again,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{TrackCountEvent::pendingEvent()}
 */
TEST_F(trackcount_event_unit_test, pendingEvent_false2)
{
    /*Initial setup*/
    LwSciStreamEvent event;

    //Create a TrackCountEvent instance with maximum allowed count of 10,
    //for 'LwSciStreamEventType_SyncCount' event type.
    LwSciStream::TrackCountEvent Count{10U, LwSciStreamEventType_SyncCount};

    //set the count value to 5.
    Count.set(5U);

    //Retrieve the pending event on Count
    Count.pendingEvent(event);

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, Count.pendingEvent(event));
}

} // namespace LwSciStream


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
