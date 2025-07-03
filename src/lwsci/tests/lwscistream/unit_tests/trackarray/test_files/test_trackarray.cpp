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
#include "lwscistream_common.h"
#include "producer.h"
#include "pool.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include "lwscistream_types.h"
#include "lwscistream_LwSciCommonPanic_mock.h"

bool flag1 = false;
bool flag2 = false;
bool flag3 = false;
bool flag4 = false;

class trackarray_unit_test : public LwSciStreamTest
{
public:
    trackarray_unit_test()
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

    ~trackarray_unit_test()
    {
        // cleanup any pending stuff, but no exceptions allowed
    }

    // put in any custom data members that you need
};

/**
 * @testname{trackarray_unit_test.TrackBits.SetOnce_true1}
 * @testcase{21730786}
 * @verify{19406814}
 * @testpurpose{Test positive scenario of TrackBits::setOnce(),
 * to set specified bits if none of them are already set and return true after successful
 * setting of the bits to 0.}
 * @testbehavior{
 * Setup:
 *     Create TrackBits object obj.
 *
 * Call to TrackBits::setOnce(),with 0U as argument,should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Boundary value}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   None of the specified bits are set}
 * @verifyFunction{TrackBits::setOnce()}
 */
TEST_F(trackarray_unit_test, SetOnce_true1)
{
    /* initial set up */

    LwSciStream::TrackBits obj;

    /*Test case*/

    //minimum value
    bool isTrue = obj.setOnce(0);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.SetOnce_true2}
 * @testcase{21730793}
 * @verify{19406814}
 * @testpurpose{Test positive scenario of TrackBits::setOnce(), to set
 * specified bits, if none of them are already set and return true after successful setting of the bits to 500.}
 * @testbehavior{
 * Setup:
 *     Create TrackBits object obj.
 *
 * Call to TrackBits::setOnce(),with 500U as argument,should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   None of the specified bits are set}
 * @verifyFunction{TrackBits::setOnce()}
 */
TEST_F(trackarray_unit_test, SetOnce_true2)
{
    /* initial set up */

    LwSciStream::TrackBits obj;

    /*Test case*/

    //minimum value
    bool isTrue = obj.setOnce(500);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.SetOnce_true3}
 * @testcase{21730799}
 * @verify{19406814}
 * @testpurpose{Test positive scenario of TrackBits::setOnce()
 * set specified bits if none of them are already set and return true after successful setting
 * of the bits to UINT64_MAX.}
 * @testbehavior{
 * Setup:
 *     Create TrackBits object obj.
 *
 * Call to TrackBits::setOnce(),with UINT64_MAX as argument,should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Boundary value}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   None of the specified bits are set}
 * @verifyFunction{TrackBits::setOnce()}
 */
TEST_F(trackarray_unit_test, SetOnce_true3)
{
    /* initial set up */

    LwSciStream::TrackBits obj;

    /*Test case*/

    //maximum value
    bool isTrue = obj.setOnce(UINT64_MAX);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits_setOnce_false2}
 * @testcase{22058918}
 * @verify{19406814}
 * @testpurpose{Test negative scenario of TrackBits::setOnce().}
 * @testbehavior{
 * Setup:
 *     1.Create TrackBits object obj.
 *     2.Call to TrackBits::setOnce(), with 3U as parameter.
 *
 * Again call of TrackBits::setOnce(), with 5U as parameter should return
 * false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::setOnce()}
 */
TEST_F(trackarray_unit_test, TrackBits_setOnce_false2)
{
    /* initial set up */

    LwSciStream::TrackBits obj;

    /*Test case*/

    //maximum value
    bool isTrue = obj.setOnce(3U);
    EXPECT_TRUE(isTrue);

    isTrue = obj.setOnce(5U);
    EXPECT_FALSE(isTrue);
}


/**
 * @testname{trackarray_unit_test.TrackBits.set_true1}
 * @testcase{21730806}
 * @verify{19406823}
 * @testpurpose{Test positive scenario of TrackBits::set(), when no bits are set.}
 * @testbehavior{
 * Setup:
 *     Creates TrackBits object obj.
 *
 * Call TrackBits::set() with 0U as argument and check if specified bits are set by calling
 * TrackBits::check() with 0U as argument,the TrackBits::check() should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Boundary value}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   None of the specified bits are set already}
 * @verifyFunction{TrackBits::set()}
 */
TEST_F(trackarray_unit_test, set_true1)
{
    /* initial set up */

    LwSciStream::TrackBits obj;
    bool isTrue;

    /*Test case*/

    obj.set(0U);
    isTrue = obj.check(0U);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.set_true2}
 * @testcase{21730814}
 * @verify{19406823}
 * @testpurpose{Test positive scenario of TrackBits::set(), to set bits with 5.}
 * @testbehavior{
 * Setup:
 *     Creates TrackBits object obj.
 *
 * Setting the bits through TrackBits::set() to 5 and further checking the set bits through
 * TrackBits::check(), should return true.}
  @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   None of the specified bits are set already}
 * @verifyFunction{TrackBits::set()}
 */
TEST_F(trackarray_unit_test, set_true2)
{
    /* initial set up */

    LwSciStream::TrackBits obj;
    bool isTrue;
    isTrue = obj.check(5U);
    EXPECT_FALSE(isTrue);

    /*Test case*/

    obj.set(5U);
    isTrue = obj.check(5U);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.set_true3}
 * @testcase{21730820}
 * @verify{19406823}
 * @testpurpose{Test positive scenario of TrackBits::set(), to set bits with UINT64_MAX.}
 * @testbehavior{
 * Setup:
 *     Create TrackBits object obj.
 *
 * Call TrackBits::set() with UINT64_MAX as argument and check if specified bits are set by calling
 * TrackBits::check() with UINT_MAX as argument. The TrackBits::check() with UINT64_MAX as parameter
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Boundary value}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *  Precondition:
 *   None of the specified bits are set already}
 * @verifyFunction{TrackBits::set()}
 */
TEST_F(trackarray_unit_test,set_true3)
{
    /* initial set up */

    LwSciStream::TrackBits obj;
    bool isTrue;
    isTrue = obj.check(UINT64_MAX);
    EXPECT_FALSE(isTrue);

    /*Test case*/

    obj.set(UINT64_MAX);
    isTrue = obj.check(UINT64_MAX);
    EXPECT_TRUE(isTrue);
}


/**
 * @testname{trackarray_unit_test.TrackBits.check_true1}
 * @testcase{21730846}
 * @verify{19406829}
 * @testpurpose{Test positive scenario of TrackBits::check() ,to check whether
 * specified bits are set}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *
 * Call to TrackBits::set(), with 300U as parameter.
 * Call to TrackBits::check(), with 300U as parameter, should return true}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::check()}
 */
TEST_F(trackarray_unit_test, check_true1)
{

    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;

    /*test case*/

    //set bit
    obj.set(300U);
    //verify with check
    isTrue = obj.check(300U);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.check_true2}
 * @testcase{21730870}
 * @verify{19406829}
 * @testpurpose{Test positive scenario of TrackBits::check(), checks whether
 * specified bits are set.}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *
 * Call to TrackBits::set(), with UINT64_MAX as parameter.
 * Call to TrackBits::check(), with UINT64_MAX as parameter, should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::check()}
 */
TEST_F(trackarray_unit_test, check_true2)
{

    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;

    /*test case*/

    //set bit
    obj.set(UINT64_MAX);
    //verify with check
    isTrue = obj.check(UINT64_MAX);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.TrackBits_extract_true1}
 * @testcase{21730877}
 * @verify{19406832}
 * @testpurpose{Test positive scenario of TrackBits::extract().}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with 0U as parameter.
 *     3.Call to TrackBits::check(),with 0U as parameter.
 *
 * Call to TrackBits::extract() with 0U as parameter, should return 0U as no bits are set.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_true1)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
      uint32_t res;

    /*test case*/

    //set and verify the bit
    bool isTrue;
    obj.set(0U);
    isTrue = obj.check(0U);
    EXPECT_TRUE(isTrue);

    //extract the single set lsb & reset it
    res = obj.extract();
    EXPECT_EQ(0U, res);
    //verify reset
    isTrue = obj.check(0U);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.TrackBits_extract_true2}
 * @testcase{21730883}
 * @verify{19406832}
 * @testpurpose{Test positive scenario of TrackBits::extract(), extract the 1U and reset.}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with 5U as parameter.
 *     3.Call to TrackBits::check(), with 5U as parameter.
 *
 * Call to TrackBits::extract() should extract 1U and reset the bit.
 * Also call to TrackBits::extract(), with 4U as parameter should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_true2)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    uint32_t res;

    /*test case*/

    //set and verify the bit
    bool isTrue;
    obj.set(5U);

    isTrue = obj.check(5U);
    EXPECT_TRUE(isTrue);

    //extract the single set lsb & reset it
    res = obj.extract();
    EXPECT_EQ(1U, res);

    res = obj.extract(4U);
    EXPECT_EQ(1U, res);

    //verify reset
    isTrue = obj.check(1U);
    EXPECT_FALSE(isTrue);

    isTrue = obj.check(4U);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits.TrackBits_extract_true3}
 * @testcase{21730890}
 * @verify{19406832}
 * @testpurpose{Test positive scenario of TrackBits::extract() , extract the 1U and reset}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with UINT64_MAX as parameter to set all bits to 1.
 *     3.Call to TrackBits::check(), with UINT64_MAX as parameter.
 *
 * Call to TrackBits::extract() should extract 1U and reset the bit.
 * Also call to TrackBits::extract(), with UINT64_MAX-1U as parameter should return true.
 * The call to TrackBits::check(), with 0U as a parameter should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_true3)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
     uint32_t res;

    /*test case*/

    //set and verify the bit
    bool isTrue;
    obj.set(UINT64_MAX);
    isTrue = obj.check(UINT64_MAX);
    EXPECT_TRUE(isTrue);

    //extract the single set lsb & reset it
    res = obj.extract();
    EXPECT_EQ(1U, res);

    isTrue = obj.extract(UINT64_MAX-1U);
    EXPECT_TRUE(isTrue);
        //verify reset

    isTrue = obj.check(0U);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits_extract_true4}
 * @testcase{21730897}
 * @verify{19406835}
 * @testpurpose{Test positive scenario of TrackBits::extract(),clears if all specified bits are set}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with 0U as parameter.
 *     3.Call to TrackBits::check(), with 0U as parameter.
 *
 * Call to TrackBits::extract(), with 0U as parameter,should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_true4)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;

    /*test case*/

    //check the extract is successful or not
    obj.set(0U);
    isTrue = obj.check(0U);
    EXPECT_TRUE(isTrue);
    isTrue = obj.extract(0U);
    EXPECT_TRUE(isTrue);
    //verify reset
    isTrue = obj.check(0U);
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits_extract_true5}
 * @testcase{21730955}
 * @verify{19406835}
 * @testpurpose{Test positive scenario of TrackBits::extract(),clears if all specified bits are set}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with 5U as parameter.
 *     3.Call to TrackBits::set(), with 3U as parameter.
 *     4.Call to TrackBits::check(), with 5U as parameter, should return true.
 *     5.Call to TrackBits::check(), with 3U as parameter, should return true..
 *
 * Call to TrackBits::extract(), with 5U as parameter,should return true.
 * Call to TrackBits::check() with 5U and 3U should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_true5)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;

    /*test case*/

    //check the extract is successful or not
    obj.set(5U);
    obj.set(3U);


    isTrue = obj.check(5U);
    EXPECT_TRUE(isTrue);
    isTrue = obj.extract(5U);
    EXPECT_TRUE(isTrue);
    isTrue = obj.check(3U);
    EXPECT_FALSE(isTrue);
    //verify reset
    isTrue = obj.check(5U);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits_extract_true6}
 * @testcase{21730961}
 * @verify{19406835}
 * @testpurpose{Test positive scenario of TrackBits::extract() ,clears if all specified bits
 * are set}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with UINT64_MAX as parameter.
 *     3.Call to TrackBits::check(), with UINT64_MAX as parameter.
 *
 * Call to TrackBits::extract(), with UINT64_MAX as parameter, should return true.
 * Call of obj.check(UINT64_MAX) should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_true6)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;
    //check the extract is successful or not
    obj.set(UINT64_MAX);
    isTrue = obj.check(UINT64_MAX);
    EXPECT_TRUE(isTrue);
    isTrue = obj.extract(UINT64_MAX);
    EXPECT_TRUE(isTrue);
    //verify reset
    isTrue = obj.check(UINT64_MAX);
    EXPECT_FALSE(isTrue);
}
//Dummy test user functions
void dummy1()
{
    flag1 = true;
}

void dummy2()
{
    flag2 = true;

}

void dummy3()
{
    flag3 = true;

}

void dummy4()
{
        //for throwing exception
    flag4 = true;
    std::function<int()> f = nullptr;
    f();
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_performAction_true1}
 * @testcase{21730968}
 * @verify{19406898}
 * @verify{19389000}
 * @verify{19389006}
 * @verify{19389012}
 * @testpurpose{Test positive scenario of TrackArrayAction::performAction()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with argument 2U.
 *     4.Call to TrackArrayAction::init(), with argument obj created in step 2.
 *
 * Call to TrackArrayAction::performAction(), with 0 and address of a function as arguments, should
 * claim the index and call the specified function and return LwSciError_Success}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::performAction()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_performAction_true1)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(true, flag1);
    EXPECT_EQ(LwSciError_Success, ret);
    flag1 = false;
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_performAction_true2}
 * @testcase{21730981}
 * @verify{19406898}
 * @testpurpose{Test positive scenario of TrackArrayAction::performAction()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as argument.
 *     4.Call to TrackArrayAction::init(), with argument obj created in step 2.
 *
 * Call to TrackArrayAction::performAction(), with 1 and address of a function as arguments,
 * performAction API should claim index and performs the specified action and
 * return LwSciError_Success}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::performAction()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_performAction_true2)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    ret = action.performAction(1, &dummy2);
    EXPECT_EQ(LwSciError_Success, ret);
    EXPECT_EQ(true, flag2);
    flag2 = false;
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_checkDone_true1}
 * @testcase{21730991}
 * @verify{19406901}
 * @testpurpose{Test positive scenario of TrackArrayAction::checkDone()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as parameter.
 *     4.Call to TrackArrayAction::init(), with obj as parameter created in step 2.
 *     5.Calls to TrackArrayAction::performAction() for both index 1 and 2 should return LwSciError_Success.
 *
 * Call to TrackArrayAction::checkDone(), should return true once all entries are done}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::checkDone()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_checkDone_true1)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    bool isTrue;
    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = action.performAction(1, &dummy2);
    EXPECT_EQ(LwSciError_Success, ret);
    isTrue = action.checkDone();
    EXPECT_TRUE(isTrue);
    EXPECT_EQ(true, flag1);
    flag1 = false;
    EXPECT_EQ(true, flag2);
    flag2 = false;
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_cycle_true1}
 * @testcase{21731006}
 * @verify{19406904}
 * @testpurpose{Test positive scenario of TrackArrayAction::cycle()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with  2U as parameter.
 *     4.Call to TrackArrayAction::init(), with object obj as parameter.
 *     5.Calls to TrackArrayAction::performAction() for both index 1 and 2 should return LwSciError_Success.
 *
 * Call to TrackArrayAction::cycle(), should return true and should reset entries successfully,
 * second round should work fine for the same entires}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::cycle()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_cycle_true1)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    bool isTrue;
    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = action.performAction(1, &dummy2);
    EXPECT_EQ(LwSciError_Success, ret);
    isTrue = action.cycle();
    EXPECT_TRUE(isTrue);
    /*restart another round*/
    ret = action.performAction(1, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = action.performAction(0, &dummy2);
    EXPECT_EQ(LwSciError_Success, ret);
    EXPECT_EQ(true, flag1);
    flag1 = false;
    EXPECT_EQ(true, flag2);
    flag2 = false;
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_cycle_true2}
 * @testcase{21731011}
 * @verify{19406904}
 * @testpurpose{Test positive scenario of TrackArrayAction::cycle()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 3U as parameter.
 *     4.Call to TrackArrayAction::init(), with object obj created in step 2 as parameter.
 *     5.Calls to TrackArrayAction::performAction() for indices 0, 1 and 2, should return LwSciError_Success.
 *
 * Call to TrackArrayAction::cycle(), should return true and should reset entries successfully,
 * second round of TrackArrayAction::performAction() should work fine with same entries}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::cycle()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_cycle_true2)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{3U};
    unsigned ret;
    ret = obj.set(3);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    bool isTrue;
    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = action.performAction(1, &dummy2);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = action.performAction(2, &dummy3);
    EXPECT_EQ(LwSciError_Success, ret);
    isTrue = action.cycle();
    EXPECT_TRUE(isTrue);
    /*restart another round*/
    ret = action.performAction(1, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = action.performAction(0, &dummy2);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = action.performAction(2, &dummy3);
    EXPECT_EQ(LwSciError_Success, ret);
    EXPECT_EQ(true, flag1);
    flag1 = false;
    EXPECT_EQ(true, flag2);
    flag2 = false;
    EXPECT_EQ(true, flag3);
    flag3 = false;
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_prepareEvent_true1}
 * @testcase{21731014}
 * @verify{19406931}
 * @verify{19389030}
 * @verify{19389036}
 * @verify{19389042}
 * @testpurpose{Test positive scenario of TrackArrayEvent::prepareEvent()}
 * @testbehavior{
 * Setup:
 *     1.Creates LwSciStream::TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(),with 2U as parameter.
 *     4.Call to TrackArrayEvent::init(), with obj and LwSciStreamEventType_PacketReady as parameters.
 *
 * Call to TrackArrayEvent::prepareEvent() with index 1U as one argument and 10U as second
 * argument should return LwSciError_Success }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayEvent::prepareEvent()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_prepareEvent_true1)
{
    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{2U};

    /*test case*/

    /*initialize */
    uint32_t ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);

    /*prepare evnt*/
    ret = event.prepareEvent(1U, 10U);
    EXPECT_EQ(LwSciError_Success, ret);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_pendingEvent_true2}
 * @testcase{21731020}
 * @verify{19406934}
 * @testpurpose{Test positive scenario of TrackArrayEvent::pendingEvent()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as parameter.
 *     4.Call to TrackArrayEvent::init(), with obj and LwSciStreamEventType_PacketReady parameters.
 *     5.Call to TrackArrayEvent::prepareEvent() with index=0U as one argument and  uint32_t
 *       10 as second argument.
 *     6.Call to TrackArrayEvent::prepareEvent() with index=1U as one argument and  uint32_t
 *       20 as second argument.
 *
 * The fist call to TrackArrayEvent::pendingEvent(), should return true and the output
 * LwSciStreamEvent type should be LwSciStreamEventType_PacketReady and error value should be 10
 * and the index should be 0U.
 * The second call to TrackArrayEvent::pendingEvent() should return true and the output
 * LwSciStreamEvent type should be LwSciStreamEventType_PacketReady and error value should be 20
 * and the index should be 1U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayEvent::pendingEvent()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_pendingEvent_true2)
{

    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{2U};

    /*test case*/
    /*initialize */
    uint32_t ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);
    LwSciStreamEvent out_event;

    /*prepare evnt*/
    ret = event.prepareEvent(0U, 10U);
    EXPECT_EQ(LwSciError_Success, ret);

    /*prepare evnt*/
    ret = event.prepareEvent(1U, 20U);
    EXPECT_EQ(LwSciError_Success, ret);


    isTrue = event.pendingEvent(out_event);
    EXPECT_EQ(out_event.type, LwSciStreamEventType_PacketReady);
    EXPECT_EQ(out_event.index, 0U);
    EXPECT_EQ(out_event.error, 10);
    EXPECT_TRUE(isTrue);

    isTrue = event.pendingEvent(out_event);
    EXPECT_EQ(out_event.type, LwSciStreamEventType_PacketReady);
    EXPECT_EQ(out_event.index, 1U);
    EXPECT_EQ(out_event.error, 20);

    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_checkUsed_true1}
 * @testcase{21731035}
 * @verify{19406937}
 * @testpurpose{Test positive scenario of TrackArrayEvent::checkUsed()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as parameter.
 *     4.Call to TrackArrayEvent::init(), with obj and LwSciStreamEventType_PacketReady as parameters.
 *     5.Call to TrackArrayEvent::prepareEvent() for both the indexes 0 and 1.
 *
 * Call to TrackArrayEvent::checkUsed() should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayEvent::checkUsed()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_checkUsed_true1)
{
    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{2U};

    /*test case*/

    //initialize
    uint32_t ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);

    ret = event.prepareEvent(0, 10);
    EXPECT_EQ(LwSciError_Success, ret);
    ret = event.prepareEvent(1, 10);
    EXPECT_EQ(LwSciError_Success, ret);

    isTrue = event.checkUsed();
    EXPECT_TRUE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_checkDone_true1}
 * @testcase{21731038}
 * @verify{19406940}
 * @testpurpose{Test positive scenario of TrackArrayEvent::checkDone()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 1U as parameter.
 *     4.Call to TrackArrayEvent::init(), with obj and LwSciStreamEventType_PacketReady parameters.
 *     5.Call to TrackArrayEvent::prepareEvent() with index as one argument and value as second
 * argument.
 *
 * Call to TrackArrayEvent::checkDone() with input argument, whose bit mask is equal to set bits,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayEvent::checkDone()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_checkDone_true1)
{
    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{2U};

    /*test case*/

    //initialize
    uint32_t ret;
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);

    ret = event.prepareEvent(0, 10);
    EXPECT_EQ(LwSciError_Success, ret);

    isTrue = event.checkDone(0U);
    EXPECT_TRUE(isTrue);
}

/*Negative test cases*/

/**
 * @testname{trackarray_unit_test.TrackBits_setOnce_false1}
 * @testcase{21731040}
 * @verify{19406814}
 * @testpurpose{Test negative scenario of TrackBits::setOnce().}
 * @testbehavior{
 * Setup:
 *     1.Create TrackBits object obj.
 *     2.Call to TrackBits::setOnce(), with 5U as parameter.
 *
 * Again call of TrackBits::setOnce(), with 5U as parameter should return
 * false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::setOnce()}
 */
TEST_F(trackarray_unit_test, TrackBits_setOnce_false1)
{
    /* initial set up */

    LwSciStream::TrackBits obj;
    bool isTrue = obj.setOnce(5u);

    /*Test case*/

    isTrue = obj.setOnce(5u);
    EXPECT_FALSE(isTrue);
}



/**
 * @testname{trackarray_unit_test.TrackBits_check_false2}
 * @testcase{21731047}
 * @verify{19406829}
 * @testpurpose{Test negative scenario of TrackBits::check().}
 * @testbehavior{
 * Setup:
 *     Creates TrackBits object obj.
 *
 * Call TrackBits::set(),with 1U as parameter and TrackBits::check(), with 2U as parameter,
 * should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::check()}
 */
TEST_F(trackarray_unit_test, TrackBits_check_false2)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;

    /*test case*/

    //set 1st bit
    obj.set(1U);
    //verify 2nd bit should not set
    isTrue = obj.check(2U);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits_check_false3}
 * @testcase{22058921}
 * @verify{19406829}
 * @testpurpose{Test negative scenario of TrackBits::check().}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *
 * Call TrackBits::set(),with 1U as parameter and TrackBits::check(), with 3U as parameter,
 * should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackBits::check()}
 */
TEST_F(trackarray_unit_test, TrackBits_check_false3)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;

    /*test case*/

    //set 1st bit
    obj.set(1U);
    //verify 2nd bit should not set
    isTrue = obj.check(3U);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits_extract_false2}
 * @testcase{21731049}
 * @verify{19406832}
 * @testpurpose{Test negative scenario of TrackBits::extract() ,clears if the specified bits
 * are set.}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with 5U as parameter.
 *     3.Call to TrackBits::check(), with 5U as parameter.
 *
 * Call to TrackBits::extract(), with 3U as parameter,should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_false2)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;
    obj.set(5U);
    isTrue = obj.check(5U);
    EXPECT_TRUE(isTrue);

    /*test case*/

    //check the extract failure
    isTrue = obj.extract(3U);
    EXPECT_FALSE(isTrue);
    //verify reset as already 2U bits are null it should return boolean false
    isTrue = obj.check(3U);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackBits_extract_false3}
 * @testcase{22058924}
 * @verify{19406832}
 * @testpurpose{Test negative scenario of TrackBits::extract() ,clears if the specified bits
 * are set.}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackBits object obj.
 *     2.Call to TrackBits::set(), with 5U as parameter.
 *     3.Call to TrackBits::check(), with 5U as parameter.
 *
 * Call to TrackBits::extract(), with 3U as parameter,should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackBits::extract()}
 */
TEST_F(trackarray_unit_test, TrackBits_extract_false3)
{
    /*initial set up*/

    LwSciStream::TrackBits obj;
    bool isTrue;
    obj.set(5U);
    isTrue = obj.check(5U);
    EXPECT_TRUE(isTrue);

    /*test case*/

    //check the extract failure
    isTrue = obj.extract(3U);
    EXPECT_FALSE(isTrue);
    //verify reset as already 2U bits are null it should return boolean false
    isTrue = obj.check(3U);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_performAction_false1}
 * @testcase{21731052}
 * @verify{19406898}
 * @testpurpose{Test negative scenario of TrackArrayAction::performAction()}
 * @testbehavior{
 * Setup:
 *     Creates TrackArrayAction object action.
 *
 * Call to TrackArrayAction::performAction(), should return LwSciError_IlwalidState as
 * action.init() not done}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::performAction()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_performAction_false1)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);

    /*test case */
    //action.init not performed it should return invalid state
    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(LwSciError_IlwalidState, ret);
}


/**
 * @testname{trackarray_unit_test.TrackArrayAction_performAction_false2}
 * @testcase{21731054}
 * @verify{19406898}
 * @testpurpose{Test negative scenario of TrackArrayAction::performAction()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as parameter.
 *     4.Call TrackArrayAction::init(), with object obj created in step 2 as parameter.
 *
 * Call to TrackArrayAction::performAction(), with 5U and address of a function as parameters should
 * return LwSciError_BadParameter}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::performAction()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_performAction_false2)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);

    /*test case */

    //after init ,if the index value here 5 which is invalid should return LwSciError_BadParameter
    action.init(obj);
    ret = action.performAction(5, &dummy1);
    EXPECT_EQ(LwSciError_BadParameter, ret);

}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_performAction_false3}
 * @testcase{21731057}
 * @verify{19406898}
 * @testpurpose{Test negative scenario of TrackArrayAction::performAction()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as parameter.
 *     4.Call TrackArrayAction::init(), with object obj created in step 2 as parameter.
 *     5.Call TrackArrayAction::performAction(), with 0U and address of a function as parameters.
 *
 * Call to TrackArrayAction::performAction(), with 0  and address of a function should return
 * LwSciError_IlwalidState}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayAction::performAction()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_performAction_false3)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);

    /*test case */

    //do init
    action.init(obj);

    //if the index is already claimed should return LwSciError_IlwalidState
    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);

    ret = action.performAction(0, &dummy2);
    EXPECT_EQ(LwSciError_IlwalidState, ret);

}


/**
 * @testname{trackarray_unit_test.TrackArrayAction_checkDone_false1}
 * @testcase{21731071}
 * @verify{19406901}
 * @testpurpose{Test negative scenario of TrackArrayAction::checkDone()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(),with 2U as argument.
 *     4.Call to TrackArrayAction::init(), with object obj created in step 2 as parameter.
 *     5.Call to TrackArrayAction::PerformAction(), with 0u and address of a function as parameters.
 *
 * Call to  TrackArrayAction::checkDone(), should return false as only one entry is done.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayAction::checkDone()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_checkDone_false1)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    bool isTrue;
    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);
    // As only one entry is done so it should return false
    isTrue = action.checkDone();
    EXPECT_FALSE(isTrue);

    EXPECT_EQ(true, flag1);
    flag1 = false;
}


/**
 * @testname{trackarray_unit_test.TrackArrayAction_checkDone_false2}
 * @testcase{21731074}
 * @verify{19406901}
 * @testpurpose{Test negative scenario of TrackArrayAction::checkDone()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2u as parameter.
 *
 * Call to  TrackArrayAction::checkDone(), should return false as TrackArrayAction.init() is not done.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayAction::checkDone()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_checkDone_false2)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);

    /*test case */

    bool isTrue;
    // As action.init is not performed it should return false
    isTrue = action.checkDone();
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_cycle_false1}
 * @testcase{21731077}
 * @verify{19406904}
 * @testpurpose{Test negative scenario of TrackArrayAction::cycle()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as parameter.
 *     4.Call to TrackArrayAction::init() with object obj created in step 2 as parameter.
 *     5.Call to TrackArrayAction::performAction(),with 0 as one argument and address of a function.
 *
 * Call to TrackArrayAction::cycle(), should return false as all entries are not done}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayAction::cycle()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_cycle_false1)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    bool isTrue;
    ret = action.performAction(0, &dummy1);
    EXPECT_EQ(LwSciError_Success, ret);

    //all entries are not done so it should return false boolean
    isTrue = action.cycle();
    EXPECT_FALSE(isTrue);

    EXPECT_EQ(true, flag1);
    flag1 = false;
}

/**
 * @testname{trackarray_unit_test.TrackArrayAction_cycle_false2}
 * @testcase{21731081}
 * @verify{19406904}
 * @testpurpose{Test negative scenario of TrackArrayAction::cycle()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 1u as parameter.
 *
 * Call to TrackArrayAction::cycle(), should return false as TrackArrayAction.init() is not done.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::cycle()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_cycle_false2)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);

    /*test case */

    bool isTrue;
    //as action.init not done should return false
    isTrue = action.cycle();
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_prepareEvent_false1}
 * @testcase{21731084}
 * @verify{19406931}
 * @testpurpose{Test negative scenario of TrackArrayEvent::prepareEvent()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *
 * Call to TrackArrayEvent::prepareEvent() with index=0 and value=10 ,should return
 * LwSciError_IlwalidState as event.init is not performed}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayEvent::prepareEvent()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_prepareEvent_false1)
{
    /*initial set up*/
    uint32_t ret;
    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{1U};

    /*test case*/

    //prepare event should return LwSciError_IlwalidState as the TrackArrayEvent instance
    //is not initialized.
    ret = event.prepareEvent(0, 10);
    EXPECT_EQ(LwSciError_IlwalidState, ret);
}


/**
 * @testname{trackarray_unit_test.TrackArrayEvent_prepareEvent_false2}
 * @testcase{21731087}
 * @verify{19406931}
 * @testpurpose{Test negative scenario of TrackArrayEvent::prepareEvent()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call Trackcount::set(),with 1U as parameter.
 *     3.Call TrackArrayEvent::init() with object obj created in step 2 and
 *       LwSciStreamEventType_PacketReady as parameters.
 *
 * Call to TrackArrayEvent::prepareEvent() with index 2 which is greater than the trackCount value of 1
 * and val of 10 as parameters,should return LwSciError_BadParameter, as here index is invalid}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayEvent::prepareEvent()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_prepareEvent_false2)
{
    /*initial set up*/
    uint32_t ret;
    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{1U};

    /*test case*/

    //initialize and give invalid index should return LwSciError_BadParameter
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);
    ret = event.prepareEvent(2, 10);
    EXPECT_EQ(LwSciError_BadParameter, ret);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_prepareEvent_false3}
 * @testcase{21731089}
 * @verify{19406931}
 * @testpurpose{Test negative scenario of TrackArrayEvent::prepareEvent()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call TrackArrayEvent::init() with object obj created in step 2 and
 *       LwSciStreamEventType_PacketReady as parameters.
 *     4.Call TrackArrayEvent::prepareEvent(), with 0U and 10U as parameters.
 *
 * Call to TrackArrayEvent::prepareEvent(), with 0U and 10U as parameters, should return
 * LwSciError_IlwalidState, as here index is already claimed}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayEvent::prepareEvent()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_prepareEvent_false3)
{
    /*initial set up*/
    uint32_t ret;
    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{1U};

    /*test case*/

    //initialize
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);
    //if the entry in the given index is already claimed should return LwSciError_IlwalidState
    ret = event.prepareEvent(0,10);
    ret = event.prepareEvent(0,10);
    EXPECT_EQ(LwSciError_IlwalidState, ret);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_pendingEvent_false1}
 * @testcase{21731093}
 * @verify{19406934}
 * @testpurpose{Test negative scenario of
 * TrackArrayEvent::pendingEvent()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set() with 2U as parameter.
 *     4.Call to TrackArrayEvent::init(), with obj and LwSciStreamEventType parameters.
 *
 * Call to TrackArrayEvent::pendingEvent() with LwSciStreamEvent event an argument
 * should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayEvent::pendingEvent()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_pendingEvent_false1)
{

    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{2U};

    /*test case*/

    /*initialize */
    uint32_t ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);
    LwSciStreamEvent eventType;

    // check LwSciStreamEvent is retrieved should fail as no event is available
    isTrue = event.pendingEvent(eventType);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_checkUsed_false1}
 * @testcase{21731097}
 * @verify{19406937}
 * @testpurpose{Test negative scenario of TrackArrayEvent::checkUsed()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(),with 2U as parameter.
 *     4.Call to TrackArrayEvent::init() with obj and LwSciStreamEventType parameters.
 *     5.Call  TrackArrayEvent::prepareEvent() with index=1 and value=20.
 *
 * Call to TrackArrayEvent::checkUsed() should return False}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayEvent::checkUsed()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_checkUsed_false1)
{
    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{4U};

    /*test case*/

    //initialize
    uint32_t ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);

    /*prepare evnt*/
    ret = event.prepareEvent(1, 20);
    EXPECT_EQ(LwSciError_Success, ret);

    isTrue = event.checkUsed();
    EXPECT_FALSE(isTrue);
}


/**
 * @testname{trackarray_unit_test.TrackArrayEvent_checkUsed_false2}
 * @testcase{21731099}
 * @verify{19406937}
 * @testpurpose{Test negative scenario of TrackArrayEvent::checkUsed()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 1U as parameter.
 *
 * Call to TrackArrayEvent::checkUsed() should return False }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayEvent::checkUsed()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_checkUsed_false2)
{
    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{4U};

    /*test case*/

    //initialize
    uint32_t ret;
    ret = obj.set(1);
    EXPECT_EQ(LwSciError_Success, ret);

    isTrue = event.checkUsed();
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_checkDone_false1}
 * @testcase{21731102}
 * @verify{19406940}
 * @testpurpose{Test negative scenario of TrackArrayEvent::checkDone()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U as parameter.
 *     4.Call to TrackArrayEvent::init() with obj and LwSciStreamEventType parameters.
 *     5.Call to TrackArrayEvent::prepareEvent() with index=0, value=20 as two arguments.
 *
 * Call to TrackArrayEvent::checkDone() with argument bits=1, should return False }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayEvent::checkDone()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_checkDone_false1)
{
    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{2U};

    /*test case*/

    //initialize
    uint32_t ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    event.init(obj, LwSciStreamEventType_PacketReady);
    //prepare one event
    ret = event.prepareEvent(0,20);
    EXPECT_EQ(LwSciError_Success, ret);
    //check for un prepared event should return
    isTrue = event.checkDone(1);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{trackarray_unit_test.TrackArrayEvent_checkDone_false2}
 * @testcase{21731107}
 * @verify{19406940}
 * @testpurpose{Test negative scenario of TrackArrayEvent::checkDone()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayEvent(LwSciError) object event.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with 2U parameter.
 *
 * Call to TrackArrayEvent::checkDone() with 1U as parameter should return False }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.
 *   }
 * @verifyFunction{TrackArrayEvent::checkDone()}
 */
TEST_F(trackarray_unit_test, TrackArrayEvent_checkDone_false2)
{
    /*initial set up*/

    bool isTrue;
    LwSciStream::TrackArrayEvent<LwSciError> event;
    LwSciStream::TrackCount obj{2U};

    /*test case*/

    //initialize
    uint32_t ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);

    //check for should return false as init is not performed
    isTrue = event.checkDone(1);
    EXPECT_FALSE(isTrue);
}


//Coverage gap testcases
/**
 * @testname{trackarray_unit_test.TrackArrayAction_performAction_false4}
 * @testcase{21821492}
 * @verify{19406898}
 * @verify{19780878}
 * @verify{19406901}
 * @verify{19780890}
 * @verify{19780902}
 * @testpurpose{Test negative scenario of TrackArrayAction::performAction()}
 * @testbehavior{
 * Setup:
 *     1.Creates TrackArrayAction object action.
 *     2.Create TrackCount object obj.
 *     3.Call to Trackcount::set(), with argument 2U.
 *     4.Call to TrackArrayAction::init(), with argument obj created in step 2.
 *
 * Call to TrackArrayAction::performAction(), with valid index of 0 and address of a function
 * throws bad_function_call execption, should cause LwSciError_BadParameter to be returned.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is directly usable on target.}
 * @testinput{No external input to the test application is required.}
 * @verifyFunction{TrackArrayAction::performAction()}
 */
TEST_F(trackarray_unit_test, TrackArrayAction_performAction_false4)
{
    /*initial set up*/

    LwSciStream::TrackArrayAction action;
    LwSciStream::TrackCount obj{2U};
    unsigned ret;
    ret = obj.set(2);
    EXPECT_EQ(LwSciError_Success, ret);
    action.init(obj);

    /*test case */

    ret = action.performAction(0, &dummy4);
    EXPECT_EQ(LwSciError_BadParameter, ret);

    EXPECT_EQ(true, flag4);
    flag4 = false;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
