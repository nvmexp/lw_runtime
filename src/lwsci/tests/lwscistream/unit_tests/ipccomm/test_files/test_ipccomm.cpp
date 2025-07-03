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
#include "ipccomm.h"
#include "common_includes.h"
#include "glob_test_vars.h"
#include "ipccomm_common.h"

//==============================================
// Define ipccomm_unit_test suite
//==============================================
class ipccomm_unit_test: public ::testing::Test {
public:
    ipccomm_unit_test( ) {
        // initialization code here
    }

    void SetUp( ) {
        // code here will execute just before the test ensues
    }

    void TearDown( ) {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~ipccomm_unit_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

class ipcsendbuffer_unit_test: public ::testing::Test {
public:
    ipcsendbuffer_unit_test( ) {
        // initialization code here
    }

    void SetUp( ) {
        // code here will execute just before the test ensues
    }

    void TearDown( ) {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~ipcsendbuffer_unit_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};

class ipcrecvbuffer_unit_test: public ::testing::Test {
public:
    ipcrecvbuffer_unit_test( ) {
        // initialization code here
    }

    void SetUp( ) {
        // code here will execute just before the test ensues
    }

    void TearDown( ) {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~ipcrecvbuffer_unit_test( )  {
        // cleanup any pending stuff, but no exceptions and no gtest
        // ASSERT* allowed.
    }

    // put in any custom data members that you need
};


namespace LwSciStream {

/**
 * @testname{ipccomm_unit_test.isInitSuccess_true}
 * @testcase{22059761}
 * @verify{19652046}
 * @testpurpose{Test positive scenario of IpcComm::isInitSuccess().}
 * @testbehavior{
 * Setup:
 *   Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::isInitSuccess() API from ipccomm object,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::isInitSuccess()}
 */
TEST_F(ipccomm_unit_test, isInitSuccess_true)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    LwSciIpcInit();

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(true, ipccomm.isInitSuccess());
}

/**
 * @testname{ipccomm_unit_test.isInitSuccess_false1}
 * @testcase{22059764}
 * @verify{19652046}
 * @testpurpose{Test negative scenario of IpcComm::isInitSuccess() when IpcComm
 * initialization failed.}
 * @testbehavior{
 * Setup:
 *  1. Stub the implementation of LwSciIpcGetEndpointInfo() to return error code
 *     other than LwSciError_Success.
 *  2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *     the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::isInitSuccess() API from ipccomm object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::isInitSuccess()}
 */
TEST_F(ipccomm_unit_test, isInitSuccess_false1)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    test_comm.LwSciIpcGetEndpointInfo_fail = true;
    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, ipccomm.isInitSuccess());
}


/**
 * @testname{ipccomm_unit_test.isInitSuccess_false2}
 * @testcase{22059769}
 * @verify{19652046}
 * @testpurpose{Test negative scenario of IpcComm::isInitSuccess() when IpcComm
 * initialization failed.}
 * @testbehavior{
 * Setup:
 *  1. Stub the implementation of ChannelCreate_r() to return -1.
 *  2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *     the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::isInitSuccess() API from ipccomm object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::isInitSuccess()}
 */
TEST_F(ipccomm_unit_test, isInitSuccess_false2)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    test_comm.ChannelCreate_r_fail = true;
    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, ipccomm.isInitSuccess());
}

/**
 * @testname{ipccomm_unit_test.isInitSuccess_false3}
 * @testcase{22059771}
 * @verify{19652046}
 * @testpurpose{Test negative scenario of IpcComm::isInitSuccess() when IpcComm
 * initialization failed.}
 * @testbehavior{
 * Setup:
 *  1. Stub the implementation of ConnectAttach_r()to return -1.
 *  2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *     the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::isInitSuccess() API from ipccomm object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::isInitSuccess()}
 */
TEST_F(ipccomm_unit_test, isInitSuccess_false3)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    test_comm.ConnectAttach_r_fail = true;
    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, ipccomm.isInitSuccess());
}

/**
 * @testname{ipccomm_unit_test.isInitSuccess_false4}
 * @testcase{22059775}
 * @verify{19652046}
 * @testpurpose{Test negative scenario of IpcComm::isInitSuccess() when IpcComm
 * initialization failed.}
 * @testbehavior{
 * Setup:
 *  1. Stub the implementation of LwSciIpcSetQnxPulseParam() to return LwSciError_BadParameter.
 *  2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *     the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::isInitSuccess() API from ipccomm object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::isInitSuccess()}
 */
TEST_F(ipccomm_unit_test, isInitSuccess_false4)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    test_comm.LwSciIpcSetQnxPulseParam_fail = true;
    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, ipccomm.isInitSuccess());
}

/**
 * @testname{ipccomm_unit_test.isInitSuccess_false5}
 * @testcase{22059776}
 * @verify{19652046}
 * @testpurpose{Test negative scenario of IpcComm::isInitSuccess() when IpcComm
 * initialization failed.}
 * @testbehavior{
 * Setup:
 *  1. Stub the implementation of LwSciIpcGetEndpointInfo() to
 *     return channel frame size as 0.
 *  2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *     the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::isInitSuccess() API from ipccomm object,
 * should return false.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::isInitSuccess()}
 */
TEST_F(ipccomm_unit_test, isInitSuccess_false5)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    test_comm.LwSciIpcGetEndpointInfo_Ilwalid_size = true;
    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, ipccomm.isInitSuccess());
}

/**
 * @testname{ipccomm_unit_test.getFrameSize_Success}
 * @testcase{22059780}
 * @verify{19652133}
 * @testpurpose{Test positive scenario of IpcComm::getFrameSize().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEndpointInfo() to return
 *    IPC channel frame size set to 512 bytes.
 *
 *   The call of IpcComm::getFrameSize() API from ipccomm object,
 * should return the frame size value as 512 bytes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::getFrameSize()}
 */
TEST_F(ipccomm_unit_test, getFrameSize_Success)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(512U, ipccomm.getFrameSize());
}

/**
 * @testname{ipccomm_unit_test.waitForEvent_Success1}
 * @testcase{22059783}
 * @verify{19652136}
 * @testpurpose{Test positive scenario of IpcComm::waitForEvent() when event
 * LW_SCI_IPC_EVENT_WRITE is available and there is a pending write request.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to create
 *      LW_SCI_IPC_EVENT_WRITE event.
 *   3. Stub the implementation of MsgReceivePulse_r() to return LwSciError_Success.
 *
 *   The call of IpcComm::waitForEvent() API from ipccomm object,
 * should return LwSciError_Success along with writeReady flag set and should
 * ilwoke LwSciIpcGetEvent() twice and MsgReceivePulse_r() only once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.
 *   - MsgReceivePulse_r() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::waitForEvent()}
 */
TEST_F(ipccomm_unit_test, waitForEvent_Success1)
{
    IpcQueryFlags result;

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcGetEvent_Write_Pending = true;
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    test_function_call.MsgReceivePulse_r_counter = 0U;
    ///////////////////////
    //     Test code     //
    ///////////////////////
    result = ipccomm.waitForEvent();
    EXPECT_EQ(LwSciError_Success, result.err);
    EXPECT_EQ(true, result.writeReady);
    test_comm.LwSciIpcGetEvent_Write_Pending = false;
    test_comm.counter = 0;

    EXPECT_EQ(2U, test_function_call.LwSciIpcGetEvent_counter);
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    EXPECT_EQ(1U, test_function_call.MsgReceivePulse_r_counter);
    test_function_call.MsgReceivePulse_r_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.waitForEvent_Success2}
 * @testcase{22059785}
 * @verify{19652136}
 * @testpurpose{Test positive scenario of IpcComm::waitForEvent() when event
 * LW_SCI_IPC_EVENT_READ is available.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to create
 *      LW_SCI_IPC_EVENT_READ event.
 *   3. Stub the implementation of MsgReceivePulse_r() to return LwSciError_Success.
 *
 *   The call of IpcComm::waitForEvent() API from ipccomm object,
 * should return LwSciError_Success along with readReady flag set and should
 * ilwoke LwSciIpcGetEvent() twice and MsgReceivePulse_r() only once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.
 *   - MsgReceivePulse_r() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::waitForEvent()}
 */
TEST_F(ipccomm_unit_test, waitForEvent_Success2)
{
    IpcQueryFlags result;

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcGetEvent_Read_Pending = true;
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    test_function_call.MsgReceivePulse_r_counter = 0U;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    result = ipccomm.waitForEvent();
    EXPECT_EQ(LwSciError_Success, result.err);
    EXPECT_EQ(true, result.readReady);
    test_comm.LwSciIpcGetEvent_Read_Pending = false;
    test_comm.counter = 0;

    EXPECT_EQ(2U, test_function_call.LwSciIpcGetEvent_counter);
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    EXPECT_EQ(1U, test_function_call.MsgReceivePulse_r_counter);
    test_function_call.MsgReceivePulse_r_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.waitForEvent_Success3}
 * @testcase{22059789}
 * @verify{19652136}
 * @testpurpose{Test positive scenario of IpcComm::waitForEvent() when
 * pulse indicating a disconnect is requested.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to
 *      return LwSciError_Success without any LwSciIpc read/write events set.
 *   3. Stub the implementation of MsgReceivePulse_r() to return LwSciError_Success.
 *
 *   The call of IpcComm::waitForEvent() API from ipccomm object,
 * should return LwSciError_Success with none of the read/write flags set and
 * should ilwoke LwSciIpcGetEvent() and MsgReceivePulse_r() only once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.
 *   - MsgReceivePulse_r() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::waitForEvent()}
 */
TEST_F(ipccomm_unit_test, waitForEvent_Success3)
{
    IpcQueryFlags result;

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcGetEvent_Disconnect_Request = true;
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    test_function_call.MsgReceivePulse_r_counter = 0U;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    result = ipccomm.waitForEvent();
    EXPECT_EQ(LwSciError_Success, result.err);
    EXPECT_EQ(false, result.writeReady);
    EXPECT_EQ(false, result.readReady);
    test_comm.LwSciIpcGetEvent_Disconnect_Request = false;

    EXPECT_EQ(1U, test_function_call.LwSciIpcGetEvent_counter);
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    EXPECT_EQ(1U, test_function_call.MsgReceivePulse_r_counter);
    test_function_call.MsgReceivePulse_r_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.waitForEvent_StreamInternalError}
 * @testcase{22059791}
 * @verify{19652136}
 * @testpurpose{Test negative scenario of IpcComm::waitForEvent() when
 * LwSciIpcGetEvent() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *   2. Inject fault in LwSciIpcGetEvent() to return LwSciError_StreamInternalError.
 *   3. Stub the implementation of MsgReceivePulse_r() to return LwSciError_Success.
 *
 *   The call of IpcComm::waitForEvent() API from ipccomm object,
 * should return LwSciError_StreamInternalError with none of the read/write flags set.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_StreamInternalError.
 *   - MsgReceivePulse_r() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::waitForEvent()}
 */
TEST_F(ipccomm_unit_test, waitForEvent_StreamInternalError)
{
    IpcQueryFlags result;

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcGetEvent_fail = true;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    result = ipccomm.waitForEvent();
    EXPECT_EQ(LwSciError_StreamInternalError, result.err);
    EXPECT_EQ(false, result.writeReady);
    EXPECT_EQ(false, result.readReady);
}

/**
 * @testname{ipccomm_unit_test.waitForEvent_Timeout}
 * @testcase{22059795}
 * @verify{19652136}
 * @testpurpose{Test negative scenario of IpcComm::waitForEvent() when
 * MsgReceivePulse_r() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to return LwSciError_Success.
 *   3. Inject fault in MsgReceivePulse_r() to return LwSciError_Timeout.
 *
 *   The call of IpcComm::waitForEvent() API from ipccomm object,
 * should return LwSciError_Timeout with none of the read/write flags set.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.
 *   - MsgReceivePulse_r() returns LwSciError_Timeout.}
 * @verifyFunction{IpcComm::waitForEvent()}
 */
TEST_F(ipccomm_unit_test, waitForEvent_Timeout)
{
    IpcQueryFlags result;

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcGetEvent_Write_Pending = true;
    test_comm.MsgReceivePulse_r_fail = true;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    result = ipccomm.waitForEvent();
    EXPECT_EQ(LwSciError_Timeout, result.err);
    test_comm.LwSciIpcGetEvent_Write_Pending = false;
    test_comm.counter = 0;
}

/**
 * @testname{ipccomm_unit_test.waitForConnection_Success}
 * @testcase{22059798}
 * @verify{19652145}
 * @testpurpose{Test positive scenario of IpcComm::waitForConnection() when
 * LW_SCI_IPC_EVENT_CONN_EST_ALL event is available.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to return LwSciError_Success
 *      along with LW_SCI_IPC_EVENT_CONN_EST_ALL.
 *   3. Stub the implementation of MsgReceivePulse_r() to return LwSciError_Success.
 *
 *   The call of IpcComm::waitForConnection() API from ipccomm object,
 * should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.
 *   - MsgReceivePulse_r() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::waitForConnection()}
 */
TEST_F(ipccomm_unit_test, waitForConnection_Success)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.waitForConnection_flag = true;
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    test_function_call.MsgReceivePulse_r_counter = 0U;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Success, ipccomm.waitForConnection());
    test_comm.waitForConnection_flag = false;
    test_comm.counter = 0;

    EXPECT_EQ(2U, test_function_call.LwSciIpcGetEvent_counter);
    test_function_call.LwSciIpcGetEvent_counter = 0U;
    EXPECT_EQ(1U, test_function_call.MsgReceivePulse_r_counter);
    test_function_call.MsgReceivePulse_r_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.waitForConnection_StreamInternalError}
 * @testcase{22059802}
 * @verify{19652145}
 * @testpurpose{Test negative scenario of IpcComm::waitForConnection() when
 * LwSciIpcGetEvent() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to return LwSciError_StreamInternalError.
 *   3. Stub the implementation of MsgReceivePulse_r() to return LwSciError_Success.
 *
 *   The call of IpcComm::waitForConnection() API from ipccomm object,
 * should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_StreamInternalError.
 *   - MsgReceivePulse_r() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::waitForConnection()}
 */
TEST_F(ipccomm_unit_test, waitForConnection_StreamInternalError)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcGetEvent_fail = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_StreamInternalError, ipccomm.waitForConnection());
}

/**
 * @testname{ipccomm_unit_test.waitForConnection_Timeout}
 * @testcase{22059805}
 * @verify{19652145}
 * @testpurpose{Test negative scenario of IpcComm::waitForConnection() when
 * MsgReceivePulse_r() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to return LwSciError_Success.
 *   3. Stub the implementation of MsgReceivePulse_r() to return LwSciError_Timeout.
 *
 *   The call of IpcComm::waitForConnection() API from ipccomm object,
 * should return LwSciError_Timeout.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.
 *   - MsgReceivePulse_r() returns LwSciError_Timeout.}
 * @verifyFunction{IpcComm::waitForConnection()}
 */
TEST_F(ipccomm_unit_test, waitForConnection_Timeout)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.counter = 0;
    test_comm.waitForConnection_flag = true;
    test_comm.MsgReceivePulse_flag= true;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Timeout, ipccomm.waitForConnection());
    test_comm.waitForConnection_flag = false;
    test_comm.counter = 0;
}

/**
 * @testname{ipccomm_unit_test.signalDisconnect_Success}
 * @testcase{22059808}
 * @verify{19652160}
 * @testpurpose{Test positive scenario of IpcComm::signalDisconnect().}
 * @testbehavior{
 * Setup:
 *   Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::signalDisconnect() API from ipccomm object,
 * should signal a disconnect request by calling MsgSendPulse_r() only once and
 * return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::signalDisconnect()}
 */
TEST_F(ipccomm_unit_test, signalDisconnect_Success)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };
    test_function_call.MsgSendPulse_r_counter = 0U;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Success, ipccomm.signalDisconnect());

    EXPECT_EQ(1U, test_function_call.MsgSendPulse_r_counter);
    test_function_call.MsgSendPulse_r_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.signalDisconnect_Timeout}
 * @testcase{22059811}
 * @verify{19652160}
 * @testpurpose{Test negative scenario of IpcComm::signalDisconnect()
 * when MsgSendPulse_r() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of MsgSendPulse_r() to return LwSciError_Timeout.
 *
 *   The call of IpcComm::signalDisconnect() API from ipccomm object,
 * should return LwSciError_Timeout.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - MsgSendPulse_r() returns LwSciError_Timeout.}
 * @verifyFunction{IpcComm::signalDisconnect()}
 */
TEST_F(ipccomm_unit_test, signalDisconnect_Timeout)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.MsgSendPulse_r_fail = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Timeout, ipccomm.signalDisconnect());
}

/**
 * @testname{ipccomm_unit_test.signalWrite_Success1}
 * @testcase{22059814}
 * @verify{19652163}
 * @testpurpose{Test positive scenario of IpcComm::signalWrite() when IPC
 * connection has been established.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to return
 *      LW_SCI_IPC_EVENT_CONN_EST_ALL event.
 *   3. Call IpcComm::waitForConnection() to establish the connection.
 *
 *   The call of IpcComm::signalWrite() API from ipccomm object,
 * should signal a write request by calling MsgSendPulse_r() once
 * and return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::signalWrite()}
 */
TEST_F(ipccomm_unit_test, signalWrite_Success1)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.waitForConnection_flag = true;
    ipccomm.waitForConnection();

    test_function_call.MsgSendPulse_r_counter = 0U;

    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Success, ipccomm.signalWrite());
    test_comm.waitForConnection_flag = false;
    test_comm.counter = 0;

    EXPECT_EQ(1U, test_function_call.MsgSendPulse_r_counter);
    test_function_call.MsgSendPulse_r_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.signalWrite_Success2}
 * @testcase{22059818}
 * @verify{19652163}
 * @testpurpose{Test positive scenario of IpcComm::signalWrite() when IPC
 * connection has not been established yet.}
 * @testbehavior{
 * Setup:
 *   Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *   the IpcComm::IpcComm() constructor.
 *
 *   The call of IpcComm::signalWrite() API from ipccomm object,
 * should return LwSciError_Success without ilwoking MsgSendPulse_r().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcGetEvent() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::signalWrite()}
 */
TEST_F(ipccomm_unit_test, signalWrite_Success2)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_function_call.MsgSendPulse_r_counter = 0U;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Success, ipccomm.signalWrite());
    EXPECT_EQ(0U, test_function_call.MsgSendPulse_r_counter);
}

/**
 * @testname{ipccomm_unit_test.signalWrite_Timeout}
 * @testcase{22059819}
 * @verify{19652163}
 * @testpurpose{Test negative scenario of IpcComm::signalWrite() when
 * MsgSendPulse_r() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of LwSciIpcGetEvent() to return
 *      LW_SCI_IPC_EVENT_CONN_EST_ALL event.
 *   3. Call IpcComm::waitForConnection() to establish the connection.
 *   4. Stub the implementation of MsgSendPulse_r() to return LwSciError_Timeout.
 *
 *   The call of IpcComm::signalWrite() API from ipccomm object,
 * should return LwSciError_Timeout.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - MsgSendPulse_r() returns LwSciError_Timeout.}
 * @verifyFunction{IpcComm::signalWrite()}
 */
TEST_F(ipccomm_unit_test, signalWrite_Timeout)
{

    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.waitForConnection_flag = true;
    ipccomm.waitForConnection();
    test_comm.waitForConnection_flag = false;
    test_comm.MsgSendPulse_r_fail = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Timeout, ipccomm.signalWrite());
    test_comm.counter = 0;
}

/**
 * @testname{ipccomm_unit_test.sendMessage_Success}
 * @testcase{22059823}
 * @verify{19652169}
 * @testpurpose{Test positive scenario of IpcComm::sendMessage().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *     the IpcComm::IpcComm() constructor.
 *   2. Create an IpcSendBuffer instance with maximum size of the buffer set to
 *      1024 bytes.
 *   3. Stub the implementation of LwSciIpcWrite() to return LwSciError_Success.
 *
 *   The call of IpcComm::sendMessage() API from ipccomm object,
 * with a valid IpcSendBuffer object should ilwoke LwSciIpcWrite() only once
 * and should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcWrite() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::sendMessage()}
 */
TEST_F(ipccomm_unit_test, sendMessage_Success)
{
    LwSciStream::IpcSendBuffer buffer { 1024U };
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;
    uint32_t count = 0x1234U;

    LwSciIpcInit();

    test_function_call.LwSciIpcWrite_counter = 0U;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };
    buffer.packBegin();
    EXPECT_EQ(true, buffer.packVal(count));
    buffer.packEnd();

    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_Success, ipccomm.sendMessage(buffer));

    EXPECT_EQ(1U, test_function_call.LwSciIpcWrite_counter);
    test_function_call.LwSciIpcWrite_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.sendMessage_StreamInternalError1}
 * @testcase{22059826}
 * @verify{19652169}
 * @testpurpose{Test negative scenario of IpcComm::sendMessage() when
 * Message to send is too big to fit into one IPC frame.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *     the IpcComm::IpcComm() constructor.
 *   2. Create an IpcSendBuffer instance with maximum size of the buffer set to
 *      1024 bytes.
 *   3. Prepare the message and make sure it exceeds the channel frame size.
 *
 *   The call of IpcComm::sendMessage() API from ipccomm object,
 * with a valid IpcSendBuffer object, should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcWrite() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::sendMessage()}
 */
TEST_F(ipccomm_unit_test, sendMessage_StreamInternalError1)
{
    LwSciStream::IpcSendBuffer buffer { 1024U };
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;
    uint32_t count = 0x1234U;

    LwSciIpcInit();

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };
    buffer.packBegin();
    for (uint32_t i=0; i<(1012U/sizeof(uint32_t)); i++)
    {
        EXPECT_EQ(true, buffer.packVal(count));
    }
    buffer.packEnd();

    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_StreamInternalError, ipccomm.sendMessage(buffer));
}

/**
 * @testname{ipccomm_unit_test.sendMessage_StreamInternalError2}
 * @testcase{22059830}
 * @verify{19652169}
 * @testpurpose{Test negative scenario of IpcComm::sendMessage() when
 * LwSciIpcWrite() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Create an IpcSendBuffer instance with maximum size of the buffer set to
 *      1024 bytes.
 *   3. Stub the implementation of LwSciIpcWrite() to return LwSciError_StreamInternalError.
 *
 *   The call of IpcComm::sendMessage() API from ipccomm object,
 * with a valid IpcSendBuffer object, should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcWrite() returns LwSciError_StreamInternalError.}
 * @verifyFunction{IpcComm::sendMessage()}
 */
TEST_F(ipccomm_unit_test, sendMessage_StreamInternalError2)
{
    LwSciStream::IpcSendBuffer buffer { 1024U };
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;
    uint32_t count = 0x1234U;

    LwSciIpcInit();

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };
    buffer.packBegin();
    EXPECT_EQ(true, buffer.packVal(count));
    buffer.packEnd();

    test_comm.LwSciIpcWrite_fail = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////

    EXPECT_EQ(LwSciError_StreamInternalError, ipccomm.sendMessage(buffer));
}

/**
 * @testname{ipccomm_unit_test.readFrame_Success}
 * @testcase{22059833}
 * @verify{19652175}
 * @testpurpose{Test positive scenario of IpcComm::readFrame().}
 * @testbehavior{
 * Setup:
 *  1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *    the IpcComm::IpcComm() constructor.
 *  2. Create an IpcRecvBuffer instance with maximum size set to 512 bytes.
 *  3. Call IpcRecvBuffer::setRecvInfo() to set the size of the received data to 412 bytes.
 *  4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *
 *   The call of IpcComm::readFrame() API from ipccomm object,
 * with a valid IpcRecvBuffer object, should call LwSciIpcRead() only once
 * and return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcRead() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::readFrame()}
 */
TEST_F(ipccomm_unit_test, readFrame_Success)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;
    LwSciStream::IpcRecvBuffer buffer { 512U };

    buffer.setRecvInfo(412U);

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_function_call.LwSciIpcRead_counter = 0U;
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(buffer));

    EXPECT_EQ(1U, test_function_call.LwSciIpcRead_counter);
    test_function_call.LwSciIpcRead_counter = 0U;
}

/**
 * @testname{ipccomm_unit_test.readFrame_StreamInternalError}
 * @testcase{22059836}
 * @verify{19652175}
 * @testpurpose{Test negative scenario of IpcComm::readFrame() when total
 * read bytes not the same as the requested size.}
 * @testbehavior{
 * Setup:
 *  1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *    the IpcComm::IpcComm() constructor.
 *  2. Create an IpcRecvBuffer instance with maximum size set to 512 bytes.
 *  3. Call IpcRecvBuffer::setRecvInfo() to set the size of the received data to 411 bytes.
 *  4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *
 *   The call of IpcComm::readFrame() API from ipccomm object,
 * with a valid IpcRecvBuffer object, should return LwSciError_StreamInternalError.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcRead() returns LwSciError_Success.}
 * @verifyFunction{IpcComm::readFrame()}
 */
TEST_F(ipccomm_unit_test, readFrame_StreamInternalError)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;
    LwSciStream::IpcRecvBuffer buffer { 512U };

    buffer.setRecvInfo(411U);

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcRead_flag = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_StreamInternalError, ipccomm.readFrame(buffer));
}

/**
 * @testname{ipccomm_unit_test.readFrame_BadParameter}
 * @testcase{22059839}
 * @verify{19652175}
 * @testpurpose{Test negative scenario of IpcComm::readFrame() when total
 * read bytes not the same as the requested size.}
 * @testbehavior{
 * Setup:
 *  1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *    the IpcComm::IpcComm() constructor.
 *  2. Create an IpcRecvBuffer instance with maximum size set to 512 bytes.
 *  3. Call IpcRecvBuffer::setRecvInfo() to set the size of the received data to 412 bytes.
 *  4. Stub the implementation of LwSciIpcRead() to return LwSciError_BadParameter.
 *
 *   The call of IpcComm::readFrame() API from ipccomm object,
 * with a valid IpcRecvBuffer object, should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *  Following LwSciIpc API calls are replaced with stubs:
 *   - LwSciIpcRead() returns LwSciError_BadParameter.}
 * @verifyFunction{IpcComm::readFrame()}
 */
TEST_F(ipccomm_unit_test, readFrame_BadParameter)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;
    LwSciStream::IpcRecvBuffer buffer { 512U };

    buffer.setRecvInfo(412U);

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcRead_fail = true;
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_BadParameter, ipccomm.readFrame(buffer));
}

/**
 * @testname{ipccomm_unit_test.flushWriteSignals_Success1}
 * @testcase{22059841}
 * @verify{19652184}
 * @testpurpose{Test positive scenario of IpcComm::flushWriteSignals() when
 * there are enqueued write requests.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Call IpcComm::signalWrite() to enqueue write requests.
 *   3. Stub the implementation of MsgSendPulse_r() to return  LwSciError_Success.
 *
 *   The call of IpcComm::flushWriteSignals() API from ipccomm object,
 * should signal a write request by calling MsgSendPulse_r() only once
 * and return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::flushWriteSignals()}
 */
TEST_F(ipccomm_unit_test, flushWriteSignals_Success1)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    ipccomm.signalWrite();

    test_function_call.MsgSendPulse_r_counter = 0U;
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, ipccomm.flushWriteSignals());

    EXPECT_EQ(1U, test_function_call.MsgSendPulse_r_counter);
    test_function_call.MsgSendPulse_r_counter = 0U;

}

/**
 * @testname{ipccomm_unit_test.flushWriteSignals_Success2}
 * @testcase{22059844}
 * @verify{19652184}
 * @testpurpose{Test positive scenario of IpcComm::flushWriteSignals() when
 * there are no enqueued write requests.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Stub the implementation of MsgSendPulse_r() to return  LwSciError_Success.
 *
 *   The call of IpcComm::flushWriteSignals() API from ipccomm object,
 * should return LwSciError_Success and should not ilwoke MsgSendPulse_r().}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::flushWriteSignals()}
 */
TEST_F(ipccomm_unit_test, flushWriteSignals_Success2)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_function_call.MsgSendPulse_r_counter = 0U;
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Success, ipccomm.flushWriteSignals());

    // Counter value should be 0U which indicates MsgSendPulse_r()
    // is not called
    EXPECT_EQ(0U, test_function_call.MsgSendPulse_r_counter);
}

/**
 * @testname{ipccomm_unit_test.flushWriteSignals_Timeout}
 * @testcase{22059847}
 * @verify{19652184}
 * @testpurpose{Test negative scenario of IpcComm::flushWriteSignals() when
 * MsgSendPulse_r() failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   2. Call IpcComm::signalWrite() to enqueue write requests.
 *   3. Stub the implementation of MsgSendPulse_r() to return  LwSciError_Timeout.
 *
 *   The call of IpcComm::flushWriteSignals() API from ipccomm object,
 * should return LwSciError_Timeout.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::flushWriteSignals()}
 */
TEST_F(ipccomm_unit_test, flushWriteSignals_Timeout)
{
    /*Initial setup*/
    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };
    ipccomm.signalWrite();

    test_comm.MsgSendPulse_r_fail = true;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(LwSciError_Timeout, ipccomm.flushWriteSignals());
}

/**
 * @testname{ipcsendbuffer_unit_test.isInitSuccess_true}
 * @testcase{22059850}
 * @verify{19676238}
 * @testpurpose{Test positive scenario of IpcSendBuffer::isInitSuccess().}
 * @testbehavior{
 * Setup:
 *   Create an IpcSendBuffer instance with a valid size by calling
 *   the IpcSendBuffer::IpcSendBuffer() constructor.
 *
 *   The call of IpcSendBuffer::isInitSuccess() API from sendBuffer object,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::isInitSuccess()}
 */
TEST_F(ipcsendbuffer_unit_test, isInitSuccess_true)
{
    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 1024U };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, sendBuffer.isInitSuccess());
}

/**
 * @testname{ipcsendbuffer_unit_test.packBegin_Success}
 * @testcase{22059853}
 * @verify{19676322}
 * @testpurpose{Test positive scenario of IpcSendBuffer::packBegin().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Call IpcSendBuffer::packVal() thrice to pack 4 bytes of data each and
 *      should return true for each call.
 *   3. Calling IpcSendBuffer::packVal() fourth time should return false as the
 *      buffer exhausted.
 *
 *   The call of IpcSendBuffer::packBegin() API from sendBuffer object,
 * should reset buffer and calling IpcSendBuffer::packVal() should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packBegin()}
 */
TEST_F(ipcsendbuffer_unit_test, packBegin_Success)
{
    uint32_t count = 0x1234U;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 12U };

    EXPECT_EQ(true, sendBuffer.packVal(count));
    EXPECT_EQ(true, sendBuffer.packVal(count));
    EXPECT_EQ(true, sendBuffer.packVal(count));
    EXPECT_EQ(false, sendBuffer.packVal(count));

    ///////////////////////
    //     Test code     //
    ///////////////////////
    sendBuffer.packBegin();

    EXPECT_EQ(true, sendBuffer.packVal(count));
}

/**
 * @testname{ipcsendbuffer_unit_test.packVal_true}
 * @testcase{22059876}
 * @verify{19676328}
 * @testpurpose{Test positive scenario of IpcSendBuffer::packVal().}
 * @testbehavior{
 * Setup:
 *   Create an IpcSendBuffer instance with a valid size(4 bytes) by calling
 *   the IpcSendBuffer::IpcSendBuffer() constructor.
 *
 *   The call of IpcSendBuffer::packVal() API from sendBuffer object to pack
 *   4 bytes of data, should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packVal()}
 */
TEST_F(ipcsendbuffer_unit_test, packVal_true)
{
    uint32_t count = 0x1234U;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { sizeof(count) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, sendBuffer.packVal(count));
}

/**
 * @testname{ipcsendbuffer_unit_test.packVal_false}
 * @testcase{22059877}
 * @verify{19676328}
 * @testpurpose{Test negative scenario of IpcSendBuffer::packVal() when
 * packing failed due to insufficient size of the IpcSendBuffer.}
 * @testbehavior{
 * Setup:
 *   Create an IpcSendBuffer instance with a valid size(2 bytes) by calling
 *   the IpcSendBuffer::IpcSendBuffer() constructor.
 *
 *   The call of IpcSendBuffer::packVal() API from sendBuffer object to pack
 *  4 bytes of data, should return false due to insufficient size of the
 *  IpcSendBuffer(2 bytes).}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packVal()}
 */
TEST_F(ipcsendbuffer_unit_test, packVal_false)
{
    uint32_t count = 0x1234U;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 2U };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, sendBuffer.packVal(count));
}


/**
 * @testname{ipcsendbuffer_unit_test.packValAndBlob_true1}
 * @testcase{22059880}
 * @verify{19676385}
 * @testpurpose{Test positive scenario of IpcSendBuffer::packValAndBlob().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size(1024 bytes) by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Prepare a value(4 bytes) and data blob(12 bytes) to be packed in the buffer.
 *
 *   The call of IpcSendBuffer::packValAndBlob() API from sendBuffer object,
 *   should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packValAndBlob()}
 */
TEST_F(ipcsendbuffer_unit_test, packValAndBlob_true1)
{
    uint32_t count = 0x1234U;
    uint32_t data;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 1024 };

    LwSciStream::IpcSendBuffer::CBlob blob { 4U, static_cast<void const*>(&data) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, sendBuffer.packValAndBlob(count, blob));
}

/**
 * @testname{ipcsendbuffer_unit_test.packValAndBlob_true2}
 * @testcase{22059882}
 * @verify{19676385}
 * @testpurpose{Test positive scenario of IpcSendBuffer::packValAndBlob().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size(1024 bytes) by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Prepare a value(4 bytes) and data blob(8 bytes) to be packed in the buffer.
 *
 *   The call of IpcSendBuffer::packValAndBlob() API from sendBuffer object,
 *   should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packValAndBlob()}
 */
TEST_F(ipcsendbuffer_unit_test, packValAndBlob_true2)
{
    uint32_t count = 0x1234U;
    uint32_t data;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 1024 };

    LwSciStream::IpcSendBuffer::CBlob blob { 0U, static_cast<void const*>(&data) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, sendBuffer.packValAndBlob(count, blob));
}

/**
 * @testname{ipcsendbuffer_unit_test.packValAndBlob_false1}
 * @testcase{22059883}
 * @verify{19676385}
 * @testpurpose{Test negative scenario of IpcSendBuffer::packValAndBlob() when
 * packing the given @a val failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size(2 bytes) by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Prepare a value(4 bytes) and data blob(12 bytes) to be packed in the buffer.
 *
 *   The call of IpcSendBuffer::packValAndBlob() API from sendBuffer object,
 *   should return false due to insufficient size of the IpcSendBuffer(2 bytes).}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packValAndBlob()}
 */
TEST_F(ipcsendbuffer_unit_test, packValAndBlob_false1)
{
    uint32_t count = 0x1234U;
    uint32_t data;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 2U };

    LwSciStream::IpcSendBuffer::CBlob blob { 4U, static_cast<void const*>(&data) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, sendBuffer.packValAndBlob(count, blob));
}

/**
 * @testname{ipcsendbuffer_unit_test.packValAndBlob_false2}
 * @testcase{22059886}
 * @verify{19676385}
 * @testpurpose{Test negative scenario of IpcSendBuffer::packValAndBlob() when
 * packing CBlob::size failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size(4 bytes) by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Prepare a value(4 bytes) and data blob(12 bytes) to be packed in the buffer.
 *
 *   The call of IpcSendBuffer::packValAndBlob() API from sendBuffer object,
 *   should return false due to insufficient size of the IpcSendBuffer(4 bytes).}
 * @testmethod{Requirements Based}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packValAndBlob()}
 */
TEST_F(ipcsendbuffer_unit_test, packValAndBlob_false2)
{
    uint32_t count = 0x1234U;
    uint32_t data;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { sizeof(count) };

    LwSciStream::IpcSendBuffer::CBlob blob { 4U, static_cast<void const*>(&data) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, sendBuffer.packValAndBlob(count, blob));
}

/**
 * @testname{ipcsendbuffer_unit_test.packValAndBlob_false3}
 * @testcase{22059888}
 * @verify{19676385}
 * @testpurpose{Test negative scenario of IpcSendBuffer::packValAndBlob() when
 * packing the CBlob::data failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size(12 bytes) by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Prepare a value(4 bytes) and data blob(12 bytes) to be packed in the buffer.
 *
 *   The call of IpcSendBuffer::packValAndBlob() API from sendBuffer object,
 *   should return false due to insufficient size of the IpcSendBuffer(12 bytes).}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packValAndBlob()}
 */
TEST_F(ipcsendbuffer_unit_test, packValAndBlob_false3)
{
    uint32_t count = 0x1234U;
    uint32_t data = 56U;

    LwSciStream::IpcSendBuffer sendBuffer { 3*sizeof(count) };

    LwSciStream::IpcSendBuffer::CBlob blob { 4U, static_cast<void const*>(&data) };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, sendBuffer.packValAndBlob(count, blob));

}

/**
 * @testname{ipcsendbuffer_unit_test.packEnd_Success}
 * @testcase{22059890}
 * @verify{19676388}
 * @testpurpose{Test positive scenario of IpcSendBuffer::packEnd().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size(8 bytes) by calling
 *   the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Call IpcSendBuffer::packVal() to pack the given value(4 bytes).
 *
 *   The call of IpcSendBuffer::packEnd() API from sendBuffer object,
 * completes packing of data, storing size in the buffer.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::packEnd()}
 */
TEST_F(ipcsendbuffer_unit_test, packEnd_Success)
{
    uint32_t count = 0x1234U;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 8U };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, sendBuffer.packVal(count));

    sendBuffer.packEnd();

    LwSciStream::IpcSendBuffer::CBlob blob1 { sendBuffer.getSendInfo() };
    EXPECT_EQ(sizeof(count), blob1.size);
}

/**
 * @testname{ipcsendbuffer_unit_test.getSendInfo_Success1}
 * @testcase{22059856}
 * @verify{19676418}
 * @testpurpose{Test positive scenario of IpcSendBuffer::getSendInfo().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Begin the packing using IpcSendBuffer::packBegin(), Pack a value using
 *      IpcSendBuffer::packVal() and end the packing using IpcSendBuffer::packEnd().
 *
 *   The call of IpcSendBuffer::getSendInfo() API from sendBuffer object,
 * should return CBlob, containing the pointer and size of the data.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::getSendInfo()}
 */
TEST_F(ipcsendbuffer_unit_test, getSendInfo_Success1)
{
    uint32_t count = 0x1234U;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 12U };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    sendBuffer.packBegin();
    EXPECT_EQ(true, sendBuffer.packVal(count));
    sendBuffer.packEnd();

    LwSciStream::IpcSendBuffer::CBlob blob1 { sendBuffer.getSendInfo() };
    EXPECT_EQ(12U, blob1.size);
    EXPECT_EQ(0x1234, *((uint32_t*)blob1.data+2U));

}

/**
 * @testname{ipcsendbuffer_unit_test.getSendInfo_Success2}
 * @testcase{22059858}
 * @verify{19676418}
 * @testpurpose{Test positive scenario of IpcSendBuffer::getSendInfo().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcSendBuffer instance with a valid size by calling
 *      the IpcSendBuffer::IpcSendBuffer() constructor.
 *   2. Begin the packing using IpcSendBuffer::packBegin(), Pack multiple values using
 *      IpcSendBuffer::packVal() and end the packing using IpcSendBuffer::packEnd().
 *
 *   The call of IpcSendBuffer::getSendInfo() API from sendBuffer object,
 * should return CBlob, containing the pointer and size of the data.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcSendBuffer::getSendInfo()}
 */
TEST_F(ipcsendbuffer_unit_test, getSendInfo_Success2)
{
    uint32_t count = 0x1234U;
    uint32_t count1 = 0x5678U;

    // Create an IpcSendBuffer object
    LwSciStream::IpcSendBuffer sendBuffer { 16U };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    sendBuffer.packBegin();
    EXPECT_EQ(true, sendBuffer.packVal(count));
    EXPECT_EQ(true, sendBuffer.packVal(count1));
    sendBuffer.packEnd();

    LwSciStream::IpcSendBuffer::CBlob blob1 { sendBuffer.getSendInfo() };
    EXPECT_EQ(16U, blob1.size);
    EXPECT_EQ(0x1234, *((uint32_t*)blob1.data+2U));
    EXPECT_EQ(0x5678, *((uint32_t*)blob1.data+3U));

}

/**
 * @testname{ipcrecvbuffer_unit_test.isInitSuccess_true}
 * @testcase{22059860}
 * @verify{19676427}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::isInitSuccess().}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with a valid size by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::isInitSuccess() API from recvBuffer object,
 * should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::isInitSuccess()}
 */
TEST_F(ipcrecvbuffer_unit_test, isInitSuccess_true)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 1024U };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.isInitSuccess());
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackReady_true}
 * @testcase{22059863}
 * @verify{19676460}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackReady() when
 * full message is ready for unpacking.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success
 *      (Received bytes: 12 and received data bytes[0..7]: 12U
 *       and data bytes[8-11]: 0xABCD.
 *       data bytes[0..7]: Indicates the message size(number of bytes must be
 *       received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackReady() API from recvBuffer object,
 * should return true as the required number of bytes for unpacking a message
 * are received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackReady()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackReady_true)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackReady());
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackReady_false1}
 * @testcase{22059866}
 * @verify{19676460}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackReady() when
 * full message is not ready for unpacking.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success
 *      (Received bytes: 12 and received data bytes[0..7]: 120U
 *       and data bytes[8-11]: 0xABCD.
 *       data bytes[0..7]: Indicates the message size(number of bytes must be
 *       received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackReady() API from recvBuffer object,
 * should return false as the required number of bytes for unpacking message
 * are not received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackReady()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackReady_false1)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.LwSciIpcRead_flag = true;
    EXPECT_EQ(LwSciError_StreamInternalError, ipccomm.readFrame(recvBuffer));

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackReady());
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackReady_false2}
 * @testcase{22059893}
 * @verify{19676460}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackReady() when
 * there is no message received yet.}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::unpackReady() API from recvBuffer object,
 * should return false as there is no message received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackReady()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackReady_false2)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackReady());
}


/**
 * @testname{ipcrecvbuffer_unit_test.unpackBegin_true}
 * @testcase{22059896}
 * @verify{19676508}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackBegin() when
 * full message is ready for unpacking.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success
 *      (Received bytes: 12 and received data bytes[0..7]: 12U
 *       and data bytes[8-11]: 0xABCD.
 *       data bytes[0..7]: Indicates the message size(number of bytes must be
 *       received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackBegin() API from recvBuffer object,
 * should reset buffer to prepare for unpacking a new message.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackBegin()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackBegin_true)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackBegin());
}


/**
 * @testname{ipcrecvbuffer_unit_test.unpackBegin_false}
 * @testcase{22059901}
 * @verify{19676508}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackBegin() when
 * full message is not ready for unpacking.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success
 *      (Received bytes: 12 and received data bytes[0..7]: 120U
 *       and data bytes[8-11]: 0xABCD.
 *       data bytes[0..7]: Indicates the message size(number of bytes must be
 *       received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackBegin() API from recvBuffer object,
 * should return false as the full message is not ready for unpacking.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackBegin()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackBegin_false)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };
    test_comm.LwSciIpcRead_flag = true;
    EXPECT_EQ(LwSciError_StreamInternalError, ipccomm.readFrame(recvBuffer));

    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackBegin());
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackVal_true}
 * @testcase{22059902}
 * @verify{19676520}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackVal().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack uint32_t:
 *      (Received message data bytes[0..7]: 12U and data bytes[8-11]: 0xABCD.
 *       data bytes[0..7]: Indicates the message size(number of bytes must be
 *       received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackVal() API from recvBuffer object,
 * should return true and unpacked uint32_t value should match with the data
 * bytes[8-11].}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackVal()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackVal_true)
{
    uint32_t count;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackVal = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackVal(count));
    EXPECT_EQ(0xABCD, count);
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackVal_false}
 * @testcase{22059905}
 * @verify{19676520}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackVal() when
 * unpacking a value failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack uint32_t:
 *      (Received message data bytes[0..7]: 8U and data bytes[8-11]: 0xABCD.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackVal() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackVal()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackVal_false)
{
    uint32_t count;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackVal = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackVal(count));
}


/**
 * @testname{ipcrecvbuffer_unit_test.unpackValAndBlob_true1}
 * @testcase{22059919}
 * @verify{19676535}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackValAndBlob().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack uint32_t and CBlob:
 *      (Received message data bytes[0..7]: 28U and data bytes[8-11]: 0xABCD
 *      data bytes[12-19]: 0x8U and data bytes[20-27]: 0x9876U.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackValAndBlob() API from recvBuffer object,
 * should return true and unpacked uint32_t, CBlob::size, CBlob::data should match
 * with data bytes[8-11], data bytes[12-19] and data bytes[20-27] respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackValAndBlob()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackValAndBlob_true1)
{
    uint32_t count;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackValAndBlob = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackValAndBlob(count, blob1));
    EXPECT_EQ(0xABCD, count);
    EXPECT_EQ(0x8U, blob1.size);
    EXPECT_EQ(0x9876, *(uint32_t*)blob1.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackValAndBlob_true2}
 * @testcase{22059922}
 * @verify{19676535}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackValAndBlob().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack uint32_t and CBlob:
 *      (Received message data bytes[0..7]: 28U and data bytes[8-11]: 0xABCD
 *      data bytes[12-19]: 0U and data bytes[20-27]: 0U
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackValAndBlob() API from recvBuffer object,
 *  should return true and unpacked uint32_t, CBlob::size, CBlob::data should
 *  match with data bytes[8-11], data bytes[12-19] and data bytes[20-27] respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackValAndBlob()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackValAndBlob_true2)
{
    uint32_t count;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackValAndBlob = 5U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackValAndBlob(count, blob1));
    EXPECT_EQ(0xABCD, count);
    EXPECT_EQ(0U, blob1.size);
    EXPECT_EQ(nullptr, blob1.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackValAndBlob_false1}
 * @testcase{22059924}
 * @verify{19676535}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackValAndBlob() when
 * unpacking @a val failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack uint32_t and CBlob:
 *      (Received message data bytes[0..7]: 8U and data bytes[8-11]: 0xABCD
 *      data bytes[12-19]: 0x8U and data bytes[20-27]: 0x9876U
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackValAndBlob() API from recvBuffer object,
 *  should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackValAndBlob()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackValAndBlob_false1)
{
    uint32_t count;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackValAndBlob = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackValAndBlob(count, blob1));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackValAndBlob_false2}
 * @testcase{22059927}
 * @verify{19676535}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackValAndBlob() when
 * unpacking CBlob::size failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack uint32_t and CBlob:
 *     (Received message data bytes[0..7]: 12U and data bytes[8-11]: 0xABCD
 *      data bytes[12-19]: 0x8U and data bytes[20-27]: 0x9876U
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackValAndBlob() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackValAndBlob()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackValAndBlob_false2)
{
    uint32_t count;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackValAndBlob = 3U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackValAndBlob(count, blob1));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackValAndBlob_false3}
 * @testcase{22059930}
 * @verify{19676535}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackValAndBlob() when
 * unpacking CBlob::data failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack uint32_t and CBlob:
 *      (Received message data bytes[0..7]: 20U and data bytes[8-11]: 0xABCD
 *      data bytes[12-19]: 0x8U and data bytes[20-27]: 0x9876U
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackValAndBlob() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackValAndBlob()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackValAndBlob_false3)
{
    uint32_t count;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackValAndBlob = 4U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackValAndBlob(count, blob1));
}


/**
 * @testname{ipcrecvbuffer_unit_test.unpackEnd_Success}
 * @testcase{22059935}
 * @verify{19676544}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackEnd().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with size(512 bytes) by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Call IpcRecvBuffer::setRecvInfo() with size of the received data
 *     set to 100 bytes.
 *   3. Call IpcRecvBuffer::getRecvInfo() to check the remaining buffer size
 *      which should be equal to 412 bytes.
 *
 *   The call of IpcRecvBuffer::unpackEnd() API from recvBuffer object,
 * resets the buffer to receive the next message and call of IpcRecvBuffer::getRecvInfo()
 * should return the size equal to 512 bytes.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackEnd()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackEnd_Success)
{

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    LwSciStream::IpcRecvBuffer::VBlob result;

    recvBuffer.setRecvInfo(100);

    result = recvBuffer.getRecvInfo();

    EXPECT_EQ(412U, result.size);
    EXPECT_NE(nullptr, result.data);

    ///////////////////////
    //     Test code     //
    ///////////////////////

    recvBuffer.unpackEnd();

    result = recvBuffer.getRecvInfo();
    EXPECT_EQ(512U, result.size);
    EXPECT_NE(nullptr, result.data);
}


/**
 * @testname{ipcrecvbuffer_unit_test.unpackFenceExport_true}
 * @testcase{22059938}
 * @verify{19676556}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackFenceExport().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack LwSciSyncFenceIpcExportDescriptor:
 *      (Received message data bytes[0..7]: 64U and data bytes[8-15]: 0xABCD
 *      data bytes[16-23]: 0x1234 and data bytes[24-31]: 0x5678 and
 *      data bytes[32-39]: 0x9AFF and data bytes[40-47]: 0xABFF and
 *      data bytes[48-55]: 0xACFF and data bytes[56-63]:0xADFF
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackFenceExport() API from recvBuffer object,
 * should return true and the unpacked LwSciSyncFenceIpcExportDescriptor::payload[0-6]
 * should match from data bytes[8-15] to data bytes[56-63] respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackFenceExport()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackFenceExport_true)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciSyncFenceIpcExportDescriptor desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackFenceExport = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackFenceExport(desc));
    EXPECT_EQ(0xABCD, desc.payload[0]);
    EXPECT_EQ(0x1234, desc.payload[1]);
    EXPECT_EQ(0x5678, desc.payload[2]);
    EXPECT_EQ(0x9AFF, desc.payload[3]);
    EXPECT_EQ(0xABFF, desc.payload[4]);
    EXPECT_EQ(0xACFF, desc.payload[5]);
    EXPECT_EQ(0xADFF, desc.payload[6]);

}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackFenceExport_false}
 * @testcase{22059941}
 * @verify{19676556}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackFenceExport() when
 * unpacking failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack LwSciSyncFenceIpcExportDescriptor:
 *      (Received message data bytes[0..7]: 20U and data bytes[8-15]: 0xABCD
 *      data bytes[16-23]: 0x1234 and data bytes[24-31]: 0x5678 and
 *      data bytes[32-39]: 0x9AFF and data bytes[40-47]: 0xABFF and
 *      data bytes[48-55]: 0xACFF and data bytes[56-63]:0xADFF
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackFenceExport() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackFenceExport()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackFenceExport_false)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciSyncFenceIpcExportDescriptor desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackFenceExport = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackFenceExport(desc));

}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackBufObjExport_true}
 * @testcase{22059944}
 * @verify{19676562}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackBufObjExport().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack LwSciBufObjIpcExportDescriptor:
 *      (Received message data bytes[0..7]: 264U and next every 8 data bytes set
 *      with value of 0x1234+i(where i:[1..32])
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackBufObjExport() API from recvBuffer object,
 * should return true and the unpacked LwSciBufObjIpcExportDescriptor::data[0..31]
 * should match with the data bytes received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackBufObjExport()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackBufObjExport_true)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    LwSciBufObjIpcExportDescriptor desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackBufObjExport = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackBufObjExport(desc));

    for (uint8_t i=1; i<=32;i++)
    {
        EXPECT_EQ(0x1234+i, desc.data[i-1]);
    }
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackBufObjExport_false}
 * @testcase{22059947}
 * @verify{19676562}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackBufObjExport() when
 * unpacking failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack LwSciBufObjIpcExportDescriptor:
 *      (Received message data bytes[0..7]: 20U
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackBufObjExport() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackBufObjExport()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackBufObjExport_false)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    LwSciBufObjIpcExportDescriptor desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackBufObjExport = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackBufObjExport(desc));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgSyncAttr_true}
 * @testcase{22059950}
 * @verify{19676571}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackMsgSyncAttr().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgSyncAttr and CBlob:
 *      (Received message data bytes[0..7]: 32U and data bytes[8]: 1U and
 *       data bytes[9]: 8U and data bytes[17]: 0x76U and data bytes[18]: 0x98U.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgSyncAttr() API from recvBuffer object,
 *  should return true and unpacked MsgSyncAttr::synchronousOnly, CBlob::size,
 *  CBlob::data should match with data bytes[8], data bytes[9] and
 *  data bytes[17..18] respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgSyncAttr()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgSyncAttr_true)
{
    uint32_t count;
    MsgSyncAttr val;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgSyncAttr = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackMsgSyncAttr(val, blob1));
    EXPECT_EQ(true, val.synchronousOnly);
    EXPECT_EQ(0x8U, blob1.size);
    EXPECT_EQ(0x9876, *(uint32_t*)blob1.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgSyncAttr_false}
 * @testcase{22059952}
 * @verify{19676571}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgSyncAttr() when
 * unpacking failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgSyncAttr and CBlob:
 *      (Received message data bytes[0..7]: 10U and data bytes[8]: 1U and
 *       data bytes[9]: 8U and data bytes[17]: 0x76U and data bytes[18]: 0x98U
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgSyncAttr() API from recvBuffer object,
 *  should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgSyncAttr()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgSyncAttr_false)
{
    uint32_t count;
    MsgSyncAttr val;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgSyncAttr = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgSyncAttr(val, blob1));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgElemAttr_true}
 * @testcase{22059954}
 * @verify{19676574}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackMsgElemAttr().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgElemAttr and CBlob:
 *      (Received message data bytes[0..7]: 40U and data bytes[8-11]: 1U and
 *       data bytes[12-15]: 2U and data bytes[16-19]: 0x1U and data bytes[20-27]:0x8U
 *       data bytes[28-35]: 0x9BFF.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgElemAttr() API from recvBuffer object,
 *  should return true and unpacked MsgElemAttr::index, MsgElemAttr::type,
 *  MsgElemAttr::mode, CBlob::size, CBlob::data should match with the data bytes[8-11],
 *  data bytes[12-15], data bytes[16-19], data bytes[20-27]
 *  and data bytes[28-35] respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgElemAttr()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgElemAttr_true)
{
    uint32_t count;
    MsgElemAttr val;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgElemAttr = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackMsgElemAttr(val, blob1));
    EXPECT_EQ(0x1U, val.index);
    EXPECT_EQ(0x2U, val.type);
    EXPECT_EQ(0x1U, val.mode);
    EXPECT_EQ(0x8U, blob1.size);
    EXPECT_EQ(0x9BFF, *(uint32_t*)blob1.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgElemAttr_false1}
 * @testcase{22059957}
 * @verify{19676574}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgElemAttr() when
 * unpacking MsgElemAttr::index failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgElemAttr and CBlob:
 *      (Received message data bytes[0..7]: 8U and data bytes[8-11]: 1U and
 *       data bytes[12-15]: 2U and data bytes[16-19]: 0x1U and data bytes[20-27]:0x8U
 *       data bytes[28-35]: 0x9BFF.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgElemAttr() API from recvBuffer object,
 *  should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgElemAttr()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgElemAttr_false1)
{
    uint32_t count;
    MsgElemAttr val;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgElemAttr = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgElemAttr(val, blob1));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgElemAttr_false2}
 * @testcase{22059960}
 * @verify{19676574}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgElemAttr() when
 * unpacking MsgElemAttr::type failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgElemAttr and CBlob:
 *      (Received message data bytes[0..7]: 12U and data bytes[8-11]: 1U and
 *       data bytes[12-15]: 2U and data bytes[16-19]: 0x1U and data bytes[20-27]:0x8U
 *       data bytes[28-35]: 0x9BFF.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgElemAttr() API from recvBuffer object,
 *  should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgElemAttr()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgElemAttr_false2)
{
    uint32_t count;
    MsgElemAttr val;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgElemAttr = 3U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgElemAttr(val, blob1));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgElemAttr_false3}
 * @testcase{22059962}
 * @verify{19676574}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgElemAttr() when
 * unpacking MsgElemAttr::mode and CBlob failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgElemAttr and CBlob:
 *      (Received message data bytes[0..7]: 16U and data bytes[8-11]: 1U and
 *       data bytes[12-15]: 2U and data bytes[16-19]: 0x1U and data bytes[20-27]:0x8U
 *       data bytes[28-35]: 0x9BFF.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgElemAttr() API from recvBuffer object,
 *  should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgElemAttr()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgElemAttr_false3)
{
    uint32_t count;
    MsgElemAttr val;

    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 100U };
    LwSciStream::IpcRecvBuffer::CBlob blob1;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgElemAttr = 4U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgElemAttr(val, blob1));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgPacketBuffer_true}
 * @testcase{22059965}
 * @verify{19676580}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackMsgPacketBuffer().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgPacketBuffer:
 *      (Received message data bytes[0..7]: 276U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U and next every 8 bytes set to pattern: 0x1234+3*i
 *       where i:[0..31].
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgPacketBuffer() API from recvBuffer object,
 * should return true and unpacked MsgPacketBuffer::handle, MsgPacketBuffer::index,
 * MsgPacketBuffer::bufObjDesc::data should match with data bytes[8-11],
 * data bytes[12-15] and remaining data bytes in the received buffer respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgPacketBuffer()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgPacketBuffer_true)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgPacketBuffer desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgPacketBuffer = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackMsgPacketBuffer(desc));

    EXPECT_EQ(0x1234, desc.handle);
    EXPECT_EQ(0x1, desc.index);
    for (uint8_t i=0; i<32;i++)
    {
        EXPECT_EQ(0x1234+3*i, desc.bufObjDesc.data[i]);
    }
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgPacketBuffer_false1}
 * @testcase{22059968}
 * @verify{19676580}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgPacketBuffer()
 * when unpacking MsgPacketBuffer::handle failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgPacketBuffer:
 *      (Received message data bytes[0..7]: 8U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgPacketBuffer() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgPacketBuffer()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgPacketBuffer_false1)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgPacketBuffer desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgPacketBuffer = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgPacketBuffer(desc));

}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgPacketBuffer_false2}
 * @testcase{22059971}
 * @verify{19676580}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgPacketBuffer()
 * when unpacking MsgPacketBuffer::index failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgPacketBuffer:
 *      (Received message data bytes[0..7]: 16U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgPacketBuffer() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgPacketBuffer()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgPacketBuffer_false2)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgPacketBuffer desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgPacketBuffer = 3U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgPacketBuffer(desc));

}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgPacketBuffer_false3}
 * @testcase{22059975}
 * @verify{19676580}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgPacketBuffer()
 * when unpacking MsgPacketBuffer::bufObjDesc failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgPacketBuffer:
 *      (Received message data bytes[0..7]: 20U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgPacketBuffer() API from recvBuffer object,
 * should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgPacketBuffer()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgPacketBuffer_false3)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgPacketBuffer desc;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgPacketBuffer = 4U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgPacketBuffer(desc));

}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgStatus_true}
 * @testcase{22059978}
 * @verify{19676586}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::unpackMsgStatus().}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgStatus:
 *      (Received message data bytes[0..7]: 24U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U and data bytes[16-19]: LwSciError_BadParameter.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgStatus() API from recvBuffer object,
 *   should return true and unpacked MsgStatus::handle, MsgStatus::index,
 *   MsgStatus::status should match with the data bytes[8-11], data bytes[12-15]
 *   and data bytes[16-19] respectively.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgStatus()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgStatus_true)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgStatus val;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgStatus = 1U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(true, recvBuffer.unpackMsgStatus(val));

    EXPECT_EQ(0x1234, val.handle);
    EXPECT_EQ(0x1, val.index);
    EXPECT_EQ(LwSciError_BadParameter, val.status);

}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgStatus_false1}
 * @testcase{22059981}
 * @verify{19676586}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgStatus() when
 * unpacking MsgStatus::handle failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgStatus:
 *      (Received message data bytes[0..7]: 8U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U and data bytes[16-19]: LwSciError_BadParameter.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgStatus() API from recvBuffer object,
 *   should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgStatus()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgStatus_false1)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgStatus val;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgStatus = 2U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgStatus(val));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgStatus_false2}
 * @testcase{22059984}
 * @verify{19676586}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgStatus() when
 * unpacking MsgStatus::index failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgStatus:
 *      (Received message data bytes[0..7]: 16U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U and data bytes[16-19]: LwSciError_BadParameter.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgStatus() API from recvBuffer object,
 *  should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgStatus()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgStatus_false2)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgStatus val;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgStatus = 3U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgStatus(val));
}

/**
 * @testname{ipcrecvbuffer_unit_test.unpackMsgStatus_false3}
 * @testcase{22059985}
 * @verify{19676586}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::unpackMsgStatus() when
 * unpacking MsgStatus::status failed.}
 * @testbehavior{
 * Setup:
 *   1. Create an IpcRecvBuffer instance with a valid size by calling
 *      the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *   2. Create an IpcComm instance with a valid LwSciIpcEndpoint by calling
 *      the IpcComm::IpcComm() constructor.
 *   3. Ilwoke IpcComm::readFrame() to receive the message.
 *   4. Stub the implementation of LwSciIpcRead() to return LwSciError_Success.
 *      To unpack MsgStatus:
 *      (Received message data bytes[0..7]: 20U and data bytes[8-11]: 0x1234 and
 *       data bytes[12-15]: 1U and data bytes[16-19]: LwSciError_BadParameter.
 *      data bytes[0..7]: Indicates the message size(number of bytes must be
 *      received for unpacking a message.))
 *
 *   The call of IpcRecvBuffer::unpackMsgStatus() API from recvBuffer object,
 *  should return false as the unpacking failed due to incorrect message
 * size received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::unpackMsgStatus()}
 */
TEST_F(ipcrecvbuffer_unit_test, unpackMsgStatus_false3)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };
    MsgStatus val;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    // Create an IpcComm object
    LwSciStream::IpcComm ipccomm { ipc };

    test_comm.unpackMsgStatus = 4U;
    EXPECT_EQ(LwSciError_Success, ipccomm.readFrame(recvBuffer));

    EXPECT_EQ(true, recvBuffer.unpackBegin());
    ///////////////////////
    //     Test code     //
    ///////////////////////
    EXPECT_EQ(false, recvBuffer.unpackMsgStatus(val));
}

/**
 * @testname{ipcrecvbuffer_unit_test.getRecvInfo_Success1}
 * @testcase{22059989}
 * @verify{19676592}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::getRecvInfo().}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with size (512 bytes) by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::getRecvInfo() API from recvBuffer object,
 * should return VBlob, containing the pointer that the caller can write to, and the size.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::getRecvInfo()}
 */
TEST_F(ipcrecvbuffer_unit_test, getRecvInfo_Success1)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    LwSciStream::IpcRecvBuffer::VBlob result;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    result = recvBuffer.getRecvInfo();

    EXPECT_EQ(512U, result.size);
    EXPECT_NE(nullptr, result.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.getRecvInfo_Success2}
 * @testcase{22059994}
 * @verify{19676592}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::getRecvInfo().}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with size (0 bytes) by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::getRecvInfo() API from recvBuffer object,
 * should return VBlob, containing a nullptr and size set to 0U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcRecvBuffer::getRecvInfo()}
 */
TEST_F(ipcrecvbuffer_unit_test, getRecvInfo_Success2)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 0U };

    LwSciStream::IpcRecvBuffer::VBlob result;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    result = recvBuffer.getRecvInfo();

    EXPECT_EQ(0U, result.size);
    EXPECT_EQ(nullptr, result.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.setRecvInfo_Success1}
 * @testcase{22059996}
 * @verify{19676622}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::setRecvInfo() with
 * received data size set to 0 bytes.}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with a size(512 bytes) by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::setRecvInfo() API from recvBuffer object,
 * with size set to 0 bytes, updates the total size of the data
 * that has been received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::setRecvInfo()}
 */
TEST_F(ipcrecvbuffer_unit_test, setRecvInfo_Success1)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    LwSciStream::IpcRecvBuffer::VBlob result;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    recvBuffer.setRecvInfo(0U);

    result = recvBuffer.getRecvInfo();

    EXPECT_EQ(512U, result.size);
    EXPECT_NE(nullptr, result.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.setRecvInfo_Success2}
 * @testcase{22059999}
 * @verify{19676622}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::setRecvInfo() with
 * received data size set to 512 bytes.}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with a size(512 bytes) by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::setRecvInfo() API from recvBuffer object,
 * with size set to 512 bytes, updates the total size of the data
 * that has been received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::setRecvInfo()}
 */
TEST_F(ipcrecvbuffer_unit_test, setRecvInfo_Success2)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    LwSciStream::IpcRecvBuffer::VBlob result;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    recvBuffer.setRecvInfo(512U);

    result = recvBuffer.getRecvInfo();

    EXPECT_EQ(0U, result.size);
    EXPECT_EQ(nullptr, result.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.setRecvInfo_Success3}
 * @testcase{22060002}
 * @verify{19676622}
 * @testpurpose{Test positive scenario of IpcRecvBuffer::setRecvInfo() with
 * received data size set to 255 bytes.}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with a size(512 bytes) by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::setRecvInfo() API from recvBuffer object,
 * with size set to 512 bytes, updates the total size of the data
 * that has been received.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::setRecvInfo()}
 */
TEST_F(ipcrecvbuffer_unit_test, setRecvInfo_Success3)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    LwSciStream::IpcRecvBuffer::VBlob result;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    recvBuffer.setRecvInfo(255U);

    result = recvBuffer.getRecvInfo();

    EXPECT_EQ(257, result.size);
    EXPECT_NE(nullptr, result.data);
}

/**
 * @testname{ipcrecvbuffer_unit_test.setRecvInfo_failure}
 * @testcase{22060004}
 * @verify{19676622}
 * @testpurpose{Test negative scenario of IpcRecvBuffer::setRecvInfo() when
 * tried to set the received data size more than the maximum allocated buffer.}
 * @testbehavior{
 * Setup:
 *   Create an IpcRecvBuffer instance with a size(512 bytes) by calling
 *   the IpcRecvBuffer::IpcRecvBuffer() constructor.
 *
 *   The call of IpcRecvBuffer::setRecvInfo() API from recvBuffer object,
 * with size set to 513 bytes, returns without performing any action.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{IpcComm::setRecvInfo()}
 */
TEST_F(ipcrecvbuffer_unit_test, setRecvInfo_failure)
{
    // Create an IpcRecvBuffer object
    LwSciStream::IpcRecvBuffer recvBuffer { 512U };

    LwSciStream::IpcRecvBuffer::VBlob result;

    LwSciIpcEndpoint ipc = IPCSRC_ENDPOINT;

    ///////////////////////
    //     Test code     //
    ///////////////////////
    recvBuffer.setRecvInfo(513U);

    result = recvBuffer.getRecvInfo();

    EXPECT_EQ(512U, result.size);
    EXPECT_NE(nullptr, result.data);
}

} // namespace LwSciStream


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
