import pydcgm
import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import dcgm_client_internal
import logger
import test_utils
import dcgm_fields
import dcgm_internal_helpers
import option_parser
import DcgmDiag
import dcgm_errors

import threading
import time
import sys
import os
import signal
import utils
import json

from ctypes import *
from apps.app_runner import AppRunner
from apps.dcgmi_app import DcgmiApp
from dcgm_internal_helpers import InjectionThread


def injection_wrapper(handle, gpuId, fieldId, value, isInt):
    # Sleep 1 second so that the insertion happens after the test run begins while not prolonging things
    time.sleep(1)
    if isInt:
        ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, fieldId, value, 0)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, fieldId, value, 5)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, fieldId, value, 10)
        assert ret == dcgm_structs.DCGM_ST_OK
    else:
        ret = dcgm_internal_helpers.inject_field_value_fp64(handle, gpuId, fieldId, value, 0)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_fp64(handle, gpuId, fieldId, value, 5)
        assert ret == dcgm_structs.DCGM_ST_OK
        ret = dcgm_internal_helpers.inject_field_value_fp64(handle, gpuId, fieldId, value, 10)
        assert ret == dcgm_structs.DCGM_ST_OK

def check_diag_result_fail(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL

def check_diag_result_pass(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_PASS

def diag_result_assert_fail(response, gpuIndex, testIndex, msg, errorCode):
    # Instead of checking that it failed, just make sure it didn't pass because we want to ignore skipped
    # tests or tests that did not run.
    assert response.perGpuResponses[gpuIndex].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_PASS, msg
    if response.version == dcgm_structs.dcgmDiagResponse_version5:
        codeMsg = "Failing test expected error code %d, but found %d" % \
                    (errorCode, response.perGpuResponses[gpuIndex].results[testIndex].error.code)
        assert response.perGpuResponses[gpuIndex].results[testIndex].error.code == errorCode, codeMsg

def diag_result_assert_pass(response, gpuIndex, testIndex, msg):
    # Instead of checking that it passed, just make sure it didn't fail because we want to ignore skipped
    # tests or tests that did not run.
    assert response.perGpuResponses[gpuIndex].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_FAIL, msg
    if response.version == dcgm_structs.dcgmDiagResponse_version5:
        codeMsg = "Passing test somehow has a non-zero error code!"
        assert response.perGpuResponses[gpuIndex].results[testIndex].error.code == 0, codeMsg

def helper_test_dcgm_diag_dbe_insertion(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr='diagnostic', paramsStr='diagnostic.test_duration=8')
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuIds[0],
                        dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 1, 5)
    assert ret == dcgm_structs.DCGM_ST_OK, "Could not insert an error to test forced failure"
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuIds[0],
                        dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 1, 15)
    assert ret == dcgm_structs.DCGM_ST_OK, "Could not insert an error to test forced failure"
    response = dd.Execute(handle)
    errorStr = "Expected results for %d GPUs, but found %d" % (len(gpuIds), response.gpuCount)
    assert response.gpuCount == len(gpuIds), errorStr
    diag_result_assert_fail(response, gpuIds[0], dcgm_structs.DCGM_DIAGNOSTIC_INDEX,
                        "Expected the diagnostic test to fail because we injected a DBE", dcgm_errors.DCGM_FR_FIELD_VIOLATION)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_dbe_insertion_embedded(handle, gpuIds):
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skip test for more debugging")
    helper_test_dcgm_diag_dbe_insertion(handle, gpuIds)

def helper_check_diag_empty_group(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v1()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version1
    runDiagInfo.groupId = groupObj.GetId()
    runDiagInfo.validate = 1

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_GROUP_IS_EMPTY)):
        dcgm_agent.dcgmActiolwalidate_v2(handle, runDiagInfo)

    # Now make sure everything works well with a group
    groupObj.AddGpu(gpuIds[0])
    response = dcgm_agent.dcgmActiolwalidate_v2(handle, runDiagInfo)
    assert response, "Should have received a response now that we have a non-empty group"

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_helper_embedded_check_diag_empty_group(handle, gpuIds):
    helper_check_diag_empty_group(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_helper_standalone_check_diag_empty_group(handle, gpuIds):
    helper_check_diag_empty_group(handle, gpuIds)

def diag_assert_error_found(response, gpuId, testIndex, errorStr):
    if response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP and \
       response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
        if response.version == dcgm_structs.dcgmDiagResponse_version5:
            warningFound = response.perGpuResponses[gpuId].results[testIndex].error.msg
        else:
            warningFound = response.perGpuResponses[gpuId].results[testIndex].warning

        assert warningFound.find(errorStr) != -1, "Expected to find '%s' as a warning, but found '%s'" % (errorStr, warningFound)

def diag_assert_error_not_found(response, gpuId, testIndex, errorStr):
    if response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_SKIP and \
       response.perGpuResponses[gpuId].results[testIndex].result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
        if response.version == dcgm_structs.dcgmDiagResponse_version5:
            warningFound = response.perGpuResponses[gpuId].results[testIndex].error.msg
        else:
            warningFound = response.perGpuResponses[gpuId].results[testIndex].warning
        assert warningFound.find(errorStr) == -1, "Expected not to find '%s' as a warning, but found it: '%s'" % (errorStr, warningFound)

def helper_check_diag_thermal_violation(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr='diagnostic', paramsStr='diagnostic.test_duration=10')

    # kick off a thread to inject the failing value while I run the diag
    diag_thread = threading.Thread(target=injection_wrapper,
                                   args =[handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
                                          9223372036854775792, True])
    diag_thread.start()
    response = dd.Execute(handle)
    diag_thread.join()

    assert response.gpuCount == len(gpuIds), "Expected %d gpus, but found %d reported" % (len(gpuIds), response.gpuCount)
    for gpuIndex in range(response.gpuCount):
        diag_assert_error_not_found(response, gpuIndex, dcgm_structs.DCGM_DIAGNOSTIC_INDEX, "Thermal violations")

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_embedded_check_diag_thermal_violation(handle, gpuIds):
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skip test for more debugging")
    helper_check_diag_thermal_violation(handle, gpuIds)

def helper_check_diag_high_temp_fail(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr='diagnostic', paramsStr='diagnostic.test_duration=10')

    # kick off a thread to inject the failing value while I run the diag
    diag_thread = threading.Thread(target=injection_wrapper,
                                   args =[handle, gpuIds[0], dcgm_fields.DCGM_FI_DEV_GPU_TEMP, 120, True])
    diag_thread.start()
    response = dd.Execute(handle)
    diag_thread.join()

    assert response.gpuCount == len(gpuIds), "Expected %d gpus, but found %d reported" % (len(gpuIds), response.gpuCount)
    diag_result_assert_fail(response, gpuIds[0], dcgm_structs.DCGM_DIAGNOSTIC_INDEX, "Expected a failure due to 120 degree inserted temp.", dcgm_errors.DCGM_FR_TEMP_VIOLATION)


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_embedded_check_diag_high_temperature(handle, gpuIds):
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skip test for more debugging")
    helper_check_diag_high_temp_fail(handle, gpuIds)


def helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuId):
    """
    Verifies that the dcgmActiolwalidate_v2 API supports older versions of the dcgmRunDiag struct
    by using the old structs to run a short validation test.
    """

    def test_dcgm_run_diag(drd, version):
        drd.validate = 1 # run a short test
        drd.gpuList = str(gpuId)
        # This will throw an exception on error
        ret = dcgm_agent.dcgmActiolwalidate_v2(handle, drd, version)

    # Test version 1
    drd = dcgm_structs.c_dcgmRunDiag_v1()
    test_dcgm_run_diag(drd, dcgm_structs.dcgmRunDiag_version1)

    # Test version 2
    drd = dcgm_structs.c_dcgmRunDiag_v2()
    test_dcgm_run_diag(drd, dcgm_structs.dcgmRunDiag_version2)

    # Test version 3
    drd = dcgm_structs.c_dcgmRunDiag_v3()
    test_dcgm_run_diag(drd, dcgm_structs.dcgmRunDiag_version3)

    # Test version 4
    drd = dcgm_structs.c_dcgmRunDiag_v4()
    test_dcgm_run_diag(drd, dcgm_structs.dcgmRunDiag_version4)

    # Test version 5
    drd = dcgm_structs.c_dcgmRunDiag_v5()
    test_dcgm_run_diag(drd, dcgm_structs.dcgmRunDiag_version5)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_run_diag_backwards_compatibility_embedded(handle, gpuIds):
    helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_run_diag_backwards_compatibility_standalone(handle, gpuIds):
    helper_check_dcgm_run_diag_backwards_compatibility(handle, gpuIds[0])


def helper_test_throttle_mask(handle, gpuId):
    """
    Verifies that the throttle ignore mask ignores the masked throttling reasons.
    Since the test runs several instances of the sm stress test with duration set to 2 (and waits for inserted errors),
    the test can take about 40 - 50 seconds to finish.
    """
    #####
    # First check whether the GPU is healthy
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=2",
                           version=dcgm_structs.dcgmRunDiag_version)
    dd.SetThrottleMask(0) # We explicitly want to fail for throttle reasons since this test inserts throttling errors 
                          # for verification
    response = dd.Execute(handle)
    if not check_diag_result_pass(response, gpuId, dcgm_structs.DCGM_SM_PERF_INDEX):
        test_utils.skip_test("Skipping because GPU %s does not pass SM Perf test. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)

    # Valid throttling reasons:
    DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN = 0x8
    DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL = 0x20
    DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL = 0x40
    DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE = 0x80

    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS

    #####
    # Helper method for inserting errors and performing the diag
    def perform_diag_with_throttle_mask_and_verify(inserted_error, throttle_mask, shouldPass, failureMsg):
        offset = 0
        interval = 0.1
        if throttle_mask is not None:
            dd.SetThrottleMask(throttle_mask)

        injection_thread = InjectionThread(handle, gpuId, fieldId, inserted_error, offset)
        injection_thread.start()
        # Verify that the inserted values are visible in DCGM before starting the diag
        assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, inserted_error, checkInterval=interval, maxWait=5), \
            "Expected inserted values to be visible in DCGM"

        # Start the diag
        response = dd.Execute(handle)

        # Stop injecting values
        injection_thread.Stop()
        injection_thread.join()
        # Ensure there were no errors inserting values
        assert injection_thread.retCode == dcgm_structs.DCGM_ST_OK, \
            "There was an error inserting values into dcgm. Return code: %s" % injection_thread.retCode

        # Check for pass or failure as per the shouldPass parameter
        failureMsg += "\nGot result: %s (\ninfo: %s,\n warning: %s)" % \
            (response.perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].result,
             response.perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].info,
             response.perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].error.msg)
        if shouldPass:    
            assert check_diag_result_pass(response, gpuId, dcgm_structs.DCGM_SM_PERF_INDEX), failureMsg
        else:
            assert check_diag_result_fail(response, gpuId, dcgm_structs.DCGM_SM_PERF_INDEX), failureMsg

    #####
    # Insert a throttling error and verify that the test fails
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask=0, shouldPass=False, failureMsg="Expected test to fail because of throttling"
    )

    # Insert throttling error and set throttle mask to ignore it (as integer value)
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN, shouldPass=True, 
        failureMsg="Expected test to pass because throttle mask (interger bitmask) ignores the throttle reason"
    )

    # Insert throttling error and set throttle mask to ignore it (as string name)
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask="HW_SLOWDOWN", shouldPass=True, 
        failureMsg="Expected test to pass because throttle mask (named reason) ignores the throttle reason"
    )

    # Insert two throttling errors and set throttle mask to ignore only one (as integer)
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN | DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL, 
        throttle_mask=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN, shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (interger bitmask) ignores one of the throttle reasons"
    )

    # Insert two throttling errors and set throttle mask to ignore only one (as string name)
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN | DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL, 
        throttle_mask="HW_SLOWDOWN", shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (named reason) ignores one of the throttle reasons"
    )

    # Insert throttling error and set throttle mask to ignore a different reason (as integer value)
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask=DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE, shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (interger bitmask) ignores different throttle reason"
    )

    # Insert throttling error and set throttle mask to ignore a different reason (as string name)
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
        throttle_mask="HW_POWER_BRAKE", shouldPass=False, 
        failureMsg="Expected test to fail because throttle mask (named reason) ignores different throttle reason"
    )

    # Clear throttling reasons and mask to verify test passes
    dd.SetThrottleMask("")
    perform_diag_with_throttle_mask_and_verify(
        inserted_error=0, throttle_mask=None, shouldPass=True,
        failureMsg="Expected test to pass because there is no throttling"
    )

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_diag_throttle_mask(handle, gpuIds):
    # This test does not work well with the embedded host engine due to race conditions for inserting failure
    # values.
    helper_test_throttle_mask(handle, gpuIds[0])


def helper_check_diag_stop_on_interrupt_signals(handle, gpuId):
    """
    Verifies that a launched diag is stopped when the dcgmi exelwtable recieves a SIGINT, SIGHUP, SIGQUIT, or SIGTERM
    signal.
    """
    # First check whether the GPU is healthy/supported
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=2",
                           version=dcgm_structs.dcgmRunDiag_version3)
    response = dd.Execute(handle)
    if not check_diag_result_pass(response, gpuId, dcgm_structs.DCGM_SM_PERF_INDEX):
        test_utils.skip_test("Skipping because GPU %s does not pass SM Stress test. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)

    # paths to dcgmi exelwtable
    paths = {
        "Linux_32bit": "./apps/x86/dcgmi",
        "Linux_64bit": "./apps/amd64/dcgmi",
        "Linux_ppc64le": "./apps/ppc64le/dcgmi",
        "Linux_aarch64": "./apps/aarch64/dcgmi"
    }
    # Verify test is running on a supported platform
    if utils.platform_identifier not in paths:
        test_utils.skip_test("Dcgmi is not supported on the current platform.")
    dcgmi_path = paths[utils.platform_identifier]

    def verify_exit_code_on_signal(signum):
        # Ensure that host engine is ready to launch a new diagnostic
        dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr='1')
        success = False
        start = time.time()
        while not success and (time.time() - start) <= 3:
            try:
                dd.Execute(handle)
                success = True
            except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_DIAG_ALREADY_RUNNING):
                # Only acceptable error due to small race condition between the lwvs process exiting and 
                # hostengine actually processing the exit. We try for a maximum of 3 seconds since this 
                # should be rare and last only for a short amount of time
                time.sleep(0.5)

        diagApp = AppRunner(dcgmi_path, args=["diag", "-r", "SM Stress", "-i", "%s" % gpuId,
                                              "-d", "5", "--debugLogFile", "/tmp/lwvs.log"])
        # Start the diag
        diagApp.start(timeout=40)
        logger.info("Launched dcgmi process with pid: %s" % diagApp.getpid())
        
        # Ensure diag is running before sending interrupt signal
        running, debug_output = dcgm_internal_helpers.check_lwvs_process(want_running=True, attempts=50)
        assert running, "The lwvs process did not start within 25 seconds: %s" % (debug_output)
        # There is a small race condition here - it is possible that the hostengine sends a SIGTERM before the 
        # lwvs process has setup a signal handler, and so the lwvs process does not stop when SIGTERM is sent. 
        # We sleep for 1 second to reduce the possibility of this scenario
        time.sleep(1)
        diagApp.signal(signum)
        retCode = diagApp.wait()
        # Check the return code and stdout/stderr output before asserting for better debugging info
        if retCode != (signum + 128):
            logger.error("Got retcode '%s' from launched diag." % retCode)
            if diagApp.stderr_lines or diagApp.stdout_lines:
                logger.info("dcgmi output:")
                for line in diagApp.stdout_lines:
                    logger.info(line)
                for line in diagApp.stderr_lines:
                    logger.error(line)
        assert retCode == (signum + 128)
        # Since the app returns a non zero exit code, we call the validate method to prevent false
        # failures from the test framework
        diagApp.validate()
        # Give the launched lwvs process 15 seconds to terminate.
        not_running, debug_output = dcgm_internal_helpers.check_lwvs_process(want_running=False, attempts=50)
        assert not_running, "The launched lwvs process did not terminate within 25 seconds. pgrep output:\n%s" \
                % debug_output

    # Verify return code on SIGINT
    # We simply verify the return code because explicitly checking whether the lwvs process has terminated is
    # clunky and error-prone
    logger.info("Testing stop on SIGINT")
    verify_exit_code_on_signal(signal.SIGINT)

    # Verify return code on SIGHUP
    logger.info("Testing stop on SIGHUP")
    verify_exit_code_on_signal(signal.SIGHUP)

    # Verify return code on SIGQUIT
    logger.info("Testing stop on SIGQUIT")
    verify_exit_code_on_signal(signal.SIGQUIT)

    # Verify return code on SIGTERM
    logger.info("Testing stop on SIGTERM")
    verify_exit_code_on_signal(signal.SIGTERM)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_diag_stop_on_signal_embedded(handle, gpuIds):
    if not option_parser.options.developer_mode:
        # This test can run into a race condition when using embedded host engine, which can cause lwvs to 
        # take >60 seconds to terminate after receiving a SIGTERM.
        test_utils.skip_test("Skip test for more debugging")
    helper_check_diag_stop_on_interrupt_signals(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_diag_stop_on_signal_standalone(handle, gpuIds):
    helper_check_diag_stop_on_interrupt_signals(handle, gpuIds[0])

def helper_verify_log_file_creation(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="targeted stress", paramsStr="targeted stress.test_duration=10")
    logname = '/tmp/tmp_test_debug_log'
    dd.SetDebugLogFile(logname)
    dd.SetDebugLevel(5)
    response = dd.Execute(handle)
    
    if len(response.systemError.msg) == 0:
        skippedAll = True
        passedCount = 0
        errors = ""
        for gpuIndex in range(response.gpuCount):
            resultType = response.perGpuResponses[gpuIndex].results[dcgm_structs.DCGM_TARGETED_PERF_INDEX].result
            if resultType not in [dcgm_structs.DCGM_DIAG_RESULT_SKIP, dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN]:
                skippedAll = False
                if resultType == dcgm_structs.DCGM_DIAG_RESULT_PASS:
                    passedCount = passedCount + 1
                else:
                    warning = response.perGpuResponses[gpuIndex].results[dcgm_structs.DCGM_TARGETED_PERF_INDEX].error.msg
                    if len(warning):
                        errors = "%s, GPU %d failed: %s" % (errors, gpuIndex, warning)

        if skippedAll == False:
            detailedMsg = "passed on %d of %d GPUs" % (passedCount, response.gpuCount)
            if len(errors):
                detailedMsg = "%s and had these errors: %s" % (detailedMsg, errors)
                logger.info(detailedMsg)
            assert os.path.isfile(logname), "Logfile '%s' was not created and %s" % (logname, detailedMsg)
        else:
            logger.info("The diagnostic was skipped, so we cannot run this test.")
    else:
        logger.info("The diagnostic had a problem when exelwting, so we cannot run this test.")


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_verify_log_file_creation_embedded(handle, gpuIds):
    # The standalone test passes, but this one doesn't, so I will come back to it.
    if not option_parser.options.developer_mode:
        test_utils.skip_test("Skip test for more debugging")
    helper_verify_log_file_creation(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_verify_log_file_creation_standalone(handle, gpuIds):
    helper_verify_log_file_creation(handle, gpuIds)

def helper_throttling_masking_failures(handle, gpuId):
    #####
    # First check whether the GPU is healthy
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=2",
                           version=dcgm_structs.dcgmRunDiag_version3)
    dd.SetThrottleMask(0) # We explicitly want to fail for throttle reasons since this test inserts throttling errors 
                          # for verification
    response = dd.Execute(handle)
    if not check_diag_result_pass(response, gpuId, dcgm_structs.DCGM_SM_PERF_INDEX):
        test_utils.skip_test("Skipping because GPU %s does not pass SM Perf test. "
                             "Please verify whether the GPU is supported and healthy." % gpuId)
    
    #####
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=30",
                           version=dcgm_structs.dcgmRunDiag_version3)
    dd.SetThrottleMask(0)

    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS

    inject_benign = InjectionThread(handle, gpuId, fieldId, 3)
    # Use an offset of 5 to make these errors start after the benign values
    inject_error = InjectionThread(handle, gpuId, fieldId, dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN, 5)

    response = [None]
    def run(dd, response):
        response[0] = dd.Execute(handle)

    logger.info("Injecting benign errors")
    inject_benign.start()

    t = threading.Thread(target=run, args=[dd, response])

    logger.info("Started diag")
    t.start()
    running, debug_output = dcgm_internal_helpers.check_lwvs_process(want_running=True)
    assert running, "Lwvs did not start within 10 seconds. Details: %s" % (debug_output)
    
    # Verify that the inserted benign values are visible in DCGM before inserting actual errors
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, 3, checkInterval=0.1, maxWait=5), \
            "Expected inserted benign values to be visible in DCGM"
    
    logger.info("Injecting actual errors")
    inject_error.start()
    # Verify that the inserted values are visible in DCGM
    # Max wait of 8 is because of 5 second offset + 2 seconds required for 20 matches + 1 second buffer.
    assert dcgm_internal_helpers.verify_field_value(gpuId, fieldId, 8, checkInterval=0.1, numMatches=20, maxWait=8), \
            "Expected inserted errors to be visible in DCGM"
                                                                                                     
    # Real errors should be visible now - ensure the test fails after waiting for it to exit
    inject_benign.Stop()
    inject_error.Stop()
    inject_benign.join()
    inject_error.join()
    assert inject_benign.retCode == dcgm_structs.DCGM_ST_OK, "benign injection failed: %s" % inject_benign.retCode
    assert inject_error.retCode == dcgm_structs.DCGM_ST_OK, "error injection failed: %s" % inject_error.retCode
                
    logger.info("Waiting for diag to finish")
    t.join()

    assert check_diag_result_fail(response[0], gpuId, dcgm_structs.DCGM_SM_PERF_INDEX), \
            "Test should have failed.\nGot result: %s (\ninfo: %s,\n warning: %s)" % \
            (response[0].perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].result,
             response[0].perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].info,
             response[0].perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].error.msg)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_throttling_masking_failures_standalone(handle, gpuIds):
    if test_utils.is_throttling_masked_by_lwvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the throttling we inject for this test")
    helper_throttling_masking_failures(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_handle_conlwrrency_standalone(handle, gpuIds):
    '''
    Test that we can use a DCGM handle conlwrrently with a diagnostic running
    '''
    diagDuration = 10

    gpuId = gpuIds[0]

    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=%d" % diagDuration,
                           version=dcgm_structs.dcgmRunDiag_version3)
    
    response = [None]
    def run(dd, response):
        response[0] = dd.Execute(handle)
    
    diagStartTime = time.time()
    threadObj = threading.Thread(target=run, args=[dd, response])
    threadObj.start()

    #Give threadObj a head start on its 10 second run
    time.sleep(1.0)

    firstReturnedRequestLatency = None
    numConlwrrentCompleted = 0
    sleepDuration = 1.0

    while threadObj.isAlive():
        #Make another request on the handle conlwrrently
        moduleStatuses = dcgm_agent.dcgmModuleGetStatuses(handle)
        secondRequestLatency = time.time() - diagStartTime
        numConlwrrentCompleted += 1

        if firstReturnedRequestLatency is None:
            firstReturnedRequestLatency = secondRequestLatency
        
        time.sleep(sleepDuration)
    
    diagThreadEndTime = time.time()
    diagDuration = diagThreadEndTime - diagStartTime
    
    if firstReturnedRequestLatency is None:
        test_utils.skip_test("Diag returned instantly. It is probably not supported for gpuId %u" % gpuId)
    
    logger.info("Completed %d conlwrrent requests. Diag ran for %.1f seconds" % (numConlwrrentCompleted, diagDuration))
    
    #We should have been able to complete a request every 2 seconds if we slept for 1 (conservatively)
    numShouldHaveCompleted = int((diagDuration / sleepDuration) / 2.0)
    assert numConlwrrentCompleted >= numShouldHaveCompleted, "Expected at least %d conlwrrent tests completed. Got %d" % (numShouldHaveCompleted, numConlwrrentCompleted)


def helper_per_gpu_responses_api(handle, gpuIds):
    """
    Verify that pass/fail status for diagnostic tests are reported on a per GPU basis via dcgmActiolwalidate API call
    """
    #####
    # First check whether the GPUs are healthy
    failGpuId = gpuIds[0]
    dd = DcgmDiag.DcgmDiag(gpuIds=[failGpuId], testNamesStr="SM Stress", paramsStr="sm stress.test_duration=2",
                           version=dcgm_structs.dcgmRunDiag_version3)
    dd.SetThrottleMask(0) # We explicitly want to fail for throttle reasons since this test inserts throttling errors 
                          # for verification
    response = dd.Execute(handle)
    if not check_diag_result_pass(response, failGpuId, dcgm_structs.DCGM_SM_PERF_INDEX):
        test_utils.skip_test("Skipping because GPU %s does not pass SM Perf test. "
                             "Please verify whether the GPU is supported and healthy." % failGpuId)
    
    #####
    # Verify API response
    dd = DcgmDiag.DcgmDiag(gpuIds=gpuIds, testNamesStr="SM Stress", paramsStr="sm stress.test_duration=5",
                           version=dcgm_structs.dcgmRunDiag_version3)
    dd.SetThrottleMask(0)
    response = [None]
    def run(dd, response):
        response[0] = dd.Execute(handle) # dcgmActiolwalidate_v2 called by this method

    # Setup injection app    
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
    inject_error = InjectionThread(handle, failGpuId, fieldId, dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN)
    logger.info("Injecting HW_SLOWDOWN throttle error for GPU %s" % failGpuId)
    inject_error.start()
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(failGpuId, fieldId, 8, maxWait=5), \
            "Expected inserted values to be visible in DCGM"

    t = threading.Thread(target=run, args=[dd, response])
    logger.info("Started diag")
    t.start()
                                                                                                     
    # Wait for diag to finish
    logger.info("Waiting for diag to finish")
    t.join()

    # Stop error insertion
    logger.info("Stopped error injection")
    inject_error.Stop()
    inject_error.join()
    assert inject_error.retCode == dcgm_structs.DCGM_ST_OK, "Error injection failed: %s" % inject_error.retCode

    # Verify that responses are reported on a per gpu basis. Ensure the first GPU failed, and all others passed
    for i, gpuId in enumerate(gpuIds):
        desired_result = dcgm_structs.DCGM_DIAG_RESULT_PASS
        if i == failGpuId:
            desired_result = dcgm_structs.DCGM_DIAG_RESULT_FAIL

        assert response[0].perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].result == desired_result, \
            "Expected GPU %s to %s.\nGot result: %s (\ninfo: %s,\n warning: %s)" % \
            (gpuId, "fail" if i == failGpuId else "pass",
             response[0].perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].result,
             response[0].perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].info,
             response[0].perGpuResponses[gpuId].results[dcgm_structs.DCGM_SM_PERF_INDEX].error.msg)

def helper_per_gpu_responses_dcgmi(handle, gpuIds):
    """
    Verify that pass/fail status for diagnostic tests are reported on a per GPU basis via dcgmi (for both normal stdout 
    and JSON output).
    """
    def print_output(app):
            for line in app.stdout_lines:
                logger.info(line)
            for line in app.stderr_lines:
                logger.error(line)

    def verify_successful_dcgmi_run(app):    
        app.start(timeout=20)
        logger.info("Started dcgmi diag with pid %s" % app.getpid())
        retcode = app.wait()
        # dcgm returns DCGM_ST_LWVS_ERROR on diag failure (which is expected here).
        expected_retcode = c_uint8(dcgm_structs.DCGM_ST_LWVS_ERROR).value
        if retcode != expected_retcode:
            if app.stderr_lines or app.stdout_lines:
                    logger.info("dcgmi output:")
                    print_output(app)
        assert retcode == expected_retcode, \
            "Expected dcgmi diag to have retcode %s. Got return code %s" % (expected_retcode, retcode)
        app.validate() # non-zero exit code must be validated
    
    # Setup injection app
    fieldId = dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
    inject_error = InjectionThread(handle, gpuIds[0], fieldId, dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN)
    logger.info("Injecting HW_SLOWDOWN throttle error for GPU %s" % gpuIds[0])
    inject_error.start()
    # Verify that the inserted values are visible in DCGM before starting the diag
    assert dcgm_internal_helpers.verify_field_value(gpuIds[0], fieldId, 8, maxWait=5), \
            "Expected inserted values to be visible in DCGM"

    # Verify dcgmi output
    gpuIdStrings = list(map(str, gpuIds))
    gpuList = ",".join(gpuIdStrings)
    args = ["diag", "-r", "SM Stress", "-p", "sm stress.test_duration=5", "-i", gpuList, "--throttle-mask", "0"]
    dcgmiApp = DcgmiApp(args=args)

    logger.info("Verifying stdout output")
    verify_successful_dcgmi_run(dcgmiApp)
    # Verify dcgmi output shows per gpu results (crude approximation of verifying correct console output)
    stress_header_found = False
    fail_gpu_found = False
    fail_gpu_text = "Fail - GPU: %s" % gpuIds[0]
    check_for_warning = False
    warning_found = False
    for line in dcgmiApp.stdout_lines:
        if not stress_header_found:
            if "Stress" not in line:
                continue
            stress_header_found = True
            continue
        if not fail_gpu_found:
            if fail_gpu_text not in line:
                continue
            fail_gpu_found = True
            check_for_warning = True
            continue
        if check_for_warning:
            if "Warning" in line:
                warning_found = True
            break

    if not (stress_header_found and fail_gpu_found and warning_found):
        logger.info("dcgmi output:")
        print_output(dcgmiApp)

    assert stress_header_found, "Expected to see 'Stress' header in output"
    assert fail_gpu_found, "Expected to see %s in output" % fail_gpu_text
    assert warning_found, "Expected to see 'Warning' in output after GPU failure text"

    # Verify JSON output
    logger.info("Verifying JSON output")
    args.append("-j")
    dcgmiApp = DcgmiApp(args=args)
    verify_successful_dcgmi_run(dcgmiApp)

    # Stop error insertion
    logger.info("Stopped error injection")
    inject_error.Stop()
    inject_error.join()
    assert inject_error.retCode == dcgm_structs.DCGM_ST_OK, "Error injection failed: %s" % inject_error.retCode

    # Verify per GPU results
    json_output = "\n".join(dcgmiApp.stdout_lines)
    output = json.loads(json_output)
    verifed = False
    if (len(output.get("DCGM GPU Diagnostic", {}).get("test_categories", [])) == 2
            and output["DCGM GPU Diagnostic"]["test_categories"][1].get("category", None) == "Stress"
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["name"] == "SM Stress"
            and len(output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"]) >= 2
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["gpu_ids"] == str(gpuIds[0])
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"][0]["status"] == "Fail"
            and output["DCGM GPU Diagnostic"]["test_categories"][1]["tests"][0]["results"][1]["status"] == "Pass"):
        verifed = True

    if not verifed:
        print_output(dcgmiApp)

    assert verifed, "dcgmi JSON output did not pass verification"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgm_diag_per_gpu_responses_standalone(handle, gpuIds):
    if test_utils.is_throttling_masked_by_lwvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the throttling we inject for this test")
    if len(gpuIds) < 2:
        test_utils.skip_test("Skipping because this test requires 2 or more GPUs with same SKU")
    logger.info("Starting test for per gpu responses (API call)")
    helper_per_gpu_responses_api(handle, gpuIds)
    logger.info("Starting test for per gpu responses (dcgmi output)")
    helper_per_gpu_responses_dcgmi(handle, gpuIds)
