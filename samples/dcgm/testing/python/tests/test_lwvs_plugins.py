import pydcgm
import dcgm_structs
import logger
import test_utils
import dcgm_fields
import dcgm_internal_helpers
import DcgmDiag
import option_parser

import signal
import threading
import time

from dcgm_internal_helpers import inject_value

################# General helpers #################
def check_diag_result_fail(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL

def check_diag_result_pass(response, gpuIndex, testIndex):
    return response.perGpuResponses[gpuIndex].results[testIndex].result == dcgm_structs.DCGM_DIAG_RESULT_PASS

################# General tests #################

##### Fail early behavior tests
def verify_early_fail_checks_for_test(handle, gpuId, test_name, testIndex):
    """
    Helper method for verifying the fail early checks for the specified test.
    """
    if testIndex == dcgm_structs.DCGM_TARGETED_POWER_INDEX and not option_parser.options.developer_mode:
        # Skip this test since Targeted Power always fails when duration is less than 30 seconds
        test_utils.skip_test("Skipping fail early verification for Targeted Power test. Use developer mode "
                             "to run this test.")
    duration = 2 if testIndex != dcgm_structs.DCGM_TARGETED_POWER_INDEX else 30 # Prevent false failures due to min
                                                                                # duration requirements for Targeted Power
    paramsStr = "%s.test_duration=%s" % (test_name, duration)

    data = [None]
    def runDiag(dd, data): # Simple helper method to run a diag (used as thread target)
        data[0] = dd.Execute(handle)

    ###
    # First verify that the given test passes for the gpu.
    # If it doesn't pass, skip test and add note to check GPU health
    logger.info("Checking whether %s test passes on GPU %s" % (test_name, gpuId))
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=test_name, paramsStr=paramsStr)
    dd.SetThrottleMask(0)
    test_name_no_spaces = test_name.replace(" ", "_")
    logname = '/tmp/lw_' + test_name_no_spaces + '%s.log'
    dd.SetDebugLogFile(logname % 1)
    dd.SetDebugLevel(5)
    response = dd.Execute(handle)
    if not check_diag_result_pass(response, gpuId, testIndex):
        test_utils.skip_test("Skipping because GPU %s does not pass %s test. "
                             "Please verify whether the GPU is healthy." % (gpuId, test_name))

    ###
    # Next, verify that the given test passes for the gpu when fail early checks are enabled and no errors are inserted
    logger.info("Checking whether %s test passes on GPU %s with fail early enabled" % (test_name, gpuId))
    duration = 15 if testIndex != dcgm_structs.DCGM_TARGETED_POWER_INDEX else 30 # Prevent false failures due to min
                                                                                 # duration requirements for Targeted Power
    paramsStr = "%s.test_duration=%s" % (test_name, duration)
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=test_name, paramsStr=paramsStr)
    dd.SetThrottleMask(0)
    dd.SetFailEarly(checkInterval=2) # enable fail early checks
    dd.SetDebugLogFile(logname % 2)
    dd.SetDebugLevel(5)

    result_thread = threading.Thread(target=runDiag, args=[dd, data])
    result_thread.start()

    # Ensure lwvs process has started
    running, debug_output = dcgm_internal_helpers.check_lwvs_process(want_running=True)
    assert running, "Lwvs process did not start within 10 seconds. pgrep output: %s" % debug_output

    start = time.time()
    result_thread.join()
    end = time.time()

    assert check_diag_result_pass(data[0], gpuId, testIndex), \
        "Expected %s test to pass with fail early enabled and no inserted errors" % test_name
    assert (end - start) >= duration * 0.9, \
        "Expected %s test to run for at least %ss, but it only ran for %ss." % (test_name, duration, end - start)

    ###
    # Verify fail early behavior by inserting an error.
    # Setup test parameters
    duration = 20 if testIndex != dcgm_structs.DCGM_TARGETED_POWER_INDEX else 30 # Prevent false failures due to min
                                                                                 # duration requirements for Targeted Power
    paramsStr = "%s.test_duration=%s" % (test_name, duration)
    response = None
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=test_name, paramsStr=paramsStr)
    dd.SetThrottleMask(0)
    dd.SetFailEarly(checkInterval=2) # enable fail early checks
    dd.SetDebugLogFile(logname % 3)

    # Setup threads / processes
    result_thread = threading.Thread(target=runDiag, args=[dd, data])
    inject_error = dcgm_internal_helpers.InjectionThread(handle, gpuId,
        dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN)

    logger.info("Verifying fail early behavior for %s test by inserting throttling errors." % test_name)
    # Start inserting errors
    inject_error.start()
    # Ensure that inserted errors are visible
    assert \
        dcgm_internal_helpers.verify_field_value(gpuId, dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS,
                                                 dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN,
                                                 checkInterval=0.1, numMatches=5), \
        "Expected inserted value for CLOCK_THROTTLE_REASONS to be visible in DCGM"

    # Start test thread
    result_thread.start()
    # Ensure lwvs process has started
    running, debug_output = dcgm_internal_helpers.check_lwvs_process(want_running=True)
    assert running, "Lwvs process did not start within 10 seconds. pgrep output: %s" % debug_output
    start = time.time()
    
    # Give the test time to exit and verify that the test exits early
    # Test should exit within 75% of test duration if it is going to fail early. Ideally, it should exit within 
    # 2 failure checks (~ 4 seconds of test start), but we provide bigger buffer to account for delays in starting 
    # the test
    result_thread.join(15)
    test_exited_early = not result_thread.isAlive() # Cache thread isAlive value until we verify it
    end = time.time()

    # Stop the injection app
    inject_error.Stop()
    inject_error.join()
    # Verify injection app stopped correctly
    assert inject_error.retCode == dcgm_structs.DCGM_ST_OK, \
        "There was an error inserting values into dcgm. Return code: %s" % inject_error.retCode

    if not test_exited_early:
        # Wait for the launched diag to end
        result_thread.join()
        end = time.time()
    
    response = data[0]
    # Check whether test exited early
    assert test_exited_early, \
        "Expected %s test to exit early. Test took %ss to complete.\nGot result: %s (\ninfo: %s,\n warning: %s)" \
            % (test_name, (end - start),
               response.perGpuResponses[gpuId].results[testIndex].result,
               response.perGpuResponses[gpuId].results[testIndex].info,
               response.perGpuResponses[gpuId].results[testIndex].error.msg)

    # Verify the test failed
    assert check_diag_result_fail(response, gpuId, testIndex), \
        "Expected %s test to fail due to inserted throttling errors.\nGot result: %s (\ninfo: %s,\n warning: %s)" % \
            (test_name, response.perGpuResponses[gpuId].results[testIndex].result,
             response.perGpuResponses[gpuId].results[testIndex].info,
             response.perGpuResponses[gpuId].results[testIndex].error.msg)

    ###
    # Rerun the test with a shorter duration to verify that the test passes now that there are no inserted errors
    duration = 3 if testIndex != dcgm_structs.DCGM_TARGETED_POWER_INDEX else 30 # Prevent false failures due to min
                                                                                # duration requirements for Targeted Power
    paramsStr = "%s.test_duration=%s" % (test_name, duration)

    logger.info("Verifying that test passes once throttling errors are removed.")
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId], testNamesStr=test_name, paramsStr=paramsStr)
    dd.SetThrottleMask(0)
    dd.SetFailEarly(checkInterval=3) # enable fail early checks
    dd.SetDebugLogFile(logname % 4)
    # Reset throttle error
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, 0, 0)
    # Sleep to ensure no pending errors left
    time.sleep(1)

    response = dd.Execute(handle)
    # Verify the test passed
    assert check_diag_result_pass(response, gpuId, testIndex), \
        "Expected %s test to pass because there is no throttling\nGot result: %s (\ninfo: %s,\n warning: %s)" % \
            (test_name, response.perGpuResponses[gpuId].results[testIndex].result,
             response.perGpuResponses[gpuId].results[testIndex].info,
             response.perGpuResponses[gpuId].results[testIndex].error.msg)

def helper_verify_fail_early_checks(handle, gpuId):
    """
    Verifies that the fail early checks are performed by the Targeted Stress, Targeted Power, SM Stress,
    and Diagnostic tests.
    """
    # Verify SM Stress
    verify_early_fail_checks_for_test(handle, gpuId, "SM Stress", dcgm_structs.DCGM_SM_PERF_INDEX)

    # Verify Diagnostic
    verify_early_fail_checks_for_test(handle, gpuId, "Diagnostic", dcgm_structs.DCGM_DIAGNOSTIC_INDEX)

    # Verify Targeted Stress
    verify_early_fail_checks_for_test(handle, gpuId, "Targeted Stress", dcgm_structs.DCGM_TARGETED_PERF_INDEX)

    # Verify Targeted Power (do this last so that other tests can be verified if developer mode is not enabled)
    verify_early_fail_checks_for_test(handle, gpuId, "Targeted Power", dcgm_structs.DCGM_TARGETED_POWER_INDEX)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_lwvs_plugin_fail_early_checks_standalone(handle, gpuIds):
    if test_utils.is_throttling_masked_by_lwvs(handle, gpuIds[0], dcgm_fields.DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN):
        test_utils.skip_test("Skipping because this SKU ignores the throttling we inject for this test")
    # Embedded host engine is not used for this test since it causes connection errors when using a separate
    # process for injecting throttling errors
    helper_verify_fail_early_checks(handle, gpuIds[0])

################# Software plugin tests #################
def check_software_result_pass(response, index):
    assert 0 <= index < dcgm_structs.LEVEL_ONE_MAX_RESULTS
    return response.levelOneResults[index].result == dcgm_structs.DCGM_DIAG_RESULT_PASS

def check_software_result_pass_all(response):
    for result in response.levelOneResults:
        # ignore tests that are not run
        if result.result != dcgm_structs.DCGM_DIAG_RESULT_PASS \
                and result.result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
            return False
    return True

def check_software_result_fail(response, index):
    assert 0 <= index < dcgm_structs.LEVEL_ONE_MAX_RESULTS
    return response.levelOneResults[index].result == dcgm_structs.DCGM_DIAG_RESULT_FAIL

def check_software_result_fail_all(response):
    for result in response.levelOneResults:
        # ignore tests that are not run
        if result.result != dcgm_structs.DCGM_DIAG_RESULT_FAIL \
                and result.result != dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
            return False
    return True


def helper_check_software_page_retirements_fail_on_pending_retirements(handle, gpuId):
    """
    Ensure that the software test for page retirements fails when there are pending page retirements.
    """
    # First verify that the software test passes for the gpu.
    # If it doesn't pass, skip test and add note to check GPU health
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId])
    response = dd.Execute(handle)
    if not check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT):
        test_utils.skip_test("Skipping because GPU %s does not pass software page retirement test. "
                             "Please verify whether the GPU is healthy." % gpuId)

    # Inject some pending page retirements
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 1, -30, True)
    response = dd.Execute(handle)
    # Ensure software test failed due to pending page retirments
    assert check_software_result_fail(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
        "Expected software test to fail due to pending page retirements in the GPU"

    # Reset injected value
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING, 0, -30, True)
    # Ensure diag passes now
    response = dd.Execute(handle)
    assert check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
        "Expected software test to pass"

# TODO: Figure out why embedded values are not detected when using the embedded host engine
# @test_utils.run_with_embedded_host_engine()
# @test_utils.run_only_with_live_gpus()
# def test_lwvs_plugin_software_pending_page_retirements_embedded(handle, gpuIds):
#     helper_check_software_page_retirements_fail_on_pending_retirements(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_lwvs_plugin_software_pending_page_retirements_standalone(handle, gpuIds):
    # This test only works with the standalone host engine - the injected values are not detected when
    # using the embedded host engine
    helper_check_software_page_retirements_fail_on_pending_retirements(handle, gpuIds[0])


def helper_check_software_page_retirements_fail_total_retirements(handle, gpuId):
    """
    Ensure that the software test for page retirements fails when there are mroe than 60 page retirements.
    """
    # First verify that the software test passes for the gpu. If it doesn't pass, skip test and add note to check GPU health
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuId])
    response = dd.Execute(handle)
    if not check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT):
        test_utils.skip_test("Skipping because GPU %s does not pass software page retirement test. "
                             "Please verify whether the GPU is healthy." % gpuId)

    # Inject enough page retirements to cause failure
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, 30, -30, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 30, -30, True)
    response = dd.Execute(handle)
    assert check_software_result_fail(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
           "Expected software test to fail due to 60 total page retirements in the GPU"

    # Ensure 59 pages pass injected value
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 29, -30, True)
    # Ensure diag passes now
    response = dd.Execute(handle)
    assert check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
           "Expected software test to pass since there are less than 60 total retired pages"

    # Reset retired pages count and verify pass
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE, 0, -30, True)
    inject_value(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE, 0, -30, True)
    # Ensure diag still passes
    response = dd.Execute(handle)
    assert check_software_result_pass(response, dcgm_structs.DCGM_SWTEST_PAGE_RETIREMENT), \
           "Expected software test to pass since there are no retired pages"

# TODO: Figure out why embedded values are not detected when using the embedded host engine
# @test_utils.run_with_embedded_host_engine()
# @test_utils.run_only_with_live_gpus()
# def test_lwvs_plugin_software_total_page_retirements_embedded(handle, gpuIds):
#     helper_check_software_page_retirements_fail_total_retirements(handle, gpuIds[0])

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_lwvs_plugin_software_total_page_retirements_standalone(handle, gpuIds):
    # This test only works with the standalone host engine - the injected values are not detected when
    # using the embedded host engine
    helper_check_software_page_retirements_fail_total_retirements(handle, gpuIds[0])
