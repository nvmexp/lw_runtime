import getpass
import os
import re
import string
import apps
import shlex
import dcgm_structs
import dcgm_agent_internal
import utils
import test_utils
import logger
import option_parser
import GcovAggregator
import time
import subprocess

from apps import DecodeLogsApp
from tests.lwswitch_tests import test_lwswitch_utils

MAX_DECODE_SIZE_KB = 1024 * 1024 # 1 GB

def log_elwironment_info():
    if utils.is_linux():
        logger.info("Xorg running:        %s" % test_utils.is_xorg_running())
    logger.info("Platform identifier: %s" % utils.platform_identifier)
    logger.info("Bare metal:          %s" % utils.is_bare_metal_system())
    logger.info("Running as user:     %s" % getpass.getuser())
    logger.debug("ELW : %s" % string.join(map(str, sorted(os.elwiron.items())), "\n"))
        
##################################################################################
### If configured, has our gcov file aggregator generate a fresh round of gcov
### files based on the .gcdas, aggregate our totals into the master gcov files,
### and then clean up.
### NOTE: This needs to be called every time a process dies because the .gcdas will
### be overwritten on a restart.
##################################################################################
def generate_and_aggregate_coverage_files():
    if GcovAggregator.g_gcovAggregator:
        GcovAggregator.g_gcovAggregator.UpdateCoverage()

def decode_files_if_exist(filenames):
    '''
    Decode the given logfile filenames if they exist on the file system
    '''
    for filename in filenames:
        if filename is None:
            continue
        if os.path.isfile(filename):
            if os.path.getsize(filename) <= MAX_DECODE_SIZE_KB * 1024:
                with test_utils.SubTest("decode log %s" % filename):
                    DecodeLogsApp(filename).run()
            else:
                logger.info("not decoding %s because it is bigger than %d MB" % 
                            (utils.shorten_path(filename), MAX_DECODE_SIZE_KB / 1024))

##################################################################################
### Kills the specified processes. If murder is specified, then they are kill -9'ed
### instead of nicely killed.
##################################################################################
def kill_process_ids(process_ids, murder):
    running = False
    for pid in process_ids:
        if not pid:
            break
        running = True
        if murder:
            cmd = 'kill -9 %s' % pid
        else:
            cmd = 'kill %s' % pid
            runner = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            output, error = runner.communicate()

    return running

##################################################################################
### Cleans up the hostengine if needed. If we can't clean it up, then we will 
### abort the testing framework.
##################################################################################
def kill_hostengine_if_needed():
    running = False
    need_to_validate = False
    for i in range(0,2):
        process_ids = test_utils.check_for_running_hostengine_and_log_details(True)
        running = kill_process_ids(process_ids, False)

        if running == False:
            break
        need_to_validate = True
        time.sleep(.5)

    if running:
        for i in range(0,2):
            process_ids = test_utils.check_for_running_hostengine_and_log_details(True)
            running = kill_process_ids(process_ids, True)

        msg = "Cannot run test! An instance of lw-hostengine is running and cannot be killed."
        msg += " Ensure lw-hostengine is stopped before running the tests."
        pids = test_utils.check_for_running_hostengine_and_log_details(False)
        assert not pids, msg

def run_tests():
    '''
    testDir: Subdirectory to look for tests in. For example: "tests" for LWML

    '''
    with test_utils.SubTest("Main"):

        log_elwironment_info()

        test_utils.RestoreDefaultElwironment.restore_elw()
        try:
            dcgm_structs._dcgmInit()
            
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_LIBRARY_NOT_FOUND):
            logger.warning("DCGM Library hasn't been found in the system, is the driver correctly installed?")

            if utils.is_linux() and utils.is_32bit() and utils.is_system_64bit():
                # 32bit test on 64bit system
                logger.warning("Make sure that you've installed driver with both 64bit and 32bit binaries (e.g. not -internal.run or -no-compact32.run)")
            raise

        with test_utils.RunEmbeddedHostEngine() as handle:
            dcgmGpuCount = test_utils.log_gpu_information(handle)
            if dcgmGpuCount < 1:
                logger.error("No DCGM-whitelisted GPUs found. Skipping tests.")
                return
        
        with test_utils.SubTest("restore state", quiet=True):
            test_utils.RestoreDefaultElwironment.restore() # restore the lwml settings

        test_content = test_utils.get_test_content()
        lwswitchModuleCounter = 0
        try:
            for module in test_content:
                # Skip all tests in lwswitch module if --lwswitch option is not passed in
                if module[0].__name__.split(".")[1] == "lwswitch_tests":
                    if not test_utils.is_lwswitch_detected():
                        continue

                    if not utils.is_bare_metal_system():
                        logger.info("LWSwitch tests are not supported in virtualization environment.")
                        continue

                    if not test_lwswitch_utils.is_dgx_2_full_topology():
                        logger.info("LWSwitch tests are not supported with partial topology.")
                        continue

                    # lwpu-fabricmanager service has to be stopped before the test
                    lwidia_fabricmanager_running = test_utils.is_lwidia_fabricmanager_running()

                    # stop fabricmanager service before fm tests
                    # because lw-hostengine app will be used
                    if lwidia_fabricmanager_running:
                        test_utils.stop_lwidia_fabricmanager()

                    if "with_running_fm" in module[0].__name__:
                        lwswitchModuleCounter += 1
                        # tests that need to have fabric manager already running
                        fmArgs = "-l -g --log-level 4 --log-rotate --log-filename /var/log/fabricmanager.log"
                        he_app = apps.LwHostEngineApp(shlex.split(fmArgs))
                        if lwswitchModuleCounter <= 2:
                            he_app.start(timeout=900)
                            logger.info("Start fabric manager pid: %s" % he_app.getpid())

                            # Wait for the fabric manager ready
                            test_utils.wait_for_fabric_manager_ready()

                        with test_utils.SubTest("module %s" % module[0].__name__):
                            for function in module[1]:
                                test_utils.run_subtest(function)
                                generate_and_aggregate_coverage_files()

                        if test_utils.is_hostengine_running():
                            logger.info("Stop fabric manager")
                            he_app.terminate()
                            he_app.validate()
                    else:
                        # tests that do not need running fabric manager
                        with test_utils.SubTest("module %s" % module[0].__name__):
                            for function in module[1]:
                                test_utils.run_subtest(function)
                                generate_and_aggregate_coverage_files()

                    # restore lwpu-fabricmanager service if it was running before tests
                    if lwidia_fabricmanager_running:
                        test_utils.start_lwidia_fabricmanager()

                else:
                    # Attempt to clean up stranded processes instead of aborting
                    kill_hostengine_if_needed()

                    with test_utils.SubTest("module %s" % module[0].__name__):
                        for function in module[1]:
                            test_utils.run_subtest(function)
                            with test_utils.SubTest("%s - restore state" % (function.__name__), quiet=True):
                                test_utils.RestoreDefaultElwironment.restore()

                            generate_and_aggregate_coverage_files()


        finally:
            # SubTest might return KeyboardInterrupt exception. We should try to restore
            # state before closing
            with test_utils.SubTest("restore state", quiet=True):
                test_utils.RestoreDefaultElwironment.restore()
            #dcgm_structs.dcgmShutdown()

        #Decode the DCGM and LWML logs
        decode_files_if_exist([logger.dcgm_trace_log_filename, logger.lwml_trace_log_filename])


_test_info_split_non_verbose = re.compile("\n *\n") # Matches empty line that separates short from long version of function_doc
_test_info_split_verbose_first_newlines = re.compile("^[\n ]*\n") # Matches empty lines at the beginning of string
_test_info_split_verbose_last_newlines = re.compile("[\n ]*$") # Matches empty lines at the end of the string
def print_test_info():
    """
    testDir: Subdirectory to look for tests in
    """
    #Colwert module subdirectory into module dot path like tests/lwvs/x => tests.lwvs.x
    testDirWithDots = test_utils.test_directory.replace("/", ".")

    test_content = test_utils.get_test_content()
    for module in test_content:
        module_name = module[0].__name__
        module_name = module_name.replace("%s." % testDirWithDots, "", 1) # all tests are in testDir. module there's no use in printing that
        for function in module[1]:
            function_name = function.__name__
            function_doc = function.__doc__
            if function_doc is None:
                # verbose output uses indentation of the original string
                function_doc = "    Missing doc"
            
            if option_parser.options.verbose:
                # remove new lines at the beginning and end of the function_doc
                function_doc = _test_info_split_verbose_first_newlines.sub("", function_doc)
                function_doc = _test_info_split_verbose_last_newlines.sub("", function_doc)
                print "%s.%s:\n%s\n" % (module_name, function_name, function_doc)
            else:
                # It's non verbose output so just take the first part of the description (up to first double empty line) 
                function_doc = _test_info_split_non_verbose.split(function_doc)[0]
                # remove spaces at beginning of each line (map strip), remove empty lines (filter bool) and make it one line (string join)
                function_doc = string.join(filter(bool, map(string.strip, function_doc.split("\n"))), " ")
                print "%s.%s:\n\t%s" % (module_name, function_name, function_doc)
