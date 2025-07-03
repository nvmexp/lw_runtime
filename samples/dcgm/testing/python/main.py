#! /usr/bin/elw python

# Early check to make sure we're at a supported version
# Several framework features require 2.7 and higher
# We've yet to vet 3.0+
import sys
if sys.version < '2.7':
    print 'DCGM Testing framework requires at least Python 2.7. Python %s is not supported' % sys.version
    sys.exit(1)
if sys.version >= '3':
    print('DCGM Testing framework is not compatible with Python 3. Please run using Python 2.7')
    sys.exit(1)

import os
import platform
import test_utils
import option_parser
import logger
import utils
import linting
import shutil
import GcovAggregator

from subprocess import check_output, Popen, CalledProcessError
from run_tests import run_tests
from run_tests import print_test_info

def is_file_binary(FileName):
    """ Checks for binary files and skips logging if True """
    try:
        with open(FileName, 'rb') as f:
            # Files with null bytes are binary
            if b'\x00' in f.read():
                print "\n=========================== " + FileName + " ===========================\n"
                print("File is binary, skipping log output!")
                return True
            else:
                return False
    except IOError:
        pass

def _summarize_tests():

    test_root = test_utils.SubTest.get_all_subtests()[0]
    tests_ok_count = test_root.stats[test_utils.SubTest.SUCCESS]
    tests_fail_count = test_root.stats[test_utils.SubTest.FAILED]
    tests_waived_count = test_root.stats[test_utils.SubTest.SKIPPED]
    tests_count = tests_ok_count + tests_fail_count

    # Dump all log output in Eris
    if tests_fail_count > 0 and option_parser.options.eris:
        logPath = os.path.join(logger.default_log_dir, logger.log_dir)
        logFiles = os.listdir(logPath)
        for logFile in logFiles:
            if not is_file_binary(os.path.join(logPath,logFile)):
                print "\n=========================== " + logFile + " ===========================\n"
                with open(os.path.join(logPath,logFile), "r") as f:
                    shutil.copyfileobj(f, sys.stdout)
    
    logger.info("\n========== TEST SUMMARY ==========\n")
    logger.info("Passed: {}".format(tests_ok_count))
    logger.info("Failed: {}".format(tests_fail_count))
    logger.info("Waived: {}".format(tests_waived_count))
    logger.info("Total:  {}".format(tests_count))

    tests_completed_ratio = 0.0
    if tests_count > 0.0:
        tests_completed_ratio = float(tests_ok_count) / (float(tests_count) - (float(tests_fail_count / 2)))
    logger.info("Score:  %.2f" % (100.0 * tests_completed_ratio))
    logger.info("==================================\n\n")

    warnings_count = logger.messages_level_counts[logger.WARNING]
    if warnings_count > 0:
        logger.warning("Framework encountered %d warning%s" % (warnings_count, utils.plural_s(warnings_count)))

    if tests_ok_count < tests_count:
        logger.info()
        logger.info("Bug filing instructions:")
        logger.info(" * In bug description please include first and last error")
        logger.info(" * Also attach %s file (it already contains lwml trace logs, lwpu-bug report and stdout)" % (logger.log_archive_filename))
        
    if not option_parser.options.lint:
        logger.warning('You have skipped running the linter against your python files.  '
                       + 'YOU MUST DO THIS BEFORE CHECKING IN ANY CODE')

def _run_burn_in_tests():
    file_name = "burn_in_stress.py"
    if os.path.exists(file_name):
        logger.info("\nRunning a single iteration of Burn-in Stress Test! \nPlease wait...\n")

        #updates environment for the child process
        elw = os.elwiron.copy()

        #remove elw. variables below to prevent log file locks
        if "__DCGM_DBG_FILE" in elw: del elw["__DCGM_DBG_FILE"]
        if "__LWML_DBG_FILE" in elw: del elw["__LWML_DBG_FILE"]

        burn = Popen(["python", file_name, "-t", "3"], stdout=None, stderr=None, elw = elw)
        
        if burn.pid == None:
            assert False, "Failed to launch Burn-in Tests" 
        burn.wait()
    else:
        logger.warning("burn_in_stress.py script not found!")

class TestFrameworkSetup(object):
    def __enter__(self):
        '''Initialize the test framework or exit on failure'''
        
        # Make sure that the MPS server is disabled before running the test-suite
        if utils.is_mps_server_running():
            print('DCGM Testing framework is not interoperable with MPS server. Please disable MPS server.')
            sys.exit(1)
        
        # Various setup steps
        option_parser.parse_options() 
        utils.verify_user_file_permissions()
        if not test_utils.noLogging:
            logger.setup_elwironment()

            if logger.log_dir:
                logger.close()
            
        option_parser.validate()
        
        if not test_utils.is_framework_compatible():
            logger.fatal("The test framework and dcgm versions are incompatible. Exiting Test Framework.")
            sys.exit(1)

        # Directory where DCGM test*.py files reside
        test_utils.set_tests_directory('tests')

        # Verify that package architecture matches python architecture
        if utils.is_64bit():
            # ignore this check on ppc64le and armv8 for now
            if not (platform.machine() == "ppc64le" or platform.machine() == "aarch64"):
                if not os.path.exists(os.path.join(utils.script_dir, "apps/amd64")):
                    print("Testing package is missing 64bit binaries, are you sure you're using package of correct architecture?")
                    sys.exit(1)
        else:
            if not os.path.exists(os.path.join(utils.script_dir, "apps/x86")):
                print("Testing package is missing 32bit binaries, are you sure you're using package of correct architecture?")
                sys.exit(1)

        # Stops the framework if running python 32bits on 64 bits OS
        if utils.is_windows():
            if os.name == "nt" and "32 bit" in sys.version and platform.machine() == "AMD64":
                print("Running Python 32-bit on a 64-bit OS is not supported. Please install Python 64-bit")
                sys.exit(1)

        if utils.is_linux():
            python_exec = str(sys.exelwtable)
            python_arch = check_output(["file", "-L", python_exec])

            if "32-bit" in python_arch and utils.is_64bit() == True:
                print("Running Python 32-bit on a 64-bit OS is not supported. Please install Python 64-bit")
                sys.exit(1)

        #Tell DCGM how to find our testing package's LWVS
        if not os.path.isfile('./apps/lwvs/lwvs'):
            logger.warning("LWVS is missing from the test framework install. Hopefully it's installed.")
        else:
            lwvsDir = os.getcwd() + '/apps/lwvs'
            logger.debug("LWVS directory: %s" % lwvsDir)
            os.elwiron['LWVS_BIN_PATH'] = lwvsDir #The elw variable parser in DcgmDiagManager is only the directory

    def __exit__(self, type, value, traceback):
        logger.close()
        pass

def main():

    with TestFrameworkSetup():

        if not option_parser.options.no_elw_check:
            if not test_utils.is_test_elwironment_sane():
                logger.warning("The test environment does not seem to be healthy, test framework cannot continue.")
                sys.exit(1)

        if not option_parser.options.no_process_check:
            if not test_utils.are_gpus_free():
                sys.exit(1)
        else:
            logger.warning("Not checking for other processes using the GPU(s), test failures may occur.")

        if option_parser.options.test_info:
            print_test_info()
            return

        if option_parser.options.clear_lint_artifacts:
            linting.clear_lint_artifacts()

        if option_parser.options.lint:
            errFileCount = linting.pylint_dcgm_files()
            if errFileCount > 0:
                sys.exit(1)

        if option_parser.options.coverage:
            GcovAggregator.g_gcovAggregator = GcovAggregator.GcovAggregator(option_parser.options.coverage)

        if test_utils.noLogging:
            run_tests()
        else:
            logger.run_with_coverage(run_tests())

        _summarize_tests()

        # Runs a single iteration of burn_in_stress test
        if option_parser.options.burn:
            _run_burn_in_tests()

if __name__ == '__main__':
    main()
