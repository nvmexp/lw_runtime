from __future__ import print_function

import os
import sys
import copy
import test_utils
import utils
import pydcgm
import option_parser
import time
import glob
import shutil
import json
import logger
import argparse
import LwidiaSmiChecker
import DcgmiDiag
import dcgm_fields
import version

from subprocess import Popen, STDOUT, PIPE, check_output

logFile = "lwvs_diag.log"
debugFile = "lwvs_debug.log"
goldelwaluesFile = "/tmp/golden_values.yml"
LOAD_TOO_HIGH_TO_TRAIN = 1010101010
PASSED_COUNT = 0
FAILED_COUNT = 0
WAIVED_COUNT = 0


################################################################################
def remove_file_yolo(filename):
    '''
    Try to remove a file, not caring if any error oclwrs
    '''
    try:
        os.remove(filename)
    except:
        pass


################################################################################
def make_symlink_if_needed(lwrrent_location):
    lwvs_hardcoded_path = '/usr/share/lwpu-validation-suite/lwvs'
    lwvs_hardcoded_dir = '/usr/share/lwpu-validation-suite'
    if not os.path.isfile(lwvs_hardcoded_path):
        if not os.path.exists(lwvs_hardcoded_dir):
            os.makedirs(lwvs_hardcoded_dir)

        lwvs_path = "%s/lwvs" % (lwrrent_location)
        try:
            os.symlink(lwvs_path, lwvs_hardcoded_path)
        except OSError as msg:
            with open("lwvs_error.log", "w") as f:
                f.write("Error: %s - Message: %s\n" % (msg.errno, msg.strerror))

        return lwvs_hardcoded_path

    return None


################################################################################
def setupElwironment(cmdArgs):
    """
    Function to prepare the test environment
    """
    message = ''

    # Verify if GPUs are free before running the tests
    if not test_utils.are_gpus_free():
        print("Some GPUs are in use, please check the workload and try again")
        sys.exit(1)

    if test_utils.is_framework_compatible() is False:
        print("run_dcgm_diagnostic.py found to be a different version than DCGM. Exiting")
        sys.exit(1)
    else:
        print("Running against CL %s" % version.CHANGE_LIST)

    # Enable persistence mode or the tests will fail
    if utils.is_root():
        print("\nEnabling persistence mode")
        lwsmi_cmd = "lwpu-smi -pm 1"
        try:
            lwsmi_ret = Popen(lwsmi_cmd, stdout=PIPE, stderr=PIPE, shell=True)
            (message, error) = lwsmi_ret.communicate()
            if message:
                print(message)
            if error:
                print(error)
        except OSError as e:
            print("Failed to enable persistence mode.\nError:\n%s" % e)
            time.sleep(1)
    else:
        print("\nWarning! Please make sure to enable persistence mode")
        time.sleep(1)

    # Collects the output of "lwpu-smi -q" and prints it out on the screen for debugging
    print("\n###################### LWSMI OUTPUT FOR DEBUGGING ONLY ##########################")

    lwsmi_cmd = "lwpu-smi -q"
    try:
        lwsmi_ret = Popen(lwsmi_cmd, stdout=PIPE, stderr=PIPE, shell=True)
        (message, error) = lwsmi_ret.communicate()
        if message:
            print(message)
        if error:
            print(error)
    except OSError as e:
        print("Unable to collect \"lwpu-smi -q\" output.\nError:\n%s" % e)
        pass

    print("\n###################### LWSMI OUTPUT FOR DEBUGGING ONLY ##########################\n\n")

    # Tries to remove older log files
    remove_file_yolo(logFile)
    remove_file_yolo(debugFile)

    print("============= TEST CONFIG ==========")
    print("VULCAN EVN:   {}".format(cmdArgs.vulcan))
    print("TEST CYLES:   {}".format(cmdArgs.cycles))
    print("DEVICE LIST:  {}".format(cmdArgs.device_id))
    print("TRAINING:     {}".format(not cmdArgs.notrain))
    print("====================================")


def trimJsonText(text):
    return text[text.find('{'):text.rfind('}') + 1]


NAME_FIELD = "name"
RESULTS_FIELD = "results"
WARNING_FIELD = "warnings"
STATUS_FIELD = "status"
INFO_FIELD = "info"
GPU_FIELD = "gpu_ids"

DIAG_THROTTLE_WARNING = "Clocks are being throttled for"
DIAG_DBE_WARNING = "ecc_dbe_volatile_total"
DIAG_ECC_MODE_WARNING = "because ECC is not enabled on GPU"
DIAG_INFOROM_WARNING = "Error calling LWML API lwmlDeviceValidateInforom"
DIAG_THERMAL_WARNING = "Thermal violations totaling "

DIAG_THROTTLE_SUGGEST = "A GPU's clocks are being throttled due to a cooling issue. Please make sure your GPUs are properly cooled."
DIAG_DBE_SUGGEST = "This GPU needs to be drained and reset to clear the non-recoverable double bit errors."
DIAG_ECC_MODE_SUGGEST = "Run 'lwpu-smi -i <gpu id> -e 1' and then reboot to enable ECC memory."
DIAG_INFOROM_SUGGEST = "A GPU's inforom is corrupt. You should re-flash it with iromflsh or replace the GPU. Run lwpu-smi without arguments to see which GPU."
DIAG_THERMAL_SUGGEST = "A GPU has thermal violations happening. Please make sure your GPUs are properly cooled."

errorTuples = [(dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, DIAG_THROTTLE_WARNING, DIAG_THROTTLE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, DIAG_DBE_WARNING, DIAG_DBE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_ECC_LWRRENT, DIAG_ECC_MODE_WARNING, DIAG_ECC_MODE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_INFOROM_CONFIG_VALID, DIAG_INFOROM_WARNING, DIAG_INFOROM_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, DIAG_THERMAL_WARNING, DIAG_THERMAL_SUGGEST)
               ]


class FailedTestInfo():
    def __init__(self, testname, warning, gpuInfo):
        self.m_warning = warning
        self.m_testname = testname
        self.m_info = ''
        self.m_gpuField = gpuInfo
        self.m_gpuId = None
        if gpuInfo:
            self.m_gpuId = int(gpuInfo)
        self.m_fieldId = None
        self.m_suggestion = ''
        self.m_evaluatedMsg = ''
        for errorTuple in errorTuples:
            if self.m_warning.find(errorTuple[1]) != -1:
                # Matched, record field ID and suggestion
                self.m_fieldId = errorTuple[0]
                self.m_suggestion = errorTuple[2]

    def SetInfo(self, info):
        self.m_info = info

    def GetFullError(self):
        if self.m_evaluatedMsg:
            return self.m_evaluatedMsg

        full = self.m_warning
        if not full:
            full = "SOMEHOW WE ARE FAILING A TEST WITH NO WARNING MESSAGE!!!"
        if self.m_info:
            full += "\n%s" % self.m_info
        if self.m_gpuField:
            full += "\n for GPU(s) %s" % self.m_gpuField

        return full

    def GetFieldId(self):
        return self.m_fieldId

    def GetGpuId(self):
        return self.m_gpuId

    def GetWarning(self):
        return self.m_warning

    def GetTestname(self):
        return self.m_testname

    def SetFailureMessage(self, val, correct_val):
        fieldName = dcgm_fields.DcgmFieldGetTagById(self.m_fieldId)
        if fieldName is None:
            fieldName = "Cannot find field id %d" % self.m_fieldId
        if val is None:
            # Our Lwpu-smi checker doesn't support this value yet
            self.m_evaluatedMsg = "%s\nOur lwpu-smi checker doesn't support evaluating field %s yet." % \
                                  (self.GetFullError(), fieldName)
        elif val != correct_val:
            self.m_evaluatedMsg = None
            if (self.m_fieldId):
                self.m_evaluatedMsg = "%s\nlwidia-smi found a value of %s for field %s instead of %s" % \
                                      (self.GetFullError(), str(val), fieldName, str(correct_val))
            else:
                self.m_evaluatedMsg = self.GetFullError()
        else:
            self.m_evaluatedMsg = "%s\nlwidia-smi found the correct value %s for field %s" %\
                                  (self.GetFullError(), str(val), fieldName)


class TestRunner():

    ################################################################################
    def __init__(self, cycles, dcgmiDiag, vulcan, notrain):
        self.cycles = int(cycles)
        self.dcgmiDiag = dcgmiDiag 
        self.notrain = notrain
        self.failed_runs = 0
        self.vulcan = vulcan
        self.failing_tests = {}
        # The exclusion list is a list of [textToSearchFor, whatToPrintIfFound] entries
        self.exclusions = [
            [DIAG_INFOROM_WARNING, DIAG_INFOROM_SUGGEST],
            [DIAG_THROTTLE_WARNING, DIAG_THROTTLE_SUGGEST],
            [DIAG_THERMAL_WARNING, DIAG_THERMAL_SUGGEST],
        ]

    ################################################################################
    def findFailedTests(self, jsondict, failed_list):
        if not isinstance(jsondict, dict):
            # Only inspect dictionaries
            return

        if RESULTS_FIELD in jsondict:
            # We've found the tst dictionary
            testname = jsondict[NAME_FIELD]
            for item in jsondict[RESULTS_FIELD]:
                if item[STATUS_FIELD] == "Fail":
                    warn = '' 
                    gpuInfo = ''
                    if WARNING_FIELD in item:
                        warn = item[WARNING_FIELD]
                    if GPU_FIELD in item:
                        gpuInfo = item[GPU_FIELD]
                    failed_test = FailedTestInfo(testname, warn, gpuInfo)
                    if INFO_FIELD in item:
                        failed_test.SetInfo(item[INFO_FIELD])

                    failed_list.append(failed_test)
        else:
            for key in jsondict:
                if isinstance(jsondict[key], list):
                    for item in jsondict[key]:
                        self.findFailedTests(item, failed_list)
                else:
                    self.findFailedTests(jsondict[key], failed_list)

    ################################################################################
    def identifyFailingTests(self, jsondict, run_index, nsc):
        failed_list = []
        if 'runtime_error' in jsondict:
            # Experienced a complete failure while trying to run the diagnostic. No need
            # to parse for further errors because there will be no other json entries.
            failInfo = FailedTestInfo('System_Failure', jsondict['runtime_error'], 'ALL')
            failed_list.append(failInfo)
        else:
            self.findFailedTests(jsondict, failed_list)
            for failInfo in failed_list:
                fieldId = failInfo.GetFieldId()
                if fieldId:
                    val, correct_val = nsc.GetErrorValue(failInfo.GetGpuId(), fieldId)
                    failInfo.SetFailureMessage(val, correct_val)
        self.failing_tests[run_index] = failed_list

    ################################################################################
    def matchesExclusion(self, warning):
        for exclusion in self.exclusions:
            if warning.find(exclusion[0]) != -1:
                return exclusion[1]

        return None

    def getErrorMessage(self, failureInfo, runIndex, recommendation):
        msg = ''
        if recommendation:
            msg = "Iteration %d test '%s' is ignoring error '%s' : %s" % \
                (runIndex, failureInfo.GetTestname(), failureInfo.GetFullError(), recommendation)
        else:
            msg = "Iteration %d test '%s' failed: '%s'" % \
                (runIndex, failureInfo.GetTestname(), failureInfo.GetFullError())

        return msg

    ################################################################################
    def checkForErrors(self):
        '''
        Check the LWVS JSON output for errors, filtering out any errors that are elwironmental rather
        than LWVS bugs. Returns a count of the number of errors. Anything > 0 will result in bugs against
        LWVS.

        Returns a tuple of [numErrors, numExclusions]
        '''
        numErrors = 0
        numExclusions = 0
        failureDetails = []
        for key in self.failing_tests:
            runFailures = 0
            for failureInfo in self.failing_tests[key]:
                recommendation = self.matchesExclusion(failureInfo.GetWarning())
                if recommendation:
                    print(self.getErrorMessage(failureInfo, key, recommendation))
                    numExclusions += 1
                else:
                    failureDetails.append(self.getErrorMessage(failureInfo, key, None))
                    runFailures += 1

            if runFailures > 0:
                self.failed_runs += 1
                numErrors += runFailures

        for failure in failureDetails:
            print("%s\n" % failure)

        return [numErrors, numExclusions]

    ################################################################################
    def run_command(self, cycles):
        """
        Helper method to run a give command
        """

        print("Running command: %s " % " ".join(self.dcgmiDiag.BuildDcgmiCommand()))
        ret = 0
        for runIndex in range(cycles):
            self.dcgmiDiag.Run()
            self.failing_tests[runIndex] = self.dcgmiDiag.failed_list
            if self.dcgmiDiag.diagRet and not ret:
                ret = self.dcgmiDiag.diagRet

        # Get the number of actual errors in the output
        failCount, exclusionCount = self.checkForErrors()

        if self.vulcan:
            print(self.dcgmiDiag.lastStdout)
            if self.dcgmiDiag.lastStderr:
                print(self.dcgmiDiag.lastStderr)

        if (failCount != 0):
            if self.failed_runs > 0:
                print("%d of %d runs Failed. Please attach %s and %s to your bug report."
                      % (self.failed_runs, cycles, logFile, debugFile))
            print("ExclusionCount: %d" % exclusionCount)
            print("FailCount: %d" % failCount)
            print("&&&& FAILED")
            print("Diagnostic test failed with code %d.\n" % ret)

            # Popen returns 0 even if diag test fails, so failing here
            return 1
        else:
            print("Success")

        return 0

    ################################################################################
    def run_without_training(self):
        self.dcgmiDiag.SetTraining(False)
        self.dcgmiDiag.SetRunMode(3)
        self.dcgmiDiag.SetConfigFile(None)
        ret = self.run_command(self.cycles)
        return ret

    ################################################################################
    def run_with_training(self):
        retries = 0
        maxRetries = 180
        # Sleep for up to three minutes to see if we can train
        while os.getloadavg()[0] > .5:
            if retries >= maxRetries:
                print("Skipping training! Cannot train because the load is above .5 after %d seconds\n" % maxRetries)
                print("This is not a bug, please make sure that no other workloads are using up the system's resources")
                return LOAD_TOO_HIGH_TO_TRAIN
            retries += 1
            time.sleep(1)

        print("\nTraining once to generate golden values... This may take a while, please wait...\n")

        self.dcgmiDiag.SetRunMode(None)
        self.dcgmiDiag.SetTraining(True)

        ret = self.run_command(1)  # Always train only once
        return ret

    ################################################################################
    def run_with_golden_values(self):
        print("Running tests using existing golden values from %s" % goldelwaluesFile)
        self.dcgmiDiag.SetConfig(goldelwaluesFile)
        self.dcgmiDiag.SetRunMode(3)
        self.dcgmiDiag.SetTraining(False)
        print("Generated golden values file: \n")
        try:
            with open(goldelwaluesFile, 'r') as f:
                print(f.read())
        except IOError as e:
            print("Error attempting to dump the golden values file '%s' : '%s'" % (goldelwaluesFile, str(e)))
            print("Attempting to continue...")

        if os.path.isfile(goldelwaluesFile):
            ret = self.run_command(1)  # Run against the golden values only once
            return ret
        else:
            print("Unable to find golden values file")
            return -1


################################################################################
def checkCmdLine(cmdArgs, settings):

    if cmdArgs.vulcan:
        settings['vulcan'] = True
        cmdArgs.device_id = None
    else:
        settings['vulcan'] = False

    if cmdArgs.device_id:
        # Verify devices have been specified correctly
        if len(cmdArgs.device_id) > 1 and ("," in cmdArgs.device_id):
            gpuIds = cmdArgs.device_id.split(",")
            for gpuId in gpuIds:
                if not gpuId.isdigit():  # despite being named isdigit(), ensures the string is a valid unsigned integer
                    print("Please specify a comma separated list of device IDs.")
                    sys.exit(1)
        elif len(cmdArgs.device_id) > 1 and ("," not in cmdArgs.device_id):
            print("Please specify a comma separated list of device IDs.")
            sys.exit(1)
        elif len(cmdArgs.device_id) == 1:
            if not cmdArgs.device_id[0].isdigit():
                print("\"{}\" is not a valid device ID, please provide a number instead.".format(cmdArgs.device_id[0]))
                sys.exit(1)
        else:
            print("Device list validated successfully")

    settings['dev_id'] = cmdArgs.device_id
    settings['cycles'] = cmdArgs.cycles
    settings['notrain'] = cmdArgs.notrain

################################################################################
def getGoldelwalueDebugFiles():
    # This method copies debug files for golden values to the current directory.
    gpuIdMetricsFileList = glob.glob('/tmp/dcgmgd_withgpuids*')
    gpuIdMetricsFile = None
    allMetricsFileList = glob.glob('/tmp/dcgmgd[!_]*')
    allMetricsFile = None

    if gpuIdMetricsFileList:
        # Grab latest debug file
        gpuIdMetricsFileList.sort()
        gpuIdMetricsFile = gpuIdMetricsFileList[-1]
    if allMetricsFileList:
        # Grab latest debug file
        allMetricsFileList.sort()
        allMetricsFile = allMetricsFileList[-1]

    fileList = []
    try:
        if gpuIdMetricsFile is not None:
            shutil.copyfile(gpuIdMetricsFile, './dcgmgd_withgpuids.txt')
            fileList.append('dcgmgd_withgpuids.txt')
        if allMetricsFile is not None:
            shutil.copyfile(allMetricsFile, './dcgmgd_allmetrics.txt')
            fileList.append('dcgmgd_allmetrics.txt')
        if os.path.isfile(goldelwaluesFile):
            shutil.copyfile(goldelwaluesFile, './golden_values.yml')
            fileList.append('golden_values.yml')
    except (IOError, OSError) as e:
        print("There was an error copying the debug files to the current directory %s" % e)

    if fileList:
        print("Please attach %s to the bug report." % fileList)
    else:
        print("No debug files were copied.")


################################################################################
def parseCommandLine():

    parser = argparse.ArgumentParser(description="DCGM DIAGNOSTIC TEST FRAMEWORK")
    parser.add_argument("-v", "--vulcan", action="store_true", help="Runs with vulcan in Eris environment")
    parser.add_argument("-c", "--cycles", required=True, help="Number of test cycles to run, all tests are one cycle.")
    parser.add_argument("-d", "--device-id", help="Comma separated list of lwml device ids.")
    parser.add_argument("-n", "--notrain", action="store_true", help="Disable training for golden values. Enabled by default.")

    args = parser.parse_args()

    return args

################################################################################
def main(cmdArgs):

    settings = {}
    checkCmdLine(cmdArgs, settings)

    # Prepare the test environment and setup step
    option_parser.initialize_as_stub()
    setupElwironment(cmdArgs)

    # Check if lwvs is installed and define location of dcgmi (not needed in Eris)
    if settings['vulcan'] is False:
        location = "/usr/bin/dcgmi"
        config = "/etc/lwpu-validation-suite/lwvs.conf"

        if not os.path.isfile(location):
            print("LWVS is NOT installed in: %s\n" % location)
            sys.exit(1)

        if not os.path.isfile(config):
            print("LWVS default configuration files is NOT present in: %s\n" % config)
            sys.exit(1)

    # Build a lwvs command list. Each element is an argument
    pluginPathSpec = None
    prefix = None
    lwrrent_location = os.path.realpath(sys.path[0])
    if settings['vulcan']:
        os.elwiron["LWVS_BIN_PATH"] = lwrrent_location
        # Specify the plugin path because ERIS doesn't have the plugins in the default location
        pluginPathSpec = "%s/plugins" % (lwrrent_location)
        if "lwvs" in lwrrent_location:
            prefix = lwrrent_location.strip("lwvs") + "/dcgm"
        else:
            prefix = os.path.join(lwrrent_location, "dcgm")

    removeLater = make_symlink_if_needed(lwrrent_location)

    # Get a list of gpus to run against
    gpuIdStr = ''
    if settings['dev_id'] is None:
        # None specified on the command line. Build compatible lists of GPUs
        dcgmHandle = pydcgm.DcgmHandle(ipAddress=None)
        gpuIds = dcgmHandle.GetSystem().discovery.GetAllSupportedGpuIds()
        gpuGroups = test_utils.group_gpu_ids_by_sku(dcgmHandle.handle, gpuIds)
        if len(gpuGroups) > 1:
            print("This system has more than one GPU SKU; DCGM Diagnostics is defaulting to just GPU(s) %s" %
                  gpuGroups[0])
        gpuGroup = gpuGroups[0]
        gpuIdStr = ",".join(map(str, gpuGroup))
        del(dcgmHandle)
        dcgmHandle = None
    else:
        gpuIdStr = settings['dev_id']

    dcgmiDiag = DcgmiDiag.DcgmiDiag(gpuIds=gpuIdStr, dcgmiPrefix=prefix, runMode=3, debugLevel=5, \
                debugFile=debugFile, pluginPath=pluginPathSpec)

    # Start tests
    run_test = TestRunner(settings['cycles'], dcgmiDiag, settings['vulcan'], settings['notrain'])

    if settings['notrain']:
        print("\nRunning with training mode disabled... This may take a while, please wait...\n")
        ret = run_test.run_without_training()
        if ret != 0:
            print("&&&& FAILED")
            return ret
    else:
        ret = run_test.run_without_training()
        if ret != 0:
            print("Exiting early after a failure in the initial runs of the diagnostic")
            print("&&&& FAILED")
            return ret

        ret = run_test.run_with_training()
        if ret == LOAD_TOO_HIGH_TO_TRAIN:
            return 0
        elif ret != 0:
            getGoldelwalueDebugFiles()
            print("Exiting early after a failure while training the diagnostic")
            print("&&&& FAILED")
            return ret

        # Disable running against the generated golden values for a time until we fix DCGM-1134
        # ret = run_test.run_with_golden_values()
        # if ret != 0:
        #     getGoldelwalueDebugFiles()
        #     print("Failed when running against the generated golden values")
        #     return ret

    if removeLater is not None:
        remove_file_yolo(removeLater)

    return ret

if __name__ == "__main__":
    cmdArgs = parseCommandLine()
    ret = main(cmdArgs)

    if os.path.isfile(logFile):
        with open(logFile, "r") as f:
            log_content = f.readlines()
            for log in log_content:
                if "Pass" in log:
                    PASSED_COUNT += 1
                elif "Fail" in log:
                    FAILED_COUNT += 1
                elif "Skip" in log:
                    WAIVED_COUNT += 1

        logger.info("\n========== TEST SUMMARY ==========\n")
        logger.info("Passed: {}".format(PASSED_COUNT))
        logger.info("Failed: {}".format(FAILED_COUNT))
        logger.info("Waived: {}".format(WAIVED_COUNT))
        logger.info("Total:  {}".format(PASSED_COUNT + FAILED_COUNT + WAIVED_COUNT))
        logger.info("Cycles: {}".format(cmdArgs.cycles))
        logger.info("==================================\n\n")
    else:
        print("Unable to provide test summary due to missing log file")

    sys.exit(ret)
