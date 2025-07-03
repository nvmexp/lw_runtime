import subprocess
import json

import dcgm_structs
import dcgm_agent
import dcgm_fields
import LwidiaSmiChecker

def trimJsonText(text):
    return text[text.find('{'):text.rfind('}') + 1]

logFile = "lwvs_diag.log"

NAME_FIELD = "name"
RESULTS_FIELD = "results"
WARNING_FIELD = "warnings"
STATUS_FIELD = "status"
INFO_FIELD = "info"
GPU_FIELD = "gpu_ids"

DIAG_THROTTLE_WARNING = "Clocks are being throttled for"
DIAG_DBE_WARNING = "ecc_dbe_volatile_total"
DIAG_ECC_MODE_WARNING = "Skipping test because ECC is not enabled on this GPU"
DIAG_INFOROM_WARNING = "lwmlDeviceValidateInforom for lwml device"
DIAG_THERMAL_WARNING = "Thermal violations totaling "
DIAG_VARY_WARNING    = "The results of training DCGM GPU Diagnostic cannot be trusted because they vary too much from run to run"

DIAG_THROTTLE_SUGGEST = "A GPU's clocks are being throttled due to a cooling issue. Please make sure your GPUs are properly cooled."
DIAG_DBE_SUGGEST = "This GPU needs to be drained and reset to clear the non-recoverable double bit errors."
DIAG_ECC_MODE_SUGGEST = "Run lwpu-smi -i <gpu id> -e 1 and then reboot to enable."
DIAG_INFOROM_SUGGEST = "A GPU's inforom is corrupt. You should re-flash it with iromflash or replace the GPU. run lwpu-smi without arguments to see which GPU."
DIAG_THERMAL_SUGGEST = "A GPU has thermal violations happening. Please make sure your GPUs are properly cooled."
DIAG_VARY_SUGGEST = "Please check for transient conditions on this machine that can disrupt consistency from run to run"

errorTuples = [(dcgm_fields.DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, DIAG_THROTTLE_WARNING, DIAG_THROTTLE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, DIAG_DBE_WARNING, DIAG_DBE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_ECC_LWRRENT, DIAG_ECC_MODE_WARNING, DIAG_ECC_MODE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_INFOROM_CONFIG_VALID, DIAG_INFOROM_WARNING, DIAG_INFOROM_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, DIAG_THERMAL_WARNING, DIAG_THERMAL_SUGGEST)
               ]


################################################################################
class FailedTestInfo():
    ################################################################################
    def __init__(self, testname, warning, gpuInfo=None):
        self.m_warning = warning
        self.m_testname = testname
        self.m_info = ''
        self.m_gpuField = gpuInfo
        self.m_gpuId = None
        self.m_isAnError = True
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

    ################################################################################
    def SetInfo(self, info):
        self.m_info = info

    ################################################################################
    def GetFullError(self):
        if self.m_evaluatedMsg:
            return self.m_evaluatedMsg

        if not self.m_warning:
            full = "%s is reported as failed but has no warning message" % self.m_testname
        else:
            full = "%s failed: '%s'" % (self.m_testname, self.m_warning)
        if self.m_info:
            full += "\n%s" % self.m_info
        if self.m_gpuField:
            full += "\n for GPU(s) %s" % self.m_gpuField

        return full

    ################################################################################
    def GetFieldId(self):
        return self.m_fieldId

    ################################################################################
    def GetGpuId(self):
        return self.m_gpuId

    ################################################################################
    def GetWarning(self):
        return self.m_warning

    ################################################################################
    def GetTestname(self):
        return self.m_testname

    ################################################################################
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
            self.m_isAnError = False # lwpu-smi reports an error in this field, so this is not a DCGM mistake
            if (self.m_fieldId):
                self.m_evaluatedMsg = "%s\nlwidia-smi found a value of %s for field %s instead of %s" % \
                                      (self.GetFullError(), str(val), fieldName, str(correct_val))
            else:
                self.m_evaluatedMsg = self.GetFullError()
        else:
            self.m_evaluatedMsg = "%s\nlwidia-smi found the correct value %s for field %s" %\
                                  (self.GetFullError(), str(val), fieldName)

    ################################################################################
    def IsAnError(self):
        return not self.m_isAnError


################################################################################
class DcgmiDiag:

    ################################################################################
    def __init__(self, gpuIds=None, testNamesStr='', paramsStr='', verbose=True, train=False, forceTrain=False, 
                 dcgmiPrefix='', runMode=0, configFile='', debugLevel=0, debugFile='', pluginPath=''):
        self.gpuList = gpuIds
        self.testNamesStr = testNamesStr
        self.paramsStr = paramsStr
        self.verbose = verbose
        self.train = train
        self.forceTrain = forceTrain
        self.dcgmiPrefix = dcgmiPrefix
        self.runMode = runMode
        self.configFile = configFile
        self.debugLevel = debugLevel
        self.debugFile = debugFile
        self.pluginPath = pluginPath

    ################################################################################
    def BuildDcgmiCommand(self):
        cmd = []

        if self.dcgmiPrefix:
            cmd.append("%s/dcgmi" % self.dcgmiPrefix)
        else:
            cmd.append("dcgmi")

        cmd.append("diag")

        if self.train:
            # Ignore run options if we are training
            cmd.append('--train')
            
            if self.forceTrain:
                cmd.append('--force')
        else:
            if self.runMode == 0:
                # Use the test names string if a run mode was not specified
                cmd.append('-r')

                if self.testNamesStr:
                    cmd.append(self.testNamesStr)
                else:
                    # default to running level 3 tests
                    cmd.append('3')
            else:
                # If the runMode has been specified, then use that over the test names string
                cmd.append('-r')
                cmd.append(str(self.runMode))

            if self.paramsStr:
                cmd.append('-p')
                cmd.append(self.paramsStr)

        if self.debugFile:
            cmd.append('--debugLogFile')
            cmd.append(self.debugFile)

            if self.debugLevel:
                cmd.append('-d')
                cmd.append(str(self.debugLevel))

        cmd.append('-j')
        
        if self.verbose:
            cmd.append('-v')

        if self.configFile:
            cmd.append('-c')
            cmd.append(self.configFile)

        if self.gpuList:
            cmd.append('-i')
            cmd.append(self.gpuList)

        if self.pluginPath:
            cmd.append('--plugin-path')
            cmd.append(self.pluginPath)

        return cmd
    
    ################################################################################
    def AddGpuList(self, gpu_list):
        self.gpuList = gpu_list
    
    ################################################################################
    def FindFailedTests(self, jsondict, failed_list):
        if not isinstance(jsondict, dict):
            # Only inspect dictionaries
            return

        if RESULTS_FIELD in jsondict:
            # We've found the test dictionary
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
                        self.FindFailedTests(item, failed_list)
                else:
                    self.FindFailedTests(jsondict[key], failed_list)
    
    ################################################################################
    def IdentifyFailingTests(self, jsondict, nsc):
        failed_list = []
        if 'runtime_error' in jsondict:
            # Experienced a complete failure while trying to run the diagnostic. No need
            # to parse for further errors because there will be no other json entries.
            failInfo = FailedTestInfo('System_Failure', jsondict['runtime_error'], 'ALL')
            failed_list.append(failInfo)
        else:
            self.FindFailedTests(jsondict, failed_list)
            for failInfo in failed_list:
                fieldId = failInfo.GetFieldId()
                if fieldId:
                    val, correct_val = nsc.GetErrorValue(failInfo.GetGpuId(), fieldId)
                    failInfo.SetFailureMessage(val, correct_val)

        return failed_list
    
    ################################################################################
    def SetAndCheckOutput(self, stdout, stderr, ret=0, nsc=None):
        self.lastStdout = stdout
        self.lastStderr = stderr
        self.diagRet = ret
        if not nsc:
            nsc = LwidiaSmiChecker.LwidiaSmiJob()
        return self.CheckOutput(nsc)

    ################################################################################
    def CheckOutput(self, nsc):
        failed_list = []
        
        if self.lastStdout:
            try:
                jsondict = json.loads(trimJsonText(self.lastStdout))
            except ValueError as e:
                print("Couldn't parse json from '%s'" % self.lastStdout)
                return None, 1

            failed_list = self.IdentifyFailingTests(jsondict, nsc)

            # Saves diag stdout into a log file - use append to get multiple runs in 
            # the same file if we're called repeatedly.
            with open(logFile, "a") as f:
                f.seek(0)
                f.write(str(self.lastStdout))

        if self.lastStderr:
            # Saves diag stderr into a log file - use append to get multiple runs in 
            # the same file if we're called repeatedly.
            with open(logFile, "a") as f:
                f.seek(0)
                f.write(str(self.lastStderr))

        return failed_list, self.diagRet

    ################################################################################
    def __RunDcgmiDiag__(self, cmd):
        self.lastCmd = cmd
        self.lastStdout = ''
        self.lastStderr = ''

        nsc = LwidiaSmiChecker.LwidiaSmiJob()
        nsc.start()
        runner = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (self.lastStdout, self.lastStderr) = runner.communicate()
        self.diagRet = runner.returncode
        nsc.m_shutdownFlag.set()
        nsc.join()

        return self.CheckOutput(nsc)
    
    ################################################################################
    def DidIFail(self):
        if self.failed_list:
            for failure in self.failed_list:
                if failure.IsAnError():
                    return True
                    break

        if self.diagRet != 0:
            return True

        return False
    
    ################################################################################
    def RunDcgmiDiag(self, training, config_file, runMode=0):
        oldTrain = self.train
        oldConfig = self.configFile
        oldRunMode = self.runMode

        if config_file:
            self.train = False
            self.configFile = config_file
        elif training:
            self.train = True
            self.configFile = ''
        else:
            self.configFile = ''
            self.train = False

        if runMode:
            self.runMode = runMode

        cmd = self.BuildDcgmiCommand()
        self.failed_list, self.diagRet = self.__RunDcgmiDiag__(cmd)

        self.train = oldTrain
        self.configFile = oldConfig
        self.runMode = oldRunMode

        return self.DidIFail()

    ################################################################################
    def RunWithTraining(self):
        return self.RunDcgmiDiag(True, None)

    ################################################################################
    def RunWithoutTraining(self):
        return self.RunDcgmiDiag(False, None)

    ################################################################################
    def RunAgainstGoldelwalues(self, goldelwaluesFile):
        return self.RunDcgmiDiag(False, goldelwaluesFile)

    ################################################################################
    def RunAtLevel(self, runMode, configFile=None):
        if runMode < 1 or runMode > 3:
            return dcgm_structs.DCGM_ST_BADPARAM

        return self.RunDcgmiDiag(False, configFile, runMode)

    ################################################################################
    def Run(self):
        cmd = self.BuildDcgmiCommand()
        self.failed_list, self.diagRet = self.__RunDcgmiDiag__(cmd)
        return self.DidIFail()

    ################################################################################
    def SetTraining(self, train):
        self.train = train

    ################################################################################
    def SetConfigFile(self, config_file):
        self.configFile = config_file
    
    ################################################################################
    def SetRunMode(self, run_mode):
        self.runMode = run_mode

    ################################################################################
    def PrintFailures(self):
        for failure in self.failed_list:
            print failure.GetFullError()

    ################################################################################
    def PrintLastRunStatus(self):
        print "Ran '%s' and got return code %d" % (self.lastCmd, self.diagRet)
        print "stdout: \n\n%s" % self.lastStdout
        if self.lastStderr:
            print "\nstderr: \n\n%s" % self.lastStderr
        else:
            print "\nNo stderr output"
        self.PrintFailures()

def main():
    dd = DcgmiDiag()
    failed = dd.Run()
    dd.PrintLastRunStatus()

if __name__ == '__main__':
    main()
