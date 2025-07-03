import pydcgm
import dcgm_structs
import dcgm_agent_internal
import dcgm_agent
import dcgm_client_internal
import logger
import test_utils
import dcgm_fields
import apps
import dcgmvalue
import DcgmSystem
from dcgm_structs import dcgmExceptionClass, DCGM_ST_NOT_CONFIGURED
import dcgm_structs_internal
import utils
import option_parser

import string
import time
from ctypes import *
import sys
import os
import pprint
from sys import stdout

def _run_dcgmi_command(args):
    ''' run a command then return (retcode, stdout_lines, stderr_lines) '''
    dcgmi = apps.DcgmiApp(args)
    dcgmi.start()
    retValue = dcgmi.wait()
    dcgmi.validate()
    return retValue, dcgmi.stdout_lines, dcgmi.stderr_lines

def _is_eris_diag_inforom_failure(args, stdout_lines):
    INFOROM_FAILURE_STRING = 'lwmlDeviceValidateInforom for lwml device'
    if not option_parser.options.eris:
        # This is used to skip diag tests. We only want to do that on Eris
        return False
    if len(args) > 0 and args[0] == 'diag' and INFOROM_FAILURE_STRING in stdout_lines:
        return True
    return False

def _assert_valid_dcgmi_results(args, retValue, stdout_lines, stderr_lines):
    assert (len(stdout_lines) > 0), 'No output detected for args "%s"' % ' '.join(args[1:])

    if _is_eris_diag_inforom_failure(args, stdout_lines):
        # If we see inforom corruption, the test should not fail
        test_utils.skip_test('Detected corrupt inforom for diag test')
        return
    
    if retValue != c_ubyte(dcgm_structs.DCGM_ST_OK).value:
        if retValue == c_ubyte(dcgm_structs.DCGM_ST_LWVS_ERROR).value:
            # DCGM_ST_LWVS_ERROR means LWVS ran but returned a bad result. In other words, the arguments were
            # valid and this return code means you had valid arguments.
            return
        logger.error('Valid test - Function returned error code: %s . Args used: "%s"' % (retValue, ' '.join(args[1:]))) 
        logger.error('Stdout:')
        for line in stdout_lines:
            logger.error('\t'+line)
        logger.error('Stderr:')
        for line in stderr_lines:
            logger.error('\t'+line)
        assert False, "See errors above."
    
    errLines = _lines_with_errors(stdout_lines)
    assert len(errLines) == 0, "Found errors in output.  Offending lines: \n%s" % '\n'.join(errLines)
        

def _assert_ilwalid_dcgmi_results(args, retValue, stdout_lines, stderr_lines):
    assert retValue != c_ubyte(dcgm_structs.DCGM_ST_OK).value, \
           'Invalid test - Function returned error code: %s . Args used: "%s"' \
           % (retValue, ', '.join(args[1:]))
            
    assert len(_lines_with_errors(stderr_lines + stdout_lines)) >= 1, \
            'Function did not display error message for args "%s". Returned: %s\nstdout: %s\nstderr: %s' \
            % (' '.join(args[1:]), retValue, '\n'.join(stdout_lines), '\n'.join(stderr_lines))

def _lines_with_errors(lines):
    errorLines = []

    errorStrings = [
        'error',
        'invalid',
        'incorrect',
        'unexpected'
    ]
    exceptionStrings = [
        'lwlink error',
        'flit error',
        'data error',
        'replay error',
        'recovery error',
        'ecc error',
        'xid error'
    ]

    for line in lines:
        lineLower = line.lower()

        for errorString in errorStrings:
            if not errorString in lineLower:
                continue

            wasExcepted = False
            for exceptionString in exceptionStrings:
                if exceptionString in lineLower:
                    wasExcepted = True
                    break
            if wasExcepted:
                continue

            errorLines.append(line)

    return errorLines

def _create_dcgmi_group(groupType=dcgm_structs.DCGM_GROUP_EMPTY):
    ''' Create an empty group and return its group ID '''
    createGroupArgs = ["group", "-c", "test_group"]

    if groupType == dcgm_structs.DCGM_GROUP_DEFAULT:
        createGroupArgs.append('--default')
    elif groupType == dcgm_structs.DCGM_GROUP_DEFAULT_LWSWITCHES:
        createGroupArgs.append('--defaultlwswitches')
    
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(createGroupArgs)
    _assert_valid_dcgmi_results(createGroupArgs, retValue, stdout_lines, stderr_lines)
    
    # dcgmi "group -c" outputs a line like 'Successfully created group "test_group" with a group ID of 2'
    # so we capture the last word as the group ID (it doesn't seem like there's a better way)
    # colwert to int so that if it's not an int, an exception is raised
    return int(stdout_lines[0].strip().split()[-1])

def _test_valid_args(argsList):
    for args in argsList:
        retValue, stdout_lines, stderr_lines = _run_dcgmi_command(args)
        _assert_valid_dcgmi_results(args, retValue, stdout_lines, stderr_lines)
        
def _test_ilwalid_args(argsList):
    for args in argsList:
        retValue, stdout_lines, stderr_lines = _run_dcgmi_command(args)
        _assert_ilwalid_dcgmi_results(args, retValue, stdout_lines, stderr_lines)
        
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgmi_group(handle, gpuIds):
    """
    Test DCGMI group
    """
     
    DCGM_ALL_GPUS = dcgm_structs.DCGM_GROUP_ALL_GPUS
     
    groupId = str(_create_dcgmi_group())
    
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["group", "-l", ""],                            # list groups
            ["group", "-g", groupId, "-i"],                 # get info on created group
            ["group", "-g", groupId, "-a", str(gpuIds[0])], # add gpu to group
            ["group", "-g", groupId, "-r", str(gpuIds[0])], # remove that gpu from the group
            ["group", "-g", groupId, "-a", "gpu:" + str(gpuIds[0])], # add gpu to group with gpu tag
            ["group", "-g", groupId, "-r", "gpu:" + str(gpuIds[0])], # remove that gpu from the group with gpu tag
            ["group", "-d", groupId, ],                     # delete the group
            ["group", "-g", "0", "-i"],                     # Default group can be fetched by ID as long as group IDs start at 0
    ])
         
    nonExistantGroupId = str(int(groupId) + 10)
     
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["group", "-c", "--default"],                       # Can't create a group called --default
            ["group", "-c", "--add"],                           # Can't create a group called --add
            ["group", "-c", "-a"],                              # Can't create a group called -a
            ["group", "-g", nonExistantGroupId, "-a", str(gpuIds[0])], # Can't add to a group that doesn't exist
            ["group", "-g", groupId, "-a", "129"],              # Can't add a GPU that doesn't exist
            ["group", "-g", groupId, "-r", "129"],              # Can't remove a GPU that doesn't exist
            ["group", "-g", nonExistantGroupId, "-r", str(gpuIds[0])],  # Can't remove from a group that does't exist
            ["group", "-g", "0", "-r", "0"],                    # Can't remove from the default group (ID 0)
            ["group", "-g", str(DCGM_ALL_GPUS), "-r", str(gpuIds[0])], # Can't remove from the default group w/ handle
            ["group", "-d", "0"],                               # Can't delete the default group (ID 0)
            ["group", "-d", str(DCGM_ALL_GPUS)],                # Can't delete the default group w/ handle
            ["group", "-d", nonExistantGroupId]                 # Can't delete a group that doesnt exist
    ])
 
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_lwswitches(2)
def test_dcgmi_group_lwswitch(handle, switchIds):

    groupId = str(_create_dcgmi_group(groupType=dcgm_structs.DCGM_GROUP_DEFAULT_LWSWITCHES))

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["group", "-g", groupId, "-i"],                 # get info on created group
            ["group", "-g", groupId, "-r", "lwswitch:%s" % str(switchIds[0])], # remove a switch from the group
            ["group", "-g", groupId, "-a", "lwswitch:%s" % str(switchIds[0])], # add a switch to group
            ["group", "-g", groupId, "-r", "lwswitch:%s" % str(switchIds[1])], # remove a 2nd switch from the group
            ["group", "-g", groupId, "-a", "lwswitch:%s" % str(switchIds[1])], # add a 2nd switch to group
            ["group", "-g", groupId, "-r", "lwswitch:%s,lwswitch:%s" % (str(switchIds[0]), str(switchIds[1]))], # remove both switches at once
            ["group", "-g", groupId, "-a", "lwswitch:%s,lwswitch:%s" % (str(switchIds[0]), str(switchIds[1]))], # add both switches at once
    ])
         
    nonExistantGroupId = str(int(groupId) + 10)
     
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["group", "-g", groupId, "-r", "taco:%s" % str(switchIds[0])], # remove a switch from an invalid entityGroupId
            ["group", "-g", groupId, "-a", "taco:%s" % str(switchIds[0])], # add a switch to an invalid entityGroupId
    ])

#Returns True if searchString oclwrs in any of lines[]. False otherwise
def helper_text_in_output(searchString, lines):
    for line in lines:
        if searchString in line:
            return True
    
    return False

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_dmon_nowatch(handle, gpuIds):
    
    allGpusCsv = ",".join(map(str,gpuIds))
    args = ["dmon", "-e", "150", "-c", "1", "-i", allGpusCsv] 
    noWatchArgs = ["dmon", "-e", "150", "-c", "1", "-i", allGpusCsv, "--no-watch"] # run dmon without watching fields. This should succeed, even though the values will be N/W

    #Make sure we find "N/W" in the output
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(noWatchArgs)
    assert True == helper_text_in_output("N/W", stdout_lines)
    assert retValue == dcgm_structs.DCGM_ST_OK, "%d != DCGM_ST_OK" % retValue

    #Now run it with actual watches and verify we don't get N/W
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(args)
    assert False == helper_text_in_output("N/W", stdout_lines)
    assert retValue == dcgm_structs.DCGM_ST_OK, "%d != DCGM_ST_OK" % retValue
    
    #Run again after watches were established and cleared. Make sure we find "N/W" in the output
    retValue, stdout_lines, stderr_lines = _run_dcgmi_command(noWatchArgs)
    assert True == helper_text_in_output("N/W", stdout_lines)
    assert retValue == dcgm_structs.DCGM_ST_OK, "%d != DCGM_ST_OK" % retValue


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_config(handle, gpuIds):
    """
    Test DCGMI config
    """
    
    # Getting GPU power limits
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    groupObj = dcgmSystem.GetEmptyGroup("test1")
    groupId = str(groupObj.GetId().value)
    assert len(gpuIds) > 0, "Failed to get devices from the node"

    for gpuId in gpuIds:
        groupObj.AddGpu(gpuId)
        gpuAttrib = dcgmSystem.discovery.GetGpuAttributes(gpuId)
        dft_pwr = str(gpuAttrib.powerLimits.defaultPowerLimit)
        max_pwr = str(gpuAttrib.powerLimits.maxPowerLimit)

    ## keep args in this order. Changing it may break the test
    validArgsTestList = [
            ["config", "--get", "-g", groupId],                    # get default group configuration
            ["config", "--get", "-g", "0"],                        # get default group configuration by ID. This will work as long as group IDs start at 0
            ["config", "-g", groupId, "--set", "-P", dft_pwr],     # set default power limit
            ["config", "-g", groupId, "--set", "-P", max_pwr],     # set max power limit
            ["config", "--get", "-g", groupId, "--verbose"],       # get verbose default group configuration
            ["config", "--enforce", "-g", groupId],                # enforce default group configuration
            ["config", "--set", "-c", "0", "-g", "0" ],            # set group configuration on default group by ID
            ["config", "--enforce", "-g", "0" ]                    # enforce group configuration on default group by ID
    ]

    #Config management only works when the host engine is running as root
    if utils.is_root():
        _test_valid_args(validArgsTestList)
    else:
        _test_ilwalid_args(validArgsTestList)

    del(groupObj)
    
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["config", "--get", "-g", "9999"],                 # Can't get config of group that doesn't exist
            ["config", "--get", "-g", "9999", "--verbose"],    # Can't get config of group that doesn't exist
            ["config", "--set", ""],                        # Can't set group configuration to nothing
            ["config", "--set", "-c", "5"],                 # Can't set an invalid compute mode
            ["config", "--enforce", "-g", "9999"]           # Can't enforce a configuration of group that doesn't exist
    ])
 
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus() #Use injected GPUs for policy so this doesn't fail on VdChip and Lwdqro
def test_dcgmi_policy(handle, gpuIds):
    """
     Test DCGMI policy
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()
    groupObj = dcgmSystem.GetGroupWithGpuIds("testgroup", gpuIds)
    groupIdStr = str(groupObj.GetId().value)

    DCGM_ALL_GPUS = dcgm_structs.DCGM_GROUP_ALL_GPUS
     
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["policy", "--get", "", "-g", groupIdStr],                 # get default group policy
            ["policy", "--get", "-g", "0"],                            # get default group policy by ID. this will fail if groupIds ever start from > 0
            ["policy", "--get", "--verbose", "-g", groupIdStr],        # get verbose default group policy
            ["policy", "--set", "0,0", "-p", "-e", "-g", groupIdStr],  # set default group policy 
            ["policy", "--set", "1,2", "-p", "-e", "-g", groupIdStr],  # set default group policy 
            ["policy", "--set", "1,0", "-x", "-g", groupIdStr],        # set monitoring of xid errors
            ["policy", "--set", "1,0", "-x", "-n", "-g", groupIdStr],  # set monitoring of xid errors and lwlink errors
            #["policy", "--reg", ""]                                   # register default group policy (causes timeout)     
    ])
     
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["policy", "--get", "-g", "1000"],               # Can't get policy of group that doesn't exist
            ["policy", "--get", "-g", "1000", "--verbose"],  # Can't get policy of group that doesn't exist
            ["policy", "--set", "-p"],                       # Can't set group policy w/ no action/validaion
            ["policy", "--set", "0,0"],                      # Can't set group policy w/ no watches
            ["policy", "--set", "0,0", "-p", "-g", "1000" ], # Can't set group policy on group that doesn't exist
            ["policy", "--reg", "-g", "1000"]                # Can't register a policy of group that doesn't exist
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgmi_health(handle, gpuIds):
    """
      Test DCGMI Health
    """
     
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["health", "--fetch", ""],                  # get default group health
            ["health", "--set", "pmit"],                # set group health
            ["health", "--clear", ""]                   # clear group health watches
    ])
                
    #Create group for testing
    groupId = str(_create_dcgmi_group())
    nonExistantGroupId = str(int(groupId) + 10)
      
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["health", "--fetch", "-g", nonExistantGroupId],    # Can't get health of group that doesn't exist
            ["health", "--set", "a", "-g", nonExistantGroupId], # Can't get health of group that doesn't exist
            ["health", "--set", "pp"],                          # Can't set health of group with multiple of same tag
            ["health", "--get", "ap"],                          # Can't set health to all plus another tag 
            ["health", "--set", ""],                            # Can't set group health w/ no arguments
            ["health", "--check", "-g", nonExistantGroupId],    # Can't check health of group that doesn't exist
            ["health", "--check", "-g", groupId]                # Can't check health of group that has no watches enabled
    ])
 
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgmi_discovery(handle, gpuIds):
    """
    Test DCGMI discovery 
    """
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["discovery", "--list", ""],                  # list gpus
            ["discovery", "--info", "aptc"],              # check group info
            ["discovery", "--info", "aptc", "--verbose"]  # checl group info verbose
    ])
    
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["discovery", "--info", "a", "-g", "2"],               # Cant check info on group that doesn't exist
            ["discovery", "--info", "a", "--gpuid", "123"]        # Cant check info on gpu that doesn't exist
    ])
     
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_diag(handle, gpuIds):
    """
    Test DCGMI diagnostics 
    """
    allGpusCsv = ",".join(map(str,gpuIds))
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["diag", "--run", "1", "-i", allGpusCsv], # run diagnostic other settings lwrrently run for too long
           ["diag", "--run", "1", "-i", allGpusCsv, "--parameters", "diagnostic.test_duration=30", "--fail-early"], # verifies --fail-early option
           ["diag", "--run", "1", "-i", allGpusCsv, "--parameters", "diagnostic.test_duration=30", "--fail-early", "--check-interval", "3"], # verifies --check-interval option
           ["diag", "--run", "1", "--throttle-mask", "HW_SLOWDOWN"], # verifies that --throttle-mask with HW_SLOWDOWN reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "SW_THERMAL"], # verifies that --throttle-mask with SW_THERMAL reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "HW_THERMAL"], # verifies that --throttle-mask with HW_THERMAL reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "HW_POWER_BRAKE"], # verifies that --throttle-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "HW_SLOWDOWN,SW_THERMAL,HW_POWER_BRAKE"], # verifies that --throttle-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "SW_THERMAL,HW_THERMAL,HW_SLOWDOWN"], # verifies that --throttle-mask with HW_POWER_BRAKE reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "40"], # verifies that --throttle-mask with HW_SLOWDOWN (8) and SW_THERMAL (32) reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "96"], # verifies that --throttle-mask with SW_THERMAL (32) and HW_THERMAL (64) reason to be ignored
           ["diag", "--run", "1", "--throttle-mask", "232"], # verifies that --throttle-mask with ALL reasons to be ignored
           ["diag", "--run", "1", "--plugin-path", "/usr/share/lwpu-validation-suite/plugins"], # verifies --plugin-path actually sees the plugins on a specified path

    ])
      
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["diag", "--run", "-g", "2"],           # Can't run on group that doesn't exist
            ["diag", "--run", "4"],                 # Can't run with a test number that doesn't exist
            ["diag", "--run", "\"roshar stress\""], # Can't run a non-existent test name
            ["diag", "--run", "3", "--parameters", "dianarstic.test_duration=40"],
            ["diag", "--run", "3", "--parameters", "diagnostic.test_durration=40"],
            ["diag", "--run", "3", "--parameters", "pcie.h2d_d2h_singgle_pinned.iterations=4000"],
            ["diag", "--run", "3", "--parameters", "pcie.h2d_d2h_single_pinned.itterations=4000"],
            ["diag", "--run", "3", "--parameters", "bob.tom=maybe"],
            ["diag", "--run", "3", "--parameters", "truck=slow"],
            ["diag", "--run", "3", "--parameters", "now this is a story all about how"],
            ["diag", "--run", "3", "--parameters", "my=life=got=flipped=turned=upside=down"],
            ["diag", "--run", "3", "--parameters", "and.i'd.like.to.take.a=minute=just.sit=right=there"],
            ["diag", "--run", "3", "--parameters", "i'll tell you=how.I.became the=prince of .a town called"],
            ["diag", "--run", "3", "--parameters", "Bel-Air"],
            ["diag", "--run", "sm stress", "--train"],
            ["diag", "--run", "1", "--force"],
            ["diag", "--run", "2", "--training-variance", "10"],
            ["diag", "--run", "1", "-i", allGpusCsv, "--parameters", "diagnostic.test_duration=30", "--fail-early 10"], # verifies --fail-early does not accept parameters
            ["diag", "--run", "1", "--parameters", "diagnostic.test_duration=30", "--fail-early", "--check-interval -1"], # no negative numbers allowed
            ["diag", "--run", "1", "--parameters", "diagnostic.test_duration=30", "--fail-early", "--check-interval 350"], # no numbers > 300 allowed
            ["diag", "--run", "1", "--parameters", "diagnostic.test_duration=30", "--check-interval 10"], # requires --fail-early parameter
            ["diag", "--run", "1", "--throttle-mask", "HW_ZSLOWDOWN"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask", "SW_TRHERMAL"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask", "HWT_THERMAL"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask", "HW_POWER_OUTBRAKE"], # verifies that --throttle-mask incorrect reason does not work
            ["diag", "--run", "1", "--throttle-mask -10"], # verifies that --throttle-mask does not accept incorrect values for any reasons to be ignored
            ["diag", "--run", "1", "-i", "0-1-2-3-4"], # Make sure -i is a comma-separated list of integers
            ["diag", "--run", "1", "-i", "roshar"], # Make sure -i is a comma-separated list of integers
            ["diag", "--run", "1", "-i", "a,b,c,d,e,f"], # Make sure -i is a comma-separated list of integers
            # ["diag", "--run", "1", "--plugin-path", "/usr/share/lwpu-validation-suite/unplugins"], # verifies --plugin-path fails if the plugins path is not specified correctly

    ])
  
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_as_root()
def test_dcgmi_stats(handle, gpuIds):
    """
     Test DCGMI Stats
    """
   
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["stats", "--enable"],                  # Enable watches
            #["stats", "--pid", "100"],                  # check pid
            #["stats", "--pid", "100", "--verbose"],     # check pid verbose (run test process and enable these if wanted)

            ["stats", "--jstart", "1"],             #start a job with Job ID 1
            ["stats", "--jstop", "1"],              #Stop the job
            ["stats", "--job", "1"],                #Print stats for the job
            ["stats", "--jremove", "1"],            #Remove the job the job
            ["stats", "--jremoveall"],              #Remove all jobs
            ["stats", "--disable"],                 #disable watches

            ["stats", "--jstart", "1"],             #start another job with Job ID 1. This should work due to jremove above. Also, setup the jstart failure below
    ])

    #Create group for testing
    groupId = str(_create_dcgmi_group())
    nonExistantGroupId = str(int(groupId) + 10)
      
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["stats", "--pid", "100", "-g", groupId],            # Can't view stats with out watches enabled
            ["stats", "enable", "-g", nonExistantGroupId],       # Can't enable watches on group that doesn't exist
            ["stats", "disable", "-g", nonExistantGroupId],      # Can't disable watches on group that doesn't exist
            ["stats", "--jstart", "1"],                          # Cant start the job with a job ID which is being used
            ["stats", "--jstop", "3"],                           # Stop an invalid job id
            ["stats", "--jremove", "3"],                         # Remove an invalid job id
            ["stats", "--job", "3"]                              # Get stats for an invalid job id
    ])
        
@test_utils.run_with_standalone_host_engine(20, ["--port", "5545"])
@test_utils.run_with_initialized_client("127.0.0.1:5545")
def test_dcgmi_port(handle):
    """
    Test DCGMI port - does dcgmi group testing using port 5545
    """
    
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
            ["group", "--host", "localhost:5545", "-l", ""],      # list groups
    ])
    
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["group", "--host", "localhost:5545", "-c", "--default"],      # Can't create a group called --default
    ])

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
def test_dcgmi_field_groups(handle):
    """
    Test DCGMI field groups - test the dcgmi commands under "fieldgroup"
    """

    _test_valid_args([
        ["fieldgroup", "-l"],
        ["fieldgroup", "-i", "-g", "1"],                    # show internal field group
        ["fieldgroup", "-c", "my_group", "-f", "1,2,3"]     # Add a field group
    ])

    _test_ilwalid_args([
        ["introspect", "-d", "-g", "1"],                    # Delete internal group. Bad
        ["introspect", "-i", "-g", "100000"],               # Info for invalid group
    ])

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
def test_dcgmi_introspect(handle):
    """
    Test DCGMI introspection - test the dcgmi commands under "introspection"
    """
    
    _test_valid_args([
        ["introspect", "--enable"],
        ["introspect", "--show", "--hostengine"],           # show hostengine
        ["introspect", "-s", "-H"],                         # short form
        ["introspect", "--show", "--all-fields"],           # show all fields
        ["introspect", "--show", "-F"],                     # short form
        ["introspect", "--show", "-f", "all"],              # show all field groups
        ["introspect", "--show", "--field-group", "all"],   # long form
        ["introspect", "--show", "-H", "-F", "-f", "all"],  # show everything
        ["introspect", "--disable"],
    ])
    
    _test_ilwalid_args([
        ["introspect", "--show", "-H"],         # all "show" commands should fail since introspection is disabled
        ["introspect", "--show", "-F"],
        ["introspect", "--show", "-f", "all"],
    ])
    
    # turn on introspection again to test more invalid args
    _test_valid_args([["introspect", "--enable"]])
    
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
def test_dcgmi_introspect_enable_disable(handle):
    """
    Test that the dcgmi commands for enabling/disabling introspection actually 
    do as they say
    """
    _run_dcgmi_command(["introspect", "--enable"])
    
    dcgmHandle = pydcgm.DcgmHandle(handle)
    system = pydcgm.DcgmSystem(dcgmHandle)
    
    # throws exception if it's not enabled
    mem = system.introspect.memory.GetForHostengine()
    
    _run_dcgmi_command(["introspect", "--disable"])
    with test_utils.assert_raises(dcgmExceptionClass(DCGM_ST_NOT_CONFIGURED)):
        mem = system.introspect.memory.GetForHostengine()
    
@test_utils.run_with_standalone_host_engine(320)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgmi_lwlink(handle, gpuIds):
    """
    Test dcgmi to display lwlink error counts
    """
    
    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["lwlink", "-e", "-g", str(gpuIds[0])],  # run the working lwlink command for gpuId[0]
           ["lwlink", "-s"]                         # Link status should work without parameters
    ])
      
    _test_ilwalid_args([
           ["lwlink","-e"],                         # -e option requires -g option
           ["lwlink","-e -s"]                       # -e and -s are mutually exclusive
    ])


def helper_make_switch_string(switchId):
    return "lwswitch:" + str(switchId)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_with_injection_lwswitches(2)
def test_dcgmi_dmon(handle, gpuIds, switchIds):
    """
    Test dcgmi to display dmon values
    """
    gpuGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT))
    switchGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT_LWSWITCHES))

    logger.info("Injected switch IDs:" + str(switchIds))

    # Creates a comma separated list of gpus
    allGpusCsv = ",".join(map(str,gpuIds))
    #Same for switches but predicate each one with lwswitch
    allSwitchesCsv = ",".join(map(helper_make_switch_string,switchIds))

    switchFieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P00

    #Inject a value for a field for each switch so we can retrieve it
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = switchFieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()-5) * 1000000.0) #5 seconds ago
    field.value.i64 = 0
    for switchId in switchIds:
        ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH, 
                                                             switchId, field)

    _test_valid_args([
        ["dmon", "-e", "150,155","-c","1"],                          # run the dmon for default gpu group.
        ["dmon", "-e", "150,155","-c","1","-g",gpuGroupId],          # run the dmon for a specified gpu group
        ["dmon", "-e", "150,155","-c","1","-g",'all_gpus'],          # run the dmon for a specified group
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",'all_lwswitches'], # run the dmon for a specified group - Reenable after DCGM-413 is fixed
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",switchGroupId], # run the dmon for a specified group
        ["dmon", "-e", "150,155","-c","1","-d","2000"],              # run the dmon for delay mentioned and default gpu group. 
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i",allGpusCsv], # run the dmon for devices mentioned and mentioned delay.
        ["dmon", "-e", str(switchFieldId),"-c","1","-d","2000","-i",allSwitchesCsv] # run the dmon for devices mentioned and mentioned delay.
    ])

    #Run tests that take a gpuId as an argument
    for gpu in gpuIds:
        _test_valid_args([
               ["dmon", "-e", "150","-c","1","-i",str(gpu)],                    # run the dmon for one gpu.
               ["dmon", "-e", "150","-c","1","-i",'gpu:'+str(gpu)],             # run the dmon for one gpu, tagged as gpu:.
               ["dmon", "-e", "150","-c","1","-i",str(gpu)],                    # run the dmon for mentioned devices and count value.
               ["dmon", "-e", "150,155","-c","1","-i",str(gpu)]                 # run the dmon for devices mentioned, default delay and field values that are provided.
        ])
    
    #Run tests that take a lwSwitch as an argument
    for switchId in switchIds:
        _test_valid_args([
               ["dmon", "-e", str(switchFieldId),"-c","1","-i",'lwswitch:'+str(switchId)], # run the dmon for one lwswitch, tagged as lwswitch:.
        ])

    _test_ilwalid_args([
           ["dmon","-c","1"],                                                       # run without required fields.
           ["dmon", "-e", "-150","-c","1","-i","1"],                                # run with invalid field id.
           ["dmon", "-e", "150","-c","1","-i","-2"],                                # run with invalid gpu id.
           ["dmon", "-e", "150","-c","1","-i","gpu:999"],                           # run with invalid gpu id.
           ["dmon", "-e", "150","-c","1","-g","999"],                               # run with invalid group id.
           ["dmon", "-i", "%s,%d" % (allGpusCsv, len(gpuIds)), "-e", "150", "-c", "1"],     # run with invalid number of devices.
           ["dmon", "-e", "150","f","0","-c","1","-i","0,1,765"],                   # run with invalid device id (non existing id).
           ["dmon", "-e", "150","-c","-1","-i","1"],                                # run with invalid count value.
           ["dmon", "-e", "150","-c","1","-i","1","-d","-1"],                       # run with invalid delay (negative value).
           ["dmon", "-f", "-9","-c","1","-i","1","-d","10000"],                     # run with invalid field Id.
           ["dmon", "-f", "150","-c", "1", "-i","0", "-d", "99" ]                   # run with invalid delay value.
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.run_with_injection_lwswitches(2)
def test_dcgmi_lwlink_lwswitches(handle, gpuIds, switchIds):
    """
    Test dcgmi to display dmon values
    """
    gpuGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT))
    switchGroupId = str(_create_dcgmi_group(dcgm_structs.DCGM_GROUP_DEFAULT_LWSWITCHES))

    logger.info("Injected switch IDs:" + str(switchIds))

    _test_valid_args([
           ["lwlink", "-s"]                       # Link status should work without parameters
    ])

    # Creates a comma separated list of gpus
    allGpusCsv = ",".join(map(str,gpuIds))
    #Same for switches but predicate each one with lwswitch
    allSwitchesCsv = ",".join(map(helper_make_switch_string,switchIds))

    switchFieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P00

    #Inject a value for a field for each switch so we can retrieve it
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = switchFieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()-5) * 1000000.0) #5 seconds ago
    field.value.i64 = 0
    for switchId in switchIds:
        ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH, 
                                                             switchId, field)

    _test_valid_args([
        ["dmon", "-e", "150,155","-c","1"],                          # run the dmon for default gpu group.
        ["dmon", "-e", "150,155","-c","1","-g",gpuGroupId],          # run the dmon for a specified gpu group
        ["dmon", "-e", "150,155","-c","1","-g",'all_gpus'],          # run the dmon for a specified group
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",'all_lwswitches'], # run the dmon for a specified group - Reenable after DCGM-413 is fixed
        ["dmon", "-e", str(switchFieldId),"-c","1","-g",switchGroupId], # run the dmon for a specified group
        ["dmon", "-e", "150,155","-c","1","-d","2000"],              # run the dmon for delay mentioned and default gpu group. 
        ["dmon", "-e", "150,155","-c","1","-d","2000","-i",allGpusCsv], # run the dmon for devices mentioned and mentioned delay.
        ["dmon", "-e", str(switchFieldId),"-c","1","-d","2000","-i",allSwitchesCsv] # run the dmon for devices mentioned and mentioned delay.
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgmi_modules(handle, gpuIds):
    """
    Test DCGMI modules 
    """

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["modules", "--list"],
           ["modules", "--blacklist", "5"],
           ["modules", "--blacklist", "policy"],
    ])
      
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["modules", "--list", "4"],
            ["modules", "--blacklist", "20"],
            ["modules", "--blacklist", "notamodule"],
    ])

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_dcgmi_profile(handle, gpuIds):
    """
    Test DCGMI "profile" subcommand
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetGroupWithGpuIds('mygroup', gpuIds)

    # Creates a comma separated list of gpus
    allGpusCsv = ",".join(map(str,gpuIds))

    #See if these GPUs even support profiling. This will bail out for non-Tesla or Pascal or older SKUs
    try:
        supportedMetrics = dcgmGroup.profiling.GetSupportedMetricGroups()
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_PROFILING_NOT_SUPPORTED) as e:
        test_utils.skip_test("Profiling is not supported for gpuIds %s" % str(gpuIds))
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED) as e:
        test_utils.skip_test("The profiling module could not be loaded")

    ## keep args in this order. Changing it may break the test
    _test_valid_args([
           ["profile", "--list", "-i", allGpusCsv],
           ["profile", "--list", "-g", str(dcgmGroup.GetId().value)],
           ["profile", "--pause"], #Pause followed by resume
           ["profile", "--resume"],
           ["profile", "--pause"], #Double pause and double resume should be fine
           ["profile", "--pause"],
           ["profile", "--resume"],
           ["profile", "--resume"],
    ])
      
    ## keep args in this order. Changing it may break the test
    _test_ilwalid_args([
            ["profile", "--list", "--pause", "--resume"], #mutually exclusive flags
            ["profile", "--pause", "--resume"], #mutually exclusive flags
            ["profile", "--list", "-i", "999"], #Invalid gpuID
            ["profile", "--list", "-i", allGpusCsv + ",taco"], #Invalid gpu at end
            ["profile", "--list", "-g", "999"], #Invalid group
    ])
