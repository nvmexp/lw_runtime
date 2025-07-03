#include <stddef.h>
#include <string>
#include <iostream>
#include <fstream>

#include "TestDiagManager.h"
#include "DcgmDiagManager.h"
#include "DcgmDiagCommon.h"
#include "DcgmDiagResponseWrapper.h"
#include "TestDiagManagerStrings.h"
#include "DcgmError.h"

TestDiagManager::TestDiagManager() {}
TestDiagManager::~TestDiagManager() {}

int TestDiagManager::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    return 0;
}

int TestDiagManager::Run()
{
    int st;
    int Nfailed = 0;

    st = TestPositiveDummyExelwtable();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestPositiveDummyExelwtable FAILED with %d\n", st);
        if(st < 0)
            return -1;
    } else
        printf("TestDiagManager::TestPositiveDummyExelwtable PASSED\n");

    st = TestNegativeDummyExelwtable();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestNegativeDummyExelwtable FAILED with %d\n", st);
        if(st < 0)
            return -1;
    } else
        printf("TestDiagManager::TestNegativeDummyExelwtable PASSED\n");

    st = TestCreateLwvsCommand();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestCreateLwvsCommand FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestCreateLwvsCommand PASSED\n");
     
    st = TestPopulateRunDiag();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestPopulateRunDiag FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestPopulateRunDiag PASSED\n");

    st = TestFillResponseStructure();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestFillResponseStructure FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestFillRespsonseStructure PASSED\n");

    st = TestPerformExternalCommand();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestPerformExternalCommand FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestPerformExternalCommand PASSED\n");

    st = TestErrorsFromLevelOne();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestErrorsFromLevelOne FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestErrorsFromLevelOne PASSED\n");

    if(Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return -1;
    }
    return 0;    
}


int TestDiagManager::Cleanup()
{
    return 0;
}

std::string TestDiagManager::GetTag()
{
    return std::string("diagmanager");
}

void TestDiagManager::CreateDummyScript()
{
    std::ofstream dummyScript;
    dummyScript.open("dummy_script");
    dummyScript << "#!/bin/bash" << std::endl;
    dummyScript << "echo \"Dummy script successfully exelwted with args $1 $2 $3\"" << std::endl;
    dummyScript.close();
    system("chmod 755 dummy_script");
}

void TestDiagManager::RemoveDummyScript()
{
    system("rm -f dummy_script");
}

int TestDiagManager::TestPositiveDummyExelwtable()
{
    
    std::string output;
    dcgmReturn_t result; 

    /*  this is, in practice, really bad... ok if the only public calls made are
     *  the Perform*Execute calls directly 
     */
    DcgmDiagManager * am = new DcgmDiagManager(NULL, NULL);

    CreateDummyScript();
    result = am->PerformDummyTestExelwte(&output);

    delete(am);
    am = 0;

    if (result != DCGM_ST_OK)
        return -1;

    DEBUG_STDERR(output);

    return 0;
}

int TestDiagManager::TestNegativeDummyExelwtable()
{
    std::string output;
    dcgmReturn_t result; 

    DcgmDiagManager * am = new DcgmDiagManager(NULL, NULL);

    RemoveDummyScript();
    result = am->PerformDummyTestExelwte(&output);

    delete(am);
    am = 0;

    if (result == DCGM_ST_OK)
        return -1;

    return 0;
}


// Helper method for TestCreateLwvsCommand
std::string ColwertVectorToCommandString(std::vector<std::string>& v)
{
    std::string out;
    for (unsigned int i = 0; i < v.size(); i++)
    {
        // The CreateLwvsCommand test expects args with spaces to be quoted. We insert the quotes here because 
        // DcgmDiagManager no longer needs to add quotes - the args are directly passed to lwvs via argv
        if (v[i].find(' ') == std::string::npos)
        {
            out += v[i];
        }
        else
        {
            out += "\"" + v[i] + "\"";
        }
        if (i < v.size() - 1)
        {
            out += " ";
        }
    }
    return out;
}

int TestDiagManager::TestCreateLwvsCommand()
{
    dcgmReturn_t                result;
    std::string                 command;
    std::vector<std::string>    cmdArgs;
    dcgmRunDiag_t               drd = {0};
    DcgmDiagManager             am(NULL, NULL);

    std::string lwvsBinPath;
    std::vector<std::string>expected;

    const char *lwvsPathElw = std::getelw("LWVS_BIN_PATH");
    if(lwvsPathElw)
        lwvsBinPath = std::string(lwvsPathElw) + "/lwvs";
    else
        lwvsBinPath = "/usr/share/lwpu-validation-suite/lwvs";

    expected.push_back(lwvsBinPath + " -j -z --specifiedtest long --configless");
    expected.push_back(lwvsBinPath + " -j -z --specifiedtest \"memory bandwidth,sm stress,targeted stress\" --configless");
    expected.push_back(lwvsBinPath + " -j -z --specifiedtest \"memory bandwidth,sm stress,targeted stress\" --parameters \"memory bandwidth.minimum_bandwidth=5000;sm perf.target_stress=8500;targeted stress.test_duration=600\" --configless");

    // When no test names are specified, none isn't valid
    drd.validate = DCGM_POLICY_VALID_NONE;
    result = am.CreateLwvsCommand(cmdArgs, &drd);
    if (result == 0) {
        // Should've failed
        return -1;
    }

    // Check a valid scenario
    cmdArgs.clear();
    drd.validate = DCGM_POLICY_VALID_SV_LONG;
    result = am.CreateLwvsCommand(cmdArgs, &drd);
    if (result == DCGM_ST_OK) {
        command = ColwertVectorToCommandString(cmdArgs);
        if (command != expected[0]) {
            fprintf(stderr, "Expected '%s' but got '%s'\n", expected[0].c_str(), command.c_str());
            return -1;
        }
    }

    // Specify the test names -- validate should get ignored
    cmdArgs.clear();
    snprintf(drd.testNames[0], sizeof(drd.testNames[0]), "memory bandwidth");
    snprintf(drd.testNames[1], sizeof(drd.testNames[1]), "sm stress");
    snprintf(drd.testNames[2], sizeof(drd.testNames[2]), "targeted stress");
    result = am.CreateLwvsCommand(cmdArgs, &drd);
    if (result == DCGM_ST_OK) {
        command = ColwertVectorToCommandString(cmdArgs);
        if (command != expected[1]) {
            fprintf(stderr, "Expected '%s' but got '%s'\n", expected[1].c_str(), command.c_str());
            return -1;
        }
    }

    // Add test parameters
    cmdArgs.clear();
    snprintf(drd.testParms[0], sizeof(drd.testParms[0]), "memory bandwidth.minimum_bandwidth=5000");
    snprintf(drd.testParms[1], sizeof(drd.testParms[1]), "sm perf.target_stress=8500");
    snprintf(drd.testParms[2], sizeof(drd.testParms[2]), "targeted stress.test_duration=600");
    result = am.CreateLwvsCommand(cmdArgs, &drd);
    if (result == DCGM_ST_OK) {
        command = ColwertVectorToCommandString(cmdArgs);
        if (command != expected[2]) {
            fprintf(stderr, "Expected '%s' but got '%s'\n", expected[2].c_str(), command.c_str());
            return -1;
        }
    }

    cmdArgs.clear();
    memset(&drd, 0, sizeof(drd));
    drd.validate = DCGM_POLICY_VALID_SV_LONG;
    drd.flags = DCGM_RUN_FLAGS_TRAIN | DCGM_RUN_FLAGS_FORCE_TRAIN;
    drd.trainingIterations = 3;
    drd.trainingVariance = 2;
    drd.trainingTolerance = 15;
    drd.version = dcgmRunDiag_version4;
    snprintf(drd.goldelwaluesFile, sizeof(drd.goldelwaluesFile), "goodasgold.yml");
    result = am.CreateLwvsCommand(cmdArgs, &drd);
    if (result == DCGM_ST_OK)
    {
        static const char *expected[] =
        { 
            "--train",
            "--force",
            "--training-iterations",
            "3",
            "--training-variance",
            "2",
            "--training-tolerance",
            "15",
            "--golden-values-filename",
            "/tmp/goodasgold.yml"
        };

        // We don't need to test 0 (lwvs path), 1 (-j), 2 (-z), or 3 (--specifiedtest)
        if (cmdArgs.size() < 16)
        {
            fprintf(stderr, "Not enough arguments populated in cmdArgs!\n");
        }

        for (int i = 0; i < 10; i++)
        {
            if (cmdArgs[i+3] != expected[i])
            {
                fprintf(stderr, "Expected '%s' at position %d, but found '%s'\n",
                        expected[i], i+4, cmdArgs[i+4].c_str());
                return -1;
            }
        }
    }

    return result;
}
    
int TestDiagManager::TestPopulateRunDiag()
{
    dcgmReturn_t  result = DCGM_ST_OK;
    dcgmRunDiag_t drd = { 0 };

    drd.version = dcgmRunDiag_version5;

    static const std::string pluginPath("/home/dalinar/unityPlugins");
    static const std::string goldenFilename("/tmp/custom.yml");
    std::string error;

    // Basic test, nothing should get populated
    dcgm_diag_common_populate_run_diag(drd, "1", "", "", "", false, false, "", "", 0, "", pluginPath, true, true, 
                                       10, 7, 12, goldenFilename, 1, true, 3, error);
    if (strlen(drd.testNames[0]) != 0)
    {
        fprintf(stderr, "Expected testNames to be empty but found '%s'\n", drd.testNames[0]);
        return -1;
    }
    else if (strlen(drd.testParms[0]) != 0)
    {
        fprintf(stderr, "Expected testParameters to be empty but found '%s'\n", drd.testParms[0]); 
        return -1;
    }
    else if (drd.flags & DCGM_RUN_FLAGS_STATSONFAIL)
    {
        fprintf(stderr, "Expected statsOnlyOnFail to be unset, but it is set.\n"); 
        return -1;
    }
    else if (drd.flags & DCGM_RUN_FLAGS_VERBOSE)
    {
        fprintf(stderr, "Expected verbose to be unset, but it is set.\n"); 
        return -1;
    }
    else if (strlen(drd.debugLogFile) > 0)
    {
        fprintf(stderr, "The debug log file shouldn't be set, but found '%s'.\n", drd.debugLogFile);
        return -1;
    }
    else if (strlen(drd.statsPath) > 0)
    {
        fprintf(stderr, "The stats path shouldn't be set, but found '%s'.\n", drd.statsPath);
        return -1;
    }
    else if (drd.debugLevel != 0)
    {
        fprintf(stderr, "Debug level should be 0 but was %u.\n", drd.debugLevel);
        return -1;
    }

    if (pluginPath != drd.pluginPath)
    {
        fprintf(stderr, "Plugin path should be '%s', but found '%s'.\n", pluginPath.c_str(), drd.pluginPath);
        return -1;
    }

    if (strcmp(drd.configFileContents, ""))
    {
        fprintf(stderr, "Config file should be empty (no config file), but found %s.\n", drd.configFileContents);
        return -1;
    }

    if (strcmp(drd.throttleMask, ""))
    {
        fprintf(stderr, "The throttle mask should be empty, but found '%s'.\n", drd.throttleMask);
        return -1;
    }

    if (!(drd.flags & DCGM_RUN_FLAGS_TRAIN))
    {
        fprintf(stderr, "Expected train flag to be set but it is unset.\n"); 
        return -1;
    }
    
    if (!(drd.flags & DCGM_RUN_FLAGS_FORCE_TRAIN))
    {
        fprintf(stderr, "Expected force train flag to be set but it is unset.\n"); 
        return -1;
    }

    if (drd.trainingIterations != 10)
    {
        fprintf(stderr, "Expected training iterations to be set to 10, but found %u.\n",
                drd.trainingIterations);
        return -1;
    }

    if (drd.trainingVariance != 7)
    {
        fprintf(stderr, "Expected training variance to be set to 7, but found %u.\n",
                drd.trainingVariance);
        return -1;
    }

    if (drd.trainingTolerance != 12)
    {
        fprintf(stderr, "Expected training tolerance to be set to 12, but found %u.\n",
                drd.trainingTolerance);
        return -1;
    }

    if (goldenFilename != drd.goldelwaluesFile)
    {
        fprintf(stderr, "Expected the golden values file to be set to '%s', but found '%s',\n",
                goldenFilename.c_str(), drd.goldelwaluesFile);
        return -1;
    }

    if (drd.groupId != (dcgmGpuGrp_t)1)
    {
        fprintf(stderr, "Expected groupid to be 1, but found %llu.\n", (unsigned long long) drd.groupId);
        return -1;
    }

    if (!(drd.flags & DCGM_RUN_FLAGS_FAIL_EARLY))
    {
        fprintf(stderr, "Expected fail early flag to be set, but it is unset.\n");
        return -1;
    }

    if (drd.failCheckInterval != 3)
    {
        fprintf(stderr, "Expected failCheckInterval to be 3, but found %u.\n", drd.failCheckInterval);
        return -1;
    }

    // Ensure fail early flag and interval is not set for v4
    drd = dcgmRunDiag_t(); // Need to reset the struct
    drd.version = dcgmRunDiag_version4;
    dcgm_diag_common_populate_run_diag(drd, "1", "", "", "", false, false, "", "", 0, "", pluginPath, true, true, 
                                       10, 7, 12, goldenFilename, 1, true, 3, error);

    if (drd.flags & DCGM_RUN_FLAGS_FAIL_EARLY)
    {
        fprintf(stderr, "Expected fail early flag to be unset, but it is set.\n");
        return -1;
    }

    if (drd.failCheckInterval != 0)
    {
        fprintf(stderr, "Expected failCheckInterval to be 0, but found %u.\n", drd.failCheckInterval);
        return -1;
    }

    const char *tn1[] = { "pcie", "targeted power" };
    const char *tp1[] = { "pcie.test_pinned=false", "pcie.min_bandwidth=25000", "targeted power.temperature_max=82" };
    std::string debugFileName("kaladin");
    std::string statsPath("/home/aimian/");
    std::string throttleMask("HW_SLOWDOWN");
    drd = dcgmRunDiag_t(); // Need to reset the struct
    drd.version = dcgmRunDiag_version4;
    dcgm_diag_common_populate_run_diag(drd, "pcie,targeted power", 
        "pcie.test_pinned=false;pcie.min_bandwidth=25000;targeted power.temperature_max=82", "", "0,1,2", true, 
        true, debugFileName, statsPath, 3, throttleMask, "", false, false, 0, 0, 0, "", 0, false, 3, error);
    for (int i = 0; i < 2; i++)
    {
        if (strcmp(drd.testNames[i], tn1[i]))
        {
            fprintf(stderr, "Test name %d. Expected '%s', found '%s'\n", i, tn1[i], drd.testNames[i]);
            return -1;
        }
    }
    for (int i = 0; i < 3; i++)
    {
        if (strcmp(drd.testParms[i], tp1[i]))
        {
            fprintf(stderr, "Test parameter %d. Expected '%s', found '%s'\n", i, tp1[i], drd.testParms[i]);
            return -1;
        }
    }

    if (strcmp(drd.gpuList, "0,1,2"))
    {
        fprintf(stderr, "Gpu list should be '0,1,2' but found '%s'\n", drd.gpuList);
        return -1;
    } 

    if (!(drd.flags & DCGM_RUN_FLAGS_STATSONFAIL))
    {
        fprintf(stderr, "Expected stats only on failure to be set, but it wasn't.\n");
        return -1;
    }

    if (!(drd.flags & DCGM_RUN_FLAGS_VERBOSE))
    {
        fprintf(stderr, "Expected verbose to be set, but it wasn't.\n");
        return -1;
    }
    
    if (debugFileName != drd.debugLogFile)
    {
        fprintf(stderr, "Expected the debug file name to be '%s', but found '%s'\n", debugFileName.c_str(),
                drd.debugLogFile);
        return -1;
    }

    if (statsPath != drd.statsPath)
    {
        fprintf(stderr, "Expected the stats path to be '%s', but found '%s'.\n", statsPath.c_str(), drd.statsPath);
        return -1;
    }

    if (drd.debugLevel != 3)
    {
        fprintf(stderr, "Debug level should be 3, but found %u.\n", drd.debugLevel);
        return -1;
    }

    // Config file is set, but should be blank since empty contents were given
    if (strcmp(drd.configFileContents, ""))
    {
        fprintf(stderr, "Config file should be empty, but found %s.\n", drd.configFileContents);
        return -1;
    }

    if (throttleMask != drd.throttleMask)
    {
        fprintf(stderr, "Expected throttle mask to be '%s', but found '%s'.\n", 
                throttleMask.c_str(), drd.throttleMask);
        return -1;
    }

    if (drd.pluginPath[0] != '\0')
    {
        fprintf(stderr, "Plugin path should be empty, but found '%s'.\n", drd.pluginPath);
        return -1;
    }

    if (drd.groupId != (dcgmGpuGrp_t)0)
    {
        fprintf(stderr, "Expected groupid to be 0, but found %llu.\n", (unsigned long long) drd.groupId);
        return -1;
    }

    if (drd.flags & DCGM_RUN_FLAGS_TRAIN)
    {
        fprintf(stderr, "Expected train flag to be unset but it is set.\n"); 
        return -1;
    }
    
    if (drd.flags & DCGM_RUN_FLAGS_FORCE_TRAIN)
    {
        fprintf(stderr, "Expected force train flag to be unset but it is set.\n"); 
        return -1;
    }

    if (drd.trainingIterations != 0)
    {
        fprintf(stderr, "Expected training iterations to be set to 0, but found %u.\n",
                drd.trainingIterations);
        return -1;
    }

    if (drd.trainingVariance != 0)
    {
        fprintf(stderr, "Expected training variance to be set to 0, but found %u.\n",
                drd.trainingVariance);
        return -1;
    }

    if (drd.trainingTolerance != 0)
    {
        fprintf(stderr, "Expected training tolerance to be set to 0, but found %u.\n",
                drd.trainingTolerance);
        return -1;
    }

    if (drd.goldelwaluesFile[0] != '\0')
    {
        fprintf(stderr, "Expected the golden values file to be empty, but found '%s',\n",
                drd.goldelwaluesFile);
        return -1;
    }

    if (drd.flags & DCGM_RUN_FLAGS_FAIL_EARLY)
    {
        fprintf(stderr, "Expected fail early flag to be unset, but it is set.\n");
        return -1;
    }

    if (drd.failCheckInterval != 0)
    {
        fprintf(stderr, "Expected failCheckInterval to be 0, but found %u.\n", drd.failCheckInterval);
        return -1;
    }

    // Config file is set to correct value
    dcgm_diag_common_populate_run_diag(drd, "pcie,targeted power", "pcie.test_pinned=false", "configfile", "0", true,
                                       true, debugFileName, statsPath, 3, "", "", false, false, 0, 0, 0, "", 0, 
                                       false, 0, error);
    if (strcmp(drd.configFileContents, "configfile"))
    {
        fprintf(stderr, "Config file should be 'configfile', but found %s.\n", drd.configFileContents);
        return -1;
    }

    // Config file is NOT set when drd.version = v1
    drd = dcgmRunDiag_t(); // Need to reset the struct
    drd.version = dcgmRunDiag_version1;
    dcgm_diag_common_populate_run_diag(drd, "pcie,targeted power", "", "configfile", "0", true, true, debugFileName, 
                                       statsPath, 3, "", "", false, false, 0, 0, 0, "", 0, false, 0, error);
    if (strlen(drd.configFileContents) > 0)
    {
        fprintf(stderr, "Config file shouldn't be set (version 1), but found %s.\n", drd.configFileContents);
        return -1;
    }

    // ThrottleIgnoreReasons is NOT set when drd.version = v2
    drd = dcgmRunDiag_t(); // Need to reset the struct
    drd.version = dcgmRunDiag_version2;
    dcgm_diag_common_populate_run_diag(drd, "pcie,targeted power", "", "configfile", "0", true, true, debugFileName, 
                                       statsPath, 3, throttleMask, "", false, false, 0, 0, 0, "", 0, false, 0, error);
    if (strlen(drd.throttleMask) > 0)
    {
        fprintf(stderr, "Throttle mask shouldn't be set (version 2), but found %s.\n", 
                drd.throttleMask);
        return -1;
    }


    return result;
}

int TestDiagManager::TestErrorsFromLevelOne()
{
    DcgmCacheManager  dcm;
    std::string errorReported("Persistence mode for GPU 1 is lwrrently disabled. LWVS requires persistence mode to be enabled. Enable persistence mode by running (as root): lwpu-smi -i 1 -pm 1");

    std::string rawJsonOutput("{ \"DCGM GPU Diagnostic\" : { \"test_categories\" : [ { \"category\" : \"Deployment\", \"tests\" : [ ");
    rawJsonOutput += "{ \"name\" : \"Blacklist\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"PASS\" } ] }, ";
    rawJsonOutput += "{ \"name\" : \"LWML Library\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"PASS\" } ] }, ";
    rawJsonOutput += "{ \"name\" : \"LWCA Main Library\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"PASS\" } ] }, ";
    rawJsonOutput += "{ \"name\" : \"Permissions and OS-related Blocks\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"PASS\" } ] }, ";
    rawJsonOutput += "{ \"name\" : \"Persistence Mode\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"FAIL\", \"warnings\" : [ \"";
    rawJsonOutput += errorReported;
    rawJsonOutput += "\" ] } ] }, ";
    rawJsonOutput += "{ \"name\" : \"Elwironmental Variables\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"SKIP\" } ] }, ";
    rawJsonOutput += "{ \"name\" : \"Page Retirement\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"SKIP\" } ] }, ";
    rawJsonOutput += "{ \"name\" : \"Graphics Processes\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"SKIP\" } ] }, ";
    rawJsonOutput += "{ \"name\" : \"Inforom\", \"results\" : [ { \"gpu_ids\" : \"0,1,2,3\", \"status\" : \"SKIP\" } ] } ";
    rawJsonOutput += "] } ], \"version\" : \"1.3\" } }";

    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    DcgmDiagManager am(&dcm, NULL);
    dcgmDiagResponse_t response;
    response.version = dcgmDiagResponse_version;
    DcgmDiagResponseWrapper drw;
    drw.SetVersion5(&response);

    dcgmReturn_t ret = am.FillResponseStructure(rawJsonOutput, drw, 0, DCGM_ST_OK);

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Failed to parse JSON, fix the JSON input!");
        return -1;
    }

    // Make sure that we are getting the error message recorded correctly
    for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
    {
        if (i == DCGM_SWTEST_LWDA_RUNTIME_LIBRARY)
        {
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fprintf(stderr, "Test %d should not have run, but result was %d\n",
                        i, response.levelOneResults[i].status);
                return -1;
            }
        }
        else if (i < DCGM_SWTEST_PERSISTENCE_MODE)
        {
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_PASS)
            {
                fprintf(stderr, "Test %d should have passed, but result was %d\n",
                        i, response.levelOneResults[i].status);
                return -1;
            }
        }
        else if (i == DCGM_SWTEST_PERSISTENCE_MODE)
        {
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_FAIL)
            {
                fprintf(stderr, "Test %d should have failed, but result was %d\n",
                        i, response.levelOneResults[i].status);
                return -1;
            }

            if (errorReported != response.levelOneResults[i].error.msg)
            {
                fprintf(stderr, "Expected error '%s' but found '%s'\n", errorReported.c_str(),
                        response.levelOneResults[i].error.msg);
                return -1;
            }
        }
        else
        {
            // After the failed tests, all these others should be skipped
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_SKIP)
            {
                fprintf(stderr, "Test %d should have been skipped, but result was %d\n",
                        i, response.levelOneResults[i].status);
                return -1;
            }
        }
    }

    return DCGM_ST_OK;
}

int TestDiagManager::TestFillResponseStructure()
{
    dcgmReturn_t result = DCGM_ST_OK;
    DcgmCacheManager  dcm;
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    DcgmDiagManager am(&dcm, NULL);
    dcgmDiagResponse_t response;
    response.version = dcgmDiagResponse_version;
    DcgmDiagResponseWrapper drw;
    drw.SetVersion5(&response);

    dcgmReturn_t ret = am.FillResponseStructure(LWVS_1_3_JSON, drw, 0, DCGM_ST_OK);
    if (response.gpuCount != 4)
    {
        fprintf(stderr, "Expected 4 gpus, got %d\n", response.gpuCount);
        return -1;
    }

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected successful parsing, but failed!\n");
        return -1;
    }

    for (int i = 0; i < 4; i++) 
    {
        if (response.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].status != DCGM_DIAG_RESULT_FAIL)
        {
            fprintf(stderr, "Expected the Sm Stress test to have failed with gpu %d, but got %d\n",
                    i, response.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].status);
            return -1;
        }

        // Make sure all of the GPUs except GPU 1 have warnings
        if ((strlen(response.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].error.msg) == 0) &&
            (i != 1)) 
        {
            fprintf(stderr, "Gpu %d should have a warning for Sm Stress, but doesn't\n", i);
            return -1;
        }
        
        if (strlen(response.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].info) == 0)
        {
            fprintf(stderr, "Gpu %d has information reported, but we captured none\n", i);
            return -1;
        }
        
        if (response.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr, "Expected the Diagnostic test to have been skipped with gpu %d, but got %d\n",
                    i, response.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].status);
            return -1;
        }
        
        if (response.perGpuResponses[i].results[DCGM_TARGETED_PERF_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr, "Expected the Targeted Stress test to have been skipped with gpu %d, but got %d\n",
                    i, response.perGpuResponses[i].results[DCGM_TARGETED_PERF_INDEX].status);
            return -1;
        }
        
        if (response.perGpuResponses[i].results[DCGM_TARGETED_POWER_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr, "Expected the Targeted Power test to have been skipped with gpu %d, but got %d\n",
                    i, response.perGpuResponses[i].results[DCGM_TARGETED_POWER_INDEX].status);
            return -1;
        }

        if (i == 0)
        {
            if (response.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status != DCGM_DIAG_RESULT_SKIP)
            {
                fprintf(stderr, "Expected GPU %d's Memory test to have been skipped, but found result %d\n",
                        i, response.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status);
                return -1;
            }
        }
        else
        {
            if (response.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status != DCGM_DIAG_RESULT_PASS)
            {
                fprintf(stderr, "Expected the Memory test to have been passed with gpu %d, but got %d\n",
                        i, response.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status);
                return -1;
            }
        }
    }

    const char *testNames[] =
    {
        "Blacklist",
        "LWML Library",
        "LWCA Main Library",
        "LWCA SDK Library",
        "Permissions",
        "Persistence Mode",
        "Environment",
        "Page Retirement",
        "Graphics Processes",
        "Inforom"
    };

    for (int i = 0; i < DCGM_SWTEST_COUNT; i++)
    {
        if (i == DCGM_SWTEST_LWDA_RUNTIME_LIBRARY)
        {
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fprintf(stderr, "Test %d should not have run, but result was %d\n",
                        i, response.levelOneResults[i].status);
                return -1;
            }
        }
        else if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_PASS)
        {
            fprintf(stderr, "%s test should've passed but got %d\n",
                    testNames[i], response.levelOneResults[i].status);

            return -1;
        }
    }

    std::string trainingMsg("Successfully trained the diagnostic. The golden values file is here \
                             /tmp/golden_values.yml\n");
    std::string trainingJson("{\"DCGM GPU Diagnostic\" : { \"Training Result\" : \"");
    trainingJson += trainingMsg;
    trainingJson += "\", \"version\" : \"1.3\" } }";
    dcgmDiagResponse_t trainingResponse;
    trainingResponse.version = dcgmDiagResponse_version;
    DcgmDiagResponseWrapper trainingDrw;
    trainingDrw.SetVersion5(&trainingResponse);
    
    ret = am.FillResponseStructure(trainingJson, trainingDrw, 0, DCGM_ST_OK);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected successful parsing, but failed!\n");
        return -1;
    }

    if (trainingResponse.trainingMsg != trainingMsg)
    {
        fprintf(stderr, "Expected msg '%s', but found '%s'\n", trainingMsg.c_str(),
                trainingResponse.trainingMsg);
        return -1;
    }

    /* Version 1.7 / Per GPU JSON */
    dcgmDiagResponse_t perGpuResponse;
    perGpuResponse.version = dcgmDiagResponse_version;
    DcgmDiagResponseWrapper perGpuDrw;
    perGpuDrw.SetVersion5(&perGpuResponse);

    fprintf(stdout, "Checking Per GPU (v1.7) JSON parsing");

    ret = am.FillResponseStructure(PER_GPU_JSON, perGpuDrw, 0, DCGM_ST_OK);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected successful parsing, but got %d instead!\n", ret);
        return -1;
    }

    if (perGpuResponse.gpuCount != 4)
    {
        fprintf(stderr, "Expected 4 gpus, got %d\n", perGpuResponse.gpuCount);
        return -1;
    }

    for (int i = 0; i < 4; i++) 
    {
        if (i != 1)
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].status != DCGM_DIAG_RESULT_FAIL)
            {
                fprintf(stderr, "Expected the Sm Stress test to have failed with gpu %d, but got %d\n",
                        i, perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].status);
                return -1;
            }
        }
        else
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].status != DCGM_DIAG_RESULT_PASS)
            {
                fprintf(stderr, "Expected the Sm Stress test to have passed for gpu %d, but got %d\n",
                        i, perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].status);
                return -1;
            }
        }

        // Make sure all of the GPUs except GPU 1 have warnings
        if ((perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].error.msg[0] == '\0') &&
            (i != 1)) 
        {
            fprintf(stderr, "Gpu %d should have a warning for Sm Stress, but doesn't\n", i);
            return -1;
        }
        
        if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].info[0] == '\0')
        {
            fprintf(stderr, "Gpu %d has information reported, but we captured none\n", i);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].error.code != DCGM_FR_STRESS_LEVEL &&
            i != 1)
        {
            fprintf(stderr, "Expected the SM Stress test to have error code %d, but found %d\n",
                    DCGM_FR_STRESS_LEVEL, 
                    perGpuResponse.perGpuResponses[i].results[DCGM_SM_PERF_INDEX].error.code);
            return -1;
        }
        
        if (perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr, "Expected the Diagnostic test to have been skipped with gpu %d, but got %d\n",
                    i, perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].status);
            return -1;
        }
    
        if (perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].error.code != 1)
        {
            // Code 1 is used above because the error reported is from old, saved json and doesn't correspond
            // to an error it's actually possible to get today. I manually added it.
            fprintf(stderr, "Expected the diagnostic test to have error code 1, but found %d\n",
                    perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].error.code);
            return -1;
        }
        
        if (perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_PERF_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr, "Expected the Targeted Stress test to have been skipped with gpu %d, but got %d\n",
                    i, perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_PERF_INDEX].status);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr, "Expected the Memory Bandwidth test to have been skipped with gpu %d, but got %d\n",
                    i, perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].status);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].error.code != DCGM_FR_TEST_DISABLED)
        {
            fprintf(stderr, "Expected the Memory Bandwidth test to have error code %d, but found %d.\n",
                    DCGM_FR_TEST_DISABLED,
                    perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].error.code);
            return -1;
        }
        
        if (perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_POWER_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr, "Expected the Targeted Power test to have been skipped with gpu %d, but got %d\n",
                    i, perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_POWER_INDEX].status);
            return -1;
        }

        if (i == 0)
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status != DCGM_DIAG_RESULT_SKIP)
            {
                fprintf(stderr, "Expected GPU %d's Memory test to have been skipped, but found result %d\n",
                        i, perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status);
                return -1;
            }
        }
        else
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status != DCGM_DIAG_RESULT_PASS)
            {
                fprintf(stderr, "Expected the Memory test to have been passed with gpu %d, but got %d\n",
                        i, perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status);
                return -1;
            }
        }
    }

    if (perGpuResponse.perGpuResponses[0].results[DCGM_MEMORY_INDEX].error.code != DCGM_FR_ECC_DISABLED)
    {
        fprintf(stderr, "Expected to find error code %d but found %d\n", DCGM_FR_ECC_DISABLED, 
                perGpuResponse.perGpuResponses[0].results[DCGM_MEMORY_INDEX].error.code);
        return -1;
    }

    for (int i = 0; i < DCGM_SWTEST_COUNT; i++)
    {
        if (i == DCGM_SWTEST_LWDA_RUNTIME_LIBRARY)
        {
            if (perGpuResponse.levelOneResults[i].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fprintf(stderr, "Test %d should not have run, but result was %d\n",
                        i, perGpuResponse.levelOneResults[i].status);
                return -1;
            }
        }
        else if (perGpuResponse.levelOneResults[i].status != DCGM_DIAG_RESULT_PASS)
        {
            fprintf(stderr, "%s test should've passed but got %d\n",
                    testNames[i], perGpuResponse.levelOneResults[i].status);

            return -1;
        }
    }

    return result;
}

void TestDiagManager::CreateDummyFailScript()
{
    std::ofstream dummyScript;
    dummyScript.open("dummy_script");
    dummyScript << "#!/bin/bash" << std::endl;
    dummyScript << "kaladin_dalinar_roshar_adolin_renarin_shallan_jasnah" << std::endl;
    dummyScript.close();
    system("chmod 755 dummy_script");
}

int TestDiagManager::TestPerformExternalCommand()
{
    std::string output;
    std::vector<std::string> dummyCmds;
    dummyCmds.push_back("dummy"); // added to DcgmDiagManager so we can generate stderr
    dcgmReturn_t result = DCGM_ST_OK;

    DcgmDiagManager am(NULL, NULL);

    // Make the script that will fail
    CreateDummyFailScript();
    am.PerformExternalCommand(dummyCmds, &output);
    if (output.size() == 0)
    {
        fprintf(stderr, "TestPerformExternalCommand should've captured stderr, but failed to do so.\n");
        return -1;
    }

    // Clean up 
    RemoveDummyScript();

    return result;
}
