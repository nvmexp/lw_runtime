#include <stddef.h>
#include <string>
#include <iostream>
#include <fstream>

#include "TestDiag.h"
#include "Diag.h"
#include "PluginStrings.h"
#include "LwvsJsonStrings.h"

TestDiag::TestDiag()
{
}

TestDiag::~TestDiag()
{
}

int TestDiag::Run()
{
    int st;
    int Nfailed = 0;

    st = TestHelperGetPluginName();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiag::TestHelperGetPluginName FAILED with %d.\n", st);
    }
    else
        fprintf(stdout, "TestDiag::TestHelperGetPluginName PASSED.\n");

    st = TestHelperJsonAddResult();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiag::TestHelperJsonAddResult FAILED with %d.\n", st);
    }
    else
        fprintf(stdout, "TestDiag::TestHelperJsonAddResult PASSED.\n");

    st = TestHelperJsonAddBasicTests();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiag::TestHelperJsonAddBasicTests FAILED with %d.\n", st);
    }
    else
        fprintf(stdout, "TestDiag::TestHelperJsonAddBasicTests PASSED.\n");

    st = TestHelperJsonBuildOutput();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiag::TestHelperDisplayAsJson FAILED with %d.\n", st);
    }
    else
        fprintf(stdout, "TestDiag::TestHelperDisplayAsJson PASSED.\n");

    st = TestGetFailureResult();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiag::TestGetFailureResult FAILED with %d.\n", st);
    }
    else
        fprintf(stdout, "TestDiag::TestGetFailureResult PASSED.\n");

    st = TestPopulateGpuList();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiag::TestPopulateGpuList FAILED with %d.\n", st);
    }
    else
        fprintf(stdout, "TestDiag::TestPopulateGpuList PASSED.\n");

    return Nfailed;
}

int TestDiag::TestPopulateGpuList()
{
    std::vector<unsigned int> gpuVec;
    dcgmDiagResponse_t diagResult;

    int  ret = 0;
    Diag d;
    d.InitializeDiagResponse(diagResult);

    // Test initial conditions
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (diagResult.perGpuResponses[i].gpuId != DCGM_MAX_NUM_DEVICES)
        {
            fprintf(stderr, "Gpu Id wasn't initialized correctly for the %dth position", i);
            ret = -1;
        }
    }

    if (diagResult.gpuCount != 0)
    {
        fprintf(stderr, "Gpu count should be 0 but was %d.\n", diagResult.gpuCount);
        ret = -1;
    }

    if (diagResult.version != dcgmDiagResponse_version)
    {
        fprintf(stderr, "Version should be %u but was %u.\n", dcgmDiagResponse_version, diagResult.version);
        ret = -1;
    }

    d.PopulateGpuList(diagResult, gpuVec);
    if (gpuVec.size() != 0)
    {
        fprintf(stderr, "Shouldn't have added gpus to the list from an empty diagResult, but we did.\n");
        ret = -1;
    }

    // The server code will set the gpuId sometimes on GPUs that haven't run, but when it does
    // it'll initialize the tests to not have run, so let's set up ours that way
    for (unsigned int i = 0; i < DCGM_PER_GPU_TEST_COUNT; i++)
    {
        diagResult.perGpuResponses[0].results[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        diagResult.perGpuResponses[1].results[i].status = DCGM_DIAG_RESULT_PASS;
    }
    diagResult.perGpuResponses[0].gpuId = 0;
    diagResult.perGpuResponses[1].gpuId = 1;
    diagResult.gpuCount = 1;

    d.PopulateGpuList(diagResult, gpuVec);

    if (gpuVec.size() != diagResult.gpuCount)
    {
        fprintf(stderr, "Expected %u GPUs in the list, but found %u.\n", diagResult.gpuCount,
                (unsigned)gpuVec.size());
        ret = -1;
    }

    return ret;
}

int TestDiag::TestGetFailureResult()
{
    int ret = 0;
    Diag d;
    dcgmDiagResponse_t diagResult = { 0 };
    diagResult.levelOneTestCount = DCGM_SWTEST_COUNT;

    dcgmReturn_t drt = d.GetFailureResult(diagResult);
    if (drt != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected a zero initialized diagResult to return success, but got '%s'.\n",
                errorString(drt));
        ret = -1;
    }

    diagResult.levelOneResults[DCGM_SWTEST_INFOROM].status = DCGM_DIAG_RESULT_FAIL;
    drt = d.GetFailureResult(diagResult);

    if (drt != DCGM_ST_LWVS_ERROR)
    {
        ret = -1;
        fprintf(stderr, "Expected an error when the inforom test returns a failure, but got success.\n");
    }

    // Clear this failure
    diagResult.levelOneResults[DCGM_SWTEST_INFOROM].status = DCGM_DIAG_RESULT_WARN;
    drt = d.GetFailureResult(diagResult);
    if (drt != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected a diagResult with only a warning to return success, but got '%s'.\n",
                errorString(drt));
        ret = -1;
    }

    diagResult.gpuCount = 1;
    diagResult.perGpuResponses[3].results[DCGM_PCI_INDEX].status = DCGM_DIAG_RESULT_FAIL;
    drt = d.GetFailureResult(diagResult);

    if (drt != DCGM_ST_OK)
    {
        fprintf(stderr, "Shouldn't have picked up a failure outside of our gpu count.\n");
        ret = -1;
    }

    diagResult.gpuCount = 4;
    drt = d.GetFailureResult(diagResult);
    if (drt != DCGM_ST_LWVS_ERROR)
    {
        ret = -1;
        fprintf(stderr, "Expected an error when the PCI test returns a failure, but got success.\n");
    }

    return ret;
}

std::string TestDiag::GetTag()
{
    return std::string("diag");
}

int TestDiag::TestHelperGetPluginName()
{
    int ret = 0;

    Diag d;

    std::string name = d.HelperGetPluginName(DCGM_MEMORY_INDEX);
    if (name != "GPU Memory")
    {
        fprintf(stderr, "Expected 'GPU Memory' to be the name for test index %d, but got '%s'.\n",
                static_cast<int>(DCGM_MEMORY_INDEX), name.c_str());
        ret = -1;
    }

    name = d.HelperGetPluginName(DCGM_DIAGNOSTIC_INDEX);
    if (name != GPUBURN_PLUGIN_NAME)
    {
        fprintf(stderr, "Expected '%s' to be the name for test index %d, but got '%s'.\n",
                GPUBURN_PLUGIN_NAME, static_cast<int>(DCGM_DIAGNOSTIC_INDEX), name.c_str());
        ret = -1;
    }

    name = d.HelperGetPluginName(DCGM_PCI_INDEX);
    if (name != BG_PLUGIN_NAME)
    {
        fprintf(stderr, "Expected '%s' to be the name for test index %d, but got '%s'.\n",
                BG_PLUGIN_NAME, static_cast<int>(DCGM_PCI_INDEX), name.c_str());
        ret = -1;
    }

    name = d.HelperGetPluginName(DCGM_SM_PERF_INDEX);
    if (name != SMPERF_PLUGIN_NAME)
    {
        fprintf(stderr, "Expected '%s' to be the name for test index %d, but got '%s'.\n",
                SMPERF_PLUGIN_NAME, static_cast<int>(DCGM_SM_PERF_INDEX), name.c_str());
        ret = -1;
    }

    name = d.HelperGetPluginName(DCGM_TARGETED_PERF_INDEX);
    if (name != CPERF_PLUGIN_NAME)
    {
        fprintf(stderr, "Expected '%s' to be the name for test index %d, but got '%s'.\n",
                CPERF_PLUGIN_NAME, static_cast<int>(DCGM_TARGETED_PERF_INDEX), name.c_str());
        ret = -1;
    }

    name = d.HelperGetPluginName(DCGM_TARGETED_POWER_INDEX);
    if (name != CP_PLUGIN_NAME)
    {
        fprintf(stderr, "Expected '%s' to be the name for test index %d, but got '%s'.\n",
                CP_PLUGIN_NAME, static_cast<int>(DCGM_TARGETED_POWER_INDEX), name.c_str());
        ret = -1;
    }

    name = d.HelperGetPluginName(DCGM_MEMORY_BANDWIDTH_INDEX);
    if (name != MEMBW_PLUGIN_NAME)
    {
        fprintf(stderr, "Expected '%s' to be the name for test index %d, but got '%s'.\n",
                MEMBW_PLUGIN_NAME, static_cast<int>(DCGM_MEMORY_BANDWIDTH_INDEX), name.c_str());
        ret = -1;
    }

    return ret;
}

int TestDiag::TestHelperJsonAddResult()
{
    int ret = 0;
    Diag d;
    dcgmDiagResponsePerGpu_v2 gpuResponse = { 0 };
    unsigned int testIndex = 0;
    unsigned int gpuIndex = 0;
    Json::Value testEntry;
    size_t i = 0;

    gpuResponse.results[testIndex].status = DCGM_DIAG_RESULT_NOT_RUN;

    if (d.HelperJsonAddResult(gpuResponse, testEntry, gpuIndex, testIndex, i) == true)
    {
        fprintf(stderr, "Expected a false return for adding a result for an empty test, but got true.\n");
        ret = -1;
    }

    if (testEntry[LWVS_RESULTS].empty() == false)
    {
        fprintf(stderr, "Should not have added a result that didn't run, but we did.\n");
        ret = -1;
    }

    gpuResponse.results[testIndex].status = DCGM_DIAG_RESULT_PASS;
    if (d.HelperJsonAddResult(gpuResponse, testEntry, gpuIndex, testIndex, i) == false)
    {
        fprintf(stderr, "Expected a true return for adding a result for a passing test, but got false.\n");
        ret = -1;
    }

    if (testEntry[LWVS_RESULTS].empty() == true)
    {
        fprintf(stderr, "Should have added a result for a passing test, but we didn't.\n");
        ret = -1;
    }
    
    i++;
    gpuResponse.results[testIndex].status = DCGM_DIAG_RESULT_FAIL;
    std::string warning("Stormlight usage of 100 broams exceeded expected usage of 80 broams.\n");
    std::string info("Stormlight was used at a rate of 10 broams / second.");
    snprintf(gpuResponse.results[testIndex].error.msg, sizeof(gpuResponse.results[testIndex].error.msg),
             warning.c_str());
    snprintf(gpuResponse.results[testIndex].info, sizeof(gpuResponse.results[testIndex].info),
             info.c_str());
    if (d.HelperJsonAddResult(gpuResponse, testEntry, gpuIndex, testIndex, i) == false)
    {
        fprintf(stderr, "Expected a true return for adding a result for a failing test, but got false.\n");
        ret = -1;
    }

    if (testEntry[LWVS_RESULTS][1].empty() == true)
    {
        fprintf(stderr, "Should have added a result for a failing test, but we didn't.\n");
        ret = -1;
    }
    else
    {
        Json::Value &result = testEntry[LWVS_RESULTS][1];
        if (result[LWVS_GPU_IDS].asString() != "0")
        {
            fprintf(stderr, "Gpu id should have been '0' but was '%s'.\n",
                    result[LWVS_GPU_IDS].asString().c_str());
            ret = -1;
        }

        if (result[LWVS_STATUS].asString() != "Fail")
        {
            fprintf(stderr, "Expected a status of 'Fail' but was '%s'.\n", result[LWVS_STATUS].asString().c_str());
            ret = -1;
        }

        if (result[LWVS_WARNINGS].asString() != warning)
        {
            fprintf(stderr, "Expected warning to be '%s' but was '%s'.\n", warning.c_str(),
                    result[LWVS_WARNINGS].asString().c_str());
            ret = -1;
        }

        if (result[LWVS_INFO].asString() != info)
        {
            fprintf(stderr, "Expected info to be '%s' but was '%s'.\n", info.c_str(),
                    result[LWVS_INFO].asString().c_str());
            ret = -1;
        }
    }

    return ret;
}

int TestDiag::TestHelperJsonAddBasicTests()
{
    int ret = 0;
    Diag d;
    dcgmDiagResponse_t r = { 0 };
    Json::Value output;
    int categoryIndex = 0;

    r.levelOneTestCount = DCGM_SWTEST_COUNT;
    for (int i = 0; i < DCGM_SWTEST_COUNT; i++)
        r.levelOneResults[i].status = DCGM_DIAG_RESULT_PASS;

    d.HelperJsonAddBasicTests(output, categoryIndex, r);

    if (output[LWVS_NAME].empty() == true)
    {
        fprintf(stderr, "Failed to create the DCGM Diagnostic entry.\n");
        return -1;
    }
    
    Json::Value &diag = output[LWVS_NAME];
    if (diag[LWVS_HEADERS].empty() == true || diag[LWVS_HEADERS].isArray() == false)
    {
        fprintf(stderr, "Failed to create the category array.\n");
        return -1;
    }

    Json::Value &category = diag[LWVS_HEADERS][0];
    if (category[LWVS_HEADER].asString() != "Deployment")
    {
        fprintf(stderr, "Category name should be 'Deployment' but is '%s'.\n",
                category[LWVS_HEADER].asString().c_str());
        ret = -1;
    }

    if (category[LWVS_TESTS].empty() == true || category[LWVS_TESTS].isArray() == false)
    {
        fprintf(stderr, "Failed to create the test array.\n");
        return -1;
    }

    Json::Value &tests = category[LWVS_TESTS];
    if (tests.size() < 10)
    {
        fprintf(stderr, "There should be 10 tests in the Deployment category, but there weren't.\n");
        ret = -1;
    }

    return ret;
}
    
int TestDiag::TestHelperJsonTestEntry(Json::Value &testEntry, int gpuIndex,
        const std::string &status, const std::string &warning)
{
    Json::Value &results = testEntry[LWVS_RESULTS];
    if (results.empty() || results.isArray() == false || static_cast<int>(results.size()) <= gpuIndex)
    {
        fprintf(stderr, "Test entry json isn't formatted as expected.\n");
        return -1;
    }

    Json::Value &result = results[gpuIndex];
    if (result[LWVS_STATUS].empty() == true)
    {
        fprintf(stderr, "Result entry had no status populated.\n");
        return -1;
    }

    if (result[LWVS_STATUS].asString() != status)
    {
        fprintf(stderr, "Test entry had expected status '%s' but found '%s'.\n",
                status.c_str(), result[LWVS_STATUS].asString().c_str());
        return -1;
    }

    if (warning.size() > 0 && result[LWVS_WARNINGS].asString() != warning)
    {
        fprintf(stderr, "Test entry had expected warning '%s' but found '%s'.\n",
                warning.c_str(), results[LWVS_WARNINGS].asString().c_str());
        return -1;
    }

    return 0;
}

int TestDiag::TestHelperJsonBuildOutput()
{
    int ret = 0;
    Diag d;
    dcgmDiagResponse_t r = { 0 };
    Json::Value output;
    int categoryIndex = 0;
    std::vector<unsigned int> gpuIndices;
    const char *warnings[] =
    {
        "Voidspren are dangerous.",
        "Vorin herecy isn't meaningful.",
        "Hoid is only watching out for Hoid.",
        "The Unmade can't be held forever.",
        "The Heralds probably aren't mentally stable.",
        "Taravangian makes Sadaes look innocent.",
        "Amaram is a jerk."
    };

    for (int i = 0; i < DCGM_SWTEST_COUNT; i++)
        r.levelOneResults[i].status = DCGM_DIAG_RESULT_PASS;

    gpuIndices.push_back(0);
    gpuIndices.push_back(1);

    for (int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
            {
                r.perGpuResponses[i].results[j].status = DCGM_DIAG_RESULT_PASS;
            }
        }
        else if (i == 1)
        {
            for (int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
            {
                r.perGpuResponses[i].results[j].status = DCGM_DIAG_RESULT_FAIL;
                snprintf(r.perGpuResponses[i].results[j].error.msg,
                         sizeof(r.perGpuResponses[i].results[j].error.msg),
                         "%s", warnings[j]);
            }
        }
        else
        {
            for (int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
            {
                r.perGpuResponses[i].results[j].status = DCGM_DIAG_RESULT_NOT_RUN;
            }
        }
    }

    d.HelperJsonBuildOutput(output, r, gpuIndices);

    if (output[LWVS_NAME].empty() == true ||
        output[LWVS_NAME][LWVS_HEADERS].empty() == true ||
        output[LWVS_NAME][LWVS_HEADERS].isArray() == false)
    {
        fprintf(stderr, "Json isn't structured correctly.\n");
        return -1;
    }

    Json::Value &headers = output[LWVS_NAME][LWVS_HEADERS];
    if (headers.size() != 4)
    {
        fprintf(stderr, "Didn't get 4 headers like expected.\n");
        return -1;
    }

    Json::Value &integration = headers[1];
    Json::Value &hardware = headers[2];
    Json::Value &stress = headers[3];

    if (integration[LWVS_TESTS].empty() == true || integration[LWVS_TESTS].isArray() == false ||
        hardware[LWVS_TESTS].empty() == true || hardware[LWVS_TESTS].isArray() == false ||
        stress[LWVS_TESTS].empty() == true || stress[LWVS_TESTS].isArray() == false)
    {
        fprintf(stderr, "Plugin categories should have arrays of tests, but don't.\n");
        return -1;
    }

    if (TestHelperJsonTestEntry(integration[LWVS_TESTS][0], 0, "Pass", ""))
    {
        fprintf(stderr, "Failure with PCIe gpu 0.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(integration[LWVS_TESTS][0], 1, "Fail", warnings[DCGM_PCI_INDEX]))
    {
        fprintf(stderr, "Failure with PCIe gpu 1.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(hardware[LWVS_TESTS][0], 0, "Pass", ""))
    {
        fprintf(stderr, "Failure with GPU Memory gpu 0.\n");
        ret = -1;
    }
    
    if (TestHelperJsonTestEntry(hardware[LWVS_TESTS][0], 1, "Fail", warnings[DCGM_MEMORY_INDEX]))
    {
        fprintf(stderr, "Failure with GPU Memory gpu 1.\n");
        ret = -1;
    }
    if (TestHelperJsonTestEntry(hardware[LWVS_TESTS][1], 0, "Pass", ""))
    {
        fprintf(stderr, "Failure with Diagnostic gpu 0.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(hardware[LWVS_TESTS][1], 1, "Fail", warnings[DCGM_DIAGNOSTIC_INDEX]))
    {
        fprintf(stderr, "Failure with Diagnostic gpu 1.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][0], 0, "Pass", ""))
    {
        fprintf(stderr, "Failure with SM Stress gpu 0.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][0], 1, "Fail", warnings[DCGM_SM_PERF_INDEX]))
    {
        fprintf(stderr, "Failure with SM Stress gpu 1.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][1], 0, "Pass", ""))
    {
        fprintf(stderr, "Failure with Targeted Stress gpu 0.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][1], 1, "Fail", warnings[DCGM_TARGETED_PERF_INDEX]))
    {
        fprintf(stderr, "Failure with Targeted Stress gpu 1.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][2], 0, "Pass", ""))
    {
        fprintf(stderr, "Failure with Targeted Power gpu 0.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][2], 1, "Fail", warnings[DCGM_TARGETED_POWER_INDEX]))
    {
        fprintf(stderr, "Failure with Targeted Power gpu 1.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][3], 0, "Pass", ""))
    {
        fprintf(stderr, "Failure with Memory Bandwidth gpu 0.\n");
        ret = -1;
    }

    if (TestHelperJsonTestEntry(stress[LWVS_TESTS][3], 1, "Fail", warnings[DCGM_MEMORY_BANDWIDTH_INDEX]))
    {
        fprintf(stderr, "Failure with Memory Bandwidth gpu 1.\n");
        ret = -1;
    }

    return ret;
}

