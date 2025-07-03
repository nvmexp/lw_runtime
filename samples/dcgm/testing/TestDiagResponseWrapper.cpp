#include <iostream>
#include <string.h>

#include "TestDiagResponseWrapper.h"
#include "DcgmDiagResponseWrapper.h"
#include "LwvsJsonStrings.h"

TestDiagResponseWrapper::TestDiagResponseWrapper()
{
}

TestDiagResponseWrapper::~TestDiagResponseWrapper()
{
}

int TestDiagResponseWrapper::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    return 0;
}

int TestDiagResponseWrapper::Cleanup()
{
    return 0;
}

std::string TestDiagResponseWrapper::GetTag()
{
    return std::string("diagresponsewrapper");
}

int TestDiagResponseWrapper::Run()
{
    int nFailed = 0;

    int st = TestInitializeDiagResponse();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestInitializeDiagResponse FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestInitializeDiagResponse PASSED\n");

    st = TestSetPerGpuResponseState();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestSetPerGpuResponseState FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestSetPerGpuResponseState PASSED\n");

    st = TestAddPerGpuMessage();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestAddPerGpuMessage FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestAddPerGpuMessage PASSED\n");

    st = TestSetGpuIndex();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestSetGpuIndex FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestSetGpuIndex PASSED\n");

    st = TestGetBasicTestResultIndex();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestGetBasicTestResultIndex FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestGetBasicTestResultIndex PASSED\n");

    st = TestRecordSystemError();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestRecordSystemError FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestRecordSystemError PASSED\n");

    st = TestAddErrorDetail();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestAddErrorDetail FAILED with %d\n", st);
    }
    else
    {
        printf("TestDiagManager::TestAddErrorDetail PASSED\n");
    }

    return st;
}

int TestDiagResponseWrapper::TestInitializeDiagResponse()
{
    DcgmDiagResponseWrapper r1;
    DcgmDiagResponseWrapper r2;
    DcgmDiagResponseWrapper r3;
    dcgmDiagResponse_v3     rv3 = {0};
    dcgmDiagResponse_v4     rv4 = {0};
    dcgmDiagResponse_v5     rv5 = {0};

    // These should be no ops, but make sure there's no crash
    r1.InitializeResponseStruct(6);
    r2.InitializeResponseStruct(6);
    r3.InitializeResponseStruct(6);

    // Set versions and make sure the state is valid
    r1.SetVersion3(&rv3);
    r2.SetVersion4(&rv4);
    r3.SetVersion5(&rv5);
    
    r1.InitializeResponseStruct(6);
    r2.InitializeResponseStruct(4);
    r3.InitializeResponseStruct(8);

    if (rv3.gpuCount != 6)
    {
        fprintf(stderr, "Gpu count was set to %d, but it should've been 6", rv3.gpuCount);
        return -1;
    }

    if (rv4.gpuCount != 4)
    {
        fprintf(stderr, "Gpu count was set to %d, but it should've been 4", rv4.gpuCount);
        return -1;
    }

    if (rv5.gpuCount != 8)
    {
        fprintf(stderr, "Gpu count was set to %d, but it should've been 8", rv5.gpuCount);
        return -1;
    }

    if (rv3.version != dcgmDiagResponse_version3)
    {
        fprintf(stderr, "Diag Response version3 wasn't set correctly");
        return -1;
    }

    if (rv4.version != dcgmDiagResponse_version4)
    {
        fprintf(stderr, "Diag Response version wasn't set correctly");
        return -1;
    }

    if (rv5.version != dcgmDiagResponse_version5)
    {
        fprintf(stderr, "Diag Response version wasn't set correctly");
        return -1;
    }

    for (unsigned int i = 0; i < 8; i++)
    {
        for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
        {
            if (i < 6)
            {
                if (rv3.perGpuResponses[i].results[j].status != DCGM_DIAG_RESULT_NOT_RUN)
                {
                    fprintf(stderr, "Initial test status wasn't set correctly");
                    return -1;
                }
            }

            if (i < 4)
            {
                if (rv4.perGpuResponses[i].results[j].status != DCGM_DIAG_RESULT_NOT_RUN)
                {
                    fprintf(stderr, "Initial test status wasn't set correctly");
                    return -1;
                }
            }
                
            if (rv5.perGpuResponses[i].results[j].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fprintf(stderr, "Initial test status wasn't set correctly");
                return -1;
            }
        }
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestSetPerGpuResponseState()
{
    DcgmDiagResponseWrapper r1;
    DcgmDiagResponseWrapper r2;
    DcgmDiagResponseWrapper r3;
    dcgmDiagResponse_v3     rv3 = {0};
    dcgmDiagResponse_v4     rv4 = {0};
    dcgmDiagResponse_v5     rv5 = {0};

    r1.SetVersion3(&rv3);
    r2.SetVersion4(&rv4);
    r3.SetVersion5(&rv5);

    r1.InitializeResponseStruct(8);
    r2.InitializeResponseStruct(8);
    r3.InitializeResponseStruct(8);

    // Make sure we avoid a crash
    r1.SetPerGpuResponseState(28, DCGM_DIAG_RESULT_PASS, 0);

    r1.SetPerGpuResponseState(0, DCGM_DIAG_RESULT_PASS, 0);
    r2.SetPerGpuResponseState(0, DCGM_DIAG_RESULT_FAIL, 0);
    r3.SetPerGpuResponseState(0, DCGM_DIAG_RESULT_PASS, 0);

    if (rv3.perGpuResponses[0].results[0].status != DCGM_DIAG_RESULT_PASS)
    {
        fprintf(stderr, "GPU 0 test 0 should be PASS, but is %d\n", rv3.perGpuResponses[0].results[0].status);
        return -1;
    }
    
    if (rv4.perGpuResponses[0].results[0].status != DCGM_DIAG_RESULT_FAIL)
    {
        fprintf(stderr, "GPU 0 test 0 should be PASS, but is %d\n", rv4.perGpuResponses[0].results[0].status);
        return -1;
    }

    if (rv5.perGpuResponses[0].results[0].status != DCGM_DIAG_RESULT_PASS)
    {
        fprintf(stderr, "GPU 0 test 0 should be PASS, but is %d\n", rv5.perGpuResponses[0].results[0].status);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestAddPerGpuMessage()
{
    DcgmDiagResponseWrapper r1;
    DcgmDiagResponseWrapper r2;
    DcgmDiagResponseWrapper r3;
    dcgmDiagResponse_v3     rv3 = {0};
    dcgmDiagResponse_v4     rv4 = {0};
    dcgmDiagResponse_v5     rv5 = {0};

    r1.SetVersion3(&rv3);
    r2.SetVersion4(&rv4);
    r3.SetVersion5(&rv5);

    r1.InitializeResponseStruct(8);
    r2.InitializeResponseStruct(8);
    r3.InitializeResponseStruct(8);

    static const std::string warn("That 3rd ideal can be tricky");
    static const std::string info("There are 5 ideals of radiance");

    r1.AddPerGpuMessage(0, warn, 0, true);
    r2.AddPerGpuMessage(0, info, 0, false);
    r3.AddPerGpuMessage(0, warn, 0, true);

    if (warn != rv3.perGpuResponses[0].results[0].warning)
    {
        fprintf(stderr, "GPU 0 test 0 warning should be '%s', but found '%s'.\n",
                warn.c_str(), rv3.perGpuResponses[0].results[0].warning);
        return -1;
    }
    
    if (info != rv4.perGpuResponses[0].results[0].info)
    {
        fprintf(stderr, "GPU 0 test 0 warning should be '%s', but found '%s'.\n",
                info.c_str(), rv4.perGpuResponses[0].results[0].info);
        return -1;
    }

    if (warn != rv5.perGpuResponses[0].results[0].error.msg)
    {
        fprintf(stderr, "GPU 0 test 0 warning should be '%s', but found '%s'.\n",
                warn.c_str(), rv5.perGpuResponses[0].results[0].error.msg);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestSetGpuIndex()
{
    DcgmDiagResponseWrapper r1;
    DcgmDiagResponseWrapper r2;
    DcgmDiagResponseWrapper r3;
    dcgmDiagResponse_v3     rv3 = {0};
    dcgmDiagResponse_v4     rv4 = {0};
    dcgmDiagResponse_v5     rv5 = {0};

    r1.SetVersion3(&rv3);
    r2.SetVersion4(&rv4);
    r3.SetVersion5(&rv5);

    r2.SetGpuIndex(0);
    r1.SetGpuIndex(1);
    r3.SetGpuIndex(2);

    if (rv4.perGpuResponses[0].gpuId != 0)
    {
        fprintf(stderr, "Slot 0 should have gpu id 0 but is %u\n", rv4.perGpuResponses[0].gpuId);
        return -1;
    }

    if (rv3.perGpuResponses[1].gpuId != 1)
    {
        fprintf(stderr, "Slot 1 should have gpu id 1 but is %u\n", rv3.perGpuResponses[1].gpuId);
        return -1;
    }

    if (rv5.perGpuResponses[2].gpuId != 2)
    {
        fprintf(stderr, "Slot 2 should have gpu id 2 but is %u\n", rv5.perGpuResponses[2].gpuId);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestGetBasicTestResultIndex()
{
    DcgmDiagResponseWrapper drw;

    if (drw.GetBasicTestResultIndex(blacklistName) != DCGM_SWTEST_BLACKLIST)
    {
        fprintf(stderr, "%s didn't match its index.\n", blacklistName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(lwmlLibName) != DCGM_SWTEST_LWML_LIBRARY)
    {
        fprintf(stderr, "%s didn't match its index.\n", lwmlLibName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(lwdaMainLibName) != DCGM_SWTEST_LWDA_MAIN_LIBRARY)
    {
        fprintf(stderr, "%s didn't match its index.\n", lwdaMainLibName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(lwdaTkLibName) != DCGM_SWTEST_LWDA_RUNTIME_LIBRARY)
    {
        fprintf(stderr, "%s didn't match its index.\n", lwdaTkLibName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(permissionsName) != DCGM_SWTEST_PERMISSIONS)
    {
        fprintf(stderr, "%s didn't match its index.\n", permissionsName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(persistenceName) != DCGM_SWTEST_PERSISTENCE_MODE)
    {
        fprintf(stderr, "%s didn't match its index.\n", persistenceName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(elwName) != DCGM_SWTEST_ELWIRONMENT)
    {
        fprintf(stderr, "%s didn't match its index.\n", elwName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(pageRetirementName) != DCGM_SWTEST_PAGE_RETIREMENT)
    {
        fprintf(stderr, "%s didn't match its index.\n", pageRetirementName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(graphicsName) != DCGM_SWTEST_GRAPHICS_PROCESSES)
    {
        fprintf(stderr, "%s didn't match its index.\n", graphicsName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(inforomName) != DCGM_SWTEST_INFOROM)
    {
        fprintf(stderr, "%s didn't match its index.\n", inforomName.c_str());
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestRecordSystemError()
{
    DcgmDiagResponseWrapper r1;
    DcgmDiagResponseWrapper r2;
    DcgmDiagResponseWrapper r3;
    dcgmDiagResponse_v3     rv3 = {0};
    dcgmDiagResponse_v4     rv4 = {0};
    dcgmDiagResponse_v5     rv5 = {0};

    r1.SetVersion3(&rv3);
    r2.SetVersion4(&rv4);
    r3.SetVersion5(&rv5);

    r1.InitializeResponseStruct(8);
    r2.InitializeResponseStruct(8);
    r3.InitializeResponseStruct(8);

    static const std::string horrible("You've Moash'ed things horribly");

    r1.RecordSystemError(horrible);
    r2.RecordSystemError(horrible);
    r3.RecordSystemError(horrible);

    if (horrible != rv3.systemError)
    {
        fprintf(stderr, "V3 should've had system error '%s', but found '%s'.\n",
                horrible.c_str(), rv3.systemError);
        return -1;
    }

    if (horrible != rv4.systemError)
    {
        fprintf(stderr, "V4 should've had system error '%s', but found '%s'.\n",
                horrible.c_str(), rv4.systemError);
        return -1;
    }

    if (horrible != rv5.systemError.msg)
    {
        fprintf(stderr, "V4 should've had system error '%s', but found '%s'.\n",
                horrible.c_str(), rv5.systemError.msg);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestAddErrorDetail()
{
    DcgmDiagResponseWrapper r1;
    DcgmDiagResponseWrapper r2;
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v3     rv3;
    dcgmDiagResponse_v4     rv4;
    dcgmDiagResponse_v5     rv5;

    r1.SetVersion3(&rv3);
    r2.SetVersion4(&rv4);
    r3.SetVersion5(&rv5);

    dcgmDiagErrorDetail_t ed;
    snprintf(ed.msg, sizeof(ed.msg), "Egads! Kaladin failed to say his fourth ideal.");
    ed.code = 20;
    
    r1.AddErrorDetail(0, 0, "Diagnostic", ed, DCGM_DIAG_RESULT_FAIL);
    r2.AddErrorDetail(0, 0, "Diagnostic", ed, DCGM_DIAG_RESULT_FAIL);
    r3.AddErrorDetail(0, 0, "Diagnostic", ed, DCGM_DIAG_RESULT_FAIL);

    if (strcmp(rv3.perGpuResponses[0].results[0].warning, ed.msg))
    {
        fprintf(stderr, "Expected to find warning '%s', but found '%s'\n", ed.msg,
                rv3.perGpuResponses[0].results[0].warning);
        return -1;
    }

    if (strcmp(rv4.perGpuResponses[0].results[0].warning, ed.msg))
    {
        fprintf(stderr, "Expected to find warning '%s', but found '%s'\n", ed.msg,
                rv4.perGpuResponses[0].results[0].warning);
        return -1;
    }

    if (strcmp(rv5.perGpuResponses[0].results[0].error.msg, ed.msg))
    {
        fprintf(stderr, "Expected to find warning '%s', but found '%s'\n", ed.msg,
                rv5.perGpuResponses[0].results[0].error.msg);
        return -1;
    }

    if (rv5.perGpuResponses[0].results[0].error.code != ed.code)
    {
        fprintf(stderr, "Expected to find code %u, but found %u\n", ed.code,
                rv5.perGpuResponses[0].results[0].error.code);
        return -1;
    }

    r1.AddErrorDetail(0, DCGM_PER_GPU_TEST_COUNT, "Inforom", ed, DCGM_DIAG_RESULT_FAIL);
    r2.AddErrorDetail(0, DCGM_PER_GPU_TEST_COUNT, "Inforom", ed, DCGM_DIAG_RESULT_FAIL);
    r3.AddErrorDetail(0, DCGM_PER_GPU_TEST_COUNT, "Inforom", ed, DCGM_DIAG_RESULT_FAIL);

    if (rv3.inforom != DCGM_DIAG_RESULT_FAIL)
    {
        fprintf(stderr, "Expected to find a failure, but found %d\n", rv3.inforom);
        return -1;
    }

    if (strcmp(rv4.levelOneResults[DCGM_SWTEST_INFOROM].warning, ed.msg))
    {
        fprintf(stderr, "Expected to find error message '%s', but found '%s'\n", ed.msg,
                rv4.levelOneResults[DCGM_SWTEST_INFOROM].warning);
        return -1;
    }

    if (strcmp(rv5.levelOneResults[DCGM_SWTEST_INFOROM].error.msg, ed.msg))
    {
        fprintf(stderr, "Expected to find error message '%s', but found '%s'\n", ed.msg,
                rv5.levelOneResults[DCGM_SWTEST_INFOROM].error.msg);
        return -1;
    }

    if (rv5.levelOneResults[DCGM_SWTEST_INFOROM].error.code != ed.code)
    {
        fprintf(stderr, "Expected to find error code %u, but found %u\n", ed.code,
                rv5.levelOneResults[DCGM_SWTEST_INFOROM].error.code);
        return -1;
    }

    return DCGM_ST_OK;
}

