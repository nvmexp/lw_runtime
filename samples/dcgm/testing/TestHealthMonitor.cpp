#include "TestHealthMonitor.h"
#include "dcgm_agent_internal.h"
#include <stddef.h>
#include <ctime>
#include <iostream>
#include <string.h>

TestHealthMonitor::TestHealthMonitor() {}
TestHealthMonitor::~TestHealthMonitor() {}

extern const etblDCGMEngineTestInternal *g_pEtbl;

int TestHealthMonitor::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    m_gpus = gpus;
    return 0;
}

int TestHealthMonitor::Run()
{
    int st; 
    int Nfailed = 0;

    st = TestHMSet();
    if(st)
    {   
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM set FAILED with %d\n", st);
        if(st < 0)
            return -1; 
    }   
    else
        printf("TestHealthMonitor::Test HM set PASSED\n");

    st = TestHMCheckPCIe();
    if(st)
    {   
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (PCIe) FAILED with %d\n", st);
        if(st < 0)
            return -1; 
    }   
    else
        printf("TestHealthMonitor::Test HM check (PCIe) PASSED\n");

    st = TestHMCheckMemSbe();
    if(st)
    {   
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Mem,Sbe) FAILED with %d\n", st);
        if(st < 0)
            return -1; 
    }   
    else
        printf("TestHealthMonitor::Test HM check (Mem,Sbe) PASSED\n");
    
    st = TestHMCheckMemDbe();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Mem,Dbe) FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (Mem,Dbe) PASSED\n");

    st = TestHMCheckInforom();
    if(st)
    {   
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (InfoROM) FAILED with %d\n", st);
        if(st < 0)
            return -1; 
    }   
    else
        printf("TestHealthMonitor::Test HM check (InfoROM) PASSED\n");
    st = TestHMCheckThermal();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Thermal) FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (Thermal) PASSED\n");
    st = TestHMCheckPower();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (Power) FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (Power) PASSED\n");

    st = TestHMCheckLWLink();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestHealthMonitor::Test HM check (LWLink) FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestHealthMonitor::Test HM check (LWLink) PASSED\n");


    if(Nfailed > 0)
    {   
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }   

    return 0;
}

int TestHealthMonitor::Cleanup()
{
    return 0;
}

std::string TestHealthMonitor::GetTag()
{
    return std::string("healthmonitor");
}

int TestHealthMonitor::TestHMSet()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_MEM);
    dcgmHealthSystems_t oldSystems;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
        return -1;

    result = dcgmHealthSet(m_dcgmHandle, groupId, dcgmHealthSystems_t(0));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthGet(m_dcgmHandle, groupId, &oldSystems);
    if (result != DCGM_ST_OK)
        goto cleanup;

    if (oldSystems != (dcgmHealthSystems_t)0)
    {
        result = DCGM_ST_GENERIC_ERROR;
        goto cleanup;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthGet(m_dcgmHandle, groupId, &oldSystems);
    if (result != DCGM_ST_OK)
        goto cleanup;

    if (oldSystems != newSystems)
    {
        result = DCGM_ST_GENERIC_ERROR;
        goto cleanup;
    }

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
 
    return result;
}

int TestHealthMonitor::TestHMCheckMemDbe()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response = {0};
    dcgmInjectFieldValue_t fv;
    dcgmGroupInfo_t groupInfo;

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM);
    response.version = dcgmHealthResponse_version;

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
        goto cleanup;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupCreate failed with %d\n", (int)result);
        goto cleanup;
    }

    memset(&groupInfo, 0, sizeof(groupInfo));
    groupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        goto cleanup;
    }

    if(groupInfo.count < 1)
    {
        printf("Skipping TestHMCheckMemDbe due to no GPUs being present");
        result = DCGM_ST_OK; /* Don't fail */
        goto cleanup;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineHealthSet failed with %d\n", (int)result);
        goto cleanup;
    }

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_ECC_DBE_VOL_TOTAL;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 0;
    fv.ts = (std::time(0)*1000000) - 60000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, groupInfo.entityList[0].entityId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        goto cleanup;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        goto cleanup;
    }

    fv.fieldId = DCGM_FI_DEV_ECC_DBE_VOL_TOTAL;
    fv.value.i64 = 5;
    fv.ts = (std::time(0)*1000000) ;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, groupInfo.entityList[0].entityId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        goto cleanup;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        goto cleanup;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_FAIL)
    {
        fprintf(stderr, "response.overallHealth %d != DCGM_HEALTH_RESULT_FAIL\n",
                (int)response.overallHealth);
        result = DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.entities[0].systems[0].errors[0].msg << std::endl;

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);

    return result;
}


int TestHealthMonitor::TestHMCheckMemSbe()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response = {0};
    dcgmInjectFieldValue_t fv;
    dcgmGroupInfo_t groupInfo;
 
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_MEM);
    response.version = dcgmHealthResponse_version;

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
        goto cleanup;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupCreate failed with %d\n", (int)result);
        goto cleanup;
    }

    memset(&groupInfo, 0, sizeof(groupInfo));
    groupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        goto cleanup;
    }

    if(groupInfo.count < 1)
    {
        printf("Skipping TestHMCheckMemSbe due to no GPUs being present");
        result = DCGM_ST_OK; /* Don't fail */
        goto cleanup;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineHealthSet failed with %d\n", (int)result);
        goto cleanup;
    }

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.ts = (std::time(0)*1000000) - 5000000;
    fv.fieldId = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
    fv.value.i64 = 0;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, groupInfo.entityList[0].entityId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        goto cleanup;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        goto cleanup;
    }

    fv.fieldId = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
    fv.value.i64 = 20;
    fv.ts = (std::time(0)*1000000) ;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, groupInfo.entityList[0].entityId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        goto cleanup;
    }
    

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "dcgmEngineHealthCheck failed with %d\n", (int)result);
        goto cleanup;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
    {
        fprintf(stderr, "response.overallHealth %d != DCGM_HEALTH_RESULT_WARN\n", (int)response.overallHealth);
        result = DCGM_ST_GENERIC_ERROR;
    }

    std::cout << response.entities[0].systems[0].errors[0].msg << std::endl;

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
 
    return result;

}

int TestHealthMonitor::TestHMCheckPCIe()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response = {0};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;
 
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_PCIE);
    response.version = dcgmHealthResponse_version;

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
        goto cleanup;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
        goto cleanup;

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_PCIE_REPLAY_COUNTER;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 0;
    fv.ts = (std::time(0)*1000000) - 50000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
        goto cleanup;

    fv.value.i64 = 100;
    fv.ts = (std::time(0)*1000000) ;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
        goto cleanup;

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    std::cout << response.entities[0].systems[0].errors[0].msg << std::endl;

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
 
    return result;

}

int TestHealthMonitor::TestHMCheckInforom()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response = {0};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;
 
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_INFOROM);
    response.version = dcgmHealthResponse_version;

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
        goto cleanup;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
        goto cleanup;

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_INFOROM_CONFIG_VALID;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 0; // inject that it is invalid
    fv.ts = (std::time(0)*1000000) ;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
        goto cleanup;

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    std::cout << response.entities[0].systems[0].errors[0].msg << std::endl;

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
 
    return result;

}

int TestHealthMonitor::TestHMCheckThermal()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response = {0};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_THERMAL);
    response.version = dcgmHealthResponse_version;

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
        goto cleanup;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
        goto cleanup;

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_THERMAL_VIOLATION;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 0;
    fv.ts = (std::time(0)*1000000) - 50000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA && result != DCGM_ST_STALE_DATA)
        goto cleanup;

    fv.value.i64 = 1000;
    fv.ts = (std::time(0)*1000000) ;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
        goto cleanup;

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    std::cout << response.entities[0].systems[0].errors[0].msg << std::endl;

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);

    return result;
}

int TestHealthMonitor::TestHMCheckPower()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response = {0};
    dcgmInjectFieldValue_t fv;
    unsigned int gpuId = m_gpus[0].gpuId;

    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_POWER);
    response.version = dcgmHealthResponse_version;

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
        goto cleanup;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId);
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
        goto cleanup;

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_POWER_VIOLATION;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 0;
    fv.ts = (std::time(0)*1000000) - 50000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA && result != DCGM_ST_STALE_DATA)
        goto cleanup;

    fv.value.i64 = 1000;
    fv.ts = (std::time(0)*1000000) ;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
        goto cleanup;

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
        goto cleanup;

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
        result = DCGM_ST_GENERIC_ERROR;

    std::cout << response.entities[0].systems[0].errors[0].msg << std::endl;

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);

    return result;

}

int TestHealthMonitor::TestHMCheckLWLink()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmHealthResponse_t response = {0};
    dcgmInjectFieldValue_t fv;
    dcgmGroupInfo_t groupInfo; 
    unsigned int gpuId;
    dcgmHealthSystems_t newSystems = dcgmHealthSystems_t(DCGM_HEALTH_WATCH_LWLINK);
    response.version = dcgmHealthResponse_version;

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Cannot get export table: '%s'\n", errorString(result));
        goto cleanup;
    }

    // Create a group consisting of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Cannot create group 'TEST1': '%s'\n", errorString(result));
        goto cleanup;
    }

    //Get the group Info
    memset(&groupInfo, 0, sizeof(groupInfo));
    groupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        goto cleanup;
    }

    //Skip the test if no GPU is found
    if(groupInfo.count < 1)
    {
        printf("Skipping TestHMCheckLWLink due to no GPUs being present\n");
        result = DCGM_ST_OK; /* Don't fail */
        goto cleanup;
    }

    //Save the first GPU Id in the list
    gpuId = groupInfo.entityList[0].entityId;
    
    //Destroy the group and create a new group with the first GPU in the list
    if (groupId)
    {
        result = dcgmGroupDestroy(m_dcgmHandle, groupId);
        if (result != DCGM_ST_OK)
        {
            fprintf(stderr, "dcgmEngineGroupDestroy  failed with %d\n", (int)result);
            return result;
        }
        groupId = 0;
    }

    // Create an empty group
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_EMPTY, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Cannot create group 'TEST1': '%s'\n", errorString(result));
        goto cleanup;
    }

    //Add saved gpudId to the empty group
    result = dcgmGroupAddDevice(m_dcgmHandle, groupId, gpuId); 
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Cannot add device %u to group '%s'\n", gpuId, errorString(result));
        goto cleanup;
    }

    result = dcgmHealthSet(m_dcgmHandle, groupId, newSystems);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to set LWLINK health watch: '%s'\n", errorString(result));
        goto cleanup;
    }

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 0;
    fv.ts = (std::time(0)*1000000) - 50000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to inject a 0 value for an LWLINK field: '%s'\n",
                errorString(result));
        goto cleanup;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK && result != DCGM_ST_NO_DATA)
    {
        fprintf(stderr, "Unable to check the health watches for this system: '%s'\n", errorString(result));
        goto cleanup;
    }
    
    // Ensure the initial lwlink health is good otherwise report and skip test
    if (response.overallHealth != DCGM_HEALTH_RESULT_PASS)
    {
        printf("Skipping TestHealthMonitor::Test HM check (LWLink). "
               "Test cannot run since LWLink health check did not pass.\n");
        result = DCGM_ST_OK;
        goto cleanup;
    }

    fv.value.i64 = 0;
    fv.ts = (std::time(0)*1000000) - 50000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to inject an error to trigger the LWLINK health failure: '%s'\n",
                errorString(result));
        goto cleanup;
    }

    fv.value.i64 = 1;
    fv.ts =(std::time(0)*1000000) ;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, gpuId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to inject a second error to trigger the LWLINK health failure: '%s'\n",
                errorString(result));
        goto cleanup;
    }

    result = dcgmHealthCheck(m_dcgmHandle, groupId, &response);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Unable to check the LWLINK health watches after injecting a failure: '%s'\n",
                errorString(result));
        goto cleanup;
    }

    if (response.overallHealth != DCGM_HEALTH_RESULT_WARN)
    {
        result = DCGM_ST_GENERIC_ERROR;
        fprintf(stderr, "Did not get a health watch warning even though we injected errors.\n");
    }

    std::cout << response.entities[0].systems[0].errors[0].msg << std::endl;

cleanup:
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
 
    return result;

}

