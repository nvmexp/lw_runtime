#define DCGM_INIT_UUID 
#include "dcgm_agent_internal.h"
#include "TestPolicyManager.h"
#include "timelib.h"
#include <unistd.h>
#include <stddef.h>
#include <ctime>
#include <iostream>
#include <string.h>

const etblDCGMEngineTestInternal *g_pEtbl = NULL;

TestPolicyManager::TestPolicyManager() {}
TestPolicyManager::~TestPolicyManager() {}

int TestPolicyManager::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    return 0;
}

static bool g_cbackBegin = false;
static bool g_cbackEnd = false;

int callback1(void * data)
{
    g_cbackBegin = true;
    return 0;
}

int callback2(void * data)
{
    g_cbackEnd = true;
    return 0;
}


int TestPolicyManager::Run()
{
    int st; 
    int Nfailed = 0;

    st = TestPolicySetGet();
    if(st)
    {   
        Nfailed++;
        fprintf(stderr, "TestPolicyManager::Test policy set/get FAILED with %d\n", st);
        if(st < 0)
            return -1; 
    }   
    
    printf("TestPolicyManager::Test policy set/get PASSED\n");

    st = TestPolicyRegUnreg();
    if(st)
    {   
        Nfailed++;
        fprintf(stderr, "TestPolicyManager::Test policy reg/unreg FAILED with %d\n", st);
        if(st < 0)
            return -1; 
    }
    
    printf("TestPolicyManager::Test policy reg/unreg PASSED\n");

	st = TestPolicyRegUnregXID();
	if (st)
	{
        Nfailed++;
        fprintf(stderr, "TestPolicyManager::Test policy reg/unreg for XID errors FAILED with %d\n", st);
        if(st < 0)
            return -1; 
	}

	printf("TestPolicyManager::Test policy reg/unreg with XID PASSED\n");

    if(Nfailed > 0)
    {   
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }   

    return 0;
}

int TestPolicyManager::Cleanup()
{
    return 0;
}

std::string TestPolicyManager::GetTag()
{
    return std::string("policymanager");
}

int TestPolicyManager::TestPolicyRegUnreg()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupInfo_t groupInfo;
    dcgmPolicyCondition_t condition = (dcgmPolicyCondition_t)0;
    dcgmStatus_t statusHandle = 0;
    dcgmInjectFieldValue_t fv;
    timelib64_t start;
    timelib64_t now;

    TestPolicySetGet();

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmInternalGetExportTable failed with %d\n", (int)result);
        goto cleanup;
    }

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupCreate failed with %d\n", (int)result);
        goto cleanup;
    }

    groupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        goto cleanup;
    }

    result = dcgmStatusCreate(&statusHandle);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineStatusCreate failed with %d\n", (int)result);
        goto cleanup;
    }

    dcgmPolicy_t policy;
    memset(&policy, 0, sizeof(policy));
    policy.version = dcgmPolicy_version;
    policy.condition = DCGM_POLICY_COND_DBE;
    policy.parms[0].tag = dcgmPolicyConditionParms_t::LLONG;
    policy.parms[0].val.llval = 4; /* Fail with > 4 */

    result = dcgmPolicySet(m_dcgmHandle, groupId, &policy, &statusHandle);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        printf("dcgmPolicySet was not supported. This is expected for VdChip and Lwdqro hardware.\n");
        result = DCGM_ST_OK;
        goto cleanup;
    }
    else if(result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmPolicySet failed with %d\n", (int)result);
        goto cleanup;
    }

    condition = DCGM_POLICY_COND_DBE;
    result = dcgmPolicyRegister(m_dcgmHandle, groupId, condition, &callback1, &callback2);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        printf("dcgmPolicyRegister was not supported. This is expected for VdChip and Lwdqro hardware.\n");
        result = DCGM_ST_OK;
        goto cleanup;
    }
    else if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmPolicyRegister failed with %d\n", (int)result);
        goto cleanup;
    }

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_ECC_LWRRENT;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 1;
    fv.ts = (std::time(0)*1000000)+60000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, groupInfo.entityList[0].entityId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        goto cleanup;
    }

    fv.fieldId = DCGM_FI_DEV_ECC_DBE_VOL_DEV;
    fv.value.i64 = policy.parms[0].val.llval + 1;
    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, groupInfo.entityList[0].entityId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        goto cleanup;
    }

    /* This is a no-op now, but we should verify that it actually returns 0 */
    result = dcgmPolicyTrigger(m_dcgmHandle);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEnginePolicyTrigger failed with %d\n", (int)result);
        goto cleanup;
    }

    // wait for the callbacks for up to 15 seconds
    start = timelib_usecSince1970();
    now = start;
    while ((!g_cbackBegin || !g_cbackEnd) && (now - start < 15 * 1000000))
    {
        usleep(10000); // 10ms
        now = timelib_usecSince1970();
    }

    if (!g_cbackBegin || !g_cbackEnd)
    {
        result = DCGM_ST_GENERIC_ERROR;
        fprintf(stderr, "cbackBegin %d, cbackEnd %d\n", g_cbackBegin, g_cbackEnd);
        goto cleanup;
    }

    result = dcgmPolicyUnregister(m_dcgmHandle, groupId, condition);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEnginePolicyUnregister failed with %d\n", (int)result);
        goto cleanup;
    }
cleanup:
    if (statusHandle)
        dcgmStatusDestroy(statusHandle);
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
 
    return result;
}

int TestPolicyManager::TestPolicySetGet()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupInfo_t groupInfo;
    dcgmPolicy_t *policy;
    dcgmPolicy_t *newPolicy;
    dcgmStatus_t statusHandle = 0;

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
        return -1;

    groupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
        return -1;

    result = dcgmStatusCreate(&statusHandle);
    if (result != DCGM_ST_OK)
        return -1;

    policy = (dcgmPolicy_t *) malloc(sizeof(dcgmPolicy_t) * groupInfo.count);
    newPolicy = (dcgmPolicy_t *) malloc(sizeof(dcgmPolicy_t) * groupInfo.count);
    if (!policy || !newPolicy)
        return -1;

    // need to free policy from here on
    for (unsigned int i = 0; i < groupInfo.count; i++)
    {
        policy[i].version = dcgmPolicy_version;
        newPolicy[i].version = dcgmPolicy_version;
    }

    result = dcgmPolicyGet(m_dcgmHandle, groupId, groupInfo.count, policy, statusHandle);
    if (result != DCGM_ST_OK)
        goto cleanup;

    // just check the version for now
    if (policy->version != dcgmPolicy_version)
    {
        result = DCGM_ST_VER_MISMATCH;
        goto cleanup;
    }

    policy->condition = DCGM_POLICY_COND_DBE;
    policy->parms[0].tag = dcgmPolicyConditionParms_t::BOOL;
    policy->parms[0].val.boolean = true;

    result = dcgmPolicySet(m_dcgmHandle, groupId, &policy[0], statusHandle);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        printf("dcgmPolicySet was not supported. This is expected for VdChip and Lwdqro hardware.\n");
        result = DCGM_ST_OK;
        goto cleanup;
    }
    else if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmPolicySet returned %d\n", (int)result);
        goto cleanup;
    }

    result = dcgmPolicyGet(m_dcgmHandle, groupId, groupInfo.count, newPolicy, statusHandle);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmPolicyGet returned %d\n", (int)result);
        goto cleanup;
    }

    if (policy->parms[0].tag == newPolicy->parms[0].tag &&
        policy->parms[0].val.boolean == newPolicy->parms[0].val.boolean)
        result = DCGM_ST_OK;
	else
		result = DCGM_ST_GENERIC_ERROR;

cleanup:
    if (policy)
        free (policy);
    if (newPolicy)
        free (newPolicy);
    if (statusHandle)
        dcgmStatusDestroy(statusHandle);
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
    return result;
}

int TestPolicyManager::TestPolicyRegUnregXID()
{
    dcgmGpuGrp_t groupId = 0;
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmGroupInfo_t groupInfo;
    dcgmPolicyCondition_t condition = (dcgmPolicyCondition_t)0;
    dcgmStatus_t statusHandle = 0;
    dcgmInjectFieldValue_t fv;
    timelib64_t start;
    timelib64_t now;
    dcgmPolicy_t policy;

	// clear global variables
	g_cbackBegin = false;
	g_cbackEnd = false;

    TestPolicySetGet();

    // for the inject calls
    result = dcgmInternalGetExportTable((const void**)&g_pEtbl, &ETID_DCGMEngineTestInternal);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmInternalGetExportTable failed with %d\n", (int)result);
        goto cleanup;
    }

    // Create a group that consists of all GPUs
    result = dcgmGroupCreate(m_dcgmHandle, DCGM_GROUP_DEFAULT, (char *)"TEST1", &groupId);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupCreate failed with %d\n", (int)result);
        goto cleanup;
    }

    groupInfo.version = dcgmGroupInfo_version;
    result = dcgmGroupGetInfo(m_dcgmHandle, groupId, &groupInfo);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineGroupGetInfo failed with %d\n", (int)result);
        goto cleanup;
    }

    result = dcgmStatusCreate(&statusHandle);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineStatusCreate failed with %d\n", (int)result);
        goto cleanup;
    }

    memset(&policy, 0, sizeof(policy));
    policy.version = dcgmPolicy_version;
    policy.condition = DCGM_POLICY_COND_XID;
    policy.parms[6].tag = dcgmPolicyConditionParms_t::BOOL;
    policy.parms[6].val.boolean = true;

    result = dcgmPolicySet(m_dcgmHandle, groupId, &policy, statusHandle);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        printf("dcgmPolicySet was not supported. This is expected for VdChip and Lwdqro hardware.\n");
        result = DCGM_ST_OK;
        goto cleanup;
    }
    else if (result != DCGM_ST_OK)
    {
        fprintf(stderr,"dcgmPolicySet failed with %d\n", (int)result);
        goto cleanup;
    }

    condition = DCGM_POLICY_COND_XID;
    result = dcgmPolicyRegister(m_dcgmHandle, groupId, condition, &callback1, &callback2);
    if (result == DCGM_ST_NOT_SUPPORTED)
    {
        printf("dcgmPolicyRegister was not supported. This is expected for VdChip and Lwdqro GPUs.");
        result = DCGM_ST_OK;
        goto cleanup;
    }
    else if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEnginePolicyRegister failed with %d\n", (int)result);
        goto cleanup;
    }

    fv.version = dcgmInjectFieldValue_version;
    fv.fieldId = DCGM_FI_DEV_XID_ERRORS;
    fv.fieldType = DCGM_FT_INT64;
    fv.status = 0;
    fv.value.i64 = 1;
    fv.ts = (std::time(0)*1000000)+60000000;

    result = DCGM_CALL_ETBL(g_pEtbl, fpdcgmInjectFieldValue, (m_dcgmHandle, groupInfo.entityList[0].entityId, &fv));
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "fpEngineInjectFieldValue failed with %d\n", (int)result);
        goto cleanup;
    }

    result = dcgmPolicyTrigger(m_dcgmHandle);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEnginePolicyTrigger failed with %d\n", (int)result);
        goto cleanup;
    }

    // wait for the callbacks for up to 15 seconds
    start = timelib_usecSince1970();
    now = start;
    while ((!g_cbackBegin || !g_cbackEnd) && (now - start < 15 * 1000000))
    {
        usleep(10000); // 10ms
        now = timelib_usecSince1970();
    }

    if (!g_cbackBegin || !g_cbackEnd)
    {
        result = DCGM_ST_GENERIC_ERROR;
        fprintf(stderr, "cbackBegin %d, cbackEnd %d\n", g_cbackBegin, g_cbackEnd);
        goto cleanup;
    }

    result = dcgmPolicyUnregister(m_dcgmHandle, groupId, condition);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEnginePolicyUnregister failed with %d\n", (int)result);
        goto cleanup;
    }
cleanup:
    if (statusHandle)
        dcgmStatusDestroy(statusHandle);
    if (groupId)
        dcgmGroupDestroy(m_dcgmHandle, groupId);
 
    return result;
}
