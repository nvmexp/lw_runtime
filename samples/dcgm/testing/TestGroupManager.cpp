/* 
 * File:   TestGroupManager.cpp
 */

#include "TestGroupManager.h"
#include "LwcmGroup.h"
#include "LwcmSettings.h"
#include <sstream>

TestGroupManager::TestGroupManager() {
}

TestGroupManager::~TestGroupManager() {
}

/*************************************************************************/
std::string TestGroupManager::GetTag()
{
    return std::string("groupmanager");
}

/*************************************************************************/
int TestGroupManager::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    m_gpus = gpus;
    return 0;
}

/*************************************************************************/
int TestGroupManager::Run()
{
    int st;
    int Nfailed = 0;    
    
    st = TestGroupCreation();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestGroupManager::TestGroupCreation FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestGroupManager::TestGroupCreation PASSED\n");
    
    
    st = TestGroupManageGpus();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestGroupManager::TestGroupManageGpus FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestGroupManager::TestGroupManageGpus PASSED\n");
    
    st = TestGroupReportErrOnDuplicate();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestGroupManager::TestGroupReportErrOnDuplicate FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestGroupManager::TestGroupReportErrOnDuplicate PASSED\n");     
    
    
    st = TestGroupGetAllGrpIds();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestGroupManager::TestGroupGetAllGrpIds FAILED with %d\n", st);
        if(st < 0)
            return -1;        
        
    } else {
        printf("TestGroupManager::TestGroupGetAllGrpIds PASSED\n");     
    }

    st = TestDefaultGpusAreDynamic();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestGroupManager::TestDefaultGpusAreDynamic FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestGroupManager::TestDefaultGpusAreDynamic PASSED\n");

    st = TestDefaultLwSwitchesAreDynamic();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestGroupManager::TestDefaultLwSwitchesAreDynamic FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestGroupManager::TestDefaultLwSwitchesAreDynamic PASSED\n");


    if(Nfailed > 0)
    {
        fprintf(stderr, "TestGroupManager had %d non-fatal test failures. Failing", Nfailed);
        return 1;
    } 
    
    return 0;
}

/*************************************************************************/
int TestGroupManager::Cleanup()
{
    return 0;
}

/*************************************************************************/
static DcgmCacheManager *GetCacheManagerInstance(void)
{
    DcgmCacheManager *cacheManager = new DcgmCacheManager();
    cacheManager->Init(1, 3600.0);
    return cacheManager;
}

/*************************************************************************/
int TestGroupManager::TestGroupCreation()
{
    unsigned int groupId;
    int st, retSt = 0;    
    
    DcgmCacheManager *cacheManager = GetCacheManagerInstance();

    LwcmGroupManager *pLwcmGrpManager;
    pLwcmGrpManager = new LwcmGroupManager(cacheManager);

    st = pLwcmGrpManager->AddNewGroup(0, "Test1", DCGM_GROUP_DEFAULT, &groupId);
    if (DCGM_ST_OK != st) {
        fprintf(stderr, "pLwcmGrpManager->AddNewGroup returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    st = pLwcmGrpManager->RemoveGroup(0, groupId);
    if (DCGM_ST_OK != st) {
        fprintf(stderr, "pLwcmGrpManager->RemoveGroup returned %d\n", st);
        retSt = 2;
        goto CLEANUP;
    }
    
CLEANUP:
    delete pLwcmGrpManager;
    pLwcmGrpManager = NULL;
    delete cacheManager;
    
    return retSt;
}

/*****************************************************************************/
int TestGroupManager::HelperOperationsOnGroup(LwcmGroupManager *pLwcmGrpManager, unsigned int groupId,
                                              string groupName, DcgmCacheManager *cacheManager)
{
    unsigned int fetchedGroupId;
    int st;    
    string str;
    std::vector<dcgmGroupEntityPair_t>groupEntities;
    
    for (unsigned int i = 0; i < m_gpus.size(); i++) {
        st = pLwcmGrpManager->AddEntityToGroup(0, groupId, DCGM_FE_GPU, m_gpus[i].gpuId);
        if (DCGM_ST_OK != st) {
            fprintf(stderr, "pLwcmGrp->AddEntityToGroup returned %d\n", st);
            return 1;
        }    
    }

    st = pLwcmGrpManager->GetGroupEntities(0, groupId, groupEntities);
    if(st != DCGM_ST_OK)
    {
        fprintf(stderr, "pLwcmGrpManager->GetGroupEntities() Failed with %d", st);
        return 2;
    }

    if(groupEntities.size() != m_gpus.size())
    {
        fprintf(stderr, "GPU ID size mismatch %u != %u\n",
                (unsigned int)groupEntities.size(), (unsigned int)m_gpus.size());
        return 2;
    }
    
    for (int i = m_gpus.size() - 1; i >= 0; i--) {
        st = pLwcmGrpManager->RemoveEntityFromGroup(0, groupId, DCGM_FE_GPU, m_gpus[i].gpuId);
        if (DCGM_ST_OK != st) {
            fprintf(stderr, "pLwcmGrp->RemoveEntityFromGroup returned %d\n", st);
            return 3;
        }    
    }
    
    st = pLwcmGrpManager->RemoveEntityFromGroup(0, groupId, DCGM_FE_GPU, m_gpus[0].gpuId);
    if (DCGM_ST_BADPARAM != st) {
        fprintf(stderr, "pLwcmGrp->RemoveEntityFromGroup should return DCGM_ST_BADPARAM. Returned : %d\n", st);
        return 4;
    }    
    
    
    str = pLwcmGrpManager->GetGroupName(0, groupId);
    if (str.compare(groupName)) {
        fprintf(stderr, "pLwcmGrp->GetGroupName failed to match the group name\n");
        return 5;
    }

    return 0;
}

/*****************************************************************************/
int TestGroupManager::TestGroupManageGpus()
{
    int st, retSt = 0;  
    vector<unsigned int> vecGroupIds;
    unsigned int numGroups = 20;
    vector<dcgmGroupEntityPair_t> groupEntities;

    DcgmCacheManager *cacheManager = GetCacheManagerInstance();

    LwcmGroupManager *pLwcmGrpManager;
    pLwcmGrpManager = new LwcmGroupManager(cacheManager);
    
    for (unsigned int g = 0; g < numGroups; ++g) 
    {
        string groupName;
        unsigned int groupId;
        
        stringstream s;
        s << g;
        groupName = "Test" + s.str();

        st = pLwcmGrpManager->AddNewGroup(0, groupName, DCGM_GROUP_EMPTY, &groupId);
        if (DCGM_ST_OK != st) {
            fprintf(stderr, "pLwcmGrpManager->AddNewGroup returned %d\n", st);
            retSt = 1;
            goto CLEANUP;
        }
        
        vecGroupIds.push_back(groupId);

        st = pLwcmGrpManager->GetGroupEntities(0, groupId, groupEntities);
        if(st != DCGM_ST_OK) {
            fprintf(stderr, "pLwcmGrpManager->GetGroupEntities returned %d\n", st);
            retSt = 3;
            goto CLEANUP;
        }
        
        st = HelperOperationsOnGroup(pLwcmGrpManager, groupId, groupName, cacheManager);
        if (DCGM_ST_OK != st) {
            retSt = 4;
            goto CLEANUP;                    
        }
    }
    
    for (unsigned int g = 0; g < numGroups; ++g)
    {
        st = pLwcmGrpManager->RemoveGroup(0, vecGroupIds[g]);
        if (DCGM_ST_OK != st) {
            fprintf(stderr, "pLwcmGrpManager->RemoveGroup returned %d\n", st);
            retSt = 5;
            goto CLEANUP;              
        }        
    }    
    
CLEANUP:
    delete pLwcmGrpManager;
    delete cacheManager;
    return retSt;
}

/*****************************************************************************/
int TestGroupManager::TestGroupReportErrOnDuplicate()
{
    int st, retSt = 0;
    unsigned int groupId;

    DcgmCacheManager *cacheManager = GetCacheManagerInstance();

    LwcmGroupManager *pLwcmGrpManager;
    pLwcmGrpManager = new LwcmGroupManager(cacheManager);
    
    st = pLwcmGrpManager->AddNewGroup(0, "Test1", DCGM_GROUP_EMPTY, &groupId);
    if (DCGM_ST_OK != st) {
        fprintf(stderr, "pLwcmGrpManager->AddNewGroup returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    } 

    st = pLwcmGrpManager->AddEntityToGroup(0, groupId, DCGM_FE_GPU, m_gpus[0].gpuId);
    if (DCGM_ST_OK != st) {
        fprintf(stderr, "pLwcmGrp->AddEntityToGroup returned %d\n", st);
        retSt = 4;
        goto CLEANUP;
    }  
    
    st = pLwcmGrpManager->AddEntityToGroup(0, groupId, DCGM_FE_GPU, m_gpus[0].gpuId);
    if (DCGM_ST_BADPARAM != st) {
        fprintf(stderr, "pLwcmGrp->AddEntityToGroup must fail for duplicate entry %d\n", st);
        retSt = 5;
        goto CLEANUP;
    }       
    
    st = pLwcmGrpManager->RemoveGroup(0, groupId);
    if (DCGM_ST_OK != st) {
        fprintf(stderr, "pLwcmGrpManager->RemoveGroup returned %d\n", st);
        retSt = 6;
        goto CLEANUP;        
    }
   
CLEANUP:
    delete pLwcmGrpManager;
    delete cacheManager;
    return retSt;
}

int TestGroupManager::TestGroupGetAllGrpIds()
{
    LwcmGroup *pLwcmGrp;    
    int st, retSt = 0;
    unsigned int groupId;
    unsigned int max_groups, index;
    unsigned int groupIdList[DCGM_MAX_NUM_GROUPS];
    unsigned int count;

    DcgmCacheManager *cacheManager = GetCacheManagerInstance();

    LwcmGroupManager *pLwcmGrpManager;
    pLwcmGrpManager = new LwcmGroupManager(cacheManager);
    

    max_groups = 10;
    for (index = 0; index < max_groups; index++) {
        st = pLwcmGrpManager->AddNewGroup(0, "Test", DCGM_GROUP_EMPTY, &groupId);
        if (DCGM_ST_OK != st) {
            fprintf(stderr, "pLwcmGrpManager->AddNewGroup returned %d\n", st);
            retSt = 1;
            goto CLEANUP;
        }
    }
    
    retSt = pLwcmGrpManager->GetAllGroupIds(0, groupIdList, &count);
    if (0 != retSt) {
        retSt = 2;
        goto CLEANUP;
    }
    
    if (count != max_groups + 2) { // +2 for the default group
        retSt = 3;
        goto CLEANUP;        
    }
    
    retSt = pLwcmGrpManager->RemoveAllGroupsForConnection(0);
    if (0 != retSt) {
        retSt = 4;
        goto CLEANUP;
    }
    
CLEANUP:
    delete pLwcmGrpManager;
    delete cacheManager;
    return retSt;    
}

/*************************************************************************/
int TestGroupManager::TestDefaultGpusAreDynamic()
{
    unsigned int groupId;
    bool found = false;
    int retSt = 0;
    unsigned int i;
    dcgmReturn_t dcgmReturn; 
    unsigned int fakeEntityId;
    size_t beforeSize, afterSize;
    std::vector<dcgmGroupEntityPair_t>entities;
    
    DcgmCacheManager *cacheManager = GetCacheManagerInstance();
    LwcmGroupManager *groupManager = new LwcmGroupManager(cacheManager);

    groupId = groupManager->GetAllGpusGroup();
    dcgmReturn = groupManager->GetGroupEntities(0, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Got error %d from GetGroupEntities()", dcgmReturn);
        retSt = 100;
        goto CLEANUP;
    }

    beforeSize = entities.size();

    if(beforeSize >= DCGM_MAX_NUM_DEVICES)
    {
        printf("TestGroupDefaultsAreDynamic Skipping test due to already having %d GPUs", 
               (int)beforeSize);
        retSt = 0;
        goto CLEANUP;
    }

    /* Add a fake GPU and make sure it appears in the entity list */
    fakeEntityId = cacheManager->AddFakeGpu();

    dcgmReturn = groupManager->GetGroupEntities(0, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Got error %d from GetGroupEntities()", dcgmReturn);
        retSt = 200;
        goto CLEANUP;
    }

    afterSize = entities.size();

    if(afterSize != beforeSize + 1)
    {
        fprintf(stderr, "Expected afterSize %d to be beforeSize %d + 1", 
                (int)afterSize, (int)beforeSize);
        retSt = 300;
        goto CLEANUP;
    }

    found = false;
    for(i = 0; i < entities.size(); i++)
    {
        if(fakeEntityId == entities[i].entityId && 
           entities[i].entityGroupId == DCGM_FE_GPU)
        {
            found = true;
            break;
        }
    }

    if(!found)
    {
        fprintf(stderr, "Unable to find GPU %u in list of %d", 
                fakeEntityId, (int)entities.size());
        retSt = 400;
        goto CLEANUP;
    }
    
CLEANUP:
    delete groupManager;
    groupManager = NULL;
    delete cacheManager;
    
    return retSt;
}

/*************************************************************************/
int TestGroupManager::TestDefaultLwSwitchesAreDynamic()
{
    unsigned int groupId;
    bool found = false;
    int retSt = 0;
    unsigned int i;
    dcgmReturn_t dcgmReturn; 
    unsigned int fakeEntityId;
    size_t beforeSize, afterSize;
    std::vector<dcgmGroupEntityPair_t>entities;
    
    DcgmCacheManager *cacheManager = GetCacheManagerInstance();
    LwcmGroupManager *groupManager = new LwcmGroupManager(cacheManager);

    groupId = groupManager->GetAllLwSwitchesGroup();
    dcgmReturn = groupManager->GetGroupEntities(0, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Got error %d from GetGroupEntities()", dcgmReturn);
        retSt = 100;
        goto CLEANUP;
    }

    beforeSize = entities.size();

    if(beforeSize >= DCGM_MAX_NUM_SWITCHES)
    {
        printf("TestGroupDefaultsAreDynamic Skipping test due to already having %d LwSwitches", 
               (int)beforeSize);
        retSt = 0;
        goto CLEANUP;
    }

    /* Add a fake LwSwitch and make sure it appears in the entity list */
    fakeEntityId = cacheManager->AddFakeLwSwitch();

    dcgmReturn = groupManager->GetGroupEntities(0, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Got error %d from GetGroupEntities()", dcgmReturn);
        retSt = 200;
        goto CLEANUP;
    }

    afterSize = entities.size();

    if(afterSize != beforeSize + 1)
    {
        fprintf(stderr, "Expected afterSize %d to be beforeSize %d + 1", 
                (int)afterSize, (int)beforeSize);
        retSt = 300;
        goto CLEANUP;
    }

    found = false;
    for(i = 0; i < entities.size(); i++)
    {
        if(fakeEntityId == entities[i].entityId && 
           entities[i].entityGroupId == DCGM_FE_SWITCH)
        {
            found = true;
            break;
        }
    }

    if(!found)
    {
        fprintf(stderr, "Unable to find LwSwitch %u in list of %d", 
                fakeEntityId, (int)entities.size());
        retSt = 400;
        goto CLEANUP;
    }
    
CLEANUP:
    delete groupManager;
    groupManager = NULL;
    delete cacheManager;
    
    return retSt;
}

/*************************************************************************/

