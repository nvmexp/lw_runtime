/* 
 * File:   LwcmGroup.cpp
 */


#include "LwcmGroup.h"
#include "logging.h"
#include "LwcmSettings.h"
#include <stdexcept>
#include "LwcmCacheManager.h"

/*****************************************************************************
 * Implementation for Group Manager Class
 *****************************************************************************/

/*****************************************************************************/
LwcmGroupManager::LwcmGroupManager(DcgmCacheManager *cacheManager)
{
    mGroupIdSequence = 0;
    mNumGroups = 0;
    mpCacheManager = cacheManager;
    lwosInitializeCriticalSection(&mLock);
    CreateDefaultGroups();
}

/*****************************************************************************/
LwcmGroupManager::~LwcmGroupManager()
{
    /* Go through the list of map and remove all the entries for LwcmGroup */
    Lock();

    GroupIdMap::iterator it;
    for(it = mGroupIdMap.begin(); it != mGroupIdMap.end(); ++it)
    {
        LwcmGroup *pLwcmGroup = it->second;
        delete(pLwcmGroup);
    }
    mGroupIdMap.clear();
    mNumGroups = 0;
    
    Unlock();
    lwosDeleteCriticalSection(&mLock);
}

/*****************************************************************************/
int LwcmGroupManager::Lock()
{
    lwosEnterCriticalSection(&mLock);
    return DCGM_ST_OK;    
}

/*****************************************************************************/
int LwcmGroupManager::Unlock()
{
    lwosLeaveCriticalSection(&mLock);
    return DCGM_ST_OK;    
}

/*****************************************************************************/
unsigned int LwcmGroupManager::GetNextGroupId()
{
    return lwosInterlockedIncrement(&mGroupIdSequence) - 1; // subtract one to start at group IDs at 0
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::CreateDefaultGroups()
{
    dcgmReturn_t dcgmRet;
    
    dcgmRet = AddNewGroup(DCGM_CONNECTION_ID_NONE, "DCGM_ALL_SUPPORTED_GPUS", DCGM_GROUP_DEFAULT, &mAllGpusGroupId);
    if(dcgmRet)
    {
        std::string error;
        error = "Default group creation failed. Error: ";
        error += errorString(dcgmRet);
        throw std::runtime_error(error);
    }

    dcgmRet = AddNewGroup(DCGM_CONNECTION_ID_NONE, "DCGM_ALL_SUPPORTED_LWSWITCHES", DCGM_GROUP_DEFAULT_LWSWITCHES, &mAllLwSwitchesGroupId);
    if(dcgmRet)
    {
        std::string error;
        error = "Default LwSwitch group creation failed. Error: ";
        error += errorString(dcgmRet);
        throw std::runtime_error(error);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
unsigned int LwcmGroupManager::GetAllGpusGroup()
{
    return mAllGpusGroupId;
}

/*****************************************************************************/
unsigned int LwcmGroupManager::GetAllLwSwitchesGroup()
{
    return mAllLwSwitchesGroupId;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::AddAllEntitiesToGroup(LwcmGroup *pLwcmGrp, 
                                                     dcgm_field_entity_group_t entityGroupId)
{
    dcgmReturn_t dcgmReturn; 
    dcgmReturn_t retSt = DCGM_ST_OK;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<dcgmGroupEntityPair_t>::iterator entityIt;


    dcgmReturn = mpCacheManager->GetAllEntitiesOfEntityGroup(1, entityGroupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got error %d from GetAllEntitiesOfEntityGroup()", dcgmReturn);
        return dcgmReturn;
    }

    if(entities.size() < 1)
    {
        PRINT_WARNING("%u", "Got 0 entities from GetAllEntitiesOfEntityGroup() of eg %u", entityGroupId);
    }

    /* Add the returned GPUs to our newly-created group */
    for(entityIt = entities.begin(); entityIt != entities.end(); ++entityIt)
    {
        dcgmReturn = pLwcmGrp->AddEntityToGroup((*entityIt).entityGroupId, (*entityIt).entityId);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %u %u %u", "Error %d from AddEntityToGroup(gid %u, eg %u, eid %u", 
                        (int)dcgmReturn, pLwcmGrp->GetGroupId(), (*entityIt).entityGroupId, (*entityIt).entityId);
            retSt = dcgmReturn;
            break;
        }
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::AddNewGroup(dcgm_connection_id_t connectionId, 
                                           string groupName, dcgmGroupType_t type, 
                                           unsigned int *pGroupId)
{
    unsigned int newGroupId;
    LwcmGroup   *pLwcmGrp;
    dcgmReturn_t dcgmReturn;
    
    if (NULL == pGroupId) {
        return DCGM_ST_BADPARAM;
    }
    
    Lock();
    
    if (mNumGroups >= DCGM_MAX_NUM_GROUPS + 2) 
    {
        PRINT_ERROR("", "Add Group: Max number of groups already configured");
        Unlock();
        return DCGM_ST_MAX_LIMIT;
    }

    newGroupId = GetNextGroupId();
    pLwcmGrp = new LwcmGroup(connectionId, groupName, newGroupId, mpCacheManager);
    
    if (type == DCGM_GROUP_DEFAULT) 
    {
        /* All GPUs on the node should be added to the group */
        dcgmReturn = AddAllEntitiesToGroup(pLwcmGrp, DCGM_FE_GPU);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "Got error %d from AddAllEntitiesToGroup()", dcgmReturn);
            Unlock();
            delete(pLwcmGrp);
            return dcgmReturn;
        }
    }
    else if (type == DCGM_GROUP_DEFAULT_LWSWITCHES)
    {
        /* All LwSwitches on the node should be added to the group */
        dcgmReturn = AddAllEntitiesToGroup(pLwcmGrp, DCGM_FE_SWITCH);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "Got error %d from AddAllEntitiesToGroup()", dcgmReturn);
            Unlock();
            delete(pLwcmGrp);
            return dcgmReturn;
        }
    }
    
    mGroupIdMap[newGroupId] = pLwcmGrp;
    *pGroupId = newGroupId;
    mNumGroups++;
    Unlock();

    PRINT_DEBUG("%u %s", "Added GroupId %u name %s", *pGroupId, groupName.c_str());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::RemoveGroup(dcgm_connection_id_t connectionId, unsigned int groupId) 
{
    LwcmGroup   *pLwcmGrp;
    GroupIdMap::iterator itGroup;
    std::vector<lwcmGroupRemoveCBEntry_t>::iterator removeCBIter;

    Lock();
    
    itGroup = mGroupIdMap.find(groupId);
    if (itGroup == mGroupIdMap.end()) {
        Unlock();
        PRINT_ERROR("%d","Delete Group: Not able to find entry corresponding to the group ID %d", groupId);        
        return DCGM_ST_NOT_CONFIGURED;
    } else {
        pLwcmGrp = itGroup->second;
        if (NULL == pLwcmGrp) {
            Unlock();
            PRINT_ERROR("%d","Delete Group: Invalid entry corresponding to the group ID %d", groupId);
            return DCGM_ST_GENERIC_ERROR;        
        }        
        
        delete pLwcmGrp;
        pLwcmGrp = NULL;
        mGroupIdMap.erase(itGroup);
    }

    mNumGroups--;


    /* Leaving this inside the lock for now for consistency. We will have to revisit
       this if it causes deadlocks between modules */
    for(removeCBIter = mOnRemoveCBs.begin(); removeCBIter != mOnRemoveCBs.end(); ++removeCBIter)
    {
        (*removeCBIter).callback(groupId, (*removeCBIter).userData);
    }
    
    Unlock();

    PRINT_DEBUG("%u", "Removed GroupId %u", groupId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::RemoveAllGroupsForConnection(dcgm_connection_id_t connectionId) 
{
    LwcmGroup *pLwcmGroup;
    GroupIdMap::iterator itGroup;
    std::vector<unsigned int>removeGroupIds;
    std::vector<unsigned int>::iterator removeIt;
    unsigned int groupId;

    Lock();

    for(itGroup = mGroupIdMap.begin(); itGroup != mGroupIdMap.end(); ++itGroup)
    {
        pLwcmGroup = itGroup->second;
        if(!pLwcmGroup)
            continue;
        groupId = pLwcmGroup->GetGroupId();
        
        if(connectionId == pLwcmGroup->GetConnectionId())
        {
            PRINT_DEBUG("%u %u", "RemoveAllGroupsForConnection queueing removal of connectionId %u, groupId %u",
                        connectionId, groupId);
            removeGroupIds.push_back(groupId);
        }
    }

    Unlock(); /* Unlock since RemoveGroup() will acquire the lock for each groupId */

    for(removeIt = removeGroupIds.begin(); removeIt != removeGroupIds.end(); ++removeIt)
    {
        RemoveGroup(connectionId, *removeIt);
    }

    PRINT_DEBUG("%u %u", "Removed %u groups for connectionId %u", (unsigned int)removeGroupIds.size(), 
                (unsigned int)connectionId);

    return DCGM_ST_OK;    
}

/*****************************************************************************/
LwcmGroup* LwcmGroupManager::GetGroupById(dcgm_connection_id_t connectionId, unsigned int groupId) 
{
    LwcmGroup   *pLwcmGrp;
    GroupIdMap::iterator itGroup;

    itGroup = mGroupIdMap.find(groupId);
    if (itGroup == mGroupIdMap.end()) {
        PRINT_ERROR("%d","Get Group: Not able to find entry corresponding to the group ID %d", groupId);
        return NULL;
    } else {
        pLwcmGrp = itGroup->second;
        if (NULL == pLwcmGrp) {
            PRINT_ERROR("%d","Get Group: Invalid entry corresponding to the group ID %d", groupId);
            return NULL;        
        }        
    }

    return pLwcmGrp;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::GetGroupEntities(dcgm_connection_id_t connectionId, unsigned int groupId,
                                                std::vector<dcgmGroupEntityPair_t> &entities)
{
    dcgmReturn_t ret;

    Lock();
    /* See if this is one of the special fully-dynamic all-entity groups */
    if(groupId == mAllGpusGroupId || groupId == mAllLwSwitchesGroupId)
    {
        dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;
        if(groupId == mAllLwSwitchesGroupId)
            entityGroupId = DCGM_FE_SWITCH;

        ret = mpCacheManager->GetAllEntitiesOfEntityGroup(1, entityGroupId, entities);
        if(ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %u", "GetGroupEntities Got error %d from GetAllEntitiesOfEntityGroup() for groupId %u", 
                        ret, groupId);
        }
        else
            PRINT_DEBUG("%u %u", "GetGroupEntities got %u entities for dynamic group %u", 
                        (unsigned int)entities.size(), groupId);
        Unlock();
        return ret;
    }

    /* This is a regular group. Just return its list */
    LwcmGroup *groupObj = GetGroupById(connectionId, groupId);
    if(!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    ret = groupObj->GetEntities(entities);
    Unlock();
    return ret;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::GetGroupGpuIds(dcgm_connection_id_t connectionId, unsigned int groupId,
                                              std::vector<unsigned int> &gpuIds)
{
    std::vector<dcgmGroupEntityPair_t>::iterator entityIter;
    std::vector<dcgmGroupEntityPair_t> entities;
    dcgmReturn_t ret = GetGroupEntities(connectionId, groupId, entities);
    if(ret != DCGM_ST_OK)
        return ret;

    for(entityIter = entities.begin(); entityIter != entities.end(); ++entityIter)
    {
        if((*entityIter).entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back((*entityIter).entityId);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::string LwcmGroupManager::GetGroupName(dcgm_connection_id_t connectionId, unsigned int groupId)
{
    std::string ret;

    Lock();
    LwcmGroup *groupObj = GetGroupById(connectionId, groupId);
    if(!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return ret;
    }

    ret = groupObj->GetGroupName();
    Unlock();
    return ret;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::AddEntityToGroup(dcgm_connection_id_t connectionId, 
                                                unsigned int groupId, 
                                                dcgm_field_entity_group_t entityGroupId, 
                                                dcgm_field_eid_t entityId)
{
    dcgmReturn_t ret;

    Lock();
    LwcmGroup *groupObj = GetGroupById(connectionId, groupId);
    if(!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    ret = groupObj->AddEntityToGroup(entityGroupId, entityId);
    Unlock();

    PRINT_DEBUG("%u %u %u %u %d", "conn %u, groupId %u added eg %u, eid %u. ret %d",
                connectionId, groupId, entityGroupId, entityId, (int)ret);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::RemoveEntityFromGroup(dcgm_connection_id_t connectionId, 
                                                     unsigned int groupId, 
                                                     dcgm_field_entity_group_t entityGroupId, 
                                                     dcgm_field_eid_t entityId)
{
    dcgmReturn_t ret;

    Lock();
    LwcmGroup *groupObj = GetGroupById(connectionId, groupId);
    if(!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    ret = groupObj->RemoveEntityFromGroup(entityGroupId, entityId);
    Unlock();

    PRINT_DEBUG("%u %u %u %u %d", "conn %u, groupId %u removed eg %u, eid %u. ret %d",
                connectionId, groupId, entityGroupId, entityId, (int)ret);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::AreAllTheSameSku(dcgm_connection_id_t connectionId, unsigned int groupId,
                                                int *areAllSameSku)
{
    dcgmReturn_t ret;

    if(!areAllSameSku)
        return DCGM_ST_BADPARAM;

    Lock();
    LwcmGroup *groupObj = GetGroupById(connectionId, groupId);
    if(!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    *areAllSameSku = groupObj->AreAllTheSameSku();
    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::verifyAndUpdateGroupId(unsigned int *groupId)
{

    if (*groupId == DCGM_GROUP_ALL_GPUS)
    { // must be before test below since DCGM_GROUP_ALL_GPUS is a large number
         *groupId = mAllGpusGroupId;
    }
    else if(*groupId == DCGM_GROUP_ALL_LWSWITCHES)
    {
        *groupId = mAllLwSwitchesGroupId;
    }

    /* Check that the groupId is actually a valid group */
    dcgmReturn_t ret = DCGM_ST_OK;

    Lock();
    LwcmGroup *groupObj = GetGroupById(0, *groupId);
    if(!groupObj)
    {
        PRINT_DEBUG("%u", "Group %u not found", *groupId);
        ret = DCGM_ST_NOT_CONFIGURED;
    }
    Unlock();

    return ret;
}

/*****************************************************************************/
int LwcmGroupManager::GetAllGroupIds(dcgm_connection_id_t connectionId, unsigned int groupIdList[], 
                                     unsigned int *pCount)
{
    LwcmGroup *pLwcmGrp;
    GroupIdMap::iterator itGroup;
    unsigned int count = 0;

    Lock();
    
    for (itGroup = mGroupIdMap.begin(); itGroup != mGroupIdMap.end(); ++itGroup) 
    {
        pLwcmGrp = itGroup->second;
        if (NULL == pLwcmGrp) {
            PRINT_ERROR("%u", "NULL LwcmGroup() at groupId %u", itGroup->first);
            continue;
        }

        groupIdList[count++] = pLwcmGrp->GetGroupId();
    }

    *pCount = count;
    Unlock();
    return 0;
}

/*****************************************************************************/
void LwcmGroupManager::OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection)
{
    RemoveAllGroupsForConnection(connectionId);
}

/*****************************************************************************/
dcgmReturn_t LwcmGroupManager::AllSupportPolicyManagement(dcgm_connection_id_t connectionId, unsigned int groupId, 
                                                          int *supportPolicyMgmt)
{
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<dcgmGroupEntityPair_t>::iterator entityIter;
    dcgmGpuBrandType_t brand;

    *supportPolicyMgmt = 1;

    dcgmReturn = GetGroupEntities(connectionId, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    for(entityIter = entities.begin(); entityIter != entities.end(); ++entityIter)
    {
        if((*entityIter).entityGroupId != DCGM_FE_GPU)
        {
            /* Only GPU entities have the policy management restriction */
            continue;
        }

        brand = mpCacheManager->GetGpuBrand((*entityIter).entityId);
        if(brand != DCGM_GPU_BRAND_TESLA && brand != DCGM_GPU_BRAND_QUADRO)
        {
            *supportPolicyMgmt = 0;
            PRINT_DEBUG("%u %u %u", "groupId %u does not support policy management because gpuId %u is not tesla or lwdqro (%u)",
                        groupId, (*entityIter).entityId, brand);
            return DCGM_ST_OK;
        }
    }

    PRINT_DEBUG("%u", "groupId %u supports policy management.", groupId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void LwcmGroupManager::SubscribeForGroupEvents(dcgmOnRemoveGroup_f onRemoveCB, void *userData)
{
    lwcmGroupRemoveCBEntry_t insertEntry;

    insertEntry.callback = onRemoveCB;
    insertEntry.userData = userData;

    Lock();

    mOnRemoveCBs.push_back(insertEntry);
    
    Unlock();
}

/*****************************************************************************
 * Group Class Implementation
 *****************************************************************************/

/*****************************************************************************/
LwcmGroup::LwcmGroup(dcgm_connection_id_t connectionId, string name, 
                     unsigned int groupId, DcgmCacheManager *cacheManager)
{
    mGroupId = groupId;
    mName = name;
    mpCacheManager = cacheManager;
    mConnectionId = connectionId;
}

/*****************************************************************************/
LwcmGroup::~LwcmGroup() 
{
    mEntityList.clear();
}

/*****************************************************************************/
dcgmReturn_t LwcmGroup::AddEntityToGroup(dcgm_field_entity_group_t entityGroupId, 
                                         dcgm_field_eid_t entityId)
{
    dcgmGroupEntityPair_t insertEntity;

    DcgmcmGpuStatus_t entityStatus = mpCacheManager->GetEntityStatus(entityGroupId, entityId);
    if(entityStatus != DcgmcmGpuStatusOk && entityStatus != DcgmcmGpuStatusFakeGpu)
    {
        PRINT_ERROR("%u %u %d", "eg %u, eid %u is in status %d. Not adding to group.",
                    entityGroupId, entityId, entityStatus);
        if(entityStatus == DcgmcmGpuStatusUnsupported)
            return DCGM_ST_GPU_NOT_SUPPORTED;
        else if(entityStatus == DcgmcmGpuStatusGpuLost)
            return DCGM_ST_GPU_IS_LOST;
        else
            return DCGM_ST_BADPARAM; /* entity is bad */
    }

    insertEntity.entityGroupId = entityGroupId;
    insertEntity.entityId = entityId;
    
    /* Check if entity is already added to the group */
    for (unsigned int i = 0; i < mEntityList.size(); ++i) 
    {
        if (mEntityList[i].entityGroupId == insertEntity.entityGroupId && 
            mEntityList[i].entityId == insertEntity.entityId)
        {
            PRINT_WARNING("%u %u %u", "AddEntityToGroup groupId %u eg %u, eid %u was already in the group", 
                          mGroupId, entityGroupId, entityId);
            return DCGM_ST_BADPARAM;
        }
    }
    
    mEntityList.push_back(insertEntity);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroup::RemoveEntityFromGroup(dcgm_field_entity_group_t entityGroupId, 
                                              dcgm_field_eid_t entityId)
{
    for (unsigned int i = 0; i < mEntityList.size(); ++i) 
    {
        if (mEntityList[i].entityGroupId == entityGroupId && 
            mEntityList[i].entityId == entityId)
        {
            mEntityList.erase(mEntityList.begin() + i);
            return DCGM_ST_OK;
        }
    }

    PRINT_ERROR("%u %u %u", "Tried to remove eg %u, eid %u from groupId %u. was not found.",
                entityGroupId, entityId, GetGroupId());
    return DCGM_ST_BADPARAM;
}

/*****************************************************************************/
string LwcmGroup::GetGroupName() 
{
    return mName;
}

/*****************************************************************************/
unsigned int LwcmGroup::GetGroupId() 
{
    return mGroupId;
}

/*****************************************************************************/
dcgm_connection_id_t LwcmGroup::GetConnectionId()
{
    return mConnectionId;
}

/*****************************************************************************/
dcgmReturn_t LwcmGroup::GetEntities(std::vector<dcgmGroupEntityPair_t> &entities)
{
    entities = mEntityList;
    return DCGM_ST_OK;
}

/*****************************************************************************/
int LwcmGroup::AreAllTheSameSku(void)
{
    unsigned int i;
    std::vector<unsigned int>gpuIds;

    /* Make a copy of the gpuIds. We're passing by ref to AreAllGpuIdsSameSku() */
    for(i = 0; i < mEntityList.size(); i++)
    {
        if(mEntityList[i].entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back(mEntityList[i].entityId);
    }

    int allTheSame = mpCacheManager->AreAllGpuIdsSameSku(gpuIds);
    return allTheSame;
}

/*****************************************************************************/

