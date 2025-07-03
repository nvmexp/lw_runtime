/* 
 * File:   LwcmHostEngineHandler.cpp
 */

#include "LwcmHostEngineHandler.h"
#include "lwcm_util.h"
#include "LwcmSettings.h"
#include "LwcmStatus.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "dlfcn.h" //dlopen, dlsym..etc
#include "logging.h"
#include "lwcmvalue.h"
#include "DcgmModule.h"
#include "DcgmModuleIntrospect.h"
#include "DcgmMetadataMgr.h"
#include "DcgmModuleHealth.h"
#include "DcgmModulePolicy.h"
#ifdef DCGM_BUILD_LWSWITCH_MODULE
    #include "DcgmModuleLwSwitch.h"
#endif
#ifdef DCGM_BUILD_VGPU_MODULE
    #include "DcgmModuleVgpu.h"
#endif
#include "dcgm_health_structs.h"
#include "dcgm_profiling_structs.h"

#include "lwml.h"
#include "LwcmGroup.h"

using namespace std;

LwcmHostEngineHandler* LwcmHostEngineHandler::mpHostEngineHandlerInstance = NULL;

/*****************************************************************************
 Constructor for LWCM Host Engine. Ilwokes the server constructor with the
 port number of listening socket
 *****************************************************************************/
LwcmHosEngineServer::LwcmHosEngineServer(unsigned short portNumber, char *socketPath, unsigned int isConnectionTCP)
                   : LwcmServer(portNumber, socketPath, isConnectionTCP)
{

}

/*****************************************************************************
 Destructor for LWCM Host Engine Server
 *****************************************************************************/
LwcmHosEngineServer::~LwcmHosEngineServer()
{

}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::TranslateBitmapToGpuVector(uint64_t gpuBitmap, std::vector<unsigned int> &gpuIds)

{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (gpuBitmap == 0)
    {
        unsigned int gId = DCGM_GROUP_ALL_GPUS;
        ret = mpGroupManager->verifyAndUpdateGroupId(&gId);

        if (ret == DCGM_ST_OK)
        {
            ret = mpGroupManager->GetGroupGpuIds(0, gId, gpuIds);
        }
    }
    else
    {
        unsigned int gpuId = 0;
        for (uint64_t i = 0x1; gpuBitmap != 0; i <<= 1, gpuId++)
        {
            if ((gpuBitmap & i) != 0)
            {
                // Bit is set, record this gpu
                gpuIds.push_back(gpuId);
            }

            // Clear that bit
            gpuBitmap &= ~i;
        }
    }

    return ret;
}

/*****************************************************************************/
void LwcmHostEngineHandler::RemoveUnhealthyGpus(std::vector<unsigned int> &gpuIds)
{
    std::vector<unsigned int> healthyGpus;
    dcgmReturn_t dcgmReturn;
    dcgm_health_msg_check_gpus_t msg;

    /* Prepare a health check RPC to the health module */
    memset(&msg, 0, sizeof(msg));
    
    if(gpuIds.size() > DCGM_MAX_NUM_DEVICES)
    {
        PRINT_ERROR("%d", "Too many GPU ids: %d. Truncating.", (int)gpuIds.size());
    }

    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_GPUS;
    msg.header.version = dcgm_health_msg_check_gpus_version;
    
    msg.systems = DCGM_HEALTH_WATCH_ALL;
    msg.startTime = 0;
    msg.endTime = 0;
    msg.response.version = dcgmHealthResponse_version1;
    msg.numGpuIds = DCGM_MIN(gpuIds.size(), DCGM_MAX_NUM_DEVICES);
    

    for (size_t i = 0; i < msg.numGpuIds; i++)
    {
        msg.gpuIds[i] = gpuIds[i];
    }

    dcgmReturn = ProcessModuleCommand(&msg.header);
    if(dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
    {
        PRINT_DEBUG("", "RemoveUnhealthyGpus not filtering due to health module not being loaded.");
        return;
    }
    else if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "ProcessModuleCommand failed with %d", dcgmReturn);
        return;
    }

    for (size_t i = 0; i < gpuIds.size(); i++)
    {
        int responseIndex = -1;
        for (unsigned int j = 0; j < DCGM_MAX_NUM_DEVICES; j++)
        {
            if (msg.response.gpu[j].gpuId == gpuIds[i])
            {
                responseIndex = j;

                break;
            }
        }

        if (responseIndex == -1)
            healthyGpus.push_back(gpuIds[i]);
        else
        {
            if (msg.response.gpu[responseIndex].overallHealth != DCGM_HEALTH_RESULT_FAIL)
                healthyGpus.push_back(gpuIds[i]);
        }
    }

    gpuIds.clear();
    gpuIds = healthyGpus;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessSelectGpusByTopology(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    uint64_t inputGpus;
    uint64_t outputGpus = 0;
    uint32_t numGpus;
    uint64_t hints;
    std::vector<unsigned int> gpuIds;

    lwcm::SchedulerHintRequest shr = pCmd->arg(0).schedulerhintrequest();

    if (shr.version() != dcgmTopoSchedHint_version1)
    {
        PRINT_ERROR("", "Incorrect version for getting a topology-based gpu scheduler hint.");
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    inputGpus = shr.inputgpuids();
    numGpus = shr.numgpus();
    hints = shr.hintflags();

    ret = TranslateBitmapToGpuVector(inputGpus, gpuIds);

    if (ret == DCGM_ST_OK)
    {
        if ((hints & DCGM_TOPO_HINT_F_IGNOREHEALTH) == 0)
        {
            RemoveUnhealthyGpus(gpuIds);
        }

        ret = mpCacheManager->SelectGpusByTopology(gpuIds, numGpus, outputGpus);
    }

    pCmd->mutable_arg(0)->set_i64(outputGpus);

    return ret;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessClientLogin(lwcm::Command *pCmd, bool *pIsComplete,
                                              LwcmServerConnection *pConnection)
{
    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_clientlogin())
    {
        PRINT_ERROR("", "CLIENT_LOGIN missing args or clientlogin");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const lwcm::ClientLogin *clientLogin = &pCmd->arg(0).clientlogin();

    bool persistAfterDisconnect = false;
    if (!clientLogin->has_persistafterdisconnect())
    {
        PRINT_DEBUG("%u", "connectionId %u Missing persistafterdisconnect",
                    pConnection->GetConnectionId());
        persistAfterDisconnect = false;
    }
    else
    {
        persistAfterDisconnect = (bool)clientLogin->persistafterdisconnect();
        PRINT_DEBUG("%d %u", "persistAfterDisconnect %d for connectionId %u",
                    persistAfterDisconnect, pConnection->GetConnectionId());
    }

    pConnection->SetPersistAfterDisconnect(persistAfterDisconnect);

    pCmd->set_status(DCGM_ST_OK);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGroupCreate(lwcm::Command *pCmd, bool *pIsComplete,
                                              LwcmServerConnection *pConnection,
                                              dcgm_connection_id_t pConnectionId)

{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_grpinfo())
    {
        LW_ASSERT(0);
        DEBUG_STDERR("Group create info argument is not set");
        PRINT_ERROR("", "Group create info argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    lwcm::GroupInfo *pLwcmGrpInfo = pCmd->mutable_arg(0)->mutable_grpinfo();
    unsigned int groupId;
    dcgmReturn_t lwcmRet;

    /* If group name is not specified as meta data then return error to the caller */
    if (!pLwcmGrpInfo->has_groupname() || !pLwcmGrpInfo->has_grouptype())
    {
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    lwcmRet = mpGroupManager->AddNewGroup(pConnectionId, pLwcmGrpInfo->groupname(),
            (dcgmGroupType_t)pLwcmGrpInfo->grouptype(), &groupId);
    if (DCGM_ST_OK != lwcmRet)
    {
        pCmd->set_status(lwcmRet);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }


    pLwcmGrpInfo->set_groupid(groupId);
    pCmd->set_status(DCGM_ST_OK);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessAddRemoveGroup(lwcm::Command *pCmd, bool *pIsComplete,
                                                 LwcmServerConnection *pConnection,
                                                 dcgm_connection_id_t pConnectionId)

{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_grpinfo())
    {
        LW_ASSERT(0);
        DEBUG_STDERR("Group add/remove device : Argument is not set");
        PRINT_ERROR("", "Group add/remove device : Argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const lwcm::GroupInfo *pLwcmGrpInfo = &(pCmd->arg(0).grpinfo());
    unsigned int groupId;
    dcgmReturn_t lwcmRet;

    /* If group name is not specified as meta data then return error to the caller */
    if (!pLwcmGrpInfo->has_groupid() || (0 == pLwcmGrpInfo->entity_size())) 
    {
        PRINT_ERROR("", "Group add/remove device: Group ID or GPU IDs not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Verify group id is valid */
    groupId = pLwcmGrpInfo->groupid();
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if ((unsigned int) groupId == mpGroupManager->GetAllGpusGroup() ||
        (unsigned int) groupId == mpGroupManager->GetAllLwSwitchesGroup())
    {
        pCmd->set_status(DCGM_ST_NOT_CONFIGURED);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    for (int i = 0; i < pLwcmGrpInfo->entity_size(); ++i)
    {

        if (pCmd->cmdtype() == lwcm::GROUP_ADD_DEVICE)
        {
            lwcmRet = mpGroupManager->AddEntityToGroup(pConnectionId, groupId, 
                                                       (dcgm_field_entity_group_t)pLwcmGrpInfo->entity(i).entitygroupid(),
                                                       (dcgm_field_eid_t)pLwcmGrpInfo->entity(i).entityid());
            if (DCGM_ST_OK != lwcmRet)
            {
                PRINT_ERROR("%d", "AddEntityToGroup returned %d", (int)lwcmRet);
                pCmd->set_status(lwcmRet);
                *pIsComplete = true;
                return DCGM_ST_OK;
            }
        } 
        else
        {
            lwcmRet = mpGroupManager->RemoveEntityFromGroup(pConnectionId, groupId, 
                                                            (dcgm_field_entity_group_t)pLwcmGrpInfo->entity(i).entitygroupid(),
                                                            (dcgm_field_eid_t)pLwcmGrpInfo->entity(i).entityid());
            if (DCGM_ST_OK != lwcmRet)
            {
                PRINT_ERROR("%d", "RemoveEntityFromGroup returned %d", (int)lwcmRet);
                pCmd->set_status(lwcmRet);
                *pIsComplete = true;
                return DCGM_ST_OK;
            }
        }
    }

    *pIsComplete = true;
    pCmd->set_status(DCGM_ST_OK);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGroupDestroy(lwcm::Command *pCmd, bool *pIsComplete,
                                               LwcmServerConnection *pConnection,
                                               dcgm_connection_id_t pConnectionId)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_grpinfo())
    {
        LW_ASSERT(0);
        DEBUG_STDERR("Group destroy info argument is not set");
        PRINT_ERROR("", "Group destroy info argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    
    const lwcm::GroupInfo *pLwcmGrpInfo = &(pCmd->arg(0).grpinfo());
    dcgmReturn_t lwcmRet;
    unsigned int groupId;
    LwcmGroup *pLwcmGrp;

    /* If group id is not specified return error to the caller */
    if (!pLwcmGrpInfo->has_groupid())
    {
        DEBUG_STDERR("Group destroy: Group ID is not specified");
        PRINT_ERROR("", "Group destroy: Group ID is not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = pLwcmGrpInfo->groupid();
    /* Verify group id is valid */
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    // Check if were delting the default group
    if ((unsigned int) groupId == mpGroupManager->GetAllGpusGroup() ||
        (unsigned int) groupId == mpGroupManager->GetAllLwSwitchesGroup())
    {
        pCmd->set_status(DCGM_ST_NOT_CONFIGURED);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
    }
    else
    {
        lwcmRet = mpGroupManager->RemoveGroup(pConnectionId, groupId);
        if (DCGM_ST_OK != lwcmRet)
        {
            DEBUG_STDERR("Group destroy: Can't delete the group");
            PRINT_ERROR("", "Group destroy: Can't delete the group");
            pCmd->set_status(lwcmRet);
            *pIsComplete = true;
        }
        else
        {
            pCmd->set_status(DCGM_ST_OK);
            *pIsComplete = true;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGroupInfo(lwcm::Command *pCmd, bool *pIsComplete,
                                            dcgm_connection_id_t pConnectionId)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_grpinfo())
    {
        LW_ASSERT(0);
        DEBUG_STDERR("Group Get Info info argument is not set");
        PRINT_ERROR("", "Group Get Info info argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
    }
        
    lwcm::GroupInfo *pLwcmGrpInfo = pCmd->mutable_arg(0)->mutable_grpinfo();
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t lwcmRet;
    unsigned int groupId;
    std::vector<dcgmGroupEntityPair_t>entities;

    /* If group id is not specified return error to the caller */
    if (!pLwcmGrpInfo->has_groupid())
    {
        DEBUG_STDERR("Group Get Info: Group ID is not specified");
        PRINT_ERROR("", "Group Get Info: Group ID is not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    groupId = pLwcmGrpInfo->groupid();
    /* Verify group id is valid */
    ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    pLwcmGrpInfo->set_groupname(mpGroupManager->GetGroupName(pConnectionId, groupId));

    ret = mpGroupManager->GetGroupEntities(pConnectionId, groupId, entities);
    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        pCmd->set_status(ret);
        *pIsComplete = true;
    }
    else
    {
        for (unsigned int index = 0; index < entities.size(); index++)
        {
            lwcm::EntityIdPair *eidPair = pLwcmGrpInfo->add_entity();
            eidPair->set_entitygroupid((unsigned int)entities[index].entityGroupId);
            eidPair->set_entityid((unsigned int)entities[index].entityId);
        }

        pCmd->set_status(DCGM_ST_OK);
        *pIsComplete = true;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGroupGetallIds(lwcm::Command *pCmd, bool *pIsComplete,
                                                 dcgm_connection_id_t pConnectionId)

{
    if (pCmd->opmode() != lwcm::OPERATION_SYSTEM)
    {
        PRINT_ERROR("", "Error: Get All Group Ids expected to be processed as a system command");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    unsigned int groupIdList[DCGM_MAX_NUM_GROUPS + 1];
    unsigned int count = 0, index = 0;
    int ret;
    lwcm::FieldMultiValues *pListGrpIds;

    /* Allocated list of group Ids to be returned back to the client */
    pListGrpIds = new lwcm::FieldMultiValues;

    /* Set the allocated values to the protobuf message */
    pCmd->add_arg()->set_allocated_fieldmultivalues(pListGrpIds);

    /* Ilwoke method to get all the groups from the system */
    ret = mpGroupManager->GetAllGroupIds(pConnectionId, groupIdList, &count);
    if (ret < 0)
    {
        PRINT_ERROR("%d", "Group Get All Ids returned error : %d", ret);
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Go through the list of group Ids and update the protobuf message */
    for (index = 0; index < count; index++)
    {
        /* Workaround for bug 1700109: don't show internal group IDs to users */
        if (groupIdList[index] == mpGroupManager->GetAllGpusGroup() || 
            groupIdList[index] == mpGroupManager->GetAllLwSwitchesGroup())
            continue;

        lwcm::Value* pLwcmValue = pListGrpIds->add_vals();
        pLwcmValue->set_i64(groupIdList[index]);
    }

    pCmd->set_status(DCGM_ST_OK);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessDiscoverDevices(lwcm::Command *pCmd, bool *pIsComplete)

{
    lwcm::FieldMultiValues *pListGpuIds;
    int onlySupported = 0; /* Default to returning old GPUs for old clients */

    if (pCmd->opmode() != lwcm::OPERATION_SYSTEM)
    {
        PRINT_WARNING("Wrong opmode for device discovering: %d",
                      "DISCOVER_DEVICES is only allowed for opmode lwcm::OPERATION_SYSTEM. Found opmode: %d",
                      static_cast<int>(pCmd->opmode()));
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Did the client provide arguments? */
    if (pCmd->arg_size())
    {
        if (pCmd->arg(0).has_i32())
            onlySupported = pCmd->arg(0).i32();
        /* Clear out the parameters received from the client */
        pCmd->clear_arg();
    }

    pListGpuIds = new lwcm::FieldMultiValues;
    pCmd->add_arg()->set_allocated_fieldmultivalues(pListGpuIds);

    PRINT_DEBUG("%d", "DISCOVER_DEVICES onlySupported %d", onlySupported);

    int ret = LwcmHostEngineHandler::Instance()->GetLwcmGpuIds(pListGpuIds, onlySupported);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetEntityList(lwcm::Command *pCmd, bool *pIsComplete)
{
    lwcm::EntityList *entityList;
    int onlySupported = 1;
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<dcgmGroupEntityPair_t>::iterator entityIter;

    /* Did the client provide arguments? */
    if (!pCmd->arg_size() || !pCmd->arg(0).has_entitylist())
    {
        PRINT_ERROR("", "GET_ENTITY_LIST was malformed.");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    entityList = pCmd->mutable_arg(0)->mutable_entitylist();
    
    if (entityList->has_entitygroupid())
        entityGroupId = (dcgm_field_entity_group_t)entityList->entitygroupid();
    else
        PRINT_DEBUG("", "GET_ENTITY_LIST had no entitygroupid");

    if (entityList->has_onlysupported())
        onlySupported = (dcgm_field_entity_group_t)entityList->onlysupported();
    else
        PRINT_DEBUG("", "GET_ENTITY_LIST had no onlysupported");

    int ret = mpCacheManager->GetAllEntitiesOfEntityGroup(onlySupported, entityGroupId, entities);
    if (ret)
    {
        PRINT_DEBUG("%d %u %d", "GetAllEntitiesOfEntityGroup(os %d, eg %u) returned %d", 
                    onlySupported, entityGroupId, ret);
        pCmd->set_status(ret);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    for (entityIter = entities.begin(); entityIter != entities.end(); ++entityIter)
    {
        lwcm::EntityIdPair *entityPair = entityList->add_entity();
        entityPair->set_entitygroupid((*entityIter).entityGroupId);
        entityPair->set_entityid((*entityIter).entityId);
    }

    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessSaveCachedStats(lwcm::Command *pCmd, bool *pIsComplete)
{

    if (!pCmd->arg_size() || !pCmd->arg(0).has_cachemanagersave())
    {
        /* Since this is a set command. This should never happen, the
           LWCMI (LW agent) must populate the command with the configuration */
        LW_ASSERT(0);
        DEBUG_STDERR("Save Cache parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
        
    lwcm::CacheManagerSave *pCacheManagerSave = pCmd->mutable_arg(0)->mutable_cachemanagersave();

    int ret = LwcmHostEngineHandler::Instance()->SaveCachedStats(pCacheManagerSave);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg();      // Clear arg as it's not needed anymore

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessLoadCachedStats(lwcm::Command *pCmd, bool *pIsComplete)
{

    if (!pCmd->arg_size() || !pCmd->arg(0).has_cachemanagerload())
    {
        /* Since this is a set command. This should never happen.
           LWCMI (LW agent) must populate the command with the configuration */
        LW_ASSERT(0);
        DEBUG_STDERR("Load Cache parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    lwcm::CacheManagerLoad *pCacheManagerLoad = pCmd->mutable_arg(0)->mutable_cachemanagerload();

    int ret = LwcmHostEngineHandler::Instance()->LoadCachedStats(pCacheManagerLoad);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg();      // Clear arg as it's not needed anymore

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessInjectFieldValue(lwcm::Command *pCmd, bool *pIsComplete)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_injectfieldvalue())
    {
        /* Since this is a set command. This should never happen.
           LWCMI (LW agent) must populate the command with the configuration */
        DEBUG_STDERR("INJECT_FIELD_VALUE parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
        
    lwcm::InjectFieldValue *pInjectFieldValue = pCmd->mutable_arg(0)->mutable_injectfieldvalue();
    dcgm_field_entity_group_t entityGroupId;
    dcgm_field_eid_t entityId;


    if (!pCmd->has_id())
        entityId = 0; /* Can be true for global fields */
    else
        entityId = pCmd->id();

    /* Handle when it's passed via the message vs the command */
    if (pInjectFieldValue->has_entityid())
        entityId = pInjectFieldValue->entityid();

    if (!pCmd->has_entitygroupid())
        entityGroupId = DCGM_FE_GPU; /* Support old clients that won't set entityGroupId */
    else
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    
    /* Handle when it's passed via the message vs the command */
    if (pInjectFieldValue->has_entitygroupid())
        entityGroupId = (dcgm_field_entity_group_t)pInjectFieldValue->entitygroupid();

    int ret = LwcmHostEngineHandler::Instance()->InjectFieldValue(entityGroupId, entityId, pInjectFieldValue);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg();      // Clear arg as it's not needed anymore

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetFieldLatestValue(lwcm::Command *pCmd, bool *pIsComplete)
{
    if (!pCmd->arg_size() && pCmd->has_id())
    {
        DEBUG_STDERR("Requested Field value and id must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
        
    lwcm::FieldValue *pFieldValue = (lwcm::FieldValue *)&(pCmd->arg(0).fieldvalue());
    dcgm_field_entity_group_t entityGroupId;

    /* Stay compatible with old protocols that don't provide entityGroupId */
    entityGroupId = DCGM_FE_GPU;
    if(pCmd->has_entitygroupid())
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();

    int ret = LwcmHostEngineHandler::Instance()->GetFieldValue(entityGroupId, pCmd->id(), pFieldValue->fieldid(), pFieldValue);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetFieldMultipleValues(lwcm::Command *pCmd, bool *pIsComplete)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_fieldmultivalues())
    {
        LW_ASSERT(0);
        DEBUG_STDERR("Requested Field multi value must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
        
    lwcm::FieldMultiValues *pFieldMultiValues = pCmd->mutable_arg(0)->mutable_fieldmultivalues();
    int ret;

    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    if (pCmd->has_entitygroupid())
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    else
        PRINT_WARNING("", "entityGroupId missing. Probably old client.");

    ret = LwcmHostEngineHandler::Instance()->GetFieldMultipleValues(entityGroupId, pCmd->id(), pFieldMultiValues);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessWatchFieldValue(lwcm::Command *pCmd, bool *pIsComplete,
                                                  DcgmWatcher &dcgmWatcher)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_watchfieldvalue())
    {
        /* Since this is a set command. This should never happen.
           DCGMI (LW agent) must populate the command with the configuration */
        DEBUG_STDERR("WATCH_FIELD_VALUE parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const lwcm::WatchFieldValue *pWatchFieldValue = &(pCmd->arg(0).watchfieldvalue());
    int ret;

    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    if (pCmd->has_entitygroupid())
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    else
        PRINT_WARNING("", "entityGroupId missing. Probably old client.");

    ret = LwcmHostEngineHandler::Instance()->WatchFieldValue(entityGroupId, pCmd->id(), pWatchFieldValue, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg();      // Clear arg as it's not needed anymore
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessUnwatchFieldValue(lwcm::Command *pCmd, bool *pIsComplete,
                                                    DcgmWatcher &dcgmWatcher)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_unwatchfieldvalue())
    {
        /* Since this is a set command. This should never happen.
           LWCMI (LW agent) must populate the command with the configuration */
        DEBUG_STDERR("UNWATCH_FIELD_VALUE parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const lwcm::UnwatchFieldValue *pUnwatchFieldValue = &(pCmd->arg(0).unwatchfieldvalue());
    int ret;

    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    if(pCmd->has_entitygroupid())
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    else
        PRINT_WARNING("", "entityGroupId missing. Probably old client.");

    ret = LwcmHostEngineHandler::Instance()->UnwatchFieldValue(entityGroupId, pCmd->id(), pUnwatchFieldValue, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg();      // Clear arg as it's not needed anymore

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessUpdateAllFields(lwcm::Command *pCmd, bool *pIsComplete)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_updateallfields())
    {
        /* Since this is a set command. This should never happen.
           LWCMI (LW agent) must populate the command with the configuration */
        LW_ASSERT(0);
        DEBUG_STDERR("UPDATE_ALL_FIELDS parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const lwcm::UpdateAllFields *pUpdateAllFields = &(pCmd->arg(0).updateallfields());

    int ret = LwcmHostEngineHandler::Instance()->UpdateAllFields(pUpdateAllFields);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg();      // Clear arg as it's not needed anymore
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessCacheManagerFieldInfo(lwcm::Command *pCmd, bool *pIsComplete)
{
    if (!pCmd->arg_size() || !pCmd->arg(0).has_cachemanagerfieldinfo())
    {
        /* Since this is a set command. This should never happen.
           LWCMI (LW agent) must populate the command with the configuration */
        LW_ASSERT(0);
        DEBUG_STDERR("CACHE_MANAGER_FIELD_INFO parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    dcgmCacheManagerFieldInfo_t fieldInfo = {0};
    std::string inFieldInfoStr = pCmd->arg(0).cachemanagerfieldinfo();

    if(inFieldInfoStr.size() != sizeof(fieldInfo))
    {
        PRINT_ERROR("%d %d", "Got CACHE_MANAGER_FIELD_INFO size %d. Expected %d",
                (int)inFieldInfoStr.size(), (int)sizeof(fieldInfo));
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Copy to a temp buffer so we aren't overwriting the string's c_str() */
    memcpy(&fieldInfo, (dcgmCacheManagerFieldInfo_t *)inFieldInfoStr.c_str(),
            sizeof(dcgmCacheManagerFieldInfo_t));

    int ret = LwcmHostEngineHandler::Instance()->GetCacheManagerFieldInfo(&fieldInfo);
    pCmd->set_status(ret);
    *pIsComplete = true;
    /* Set the memory contents from the temp buffer */
    pCmd->mutable_arg(0)->set_cachemanagerfieldinfo(&fieldInfo, sizeof(fieldInfo));

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessWatchFields(lwcm::Command *pCmd, bool *pIsComplete,
                                              DcgmWatcher &dcgmWatcher)
{
    unsigned int groupId;
    lwcm::WatchFields *pWatchFields;

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Config Get Err: Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = pCmd->id();
    /* Verify group id is valid */
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_watchfields())
    {
        PRINT_ERROR("", "WATCH_FIELDS parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_BADPARAM;
    }
    
    pWatchFields = pCmd->mutable_arg(0)->mutable_watchfields();

    if(pWatchFields->version() != dcgmWatchFields_version)
    {
        PRINT_ERROR("%d %d", "WATCH_FIELDS version mismatch read %d != expected %d",
                pWatchFields->version(), dcgmWatchFields_version);
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if(!pWatchFields->has_fieldgroupid() || !pWatchFields->has_maxkeepage() ||
            !pWatchFields->has_maxkeepsamples() || !pWatchFields->has_updatefreq())
    {
        PRINT_ERROR("", "WATCH_FIELDS missing field");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = WatchFieldGroup(groupId, (dcgmGpuGrp_t)pWatchFields->fieldgroupid(),
                          pWatchFields->updatefreq(), pWatchFields->maxkeepage(), pWatchFields->maxkeepsamples(),
                          dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg();      // Clear arg as it's not needed anymore
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessUnwatchFields(lwcm::Command *pCmd, bool *pIsComplete,
                                                DcgmWatcher &dcgmWatcher)
{
    unsigned int groupId;
    lwcm::UnwatchFields *pUnwatchFields;

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "UNWATCH_FIELDS: Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = pCmd->id();
    /* Verify group id is valid */
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_unwatchfields())
    {
        PRINT_ERROR("", "UNWATCH_FIELDS parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_BADPARAM;
    }

    pUnwatchFields = pCmd->mutable_arg(0)->mutable_unwatchfields();

    /* redundant check for fieldgroupid, but keeping it here in case we have optional fields in the future */
    if (!pUnwatchFields->has_fieldgroupid())
    {
        PRINT_ERROR("", "WATCH_FIELDS missing field");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = UnwatchFieldGroup(groupId, (dcgmGpuGrp_t)pUnwatchFields->fieldgroupid(), dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg(); // Clear arg as it's not needed anymore
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetPidInfo(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    int groupId;

    if (pCmd->opmode() != lwcm::OPERATION_GROUP_ENTITIES)
    {
        PRINT_ERROR("", "GET_PID_INFORMATION only works on groupIds");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Config Get Err: Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    // No group Id verification needed as its handled in GetProcessInfo

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from GET_PID_INFORMATION");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    ret = GetProcessInfo(pCmd->id(), (dcgmPidInfo_t *)pCmd->arg(0).blob().c_str());
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessFieldGroupCreate(lwcm::Command *pCmd, bool *pIsComplete,
                                                   DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmFieldGroupInfo_t *fieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_CREATE");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    fieldGrpInfo = (dcgmFieldGroupInfo_t *)pCmd->arg(0).blob().c_str();
    if (fieldGrpInfo->version != dcgmFieldGroupInfo_version)
    {
        PRINT_ERROR("%d %d", "FIELD_GROUP_CREATE version mismatch %d != %d",
                    fieldGrpInfo->version, dcgmFieldGroupInfo_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    std::vector<unsigned short>fieldIds(fieldGrpInfo->fieldIds, fieldGrpInfo->fieldIds+fieldGrpInfo->numFieldIds);
    /* This call will set fieldGrpInfo->fieldGroupId */
    ret = mpFieldGroupManager->AddFieldGroup(fieldGrpInfo->fieldGroupName,
                                             fieldIds,
                                             &fieldGrpInfo->fieldGroupId, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessFieldGroupDestroy(lwcm::Command *pCmd, bool *pIsComplete,
                                                    DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmFieldGroupInfo_t *fieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_DESTROY");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    fieldGrpInfo = (dcgmFieldGroupInfo_t *)pCmd->arg(0).blob().c_str();
    if (fieldGrpInfo->version != dcgmFieldGroupInfo_version)
    {
        PRINT_ERROR("%d %d", "FIELD_GROUP_DESTROY version mismatch %d != %d",
                    fieldGrpInfo->version, dcgmFieldGroupInfo_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Note: passing user-created flag */
    ret = mpFieldGroupManager->RemoveFieldGroup(fieldGrpInfo->fieldGroupId, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessFieldGroupGetOne(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    dcgmFieldGroupInfo_t *fieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_GET_ONE");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    fieldGrpInfo = (dcgmFieldGroupInfo_t *)pCmd->arg(0).blob().c_str();
    if (fieldGrpInfo->version != dcgmFieldGroupInfo_version)
    {
        PRINT_ERROR("%d %d", "FIELD_GROUP_GET_ONE version mismatch %d != %d",
                    fieldGrpInfo->version, dcgmFieldGroupInfo_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = mpFieldGroupManager->PopulateFieldGroupInfo(fieldGrpInfo);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessFieldGroupGetAll(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    dcgmAllFieldGroup_t *allFieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_GET_ALL");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    allFieldGrpInfo = (dcgmAllFieldGroup_t *)pCmd->arg(0).blob().c_str();
    if (allFieldGrpInfo->version != dcgmAllFieldGroup_version)
    {
        PRINT_ERROR("%d %d", "FIELD_GROUP_GET_ALL version mismatch %d != %d",
                allFieldGrpInfo->version, dcgmAllFieldGroup_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = mpFieldGroupManager->PopulateFieldGroupGetAll(allFieldGrpInfo);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessWatchRedefined(lwcm::Command *pCmd, bool *pIsComplete,
                                                 DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmWatchPredefined_t *watchPredef;
    dcgmFieldGrp_t fieldGroupId;
    unsigned int groupId;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from WATCH_PREDEFINED");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    watchPredef = (dcgmWatchPredefined_t *)pCmd->arg(0).blob().c_str();
    if (watchPredef->version != dcgmWatchPredefined_version)
    {
        PRINT_ERROR("%d %d", "WATCH_PREDEFINED version mismatch %d != %d",
                watchPredef->version, dcgmWatchPredefined_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = (unsigned int)(intptr_t)watchPredef->groupId;
    ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (ret != DCGM_ST_OK)
    {
        pCmd->set_status(ret);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    switch(watchPredef->watchPredefType)
    {
        case DCGM_WATCH_PREDEF_PID: /* Intentional fall-through */
        case DCGM_WATCH_PREDEF_JOB:
            fieldGroupId = mFieldGroupPidAndJobStats;
            break;

        case DCGM_WATCH_PREDEF_ILWALID:
        default:
            PRINT_ERROR("%d", "Invalid watchPredefType %d", (int)watchPredef->watchPredefType);
            pCmd->set_status(DCGM_ST_BADPARAM);
            *pIsComplete = true;
            return DCGM_ST_OK;
    }

    ret = WatchFieldGroup(groupId, fieldGroupId,
                          watchPredef->updateFreq, watchPredef->maxKeepAge,
                          watchPredef->maxKeepSamples, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessJobStartStats(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    string jobId;
    unsigned int groupId;

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "JOB_START Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    
    groupId = pCmd->id();
    
    /* Verify group id is valid */
    ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "JOB_START Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }            
    
    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()))
    {
        PRINT_ERROR("", "JOB_START Error: Job id is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    
    jobId = pCmd->arg(0).str();       
    ret = JobStartStats(jobId, groupId);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessJobStopStats(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    string jobId;

    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()))
    {
        PRINT_ERROR("", "JOB_START Error: Job id is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    
    jobId = pCmd->arg(0).str();       
    ret = JobStopStats(jobId);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessJobRemove(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    string jobId;

    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()))
    {
        PRINT_ERROR("", "JOB_REMOVE Error: Job id is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    jobId = pCmd->arg(0).str();
    ret = JobRemove(jobId);
    pCmd->set_status(ret);
    *pIsComplete = true;
        
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessJobGetInfo(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    string jobId;

    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()) || !(pCmd->arg(1).has_blob()))
    {
        PRINT_ERROR("", "JOB_START Error: Job id or output struct is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    
    jobId = pCmd->arg(0).str();
    ret = JobGetStats( jobId, (dcgmJobInfo_t *)pCmd->arg(1).blob().c_str());
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetTopologyAffinity(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmcm_sample_t sample;
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    dcgmAffinity_t *affinity_p;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned int>dcgmGpuIds;
    dcgmAffinity_t gpuAffinity;

    gpuAffinity.numGpus = 0;

    if (pCmd->opmode() != lwcm::OPERATION_GROUP_ENTITIES)
    {
        PRINT_ERROR("", "GET_TOPOLOGY_INFO_AFFINITY only works on groupIds");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void*)&gpuAffinity, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Get affinity Err: Group ID not specified");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void*)&gpuAffinity, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }
    else
        groupId = pCmd->id();

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuAffinity, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    dcgmReturn = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuAffinity, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    for (int i = 0; i < (int)entities.size(); i++)
    {
        /* Only consider GPUs */
        if (entities[i].entityGroupId != DCGM_FE_GPU)
            continue;
        
        dcgmGpuIds.push_back(entities[i].entityId);
    }

    if (dcgmGpuIds.size() < 1)
    {
        PRINT_DEBUG("%d", "No GPUs in group %d", groupId);
        finalizeCmd(pCmd, DCGM_ST_NO_DATA, pIsComplete, (void*)&gpuAffinity, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    // retrieve the latest sample of PCI topology information
    dcgmReturn = mpCacheManager->GetLatestSample(DCGM_FE_GPU, dcgmGpuIds[0], DCGM_FI_GPU_TOPOLOGY_AFFINITY, 
                                                 &sample, 0);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: unable to retrieve affinity information");
        finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuAffinity, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    affinity_p = (dcgmAffinity_t *) sample.val.blob;

    // now run through the topology list comparing it to the group GPU list and copy over
    // applicable elements
    for (unsigned int elNum = 0; elNum < affinity_p->numGpus; elNum++)
    {
        if (std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), affinity_p->affinityMasks[elNum].dcgmGpuId) != dcgmGpuIds.end())
        {
            memcpy(gpuAffinity.affinityMasks[gpuAffinity.numGpus].bitmask,
                   affinity_p->affinityMasks[elNum].bitmask, sizeof(unsigned long) * DCGM_AFFINITY_BITMASK_ARRAY_SIZE);
            gpuAffinity.affinityMasks[gpuAffinity.numGpus].dcgmGpuId = affinity_p->affinityMasks[elNum].dcgmGpuId;

            gpuAffinity.numGpus++;
        }
    }
    free(affinity_p); // must delete sample value because it is a type that is malloc'ed

    finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuAffinity, sizeof(dcgmTopology_t));

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetTopologyIO(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmcm_sample_t sample;
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    dcgmTopology_t *topologyPci_p;
    dcgmTopology_t *topologyLwLink_p;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned int> dcgmGpuIds;
    LwcmGroup *pLwcmGrp;
    dcgmTopology_t gpuTopology;

    // always return this struct so that if we return DCGM_ST_NO_DATA that people can still
    // rely on numElements being 0 instead of uninitialized
    gpuTopology.version = dcgmTopology_version;
    gpuTopology.numElements = 0;

    if (pCmd->opmode() != lwcm::OPERATION_GROUP_ENTITIES)
    {
        PRINT_ERROR("", "GET_TOPOLOGY_INFO_IO only works on groupIds");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Get topology Err: Group ID not specified");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }
    else
        groupId = pCmd->id();

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    dcgmReturn = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    for (int i = 0; i < (int)entities.size(); i++)
    {
        /* Only consider GPUs */
        if (entities[i].entityGroupId != DCGM_FE_GPU)
            continue;
        
        dcgmGpuIds.push_back(entities[i].entityId);
    }

    if (dcgmGpuIds.size() < 1)
    {
        PRINT_DEBUG("%d", "No GPUs in group %d", groupId);
        finalizeCmd(pCmd, DCGM_ST_NO_DATA, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    // retrieve the latest sample of PCI topology information
    dcgmReturn = mpCacheManager->GetLatestSample(DCGM_FE_GPU, dcgmGpuIds[0], DCGM_FI_GPU_TOPOLOGY_PCI, 
                                                 &sample, 0);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: unable to retrieve topology information");
        finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    topologyPci_p = (dcgmTopology_t *) sample.val.blob;

    // retrieve the latest sample of LWLINK topology information
    dcgmReturn = mpCacheManager->GetLatestSample(DCGM_FE_GPU, dcgmGpuIds[0], DCGM_FI_GPU_TOPOLOGY_LWLINK, 
                                                 &sample, 0);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: unable to retrieve topology information");
        finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    topologyLwLink_p = (dcgmTopology_t *) sample.val.blob;
    
    // now run through the topology list comparing it to the group GPU list and copy over
    // applicable elements
    for (unsigned int elNum = 0; elNum < topologyPci_p->numElements; elNum++)
    {
        // PCI is the master here as all GPUs will have *some* PCI relationship
        // only peek info the LWLINK topology info if we've found a match for PCI
        if (std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), topologyPci_p->element[elNum].dcgmGpuA) != dcgmGpuIds.end() &&
            std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), topologyPci_p->element[elNum].dcgmGpuB) != dcgmGpuIds.end())   // both gpus are in our list
        {
            gpuTopology.element[gpuTopology.numElements].dcgmGpuA = topologyPci_p->element[elNum].dcgmGpuA;
            gpuTopology.element[gpuTopology.numElements].dcgmGpuB = topologyPci_p->element[elNum].dcgmGpuB;
            gpuTopology.element[gpuTopology.numElements].path = topologyPci_p->element[elNum].path;
            gpuTopology.element[gpuTopology.numElements].AtoBLwLinkIds = 0; // set to zero just in case there is no LWLINK
            gpuTopology.element[gpuTopology.numElements].BtoALwLinkIds = 0; // set to zero just in case there is no LWLINK
            for (unsigned int lwLinkElNum = 0; lwLinkElNum < topologyLwLink_p->numElements; lwLinkElNum++)
            {
                if (topologyLwLink_p->element[lwLinkElNum].dcgmGpuA == topologyPci_p->element[elNum].dcgmGpuA &&
                    topologyLwLink_p->element[lwLinkElNum].dcgmGpuB == topologyPci_p->element[elNum].dcgmGpuB)
                {
                    gpuTopology.element[gpuTopology.numElements].path = (dcgmGpuTopologyLevel_t)((int)gpuTopology.element[gpuTopology.numElements].path | 
                                                                        (int)topologyLwLink_p->element[lwLinkElNum].path);
                    gpuTopology.element[gpuTopology.numElements].AtoBLwLinkIds = topologyLwLink_p->element[lwLinkElNum].AtoBLwLinkIds;
                    gpuTopology.element[gpuTopology.numElements].BtoALwLinkIds = topologyLwLink_p->element[lwLinkElNum].BtoALwLinkIds;
                }
            }
            gpuTopology.numElements++;
        }
    }

    free(topologyPci_p); // must free the sample blob that was malloc'ed
    free(topologyLwLink_p); // must free the sample blob that was malloc'ed

    finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void*)&gpuTopology, sizeof(dcgmTopology_t));

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessModuleCommandWrapper(lwcm::Command *pCmd, bool *pIsComplete, 
                                                                dcgm_connection_id_t connectionId,
                                                                dcgm_request_id_t requestId)
{
    dcgm_module_command_header_t *moduleCommand;
    dcgmReturn_t dcgmReturn;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "MODULE_COMMAND was missing arg or blob");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;

        return DCGM_ST_OK;
    }

    moduleCommand = (dcgm_module_command_header_t *)pCmd->mutable_arg(0)->blob().c_str();
    if(moduleCommand->length != pCmd->mutable_arg(0)->blob().size())
    {
        PRINT_ERROR("%u %u", "Got size mismatch with module command. %u != %u", 
                    moduleCommand->length, (unsigned int)pCmd->mutable_arg(0)->blob().size());
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_BADPARAM;
    }

    /* We are the only authority on the connectionId. Overwrite what the remote client put */
    moduleCommand->connectionId = connectionId;
    
    /* Use the passed-in request ID if the message's request ID hasn't been set yet */
    if(!moduleCommand->requestId)
        moduleCommand->requestId = requestId;

    dcgmReturn = ProcessModuleCommand(moduleCommand);

    /* Set the returned blob to the response */
    pCmd->mutable_arg(0)->set_blob(moduleCommand, moduleCommand->length);

    pCmd->set_status(dcgmReturn);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessCreateFakeEntities(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;

    if(pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from CREATE_FAKE_ENTITIES");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmCreateFakeEntities_t *createFakeEntities = (dcgmCreateFakeEntities_t *)pCmd->arg(0).blob().c_str();

    if(createFakeEntities->version != dcgmCreateFakeEntities_version)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    for (unsigned int i = 0; i < createFakeEntities->numToCreate; i++)
    {
        if (createFakeEntities->entityGroupId == DCGM_FE_GPU)
        {
            createFakeEntities->entityId[i] = mpCacheManager->AddFakeGpu();
            if (createFakeEntities->entityId[i] == DCGM_GPU_ID_BAD)
            {
                PRINT_ERROR("", "Got bad fake gpuId DCGM_GPU_ID_BAD from cache manager");
                pCmd->set_status(DCGM_ST_GENERIC_ERROR);
                *pIsComplete = true;
                return DCGM_ST_GENERIC_ERROR;
            }
        }
        else if (createFakeEntities->entityGroupId == DCGM_FE_SWITCH)
        {
            createFakeEntities->entityId[i] = mpCacheManager->AddFakeLwSwitch();
            if(createFakeEntities->entityId[i] == DCGM_GPU_ID_BAD)
            {
                PRINT_ERROR("", "Got bad fake lwSwitch ID DCGM_GPU_ID_BAD from cache manager");
                pCmd->set_status(DCGM_ST_GENERIC_ERROR);
                *pIsComplete = true;
                return DCGM_ST_GENERIC_ERROR;
            }
        }
        else
        {
            PRINT_ERROR("%u", "CREATE_FAKE_ENTITIES got unhandled eg %u", 
                        createFakeEntities->entityGroupId);
            pCmd->set_status(DCGM_ST_NOT_SUPPORTED);
            *pIsComplete = true;
            return DCGM_ST_NOT_SUPPORTED;
        }
    }

    pCmd->set_status(DCGM_ST_OK);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetLwLinkLinkStatus(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;

    if(pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from GET_LWLINK_LINK_STATUS");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmLwLinkStatus_v2 lwLinkStatus2;
    dcgmLwLinkStatus_v1 lwLinkStatus1;

    switch (pCmd->arg(0).blob().size())
    {
        case sizeof(dcgmLwLinkStatus_v1):
            memcpy(&lwLinkStatus1, pCmd->arg(0).blob().c_str(), sizeof(lwLinkStatus1));

            if (lwLinkStatus1.version != dcgmLwLinkStatus_version1)
            {
                pCmd->set_status(DCGM_ST_VER_MISMATCH);
                *pIsComplete = true;
                return DCGM_ST_VER_MISMATCH;
            }
            
            ret = mpCacheManager->PopulateLwLinkLinkStatus(lwLinkStatus1);
            
            /* Set the response blob */
            pCmd->mutable_arg(0)->set_blob(&lwLinkStatus1, sizeof(lwLinkStatus1));
            break;

        case sizeof(dcgmLwLinkStatus_v2):
            memcpy(&lwLinkStatus2, pCmd->arg(0).blob().c_str(), sizeof(lwLinkStatus2));
            if (lwLinkStatus2.version != dcgmLwLinkStatus_version2)
            {
                pCmd->set_status(DCGM_ST_VER_MISMATCH);
                *pIsComplete = true;
                return DCGM_ST_VER_MISMATCH;
            }

            ret = mpCacheManager->PopulateLwLinkLinkStatus(lwLinkStatus2);

            /* Set the response blob */
            pCmd->mutable_arg(0)->set_blob(&lwLinkStatus2, sizeof(lwLinkStatus2));
            break;

        default:
            pCmd->set_status(DCGM_ST_VER_MISMATCH);
            *pIsComplete = true;
            return DCGM_ST_VER_MISMATCH;
    }



    pCmd->set_status(ret);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessSetLwLinkLinkStatus(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;

    if(pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from SET_LWLINK_LINK_STATUS");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmSetLwLinkLinkState_v1 linkState;

    if(pCmd->arg(0).blob().size() != sizeof(linkState))
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }
    /* Make a local copy of the request so we're not messing with protobuf memory */
    memcpy(&linkState, pCmd->arg(0).blob().c_str(), sizeof(linkState));

    if(linkState.version != dcgmSetLwLinkLinkState_version1)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    ret = mpCacheManager->SetEntityLwLinkLinkState(linkState.entityGroupId, linkState.entityId, 
                                                   linkState.linkId, linkState.linkState);

    pCmd->set_status(ret);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessModuleBlacklist(lwcm::Command *pCmd)
{
    dcgmReturn_t ret;

    if(pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from MODULE_BLACKLIST");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmModuleBlacklist_v1 msg;

    if(pCmd->arg(0).blob().size() != sizeof(msg))
    {
        PRINT_ERROR("", "MODULE_BLACKLIST size mismatch");
        return DCGM_ST_VER_MISMATCH;
    }
    /* Make a local copy of the request so we're not messing with protobuf memory */
    memcpy(&msg, pCmd->arg(0).blob().c_str(), sizeof(msg));

    if(msg.version != dcgmModuleBlacklist_version1)
    {
        PRINT_ERROR("%X %X", "MODULE_BLACKLIST version mismatch x%X != x%X", 
                    msg.version, dcgmModuleBlacklist_version1);
        return DCGM_ST_VER_MISMATCH;
    }

    if(msg.moduleId <= DcgmModuleIdCore || msg.moduleId >= DcgmModuleIdCount)
    {
        PRINT_ERROR("%u", "Invalid moduleId %u", msg.moduleId);
        return DCGM_ST_BADPARAM;
    }

    /* Lock the host engine so states don't change under us */
    Lock();

    /* React to this based on the current module status */
    switch(m_modules[msg.moduleId].status)
    {
        case DcgmModuleStatusNotLoaded:
            break; /* Will blacklist below */
        
        case DcgmModuleStatusBlacklisted:
            Unlock();
            PRINT_DEBUG("%u", "Module ID %u is already blacklisted.", msg.moduleId);
            return DCGM_ST_OK;
        
        case DcgmModuleStatusFailed:
            PRINT_DEBUG("%u", "Module ID %u already failed to load. Setting to blacklisted.", msg.moduleId);
            break;

        case DcgmModuleStatusLoaded:
            Unlock();
            PRINT_WARNING("%u", "Could not blacklist module %u that was already loaded.", msg.moduleId);
            return DCGM_ST_IN_USE;

        /* Not adding a default case here so adding future states will cause a compiler error */
    }

    PRINT_INFO("%u", "Blacklisting module %u", msg.moduleId);
    m_modules[msg.moduleId].status = DcgmModuleStatusBlacklisted;

    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessModuleGetStatuses(lwcm::Command *pCmd)
{
    dcgmReturn_t ret;
    unsigned int moduleId;

    if(pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from MODULE_GET_STATUSES");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmModuleGetStatuses_v1 msg;

    if(pCmd->arg(0).blob().size() != sizeof(msg))
    {
        PRINT_ERROR("", "MODULE_GET_STATUSES size mismatch");
        return DCGM_ST_VER_MISMATCH;
    }
    /* Make a local copy of the request so we're not messing with protobuf memory */
    memcpy(&msg, pCmd->arg(0).blob().c_str(), sizeof(msg));

    if(msg.version != dcgmModuleGetStatuses_version)
    {
        PRINT_ERROR("%X %X", "MODULE_BLACKLIST version mismatch x%X != x%X", 
                    msg.version, dcgmModuleGetStatuses_version);
        return DCGM_ST_VER_MISMATCH;
    }

    /* Note: not locking here because we're not looking at anything sensitive */

    msg.numStatuses = 0;
    for(moduleId = DcgmModuleIdCore; 
        moduleId < DcgmModuleIdCount && msg.numStatuses < DCGM_MODULE_STATUSES_CAPACITY; 
        moduleId++)
    {
        msg.statuses[msg.numStatuses].id = m_modules[moduleId].id;
        msg.statuses[msg.numStatuses].status = m_modules[moduleId].status;
        msg.numStatuses++;
    }

    /* Set the response blob */
    pCmd->mutable_arg(0)->set_blob(&msg, sizeof(msg));

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetMultipleLatestValues(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    dcgmGetMultipleLatestValues_t msg;
    unsigned int i;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned short>fieldIds;

    *pIsComplete = true; /* Just set this once */

    if(pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Payload missing from from GET_MULTIPLE_LATEST_VALUES");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    /* Make a copy of the message since we're going to modify it */
    memcpy(&msg, pCmd->arg(0).blob().c_str(), pCmd->arg(0).blob().size());

    /* Colwert the entity group to a list of entities */
    if(!msg.entitiesCount)
    {
        unsigned int groupId = (uintptr_t)msg.groupId;

        /* If this is a special group ID, colwert it to a real one */
        ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
        if(ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %p", "Got st %d from verifyAndUpdateGroupId. groupId %p", 
                        ret, msg.groupId);
            pCmd->set_status(DCGM_ST_OK);
            return ret;
        }

        ret = mpGroupManager->GetGroupEntities(0, groupId, entities);
        if(ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %p", "Got st %d from GetGroupEntities. groupId %p", 
                        ret, msg.groupId);
            pCmd->set_status(DCGM_ST_OK);
            return ret;
        }
    }
    else
    {
        /* Use the list from the message */
        entities.insert(entities.end(), &msg.entities[0], 
                            &msg.entities[msg.entitiesCount]);
    }

    /* Colwert the fieldGroupId to a list of field IDs */
    if(!msg.fieldIdCount)
    {
        ret = mpFieldGroupManager->GetFieldGroupFields(msg.fieldGroupId, fieldIds);
        if(ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %p", "Got st %d from GetFieldGroupFields. fieldGroupId %p", 
                        ret, msg.fieldGroupId);
            pCmd->set_status(DCGM_ST_OK);
            return ret;
        }
    }
    else
    {
        /* Use the list from the message */
        fieldIds.insert(fieldIds.end(), &msg.fieldIds[0], 
                        &msg.fieldIds[msg.fieldIdCount]);
    }

    /* Create the fvBuffer after we know how many field IDs we'll be retrieving */
    size_t initialCapacity = FVBUFFER_GUESS_INITIAL_CAPACITY(entities.size(), fieldIds.size());
    DcgmFvBuffer fvBuffer(initialCapacity);

    /* Make a batch request to the cache manager to fill a fvBuffer with all of the values */
    if(msg.flags & DCGM_FV_FLAG_LIVE_DATA)
        ret = mpCacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    else
        ret = mpCacheManager->GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);
    if(ret != DCGM_ST_OK)
    {
        pCmd->set_status(ret);
        return ret;    
    }

    const char *fvBufferBytes = fvBuffer.GetBuffer();
    size_t bufferSize = 0, elementCount = 0;

    fvBuffer.GetSize(&bufferSize, &elementCount);

    if(!fvBufferBytes || !bufferSize)
    {
        PRINT_ERROR("%p %d", "Unexpected fvBuffer %p, fvBufferBytes %d", 
                    fvBufferBytes, (int)bufferSize);
        ret = DCGM_ST_GENERIC_ERROR;
        pCmd->set_status(ret);
        return ret;
    }

    /* Set pCmd->blob with the contents of the FV buffer */
    pCmd->mutable_arg(0)->set_blob(fvBufferBytes, bufferSize);
    pCmd->set_status(ret);
    return ret;

}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessGetFieldSummary(lwcm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from GET_FIELD_SUMMARY");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return ret;
    }

    dcgmFieldSummaryRequest_v1 fieldSummary;

    if (pCmd->arg(0).blob().size() != sizeof(fieldSummary))
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return ret;
    }

    // Make a local copy of the request to avoid stomping on memory
    memcpy(&fieldSummary, pCmd->arg(0).blob().c_str(), sizeof(fieldSummary));

    if (fieldSummary.version != dcgmFieldSummaryRequest_version1)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return ret;
    }

    dcgm_field_meta_p fm = DcgmFieldGetById(fieldSummary.fieldId);

    if (fm == 0)
    {
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return ret;
    }

    int numSummaryTypes = 0;
    timelib64_t startTime = fieldSummary.startTime;
    timelib64_t endTime = fieldSummary.endTime;
    dcgm_field_entity_group_t entityGroupId = static_cast<dcgm_field_entity_group_t>(fieldSummary.entityGroupId);

    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    memset(&summaryTypes, 0, sizeof(summaryTypes));

    for (int i = 0; i < static_cast<int>(DcgmcmSummaryTypeSize); i++)
    {
        if ((fieldSummary.summaryTypeMask & 0x1 << i) != 0)
        {
            summaryTypes[numSummaryTypes] = static_cast<DcgmcmSummaryType_t>(i);
            numSummaryTypes++;
        }
    }
                    
    fieldSummary.response.fieldType = fm->fieldType;
    fieldSummary.response.summaryCount = numSummaryTypes;

    switch (fm->fieldType)
    {
        case DCGM_FT_DOUBLE:
            {
                double dSummaryValues[DcgmcmSummaryTypeSize];
                
                ret = mpCacheManager->GetFp64SummaryData(entityGroupId, fieldSummary.entityId,
                                                         fieldSummary.fieldId, numSummaryTypes, summaryTypes,
                                                         dSummaryValues, startTime, endTime, NULL, NULL);
                if (ret == DCGM_ST_OK)
                {
                    // Copy the values back into the response
                    for (int i = 0; i < numSummaryTypes; i++)
                        fieldSummary.response.values[i].fp64 = dSummaryValues[i];
                }

                break;
            }
        case DCGM_FT_INT64:
            {
                long long iSummaryValues[DcgmcmSummaryTypeSize];
    
                ret = mpCacheManager->GetInt64SummaryData(entityGroupId, fieldSummary.entityId,
                                                          fieldSummary.fieldId, numSummaryTypes, summaryTypes,
                                                          iSummaryValues, startTime, endTime, NULL, NULL);

                if (ret == DCGM_ST_OK)
                {
                    // Copy the values back into the response
                    for (int i = 0; i < numSummaryTypes; i++)
                        fieldSummary.response.values[i].i64 = iSummaryValues[i];
                }

                break;
            }
        default:
            {
                // We only support this call for int64 and doubles 
                ret = DCGM_ST_FIELD_UNSUPPORTED_BY_API;
                break;
            }
    }

    if (ret == DCGM_ST_OK)
    {
        /* Set the response blob */
        pCmd->mutable_arg(0)->set_blob(&fieldSummary, sizeof(fieldSummary));
    }

    pCmd->set_status(ret);
    *pIsComplete = true;

    return ret;
}

/*****************************************************************************/
int LwcmHostEngineHandler::ProcessRequest(lwcm::Command *pCmd, bool *pIsComplete, 
                                          LwcmServerConnection* pConnection, 
                                          dcgm_request_id_t requestId)
{
    int ret = 0;
    dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE;
    
    /* Only set the connectionId if we're actually going to clean up requests
       You can still get the connection ID from pConnection->GetConnectionId() */
    if(pConnection && !pConnection->GetPersistAfterDisconnect())
    {
        connectionId = pConnection->GetConnectionId();
    }
    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    PRINT_DEBUG("%d %u", "processing request of type %d for connectionId %u", 
                pCmd->cmdtype(), connectionId);

    switch (pCmd->cmdtype())
    {
        case lwcm::CLIENT_LOGIN:
        {
            ret = ProcessClientLogin(pCmd, pIsComplete, pConnection);
            break;
        }

        case lwcm::GROUP_CREATE:
        {
            ret = ProcessGroupCreate(pCmd, pIsComplete, pConnection, connectionId);
            break;
        }

        case lwcm::GROUP_ADD_DEVICE:    /* fall-through is intentional */
        case lwcm::GROUP_REMOVE_DEVICE:
        {
            ret = ProcessAddRemoveGroup(pCmd, pIsComplete, pConnection, connectionId);
            break;
        }

        case lwcm::GROUP_DESTROY:
        {
            ret = ProcessGroupDestroy(pCmd, pIsComplete, pConnection, connectionId);
            break;
        }

        case lwcm::GROUP_INFO:
        {
            ret = ProcessGroupInfo(pCmd, pIsComplete, connectionId);
            break;
        }

        case lwcm::GROUP_GETALL_IDS:
        {
            ret = ProcessGroupGetallIds(pCmd, pIsComplete, connectionId);
            break;
        }

        case lwcm::DISCOVER_DEVICES:
        {
            ret = ProcessDiscoverDevices(pCmd, pIsComplete);
            break;
        }

        case lwcm::GET_ENTITY_LIST:
        {
            ret = ProcessGetEntityList(pCmd, pIsComplete);
            break;
        }

        case lwcm::SAVE_CACHED_STATS:

            ret = ProcessSaveCachedStats(pCmd, pIsComplete);
            break;

        case lwcm::LOAD_CACHED_STATS:

            ret = ProcessLoadCachedStats(pCmd, pIsComplete);
            break;

        case lwcm::INJECT_FIELD_VALUE:
        {
            ret = ProcessInjectFieldValue(pCmd, pIsComplete);
            break;
        }

        case lwcm::GET_FIELD_LATEST_VALUE:

            ret = ProcessGetFieldLatestValue(pCmd, pIsComplete);
            break;

        case lwcm::GET_FIELD_MULTIPLE_VALUES:
        {
            ret = ProcessGetFieldMultipleValues(pCmd, pIsComplete);
            break;
        }

        case lwcm::WATCH_FIELD_VALUE:
        {
            ret = ProcessWatchFieldValue(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case lwcm::UNWATCH_FIELD_VALUE:
        {
            ret = ProcessUnwatchFieldValue(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case lwcm::UPDATE_ALL_FIELDS:
        {
            ret = ProcessUpdateAllFields(pCmd, pIsComplete);
            break;
        }

        case lwcm::CACHE_MANAGER_FIELD_INFO:
        {
            ret = ProcessCacheManagerFieldInfo(pCmd, pIsComplete);
            break;
        }

        case lwcm::WATCH_FIELDS:
        {
            ret = ProcessWatchFields(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case lwcm::UNWATCH_FIELDS:
        {
            ret = ProcessUnwatchFields(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case lwcm::GET_PID_INFORMATION:
        {
            ret = ProcessGetPidInfo(pCmd, pIsComplete);
            break;
        }
        
        case lwcm::FIELD_GROUP_CREATE:
        {
            ret = ProcessFieldGroupCreate(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case lwcm::FIELD_GROUP_DESTROY:
        {
            ret = ProcessFieldGroupDestroy(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case lwcm::FIELD_GROUP_GET_ONE:
        {
            ret = ProcessFieldGroupGetOne(pCmd, pIsComplete);
            break;
        }

        case lwcm::FIELD_GROUP_GET_ALL:
        {
            ret = ProcessFieldGroupGetAll(pCmd, pIsComplete);
            break;
        }

        case lwcm::WATCH_PREDEFINED:
        {
            ret = ProcessWatchRedefined(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case lwcm::JOB_START_STATS:
        {
            ret = ProcessJobStartStats(pCmd, pIsComplete);
            break;            
        }
            
        case lwcm::JOB_STOP_STATS:
        {
            ret = ProcessJobStopStats(pCmd, pIsComplete);
            break;            
        }
        
        case lwcm::JOB_REMOVE:
        {
            ret = ProcessJobRemove(pCmd, pIsComplete);
            break;
        }

        case lwcm::JOB_REMOVE_ALL:
        {
            dcgmReturn_t ret = JobRemoveAll();
            pCmd->set_status(ret);
            *pIsComplete = true;
            break;
        }

        case lwcm::JOB_GET_INFO:
        {
            ret = ProcessJobGetInfo(pCmd, pIsComplete);
            break;
        }

        case lwcm::GET_TOPOLOGY_INFO_AFFINITY:
        {
            ret = ProcessGetTopologyAffinity(pCmd, pIsComplete);
            break;

        }
       
        case lwcm::GET_TOPOLOGY_INFO_IO:
        {
            ret = ProcessGetTopologyIO(pCmd, pIsComplete);
            break;
        }
        
        case lwcm::SELECT_GPUS_BY_TOPOLOGY:
        {
            ret = ProcessSelectGpusByTopology(pCmd, pIsComplete);
            break;
        }

        case lwcm::GET_FIELD_SUMMARY:
        {
            ret = ProcessGetFieldSummary(pCmd, pIsComplete);
            break;
        }

        case lwcm::MODULE_COMMAND:
        {
            ret = ProcessModuleCommandWrapper(pCmd, pIsComplete, connectionId,
                                              requestId);
            break;
        }

        case lwcm::CREATE_FAKE_ENTITIES:
        {
            ret = ProcessCreateFakeEntities(pCmd, pIsComplete);
            break;
        }

        case lwcm::GET_LWLINK_LINK_STATUS:
        {
            ret = ProcessGetLwLinkLinkStatus(pCmd, pIsComplete);
            break;
        }

        case lwcm::GET_MULTIPLE_LATEST_VALUES:
        {
            ret = ProcessGetMultipleLatestValues(pCmd, pIsComplete);
            break;
        }

        case lwcm::SET_LWLINK_LINK_STATUS:
        {
            ret = ProcessSetLwLinkLinkStatus(pCmd, pIsComplete);
            break;
        }

        case lwcm::MODULE_BLACKLIST:
        {
            dcgmReturn_t ret = ProcessModuleBlacklist(pCmd);
            pCmd->set_status(ret);
            *pIsComplete = true;
            break;
        }

        case lwcm::MODULE_GET_STATUSES:
        {
            dcgmReturn_t ret = ProcessModuleGetStatuses(pCmd);
            pCmd->set_status(ret);
            *pIsComplete = true;
            break;
        }

        default:
            // Unknown command
            PRINT_ERROR("", "Unknown command.");
            pCmd->set_status(DCGM_ST_BADPARAM);
            break;
    }

    return ret;
}

/*****************************************************************************/
void LwcmHostEngineHandler::finalizeCmd(lwcm::Command *pCmd, dcgmReturn_t cmdStatus, bool *&pIsComplete, void* returnArg, size_t returnArgSize)
{
    pCmd->add_arg()->set_blob(returnArg, returnArgSize);
    pCmd->set_status(cmdStatus);
    *pIsComplete = true;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::SendRawMessageToEmbeddedClient(unsigned int msgType, 
                                                           dcgm_request_id_t requestId,
                                                           void *msgData, int msgLength)
{

    watchedRequests_t::iterator requestIt;
    
    /* Embedded client */
    if(!requestId)
    {
        PRINT_ERROR("", "Can't SendRawMessageToEmbeddedClient() with 0 requestId");
        return DCGM_ST_GENERIC_ERROR;
    }

    Lock();

    requestIt = m_watchedRequests.find(requestId);
    if(requestIt == m_watchedRequests.end())
    {
        PRINT_ERROR("%u", "SendRawMessageToEmbeddedClient unable to find requestId %u", 
                    requestId);
        Unlock();
        return DCGM_ST_BADPARAM;
    }

    /* ProcessMessage is expecting an allocated message */
    LwcmMessage *msg = new LwcmMessage();
    msg->UpdateMsgHdr(msgType, requestId, DCGM_PROTO_ST_SUCCESS, msgLength);
    msg->UpdateMsgContent((char *)msgData, msgLength);

    requestIt->second->ProcessMessage(msg);
    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::SendRawMessageToClient(dcgm_connection_id_t connectionId, 
                                                           unsigned int msgType, 
                                                           dcgm_request_id_t requestId,
                                                           void *msgData, int msgLength)
{
    if(!connectionId)
    {
        return SendRawMessageToEmbeddedClient(msgType, requestId, 
                                              msgData, msgLength);
    }

    /* Remote case */
    if(!mpServerObj)
    {
        PRINT_ERROR("", "mpServerObj was NULL");
        return DCGM_ST_GENERIC_ERROR;
    }

    return mpServerObj->SendRawMessageToClient(connectionId, msgType, 
                                               requestId, msgData, msgLength);
}


/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::ProcessModuleCommand(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;

    if(moduleCommand->moduleId <= DcgmModuleIdCore || moduleCommand->moduleId >= DcgmModuleIdCount)
    {
        PRINT_ERROR("%u", "Invalid module id: %u", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }

    /* Is the module loaded? */
    if(!m_modules[moduleCommand->moduleId].ptr)
    {
        dcgmReturn = LoadModule(moduleCommand->moduleId);
        if(dcgmReturn != DCGM_ST_OK)
            return dcgmReturn;
    }

    /* Dispatch the message */
    dcgmReturn = m_modules[moduleCommand->moduleId].ptr->ProcessMessage(moduleCommand);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    return mpCacheManager->GetCacheManagerFieldInfo(fieldInfo);
}

/*****************************************************************************/
void LwcmHostEngineHandler::OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection)
{
    if(mpGroupManager)
        mpGroupManager->OnConnectionRemove(connectionId, pConnection);
    if(mpFieldGroupManager)
        mpFieldGroupManager->OnConnectionRemove(connectionId, pConnection);
    /* Call the cache manager last since the rest of the modules refer to it */
    if(mpCacheManager)
        mpCacheManager->OnConnectionRemove(connectionId, pConnection);
    
    /* Notify each module about the client disconnect */
    for(unsigned int moduleIndex = 0; moduleIndex < DcgmModuleIdCount; moduleIndex++)
    {
        if(m_modules[moduleIndex].ptr)
            m_modules[moduleIndex].ptr->OnClientDisconnect(connectionId);
    }
}

/*****************************************************************************/    
int LwcmHostEngineHandler::HandleCommands(vector<lwcm::Command *> *pVecCmdsToProcess, bool *pIsComplete, 
                                          LwcmServerConnection* pConnection, dcgm_request_id_t requestId)
{

    vector<lwcm::Command *>::iterator cmdIterator;
    lwcm::Command  *pCmd;


    for (cmdIterator = pVecCmdsToProcess->begin(); cmdIterator != pVecCmdsToProcess->end(); ++cmdIterator)
    {
        pCmd = *(cmdIterator);
        (void)ProcessRequest(pCmd, pIsComplete, pConnection, requestId);
        /* Give the caller our timestamp */
        pCmd->set_timestamp(timelib_usecSince1970());
    }

    return 0;
}

/*****************************************************************************/
int LwcmHosEngineServer::SendDataToClient(LwcmProtobuf *protoObj, 
        LwcmServerConnection* pConnection, dcgm_request_id_t requestId, unsigned int msgType)
{
    // LwcmProtobuf protoObj;
    char *msgToSend;
    unsigned int msgLen;    
    LwcmMessage lwcmReply;
    int st;

    protoObj->GetEncodedMessage(&msgToSend, &msgLen);

    // DEBUG_STDOUT("Length of Message to send to Client: " << msgLen);

    /* Ilwoke transport module to send the message */
    lwcmReply.UpdateMsgHdr(msgType, requestId, DCGM_PROTO_ST_SUCCESS, msgLen);
    lwcmReply.UpdateMsgContent(msgToSend, msgLen);

    st = pConnection->SetOutputBuffer(&lwcmReply);
    if (st) {
        return st;
    }

    PRINT_DEBUG("%u %u %X %u", "Sent protobuf message length %u, requestId %u, msgType x%X to connectionId %u", 
                msgLen, requestId, msgType, pConnection->GetConnectionId());

    return 0;
}

/*****************************************************************************/
dcgmReturn_t LwcmHosEngineServer::SendRawMessageToClient(dcgm_connection_id_t connectionId, 
                                                         unsigned int msgType, 
                                                         dcgm_request_id_t requestId,
                                                         void *msgData, int msgLength)
{
    LwcmServerConnection *pConnection = (LwcmServerConnection *)GetConnectionHandler()->GetConnectionEntry(connectionId);
    if(!pConnection)
    {
        PRINT_ERROR("%u %u", "Unable to find connectionId %u. Discarding msgType %u",
                    connectionId, msgType);
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    /* Note: We have to DecrReference after this point or we'll never free pConnection */
    
    LwcmMessage lwcmMessage;
    lwcmMessage.UpdateMsgHdr(msgType, requestId, DCGM_PROTO_ST_SUCCESS, msgLength);
    lwcmMessage.UpdateMsgContent((char *)msgData, msgLength);
    pConnection->SetOutputBuffer(&lwcmMessage);

    pConnection->DecrReference();

    PRINT_DEBUG("%d %u %X %u", "Sent raw message length %d, requestId %u, msgType x%X to connectionId %u", 
                msgLength, requestId, msgType, connectionId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void LwcmHosEngineServer::OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection)
{
    LwcmHostEngineHandler::Instance()->OnConnectionRemove(connectionId, pConnection);
}

/*****************************************************************************/
int LwcmHosEngineServer::OnRequest(dcgm_request_id_t requestId, LwcmServerConnection* pConnection)
{

    LwcmMessage *pMessageRecvd;                     /* Pointer to LWCM Message */
    LwcmRequest *pLwcmServerRequest;                /* Pointer to LWCM Request */
    LwcmProtobuf protoObj;                          /* Protobuf object to send or recv the message */
    vector<lwcm::Command *> vecCmds;                /* To store reference to commands inside the protobuf message */
    bool isComplete;                                /* Flag to determine if the request handling is complete */
    int st;                                         /* Status code */
    int retSt = 0;                                  /* Status code to return from this function. 0 = success. < 0 on error */
    int messageCount;                               /* The number of messages included in this command */

    if (!pConnection) {
        PRINT_ERROR("", "Null pConnection");
        return -1;
    }

    /* No need to pConnection->IncrRef() here as the parent method already does this */

    /**
     * Before processing the request check if the connection is still in 
     * active state. If the connection is not active then don't even proceed 
     * and mark the request as completed. The CompleteRequest will delete the
     * connection bindings if this request is the last entity holding on to the 
     * connection even when the connection is in inactive state.
     */
    if (!pConnection->IsConnectionActive()) 
    {
        PRINT_DEBUG("%u", "Connection %u is inactive", pConnection->GetConnectionId());
        retSt = -1;
        goto CLEANUP;
    }

    pLwcmServerRequest = pConnection->GetRequest(requestId);
    if (NULL == pLwcmServerRequest) 
    {
        PRINT_ERROR("%d %u", "Failed to get Info for request id %u, connectionId %u", 
                    (unsigned int)requestId, pConnection->GetConnectionId());
        retSt = -1;
        goto CLEANUP;
    }    

    messageCount = pLwcmServerRequest->MessageCount();
    if (messageCount != 1) 
    {
        PRINT_ERROR("%d %u", "Error: Expected single message for the request. Got %d for connectionId %u", 
                    messageCount, pConnection->GetConnectionId());
        retSt = -1;
        goto CLEANUP;
    }

    // Get the message received corresponding to the request id
    pMessageRecvd = pLwcmServerRequest->GetNextMessage();
    if (NULL == pMessageRecvd) 
    {
        PRINT_ERROR("%u %u", "Failed to get message for request id %u for connectionId %u", 
                    (unsigned int)requestId, pConnection->GetConnectionId());
        retSt = -1;
        goto CLEANUP;
    }

    retSt = protoObj.ParseRecvdMessage((char *)pMessageRecvd->GetContent(), pMessageRecvd->GetLength(), &vecCmds);
    if(retSt != 0)
    {
        PRINT_ERROR("%d %u", "ParseRecvdMessage returned %d for connectionId %u", retSt, pConnection->GetConnectionId());
        goto CLEANUP;
    }

    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pMessageRecvd;
    pMessageRecvd = NULL;

    if (0 != LwcmHostEngineHandler::Instance()->HandleCommands(&vecCmds, &isComplete, pConnection, requestId)) 
    {
        retSt = -1;
        goto CLEANUP;
    }

    retSt = SendDataToClient(&protoObj, pConnection, requestId, DCGM_MSG_PROTO_RESPONSE);

CLEANUP:
    pConnection->CompleteRequest(requestId);
    return retSt;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::WatchHostEngineFields(void)
{

    std::vector<unsigned short>fieldIds;
    dcgmReturn_t dcgmReturn;
    DcgmWatcher watcher(DcgmWatcherTypeHostEngine, DCGM_CONNECTION_ID_NONE);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_DEV_ECC_LWRRENT); /* Can really only change once per driver reload. LWML caches this so it's virtually a no-op */

    fieldIds.push_back(DCGM_FI_DEV_VIRTUAL_MODE); /* Used by dcgmDeviceAttributes_t */
    fieldIds.push_back(DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS); /* Used by dcgmVgpuDeviceAttributes_t */
    fieldIds.push_back(DCGM_FI_DEV_VGPU_INSTANCE_IDS); /* Used by dcgmVgpuDeviceAttributes_t */

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_30SEC", fieldIds, &mFieldGroup30Sec, watcher);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "AddFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Max number of entries 14400/30 entries
    dcgmReturn = WatchFieldGroup(mpGroupManager->GetAllGpusGroup(), mFieldGroup30Sec, 30000000, 14400.0, 480, watcher);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "WatchFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    fieldIds.clear();
    /* Needed by the scheduler hint APIs */
    fieldIds.push_back(DCGM_FI_GPU_TOPOLOGY_PCI);
    fieldIds.push_back(DCGM_FI_GPU_TOPOLOGY_LWLINK);
    fieldIds.push_back(DCGM_FI_GPU_TOPOLOGY_AFFINITY);
    /* Needed as it is the static info related to GPU attribute associated with vGPU */
    fieldIds.push_back(DCGM_FI_DEV_SUPPORTED_TYPE_INFO);

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_HOURLY", fieldIds, &mFieldGroupHourly, watcher);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "AddFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Max number of entries 14400/3600 entries. Include non-DCGM GPUs
    dcgmReturn = WatchFieldGroupAllGpus(mFieldGroupHourly, 3600000000, 14400.0, 4, 0, watcher);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "WatchFieldGroupAllGpus returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process / job stats fields. Just add the group. The user will watch the fields */
    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_DEV_ACCOUNTING_DATA);
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_TX_THROUGHPUT);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_RX_THROUGHPUT);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL);
    fieldIds.push_back(DCGM_FI_DEV_MEM_COPY_UTIL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_COMPUTE_PIDS);
    fieldIds.push_back(DCGM_FI_DEV_GRAPHICS_PIDS);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_SYNC_BOOST_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL_SAMPLES);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_SBE);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_DBE);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_PENDING);
    fieldIds.push_back(DCGM_FI_DEV_INFOROM_CONFIG_VALID);

    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);

    /* Add Watch for LWLINK flow control CRC Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL);
    /* Add Watch for LWLINK data CRC Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL);
    /* Add Watch for LWLINK Replay Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL);
    /* Add Watch for LWLINK Recovery Error Counter for all the lanes*/
    fieldIds.push_back(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL);
    
    //reliability violation time
    //board violation time
    //low utilization time

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_JOB", fieldIds, &mFieldGroupPidAndJobStats, watcher);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "AddFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static void HostEngineOnGroupEventCB(unsigned int groupId, void *userData)
{
    LwcmHostEngineHandler *hostEngineHandler = (LwcmHostEngineHandler *)userData;

    hostEngineHandler->OnGroupRemove(groupId);
}

/*****************************************************************************/
void LwcmHostEngineHandler::OnGroupRemove(unsigned int groupId)
{
    /* Notify each module about the group removal */
    for(unsigned int moduleIndex = 0; moduleIndex < DcgmModuleIdCount; moduleIndex++)
    {
        if(m_modules[moduleIndex].ptr)
            m_modules[moduleIndex].ptr->OnGroupRemove(groupId);
    }
}

/*****************************************************************************/
void LwcmHostEngineHandler::OnFvUpdates(DcgmFvBuffer *fvBuffer, DcgmWatcherType_t *watcherTypes, 
                                        int numWatcherTypes, void *userData)
{
    static dcgmModuleId_t watcherToModuleMap[DcgmWatcherTypeCount] = 
        {DcgmModuleIdCore, DcgmModuleIdCore, DcgmModuleIdHealth,
         DcgmModuleIdPolicy, DcgmModuleIdCore, DcgmModuleIdCore, DcgmModuleIdLwSwitch};
    
    /* Dispatch each watcher to the corresponding module */
    dcgmModuleId_t destinationModuleId;
    int i;

    for(i = 0; i < numWatcherTypes; i++)
    {
        destinationModuleId = watcherToModuleMap[watcherTypes[i]];
        if(destinationModuleId == DcgmModuleIdCore)
        {
            PRINT_ERROR("%u", "Unhandled watcherType %u can't be dispatched to a module.", 
                        watcherTypes[i]);
            continue;
        }

        if(!m_modules[destinationModuleId].ptr)
        {
            PRINT_DEBUG("%u", "Skipping FV update for moduleId %u that is not loaded.", 
                        destinationModuleId);
            continue;
        }

        /* Module is loaded! Dispatch the callback */
        m_modules[destinationModuleId].ptr->OnFieldValuesUpdate(fvBuffer);
    }
}

/*****************************************************************************/
static void lwHostEngineFvCallback(DcgmFvBuffer *fvBuffer, DcgmWatcherType_t *watcherTypes, 
                                   int numWatcherTypes, void *userData)
{
     LwcmHostEngineHandler *hostEngineHandler = (LwcmHostEngineHandler *)userData;

     hostEngineHandler->OnFvUpdates(fvBuffer, watcherTypes, numWatcherTypes, userData);
}

/*****************************************************************************
 Constructor for LWCM Host Engine Handler
 *****************************************************************************/
LwcmHostEngineHandler::LwcmHostEngineHandler(dcgmOperationMode_t mode)
{
    int index;
    int ret;
    dcgmReturn_t lwcmRet;

    mpServerObj = NULL;
    mpCacheManager = NULL;

    m_nextWatchedRequestId = 1;

    memset(&m_modules, 0, sizeof(m_modules));
    /* Do explicit initialization of the modules */
    for(unsigned int i = 0; i < DcgmModuleIdCount; i++)
    {
        m_modules[i].id = (dcgmModuleId_t)i;
        m_modules[i].status = DcgmModuleStatusNotLoaded;
    }
    /* Core module is always loaded */
    m_modules[DcgmModuleIdCore].status = DcgmModuleStatusLoaded;
    /* Set module filenames */
    m_modules[DcgmModuleIdLwSwitch].filename = "libdcgmmodulelwswitch.so.1";
    m_modules[DcgmModuleIdVGPU].filename = "libdcgmmodulevgpu.so.1";
    m_modules[DcgmModuleIdIntrospect].filename = "libdcgmmoduleintrospect.so.1";
    m_modules[DcgmModuleIdHealth].filename = "libdcgmmodulehealth.so.1";
    m_modules[DcgmModuleIdPolicy].filename = "libdcgmmodulepolicy.so.1";
    m_modules[DcgmModuleIdConfig].filename = "libdcgmmoduleconfig.so.1";
    m_modules[DcgmModuleIdDiag].filename = "libdcgmmodulediag.so.1";
    m_modules[DcgmModuleIdProfiling].filename = "libdcgmmoduleprofiling.so.1";

    /* Make sure we can catch any signal sent to threads by LwcmThread */
    LwcmThread::InstallSignalHandler();

    if (LWML_SUCCESS != lwmlInit()) {
        throw std::runtime_error("Error: Failed to initialize LWML");
    }

    char driverVersion[80];
    lwmlSystemGetDriverVersion(driverVersion, 80);
    if (strcmp(driverVersion, DCGM_MIN_DRIVER_VERSION) < 0)
    {
        throw std::runtime_error("Driver " + std::string(driverVersion) + " is unsupported. Must be at least "
                                 + std::string(DCGM_MIN_DRIVER_VERSION) + ".");
    }

    lwosInitializeCriticalSection(&m_lock);

    ret = DcgmFieldsInit();
    if (ret) {
        std::stringstream ss;
        ss << "DCGM Fields Init Failed. Error: " << ret; 
        throw std::runtime_error(ss.str());
    }

    unsigned int lwmlDeviceCount = 0;
    lwmlReturn_t lwmlSt = lwmlDeviceGetCount(&lwmlDeviceCount);
    if (lwmlSt != LWML_SUCCESS)
    {
        std::stringstream ss;
        ss << "Unable to get the LWML device count. LWML Error: " << lwmlSt;
        throw std::runtime_error(ss.str());
    }
    
    if(lwmlDeviceCount > DCGM_MAX_NUM_DEVICES)
    {
        std::stringstream ss;
        ss << "DCGM only supports up to " << DCGM_MAX_NUM_DEVICES << " GPUs. " 
           << lwmlDeviceCount << " GPUs were found in the system.";
        throw std::runtime_error(ss.str());
    }
    else if (lwmlDeviceCount == 0)
    {
        throw std::runtime_error("DCGM Failed to find any GPUs on the node.");
    }

    mpCacheManager = new DcgmCacheManager();

    /* Don't do anything before you call mpCacheManager->Init() */

    if (mode == DCGM_OPERATION_MODE_AUTO)
    {
        ret = mpCacheManager->Init(0, 86400.0);
        if (ret)
        {
            std::stringstream ss;
            ss << "CacheManager Init Failed. Error: " << ret; 
            throw std::runtime_error(ss.str());
        }
    } else {
        ret = mpCacheManager->Init(1, 14400.0);
        if (ret)
        {
            std::stringstream ss;
            ss << "CacheManager Init Failed. Error: " << ret; 
            throw std::runtime_error(ss.str());
        }        
    }

    lwcmRet = mpCacheManager->SubscribeForFvUpdates(lwHostEngineFvCallback, this);
    if(lwcmRet != DCGM_ST_OK)
    {
        throw std::runtime_error("DCGM was unable to subscribe for cache manager updates.");
    }

    /* Initialize the group manager before we add our default watches */
    mpGroupManager = new LwcmGroupManager(mpCacheManager);
    mpGroupManager->SubscribeForGroupEvents(HostEngineOnGroupEventCB, this);

    mpFieldGroupManager = new DcgmFieldGroupManager();

    /* Watch internal fields before we start the cache manager update thread */
    lwcmRet = WatchHostEngineFields();
    if(lwcmRet)
    {
        throw std::runtime_error("WatchHostEngineFields failed.");
    }

    /* Start the cache manager update thread */
    ret = mpCacheManager->Start();
    if(ret)
    {
        std::stringstream ss;
        ss << "CacheManager Start Failed. Error: " << ret; 
        throw std::runtime_error(ss.str());        
    }

    /* Wait for a round of updates to occur so that we can safely query values */
    ret = mpCacheManager->UpdateAllFields(1);
    if(ret)
    {
        std::stringstream ss;
        ss << "CacheManager UpdateAllFields. Error: " << ret;
        throw std::runtime_error(ss.str());
    }
}

/*****************************************************************************
 Destructor for LWCM Host Engine Handler
 *****************************************************************************/
LwcmHostEngineHandler::~LwcmHostEngineHandler() 
{
    /* Make sure that server is stopped first so that no new connections or 
     * requests are accepted by the HostEngine.
     * Always keep this first */

    if (NULL != mpServerObj) {
        mpServerObj->StopServer();
    }

    /* Free sub-modules before we unload core modules */
    for(int i = 0; i < DcgmModuleIdCount; i++)
    {
        if(m_modules[i].ptr && m_modules[i].freeCB)
        {
            m_modules[i].freeCB(m_modules[i].ptr);
            m_modules[i].ptr = 0;
        }

        m_modules[i].allocCB = 0;
        m_modules[i].freeCB = 0;

        if(m_modules[i].dlopenPtr)
        {
            dlclose(m_modules[i].dlopenPtr);
            m_modules[i].dlopenPtr = 0;
        }
    }
    
    deleteNotNull(mpCacheManager);
    deleteNotNull(mpFieldGroupManager);

    //DcgmFieldsTerm(); //Not doing this for now due to bug 1787570, comment 1

    /* Shutdown protobuf library at HostEngine side */
    // WAR for bug 2347865
    // google::protobuf::ShutdownProtobufLibrary();

    /* Remove all the connections. Keep it after modules referencing the connections */
    deleteNotNull(mpServerObj);
    deleteNotNull(mpGroupManager);

    /* Remove lingering tracked rquests */
    RemoveAllTrackedRequests();

    lwosDeleteCriticalSection(&m_lock);

    if (LWML_SUCCESS != lwmlShutdown()) {
        throw std::runtime_error("Error: Failed to ShutDown LWML");
    }       
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::LoadModule(dcgmModuleId_t moduleId)
{
    if(moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        PRINT_ERROR("%u", "Invalid moduleId %u", moduleId);
        return DCGM_ST_BADPARAM;
    }

    /* Is the module already loaded? */
    if(m_modules[moduleId].ptr)
        return DCGM_ST_OK;
    
    if(m_modules[moduleId].status == DcgmModuleStatusBlacklisted ||
      m_modules[moduleId].status == DcgmModuleStatusFailed)
    {
        PRINT_WARNING("%u %u", "Skipping loading of module %u in status %u", 
                      moduleId, m_modules[moduleId].status);
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Get the lock so we don't try to load the module from two threads */
    Lock();

    if(m_modules[moduleId].ptr)
    {
        /* Module was loaded by another thread while we were getting the lock */
        Unlock();
        return DCGM_ST_OK;
    }

    /* Do we have a library name to open? */
    if(!m_modules[moduleId].filename)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        Unlock();
        PRINT_ERROR("%u", "Failed to load module %u - no filename", moduleId);
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Try to load the library */
    m_modules[moduleId].dlopenPtr = dlopen(m_modules[moduleId].filename, RTLD_NOW);
    if(!m_modules[moduleId].dlopenPtr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        Unlock();
        PRINT_ERROR("%u %s %s", "Failed to load module %u - dlopen(%s) returned: %s", 
                    moduleId, m_modules[moduleId].filename, dlerror());
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Get all of the function pointers we need */
    m_modules[moduleId].allocCB = (dcgmModuleAlloc_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_alloc_module_instance");
    m_modules[moduleId].freeCB = (dcgmModuleFree_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_free_module_instance");
    if(!m_modules[moduleId].allocCB || !m_modules[moduleId].freeCB)
    {
        PRINT_ERROR("%p %p %s", "dcgm_alloc_module_instance (%p) or dcgm_free_module_instance (%p) was missing from %s",
                    m_modules[moduleId].allocCB, m_modules[moduleId].freeCB,
                    m_modules[moduleId].filename);
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        dlclose(m_modules[moduleId].dlopenPtr);
        m_modules[moduleId].dlopenPtr = 0;
        Unlock();
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Call the constructor (finally). Note that constructors can throw runtime errors. We should treat that as
       the constructor failing and mark the module as failing to load. */
    try
    {
        m_modules[moduleId].ptr = m_modules[moduleId].allocCB();
    }
    catch(const std::runtime_error& e)
    {
        PRINT_ERROR("", "Caught std::runtime error from allocCB()");
        /* m_modules[moduleId].ptr will remain null, which is handled below */
    }
    
    if(!m_modules[moduleId].ptr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        dlclose(m_modules[moduleId].dlopenPtr);
        m_modules[moduleId].dlopenPtr = 0;
        PRINT_ERROR("%u", "Failed to load module %u", moduleId);
    }
    else
    {
        m_modules[moduleId].status = DcgmModuleStatusLoaded;
        PRINT_INFO("%u", "Loaded module %u", moduleId);
    }

    Unlock();

    if(m_modules[moduleId].status == DcgmModuleStatusLoaded)
        return DCGM_ST_OK;
    else
        return DCGM_ST_MODULE_NOT_LOADED;
}


/*****************************************************************************/
int LwcmHostEngineHandler::Lock()
{
    lwosEnterCriticalSection(&m_lock);
    return DCGM_ST_OK;
}

/*****************************************************************************/
int LwcmHostEngineHandler::Unlock()
{
    lwosLeaveCriticalSection(&m_lock);
    return DCGM_ST_OK;
}

/*
 * Pass pointer by reference so that we can set it to NULL afterwards
 */
template<typename T>
void LwcmHostEngineHandler::deleteNotNull(T *&obj)
{
    if (NULL != obj) {
        delete obj;
        obj = NULL;
    }
}

/*****************************************************************************
 This method initializes and returns the singleton instance to LWCM Host Engine Handler
 *****************************************************************************/
LwcmHostEngineHandler* LwcmHostEngineHandler::Init(dcgmOperationMode_t mode)
{
    if (!mpHostEngineHandlerInstance) {
        try {
            mpHostEngineHandlerInstance = new LwcmHostEngineHandler(mode);
        }
        catch (const std::runtime_error &e)
        {
            fprintf(stderr, "%s\n", e.what());
            deleteNotNull(mpHostEngineHandlerInstance);

            return NULL;
        }
    }
    return mpHostEngineHandlerInstance;
}

/*****************************************************************************
 This method returns the singleton instance to LWCM Host Engine Handler
 *****************************************************************************/
LwcmHostEngineHandler* LwcmHostEngineHandler::Instance()
{
    return mpHostEngineHandlerInstance;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::SaveCachedStats(lwcm::CacheManagerSave *cacheManagerSave)
{
    if(cacheManagerSave->filetype() != (lwcm::CacheManagerFileType) DCGM_STATS_FILE_TYPE_JSON ||
            !cacheManagerSave->filename().c_str()[0])
        return DCGM_ST_BADPARAM;

    return (dcgmReturn_t)mpCacheManager->SaveCache(cacheManagerSave->filename());
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::LoadCachedStats(lwcm::CacheManagerLoad *cacheManagerLoad)
{
    if(cacheManagerLoad->filetype() != (lwcm::CacheManagerFileType) DCGM_STATS_FILE_TYPE_JSON ||
            !cacheManagerLoad->filename().c_str()[0])
        return DCGM_ST_BADPARAM;

    return (dcgmReturn_t)mpCacheManager->LoadCache(cacheManagerLoad->filename());
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                    dcgm_field_eid_t entityId,
                                                    const lwcm::WatchFieldValue *watchFieldValue,
                                                    DcgmWatcher watcher)
{
    if(!watchFieldValue || !watchFieldValue->has_fieldid() ||
            !watchFieldValue->has_maxkeepage() || !watchFieldValue->has_updatefreq())
    {
        PRINT_ERROR("", "Bad parameter in WatchFieldValue");
        return DCGM_ST_BADPARAM;
    }

    return (dcgmReturn_t)mpCacheManager->AddFieldWatch(entityGroupId, entityId,
            (unsigned short)watchFieldValue->fieldid(),
            (timelib64_t)watchFieldValue->updatefreq(),
            watchFieldValue->maxkeepage(),
            watchFieldValue->maxkeepsamples(), watcher, false);
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                    dcgm_field_eid_t entityId,
                                                    unsigned short dcgmFieldId,
                                                    timelib64_t monitorFrequencyUsec,
                                                    double maxSampleAge,
                                                    int maxKeepSamples, DcgmWatcher watcher)
{
    return (dcgmReturn_t)mpCacheManager->AddFieldWatch(entityGroupId, entityId, dcgmFieldId,
                                monitorFrequencyUsec, maxSampleAge, maxKeepSamples, watcher, false);
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                      dcgm_field_eid_t entityId,
                                                      const lwcm::UnwatchFieldValue *unwatchFieldValue,
                                                      DcgmWatcher watcher)
{
    int clearCache;

    if(!unwatchFieldValue || !unwatchFieldValue->has_fieldid())
        return DCGM_ST_BADPARAM;

    clearCache = 1; /* Default to true */
    if(unwatchFieldValue->has_clearcache())
        clearCache = unwatchFieldValue->clearcache();

    return (dcgmReturn_t)mpCacheManager->RemoveFieldWatch(entityGroupId, entityId,
                                                          (unsigned short)unwatchFieldValue->fieldid(),
                                                          clearCache, watcher);
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                      dcgm_field_eid_t entityId,
                                                      unsigned short dcgmFieldId,
                                                      int clearCache, DcgmWatcher watcher)
{
    return (dcgmReturn_t)mpCacheManager->RemoveFieldWatch(entityGroupId, entityId, dcgmFieldId, 
                                                          clearCache, watcher);
}


/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::UpdateAllFields(const lwcm::UpdateAllFields *updateAllFields)
{
    int waitForUpdate;

    if(!updateAllFields)
        return DCGM_ST_BADPARAM;

    waitForUpdate = 0;
    if(updateAllFields->has_waitforupdate())
        waitForUpdate = updateAllFields->waitforupdate();

    return (dcgmReturn_t)mpCacheManager->UpdateAllFields(waitForUpdate);
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::InjectFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                     dcgm_field_eid_t entityId,
                                                     lwcm::InjectFieldValue *injectFieldValue)
{
    dcgmcm_sample_t sample = {0};
    lwcm::FieldValue *fieldValue = 0;
    lwcm::Value *value = 0;
    std::string tempStr;
    dcgm_field_meta_p fieldMeta = 0;

    if(!injectFieldValue->has_fieldvalue())
        return DCGM_ST_BADPARAM;

    if(injectFieldValue->version() != dcgmInjectFieldValue_version)
        return DCGM_ST_VER_MISMATCH;

    if(!injectFieldValue->has_version())
        return DCGM_ST_BADPARAM;
    fieldValue = injectFieldValue->mutable_fieldvalue();

    if(!fieldValue->has_fieldtype() || !fieldValue->has_val() || !fieldValue->has_fieldid())
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(fieldValue->fieldid());
    if(!fieldMeta)
        return DCGM_ST_BADPARAM;

    if(fieldValue->has_ts())
        sample.timestamp = fieldValue->ts();

    value = fieldValue->mutable_val();

    switch(fieldValue->fieldtype())
    {
        case lwcm::INT64:
            if(!value->has_i64())
                return DCGM_ST_BADPARAM;
            if(fieldMeta->fieldType != DCGM_FT_INT64)
                return DCGM_ST_BADPARAM;
            sample.val.i64 = value->i64();
            break;

        case lwcm::DBL:
            if(!value->has_dbl())
                return DCGM_ST_BADPARAM;
            if(fieldMeta->fieldType != DCGM_FT_DOUBLE)
                return DCGM_ST_BADPARAM;

            sample.val.d = value->dbl();
            break;

        case lwcm::STR:
            if(!value->has_str())
                return DCGM_ST_BADPARAM;
            if(fieldMeta->fieldType != DCGM_FT_STRING)
                return DCGM_ST_BADPARAM;

            tempStr = value->str().c_str();
            sample.val.str = (char *)tempStr.c_str();
            sample.val2.ptrSize = strlen(sample.val.str)+1;
            /* Note: sample.val.str is only valid as long as tempStr doesn't change */
            break;

        default:
            return DCGM_ST_BADPARAM;
    }

    return mpCacheManager->InjectSamples(entityGroupId, entityId, fieldValue->fieldid() , &sample, 1);
}

/*****************************************************************************/
dcgmReturn_t  LwcmHostEngineHandler::GetLwcmGpuIds(lwcm::FieldMultiValues *pLwcmFieldMultiValues,
                                                   int onlySupported)
{
    unsigned int i;
    std::vector<unsigned int>gpuIds;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = mpCacheManager->GetGpuIds(onlySupported, gpuIds);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Can't find devices at host engine. got error %d", (int)dcgmReturn);
                    pLwcmFieldMultiValues->set_status(DCGM_ST_INIT_ERROR);
        return dcgmReturn;
    }

    PRINT_DEBUG("%d %d", "Got %d gpus from the cache manager. onlySupported %d",
                (int)gpuIds.size(), onlySupported);

    pLwcmFieldMultiValues->set_fieldtype(DCGM_FT_INT64);

    for (i = 0; i < gpuIds.size(); i++)
    {
        int gpuId;

        gpuId = gpuIds[i];
        lwcm::Value* pLwcmValue = pLwcmFieldMultiValues->add_vals();
        pLwcmValue->set_i64(gpuId);
    }

    pLwcmFieldMultiValues->set_status(DCGM_ST_OK);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t  LwcmHostEngineHandler::GetLwcmGpuIds(std::vector<unsigned int> &gpuIds,
                                                   int onlySupported)
{
    return (dcgmReturn_t)mpCacheManager->GetGpuIds(onlySupported, gpuIds);
}


/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::GetLwcmGpuArch(dcgm_field_eid_t entityId,
                                                   lwmlChipArchitecture_t &arch)
{
    return (dcgmReturn_t)mpCacheManager->GetGpuArch(entityId, arch);
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::GetFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                  dcgm_field_eid_t entityId, unsigned int fieldId,
                                                  lwcm::FieldValue* pLwcmFieldValue)
{
    dcgmcm_sample_t sample;
    dcgm_field_meta_p pFieldMetaData;
    dcgmReturn_t ret;

    /* Get Meta data corresponding to the fieldID */
    pFieldMetaData = DcgmFieldGetById(fieldId);
    if (NULL == pFieldMetaData) {
        pLwcmFieldValue->set_status(DCGM_ST_UNKNOWN_FIELD);
        mpCacheManager->FreeSamples(&sample, 1, (unsigned short)fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if(pFieldMetaData->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing entityGroupId to be NONE");
        entityGroupId = DCGM_FE_NONE;
    }

    /* Get Latest sample from cache manager */
    ret = mpCacheManager->GetLatestSample(entityGroupId, entityId, fieldId, &sample, 0);
    if (ret) {
        pLwcmFieldValue->set_status(ret);
        // reduce the logging level as this may pollute the log file when there is continuous filed watch
        PRINT_DEBUG("%u %u %u %d", "Get latest Sample for field ID %u on eg %u, eid %u failed with error %d",
                    fieldId,entityGroupId, entityId, ret);
        return ret;
    }

    pLwcmFieldValue->set_version(dcgmFieldValue_version1);
    pLwcmFieldValue->set_ts(sample.timestamp);

    pLwcmFieldValue->set_fieldid(fieldId);
    pLwcmFieldValue->set_fieldtype(pFieldMetaData->fieldType);
    lwcm::Value *pLwcmVal = pLwcmFieldValue->mutable_val();

    /* Update pcmd based on the field type */
    switch (pFieldMetaData->fieldType) {
        case DCGM_FT_DOUBLE:
            pLwcmVal->set_dbl(sample.val.d);
            break;

        case DCGM_FT_STRING:
            pLwcmVal->set_str(sample.val.str);
            break;

        case DCGM_FT_INT64:        /* Fall-through is intentional */
        case DCGM_FT_TIMESTAMP:
            pLwcmVal->set_i64(sample.val.i64);
            break;

        case DCGM_FT_BINARY:
            pLwcmVal->set_blob(sample.val.blob, sample.val2.ptrSize);
            break;

        default:
            LW_ASSERT(0);
            DEBUG_STDERR("Update code to support additional Field Types");
            mpCacheManager->FreeSamples(&sample, 1, (unsigned short)fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }

    pLwcmFieldValue->set_status(DCGM_ST_OK);
    mpCacheManager->FreeSamples(&sample, 1, (unsigned short)fieldId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::GetLatestSample(dcgm_field_entity_group_t entityGroupId, 
                                                    dcgm_field_eid_t entityId,
                                                    unsigned short dcgmFieldId,
                                                    dcgmcm_sample_p sample)
{
    return (dcgmReturn_t)mpCacheManager->GetLatestSample(entityGroupId, entityId, dcgmFieldId, sample, 0);
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::GetFieldMultipleValues(dcgm_field_entity_group_t entityGroupId,
                                                           dcgm_field_eid_t entityId,
                                                           lwcm::FieldMultiValues *pFieldMultiValues)
{
    dcgmReturn_t lwcmSt;
    int i;
    int fieldId = 0;
    dcgm_field_meta_p fieldMeta = 0;
    int MsampleBuffer = 0; /* Allocated count of sampleBuffer[] */
    int NsampleBuffer = 0; /* Number of values in sampleBuffer[] that are valid */
    dcgmcm_sample_p sampleBuffer = 0;
    dcgmReturn_t retSt = DCGM_ST_OK;
    timelib64_t startTs = 0, endTs = 0;
    dcgmOrder_t order;
    lwcm::Value *pAddValue = 0;

    if(!pFieldMultiValues || !pFieldMultiValues->has_fieldid() ||
            !pFieldMultiValues->has_orderflag() || !pFieldMultiValues->has_maxcount())
    {
        pFieldMultiValues->set_status(DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    fieldId = pFieldMultiValues->fieldid();

    /* Get Meta data corresponding to the fieldID */
    fieldMeta = DcgmFieldGetById(fieldId);
    if (!fieldMeta)
    {
        pFieldMultiValues->set_status(DCGM_ST_UNKNOWN_FIELD);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if(fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing entityGroupId to be NONE");
        entityGroupId = DCGM_FE_NONE;
    }

    pFieldMultiValues->set_version(dcgmGetMultipleValuesForField_version1);
    pFieldMultiValues->set_fieldtype(fieldMeta->fieldType);

    if(pFieldMultiValues->has_startts())
        startTs = (timelib64_t)pFieldMultiValues->startts();
    if(pFieldMultiValues->has_endts())
        endTs = (timelib64_t)pFieldMultiValues->endts();
    order = (dcgmOrder_t)pFieldMultiValues->orderflag();

    MsampleBuffer = pFieldMultiValues->maxcount();
    if(MsampleBuffer < 1)
    {
        pFieldMultiValues->set_status(DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    /* We are allocated the entire buffer of samples. Set a reasonable limit */
    if(MsampleBuffer > 10000)
        MsampleBuffer = 10000;

    sampleBuffer = (dcgmcm_sample_p)malloc(MsampleBuffer * sizeof(sampleBuffer[0]));
    if(!sampleBuffer) {
        PRINT_ERROR("%lu", "failed malloc for %lu bytes", MsampleBuffer * sizeof(sampleBuffer[0]));
        pFieldMultiValues->set_status(DCGM_ST_MEMORY);
        return DCGM_ST_MEMORY;
    }
    /* GOTO CLEANUP BELOW THIS POINT */

    NsampleBuffer = MsampleBuffer;
    lwcmSt = mpCacheManager->GetSamples(entityGroupId, entityId, fieldId, sampleBuffer, &NsampleBuffer,
                                        startTs, endTs, order);
    if(lwcmSt != DCGM_ST_OK)
    {
        retSt = lwcmSt;
        pFieldMultiValues->set_status(retSt);
        goto CLEANUP;
    }
    /* NsampleBuffer now contains the number of valid records returned from our query */

    /* There shouldn't be any elements in here but let's just be sure */
    pFieldMultiValues->clear_vals();

    /* Add each of the samples to the return type */
    for(i=0; i < NsampleBuffer; i++)
    {
        pAddValue = pFieldMultiValues->add_vals();

        pAddValue->set_timestamp(sampleBuffer[i].timestamp);

        switch (fieldMeta->fieldType)
        {
            case DCGM_FT_DOUBLE:
                pAddValue->set_dbl(sampleBuffer[i].val.d);
                break;

            case DCGM_FT_STRING:
                pAddValue->set_str(sampleBuffer[i].val.str);
                break;

            case DCGM_FT_INT64: /* Fall-through is intentional */
            case DCGM_FT_TIMESTAMP:
                pAddValue->set_i64(sampleBuffer[i].val.i64);
                break;

            case DCGM_FT_BINARY:
                pAddValue->set_blob(sampleBuffer[i].val.blob, sampleBuffer[i].val2.ptrSize);
                break;

            default:
                LW_ASSERT(0);
                DEBUG_STDERR("Update code to support additional Field Types");
                retSt = DCGM_ST_GENERIC_ERROR;
                goto CLEANUP;
        }
    }

    pFieldMultiValues->set_maxcount(pFieldMultiValues->vals_size());
    pFieldMultiValues->set_status(retSt);

    CLEANUP:
    if(sampleBuffer)
    {
        if(NsampleBuffer)
            mpCacheManager->FreeSamples(sampleBuffer, NsampleBuffer, (unsigned short)fieldId);
        free(sampleBuffer);
        sampleBuffer = 0;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::GetValuesForFields(dcgm_field_entity_group_t entityGroupId,
                                                       dcgm_field_eid_t entityId,
                                                       unsigned int fieldIds[], unsigned int count,
                                                       lwcm::FieldValue values[])
{
    unsigned int index;

    for (index = 0; index < count; ++index) {
        (void) GetFieldValue(entityGroupId, entityId, fieldIds[index], &values[index]);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::HelperGetInt64StatSummary(dcgm_field_entity_group_t entityGroupId,
                                                              dcgm_field_eid_t entityId, unsigned short fieldId,
                                                              dcgmStatSummaryInt64_t *summary,
                                                              long long startTime, long long endTime)
{
    dcgmReturn_t dcgmReturn;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    long long summaryValues[DcgmcmSummaryTypeSize];

    int numSummaryTypes = 3; /* Should match count below */
    summaryTypes[0] = DcgmcmSummaryTypeMinimum;
    summaryTypes[1] = DcgmcmSummaryTypeMaximum;
    summaryTypes[2] = DcgmcmSummaryTypeAverage;

    dcgmReturn = mpCacheManager->GetInt64SummaryData(entityGroupId, entityId, fieldId,
                                                     numSummaryTypes, summaryTypes,
            summaryValues, startTime, endTime, NULL, NULL);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Should be same indexes as summaryTypes assignments above */
    summary->milwalue = summaryValues[0];
    summary->maxValue = summaryValues[1];
    summary->average = summaryValues[2];

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::HelperGetInt32StatSummary(dcgm_field_entity_group_t entityGroupId,
                                                              dcgm_field_eid_t entityId, unsigned short fieldId,
                                                              dcgmStatSummaryInt32_t *summary,
                                                              long long startTime, long long endTime)
{
    dcgmStatSummaryInt64_t summary64;

    dcgmReturn_t dcgmReturn = HelperGetInt64StatSummary(entityGroupId, entityId, fieldId,
                                                        &summary64, startTime, endTime);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    summary->average = lwcmvalue_int64_to_int32(summary64.average);
    summary->maxValue = lwcmvalue_int64_to_int32(summary64.maxValue);
    summary->milwalue = lwcmvalue_int64_to_int32(summary64.milwalue);
    return DCGM_ST_OK;
}

/*************************************************************************************/
/* Helper to fill destPids[] with unique entries from srcPids it doesn't have already */
static void mergeUniquePids(unsigned int *destPids, int *destPidsSize, int maxDestPids,
        unsigned int *srcPids, int srcPidsSize)
{
    int i, j, havePid;

    if((*destPidsSize) >= maxDestPids)
        return; /* destPids is already full */

    for(i = 0; i < srcPidsSize; i++)
    {
        havePid = 0;
        for(j = 0; j < (*destPidsSize); j++)
        {
            if(srcPids[i] == destPids[j])
            {
                havePid = 1;
                break;
            }
        }

        if(havePid)
            continue;

        destPids[*destPidsSize] = srcPids[i];
        (*destPidsSize)++;

        if((*destPidsSize) >= maxDestPids)
            return; /* destPids is full */
    }
}

/*************************************************************************************/
/* Helper to fill destPidInfo[] with unique entries from srcPidInfo it doesn't have already */

static void mergeUniquePidInfo(dcgmProcessUtilInfo_t *destPidInfo, int *destPidInfoSize, int maxDestPidInfo,
        dcgmProcessUtilInfo_t*srcPidInfo, int srcPidInfoSize)
{
    int i, j, havePid;

    if((*destPidInfoSize) >= maxDestPidInfo)
        return; /* destPids is already full */

    for(i = 0; i < srcPidInfoSize; i++)
    {
        havePid = 0;
        for(j = 0; j < (*destPidInfoSize); j++)
        {
            if(srcPidInfo[i].pid == destPidInfo[j].pid)
            {
                havePid = 1;
                break;
            }
        }

        if(havePid)
            continue;

        destPidInfo[*destPidInfoSize].pid = srcPidInfo[i].pid;
        destPidInfo[*destPidInfoSize].smUtil = srcPidInfo[i].smUtil;
        destPidInfo[*destPidInfoSize].memUtil = srcPidInfo[i].memUtil;
        (*destPidInfoSize)++;

        if((*destPidInfoSize) >= maxDestPidInfo)
            return; /* destPids is full */
    }
}


/*************************************************************************************/
/* Helper to find and fill the Utilization rates in pidInfo for the pid in pidInfo*/

static void findPidUtilInfo(dcgmProcessUtilSample_t* smUtil, unsigned int numSmUtilVal, dcgmProcessUtilSample_t* memUtil, unsigned int numMemUtilVal,
                                                        dcgmProcessUtilInfo_t*pidInfo)
{
    unsigned int smUtilIter = 0, memUtilIter =0, utilIter = 0;
    int pidFound = 0;

    /* Copy the SM Util first*/
    for(smUtilIter = 0; smUtilIter < numSmUtilVal ; smUtilIter++)
    {
        if(pidInfo->pid == smUtil[smUtilIter].pid)
        {
            pidInfo->smUtil = smUtil[smUtilIter].util;
            pidFound = 1;
            break;
        }
    }

    if(!pidFound)
        pidInfo->smUtil = DCGM_INT32_BLANK;

    /* Reset pidFound Variable */
    pidFound = 0;
    
    /* Update the Mem Util */
    for(memUtilIter = 0; memUtilIter < numMemUtilVal ; memUtilIter++)
    {
        if(pidInfo->pid == memUtil[memUtilIter].pid)
        {
            pidInfo->memUtil = memUtil[memUtilIter].util;
            pidFound = 1;
            break;
        }
    }

    if(!pidFound)
        pidInfo->memUtil = DCGM_INT32_BLANK;
}


/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::GetProcessInfo(unsigned int groupId, dcgmPidInfo_t *pidInfo)
{
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned int>gpuIds;
    std::vector<unsigned int>::iterator gpuIdIt;
    dcgmPidSingleInfo_t *singleInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    dcgmDevicePidAccountingStats_t accountingInfo;
    long long startTime, endTime, i64Val;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    int i, j;
    double doubleVal;
    int Msamples = 10; /* Should match size of samples[] */
    dcgmcm_sample_t samples[10];
    int havePid;
    dcgmStatSummaryInt32_t blankSummary32 = {DCGM_INT32_BLANK, DCGM_INT32_BLANK, DCGM_INT32_BLANK};
    dcgmStatSummaryInt64_t blankSummary64 = {DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK};

    /* Sanity check the incoming parameters */
    if(!pidInfo->pid)
    {
        PRINT_WARNING("", "No PID provided in request");
        return DCGM_ST_BADPARAM;
    }

    if(pidInfo->version != dcgmPidInfo_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d", dcgmPidInfo_version, pidInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }


    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn){
        PRINT_ERROR("", "Error: Bad group id parameter");
        return dcgmReturn;
    }

    /* Resolve the groupId -> entities[] -> gpuIds[] */
    dcgmReturn = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process stats are only supported for GPUs for now */
    for(i = 0; i < (int)entities.size(); i++)
    {
        if(entities[i].entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back(entities[i].entityId);
    }

    /* Prepare a health response to be populated once we have startTime and endTime */
    dcgmHealthResponse_v1 response;
    memset(&response, 0, sizeof(response));

    /* Zero the structures */
    memset(&pidInfo->gpus[0], 0, sizeof(pidInfo->gpus));
    memset(&pidInfo->summary, 0, sizeof(pidInfo->summary));

    /* Initialize summary information */
    pidInfo->summary.pcieRxBandwidth = blankSummary64;
    pidInfo->summary.pcieTxBandwidth = blankSummary64;
    pidInfo->summary.powerViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pidInfo->summary.thermalViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pidInfo->summary.energyConsumed = DCGM_INT64_BLANK;
    pidInfo->summary.pcieReplays = 0;
    pidInfo->summary.smUtilization = blankSummary32;
    pidInfo->summary.memoryUtilization = blankSummary32;
    pidInfo->summary.eccSingleBit = 0;
    pidInfo->summary.eccDoubleBit = 0;
    pidInfo->summary.memoryClock = blankSummary32;
    pidInfo->summary.smClock = blankSummary32;

    for(gpuIdIt = gpuIds.begin(); gpuIdIt != gpuIds.end(); ++gpuIdIt)
    {
        singleInfo = &pidInfo->gpus[pidInfo->numGpus];
        singleInfo->gpuId = *gpuIdIt;

        dcgmReturn = mpCacheManager->GetLatestProcessInfo(singleInfo->gpuId, pidInfo->pid, &accountingInfo);
        if(dcgmReturn == DCGM_ST_NO_DATA)
        {
            PRINT_DEBUG("%u %u", "Pid %u did not run on gpuId %u", pidInfo->pid, singleInfo->gpuId);
            continue;
        }

        if(dcgmReturn == DCGM_ST_NOT_WATCHED)
        {
            PRINT_DEBUG("%u %u", "Fields are not watched. Cannot get info for pid %u on GPU %u", pidInfo->pid, singleInfo->gpuId);
            continue;
        }

        /* Increment GPU count now that we know the process ran on this GPU */
        pidInfo->numGpus++;

        startTime = (long long)accountingInfo.startTimestamp;
        if(!pidInfo->summary.startTime || startTime < pidInfo->summary.startTime)
            pidInfo->summary.startTime = startTime;

        
        if (0 == accountingInfo.activeTimeUsec) // Implies that the process is still running
        {
            endTime = (long long)timelib_usecSince1970();
            pidInfo->summary.endTime = 0;    // Set end-time to 0 if the process is act
        }
        else
        {
            endTime = (long long)accountingInfo.startTimestamp + accountingInfo.activeTimeUsec;
            pidInfo->summary.endTime = endTime;
        }
        
        singleInfo->startTime = pidInfo->summary.startTime;
        singleInfo->endTime = pidInfo->summary.endTime;
        
        /* See if the energy counter is supported. If so, use that rather than integrating the power usage */
        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            singleInfo->energyConsumed = i64Val;
        }
        else
        {
            /* No energy counter. Integrate power usage */
            PRINT_DEBUG("", "No energy counter. Using power_usage");
            summaryTypes[0] = DcgmcmSummaryTypeIntegral;
            mpCacheManager->GetFp64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_POWER_USAGE, 1, &summaryTypes[0],
                    &doubleVal, startTime, endTime, NULL, NULL);
            if(!DCGM_FP64_IS_BLANK(doubleVal))
                doubleVal /= 1000.0; /* colwert from usec watts to milliwatt seconds */
            singleInfo->energyConsumed = lwcmvalue_double_to_int64(doubleVal);
        }

        /* Update summary value, handling blank case */
        if(!DCGM_INT64_IS_BLANK(singleInfo->energyConsumed))
        {
            if(!DCGM_INT64_IS_BLANK(pidInfo->summary.energyConsumed))
                pidInfo->summary.energyConsumed += singleInfo->energyConsumed;
            else
                pidInfo->summary.energyConsumed = singleInfo->energyConsumed;
        }

        HelperGetInt64StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_PCIE_RX_THROUGHPUT, &singleInfo->pcieRxBandwidth,                                   startTime, endTime);
        HelperGetInt64StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_PCIE_TX_THROUGHPUT, &singleInfo->pcieTxBandwidth,
                startTime, endTime);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_PCIE_REPLAY_COUNTER, 1, &summaryTypes[0],
                                            &singleInfo->pcieReplays, startTime, endTime, NULL, NULL);
        pidInfo->summary.pcieReplays += singleInfo->pcieReplays;



        HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL, &singleInfo->smUtilization,
                startTime, endTime);
        HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_COPY_UTIL, &singleInfo->memoryUtilization,
                startTime, endTime);

        
        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, 1, &summaryTypes[0],
                &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->eccSingleBit = lwcmvalue_int64_to_int32(i64Val);
        if(!DCGM_INT32_IS_BLANK(singleInfo->eccSingleBit))
            pidInfo->summary.eccSingleBit += singleInfo->eccSingleBit;

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 1, &summaryTypes[0],
                &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->eccDoubleBit = lwcmvalue_int64_to_int32(i64Val);
        if(!DCGM_INT32_IS_BLANK(singleInfo->eccDoubleBit))
            pidInfo->summary.eccDoubleBit += singleInfo->eccDoubleBit;

        HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SM_CLOCK, &singleInfo->smClock,
                startTime, endTime);

        HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_CLOCK, &singleInfo->memoryClock,
                startTime, endTime);

        singleInfo->numXidCriticalErrors = Msamples;
        dcgmReturn = mpCacheManager->GetSamples(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_XID_ERRORS,
                samples, &singleInfo->numXidCriticalErrors,
                startTime, endTime, DCGM_ORDER_ASCENDING);
        for(i = 0; i < singleInfo->numXidCriticalErrors; i++)
        {
            singleInfo->xidCriticalErrorsTs[i] = samples[i].timestamp;
            if(pidInfo->summary.numXidCriticalErrors < (int)LWML_NUMELMS(pidInfo->summary.xidCriticalErrorsTs))
            {
                pidInfo->summary.xidCriticalErrorsTs[pidInfo->summary.numXidCriticalErrors] = samples[i].timestamp;
                pidInfo->summary.numXidCriticalErrors++;
            }
        }
        mpCacheManager->FreeSamples(samples, singleInfo->numXidCriticalErrors,
                DCGM_FI_DEV_XID_ERRORS);

        singleInfo->numOtherComputePids = (int)LWML_NUMELMS(singleInfo->otherComputePids);
        dcgmReturn = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_COMPUTE_PIDS,
                pidInfo->pid,singleInfo->otherComputePids,
                (unsigned int *)&singleInfo->numOtherComputePids,
                startTime, endTime);

        mergeUniquePids(pidInfo->summary.otherComputePids, &pidInfo->summary.numOtherComputePids,
                (int)LWML_NUMELMS(pidInfo->summary.otherComputePids),
                singleInfo->otherComputePids, singleInfo->numOtherComputePids);

        singleInfo->numOtherGraphicsPids = (int)LWML_NUMELMS(singleInfo->otherGraphicsPids);
        dcgmReturn = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GRAPHICS_PIDS,
                pidInfo->pid, singleInfo->otherGraphicsPids,
                (unsigned int *)&singleInfo->numOtherGraphicsPids,
                startTime, endTime);

        mergeUniquePids(pidInfo->summary.otherGraphicsPids, &pidInfo->summary.numOtherGraphicsPids,
                (int)LWML_NUMELMS(pidInfo->summary.otherGraphicsPids),
                singleInfo->otherGraphicsPids, singleInfo->numOtherGraphicsPids);

        singleInfo->maxGpuMemoryUsed = accountingInfo.maxMemoryUsage;
        pidInfo->summary.maxGpuMemoryUsed = accountingInfo.maxMemoryUsage;

        /* Get the unique utilization sample for PIDs from the utilization Sample */
        dcgmProcessUtilSample_t smUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueSmSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL_SAMPLES, pidInfo->pid, smUtil, &numUniqueSmSamples, startTime, endTime);

        dcgmProcessUtilSample_t memUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueMemSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES, pidInfo->pid ,memUtil, &numUniqueMemSamples, startTime, endTime);

        /* Update the process utilization in the pidInfo*/
        singleInfo->processUtilization.pid = pidInfo->pid;
        singleInfo->processUtilization.smUtil = smUtil[0].util;
        singleInfo->processUtilization.memUtil = memUtil[0].util;

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_POWER_VIOLATION, 1, &summaryTypes[0],
                &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->powerViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pidInfo->summary.powerViolationTime))
                pidInfo->summary.powerViolationTime += i64Val;
            else
                pidInfo->summary.powerViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_THERMAL_VIOLATION, 1, &summaryTypes[0],
                &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->thermalViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pidInfo->summary.thermalViolationTime))
                pidInfo->summary.thermalViolationTime += i64Val;
            else
                pidInfo->summary.thermalViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_BOARD_LIMIT_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->boardLimitViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pidInfo->summary.boardLimitViolationTime))
                pidInfo->summary.boardLimitViolationTime += i64Val;
            else
                pidInfo->summary.boardLimitViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_LOW_UTIL_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->lowUtilizationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pidInfo->summary.lowUtilizationTime))
                pidInfo->summary.lowUtilizationTime += i64Val;
            else
                pidInfo->summary.lowUtilizationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_RELIABILITY_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->reliabilityViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pidInfo->summary.reliabilityViolationTime))
                pidInfo->summary.reliabilityViolationTime += i64Val;
            else
                pidInfo->summary.reliabilityViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SYNC_BOOST_VIOLATION, 1, &summaryTypes[0],
                &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->syncBoostTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pidInfo->summary.syncBoostTime))
                pidInfo->summary.syncBoostTime += i64Val;
            else
                pidInfo->summary.syncBoostTime = i64Val;
        }

        /* Update the Health Response - once version has been set, it's been populated. */
        if(response.version != 0)
            HelperHealthCheckV1(groupId, startTime, endTime, &response);

        /* Update the overall Health of the system */
        pidInfo->summary.overallHealth = response.overallHealth;

        /* Find the matching gpuId*/
        unsigned int gpuIndex = 0;
        int found = 0;
        for(gpuIndex = 0; gpuIndex  < response.gpuCount ; gpuIndex ++)
        {
            if(response.gpu[gpuIndex ].gpuId == singleInfo->gpuId)
            {
                found = 1;
                break;
             }
        }

        if(found == 0)
            continue;

        /* Update the Incident Count, overall health of the gpu*/
        singleInfo->incidentCount = response.gpu[gpuIndex].incidentCount;
        singleInfo->overallHealth = response.gpu[gpuIndex].overallHealth;

        for(unsigned int incident= 0; incident < singleInfo->incidentCount ; incident++)
        {
            singleInfo->systems[incident].system = response.gpu[gpuIndex].systems[incident].system;
            singleInfo->systems[incident].health = response.gpu[gpuIndex].systems[incident].health;
        }
        
    }

    if(!pidInfo->numGpus)
    {

        if(dcgmReturn == DCGM_ST_NOT_WATCHED)
        {
            return DCGM_ST_NOT_WATCHED;
        }
        else
        {
            PRINT_DEBUG("%u", "Pid %u ran on no GPUs", pidInfo->pid);
            return DCGM_ST_NO_DATA;
        }
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::JobStartStats(string jobId, unsigned int groupId)
{
    jobIdMap_t::iterator it;
    dcgmReturn_t ret;
    
    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end()) {
        /* Insert it as a record */
        jobRecord_t record;
        record.startTime = timelib_usecSince1970();
        record.endTime = 0;
        record.groupId = groupId;
        mJobIdMap.insert(make_pair(jobId, record));
        Unlock();
    } 
    else {
        Unlock();
        PRINT_ERROR("%s", "Duplicate JobId as input : %s", jobId.c_str());
        /* Implies that the entry corresponding to the job id already exists */
        return DCGM_ST_DUPLICATE_KEY;
    }
    
    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::JobStopStats(string jobId)
{
    jobIdMap_t::iterator it;
    dcgmReturn_t ret;
    
    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end()) {
        Unlock();
        PRINT_ERROR("%s", "Can't find entry corresponding to the Job Id : %s", jobId.c_str());
        return DCGM_ST_NO_DATA;
    } else {
        jobRecord_t  *pRecord;
        pRecord = &(it->second);
        pRecord->endTime = timelib_usecSince1970();
    }
    Unlock();
    
    return DCGM_ST_OK;    
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::HelperHealthCheckV1(unsigned int groupId, 
                                                        long long startTime, 
                                                        long long endTime, 
                                                        dcgmHealthResponse_v1 *response)
{
    dcgm_health_msg_check_v1 msg;

    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_V1;
    msg.header.version = dcgm_health_msg_check_version1;

    msg.groupId = (dcgmGpuGrp_t)(uintptr_t)groupId;
    msg.startTime = startTime;
    msg.endTime = endTime;
    
    dcgmReturn_t dcgmReturn = ProcessModuleCommand(&msg.header);
    if(dcgmReturn != DCGM_ST_OK)
    {
        if(dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
            PRINT_DEBUG("", "Health check skipped due to module not being loaded.");
        else
            PRINT_ERROR("%d", "Health check failed with %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    memcpy(response, &msg.response, sizeof(*response));
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::JobGetStats( string jobId, dcgmJobInfo_t* pJobInfo)
{
    jobIdMap_t::iterator it;
    jobRecord_t  *pRecord;
    int groupId;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned int>gpuIds;
    std::vector<unsigned int>::iterator gpuIdIt;
    dcgmGpuUsageInfo_t *singleInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    dcgmDevicePidAccountingStats_t accountingInfo;
    long long startTime, endTime, i64Val;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    int i, j;
    double doubleVals[DcgmcmSummaryTypeSize];
    int Msamples = 10; /* Should match size of samples[] */
    dcgmcm_sample_t samples[10];
    dcgmStatSummaryInt32_t blankSummary32 = {DCGM_INT32_BLANK, DCGM_INT32_BLANK, DCGM_INT32_BLANK};
    dcgmStatSummaryInt64_t blankSummary64 = {DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK};
    dcgmStatSummaryFp64_t blankSummaryFP64 = {DCGM_FP64_BLANK, DCGM_FP64_BLANK, DCGM_FP64_BLANK};
    int fieldValue;
    
    if(pJobInfo->version != dcgmJobInfo_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d", dcgmJobInfo_version, pJobInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }    
    
    /* If entry can't be found then return error back to the caller */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end()) {
        Unlock();
        PRINT_ERROR("%s", "Can't find entry corresponding to the Job Id : %s", jobId.c_str());
        return DCGM_ST_NO_DATA;
    } else {
        pRecord = &it->second;
    }
    
    groupId = pRecord->groupId;
    startTime = pRecord->startTime;
    
    if (pRecord->endTime == 0) {
        endTime = (long long)timelib_usecSince1970();
    } else {
        endTime = (long long)pRecord->endTime;
    }
    Unlock();
    
    if (startTime > endTime) {
        PRINT_ERROR("%llu %llu", "Get job stats. Start time is greater than end time. start time: %llu end time: %llu", startTime, endTime);
        return DCGM_ST_GENERIC_ERROR;
    }
    
    /* Resolve the groupId -> entities[] -> gpuIds[] */
    dcgmReturn = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process stats are only supported for GPUs for now */
    for(i = 0; i < (int)entities.size(); i++)
    {
        if(entities[i].entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back(entities[i].entityId);
    }

    /* Initialize a health response to be populated later */
    dcgmHealthResponse_v1 response;
    memset(&response, 0, sizeof(response));

    /* Zero the structures */
    pJobInfo->numGpus = 0;
    memset(&pJobInfo->gpus[0], 0, sizeof(pJobInfo->gpus));
    memset(&pJobInfo->summary, 0, sizeof(pJobInfo->summary));

    /* Initialize summary information */
    pJobInfo->summary.gpuId = DCGM_INT32_BLANK;
    pJobInfo->summary.pcieRxBandwidth = blankSummary64;
    pJobInfo->summary.pcieTxBandwidth = blankSummary64;
    pJobInfo->summary.powerViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.thermalViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.reliabilityViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.boardLimitViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.lowUtilizationTime = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.syncBoostTime = DCGM_INT64_BLANK;
    pJobInfo->summary.energyConsumed = DCGM_INT64_BLANK;
    pJobInfo->summary.pcieReplays = DCGM_INT64_BLANK;
    pJobInfo->summary.smUtilization = blankSummary32;
    pJobInfo->summary.memoryUtilization = blankSummary32;
    pJobInfo->summary.eccSingleBit = DCGM_INT32_BLANK;
    pJobInfo->summary.eccDoubleBit = DCGM_INT32_BLANK;
    pJobInfo->summary.memoryClock = blankSummary32;
    pJobInfo->summary.smClock = blankSummary32;
    pJobInfo->summary.powerUsage = blankSummaryFP64;

    /* Update the start and end time in the summary*/
    pJobInfo->summary.startTime = startTime;
    pJobInfo->summary.endTime = endTime;
    
    for(gpuIdIt = gpuIds.begin(); gpuIdIt != gpuIds.end(); ++gpuIdIt)
    {
        singleInfo = &pJobInfo->gpus[pJobInfo->numGpus];
        singleInfo->gpuId = *gpuIdIt;
        
        /* Increment GPU count now that we know the process ran on this GPU */
        pJobInfo->numGpus++;

        summaryTypes[0] = DcgmcmSummaryTypeIntegral;
        summaryTypes[1] = DcgmcmSummaryTypeMinimum;
        summaryTypes[2] = DcgmcmSummaryTypeMaximum;
        summaryTypes[3] = DcgmcmSummaryTypeAverage;

        mpCacheManager->GetFp64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_POWER_USAGE, 4, &summaryTypes[0],
                &doubleVals[0], startTime, endTime, NULL, NULL);
        
        /* See if the energy counter is supported. If so, use that rather than integrating the power usage */
        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            singleInfo->energyConsumed = i64Val;
        }
        else
        {
            /* No energy counter. Integrate power usage */
            PRINT_DEBUG("", "No energy counter. Using power_usage");

            if(!DCGM_FP64_IS_BLANK(doubleVals[0]))
                doubleVals[0] /= 1000.0; /* colwert from usec watts to milliwatt seconds */
            singleInfo->energyConsumed = lwcmvalue_double_to_int64(doubleVals[0]);
        }

        /* Update summary value, handling blank case */
        if(!DCGM_INT64_IS_BLANK(singleInfo->energyConsumed))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.energyConsumed))
                pJobInfo->summary.energyConsumed += singleInfo->energyConsumed;
            else
                pJobInfo->summary.energyConsumed = singleInfo->energyConsumed;
        }

        singleInfo->powerUsage.milwalue = doubleVals[1]; /* Same indexes as summaryTypes[] */
        singleInfo->powerUsage.maxValue = doubleVals[2];
        singleInfo->powerUsage.average = doubleVals[3];

        /* Update summary value for average, handling blank case */
        if(!DCGM_FP64_IS_BLANK(singleInfo->powerUsage.average))
        {
            if(!DCGM_FP64_IS_BLANK(pJobInfo->summary.powerUsage.average))
                pJobInfo->summary.powerUsage.average += singleInfo->powerUsage.average;
            else
                pJobInfo->summary.powerUsage.average = singleInfo->powerUsage.average;
        }

        /* Note: we aren't populating minimum and maximum summary values because they don't make sense across
         * GPUS. One GPUs minimum could occur at a different time than another GPU's minimum
         */

        LwcmHostEngineHandler::Instance()->HelperGetInt64StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_PCIE_RX_THROUGHPUT, &singleInfo->pcieRxBandwidth, startTime, endTime);
        LwcmHostEngineHandler::Instance()->HelperGetInt64StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_PCIE_TX_THROUGHPUT, &singleInfo->pcieTxBandwidth, startTime, endTime);

        /* If the PCIE Tx BW is blank, update the average with the PCIE Tx BW value as 0 for this GPU*/
        if( DCGM_INT64_IS_BLANK(singleInfo->pcieTxBandwidth.average))
            fieldValue = 0;
        else
            fieldValue = singleInfo->pcieTxBandwidth.average;

        pJobInfo->summary.pcieTxBandwidth.average = (pJobInfo->summary.pcieTxBandwidth.average * (pJobInfo->numGpus -1) + fieldValue)/(pJobInfo->numGpus);

        /* If the PCIE Rx BW is blank, update the average with the PCIE Rx BW value as 0 for this GPU*/
        if(DCGM_INT64_IS_BLANK(singleInfo->pcieRxBandwidth.average))
            fieldValue = 0;
        else
            fieldValue = singleInfo->pcieRxBandwidth.average;

        pJobInfo->summary.pcieRxBandwidth.average = (pJobInfo->summary.pcieRxBandwidth.average * (pJobInfo->numGpus -1) + fieldValue) / (pJobInfo->numGpus);
        
        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_PCIE_REPLAY_COUNTER, 1, &summaryTypes[0],
                &singleInfo->pcieReplays, startTime, endTime, NULL, NULL);
        if(!DCGM_INT64_IS_BLANK(singleInfo->pcieReplays))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.pcieReplays))
                pJobInfo->summary.pcieReplays = singleInfo->pcieReplays;
            else
                pJobInfo->summary.pcieReplays += singleInfo->pcieReplays;
        }
        
        singleInfo->startTime = startTime;
        singleInfo->endTime = endTime;

        LwcmHostEngineHandler::Instance()->HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL, &singleInfo->smUtilization,
                startTime, endTime);
        
        /* If the SM utilization is blank, update the average with the SM utilization value as 0 for this GPU*/
        if( DCGM_INT32_IS_BLANK(singleInfo->smUtilization.average))
            fieldValue = 0;
        else
            fieldValue = singleInfo->smUtilization.average;

        pJobInfo->summary.smUtilization.average = (pJobInfo->summary.smUtilization.average * (pJobInfo->numGpus-1) + fieldValue) / (pJobInfo->numGpus);
        
        LwcmHostEngineHandler::Instance()->HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_COPY_UTIL, &singleInfo->memoryUtilization,
                startTime, endTime);

        /* If  mem utilization is blank, update the average with the mem utilization value as 0 for this GPU*/
        if( DCGM_INT32_IS_BLANK( singleInfo->memoryUtilization.average))
            fieldValue = 0;
        else
            fieldValue = singleInfo->memoryUtilization.average;

        pJobInfo->summary.memoryUtilization.average = (pJobInfo->summary.memoryUtilization.average * (pJobInfo->numGpus-1) + fieldValue) / (pJobInfo->numGpus);


        
        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->eccSingleBit = lwcmvalue_int64_to_int32(i64Val);
        
        if(!DCGM_INT32_IS_BLANK(singleInfo->eccSingleBit))
        {
            if(DCGM_INT32_IS_BLANK(pJobInfo->summary.eccSingleBit))
                pJobInfo->summary.eccSingleBit = singleInfo->eccSingleBit;
            else
                pJobInfo->summary.eccSingleBit += singleInfo->eccSingleBit;
        }
        
        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->eccDoubleBit = lwcmvalue_int64_to_int32(i64Val);
        
        if(!DCGM_INT32_IS_BLANK(singleInfo->eccDoubleBit))
        {
            if(DCGM_INT32_IS_BLANK(pJobInfo->summary.eccDoubleBit))
                pJobInfo->summary.eccDoubleBit = singleInfo->eccDoubleBit;
            else
                pJobInfo->summary.eccDoubleBit += singleInfo->eccDoubleBit;
        }
        
        LwcmHostEngineHandler::Instance()->HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SM_CLOCK, &singleInfo->smClock,
                startTime, endTime);

        /* If  SM clock is blank, update the average with the SM  clock value as 0 for this GPU*/
        if( DCGM_INT32_IS_BLANK( singleInfo->smClock.average))
            fieldValue = 0;
        else
            fieldValue = singleInfo->smClock.average;

        pJobInfo->summary.smClock.average = (pJobInfo->summary.smClock.average * (pJobInfo->numGpus-1) + fieldValue) / (pJobInfo->numGpus);
        
        LwcmHostEngineHandler::Instance()->HelperGetInt32StatSummary(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_CLOCK, &singleInfo->memoryClock,
                startTime, endTime);

        /* If memory clock is blank, update the average with the memory clock  value as 0 for this GPU*/
        if( DCGM_INT32_IS_BLANK( singleInfo->memoryClock.average))
            fieldValue = 0;
        else
            fieldValue = singleInfo->memoryClock.average;

        pJobInfo->summary.memoryClock.average = (pJobInfo->summary.memoryClock.average * (pJobInfo->numGpus-1) + fieldValue) / (pJobInfo->numGpus);
        

        singleInfo->numXidCriticalErrors = Msamples;
        dcgmReturn = mpCacheManager->GetSamples(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_XID_ERRORS,
                                                samples, &singleInfo->numXidCriticalErrors,
                                                startTime, endTime, DCGM_ORDER_ASCENDING);
        for(i = 0; i < singleInfo->numXidCriticalErrors; i++)
        {
            singleInfo->xidCriticalErrorsTs[i] = samples[i].timestamp;
            if(pJobInfo->summary.numXidCriticalErrors < (int)LWML_NUMELMS(pJobInfo->summary.xidCriticalErrorsTs))
            {
                pJobInfo->summary.xidCriticalErrorsTs[pJobInfo->summary.numXidCriticalErrors] = samples[i].timestamp;
                pJobInfo->summary.numXidCriticalErrors++;
            }
        }
        mpCacheManager->FreeSamples(samples, singleInfo->numXidCriticalErrors,
                DCGM_FI_DEV_XID_ERRORS);

        singleInfo->numComputePids = (int)LWML_NUMELMS(singleInfo->computePidInfo);
        dcgmReturn = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_COMPUTE_PIDS,
                                                       0, singleInfo->computePidInfo,
                                                       (unsigned int *)&singleInfo->numComputePids,
                                                       startTime, endTime);

        mergeUniquePidInfo(pJobInfo->summary.computePidInfo, &pJobInfo->summary.numComputePids,
                (int)LWML_NUMELMS(pJobInfo->summary.computePidInfo),
                singleInfo->computePidInfo, singleInfo->numComputePids);

        singleInfo->numGraphicsPids = (int)LWML_NUMELMS(singleInfo->graphicsPidInfo);
        dcgmReturn = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GRAPHICS_PIDS,
                0,singleInfo->graphicsPidInfo,
                (unsigned int *)&singleInfo->numGraphicsPids,
                startTime, endTime);

        mergeUniquePidInfo(pJobInfo->summary.graphicsPidInfo, &pJobInfo->summary.numGraphicsPids,
                (int)LWML_NUMELMS(pJobInfo->summary.graphicsPidInfo),
                singleInfo->graphicsPidInfo, singleInfo->numGraphicsPids);

        /* Get the max memory usage for the GPU and summary option for compute PIDs */
        for (i = 0; i < singleInfo->numComputePids; i++) {
            // Get max memory usage for all the processes on the GPU
            dcgmReturn = mpCacheManager->GetLatestProcessInfo(singleInfo->gpuId, singleInfo->computePidInfo[i].pid, &accountingInfo);
            if (DCGM_ST_OK == dcgmReturn) {
                if ((long long)accountingInfo.maxMemoryUsage > singleInfo->maxGpuMemoryUsed) {
                    singleInfo->maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
                
                if ((long long)accountingInfo.maxMemoryUsage > pJobInfo->summary.maxGpuMemoryUsed) {
                    pJobInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
            }
        }
        
        /* Get the max memory usage for the GPU and summary option for Graphics PIDs */
        for (i = 0; i < singleInfo->numGraphicsPids; i++) {
            // Get max memory usage for all the processes on the GPU
            dcgmReturn = mpCacheManager->GetLatestProcessInfo(singleInfo->gpuId, singleInfo->graphicsPidInfo[i].pid, &accountingInfo);
            if (DCGM_ST_OK == dcgmReturn) {
                if ((long long)accountingInfo.maxMemoryUsage > singleInfo->maxGpuMemoryUsed) {
                    singleInfo->maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
                
                if ((long long)accountingInfo.maxMemoryUsage > pJobInfo->summary.maxGpuMemoryUsed) {
                    pJobInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
            }
        }        

        /* Get the unique utilization sample for PIDs from the utilization Sample */
        dcgmProcessUtilSample_t smUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueSmSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL_SAMPLES, 0,
                                              smUtil, &numUniqueSmSamples, startTime, endTime);

        dcgmProcessUtilSample_t memUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueMemSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES, 0,
                                              memUtil, &numUniqueMemSamples, startTime, endTime);

        
        /* Merge the SM and MEM utilization rates for various PIDs */
        for(i = 0; i < singleInfo->numComputePids; i++)
            findPidUtilInfo(smUtil, numUniqueSmSamples, memUtil, numUniqueMemSamples, &singleInfo->computePidInfo[i]);

        for(i = 0; i < singleInfo->numGraphicsPids; i++)
            findPidUtilInfo(smUtil,numUniqueSmSamples, memUtil, numUniqueMemSamples, &singleInfo->graphicsPidInfo[i]);

            
        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_POWER_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->powerViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.powerViolationTime))
                pJobInfo->summary.powerViolationTime += i64Val;
            else
                pJobInfo->summary.powerViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_THERMAL_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->thermalViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.thermalViolationTime))
                pJobInfo->summary.thermalViolationTime += i64Val;
            else
                pJobInfo->summary.thermalViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_RELIABILITY_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->reliabilityViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.reliabilityViolationTime))
                pJobInfo->summary.reliabilityViolationTime += i64Val;
            else
                pJobInfo->summary.reliabilityViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_BOARD_LIMIT_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->boardLimitViolationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.boardLimitViolationTime))
                pJobInfo->summary.boardLimitViolationTime += i64Val;
            else
                pJobInfo->summary.boardLimitViolationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_LOW_UTIL_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->lowUtilizationTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.lowUtilizationTime))
                pJobInfo->summary.lowUtilizationTime += i64Val;
            else
                pJobInfo->summary.lowUtilizationTime = i64Val;
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SYNC_BOOST_VIOLATION, 1, &summaryTypes[0],
                                            &i64Val, startTime, endTime, NULL, NULL);
        singleInfo->syncBoostTime = i64Val;
        if(!DCGM_INT64_IS_BLANK(i64Val))
        {
            if(!DCGM_INT64_IS_BLANK(pJobInfo->summary.syncBoostTime))
                pJobInfo->summary.syncBoostTime += i64Val;
            else
                pJobInfo->summary.syncBoostTime = i64Val;
        }

        /* Update the Health Response if we haven't retrieved it yet */
        if(!response.version)
            HelperHealthCheckV1(groupId,startTime,endTime,&response);

        /* Update the overallHealth of the system */
        pJobInfo->summary.overallHealth = response.overallHealth;

        /* Find the matching GPUId */
        unsigned int gpuIndex = 0;
        int found = 0;

        for(gpuIndex = 0; gpuIndex  < response.gpuCount ; gpuIndex ++)
        {
            if(response.gpu[gpuIndex ].gpuId == singleInfo->gpuId)
            {
                found = 1;
                break;
             }
        }

        if(found == 0)
            continue;

        /* Update the health of the GPU */
        singleInfo->incidentCount = response.gpu[gpuIndex].incidentCount;
        singleInfo->overallHealth = response.gpu[gpuIndex].overallHealth;

        for(unsigned int incident = 0; incident < singleInfo->incidentCount; incident++)
        {
            singleInfo->systems[incident].system = response.gpu[gpuIndex].systems[incident].system;
            singleInfo->systems[incident].health = response.gpu[gpuIndex].systems[incident].health;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::JobRemove(string jobId)
{
    jobIdMap_t::iterator it;
    dcgmReturn_t ret;

    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        Unlock();
        PRINT_ERROR("%s", "JobRemove: Can't find jobId : %s", jobId.c_str());
        return DCGM_ST_NO_DATA;
    }

    mJobIdMap.erase(it);
    Unlock();

    PRINT_DEBUG("%s", "JobRemove: Removed jobId %s", jobId.c_str());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::JobRemoveAll(void)
{
    jobIdMap_t::iterator it;
    dcgmReturn_t ret;

    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    mJobIdMap.clear();
    Unlock();

    PRINT_DEBUG("", "JobRemoveAll: Removed all jobs");
    return DCGM_ST_OK;
}

/*****************************************************************************/
static void helper_get_prof_field_ids(std::vector<unsigned short> &fieldIds, 
                                      std::vector<unsigned short> &profFieldIds)
{
    profFieldIds.clear();

    for(std::vector<unsigned short>::iterator it = fieldIds.begin(); 
        it != fieldIds.end(); ++it)
    {
        if(*it >= DCGM_FI_PROF_FIRST_ID && *it <= DCGM_FI_PROF_LAST_ID)
        {
            profFieldIds.push_back(*it);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::WatchFieldGroup(unsigned int groupId, dcgmFieldGrp_t fieldGroupId,
                                                    timelib64_t monitorFrequencyUsec, double maxSampleAge,
                                                    int maxKeepSamples, DcgmWatcher watcher)
{
    int i, j;
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned short>fieldIds;
    std::vector<unsigned short>profFieldIds;
    dcgmReturn_t retSt = DCGM_ST_OK;

    dcgmReturn = mpGroupManager->GetGroupEntities(watcher.connectionId, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got %d from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }

    PRINT_DEBUG("%d %d", "Got %d entities and %d fields", (int)entities.size(), (int)fieldIds.size());

    for(i=0; i<(int)entities.size(); i++)
    {
        for(j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->AddFieldWatch(entities[i].entityGroupId, entities[i].entityId, 
                                                       fieldIds[j], monitorFrequencyUsec, 
                                                       maxSampleAge, maxKeepSamples, watcher, false);
            if(dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%u %u %d %d", "AddFieldWatch(%u, %u, %d) returned %d", 
                            entities[i].entityGroupId, entities[i].entityId, (int)fieldIds[j], 
                            (int)dcgmReturn);
                retSt = dcgmReturn;
                goto GETOUT;
            }
        }
    }

    /* Add profiling watches after the watches exist in the cache manager so that
       quota policy is in place */
    helper_get_prof_field_ids(fieldIds, profFieldIds);

    if(profFieldIds.size() < 1)
        return DCGM_ST_OK; /* No prof fields. Just return */
    
    dcgm_profiling_msg_watch_fields_t msg;
    memset(&msg, 0, sizeof(msg));

    if(profFieldIds.size() > DCGM_ARRAY_CAPACITY(msg.watchFields.fieldIds))
    {
        PRINT_ERROR("%d", "Too many prof field IDs %d for request DCGM_PROFILING_SR_WATCH_FIELDS",
                    (int)profFieldIds.size());
        
        retSt = DCGM_ST_GENERIC_ERROR;
        goto GETOUT;
    }

    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_WATCH_FIELDS;
    msg.header.connectionId = watcher.connectionId;
    msg.header.version = dcgm_profiling_msg_watch_fields_version;
    msg.watchFields.version = dcgmProfWatchFields_version;
    msg.watchFields.groupId = (void *)(intptr_t)groupId;
    msg.watchFields.numFieldIds = profFieldIds.size();
    memcpy(&msg.watchFields.fieldIds[0], &profFieldIds[0], 
           profFieldIds.size() * sizeof(msg.watchFields.fieldIds[0]));
    msg.watchFields.updateFreq = monitorFrequencyUsec;
    msg.watchFields.maxKeepAge = maxSampleAge;
    msg.watchFields.maxKeepSamples = maxKeepSamples;

    dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&msg);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "DCGM_PROFILING_SR_WATCH_FIELDS failed with %d", dcgmReturn);
        retSt = dcgmReturn;
        goto GETOUT;
    }

GETOUT:
    if(retSt != DCGM_ST_OK)
    {
        /* Clean up any watches that were established since at least one failed */
        UnwatchFieldGroup(groupId, fieldGroupId, watcher);
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::UnwatchFieldGroup(unsigned int groupId, dcgmFieldGrp_t fieldGroupId, 
                                                      DcgmWatcher watcher)
{
    int i, j;
    dcgmReturn_t dcgmReturn;
    dcgmReturn_t retSt = DCGM_ST_OK;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned short>fieldIds;
    std::vector<unsigned short>profFieldIds;

    dcgmReturn = mpGroupManager->GetGroupEntities(watcher.connectionId, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got %d from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }

    PRINT_DEBUG("%d %d", "Got %d entities and %d fields", (int)entities.size(), (int)fieldIds.size());

    for(i=0; i<(int)entities.size(); i++)
    {
        for(j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->RemoveFieldWatch(entities[i].entityGroupId, entities[i].entityId, 
                                                          fieldIds[j], 1, watcher);
            if(dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%u %u %d %d", "RemoveFieldWatch(%u, %u, %d) returned %d", 
                            entities[i].entityGroupId, entities[i].entityId, 
                            (int)fieldIds[j], (int)dcgmReturn);
                retSt = dcgmReturn;
                /* Keep going so we don't leave watches active */
            }
        }
    }

    /* Send a module command to the profiling module to unwatch any fieldIds */
    helper_get_prof_field_ids(fieldIds, profFieldIds);

    if(profFieldIds.size() < 1)
        return retSt; /* No prof fields. Just return */
    
    dcgm_profiling_msg_unwatch_fields_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.header.length = sizeof(msg);
    msg.header.moduleId = DcgmModuleIdProfiling;
    msg.header.subCommand = DCGM_PROFILING_SR_UNWATCH_FIELDS;
    msg.header.connectionId = watcher.connectionId;
    msg.header.version = dcgm_profiling_msg_unwatch_fields_version;
    msg.unwatchFields.version = dcgmProfUnwatchFields_version;
    msg.unwatchFields.groupId = (void *)(intptr_t)groupId;

    dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&msg);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "DCGM_PROFILING_SR_UNWATCH_FIELDS failed with %d", dcgmReturn);
        retSt = dcgmReturn;
    }


    return retSt;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::WatchFieldGroupAllGpus(dcgmFieldGrp_t fieldGroupId,
                                                           timelib64_t monitorFrequencyUsec, double maxSampleAge,
                                                           int maxKeepSamples, int activeOnly, DcgmWatcher watcher)
{
    int i, j;
    dcgmReturn_t dcgmReturn;
    std::vector<unsigned int>gpuIds;
    std::vector<unsigned short>fieldIds;

    dcgmReturn = mpCacheManager->GetGpuIds(activeOnly, gpuIds);

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got %d from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }


    PRINT_DEBUG("%d %d", "Got %d gpus and %d fields", (int)gpuIds.size(), (int)fieldIds.size());

    for(i=0; i<(int)gpuIds.size(); i++)
    {
        for(j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuIds[i], fieldIds[j], monitorFrequencyUsec, 
                                                       maxSampleAge, maxKeepSamples, watcher, false);
            if(dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %d %d", "AddFieldWatch(%d, %d) returned %d", (int)gpuIds[i], (int)fieldIds[j], (int)dcgmReturn);
                return DCGM_ST_GENERIC_ERROR;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::AddRequestWatcher(LwcmRequest *request, 
                                                      dcgm_request_id_t *requestId)
{
    if(!request || !requestId)
        return DCGM_ST_BADPARAM;

    Lock();

    m_nextWatchedRequestId++;

    /* Search for a nonzero, unused request ID. This should only take more than one 
       loop if we've served more than 4 billion requests */
    while(!m_nextWatchedRequestId || 
          m_watchedRequests.find(m_nextWatchedRequestId) != m_watchedRequests.end())
    {
        m_nextWatchedRequestId++;
    }
    
    request->SetRequestId(m_nextWatchedRequestId);
    *requestId = m_nextWatchedRequestId;
    m_watchedRequests[m_nextWatchedRequestId] = request;

    /* Log while we still have the lock */
    PRINT_DEBUG("%u %p", "Assigned requestId %u to request %p", 
                m_nextWatchedRequestId, request);
    Unlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/
void LwcmHostEngineHandler::NotifyRequestOfCompletion(dcgm_connection_id_t connectionId, 
                                                      dcgm_request_id_t requestId)
{
    if(!connectionId)
    {
        /* Local request. Just remove our object */
        Lock();

        watchedRequests_t::iterator it = m_watchedRequests.find(requestId);
        if(it == m_watchedRequests.end())
            PRINT_ERROR("%u", "Unable to find requestId %u", requestId);
        else
        {
            m_watchedRequests.erase(it);
            PRINT_DEBUG("%u", "Removed requestId %u", requestId);
        }
        Unlock();
        return;
    }

    dcgm_msg_request_notify_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.requestId = requestId;

    SendRawMessageToClient(connectionId, DCGM_MSG_REQUEST_NOTIFY, 
                           requestId, &msg, sizeof(msg));
}

/*****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::RemoveAllTrackedRequests(void)
{
    PRINT_DEBUG("", "Entering RemoveAllTrackedRequests");

    Lock();

    watchedRequests_t::iterator it;

    for(it = m_watchedRequests.begin(); it != m_watchedRequests.end(); ++it)
    {
        if(!it->second)
        {
            PRINT_ERROR("%u", "NULL request at ID %u", it->first);
        }
        else
        {
            PRINT_DEBUG("%p %u", "Deleted request %p at ID %u", 
                        it->second, it->first);
            delete(it->second);
        }
    }

    m_watchedRequests.clear();
    
    Unlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/

/*****************************************************************************
 This method is used to start LWCM Host Engine in listening mode
 *****************************************************************************/
dcgmReturn_t LwcmHostEngineHandler::RunServer(unsigned short portNumber, char *socketPath, unsigned int isConnectionTCP)
{
    try {
        mpServerObj = new LwcmHosEngineServer(portNumber, socketPath, isConnectionTCP);
    } catch (std::runtime_error &e) {
        PRINT_ERROR("%s", "ERROR: %s", e.what());
        DEBUG_STDERR(e.what());

        if (mpServerObj) {
            delete mpServerObj;
            mpServerObj = NULL;
        }

        return DCGM_ST_INIT_ERROR;
    }    

    /* Start the server */
    if (0 != mpServerObj->Start()) {
        delete mpServerObj;
        mpServerObj = NULL;        
        return DCGM_ST_INIT_ERROR;
    }

    /* Wait for the notification that the server has started running. 
       Return error if the server can't be started */
    if (0 != mpServerObj->WaitToStart()) {
        delete mpServerObj;
        mpServerObj = NULL;          
        return DCGM_ST_INIT_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
LwcmHosEngineServer* LwcmHostEngineHandler::GetServer()
{
    return mpServerObj;
}

/*****************************************************************************
 This method deletes the LWCM Host Engine Handler Instance
 *****************************************************************************/
void LwcmHostEngineHandler::Cleanup()
{
    if (NULL != mpHostEngineHandlerInstance) {
        delete mpHostEngineHandlerInstance;
        mpHostEngineHandlerInstance = NULL;
    }
}
