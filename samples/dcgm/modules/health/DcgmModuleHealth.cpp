#include "DcgmModuleHealth.h"
#include "dcgm_structs.h"
#include "logging.h"
#include "DcgmLogging.h"
#include "LwcmHostEngineHandler.h"

/*****************************************************************************/
DcgmModuleHealth::DcgmModuleHealth()
{
    mpCacheManager = LwcmHostEngineHandler::Instance()->GetCacheManager();
    if(!mpCacheManager)
    {
        const char *errorStr = "DcgmModuleHealth was unable to find the cache manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpGroupManager = LwcmHostEngineHandler::Instance()->GetGroupManager();
    if(!mpGroupManager)
    {
        const char *errorStr = "DcgmModuleHealth was unable to find the group manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpHealthWatch = new DcgmHealthWatch(mpGroupManager, mpCacheManager);
    if(!mpHealthWatch)
    {
        const char *errorStr = "DcgmModuleHealth was unable to allocate the health class.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }   
}

/*****************************************************************************/
DcgmModuleHealth::~DcgmModuleHealth()
{
    delete mpHealthWatch;
    mpCacheManager = 0; /* Not owned by us */
    mpGroupManager = 0; /* Not owned by us */
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessSetSystems(dcgm_health_msg_set_systems_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_set_systems_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter %u", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpHealthWatch->SetWatches(groupId, msg->systems, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Set Health Watches Err: Unable to set watches");
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessGetSystems(dcgm_health_msg_get_systems_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_get_systems_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter: %u", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpHealthWatch->GetWatches(groupId, &msg->systems);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "GetWatches failed with %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessCheckV1(dcgm_health_msg_check_v1 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_check_version1);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter: %u", dcgmReturn);
        return dcgmReturn;
    }

    /* MonitorWatches is expecting a zeroed struct */
    memset(&msg->response, 0, sizeof(msg->response));

    dcgmReturn = mpHealthWatch->MonitorWatchesV1(groupId, msg->startTime, msg->endTime, &msg->response);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessCheckV2(dcgm_health_msg_check_v2 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_check_version2);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter: %u", dcgmReturn);
        return dcgmReturn;
    }

    /* MonitorWatches is expecting a zeroed struct */
    memset(&msg->response, 0, sizeof(msg->response));
    msg->response.version = dcgmHealthResponse_version2;

    dcgmReturn = mpHealthWatch->MonitorWatchesV2(groupId, msg->startTime, msg->endTime, &msg->response);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessCheckV3(dcgm_health_msg_check_v3 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_check_version3);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter: %u", dcgmReturn);
        return dcgmReturn;
    }

    /* MonitorWatches is expecting a zeroed struct */
    memset(&msg->response, 0, sizeof(msg->response));
    msg->response.version = dcgmHealthResponse_version3;

    dcgmReturn = mpHealthWatch->MonitorWatchesV2(groupId, msg->startTime, msg->endTime, &msg->response);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessCheckGpus(dcgm_health_msg_check_gpus_t *msg)
{
    unsigned int gpuIdIndex;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_check_gpus_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if(!msg->systems || !msg->numGpuIds)
    {
        PRINT_ERROR("%u %u", "Systems %u or numGpuIds %u was missing.", 
                    msg->systems, msg->numGpuIds);
        return DCGM_ST_BADPARAM;
    }

    /* MonitorWatches is expecting a zeroed struct, except the version */
    memset(&msg->response, 0, sizeof(msg->response));
    msg->response.version = dcgmHealthResponse_version1;

    for(gpuIdIndex = 0; gpuIdIndex < msg->numGpuIds; gpuIdIndex++)
    {
        dcgmReturn = mpHealthWatch->MonitorWatchesForGpu(msg->gpuIds[gpuIdIndex], msg->startTime, 
                                                         msg->endTime, msg->systems, &msg->response);
        if(dcgmReturn != DCGM_ST_OK)
            break;
    }

    
    return dcgmReturn;
}

/*****************************************************************************/
void DcgmModuleHealth::OnGroupRemove(unsigned int groupId)
{
    mpHealthWatch->OnGroupRemove(groupId);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_HEALTH_SR_GET_SYSTEMS:
            retSt = ProcessGetSystems((dcgm_health_msg_get_systems_t *)moduleCommand);
            break;

        case DCGM_HEALTH_SR_SET_SYSTEMS:
            retSt = ProcessSetSystems((dcgm_health_msg_set_systems_t *)moduleCommand);
            break;

        case DCGM_HEALTH_SR_CHECK_V1:
            retSt = ProcessCheckV1((dcgm_health_msg_check_v1 *)moduleCommand);
            break;

        case DCGM_HEALTH_SR_CHECK_V2:
            retSt = ProcessCheckV2((dcgm_health_msg_check_v2 *)moduleCommand);
            break;

        case DCGM_HEALTH_SR_CHECK_V3:
            retSt = ProcessCheckV3(reinterpret_cast<dcgm_health_msg_check_v3 *>(moduleCommand));
            break;

        case DCGM_HEALTH_SR_CHECK_GPUS:
            retSt = ProcessCheckGpus((dcgm_health_msg_check_gpus_t *)moduleCommand);
            break;

        default:
            PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
            return DCGM_ST_BADPARAM;
            break;
    }

    return retSt;
}

/*****************************************************************************/
extern "C" DcgmModule *dcgm_alloc_module_instance(void)
{
    return (DcgmModule *)new DcgmModuleHealth();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}

/*****************************************************************************/
