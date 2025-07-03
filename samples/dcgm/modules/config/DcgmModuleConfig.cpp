
#include "DcgmModuleConfig.h"
#include "dcgm_structs.h"
#include "logging.h"
#include "DcgmLogging.h"
#include "LwcmHostEngineHandler.h"
#include "DcgmConfigManager.h"

/*****************************************************************************/
DcgmModuleConfig::DcgmModuleConfig()
{
    mpCacheManager = LwcmHostEngineHandler::Instance()->GetCacheManager();
    if(!mpCacheManager)
    {
        const char *errorStr = "DcgmModuleConfig was unable to find the cache manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpGroupManager = LwcmHostEngineHandler::Instance()->GetGroupManager();
    if(!mpGroupManager)
    {
        const char *errorStr = "DcgmModuleConfig was unable to find the group manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpConfigManager = new DcgmConfigManager(mpCacheManager, mpGroupManager);
    if(!mpConfigManager)
    {
        const char *errorStr = "DcgmModuleConfig was unable to allocate the config class.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }   
}

/*****************************************************************************/
DcgmModuleConfig::~DcgmModuleConfig()
{
    delete mpConfigManager;
    mpCacheManager = 0; /* Not owned by us */
    mpGroupManager = 0; /* Not owned by us */
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessSetConfig(dcgm_config_msg_set_v1 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_set_version);
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

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    dcgmReturn = mpConfigManager->SetConfig(groupId, &msg->config, &statusList);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "SetConfig returned %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessGetConfig(dcgm_config_msg_get_v1 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_get_version);
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

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    if (msg->reqType == DCGM_CONFIG_TARGET_STATE)
        dcgmReturn = mpConfigManager->GetTargetConfig(groupId, &msg->numConfigs, msg->configs, &statusList);
    else if(msg->reqType == DCGM_CONFIG_LWRRENT_STATE)
        dcgmReturn = mpConfigManager->GetLwrrentConfig(groupId, &msg->numConfigs, msg->configs, &statusList);
    else
    {
        PRINT_ERROR("%u", "Bad reqType %u", msg->reqType);
        return DCGM_ST_BADPARAM;
    }

    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "GetConfig failed with %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessEnforceConfigGroup(dcgm_config_msg_enforce_group_v1 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_enforce_group_version);
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

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    dcgmReturn = mpConfigManager->EnforceConfigGroup(groupId, &statusList);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessEnforceConfigGpu(dcgm_config_msg_enforce_gpu_v1 *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_enforce_gpu_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    dcgmReturn = mpConfigManager->EnforceConfigGpu(msg->gpuId, &statusList);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_CONFIG_SR_GET:
            retSt = ProcessGetConfig((dcgm_config_msg_get_v1 *)moduleCommand);
            break;

        case DCGM_CONFIG_SR_SET:
            retSt = ProcessSetConfig((dcgm_config_msg_set_v1 *)moduleCommand);
            break;

        case DCGM_CONFIG_SR_ENFORCE_GROUP:
            retSt = ProcessEnforceConfigGroup((dcgm_config_msg_enforce_group_v1 *)moduleCommand);
            break;

        case DCGM_CONFIG_SR_ENFORCE_GPU:
            retSt = ProcessEnforceConfigGpu((dcgm_config_msg_enforce_gpu_v1 *)moduleCommand);
            break;

        default:
            PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
            return DCGM_ST_BADPARAM;
            break;
    }

    return retSt;
}

/*****************************************************************************/
void DcgmModuleConfig::OnClientDisconnect(dcgm_connection_id_t connectionId)
{
    mpConfigManager->OnClientDisconnect(connectionId);
}

/*****************************************************************************/
extern "C" DcgmModule *dcgm_alloc_module_instance(void)
{
    return (DcgmModule *)new DcgmModuleConfig();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}

/*****************************************************************************/
