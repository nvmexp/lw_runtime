#include "DcgmModulePolicy.h"
#include "dcgm_structs.h"
#include "logging.h"
#include "DcgmLogging.h"
#include "LwcmHostEngineHandler.h"

/*****************************************************************************/
DcgmModulePolicy::DcgmModulePolicy()
{
    mpCacheManager = LwcmHostEngineHandler::Instance()->GetCacheManager();
    if(!mpCacheManager)
    {
        const char *errorStr = "DcgmModulePolicy was unable to find the cache manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpGroupManager = LwcmHostEngineHandler::Instance()->GetGroupManager();
    if(!mpGroupManager)
    {
        const char *errorStr = "DcgmModulePolicy was unable to find the group manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpPolicyManager = new DcgmPolicyManager(LwcmHostEngineHandler::Instance());
    if(!mpPolicyManager)
    {
        const char *errorStr = "DcgmModulePolicy was unable to allocate the policy class.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }
}

/*****************************************************************************/
DcgmModulePolicy::~DcgmModulePolicy()
{
    delete mpPolicyManager;
    mpPolicyManager = 0;
    mpCacheManager = 0; /* Not owned by us */
    mpGroupManager = 0; /* Not owned by us */
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessGetPolicies(dcgm_policy_msg_get_policies_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_get_policies_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    return mpPolicyManager->ProcessGetPolicies(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessSetPolicy(dcgm_policy_msg_set_policy_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_set_policy_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    return mpPolicyManager->ProcessSetPolicy(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessRegister(dcgm_policy_msg_register_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_register_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    return mpPolicyManager->RegisterForPolicy(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessUnregister(dcgm_policy_msg_unregister_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_unregister_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    return mpPolicyManager->UnregisterForPolicy(msg);
}

/*****************************************************************************/
void DcgmModulePolicy::OnClientDisconnect(dcgm_connection_id_t connectionId)
{
    mpPolicyManager->OnClientDisconnect(connectionId);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_POLICY_SR_GET_POLICIES:
            retSt = ProcessGetPolicies((dcgm_policy_msg_get_policies_t *)moduleCommand);
            break;

        case DCGM_POLICY_SR_SET_POLICY:
            retSt = ProcessSetPolicy((dcgm_policy_msg_set_policy_t *)moduleCommand);
            break;

        case DCGM_POLICY_SR_REGISTER:
            retSt = ProcessRegister((dcgm_policy_msg_register_t *)moduleCommand);
            break;

        case DCGM_POLICY_SR_UNREGISTER:
            retSt = ProcessUnregister((dcgm_policy_msg_unregister_t *)moduleCommand);
            break;

        default:
            PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
            return DCGM_ST_BADPARAM;
            break;
    }

    return retSt;
}

/*************************************************************************/
void DcgmModulePolicy::OnFieldValuesUpdate(DcgmFvBuffer *fvBuffer)
{
    mpPolicyManager->OnFieldValuesUpdate(fvBuffer);
}

/*****************************************************************************/
extern "C" DcgmModule *dcgm_alloc_module_instance(void)
{
    return (DcgmModule *)new DcgmModulePolicy();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}

/*****************************************************************************/
