
#include "DcgmModuleDiag.h"
#include "dcgm_structs.h"
#include "logging.h"
#include "DcgmLogging.h"
#include "LwcmHostEngineHandler.h"
#include "DcgmConfigManager.h"
#include "DcgmDiagResponseWrapper.h"

/*****************************************************************************/
DcgmModuleDiag::DcgmModuleDiag()
{
    mpCacheManager = LwcmHostEngineHandler::Instance()->GetCacheManager();
    if(!mpCacheManager)
    {
        const char *errorStr = "DcgmModuleDiag was unable to find the cache manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpGroupManager = LwcmHostEngineHandler::Instance()->GetGroupManager();
    if(!mpGroupManager)
    {
        const char *errorStr = "DcgmModuleDiag was unable to find the group manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpDiagManager = new DcgmDiagManager(mpCacheManager, mpGroupManager);
    if(!mpDiagManager)
    {
        const char *errorStr = "DcgmModuleDiag was unable to allocate the config class.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }   
}

/*****************************************************************************/
DcgmModuleDiag::~DcgmModuleDiag()
{
    delete mpDiagManager;
    mpCacheManager = 0; /* Not owned by us */
    mpGroupManager = 0; /* Not owned by us */
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessRunLegacyV1(dcgm_diag_msg_run_v1 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    drw.SetVersion3(&msg->diagResponse);

    dcgmReturn = mpDiagManager->RunDiagAndAction(&msg->runDiag, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "RunDiagAndAction returned %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessRunLegacyV2(dcgm_diag_msg_run_v2 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    drw.SetVersion4(&msg->diagResponse);

    dcgmReturn = mpDiagManager->RunDiagAndAction(&msg->runDiag, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "RunDiagAndAction returned %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessRun(dcgm_diag_msg_run_t *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        drw.SetVersion5(&msg->diagResponse);
    }

    dcgmReturn = mpDiagManager->RunDiagAndAction(&msg->runDiag, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "RunDiagAndAction returned %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessStop(dcgm_diag_msg_stop_t *msg)
{
    return mpDiagManager->StopRunningDiag();
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_DIAG_SR_RUN:
        
            if (DCGM_ST_OK == CheckVersion(moduleCommand, dcgm_diag_msg_run_version1))
            {
                retSt = ProcessRunLegacyV1((dcgm_diag_msg_run_v1 *)moduleCommand);
            }
            else if (DCGM_ST_OK == CheckVersion(moduleCommand, dcgm_diag_msg_run_version2))
            {
                retSt = ProcessRunLegacyV2((dcgm_diag_msg_run_v2 *)moduleCommand);
            }
            else
            {
                retSt = ProcessRun((dcgm_diag_msg_run_t *)moduleCommand);
            }

            break;
        
        case DCGM_DIAG_SR_STOP:
            retSt = ProcessStop((dcgm_diag_msg_stop_t *)moduleCommand);
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
    return (DcgmModule *)new DcgmModuleDiag();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}

/*****************************************************************************/
