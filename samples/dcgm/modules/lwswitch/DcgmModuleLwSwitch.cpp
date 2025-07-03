
#include <stdexcept>
#include <syslog.h>

#include "DcgmModuleLwSwitch.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmLocalFabricManager.h"
#include "dcgm_lwswitch_structs.h"
#include "LwcmProtobuf.h"
#include "dcgm_structs.h"
#include "logging.h"

/*****************************************************************************/
DcgmModuleLwSwitch::DcgmModuleLwSwitch()
{
    globalFabricManager = NULL;
    localFabricManager = NULL;
}

/*****************************************************************************/
DcgmModuleLwSwitch::~DcgmModuleLwSwitch()
{

}

/*****************************************************************************/
dcgmReturn_t
DcgmModuleLwSwitch::ProcessShutdown(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_shutdown_t *shutdownMsg = (dcgm_lwswitch_msg_shutdown_t *)moduleCommand;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_lwswitch_msg_shutdown_version);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    // exit GlobalFM first as it has to send deinit to LocalFMs
    if (shutdownMsg->stopGlobal && globalFabricManager)
    {
        delete globalFabricManager;
        globalFabricManager = NULL;
        PRINT_DEBUG("", "Stop Global Fabric Manager.");
    }

    /* Do work here to stop the LwSwitch module in the host engine */
    if (shutdownMsg->stopLocal && localFabricManager)
    {
        delete localFabricManager;
        localFabricManager = NULL;
        PRINT_DEBUG("", "Stop Local Fabric Manager.");
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t
DcgmModuleLwSwitch::ProcessStart(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_start_t *startMsg = (dcgm_lwswitch_msg_start_t *)moduleCommand;
    char *domainSocketPath = NULL, *stateFilename = NULL;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_lwswitch_msg_start_version);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Version matches */
    PRINT_DEBUG("", "Got start message");

    if ( strnlen(startMsg->domainSocketPath, FM_DOMAIN_SOCKET_PATH_LEN_MAX) > 0 )
    {
        domainSocketPath = startMsg->domainSocketPath;
    }

    if ( strnlen(startMsg->stateFilename, FM_FILE_PATH_LEN_MAX) > 0 )
    {
        stateFilename = startMsg->stateFilename;
    }

    /* Do work here to start the LwSwitch module in the host engine */
    if (startMsg->startLocal) 
    {
        try
        {
            localFabricManager = startMsg->startingPort > 0 ?
                new DcgmLocalFabricManager(startMsg->sharedFabric == 0 ? false : true, startMsg->bindInterfaceIp, startMsg->startingPort, domainSocketPath) :
                new DcgmLocalFabricManager(startMsg->sharedFabric == 0 ? false : true, startMsg->bindInterfaceIp, FM_CONTROL_CONN_PORT, domainSocketPath);
        }
        catch (const std::runtime_error &e)
        {
            syslog(LOG_NOTICE, e.what());
            fprintf(stderr, "%s\n", e.what());
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    if (startMsg->startGlobal)
    {
        try
        {
            globalFabricManager = startMsg->startingPort > 0 ?
                new DcgmGlobalFabricManager(startMsg->sharedFabric == 0 ? false : true,
                                            startMsg->restart == 0 ? false : true,
                                            startMsg->startingPort,
                                            domainSocketPath,
                                            stateFilename) :
                new DcgmGlobalFabricManager(startMsg->sharedFabric == 0 ? false : true,
                                            startMsg->restart == 0 ? false : true,
                                            FM_CONTROL_CONN_PORT,
                                            domainSocketPath,
                                            stateFilename);
        }
        catch (const std::runtime_error &e)
        {
            syslog(LOG_NOTICE, e.what());
            fprintf(stderr, "%s\n", e.what());
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t
DcgmModuleLwSwitch::ProcessGetSupportedFabricPartitions(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_get_fabric_partition_t *getPartitionMsg = (dcgm_lwswitch_msg_get_fabric_partition_t *)moduleCommand;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_lwswitch_msg_get_fabric_partition_version);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("", "ProcessGetSupportedFabricPartitions: version mismatch");
        return dcgmReturn;
    }

    // make sure that globalFM is started
    if (NULL == globalFabricManager) 
    {
        PRINT_ERROR("", "Global Fabric Manager is not started while requesting for supported fabric partitions.");
        return DCGM_ST_GENERIC_ERROR;
    }

    return globalFabricManager->getSupportedFabricPartitions(getPartitionMsg->dcgmFabricPartition);
}

/*****************************************************************************/
dcgmReturn_t
DcgmModuleLwSwitch::ProcessActivateFabricPartition(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_activate_fabric_partition_t *activatePartitionMsg = (dcgm_lwswitch_msg_activate_fabric_partition_t *)moduleCommand;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_lwswitch_msg_activate_fabric_partition_version);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("", "ProcessActivateFabricPartition: version mismatch");
        return dcgmReturn;
    }

    // make sure that globalFM is started
    if (NULL == globalFabricManager) 
    {
        PRINT_ERROR("", "Global Fabric Manager is not started while requesting to activate fabric partition.");
        return DCGM_ST_GENERIC_ERROR;
    }

    return globalFabricManager->activateFabricPartition(activatePartitionMsg->partitionId);
}

/*****************************************************************************/
dcgmReturn_t
DcgmModuleLwSwitch::ProcessDeactivateFabricPartition(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_deactivate_fabric_partition_t *deactivatePartitionMsg = (dcgm_lwswitch_msg_deactivate_fabric_partition_t *)moduleCommand;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_lwswitch_msg_deactivate_fabric_partition_version);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("", "ProcessDeactivateFabricPartition: version mismatch");
        return dcgmReturn;
    }

    // make sure that globalFM is started
    if (NULL == globalFabricManager) 
    {
        PRINT_ERROR("", "Global Fabric Manager is not started while requesting to deactivate fabric partition.");
        return DCGM_ST_GENERIC_ERROR;
    }

    return globalFabricManager->deactivateFabricPartition(deactivatePartitionMsg->partitionId);
}

/*****************************************************************************/
dcgmReturn_t
DcgmModuleLwSwitch::ProcessSetActivatedFabricPartitions(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_lwswitch_msg_set_activated_fabric_partitions_t *setPartitionMsg = (dcgm_lwswitch_msg_set_activated_fabric_partitions_t *)moduleCommand;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_lwswitch_msg_set_activated_fabric_partitions_version);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("", "ProcessSetActivatedFabricPartitions: version mismatch");
        return dcgmReturn;
    }

    // make sure that globalFM is started
    if (NULL == globalFabricManager)
    {
        PRINT_ERROR("", "Global Fabric Manager is not started while requesting for setting activated fabric partitions.");
        return DCGM_ST_GENERIC_ERROR;
    }

    return globalFabricManager->setActivatedFabricPartitions(setPartitionMsg->dcgmFabricPartition);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleLwSwitch::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_LWSWITCH_SR_START:
            retSt = ProcessStart(moduleCommand);
            break;

        case DCGM_LWSWITCH_SR_SHUTDOWN:
            retSt = ProcessShutdown(moduleCommand);
            break;

        case DCGM_LWSWITCH_SR_GET_SUPPORTED_FABRIC_PARTITIONS:
            retSt = ProcessGetSupportedFabricPartitions(moduleCommand);
            break;

        case DCGM_LWSWITCH_SR_ACTIVATE_FABRIC_PARTITION:
            retSt = ProcessActivateFabricPartition(moduleCommand);
            break;

        case DCGM_LWSWITCH_SR_DEACTIVATE_FABRIC_PARTITION:
            retSt = ProcessDeactivateFabricPartition(moduleCommand);
            break;

        case DCGM_LWSWITCH_SR_SET_ACTIVATED_FABRIC_PARTITIONS:
            retSt = ProcessSetActivatedFabricPartitions(moduleCommand);
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
    return (DcgmModule *)new DcgmModuleLwSwitch();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}

/*****************************************************************************/
