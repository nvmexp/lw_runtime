#include "DcgmModuleVgpu.h"
#include "dcgm_vgpu_structs.h"
#include "LwcmProtobuf.h"
#include "dcgm_structs.h"
#include "logging.h"

/*****************************************************************************/
DcgmModuleVgpu::DcgmModuleVgpu()
{

}

/*****************************************************************************/
DcgmModuleVgpu::~DcgmModuleVgpu()
{

}

/*****************************************************************************/
dcgmReturn_t DcgmModuleVgpu::ProcessShutdown(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_vgpu_msg_shutdown_t *shutdownMsg = (dcgm_vgpu_msg_shutdown_t *)moduleCommand;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_vgpu_msg_shutdown_version);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Do work here to start the vGPU module in the host engine */

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleVgpu::ProcessStart(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;
    dcgm_vgpu_msg_start_t *startMsg = (dcgm_vgpu_msg_start_t *)moduleCommand;

    dcgmReturn = CheckVersion(moduleCommand, dcgm_vgpu_msg_start_version);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    /* Do work here to start the vGPU module in the host engine */

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleVgpu::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_VGPU_SR_START:
            retSt = ProcessStart(moduleCommand);
            break;

        case DCGM_VGPU_SR_SHUTDOWN:
            retSt = ProcessShutdown(moduleCommand);
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
    return (DcgmModule *)new DcgmModuleVgpu();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}


/*****************************************************************************/
