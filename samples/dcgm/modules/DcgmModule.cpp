
#include "DcgmModule.h"

/*****************************************************************************/
DcgmModule::~DcgmModule()
{

}

/*****************************************************************************/
dcgmReturn_t DcgmModule::CheckVersion(dcgm_module_command_header_t *moduleCommand, unsigned int compareVersion)
{
    if(!moduleCommand)
        return DCGM_ST_BADPARAM;
    
    if(moduleCommand->version != compareVersion)
        return DCGM_ST_VER_MISMATCH;

    return DCGM_ST_OK;
}

/*****************************************************************************/
