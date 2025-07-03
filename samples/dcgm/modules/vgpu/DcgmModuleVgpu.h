#ifndef DCGMMODULEVGPU_H
#define DCGMMODULEVGPU_H

#include "DcgmModule.h"

class DcgmModuleVgpu : public DcgmModule
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleVgpu();
    virtual ~DcgmModuleVgpu(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     *
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /* Helpers for processing various individual module commands */
    dcgmReturn_t ProcessStart(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessShutdown(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
};


#endif //DCGM_MODULE_VGPU
