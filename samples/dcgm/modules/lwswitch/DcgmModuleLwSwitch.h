#ifndef DCGMMODULELWSWITCH_H
#define DCGMMODULELWSWITCH_H

#include "DcgmModule.h"

/* Forward define these classes so we don't need to bring all of the include 
   baggage of their header files */
class DcgmGlobalFabricManager;
class DcgmLocalFabricManager;

class DcgmModuleLwSwitch : public DcgmModule
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleLwSwitch();
    virtual ~DcgmModuleLwSwitch(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     *
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/

private:

    /* Helpers for processing various individual module commands */
    dcgmReturn_t ProcessStart(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessShutdown(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessGetSupportedFabricPartitions(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessActivateFabricPartition(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessDeactivateFabricPartition(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessSetActivatedFabricPartitions(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/

    DcgmGlobalFabricManager *globalFabricManager;
    DcgmLocalFabricManager *localFabricManager;
};


#endif //DCGM_MODULE_LWSWITCH
