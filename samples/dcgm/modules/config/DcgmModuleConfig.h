#ifndef DCGMMODULECONFIG_H
#define DCGMMODULECONFIG_H

#include "DcgmModule.h"
#include "DcgmConfigManager.h"
#include "dcgm_config_structs.h"

class DcgmModuleConfig : public DcgmModule
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleConfig();
    virtual ~DcgmModuleConfig(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /*
     * Virtual method for this module to handle when a client disconnects from
     * DCGM.
     * 
     */    
    void OnClientDisconnect(dcgm_connection_id_t connectionId);

    /*************************************************************************/
private:

    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessGetConfig(dcgm_config_msg_get_v1 *msg);
    dcgmReturn_t ProcessSetConfig(dcgm_config_msg_set_v1 *msg);
    dcgmReturn_t ProcessEnforceConfigGroup(dcgm_config_msg_enforce_group_v1 *msg);
    dcgmReturn_t ProcessEnforceConfigGpu(dcgm_config_msg_enforce_gpu_v1 *msg);

    /*************************************************************************/
    /* Private member variables */
    DcgmConfigManager *mpConfigManager; /* Pointer to the worker class for this module */
    DcgmCacheManager *mpCacheManager;   /* Cached pointer to the cache manager. Not owned by this class */
    LwcmGroupManager *mpGroupManager;   /* Cached pointer to the group manager. Not owned by this class */
};


#endif //DCGMMODULECONFIG_H
