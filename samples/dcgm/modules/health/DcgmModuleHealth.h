#ifndef DCGMMODULEHEALTH_H
#define DCGMMODULEHEALTH_H

#include "DcgmModule.h"
#include "DcgmHealthWatch.h"
#include "dcgm_health_structs.h"

class DcgmModuleHealth : public DcgmModule
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleHealth();
    virtual ~DcgmModuleHealth(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /* 
     * Process an entity group being destroyed (inherited from DcgmModule)
     */
    void OnGroupRemove(unsigned int groupId);

    /*************************************************************************/
private:

    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessGetSystems(dcgm_health_msg_get_systems_t *msg);
    dcgmReturn_t ProcessSetSystems(dcgm_health_msg_set_systems_t *msg);
    dcgmReturn_t ProcessCheckV1(dcgm_health_msg_check_v1 *msg);
    dcgmReturn_t ProcessCheckV2(dcgm_health_msg_check_v2 *msg);
    dcgmReturn_t ProcessCheckV3(dcgm_health_msg_check_v3 *msg);
    dcgmReturn_t ProcessCheckGpus(dcgm_health_msg_check_gpus_t *msg);

    /*************************************************************************/
    /* Private member variables */
    DcgmHealthWatch *mpHealthWatch;     /* Pointer to the worker class for this module */
    DcgmCacheManager *mpCacheManager;   /* Cached pointer to the cache manager. Not owned by this class */
    LwcmGroupManager *mpGroupManager;   /* Cached pointer to the group manager. Not owned by this class */
};


#endif //DCGMMODULEHEALTH_H
