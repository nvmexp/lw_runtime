#ifndef DCGMMODULEPOLICY_H
#define DCGMMODULEPOLICY_H

#include "DcgmModule.h"
#include "DcgmPolicyManager.h"
#include "dcgm_policy_structs.h"

class DcgmModulePolicy : public DcgmModule
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModulePolicy();
    virtual ~DcgmModulePolicy(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /* 
     * Process a client disconnecting (inherited from DcgmModule)
     */
    void OnClientDisconnect(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /* 
     * Process a field value we are watching updating (inherited from DcgmModule)
     */
    void OnFieldValuesUpdate(DcgmFvBuffer *fvBuffer);

    /*************************************************************************/
private:

    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessGetPolicies(dcgm_policy_msg_get_policies_t *msg);
    dcgmReturn_t ProcessSetPolicy(dcgm_policy_msg_set_policy_t *msg);
    dcgmReturn_t ProcessRegister(dcgm_policy_msg_register_t *msg);
    dcgmReturn_t ProcessUnregister(dcgm_policy_msg_unregister_t *msg);

    /*************************************************************************/
    /* Private member variables */
    DcgmPolicyManager *mpPolicyManager; /* Pointer to the worker class for this module */
    DcgmCacheManager *mpCacheManager;   /* Cached pointer to the cache manager. Not owned by this class */
    LwcmGroupManager *mpGroupManager;   /* Cached pointer to the group manager. Not owned by this class */
};


#endif //DCGMMODULEPOLICY_H
