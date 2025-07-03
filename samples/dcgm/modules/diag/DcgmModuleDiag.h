#ifndef DCGMMODULEDIAG_H
#define DCGMMODULEDIAG_H

#include "DcgmModule.h"
#include "DcgmDiagManager.h"
#include "dcgm_diag_structs.h"

class DcgmModuleDiag : public DcgmModule
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleDiag();
    virtual ~DcgmModuleDiag(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
private:

    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessRun(dcgm_diag_msg_run_t *msg);
    dcgmReturn_t ProcessRunLegacyV1(dcgm_diag_msg_run_v1 *msg);
    dcgmReturn_t ProcessRunLegacyV2(dcgm_diag_msg_run_v2 *msg);
    dcgmReturn_t ProcessStop(dcgm_diag_msg_stop_t *msg);

    /*************************************************************************/
    /* Private member variables */
    DcgmDiagManager *mpDiagManager; /* Pointer to the worker class for this module */
    DcgmCacheManager *mpCacheManager;   /* Cached pointer to the cache manager. Not owned by this class */
    LwcmGroupManager *mpGroupManager;   /* Cached pointer to the group manager. Not owned by this class */
};


#endif //DCGMMODULEDIAG_H
