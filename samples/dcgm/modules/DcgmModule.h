#ifndef DCGMMODULE_H
#define DCGMMODULE_H

#include <string>
#include <stdexcept>
#include "dcgm_structs.h"
#include "dcgm_module_structs.h"
#include "LwcmProtobuf.h"
#include "DcgmFvBuffer.h"

/* Base class for a DCGM Module that will be plugged into the host engine to service
 * requests. Extend this class with your own.
 */

class DcgmModule
{
public:
    /*************************************************************************/
    virtual ~DcgmModule(); /* Virtual destructor because of ancient compiler */

    /*************************************************************************/
    /*
     * Helper method to look at the first 4 bytes of the blob field of a moduleCommand
     * and compare the version against the expected version of the message
     *
     * Returns: DCGM_ST_OK if the versions match
     *          DCGM_ST_VER_MISMATCH if the versions mismatch
     *          DCGM_ST_? on other error
     *
     */
    dcgmReturn_t CheckVersion(dcgm_module_command_header_t *moduleCommand, unsigned int compareVersion);

    /*************************************************************************/
    /* 
     * Virtual method to process a DCGM Module command
     *
     * moduleCommand contains the command for this module. Call moduleCommand->set_blob()
     * to set the bytes that will be returned to the caller on the client side.
     *
     * Returns: DCGM_ST_OK if processing the command succeeded
     *          DCGM_ST_? enum value on error. Will be returned to the caller
     */
    virtual
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) = 0;

    /*************************************************************************/
    /*
     * Virtual method for this module to handle when an entity group is removed
     * from DCGM. 
     * 
     * An empty fuction is provided by default, which ignores this event.
     */
    virtual
    void OnGroupRemove(unsigned int groupId) {};

    /*************************************************************************/
    /*
     * Virtual method for this module to handle when a client disconnects from
     * DCGM.
     * 
     * An empty fuction is provided by default, which ignores this event.
     */
    virtual
    void OnClientDisconnect(dcgm_connection_id_t connectionId) {};

    /*************************************************************************/
    /* 
     * Virtual method for this module to process a field value that updated in
     * the cache manager that this module subscribed for
     */
    virtual
    void OnFieldValuesUpdate(DcgmFvBuffer *fvBuffer) {};

    /*************************************************************************/
};

/* Callback functions for allocating and freeing DcgmModules. These are found
   in the modules' shared library with dlsym */
typedef DcgmModule *(*dcgmModuleAlloc_f)(void);
typedef void (*dcgmModuleFree_f)(DcgmModule *);

#endif //DCGMMODULE_H
