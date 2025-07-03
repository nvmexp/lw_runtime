#ifndef DCGMMODULEAPI_H
#define DCGMMODULEAPI_H

#include "LwcmProtocol.h"
#include "LwcmRequest.h"
#include "dcgm_module_structs.h"

/*****************************************************************************/
/*
 * Helper to send a blocking module request to the DCGM host engine
 *
 * moduleCommand should be both an input and output structure of parameters to
 * your request and results from your request. All module commands 
 * (EX dcgm_vgpu_msg_start_v1, dcgm_vgpu_msg_shutdown_v1) have this structure
 * at the front. 
 * If request is provided, then that class's ProcessMessage class can be used
 * to peek at messages as they are received for this request and possibly notify
 * clients about state changes. Note that doing this will leave the request open
 * until it's removed from the host engine or connection.
 *
 * Note: Timeout is lwrrently only used for remote requests. 
 */
dcgmReturn_t dcgmModuleSendBlockingFixedRequest(dcgmHandle_t pLwcmHandle,
                                                dcgm_module_command_header_t *moduleCommand,
                                                LwcmRequest *request=0, 
                                                unsigned int timeout=60000);

/*****************************************************************************/

#endif //DCGMMODULEAPI_H
