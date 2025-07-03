#pragma once

/* DCGM Module messages used for communicating with core DCGM */

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Core Subrequest IDs */
#define DCGM_CORE_SR_CLIENT_DISCONNECT 1 /* Notify modules that a client logged out */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/**
 * Subrequest DCGM_CORE_SR_CLIENT_DISCONNECT
 */
typedef struct dcgm_core_msg_client_disconnect_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgm_connection_id_t connectionId; /* ID of the client that logged out */
} dcgm_core_msg_client_disconnect_v1;

#define dcgm_core_msg_client_disconnect_version1 MAKE_DCGM_VERSION(dcgm_core_msg_client_disconnect_v1,1)
#define dcgm_core_msg_client_disconnect_version dcgm_core_msg_client_disconnect_version1

typedef dcgm_core_msg_client_disconnect_v1 dcgm_core_msg_client_disconnect_t;

