#ifndef DCGM_DIAG_STRUCTS_H
#define DCGM_DIAG_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
#define DCGM_DIAG_SR_RUN                     1
#define DCGM_DIAG_SR_STOP                    2
#define DCGM_DIAG_SR_COUNT                   2 /* Keep as last entry with same value as highest number */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 1
 */
typedef struct dcgm_diag_msg_run_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmPolicyAction_t action;       /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_t runDiag;           /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v3 diagResponse; /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v1;

#define dcgm_diag_msg_run_version1 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v1,1)

/**
 * Subrequest DCGM_DIAG_SR_RUN version 2
 */
typedef struct dcgm_diag_msg_run_v2
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmPolicyAction_t action;       /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_t runDiag;           /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v4 diagResponse; /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v2;

#define dcgm_diag_msg_run_version2 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v2,2)

/**
 * Subrequest DCGM_DIAG_SR_RUN version 3
 */
typedef struct dcgm_diag_msg_run_v3
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmPolicyAction_t action;       /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_t runDiag;           /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v5 diagResponse; /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v3;

#define dcgm_diag_msg_run_version3 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v3, 3)
#define dcgm_diag_msg_run_version dcgm_diag_msg_run_version3

typedef dcgm_diag_msg_run_v3 dcgm_diag_msg_run_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_STOP version 1
 */
typedef struct dcgm_diag_msg_stop_v1
{
    dcgm_module_command_header_t header; /* Command header */
} dcgm_diag_msg_stop_v1;

#define dcgm_diag_msg_stop_version1 MAKE_DCGM_VERSION(dcgm_diag_msg_stop_v1,1)
#define dcgm_diag_msg_stop_version dcgm_diag_msg_stop_version1

typedef dcgm_diag_msg_stop_v1 dcgm_diag_msg_stop_t;

/*****************************************************************************/

#endif //DCGM_DIAG_STRUCTS_H
