#ifndef DCGM_VGPU_STRUCTS_H
#define DCGM_VGPU_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* vGPU Subrequest IDs */
#define DCGM_VGPU_SR_START     1 /* Start the vGPU module */
#define DCGM_VGPU_SR_SHUTDOWN  2 /* Stop the vGPU module */
#define DCGM_VGPU_SR_COUNT     3 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/
typedef struct dcgm_vgpu_msg_start_v1
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned int unused;
} dcgm_vgpu_msg_start_v1;

#define dcgm_vgpu_msg_start_version1 MAKE_DCGM_VERSION(dcgm_vgpu_msg_start_t,1)
#define dcgm_vgpu_msg_start_version dcgm_vgpu_msg_start_version1

typedef dcgm_vgpu_msg_start_v1 dcgm_vgpu_msg_start_t;

/*****************************************************************************/
typedef struct dcgm_vgpu_msg_shutdown_v1
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned int unused;
} dcgm_vgpu_msg_shutdown_v1;

#define dcgm_vgpu_msg_shutdown_version1 MAKE_DCGM_VERSION(dcgm_vgpu_msg_shutdown_t,1)
#define dcgm_vgpu_msg_shutdown_version dcgm_vgpu_msg_shutdown_version1

typedef dcgm_vgpu_msg_shutdown_v1 dcgm_vgpu_msg_shutdown_t;

/*****************************************************************************/

#endif //DCGM_VGPU_STRUCTS_H
