#ifndef DCGM_HEALTH_STRUCTS_H
#define DCGM_HEALTH_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
#define DCGM_HEALTH_SR_GET_SYSTEMS             1
#define DCGM_HEALTH_SR_SET_SYSTEMS             2
#define DCGM_HEALTH_SR_CHECK_V1                3
#define DCGM_HEALTH_SR_CHECK_V2                4
#define DCGM_HEALTH_SR_CHECK_GPUS              5
#define DCGM_HEALTH_SR_CHECK_V3                6
#define DCGM_HEALTH_SR_COUNT                   7 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/**
 * Subrequest DCGM_HEALTH_SR_GET_SYSTEMS
 */
typedef struct dcgm_health_msg_get_systems_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;         /*  IN: Group ID to get the health systems of */
    dcgmHealthSystems_t systems;  /* OUT: Health systems of the group */
} dcgm_health_msg_get_systems_v1;

#define dcgm_health_msg_get_systems_version1 MAKE_DCGM_VERSION(dcgm_health_msg_get_systems_v1,1)
#define dcgm_health_msg_get_systems_version dcgm_health_msg_get_systems_version1

typedef dcgm_health_msg_get_systems_v1 dcgm_health_msg_get_systems_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_SET_SYSTEMS
 */
typedef struct dcgm_health_msg_set_systems_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;         /*  IN: Group ID to get the health systems of */
    dcgmHealthSystems_t systems;  /*  IN: Health systems to set for the group */
} dcgm_health_msg_set_systems_v1;

#define dcgm_health_msg_set_systems_version1 MAKE_DCGM_VERSION(dcgm_health_msg_set_systems_v1,1)
#define dcgm_health_msg_set_systems_version dcgm_health_msg_set_systems_version1

typedef dcgm_health_msg_set_systems_v1 dcgm_health_msg_set_systems_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_CHECK_V1
 */
typedef struct dcgm_health_msg_check_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;           /*  IN: Group ID to check the health systems of */
    long long startTime;            /*  IN: Earliest timestamp to health check in usec since 1970. 0=for all time */
    long long endTime;              /*  IN: Latest timestamp to health check in usec since 1970. 0=for all time */
    dcgmHealthResponse_v1 response; /* OUT: Health of the entities of group groupId */
} dcgm_health_msg_check_v1;

#define dcgm_health_msg_check_version1 MAKE_DCGM_VERSION(dcgm_health_msg_check_v1,1)

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_CHECK_V2
 */
typedef struct dcgm_health_msg_check_v2
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;           /*  IN: Group ID to check the health systems of */
    long long startTime;            /*  IN: Earliest timestamp to health check in usec since 1970. 0=for all time */
    long long endTime;              /*  IN: Latest timestamp to health check in usec since 1970. 0=for all time */
    dcgmHealthResponse_v2 response; /* OUT: Health of the entities of group groupId */
} dcgm_health_msg_check_v2;

#define dcgm_health_msg_check_version2 MAKE_DCGM_VERSION(dcgm_health_msg_check_v2,1)

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_CHECK_V3
 */
typedef struct dcgm_health_msg_check_v3
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;           /*  IN: Group ID to check the health systems of */
    long long startTime;            /*  IN: Earliest timestamp to health check in usec since 1970. 0=for all time */
    long long endTime;              /*  IN: Latest timestamp to health check in usec since 1970. 0=for all time */
    dcgmHealthResponse_v3 response; /* OUT: Health of the entities of group groupId */
} dcgm_health_msg_check_v3;

#define dcgm_health_msg_check_version3 MAKE_DCGM_VERSION(dcgm_health_msg_check_v3, 3)

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_CHECK_GPUS (Only used internally)
 */
typedef struct dcgm_health_msg_check_gpus_t
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmHealthSystems_t systems;    /*  IN: Health systems to check for the provided gpuIds */
    unsigned int numGpuIds;         /*  IN: Number of populated entries in gpuIds */
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES]; /*  IN: GPU ID to check the health systems of */
    long long startTime;            /*  IN: Earliest timestamp to health check in usec since 1970. 0=for all time */
    long long endTime;              /*  IN: Latest timestamp to health check in usec since 1970. 0=for all time */
    dcgmHealthResponse_v1 response; /* OUT: Health of gpuId */
} dcgm_health_msg_check_gpus_t;

#define dcgm_health_msg_check_gpus_version MAKE_DCGM_VERSION(dcgm_health_msg_check_gpus_t,1)

/*****************************************************************************/

#endif //DCGM_HEALTH_STRUCTS_H
