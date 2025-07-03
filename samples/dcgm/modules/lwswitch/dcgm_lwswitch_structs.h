#ifndef DCGM_LWSWITCH_STRUCTS_H
#define DCGM_LWSWITCH_STRUCTS_H

#include "dcgm_module_structs.h"
#include "dcgm_module_fm_structs_internal.h"

/*****************************************************************************/
/* LWSwitch Subrequest IDs */
#define DCGM_LWSWITCH_SR_START                                  1 /* Start the LWSwitch module */
#define DCGM_LWSWITCH_SR_SHUTDOWN                               2 /* Stop the LWSwitch module */
#define DCGM_LWSWITCH_SR_GET_SUPPORTED_FABRIC_PARTITIONS        3 /* Query all the available fabric partitions */
#define DCGM_LWSWITCH_SR_ACTIVATE_FABRIC_PARTITION              4 /* Activate a supported fabric partition */
#define DCGM_LWSWITCH_SR_DEACTIVATE_FABRIC_PARTITION            5 /* Deactivate a supported fabric partition */
#define DCGM_LWSWITCH_SR_SET_ACTIVATED_FABRIC_PARTITIONS        6 /* Set activated fabric partitions */

#define DCGM_LWSWITCH_SR_COUNT                                  7 /* Keep as last entry and 1 greater */

#define FM_FILE_PATH_LEN_MAX            256
#define FM_DOMAIN_SOCKET_PATH_LEN_MAX   256

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/
#define MAX_IP_ADDR_LEN 32
typedef struct dcgm_lwswitch_msg_start_v1
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned char startLocal;
    unsigned char startGlobal;
    unsigned short startingPort;
    unsigned char  sharedFabric;
    char bindInterfaceIp[MAX_IP_ADDR_LEN];
} dcgm_lwswitch_msg_start_v1;

typedef struct dcgm_lwswitch_msg_start_v2
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned char startLocal;
    unsigned char startGlobal;
    unsigned short startingPort;
    unsigned char  sharedFabric;
    char domainSocketPath[FM_DOMAIN_SOCKET_PATH_LEN_MAX];
} dcgm_lwswitch_msg_start_v2;

typedef struct dcgm_lwswitch_msg_start_v3
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned char startLocal;
    unsigned char startGlobal;
    unsigned short startingPort;
    unsigned char  sharedFabric;
    char bindInterfaceIp[MAX_IP_ADDR_LEN];
    char domainSocketPath[FM_DOMAIN_SOCKET_PATH_LEN_MAX];
} dcgm_lwswitch_msg_start_v3;

typedef struct dcgm_lwswitch_msg_start_v4
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned char startLocal;
    unsigned char startGlobal;
    unsigned short startingPort;
    unsigned char  sharedFabric;
    unsigned char  restart;
    char bindInterfaceIp[MAX_IP_ADDR_LEN];
    char domainSocketPath[FM_DOMAIN_SOCKET_PATH_LEN_MAX];
    char stateFilename[FM_FILE_PATH_LEN_MAX];
} dcgm_lwswitch_msg_start_v4;

#define dcgm_lwswitch_msg_start_version1 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_start_t,1)
#define dcgm_lwswitch_msg_start_version2 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_start_t,2)
#define dcgm_lwswitch_msg_start_version3 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_start_t,3)
#define dcgm_lwswitch_msg_start_version4 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_start_t,4)

#define dcgm_lwswitch_msg_start_version dcgm_lwswitch_msg_start_version4
typedef dcgm_lwswitch_msg_start_v4 dcgm_lwswitch_msg_start_t;

/*****************************************************************************/
typedef struct dcgm_lwswitch_msg_shutdown_v1
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned char stopLocal;
    unsigned char stopGlobal;
} dcgm_lwswitch_msg_shutdown_v1;

#define dcgm_lwswitch_msg_shutdown_version1 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_shutdown_t,1)
#define dcgm_lwswitch_msg_shutdown_version dcgm_lwswitch_msg_shutdown_version1

typedef dcgm_lwswitch_msg_shutdown_v1 dcgm_lwswitch_msg_shutdown_t;

/*****************************************************************************/

/*****************************************************************************/
typedef struct dcgm_lwswitch_msg_get_fabric_partition_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmFabricPartitionList_t dcgmFabricPartition;
} dcgm_lwswitch_msg_get_fabric_partition_v1;

#define dcgm_lwswitch_msg_get_fabric_partition_version1 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_get_fabric_partition_t,1)
#define dcgm_lwswitch_msg_get_fabric_partition_version dcgm_lwswitch_msg_get_fabric_partition_version1

typedef dcgm_lwswitch_msg_get_fabric_partition_v1 dcgm_lwswitch_msg_get_fabric_partition_t;

/*****************************************************************************/
typedef struct dcgm_lwswitch_msg_activate_fabric_partition_v1
{
    dcgm_module_command_header_t header; /* Command header */
    unsigned int partitionId;
} dcgm_lwswitch_msg_activate_fabric_partition_v1;

#define dcgm_lwswitch_msg_activate_fabric_partition_version1 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_activate_fabric_partition_t,1)
#define dcgm_lwswitch_msg_activate_fabric_partition_version dcgm_lwswitch_msg_activate_fabric_partition_version1

typedef dcgm_lwswitch_msg_activate_fabric_partition_v1 dcgm_lwswitch_msg_activate_fabric_partition_t;

/*****************************************************************************/
typedef struct dcgm_lwswitch_msg_deactivate_fabric_partition_v1
{
    dcgm_module_command_header_t header; /* Command header */
    unsigned int partitionId;
} dcgm_lwswitch_msg_deactivate_fabric_partition_v1;

#define dcgm_lwswitch_msg_deactivate_fabric_partition_version1 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_deactivate_fabric_partition_t,1)
#define dcgm_lwswitch_msg_deactivate_fabric_partition_version dcgm_lwswitch_msg_deactivate_fabric_partition_version1

typedef dcgm_lwswitch_msg_deactivate_fabric_partition_v1 dcgm_lwswitch_msg_deactivate_fabric_partition_t;

/*****************************************************************************/
typedef struct dcgm_lwswitch_msg_set_activated_fabric_partitions_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmActivatedFabricPartitionList_t dcgmFabricPartition;
} dcgm_lwswitch_msg_set_activated_fabric_partitions_v1;

#define dcgm_lwswitch_msg_set_activated_fabric_partitions_version1 MAKE_DCGM_VERSION(dcgm_lwswitch_msg_set_activated_fabric_partitions_t,1)
#define dcgm_lwswitch_msg_set_activated_fabric_partitions_version dcgm_lwswitch_msg_set_activated_fabric_partitions_version1

typedef dcgm_lwswitch_msg_set_activated_fabric_partitions_v1 dcgm_lwswitch_msg_set_activated_fabric_partitions_t;

#endif //DCGM_LWSWITCH_STRUCTS_H
