/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#ifndef FMLIB_API_H
#define FMLIB_API_H

typedef struct fm_msg_get_fabric_partition_v1
{
    unsigned int version;
    fmFabricPartitionList_t fmFabricPartitionList;
} fm_msg_get_fabric_partition_v1;

typedef fm_msg_get_fabric_partition_v1 fm_msg_get_fabric_partition_t;
#define fm_msg_get_fabric_partition_version1 MAKE_FM_PARAM_VERSION(fm_msg_get_fabric_partition_t,1)
#define fm_msg_get_fabric_partition_version fm_msg_get_fabric_partition_version1

typedef struct fm_msg_activate_fabric_partition
{
    unsigned int version;
    fmFabricPartitionId_t partitionId;
} fm_msg_activate_fabric_partition_t;

#define fm_msg_activate_fabric_partition_version1 MAKE_FM_PARAM_VERSION(fm_msg_activate_fabric_partition_t,1)
#define fm_msg_activate_fabric_partition_version fm_msg_activate_fabric_partition_version1

typedef struct fm_msg_activate_fabric_partition_vfs
{
    unsigned int version;
    fmFabricPartitionId_t partitionId;
    fmPciDevice_t vfList[FM_MAX_NUM_GPUS];
    unsigned int numVfs;
} fm_msg_activate_fabric_partition_vfs_t;

#define fm_msg_activate_fabric_partition_vfs_version1 MAKE_FM_PARAM_VERSION(fm_msg_activate_fabric_partition_vfs_t,1)
#define fm_msg_activate_fabric_partition_vfs_version fm_msg_activate_fabric_partition_vfs_version1

typedef struct fm_msg_deactivate_fabric_partition
{
    unsigned int version;
    fmFabricPartitionId_t partitionId;
} fm_msg_deactivate_fabric_partition_t;

#define fm_msg_deactivate_fabric_partition_version1 MAKE_FM_PARAM_VERSION(fm_msg_deactivate_fabric_partition_t,1)
#define fm_msg_deactivate_fabric_partition_version fm_msg_deactivate_fabric_partition_version1

typedef struct fm_msg_set_activated_fabric_partition_list_v1
{
    unsigned int version;
    fmActivatedFabricPartitionList_t fmActivatedFabricPartitionList;
} fm_msg_set_activated_fabric_partition_List_v1;

typedef fm_msg_set_activated_fabric_partition_list_v1 fm_msg_set_activated_fabric_partition_list_t;
#define fm_msg_set_activated_fabric_partition_version1 MAKE_FM_PARAM_VERSION(fm_msg_set_activated_fabric_partition_list_v1,1)
#define fm_msg_set_activated_fabric_partition_version fm_msg_set_activated_fabric_partition_version1

typedef struct fm_msg_get_lwlink_failed_devices_v1
{
    unsigned int version;
    fmLwlinkFailedDevices_t fmLwlinkFailedDevices;
} fm_msg_get_lwlink_failed_devices_v1;

typedef fm_msg_get_lwlink_failed_devices_v1 fm_msg_get_lwlink_failed_devices_t;
#define fm_msg_get_lwlink_failed_devices_version1 MAKE_FM_PARAM_VERSION(fm_msg_get_lwlink_failed_devices_t,1)
#define fm_msg_get_lwlink_failed_devices_version fm_msg_get_lwlink_failed_devices_version1

typedef struct fm_msg_get_unsupported_fabric_partition_v1
{
    unsigned int version;
    fmUnsupportedFabricPartitionList_t fmUnsupportedFabricPartitionList;
} fm_msg_get_unsupported_fabric_partition_v1;

typedef fm_msg_get_unsupported_fabric_partition_v1 fm_msg_get_unsupported_fabric_partition_t;
#define fm_msg_get_unsupported_fabric_partition_version1 MAKE_FM_PARAM_VERSION(fm_msg_get_unsupported_fabric_partition_t,1)
#define fm_msg_get_unsupported_fabric_partition_version fm_msg_get_unsupported_fabric_partition_version1

#endif /* LW_FM_TYPES_H */

