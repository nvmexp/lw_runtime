/*
 * Copyright 1993-2018 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

 #ifndef DCGM_MODULE_FM_STRUCTS_INTERNAL_H
#define DCGM_MODULE_FM_STRUCTS_INTERNAL_H

/* Make sure that dcgm_structs.h is loaded first. This file depends on it */
#include "dcgm_structs.h"
#include "dcgm_agent.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DCGM_MAX_FABRIC_PARTITIONS 31

typedef struct
{
    unsigned int physicalId;
    char uuid[DCGM_DEVICE_UUID_BUFFER_SIZE];
    char pciBusId[DCGM_MAX_STR_LENGTH];
} dcgmFabricPartitionGpuInfo_t;

typedef struct
{
    unsigned int partitionId;
    unsigned int isActive;
    unsigned int numGpus;
    dcgmFabricPartitionGpuInfo_t gpuInfo[DCGM_MAX_NUM_DEVICES];
} dcgmFabricPartitionInfo_t;

typedef struct
{
    unsigned int version;
    unsigned int numPartitions;
    dcgmFabricPartitionInfo_t partitionInfo[DCGM_MAX_FABRIC_PARTITIONS];
} dcgmFabricPartitionList_v1;

typedef dcgmFabricPartitionList_v1 dcgmFabricPartitionList_t;
#define dcgmFabricPartitionList_version1 MAKE_DCGM_VERSION(dcgmFabricPartitionList_v1, 1)
#define dcgmFabricPartitionList_version dcgmFabricPartitionList_version1

typedef struct
{
    unsigned int version;
    unsigned int numPartitions;
    unsigned int partitionIds[DCGM_MAX_FABRIC_PARTITIONS];
} dcgmActivatedFabricPartitionList_v1;

typedef dcgmActivatedFabricPartitionList_v1 dcgmActivatedFabricPartitionList_t;
#define dcgmActivatedFabricPartitionList_version1 MAKE_DCGM_VERSION(dcgmActivatedFabricPartitionList_v1, 1)
#define dcgmActivatedFabricPartitionList_version dcgmActivatedFabricPartitionList_version1

#ifdef __cplusplus
}
#endif

#endif  /* DCGM_MODULE_FM_STRUCTS_INTERNAL_H */
