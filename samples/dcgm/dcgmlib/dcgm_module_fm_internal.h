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

#ifndef DCGM_MODULE_FM_INTERNAL_H
#define DCGM_MODULE_FM_INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "dcgm_structs.h"
#include "dcgm_module_fm_structs_internal.h"
#include "dcgm_uuid.h"

/********************************************************************************/
/* Definition of all the internal DCGM APIs exposed by Fabric Manager           */
/********************************************************************************/

// GUIDS for internal APIs
DCGM_DEFINE_UUID(ETID_DCGMModuleFMInternal,
                 0x5bdeafe8, 0xb2f3, 0x4fd5, 0xb3, 0x8c, 0xfe, 0x9b, 0x9d, 0xc3, 0x8f, 0x1e);

typedef struct etblDCGMModuleFMInternal_st {
    /// This export table supports versioning by adding to the end without changing
    /// the ETID.  The struct_size field will always be set to the size in bytes of
    /// the entire export table structure.
    size_t struct_size;

    // 1
    /**
     * This method is used to query all the available fabric partitions in an LWSwitch based system.
     * These fabric partitions allow users to assign specified GPUs to a guestOS as part of multitenancy
     * with necessary LWLink isolation.
     *
     * @param pDcgmHandle
     * @param pDcgmFabricPartition
     * @return
     */
    dcgmReturn_t(*fpdcgmGetSupportedFabricPartitions)(dcgmHandle_t pDcgmHandle, dcgmFabricPartitionList_t *pDcgmFabricPartition);

    // 2
    /**
     * This method is used to activate a supported fabric partition in an LWSwitch based system.
     *
     * @param pDcgmHandle
     * @param partitionId
     * @return
     */

    dcgmReturn_t(*fpdcgmActivateFabricPartition)(dcgmHandle_t pDcgmHandle, unsigned int partitionId);

    // 3
    /**
     * This method is used to deactivate a previously activated fabric partition in an LWSwitch based system.
     *
     * @param pDcgmHandle
     * @param partitionId
     * @return
     */

    dcgmReturn_t(*fpdcgmDeactivateFabricPartition)(dcgmHandle_t pDcgmHandle, unsigned int partitionId);

    // 4
    /**
     * This method is used to set a list of lwrrently activated fabric partitions to Fabric Manager after its restart.
     *
     * @param pDcgmHandle
     * @param pDcgmActivatedPartitionList
     * @return
     *  DCGM_ST_OK – Success
     *  DCGM_ST_BADPARAM - A bad parameter was passed.
     *  DCGM_ST_NOT_SUPPORTED – Requested feature is not supported or enabled
     *  DCGM_ST_NOT_CONFIGURED – Fabric Manager is initializing and no data available.
     */
    dcgmReturn_t(*fpdcgmSetActivatedFabricPartitions)(dcgmHandle_t pDcgmHandle,  dcgmActivatedFabricPartitionList_t *pDcgmActivatedPartitionList);

} etblDCGMModuleFMInternal;


#ifdef __cplusplus
}
#endif

#endif /* DCGM_MODULE_FM_INTERNAL_H */
