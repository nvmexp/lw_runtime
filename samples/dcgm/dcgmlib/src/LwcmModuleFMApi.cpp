extern "C" {
#include <stdio.h>
#include <stdint.h>
#include "dcgm_module_fm_internal.h"
#include "lwcm_util.h"
#include "lwcmvalue.h"
#include "spinlock.h"
#include "dcgm_lwswitch_structs.h"
}

#include "LwcmSettings.h"
#include "LwcmStatus.h"
#include "DcgmLogging.h"
#include "DcgmFvBuffer.h"
#include "DcgmModuleApi.h"

// Wrap each dcgmFunction with apiEnter and apiExit
#define DCGM_INT_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)         \
        static dcgmReturn_t tsapiFuncname argtypes ;                                  \
        static dcgmReturn_t dcgmFuncname argtypes                                     \
        {                                                                             \
            dcgmReturn_t result;                                                      \
            PRINT_ERROR("Entering %s%s " fmt,                                          \
                        "Entering %s%s " fmt, #dcgmFuncname, #argtypes, ##__VA_ARGS__);\
            result = apiEnter();                                                      \
            if (result != DCGM_ST_OK)                                                 \
            {                                                                         \
                return result;                                                        \
            }                                                                         \
            result = tsapiFuncname(__VA_ARGS__);                                      \
            apiExit();                                                                \
            return result;                                                            \
        }

extern "C" {
#include "dcgm_module_fm_entry_point.h"
}

#include "LwcmClientHandler.h"
#include "LwcmHostEngineHandler.h"


// Instructions:
//
// - Try to make Export Tables backward binary compatible
// - Number all internal functions. Otherwise it's hard to make integrations properly
// - Don't remove rows. Deprecate old functions by putting NULL instead
// - When you do integrations make sure to pad missing functions with NULLs
// - Never renumber functions when integrating. Numbers of functions should always match the
//   module_* numbering
DCGM_INIT_EXTERN_CONST etblDCGMModuleFMInternal g_etblDCGMModuleFMInternal =
{  
        sizeof (g_etblDCGMModuleFMInternal),
        dcgmGetSupportedFabricPartitions,       // 1
        dcgmActivateFabricPartition,            // 2
        dcgmDeactivateFabricPartition,          // 3
        dcgmSetActivatedFabricPartitions,       // 4
};


/*****************************************************************************/
dcgmReturn_t tsapiGetSupportedFabricPartitions(dcgmHandle_t pDcgmHandle,
                                               dcgmFabricPartitionList_t *pDcgmFabricPartition)
{
    dcgmReturn_t dcgmReturn;

    if (NULL == pDcgmFabricPartition) {
        PRINT_ERROR("", "called GetSupportedFabricPartitions with invalid arguments");
        return DCGM_ST_BADPARAM;
    }

    if (pDcgmFabricPartition->version != dcgmFabricPartitionList_version1) {
        PRINT_ERROR("", "called GetSupportedFabricPartitions with unsupported version");
        return DCGM_ST_VER_MISMATCH;
    }

    dcgm_lwswitch_msg_get_fabric_partition_t getPartitionMsg;
    getPartitionMsg.header.version = dcgm_lwswitch_msg_get_fabric_partition_version1;
    getPartitionMsg.header.length = sizeof(getPartitionMsg);
    getPartitionMsg.header.moduleId = DcgmModuleIdLwSwitch;
    getPartitionMsg.header.subCommand = DCGM_LWSWITCH_SR_GET_SUPPORTED_FABRIC_PARTITIONS;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &getPartitionMsg.header);

    if (dcgmReturn == DCGM_ST_OK) {
        // copy the partition information to caller buffer
        memcpy(pDcgmFabricPartition, &getPartitionMsg.dcgmFabricPartition,
               sizeof(getPartitionMsg.dcgmFabricPartition));
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiActivateFabricPartition(dcgmHandle_t pDcgmHandle,
                                          unsigned int partitionId)
{
    dcgm_lwswitch_msg_activate_fabric_partition_t activateMsg;
    activateMsg.header.version = dcgm_lwswitch_msg_activate_fabric_partition_version1;
    activateMsg.header.length = sizeof(activateMsg);
    activateMsg.header.moduleId = DcgmModuleIdLwSwitch;
    activateMsg.header.subCommand = DCGM_LWSWITCH_SR_ACTIVATE_FABRIC_PARTITION;
    activateMsg.partitionId = partitionId;

    return dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &activateMsg.header);
}

/*****************************************************************************/
dcgmReturn_t tsapiDeactivateFabricPartition(dcgmHandle_t pDcgmHandle,
                                            unsigned int partitionId)
{
    dcgm_lwswitch_msg_deactivate_fabric_partition_t deactivateMsg;
    deactivateMsg.header.version = dcgm_lwswitch_msg_deactivate_fabric_partition_version1;
    deactivateMsg.header.length = sizeof(deactivateMsg);
    deactivateMsg.header.moduleId = DcgmModuleIdLwSwitch;
    deactivateMsg.header.subCommand = DCGM_LWSWITCH_SR_DEACTIVATE_FABRIC_PARTITION;
    deactivateMsg.partitionId = partitionId;

    return dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &deactivateMsg.header);
}

/*****************************************************************************/
dcgmReturn_t tsapiSetActivatedFabricPartitions(dcgmHandle_t pDcgmHandle,
        dcgmActivatedFabricPartitionList_t *pDcgmActivatedPartitionList)
{
    if (pDcgmActivatedPartitionList == NULL) {
        PRINT_ERROR("", "called SetActivatedFabricPartitions with invalid arguments");
        return DCGM_ST_BADPARAM;
    }

    dcgm_lwswitch_msg_set_activated_fabric_partitions_t setActivatedPartitionMsg;
    setActivatedPartitionMsg.header.version = dcgm_lwswitch_msg_set_activated_fabric_partitions_version1;
    setActivatedPartitionMsg.header.length = sizeof(setActivatedPartitionMsg);
    setActivatedPartitionMsg.header.moduleId = DcgmModuleIdLwSwitch;
    setActivatedPartitionMsg.header.subCommand = DCGM_LWSWITCH_SR_SET_ACTIVATED_FABRIC_PARTITIONS;
    setActivatedPartitionMsg.dcgmFabricPartition = *pDcgmActivatedPartitionList;

    return dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &setActivatedPartitionMsg.header);
}
