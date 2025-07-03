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

/*
 * File: dcgm_structs.h
 */

#ifndef DCGM_STRUCTS_H
#define DCGM_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "dcgm_fields.h"  
#include <stdint.h>

/***************************************************************************************************/
/** @defgroup lwmlReturnEnums Enums and Macros
 *  @{
 */
/***************************************************************************************************/    

/**
 * Represents value of the field which can be returned by Host Engine in case the
 * operation is not successful
 *
 */
#ifndef DCGM_BLANK_VALUES
#define DCGM_BLANK_VALUES
    
/**
 * Base value for 32 bits integer blank. can be used as an unspecified blank 
 */
#define DCGM_INT32_BLANK 0x7ffffff0
    
/**
 * Base value for 64 bits integer blank. can be used as an unspecified blank 
 */
#define DCGM_INT64_BLANK 0x7ffffffffffffff0

/**
 * Base value for double blank. 2 ** 47. FP 64 has 52 bits of mantissa,
 * so 47 bits can still increment by 1 and represent each value from 0-15 
 */
#define DCGM_FP64_BLANK 140737488355328.0

/**
 * Base value for string blank.
 */
#define DCGM_STR_BLANK "<<<NULL>>>"

/** 
 * Represents an error where INT32 data was not found 
 */
#define DCGM_INT32_NOT_FOUND          (DCGM_INT32_BLANK+1)
    
/** 
 * Represents an error where INT64 data was not found 
 */
#define DCGM_INT64_NOT_FOUND          (DCGM_INT64_BLANK+1)
    
/** 
 * Represents an error where FP64 data was not found 
 */    
#define DCGM_FP64_NOT_FOUND           (DCGM_FP64_BLANK+1.0)
    
/** 
 * Represents an error where STR data was not found 
 */        
#define DCGM_STR_NOT_FOUND            "<<<NOT_FOUND>>>"

/** 
 * Represents an error where fetching the INT32 value is not supported 
 */
#define DCGM_INT32_NOT_SUPPORTED           (DCGM_INT32_BLANK+2)
    
/** 
 * Represents an error where fetching the INT64 value is not supported 
 */    
#define DCGM_INT64_NOT_SUPPORTED           (DCGM_INT64_BLANK+2)
    
/** 
 * Represents an error where fetching the FP64 value is not supported 
 */        
#define DCGM_FP64_NOT_SUPPORTED            (DCGM_FP64_BLANK+2.0)
    
/** 
 * Represents an error where fetching the STR value is not supported 
 */            
#define DCGM_STR_NOT_SUPPORTED             "<<<NOT_SUPPORTED>>>"

/**
 *  Represents and error where fetching the INT32 value is not allowed with our current credentials 
 */
#define DCGM_INT32_NOT_PERMISSIONED        (DCGM_INT32_BLANK+3)
    
/**
 *  Represents and error where fetching the INT64 value is not allowed with our current credentials 
 */    
#define DCGM_INT64_NOT_PERMISSIONED        (DCGM_INT64_BLANK+3)
    
/**
 *  Represents and error where fetching the FP64 value is not allowed with our current credentials 
 */        
#define DCGM_FP64_NOT_PERMISSIONED         (DCGM_FP64_BLANK+3.0)
    
/**
 *  Represents and error where fetching the STR value is not allowed with our current credentials 
 */            
#define DCGM_STR_NOT_PERMISSIONED          "<<<NOT_PERM>>>"

/** 
 * Macro to check if a INT32 value is blank or not 
 */
#define DCGM_INT32_IS_BLANK(val) (((val) >= DCGM_INT32_BLANK) ? 1 : 0)
    
/** 
 * Macro to check if a INT64 value is blank or not 
 */    
#define DCGM_INT64_IS_BLANK(val) (((val) >= DCGM_INT64_BLANK) ? 1 : 0)
    
/** 
 * Macro to check if a FP64 value is blank or not 
 */        
#define DCGM_FP64_IS_BLANK(val) (((val) >= DCGM_FP64_BLANK ? 1 : 0))
    
/** 
 * Macro to check if a STR value is blank or not 
 * Works on (char *). Looks for <<< at first position and >>> inside string
 */        
#define DCGM_STR_IS_BLANK(val) (val == strstr(val, "<<<") && strstr(val, ">>>"))

#endif //DCGM_BLANK_VALUES

/**
 * Max number of GPUs supported by DCGM
 */    
#define DCGM_MAX_NUM_DEVICES   16

/**
 * Number of LwLink links per GPU supported by DCGM
 * This is 12 for Ampere, 6 for Volta, and 4 for Pascal
 */
#define DCGM_LWLINK_MAX_LINKS_PER_GPU 12

/**
 * Maximum LwLink links pre-Ampere
 */
#define DCGM_LWLINK_MAX_LINKS_PER_GPU_LEGACY1 6

/**
 * Max number of LwSwitches supported by DCGM 
 **/
#define DCGM_MAX_NUM_SWITCHES 12

/**
 * Number of LwLink links per LwSwitch supported by DCGM
 */
#define DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH 18

/**
 * Maximum number of vGPU instances per physical GPU
 */
#define DCGM_MAX_VGPU_INSTANCES_PER_PGPU 32

/**
 * Max number of vGPUs supported on DCGM
 */
#define DCGM_MAX_NUM_VGPU_DEVICES   DCGM_MAX_NUM_DEVICES * DCGM_MAX_VGPU_INSTANCES_PER_PGPU

/**
 * Max length of the DCGM string field
 */
#define DCGM_MAX_STR_LENGTH     256

/**
 * Max number of clocks supported for a device
 */
#define DCGM_MAX_CLOCKS         256

/**
 * Max limit on the number of groups supported by DCGM
 */
#define DCGM_MAX_NUM_GROUPS      64

/**
 * Max number of active FBC sessions
 */
#define DCGM_MAX_FBC_SESSIONS   256
    

/**
 * Represents the size of a buffer that holds a vGPU type Name or vGPU class type or name of process running on vGPU instance.
 */
#define DCGM_VGPU_NAME_BUFFER_SIZE     64

/**
 * Represents the size of a buffer that holds a vGPU license string
 */
#define DCGM_GRID_LICENSE_BUFFER_SIZE  128

/**
 * Default compute mode -- multiple contexts per device
 */
#define DCGM_CONFIG_COMPUTEMODE_DEFAULT            0
    
/**
 * Compute-prohibited mode -- no contexts per device
 */
#define DCGM_CONFIG_COMPUTEMODE_PROHIBITED         1
    
/**
 * Compute-exclusive-process mode -- only one context per device, usable from multiple threads at 
 * a time
 */
#define DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS  2
    

/**
 * Default Port Number for DCGM Host Engine
 */
#define DCGM_HE_PORT_NUMBER 5555


/**
 * Creates a unique version number for each struct
 */
#define MAKE_DCGM_VERSION(typeName,ver) (unsigned int)(sizeof(typeName) | ((ver)<<24))

/***************************************************************************************************/




/**
 * Operation mode for DCGM
 * 
 * DCGM can run in auto-mode where it runs additional threads in the background to collect 
 * any metrics of interest and auto manages any operations needed for policy management.
 * 
 * DCGM can also operate in manual-mode where it's exelwtion is controlled by the user. In
 * this mode, the user has to periodically call APIs such as \ref dcgmPolicyTrigger and
 * \ref dcgmUpdateAllFields which tells DCGM to wake up and perform data collection and
 * operations needed for policy management.
 */
typedef enum dcgmOperationMode_enum 
{
    DCGM_OPERATION_MODE_AUTO   = 1,
    DCGM_OPERATION_MODE_MANUAL = 2
} dcgmOperationMode_t;
    
/**
 * When more than one value is returned from a query, which order should it be returned in?
 */
typedef enum dcgmOrder_enum
{
    DCGM_ORDER_ASCENDING  = 1, //!< Data with earliest (lowest) timestamps returned first
    DCGM_ORDER_DESCENDING = 2  //!< Data with latest (highest) timestamps returned first
} dcgmOrder_t;

/** 
 * Return values for DCGM API calls. 
 */
typedef enum dcgmReturn_enum
{
    DCGM_ST_OK                   =  0,  //!< Success
    DCGM_ST_BADPARAM             = -1,  //!< A bad parameter was passed to a function
    DCGM_ST_GENERIC_ERROR        = -3,  //!< A generic, unspecified error
    DCGM_ST_MEMORY               = -4,  //!< An out of memory error oclwrred
    DCGM_ST_NOT_CONFIGURED       = -5,  //!< Setting not configured
    DCGM_ST_NOT_SUPPORTED        = -6,  //!< Feature not supported
    DCGM_ST_INIT_ERROR           = -7,  //!< DCGM Init error
    DCGM_ST_LWML_ERROR           = -8,  //!< When LWML returns error
    DCGM_ST_PENDING              = -9,  //!< Object is in pending state of something else
    DCGM_ST_UNINITIALIZED        = -10, //!< Object is in undefined state
    DCGM_ST_TIMEOUT              = -11, //!< Requested operation timed out
    DCGM_ST_VER_MISMATCH         = -12, //!< Version mismatch between received and understood API
    DCGM_ST_UNKNOWN_FIELD        = -13, //!< Unknown field id
    DCGM_ST_NO_DATA              = -14, //!< No data is available
    DCGM_ST_STALE_DATA           = -15, //!< Data is considered stale
    DCGM_ST_NOT_WATCHED          = -16, //!< The given field id is not being updated by the cache manager
    DCGM_ST_NO_PERMISSION        = -17, //!< Do not have permission to perform the desired action
    DCGM_ST_GPU_IS_LOST          = -18, //!< GPU is no longer reachable
    DCGM_ST_RESET_REQUIRED       = -19, //!< GPU requires a reset
    DCGM_ST_FUNCTION_NOT_FOUND   = -20, //!< The function that was requested was not found (bindings only error)
    DCGM_ST_CONNECTION_NOT_VALID = -21, //!< The connection to the host engine is not valid any longer 
    DCGM_ST_GPU_NOT_SUPPORTED    = -22, //!< This GPU is not supported by DCGM
    DCGM_ST_GROUP_INCOMPATIBLE   = -23, //!< The GPUs of the provided group are not compatible with each other for the requested operation
    DCGM_ST_MAX_LIMIT            = -24, //!< Max limit reached for the object
    DCGM_ST_LIBRARY_NOT_FOUND    = -25, //!< DCGM library could not be found
    DCGM_ST_DUPLICATE_KEY        = -26, //!< Duplicate key passed to a function
    DCGM_ST_GPU_IN_SYNC_BOOST_GROUP = -27, //!<GPU is already a part of a sync boost group
    DCGM_ST_GPU_NOT_IN_SYNC_BOOST_GROUP = -28, //!<GPU is not a part of a sync boost group
    DCGM_ST_REQUIRES_ROOT        = -29,  //!< This operation cannot be performed when the host engine is running as non-root
    DCGM_ST_LWVS_ERROR           = -30, //!< DCGM GPU Diagnostic was successfully exelwted, but reported an error.
    DCGM_ST_INSUFFICIENT_SIZE    = -31, //!< An input argument is not large enough
    DCGM_ST_FIELD_UNSUPPORTED_BY_API = -32, //!< The given field ID is not supported by the API being called
    DCGM_ST_MODULE_NOT_LOADED    = -33, //!< This request is serviced by a module of DCGM that is not lwrrently loaded
    DCGM_ST_IN_USE               = -34, //!< The requested operation could not be completed because the affected resource is in use
    DCGM_ST_GROUP_IS_EMPTY       = -35, //!< This group is empty and the requested operation is not valid on an empty group
    DCGM_ST_PROFILING_NOT_SUPPORTED = -36, //!< Profiling is not supported for this group of GPUs or GPU.
    DCGM_ST_PROFILING_LIBRARY_ERROR = -37,  //!< The third-party Profiling module returned an unrecoverable error.
    DCGM_ST_PROFILING_MULTI_PASS = -38, //!< The requested profiling metrics cannot be collected in a single pass
    DCGM_ST_DIAG_ALREADY_RUNNING = -39, //!< A diag instance is already running, cannot run a new diag until the current one finishes.
    DCGM_ST_DIAG_BAD_JSON = -40,        //!< The DCGM GPU Diagnostic returned JSON that cannot be parsed
    DCGM_ST_DIAG_BAD_LAUNCH = -41,       //!< Error while launching the DCGM GPU Diagnostic
    DCGM_ST_DIAG_VARIANCE = -42, //!< There is too much variance while training the diagnostic
    DCGM_ST_DIAG_THRESHOLD_EXCEEDED = -43, //!< A field value met or exceeded the error threshold.
    DCGM_ST_INSUFFICIENT_DRIVER_VERSION = -44 //The installed driver version is insufficient for this API
} dcgmReturn_t;

static const char* errorString(dcgmReturn_t result)
{
    switch (result)
    {
        case DCGM_ST_OK:
            return "Success";
        case DCGM_ST_BADPARAM:
            return "Bad parameter passed to function";
        case DCGM_ST_GENERIC_ERROR:
            return "Generic unspecified error";
        case DCGM_ST_MEMORY:
            return "Out of memory error";
        case DCGM_ST_NOT_CONFIGURED:
            return "Setting not configured";
        case DCGM_ST_NOT_SUPPORTED:
            return "Feature not supported";
        case DCGM_ST_INIT_ERROR:
            return "DCGM initialization error";
        case DCGM_ST_LWML_ERROR:
            return "LWML error";
        case DCGM_ST_PENDING:
            return "Object is in a pending state";
        case DCGM_ST_UNINITIALIZED:
            return "Object is in an undefined state";
        case DCGM_ST_TIMEOUT:
            return "Timeout";
        case DCGM_ST_VER_MISMATCH:
            return "API version mismatch";
        case DCGM_ST_UNKNOWN_FIELD:
            return "Unknown field identifier";
        case DCGM_ST_NO_DATA:
            return "No data is available";
        case DCGM_ST_STALE_DATA:
            return "Only stale data is available";
        case DCGM_ST_NOT_WATCHED:
            return "Field is not being watched";
        case DCGM_ST_NO_PERMISSION:
            return "No permission";
        case DCGM_ST_GPU_IS_LOST:
            return "GPU is lost";
        case DCGM_ST_RESET_REQUIRED:
            return "GPU requires reset";
        case DCGM_ST_CONNECTION_NOT_VALID:
            return "Host engine connection invalid/disconnected";
        case DCGM_ST_GPU_NOT_SUPPORTED:
            return "This GPU is not supported by DCGM";
        case DCGM_ST_GROUP_INCOMPATIBLE:
            return "The GPUs of this group are incompatible with each other for the requested operation";
        case DCGM_ST_MAX_LIMIT:
            return "Max limit reached for the object";
        case DCGM_ST_LIBRARY_NOT_FOUND:
            return "DCGM library could not be found";
        case DCGM_ST_DUPLICATE_KEY:
            return "Duplicate Key passed to function";
        case DCGM_ST_GPU_IN_SYNC_BOOST_GROUP:
            return "GPU is a part of a Sync Boost Group";
        case DCGM_ST_GPU_NOT_IN_SYNC_BOOST_GROUP:
            return "GPU is not a part of Sync Boost Group";
        case DCGM_ST_REQUIRES_ROOT:
            return "Host engine is running as non-root";
        case DCGM_ST_LWVS_ERROR:
            return "DCGM GPU Diagnostic returned an error";
        case DCGM_ST_INSUFFICIENT_SIZE:
            return "An input argument is not large enough";
        case DCGM_ST_FIELD_UNSUPPORTED_BY_API:
            return "The given field ID is not supported by the API being called";
        case DCGM_ST_MODULE_NOT_LOADED:
            return "This request is serviced by a module of DCGM that is not lwrrently loaded";
        case DCGM_ST_IN_USE:
            return "The requested operation could not be completed because the affected resource is in use";
        case DCGM_ST_GROUP_IS_EMPTY:
            return "The specified group is empty, and this operation is incompatible with an empty group";
        case DCGM_ST_PROFILING_NOT_SUPPORTED:
            return "Profiling is not supported for this group of GPUs or GPU";
        case DCGM_ST_PROFILING_LIBRARY_ERROR: 
            return "The third-party Profiling module returned an unrecoverable error";
        case DCGM_ST_PROFILING_MULTI_PASS:
            return "The requested profiling metrics cannot be collected in a single pass";
        case DCGM_ST_DIAG_ALREADY_RUNNING:
            return "A diag instance is already running, cannot run a new diag until the current one finishes";
        case DCGM_ST_DIAG_BAD_JSON:
            return "The GPU Diagnostic returned Json that cannot be parsed.";
        case DCGM_ST_DIAG_BAD_LAUNCH:
            return "Error while launching the GPU Diagnostic.";
        case DCGM_ST_DIAG_VARIANCE:
            return "The results of training DCGM GPU Diagnostic cannot be trusted because they vary too much from run to run";
        case DCGM_ST_DIAG_THRESHOLD_EXCEEDED:
            return "A field value met or exceeded the error threshold.";
        case DCGM_ST_INSUFFICIENT_DRIVER_VERSION:
            return "The installed driver version is insufficient for this API";
        default:
            // Wrong error codes should be handled by the caller
            return 0;
    }
}

/**
 * Type of GPU groups
 */
typedef enum dcgmGroupType_enum     
{
    DCGM_GROUP_DEFAULT = 0,     //!< All the GPUs on the node are added to the group
    DCGM_GROUP_EMPTY   = 1,     //!< Creates an empty group
    DCGM_GROUP_DEFAULT_LWSWITCHES = 2 //!< All LwSwitches of the node are added to the group
} dcgmGroupType_t;

/**
 * Identifies for special DCGM groups
 */
#define DCGM_GROUP_ALL_GPUS       0x7fffffff
#define DCGM_GROUP_ALL_LWSWITCHES 0x7ffffffe

/** 
 * Maximum number of entities per entity group
 */
#define DCGM_GROUP_MAX_ENTITIES 64

/**
 * Represents the type of configuration to be fetched from the GPUs
 */
typedef enum dcgmConfigType_enum
{
    DCGM_CONFIG_TARGET_STATE = 0,          //!< The target configuration values to be applied
    DCGM_CONFIG_LWRRENT_STATE = 1          //!< The current configuration state
}dcgmConfigType_t;

/**
 * Represents the power cap for each member of the group.
 */
typedef enum dcgmConfigPowerLimitType_enum
{
    DCGM_CONFIG_POWER_CAP_INDIVIDUAL    = 0, //!< Represents the power cap to be applied for each member of the group
    DCGM_CONFIG_POWER_BUDGET_GROUP      = 1  //!< Represents the power budget for the entire group
}dcgmConfigPowerLimitType_t;

/** @} */


/***************************************************************************************************/
/** @defgroup dcgmStructs Structure definitions
 *  @{
 */
/***************************************************************************************************/
typedef void *dcgmHandle_t;   //!< Identifier for DCGM Handle
typedef void *dcgmGpuGrp_t;   //!< Identifier for a group of GPUs. A group can have one or more GPUs
typedef void *dcgmFieldGrp_t; //!< Identifier for a group of fields.
typedef void *dcgmStatus_t;   //!< Identifier for list of status codes

/**
 * Connection options for dcgmConnect_v2 (v1)
 * 
 * NOTE: This version is deprecated. use dcgmConnectV2Params_v2
 */
typedef struct 
{
    unsigned int version;                //!< Version number. Use dcgmConnectV2Params_version
    unsigned int persistAfterDisconnect; /*!< Whether to persist DCGM state modified by this conection 
                                              once the connection is terminated. Normally, all field 
                                              watches created by a connection are removed once a 
                                              connection goes away.
                                              1 = do not clean up after this connection. 
                                              0 = clean up after this connection */
} dcgmConnectV2Params_v1;
 
 /**
  * Version 1 for \ref dcgmConnectV2Params_v1
  */
 #define dcgmConnectV2Params_version1 MAKE_DCGM_VERSION(dcgmConnectV2Params_v1, 1)

/**
 * Connection options for dcgmConnect_v2 (v2)
 */
typedef struct 
{
    unsigned int version;                //!< Version number. Use dcgmConnectV2Params_version
    unsigned int persistAfterDisconnect; /*!< Whether to persist DCGM state modified by this conection 
                                              once the connection is terminated. Normally, all field 
                                              watches created by a connection are removed once a 
                                              connection goes away.
                                              1 = do not clean up after this connection. 
                                              0 = clean up after this connection */
    unsigned int timeoutMs;              /*!< When attempting to connect to the specified host engine, 
                                              how long should we wait in milliseconds before giving up */
    unsigned int addressIsUnixSocket;    /*!< Whether or not the passed-in address is a unix socket filename (1)
                                              or a TCP/IP address (0) */
} dcgmConnectV2Params_v2;

/**
 * Typedef for \ref dcgmConnectV2Params_v2
 */
 typedef dcgmConnectV2Params_v2 dcgmConnectV2Params_t;
 
 /**
  * Version 2 for \ref dcgmConnectV2Params_v2
  */
 #define dcgmConnectV2Params_version2 MAKE_DCGM_VERSION(dcgmConnectV2Params_v2, 2)
 
 /**
  * Latest version for \ref dcgmConnectV2Params_t
  */
 #define dcgmConnectV2Params_version dcgmConnectV2Params_version2

/**
 * Structure to store information for DCGM group
 */
typedef struct
{
    unsigned int version;                         //!< Version Number (use dcgmGroupInfo_version1)
    unsigned int count;                           //!< count of GPU IDs returned in \a gpuIdList
    unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES]; //!< List of GPU Ids part of the group
    char groupName[DCGM_MAX_STR_LENGTH];          //!< Group Name
}dcgmGroupInfo_v1;

/**
 * Version 1 for \ref dcgmGroupInfo_v1
 */
#define dcgmGroupInfo_version1 MAKE_DCGM_VERSION(dcgmGroupInfo_v1, 1)

/**
 * Represents a entityGroupId + entityId pair to uniquely identify a given entityId inside
 * a group of entities
 */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId;   //!< Entity Group ID entity belongs to
    dcgm_field_eid_t entityId;                 //!< Entity ID of the entity
} dcgmGroupEntityPair_t;

/**
 * Structure to store information for DCGM group
 */
typedef struct
{
    unsigned int version;                         //!< Version Number (use dcgmGroupInfo_version2)
    unsigned int count;                           //!< count of entityIds returned in \a entityList
    char groupName[DCGM_MAX_STR_LENGTH];          //!< Group Name
    dcgmGroupEntityPair_t entityList[DCGM_GROUP_MAX_ENTITIES]; //!< List of the entities that are in this group
}dcgmGroupInfo_v2;

/**
 * Typedef for \ref dcgmGroupInfo_v2
 */
typedef dcgmGroupInfo_v2 dcgmGroupInfo_t;

/**
 * Version 2 for \ref dcgmGroupInfo_v2
 */
#define dcgmGroupInfo_version2 MAKE_DCGM_VERSION(dcgmGroupInfo_v2, 2)

/**
 * Latest version for \ref dcgmGroupInfo_t
 */
#define dcgmGroupInfo_version dcgmGroupInfo_version2

/**
 * Maximum number of field groups that can exist
 */
#define DCGM_MAX_NUM_FIELD_GROUPS 64

/**
 * Maximum number of field IDs that can be in a single field group
 */
#define DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP 128

/**
 * Structure to represent information about a field group
 */
typedef struct
{
    unsigned int version;                                //!< Version number (dcgmFieldGroupInfo_version)
    unsigned int numFieldIds;                            //!< Number of entries in fieldIds[] that are valid
    dcgmFieldGrp_t fieldGroupId;                         //!< ID of this field group
    char fieldGroupName[DCGM_MAX_STR_LENGTH];            //!< Field Group Name
    unsigned short fieldIds[DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP]; //!< Field ids that belong to this group
} dcgmFieldGroupInfo_v1;

typedef dcgmFieldGroupInfo_v1 dcgmFieldGroupInfo_t;

/**
 * Version 1 for dcgmFieldGroupInfo_v1
 */
#define dcgmFieldGroupInfo_version1 MAKE_DCGM_VERSION(dcgmFieldGroupInfo_v1, 1)

/**
 * Latest version for dcgmFieldGroupInfo_t
 */
#define dcgmFieldGroupInfo_version dcgmFieldGroupInfo_version1


typedef struct
{
    unsigned int version;                                        //!< Version number (dcgmAllFieldGroupInfo_version)
    unsigned int numFieldGroups;                                 //!< Number of entries in fieldGroups[] that are populated
    dcgmFieldGroupInfo_t fieldGroups[DCGM_MAX_NUM_FIELD_GROUPS]; //!< Info about each field group
} dcgmAllFieldGroup_v1;

typedef dcgmAllFieldGroup_v1 dcgmAllFieldGroup_t;

/**
 * Version 1 for dcgmAllFieldGroup_v1
 */
#define dcgmAllFieldGroup_version1 MAKE_DCGM_VERSION(dcgmAllFieldGroup_v1, 1)

/**
 * Latest version for dcgmAllFieldGroup_t
 */
#define dcgmAllFieldGroup_version dcgmAllFieldGroup_version1


/**
 * Structure to represent error attributes
 */
typedef struct
{
    unsigned int gpuId;      //!<  Represents GPU ID
    short        fieldId;    //!<  One of DCGM_FI_?
    int          status;     //!<  One of DCGM_ST_?
}dcgmErrorInfo_t;

/**
 * Represents a set of memory, SM, and video clocks for a device. This can be current values or a target values based on context
 */
typedef struct
{
    int version;                //!< Version Number (dcgmClockSet_version)
    unsigned int memClock;      //!< Memory Clock  (Memory Clock value OR DCGM_INT32_BLANK to Ignore/Use compatible value with smClk)
    unsigned int smClock;       //!< SM Clock      (SM Clock value OR DCGM_INT32_BLANK to Ignore/Use compatible value with memClk)
}dcgmClockSet_v1;

/**
 * Typedef for \ref dcgmClockSet_v1
 */
typedef dcgmClockSet_v1  dcgmClockSet_t;

/**
 * Version 1 for \ref dcgmClockSet_v1
 */
#define dcgmClockSet_version1 MAKE_DCGM_VERSION(dcgmClockSet_v1, 1)

/**
 * Latest version for \ref dcgmClockSet_t
 */
#define dcgmClockSet_version dcgmClockSet_version1

/**
 * Represents list of supported clock sets for a device
 */
typedef struct {
    unsigned int version;                       //!< Version Number (dcgmDeviceSupportedClockSets_version)
    unsigned int count;                         //!< Number of supported clocks
    dcgmClockSet_t clockSet[DCGM_MAX_CLOCKS];   //!< Valid clock sets for the device. Upto \ref count entries are filled
}dcgmDeviceSupportedClockSets_v1;

/**
 * Typedef for \ref dcgmDeviceSupportedClockSets_v1
 */
typedef dcgmDeviceSupportedClockSets_v1 dcgmDeviceSupportedClockSets_t;

/**
 * Version 1 for \ref dcgmDeviceSupportedClockSets_v1
 */
#define dcgmDeviceSupportedClockSets_version1 MAKE_DCGM_VERSION(dcgmDeviceSupportedClockSets_v1, 1)

/**
 * Latest version for \ref dcgmDeviceSupportedClockSets_t
 */
#define dcgmDeviceSupportedClockSets_version dcgmDeviceSupportedClockSets_version1

/**
 * Represents accounting data for one process
 */
typedef struct {
    unsigned int version;                       //!< Version Number. Should match dcgmDevicePidAccountingStats_version
    unsigned int pid;                           //!< Process id of the process these stats are for
    unsigned int gpuUtilization;                //!< Percent of time over the process's lifetime during which one or more kernels was exelwting on the GPU.
                                                //! Set to DCGM_INT32_NOT_SUPPORTED if is not supported
    unsigned int memoryUtilization;             //!< Percent of time over the process's lifetime during which global (device) memory was being read or written.
                                                //! Set to DCGM_INT32_NOT_SUPPORTED if is not supported
    unsigned long long maxMemoryUsage;          //!< Maximum total memory in bytes that was ever allocated by the process.
                                                //! Set to DCGM_INT64_NOT_SUPPORTED if is not supported
    unsigned long long startTimestamp;          //!< CPU Timestamp in usec representing start time for the process
    unsigned long long activeTimeUsec;          //!< Amount of time in usec during which the compute context was active. Note that
                                                //! this does not mean the context was being used. endTimestamp
                                                //! can be computed as startTimestamp + activeTime
} dcgmDevicePidAccountingStats_v1;

/**
 * Typedef for \ref dcgmDevicePidAccountingStats_v1
 */
typedef dcgmDevicePidAccountingStats_v1 dcgmDevicePidAccountingStats_t;

 /**
 * Version 1 for \ref dcgmDevicePidAccountingStats_v1
 */
#define dcgmDevicePidAccountingStats_version1 MAKE_DCGM_VERSION(dcgmDevicePidAccountingStats_v1, 1)

/**
 * Latest version for \ref dcgmDevicePidAccountingStats_t
 */
#define dcgmDevicePidAccountingStats_version dcgmDevicePidAccountingStats_version1

/**
 * Represents thermal information
 */
 typedef struct {
    unsigned int version;                       //!< Version Number
    unsigned int slowdownTemp;                  //!< Slowdown temperature
    unsigned int shutdownTemp;                  //!< Shutdown temperature
}dcgmDeviceThermals_v1;

/**
 * Typedef for \ref dcgmDeviceThermals_v1
 */
typedef dcgmDeviceThermals_v1 dcgmDeviceThermals_t;

/**
 * Version 1 for \ref dcgmDeviceThermals_v1
 */
#define dcgmDeviceThermals_version1 MAKE_DCGM_VERSION(dcgmDeviceThermals_v1, 1)

/**
 * Latest version for \ref dcgmDeviceThermals_t
 */
#define dcgmDeviceThermals_version dcgmDeviceThermals_version1

/**
 * Represents various power limits
 */
typedef struct {
    unsigned int version;               //!< Version Number
    unsigned int lwrPowerLimit;         //!< Power management limit associated with this device (in W)
    unsigned int defaultPowerLimit;     //!< Power management limit effective at device boot (in W)
    unsigned int enforcedPowerLimit;    //!< Effective power limit that the driver enforces after taking into account all limiters (in W)
    unsigned int minPowerLimit;         //!< Minimum power management limit (in W)
    unsigned int maxPowerLimit;         //!< Maximum power management limit (in W)
}dcgmDevicePowerLimits_v1;

/**
 * Typedef for \ref dcgmDevicePowerLimits_v1
 */
typedef dcgmDevicePowerLimits_v1 dcgmDevicePowerLimits_t;

/**
 * Version 1 for \ref dcgmDevicePowerLimits_v1
 */
#define dcgmDevicePowerLimits_version1 MAKE_DCGM_VERSION(dcgmDevicePowerLimits_v1, 1)

/**
 * Latest version for \ref dcgmDevicePowerLimits_t
 */
#define dcgmDevicePowerLimits_version dcgmDevicePowerLimits_version1

/**
 * Represents device identifiers
 */
typedef struct
{
    unsigned int version;                           //!< Version Number (dcgmDeviceIdentifiers_version)
    char brandName[DCGM_MAX_STR_LENGTH];            //!< Brand Name
    char deviceName[DCGM_MAX_STR_LENGTH];           //!< Name of the device
    char pciBusId[DCGM_MAX_STR_LENGTH];             //!< PCI Bus ID
    char serial[DCGM_MAX_STR_LENGTH];               //!< Serial for the device
    char uuid[DCGM_MAX_STR_LENGTH];                 //!< UUID for the device
    char vbios[DCGM_MAX_STR_LENGTH];                //!< VBIOS version
    char inforomImageVersion[DCGM_MAX_STR_LENGTH];  //!< Inforom Image version
    unsigned int pciDeviceId;                       //!< The combined 16-bit device id and 16-bit vendor id
    unsigned int pciSubSystemId;                    //!< The 32-bit Sub System Device ID
    char driverVersion[DCGM_MAX_STR_LENGTH];        //!< Driver Version
    unsigned int virtualizationMode;                //!< Virtualization Mode
}dcgmDeviceIdentifiers_v1;

/**
 * Typedef for \ref dcgmDeviceIdentifiers_v1
 */
typedef dcgmDeviceIdentifiers_v1 dcgmDeviceIdentifiers_t;

/**
 * Version 1 for \ref dcgmDeviceIdentifiers_v1
 */
#define dcgmDeviceIdentifiers_version1 MAKE_DCGM_VERSION(dcgmDeviceIdentifiers_v1, 1)

/**
 * Latest version for \ref dcgmDeviceIdentifiers_t
 */
#define dcgmDeviceIdentifiers_version dcgmDeviceIdentifiers_version1

/**
 * Represents device memory and usage
 */
typedef struct
{
    unsigned int    version;    //!< Version Number (dcgmDeviceMemoryUsage_version)
    unsigned int    bar1Total;  //!< Total BAR1 size in megabytes
    unsigned int    fbTotal;    //!< Total framebuffer memory in megabytes
    unsigned int    fbUsed;     //!< Used framebuffer memory in megabytes
    unsigned int    fbFree;     //!< Free framebuffer memory in megabytes
}dcgmDeviceMemoryUsage_v1;

/**
 * Typedef for \ref dcgmDeviceMemoryUsage_v1
 */
typedef dcgmDeviceMemoryUsage_v1 dcgmDeviceMemoryUsage_t;

/**
 * Version 1 for \ref dcgmDeviceMemoryUsage_v1
 */
#define dcgmDeviceMemoryUsage_version1 MAKE_DCGM_VERSION(dcgmDeviceMemoryUsage_v1, 1)

/**
 * Latest version for \ref dcgmDeviceMemoryUsage_t
 */
#define dcgmDeviceMemoryUsage_version dcgmDeviceMemoryUsage_version1

 /**
 * Represents utilization values for vGPUs running on the device
 */
typedef struct
{
    unsigned int version;           //!< Version Number (dcgmDeviceVgpuUtilInfo_version)
    unsigned int vgpuId;            //!< vGPU instance ID
    unsigned int smUtil;            //!< GPU utilization for vGPU
    unsigned int memUtil;           //!< Memory utilization for vGPU
    unsigned int enlwtil;           //!< Encoder utilization for vGPU
    unsigned int delwtil;           //!< Decoder utilization for vGPU
}dcgmDeviceVgpuUtilInfo_v1;

/**
 * Typedef for \ref dcgmDeviceVgpuUtilInfo_v1
 */
typedef dcgmDeviceVgpuUtilInfo_v1 dcgmDeviceVgpuUtilInfo_t;

/**
 * Version 1 for \ref dcgmDeviceVgpuUtilInfo_v1
 */
#define dcgmDeviceVgpuUtilInfo_version1 MAKE_DCGM_VERSION(dcgmDeviceVgpuUtilInfo_v1, 1)

/**
 * Latest version for \ref dcgmDeviceVgpuUtilInfo_t
 */
#define dcgmDeviceVgpuUtilInfo_version dcgmDeviceVgpuUtilInfo_version1

 /**
 * Represents current encoder statistics for the given device/vGPU instance
 */
typedef struct
{
    unsigned int version;           //!< Version Number (dcgmDeviceEncStats_version)
    unsigned int sessionCount;      //!< Count of active encoder sessions
    unsigned int averageFps;        //!< Trailing average FPS of all active sessions
    unsigned int averageLatency;    //!< Encode latency in milliseconds
}dcgmDeviceEncStats_v1;

/**
 * Typedef for \ref dcgmDeviceEncStats_v1
 */
typedef dcgmDeviceEncStats_v1 dcgmDeviceEncStats_t;

/**
 * Version 1 for \ref dcgmDeviceEncStats_v1
 */
#define dcgmDeviceEncStats_version1 MAKE_DCGM_VERSION(dcgmDeviceEncStats_v1, 1)

/**
 * Latest version for \ref dcgmDeviceEncStats_t
 */
#define dcgmDeviceEncStats_version dcgmDeviceEncStats_version1

/**
 * Represents current frame buffer capture sessions statistics for the given device/vGPU instance
 */
typedef struct
{
    unsigned int version;           //!< Version Number (dcgmDeviceFbcStats_version)
    unsigned int sessionCount;      //!< Count of active FBC sessions
    unsigned int averageFps;        //!< Moving average new frames captured per second
    unsigned int averageLatency;    //!< Moving average new frame capture latency in microseconds
}dcgmDeviceFbcStats_v1;

/**
 * Typedef for \ref dcgmDeviceFbcStats_v1
 */
typedef dcgmDeviceFbcStats_v1 dcgmDeviceFbcStats_t;

/**
 * Version 1 for \ref dcgmDeviceFbcStats_v1
 */
#define dcgmDeviceFbcStats_version1 MAKE_DCGM_VERSION(dcgmDeviceFbcStats_v1, 1)

/**
 * Latest version for \ref dcgmDeviceEncStats_t
 */
#define dcgmDeviceFbcStats_version dcgmDeviceFbcStats_version1

/*
 * Represents frame buffer capture session type
 */
typedef enum dcgmFBCSessionType_enum
{
    DCGM_FBC_SESSION_TYPE_UNKNOWN = 0,     //!< Unknown
    DCGM_FBC_SESSION_TYPE_TOSYS,           //!< FB capture for a system buffer
    DCGM_FBC_SESSION_TYPE_LWDA,            //!< FB capture for a lwca buffer
    DCGM_FBC_SESSION_TYPE_VID,             //!< FB capture for a Vid buffer
    DCGM_FBC_SESSION_TYPE_HWENC,           //!< FB capture for a LWENC HW buffer
}dcgmFBCSessionType_t;

/**
 * Represents information about active FBC session on the given device/vGPU instance
 */
typedef struct
{
    unsigned int          version;              //!< Version Number (dcgmDeviceFbcSessionInfo_version)
    unsigned int          sessionId;            //!< Unique session ID
    unsigned int          pid;                  //!< Owning process ID
    unsigned int          vgpuId;               //!< vGPU instance ID (only valid on vGPU hosts, otherwise zero)
    unsigned int          displayOrdinal;       //!< Display identifier
    dcgmFBCSessionType_t  sessionType;          //!< Type of frame buffer capture session
    unsigned int          sessionFlags;         //!< Session flags
    unsigned int          hMaxResolution;       //!< Max horizontal resolution supported by the capture session
    unsigned int          vMaxResolution;       //!< Max vertical resolution supported by the capture session
    unsigned int          hResolution;          //!< Horizontal resolution requested by caller in capture call
    unsigned int          vResolution;          //!< Vertical resolution requested by caller in capture call
    unsigned int          averageFps;           //!< Moving average new frames captured per second
    unsigned int          averageLatency;       //!< Moving average new frame capture latency in microseconds
}dcgmDeviceFbcSessionInfo_v1;

/**
 * Typedef for \ref dcgmDeviceFbcSessionInfo_v1
 */
typedef dcgmDeviceFbcSessionInfo_v1 dcgmDeviceFbcSessionInfo_t;

/**
 * Version 1 for \ref dcgmDeviceFbcSessionInfo_v1
 */
#define dcgmDeviceFbcSessionInfo_version1 MAKE_DCGM_VERSION(dcgmDeviceFbcSessionInfo_v1, 1)

/**
 * Latest version for \ref dcgmDeviceFbcSessionInfo_t
 */
#define dcgmDeviceFbcSessionInfo_version dcgmDeviceFbcSessionInfo_version1

/**
 * Represents all the active FBC sessions on the given device/vGPU instance
 */
typedef struct
{
    unsigned int version;                                           //!< Version Number (dcgmDeviceFbcSessions_version)
    unsigned int sessionCount;                                      //!< Count of active FBC sessions
    dcgmDeviceFbcSessionInfo_t sessionInfo[DCGM_MAX_FBC_SESSIONS];  //!< Info about the active FBC session
}dcgmDeviceFbcSessions_v1;

/**
 * Typedef for \ref dcgmDeviceFbcSessions_v1
 */
typedef dcgmDeviceFbcSessions_v1 dcgmDeviceFbcSessions_t;

/**
 * Version 1 for \ref dcgmDeviceFbcSessions_v1
 */
#define dcgmDeviceFbcSessions_version1 MAKE_DCGM_VERSION(dcgmDeviceFbcSessions_v1, 1)

/**
 * Latest version for \ref dcgmDeviceFbcSessions_t
 */
#define dcgmDeviceFbcSessions_version dcgmDeviceFbcSessions_version1

/*
 * Represents type of encoder for capacity can be queried
 */
typedef enum dcgmEncoderQueryType_enum
{
    DCGM_ENCODER_QUERY_H264 = 0,
    DCGM_ENCODER_QUERY_HEVC = 1
}dcgmEncoderType_t;

 /**
 * Represents information about active encoder sessions on the given vGPU instance
 */
typedef struct
{
    unsigned int version;               //!< Version Number (dcgmDeviceVgpuEncSessions_version)
    union {
        unsigned int vgpuId;            //!< vGPU instance ID
        unsigned int sessionCount;
    } encoderSessionInfo;
    unsigned int       sessionId;       //!< Unique session ID
    unsigned int       pid;             //!< Process ID
    dcgmEncoderType_t  codecType;       //!< Video encoder type
    unsigned int       hResolution;     //!< Current encode horizontal resolution
    unsigned int       vResolution;     //!< Current encode vertical resolution
    unsigned int       averageFps;      //!< Moving average encode frames per second
    unsigned int       averageLatency;  //!< Moving average encode latency in milliseconds
}dcgmDeviceVgpuEncSessions_v1;

/**
 * Typedef for \ref dcgmDeviceVgpuEncSessions_v1
 */
typedef dcgmDeviceVgpuEncSessions_v1 dcgmDeviceVgpuEncSessions_t;

/**
 * Version 1 for \ref dcgmDeviceVgpuEncSessions_v1
 */
#define dcgmDeviceVgpuEncSessions_version1 MAKE_DCGM_VERSION(dcgmDeviceVgpuEncSessions_v1, 1)

/**
 * Latest version for \ref dcgmDeviceVgpuEncSessions_t
 */
#define dcgmDeviceVgpuEncSessions_version dcgmDeviceVgpuEncSessions_version1

 /**
 * Represents utilization values for processes running in vGPU VMs using the device
 */
typedef struct
{
    unsigned int version;                           //!< Version Number (dcgmDeviceVgpuProcessUtilInfo_version)
    union {
        unsigned int vgpuId;                        //!< vGPU instance ID
        unsigned int vgpuProcessSamplesCount;       //!< Count of processes running in the vGPU VM,for which utilization rates are being reported in this cycle.
    } vgpuProcessUtilInfo;
    unsigned int pid;                               //!< Process ID of the process running in the vGPU VM.
    char processName[DCGM_VGPU_NAME_BUFFER_SIZE];   //!< Process Name of process running in the vGPU VM.
    unsigned int smUtil;                            //!< GPU utilization of process running in the vGPU VM.
    unsigned int memUtil;                           //!< Memory utilization of process running in the vGPU VM.
    unsigned int enlwtil;                           //!< Encoder utilization of process running in the vGPU VM.
    unsigned int delwtil;                           //!< Decoder utilization of process running in the vGPU VM.
}dcgmDeviceVgpuProcessUtilInfo_v1;

/**
 * Typedef for \ref dcgmDeviceVgpuProcessUtilInfo_v1
 */
typedef dcgmDeviceVgpuProcessUtilInfo_v1 dcgmDeviceVgpuProcessUtilInfo_t;

/**
 * Version 1 for \ref dcgmDeviceVgpuProcessUtilInfo_v1
 */
#define dcgmDeviceVgpuProcessUtilInfo_version1 MAKE_DCGM_VERSION(dcgmDeviceVgpuProcessUtilInfo_v1, 1)

/**
 * Latest version for \ref dcgmDeviceVgpuProcessUtilInfo_t
 */
#define dcgmDeviceVgpuProcessUtilInfo_version dcgmDeviceVgpuProcessUtilInfo_version1

/**
 * Represents various IDs related to vGPU.
 */
typedef struct
{
    unsigned int    version;                                              //!< Version Number (dcgmDeviceVgpuIds_version)
    unsigned int    unusedSupportedVgpuTypeCount;                         //!< Unused Field
    unsigned int    unusedSupportedVgpuTypeIds[DCGM_MAX_NUM_DEVICES];     //!< Unused Field
    unsigned int    unusedcreatableVgpuTypeCount;                         //!< Unused Field
    unsigned int    unusedcreatableVgpuTypeIds[DCGM_MAX_NUM_DEVICES];     //!< Unused Field
}dcgmDeviceVgpuIds_v1;

/**
 * Typedef for \ref dcgmDeviceVgpuIds_v1
 */
typedef dcgmDeviceVgpuIds_v1 dcgmDeviceVgpuIds_t;

/**
 * Version 1 for \ref dcgmDeviceVgpuIds_v1
 */
#define dcgmDeviceVgpuIds_version1 MAKE_DCGM_VERSION(dcgmDeviceVgpuIds_v1, 1)

/**
 * Latest version for \ref dcgmDeviceVgpuIds_t
 */
#define dcgmDeviceVgpuIds_version dcgmDeviceVgpuIds_version1

/**
 * Represents static info related to vGPUs supported on the device.
 */
typedef struct
{
    unsigned int    version;                                            //!< Version number (dcgmDeviceVgpuTypeIdStaticInfo_version)
    union {
        unsigned int vgpuTypeId;
        unsigned int supportedVgpuTypeCount;
    } vgpuTypeInfo;                                                     //!< vGPU type ID and Supported vGPU type count
    char            vgpuTypeName[DCGM_VGPU_NAME_BUFFER_SIZE];           //!< vGPU type Name
    char            vgpuTypeClass[DCGM_VGPU_NAME_BUFFER_SIZE];          //!< Class of vGPU type
    char            vgpuTypeLicense[DCGM_GRID_LICENSE_BUFFER_SIZE];     //!< license of vGPU type
    int             deviceId;                                           //!< device ID of vGPU type
    int             subsystemId;                                        //!< Subsytem ID of vGPU type
    int             numDisplayHeads;                                    //!< Count of vGPU's supported display heads
    int             maxInstances;                                       //!< maximum number of vGPU instances creatable on a device for given vGPU type
    int             frameRateLimit;                                     //!< Frame rate limit value of the vGPU type
    int             maxResolutionX;                                     //!< vGPU display head's maximum supported resolution in X dimension
    int             maxResolutionY;                                     //!< vGPU display head's maximum supported resolution in Y dimension
    int             fbTotal;                                            //!< vGPU Total framebuffer size in megabytes
}dcgmDeviceVgpuTypeInfo_v1;

/**
 * Typedef for \ref dcgmDeviceVgpuTypeInfo_v1
 */
typedef dcgmDeviceVgpuTypeInfo_v1 dcgmDeviceVgpuTypeInfo_t;

/**
 * Version 1 for \ref dcgmDeviceVgpuTypeInfo_v1
 */
#define dcgmDeviceVgpuTypeInfo_version1 MAKE_DCGM_VERSION(dcgmDeviceVgpuTypeInfo_v1, 1)

/**
 * Latest version for \ref dcgmDeviceVgpuTypeInfo_t
 */
#define dcgmDeviceVgpuTypeInfo_version dcgmDeviceVgpuTypeInfo_version1

/**
 * Represents attributes corresponding to a device
 */
typedef struct
{
    unsigned int version;                                                                  //!< Version number (dcgmDeviceAttributes_version)
    dcgmDeviceSupportedClockSets_t clockSets;                                              //!< Supported clocks for the device
    dcgmDeviceThermals_t        thermalSettings;                                           //!< Thermal settings for the device
    dcgmDevicePowerLimits_t     powerLimits;                                               //!< Various power limits for the device
    dcgmDeviceIdentifiers_t     identifiers;                                               //!< Identifiers for the device
    dcgmDeviceMemoryUsage_t     memoryUsage;                                               //!< Memory usage info for the device
    dcgmDeviceVgpuIds_t         unusedVgpuIds;                                             //!< Unused Field
    unsigned int                unusedActiveVgpuInstanceCount;                             //!< Unused Field
    unsigned int                unusedVgpuInstanceIds[DCGM_MAX_NUM_DEVICES];               //!< Unused Field
}dcgmDeviceAttributes_v1;

/**
 * Typedef for \ref dcgmDeviceAttributes_v1
 */
typedef dcgmDeviceAttributes_v1 dcgmDeviceAttributes_t;

/**
 * Version 1 for \ref dcgmDeviceAttributes_v1
 */
#define dcgmDeviceAttributes_version1 MAKE_DCGM_VERSION(dcgmDeviceAttributes_v1, 1)

/**
 * Latest version for \ref dcgmDeviceAttributes_t
 */
#define dcgmDeviceAttributes_version dcgmDeviceAttributes_version1

/**
 * Maximum number of vGPU types per physical GPU
 */
#define DCGM_MAX_VGPU_TYPES_PER_PGPU 32

/**
 * Represents the vGPU attributes corresponding to a physical device
 */
typedef struct
{
    unsigned int                        version;                                                    //!< Version number (dcgmVgpuDeviceAttributes_version)
    unsigned int                        activeVgpuInstanceCount;                                    //!< Count of active vGPU instances on the device
    unsigned int                        activeVgpuInstanceIds[DCGM_MAX_VGPU_INSTANCES_PER_PGPU];    //!< List of vGPU instances
    unsigned int                        creatableVgpuTypeCount;                                     //!< Creatable vGPU type count
    unsigned int                        creatableVgpuTypeIds[DCGM_MAX_VGPU_TYPES_PER_PGPU];         //!< List of Creatable vGPU types
    unsigned int                        supportedVgpuTypeCount;                                     //!< Supported vGPU type count
    dcgmDeviceVgpuTypeInfo_t            supportedVgpuTypeInfo[DCGM_MAX_VGPU_TYPES_PER_PGPU];        //!< Info related to vGPUs supported on the device
    dcgmDeviceVgpuUtilInfo_t            vgpuUtilInfo[DCGM_MAX_VGPU_TYPES_PER_PGPU];                 //!< Utilizations specific to vGPU instance
    unsigned int                        gpuUtil;                                                    //!< GPU utilization
    unsigned int                        memCopyUtil;                                                //!< Memory utilization
    unsigned int                        enlwtil;                                                    //!< Encoder utilization
    unsigned int                        delwtil;                                                    //!< Decoder utilization
}dcgmVgpuDeviceAttributes_v6;

/**
 * Typedef for \ref dcgmVgpuDeviceAttributes_v6
 */
typedef dcgmVgpuDeviceAttributes_v6 dcgmVgpuDeviceAttributes_t;

/**
 * Version 6 for \ref dcgmVgpuDeviceAttributes_v6
 */
#define dcgmVgpuDeviceAttributes_version6 MAKE_DCGM_VERSION(dcgmVgpuDeviceAttributes_v6, 1)

/**
 * Latest version for \ref dcgmVgpuDeviceAttributes_t
 */
#define dcgmVgpuDeviceAttributes_version dcgmVgpuDeviceAttributes_version6

/**
 * Represents the size of a buffer that holds string related to attributes specific to vGPU instance
 */
#define DCGM_DEVICE_UUID_BUFFER_SIZE 80

/**
 * Represents attributes specific to vGPU instance
 */
typedef struct
{
    unsigned int                version;                                                //!< Version number (dcgmVgpuInstanceAttributes_version)
    char                        vmId[DCGM_DEVICE_UUID_BUFFER_SIZE];                     //!< VM ID of the vGPU instance
    char                        vmName[DCGM_DEVICE_UUID_BUFFER_SIZE];                   //!< VM name of the vGPU instance
    unsigned int                vgpuTypeId;                                             //!< Type ID of the vGPU instance
    char                        vgpuUuid[DCGM_DEVICE_UUID_BUFFER_SIZE];                 //!< UUID of the vGPU instance
    char                        vgpuDriverVersion[DCGM_DEVICE_UUID_BUFFER_SIZE];        //!< Driver version of the vGPU instance
    unsigned int                fbUsage;                                                //!< Fb usage of the vGPU instance
    unsigned int                licenseStatus;                                          //!< License status of the vGPU instance
    unsigned int                frameRateLimit;                                         //!< Frame rate limit of the vGPU instance
}dcgmVgpuInstanceAttributes_v1;

/**
 * Typedef for \ref dcgmVgpuInstanceAttributes_v1
 */
typedef dcgmVgpuInstanceAttributes_v1 dcgmVgpuInstanceAttributes_t;

/**
 * Version 1 for \ref dcgmVgpuInstanceAttributes_v1
 */
#define dcgmVgpuInstanceAttributes_version1 MAKE_DCGM_VERSION(dcgmVgpuInstanceAttributes_v1, 1)

/**
 * Latest version for \ref dcgmVgpuInstanceAttributes_t
 */
#define dcgmVgpuInstanceAttributes_version dcgmVgpuInstanceAttributes_version1

/**
 * Used to represent Performance state settings
 */
typedef struct
{
    unsigned int          syncBoost;    //!< Sync Boost Mode (0: Disabled, 1 : Enabled, DCGM_INT32_BLANK : Ignored). Note that using this setting may result in lower clocks than targetClocks
    dcgmClockSet_t        targetClocks; //!< Target clocks. Set smClock and memClock to DCGM_INT32_BLANK to ignore/use compatible values. For GPUs > Maxwell, setting this implies autoBoost=0
}dcgmConfigPerfStateSettings_t;

/**
 * Used to represents the power capping limit for each GPU in the group or to represent the power 
 * budget for the entire group
 */
typedef struct
{
    dcgmConfigPowerLimitType_t type;  //!< Flag to represent power cap for each GPU or power budget for the group of GPUs
    unsigned int val;                 //!< Power Limit in Watts (Set a value OR DCGM_INT32_BLANK to Ignore)
}dcgmConfigPowerLimit_t;

/**
 * Structure to represent default and target configuration for a device
 */
typedef struct
{
    unsigned int                  version;      //!< Version number (dcgmConfig_version)
    unsigned int                  gpuId;        //!< GPU ID
    unsigned int                  eccMode;      //!< ECC Mode  (0: Disabled, 1 : Enabled, DCGM_INT32_BLANK : Ignored)
    unsigned int                  computeMode;  //!< Compute Mode (One of DCGM_CONFIG_COMPUTEMODE_? OR DCGM_INT32_BLANK to Ignore)
    dcgmConfigPerfStateSettings_t perfState;    //!< Performance State Settings (clocks / boost mode)
    dcgmConfigPowerLimit_t        powerLimit;   //!< Power Limits
}dcgmConfig_v1;

/**
 * Typedef for \ref dcgmConfig_v1
 */
typedef dcgmConfig_v1 dcgmConfig_t;

/**
 * Version 1 for \ref dcgmConfig_v1
 */
#define dcgmConfig_version1 MAKE_DCGM_VERSION(dcgmConfig_v1, 1)

/**
 * Latest version for \ref dcgmConfig_t
 */
#define dcgmConfig_version dcgmConfig_version1

/**
 * Structure to represent default and target vgpu configuration for a device
 */
typedef struct
{
    unsigned int                  version;      //!< Version number (dcgmConfig_version)
    unsigned int                  gpuId;        //!< GPU ID
    unsigned int                  eccMode;      //!< ECC Mode  (0: Disabled, 1 : Enabled, DCGM_INT32_BLANK : Ignored)
    unsigned int                  computeMode;  //!< Compute Mode (One of DCGM_CONFIG_COMPUTEMODE_? OR DCGM_INT32_BLANK to Ignore)
    dcgmConfigPerfStateSettings_t perfState;    //!< Performance State Settings (clocks / boost mode)
    dcgmConfigPowerLimit_t        powerLimit;   //!< Power Limits
}dcgmVgpuConfig_v1;

/**
 * Typedef for \ref dcgmVgpuConfig_v1
 */
typedef dcgmVgpuConfig_v1 dcgmVgpuConfig_t;

/**
 * Version 1 for \ref dcgmVgpuConfig_v1
 */
#define dcgmVgpuConfig_version1 MAKE_DCGM_VERSION(dcgmVgpuConfig_v1, 1)

/**
 * Latest version for \ref dcgmVgpuConfig_t
 */
#define dcgmVgpuConfig_version dcgmVgpuConfig_version1

/**
 * Represents a callback to receive updates from asynchronous functions.
 * Lwrrently the only implemented callback function is dcgmPolicyRegister
 * and the void * data will be a pointer to dcgmPolicyCallbackResponse_t.
 * Ex.
 * dcgmPolicyCallbackResponse_t *callbackResponse = (dcgmPolicyCallbackResponse_t *) userData;
 * 
 */
typedef int (*fpRecvUpdates)(void *userData);

/*Remove from doxygen documentation
 *
 * Define the structure that contains specific policy information 
 */
typedef struct 
{
    // version must always be first
    unsigned int version;                   //!< Version number (dcgmPolicyViolation_version)

    unsigned int notifyOnEccDbe;            //!< true/false notification on ECC Double Bit Errors
    unsigned int notifyOnPciEvent;          //!< true/false notification on PCI Events
    unsigned int notifyOnMaxRetiredPages;   //!< number of retired pages to occur before notification
} dcgmPolicyViolation_v1;

/*Remove from doxygen documentation
 *
 * Represents the versioning for the dcgmPolicyViolation_v1 structure
 */

/*
 * Typedef for \ref dcgmPolicyViolation_v1
 */
typedef dcgmPolicyViolation_v1 dcgmPolicyViolation_t;

/*
 * Version 1 for \ref dcgmPolicyViolation_v1
 */
#define dcgmPolicyViolation_version1 MAKE_DCGM_VERSION(dcgmPolicyViolation_v1, 1)

/*
 * Latest version for \ref dcgmPolicyViolation_t
 */
#define dcgmPolicyViolation_version dcgmPolicyViolation_version1

/** 
 * Enumeration for policy conditions.
 * When used as part of dcgmPolicy_t these have corresponding parameters to 
 * allow them to be switched on/off or set specific violation thresholds
 */
typedef enum dcgmPolicyCondition_enum
{
    // these are bitwise rather than sequential
    DCGM_POLICY_COND_DBE               = 0x1,              //!< Double bit errors -- boolean in dcgmPolicyConditionParms_t
    DCGM_POLICY_COND_PCI               = 0x2,              //!< PCI events/errors -- boolean in dcgmPolicyConditionParms_t
    DCGM_POLICY_COND_MAX_PAGES_RETIRED = 0x4,              //!< Maximum number of retired pages -- number required in dcgmPolicyConditionParms_t
    DCGM_POLICY_COND_THERMAL           = 0x8,              //!< Thermal violation -- number required in dcgmPolicyConditionParms_t
    DCGM_POLICY_COND_POWER             = 0x10,             //!< Power violation -- number required in dcgmPolicyConditionParms_t
    DCGM_POLICY_COND_LWLINK            = 0x20,             //!< LWLINK errors -- boolean in dcgmPolicyConditionParms_t
    DCGM_POLICY_COND_XID               = 0x40,             //!< XID errors -- number required in dcgmPolicyConditionParms_t
} dcgmPolicyCondition_t;

#define DCGM_POLICY_COND_MAX 7

/**
 * Structure for policy condition parameters.
 * This structure contains a tag that represents the type of the value being passed
 * as well as a "val" which is a union of the possible value types.  For example,
 * to pass a true boolean: tag = BOOL, val.boolean = 1.
 */
typedef struct dcgmPolicyConditionParms_st
{
    enum {BOOL, LLONG} tag;       
    union {   
        unsigned int boolean;                
        unsigned long long llval;
    } val;
} dcgmPolicyConditionParms_t;

/**
 * Enumeration for policy modes
 */
typedef enum dcgmPolicyMode_enum
{
    DCGM_POLICY_MODE_AUTOMATED = 0,      //!< automatic mode
    DCGM_POLICY_MODE_MANUAL    = 1,      //!< manual mode
} dcgmPolicyMode_t;

/**
 * Enumeration for policy isolation modes
 */
typedef enum dcgmPolicyIsolation_enum
{
    DCGM_POLICY_ISOLATION_NONE = 0,      //!< no isolation of GPUs on error
} dcgmPolicyIsolation_t;

/**
 * Enumeration for policy actions
 */
typedef enum dcgmPolicyAction_enum
{
    DCGM_POLICY_ACTION_NONE     = 0,     //!< no action
    DCGM_POLICY_ACTION_GPURESET = 1,     //!< perform a GPU reset on violation
} dcgmPolicyAction_t;

/**
 * Enumeration for policy validation actions
 */
typedef enum dcgmPolicyValidation_enum
{
    DCGM_POLICY_VALID_NONE      = 0,      //!< no validation after an action is performed
    DCGM_POLICY_VALID_SV_SHORT  = 1,      //!< run a short System Validation on the system after failure
    DCGM_POLICY_VALID_SV_MED    = 2,      //!< run a medium System Validation test after failure
    DCGM_POLICY_VALID_SV_LONG   = 3,      //!< run a extensive System Validation test after failure
} dcgmPolicyValidation_t;

/**
 * Enumeration for policy failure responses
 */
typedef enum dcgmPolicyFailureResp_enum
{
    DCGM_POLICY_FAILURE_NONE = 0,        //!< on failure of validation perform no action
} dcgmPolicyFailureResp_t;

/** 
 * Structure to fill when a user queries for policy violations
 */
typedef struct 
{
    unsigned int gpuId;                 //!< gpu ID
    unsigned int violationOclwrred;     //!< a violation based on the bit values in \ref dcgmPolicyCondition_t
} dcgmPolicyViolationNotify_t;

/**
 * Define the structure that specifies a policy to be enforced for a GPU 
 */
typedef struct 
{
    // version must always be first
    unsigned int version;                   //!< version number (dcgmPolicy_version)

    dcgmPolicyCondition_t condition;        //!< Condition(s) to access \ref dcgmPolicyCondition_t
    dcgmPolicyMode_t mode;                  //!< Mode of operation \ref dcgmPolicyMode_t
    dcgmPolicyIsolation_t isolation;        //!< Isolation level after a policy violation \ref dcgmPolicyIsolation_t
    dcgmPolicyAction_t action;              //!< Action to perform after a policy violation \ref dcgmPolicyAction_t action
    dcgmPolicyValidation_t validation;      //!< Validation to perform after action is taken \ref dcgmPolicyValidation_t
    dcgmPolicyFailureResp_t response;       //!< Failure to validation response \ref dcgmPolicyFailureResp_t
    dcgmPolicyConditionParms_t parms[DCGM_POLICY_COND_MAX]; //!< Parameters for the \a condition fields
} dcgmPolicy_v1;

/**
 * Typedef for \ref dcgmPolicy_v1
 */
typedef dcgmPolicy_v1 dcgmPolicy_t;

/**
 * Version 1 for \ref dcgmPolicy_v1
 */
#define dcgmPolicy_version1 MAKE_DCGM_VERSION(dcgmPolicy_v1, 1)

/**
 * Latest version for \ref dcgmPolicy_t
 */
#define dcgmPolicy_version dcgmPolicy_version1


/**
 * Define the ECC DBE return structure
 */
typedef struct
{
    long long timestamp;                                //!< timestamp of the error
    enum {L1, L2, DEVICE, REGISTER, TEXTURE} location;  //!< location of the error
    unsigned int numerrors;                             //!< number of errors
} dcgmPolicyConditionDbe_t;

/**
 * Define the PCI replay error return structure
 */
typedef struct
{
    long long timestamp;                                //!< timestamp of the error
    unsigned int counter;                               //!< value of the PCIe replay counter
} dcgmPolicyConditionPci_t;

/**
 * Define the maximum pending retired pages limit return structure
 */
typedef struct
{
    long long timestamp;                                //!< timestamp of the error
    unsigned int sbepages;                              //!< number of pending pages due to SBE
    unsigned int dbepages;                              //!< number of pending pages due to DBE
} dcgmPolicyConditionMpr_t;

/** 
 * Define the thermal policy violations return structure
 */
typedef struct
{
    long long timestamp;                                //!< timestamp of the error
    unsigned int thermalViolation;                      //!< Temperature reached that violated policy
} dcgmPolicyConditionThermal_t;

/** 
 * Define the power policy violations return structure
 */
typedef struct
{
    long long timestamp;                                //!< timestamp of the error
    unsigned int powerViolation;                        //!< Power value reached that violated policy
} dcgmPolicyConditionPower_t;

/** 
 * Define the lwlink policy violations return structure
 */
typedef struct
{
    long long timestamp;                //!< timestamp of the error
    unsigned short fieldId;             //!<Lwlink counter field ID that violated policy
    unsigned int counter;               //!< Lwlink counter value that violated policy
} dcgmPolicyConditionLwlink_t;

/**
 * Define the xid policy violations return structure
 */
typedef struct
{
    long long timestamp;        //!< Timestamp of the error
    unsigned int errnum;        //!< The XID error number
} dcgmPolicyConditionXID_t;


/** 
 * Define the structure that is given to the callback function
 */
typedef struct
{
    //version must always be first
    unsigned int version;                     //!< version number (dcgmPolicyCallbackResponse_version)

    dcgmPolicyCondition_t condition;          //!< Condition that was violated 
    union {   
        dcgmPolicyConditionDbe_t dbe;         //!< ECC DBE return structure
        dcgmPolicyConditionPci_t pci;         //!< PCI replay error return structure
        dcgmPolicyConditionMpr_t mpr;         //!< Max retired pages limit return structure
        dcgmPolicyConditionThermal_t thermal; //!< Thermal policy violations return structure
        dcgmPolicyConditionPower_t power;     //!< Power policy violations return structure
        dcgmPolicyConditionLwlink_t lwlink;   //!< Lwlink policy violations return structure
        dcgmPolicyConditionXID_t xid;         //!< XID policy violations return structure
    } val;
} dcgmPolicyCallbackResponse_v1;


/**
 * Typedef for \ref dcgmPolicyCallbackResponse_v1
 */
typedef dcgmPolicyCallbackResponse_v1 dcgmPolicyCallbackResponse_t;

/**
 * Version 1 for \ref dcgmPolicyCallbackResponse_v1
 */
#define dcgmPolicyCallbackResponse_version1 MAKE_DCGM_VERSION(dcgmPolicyCallbackResponse_v1, 1)

/**
 * Latest version for \ref dcgmPolicyCallbackResponse_t
 */
#define dcgmPolicyCallbackResponse_version dcgmPolicyCallbackResponse_version1


#define DCGM_MAX_BLOB_LENGTH    4096 //!<  Set above size of largest blob entry. Lwrrently this is dcgmDeviceVgpuTypeInfo_v1

/**
 * This structure is used to represent value for the field to be queried.
 */
typedef struct
{
    // version must always be first
    unsigned int version;               //!< version number (dcgmFieldValue_version1)

    unsigned short fieldId;             //!<  One of DCGM_FI_?
    unsigned short fieldType;           //!< One of DCGM_FT_?
    int     status;                     //!< Status for the querying the field. DCGM_ST_OK or one of DCGM_ST_?
    int64_t ts;                         //!< Timestamp in usec since 1970 */
    union {
        int64_t i64;      //!<  Int64 value
        double  dbl;      //!< Double value
        char    str[DCGM_MAX_STR_LENGTH]; //!< NULL terminated string
        char    blob[DCGM_MAX_BLOB_LENGTH]; //!< Binary blob
    } value;            //!< Value
}dcgmFieldValue_v1;

/**
 * Version 1 for \ref dcgmFieldValue_v1
 */
#define dcgmFieldValue_version1 MAKE_DCGM_VERSION(dcgmFieldValue_v1, 1)

/**
 * This structure is used to represent value for the field to be queried.
 */
typedef struct
{
    // version must always be first
    unsigned int version;               //!< version number (dcgmFieldValue_version2)
    dcgm_field_entity_group_t entityGroupId; //!< Entity group this field value's entity belongs to
    dcgm_field_eid_t entityId;          //!< Entity this field value belongs to
    unsigned short fieldId;             //!<  One of DCGM_FI_?
    unsigned short fieldType;           //!< One of DCGM_FT_?
    int     status;                     //!< Status for the querying the field. DCGM_ST_OK or one of DCGM_ST_?
    unsigned int unused;                //!< Unused for now to align ts to an 8-byte boundary. */
    int64_t ts;                         //!< Timestamp in usec since 1970 */
    union {
        int64_t i64;      //!<  Int64 value
        double  dbl;      //!< Double value
        char    str[DCGM_MAX_STR_LENGTH]; //!< NULL terminated string
        char    blob[DCGM_MAX_BLOB_LENGTH]; //!< Binary blob
    } value;            //!< Value
}dcgmFieldValue_v2;

/**
 * Version 2 for \ref dcgmFieldValue_v2
 */
#define dcgmFieldValue_version2 MAKE_DCGM_VERSION(dcgmFieldValue_v2, 2)

/** 
 * Field value flags used by \ref dcgmEntitiesGetLatestValues
 **/
#define DCGM_FV_FLAG_LIVE_DATA     0x00000001 /** Retrieve live data from the driver rather than cached data. 
                                                  Warning: Setting this flag will result in multiple calls to the
                                                  LWPU driver that will be much slower than retrieving a cached 
                                                  value. */

/**
 * User callback function for processing one or more field updates. This callback will
 * be ilwoked one or more times per field until all of the expected field values have been
 * enumerated. It is up to the callee to detect when the field id changes
 *
 * @param gpuId                IN: GPU ID of the GPU this field value set belongs to
 * @param values               IN: Field values. These values must be copied as they will
 *                                 be destroyed as soon as this call returns.
 * @param numValues            IN: Number of entries that are valid in values[]
 * @param userData             IN: User data pointer passed to the update function that generated
 *                                 this callback
 *
 * Returns 0 if OK
 *        <0 if enumeration should stop. This allows to callee to abort field value enumeration.
 *
 */
typedef int (*dcgmFieldValueEnumeration_f)(unsigned int gpuId, dcgmFieldValue_v1 *values,
                                           int numValues, void *userData);

/**
 * User callback function for processing one or more field updates. This callback will
 * be ilwoked one or more times per field until all of the expected field values have been
 * enumerated. It is up to the callee to detect when the field id changes
 *
 * @param entityGroupId        IN: entityGroup of the entity this field value set belongs to
 * @param entityId             IN: Entity this field value set belongs to
 * @param values               IN: Field values. These values must be copied as they will
 *                                 be destroyed as soon as this call returns.
 * @param numValues            IN: Number of entries that are valid in values[]
 * @param userData             IN: User data pointer passed to the update function that generated
 *                                 this callback
 *
 * Returns 0 if OK
 *        <0 if enumeration should stop. This allows to callee to abort field value enumeration.
 *
 */
typedef int (*dcgmFieldValueEntityEnumeration_f)(dcgm_field_entity_group_t entityGroupId, 
                                                 dcgm_field_eid_t entityId,
                                                 dcgmFieldValue_v1 *values,
                                                 int numValues, void *userData);


/**
 * Summary of time series data in int64 format.
 * Each value will either be set or be a BLANK value. Check for
 * blank with the DCGM_INT64_IS_BLANK() macro. See dcgmvalue.h for the actual
 * values of BLANK values */
typedef struct
{
    long long milwalue; //!< Minimum value of the samples looked at 
    long long maxValue; //!< Maximum value of the samples looked at 
    long long average;  //!< Simple average of the samples looked at. Blank values are ignored for this callwlation
} dcgmStatSummaryInt64_t;

/**
 * Same as dcgmStatSummaryInt64_t, but with 32-bit integer values 
 */
typedef struct
{
    int milwalue; //!< Minimum value of the samples looked at
    int maxValue; //!< Maximum value of the samples looked at
    int average;  //!< Simple average of the samples looked at. Blank values are ignored for this callwlation
} dcgmStatSummaryInt32_t;

/**
 * Summary of time series data in double-precision format.
 * Each value will either be set or be a BLANK value. Check for
 * blank with the DCGM_FP64_IS_BLANK() macro. See dcgmvalue.h for the actual
 * values of BLANK values */
typedef struct
{
    double milwalue; //!< Minimum value of the samples looked at
    double maxValue; //!< Maximum value of the samples looked at
    double average;  //!< Simple average of the samples looked at. Blank values are ignored for this callwlation
} dcgmStatSummaryFp64_t;

/**
 * Systems structure used to enable or disable health watch systems
 */
typedef enum dcgmHealthSystems_enum
{
    DCGM_HEALTH_WATCH_PCIE      = 0x1,                     //!< PCIe system watches (must have 1m of data before query)
    DCGM_HEALTH_WATCH_LWLINK    = 0x2,                     //!< LWLINK system watches
    DCGM_HEALTH_WATCH_PMU       = 0x4,                     //!< Power management unit watches
    DCGM_HEALTH_WATCH_MLW       = 0x8,                     //!< Microcontroller unit watches
    DCGM_HEALTH_WATCH_MEM       = 0x10,                    //!< Memory watches
    DCGM_HEALTH_WATCH_SM        = 0x20,                    //!< Streaming multiprocessor watches
    DCGM_HEALTH_WATCH_INFOROM   = 0x40,                    //!< Inforom watches
    DCGM_HEALTH_WATCH_THERMAL   = 0x80,                    //!< Temperature watches (must have 1m of data before query)
    DCGM_HEALTH_WATCH_POWER     = 0x100,                   //!< Power watches (must have 1m of data before query)
    DCGM_HEALTH_WATCH_DRIVER    = 0x200,                   //!< Driver-related watches
    DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL = 0x400,           //!< Non-fatal errors in LwSwitch
    DCGM_HEALTH_WATCH_LWSWITCH_FATAL = 0x800,              //!< Fatal errors in LwSwitch

    // ...
    DCGM_HEALTH_WATCH_ALL       = 0xFFFFFFFF               //!< All watches enabled
} dcgmHealthSystems_t;

#define DCGM_HEALTH_WATCH_COUNT_V1 10                      //!< For iterating through the dcgmHealthSystems_v1 enum
#define DCGM_HEALTH_WATCH_COUNT_V2 12                      //!< For iterating through the dcgmHealthSystems_v2 enum

/**
 * Health Watch test results
 */
typedef enum dcgmHealthWatchResult_enum
{
    DCGM_HEALTH_RESULT_PASS = 0,                       //!< All results within this system are reporting normal
    DCGM_HEALTH_RESULT_WARN = 10,                       //!< A warning has been issued, refer to the response for more information
    DCGM_HEALTH_RESULT_FAIL = 20,                       //!< A failure has been issued, refer to the response for more information
} dcgmHealthWatchResults_t;

/**
 * Health Response structure version 1. GPU Only
 */
typedef struct 
{
    unsigned int version;                                //!< version number (dcgmHealthResponse_version)
    dcgmHealthWatchResults_t overallHealth;              //!< The overall health of the system.  \ref dcgmHealthWatchResults_t
    unsigned int gpuCount;                               //!< The number of GPUs with warnings/errors
    struct {
        unsigned int gpuId;                              //!< GPU ID for which this data is valid
        dcgmHealthWatchResults_t overallHealth;          //!< overall health of this GPU
        unsigned int incidentCount;                      //!< The number of systems that encountered a warning/error
        struct {
            dcgmHealthSystems_t system;                  //!< system to which this information belongs
            dcgmHealthWatchResults_t health;             //!< health of the specified system on this GPU
            char errorString[1024];                      //!< information about the error(s) or warning(s) flagged
        } systems[DCGM_HEALTH_WATCH_COUNT_V1];         
    } gpu[DCGM_MAX_NUM_DEVICES];
} dcgmHealthResponse_v1;

/**
 * Version 1 for \ref dcgmHealthResponse_v1
 */
#define dcgmHealthResponse_version1 MAKE_DCGM_VERSION(dcgmHealthResponse_v1, 1)

/**
 * Health Response structure version 2 - LwSwitch-compatible
 */
typedef struct 
{
    unsigned int version;                                //!< version number (dcgmHealthResponse_version)
    dcgmHealthWatchResults_t overallHealth;              //!< The overall health of the system.  \ref dcgmHealthWatchResults_t
    unsigned int entityCount;                            //!< The number of entities with warnings/errors
    struct {
        dcgm_field_entity_group_t entityGroupId;         //!< entity group entityId belongs to
        dcgm_field_eid_t entityId;                       //!< entity for which this data is valid
        dcgmHealthWatchResults_t overallHealth;          //!< overall health of this entity
        unsigned int incidentCount;                      //!< The number of systems that encountered a warning/error
        struct {
            dcgmHealthSystems_t system;                  //!< system to which this information belongs
            dcgmHealthWatchResults_t health;             //!< health of the specified system on this entity
            char errorString[1024];                      //!< information about the error(s) or warning(s) flagged
        } systems[DCGM_HEALTH_WATCH_COUNT_V2];
    } entities[DCGM_GROUP_MAX_ENTITIES];
} dcgmHealthResponse_v2;

/**
 * Version 2 for \ref dcgmHealthResponse_v2
 */
#define dcgmHealthResponse_version2 MAKE_DCGM_VERSION(dcgmHealthResponse_v2, 2)

typedef struct
{
    char         msg[1024];
    unsigned int code;
} dcgmDiagErrorDetail_t;

/**
 * Health Response structure version 3 - LwSwitch-compatible and uses error codes for easier processing
 */
typedef struct 
{
    unsigned int version;                                //!< version number (dcgmHealthResponse_version)
    dcgmHealthWatchResults_t overallHealth;              //!< The overall health of the system.  \ref dcgmHealthWatchResults_t
    unsigned int entityCount;                            //!< The number of entities with warnings/errors
    struct {
        dcgm_field_entity_group_t entityGroupId;         //!< entity group entityId belongs to
        dcgm_field_eid_t entityId;                       //!< entity for which this data is valid
        dcgmHealthWatchResults_t overallHealth;          //!< overall health of this entity
        unsigned int incidentCount;                      //!< The number of systems that encountered a warning/error
        struct {
            dcgmHealthSystems_t system;                  //!< system to which this information belongs
            dcgmHealthWatchResults_t health;             //!< health of the specified system on this entity
            dcgmDiagErrorDetail_t errors[4];             //!< Information about the error(s) and their error codes
            unsigned int          errorCount;            //!< count of errors so far for this system
        } systems[DCGM_HEALTH_WATCH_COUNT_V2];
    } entities[DCGM_GROUP_MAX_ENTITIES];
} dcgmHealthResponse_v3;

/**
 * Version 3 for \ref dcgmHealthResponse_v3
 */
#define dcgmHealthResponse_version3 MAKE_DCGM_VERSION(dcgmHealthResponse_v3, 3)

/**
 * Typedef for \ref dcgmHealthResponse_v3
 */
typedef dcgmHealthResponse_v3 dcgmHealthResponse_t;

/**
 * Latest version for \ref dcgmHealthResponse_t
 */
#define dcgmHealthResponse_version dcgmHealthResponse_version3


#define DCGM_MAX_PID_INFO_NUM 16
/** 
 * per process utilization rates
 */
typedef struct
{
    unsigned int pid;
    double smUtil;
    double memUtil;
}dcgmProcessUtilInfo_t;

/**
 *Internal structure used to get the PID and the corresponding utilization rate
 */
typedef struct
{
    double util;
    unsigned int pid;
}dcgmProcessUtilSample_t;


/**
 * Info corresponding to single PID
 */
typedef struct
{
    unsigned int gpuId;                     //!< ID of the GPU this pertains to. GPU_ID_ILWALID = summary information for multiple GPUs

    /* All of the following are during the process's lifetime */

    long long energyConsumed;               //!< Energy consumed by the gpu in milliwatt-seconds
    dcgmStatSummaryInt64_t pcieRxBandwidth; //!< PCI-E bytes read from the GPU 
    dcgmStatSummaryInt64_t pcieTxBandwidth; //!< PCI-E bytes written to the GPU 
    long long pcieReplays;                  //!< Count of PCI-E replays that oclwrred 
    long long startTime;                      //!< Process start time in microseconds since 1970
    long long endTime;                        //!< Process end time in microseconds since 1970 or reported as 0 if the process is not completed
    dcgmProcessUtilInfo_t processUtilization; //!< Process SM and Memory Utilization (in percent)
    dcgmStatSummaryInt32_t smUtilization;     //!< GPU SM Utilization in percent 
    dcgmStatSummaryInt32_t memoryUtilization; //!< GPU Memory Utilization in percent
    unsigned int eccSingleBit;                //!< Count of ECC single bit errors that oclwrred 
    unsigned int eccDoubleBit;                //!< Count of ECC double bit errors that oclwrred 
    dcgmStatSummaryInt32_t memoryClock;       //!< Memory clock in MHz 
    dcgmStatSummaryInt32_t smClock;           //!< SM clock in MHz

    int numXidCriticalErrors;                 //!< Number of valid entries in xidCriticalErrorsTs
    long long xidCriticalErrorsTs[10];        //!< Timestamps of the critical XID errors that oclwrred

    int numOtherComputePids;                  //!< Count of otherComputePids entries that are valid 
    unsigned int otherComputePids[DCGM_MAX_PID_INFO_NUM];        //!< Other compute processes that ran. 0=no process 

    int numOtherGraphicsPids;                 //!< Count of otherGraphicsPids entries that are valid 
    unsigned int otherGraphicsPids[DCGM_MAX_PID_INFO_NUM];       //!< Other graphics processes that ran. 0=no process

    long long maxGpuMemoryUsed;               //!< Maximum amount of GPU memory that was used in bytes

    long long powerViolationTime;             //!< Number of microseconds we were at reduced clocks due to power violation
    long long thermalViolationTime;           //!< Number of microseconds we were at reduced clocks due to thermal violation
    long long reliabilityViolationTime;       //!< Amount of microseconds we were at reduced clocks due to the reliability limit 
    long long boardLimitViolationTime;        //!< Amount of microseconds we were at reduced clocks due to being at the board's max voltage
    long long lowUtilizationTime;             //!< Amount of microseconds we were at reduced clocks due to low utilization
    long long syncBoostTime;                  //!< Amount of microseconds we were at reduced clocks due to sync boost
    dcgmHealthWatchResults_t overallHealth;              //!< The overall health of the system .  \ref dcgmHealthWatchResults_t
    unsigned int incidentCount;
    struct {
        dcgmHealthSystems_t system;                  //!< system to which this information belongs
        dcgmHealthWatchResults_t health;             //!< health of the specified system on this GPU
    } systems[DCGM_HEALTH_WATCH_COUNT_V1];
} dcgmPidSingleInfo_t;

/**
 * To store process statistics
 */
typedef struct
{
    unsigned int version;         //!< Version of this message  (dcgmPidInfo_version)
    unsigned int pid;             //!< PID of the process
    unsigned int unused;
    int numGpus;                  //!< Number of GPUs that are valid in GPUs
    dcgmPidSingleInfo_t summary;  //!< Summary information for all GPUs listed in gpus[] 
    dcgmPidSingleInfo_t gpus[16]; //!< Per-GPU information for this PID
} dcgmPidInfo_v1;

/**
 * Typedef for \ref dcgmPidInfo_v1
 */
typedef dcgmPidInfo_v1 dcgmPidInfo_t;

 /**
 * Version 1 for \ref dcgmPidInfo_v1
 */
#define dcgmPidInfo_version1 MAKE_DCGM_VERSION(dcgmPidInfo_v1, 1)

/**
 * Latest version for \ref dcgmPidInfo_t
 */
#define dcgmPidInfo_version dcgmPidInfo_version1

/**
 * Info corresponding to the job on a GPU
 */
typedef struct
{
    unsigned int gpuId;                     //!< ID of the GPU this pertains to. GPU_ID_ILWALID = summary information for multiple GPUs

    /* All of the following are during the job's lifetime */

    long long energyConsumed;                 //!< Energy consumed in milliwatt-seconds
    dcgmStatSummaryFp64_t powerUsage;         //!< Power usage Min/Max/Avg in watts
    dcgmStatSummaryInt64_t pcieRxBandwidth;   //!< PCI-E bytes read from the GPU
    dcgmStatSummaryInt64_t pcieTxBandwidth;   //!< PCI-E bytes written to the GPU 
    long long pcieReplays;                    //!< Count of PCI-E replays that oclwrred 
    long long startTime;                      //!< User provided job start time in microseconds since 1970
    long long endTime;                        //!< User provided job end time in microseconds since 1970 
    dcgmStatSummaryInt32_t smUtilization;       //!< GPU SM Utilization in percent 
    dcgmStatSummaryInt32_t memoryUtilization;   //!< GPU Memory Utilization in percent
    unsigned int eccSingleBit;                //!< Count of ECC single bit errors that oclwrred 
    unsigned int eccDoubleBit;                //!< Count of ECC double bit errors that oclwrred 
    dcgmStatSummaryInt32_t memoryClock;       //!< Memory clock in MHz 
    dcgmStatSummaryInt32_t smClock;           //!< SM clock in MHz

    int numXidCriticalErrors;                 //!< Number of valid entries in xidCriticalErrorsTs
    long long xidCriticalErrorsTs[10];        //!< Timestamps of the critical XID errors that oclwrred

    int numComputePids;                       //!< Count of computePids entries that are valid 
    dcgmProcessUtilInfo_t computePidInfo[DCGM_MAX_PID_INFO_NUM];             //!< List of compute processes that ran during the job. 0=no process 

    int numGraphicsPids;                      //!< Count of graphicsPids entries that are valid 
    dcgmProcessUtilInfo_t graphicsPidInfo[DCGM_MAX_PID_INFO_NUM];            //!< List of compute processes that ran during the job. 0=no process
    
    long long maxGpuMemoryUsed;               //!< Maximum amount of GPU memory that was used in bytes

    long long powerViolationTime;             //!< Number of microseconds we were at reduced clocks due to power violation
    long long thermalViolationTime;           //!< Number of microseconds we were at reduced clocks due to thermal violation
    long long reliabilityViolationTime;       //!< Amount of microseconds we were at reduced clocks due to the reliability limit 
    long long boardLimitViolationTime;        //!< Amount of microseconds we were at reduced clocks due to being at the board's max voltage
    long long lowUtilizationTime;             //!< Amount of microseconds we were at reduced clocks due to low utilization
    long long syncBoostTime;                  //!< Amount of microseconds we were at reduced clocks due to sync boost
    dcgmHealthWatchResults_t overallHealth;              //!< The overall health of the system .  \ref dcgmHealthWatchResults_t
    unsigned int incidentCount;
    struct {
        dcgmHealthSystems_t system;                  //!< system to which this information belongs
        dcgmHealthWatchResults_t health;             //!< health of the specified system on this GPU
    } systems[DCGM_HEALTH_WATCH_COUNT_V1];
} dcgmGpuUsageInfo_t;


/**
 * To store job statistics
 * The following fields are not applicable in the summary info:
 * - pcieRxBandwidth (Min/Max)
 * - pcieTxBandwidth (Min/Max)
 * - smUtilization (Min/Max)
 * - memoryUtilization (Min/Max)
 * - memoryClock (Min/Max)
 * - smClock (Min/Max)
 * - processSamples
 *
 * The average value in the above fields (in the summary) is the
 * average of the averages of respective fields from all GPUs
 */
typedef struct
{
    unsigned int version;         //!< Version of this message  (dcgmPidInfo_version)
    int numGpus;                  //!< Number of GPUs that are valid in gpus[]
    dcgmGpuUsageInfo_t summary;   //!< Summary information for all GPUs listed in gpus[] 
    dcgmGpuUsageInfo_t gpus[16];  //!< Per-GPU information for this PID
} dcgmJobInfo_v2;

/**
 * Typedef for \ref dcgmJobInfo_v2
 */
typedef dcgmJobInfo_v2 dcgmJobInfo_t;

 /**
 * Version 2 for \ref dcgmJobInfo_v2
 */
#define dcgmJobInfo_version2 MAKE_DCGM_VERSION(dcgmJobInfo_v2, 2)

/**
 * Latest version for \ref dcgmJobInfo_t
 */
#define dcgmJobInfo_version dcgmJobInfo_version2


/**
 * Running process information for a compute or graphics process
 */
typedef struct
{
    unsigned int version;          //!< Version of this message (dcgmRunningProcess_version)
    unsigned int pid;              //!< PID of the process
    unsigned long long memoryUsed; //!< GPU memory used by this process in bytes.
} dcgmRunningProcess_v1;

/**
 * Typedef for \ref dcgmRunningProcess_v1
 */
typedef dcgmRunningProcess_v1 dcgmRunningProcess_t;

 /**
 * Version 1 for \ref dcgmRunningProcess_v1
 */
#define dcgmRunningProcess_version1 MAKE_DCGM_VERSION(dcgmRunningProcess_v1, 1)

/**
 * Latest version for \ref dcgmRunningProcess_t
 */
#define dcgmRunningProcess_version dcgmRunningProcess_version1

/**
 * Enumeration for diagnostic levels
 */
typedef enum
{
    DCGM_DIAG_LVL_ILWALID   = 0,      //!< Uninitialized
    DCGM_DIAG_LVL_SHORT     = 10,     //!< run a very basic health check on the system
    DCGM_DIAG_LVL_MED       = 20,     //!< run a medium-length diagnostic (a few minutes)
    DCGM_DIAG_LVL_LONG      = 30,     //!< run a extensive diagnostic (several minutes)
} dcgmDiagnosticLevel_t;

/**
 * Diagnostic test results
 */
typedef enum dcgmDiagResult_enum
{
    DCGM_DIAG_RESULT_PASS = 0,            //!< This test passed as diagnostics
    DCGM_DIAG_RESULT_SKIP = 1,            //!< This test was skipped
    DCGM_DIAG_RESULT_WARN = 2,            //!< This test passed with warnings
    DCGM_DIAG_RESULT_FAIL = 3,            //!< This test failed the diagnostics
    DCGM_DIAG_RESULT_NOT_RUN = 4,         //!< This test wasn't exelwted
} dcgmDiagResult_t;

typedef struct
{
    dcgmDiagResult_t status;              //!< The result of the test
    char             warning[1024];       //!< Warning returned from the test, if any
    char             info[1024];          //!< Information details returned from the test, if any
} dcgmDiagTestResult_v1;

typedef struct
{
    dcgmDiagResult_t      status;              //!< The result of the test
    dcgmDiagErrorDetail_t error;               //!< The error message and error code, if any
    char                  info[1024];          //!< Information details returned from the test, if any
} dcgmDiagTestResult_v2;


/**
 * Diagnostic per gpu tests - fixed indices for dcgmDiagResponsePerGpu_t.results[]
 */
typedef enum dcgmPerGpuTestIndices_enum
{
    DCGM_MEMORY_INDEX           = 0, //!< Memory test index
    DCGM_DIAGNOSTIC_INDEX       = 1, //!< Diagnostic test index
    DCGM_PCI_INDEX              = 2, //!< PCIe test index
    DCGM_SM_PERF_INDEX          = 3, //!< SM Stress test index
    DCGM_TARGETED_PERF_INDEX    = 4, //!< Targeted Stress test index
    DCGM_TARGETED_POWER_INDEX   = 5, //!< Targeted Power test index
    DCGM_MEMORY_BANDWIDTH_INDEX = 6, //!< Memory bandwidth test index
} dcgmPerGpuTestIndices_t;

// This test is only run by itself, so it can use the 0 slot
#define DCGM_CONTEXT_CREATE_INDEX 0

// Sync with dcgmPerGpuTestIndices_enum
#define DCGM_PER_GPU_TEST_COUNT 7

/**
 * Per GPU diagnostics result structure
 */
typedef struct
{
    unsigned int gpuId;                   //!< ID for the GPU this information pertains
    unsigned int hwDiagnosticReturn;      //!< Per GPU hardware diagnostic test return code
    dcgmDiagTestResult_v1 results[DCGM_PER_GPU_TEST_COUNT]; //!< Array with a result for each per-gpu test
} dcgmDiagResponsePerGpu_v1;

typedef struct
{
    unsigned int gpuId;                   //!< ID for the GPU this information pertains
    unsigned int hwDiagnosticReturn;      //!< Per GPU hardware diagnostic test return code
    dcgmDiagTestResult_v2 results[DCGM_PER_GPU_TEST_COUNT]; //!< Array with a result for each per-gpu test
} dcgmDiagResponsePerGpu_v2;

/**
 * Global diagnostics result structure
 */
typedef struct
{
    unsigned int version;                                        //!< version number (dcgmDiagResult_version)
    unsigned int gpuCount;                                       //!< number of valid per GPU results

    dcgmDiagResult_t blacklist;                                  //!< test for presence of blacklisted drivers (e.g. nouveau)
    dcgmDiagResult_t lwmlLibrary;                                //!< test for presence (and version) of LWML lib
    dcgmDiagResult_t lwdaMainLibrary;                            //!< test for presence (and version) of LWCA lib
    dcgmDiagResult_t lwdaRuntimeLibrary;                         //!< test for presence (and version) of LWCA RT lib
    dcgmDiagResult_t permissions;                                //!< test for character device permissions
    dcgmDiagResult_t persistenceMode;                            //!< test for persistence mode enabled
    dcgmDiagResult_t environment;                                //!< test for LWCA environment vars that may slow tests
    dcgmDiagResult_t pageRetirement;                             //!< test for pending frame buffer page retirement
    dcgmDiagResult_t inforom;                                    //!< test for inforom corruption
    dcgmDiagResult_t graphicsProcesses;                          //!< test for graphics processes running
    dcgmDiagResponsePerGpu_v1 perGpuResponses[DCGM_MAX_NUM_DEVICES];  //!< per GPU test results
    char systemError[1024];                                      //!< System-wide error reported from LWVS
} dcgmDiagResponse_v3;


#define DCGM_SWTEST_COUNT     10
#define LEVEL_ONE_MAX_RESULTS 16

typedef enum dcgmSoftwareTest_enum
{
    DCGM_SWTEST_BLACKLIST            = 0, // test for presence of blacklisted drivers (e.g. nouveau)
    DCGM_SWTEST_LWML_LIBRARY         = 1, // test for presence (and version) of LWML lib
    DCGM_SWTEST_LWDA_MAIN_LIBRARY    = 2, // test for presence (and version) of LWCA lib
    DCGM_SWTEST_LWDA_RUNTIME_LIBRARY = 3, // test for presence (and version) of LWCA RT lib
    DCGM_SWTEST_PERMISSIONS          = 4, // test for character device permissions
    DCGM_SWTEST_PERSISTENCE_MODE     = 5, // test for persistence mode enabled
    DCGM_SWTEST_ELWIRONMENT          = 6, // test for LWCA environment vars that may slow tests
    DCGM_SWTEST_PAGE_RETIREMENT      = 7, // test for pending frame buffer page retirement
    DCGM_SWTEST_GRAPHICS_PROCESSES   = 8, // test for graphics processes running
    DCGM_SWTEST_INFOROM              = 9, // test for inforom corruption
} dcgmSoftwareTest_t;

/**
 * Global diagnostics result structure
 */
typedef struct
{
    unsigned int version;                                             //!< version number (dcgmDiagResult_version)
    unsigned int gpuCount;                                            //!< number of valid per GPU results
    unsigned int levelOneTestCount;                                   //!< number of valid levelOne results

    dcgmDiagTestResult_v1     levelOneResults[LEVEL_ONE_MAX_RESULTS]; //!< Basic, system-wide test results.
    dcgmDiagResponsePerGpu_v1 perGpuResponses[DCGM_MAX_NUM_DEVICES];  //!< per GPU test results
    char systemError[1024];                                           //!< System-wide error reported from LWVS
    char trainingMsg[1024];                                           //!< Training Message
} dcgmDiagResponse_v4;

typedef struct
{
    unsigned int version;                                             //!< version number (dcgmDiagResult_version)
    unsigned int gpuCount;                                            //!< number of valid per GPU results
    unsigned int levelOneTestCount;                                   //!< number of valid levelOne results

    dcgmDiagTestResult_v2     levelOneResults[LEVEL_ONE_MAX_RESULTS]; //!< Basic, system-wide test results.
    dcgmDiagResponsePerGpu_v2 perGpuResponses[DCGM_MAX_NUM_DEVICES];  //!< per GPU test results
    dcgmDiagErrorDetail_t     systemError;                            //!< System-wide error reported from LWVS
    char                      trainingMsg[1024];                      //!< Training Message
} dcgmDiagResponse_v5;

/**
 * Typedef for \ref dcgmDiagResponse_v5
 */
typedef dcgmDiagResponse_v5 dcgmDiagResponse_t;

/**
 * Version 3 for \ref dcgmDiagResponse_v3
 */
#define dcgmDiagResponse_version3 MAKE_DCGM_VERSION(dcgmDiagResponse_v3, 3)

/**
 * Version 4 for \ref dcgmDiagResponse_v4
 */
#define dcgmDiagResponse_version4 MAKE_DCGM_VERSION(dcgmDiagResponse_v4, 4)

/**
 * Version 5 for \ref dcgmDiagResponse_v5
 */
#define dcgmDiagResponse_version5 MAKE_DCGM_VERSION(dcgmDiagResponse_v5, 5)

/**
 * Latest version for \ref dcgmDiagResponse_t
 */
#define dcgmDiagResponse_version dcgmDiagResponse_version5

/**
 * Represents level relationships within a system between two GPUs
 * The enums are spaced to allow for future relationships.  These match
 * the definitions in lwml.h
 */
typedef enum dcgmGpuLevel_enum
{
    // PCI connectivity states
    DCGM_TOPOLOGY_BOARD              = 0x1,   //!< multi-GPU board
    DCGM_TOPOLOGY_SINGLE             = 0x2,   //!< all devices that only need traverse a single PCIe switch
    DCGM_TOPOLOGY_MULTIPLE           = 0x4,   //!< all devices that need not traverse a host bridge
    DCGM_TOPOLOGY_HOSTBRIDGE         = 0x8,   //!< all devices that are connected to the same host bridge
    DCGM_TOPOLOGY_CPU                = 0x10,  //!< all devices that are connected to the same CPU but possibly multiple host bridges
    DCGM_TOPOLOGY_SYSTEM             = 0x20,  //!< all devices in the system

    // LWLINK connectivity states
    DCGM_TOPOLOGY_LWLINK1            = 0x0100, //!< GPUs connected via a single LWLINK link
    DCGM_TOPOLOGY_LWLINK2            = 0x0200, //!< GPUs connected via two LWLINK links
    DCGM_TOPOLOGY_LWLINK3            = 0x0400, //!< GPUs connected via three LWLINK links
    DCGM_TOPOLOGY_LWLINK4            = 0x0800, //!< GPUs connected via four LWLINK links
    DCGM_TOPOLOGY_LWLINK5            = 0x1000, //!< GPUs connected via five LWLINK links
    DCGM_TOPOLOGY_LWLINK6            = 0x2000, //!< GPUs connected via six LWLINK links
} dcgmGpuTopologyLevel_t;

// the PCI paths are the lower 8 bits of the path information
#define DCGM_TOPOLOGY_PATH_PCI(x) (dcgmGpuTopologyLevel_t) ((unsigned int)(x) & 0xFF)

// the LWLINK paths are the upper 24 bits of the path information
#define DCGM_TOPOLOGY_PATH_LWLINK(x) (dcgmGpuTopologyLevel_t) ((unsigned int)(x) & 0xFFFFFF00)

#define DCGM_AFFINITY_BITMASK_ARRAY_SIZE 8

/**
 * Device topology information
 */
typedef struct
{
    unsigned int version;                                                        //!< version number (dcgmDeviceTopology_version)

    unsigned long cpuAffinityMask[DCGM_AFFINITY_BITMASK_ARRAY_SIZE];            //!< affinity mask for the specified GPU
                                                                                //!<   a 1 represents affinity to the CPU in that bit position
                                                                                //!<   supports up to 256 cores
    unsigned int numGpus;                                                       //!< number of valid entries in gpuPaths

    struct {
        unsigned int gpuId;                                                     //!< gpuId to which the path represents
        dcgmGpuTopologyLevel_t path;                                            //!< path to the gpuId from this GPU. Note that this is a bitmask
                                                                                //!<   of DCGM_TOPOLOGY_* values and can contain both PCIe topology
                                                                                //!<   and LwLink topology where applicable. For instance:
                                                                                //!<   0x210 = DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_LWLINK2
                                                                                //!<   Use the macros DCGM_TOPOLOGY_PATH_LWLINK and 
                                                                                //!<   DCGM_TOPOLOGY_PATH_PCI to mask the LwLink and PCI paths, respectively.
        unsigned int localLwLinkIds;                                            //!< bits representing the local links connected to gpuId
                                                                                //!< e.g. if this field == 3, links 0 and 1 are connected, 
                                                                                //!< field is only valid if LWLINKS actually exist between GPUs
    } gpuPaths[DCGM_MAX_NUM_DEVICES - 1];
} dcgmDeviceTopology_v1;

/**
 * Typedef for \ref dcgmDeviceTopology_v1
 */
typedef dcgmDeviceTopology_v1 dcgmDeviceTopology_t;

/**
 * Version 1 for \ref dcgmDeviceTopology_v1
 */
#define dcgmDeviceTopology_version1 MAKE_DCGM_VERSION(dcgmDeviceTopology_v1, 1)

/**
 * Latest version for \ref dcgmDeviceTopology_t
 */
#define dcgmDeviceTopology_version dcgmDeviceTopology_version1

/**
 * Group topology information
 */
typedef struct
{
    unsigned int version;                                                        //!< version number (dcgmGroupTopology_version)

    unsigned long groupCpuAffinityMask[DCGM_AFFINITY_BITMASK_ARRAY_SIZE];        //!< the CPU affinity mask for all GPUs in the group
                                                                                 //!<   a 1 represents affinity to the CPU in that bit position
                                                                                 //!<   supports up to 256 cores
    unsigned int numaOptimalFlag;                                                //!< a zero value indicates that 1 or more GPUs
                                                                                //!<   in the group have a different CPU affinity and thus
                                                                                //!<   may not be optimal for certain algorithms
    dcgmGpuTopologyLevel_t slowestPath;                                            //!< the slowest path amongst GPUs in the group
} dcgmGroupTopology_v1;

/**
 * Typedef for \ref dcgmGroupTopology_v1
 */
typedef dcgmGroupTopology_v1 dcgmGroupTopology_t;

/**
 * Version 1 for \ref dcgmGroupTopology_v1
 */
#define dcgmGroupTopology_version1 MAKE_DCGM_VERSION(dcgmGroupTopology_v1, 1)

/**
 * Latest version for \ref dcgmGroupTopology_t
 */
#define dcgmGroupTopology_version dcgmGroupTopology_version1

/**
 * Identifies a level to retrieve field introspection info for
 */
typedef enum dcgmIntrospectLevel_enum
{
    DCGM_INTROSPECT_LVL_ILWALID = 0,     //!< Invalid value
    DCGM_INTROSPECT_LVL_FIELD = 1,       //!< Introspection data is grouped by field ID
    DCGM_INTROSPECT_LVL_FIELD_GROUP = 2, //!< Introspection data is grouped by field group
    DCGM_INTROSPECT_LVL_ALL_FIELDS,      //!< Introspection data is aggregated for all fields
} dcgmIntrospectLevel_t;

/**
 * Identifies the retrieval context for introspection API calls.
 */
typedef struct {
    unsigned int version;                //!< version number (dcgmIntrospectContext_version)
    dcgmIntrospectLevel_t introspectLvl; //!<Introspect Level \ref dcgmIntrospectLevel_t
    union
    {
        dcgmGpuGrp_t fieldGroupId;       //!< Only needed if \ref introspectLvl is DCGM_INTROSPECT_LVL_FIELD_GROUP
        unsigned short fieldId;          //!< Only needed if \ref introspectLvl is DCGM_INTROSPECT_LVL_FIELD
        unsigned long long contextId;    //!< Overloaded way to access both fieldGroupId and fieldId
    };
} dcgmIntrospectContext_v1;

/**
* Typedef for \ref dcgmIntrospectContext_v1
*/
typedef dcgmIntrospectContext_v1 dcgmIntrospectContext_t;

/**
 * Version 1 for \ref dcgmIntrospectContext_t
 */
#define dcgmIntrospectContext_version1 MAKE_DCGM_VERSION(dcgmIntrospectContext_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectContext_t
 */
#define dcgmIntrospectContext_version dcgmIntrospectContext_version1

/**
 * DCGM Exelwtion time info for a set of fields
 */
typedef struct
{
    unsigned int version;            //!< version number (dcgmIntrospectFieldsExecTime_version)

    long long meanUpdateFreqUsec;    //!< the mean update frequency of all fields

    double recentUpdateUsec;         //!< the sum of every field's most recent exelwtion time after they
                                     //!< have been normalized to \ref meanUpdateFreqUsec".
                                     //!< This is roughly how long it takes to update fields every \ref meanUpdateFreqUsec

    long long totalEverUpdateUsec;   //!< The total amount of time, ever, that has been spent updating all the fields
} dcgmIntrospectFieldsExecTime_v1;

/**
 * Typedef for \ref dcgmIntrospectFieldsExecTime_t
 */
typedef dcgmIntrospectFieldsExecTime_v1 dcgmIntrospectFieldsExecTime_t;

/**
 * Version 1 for \ref dcgmIntrospectFieldsExecTime_t
 */
#define dcgmIntrospectFieldsExecTime_version1 MAKE_DCGM_VERSION(dcgmIntrospectFieldsExecTime_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectFieldsExecTime_t
 */
#define dcgmIntrospectFieldsExecTime_version dcgmIntrospectFieldsExecTime_version1

/**
 * Full introspection info for field exelwtion time
 */
typedef struct {
    unsigned int version;                 //!< version number (dcgmIntrospectFullFieldsExecTime_version)

    dcgmIntrospectFieldsExecTime_v1 aggregateInfo;  //!< info that includes global and device scope

    int hasGlobalInfo;                          //!< 0 means \ref globalInfo is populated, !0 means it's not
    dcgmIntrospectFieldsExecTime_v1 globalInfo; //!< info that only includes global field scope

    unsigned short gpuInfoCount;                            //!< count of how many entries in \ref gpuInfo are populated
    unsigned int gpuIdsForGpuInfo[DCGM_MAX_NUM_DEVICES];    //!< the GPU ID at a given index identifies which gpu
                                                            //!< the corresponding entry in \ref gpuInfo is from

    dcgmIntrospectFieldsExecTime_v1 gpuInfo[DCGM_MAX_NUM_DEVICES];    //!< info that is separated by the
                                                                      //!< GPU ID that the watches were for
} dcgmIntrospectFullFieldsExecTime_v1;

/**
* typedef for \ref dcgmIntrospectFullFieldsExecTime_v1
*/
typedef dcgmIntrospectFullFieldsExecTime_v1 dcgmIntrospectFullFieldsExecTime_t;

/**
 * Version 1 for \ref dcgmIntrospectFullFieldsExecTime_t
 */
#define dcgmIntrospectFullFieldsExecTime_version1 MAKE_DCGM_VERSION(dcgmIntrospectFullFieldsExecTime_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectFullFieldsExecTime_t
 */
#define dcgmIntrospectFullFieldsExecTime_version dcgmIntrospectFullFieldsExecTime_version1

/**
 * State of DCGM metadata gathering.  If it is set to DISABLED then "Metadata" API
 * calls to DCGM are not supported.
 */
typedef enum dcgmIntrospectState_enum {
    DCGM_INTROSPECT_STATE_DISABLED = 0,
    DCGM_INTROSPECT_STATE_ENABLED = 1
} dcgmIntrospectState_t;

/**
 * DCGM Memory usage information
 */
typedef struct
{
    unsigned int version;     //!< version number (dcgmIntrospectMemory_version)
    long long bytesUsed;      //!< number of bytes
} dcgmIntrospectMemory_v1;

/**
 * Typedef for \ref dcgmIntrospectMemory_t
 */
typedef dcgmIntrospectMemory_v1 dcgmIntrospectMemory_t;

/**
 * Version 1 for \ref dcgmIntrospectMemory_t
 */
#define dcgmIntrospectMemory_version1 MAKE_DCGM_VERSION(dcgmIntrospectMemory_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectMemory_t
 */
#define dcgmIntrospectMemory_version dcgmIntrospectMemory_version1


/**
 * Full introspection info for field memory
 */
typedef struct {
    unsigned int version;                 //!< version number (dcgmIntrospectFullMemory_version)

    dcgmIntrospectMemory_v1 aggregateInfo;//!< info that includes global and device scope

    int hasGlobalInfo;                    //!< 0 means \ref globalInfo is populated, !0 means it's not
    dcgmIntrospectMemory_v1 globalInfo;   //!< info that only includes global field scope

    unsigned short gpuInfoCount;          //!< count of how many entries in \ref gpuInfo are populated
    unsigned int gpuIdsForGpuInfo[DCGM_MAX_NUM_DEVICES];  //!< the GPU ID at a given index identifies which gpu
                                                          //!< the corresponding entry in \ref gpuInfo is from

    dcgmIntrospectMemory_v1 gpuInfo[DCGM_MAX_NUM_DEVICES];  //!< info that is divided by the
                                                            //!< GPU ID that the watches were for
} dcgmIntrospectFullMemory_v1;

/**
* typedef for \ref dcgmIntrospectFullMemory_v1
*/
typedef dcgmIntrospectFullMemory_v1 dcgmIntrospectFullMemory_t;

/**
 * Version 1 for \ref dcgmIntrospectFullMemory_t
 */
#define dcgmIntrospectFullMemory_version1 MAKE_DCGM_VERSION(dcgmIntrospectFullMemory_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectFullMemory_t
 */
#define dcgmIntrospectFullMemory_version dcgmIntrospectFullMemory_version1

/**
 * DCGM CPU Utilization information.  Multiply values by 100 to get them in %.
 */
typedef struct
{
    unsigned int version; //!< version number (dcgmMetadataCpuUtil_version)
    double total;         //!< fraction of device's CPU resources that were used
    double kernel;        //!< fraction of device's CPU resources that were used in kernel mode
    double user;          //!< fraction of device's CPU resources that were used in user mode
} dcgmIntrospectCpuUtil_v1;

/**
 * Typedef for \ref dcgmIntrospectCpuUtil_t
 */
typedef dcgmIntrospectCpuUtil_v1 dcgmIntrospectCpuUtil_t;

/**
 * Version 1 for \ref dcgmIntrospectCpuUtil_t
 */
#define dcgmIntrospectCpuUtil_version1 MAKE_DCGM_VERSION(dcgmIntrospectCpuUtil_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectCpuUtil_t
 */
#define dcgmIntrospectCpuUtil_version dcgmIntrospectCpuUtil_version1

#define DCGM_MAX_CONFIG_FILE_LEN  10000
#define DCGM_MAX_TEST_NAMES     20
#define DCGM_MAX_TEST_NAMES_LEN 50
#define DCGM_MAX_TEST_PARMS     100
#define DCGM_MAX_TEST_PARMS_LEN 100
#define DCGM_GPU_LIST_LEN       50
#define DCGM_FILE_LEN           30
#define DCGM_PATH_LEN           128
#define DCGM_THROTTLE_MASK_LEN 50

// Flags options for running the GPU diagnostic
#define DCGM_RUN_FLAGS_VERBOSE     0x0001 // Output in verbose mode; include information as well as warnings
#define DCGM_RUN_FLAGS_STATSONFAIL 0x0002 // Output stats only on failure
#define DCGM_RUN_FLAGS_TRAIN       0x0004 // Train DCGM diagnostic and output a configuration file with golden values
#define DCGM_RUN_FLAGS_FORCE_TRAIN 0x0008 // Ignore warnings against training the diagnostic and train anyway
#define DCGM_RUN_FLAGS_FAIL_EARLY  0x0010 // Enable fail early checks for the Targeted Stress, Targeted Power, SM Stress, and Diagnostic tests

typedef struct
{
    unsigned int  version; //! < version of this message
    unsigned int  flags; //! < flags specifying binary options for running it. See DCGM_RUN_FLAGS_*
    unsigned int  debugLevel; //! < 0-5 for the debug level the GPU diagnostic will use for logging.
    dcgmGpuGrp_t  groupId; //! < group of GPUs to verify. Cannot be specified together with gpuList.
    dcgmPolicyValidation_t validate; //! < 0-3 for which tests to run. Optional.
    char          testNames[DCGM_MAX_TEST_NAMES][DCGM_MAX_TEST_NAMES_LEN]; //! < Specifed list of test names. Optional.
    char          testParms[DCGM_MAX_TEST_PARMS][DCGM_MAX_TEST_PARMS_LEN]; //! < Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
    char          gpuList[DCGM_GPU_LIST_LEN]; //! < Comma-separated list of gpus. Cannot be specified with the groupId.
    char          debugLogFile[DCGM_FILE_LEN]; //! < Alternate name for the debug log file that should be used
    char          statsPath[DCGM_PATH_LEN]; //! < Path that the plugin's statistics files should be written to
} dcgmRunDiag_v1;

typedef struct
{
    unsigned int  version; //! < version of this message
    unsigned int  flags; //! < flags specifying binary options for running it. See DCGM_RUN_FLAGS_*
    unsigned int  debugLevel; //! < 0-5 for the debug level the GPU diagnostic will use for logging.
    dcgmGpuGrp_t  groupId; //! < group of GPUs to verify. Cannot be specified together with gpuList.
    dcgmPolicyValidation_t validate; //! < 0-3 for which tests to run. Optional.
    char          testNames[DCGM_MAX_TEST_NAMES][DCGM_MAX_TEST_NAMES_LEN]; //! < Specifed list of test names. Optional.
    char          testParms[DCGM_MAX_TEST_PARMS][DCGM_MAX_TEST_PARMS_LEN]; //! < Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
    char          gpuList[DCGM_GPU_LIST_LEN]; //! < Comma-separated list of gpus. Cannot be specified with the groupId.
    char          debugLogFile[DCGM_FILE_LEN]; //! < Alternate name for the debug log file that should be used
    char          statsPath[DCGM_PATH_LEN]; //! < Path that the plugin's statistics files should be written to
    char          configFileContents[DCGM_MAX_CONFIG_FILE_LEN]; //! < Contents of lwvs config file (likely yaml)
} dcgmRunDiag_v2;

typedef struct
{
    unsigned int  version; //! < version of this message
    unsigned int  flags; //! < flags specifying binary options for running it. See DCGM_RUN_FLAGS_*
    unsigned int  debugLevel; //! < 0-5 for the debug level the GPU diagnostic will use for logging.
    dcgmGpuGrp_t  groupId; //! < group of GPUs to verify. Cannot be specified together with gpuList.
    dcgmPolicyValidation_t validate; //! < 0-3 for which tests to run. Optional.
    char          testNames[DCGM_MAX_TEST_NAMES][DCGM_MAX_TEST_NAMES_LEN]; //! < Specifed list of test names. Optional.
    char          testParms[DCGM_MAX_TEST_PARMS][DCGM_MAX_TEST_PARMS_LEN]; //! < Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
    char          gpuList[DCGM_GPU_LIST_LEN]; //! < Comma-separated list of gpus. Cannot be specified with the groupId.
    char          debugLogFile[DCGM_FILE_LEN]; //! < Alternate name for the debug log file that should be used
    char          statsPath[DCGM_PATH_LEN]; //! < Path that the plugin's statistics files should be written to
    char          configFileContents[DCGM_MAX_CONFIG_FILE_LEN]; //! < Contents of lwvs config file (likely yaml)
    char          throttleMask[DCGM_THROTTLE_MASK_LEN]; //! < Throttle reasons to ignore as either integer mask or csv list of reasons
} dcgmRunDiag_v3;

typedef struct
{
    unsigned int  version; //! < version of this message
    unsigned int  flags; //! < flags specifying binary options for running it. See DCGM_RUN_FLAGS_*
    unsigned int  debugLevel; //! < 0-5 for the debug level the GPU diagnostic will use for logging.
    dcgmGpuGrp_t  groupId; //! < group of GPUs to verify. Cannot be specified together with gpuList.
    dcgmPolicyValidation_t validate; //! < 0-3 for which tests to run. Optional.
    char          testNames[DCGM_MAX_TEST_NAMES][DCGM_MAX_TEST_NAMES_LEN]; //! < Specifed list of test names. Optional.
    char          testParms[DCGM_MAX_TEST_PARMS][DCGM_MAX_TEST_PARMS_LEN]; //! < Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
    char          gpuList[DCGM_GPU_LIST_LEN]; //! < Comma-separated list of gpus. Cannot be specified with the groupId.
    char          debugLogFile[DCGM_FILE_LEN]; //! < Alternate name for the debug log file that should be used
    char          statsPath[DCGM_PATH_LEN]; //! < Path that the plugin's statistics files should be written to
    char          configFileContents[DCGM_MAX_CONFIG_FILE_LEN]; //! < Contents of lwvs config file (likely yaml)
    char          throttleMask[DCGM_THROTTLE_MASK_LEN]; //! < Throttle reasons to ignore as either integer mask or csv list of reasons
    char          pluginPath[DCGM_PATH_LEN]; //! < Custom path to the diagnostic plugins
    unsigned int  trainingIterations; //! < Number of iterations for training 
    unsigned int  trainingVariance; //! < Acceptable training variance as a percentage of the value. (0-100)
    unsigned int  trainingTolerance; //! < Acceptable training tolerance as a percentage of the value. (0-100)
    char          goldelwaluesFile[DCGM_PATH_LEN]; //! < The path where the golden values should be recorded
} dcgmRunDiag_v4;

typedef struct
{
    unsigned int  version; //! < version of this message
    unsigned int  flags; //! < flags specifying binary options for running it. See DCGM_RUN_FLAGS_*
    unsigned int  debugLevel; //! < 0-5 for the debug level the GPU diagnostic will use for logging.
    dcgmGpuGrp_t  groupId; //! < group of GPUs to verify. Cannot be specified together with gpuList.
    dcgmPolicyValidation_t validate; //! < 0-3 for which tests to run. Optional.
    char          testNames[DCGM_MAX_TEST_NAMES][DCGM_MAX_TEST_NAMES_LEN]; //! < Specifed list of test names. Optional.
    char          testParms[DCGM_MAX_TEST_PARMS][DCGM_MAX_TEST_PARMS_LEN]; //! < Parameters to set for specified tests in the format: testName.parameterName=parameterValue. Optional.
    char          gpuList[DCGM_GPU_LIST_LEN]; //! < Comma-separated list of gpus. Cannot be specified with the groupId.
    char          debugLogFile[DCGM_PATH_LEN]; //! < Alternate name for the debug log file that should be used
    char          statsPath[DCGM_PATH_LEN]; //! < Path that the plugin's statistics files should be written to
    char          configFileContents[DCGM_MAX_CONFIG_FILE_LEN]; //! < Contents of lwvs config file (likely yaml)
    char          throttleMask[DCGM_THROTTLE_MASK_LEN]; //! < Throttle reasons to ignore as either integer mask or csv list of reasons
    char          pluginPath[DCGM_PATH_LEN]; //! < Custom path to the diagnostic plugins
    unsigned int  trainingIterations; //! < Number of iterations for training 
    unsigned int  trainingVariance; //! < Acceptable training variance as a percentage of the value. (0-100)
    unsigned int  trainingTolerance; //! < Acceptable training tolerance as a percentage of the value. (0-100)
    char          goldelwaluesFile[DCGM_PATH_LEN]; //! < The path where the golden values should be recorded
    unsigned int  failCheckInterval; //! < How often the fail early checks should occur when enabled.
} dcgmRunDiag_v5;

/**
 * Typedef for \ref dcgmRunDiag_t
 */
typedef dcgmRunDiag_v5 dcgmRunDiag_t;

/**
 * Version 1 for \ref dcgmRunDiag_t
 */
#define dcgmRunDiag_version1 MAKE_DCGM_VERSION(dcgmRunDiag_v1, 1)

/**
 * Version 2 for \ref dcgmRunDiag_t
 */
#define dcgmRunDiag_version2 MAKE_DCGM_VERSION(dcgmRunDiag_v2, 2)

/**
 * Version 3 for \ref dcgmRunDiag_t
 */
#define dcgmRunDiag_version3 MAKE_DCGM_VERSION(dcgmRunDiag_v3, 3)

/**
 * Version 4 for \ref dcgmRunDiag_t
 */
#define dcgmRunDiag_version4 MAKE_DCGM_VERSION(dcgmRunDiag_v4, 4)

/**
 * Version 5 for \ref dcgmRunDiag_t
 */
#define dcgmRunDiag_version5 MAKE_DCGM_VERSION(dcgmRunDiag_v5, 5)

/**
 * Latest version for \ref dcgmRunDiag_t
 */
#define dcgmRunDiag_version dcgmRunDiag_version5

/**
 * Flags for dcgmGetEntityGroupEntities's flags parameter
 */
#define DCGM_GEGE_FLAG_ONLY_SUPPORTED 0x00000001 //!< Only return entities that are supported by DCGM. 
                                                 //!< This mimics the behavior of dcgmGetAllSupportedDevices().

/**
 * Identifies a GPU LWLink error type returned by DCGM_FI_DEV_GPU_LWLINK_ERRORS
 */
typedef enum dcgmGpuLWLinkErrorType_enum
{
    DCGM_GPU_LWLINK_ERROR_RECOVERY_REQUIRED = 1,  //!< LWLink link recovery error oclwrred
    DCGM_GPU_LWLINK_ERROR_FATAL,         //!< LWLink link fatal error oclwrred
} dcgmGpuLWLinkErrorType_t;

/* Topology hints for dcgmSelectGpusByTopology() */
#define DCGM_TOPO_HINT_F_NONE         0x00000000 /* No hints specified */
#define DCGM_TOPO_HINT_F_IGNOREHEALTH 0x00000001 /* Ignore the health of the GPUs when picking GPUs for job 
                                                    exection. By default, only healthy GPUs are considered. */

typedef struct
{
    unsigned int version; //! < version of this message
    uint64_t     inputGpuIds; //! < bitmask of the GPU ids to choose from
    uint32_t     numGpus; //! < the number of gpus that DCGM should choose
    uint64_t     hintFlags; //! < Hints to ignore certain factors for the scheduling hint
} dcgmTopoSchedHint_v1;

typedef dcgmTopoSchedHint_v1 dcgmTopoSchedHint_t;

#define dcgmTopoSchedHint_version1 MAKE_DCGM_VERSION(dcgmTopoSchedHint_v1, 1)

/**
 * LwLink link states
 */
typedef enum dcgmLwLinkLinkState_enum
{
    DcgmLwLinkLinkStateNotSupported = 0, //!< LwLink is unsupported by this GPU (Default for GPUs)
    DcgmLwLinkLinkStateDisabled     = 1, //!< LwLink is supported for this link but this link is disabled (Default for LwSwitches)
    DcgmLwLinkLinkStateDown         = 2, //!< This LwLink link is down (inactive)
    DcgmLwLinkLinkStateUp           = 3  //!< This LwLink link is up (active)
} dcgmLwLinkLinkState_t;

/** 
 * State of LwLink links for a GPU 
 */
typedef struct
{
    dcgm_field_eid_t entityId; //!< Entity ID of the GPU (gpuId)
    dcgmLwLinkLinkState_t linkState[DCGM_LWLINK_MAX_LINKS_PER_GPU_LEGACY1]; //!< Per-GPU link states
} dcgmLwLinkGpuLinkStatus_v1;

typedef struct
{
    dcgm_field_eid_t entityId; //!< Entity ID of the GPU (gpuId)
    dcgmLwLinkLinkState_t linkState[DCGM_LWLINK_MAX_LINKS_PER_GPU]; //!< Per-GPU link states
} dcgmLwLinkGpuLinkStatus_v2;

/** 
 * State of LwLink links for a LwSwitch 
 */
typedef struct
{
    dcgm_field_eid_t entityId; //!< Entity ID of the LwSwitch (physicalId)
    dcgmLwLinkLinkState_t linkState[DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH]; //!< Per-LwSwitch link states
} dcgmLwLinkLwSwitchLinkStatus_t;

/**
 * Status of all of the LwLinks in a given system
 */
typedef struct
{
    unsigned int version;       //!< Version of this request. Should be dcgmLwLinkStatus_version1
    unsigned int numGpus;       //!< Number of entries in gpus[] that are populated
    dcgmLwLinkGpuLinkStatus_v1 gpus[DCGM_MAX_NUM_DEVICES]; //!< Per-GPU LwLink link statuses
    unsigned int numLwSwitches; //!< Number of entries in lwSwitches[] that are populated
    dcgmLwLinkLwSwitchLinkStatus_t lwSwitches[DCGM_MAX_NUM_SWITCHES]; //!< Per-LwSwitch link statuses
} dcgmLwLinkStatus_v1;

/**
 * Version 1 of dcgmLwLinkStatus
 */
#define dcgmLwLinkStatus_version1 MAKE_DCGM_VERSION(dcgmLwLinkStatus_v1, 1)

typedef struct
{
    unsigned int version;       //!< Version of this request. Should be dcgmLwLinkStatus_version1
    unsigned int numGpus;       //!< Number of entries in gpus[] that are populated
    dcgmLwLinkGpuLinkStatus_v2 gpus[DCGM_MAX_NUM_DEVICES]; //!< Per-GPU LwLink link statuses
    unsigned int numLwSwitches; //!< Number of entries in lwSwitches[] that are populated
    dcgmLwLinkLwSwitchLinkStatus_t lwSwitches[DCGM_MAX_NUM_SWITCHES]; //!< Per-LwSwitch link statuses
} dcgmLwLinkStatus_v2;

typedef dcgmLwLinkStatus_v2 dcgmLwLinkStatus_t;

/**
 * Version 2 of dcgmLwLinkStatus
 */
#define dcgmLwLinkStatus_version2 MAKE_DCGM_VERSION(dcgmLwLinkStatus_v2, 2)

/* Bitmask values for dcgmGetFieldIdSummary - Sync with DcgmcmSummaryType_t */
#define DCGM_SUMMARY_MIN          0x00000001
#define DCGM_SUMMARY_MAX          0x00000002
#define DCGM_SUMMARY_AVG          0x00000004
#define DCGM_SUMMARY_SUM          0x00000008
#define DCGM_SUMMARY_COUNT        0x00000010
#define DCGM_SUMMARY_INTEGRAL     0x00000020
#define DCGM_SUMMARY_DIFF         0x00000040
#define DCGM_SUMMARY_SIZE         7

/* dcgmSummaryResponse_t is part of dcgmFieldSummaryRequest, so it uses dcgmFieldSummaryRequest's version. */

typedef struct
{
    unsigned int fieldType;    //! < type of field that is summarized (int64 or fp64)
    unsigned int summaryCount; //! < the number of populated summaries in \ref values
    union
    {
        int64_t i64;
        double  fp64;
    } values[DCGM_SUMMARY_SIZE]; //! < array for storing the values of each summary. The summaries are stored
                                 //! < in order. For example, if MIN AND MAX are requested, then 0 will be MIN
                                 //! < and 1 will be MAX. If AVG and DIFF were requested, then AVG would be 0
                                 //! < and 1 would be DIFF
} dcgmSummaryResponse_t;

typedef struct
{
    unsigned int              version;         //! < version of this message - dcgmFieldSummaryRequest_v1
    unsigned short            fieldId;         //! < field id to be summarized
    dcgm_field_entity_group_t entityGroupId;    //! < the type of entity whose field we're getting
    dcgm_field_eid_t          entityId;        //! < ordinal id for this entity
    uint32_t                  summaryTypeMask; //! < bitmask of DCGM_SUMMARY_*, the requested summaries
    uint64_t                  startTime;       //! < start time for the interval being summarized. 0 means to use
                                               //! < any data before.
    uint64_t                  endTime;         //! < end time for the interval being summarized. 0 means to use
                                               //! < any data after.
    dcgmSummaryResponse_t     response;        //! < response data for this request
} dcgmFieldSummaryRequest_v1;

typedef dcgmFieldSummaryRequest_v1 dcgmFieldSummaryRequest_t;

#define dcgmFieldSummaryRequest_version1 MAKE_DCGM_VERSION(dcgmFieldSummaryRequest_v1, 1)

/**
 * Module IDs
 */
typedef enum
{
    DcgmModuleIdCore           = 0, //!< Core DCGM - always loaded
    DcgmModuleIdLwSwitch       = 1, //!< LwSwitch Module
    DcgmModuleIdVGPU           = 2, //!< VGPU Module
    DcgmModuleIdIntrospect     = 3, //!< Introspection Module
    DcgmModuleIdHealth         = 4, //!< Health Module
    DcgmModuleIdPolicy         = 5, //!< Policy Module
    DcgmModuleIdConfig         = 6, //!< Config Module
    DcgmModuleIdDiag           = 7, //!< GPU Diagnostic Module
    DcgmModuleIdProfiling      = 8, //!< Profiling Module
    
    DcgmModuleIdCount               //!< Always last. 1 greater than largest value above
} dcgmModuleId_t;

/**
 * Module Status. Modules are lazy loaded, so they will be in status DcgmModuleStatusNotLoaded
 * until they are used. One modules are used, they will move to another status. 
 */
typedef enum
{
    DcgmModuleStatusNotLoaded = 0,   //!< Module has not been loaded yet
    DcgmModuleStatusBlacklisted = 1, //!< Module has been blacklisted from being loaded
    DcgmModuleStatusFailed = 2,      //!< Loading the module failed
    DcgmModuleStatusLoaded = 3,      //!< Module has been loaded
} dcgmModuleStatus_t;

/**
 * Status of all of the modules of the host engine
 */
typedef struct
{
    dcgmModuleId_t     id;     //!< ID of this module
    dcgmModuleStatus_t status; //!< Status of this module
} dcgmModuleGetStatusesModule_t;

#define DCGM_MODULE_STATUSES_CAPACITY 16 //!< This is larger than DcgmModuleIdCount so we can add modules without versioning this request

typedef struct
{
    unsigned int version;       //!< Version of this request. Should be dcgmModuleGetStatuses_version1
    unsigned int numStatuses;   //!< Number of entries in statuses[] that are populated
    dcgmModuleGetStatusesModule_t statuses[DCGM_MODULE_STATUSES_CAPACITY]; //!< Per-module status information
} dcgmModuleGetStatuses_v1;

/**
 * Version 1 of dcgmModuleGetStatuses
 */
#define dcgmModuleGetStatuses_version1 MAKE_DCGM_VERSION(dcgmModuleGetStatuses_v1, 1)
#define dcgmModuleGetStatuses_version dcgmModuleGetStatuses_version1
typedef dcgmModuleGetStatuses_v1 dcgmModuleGetStatuses_t;


/**
 * Structure to return all of the profiling metric groups that are available for the given groupId.
 * 
 */

#define DCGM_PROF_MAX_NUM_GROUPS  10  //!< Maximum number of metric ID groups that can exist in DCGM
#define DCGM_PROF_MAX_FIELD_IDS_PER_GROUP 8 //!< Maximum number of field IDs that can be in a single DCGM profiling metric group

typedef struct
{
    unsigned short majorId;  /** Major ID of this metric group. Metric groups with the same majorId cannot be
                                 watched conlwrrently with other metric groups with the same majorId */
    unsigned short minorId;  /** Minor ID of this metric group. This distinguishes metric groups within the same
                                 major metric group from each other */
    unsigned int numFieldIds;/** Number of field IDs that are populated in fieldIds[] */
    unsigned short fieldIds[DCGM_PROF_MAX_FIELD_IDS_PER_GROUP]; /** DCGM Field IDs that are part of this profiling group. 
                                 See DCGM_FI_PROF_* definitions in dcgm_fields.h for details. */
} dcgmProfMetricGroupInfo_t;

typedef struct
{
    /* Input parameters */
    unsigned int version;         /** Version of this request. Should be dcgmProfGetMetricGroups_version */
    unsigned int unused;          /** Not used for now. Set to 0 */
    dcgmGpuGrp_t groupId;         /** Group of GPUs we should get the metric groups for. These must all be the
                                      exact same GPU or DCGM_ST_GROUP_INCOMPATIBLE will be returned */
    /* Output */
    unsigned int numMetricGroups; /** Number of entries in metricGroups[] that are populated */
    unsigned int unused1;         /** Not used for now. Set to 0 */
    dcgmProfMetricGroupInfo_t metricGroups[DCGM_PROF_MAX_NUM_GROUPS]; /* Info for each metric group */
} dcgmProfGetMetricGroups_v2;

/**
 * Version 1 of dcgmProfGetMetricGroups_t
 */
#define dcgmProfGetMetricGroups_version2 MAKE_DCGM_VERSION(dcgmProfGetMetricGroups_v2, 2)
#define dcgmProfGetMetricGroups_version dcgmProfGetMetricGroups_version2
typedef dcgmProfGetMetricGroups_v2 dcgmProfGetMetricGroups_t;

/**
 * Structure to pass to dcgmProfWatchFields() when watching profiling metrics
 */
typedef struct
{
    unsigned int version; /** Version of this request. Should be dcgmProfWatchFields_version */
    dcgmGpuGrp_t groupId; /** Group ID representing collection of one or more GPUs. Look at
                              \ref dcgmGroupCreate for details on creating the group.
                              Alternatively, pass in the group id as \a DCGM_GROUP_ALL_GPUS
                              to perform operation on all the GPUs. The GPUs of the group must all be
                              identical or DCGM_ST_GROUP_INCOMPATIBLE will be returned by this API. */
    unsigned int numFieldIds; /** Number of field IDs that are being passed in fieldIds[] */
    unsigned short fieldIds[16]; /** DCGM_FI_PROF_? field IDs to watch */
    long long updateFreq; /** How often to update this field in usec. Note that profiling metrics may need to be
                              sampled more frequently than this value. See dcgmProfMetricGroupInfo_t.minUpdateFreqUsec
                              of the metric group matching metricGroupTag to see what this minimum is. If 
                              minUpdateFreqUsec < updateFreq then samples will be aggregated to updateFreq intervals
                              in DCGM's internal cache. */
    double maxKeepAge;    /** How long to keep data for every fieldId in seconds */
    int maxKeepSamples;   /** Maximum number of samples to keep for each fieldId. 0=no limit */
    unsigned int flags;   /** For future use. Set to 0 for now. */
} dcgmProfWatchFields_v1;

/**
 * Version 1 of dcgmProfWatchFields_v1
 */
#define dcgmProfWatchFields_version1 MAKE_DCGM_VERSION(dcgmProfWatchFields_v1, 1)
#define dcgmProfWatchFields_version dcgmProfWatchFields_version1
typedef dcgmProfWatchFields_v1 dcgmProfWatchFields_t;

/**
 * Structure to pass to dcgmProfUnwatchFields when unwatching profiling metrics
 */
typedef struct
{
    unsigned int version; /** Version of this request. Should be dcgmProfUnwatchFields_version */
    dcgmGpuGrp_t groupId; /** Group ID representing collection of one or more GPUs. Look at
                              \ref dcgmGroupCreate for details on creating the group.
                              Alternatively, pass in the group id as \a DCGM_GROUP_ALL_GPUS
                              to perform operation on all the GPUs. The GPUs of the group must all be
                              identical or DCGM_ST_GROUP_INCOMPATIBLE will be returned by this API. */
    unsigned int flags;   /** For future use. Set to 0 for now. */
} dcgmProfUnwatchFields_v1;

/**
 * Version 1 of dcgmProfUnwatchFields_v1
 */
#define dcgmProfUnwatchFields_version1 MAKE_DCGM_VERSION(dcgmProfUnwatchFields_v1, 1)
#define dcgmProfUnwatchFields_version dcgmProfUnwatchFields_version1
typedef dcgmProfUnwatchFields_v1 dcgmProfUnwatchFields_t;

/**
 * Structure to describe the DCGM build environment
 */
typedef struct
{
    unsigned int version;                    //!< Version of this message
    char changelist[DCGM_MAX_STR_LENGTH];    //!< Changelist number from which DCGM was built
    char platform[DCGM_MAX_STR_LENGTH];      //!< Builder platform - uname result without hostname
    char branch[DCGM_MAX_STR_LENGTH];        //!< Name of the branch where DCGM was built
    char driverVersion[DCGM_MAX_STR_LENGTH]; //!< The version of LWpu driver DCGM was linked with
    char buildDate[DCGM_MAX_STR_LENGTH];     //!< Date of the build
} dcgmVersionInfo_v1;

/**
 * Version 1 of dcgmVersionInfo_v1;
 */
#define dcgmVersionInfo_version1 MAKE_DCGM_VERSION(dcgmVersionInfo_v1, 1)
#define dcgmVersionInfo_version dcgmVersionInfo_version1
typedef dcgmVersionInfo_v1 dcgmVersionInfo_t;

/** @} */

#ifdef    __cplusplus
}
#endif

#endif    /* DCGM_STRUCTS_H */
