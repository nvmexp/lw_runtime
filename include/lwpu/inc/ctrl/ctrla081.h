/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrla081.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrla082.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"
#include "lw_vgpu_types.h"
/* LWA081_VGPU_CONFIG control commands and parameters */

#define LWA081_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0xA081, LWA081_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWA081_CTRL_RESERVED                 (0x00)
#define LWA081_CTRL_VGPU_CONFIG              (0x01)

#define LWA081_CTRL_VGPU_CONFIG_ILWALID_TYPE 0x00
#define LWA081_MAX_VGPU_TYPES_PER_PGPU       0x20
#define LWA081_MAX_VGPU_PER_PGPU             32
#define LWA081_VM_UUID_SIZE                  16
#define LWA081_VGPU_STRING_BUFFER_SIZE       32
#define LWA081_VGPU_SIGNATURE_SIZE           128
#define LWA081_VM_NAME_SIZE                  128
#define LWA081_PCI_CONFIG_SPACE_SIZE         0x100
#define LWA081_PGPU_METADATA_STRING_SIZE     256
#define LWA081_EXTRA_PARAMETERS_SIZE         1024

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_SET_INFO
 *
 * This command sets the vGPU config information in RM
 *
 * Parameters:
 *
 * discardVgpuTypes [IN]
 *  This parameter specifies if existing vGPU configuration should be
 *  discarded for given pGPU
 *
 * vgpuInfo [IN]
 *  This parameter specifies virtual GPU type information
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_SET_INFO (0xa0810101) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_INFO_PARAMS_MESSAGE_ID" */

/*
 * LWA081_CTRL_VGPU_CONFIG_INFO
 *
 * This structure represents the per vGPU information
 *
 */
typedef struct LWA081_CTRL_VGPU_INFO {
    // This structure should be in sync with LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO_PARAMS
    LwU32 vgpuType;
    LwU8  vgpuName[LWA081_VGPU_STRING_BUFFER_SIZE];
    LwU8  vgpuClass[LWA081_VGPU_STRING_BUFFER_SIZE];
    LwU8  vgpuSignature[LWA081_VGPU_SIGNATURE_SIZE];
    LwU8  license[LW_GRID_LICENSE_INFO_MAX_LENGTH];
    LwU32 maxInstance;
    LwU32 numHeads;
    LwU32 maxResolutionX;
    LwU32 maxResolutionY;
    LwU32 maxPixels;
    LwU32 frlConfig;
    LwU32 lwdaEnabled;
    LwU32 eccSupported;
    LwU32 gpuInstanceSize;
    LwU32 multiVgpuSupported;
    LW_DECLARE_ALIGNED(LwU64 vdevId, 8);
    LW_DECLARE_ALIGNED(LwU64 pdevId, 8);
    LW_DECLARE_ALIGNED(LwU64 fbLength, 8);
    LW_DECLARE_ALIGNED(LwU64 mappableVideoSize, 8);
    LW_DECLARE_ALIGNED(LwU64 fbReservation, 8);
    LwU32 encoderCapacity;
    LW_DECLARE_ALIGNED(LwU64 bar1Length, 8);
    LwU32 frlEnable;
    LwU8  adapterName[LW2080_GPU_MAX_NAME_STRING_LENGTH];
    LwU16 adapterName_Unicode[LW2080_GPU_MAX_NAME_STRING_LENGTH];
    LwU8  shortGpuNameString[LW2080_GPU_MAX_NAME_STRING_LENGTH];
    LwU8  licensedProductName[LW_GRID_LICENSE_INFO_MAX_LENGTH];
    LwU32 vgpuExtraParams[LWA081_EXTRA_PARAMETERS_SIZE];
    LwU32 ftraceEnable;
    LwU32 gpuDirectSupported;
    LwU32 lwlinkP2PSupported;
    // used only by LWML
    LwU32 gpuInstanceProfileId;
} LWA081_CTRL_VGPU_INFO;


/*
 * LWA081_CTRL_VGPU_CONFIG_INFO_PARAMS
 *
 * This structure represents the vGPU configuration information
 *
 */
#define LWA081_CTRL_VGPU_CONFIG_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA081_CTRL_VGPU_CONFIG_INFO_PARAMS {
    LwBool discardVgpuTypes;
    LW_DECLARE_ALIGNED(LWA081_CTRL_VGPU_INFO vgpuInfo, 8);
} LWA081_CTRL_VGPU_CONFIG_INFO_PARAMS;


/*
 * LWA081_CTRL_VGPU_CONFIG_ENUMERATE_VGPU_PER_PGPU
 *
 * This command enumerates list of vGPU guest instances per pGpu
 *
 * Parameters:
 *
 * vgpuType [OUT]
 *  This parameter specifies the virtual GPU type for this physical GPU
 *
 * numVgpu [OUT]
 *  This parameter specifies the number of virtual GPUs created on this physical GPU
 *
 * guestInstanceInfo [OUT]
 *  This parameter specifies an array containing guest instance's information for
 *  all instances created on this physical GPU
 *
 * guestVgpuInfo [OUT]
 *  This parameter specifies an array containing guest vgpu's information for
 *  all vGPUs created on this physical GPU
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_ENUMERATE_VGPU_PER_PGPU (0xa0810102) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_ENUMERATE_VGPU_PER_PGPU_PARAMS_MESSAGE_ID" */

/*
 * LWA081_GUEST_VM_INFO
 *
 * This structure represents vGPU guest's (VM's) information
 *
 * vmPid [OUT]
 *  This param specifies the vGPU plugin process ID
 * vmIdType [OUT]
 *  This param specifies the VM ID type, i.e. DOMAIN_ID or UUID
 * guestOs [OUT]
 *  This param specifies the vGPU guest OS type
 * migrationProhibited [OUT]
 *  This flag indicates whether migration is prohibited for VM or not
 * guestNegotiatedVgpuVersion [OUT]
 *  This param specifies the vGPU version of guest driver after negotiation
 * frameRateLimit [OUT]
 *  This param specifies the current value of FRL set for guest
 * licensed [OUT]
 *  This param specifies whether the VM is Unlicensed/Licensed
 * licenseState [OUT]
 *  This param specifies the current state of the GRID license state machine
 * licenseExpiryTimestamp [OUT]
 *  License expiry time in seconds since UNIX epoch
 * licenseExpiryStatus [OUT]
 *  License expiry status
 * guestDriverVersion [OUT]
 *  This param specifies the driver version of the driver installed on the VM
 * guestDriverBranch [OUT]
 *  This param specifies the driver branch of the driver installed on the VM
 * vmName [OUT]
 *  This param stores the name assigned to VM (KVM only)
 * guestVmInfoState [OUT]
 *  This param stores the current state of guest dependent fields
 *
 */
typedef struct LWA081_GUEST_VM_INFO {
    LwU32               vmPid;
    VM_ID_TYPE          vmIdType;
    LwU32               guestOs;
    LwU32               migrationProhibited;
    LwU32               guestNegotiatedVgpuVersion;
    LwU32               frameRateLimit;
    LwBool              licensed;
    LwU32               licenseState;
    LwU32               licenseExpiryTimestamp;
    LwU8                licenseExpiryStatus;
    LW_DECLARE_ALIGNED(VM_ID guestVmId, 8);
    LwU8                guestDriverVersion[LWA081_VGPU_STRING_BUFFER_SIZE];
    LwU8                guestDriverBranch[LWA081_VGPU_STRING_BUFFER_SIZE];
    LwU8                vmName[LWA081_VM_NAME_SIZE];
    GUEST_VM_INFO_STATE guestVmInfoState;
} LWA081_GUEST_VM_INFO;

/*
 * LWA081_GUEST_VGPU_DEVICE
 *
 * This structure represents host vgpu device's (assigned to VM) information
 *
 * eccState [OUT]
 *  This parameter specifies the ECC state of the virtual GPU.
 *  One of LWA081_CTRL_ECC_STATE_xxx values.
 * bDriverLoaded [OUT]
 *  This parameter specifies whether driver is loaded on this particular vGPU.
 * swizzId [OUT]
 *  This param specifies the GPU Instance ID or Swizz ID
 *
 */
typedef struct LWA081_HOST_VGPU_DEVICE {
    LwU32  vgpuType;
    LwU32  vgpuDeviceInstanceId;
    LW_DECLARE_ALIGNED(LwU64 vgpuPciId, 8);
    LwU8   vgpuUuid[VM_UUID_SIZE];
    LwU8   mdevUuid[VM_UUID_SIZE];
    LwU32  encoderCapacity;
    LW_DECLARE_ALIGNED(LwU64 fbUsed, 8);
    LwU32  eccState;
    LwBool bDriverLoaded;
    LwU32  swizzId;
} LWA081_HOST_VGPU_DEVICE;

/* ECC state values */
#define LWA081_CTRL_ECC_STATE_UNKNOWN       0
#define LWA081_CTRL_ECC_STATE_NOT_SUPPORTED 1
#define LWA081_CTRL_ECC_STATE_DISABLED      2
#define LWA081_CTRL_ECC_STATE_ENABLED       3

/*
 * LWA081_VGPU_GUEST
 *
 * This structure represents a vGPU guest
 *
 */
typedef struct LWA081_VGPU_GUEST {
    LW_DECLARE_ALIGNED(LWA081_GUEST_VM_INFO guestVmInfo, 8);
    LW_DECLARE_ALIGNED(LWA081_HOST_VGPU_DEVICE vgpuDevice, 8);
} LWA081_VGPU_GUEST;

/*
 * LWA081_CTRL_VGPU_CONFIG_ENUMERATE_VGPU_PER_PGPU_PARAMS
 *
 * This structure represents the information of vGPU guest instances per pGpu
 *
 */
#define LWA081_CTRL_VGPU_CONFIG_ENUMERATE_VGPU_PER_PGPU_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWA081_CTRL_VGPU_CONFIG_ENUMERATE_VGPU_PER_PGPU_PARAMS {
    LwU32 vgpuType;
    LwU32 numVgpu;
    LW_DECLARE_ALIGNED(LWA081_VGPU_GUEST vgpuGuest[LWA081_MAX_VGPU_PER_PGPU], 8);
} LWA081_CTRL_VGPU_CONFIG_ENUMERATE_VGPU_PER_PGPU_PARAMS;

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_VGPU_TYPE_INFO
 *
 * This command fetches vGPU type info from RM.
 *
 * Parameters:
 *
 * vgpuType [IN]
 *  This parameter specifies the virtual GPU type for which vGPU info should be returned.
 *
 * vgpuTypeInfo [OUT]
 *  This parameter returns LWA081_CTRL_VGPU_INFO data for the vGPU type specified by vgpuType.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */

#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_VGPU_TYPE_INFO (0xa0810103) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPE_INFO_PARAMS_MESSAGE_ID" */

/*
 * LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPE_INFO_PARAMS
 *
 */
#define LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPE_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPE_INFO_PARAMS {
    LwU32 vgpuType;
    LW_DECLARE_ALIGNED(LWA081_CTRL_VGPU_INFO vgpuTypeInfo, 8);
} LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPE_INFO_PARAMS;

/*
 * LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPES_PARAMS
 *      This structure represents supported/creatable vGPU types on a pGPU
 */
typedef struct LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPES_PARAMS {
    /*
     * [OUT] Count of supported/creatable vGPU types on a pGPU
     */
    LwU32 numVgpuTypes;

    /*
     * [OUT] - Array of vGPU type ids supported/creatable on a pGPU
     */
    LwU32 vgpuTypes[LWA081_MAX_VGPU_TYPES_PER_PGPU];
} LWA081_CTRL_VGPU_CONFIG_GET_VGPU_TYPES_PARAMS;


/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_SUPPORTED_VGPU_TYPES
 *
 * This command fetches count and list of vGPU types supported on a pGpu from RM
 *
 * Parameters:
 *
 * numVgpuTypes [OUT]
 *  This parameter returns the number of vGPU types supported on this pGPU
 *
 * vgpuTypes [OUT]
 *  This parameter returns list of supported vGPUs types on this pGPU
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_NOT_SUPPORTED
 */

#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_SUPPORTED_VGPU_TYPES (0xa0810104) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | 0x4" */

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_CREATABLE_VGPU_TYPES
 *
 * This command fetches count and list of vGPU types creatable on a pGpu from RM
 *
 * Parameters:
 *
 * numVgpuTypes [OUT]
 *  This parameter returns the number of vGPU types creatable on this pGPU
 *
 * vgpuTypes [OUT]
 *  This parameter returns list of creatable vGPUs types on this pGPU
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_NOT_SUPPORTED
 */

#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_CREATABLE_VGPU_TYPES (0xa0810105) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | 0x5" */

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the associated subdevice/pGPU.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the associated subdevice object.
 *
 *   event
 *     This parameter specifies the type of event to which the specified
 *     action is to be applied.  This parameter must specify a valid
 *     LWA081_NOTIFIERS value (see cla081.h for more details) and should
 *     not exceed one less LWA081_NOTIFIERS_MAXCOUNT.
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LWA081_CTRL_SET_EVENT_NOTIFICATION_DISABLE
 *         This action disables event notification for the specified
 *         event for the associated subdevice object.
 *       LWA081_CTRL_SET_EVENT_NOTIFICATION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event for the associated subdevice object.
 *       LWA081_CTRL_SET_EVENT_NOTIFICATION_REPEAT
 *         This action enables repeated event notification for the specified
 *         event for the associated system controller object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_EVENT_SET_NOTIFICATION   (0xa0810106) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LWA081_CTRL_VGPU_CONFIG_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWA081_CTRL_VGPU_CONFIG_EVENT_SET_NOTIFICATION_PARAMS {
    LwU32 event;
    LwU32 action;
} LWA081_CTRL_VGPU_CONFIG_EVENT_SET_NOTIFICATION_PARAMS;


/* valid event action values */
#define LWA081_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LWA081_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LWA081_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_NOTIFY_START
 *
 * This command notifies the lwpu-vgpu-vfio module with start status.
 * It notifies whether start has been successful or not.
 *
 *   mdevUuid
 *     This parameter specifies the uuid of the mdev device for which start has
 *     been called.
 *   vmUuid
 *     The UUID of VM for which vGPU has been created.
 *   vmName
 *     The name of VM for which vGPU has been created.
 *   returnStatus
 *     This parameter species whether the vGPU plugin is initialized or not.
 *     it specifies the error code in case plugin initialization has failed
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_NOTIFY_START          (0xa0810107) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_NOTIFY_START_PARAMS_MESSAGE_ID" */

/*
 * LWA081_CTRL_VGPU_CONFIG_NOTIFY_START_PARAMS
 * This structure represents information of plugin init status.
 */
#define LWA081_CTRL_VGPU_CONFIG_NOTIFY_START_PARAMS_MESSAGE_ID (0x7U)

typedef struct LWA081_CTRL_VGPU_CONFIG_NOTIFY_START_PARAMS {
    LwU8  mdevUuid[VM_UUID_SIZE];
    LwU8  vmUuid[VM_UUID_SIZE];
    LwU8  vmName[LWA081_VM_NAME_SIZE];
    LwU32 returnStatus;
} LWA081_CTRL_VGPU_CONFIG_NOTIFY_START_PARAMS;

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_MDEV_REGISTER
 *
 * This command register the GPU to Linux kernel's mdev module for vGPU on KVM.
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_MDEV_REGISTER                      (0xa0810109) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | 0x9" */

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_SET_VGPU_INSTANCE_ENCODER_CAPACITY
 *
 * This command is used to set vGPU instance's (represented by vgpuUuid) encoder Capacity.
 *
 *   vgpuUuid
 *     This parameter specifies the uuid of vGPU assigned to VM.
 *   encoderCapacity
 *     Encoder capacity value from 0 to 100. Value of 0x00 indicates encoder performance
 *     may be minimal for this GPU and software should fall back to CPU-based encode.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LWA081_CTRL_CMD_VGPU_CONFIG_SET_VGPU_INSTANCE_ENCODER_CAPACITY (0xa0810110) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_VGPU_INSTANCE_ENCODER_CAPACITY_PARAMS_MESSAGE_ID" */

/*
 * LWA081_CTRL_VGPU_CONFIG_VGPU_INSTANCE_ENCODER_CAPACITY_PARAMS
 *
 * This structure represents  encoder capacity for vgpu instance.
 */
#define LWA081_CTRL_VGPU_CONFIG_VGPU_INSTANCE_ENCODER_CAPACITY_PARAMS_MESSAGE_ID (0x10U)

typedef struct LWA081_CTRL_VGPU_CONFIG_VGPU_INSTANCE_ENCODER_CAPACITY_PARAMS {
    LwU8  vgpuUuid[VM_UUID_SIZE];
    LwU32 encoderCapacity;
} LWA081_CTRL_VGPU_CONFIG_VGPU_INSTANCE_ENCODER_CAPACITY_PARAMS;

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_VGPU_FB_USAGE
 *
 * This command is used to get the FB usage of all vGPU instances running on a GPU.
 *
 *  vgpuCount
 *      This field specifies the number of vGPU devices for which FB usage is returned.
 *  vgpuFbUsage
 *      This is an array of type LWA081_VGPU_FB_USAGE, which contains a list of vGPUs
 *      and their corresponding FB usage in bytes;
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_VGPU_FB_USAGE (0xa0810111) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_GET_VGPU_FB_USAGE_PARAMS_MESSAGE_ID" */

typedef struct LWA081_VGPU_FB_USAGE {
    LwU8 vgpuUuid[VM_UUID_SIZE];
    LW_DECLARE_ALIGNED(LwU64 fbUsed, 8);
} LWA081_VGPU_FB_USAGE;

/*
 * LWA081_CTRL_VGPU_CONFIG_GET_VGPU_FB_USAGE_PARAMS
 *
 * This structure represents the FB usage information of vGPU instances running on a GPU.
 */
#define LWA081_CTRL_VGPU_CONFIG_GET_VGPU_FB_USAGE_PARAMS_MESSAGE_ID (0x11U)

typedef struct LWA081_CTRL_VGPU_CONFIG_GET_VGPU_FB_USAGE_PARAMS {
    LwU32 vgpuCount;
    LW_DECLARE_ALIGNED(LWA081_VGPU_FB_USAGE vgpuFbUsage[LWA081_MAX_VGPU_PER_PGPU], 8);
} LWA081_CTRL_VGPU_CONFIG_GET_VGPU_FB_USAGE_PARAMS;

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_MIGRATION_CAP
 *
 * This command is used to query whether pGPU is live migration capable or not.
 *
 *  bMigrationCap
 *      Set to LW_TRUE if pGPU is migration capable.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_REQUEST
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_MIGRATION_CAP (0xa0810112) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_CMD_VGPU_CONFIG_GET_MIGRATION_CAP_PARAMS_MESSAGE_ID" */

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_MIGRATION_CAP_PARAMS
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_MIGRATION_CAP_PARAMS_MESSAGE_ID (0x12U)

typedef struct LWA081_CTRL_CMD_VGPU_CONFIG_GET_MIGRATION_CAP_PARAMS {
    LwBool bMigrationCap;
} LWA081_CTRL_CMD_VGPU_CONFIG_GET_MIGRATION_CAP_PARAMS;

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_HOST_FB_RESERVATION
 *
 * This command is used to get the host FB requirements
 *
 *  hostReservedFb [OUT]
 *      Amount of FB reserved for the host
 *  eccAndPrReservedFb [OUT]
 *      Amount of FB reserved for the ecc and page retirement
 *  totalReservedFb [OUT]
 *      Total FB reservation
 *  vgpuTypeId [IN]
 *      The Type ID for VGPU profile
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_HOST_FB_RESERVATION (0xa0810113) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_GET_HOST_FB_RESERVATION_PARAMS_MESSAGE_ID" */

#define LWA081_CTRL_VGPU_CONFIG_GET_HOST_FB_RESERVATION_PARAMS_MESSAGE_ID (0x13U)

typedef struct LWA081_CTRL_VGPU_CONFIG_GET_HOST_FB_RESERVATION_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 hostReservedFb, 8);
    LW_DECLARE_ALIGNED(LwU64 eccAndPrReservedFb, 8);
    LW_DECLARE_ALIGNED(LwU64 totalReservedFb, 8);
    LwU32 vgpuTypeId;
} LWA081_CTRL_VGPU_CONFIG_GET_HOST_FB_RESERVATION_PARAMS;

/*
 * LWA081_CTRL_CMD_VGPU_CONFIG_GET_PGPU_METADATA_STRING
 *
 * This command is used to get the pGpu metadata string.
 *
 * pGpuString
 *     String holding pGpu Metadata
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_REQUEST
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_PGPU_METADATA_STRING (0xa0810114) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_GET_PGPU_METADATA_STRING_PARAMS_MESSAGE_ID" */

#define LWA081_CTRL_VGPU_CONFIG_GET_PGPU_METADATA_STRING_PARAMS_MESSAGE_ID (0x14U)

typedef struct LWA081_CTRL_VGPU_CONFIG_GET_PGPU_METADATA_STRING_PARAMS {
    LwU8 pGpuString[LWA081_PGPU_METADATA_STRING_SIZE];
} LWA081_CTRL_VGPU_CONFIG_GET_PGPU_METADATA_STRING_PARAMS;

#define LWA081_CTRL_CMD_VGPU_CONFIG_GET_DOORBELL_EMULATION_SUPPORT (0xa0810115) /* finn: Evaluated from "(FINN_LWA081_VGPU_CONFIG_VGPU_CONFIG_INTERFACE_ID << 8) | LWA081_CTRL_VGPU_CONFIG_GET_DOORBELL_EMULATION_SUPPORT_PARAMS_MESSAGE_ID" */

#define LWA081_CTRL_VGPU_CONFIG_GET_DOORBELL_EMULATION_SUPPORT_PARAMS_MESSAGE_ID (0x15U)

typedef struct LWA081_CTRL_VGPU_CONFIG_GET_DOORBELL_EMULATION_SUPPORT_PARAMS {
    LwBool doorbellEmulationEnabled;
} LWA081_CTRL_VGPU_CONFIG_GET_DOORBELL_EMULATION_SUPPORT_PARAMS;

/* _ctrlA081vgpuconfig_h_ */
