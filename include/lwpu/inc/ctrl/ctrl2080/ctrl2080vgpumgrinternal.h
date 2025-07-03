/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl2080/ctrl2080vgpumgrinternal.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrla081.h"

/*
 * LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK
 *
 * This command is used to bootload GSP VGPU plugin task.
 * Can be called only with SR-IOV and with VGPU_GSP_PLUGIN_OFFLOAD feature.
 *
 * dbdf                        - domain (31:16), bus (15:8), device (7:3), function (2:0)
 * gfid                        - Gfid
 * vgpuType                    - The Type ID for VGPU profile
 * vmPid                       - Plugin process ID of vGPU guest instance
 * swizzId                     - SwizzId
 * numChannels                 - Number of channels
 * numPluginChannels           - Number of plugin channels
 * bDisableSmcPartitionRestore - If set to true, SMC default exelwtion partition
 *                               save/restore will not be done in host-RM
 * guestFbPhysAddrList         - list of VMMU segment aligned physical address of guest FB memory
 * guestFbLengthList           - list of guest FB memory length in bytes
 * pluginHeapMemoryPhysAddr    - plugin heap memory offset
 * pluginHeapMemoryLength      - plugin heap memory length in bytes
 */
#define LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK (0x20804001) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_MAX_VMMU_SEGMENTS                                   384

#define LW2080_CTRL_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK_PARAMS {
    LwU32  dbdf;
    LwU32  gfid;
    LwU32  vgpuType;
    LwU32  vmPid;
    LwU32  swizzId;
    LwU32  numChannels;
    LwU32  numPluginChannels;
    LwBool bDisableDefaultSmcExecPartRestore;
    LwU32  numGuestFbSegments;
    LW_DECLARE_ALIGNED(LwU64 guestFbPhysAddrList[LW2080_CTRL_MAX_VMMU_SEGMENTS], 8);
    LW_DECLARE_ALIGNED(LwU64 guestFbLengthList[LW2080_CTRL_MAX_VMMU_SEGMENTS], 8);
    LW_DECLARE_ALIGNED(LwU64 pluginHeapMemoryPhysAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 pluginHeapMemoryLength, 8);
    LW_DECLARE_ALIGNED(LwU64 ctrlBuffOffset, 8);
} LW2080_CTRL_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK_PARAMS;

/*
 * LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK
 *
 * This command is used to shutdown GSP VGPU plugin task.
 * Can be called only with SR-IOV and with VGPU_GSP_PLUGIN_OFFLOAD feature.
 *
 * gfid                        - Gfid
 */
#define LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK (0x20804002) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK_PARAMS {
    LwU32 gfid;
} LW2080_CTRL_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK_PARAMS;

/*
 * LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE
 *
 * This command is used to add a new vGPU config to the pGPU in physical RM.
 * Unlike LWA081_CTRL_CMD_VGPU_CONFIG_SET_INFO, it does no validation
 * and is only to be used internally.
 *
 * discardVgpuTypes [IN]
 *  This parameter specifies if existing vGPU configuration should be
 *  discarded for given pGPU
 *
 * vgpuInfo [IN]
 *   This parameter specifies virtual GPU type information
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE (0x20804003) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE_PARAMS {
    LwBool discardVgpuTypes;
    LW_DECLARE_ALIGNED(LWA081_CTRL_VGPU_INFO vgpuInfo, 8);
} LW2080_CTRL_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE_PARAMS;

/*
 * LW2080_GUEST_VM_INFO
 *
 * This structure represents vGPU guest's (VM's) information
 *
 * vmPid [OUT]
 *  This param specifies the vGPU plugin process ID
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
 * guestVmInfoState [OUT]
 *  This param stores the current state of guest dependent fields
 *
 */
typedef struct LW2080_GUEST_VM_INFO {
    LwU32               vmPid;
    LwU32               guestOs;
    LwU32               migrationProhibited;
    LwU32               guestNegotiatedVgpuVersion;
    LwU32               frameRateLimit;
    LwBool              licensed;
    LwU32               licenseState;
    LwU32               licenseExpiryTimestamp;
    LwU8                licenseExpiryStatus;
    LwU8                guestDriverVersion[LWA081_VGPU_STRING_BUFFER_SIZE];
    LwU8                guestDriverBranch[LWA081_VGPU_STRING_BUFFER_SIZE];
    GUEST_VM_INFO_STATE guestVmInfoState;
} LW2080_GUEST_VM_INFO;

/*
 * LW2080_GUEST_VGPU_DEVICE
 *
 * This structure represents host vgpu device's (assigned to VM) information
 *
 * gfid [OUT]
 *  This parameter specifies the gfid of vGPU assigned to VM.
 * vgpuPciId [OUT]
 *  This parameter specifies vGPU PCI ID
 * vgpuDeviceInstanceId [OUT]
 *  This paramter specifies the vGPU device instance per VM to be used for supporting
 *  multiple vGPUs per VM.
 * fbUsed [OUT]
 *  This parameter specifies FB usage in bytes
 * eccState [OUT]
 *  This parameter specifies the ECC state of the virtual GPU.
 *  One of LWA081_CTRL_ECC_STATE_xxx values.
 * bDriverLoaded [OUT]
 *  This parameter specifies whether driver is loaded on this particular vGPU.
 *
 */
typedef struct LW2080_HOST_VGPU_DEVICE {
    LwU32  gfid;
    LW_DECLARE_ALIGNED(LwU64 vgpuPciId, 8);
    LwU32  vgpuDeviceInstanceId;
    LW_DECLARE_ALIGNED(LwU64 fbUsed, 8);
    LwU32  encoderCapacity;
    LwU32  eccState;
    LwBool bDriverLoaded;
} LW2080_HOST_VGPU_DEVICE;

/*
 * LW2080_VGPU_GUEST
 *
 * This structure represents a vGPU guest
 *
 */
typedef struct LW2080_VGPU_GUEST {
    LW2080_GUEST_VM_INFO guestVmInfo;
    LW_DECLARE_ALIGNED(LW2080_HOST_VGPU_DEVICE vgpuDevice, 8);
} LW2080_VGPU_GUEST;

/*
 * LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_ENUMERATE_VGPU_PER_PGPU
 *
 * This command enumerates list of vGPU guest instances per pGpu
 *
 * numVgpu [OUT]
 *  This parameter specifies the number of virtual GPUs created on this physical GPU
 *
 * vgpuGuest [OUT]
 *  This parameter specifies an array containing guest vgpu's information for
 *  all vGPUs created on this physical GPU
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_ENUMERATE_VGPU_PER_PGPU (0x20804004) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_VGPU_MGR_INTERNAL_ENUMERATE_VGPU_PER_PGPU_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_VGPU_MGR_INTERNAL_ENUMERATE_VGPU_PER_PGPU_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_VGPU_MGR_INTERNAL_ENUMERATE_VGPU_PER_PGPU_PARAMS {
    LwU32 numVgpu;
    LW_DECLARE_ALIGNED(LW2080_VGPU_GUEST vgpuGuest[LWA081_MAX_VGPU_PER_PGPU], 8);
} LW2080_CTRL_VGPU_MGR_INTERNAL_ENUMERATE_VGPU_PER_PGPU_PARAMS;

/*
 * LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_CLEAR_GUEST_VM_INFO
 *
 * This command is used clear guest vm info. It should be used when
 * LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_GUEST_LIFE_CYCLE_STATE
 * is called with LWA081_NOTIFIERS_EVENT_VGPU_GUEST_DESTROYED state.
 *
 * gfid [IN]
 *  This parameter specifies the gfid of vGPU assigned to VM.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_CLEAR_GUEST_VM_INFO (0x20804005) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_VGPU_MGR_INTERNAL_CLEAR_GUEST_VM_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_VGPU_MGR_INTERNAL_CLEAR_GUEST_VM_INFO_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_VGPU_MGR_INTERNAL_CLEAR_GUEST_VM_INFO_PARAMS {
    LwU32 gfid;
} LW2080_CTRL_VGPU_MGR_INTERNAL_CLEAR_GUEST_VM_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_GET_VGPU_FB_USAGE
 *
 * This command is used to get the FB usage of all vGPU instances running on a GPU.
 *
 * vgpuCount [OUT]
 *  This field specifies the number of vGPU devices for which FB usage is returned.
 * vgpuFbUsage [OUT]
 *  This is an array of type LW2080_VGPU_FB_USAGE, which contains a list of vGPU gfid
 *  and their corresponding FB usage in bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_GET_VGPU_FB_USAGE (0x20804006) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_VGPU_MGR_INTERNAL_GET_VGPU_FB_USAGE_PARAMS_MESSAGE_ID" */

typedef struct LW2080_VGPU_FB_USAGE {
    LwU32 gfid;
    LW_DECLARE_ALIGNED(LwU64 fbUsed, 8);
} LW2080_VGPU_FB_USAGE;

#define LW2080_CTRL_VGPU_MGR_INTERNAL_GET_VGPU_FB_USAGE_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_VGPU_MGR_INTERNAL_GET_VGPU_FB_USAGE_PARAMS {
    LwU32 vgpuCount;
    LW_DECLARE_ALIGNED(LW2080_VGPU_FB_USAGE vgpuFbUsage[LWA081_MAX_VGPU_PER_PGPU], 8);
} LW2080_CTRL_VGPU_MGR_INTERNAL_GET_VGPU_FB_USAGE_PARAMS;

/*
 * LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_SET_VGPU_ENCODER_CAPACITY
 *
 * This command is used to set vGPU instance's (represented by gfid) encoder Capacity.
 *
 * gfid [IN]
 *  This parameter specifies the gfid of vGPU assigned to VM.
 * encoderCapacity [IN]
 *  Encoder capacity value from 0 to 100. Value of 0x00 indicates encoder performance
 *  may be minimal for this GPU and software should fall back to CPU-based encode.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW2080_CTRL_CMD_VGPU_MGR_INTERNAL_SET_VGPU_ENCODER_CAPACITY (0x20804007) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_VGPU_MGR_INTERNAL_SET_VGPU_ENCODER_CAPACITY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_VGPU_MGR_INTERNAL_SET_VGPU_ENCODER_CAPACITY_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_VGPU_MGR_INTERNAL_SET_VGPU_ENCODER_CAPACITY_PARAMS {
    LwU32 gfid;
    LwU32 encoderCapacity;
} LW2080_CTRL_VGPU_MGR_INTERNAL_SET_VGPU_ENCODER_CAPACITY_PARAMS;

/*
 * LW2080_CTRL_CMD_INTERNAL_VGPU_PLUGIN_CLEANUP
 *
 * This command is used to cleanup all the GSP VGPU plugin task allocated resources after its shutdown.
 * Can be called only with SR-IOV and with VGPU_GSP_PLUGIN_OFFLOAD feature.
 *
 * gfid [IN]
 *  This parameter specifies the gfid of vGPU assigned to VM.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_INTERNAL_VGPU_PLUGIN_CLEANUP (0x20804008) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_VGPU_MGR_INTERNAL_INTERFACE_ID << 8) | LW2080_CTRL_INTERNAL_VGPU_PLUGIN_CLEANUP_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_INTERNAL_VGPU_PLUGIN_CLEANUP_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_INTERNAL_VGPU_PLUGIN_CLEANUP_PARAMS {
    LwU32 gfid;
} LW2080_CTRL_INTERNAL_VGPU_PLUGIN_CLEANUP_PARAMS;

/* _ctrl2080vgpumgrinternal_h_ */
