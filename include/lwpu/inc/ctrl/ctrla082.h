/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrla082.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl0080/ctrl0080fifo.h" // LW0080_CTRL_CMD_FIFO_IDLE_CHANNELS_MAX_CHANNELS
#include "ctrl/ctrl2080/ctrl2080gpu.h" // LW2080_GPU_MAX_NAME_STRING_LENGTH
#include "lw_vgpu_types.h" // VM_ID_TYPE
/* LWA082_HOST_VGPU_DEVICE control commands and parameters */
#define LWA082_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0xA082U, LWA082_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWA082_CTRL_RESERVED                                (0x00)
#define LWA082_CTRL_HOST_VGPU_DEVICE                        (0x01)

#define LWA082_DRIVER_VER_SIZE                              16
#define LWA082_BRANCH_VER_SIZE                              32
#define LWA082_VGPU_NAME_SIZE                               32
#define LWA082_VGPU_SIGNATURE_SIZE                          128
#define LWA082_MAX_VGPU_DEVICES_PER_PGPU                    32
#define LWA082_EXTRA_PARAMETERS_SIZE                        1024
#define LWA082_VGPU_PCI_ID_SIZE                             32
#define LWA082_ENGINE_TYPES_MAX                             128

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO
 *
 * This command gets the vGPU type information of host vgpu device.
 *
 * Parameters:
 *
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO_PARAMS [OUT]
 *  This parameter specifies the vGPU type information of the guest vGPU device
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_OBJECT_NOT_FOUND
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO (0xa0820102) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO_PARAMS {
    // This structure should be in sync with LWA081_CTRL_VGPU_INFO
    LwU32 vgpuType;
    LwU8  vgpuName[LWA082_VGPU_NAME_SIZE];
    LwU8  vgpuClass[LWA082_VGPU_NAME_SIZE];
    LwU8  vgpuSignature[LWA082_VGPU_SIGNATURE_SIZE];
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
    LwU8  vgpuExtraParams[LWA082_EXTRA_PARAMETERS_SIZE];
    LwU32 ftraceEnable;
    LwU32 gpuDirectSupported;
    LwU32 lwlinkP2PSupported;
} LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_TYPE_INFO_PARAMS;

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VM_INFO
 *
 * This command sets the vGPU guest VM's information
 *
 * Parameter:
 *
 * bDriverLoaded [IN]
 *  This parameter specifies whether driver is loaded on this particular vGPU.
 *
 * clearGuestInfo [IN]
 *  This parameter specifies whether to clear the vGPU guest VM's information.
 *  And whenever clearGuestInfo is true, guestDriverVersion and guestDriverBranch
 *  should be set to "Not Available".
 *
 * guestOs [IN]
 *  This parameter specifies the guest OS type.
 *
 * frameRateLimit [IN]
 *  This parameter specifies the current value of FRL set for the vGPU guest instance (VM),
 *  or if it is disabled.
 *
 * migrationProhibited [IN]
 *  This parameter specifies whether migration is prohibited for the vGPU guest instance (VM).

 * guestDriverVersion [IN]
 *  This parameter specifies the guest driver version
 *
 * guestDriverBranch [IN]
 *  This parameter specifies the guest branch
 *
 * vgpuPciId [IN]
 *  This parameter specifies the PCI id of the GPU assigned to the vGPU guest instance
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VM_INFO (0xa0820103) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_INFO_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_INFO_PARAMS {
    LwBool bDriverLoaded;
    LwBool clearGuestInfo;
    LwU32  guestOs;
    LwU32  frameRateLimit;
    LwU32  migrationProhibited;
    LwU32  guestNegotiatedVgpuVersion;
    LwU8   guestDriverVersion[LWA082_DRIVER_VER_SIZE];
    LwU8   guestDriverBranch[LWA082_BRANCH_VER_SIZE];
    LW_DECLARE_ALIGNED(LwU64 vgpuPciId, 8);
} LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_INFO_PARAMS;

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VM_LICENSE_INFO
 *
 * This command sets the vGPU guest VM's license information
 *
 * Parameter:
 *
 * licensed [IN]
 *  This parameter specifies the vGPU guest is licensed/unlicensed
 *
 * licenseState [IN]
 *  This parameter specifies the current state of RM's GRID license state machine in VM
 *
 * fpsValue [IN]
 *  This parameter specifies the current fps value in VM
 *
 * licenseExpiryTimestamp [IN]
 *  This parameter specifies the current value of license expiry in seconds since epoch time
 *
 * licenseExpiryStatus [IN]
 *  This parameter specifies the license expiry status
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VM_LICENSE_INFO (0xa0820104) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_LICENSE_INFO_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_LICENSE_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_LICENSE_INFO_PARAMS {
    LwBool licensed;
    LwU32  licenseState;
    LwU32  fpsValue;
    LwU32  licenseExpiryTimestamp;
    LwU8   licenseExpiryStatus;
} LWA082_CTRL_HOST_VGPU_DEVICE_SET_VM_LICENSE_INFO_PARAMS;

/* valid event action values */
#define LWA082_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE                 (0x00000000)
#define LWA082_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE                  (0x00000001)
#define LWA082_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT                  (0x00000002)

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_DEVICE_ENCODER_CAPACITY
 *
 * This command used to get encoder capacity for vgpu from host RM.
 *
 * Parameters:

 * encoderCapacity [OUT]
 *   Encoder capacity value from 0 to 100. Value of 0x00 indicates encoder performance
 *   may be minimal for this GPU and software should fall back to CPU-based encode.
 *
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_VGPU_DEVICE_ENCODER_CAPACITY (0xa082010d) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_GET_VGPU_DEVICE_ENCODER_CAPACITY_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_GET_VGPU_DEVICE_ENCODER_CAPACITY_PARAMS_MESSAGE_ID (0xDU)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_GET_VGPU_DEVICE_ENCODER_CAPACITY_PARAMS {
    LwU32 encoderCapacity;
} LWA082_CTRL_HOST_VGPU_DEVICE_GET_VGPU_DEVICE_ENCODER_CAPACITY_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VGPU_FB_USAGE
 *
 * This command is used to update FB usage for vgpu in host RM.
 *
 * Parameters:
 *
 * fbUsed [IN]
 *  Current FB used value in bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VGPU_FB_USAGE (0xa0820110) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_FB_USAGE_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_FB_USAGE_PARAMS_MESSAGE_ID (0x10U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_FB_USAGE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 fbUsed, 8);
} LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_FB_USAGE_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_S_CHID_BASE
 *
 * This command is used to get the system channel ID base for this guest. This
 * applies only to SR-IOV. If SR-IOV is disabled, this will return 0.
 *
 * Parameters:
 *
 * engineCount [IN]
 *    # of engines in the engine list of this guest
 *
 * engineList [IN]
 *    The list of LW2080_ENGINE_TYPE engines that belong to this guest.
 *    The entries may be local engine ids within the
 *    guest's partition in case SMC is enabled
 *
 * sChidBaseList [OUT]
 *    The list of schid bases for each engine in engineList
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_S_CHID_BASE (0xa0820111) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_GET_S_CHID_BASE_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_GET_S_CHID_BASE_PARAMS_MESSAGE_ID (0x11U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_GET_S_CHID_BASE_PARAMS {
    LwU32 engineCount; // [in]
    LwU32 engineList[LWA082_ENGINE_TYPES_MAX]; // [in]
    LwU32 sChidBaseList[LWA082_ENGINE_TYPES_MAX]; // [out]
} LWA082_CTRL_HOST_VGPU_DEVICE_GET_S_CHID_BASE_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VGPU_VMBUS
 *
 * This command is used to set VMBUS handle in RM.
 *
 * Parameters:
 *
 * hVmbus [IN]
 *  handle to vmbus.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_VGPU_VMBUS (0xa0820113) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_VMBUS_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_VMBUS_PARAMS_MESSAGE_ID (0x13U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_VMBUS_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 hVmbus, 8);
} LWA082_CTRL_HOST_VGPU_DEVICE_SET_VGPU_VMBUS_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESET_VGPU_VMBUS
 *
 * This command is used to reset VMBUS handle in RM.
 *
 * Parameters:
 *
 * hVmbus [IN]
 *  handle to vmbus.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESET_VGPU_VMBUS (0xa0820114) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_RESET_VGPU_VMBUS_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_RESET_VGPU_VMBUS_PARAMS_MESSAGE_ID (0x14U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_RESET_VGPU_VMBUS_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 hVmbus, 8);
} LWA082_CTRL_HOST_VGPU_DEVICE_RESET_VGPU_VMBUS_PARAMS;


/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_SHARED_MEMORY
 *
 * This command is used to set shared memory parameters
 * in host vgpu device in RM.
 *
 * Parameters:
 *
 * hSharedMemory [IN]
 *   Handle to shared memory
 * hSharedMemoryClient
 *   Handle to shared memory client
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_SHARED_MEMORY (0xa0820115) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_SET_SHARED_MEMORY_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_SET_SHARED_MEMORY_PARAMS_MESSAGE_ID (0x15U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_SET_SHARED_MEMORY_PARAMS {
    LwHandle hSharedMemory;
    LwHandle hSharedMemoryClient;
} LWA082_CTRL_HOST_VGPU_DEVICE_SET_SHARED_MEMORY_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_COMPLETE_VMBUS_PACKET
 *
 * VMbus packet is the message sent from host to guest to pin memory.
 * The memory gets unpinned once we mark this vm bus packet as complete.
 * This command is used to complete the vmbus packet.
 *
 * Parameters:
 *
 * vmBusPacketId [IN]
 *  VMBus Packet ID which needs to be completed
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_COMPLETE_VMBUS_PACKET (0xa0820117) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_COMPLETE_VMBUS_PACKET_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_COMPLETE_VMBUS_PACKET_PARAMS_MESSAGE_ID (0x17U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_COMPLETE_VMBUS_PACKET_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 vmBusPacketId, 8);
} LWA082_CTRL_HOST_VGPU_DEVICE_COMPLETE_VMBUS_PACKET_PARAMS;

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_ECC_STATUS
 *
 * This command sets the guest vgpu device's ECC status information
 *
 * Parameter:
 *
 * bEccEnabled [IN]
 *  This parameter specifies the guests ECC status
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_ECC_STATUS (0xa0820118) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_SET_ECC_STATUS_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_SET_ECC_STATUS_PARAMS_MESSAGE_ID (0x18U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_SET_ECC_STATUS_PARAMS {
    LwBool bEccEnabled;
} LWA082_CTRL_HOST_VGPU_DEVICE_SET_ECC_STATUS_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SAVE_VF_BAR_BLOCK_VALUE
 *
 * This command is used save VF Bar regions.
 *
 * regBar1Block
 *   This will contain VF's BAR1 block register value.
 *
 * regBar1BlockHigh
 *   This will contain VF's BAR1 block high register value.
 *
 * regBar2Block
 *   This will contain VF's BAR2 block register value
 *
 * regBar2BlockHigh
 *   This will contain VF's BAR2 block high register value
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_TIMEOUT
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SAVE_VF_BAR_BLOCK_VALUE (0xa0820119) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x19" */

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_VF_BAR_BLOCK_VALUE_PARAMS {
    LwU32 regBar1Block;
    LwU32 regBar1BlockHigh;
    LwU32 regBar2Block;
    LwU32 regBar2BlockHigh;
} LWA082_CTRL_HOST_VGPU_DEVICE_VF_BAR_BLOCK_VALUE_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_VF_BAR_BLOCK_VALUE
 *
 * This command is used restore VF Bar regions.
 *
 * Parameters:
 *   This command uses LWA082_CTRL_HOST_VGPU_DEVICE_VF_BAR_BLOCK_VALUE_PARAMS
 *   for providing BAR block values
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_TIMEOUT
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_VF_BAR_BLOCK_VALUE (0xa082011a) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x1A" */

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SAVE_INTR_REG_VALUE
 *
 * This command is used save interrupt register values.
 *
 * intr_top_en_set
 *   This will contain value stored in CPU_INTR_TOP_EN_SET register.
 *
 * intr_leaf_en_set[]
 *   This will contain the INTR_LEAF_EN_SET register values
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SAVE_INTR_REG_VALUE        (0xa082011b) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x1B" */

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_INTR_REG_VALUE
 *
 * This command is used restore interrupt register values.
 *
 * intr_top_en_set
 *   This will contain value stored in CPU_INTR_TOP_EN_SET register.
 *
 * intr_leaf_en_set[]
 *   This will contain the INTR_LEAF_EN_SET register values
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_INTR_REG_VALUE     (0xa082011c) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x1C" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_INTR_LEAF_EN_SET_ARRAY_SIZE    16
typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_UPDATE_INTR_REG_VALUE_PARAMS {
    LwU32 intr_top_en_set;
    LwU32 intr_leaf_en_set[LWA082_CTRL_HOST_VGPU_DEVICE_INTR_LEAF_EN_SET_ARRAY_SIZE];
} LWA082_CTRL_HOST_VGPU_DEVICE_UPDATE_INTR_REG_VALUE_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GER_BAR_MAPPING_RANGES
 *
 * This command is used to get Bar mapping ranges in RM.
 *
 * Parameters:
 * offsets [OUT]
 *  Offsets of the ranges
 * sizes [OUT]
 *  Sizes of the ranges
 * mitigated [OUT]
 *  Specifies whether it's mitigated range
 * numRanges [OUT]
 *  Number of ranges
 *
 * osPageSize [IN]
 *  Page size.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_BAR_MAPPING_RANGES (0xa082011e) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_GET_BAR_MAPPING_RANGES_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_MAX_BAR_MAPPING_RANGES     10

#define LWA082_CTRL_HOST_VGPU_DEVICE_GET_BAR_MAPPING_RANGES_PARAMS_MESSAGE_ID (0x1EU)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_GET_BAR_MAPPING_RANGES_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 offsets[LWA082_CTRL_HOST_VGPU_DEVICE_MAX_BAR_MAPPING_RANGES], 8);
    LW_DECLARE_ALIGNED(LwU64 sizes[LWA082_CTRL_HOST_VGPU_DEVICE_MAX_BAR_MAPPING_RANGES], 8);
    LwU32  numRanges;
    LwU32  osPageSize;
    LwBool mitigated[LWA082_CTRL_HOST_VGPU_DEVICE_MAX_BAR_MAPPING_RANGES];
} LWA082_CTRL_HOST_VGPU_DEVICE_GET_BAR_MAPPING_RANGES_PARAMS;

/*
 *LWA082_CTRL_CMD_HOST_VGPU_DEVICE_FREE_STATES
 *
 * This command is used to free the GR Global context buffers allocated
 * in HostVgpuDevice when guest is rebooted.
 *
 * Parameters:
 *
 * flags [IN]
 *   Specifies what component of HostVgpuDevice to free.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_FREE_STATES                       (0xa082011f) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_FREE_STATES_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_FLAGS_GLOBAL_GR_CONTEXT_BUFFERS          0:0
#define LWA082_CTRL_HOST_VGPU_DEVICE_FLAGS_GLOBAL_GR_CONTEXT_BUFFERS_TRUE  0x1
#define LWA082_CTRL_HOST_VGPU_DEVICE_FLAGS_GLOBAL_GR_CONTEXT_BUFFERS_FALSE 0x0

#define LWA082_CTRL_HOST_VGPU_DEVICE_FREE_STATES_PARAMS_MESSAGE_ID (0x1FU)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_FREE_STATES_PARAMS {
    LwU32 flags;
} LWA082_CTRL_HOST_VGPU_DEVICE_FREE_STATES_PARAMS;

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_DEFAULT_EXEC_PARTITION
 *
 * This command restores the vGPU device's default EXEC partition saved in
 * HOST_VGPU_DEVICE.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_DEFAULT_EXEC_PARTITION (0xa0820120) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x20" */

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SAVE_VF_FAULT_BUFFER_INFO
 *
 * This command is used save VF fault buffer info
 *
 * nonReplayableFaultBufferHi
 *   This will contain VF's fault buffer location MSB bits
 *
 * nonReplayableFaultBufferLo
 *   This will contain VF's fault buffer location LSB bits
 *
 * nonReplayableFaultBufferSize
 *   This will contain VF's fault buffer size
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SAVE_VF_FAULT_BUFFER_INFO      (0xa0820123) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x23" */

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_VF_FAULT_BUFFER_INFO_PARAMS {
    LwU32 nonReplayableFaultBufferHi;
    LwU32 nonReplayableFaultBufferLo;
    LwU32 nonReplayableFaultBufferSize;
} LWA082_CTRL_HOST_VGPU_DEVICE_VF_FAULT_BUFFER_INFO_PARAMS;

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_VF_FAULT_BUFFER_INFO
 *
 * This command is used restore VF fault buffer info
 *
 * Parameters:
 *   This command uses LWA082_CTRL_HOST_VGPU_DEVICE_VF_FAULT_BUFFER_INFO_PARAMS
 *   for providing fault buffer info
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_RESTORE_VF_FAULT_BUFFER_INFO (0xa0820124) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x24" */

/* LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_IDLE_STATUS
 *
 * This command is used to ascertain whether 1 or more channels belonging to a certain guest is
 * idle or not. This information helps tell the plugin which channels to wait on.
 *
 * Parameters:
 *
 * numChIds
 *   Number of virtual/physical channel ID's to collect idle status on.
 *
 * chIds
 *   Array of channel information for collecting idle status on.
 *
 *   engineId
 *     Physical engine ID (LW2080_ENGINE_TYPE) that the channel is running on.
 *
 *   idleStatus [OUT]
 *     True if channel is idle, False otherwise
 *
 *   chId
 *     Channel ID, this is a virtual channel ID if SR-IOV is enabled but physical (sChId) otherwise.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_GET_IDLE_STATUS              (0xa0820126) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_GET_IDLE_STATUS_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_GET_IDLE_STATUS_PARAMS_MESSAGE_ID (0x26U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_GET_IDLE_STATUS_PARAMS {
    LwU32 numChIds;
    struct {
        LwU16  chId;
        LwU8   engineId;
        LwBool bIdle;
    } chIds[LW0080_CTRL_CMD_FIFO_IDLE_CHANNELS_MAX_CHANNELS];
} LWA082_CTRL_HOST_VGPU_DEVICE_GET_IDLE_STATUS_PARAMS;

/* These defines are applicable per VM i.e. replicated per VM. */
#define LW_VF_SCRATCH_REG_SIZE                                   16 /* LW_VIRTUAL_FUNCTION_PRIV_MAILBOX_SCRATCH__SIZE_1 */

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_READ_VF_SCRATCH_INDEXES
 * 
 * This command is used to read scratch registers inside GSP-Plugin.
 *
 * Parameters:
 *   vfScratchRegNum
 *     - Number of scratch registers we want to read.
 *   vfScratchRegIndex
 *     - This field has the indexes of VF scratch registers we want to read/write.
 *
 *   vfScratchRegVal
 *     - This field has the value stored in VF scratch registers we want to read/write.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_READ_VF_SCRATCH_INDEXES (0xa0820127) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x27" */

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_VF_SCRATCH_INDEXES_PARAMS {
    LwU32 vfScratchRegNum;
    LwU32 vfScratchRegIndex[LW_VF_SCRATCH_REG_SIZE]; // [IN]
    LwU32 vfScratchRegVal[LW_VF_SCRATCH_REG_SIZE]; // [OUT] for read and [IN] for write
} LWA082_CTRL_HOST_VGPU_DEVICE_VF_SCRATCH_INDEXES_PARAMS;

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_WRITE_VF_SCRATCH_INDEXES
 * 
* Parameters:
 *   This command is used to write VF scratch registers inside GSP-Plugin.
 *   This command uses LWA082_CTRL_HOST_VGPU_DEVICE_VF_SCRATCH_INDEXES_PARAMS
 *   for providing vf scratch register values.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_WRITE_VF_SCRATCH_INDEXES (0xa0820128) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | 0x28" */

/*
 * LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP
 *
 * This structure describes register operation information for use with
 * the LWA082_CTRL_CMD_HOST_VGPU_DEVICE_EXEC_VF_REG_OPS command.  The structure describes
 * a single register operation.  The operation can be a read or write and
 * can involve either 32bits or 64bits of data.
 *
 * For read operations, the operation takes the following form:
 *
 *   regValue = read(bar0 + regOffset)
 *
 * For write operations, the operation takes the following form:
 *
 *   new = ((read(bar0 + regOffset) & ~regAndNMask) | regValue)
 *   write(bar0 + regOffset, new)
 *
 * Details on the parameters follow:
 *
 *   regOp
 *     This field specifies the operation to be applied to the register
 *     specified by the regOffset parameter.  Valid values for this
 *     parameter are:
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_READ_32
 *         The register operation should be a 32bit register read.
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_WRITE_32
 *         The register operation should be a 32bit register write.
 *   regType
 *     This field specifies the type of the register specified by the
 *     regOffset parameter.  Valid values for this parameter are:
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_MMU_FAULT
 *         The registers used to set the replayable and non-replayable fault
 *         buffers configurations.
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_MMU_TLB_ILWALIDATE
 *         The registers used to perform a TLB ilwalidate.
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_ACCESS_COUNTER
 *         The registers used by the Access Counter feature in HUB.
 *   regStatus
 *     This field returns the completion status for the associated register
 *     operation in the form of a bitmask.  Possible status values for this
 *     field are:
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_SUCCESS
 *         This value indicates the operation completed successfully.
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_ILWALID_OP
 *         This bit value indicates that the regOp value is not valid.
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_ILWALID_TYPE
 *         This bit value indicates that the regType value is not valid.
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_ILWALID_OFFSET
 *         This bit value indicates that the regOffset value is invalid.
 *         The regOffset value must be within the legal BAR0 range for the
 *         associated GPU and must target a supported register with a
 *         supported operation.
 *       LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_UNSUPPORTED_OFFSET
 *         This bit value indicates that the operation to the register
 *         specified by the regOffset value is not supported for the
 *         associated GPU.
 *   regOffset
 *     This field specifies the register offset to access.  The specified
 *     offset must be a valid BAR0 offset for the associated VF.
 *   regValue
 *     This field contains the register value.
 *     For read operations, this value returns the current value of the
 *     register specified by regOffset.  For write operations, this field
 *     specifies the logical OR value applied to the current value
 *     contained in the register specified by regOffset.
 *   regAndNMask
 *     This field contains the mask used to clear a desired field from
 *     the current value contained in the register specified by regOffsetLo.
 *     This field is negated and ANDed to this current register value.
 *     This field is only used for write operations.  This field is ignored
 *     for read operations.
 */
typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP {
    LwU8  regOp;
    LwU8  regType;
    LwU8  regStatus;
    LwU32 regOffset;
    LwU32 regValue;
    LwU32 regAndNMask;
} LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP;

/* valid regOp values */
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_READ_32                 (0x00000000)
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_WRITE_32                (0x00000001)

/* valid regType values */
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_MMU_FAULT          (0x00000000)
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_MMU_TLB_ILWALIDATE (0x00000001)
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_ACCESS_COUNTER     (0x00000002)

/* valid regStatus values */
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_SUCCESS          (0x00000000)
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_ILWALID_OP       (0x00000001)
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_ILWALID_TYPE     (0x00000002)
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_ILWALID_OFFSET   (0x00000004)
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_STATUS_UNSUPPORTED_OP   (0x00000008)

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_EXEC_VF_REG_OPS
 *
 * This command is used to submit a buffer containing one or more
 * LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP structures for processing.  Each entry in the
 * buffer specifies a single read or write operation.  Each entry is checked
 * for validity in an initial pass over the buffer with the results for
 * each operation stored in the corresponding regStatus field. If any invalid
 * entries are found during this intial pass then none of the operations are exelwted.
 * Entries are processed in order within each regType with LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_MMU_FAULT
 * entries processed first followed by LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP_TYPE_MMU_TLB_ILWALIDATE and so on.
 *
 *   regOpCount
 *     This field specifies the number of entries on the caller's regOps
 *     list.
 *   regOps
 *     This field specifies an inlined array of regOps from which the desired
 *     register information is to be
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

/* setting this to 128 keeps it below 4k in size */
#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OPS_ARRAY_MAX              128

#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_EXEC_VF_REG_OPS               (0xa0820129) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_VF_EXEC_REG_OPS_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_VF_EXEC_REG_OPS_PARAMS_MESSAGE_ID (0x29U)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_VF_EXEC_REG_OPS_PARAMS {
    LwU32                                  regOpCount;
    LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OP regOps[LWA082_CTRL_HOST_VGPU_DEVICE_VF_REG_OPS_ARRAY_MAX];
} LWA082_CTRL_HOST_VGPU_DEVICE_VF_EXEC_REG_OPS_PARAMS;

/* valid action values */
//  @todo Define actual action values.

/*
 * LWA082_CTRL_CMD_SEND_EVENT_NOTIFICATION
 *
 * This command triggers a software event for the associated hostvgpudevice.
 * We will use this command only in GSP-Firmware. This is used for GSP-Plugin to communicate with CPU-RM.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWA082_CTRL_CMD_SEND_EVENT_NOTIFICATION (0xa082012a) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_SEND_EVENT_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_SEND_EVENT_NOTIFICATION_PARAMS_MESSAGE_ID (0x2AU)

typedef struct LWA082_CTRL_SEND_EVENT_NOTIFICATION_PARAMS {
    //
    // @todo: We will define the actual event values later based on the use case.
    // These event values are only for Test purpose.
    //
    LwU32 eventIndex;
} LWA082_CTRL_SEND_EVENT_NOTIFICATION_PARAMS;

/*
 * LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_RESTORE_STATE
 *
 * This command is used to notify RM whether vGPU VM is restoring during vGPU VM migration.
 *
 *   bRestore
 *     This field specifies whether vGPU VM is restoring or not.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LWA082_CTRL_CMD_HOST_VGPU_DEVICE_SET_RESTORE_STATE (0xa082012b) /* finn: Evaluated from "(FINN_LWA082_HOST_VGPU_DEVICE_HOST_VGPU_DEVICE_INTERFACE_ID << 8) | LWA082_CTRL_HOST_VGPU_DEVICE_SET_RESTORE_STATE_PARAMS_MESSAGE_ID" */

#define LWA082_CTRL_HOST_VGPU_DEVICE_SET_RESTORE_STATE_PARAMS_MESSAGE_ID (0x2BU)

typedef struct LWA082_CTRL_HOST_VGPU_DEVICE_SET_RESTORE_STATE_PARAMS {
    LwBool bRestore;
} LWA082_CTRL_HOST_VGPU_DEVICE_SET_RESTORE_STATE_PARAMS;

/* _ctrlA082hostvgpudevice_h_ */
