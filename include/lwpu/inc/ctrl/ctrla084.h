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
// Source file: ctrl/ctrla084.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#include "lw_vgpu_types.h"


#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h" // LW2080_GPU_MAX_GID_LENGTH
#include "ctrl/ctrl2080/ctrl2080fb.h" // LW2080_CTRL_FB_OFFLINED_PAGES_MAX_PAGES

/* LWA084_HOST_VGPU_DEVICE control commands and parameters */
#define LWA084_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0xA084U, LWA084_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWA084_CTRL_RESERVED                                         (0x00)
#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL                          (0x01)

/*
 * LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_DEVICE_INFO
 *
 * This command sets the guest vgpu device's information
 *
 * Parameter:
 *
 * vgpuUuid [IN]
 *  This parameter specifies the universaly unique identifier of the guest vGPU device
 *
 * vgpuDeviceInstanceId [IN]
 *  This parameter specifies the vGPU device instance
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_DEVICE_INFO (0xa0840101) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_DEVICE_INFO_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_DEVICE_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_DEVICE_INFO_PARAMS {
    LwU8  vgpuUuid[LW2080_GPU_MAX_GID_LENGTH];
    LwU32 vgpuDeviceInstanceId;
} LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_DEVICE_INFO_PARAMS;

/* LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_GUEST_LIFE_CYCLE_STATE
 *
 * This command triggers the notifier for vGPU guest.
 *
 * Parameters:
 *
 * vmLifeCycleState[IN]
 * The life cycle event of the vGPU guest. This can be:
 *   LWA081_NOTIFIERS_EVENT_VGPU_GUEST_DESTROYED
 *   LWA081_NOTIFIERS_EVENT_VGPU_GUEST_CREATED
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_GUEST_LIFE_CYCLE_STATE (0xa0840102) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_GUEST_LIFE_CYCLE_STATE_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_GUEST_LIFE_CYCLE_STATE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_GUEST_LIFE_CYCLE_STATE_PARAMS {
    LwU32 vmLifeCycleState;
} LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_VGPU_GUEST_LIFE_CYCLE_STATE_PARAMS;

/*
 * LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_BL_PAGE_PATCHINFO
 *
 * This command is used to copy black listed page mapping info from plugin to host vgpu device
 *
 *   guestFbSegmentPageSize [in]
 *     Guest FB segment page size
 *
 *   blPageCount [in]
 *     blacklisted page count in the guest FB range
 *
 *   gpa [in]
 *     This array represents guest page address list of black listed page
 *
 *   hMemory [in]
 *     This array represents memory handle list of good page
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 */
#define LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_BL_PAGE_PATCHINFO (0xa0840103) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_BL_PAGE_PATCHINFO_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_BL_PAGE_PATCHINFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_BL_PAGE_PATCHINFO_PARAMS {
    LwU32    guestFbSegmentPageSize;
    LwU32    blPageCount;
    LW_DECLARE_ALIGNED(LwU64 gpa[LW2080_CTRL_FB_OFFLINED_PAGES_MAX_PAGES], 8);
    LwHandle hMemory[LW2080_CTRL_FB_OFFLINED_PAGES_MAX_PAGES];
} LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_BL_PAGE_PATCHINFO_PARAMS;

/*
 * LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS
 *
 * Config space access for the virtual function
 *
 * Parameters:
 * 
 * offset [IN]
 *   Offset within the config space
 *
 * numBytes [IN]
 *   Number of bytes to be read/written: 1/2/4
 *
 * accessType [IN]
 *   To indicate whether it is a read operation or a write operation
 *
 * value [INOUT]
 *   Value to be read or written
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS            (0xa0840104) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS_TYPE_READ  0x1
#define LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS_TYPE_WRITE 0x2

#define LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS_PARAMS {
    LwU16 offset;
    LwU8  numBytes;
    LwU8  accessType;
    LwU32 value;
} LWA084_CTRL_CMD_VF_CONFIG_SPACE_ACCESS_PARAMS;

/*
 * LWA084_CTRL_CMD_BIND_FECS_EVTBUF
 *
 * This command is used to create a bind-point on a host system
 * for the collection of guest VM FECS events
 *
 *  hEventBufferClient[IN]
 *      The client of the event buffer to bind to
 *
 *  hEventBufferSubdevice[IN]
 *      The subdevice of the event buffer to bind to
 *
 *  hEventBuffer[IN]
 *      The event buffer to bind to
 *
 *  reasonCode[OUT]
 *      The reason for failure (if any); see LW2080_CTRL_GR_FECS_BIND_EVTBUF_REASON_CODE
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWA084_CTRL_CMD_BIND_FECS_EVTBUF (0xa0840105) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_BIND_FECS_EVTBUF_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_BIND_FECS_EVTBUF_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWA084_CTRL_BIND_FECS_EVTBUF_PARAMS {
    LwHandle hEventBufferClient;
    LwHandle hEventBufferSubdevice;
    LwHandle hEventBuffer;
    LwU32    reasonCode;
} LWA084_CTRL_BIND_FECS_EVTBUF_PARAMS;

/*
 * LWA084_CTRL_CMD_TRIGGER_PRIV_DOORBELL
 *
 * The command will trigger the specified interrupt on the host from CPU Plugin.
 *
 *   handle[IN]
 *      - An opaque handle that will be passed in along with the interrupt
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWA084_CTRL_CMD_TRIGGER_PRIV_DOORBELL (0xa0840106) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_TRIGGER_PRIV_DOORBELL_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_TRIGGER_PRIV_DOORBELL_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWA084_CTRL_TRIGGER_PRIV_DOORBELL_PARAMS {
    LwU32 handle;
} LWA084_CTRL_TRIGGER_PRIV_DOORBELL_PARAMS;

/* valid action values */
#define LWA084_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LWA084_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LWA084_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)

/*
 * LWA084_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the associated host vgpu device.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the associated host vgpu device object.
 */
#define LWA084_CTRL_CMD_EVENT_SET_NOTIFICATION (0xa0840107) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x7U)

typedef struct LWA084_CTRL_EVENT_SET_NOTIFICATION_PARAMS {
    //
    // @todo: We will define the actual event values later based on the use case.
    // These event values are only for Test purpose.
    //
    LwU32  event;
    LwU32  action;
    LwBool bNotifyState;
} LWA084_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_SRIOV_STATE
 *
 * This command is used to set SRIOV state parameters in RM.
 *
 * Parameters:
 *
 * numPluginChannels [IN]
 *   Number of channels required by plugin
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_SRIOV_STATE                 (0xa0840108) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_SRIOV_STATE_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_SRIOV_STATE_MAX_PLUGIN_CHANNELS 5

#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_SRIOV_STATE_PARAMS_MESSAGE_ID (0x8U)

typedef struct LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_SRIOV_STATE_PARAMS {
    LwU32 numPluginChannels;
} LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_SRIOV_STATE_PARAMS;

/* LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID
 *
 * This command is used to set/unset VM ID parameters in host vgpu device in RM.
 *
 * Parameters:
 *
 * action
 *   This parameter specifies the desired set guest id action.
 *   Valid set guest id actions include:
 *     LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_ACTION_SET
 *       This action sets the VM ID information in host vgpu device.
 *     LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_ACTION_UNSET
 *       This action unsets the VM ID information in host vgpu device.
 * vmPid [IN]
 *   VM process ID
 * vmIdType[IN]
 *   VM ID type whether it's UUID or DOMAIN_ID
 * guestVmId[IN]
 *   VM ID
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LWA084_CTRL_CMD_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID (0xa0840109) /* finn: Evaluated from "(FINN_LWA084_HOST_VGPU_DEVICE_KERNEL_HOST_VGPU_DEVICE_KERNEL_INTERFACE_ID << 8) | LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_PARAMS_MESSAGE_ID" */

#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_PARAMS_MESSAGE_ID (0x9U)

typedef struct LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_PARAMS {
    LwU8       action;
    LwU32      vmPid;
    VM_ID_TYPE vmIdType;
    LW_DECLARE_ALIGNED(VM_ID guestVmId, 8);
} LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_PARAMS;

#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_ACTION_SET   (0x00000000)
#define LWA084_CTRL_HOST_VGPU_DEVICE_KERNEL_SET_GUEST_ID_ACTION_UNSET (0x00000001)

/* _ctrla084hostvgpudevicekernel_h_ */
