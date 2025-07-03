/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0080/ctrl0080gpu.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0080/ctrl0080base.h"
#include "lwlimits.h"


/* LW01_DEVICE_XX/LW03_DEVICE gpu control commands and parameters */

/*
 * LW0080_CTRL_CMD_GPU_GET_CLASSLIST
 *
 * This command returns supported class information for the specified device.
 * If the device is comprised of more than one GPU, the class list represents
 * the set of supported classes common to all GPUs within the device.
 *
 * It has two modes:
 *
 * If the classList pointer is NULL, then this command returns the number
 * of classes supported by the device in the numClasses field.  The value
 * should then be used by the client to allocate a classList buffer
 * large enough to hold one 32bit value per numClasses entry.
 *
 * If the classList pointer is non-NULL, then this command returns the
 * set of supported class numbers in the specified buffer.
 *
 *   numClasses
 *     If classList is NULL, then this parameter will return the
 *     number of classes supported by the device.  If classList is non-NULL,
 *     then this parameter indicates the number of entries in classList.
 *   classList
 *     This parameter specifies a pointer to the client's buffer into
 *     which the supported class numbers should be returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW0080_CTRL_CMD_GPU_GET_CLASSLIST (0x800201) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_CLASSLIST_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_CLASSLIST_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_GPU_GET_CLASSLIST_PARAMS {
    LwU32 numClasses;
    LW_DECLARE_ALIGNED(LwP64 classList, 8);
} LW0080_CTRL_GPU_GET_CLASSLIST_PARAMS;

/**
 * LW0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES
 *
 * This command returns the number of subdevices for the device.
 *
 *   numSubDevices
 *     This parameter returns the number of subdevices within the device.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES (0x800280) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_NUM_SUBDEVICES_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_NUM_SUBDEVICES_PARAMS_MESSAGE_ID (0x80U)

typedef struct LW0080_CTRL_GPU_GET_NUM_SUBDEVICES_PARAMS {
    LwU32 numSubDevices;
} LW0080_CTRL_GPU_GET_NUM_SUBDEVICES_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_GET_VIDLINK_ORDER
 *
 * This command returns the video link order of each subdevice id inside the
 * device.  This call can only be made after SLI is enabled.  This call is
 * intended for 3D clients to use to determine the vidlink order of the
 * devices.  The Display Output Parent will always be the first subdevice
 * mask listed in the array.  Note that this command should not be used in
 * case of bridgeless SLI.  The order of the subdevices returned by this
 * command will not be correct in case of bridgeless SLI.
 *
 *   ConnectionCount
 *     Each HW can provide 1 or 2 links between all GPUs in a device.  This
 *     number tells how many links are available between GPUs.  This data
 *     also represents the number of conlwrrent SLI heads that can run at
 *     the same time over this one device.
 *
 *   Order
 *     This array returns the order of subdevices that are used through
 *     the vidlink for display output.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0080_CTRL_CMD_GPU_GET_VIDLINK_ORDER (0x800281) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_VIDLINK_ORDER_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_VIDLINK_ORDER_PARAMS_MESSAGE_ID (0x81U)

typedef struct LW0080_CTRL_GPU_GET_VIDLINK_ORDER_PARAMS {
    LwU32 ConnectionCount;
    LwU32 Order[LW_MAX_SUBDEVICES];
} LW0080_CTRL_GPU_GET_VIDLINK_ORDER_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_SET_DISPLAY_OWNER
 *
 * This command sets display ownership within the device to the specified
 * subdevice instance.  The actual transfer of display ownership will take
 * place at the next modeset.
 *
 *   subDeviceInstance
 *     This member specifies the subdevice instance of the new display
 *     owner.  The subdevice instance must be in the legal range
 *     indicated by the LW0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES command.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0080_CTRL_CMD_GPU_SET_DISPLAY_OWNER (0x800282) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_SET_DISPLAY_OWNER_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_SET_DISPLAY_OWNER_PARAMS_MESSAGE_ID (0x82U)

typedef struct LW0080_CTRL_GPU_SET_DISPLAY_OWNER_PARAMS {
    LwU32 subDeviceInstance;
} LW0080_CTRL_GPU_SET_DISPLAY_OWNER_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_GET_DISPLAY_OWNER
 *
 * This command returns the subdevice instance of the current display owner
 * within the device.
 *
 *   subDeviceInstance
 *     This member returns the subdevice instance of the current display
 *     owner.  The subdevice instance will be in the legal range
 *     indicated by the LW0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES command.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0080_CTRL_CMD_GPU_GET_DISPLAY_OWNER (0x800283) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_DISPLAY_OWNER_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_DISPLAY_OWNER_PARAMS_MESSAGE_ID (0x83U)

typedef struct LW0080_CTRL_GPU_GET_DISPLAY_OWNER_PARAMS {
    LwU32 subDeviceInstance;
} LW0080_CTRL_GPU_GET_DISPLAY_OWNER_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_SET_VIDLINK
 *
 * This command enables or disables the VIDLINK of all subdevices in the
 * current SLI configuration.
 *
 *   enable
 *     Enables or disables the vidlink
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_GPU_SET_VIDLINK (0x800285) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_SET_VIDLINK_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_SET_VIDLINK_PARAMS_MESSAGE_ID (0x85U)

typedef struct LW0080_CTRL_GPU_SET_VIDLINK_PARAMS {
    LwU32 enable;
} LW0080_CTRL_GPU_SET_VIDLINK_PARAMS;

#define LW0080_CTRL_GPU_SET_VIDLINK_ENABLE_FALSE                 (0x00000000)
#define LW0080_CTRL_GPU_SET_VIDLINK_ENABLE_TRUE                  (0x00000001)

/* commands */
#define LW0080_CTRL_CMD_GPU_VIDEO_POWERGATE_GET_STATUS           0
#define LW0080_CTRL_CMD_GPU_VIDEO_POWERGATE_POWERDOWN            1
#define LW0080_CTRL_CMD_GPU_VIDEO_POWERGATE_POWERUP              2

/* status */
#define LW0080_CTRL_CMD_GPU_VIDEO_POWERGATE_STATUS_POWER_ON      0
#define LW0080_CTRL_CMD_GPU_VIDEO_POWERGATE_STATUS_POWERING_DOWN 1
#define LW0080_CTRL_CMD_GPU_VIDEO_POWERGATE_STATUS_GATED         2
#define LW0080_CTRL_CMD_GPU_VIDEO_POWERGATE_STATUS_POWERING_UP   3

/*
 * LW0080_CTRL_CMD_GPU_MODIFY_SW_STATE_PERSISTENCE
 *
 * This command is used to enable or disable the persistence of a GPU's
 * software state when no clients exist. With persistent software state enabled
 * the GPU's software state is not torn down when the last client exits, but is
 * retained until either the kernel module unloads or persistent software state
 * is disabled.
 *
 *  newState
 *    This input parameter is used to enable or disable the persistence of the
 *    software state of all subdevices within the device.
 *    Possible values are:
 *      LW0080_CTRL_GPU_SW_STATE_PERSISTENCE_ENABLED
 *      LW0080_CTRL_GPU_SW_STATE_PERSISTENCE_DISABLED
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0080_CTRL_CMD_GPU_MODIFY_SW_STATE_PERSISTENCE          (0x800287) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_MODIFY_SW_STATE_PERSISTENCE_PARAMS_MESSAGE_ID" */

/* Possible values of persistentSwState */
#define LW0080_CTRL_GPU_SW_STATE_PERSISTENCE_ENABLED             (0x00000000)
#define LW0080_CTRL_GPU_SW_STATE_PERSISTENCE_DISABLED            (0x00000001)

#define LW0080_CTRL_GPU_MODIFY_SW_STATE_PERSISTENCE_PARAMS_MESSAGE_ID (0x87U)

typedef struct LW0080_CTRL_GPU_MODIFY_SW_STATE_PERSISTENCE_PARAMS {
    LwU32 newState;
} LW0080_CTRL_GPU_MODIFY_SW_STATE_PERSISTENCE_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_QUERY_SW_STATE_PERSISTENCE
 *
 *   swStatePersistence
 *     This parameter returns a value indicating if persistent software
 *     state is lwrrently enabled or not for the specified GPU. See the
 *     description of LW0080_CTRL_CMD_GPU_MODIFY_SW_STATE_PERSISTENCE.
 *     Possible values are:
 *       LW0080_CTRL_GPU_SW_STATE_PERSISTENCE_ENABLED
 *       LW0080_CTRL_GPU_SW_STATE_PERSISTENCE_DISABLED
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0080_CTRL_CMD_GPU_QUERY_SW_STATE_PERSISTENCE (0x800288) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_QUERY_SW_STATE_PERSISTENCE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_QUERY_SW_STATE_PERSISTENCE_PARAMS_MESSAGE_ID (0x88U)

typedef struct LW0080_CTRL_GPU_QUERY_SW_STATE_PERSISTENCE_PARAMS {
    LwU32 swStatePersistence;
} LW0080_CTRL_GPU_QUERY_SW_STATE_PERSISTENCE_PARAMS;

/**
 * LW0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE
 *
 * This command returns a value indicating virtualization mode in
 * which the GPU is running.
 *
 *   virtualizationMode
 *     This parameter returns the virtualization mode of the device.
 *     Possible values are:
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_NONE
 *         This value indicates that there is no virtualization mode associated with the
 *         device (i.e. it's a baremetal GPU).
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_NMOS
 *         This value indicates that the device is associated with the NMOS.
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_VGX
 *         This value indicates that the device is associated with VGX(guest GPU).
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST_VGPU
 *         This value indicates that the device is associated with vGPU(host GPU).
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST_VSGA
 *         This value indicates that the device is associated with vSGA(host GPU).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE   (0x800289) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_VIRTUALIZATION_MODE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_VIRTUALIZATION_MODE_NONE      (0x00000000)
#define LW0080_CTRL_GPU_VIRTUALIZATION_MODE_NMOS      (0x00000001)
#define LW0080_CTRL_GPU_VIRTUALIZATION_MODE_VGX       (0x00000002)
#define LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST      (0x00000003)
#define LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST_VGPU LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST
#define LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST_VSGA (0x00000004)

#define LW0080_CTRL_GPU_GET_VIRTUALIZATION_MODE_PARAMS_MESSAGE_ID (0x89U)

typedef struct LW0080_CTRL_GPU_GET_VIRTUALIZATION_MODE_PARAMS {
    LwU32 virtualizationMode;
} LW0080_CTRL_GPU_GET_VIRTUALIZATION_MODE_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW0080_CTRL_CMD_GPU_SET_OPERATION_MODE
 *
 * This command changes the GPU Operation Mode
 * configuration setting for a GPU given its
 * device handle.  The value specified is
 * stored in non-volatile (inforom) memory on the board and will take
 * effect with the next VBIOS POST. The IMMEDIATE bit can be specified
 * only to go from GOM A to GOM E and vice versa (for toggling Double
 * Precision). This change takes place immediately.
 *
 *   newOperationMode
 *     The new configuration setting to take effect with
 *     the next VBIOS POST.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_GPU_SET_OPERATION_MODE (0x80028a) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_SET_OPERATION_MODE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_SET_OPERATION_MODE_PARAMS_MESSAGE_ID (0x8AU)

typedef struct LW0080_CTRL_GPU_SET_OPERATION_MODE_PARAMS {
    LwU32 newOperationMode;
} LW0080_CTRL_GPU_SET_OPERATION_MODE_PARAMS;

#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_CONFIGURATION              8:0
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_CONFIGURATION_A (0x00000001)
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_CONFIGURATION_B (0x00000002)
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_CONFIGURATION_C (0x00000004)
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_CONFIGURATION_D (0x00000008)
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_CONFIGURATION_E (0x00000010)
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_IMMEDIATE                31:31
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_IMMEDIATE_FALSE (0x00000000)
#define LW0080_CTRL_GPU_SET_OPERATION_MODE_NEW_OPERATION_MODE_IMMEDIATE_TRUE  (0x00000001)

/*
 * LW0080_CTRL_CMD_GPU_QUERY_D_PRECISION_CAPABILITIES
 *
 * This command returns whether Double Precision is enabled or
 * disabled on GPUs in the associated device.  If Double Presision
 * is not supported for the GPUs (i.e. the GPUs are not VdChip
 * GPUs) then a NOT_SUPPORTED status value is returned.
 *
 *   bIsDpEnabled
 *     This parameter returns if Double Precision is enabled or
 *     disabled.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_GPU_QUERY_D_PRECISION_CAPABILITIES                    (0x80028b) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_QUERY_D_PRECISION_CAPABILITIES_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_QUERY_D_PRECISION_CAPABILITIES_PARAMS_MESSAGE_ID (0x8BU)

typedef struct LW0080_CTRL_GPU_QUERY_D_PRECISION_CAPABILITIES_PARAMS {
    LwBool bIsDpEnabled;
} LW0080_CTRL_GPU_QUERY_D_PRECISION_CAPABILITIES_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_GET_SPARSE_TEXTURE_COMPUTE_MODE
 *
 * This command returns the setting information for sparse texture compute
 * mode optimization on the associated GPU. This setting indicates how the
 * large page size should be selected by the RM for the GPU.
 *
 *   defaultSetting
 *     This field specifies what the OS default setting is for the associated
 *     GPU. See LW0080_CTRL_CMD_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE for a list
 *     of possible values.
 *   lwrrentSetting
 *     This field specifies which optimization mode was applied when the
 *     driver was loaded. See
 *     LW0080_CTRL_CMD_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE for a list of
 *     possible values.
 *   pendingSetting
 *     This field specifies which optimization mode will be applied on the
 *     next driver reload. See
 *     LW0080_CTRL_CMD_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE for a list of
 *     possible values.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0080_CTRL_CMD_GPU_GET_SPARSE_TEXTURE_COMPUTE_MODE (0x80028c) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS_MESSAGE_ID (0x8LW)

typedef struct LW0080_CTRL_GPU_GET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS {
    LwU32 defaultSetting;
    LwU32 lwrrentSetting;
    LwU32 pendingSetting;
} LW0080_CTRL_GPU_GET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE
 *
 * This command sets the pending setting for sparse texture compute mode. This
 * setting indicates how the large page size should be selected by the RM for
 * the GPU on the next driver reload.
 *
 *   setting
 *     This field specifies which use case the RM should optimize the large
 *     page size for on the next driver reload. Possible values for this
 *     field are:
 *       LW0080_CTRL_GPU_SPARSE_TEXTURE_COMPUTE_MODE_DEFAULT
 *         This value indicates that the RM should use the default setting for
 *         the GPU's large page size. The default setting is reported by
 *         LW0080_CTRL_CMD_GPU_GET_SPARSE_TEXTURE_COMPUTE_MODE.
 *       LW0080_CTRL_GPU_SPARSE_TEXTURE_COMPUTE_MODE_OPTIMIZE_COMPUTE
 *         This value indicates that the RM should select the GPU's large page
 *         size to optimize for compute use cases.
 *       LW0080_CTRL_GPU_SPARSE_TEXTURE_COMPUTE_MODE_OPTIMIZE_SPARSE_TEXTURE
 *         This value indicates that the RM should select the GPU's large page
 *         size to optimize for sparse texture use cases.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 */
#define LW0080_CTRL_CMD_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE (0x80028d) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS_MESSAGE_ID (0x8DU)

typedef struct LW0080_CTRL_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS {
    LwU32 setting;
} LW0080_CTRL_GPU_SET_SPARSE_TEXTURE_COMPUTE_MODE_PARAMS;

/* Possible sparse texture compute mode setting values */
#define LW0080_CTRL_GPU_SPARSE_TEXTURE_COMPUTE_MODE_DEFAULT                 0
#define LW0080_CTRL_GPU_SPARSE_TEXTURE_COMPUTE_MODE_OPTIMIZE_COMPUTE        1
#define LW0080_CTRL_GPU_SPARSE_TEXTURE_COMPUTE_MODE_OPTIMIZE_SPARSE_TEXTURE 2

/*
 * LW0080_CTRL_CMD_GPU_GET_VGX_CAPS
 *
 * This command gets the VGX capability of the GPU depending on the status of
 * the VGX hardware fuse.
 *
 *   isVgx
 *     This field is set to LW_TRUE is VGX fuse is enabled for the GPU otherwise
 *     it is set to LW_FALSE.
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_GPU_GET_VGX_CAPS                                    (0x80028e) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_VGX_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_VGX_CAPS_PARAMS_MESSAGE_ID (0x8EU)

typedef struct LW0080_CTRL_GPU_GET_VGX_CAPS_PARAMS {
    LwBool isVgx;
} LW0080_CTRL_GPU_GET_VGX_CAPS_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_SET_VIRTUALIZATION_MODE
 *
 * This command set a value indicating virtualization mode in
 * which the GPU is running. Virtualization mode of a GPU can
 * only be set when it is running in host hypervisor mode.
 *
 *   virtualizationMode
 *     This parameter set the virtualization mode of the device.
 *     Possible values are:
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST_VGPU
 *         This value indicates that the device is associated with vGPU((host GPU).
 *       LW0080_CTRL_GPU_VIRTUALIZATION_MODE_HOST_VSGA
 *         This value indicates that the device is associated with vSGA(host GPU).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0080_CTRL_CMD_GPU_SET_VIRTUALIZATION_MODE (0x80028f) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_SET_VIRTUALIZATION_MODE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_SET_VIRTUALIZATION_MODE_PARAMS_MESSAGE_ID (0x8FU)

typedef struct LW0080_CTRL_GPU_SET_VIRTUALIZATION_MODE_PARAMS {
    LwU32 virtualizationMode;
} LW0080_CTRL_GPU_SET_VIRTUALIZATION_MODE_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_VIRTUALIZATION_SWITCH_TO_VGA
 *
 * This command notifies the hypervisor to switch its "backdoor VNC" view back
 * to the console.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0080_CTRL_CMD_GPU_VIRTUALIZATION_SWITCH_TO_VGA (0x800290) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | 0x90" */

/*
 * LW0080_CTRL_CMD_GPU_GET_SRIOV_CAPS
 *
 * This command is used to query GPU SRIOV capabilities
 *   totalVFs
 *     Total number of virtual functions supported.
 *
 *   firstVfOffset
 *     Offset of the first VF.
 *
 *   vfFeatureMask
 *     Bitmask of features managed by the guest
 *
 *   FirstVFBar0Address
 *     Address of BAR0 region of first VF.
 *
 *   FirstVFBar1Address
 *     Address of BAR1 region of first VF.
 *
 *   FirstVFBar2Address
 *     Address of BAR2 region of first VF.
 *
 *   bar0Size
 *     Size of BAR0 region on VF.
 *
 *   bar1Size
 *     Size of BAR1 region on VF.
 *
 *   bar2Size
 *     Size of BAR2 region on VF.
 *
 *   b64bitBar0
 *     If the VF BAR0 is 64-bit addressable.
 *
 *   b64bitBar1
 *     If the VF BAR1 is 64-bit addressable.
 *
 *   b64bitBar2
 *     If the VF BAR2 is 64-bit addressable.
 *
 *   bSriovEnabled
 *     Flag for SR-IOV enabled or not.
 *
 *   bSriovHeavyEnabled
 *     Flag for whether SR-IOV is enabled in standard or heavy mode.
 *
 *   bEmulateVFBar0TlbIlwalidationRegister
 *     Flag for whether VF's TLB Ilwalidate Register region needs emulation.
 *
 *   bClientRmAllocatedCtxBuffer
 *     Flag for whether engine ctx buffer is managed by client RM.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_GPU_GET_SRIOV_CAPS               (0x800291) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS_MESSAGE_ID (0x91U)

typedef struct LW0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS {
    LwU32  totalVFs;
    LwU32  firstVfOffset;
    LwU32  vfFeatureMask;
    LW_DECLARE_ALIGNED(LwU64 FirstVFBar0Address, 8);
    LW_DECLARE_ALIGNED(LwU64 FirstVFBar1Address, 8);
    LW_DECLARE_ALIGNED(LwU64 FirstVFBar2Address, 8);
    LW_DECLARE_ALIGNED(LwU64 bar0Size, 8);
    LW_DECLARE_ALIGNED(LwU64 bar1Size, 8);
    LW_DECLARE_ALIGNED(LwU64 bar2Size, 8);
    LwBool b64bitBar0;
    LwBool b64bitBar1;
    LwBool b64bitBar2;
    LwBool bSriovEnabled;
    LwBool bSriovHeavyEnabled;
    LwBool bEmulateVFBar0TlbIlwalidationRegister;
    LwBool bClientRmAllocatedCtxBuffer;
} LW0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



// Update this macro if new HW exceeds GPU Classlist MAX_SIZE
#define LW0080_CTRL_GPU_CLASSLIST_MAX_SIZE   116

#define LW0080_CTRL_CMD_GPU_GET_CLASSLIST_V2 (0x800292) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS_MESSAGE_ID (0x92U)

typedef struct LW0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS {
    LwU32 numClasses;                                       // __OUT__
    LwU32 classList[LW0080_CTRL_GPU_CLASSLIST_MAX_SIZE];    // __OUT__
} LW0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS;

/*
 * LW0080_CTRL_CMD_GPU_FIND_SUBDEVICE_HANDLE
 *
 * Find a subdevice handle allocated under this device
 */
#define LW0080_CTRL_CMD_GPU_FIND_SUBDEVICE_HANDLE (0x800293) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_FIND_SUBDEVICE_HANDLE_PARAM_MESSAGE_ID" */

#define LW0080_CTRL_GPU_FIND_SUBDEVICE_HANDLE_PARAM_MESSAGE_ID (0x93U)

typedef struct LW0080_CTRL_GPU_FIND_SUBDEVICE_HANDLE_PARAM {
    LwU32    subDeviceInst;         // [in]
    LwHandle hSubDevice;            // [out]
} LW0080_CTRL_GPU_FIND_SUBDEVICE_HANDLE_PARAM;

/*
 * LW0080_CTRL_CMD_GPU_GET_BRAND_CAPS
 *
 * This command gets branding information for the device.
 *
 *   brands
 *     Mask containing branding information. A bit in this
 *     mask is set if the GPU has particular branding.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW0080_CTRL_GPU_GET_BRAND_CAPS_QUADRO    LWBIT(0)
#define LW0080_CTRL_GPU_GET_BRAND_CAPS_LWS       LWBIT(1)
#define LW0080_CTRL_GPU_GET_BRAND_CAPS_TITAN     LWBIT(2)

#define LW0080_CTRL_CMD_GPU_GET_BRAND_CAPS (0x800294) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_GET_BRAND_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_GET_BRAND_CAPS_PARAMS_MESSAGE_ID (0x94U)

typedef struct LW0080_CTRL_GPU_GET_BRAND_CAPS_PARAMS {
    LwU32 brands;
} LW0080_CTRL_GPU_GET_BRAND_CAPS_PARAMS;

/*
 * These are the per-VF BAR1 sizes that we support in MB.
 * They are used with the LW0080_CTRL_GPU_SET_VGPU_VF_BAR1_SIZE control call and
 * should match the LW_XVE_BAR1_CONFIG_SIZE register defines.
 */
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_64M       64
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_128M      128
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_256M      256
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_512M      512
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_1G        1024
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_2G        2048
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_4G        4096
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_8G        8192
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_16G       16384
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_32G       32768
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_64G       65536
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_128G      131072
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_MIN       LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_64M
#define LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_MAX       LW0080_CTRL_GPU_VGPU_VF_BAR1_SIZE_128G

#define LW0080_CTRL_GPU_VGPU_NUM_VFS_ILWALID        LW_U32_MAX

/*
 * LW0080_CTRL_GPU_SET_VGPU_VF_BAR1_SIZE
 *
 * @brief Resize BAR1 per-VF on the given GPU
 *   vfBar1SizeMB[in]   size of per-VF BAR1 size in MB
 *   numVfs[out]        number of VFs that can be created given the new BAR1 size
 */
#define LW0080_CTRL_GPU_SET_VGPU_VF_BAR1_SIZE (0x800296) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GPU_INTERFACE_ID << 8) | LW0080_CTRL_GPU_SET_VGPU_VF_BAR1_SIZE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GPU_SET_VGPU_VF_BAR1_SIZE_PARAMS_MESSAGE_ID (0x96U)

typedef struct LW0080_CTRL_GPU_SET_VGPU_VF_BAR1_SIZE_PARAMS {
    LwU32 vfBar1SizeMB;
    LwU32 numVfs;
} LW0080_CTRL_GPU_SET_VGPU_VF_BAR1_SIZE_PARAMS;

/* _ctrl0080gpu_h_ */
