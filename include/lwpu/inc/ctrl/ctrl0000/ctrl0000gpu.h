/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2019 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000gpu.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0000/ctrl0000base.h"
#include "ctrl/ctrl0000/ctrl0000system.h"
#include "ctrl/ctrlxxxx.h"
#include "lwlimits.h"

/* LW01_ROOT (client) GPU control commands and parameters */

/*
 * LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS
 *
 * This command returns a table of attached gpuId values.
 * The table is LW0000_CTRL_GPU_MAX_ATTACHED_GPUS entries in size.
 *
 *   gpuIds[]
 *     This parameter returns the table of attached GPU IDs.
 *     The GPU ID is an opaque platform-dependent value that can be used
 *     with the LW0000_CTRL_CMD_GPU_GET_ID_INFO command to retrieve
 *     additional information about the GPU.  The valid entries in gpuIds[]
 *     are contiguous, with a value of LW0000_CTRL_GPU_ILWALID_ID indicating
 *     the invalid entries.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS (0x201) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_MAX_ATTACHED_GPUS    32
#define LW0000_CTRL_GPU_ILWALID_ID           (0xffffffff)

#define LW0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS {
    LwU32 gpuIds[LW0000_CTRL_GPU_MAX_ATTACHED_GPUS];
} LW0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS;

/*
 * Deprecated. Please use LW0000_CTRL_CMD_GPU_GET_ID_INFO_V2 instead.
 */
#define LW0000_CTRL_CMD_GPU_GET_ID_INFO (0x202) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_ID_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_MAX_SZNAME      128

#define LW0000_CTRL_NO_NUMA_NODE        (-1)

#define LW0000_CTRL_GPU_GET_ID_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_GPU_GET_ID_INFO_PARAMS {
    LwU32 gpuId;
    LwU32 gpuFlags;
    LwU32 deviceInstance;
    LwU32 subDeviceInstance;
    LW_DECLARE_ALIGNED(LwP64 szName, 8);
    LwU32 sliStatus;
    LwU32 boardId;
    LwU32 gpuInstance;
    LwS32 numaId;
} LW0000_CTRL_GPU_GET_ID_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_ID_INFO_V2
 * This command returns GPU instance information for the specified GPU.
 *
 *   [in] gpuId
 *     This parameter should specify a valid GPU ID value.  If there
 *     is no GPU present with the specified ID, a status of
 *     LW_ERR_ILWALID_ARGUMENT is returned.
 *   [out] gpuFlags
 *     This parameter returns various flags values for the specified GPU.
 *     Valid flag values include:
 *       LW0000_CTRL_GPU_ID_INFO_IN_USE
 *         When true this flag indicates there are client references
 *         to the GPU in the form of device class instantiations (see
 *         LW01_DEVICE or LW03_DEVICE descriptions for details).
 *       LW0000_CTRL_GPU_ID_INFO_LINKED_INTO_SLI_DEVICE
 *         When true this flag indicates the GPU is linked into an
 *         active SLI device.
 *       LW0000_CTRL_GPU_ID_INFO_MOBILE
 *         When true this flag indicates the GPU is a mobile GPU.
 *       LW0000_CTRL_GPU_ID_BOOT_MASTER
 *         When true this flag indicates the GPU is the boot master GPU.
 *       LW0000_CTRL_GPU_ID_INFO_QUADRO
 *         When true this flag indicates the GPU is a Lwdqro GPU.
 *       LW0000_CTRL_GPU_ID_INFO_SOC
 *         When true this flag indicates the GPU is part of a
 *         System-on-Chip (SOC).
 *       LW0000_CTRL_GPU_ID_INFO_ATS_ENABLED
 *         When ATS is enabled on the system.
 *   [out] deviceInstance
 *     This parameter returns the broadcast device instance number associated
 *     with the specified GPU.  This value can be used to instantiate
 *     a broadcast reference to the GPU using the LW01_DEVICE classes.
 *   [out] subDeviceInstance
 *     This parameter returns the unicast subdevice instance number
 *     associated with the specified GPU.  This value can be used to
 *     instantiate a unicast reference to the GPU using the LW20_SUBDEVICE
 *     classes.
 *   [out] sliStatus
 *     This parameters returns the SLI status for the specified GPU.
 *     Legal values for this member are described by LW0000_CTRL_SLI_STATUS.
 *   [out] boardId
 *     This parameter returns the board ID value with which the
 *     specified GPU is associated.  Multiple GPUs can share the
 *     same board ID in multi-GPU configurations.
 *   [out] gpuInstance
 *     This parameter returns the GPU instance number for the specified GPU.
 *     GPU instance numbers are assigned in bus-probe order beginning with
 *     zero and are limited to one less the number of GPUs in the system.
 *   [out] numaId
 *     This parameter returns the ID of NUMA node for the specified GPU.
 *     In case there is no NUMA node, LW0000_CTRL_NO_NUMA_NODE is returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_GET_ID_INFO_V2 (0x205) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS {
    LwU32 gpuId;
    LwU32 gpuFlags;
    LwU32 deviceInstance;
    LwU32 subDeviceInstance;
    LwU32 sliStatus;
    LwU32 boardId;
    LwU32 gpuInstance;
    LwS32 numaId;
} LW0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS;


/* valid flags values */
#define LW0000_CTRL_GPU_ID_INFO_IN_USE                             0:0
#define LW0000_CTRL_GPU_ID_INFO_IN_USE_FALSE                 (0x00000000)
#define LW0000_CTRL_GPU_ID_INFO_IN_USE_TRUE                  (0x00000001)
#define LW0000_CTRL_GPU_ID_INFO_LINKED_INTO_SLI_DEVICE             1:1
#define LW0000_CTRL_GPU_ID_INFO_LINKED_INTO_SLI_DEVICE_FALSE (0x00000000)
#define LW0000_CTRL_GPU_ID_INFO_LINKED_INTO_SLI_DEVICE_TRUE  (0x00000001)
#define LW0000_CTRL_GPU_ID_INFO_MOBILE                             2:2
#define LW0000_CTRL_GPU_ID_INFO_MOBILE_FALSE                 (0x00000000)
#define LW0000_CTRL_GPU_ID_INFO_MOBILE_TRUE                  (0x00000001)
#define LW0000_CTRL_GPU_ID_INFO_BOOT_MASTER                        3:3
#define LW0000_CTRL_GPU_ID_INFO_BOOT_MASTER_FALSE            (0x00000000)
#define LW0000_CTRL_GPU_ID_INFO_BOOT_MASTER_TRUE             (0x00000001)
#define LW0000_CTRL_GPU_ID_INFO_QUADRO                             4:4
#define LW0000_CTRL_GPU_ID_INFO_QUADRO_FALSE                 (0x00000000)
#define LW0000_CTRL_GPU_ID_INFO_QUADRO_TRUE                  (0x00000001)
#define LW0000_CTRL_GPU_ID_INFO_SOC                                5:5
#define LW0000_CTRL_GPU_ID_INFO_SOC_FALSE                    (0x00000000)
#define LW0000_CTRL_GPU_ID_INFO_SOC_TRUE                     (0x00000001)
#define LW0000_CTRL_GPU_ID_INFO_ATS_ENABLED                        6:6
#define LW0000_CTRL_GPU_ID_INFO_ATS_ENABLED_FALSE            (0x00000000)
#define LW0000_CTRL_GPU_ID_INFO_ATS_ENABLED_TRUE             (0x00000001)

/*
 * LW0000_CTRL_CMD_GPU_GET_INIT_STATUS
 *
 * This command returns the initialization status for the specified GPU, and
 * will return LW_ERR_ILWALID_STATE if called prior to GPU
 * initialization.
 *
 *   gpuId
 *     This parameter should specify a valid GPU ID value.  If there
 *     is no GPU present with the specified ID, a status of
 *     LW_ERR_ILWALID_ARGUMENT is returned.
 *   status
 *     This parameter returns the status code identifying the initialization
 *     state of the GPU. If this parameter has the value LW_OK,
 *     then no errors were detected during GPU initialization. Otherwise, this
 *     parameter specifies the top-level error that was detected during GPU
 *     initialization. Note that a value of LW_OK only means that
 *     no errors were detected during the actual GPU initialization, and other
 *     errors may have oclwrred that prevent the GPU from being attached or
 *     accessible via the LW01_DEVICE or LW20_SUBDEVICE classes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW0000_CTRL_CMD_GPU_GET_INIT_STATUS                  (0x203) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_INIT_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_INIT_STATUS_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0000_CTRL_GPU_GET_INIT_STATUS_PARAMS {
    LwU32 gpuId;
    LwU32 status;
} LW0000_CTRL_GPU_GET_INIT_STATUS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_DEVICE_IDS
 *
 * This command returns a mask of valid device IDs.  These device IDs
 * can be used to instantiate the LW01_DEVICE_0 class (see LW01_DEVICE_0
 * for more information).
 *
 *   deviceIds
 *     This parameter returns the mask of valid device IDs.  Each enabled bit
 *     in the mask corresponds to a valid device instance.  Valid device
 *     instances can be used to initialize the LW0080_ALLOC_PARAMETERS
 *     structure when using LwRmAlloc to instantiate device handles.  The
 *     number of device IDs will not exceed LW_MAX_DEVICES in number.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0000_CTRL_CMD_GPU_GET_DEVICE_IDS (0x204) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_DEVICE_IDS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_DEVICE_IDS_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0000_CTRL_GPU_GET_DEVICE_IDS_PARAMS {
    LwU32 deviceIds;
} LW0000_CTRL_GPU_GET_DEVICE_IDS_PARAMS;

/*
 * LW0000_CTRL_GPU_SLI_CONFIG
 *
 * This structure describes a single SLI configuration.
 *
 *   sliInfo
 *     This parameter contains any flags associated with the configuration.
 *     Valid info values include:
 *       LW0000_CTRL_GPU_SLI_INFO_ACTIVE (out)
 *         When set to _TRUE this flag indicates that the configuration has
 *         been activated by a call to LW0000_CTRL_CMD_GPU_LINK_SLI_DEVICE.
 *         When set to _FALSE this flag indicates that the configuration
 *         is inactive.
 *       LW0000_CTRL_GPU_SLI_INFO_VIDLINK (out)
 *         When set to _PRESENT this flag indicates that the configuration
 *         has a video link (or bridge).  When set to _NOT_PRESENT this flag
 *         indicates the configuration does not have a video link.
 *       LW0000_CTRL_GPU_SLI_INFO_ENABLE_SLI_BY_DEFAULT (out)
 *         When set to _TRUE this flag indicates that the SLI configuration
 *         must be enabled by default.  When to set to _FALSE this flag
 *         indicates that the SLI configuration can be optionally enabled.
 *       LW0000_CTRL_GPU_SLI_INFO_MULTI_GPU (out)
 *         When set to _TRUE this flag indicates that this is a
 *         "Multi-GPU"-labeled configuration.
 *       LW0000_CTRL_GPU_SLI_INFO_GX2_BOARD (out)
 *         When set to _TRUE this flag indicates that the GPUs comprising
 *         this SLI configuration are Dagwoods.
 *       LW0000_CTRL_GPU_SLI_INFO_DYNAMIC_ALLOWED (out)
 *         When set to _TRUE this flag indicates that a Dynamic SLI
 *         transition is allowed.  When _FALSE a reboot is required.
 *       LW0000_CTRL_GPU_SLI_INFO_VIDLINK_CONNECTOR (out)
 *         When set to _PRESENT this flag indicates that the configuration
 *         has video link (or bridge) connectors on all GPUs.
 *         When set to _NOT_PRESENT this flag indicates the configuration does
 *         does not have video link connctors on all GPUs.
 *       LW0000_CTRL_GPU_SLI_INFO_BROADCAST - DEPRECATED (out)
 *         When set to _TRUE this flag indicates that broadcast mode is
 *         supported by this SLI configuration.  When set to _FALSE this flag
 *         indicates that broadcast mode is not supported by this SLI
 *         configuration.
 *       LW0000_CTRL_GPU_SLI_INFO_UNICAST - DEPRECATED (out)
 *         When set to _TRUE this flag indicates that unicast mode is
 *         supported by this SLI configuration.  When set to _FALSE this flag
 *         indicates that unicast mode is not supported by this SLI
 *         configuration.
 *       LW0000_CTRL_GPU_SLI_INFO_SLI_MOSAIC_ONLY (out)
 *         When set to _TRUE this flag indicates that only a
 *         Mosaic mode is supported by this configuration.
 *       LW0000_CTRL_GPU_SLI_INFO_4_WAY_SLI (out)
 *         When set to _TRUE this flag indicates that this is a
 *         "4-Way-SLI" labeled configuration - four discrete boards.
 *       LW0000_CTRL_GPU_SLI_INFO_BASE_MOSAIC_ONLY (out)
 *         When set to _TRUE this flag indicates that base mosaic is
 *         only supported.  That is, from an end-user point of view,
 *         SLI is not allowed.
 *       LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_MOSAIC (out)
 *         When set to _TRUE this flag indicates that in addition to
 *         SLI support, the topology supports SLI Mosaic.
 *       LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_BASE_MOSAIC (out)
 *         When set to _TRUE this flag indicates that in addition to
 *         SLI support, the topology supports SLI BaseMosaic.
 *       LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO (out)
 *         When set to _TRUE this flag indicates that Dual MIO has been
 *         activated in this GPU topology.
 *       LW0000_CTRL_GPU_SLI_INFO_GPUS_DUAL_MIO_CAPABLE (out)
 *         When set to _TRUE this flag indicates that all GPUs this topology
 *         support Dual MIO
 *       LW0000_CTRL_GPU_SLI_INFO_LWLINK (out)
 *         When set to _PRESENT this flag indicates that the configuration
 *         has Lwlink on all GPUs
 *         When set to _NOT_PRESENT this flag indicates the configuration
 *         does not have Lwlink on all GPUs.
 *       LW0000_CTRL_GPU_SLI_INFO_LWLINK_CONNECTOR (out)
 *         When set to _PRESENT this flag indicates that the configuration
 *         has Lwlink connectors on all GPUs
 *         When set to _NOT_PRESENT this flag indicates the configuration
 *         does not have Lwlink connctors on all GPUs.
 *       LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO_POSSIBLE  (out)
 *         When set to _TRUE this flag indicates that Dual Mio is possible
 *         with this topology.
 *       LW0000_CTRL_GPU_SLI_INFO_HIGH_SPEED_VIDLINK  (out)
 *         When set to _TRUE this flag indicates that all GPUs in this topology are
 *         connected with one or two high speed video bridges.
 *       LW0000_CTRL_GPU_SLI_INFO_WS_OVERRIDE (out)
 *         When set to _TRUE this flag indicates that the GPU is internal, for which
 *         the SLI rules are relaxed: do not check for Lwdqro/Mobile restrictions
 *         related to the platform.
 *       LW0000_CTRL_GPU_SLI_INFO_PREPASCAL (out)
 *         When set to _TRUE this flag indicates that the GPU family is Maxwell
 *         or below.
 *       LW0000_CTRL_GPU_SLI_INFO_DD_ALT_SET_VIDLINK (in)
 *         This flag is an input and can be set by the client.
 *         When set to _TRUE this flag indicates the client will call LW0080_CTRL_GPU_SET_VIDLINK
 *         before modeset. If this flag is enabled, RM tracks any additional states required to ensure
 *         MIO power is enabled during SV2 for all necessary conditions. Client would call
 *         LW0080_CTRL_GPU_SET_VIDLINK after modeset only for power saving tweaks.
 *       LW0000_CTRL_GPU_SLI_INFO_NO_BRIDGE_REQUIRED (out)
 *         When set to _TRUE this flag indicates that no bridge is required
 *
 *   displayGpuIndex
 *     This member contains the index into the gpuIds[] array pointing to
 *     the display GPU.
 *   gpuCount
 *     This member contains the number of GPU IDs comprising the SLI device
 *     and stored in the gpuIds[] array.
 *   gpuIds
 *     This member contains the array of GPU IDs comprising the SLI device.
 *     Valid entries are contiguous, beginning with the first entry in the
 *     list.  Invalid entries contain LW0000_CTRL_GPU_ILWALID_ID.
 *   masterGpuIndex
 *     This member contains the index into the gpuIds[] array pointing to
 *     the master GPU.
 *   noDisplayGpuMask
 *     This member contains an index mask into the gpuIds[] array pointing to
 *     the GPUs that cannot have active displays when SLI is enabled.
 */
typedef struct LW0000_CTRL_GPU_SLI_CONFIG {
    LwU32 sliInfo;
    LwU32 displayGpuIndex;
    LwU32 gpuCount;
    LwU32 gpuIds[LW_MAX_SUBDEVICES];
    LwU32 masterGpuIndex;
    LwU32 noDisplayGpuMask;
} LW0000_CTRL_GPU_SLI_CONFIG;

/* valid flags values */
#define LW0000_CTRL_GPU_SLI_INFO_ACTIVE                            0:0
#define LW0000_CTRL_GPU_SLI_INFO_ACTIVE_FALSE                     (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_ACTIVE_TRUE                      (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_VIDLINK                           1:1
#define LW0000_CTRL_GPU_SLI_INFO_VIDLINK_NOT_PRESENT              (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_VIDLINK_PRESENT                  (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_ENABLE_SLI_BY_DEFAULT             2:2
#define LW0000_CTRL_GPU_SLI_INFO_ENABLE_SLI_BY_DEFAULT_FALSE      (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_ENABLE_SLI_BY_DEFAULT_TRUE       (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_MULTI_GPU                         3:3
#define LW0000_CTRL_GPU_SLI_INFO_MULTI_GPU_FALSE                  (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_MULTI_GPU_TRUE                   (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_ENABLE_CORELOGIC_BROADCAST        4:4          // DEPRECATED
#define LW0000_CTRL_GPU_SLI_INFO_ENABLE_CORELOGIC_BROADCAST_FALSE (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_ENABLE_CORELOGIC_BROADCAST_TRUE  (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_IMPLICIT_SLI                      5:5
#define LW0000_CTRL_GPU_SLI_INFO_IMPLICIT_SLI_FALSE               (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_IMPLICIT_SLI_TRUE                (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_GX2_BOARD                         6:6
#define LW0000_CTRL_GPU_SLI_INFO_GX2_BOARD_FALSE                  (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_GX2_BOARD_TRUE                   (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_DYNAMIC_ALLOWED                   7:7
#define LW0000_CTRL_GPU_SLI_INFO_DYNAMIC_ALLOWED_FALSE            (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_DYNAMIC_ALLOWED_TRUE             (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_VIDLINK_CONNECTOR                 8:8
#define LW0000_CTRL_GPU_SLI_INFO_VIDLINK_CONNECTOR_NOT_PRESENT    (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_VIDLINK_CONNECTOR_PRESENT        (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_BROADCAST                          9:9         // DEPRECATED
#define LW0000_CTRL_GPU_SLI_INFO_BROADCAST_FALSE                  (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_BROADCAST_TRUE                   (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_UNICAST                           10:10        // DEPRECATED
#define LW0000_CTRL_GPU_SLI_INFO_UNICAST_FALSE                    (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_UNICAST_TRUE                     (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_SLI_MOSAIC_ONLY                   11:11
#define LW0000_CTRL_GPU_SLI_INFO_SLI_MOSAIC_ONLY_FALSE            (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_SLI_MOSAIC_ONLY_TRUE             (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_4_WAY_SLI                         12:12
#define LW0000_CTRL_GPU_SLI_INFO_4_WAY_SLI_FALSE                  (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_4_WAY_SLI_TRUE                   (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_BASE_MOSAIC_ONLY                  13:13
#define LW0000_CTRL_GPU_SLI_INFO_BASE_MOSAIC_ONLY_FALSE           (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_BASE_MOSAIC_ONLY_TRUE            (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_MOSAIC                  14:14
#define LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_MOSAIC_FALSE           (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_MOSAIC_TRUE            (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_BASE_MOSAIC             15:15
#define LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_BASE_MOSAIC_FALSE      (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_ALLOW_SLI_BASE_MOSAIC_TRUE       (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO                          16:16
#define LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO_FALSE                   (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO_TRUE                    (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_GPUS_DUAL_MIO_CAPABLE             17:17
#define LW0000_CTRL_GPU_SLI_INFO_GPUS_DUAL_MIO_CAPABLE_FALSE      (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_GPUS_DUAL_MIO_CAPABLE_TRUE       (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_LWLINK                            18:18
#define LW0000_CTRL_GPU_SLI_INFO_LWLINK_NOT_PRESENT               (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_LWLINK_PRESENT                   (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_LWLINK_CONNECTOR                  19:19
#define LW0000_CTRL_GPU_SLI_INFO_LWLINK_CONNECTOR_NOT_PRESENT     (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_LWLINK_CONNECTOR_PRESENT         (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO_POSSIBLE                 21:21
#define LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO_POSSIBLE_FALSE          (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_DUAL_MIO_POSSIBLE_TRUE           (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_HIGH_SPEED_VIDLINK                22:22
#define LW0000_CTRL_GPU_SLI_INFO_HIGH_SPEED_VIDLINK_FALSE         (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_HIGH_SPEED_VIDLINK_TRUE          (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_WS_OVERRIDE                       23:23
#define LW0000_CTRL_GPU_SLI_INFO_WS_OVERRIDE_FALSE                (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_WS_OVERRIDE_TRUE                 (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_NO_BRIDGE_REQUIRED                24:24
#define LW0000_CTRL_GPU_SLI_INFO_NO_BRIDGE_REQUIRED_FALSE         (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_NO_BRIDGE_REQUIRED_TRUE          (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_PREPASCAL                         25:25
#define LW0000_CTRL_GPU_SLI_INFO_PREPASCAL_FALSE                  (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_PREPASCAL_TRUE                   (0x00000001)
#define LW0000_CTRL_GPU_SLI_INFO_DD_ALT_SET_VIDLINK                26:26
#define LW0000_CTRL_GPU_SLI_INFO_DD_ALT_SET_VIDLINK_FALSE         (0x00000000)
#define LW0000_CTRL_GPU_SLI_INFO_DD_ALT_SET_VIDLINK_TRUE          (0x00000001)


/*
 * Deprecated. Please use LW0000_CTRL_CMD_GPU_GET_VALID_SLI_CONFIGS_V2 instead.
 */
#define LW0000_CTRL_CMD_GPU_GET_VALID_SLI_CONFIGS                 (0x210) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_PARAMS {
    LwU32 sliStatus;
    LwU32 sliConfigCount;
    LW_DECLARE_ALIGNED(LwP64 sliConfigList, 8);
    LwU8  bRecheckSliCookie;
} LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_VALID_SLI_CONFIGS_V2
 *
 * This command returns the number and specification of valid SLI
 * configurations.  A valid SLI configuration is comprised of two or more GPUs,
 * and can be used to create an SLI device using the
 * LW0000_CTRL_CMD_LINK_SLI_DEVICE command.  The list of entries returned by
 * this command includes all possible SLI configurations, including those that
 * describe active SLI devices.
 *
 *   [out] sliStatus
 *     This parameter returns the system-wide SLI status.  If sliConfigCount
 *     contains 0, this parameter will indicate the reason(s) that
 *     no valid configurations are available.
 *   [out] sliConfigCount
 *     This parameter always returns the total number of valid SLI device
 *     configurations available in the system.  A value of 0 indicates
 *     that there are no valid configurations available.
 *   [out] sliConfigList
 *     RM will copy sliConfigCount entries each describing a configuration into
 *     this buffer.  The fixed size of this array should be able to hold the
 *     maximum possible number of configs, per bug 2010268.
 *   [in] bRecheckSliCookie
 *     Recheck the SLI cookie in case it has not been retrieved correctly
 *     by the driver due to OS timing problems at boot on Windows only.
 *     This flag is not needed on other oses than Windows.
 *     It applies only once, and is set only from the LWAPI interface.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_GET_VALID_SLI_CONFIGS_V2 (0x21d) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_V2_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SLI_CONFIG_LIST_SIZE             256

#define LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_V2_PARAMS_MESSAGE_ID (0x1DU)

typedef struct LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_V2_PARAMS {
    LwU32                      sliStatus;
    LwU32                      startingOffset;
    LwU32                      sliConfigCount;
    LW0000_CTRL_GPU_SLI_CONFIG sliConfigList[LW0000_CTRL_SLI_CONFIG_LIST_SIZE];
    LwU8                       bRecheckSliCookie;
} LW0000_CTRL_GPU_GET_VALID_SLI_CONFIGS_V2_PARAMS;

/* sliStatus values */
#define LW0000_CTRL_SLI_STATUS_OK                         (0x00000000)
#define LW0000_CTRL_SLI_STATUS_ILWALID_GPU_COUNT          (0x00000001)
#define LW0000_CTRL_SLI_STATUS_OS_NOT_SUPPORTED           (0x00000002)
#define LW0000_CTRL_SLI_STATUS_OS_ERROR                   (0x00000004)
#define LW0000_CTRL_SLI_STATUS_NO_VIDLINK                 (0x00000008)
#define LW0000_CTRL_SLI_STATUS_INSUFFICIENT_LINK_WIDTH    (0x00000010)
#define LW0000_CTRL_SLI_STATUS_CPU_NOT_SUPPORTED          (0x00000020)
#define LW0000_CTRL_SLI_STATUS_GPU_NOT_SUPPORTED          (0x00000040)
#define LW0000_CTRL_SLI_STATUS_BUS_NOT_SUPPORTED          (0x00000080)
#define LW0000_CTRL_SLI_STATUS_CHIPSET_NOT_SUPPORTED      (0x00000100)
#define LW0000_CTRL_SLI_STATUS_GPU_MISMATCH               (0x00000400)
#define LW0000_CTRL_SLI_STATUS_ARCH_MISMATCH              (0x00000800)
#define LW0000_CTRL_SLI_STATUS_IMPL_MISMATCH              (0x00001000)
#define LW0000_CTRL_SLI_STATUS_SLI_WITH_TCC_NOT_SUPPORTED (0x00002000)
#define LW0000_CTRL_SLI_STATUS_PCI_ID_MISMATCH            (0x00004000)
#define LW0000_CTRL_SLI_STATUS_FB_MISMATCH                (0x00008000)
#define LW0000_CTRL_SLI_STATUS_VBIOS_MISMATCH             (0x00010000)
#define LW0000_CTRL_SLI_STATUS_QUADRO_MISMATCH            (0x00020000)
#define LW0000_CTRL_SLI_STATUS_BUS_TOPOLOGY_ERROR         (0x00040000)
#define LW0000_CTRL_SLI_STATUS_CONFIGSPACE_ACCESS_ERROR   (0x00080000)
#define LW0000_CTRL_SLI_STATUS_INCONSISTENT_CONFIG_SPACE  (0x00100000)
#define LW0000_CTRL_SLI_STATUS_CONFIG_NOT_SUPPORTED       (0x00200000)
#define LW0000_CTRL_SLI_STATUS_RM_NOT_SUPPORTED           (0x00400000)
#define LW0000_CTRL_SLI_STATUS_GPU_DRAINING               (0x00800000)
#define LW0000_CTRL_SLI_STATUS_MOBILE_MISMATCH            (0x01000000)
#define LW0000_CTRL_SLI_STATUS_ECC_MISMATCH               (0x02000000)
#define LW0000_CTRL_SLI_STATUS_INSUFFICIENT_FB            (0x04000000)
#define LW0000_CTRL_SLI_STATUS_SLI_COOKIE_NOT_PRESENT     (0x08000000)
#define LW0000_CTRL_SLI_STATUS_SLI_FINGER_NOT_SUPPORTED   (0x10000000)
#define LW0000_CTRL_SLI_STATUS_SLI_WITH_ECC_NOT_SUPPORTED (0x20000000)
#define LW0000_CTRL_SLI_STATUS_GR_MISMATCH                (0x40000000)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0000_CTRL_CMD_GPU_GET_SLI_CONFIG_INFO
 *
 * This command returns SLI configuration state for the specified set of
 * GPU IDs.  SLI configuration state includes the LW0000_CTRL_SLI_STATUS
 * value for the set of GPU IDs.  An sliStatus value of
 * LW0000_CTRL_SLI_STATUS_OK indicates the set of GPU IDs represents a
 * valid SLI configuration.  When ilwoked on a valid SLI configuration
 * this command returns additional SLI configuration state, including
 * the LW0000_CTRL_SLI_INFO value and display ownership settings, in the
 * associated LW0000_CTRL_GPU_SLI_CONFIG structure.
 *
 *   sliConfig
 *     This member specifies the LW0000_CTRL_GPU_SLI_CONFIG structure
 *     describing the set of GPU IDs for which SLI information
 *     should be obtained.  The gpuCount and gpuIds members should be
 *     initialized to represent the desired set of GPUs.  The remaining
 *     fields will be used to return the current settings for the
 *     corresponding SLI configuration.
 *   sliStatus
 *     This member contains the SLI status for the specified SLI configuration.
 *     Legal values for this member are described by LW0000_CTRL_SLI_STATUS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_GET_SLI_CONFIG_INFO           (0x211) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_SLI_CONFIG_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_SLI_CONFIG_INFO_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW0000_CTRL_GPU_GET_SLI_CONFIG_INFO_PARAMS {
    LW0000_CTRL_GPU_SLI_CONFIG sliConfig;
    LwU32                      sliStatus;
} LW0000_CTRL_GPU_GET_SLI_CONFIG_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_LINK_SLI_DEVICE
 *
 * This command enables a new SLI device by linking the specified set of
 * GPUs contained in the SLI device description.  An SLI device is a
 * device that contains two or more GPUs.
 *
 *   deviceInstance
 *     This parameter will contain the device instance number assigned
 *     by the RM to the new SLI device.  This device instance number
 *     can be used to allocate instances of the device (see the description
 *     of the LW01_DEVICE_0 class for more details).
 *   sliConfig
 *     This parameter specifies the description of the new SLI device the
 *     RM is to create by linking the specified GPUs.  The contents of
 *     this parameter typically contain information retrieved with the
 *     LW0000_CTRL_CMD_GPU_GET_SLI_DEVICES command.  Each of the GPUs
 *     contained in the configuration must not be part of an existing
 *     SLI device.  An attempt to link a GPU into more than one SLI
 *     device will fail with a STATE_IN_USE error.
 *
 * After a link operation the device and subdevice tables maintained
 * by the RM will have changed.  Clients need to refresh their GPU
 * topology information and recreate handles to these resources to ensure
 * correct operation.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 */
#define LW0000_CTRL_CMD_GPU_LINK_SLI_DEVICE (0x212) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_LINK_SLI_DEVICE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_LINK_SLI_DEVICE_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW0000_CTRL_GPU_LINK_SLI_DEVICE_PARAMS {
    LwU32                      deviceInstance;
    LW0000_CTRL_GPU_SLI_CONFIG sliConfig;
} LW0000_CTRL_GPU_LINK_SLI_DEVICE_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_UNLINK_SLI_DEVICE
 *
 * This command disables the specified SLI device by unlinking the
 * associated GPUs.  If the specified SLI device includes only a single
 * GPU then no action is taken and the command returns a successful status.
 *
 *   deviceInstance
 *     This parameter specifies the the device instance of the SLI device
 *     to unlink.
 *
 * After an unlink operation the device and subdevice tables maintained
 * by the RM will have changed.  Clients need to refresh their GPU
 * topology information and recreate handles to these resources to ensure
 * correct operation.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_UNLINK_SLI_DEVICE (0x213) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_UNLINK_SLI_DEVICE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_UNLINK_SLI_DEVICE_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW0000_CTRL_GPU_UNLINK_SLI_DEVICE_PARAMS {
    LwU32 deviceInstance;
} LW0000_CTRL_GPU_UNLINK_SLI_DEVICE_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0000_CTRL_CMD_GPU_GET_PROBED_IDS
 *
 * This command returns a table of probed gpuId values.
 * The table is LW0000_CTRL_GPU_MAX_PROBED_GPUS entries in size.
 *
 *   gpuIds[]
 *     This parameter returns the table of probed GPU IDs.
 *     The GPU ID is an opaque platform-dependent value that can
 *     be used with the LW0000_CTRL_CMD_GPU_ATTACH_IDS and
 *     LW0000_CTRL_CMD_GPU_DETACH_ID commands to attach and detach
 *     the GPU.
 *     The valid entries in gpuIds[] are contiguous, with a value
 *     of LW0000_CTRL_GPU_ILWALID_ID indicating the invalid entries.
 *   excludedGpuIds[]
 *     This parameter returns the table of excluded GPU IDs.
 *     An excluded GPU ID is an opaque platform-dependent value that
 *     can be used with LW0000_CTRL_CMD_GPU_GET_PCI_INFO and
 *     LW0000_CTRL_CMD_GPU_GET_UUID_INFO.
 *     The valid entries in excludedGpuIds[] are contiguous, with a value
 *     of LW0000_CTRL_GPU_ILWALID_ID indicating the invalid entries.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0000_CTRL_CMD_GPU_GET_PROBED_IDS (0x214) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_MAX_PROBED_GPUS    LW_MAX_DEVICES

#define LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS {
    LwU32 gpuIds[LW0000_CTRL_GPU_MAX_PROBED_GPUS];
    LwU32 excludedGpuIds[LW0000_CTRL_GPU_MAX_PROBED_GPUS];
} LW0000_CTRL_GPU_GET_PROBED_IDS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_PCI_INFO
 *
 * This command takes a gpuId and returns PCI bus information about
 * the device. If the OS does not support returning PCI bus
 * information, this call will return LW_ERR_NOT_SUPPORTED
 *
 *   gpuId
 *     This parameter should specify a valid GPU ID value.  If there
 *     is no GPU present with the specified ID, a status of
 *     LW_ERR_ILWALID_ARGUMENT is returned.
 *
 *   domain
 *     This parameter returns the PCI domain of the GPU.
 *
 *   bus
 *     This parameter returns the PCI bus of the GPU.
 *
 *   slot
 *     This parameter returns the PCI slot of the GPU.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_GET_PCI_INFO (0x21b) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS_MESSAGE_ID (0x1BU)

typedef struct LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS {
    LwU32 gpuId;
    LwU32 domain;
    LwU16 bus;
    LwU16 slot;
} LW0000_CTRL_GPU_GET_PCI_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_ATTACH_IDS
 *
 * This command attaches the GPUs with the gpuIds matching those in
 * the table provided by the client.
 * The table is LW0000_CTRL_GPU_MAX_PROBED_GPUS entries in size.
 *
 *   gpuIds[]
 *     This parameter holds the table of gpuIds to attach. At least
 *     one gpuId must be specified; clients may use the special
 *     gpuId value LW0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS to indicate
 *     that all probed GPUs are to be attached.
 *     The entries in gpuIds[] must be contiguous, with a value of
 *     LW0000_CTRL_GPU_ILWALID_ID to indicate the first invalid
 *     entry.
 *     If one or more of the gpuId values do not specify a GPU found
 *     in the system, the LW_ERR_ILWALID_ARGUMENT error
 *     status is returned.
 *
 *   failedId
 *     If LW0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS is specified and
 *     a GPU cannot be attached, the LW0000_CTRL_CMD_GPU_ATTACH_IDS
 *     command returns an error code and saves the failing GPU's
 *     gpuId in this field.
 *
 * If a table of gpuIds is provided, these gpuIds will be validated
 * against the RM's table of probed gpuIds and attached in turn,
 * if valid; if LW0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS is used, all
 * probed gpuIds will be attached, in the order the associated GPUs
 * were probed in by the RM.
 *
 * If a gpuId fails to attach, this gpuId is stored in the failedId
 * field. Any GPUs attached by the command prior the failure are
 * detached.
 *
 * If multiple clients use LW0000_CTRL_CMD_GPU_ATTACH_IDS to attach
 * a gpuId, the RM ensures that the gpuId won't be detached until
 * all clients have issued a call to LW0000_CTRL_CMD_GPU_DETACH_IDS
 * to detach the gpuId (or have terminated).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OPERATING_SYSTEM
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_IRQ_EDGE_TRIGGERED
 *   LW_ERR_IRQ_NOT_FIRING
 */
#define LW0000_CTRL_CMD_GPU_ATTACH_IDS        (0x215) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_ATTACH_IDS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS (0x0000ffff)

#define LW0000_CTRL_GPU_ATTACH_IDS_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW0000_CTRL_GPU_ATTACH_IDS_PARAMS {
    LwU32 gpuIds[LW0000_CTRL_GPU_MAX_PROBED_GPUS];
    LwU32 failedId;
} LW0000_CTRL_GPU_ATTACH_IDS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_DETACH_IDS
 *
 * This command detaches the GPUs with the gpuIds matching those in
 * the table provided by the client.
 * The table is LW0000_CTRL_GPU_MAX_ATTACHED_GPUS entries in size.
 *
 *   gpuIds[]
 *     This parameter holds the table of gpuIds to detach. At least
 *     one gpuId must be specified; clients may use the special
 *     gpuId LW0000_CTRL_GPU_DETACH_ALL_ATTACHED_IDS to indicate that
 *     all attached GPUs are to be detached.
 *     The entries in gpuIds[] must be contiguous, with a value of
 *     LW0000_CTRL_GPU_ILWALID_ID to indicate the first invalid
 *     entry.
 *     If one or more of the gpuId values do not specify a GPU found
 *     in the system, the LW_ERR_ILWALID_ARGUMENT error
 *     status is returned.
 *
 * If a table of gpuIds is provided, these gpuIds will be validated
 * against the RM's list of attached gpuIds; each valid gpuId is
 * detached immediately if it's no longer in use (i.e. if there are
 * no client references to the associated GPU in the form of
 * device class instantiations (see the LW01_DEVICE or LW03_DEVICE
 * descriptions for details)) and if no other client still requires
 * the associated GPU to be attached.
 *
 * If a given gpuId can't be detached immediately, it will instead
 * be detached when the last client reference is freed or when
 * the last client that issued LW0000_CTRL_CMD_GPU_ATTACH_IDS for
 * this gpuId either issues LW0000_CTRL_CMD_GPU_DETACH_IDS or exits
 * without detaching the gpuId explicitly.
 *
 * Clients may use the LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS command
 * to obtain a table of the attached gpuIds.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OPERATING_SYSTEM
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_DETACH_IDS          (0x216) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_DETACH_IDS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_DETACH_ALL_ATTACHED_IDS (0x0000ffff)

#define LW0000_CTRL_GPU_DETACH_IDS_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW0000_CTRL_GPU_DETACH_IDS_PARAMS {
    LwU32 gpuIds[LW0000_CTRL_GPU_MAX_ATTACHED_GPUS];
} LW0000_CTRL_GPU_DETACH_IDS_PARAMS;


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * Deprecated. Please use LW0000_CTRL_CMD_GPU_GET_ILWALID_SLI_CONFIGS_V2 instead.
 */
#define LW0000_CTRL_CMD_GPU_GET_ILWALID_SLI_CONFIGS (0x217) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_PARAMS_MESSAGE_ID (0x17U)

typedef struct LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_PARAMS {
    LwU32 sliConfigCount;
    LW_DECLARE_ALIGNED(LwP64 sliConfigList, 8);
    LwU8  bRecheckSliCookie;
} LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_ILWALID_SLI_CONFIGS_V2
 *
 * This command returns invalid SLI configurations.  An invalid SLI
 * configuration is comprised of two or more GPUs that can not be
 * linked into an SLI device using the LW0000_CTRL_CMD_LINK_SLI_DEVICE
 * command due to incompatibilities, SLI policy violations or other
 * problems.  The list of entries returned by this command includes
 * all invalid SLI configurations.
 *
 *   [out] sliConfigCount
 *     This parameter always returns the total number of invalid SLI device
 *     configurations found in the system.  A value of 0 indicates
 *     that there are no invalid configurations.
 *   [out] sliConfigList
 *     RM will copy sliConfigCount entries each detailing an invalid
 *     configuration into this array, and set sliConfigCount accordingly.
 *   [in] bRecheckSliCookie
 *     Recheck the SLI cookie in case it has not been retrieved correctly
 *     by the driver due to OS timing problems at boot on Windows only.
 *     This flag is not needed on other oses than Windows.
 *     It applies only once, and is set only from the LWAPI interface.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_GET_ILWALID_SLI_CONFIGS_V2 (0x21c) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_V2_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_V2_PARAMS_MESSAGE_ID (0x1LW)

typedef struct LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_V2_PARAMS {
    LwU32                      startingOffset;
    LwU32                      sliConfigCount;
    LW0000_CTRL_GPU_SLI_CONFIG sliConfigList[LW0000_CTRL_SLI_CONFIG_LIST_SIZE];
    LwU8                       bRecheckSliCookie;
} LW0000_CTRL_GPU_GET_ILWALID_SLI_CONFIGS_V2_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_SLI_SYSTEM_PROPERTIES
 *
 * This command returns the system properties mask that the SLI approval
 * mechanism uses to make its decision.
 *
 *   gpuCount
 *     This member specifies the number of GPUs defined in the gpuIds[] array.
 *   gpuIds
 *     This member specifies an array of GPU IDs that are in a valid
 *     or invalid SLI config.
 *   sliSystemPropertiesMask
 *     This member returns the the system properties mask the SLI approval
 *     mechanism uses to make its decision.  Valid sliSystemPropertiesMask
 *     values include:
 *       LW0000_CTRL_SLI_SYSPROP_IS_MOBILE
 *         This is a mobile configiguration.
 *       LW0000_CTRL_SLI_SYSPROP_IS_E551
 *         This is an E551 configuration (deprecated).
 *       LW0000_CTRL_SLI_SYSPROP_IS_BR02
 *         This configuration has a BR02.
 *       LW0000_CTRL_SLI_SYSPROP_IS_BR03
 *         Each GPU in the configuration is behind a BR03.
 *       LW0000_CTRL_SLI_SYSPROP_IS_BR04
 *         Each GPU in the configuration is behind a BR04.
 *       LW0000_CTRL_SLI_SYSPROP_IS_BR_ANY
 *         This configuration config has zero or more BR03s or BR04s.
 *       LW0000_CTRL_SLI_SYSPROP_IS_CPU_XEON
 *         The system has a Xeon CPU.
 *       LW0000_CTRL_SLI_SYSPROP_IS_QUADRO
 *         The configuration is Lwdqro only.
 *       LW0000_CTRL_SLI_SYSPROP_IS_OS_WDDM
 *         The operating system is WDDM based (Vista, Win7, Win8 Win10...).
 *       LW0000_CTRL_SLI_SYSPROP_IS_2BR03
 *         The configuration has a minimum of two cascaded BR03s behind
 *         each GPU.
 *       LW0000_CTRL_SLI_SYSPROP_IS_3BR03
 *         The configuration has minimum of three cascaded BR03s behind
 *         each GPU.
 *       LW0000_CTRL_SLI_SYSPROP_IS_OS_XP
 *         The operating system is XP.
 *       LW0000_CTRL_SLI_SYSPROP_IS_VIDEO_BRIDGE
 *         There is a video bridge present in the configuration, cirlwlar or not
 *       LW0000_CTRL_SLI_SYSPROP_IS_LWLINK
 *         There is a LwLink present in the configuration
 *       LW0000_CTRL_SLI_SYSPROP_IS_OS_UNIX
 *         The operating system is Unix.
 *       LW0000_CTRL_SLI_SYSPROP_IS_GEFORCE
 *         The configuration is VdChip only.
 *       LW0000_CTRL_SLI_SYSPROP_IS_OS_MODS
 *         The operating system is MODS.
 *       LW0000_CTRL_SLI_SYSPROP_IS_COMMON_BR03
 *         All GPUs in the configuration have a common BR03.
 *       LW0000_CTRL_SLI_SYSPROP_IS_COMMON_BR04
 *         All GPUs in the configuration have a common BR04.
 *       LW0000_CTRL_SLI_SYSPROP_IS_SHARED_BR03
 *         Each GPU in the configuration shares a BR03 with another GPU.
 *       LW0000_CTRL_SLI_SYSPROP_IS_SHARED_BR04
 *         Each GPU in the configuration shares a BR04 with another GPU.
 *       LW0000_CTRL_SLI_SYSPROP_IS_2BR04
 *         There is a minimum of two cascaded BR04s behind each GPU in the
 *         configuration.
 *       LW0000_CTRL_SLI_SYSPROP_IS_BR04_REV_A03
 *         There is a BR04 A03 in the system.
 *       LW0000_CTRL_SLI_SYSPROP_IS_MXM_INTERPOSER
 *         All GPUS in the configuration are behind an MXM interposer card.
 *       LW0000_CTRL_SLI_SYSPROP_IS_NO_BR_3RDPARTY
 *         All non-root port bridges in the system other than those
 *         contained in dagwoods are supported.
 *       LW0000_CTRL_SLI_SYSPROP_IS_SLI_APPROVAL_COOKIE
 *         The SBIOS has an SLI approval cookie.
 *       LW0000_CTRL_SLI_SYSPROP_IS_BR04_PRESENT
 *         There is a BR04 present in the system. The GPUs in the
 *         configuration are not necessarily behind a BR04.
 *       LW0000_CTRL_SLI_SYSPROP_IS_2BR04_NOT_CASCADED
 *         There is a minimum of two non-cascaded BR04s in the system.
 *       LW0000_CTRL_SLI_SYSPROP_IS_4_WAY_SLI_APPROVAL_COOKIE
 *         The SBIOS has an SLI approval cookie enabling 4-way SLI.
 *       LW0000_CTRL_SLI_SYSPROP_IS_TEMPLATE_APPROVAL_COOKIE
 *         The SBIOS has an template approval cookie. It does not mean it is valid.
 *       LW0000_CTRL_SLI_SYSPROP_IS_CIRLWLAR_VIDEO_BRIDGE
 *         There is a cirlwlar video bridge present in the configuration.
 *       LW0000_CTRL_SLI_SYSPROP_IS_ALLOW_GEFORCE_ON_WORKSTATION
 *         This system allows Vdchip SLI on workstation systems.
 *       LW0000_CTRL_SLI_SYSPROP_IS_GPU_DOES_NOT_SUPPORT_SLI
 *         One or more GPUs in the configuration does not support SLI
 *       LW0000_CTRL_SLI_SYSPROP_IS_GX2
 *         Each GPU in the configuration is a GX2.
 *       LW0000_CTRL_SLI_SYSPROP_IS_GPU_SUPPORTS_BASE_MOSAIC
 *         Each GPU in the configuration supports base mosaic mode.
 *       LW0000_CTRL_SLI_SYSPROP_IS_NO_ONBOARD_BR04
 *         BR04 device is not present.
 *       LW0000_CTRL_SLI_SYSPROP_IS_NO_UNSUPPORTED_PCI_BRIDGE
 *         All GPU's are present on supported bridge.
 *       LW0000_CTRL_SLI_SYSPROP_IS_PLX
 *         Each GPU in the gpu array is behind a PLX.
 *       LW0000_CTRL_SLI_SYSPROP_IS_COMMON_PLX
 *         All GPUs in the gpu array have a common PLX.
 *       LW0000_CTRL_SLI_SYSPROP_IS_SHARED_PLX
 *         Each GPU in the gpu array shares a PLX with another GPU.
 *       LW0000_CTRL_SLI_SYSPROP_IS_QSYNC
 *         Each GPU in the gpu array is connected to Lwdqro Sync card.
 *       LW0000_CTRL_SLI_SYSPROP_IS_LWLINK
 *         Each GPU in the gpu array is connected via LwLink
 *       LW0000_CTRL_SLI_SYSPROP_IS_WS_OVERRIDE
 *         Each GPU in the gpu array has a VBIOS indicating we should ignore
 *         the VDChip/Lwdqro/Mobile attributes when deteriimning SLI Approval.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_GET_SLI_SYSTEM_PROPERTIES           (0x218) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_SLI_SYSTEM_PROPERTIES_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_SLI_SYSPROP_IS_MOBILE                       (0x0000000000000001ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_E551                         (0x0000000000000004ULL) // Deprecated
#define LW0000_CTRL_SLI_SYSPROP_IS_BR02                         (0x0000000000000008ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_BR03                         (0x0000000000000010ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_BR04                         (0x0000000000000020ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_BR_ANY                       (0x0000000000000040ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_CPU_XEON                     (0x0000000000000080ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_QUADRO                       (0x0000000000000100ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_OS_VISTA                     (0x0000000000000200ULL) // To be removed
#define LW0000_CTRL_SLI_SYSPROP_IS_OS_WDDM                      (0x0000000000000200ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_2BR03                        (0x0000000000000400ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_3BR03                        (0x0000000000000800ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_OS_XP                        (0x0000000000001000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_VIDEO_BRIDGE                 (0x0000000000002000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_OS_UNIX                      (0x0000000000004000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_GEFORCE                      (0x0000000000008000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_OS_MODS                      (0x0000000000040000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_COMMON_BR03                  (0x0000000000080000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_COMMON_BR04                  (0x0000000000100000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_SHARED_BR03                  (0x0000000000200000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_SHARED_BR04                  (0x0000000000400000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_2BR04                        (0x0000000000800000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_BR04_REV_A03                 (0x0000000001000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_MXM_INTERPOSER               (0x0000000002000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_NO_BR_3RDPARTY               (0x0000000004000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_SLI_APPROVAL_COOKIE          (0x0000000008000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_BR04_PRESENT                 (0x0000000010000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_2BR04_NOT_CASCADED           (0x0000000020000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_4_WAY_SLI_APPROVAL_COOKIE    (0x0000000040000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_TEMPLATE_APPROVAL_COOKIE     (0x0000000080000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_CIRLWLAR_VIDEO_BRIDGE        (0x0000000100000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_ALLOW_GEFORCE_ON_WORKSTATION (0x0000000200000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_GPU_DOES_NOT_SUPPORT_SLI     (0x0000000400000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_GX2                          (0x0000000800000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_GPU_SUPPORTS_BASE_MOSAIC     (0x0000001000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_NO_VIDEO_BRIDGE              (0x0000002000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_P2P_WRITE_ALLOWED            (0x0000004000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_P2P_READ_ALLOWED             (0x0000008000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_NO_ONBOARD_BR04              (0x0000010000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_NO_UNSUPPORTED_PCI_BRIDGE    (0x0000020000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_PLX                          (0x0000040000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_COMMON_PLX                   (0x0000080000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_SHARED_PLX                   (0x0000100000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_SLI_APPROVAL_COOKIE_RECHECK  (0x0000200000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_QSYNC                        (0x0000400000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_LWLINK                       (0x0000800000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_WS_OVERRIDE                  (0x0001000000000000ULL)
#define LW0000_CTRL_SLI_SYSPROP_IS_PREPASCAL                    (0x0004000000000000ULL)

// This name is deprecated and should NOT be used in new applications
#define LW0000_CTRL_SLI_SYSPROP_IS_GPU_GT200_AND_ABOVE          LW0000_CTRL_SLI_SYSPROP_IS_GPU_SUPPORTS_BASE_MOSAIC

#define LW0000_CTRL_GPU_GET_SLI_SYSTEM_PROPERTIES_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW0000_CTRL_GPU_GET_SLI_SYSTEM_PROPERTIES_PARAMS {
    LwU32 gpuCount;
    LwU32 gpuIds[LW0000_CTRL_GPU_MAX_ATTACHED_GPUS];
    LW_DECLARE_ALIGNED(LwU64 sliSystemPropertiesMask, 8);
} LW0000_CTRL_GPU_GET_SLI_SYSTEM_PROPERTIES_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_VIDEO_LINKS
 *
 * This command returns information about video bridge connections
 * detected between GPUs in the system, organized as a table
 * with one row per attached GPU and none, one or more peer GPUs
 * listed in the columns of each row, if connected to the row head
 * GPU via a video bridge.
 *
 *   gpuId
 *     For each row, this field holds the GPU ID of the GPU
 *     whose connections are listed in the row.
 *
 *   connectedGpuIds
 *     For each row, this table holds the GPU IDs of the
 *     GPUs connected to the GPU identified via the 'gpuId'
 *     field.
 *
 *   links
 *     This table holds information about the video bridges
 *     connected between GPUs in the system.  Each row
 *     represents connections to a single GPU.
 *
 * Please note: the table only reports video links between already
 * attached GPUs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_GPU_GET_VIDEO_LINKS (0x219) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_VIDEO_LINKS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_MAX_VIDEO_LINKS     8

typedef struct LW0000_CTRL_GPU_VIDEO_LINKS {
    LwU32 gpuId;
    LwU32 connectedGpuIds[LW0000_CTRL_GPU_MAX_VIDEO_LINKS];
} LW0000_CTRL_GPU_VIDEO_LINKS;

#define LW0000_CTRL_GPU_GET_VIDEO_LINKS_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW0000_CTRL_GPU_GET_VIDEO_LINKS_PARAMS {
    LW0000_CTRL_GPU_VIDEO_LINKS links[LW0000_CTRL_GPU_MAX_ATTACHED_GPUS];
} LW0000_CTRL_GPU_GET_VIDEO_LINKS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_ACTIVATE_SLI_CONFIG
 *
 * This command activates (or enables) the specified SLI configuration.  This
 * state is reflected in the LW0000_CTRL_GPU_SLI_INFO_ACTIVE flag (see
 * description of LW0000_CTRL_GPU_SLI_CONFIG above for more details).
 *
 *   sliConfig
 *     This member specifies the LW0000_CTRL_GPU_SLI_CONFIG structure
 *     describing the set of GPU IDs for which SLI information
 *     should be obtained.  The gpuCount and gpuIds members should be
 *     initialized to represent the desired set of GPUs.  The remaining
 *     fields will be used to return the current settings for the
 *     corresponding SLI configuration.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_ACTIVATE_SLI_CONFIG (0x21a) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_ACTIVATE_SLI_CONFIG_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_ACTIVATE_SLI_CONFIG_PARAMS_MESSAGE_ID (0x1AU)

typedef struct LW0000_CTRL_GPU_ACTIVATE_SLI_CONFIG_PARAMS {
    LW0000_CTRL_GPU_SLI_CONFIG sliConfig;
} LW0000_CTRL_GPU_ACTIVATE_SLI_CONFIG_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_INIT_SCALABILITY
 *
 * This command initializes SLI scalability
 * attributes for the specified gpuId.
 *
 * This command takes a gpu Id as argument.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GPU_INIT_SCALABILITY (0x230) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_INIT_SCALABILITY_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_INIT_SCALABILITY_PARAMS_MESSAGE_ID (0x30U)

typedef struct LW0000_CTRL_GPU_INIT_SCALABILITY_PARAMS {
    LwU32 gpuId;
} LW0000_CTRL_GPU_INIT_SCALABILITY_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0000_CTRL_CMD_GPU_GET_SVM_SIZE
 *
 * This command is used to get the SVM size.
 *
 *   gpuId
 *     This parameter uniquely identifies the GPU whose associated
 *     SVM size is to be returned. The value of this field must
 *     match one of those in the table returned by
 *     LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS
 *
 *   SvmSize
 *     SVM size is returned in this.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *
 */
#define LW0000_CTRL_CMD_GPU_GET_SVM_SIZE (0x240) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_SVM_SIZE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_SVM_SIZE_PARAMS_MESSAGE_ID (0x40U)

typedef struct LW0000_CTRL_GPU_GET_SVM_SIZE_PARAMS {
    LwU32 gpuId;
    LwU32 svmSize;
} LW0000_CTRL_GPU_GET_SVM_SIZE_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0000_CTRL_CMD_GPU_NOT_AT_END_OF_LINEAR_SLI_BRIDGE
 *
 * This command is used to query if the GPU is not at the end of a linear
 * SLI video bridge
 *
 *   gpuId
 *     This parameter uniquely identifies the GPU whose associated
 *     position in a linear SLI video bridge is to be returned.
 *     The value of this field must match one of those in the table returned by
 *     LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS
 *
 *   notAtEndOfLinearSliBridge
 *     Value is TRUE if:
 *       - The GPU is in an SLI configuration, and
 *       - The video link is linear (not cirlwlar), and
 *       - The GPU is not at the end of this video link.
 *     Value id FALSE otherwise.

 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0000_CTRL_CMD_GPU_NOT_AT_END_OF_LINEAR_SLI_BRIDGE (0x250) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_NOT_AT_END_OF_LINEAR_SLI_BRIDGE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_NOT_AT_END_OF_LINEAR_SLI_BRIDGE_PARAMS_MESSAGE_ID (0x50U)

typedef struct LW0000_CTRL_GPU_NOT_AT_END_OF_LINEAR_SLI_BRIDGE_PARAMS {
    LwU32 gpuId;
    LwU32 notAtEndOfLinearSliBridge;
} LW0000_CTRL_GPU_NOT_AT_END_OF_LINEAR_SLI_BRIDGE_PARAMS;


/*
 * LW0000_CTRL_GPU_GFA_SESSION_ID
 *
 * This structure is contains session ID information for use
 * with the LW0000_CTRL_CMD_GPU_GFA_GET_GRAPHICS_IDS and
 * LW0000_CTRL_CMD_GPU_GFA_AUTHENTICATE commands.
 *
 *   key
 *     When used with the GET_GRAPHICS_IDS command this field contains
 *     the encryption key used to encrypt the value stored in the id field.
 *     This field is unused when used with the AUTHENTICATE command.
 *   id
 *     When used with the GET_GRAPHICS_IDS command this field holds the
 *     encrypted session ID value.  When used with the AUTHENTICATE command
 *     this field holds the raw session ID value received from the GFA
 *     server.
 */
typedef struct LW0000_CTRL_GPU_GFA_SESSION_ID {
    LwU32 key;
    LwU32 id;
} LW0000_CTRL_GPU_GFA_SESSION_ID;

/*
 * LW0000_CTRL_GPU_GFA_SESSION_CONTAINER
 *
 * This structure contains session
 *
 *   keyOut
 *     This parameter returns the key with which the data returned from
 *     LW0000_CTRL_CMD_GPU_GFA_GET_GRAPHICS_IDS is encrypted.
 *   sessionId
 *     This field contains session ID information.  See the description of
 *     the LW0000_CTRL_GPU_GFA_SESSION_ID structure for more information.
 */
typedef struct LW0000_CTRL_GPU_GFA_SESSION_CONTAINER {
    LwU32                          keyOut;
    LW0000_CTRL_GPU_GFA_SESSION_ID sessionId;
} LW0000_CTRL_GPU_GFA_SESSION_CONTAINER;

/*
 * LW0000_CTRL_CMD_GPU_GET_QUADRO_SLI_FINGER_STATUS
 *
 * This command is used to query the SLI finger connection status of the
 * specified GPU.  Kepler and above Lwdqro GPUs have two SLI fingers.  One
 * of these SLI fingers is reserved for an EXTDEV (GSYNC/SDI) device and
 * the other is for a SLI bridge.
 *
 *   gpuId
 *     This parameter uniquely identifies the GPU whose SLI finger info
 *     needs to be returned. The value of this field must match one of
 *     those in the table returned by LW0000_CTRL_CMD_GPU_GET_ATTACHED_IDS.
 *
 *   sliFingerStatus
 *     This field will give information of SLI finger for Lwdqro boards.
 *       LW0000_CTRL_GPU_QUADRO_SLI_FINGER_STATUS_OK
 *         GPU and EXTDEV if preset are connected over correct SLI fingers.
 *       LW0000_CTRL_GPU_QUADRO_SLI_FINGER_STATUS_VIDEO_BRIDGE_NOT_SUPPORTED
 *         The video bridge is connected in wrong SLI finger.
 *       LW0000_CTRL_GPU_QUADRO_SLI_FINGER_STATUS_EXTDEV_NOT_SUPPORTED
 *         The EXTDEV (GSYNC/SDI) cable is connected to wrong SLI finger.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0000_CTRL_CMD_GPU_GET_QUADRO_SLI_FINGER_STATUS (0x273) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_QUADRO_SLI_FINGER_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_QUADRO_SLI_FINGER_STATUS_PARAMS_MESSAGE_ID (0x73U)

typedef struct LW0000_CTRL_GPU_GET_QUADRO_SLI_FINGER_STATUS_PARAMS {
    LwU32 gpuId;
    LwU32 sliFingerStatus;
} LW0000_CTRL_GPU_GET_QUADRO_SLI_FINGER_STATUS_PARAMS;

/* legal sliFingerStatus values */
#define LW0000_CTRL_GPU_QUADRO_SLI_FINGER_STATUS_OK                         (0x00000000)
#define LW0000_CTRL_GPU_QUADRO_SLI_FINGER_STATUS_VIDEO_BRIDGE_NOT_SUPPORTED (0x00000001)
#define LW0000_CTRL_GPU_QUADRO_SLI_FINGER_STATUS_EXTDEV_NOT_SUPPORTED       (0x00000002)


/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0000_CTRL_CMD_GPU_GET_UUID_INFO
 *
 * This command returns requested information pertaining to the GPU
 * specified by the GPU UUID passed in.
 *
 * Generally only GPUs that have been attached are visible to this call. Therefore
 * queries on unattached GPUs will fail with LW_ERR_OBJECT_NOT_FOUND.  However, 
 * a query for a SHA1 UUID may succeed for an unattached GPU in cases where the GID
 * is cached, such as an excluded GPU.
 *
 *   gpuGuid (INPUT)
 *     The GPU UUID of the gpu whose parameters are to be returned. Refer to
 *     LW0000_CTRL_CMD_GPU_GET_ID_INFO for more information.
 *
 *   flags (INPUT)
 *     The _FORMAT* flags designate ascii string format or a binary format.
 *
 *     The _TYPE* flags designate either SHA-1-based (32-hex-character) or
 *     SHA-256-based (64-hex-character).
 *
 *   gpuId (OUTPUT)
 *     The GPU ID of the GPU identified by gpuGuid. Refer to
 *     LW0000_CTRL_CMD_GPU_GET_ID_INFO for more information.
 *
 *   deviceInstance (OUTPUT)
 *     The device instance of the GPU identified by gpuGuid. Refer to
 *     LW0000_CTRL_CMD_GPU_GET_ID_INFO for more information.
 *
 *   subdeviceInstance (OUTPUT)
 *     The subdevice instance of the GPU identified by gpuGuid. Refer to
 *     LW0000_CTRL_CMD_GPU_GET_ID_INFO for more information.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *
 */
#define LW0000_CTRL_CMD_GPU_GET_UUID_INFO                                   (0x274) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_UUID_INFO_PARAMS_MESSAGE_ID" */

/* maximum possible number of bytes of GID information */
#define LW0000_GPU_MAX_GID_LENGTH                                           (0x00000100)

#define LW0000_CTRL_GPU_GET_UUID_INFO_PARAMS_MESSAGE_ID (0x74U)

typedef struct LW0000_CTRL_GPU_GET_UUID_INFO_PARAMS {
    LwU8  gpuUuid[LW0000_GPU_MAX_GID_LENGTH];
    LwU32 flags;
    LwU32 gpuId;
    LwU32 deviceInstance;
    LwU32 subdeviceInstance;
} LW0000_CTRL_GPU_GET_UUID_INFO_PARAMS;

#define LW0000_CTRL_CMD_GPU_GET_UUID_INFO_FLAGS_FORMAT                       1:0
#define LW0000_CTRL_CMD_GPU_GET_UUID_INFO_FLAGS_FORMAT_ASCII  (0x00000000)
#define LW0000_CTRL_CMD_GPU_GET_UUID_INFO_FLAGS_FORMAT_BINARY (0x00000002)

#define LW0000_CTRL_CMD_GPU_GET_UUID_INFO_FLAGS_TYPE                         2:2
#define LW0000_CTRL_CMD_GPU_GET_UUID_INFO_FLAGS_TYPE_SHA1     (0x00000000)
#define LW0000_CTRL_CMD_GPU_GET_UUID_INFO_FLAGS_TYPE_SHA256   (0x00000001)

/*
 * LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID
 *
 * This command returns the GPU UUID for the provided GPU ID.
 * Note that only GPUs that have been attached are visible to this call.
 * Therefore queries on unattached GPUs will fail
 * with LW_ERR_OBJECT_NOT_FOUND.
 *
 *   gpuId (INPUT)
 *     The GPU ID whose parameters are to be returned. Refer to
 *     LW0000_CTRL_CMD_GPU_GET_ID_INFO for more information.
 *
 *   flags (INPUT)
 *
 *     LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_ASCII
 *       This value is used to request the GPU UUID in ASCII format.
 *     LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_BINARY
 *       This value is used to request the GPU UUID in binary format.
 *
 *     LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_TYPE_SHA1
 *       This value is used to request that the GPU UUID value
 *       be SHA1-based (32-hex-character).
 *     LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_TYPE_SHA256
 *       This value is used to request that the GPU UUID value
 *       be SHA256-based (64-hex-character).
 *
 *   gpuUuid[LW0000_GPU_MAX_GID_LENGTH] (OUTPUT)
 *     The GPU UUID of the GPU identified by GPU ID. Refer to
 *     LW0000_CTRL_CMD_GPU_GET_ID_INFO for more information.
 *
 *   uuidStrLen (OUTPUT)
 *     The length of the UUID returned which is related to the format that
 *     was requested using flags.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID              (0x275) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS_MESSAGE_ID (0x75U)

typedef struct LW0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS {
    LwU32 gpuId;
    LwU32 flags;
    LwU8  gpuUuid[LW0000_GPU_MAX_GID_LENGTH];
    LwU32 uuidStrLen;
} LW0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS;

/* valid format values */
#define LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT                       1:0
#define LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_ASCII  (0x00000000)
#define LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_FORMAT_BINARY (0x00000002)

/*valid type values*/
#define LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_TYPE                         2:2
#define LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_TYPE_SHA1     (0x00000000)
#define LW0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID_FLAGS_TYPE_SHA256   (0x00000001)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0000_CTRL_CMD_GPU_GET_DRIVER_SLI_HACK_STATUS
 *
 * This command queries the status of the driver hack detection
 * oclwrring during the SLI approval process.
 *
 *
 *   bDriverHacked
 *      Set to true if the hack was detected during SLI Approval
 *
 * Possible status values returned are:
 *   LW_OK
 *
 */
#define LW0000_CTRL_CMD_GPU_GET_DRIVER_SLI_HACK_STATUS               (0x276) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_DRIVER_SLI_HACK_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_DRIVER_SLI_HACK_STATUS_PARAMS_MESSAGE_ID (0x76U)

typedef struct LW0000_CTRL_GPU_GET_DRIVER_SLI_HACK_STATUS_PARAMS {
    LwBool bDriverHacked;
} LW0000_CTRL_GPU_GET_DRIVER_SLI_HACK_STATUS_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_SLI_SKIP_FB_SIZE_COMPARE
 *
 * This command allows SLI between GPUs with different FB sizes.
 *
 *   bEnable
 *      Set to true to allow SLI between GPUs with different FB sizes.
 *
 * Possible status values returned are:
 *   LW_OK
 *
 */
#define LW0000_CTRL_CMD_GPU_SLI_SKIP_FB_SIZE_COMPARE (0x277) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_SLI_SKIP_FB_SIZE_COMPARE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_SLI_SKIP_FB_SIZE_COMPARE_PARAMS_MESSAGE_ID (0x77U)

typedef struct LW0000_CTRL_GPU_SLI_SKIP_FB_SIZE_COMPARE_PARAMS {
    LwBool bEnable;
} LW0000_CTRL_GPU_SLI_SKIP_FB_SIZE_COMPARE_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0000_CTRL_CMD_GPU_MODIFY_DRAIN_STATE
 *
 * This command is used to enter or exit the so called "drain" state.
 * When this state is enabled, the existing clients continue exelwting
 * as usual, however no new client connections are allowed.
 * This is done in order to "drain" the system of the running clients
 * in preparation to selectively powering down the GPU.
 * No GPU can enter a bleed state if that GPU is in an SLI group.
 * In that case, LW_ERR_IN_USE is returned.
 * Requires administrator privileges.
 *
 * It is expected, that the "drain" state will be eventually deprecated
 * and replaced with another mechanism to quiesce a GPU (Bug 1718113).
 *
 *  gpuId (INPUT)
 *    This parameter should specify a valid GPU ID value.  If there
 *    is no GPU present with the specified ID, a status of
 *    LW_ERR_ILWALID_ARGUMENT is returned.
 *  newState (INPUT)
 *    This input parameter is used to enter or exit the "drain"
 *    software state of the GPU specified by the gpuId parameter.
 *    Possible values are:
 *      LW0000_CTRL_GPU_DRAIN_STATE_ENABLED
 *      LW0000_CTRL_GPU_DRAIN_STATE_DISABLED
 *  flags (INPUT)
 *    LW0000_CTRL_GPU_DRAIN_STATE_FLAG_REMOVE_DEVICE
 *      if set, upon reaching quiescence, a request will be made to
 *      the OS to "forget" the PCI device associated with the
 *      GPU specified by the gpuId parameter, in case such a request
 *      is supported by the OS. Otherwise, LW_ERR_NOT_SUPPORTED
 *      will be returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_IN_USE
 */

#define LW0000_CTRL_CMD_GPU_MODIFY_DRAIN_STATE         (0x278) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_MODIFY_DRAIN_STATE_PARAMS_MESSAGE_ID" */

/* Possible values of newState */
#define LW0000_CTRL_GPU_DRAIN_STATE_DISABLED           (0x00000000)
#define LW0000_CTRL_GPU_DRAIN_STATE_ENABLED            (0x00000001)

/* Defined bits for the "flags" argument */
#define LW0000_CTRL_GPU_DRAIN_STATE_FLAG_REMOVE_DEVICE (0x00000001)
#define LW0000_CTRL_GPU_DRAIN_STATE_FLAG_LINK_DISABLE  (0x00000002)

#define LW0000_CTRL_GPU_MODIFY_DRAIN_STATE_PARAMS_MESSAGE_ID (0x78U)

typedef struct LW0000_CTRL_GPU_MODIFY_DRAIN_STATE_PARAMS {
    LwU32 gpuId;
    LwU32 newState;
    LwU32 flags;
} LW0000_CTRL_GPU_MODIFY_DRAIN_STATE_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE
 *
 *  gpuId (INPUT)
 *    This parameter should specify a valid GPU ID value.  If there
 *    is no GPU present with the specified ID, a status of
 *    LWOS_STATUS_ERROR_ILWALID_ARGUMENT is returned.
 *  drainState (OUTPUT)
 *    This parameter returns a value indicating if the "drain"
 *    state is lwrrently enabled or not for the specified GPU. See the
 *    description of LW0000_CTRL_CMD_GPU_MODIFY_DRAIN_STATE.
 *    Possible values are:
 *      LW0000_CTRL_GPU_DRAIN_STATE_ENABLED
 *      LW0000_CTRL_GPU_DRAIN_STATE_DISABLED
 *  flags (OUTPUT)
 *    LW0000_CTRL_GPU_DRAIN_STATE_FLAG_REMOVE_DEVICE
 *      if set, upon reaching quiesence, the GPU device will be
 *      removed automatically from the kernel space, similar
 *      to what writing "1" to the sysfs "remove" node does.
 *    LW0000_CTRL_GPU_DRAIN_STATE_FLAG_LINK_DISABLE
 *      after removing the GPU, also disable the parent bridge's
 *      PCIe link. This flag can only be set in conjunction with
 *      LW0000_CTRL_GPU_DRAIN_STATE_FLAG_REMOVE_DEVICE, and then
 *      only when the GPU is already idle (not attached).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE (0x279) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_QUERY_DRAIN_STATE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_QUERY_DRAIN_STATE_PARAMS_MESSAGE_ID (0x79U)

typedef struct LW0000_CTRL_GPU_QUERY_DRAIN_STATE_PARAMS {
    LwU32 gpuId;
    LwU32 drainState;
    LwU32 flags;
} LW0000_CTRL_GPU_QUERY_DRAIN_STATE_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_DISCOVER
 *
 * This request asks the OS to scan the PCI tree or a sub-tree for GPUs,
 * that are not yet known to the OS, and to make them available for use.
 * If all of domain:bus:slot.function are zeros, the entire tree is scanned,
 * otherwise the parameters identify the bridge device, that roots the
 * subtree to be scanned.
 * Requires administrator privileges.
 *
 *  domain (INPUT)
 *    PCI domain of the bridge
 *  bus (INPUT)
 *    PCI bus of the bridge
 *  slot (INPUT)
 *    PCI slot of the bridge
 *  function (INPUT)
 *    PCI function of the bridge
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_OPERATING_SYSTEM
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0000_CTRL_CMD_GPU_DISCOVER (0x27a) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | 0x7A" */

typedef struct LW0000_CTRL_GPU_DISCOVER_PARAMS {
    LwU32 domain;
    LwU8  bus;
    LwU8  slot;
    LwU8  function;
} LW0000_CTRL_GPU_DISCOVER_PARAMS;

/*
 * LW0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE
 *
 * This command is used to get the content of the MemOp (LWCA Memory Operation)
 * enablement mask, which can be overridden by using the MemOpOverride RegKey.
 *
 * The enableMask member must be treated as a bitmask, where each bit controls
 * the enablement of a feature.
 *
 * So far, the only feature which is defined controls to whole MemOp APIs.
 *
 * Possible status values returned are:
 *   LW_OK
 *
 */
#define LW0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE (0x27b) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_GET_MEMOP_ENABLE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_GET_MEMOP_ENABLE_PARAMS_MESSAGE_ID (0x7BU)

typedef struct LW0000_CTRL_GPU_GET_MEMOP_ENABLE_PARAMS {
    LwU32 enableMask;
} LW0000_CTRL_GPU_GET_MEMOP_ENABLE_PARAMS;

#define LW0000_CTRL_GPU_FLAGS_MEMOP_ENABLE   (0x00000001)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL
 */
#define LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL (0x280) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | 0x80" */

typedef struct LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_MOTHERBOARD {
    LwU8   cpu;
    LwU16  chipset_vendor_id;
    LwU16  chipset_device_id;
    LwU32  chipset_sli_bond;
    LwBool is_not_hostbridge_id;
    LwBool bReferenced;
} LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_MOTHERBOARD;

typedef struct LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_GPU {
    LwU32 pmc_boot_value;
    LwU32 pci_device_id;
    LwU32 pci_ss_device_id;
    LwU32 instance;
} LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_GPU;

typedef struct LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_GRAPHICSBOARD {
    LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_GPU gpu[LW_MAX_DEVICES];
    LwU32                                    nb_gpu;
    LwU32                                    itf;
    LwU32                                    pcielinewidth;
    LwU32                                    br02Count;
    LwU32                                    br03Count;
    LwU32                                    br04Count;
    LwU32                                    plxCount;
    LwU8                                     br04Rev;
    LwBool                                   is_mobile; // This represents all GPUs on the board
    LwU32                                    RamTopOfMemoryMb; // This represents all GPUs on the board
    LwBool                                   supports_sli; // This represents all GPUs on the board
    LwBool                                   is_quadro;// This represents all GPUs on the board
    LwBool                                   no_cipher;// The board does not have the cipher engine allocated
    LwBool                                   no_ppp;// The board does not have the ppp engine allocated
    LwBool                                   bReferenced;
} LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_GRAPHICSBOARD;

typedef struct LW0000_CTRL_GPU_GET_SLI_APPROVAL_PARAMS {
    LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_MOTHERBOARD   motherboard;
    LW0000_CTRL_CMD_GPU_GET_SLI_APPROVAL_GRAPHICSBOARD graphicsboard[LW_MAX_DEVICES];
    char                                               sli_cookie[LW0000_SYSTEM_MAX_APPROVAL_COOKIE_STRING_BUFFER];
    LwU32                                              nb_graphicsboard;
    LwU32                                              slibridge;
    LwU8                                               qsync;
    LwU32                                              os;
    LwU32                                              br03Cascaded;
    LwU32                                              br04Cascaded;
    LwU32                                              br04NotCascaded;
    LwU8                                               br04Rev;
    LwU32                                              plxCount;
    LwU32                                              sli_status;
    LwU32                                              hacklevel;
    LwBool                                             has_sli_cookie;
    LwU16                                              chipset_ss_vendor_id;
    LwU16                                              chipset_ss_device_id;
} LW0000_CTRL_GPU_GET_SLI_APPROVAL_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0000_CTRL_CMD_GPU_DISABLE_LWLINK_INIT
 *
 * This privileged command is used to disable initialization for the LWLinks
 * provided in the mask.
 *
 * The mask must be applied before the GPU is attached. DISABLE_LWLINK_INIT
 * is an NOP for non-LWLink GPUs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_IN_USE
 *
 */
#define LW0000_CTRL_CMD_GPU_DISABLE_LWLINK_INIT (0x281) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_DISABLE_LWLINK_INIT_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_DISABLE_LWLINK_INIT_PARAMS_MESSAGE_ID (0x81U)

typedef struct LW0000_CTRL_GPU_DISABLE_LWLINK_INIT_PARAMS {
    LwU32  gpuId;
    LwU32  mask;
    LwBool bSkipHwLwlinkDisable;
} LW0000_CTRL_GPU_DISABLE_LWLINK_INIT_PARAMS;


#define LW0000_CTRL_GPU_LEGACY_CONFIG_MAX_PARAM_DATA     0x00000175
#define LW0000_CTRL_GPU_LEGACY_CONFIG_MAX_PROPERTIES_IN  6
#define LW0000_CTRL_GPU_LEGACY_CONFIG_MAX_PROPERTIES_OUT 5

/*
 * LW0000_CTRL_CMD_GPU_LEGACY_CONFIG
 *
 * Path to use legacy RM GetConfig/Set API. This API is being phased out.
 */
#define LW0000_CTRL_CMD_GPU_LEGACY_CONFIG                (0x282) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_LEGACY_CONFIG_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_LEGACY_CONFIG_PARAMS_MESSAGE_ID (0x82U)

typedef struct LW0000_CTRL_GPU_LEGACY_CONFIG_PARAMS {
    LwHandle hContext;    /* [in]  - Handle of object to perform operation on (Device, Subdevice, etc) */
    LwU32    opType;      /* [in]  - Type of API */
    LwV32    index;       /* [in]  - command type */
    LwU32    dataType;    /* [out] - data union type */

    union {
        struct {
            LwV32 value;
        } configGet;
        struct {
            LwU32 newValue;
            LwU32 oldValue;
        } configSet;
        struct {
            LwU8  paramData[LW0000_CTRL_GPU_LEGACY_CONFIG_MAX_PARAM_DATA];
            LwU32 paramSize;
        } configEx;
        struct {
            LwU32 propertyId;
            LwU32 propertyIn[LW0000_CTRL_GPU_LEGACY_CONFIG_MAX_PROPERTIES_IN];
            LwU32 propertyOut[LW0000_CTRL_GPU_LEGACY_CONFIG_MAX_PROPERTIES_OUT];
        } reservedProperty;
    } data;
} LW0000_CTRL_GPU_LEGACY_CONFIG_PARAMS;

#define LW0000_CTRL_GPU_LEGACY_CONFIG_OP_TYPE_GET      (0x00000000)
#define LW0000_CTRL_GPU_LEGACY_CONFIG_OP_TYPE_SET      (0x00000001)
#define LW0000_CTRL_GPU_LEGACY_CONFIG_OP_TYPE_GET_EX   (0x00000002)
#define LW0000_CTRL_GPU_LEGACY_CONFIG_OP_TYPE_SET_EX   (0x00000003)
#define LW0000_CTRL_GPU_LEGACY_CONFIG_OP_TYPE_RESERVED (0x00000004)

/*
 * LW0000_CTRL_CMD_IDLE_CHANNELS
 */
#define LW0000_CTRL_CMD_IDLE_CHANNELS                  (0x283) /* finn: Evaluated from "(FINN_LW01_ROOT_GPU_INTERFACE_ID << 8) | LW0000_CTRL_GPU_IDLE_CHANNELS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GPU_IDLE_CHANNELS_PARAMS_MESSAGE_ID (0x83U)

typedef struct LW0000_CTRL_GPU_IDLE_CHANNELS_PARAMS {
    LwHandle hDevice;
    LwHandle hChannel;
    LwV32    numChannels;
    /* C form: LwP64 phClients LW_ALIGN_BYTES(8); */
    LW_DECLARE_ALIGNED(LwP64 phClients, 8);
    /* C form: LwP64 phDevices LW_ALIGN_BYTES(8); */
    LW_DECLARE_ALIGNED(LwP64 phDevices, 8);
    /* C form: LwP64 phChannels LW_ALIGN_BYTES(8); */
    LW_DECLARE_ALIGNED(LwP64 phChannels, 8);
    LwV32    flags;
    LwV32    timeout;
} LW0000_CTRL_GPU_IDLE_CHANNELS_PARAMS;

/* _ctrl0000gpu_h_ */

