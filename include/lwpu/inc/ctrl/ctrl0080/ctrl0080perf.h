/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2019 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080perf.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE perf control commands and parameters */

/*
 * LW0080_CTRL_CMD_PERF_GET_CAPS
 *
 * This command returns the set of performance capabilities for the device
 * in the form of an array of unsigned bytes.  Performance capabilities
 * include supported features and required workarounds for the performance
 * management subsystem(s) in the device, each represented by a byte
 * offset into the table and a bit position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0080_CTRL_PERF_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the power caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_POINTER
 */
#define LW0080_CTRL_CMD_PERF_GET_CAPS (0x801901) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_GET_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_GET_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_PERF_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
} LW0080_CTRL_PERF_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0080_CTRL_PERF_GET_CAP(tbl,c)         (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW0080_CTRL_PERF_CAPS_RMSTRESS                              0:0x01
#define LW0080_CTRL_PERF_CAPS_CONSTANT_MCLK                         0:0x02
#define LW0080_CTRL_PERF_CAPS_2D_STRESS_TEST                        0:0x04
#define LW0080_CTRL_PERF_CAPS_DX_WAR_BUG_3124034                    0:0x08

/* size in bytes of power caps table */
#define LW0080_CTRL_PERF_CAPS_TBL_SIZE                             1

/*
 * Following bit field can be used to query whether GPU supports power virus
 * application profiling.
 */
#define LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_APP_PROFILING                0:0
#define LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_APP_PROFILING_NO  (0x00000000)
#define LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_APP_PROFILING_YES (0x00000001)

/*
 * LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_PARAMS
 *
 * Query whether GPU supports power virus application profiling.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0080_CTRL_CMD_PERF_GET_POWER_VIRUS_SUPPORT               (0x801902) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_PARAMS {
    LwU32 mode;
} LW0080_CTRL_PERF_GET_POWER_VIRUS_SUPPORT_PARAMS;

/*
 * LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION
 *
 * Possible values are:
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_ILWALID
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_RUNNING
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_TERMINATED
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_END
 *
 * Power virus state can be either running or terminated.
 */
typedef enum LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION {
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_ILWALID = 0,
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_RUNNING = 1,
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_TERMINATED = 2,
    // Add new entries here
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION_END = 3, // Should always be last entry
} LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION;

/*
 * LW0080_CTRL_PERF_POWER_VIRUS_APP_ID
 *
 * Possible values are:
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_ILWALID
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_FURMARK
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_OCCT
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_EVGA_OC_SCANNER
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_MSI_KOMBUSTOR
 *      LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_END
 *
 * Add an entry here for supporting additional power virus applications.
 */
typedef enum LW0080_CTRL_PERF_POWER_VIRUS_APP_ID {
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_ILWALID = 0,
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_FURMARK = 1,         //OGL
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_OCCT = 2,            //DX
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_EVGA_OC_SCANNER = 3, //OGL
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_MSI_KOMBUSTOR = 4,
    // Add new entries here
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ID_END = 5, // Should always be last entry
} LW0080_CTRL_PERF_POWER_VIRUS_APP_ID;

/*
 * LW0080_CTRL_CMD_PERF_SET_POWER_VIRUS_APP_NOTIFICATION
 *
 *   action
 *      Set by the client to convey application status (either running or terminated).
 *
 *   applicationID
 *      Set by the client to respective application ID.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_REQUEST
 */
#define LW0080_CTRL_CMD_PERF_SET_POWER_VIRUS_APP_NOTIFICATION (0x801903) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_SET_POWER_VIRUS_APP_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_SET_POWER_VIRUS_APP_NOTIFICATION_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0080_CTRL_PERF_SET_POWER_VIRUS_APP_NOTIFICATION_PARAMS {
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ACTION action;
    LW0080_CTRL_PERF_POWER_VIRUS_APP_ID     applicationID;
} LW0080_CTRL_PERF_SET_POWER_VIRUS_APP_NOTIFICATION_PARAMS;

/*
 * LW0080_CTRL_CMD_PERF_NOTIFY_SREEN_SAVER_STATE
 *
 *  This RM CTRL is deprecated as a part of the Unlinked SLI project.
 *  Please, instead use the unicast version LW2080_CTRL_CMD_PERF_NOTIFY_SREEN_SAVER_STATE
 *
 *  This command will notify the current state of screen saver to RM.
 *  Based on the screen saver state RM will tune the performance.
 *
 *   bRunning
 *     When set to TRUE this parameter indicates that the screen saver is in
 *     running state.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_PERF_SET_SCREEN_SAVER_STATE (0x801904) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS {
    LwBool bRunning;
} LW0080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS;

/*
 * LW0080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_GET_INFO
 *
 * This command returns the static information pertaining to SLI GPU Boost
 * synchronization.
 *
 *  bEnable
 *      When set to TRUE, this parameter indicates that the feature is enabled.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_GET_INFO (0x801905) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS {
    LwBool bEnable;
} LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS;

/*
 * LW0080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_GET_CONTROL
 *
 * This command returns the control information pertaining to SLI GPU Boost
 * synchronization.
 *
 *  bActivate
 *      When set to TRUE, this parameter indicates that the feature is active.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_GET_CONTROL (0x801906) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | 0x6" */
/*
 * LW0080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_SET_CONTROL
 *
 * This command sets the control information pertaining to SLI GPU Boost
 * synchronization.
 *
 *  bActivate
 *      When set to TRUE, feature will be activated.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_ARGUMENT
 *  LW_ERR_ILWALID_REQUEST
 *  LW_ERR_ILWALID_STATE
 */
#define LW0080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_SET_CONTROL (0x801907) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_CONTROL_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_CONTROL_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_CONTROL_PARAMS {
    LwBool bActivate;
} LW0080_CTRL_PERF_SLI_GPU_BOOST_SYNC_CONTROL_PARAMS;

/*
 * LW0080_CTRL_CMD_PERF_ADJUST_LIMIT_BY_PERFORMANCE
 *
 * This command can be used by a controller outside of RM to trigger a change
 * in perf limit based on the given performance parameters.
 *
 *   flags
 *     This parameter specifies LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_XXX flags:
 *     _CLEAR_YES to clear current limit
 *     _FULLSCREEN_YES to specify that the client is running in fullscreen mode
 *     _VIDEO_YES to specify a GameStream/ShadowPlay client
 *     _VR_YES to specify a VR display client
 *     _REFCNT_INC_YES to increment reference counter for max perf request
 *     _REFCNT_DEC_YES to decrement reference counter for max perf request
 *     _VR_APP_YES to specify a VR application client
 *     _WHISPER_ENABLE to apply WhisperMode cap
 *     _WHISPER_DISABLE to clear WhisperMode cap
 *     _ANSEL_YES to specify an Ansel overhead request
 *   evalUs
 *     This parameter specifies how long the client spent evaluating before
 *     making this call to adjust limit.
 *   current
 *     This parameter specifies the current performance (lower is faster).
 *   target
 *     This parameter specifies the target performance (lower is faster).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_PERF_ADJUST_LIMIT_BY_PERFORMANCE (0x801908) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW0080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS {
    LwU32 flags;
    LwU32 evalUs;
    LwU32 current;
    LwU32 target;
} LW0080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS;

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR                        0:0
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR_NO          (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR_YES         (0x00000001)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN                   1:1
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN_NO     (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN_YES    (0x00000001)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO                        2:2
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO_NO          (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO_YES         (0x00000001)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR                           3:3
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_NO             (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_YES            (0x00000001)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC                   4:4
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC_NO     (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC_YES    (0x00000001)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC                   5:5
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC_NO     (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC_YES    (0x00000001)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP                       6:6
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP_NO         (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP_YES        (0x00000001)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER                      8:7
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_NO_CHANGE (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_ENABLE    (0x00000001)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_DISABLE   (0x00000002)

#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL                        9:9
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL_NO          (0x00000000)
#define LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL_YES         (0x00000001)

// Current to target ratio is capped to 100x
#define LW0080_CTRL_PERF_ADJUST_LIMIT_C2T_RATIO_MAX          (100)

/*
 * LW0080_CTRL_CMD_PERF_LWDA_LIMIT_SET_CONTROL
 *
 * This command is deprecated as a part of the Unlinked SLI project.
 * Please, instead use the unicast version LW2080_CTRL_CMD_PERF_LWDA_LIMIT_SET_CONTROL
 *
 * This command sets the control information pertaining to Lwca limit.
 *
 *  bLwdaLimit
 *      When set to TRUE, clocks will be limited based on Lwca.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_ARGUMENT
 *  LW_ERR_ILWALID_REQUEST
 *  LW_ERR_ILWALID_STATE
 */
#define LW0080_CTRL_CMD_PERF_LWDA_LIMIT_SET_CONTROL          (0x801909) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW0080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS {
    LwBool bLwdaLimit;
} LW0080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS;


#define LW0080_CTRL_CMD_PERF_GET_CAPS_V2 (0x801910) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_PERF_INTERFACE_ID << 8) | LW0080_CTRL_PERF_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_PERF_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW0080_CTRL_PERF_GET_CAPS_V2_PARAMS {
    LwU8 capsTbl[LW0080_CTRL_PERF_CAPS_TBL_SIZE];
} LW0080_CTRL_PERF_GET_CAPS_V2_PARAMS;

/* _ctrl0080perf_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

