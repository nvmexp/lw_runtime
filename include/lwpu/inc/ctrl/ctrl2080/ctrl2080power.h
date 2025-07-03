/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080power.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/*
 * LW2080_CTRL_CMD_GPU_POWER_ON_OFF
 *
 * This command exelwtes the save context required before powering-off the GPU
 * or the restore context required after powering-on the GPU. To be used by
 * GC6 or Rail-Gating or GC5
 *
 *  params.bUseHwTimer
 *    Specifies whether the hw timer is to be used to wake up from GC5/GC6
 *  params.timeToWakeUs
 *    Specifies the number of microseconds before waking up from GC5/GC6
 *  params.bAutoSetupHwTimer
 *    Specifies whether the HW(PMU) should figure out the wakeup timer by itself
 *    based on the programmed ptimer alarm value.
 *    Client needs to set the bUseHwTimer as well to use this mode.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_READY
 *   LW_ERR_TIMEOUT
 *   LW_ERR_GENERIC
 *
 * For GC6 transitions, these errors are non-fatal on entry -
 *      LW_ERR_NOT_SUPPORTED
 *      LW_ERR_ILWALID_STATE
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_NOT_READY
 */
#define LW2080_CTRL_CMD_GPU_POWER_ON_OFF (0x20802701) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GPU_POWER_ON_OFF_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_POWER_ON_OFF_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_GPU_POWER_ON_OFF_PARAMS {
    LwU32 action;
    struct {
        LwBool bUseHwTimer;
        LwU32  timeToWakeUs;
        LwBool bAutoSetupHwTimer;
        LwBool bIsGpuSelfWake;
        LwBool bIsS3Transition;
        LwBool bIsRTD3Transition;
        LwBool bIsRTD3CoreRailPowerLwt;
        LwBool bIsHdaOSPowerManaged;
        LwBool bIsRTD3HotTransition;   //output
    } params;
} LW2080_CTRL_GPU_POWER_ON_OFF_PARAMS;

#define LW2080_CTRL_GPU_POWER_ON_OFF_RG_SAVE                     (0x00000001)
#define LW2080_CTRL_GPU_POWER_ON_OFF_RG_RESTORE                  (0x00000002)
#define LW2080_CTRL_GPU_POWER_ON_OFF_GC6_ENTER                   (0x00000003)
#define LW2080_CTRL_GPU_POWER_ON_OFF_GC6_EXIT                    (0x00000004)
#define LW2080_CTRL_GPU_POWER_ON_OFF_GC5_ENTER                   (0x00000005)
#define LW2080_CTRL_GPU_POWER_ON_OFF_GC5_EXIT                    (0x00000006)
#define LW2080_CTRL_GPU_POWER_ON_OFF_GC5_ACTIVATE                (0x00000007)
#define LW2080_CTRL_GPU_POWER_ON_OFF_GC5_DEACTIVATE              (0x00000008)
#define LW2080_CTRL_GPU_POWER_ON_OFF_MSHYBRID_GC6_ENTER          (0x00000009)
#define LW2080_CTRL_GPU_POWER_ON_OFF_MSHYBRID_GC6_EXIT           (0x0000000A)
#define LW2080_CTRL_GPU_POWER_ON_OFF_FAST_GC6_ENTER              (0x0000000B)
#define LW2080_CTRL_GPU_POWER_ON_OFF_FAST_GC6_EXIT               (0x0000000C)

/*
 * LW2080_CTRL_CMD_POWER_FEATURES_SUPPORTED
 *
 * This command checks which power features are supported
 *
 *  powerFeaturesSupported
 *    Bit mask that specifies power features supported
 *
 * Possible status return values are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_POWER_FEATURES_SUPPORTED                 (0x20802702) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_POWER_FEATURES_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_RG                  0:0
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_RG_FALSE            (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_RG_TRUE             (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6                 1:1
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_FALSE           (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_TRUE            (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC5                 2:2
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC5_FALSE           (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC5_TRUE            (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_MSHYBRID            3:3
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_MSHYBRID_FALSE      (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_MSHYBRID_TRUE       (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_RTD3            4:4
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_RTD3_FALSE      (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_RTD3_TRUE       (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_FGC6                5:5
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_FGC6_FALSE          (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_FGC6_TRUE           (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_RTD3_SLI        6:6
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_RTD3_SLI_FALSE  (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GC6_RTD3_SLI_TRUE   (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_LEGACY_GC6          7:7
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_LEGACY_GC6_FALSE    (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_LEGACY_GC6_TRUE     (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GCOFF               8:8
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GCOFF_FALSE         (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GCOFF_TRUE          (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_RTD3_GCOFF          9:9
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_RTD3_GCOFF_FALSE    (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_RTD3_GCOFF_TRUE     (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_AGC6                10:10
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_AGC6_FALSE          (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_AGC6_TRUE           (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_APSTATE             11:11
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_APSTATE_FALSE       (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_APSTATE_TRUE        (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_LEGACY_GCOFF        12:12
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_LEGACY_GCOFF_FALSE  (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_LEGACY_GCOFF_TRUE   (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GCOFF_CB_PERF       13:13
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GCOFF_CB_PERF_FALSE (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_GCOFF_CB_PERF_TRUE  (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_SUPPORTED_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_POWER_FEATURES_SUPPORTED_PARAMS {
    LwU32 powerFeaturesSupported;
} LW2080_CTRL_POWER_FEATURES_SUPPORTED_PARAMS;

/*
 * LW2080_CTRL_CMD_POWER_FEATURE_GC6_INFO
 *
 * This command is used to retrieve GC6 debug/status information
 *
 *  gc6InfoMask
 *    Bit mask that specifies the various GC6 status/debug information
 *  lwsrInfoMask
 *    Bit mask that specifies the various LWSR status/debug information
 *  sbiosCapsMaskGC6
 *    Bit mask returned by JT_FUNC_CAPS (see also v0.71 of JT spec):
 *      _JTE    : JT Enabled
 *      _LWSE   : LWSR Enabled
 *      _PPR    : Panel Power Rail
 *      _SRPR   : Self-Refresh Controller Power Rail
 *      _FBPR   : FB Power Rail
 *      _GPR    : GPU Power Rail
 *      _GCR    : GC6 ROM
 *      _PTH    : Panic Trap Handler
 *      _NOT    : Supports Notify
 *      _MXRV   : Highest Revision Level Supported
 *  gc6CapsMask
 *    Bit mask that specifies GC6 caps in the driver
 *
 * Possible status return values are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_POWER_FEATURE_GET_GC6_INFO                         (0x20802703) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_POWER_FEATURE_GC6_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_FBCLAMP_ENABLED             0:0
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_FBCLAMP_ENABLED_TRUE      (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_FBCLAMP_ENABLED_FALSE     (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_IFR_ENABLED                 1:1
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_IFR_ENABLED_TRUE          (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_IFR_ENABLED_FALSE         (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_SBIOS_ENABLED                     2:2
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_SBIOS_ENABLED_TRUE              (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_SBIOS_ENABLED_FALSE             (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_RM_REGKEY_ROMLESS_ENABLED         3:3
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_RM_REGKEY_ROMLESS_ENABLED_TRUE  (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_RM_REGKEY_ROMLESS_ENABLED_FALSE (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_RM_REGKEY_ROM_ENABLED             4:4
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_RM_REGKEY_ROM_ENABLED_TRUE      (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_RM_REGKEY_ROM_ENABLED_FALSE     (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_PGISLAND_PRESENT                  5:5
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_PGISLAND_PRESENT_TRUE           (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_PGISLAND_PRESENT_FALSE          (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_SUPPORTED                   6:6
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_SUPPORTED_TRUE            (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_VBIOS_SUPPORTED_FALSE           (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_CHIP_SUPPORTED                    7:7
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_CHIP_SUPPORTED_TRUE             (0x00000000)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_CHIP_SUPPORTED_FALSE            (0x00000001)

#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_JTE                 0:0
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_JTE_FALSE                      (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_JTE_TRUE                       (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_LWSE                2:1
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_LWSE_TRUE                      (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_LWSE_FALSE                     (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_PPR                 4:3
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_PPR_FB_CLAMP                   (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_PPR_SEPARATE                   (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_PPR_SUSPEND                    (0x00000002)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_SRPR                5:5
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_SRPR_DEFAULT                   (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_SRPR_SUSPEND                   (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_FBPR                7:6
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_FBPR_MINIMAL                   (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_FBPR_SPLIT                     (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_FBPR_SUSPEND                   (0x00000002)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_FBPR_SPLIT_SUSPEND             (0x00000003)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_GPR                 9:8
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_GPR_COMBINED                   (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_GPR_SEPARATE                   (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_GCR                 10:10
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_GCR_SPI_ROM                    (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_GPR_INTEGRATED_ROM             (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_PTH                 11:11
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_PTH_SMI                        (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_PTH_NO_SMI                     (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_NOT                 11:11
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_NOT_CG                         (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_NOT_CRCG                       (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_MXRV                30:20


#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ROMLESS         0:0
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ROMLESS_TRUE                   (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ROMLESS_FALSE                  (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ROM             1:1
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ROM_TRUE                       (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ROM_FALSE                      (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_LWSR            2:2
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_LWSR_TRUE                      (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_LWSR_FALSE                     (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_TDR             3:3
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_TDR_TRUE                       (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_TDR_FALSE                      (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_SELFWAKE        4:4
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_SELFWAKE_TRUE                  (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_SELFWAKE_FALSE                 (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ISREXIT         5:5
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ISREXIT_TRUE                   (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_ISREXIT_FALSE                  (0x00000001)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_S3GC6           6:6
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_S3GC6_TRUE                     (0x00000000)
#define LW2080_CTRL_POWER_FEATURES_GC6_CAPS_S3GC6_FALSE                    (0x00000001)
#define LW2080_CTRL_POWER_FEATURE_GC6_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_POWER_FEATURE_GC6_INFO_PARAMS {
    LwU32 gc6InfoMask;
// VBIOS supports fbclamp deassert on gc6 exit
// VBIOS supports IFR on GC6 exit
// SBIOS supports GC6 functions
// Romless GC6 is enabled due to regkey
// Rom GC6 is enabled due to regkey
// GPU is using always-on power islands for GC6
// VBIOS supports GC6
// Chip supports GC6
    LwU32 lwsrInfoMask;
    // TODO. Add masks for LWSR

    LwU32 sbiosCapsMask;
    LwU32 gc6CapsMask;
} LW2080_CTRL_POWER_FEATURE_GC6_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_POWER_GET_GC6_REFCOUNT_STATE
 *
 * This command is used to retrieve the current GC6 refcount state
 *
 *  bRefcountHeld
 *    Whether refcount is held by RM
 *
 * Possible status return values are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_POWER_GET_GC6_REFCOUNT_STATE (0x20802704) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_POWER_GC6_GET_REFCOUNT_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_POWER_GC6_GET_REFCOUNT_STATE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_POWER_GC6_GET_REFCOUNT_STATE_PARAMS {
    LwBool bRefcountHeld;
} LW2080_CTRL_POWER_GC6_GET_REFCOUNT_STATE_PARAMS;

/*
 * LW2080_CTRL_CMD_GC6_PROGRAM_SCI_WAKEUP_TIMER
 *
 * This command is obsolete.
 * See LW2080_CTRL_CMD_GPU_POWER_ON_OFF to set wake timer for GC6.
 *
 */
#define LW2080_CTRL_CMD_GC6_PROGRAM_SCI_WAKEUP_TIMER (0x20802705) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | 0x5" */ // deprecated, removed from RM

typedef struct LW2080_CTRL_GC6_PROGRAM_SCI_WAKEUP_TIMER_PARAMS {
    LwU32 wakeupTimeUs;
} LW2080_CTRL_GC6_PROGRAM_SCI_WAKEUP_TIMER_PARAMS;

/*
 * LW2080_CTRL_CMD_GC6PLUS_IS_ISLAND_LOADED
 *
 * This command returns whether the GC6+ SCI/BSI islands are loaded or not.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_GC6PLUS_IS_ISLAND_LOADED (0x20802706) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GC6PLUS_IS_ISLAND_LOADED_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GC6PLUS_IS_ISLAND_LOADED_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_GC6PLUS_IS_ISLAND_LOADED_PARAMS {
    LwBool bIsIslandLoaded;
} LW2080_CTRL_GC6PLUS_IS_ISLAND_LOADED_PARAMS;

/*
* LW2080_CTRL_CMD_GCX_GET_WAKEUP_REASON
*
* This command returns the REASON of wakeup/abort for GC5/GC6M.
*    selectPowerState
*        The state for which wakeup reason(s) are being requested.
*    statId
*        RM fill the id the statistic going to be fetched. (count)
*
* Valid for both GC5 and GC6
*   sciIntrStatus0
*   sciIntrStatus1
*       The interrupt logged by SCI while we were in GCx state.
*       This is a verbatim copy of SCI_INTERRUPT_STATUS registers.
*
* if (selectPowerState == LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5MINUS_SSC)
*   gc5ExitType
*        Type of GC5 exit. Ie exit vs abort.
*   gc5AbortCode
*        Reason for abort.
*   pmcIntr0
*   pmcIntr1
*        Verbatim copy of PMC_INTR registers in the event of a GC5 abort.
*
* Possible status return values are:
* LW_OK
* LW_ERR_ILWALID_ARGUMENT
*/
#define LW2080_CTRL_CMD_GCX_GET_WAKEUP_REASON                                        (0x20802707) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GCX_GET_WAKEUP_REASON_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC6                                        0x0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5MINUS_SSC                               0x1

#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_EXIT                               0x0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT                              0x1

#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_WITH_DEEP_L1                       0x0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_WITH_L1_1                          0x1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_WITH_L1_2                          0x2

#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_DEEP_L1_ENTRY_TIMEOUT        0:0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_DEEP_L1_ENTRY_TIMEOUT_NO     0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_DEEP_L1_ENTRY_TIMEOUT_YES    1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_DEEP_L1_EXIT                 1:1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_DEEP_L1_EXIT_NO              0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_DEEP_L1_EXIT_YES             1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_MSCG_ABORT                   2:2
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_MSCG_ABORT_NO                0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_MSCG_ABORT_YES               1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_RTOS_ABORT                   3:3
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_RTOS_ABORT_NO                0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_RTOS_ABORT_YES               1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_AZALIA_ACTIVE                4:4
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_AZALIA_ACTIVE_NO             0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_AZALIA_ACTIVE_YES            1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_HOST_NOT_IDLE                5:5
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_HOST_NOT_IDLE_NO             0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_HOST_NOT_IDLE_YES            1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_PENDING_PMC_INTR             6:6
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_PENDING_PMC_INTR_NO          0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_PENDING_PMC_INTR_YES         1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_PENDING_SCI_INTR             7:7
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_PENDING_SCI_INTR_NO          0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_PENDING_SCI_INTR_YES         1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_THERM_I2CS_BUSY              8:8
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_THERM_I2CS_BUSY_NO           0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_THERM_I2CS_BUSY_YES          1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_SPELWLATIVE_PTIMER_ALARM     9:9
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_SPELWLATIVE_PTIMER_ALARM_NO  0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_SPELWLATIVE_PTIMER_ALARM_YES 1
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_LATE_DEEP_L1_EXIT            10:10
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_LATE_DEEP_L1_EXIT_NO         0
#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_GC5_SSC_ABORT_LATE_DEEP_L1_EXIT_YES        1

#define LW2080_CTRL_GCX_GET_WAKEUP_REASON_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_GCX_GET_WAKEUP_REASON_PARAMS {
    LwU32 selectPowerState;
    LwU32 statId;
    LwU8  gc5ExitType;
    LwU8  deepL1Type;
// SSC can be done with deepL1, L1.1, L1.2
    LwU16 gc5AbortCode;
// DeepL1 not entered within expected Timeout
// DeepL1 exit
// MSCG aborted
// RTOS sent Abort to GCX task
// Azalia Active
// Host not idle
// Pending PMC Interrupt
// Pending SCI Interrupt
// Therm I2C slave not Idle
// Spelwlative abort for ptimer alarm
// "Late" Deep L1 exit - after disabling internal events from causing deep L1 wakeups
    LwU32 sciIntr0;
    LwU32 sciIntr1;
    LwU32 pmcIntr0;
    LwU32 pmcIntr1;
} LW2080_CTRL_GCX_GET_WAKEUP_REASON_PARAMS;

/*
 * LW2080_CTRL_CMD_GC6_GET_WAKEUP_REASON_NO_LOCK
 *
 * This command returns the GC6 wake up reason.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_GC6_GET_WAKEUP_REASON_NO_LOCK            (0x20802708) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_UNKNOWN        0
#define LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_TIMER_EXPIRED  1
#define LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_DEVICE_HOTPLUG 2
#define LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_DRIVER         3
#define LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_PARAMS {
    LwU8 wakeupReason;
} LW2080_CTRL_GC6_GET_WAKEUP_REASON_NO_LOCK_PARAMS;

/*
 * LW2080_CTRL_CMD_POWER_GET_POWER_CYCLE_LIMIT
 *
 * This command returns the target number of power cycles per year
 * based on pre-process limits for fuse block reads
 *
 * Possible status return values are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_POWER_GET_POWER_CYCLE_LIMIT (0x20802709) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_POWER_GET_POWER_CYCLE_LIMIT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_POWER_GET_POWER_CYCLE_LIMIT_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_POWER_GET_POWER_CYCLE_LIMIT_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 numPowerCyclesPerYear, 8);
} LW2080_CTRL_POWER_GET_POWER_CYCLE_LIMIT_PARAMS;

#define LW2080_CTRL_CMD_GET_RTD3_INFO (0x2080270a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GET_RTD3_INFO_PARAMS_MESSAGE_ID" */
/*
 * LW2080_CTRL_CMD_GET_RTD3_INFO
 *
 * This command returns the RTD3 support information from VBIOS
 * Including both RTD3-GC6 and RTD3-GCOFF
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE if it's ilwokded before driver state load
 */
#define LW2080_CTRL_CMD_GET_RTD3_INFO_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_CMD_GET_RTD3_INFO_PARAMS {

    // TODO-SC use bit mask
    LwBool bIsRTD3SupportedByChip;
    LwBool bIsRTD3GC6SupportedByChip;
    LwBool bIsRTD3GCOffSupportedByChip;
    LwBool bIsRTD3VoltageSourceSwitchSupported;
    LwBool bIsRTD3IFRPathSupported;
    LwBool bIsRTD3SupportedInUnix;

    struct {
        LwBool bIsRTD3GC6TotalBoardPowerUnitMA;
        LwU16  RTD3GC6TotalBoardPower;
        LwU16  RTD3GC6PerstDelay;
    } rtd3gc6;

    struct {
        LwBool bIsRTD3GCOffTotalBoardPowerUnitMA;
        LwU16  RTD3GCOffTotalBoardPower;
    } rtd3gcoff;
} LW2080_CTRL_CMD_GET_RTD3_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GPU_POWER_PRE_POST_POWER_ON_OFF
 *
 * This command performs the pre-powerOff or post-powerOn actions on the GPU.
 * For example, before powering off the GPU, lwlinks (if any available and if
 * support L2 state) should be transitioned to L2 state and after powering on
 * the GPU, they should be transitioned back to L0 state
 *
 * action   Whether to perform the steps before powering off the GPU
 *          of to perform the steps after powering on the GPU
 *
 * Possible status return values are:
 *   LW_OK                            Success
 *   LW_ERR_ILWALID_ARGUMENT          Invalid action
 *   LW_WARN_MORE_PROCESSING_REQUIRED Power state transition in progress
 *
 */
#define LW2080_CTRL_CMD_GPU_POWER_PRE_POST_POWER_ON_OFF (0x2080270b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GPU_POWER_PRE_POST_POWER_ON_OFF_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPU_POWER_PRE_POST_POWER_ON_OFF_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_GPU_POWER_PRE_POST_POWER_ON_OFF_PARAMS {
    LwU32 action;
} LW2080_CTRL_GPU_POWER_PRE_POST_POWER_ON_OFF_PARAMS;

// Possible values for action
#define LW2080_CTRL_GPU_POWER_PRE_POWER_OFF (0x00000001)
#define LW2080_CTRL_GPU_POWER_POST_POWER_ON (0x00000002)

/*!
 * @brief GC6 flavor ids
 */
typedef enum LW2080_CTRL_GC6_FLAVOR_ID {
    LW2080_CTRL_GC6_FLAVOR_ID_MSHYBRID = 0,
    LW2080_CTRL_GC6_FLAVOR_ID_OPTIMUS = 1,
    LW2080_CTRL_GC6_FLAVOR_ID_LWSR = 2,
    LW2080_CTRL_GC6_FLAVOR_ID_LWSR_FGC6 = 3,
    LW2080_CTRL_GC6_FLAVOR_ID_MAX = 4,
} LW2080_CTRL_GC6_FLAVOR_ID;

/*!
 * @brief GC6 step ids
 */
typedef enum LW2080_CTRL_GC6_STEP_ID {
    LW2080_CTRL_GC6_STEP_ID_SR_ENTRY = 0,
    LW2080_CTRL_GC6_STEP_ID_GPU_OFF = 1,
    LW2080_CTRL_GC6_STEP_ID_MAX = 2,
} LW2080_CTRL_GC6_STEP_ID;

typedef struct LW2080_CTRL_GC6_FLAVOR_INFO {
    LW2080_CTRL_GC6_FLAVOR_ID flavorId;
    LwU32                     stepMask;
} LW2080_CTRL_GC6_FLAVOR_INFO;

/*
 * LW2080_CTRL_CMD_GET_GC6_FLAVOR_INFO
 *
 * This command returns the step mask of the requested GC6 flavors
 *
 * Possible status return values are:
 *   LW_OK     Success
 */
#define LW2080_CTRL_CMD_GET_GC6_FLAVOR_INFO (0x2080270c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GET_GC6_FLAVOR_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GET_GC6_FLAVOR_INFO_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW2080_CTRL_GET_GC6_FLAVOR_INFO_PARAMS {
    LW2080_CTRL_GC6_FLAVOR_INFO flavorInfo[LW2080_CTRL_GC6_FLAVOR_ID_MAX];
} LW2080_CTRL_GET_GC6_FLAVOR_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GC6_ENTRY
 *
 * This command exelwtes the steps of GC6 entry sequence
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED (non-fatal)
 *   LW_ERR_ILWALID_STATE (non-fatal)
 *   LW_ERR_ILWALID_ARGUMENT (non-fatal)
 *   LW_ERR_NOT_READY (non-fatal)
 *   LW_ERR_TIMEOUT
 *   LW_ERR_GENERIC
 */
#define LW2080_CTRL_CMD_GC6_ENTRY (0x2080270d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GC6_ENTRY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GC6_ENTRY_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW2080_CTRL_GC6_ENTRY_PARAMS {
    LW2080_CTRL_GC6_FLAVOR_ID flavorId;
    LwU32                     stepMask;
    struct {
        LwBool bUseHwTimer;
        LwU32  timeToWakeUs;
        LwBool bIsRTD3Transition;
        LwBool bIsRTD3CoreRailPowerLwt;
        LwBool bIsHdaOSPowerManaged;
        LwBool bSkipPstateSanity;
    } params;
} LW2080_CTRL_GC6_ENTRY_PARAMS;


/*
 * LW2080_CTRL_CMD_GC6_EXIT
 *
 * This command exelwtes the steps of GC6 exit sequence
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_GENERIC
 */
#define LW2080_CTRL_CMD_GC6_EXIT (0x2080270e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_GC6_EXIT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GC6_EXIT_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW2080_CTRL_GC6_EXIT_PARAMS {
    LW2080_CTRL_GC6_FLAVOR_ID flavorId;
    struct {
        LwBool bIsGpuSelfWake;
        LwBool bIsRTD3Transition;
        LwBool bIsHdaOSPowerManaged;
        LwBool bIsRTD3HotTransition;   //output
    } params;
} LW2080_CTRL_GC6_EXIT_PARAMS;

/*
 * LW2080_CTRL_CMD_SET_GC6_RTD3_INFO
 *
 * This command provides a interface to overwrite RM RTD3 settings
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_SET_GC6_RTD3_INFO            (0x2080270f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_POWER_INTERFACE_ID << 8) | LW2080_CTRL_CMD_SET_GC6_RTD3_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_SET_RTD3_TEST_PARAM_ENABLE                  0:0
#define LW2080_CTRL_SET_RTD3_TEST_PARAM_ENABLE_FALSE (0x00000000)
#define LW2080_CTRL_SET_RTD3_TEST_PARAM_ENABLE_TRUE  (0x00000001)
#define LW2080_CTRL_SET_RTD3_TEST_FORCE_D3HOT                   1:1
#define LW2080_CTRL_SET_RTD3_TEST_FORCE_D3HOT_FALSE  (0x00000000)
#define LW2080_CTRL_SET_RTD3_TEST_FORCE_D3HOT_TRUE   (0x00000001)
#define LW2080_CTRL_CMD_SET_GC6_RTD3_INFO_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW2080_CTRL_CMD_SET_GC6_RTD3_INFO_PARAMS {
    LwBool bIsRTD3SupportedBySystem;
    LwU8   RTD3D3HotTestParams;
} LW2080_CTRL_CMD_SET_GC6_RTD3_INFO_PARAMS;

/* _ctrl2080power_h_ */

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

