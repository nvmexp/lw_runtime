/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080perf.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "lwfixedtypes.h"
#include "ctrl/ctrl2080/ctrl2080base.h"


#include "lwmisc.h"
//
// XAPICHK/XAPI_TEST chokes on the "static LWINLINE" defines in lwmisc.h.
// However, we don't need any of those definitions for those tests (XAPICHK is a
// syntactical check, not a functional test).  So, instead, just #define out the
// macros referenced below.
//
#include "ctrl/ctrl2080/ctrl2080clk.h"
#include "ctrl/ctrl2080/ctrl2080gpumon.h"
#include "ctrl/ctrl2080/ctrl2080volt.h"
#include "ctrl/ctrl2080/ctrl2080vfe.h"
#include "ctrl/ctrl2080/ctrl2080pmumon.h"
#include "ctrl/ctrl0080/ctrl0080perf.h"
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "ctrl/ctrl2080/ctrl2080perf_opaque_non_privileged.h"
#include "ctrl/ctrl2080/ctrl2080perf_opaque_privileged.h"
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW_SUBPROC_NAME_MAX_LENGTH          100

/* LW20_SUBDEVICE_XX perf control commands and parameters */

/*
 * LW2080_CTRL_CMD_PERF_GET_TABLE_INFO
 *
 * This command is obsolete.
 * Please use LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO.
 */
#define LW2080_CTRL_CMD_PERF_GET_TABLE_INFO (0x20802001) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_TABLE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_TABLE_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_PERF_GET_TABLE_INFO_PARAMS {
    LwU32 flags;
    LwU32 numLevels;
    LwU32 numPerfClkDomains;
    LwU32 perfClkDomains;
} LW2080_CTRL_PERF_GET_TABLE_INFO_PARAMS;
#define LW2080_CTRL_PERF_TABLE_FLAGS_MAXPERF                       0:0
#define LW2080_CTRL_PERF_TABLE_FLAGS_MAXPERF_NOT_SUPPORTED (0x00000000)
#define LW2080_CTRL_PERF_TABLE_FLAGS_MAXPERF_SUPPORTED     (0x00000001)

/*
 * LW2080_CTRL_CMD_PERF_GET_LEVEL_INFO
 * LW2080_CTRL_CMD_PERF_SET_LEVEL_INFO
 * LW2080_CTRL_CMD_PERF_TEST_LEVEL
 *
 * These commands are obsolete.
 * Please use LW2080_CTRL_CMD_PERF_GET_PSTATE2_INFO_V2
 * and LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO_V2.
 */
typedef struct LW2080_CTRL_PERF_GET_CLK_INFO {
    LwU32 flags;
    LwU32 domain;
    LwU32 lwrrentFreq;
    LwU32 defaultFreq;
    LwU32 minFreq;
    LwU32 maxFreq;
} LW2080_CTRL_PERF_GET_CLK_INFO;
#define LW2080_CTRL_CMD_PERF_GET_LEVEL_INFO (0x20802002) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_LEVEL_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_LEVEL_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_PERF_GET_LEVEL_INFO_PARAMS {
    LwU32 level;
    LwU32 flags;
    LW_DECLARE_ALIGNED(LwP64 perfGetClkInfoList, 8);
    LwU32 perfGetClkInfoListSize;
} LW2080_CTRL_PERF_GET_LEVEL_INFO_PARAMS;
#define LW2080_CTRL_PERF_GET_LEVEL_INFO_FLAGS_TYPE                 0:0
#define LW2080_CTRL_PERF_GET_LEVEL_INFO_FLAGS_TYPE_DEFAULT   (0x00000000)
#define LW2080_CTRL_PERF_GET_LEVEL_INFO_FLAGS_TYPE_OVERCLOCK (0x00000001)
#define LW2080_CTRL_PERF_GET_LEVEL_INFO_FLAGS_MODE                 2:1
#define LW2080_CTRL_PERF_GET_LEVEL_INFO_FLAGS_MODE_NONE      (0x00000000)
#define LW2080_CTRL_PERF_GET_LEVEL_INFO_FLAGS_MODE_DESKTOP   (0x00000001)
#define LW2080_CTRL_PERF_GET_LEVEL_INFO_FLAGS_MODE_MAXPERF   (0x00000002)
typedef struct LW2080_CTRL_PERF_SET_CLK_INFO {
    LwU32 domain;
    LwU32 targetFreq;
} LW2080_CTRL_PERF_SET_CLK_INFO;
#define LW2080_CTRL_CMD_PERF_SET_LEVEL_INFO (0x20802003) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_LEVEL_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_LEVEL_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_PERF_SET_LEVEL_INFO_PARAMS {
    LwU32 level;
    LwU32 flags;
    LW_DECLARE_ALIGNED(LwP64 perfSetClkInfoList, 8);
    LwU32 perfSetClkInfoListSize;
} LW2080_CTRL_PERF_SET_LEVEL_INFO_PARAMS;

#define LW2080_CTRL_CMD_PERF_TEST_LEVEL (0x20802004) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x4" */ // Deprecated (removed from RM)

typedef struct LW2080_CTRL_PERF_TEST_LEVEL_PARAMS {
    LwU32 level;
    LwU32 flags;
    LwU32 result;
    LW_DECLARE_ALIGNED(LwP64 perfSetClkInfoList, 8);
    LwU32 perfSetClkInfoListSize;
} LW2080_CTRL_PERF_TEST_LEVEL_PARAMS;

/*
 * LW2080_CTRL_PERF_MODE
 *
 * This set of macros is used to identify supported performance modes.
 *
 * Valid performance modes include:
 *   LW2080_CTRL_PERF_MODE_DESKTOP
 *   LW2080_CTRL_PERF_MODE_MAXPERF
 */
#define LW2080_CTRL_PERF_MODE_DESKTOP (0x00000001)
#define LW2080_CTRL_PERF_MODE_MAXPERF (0x00000002)

/*
 * LW2080_CTRL_CMD_PERF_GET_MODE
 *
 * This command returns the specified performance mode for the associated
 * subdevice.
 *
 *   flags
 *     This parameter is lwrrently unused.
 *   mode
 *     This parameter returns the current performance mode.
 *     Valid values for this parameter include:
 *       LW2080_CTRL_PERF_MODE_DESKTOP
 *         The current performance mode enables desktop performance level(s).
 *       LW2080_CTRL_PERF_MODE_MAXPERF
 *         The current performance mode enables maximum perforrmance level(s).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_GET_MODE (0x20802005) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_MODE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_PERF_GET_MODE_PARAMS {
    LwU32 flags;
    LwU32 mode;
} LW2080_CTRL_PERF_GET_MODE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_MODE
 *
 * This command enables the specified performance mode for the associated
 * subdevice.
 *
 *   flags
 *     This parameter sets any flags for the new performance mode.
 *     Legal values for this parameter include:
 *       LW2080_CTRL_PERF_SET_MODE_FLAGS_BOOST_ENABLE
 *         This flag specifies that the new performance mode should
 *         should be entered immediately.
 *       LW2080_CTRL_PERF_SET_MODE_FLAGS_BOOST_DISABLE
 *         This flag specifies that the new performance mode does not
 *         need to be entered immediately.
 *   mode
 *     This parameter specifies the desired new performance mode for the
 *     subdevice.  Valid values for this parameter include:
 *       LW2080_CTRL_PERF_MODE_DESKTOP
 *         This performance mode enables the desktop performance level(s)
 *         in the performance table.
 *       LW2080_CTRL_PERF_MODE_MAXPERF
 *         This performance mode enables the maximum performance level(s)
 *         in the performance table.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_SET_MODE (0x20802006) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_MODE_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_PERF_SET_MODE_PARAMS {
    LwU32 flags;
    LwU32 mode;
} LW2080_CTRL_PERF_SET_MODE_PARAMS;

/* valid flags values */
#define LW2080_CTRL_PERF_SET_MODE_FLAGS_BOOST                      0:0
#define LW2080_CTRL_PERF_SET_MODE_FLAGS_BOOST_DISABLE (0x00000000)
#define LW2080_CTRL_PERF_SET_MODE_FLAGS_BOOST_ENABLE  (0x00000001)

/*
 * LW2080_CTRL_CMD_PERF_GET_ACTIVE_CLOCKING
 * LW2080_CTRL_CMD_PERF_SET_ACTIVE_CLOCKING
 *
 * These commands are no longer supported.
 */
#define LW2080_CTRL_CMD_PERF_GET_ACTIVE_CLOCKING      (0x20802007) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x7" */

typedef struct LW2080_CTRL_PERF_GET_ACTIVE_CLOCKING_PARAMS {
    LwU32 flags;
} LW2080_CTRL_PERF_GET_ACTIVE_CLOCKING_PARAMS;
#define LW2080_CTRL_PERF_GET_ACTIVE_CLOCKING_FLAGS                 0:0
#define LW2080_CTRL_PERF_GET_ACTIVE_CLOCKING_FLAGS_DISABLED (0x00000000)
#define LW2080_CTRL_PERF_GET_ACTIVE_CLOCKING_FLAGS_ENABLED  (0x00000001)
#define LW2080_CTRL_CMD_PERF_SET_ACTIVE_CLOCKING            (0x20802008) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x8" */

typedef struct LW2080_CTRL_PERF_SET_ACTIVE_CLOCKING_PARAMS {
    LwU32 flags;
} LW2080_CTRL_PERF_SET_ACTIVE_CLOCKING_PARAMS;
#define LW2080_CTRL_PERF_SET_ACTIVE_CLOCKING_FLAGS                 0:0
#define LW2080_CTRL_PERF_SET_ACTIVE_CLOCKING_FLAGS_DISABLE (0x00000000)
#define LW2080_CTRL_PERF_SET_ACTIVE_CLOCKING_FLAGS_ENABLE  (0x00000001)

/*
 * LW2080_CTRL_CMD_PERF_GET_LWRRENT_LEVEL
 *
 * This command is obsolete.
 * Please use LW2080_CTRL_CMD_PERF_GET_LWRRENT_PSTATE.
 */
#define LW2080_CTRL_CMD_PERF_GET_LWRRENT_LEVEL             (0x20802009) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_LWRRENT_LEVEL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_LWRRENT_LEVEL_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_PERF_GET_LWRRENT_LEVEL_PARAMS {
    LwU32 lwrrLevel;
} LW2080_CTRL_PERF_GET_LWRRENT_LEVEL_PARAMS;


/*
 * LW2080_CTRL_CMD_PERF_BOOST
 *
 * This command can be used to boost P-State up one level or to the highest for a limited
 * duration for the associated subdevice. Boosts from different clients are being tracked
 * independently. Note that there are other factors that can limit P-States so the resulting
 * P-State may differ from expectation.
 *
 *   flags
 *     This parameter specifies the actual command. _CLEAR is to clear existing boost.
 *     _BOOST_1LEVEL is to boost P-State one level higher. _BOOST_TO_MAX is to boost
 *     to the highest P-State.
 *   duration
 *     This parameter specifies the duration of the boost in seconds. This has to be less
 *     than LW2080_CTRL_PERF_BOOST_DURATION_MAX.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_PERF_BOOST_FLAGS_CMD                1:0
#define LW2080_CTRL_PERF_BOOST_FLAGS_CMD_CLEAR        (0x00000000)
#define LW2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_1LEVEL (0x00000001)
#define LW2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX (0x00000002)

#define LW2080_CTRL_PERF_BOOST_FLAGS_LWDA               4:4
#define LW2080_CTRL_PERF_BOOST_FLAGS_LWDA_NO          (0x00000000)
#define LW2080_CTRL_PERF_BOOST_FLAGS_LWDA_YES         (0x00000001)

#define LW2080_CTRL_PERF_BOOST_FLAGS_ASYNC              5:5
#define LW2080_CTRL_PERF_BOOST_FLAGS_ASYNC_NO         (0x00000000)
#define LW2080_CTRL_PERF_BOOST_FLAGS_ASYNC_YES        (0x00000001)

#define LW2080_CTRL_PERF_BOOST_DURATION_MAX           3600 //The duration can be specified up to 1 hour
#define LW2080_CTRL_PERF_BOOST_DURATION_INFINITE      0xffffffff // If set this way, the boost will last until cleared.

#define LW2080_CTRL_CMD_PERF_BOOST                    (0x2080200a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_BOOST_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_BOOST_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_PERF_BOOST_PARAMS {
    LwU32 flags;
    LwU32 duration;
} LW2080_CTRL_PERF_BOOST_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_LWRRENT_LEVEL
 *
 * This command is obsolete.
 * Please use LW2080_CTRL_CMD_PERF_SET_FORCE_PSTATE_EX.
 */
#define LW2080_CTRL_CMD_PERF_SET_LWRRENT_LEVEL (0x20802010) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_LWRRENT_LEVEL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_LWRRENT_LEVEL_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_PERF_SET_LWRRENT_LEVEL_PARAMS {
    LwU32 lwrrLevel;
} LW2080_CTRL_PERF_SET_LWRRENT_LEVEL_PARAMS;


/*
 * LW2080_CTRL_CMD_PERF_GET_CLK_CTRL
 * LW2080_CTRL_CMD_PERF_SET_CLK_CTRL
 *
 * These commands are no longer supported.
 */
#define LW2080_CTRL_PERF_CLK_CTRL_GRAPHICS                         0:0
#define LW2080_CTRL_PERF_CLK_CTRL_GRAPHICS_DISABLED   (0x00000000)
#define LW2080_CTRL_PERF_CLK_CTRL_GRAPHICS_ENABLED    (0x00000001)
#define LW2080_CTRL_PERF_CLK_CTRL_MEMORY                           1:1
#define LW2080_CTRL_PERF_CLK_CTRL_MEMORY_DISABLED     (0x00000000)
#define LW2080_CTRL_PERF_CLK_CTRL_MEMORY_ENABLED      (0x00000001)
#define LW2080_CTRL_PERF_CLK_CTRL_THERMAL                          2:2 // Deprecated (NJ-TODO)
#define LW2080_CTRL_PERF_CLK_CTRL_THERMAL_DISABLED    (0x00000000)
#define LW2080_CTRL_PERF_CLK_CTRL_THERMAL_ENABLED     (0x00000001)
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_GRAPHICS                  4:4
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_GRAPHICS_YES (0x00000000)
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_GRAPHICS_NO  (0x00000001)
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_MEMORY                    5:5
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_MEMORY_YES   (0x00000000)
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_MEMORY_NO    (0x00000001)
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_THERMAL                   6:6
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_THERMAL_YES  (0x00000000)
#define LW2080_CTRL_PERF_CLK_CTRL_SELECT_THERMAL_NO   (0x00000001)
#define LW2080_CTRL_CMD_PERF_GET_CLK_CTRL             (0x20802020) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_CLK_CTRL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_CLK_CTRL_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW2080_CTRL_PERF_GET_CLK_CTRL_PARAMS {
    LwU32 clkCtrl;
} LW2080_CTRL_PERF_GET_CLK_CTRL_PARAMS;
#define LW2080_CTRL_CMD_PERF_SET_CLK_CTRL (0x20802021) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_CLK_CTRL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_CLK_CTRL_PARAMS_MESSAGE_ID (0x21U)

typedef struct LW2080_CTRL_PERF_SET_CLK_CTRL_PARAMS {
    LwU32 clkCtrl;
} LW2080_CTRL_PERF_SET_CLK_CTRL_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_CORE_VOLTAGE_STATUS
 * LW2080_CTRL_CMD_PERF_SET_CORE_VOLTAGE_STATUS
 *
 * These commands are no longer supported.
 */
#define LW2080_CTRL_PERF_CORE_VOLTAGE_STATUS                       0:0
#define LW2080_CTRL_PERF_CORE_VOLTAGE_STATUS_DISABLED (0x00000000)
#define LW2080_CTRL_PERF_CORE_VOLTAGE_STATUS_ENABLED  (0x00000001)
#define LW2080_CTRL_CMD_PERF_GET_CORE_VOLTAGE_STATUS  (0x20802030) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_CORE_VOLTAGE_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_CORE_VOLTAGE_STATUS_PARAMS_MESSAGE_ID (0x30U)

typedef struct LW2080_CTRL_PERF_GET_CORE_VOLTAGE_STATUS_PARAMS {
    LwU32 lwrrStatus;
} LW2080_CTRL_PERF_GET_CORE_VOLTAGE_STATUS_PARAMS;
#define LW2080_CTRL_CMD_PERF_SET_CORE_VOLTAGE_STATUS (0x20802031) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_CORE_VOLTAGE_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_CORE_VOLTAGE_STATUS_PARAMS_MESSAGE_ID (0x31U)

typedef struct LW2080_CTRL_PERF_SET_CORE_VOLTAGE_STATUS_PARAMS {
    LwU32 lwrrStatus;
    LwU32 newStatus;
} LW2080_CTRL_PERF_SET_CORE_VOLTAGE_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_SAMPLED_VOLTAGE_INFO
 *
 * This command can be used to get sampled voltage information for the
 * associated subdevice.
 *
 *   voltage
 *     This parameter returns the sampled voltage (in millivolts) for
 *     the subdevice.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_GET_SAMPLED_VOLTAGE_INFO (0x20802033) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_SAMPLED_VOLTAGE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_SAMPLED_VOLTAGE_INFO_PARAMS_MESSAGE_ID (0x33U)

typedef struct LW2080_CTRL_PERF_GET_SAMPLED_VOLTAGE_INFO_PARAMS {
    LwU32 voltage;
} LW2080_CTRL_PERF_GET_SAMPLED_VOLTAGE_INFO_PARAMS;

/*
 * LW2080_CTRL_PERF_PERFMON_SAMPLE
 *
 * This structure describes per-clock domain information obtained with
 * the LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE command.
 *
 *   clkDomain
 *     This parameter specifies the performance clock domain.  Legal
 *     values include:
 *       LW2080_CTRL_CLK_DOMAIN_LWCLK
 *       LW2080_CTRL_CLK_DOMAIN_MCLK
 *   clkPercentBusy
 *     This parameter contains the percentage during the sample that
 *     the clock remains busy.
 *   clkPercentIncr
 *     This parameter contains the percentage during the sample that
 *     the clock is increasing.
 *   clkPercentDecr
 *     This parameter contains the percentage during the sample that
 *     the clock is decreasing.
 */
typedef struct LW2080_CTRL_PERF_PERFMON_SAMPLE {
    LwU32 clkDomain;
    LwU32 clkPercentBusy;
    LwU32 clkPercentIncr;
    LwU32 clkPercentDecr;
} LW2080_CTRL_PERF_PERFMON_SAMPLE;

#define LW2080_CTRL_PERF_PERFMON_SAMPLE_PERCENT_BUSY(sample) ((sample).clkPercentBusy)

/*
 * LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE
 *
 * This command can be used to obtain performance monitor samples from
 * the associated subdevice.
 * This command is not supported with SMC enabled.
 *
 *   clkSamplesListSize
 *      This parameter specifies the number of clock domains to be sampled.
 *   clkSamplesList
 *     This field specifies a pointer in the caller's address space to the
 *     buffer into which the performance monitor samples are to be returned.
 *     This buffer must be at least as big as clkSamplesListSize multiplied
 *     by the size of the LW2080_CTRL_PERF_PERFMON_SAMPLE structure.
 *   clkInfoListSize
 *      This parameter specifies the number of clock domains to be queried.
 *   clkInfoList
 *     This field specifies a pointer in the caller's address space to the
 *     buffer into which the clocks info to be returned.
 *     This buffer must be at least as big as clkSamplesListSize multiplied
 *     by the size of the LW2080_CTRL_CLK_INFO structure.
 *   samplingPeriodUs
 *     This field returns the sampling period in microseconds.
 *   clkInfoFlags
 *     This field specifies whether gpu clocks remained constant or changed
 *     during current sampling period.
 *     Legal values for this field include:
 *       LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_CLKINFOFLAGS_CHANGED_NO
 *         clocks remained constant during current sampling period.
 *       LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_CLKINFOFLAGS_CHANGED_YES
 *         clocks might have changed during current sampling period.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * Deprecated, please use LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE_V3.
 */
#define LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE    (0x20802040) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE_V2
 *
 * Adds clkInfoListSize, clkInfoList and clkInfoFlags fields to the original
 * PARAMS structure to get the current clks and a flag indicating that the
 * clocks might have changed.
 *
 * Deprecated, please use LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE_V3.
 */
#define LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE_V2 (0x20802094) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x94" */

/*
 * LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE_V3
 *
 * Same as LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE_V2 but without embedded pointers.
 */
#define LW2080_CTRL_CMD_PERF_GET_PERFMON_SAMPLE_V3 (0x20802097) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_V3_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_PARAMS_MESSAGE_ID (0x40U)

typedef struct LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_PARAMS {
    LwU32 clkSamplesListSize;
    LW_DECLARE_ALIGNED(LwP64 clkSamplesList, 8);
    LwU32 clkInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 clkInfoList, 8);
    LwU32 samplingPeriodUs;
    LwU32 clkInfoFlags;
} LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_PARAMS;

#define LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_V3_PARAMS_MESSAGE_ID (0x97U)

typedef struct LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_V3_PARAMS {
    LwU32                           clkSamplesListSize;
    LW2080_CTRL_PERF_PERFMON_SAMPLE clkSamplesList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
    LwU32                           clkInfoListSize;
    LW2080_CTRL_CLK_INFO            clkInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
    LwU32                           samplingPeriodUs;
    LwU32                           clkInfoFlags;
} LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_V3_PARAMS;

/* valid clocks info changed values */
#define LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_CLKINFOFLAGS_CHANGED_NO  (0x00000000)
#define LW2080_CTRL_PERF_GET_PERFMON_SAMPLE_CLKINFOFLAGS_CHANGED_YES (0x00000001)

/*
 * LW2080_CTRL_PERF_PERFMON_SENSOR_CNTR
 *
 * Describes per-sensor information obtained with the
 * LW2080_CTRL_CMD_PERF_GET_PERFMON_SENSOR_CNTR_INFO command.
 */

typedef struct LW2080_CTRL_PERF_PERFMON_SENSOR_CNTR {
    /*!
     * Out: perf clock domain
     */
    LwU32  clkDomain;
    /*!
     * Out: engines with which counter is associated
     */
    LwU32  engineMask;
    /*!
     * Out: utilization percentage
     */
    LwU32  percentBusy;
    /*!
     * Out: additional semantic information for counters tracking PCI-E util
     *      metrics. See LW2080_CTRL_PERF_PCIE_*  for possible values
     */
    LwU8   pcieType;
    /*!
     * Out: whether this entry corresponds to a valid counter or not
     */
    LwBool bIsPresent;
} LW2080_CTRL_PERF_PERFMON_SENSOR_CNTR;

/*!
 * Macros for additional semantic information for PCI-E utilization counters
 */
#define LW2080_CTRL_PERF_NON_PCIE_SENSOR_CNTR             0x00
#define LW2080_CTRL_PERF_PCIE_TX_SENSOR_CNTR              0x01
#define LW2080_CTRL_PERF_PCIE_RX_SENSOR_CNTR              0x02

/*
 * LW2080_CTRL_CMD_PERF_GET_PERFMON_SENSOR_CNTR_INFO
 *
 * This command can be used to obtain sensor counter information from
 * the associated sub-device.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_GET_PERFMON_SENSOR_CNTR_INFO (0x20802041) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PERFMON_SENSOR_CNTR_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PERFMON_SENSOR_CNTR_INFO_PARAMS_MESSAGE_ID (0x41U)

typedef struct LW2080_CTRL_PERF_GET_PERFMON_SENSOR_CNTR_INFO_PARAMS {
    /*!
     * In: Number of sensor counters in the list
     */
    LwU32 sensorCntrListSize;
    /*!
     * Out: pointer in the caller's address space to the buffer into which the
     *      sensor counter information has to be returned. This buffer must be
     *      at least as big as sensorCntrListSize multiplied by the size of
     *      the LW2080_CTRL_PERF_PERFMON_SENSOR_CNTR structure.
     */
    LW_DECLARE_ALIGNED(LwP64 sensorCntrList, 8);
    /*!
     * Out: Sampling Period in microseconds.
     */
    LwU32 samplingPeriodUs;
} LW2080_CTRL_PERF_GET_PERFMON_SENSOR_CNTR_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_PERFMON_MAX_SENSOR_CNTR
 *
 * This command can be used to obtain maximum number of sensor counters
 * available in the system
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW2080_CTRL_CMD_PERF_GET_PERFMON_MAX_SENSOR_CNTR (0x20802042) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PERFMON_MAX_SENSOR_CNTR_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PERFMON_MAX_SENSOR_CNTR_PARAMS_MESSAGE_ID (0x42U)

typedef struct LW2080_CTRL_PERF_GET_PERFMON_MAX_SENSOR_CNTR_PARAMS {
    /*!
     * Out: Maximum number of perf sensor counters supported
     */
    LwU32 maxSensorCntr;
} LW2080_CTRL_PERF_GET_PERFMON_MAX_SENSOR_CNTR_PARAMS;

/*
 * LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO
 *
 * This structure describes adaptive clock information:
 *
 *   select
 *     This parameter is obsolete.
 *     The status is uniform across all modes of operation.
 *   status
 *     This field specifies whether adaptive clocking is enabled or disabled.
 *     Legal values for this field include:
 *       LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED
 *         Adapative clocking is disabled.
 *       LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED
 *         Adapative clocking is enabled.
 *
 */
typedef struct LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_PARAMS {
    LwU32 select;
    LwU32 status;
} LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_PARAMS;

/* valid adaptive clocking info status values */
#define LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_STATUS_DISABLED (0x00000000)
#define LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED  (0x00000001)

/*
 * LW2080_CTRL_CMD_PERF_SET_ADAPTIVE_CLOCKING_INFO
 *
 * This command can be used to set adaptive clocking status.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_SET_ADAPTIVE_CLOCKING_INFO         (0x20802053) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_ADAPTIVE_CLOCKING_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_ADAPTIVE_CLOCKING_INFO_PARAMS_MESSAGE_ID (0x53U)

typedef struct LW2080_CTRL_PERF_SET_ADAPTIVE_CLOCKING_INFO_PARAMS {
    LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_PARAMS adaptiveClockingInfo;
} LW2080_CTRL_PERF_SET_ADAPTIVE_CLOCKING_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_ADAPTIVE_CLOCKING_INFO
 *
 * This command can be used to retrieve current adaptive clocking status.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_ADAPTIVE_CLOCKING_INFO (0x20802054) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_ADAPTIVE_CLOCKING_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_ADAPTIVE_CLOCKING_INFO_PARAMS_MESSAGE_ID (0x54U)

typedef struct LW2080_CTRL_PERF_GET_ADAPTIVE_CLOCKING_INFO_PARAMS {
    LW2080_CTRL_PERF_ADAPTIVE_CLOCKING_INFO_PARAMS adaptiveClockingInfo;
} LW2080_CTRL_PERF_GET_ADAPTIVE_CLOCKING_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_POWER_MIZER_INFO
 * LW2080_CTRL_CMD_PERF_GET_POWER_MIZER_INFO
 *
 * These commands are obsolete.
 * Please use LW2080_CTRL_CMD_PERF_SET_PSTATE_CLIENT_LIMITS
 * and LW2080_CTRL_CMD_PERF_GET_PSTATE_CLIENT_LIMITS.
 */
#define LW2080_CTRL_PERF_POWER_MIZER_SELECT_AC_HARD   (0x00000000)
#define LW2080_CTRL_PERF_POWER_MIZER_SELECT_AC_SOFT   (0x00000001)
#define LW2080_CTRL_PERF_POWER_MIZER_SELECT_BATT_HARD (0x00000002)
#define LW2080_CTRL_PERF_POWER_MIZER_SELECT_BATT_SOFT (0x00000003)
typedef struct LW2080_CTRL_PERF_POWER_MIZER_INFO_PARAMS {
    LwU32 select;
    LwU32 state;
} LW2080_CTRL_PERF_POWER_MIZER_INFO_PARAMS;
#define LW2080_CTRL_PERF_POWER_MIZER_MAX          (0x00000001)
#define LW2080_CTRL_PERF_POWER_MIZER_BALANCED     (0x00000002)
#define LW2080_CTRL_PERF_POWER_MIZER_MAX_BATT     (0x00000003)
#define LW2080_CTRL_CMD_PERF_SET_POWER_MIZER_INFO (0x20802055) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_POWER_MIZER_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_POWER_MIZER_INFO_PARAMS_MESSAGE_ID (0x55U)

typedef struct LW2080_CTRL_PERF_SET_POWER_MIZER_INFO_PARAMS {
    LW2080_CTRL_PERF_POWER_MIZER_INFO_PARAMS powerMizerInfo;
} LW2080_CTRL_PERF_SET_POWER_MIZER_INFO_PARAMS;
#define LW2080_CTRL_CMD_PERF_GET_POWER_MIZER_INFO (0x20802056) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_POWER_MIZER_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_POWER_MIZER_INFO_PARAMS_MESSAGE_ID (0x56U)

typedef struct LW2080_CTRL_PERF_GET_POWER_MIZER_INFO_PARAMS {
    LW2080_CTRL_PERF_POWER_MIZER_INFO_PARAMS powerMizerInfo;
} LW2080_CTRL_PERF_GET_POWER_MIZER_INFO_PARAMS;

/*
 * LW2080_CTRL_PERF_POWERSTATE
 *
 * This structure describes power state information.
 *
 *    powerState
 *       This parameter specifies the type of power source.
 *       Legal values for this parameter include:
 *          LW2080_CTRL_PERF_POWER_SOURCE_AC
 *             This values indicates that the power state is AC.
 *          LW2080_CTRL_PERF_POWER_SOURCE_BATTERY
 *             This values indicates that the power state is battery.
 */
#define LW2080_CTRL_PERF_POWER_SOURCE_AC      (0x00000000)
#define LW2080_CTRL_PERF_POWER_SOURCE_BATTERY (0x00000001)

typedef struct LW2080_CTRL_PERF_POWERSTATE_PARAMS {
    LwU32 powerState;
} LW2080_CTRL_PERF_POWERSTATE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_POWERSTATE
 *
 * This command can be used to find out whether the perf power state is AC/battery.
 *
 *    powerStateInfo
 *       This parameter specifies the power source type.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_PERF_GET_POWERSTATE (0x2080205a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_POWERSTATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_POWERSTATE_PARAMS_MESSAGE_ID (0x5AU)

typedef struct LW2080_CTRL_PERF_GET_POWERSTATE_PARAMS {
    LW2080_CTRL_PERF_POWERSTATE_PARAMS powerStateInfo;
} LW2080_CTRL_PERF_GET_POWERSTATE_PARAMS;


/*
 * LW2080_CTRL_CMD_PERF_SET_POWERSTATE
 *
 * This command can be used to set the perf power state as AC or battery.
 *
 *    powerStateInfo
 *       This parameter specifies the power source type to set.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_SET_POWERSTATE (0x2080205b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_POWERSTATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_POWERSTATE_PARAMS_MESSAGE_ID (0x5BU)

typedef struct LW2080_CTRL_PERF_SET_POWERSTATE_PARAMS {
    LW2080_CTRL_PERF_POWERSTATE_PARAMS powerStateInfo;
} LW2080_CTRL_PERF_SET_POWERSTATE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_NOTIFY_VIDEOEVENT
 *
 * This command can be used by video driver to notify RM concerning
 * performance related events.
 *
 *    videoEvent
 *       This parameter specifies the video event to notify.
 *       Legal values for this parameter include:
 *          LW2080_CTRL_PERF_VIDEOEVENT_STREAM_HD_START
 *          LW2080_CTRL_PERF_VIDEOEVENT_STREAM_HD_STOP
 *             These values indicate that a HD video stream (less than 4K)
 *             has started/stopped.
 *          LW2080_CTRL_PERF_VIDEOEVENT_STREAM_SD_START
 *          LW2080_CTRL_PERF_VIDEOEVENT_STREAM_SD_STOP
 *             These are now obsolete in new products as we no longer
 *             need to differentiate between SD and HD.
 *          LW2080_CTRL_PERF_VIDEOEVENT_STREAM_4K_START
 *          LW2080_CTRL_PERF_VIDEOEVENT_STREAM_4K_STOP
 *             These value indicates that a 4K video stream (3840x2160 pixels
 *             or higher) has started/stopped.
 *          LW2080_CTRL_PERF_VIDEOEVENT_OFA_START
 *          LW2080_CTRL_PERF_VIDEOEVENT_OFA_STOP
 *             These value indicates that Optical Flow Accelerator usage has
 *             started/stopped.
 *       The following flags may be or'd into the event value:
 *          LW2080_CTRL_PERF_VIDEOEVENT_FLAG_LINEAR_MODE
 *              The stream operates BSP/VP2 or MSVLD/MSPDEC communication in
 *              linear mode (default is ring mode).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_PERF_NOTIFY_VIDEOEVENT (0x2080205d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_NOTIFY_VIDEOEVENT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_NOTIFY_VIDEOEVENT_PARAMS_MESSAGE_ID (0x5DU)

typedef struct LW2080_CTRL_PERF_NOTIFY_VIDEOEVENT_PARAMS {
    LwU32 videoEvent;
} LW2080_CTRL_PERF_NOTIFY_VIDEOEVENT_PARAMS;

#define LW2080_CTRL_PERF_VIDEOEVENT_EVENT_MASK       (0x0000ffff)
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_HD_START  (0x00000001)
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_HD_STOP   (0x00000002)
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_START     LW2080_CTRL_PERF_VIDEOEVENT_STREAM_HD_START
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_STOP      LW2080_CTRL_PERF_VIDEOEVENT_STREAM_HD_STOP
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_SD_START  (0x00000003)
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_SD_STOP   (0x00000004)
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_4K_START  (0x00000005)
#define LW2080_CTRL_PERF_VIDEOEVENT_STREAM_4K_STOP   (0x00000006)
#define LW2080_CTRL_PERF_VIDEOEVENT_OFA_START        (0x00000007)
#define LW2080_CTRL_PERF_VIDEOEVENT_OFA_STOP         (0x00000008)
#define LW2080_CTRL_PERF_VIDEOEVENT_FLAG_LINEAR_MODE (0x00010000)

/*
 * LW2080_CTRL_CMD_PERF_VIDEO_GET_STATUS
 *
 * This command returns current status of notified video events.
 *    streamCount4k
 *        Number of 4K video streams (3840x2160+ pixels) that RM is aware of.
 *    streamCount1080p
 *        Number of less-than-4K video streams (SD, HD) that RM is aware of.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_PERF_VIDEO_GET_STATUS        (0x2080205e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_VIDEO_STATUS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_VIDEO_STATUS_MESSAGE_ID (0x5EU)

typedef struct LW2080_CTRL_PERF_VIDEO_STATUS {
    LwU32 streamCount4k;
    LwU32 streamCount1080p;
} LW2080_CTRL_PERF_VIDEO_STATUS;

/*
 * LW2080_CTRL_PERF_VOLTAGE_DOMAINS
 *
 * This is NOT supported on Pstate 3.0
 *
 * These are definitions of voltage domains relevant to perf.
 * NOT_SUPPORTED - return info that there is NO voltage domain support on Pstate3.0
 * CORE          - power supply to GPU
 * FB            - power supply to the video memory and FB interface
 * MAX           - arbitrary maximum of 16 voltages - this number needs to stay in sync
 *                 with LWAPI_MAX_GPU_PERF_VOLTAGES.
 *
 */
#define LW2080_CTRL_PERF_VOLTAGE_DOMAINS_UNDEFINED                   (0x00000000)
#define LW2080_CTRL_PERF_VOLTAGE_DOMAINS_CORE                        (0x00000001)
#define LW2080_CTRL_PERF_VOLTAGE_DOMAINS_FB                          (0x00000002)
#define LW2080_CTRL_PERF_VOLTAGE_DOMAINS_COLD_CORE                   (0x00000004)
#define LW2080_CTRL_PERF_VOLTAGE_DOMAINS_CORE_NOMINAL                (0x00000008)
#define LW2080_CTRL_PERF_VOLTAGE_DOMAINS_NOT_SUPPORTED               (0x00000000)

/*
 * LW2080_CTRL_PERF_VIRTUAL_PSTATES
 *
 * These are special definitions of virtual performance states values.
 * INDEX_ONLY means to reference the virtual P-state index (number) directly.
 * Not all special vPstates are available on a given system.
 */
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_INDEX_ONLY                  (0x00000000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_D2                          (0x00000001)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_D3                          (0x00000002)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_D4                          (0x00000004)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_D5                          (0x00000008)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_OVER_LWRRENT                (0x00000010)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_VRHOT                       (0x00000020)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_MAX_BATT                    (0x00000040)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_MAX_SLI                     (0x00000080)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_MAX_THERM_SUSTAIN           (0x00000100)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_BOOST                       (0x00000200)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_TURBO_BOOST                 (0x00000400)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_RATED_TDP                   (0x00000800)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_SLOWDOWN_PWR                (0x00001000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_MID_POINT                   (0x00002000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_INFLECTION0                 (0x00004000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_FULL_DEFLECTION             (0x00008000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_MAX_LWSTOMER_BOOST          (0x00010000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_INFLECTION1                 (0x00020000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_INFLECTION2                 (0x00040000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_WHISPER_MODE                (0x00080000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_DLPPM_1X_ESTIMATION_MINIMUM (0x00100000)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_DLPPC_1X_SEARCH_MINIMUM     (0x00200000)

/*
 * LW2080_CTRL_PERF_VIRTUAL_PSTATE_NUM
 *
 * These are definitions for virtual performance state numbers, mostly for
 * documentation.
 */
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_NUM_VP0                     (0)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_NUM_VP1                     (1)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_NUM_MAX                     (254)
#define LW2080_CTRL_PERF_VIRTUAL_PSTATES_NUM_SKIP                    (0x000000FF)

/*
 * LW2080_CTRL_PERF_PSTATES
 *
 * These are definitions of performance states (P-states) values.
 * P0 has the maximum performance capability and consumes maximum
 * power. P1 has a lower perf and power than P0, and so on.
 * For LWPU GPUs, the following definitions are made:
 * P0    - maximum 3D performance
 * P1    - original P0 when active clocked
 * P2-P3 - balanced 3D performance-power
 * P8    - basic HD video playback
 * P10   - SD video playback
 * P12   - minimum idle power
 * P15   - max possible P-state under current scheme (lwrrently not used)
 * Not all P-states are available on a given system.
 */
#define LW2080_CTRL_PERF_PSTATES_UNDEFINED                           (0x00000000)
#define LW2080_CTRL_PERF_PSTATES_CLEAR_FORCED                        (0x00000000)
#define LW2080_CTRL_PERF_PSTATES_P0                                  (0x00000001)
#define LW2080_CTRL_PERF_PSTATES_P1                                  (0x00000002)
#define LW2080_CTRL_PERF_PSTATES_P2                                  (0x00000004)
#define LW2080_CTRL_PERF_PSTATES_P3                                  (0x00000008)
#define LW2080_CTRL_PERF_PSTATES_P4                                  (0x00000010)
#define LW2080_CTRL_PERF_PSTATES_P5                                  (0x00000020)
#define LW2080_CTRL_PERF_PSTATES_P6                                  (0x00000040)
#define LW2080_CTRL_PERF_PSTATES_P7                                  (0x00000080)
#define LW2080_CTRL_PERF_PSTATES_P8                                  (0x00000100)
#define LW2080_CTRL_PERF_PSTATES_P9                                  (0x00000200)
#define LW2080_CTRL_PERF_PSTATES_P10                                 (0x00000400)
#define LW2080_CTRL_PERF_PSTATES_P11                                 (0x00000800)
#define LW2080_CTRL_PERF_PSTATES_P12                                 (0x00001000)
#define LW2080_CTRL_PERF_PSTATES_P13                                 (0x00002000)
#define LW2080_CTRL_PERF_PSTATES_P14                                 (0x00004000)
#define LW2080_CTRL_PERF_PSTATES_P15                                 (0x00008000)
#define LW2080_CTRL_PERF_PSTATES_MAX                                 LW2080_CTRL_PERF_PSTATES_P15
#define LW2080_CTRL_PERF_PSTATES_SKIP_ENTRY                          (0x10000) /* finn: Evaluated from "(LW2080_CTRL_PERF_PSTATES_MAX << 1)" */
#define LW2080_CTRL_PERF_PSTATES_ALL                                 (0xffff) /* finn: Evaluated from "(LW2080_CTRL_PERF_PSTATES_MAX | (LW2080_CTRL_PERF_PSTATES_MAX - 1))" */

/*
 * LW2080_CTRL_PERF_PSTATE_FALLBACK
 *
 * These are definitions of P-state fallback strategy, which determines
 * the P-state to use for the function call when the specified P-state
 * is not available.
 * RETURN_ERROR - do not fallback, just return error
 * HIGHER_PERF  - fallback to a higher perforamnce P-state, or to
 *                the P-state with the highest possible performance
 *                if no higher performance P-state is found.
 * LOWER_PERF   - fallback to a lower performance P-state, or to
 *                the P-state with the lowest possible performance
 *                if no lower performance P-state is found.
 */
#define LW2080_CTRL_PERF_PSTATE_FALLBACK_RETURN_ERROR                (0x00000000)
#define LW2080_CTRL_PERF_PSTATE_FALLBACK_HIGHER_PERF                 (0x00000001)
#define LW2080_CTRL_PERF_PSTATE_FALLBACK_LOWER_PERF                  (0x00000002)

/*
 * LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO
 *
 * This command returns general performance states information for the
 * associated subdevice. This information includes the available
 * P-states and supported clock and voltage domains.
 *
 *   flags
 *     This parameter returns P-states flags.
 *     The valid P-states flag values are
 *     LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PERFMON
 *         This flag indicates if perfmon is enabled.
 *     LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYN_PSTATE_CAPABLE
 *         This flag indicates if Dynamic P-State is enabled.
 *     LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYNAMIC_PSTATE
 *     LW2080_CTRL_PERF_GET_PSTATES_FLAGS_ASLM
 *         This flag indicates if ASLM is enabled in RM, _ENABLE_UPONLY means
 *         RM is limited to increase link width but not decrease.
 *     LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PEXSPEED_CHANGE
 *         This flag indicates if RM would attempt PCI-E speed change per P-State.
 *   pstates
 *     This parameter returns the mask of available P-states. Each enabled
 *     bit in the mask represents the availability of the corresponding
 *     P-state, as defined in LW2080_CTRL_PERF_PSTATES values.
 *   virtualPstates
 *     This parameter returns the mask of available special vPstates. Each enabled
 *     bit in the mask represents the availability of the corresponding
 *     special vPstate, as defined in LW2080_CTRL_PERF_VIRTUAL_PSTATES values.
 *   virtualPstatesNum
 *     This parameter returns the total number of vPstates.
 *   perfClkDomains
 *     This parameter returns the mask of supported clock domains. Each
 *     enabled bit in the mask represents programming support for the corresponding
 *     clock domain, as defined in LW2080_CTRL_CLK_DOMAIN values
 *     in ctrl2080clk.h.
 *   perfVoltageDomains
 *     This parameter returns the mask of supported voltage domains. Each
 *     enabled bit in the mask represents the support for the corresponding
 *     voltage domain, as defined in LW2080_CTRL_PERF_VOLTAGE_DOMAINS values.
 *   perfClkDomainsReported
 *     This parameter returns the mask of supported clock domains. Each
 *     enabled bit in the mask represents reporting support for the corresponding
 *     clock domain, as defined in LW2080_CTRL_CLK_DOMAIN values
 *     in ctrl2080clk.h.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO                        (0x20802060) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PSTATES_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PSTATES_INFO_PARAMS_MESSAGE_ID (0x60U)

typedef struct LW2080_CTRL_PERF_GET_PSTATES_INFO_PARAMS {
    LwU32 flags;
    LwU32 pstates;
    LwU32 virtualPstates;
    LwU32 virtualPstatesNum;
    LwU32 perfClkDomains;
    LwU32 perfVoltageDomains;
    LwU32 perfClkDomainsReported;
} LW2080_CTRL_PERF_GET_PSTATES_INFO_PARAMS;

#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PERFMON                        0:0
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PERFMON_DISABLED        (0x00000000)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PERFMON_ENABLED         (0x00000001)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYN_PSTATE_CAPABLE             1:1
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYN_PSTATE_CAPABLE_OFF  (0x00000000)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYN_PSTATE_CAPABLE_ON   (0x00000001)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYNAMIC_PSTATE                 2:2
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYNAMIC_PSTATE_DISABLE  (0x00000000)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_DYNAMIC_PSTATE_ENABLE   (0x00000001)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_ASLM                           4:3
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_ASLM_DISABLE            (0x00000000)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_ASLM_ENABLE_UPONLY      (0x00000001)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_ASLM_ENABLE_BOTH        (0x00000002)

#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PEXSPEED_CHANGE                  5:5
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PEXSPEED_CHANGE_DISABLE (0x00000000)
#define LW2080_CTRL_PERF_GET_PSTATES_FLAGS_PEXSPEED_CHANGE_ENABLE  (0x00000001)

/*
 * Structure describing deviation of given parameter (voltage, frequency, ...)
 * from it's nominal value.
 */
typedef struct LW2080_CTRL_PERF_PARAM_DELTA {
    LwS32 value;  //<! deviation of current settings from nominal value
    struct {
        LwS32 min;    //<! min allowed value of this deviation
        LwS32 max;    //<! max allowed value of this deviation
    } valueRange;
} LW2080_CTRL_PERF_PARAM_DELTA;

/*
 * LW2080_CTRL_PERF_CLK_DOM_INFO
 *
 * This structure describes the clock domains related to performance.
 *
 *   domain
 *     This parameter specifies/returns the clock domain .
 *     as defined in LW2080_CTRL_CLK_DOMAIN.
 *   flags
 *     This parameter specifies/returns the clock flags.
 *       LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_PLL
 *         This flag indicates that the clock source should
 *         be forced to a PLL (and not the bypass clock source).
 *       LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_BYPASS
 *         This flag indicates that the clock source should
 *         be forced to the bypass clock source.
 *       LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_UPDATED
 *         This flag indicates that the clock info has been updated.
 *         Used by LW2080_CTRL_CMD_PERF_GET_RATIO_CLK_FREQ.
 *       LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_USE_DELTA
 *         This flag indicates that field freqDelta is populated and that it
 *         should be used within SET call(s) instead of absolute frequency.
 *   freq
 *     This parameter specifies/returns the clock frequency in kHz.
 *   freqDelta
 *     This parameter specifies offset of 'freq' from nominal CLK settings in kHz.
 */
typedef struct LW2080_CTRL_PERF_CLK_DOM_INFO {
    LwU32 domain;
    LwU32 flags;
    LwU32 freq;
    LwS32 freqDelta;
} LW2080_CTRL_PERF_CLK_DOM_INFO;

#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_PLL              0:0
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_PLL_DISABLE    (0x00000000)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_PLL_ENABLE     (0x00000001)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_BYPASS           1:1
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_BYPASS_DISABLE (0x00000000)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_BYPASS_ENABLE  (0x00000001)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_UPDATED                2:2
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_UPDATED_NO           (0x00000000)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_UPDATED_YES          (0x00000001)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_USE_DELTA              3:3
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_USE_DELTA_NO         (0x00000000)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_USE_DELTA_YES        (0x00000001)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_NAFLL            4:4
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_NAFLL_DISABLE  (0x00000000)
#define LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_FORCE_NAFLL_ENABLE   (0x00000001)

/*
 * LW2080_CTRL_PERF_CLK_DOM2_INFO
 *
 * This structure describes the clock domains related to P-states 2.x,
 * and represents the extension of LW2080_CTRL_PERF_CLK_DOM_INFO structure.
 *
 *   domain
 *     This parameter specifies the clock domain as defined in
 *     LW2080_CTRL_CLK_DOMAIN.
 *   flags
 *     This parameter specifies/returns the clock flags.
 *       LW2080_CTRL_PERF_CLK_DOM2_INFO_FLAGS_USAGE
 *         This field defines clock domain usage. Following defines are re-used:
 *           LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_<xyz>
 *   minFreq
 *     This parameter returns the min clock frequency in kHz.
 *   maxFreq
 *     This parameter returns the max clock frequency in kHz.
 */
typedef struct LW2080_CTRL_PERF_CLK_DOM2_INFO {
    LwU32 domain;
    LwU32 flags;
    LwU32 minFreq;
    LwU32 maxFreq;
} LW2080_CTRL_PERF_CLK_DOM2_INFO;

#define LW2080_CTRL_PERF_CLK_DOM2_INFO_FLAGS_USAGE                 2:0

/*!
 * LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_LOGICAL
 *
 * This structure describes a logical voltage (in uV) corresponding to a voltage
 * domains, as related to performance.
 *
 *   logicalVoltageuV
 *     This parameter specifies the logical voltage (in uV) corresponding to
 *     this voltage domain.
 */
typedef struct LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_LOGICAL {
    LwU32 logicalVoltageuV;
} LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_LOGICAL;

/*!
 * LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VDT
 *
 * This structure describes a VDT (Voltage Descriptor Table) index corresponding
 * to a voltage domains, as related to performance.
 *
 *   vdtIndex
 *     This parameter specifies the VDT index corresponding to this voltage
 *     domain.
 */
typedef struct LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VDT {
    LwU8 vdtIndex;
} LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VDT;

/*!
 * LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VFE
 *
 * This structure describes a VFE Equation index corresponding
 * to a voltage domains, as related to performance.
 *
 *   vfeEquIndex
 *     This parameter specifies the VFE index corresponding to this voltage
 *     domain.
 */
typedef struct LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VFE {
    LW2080_CTRL_PERF_VFE_EQU_IDX vfeEquIndex;
} LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VFE;

/*!
 * Macros defining the frequency type that needs to be selected for given Pstate
 * index and voltage domain.
 */
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_PSTATE_FREQ_TYPE_MIN 0x01
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_PSTATE_FREQ_TYPE_MAX 0x02
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_PSTATE_FREQ_TYPE_NOM 0x03

/*!
 * LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_PSTATE
 *
 * This structure describes a PSTATE index corresponding to a voltage domains,
 * as related to performance.
 *
 *   pstateIndex
 *     This parameter specifies the PSTATE index corresponding to this voltage
 *     domain.
 *   freqType
 *     This parameter specifies the frequency type
 *     ref@ LW2080_CTRL_PERF_VOLT_DOM_INFO_PSTATE_FREQ_TYPE_(XYZ)
 *     corresponding to this voltage domain.
 */
typedef struct LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_PSTATE {
    LwU8 pstateIndex;
    LwU8 freqType;
} LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_PSTATE;

/*!
 * LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VPSTATE
 *
 * This structure describes a VPSTATE index corresponding to a voltage domains,
 * as related to performance.
 *
 *   vpstateIndex
 *     This parameter specifies the Virtual Pstate index corresponding to
 *     this voltage domain.
 */
typedef struct LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VPSTATE {
    LwU8 vpstateIndex;
} LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VPSTATE;

/*!
 * LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_FREQUENCY
 *
 * This structure describes a CLOCK DOMAIN and FREQUENCY.
 *
 *   clkDomain
 *     This parameter specifies the clock domain
 *   freqKHz
 *     This parameter specifies the frequency in KHz
 */
typedef struct LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_FREQUENCY {
    LwU32 clkDomain;
    LwU32 freqKHz;
} LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_FREQUENCY;

/*!
 * LW2080_CTRL_PERF_VOLT_DOM_INFO
 *
 * This structure describes the voltage settings corresponding to a voltage
 * domains, as related to performance.
 *
 *   domain
 *     This parameter specifies/returns the voltage domain as defined in
 *     LW2080_CTRL_PERF_VOLTAGE_DOMAINS.
 *   flags
 *     This parameter specifies/returns the voltage flags. They are defined
 *     together with @see LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO structure.
 *  voltageDomain
 *     Voltage domain as specified by the LW2080_CTRL_VOLT_VOLT_DOMAIN_<xyz> macros.
 *   type
 *     This parameter specifies/returns the type of voltage setting corresponding
 *     to this voltage domain.  This field is used to determine how to interpret
 *     the data union.
 *   data
 *     This parameter is a union to specify/return the type-specific voltage
 *     settings corresponding to the voltage domain.  The type parameter is used
 *     to interpret this union.
 *   voltageDeltauV
 *     This parameter specifies/returns the value of voltage delta (in uV) that
 *     represent the current overvoltage request/settings. Caller may use this
 *     value together with 'lwrrTargetVoltageuV' to callwlate nominal voltage.
 *   lwrrTargetVoltageuV
 *     This parameter is used to return the current target voltage (in uV) as
 *     evaluated by the current voltage setting corresponding to this voltage
 *     domain.  NOTE: This value may be dynamic, voltages settings (such as VDT
 *     indexes) can point to formulas which can dynamically change at run time
 *     (e.g. temperature-dependence).  This value is only provided as a helper
 *     to the caller to evaluate the voltage settings in concrete terms.
 *
 * @note  This structure is used within multiple RmCtrl calls. Assess all
 *        existing use-cases before making modifications/enhancements.
 */
typedef struct LW2080_CTRL_PERF_VOLT_DOM_INFO {
    LwU32 domain;
    LwU32 flags;
    LwU8  voltageDomain;

    LwU8  type;
    union {
        LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_LOGICAL   logical;
        LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VDT       vdt;
        LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VFE       vfe;
        LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_PSTATE    pstate;
        LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_VPSTATE   vpstate;
        LW2080_CTRL_PERF_VOLT_DOM_INFO_DATA_FREQUENCY freq;
    } data;

    LW2080_CTRL_PERF_PARAM_DELTA voltageDeltauV;

    LwU32                        lwrrTargetVoltageuV;
} LW2080_CTRL_PERF_VOLT_DOM_INFO;

/*!
 * Defines of the supported voltage types:
 *
 *  LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_LOGICAL
 *      Logical voltage is used to encode voltage settings.
 *
 *  LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_VDT
 *      VDT index is used to encode voltage settings.
 *
 *  LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_DELTA_ONLY
 *      Voltage delta (relative to nominal value) is used to encode voltage
 *      settings. Supported only within LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO
 *      and LW2080_CTRL_CMD_PERF_SET_PSTATES20_DATA commands. This type does
 *      not require type specific parameters.
 *  LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_VFE
 *      VFE index is used to encode voltage settings.
 *  LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_PSTATE
 *      PSTATE index is used to encode voltage settings.
 *  LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_VPSTATE
 *      VPSTATE index is used to encode voltage settings.
 */
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_LOGICAL    0x00
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_VDT        0x01
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_DELTA_ONLY 0x02
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_VFE        0x03
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_PSTATE     0x04
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_VPSTATE    0x05
#define LW2080_CTRL_PERF_VOLT_DOM_INFO_TYPE_FREQUENCY  0x06

/*
 * Deprecated. Please use LW2080_CTRL_CMD_PERF_GET_VIRTUAL_PSTATE_INFO_V2 instead.
 */
#define LW2080_CTRL_CMD_PERF_GET_VIRTUAL_PSTATE_INFO   (0x20802016) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_PARAMS {
    LwU32 virtualPstate;
    LwU8  index;
    LwU32 flags;
    LwU32 pstate;
    LwU32 perfClkDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfClkDomInfoList, 8);
} LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_PSTATE_INFO
 *
 * This command is an older and deprecated version of the
 * LW2080_CTRL_CMD_PERF_GET_PSTATE2_INFO_V2, that does not include additional
 * P-states 2.x information.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_PSTATE_INFO (0x20802061) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PSTATE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PSTATE_INFO_PARAMS_MESSAGE_ID (0x61U)

typedef struct LW2080_CTRL_PERF_GET_PSTATE_INFO_PARAMS {
    LwU32 pstate;
    LwU32 flags;
    LwU32 perfClkDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfClkDomInfoList, 8);
    LwU32 perfVoltDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfVoltDomInfoList, 8);
} LW2080_CTRL_PERF_GET_PSTATE_INFO_PARAMS;

/* valid flags values - keep in synch with SET_PSTATE_INFO */
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELIMIT                         0:0  // DEPRECATED
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELIMIT_GEN2          (0x00000000) // DEPRECATED
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELIMIT_GEN1          (0x00000001) // DEPRECATED
#define LW2080_CTRL_GET_PSTATE_FLAG_OVERCLOCKED                       1:1
#define LW2080_CTRL_GET_PSTATE_FLAG_OVERCLOCKED_FALSE       (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_OVERCLOCKED_TRUE        (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_OVERCLOCKABLE                     2:2
#define LW2080_CTRL_GET_PSTATE_FLAG_OVERCLOCKABLE_FALSE     (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_OVERCLOCKABLE_TRUE      (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH                     7:4
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_UNDEFINED (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_1         (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_2         (0x00000002)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_4         (0x00000003)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_8         (0x00000004)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_12        (0x00000005)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_16        (0x00000006)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH_32        (0x00000007)
#define LW2080_CTRL_GET_PSTATE_FLAG_VPS                              15:8
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKSPEED                   19:16
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKSPEED_UNDEFINED (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKSPEED_GEN1      (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKSPEED_GEN2      (0x00000002)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKSPEED_GEN3      (0x00000003)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_L0S_L1                 21:20
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_L0S                    20:20
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_L0S_ENABLE     (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_L0S_DISABLE    (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_L1                     21:21
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_L1_ENABLE      (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_L1_DISABLE     (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_DEEPL1                 22:22
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_DEEPL1_ENABLE  (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_PCIECAPS_DEEPL1_DISABLE (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_ORIGINAL_SETTINGS               28:28
#define LW2080_CTRL_GET_PSTATE_FLAG_ORIGINAL_SETTINGS_FALSE (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_ORIGINAL_SETTINGS_TRUE  (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_DEFAULT_SETTINGS                29:29
#define LW2080_CTRL_GET_PSTATE_FLAG_DEFAULT_SETTINGS_FALSE  (0x00000000)
#define LW2080_CTRL_GET_PSTATE_FLAG_DEFAULT_SETTINGS_TRUE   (0x00000001)
#define LW2080_CTRL_GET_PSTATE_FLAG_RSVD                            31:30

/*
 * Deprecated. Please use LW2080_CTRL_CMD_PERF_GET_PSTATE2_INFO_V2 instead.
 */
#define LW2080_CTRL_CMD_PERF_GET_PSTATE2_INFO               (0x20802067) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PSTATE2_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PSTATE2_INFO_PARAMS_MESSAGE_ID (0x67U)

typedef struct LW2080_CTRL_PERF_GET_PSTATE2_INFO_PARAMS {
    LwU32 pstate;
    LwU32 flags;
    LwU32 perfClkDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfClkDomInfoList, 8);
    LW_DECLARE_ALIGNED(LwP64 perfClkDom2InfoList, 8);
    LwU32 perfVoltDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfVoltDomInfoList, 8);
} LW2080_CTRL_PERF_GET_PSTATE2_INFO_PARAMS;

/*
 * Deprecated. Please use LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO_V2 instead.
 */
#define LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO (0x20802062) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_PSTATE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_PSTATE_INFO_PARAMS_MESSAGE_ID (0x62U)

typedef struct LW2080_CTRL_PERF_SET_PSTATE_INFO_PARAMS {
    LwU32 pstate;
    LwU32 flags;
    LwU32 perfClkDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfClkDomInfoList, 8);
    LwU32 perfVoltDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfVoltDomInfoList, 8);
} LW2080_CTRL_PERF_SET_PSTATE_INFO_PARAMS;

/* valid flags values - keep in synch with GET_PSTATE_INFO */
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_RSVD0                             2:0
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_RSVD1                             7:4
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_RSVD2                           29:28
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_MODE                            30:30
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_MODE_OVERCLOCKING  (0x00000000)
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_MODE_INTERNAL_TEST (0x00000001)
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_TEST_ONLY                       31:31
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_TEST_ONLY_FALSE    (0x00000000)
#define LW2080_CTRL_SET_PSTATE_INFO_FLAG_TEST_ONLY_TRUE     (0x00000001)


/*!
 * LW2080_CTRL_CMD_PERF_VF_TABLES_GET_INFO
 *
 * Returns static information about the size of the VF Indexes and VF Entries
 * Tables.  Should be called first to determine the range of acceptable values
 * to use for both LW2080_CTRL_CMD_PERF_VF_TABLES_ENTRIES_GET_INFO and
 * LW2080_CTRL_CMD_PERF_VF_TABLES_ENTRIES_SET_INFO.
 *
 * For documentation of parameters please see @ref
 * LW2080_CTRL_PERF_VF_TABLES_GET_INFO_PARAMS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_VF_TABLES_GET_INFO             (0x208020a0) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_VF_TABLES_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * LW2080_CTRL_PERF_VF_TABLES_GET_INFO_PARAMS
 *
 * Argument structure for LW2080_CTRL_CMD_PERF_VF_TABLES_GET_INFO.  Contains the
 * size of the VF Indexes and VF Entries tables.
 */
#define LW2080_CTRL_PERF_VF_TABLES_GET_INFO_PARAMS_MESSAGE_ID (0xA0U)

typedef struct LW2080_CTRL_PERF_VF_TABLES_GET_INFO_PARAMS {
    /*!
     * Out: Number of entries in the VF Indexes table.
     */
    LwU32 numIndexes;
    /*!
     * Out: Number of entries in the VF Entries table.
     */
    LwU32 numEntries;
} LW2080_CTRL_PERF_VF_TABLES_GET_INFO_PARAMS;


/*!
 * LW2080_CTRL_CMD_PERF_VF_TABLES_ENTRIES_GET_INFO
 *
 * Returns the current values in a set of VF Indexes and/or VF Entries Tables
 * entries, as specified by the client.  Used to dump the current state of the
 * VF tables.
 *
 * For documentation of parameters please see @ref
 * LW2080_CTRL_PERF_VF_TABLES_ENTRIES_INFO_PARAMS
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_VF_TABLES_ENTRIES_GET_INFO (0x208020a1) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xA1" */

/*!
 * LW2080_CTRL_CMD_PERF_VF_TABLES_ENTRIES_SET_INFO
 *
 * Sets the current values in a set of VF Indexes and/or VF Entries Tables
 * entries, as specified by the client.  Used to lwstomize the state of the
 * VF tables at run-time.
 *
 * For documentation of parameters please see @ref
 * LW2080_CTRL_PERF_VF_TABLES_ENTRIES_INFO_PARAMS
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_VF_TABLES_ENTRIES_SET_INFO (0x208020a2) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xA2" */

/*!
 * LW2080_CTRL_PERF_VF_INDEXES_TABLE_INFO
 *
 * This structure represents an entry in the VF Indexes Table.
 */
typedef struct LW2080_CTRL_PERF_VF_INDEXES_TABLE_INFO {
    /*!
     * In: Index of this VF Indexes Table entry
     */
    LwU32 index;

    /*!
     * In/Out: Pstate index corresponding to this index.
     */
    LwU32 pstate;
    /*!
     * In/Out: Clock domain corresponding to this index.
     */
    LwU32 domain;
    /*!
     * Index into the VF Entries Table of the first entry in this logical table.
     */
    LwU8  entryIndexFirst;
    /*!
     * Index into the VF Entries Table of the last entry in this logical table.
     */
    LwU8  entryIndexLast;
} LW2080_CTRL_PERF_VF_INDEXES_TABLE_INFO;

#define LW2080_CTRL_PERF_VF_TABLE_ENTRY_STEP_SIZE_NONE 0
#define LW2080_CTRL_PERF_VF_TABLE_ENTRY_STEP_SIZE_ALL  1

/*!
 * LW2080_CTRL_PERF_VF_ENTRIES_TABLE_INFO
 *
 * This structure represents an entry in the VF Entries Table.
 */
typedef struct LW2080_CTRL_PERF_VF_ENTRIES_TABLE_INFO {
    /*!
     * In: Index of this VF Entries Table entry
     */
    LwU32                          index;

    /*!
     * In/Out: Max clock frequency which can be supported in this entry.
     */
    LwU32                          maxFreqKHz;
    /*!
     * In/Out: Frequency step size allowed in this entry.
     */
    LwU32                          freqStepSizeKHz;
    /*!
     * In/Out: LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_<xyz>
     */
    LwU32                          flags;
    /*!
     * In/Out: Voltage required to the max clock frequency.
     */
    LW2080_CTRL_PERF_VOLT_DOM_INFO voltageInfo;
} LW2080_CTRL_PERF_VF_ENTRIES_TABLE_INFO;

/*!
 * LW2080_CTRL_PERF_VF_TABLES_ENTRIES_INFO_PARAMS
 *
 * Structure representing a set of VF Indexes and VF Entries Tables entries for
 * which the client either wants to access or mutate their settings.
 */
typedef struct LW2080_CTRL_PERF_VF_TABLES_ENTRIES_INFO_PARAMS {
    /*!
     * In: Number of LW2080_CTRL_PERF_VF_INDEXES_TABLE_INFO structures to find
     * in the buffer pointed at by @ref vfIndexesInfoList.
     */
    LwU32 vfIndexesInfoListSize;
    /*!
     * In/Out: Pointer to the client's buffer of size >= @ref
     * vfIndexesInfoListSize * sizeof (LW2080_CTRL_PERF_VF_INDEXES_TABLE_INFO)
     * in which the RM can find the specified set of VF Indexes Table entries to
     * operate upon.
     */
    LW_DECLARE_ALIGNED(LwP64 vfIndexesInfoList, 8);
    /*!
     * In: Number of LW2080_CTRL_PERF_VF_ENTRIES_TABLE_INFO structures to find
     * in the buffer pointed at by @ref vfEntriesInfoList.
     */
    LwU32 vfEntriesInfoListSize;
    /*!
     * In/Out: Pointer to the client's buffer of size >= @ref
     * vfEntriesInfoListSize * sizeof (LW2080_CTRL_PERF_VF_ENTRIES_TABLE_INFO)
     * in which the RM can find the specified set of VF Entries Table entries to
     * operate upon.
     */
    LW_DECLARE_ALIGNED(LwP64 vfEntriesInfoList, 8);
} LW2080_CTRL_PERF_VF_TABLES_ENTRIES_INFO_PARAMS;


/*
 * LW2080_CTRL_CMD_PERF_LOOK_UP_VOLTAGE
 *
 * This command, when given a pstate, clock domain, and frequency, returns
 * the corresponding voltage information.
 *
 *   flags
 *     This field specifies the flags for the operation.
 *     It is reserved for future use.
 *   pstate
 *     This parameter specifies the target P-state for the inquiry.
 *   domain
 *     This parameter specifies the target decoupled clock domain for the
 *     inquiry.
 *   freq
 *     This parameter specifies the target clock freq, in KHz, for the inquiry.
 *   perfVoltDomInfoListSize
 *     This parameter specifies the number of performance voltage domain
 *     entries to return in the associated perfVoltDomInfoList buffer.
 *     This parameter should be set to the number of enabled bits in
 *     perfVoltageDomains parameter returned by the
 *     LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO command.
 *   perfVoltDomInfoList
 *     This field specifies a pointer in the caller's address space to the
 *     buffer into which the performance voltage information is to be returned.
 *     This buffer must be at least as big as perfVoltDomInfoListSize multiplied
 *     by the size of the LW2080_CTRL_PERF_VOLT_DOM_INFO structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_LOOK_UP_VOLTAGE (0x20802066) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_LOOK_UP_VOLTAGE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_LOOK_UP_VOLTAGE_PARAMS_MESSAGE_ID (0x66U)

typedef struct LW2080_CTRL_PERF_LOOK_UP_VOLTAGE_PARAMS {
    LwU32 flags;
    LwU32 pstate;
    LwU32 domain;
    LwU32 freq;
    LwU32 perfVoltDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfVoltDomInfoList, 8);
} LW2080_CTRL_PERF_LOOK_UP_VOLTAGE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_RATIO_CLK_FREQ
 *
 * This command, when given a pstate, the decoupled clock domain, and
 * frequency, returns the corresponding ratio clock frequency information.
 *
 *   flags
 *     This field specifies the flags for the operation.
 *     It is reserved for future use.
 *   pstate
 *     This parameter specifies the P-state for the inquiry.
 *   domain
 *     This parameter specifies the decoupled clock domain for the inquiry.
 *   freq
 *     This parameter specifies the decoupled clock frequency, in KHz.
 *   perfClkDomInfoListSize
 *     This parameter specifies the number of performance clock domain
 *     entries to return in the associated perfClkDomInfoList buffer.
 *     This parameter should be set to the number of enabled bits in
 *     perfClkDomains parameter returned by the
 *     LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO command.
 *   perfClkDomInfoList
 *     This field specifies a pointer in the caller's address space to the
 *     buffer into which the performance clock information is to be returned.
 *     This buffer must be at least as big as perfClkDomInfoListSize multiplied
 *     by the size of the LW2080_CTRL_PERF_CLK_DOM_INFO structure.
 *     Each updated entry will have LW2080_CTRL_PERF_CLK_DOM_INFO_FLAGS_UPDATED
 *     set to _YES.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_RATIO_CLK_FREQ (0x20802069) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_RATIO_CLK_FREQ_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_RATIO_CLK_FREQ_PARAMS_MESSAGE_ID (0x69U)

typedef struct LW2080_CTRL_PERF_GET_RATIO_CLK_FREQ_PARAMS {
    LwU32 flags;
    LwU32 pstate;
    LwU32 domain;
    LwU32 freq;
    LwU32 perfClkDomInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 perfClkDomInfoList, 8);
} LW2080_CTRL_PERF_GET_RATIO_CLK_FREQ_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_SUPPORTED_VOLTAGE_DOMAINS
 *
 * This command returns the set of voltage domains supported/controllable on the
 * current board.
 *
 *   flags
 *     This parameter returns flags about the voltage domains.  However, none
 *     are lwrrently supported.
 *   voltageDomains
 *     This parameter returns the mask of supported voltage domains. Each
 *     enabled bit in the mask represents the support for the corresponding
 *     voltage domain, as defined in LW2080_CTRL_PERF_VOLTAGE_DOMAINS values.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_SUPPORTED_VOLTAGE_DOMAINS (0x20802063) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_SUPPORTED_VOLTAGE_DOMAINS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_SUPPORTED_VOLTAGE_DOMAINS_PARAMS_MESSAGE_ID (0x63U)

typedef struct LW2080_CTRL_PERF_GET_SUPPORTED_VOLTAGE_DOMAINS_PARAMS {
    LwU32 flags;
    LwU32 voltageDomains;
} LW2080_CTRL_PERF_GET_SUPPORTED_VOLTAGE_DOMAINS_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS
 *
 * This command returns the set of voltage domains supported/controllable on the
 * current board.
 *
 *   voltageDomain
 *     This parameter specifies the mask of the voltage domain for which to
 *     return the number of supported levels. Valid domains are defined in
 *     LW2080_CTRL_PERF_VOLTAGE_DOMAINS values.  Should be returned as supported
 *     by LW2080_CTRL_CMD_PERF_GET_VOLTAGE_DOMAINS_INFO.
 *   flags
 *     Used to specify certain features for request and to return information
 *     about the domain.  Valid flags include:
 *       LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_FLAGS_INTERNAL - [in]
 *           Used to specify that the request comes from an internal/privileged
 *           LW client, such as MODS.  Will ignore all internal voltage
 *           protections (i.e. limiting to only boards which are approved for
 *           overvoltaging) and return all voltages supported by the board.
 *   voltageDomainNumLevels
 *     The number of levels supported by the voltage domain.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS (0x20802064) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_PARAMS_MESSAGE_ID (0x64U)

typedef struct LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_PARAMS {
    LwU32 voltageDomain;
    LwU32 flags;
    LwU32 voltageDomainNumLevels;
} LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_PARAMS;

#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_FLAGS_INTERNAL       0:0
#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_FLAGS_INTERNAL_FALSE (0x00000000)
#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_FLAGS_INTERNAL_TRUE  (0x00000001)
#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_NUM_LEVELS_FLAGS_RSVD           31:1

/*
 * LW2080_CTRL_CMD_PERF_GET_VOLTAGE_DOMAIN_LEVELS
 *
 * This command returns the set of voltage levels supported/controllable in the
 * specified domain on the current board.
 *
 *   voltageDomain
 *     This parameter specifies the voltage domain for which voltage levels
 *     should be returned.  Valid entries are defined in
 *     LW2080_CTRL_PERF_VOLTAGE_DOMAINS.
 *   flags
 *     Used to specify certain features for request and to return information
 *     about the domain.  Valid flags include:
 *       LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_FLAGS_INTERNAL - [in]
 *           Used to specify that the request comes from an internal/privileged
 *           LW client, such as MODS.  Will ignore all internal voltage
 *           protections (i.e. limiting to only boards which are approved for
 *           overvoltaging) and return all voltages supported by the board.
 *   voltageDomainLevelsListSize
 *     This parameter specifies the number of performance voltage domain levels
 *     entries to return in the associated perfVoltDomInfoList buffer.  This
 *     parameter should be set to the number of entries returned in the
 *     voltageDomainNumLevels parameter by the
 *     LW2080_CTRL_CMD_GET_VOLTAGE_DOMAINS_INFO command.
 *   voltageDomainLevelsList
 *     This field specifies a pointer in the caller's address space to the
 *     buffer into which the performance voltage domain levels information is to
 *     be returned.  This buffer must be at least as big as
 *     voltageDomainLevelsListSize multiplied by the size of the a DWORD (4
 *     bytes).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_VOLTAGE_DOMAIN_LEVELS                      (0x20802065) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_PARAMS_MESSAGE_ID (0x65U)

typedef struct LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_PARAMS {
    LwU32 voltageDomain;
    LwU32 flags;
    LwU32 voltageDomainLevelsListSize;
    LW_DECLARE_ALIGNED(LwP64 voltageDomainLevelsList, 8);
} LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_PARAMS;

#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_FLAGS_INTERNAL           0:0
#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_FLAGS_INTERNAL_FALSE (0x00000000)
#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_FLAGS_INTERNAL_TRUE  (0x00000001)
#define LW2080_CTRL_PERF_GET_VOLTAGE_DOMAIN_LEVELS_FLAGS_RSVD              31:1

/*
 * LW2080_CTRL_CMD_PERF_GET_LWRRENT_PSTATE
 *
 * This command returns the current performance state of the GPU.
 *
 *   lwrrPstate
 *     This parameter returns the current P-state, as defined in
 *     LW2080_CTRL_PERF_PSTATES values.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_PERF_GET_LWRRENT_PSTATE                         (0x20802068) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_LWRRENT_PSTATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_LWRRENT_PSTATE_PARAMS_MESSAGE_ID (0x68U)

typedef struct LW2080_CTRL_PERF_GET_LWRRENT_PSTATE_PARAMS {
    LwU32 lwrrPstate;
} LW2080_CTRL_PERF_GET_LWRRENT_PSTATE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_FORCE_PSTATE
 *
 * This command forces the GPU into the specified performance state.
 *
 *   forcePstate
 *     This parameter specifies the target P-state. Possible P-states
 *     are defined in LW2080_CTRL_PERF_PSTATES. Specify CLEAR_FORCED
 *     to stop forcing to any P-states.
 *   fallback
 *     This parameter specifies the fallback strategy when the target
 *     P-state is not available. Possible fallback values are defined
 *     in LW2080_CTRL_PERF_PSTATE_FALLBACK.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_SET_FORCE_PSTATE (0x20802070) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_FORCE_PSTATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_PARAMS_MESSAGE_ID (0x70U)

typedef struct LW2080_CTRL_PERF_SET_FORCE_PSTATE_PARAMS {
    LwU32 forcePstate;
    LwU32 fallback;
} LW2080_CTRL_PERF_SET_FORCE_PSTATE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_FORCE_PSTATE_EX
 *
 * This command forces the GPU into the specified performance state,
 * allowing various options as specified in the flags field.
 *
 *   forcePstate
 *     This parameter specifies the target P-state. Possible P-states
 *     are defined in LW2080_CTRL_PERF_PSTATES. Specify CLEAR_FORCED
 *     to stop forcing to any P-states.
 *   fallback
 *     This parameter specifies the fallback strategy when the target
 *     P-state is not available. Possible fallback values are defined
 *     in LW2080_CTRL_PERF_PSTATE_FALLBACK.
 *   flags
 *     This parameter allows control over behavior for how the p-state
 *     is changed. Possible values are defined in
 *     LW2080_CTRL_PERF_PSTATE_CHANGE_FLAGS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_SET_FORCE_PSTATE_EX (0x20802071) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_PARAMS_MESSAGE_ID (0x71U)

typedef struct LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_PARAMS {
    LwU32 forcePstate;
    LwU32 fallback;
    LwU32 flags;
} LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_PARAMS;

#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_ASYNC                  0:0
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_ASYNC_DISABLED       (0x00000000)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_ASYNC_ENABLED        (0x00000001)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_PRIORITY               1:1
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_PRIORITY_NORM        (0x00000000)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_PRIORITY_MODS        (0x00000001)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_VBLANK_WAIT                  2:2
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_VBLANK_WAIT_DEFAULT  (0x00000000)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_VBLANK_WAIT_SKIP     (0x00000001)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_VOLTAGE2                     4:3
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_VOLTAGE2_NO_CHANGE   (0x00000000)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_VOLTAGE2_PSTATE_ONLY (0x00000001)
#define LW2080_CTRL_PERF_SET_FORCE_PSTATE_EX_FLAGS_VOLTAGE2_NORMAL      (0x00000002)

/*
 * LW2080_CTRL_CMD_PERF_GET_PSTATE_CLIENT_LIMITS
 *
 * This command returns hard/soft limit of performance state (P-State)
 *
 *   limitType
 *     This parameter specifies hard or soft limit of the max P-state. Possible
 *     type are defined as LW2080_CTRL_PERF_PSTATE_LIMIT_*.
 *
 *   PstateLimit
 *     This parameter returns the max P-state of client soft or hard limit.
 *     P-states values are defined in LW2080_CTRL_PERF_PSTATES.  It returns
 *     PSTATES_P0 when the limit is not set.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_PERF_PSTATE_CLIENT_LIMIT_HARD                       (0x00000001)
#define LW2080_CTRL_PERF_PSTATE_CLIENT_LIMIT_SOFT                       (0x00000002)

#define LW2080_CTRL_CMD_PERF_GET_PSTATE_CLIENT_LIMITS                   (0x20802072) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PSTATE_CLIENT_LIMITS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PSTATE_CLIENT_LIMITS_PARAMS_MESSAGE_ID (0x72U)

typedef struct LW2080_CTRL_PERF_GET_PSTATE_CLIENT_LIMITS_PARAMS {
    LwU32 limitId;
    LwU32 PstateLimit;
} LW2080_CTRL_PERF_GET_PSTATE_CLIENT_LIMITS_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_PSTATE_CLIENT_LIMITS
 *
 * This command sets hard/soft limit of performance state (P-State)
 *
 *   limitType
 *     This parameter specifies hard or soft limit of the max P-state. Possible
 *     value are defined as LW2080_CTRL_PERF_PSTATE_LIMIT_*.
 *
 *   PstateLimit
 *     This parameter specifies the performance state to be the max P-state
 *     for hard limit or soft limit.  Possible P-states are defined in
 *     LW2080_CTRL_PERF_PSTATES.  Specify PSTATES_P0 to stop the P-state limit.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_SET_PSTATE_CLIENT_LIMITS (0x20802073) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_PSTATE_CLIENT_LIMITS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_PSTATE_CLIENT_LIMITS_PARAMS_MESSAGE_ID (0x73U)

typedef struct LW2080_CTRL_PERF_SET_PSTATE_CLIENT_LIMITS_PARAMS {
    LwU32 limitId;
    LwU32 PstateLimit;
} LW2080_CTRL_PERF_SET_PSTATE_CLIENT_LIMITS_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_ENABLE_OVERCLOCKED_PSTATES
 *
 * This command enables or disables overclocked P-states
 *
 *   enable
 *     This parameter specifies to enable or to disable overclocked P-states
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_ENABLE_OVERCLOCKED_PSTATES (0x20802074) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_ENABLE_OVERCLOCKED_PSTATES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_ENABLE_OVERCLOCKED_PSTATES_PARAMS_MESSAGE_ID (0x74U)

typedef struct LW2080_CTRL_PERF_ENABLE_OVERCLOCKED_PSTATES_PARAMS {
    LwU8 enable;
} LW2080_CTRL_PERF_ENABLE_OVERCLOCKED_PSTATES_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_PSTATE_LIMIT_EX
 *
 * Allows client to set or clear any limit in any units.
 * This is a wrapper for perfSetLimit.  It is used for internal testing only.
 * Do not export to LWAPI.  All parameters are untranslated.  See perfSetLimit
 * function header comments.
 */
#define LW2080_CTRL_CMD_PERF_SET_PSTATE_LIMIT_EX (0x20802075) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_PSTATE_LIMIT_EX_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_PSTATE_LIMIT_EX_PARAMS_MESSAGE_ID (0x75U)

typedef struct LW2080_CTRL_PERF_SET_PSTATE_LIMIT_EX_PARAMS {
    LwU32 reason;
    LwU32 value;
    LwU32 domain;
    LwU32 flags;
} LW2080_CTRL_PERF_SET_PSTATE_LIMIT_EX_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_POWER_CONNECTOR_STATUS
 *
 * This command will get the performance statue information for each
 * supported power connector on the associated GPU.
 *
 *  connectorsPresent
 *    This mask reports the set of connectors present on the board.
 *  bootStatus
 *    This mask reports the set of connectors connected at boot time.
 *  lwrrStatus
 *    This mask reports the set of connectors associated with the current
 *    performance level.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */

/* power connector mask format */
#define LW2080_CTRL_PERF_POWER_CONNECTOR_STATUS_AUX                 0:0
#define LW2080_CTRL_PERF_POWER_CONNECTOR_STATUS_AUX_DISCONN (0x00000000)
#define LW2080_CTRL_PERF_POWER_CONNECTOR_STATUS_AUX_CONN    (0x00000001)

#define LW2080_CTRL_CMD_PERF_GET_POWER_CONNECTOR_STATUS     (0x20802088) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_POWER_CONNECTOR_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_POWER_CONNECTOR_PARAMS_MESSAGE_ID (0x88U)

typedef struct LW2080_CTRL_PERF_GET_POWER_CONNECTOR_PARAMS {
    LwU32 connectorsPresent;
    LwU32 bootStatus;
    LwU32 lwrrStatus;
} LW2080_CTRL_PERF_GET_POWER_CONNECTOR_PARAMS;

/*
 * LW2080_CTRL_PERF_LIMIT_INFO
 *
 * This structure contains descriptive information about a defined performance
 * limiting factor.
 *   suspendId
 *     This field is the unique Id for a defined performance limit
 *   limitType
 *     This field is a mask of the type of limitation this limit
 *     represents. Valid values are:
 *       LW2080_CTRL_PERF_SUSPEND_LIMIT_TYPE_MIN
 *         The p-state associated with the suspend is the minimum allowed
 *         p-state.
 *       LW2080_CTRL_PERF_SUSPEND_LIMIT_TYPE_MAX
 *         The p-state associated with the suspend is themaximum allowed
 *         p-state.
 *       Both flags can be used indicating that the p-state is used as both
 *       a maximum and minimum limit.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INFO_DEPRECATED {
    LwU8 limitId;
    LwU8 limitType;
} LW2080_CTRL_PERF_LIMIT_INFO_DEPRECATED;

/*!
 * P-States limiting factors from lowest to highest priority
 *
 * ** This enum must always match these tables:
 *
 *  drivers/resman/kernel/perfctl/lw/2x/perf_limit_2x.c:PstateLimitConstData2x[]
 *  drivers/resman/kernel/perfctl/lw/35/perf_limit_35.c:PstateLimitConstData35[]
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/PState_Limiting_Factors
 */
typedef LwU32 LW2080_CTRL_PERF_PERF_LIMIT_ID;

#define LW2080_CTRL_PERF_PERF_LIMIT_ID_NOT_SET                                     (0U) //<! No limit is set.  Used for reporting purposes.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_UTIL_DOMAIN_GROUP0                          (1U) //<! Utilization for domain group 0.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_UTIL_DOMAIN_GROUP1                          (2U) //<! Utilization for domain group 1.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERF_CF_CONTROLLER_DRAM_MIN                 (3U) //<! PERF CF Controller min limit for memory clock.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERF_CF_CONTROLLER_GPC_MIN                  (4U) //<! PERF CF Controller min limit for graphics clock.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERF_CF_CONTROLLER_LWD_MIN                  (5U) //<! PERF CF Controller min limit for video clock.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_JPAC_PSTATE_MIN                             (6U) //<! Floor Limit from JPAC (_PSTATE)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_AFTER_UTIL                                  (7U)                                      //<! *** NOT A LIMIT ENTRY *** Starting marker above utilization.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_GPU_IS_IDLE                                 LW2080_CTRL_PERF_PERF_LIMIT_ID_AFTER_UTIL  //<! Aggressive Switching when GPU is idle for group 0
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_GPU_IS_IDLE_GROUP1                          (8U)                                      //<! Aggressive Switching when GPU is idle for group 1
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERFORMANCE_CAP0                            (9U)                                      //<! Cap based on performance data (GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERFORMANCE_CAP1                            (10U)                                      //<! Cap based on performance data (P-state)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_BOOST_LOW                                   (11U) //<! Lower priority boost (for non-LWCA)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERF_CF_CONTROLLER_DRAM_MAX                 (12U) //<! PERF CF Controller max limit for memory clock.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERF_CF_CONTROLLER_GPC_MAX                  (13U) //<! PERF CF Controller max limit for graphics clock.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERF_CF_CONTROLLER_LWD_MAX                  (14U) //<! PERF CF Controller max limit for video clock.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PERF_CF_CONTROLLER_XBAR_MAX                 (15U) //<! PERF CF Controller max limit for XBAR clock.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_BOOST                                       (16U) //<! Higher priority boost
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_DOM_GRP_0_MIN                           (17U) //<! Minimum P-state for bridgeless SLI.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PM_DYNAMIC                                  (18U) //<! Used to disable PowerMizer PERFSTATE
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_NO_CLIENT                                   (19U) //<! ClientEnabledPerf when PERFMON_2D is disabled
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SCREEN_SAVER                                (20U) //<! Notification of screen saver
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_NO_HIRES                                    (21U) //<! FSDOS, etc
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_DEEP_IDLE                                   (22U) //<! Hybrid deep idle notification
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OS_LEVEL                                    (23U) //<! OS Level Perf State change
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_VPS                                         (24U) //<! Virtual P-States limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_JPAC_PSTATE_MAX                             (25U) //<! Limit from JPAC (_PSTATE)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_JPAC_GPC_MAX                                (26U) //<! Limit from JPAC (_GPCCLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SPDIFF_GLITCH                               (27U) //<! Pstate change causes SPDIFF to glitch
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_POWERMIZER                                  (28U) //<! Powermizer
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_0_MAX                                (29U) //<! Client limit 0 max : Need to be removed in favor of new LOOSE limits below
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_0_MIN                                (30U) //<! Client limit 0 min : Need to be removed in favor of new LOOSE limits below
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_1_MAX                                (31U) //<! Client limit 1 max : Need to be removed in favor of new LOOSE limits below
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_1_MIN                                (32U) //<! Client limit 1 min : Need to be removed in favor of new LOOSE limits below
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_2_MAX                                (33U) //<! Client limit 2 max : Need to be removed in favor of new LOOSE limits below
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_2_MIN                                (34U) //<! Client limit 2 min : Need to be removed in favor of new LOOSE limits below
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_INTERSECT                        (35U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_PSTATE_MAX                (36U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_PSTATE_MIN                (37U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_DRAM_MAX                  (38U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOCK_DRAM_MAX                    (39U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_PCIE_MAX                  (40U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_GPC_MAX                   (41U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_DISP_MAX                  (42U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_XBAR_MAX                  (43U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOCK_XBAR_MAX                    (44U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_DRAM_MIN                  (45U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOCK_DRAM_MIN                    (46U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_PCIE_MIN                  (47U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_GPC_MIN                   (48U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_DISP_MIN                  (49U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_STRICT_XBAR_MIN                  (50U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOCK_XBAR_MIN                    (51U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_0_MAX                      (52U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_0_MIN                      (53U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_1_MAX                      (54U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_1_MIN                      (55U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_2_MAX                      (56U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_2_MIN                      (57U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_3_MAX                      (58U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_3_MIN                      (59U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_4_MAX                      (60U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_4_MIN                      (61U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_5_MAX                      (62U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_5_MIN                      (63U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_6_MAX                      (64U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_6_MIN                      (65U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_7_MAX                      (66U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_7_MIN                      (67U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_8_MAX                      (68U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_8_MIN                      (69U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_9_MAX                      (70U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOW_LOOSE_9_MIN                      (71U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_APPLICATIONCLOCKS                           (72U) //<! Sets max to Application clock
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OVERCLOCK                                   (73U) //<! For enabling/disabling Overclocked PerfLevel
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RC_ERROR                                    (74U) //<! Notification of RC error
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_BUG_535734                                  (75U) //<! WAR to make sure dispclk >= sorclk when doing HDCP tasks
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MIN_FAN_LEVEL                               (76U) //<! Minimum fan level
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_VIDEO_WAR                                   (77U) //<! Various WARs for SD/HD video playback
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_BUG_660789                                  (78U) //<! WAR to make sure we don't switch to P12 on GF100 P1030's. This needs to be lower priority than UNLOAD_DRIVER
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_DISPLAY_GLITCH                              (79U) //<! Pstate change causes display to glitch=
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_ISMODEPOSSIBLE                              (80U) //<! IMP requirement
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_UNLOAD_DRIVER                               (81U) //<! Locking to a P-State between unload/load
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_ISMODEPOSSIBLE_DISP                         (82U) //<! IMP DISPCLK requirement
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_BANDWIDTH_HCLONE                            (83U) //<! Bandwidth requirement in hclone mode
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SUSPEND_POWER                               (84U) //<! Locking to a P-State during suspend
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_EXCEPT_VIDEO                                (85U) //<! Limit perf/power except during video playback
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_BIF_USBC_PMU_PCIE_MIN                       (86U) //<! PMU - Lock the pcie genspeed to max for USB-C isochronous traffic
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_BIF_USBC_RM_PCIE_MIN                        (87U) //<! RM - Lock the pcie genspeed to max for USB-C isochronous traffic
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT                                   (88U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_LOGIC_1_MAX                       (89U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_LOGIC_2_MAX                       (90U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_SRAM_0_MAX                        (91U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_SRAM_1_MAX                        (92U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_SRAM_2_MAX                        (93U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_LOGIC_0_MIN                       (94U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_LOGIC_1_MIN                       (95U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_LOGIC_2_MIN                       (96U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_SRAM_0_MIN                        (97U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_SRAM_1_MIN                        (98U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_INTERSECT_SRAM_2_MIN                        (99U)   //<! TODO: Remove all except _INTERSECT. Rest are deprecated
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_PSTATE_MAX                    (100U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_PSTATE_MIN                    (101U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_DRAM_MAX                      (102U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOCK_DRAM_MAX                        (103U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_PCIE_MAX                      (104U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_GPC_MAX                       (105U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_DISP_MAX                      (106U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_XBAR_MAX                      (107U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOCK_XBAR_MAX                        (108U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_DRAM_MIN                      (109U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOCK_DRAM_MIN                        (110U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_PCIE_MIN                      (111U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_GPC_MIN                       (112U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_DISP_MIN                      (113U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_STRICT_XBAR_MIN                      (114U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOCK_XBAR_MIN                        (115U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_0_MAX                          (116U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_1_MAX                          (117U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_2_MAX                          (118U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_3_MAX                          (119U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_4_MAX                          (120U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_5_MAX                          (121U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_6_MAX                          (122U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_7_MAX                          (123U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_8_MAX                          (124U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_9_MAX                          (125U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_0_MIN                          (126U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_1_MIN                          (127U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_2_MIN                          (128U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_3_MIN                          (129U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_4_MIN                          (130U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_5_MIN                          (131U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_6_MIN                          (132U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_7_MIN                          (133U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_8_MIN                          (134U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_LOOSE_9_MIN                          (135U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_FORCED                                      (136U) //<! Forced to the specified level
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_FORCED_DACPERFTEST                          (137U) //<! Forced to the specified level for DAC perf test
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_FORCED_LINKTRAIN                            (138U) //<! Forced to the specified level for link training
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_START_HARDCAPS                              (139U) //<!*** NOT A LIMIT ENTRY *** Starting marker for hardcaps.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SETUP_LOCK LW2080_CTRL_PERF_PERF_LIMIT_ID_START_HARDCAPS //<! A max pstate to lock to when performing LW2080_CTRL_CMD_PERF_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SCALE_SETUP
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_LOCKED_DRIVER                               (140U) //<! Limit locked driver to a P-state.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_DISP_RESTRICT_GPC_MAX                       (141U) //<! limit to restrict GPCCLK on demand from Display
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_UNLOAD_DRIVER_PSTATE                        (142U) //<! Boot P-State driver locks when unload
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_UNLOAD_DRIVER_VOLTAGE_RAIL_0                (143U) //<! Boot voltage driver locks when unload
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_UNLOAD_DRIVER_VOLTAGE_RAIL_1                (144U) //<! Boot voltage driver locks when unload
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_UNLOAD_DRIVER_DISP                          (145U) //<! Boot dispclk driver locks when unload
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_LWSTOMER_BOOST_MAX                          (146U) //<! Sets max limit on auto-boost mode
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OOB_CLOCK_LIMIT_MAX                         (147U) //<! Sets oob clock supersede customer boost max
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OOB_CLOCK_LIMIT_MIN                         (148U) //<! Sets oob clock supersede customer boost max
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RATED_TDP_MAX                               (149U) //<! Limit clocks to rated TDP.  Used when there is no GPS or PowerCapping.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RATED_TDP_MIN                               (150U) //<! Lock clocks to rated TDP.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_ECC_MAX                                     (151U) //<! Restrict Pstate to max for ECC only on Turing DT Lwdqro and later chips.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_LWDA_MAX                                    (152U) //<! Maximum P-state safe for LWCA.
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_DRAM                           (153U) //<! Thermal policy limit for Domain Group 0 (_PSTATE)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_GPC                            (154U) //<! Thermal policy limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_XBAR                           (155U) //<! Thermal policy limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_LWVDD                          (156U) //<! Thermal policy limit for LWVDD
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_EDP_POLICY_DOM_GRP_1                        (157U) //<! EDP policy limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OVERVOLTAGE_LWVDD                           (158U) //<! LWVDD Overvoltage Limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OVERVOLTAGE_MSVDD                           (159U) //<! MSVDD Overvoltage Limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_ALT_LWVDD                       (160U) //<! LWVDD Reliability Limit - Alternate
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_ALT_MSVDD                       (161U) //<! MSVDD Reliability Limit - Alternate
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_LWVDD                           (162U) //<! LWVDD Reliability Limit - Default
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_MSVDD                           (163U) //<! MSVDD Reliability Limit - Default
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_POWERMIZER_HARD                             (164U) //<! Powermizer hard cap
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_CLIENT_HARD                                 (165U) //<! Client hard limit set by User APP
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_THERMAL                                     (166U) //<! Notification of Thermal event
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SYSPERF                                     (167U) //<! System performance limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_AUX_PWR_STATE                               (168U) //<! Auxiliary Power State changes
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PWR_SUPPLY_CAPACITY                         (169U) //<! Dynamic power supply capacity changes
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PWR_POLICY_DRAM                             (170U) //<! PMU limit for Domain Group 0 (MCLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PWR_POLICY_GPC                              (171U) //<! PMU limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PWR_POLICY_XBAR                             (172U) //<! PMU limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_JPPC_PSTATE_MAX                             (173U) //<! Limit from JPPC (_PSTATE)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_JPPC_GPC_MAX                                (174U) //<! Limit from JPPC (_GPCCLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_GPU_BOOST_DRAM                          (175U) //<! SLI GPU Boost limit for Domain Group 0 (_PSTATE)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_GPU_BOOST_GPC                           (176U) //<! SLI GPU Boost limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_GPU_BOOST_XBAR                          (177U) //<! SLI GPU Boost limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SW_BATTPOWER                                (178U) //<! SW-only battery power req
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_EXT_PERF_CONTROL                            (179U) //<! External Perf control via gpio (also MXM battery power req)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MXM_ACPOWER                                 (180U) //<! MXM AC power requirement
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_AUX_POWER                                   (181U) //<! Notification of no Aux Power
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_VMIN_LWVDD                                  (182U) //<! LWVDD Vmin Limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_VMIN_MSVDD                                  (183U) //<! MSVDD Vmin Limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_GPU_STATE_LOAD_BOOST_DOM_GRP_0              (184U) //<! GPU state load boost limit for Domain Group 0 (_PSTATE)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_GPU_STATE_LOAD_BOOST_DOM_GRP_1              (185U) //<! GPU state load boost limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RAM_ASSIST_UNLOAD_DRIVER_VOLTAGE_RAIL_0     (186U) //<! Boot voltage driver floor when unload for RAM Assist
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RAM_ASSIST_UNLOAD_DRIVER_VOLTAGE_RAIL_1     (187U) //<! Boot voltage driver floor when unload for RAM Assist
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_INTERSECT                        (188U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_PSTATE_MAX                (189U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_PSTATE_MIN                (190U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_DRAM_MAX                  (191U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOCK_DRAM_MAX                    (192U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_PCIE_MAX                  (193U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_GPC_MAX                   (194U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_DISP_MAX                  (195U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_XBAR_MAX                  (196U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOCK_XBAR_MAX                    (197U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_DRAM_MIN                  (198U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOCK_DRAM_MIN                    (199U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_PCIE_MIN                  (200U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_GPC_MIN                   (201U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_DISP_MIN                  (202U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_STRICT_XBAR_MIN                  (203U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOCK_XBAR_MIN                    (204U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_0_MAX                      (205U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_0_MIN                      (206U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_1_MAX                      (207U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_1_MIN                      (208U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_2_MAX                      (209U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_2_MIN                      (210U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_3_MAX                      (211U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_3_MIN                      (212U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_4_MAX                      (213U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_4_MIN                      (214U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_5_MAX                      (215U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_5_MIN                      (216U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_6_MAX                      (217U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_6_MIN                      (218U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_7_MAX                      (219U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_7_MIN                      (220U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_8_MAX                      (221U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_8_MIN                      (222U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_9_MAX                      (223U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOOSE_9_MIN                      (224U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES                                  (225U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LWVDD                            (226U)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_MSVDD                            (227U)


/*!
 * Number of suspend factors
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_NUM_LIMITS                                  (228U)

#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PMU_DOM_GRP_0                               LW2080_CTRL_PERF_PERF_LIMIT_ID_PWR_POLICY_DRAM
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_PMU_DOM_GRP_1                               LW2080_CTRL_PERF_PERF_LIMIT_ID_PWR_POLICY_GPC
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_GPU_BOOST_DOM_GRP_0                     LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_GPU_BOOST_DRAM
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_GPU_BOOST_DOM_GRP_1                     LW2080_CTRL_PERF_PERF_LIMIT_ID_SLI_GPU_BOOST_GPC
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_DOM_GRP_0                      LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_DRAM        //<! Thermal policy limit for Domain Group 0 (_PSTATE)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_DOM_GRP_1                      LW2080_CTRL_PERF_PERF_LIMIT_ID_THERM_POLICY_GPC         //<! Thermal policy limit for Domain Group 1 (_GPC2CLK)
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_ALT_SRAM                        LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_ALT_MSVDD    //<! SRAM Reliability Limit - Alternate
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_VMIN_SRAM                                   LW2080_CTRL_PERF_PERF_LIMIT_ID_VMIN_MSVDD               //<! SRAM Vmin Limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_SRAM                             LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_MSVDD         //<! SRAM MODS rules
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_SRAM                            LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_MSVDD        //<! SRAM Reliability Limit - Default
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OVERVOLTAGE_SRAM                            LW2080_CTRL_PERF_PERF_LIMIT_ID_OVERVOLTAGE_MSVDD        //<! SRAM Overvoltage Limit
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_OVERVOLTAGE_LOGIC                           LW2080_CTRL_PERF_PERF_LIMIT_ID_OVERVOLTAGE_LWVDD
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_ALT_LOGIC                       LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_ALT_LWVDD
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_LOGIC                           LW2080_CTRL_PERF_PERF_LIMIT_ID_RELIABILITY_LWVDD
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_VMIN_LOGIC                                  LW2080_CTRL_PERF_PERF_LIMIT_ID_VMIN_LWVDD
#define LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LOGIC                            LW2080_CTRL_PERF_PERF_LIMIT_ID_MODS_RULES_LWVDD


/*
 * LW2080_CTRL_CMD_PERF_GET_LIMITS_INFO
 *
 * This command returns the number of defined limits and information
 * describing each limit
 *
 *   perfNumLimits
 *     This field returns the number of defined limits.
 *   perfLimitInfoList
 *     This field is an array of LW2080_CTRL_PERF_LIMIT_INFO structures
 *     in which each limit's information is returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_PERF_LIMIT_PERFMON                                             0x00U
#define LW2080_CTRL_PERF_LIMIT_NO_CLIENT                                           0x01U
#define LW2080_CTRL_PERF_LIMIT_SCREEN_SAVER                                        0x02U
#define LW2080_CTRL_PERF_LIMIT_NO_HIRES                                            0x03U
#define LW2080_CTRL_PERF_LIMIT_OS_LEVEL                                            0x04U
#define LW2080_CTRL_PERF_LIMIT_SPDIFF_GLITCH                                       0x05U
#define LW2080_CTRL_PERF_LIMIT_DISPLAY_GLITCH                                      0x06U
#define LW2080_CTRL_PERF_LIMIT_UNLOAD_DRIVER                                       0x07U
#define LW2080_CTRL_PERF_LIMIT_POWERMIZER                                          0x08U
#define LW2080_CTRL_PERF_LIMIT_STRESSTEST_FAILURE                                  0x09U
#define LW2080_CTRL_PERF_LIMIT_RC_ERROR                                            0x0aU
#define LW2080_CTRL_PERF_LIMIT_MIN_FAN_LEVEL                                       0x0bU
#define LW2080_CTRL_PERF_LIMIT_MCLK_CLONE                                          0x0lw
#define LW2080_CTRL_PERF_LIMIT_OVERLAY                                             0x0dU
#define LW2080_CTRL_PERF_LIMIT_HIGHRES                                             0x0eU  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_BANDWIDTHFACTOR                                     0x0fU  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_HD_FRAMEDROP_WAR                                    0x10U
#define LW2080_CTRL_PERF_LIMIT_ISMODEPOSSIBLE                                      0x11U
#define LW2080_CTRL_PERF_LIMIT_HYBRID                                              0x12U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_ILWALID_SYSCON                                      0x13U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_STRESSTEST_SETUP                                    0x14U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_FORCED                                              0x15U
#define LW2080_CTRL_PERF_LIMIT_FORCED_DACPERFTEST                                  0x16U
#define LW2080_CTRL_PERF_LIMIT_FORCED_ACSHMOO                                      0x17U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_FORCED_STRESSTEST                                   0x18U
#define LW2080_CTRL_PERF_LIMIT_POWERMIZER_HARD                                     0x19U
#define LW2080_CTRL_PERF_LIMIT_THERMAL                                             0x1aU
#define LW2080_CTRL_PERF_LIMIT_SYSPERF                                             0x1bU
#define LW2080_CTRL_PERF_LIMIT_PWR_SUPPLY_CAPACITY                                 0x1lw
#define LW2080_CTRL_PERF_LIMIT_SW_BATTPOWER                                        0x1dU
#define LW2080_CTRL_PERF_LIMIT_EXT_PERF_CONTROL                                    0x1eU
#define LW2080_CTRL_PERF_LIMIT_MXM_ACPOWER                                         0x1fU
#define LW2080_CTRL_PERF_LIMIT_AUX_POWER                                           0x20U
#define LW2080_CTRL_PERF_LIMIT_SHORT_VBLANK                                        0x21U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_3D_WAR                                              0x22U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_DEEP_IDLE                                           0x23U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_SOFT                                         0x24U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_CLIENT_HARD                                         0x25U
#define LW2080_CTRL_PERF_LIMIT_OVERCLOCK                                           0x26U
#define LW2080_CTRL_PERF_LIMIT_FORCED_LINKTRAIN                                    0x27U
#define LW2080_CTRL_PERF_LIMIT_POWER_BALANCE                                       0x28U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_BUG_535734                                          0x29U
#define LW2080_CTRL_PERF_LIMIT_BOOST                                               0x2aU
#define LW2080_CTRL_PERF_LIMIT_PM_DYNAMIC                                          0x2bU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES                                          0x2lw
#define LW2080_CTRL_PERF_LIMIT_EXCEPT_VIDEO                                        0x2dU
#define LW2080_CTRL_PERF_LIMIT_SDI_INPUT_CAPTURE                                   0x2eU  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_BANDWIDTH_HCLONE                                    0x2fU
#define LW2080_CTRL_PERF_LIMIT_VPS_DISPLAY                                         0x30U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_VPS                                                 0x31U
#define LW2080_CTRL_PERF_LIMIT_CANOAS_MODE                                         0x32U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_BUG_660789                                          0x33U
#define LW2080_CTRL_PERF_LIMIT_P1020_WAR                                           0x34U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_LOCKED_DRIVER                                       0x35U
#define LW2080_CTRL_PERF_LIMIT_PMU_OVERRIDE                                        0x36U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_0_MAX                                        0x37U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_0_MIN                                        0x38U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_1_MAX                                        0x39U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_1_MIN                                        0x3aU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_2_MAX                                        0x3bU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_2_MIN                                        0x3lw
#define LW2080_CTRL_PERF_LIMIT_PERFMON_GROUP_1                                     0x3dU
#define LW2080_CTRL_PERF_LIMIT_PERFMON_GROUP_2                                     0x3eU  //<! RESERVED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_PERFMON_GROUP_3                                     0x3fU  //<! RESERVED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_RATED_TDP_MAX                                       0x40U
#define LW2080_CTRL_PERF_LIMIT_OVERVOLTAGE_LOGIC                                   0x41U
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_LOGIC                                   0x42U
#define LW2080_CTRL_PERF_LIMIT_PMU_DOM_GRP_1                                       0x43U
#define LW2080_CTRL_PERF_LIMIT_AUX_PWR_STATE                                       0x44U
#define LW2080_CTRL_PERF_LIMIT_PERFORMANCE_CAP                                     0x45U
#define LW2080_CTRL_PERF_LIMIT_THERM_POLICY_DOM_GRP_1                              0x46U
#define LW2080_CTRL_PERF_LIMIT_PERFORMANCE_CAP1                                    0x47U
#define LW2080_CTRL_PERF_LIMIT_THERM_POLICY_DOM_GRP_0                              0x48U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_DRAM_MAX                              0x49U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_DRAM_MIN                              0x4aU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_GPC_MAX                               0x4bU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_GPC_MIN                               0x4lw
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_ALT_LOGIC                               0x4dU
#define LW2080_CTRL_PERF_LIMIT_THERM_POLICY_LWVDD                                  0x4eU
#define LW2080_CTRL_PERF_LIMIT_RATED_TDP_MIN                                       0x4fU
#define LW2080_CTRL_PERF_LIMIT_EDP_POLICY_DOM_GRP_1                                0x50U
#define LW2080_CTRL_PERF_LIMIT_SLI_GPU_BOOST_DOM_GRP_0                             0x51U
#define LW2080_CTRL_PERF_LIMIT_SLI_GPU_BOOST_DOM_GRP_1                             0x52U
#define LW2080_CTRL_PERF_LIMIT_ISMODEPOSSIBLE_DISP                                 0x53U
#define LW2080_CTRL_PERF_LIMIT_LWDA_MAX                                            0x54U
#define LW2080_CTRL_PERF_LIMIT_GPU_IS_IDLE                                         0x55U
#define LW2080_CTRL_PERF_LIMIT_SUSPEND_POWER                                       0x56U
#define LW2080_CTRL_PERF_LIMIT_GPU_IS_IDLE_GROUP1                                  0x57U
#define LW2080_CTRL_PERF_LIMIT_SLI_DOM_GRP_0_MIN                                   0x58U
#define LW2080_CTRL_PERF_LIMIT_APPLICATIONCLOCKS                                   0x59U
#define LW2080_CTRL_PERF_LIMIT_LWSTOMER_BOOST_MAX                                  0x5aU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_DISP_MAX                              0x5bU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_DISP_MIN                              0x5lw
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_0_MAX                                  0x5dU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_1_MAX                                  0x5eU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_2_MAX                                  0x5fU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_0_MIN                                  0x60U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_1_MIN                                  0x61U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_2_MIN                                  0x62U
#define LW2080_CTRL_PERF_LIMIT_INTERSECT                                           0x63U
//TODO: Remove these limits. Clean LwAPI
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_LOGIC_1_MAX                               0x64U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_LOGIC_2_MAX                               0x65U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_SRAM_0_MAX                                0x66U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_SRAM_1_MAX                                0x67U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_SRAM_2_MAX                                0x68U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_LOGIC_0_MIN                               0x69U  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_LOGIC_1_MIN                               0x6aU  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_LOGIC_2_MIN                               0x6bU  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_SRAM_0_MIN                                0x6lw  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_SRAM_1_MIN                                0x6dU  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_SRAM_2_MIN                                0x6eU  //<! DEPRECATED - No matching RM PERF_LIMIT_ENTRY
#define LW2080_CTRL_PERF_LIMIT_OVERVOLTAGE_SRAM                                    0x6fU
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_ALT_SRAM                                0x70U
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_SRAM                                    0x71U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOGIC                                    0x72U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_SRAM                                     0x73U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_INTERSECT                                0x74U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_DRAM_MAX                          0x75U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_GPC_MAX                           0x76U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_DISP_MAX                          0x77U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_DRAM_MIN                          0x78U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_GPC_MIN                           0x79U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_DISP_MIN                          0x7aU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_0_MAX                              0x7bU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_1_MAX                              0x7lw
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_2_MAX                              0x7dU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_0_MIN                              0x7eU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_1_MIN                              0x7fU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_2_MIN                              0x80U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_INTERSECT                                0x81U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_DRAM_MAX                          0x82U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_GPC_MAX                           0x83U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_DISP_MAX                          0x84U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_DRAM_MIN                          0x85U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_GPC_MIN                           0x86U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_DISP_MIN                          0x87U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_0_MAX                              0x88U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_1_MAX                              0x89U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_2_MAX                              0x8aU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_0_MIN                              0x8bU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_1_MIN                              0x8lw
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_2_MIN                              0x8dU
#define LW2080_CTRL_PERF_LIMIT_VMIN_LOGIC                                          0x8eU
#define LW2080_CTRL_PERF_LIMIT_VMIN_SRAM                                           0x8fU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_PSTATE_MIN                        0x90U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_PSTATE_MAX                        0x91U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_PSTATE_MAX                            0x92U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_PSTATE_MIN                            0x93U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_PSTATE_MAX                        0x94U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_PSTATE_MIN                        0x95U
#define LW2080_CTRL_PERF_LIMIT_GPU_STATE_LOAD_BOOST_DOM_GRP_0                      0x96U
#define LW2080_CTRL_PERF_LIMIT_GPU_STATE_LOAD_BOOST_DOM_GRP_1                      0x97U
#define LW2080_CTRL_PERF_LIMIT_UNLOAD_DRIVER_PSTATE                                0x98U
#define LW2080_CTRL_PERF_LIMIT_UNLOAD_DRIVER_VOLTAGE_RAIL_0                        0x99U
#define LW2080_CTRL_PERF_LIMIT_UNLOAD_DRIVER_VOLTAGE_RAIL_1                        0x9aU
#define LW2080_CTRL_PERF_LIMIT_UNLOAD_DRIVER_DISP                                  0x9bU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_PCIE_MAX                          0x9lw
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_PCIE_MIN                          0x9dU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_PCIE_MAX                              0x9eU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_PCIE_MIN                              0x9fU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_PCIE_MAX                          0xa0U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_PCIE_MIN                          0xa1U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_3_MAX                                  0xa2U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_4_MAX                                  0xa3U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_5_MAX                                  0xa4U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_6_MAX                                  0xa5U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_7_MAX                                  0xa6U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_8_MAX                                  0xa7U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_9_MAX                                  0xa8U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_3_MIN                                  0xa9U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_4_MIN                                  0xaaU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_5_MIN                                  0xabU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_6_MIN                                  0xalw
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_7_MIN                                  0xadU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_8_MIN                                  0xaeU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_9_MIN                                  0xafU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_3_MAX                              0xb0U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_4_MAX                              0xb1U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_5_MAX                              0xb2U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_6_MAX                              0xb3U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_7_MAX                              0xb4U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_8_MAX                              0xb5U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_9_MAX                              0xb6U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_3_MIN                              0xb7U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_4_MIN                              0xb8U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_5_MIN                              0xb9U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_6_MIN                              0xbaU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_7_MIN                              0xbbU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_8_MIN                              0xblw
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOOSE_9_MIN                              0xbdU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_3_MAX                              0xbeU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_4_MAX                              0xbfU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_5_MAX                              0xc0U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_6_MAX                              0xc1U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_7_MAX                              0xc2U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_8_MAX                              0xc3U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_9_MAX                              0xc4U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_3_MIN                              0xc5U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_4_MIN                              0xc6U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_5_MIN                              0xc7U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_6_MIN                              0xc8U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_7_MIN                              0xc9U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_8_MIN                              0xcaU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOOSE_9_MIN                              0xcbU
#define LW2080_CTRL_PERF_LIMIT_DISP_RESTRICT_GPC_MAX                               0xclw
#define LW2080_CTRL_PERF_LIMIT_PERF_CF_CONTROLLER_DRAM_MIN                         0xd0U
#define LW2080_CTRL_PERF_LIMIT_PERF_CF_CONTROLLER_GPC_MIN                          0xd1U
#define LW2080_CTRL_PERF_LIMIT_PERF_CF_CONTROLLER_LWD_MIN                          0xd2U
#define LW2080_CTRL_PERF_LIMIT_BIF_USBC_PMU_PCIE_MIN                               0xd3U
#define LW2080_CTRL_PERF_LIMIT_BIF_USBC_RM_PCIE_MIN                                0xd4U
#define LW2080_CTRL_PERF_LIMIT_PERF_CF_CONTROLLER_DRAM_MAX                         0xd5U
#define LW2080_CTRL_PERF_LIMIT_PERF_CF_CONTROLLER_GPC_MAX                          0xd6U
#define LW2080_CTRL_PERF_LIMIT_PERF_CF_CONTROLLER_LWD_MAX                          0xd7U
#define LW2080_CTRL_PERF_LIMIT_ECC_MAX                                             0xd8U
#define LW2080_CTRL_PERF_LIMIT_OOB_CLOCK_LIMIT_MAX                                 0xd9U
#define LW2080_CTRL_PERF_LIMIT_OOB_CLOCK_LIMIT_MIN                                 0xdaU
#define LW2080_CTRL_PERF_LIMIT_PERF_CF_CONTROLLER_XBAR_MAX                         0xdbU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOCK_DRAM_MAX                            0xdlw
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_XBAR_MAX                          0xddU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOCK_XBAR_MAX                            0xdeU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOCK_DRAM_MIN                            0xdfU
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_STRICT_XBAR_MIN                          0xe1U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOW_LOCK_XBAR_MIN                            0xe2U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOCK_DRAM_MAX                                0xe3U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_XBAR_MAX                              0xe4U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOCK_XBAR_MAX                                0xe5U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOCK_DRAM_MIN                                0xe6U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_STRICT_XBAR_MIN                              0xe7U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_LOCK_XBAR_MIN                                0xe8U
#define LW2080_CTRL_PERF_LIMIT_THERM_POLICY_XBAR                                   0xe9U
#define LW2080_CTRL_PERF_LIMIT_PWR_POLICY_XBAR                                     0xeaU
#define LW2080_CTRL_PERF_LIMIT_SLI_GPU_BOOST_XBAR                                  0xebU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOCK_DRAM_MAX                            0xelw
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_XBAR_MAX                          0xedU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOCK_XBAR_MAX                            0xeeU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOCK_DRAM_MIN                            0xefU
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_STRICT_XBAR_MIN                          0xf0U
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOCK_XBAR_MIN                            0xf1U
#define LW2080_CTRL_PERF_LIMIT_BOOST_LOW                                           0xf2U
#define LW2080_CTRL_PERF_LIMIT_JPAC_PSTATE_MAX                                     0xf3U
#define LW2080_CTRL_PERF_LIMIT_JPAC_GPC_MAX                                        0xf4U
#define LW2080_CTRL_PERF_LIMIT_JPAC_PSTATE_MIN                                     0xf5U
#define LW2080_CTRL_PERF_LIMIT_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SETUP_LOCK         0xf6U
#define LW2080_CTRL_PERF_LIMIT_JPPC_PSTATE_MAX                                     0xf7U
#define LW2080_CTRL_PERF_LIMIT_JPPC_GPC_MAX                                        0xf8U
#define LW2080_CTRL_PERF_LIMIT_RAM_ASSIST_UNLOAD_DRIVER_VOLTAGE_RAIL_0             0xf9U
#define LW2080_CTRL_PERF_LIMIT_RAM_ASSIST_UNLOAD_DRIVER_VOLTAGE_RAIL_1             0xfaU

// NOTE: UPDATE engines.proto to reflect added perf limits to this list!!
// see also LWAPI_PERF_LIMIT_ID_MAX_LIMITS
#define LW2080_CTRL_PERF_LIMIT_ILWALID                                             0xff
#define LW2080_CTRL_PERF_MAX_LIMITS                                                0x100

#define LW2080_CTRL_PERF_LIMIT_PWR_POLICY_DRAM                                     LW2080_CTRL_PERF_LIMIT_PMU_OVERRIDE             // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_PWR_POLICY_GPC                                      LW2080_CTRL_PERF_LIMIT_PMU_DOM_GRP_1            // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_SLI_GPU_BOOST_DRAM                                  LW2080_CTRL_PERF_LIMIT_SLI_GPU_BOOST_DOM_GRP_0  // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_SLI_GPU_BOOST_GPC                                   LW2080_CTRL_PERF_LIMIT_SLI_GPU_BOOST_DOM_GRP_1  // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_THERM_POLICY_DRAM                                   LW2080_CTRL_PERF_LIMIT_THERM_POLICY_DOM_GRP_0   // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_THERM_POLICY_GPC                                    LW2080_CTRL_PERF_LIMIT_THERM_POLICY_DOM_GRP_1   // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_ALT_MSVDD                               LW2080_CTRL_PERF_LIMIT_RELIABILITY_ALT_SRAM     // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_VMIN_MSVDD                                          LW2080_CTRL_PERF_LIMIT_VMIN_SRAM                // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_MSVDD                                    LW2080_CTRL_PERF_LIMIT_MODS_RULES_SRAM          // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_MSVDD                                   LW2080_CTRL_PERF_LIMIT_RELIABILITY_SRAM         // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_OVERVOLTAGE_MSVDD                                   LW2080_CTRL_PERF_LIMIT_OVERVOLTAGE_SRAM         // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_OVERVOLTAGE_LWVDD                                   LW2080_CTRL_PERF_LIMIT_OVERVOLTAGE_LOGIC        // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_ALT_LWVDD                               LW2080_CTRL_PERF_LIMIT_RELIABILITY_ALT_LOGIC    // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_RELIABILITY_LWVDD                                   LW2080_CTRL_PERF_LIMIT_RELIABILITY_LOGIC        // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_VMIN_LWVDD                                          LW2080_CTRL_PERF_LIMIT_VMIN_LOGIC               // <! Renamnes starting Pstate 4.0
#define LW2080_CTRL_PERF_LIMIT_MODS_RULES_LWVDD                                    LW2080_CTRL_PERF_LIMIT_MODS_RULES_LOGIC         // <! Renamnes starting Pstate 4.0

// PP-TODO : Temporary to unblock RM - LWAPI sync up
#define LW2080_CTRL_PERF_LIMIT_JPAC_PSTATE                                         LW2080_CTRL_PERF_LIMIT_JPAC_PSTATE_MAX
#define LW2080_CTRL_PERF_LIMIT_JPAC_GPC                                            LW2080_CTRL_PERF_LIMIT_JPAC_GPC_MAX
//
// OBSOLETE LIMIT NAMES
// Use the new names -- will be defeatured.
//
#define LW2080_CTRL_PERF_LIMIT_RATED_TDP                                           LW2080_CTRL_PERF_LIMIT_RATED_TDP_MAX
#define LW2080_CTRL_PERF_LIMIT_INTERNAL_CLIENT_0_MAX                               LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_0_MAX
#define LW2080_CTRL_PERF_LIMIT_INTERNAL_CLIENT_0_MIN                               LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_0_MIN
#define LW2080_CTRL_PERF_LIMIT_INTERNAL_CLIENT_1_MAX                               LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_1_MAX
#define LW2080_CTRL_PERF_LIMIT_INTERNAL_CLIENT_1_MIN                               LW2080_CTRL_PERF_LIMIT_CLIENT_LOOSE_1_MIN
#define LW2080_CTRL_PERF_LIMIT_VDT_OVERVOLTAGE                                     LW2080_CTRL_PERF_LIMIT_OVERVOLTAGE_LOGIC
#define LW2080_CTRL_PERF_LIMIT_VDT_RELIABILITY_ALT                                 LW2080_CTRL_PERF_LIMIT_RELIABILITY_ALT_LOGIC
#define LW2080_CTRL_PERF_LIMIT_VDT_RELIABILITY                                     LW2080_CTRL_PERF_LIMIT_RELIABILITY_LOGIC
#define LW2080_CTRL_PERF_LIMIT_INTERSECT_LOGIC_0_MAX                               LW2080_CTRL_PERF_LIMIT_INTERSECT

#define LW2080_CTRL_PERF_LIMIT_NAME_MAX_LENGTH                                     32

#define LW2080_CTRL_PERF_LIMIT_TYPE_MIN                                            0x01
#define LW2080_CTRL_PERF_LIMIT_TYPE_MAX                                            0x02
#define LW2080_CTRL_PERF_LIMIT_TYPE_BOTH                                           0x03


#define LW2080_CTRL_CMD_PERF_GET_LIMITS_INFO                                       (0x20802090) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_LIMITS_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_LIMITS_INFO_PARAMS_MESSAGE_ID (0x90U)

typedef struct LW2080_CTRL_PERF_GET_LIMITS_INFO_PARAMS {
    LwU32                                  perfNumLimits;
    LW2080_CTRL_PERF_LIMIT_INFO_DEPRECATED perfLimitInfoList[LW2080_CTRL_PERF_MAX_LIMITS];
} LW2080_CTRL_PERF_GET_LIMITS_INFO_PARAMS;

/*
 * LW2080_CTRL_PERF_LIMIT_STATUS
 *
 * This structure contains the current state of a defined performance
 * limiting factor.
 *
 *   suspendId
 *     This field is the Id of a suspend
 *   pstate
 *     This field is the current p-state associated with the limit, represented
 *     by a number ranging from 0 to 15. A value of 0xff indicates an unknown
 *     P-State.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_STATUS {
    LwU8 limitId;
    LwU8 pstate;
} LW2080_CTRL_PERF_LIMIT_STATUS;

/*
 * LW2080_CTRL_CMD_PERF_GET_ACTIVE_LIMITS
 *
 * This command returns a list of each active limiting factor and its
 * associated p-state.
 *
 *   perfNumActiveLimits
 *     This field specifies the number of active limit entries being
 *     returned in the associated perfLimitInfoList array.
 *
 *   perfLimitStatusList
 *     This field is an array of LW2080_CTRL_PERF_LIMIT_STATUS structures in
 *     which each active performance limit's status is returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */

#define LW2080_CTRL_CMD_PERF_GET_ACTIVE_LIMITS (0x20802091) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_ACTIVE_LIMITS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_ACTIVE_LIMITS_PARAMS_MESSAGE_ID (0x91U)

typedef struct LW2080_CTRL_PERF_GET_ACTIVE_LIMITS_PARAMS {
    LwU32                         perfNumActiveLimits;
    LW2080_CTRL_PERF_LIMIT_STATUS perfLimitStatusList[LW2080_CTRL_PERF_MAX_LIMITS];
} LW2080_CTRL_PERF_GET_ACTIVE_LIMITS_PARAMS;


/*!
 * LW2080_CTRL_CMD_PERF_LIMITS_GET_INFO
 *
 * This command returns the static information for a specified of PERF_LIMIT
 * ids, primarily whether the PERF_LIMIT is a minimum or a maximum limit.
 *
 * For documentation of parameters, see @ref
 * LW2080_CTRL_PERF_LIMITS_INFO_PARAMS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *
 * Deprecated, please use LW2080_CTRL_CMD_PERF_LIMITS_GET_INFO_V2 instead.
 */
#define LW2080_CTRL_CMD_PERF_LIMITS_GET_INFO          (0x20802076) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x76" */

/*!
 * Flags values for LW2080_CTRL_PERF_LIMIT_INFO.
 *
 * _MIN
 *     This PERF_LIMIT is a minimum limit.  This is not mutually exclusive with
 *     _MAX - if both are set, then this PERF_LIMIT locks to the specified
 *     value.
 * _MAX
 *     This PERF_LIMIT is a maximum limit.  This is not mutually exclusive with
 *     _MIN - if both are set, then this PERF_LIMIT locks to the specified
 *     value.
 *
 * _MINMAX
 *     Alias for _MIN and _MAX above.  Sometimes it is more colwenient to
 *     refer to them together.
 *   _NONE   Not valid.
 *   _MIN    This PERF_LIMIT is a minimum limit.
 *   _MAX    This PERF_LIMIT is a maximum limit.
 *   _BOTH   This PERF_LIMIT is both a min and max.
 *
 * LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_API
 *   Sets LW2080_CTRL_PERF_DECREASE_REASON_API_TRIGGERED.
 *
 * LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_FORCED
 *   Indicates a "forced" limit.
 *
 * LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_SOURCE
 *   For PMU-based arbitration, indicates whether an RM client or PMU client
 *   specifies the limit. For RM-based arbitration, this field is ignored.
 */
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MIN                                0:0
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MIN_FALSE   (0x00000000)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MIN_TRUE    (0x00000001)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MAX                                1:1
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MAX_FALSE   (0x00000000)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MAX_TRUE    (0x00000001)

#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MINMAX                             1:0
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MINMAX_NONE (0x00000000)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MINMAX_MIN  (0x00000001)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MINMAX_MAX  (0x00000002)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_MINMAX_BOTH (0x00000003)

#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_API                                2:2
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_API_NO      (0x00000000)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_API_YES     (0x00000001)

#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_FORCED                             3:3
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_FORCED_NO   (0x00000000)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_FORCED_YES  (0x00000001)

#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_SOURCE                             5:4
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_SOURCE_NONE (0x00000000)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_SOURCE_CPU  (0x00000001)
#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_SOURCE_PMU  (0x00000002)

#define LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_RSVD                              31:6

/*!
 * Structure representing the static information corresponding to a PERF_LIMIT.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INFO {
    /*!
     * In: LW2080_CTRL_PERF_LIMIT_<xyz>
     */
    LwU32 limitId;
    /*!
     * Out: LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_<xyz>
     */
    LwU32 flags;
    /*!
    * Out: Priority of the flag.  Higher number ~ higher priority.  This is
    * really just the internal RM index.
     */
    LwU32 priority;
    /*!
     * Out: String with user-friendly name of the limit.
     */
    char  szName[LW2080_CTRL_PERF_LIMIT_NAME_MAX_LENGTH];
} LW2080_CTRL_PERF_LIMIT_INFO;

/*!
 * Structure representing a set of PERF_LIMITs queried by the caller.
 *
 * Deprecated, please use LW2080_CTRL_PERF_LIMITS_INFO_V2_PARAMS instead.
 */
typedef struct LW2080_CTRL_PERF_LIMITS_INFO_PARAMS {
    /*!
     * In: Number of valid entries in the @ref limits array.
     */
    LwU32 numLimits;
    /*!
     * Out: Pointer to array of PERF_LIMITs to query.  Valid indexes from 0 to @ref
     * numLimits - 1.  Buffer pointed to must be at least as big as numLimits
     * multiplied by the size of LW2080_CTRL_PERF_LIMIT_INFO.
     */
    LW_DECLARE_ALIGNED(LwP64 pLimits, 8);
} LW2080_CTRL_PERF_LIMITS_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_LIMITS_GET_INFO_V2
 *
 * Same as LW2080_CTRL_CMD_PERF_LIMITS_GET_INFO but with fixed sized array and list-style naming.
 */
#define LW2080_CTRL_CMD_PERF_LIMITS_GET_INFO_V2 (0x2080202d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_LIMITS_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_LIMITS_INFO_V2_PARAMS_MESSAGE_ID (0x2DU)

typedef struct LW2080_CTRL_PERF_LIMITS_INFO_V2_PARAMS {
    /*!
     * In: Number of valid entries in the @ref limits array.
     */
    LwU32                       numLimits;
    /*!
     * Out: Array of PERF_LIMITs to query.
     */
    LW2080_CTRL_PERF_LIMIT_INFO limitsList[LW2080_CTRL_PERF_MAX_LIMITS];
} LW2080_CTRL_PERF_LIMITS_INFO_V2_PARAMS;

/*!
 * Encoding for type of PERF_LIMIT specified by callers as input to
 * perfSetLimit():
 *
 * _DISABLED
 *     PERF_LIMIT is lwrrently disabled.
 * _PSTATE
 *     PERF_LIMIT is specified with a pstate.
 * _FREQ
 *     PERF_LIMIT is specified with a frequency value on a clock domain.
 * _VOLTAGE
 *     PERF_LIMIT is specified with a voltage value, applied to domain group.
 * _UNSUPPORTED
 *     PERF_LIMIT is specified with a type supported by the RM internally, but
 *     not yet supported by RMCTRL.  Support for this type needs to be added to
 *     RMCTRL.
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_DISABLED    0x00000000
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_PSTATE      0x00000001
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_FREQ        0x00000002
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VPSTATE     0x00000003
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VOLTAGE_2X  0x00000004
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VOLTAGE     LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VOLTAGE_2X
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VPSTATE_IDX 0x00000005
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VOLTAGE_3X  0x00000006
#define LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_UNSUPPORTED 0xFFFFFFFF

/*!
 * Encoding for PERF_LIMIT_INPUT flags specified by callers as input to
 * perfSetLimit():
 *
 * Lwrrently all RSVD.
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_FLAGS_RSVD                             31:0

/*!
 * Enumeration for pstate points. Used to select clock values for
 * input pstate index
 */
typedef LwU32 LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT;

/*!
 * Choose nominal clocks for given pstate Index
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_NOM              (0x00U)
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT_NOM  LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_NOM
/*!
 * Choose minimum clocks for given pstate Index
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_MIN              (0x01U)
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT_MIN  LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_MIN
/*!
 * Choose maximum clocks for given pstate Index
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_MAX              (0x02U)
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT_MAX  LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_MAX
/*!
 * Choose maximum clocks for given pstate Index
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_MID              (0x03U)
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT_MID  LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_MID

/*!
 * Last point. Add all new points before this.
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_LAST             (0x04U)
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT_LAST LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_LAST

/*!
 * @copydoc LW2080_CTRL_PERF_PERF_LIMIT_INPUT_DATA_PSTATE_POINT
 *
 * To remain compatibility with existing clients. Remove once clients have
 * colwerted to new enumeration.
 */
typedef LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT;

/*!
 * Input data corresponding to LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_PSTATE.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE {
    /*!
     * LW2080_CTRL_PERF_PSTATES_<xyz>
     */
    LwU32                                          pstateId;
    /*!
     * LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT_<xyz>
     */
    LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE_POINT point;
} LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE;
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE *PLW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE;

/*!
 * Input data corresponding to LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_FREQ.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_FREQ {
    /*!
     * Frequency (KHz)
     */
    LwU32 freqKHz;
    /*!
     * LW2080_CTRL_CLK_DOMAIN_<xyz>
     */
    LwU32 domain;
} LW2080_CTRL_PERF_LIMIT_INPUT_DATA_FREQ;
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_FREQ *PLW2080_CTRL_PERF_LIMIT_INPUT_DATA_FREQ;

/*!
 * Reserved value for invalid/skipped vP-state.
 */
#define LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VPSTATE_ILWALID 0xFFFFFFFF

/*!
 * Input data corresponding to LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VPSTATE.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VPSTATE {
    /*!
     * vP-state number.
     */
    LwU32 vpstate;

    /*!
     * vP-state index.
     */
    LwU32 vpstateIdx;
} LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VPSTATE;
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VPSTATE *PLW2080_CTRL_PERF_LIMIT_INPUT_DATA_VPSTATE;

/*!
 * Input data corresponding to LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VOLTAGE.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE_2X {
    /*!
     * Voltage input information.
     */
    LW2080_CTRL_PERF_VOLT_DOM_INFO info;
    /*!
     * Decoupled clock domain mask (of LW2080_CTRL_CLK_DOMAIN_XXX) to cap with voltage info.
     *
     * ~~~PS20 TODO~~~ This can be moved to the main _INPUT structure when we
     * add a target clock domain mask for all PERF_LIMIT specifications.
     */
    LwU32                          clkDomain;
} LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE_2X;
typedef LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE_2X LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE;
typedef LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE *PLW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE;

/*!
 * Max elements for PERF_LIMIT_VOLTAGE_DATA
 */
#define LW2080_CTRL_PERF_LIMIT_VOLTAGE_DATA_ELEMENTS_MAX 8

/*!
 * Input data corresponding to LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_VOLTAGE_3X.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE_3X {
    /*!
     * Volt domain index for input elements
     */
    LwU32                          voltDomain;
    /*!
     * Number of elements input
     */
    LwU32                          numElements;
    /*!
     * Voltage input information.
     */
    LW2080_CTRL_PERF_VOLT_DOM_INFO info[LW2080_CTRL_PERF_LIMIT_VOLTAGE_DATA_ELEMENTS_MAX];
    /*!
     * Global voltage delta. This is applied to every element along with
     * each element's local delta.
     */
    LwS32                          deltauV;
    /*!
     * Max target voltage across all elements.
     */
    LwU32                          lwrrTargetVoltageuV;
} LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE_3X;

/*!
 * Union of type-specific input data.
 */



/*!
 * Structure encoding the input to perfSetLimit() by the caller, specifying how
 * the caller wanted the PERF_LIMIT to affect perf/clocks.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_INPUT {
    /*!
     * LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_<xyz>
     */
    LwU32 type;

    /*!
     * @ref LW2080_CTRL_PERF_LIMIT_INPUT_FLAGS_<xyz>
     */
    LwU32 flags;

    /*!
     * Union of type-specific data.
     */
    union {
        LW2080_CTRL_PERF_LIMIT_INPUT_DATA_PSTATE     pstate;
        LW2080_CTRL_PERF_LIMIT_INPUT_DATA_FREQ       freq;
        LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VPSTATE    vpstate;
        LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE_2X volt;
        LW2080_CTRL_PERF_LIMIT_INPUT_DATA_VOLTAGE_3X volt3x;
    } data;
} LW2080_CTRL_PERF_LIMIT_INPUT;

/*!
 * Reserved value for invalid/skipped entries.
 */
#define LW2080_CTRL_PERF_LIMIT_OUTPUT_VALUE_ILWALID 0xFFFFFFFF

/*!
 * Structure encoding the output PERF_LIMIT values as determined by
 * perfSetLimit(), the domain group values which are affecting perf/clocks.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_OUTPUT {
    /*!
     * Boolean indicating whether the PERF_LIMIT is lwrrently enabled/disabled.
     */
    LwBool bEnabled;
    /*!
     * Decoupled clock domain (of LW2080_CTRL_CLK_DOMAIN_XXX) affected by the PERF_LIMIT or _UNDEFINED for P-state.
     */
    LwU32  clkDomain;
    /*!
     * Domain-group-specific value enforced by the PERF_LIMIT.
     */
    LwU32  value;
} LW2080_CTRL_PERF_LIMIT_OUTPUT;

/*!
 * LW2080_CTRL_CMD_PERF_LIMITS_GET_STATUS
 *
 * This command returns the current status of a specified of PERF_LIMIT ids,
 * which are enforcing various limitations on perf/clocks as inputs to the perf
 * clock control arbiter.
 *
 * For documentation of parameters, see @ref
 * LW2080_CTRL_PERF_LIMITS_GET_STATUS_PARAMS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *
 * Deprecated, please use LW2080_CTRL_CMD_PERF_LIMITS_GET_STATUS_V2 instead.
 */
#define LW2080_CTRL_CMD_PERF_LIMITS_GET_STATUS (0x20802077) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x77" */

/*!
 * Structure representing the status of a given PERF_LIMIT.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_GET_STATUS {
    /*!
     * In: LW2080_CTRL_PERF_LIMIT_<xyz> requested by the caller.
     */
    LwU32                         limitId;
    /*!
     * Out: Input values specified to perfSetLimit() by caller.
     */
    LW2080_CTRL_PERF_LIMIT_INPUT  input;
    /*!
     * Out: Output values as determined by perfSetLimit().
     */
    LW2080_CTRL_PERF_LIMIT_OUTPUT output;
} LW2080_CTRL_PERF_LIMIT_GET_STATUS;

/*!
 * Structure representing a set of PERF_LIMITs queried by the caller.
 *
 * Deprecated, please use LW2080_CTRL_PERF_LIMITS_GET_STATUS_V2_PARAMS instead.
 */
typedef struct LW2080_CTRL_PERF_LIMITS_GET_STATUS_PARAMS {
    /*!
     * In: Number of valid entries in the @ref limits array.
     */
    LwU32 numLimits;
    /*!
     * In: Pointer to array of PERF_LIMITs to query.  Valid indexes from 0 to @ref
     * numLimits - 1.  Buffer pointed to must be at least as big as numLimits
     * multiplied by the size of LW2080_CTRL_PERF_LIMIT_GET_STATUS.
     */
    LW_DECLARE_ALIGNED(LwP64 pLimits, 8);
} LW2080_CTRL_PERF_LIMITS_GET_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_LIMITS_SET_STATUS
 *
 * This command sets the current status of a specified of PERF_LIMIT ids,
 * to enforce limitations on perf/clocks as inputs to the perf
 * clock control arbiter.
 *
 * Will update the settings for the supplied set of PERF_LIMITs and then trigger
 * a pstate/clock change to apply the new limitations to the HW.
 *
 * For documentation of parameters, see @ref
 * LW2080_CTRL_PERF_LIMITS_SET_STATUS_PARAMS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *
 * Deprecated, please use LW2080_CTRL_CMD_PERF_LIMITS_SET_STATUS_V2 instead.
 */
#define LW2080_CTRL_CMD_PERF_LIMITS_SET_STATUS (0x20802078) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x78" */

/*!
 * Structure representing the requested status/settings for a PERF_LIMIT.
 */
typedef struct LW2080_CTRL_PERF_LIMIT_SET_STATUS {
    /*!
     * In: LW2080_CTRL_PERF_LIMIT_<xyz> requested by the caller.
     */
    LwU32                        limitId;
    /*!
     * In: Input values to be specified to perfSetLimit().
     */
    LW2080_CTRL_PERF_LIMIT_INPUT input;
} LW2080_CTRL_PERF_LIMIT_SET_STATUS;

/*!
 * Flags for LW2080_CTRL_PERF_LIMITS_SET_STATUS_PARAMS.
 *
 * _ASYNC
 *     Specifies whether pstate/clocks change after applying new settings should
 *     be asynchronous.
 */
#define LW2080_CTRL_PERF_LIMITS_SET_STATUS_FLAGS_ASYNC                       0:0
#define LW2080_CTRL_PERF_LIMITS_SET_STATUS_FLAGS_ASYNC_FALSE 0x00000000
#define LW2080_CTRL_PERF_LIMITS_SET_STATUS_FLAGS_ASYNC_TRUE  0x00000001
#define LW2080_CTRL_PERF_LIMITS_SET_STATUS_FLAGS_RSVD                       31:1

/*!
 * Structure representing a set of PERF_LIMITs set by the caller.
 *
 * Deprecated, please use LW2080_CTRL_PERF_LIMITS_SET_STATUS_V2_PARAMS instead.
 */
typedef struct LW2080_CTRL_PERF_LIMITS_SET_STATUS_PARAMS {
    /*!
     * In: Number of valid entries in the @ref limits array.
     */
    LwU32 numLimits;
    /*!
     * In: LW2080_CTRL_PERF_LIMITS_SET_STATUS_FLAGS_<xyz>
     */
    LwU32 flags;
    /*!
     * In: Pointer to array of PERF_LIMITs to set.  Valid indexes from 0 to @ref
     * numLimits - 1.  Buffer pointed to must be at least as big as numLimits
     * multiplied by the size of LW2080_CTRL_PERF_LIMIT_SET_STATUS.
     */
    LW_DECLARE_ALIGNED(LwP64 pLimits, 8);
} LW2080_CTRL_PERF_LIMITS_SET_STATUS_PARAMS;

/*!
 * V2 of LW2080_CTRL_CMD_PERF_LIMITS_SET_STATUS that doesn't use an embedded pointer.
 */
#define LW2080_CTRL_CMD_PERF_LIMITS_SET_STATUS_V2 (0x208020af) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_LIMITS_SET_STATUS_V2_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing a set of PERF_LIMITs set by the caller.
 */
#define LW2080_CTRL_PERF_LIMITS_SET_STATUS_V2_PARAMS_MESSAGE_ID (0xAFU)

typedef struct LW2080_CTRL_PERF_LIMITS_SET_STATUS_V2_PARAMS {
    /*!
     * In: Number of valid entries in the @ref limits array.
     */
    LwU32                             numLimits;
    /*!
     * In: LW2080_CTRL_PERF_LIMITS_SET_STATUS_FLAGS_<xyz>
     */
    LwU32                             flags;
    /*!
     * In: Pointer to array of PERF_LIMITs to set.  Valid indexes from 0 to @ref
     * numLimits - 1.
     */
    LW2080_CTRL_PERF_LIMIT_SET_STATUS limitsList[LW2080_CTRL_PERF_MAX_LIMITS];
} LW2080_CTRL_PERF_LIMITS_SET_STATUS_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_MXM_POWER_LEVEL
 *
 * This command allows the forcing of a performance level based on auxiliary
 * power states defined in MXM SIS.
 *
 *   powerLevel
 *     This parameter specifies the target auxiliary Power state. Legal
 *     aux power states for this parameter are defined in ctrl0000system.h.
 *     see the LW0000_CTRL_SYSTEM_EVENT_POWER_LEVEL #defines.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_SET_MXM_POWER_LEVEL (0x20802092) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x92" */

typedef struct LW2080_CTRL_PERF_SET_MXM_POWER_LEVEL_PARAMS {
    LwU32 powerLevel;
} LW2080_CTRL_PERF_SET_MXM_POWER_LEVEL_PARAMS;

/* Aux power states for MXM */
#define LW2080_CTRL_PERF_SET_MXM_POWER_LEVEL_P0  (0x00000000)
#define LW2080_CTRL_PERF_SET_MXM_POWER_LEVEL_P1  (0x00000001)
#define LW2080_CTRL_PERF_SET_MXM_POWER_LEVEL_P2  (0x00000002)
#define LW2080_CTRL_PERF_SET_MXM_POWER_LEVEL_P3  (0x00000003)
#define LW2080_CTRL_PERF_SET_MXM_POWER_LEVEL_P4  (0x00000004)

/*
 * LW2080_CTRL_CMD_PERF_SET_AUX_POWER_STATE
 *
 * This command allows the forcing of a performance level based on auxiliary
 * power-states.
 *
 *   powerState
 *     This parameter specifies the target auxiliary Power state. Legal aux
 *     power-states for this parameter are defined by the
 *     LW2080_CTRL_PERF_AUX_POWER_STATE_P* definitions that follow.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_SET_AUX_POWER_STATE (0x20802092) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_AUX_POWER_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_AUX_POWER_STATE_PARAMS_MESSAGE_ID (0x92U)

typedef struct LW2080_CTRL_PERF_SET_AUX_POWER_STATE_PARAMS {
    LwU32 powerState;
} LW2080_CTRL_PERF_SET_AUX_POWER_STATE_PARAMS;

#define LW2080_CTRL_PERF_AUX_POWER_STATE_P0    (0x00000000)
#define LW2080_CTRL_PERF_AUX_POWER_STATE_P1    (0x00000001)
#define LW2080_CTRL_PERF_AUX_POWER_STATE_P2    (0x00000002)
#define LW2080_CTRL_PERF_AUX_POWER_STATE_P3    (0x00000003)
#define LW2080_CTRL_PERF_AUX_POWER_STATE_P4    (0x00000004)
#define LW2080_CTRL_PERF_AUX_POWER_STATE_COUNT (0x00000005)

/*!
 * Structure representing a set of PERF_LIMITs queried by the caller.
 */
#define LW2080_CTRL_PERF_LIMITS_GET_STATUS_V2_PARAMS_MESSAGE_ID (0x79U)

typedef struct LW2080_CTRL_PERF_LIMITS_GET_STATUS_V2_PARAMS {
    /*!
     * In: Number of valid entries in the @ref limits array.
     */
    LwU32                             numLimits;
    /*!
     * In: Array of PERF_LIMITs to query.  Valid indexes from 0 to @ref
     * numLimits - 1.
     */
    LW2080_CTRL_PERF_LIMIT_GET_STATUS limitsList[LW2080_CTRL_PERF_MAX_LIMITS];
} LW2080_CTRL_PERF_LIMITS_GET_STATUS_V2_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_LIMITS_GET_STATUS_V2
 *
 * This command returns the current status of a specified of PERF_LIMIT ids,
 * which are enforcing various limitations on perf/clocks as inputs to the perf
 * clock control arbiter.
 *
 * For documentation of parameters, see @ref
 * LW2080_CTRL_PERF_LIMITS_GET_STATUS_PARAMS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_PERF_LIMITS_GET_STATUS_V2 (0x20802079) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_LIMITS_GET_STATUS_V2_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_PERF_RESERVE_PERFMON_HW
 *
 *  This command reserves HW Performance Monitoring capabilities for exclusive
 *  use by the requester.  If the HW Performance Monitoring capabilities are
 *  lwrrently in use then LWOS_STATUS_ERROR_STATE_IN_USE is returned.
 *
 *   bAcquire
 *     When set to TRUE this parameter indicates that the client wants to
 *     acquire the Performance Monitoring capabilities on the subdevice.
 *     When set to FALSE this parameter releases the Performance Monitoring
 *     capabilities on the subdevice.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_ERROR_STATE_IN_USE
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_PERF_RESERVE_PERFMON_HW   (0x20802093) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_RESERVE_PERFMON_HW_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_RESERVE_PERFMON_HW_PARAMS_MESSAGE_ID (0x93U)

typedef struct LW2080_CTRL_PERF_RESERVE_PERFMON_HW_PARAMS {
    LwBool bAcquire;
} LW2080_CTRL_PERF_RESERVE_PERFMON_HW_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_VPS_GET_PWM
 *
 *  This command gets the current virtual P-States PWM status.
 *  When running, we will PWM between a high P-State and a low
 *  P-State for their respective durations.
 *
 *   flags
 *     LW2080_CTRL_PERF_VPS_PWM_FLAGS_ENABLE
 *       This field specifies whether virtual P-States PWM is
 *       enabled or not.
 *     LW2080_CTRL_PERF_VPS_PWM_FLAGS_RUNNING
 *       If virtual P-States PWM is enabled, this field specifies
 *       whether it is lwrrently PWM-ing or not.
 *     LW2080_CTRL_PERF_VPS_PWM_FLAGS_STATE
 *       If virtual P-States is lwrrently PWM-ing, this field
 *       specifies the current PWM state (high or low).
 *   highPstate
 *     If virtual P-States PWM is enabled, this field specifies
 *     the P-State used for high state.
 *   lowPstate
 *     If virtual P-States PWM is enabled, this field specifies
 *     the P-State used for high state.
 *   highUs
 *     This field specifies the duration we would stay in the high
 *     P-State.
 *   lowUs
 *     This field specifies the duration we would stay in the low
 *     P-State.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_VPS_GET_PWM             (0x2080200d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xD" */

#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_ENABLE           0:0
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_ENABLE_FALSE  (0x00000000)
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_ENABLE_TRUE   (0x00000001)
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_RUNNING          1:1
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_RUNNING_FALSE (0x00000000)
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_RUNNING_TRUE  (0x00000001)
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_STATE            2:2
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_STATE_LOW     (0x00000000)
#define LW2080_CTRL_PERF_VPS_PWM_FLAGS_STATE_HIGH    (0x00000001)

typedef struct LW2080_CTRL_PERF_VPS_PWM_PARAMS {
    LwU32 flags;
    LwU32 highPstate;
    LwU32 lowPstate;
    LwU32 highUs;
    LwU32 lowUs;
} LW2080_CTRL_PERF_VPS_PWM_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_VPS_SET_INFO
 *
 *  This command sets the target virtual P-States limit.
 *
 *   flags
 *     LW2080_CTRL_PERF_VPS_INFO_FLAGS_ENABLE
 *       This field specifies whether to enable virtual P-States
 *       limit or not.
 *   target
 *     If LW2080_CTRL_PERF_VPS_INFO_FLAGS_ENABLE == TRUE,
 *     this field specifies the new target virtual P-State limit.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_VPS_SET_INFO            (0x2080200e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xE" */

/*
 * LW2080_CTRL_CMD_PERF_VPS_GET_INFO
 *
 *  This command gets the current actual virtual P-States limit.
 *
 *   flags
 *     LW2080_CTRL_PERF_VPS_INFO_FLAGS_ENABLE
 *       This field indicates whether virtual P-States limit is
 *       enabled or not.
 *   actual
 *     This field reports the actual virtual P-States limit.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_VPS_GET_INFO            (0x2080200f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xF" */

#define LW2080_CTRL_PERF_VPS_INFO_FLAGS_ENABLE          0:0
#define LW2080_CTRL_PERF_VPS_INFO_FLAGS_ENABLE_FALSE (0x00000000)
#define LW2080_CTRL_PERF_VPS_INFO_FLAGS_ENABLE_TRUE  (0x00000001)

typedef struct LW2080_CTRL_PERF_VPS_INFO_PARAMS {
    LwU32 flags;
    LwU8  target;
    LwU8  actual;
} LW2080_CTRL_PERF_VPS_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_FSTATE
 *
 *  This command gets the current and max F-states.
 *
 *   flags
 *     It is reserved for future use.
 *   current
 *     This field reports the current F-state.
 *   max
 *     This field reports the maximum F-state.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_FSTATE (0x20802015) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_FSTATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_FSTATE_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW2080_CTRL_PERF_GET_FSTATE_PARAMS {
    LwU32 flags;
    LwU8  current;
    LwU8  max;
} LW2080_CTRL_PERF_GET_FSTATE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_VPS_SET_PERIOD
 *
 *  This command sets the virtual P-States PWM period. It can only be changed
 *  when virtual P-States limit is disabled.
 *
 *   flags
 *     Reserved for future use.
 *   periodUs
 *     This field specifies the new PWM period in us.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_PERF_VPS_SET_PERIOD (0x20802011) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x11" */

typedef struct LW2080_CTRL_PERF_VPS_PERIOD_PARAMS {
    LwU32 flags;
    LwU32 periodUs;
} LW2080_CTRL_PERF_VPS_PERIOD_PARAMS;

#define LW2080_CTRL_PERF_VPS_PERIOD_AUTO              (0x00000000)

/*
 * LW2080_CTRL_CMD_PERF_ENSURE_HCLONE_BANDWIDTHS
 *
 *  This command disable ASLM and sets a pstate which satisfy copy engine
 *  bandwidth requirement in hclone mode and release these limits when we
 *  are no longer in hclone mode (hcloneBandwidthReq is 0).
 *
 *   hcloneBandwidthReq
 *     The bandwidth required (in MB/s) for hclone mode.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_ENSURE_HCLONE_BANDWIDTHS (0x20802012) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x12" */

typedef struct LW2080_CTRL_PERF_ENSURE_HCLONE_BANDWIDTHS_PARAMS {
    LwU32 hcloneBandwidthReq;
} LW2080_CTRL_PERF_ENSURE_HCLONE_BANDWIDTHS_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_VPS_HANDLE_SBIOS_EVENT
 *
 *  This command is issed to handle SBIOS event requesting
 *  change in associated GPU's power settings.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_PERF_VPS_HANDLE_SBIOS_EVENT                  (0x20802013) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x13" */

/*
 * Reasons for GPU perf decrease
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_SHUTDOWN
 *       GPU slowdown/shutdown.
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_PROTECTION
 *       POR thermal protection.
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_AVERAGE_POWER
 *       Power capping - Slow/Pstate cap
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_PEAK_POWER
 *       Power capping - Moderately Fast
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_INSUFFICIENT_POWER
 *       Power connector missing
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_AC_BATT
 *       AC->BATT event.
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_API_TRIGGERED
 *       API triggered slowdown.
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_POWER_BRAKE
 *       External Power Brake event.
 *
 *   LW2080_CTRL_PERF_DECREASE_REASON_UNKNOWN
 *       Unknown reason.
 */

#define LW2080_CTRL_PERF_DECREASE_NONE                               0

#define LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_SHUTDOWN                       0:0
#define LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_SHUTDOWN_INACTIVE   (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_SHUTDOWN_ACTIVE     (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_PROTECTION                     1:1
#define LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_PROTECTION_INACTIVE (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_THERMAL_PROTECTION_ACTIVE   (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_AVERAGE_POWER                          2:2
#define LW2080_CTRL_PERF_DECREASE_REASON_AVERAGE_POWER_INACTIVE      (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_AVERAGE_POWER_ACTIVE        (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_PEAK_POWER                             3:3
#define LW2080_CTRL_PERF_DECREASE_REASON_PEAK_POWER_INACTIVE         (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_PEAK_POWER_ACTIVE           (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_INSUFFICIENT_POWER                     4:4
#define LW2080_CTRL_PERF_DECREASE_REASON_INSUFFICIENT_POWER_INACTIVE (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_INSUFFICIENT_POWER_ACTIVE   (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_AC_BATT                                5:5
#define LW2080_CTRL_PERF_DECREASE_REASON_AC_BATT_INACTIVE            (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_AC_BATT_ACTIVE              (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_API_TRIGGERED                          6:6
#define LW2080_CTRL_PERF_DECREASE_REASON_API_TRIGGERED_INACTIVE      (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_API_TRIGGERED_ACTIVE        (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_POWER_BRAKE                            7:7
#define LW2080_CTRL_PERF_DECREASE_REASON_POWER_BRAKE_INACTIVE        (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_POWER_BRAKE_ACTIVE          (0x00000001)

#define LW2080_CTRL_PERF_DECREASE_REASON_UNKNOWN                              31:31
#define LW2080_CTRL_PERF_DECREASE_REASON_UNKNOWN_INACTIVE            (0x00000000)
#define LW2080_CTRL_PERF_DECREASE_REASON_UNKNOWN_ACTIVE              (0x00000001)

/*!
 * LW2080_CTRL_CMD_PERF_GET_PERF_DECREASE_INFO
 *
 * Retrieves current system perf decrease info in form of two masks.
 *
 * thermalMask
 *      Perf decrease due to any of LW2080_CTRL_PERF_DECREASE_REASON_<xyz>
 *      reasons that result in thermal slowdown.
 *
 * pstateMask
 *      Perf decrease due to any of LW2080_CTRL_PERF_DECREASE_REASON_<xyz>
 *      reasons that result in pstate cap.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_POINTER
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_PERF_DECREASE_INFO                  (0x20802014) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_CMD_PERF_GET_PERF_DECREASE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_PERF_GET_PERF_DECREASE_INFO_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW2080_CTRL_CMD_PERF_GET_PERF_DECREASE_INFO_PARAMS {
    LwU32 thermalMask;
    LwU32 pstateMask;
} LW2080_CTRL_CMD_PERF_GET_PERF_DECREASE_INFO_PARAMS;

//
// NJ-TODO: Defines for flags will be added at the same time when they will get
// introduced to the implementation of the RmCtrl calls.
//

/*!
 * LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_SINGLE
 *
 * This structure encapsulates clock domain information specific to _FIXED and
 * _PSTATE usage types (LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_<xyz>).
 *
 *  freqkHz
 *    This parameter returns the clock frequency in kHz (including overclocking).
 */
typedef struct LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_SINGLE {
    LwU32 freqkHz;
} LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_SINGLE;

/*!
 * LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_RANGE
 *
 * This structure encapsulates clock domain information specific to _DECOUPLED
 * and _RATIO usage types (LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_<xyz>).
 *
 *  minFreqkHz
 *    This parameter returns the min clock frequency in kHz.
 *  maxFreqkHz
 *    This parameter returns the max clock frequency in kHz. If this clock domain
 *    was overclocked than the settings will reflect on this (upper) boundary.
 *  voltageDomain
 *    This parameter returns the ID (LW2080_CTRL_PERF_VOLTAGE_DOMAINS_<xyz>) of
 *    the voltage domain that is linked with this clock domain.
 *  minFreqVoltageuV
 *    This parameter returns the min voltage in uV requested by clock domain
 *    when using LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_RANGE::minFreqkHz.
 *  maxFreqVoltageuV
 *    This parameter returns the min voltage in uV requested by clock domain
 *    when using LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_RANGE::maxFreqkHz.
 */
typedef struct LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_RANGE {
    LwU32 minFreqkHz;
    LwU32 maxFreqkHz;
    LwU32 voltageDomain;
    LwU32 minFreqVoltageuV;
    LwU32 maxFreqVoltageuV;
} LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_RANGE;

/*!
 * LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO
 *
 * This structure describes single clock domain.
 *
 *  domain [IN]
 *    This parameter specifies an unique clock domain identifier as defined in
 *    LW2080_CTRL_CLK_DOMAIN_<xyz>
 *  flags
 *    This parameter returns flags specific to current clock domain.
 *      LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_FLAGS_EDITABLE
 *        This field indicates that current clock domain can be modified.
 *  type
 *    This parameter returns the usage type of clock domain as defined in
 *    LW2080_CTRL_CLK_PSTATES2_INFO_FLAGS_USAGE_<xyz>
 *  data
 *    This union holds clock domain usage type specific information
 *    (@see LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO::type).
 *  freqDeltakHz [IN-SET]
 *    This parameter returns the deviation of current clock frequency settings
 *    LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO::data.*** from the nominal settings
 *    (from the non-over-clocked state).
 */
typedef struct LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO {
    LwU32 domain;
    LwU32 flags;

    LwU8  type;
    union {
        LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_SINGLE fixed;
        LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_SINGLE pstate;
        LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_RANGE  decoupled;
        LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_RANGE  ratio;
    } data;

    LW2080_CTRL_PERF_PARAM_DELTA freqDeltakHz;
} LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO;

#define LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_FLAGS_EDITABLE       0:0
#define LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_FLAGS_EDITABLE_NO  (0x00000000)
#define LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO_FLAGS_EDITABLE_YES (0x00000001)

/*!
 * LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO
 *
 * @see  LW2080_CTRL_PERF_VOLT_DOM_INFO
 *
 * Reusing existing structure to reduce RM code size increase since it contains
 * all necessary fields.
 *
 * Updated parameters:
 *
 *   flags
 *     This parameter returns flags specific to current voltage domain.
 *       LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO_FLAGS_EDITABLE
 *         This field indicates that current voltage domain can be modified.
 */
typedef LW2080_CTRL_PERF_VOLT_DOM_INFO LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO;

#define LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO_FLAGS_EDITABLE      0:0
#define LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO_FLAGS_EDITABLE_NO           (0x00000000)
#define LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO_FLAGS_EDITABLE_YES          (0x00000001)

#define LW2080_CTRL_PERF_PSTATE20_FLAGS_EDITABLE                            0:0
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_EDITABLE_NO                         (0x00000000)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_EDITABLE_YES                        (0x00000001)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L0S                        1:1
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L0S_ENABLE                 (0x00000001)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L0S_DISABLE                (0x00000000)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L1                         2:2
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L1_ENABLE                  (0x00000001)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L1_DISABLE                 (0x00000000)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_DEEPL1                     3:3
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_DEEPL1_ENABLE              (0x00000001)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_DEEPL1_DISABLE             (0x00000000)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_SET_ASPM                   4:4
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_SET_ASPM_ENABLE            (0x00000001)
#define LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_SET_ASPM_DISABLE           (0x00000000)

// Keep this in sync with LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_FLAGS.
#define LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_FLAGS_RSVD0              0:0
#define LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_FLAGS_MODE               1:1
#define LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_FLAGS_MODE_OVERCLOCKING  (0x00000000)
#define LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_FLAGS_MODE_INTERNAL_TEST (0x00000001)

/*!
 * To restrict the RMCTRL size to XAPI MAX limit, we are using safe limit for
 * maximum possible clock domains.
 * TO_DO: Remove this HACK once we solve XAPI LIMIT issue/ Or come up with better
 * solution to meet the MAX limit.
 */
#define LW2080_CTRL_PERF_PSTATE_MAX_CLK_DOMAINS                             16U


/*!
 * Enumeration of the PERF_PSTATE flags.
 */
#define LW2080_CTRL_PERF_PSTATE_FLAGS_RSVD1                                    0:0  // RSVD1
#define LW2080_CTRL_PERF_PSTATE_FLAGS_LOW_PWR_EXCLUDE                          1:1  // Level is excluded from use while on low power
#define LW2080_CTRL_PERF_PSTATE_FLAGS_LOW_PWR_EXCLUDE_NO                    0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_LOW_PWR_EXCLUDE_YES                   0x00000001
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_EXCLUDE                         2:2  // Level is excluded from use while ext. perf control
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_EXCLUDE_NO                   0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_EXCLUDE_YES                  0x00000001
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_SLI_EXCLUDE                     3:3  // Level is excluded from use while ext. perf control (SLI) bit
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_SLI_EXCLUDE_NO               0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_SLI_EXCLUDE_YES              0x00000001
#define LW2080_CTRL_PERF_PSTATE_FLAGS_OVERCLOCKING                             5:5  // Level for Notebook Overclocking
#define LW2080_CTRL_PERF_PSTATE_FLAGS_OVERCLOCKING_DISABLED                 0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_OVERCLOCKING_ENABLED                  0x00000001
#define LW2080_CTRL_PERF_PSTATE_FLAGS_RSVD2                                    6:6  // RSVD2
#define LW2080_CTRL_PERF_PSTATE_FLAGS_LWDA_SAFE                                7:7  // P-state level is safe for LWCA.
#define LW2080_CTRL_PERF_PSTATE_FLAGS_LWDA_SAFE_DISABLED                    0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_LWDA_SAFE_ENABLED                     0x00000001
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_CAP                             9:8  // Ext. perf control p-state cap behavior
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_CAP_RM_DEFAULT               0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_CAP_GLITCHY                  0x00000001
#define LW2080_CTRL_PERF_PSTATE_FLAGS_EXT_PERF_CAP_WAIT_VBLANK              0x00000002
#define LW2080_CTRL_PERF_PSTATE_FLAGS_OVOC                                   10:10  // Enable OV/OC in Pstate 3.0
#define LW2080_CTRL_PERF_PSTATE_FLAGS_OVOC_DISABLED                         0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_OVOC_ENABLED                          0x00000001
#define LW2080_CTRL_PERF_PSTATE_FLAGS_DECREASE_THRESHOLD_IGNORE_FB           11:11  // Ignore level when callwlating FB utilization decrease threshold
#define LW2080_CTRL_PERF_PSTATE_FLAGS_DECREASE_THRESHOLD_IGNORE_FB_NO       0x00000000
#define LW2080_CTRL_PERF_PSTATE_FLAGS_DECREASE_THRESHOLD_IGNORE_FB_YES      0x00000001

/*!
 * Enumeration of the PERF_PSTATE levels.
 *
 * PP-TODO : Get rid of perf levels macro
 * @ref https://rmopengrok.lwpu.com/source/xref/chips_a/drivers/resman/kernel/inc/perf/pstate.h#101
 */
#define LW2080_CTRL_PERF_PSTATE_LEVEL_MAX                                   10
#define LW2080_CTRL_PERF_PSTATE_LEVEL_ILWALID                               LW_U8_MAX
#define LW2080_CTRL_PERF_PSTATE_INDEX_ILWALID                               LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Enumeration of the PERF_PSTATE feature version.
 *
 * _2X - Legacy implementation of PERF_PSTATE used in pstates 2.0 and earlier.
 * _30 - PERF_PSTATE implementation used with pstates 3.0 only
 * _35 - PERF_PSTATE implementation used with pstates 3.5 only
 * _3X - PERF_PSTATE implementation used with pstates 3.0 and later. Represents super of 30 and 35.
 * _40 - PERF_PSTATE implementation used with pstates 4.0 and later.
 */
#define LW2080_CTRL_PERF_PSTATE_VERSION_ILWALID                             0x00
#define LW2080_CTRL_PERF_PSTATE_VERSION_2X                                  0x20
#define LW2080_CTRL_PERF_PSTATE_VERSION_3X                                  0xFE
#define LW2080_CTRL_PERF_PSTATE_VERSION_30                                  0x30
#define LW2080_CTRL_PERF_PSTATE_VERSION_35                                  0x35
#define LW2080_CTRL_PERF_PSTATE_VERSION_40                                  0x40
#define LW2080_CTRL_PERF_PSTATE_VERSION_DISABLED                            LW_U8_MAX

/*!
 * LW2080_CTRL_PERF_PSTATE_TYPE
 *
 * Represents which PSTATE this object is referring to.
 * The _2X refers to PSTATE used in Pstate 2.0.
 * The _30 refers to PSTATE used in Pstate 3.0.
 * The _35 refers to PSTATE used in Pstate 3.5.
 */
#define LW2080_CTRL_PERF_PSTATE_TYPE_BASE                                   0x00
#define LW2080_CTRL_PERF_PSTATE_TYPE_2X                                     0x01
#define LW2080_CTRL_PERF_PSTATE_TYPE_3X                                     0x02
#define LW2080_CTRL_PERF_PSTATE_TYPE_30                                     0x03
#define LW2080_CTRL_PERF_PSTATE_TYPE_35                                     0x04
// Insert new types here and increment _MAX
#define LW2080_CTRL_PERF_PSTATE_TYPE_MAX                                    0x05
#define LW2080_CTRL_PERF_PSTATE_TYPE_UNKNOWN                                0xFF

/*!
 * Structure describing PERF_PSTATE_2X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_INFO_2X {
    /*!
     * Lwrrently we are using the legacy RMCTRLs for PSTATE 2.0.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PSTATE_INFO_2X;
typedef struct LW2080_CTRL_PERF_PSTATE_INFO_2X *PLW2080_CTRL_PERF_PSTATE_INFO_2X;

/*!
 * LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_FIXED
 *
 * This structure encapsulates clock domain information specific to _FIXED
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_FIXED {
    /*!
     * This parameter returns the clock frequency in kHz (including overclocking).
     */
    LwU32 freqkHz;
} LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_FIXED;

/*!
 * LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_DECOUPLED
 *
 * This structure represents the Pstate clock entry represented in the VBIOS
 * Pstate 5.0 Table for decoupled master and slave clocks.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_DECOUPLED {
    LwU32 targetFreqKHz;     // Target (before clkConfig) and actual (after clkConfig)
    LwU32 freqRangeMinKHz;   // Minimum frequency in KHz.
    LwU32 freqRangeMaxKHz;   // Maximum frequency in KHz.
} LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_DECOUPLED;

typedef LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_DECOUPLED LW2080_CTRL_PERF_PSTATE_30_CLOCK_ENTRY;
typedef struct LW2080_CTRL_PERF_PSTATE_CLK_DOM_INFO_DECOUPLED *PLW2080_CTRL_PERF_PSTATE_30_CLOCK_ENTRY;

/*!
 * CLK_DOMAIN type-specific data union.  Discriminated by
 * CLK_DOMAIN::super.type.
 */


/*!
 * This structure encapsulates static POR VBIOS information for a min/nom/max
 * tuple of a Pstate's clock entry.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_INFO {
    /*!
     * This freq represent the actual PSTATE VBIOS value without any frequency
     * OC offsets adjustment. When user goes to debug mode, we will reset the
     * pstate freq tuple to this value.
     */
    LwU32 origFreqkHz;
    /*!
     * This freq value represent the default POR value. This is basically the
     * ref@ origFreqkHz + Factory OC offset. We consider Factory OC offset as
     * change in POR. The PSTATE GET INFO will always return this value.
     */
    LwU32 porFreqkHz;
} LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_INFO;

/*!
 * This structure encapsulates dynamic runtime information for a min/nom/max
 * tuple of a Pstate's clock entry.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_STATUS {
    /*!
     * Current frequency value in given clock entry
     * This is the latest PSTATE Freq value with all freq oc offsets adjusted.
     * RM Mostly use this value unless someone requests for un-offseted freq
     * value. (ex. debug mode)
     *
     * On PSTATE 2.0, we do not add the freq OC offset dynamically, So no point
     * updating the base freq value. Also it is very risky to update the entire
     * PSTATE 2.0 code now, so we are not going to change the behavior on
     * PSTATE 2.0.
     * On PSTATE 2.0, freqkHz == baseFreqkHz and origFreqkHz == porFreqkHz
     */
    LwU32 freqkHz;
    /*!
     * @ref freqkHz trimmed with the CLK_DOMAIN's VF lwrve's maximum frequency -
     * i.e. the value returned via @ref ClkDomain3XProgMaxFreqMHzGet().
     */
    LwU32 freqVFMaxkHz;
} LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_STATUS;

/*!
 * This structure encapsulates runtime controllable information for a
 * min/nom/max tuple of a Pstate's clock entry.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_CONTROL {
    /*!
     * Current frequency base value in given clock entry. At system boot base
     * value is set to VBIOS value ref@ origFreqkHz. On every VF change, we
     * re-run the frequency OC adjustment on top of this base pstate freq value.
     * PSTATE (GET | SET) Control cmds directly update the base value
     */
    LwU32 baseFreqkHz;
} LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_CONTROL;

/*!
 * This structure represents all static POR VBIOS information associated with
 * one of a Pstate's clock entry's tuple.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_INFO {
    /*!
     * This parameter returns the usage type of clock domain as defined in
     * LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_<xyz> (ctrl2080clk.h)
     */
    LwU8                                         type;

    /*!
     * CLOCK INFO data for the minimum frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_INFO min;
    /*!
     * CLOCK INFO data for the maximum frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_INFO max;
    /*!
     * CLOCK INFO data for the nominal frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_INFO nom;
} LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_INFO;

/*!
 * This structure represents all dynamic runtime information associated with
 * one of a Pstate's clock entry's tuple.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_STATUS {
    /*!
     * This parameter returns the usage type of clock domain as defined in
     * LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_<xyz> (ctrl2080clk.h)
     */
    LwU8                                           type;

    /*!
     * CLOCK STATUS data for the minimum frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_STATUS min;
    /*!
     * CLOCK STATUS data for the maximum frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_STATUS max;
    /*!
     * CLOCK STATUS data for the nominal frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_STATUS nom;
} LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_STATUS;

/*!
 * This structure represents all dynamic runtime information associated with
 * one of a Pstate's clock entry's tuple.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_STATUS_35 {
    LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_STATUS super;
    /*!
     * CLOCK STATUS data for the maximum frequency supported by Vmin at this PSTATE.
     */
    LwU32                                      freqMaxAtVminkHz;
} LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_STATUS_35;

/*!
 * Define unspported Fmax@Vmin value
 */
#define PERF_PSTATE_FMAX_AT_VMIN_UNSUPPORTED LW_U32_MAX

/*!
 * This structure represents all runtime controllable information associated
 * with one of a Pstate's clock entry's tuple.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_CONTROL {
    /*!
     * This parameter returns the usage type of clock domain as defined in
     * LW2080_CTRL_CLK_CLK_DOMAIN_TYPE_<xyz> (ctrl2080clk.h)
     */
    LwU8                                            type;

    /*!
     * CLOCK CONTROL data for the minimum frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_CONTROL min;
    /*!
     * CLOCK CONTROL data for the maximum frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_CONTROL max;
    /*!
     * CLOCK CONTROL data for the nominal frequency of this PSTATE.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY_CONTROL nom;
} LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_CONTROL;

/*!
 * @brief This structure represents a clock frequency for a given clock entry
 * in the VBIOS Performance Table v5.0+.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY {
    /*!
     * Current frequency value in given clock entry
     * This is the latest PSTATE Freq value with all freq oc offsets adjusted.
     * RM Mostly use this value unless someone requests for un-offseted freq
     * value. (ex. debug mode)
     *
     * On PSTATE 2.0, we do not add the freq OC offset dynamically, So no point
     * updating the base freq value. Also it is very risky to update the entire
     * PSTATE 2.0 code now, so we are not going to change the behavior on
     * PSTATE 2.0.
     * On PSTATE 2.0, freqkHz == baseFreqkHz and origFreqkHz == porFreqkHz
     */
    LwU32 freqkHz;

    /*!
     * @ref freqkHz trimmed with the CLK_DOMAIN's VF lwrve's maximum frequency -
     * i.e. the value returned via @ref ClkDomain3XProgMaxFreqMHzGet().
     */
    LwU32 freqVFMaxkHz;

    /*!
     * Current frequency base value in given clock entry. At system boot base
     * value is set to VBIOS value ref@ origFreqkHz. On every VF change, we
     * re-run the frequency OC adjustment on top of this base Pstate freq value.
     * PSTATE (GET | SET) Control cmds directly update the base value
     */
    LwU32 baseFreqkHz;

    /*!
     * This freq represent the actual PSTATE VBIOS value without any frequency
     * OC offsets adjustment. When user goes to debug mode, we will reset the
     * Pstate freq tuple to this value.
     */
    LwU32 origFreqkHz;

    /*!
     * This freq value represent the default POR value. This is basically the
     * ref@ origFreqkHz + Factory OC offset. We consider Factory OC offset as
     * change in POR. The PSTATE GET INFO will always return this value.
     */
    LwU32 porFreqkHz;
} LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY;
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY *PLW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY;

/*!
 * @brief PSTATE Clock Entry.
 *
 * This structure represents a Pstate clock entry represented in the VBIOS
 * Performance Table v5.0+.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY {
    /*!
     * Clock tuple defining the minimum frequencies for a clock domain's
     * Pstate range.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY min;

    /*!
     * Clock tuple defining the maximum frequencies for a clock domain's
     * Pstate range.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY max;

    /*!
     * Clock tuple defining the nominal frequency in  a clock domain's
     * Pstate range.
     *
     * @note nom should be less or equal to
     * LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY::max and greater or equal to
     * LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY::min
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_FREQUENCY nom;
} LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY;
typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY *PLW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY;

typedef struct LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_35 {
    /*!
     * Super structure to be place first. Contains original clock entry information
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY super;
    /*!
     * CLOCK data for the maximum frequency supported by Vmin at this PSTATE.
     */
    LwU32                               freqMaxAtVminkHz;
} LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_35;

/*!
 * @brief PSTATE Volt Entry.
 *
 * This structure represents a Pstate volt entry arry index by volt rail index
 * as per the volt raid VBIOS table
 */
typedef struct LW2080_CTRL_PERF_PSTATE_VOLT_RAIL {
    /*!
     * Volt tuple defining the minimum voltage needed for this
     * Pstate range.
     */
    LwU32 vMinuV;
} LW2080_CTRL_PERF_PSTATE_VOLT_RAIL;

/*!
 * Structure describing PERF_VPSTATE_3X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_INFO_3X {
    /*!
     * Index into PCIE Table
     * Default value = 0xFF (Invalid)
     * wiki : https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/LPWR_VBIOS_Table#PCIE_Table
     */
    LwU8                                     pcieIdx;

    /*!
     * Index into LWLINK Table
     * Default value = 0xFF (Invalid)
     * wiki : https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/LPWR_VBIOS_Table#LWLINK_Table
     */
    LwU8                                     lwlinkIdx;

    /*!
     * Array of CLK_DOMAIN entries.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_INFO clkEntries[LW2080_CTRL_PERF_PSTATE_MAX_CLK_DOMAINS];
} LW2080_CTRL_PERF_PSTATE_INFO_3X;

/*!
 * PERF_PSTATE type-specific data union.  Discriminated by
 * PERF_PSTATE::super.type.
 */


/*!
 * Structure describing PERF_PSTATE static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
    * This parameter specifies an unique P-state identifier as defined in
    * LW2080_CTRL_PERF_PSTATES_<xyz>.
    */
    LwU32                pstateID;
    /*!
     * This parameter returns flags specific to current P-state.
     *
     * @ref LW2080_CTRL_PERF_PSTATE_FLAGS_*
     */
    LwU32                flags;
    /*!
     * Perf level of the PSTATE associated with this object. This parameter is
     * the index in the dense legacy PSTATE packing, which is the VBIOS indexing
     * but compressed so that there are no gaps.
     */
    LwU8                 level;
    /*!
     * Index to LPWR Table
     * Default value = 0xFF (Invalid)
     * wiki : https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/LPWR_VBIOS_Table
     */
    LwU8                 lpwrEntryIdx;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PSTATE_INFO_2X v2x;
        LW2080_CTRL_PERF_PSTATE_INFO_3X v3x;
    } data;
} LW2080_CTRL_PERF_PSTATE_INFO;
typedef struct LW2080_CTRL_PERF_PSTATE_INFO *PLW2080_CTRL_PERF_PSTATE_INFO;

/*!
 * Structure describing PERF_PSTATES static information/POR.  Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PSTATES_INFO_MESSAGE_ID (0xA7U)

typedef struct LW2080_CTRL_PERF_PSTATES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32  super;
    /*!
     * PERF_PSTATE version.  @ref LW2080_CTRL_PERF_PSTATE_VERSION_<xyz>
     */
    LwU8                         version;
    /*!
     * Array of PERF_VPSTATE structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PSTATE_INFO pstates[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PSTATES_INFO;
typedef struct LW2080_CTRL_PERF_PSTATES_INFO *PLW2080_CTRL_PERF_PSTATES_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PSTATES_GET_INFO
 *
 * This command returns PERF_PSTATES static object information/POR as specified
 * by the VBIOS in pstate Table.
 *
 * See @ref LW2080_CTRL_PERF_PSTATES_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PSTATES_GET_INFO (0x208020a7) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PSTATES_INFO_MESSAGE_ID" */

/*!
 * Structure representing the dynamic status of the parameters associated with
 * PERF_PSTATE_2X.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_STATUS_2X {
    /*!
     * Lwrrently we are using the legacy RMCTRLs for PSTATE 2.0.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PSTATE_STATUS_2X;

/*!
 * Structure representing the dynamic status of the parameters associated with
 * PERF_PSTATE_30.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_STATUS_30 {
    /*!
     * Array of CLK_DOMAIN entries.
     * Array is indexed by the Clocks Table (CLK_DOMAIN) indexes.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_STATUS clkEntries[LW2080_CTRL_PERF_PSTATE_MAX_CLK_DOMAINS];
} LW2080_CTRL_PERF_PSTATE_STATUS_30;

/*!
 * Structure representing the dynamic status of the parameters associated with
 * PERF_PSTATE_35.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_STATUS_35 {
    /*!
     * Array of CLK_DOMAIN entries.
     * Array is indexed by the Clocks Table (CLK_DOMAIN) indexes.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_STATUS_35 clkEntries[LW2080_CTRL_PERF_PSTATE_MAX_CLK_DOMAINS];

    /*!
     * Array of voltRails containing Vmin for each pstate
     */
    LW2080_CTRL_PERF_PSTATE_VOLT_RAIL             voltRails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_PSTATE_STATUS_35;

/*!
 * PERF_PSTATE type-specific data union.  Discriminated by
 * PERF_PSTATE::super.type.
 */


/*!
 * Structure representing the dynamic status of the parameters associated with
 * PERF_PSTATE. Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PSTATE_STATUS_2X v2x;
        LW2080_CTRL_PERF_PSTATE_STATUS_30 v30;
        LW2080_CTRL_PERF_PSTATE_STATUS_35 v35;
    } data;
} LW2080_CTRL_PERF_PSTATE_STATUS;

/*!
 * Structure representing the dynamic status of the parameters associated with
 * PERF_PSTATES. Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PSTATES_STATUS_MESSAGE_ID (0xA8U)

typedef struct LW2080_CTRL_PERF_PSTATES_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32    super;
    /*!
     * Array of PERF_VPSTATE structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PSTATE_STATUS pstates[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PSTATES_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PSTATES_GET_STATUS
 *
 * This command returns PERF_PSTATES dynamic status as specified by the
 * PERF_PSTATES entries in the PSTATE Table.
 *
 * See @ref LW2080_CTRL_PERF_PSTATES_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_PERF_PSTATES_GET_STATUS (0x208020a8) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PSTATES_STATUS_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with PERF_PSTATE_2X.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CONTROL_2X {
    /*!
     * Lwrrently we are using the legacy RMCTRLs for PSTATE 2.0.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PSTATE_CONTROL_2X;

/*!
 * Structure representing the control parameters associated with PERF_VPSTATE_3X.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CONTROL_3X {
    /*!
     * copydoc@ LW2080_CTRL_PERF_VOLT_DOM_INFO
     * Mainly use for Pstate Vmin for each voltage domain (SRAM & LOGIC).
     * Array is indexed by the Voltage Rail Table (VOLTAGE_RAIL) indexes.
     */
    LW2080_CTRL_PERF_VOLT_DOM_INFO              voltageInfo[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];

    /*!
     * Array of CLK_DOMAIN entries.
     * Array is indexed by the Clocks Table (CLK_DOMAIN) indexes.
     */
    LW2080_CTRL_PERF_PSTATE_CLOCK_ENTRY_CONTROL clkEntries[LW2080_CTRL_PERF_PSTATE_MAX_CLK_DOMAINS];
} LW2080_CTRL_PERF_PSTATE_CONTROL_3X;

/*!
 * PERF_PSTATE type-specific data union.  Discriminated by
 * PERF_PSTATE::super.type.
 */


/*!
 * Structure representing the control parameters associated with PERF_PSTATE. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PSTATE_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PSTATE_CONTROL_2X v2x;
        LW2080_CTRL_PERF_PSTATE_CONTROL_3X v3x;
    } data;
} LW2080_CTRL_PERF_PSTATE_CONTROL;

/*!
 * Structure representing the control parameters associated with PERF_PSTATES. Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_PERF_PSTATES_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32     super;
    /*!
     * Array of PERF_VPSTATE structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PSTATE_CONTROL pstates[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PSTATES_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_PSTATES_GET_CONTROL
 *
 * This command returns PERF_PSTATES control parameters as specified by the
 * PERF_PSTATES entries in the PSTATE Table.
 *
 * See @ref LW2080_CTRL_PERF_PSTATES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PSTATES_GET_CONTROL  (0x208020a9) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xA9" */

/*!
 * LW2080_CTRL_CMD_PERF_PSTATES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_PSTATES entries in the PSTATE Table and applies these new parameters to
 * the set of PERF_PSTATES entries.
 *
 * See @ref LW2080_CTRL_PERF_PSTATES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PSTATES_SET_CONTROL  (0x208020aa) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xAA" */


/*!
 * Enumeration of PERF_PERF_MODE class types.
 */
#define LW2080_CTRL_PERF_PERF_MODE_TYPE_1X        0x00U
/*! @} */

/*!
 * Enumerations of the PERF_PERF_MODE Id HALs, as specified by the VBIOS Performance Mode Table.
 *
 * https://confluence.lwpu.com/display/RMPER/Performance+Mode#PerformanceMode-VBIOS
 * @{
 */
#define LW2080_CTRL_PERF_PERF_MODE_HAL_0          0x00U
/*! @} */

/*!
 * Enumeration of performance mode ids.
 * @{
 */
#define LW2080_CTRL_PERF_PERF_MODE_ID_ILWALID     0xFFU
#define LW2080_CTRL_PERF_PERF_MODE_ID_PERF_MODE_0 0x00U
#define LW2080_CTRL_PERF_PERF_MODE_ID_MAX         0x01U
/*! @} */

/*!
 * Union for config item engage Value
 */


typedef struct LW2080_CTRL_PERF_PERF_MODE_CONFIG_VALUE {
    /*!
     * Boolean flag indicating the specific type in
     * LW2080_CTRL_PERF_PERF_MODE_CONFIG_VALUE (signedValue or
     * unsignedValue)
     */
    LwBool bSigned;
    /*!
     * Type-specific data union.
     */
    union {
        /*!
         * config item engage value when its signed integer
         */
        LwS32 signedValue;
        /*!
         * config item engage value when its unsigned integer
         */
        LwU32 unsignedValue;
    } data;
} LW2080_CTRL_PERF_PERF_MODE_CONFIG_VALUE;

/*!
 * Structure describing PERF_PERF_MODE configuration item.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODE_CONFIG_ITEM {
    /*!
     * Index of Performance Mode Config Table Entry
     */
    LwU8                                    configIdx;

    /*!
     * Number of fractional bits in @ref valDefault.
     */
    LwU8                                    numFracBits;

    /*!
     * Engage value of the configuration item.
     */
    LW2080_CTRL_PERF_PERF_MODE_CONFIG_VALUE engageVal;
} LW2080_CTRL_PERF_PERF_MODE_CONFIG_ITEM;

/*!
 * Structure describing PERF_PERF_MODE static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ                   super;

    /*!
     * Performance Mode Id.
     * @ ref LW2080_CTRL_PERF_PERF_MODE_ID_MAX
     */
    LwU8                                   modeId;

    /*!
     * Priority of the mode.
     */
    LwU8                                   priority;

    /*!
     * Mask of conflicting performance modes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32       conflictModes;

    /*!
     * Total number of supported config items
     */
    LwU8                                   numConfigItems;
    /*!
     * Array of configuration items for this mode.
     */
    LW2080_CTRL_PERF_PERF_MODE_CONFIG_ITEM configItems[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_MODE_INFO;

/*!
 * Structure describing PERF_PERF_MODES static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_MODES_INFO_MESSAGE_ID (0x35U)

typedef struct LW2080_CTRL_PERF_PERF_MODES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32     super;

    /*!
     * HAL index.
     * @ref LW2080_CTRL_PERF_PERF_MODE_HAL_<x>
     */
    LwU8                            halIdx;

    /*!
     * Map of mode id to mode index as per VBIOS POR.
     * @ref LW2080_CTRL_PERF_PERF_MODE_ID_<xyz>
     */
    LwBoardObjIdx                   modeIdToIdxMap[LW2080_CTRL_PERF_PERF_MODE_ID_MAX];

    /*!
     * Array of PERF_PERF_MODE structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_MODE_INFO modes[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_MODES_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_MODES_GET_INFO
 *
 * This command returns PERF_PERF_MODES static object information/POR as specified
 * by the VBIOS in the Performance Mode Table.
 *
 * The PERF_PERF_MODE objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_MODES_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_MODES_GET_INFO (0x20802035) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_MODES_INFO_MESSAGE_ID" */

/*!
 * Structure describing PERF_PERF_MODE dynamic information. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODE_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * Boolean tracking whether the performance mode is enabled.
     */
    LwBool               bEnabled;
} LW2080_CTRL_PERF_PERF_MODE_STATUS;

/*!
 * Structure describing PERF_PERF_MODES dynamic information. Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODES_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32       super;

    /*!
     * Mask of lwrrently enabled performance modes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32  enabledModes;

    /*!
     * Mask of lwrrently engaged performance modes.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32  engagedModes;

    /*!
     * Array of PERF_PERF_MODE structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_MODE_STATUS modes[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_MODES_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_MODES_GET_STATUS
 *
 * This command returns the PERF_PERF_MODES dynamic state information associated by the
 * performance mode functionality
 *
 * See @ref LW2080_CTRL_PERF_PERF_MODES_STATUS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_PERF_PERF_MODES_GET_STATUS (0x20802036) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x36" */

/*!
 * Structure describing PERF_PERF_MODE control params. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODE_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * Boolean tracking whether the performance mode is engaged.
     */
    LwBool               bEngaged;
} LW2080_CTRL_PERF_PERF_MODE_CONTROL;

/*!
 * Structure describing PERF_PERF_MODES control params. Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODES_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32        super;

    /*!
     * Array of PERF_PERF_MODE structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_MODE_CONTROL modes[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_MODES_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_MODES_GET_CONTROL
 *
 * This command returns PERF_PERF_MODES control parameters as specified
 * by the VBIOS in the Performance Mode Table.
 *
 * The PERF_PERF_MODE objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_MODES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_MODES_GET_CONTROL (0x20802037) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x37" */

/*!
 * LW2080_CTRL_CMD_PERF_PERF_MODES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_PERF_MODES entries in the Performance Mode Table, and applies these
 * new parameters to the set of PERF_PERF_MODES entries.
 *
 * The PERF_PERF_MODE objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_MODES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_PERF_MODES_SET_CONTROL (0x20802038) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x38" */

/*!
 * Enumeration of PERF_PERF_MODE_CONFIG class types.
 */
#define LW2080_CTRL_PERF_PERF_MODE_CONFIG_TYPE_VFE  0x00U
/*! @} */

/*!
 * Structure describing PERF_PERF_MODE_CONFIG static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODE_CONFIG_INFO_VFE {
    /*!
     * Unique identifier of globally specified VFE variable for this
     * mode of operation.
     */
    LwU8 uniqueId;
} LW2080_CTRL_PERF_PERF_MODE_CONFIG_INFO_VFE;

/*!
 * PERF_PERF_MODE_CONFIG type-specific data union. Discriminated by
 * PERF_PERF_MODE_CONFIG::super.type.
 */


/*!
 * Enumeration of POR settings to control how to restore the value on disengage.
 * @{
 */
#define LW2080_CTRL_PERF_PERF_MODE_CONFIG_DISENGAGE_RESTORE_POR  0x00
#define LW2080_CTRL_PERF_PERF_MODE_CONFIG_DISENGAGE_RESTORE_LAST 0x01
/*! @} */

/*!
 * Structure describing PERF_PERF_MODE_CONFIG static information/POR. Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODE_CONFIG_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * POR settings to control how to restore the value on disengage.
     * @ref LW2080_CTRL_PERF_PERF_MODE_CONFIG_DISENGAGE_RESTORE_<xyz>
     */
    LwU8                 restoreOnDisengage;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_MODE_CONFIG_INFO_VFE vfe;
    } data;
} LW2080_CTRL_PERF_PERF_MODE_CONFIG_INFO;

/*!
 * Structure describing PERF_PERF_MODE_CONFIGS static information/POR. Implements the
 * BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_MODE_CONFIGS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32            super;

    /*!
     * Array of PERF_PERF_MODE_CONFIG structures. Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_MODE_CONFIG_INFO configs[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_MODE_CONFIGS_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_PERF_MODE_CONFIGS_GET_INFO
 *
 * This command returns PERF_PERF_MODE_CONFIGS static object information/POR as specified
 * by the VBIOS in the Performance Mode Table.
 *
 * The PERF_PERF_MODE_CONFIG objects are indexed per how they are specified in the VBIOS
 * table.
 *
 * See @ref LW2080_CTRL_PERF_PERF_MODE_CONFIGS_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_PERF_MODE_CONFIGS_GET_INFO    (0x20802039) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x39" */

/*!
 * LW2080_CTRL_CMD_PERF_GET_PERF_TEST_SPEC_TABLE_INFO
 *
 * This command gets the performance test specification table which is read from
 * VBIOS and stored in pPerf. Will be used in MODS test 275.
 */
#define LW2080_CTRL_CMD_PERF_GET_PERF_TEST_SPEC_TABLE_INFO (0x20802034) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PERF_TEST_SPEC_TABLE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PERF_TEST_SPEC_TABLE_INFO_PARAMS_MESSAGE_ID (0x34U)

typedef struct LW2080_CTRL_PERF_GET_PERF_TEST_SPEC_TABLE_INFO_PARAMS {
    /*!
     * This parameter specifies target junction temperature in C.
     * This would be the exact value as LwTemp, which is 24 bit integer part, 8
     * bit fractional part.
     */
    LwS32  tjTempC;

    /*!
     * This parameter specifies maximum acoustic tach speed in RPM.
     */
    LwU16  maxTachSpeedRPM;

    /*!
     * This parameter specifies base power limit in mW.
     */
    LwU32  basePwrLimitmW;

    /*!
     * This parameter specifies power vector index.
     */
    LwU8   pwrVectIdx;

    /*!
     * This parameter specifies preheat time for test in seconds.
     */
    LwU8   preheatTimeS;

    /*!
     * This parameter specifies base test time in seconds.
     */
    LwU8   baseTestTimeS;

    /*!
     * This parameter specifies boost test time in seconds.
     */
    LwU8   boostTestTimeS;

    /*!
     * This parameter specifies test failure Junction temperature in C.
     */
    LwTemp fjTempC;
} LW2080_CTRL_PERF_GET_PERF_TEST_SPEC_TABLE_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_RATED_TDP_GET_INFO
 *
 * This command returns the static state describing the RATED_TDP power/perf
 * control capabilities on this GPU per the clock/pstate/vpstate POR.
 *
 * The RATED_TDP point, specified per the VPstate Table, is the maximum
 * sustainable clock and voltage point w.r.t. thermal/power for the given TDP
 * app/workload - i.e. the highest guaranteed clocks with respect to the various
 * power controllers.
 *
 * The RM's RATED_TDP functionality exposes the ability for various clients to
 * expose or limit VF points above RATED_TDP - effectively limiting
 * power-/thermal-aware boost (GPUBoost, etc.).  This functionality can be
 * useful to help eliminate performance variation for profiling or for
 * synchronized workloads.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/VPState_Table
 * https://wiki.lwpu.com/engwiki/index.php/Resman/PState/Data_Tables/VPState_Table/1.0_Spec
 *
 * See @ref LW2080_CTRL_PERF_RATED_TDP_INFO_PARAMS for documentation of
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 */
#define LW2080_CTRL_CMD_PERF_RATED_TDP_GET_INFO (0x2080206c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_RATED_TDP_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure for static information describing the RATED_TDP functionality.
 */
#define LW2080_CTRL_PERF_RATED_TDP_INFO_PARAMS_MESSAGE_ID (0x6LW)

typedef struct LW2080_CTRL_PERF_RATED_TDP_INFO_PARAMS {
    /*!
     * [out] - Boolean describing whether the RATED_TDP feature is supported on
     * this GPU.
     */
    LwU8 bSupported;
} LW2080_CTRL_PERF_RATED_TDP_INFO_PARAMS;

/*!
 * Enumeration of the RATED_TDP arbitration clients which make requests to force
 * enable/disable VF points above the RATED_TDP point.
 *
 * These clients are sorted in descending priority - the RM will arbitrate
 * between all clients in order of priority, taking as output the first client
 * whose input action != @ref LW2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT.
 */
typedef enum LW2080_CTRL_PERF_RATED_TDP_CLIENT {
    /*!
     * Internal RM client corresponding to the RM's internal state and features.
     * The RM client will either allow default behavior (@ref
     * LW2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT) or will limit to RATED_TDP
     * (@ref LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT) when no power
     * controllers are active.
     */
    LW2080_CTRL_PERF_RATED_TDP_CLIENT_RM = 0,
    /*!
     * This Client is specifically for Bug 1785342 where we need to limit the TDP
     * to Min value on boot. And clear the Max TDP limit.
     */
    LW2080_CTRL_PERF_RATED_TDP_CLIENT_WAR_BUG_1785342 = 1,
    /*!
     * Global client request.  This client is expected to be used by a global
     * switch functionality in an end-user tool, such as EVGA Precision, to
     * either force enabling boost above RATED_TDP (@ref
     * LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED) or to force limiting to
     * RATED_TDP (@ref LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT) across the
     * board, regardless of any app-profie settings.
     */
    LW2080_CTRL_PERF_RATED_TDP_CLIENT_GLOBAL = 2,
    /*!
     * Operating system request.  This client is expected to be used by the
     * operating system to set @ref LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK
     * for performance profiling.
     */
    LW2080_CTRL_PERF_RATED_TDP_CLIENT_OS = 3,
    /*!
     * App profile client requests.  This client is expected to be used by the
     * app-profile settings to either default to whatever was requested by
     * higher-priority clients (@ref LW2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT)
     * or to limit to RATED_TDP (@ref
     * LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT) for apps which have shown
     * bad behavior when boosting.
     */
    LW2080_CTRL_PERF_RATED_TDP_CLIENT_PROFILE = 4,
    /*!
     * Number of supported clients.
     *
     * @Note MUST ALWAYS BE LAST!
     */
    LW2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS = 5,
} LW2080_CTRL_PERF_RATED_TDP_CLIENT;

/*!
 * Enumeration RATED_TDP actions - these are the requested actions clients can
 * make to change the behavior of the RATED_TDP functionality.
 */
typedef enum LW2080_CTRL_PERF_RATED_TDP_ACTION {
    /*!
     * The default action - meaning no explicit request from the client other
     * than to take the default behavior (allowing boosting above RATED_TDP) or
     * any explicit actions from lower priority clients.
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT = 0,
    /*!
     * Force allow boosting above RATED_TDP - this action explicitly requests
     * boosting above RATED_TDP, preventing lower priority clients to limit to
     * RATED_TDP.
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED = 1,
    /*!
     * Force to limit above RATED_TDP - this action explicitly requests to limit
     * to RATED_TDP.  This is the opposite of the default behavior to allow
     * boosting above RATED_TDP.  Clients specify this action when they
     * explicitly need boost to be disabled (e.g. eliminating perf variation,
     * special apps which exhibit bad behavior, etc.).
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT = 2,
    /*!
     * Lock to RATED_TDP - this action requests the clocks to be fixed at the
     * RATED_TDP.  Used for achieving stable clocks required for profiling.
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK = 3,
    /*!
     * Lock to Min TDP - This requests min to be fixed at RATED_TDP but allow
     * boosting for max
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_FLOOR = 4,
} LW2080_CTRL_PERF_RATED_TDP_ACTION;

/*!
 * LW2080_CTRL_CMD_PERF_RATED_TDP_GET_STATUS
 *
 * This command returns the dynamic state of the RATED_TDP power/perf control
 * feature of this GPU.  This includes dumping all client requests for boosting
 * above or limiting to RATED_TDP, as well as the current output action for
 * boost/limit.
 *
 * This command is useful for debugging the state of the RATED_TDP feature to
 * figure out which component are limiting to RATED_TDP.
 *
 * See @ref LW2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS for documentation of
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_RATED_TDP_GET_STATUS (0x2080206d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure describing dynamic state of the RATED_TDP feature.
 */
#define LW2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_MESSAGE_ID (0x6DU)

typedef struct LW2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS {
    /*!
     * Structure of internal RM state - these values are used to determine the
     * behavior of LW2080_CTRL_PERF_RATED_TDP_CLIENT_RM per the RM's @ref
     * perfPwrRatedTdpLimitRegisterClientActive() interface.
     */
    struct {
        /*!
         * [out] - Mask of active client controllers (@ref
         * PERF_PWR_RATED_TDP_CLIENT) which are lwrrently regulating TDP.  When
         * this mask is zero, LW2080_CTRL_PERF_RATED_TDP_CLIENT_RM will request
         * LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT.
         */
        LwU32 clientActiveMask;
        /*!
         * [out] - Boolean indicating that user has requested locking to
         * RATED_TDP vPstate via corresponding regkey
         * LW_REG_STR_RM_PERF_RATED_TDP_LIMIT.  When the boolean value is true,
         * LW2080_CTRL_PERF_RATED_TDP_CLIENT_RM will request
         * LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT.
         */
        LwU8  bRegkeyLimitRatedTdp;
    } rm;

    /*!
     * [out] - Arbitrated output action of all client requests (@ref inputs).
     * This is the current state of the RATED_TDP feature.  Will only be @ref
     * LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED or @ref
     * LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT.
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION output;
    /*!
     * [out] - Array of input client request actions, indexed via @ref
     * LW2080_CTRL_PERF_RATED_TDP_CLIENT_<xyz>.  RM will arbitrate between these
     * requests, choosing the highest priority request != @ref
     * LW2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT or fallback to choosing @ref
     * LW2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED.
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION inputs[LW2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS];
} LW2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_RATED_TDP_GET_CONTROL
 *
 * This command retrieves the current requested RATED_TDP action corresponding
 * to the specified client.
 *
 * See @ref LW2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS for documentation of
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_RATED_TDP_GET_CONTROL (0x2080206e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x6E" */

/*!
 * LW2080_CTRL_CMD_PERF_RATED_TDP_SET_CONTROL
 *
 * This command sets the requested RATED_TDP action corresponding to the
 * specified client.  @Note, however, that this command is unable to set @ref
 * LW2080_CTRL_PERF_RATED_TDP_CLIENT_RM.
 *
 * See @ref LW2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS for documentation of
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_RATED_TDP_SET_CONTROL (0x2080206f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x6F" */

/*!
 * Structure containing the requested action for a RATED_TDP client (@ref
 * LW2080_CTRL_PERF_RATED_TDP_CLIENT).
 */
typedef struct LW2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS {
    /*!
     * [in] - Specified client for request.
     */
    LW2080_CTRL_PERF_RATED_TDP_CLIENT client;
    /*!
     * [in/out] - Client's requested action.
     */
    LW2080_CTRL_PERF_RATED_TDP_ACTION input;
} LW2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_PERF_POLICIES_VERSION
 *
 * Represents which PERF_POLICIES version is using.
 * _2X - Legacy implementation of PERF_LIMITS used in P-states 3.1 and earlier.
 * _35 - PERF_LIMITS implementation used with P-states 3.5 and later.
 * _PMU - Virtual class represent any PERF_LIMITS version which is implemented
 * on the PMU, not the RM.
 */
#define LW2080_CTRL_PERF_POLICIES_VERSION_2X         0x20
#define LW2080_CTRL_PERF_POLICIES_VERSION_35         0x35
#define LW2080_CTRL_PERF_POLICIES_VERSION_PMU        0xFE
#define LW2080_CTRL_PERF_POLICIES_VER_UNKNOWN        0xFF

/*!
 * Enumeration of PERF_POLICY class types.
 */
#define LW2080_CTRL_PERF_PERF_POLICY_SW_TYPE_2X      0x00
#define LW2080_CTRL_PERF_PERF_POLICY_SW_TYPE_35      0x01
#define LW2080_CTRL_PERF_PERF_POLICY_SW_TYPE_PMU     0x02
#define LW2080_CTRL_PERF_PERF_POLICY_SW_TYPE_UNKNOWN 0xFF

/*!
 * Macro indicating whether this policy is a MIN limit.
 */
#define LW2080_CTRL_PERF_POLICY_SW_FLAGS_MIN                                0:0
#define LW2080_CTRL_PERF_POLICY_SW_FLAGS_MIN_CLEAR   0x0
#define LW2080_CTRL_PERF_POLICY_SW_FLAGS_MIN_SET     0x1

/*!
 * Macro indicating whether this policy is a MAX limit.
 */
#define LW2080_CTRL_PERF_POLICY_SW_FLAGS_MAX                                1:1
#define LW2080_CTRL_PERF_POLICY_SW_FLAGS_MAX_CLEAR   0x0
#define LW2080_CTRL_PERF_POLICY_SW_FLAGS_MAX_SET     0x1

/*!
 * LW2080_CTRL_CMD_PERF_POLICIES_GET_INFO
 *
 * This info includes supported SW and HW perf policies on system, as well as
 * supported performance points.
 * Each SW policy has an individual perf request. We will pick
 * up lowest one. HW policy will enforce slowdown and therefore cap
 * performance. By observing performance policies end-user could see which
 * performance policy is limiting board's current performance.
 *
 * This command gets static performance policies information. See
 * @ref LW2080_CTRL_PERF_POLICIES_INFO_PARAMS for parameter documentation.
 */
#define LW2080_CTRL_CMD_PERF_POLICIES_GET_INFO       (0x20802080) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_POLICIES_INFO_MESSAGE_ID" */

//
// TODO-JBH: When the bugfix_main, dev_a, and chips_a changes have been
// propagated to all the other branches, define the following to have the
// the branches switch over to the new implementation. Afterwards, clean the
// branches and remove this #define.
//
#define LW2080_CTRL_PERF_POLICIES_35

/*!
 * Enumeration for SW performance policy.
 */
typedef LwU8 LW2080_CTRL_PERF_POLICY_SW_ID;

/*!
 * Power. Indicating perf is limited by total power limit.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_POWER              (0x00U)
/*!
 * Thermal. Indicating perf is limited by temperature limit.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_THERMAL            (0x01U)
/*!
 * VDT Reliability. Indicating perf is limited by reliability voltage
 * limit.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_RELIABILITY        (0x02U)
/*!
 * Operating. Indicating perf is limited by board's maximum operating
 * voltage limit.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_OPERATING          (0x03U)
/*!
 * Utilization. Indicating perf is limited by low GPC2CLK utilization.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_UTILIZATION        (0x04U)
/*!
 * GPU Boost Synchronization for SLI system.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_SLI_GPU_BOOST_SYNC (0x05U)
/*!
 * Number of available SW performance policies, not including INVALID.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_NUM                (0x06U)
/*!
 * Invalid PERF_POLICY ID.
 */
#define LW2080_CTRL_PERF_POLICY_SW_ID_ILWALID            (0xFFU)

/*!
 * Enumeration of SW performance points. Each point represents a specific upper
 * limit of performance.
 */
typedef enum LW2080_CTRL_PERF_POINT_ID {
    /*!
     * Max clock. Indicating the request is to set GPC2CLK below maximum
     * clock.
     */
    LW2080_CTRL_PERF_POINT_ID_MAX_CLOCK = 0,
    /*!
     * Turbo Boost. Indicating the request is to set GPC2CLK below turbo boost.
     */
    LW2080_CTRL_PERF_POINT_ID_TURBO_BOOST = 1,
    /*!
     * 3D Boost. Indicating the request is to set GPC2CLK below 3D boost.
     */
    LW2080_CTRL_PERF_POINT_ID_3D_BOOST = 2,
    /*!
     * Rated TDP. Indicating the request is to set GPC2CLK below rated TDP.
     */
    LW2080_CTRL_PERF_POINT_ID_RATED_TDP = 3,
    /*!
     * Max Customer Boost. Indicating the request is to set GPC2CLK below max customer boost.
     */
    LW2080_CTRL_PERF_POINT_ID_MAX_LWSTOMER_BOOST = 4,
    /*!
     * Display Clock Intersect. Indicating the request is to set GPC2CLK such that it intersects DISPCLK.
     */
    LW2080_CTRL_PERF_POINT_ID_DISPLAY_CLOCK_INTERSECT = 5,
    /*!
     * Number of available performance points. Must be last!
     */
    LW2080_CTRL_PERF_POINT_ID_NUM = 6,
} LW2080_CTRL_PERF_POINT_ID;

/*!
 * Structure containing performance policy info.
 */
typedef struct LW2080_CTRL_PERF_POLICY_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
} LW2080_CTRL_PERF_POLICY_INFO;
typedef struct LW2080_CTRL_PERF_POLICY_INFO *PLW2080_CTRL_PERF_POLICY_INFO;

/*!
 * Structure containing performance policies info.
 */
#define LW2080_CTRL_PERF_POLICIES_INFO_MESSAGE_ID (0x80U)

typedef struct LW2080_CTRL_PERF_POLICIES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32  super;

    /*!
     * PERF_POLICIES version. @ref LW2080_CTRL_PERF_POLICIES_VERSION
     */
    LwU8                         version;

    /*!
     * Output: Mask of supported SW perf points.
     */
    LwU32                        supportedSWPointMask;

    /*!
     * Output: Mask of supported SW perf policies.
     * TODO-JBH: Remove. Get policy mask from super.objMask.
     */
    LwU32                        supportedSWPolicyMask;

    /*!
     * Array of LW2080_CTRL_PERF_POLICY_INFO structures. Has valid indexes
     * corresponding to the bits in @ref super.objMask.
     */
    LW2080_CTRL_PERF_POLICY_INFO policies[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_POLICIES_INFO;
typedef struct LW2080_CTRL_PERF_POLICIES_INFO *PLW2080_CTRL_PERF_POLICIES_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_POLICIES_GET_STATUS
 *
 * This command gets performance policies status. Status includes all run-time
 * data related to performance policies. See
 * @ref LW2080_CTRL_PERF_POLICIES_STATUS for parameter documentation
 */
#define LW2080_CTRL_CMD_PERF_POLICIES_GET_STATUS (0x20802081) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_POLICIES_STATUS_MESSAGE_ID" */

/*!
 * Structure representing one SW perf policy's status.
 */
typedef struct LW2080_CTRL_PERF_POINT_VIOLATION_STATUS {
    /*!
     * Output: Mask of perf points that are lwrrently violated.
     */
    LwU32         perfPointMask;
    /*!
     * Output: Violation time for each perf point.
     */
    LwU64_ALIGN32 perfPointTimeNs[LW2080_CTRL_PERF_POINT_ID_NUM];
} LW2080_CTRL_PERF_POINT_VIOLATION_STATUS;

/*!
 * Helper macro to initialize the perf policy status.
 *
 * @param[out]  _pStatus  LW2080_CTRL_PERF_POINT_VIOLATION_STATUS structure to
 *                        initialize.
 */
#define LW2080_CTRL_PERF_POINT_VIOLATION_STATUS_INIT(_pStatus)                \
    do                                                                        \
    {                                                                         \
        LWMISC_MEMSET((_pStatus), 0x00,                                       \
            LW_SIZEOF32(LW2080_CTRL_PERF_POINT_VIOLATION_STATUS));            \
    }                                                                         \
    while (LW_FALSE)

/*!
 * Structure representing one SW perf policy's status.
 */
typedef struct LW2080_CTRL_PERF_POLICY_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ                    super;

    /*!
     * Status data.
     */
    LW2080_CTRL_PERF_POINT_VIOLATION_STATUS violationStatus;
} LW2080_CTRL_PERF_POLICY_STATUS;

/*!
 * Structure to hold perf policies status data
 */
#define LW2080_CTRL_PERF_POLICIES_STATUS_MESSAGE_ID (0x81U)

typedef struct LW2080_CTRL_PERF_POLICIES_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32             super;

    /*!
     * Version of the policies structure. Values specified in
     * @ref LW2080_CTRL_PERF_POLICIES_VERSION.
     */
    LwU8                                    version;

    /*!
     * Input: Requested SW policies mask by user. Will only retrieve status for
     * designated policies.
     */
    LwU32                                   requestedSWPolicyMask;

    /*!
     * Output: Reference time's output. Units in nano-seconds.
     */
    LwU64_ALIGN32                           referenceTimeNs;

    /*!
     * Output: System's global status and violation time for each perf point.
     */
    LW2080_CTRL_PERF_POINT_VIOLATION_STATUS globalViolationStatus;

    /*!
     * Mask of lwrrently limiting policies.
     */
    LwU32                                   limitingPoliciesMask;

    /*!
     * Array of LW2080_CTRL_PERF_POLICY_STATUS structures. Has valid indexes
     * corresponding to the bits in @ref super.objMask.
     */
    LW2080_CTRL_PERF_POLICY_STATUS          policies[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_POLICIES_STATUS;
typedef struct LW2080_CTRL_PERF_POLICIES_STATUS *PLW2080_CTRL_PERF_POLICIES_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_POLICIES_GET_SAMPLES
 *
 * This command obtains performance policies samples. Status includes all
 * run-time data related to performance policies. See
 * @ref LW2080_CTRL_PERF_POLICIES_STATUS for parameter documentation
 */
#define LW2080_CTRL_CMD_PERF_POLICIES_GET_SAMPLES          (0x20802095) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_GET_SAMPLES_PARAMS_MESSAGE_ID" */

/*!
 * @brief Maximum size of the cirlwlar buffer of samples to keep a history.
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_SAMPLE_COUNT (50U)

/*!
 * @brief An individual data sample of perf policies.
 */
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_SAMPLE {
    /*!
     * @brief Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * @brief The perf policies' status.
     */
    LW2080_CTRL_PERF_POLICIES_STATUS data;
} LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_SAMPLE;
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_SAMPLE *PLW2080_CTRL_PERF_PMUMON_PERF_POLICIES_SAMPLE;

/*!
 * @brief Container of @ref LW2080_CTRL_PERF_POLICIES_STATUS stored in a
 * cirlwlar buffer. Allows clients to obtain perf. policy status from the
 * super surface instead of making an RPC call.
 */
#define LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_GET_SAMPLES_PARAMS_MESSAGE_ID (0x95U)

typedef struct LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_GET_SAMPLES_PARAMS {
    /*!
     * @brief Metadata for the samples.
     */
    LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;

    /*!
     * @brief A collection of data samples for perf policy data.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_SAMPLE samples[LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_SAMPLE_COUNT], 8);
} LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_GET_SAMPLES_PARAMS;
typedef struct LW2080_CTRL_PERF_PMUMON_PERF_POLICIES_GET_SAMPLES_PARAMS *PLW2080_CTRL_PERF_PMUMON_PERF_POLICIES_GET_SAMPLES_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_VF_POINT_OPS
 *
 * Allows client to set a VF point and cap to it, get the current capped
 * VF point and clear a previously capped VF point.
 * Limitation(s):
 *   1. Supported on cheetah alone.
 *
 */
#define LW2080_CTRL_CMD_PERF_VF_POINT_OPS (0x20802082) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_VF_POINT_OPS_PARAMS_MESSAGE_ID" */

typedef struct LW2080_CTRL_PERF_VF_POINT_OPS_DOMAINS {
    /*!
     * In: Indicates the requested operation, must be one of the defines
     * in LW2080_CTRL_PERF_VF_POINT_OPS_OP_XXX.
     */
    LwU8  operation;
    /*!
     * In: Indicates the reason associated with the request, must be one
     * of the defines in LW2080_CTRL_PERF_VF_POINT_OPS_REASON_XXX. This is
     * only required for operation SET and CLEAR. This is unused for OP_GET.
     */
    LwU8  reason;
    /*!
     * In/Out: Voltage in uV.
     *   For a OP_SET call, this provides the new value to be capped to (IN).
     *   Either frequency or voltage must be set. Both cannot be non-zero.
     *   For a OP_GET call, this provides the current value (OUT).
     *   For a OP_CLEAR call, this field is unused.
     */
    LwU32 voltuV;
    /*!
     * In: frequency requested in kHz
     *   For a OP_SET call, this provides the new value to be capped to (IN).
     *   Either frequency or voltage must be set. Both cannot be non-zero.
     *   For a OP_GET call, this provides the current value (OUT).
     *   For a OP_CEAR call, this field is unused.
     */
    LwU32 freqKHz;
    /*!
     * In/Out: Indicates the clk domain for this entry. Should be one of
     * LW2080_CTRL_CLK_DOMAIN_XXX defined in ctrl2080clk.h.
     * Valid ones for CheetAh: LW2080_CTRL_CLK_DOMAIN_GPC2CLK.
     */
    LwU32 clkDomain;
} LW2080_CTRL_PERF_VF_POINT_OPS_DOMAINS;

/*!
 * Maximum number of domains that this control call accepts.
 * We only need 1 today for gpc2clk, keeping room for more
 * domains in case we need to use them.
 */
#define LW2080_CTRL_PERF_VF_POINT_OPS_DOMAINS_MAX 0x3

#define LW2080_CTRL_PERF_VF_POINT_OPS_PARAMS_MESSAGE_ID (0x82U)

typedef struct LW2080_CTRL_PERF_VF_POINT_OPS_PARAMS {
    /*!
     * In: Indicates number of valid entries in the array vfDomains.
     */
    LwU32                                 numOfEntries;
    /*!
     * In/Out: Array of LW2080_CTRL_PERF_VF_POINT_OPS_DOMAINS.
     */
    LW2080_CTRL_PERF_VF_POINT_OPS_DOMAINS vfDomains[LW2080_CTRL_PERF_VF_POINT_OPS_DOMAINS_MAX];
} LW2080_CTRL_PERF_VF_POINT_OPS_PARAMS;

/*!
 * Valid operations defines
 */
#define LW2080_CTRL_PERF_VF_POINT_OPS_OP_GET                       0x0
#define LW2080_CTRL_PERF_VF_POINT_OPS_OP_SET                       0x1
#define LW2080_CTRL_PERF_VF_POINT_OPS_OP_CLEAR                     0x2

/*!
 * Valid reasons defines
 */
#define LW2080_CTRL_PERF_VF_POINT_OPS_REASON_SW_THERMAL_THROTTLING LW2080_CTRL_PERF_LIMIT_THERM_POLICY_DOM_GRP_1
#define LW2080_CTRL_PERF_VF_POINT_OPS_REASON_EDP                   LW2080_CTRL_PERF_LIMIT_EDP_POLICY_DOM_GRP_1
#define LW2080_CTRL_PERF_VF_POINT_OPS_REASON_FORCED                LW2080_CTRL_PERF_LIMIT_FORCED
#define LW2080_CTRL_PERF_VF_POINT_OPS_REASON_MODS_RULES            LW2080_CTRL_PERF_LIMIT_MODS_RULES
#define LW2080_CTRL_PERF_VF_POINT_OPS_REASON_OS_LEVEL              LW2080_CTRL_PERF_LIMIT_OS_LEVEL

/*!
 * LW2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES
 *
 * This command returns perfmon gpu monitoring utilization samples.
 * This command is not supported with SMC enabled.
 *
 * See LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_PARAM for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * Deprecated, please use LW2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2.
 */
#define LW2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES       (0x20802083) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_PARAM_MESSAGE_ID" */

/*!
 * This struct represents the GPU monitoring perfmon sample for an engine.
 */
typedef struct LW2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE {
    /*!
     * Percentage during the sample that the engine remains busy. This
     * is in units of pct*100.
     */
    LwU32 util;
    /*!
     * Scaling factor to colwert utilization from full GPU to per vGPU.
     */
    LwU32 vgpuScale;
    /*!
     * Process ID of the process that was active on the engine when the
     * sample was taken. If no process is active then LW2080_GPUMON_PID_ILWALID
     * will be returned.
     */
    LwU32 procId;
    /*!
     * Process ID of the process in the vGPU VM that was active on the engine when
     * the sample was taken. If no process is active then LW2080_GPUMON_PID_ILWALID
     * will be returned.
     */
    LwU32 subProcessID;
     /*!
     * Process name of the process in the vGPU VM that was active on the engine when
     * the sample was taken. If no process is active then NULL will be returned.
     */
    char  subProcessName[LW_SUBPROC_NAME_MAX_LENGTH];
} LW2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE;

/*!
 * This struct represents the GPU monitoring perfmon sample.
 */
typedef struct LW2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE {
    /*!
     * Base GPU monitoring sample.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_GPUMON_SAMPLE base, 8);
    /*!
     * FB bandwidth utilization sample.
     */
    LW2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE fb;
    /*!
     * GR utilization sample.
     */
    LW2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE gr;
    /*!
     * LW ENCODER utilization sample.
     */
    LW2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE lwenc;
    /*!
     * LW DECODER utilization sample.
     */
    LW2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE lwdec;
} LW2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE;

/*!
 * This struct represents the GPU monitoring samples of perfmon values that
 * client wants the access to.
 */
#define LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_PARAM_MESSAGE_ID (0x83U)

typedef LW2080_CTRL_GPUMON_SAMPLES LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_PARAM;

/*!
 * Number of GPU monitoring sample in their respective buffers.
 */
#define LW2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL       100

#define LW2080_CTRL_PERF_GPUMON_PERFMON_UTIL_BUFFER_SIZE           \
    LW_SIZEOF32(LW2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE) *     \
    LW2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL

/*!
 * LW2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2
 *
 * This command returns perfmon gpu monitoring utilization samples.
 * This command is not supported with SMC enabled.
 *
 * See LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_PARAM_V2 for documentation
 * on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * Note this is the same as LW2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES
 * but without the embedded pointer.
 *
 */
#define LW2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2 (0x20802096) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_MESSAGE_ID" */

/*!
 *  This structure represents the GPU monitoring samples of utilization values that
 *  the client wants access to.
 */
#define LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_MESSAGE_ID (0x96U)

typedef struct LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS {
    /*!
    * Type of the sample, see LW2080_CTRL_GPUMON_SAMPLE_TYPE_* for reference.
    */
    LwU8  type;
    /*!
    * Size of the buffer, this should be
    * bufSize ==  LW2080_CTRL_*_GPUMON_SAMPLE_COUNT_*
    *    sizeof(derived type of LW2080_CTRL_GPUMON_SAMPLE).
    */
    LwU32 bufSize;
    /*!
    * Number of samples in ring buffer.
    */
    LwU32 count;
    /*!
    * tracks the offset of the tail in the cirlwlar queue array pSamples.
    */
    LwU32 tracker;
    /*!
    * A cirlwlar queue with size == bufSize.
    *
    * @note This cirlwlar queue wraps around after 10 seconds of sampling,
    * and it is clients' responsibility to query within this time frame in
    * order to avoid losing samples.
    * @note With one exception, this queue contains last 10 seconds of samples
    * with tracker poiniting to oldest entry and entry before tracker as the
    * newest entry. Exception is when queue is not full (i.e. tracker is
    * pointing to a zeroed out entry), in that case valid entries are between 0
    * and tracker.
    * @note Clients can store tracker from previous query in order to provide
    * samples since last read.
    */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE samples[LW2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL], 8);
} LW2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_GET_LOCKED_CLOCKS_MODE_STATUS
 *
 * This command is used to query the LockedClocksMode setting of the system.
 * LockedClocksMode is a setting to disallow boost and set max clock value.
 * Tesla GPU's traditionally had boost clocks disabled, but GK210_and_later
 * Tesla chips will allow boost clocks and provide user an option to disallow
 * boost clocks if required.
 *
 * bToggleBoostClocksSupported
 *      This is read from VBIOS and controls users from changing the
 *      LockedClocksMode setting. This field is non-editable.
 *
 * bLockedClocksModeEnabledByDefault
 *      Initially holds the value from vbios. This will be system
 *      LockedClocksMode when no other clients want to override LockedClocksMode.
 *      This flag can be edited only by privileged user.
 *
 * bLockedClocksModeLwrrentlyEnabled
 *      This holds the active system LockedClocksMode.
 *
 * bClientToggleLockedClocksMode
 *      This flag controls clients from overriding the LockedClocksMode. This
 *      flag can be edited only by privileged user.
 *
 * See @ref LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_MODE_STATUS_PARAM for
 * documentation of parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_LOCKED_CLOCKS_MODE_STATUS (0x20802084) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_MODE_STATUS_PARAM_MESSAGE_ID" */

/*!
 * This struct contains ToggleSupportedFlag and LockedClocksModeEnabled flag.
 */
#define LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_MODE_STATUS_PARAM_MESSAGE_ID (0x84U)

typedef struct LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_MODE_STATUS_PARAM {
    LwBool bToggleBoostClocksSupported;
    LwBool bLockedClocksModeEnabledByDefault;
    LwBool bLockedClocksModeLwrrentlyEnabled;
    LwBool bClientToggleLockedClocksMode;
} LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_MODE_STATUS_PARAM;

/*!
 * LW2080_CTRL_CMD_PERF_SET_LOCKED_CLOCKS_MODE_STATUS
 *
 * This command is used to change the active LockedClocksMode. It can
 * be either enabled or disabled. Clients can use this call to change the state
 * when no other clients have requested for locked clock mode setting or add
 * refcnt to existing state. When this call succeeds, the system will be in the
 * requested LockedClocksMode state and the client is assured that the system
 * stays in this mode till the requested client releases the state or is killed.
 *
 * bLockedClocksModeEnabled
 *      This will be current system LockedClocksMode.
 *
 * See @ref LW2080_CTRL_PERF_SET_LOCKED_CLOCKS_MODE_STATUS_PARAM for
 * documentation of parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_OPERATION
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_PERF_SET_LOCKED_CLOCKS_MODE_STATUS (0x20802085) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_LOCKED_CLOCKS_MODE_STATUS_PARAM_MESSAGE_ID" */

/*!
 * This struct contains LockedClocksModeEnabled flag.
 */
#define LW2080_CTRL_PERF_SET_LOCKED_CLOCKS_MODE_STATUS_PARAM_MESSAGE_ID (0x85U)

typedef struct LW2080_CTRL_PERF_SET_LOCKED_CLOCKS_MODE_STATUS_PARAM {
    LwBool bLockedClocksModeEnabled;
} LW2080_CTRL_PERF_SET_LOCKED_CLOCKS_MODE_STATUS_PARAM;

/*!
 * LW2080_CTRL_CMD_PERF_SET_DEFAULT_LOCKED_CLOCKS_MODE_STATUS
 *
 * This command is used to set the default LockedClocksMode and control
 * clients from changing locked clock mode setting. This call is reserved only
 * for privileged users.
 *
 * bLockedClocksModeEnabled
 *      This will be system LockedClocksMode setting when no other clients want
 *      to override lockedclock mode setting.
 *
 * bClientToggleLockedClocksMode
 *      Allow/Disallow clients to change the LockedClocksMode flag.
 *
 * bForceLockedClocksMode
 *      Force the mode to apply immediately.
 *
 * See @ref LW2080_CTRL_PERF_SET_DEFAULT_LOCKED_CLOCKS_MODE_STATUS_PARAM for
 * documentation of parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_SET_DEFAULT_LOCKED_CLOCKS_MODE_STATUS (0x20802086) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_DEFAULT_LOCKED_CLOCKS_MODE_STATUS_PARAM_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_DEFAULT_LOCKED_CLOCKS_MODE_STATUS_PARAM_MESSAGE_ID (0x86U)

typedef struct LW2080_CTRL_PERF_SET_DEFAULT_LOCKED_CLOCKS_MODE_STATUS_PARAM {
    LwBool bLockedClocksModeEnabled;
    LwBool bClientToggleLockedClocksMode;
    LwBool bForceLockedClocksMode;
} LW2080_CTRL_PERF_SET_DEFAULT_LOCKED_CLOCKS_MODE_STATUS_PARAM;

/*
 * LW2080_CTRL_CMD_PERF_GET_VID_ENG_PERFMON_SAMPLE
 *
 * This command can be used to obtain video decoder utilization of
 * the associated subdevice.
 * This command is not supported with SMC enabled.
 *
 *   engineType
 *     This parameter will allow clients to set type of video
 *     engine in question. It can be LWENC or LWDEC.
 *   clkPercentBusy
 *     This parameter contains the percentage during the sample that
 *     the clock remains busy.
 *   samplingPeriodUs
 *     This field returns the sampling period in microseconds.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_VID_ENG_PERFMON_SAMPLE (0x20802087) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS_MESSAGE_ID" */

typedef enum LW2080_CTRL_CMD_PERF_VID_ENG {
    /*!
     * GPU Video encoder engine.
     */
    LW2080_CTRL_CMD_PERF_VID_ENG_LWENC = 1,

    /*!
     * GPU video decoder engine.
     */
    LW2080_CTRL_CMD_PERF_VID_ENG_LWDEC = 2,
} LW2080_CTRL_CMD_PERF_VID_ENG;

#define LW2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS_MESSAGE_ID (0x87U)

typedef struct LW2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS {
    LW2080_CTRL_CMD_PERF_VID_ENG engineType;
    LwU32                        clkPercentBusy;
    LwU32                        samplingPeriodUs;
} LW2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GPU_IS_IDLE
 *
 * This command notifies RM to make p state switching aggressive by setting
 * required limiting factors to speed up GC6 Entry initiation.
 *
 *  prevPstate [out]
 *      This parameter will contain the pstate before the switch was initiated
 *
 * Possible status return values are:
 *   LW_OK : If P State Switch is successful
 *   LW_ILWALID_STATE : If unable to access P State structure
 *   LWOS_STATUS_ERROR   : If P State Switch is unsuccessful
 */
#define LW2080_CTRL_CMD_PERF_GPU_IS_IDLE (0x20802089) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GPU_IS_IDLE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GPU_IS_IDLE_PARAMS_MESSAGE_ID (0x89U)

typedef struct LW2080_CTRL_PERF_GPU_IS_IDLE_PARAMS {
    LwU32 prevPstate;
    LwU32 action;
} LW2080_CTRL_PERF_GPU_IS_IDLE_PARAMS;

#define LW2080_CTRL_PERF_GPU_IS_IDLE_TRUE             (0x00000001)
#define LW2080_CTRL_PERF_GPU_IS_IDLE_FALSE            (0x00000002)

/*
 * LW2080_CTRL_CMD_PERF_AGGRESSIVE_PSTATE_NOTIFY
 *
 * This command is for the KMD Aggressive P-state feature. Please reference:
 * https://confluence.lwpu.com/display/OP/RID+64943+-+Aggressive+P+state+%3A+P-state+boosting.
 *
 *  bGpuIsIdle [in]
 *      When true, applies cap to lowest P-state/GPCCLK. When false, releases cap.
 *  idleTimeUs [in]
 *      The amount of time (in microseconds) the GPU was idle since previous
 *      call, part of the GPU utilization data from KMD.
 *  busyTimeUs [in]
 *      The amount of time (in microseconds) the GPU was not idle since
 *      previous call, part of the GPU utilization data from KMD.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_AGGRESSIVE_PSTATE_NOTIFY (0x2080208f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_AGGRESSIVE_PSTATE_NOTIFY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_AGGRESSIVE_PSTATE_NOTIFY_PARAMS_MESSAGE_ID (0x8FU)

typedef struct LW2080_CTRL_PERF_AGGRESSIVE_PSTATE_NOTIFY_PARAMS {
    LwBool bGpuIsIdle;
    LwBool bRestoreToMax;
    LW_DECLARE_ALIGNED(LwU64 idleTimeUs, 8);
    LW_DECLARE_ALIGNED(LwU64 busyTimeUs, 8);
} LW2080_CTRL_PERF_AGGRESSIVE_PSTATE_NOTIFY_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_GET_LOCKED_CLOCKS_CLIENT_PID
 *
 * This command is used to get the list of all process id who have toggled
 * LockedClocksMode. This call is reserved only for privileged users.
 *
 * count
 *      Min of (Number of processes lwrrently using toggle LockedClocksMode,
 *           LW2080_CTRL_PERF_MAX_LOCKED_CLOCKS_CLIENT_PID).
 *
 * pid
 *      Array of process id's.
 *
 * bPidExceedsMax
 *      Set if the number of processes lwrrently using toggle LockedClocksMode is
 *      greater than LW2080_CTRL_PERF_MAX_LOCKED_CLOCKS_CLIENT_PID.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_PERMISSIONS
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_LOCKED_CLOCKS_CLIENT_PID (0x2080208a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_CLIENT_PID_PARAM_MESSAGE_ID" */

#define LW2080_CTRL_PERF_MAX_LOCKED_CLOCKS_CLIENT_PID     (256)

#define LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_CLIENT_PID_PARAM_MESSAGE_ID (0x8AU)

typedef struct LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_CLIENT_PID_PARAM {
    LwU32  count;
    LwU32  pid[LW2080_CTRL_PERF_MAX_LOCKED_CLOCKS_CLIENT_PID];
    LwBool bPidExceedsMax;
} LW2080_CTRL_PERF_GET_LOCKED_CLOCKS_CLIENT_PID_PARAM;

/*!
 * LW2080_CTRL_PERF_LOCK_PEX_GEN2
 *
 * This command will hold PEX Gen to Gen 2, if Gen 2 and Gen 3 are supported.
 * This structure is solely used for bug 1606797 on Kepler. Do not use it
 * for other purpose.
 */
#define LW2080_CTRL_CMD_PERF_LOCK_PEX_GEN2 (0x2080208b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_LOCK_PEX_GEN2_PARAM_MESSAGE_ID" */

#define LW2080_CTRL_PERF_LOCK_PEX_GEN2_PARAM_MESSAGE_ID (0x8BU)

typedef struct LW2080_CTRL_PERF_LOCK_PEX_GEN2_PARAM {
    LwBool bEnabled;
} LW2080_CTRL_PERF_LOCK_PEX_GEN2_PARAM;

/* ------------------------ Debug Mode APIs -------------------------------- */

/*!
 * Structure describing Debug mode static information/POR.
 */
#define LW2080_CTRL_PERF_DEBUG_MODE_GET_INFO_MESSAGE_ID (0xB9U)

typedef struct LW2080_CTRL_PERF_DEBUG_MODE_GET_INFO {
    /*!
     * The new APIs are only supported on pascal and later GPUs.
     * If this flag is NOT set, client must discard all other info
     * and use the legacy APIs.
     */
    LwU8   bApiSupported;

    /*!
     * TRUE     If we have Debug mode support enable on given chip
     * FALSE    Otherwise.
     */
    LwBool bSupported;

    /*!
     * TRUE     If the given board is factory over clocked board
     * FALSE    If the given board is reference boards.
     */
    LwBool bFactoryOverclocked;
} LW2080_CTRL_PERF_DEBUG_MODE_GET_INFO;
typedef struct LW2080_CTRL_PERF_DEBUG_MODE_GET_INFO *PLW2080_CTRL_PERF_DEBUG_MODE_GET_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_DEBUG_MODE_GET_INFO
 *
 * This command returns static object information/POR as specified
 * by the VBIOS for debug mode feature.
 *
 * See @ref LW2080_CTRL_PERF_DEBUG_MODE_GET_INFO for documentation on the
 * parameters.
 */
#define LW2080_CTRL_CMD_PERF_DEBUG_MODE_GET_INFO (0x208020b9) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_DEBUG_MODE_GET_INFO_MESSAGE_ID" */

/*!
 * Structure describing dynamic status of Debug mode params.
 */
#define LW2080_CTRL_PERF_DEBUG_MODE_GET_STATUS_MESSAGE_ID (0xBAU)

typedef struct LW2080_CTRL_PERF_DEBUG_MODE_GET_STATUS {
    /*!
     * TRUE     If there is any amount of over clocking on the GPU.
     * FALSE    Otherwise.
     *
     * This flag represents sum of factory and user applied OC.
     */
    LwBool bOverClocked;
} LW2080_CTRL_PERF_DEBUG_MODE_GET_STATUS;
typedef struct LW2080_CTRL_PERF_DEBUG_MODE_GET_STATUS *PLW2080_CTRL_PERF_DEBUG_MODE_GET_STATUS;

/*!
 * LW2080_CTRL_CMD_PERF_DEBUG_MODE_GET_STATUS
 *
 * This command returns the dynamic state information associated with the
 * debug mode feature.
 *
 * See @ref LW2080_CTRL_PERF_DEBUG_MODE_GET_STATUS for documentation on the
 * parameters.
 */
#define LW2080_CTRL_CMD_PERF_DEBUG_MODE_GET_STATUS (0x208020ba) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_DEBUG_MODE_GET_STATUS_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with Debug mode.
 */
typedef struct LW2080_CTRL_PERF_DEBUG_MODE_CONTROL {
    /*!
     * Boolean flag that will be use by client to enable / disable the
     * over clocking.
     */
    LwBool bEnabled;
} LW2080_CTRL_PERF_DEBUG_MODE_CONTROL;
typedef struct LW2080_CTRL_PERF_DEBUG_MODE_CONTROL *PLW2080_CTRL_PERF_DEBUG_MODE_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_DEBUG_MODE_GET_CONTROL
 *
 * This command returns debug mode feature specific control parameters.
 *
 * See @ref LW2080_CTRL_PERF_DEBUG_MODE_CONTROL for documentation on the
 * parameters.
 */
#define LW2080_CTRL_CMD_PERF_DEBUG_MODE_GET_CONTROL (0x208020bb) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xBB" */

/*!
 * LW2080_CTRL_CMD_PERF_DEBUG_MODE_SET_CONTROL
 *
 * This command accepts client-specified control parameters for debug mode
 * feature, and applies these new parameters values.
 *
 *
 * See LW2080_CTRL_PERF_DEBUG_MODE_CONTROL for documentation on the
 * parameters.
 */
#define LW2080_CTRL_CMD_PERF_DEBUG_MODE_SET_CONTROL (0x208020bc) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xBC" */

/*
 * LW2080_CTRL_CMD_PERF_CHECK_DEFAULT_MODE
 *
 * Facilitates a way to enable end-users to check whether their clocks are in
 * default-mode or debug-mode with reference to board POR clocks.
 */
#define LW2080_CTRL_CMD_PERF_CHECK_DEFAULT_MODE     (0x2080208c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CHECK_DEFAULT_MODE_PARAMS_MESSAGE_ID" */

/*!
 * LW2080_CTRL_PERF_CHECK_DEFAULT_MODE_PARAMS
 */
#define LW2080_CTRL_PERF_CHECK_DEFAULT_MODE_PARAMS_MESSAGE_ID (0x8LW)

typedef struct LW2080_CTRL_PERF_CHECK_DEFAULT_MODE_PARAMS {
    /*!
     * TRUE     If the GPU is part of supported GPUs selected by marketing.
     * FALSE    Otherwise.
     */
    LwBool bSupported;

    /*!
     * if bMode =
     * 1 - Base Clk is equal to LW POR
     * 0 - Base Clk is not equal to LW POR
     */
    LwBool bMode;

    /*!
     * PP-TODO : Remove this param once LWAPIs are updated to use the new param
     * @ref bFactoryOverclocked
     */
    LwBool bIsBaseClkChangeAllow;

    /*!
     * TRUE     If the given board is factory over clocked board
     * FALSE    If the given board is reference boards.
     *
     * @note    Static param and does not change at run-time
     */
    LwBool bFactoryOverclocked;
} LW2080_CTRL_PERF_CHECK_DEFAULT_MODE_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_SET_DEFAULT_MODE
 *
 * Facilitates a way to enable end-users to reduce their clocks to
 * reference board POR clocks (default-mode or debug-mode)
 */
#define LW2080_CTRL_CMD_PERF_SET_DEFAULT_MODE (0x2080208d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_DEFAULT_MODE_PARAMS_MESSAGE_ID" */

/*!
 * LW2080_CTRL_PERF_SET_DEFAULT_MODE_PARAMS
 */
#define LW2080_CTRL_PERF_SET_DEFAULT_MODE_PARAMS_MESSAGE_ID (0x8DU)

typedef struct LW2080_CTRL_PERF_SET_DEFAULT_MODE_PARAMS {
    /*!
     * PP-TODO : Remove this param once LWAPIs are updated to use the new param
     * @ref bEnabled
     */
    LwBool bMode;

    /*!
     * Boolean flag that will be use by client to enable / disable the
     * over clocking.
     *
     * LW_TRUE  - Enable debug mode
     * LW_FALSE - Disable debug mode
     */
    LwBool bEnabled;

    /*!
     * Difference between the base clk value of LW POR and Factory OC Board.
     * This difference is use to shift the VF lwrve with new freqDelta offset.
     */
    LwS32  freqDeltaKhz;
} LW2080_CTRL_PERF_SET_DEFAULT_MODE_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_GET_DEFAULT_MODE
 *
 * Facilitates a way to to check whether the debug mode is enabled / disabled.
 */
#define LW2080_CTRL_CMD_PERF_GET_DEFAULT_MODE (0x2080208e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_DEFAULT_MODE_PARAMS_MESSAGE_ID" */

/*!
 * LW2080_CTRL_PERF_GET_DEFAULT_MODE_PARAMS
 */
#define LW2080_CTRL_PERF_GET_DEFAULT_MODE_PARAMS_MESSAGE_ID (0x8EU)

typedef struct LW2080_CTRL_PERF_GET_DEFAULT_MODE_PARAMS {
    /*!
     * Boolean flag that will be use by client to check whether the
     * debug mode feature is enable / disable.
     *
     * LW_TRUE  - Enable debug mode
     * LW_FALSE - Disable debug mode
     */
    LwBool bEnabled;
} LW2080_CTRL_PERF_GET_DEFAULT_MODE_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_VF_CHANGE_INJECT
 *
 * This command injects a VF change for a list of clock domain(s) and a
 * list of voltage rail(s) either into the PERF change sequencer, or directly
 * to the PMU if the BYPASS_CHANGE_SEQ flag is set.
 *
 * Note: The BYPASS_CHANGE_SEQ should be set only if a client wants to change
 * the NAFLL clocks on a platform (like emulation) that doesn't support voltage
 * changes. Please don't set this flag if you don't know what its supposed to do.
 *
 * The clients can specify the clock domains and voltage rails in any order
 * respectively. The change sequence engine will however change the clocks
 * and voltages per specific ordering rules that the clients need not be aware
 * of. The important point is that the clients should *not* assume that RM will
 * change the clocks or voltages in the order they gave in the list.
 *
 * See @ref LW2080_CTRL_PERF_VF_CHANGE_INJECT_PARAMS for documentation on the
 * parameters.
 *
 * Return values are specified per @ref lw2080CtrlCmdPerfVfChangeInject.
 */
#define LW2080_CTRL_CMD_PERF_VF_CHANGE_INJECT                         (0x208020a3) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xA3" */

/*
 *@ref flags field of LW2080_CTRL_PERF_VF_CHANGE_INJECT_PARAMS structure
 */
#define LW2080_CTRL_PERF_VF_CHANGE_INJECT_FLAGS_ADC_OVERRIDE                 0:0
#define LW2080_CTRL_PERF_VF_CHANGE_INJECT_FLAGS_ADC_OVERRIDE_NO       0x00000000
#define LW2080_CTRL_PERF_VF_CHANGE_INJECT_FLAGS_ADC_OVERRIDE_YES      0x00000001

#define LW2080_CTRL_PERF_VF_CHANGE_INJECT_FLAGS_BYPASS_CHANGE_SEQ            1:1
#define LW2080_CTRL_PERF_VF_CHANGE_INJECT_FLAGS_BYPASS_CHANGE_SEQ_NO  0x00000000
#define LW2080_CTRL_PERF_VF_CHANGE_INJECT_FLAGS_BYPASS_CHANGE_SEQ_YES 0x00000001

/*!
 * @ref LW2080_CTRL_PERF_VF_CHANGE_INJECT_PARAMS
 */
typedef struct LW2080_CTRL_PERF_VF_CHANGE_INJECT_PARAMS {
    /*!
     * [In] Additional flags to indicate override, FFR set/clear etc
     * @ref LW2080_CTRL_PERF_VF_CHANGE_INJECT_FLAGS_<XYZ>
     */
    LwU8                            flags;

    /*!
     * [In/Out] List of clock domains for which a change of frequency is
     * required
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_LIST clkList;

    /*!
     * [In] List of voltage rail for which a change of voltage is required
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_LIST voltList;
} LW2080_CTRL_PERF_VF_CHANGE_INJECT_PARAMS;

/*!
 * Special define to represent an invalid PERF_VPSTATE index.
 */
#define LW2080_CTRL_PERF_VPSTATE_INDEX_ILWALID                    LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Special define to represent resman's number of domain groups.
 */
#define LW2080_CTRL_PERF_VPSTATE_2X_DOMAIN_GROUP_NUM              0x03

/*!
 * Enumeration of the PERF_VPSTATE feature version.
 *
 * _2X - Legacy implementation of PERF_VPSTATE used in pstates 2.0 and earlier.
 * _3X - PERF_VPSTATE implementation used with pstates 3.0 and later.
 */
#define LW2080_CTRL_PERF_VPSTATE_VERSION_2X                       0x20
#define LW2080_CTRL_PERF_VPSTATE_VERSION_3X                       0x30

/*!
 * LW2080_CTRL_PERF_VPSTATE_TYPE
 *
 * Represents which VPSTATE this object is referring to.
 * The _2X refers to VPSTATE used in Pstate 2.0.
 * The _3X refers to VPSTATE used in Pstate 3.0.
 */
#define LW2080_CTRL_PERF_VPSTATE_TYPE_2X                          0x02
#define LW2080_CTRL_PERF_VPSTATE_TYPE_3X                          0x03
#define LW2080_CTRL_PERF_VPSTATE_TYPE_UNKNOWN                     0xFF

/*!
 *
 * MACROs of VPSTATE indexes/names.
 * These indexes are used to abstract away implementation details from
 * client interfaces - e.g. a client can request the "BOOST" vpstate without
 * having to know which vpstate index.
 *
 * _D2          // vP-state index for 0xD2 state.
 * _D3          // vP-state index for 0xD3 state.
 * _D4          // vP-state index for 0xD4 state.
 * _D5          // vP-state index for 0xD5 state.
 * _OVERLWR     // Fastest vP-state when brick/aux over-current event oclwrs.
 * _VRHOT       // Fastest vP-state when GPU voltage regulator over-temperature event oclwrs.
 * _MAXBATT     // Fastest vP-state index available for battery.
 * _MAXSLI      // Fastest vP-state index available for SLI mode.
 * _MAXTHERMSUS // Fastest thermally sustainable vP-state index.
 * _SLOWDOWNPWR // vP-state index for the Slowdown Power Threshold
 * _MIDPOINT    // vP-state index for the Mid-Point - Used only by MODS for man. diag testing
 * _FLAGS       // vP-state header flags byte - though we don't use this variable
 * _INFLECTION0 // vP-state index for power policy inflection point 0
 * _FULLDEFLECT // vP-state index for full deflection point
 * _IMPFIRST    // First IMP VPstate.
 * _IMPLAST     // Last IMP VPstate.
 * _RATEDTDP    // vP-state representing Rated TDP.
 * _BOOST       // vP-state to use for 3D boost and sudden increase in utilization.
 * _TURBOBOOST  // vP-state to use for displaying turbo boost clocks to the user.
 * _INFLECTION1 // vP-state index for power policy inflection point 1
 * _INFLECTION2 // vP-state index for power policy inflection point 2
 * _NUM_INDEXES // vP-state total number of indexes.
 */
#define LW2080_CTRL_PERF_VPSTATES_IDX_D2                          0x00    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_D3                          0x01    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_D4                          0x02    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_D5                          0x03    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_OVERLWR                     0x04    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_VRHOT                       0x05    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_MAXBATT                     0x06    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_MAXSLI                      0x07    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_MAXTHERMSUS                 0x08    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_SLOWDOWNPWR                 0x09    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_MIDPOINT                    0x0a    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_FLAGS                       0x0b    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_INFLECTION0                 0x0c    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_FULLDEFLECT                 0x0d    // PSTATES 2.0
#define LW2080_CTRL_PERF_VPSTATES_IDX_IMPFIRST                    0x0e    // PSTATES 3.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_IMPLAST                     0x0f    // PSTATES 3.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_RATEDTDP                    0x10    // PSTATES 2.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_BOOST                       0x11    // PSTATES 2.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_TURBOBOOST                  0x12    // PSTATES 2.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_MAX_LWSTOMER_BOOST          0x13    // PSTATES 2.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_INFLECTION1                 0x14    // PSTATES 3.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_INFLECTION2                 0x15    // PSTATES 3.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_WHISPER_MODE                0x16    // PSTATES 3.0+
#define LW2080_CTRL_PERF_VPSTATES_IDX_DLPPM_1X_ESTIMATION_MINIMUM 0x17
#define LW2080_CTRL_PERF_VPSTATES_IDX_DLPPC_1X_SEARCH_MINIMUM     0x18
#define LW2080_CTRL_PERF_VPSTATES_IDX_NUM_INDEXES                 0x19

typedef LwU8 LW2080_CTRL_PERF_VPSTATES_IDX;

 /*!
 * Structure describing PERF_VPSTATE_2X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_INFO_2X {
    LwU8  vPstateNum;                    // Externally visible vP-state number.
    LwU32 reqPower10mW;                  // Maximum power required for this vP-state.
    LwU32 reqSlowdownPower10mW;          // Maximum power required for this vP-state when slowdown is on.
    LwU32 value[LW2080_CTRL_PERF_VPSTATE_2X_DOMAIN_GROUP_NUM];    // Value to set for each domain group.
} LW2080_CTRL_PERF_VPSTATE_INFO_2X;

/*!
 * Entry representing a single clock domain inside a @ref LW2080_CTRL_PERF_VPSTATE_XYZ_3X entry.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_3X_CLOCK_ENTRY {
    LwU16 targetFreqMHz;                  // Target frequency in MHz.
    LwU16 minEffFreqMHz;                  // Minimum effective frequency in MHz.
    LwU16 origTargetFreqMHz;              // Original frequency value from Vpstate Table.
    LwU16 porFreqMHz;                     // Frequency value which is POR for this GPU
                                            // either original value from Vpstate Table or overridden value from Factory OC Table
} LW2080_CTRL_PERF_VPSTATE_3X_CLOCK_ENTRY;
typedef struct LW2080_CTRL_PERF_VPSTATE_3X_CLOCK_ENTRY *PLW2080_CTRL_PERF_VPSTATE_3X_CLOCK_ENTRY;

/*!
 * Entry representing a single clock domain inside a @ref LW2080_CTRL_PERF_VPSTATE_XYZ_3X entry.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_3X_CONTROL_CLOCK_ENTRY {
    LwU16 targetFreqMHz;                  // Target frequency in MHz.
    LwU16 minEffFreqMHz;                  // Minimum effective frequency in MHz.
} LW2080_CTRL_PERF_VPSTATE_3X_CONTROL_CLOCK_ENTRY;
typedef struct LW2080_CTRL_PERF_VPSTATE_3X_CONTROL_CLOCK_ENTRY *PLW2080_CTRL_PERF_VPSTATE_3X_CONTROL_CLOCK_ENTRY;

/*!
 * Structure describing PERF_VPSTATE_3X static information/POR.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_INFO_3X {
    LwU8                                    pstateIdx;                      // index pointing to the P-state associated with this Vpstate.
    /*!
     * Array of clocks specified by this Vpstate.
     */
    LW2080_CTRL_PERF_VPSTATE_3X_CLOCK_ENTRY clocks[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_VPSTATE_INFO_3X;

/*!
 * PERF_VPSTATE type-specific data union.  Discriminated by
 * PERF_VPSTATE::super.type.
 */


/*!
 * Structure describing PERF_VPSTATE static information/POR.  Implements the
 * BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_VPSTATE_INFO_2X v2x;
        LW2080_CTRL_PERF_VPSTATE_INFO_3X v3x;
    } data;
} LW2080_CTRL_PERF_VPSTATE_INFO;

/*!
 * Structure describing PERF_VPSTATES static information/POR.  Implements the
 * BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_VPSTATES_INFO_MESSAGE_ID (0xA4U)

typedef struct LW2080_CTRL_PERF_VPSTATES_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255  super;

    /*!
     * PERF_VPSTATE version.  @ref LW2080_CTRL_PERF_VPSTATE_VERSION_<xyz>
     */
    LwU8                          version;

    /*!
     * Domain Groups available in this VPstate table. For Pstate 2.0. This is typically
     * Pstate and GPC2CLK (2). For Pstate 3.0, this will be number of enabled
     * clock domains.
     */
    LwU32                         nDomainGroups;

    /*!
     * Array of VPSTATE indexes/names - ref@ LW2080_CTRL_PERF_VPSTATES_IDX_****
     * These indexes are used to abstract away implementation details from
     * client interfaces - e.g. a client can request the "BOOST" vpstate without
     * having to know which vpstate index.
     *
     * @note LW2080_CTRL_PERF_VPSTATE_INDEX_ILWALID indicates that a vpstate
     * index is not present/specified on this GPU.
     */
    LwU8                          vpstateIdx[LW2080_CTRL_PERF_VPSTATES_IDX_NUM_INDEXES];

    /*!
     * Array of PERF_VPSTATE structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_VPSTATE_INFO vpstates[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_VPSTATES_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_VPSTATES_GET_INFO
 *
 * This command returns PERF_VPSTATES static object information/POR as specified
 * by the VBIOS in vpstate Table.
 *
 * See @ref LW2080_CTRL_PERF_VPSTATES_INFO for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_PERF_VPSTATES_GET_INFO (0x208020a4) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_VPSTATES_INFO_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with PERF_VPSTATE_2X.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_CONTROL_2X {
    /*!
     * Value to set for each domain group.
     */
    LwU32 value[LW2080_CTRL_PERF_VPSTATE_2X_DOMAIN_GROUP_NUM];
} LW2080_CTRL_PERF_VPSTATE_CONTROL_2X;

/*!
 * Structure representing the control parameters associated with PERF_VPSTATE_3X.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_CONTROL_3X {
    /*!
     * Index pointing to the P-state associated with this Vpstate.
     */
    LwU8                                            pstateIdx;
    /*!
     * Array of clocks specified by this Vpstate.
     */
    LW2080_CTRL_PERF_VPSTATE_3X_CONTROL_CLOCK_ENTRY clocks[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_PERF_VPSTATE_CONTROL_3X;

/*!
 * PERF_VPSTATE type-specific data union.  Discriminated by
 * PERF_VPSTATE::super.type.
 */


/*!
 * Structure representing the control parameters associated with PERF_VPSTATE.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_VPSTATE_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant.  Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_VPSTATE_CONTROL_2X v2x;
        LW2080_CTRL_PERF_VPSTATE_CONTROL_3X v3x;
    } data;
} LW2080_CTRL_PERF_VPSTATE_CONTROL;

/*!
 * Structure representing the control parameters associated with PERF_VPSTATES.
 * Implements the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_PERF_VPSTATES_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255     super;

    /*!
     * bOriginal = LW_TRUE  : VBIOS stored value
     * bOriginal = LW_FALSE : Current VPSTATEs value
     * We will only allow to GET ORIGINAL value so that user can use it to
     * reset the VPSTATEs to VBIOS value, User will NOT be able to change
     * Original value.
     */
    LwBool                           bOriginal;

    /*!
     * Array of PERF_VPSTATE structures.  Has valid indexes corresponding to the
     * bits set in @ref super.objMask.
     */
    LW2080_CTRL_PERF_VPSTATE_CONTROL vpstates[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_VPSTATES_CONTROL;

/*!
 * LW2080_CTRL_CMD_PERF_VPSTATES_GET_CONTROL
 *
 * This command returns PERF_VPSTATES control parameters as specified by the
 * VBIOS in the VPSTATE Table.
 *
 * See @ref LW2080_CTRL_PERF_VPSTATES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetControl.
 */
#define LW2080_CTRL_CMD_PERF_VPSTATES_GET_CONTROL         (0x208020a5) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xA5" */

/*!
 * LW2080_CTRL_CMD_PERF_VPSTATES_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * PERF_VPSTATES entries in the VPSTATE Table and applies these new parameters
 * to the set of PERF_VPSTATES entries.
 *
 *
 * See @ref LW2080_CTRL_PERF_VPSTATES_CONTROL for documentation on the
 * parameters.
 *
 * Return values are specified per @ref BoardObjGrpSetControl.
 */
#define LW2080_CTRL_CMD_PERF_VPSTATES_SET_CONTROL         (0x208020a6) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xA6" */

/*-----------------------------  CHANGES_SEQ --------------------------------*/

/*!
 * Enumeration of the PERF CHANGE_SEQ feature version.
 *
 * _2X  - Legacy implementation of CHANGE_SEQ used in pstates 3.0 and earlier.
 * _PMU - Represents PMU based perf change sequence class and its sub-classes.
 * _31  - CHANGE_SEQ implementation used with pstates 3.1 and later.
 * _35  - CHANGE_SEQ implementation used with pstates 3.5 and later.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_UNKNOWN       LW_U8_MAX
#define LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_2X            0x01
#define LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_PMU           0x02
#define LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_31            0x03
#define LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_35            0x04

/*!
 * Flags to provide information about the input perf change request.
 * This flags will be used to understand the type of perf change req.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_NONE           0x00
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_FORCE                         LWBIT(0)
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_FORCE_CLOCKS                  LWBIT(1)
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_ASYNC                         LWBIT(2)
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_SKIP_VBLANK_WAIT              LWBIT(3)

/*!
 * perfStateTrigger() called from deferred WORKITEM - don't need to
 * defer beginning of VF change to WORKITEM.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_FROM_WORKITEM                        LWBIT(5)

/*!
 * INVALID value for changeSeqId, indicating no actual CHANGE (i.e. no CHANGE
 * was queued).
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_SEQ_ID_ILWALID LW_U32_MAX

/*!
 * Helper macro to increment @ref CHANGE_SEQ::changeSeqId.  This helper
 * function can be reused within CHANGE_SEQ_PMU implementations
 * whenever queueing a new input change.
 *
 * Takes care to avoid incrementing to _ILWALID value, such that an INVALID @ref
 * will never match with @ref CHANGE_SEQ_PMU::changeSeqId.
 *
 * @param[in/out] pSeqId - Pointer to the changeSeqId to increment.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_SEQ_ID_INCREMENT(pSeqId)                   \
    do {                                                                              \
        (*(pSeqId)) =                                                                 \
            ((*(pSeqId)) == LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_SEQ_ID_ILWALID - 1U) ? \
                0U : (*(pSeqId)) + 1U;                                                \
    } while (LW_FALSE)

/*!
 * Helper macro to determine if a seqId has been completed by the
 * CHANGE_SEQ or not.
 *
 * This is done by taking the delta between the requested/waited and the last
 * sequence IDs and comparing it against 0. If the delta is greater than or equal
 * to 0, then the seq. ID has been processed.
 *
 * @param[in]  seqIdLast      Sequence ID of the last change which CHANGE_SEQ has completed.
 * @param[in]  seqIdWait      Sequence ID which caller wants to know has been completed
 *
 * @return LW_TRUE if the seqIdWait has been completed; LW_FALSE otherwise
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_SEQ_ID_IS_COMPLETED(seqIdLast, seqIdWait) \
  (((LwS32)(seqIdLast) - (LwS32)(seqIdWait)) >= 0)

/*!
 * Structure representing input change struct for clock programming.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_CLK {
   /*!
     * [In] Target Frequency to be set in KHz.
     * This should ideally have been in MHz, but the clocks code and the
     * interfaces in RM takes clock values in KHz. To avoid changing all the
     * interfaces, keep the unit in kHz here.
     */
    LwU32 clkFreqkHz;
} LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_CLK;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_CLK *PLW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_CLK;

/*!
 * Structure representing input change struct for voltage programming.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT {
    /*!
     * VOLT_RAIL voltage values (uV). Indexes correspond to indexes into
     * VOLT_RAILS BOARDOBJGRP. This array has valid indexes corresponding to
     * bits set in @ref voltRailsMask
     */
    LwU32 voltageuV;

    /*!
     * V_{min, noise-unaware} - The minimum voltage (uV) with respect to
     * noise-unaware constraints on this VOLT_RAIL. Indexes correspond to
     * indexes into VOLT_RAILS BOARDOBJGRP. This array has valid indexes
     * corresponding to bits set in @ref voltRailsMask
     */
    LwU32 voltageMinNoiseUnawareuV;
} LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT *PLW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT;

/*!
 * Structure representing input change struct. Client of perf change sequence
 * will queue the change request using this input struct. CHange sequecer will
 * internally generate the @ref LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE struct
 * from this input struct for implementing the change.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT {
    /*!
     * Pstate index to set.
     */
    LwU32                                         pstateIndex;

    /*!
     * Flags for perf change req. @ref LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_<XYZ>
     */
    LwU32                                         flags;

    /*!
     * Used to check if VF data is stale.
     */
    LwU32                                         vfPointsCacheCounter;

    /*!
     * Mask of CLK_DOMAINs to be programmed (1-1 mapped with CLK_DOMAIN index).
     * For each clock domain to be programmed, client will set the corresponding
     * bit in the mask and fill-in the target frequency @ref clkFreqkHz for that
     * clock domain.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32              clkDomainsMask;

    /*!
     * CLK_DOMAIN change information corresponding to the mask bit set in
     * @ref clkDomainsMask
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_CLK  clk[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];

    /*!
     * Mask of VOLT_RAILs to be programmed (1-1 mapped with VOLT_RAIL index).
     * For each volt rail to be programmed, client will set the corresponding
     * bit in the mask and fill-in the target voltage @ref clkFreqkHz for that
     * volt rail.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32              voltRailsMask;

    /*!
     * VOLT_RAIL change information corresponding to the mask bit set in
     * @ref voltRailsMask
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT volt[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT *PLW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT;

/*!
 * Structure representing input voltage offset struct per rail.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET_ITEM {
    /*!
     * Mask of valid offset.
     * Voltage margin offset will be applied via VOLT_RAIL whereas
     * CLFC and CLVC will be applied via clock controllers.
     */
    LwU32 offsetMask;

    /*!
     * Array of Voltage offset (uV). Index in this array is statically mapped
     * to @ref LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_<XYZ>
     */
    LwS32 voltOffsetuV[LW2080_CTRL_VOLT_VOLT_RAIL_OFFSET_MAX];
} LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET_ITEM;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET_ITEM *PLW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET_ITEM;

/*!
 * Structure representing input change voltage offset struct.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET {
    /*!
     * Mask of VOLT_RAILs to be programmed (1-1 mapped with VOLT_RAIL index).
     * For each volt rail to be programmed, client will set the corresponding
     * bit in the mask and fill-in the target voltage offset for that volt rail.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                          voltRailsMask;

    /*!
     * By design controllers send delta on top of existing delta so we need
     * special boolean to indicate to change sequencer that we need to override
     * the voltage offset to zero. This is generally required when we disable
     * controllers or update their allowable offset range.
     */
    LwBool                                                    bOverrideVoltOffset;

    /*!
     * Boolean indicating whether perf change is forced.
     * In case of voltage margin offset update, the arbiter
     * will trigger perf change via the ilwalidation code path
     * so force change is not required.
     */
    LwBool                                                    bForceChange;

    /*!
     * VOLT_RAIL change information corresponding to the mask bit set in
     * @ref voltRailsMask
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET_ITEM rails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET *PLW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT_VOLT_OFFSET;

/*!
 * This struct contain pmu perf change sequence specific change parameters.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_PMU {
    /*!
     * Pmu Perf change sequence id. It is assigned at the time of enqueue.
     * After completion of each perf change request, pmu perf change
     * sequence will send notifications to all the clients blocked
     * on the sync perf change requests whose sequence id is less than
     * or equal to the latest completed change.
     */
    LwU32 seqId;
} LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_PMU;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_PMU *PLW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_PMU;

/*!
 * CHANGE_SEQ version-specific data union. Discriminated by
 * CHANGE_SEQ::version.
 */


/*!
 * @brief This struct contain all the information required to successfully
 * complete given perf change request.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE {
    /*!
     * List of clock domains domains and the values in KHz to set.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_LIST clkList;
    /*!
     * List of target voltage domains and the values in uV to set.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_LIST voltList;
    /*!
     * Pstate index to set.
     */
    LwU32                           pstateIndex;
    /*!
     * Flags for perf change req. @ref LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_<XYZ>
     */
    LwU32                           flags;
    /*!
     * Used to check if VF data is stale.
     */
    LwU32                           vfPointsCacheCounter;
    /*!
     * Current tFAW value to be programmed in HW
     */
    LwU8                            tFAW;
    /*!
     * Following parameters are private member of perf change sequence for
     * internal tracking and debugging purpose. Clients MUST NOT update
     * these parameters.
     */

    /*!
     * Perf Change Sequence Structure Version.
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_<xyz>.
     */
    LwU8                            version;

    /*!
     * Version-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_PMU pmu;
    } data;
} LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE *PLW2080_CTRL_PERF_CHANGE_SEQ_CHANGE;

/*!
 * Enumeration of PERF LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT - unique
 * identifiers for each different clients of perf change sequence lock.
 */
typedef enum LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT {
    /*!
     * RM will use this to lock the change sequence during system boot and
     * suspend / resume operation.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_RM_INIT = 0,
    /*!
     * This is used to lock the change sequence till the PMU bootstrap.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_PMU_INIT = 1,
    /*!
     * This is used by DISP clock for locking change sequence during modeset.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_DISP_MODESET = 2,
    /*!
     * This is used by DISP clock for locking change sequence during channel
     * allocation.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_DISP_CHANNEL_ALLOC = 3,
    /*!
     * This is used by MCLK clock for locking change sequence during FB memory
     * programming.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_MCLK_GDDR5_INIT = 4,
    /*!
     * [Deprecated] In legacy code, this is used while updating GPU state.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_STATE_UPDATE = 5,
    /*!
     * This is used by RMCTRL to switch the PCI-E bus speed.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_PCIE_BUS = 6,
    /*!
     * This is used by back door APIs to update clk/voltage/pstate.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_BACKDOOR_API = 7,
    /*!
     * This is used by perf change sequence RMCTRL calls to lock perf change
     * sequence on external clients (PERFDEBUG / MODS) request.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_API = 8,
    /*!
     * This is used by FBFLCN to stop perf change until FBFLCN sends
     * its queues information down to PMU.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_FBFLCN_INIT = 9,

    /*!
     * This is used by window reserved method in RM for locking change
     * sequencer while updating MCLK switch watermarks.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_UPDATE_MCLK_WATERMARKS = 10,

    /*!
     * This is used by CLIENT_PERF_CF_PWR_MODEL_PROFILE_SETUP rmctrl
     * to lock change sequencer before performing pwr model profile estimations.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_CLIENT_PERF_CF_PWR_MODEL_PROFILE_SETUP = 11,
} LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT;

/*!
 * Enumeration of PERF CHANGE_SEQ_STEP_IDs - unique identifiers for each
 * different step possible in a CHANGE_SEQ_SCRIPT.
 *
 * @note A CHANGE_SEQ_SCRIPT is not required to include each CHANGE_SEQ_STEP_ID.
 * The CHANGE_SEQ will choose which CHANGE_SEQ_STEP_IDs are required per the
 * semantics of the requested change.
 */
typedef enum LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID {
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_NONE = 0,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_WAIT_FOR_MODESET = 1,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_KMD_NOTIFY = 2,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_PRE_HW = 3,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_NOISE_UNAWARE_CLKS = 4,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_VOLT_INCREASE = 5,
    LW2080_CTRL_PERF_CHANGE_SEQ_30_STEP_ID_NOISE_AWARE_CLKS_INCREASE = 6,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LINK_SPEED = 7,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LINK_WIDTH = 8,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_ASPM = 9,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_DEEP_L1 = 10,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_RAM_SVOP = 11,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LWVDD_PSI = 12,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_VOLT_DECREASE = 13,
    LW2080_CTRL_PERF_CHANGE_SEQ_30_STEP_ID_NOISE_AWARE_CLKS_DECREASE = 14,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LTR_VALUE = 15,
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_LPWR_FEATURES = 16,

    // Non-blocking operations that always come after hardware changes
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_POST_HW = 17,

    // This must always be defined last
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_MAX_STEPS = 18,
} LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID;
typedef enum LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID *PLW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID;

/*!
 * Enumeration of CHANGE_SEQ_2X_SCRIPT_STATE - the state of a CHANGE_SEQ_SCRIPT
 * as the CHANGE_SEQ generates and then processes it.
 */
typedef enum LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE {
    /*!
     * The CHANGE_SEQ_SCRIPT is unused - i.e. the CHANGE_SEQ_SCRIPT structure
     * does not contain a sequence of CHANGE_SEQ_STEP_IDs to execute.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_NONE = 0,
    /*!
     * The CHANGE_SEQ_SCRIPT is initialized with a sequence of
     * CHANGE_SEQ_STEP_IDs to execute, but the CHANGE_SEQ has not yet begun
     * processing it.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_INITIALIZED = 1,
    /*!
     * The CHANGE_SEQ is processing the CHANGE_SEQ_SCRIPT and has not yet
     * finished.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_IN_PROGRESS = 2,
    /*!
     * The CHANGE_SEQ has finished processing the CHANGE_SEQ_SCRIPT.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_DONE = 3,
} LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE;
typedef enum LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE *PLW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE;

/*!
 * PP-TODO : Deprecated, Need to delete them after updating LWAPIs
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_NONE                      LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_NONE
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_WAIT_FOR_MODESET          LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_WAIT_FOR_MODESET
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_KMD_NOTIFY                LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_KMD_NOTIFY
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_PRE_HW                    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_PRE_HW
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_NOISE_UNAWARE_CLKS        LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_NOISE_UNAWARE_CLKS
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_VOLT_INCREASE             LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_VOLT_INCREASE
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_NOISE_AWARE_CLKS_INCREASE LW2080_CTRL_PERF_CHANGE_SEQ_30_STEP_ID_NOISE_AWARE_CLKS_INCREASE
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_SET_LINK_SPEED            LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LINK_SPEED
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_SET_LINK_WIDTH            LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LINK_WIDTH
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_SET_ASPM                  LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_ASPM
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_SET_DEEP_L1               LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_DEEP_L1
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_SET_RAM_SVOP              LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_RAM_SVOP
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_SET_LWVDD_PSI             LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LWVDD_PSI
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_VOLT_DECREASE             LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_VOLT_DECREASE
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_NOISE_AWARE_CLKS_DECREASE LW2080_CTRL_PERF_CHANGE_SEQ_30_STEP_ID_NOISE_AWARE_CLKS_DECREASE
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_SET_LTR_VALUE             LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_SET_LTR_VALUE
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_LPWR_FEATURES             LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_LPWR_FEATURES
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_POST_HW                   LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_POST_HW
#define LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID_MAX_STEPS                 LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_MAX_STEPS


typedef LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID LW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID;
typedef enum LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID *PLW2080_CTRL_PERF_CHANGE_SEQ_STEP_ID;

/*!
 * PP-TODO : Deprecated, Need to delete them after updating LWAPIs
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STATE_NONE        LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_NONE
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STATE_INITIALIZED LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_INITIALIZED
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STATE_IN_PROGRESS LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_IN_PROGRESS
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STATE_DONE        LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE_DONE

typedef LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STATE;
typedef enum LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE *PLW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STATE;

/*!
 * Structure representing the profiling data for a CHANGE_SEQ_SCRIPT_STEP.  This
 * is a collection of timestamps for various metrics.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP_PROFILING {
    /*!
     * Total elapsed time (ns) to process the CHANGE_SEQ_SCRIPT_STEP - i.e. the
     * measured difference in wall time.  This number is measured between the
     * time the CHANGE_SEQ begins processing the step and the time CHANGE_SEQ
     * determines the step is finished.  That value may include communication
     * overhead/delays and may not be an accurate representation of how long the
     * step took in HW, nor is it a measurent of how long the RM took to process
     * the step while holding the RM lock.
     */
    LwU64_ALIGN32 totalTimens;

    /*!
     * Total RM thread time (ns) spent processing the CHANGE_SEQ_SCRIPT_STEP -
     * i.e. the amount of time the CHNAGE_SEQ spent processing this
     * CHANGE_SEQ_SCRIPT_STEP while holding the RM lock.
     *
     * With asynchronous changes, this number may be the sum of several
     * different RM threads processing the thread - e.g. one to start the
     * asynchronous operation, one to complete the operation when the
     * asynchronous operation finishes and resumes the CHANGE_SEQ.
     *
     * This number is important, and should be minimized, as long-running
     * threads holding the RM lock can delay interrupts, which can have negative
     * impact on user-experience (i.e. stutter, delayed interrupt notification,
     * etc.).
     *
     * This number should always be <= @ref totalTimens.
     */
    LwU64_ALIGN32 rmThreadTimens;

    /*!
     * Total PMU thread time (ns) spent processing the CHANGE_SEQ_SCRIPT_STEP -
     * i.e. the amount of time the CHNAGE_SEQ spent processing this
     * CHANGE_SEQ_SCRIPT_STEP in PMU.
     *
     * This number should always be <= @ref totalTimens.
     */
    LwU64_ALIGN32 pmuThreadTimens;
} LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP_PROFILING;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP_PROFILING *PLW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP_PROFILING;

/*!
 * Structure representing a single STEP in CHANGE_SEQ_SCRIPT.  Contains the
 * identifying information about the STEP and (eventually) any applicable
 * parameters, as well as diagnostic state.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STEP {
    /*!
     * LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID of this STEP.  Indicates the
     * operation to be done at this step.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID            stepId;

    LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP_PROFILING profiling;
} LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STEP;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STEP *PLW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STEP;

/*!
 * PP-TODO : Deprecated, Need to delete them after updating LWAPIs
 */
typedef LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STEP LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STEP *PLW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP;

/*!
 * Structure representing the profiling data for a given CHANGE_SEQ thread when
 * processing a CHANGE_SEQ_SCRIPT.
 *
 * With asysnchronous CHANGE_SEQ processing, the CHANGE_SEQ thread may
 * encounter an asynchronous operation (i.e. status ==
 * LW_WARN_MORE_PROCESSING_REQUIRED).  In this case, the thread will return up
 * the stack and release the RM lock, depending on the notification of
 * completion of the asynchronous operation to kick off a new CHANGE_SEQ thread.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING_THREAD {
    /*!
     * Mask of step indexes (i.e. indexes into the @ref
     * CHANGE_SEQ_SCRIPT::steps[] array) which this thread processed.
     */
    LwU32         stepMask;
    /*!
     * Total elapsed time (ns) of this thread while processing the
     * CHANGE_SEQ_SCRIPT.  This will be the entire time this thread processed
     * the CHANGE_SEQ_SCRIPT while holding the RM lock.
     *
     * This number is important because it has a direct impact on user
     * experience - if a thread holds the RM lock for too long, it can delay
     * interrupts which may be important for user experience (causing stutter or
     * delayed notification of interrupts).  The RM should aim to always keep
     * this number far below 1ms (maybe even 500us) for asynchronous changes.
     */
    LwU64_ALIGN32 timens;
} LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING_THREAD;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING_THREAD *PLW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING_THREAD;

/*!
 * Maximum number of threads the CHANGE_SEQ_SCRIPT can support in its profiling
 * data.
 *
 * Current value (8) is *far* above what we expect to see in actuality (probably
 * 3-4).
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_MAX_PROFILING_THREADS 8

/*!
 * Structure representing the profiling data for an overall CHANGE_SEQ_SCRIPT.
 * This is a collection of timestamps for various metrics.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING {
    /*!
     * Total elapsed time (ns) to process the CHANGE_SEQ_SCRIPT - i.e. the
     * measured difference in wall time.  This number is measured between the
     * time the CHANGE_SEQ begins processing the CHANGE_SEQ_SCRIPT and the time
     * CHANGE_SEQ determines the CHANGE_SEQ_SCRIPT is finished.  That value may
     * include communication overhead/delays and may not be an accurate
     * representation of how long the step took in HW, nor is it a measurent of
     * how long the RM took to process the step while holding the RM lock.
     */
    LwU64_ALIGN32                                       totalTimens;

    /*!
     * Total elapsed time (ns) to build the CHANGE_SEQ_SCRIPT.
     */
    LwU64_ALIGN32                                       totalBuildTimens;

    /*!
     * Total elapsed time (ns) to execute the CHANGE_SEQ_SCRIPT.
     */
    LwU64_ALIGN32                                       totalExelwtionTimens;

    /*!
     * Total number of threads in the CHANGE_SEQ required to process this
     * CHANGE_SEQ_SCRIPT.  Defines valid indexes in the @ref rmThreads[] array.
     */
    LwU8                                                numRmThreads;
    /*!
     * Array of profiling data for each CHANGE_SEQ thread.  Has valid indexes in
     * the range [0, @ref numRmThreads).
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING_THREAD rmThreads[LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_MAX_PROFILING_THREADS];
} LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING *PLW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING;

/*!
 * Structure to store all the common data of a perf change sequence. This makes
 * it easier to be accessed by RMCTRL to expose profiling information.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT {
    /*!
     * State of the CHANGE_SEQ_SCRIPT object.  Tracks where the CHANGE_SEQ
     * is in processing this script.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STATE  state;
    /*!
     * Array of LW2080_CTRL_PERF_CHANGE_SEQ_STEPs to be exectuted in this
     * CHANGE_SEQ_SCRIPT.  Has valid indexes in range [0, @ref numSteps).
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT_STEP   steps[LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID_MAX_STEPS];
    /*!
     * Total number of steps in the CHANGES_SEQ_SCRIPT.  Specifies valid indexes
     * in the @ref majorSteps[] array.
     */
    LwU32                                        numSteps;
    /*!
     * Index into @ref majorSteps[] array of current step that CHANGE_SEQ is
     * processing.
     */
    LwU32                                        lwrStepIndex;
    /*!
     * Profiling information for the overall PERF_CHANGE_SEQ_SCRIPT.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING profiling;
} LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT *PLW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT;

/*!
 * PP-TODO : Deprecated, Need to delete them after updating LWAPIs
 */
typedef LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT *PLW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT;

/*!
 * Enumeration of PERF CHANGE_SEQ_PMU_STEP_IDs - unique identifiers for each
 * different step possible in a CHANGE_SEQ_PMU_SCRIPT.
 *
 * @note
 * A CHANGE_SEQ_PMU_SCRIPT is not required to include each CHANGE_SEQ_PMU_STEP_ID.
 * The CHANGE_SEQ will choose which CHANGE_SEQ_PMU_STEP_IDs are required per the
 * semantics of the requested change.
 */
typedef LwU32 LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID;
typedef LwU32 *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID;

/*!
 * Common steps across all perf change sequence versions
 */

/*!
 * This is a dummy step to occupy index 0.
 * This step is used just for super class pointer.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_NONE                     (0U)

/*!
 * Perform all necessary initialization at the start of every perf change.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CHANGE
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_PRE_CHANGE_RM            (1U)
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_PRE_CHANGE_PMU           (2U)

/*!
 * Perform all necessary completions at the end of every perf change.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CHANGE
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_POST_CHANGE_RM           (3U)
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_POST_CHANGE_PMU          (4U)

/*!
 * Perform all necessary initialization at the start of every PSTATE change.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_PSTATE
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_PRE_PSTATE_RM            (5U)
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_PRE_PSTATE_PMU           (6U)

/*!
 * Perform all necessary completions at the end of every PSTATE change.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_PSTATE
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_POST_PSTATE_RM           (7U)
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_POST_PSTATE_PMU          (8U)

/*!
 * Update / Set all voltage including noise-unaware Vmin.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_VOLT                     (9U)

/*!
 * Update LPWR features state based on the new PSTATE.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_LPWR                     (10U)

/*!
 * Update BIF features state based on the new PSTATE.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_BIF
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_BIF                      (11U)

/*!
 * Perf change sequence version 31 specific steps
 */

/*!
 * Update / Set all noise-unaware clocks
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_31_STEP_ID_NOISE_UNAWARE_CLKS        (12U)
/*!
 * Update / Set all noise-aware clocks
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_31_STEP_ID_NOISE_AWARE_CLKS          (13U)

/*!
 * Perf change sequence version 35 specific steps
 */

/*!
 * Update / Set all decreasing clocks and special case NAFLL
 * clocks as pre voltage step.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_35_STEP_ID_PRE_VOLT_CLKS             (14U)

/*!
 * Update / Set all increasing clocks as post voltage step.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_35_STEP_ID_POST_VOLT_CLKS            (15U)

/*!
 * Evaluate and program clock monitors threshold.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLK_MON
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_35_STEP_ID_CLK_MON                   (16U)

/*!
 * Clock steps specifically designed on TURING to bypass
 * clocks 3.0 code paths. This will be only used on
 * TURING AUTO SKUs that do not support non-NAFLL clocks.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_35_STEP_ID_PRE_VOLT_NAFLL_CLKS       (17U)

/*!
 * Update / Set all increasing clocks as post voltage step.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_35_STEP_ID_POST_VOLT_NAFLL_CLKS      (18U)

/*!
 * Deinit Clocks specified in clock list.
 * This step will be used by features like GPC-RG to disable HW access
 * of GPC chiplet such as GPC LUTs, GPC ADCs and GPC Clock Counters.
 *
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_CLK_DEINIT              (19U)

/*!
 * Init NAFLL devices.
 * This step will be used by features like GPC-RG to initialize NAFLL
 * devices at boot frequency into FFR regime following the same sequence
 * as of VBIOS boot.
 *
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_CLK_INIT                (20U)

/*!
 * Init Clocks specified in clock list.
 * This step will be used by features like GPC-RG to init and enable HW
 * access of GPC chiplet such as GPC LUTs, GPC ADCs and GPC Clock Counters.
 *
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_CLK_RESTORE             (21U)

/*!
 * Gate voltage rail specified in VOLT LIST.
 * This step will be used by features like GPC-RG to gate the LWVDD
 * voltage rail and its dependencies.
 *
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_VOLT_GATE               (22U)

/*!
 * Ungate voltage rail specified in VOLT LIST.
 * This step will be used by features like GPC-RG to ungate the LWVDD
 * voltage rail and its dependencies.
 *
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_VOLT_UNGATE             (23U)

/*!
 * Perform LPWR sequence pre and post voltage gate.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_PRE_VOLT_GATE_LPWR      (24U)
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_POST_VOLT_GATE_LPWR     (25U)

/*!
 * Perform LPWR sequence pre and post voltage ungate.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_PRE_VOLT_UNGATE_LPWR    (26U)
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_STEP_ID_POST_VOLT_UNGATE_LPWR   (27U)

/*!
 * Perf change sequence version 35 temporarity XBAR boost around MCLK steps
 */

/*!
 * Update / Set all decreasing clocks and special case NAFLL
 * clocks as pre voltage step for temporarily XBAR boost.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_35_STEP_ID_XBAR_BOOST_PRE_VOLT_CLKS  (28U)

/*!
 * Update / Set all increasing clocks and special case NAFLL
 * clocks as post voltage step for temporarily XBAR boost.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_35_STEP_ID_XBAR_BOOST_POST_VOLT_CLKS (29U)

/*!
 * Update / Set all voltage including noise-unaware Vmin for temporarily XBAR boost.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_XBAR_BOOST_VOLT          (30U)

/*!
 * Update memory settings for memory tuning.
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_MEM_TUNE
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_MEM_TUNE                 (31U)

/*!
 * This must always be defined last
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID_MAX_STEPS                (32U)

/*!
 * Macro defining the max allowed steps in given group of perf change scripts.
 * These numbers are generated based on worst case max step ids that are
 * supported at any going point for a given script.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_VF_SWITCH_MAX_STEPS           (14U)
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_LPWR_MAX_STEPS                (4U)

/*!
 * Enumeration of CHANGE_SEQ_PMU_SCRIPT_STATE - the state of a
 * CHANGE_SEQ_PMU_SCRIPT as the CHANGE_SEQ_PMU generates and then processes it.
 */
typedef LwU32 LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE;
typedef LwU32 *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE;

/*!
 * The CHANGE_SEQ_PMU_SCRIPT is idle - i.e. the CHANGE_SEQ_PMU_SCRIPT
 * structure does not contain a sequence of CHANGE_SEQ_PMU_STEP_IDs to
 * execute. Possible next steps are :: _LOCKED or _IN_PROGRESS
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE_NONE        (0x00U)

/*!
 * The CHANGE_SEQ_PMU_SCRIPT is locked by clients. This client could be RM
 * or any other falcon including PMU who want to ensure that there is NO
 * perf change while they are processing certain task / operation.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE_LOCKED      (0x01U)

/*!
 * The CHANGE_SEQ_PMU is processing the CHANGE_SEQ_PMU_SCRIPT and has not yet
 * finished.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE_IN_PROGRESS (0x02U)

/*!
 * The CHANGE_SEQ_PMU is waiting for new change as last change was SKIPPED.
 * Possible reason for skipping change request -
 *      VF lwrve changed between the time change was queued and change exelwtion begin.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE_WAITING     (0x03U)

/*!
 * The CHANGE_SEQ hit error during build / exelwtion of change sequencer script.
 * Once Change sequencer goes into _ERROR state, there is possible future state
 * transition, in other words, it does not know how to recover to sane state from
 * _ERROR state.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE_ERROR       (0x04U)

/*!
 * Structure representing global data shared by all the perf change sequence pmu
 * steps.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER {
    /*!
     * LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID of this STEP. Indicates the
     * operation to be done at this step.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID           stepId;

    /*!
     * Profiling information for exelwtion of step represented by @ref stepId.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_STEP_PROFILING profiling;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER;

/*!
 * Structure representing data needed to perform actions necessary at the start/end
 * of each change.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CHANGE {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;

    /*!
     * Pstate index to set.
     */
    LwU32                                             pstateIndex;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CHANGE;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CHANGE *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CHANGE;

/*!
 * Structure representing data needed to perform actions necessary at the start/end
 * of each pstate change.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_PSTATE {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;

    /*!
     * Pstate index to set.
     */
    LwU32                                             pstateIndex;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_PSTATE;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_PSTATE *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_PSTATE;

/*!
 * Structure representing data needed to perform actions necessary in LPWR state
 * based on new pstate.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;

    /*!
     * Pstate index to set.
     */
    LwU32                                             pstateIndex;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR;

/*!
 * Structure representing data needed to perform actions necessary in BIF state
 * based on new pstate.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_BIF {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;

    /*!
     * Pstate index to set.
     */
    LwU32                                             pstateIndex;

    /*!
     * Index into PCIE Table
     * Default value = 0xFF (Invalid)
     * wiki : https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/LPWR_VBIOS_Table#PCIE_Table
     */
    LwU8                                              pcieIdx;

    /*!
     * Index into LWLINK Table
     * Default value = 0xFF (Invalid)
     * wiki : https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components/LowPower/LPWR_VBIOS_Table#LWLINK_Table
     */
    LwU8                                              lwlinkIdx;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_BIF;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_BIF *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_BIF;

/*!
 * Structure representing data needed to update the clocks during perf change.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;

    /*!
     * Ordered list of noise aware / noise unaware clocks for a P-state
     * or decoupled clock change.
     */
    LW2080_CTRL_CLK_CLK_DOMAIN_LIST                   clkList;

    /*!
     * ADC SW override list. @ref LW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST
     */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST              adcSwOverrideList;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS;

/*!
 * Structure representing data needed to update memory settings for tuning
 * memory as per the dynamic performance requirements.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_MEM_TUNE {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;
    /*!
     * Current tFAW value to be programmed in HW
     */
    LwU8                                              tFAW;
    /*!
     * Current memory clock frequency. This is required to determine
     * the tRRD value based on above mode of operation @ref tFAW
     */
    LwU32                                             mclkFreqKHz;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_MEM_TUNE;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_MEM_TUNE *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_MEM_TUNE;

/*!
 * Structure representing data needed to update the voltage related parameters
 * during the perf change. It includes setting voltage min and updating the lwrr
 * voltage value.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;

    /*!
     * List of voltage rails items that needs to be set during the
     * VOLTAGE_STEP as part of a p-state or a decoupled clocks change. This
     * list is also sent to the PMU when the noise aware clocks are changed
     * to decide the regime in which those clocks will operate.
     */
    LW2080_CTRL_VOLT_VOLT_RAIL_LIST                   voltList;

    /*!
     * ADC SW override list. @ref LW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST
     */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LW2080_CTRL_CLK_ADC_SW_OVERRIDE_LIST              adcSwOverrideList;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT;

/*!
 * Structure representing data needed to update the clock monitors
 * thresholds during perf change.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLK_MON {
    /*!
     * Super class. Must always be first object in structure.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER super;

    /*!
     * Ordered list of clocks that require clock monitor programming
     */
    LW2080_CTRL_CLK_DOMAIN_CLK_MON_LIST               clkMonList;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLK_MON;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLK_MON *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLK_MON;

/*!
 * _CHANGE_SEQ_PMU type-specific data union.  Discriminated by
 * @ref LW2080_CTRL_PERF_CHANGE_SEQ_2X_STEP_ID
 */
typedef union LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_DATA {
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_SUPER    super;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CHANGE   change;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_PSTATE   pstate;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_LPWR     lpwr;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_BIF      bif;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLKS     clk;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_VOLT     volt;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_CLK_MON  clkMon;

    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_MEM_TUNE memTune;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_DATA;

typedef union LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_DATA *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_DATA;

/*!
 * Structure representing a single STEP in CHANGE_SEQ_PMU_SCRIPT.  Contains the
 * identifying information about the STEP and (eventually) any applicable
 * parameters, as well as diagnostic state.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP {
    /*!
     * LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID of this STEP. Indicates the
     * operation to be done at this step.
     * This is redundant param to support XAPI limitations. RMCTRL will
     * update this param.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEP_ID          stepId;

    /*!
     * Type-specific data union.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP_DATA data;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP;

/*!
 * Structure to store all the common data of a perf change sequence steps.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_HEADER {
    /*!
     * The order of all changes - pstates and decoupled clocks will be
     * determined by the voltages if they are moving in same direction
     * and clocks if they are moving in different directions.
     */
    LwBool                                       bIncrease;

    /*!
     * Script Id.
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_<xyz>
     */
    LwU8                                         scriptId;

    /*!
     * Total number of steps in the CHANGES_SEQ_PMU_SCRIPT. Specifies valid
     * indexes in the @ref steps[] array.
     */
    LwU8                                         numSteps;

    /*!
     * Index into @ref steps[] array of current step that CHANGE_SEQ_PMU is
     * processing.
     */
    LwU8                                         lwrStepIndex;

    /*!
     * Profiling information for the overall PERF_CHANGE_SEQ_SCRIPT.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_PROFILING profiling;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_HEADER;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_HEADER *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_HEADER;

/*!
 * Structure to store all the common data of a perf change sequence. This makes
 * it easier to be accessed by RMCTRL to expose profiling information.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT {
    /*!
     * Header struct containing global change sequence parameters. These params
     * are shared among all the change sequence steps.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_HEADER hdr;

    /*!
     * Stores the change request lwrrently being processed by the daemon.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE            changeLwrr;

    /*!
     * Array of LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEPs to be exectuted in this
     * CHANGE_SEQ_PMU_SCRIPT.  Has valid indexes in range [0, @ref numSteps).
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP   steps[LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_VF_SWITCH_MAX_STEPS];
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT;

/*!
 * Enumeration of Perf Change Sequencer supported LPWR scripts.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_LPWR_ENTRY     0U
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_LPWR_EXIT      1U
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_LPWR_POST      2U
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_LPWR_MAX       3U

/*!
 * Enumeration of Perf Change Sequencer supported PERF scripts.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_PERF_VF_SWITCH 0U
#define LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_PERF_MAX       1U

/*!
 * Structure describing the complete LPWR script. There will be multiple
 * instance of this script to execute entry, exit and post processing
 * of LPWR feature.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPT {
    /*!
     * Sequence id of LPWR Script. This sequence id will be incremented by one
     * on every script build and exelwtion.
     *
     * Specific for GPC-RG LPWR script, this sequence id will be used to sanity
     * check invalid request to process exit script without processing entry
     * script as well as to discard redundant request of post script exelwtion
     * in cases where VF switch inline exelwtes the post script.
     */
    LwU32                                         seqId;

    /*!
     * Header struct containing global change sequence parameters. These params
     * are shared among all the change sequence steps.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_HEADER hdr;

    /*!
     * Array of LW2080_CTRL_PERF_CHANGE_SEQ_PMU_STEPs to be exectuted in this
     * CHANGE_SEQ_PMU_SCRIPT.  Has valid indexes in range [0, @ref numSteps).
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STEP   steps[LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_LPWR_MAX_STEPS];
} LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPT;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPT *PLW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPT;

/*!
 * LPWR feature script id that will be used to get the pointer to the
 * group of perf LPWR scripts @ref LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPTS
 * for that feature.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPTS_ID_GPC_RG 0x00

/*!
 * Structure describing ALL supported LPWR scripts.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPTS {
    /*!
     * Global Sequence id of all LPWR Scripts of given LPWR feature.
     * This will be incremented only on the entry script exelwtion.
     */
    LwU32                                   seqId;

    /*!
     * Boolean tracking perf state with respect to the given LPWR feature.
     * This boolean will be set to true when perf starts the processing of
     * the entry script and will be set to false when perf completes the
     * processing of the post processing script (perf restore).
     */
    LwBool                                  bEngaged;

    /*!
     * Array of all supported LPWR scripts index by @ref LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_LPWR_<xyz>
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPT lpwrScripts[LW2080_CTRL_PERF_CHANGE_SEQ_SCRIPT_ID_LPWR_MAX];
} LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPTS;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPTS *PLW2080_CTRL_PERF_CHANGE_SEQ_LPWR_SCRIPTS;

/*!
 * Structure describing perf change sequence static information/POR.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_INFO_PMU {
    /*!
     * Mask of the steps Ids that PMU is advertizing to CPU.
     * CPU must claim them back to PMU via SET_INFO.
     */
    LwU32 cpuAdvertisedStepIdMask;

    /*!
     * Mask of the steps Ids that will be exelwted on CPU.
     */
    LwU32 cpuStepIdMask;
} LW2080_CTRL_PERF_CHANGE_SEQ_INFO_PMU;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_INFO_PMU *PLW2080_CTRL_PERF_CHANGE_SEQ_INFO_PMU;

/*!
 * CHANGE_SEQ Version-specific data union.  Discriminated by
 * CHANGE_SEQ::version.
 */


/*!
 * Structure describing perf change sequence static information/POR.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_INFO_MESSAGE_ID (0xABU)

typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_INFO {
    /*!
     * Perf Change Sequence Structure Version -
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_<xyz>.
     */
    LwU8 version;

    /*!
     * Boolean indicating whether the given version of perf change sequence has
     * PMU support enabled.
     */
    LwU8 bEnabledPmuSupport;

    /*!
     * Version-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CHANGE_SEQ_INFO_PMU pmu;
    } data;
} LW2080_CTRL_PERF_CHANGE_SEQ_INFO;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_INFO *PLW2080_CTRL_PERF_CHANGE_SEQ_INFO;

/*!
 * LW2080_CTRL_CMD_PERF_CHANGE_SEQ_GET_INFO
 *
 * This command returns CHANGE_SEQ static object information/POR.
 *
 * See @ref LW2080_CTRL_PERF_CHANGE_SEQ_INFO for documentation on the
 * parameters.
 */
#define LW2080_CTRL_CMD_PERF_CHANGE_SEQ_GET_INFO (0x208020ab) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CHANGE_SEQ_INFO_MESSAGE_ID" */

/*!
 * Structure representing the dynamic status of the parameters associated with
 * perf change sequence 2X
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_2X {
    /*!
     * Profiling data from the last change sequence script exelwted.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_2X_SCRIPT scriptLast;
} LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_2X;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_2X *PLW2080_CTRL_PERF_CHANGE_SEQ_STATUS_2X;

/*!
 * Flags to provide notifications about the completion event from pmu perf
 * change sequence.
 */

/*!
 * Completion event notification for perf change sequence lock request
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_COMPLETION_LOCK_ACQUIRED                0:0
#define LW2080_CTRL_PERF_CHANGE_SEQ_COMPLETION_LOCK_ACQUIRED_FALSE 0x00
#define LW2080_CTRL_PERF_CHANGE_SEQ_COMPLETION_LOCK_ACQUIRED_TRUE  0x01

/*!
 * Represents sync change queue size to determine the max allowed pending sync
 * perf change request at any given time.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_QUEUE_SIZE         0x08U

/*!
 * Enumeration of PERF LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT - unique
 * identifiers for each different clients of sync perf change requests.
 */
typedef LwU32 LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT;

/*!
 * Always at index 0.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_ILWALID (0x00U)

/*!
 * This is used to inform that the sync change was requested by RM.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_RM      (0x01U)

/*!
 * This is used to inform that the sync change was requested by PMU.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_PMU     (0x02U)

/*!
 * Structure to store all the pending sync change requests callback information.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_QUEUE {
    /*!
     * Perf change sequence id. It is assigned at the time of enqueue.
     * After completion of each perf change request, pmu perf change
     * sequence will send notifications to all the clients blocked
     * on the sync perf change requests whose sequence id is less than
     * or equal to the latest completed change.
     */
    LwU32                                          seqId;
    /*!
     * Client associated with the ref@ seqId. PMU perf change sequence
     * will use the client information to send the async notifications
     * to the client.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT clientId;
} LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_QUEUE;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_QUEUE *PLW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_QUEUE;

/*!
 * Represent count of bins used for histogram.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_BIN_COUNT  32U

/*!
 * Represent size of cirlwlar queue.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_QUEUE_SIZE 16U

/*!
 * @brief   Structure holding change's profiling data.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_QUEUE {
    /*!
     * Start time of PMU client's change request pass down from
     * clients.
     */
    LwU64_ALIGN32 timeStart;

    /*!
     * Perf change sequence id. It is assigned at the time of enqueue.
     * After completion of each perf change request, pmu perf change
     * sequence will use this id to update histogram data.
     */
    LwU32         seqId;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_QUEUE;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_QUEUE *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_QUEUE;

/*!
 * @brief   Structure holding task's LWOS profiling data.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING {
    /*!
     * Queue to store each perf change Request
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_QUEUE queue[LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_QUEUE_SIZE];

    LwU32                                                  timeMax;
    LwU32                                                  timeMin;

    LwU8                                                   smallBinCount;
    LwU8                                                   smallBinSizeLog2;
    LwU8                                                   bigBinSizeLog2;
    LwU32                                                  execTimeBins[LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING_BIN_COUNT];
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING;

/*!
 * Structure representing the dynamic parameters associated with pmu perf change
 * sequence.
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU {
    /*!
     * State of the CHANGE_SEQ_PMU_SCRIPT object. Tracks where the
     * CHANGE_SEQ is in-processing this script.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT_STATE         state;

    /*!
     * Boolean indicating if there is any pending lock request.
     * If the change sequence is in the middle of processing any change while
     * it receives request to lock change seq, it will set this flag to LW_TRUE.
     * When perf task receives a completion signal from perf daemon task, it
     * will check this flag. If the flag is set, it will send async message
     * to the client (RM) about the availability of lock and set the state to
     * LOCKED.
     */
    LwBool                                               bLockWaiting;

    /*!
     * Global perf change sequence id counter. It will be incremented when new
     * perf change req. is enqueued. The latest counter value will be assigned
     * to each input change request.
     *
     * After completion of each perf change request, pmu perf change sequence
     * will send notifications to all the clients blocked on the sync perf
     * change requests whose sequence id is less than or equal to the latest
     * completed change.
     */
    LwU32                                                seqIdCounter;

    /*!
     * @copydoc LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_QUEUE
     *
     * The trick to remove something from the queue is by setting the client to
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_ILWALID
     *
     * @note - This struct is private to perf change sequence and clients of
     * perf change sequence MUST not access it for any purpose.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_CLIENT_QUEUE syncChangeQueue[LW2080_CTRL_PERF_CHANGE_SEQ_SYNC_CHANGE_QUEUE_SIZE];

    /*!
     * PP-TODO::
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_CHANGE_PROFILING     profiling;
} LW2080_CTRL_PERF_CHANGE_SEQ_PMU;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_PMU *PLW2080_CTRL_PERF_CHANGE_SEQ_PMU;

/*!
 * Structure representing the dynamic status of the parameters associated with
 * perf change sequence pmu
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_PMU {
    /*!
     * Boolean caching the lock state information of pmu perf change sequence.
     * During RM init, it will store all the input lock requests and send it
     * down to pmu during PMU INIT event.
     *
     * @note    This param is not initialized at each perfChangeSeqStateInit to
     *          remember the input requests from already initialized objects.
     */
    LwBool                                 bLock;

    /*!
     * Mask of events that RM perf change sequence is waiting from pmu perf
     * change sequence.
     */
    LwU32                                  eventMaskPending;

    /*!
     * Mask of events that RM perf change sequence received from pmu perf
     * change sequence.
     *
     * @note RM will use both pending and received mask for sanity checking
     * the events received from PMU. RM will clear the event bit in both masks
     * upon successful completion of event.
     */
    LwU32                                  eventMaskReceived;

    /*!
     * Latest pending / completed Perf change sequence id. It is updated
     * upon successful enqueue of perf change request into PMU.
     * When RM received sync change completion event from PMU, it will
     * sanity check that the input pmu sequence id is greater than or
     * equal to the latest pending ref@ seqId.
     */
    LwU32                                  seqId;

    /*!
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_PMU
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU        pmu;

    /*!
     * The last change sequence script which the CHANGE_SEQ processed.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_PMU_SCRIPT scriptLast;
} LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_PMU;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_PMU *PLW2080_CTRL_PERF_CHANGE_SEQ_STATUS_PMU;

/*!
 * CHANGE_SEQ type-specific data union.  Discriminated by
 * CHANGE_SEQ::type.
 */



/*!
 * Structure representing the dynamic status of the parameters associated with
 * perf change sequence.
 */
#define LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_MESSAGE_ID (0xALW)

typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_STATUS {
    /*!
     * Perf Change Sequence Structure Version -
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_<xyz>.
     */
    LwU8                               version;

    /*!
     * Mask of RM clients that could request the change sequence lock.
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT
     */
    LwU32                              clientLockMask;

    /*!
     * Latest completed perf change request.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE changeLast;

    /*!
     * Version-specific data union.
     */
    union {
        LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_2X  v2x;
        LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_PMU pmu;
    } data;
} LW2080_CTRL_PERF_CHANGE_SEQ_STATUS;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_STATUS *PLW2080_CTRL_PERF_CHANGE_SEQ_STATUS;

 /*!
 * This command returns perf change sequence status data.
 *
 * See @ref LW2080_CTRL_PERF_CHANGE_SEQ_STATUS for documentation on the
 * parameters.
 *
 * @return LW_OK
 *      RMCTRL control data successfully applied.
 *
 * @return LW_ERR_ILWALID_POINTER
 *      One or more of the pointers was NULL.
 *
 * @return LW_ERR_ILWALID_OBJECT
 *      Was not able to obtain the specified object within the object group.
 *
 * @return LW_ERR_ILWALID_ARGUMENT
 *     User-provided input was incorrect/invalid.
 */
#define LW2080_CTRL_CMD_PERF_CHANGE_SEQ_GET_STATUS (0x208020ac) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_CHANGE_SEQ_STATUS_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with
 * perf change sequence
 */
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL {
    /*!
     * Perf Change Sequence Structure Version -
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_VERSION_<xyz>.
     */
    LwU8                                     version;

    /*!
     * Mask of clock domain entries that needs to be excluded during the perf
     * change sequence process. Client will provide this mask through set
     * control RMCTRL.
     * PS: This is the MASK of clock domain indices
     *     The mask identifying all modifiable domains will be determined by
     *     (ClkDomainsMask | clkDomainsInclusionMask) & ~clkDomainsExclusionMask
     */
    LW2080_CTRL_BOARDOBJGRP_E32              clkDomainsExclusionMask;

    /*!
     * Mask of clock domain entries that needs to be included during the perf
     * change sequence process. Client will provide this mask through set
     * control RMCTRL.
     * PS: This is the MASK of clock domain indices
     *     The mask identifying all modifiable domains will be determined by
     *     (ClkDomainsMask | clkDomainsInclusionMask) & ~clkDomainsExclusionMask
     */
    LW2080_CTRL_BOARDOBJGRP_E32              clkDomainsInclusionMask;

    /*!
     * Perf change sequencer step id exclusion mask. Each bit represents
     * bit position of LW2080_CTRL_PERF_CHANGE_SEQ_<abc>_STEP_ID_<xyz>
.    * Client could update this mask to exclude specific change seq script
     * step from the script exelwtion.
     *
     * @note Change Sequencer assumes client knows what action they are
     * requesting and therefore it will blindly exclude the step based on
     * client's request. It does not perform any dependency tracking.
     */
    LwU32                                    stepIdExclusionMask;

    /*!
     * Request to lock perf change sequence.
     * @ref LW2080_CTRL_PERF_CHANGE_SEQ_LOCK_CLIENT_API
     */
    LwBool                                   bLock;

    /*!
     * [in] Client will set this boolean to LW_TRUE if they want to queue
     * the @ref change request. This is required as we do not want to always
     * queue the @ref change request. Also it could be possible that client
     * just want to force trigger the change with same values to update the
     * HW / SW status.
     */
    LwBool                                   bChangeRequested;

    /*!
     * [out] Latest completed perf change request.
     * [in]  Client's Perf Change request
     * SOON TO BE DEPRECATED IN FAVOR OF LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE       change;

    /*!
     * [out] Latest completed perf change request.
     * [in]  Client's Perf Change request
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT changeInput;
} LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL;
typedef struct LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL *PLW2080_CTRL_PERF_CHANGE_SEQ_CONTROL;

/*!
 * PP-TODO - For backward compatibility
 */
typedef LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL_PARAM;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LW2080_CTRL_CMD_PERF_CHANGE_SEQ_SET_CONTROL_FINN_PARAMS_MESSAGE_ID (0xAEU)

typedef struct LW2080_CTRL_CMD_PERF_CHANGE_SEQ_SET_CONTROL_FINN_PARAMS {
    LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL_PARAM params;
} LW2080_CTRL_CMD_PERF_CHANGE_SEQ_SET_CONTROL_FINN_PARAMS;



/*!
 * LW2080_CTRL_CMD_PERF_CHANGE_SEQ_GET_CONTROL
 *
 * This command accepts client-specified control parameters for perf change
 * sequence and applies these new parameters to the perf change sequence.
 *
 * See @ref LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL for documentation on the
 * parameters.
 *
 * @return LW_OK
 *      RMCTRL control data successfully applied.
 *
 * @return LW_ERR_ILWALID_POINTER
 *      One or more of the pointers was NULL.
 *
 * @return LW_ERR_ILWALID_OBJECT
 *      Was not able to obtain the specified object within the object group.
 *
 * @return LW_ERR_ILWALID_ARGUMENT
 *     User-provided input was incorrect/invalid.
 */
#define LW2080_CTRL_CMD_PERF_CHANGE_SEQ_GET_CONTROL                 (0x208020ad) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xAD" */

/*!
 * This command returns perf change sequence control parameters.
 *
 * See @ref LW2080_CTRL_PERF_CHANGE_SEQ_CONTROL for documentation on the
 * parameters.
 *
 * @return LW_OK
 *      RMCTRL control data successfully applied.
 *
 * @return LW_ERR_ILWALID_POINTER
 *      One or more of the pointers was NULL.
 *
 * @return LW_ERR_ILWALID_OBJECT
 *      Was not able to obtain the specified object within the object group.
 *
 * @return LW_ERR_ILWALID_ARGUMENT
 *     User-provided input was incorrect/invalid.
 */
#define LW2080_CTRL_CMD_PERF_CHANGE_SEQ_SET_CONTROL                 (0x208020ae) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0xAE" */

/*-----------------------------  CHANGES_SEQ --------------------------------*/

/*!
 * LW2080_CTRL_CMD_PERF_SET_OVERCLOCKING_TABLE_INFOROM
 *
 * This command updates the INFOROM's Overclocking table with the frequency for
 * a particular Clock/Vpstate entry (called from SMBPBI)
 *
 *   [in] clockOrVpStateEntry
 *     Specifies if CLOCK / VPSTATE
 *   [in] entryIndex
 *     Entry Index to be modified
 *   [in] maxClockMHz
 *     Clock limit value
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_OVERCLOCKING_TABLE_INFOROM_SET_CONTROL (0x208020b0) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_OVERCLOCKING_TABLE_INFOROM_CONTROL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_OVERCLOCKING_TABLE_INFOROM_CONTROL_PARAMS_MESSAGE_ID (0xB0U)

typedef struct LW2080_CTRL_PERF_OVERCLOCKING_TABLE_INFOROM_CONTROL_PARAMS {
    LwU32 clockOrVpStateEntry;
    LwU32 entryIndex;
    LwU32 maxClockMHz;
} LW2080_CTRL_PERF_OVERCLOCKING_TABLE_INFOROM_CONTROL_PARAMS;

/*------------------------------ PERF_LIMITS --------------------------------*/
/*!
 * Enumeration of @ref PERF_LIMIT_SET_NORMALIZED value types.  Used to determine
 * how to interpret the @ref PERF_LIMIT_SET_NORMALIZED_DATA union.
 */
typedef LwU32 LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE;
typedef LwU32 *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE;

/*!
 * Disabled output type.  This is a placeholder type for an _OUTPUT
 * structure which should never be referenced.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_DISABLED    (0x00U)
/*!
 * P-state index
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_PSTATE_IDX  (0x01U)
/*!
 * Frequency
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_FREQUENCY   (0x02U)
/*!
 * vP-state index
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_VPSTATE_IDX (0x03U)
/*!
 * Voltage values to be compared against domain group values for voltage.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_VOLTAGE     (0x04U)
/*!
 * Number of types.  Must be last.  Used for validation.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_COUNT       (0x05U)


/*!
 * Input data corresponding to LW2080_CTRL_PERF_LIMIT_INPUT_TYPE_PSTATE.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE {
    /*!
     * P-state index
     */
    LwU32                                                      pstateIdx;
    /*!
     * @copydoc LW2080_CTRL_PERF_PERF_LIMIT_INPUT_DATA_PSTATE_POINT
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE_POINT point;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE;

/*!
 * Client input data corresponding to a client frequency input limit.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_FREQ {
    /*!
     * Clock domain index for frequency.
     */
    LwU8  clkDomainIdx;
    /*!
     * Frequency limit (kHz)
     */
    LwU32 freqKHz;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_FREQ;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_FREQ *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_FREQ;

/*!
 * Input data corresponding to @ref
 * LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_VPSTATE_IDX.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VPSTATE {
    /*!
     * vP-state index.
     */
    LwU32 vpstateIdx;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VPSTATE;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VPSTATE *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VPSTATE;

/*!
 * Enumeration of the voltage PERF_LIMIT types.
 */
typedef enum LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE {
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE_UNKNOWN = 0,
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE_LOGICAL = 1,
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE_VFE = 2,
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE_PSTATE = 3,
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE_VPSTATE = 4,
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE_FREQUENCY = 5,
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE_COUNT = 6,
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE;
typedef enum LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE;

/*!
 * Input data for a voltage PERF_LIMIT of sub-type
 * @ref LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_TYPE_LOGICAL.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_LOGICAL {
    /*!
     * Voltage value of the limit in uV.
     */
    LwU32 voltageuV;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_LOGICAL;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_LOGICAL *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_LOGICAL;

/*!
 * Input data for a voltage PERF_LIMIT of sub-type
 * @ref LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_TYPE_VFE.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VFE {
    /*!
     * VFE equation index of the entry that specifies the voltage limit.
     */
    LW2080_CTRL_PERF_VFE_EQU_IDX vfeEquIdx;
    /*!
     * Index into the PMU VFE_EQU_MONITOR which will be evaluating @ref
     * vfeEquIdx.
     *
     * @note This is an record-keeping/debugging value and is referenced
     * only as input, never as input.
     */
    LwU8                         vfeEquMonIdx;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VFE;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VFE *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VFE;

/*!
 * Input data for a voltage PERF_LIMIT of sub-type
 * @ref LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_TYPE_PSTATE.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_PSTATE {
    /*!
     * Pstate index of the entry that specifies the voltage limit.
     */
    LwU8 pstateIdx;

    /*!
     * Frequency type @ref LW2080_CTRL_PERF_VOLT_DOM_INFO_PSTATE_FREQ_TYPE_<xyz>
     */
    LwU8 freqType;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_PSTATE;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_PSTATE *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_PSTATE;

/*!
 * Input data for a voltage PERF_LIMIT of sub-type
 * @ref LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_TYPE_VPSTATE.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VPSTATE {
    /*!
     * Vpstate index of the entry that specifies the voltage limit.
     */
    LwU32 vpstateIdx;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VPSTATE;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VPSTATE *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VPSTATE;

/*!
 * Input data for a voltage PERF_LIMIT of sub-type
 * @ref LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_TYPE_FREQUENCY.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_FREQUENCY {
    /*!
     * Clock domain index for frequency.
     */
    LwU8  clkDomainIdx;

    /*!
     * Frequency in KHz
     */
    LwU32 freqKHz;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_FREQUENCY;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_FREQUENCY *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_FREQUENCY;

/*!
 * Union of type-specific input data.
 */


typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT {
    /*!
     * @ref LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_TYPE_<xyz>
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_TYPE type;

    /*!
     * Union of type-specific data.
     */
    union {
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_LOGICAL   logical;
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VFE       vfe;
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_PSTATE    pstate;
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_VPSTATE   vpstateIdx;
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT_DATA_FREQUENCY frequency;
    } data;

    /*!
     * Voltage delta per element. This is applied along with the global delta
     */
    LwS32 deltauV;

    /*!
     * Evaluated voltage - i.e. the MAX of all ELEMENTs.  This value is output
     * only, useful for debugging APIs.
     */
    LwU32 lwrrVoltageuV;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT;

/*!
 * Structure representing voltage PERF_LIMIT for the specified VOLT_RAIL.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE {
    /*!
     * Index of the VOLT_RAIL for the voltage PERF_LIMIT.
     */
    LwU8                                                          voltRailIdx;

    /*!
     * Number of VOLTAGE_DATA_ELEMENT entries
     */
    LwU8                                                          numElements;

    /*!
     * Voltage delta. This a global delta which is applied along
     * with every element's local delta
     */
    LwS32                                                         deltauV;

    /*!
     * Evaluated voltage - i.e. the MAX of all ELEMENTs.  This value is output
     * only, useful for debugging APIs.
     */
    LwU32                                                         lwrrVoltageuV;

    /*!
     * Voltage PERF_LIMIT type-specific data.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE_ELEMENT elements[LW2080_CTRL_PERF_LIMIT_VOLTAGE_DATA_ELEMENTS_MAX];
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE;

/*!
 * Union of type-specific input data.
 */


/*!
 * Client input data for a PERF_LIMIT.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT {
    /*!
     * LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_<xyz>
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE type;

    /*!
     * Union of type-specific data.
     */
    union {
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_PSTATE  pstate;
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_FREQ    freq;
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VPSTATE vpstate;
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_DATA_VOLTAGE voltage;
    } data;
} LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT *PLW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT;

/*!
 * Helper macro to initialzae a LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT
 * structure. Will initialize the client input to be DISABLED.
 *
 * @param[in,out]  pClientInput
 *     LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT pointer to initialize.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_INIT(pClientInput)         \
do {                                                                        \
    LWMISC_MEMSET((pClientInput), 0x00,                                     \
        LW_SIZEOF32(LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT));             \
    (pClientInput)->type =                                                  \
        LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_DISABLED;             \
} while (LW_FALSE)

/*!
 * Helper macro which indicates whether a given
 * LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT structure is active or not
 * (i.e. LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE != _DISABLED).
 *
 * @param[in]  _pClientInput
 *     LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT pointer
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_ACTIVE(_pClientInput)     \
    (LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE_DISABLED !=             \
        (_pClientInput)->type)

/*!
 * Limit ranges to be used for arbitration.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE {
    /*!
     * Minimum value. May be P-state index, clock frequency, or voltage.
     */
    LwU32 milwalue;
    /*!
     * Maximum value. May be P-state index, clock frequency, or voltage.
     */
    LwU32 maxValue;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE;

/*!
 * Helper macro to set a LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE
 * struct.
 *
 * @param[out]  _pRange
 *     LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE pointer to set.
 * @param[in]   _milwalue
 *     Value to set in @ref LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE::milwalue.
 * @param[in]   _maxValue
 *     Value to set in @ref LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE::maxValue.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE_SET(_pRange, _milwalue, _maxValue) \
    do {                                                                                  \
        (_pRange)->milwalue = (_milwalue);                                                \
        (_pRange)->maxValue = (_maxValue);                                                \
    } while (LW_FALSE)

/*!
 * Helper macro to initialize a LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE
 * struct to the default completely unbound range [@ref LW_U32_MIN, @ref
 * LW_U32_MAX].
 *
 * @param[out]  _pRange
 *     LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE pointer to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE_INIT(_pRange)            \
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE_SET((_pRange),               \
        LW_U32_MIN, LW_U32_MAX);

/*!
 * Enumeration of LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLEs as
 * indexed into various tuple arrays in ARBITRATION_OUTPUT structures.  For
 * example, see @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT::tuples[], @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35::tuples[].
 */
typedef enum LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX {
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_MIN = 0,
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_MAX = 1,
    /*!
     * Must always be last enumeration value!
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_NUM_IDXS = 2,
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX;
typedef enum LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX;

/*!
 * Per-PERF_LIMIT arbitration input data.  This data will used as part of
 * arbitration via @ref perfSetArbitrationTableEntry().
 *
 * PERF_LIMIT 3X-specific data.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X {
    /*!
     * We need an entry for every clock domain, since limits affect domains
     * individually.  This array has the same index as Perf.clkDomain.
     *
     * CRPTODO - Rename to "clkDomains".
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE clocks[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];

    /*!
     * We need an entry for each voltage rail.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE voltRails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];

    /*!
     * And one more entry for the P-states.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE pstate;
} LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X *PLW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X;

/*!
 * Helper macro to initialize a LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X
 * struct.
 *
 * @param[out]   _pArbInput3x
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X_INIT(_pArbInput3x)     \
    do {                                                                        \
        LwU8 _i;                                                                \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE_INIT(                    \
            &((_pArbInput3x)->pstate));                                         \
        for (_i = 0; _i < LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS; ++_i)  \
        {                                                                       \
            LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE_INIT(                \
                &((_pArbInput3x)->clocks[_i]));                                 \
        }                                                                       \
        for (_i = 0; _i < LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS; ++_i)    \
        {                                                                       \
            LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE_INIT(                \
                &((_pArbInput3x)->voltRails[_i]));                              \
        }                                                                       \
    } while (LW_FALSE)

/*!
 * Default arbitration output tuple.  A tuple of pstate index and CLK_DOMAIN
 * frequencies.
 *
 * This struct differs from @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE in the following ways:
 *
 * 1. This struct does not contain any VOLT_RAIL values.  VOLT_RAIL voltages cannot
 * be directly compared.  They are instead a function of the final arbitrated
 * CLK_DOMAIN frequencies.  So, default arbitration doesn't bother with them.
 *
 * 2. This struct doesn't contain any limitIdxs.  There is no concept of
 * PERF_LIMIT priorities before the arbitration algorithm runs.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE {
    /*!
     * PSTATE index for this tuple.
     */
    LwU32 pstateIdx;
    /*!
     * CLK_DOMAIN frequency values (kHz).  Indexes correspond to indexes into
     * CLK_DOMAINS BOARDOBJGRP.  This array has valid indexes corresponding to
     * bits set in @ref
     * LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35::clkDomainsMask.
     */
    LwU32 clkDomains[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
    /*!
     * VOLT_RAIL voltage values (uV).  Indexes correspond to indexes into
     * VOLT_RAILS BOARDOBJGRP.  This array has valid indexes corresponding to
     * bits set in @ref
     * LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35::voltRailsMask.
     */
    LwU32 voltRails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE *PLW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE;

/*!
 * Helper macro to initialize a @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_INPUT_35_TUPLE.
 *
 * @param[out]   _pArbInput35Tuple
 *     Pointer to @ref LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_INPUT_35_TUPLE
 *     to initialize.
 * @param[in]    _value
 *     Value to which all elements of the tuple should be initialized.  Expected
 *     to be either @ref LW_U32_MIN or @ref LW_U32_MAX.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE_INIT(_pArbInput35Tuple, \
            _value)                                                                    \
    do {                                                                               \
        LwU8 _i;                                                                       \
        (_pArbInput35Tuple)->pstateIdx = (_value);                                     \
        for (_i = 0; _i < LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS; _i++)         \
            { (_pArbInput35Tuple)->clkDomains[_i] = (_value); }                        \
        for (_i = 0; _i < LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS; _i++)           \
            { (_pArbInput35Tuple)->voltRails[_i] = (_value); }                         \
    } while (LW_FALSE)

/*!
 * Represents a value in an ARBITRATION_OUTPUT tuple.  This structure includes a
 * unitless value (units are implied by the context of this structure within the
 * TUPLE).  It also includes the LW2080_CTRL_PERF_PERF_LIMIT_ID which
 * arbitration determined bound to this value with its input.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE {
    /*!
     * Unitless value.
     */
    LwU32                          value;
    /*!
     * @ref LW2080_CTRL_PERF_PERF_LIMIT_ID which arbitration determined bound to
     * this value with its input.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_ID limitIdx;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE;

/*!
 * PERF_LIMIT_35 arbitration input data.  Populated from @ref
 * LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT via @ref
 * _perfLimit35ClientInputToArbInput() and then applied to @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35 via @ref
 * _perfLimit35ArbInputToArbOutput.
 *
 * Extends @ref LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_3X as a
 * super-class.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35 {
    /*!
     * Mask of CLK_DOMAINs in @ref super which have been set via @ref
     * _perfLimit35ClientInputToArbInput().  This book-keeping mask is used for
     * determining which CLK_DOMAIN values still need to be determined.
     *
     * As a sanity check, @ref _perfLimit35ClientInputToArbInput() will confirm
     * that this mask is equal to PERF_LIMITS_ARBITRATION_OUTPUT::clkDomainsMask
     * after all CLIENT_COLWERSION is complete - i.e. that code has determined a
     * complete tuple of PSTATE and CLK_DOMAIN values which can be applied to
     * PERF_LIMITS_ARBITRATION_OUTPUT.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                            clkDomainsMask;
    /*!
     * Mask of VOLT_RAILS in @ref super which have been set via @ref
     * _perfLimit35ClientInputToArbInput().
     *
     * Most PERF_LIMIT_35 objects won't specify any PERF_LIMIT_ARBITRATION_INPUT
     * values for VOLT_RAILS.  Most VOLT_RAIL values are determined from the
     * (PSTATE, CLK_DOMAINS) tuple values in PERF_LIMITS_ARBITRATION_OUTPUT.
     * The exception is PERF_LIMIT_35 objects with
     * LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT_TYPE = _VOLTAGE, which want to
     * bound VOLT_RAILs.  This masks tracks when that data is specified in the
     * PERF_LIMIT_ARBITRATION_INPUT and must be applied to
     * PERF_LIMITS_ARBITRATION_OUTPUT.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                            voltRailsMask;
    /*!
     * Array of LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_INPUT_35_TUPLE
     * structures.  Indexed per @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX - i.e. 0 =
     * _MIN, 1 = _MAX.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE      tuples[LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_NUM_IDXS];
    /*!
     * Array used to store the maximum loose voltage limit applied to this
     * particular arbitration input.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE voltageMaxLooseuV[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35 *PLW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35;

/*!
 * Helper macro to initialize a LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35
 * struct.
 *
 * @param[out]   _pArbInput35
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35 to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_INIT(_pArbInput35)      \
    do {                                                                         \
        LwU8 _i;                                                                 \
        LW2080_CTRL_BOARDOBJGRP_MASK_E32_INIT(                                   \
            &((_pArbInput35)->clkDomainsMask.super));                            \
        LW2080_CTRL_BOARDOBJGRP_MASK_E32_INIT(                                   \
            &((_pArbInput35)->voltRailsMask.super));                             \
        LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE_INIT(             \
            &((_pArbInput35)->tuples[                                            \
                LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_MIN]), \
            LW_U32_MIN);                                                         \
        LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35_TUPLE_INIT(             \
            &((_pArbInput35)->tuples[                                            \
                LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_MAX]), \
            LW_U32_MAX);                                                         \
        for (_i = 0; _i <LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS; _i++)      \
        {                                                                        \
            (_pArbInput35)->voltageMaxLooseuV[_i].limitIdx =                     \
                LW2080_CTRL_PERF_PERF_LIMIT_ID_NOT_SET;                          \
            (_pArbInput35)->voltageMaxLooseuV[_i].value = LW_U32_MAX;            \
        }                                                                        \
    } while (LW_FALSE)

/*!
 * Default arbitration output tuple.  A tuple of pstate index and CLK_DOMAIN
 * frequencies.
 *
 * This struct differs from @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE in the following ways:
 *
 * 1. This struct does not contain any VOLT_RAIL values.  VOLT_RAIL voltages cannot
 * be directly compared.  They are instead a function of the final arbitrated
 * CLK_DOMAIN frequencies.  So, default arbitration doesn't bother with them.
 *
 * 2. This struct doesn't contain any limitIdxs.  There is no concept of
 * PERF_LIMIT priorities before the arbitration algorithm runs.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT_TUPLE {
    /*!
     * PSTATE index for this tuple.
     */
    LwU32 pstateIdx;
    /*!
     * CLK_DOMAIN frequency values (kHz).  Indexes correspond to indexes into
     * CLK_DOMAINS BOARDOBJGRP.  This array has valid indexes corresponding to
     * bits set in @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT::clkDomainsMask.
     */
    LwU32 clkDomains[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT_TUPLE;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT_TUPLE *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT_TUPLE;


/*!
 * Default arbitration output struct.  The "default" arbitration output is the
 * most loose min/max bounds on PSTATE and CLK_DOMAINs within which PERF_LIMITs
 * can bound even further.
 *
 * On each run of the arbitration algorithm, these values are copied into the
 * ARBITRATION_OUTPUT struct's respective min and max tuples and then
 * arbitration algorithm will move those tuples tighter per the input specified
 * in the individual PERF_LIMITs.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT {
    /*!
     * Mask of indexes of CLK_DOMAINs which have frequency settings specified in
     * this struct.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                              clkDomainsMask;
    /*!
     * Array of LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT_TUPLE
     * structures.  Indexed per @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX - i.e. 0 =
     * _MIN, 1 = _MAX.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT_TUPLE tuples[LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_NUM_IDXS];
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT;

/*!
 * INIT value for @ref PERF_LIMITS::arbSeqId.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARB_SEQ_ID_INIT    0

/*!
 * INVALID value for arbSeqId, indicating that the data stored within
 * the @ref LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT is invalid and must
 * be generated via @ref perfLimitsArbitrate().
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARB_SEQ_ID_ILWALID LW_U32_MAX

/*!
 * Helper macro to increment @ref PERF_LIMITS::arbSeqId.  This helper
 * function can be reused within both the RM and PMU PERF_LIMITS implementations
 * whenever state upon which @ref perfLimitsArbitrate() depends (PERF_LIMIT
 * values, VF, (V)PSTATE, etc.) changes.
 *
 * Takes care to avoid incrementing to INVALID value, such that an INVALID @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT::arbSeqId will never
 * match with @ref PERF_LIMITS::arbSeqId.
 *
 * @param[in/out] pSeqId - Pointer to the arbSeqId to increment.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARB_SEQ_ID_INCREMENT(pSeqId)                \
    do {                                                                         \
        (*pSeqId) =                                                              \
            ((*pSeqId) == LW2080_CTRL_PERF_PERF_LIMITS_ARB_SEQ_ID_ILWALID - 1U) ?\
                0U : (*pSeqId) + 1U;                                             \
    } while (LW_FALSE)

#define LW2080_CTRL_PERF_PERF_CHANGE_SEQ_SEQ_ID_INCREMENT(pSeqId)                \
            LW2080_CTRL_PERF_PERF_LIMITS_ARB_SEQ_ID_INCREMENT(pSeqId)

/*!
 * Structure representing ARBIRATION_OUTPUT cache state.  This contains all the
 * necessary state data from when ARBIRATION_OUTPUT was last generated/cached
 * and is compared against the current state to determine whether the cached
 * data can be reused or the arbitration algorithm needs to repopulate the
 * structure.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA {
    /*!
     * @ref perfLimitsArbitrate() sequence ID when this ARBITRATION_OUTPUT was
     * generated.  This value can be compared against the
     * PERF_LIMITS::arbSeqId to confirm whether the data is still valid
     * or whether its possibly stale due to underlying state differences.
     */
    LwU32                             arbSeqId;
    /*!
     * Cached mask of excluded PERF_LIMITS which were used to generate this
     * ARBITRATION_OUTPUT_2X.  The last pLimitMaskExclude passed to @ref
     * perfLimitsArbitrate().  Used to determine whether this
     * ARBITRATION_OUTPUT_2X is stale and arbitration algorithm needs to be
     * re-run.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskExclude;
    /*!
     * Boolean indicating whether the ARBIRATION_OUTPUT is dirty.  If LW_TRUE,
     * the arbitration algorithm decided to re-run and has populated the
     * ARBITRATION_OUTPUT with new values.  If LW_FALSE, the arbitration
     * algorithm decided that the current cached ARBITRATION_OUTPUT is still
     * valid and did not populate it.
     */
    LwBool                            bDirty;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA;

/*!
 * Helper macro to init a
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA struct.
 *
 * @param[in] _pCacheData
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA
 *     struct to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA_INIT(_pCacheData) \
    do {                                                                             \
        LWMISC_MEMSET((_pCacheData), 0x0,                                            \
            LW_SIZEOF32(LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA));\
        (_pCacheData)->arbSeqId = LW2080_CTRL_PERF_PERF_LIMITS_ARB_SEQ_ID_ILWALID;   \
    } while (LW_FALSE)

/*!
 * Helper macro to set the variables within a
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE.
 *
 * @param[out]  _pTupleValue
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE to set.
 * @param[in]   _value
 *     value to set.
 * @param[in]   _limtiIdx
 *     limitIdx to set.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE_SET(_pTupleValue, _value, _limitIdx) \
do {                                                                                                     \
    (_pTupleValue)->value    = (_value);                                                                 \
    (_pTupleValue)->limitIdx = (_limitIdx);                                                              \
} while (LW_FALSE)

/*!
 * Helper macro to set the variables within a
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE based on a MAX()
 * comparison of @ref _value and @ref _pTupleValue->value.
 *
 * @param[out]  _pTupleValue
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE to set.
 * @param[in]   _value
 *     value to set if greater than @ref _pTupleValue->max
 * @param[in]   _limtiIdx
 *     limitIdx to set.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE_MAX(_pTupleValue, _value, _limitIdx) \
do {                                                                                                     \
    if ((_value) > (_pTupleValue)->value)                                                                \
    {                                                                                                    \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE_SET(                                 \
            (_pTupleValue), (_value), (_limitIdx));                                                      \
    }                                                                                                    \
} while (LW_FALSE)

/*!
 * Helper macro to set the variables within a
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE based on a MIN()
 * comparison of @ref _value and @ref _pTupleValue->value.
 *
 * @param[out]  _pTupleValue
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE to set.
 * @param[in]   _value
 *     value to set if less than @ref _pTupleValue->max
 * @param[in]   _limtiIdx
 *     limitIdx to set.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE_MIN(_pTupleValue, _value, _limitIdx) \
do {                                                                                                     \
    if ((_value) < (_pTupleValue)->value)                                                                \
    {                                                                                                    \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE_SET(                                 \
            (_pTupleValue), (_value), (_limitIdx));                                                      \
    }                                                                                                    \
} while (LW_FALSE)

/*!
 * Arbitration output tuple values for a given CLK_DOMAIN.
 *
 * The CLK_DOMAIN index/ID is determined by context of usage within
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN {
    /*!
     * The CLK_DOMAIN's arbitrated output frequency.  The frequency value is in
     * units of kHz.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE freqkHz;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN;

/*!
 * Arbitration output tuple values for a given VOLT_RAIL.
 *
 * The VOLT_RAIL index/ID is determined by context of usage within
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL {
    /*!
     * The VOLT_RAIL's arbitrated output voltage.  The voltage value is in units
     * of uV.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE voltageuV;
    /*!
     * The VOLT_RAIL's arbitrated output noise-unaware minimum voltage.  The
     * voltage value is in units of uV.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE voltageNoiseUnawareMinuV;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL;

/*!
 * A tuple of pstate index, CLK_DOMAIN frequencies, and VOLTAGE_RAIL voltages as
 * specified together a single operating point per the output of the PERF_LIMITS
 * arbitration routine.
 *
 * This structure contains some limited debugging information to explain how the
 * PERF_LIMITs bound to these values.
 *
 * This structure contains caching data which represents the conditions under
 * which this tuple was generated, which can later be used the PERF_LIMITS
 * arbitration routine may use to decide that the current tuple won't change and
 * arbitration can be skipped.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE {
    /*!
     * ARBITATION_OUTPUT caching data.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA       cacheData;
    /*!
     * Arbitrated PSTATE output.   Current usecase use units of PSTATE_LEVEL.
     *
     * CRPTODO - Switch over to pstateIdx.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE      pstateIdx;
    /*!
     * Arbitrated CLK_DOMAINS output.  Indexed by the indexes of the CLK_DOMAINS
     * BOARDOBJGRP.
     *
     * Has valid indexes corresponding to @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT::clkDomainsMask.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN clkDomains[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
    /*!
     * Arbitrated VOLT_RAILS output.  Indexed by the indexes of the VOLT_RAILS
     * BOARDOBJGRP.
     *
     * Has valid indexes corresponding to @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT::voltRailMask.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL  voltRails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE;

/*!
 * Helper macro to initialize a LW2080_CTRL_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE
 * structure.
 *
 * @param[in] _pTuple
 *     Pointer to LW2080_CTRL_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_INIT(_pTuple) \
    do {                                                                    \
        LWMISC_MEMSET((_pTuple), 0x0,                                       \
            LW_SIZEOF32(                                                    \
                LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE));    \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA_INIT(    \
            &((_pTuple)->cacheData));                                       \
    } while (LW_FALSE)

/*!
 * Generic PERF_LIMITS arbitration output structure.
 *
 * Output of the arbitration routine (@ref perfLimitsArbitrate())
 * which contains the min and max tuples of arbitrated/target clocks and
 * voltages for the GPU.  This structure also contains various deubgging
 * information about the PERF_LIMITs which determined that output tuple.
 *
 * This structure is completely PERF_LIMITS-type/version-independent, it
 * contains only generic arbitration output which is expected from any/all
 * versions of PERF_LIMITS.  Sub-classes may implement/extend this structure for
 * their type-specific information.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT {
    /*!
     * PERF_LIMITS version which populated this ARBITRATION_OUTPUT struct.  Used
     * for dynamic casting of this generic ARBITRATION_OUTPUT structure to any
     * type-specific ARBITRATION_OUTPUT structure which might extend it.
     *
     * Values specified in @ref LW2080_CTRL_PERF_PERF_LIMITS_VERSION_<XYZ>.
     */
    LwU8                                                  version;
    /*!
     * Mask of arbitrated CLK_DOMAINs, corresponding to indexes in the
     * CLK_DOMAINS BOARDOBJGRP.
     *
     * Will be a subset of the overall CLK_DOMAINS BOARDOBJGRP.  Valid indexes
     * may be omitted if the arbiter decides they should not be arbitrated
     * (e.g. FIXED domains are not arbitrated).
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                      clkDomainsMask;
    /*!
     * Mask of arbitrated VOLT_RAILs, corresponding to indexes in the
     * VOLT_RAILS BOARDOBJGRP.
     *
     * Will be a subset of the overall VOLT_RAILS BOARDOBJGRP.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                      voltRailsMask;
    /*!
     * Arbitrated result tuple.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE tuple;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT;

/*!
 * Helper macro to initialize a LW2080_CTRL_PERF_LIMITS_ARBITRATION_OUTPUT structure.
 *
 * @param[in] _pArbOutput  Pointer to LW2080_CTRL_PERF_LIMITS_ARBITRATION_OUTPUT to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_INIT(_pArbOutput)  \
    do {                                                                   \
        LWMISC_MEMSET((_pArbOutput), 0x0,                                  \
            LW_SIZEOF32(LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT)); \
        (_pArbOutput)->version = LW2080_CTRL_PERF_PERF_LIMITS_VER_UNKNOWN; \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_INIT(        \
            &((_pArbOutput)->tuple));                                      \
    } while (LW_FALSE)


/*!
 * Domain arbitration output.  This structure contains output arbitrated/target
 * value for a domain (pstate, clock, voltage) as well as various debugging
 * state to determine exactly how the target value was determined.
 *
 * This structure is the SUPER structure for all arbitration output.  It is
 * extended by various sub-classes which may contain extra data.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY {
    /*!
     * Clock or voltage domain index.
     */
    LwU8                                           domainIdx;
    /*!
     * Allowed range, including utilization limits
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE range;
    /*!
     * Range of noise-unaware values - corresponding to the min and max clock
     * tuples.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_RANGE rangeNoiseUnaware;
    /*!
     * Minimum value without utilization limits.
     */
    LwU32                                          milwalueNoUtil;
    /*!
     * Priority of the limit that set milwalue.
     */
    LwU8                                           minPriority;
    /*!
     * Priority of the limit that set milwalueNoUtil.
     */
    LwU8                                           minPriorityNoUtil;
    /*!
     * Priority of the limit that set @ref rangeNoiseUnaware::milwalue.
     */
    LwU8                                           minPriorityNoiseUnaware;
    /*!
     * Priority of the limit that set maxValue.
     */
    LwU8                                           maxPriority;
    /*!
     * Priority of the limit that set @ref rangeNoiseUnaware::maxValue.
     */
    LwU8                                           maxPriorityNoiseUnaware;
    /*!
     * Target clock frequency, Pstate level, or voltage to set.
     */
    LwU32                                          lwrValueToSet;
} LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY *PLW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY;

/*!
 * Voltage Domain Arbitration Output.  This structure extends @ref
 * PERF_ARBITRATION_ENTRY_SUPER to track the output of arbitration on a voltage
 * domain.  In addition to the _SUPER data, it includes the allowed range of
 * values voltage values (both v_{target} and v_{min, noise-unaware}) and the
 * v_{min, noise-unaware}.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT {
    /*!
     * SUPER structure.  Common to all LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY types.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY super;
    /*!
     * The current V_{min, noise-unaware} value.  This value is chosen from the
     * either the min or max noise-unaware voltage values, per the specified
     * semantics.
     */
    LwU32                                         lwrVoltageMinNoiseUnawareuV;
    /*!
     * A loose voltage min/floor to be applied to both min and max voltage
     * values without affecting the target clocks.
     * @ref LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT::super.range
     * contains the minimum voltages for the current tuple of arbitrated clocks.
     * It does not actually represent any type of direct bounding on the voltages.
     * However, loose voltage limits (@ref PERF_LIMIT::clkDomainStrictPropagationMask ==
     * 0) wish to actually bound the voltage.
     *
     * A loose voltage maximum can be applied once to the @ref
     * LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT::super.range and then
     * be discarded.  Any subsequent higher minimum voltages (per a higher-
     * priority PERF_LIMIT) will be taken simply by priority - higher priority
     * min above the lower priority max.  However, a lower voltage should not
     * override the min due to max of mins - i.e. a higher priority min below a
     * lower priority min.  In this case, we need to bound/floor subsequent
     * voltage requests to the loose voltage min.
     */
    LwU32                                         milwalueLoose;
    /*!
     * PERF_LIMIT ID of the last PERF_LIMIT to set the loose voltage min.
     */
    LwU8                                          minPriorityLoose;
} LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT *PLW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT;

/*!
 * PERF_LIMITS_2X arbitration output structure.
 *
 * Output of the arbitration routine (@ref perfLimits2xArbitrate())
 * which contains the min and max tuples of arbitrated/target clocks and
 * voltages for the GPU.  This structure also contains various deubgging
 * information about the PERF_LIMITs which determined that output tuple.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_2X {
    /*!
     * ARBITRATION_OUTPUT caching data.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA cacheData;
    /*!
     * Used to check if VF data is stale.
     */
    LwU32                                                      vfPointsCacheCounter;
    /*!
     * Entries for each clock domain. Limits affect domains individually.
     *
     * TODO: Colwert LW2080_CTRL_PERF_PERF_LIMIT_ARBITRAATION_ENTRY to a
     * BOARDOBJGRP and iterate over each element in the group.
     */
    LwU32                                                      nNumClockEntries;
    LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY              clocks[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
    /*!
     * Entries for each voltage rail.
     *
     * TODO: Colwert LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT to a
     * BOARDOBJGRP and iterate over each element in the group.
     */
    LwU32                                                      nNumVoltRailEntries;
    LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY_VOLT         voltRails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
    /*!
     * P-state entry.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_ENTRY              pstate;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_2X;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_2X *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_2X;

/*!
 * Helper macro to initialize a LW2080_CTRL_PERF_LIMITS_ARBITRATION_OUTPUT_2X
 * structure.
 *
 * @param[in] pArbOutput2x
 *     Pointer to LW2080_CTRL_PERF_LIMITS_ARBITRATION_OUTPUT_2X to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_2X_INIT(pArbOutput2x)  \
    do {                                                                       \
        LWMISC_MEMSET((pArbOutput2x), 0x0,                                     \
            LW_SIZEOF32(LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_2X));  \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_CACHE_DATA_INIT(       \
            &((pArbOutput2x)->cacheData));                                     \
    } while (LW_FALSE)

/*!
 * PERF_LIMIT_35 algorithm-specific CLK_DOMAIN arbitration output tuple.  This
 * structure extends
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN to provide
 * generic PERF_LIMIT arbitration output tuple data, as well as
 * PERF_LIMIT_35-specific data and debugging state.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN_35 {
    /*!
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN super
     * structure.  Must always be first element in structure.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN super;

    /*!
     * Minimum noise-unaware frequency (kHz) for this CLK_DOMAIN.  For a
     * noise-unaware CLK_DOMAIN, this should always be equal to the @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN::freqkHz.
     * For a noise-aware CLK_DOMAIN, this value is set by any noise-intolerant
     * PERF_LIMITs - i.e. PERF_LIMITs which are setting limits which most be
     * guaranteed (usually due to QoS requirements) even for noise.
     *
     * The arbitration algorithm uses this value to generate the @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE::voltageNoiseUnawareMinuv.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE      freqNoiseUnawareMinkHz;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN_35 *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN_35;

/*!
 * PERF_LIMIT_35 algorithm-specific VOLT_RAIL arbitration output tuple.  This
 * structure extends
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL to provide
 * generic PERF_LIMIT arbitration output tuple data, as well as
 * PERF_LIMIT_35-specific data and debugging state.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL_35 {
    /*!
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL super
     * structure.  Must always be first element in structure.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL super;
    /*!
     * Minimum loose voltage (uV) on this VOLT_RAIL.  Loose voltage limits bound
     * the VOLT_RAIL voltage without any impact on any CLK_DOMAIN frequencies.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE     voltageMinLooseuV;
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL_35 *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL_35;

/*!
 * PERF_LIMITS_35 arbitrated output tuple.  This is a tuple of PSTATE,
 * CLK_DOMAIN, and VOLT_RAIL values.  These values include both final arbitrated
 * output values (e.g. PSTATE index, CLK_DOMAIN frequencies, VOLT_RAIL voltages)
 * but also various intermediate book-keeping and debugging data.
 *
 * This structure is a superset of @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE, but intentionally
 * doesn't extend it as a "super" so that all of the extra data can be
 * referenced within the same structure.  Instead, ilwidivdual CLK_DOMAIN and
 * VOLT_RAIL _35 structures extend the common structs as super.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35 {
    /*!
     * Arbitrated PSTATE index output.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE         pstateIdx;
    /*!
     * Arbitrated CLK_DOMAINS output.  Indexed by the indexes of the CLK_DOMAINS
     * BOARDOBJGRP.
     *
     * Has valid indexes corresponding to @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT::clkDomainsMask.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN_35 clkDomains[LW2080_CTRL_CLK_CLK_DOMAIN_CLIENT_MAX_DOMAINS];
    /*!
     * Arbitrated VOLT_RAILS output.  Indexed by the indexes of the VOLT_RAILS
     * BOARDOBJGRP.
     *
     * Has valid indexes corresponding to @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT::voltRailMask.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VOLT_RAIL_35  voltRails[LW2080_CTRL_VOLT_VOLT_RAIL_CLIENT_MAX_RAILS];
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35 *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35;

/*!
 * Helper macro to init a @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35 struct.
 *
 * @param[in] _pArbOutputTuple35
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35
 *     struct to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35_INIT(_pArbOutputTuple35) \
    do {                                                                                  \
        LWMISC_MEMSET((_pArbOutputTuple35), 0x0,                                          \
            LW_SIZEOF32(LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35));       \
    } while (LW_FALSE)

/*!
 * PERF_LIMITS_35 arbitration output structure.
 *
 * This structure extends @ref LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT
 * to provide both generic PERF_LIMITS-type/version-independent output, as well
 * as PERF_LIMITS_35-specific arbitration algorithm state and debugging data.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35 {
    /*!
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT super data.  Must always
     * be first element in struct.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT          super;
    /*!
     * Array of LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35
     * structures.  Indexed per @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX - i.e. 0 =
     * _MIN, 1 = _MAX.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35 tuples[LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_IDX_NUM_IDXS];
} LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35 *PLW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35;

/*!
 * Helper macro to init a @ref
 * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35 struct.
 *
 * @param[in] _pArbOutputTuple35
 *     Pointer to LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35
 *     struct to init.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35_INIT(_pArbOutput35)          \
    do {                                                                                \
        LwU8 _i;                                                                        \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_INIT(                           \
            &((_pArbOutput35)->super));                                                 \
        for (_i = 0;                                                                    \
            _i < LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35_TUPLE_IDX_NUM_IDXS; \
            _i++)                                                                       \
        {                                                                               \
            LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_35_INIT(              \
                &((_pArbOutput35)->tuples[i]));                                         \
        }                                                                               \
    } while (LW_FALSE)

/*!
 * Contains the last set of GPU Boost Sync limits sent to the RM from the PMU.
 * Only sends the P-state index and GPCCLK values, as that all GPU Boost Sync
 * cares about. Add more clocks as needed.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_GPU_BOOST_SYNC {
    /*!
     * Specifies if GPU Boost Sync is enabled or not.
     */
    LwBool                                                           bEnabled;

    /*!
     * Specifies if synchronization is required. This happens when toggling
     * the enable flag.
     */
    LwBool                                                           bForce;

    /*!
     * GPU Boost Sync limit for P-state index.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE      pstateIdx;

    /*!
     * GPU Boost Sync limit for GPCCLK.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_CLK_DOMAIN gpcclkkHz;
} LW2080_CTRL_PERF_PERF_LIMITS_GPU_BOOST_SYNC;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_GPU_BOOST_SYNC *PLW2080_CTRL_PERF_PERF_LIMITS_GPU_BOOST_SYNC;

/*!
 * Helper macro to initialize a LW2080_CTRL_PERF_PERF_LIMITS_GPU_BOOST_SYNC
 * structure.
 *
 * @param[out]  pGpuBoostSync  LW2080_CTRL_PERF_PERF_LIMITS_GPU_BOOST_SYNC to
 *                             initialize.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_STATUS_PMU_INIT(pGpuBoostSync)         \
    do {                                                                    \
        (pGpuBoostSync)->bEnabled = LW_FALSE;                               \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE_SET(    \
            &((pGpuBoostSync)->pstateIdx), LW_U32_MAX,                      \
            LW2080_CTRL_PERF_PERF_LIMIT_ID_NOT_SET);                        \
        LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_TUPLE_VALUE_SET(    \
            &((pGpuBoostSync)->gpcclkkHz.freqkHz), LW_U32_MAX,              \
            LW2080_CTRL_PERF_PERF_LIMIT_ID_NOT_SET);                        \
    } while (LW_FALSE)

/*!
 * LW2080_CTRL_PERF_PERF_LIMITS_VERSION
 *
 * Represents which PERF_LIMITS version is using.
 * _2X    - Legacy implementation of PERF_LIMITS used in P-states 3.1 and earlier.
 * _35    - PERF_LIMITS implementation used with P-states 3.5 and later.
 * _35_10 - PERF_LIMITS implementation used with P-states 3.5 version 1.0 (TURING and GA100 Only).
 * _40    - PERF_LIMITS implementation used with P-states 4.0 (GA10x and later).
 * _PMU   - Virtual class represent any PERF_LIMITS version which is implemented
 * on the PMU, not the RM.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_VERSION_SUPER 0x0
#define LW2080_CTRL_PERF_PERF_LIMITS_VERSION_2X    0x1
#define LW2080_CTRL_PERF_PERF_LIMITS_VERSION_35    0x2
#define LW2080_CTRL_PERF_PERF_LIMITS_VERSION_35_10 0x3
#define LW2080_CTRL_PERF_PERF_LIMITS_VERSION_40    0x5
#define LW2080_CTRL_PERF_PERF_LIMITS_VERSION_PMU   0x6
#define LW2080_CTRL_PERF_PERF_LIMITS_VER_UNKNOWN   0xFF

/*!
 * Enumeration of PERF_LIMIT class types.
 */
#define LW2080_CTRL_PERF_PERF_LIMIT_TYPE_2X        0x00
#define LW2080_CTRL_PERF_PERF_LIMIT_TYPE_35        0x01
#define LW2080_CTRL_PERF_PERF_LIMIT_TYPE_PMU       0x02
#define LW2080_CTRL_PERF_PERF_LIMIT_TYPE_35_10     0x03
#define LW2080_CTRL_PERF_PERF_LIMIT_TYPE_40        0x04
#define LW2080_CTRL_PERF_PERF_LIMIT_TYPE_UNKNOWN   0xFF

/*!
 * Structure describing PERF_LIMITS_PMU static information.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_INFO_PMU {
    /*!
     * Reserved for future use.
     */
    LwU32 rsvd;
} LW2080_CTRL_PERF_PERF_LIMIT_INFO_PMU;

/*!
 * Structure describing PERF_LIMITS_35 static information.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_INFO_35 {
    /*!
     * Super class. Must always be first.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_INFO_PMU super;

    /*!
     * Boolean indicating whether STRICT CLK_DOMAIN propgation should be
     * constrained to the ARBITRATION_INPUT PSTATE or not.
     */
    LwBool                               bStrictPropagationPstateConstrained;

    /*!
     * Boolean indicating whether V_{min, noise-unaware} needs to be updated
     * based on V_{min} value determined using the clock domain strcit
     * propagation mask.
     */
    LwBool                               bForceNoiseUnawareStrictPropgation;

    // PP-TODO : Remove these duplicates after updating LWAPIs.

    /*!
     * Mask for clock domains to strictly limit. One bit for each clkDomainIdx.
     *
     * When the bit for a domain is not set, arbitration will propagate as
     * loosely as possible. For example, a clock will only be adjusted when
     * it is outside of the permissible range for the P-state range.
     *
     * When the bit for a domain is set, arbitration will find matching
     * frequencies across domains using ratios or V/F intersects.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32     clkDomainStrictPropagationMask;
    /*!
     * Mask of clock domains (mask of indexes into the CLK_DOMAINs BOARDOBJGRP)
     * whose V_{min} values should be included in V_{min, noise-unaware},
     * regardless of their programming/sourcing mode, when determining the
     * voltage requirements for this PERF_LIMIT.
     *
     * This mask is used to indicate that even when the clock is
     * programmed/sourced by a noise-aware frequency generator (e.g. NAFLL), the
     * V_{min}s per the CLK_DOMAIN's VF lwrve should be respected.  This is used
     * by PERF_LIMITs which have strict QoS requirements on noise-aware
     * CLK_DOMAINs - e.g. IMP requirements on SYSCLK, XBARCLK, LTCCLK.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32     clkDomainsMaskForceVminNoiseUnaware;
} LW2080_CTRL_PERF_PERF_LIMIT_INFO_35;

/*!
 * Structure describing PERF_LIMITS_35_10 static information.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_INFO_35_10 {
    /*!
     * Super class. Must always be first.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_INFO_35 super;

    /*!
     * Mask for clock domains to strictly limit. One bit for each clkDomainIdx.
     *
     * When the bit for a domain is not set, arbitration will propagate as
     * loosely as possible. For example, a clock will only be adjusted when
     * it is outside of the permissible range for the P-state range.
     *
     * When the bit for a domain is set, arbitration will find matching
     * frequencies across domains using ratios or V/F intersects.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32    clkDomainStrictPropagationMask;
} LW2080_CTRL_PERF_PERF_LIMIT_INFO_35_10;

/*!
 * Structure describing PERF_LIMITS_40 static information.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_INFO_40 {
    /*!
     * Super class. Must always be first.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_INFO_35 super;
} LW2080_CTRL_PERF_PERF_LIMIT_INFO_40;

/*!
 * PERF_LIMITS type-specific data union. Discriminated by PERF_LIMIT::super.type.
 */


/*!
 * Structure representing the static information of the parameters associated
 * with PERF_LIMITS. Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ          super;
    /*!
     * @ref LW2080_CTRL_PERF_LIMIT_INFO_FLAGS_<xyz>
     */
    LwU8                          flags;
    /*!
     * String with user-friendly name of the limit.
     */
    char                          szName[LW2080_CTRL_PERF_LIMIT_NAME_MAX_LENGTH];
    /*!
     * Propagation regime to be used for propagation of settings to
     * CLK_DOMAINs.
     */
    LwU8                          propagationRegime;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                          type;
    /*!
     * The perf policy ID this limit maps to.
     */
    LW2080_CTRL_PERF_POLICY_SW_ID perfPolicyId;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_LIMIT_INFO_PMU   pmu;
        LW2080_CTRL_PERF_PERF_LIMIT_INFO_35    v35;
        LW2080_CTRL_PERF_PERF_LIMIT_INFO_35_10 v35V10;
        LW2080_CTRL_PERF_PERF_LIMIT_INFO_40    v40;
    } data;
} LW2080_CTRL_PERF_PERF_LIMIT_INFO;

/*!
 * Structure representing the static information of the parameters associated
 * with PERF_LIMITS. Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_INFO_MESSAGE_ID (0x22U)

typedef struct LW2080_CTRL_PERF_PERF_LIMITS_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255      super;
    /*!
     * PERF_LIMITS version. @ref LW2080_CTRL_PERF_PERF_LIMITS_VERSION
     */
    LwU8                              version;
    /*!
     * Suspend masks used by the arbiter.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskNone;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskAll;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskOsLevel;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskForced;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskNotForced;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskHardcaps;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskNotHardcaps;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskGpuBoost;
    LW2080_CTRL_BOARDOBJGRP_MASK_E255 limitMaskNotGpuBoost;
    /*!
     * Array of LW2080_CTRL_PERF_PERF_LIMIT_INFO structures. Has valid indexes
     * corresponding to the bits in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_INFO  limits[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_LIMITS_INFO;

#define LW2080_CTRL_CMD_PERF_PERF_LIMITS_GET_INFO (0x20802022) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_LIMITS_INFO_MESSAGE_ID" */

/*!
 * Structure describing PERF_LIMIT_2X dynamic status.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS_2X {
    /*!
     * Input values specified to perfSetLimit() by caller.
     *
     * @note These are legacy structures used by old APIs.
     */
    LW2080_CTRL_PERF_LIMIT_INPUT  input;
    /*!
     * Output values as determined by arbitration algorithm.
     *
     * @note These are legacy structures used by old APIs.
     */
    LW2080_CTRL_PERF_LIMIT_OUTPUT output;
} LW2080_CTRL_PERF_PERF_LIMIT_STATUS_2X;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS_2X *PLW2080_CTRL_PERF_PERF_LIMIT_STATUS_2X;

/*!
 * Structure describing PERF_LIMIT_PMU dynamic status.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS_PMU {
    /*!
     * Client-based input for the limit.
     *
     * @note CRPTODO: This will be removed in a follow-up CL.  See @ref
     * LW2080_CTRL_PERF_PERF_LIMIT_STATUS::clientInput.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT clientInput;
} LW2080_CTRL_PERF_PERF_LIMIT_STATUS_PMU;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS_PMU *PLW2080_CTRL_PERF_PERF_LIMIT_STATUS_PMU;

/*!
 * Structure describing PERF_LIMIT_35 dynamic status.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS_35 {
    /*!
     * Super class. Must always be first.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_STATUS_PMU           super;
    /*!
     * Arbitration-based input for the limit.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_ARBITRATION_INPUT_35 arbInput;
} LW2080_CTRL_PERF_PERF_LIMIT_STATUS_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS_35 *PLW2080_CTRL_PERF_PERF_LIMIT_STATUS_35;

/*!
 * PERF_LIMIT type-specific data union. Discriminated by PERF_LIMIT::super.type.
 */


/*!
 * Structure representing the dynamic status of the parameters associated with
 * PERF_LIMITS. Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ                     super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                                     type;
    /*!
     * Client-based input for the limit.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT clientInput;

    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_LIMIT_STATUS_2X  v2x;
        LW2080_CTRL_PERF_PERF_LIMIT_STATUS_PMU pmu;
        LW2080_CTRL_PERF_PERF_LIMIT_STATUS_35  v35;
    } data;
} LW2080_CTRL_PERF_PERF_LIMIT_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_STATUS *PLW2080_CTRL_PERF_PERF_LIMIT_STATUS;

/*!
 * PERF_LIMITS_PMU status data.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_STATUS_PMU {
    /*!
     * Mask of PERF_LIMITs which are lwrrently active.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E255                       limitMaskActive;

    /*!
     * Default arbitration output as cached in PERF_LIMITS object in PMU.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_DEFAULT arbOutputDefault;

    /*!
     * GPU Boost Sync status.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_GPU_BOOST_SYNC             gpuBoostSync;

    /*!
     * Last set of input data sent to the change sequencer.
     */
    LW2080_CTRL_PERF_CHANGE_SEQ_CHANGE_INPUT                changeSeqChange;
} LW2080_CTRL_PERF_PERF_LIMITS_STATUS_PMU;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_STATUS_PMU *PLW2080_CTRL_PERF_PERF_LIMITS_STATUS_PMU;

/*!
 * PERF_LIMITS_35 status data.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_STATUS_35 {
    /*!
     * PERF_LIMITS_PMU super class status data.  Must always be first element in
     * struct!
     */
    LW2080_CTRL_PERF_PERF_LIMITS_STATUS_PMU            super;
    /*!
     * ARBITRATION_OUTPUT_35 structure used in last call to
     * @perfLimitsArbitrateAndApply().  Returned here for debugging the current
     * state.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT_35 arbOutput35Apply;
} LW2080_CTRL_PERF_PERF_LIMITS_STATUS_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_STATUS_35 *PLW2080_CTRL_PERF_PERF_LIMITS_STATUS_35;

/*!
 * Union of type-specific PERF_LIMITs data.
 */
typedef union LW2080_CTRL_PERF_PERF_LIMITS_STATUS_DATA {
    LW2080_CTRL_PERF_PERF_LIMITS_STATUS_PMU pmu;
    LW2080_CTRL_PERF_PERF_LIMITS_STATUS_35  v35;
} LW2080_CTRL_PERF_PERF_LIMITS_STATUS_DATA;

typedef union LW2080_CTRL_PERF_PERF_LIMITS_STATUS_DATA *PLW2080_CTRL_PERF_PERF_LIMITS_STATUS_DATA;

/*!
 * Structure representing the dynamic status of the parameters associated with
 * PERF_LIMITS. Implements the BOARDOBJGRP model/interface.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_STATUS_MESSAGE_ID (0x23U)

typedef struct LW2080_CTRL_PERF_PERF_LIMITS_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255                    super;
    /*!
     * Version of the limits structure. Values specified in
     * @ref LW2080_CTRL_PERF_PERF_LIMITS_VERSION.
     */
    LwU8                                            version;
    /*!
     * Sequence ID of all calls to @ref perfLimitsArbitrateAndApply_IMPL().
     * Used to determine if perfLimitsArbitrateAndApply calls deferred to
     * WORKITEM have gone stale due to an interleaving call to
     * perfLimitsArbitrateAndApply().  WORKITEM will confirm that the sequence
     * IDs match or bail out.
     */
    LwU32                                           applySeqId;
    /*!
     * Sequence ID for @ref perfLimitsArbitrate which is used to track
     * the dependent state of PERF_LIMITS.  Whenever any of the dependent state
     * changes, this sequence ID must be incremented.
     *
     * This sequence ID will be stored in the @ref
     * LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT struct to track the
     * snapshot of the state at which the ARBITRATION_OUTPUT was computed.
     * Subsequent calls to @ref perfLimitsArbitrate will short-circuit
     * when the sequence ID of the input struct matches the global sequence ID
     * (i.e. no dependent state has changed).
     */
    LwU32                                           arbSeqId;
    /*!
     * ARBITRATION_OUTPUT structure used in last call to
     * @perfLimitsArbitrateAndApply().  Returned here for debugging the current
     * state.
     *
     * This structure provides implementation-agnostic ARBITRATION_OUTPUT
     * debugging.  To see implementation-specific debug data, interrogate the
     * @ref data union below.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_ARBITRATION_OUTPUT arbOutputApply;
    /*!
     * Version/type-specific PERF_LIMITS status data.  Discriminated by @ref
     * version field.
     */
    LW2080_CTRL_PERF_PERF_LIMITS_STATUS_DATA        data;
    /*!
     * Array of LW2080_CTRL_PERF_PERF_LIMIT_STATUS structures. Has valid indexes
     * corresponding to the bits in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_STATUS              limits[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_LIMITS_STATUS;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_STATUS *PLW2080_CTRL_PERF_PERF_LIMITS_STATUS;

#define LW2080_CTRL_CMD_PERF_PERF_LIMITS_GET_STATUS (0x20802023) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_PERF_LIMITS_STATUS_MESSAGE_ID" */

/*!
 * Flags while setting control parameters.
 *
 * LW2080_CTRL_PERF_PERF_LIMITS_FLAGS_SYNC
 *   _NO  - Performs an asynchronous limit change. Function will return
 *          after setting the new limits. Arbitration will happen
 *          asynchronously.
 *   _YES - Performs a synchronous limit change. Function will wait for
 *          arbitration to complete before returning.
 */
#define LW2080_CTRL_PERF_PERF_LIMITS_FLAGS_SYNC                           (0:0)
#define LW2080_CTRL_PERF_PERF_LIMITS_FLAGS_SYNC_NO  (0x00000000)
#define LW2080_CTRL_PERF_PERF_LIMITS_FLAGS_SYNC_YES (0x00000001)
#define LW2080_CTRL_PERF_PERF_LIMITS_FLAGS_RSVD                          (31:1)

/*!
 * Structure describing PERF_LIMITS_35 static information.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CONTROL_PMU {
    /*!
     * Client-based input for the limit.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CLIENT_INPUT rmClientInput;
} LW2080_CTRL_PERF_PERF_LIMIT_CONTROL_PMU;

/*!
 * Structure describing PERF_LIMITS_35 static information.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CONTROL_35 {
    /*!
     * Super class. Must always be first.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CONTROL_PMU super;
} LW2080_CTRL_PERF_PERF_LIMIT_CONTROL_35;

/*!
 * PERF_LIMITS type-specific data union. Discriminated by PERF_LIMIT::super.type.
 */


/*!
 * Structure representing the control parameters associated with PERF_LIMITS.
 * Implements the BOARDOBJ model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMIT_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;
    /*!
     * XAPI does not support using an element within another structure as a
     * discriminant. Placing this redundant type value here until that design
     * constraint can be fixed.
     */
    LwU8                 type;
    /*!
     * Type-specific data union.
     */
    union {
        LW2080_CTRL_PERF_PERF_LIMIT_CONTROL_PMU pmu;
        LW2080_CTRL_PERF_PERF_LIMIT_CONTROL_35  v35;
    } data;
} LW2080_CTRL_PERF_PERF_LIMIT_CONTROL;

/*!
 * PERF_LIMITS_PMU control data.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_PMU {
    /*!
     * Boolean indicating whether the arbiter should apply the minimum or
     * maximum arbitrated results.
     */
    LwBool bApplyMin;
} LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_PMU;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_PMU *PLW2080_CTRL_PERF_PERF_LIMITS_CONTROL_PMU;

/*!
 * PERF_LIMITS_35 control data.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_35 {
    /*!
     * PERF_LIMITS_PMU super class control data.  Must always be first element
     * in struct!
     */
    LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_PMU super;
} LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_35;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_35 *PLW2080_CTRL_PERF_PERF_LIMITS_CONTROL_35;

/*!
 * Union of type-specific PERF_LIMITs data.
 */


/*!
 * Structure representing the control parameters associated with PERF_LIMITS.
 * Implements the BOARDOBJGRP model/interface.
 */
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E255 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E255 super;
    /*!
     * Version of the limits structure. Values specified in
     * @ref LW2080_CTRL_PERF_PERF_LIMITS_VERSION.
     */
    LwU8                         version;
    /*!
     * @ref LW2080_CTRL_PERF_PERF_LIMITS_FLAGS_<xyz>
     */
    LwU32                        flags;
    /*!
     * Specifies to lock/unlock the arbiter. LW_TRUE locks the arbiter,
     * preventing the arbiter from changing clocks. LW_FALSE unlocks the
     * arbiter, resulting in normal operation.
     */
    LwBool                       bArbitrateAndApplyLock;
    /*!
     * Specifies whether caching is enabled or disabled. Caching helps optimize
     * arbiter run-time by caching (and thus not re-evaluating) all state which
     * should not change between arbiter runs.
     */
    LwBool                       bCachingEnabled;
    /*!
     * Version/type-specific PERF_LIMITS control data.  Discriminated by @ref
     * version field.
     */
    union {
        LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_PMU pmu;
        LW2080_CTRL_PERF_PERF_LIMITS_CONTROL_35  v35;
    } data;
    /*!
     * Array of LW2080_CTRL_PERF_PERF_LIMIT_CONTROL structures. Has valid
     * indexes corresponding to the bits in @ref super.objMask.
     */
    LW2080_CTRL_PERF_PERF_LIMIT_CONTROL limits[LW2080_CTRL_BOARDOBJGRP_E255_MAX_OBJECTS];
} LW2080_CTRL_PERF_PERF_LIMITS_CONTROL;
typedef struct LW2080_CTRL_PERF_PERF_LIMITS_CONTROL *PLW2080_CTRL_PERF_PERF_LIMITS_CONTROL;

#define LW2080_CTRL_CMD_PERF_PERF_LIMITS_GET_CONTROL    (0x20802024) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x24" */

#define LW2080_CTRL_CMD_PERF_PERF_LIMITS_SET_CONTROL    (0x20802025) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x25" */

/*
 * LW2080_CTRL_CMD_PERF_GET_VIRTUAL_PSTATE_INFO_V2
 *
 * This command returns the information for the specified virtual P-state.
 * This information includes the P-state and a list of one or more clock
 * domain entries.
 *
 *   [in] virtualPstate
 *     This parameter specifies the vPstate for which information should
 *     be returned. Valid entries are LW2080_CTRL_PERF_VIRTUAL_PSTATES
 *     values.
 *   [in,out] index
 *     When virtualPstate is specified to be _INDEX_ONLY, this parameter
 *     specifies the direct index (vP-state number) for which information
 *     should be returned.
 *   [out] flags
 *     This parameter returns P-state specific flags, the same as
 *       LW2080_CTRL_GET_PSTATE_FLAG values.
 *   [out] pstate
 *     This parameter returns the LW2080_CTRL_PERF_PSTATES of the vPstate.
 *   [in] perfClkDomInfoListSize
 *     This parameter specifies the number of performance clock domain
 *     entries to return in the associated perfClkDomInfoList buffer.
 *     This parameter should be set to the number of enabled bits in
 *     perfClkDomainsReported parameter returned by the
 *     LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO command.
 *   [in,out] perfClkDomInfoList
 *     The performance clock information is returned into this array of
 *     LW2080_CTRL_PERF_CLK_DOM_INFO structures. The domain field for each
 *     entry has to be set to a valid LW2080_CTRL_CLK_DOMAIN.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_VIRTUAL_PSTATE_INFO_V2 (0x20802026) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_V2_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_V2_PARAMS {
    LwU32                         virtualPstate;
    LwU8                          index;
    LwU32                         flags;
    LwU32                         pstate;
    LwU32                         perfClkDomInfoListSize;
    LW2080_CTRL_PERF_CLK_DOM_INFO perfClkDomInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_PERF_GET_VIRTUAL_PSTATE_INFO_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO_V2
 *
 * This command sets performance state information for the specified P-state
 * based on the provide params. This information includes a list of one or more
 * clock domain entries and a list of one or more voltage domain entries. Will
 * only overwrite with the provided susbset of clocks and voltages upon
 * successful completion. For overclocking, it will also run stress test
 * (if supported by the OS) to sanity check the configuration.
 *
 * A couple caveats when using this ctrl call for overclocking:
 *
 *   a) This feature works on existing Tesla boards, but we don't have voltage
 *   levels standardized across boards/skus which have the same marketing name,
 *   so overriding/setting the voltage is disabled on all pre-Fermi parts.  In
 *   the future we could expose this for OEMs that want to qual boards and
 *   provide their own tools, possibly through some sort of regkey.  To enable
 *   this feature on Tesla, use the regkey RMEnableTeslaOvervoltaging.
 *
 *   b) If setting the voltage is enabled on this board, will match the
 *   specified voltage to the closest, safe voltage listed in the power control
 *   table.  For Fermi, this will be any voltage <= the safe limit specified in
 *   the voltage control table (this limit will be >= P0 voltage).  To expose
 *   voltages above the safe limit, use the regkey
 *   RMEnableFermiUnqualifiedVoltages.
 *
 *   c) This functionality is limited to the P0 p-state.  To overclock other
 *   p-states, set the regkey RMEnableOverclockingAllPstates, however there
 *   could be issues with some of the sanity checking if you try to override
 *   several p-states at once.
 *
 *   [in] pstate
 *     This parameter specifies the P-state which should be
 *     overridden. Valid entries are defined in LW2080_CTRL_PERF_PSTATES values.
 *   [in] flags
 *     This parameter allows the following options:
 *       LW2080_CTRL_SET_PSTATE_INFO_FLAG_MODE
 *         The internal test mode would bypass all checks and restrictions
 *         in the overclocking mode.
 *       LW2080_CTRL_SET_PSTATE_INFO_FLAG_TEST_ONLY
 *         This flag indicates that the set pstate info should just test the
 *         settings and run the stress test, returning OK if everything will
 *         pass.  However, no settings will be permanently applied.
 *   [in] perfClkDomInfoListSize
 *     This parameter specifies the number of performance clock domain entries
 *     to override from the associated perfGetClockInfo2List array.  This
 *     parameter should be less than or equal to the number of enabled bits in
 *     perfClkDomains parameter returned by the
 *     LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO command.
 *   [in] perfClkDomInfoList
 *     The performance clock information is overridden from this array of
 *     LW2080_CTRL_PERF_CLK_DOM_INFO structures. The number of elements used is
 *     specified by perfClkDomInfoListSize.
 *   [in] perfVoltDomInfoListSize
 *     This parameter specifies the number of performance voltage domain entries
 *     to override in the associated perfGetVoltageInfo2List buffer.  This
 *     parameter should be less than or equal to the number of enabled bits in
 *     perfVoltageDomains parameter returned by the
 *     LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO command.
 *   [in] perfVoltDomInfoList
 *     The performance voltage information is overridden from this array of
 *     LW2080_CTRL_PERF_VOLT_DOM_INFO structures. The number of entries to
 *     override from this array is given by perfVoltDomInfoListSize.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO_V2 (0x20802027) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_PSTATE_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_PSTATE_INFO_V2_PARAMS_MESSAGE_ID (0x27U)

typedef struct LW2080_CTRL_PERF_SET_PSTATE_INFO_V2_PARAMS {
    LwU32                          pstate;
    LwU32                          flags;
    LwU32                          perfClkDomInfoListSize;
    LW2080_CTRL_PERF_CLK_DOM_INFO  perfClkDomInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
    LwU32                          perfVoltDomInfoListSize;
    LW2080_CTRL_PERF_VOLT_DOM_INFO perfVoltDomInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_PERF_SET_PSTATE_INFO_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_GET_PSTATE2_INFO_V2
 *
 * This command returns performance state information for the specified
 * P-state. This information includes a list of one or more clock domain
 * entries and a list of one or more voltage domain entries.
 *
 *   [in] pstate
 *     This parameter specifies the P-state for which information should
 *     be returned. Valid entries are defined in LW2080_CTRL_PERF_PSTATES
 *     values.
 *   [out] flags
 *     This parameter returns P-state specific flags.
 *     The valid P-state flag values are:
 *       LW2080_CTRL_GET_PSTATE_FLAG_PCIELIMIT
 *         This flag indicates if the PCIE limit is GEN1 or GEN2
.*       LW2080_CTRL_GET_PSTATE_FLAG_OVERCLOCKED
.*         When true this flag indicates the Pstate is overclocked.
 *       LW2080_CTRL_GET_PSTATE_FLAG_PCIELINKWIDTH
 *         This flag indicates PCIE link width.
 *         Legal values are _1 (x1) to _32 ( x32) according to PCIE spec.
 *         _UNDEFINED means it is not defined for specified p-state.
 *   [in] perfClkDomInfoListSize
 *     This parameter specifies the number of performance clock domain
 *     entries to return in the associated perfGetClockInfo2List array.
 *     This parameter should be set to the number of enabled bits in
 *     perfClkDomains parameter returned by the
 *     LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO command.
 *   [out] perfClkDomInfoList
 *     The performance clock information is returned into this array of
 *     LW2080_CTRL_PERF_CLK_DOM_INFO. The number of elements set is specified
 *     by perfClkDomInfoListSize.
 *   [out] perfClkDom2InfoList
 *     The performance clock information is returned into this array of
 *     LW2080_CTRL_PERF_CLK_DOM2_INFO. The number of elements set is specified
 *     by perfClkDomInfoListSize.
 *   [in] perfVoltDomInfoListSize
 *     This parameter specifies the number of performance voltage domain
 *     entries to return in the associated perfGetVoltageInfo2List array.
 *     This parameter should be set to the number of enabled bits in
 *     perfVoltageDomains parameter returned by the
 *     LW2080_CTRL_CMD_PERF_GET_PSTATES_INFO command.
 *   [out] perfVoltDomInfoList
 *     The performance voltage information is returned into this array of
 *     LW2080_CTRL_PERF_VOLT_DOM_INFO. The number of elements set is specified
 *     by perfGetClkInfo2ListSize.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_PSTATE2_INFO_V2 (0x20802028) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PSTATE2_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PSTATE2_INFO_V2_PARAMS_MESSAGE_ID (0x28U)

typedef struct LW2080_CTRL_PERF_GET_PSTATE2_INFO_V2_PARAMS {
    LwU32                          pstate;
    LwU32                          flags;
    LwU32                          perfClkDomInfoListSize;
    LW2080_CTRL_PERF_CLK_DOM_INFO  perfClkDomInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
    LW2080_CTRL_PERF_CLK_DOM2_INFO perfClkDom2InfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
    LwU32                          perfVoltDomInfoListSize;
    LW2080_CTRL_PERF_VOLT_DOM_INFO perfVoltDomInfoList[LW2080_CTRL_CLK_ARCH_MAX_DOMAINS];
} LW2080_CTRL_PERF_GET_PSTATE2_INFO_V2_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_GET_OPTP_STATUS
 *
 * This command returns the current status of OPTP.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_GET_OPTP_STATUS (0x20802029) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_OPTP_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_OPTP_MAX_CLIENTS    32

typedef struct LW2080_CTRL_PERF_OPTP_STATUS_CLIENT {
    /*!
     * Client ID.
     */
    LwHandle hClient;
    /*!
     * Client's current / target time in 0.01%.
     */
    LwU32    perfVal;
    /*!
     * Application is in full screen mode.
     */
    LwBool   bFullScreen;
    /*!
     * This is a video client. (GameStream, ShadowPlay, video playback, etc.)
     */
    LwBool   bVideo;
    /*!
     * This is a VR display client.
     */
    LwBool   bVr;
    /*!
     * This is a VR renderer/compositor client.
     */
    LwBool   bVrApp;
    /*!
     * Process ID of the client (for debugging).
     */
    LwU32    processId;
    /*!
     * Cached current performance.
     */
    LwU32    current;
    /*!
     * Cached target performance.
     */
    LwU32    target;
    /*!
     * Cached ansel overhead.
     */
    LwU32    anselOverhead;
} LW2080_CTRL_PERF_OPTP_STATUS_CLIENT;

#define LW2080_CTRL_PERF_GET_OPTP_STATUS_PARAMS_MESSAGE_ID (0x29U)

typedef struct LW2080_CTRL_PERF_GET_OPTP_STATUS_PARAMS {
    /*!
     * Current number of clients.
     */
    LwU8                                numClient;
    /*!
     * Client specific status.
     */
    LW2080_CTRL_PERF_OPTP_STATUS_CLIENT client[LW2080_CTRL_PERF_OPTP_MAX_CLIENTS];
} LW2080_CTRL_PERF_GET_OPTP_STATUS_PARAMS;

/*!
 * Structure representing the info parameters associated with LWDA_LIMIT.
 */
#define LW2080_CTRL_PERF_LWDA_LIMIT_INFO_MESSAGE_ID (0x2AU)

typedef struct LW2080_CTRL_PERF_LWDA_LIMIT_INFO {
    /*!
     * POR ability to allow the LWDA_MAX setting to be overridden.
     *
     * If LW_TRUE, the ability to prevent the LWDA_MAX perf limit
     * from being applied is available to RM clients; i.e.
     * LW2080_CTRL_PERF_LWDA_LIMIT_OVERRIDE_PARAMS::bLimitApplyOverride
     * can be modified.
     */
    LwBool bLimitApplyOverrideSupported;
} LW2080_CTRL_PERF_LWDA_LIMIT_INFO;
typedef struct LW2080_CTRL_PERF_LWDA_LIMIT_INFO *PLW2080_CTRL_PERF_LWDA_LIMIT_INFO;

#define LW2080_CTRL_CMD_PERF_LWDA_LIMIT_GET_INFO (0x2080202a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_LWDA_LIMIT_INFO_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with LWDA_LIMIT.
 */
typedef struct LW2080_CTRL_PERF_LWDA_LIMIT_OVERRIDE_PARAMS {
    /*!
     * Overrides the abilility of perfLwdaLimitEvaluateLimit to
     * set the LWDA_MAX perf limit.
     *
     * If LW_TRUE, the LWDA_MAX perf limit will not be set.
     */
    LwBool bLimitApplyOverride;
} LW2080_CTRL_PERF_LWDA_LIMIT_OVERRIDE_PARAMS;
typedef struct LW2080_CTRL_PERF_LWDA_LIMIT_OVERRIDE_PARAMS *PLW2080_CTRL_PERF_LWDA_LIMIT_OVERRIDE_PARAMS;

#define LW2080_CTRL_CMD_PERF_LWDA_LIMIT_GET_OVERRIDE (0x2080202b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x2B" */

#define LW2080_CTRL_CMD_PERF_LWDA_LIMIT_SET_OVERRIDE (0x2080202c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | 0x2C" */

/*
 * LW2080_CTRL_CMD_PERF_LWDA_LIMIT_SET_CONTROL
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
#define LW2080_CTRL_CMD_PERF_LWDA_LIMIT_SET_CONTROL  (0x2080203a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS_MESSAGE_ID (0x3AU)

typedef struct LW2080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS {
    LwBool bLwdaLimit;
} LW2080_CTRL_PERF_LWDA_LIMIT_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_GET_CAPS_V2
 *
 * This command returns the set of performance capabilities for the subdevice
 * in the form of an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_POINTER
 */
#define LW2080_CTRL_CMD_PERF_GET_CAPS_V2 (0x2080202e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

/* size in bytes of power caps table */
#define LW2080_CTRL_PERF_CAPS_TBL_SIZE   1

#define LW2080_CTRL_PERF_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x2EU)

typedef struct LW2080_CTRL_PERF_GET_CAPS_V2_PARAMS {
    LwU8   capsTbl[LW2080_CTRL_PERF_CAPS_TBL_SIZE];
    /*!
     * If LW_FALSE RM will return the data for the current GPU
     * If LW_TRUE RM will return consolidated data
     */
    LwBool bCapsInitialized;
} LW2080_CTRL_PERF_GET_CAPS_V2_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_GET_INFO
 *
 * This command returns the static information whether the SLI Gpu Boost
 * synchronization feature is is enabled
 *
 *  bEnabled
 *      When set to TRUE, this parameter indicates that the feature is enabled.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_PERF_SLI_GPU_BOOST_SYNC_GET_INFO (0x2080202f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS_MESSAGE_ID (0x2FU)

typedef struct LW2080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS {
    LwBool bEnabled;
} LW2080_CTRL_PERF_SLI_GPU_BOOST_SYNC_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_PERF_NOTIFY_SREEN_SAVER_STATE
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
#define LW2080_CTRL_CMD_PERF_SET_SCREEN_SAVER_STATE (0x20802077) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS_MESSAGE_ID (0x77U)

typedef struct LW2080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS {
    LwBool bRunning;
} LW2080_CTRL_PERF_SET_SCREEN_SAVER_STATE_PARAMS;


/*
 * LW2080_CTRL_CMD_PERF_ADJUST_LIMIT_BY_PERFORMANCE
 *
 * This command can be used by a controller outside of RM to trigger a change
 * in perf limit based on the given performance parameters.
 *
 *   flags
 *     This parameter specifies LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_XXX flags:
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
#define LW2080_CTRL_CMD_PERF_ADJUST_LIMIT_BY_PERFORMANCE               (0x2080203b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | (LW2080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS_MESSAGE_ID)" */

#define LW2080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS_MESSAGE_ID 0x3B

typedef LW0080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS LW2080_CTRL_PERF_ADJUST_LIMIT_BY_PERFORMANCE_PARAMS;

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR                                                                 0:0
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR_NO          LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR_YES         LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_CLEAR_YES

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN                                                            1:1
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN_NO     LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN_YES    LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_FULLSCREEN_YES

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO                                                                 2:2
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO_NO          LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO_YES         LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VIDEO_YES

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR                                                                    3:3
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_NO             LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_YES            LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_YES

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC                                                            4:4
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC_NO     LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC_YES    LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_INC_YES

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC                                                            5:5
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC_NO     LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC_YES    LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_REFCNT_DEC_YES

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP                                                                6:6
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP_NO         LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP_YES        LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_VR_APP_YES

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER                                                               8:7
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_NO_CHANGE LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_NO_CHANGE
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_ENABLE    LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_ENABLE
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_DISABLE   LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_WHISPER_DISABLE

#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL                                                                 9:9
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL_NO          LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL_NO
#define LW2080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL_YES         LW0080_CTRL_PERF_ADJUST_LIMIT_FLAG_ANSEL_YES

// Current to target ratio is capped to 100x
#define LW2080_CTRL_PERF_ADJUST_LIMIT_C2T_RATIO_MAX          LW0080_CTRL_PERF_ADJUST_LIMIT_C2T_RATIO_MAX


/* _ctrl2080perf_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

