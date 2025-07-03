/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc372/ctrlc372verif.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrlc372/ctrlc372base.h"
#include "ctrl/ctrlc372/ctrlc372chnc.h" // LWC372_CTRL_MAX_POSSIBLE_WINDOWS

/*
 * LWC372_CTRL_CMD_GET_IMP_CALC_DATA
 * 
 * This command returns various callwlated data from the last IMP call.
 */
#define LWC372_CTRL_CMD_GET_IMP_CALC_DATA (0xc3720201) /* finn: Evaluated from "(FINN_LWC372_DISPLAY_SW_VERIF_INTERFACE_ID << 8) | LWC372_CTRL_GET_IMP_CALC_DATA_PARAMS_MESSAGE_ID" */

#define LWC372_CTRL_GET_IMP_CALC_DATA_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC372_CTRL_GET_IMP_CALC_DATA_PARAMS {
    LWC372_CTRL_CMD_BASE_PARAMS base;

    LwU32                       impResult;
    LwU32                       availableBandwidthMBPS;
    LwU32                       rrrbLatencyNs;
    LwU32                       requiredDownstreamHubClkKHz;
    LwU32                       requiredDispClkKHz;

    LwU32                       maxDispClkKHz;
    LwU32                       maxHubClkKHz;
    LwU32                       requiredDramClkKHz;
    LwU32                       requiredMcClkKHz;
    LwU32                       requiredMchubClkKHz;

    LwU32                       minFBFetchRateWindowKBPS[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU32                       minFBFetchRateLwrsorKBPS[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU16                       minMempoolAllocWindow[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU16                       minMempoolAllocLwrsor[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU16                       mempoolAllocWindow[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU16                       mempoolAllocLwrsor[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU8                        fetchMeterWindow[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU8                        fetchMeterLwrsor[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU16                       requestLimitWindow[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU16                       requestLimitLwrsor[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU16                       pipeMeterWindow[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU16                       pipeMeterRatioWin[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU16                       pipeMeterLwrsor[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU16                       pipeMeterRatioLwrsor[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU8                        drainMeterWindow[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
    LwU8                        drainMeterLwrsor[LWC372_CTRL_MAX_POSSIBLE_HEADS];

    LW_DECLARE_ALIGNED(LwU64 spoolUpTimeNs[LWC372_CTRL_MAX_POSSIBLE_HEADS], 8);
    LwU32                       elvStart[LWC372_CTRL_MAX_POSSIBLE_HEADS];
    LwU32                       memfetchVblankDurationUs[LWC372_CTRL_MAX_POSSIBLE_HEADS];

    struct {
        LwU32 outputScalerInRateLinesPerSec;
    } head[LWC372_CTRL_MAX_POSSIBLE_HEADS];

    struct {
        LwU32 inputScalerInRateLinesPerSec;
    } window[LWC372_CTRL_MAX_POSSIBLE_WINDOWS];
} LWC372_CTRL_GET_IMP_CALC_DATA_PARAMS;
typedef struct LWC372_CTRL_GET_IMP_CALC_DATA_PARAMS *PLWC372_CTRL_GET_IMP_CALC_DATA_PARAMS;

/*
 * LWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS
 *
 * This command recallwlates BW and latency values in FB_IMP_PERF_LEVEL for the 
 * highest perf level. The structure in question resides in 
 * /chips_a/drivers/resman/kernel/inc/volta/volta_fb.h
 */
#define LWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS (0xc3720202) /* finn: Evaluated from "(FINN_LWC372_DISPLAY_SW_VERIF_INTERFACE_ID << 8) | LWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS_PARAMS_MESSAGE_ID" */

#define LWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS_PARAMS {
    LWC372_CTRL_CMD_BASE_PARAMS base;

    LW_DECLARE_ALIGNED(LwU64 dispClkHz, 8);
    LW_DECLARE_ALIGNED(LwU64 ramClkHz, 8);
    LW_DECLARE_ALIGNED(LwU64 l2ClkHz, 8);
    LW_DECLARE_ALIGNED(LwU64 xbarClkHz, 8);
    LW_DECLARE_ALIGNED(LwU64 hubClkHz, 8);
    LW_DECLARE_ALIGNED(LwU64 sysClkHz, 8);
} LWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS_PARAMS;
typedef struct LWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS_PARAMS *PLWC372_CTRL_CMD_IMP_STATE_LOAD_FORCE_CLOCKS_PARAMS;

/* _ctrlc372verif_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

