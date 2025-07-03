/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
//
// This file should NEVER be published.
//
#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080clk_opaque_non_privileged.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "lwfixedtypes.h"
#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080clk.h"
#include "lwmisc.h"

/*
 * LW2080_CTRL_CLK_MEASURE_FREQ
 *
 * This command is used to callwlate the real frequency of a clock by using
 * clock counters. Note that this API does NOT measure the PLL output, but
 * the ultimate clock output at the lowest point of the stream.
 *   clkDomain
 *     The clock domain for which we want to measure the frequency.
 *     For more information on clkDomain see LW2080_CTRL_CLK_GET_DOMAINS above.
 *     This API will fail if you pass down any domain that's not supported on the chip.
 *   freqKHz
 *     This field returns the callwlated frequency in units of KHz.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW2080_CTRL_CMD_CLK_MEASURE_FREQ (0x20809006) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CLK_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_CLK_MEASURE_FREQ_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CLK_MEASURE_FREQ_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_CLK_MEASURE_FREQ_PARAMS {
    LwU32 clkDomain;
    LwU32 freqKHz;
} LW2080_CTRL_CLK_MEASURE_FREQ_PARAMS;


#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

