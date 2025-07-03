/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fclk.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

typedef struct LW208F_CTRL_CLK_GET_OPERATIONAL_FREQ_LIMITS_PARAMS {
    LwU32 clkDomain;
    LwU32 minOperationalFreqKHz;
    LwU32 maxOperationalFreqKHz;
} LW208F_CTRL_CLK_GET_OPERATIONAL_FREQ_LIMITS_PARAMS;

/*
 * LW208F_CTRL_CMD_CLK_VERIFY_FREQ
 *
 * It is used to check that whether the difference between the current frequency
 * of the clock domain supplied and the corresponding frequency defined in 
 * the vbios for the gievn pstate is within the specified tolerance.
 *   pstate
 *     This parameter specifies the P-state for which the verification
 *     of clk frequencies is to be done. Valid entries are defined in 
 *     LW2080_CTRL_PERF_PSTATES values.
 *   clkDomain
 *     The clock domain for which we want to verify the frequency. Valid entries
 *     are must be one of LW2080_CTRL_CLK_DOMAIN values.
 *   tolerance100X;
 *     The maximum deviation of clock frequency that can be allowed , from 
 *     the frequency defined in the vbios for the given pstate. It is expressed
 *     as a percentage multiplied by 100.
 *
 * Possible status values returned are:
 *   LW_OK 
 *     Returned if the frequencies are within the specified range.
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_OPERATING_SYSTEM   
 */
#define LW208F_CTRL_CMD_CLK_VERIFY_FREQ (0x208f0802) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_CLK_INTERFACE_ID << 8) | LW208F_CTRL_CLK_VERIFY_FREQ_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_CLK_VERIFY_FREQ_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_CLK_VERIFY_FREQ_PARAMS {
    LwU32 pstate;
    LwU32 clkDomain;
    LwU32 tolerance100X;
} LW208F_CTRL_CLK_VERIFY_FREQ_PARAMS;

