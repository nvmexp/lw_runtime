/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021-2022 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl2080/ctrl2080perf_opaque_privileged.finn
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

#include "lwmisc.h"

#include "ctrl/ctrl2080/ctrl2080perf_opaque_non_privileged.h"


/*
 * LW2080_CTRL_CMD_PERF_SET_BOOST_FREQUENCIES
 *
 * This command sets the frequencies for the boost virtual P-state.  It is a
 * temporary command.  As time permits, it will be replaced by the standard
 * vP-state mutator, LW2080_CTRL_CMD_PERF_SET_VIRTUAL_PSTATE_INFO.
 */
#define LW2080_CTRL_CMD_PERF_SET_BOOST_FREQUENCIES (0x2080e019) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_LEGACY_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_BOOST_FREQUENCIES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_BOOST_FREQUENCIES_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW2080_CTRL_PERF_SET_BOOST_FREQUENCIES_PARAMS {
    LwU32 gpc2FreqKHz;
    LwU32 memoryFreqKHz;
} LW2080_CTRL_PERF_SET_BOOST_FREQUENCIES_PARAMS;

/*!
 * LW2080_CTRL_CMD_PERF_SET_PSTATES20_DATA
 *
 * NOTE: This CMD is deprecated on Pstate 3.0. please use the new Pstate
 * RMCTRLs. ref@ LW2080_CTRL_CMD_PERF_PSTATES_SET_CONTROL
 *
 * This command sets an information for all requested performance states.
 * This information includes clock and voltage settings across one or more
 * requested P-states.
 * It represents and enhanced version of LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO
 * command, and it allow its caller to apply all information atomically.
 * Before issuing this command, caller is responsible to populate information
 * specifying P-states of interest as well as lists of required clock and
 * voltage parameters (fields marked as IN) together with the new settings for
 * these parameters (fields marked as IN-SET).
 * For more information on use and restrictions of this call see documentation
 * of LW2080_CTRL_CMD_PERF_SET_PSTATE_INFO command.
 *
 *  flags
 *    This parameter specifies flags common to all P-states.
 *      LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_FLAGS_MODE
 *        This field is used to select either the (default) overclocking or
 *        the internal test mode. With the internal test mode, the new settings
 *        are applied directly with minimal processing and checking.
 *  numPstates [IN]
 *    This parameter specifies number of requested pstate entries to be
 *    processed.
 *  numClocks [IN]
 *    This parameter specifies number of requested perfClkDomInfoList entries
 *    to be processed
 *  numVoltages [IN]
 *    This parameter specifies number of requested perfVoltDomInfoList entries
 *    to be processed.
 *  pstate
 *    This parameter is an array of elements each one containing information
 *    describing single P-state.
 *  ov
 *    This parameter specifies data related to over-voltaging.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_PERF_SET_PSTATES20_DATA (0x2080e06b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_LEGACY_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_MESSAGE_ID (0x6BU)

typedef struct LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS {
    LwU32 flags;
    LwU32 numPstates;
    LwU32 numClocks;
    LwU32 numVoltages;
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PSTATE20 pstate[LW2080_CTRL_PERF_PSTATE20_COUNT_MAX], 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PSTATE20_OV ov, 8);
} LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS;

#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

