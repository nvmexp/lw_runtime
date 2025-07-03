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
// Source file: ctrl/ctrl2080/ctrl2080perf_opaque_non_privileged.finn
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

/* LW20_SUBDEVICE_XX perf opaque non privileged control commands and parameters */


/*!
 * This define limits the max number of P-states that can be handled by either
 * LW2080_CTRL_CMD_PERF_GET/SET_PSTATES20_DATA commands.
 *
 * @note Needs to be in sync (greater or equal) to LWAPI_MAX_GPU_PERF_PSTATES.
 */
#define LW2080_CTRL_PERF_PSTATE20_COUNT_MAX 16

/*!
 * LW2080_CTRL_PERF_PSTATE20
 *
 * This structure encapsulates all data related to a single performance state.
 *
 *  pstateID [IN]
 *    This parameter specifies an unique P-state identifier as defined in
 *    LW2080_CTRL_PERF_PSTATES_<xyz>.
 *  flags
 *    This parameter returns flags specific to current P-state.
 *      LW2080_CTRL_PERF_PSTATE20_FLAGS_EDITABLE
 *        This field indicates that at least one clock and/or voltage domain in
 *        current P-state can be modified.
 *      LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L0S
 *        This field indicates L0s capability of this pstate
 *      LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_L1
 *        This field indicates L1 capability of this pstate
 *      LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_DEEPL1
 *        This field indicates deep L1 capability of this pstate
 *      LW2080_CTRL_PERF_PSTATE20_FLAGS_PCIECAPS_SET_ASPM
 *        This field controls whether we are changing ASPM settings or not.
 *  perfClkDomInfoList [IN]
 *    This field specifies a pointer in the caller's address space to the buffer
 *    into which the performance clock information is to be returned (or
 *    retrieved from for SET call). This buffer must be at least as big as
 *    LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS::numClocks multiplied by the
 *    size of the LW2080_CTRL_PERF_PSTATE20_CLK_DOM_INFO structure.
 *  perfVoltDomInfoList [IN]
 *    This field specifies a pointer in the caller's address space to the buffer
 *    into which the performance voltage information is to be returned (or
 *    retrieved from for SET call). This buffer must be at least as big as
 *    LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS::numVoltages multiplied by the
 *    size of the LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO structure.
 */
typedef struct LW2080_CTRL_PERF_PSTATE20 {
    LwU32 pstateID;
    LwU32 flags;
    LW_DECLARE_ALIGNED(LwP64 perfClkDomInfoList, 8);
    LW_DECLARE_ALIGNED(LwP64 perfVoltDomInfoList, 8);
} LW2080_CTRL_PERF_PSTATE20;

/*!
 * LW2080_CTRL_PERF_PSTATE20_OV
 *
 * This structure encapsulates all data related to over-voltaging.
 *
 *  flags
 *    This parameter returns flags specific to OV settings.
 *      LW2080_CTRL_PERF_PSTATE20_OV_FLAGS_EDITABLE
 *        This field indicates that at least one voltage domain can be
 *        over-voltaged.
 *  numVoltages [IN]
 *    This parameter specifies number of requested perfVoltDomInfoList entries
 *    to be processed.
 *  perfVoltDomInfoList [IN]
 *    This field specifies a pointer in the caller's address space to the buffer
 *    into which the over-voltage information is to be returned (or retrieved
 *    from for SET call). This buffer must be at least as big as
 *    LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS::numVoltages multiplied by the
 *    size of the LW2080_CTRL_PERF_PSTATE20_VOLT_DOM_INFO structure.
 */
typedef struct LW2080_CTRL_PERF_PSTATE20_OV {
    LwU32 flags;
    LwU32 numVoltages;
    LW_DECLARE_ALIGNED(LwP64 perfVoltDomInfoList, 8);
} LW2080_CTRL_PERF_PSTATE20_OV;

#define LW2080_CTRL_PERF_PSTATE20_OV_FLAGS_EDITABLE     0:0
#define LW2080_CTRL_PERF_PSTATE20_OV_FLAGS_EDITABLE_NO  (0x00000000)
#define LW2080_CTRL_PERF_PSTATE20_OV_FLAGS_EDITABLE_YES (0x00000001)

/*!
 * LW2080_CTRL_CMD_PERF_GET_PSTATES20_DATA
 *
 * NOTE: This CMD is deprecated on Pstate 3.0. please use the new Pstate
 * RMCTRLs. ref@ LW2080_CTRL_CMD_PERF_PSTATES_GET_INFO
 *
 * This command returns an information about all requested performance states.
 * This information includes a list of one or more clock domain entries and
 * a list of one or more voltage domain entries.
 * It represent an enhanced version of LW2080_CTRL_CMD_PERF_GET_PSTATE2_INFO
 * command, and it allow its caller to retrieve all information atomically.
 * Before issuing this command, caller is responsible to populate information
 * specifying P-states of interest as well as lists of required clock and
 * voltage parameters (fields marked as IN).
 *
 *  flags
 *    This parameter returns flags common to all P-states.
 *      LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_FLAGS_EDITABLE
 *        This field indicates that at least one P-state contains at least one
 *        clock and/or voltage domain that can be modified. In other words that
 *        this system supports over-clocking / over-voltaging.
 *  numPstates [IN]
 *    This parameter specifies number of requested pstate entries to be
 *    processed.
 *  numClocks [IN]
 *    This parameter specifies number of requested perfClkDomInfoList entries
 *    to be processed.
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
#define LW2080_CTRL_CMD_PERF_GET_PSTATES20_DATA         (0x2080a06a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_PERF_LEGACY_NON_PRIVILEGED_INTERFACE_ID << 8) | LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_MESSAGE_ID (0x6AU)

typedef struct LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS {
    LwU32 flags;
    LwU32 numPstates;
    LwU32 numClocks;
    LwU32 numVoltages;
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PSTATE20 pstate[LW2080_CTRL_PERF_PSTATE20_COUNT_MAX], 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_PERF_PSTATE20_OV ov, 8);
} LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS;

// Keep this in sync with LW2080_CTRL_PERF_SET_PSTATES20_DATA_PARAMS_FLAGS.
#define LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_FLAGS_EDITABLE       0:0
#define LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_FLAGS_EDITABLE_NO  (0x00000000)
#define LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_FLAGS_EDITABLE_YES (0x00000001)

#define LW2080_CTRL_PERF_GET_PSTATES20_DATA_PARAMS_FLAGS_RSVD1          1:1
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

