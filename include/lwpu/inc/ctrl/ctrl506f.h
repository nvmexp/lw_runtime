/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2020 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl506f.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW50_GPFIFO control commands and parameters */

#define LW506F_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x506F, LW506F_CTRL_##cat, idx)

/* LW50_GPFIFO command categories (6bits) */
#define LW506F_CTRL_RESERVED (0x00)
#define LW506F_CTRL_GPFIFO   (0x01)
#define LW506F_CTRL_EVENT    (0x02)

/*
 * LW506F_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW506F_CTRL_CMD_NULL (0x506f0000) /* finn: Evaluated from "(FINN_LW50_CHANNEL_GPFIFO_RESERVED_INTERFACE_ID << 8) | 0x0" */

/*
 * LW506F_CTRL_CMD_RESET_ISOLATED_CHANNEL
 *
 * This command resets a channel which was isolated previously by RC recovery.
 *
 *   exceptType
 *      This input parameter specifies the type of RC error that oclwrred. See the
 *      description of the ROBUST_CHANNEL_* values in lwerror.h for valid exceptType
 *      values. info32 field of the error notifier is set with the exceptType when 
 *      the error notifier is signaled.
 *
 *   engineID
 *      This input parameter specifies the engine to be reset.  See the description 
 *      of the LW2080_ENGINE_TYPE values in cl2080.h for valid engineID values.  info16 
 *      field of the error notifier is set with the engineID when the error notifier is 
 *      signaled.    
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */




#define LW506F_CTRL_CMD_RESET_ISOLATED_CHANNEL (0x506f0105) /* finn: Evaluated from "(FINN_LW50_CHANNEL_GPFIFO_GPFIFO_INTERFACE_ID << 8) | LW506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS_MESSAGE_ID" */

/*
 * LW506F_CTRL_CMD_EVENT_SET_TRIGGER  (deprecated on Fermi+)
 *
 * This command triggers a software event for the associated channel.
 * This command accepts no parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
// #define LW506F_CTRL_CMD_EVENT_SET_TRIGGER         LW506F_CTRL_CMD(EVENT, 0x09)

#define LW506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS {
    LwU32 exceptType;
    LwU32 engineID;
} LW506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS;/* _ctrl506f.h_ */
