/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2007-2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrla06f/ctrla06finternal.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrla06f/ctrla06fbase.h"
#include "ctrl/ctrla06f/ctrla06fgpfifo.h"

/*
 * LWA06F_CTRL_CMD_INTERNAL_STOP_CHANNEL
 *
 * This command is an internal command sent from Kernel RM to Physical RM
 * to stop the channel in hardware
 *
 * Please see description of LWA06F_CTRL_CMD_STOP_CHANNEL for more information.
 *
 */
#define LWA06F_CTRL_CMD_INTERNAL_STOP_CHANNEL    (0xa06f0301) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_INTERNAL_INTERFACE_ID << 8) | 0x1" */

/*
 * LWA06F_CTRL_CMD_INTERNAL_RESET_CHANNEL
 *
 * This command is an internal command sent from Kernel RM to Physical RM
 * to perform the channel reset operations in hardware
 *
 * Please see description of LW906F_CTRL_CMD_RESET_CHANNEL for more information.
 *
 */
#define LWA06F_CTRL_CMD_INTERNAL_RESET_CHANNEL   (0xa06f0302) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_INTERNAL_INTERFACE_ID << 8) | 0x2" */

/*
 * LWA06F_CTRL_CMD_INTERNAL_GPFIFO_SCHEDULE
 *
 * This command is an internal command sent from Kernel RM to Physical RM
 * to schedule the channel in hardware
 *
 * Please see description of LWA06F_CTRL_CMD_GPFIFO_SCHEDULE for more information.
 *
 */
#define LWA06F_CTRL_CMD_INTERNAL_GPFIFO_SCHEDULE (0xa06f0303) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_INTERNAL_INTERFACE_ID << 8) | 0x3" */

/* ctrla06finternal_h */
