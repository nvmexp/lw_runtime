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
// Source file: ctrl/ctrla06f/ctrla06fbase.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/* GK100_GPFIFO control commands and parameters */

#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl906f.h"          /* A06F is partially derived from 906F */

#define LWA06F_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xA06F, LWA06F_CTRL_##cat, idx)

/* GK100_GPFIFO command categories (6bits) */
#define LWA06F_CTRL_RESERVED (0x00)
#define LWA06F_CTRL_GPFIFO   (0x01)
#define LWA06F_CTRL_EVENT    (0x02)
#define LWA06F_CTRL_INTERNAL (0x03)

/*
 * LWA06F_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 *
 */
#define LWA06F_CTRL_CMD_NULL (0xa06f0000) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrla06fbase_h_ */
