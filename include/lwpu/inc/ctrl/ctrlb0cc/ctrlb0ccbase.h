/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrlb0cc/ctrlb0ccbase.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"  // LW2080_CTRL_GPU_REG_OP
#define LWB0CC_CTRL_CMD(cat,idx)                     LWXXXX_CTRL_CMD(0xB0CC, LWB0CC_CTRL_##cat, idx)

/* MAXWELL_PROFILER command categories (6 bits) */
#define LWB0CC_CTRL_RESERVED (0x00)
#define LWB0CC_CTRL_PROFILER (0x01)
#define LWB0CC_CTRL_INTERNAL (0x02)

/*!
 * LWB0CC_CTRL_CMD_NULL
 *
 *    This command does nothing.
 *    This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LWB0CC_CTRL_CMD_NULL (0xb0cc0000) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_RESERVED_INTERFACE_ID << 8) | 0x0" */


/* _ctrlb0ccbase_h_ */
