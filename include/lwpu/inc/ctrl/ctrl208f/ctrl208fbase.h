/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl208f/ctrl208fbase.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW20_SUBDEVICE_DIAG: diagnostic class control commands and parameters */

#define LW208F_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x208F, LW208F_CTRL_##cat, idx)

/* Subdevice diag command categories (6bits) */
#define LW208F_CTRL_RESERVED     (0x00)
#define LW208F_CTRL_POWER        (0x01)
#define LW208F_CTRL_THERMAL      (0x02)
#define LW208F_CTRL_SEQ          (0x03)
#define LW208F_CTRL_FIFO         (0x04)
#define LW208F_CTRL_FB           (0x05)
#define LW208F_CTRL_MC           (0x06)
#define LW208F_CTRL_BIF          (0x07)
#define LW208F_CTRL_CLK          (0x08)
#define LW208F_CTRL_PERF         (0x09)
#define LW208F_CTRL_FBIO         (0x0A)
#define LW208F_CTRL_MMU          (0x0B)
#define LW208F_CTRL_PMU          (0x0C)
#define LW208F_CTRL_EVENT        (0x10)
#define LW208F_CTRL_GPU          (0x11)
#define LW208F_CTRL_GR           (0x12)
#define LW208F_CTRL_PMGR         (0x13)
#define LW208F_CTRL_DMA          (0x14)
// const LW208F_CTRL_TMR = (0x15); // not supported
#define LW208F_CTRL_RMFS         (0x16)
#define LW208F_CTRL_GSPMSGTIMING (0x17)
#define LW208F_CTRL_BUS          (0x18)

/*
 * LW208F_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *     LW_OK
 */
#define LW208F_CTRL_CMD_NULL     (0x208f0000) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl208fbase_h_ */
