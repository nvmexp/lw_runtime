/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2005-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl83de/ctrl83debase.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* GT200_DEBUG control commands and parameters */

#define LW83DE_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0x83DEU, LW83DE_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW83DE_CTRL_RESERVED (0x00)
#define LW83DE_CTRL_GR       (0x01)
#define LW83DE_CTRL_FIFO     (0x02)
#define LW83DE_CTRL_DEBUG    (0x03)
#define LW83DE_CTRL_INTERNAL (0x04)


/*
 * LW83DE_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW83DE_CTRL_CMD_NULL (0x83de0000) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl83debase_h_ */
