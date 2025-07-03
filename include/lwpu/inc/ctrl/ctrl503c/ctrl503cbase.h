/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2005-2019 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl503c/ctrl503cbase.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW50_THIRD_PARTY_P2P control commands and parameters */
#define LW503C_CTRL_CMD(cat,idx)  LWXXXX_CTRL_CMD(0x503C,LW503C_CTRL_##cat,idx)

/* LW50_THIRD_PARTY_P2P command categories (6bits) */
#define LW503C_CTRL_RESERVED (0x00)
#define LW503C_CTRL_P2P      (0x01)

/*
 * LW503C_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW503C_CTRL_CMD_NULL (0x503c0000) /* finn: Evaluated from "(FINN_LW50_THIRD_PARTY_P2P_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl503cbase_h_ */
