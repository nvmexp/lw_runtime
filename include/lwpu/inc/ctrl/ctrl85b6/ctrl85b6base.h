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
// Source file: ctrl/ctrl85b6/ctrl85b6base.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* GT212 PMU control command supporting structures */

#define LW85B6_CTRL_CMD(cat,idx) \
    LWXXXX_CTRL_CMD(0x85B6, LW85B6_CTRL_##cat, idx)

/* PMU command categories (6bits) */
#define LW85B6_CTRL_RESERVED (0x00)
#define LW85B6_CTRL_PMU      (0x01)

/*
 * LW85B6_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW85B6_CTRL_CMD_NULL (0x85b60000) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl85b6base_h_ */
