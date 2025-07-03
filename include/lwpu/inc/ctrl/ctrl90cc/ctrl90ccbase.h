/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2005-2022 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl90cc/ctrl90ccbase.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* GF100_PROFILER control commands and parameters */

#define LW90CC_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x90CC, LW90CC_CTRL_##cat, idx)

/* GF100_PROFILER command categories (6 bits) */
#define LW90CC_CTRL_RESERVED (0x00)
#define LW90CC_CTRL_HWPM     (0x01)
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*!
 * 90CC LWLINK category of control calls are no longer supported on GSP-RM
 * (closed source or open source)
 */
#define LW90CC_CTRL_LWLINK   (0x02)
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW90CC_CTRL_POWER    (0x03)

/*
 * LW90CC_CTRL_CMD_NULL
 *
 *    This command does nothing.
 *    This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LW90CC_CTRL_CMD_NULL (0x90cc0000) /* finn: Evaluated from "(FINN_GF100_PROFILER_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl90ccbase_h_ */
