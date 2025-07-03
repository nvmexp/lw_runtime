/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017-2022 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl000f.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"

#define LW000F_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x000f, LW000F_CTRL_##cat, idx)

/* Client command categories (6bits) */
#define LW000F_CTRL_RESERVED  (0x00U)
#define LW000F_CTRL_FM        (0x01U)
#define LW000F_CTRL_RESERVED2 (0x02U)

/*
 * LW000f_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LW000F_CTRL_CMD_NULL  (0xf0000U) /* finn: Evaluated from "(FINN_FABRIC_MANAGER_SESSION_RESERVED_INTERFACE_ID << 8) | 0x0" */



/*
 * LW000F_CTRL_CMD_SET_FM_STATE
 *
 * This command will notify RM that the fabric manager is initialized.
 *
 * RM would block P2P operations such as P2P capability reporting, LW50_P2P object
 * allocation etc. until the notification received.
 *
 * Possible status values returned are:
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_OBJECT_NOT_FOUND
 *    LW_ERR_NOT_SUPPORTED
 *    LW_OK
 */
#define LW000F_CTRL_CMD_SET_FM_STATE   (0xf0101U) /* finn: Evaluated from "(FINN_FABRIC_MANAGER_SESSION_FM_INTERFACE_ID << 8) | 0x1" */

/*
 * LW000F_CTRL_CMD_CLEAR_FM_STATE
 *
 * This command will notify RM that the fabric manager is uninitialized.
 *
 * RM would block P2P operations such as P2P capability reporting, LW50_P2P object
 * allocation etc. as soon as the notification received.
 *
 * Possible status values returned are:
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_OBJECT_NOT_FOUND
 *    LW_ERR_NOT_SUPPORTED
 *    LW_OK
 */
#define LW000F_CTRL_CMD_CLEAR_FM_STATE (0xf0102U) /* finn: Evaluated from "(FINN_FABRIC_MANAGER_SESSION_FM_INTERFACE_ID << 8) | 0x2" */

/* _ctrl000f.h_ */
