/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl9010.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW9010_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x9010, LW9010_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW9010_CTRL_RESERVED        (0x00)
#define LW9010_CTRL_VBLANK_CALLBACK (0x01)


/*
 * LW9010_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LW9010_CTRL_CMD_NULL        (0x90100000) /* finn: Evaluated from "(FINN_LW9010_VBLANK_CALLBACK_RESERVED_INTERFACE_ID << 8) | 0x0" */



/*
 * LW9010_CTRL_CMD_SET_VBLANK_NOTIFICATION
 *
 * This command is used to enable and disable vblank notifications. This 
 * is specially intended for cases where the RM client is calling from a
 * high IRQL context, where other mechanisms to toggle vblank notification
 * (such as freeing and reallocating the LW9010_VBLANK_CALLBACK object)
 * would not be suitable. As this is being ilwoked at the high IRQL, 
 * locking can be bypassed, if the LWOS54_FLAGS_LOCK_BYPASS flag is set on 
 * the control call.Here the OS will take care of the synchronization. 
 * The Windows Display Driver for Cobalt requires this, for example.
 *
 *    bSetVBlankNotifyEnable
 *       This parameter tell whether to enable or disable the Vblank notification
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW9010_CTRL_CMD_SET_VBLANK_NOTIFICATION (0x90100101) /* finn: Evaluated from "(FINN_LW9010_VBLANK_CALLBACK_INTERFACE_ID << 8) | LW9010_CTRL_CMD_SET_VBLANK_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW9010_CTRL_CMD_SET_VBLANK_NOTIFICATION_PARAMS_MESSAGE_ID (0x01U)

typedef struct LW9010_CTRL_CMD_SET_VBLANK_NOTIFICATION_PARAMS {
    LwBool bSetVBlankNotifyEnable;
} LW9010_CTRL_CMD_SET_VBLANK_NOTIFICATION_PARAMS;

/* _ctrl9010_h_ */
