/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2011-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl83de/ctrl83defifo.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl83de/ctrl83debase.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW83DE_CTRL_CMD_FIFO_SUSPEND_RESUME_CTXSW
 *
 * This command is deprecated and no longer supported.
 *
 * Possible status values returned are
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW83DE_CTRL_CMD_FIFO_SUSPEND_RESUME_CTXSW (0x83de0201) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_FIFO_INTERFACE_ID << 8) | 0x1" */

typedef struct LW83DE_CTRL_FIFO_SUSPEND_RESUME_CTXSW_PARAMS {
    LwU32 srAction;   /* Defines if it is to be a Suspend or Resume action */
} LW83DE_CTRL_FIFO_SUSPEND_RESUME_CTXSW_PARAMS;

#define LW83DE_CTRL_FIFO_SUSPEND_ACTION             0x00000001
#define LW83DE_CTRL_FIFO_RESUME_ACTION              0x00000002


/*
 * LW83DE_CTRL_CMD_FIFO_GET_CTX_RESIDENT_STATE
 *
 * This command is deprecated and no longer supported.
 *
 * Possible return status values are
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW83DE_CTRL_CMD_FIFO_GET_CTX_RESIDENT_STATE (0x83de0202) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_FIFO_INTERFACE_ID << 8) | 0x2" */

typedef struct LW83DE_CTRL_FIFO_GET_CTX_RESIDENT_STATE_PARAMS {
    LwHandle hChannel;
} LW83DE_CTRL_FIFO_GET_CTX_RESIDENT_STATE_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/* _ctrl83defifo_h_ */
