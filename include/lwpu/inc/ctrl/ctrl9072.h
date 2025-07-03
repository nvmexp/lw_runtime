/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2020 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl9072.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW9072_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x9072, LW9072_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW9072_CTRL_RESERVED (0x00)
#define LW9072_CTRL_DISP_SW  (0x01)


/*
 * LW9072_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LW9072_CTRL_CMD_NULL (0x90720000) /* finn: Evaluated from "(FINN_GF100_DISP_SW_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW9072_CTRL_CMD_NOTIFY_ON_VBLANK
 *
 * This command implements an out-of-band version of the
 * GF100_DISP_SW class's LW9072_NOTIFY_ON_VBLANK method.
 *
 * Parameters:
 *
 *   data
 *     Valid data accepted by the LW9072_NOTIFY_ON_VBLANK method.
 *   bHeadDisabled
 *     Specifies whether head is active while adding vblank
 *     callback.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LWOS_STATUS_ILWALID_PARAM_STRUCT
 *    LW_ERR_ILWALID_STATE
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW9072_CTRL_CMD_NOTIFY_ON_VBLANK (0x90720101) /* finn: Evaluated from "(FINN_GF100_DISP_SW_DISP_SW_INTERFACE_ID << 8) | LW9072_CTRL_CMD_NOTIFY_ON_VBLANK_PARAMS_MESSAGE_ID" */

#define LW9072_CTRL_CMD_NOTIFY_ON_VBLANK_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW9072_CTRL_CMD_NOTIFY_ON_VBLANK_PARAMS {
    LwU32  data;
    LwBool bHeadDisabled;
} LW9072_CTRL_CMD_NOTIFY_ON_VBLANK_PARAMS;

/* _ctrl9072.h_ */
