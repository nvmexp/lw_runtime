/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2007-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fpower.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * LW208F_CTRL_CMD_SUSPEND_RESUME_QUICK
 *
 * This command puts the GPU in D3, resumes it back to D0
 * This command is developed purely for testing purpose to do quick D3<->D0
 * transitions in a shorter amount of time under simulation.
 *
 *   srAction
 *     Specifies whether GPU should be suspend, resumed, or dirty the fb
 *     contents.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW208F_CTRL_CMD_SUSPEND_RESUME_QUICK (0x208f0101) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_POWER_INTERFACE_ID << 8) | LW208F_CTRL_SUSPEND_RESUME_QUICK_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_SUSPEND_RESUME_QUICK_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_SUSPEND_RESUME_QUICK_PARAMS {
    LwU32 srAction;
} LW208F_CTRL_SUSPEND_RESUME_QUICK_PARAMS;

#define LW208F_CTRL_CMD_SUSPEND_RESUME_ACTION_SUSPEND          (0x00000001)
#define LW208F_CTRL_CMD_SUSPEND_RESUME_ACTION_RESUME           (0x00000002)
#define LW208F_CTRL_CMD_SUSPEND_RESUME_ACTION_FB_DIRTY_FULL    (0x00000003)
#define LW208F_CTRL_CMD_SUSPEND_RESUME_ACTION_FB_DIRTY_PARTIAL (0x00000004)

/* _ctrl208fpower_h_ */

