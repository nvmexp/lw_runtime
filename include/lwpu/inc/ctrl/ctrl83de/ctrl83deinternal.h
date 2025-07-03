/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl83de/ctrl83deinternal.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl83de/ctrl83debase.h"

#define LW83DE_CTRL_CMD_INTERNAL_SM_DEBUG_MODE_ENABLE                (0x83de0401) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x1" */

#define LW83DE_CTRL_CMD_INTERNAL_SM_DEBUG_MODE_DISABLE               (0x83de0402) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x2" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_MODE_MMU_DEBUG                  (0x83de0403) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x3" */

#define LW83DE_CTRL_CMD_INTERNAL_GET_MODE_MMU_DEBUG                  (0x83de0404) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x4" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_EXCEPTION_MASK                  (0x83de0405) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x5" */

#define LW83DE_CTRL_CMD_INTERNAL_GET_EXCEPTION_MASK                  (0x83de0406) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x6" */

#define LW83DE_CTRL_CMD_INTERNAL_READ_SINGLE_SM_ERROR_STATE          (0x83de0407) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x7" */

#define LW83DE_CTRL_CMD_INTERNAL_READ_ALL_SM_ERROR_STATES            (0x83de0408) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x8" */

#define LW83DE_CTRL_CMD_INTERNAL_CLEAR_SINGLE_SM_ERROR_STATE         (0x83de0409) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x9" */

#define LW83DE_CTRL_CMD_INTERNAL_CLEAR_ALL_SM_ERROR_STATES           (0x83de040a) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0xA" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_NEXT_STOP_TRIGGER_TYPE          (0x83de040b) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0xB" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_SINGLE_STEP_INTERRUPT_HANDLING  (0x83de040c) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0xC" */

#define LW83DE_CTRL_CMD_INTERNAL_SUSPEND_CONTEXT                     (0x83de040d) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0xD" */

#define LW83DE_CTRL_CMD_INTERNAL_RESUME_CONTEXT                      (0x83de040e) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0xE" */

#define LW83DE_CTRL_CMD_INTERNAL_EXEC_REG_OPS                        (0x83de040f) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0xF" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_MODE_ERRBAR_DEBUG               (0x83de0410) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x10" */

#define LW83DE_CTRL_CMD_INTERNAL_GET_MODE_ERRBAR_DEBUG               (0x83de0411) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x11" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_SINGLE_SM_SINGLE_STEP           (0x83de0412) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x12" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_SINGLE_SM_STOP_TRIGGER          (0x83de0413) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x13" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_SINGLE_SM_RUN_TRIGGER           (0x83de0414) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x14" */

#define LW83DE_CTRL_CMD_INTERNAL_SET_SINGLE_SM_SKIP_IDLE_WARP_DETECT (0x83de0415) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x15" */

#define LW83DE_CTRL_CMD_INTERNAL_GET_SINGLE_SM_DEBUGGER_STATUS       (0x83de0416) /* finn: Evaluated from "(FINN_GT200_DEBUGGER_INTERNAL_INTERFACE_ID << 8) | 0x16" */

/* ctrl83deinternal_h */
