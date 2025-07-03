/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0073/ctrl0073dpu.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl0073/ctrl0073base.h"

/* LW04_DISPLAY_COMMON DPU control commands and parameters */

/*
 * LW0073_CTRL_CMD_DPU_UCODE_STATE
 *
 * This command is used to retrieve the internal state of the DPU ucode.  It
 * may be used to determine when the DPU is ready to accept and process
 * commands.
 *
 *  ucodeState
 *    This parameter returns the internal DPU ucode state.  Legal values for 
 *    this parameter include:
 *      LW0073_CTRL_DPU_UCODE_STATE_NONE
 *        This value indicates that the DPU ucode has not been loaded (ie. the 
 *        DPU has not been bootstrapped).  The DPU is not accepting commands at
 *        this point.
 *
 *      LW0073_CTRL_DPU_UCODE_STATE_LOADED
 *        This value indicates that the DPU ucode has been loaded but the DPU 
 *        has not yet been started. The DPU is not accepting commands at this 
 *        point.
 *
 *      LW0073_CTRL_DPU_UCODE_STATE_RUNNING
 *        This value indicates that the DPU ucode has been loaded and that the 
 *        DPU is lwrrently exelwting its bootstrapping process. The DPU is not 
 *        accepting commands at this point.
 *
 *      LW0073_CTRL_DPU_UCODE_STATE_READY
 *        This value indicates that the DPU is fully bootstrapped and ready to 
 *        accept and process commands.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0073_CTRL_CMD_DPU_UCODE_STATE (0x731501U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DPU_INTERFACE_ID << 8) | LW0073_CTRL_DPU_UCODE_STATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DPU_UCODE_STATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0073_CTRL_DPU_UCODE_STATE_PARAMS {
    LwU32 ucodeState;
} LW0073_CTRL_DPU_UCODE_STATE_PARAMS;

#define LW0073_CTRL_DPU_UCODE_STATE_NONE    (0x00000000U)
#define LW0073_CTRL_DPU_UCODE_STATE_LOADED  (0x00000001U)
#define LW0073_CTRL_DPU_UCODE_STATE_RUNNING (0x00000002U)
#define LW0073_CTRL_DPU_UCODE_STATE_READY   (0x00000003U)

/*
 * LW0073_CTRL_DPU_SURFACE_INFO
 *
 *  hMemory
 *    The memory handle associated with the surface being described.
 *
 *  offset
 *    A DPU surface may be a subregion of a larger allocation.  This offset
 *    marks the start of the surface relative to the start of the memory
 *    allocation.
 *
 *  size
 *    Used in conjunction with the offset (above) to mark the end of the 
 *    surface.
 */
typedef struct LW0073_CTRL_DPU_SURFACE_INFO {
    LwHandle hMemory;
    LwU32    offset;
    LwU32    size;
} LW0073_CTRL_DPU_SURFACE_INFO;

/*
 * LW0073_CTRL_CMD_DPU_SEND_CMD
 *
 * The DPU interface for submitting (non-blocking) a command to the DPU HW.
 * This interface is also used for setting the callback that will occur when
 * when command has completed.  The client must specify a callback function,
 * any private arguments that should be passed in the callback as well as all
 * necessary pre-allocated buffers for storing the command message.
 *
 *  cmd
 *    Represents the actual buffer containing raw command data processed by the
 *    DPU.  Refer to RM internal documentation for details of the command 
 *    format.
 *
 *  msg
 *    Represents the actual buffer that will be filled in response to exelwtion
 *    of the command (ie. the command's output).  It is the caller's
 *    responsibility to initialize this buffer.
 *
 *  queueId
 *    The logical identifier for the command queue this command is destined
 *    for.  Legal values for this parameter include:
 *      DPU_RM_CMDQ_LOG_ID 
 *        This is the logical queue for RM to send commands to DPU.  In the
 *        most case, the client should use this queue.   
 *
 *      DPU_PMU_CMDQ_LOG_ID  
 *        This is the logical queue for PMU to send commands to DPU.  The client
 *        should use this queue only for test or special purpose.
 *
 *  seqDesc
 *    [output] Will populated with a unique identifier/descriptor that
 *    identifies this command submission.  This descriptor may be used to
 *    actively query/poll for command status.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_DPU_SEND_CMD (0x731502U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DPU_INTERFACE_ID << 8) | LW0073_CTRL_DPU_SEND_CMD_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DPU_SEND_CMD_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0073_CTRL_DPU_SEND_CMD_PARAMS {
    LW0073_CTRL_DPU_SURFACE_INFO cmd;
    LW0073_CTRL_DPU_SURFACE_INFO msg;
    LwU32                        queueId;
    LwU32                        seqDesc;
} LW0073_CTRL_DPU_SEND_CMD_PARAMS;


/*
 * LW0073_CTRL_CMD_DPU_CMD_STATUS
 *
 * This command checks on the status of a command.
 *
 *  seqDesc
 *    The unique identifier/descriptor for the command that was returned when
 *    the command was first submitted.
 *
 *  seqStatus
 *    This parameter return the command's status.  Legal values for this 
 *    parameter include:
 *      LW0073_CTRL_DPU_CMD_STATUS_NONE
 *        This value indicates that a command has never submitted to the DPU 
 *        that matches the sequence descriptor provided.
 *
 *      LW0073_CTRL_DPU_CMD_STATUS_RUNNING
 *        This value indicates that the command has been issued and is still 
 *        exelwting on the DPU
 *
 *      LW0073_CTRL_DPU_CMD_STATUS_DONE
 *        This value indicates that the command was issued and has completed 
 *        exelwtion.
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_ILWALID_PARAM_STRUCT
 *  LW_ERR_INSUFFICIENT_RESOURCES
 *  LW_ERR_ILWALID_STATE
 *  LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_DPU_CMD_STATUS (0x731503U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DPU_INTERFACE_ID << 8) | LW0073_CTRL_DPU_CMD_STATUS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DPU_CMD_STATUS_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0073_CTRL_DPU_CMD_STATUS_PARAMS {
    LwU32 seqDesc;
    LwU32 seqStatus;
} LW0073_CTRL_DPU_CMD_STATUS_PARAMS;

#define LW0073_CTRL_DPU_CMD_STATUS_NONE    (0x00000000U)
#define LW0073_CTRL_DPU_CMD_STATUS_RUNNING (0x00000001U)
#define LW0073_CTRL_DPU_CMD_STATUS_DONE    (0x00000002U)

/* _ctrl0073dpu_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

