/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlc310.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/*
 * VOLTA_GSP control commands and parameters.  This class is used for both
 * GSPLite (in Volta) and GSP (in Turing_and_later).
 */

#define LWC310_CTRL_CMD(cat,idx) \
    LWXXXX_CTRL_CMD(0xC310, LWC310_CTRL_##cat, idx)

/* LWC310 command categories (6bits) */
#define LWC310_CTRL_RESERVED            (0x00)
#define LWC310_CTRL_GSP                 (0x01)

/*
 * LWC310_CTRL_CMD_GSP_UCODE_STATE
 *
 * This command is used to retrieve the internal state of the GSP ucode.  It
 * may be used to determine when the GSP is ready to accept and process
 * commands.
 *
 *  ucodeState
 *    This parameter returns the internal GSP ucode state.  Legal values for 
 *    this parameter include:
 *      LWC310_CTRL_GSP_UCODE_STATE_NONE
 *        This value indicates that the GSP ucode has not been loaded (ie. the 
 *        GSP has not been bootstrapped).  The GSP is not accepting commands at
 *        this point.
 *
 *      LWC310_CTRL_GSP_UCODE_STATE_LOADED
 *        This value indicates that the GSP ucode has been loaded but the GSP 
 *        has not yet been started. The GSP is not accepting commands at this 
 *        point.
 *
 *      LWC310_CTRL_GSP_UCODE_STATE_RUNNING
 *        This value indicates that the GSP ucode has been loaded and that the 
 *        GSP is lwrrently exelwting its bootstrapping process. The GSP is not 
 *        accepting commands at this point.
 *
 *      LWC310_CTRL_GSP_UCODE_STATE_READY
 *        This value indicates that the GSP is fully bootstrapped and ready to 
 *        accept and process commands.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LWC310_CTRL_CMD_GSP_UCODE_STATE (0xc3100101) /* finn: Evaluated from "(FINN_VOLTA_GSP_GSP_INTERFACE_ID << 8) | LWC310_CTRL_GSP_UCODE_STATE_PARAMS_MESSAGE_ID" */

#define LWC310_CTRL_GSP_UCODE_STATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC310_CTRL_GSP_UCODE_STATE_PARAMS {
    LwU32 ucodeState;
} LWC310_CTRL_GSP_UCODE_STATE_PARAMS;

#define LWC310_CTRL_GSP_UCODE_STATE_NONE    (0x00000000)
#define LWC310_CTRL_GSP_UCODE_STATE_LOADED  (0x00000001)
#define LWC310_CTRL_GSP_UCODE_STATE_RUNNING (0x00000002)
#define LWC310_CTRL_GSP_UCODE_STATE_READY   (0x00000003)

/*
 * LWC310_CTRL_GSP_SURFACE_INFO
 *
 *  hMemory
 *    The memory handle associated with the surface being described.
 *
 *  offset
 *    A GSP surface may be a subregion of a larger allocation.  This offset
 *    marks the start of the surface relative to the start of the memory
 *    allocation.
 *
 *  size
 *    Used in conjunction with the offset (above) to mark the end of the 
 *    surface.
 */
typedef struct LWC310_CTRL_GSP_SURFACE_INFO {
    LwHandle hMemory;
    LwU32    offset;
    LwU32    size;
} LWC310_CTRL_GSP_SURFACE_INFO;

/*
 * LWC310_CTRL_CMD_GSP_SEND_CMD
 *
 * The GSP interface for submitting (non-blocking) a command to the GSP HW.
 * This interface is also used for setting the callback that will occur when
 * when command has completed.  The client must specify a callback function,
 * any private arguments that should be passed in the callback as well as all
 * necessary pre-allocated buffers for storing the command message.
 *
 *  cmd
 *    Represents the actual buffer containing raw command data processed by the
 *    GSP.  Refer to RM internal documentation for details of the command 
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
 *      GSP_RM_CMDQ_LOG_ID 
 *        This is the logical queue for RM to send commands to GSP.  In the
 *        most case, the client should use this queue.   
 *
 *      GSP_PMU_CMDQ_LOG_ID  
 *        This is the logical queue for PMU to send commands to GSP.  The client
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
#define LWC310_CTRL_CMD_GSP_SEND_CMD                    (0xc3100102) /* finn: Evaluated from "(FINN_VOLTA_GSP_GSP_INTERFACE_ID << 8) | LWC310_CTRL_GSP_SEND_CMD_PARAMS_MESSAGE_ID" */

#define LWC310_CTRL_GSP_SEND_CMD_FLAGS_ROUTE_CC          0:0
#define LWC310_CTRL_GSP_SEND_CMD_FLAGS_ROUTE_CC_DISABLE (0x00000000)
#define LWC310_CTRL_GSP_SEND_CMD_FLAGS_ROUTE_CC_ENABLE  (0x00000001)

#define LWC310_CTRL_GSP_SEND_CMD_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC310_CTRL_GSP_SEND_CMD_PARAMS {
    LWC310_CTRL_GSP_SURFACE_INFO cmd;
    LWC310_CTRL_GSP_SURFACE_INFO msg;
    LwU32                        queueId;
    LwU32                        seqDesc;
    LwHandle                     hMemory;
    LwU32                        reqMsgSize;
    LwU32                        rspBufferSize;
    LwU32                        flags;
} LWC310_CTRL_GSP_SEND_CMD_PARAMS;

/*
 * LWC310_CTRL_CMD_GSP_CMD_STATUS
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
 *      LWC310_CTRL_GSP_CMD_STATUS_NONE
 *        This value indicates that a command has never submitted to the GSP 
 *        that matches the sequence descriptor provided.
 *
 *      LWC310_CTRL_GSP_CMD_STATUS_RUNNING
 *        This value indicates that the command has been issued and is still 
 *        exelwting on the GSP
 *
 *      LWC310_CTRL_GSP_CMD_STATUS_DONE
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
#define LWC310_CTRL_CMD_GSP_CMD_STATUS (0xc3100103) /* finn: Evaluated from "(FINN_VOLTA_GSP_GSP_INTERFACE_ID << 8) | LWC310_CTRL_GSP_CMD_STATUS_PARAMS_MESSAGE_ID" */

#define LWC310_CTRL_GSP_CMD_STATUS_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC310_CTRL_GSP_CMD_STATUS_PARAMS {
    LwU32 seqDesc;
    LwU32 seqStatus;
} LWC310_CTRL_GSP_CMD_STATUS_PARAMS;

#define LWC310_CTRL_GSP_CMD_STATUS_NONE    (0x00000000)
#define LWC310_CTRL_GSP_CMD_STATUS_RUNNING (0x00000001)
#define LWC310_CTRL_GSP_CMD_STATUS_DONE    (0x00000002)

/*
 * LWC310_CTRL_CMD_GSP_TEST_SUPPORTED
 *
 * Check if a test is supported on the GSP
 *
 * testId
 *     The GSP TEST cmd ID for the test
 *
 * bTestSupported
 *     Whether the given TEST cmd ID is supported on the GSP
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_NOT_SUPPORTED
 */
#define LWC310_CTRL_CMD_GSP_TEST_SUPPORTED (0xc3100104) /* finn: Evaluated from "(FINN_VOLTA_GSP_GSP_INTERFACE_ID << 8) | LWC310_CTRL_GSP_TEST_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LWC310_CTRL_GSP_TEST_SUPPORTED_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWC310_CTRL_GSP_TEST_SUPPORTED_PARAMS {
    LwU8   testId;
    LwBool bTestSupported;
} LWC310_CTRL_GSP_TEST_SUPPORTED_PARAMS;

/* _ctrlc310_h_ */
