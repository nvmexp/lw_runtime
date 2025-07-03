/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2014-2016 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlb6b9.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* MAXWELL_SEC2 control commands and parameters */

#define LWB6B9_CTRL_CMD(cat,idx) \
    LWXXXX_CTRL_CMD(0xB6B9, LWB6B9_CTRL_##cat, idx)

/* SEC2 command categories (6bits) */
#define LWB6B9_CTRL_RESERVED             (0x00)
#define LWB6B9_CTRL_SEC2                 (0x01)

/*
 * LWB6B9_CTRL_CMD_SEC2_UCODE_STATE
 *
 * This command is used to retrieve the internal state of the SEC2 ucode.  It
 * may be used to determine when the SEC2 is ready to accept and process
 * commands.
 *
 *  ucodeState
 *    This parameter returns the internal SEC2 ucode state.  Legal values for 
 *    this parameter include:
 *      LWB6B9_CTRL_SEC2_UCODE_STATE_NONE
 *        This value indicates that the SEC2 ucode has not been loaded (ie. the 
 *        SEC2 has not been bootstrapped).  The SEC2 is not accepting commands at
 *        this point.
 *
 *      LWB6B9_CTRL_SEC2_UCODE_STATE_LOADED
 *        This value indicates that the SEC2 ucode has been loaded but the SEC2 
 *        has not yet been started. The SEC2 is not accepting commands at this 
 *        point.
 *
 *      LWB6B9_CTRL_SEC2_UCODE_STATE_RUNNING
 *        This value indicates that the SEC2 ucode has been loaded and that the 
 *        SEC2 is lwrrently exelwting its bootstrapping process. The SEC2 is not 
 *        accepting commands at this point.
 *
 *      LWB6B9_CTRL_SEC2_UCODE_STATE_READY
 *        This value indicates that the SEC2 is fully bootstrapped and ready to 
 *        accept and process commands.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LWB6B9_CTRL_CMD_SEC2_UCODE_STATE (0xb6b90101) /* finn: Evaluated from "(FINN_MAXWELL_SEC2_SEC2_INTERFACE_ID << 8) | LWB6B9_CTRL_SEC2_UCODE_STATE_PARAMS_MESSAGE_ID" */

#define LWB6B9_CTRL_SEC2_UCODE_STATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWB6B9_CTRL_SEC2_UCODE_STATE_PARAMS {
    LwU32 ucodeState;
} LWB6B9_CTRL_SEC2_UCODE_STATE_PARAMS;

#define LWB6B9_CTRL_SEC2_UCODE_STATE_NONE    (0x00000000)
#define LWB6B9_CTRL_SEC2_UCODE_STATE_LOADED  (0x00000001)
#define LWB6B9_CTRL_SEC2_UCODE_STATE_RUNNING (0x00000002)
#define LWB6B9_CTRL_SEC2_UCODE_STATE_READY   (0x00000003)

/*
 * LWB6B9_CTRL_SEC2_SURFACE_INFO
 *
 *  hMemory
 *    The memory handle associated with the surface being described.
 *
 *  offset
 *    A SEC2 surface may be a subregion of a larger allocation.  This offset
 *    marks the start of the surface relative to the start of the memory
 *    allocation.
 *
 *  size
 *    Used in conjunction with the offset (above) to mark the end of the 
 *    surface.
 */
typedef struct LWB6B9_CTRL_SEC2_SURFACE_INFO {
    LwHandle hMemory;
    LwU32    offset;
    LwU32    size;
} LWB6B9_CTRL_SEC2_SURFACE_INFO;

/*
 * LWB6B9_CTRL_CMD_SEC2_SEND_CMD
 *
 * The SEC2 interface for submitting (non-blocking) a command to the SEC2 HW.
 * This interface is also used for setting the callback that will occur when
 * when command has completed.  The client must specify a callback function,
 * any private arguments that should be passed in the callback as well as all
 * necessary pre-allocated buffers for storing the command message.
 *
 *  cmd
 *    Represents the actual buffer containing raw command data processed by the
 *    SEC2.  Refer to RM internal documentation for details of the command 
 *    format.
 *
 *  msg
 *    Represents the actual buffer that will be filled in response to exelwtion
 *    of the command (ie. the command's output).  It is the caller's
 *    responsibility to initialize this buffer.
 * 
 *  payloadIn
 *    Represents an optional (may be NULL) buffer of data that the command is
 *    dependent upon for processing.  This is indended to be used for large
 *    buffers, needed by the command, that should not be queued in the SEC2
 *    command queues (due to size).
 *
 *  payloadOut
 *    Represents an optional (may be NULL) buffer that may be filled when a
 *    command produces a large amount of output.
 *
 *  inAllocOffset
 *    Represents the offset into 'cmd' where the input payload allocation
 *    structure begins.  When an input payload ('payloadIn') is not specified,
 *    this parameter should be set to zero.
 *
 *  outAllocOffset
 *    Same as 'inAllocOffset' only pertaining to 'payloadOut' 
 * 
 *  queueId
 *    The logical identifier for the command queue this command is destined
 *    for.  Legal values for this parameter include:
 *      SEC2_RM_CMDQ_LOG_ID 
 *        This is the logical queue for RM to send commands to SEC2.  In the
 *        most case, the client should use this queue.   
 *
 *      SEC2_PMU_CMDQ_LOG_ID  
 *        This is the logical queue for PMU to send commands to SEC2.  The client
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
#define LWB6B9_CTRL_CMD_SEC2_SEND_CMD (0xb6b90102) /* finn: Evaluated from "(FINN_MAXWELL_SEC2_SEC2_INTERFACE_ID << 8) | LWB6B9_CTRL_SEC2_SEND_CMD_PARAMS_MESSAGE_ID" */

#define LWB6B9_CTRL_SEC2_SEND_CMD_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWB6B9_CTRL_SEC2_SEND_CMD_PARAMS {
    LWB6B9_CTRL_SEC2_SURFACE_INFO cmd;
    LWB6B9_CTRL_SEC2_SURFACE_INFO msg;
    LWB6B9_CTRL_SEC2_SURFACE_INFO payloadIn;
    LWB6B9_CTRL_SEC2_SURFACE_INFO payloadOut;
    LwU32                         inAllocOffset;
    LwU32                         outAllocOffset;
    LwU32                         queueId;
    LwU32                         seqDesc;
} LWB6B9_CTRL_SEC2_SEND_CMD_PARAMS;


/*
 * LWB6B9_CTRL_CMD_SEC2_CMD_STATUS
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
 *      LWB6B9_CTRL_SEC2_CMD_STATUS_NONE
 *        This value indicates that a command has never submitted to the SEC2 
 *        that matches the sequence descriptor provided.
 *
 *      LWB6B9_CTRL_SEC2_CMD_STATUS_RUNNING
 *        This value indicates that the command has been issued and is still 
 *        exelwting on the SEC2
 *
 *      LWB6B9_CTRL_SEC2_CMD_STATUS_DONE
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
#define LWB6B9_CTRL_CMD_SEC2_CMD_STATUS (0xb6b90103) /* finn: Evaluated from "(FINN_MAXWELL_SEC2_SEC2_INTERFACE_ID << 8) | LWB6B9_CTRL_SEC2_CMD_STATUS_PARAMS_MESSAGE_ID" */

#define LWB6B9_CTRL_SEC2_CMD_STATUS_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWB6B9_CTRL_SEC2_CMD_STATUS_PARAMS {
    LwU32 seqDesc;
    LwU32 seqStatus;
} LWB6B9_CTRL_SEC2_CMD_STATUS_PARAMS;

#define LWB6B9_CTRL_SEC2_CMD_STATUS_NONE    (0x00000000)
#define LWB6B9_CTRL_SEC2_CMD_STATUS_RUNNING (0x00000001)
#define LWB6B9_CTRL_SEC2_CMD_STATUS_DONE    (0x00000002)

/*
 * LWB6B9_CTRL_CMD_SEC2_TEST_SUPPORTED
 *
 * Check if a test is supported on the SEC2
 *
 * testId
 *     The SEC2 TEST cmd ID for the test
 *
 * bTestSupported
 *     Whether the given TEST cmd ID is supported on the SEC2
 *
 * Possible status values returned are:
 *  LW_OK
 *  LW_ERR_NOT_SUPPORTED
 */
#define LWB6B9_CTRL_CMD_SEC2_TEST_SUPPORTED (0xb6b90104) /* finn: Evaluated from "(FINN_MAXWELL_SEC2_SEC2_INTERFACE_ID << 8) | LWB6B9_CTRL_SEC2_TEST_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LWB6B9_CTRL_SEC2_TEST_SUPPORTED_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWB6B9_CTRL_SEC2_TEST_SUPPORTED_PARAMS {
    LwU8   testId;
    LwBool bTestSupported;
} LWB6B9_CTRL_SEC2_TEST_SUPPORTED_PARAMS;

/* _ctrlb6b9.h_ */
