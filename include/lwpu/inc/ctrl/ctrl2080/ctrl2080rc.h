/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080rc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/*
 * LW2080_CTRL_CMD_RC_READ_VIRTUAL_MEM
 *
 * This command returns data read from the specified virtual memory address for
 * the associated subdevice.
 *
 *   hChannel
 *      This parameter specifies the channel object handle from which the virtual
 *      memory range applies.
 *   virtAddress
 *      This parameter specifies the GPU base virtual memory address from which data should
 *      be read.  The amount of data read is specified by the bufferSize parameter.
 *   bufferPtr
 *      This parameter specifies the buffer address in the caller's address space into which
 *      the data is to be returned.  The address must be aligned on an 8-byte boundary.
 *      The buffer must be at least as big as the value specified bufferSize parameter (in bytes).
 *   bufferSize
 *      This parameter specifies the size of the buffer referenced by the bufferPtr parameter.
 *      This parameter also indicates the total number of bytes to be returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_XLATE
 */
#define LW2080_CTRL_RC_READ_VIRTUAL_MEM_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_RC_READ_VIRTUAL_MEM_PARAMS {
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 virtAddress, 8);
    LW_DECLARE_ALIGNED(LwP64 bufferPtr, 8);
    LwU32    bufferSize;
} LW2080_CTRL_RC_READ_VIRTUAL_MEM_PARAMS;

#define LW2080_CTRL_CMD_RC_READ_VIRTUAL_MEM (0x20802204) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | LW2080_CTRL_RC_READ_VIRTUAL_MEM_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_RC_GET_ERROR_COUNT
 *
 * This command returns the number of RC errors.
 *
 *   errorCount
 *      Number of RC errors.
 *
 * Note: If SMC is enabled, mig/monitor capability must be acquired to query
 * aggregate information. Otherwise, the control call returns
 * LW_ERR_INSUFFICIENT_PERMISSIONS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_PERMISSIONS.
 */
#define LW2080_CTRL_RC_GET_ERROR_COUNT_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_RC_GET_ERROR_COUNT_PARAMS {
    LwU32 errorCount;
} LW2080_CTRL_RC_GET_ERROR_COUNT_PARAMS;

#define LW2080_CTRL_CMD_RC_GET_ERROR_COUNT      (0x20802205) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | LW2080_CTRL_RC_GET_ERROR_COUNT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_RC_ERROR_PARAMS_BUFFER_SIZE (0x2000) /* finn: Evaluated from "(8 * 1024)" */

#define LW2080_CTRL_CMD_RC_GET_ERROR            (0x20802206) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0x6" */

/*
 * LW2080_CTRL_CMD_RC_GET_ERROR_V2
 *
 * This command returns an error element in the RC error list.
 *
 *   whichBuffer
 *      Which Error to return (0 is oldest)
 *   outputRecordSize
 *      Output Size of Buffer -- Zero if error record doesn't exist
 *   recordBuffer
 *      buffer
 *
 * Note: If SMC is enabled, mig/monitor capability must be acquired to query
 * aggregate information. Otherwise, the control call returns
 * LW_ERR_INSUFFICIENT_PERMISSIONS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_PERMISSIONS.
 *
 */

#define LW2080_CTRL_RC_GET_ERROR_V2_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_RC_GET_ERROR_V2_PARAMS {

    LwU32 whichBuffer;                   // [IN]   - which error to return (0 is oldest)
    LwU32 outputRecordSize;              // [OUT]
    LwU8  recordBuffer[LW2080_CTRL_RC_ERROR_PARAMS_BUFFER_SIZE];
} LW2080_CTRL_RC_GET_ERROR_V2_PARAMS;

#define LW2080_CTRL_CMD_RC_GET_ERROR_V2            (0x20802213) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | LW2080_CTRL_RC_GET_ERROR_V2_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_RC_SET_CLEAN_ERROR_HISTORY
 *
 * This command cleans error history.
 *
 * This command has no input parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW2080_CTRL_CMD_RC_SET_CLEAN_ERROR_HISTORY (0x20802207) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0x7" */

/*
 * LW2080_CTRL_CMD_RC_GET_WATCHDOG_INFO
 *
 * This command returns information about the RC watchdog.
 *
 *   watchdogStatusFlags
 *     This output parameter is a combination of one or more of the following:
 *
 *       LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_NONE
 *         This is the value of watchdogStatusFlags if no flags are set.
 *
 *       LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_DISABLED
 *         This means that the watchdog is disabled.
 *
 *       LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_RUNNING
 *         This means that the watchdog is running.
 *
 *       LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_INITIALIZED
 *         This means that the watchdog has been initialized.
 *
 *     A typical result would be either "running and initialized", or
 *     "disabled".  However, "initialized, but not running, and not disabled"
 *     is also quite reasonable (if the computer is hibernating, for example).
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_RC_GET_WATCHDOG_INFO_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_RC_GET_WATCHDOG_INFO_PARAMS {
    LwU32 watchdogStatusFlags;
} LW2080_CTRL_RC_GET_WATCHDOG_INFO_PARAMS;

#define LW2080_CTRL_CMD_RC_GET_WATCHDOG_INFO               (0x20802209) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | LW2080_CTRL_RC_GET_WATCHDOG_INFO_PARAMS_MESSAGE_ID" */

/* valid values for watchdogStatusFlags */
#define LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_NONE        (0x00000000)
#define LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_DISABLED    (0x00000001)
#define LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_RUNNING     (0x00000002)
#define LW2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_INITIALIZED (0x00000004)

/*
 * LW2080_CTRL_CMD_RC_DISABLE_WATCHDOG
 *
 * This command disables the RC watchdog, if possible.
 * If, however, another RM client has already explicitly (via LW2080 call) enabled
 * the RC watchdog, then this method returns LW_ERR_STATE_IN_USE.
 *
 * This command, if successful, will prevent other clients from enabling the
 * watchdog until the calling RM client releases its request with
 * LW2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS or frees its LW20_SUBDEVICE.
 *
 * See LW2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG for disabling the watchdog
 * without preventing other clients from enabling it.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_RC_DISABLE_WATCHDOG                (0x2080220a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0xA" */

/*
 * LW2080_CTRL_CMD_RC_ENABLE_WATCHDOG
 *
 * This command enables the RC watchdog, if possible.
 * If, however, another RM client has already explicitly (via LW2080 call) disabled
 * the RC watchdog, then this method returns LW_ERR_STATE_IN_USE.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_RC_ENABLE_WATCHDOG                 (0x2080220b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0xB" */

/*
 * LW2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS
 *
 * This command releases all of the RM client's outstanding requests to enable
 * or disable the watchdog.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS       (0x2080220c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0xC" */

/*
 * LW2080_CTRL_CMD_SET_RC_RECOVERY/LW2080_CTRL_CMD_GET_RC_RECOVERY
 *
 * This command disables/enables RC recovery.
 *
 *   rcEnable
 *       LW2080_CTRL_CMD_SET_RC_RECOVERY_DISABLED
 *         Disable robust channel recovery.
 *
 *       LW2080_CTRL_CMD_SET_RC_RECOVERY_ENABLED
 *         Enable robust channel recovery with default breakpoint handling.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
typedef struct LW2080_CTRL_CMD_RC_RECOVERY_PARAMS {
    LwU32 rcEnable;
} LW2080_CTRL_CMD_RC_RECOVERY_PARAMS;

#define LW2080_CTRL_CMD_SET_RC_RECOVERY       (0x2080220d) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0xD" */

#define LW2080_CTRL_CMD_GET_RC_RECOVERY       (0x2080220e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0xE" */

/* valid values for rcEnable */
#define LW2080_CTRL_CMD_RC_RECOVERY_DISABLED  (0x00000000)
#define LW2080_CTRL_CMD_RC_RECOVERY_ENABLED   (0x00000001)

/*
 * LW2080_CTRL_CMD_TDR_SET_TIMEOUT_STATE
 *
 * This command can be used to set TDR timeout state.
 *
 * It can be used to indicate that a timeout has oclwrred and that a GPU
 * reset will start. It can also be used to indicate that the reset has
 * completed along with the corresponding complition status.
 *
 *   cmd
 *    This parameter is used to indicate the stage of the TDR recovery
 *    process. Legal values for this parameter are:
 *     LW2080_CTRL_TDR_SET_TIMEOUT_STATE_CMD_GPU_RESET_BEGIN
 *      This value indicates that TDR recovery is about to begin.
 *     LW2080_CTRL_TDR_SET_TIMEOUT_STATE_CMD_GPU_RESET_END
 *      This value indicates that TDR recovery has completed.
 *
 *   status
 *     This parameter is valid when the cmd parameter is set to
 *     LW2080_CTRL_TDR_SET_TIMEOUT_STATE_CMD_GPU_RESET_END. It is used
 *     to specify the completion status of the TDR recovery. Legal
 *     values for this parameter include:
 *       LW2080_CTRL_TDR_SET_TIMEOUT_STATE_STATUS_FAIL
 *         This value indicates the recovery failed.
 *       LW2080_CTRL_TDR_SET_TIMEOUT_STATE_STATUS_SUCCESS
 *         This value indicates the recovery succeeded.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_TDR_SET_TIMEOUT_STATE (0x2080220f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | LW2080_CTRL_TDR_SET_TIMEOUT_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_TDR_SET_TIMEOUT_STATE_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW2080_CTRL_TDR_SET_TIMEOUT_STATE_PARAMS {
    LwU32 cmd;
    LwS32 status;
} LW2080_CTRL_TDR_SET_TIMEOUT_STATE_PARAMS;

/* valid cmd values */
#define LW2080_CTRL_TDR_SET_TIMEOUT_STATE_CMD_GPU_RESET_BEGIN (0x00000000)
#define LW2080_CTRL_TDR_SET_TIMEOUT_STATE_CMD_GPU_RESET_END   (0x00000001)

/* valid status values */
#define LW2080_CTRL_TDR_SET_TIMEOUT_STATE_STATUS_SUCCESS      (0x00000000)
#define LW2080_CTRL_TDR_SET_TIMEOUT_STATE_STATUS_FAIL         (0x00000001)

/*
 * LW2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG
 *
 * This command disables the RC watchdog, similarly to how
 * LW2080_CTRL_CMD_RC_DISABLE_WATCHDOG does. However, unlike that command, this
 * command will not prevent another RM client from explicitly enabling the RC
 * watchdog with LW2080_CTRL_CMD_RC_ENABLE_WATCHDOG.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG              (0x20802210) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0x10" */

/*
 * LW2080_CTRL_CMD_GET_RC_INFO/LW2080_CTRL_CMD_SET_RC_INFO
 *
 * This command can be used to set robust channel parameters.
 *
 *   rcMode
 *       LW2080_CTRL_CMD_SET_RC_INFO_MODE_DISABLE
 *         Disable robust channel operation.
 *
 *       LW2080_CTRL_CMD_SET_RC_INFO_MODE_ENABLE
 *         Enable robust channel operation.
 *
 *   rcBreak
 *       LW2080_CTRL_CMD_SET_RC_INFO_BREAK_DISABLE
 *         Disable breakpoint handling during robust channel operation.
 *
 *       LW2080_CTRL_CMD_SET_RC_INFO_BREAK_ENABLE
 *         Enable breakpoint handling during robust channel operation.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
typedef struct LW2080_CTRL_CMD_RC_INFO_PARAMS {
    LwU32 rcMode;
    LwU32 rcBreak;
} LW2080_CTRL_CMD_RC_INFO_PARAMS;

#define LW2080_CTRL_CMD_SET_RC_INFO           (0x20802211) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0x11" */

#define LW2080_CTRL_CMD_GET_RC_INFO           (0x20802212) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_RC_INTERFACE_ID << 8) | 0x12" */

/* valid rcMode values */
#define LW2080_CTRL_CMD_RC_INFO_MODE_DISABLE  (0x00000000)
#define LW2080_CTRL_CMD_RC_INFO_MODE_ENABLE   (0x00000001)

/* valid rcBreak values */
#define LW2080_CTRL_CMD_RC_INFO_BREAK_DISABLE (0x00000000)
#define LW2080_CTRL_CMD_RC_INFO_BREAK_ENABLE  (0x00000001)

/* _ctrl2080rc_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

