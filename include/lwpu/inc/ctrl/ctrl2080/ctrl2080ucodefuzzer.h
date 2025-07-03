/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080ucodefuzzer.finn
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

/* Numerical ID for each ucode */
#define LW2080_UCODE_FUZZER_SEC2              1
#define LW2080_UCODE_FUZZER_PMU               2

#define LW2080_UCODE_FUZZER_MAX_CMD_SIZE      2048
#define LW2080_UCODE_FUZZER_MAX_PAYLOAD_SIZE  2048

/*
 * LW2080_CTRL_CMD_UCODE_FUZZER_RTOS_CMD
 *
 * This command is used to send commands to an RTOS ucode without
 * normal validation functions.
 *
 *  ucode
 *    This parameter corresponds to the ucode the command should be sent to.
 *    Legal values for this parameter are RTOS uCode identifiers (i.e.
 *    UCODE_FUZZER_SEC2).
 *  queueId
 *    This parameter specifies which queue the command should be sent to.
 *  sizeCmd
 *    The number of bytes of the command specified in cmd
 *  sizePayload
 *    The number of bytes of the payload specified in payload
 *  timeoutMs
 *    The timeout in milliseconds to wait for a command to complete
 *  cmd
 *    The full command data to send to the ucode
 *  payload
 *    The full payload data to send with the command
 *
 * Possible status values returned are (will vary based on ucode)
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_TIMEOUT
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW2080_CTRL_CMD_UCODE_FUZZER_RTOS_CMD (0x20803901) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_UCODE_FUZZER_INTERFACE_ID << 8) | LW2080_CTRL_UCODE_FUZZER_RTOS_CMD_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_UCODE_FUZZER_RTOS_CMD_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_UCODE_FUZZER_RTOS_CMD_PARAMS {
    LwU32 ucode;
    LwU32 queueId;
    LwU32 sizeCmd;
    LwU32 sizePayload;
    LwU32 timeoutMs;
    LwU8  cmd[LW2080_UCODE_FUZZER_MAX_CMD_SIZE];
    LwU8  payload[LW2080_UCODE_FUZZER_MAX_PAYLOAD_SIZE];
} LW2080_CTRL_UCODE_FUZZER_RTOS_CMD_PARAMS;


/*
 * LW2080_CTRL_UCODE_FUZZER_SANITIZER_COV_CONTROL_PARAMS
 *
 * Parameters struct shared by the control calls
 * LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_GET_CONTROL and
 * LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_SET_CONTROL.
 */
typedef struct LW2080_CTRL_UCODE_FUZZER_SANITIZER_COV_CONTROL_PARAMS {
    LwU32  ucode;
    LwU32  used;
    LwU32  missed;
    LwBool bEnabled;
} LW2080_CTRL_UCODE_FUZZER_SANITIZER_COV_CONTROL_PARAMS;

/*
 * LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_GET_CONTROL
 *
 * Retrieves the status of SanitizerCoverage run-time data gathering for the
 * given ucode (such as whether it is enabled or not).
 *
 *   ucode     numeric id of the desired ucode to target
 *   used      number of elements used in the ucode's DMEM buffer
 *   missed    number of callbacks missed due to the ucode's DMEM buffer
 *             being full
 *   bEnabled  whether the ucode's SanitizerCoverage run-time data gathering
 *             is enabled (LW_TRUE) or not (LW_FALSE)
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_GET_CONTROL (0x20803902) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_UCODE_FUZZER_INTERFACE_ID << 8) | 0x2" */

/*
 * LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_SET_CONTROL
 *
 * Adjusts the status of SanitizerCoverage run-time data gathering for the
 * given ucode (such as whether it is enabled or not).
 *
 *   ucode     numeric id of the desired ucode to target
 *   used      set to 0 to forcibly empty the ucode's DMEM buffer
 *             (all other values ignored)
 *   missed    set to 0 to reset the ucode's counter of callbacks missed
 *             (all other values ignored)
 *   bEnabled  whether to enable (LW_TRUE) or disable (LW_FALSE)
 *             the ucode's run-time SanitizerCoverage data gathering
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_SET_CONTROL (0x20803903) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_UCODE_FUZZER_INTERFACE_ID << 8) | 0x3" */

/*
 * LW2080_UCODE_FUZZER_SANITIZER_COV_RPC_MAX_ELEMENTS_*
 *
 * Maximum number of 64-bit elements that can be retrieved by one RPC call
 * (potentially differs per-ucode). These are used to size the buffer in the
 * respective ucode RPC parameters.
 */
#define LW2080_UCODE_FUZZER_SANITIZER_COV_RPC_MAX_ELEMENTS_PMU 128

/*
 * LW2080_UCODE_FUZZER_SANITIZER_COV_RPC_MAX_BYTES_*
 *
 * Eight times (for eight bytes per element) the corresponding 
 * LW2080_UCODE_FUZZER_SANITIZER_COV_RPC_MAX_ELEMENTS_* macro.
 */
#define LW2080_UCODE_FUZZER_SANITIZER_COV_RPC_MAX_BYTES_PMU    (0x400) /* finn: Evaluated from "(LW2080_UCODE_FUZZER_SANITIZER_COV_RPC_MAX_ELEMENTS_PMU * 8)" */

/*
 * LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_DATA_GET
 *
 * Retrieves the run-time data gathered by SanitizerCoverage for the
 * given ucode.
 *
 *   pData        destination buffer to retrieve data into
 *   ucode        numeric id of the desired ucode to retrieve data from
 *   numElements  number of elements of SanitizerCoverage data to copy
 *                (and number of elements actually copied)
 *   bDone        will be LW_TRUE when all available data has been retrieved
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_UCODE_FUZZER_SANITIZER_COV_DATA_GET    (0x20803904) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_UCODE_FUZZER_INTERFACE_ID << 8) | LW2080_CTRL_UCODE_FUZZER_SANITIZER_COV_DATA_GET_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_UCODE_FUZZER_SANITIZER_COV_DATA_GET_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_UCODE_FUZZER_SANITIZER_COV_DATA_GET_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 *pData, 8);
    LwU32  ucode;
    LwU32  numElements;
    LwBool bDone;
} LW2080_CTRL_UCODE_FUZZER_SANITIZER_COV_DATA_GET_PARAMS;

/* _ctrl2080ucodefuzzer_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

