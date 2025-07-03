/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl5070/ctrl5070seq.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl5070/ctrl5070base.h"

/*
 * LW5070_CTRL_CMD_GET_SOR_SEQ_CTL
 *
 * This command returns SOR sequencer's power up and down PCs and sequencer
 * program to be used for power up and dowm.
 *
 *      orNumber
 *          The OR number for which the seq ctrls are to be modified.
 *
 *      puPcAlt
 *          Alternate power up PC.
 *
 *      pdPc
 *          Power down PC.
 *
 *      pdPcAlt
 *          Alternate power down PC.
 *
 *      normalStart
 *          Whether normal mode is using normal or alt PC
 *
 *      safeStart
 *          Whether safe mode is using normal or alt PC
 *
 *      normalState
 *          Whether normal state is PD or PU.
 *
 *      safeState
 *          Whether safe state is PD or PU.
 *
 *      flags
 *          There is only one flag defined lwrrently
 *              1. GET_SEQ_PROG: Whether or not current seq program must be
 *                 return back. Caller should set this to _YES to read the
 *                 current seq program.
 *
 *      seqProgram
 *          The sequencer program consisting of power up and down sequences.
 *          For LW50, this consists of 16 DWORDS. The program is
 *          relevant only when GET_SEQ_PROG flags is set to _YES.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL                         (0x50700301U) /* finn: Evaluated from "(FINN_LW50_DISPLAY_SEQ_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_PU_PC_ALT_VALUE                     3:0

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_PD_PC_VALUE                         3:0

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_PD_PC_ALT_VALUE                     3:0

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_NORMAL_START_VAL                    0:0
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_NORMAL_START_VAL_NORMAL (0x00000000U)
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_NORMAL_START_VAL_ALT    (0x00000001U)

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SAFE_START_VAL                      0:0
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SAFE_START_VAL_NORMAL   (0x00000000U)
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SAFE_START_VAL_ALT      (0x00000001U)

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_NORMAL_STATE_VAL                    0:0
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_NORMAL_STATE_VAL_PD     (0x00000000U)
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_NORMAL_STATE_VAL_PU     (0x00000001U)

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SAFE_STATE_VAL                      0:0
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SAFE_STATE_VAL_PD       (0x00000000U)
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SAFE_STATE_VAL_PU       (0x00000001U)

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_FLAGS_GET_SEQ_PROG                  0:0
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_FLAGS_GET_SEQ_PROG_NO   (0x00000000U)
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_FLAGS_GET_SEQ_PROG_YES  (0x00000001U)

#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SEQ_PROG_SIZE           16U
#define LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       puPcAlt;
    LwU32                       pdPc;
    LwU32                       pdPcAlt;
    LwU32                       normalStart;
    LwU32                       safeStart;
    LwU32                       normalState;
    LwU32                       safeState;
    LwU32                       flags;
    LwU32                       seqProgram[LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_SEQ_PROG_SIZE];
} LW5070_CTRL_CMD_GET_SOR_SEQ_CTL_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_SOR_SEQ_CTL
 *
 * This command does the following in that order
 * (a) Loads a specified sequencer program for power up and down.
 * (b) Updates SOR sequencer's power up and down PCs, tells seq to SKIP
 * current wait for vsync and waits until sequencer actually SKIPs or halts
 * (see more below under SKIP_WAIT_FOR_VSYNC flag) and
 * (c) Update power settings (safe/normal start and state).
 *
 *      orNumber
 *          The OR number for which the seq ctrls are to be modified.
 *
 *      puPcAlt
 *          Alternate power up PC.
 *
 *      pdPc
 *          Power down PC.
 *
 *      pdPcAlt
 *          Alternate power down PC.
 *
 *      normalStart
 *          Whether normal mode should use normal or alt PC.
 *
 *      safeStart
 *          Whether safe mode should use normal or alt PC.
 *
 *      normalState
 *          Whether normal state should be PD or PU.
 *
 *      safeState
 *          Whether safe state should be PD or PU.
 *
 *      flags
 *          The following flags have been defined
 *              1. SKIP_WAIT_FOR_VSYNC: Should seq be forced to skip waiting
 *                 for vsync if it's lwrrently waiting on such an instruction.
 *                 If the current instruction doesn't have a wait for vsync,
 *                 SKIP will be applied to the next one and so on until
 *                 either sequencer halts or an instruction with a wait for
 *                 vsync is found. The call will block until seq halts or
 *                 SKIPs a wait for vsync.
 *              2. SEQ_PROG_PRESENT: Whether or not a new seq program has
 *                 been specified.
 *
 *      seqProgram
 *          The sequencer program consisting of power up and down sequences.
 *          For LW50, this consists of 16 DWORDS. The program is
 *          relevant only when SEQ_PROG_PRESENT flags is set to _YES.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL                               (0x50700302U) /* finn: Evaluated from "(FINN_LW50_DISPLAY_SEQ_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PU_PC_ALT_VALUE                     3:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PU_PC_ALT_SPECIFIED               31:31
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PU_PC_ALT_SPECIFIED_NO        (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PU_PC_ALT_SPECIFIED_YES       (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_VALUE                         3:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_SPECIFIED                   31:31
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_SPECIFIED_NO            (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_SPECIFIED_YES           (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_ALT_VALUE                     3:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_ALT_SPECIFIED               31:31
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_ALT_SPECIFIED_NO        (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PD_PC_ALT_SPECIFIED_YES       (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_START_VAL                    0:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_START_VAL_NORMAL       (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_START_VAL_ALT          (0x00000001U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_START_SPECIFIED            31:31
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_START_SPECIFIED_NO     (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_START_SPECIFIED_YES    (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_START_VAL                      0:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_START_VAL_NORMAL         (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_START_VAL_ALT            (0x00000001U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_START_SPECIFIED              31:31
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_START_SPECIFIED_NO       (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_START_SPECIFIED_YES      (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_STATE_VAL                    0:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_STATE_VAL_PD           (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_STATE_VAL_PU           (0x00000001U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_STATE_SPECIFIED            31:31
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_STATE_SPECIFIED_NO     (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_NORMAL_STATE_SPECIFIED_YES    (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_STATE_VAL                      0:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_STATE_VAL_PD             (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_STATE_VAL_PU             (0x00000001U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_STATE_SPECIFIED              31:31
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_STATE_SPECIFIED_NO       (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SAFE_STATE_SPECIFIED_YES      (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_FLAGS_SKIP_WAIT_FOR_VSYNC               0:0
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_FLAGS_SKIP_WAIT_FOR_VSYNC_NO  (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_FLAGS_SKIP_WAIT_FOR_VSYNC_YES (0x00000001U)

#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_FLAGS_SEQ_PROG_PRESENT                   1:1
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_FLAGS_SEQ_PROG_PRESENT_NO     (0x00000000U)
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_FLAGS_SEQ_PROG_PRESENT_YES    (0x00000001U)


#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SEQ_PROG_SIZE                 16U
#define LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       puPcAlt;
    LwU32                       pdPc;
    LwU32                       pdPcAlt;
    LwU32                       normalStart;
    LwU32                       safeStart;
    LwU32                       normalState;
    LwU32                       safeState;
    LwU32                       flags;
    LwU32                       seqProgram[LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_SEQ_PROG_SIZE];
} LW5070_CTRL_CMD_SET_SOR_SEQ_CTL_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL
 *
 * This command returns PIOR sequencer's power up and down PCs and sequencer
 * program to be used for power up and dowm.
 *
 *      orNumber
 *          The OR number for which the seq ctrls are to be modified.
 *
 *      puPcAlt
 *          Alternate power up PC.
 *
 *      pdPc
 *          Power down PC.
 *
 *      pdPcAlt
 *          Alternate power down PC.
 *
 *      normalStart
 *          Whether normal mode is using normal or alt PC
 *
 *      safeStart
 *          Whether safe mode is using normal or alt PC
 *
 *      normalState
 *          Whether normal state is PD or PU.
 *
 *      safeState
 *          Whether safe state is PD or PU.
 *
 *      flags
 *          There is only one flag defined lwrrently
 *              1. GET_SEQ_PROG: Whether or not current seq program must be
 *                 return back. Caller should set this to _YES to read the
 *                 current seq program.
 *
 *      seqProgram
 *          The sequencer program consisting of power up and down sequences.
 *          For LW50, this consists of 16 DWORDS. The program is
 *          relevant only when GET_SEQ_PROG flags is set to _YES.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL                         (0x50700303U) /* finn: Evaluated from "(FINN_LW50_DISPLAY_SEQ_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_PU_PC_ALT_VALUE                    3:0

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_PD_PC_VALUE                        3:0

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_PD_PC_ALT_VALUE                    3:0

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_NORMAL_START_VAL                   0:0
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_NORMAL_START_VAL_NORMAL (0x00000000U)
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_NORMAL_START_VAL_ALT    (0x00000001U)

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SAFE_START_VAL                     0:0
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SAFE_START_VAL_NORMAL   (0x00000000U)
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SAFE_START_VAL_ALT      (0x00000001U)

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_NORMAL_STATE_VAL                   0:0
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_NORMAL_STATE_VAL_PD     (0x00000000U)
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_NORMAL_STATE_VAL_PU     (0x00000001U)

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SAFE_STATE_VAL                     0:0
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SAFE_STATE_VAL_PD       (0x00000000U)
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SAFE_STATE_VAL_PU       (0x00000001U)

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_FLAGS_GET_SEQ_PROG                 0:0
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_FLAGS_GET_SEQ_PROG_NO   (0x00000000U)
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_FLAGS_GET_SEQ_PROG_YES  (0x00000001U)

#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SEQ_PROG_SIZE           8U
#define LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       puPcAlt;
    LwU32                       pdPc;
    LwU32                       pdPcAlt;
    LwU32                       normalStart;
    LwU32                       safeStart;
    LwU32                       normalState;
    LwU32                       safeState;
    LwU32                       flags;
    LwU32                       seqProgram[LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_SEQ_PROG_SIZE];
} LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL
 *
 * This command does the following in that order
 * (a) Loads a specified sequencer program for power up and down.
 * (b) Updates PIOR sequencer's power up and down PCs, tells seq to SKIP
 * current wait for vsync and waits until sequencer actually SKIPs or halts
 * (see more below under SKIP_WAIT_FOR_VSYNC flag) and
 * (c) Update power settings (safe/normal start and state).
 *
 *      orNumber
 *          The OR number for which the seq ctrls are to be modified.
 *
 *      puPcAlt
 *          Alternate power up PC.
 *
 *      pdPc
 *          Power down PC.
 *
 *      pdPcAlt
 *          Alternate power down PC.
 *
 *      normalStart
 *          Whether normal mode should use normal or alt PC
 *
 *      safeStart
 *          Whether safe mode should use normal or alt PC
 *
 *      normalState
 *          Whether normal state should be PD or PU.
 *
 *      safeState
 *          Whether safe state should be PD or PU.
 *
 *      flags
 *          The following flags have been defined
 *              1. SKIP_WAIT_FOR_VSYNC: Should seq be forced to skip waiting
 *                 for vsync if it's lwrrently waiting on such an instruction.
 *                 If the current instruction doesn't have a wait for vsync,
 *                 SKIP will be applied to the next one and so on until
 *                 either sequencer halts or an instruction with a wait for
 *                 vsync is found. The call will block until seq halts or
 *                 SKIPs a wait for vsync.
 *              2. SEQ_PROG_PRESENT: Whether or not a new seq program has
 *                 been specified.
 *
 *      seqProgram
 *          The sequencer program consisting of power up and down sequences.
 *          For LW50, this consists of 8 DWORDS. The program is
 *          relevant only when SEQ_PROG_PRESENT flags is set to _YES.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL                               (0x50700304U) /* finn: Evaluated from "(FINN_LW50_DISPLAY_SEQ_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PU_PC_ALT_VALUE                    3:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PU_PC_ALT_SPECIFIED              31:31
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PU_PC_ALT_SPECIFIED_NO        (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PU_PC_ALT_SPECIFIED_YES       (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_VALUE                        3:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_SPECIFIED                  31:31
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_SPECIFIED_NO            (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_SPECIFIED_YES           (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_ALT_VALUE                    3:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_ALT_SPECIFIED              31:31
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_ALT_SPECIFIED_NO        (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PD_PC_ALT_SPECIFIED_YES       (0x00000001U)


#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_START_VAL                        0:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_START_VAL_NORMAL       (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_START_VAL_ALT          (0x00000001U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_START_SPECIFIED                31:31
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_START_SPECIFIED_NO     (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_START_SPECIFIED_YES    (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_START_VAL                          0:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_START_VAL_NORMAL         (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_START_VAL_ALT            (0x00000001U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_START_SPECIFIED                  31:31
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_START_SPECIFIED_NO       (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_START_SPECIFIED_YES      (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_STATE_VAL                        0:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_STATE_VAL_PD           (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_STATE_VAL_PU           (0x00000001U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_STATE_SPECIFIED                31:31
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_STATE_SPECIFIED_NO     (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_NORMAL_STATE_SPECIFIED_YES    (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_STATE_VAL                          0:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_STATE_VAL_PD             (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_STATE_VAL_PU             (0x00000001U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_STATE_SPECIFIED                  31:31
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_STATE_SPECIFIED_NO       (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SAFE_STATE_SPECIFIED_YES      (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_FLAGS_SKIP_WAIT_FOR_VSYNC              0:0
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_FLAGS_SKIP_WAIT_FOR_VSYNC_NO  (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_FLAGS_SKIP_WAIT_FOR_VSYNC_YES (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_FLAGS_SEQ_PROG_PRESENT                  1:1
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_FLAGS_SEQ_PROG_PRESENT_NO     (0x00000000U)
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_FLAGS_SEQ_PROG_PRESENT_YES    (0x00000001U)

#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SEQ_PROG_SIZE                 8U
#define LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       orNumber;

    LwU32                       puPcAlt;
    LwU32                       pdPc;
    LwU32                       pdPcAlt;
    LwU32                       normalStart;
    LwU32                       safeStart;
    LwU32                       normalState;
    LwU32                       safeState;
    LwU32                       flags;
    LwU32                       seqProgram[LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_SEQ_PROG_SIZE];
} LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW5070_CTRL_CMD_CTRL_SEQ_PROG_SPEED
 *
 * This call allows a fast sequencer program to be selected. It's intended for
 * situations where panel sequencing is not required and the usual sequencing
 * delays cost too much time.
 *
 *     displayId
 *       The corresponding display ID. (Note that this call is lwrrently only
 *       supported for LVDS on an internal encoder, i.e. a SOR.)
 *     cmd
 *       The command to perform. Valid values are:
 *         LW5070_CTRL_SEQ_PROG_SPEED_CMD_GET
 *           Get the current state.
 *         LW5070_CTRL_SEQ_PROG_SPEED_CMD_SET
 *           Set the current state.
 *     state
 *       The state of panel sequencing for this displayId. This is an input
 *       when cmd = SET and an output when cmd = GET.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_PARAM_STRUCT
 *    LW_ERR_NOT_SUPPORTED
 *
 */

#define LW5070_CTRL_CMD_CTRL_SEQ_PROG_SPEED     (0x50700305U) /* finn: Evaluated from "(FINN_LW50_DISPLAY_SEQ_INTERFACE_ID << 8) | LW5070_CTRL_SEQ_PROG_SPEED_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_SEQ_PROG_SPEED_CMD_GET      (0x00000000U)
#define LW5070_CTRL_SEQ_PROG_SPEED_CMD_SET      (0x00000001U)

#define LW5070_CTRL_SEQ_PROG_SPEED_STATE_NORMAL (0x00000000U)
#define LW5070_CTRL_SEQ_PROG_SPEED_STATE_FAST   (0x00000001U)

#define LW5070_CTRL_SEQ_PROG_SPEED_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW5070_CTRL_SEQ_PROG_SPEED_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;

    LwU32                       displayId;

    LwU32                       cmd;
    LwU32                       state;
} LW5070_CTRL_SEQ_PROG_SPEED_PARAMS;

/* _ctrl5070seq_h_ */
