/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080unix.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX OS control commands and parameters */

/*
 * LW2080_CTRL_CMD_OS_UNIX_GC6_BLOCKER_REFCNT
 *
 * This command increases or decreases the value of the per-GPU GC6 blocker
 * refCount used by Linux kernel clients to prevent the GPU from entering GC6.
 *
 * When the refCount is non-zero, the GPU cannot enter GC6. When the refCount
 * transitions from zero to non-zero as a result of this command, the GPU will
 * automatically come out of GC6.
 *
 * action   Whether to increment or decrement the value of the refCount.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_OS_UNIX_GC6_BLOCKER_REFCNT (0x20803d01) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | LW2080_CTRL_OS_UNIX_GC6_BLOCKER_REFCNT_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_OS_UNIX_GC6_BLOCKER_REFCNT_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_OS_UNIX_GC6_BLOCKER_REFCNT_PARAMS {
    LwU32 action;
} LW2080_CTRL_OS_UNIX_GC6_BLOCKER_REFCNT_PARAMS;

// Possible values for action
#define LW2080_CTRL_OS_UNIX_GC6_BLOCKER_REFCNT_INC   (0x00000001)
#define LW2080_CTRL_OS_UNIX_GC6_BLOCKER_REFCNT_DEC   (0x00000002)

/*
 * LW2080_CTRL_CMD_OS_UNIX_ALLOW_DISALLOW_GCOFF
 *
 * RM by default allows GCOFF but when the X driver disallows to enter in GCOFF
 * then this rmcontrol sets flag as LW_FALSE and if it allows to enter in GCOFF
 * then the flag is set as LW_TRUE.
 *
 * action   Whether to allow or disallow the user mode clients to enter in GCOFF.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_OS_UNIX_ALLOW_DISALLOW_GCOFF (0x20803d02) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | LW2080_CTRL_OS_UNIX_ALLOW_DISALLOW_GCOFF_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_OS_UNIX_ALLOW_DISALLOW_GCOFF_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_OS_UNIX_ALLOW_DISALLOW_GCOFF_PARAMS {
    LwU32 action;
} LW2080_CTRL_OS_UNIX_ALLOW_DISALLOW_GCOFF_PARAMS;

// Possible values for action
#define LW2080_CTRL_OS_UNIX_ALLOW_DISALLOW_GCOFF_ALLOW    (0x00000001)
#define LW2080_CTRL_OS_UNIX_ALLOW_DISALLOW_GCOFF_DISALLOW (0x00000002)

/*
 * LW2080_CTRL_CMD_OS_UNIX_AUDIO_DYNAMIC_POWER
 *
 * GPU can have integrated HDA (High Definition Audio) controller which
 * can be in active or suspended state during dynamic power management.
 * This command will perform HDA controller wakeup (if bEnter is false) or
 * suspend (if bEnter is true).
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_OS_UNIX_AUDIO_DYNAMIC_POWER       (0x20803d03) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | LW2080_CTRL_OS_UNIX_AUDIO_DYNAMIC_POWER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_OS_UNIX_AUDIO_DYNAMIC_POWER_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_OS_UNIX_AUDIO_DYNAMIC_POWER_PARAMS {
    LwBool bEnter;
} LW2080_CTRL_OS_UNIX_AUDIO_DYNAMIC_POWER_PARAMS;

/*
 * LW2080_CTRL_CMD_OS_UNIX_INSTALL_PROFILER_HOOKS
 *
 * Initialize cyclestats HWPM support in the kernel. This will set up a callback
 * event for the channel indicated by hNotifierResource. This callback will execute
 * perf register read / write commands enqueued in the shared buffer indicated by
 * hNotifyDataMemory. Only one client may use HWPM functionality at a time.
 *
 * Additionally, if perfmonIdCount is greater than zero, mode-e HWPM streaming into
 * the buffer indicated by hSnapshotMemory will be initialized (but not turned on).
 * Data will be copied into the provided buffer every 10ms, or whenever a
 * LW2080_CTRL_CMD_OS_UNIX_FLUSH_SNAPSHOT_BUFFER command is issued.
 */
#define LW2080_CTRL_CMD_OS_UNIX_INSTALL_PROFILER_HOOKS (0x20803d04) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | LW2080_CTRL_OS_UNIX_INSTALL_PROFILER_HOOKS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_OS_UNIX_INSTALL_PROFILER_HOOKS_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_OS_UNIX_INSTALL_PROFILER_HOOKS_PARAMS {
    LwHandle hNotifierResource;
    LwU32    notifyDataSize;
    LwHandle hNotifyDataMemory;
    LwU32    perfmonIdCount;
    LwU32    snapshotBufferSize;
    LwHandle hSnapshotMemory;
} LW2080_CTRL_OS_UNIX_INSTALL_PROFILER_HOOKS_PARAMS;

/*
 * LW2080_CTRL_CMD_OS_UNIX_FLUSH_SNAPSHOT_BUFFER
 *
 * Immediately copies any pending mode-e HWPM data into the previously
 * installed snapshot buffer instead of waiting for the timer.
 */
#define LW2080_CTRL_CMD_OS_UNIX_FLUSH_SNAPSHOT_BUFFER     (0x20803d05) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | 0x5" */

/*
 * LW2080_CTRL_CMD_OS_UNIX_STOP_PROFILER
 *
 * Stop the timer responsible for copying mode-e HWPM data to the snapshot buffer.
 * The snapshot buffer must not be freed by the client before this command is issued.
 */
#define LW2080_CTRL_CMD_OS_UNIX_STOP_PROFILER             (0x20803d06) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | 0x6" */

/*
 * LW2080_CTRL_CMD_OS_UNIX_VIDMEM_PERSISTENCE_STATUS
 *
 * This command will be used by clients to check if the GPU video memory will
 * be persistent during system suspend/resume cycle.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_OS_UNIX_VIDMEM_PERSISTENCE_STATUS (0x20803d07) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | LW2080_CTRL_OS_UNIX_VIDMEM_PERSISTENCE_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_OS_UNIX_VIDMEM_PERSISTENCE_STATUS_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_OS_UNIX_VIDMEM_PERSISTENCE_STATUS_PARAMS {
    LwBool bVidmemPersistent;
} LW2080_CTRL_OS_UNIX_VIDMEM_PERSISTENCE_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_OS_UNIX_UPDATE_TGP_STATUS
 *
 * This command will be used by clients to set restore TGP flag which will
 * help to restore TGP limits when clients are destroyed.
 *
 * Possible status values returned are:
 * LW_OK
 */
#define LW2080_CTRL_CMD_OS_UNIX_UPDATE_TGP_STATUS (0x20803d08) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_OS_UNIX_INTERFACE_ID << 8) | LW2080_CTRL_OS_UNIX_UPDATE_TGP_STATUS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_OS_UNIX_UPDATE_TGP_STATUS_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_OS_UNIX_UPDATE_TGP_STATUS_PARAMS {
    LwBool bUpdateTGP;
} LW2080_CTRL_OS_UNIX_UPDATE_TGP_STATUS_PARAMS;
/* _ctrl2080unix_h_ */
