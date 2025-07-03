/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0004.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW01_TIMER control commands and parameters */

#define LW0004_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x0004, LW0004_CTRL_##cat, idx)

/* LW01_TIMER command categories (8bits) */
#define LW0004_CTRL_RESERVED (0x00)
#define LW0004_CTRL_TMR      (0x01)

/*
 * LW0004_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0004_CTRL_CMD_NULL (0x40000) /* finn: Evaluated from "(FINN_LW01_TIMER_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW0004_CTRL_CMD_TMR_SET_ALARM_NOTIFY
 *
 * This command can be used to set a PTIMER alarm to trigger at the
 * specified time in the future on the subdevice associated with this
 * LW01_TIMER object instance.
 *
 *   hEvent
 *     This parameter specifies the handle of an LW01_EVENT object instance
 *     that is to be signaled when the alarm triggers.  This LW01_EVENT
 *     object instance must have been allocated with this LW01_TIMER object
 *     instance as its parent.  If this parameter is set to LW01_NULL_OBJECT
 *     then all LW01_EVENT object instances associated with this LW01_TIMER
 *     object instance are signaled.
 *   alarmTimeUsecs
 *     This parameter specifies the relative time in nanoseconds at which
 *     the alarm should trigger.  Note that the accuracy between the alarm
 *     trigger and the subsequent notification to the caller can vary
 *     depending on system conditions.
 *
 * Possible status values returned include:
 *   LWOS_STATUS_SUCCES
 *   LWOS_STATUS_ILWALID_PARAM_STRUCT
 *   LWOS_STATUS_ILWALID_OBJECT_HANDLE
 */

#define LW0004_CTRL_CMD_TMR_SET_ALARM_NOTIFY (0x40110) /* finn: Evaluated from "(FINN_LW01_TIMER_TMR_INTERFACE_ID << 8) | LW0004_CTRL_TMR_SET_ALARM_NOTIFY_PARAMS_MESSAGE_ID" */

#define LW0004_CTRL_TMR_SET_ALARM_NOTIFY_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW0004_CTRL_TMR_SET_ALARM_NOTIFY_PARAMS {
    LwHandle hEvent;
    LW_DECLARE_ALIGNED(LwU64 alarmTimeNsecs, 8);
} LW0004_CTRL_TMR_SET_ALARM_NOTIFY_PARAMS;

/* _ctrl0004_h_ */

