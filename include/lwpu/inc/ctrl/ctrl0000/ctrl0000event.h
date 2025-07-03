/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000event.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0000/ctrl0000base.h"

#include "ctrl/ctrlxxxx.h"
#include "class/cl0000.h"
/*
 * LW0000_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification for the system events.
 * 
 *   event
 *     This parameter specifies the type of event to which the specified
 *     action is to be applied. The valid event values can be found in
 *     cl0000.h. 
 *
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LW0000_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE
 *         This action disables event notification for the specified
 *         event.
 *       LW0000_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event.
 *       LW0000_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT
 *         This action enables repeated event notification for the 
 *         specified event.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_CLIENT
 *
 */

#define LW0000_CTRL_CMD_EVENT_SET_NOTIFICATION (0x501) /* finn: Evaluated from "(FINN_LW01_ROOT_EVENT_INTERFACE_ID << 8) | LW0000_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_EVENT_SET_NOTIFICATION_PARAMS {
    LwU32 event;
    LwU32 action;
} LW0000_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* valid action values */
#define LW0000_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LW0000_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LW0000_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)

/*
 * LW0000_CTRL_CMD_GET_SYSTEM_EVENT_STATUS
 *
 * This command returns the status of the specified system event type.
 * See the description of LW01_EVENT for details on registering events.
 *
 *   event
 *     This parameter specifies the event type. Valid event type values
 *     can be found in cl0000.h.
 *   status
 *     This parameter returns the status for a given event type. Valid
 *     status values can be found in cl0000.h.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_CLIENT
 *
 */

#define LW0000_CTRL_CMD_GET_SYSTEM_EVENT_STATUS           (0x502) /* finn: Evaluated from "(FINN_LW01_ROOT_EVENT_INTERFACE_ID << 8) | LW0000_CTRL_GET_SYSTEM_EVENT_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GET_SYSTEM_EVENT_STATUS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_GET_SYSTEM_EVENT_STATUS_PARAMS {
    LwU32 event;
    LwU32 status;
} LW0000_CTRL_GET_SYSTEM_EVENT_STATUS_PARAMS;

/* _ctrl0000event_h_ */

