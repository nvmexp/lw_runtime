/*
 * SPDX-FileCopyrightText: Copyright (c) 2010-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl5070/ctrl5070event.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl5070/ctrl5070base.h"

/* LW50_DISPLAY event-related control commands and parameters */

/*
 * LW5070_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the LW50_DISPLAY object.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the LW50_DISPLAY object.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the LW50_DISPLAY
 *     parent device to which the operation should be directed.  This parameter
 *     must specify a value between zero and the total number of subdevices
 *     within the parent device.  This parameter should be set to zero for
 *     default behavior.
 *   hEvent
 *     This parameter specifies the handle of the LW01_EVENT instance
 *     to be bound to the given subDeviceInstance.
 *   event
 *     This parameter specifies the type of event to which the specified
 *     action is to be applied.  This parameter must specify a valid
 *     NOTIFIERS value of display class.
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LW5070_CTRL_SET_EVENT_NOTIFICATION_DISABLE
 *         This action disables event notification for the specified
 *         event for the associated subdevice object.
 *       LW5070_CTRL_SET_EVENT_NOTIFICATION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event for the associated subdevice object.
 *       LW5070_CTRL_SET_EVENT_NOTIFICATION_REPEAT
 *         This action enables repeated event notification for the specified
 *         event for the associated system controller object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW5070_CTRL_CMD_EVENT_SET_NOTIFICATION (0x50700901) /* finn: Evaluated from "(FINN_LW50_DISPLAY_EVENT_INTERFACE_ID << 8) | LW5070_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW5070_CTRL_EVENT_SET_NOTIFICATION_PARAMS {
    LwU32    subDeviceInstance;
    LwHandle hEvent;
    LwU32    event;
    LwU32    action;
} LW5070_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* valid action values */
#define LW5070_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LW5070_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LW5070_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)


/*
 * LW5070_CTRL_CMD_EVENT_SET_TRIGGER
 *
 * This command triggers a software event for the LW50_DISPLAY object.
 * This command accepts no parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW5070_CTRL_CMD_EVENT_SET_TRIGGER                 (0x50700902) /* finn: Evaluated from "(FINN_LW50_DISPLAY_EVENT_INTERFACE_ID << 8) | 0x2" */


/*
 * LW5070_CTRL_CMD_EVENT_SET_NOTIFIER_MEMORY
 *
 *     hMemory
 *       This parameter specifies the handle of the memory object
 *       that identifies the memory address translation for this
 *       subdevice instance's notification(s).  The beginning of the
 *       translation points to an array of notification data structures.
 *       The size of the translation must be at least large enough to hold the
 *       maximum number of notification data structures.
 *       Legal argument values must be instances of the following classes:
 *         LW01_NULL
 *         LW04_MEMORY
 *       When hMemory specifies the LW01_NULL_OBJECT value then any existing
 *       memory translation connection is cleared.  There must not be any
 *       pending notifications when this command is issued.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW5070_CTRL_CMD_EVENT_SET_MEMORY_NOTIFIES         (0x50700903) /* finn: Evaluated from "(FINN_LW50_DISPLAY_EVENT_INTERFACE_ID << 8) | LW5070_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW5070_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS {
    LwU32    subDeviceInstance;
    LwHandle hMemory;
} LW5070_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS;

#define LW5070_EVENT_MEMORY_NOTIFIES_STATUS_NOTIFIED 0
#define LW5070_EVENT_MEMORY_NOTIFIES_STATUS_PENDING  1
#define LW5070_EVENT_MEMORY_NOTIFIES_STATUS_ERROR    2



/* _ctrl5070event_h_ */
