/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc370/ctrlc370event.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlc370/ctrlc370base.h"
/* C370 is partially derived from 5070 */
#include "ctrl/ctrl5070/ctrl5070event.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * The following control calls are defined in ctrl5070event.h, but they are
 * still supported on LWC370. We redirect these control cmds to LW5070_CTRL_CMD,
 * and keep the _PARAMS unchanged for now.
 */

#define LWC370_CTRL_CMD_EVENT_SET_TRIGGER         LW5070_CTRL_CMD_EVENT_SET_TRIGGER
#define LWC370_CTRL_CMD_EVENT_SET_MEMORY_NOTIFIES LW5070_CTRL_CMD_EVENT_SET_MEMORY_NOTIFIES

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)





#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/* 
* headId
*   This parameter indicates the ID of head on which we received interrupt
* RgSemId
*   This parameter indicates the RG Semaphore Index for given head
*/
typedef struct LWC370_RG_SEM_NOTIFICATION_PARAMS {
    LwU32 headId;
    LwU32 rgSemId;
} LWC370_RG_SEM_NOTIFICATION_PARAMS;

/* 
* LWC370_VPR_POLICY enums that indicate the type of policy violation
* Each enum value is mutually exclusive and is set by the SEC POLICY
* interrupt handler. Client receives event notif data with one of these 
* set as part of the notificaion params.
*/
typedef enum LWC370_VPR_POLICY {
    LWC370_VPR_POLICY_N0_HDCP = 0,
    LWC370_VPR_POLICY_HDCP1X = 1,
    LWC370_VPR_POLICY_INTERNAL = 2,
    LWC370_VPR_POLICY_SLI = 3,
    LWC370_VPR_POLICY_HDCP22_TYPE0 = 4,
    LWC370_VPR_POLICY_HDCP22_TYPE1 = 5,
    LWC370_VPR_POLICY_RWPR = 6,
    LWC370_VPR_POLICY_VGA = 7,
} LWC370_VPR_POLICY;

/* 
* LWC370_VPR_NOTIFICATION_PARAMS - struct used to pass information from
* RM to registered clients for SEC POLICY violation events
* 
* violationPolicy
*   This parameter indicates the type of policy violation that oclwrred
* headId
*   This parameter indicates the ID of the first violating HEAD
* sorId
*   This parameter indicates the ID of the first violating SOR
* windowId
*   This parameter indicates the ID of the first violating Window
*/
typedef struct LWC370_VPR_NOTIFICATION_PARAMS {
    LWC370_VPR_POLICY violationPolicy;
    LwU32             headId;
    LwHandle          sorId;
    LwU32             windowId;
} LWC370_VPR_NOTIFICATION_PARAMS;

/* LWC370_SUBDEVICE_XX event-related control commands and parameters */

/*
 * LWC370_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the associated subdevice.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the associated subdevice object.
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
 *     LW2080_NOTIFIERS value (see cl2080.h for more details) and should
 *     not exceed one less LW2080_NOTIFIERS_MAXCOUNT.
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LWC370_CTRL_SET_EVENT_NOTIFICATION_DISABLE
 *         This action disables event notification for the specified
 *         event for the associated subdevice object.
 *       LWC370_CTRL_SET_EVENT_NOTIFICATION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event for the associated subdevice object.
 *       LWC370_CTRL_SET_EVENT_NOTIFICATION_REPEAT
 *         This action enables repeated event notification for the specified
 *         event for the associated system controller object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LWC370_CTRL_CMD_EVENT_SET_NOTIFICATION (0xc3700901) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_EVENT_INTERFACE_ID << 8) | 0x1" */

typedef LW5070_CTRL_EVENT_SET_NOTIFICATION_PARAMS LWC370_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC370_CTRL_CMD_EVENT_SET_NOTIFICATION_FINN_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC370_CTRL_CMD_EVENT_SET_NOTIFICATION_FINN_PARAMS {
    LWC370_CTRL_EVENT_SET_NOTIFICATION_PARAMS params;
} LWC370_CTRL_CMD_EVENT_SET_NOTIFICATION_FINN_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/* valid action values */
#define LWC370_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE LW5070_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE
#define LWC370_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  LW5070_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE
#define LWC370_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  LW5070_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT

/* _ctrlc370event_h_ */
