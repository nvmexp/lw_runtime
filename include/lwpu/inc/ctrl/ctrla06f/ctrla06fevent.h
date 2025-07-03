/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2007-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrla06f/ctrla06fevent.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrla06f/ctrla06fbase.h"

/*
 * LWA06F_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the associated channel.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the associated channel object.
 *
 *   event
 *     This parameter specifies the type of event to which the specified
 *     action is to be applied.  This parameter must specify a valid
 *     LWA06F_NOTIFIERS value (see cla06f.h for more details) and should
 *     not exceed one less LWA06F_NOTIFIERS_MAXCOUNT.
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LWA06F_CTRL_SET_EVENT_NOTIFICATION_ACTION_DISABLE
 *         This action disables event notification for the specified
 *         event for the associated channel object.
 *       LWA06F_CTRL_SET_EVENT_NOTIFICATION_ACTION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event for the associated channel object.
 *       LWA06F_CTRL_SET_EVENT_NOTIFICATION_ACTION_REPEAT
 *         This action enables repeated event notification for the specified
 *         event for the associated channel object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LWA06F_CTRL_CMD_EVENT_SET_NOTIFICATION (0xa06f0205) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_EVENT_INTERFACE_ID << 8) | LWA06F_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LWA06F_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWA06F_CTRL_EVENT_SET_NOTIFICATION_PARAMS {
    LwU32 event;
    LwU32 action;
} LWA06F_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* valid action values */
#define LWA06F_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LWA06F_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LWA06F_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)

/*
 * LWA06F_CTRL_CMD_EVENT_SET_TRIGGER
 *
 * This command triggers a software event for the associated channel.
 * This command accepts no parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWA06F_CTRL_CMD_EVENT_SET_TRIGGER                 (0xa06f0206) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_EVENT_INTERFACE_ID << 8) | 0x6" */


/* _ctrla06fevent_h_ */
