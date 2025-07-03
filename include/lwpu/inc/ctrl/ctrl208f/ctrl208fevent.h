/* 
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fevent.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/* LW20_SUBDEVICE_DIAG event-related control commands and parameters */

/*
 * LW208F_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the associated subdevice.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the associated subdevice object.
 *
 *   event
 *     This parameter specifies the type of event to which the specified
 *     action is to be applied.  This parameter must specify a valid
 *     LW208F_NOTIFIERS value (see cl208f.h for more details) and should
 *     not exceed one less LW208F_NOTIFIERS_MAXCOUNT.
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LW208F_CTRL_SET_EVENT_NOTIFICATION_DISABLE
 *         This action disables event notification for the specified
 *         event for the associated subdevice object.
 *       LW208F_CTRL_SET_EVENT_NOTIFICATION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event for the associated subdevice object.
 *       LW208F_CTRL_SET_EVENT_NOTIFICATION_REPEAT
 *         This action enables repeated event notification for the specified
 *         event for the associated system controller object.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW208F_CTRL_CMD_EVENT_SET_NOTIFICATION (0x208f1001) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_EVENT_INTERFACE_ID << 8) | LW208F_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_EVENT_SET_NOTIFICATION_PARAMS {
    LwU32 event;
    LwU32 action;
} LW208F_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* valid action values */
#define LW208F_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LW208F_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LW208F_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)

/*
 * LW208F_CTRL_CMD_EVENT_SET_TRIGGER
 *
 * This command triggers a software event for the associated subdevice.
 * This command accepts no parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW208F_CTRL_CMD_EVENT_SET_TRIGGER                 (0x208f1002) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_EVENT_INTERFACE_ID << 8) | 0x2" */

/* _ctrl208fevent_h_ */
