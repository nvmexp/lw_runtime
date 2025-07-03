/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlc56f.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/* AMPERE_CHANNEL_GPFIFO_A control commands and parameters */

#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl906f.h"          /* C36F is partially derived from 906F */
#include "ctrl/ctrla06f.h"          /* C36F is partially derived from a06F */
#include "ctrl/ctrlc36f.h" // This control call interface is an ALIAS of C36F
#include "cc_drv.h"

#define LWC56F_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xC36F, LWC56F_CTRL_##cat, idx)

/* AMPERE_CHANNEL_GPFIFO_A command categories (6bits) */
#define LWC56F_CTRL_RESERVED (0x00)
#define LWC56F_CTRL_GPFIFO   (0x01)
#define LWC56F_CTRL_EVENT    (0x02)

/*
 * LWC56F_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned is: LW_OK
*/
#define LWC56F_CTRL_CMD_NULL (LWC36F_CTRL_CMD_NULL)






/*
 * LWC56F_CTRL_GET_CLASS_ENGINEID
 *
 * Please see description of LW906F_CTRL_GET_CLASS_ENGINEID for more information.
 *
 */
#define LWC56F_CTRL_GET_CLASS_ENGINEID (LWC36F_CTRL_GET_CLASS_ENGINEID)

typedef LW906F_CTRL_GET_CLASS_ENGINEID_PARAMS LWC56F_CTRL_GET_CLASS_ENGINEID_PARAMS;

/*
 * LWC56F_CTRL_RESET_CHANNEL
 *
 * Please see description of LW906F_CTRL_CMD_RESET_CHANNEL for more information.
 *
 */
#define LWC56F_CTRL_CMD_RESET_CHANNEL (LWC36F_CTRL_CMD_RESET_CHANNEL)

typedef LW906F_CTRL_CMD_RESET_CHANNEL_PARAMS LWC56F_CTRL_CMD_RESET_CHANNEL_PARAMS;

/*
 * LWC56F_CTRL_CMD_GPFIFO_SCHEDULE
 *
 * Please see description of LWA06F_CTRL_CMD_GPFIFO_SCHEDULE for more information.
 *
 */
#define LWC56F_CTRL_CMD_GPFIFO_SCHEDULE (LWC36F_CTRL_CMD_GPFIFO_SCHEDULE)

typedef LWA06F_CTRL_GPFIFO_SCHEDULE_PARAMS LWC56F_CTRL_GPFIFO_SCHEDULE_PARAMS;

/*
 * LWC56F_CTRL_CMD_BIND
 *
 * Please see description of LWA06F_CTRL_CMD_BIND for more information.
 */
#define LWC56F_CTRL_CMD_BIND (LWC36F_CTRL_CMD_BIND)

typedef LWA06F_CTRL_BIND_PARAMS LWC56F_CTRL_BIND_PARAMS;

/*
 * LWC56F_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * Please see description of LWA06F_CTRL_CMD_EVENT_SET_NOTIFICATION for more information.
*/


#define LWC56F_CTRL_CMD_EVENT_SET_NOTIFICATION (LWC36F_CTRL_CMD_EVENT_SET_NOTIFICATION)

typedef LWA06F_CTRL_EVENT_SET_NOTIFICATION_PARAMS LWC56F_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* valid action values */
#define LWC56F_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE LWA06F_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE
#define LWC56F_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  LWA06F_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE
#define LWC56F_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  LWA06F_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT

/*
 * LWC56F_CTRL_CMD_EVENT_SET_TRIGGER
 *
 * Please see description of LWA06F_CTRL_CMD_EVENT_SET_TRIGGER for more information.
 */
#define LWC56F_CTRL_CMD_EVENT_SET_TRIGGER                 (LWC36F_CTRL_CMD_EVENT_SET_TRIGGER)





/*
 * LWC56F_CTRL_CMD_GET_MMU_FAULT_INFO
 *
 * Please see description of LW906F_CTRL_CMD_GET_MMU_FAULT_INFO for more information.
 *
 */
#define LWC56F_CTRL_CMD_GET_MMU_FAULT_INFO (LWC36F_CTRL_CMD_GET_MMU_FAULT_INFO)

typedef LW906F_CTRL_GET_MMU_FAULT_INFO_PARAMS LWC56F_CTRL_GET_MMU_FAULT_INFO_PARAMS;

/*
 * LWC56F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN
 *
 *    This command returns an opaque work submit token to the caller which
 *    can be used to write to doorbell register to finish submitting work.
 *
 *    workSubmitToken       The 32-bit work submit token
 *
 *    Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_OBJECT_HANDLE
 *     LW_ERR_ILWALID_ARGUMENT
 *
 */

#define LWC56F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN (LWC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN)

typedef struct LWC56F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS {
    LwU32 workSubmitToken;
} LWC56F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS;

/*
 * LWC56F_CTRL_CMD_GET_KMB
 *
 *    This command returns the Key Material Bundle (KMB) for the current channel.
 *
 *    kmb [OUT]           The KMB for the channel.
 *
 *    Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_OBJECT_HANDLE
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 */
#define LWC56F_CTRL_CMD_GET_KMB (0xc56f010b) /* finn: Evaluated from "(FINN_AMPERE_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | LWC56F_CTRL_CMD_GET_KMB_PARAMS_MESSAGE_ID" */

#define LWC56F_CTRL_CMD_GET_KMB_PARAMS_MESSAGE_ID (0xBU)

typedef struct LWC56F_CTRL_CMD_GET_KMB_PARAMS {
    CC_KMB kmb;
} LWC56F_CTRL_CMD_GET_KMB_PARAMS;

/*
 * LWC56F_CTRL_ROTATE_SELWRE_CHANNEL_IV
 *
 *    This command rotates the IVs for secure channels.
 *
 *    rotateIvType [IN]    Which IVs to rotate.
 *    updatedKmb [OUT]     Updated KMB after the IV rotation.
 *
 *    Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_OBJECT_HANDLE
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 */
#define LWC56F_CTRL_ROTATE_SELWRE_CHANNEL_IV (0xc56f010c) /* finn: Evaluated from "(FINN_AMPERE_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | LWC56F_CTRL_ROTATE_SELWRE_CHANNEL_IV_PARAMS_MESSAGE_ID" */

#define LWC56F_CTRL_ROTATE_SELWRE_CHANNEL_IV_PARAMS_MESSAGE_ID (0xLW)

typedef struct LWC56F_CTRL_ROTATE_SELWRE_CHANNEL_IV_PARAMS {
    ROTATE_IV_TYPE rotateIvType;
    CC_KMB         updatedKmb;
} LWC56F_CTRL_ROTATE_SELWRE_CHANNEL_IV_PARAMS;

/* _ctrlc56f.h_ */

