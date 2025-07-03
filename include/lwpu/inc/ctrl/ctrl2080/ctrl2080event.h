/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080event.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

#include "lw_vgpu_types.h"
/* LW20_SUBDEVICE_XX event-related control commands and parameters */

/*
 * LW2080_CTRL_CMD_EVENT_SET_NOTIFICATION
 *
 * This command sets event notification state for the associated subdevice.
 * This command requires that an instance of LW01_EVENT has been previously
 * bound to the associated subdevice object.
 *
 *   event
 *     This parameter specifies the type of event to which the specified
 *     action is to be applied.  This parameter must specify a valid
 *     LW2080_NOTIFIERS value (see cl2080.h for more details) and should
 *     not exceed one less LW2080_NOTIFIERS_MAXCOUNT.
 *   action
 *     This parameter specifies the desired event notification action.
 *     Valid notification actions include:
 *       LW2080_CTRL_SET_EVENT_NOTIFICATION_DISABLE
 *         This action disables event notification for the specified
 *         event for the associated subdevice object.
 *       LW2080_CTRL_SET_EVENT_NOTIFICATION_SINGLE
 *         This action enables single-shot event notification for the
 *         specified event for the associated subdevice object.
 *       LW2080_CTRL_SET_EVENT_NOTIFICATION_REPEAT
 *         This action enables repeated event notification for the specified
 *         event for the associated system controller object.
 *    bNotifyState
 *      This boolean is used to indicate the current state of the notifier
 *      at the time of event registration. This is optional and its semantics
 *      needs to be agreed upon by the notifier and client using the notifier
 *    info32
 *      This is used to send 32-bit initial state info with the notifier at
 *      time of event registration
 *    info16
 *      This is used to send 16-bit initial state info with the notifier at
 *      time of event registration
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_EVENT_SET_NOTIFICATION (0x20800301) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS {
    LwU32  event;
    LwU32  action;
    LwBool bNotifyState;
    LwU32  info32;
    LwU16  info16;
} LW2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS;

/* valid action values */
#define LW2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE (0x00000000)
#define LW2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_SINGLE  (0x00000001)
#define LW2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT  (0x00000002)

/* XUSB/PPC D-state defines */
#define LW2080_EVENT_DSTATE_XUSB_D0                       (0x00000000)
#define LW2080_EVENT_DSTATE_XUSB_D3                       (0x00000003)
#define LW2080_EVENT_DSTATE_XUSB_ILWALID                  (0xFFFFFFFF)
#define LW2080_EVENT_DSTATE_PPC_D0                        (0x00000000)
#define LW2080_EVENT_DSTATE_PPC_D3                        (0x00000003)
#define LW2080_EVENT_DSTATE_PPC_ILWALID                   (0xFFFFFFFF)

// HDACODEC Decice DState, D3_COLD is only for verbose mapping, it cannot be logged
typedef enum LW2080_EVENT_HDACODEC_DSTATE {
    LW2080_EVENT_HDACODEC_DEVICE_DSTATE_D0 = 0,
    LW2080_EVENT_HDACODEC_DEVICE_DSTATE_D1 = 1,
    LW2080_EVENT_HDACODEC_DEVICE_DSTATE_D2 = 2,
    LW2080_EVENT_HDACODEC_DEVICE_DSTATE_D3_HOT = 3,
    LW2080_EVENT_HDACODEC_DEVICE_DSTATE_D3_COLD = 4,
    LW2080_EVENT_HDACODEC_DEVICE_DSTATE_DSTATE_MAX = 5,
} LW2080_EVENT_HDACODEC_DSTATE;

/*
 * LW2080_CTRL_CMD_EVENT_SET_TRIGGER
 *
 * This command triggers a software event for the associated subdevice.
 * This command accepts no parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_EVENT_SET_TRIGGER         (0x20800302) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | 0x2" */

/*
 * LW2080_CTRL_CMD_EVENT_SET_NOTIFIER_MEMORY
 *
 *     hMemory
 *       This parameter specifies the handle of the memory object
 *       that identifies the memory address translation for this
 *       subdevice instance's notification(s).  The beginning of the
 *       translation points to an array of notification data structures.
 *       The size of the translation must be at least large enough to hold the
 *       maximum number of notification data structures identified by
 *       the LW2080_MAX_NOTIFIERS value.
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
#define LW2080_CTRL_CMD_EVENT_SET_MEMORY_NOTIFIES (0x20800303) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS {
    LwHandle hMemory;
} LW2080_CTRL_EVENT_SET_MEMORY_NOTIFIES_PARAMS;

#define LW2080_EVENT_MEMORY_NOTIFIES_STATUS_NOTIFIED 0
#define LW2080_EVENT_MEMORY_NOTIFIES_STATUS_PENDING  1
#define LW2080_EVENT_MEMORY_NOTIFIES_STATUS_ERROR    2

/*
 * LW2080_CTRL_CMD_EVENT_SET_SEMAPHORE_MEMORY
 *
 *     hSemMemory
 *       This parameter specifies the handle of the memory object that
 *       identifies the semaphore memory associated with this subdevice
 *       event notification.  Once this is set RM will generate an event
 *       only when there is a change in the semaphore value.  It is
 *       expected that the semaphore memory value will be updated by
 *       the GPU indicating that there is an event pending. This
 *       command is used by VGX plugin to determine which virtual
 *       machine has generated a particular event.
 *
 *     semOffset
 *       This parameter indicates the memory offset of the semaphore.
 *
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_PARAM_STRUCT
 *      LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_EVENT_SET_SEMAPHORE_MEMORY   (0x20800304) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_SET_SEMAPHORE_MEMORY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_EVENT_SET_SEMAPHORE_MEMORY_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_EVENT_SET_SEMAPHORE_MEMORY_PARAMS {
    LwHandle hSemMemory;
    LwU32    semOffset;
} LW2080_CTRL_EVENT_SET_SEMAPHORE_MEMORY_PARAMS;

/*
 * LW2080_CTRL_CMD_EVENT_SET_GUEST_MSI
 *
 *     hSemMemory
 *       This parameter specifies the handle of the memory object that
 *       identifies the semaphore memory associated with this subdevice
 *       event notification.  Once this is set RM will generate an event
 *       only when there is a change in the semaphore value.  It is
 *       expected that the semaphore memory value will be updated by
 *       the GPU indicating that there is an event pending. This
 *       command is used by VGX plugin to determine which virtual
 *       machine has generated a particular event.
 *
 *     guestMSIAddr
 *       This parameter indicates the guest allocated MSI address.
 *
 *     guestMSIData
 *       This parameter indicates the MSI data set by the guest OS.
 *
 *     vmIdType
 *       This parameter specifies the type of guest virtual machine identifier
 *
 *     guestVmId
 *       This parameter specifies the guest virtual machine identifier
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_PARAM_STRUCT
 *      LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_EVENT_SET_GUEST_MSI (0x20800305) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_SET_GUEST_MSI_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_EVENT_SET_GUEST_MSI_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_EVENT_SET_GUEST_MSI_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 guestMSIAddr, 8);
    LwU32      guestMSIData;
    LwHandle   hSemMemory;
    LwBool     isReset;
    VM_ID_TYPE vmIdType;
    LW_DECLARE_ALIGNED(VM_ID guestVmId, 8);
} LW2080_CTRL_EVENT_SET_GUEST_MSI_PARAMS;


/*
 * LW2080_CTRL_CMD_EVENT_SET_SEMA_MEM_VALIDATION
 *
 *     hSemMemory
 *       This parameter specifies the handle of the memory object that
 *       identifies the semaphore memory associated with this subdevice
 *       event notification.  Once this is set RM will generate an event
 *       only when there is a change in the semaphore value.  It is
 *       expected that the semaphore memory value will be updated by
 *       the GPU indicating that there is an event pending. This
 *       command is used by VGX plugin to determine which virtual
 *       machine has generated a particular event.
 *
 *     isSemaMemValidationEnabled
 *       This parameter used to enable/disable change in sema value check
 *       while generating an event.
 *
 * Possible status values returned are:
 *      LWOS_STATUS_SUCCESS
 *      LWOS_STATUS_ERROR_ILWALID_OBJECT_HANDLE
 *      LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */


#define LW2080_CTRL_CMD_EVENT_SET_SEMA_MEM_VALIDATION (0x20800306) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_SET_SEMA_MEM_VALIDATION_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_EVENT_SET_SEMA_MEM_VALIDATION_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_EVENT_SET_SEMA_MEM_VALIDATION_PARAMS {
    LwHandle hSemMemory;
    LwBool   isSemaMemValidationEnabled;
} LW2080_CTRL_EVENT_SET_SEMA_MEM_VALIDATION_PARAMS;


/*
 * LW2080_CTRL_CMD_EVENT_SET_VMBUS_CHANNEL
 *
 *     hSemMemory
 *       This parameter specifies the handle of the memory object that
 *       identifies the semaphore memory associated with this subdevice
 *       event notification.  Once this is set RM will generate an event
 *       only when there is a change in the semaphore value.  It is
 *       expected that the semaphore memory value will be updated by
 *       the GPU indicating that there is an event pending. This
 *       command is used by VGX plugin to determine which virtual
 *       machine has generated a particular event.
 *
 *     vmIdType
 *       This parameter specifies the type of guest virtual machine identifier
 *
 *     guestVmId
 *       This parameter specifies the guest virtual machine identifier
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_EVENT_SET_VMBUS_CHANNEL (0x20800307) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_SET_VMBUS_CHANNEL_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_EVENT_SET_VMBUS_CHANNEL_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_EVENT_SET_VMBUS_CHANNEL_PARAMS {
    LwHandle   hSemMemory;
    VM_ID_TYPE vmIdType;
    LW_DECLARE_ALIGNED(VM_ID guestVmId, 8);
} LW2080_CTRL_EVENT_SET_VMBUS_CHANNEL_PARAMS;


/*
 * LW2080_CTRL_CMD_EVENT_SET_TRIGGER_FIFO
 *
 * This command triggers a FIFO event for the associated subdevice.
 *
 *  hEvent
 *    Handle of the event that should be notified. If zero, all
 *    non-stall interrupt events for this subdevice will be notified.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_EVENT_SET_TRIGGER_FIFO (0x20800308) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS {
    LwHandle hEvent;
} LW2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS;

/*
 * LW2080_CTRL_CMD_EVENT_VIDEO_BIND_EVTBUF_FOR_UID
 *
 * This command is used to create a video bind-point to an event buffer that
 * is filtered by UID.
 *
 *  hEventBuffer[IN]
 *      The event buffer to bind to
 *
 *  recordSize[IN]
 *      The size of the FECS record in bytes
 *
 *  levelOfDetail[IN]
 *      One of LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_LOD_:
 *          FULL: Report all CtxSw events
 *          SIMPLE: Report engine start and engine end events only
 *          CUSTOM: Report events in the eventFilter field
 *      NOTE: RM may override the level-of-detail depending on the caller
 *
 *  eventFilter[IN]
 *      Bitmask of events to report if levelOfDetail is CUSTOM
 *
 *  bAllUsers[IN]
 *     Only report video data for the current user if false, for all users if true
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_EVENT_VIDEO_BIND_EVTBUF (0x20800309) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_EVENT_INTERFACE_ID << 8) | LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_PARAMS_MESSAGE_ID" */

typedef enum LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_LOD {
    LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_LOD_FULL = 0,
    LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_LOD_SIMPLE = 1,
    LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_LOD_LWSTOM = 2,
} LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_LOD;

#define LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_PARAMS {
    LwHandle                                hEventBuffer;
    LwU32                                   recordSize;
    LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_LOD levelOfDetail;
    LwU32                                   eventFilter;
    LwBool                                  bAllUsers;
} LW2080_CTRL_EVENT_VIDEO_BIND_EVTBUF_PARAMS;

/* _ctrl2080event_h_ */
