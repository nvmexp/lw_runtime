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
// Source file: ctrl/ctrl5070/ctrl5070verif.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl5070/ctrl5070base.h"
#include "ctrl5070common.h"
#include "ctrl5070impoverrides.h"

/*
 * LW5070_CTRL_CMD_SET_VBIOS_ATTENTION_EVENT
 *
 * This command sets the OS event handle that RM will call to wake the client
 * when VBIOS_ATTENTION_PENDING interrupt has been fired and cleared. By
 * setting this event, RM will also bypass the normal handling of the
 * interrupt.
 * 
 *      hEvent
 *          The handle to the event that RM will use to wake the client.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_VBIOS_ATTENTION_EVENT (0x50700601) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_VBIOS_ATTENTION_EVENT_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_VBIOS_ATTENTION_EVENT_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW5070_CTRL_CMD_SET_VBIOS_ATTENTION_EVENT_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);
} LW5070_CTRL_CMD_SET_VBIOS_ATTENTION_EVENT_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE
 *
 * This command sets the display exception restart mode. 
 * Also resets the exception counter for this specific exception mode.
 * 
 *      channel
 *          Target channel for the exception mode
 *
 *      reason
 *          Reason for the exception
 *
 *      restartMode
 *          The restart mode in case of exception
 *
 *      assert
 *          Whether RM should assert in case of exception
 *
 *      useExceptArg
 *          Whether to use client supplied exception argument
 *
 *      exceptArg
 *          The client supplied exception argument
 *
 *      hEvent
 *          The handle to the OS event that RM will use to call the client
 *          in case of exception event
 *      
 *      manualRestart
 *          Whether the client wants to manually restart the HW when an OS
 *          event is set for exception.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE                            (0x50700602) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_REASON                     2:0
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_REASON_PUSHBUFFER_ERR      (0x00000001)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_REASON_TRAP                (0x00000002)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_REASON_RESERVED_METHOD     (0x00000003)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_REASON_ILWALID_ARG         (0x00000004)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_REASON_ILWALID_STATE       (0x00000005)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_REASON_UNRESOLVABLE_HANDLE (0x00000006)

#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_RESTART_MODE               1:0
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_RESTART_MODE_RESUME        (0x00000000)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_RESTART_MODE_SKIP          (0x00000001)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_RESTART_MODE_REPLAY        (0x00000002)

#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_ASSERT                     0:0
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_ASSERT_NO                  (0x00000000)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_ASSERT_YES                 (0x00000001)

#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_USE_EXCEPT_ARG             0:0
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_USE_EXCEPT_ARG_NO          (0x00000000)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_USE_EXCEPT_ARG_YES         (0x00000001)

#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_MANUAL_RESTART             0:0
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_MANUAL_RESTART_NO          (0x00000000)
#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_MANUAL_RESTART_YES         (0x00000001)

#define LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channel;
    LwU32                       reason;
    LwU32                       restartMode;
    LwU32                       assert;
    LwU32                       useExceptArg;
    LwU32                       exceptArg;

    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);

    LwU32                       manualRestart;
} LW5070_CTRL_CMD_SET_EXCEPTION_RESTART_MODE_PARAMS;

/*
 * LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE
 *
 * This command gets the display exception restart mode.
 * 
 *      channel
 *          Target channel for the exception mode
 *
 *      reason
 *          Reason for the exception
 *
 *      restartMode
 *          The restart mode in case of exception
 *
 *      assert
 *          Whether RM should assert in case of exception
 *
 *      useExceptArg
 *          Whether to use exception argument
 *
 *      exceptArg
 *          The exception argument
 *
 *      exceptCnt
 *          Number of exceptions since last reset
 *
 *      hEvent
 *          The handle to the OS event that RM will use to call the client
 *          in case of exception event
 *
 *      manualRestart
 *          Whether the client wants to manually restart the HW when an OS
 *          event is set for exception.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE                            (0x50700603) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_REASON                     2:0
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_REASON_PUSHBUFFER_ERR      (0x00000001)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_REASON_TRAP                (0x00000002)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_REASON_RESERVED_METHOD     (0x00000003)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_REASON_ILWALID_ARG         (0x00000004)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_REASON_ILWALID_STATE       (0x00000005)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_REASON_UNRESOLVABLE_HANDLE (0x00000006)

#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_RESTART_MODE               1:0
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_RESTART_MODE_RESUME        (0x00000000)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_RESTART_MODE_SKIP          (0x00000001)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_RESTART_MODE_REPLAY        (0x00000002)

#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_ASSERT                     0:0
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_ASSERT_NO                  (0x00000000)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_ASSERT_YES                 (0x00000001)

#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_USE_EXCEPT_ARG             0:0
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_USE_EXCEPT_ARG_NO          (0x00000000)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_USE_EXCEPT_ARG_YES         (0x00000001)

#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_MANUAL_RESTART             0:0
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_MANUAL_RESTART_NO          (0x00000000)
#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_MANUAL_RESTART_YES         (0x00000001)

#define LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channel;
    LwU32                       reason;
    LwU32                       restartMode;
    LwU32                       assert;
    LwU32                       useExceptArg;
    LwU32                       exceptArg;

    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);

    LwU32                       manualRestart;
    LwU32                       exceptCnt;
} LW5070_CTRL_CMD_GET_EXCEPTION_RESTART_MODE_PARAMS;


/*
 * LW5070_CTRL_CMD_GET_SV_RESTART_MODE
 *
 * This command gets the display exception restart mode.
 * 
 *      whichSv
 *          The supervisor interrupt for which the mode is desired.
 *
 *      restartMode
 *          The restart mode RM would use when the supervisor intr happens.
 *
 *      hEvent
 *          The handle to the OS event that RM will use to call the client
 *          when the next supervisor intr happens.
 *      
 *      clientRestart
 *          Whether the client would get an opportunity to restart the HW when
 *          an OS event is sent for the supervisor event.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE                        (0x50700604) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_SV_RESTART_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_WHICH_SV               1:0
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_WHICH_SV_ONE           (0x00000000)
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_WHICH_SV_TWO           (0x00000001)
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_WHICH_SV_THREE         (0x00000002)

#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_RESTART_MODE           1:0
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_RESTART_MODE_RESUME    (0x00000000)
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_RESTART_MODE_SKIP      (0x00000001)
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_RESTART_MODE_REPLAY    (0x00000002)

#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_EXELWTE_RM_SV_CODE     0:0
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_EXELWTE_RM_SV_CODE_NO  (0x00000000)
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_EXELWTE_RM_SV_CODE_YES (0x00000001)

#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_CLIENT_RESTART         0:0
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_CLIENT_RESTART_NO      (0x00000000)
#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_CLIENT_RESTART_YES     (0x00000001)

#define LW5070_CTRL_CMD_GET_SV_RESTART_MODE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW5070_CTRL_CMD_GET_SV_RESTART_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       whichSv;
    LwU32                       restartMode;
    LwU32                       exelwteRmSvCode;
    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);

    LwU32                       clientRestart;
} LW5070_CTRL_CMD_GET_SV_RESTART_MODE_PARAMS;


/*
 * LW5070_CTRL_CMD_SET_SV_RESTART_MODE
 *
 * This command sets the display exception restart mode.
 * 
 *      whichSv
 *          The supervisor interrupt for which the mode needs to be set.
 *
 *      restartMode
 *          The restart mode RM should use when the supervisor intr happens.
 *
 *      hEvent
 *          The handle to the OS event that RM will use to call the client
 *          when the next supervisor intr happens.
 *      
 *      clientRestart
 *          Whether the client wants to manually restart the HW when an OS
 *          event is sent for the supervisor event.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE                        (0x50700605) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_SV_RESTART_MODE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_WHICH_SV               1:0
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_WHICH_SV_ONE           (0x00000000)
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_WHICH_SV_TWO           (0x00000001)
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_WHICH_SV_THREE         (0x00000002)

#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_RESTART_MODE           1:0
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_RESTART_MODE_RESUME    (0x00000000)
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_RESTART_MODE_SKIP      (0x00000001)
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_RESTART_MODE_REPLAY    (0x00000002)

#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_EXELWTE_RM_SV_CODE     0:0
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_EXELWTE_RM_SV_CODE_NO  (0x00000000)
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_EXELWTE_RM_SV_CODE_YES (0x00000001)

#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_CLIENT_RESTART         0:0
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_CLIENT_RESTART_NO      (0x00000000)
#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_CLIENT_RESTART_YES     (0x00000001)

#define LW5070_CTRL_CMD_SET_SV_RESTART_MODE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW5070_CTRL_CMD_SET_SV_RESTART_MODE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       whichSv;
    LwU32                       restartMode;
    LwU32                       exelwteRmSvCode;
    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);

    LwU32                       clientRestart;
} LW5070_CTRL_CMD_SET_SV_RESTART_MODE_PARAMS;


/*
 * LW5070_CTRL_CMD_GET_PREV_MODESWITCH_FLAGS
 *
 * This command gets the modeswitch actions taken
 * 
 *      blankHead
 *          Whether head was blanked.
 *
 *      shutdownHead
 *          Whether head was shutdown.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_PREV_MODESWITCH_FLAGS (0x50700606) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_PREV_MODESWITCH_FLAGS_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_PREV_MODESWITCH_FLAGS_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW5070_CTRL_CMD_GET_PREV_MODESWITCH_FLAGS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwBool                      blankHead[LW5070_CTRL_CMD_MAX_HEADS];
    LwBool                      shutdownHead[LW5070_CTRL_CMD_MAX_HEADS];
} LW5070_CTRL_CMD_GET_PREV_MODESWITCH_FLAGS_PARAMS;

/*
 * LW5070_CTRL_CMD_GET_OVERLAY_FLIPCOUNT
 *
 * This command gets the flip count of the specified overlay channel.
 * 
 *      channelInstance
 *          This field indicates which of the two instances of the overlay
 *          channel the cmd is meant for.
 *      
 *      value
 *          The current flip count.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_CHANNEL
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_OVERLAY_FLIPCOUNT (0x5070060b) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_OVERLAY_FLIPCOUNT_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_OVERLAY_FLIPCOUNT_VALUE                        11:0
#define LW5070_CTRL_CMD_GET_OVERLAY_FLIPCOUNT_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW5070_CTRL_CMD_GET_OVERLAY_FLIPCOUNT_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelInstance;

    LwU32                       value;
} LW5070_CTRL_CMD_GET_OVERLAY_FLIPCOUNT_PARAMS;


/*
 * LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT
 *
 * This command forces the flip count of overlay channel to a specific value.
 *
 *      channelInstance
 *          This field indicates which of the two instances of the overlay
 *          channel the cmd is meant for.
 *      
 *      forceCount
 *          The flip count value to set.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_CHANNEL
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT                    (0x5070060c) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_FORCECOUNT                    0:0
#define LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_FORCECOUNT_INIT    (0x00000000)
#define LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_FORCECOUNT_DISABLE (0x00000000)
#define LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_FORCECOUNT_ENABLE  (0x00000001)
#define LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelInstance;

    LwU32                       forceCount;
} LW5070_CTRL_CMD_SET_OVERLAY_FLIPCOUNT_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF
 *
 * This command let's RM clients receive a notification (via a notifier bit)
 * while RM is polling for a particular channel state and tell RM to block
 * inside the polling loo[ until the client sets another notifier bit.
 *
 *      channelClass
 *          This field indicates the hw class number ([50/82]7A-[50/82]7E)
 *          as defined in cl[50/82]7A-[50/82]7E.h.
 *
 *      channelInstance
 *          This field indicates which of the two instances of the channel
 *          (in case there are two. ex: base, overlay etc) the cmd is meant for.
 *          Note that core channel has only one instance and the field should
 *          be set to 0 for core channel.
 *
 *      channelState
 *          This field specifies the state for which the notification would be
 *          generated by RM if RM starts polling for the state. This is a bit
 *          mask of states. So, client can specify multiple states and
 *          RM would send notification when it starts polling for any one of
 *          those states.
 *
 *      hNotifierCtxDma
 *          Handle to the ctx dma for the notifier that is used for
 *          communication between RM and client. Set this to 0 to clear
 *          a previously set up notification.
 *      
 *      offset
 *          Offset within the notifier context dma where the notifier begins
 *          Offset must be 4 byte aligned.
 *
 * Most common usage scenario: Client wants to do a vbios grab and modeset
 * while RM is busy polling for dealloc to happen. To achieve this, client
 * would send this ctrl cmd specifying
 * (1) The desired channel (via channelClass and channelInstance)
 * (2) The channel state for which it needs notification (_DEALLOC in this
 *     example).
 * (3) A notifier ctx dma (and offset) which RM will use to tell client that it
 *     has started polling for channelState and client will use to tell RM to
 *     grab the display back, resume driver operation and finish dealloc.
 *
 * Subsequently, the client will
 * (1) Fork a thread that's polling for the bit set up by RM and the sole job
 *     of this thread is to do a vbios grab and modeset.
 * (2) Call RM to dealloc the channel. RM will hit dealloc bit, notice that
 *     client has asked for notitification and set up the notifier before
 *     it starts polling for clearance from client to resume driver operation.
 *
 * Once RM has set the bit for the forked client thread to do a vbios modeset,
 * vbios modeset will happen, and thread will tell RM to continue. This will
 * cause RM to resume driver operation, finish dealloc and unblock the parent
 * client thread which issued the call to dealloc.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF (0x5070060e) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_IDLE          LW5070_CTRL_CMD_CHANNEL_STATE_IDLE
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_WRTIDLE       LW5070_CTRL_CMD_CHANNEL_STATE_WRTIDLE
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_EMPTY         LW5070_CTRL_CMD_CHANNEL_STATE_EMPTY
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_FLUSHED       LW5070_CTRL_CMD_CHANNEL_STATE_FLUSHED
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_BUSY          LW5070_CTRL_CMD_CHANNEL_STATE_BUSY
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_DEALLOC       LW5070_CTRL_CMD_CHANNEL_STATE_DEALLOC
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_DEALLOC_LIMBO LW5070_CTRL_CMD_CHANNEL_STATE_DEALLOC_LIMBO
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_LIMBO1        LW5070_CTRL_CMD_CHANNEL_STATE_LIMBO1
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_LIMBO2        LW5070_CTRL_CMD_CHANNEL_STATE_LIMBO2
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_FCODEINIT     LW5070_CTRL_CMD_CHANNEL_STATE_FCODEINIT
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_FCODE         LW5070_CTRL_CMD_CHANNEL_STATE_FCODE
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_VBIOSINIT     LW5070_CTRL_CMD_CHANNEL_STATE_VBIOSINIT
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_VBIOSOPER     LW5070_CTRL_CMD_CHANNEL_STATE_VBIOSOPER
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_UNCONNECTED   LW5070_CTRL_CMD_CHANNEL_STATE_UNCONNECTED
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_INITIALIZE    LW5070_CTRL_CMD_CHANNEL_STATE_INITIALIZE
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_SHUTDOWN1     LW5070_CTRL_CMD_CHANNEL_STATE_SHUTDOWN1
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_SHUTDOWN2     LW5070_CTRL_CMD_CHANNEL_STATE_SHUTDOWN2
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_CHN_STATE_INIT          LW5070_CTRL_CMD_CHANNEL_STATE_INIT

#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;
    LwU32                       channelState;
    LwHandle                    hNotifierCtxDma;
    LwU32                       offset;
} LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_NOTIFICATION
 *
 * This is the structure of the notifier used in
 * LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF ctrl cmd.
 *
 *      blockRMDuringPoll
 *          This field is set by the client to _YES to block RM, _NO to let
 *          RM continue. In the example cited above, client would set this
 *          field to _YES before calling RM to dealloc the channel.
 *          Access: Client can read/write this field. RM can only read the
 *          field.
 *
 *      rmIsBusyPolling
 *          RM will set this field to
 *          (1) _YES the moment it goes into the polling loop.
 *          (2) _NO the moment it goes out of the polling loop.
 *          In the example cited above, child thread should poll for this
 *          bit to go to _YES, do a vbios modeset and then set
 *          blockRMDuringPoll to _NO. Client should init this field to _NO
 *          before calling RM to dealloc the channel.
 *          Access: Client can write to this field to clear it before it sends
 *          the ctrl cmd. Subsequently, client can only read this field. RM can
 *          read/write the field.
 */
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_BLOCK_RM_DURING_POLL_NO  0x0
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_BLOCK_RM_DURING_POLL_YES 0x1

#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_RM_IS_BUSY_POLLING_NO    0x0
#define LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_RM_IS_BUSY_POLLING_YES   0x1

typedef struct LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_NOTIFICATION {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       blockRMDuringPoll;
    LwU32                       rmIsBusyPolling;
} LW5070_CTRL_CMD_SET_CHN_STATE_POLL_NOTIF_NOTIFICATION;

/*
 * LW5070_CTRL_CMD_SET_DMI_ELV
 *
 * This command controls the timing of the early loadv signal from the
 * display memory interface (DMI) to the display software interface (DSI).
 *
 * Warning: This interface is low level and its meaning may change slightly
 *   from generation to generation.
 *
 *      head
 *          This parameter controls which channel (0 or 1) we're configuring.
 *      
 *      {what, value}
 *            The pair consisting of one of the following:
 *            
 *            LW5070_CTRL_CMD_SET_DMI_ELV_ADVANCE
 *              Prior to the end of each contract, the dmi must signal the early loadv to the
 *              dsi. This register is used to program the number of lines prior to the end of
 *              the contract that the dmi will generate the early loadv signal to the dsi.
 *              The lines here are in post-FOS units.
 *
 *            LW5070_CTRL_CMD_SET_DMI_ELV_MIN_DELAY
 *              After loadv arrives in the dmi block, MIN_DELAY is used to set the minimum
 *              amount of time in microseconds before the early loadv can be sent back to the
 *              dsi. It gives the dsi enough time to send bundles on the bundle interface.
 *              The RM needs to recallwlate MIN_DELAY whenever the resolution changes
 *              using the following formula:
 *              MIN_DELAY <= 25-50% of (H_ACTIVE * V_ACTIVE * PCLK_PERIOD(in microseconds))
 *        
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_DMI_ELV           (0x5070060f) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_DMI_ELV_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_DMI_ELV_ADVANCE   0x1
#define LW5070_CTRL_CMD_SET_DMI_ELV_MIN_DELAY 0x2
#define LW5070_CTRL_CMD_SET_DMI_ELV_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW5070_CTRL_CMD_SET_DMI_ELV_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;
    LwU32                       what;
    LwU32                       value;
} LW5070_CTRL_CMD_SET_DMI_ELV_PARAMS;

/*
 * LW5070_CTRL_DP_LINKCTL
 *
 * This structure describes linkCtl information.  The intended use is for
 * hardware verification and should not be used for normal driver behaviour.
 *
 *   dpIndex
 *     Whether DP_A or DP_B should be selected.
 *   linkCtl
 *     The linkCtl parameter is broken down as follows:
 *       LW5070_CTRL_DP_LINKCTL_ENABLE
 *         Controls whether the current DP port is enabled and data should be 
 *         driven to the associated pins.  If set to _NO, the current DP port 
 *         is not enabled and the port will be powered down.
 *       LW5070_CTRL_DP_LINKCTL_TUSIZE 
 *         Controls the Transfer unit size which as a valid range 64 to 32.
 *       LW5070_CTRL_DP_LINKCTL_SYNCMODE
 *         Specifies if the link clock and stream clock are asynchronous.
 *         The default control is DISABLE which is asynchronous.  If set
 *          to ENABLE, the link clock and stream clock are synchronous.
 *         user-accessible external digital port.
 *       LW5070_CTRL_DP_LINKCTL_SCRAMBLEREN  
 *         Controls whether the DP transmitter will scrambles data symbols 
 *         before transmission.  Set this value to DISABLE to disable scrambler
 *         and transmit all. The default control is scrambler enabled.
 *       LW5070_CTRL_DP_LINKCTL_ENHANCEDFRAME
 *         Controls whether enhanced framing symbol sequence for BS, SR, CPBS, 
 *         and CPSR are supported.  Set this value to ENABLE to be supported.
 *       LW5070_CTRL_DP_LINKCTL_LANECOUNT   
 *         This field controls # of lanes 0, 1, 2, 4 or 8 for DP configuration.
 *       LW5070_CTRL_DP_LINKCTL_CHANNELCODING 
 *         This field shall be set to 1 if the DP rcvr supports the Main Link 
 *         channel coding specification as specified in ANSI X3.230-1994, 
 *         clause 11. Reserved for future expansion if a different coding 
 *         scheme is used later.  For now this should always be set to _ENABLE.
 *       LW5070_CTRL_DP_LINKCTL_FORMAT_MODE
 *         This field is used to select between DP multistream and single stream
 *         mode in the SOR.
 *            LW5070_CTRL_DP_LINKCTL_FORMAT_MODE_SINGLE_STREAM
 *              This value selects DP single stream mode.
 *            LW5070_CTRL_DP_LINKCTL_FORMAT_MODE_MULTI_STREAM
 *              This value selects DP multi stream mode.
 *       LW5070_CTRL_DP_LINKCTL_TRAININGPTTRN  
 *         Controls the link training pattern setting
 *         _NOPATTERN: training not in progress or disabled.
 *         _TRAINING1: training pattern 1
 *         _TRAINING2: training pattern 2
 *         _TRAINING3: training pattern 3
 *       LW5070_CTRL_DP_LINKCTL_LINKQUALPTTRN  
 *         Controls which link quality pattern set is transmitted.
 *         _NOPATTERN: Link quality test pattern is not transmitted
 *         _D102: D10.2 test pattern (unscrambled) transmitted 
 *                (same as training pattern 1)
 *         _SBLERRRATE: Symbol error rate measurement pattern transmitted.
 *         _PRBS7: transmitted.
 *       LW5070_CTRL_DP_LINKCTL_COMPLIANCEPTTRN
 *         Controls whether a compliance test pattern is transmitted.  Default
 *          is none.  Set this value to 1 to transmit a color square.
 *       LW5070_CTRL_DP_LINKCTL_FORCE_IDLEPTTRN  
 *         This control allows manual setting of sending idle pattern. 
 *         Normal operation is default as automatic.
 */
typedef struct LW5070_CTRL_DP_LINKCTL_PARAMS {
    LwU32 dpIndex;

    LwU32 linkCtl;
} LW5070_CTRL_DP_LINKCTL_PARAMS;

#define LW5070_CTRL_DP_LINKCTL_INDEX_DP                            0:0
#define LW5070_CTRL_DP_LINKCTL_INDEX_DP_A                   0x00000000
#define LW5070_CTRL_DP_LINKCTL_INDEX_DP_B                   0x00000001

#define LW5070_CTRL_DP_LINKCTL_ENABLE                              0:0
#define LW5070_CTRL_DP_LINKCTL_ENABLE_NO                    0x00000000
#define LW5070_CTRL_DP_LINKCTL_ENABLE_YES                   0x00000001
#define LW5070_CTRL_DP_LINKCTL_TUSIZE                              8:2
#define LW5070_CTRL_DP_LINKCTL_TUSIZE_DEFAULT               0x00000040
#define LW5070_CTRL_DP_LINKCTL_SYNCMODE                          10:10
#define LW5070_CTRL_DP_LINKCTL_SYNCMODE_DISABLE             0x00000000
#define LW5070_CTRL_DP_LINKCTL_SYNCMODE_ENABLE              0x00000001
#define LW5070_CTRL_DP_LINKCTL_SCRAMBLEREN                       13:12
#define LW5070_CTRL_DP_LINKCTL_SCRAMBLEREN_DISABLE          0x00000000
#define LW5070_CTRL_DP_LINKCTL_SCRAMBLEREN_ENABLE_GALIOS    0x00000001
#define LW5070_CTRL_DP_LINKCTL_SCRAMBLEREN_ENABLE_FIBONACCI 0x00000002
#define LW5070_CTRL_DP_LINKCTL_ENHANCEDFRAME                     14:14
#define LW5070_CTRL_DP_LINKCTL_ENHANCEDFRAME_DISABLE        0x00000000
#define LW5070_CTRL_DP_LINKCTL_ENHANCEDFRAME_ENABLE         0x00000001
#define LW5070_CTRL_DP_LINKCTL_LANECOUNT                         20:16
#define LW5070_CTRL_DP_LINKCTL_LANECOUNT_ZERO               0x00000000
#define LW5070_CTRL_DP_LINKCTL_LANECOUNT_ONE                0x00000001
#define LW5070_CTRL_DP_LINKCTL_LANECOUNT_TWO                0x00000003
#define LW5070_CTRL_DP_LINKCTL_LANECOUNT_FOUR               0x0000000F
#define LW5070_CTRL_DP_LINKCTL_LANECOUNT_EIGHT              0x0000001F
#define LW5070_CTRL_DP_LINKCTL_CHANNELCODING                     22:22
#define LW5070_CTRL_DP_LINKCTL_CHANNELCODING_DISABLE        0x00000000
#define LW5070_CTRL_DP_LINKCTL_CHANNELCODING_ENABLE         0x00000001
#define LW5070_CTRL_DP_LINKCTL_FORMAT_MODE                       23:23
#define LW5070_CTRL_DP_LINKCTL_FORMAT_MODE_SINGLE_STREAM    0x00000000
#define LW5070_CTRL_DP_LINKCTL_FORMAT_MODE_MULTI_STREAM     0x00000001
#define LW5070_CTRL_DP_LINKCTL_TRAININGPTTRN                     25:24
#define LW5070_CTRL_DP_LINKCTL_TRAININGPTTRN_NOPATTERN      0x00000000
#define LW5070_CTRL_DP_LINKCTL_TRAININGPTTRN_TRAINING1      0x00000001
#define LW5070_CTRL_DP_LINKCTL_TRAININGPTTRN_TRAINING2      0x00000002
#define LW5070_CTRL_DP_LINKCTL_TRAININGPTTRN_TRAINING3      0x00000003
#define LW5070_CTRL_DP_LINKCTL_LINKQUALPTTRN                     27:26
#define LW5070_CTRL_DP_LINKCTL_LINKQUALPTTRN_NOPATTERN      0x00000000
#define LW5070_CTRL_DP_LINKCTL_LINKQUALPTTRN_D102           0x00000001
#define LW5070_CTRL_DP_LINKCTL_LINKQUALPTTRN_SBLERRRATE     0x00000002
#define LW5070_CTRL_DP_LINKCTL_LINKQUALPTTRN_PRBS7          0x00000003
#define LW5070_CTRL_DP_LINKCTL_COMPLIANCEPTTRN                   28:28
#define LW5070_CTRL_DP_LINKCTL_COMPLIANCEPTTRN_NOPATTERN    0x00000000
#define LW5070_CTRL_DP_LINKCTL_COMPLIANCEPTTRN_COLORSQUARE  0x00000001
#define LW5070_CTRL_DP_LINKCTL_FORCE_IDLEPTTRN                   31:31
#define LW5070_CTRL_DP_LINKCTL_FORCE_IDLEPTTRN_NO           0x00000000
#define LW5070_CTRL_DP_LINKCTL_FORCE_IDLEPTTRN_YES          0x00000001

/*
 * LW5070_CTRL_DP_CONFIG
 *
 * This structure describes config information.  The intended use is for
 * hardware verification and should not be used for normal driver behaviour.
 *
 *   dpIndex
 *     Whether DP_A or DP_B should be selected.
 *   linkCtl
 *     The linkCtl parameter is broken down as follows:
 *       LW5070_CTRL_DP_CONFIG_WATERMARK  
 *         This field defines the # of symbols that the DP logic will wait 
 *         for at the beginning of a line beore ending blanking.  This number
 *         should be based on the pixel rate and the link speed.  See 
 *         dev_disp.ref for more details on watermark setting.
 *       LW5070_CTRL_DP_CONFIG_ACTIVESYM_COUNT
 *         Controls the setting for active symbol count.
 *       LW5070_CTRL_DP_CONFIG_ACTIVESYM_FRAC 
 *         Controls the setting for active symbol count.
 *       LW5070_CTRL_DP_CONFIG_ACTIVESYM_POLARITY  
 *         Controls the active symbol polarity.
 *       LW5070_CTRL_DP_CONFIG_ACTIVESYM_CNTL  
 *         Controls the active symbol control.
 *       LW5070_CTRL_DP_CONFIG_IDLE_BEFORE_ATTACH
 *         When a head is attached to the SOR while in DisplayPort mode, 
 *         the OR will send out five idle patterns before acknowledging the 
 *         attach command. The DP spec requires that at least
 *         five idle patterns are sent before changing any timing parameters.
 *         This give the hardware time to callwlate a new M value if the pclk 
 *         frequency changed.  Controls as follows:
 *         _ENABLE: wait 5 idle patterns after attach before acknowledging it
 *         _DISABLE: acknowledge attach right away.
 *       LW5070_CTRL_DP_CONFIG_RD_RESET_VAL  
 *         When training pattern 2 begins, the running disparity of the 
 *         8B10B encoding is reset. Some panels may expect positive polarity 
 *         for the 1st symbol, some may expect negative.
 *         _POSITIVE: The internal running disparity is reset to positive, 
 *         so the first 10b symbol sent out will have negative disparity.
 *         _NEGATIVE: The internal running disparity is reset to negative, 
 *         so the first 10b symbol send out will have positive disparity.
 */
typedef struct LW5070_CTRL_DP_CONFIG_PARAMS {
    LwU32 dpIndex;

    LwU32 config;
} LW5070_CTRL_DP_CONFIG_PARAMS;

#define LW5070_CTRL_DP_CONFIG_INDEX_DP                             0:0
#define LW5070_CTRL_DP_CONFIG_INDEX_DP_A                   0x00000000
#define LW5070_CTRL_DP_CONFIG_INDEX_DP_B                   0x00000001

#define LW5070_CTRL_DP_CONFIG_WATERMARK                            5:0
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_COUNT                     14:8
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_FRAC                     19:16
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_POLARITY                 24:24
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_POLARITY_NEGATIVE  0x00000000
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_POLARITY_POSITIVE  0x00000001
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_CNTL                     26:26
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_CNTL_DISABLE       0x00000000
#define LW5070_CTRL_DP_CONFIG_ACTIVESYM_CNTL_ENABLE        0x00000001
#define LW5070_CTRL_DP_CONFIG_IDLE_BEFORE_ATTACH                 28:28
#define LW5070_CTRL_DP_CONFIG_IDLE_BEFORE_ATTACH_DISABLE   0x00000000
#define LW5070_CTRL_DP_CONFIG_IDLE_BEFORE_ATTACH_ENABLE    0x00000001
#define LW5070_CTRL_DP_CONFIG_RD_RESET_VAL                       31:31
#define LW5070_CTRL_DP_CONFIG_RD_RESET_VAL_POSITIVE        0x00000000
#define LW5070_CTRL_DP_CONFIG_RD_RESET_VAL_NEGATIVE        0x00000001

/*
 * LW5070_CTRL_CMD_SET_SF_BLANK
 *
 * This command programs the specified blanking settings for the specified SF.
 *
 *      sfNumber
 *          The SF number for which the blanking settings need to be programmed.
 *
 *      transition
 *          Whether new settings must be applied immediately or at next vsync.
 *
 *      status
 *          Whether to blank or unblank. This has the effect of
 *          clearing or enabling the override depending on the current status.
 *
 *      waitForCompletion
 *          Whether or not to wait until new settings are in effect.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_SF_BLANK                       (0x50700618) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_SF_BLANK_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_SF_BLANK_TRANSITION                             0:0
#define LW5070_CTRL_CMD_SET_SF_BLANK_TRANSITION_IMMEDIATE  0x00000000
#define LW5070_CTRL_CMD_SET_SF_BLANK_TRANSITION_NEXT_VSYNC 0x00000001

#define LW5070_CTRL_CMD_SET_SF_BLANK_STATUS                                 0:0
#define LW5070_CTRL_CMD_SET_SF_BLANK_STATUS_UNBLANK        0x00000000
#define LW5070_CTRL_CMD_SET_SF_BLANK_STATUS_BLANK          0x00000001

#define LW5070_CTRL_CMD_SET_SF_BLANK_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW5070_CTRL_CMD_SET_SF_BLANK_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       sfNumber;

    LwU32                       transition;
    LwU32                       status;
    LwBool                      bWaitForCompletion;
} LW5070_CTRL_CMD_SET_SF_BLANK_PARAMS;


#define LW5070_CTRL_IMP_MAX_OVERRIDE_ENTRIES 100

typedef struct LW5070_CTRL_IMP_OVERRIDE_ENTRY {
    LwU32 key;
    LW_DECLARE_ALIGNED(LwU64 value, 8);
} LW5070_CTRL_IMP_OVERRIDE_ENTRY;

#define LW5070_CTRL_CMD_SET_IMP_OVERRIDES_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW5070_CTRL_CMD_SET_IMP_OVERRIDES_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       numberOfEntries;
    LW_DECLARE_ALIGNED(LW5070_CTRL_IMP_OVERRIDE_ENTRY impOverrideEntries[LW5070_CTRL_IMP_MAX_OVERRIDE_ENTRIES], 8);
} LW5070_CTRL_CMD_SET_IMP_OVERRIDES_PARAMS;
typedef struct LW5070_CTRL_CMD_SET_IMP_OVERRIDES_PARAMS *PLW5070_CTRL_CMD_SET_IMP_OVERRIDES_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_IMP_OVERRIDES
 *
 * This command sets the configuration for overriding the parameters in IMP outputs. The override
 * configuration is used across all the IMP calls. This is a verif only command intended to be used 
 * in mods based tests especially on simulation for overriding the HW configuration.
 *
 * NOTE: This call ignores the override configuration for  
 * dispIMP input and  FBIMP input, where dispIMP inputs are supplied via
 * DTI, FBIMP inputs were not implemented yet.
 * 
 *  numberOfEntries
 *     This field indicates the number of Override Entries given by the client.
 * 
 *  impOverrideEntries
 *     This field indicates the entries like ("key", "value") pairs. 
 *     "key" is one of the IMP output keys like ASR, MEMPOOL etc., defined in ctrl5070impoverrides.h
 *     "value" is a value which needs to be overridden for a key.
 * 
 *     Examples:
 *     Ex1: LW5070_CTRL_IMP_OVERRIDE_SET_HEAD_NUMBER, 0,
 *          LW5070_CTRL_IMP_OVERRIDE_OUTPUT_MEMPOOL, 0x100
 *      Above Entries will Override MEMPOOL for Head number "0" with value "0x100". LW5070_CTRL_IMP_OVERRIDE_SET_HEAD_NUMBER is
 *      one of the "ControlOverrides", ControlOverrides are used for controlling the given array of Overrides like setting "HeadNumber" 
 *      or "Index" value of an Entry and deleting the stored entries. 
 * 
 *     Subsequent override entries followed by LW5070_CTRL_IMP_OVERRIDE_SET_HEAD_NUMBER will get HeadNumber as a value, in the above example it is '0'.
 * 
 *     Ex2: LW5070_CTRL_IMP_OVERRIDE_CLEAR_ALL_OVERRIDES, 0
 *          Another ControlOverride used to clear all the stored Overrides. 
 * 
 *   Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * Usage Information: To override an IMP output, clients need to call
 * LW5070_CTRL_CMD_SET_IMP_OVERRIDES with the given OverrideEntries 
 *
 */

#define LW5070_CTRL_CMD_SET_IMP_OVERRIDES (0x50700620) /* finn: Evaluated from "(FINN_LW50_DISPLAY_VERIF_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_IMP_OVERRIDES_PARAMS_MESSAGE_ID" */

/* _ctrl5070verif_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

