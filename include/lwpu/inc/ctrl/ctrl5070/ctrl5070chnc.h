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
// Source file: ctrl/ctrl5070/ctrl5070chnc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl5070/ctrl5070base.h"
#include "ctrl5070common.h"
#include "lwdisptypes.h"

#define LW5070_CTRL_CMD_NUM_DISPLAY_ID_DWORDS_PER_HEAD 2

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_CMD_IDLE_CHANNEL
 *
 * This command tries to wait or forces the desired channel state.
 *
 *      channelClass
 *          This field indicates the hw class number (507A-507E)
 *
 *      channelInstance
 *          This field indicates which of the two instances of the channel
 *          (in case there are two. ex: base, overlay etc) the cmd is meant for.
 *          Note that core channel has only one instance and the field should
 *          be set to 0 for core channel.
 *
 *      desiredChannelStateMask
 *          This field indicates the desired channel states. When more than
 *          one bit is set, RM will return whenever it finds hardware on one
 *          of the states in the bistmask.
 *          Normal options are IDLE, WRTIDLE, QUIESCENT1 and QUIESCENT2.
 *          Verif only options include EMPTY, FLUSHED and BUSY as well.
 *          Note:
 *              (1) When QUIESCENT1 or QUIESCENT2 is chosen only one bit should
 *                  be set in the bitmask. RM will ignore any other state.
 *              (2) Accelerators should not be required for QUIESCENT states as
 *                  RM tries to ensure QUIESCENT forcibly on it's own.
 *
 *      accelerators
 *          What accelerator bits should be used if RM timesout trying to
 *          wait for the desired state. This is not yet implemented since it
 *          should normally not be required to use these. Usage of accelerators
 *          should be restricted and be done very carefully as they may have
 *          undesirable effects.
 *          NOTE: accelerators should not be used directly in production code.
 *
 *      timeout
 *          Timeout to use when waiting for the desired state. This is also for
 *          future expansion and not yet implemented.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_TIMEOUT
 */
#define LW5070_CTRL_CMD_IDLE_CHANNEL                   (0x50700101) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_IDLE_CHANNEL_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_IDLE_CHANNEL_STATE_IDLE                        LW5070_CTRL_CMD_CHANNEL_STATE_IDLE
#define LW5070_CTRL_IDLE_CHANNEL_STATE_WRTIDLE                     LW5070_CTRL_CMD_CHANNEL_STATE_WRTIDLE
#define LW5070_CTRL_IDLE_CHANNEL_STATE_QUIESCENT1                  LW5070_CTRL_CMD_CHANNEL_STATE_QUIESCENT1
#define LW5070_CTRL_IDLE_CHANNEL_STATE_QUIESCENT2                  LW5070_CTRL_CMD_CHANNEL_STATE_QUIESCENT2
#define LW5070_CTRL_IDLE_CHANNEL_STATE_EMPTY                       LW5070_CTRL_CMD_CHANNEL_STATE_EMPTY

#define LW5070_CTRL_IDLE_CHANNEL_STATE_FLUSHED                     LW5070_CTRL_CMD_CHANNEL_STATE_FLUSHED
#define LW5070_CTRL_IDLE_CHANNEL_STATE_BUSY                        LW5070_CTRL_CMD_CHANNEL_STATE_BUSY

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#define LW5070_CTRL_IDLE_CHANNEL_ACCL_NONE             (0x00000000)
#define LW5070_CTRL_IDLE_CHANNEL_ACCL_IGNORE_PI                    (LWBIT(0))
#define LW5070_CTRL_IDLE_CHANNEL_ACCL_SKIP_NOTIF                   (LWBIT(1))
#define LW5070_CTRL_IDLE_CHANNEL_ACCL_SKIP_SEMA                    (LWBIT(2))
#define LW5070_CTRL_IDLE_CHANNEL_ACCL_IGNORE_INTERLOCK             (LWBIT(3))
#define LW5070_CTRL_IDLE_CHANNEL_ACCL_IGNORE_FLIPLOCK              (LWBIT(4))
#define LW5070_CTRL_IDLE_CHANNEL_ACCL_TRASH_ONLY                   (LWBIT(5))
#define LW5070_CTRL_IDLE_CHANNEL_ACCL_TRASH_AND_ABORT              (LWBIT(6))

#define LW5070_CTRL_IDLE_CHANNEL_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW5070_CTRL_IDLE_CHANNEL_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;

    LwU32                       desiredChannelStateMask;
    LwU32                       accelerators;        // For future expansion. Not yet implemented
    LwU32                       timeout;             // For future expansion. Not yet implemented
    LwBool                      restoreDebugMode;
} LW5070_CTRL_IDLE_CHANNEL_PARAMS;

/*
 * LW5070_CTRL_CMD_STOP_OVERLAY
 *
 * This command tries to turn the overlay off ASAP.
 *
 *      channelInstance
 *          This field indicates which of the two instances of the overlay
 *          channel the cmd is meant for.
 *
 *      notifyMode
 *          This field indicates the action RM should take once the overlay has
 *          been successfully stopped. The options are (1) Set a notifier
 *          (2) Set the notifier and generate and OS event
 *
 *      hNotifierCtxDma
 *          Handle to the ctx dma for the notifier that must be written once
 *          overlay is stopped. The standard LwNotification notifier structure
 *          is used.
 *
 *      offset
 *          Offset within the notifier context dma where the notifier begins
 *          Offset must be 16 byte aligned.
 *
 *      hEvent
 *          Handle to the event that RM must use to awaken the client when
 *          notifyMode is WRITE_AWAKEN.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT: Invalid notify mode
 *      LW_ERR_ILWALID_CHANNEL: When the overlay is unallocated
 *      LW_ERR_ILWALID_OWNER: Callee isn't the owner of the channel
 *      LW_ERR_ILWALID_OBJECT_HANDLE: Notif ctx dma not found
 *      LW_ERR_ILWALID_OFFSET: Bad offset within notif ctx dma
 *      LW_ERR_INSUFFICIENT_RESOURCES
 *      LW_ERR_TIMEOUT: RM timedout waiting to inject methods
 */
#define LW5070_CTRL_CMD_STOP_OVERLAY                          (0x50700102) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_STOP_OVERLAY_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_STOP_OVERLAY_NOTIFY_MODE_WRITE        (0x00000000)
#define LW5070_CTRL_CMD_STOP_OVERLAY_NOTIFY_MODE_WRITE_AWAKEN (0x00000001)

#define LW5070_CTRL_CMD_STOP_OVERLAY_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW5070_CTRL_CMD_STOP_OVERLAY_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelInstance;
    LwU32                       notifyMode;
    LwHandle                    hNotifierCtxDma;
    LwU32                       offset;
    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);
} LW5070_CTRL_CMD_STOP_OVERLAY_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_CMD_SET_SEMA_ACQ_DELAY
 *
 * This command sets the delay between semaphore reads
 *
 *      delayUs
 *          Delay between semaphore reads in microseconds.
 *          A delay of 0us will result in an error (ILWALID_ARGUMENT)
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_SEMA_ACQ_DELAY (0x50700104) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_SEMA_ACQ_DELAY_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_SEMA_ACQ_DELAY_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW5070_CTRL_CMD_SET_SEMA_ACQ_DELAY_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       delayUs;
} LW5070_CTRL_CMD_SET_SEMA_ACQ_DELAY_PARAMS;

/*
 * defines for LW5070_CTRL_CMD_SET_ERRMASK_PARAMS.errorCode
 */
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_NONE                                        (0x00000000)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507A_ILWALID_01                             (0x00000001)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507A_ILWALID_CHECK                          (0x00000002)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507A_ILWALID_ENUM_RANGE                     (0x00000003)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507B_ILWALID_CHECK                          (0x00000004)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507B_ILWALID_ENUM_RANGE                     (0x00000005)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_CHECK                          (0x00000006)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_CORE_CONSISTENCY               (0x00000007)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_ENUM_RANGE                     (0x00000008)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_IMPROPER_SURFACE_SPECIFICATION (0x00000009)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_IMPROPER_USAGE                 (0x0000000A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_INTERNAL                       (0x0000000B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_LUT_VIOLATION                  (0x0000000C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_NOTIFIER_VIOLATION             (0x0000000D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_SEMAPHORE_VIOLATION            (0x0000000E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507C_ILWALID_SURFACE_VIOLATES_CONTEXT_DMA   (0x0000000F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_BLOCKING_VIOLATION             (0x00000010)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_CHECK                          (0x00000011)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_CONFIGURATION                  (0x00000012)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_LWRSOR_VIOLATION               (0x00000013)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_ENUM_RANGE                     (0x00000014)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_IMPROPER_SURFACE_SPECIFICATION (0x00000015)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_INTERNAL                       (0x00000016)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_ILWALID_RASTER                 (0x00000017)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_ISO_VIOLATION                  (0x00000018)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_LOCKING_VIOLATION              (0x00000019)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_LUT_VIOLATION                  (0x0000001A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_NOTIFIER_VIOLATION             (0x0000001B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_SCALING_VIOLATION              (0x0000001C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_SURFACE_VIOLATES_CONTEXT_DMA   (0x0000001D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507D_ILWALID_VIEWPORT_VIOLATION             (0x0000001E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_CHECK                          (0x0000001F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_CORE_CONSISTENCY               (0x00000020)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_ENUM_RANGE                     (0x00000021)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_IMPROPER_SURFACE_SPECIFICATION (0x00000022)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_IMPROPER_USAGE                 (0x00000023)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_INTERNAL                       (0x00000024)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_NOTIFIER_VIOLATION             (0x00000025)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_SEMAPHORE_VIOLATION            (0x00000026)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_507E_ILWALID_SURFACE_VIOLATES_CONTEXT_DMA   (0x00000027)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_001                        (0x00000028)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_002                        (0x00000029)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_003                        (0x0000002A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_004                        (0x0000002B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_005                        (0x0000002C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_007                        (0x0000002D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_008                        (0x0000002E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_009                        (0x0000002F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_010                        (0x00000030)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_011                        (0x00000031)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_012                        (0x00000032)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_013                        (0x00000033)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_014                        (0x00000034)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_015                        (0x00000035)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_016                        (0x00000036)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_017                        (0x00000037)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_018                        (0x00000038)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_019                        (0x00000039)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_020                        (0x0000003A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_021                        (0x0000003B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_022                        (0x0000003C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_023                        (0x0000003D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_024                        (0x0000003E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_025                        (0x0000003F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_026                        (0x00000040)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_027                        (0x00000041)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_028                        (0x00000042)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_029                        (0x00000043)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_030                        (0x00000044)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_031                        (0x00000045)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_032                        (0x00000046)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_033                        (0x00000047)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_034                        (0x00000048)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_035                        (0x00000049)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_036                        (0x0000004A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_037                        (0x0000004B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_038                        (0x0000004C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_039                        (0x0000004D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_040                        (0x0000004E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_041                        (0x0000004F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_042                        (0x00000050)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_043                        (0x00000051)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_044                        (0x00000052)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_CORE_STATE_ERROR_045                        (0x00000053)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_001                        (0x00000054)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_002                        (0x00000055)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_003                        (0x00000056)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_004                        (0x00000057)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_005                        (0x00000058)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_006                        (0x00000059)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_007                        (0x0000005A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_008                        (0x0000005B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_009                        (0x0000005C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_010                        (0x0000005D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_011                        (0x0000005E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_012                        (0x0000005F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_013                        (0x00000060)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_014                        (0x00000061)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_015                        (0x00000062)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_016                        (0x00000063)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_017                        (0x00000064)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_018                        (0x00000065)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_019                        (0x00000066)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_020                        (0x00000067)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_021                        (0x00000068)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_022                        (0x00000069)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_023                        (0x0000006A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_024                        (0x0000006B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_025                        (0x0000006C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_026                        (0x0000006D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_027                        (0x0000006E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_028                        (0x0000006F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_029                        (0x00000070)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_030                        (0x00000071)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_031                        (0x00000072)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_032                        (0x00000073)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_033                        (0x00000074)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_034                        (0x00000075)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_035                        (0x00000076)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_036                        (0x00000077)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_037                        (0x00000078)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_038                        (0x00000079)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_039                        (0x0000007A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_040                        (0x0000007B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_041                        (0x0000007C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_042                        (0x0000007D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_043                        (0x0000007E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_044                        (0x0000007F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_045                        (0x00000080)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_046                        (0x00000081)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_047                        (0x00000082)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_048                        (0x00000083)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_049                        (0x00000084)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_050                        (0x00000085)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_051                        (0x00000086)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_052                        (0x00000087)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_053                        (0x00000088)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_BASE_STATE_ERROR_054                        (0x00000089)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_001                     (0x0000008A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_002                     (0x0000008B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_003                     (0x0000008C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_004                     (0x0000008D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_005                     (0x0000008E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_006                     (0x0000008F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_007                     (0x00000090)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_008                     (0x00000091)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_009                     (0x00000092)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_010                     (0x00000093)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_011                     (0x00000094)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_012                     (0x00000095)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_013                     (0x00000096)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_014                     (0x00000097)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_015                     (0x00000098)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_016                     (0x00000099)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_017                     (0x0000009A)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_018                     (0x0000009B)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_019                     (0x0000009C)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_020                     (0x0000009D)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_021                     (0x0000009E)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_022                     (0x0000009F)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_023                     (0x000000A0)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_024                     (0x000000A1)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_025                     (0x000000A2)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_026                     (0x000000A3)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_027                     (0x000000A4)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_028                     (0x000000A5)
#define LW5070_CTRL_SET_ERRMASK_ERRCODE_OVERLAY_STATE_ERROR_029                     (0x000000A6)

/*
 * LW5070_CTRL_CMD_SET_ERRMASK
 *
 * This command sets the mask for error check for a channel and/or method.
 *
 *      channelClass
 *          This field indicates the hw class number (507A-507E).
 *
 *      channelInstance
 *          This field indicates which of the two instances of the channel
 *          (in case there are two. ex: base, overlay etc) the cmd is meant for.
 *          Note that core channel has only one instance and the field should
 *          be set to 0 for core channel.
 *
 *      method
 *          This field indicates a specific method offset within the specified
 *          channel to apply the error masking. This field is optional.
 *
 *      mode
 *          Specifies the error masking mode.
 *
 *      errorCode
 *          Indicates the specific state error code to be suppressed. Some mode
 *          will ignore this field.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_ILWALID_CHANNEL: When the channel is unallocated
 *      LW_ERR_INSUFFICIENT_RESOURCES: All error masks are in use
 *      LW_ERR_ILWALID_OWNER: Callee isn't the owner of the channel
 *
 */
#define LW5070_CTRL_CMD_SET_ERRMASK                                                 (0x50700105) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_ERRMASK_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_ERRMASK_MODE_DISABLE                                    (0x00000000)
#define LW5070_CTRL_CMD_SET_ERRMASK_MODE_ALL                                        (0x00000001)
#define LW5070_CTRL_CMD_SET_ERRMASK_MODE_ALL_ARG                                    (0x00000002)
#define LW5070_CTRL_CMD_SET_ERRMASK_MODE_ALL_STATE                                  (0x00000003)
#define LW5070_CTRL_CMD_SET_ERRMASK_MODE_METHOD_ARG                                 (0x00000004)
#define LW5070_CTRL_CMD_SET_ERRMASK_MODE_METHOD_STATE                               (0x00000005)
#define LW5070_CTRL_CMD_SET_ERRMASK_MODE_STATE_CODE                                 (0x00000006)

#define LW5070_CTRL_CMD_SET_ERRMASK_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW5070_CTRL_CMD_SET_ERRMASK_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;
    LwU32                       method;

    LwU32                       mode;
    LwU32                       errorCode;
} LW5070_CTRL_CMD_SET_ERRMASK_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_UNFR_UPD_EVENT
 *
 * This command is used by CPL to register the unfriendly update event that RM
 * generates upon seeing an unfriendly update from vbios.
 *
 *      hEvent
 *          The event handle.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_CHANNEL
 *      LW_ERR_ILWALID_CLIENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_UNFR_UPD_EVENT (0x50700107) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_UNFR_UPD_EVENT_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_UNFR_UPD_EVENT_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW5070_CTRL_CMD_SET_UNFR_UPD_EVENT_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);
} LW5070_CTRL_CMD_SET_UNFR_UPD_EVENT_PARAMS;

/*
 * LW5070_CTRL_CMD_PREP_FOR_RESUME_FROM_UNFRIENDLY_UPDATE
 *
 * This command is used by DD before doing a modeset to resume driver operation
 * after an unfriendly vbios update which may or may not be in the middle of a
 * modeswitch.
 *
 * There are no fields in the struct below right now but we might add in future.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_CHANNEL
 *      LW_ERR_ILWALID_CLIENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_PREP_FOR_RESUME_FROM_UNFR_UPD (0x50700108) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | 0x8" */

typedef struct LW5070_CTRL_CMD_PREP_FOR_RESUME_FROM_UNFR_UPD_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
} LW5070_CTRL_CMD_PREP_FOR_RESUME_FROM_UNFR_UPD_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW5070_CTRL_CMD_IS_MODE_POSSIBLE
 *
 * This command is used by DD to determine whether or not a given mode
 * is possible given the current lwclk, mclk, dispclk and potentially some
 * other parameters that are normally hidden from it. All the parameters
 * except IsPossible (output), Force422(output), MinPstate (input/output),
 * minPerfLevel (output), CriticalWatermark (output), worstCaseMargin (output),
 * and worstCaseDomain (output) params are supplied by the caller.
 *
 *      HeadActive
 *          Whether or not the params for this head are relevant.
 *
 *      PixelClock
 *          Frequency: Pixel clk frequency in KHz.
 *          Adj1000Div1001: 1000/1001 multiplier for pixel clock.
 *
 *      RasterSize
 *          Width: Total width of the raster. Also referred to as HTotal.
 *          Height: Total height of the raster. Also referred to as VTotal.
 *
 *      RasterBlankStart
 *          X: Start of horizontal blanking for the raster.
 *          Y: Start of vertical blanking for the raster.
 *
 *      RasterBlankEnd
 *          X: End of horizontal blanking for the raster.
 *          Y: End of vertical blanking for the raster.
 *
 *      RasterVertBlank2
 *          YStart: Start of second blanking for second field for an
 *              interlaced raster. This field is irrelevant when raster is
 *              progressive.
 *          YEnd: End of second blanking for second field for an
 *              interlaced raster. This field is irrelevant when raster is
 *              progressive.
 *
 *      Control
 *          RasterStructure: Whether the raster ir progressive or interlaced.
 *
 *      OutputScaler
 *          VerticalTaps: Vertical scaler taps.
 *          HorizontalTaps: Horizontal scaler taps.
 *          Force422: Whether OutputScaler is operating in 422 mode or not.
 *
 *      ViewportSizeOut
 *          Width: Width of output viewport.
 *          Height: Height of output viewport.
 *          Both the above fields are irrelevant for G80.
 *
 *      ViewportSizeOutMin
 *          Width: Minimum possible/expected width of output viewport.
 *          Height: Minimum possible/expected height of output viewport.
 *
 *      ViewportSizeIn
 *          Width: Width of input viewport.
 *          Height: Height of input viewport.
 *
 *      Params
 *          Format: Core channel's pixel format. See the enumerants following
 *              the variable declaration for possible options.
 *          SuperSample: Whether to use X1AA or X4AA in core channel.
 *              This parameter is ignored for G80.
 *
 *      BaseUsageBounds
 *          Usable: Whether or not the base channel is expected to be used.
 *          PixelDepth: Maximum pixel depth allowed in base channel.
 *          SuperSample: Whether or not X4AA is allowed in base channel.
 *          BaseLutUsage: Base LUT Size
 *          OutputLutUsage: Output LUT size
 *
 *      OverlayUsageBounds
 *          Usable: Whether or not the overlay channel is expected to be used.
 *          PixelDepth: Maximum pixel depth allowed in overlay channel.
 *          OverlayLutUsage: Overlay LUT Size
 *
 *      BaseLutLo
 *          Enable: Specifies Core Channel's Base LUT is enable or not.
 *          Mode: Specifies the LUT Mode.
 *          NeverYieldToBase: Specifies whether NEVER_YIELD_TO_BASE is enabled or not.
 *
 *      OutputLutLo
 *          Enable: Specifies Core Channel's Output LUT is enable or not.
 *          Mode: Specifies the LUT Mode.
 *          NeverYieldToBase: Specifies whether NEVER_YIELD_TO_BASE is enabled or not.
 *
 *      outputResourcePixelDepthBPP
 *          Specifies the output pixel depth with scaler mode.
 *
 *      CriticalWatermark
 *          If MinPState is set to _NEED_MIN_PSTATE, this will return the critical
 *          watermark level at the minimum Pstate.  Otherwise, this will return
 *          the critical watermark at the level that the IMP callwlations are
 *          otherwise performed at.
 *
 *      pixelReplicateMode
 *          Specifies the replication mode whether it is X2 or X4. Need to set the parameter
 *          to OFF if there is no pixel replication.
 *
 *      numSSTLinks
 *          Number of Single Stream Transport links which will be used by the
 *          SOR.  "0" means to use the number indicated by the most recent
 *          LW0073_CTRL_CMD_DP_SINGLE_HEAD_MULTI_STREAM_MODE_SST call.
 *
 *      RequestedOperation
 *          This parameter is used to determine whether
 *          1. DD is simplying querying whether or not the specified mode is
 *             possible (REQUESTED_OPER = _QUERY) or
 *          2. DD is about to set the specified mode and RM should make
 *             appropriate preparations to make the mode possible. DD should
 *             never pass in a mode that was never indicated by RM as possible
 *             when DD queried for the possibility of the mode. This
 *             corresponds to REQUESTED_OPER = _PRE_MODESET.
 *          3. DD just finished setting the specified mode. RM can go ahead
 *             and make changes like lowering the perf level if desired. This
 *             corresponds to REQUESTED_OPER = _POST_MODESET. This parameter is
 *             useful when we are at a higher perf level in a mode that's not
 *             possible at a lower perf level and want to go to a mode that is
 *             possible even at a lower perf level. In such cases, lowering
 *             perf level before modeset is complete is dangerous as it will
 *             cause underflow. RM will wait until the end of modeset to lower
 *             the perf level.
 *
 *      options
 *          Specifies a bitmask for options.
 *            LW5070_CTRL_IS_MODE_POSSIBLE_OPTIONS_GET_MARGIN
 *              Tells IMP to callwlate worstCaseMargin and worstCaseDomain.
 *
 *      IsPossible
 *          This is the first OUT param for this call. It indicates whether
 *          or not the current mode is possible.
 *
 *      MinPState
 *          MinPState is an IO (in/out) variable; it gives the minimum p-state
 *          value at which the mode is possible on a PStates 2.0 system if the
 *          parameter is initialized by the caller with _NEED_MIN_PSTATE.  If
 *          _NEED_MIN_PSTATE is not specified, IMP query will just run at the
 *          max available perf level and return results for that pstate.
 *
 *          If the minimum pstate is required, then MasterLockMode,
 *          MasterLockPin, SlaveLockMode, and SlaveLockPin must all be
 *          initialized.
 *
 *          On a PStates 3.0 system, the return value for MinPState is
 *          undefined, but minPerfLevel can return the minimum IMP v-pstate.
 *
 *      minPerfLevel
 *          On a PStates 3.0 system, minPerfLevel returns the minimum IMP
 *          v-pstate at which the mode is possible.  On a PStates 2.0 system,
 *          minPerfLevel returns the minimum perf level at which the mode is
 *          possible.
 *
 *          minPerfLevel is valid only if MinPState is initialized to
 *          _NEED_MIN_PSTATE.
 *
 *      worstCaseMargin
 *          Returns the ratio of available bandwidth to required bandwidth,
 *          multiplied by LW5070_CTRL_IMP_MARGIN_MULTIPLIER.  Available
 *          bandwidth is callwlated in the worst case bandwidth domain, i.e.,
 *          the domain with the least available margin.  Bandwidth domains
 *          include the IMP-relevant clock domains, and possibly other virtual
 *          bandwidth domains such as AWP.
 *
 *          Note that IMP checks additional parameters besides the bandwidth
 *          margins, but only the bandwidth margin is reported here, so it is
 *          possible for a mode to have a more restrictive domain that is not
 *          reflected in the reported margin result.
 *
 *          This result is not guaranteed to be valid if the mode is not
 *          possible.
 *
 *          Note also that the result is generally callwlated for the highest
 *          pstate possible (usually P0).  But if _NEED_MIN_PSTATE is specified
 *          with the MinPState parameter, the result will be callwlated for the
 *          min possible pstate (or the highest possible pstate, if the mode is
 *          not possible).
 *
 *          The result is valid only if
 *          LW5070_CTRL_IS_MODE_POSSIBLE_OPTIONS_GET_MARGIN is set in
 *          "options".
 *
 *      worstCaseDomain
 *          Returns a short text string naming the domain for the margin
 *          returned in "worstCaseMargin".  See "worstCaseMargin" for more
 *          information.
 *
 *      bUseCachedPerfState
 *          Indicates that RM should use cached values for the fastest
 *          available perf level (v-pstate for PStates 3.0 or pstate for
 *          PStates 2.0) and dispclk.  This feature allows the query call to
 *          execute faster, and is intended to be used, for example, during
 *          mode enumeration, when many IMP query calls are made in close
 *          succession, and perf conditions are not expected to change between
 *          query calls.  When IMP has not been queried recently, it is
 *          recommended to NOT use cached values, in case perf conditions have
 *          changed and the cached values no longer reflect the current
 *          conditions.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_GENERIC
 *
 * Assumptions/Limitations:
 *      - If the caller sends any methods to alter the State Cache, before calling of
 *          the following functions:
 *               LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_QUERY_USE_SC
 *               LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_PRE_MODESET_USE_SC
 *               LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_POST_MODESET_USE_SC
 *          the caller must repeatedly issue LW5070_CTRL_CMD_GET_CHANNEL_INFO, and delay until the
 *          returned channelState is either:
 *              LW5070_CTRL_CMD_GET_CHANNEL_INFO_STATE_IDLE,
 *              LW5070_CTRL_CMD_GET_CHANNEL_INFO_STATE_WRTIDLE, or
 *              LW5070_CTRL_CMD_GET_CHANNEL_INFO_STATE_EMPTY.
 *          This ensures that all commands have reached the State Cache before RM reads
 *              them.
 *
 *
 */
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE                                                  (0x50700109) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_QUERY                        (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_PRE_MODESET                  (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_POST_MODESET                 (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_QUERY_USE_SC                 (0x00000003)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_PRE_MODESET_USE_SC           (0x00000004)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_POST_MODESET_USE_SC          (0x00000005)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_REQUESTED_OPERATION_SUPERVISOR                   (0x00000007)

#define LW5070_CTRL_IS_MODE_POSSIBLE_OPTIONS_GET_MARGIN                                   (0x00000001)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_IS_POSSIBLE_NO                                   (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_IS_POSSIBLE_YES                                  (0x00000001)

#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_UNDEFINED                                    (0x00000000)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P0                                           (0x00000001)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P1                                           (0x00000002)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P2                                           (0x00000004)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P3                                           (0x00000008)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P8                                           (0x00000100)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P10                                          (0x00000400)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P12                                          (0x00001000)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P15                                          (0x00008000)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_MAX                                          LW5070_CTRL_IS_MODE_POSSIBLE_PSTATES_P15
#define LW5070_CTRL_IS_MODE_POSSIBLE_NEED_MIN_PSTATE                                      (0x10101010)
#define LW5070_CTRL_IS_MODE_POSSIBLE_NEED_MIN_PSTATE_DEFAULT                              (0x00000000)

#define LW5070_CTRL_IMP_MARGIN_MULTIPLIER                                                 (0x00000400)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_HEAD_ACTIVE_NO                                   (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_HEAD_ACTIVE_YES                                  (0x00000001)

#define LW5070_CTRL_IS_MODE_POSSIBLE_DISPLAY_ID_SKIP_IMP_OUTPUT_CHECK                     (0xAAAAAAAA)

#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_DEFAULT                  (0x00000000)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_16_422               (0x00000001)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_18_444               (0x00000002)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_20_422               (0x00000003)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_422               (0x00000004)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_444               (0x00000005)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_30_444               (0x00000006)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_32_422               (0x00000007)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_36_444               (0x00000008)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_48_444               (0x00000009)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PIXEL_CLOCK_ADJ1000DIV1001_NO                    (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PIXEL_CLOCK_ADJ1000DIV1001_YES                   (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_CONTROL_STRUCTURE_PROGRESSIVE                    (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_CONTROL_STRUCTURE_INTERLACED                     (0x00000001)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_VERTICAL_TAPS_1                    (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_VERTICAL_TAPS_2                    (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_VERTICAL_TAPS_3                    (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_VERTICAL_TAPS_3_ADAPTIVE           (0x00000003)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_VERTICAL_TAPS_5                    (0x00000004)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_HORIZONTAL_TAPS_1                  (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_HORIZONTAL_TAPS_2                  (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_SCALER_HORIZONTAL_TAPS_8                  (0x00000002)

#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_SCALER_FORCE422_MODE_DISABLE                  (0x00000000)
#define LW5070_CTRL_IS_MODE_POSSIBLE_OUTPUT_SCALER_FORCE422_MODE_ENABLE                   (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_I8                                 (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_VOID16                             (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_VOID32                             (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_RF16_GF16_BF16_AF16                (0x00000003)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_A8R8G8B8                           (0x00000004)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_A2B10G10R10                        (0x00000005)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_A8B8G8R8                           (0x00000006)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_R5G6B5                             (0x00000007)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_FORMAT_A1R5G5B5                           (0x00000008)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_SUPER_SAMPLE_X1AA                         (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_SUPER_SAMPLE_X4AA                         (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_USABLE_USE_LWRRENT             (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_USABLE_NO                      (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_USABLE_YES                     (0x00000002)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_PIXEL_DEPTH_USE_LWRRENT        (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_PIXEL_DEPTH_8                  (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_PIXEL_DEPTH_16                 (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_PIXEL_DEPTH_32                 (0x00000003)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_PIXEL_DEPTH_64                 (0x00000004)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_SUPER_SAMPLE_USE_LWRRENT       (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_SUPER_SAMPLE_X1AA              (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_SUPER_SAMPLE_X4AA              (0x00000002)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_BASE_LUT_USAGE_NONE            (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_BASE_LUT_USAGE_257             (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_BASE_LUT_USAGE_1025            (0x00000002)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_OUTPUT_LUT_USAGE_NONE          (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_OUTPUT_LUT_USAGE_257           (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_USAGE_BOUNDS_OUTPUT_LUT_USAGE_1025          (0x00000002)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_USABLE_USE_LWRRENT          (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_USABLE_NO                   (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_USABLE_YES                  (0x00000002)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH_USE_LWRRENT     (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH_16              (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH_32              (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_PIXEL_DEPTH_64              (0x00000003)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_OVERLAY_LUT_USAGE_NONE      (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_OVERLAY_LUT_USAGE_257       (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OVERLAY_USAGE_BOUNDS_OVERLAY_LUT_USAGE_1025      (0x00000002)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_ENABLE_DISABLE                       (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_ENABLE_ENABLE                        (0x00000001)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_LORES                           (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_HIRES                           (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_INDEX_1025_UNITY_RANGE          (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_INTERPOLATE_1025_UNITY_RANGE    (0x00000003)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_INTERPOLATE_1025_XRBIAS_RANGE   (0x00000004)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_INTERPOLATE_1025_XVYCC_RANGE    (0x00000005)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_INTERPOLATE_257_UNITY_RANGE     (0x00000006)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_MODE_INTERPOLATE_257_LEGACY_RANGE    (0x00000007)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_NEVER_YIELD_TO_BASE_DISABLE          (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_BASE_LUT_LO_NEVER_YIELD_TO_BASE_ENABLE           (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_ENABLE_DISABLE                     (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_ENABLE_ENABLE                      (0x00000001)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_LORES                         (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_HIRES                         (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_INDEX_1025_UNITY_RANGE        (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_UNITY_RANGE  (0x00000003)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_XRBIAS_RANGE (0x00000004)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_INTERPOLATE_1025_XVYCC_RANGE  (0x00000005)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_INTERPOLATE_257_UNITY_RANGE   (0x00000006)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_MODE_INTERPOLATE_257_LEGACY_RANGE  (0x00000007)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_NEVER_YIELD_TO_BASE_DISABLE        (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_OUTPUT_LUT_LO_NEVER_YIELD_TO_BASE_ENABLE         (0x00000001)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PIXEL_REPLICATE_MODE_OFF                             (0x00000000)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PIXEL_REPLICATE_MODE_X2                              (0x00000001)
#define LW5070_CTRL_IS_MODE_POSSIBLE_PIXEL_REPLICATE_MODE_X4                              (0x00000002)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    struct {
        LwU32 HeadActive;
        struct {
            LwU32 Frequency;

            LwU32 Adj1000Div1001;
        } PixelClock;

        struct {
            LwU32 Width;
            LwU32 Height;
        } RasterSize;

        struct {
            LwU32 X;
            LwU32 Y;
        } RasterBlankStart;

        struct {
            LwU32 X;
            LwU32 Y;
        } RasterBlankEnd;

        struct {
            LwU32 YStart;
            LwU32 YEnd;
        } RasterVertBlank2;

        struct {
            LwU32             Structure;
/*
 * Note: For query calls, the lock modes and lock pins are used only if the min
 * pstate is required (i.e., if MinPState is set to
 * LW5070_CTRL_IS_MODE_POSSIBLE_NEED_MIN_PSTATE).
 */
            LW_DISP_LOCK_MODE MasterLockMode;
            LW_DISP_LOCK_PIN  MasterLockPin;
            LW_DISP_LOCK_MODE SlaveLockMode;
            LW_DISP_LOCK_PIN  SlaveLockPin;
        } Control;

        struct {
            LwU32  VerticalTaps;
            LwU32  HorizontalTaps;
            LwBool Force422;
        } OutputScaler;

        struct {
            LwU32 Width;
            LwU32 Height;
        } ViewportSizeOut;

        struct {
            LwU32 Width;
            LwU32 Height;
        } ViewportSizeOutMin;

        struct {
            LwU32 Width;
            LwU32 Height;
        } ViewportSizeOutMax;

        struct {
            LwU32 Width;
            LwU32 Height;
        } ViewportSizeIn;

        struct {
            LwU32 Format;
            LwU32 SuperSample;
        } Params;

        struct {
            LwU32 Usable;
            LwU32 PixelDepth;
            LwU32 SuperSample;
            LwU32 BaseLutUsage;
            LwU32 OutputLutUsage;
        } BaseUsageBounds;

        struct {
            LwU32 Usable;
            LwU32 PixelDepth;
            LwU32 OverlayLutUsage;
        } OverlayUsageBounds;

        struct {
            LwBool Enable;
            LwU32  Mode;
            LwBool NeverYieldToBase;
        } BaseLutLo;

        struct {
            LwBool Enable;
            LwU32  Mode;
            LwBool NeverYieldToBase;
        } OutputLutLo;

        LwU32 displayId[LW5070_CTRL_CMD_NUM_DISPLAY_ID_DWORDS_PER_HEAD];
        LwU32 outputResourcePixelDepthBPP;

        LwU32 CriticalWatermark; // in pixels

    } Head[LW5070_CTRL_CMD_MAX_HEADS];

    struct {
        LwU32 owner;
        LwU32 protocol;
    } Dac[LW5070_CTRL_CMD_MAX_DACS];

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    struct {
        LwU32 owner;
        LwU32 protocol;
    } Wbor[LW5070_CTRL_CMD_MAX_WBORS];
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



    struct {
//
// owner field is deprecated. In the future, all client calls should set
// ownerMask and bUseSorOwnerMask. bUseSorOwnerMask must be set in order
// to use ownerMask.
//
        LwU32 owner;
        LwU32 ownerMask; // Head mask owned this sor

        LwU32 protocol;
        LwU32 pixelReplicateMode;

        LwU8  numSSTLinks;
    } Sor[LW5070_CTRL_CMD_MAX_SORS];

    LwBool bUseSorOwnerMask;

    struct {
        LwU32 owner;
        LwU32 protocol;
    } Pior[LW5070_CTRL_CMD_MAX_PIORS];


    LwU32  RequestedOperation;
// This argument is for VERIF and INTERNAL use only
    LwU32  options;
    LwU32  IsPossible;
    LwU32  MinPState;

    LwU32  minPerfLevel;
//
// Below are the possible Output values for MinPState variable.
// Lower the p-state value higher the power consumption; if no p-states are defined on chip
// then it will return as zero.
//

//
// Below are the possible input values for MinPstate Variable, by default it callwlate
// mode is possible or not at max available p-state and return the same state in that variable.
//
    LwU32  worstCaseMargin;

//
// The callwlated margin is multiplied by a constant, so that it can be
// represented as an integer with reasonable precision.  "0x400" was chosen
// because it is a power of two, which might allow some compilers/CPUs to
// simplify the callwlation by doing a shift instead of a multiply/divide.
// (And 0x400 is 1024, which is close to 1000, so that may simplify visual
// interpretation of the raw margin value.)
//
    char   worstCaseDomain[8];

    LwBool bUseCachedPerfState;
} LW5070_CTRL_CMD_IS_MODE_POSSIBLE_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_CMD_QUERY_IS_MODE_POSSIBLE_LOG_ENTRY
 *
 * This structure describes a single mode ilwestigated by IMP. It includes
 * the relevant raster timings, pixel clocks, BASE/OVERLAY/CURSOR usage
 * bounds, DRAM timing parameters, LWCLK/MCLK frequencies and RXB/linebuffer
 * register values.
 */
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_IS_POSSIBLE_NO                      (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_IS_POSSIBLE_YES                     (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_HEAD_ACTIVE_YES                     (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_HEAD_ACTIVE_NO                      (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_PIXEL_CLOCK_ADJ1000DIV1001_YES      (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_PIXEL_CLOCK_ADJ1000DIV1001_NO       (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_CONTROL_STRUCTURE_PROGRESSIVE       (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_CONTROL_STRUCTURE_INTERLACED        (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_SWIZZLED_IH_SURFACE_ALLOCATION_YES  (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_SWIZZLED_IH_SURFACE_ALLOCATION_NO   (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_PESSIMISTIC_BASEOVLY_ALLOCATION_YES (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_PESSIMISTIC_BASEOVLY_ALLOCATION_NO  (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_DYNAMIC_MEMPOOL_ALLOCATION_YES      (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_DYNAMIC_MEMPOOL_ALLOCATION_NO       (0x00000000)

#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_MEMORY_LAYOUT_PITCH    (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_MEMORY_LAYOUT_BL       (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_PIXEL_DEPTH_8          (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_PIXEL_DEPTH_16         (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_PIXEL_DEPTH_32         (0x00000003)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_PIXEL_DEPTH_64         (0x00000004)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_SUPER_SAMPLE_X1AA      (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_SUPER_SAMPLE_X4AA      (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_IS_VALID_NO            (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_BASE_CHANNEL_IS_VALID_YES           (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_OVERLAY_CHANNEL_MEMORY_LAYOUT_PITCH (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_OVERLAY_CHANNEL_MEMORY_LAYOUT_BL    (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_OVERLAY_CHANNEL_PIXEL_DEPTH_16      (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_OVERLAY_CHANNEL_PIXEL_DEPTH_32      (0x00000002)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_OVERLAY_CHANNEL_IS_VALID_NO         (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_OVERLAY_CHANNEL_IS_VALID_YES        (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_LWRSOR_CHANNEL_MEMORY_LAYOUT_PITCH  (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_LWRSOR_CHANNEL_PIXEL_DEPTH_32       (0x00000001)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_LWRSOR_CHANNEL_IS_VALID_NO          (0x00000000)
#define LW5070_CTRL_CMD_IS_MODE_POSSIBLE_LOG_ENTRY_LWRSOR_CHANNEL_IS_VALID_YES         (0x00000001)
typedef struct LW5070_CTRL_CMD_QUERY_IS_MODE_POSSIBLE_LOG_ENTRY {
    LwU32 isPossible;
    struct {
        LwU32 isPossible;
        struct {
            LwU32 HeadActive;
            LwU32 DisplayId;
            struct {
                LwU32 Frequency;
                LwU32 Adj1000Div1001;
            } PixelClock;
            struct {
                LwU32 Width;
                LwU32 Height;
            } RasterSize;
            struct {
                LwU32 X;
                LwU32 Y;
            } RasterBlankStart;
            struct {
                LwU32 X;
                LwU32 Y;
            } RasterBlankEnd;
            struct {
                LwU32 YStart;
                LwU32 YEnd;
            } RasterVertBlank2;
            struct {
                LwU32 Structure;
            } Control;
            struct {
                LwU32 Width;
                LwU32 Height;
            } ViewportSizeOut;
            struct {
                LwU32 Width;
                LwU32 Height;
            } ViewportSizeOutMin;
            struct {
                LwU32 Width;
                LwU32 Height;
            } ViewportSizeIn;
        } Head[LW5070_CTRL_CMD_MAX_HEADS];
        struct {
            LwU32 owner;
            LwU32 protocol;
        } Dac[LW5070_CTRL_CMD_MAX_DACS];
        LwU32 reqdDispClk;
        LwU32 maxDispClk;
        LwU32 reqdDispClkRatio;
    } disp;
    struct {
        LwU32  isPossible;
        LW_DECLARE_ALIGNED(LwU64 memUtil, 8);
        LwU32  rxbCntlLoopUtil;
        LwU32  dramConlwrrency;
        LwU32  swizzledIHSurfaceAllocation;
        LwU32  pessimisticBaseOvlyAllocation;
        LwU32  dynamicMempoolAllocation;
        LwBool configStutterEnabled;

        struct {
            LwU32 defaultRxbCreditsPerPartition;
            LwU32 mempoolTotalBlocks;
            LwU32 mempoolMinBlocksPerHead;
            struct {
                LwU32 mempoolBlocks;
                LwU32 baseLbEntries;
                LwU32 ovlyLbEntries;
                LwU32 mempoolSpoolupThresh;
                LwU32 rxbCredits;
            } Head[LW5070_CTRL_CMD_MAX_HEADS];
        } regIn;
        struct {
            struct {
                LwU32 mempoolBlocks;
                LwU32 baseLbEntries;
                LwU32 ovlyLbEntries;
                LwU32 rxbCredits;
            } Head[LW5070_CTRL_CMD_MAX_HEADS];
        } regOut;
        struct {
            LwU32 BL;
            LwU32 CL;
            LwU32 tRCDRd;
            LwU32 tRCDWr;
            LwU32 tRP;
            LwU32 tR2P;
            LwU32 tW2P;
            LwU32 tR2W;
            LwU32 tW2R;
            LwU32 numPartitions;
            LwU32 numBanks;
            LwU32 numColBits;
        } dramCfg;
        struct {
            LwU32 lwclkFreqKHz;
            LwU32 mclkFreqKHz;
        } clkCfg;
        struct {
            LwU32 resX;
            LwU32 fetchRateKHz;
            LwS32 latency_margin_head_10x;

            struct {
                LwU32 memLayout;
                LwU32 pixelDepth;
                LwU32 superSample;
                LwU32 isValid;
            } Base;
            struct {
                LwU32 memLayout;
                LwU32 pixelDepth;
                LwU32 isValid;
            } Overlay;
            struct {
                LwU32 memLayout;
                LwU32 pixelDepth;
                LwU32 isValid;
            } Cursor;
        } Head[LW5070_CTRL_CMD_MAX_HEADS];
    } fb;
} LW5070_CTRL_CMD_QUERY_IS_MODE_POSSIBLE_LOG_ENTRY;

/*
 * LW5070_CTRL_CMD_QUERY_IS_MODE_POSSIBLE_LOG (deprecated on Fermi+)
 *
 * This command can be used by clients to retrieve information about modes
 * previously ilwestigated by IMP.  The data returned are intended for
 * post mortem analysis and can be used to generate input files for the
 * architecture version of fbIMP.
 *
 * This command has two modes:
 *
 * If the logEntryList pointer is NULL, this command returns the number
 * of entries in the IMP log via the 'logEntryCount' field.
 *
 * If the logEntryList pointer is non-NULL, this command returns the
 * first logEntryCount entries of the IMP log via the buffer referenced by
 * the logEntryList pointer.  The entries are removed from the IMP log.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW5070_CTRL_CMD_QUERY_IS_MODE_POSSIBLE_LOG (0x5070010a) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | 0xA" */  // Do Not Use!

typedef struct LW5070_CTRL_CMD_QUERY_IS_MODE_POSSIBLE_LOG_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       logEntryCount;
    LW_DECLARE_ALIGNED(LwP64 logEntryList, 8);
} LW5070_CTRL_CMD_QUERY_IS_MODE_POSSIBLE_LOG_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW5070_CTRL_CMD_GET_CHANNEL_INFO
 *
 * This command returns the current channel state.
 *
 *      channelClass
 *          This field indicates the hw class number (507A-507E)
 *
 *      channelInstance
 *          This field indicates which of the two instances of the channel
 *          (in case there are two. ex: base, overlay etc) the cmd is meant for.
 *          Note that core channel has only one instance and the field should
 *          be set to 0 for core channel.
 *
 *      channelState
 *          This field indicates the desired channel state in a mask form that
 *          is compatible with LW5070_CTRL_CMD_IDLE_CHANNEL. A mask format
 *          allows clients to check for one from a group of states.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 *
 * Display driver uses this call to ensure that all it's methods have
 * propagated through hardware's internal fifo
 * (LW5070_CTRL_GET_CHANNEL_INFO_STATE_NO_METHOD_PENDING) before it calls
 * RM to check whether or not the mode it set up in Assembly State Cache will
 * be possible. Note that display driver can not use completion notifier in
 * this case because completion notifier is associated with Update and Update
 * will propagate the state from Assembly to Armed and when checking the
 * possibility of a mode, display driver wouldn't want Armed state to be
 * affected.
 */
#define LW5070_CTRL_CMD_GET_CHANNEL_INFO (0x5070010b) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_CHANNEL_INFO_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_IDLE                 LW5070_CTRL_CMD_CHANNEL_STATE_IDLE
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_WRTIDLE              LW5070_CTRL_CMD_CHANNEL_STATE_WRTIDLE
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_EMPTY                LW5070_CTRL_CMD_CHANNEL_STATE_EMPTY
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_FLUSHED              LW5070_CTRL_CMD_CHANNEL_STATE_FLUSHED
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_BUSY                 LW5070_CTRL_CMD_CHANNEL_STATE_BUSY
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_DEALLOC              LW5070_CTRL_CMD_CHANNEL_STATE_DEALLOC
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_DEALLOC_LIMBO        LW5070_CTRL_CMD_CHANNEL_STATE_DEALLOC_LIMBO
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_LIMBO1               LW5070_CTRL_CMD_CHANNEL_STATE_LIMBO1
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_LIMBO2               LW5070_CTRL_CMD_CHANNEL_STATE_LIMBO2
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_FCODEINIT            LW5070_CTRL_CMD_CHANNEL_STATE_FCODEINIT
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_FCODE                LW5070_CTRL_CMD_CHANNEL_STATE_FCODE
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_VBIOSINIT            LW5070_CTRL_CMD_CHANNEL_STATE_VBIOSINIT
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_VBIOSOPER            LW5070_CTRL_CMD_CHANNEL_STATE_VBIOSOPER
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_UNCONNECTED          LW5070_CTRL_CMD_CHANNEL_STATE_UNCONNECTED
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_INITIALIZE           LW5070_CTRL_CMD_CHANNEL_STATE_INITIALIZE
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_SHUTDOWN1            LW5070_CTRL_CMD_CHANNEL_STATE_SHUTDOWN1
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_SHUTDOWN2            LW5070_CTRL_CMD_CHANNEL_STATE_SHUTDOWN2
#define LW5070_CTRL_GET_CHANNEL_INFO_STATE_NO_METHOD_PENDING    (LW5070_CTRL_GET_CHANNEL_INFO_STATE_EMPTY | LW5070_CTRL_GET_CHANNEL_INFO_STATE_WRTIDLE | LW5070_CTRL_GET_CHANNEL_INFO_STATE_IDLE)
#define LW5070_CTRL_CMD_GET_CHANNEL_INFO_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW5070_CTRL_CMD_GET_CHANNEL_INFO_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;
    LwBool                      IsChannelInDebugMode;

    LwU32                       channelState;
} LW5070_CTRL_CMD_GET_CHANNEL_INFO_PARAMS;



/*
 * LW5070_CTRL_CMD_SET_ACCL
 *
 *   This command turns accelerators on and off. The use of this command
 *   should be restricted as it may have undesirable effects. It's
 *   purpose is to provide a mechanism for clients to use the
 *   accelerator bits to get into states that are either not detectable
 *   by the RM or may take longer to reach than we think is reasonable
 *   to wait in the RM.
 *
 * LW5070_CTRL_CMD_GET_ACCL
 *
 *   This command queries the current state of the accelerators.
 *
 *      channelClass
 *          This field indicates the hw class number (507A-507E)
 *
 *      channelInstance
 *          This field indicates which of the two instances of the channel
 *          (in case there are two. ex: base, overlay etc) the cmd is meant for.
 *          Note that core channel has only one instance and the field should
 *          be set to 0 for core channel.
 *
 *      accelerators
 *          Accelerators to be set in the SET_ACCEL command. Returns the
 *          lwrrently set accelerators on the GET_ACCEL command.
 *
 *      accelMask
 *          A mask to specify which accelerators to change with the
 *          SET_ACCEL command. This field does nothing in the GET_ACCEL
 *          command.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_CHANNEL
 *      LW_ERR_ILWALID_OWNER
 *      LW_ERR_GENERIC
 *
 */

#define LW5070_CTRL_CMD_SET_ACCL (0x5070010c) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_SET_ACCL_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_ACCL (0x5070010d) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_GET_ACCL_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_ACCL_NONE    LW5070_CTRL_IDLE_CHANNEL_ACCL_NONE
#define LW5070_CTRL_ACCL_IGNORE_PI        LW5070_CTRL_IDLE_CHANNEL_ACCL_IGNORE_PI
#define LW5070_CTRL_ACCL_SKIP_NOTIF       LW5070_CTRL_IDLE_CHANNEL_ACCL_SKIP_NOTIF
#define LW5070_CTRL_ACCL_SKIP_SEMA        LW5070_CTRL_IDLE_CHANNEL_ACCL_SKIP_SEMA
#define LW5070_CTRL_ACCL_IGNORE_INTERLOCK LW5070_CTRL_IDLE_CHANNEL_ACCL_IGNORE_INTERLOCK
#define LW5070_CTRL_ACCL_IGNORE_FLIPLOCK  LW5070_CTRL_IDLE_CHANNEL_ACCL_IGNORE_FLIPLOCK
#define LW5070_CTRL_ACCL_TRASH_ONLY       LW5070_CTRL_IDLE_CHANNEL_ACCL_TRASH_ONLY
#define LW5070_CTRL_ACCL_TRASH_AND_ABORT  LW5070_CTRL_IDLE_CHANNEL_ACCL_TRASH_AND_ABORT
#define LW5070_CTRL_SET_ACCL_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW5070_CTRL_SET_ACCL_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;

    LwU32                       accelerators;
    LwU32                       accelMask;
} LW5070_CTRL_SET_ACCL_PARAMS;
#define LW5070_CTRL_GET_ACCL_PARAMS_MESSAGE_ID (0xDU)

typedef LW5070_CTRL_SET_ACCL_PARAMS LW5070_CTRL_GET_ACCL_PARAMS;

/*
 * LW5070_CTRL_CMD_STOP_BASE
 *
 * This command tries to turn the base channel off ASAP.
 *
 *      channelInstance
 *          This field indicates which of the two instances of the base
 *          channel the cmd is meant for.
 *
 *      notifyMode
 *          This field indicates the action RM should take once the base
 *          channel has been successfully stopped. The options are (1) Set a
 *          notifier (2) Set the notifier and generate and OS event
 *
 *      hNotifierCtxDma
 *          Handle to the ctx dma for the notifier that must be written once
 *          base channel is stopped. The standard LwNotification notifier
 *          structure is used.
 *
 *      offset
 *          Offset within the notifier context dma where the notifier begins
 *          Offset must be 16 byte aligned.
 *
 *      hEvent
 *          Handle to the event that RM must use to awaken the client when
 *          notifyMode is WRITE_AWAKEN.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT: Invalid notify mode
 *      LW_ERR_ILWALID_CHANNEL: When the overlay is unallocated
 *      LW_ERR_ILWALID_OWNER: Callee isn't the owner of the channel
 *      LW_ERR_ILWALID_OBJECT_HANDLE: Notif ctx dma not found
 *      LW_ERR_ILWALID_OFFSET: Bad offset within notif ctx dma
 *      LW_ERR_INSUFFICIENT_RESOURCES
 *      LW_ERR_TIMEOUT: RM timedout waiting to inject methods
 */
#define LW5070_CTRL_CMD_STOP_BASE                          (0x5070010e) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_STOP_BASE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_STOP_BASE_NOTIFY_MODE_WRITE        (0x00000000)
#define LW5070_CTRL_CMD_STOP_BASE_NOTIFY_MODE_WRITE_AWAKEN (0x00000001)

#define LW5070_CTRL_CMD_STOP_BASE_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW5070_CTRL_CMD_STOP_BASE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelInstance;
    LwU32                       notifyMode;
    LwHandle                    hNotifierCtxDma;
    LwU32                       offset;
    LW_DECLARE_ALIGNED(LwP64 hEvent, 8);
} LW5070_CTRL_CMD_STOP_BASE_PARAMS;


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

typedef struct LW5070_CTRL_STUTTER_FEATURES_OUTPUT {
    LwBool bIsValid;
    // IMP should never set BOTH asr.bIsPossible and mscg.bIsPossible.
    struct {
        LwBool bIsPossible;
        struct {
            LwU32 lowWatermark;
            LwU32 highWatermark;
            LwU32 submode;
            LwU32 mode;
        } head[LW5070_CTRL_CMD_MAX_HEADS];
        LwU32 efficiency;
    } asr;
    struct {
        LwBool bIsPossible;
        struct {
            //
            // The MSCG low watermark represents the lowest mempool oclwpancy
            // level at which pixel data must be requested in order to avoid
            // under flow.  LWM will be zero for inactive heads.
            //
            LwU32 lowWatermark;
            //
            // The MSCG high watermark represents the lowest mempool oclwpancy
            // at which we still expect to be able to save power by entering
            // MSCG. We don't actually enter MSCG at that level; normally, we
            // enter MSCG eight after filling mempool and switching to drain
            // mode, but if MSCG entry is blocked for some reason, we can
            // still enter later if it becomes unblocked before mempool drains
            // below the high watermark level.
            //
            LwU32 highWatermark;
            //
            // The MSCG wake up watermark represents the lowest mempool
            // mempool oclwpancy level at which the memory subsystem must
            // begin waking up in order to be fully awake by the time
            // we reach the low watermark.
            //
            LwU32 wakeupWatermark;
            //
            // The mode field is no longer used.  The HW mode field is
            // programmed based on the LWM setting.
            //
            LwU32 mode;
        } head[LW5070_CTRL_CMD_MAX_HEADS];
        LwU32 efficiency;
    } mscg;
} LW5070_CTRL_STUTTER_FEATURES_OUTPUT;
typedef struct LW5070_CTRL_STUTTER_FEATURES_OUTPUT *PLW5070_CTRL_STUTTER_FEATURES_OUTPUT;

typedef struct LW5070_CTRL_ASR_INPUT {
    LwBool isValid;                 // Variable for debugging purposes.
    LwU32  bytesPerMempoolEntry;    // Watermarks are is 32B units(for now)
    LwU32  ASREfficiencyThreshold;

    // LW_PFB_TIMING10_ASR2ASREX (m2clocks)
    LwU32  tXSR; // DRAM timing parameter tXSR ( Exit Self Refresh on a Read )

    // LW_PFB_TIMING10_ASR2NRD (m2clocks)
    LwU32  tXSNR;

    // LW_PFB_DRAM_ASR_ASRPD == _POWERDOWN
    LwBool powerdown;
} LW5070_CTRL_ASR_INPUT;
typedef struct LW5070_CTRL_ASR_INPUT *PLW5070_CTRL_ASR_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_HEAD_INPUT {
    LwBool HeadActive;

    struct {
        LwU32  Frequency; // pixel clk freq in KHz
        LwBool UseAdj1000Div1001;
    } PixelClock;

    struct {
        LwU32 Width;
        LwU32 Height;
    } RasterSize;

    struct {
        LwU32 X;
        LwU32 Y;
    } RasterBlankStart;

    struct {
        LwU32 X;
        LwU32 Y;
    } RasterBlankEnd;

    struct {
        LwU32 YStart;
        LwU32 YEnd;
    } RasterVertBlank2;

    struct {
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_RASTER_STRUCTURE Structure;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW_DISP_LOCK_MODE   MasterLockMode;
        LW_DISP_LOCK_PIN    MasterLockPin;
        LW_DISP_LOCK_MODE   SlaveLockMode;
        LW_DISP_LOCK_PIN    SlaveLockPin;
    } Control;

    struct {
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_VERTICAL_TAPS   VerticalTaps;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_HORIZONTAL_TAPS HorizontalTaps;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LwBool             Force422;
    } OutputScaler;

    struct {
        LwU32 Width;
        LwU32 Height;
    } ViewportSizeOut;

    struct {
        LwU32 Width;
        LwU32 Height;
    } ViewportSizeOutMin;

    struct {
        LwU32 Width;
        LwU32 Height;
    } ViewportSizeOutMax;

    struct {
        LwU32 Width;
        LwU32 Height;
    } ViewportSizeIn;

    struct {
        LwU32           CoreBytesPerPixel;
        LW_SUPER_SAMPLE SuperSample;
    } Params;

    struct {
        LwBool          Usable;
        LwBool          DistRenderUsable;
        LwU32           BytesPerPixel;
        LW_SUPER_SAMPLE SuperSample;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_LUT_USAGE    BaseLutUsage;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_LUT_USAGE    OutputLutUsage;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
    } BaseUsageBounds;

    struct {
        LwBool       Usable;
        LwU32        BytesPerPixel;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_LUT_USAGE OverlayLutUsage;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
    } OverlayUsageBounds;

    struct {
        LwBool         Enable;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_LUT_LO_MODE Mode;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LwBool         NeverYieldToBase;
    } BaseLutLo;

    struct {
        LwBool         Enable;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        LW_LUT_LO_MODE Mode;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LwBool         NeverYieldToBase;
    } OutputLutLo;

    LwU32                              displayId[LW5070_CTRL_CMD_NUM_DISPLAY_ID_DWORDS_PER_HEAD];

    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP outputResourcePixelDepthBPP;
} LW5070_CTRL_IMP_DISP_HEAD_INPUT;
typedef struct LW5070_CTRL_IMP_DISP_HEAD_INPUT *PLW5070_CTRL_IMP_DISP_HEAD_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_DAC_INPUT {
    LW_OR_OWNER     owner;
    LW_DAC_PROTOCOL protocol;
} LW5070_CTRL_IMP_DISP_DAC_INPUT;
typedef struct LW5070_CTRL_IMP_DISP_DAC_INPUT *PLW5070_CTRL_IMP_DISP_DAC_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_SOR_INPUT {
    LW_OR_OWNER             owner;
    LwU32                   ownerMask; // Head mask owned this sor

    LW_SOR_PROTOCOL         protocol;
    LW_PIXEL_REPLICATE_MODE pixelReplicateMode;
    LwU8                    numSSTLinks;
} LW5070_CTRL_IMP_DISP_SOR_INPUT;
typedef struct LW5070_CTRL_IMP_DISP_SOR_INPUT *PLW5070_CTRL_IMP_DISP_SOR_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_PIOR_INPUT {
    LW_OR_OWNER      owner;
    LW_PIOR_PROTOCOL protocol;
} LW5070_CTRL_IMP_DISP_PIOR_INPUT;
typedef struct LW5070_CTRL_IMP_DISP_PIOR_INPUT *PLW5070_CTRL_IMP_DISP_PIOR_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_WBOR_INPUT {
    LW_OR_OWNER      owner;
    LW_WBOR_PROTOCOL protocol;
} LW5070_CTRL_IMP_DISP_WBOR_INPUT;
typedef struct LW5070_CTRL_IMP_DISP_WBOR_INPUT *PLW5070_CTRL_IMP_DISP_WBOR_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_HW_SPECIFIC_INPUT {
    // This struct is used to make DispIsModePossible()
    struct {
        LwU32 MaxPixels5Tap444[LW5070_CTRL_CMD_MAX_HEADS];                      // independent of h/w.
        LwU32 MaxPixels3Tap444[LW5070_CTRL_CMD_MAX_HEADS];                      // This structure is same as in DISPHALINFO
        LwU32 MaxPixels2Tap444[LW5070_CTRL_CMD_MAX_HEADS];                      // but is populated manually for the test
    } Cap;

    LwU32 pipe2RgFifoDepth;
} LW5070_CTRL_IMP_DISP_HW_SPECIFIC_INPUT;
typedef struct LW5070_CTRL_IMP_DISP_HW_SPECIFIC_INPUT *PLW5070_CTRL_IMP_DISP_HW_SPECIFIC_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_DISP_INPUT {
    LW5070_CTRL_IMP_DISP_HEAD_INPUT        impDispHeadIn[LW5070_CTRL_CMD_MAX_HEADS];
    LW5070_CTRL_IMP_DISP_DAC_INPUT         impDacIn[LW5070_CTRL_CMD_MAX_DACS];
    LW5070_CTRL_IMP_DISP_SOR_INPUT         impSorIn[LW5070_CTRL_CMD_MAX_SORS];
    LW5070_CTRL_IMP_DISP_PIOR_INPUT        impPiorIn[LW5070_CTRL_CMD_MAX_PIORS];
    LW5070_CTRL_IMP_DISP_WBOR_INPUT        impWborIn[LW5070_CTRL_CMD_MAX_WBORS];
    LW5070_CTRL_IMP_DISP_HW_SPECIFIC_INPUT impHWspecificDispIn;
    LwBool                                 bUseSorOwnerMask;
} LW5070_CTRL_IMP_DISP_DISP_INPUT;
typedef struct LW5070_CTRL_IMP_DISP_DISP_INPUT *PLW5070_CTRL_IMP_DISP_DISP_INPUT;

typedef struct LW5070_CTRL_IMP_DISP_DISP_OUTPUT {
    LwBool isPossible;        // Can display handle the bandwidth required by this mode?
    LwU32  impFailReason;

    LwU32  reqdDispClk;

    // maxDispClk is the actual dispClk frequency, in KHz.
    LwU32  maxDispClk;

    LwU32  reqdDispClkRatio;

    LwU32  fbFetchRateKHz[LW5070_CTRL_CMD_MAX_HEADS];

    LwU32  meter[LW5070_CTRL_CMD_MAX_HEADS];

    //
    // dispClkActualKHz is the effective, or usable, dispClk frequency, after
    // applying fetch metering.
    //
    LwU32  dispClkActualKHz[LW5070_CTRL_CMD_MAX_HEADS];

    // minFillRateKHz is the required pipe to rg fifo fill rate.
    LwU32  minFillRateKHz[LW5070_CTRL_CMD_MAX_HEADS];

    LwU32  avgFetchRateKHz[LW5070_CTRL_CMD_MAX_HEADS];
} LW5070_CTRL_IMP_DISP_DISP_OUTPUT;
typedef struct LW5070_CTRL_IMP_DISP_DISP_OUTPUT *PLW5070_CTRL_IMP_DISP_DISP_OUTPUT;


#define LW5070_CTRL_IMP_BITS_PER_BYTE                                 8
#define LW5070_CTRL_IMP_BYTES_PER_GOB                                 256
#define LW5070_CTRL_IMP_BYTES_PER_SECTOR                              32
#define LW5070_CTRL_IMP_BYTES_PER_MEGABYTE                            (0x100000) /* finn: Evaluated from "(1024 * 1024)" */

// block-linear memory fetches 2x the amount of memory when linebuffer
// can't be used
#define LW5070_CTRL_IMP_BL_OVERFETCH_FOR_MISSING_LINEBUFFER           2

// filter-on-scanout filters four pixels down to one
#define LW5070_CTRL_IMP_FOS_FILTERING_AMOUNT                          4

//
// FOS is done in the partition.  So for 4xAA, 4Bpp, a block of 64 pixels
// constitutes 1024B being read from the DRAMs, which is filtered down to
// 256B.  Now that data returned to ISO across RXB is doubled; it is sent
// back in 16B chunks and since the interface is 32B, half of the data is
// 0.  Hence, we return 2x the amount of data actually used by ISO.  Here's
// (hopefully) a better explanation from Michael Sonnenberg:
//     I'm new to ISO, but my understanding is that FOS is done in the
//     partition.  So for 4xAA, 4Bpp, a block of 64 pixels constitutes
//     1024B being read from the DRAMs, which is filtered down to 256B.
//     Now that data returned to ISO across RXB is doubled; it is sent back
//     in 16B chunks and since the interface is 32B, half of the data is 0.
//     So I think this is why we see 512B sent through RXB, rather than
//     256B.
//
// Ok, even though we get two 16B returns, there is only one RXB credit
// used.  From MikeW:
//
//     ------------------------------------------
//     From: Michael Woodmansee
//     Sent:Friday, August 08, 2008 2:35 PM
//     Subject:RE: Quick ISO question.
//
//     [PraveenK] In FOS, if ISO request was for 128B, we will return only 32B
//     [PraveenK] thro RXB. I suppose we return it as 2 16B transfers (with
//     [PraveenK] half of the 32B interface as 0s), right? If so do we still
//     [PraveenK] consume only 1 RXB credit for this ISO request?
//
//     It is sent as a pair of 16B xfers to RXB, takes a single 32B RXB credit
//     and is returned to iso as a single 32B xfer.


//
// Mode is possible if we use 95% or less of available memory bandwidth
// in the worst case, and less than 1/2 in the best case; the default
// threshold can be overridden via the 'RMIMPMaxFbUtilization' RM registry
// key.
//
#define LW5070_CTRL_IMP_MAX_MODE_POSSIBLE_MEM_UTILIZATION             95

// Number of pixels wide for the cursor
#define LW5070_CTRL_IMP_LWRSOR_WIDTH                                  64
#define LW5070_CTRL_IMP_LWRSOR_HEIGHT                                 64

// GF11X LUT definitions.
#define LW5070_CTRL_IMP_NUM_LUTS                                      3
#define LW5070_CTRL_IMP_LUT_WIDTH                                     (0x802) /* finn: Evaluated from "(1025 * 2)" */

// Each LUT is eight bytes, enough to store 16-bit red, green, and blue values.
#define LW5070_CTRL_IMP_LUT_ENTRY_SIZE (LUT_PIXEL_DEPTH * 2)
#define LW5070_CTRL_IMP_LUT_SIZE_0     (0 * LW5070_CTRL_IMP_LUT_ENTRY_SIZE)
#define LW5070_CTRL_IMP_LUT_SIZE_257   (257 * LW5070_CTRL_IMP_LUT_ENTRY_SIZE)
#define LW5070_CTRL_IMP_LUT_SIZE_1025  (1025 * LW5070_CTRL_IMP_LUT_ENTRY_SIZE)

// Time spent int the RXB control loop latency - the actual round trip is
// split between lwclock and mclock domain, so we dynamically callwlate the
// frequency based on the current lw/m2 clock frequencies
// 8/6/2008 JPR: Updated the MCLOCK part from 39 to 49 based on the minimum
// measured latency in bug 413094.  In the bug we measured between 97 and
// 111.  The difference is assumed to be the difference in Pre/Act timing
// which IMP already takes into account.

#define LW5070_CTRL_IMP_RXB_CNTRL_LOOP_POST_DRAM_LWCLOCK_PART         5
#define LW5070_CTRL_IMP_RXB_CNTRL_LOOP_POST_DRAM_MCLOCK_PART_ORIG     39
#define LW5070_CTRL_IMP_RXB_CNTRL_LOOP_POST_DRAM_MCLOCK_PART_MEASURED 49

// Percentage of the RXB latency loop that, if we can't cover, inlwrs the
// aggressor traffic overhead.
#define LW5070_CTRL_IMP_RXB_CONTROL_LOOP_LATENCY_TOLERANCE            95

// Default number of total RXB credits split between the two heads.  Each
// enabled head needs a minimum of 8 credits, and credits should be
// allocated on an 8-credit basis as to cover the largest ISO request size
// of 256B == 32B * 8
#define LW5070_CTRL_IMP_DEFAULT_NUMBER_RXB_CREDITS_PER_PARTITION      32
#define LW5070_CTRL_IMP_RXB_CREDIT_ALLOCATION_QUANTA                  8

// Size, in bytes, of a single memory pool block.  Memory pool blocks
// should be allocated 8 at a time to handle the 256B pitch requests w/o
// fragmentation.
#define LW5070_CTRL_IMP_MEMORY_POOL_BLOCK_SIZE_BYTES                  32
#define LW5070_CTRL_IMP_MEMORY_POOL_ALLOCATION_QUANTA                 8

//
// Number of reserved memory pool entries that cannot be used for
// linebuffering.  From the '8' in the equation in the manuals describing
//ISO_LINE_BUFFER_HLIMIT0/1 registers:
// (LINE_BUFFER_HLIMIT0/1_BASE + LINE_BUFFER_HLIMIT0/1_OVLY) < (POOLCFG_LIMIT0/1 - Head0/1.SPOOLUP_THRESH - 8)
//
#define LW5070_CTRL_IMP_RESERVED_LWRSOR_MEMPOOL_ENTRIES               8

// All DRAM timing parameters should be less than this value - used as a
// sanity check
#define LW5070_CTRL_IMP_MAX_M2CLOCKS_FOR_DRAM_PARAM                   100

//
// Number of channels in partition
//
#define SINGLE_CHN_PER_PARTITION                                      1
#define DUAL_CHN_PER_PARTITION                                        2

//
// Number of Pixels per LWClk
//
#define ONE_PIXEL_PER_LWCLK                                           1
#define TWO_PIXEL_PER_LWCLK                                           2

typedef enum LW5070_CTRL_IMP_FBDRAM_CONLWR_STREAM {
    LW5070_CTRL_IMP_FBDRAM_CONLWR_STREAM_BASE = 0, // primary or secondary base layer
    LW5070_CTRL_IMP_FBDRAM_CONLWR_STREAM_OVLY = 1,
    LW5070_CTRL_IMP_FBDRAM_CONLWR_STREAM_NUM_STREAMS = 2,
} LW5070_CTRL_IMP_FBDRAM_CONLWR_STREAM;

// Keeping around the unused old structure to satisfy cross builds.
typedef enum LW5070_CTRL_IMP_BANDWIDTH_GOAL {
    LW5070_CTRL_BANDWIDTH_GOAL_MAX = 0,
    LW5070_CTRL_BANDWIDTH_GOAL_OPTIMAL = 1,
} LW5070_CTRL_IMP_BANDWIDTH_GOAL;

//
// The Bw and latency table on MCP systems need to be handled differently based on the
// context in which the IMP call is made. So, introduce this enum to indicate the
// necessary action
//
typedef enum LW5070_CTRL_IMP_BW_GOAL {
    LW5070_CTRL_IMP_BW_GOAL_ILWALID = 0,
    LW5070_CTRL_IMP_BW_GOAL_MAX = 1,
    LW5070_CTRL_IMP_BW_GOAL_OPTIMAL = 2,
    LW5070_CTRL_IMP_BW_GOAL_USE_ACTIVE = 3,
    LW5070_CTRL_IMP_BW_GOAL_USE_CACHED = 4,
} LW5070_CTRL_IMP_BW_GOAL;

// Maximum number of DRAM banks we support.
#define LW5070_CTRL_IMP_MAX_NUM_DRAM_BANKS 16

typedef struct LW5070_CTRL_IMP_FB_REG_INPUT {
   // Total number of blocks in the memory pool
   // LW_PFB_ISO_MEMPOOL_TOT_BLKS
    LwU32 mempoolTotalBlocks;

   // Minimum number of blocks per head
   // LW_PFB_ISO_MEMPOOL_MIN_BLKS_PER_HEAD
    LwU32 mempoolMinBlocksPerHead;

    // Number of blocks allocated to each head
    // LW_PFB_ISO_POOLCFG_LIMIT0/LW_PFB_ISO_POOLCFG_LIMIT1
    LwU32 mempoolBlocks[LW5070_CTRL_CMD_MAX_HEADS];

    // Number of entries in the memory pool used for line
    // buffering for the BASE channel on each head.
    // LW_PFB_ISO_LINE_BUFFER_HLIMIT0_BASE/LW_PFB_ISO_LINE_BUFFER_HLIMIT1_BASE
    LwU32 baseLbEntries[LW5070_CTRL_CMD_MAX_HEADS];

    // Number of entries in the memory pool used for line
    // buffering for the OVERALY channel on each head.
    // LW_PFB_ISO_LINE_BUFFER_HLIMIT0_OVLY/LW_PFB_ISO_LINE_BUFFER_HLIMIT1_OVLY
    LwU32 ovlyLbEntries[LW5070_CTRL_CMD_MAX_HEADS];

    // Spoolup thresholds for each head
    // LW_PFB_ISO_CFGHEAD_SPOOLUP_THRESH0/LW_PFB_ISO_CFGHEAD_SPOOLUP_THRESH1
    LwU32 mempoolSpoolupThresh[LW5070_CTRL_CMD_MAX_HEADS];

    // RXB credits for each head
    // LW_PFB_RXB_RDATFIFO0_SLOTS_ISO_D0/LW_PFB_RXB_RDATFIFO0_SLOTS_ISO_D1
    LwU32 rxbCredits[LW5070_CTRL_CMD_MAX_HEADS];

    // NOT a real register, but use this to select a different value for
    // the default number of RXB credits.  Setting this to '0' will use the
    // chip-defaults
    LwU32 RxbCreditsPerPartition;

    // This is the time, in 10ths of usecs, that hardware has to fully
    // spool-up.  This is the time between a contract being issued and
    // the first pixel being fetched.  If set to 0, the spoolup check
    // is bypassed.
    LwU32 available_spoolup_time[LW5070_CTRL_CMD_MAX_HEADS];

    // The shift-amount when callwlating the ISO training and/or watermark
    // thresholds.
    // LW_PFB_ISO_VFIFO_PIXEL_SHIFT
    LwU32 vfifo_pixel_shift;

    //
    // number of requests allocated per rotation of the WRR arbiter for
    // TC0/TC1 clients when display is in non-critical state.
    // LW_PFB_FBBA_IQ_ARB5_WEIGHT_LAST_TC(i)_NON_CRIT
    // Used only in MCP89 for internal latency computations
    //
    LwU32 tc0NonCritArbiterWeight;
    LwU32 tc1NonCritArbiterWeight;
} LW5070_CTRL_IMP_FB_REG_INPUT;
typedef struct LW5070_CTRL_IMP_FB_REG_INPUT *PLW5070_CTRL_IMP_FB_REG_INPUT;

// data transport bus to/from FB
typedef enum imp_tbus_type {
    LW5070_CTRL_IMP_TBUS_TYPE_NONE = 0,    // typical choice for local FB
    LW5070_CTRL_IMP_TBUS_TYPE_HT1 = 1,        // HyperTransport, version 1.x
    LW5070_CTRL_IMP_TBUS_TYPE_HT3 = 2,        // HyperTransport, version 3.x
    LW5070_CTRL_IMP_TBUS_TYPE_COUNT = 3,
} imp_tbus_type;

// memory controller type
typedef enum imp_mc_type {
    LW5070_CTRL_IMP_MC_TESLA_FB = 0,   // dGPU, Tesla family Lwpu FB
    LW5070_CTRL_IMP_MC_AMDK8_NB = 1,   // iGPU, AMD Hammer CPUs
    LW5070_CTRL_IMP_MC_AMDK10_NB = 2,  // iGPU, AMD Griffin and Greyhound CPUs
    LW5070_CTRL_IMP_MC_INTEL_NB = 3,   // iGPU, Intel Havendale and Lynnfield
    LW5070_CTRL_IMP_MC_TYPE_COUNT = 4,
} imp_mc_type;

// DDR parameters
typedef struct LW5070_CTRL_IMP_FB_DRAM_CFG_INPUT {
    // (LW_PFB_CFG0_BURST_LENGTH == 0) ? 4 : 8;
    LwU32         BL;      // DDR Burst Length

    // LW_PFB_TIMING3_QUSE
    LwU32         CL;      // Cas Latency

    // LW_PFB_TIMING2_RD_RCD
    LwU32         tRCDRd;  // act(n)  => read(n)

    // LW_PFB_TIMING2_WR_RCD
    LwU32         tRCDWr;  // act(n)  => write(n)

    // LW_PFB_TIMING0_RP
    LwU32         tRP;     // pre(n)  => act(n)

    // LW_PFB_TIMING1_R2P == 1
    LwU32         tR2P;    // read(n) => pre(n)

    // LW_PFB_TIMING1_W2P == ((WL+1) + tWR)
    LwU32         tW2P;    // write(n) => pre(n)

    // LW_PFB_TIMING1_R2W == (CL + BL/2) - WL + tDBT
    LwU32         tR2W;    // read(x)  => write(y), all values x,y

    // LW_PFB_TIMING1_W2R ==  WL + BL/2 + tCDLR;
    LwU32         tW2R;    // write(x) =>  read(y), all values x,y

    LwU32         bytesPerM2clock;   // number of bytes we return per m2clock

    // LW_PBUS_FS_FB_ENABLE
    LwU32         numPartitions;     // number of partitions

    // (LW_PFB_CFG1_BANK == 0) ? 4 : 8;
    LwU32         numBanks;          // number of banks of memory

    // LW_PFB_CFG1_COL
    LwU32         numColBits;        // number of column bits

    //
    LwU32         numCommandCyclesPerM2clock;

    // Dram type is now needs to use as IMP struct (gt21x onward)
    LwU32         dRamType;

    // which GPU we're using
    //enum imp_gpu_family gpu_family;

    // number of memory partitions (1 - 8)
    //LwU32 num_partitions;

    // lwclock frequency in MHz
    //LwU32 lwclock_freq_mhz;

    // m2clock frequency in MHz
    //LwU32 m2clock_freq_mhz;

    // The number of extra lines buffered in block-linear modes.
    // LW_PFB_ISO_LINE_BUFFER_CNT
    LwU32         lines_in_linebuffer;

    // Misc config registers - should be zero for default operation

    // When non-zero, we use an "average" latency for worst-case aggressor
    // traffic instead of the absolute worst case
    LwU32         config_use_relaxed_aggressor_latencies;

    // When non-zero, we force memory pool allocations to be split between
    // evenly two heads, regardless of whether both heads are enabled.
    LwBool        config_use_static_mempool_allocation;

    // Indicates if display compaction is available
    LwU32         configCompactionAvailable;

    LwBool        configStutterEnabled;
    //
    // Compaction surface size
    // If not 0, this specifies the size of the compaction surface that was statically
    // allocated by RM. When 0, the algorithm assumes that we have an infinite compaction surface size.
    // The amount ofcompaction surface that is actually used will be reported in
    // imp_iso_outregs.compact_surface_used_size
    //
    LwU32         compactionSurfaceSize;
    LwU32         compactionMaxSlices;

    // Normally, when config_use_static_mempool_allocation is zero, the
    // mempool is divided between the two heads proportional to their
    // bandwidth requirements.  However, IMP latency tolerance callwlations
    // assume that, in the worst-case, (resX-1)*lines_in_linebuffer pixels
    // are unable to contribute to latency tolerance.
    // When config_allocate_mempool_based_on_latency is non-zero, these
    // "hidden" pixels are considered in mempool allocation callwlations
    // so that latency tolerance of each head is better matched.
    LwU32         config_allocate_mempool_based_on_latency;

    //
    // When non-zero, enables codepaths that limit mempool and linebuffer
    // allocations to avoid potential hang conditions (bug 399597).
    //
    LwU32         config_spoolup_thresh_hang_prevention;

    //
    // When non-zero, a different algorithm is used for linebuffer allocations
    // that avoids corner cases where there can be too much linebuffering.
    //
    LwU32         config_use_alternate_lb_allocation_rule;

    // When non-zero, we use pessimistic base/overlay allocation
    // assumptions.  The default assumption is that the base/overlay allocations within a
    // single head are exactly (num_bank/2) accesses apart [1]
    LwBool        config_use_pessimistic_base_overlay_alloc;

    // Use non-swizzled bank accesses between heads.  Optimal is to swizzle
    // banks
    LwBool        config_use_nonswizzled_interhead_surface_alloc;

    // When non-zero, override the callwlated dram conlwrrency with this value.
    LwU32         config_forced_dram_conlwrrency;

    // When non-zero, ignore any RXB overhead (assumes more credits)
    LwU32         config_ignore_rxb_thresholds;

    // boolean - should we use the new IMP callwlations for RXB latency?
    // See bug 433314 for details.
    LwBool        config_use_updated_rxb_latency_calcs;

    //boolean - when non-zero we will be in a gear-shift mode.
    //this is related to gt21x where we run the rxb path @ twice the rate
    //to reduce the latency in the path between the TOL and the memory controller
    LwU32         config_use_gear_shifters;

    //
    // boolean - Can we boost the maximum utilization threshold up to 100% if we
    // are more than 50% linebuffered in all surfaces?  This will get us back
    // some highres dualview modes we had previously allowed (and it *looks*
    // like we can support w/o underflow) but now don't allow with partial
    // linebuffering handling.
    // Bug 523185 and 517220 for details.
    //
    LwU32         config_mem_utilization_100pcnt_if_50pcnt_linebuffered;

    // which transport bus we're using to connect to FB
    imp_tbus_type tbus_type;

    // which memory controller we're using
    imp_mc_type   mcType;

    // Transport bus data clock frequency (including DDR multiple if
    // applicable).  Ignored if tbus_type==LW5070_CTRL_IMP_TBUS_TYPE_NONE.
    LwU32         tbus_freq_mhz;

    // Transport data bus bit width (downstream direction).  Ignored if
    // tbus_type==LW5070_CTRL_IMP_TBUS_TYPE_NONE.
    LwU32         tbus_bit_width;

    // These inputs should come from ACPI call
    LwU32         wc_latency; // 10ths of usecs, either base_latency or total_latency from ACPI call
    LwU32         total_bw; // MB/s
    LwU32         link_bw; // MB/s. Used only on MCP7x. Unused on MCP89
    LwU32         aggressive_pm_latency; // 10ths of usecs

    LwU32         bwTableIndex;

    // Unused variable, but keeping around to satisfy cross builds.
    LwU32         targetBandwidth;

    // LW_PFB_CHANNELCFG_CHANNEL_ENABLE_32_BIT ? 1 : 2
    LwU32         numChnPerPartition;

    // Number of pixels transferred per lwclk
    LwU32         numPixelTransPerLwClk;

    // Rate at which the RXB can return data
    LwU32         RxbBytesPerLwclk;

    //
    // Specify a maximum portion of memory bandwidth ISO can use.
    // This value exist to overwrite pFbHalPvtInfo->impMaxFbUtilizationOverride
    //
    LwU32         impMaxFbUtilization;


    //
    // altVidOn tells whether or not IMP callwlations will be done under the
    // assumption that AltVid will be enabled.  We prefer to run with AltVid
    // enabled, because power consumption is reduced, but this increases memory
    // latency, so some modes can work only if AltVid is disabled.
    //
    LwBool        altVidOn;

    //
    // altVidLatencyDelta specifies how much latency will increase (relative
    // to the current or most recent IMP mode) if AltVid is turned on.
    //
    LwS32         altVidLatencyDelta;

    //
    // dllOn specifies whether the DLL is on or off.  At lower memory
    // frequencies (lower P-States), the DLLs can be left off during normal
    // operation (not just in ASR).  If the DLLs are off, ASR wakeup can be
    // much faster, because it is not necessary to wait for DLL lock.
    //
    LwBool        dllOn;

    // Start ISO registers

    // use_user_specified_work_limits - when non-zero we will use the work limits specified by
    // bankq_work_limit, tol_max_quanta, num_instantaneous_banks, and
    // rmw_limit; otherwise we use the IMP defaults
    LwU32         use_user_specified_work_limits;

    // BANKQ_WORK_LIMIT - only applicable for tesla2 architectures
    // is used to limit the number of mclks worth of work that can be
    // outstanding in each TOL bankq threadbuf.
    LwU32         bankq_work_limit;

    // TOL_QUANTA (max of TOL_QUANTA_1_BANK, TOL_QUANTA_2_BANK,
    // TOL_QUANTA_3_BANK, and TOL_QUANTA_GT3_BANK; almost always should be
    // TOL_QUANTA_1_BANK - max number of streamed requests to a single bank
    // before forcing a row/bank change.
    LwU32         tol_max_quanta;

    // NUM_INSTANTANEOUS_BANKS - only applicable for tesla2 architectures
    // Maximum number of threadbufs that can have pending work in GT200
    // (LW_PFB_TOL_CTRL1_NUM_INSTANTANEOUS_BANKS + 1)
    LwU32         num_instantaneous_banks;

    // LW_PFB_TOL_CTRL2_RMW_LIMIT
    // Maximum number of RMWs that can be active in gt200
    LwU32         rmw_limit;
} LW5070_CTRL_IMP_FB_DRAM_CFG_INPUT;
typedef struct LW5070_CTRL_IMP_FB_DRAM_CFG_INPUT *PLW5070_CTRL_IMP_FB_DRAM_CFG_INPUT;

// GPU configuration
typedef struct LW5070_CTRL_IMP_FB_CLK_INPUT {
    LwU32 PerfLevel;         // The PerfLevel to run IMP at or PERFCTL_LEVELS for "current clocks"
    LwU32 lwclkFreqKHz;      // lwclk frequency in KHz corresponding to above perfLevel
    LwU32 mclkFreqKHz;       // mclk frequency in KHz corresponding to the above perfLevel
    LwU32 hostclkFreqKHz;    // hostclk frequency in KHz corresponding to above perfLevel
    LwU32 gpuCache2clkFreqKHz;    // gpucache2clk frequency in KHz corresponding to above perfLevel

    //
    // XXXTODO: Temporary hack to unblock bugfix_main i17 rebase.
    // Get rid of the dispclkFreqKHz variable from impFbClkIn
    //
    LwU32 dispclkFreqKHz;
} LW5070_CTRL_IMP_FB_CLK_INPUT;
typedef struct LW5070_CTRL_IMP_FB_CLK_INPUT *PLW5070_CTRL_IMP_FB_CLK_INPUT;

typedef struct LW5070_CTRL_IMP_FB_HEAD_CHN_INPUT {
    LW_MEMORY_LAYOUT memoryLayout;
    LwU32            bpp;            // Bits per pixel
    LW_SUPER_SAMPLE  superSample;
    LwU32            blockHeightOverride;      // block height in gobs. Used to overwrite settings for IMP testing.
    LwBool           valid;          // Whether or not this entry should be considered when doing IMP calc
    LwU32            pcntLineNotLinebufferedS1000;  // Temporary for Bug 517220
} LW5070_CTRL_IMP_FB_HEAD_CHN_INPUT;
typedef struct LW5070_CTRL_IMP_FB_HEAD_CHN_INPUT *PLW5070_CTRL_IMP_FB_HEAD_CHN_INPUT;

typedef LW5070_CTRL_IMP_FB_HEAD_CHN_INPUT LW5070_CTRL_IMP_FB_HEAD_CORE_INPUT;
typedef LW5070_CTRL_IMP_FB_HEAD_CHN_INPUT LW5070_CTRL_IMP_FB_HEAD_BASE_INPUT;
typedef LW5070_CTRL_IMP_FB_HEAD_CHN_INPUT LW5070_CTRL_IMP_FB_HEAD_OVLY_INPUT;
typedef LW5070_CTRL_IMP_FB_HEAD_CHN_INPUT LW5070_CTRL_IMP_FB_HEAD_LWRS_INPUT;

// configuration for a single head
typedef struct LW5070_CTRL_IMP_FB_HEAD_INPUT {
    LW5070_CTRL_IMP_FB_HEAD_BASE_INPUT impFbHeadBaseIn;
    LW5070_CTRL_IMP_FB_HEAD_OVLY_INPUT impFbHeadOvlyIn;
    LW5070_CTRL_IMP_FB_HEAD_LWRS_INPUT impFbHeadLwrsIn;

    LwU32                              resX;            // screen resolution in X for this head
    LwU32                              resY;
    LwU32                              fbFetchRateKHz;  // pixel fetch rate desired by display in KHz

} LW5070_CTRL_IMP_FB_HEAD_INPUT;
typedef struct LW5070_CTRL_IMP_FB_HEAD_INPUT *PLW5070_CTRL_IMP_FB_HEAD_INPUT;

// Structure with all the parameters inferred form dispIMP
typedef struct LW5070_CTRL_IMP_FB_DISP_INPUT {
    LW5070_CTRL_IMP_FB_HEAD_INPUT impFbHeadIn[LW5070_CTRL_CMD_MAX_HEADS];
    LwU32                         reqdDispclkFreqKHz;
} LW5070_CTRL_IMP_FB_DISP_INPUT;
typedef struct LW5070_CTRL_IMP_FB_DISP_INPUT *PLW5070_CTRL_IMP_FB_DISP_INPUT;

typedef struct LW5070_CTRL_IMP_FB_INPUT {
    LW5070_CTRL_IMP_FB_REG_INPUT      impFbRegIn;
    LW5070_CTRL_IMP_FB_DRAM_CFG_INPUT impFbDramCfgIn;
    LW5070_CTRL_IMP_FB_CLK_INPUT      impFbClkIn;
    LW5070_CTRL_ASR_INPUT             impFbASRIn;
    LW5070_CTRL_IMP_FB_DISP_INPUT     impFbDispIn;
} LW5070_CTRL_IMP_FB_INPUT;
typedef struct LW5070_CTRL_IMP_FB_INPUT *PLW5070_CTRL_IMP_FB_INPUT;

typedef struct LW5070_CTRL_IMP_FB_HEAD_REG_OUTPUT {
    LwU32 memPoolBlks;   // Number of 32-byte blocks allocated for this head. LW_PFB_ISO_POOLCFG_LIMITn
    LwU32 memPoolBlksMin; // Number of 32-byte blocks allocated for this head LW_PFB_ISO_POOLCFG_LIMITn to have best perf
    LwU32 rxbCredits;    // Number of RXB credits allocated for this head. LW_PFB_RXB_RDATFIFO0_SLOTS_ISO_Dn
    LwU32 baseLbEntries; // Number of 32-byte credits in the mempool for base thread, per head (0 == disabled)
                        // LW_PFB_ISO_LINE_BUFFER_HLIMITn_BASE, LW_PFB_ISO_LINE_BUFFER_CTRL_EN_HnB
    LwU32 ovlyLbEntries; // Number of 32-byte credits in the mempool for overlay thread, per head (0 == disabled)
                        // LW_PFB_ISO_LINE_BUFFER_HLIMITn_OVLY, LW_PFB_ISO_LINE_BUFFER_CTRL_EN_HnO

    // This is the maximum amount of time, in 10ths of usecs, that hardware
    // requires to spool-up to a minimally safe level.  DMI duration values should
    // be programmed to ensure that there is at least this much time between a
    // contract being issued and the first pixel being fetched.  If this value is
    // zero, then a default spoolup time should be used.
    LwU32 required_spoolup_time;

    //
    // When non-zero, indicates that compaction should be applied to the head.  Normally
    // only one head can have compaction enabled at a time.
    //
    LwU32 headCompactable;

    LwU32 compactionPitch;

    //
    // Amount of the compacted surface that it actually used.
    // It is basically (computed pitch)*(number of used slices)
    // This value is less than compaction_surface_size.
    //
    LwU32 compactionSurfaceUsedSize;

    // Low and high watermarks, in pixels, for use by ISO_STUTTER feature.  Watermarks
    // are based on minimum fifo depth, in pixels, necessary to survive worst-case
    // latency. There are two high watermarks (HWMs), one in available pixels and the other in
    // raw 32B blocks, both of which must be exceeded before entering stutter drain mode.
    // A value of 0 is returned when watermark value not appropriate (e.g.head is disabled).
    // LW_PFB_ISO_STUTTER_HEAD0_LWM_THRESHOLD / LW_PFB_ISO_STUTTER_HEAD1_LWM_THRESHOLD
    // LW_PFB_ISO_STUTTER_HEAD0_HWM_THRESHOLD / LW_PFB_ISO_STUTTER_HEAD1_HWM_THRESHOLD
    // LW_PFB2_ISO_BLOCK_HWM(i)
    LwU32 stutter_lwm;
    LwU32 stutter_hwm;
    LwU32 stutter_blk_hwm;

    // Single watermark used by iso_hub for generating fidvid_enable signal.  Watermark
    // is based on min fifo depth, in pixels, necessary to survive worst-case
    // latency of an aggressive power-management event.
    // LW_PFB_ISO_FIDVID_HEAD0_WM_THRESHOLD/LW_PFB_ISO_FIDVID_HEAD1_WM_THRESHOLD
    LwU32 aggressive_pm_wm;

    // Mempool oclwpancy thresholds (low- and high-watermarks) for determining critical
    // states (hpiso_crit in gt209 or fifo_is_critical in gt21a) with hysteresis.  Values
    // are in 32B blocks of base pixels only.  A value of 0 is returned when watermark
    // value not appropriate (e.g. head is disabled).
    // LW_PFB_ISO_CRIT_LWM_HEAD0_THRESHOLD / LW_PFB_ISO_CRIT_LWM_HEAD1_THRESHOLD
    // LW_PFB_ISO_CRIT_HWM_HEAD0_THRESHOLD / LW_PFB_ISO_CRIT_HWM_HEAD1_THRESHOLD
    LwU32 hpiso_crit_lwm;
    LwU32 hpiso_crit_hwm;

    // igt21a critical signal generation avoids unnecessary critical assertion by monitoring
    // request service rate.  We don't assert head_is_critical signal when oclwpancy is below lwm
    // if the following request service is satisfied.  Value is minimum request rate
    // in 512B/1us = 0.5MB/s units.
    // LW_PFB_ISO_DCRIT(i)_RSR_RATE
    LwU32 rsrRate;

    // igt21a request service rate monitoring uses thresolds and hysteresis for rsr_critical
    // assertion.  Values are signed in 32B blocks.
    // LW_PFB_ISO_RSR(i)_BEHIND_THRESH, LW_PFB_ISO_RSR(i)_AHEAD_THRESH
    LwS32 rsrBehindThresh;
    LwS32 rsrAheadThresh;

    // IQ arbiter can use WRR weights to allocate TC2 bandwidth between
    // head0 and head1.  Values are in grants.
    // TODO: Put register names here
    LwU32 iqarb;

    // Per-head ISO training watermark thresholds used to ensure we don't underflow
    // when doing training.
    //     LW_PFB_TRAINING_THRESHOLD_D0/LW_PFB_TRAINING_THRESHOLD_D1
    // Not tow SW:  A value of '0' indicates that the value is invalid and
    // should *not* be written to the register.  Invalid registers can
    // occur if either a head is disabled *or* an invalid vfifo_pixel_shift
    // register was used for input.
    LwU32 perhead_iso_training_threshold;
    LwU32 perhead_iso_training_threshold_raw;

    // Critical watermark
    // LW_PFB_ARB_ISO_CRITICAL_THRESHOLD_HEAD0/1
    LwU32 iso_critical_threshhold;
    LwU32 iso_critical_threshhold_raw;
} LW5070_CTRL_IMP_FB_HEAD_REG_OUTPUT;
typedef struct LW5070_CTRL_IMP_FB_HEAD_REG_OUTPUT *PLW5070_CTRL_IMP_FB_HEAD_REG_OUTPUT;

typedef struct LW5070_CTRL_IMP_FB_REG_OUTPUT {
    LW5070_CTRL_IMP_FB_HEAD_REG_OUTPUT impFbHeadRegOut[LW5070_CTRL_CMD_MAX_HEADS];
    LwU32                              sharedRxbCredits;
} LW5070_CTRL_IMP_FB_REG_OUTPUT;
typedef struct LW5070_CTRL_IMP_FB_REG_OUTPUT *PLW5070_CTRL_IMP_FB_REG_OUTPUT;

typedef struct LW5070_CTRL_IMP_FB_OUTPUT {
    LwBool                              isPossible;     // Can FB satisfy the bandwidth demands of this mode?
    LwBool                              valid;          //
    LW5070_CTRL_IMP_FB_REG_OUTPUT       impFbRegOut;
    LW_DECLARE_ALIGNED(LwU64 memUtil, 8); // Percentage of overall DRAM bandwidth utilized for display
    LwU32                               rxbCntlLoopUtil;// Control-loop latency utilization
    LwU32                               dramConlwrrency;// DRAM conlwrrency

    // For ASR input - the maximum BW available for ISO (display)
    LwU32                               max_dram_bw_for_ISO;                   // MB/sec

    LwU32                               global_iso_training_threshold;         // LW_PFB_ISO_VFIFO_TRAINING_THRESH

    //struct imp_iso_outregs oregs0; // LB config registers for head0
    //struct imp_iso_outregs oregs1; // LB config registers for head1

    // Minimum clock values (needed to callwlate these to ensure the lwclk
    // is correct)
    //struct imp_sysclock_freqs min_sysclock_freqs;
    LwU32                               min_sysclock_freq;

    // Returned margin values are set to 99999 if the respective margin check is disabled
    // or inappopriate for the current configuration.
    LwS32                               latency_margin_10x[LW5070_CTRL_CMD_MAX_HEADS]; // margin, in 0.1us, till underflow in worst-case

    LW5070_CTRL_STUTTER_FEATURES_OUTPUT impFbASROut;
    //
    // During the IMP call for normal power (AltVid off), we make an estimate
    // of whether or not we think IMP can pass with AltVid on, and save the
    // result in altVidProbablyPossible.  If we think IMP can pass with AltVid
    // on, we will call IMP again with the DRAM parameters configured for
    // "AltVid on" to be sure, but if we don't think it will be able to pass,
    // we don't waste time with the additional IMP call(s).
    //
    LwBool                              altVidProbablyPossible;
    //
    // dram_bw_needed_for_ISO records the bandwidth needed for the current IMP
    // display configuration.
    //
    LwU32                               dram_bw_needed_for_ISO;

    //
    // critArbAddfactor records the arbiter weight to be added to the critical
    // head's arbiter weight to give more weight than non-critical heads
    // This factor needs to be subtracted from the non-critical head and is
    // callwlated assuming the number of heads as 2
    //
    LwU32                               critArbAddfactor;

    // These values are callwlated as a part of IMP
    LwU32                               iso2ixveLatencyNs; // in nanoseconds
    LwU32                               ixve2isoLatencyNs; // in nanoseconds
    LwU32                               totalIsohubIxveLatencyNs; // in nanoseconds

} LW5070_CTRL_IMP_FB_OUTPUT;
typedef struct LW5070_CTRL_IMP_FB_OUTPUT *PLW5070_CTRL_IMP_FB_OUTPUT;

//----------------------------------------------------------------------

typedef struct LW5070_CTRL_IMP_IS_MODE_POSSIBLE_INPUT {
    LW5070_CTRL_IMP_DISP_DISP_INPUT impDispIn;
    LW5070_CTRL_IMP_FB_INPUT        impFbIn;
} LW5070_CTRL_IMP_IS_MODE_POSSIBLE_INPUT;
typedef struct LW5070_CTRL_IMP_IS_MODE_POSSIBLE_INPUT *PLW5070_CTRL_IMP_IS_MODE_POSSIBLE_INPUT;

typedef struct LW5070_CTRL_IMP_IS_MODE_POSSIBLE_OUTPUT {
    LwBool                           isPossible;
    LW5070_CTRL_IMP_DISP_DISP_OUTPUT impDispOut;
    LW_DECLARE_ALIGNED(LW5070_CTRL_IMP_FB_OUTPUT impFbOut, 8);
} LW5070_CTRL_IMP_IS_MODE_POSSIBLE_OUTPUT;
typedef struct LW5070_CTRL_IMP_IS_MODE_POSSIBLE_OUTPUT *PLW5070_CTRL_IMP_IS_MODE_POSSIBLE_OUTPUT;

typedef enum LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE_FLAGS {
    LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE_FLAGS_ALL = 0,
    LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE_FLAGS_DISP = 1,
    LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE_FLAGS_FB = 2,
} LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE_FLAGS;

/*
 * LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE
 *
 * This command is intended to be used solely via mods verif. This
 * commands is similar to QUERY_IS_MODE_POSSIBLE with the exception
 * that all parameters (including DRAM timings etc) are provided by
 * the user.
 *
 * input
 *   Full input to IMP
 *
 * result
 *   Full output of IMP (see isPossible field)
 *
 * desiredImp
 *   Provides Transparency to Caller to Choose among FbImp/DispImp/Both
 *
 * PublicId
 *   Provides  access FB halified functions based on Public ID
 *   assigned with  HAL_IMPL_XXXX,this parameter chosen by FbImp Test
 *   automatically based on the fmodel and config file input "gpu.gpufamily"
 *   this is optional parameter,   Default value is 0
 *
 *   Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE (0x50700111) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_VERIF_QUERY_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_VERIF_QUERY_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW5070_CTRL_VERIF_QUERY_IS_MODE_POSSIBLE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS                        base;
    LW5070_CTRL_IMP_IS_MODE_POSSIBLE_INPUT             input;
    LW_DECLARE_ALIGNED(LW5070_CTRL_IMP_IS_MODE_POSSIBLE_OUTPUT output, 8);
    LW5070_CTRL_CMD_VERIF_QUERY_IS_MODE_POSSIBLE_FLAGS desiredImp;
    LwU32                                              PublicId;
} LW5070_CTRL_VERIF_QUERY_IS_MODE_POSSIBLE_PARAMS;

/*
 * LW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE
 *
 * This is a VERIF only backdoor to directly call the Fermi IMP algorithm.
 *
 * input
 *   Display and FB timings
 *
 * output
 *   Is mode possible, Mempool configuration, etc.
 *
 *   Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */

#define LW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE (0x50700114) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_FERMI_MAX_HEADS                    4
typedef struct LW5070_CTRL_FERMI_IMP_FB_HEAD_INPUT {
    LW_MEMORY_LAYOUT memoryLayoutBase;
    LW_MEMORY_LAYOUT memoryLayoutOverlay;
    LW_MEMORY_LAYOUT memoryLayoutLwrsor;

    LwBool           bOverlayEnabled;

    //
    // fbFetchRateKHz is the pixel fetch rate needed by the display, in KHz.
    // It is similar to pclk, but it is averaged over an entire scan line.
    // That is, we assume we have the time period of an entire line (including
    // blanking) to fetch all of the pixels needed for that line.
    //
    //     FetchRate = pclk * (HActive / HTotal)
    //
    LwU32            fbFetchRateKHz;
} LW5070_CTRL_FERMI_IMP_FB_HEAD_INPUT;

typedef struct LW5070_FERMI_LATENCY {
    LwU32 constant;     // nanoseconds
    LwU32 ramClks;
    LwU32 l2Clks;
    LwU32 xbarClks;
    LwU32 sysClks;
    LwU32 hubClks;
} LW5070_FERMI_LATENCY;

typedef struct LW5070_FERMI_CLOCKS {
    LW_DECLARE_ALIGNED(LwU64 display, 8);
    LW_DECLARE_ALIGNED(LwU64 ram, 8);
    LW_DECLARE_ALIGNED(LwU64 l2, 8);
    LW_DECLARE_ALIGNED(LwU64 xbar, 8);
    LW_DECLARE_ALIGNED(LwU64 hub, 8);
    //
    // IMP depends on sysclk, as well as other clocks.  However, IMP
    // callwlations may ignore sysclk because it is assumed that hubclk will
    // always be a more limiting factor.
    //
    LW_DECLARE_ALIGNED(LwU64 sys, 8);
} LW5070_FERMI_CLOCKS;
typedef struct LW5070_FERMI_CLOCKS *PLW5070_FERMI_CLOCKS;

typedef struct LW5070_FERMI_MIN_CLOCKS {
    LW_DECLARE_ALIGNED(LwU64 ram, 8);
    LW_DECLARE_ALIGNED(LwU64 l2, 8);
    LW_DECLARE_ALIGNED(LwU64 xbar, 8);
    LW_DECLARE_ALIGNED(LwU64 hub, 8);
    LW_DECLARE_ALIGNED(LwU64 sys, 8);
} LW5070_FERMI_MIN_CLOCKS;


#define LW5070_CTRL_CMD_FERMI_ISO_LB_BUFFERS 2

typedef LwBool lineBuffers[LW5070_CTRL_CMD_FERMI_ISO_LB_BUFFERS];

typedef struct LW5070_CTRL_FERMI_IMP_FB_INPUT {
    struct {
         // ramType = BUFFER_GDDR5, BUFFER_GDDR3_BGA136, or BUFFER_GDDR3_BGA144.
        LwU32  ramType;

        LwU32  tRC;                // ns
        LwU32  tRAS;               // ns
        LwU32  tRP;                // ns
        LwU32  tRCDRD;             // ns
        LwU32  timingRFC;          // units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  timingRP;           // units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  tWCK2MRS;           // units of MAX(CK_period, ltcclk_period/2)
                                   // On Kepler, units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  tWCK2TR;            // units of MAX(CK_period, ltcclk_period/2)
                                   // On Kepler, units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  tMRD;               // units of MAX(CK_period, ltcclk_period/2)
                                   // On Kepler, units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  tMRS2RDWCK;         // units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  tQPOPWCK;           // units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  tMRSTWCK;           // units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  EXT_BIG_TIMER;      // units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  STEP_LN;            // units of MAX(dramdiv4clk_period, ltcclk_period)
        LwU32  asrClkConstD4L1;    // units of MAX(dramdiv4clk_period, ltcclk_period)   // OBSOLETE
        LwU32  asrClkConstC1L2;    // units of MAX(CK_period, ltcclk_period/2)          // OBSOLETE
        LwU32  asrClkConstC1L1;    // units of MAX(CK_period, ltcclk_period)            // OBSOLETE
        LwU32  asrClkConstD1;      // units of dramclk_period                           // OBSOLETE

         // specifics of DRAM config which affect ASR exit time on GDDR5
        LwU32  tREFI;             // Refresh interval (ns)
        LwBool bAutoSync;         // autosync mode
        LwBool bFastExit;         // fast exit from self refresh
        LwBool bX16;              // x16/32 bit mode

         //
         // numBanks is the number of DRAM banks internal to a DRAM chip that a
         // request stream is using (typically less than the total number of
         // banks per chip).
         //
        LwU32  numBanks;

        LwU32  bytesPerClock;      // bytes per clock per DRAM chip (includes
                                   // fudge factor for overhead)
        LwU32  bytesPerClockFromBusWidth;   // bytes per clock per DRAM chip
                                   // (no fudge factor included)
        LwU32  bytesPerActivate;   // number of bytes available per RAS cycle, per bank
    } DRAM;

    struct {
        //
        // On GP102 and later, "dramChipCountPerBwUnit" is the number of DRAM
        // chips per subpa.  On GP100 and earlier products,
        // "dramChipCountPerBwUnit" is the number of DRAM chips per LTC.
        //
        LwU32 dramChipCountPerBwUnit;

        //
        // On GP102 and later, "enabledDramBwUnits" is to the total number of
        // enabled subpas on the GPU.  On GP100 and earlier products,
        // "enabledDramBwUnits" is the total number of enabled LTCs on the GPU
        // (which should be the same as "enabledLtcs", below).
        //
        LwU32 enabledDramBwUnits;

        //
        // On GP102 and later, "ltcBwUnitPipes" is the number of data pipes
        // outputting from a subpa (most likely set to "1").  On GP100 and
        // earlier products, "ltcBwUnitPipes" is equal to L2Slices.
        //
        LwU32 ltcBwUnitPipes;

        LwU32 L2Slices;            // How many L2 Slices are there per LTC (usually 2)
        LwU32 enabledLtcs;         // How many LTC instances are enabled?

        //
        // On GP102 and later, "bytesPerClock" is the number of bytes per subpa
        // output pipe per ltcclk.  On GP100 and earlier products,
        // "bytesPerClock" is the number of bytes per L2 slice per ltcclk.
        //
        LwU32 bytesPerClock;

        LwU32 ltcBytesPerClockWithDcmp; // Max bytes of decompression (DCMP)
                                        // data in and out of L2 cache per
                                        // ltcclk.
        LwU32 ltcBytesPerClockWithFos;  // Max bytes of filter-on-scanout (FOS)
                                        // data out of L2 cache per ltcclk.
        LwU32 awpPoolEntries;      // number of blocks in each AWP pool (1 pool per L2 slice)
        LwU32 bytesPerAwpBlock;
        LwU32 rrrbPoolEntries;     // number of blocks in the RRRB pool
        LwU32 bytesPerRrrbBlock;
    } FBP;

    struct {
        LwU32 bytesPerClock;        // Bytes transferred through a slice in a
                                    // single xbarclk
        LwU32 totalSlices;          // Total number of slices in the xbar,
                                    // including floorswept slices.
        //
        // maxFbpsPerXbarSlice records the number of FBPs connected to the xbar
        // slice that has the most FBPs connected to it.  In some cases, all
        // xbar slices will connect to the same number of FBPs (e.g., three in
        // an unfloorswept GF100 (see diagram for the pMinClocks->xbar
        // callwlation in fbgf100.c)).  But some xbar slices might have to
        // service more FBPs if floorsweeping creates an uneven configuration,
        // or if there are an odd number of FBPs to begin with.
        //
        // On Kepler and later, all FBPs are connected to all xbar slices, and
        // we are not concerned about uneven bandwidth, so we simply set this
        // parameter equal to "1".
        //
        LwU32 maxFbpsPerXbarSlice;

        LwU32 numHubPorts;
    } XBAR;

    struct {
        //
        // bytesPerClockX1000 is the number of bytes transferred through the
        // NISO hub in 1000 sysclks.
        //
        LwU32 bytesPerClockX1000;
    } NISO;

    struct {
        LwU32       linebufferAdditionalLines;    // Number of lines in the linebuffer  (1 for a 2-line hi, or 3 for 4-line hi)
        LwU32       linesFetchedForBlockLinear;   // How many lines of data are fetched for a BL surface?
        LwU32       linebufferTotalBlocks;        // Total number of (32 byte) memmpool blocks in the system
        LwU32       linebufferMinBlocksPerHead;
        LwU32       bytesPerClock;
        lineBuffers lineBufferingIsAllowed[LW5070_CTRL_CMD_FERMI_MAX_HEADS];  // Control to use linebuffering.
        //
        // Note that lineBufferingIsAllowed indicates whether "I want to use the line-buf or not, regardless
        // of whether the line-buf is supported" and linebufferAdditionalLines indicates if "the line-buf is
        // supported".
        //

        LwBool      maxFlipPerfRequested[LW5070_CTRL_CMD_FERMI_MAX_HEADS];
        LwBool      bMclkIhubOkToSwitchAllowed;
    } ISO;

    //
    // If dynamic clocking is enabled with PStates 3.0, perfLevel indexes the
    // 0-based IMP v-pstate that will be used for the IMP callwlations.
    //
    // If dynamic clocking is enabled with PStates 2.0, perfLevel indexes the
    // 0-based pstate (perf level) that will be used for the IMP callwlations.
    //
    // If dynamic clocking is disabled, perfLevel should be set to
    // PERFCTL_LEVELS, and IMP will use "current clocks".
    //
    LwU32                perfLevel;

    LW_DECLARE_ALIGNED(LW5070_FERMI_CLOCKS CLOCKS, 8);     // clock frequencies in Hz

    //
    // roundtripLatency measures the delay between the time when data is
    // requested at the ISO hub and the time the requested data is returned to
    // the ISO hub.  It is the sum of a fixed number of DRAM clocks, ltc
    // clocks, xbar clocks, and hub clocks, plus a constant.
    //
    LW5070_FERMI_LATENCY roundtripLatency;

    //
    // returnLatency measures only the time it takes for data to return from
    // DRAM to the ISO hub.
    //
    LW5070_FERMI_LATENCY returnLatency;                 // OBSOLETE

    struct {
        LwBool compressedSurfacesAllowed;   // Deprecated - not used by IMP.

        //
        // Note that bEccIsEnabled will be set to LW_FALSE for HBM, even though
        // HBM may have ECC enabled, because HBM ECC runs in parallel and has
        // no impact on bandwidth.
        //
        LwBool bEccIsEnabled;
        LwBool bUseStaticMempoolAllocation;

        //
        // If bForceMinMempool is set, we allocate only the minimum mempool
        // required for latency.  No additional mempool is allocated for line
        // buffering, ASR, or leftover fill.  This feature will normally be
        // disabled, but may be enabled for IMP verification.
        //
        LwBool bForceMinMempool;
    } CAPS;

    struct {
        LwBool isAllowed;
        LwU32  efficiencyThreshold;
        //
        // dllOn specifies whether the DLLs are on or off.  At lower memory
        // frequencies (lower P-States), the DLLs can be left off during normal
        // operation (not just in ASR).  If the DLLs are off, ASR wakeup can be
        // much faster, because it is not necessary to wait for DLL lock.
        //
        LwBool dllOn;
        LwU32  tXSR; // DRAM timing parameter tXSR ( Exit Self Refresh on a Read )
        LwU32  tXSNR;
        LwBool powerdown;
    } ASR;

    struct {
        LwBool bIsAllowed;
    } MSCG;

    LW5070_CTRL_FERMI_IMP_FB_HEAD_INPUT impFbHeadIn[LW5070_CTRL_CMD_FERMI_MAX_HEADS];
} LW5070_CTRL_FERMI_IMP_FB_INPUT;

typedef struct LW5070_CTRL_IMP_ISOHUB_INPUT {
    LwU32 data;
    struct {
        LwBool bMempoolCompression;// Introduced on Kepler.
    } CAPS;
    // Based on the cursor size, number of mempool entries required for cursor.
    LwU32 maxLwrsorEntries;

    //
    // Number of bytes requested in a single pitch linear memory request.
    // On Maxwell and later, the size of a pitch linear request is set equal to
    // the size of a block linear request.
    //
    LwU32 pitchFetchQuanta;
} LW5070_CTRL_IMP_ISOHUB_INPUT;
typedef struct LW5070_CTRL_IMP_ISOHUB_INPUT *PLW5070_CTRL_IMP_ISOHUB_INPUT;

typedef struct LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_INPUT {
    //
    // bGetMargin is set if the caller needs to know the bandwidth margin
    // available for the specified mode.
    //
    LwBool                          bGetMargin;
    LW5070_CTRL_IMP_DISP_DISP_INPUT impDispIn;
    LW_DECLARE_ALIGNED(LW5070_CTRL_FERMI_IMP_FB_INPUT impFbIn, 8);
    LW5070_CTRL_IMP_ISOHUB_INPUT    impIsohubIn;
} LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_INPUT;
typedef struct LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_INPUT *PLW5070_CTRL_FERMI_IS_MODE_POSSIBLE_INPUT;

typedef struct LW5070_IMP_FB_USAGE_VALUES {
    //
    // dramChip measures bandwidth for a single RAM chip.  It includes
    // overfetch and * AA factor.
    //
    LW_DECLARE_ALIGNED(LwU64 dramChip, 8);

    //
    // ltcBwUnit includes overfetch and * AA factor.  On GP102 and later, it is
    // the bandwidth per subpa.  On GP100 and earlier, it is the bandwdith per
    // ltc.
    //
    LW_DECLARE_ALIGNED(LwU64 ltcBwUnit, 8);
    // xbar includes overfetch
    LW_DECLARE_ALIGNED(LwU64 xbar, 8);
    //
    // sys - on Kepler, measures traffic from NISO hub to ISO hub; includes
    // overfetch
    //
    LW_DECLARE_ALIGNED(LwU64 sys, 8);
    // iso - measures traffic going INTO the ISO hub; includes overfetch
    LW_DECLARE_ALIGNED(LwU64 iso, 8);
    //
    // fos - L2 cache traffic devoted to filter-on-scanout (FOS) data
    // fos includes overfetch and no AA factor, per LTC
    //
    LW_DECLARE_ALIGNED(LwU64 fos, 8);
    LW_DECLARE_ALIGNED(LwU64 awp, 8);
    LW_DECLARE_ALIGNED(LwU64 rrrb, 8);
} LW5070_IMP_FB_USAGE_VALUES;
typedef struct LW5070_IMP_FB_USAGE_VALUES *PLW5070_IMP_FB_USAGE_VALUES;

#define LW5070_CTRL_ISOHUB_IMP_FAIL_NO_ERROR     0
#define LW5070_CTRL_ISOHUB_IMP_FAIL_DISPIMP      1
#define LW5070_CTRL_ISOHUB_IMP_FAIL_HUBCLK       2
#define LW5070_CTRL_ISOHUB_IMP_FAIL_MEMPOOL      3
#define LW5070_CTRL_ISOHUB_IMP_FAIL_FETCH_LIMIT  4
#define LW5070_CTRL_ISOHUB_IMP_FAIL_DMI_DURATION 5
#define LW5070_CTRL_ISOHUB_IMP_FAIL_MCLK_SWITCH  6
#define LW5070_CTRL_ISOHUB_IMP_FAIL_INTERNAL     7

#define LW5070_CTRL_DISP_IMP_FAIL_NO_ERROR       0
#define LW5070_CTRL_DISP_IMP_FAIL_BAD_PARAM      1

typedef struct LW5070_CTRL_IMP_ISOHUB_MEMPOOL_OUTPUT {
    LwU32 baseFullyLinesBuffered;
    LwU32 ovlyFullyLinesBuffered;

/*
 * Block linear surfaces cause the display to get back several lines of data
 * with each request.  Linebuffer stores data for subsequenet lines to prevent
 * the fetch from being wasted.  baseLimit and ovlyLimit specify for how many
 * pixels from each line (for base and overlay streams) can be buffered.
 */
    LwU32 baseLimit;
    LwU32 ovlyLimit;

    LwU32 latBufBlocks;
    LwU32 memPoolBlocks;
} LW5070_CTRL_IMP_ISOHUB_MEMPOOL_OUTPUT;
typedef struct LW5070_CTRL_IMP_ISOHUB_MEMPOOL_OUTPUT *PLW5070_CTRL_IMP_ISOHUB_MEMPOOL_OUTPUT;

typedef struct LW5070_CTRL_IMP_ISOHUB_OUTPUT {
    LwBool                                bIsPossible;
    LwU32                                 impFailReason;

    LW5070_CTRL_IMP_ISOHUB_MEMPOOL_OUTPUT memPoolSetting[LW5070_CTRL_CMD_FERMI_MAX_HEADS];
    LwU32                                 freeMemPoolBlocks;

    LW_DECLARE_ALIGNED(LwU64 reqdHubClk, 8);
    LwU32                                 fetchMeter[LW5070_CTRL_CMD_FERMI_MAX_HEADS];
    LwU32                                 fetchLimit[LW5070_CTRL_CMD_FERMI_MAX_HEADS];

    LwBool                                bIsMclkIhubOkToSwitchPossible;
    LwBool                                bOverrideModesetMempool;

    LW5070_CTRL_STUTTER_FEATURES_OUTPUT   sf;
    struct {
        LwU32 mclkMode;
        LwU32 mclkSubMode;
        LwU32 mclkDwcfWM;
        LwU32 mclkMidWM;
        LwS32 mclkMempool;
    } mclkSetting[LW5070_CTRL_CMD_FERMI_MAX_HEADS];

    struct {
        LwBool bEnable;
        //
        // The critical watermark represents the mempool oclwpancy level at which
        // pixel data must be requested in order to avoid underflow.
        //
        LwU32  wm;        // watermark
    } critSetting[LW5070_CTRL_CMD_FERMI_MAX_HEADS];

    LwU32 dmiDuration[LW5070_CTRL_CMD_FERMI_MAX_HEADS];
} LW5070_CTRL_IMP_ISOHUB_OUTPUT;
typedef struct LW5070_CTRL_IMP_ISOHUB_OUTPUT *PLW5070_CTRL_IMP_ISOHUB_OUTPUT;

#define MAX_DOMAIN_STRING_LENGTH (0x00000008)

typedef struct LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_OUTPUT {
    LwBool                           result;                                        // TRUE if the mode is possible.
    char                             worstCaseDomain[8];
    LwU32                            worstCaseMargin;
    LwS32                            memPool[LW5070_CTRL_CMD_FERMI_MAX_HEADS];      // Per head buffering
    LwU32                            fbFetchRateBytesPerSec[LW5070_CTRL_CMD_FERMI_MAX_HEADS];

    //
    // mempoolBaseBpp is the number of mempool bytes required for each pixel in
    // the base or core channel (whichever is larger).  For 64bpp pixels, it is
    // the downcolwerted size (four bytes instead of eight).  It will be zero
    // if the head is not active.
    //
    LwU32                            mempoolBaseCoreBpp[LW5070_CTRL_CMD_FERMI_MAX_HEADS];

    //
    // mempoolOvlyBpp is the number of mempool bytes required for each pixel in
    // the overlay channel.  It will be zero if the head is not active, or if
    // overlay is not enabled.
    //
    LwU32                            mempoolOvlyBpp[LW5070_CTRL_CMD_FERMI_MAX_HEADS];

    //
    // minClocks gives an estimate of the minimum clock frequencies necessary
    // to support the specified mode.  It is only valid if "result" is true
    // (i.e., if the mode is possible), and it is only an approximation because
    // it does not take everything into account that the clock frequencies
    // affect (e.g., the latency callwlation in _fbGetIsoRoundTripLatency()).
    //
    LW_DECLARE_ALIGNED(LW5070_FERMI_MIN_CLOCKS minClocks, 8);

    LW_DECLARE_ALIGNED(LW5070_FERMI_MIN_CLOCKS asrMinClocks, 8);

    LW_DECLARE_ALIGNED(LW5070_IMP_FB_USAGE_VALUES availableBandwidth, 8);
    LW_DECLARE_ALIGNED(LW5070_IMP_FB_USAGE_VALUES requiredBandwidth, 8);

    // required BW assuming no benefit from line buffering
    LW_DECLARE_ALIGNED(LW5070_IMP_FB_USAGE_VALUES requiredBandwidthNoLB, 8);

    //
    // bIsPossibleNoLB indicates if the mode would be possible without the
    // bandwidth benefit of line buffering.
    //
    LwBool                           bIsPossibleNoLB;

    //
    // adjustedRrrbPoolEntries is set to the number of entries that matches the
    // RRRB bandwidth with the required ISO bandwidth.  (First used on Kepler.)
    //
    LwU32                            adjustedRrrbPoolEntries;

    LW5070_CTRL_IMP_DISP_DISP_OUTPUT impDispOut;
    LW_DECLARE_ALIGNED(LW5070_CTRL_IMP_ISOHUB_OUTPUT impIsohubOut, 8);
} LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_OUTPUT;
typedef struct LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_OUTPUT *PLW5070_CTRL_FERMI_IS_MODE_POSSIBLE_OUTPUT;


#define LW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LW_DECLARE_ALIGNED(LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_INPUT input, 8);
    LW_DECLARE_ALIGNED(LW5070_CTRL_FERMI_IS_MODE_POSSIBLE_OUTPUT output, 8);
} LW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE_PARAMS;
typedef struct LW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE_PARAMS *PLW5070_CTRL_CMD_VERIF_QUERY_FERMI_IS_MODE_POSSIBLE_PARAMS;


/*
 * LW5070_CTRL_CMD_GET_PINSET_COUNT
 *
 * Get number of pinsets on this GPU.
 *
 *   pinsetCount [out]
 *     Number of pinsets on this GPU is returned in this parameter.
 *     This count includes pinsets that are not connected.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW5070_CTRL_CMD_GET_PINSET_COUNT (0x50700115) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_GET_PINSET_COUNT_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_GET_PINSET_COUNT_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW5070_CTRL_GET_PINSET_COUNT_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       pinsetCount;
} LW5070_CTRL_GET_PINSET_COUNT_PARAMS;


/*
 * LW5070_CTRL_CMD_GET_PINSET_PEER
 *
 * Retrieve the pinset/GPU that is connected to the specified pinset on
 * this GPU.
 *
 *   pinset [in]
 *     Pinset on this GPU for which peer info is to be returned must be
 *     specified in this parameter.
 *
 *   peerGpuId [out]
 *     Instance of the GPU on the other side of the connection is
 *     returned in this parameter.
 *
 *   peerPinset [out]
 *     Pinset on the other side of the connection is returned in this
 *     parameter.  If there is no connection then the value is
 *     LW5070_CTRL_CMD_GET_PINSET_PEER_PEER_PINSET_NONE.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW5070_CTRL_CMD_GET_PINSET_PEER                       (0x50700116) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_GET_PINSET_PEER_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_PINSET_PEER_PEER_GPUINSTANCE_NONE (0xffffffff)

#define LW5070_CTRL_CMD_GET_PINSET_PEER_PEER_PINSET_NONE      (0xffffffff)

#define LW5070_CTRL_GET_PINSET_PEER_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW5070_CTRL_GET_PINSET_PEER_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       pinset;

    LwU32                       peerGpuInstance;
    LwU32                       peerPinset;
} LW5070_CTRL_GET_PINSET_PEER_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW5070_CTRL_CMD_SET_RMFREE_FLAGS
 *
 * This command sets the flags for an upcoming call to RmFree().
 * After the RmFree() API runs successfully or not, the flags are cleared.
 *
 *   flags
 *     This parameter holds the LW0000_CTRL_GPU_SET_RMFREE_FLAGS_*
 *     flags to be passed for the next RmFree() command only.
 *     The flags can be one of those:
 *     - LW0000_CTRL_GPU_SET_RMFREE_FLAGS_NONE:
 *       explicitly clears the flags
 *     - LW0000_CTRL_GPU_SET_RMFREE_FLAGS_FREE_PRESERVES_HW:
 *       instructs RmFree() to preserve the HW configuration. After
 *       RmFree() is run this flag is cleared.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW5070_CTRL_CMD_SET_RMFREE_FLAGS         (0x50700117) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_SET_RMFREE_FLAGS_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_SET_RMFREE_FLAGS_NONE        0x00000000
#define LW5070_CTRL_SET_RMFREE_FLAGS_PRESERVE_HW 0x00000001
#define LW5070_CTRL_SET_RMFREE_FLAGS_PARAMS_MESSAGE_ID (0x17U)

typedef struct LW5070_CTRL_SET_RMFREE_FLAGS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       flags;
} LW5070_CTRL_SET_RMFREE_FLAGS_PARAMS;


/*
 * LW5070_CTRL_CMD_IMP_SET_GET_PARAMETER
 *
 * This command allows to set or get certain IMP parameters. Change of
 * values take effect on next modeset and is persistent across modesets
 * until the driver is unloaded or user changes the override.
 *
 *   index
 *     One of LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_XXX defines -
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IMP_ENABLE
 *       Only supports "get" operation. If FALSE, IMP is being bypassed and
 *       all Is Mode Possible queries are answered with "mode is possible"
 *       and registers normally set by IMP are not changed from their defaults.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED
 *       Should IMP consider using ASR. ASR won't be allowed unless it is set to
 *       "allowed" through both _IS_ASR_ALLOWED and _IS_ASR_ALLOWED_PER_PSTATE.
 *       Note that IMP will not run ASR and MSCG at the same time.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED_PER_PSTATE
 *       Should IMP consider using ASR when this pstate is being used. ASR won't
 *       be allowed unless it is set to "allowed" through both
 *       LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED and
 *       LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED_PER_PSTATE.
 *       So when LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED
 *       returns FALSE, IMP won't consider ASR for any p-state. Note that IMP
 *       will not run ASR and MSCG at the same time. This function is valid
 *       only on PStates 2.0 systems.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_MSCG_ALLOWED_PER_PSTATE
 *       Should IMP consider using MSCG when this pstate is being used. MSCG
 *       won't be allowed if the MSCG feature isn't enabled even if we set to
 *       "allowed" through
 *       LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_MSCG_ALLOWED_PER_PSTATE.
 *       Use LW2080_CTRL_CMD_MC_QUERY_POWERGATING_PARAMETER to query if MSCG is
 *       supported and enabled. Note that IMP will not run ASR and MSCG at the
 *       same time. This function is valid only on PStates 2.0 systems.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_STUTTER_FEATURE_PER_PSTATE
 *       Only supports "get" operation. Returns which stutter feature is being
 *       engaged in hardware when running on the given pstate. Valid values are:
 *         LW5070_CTRL_IMP_STUTTER_FEATURE_NONE
 *           This value indicates no stutter feature is enabled.
 *         LW5070_CTRL_IMP_STUTTER_FEATURE_ASR
 *           This value indicates ASR is the current enabled stutter feature.
 *         LW5070_CTRL_IMP_STUTTER_FEATURE_MSCG
 *           This value indicates MSCG is the current enabled stutter feature.
 *       Note that system will not run ASR and MSCG at the same time. This
 *       function is valid only on PStates 2.0 systems.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_STUTTER_FEATURE_PREDICTED_EFFICIENCY_PER_PSTATE
 *       Only supports "get" operation. Returns the efficiency which IMP
 *       predicted for the engaged stutter feature (ASR or MSCG) when running
 *       on the given pstate. Normally, the actual efficiency should be higher
 *       than the callwlated predicted efficiency. For MSCG, the predicted
 *       efficiency assumes no mempool compression. If compression is enabled
 *       with MSCG, the actual efficiency may be significantly higher. Returns
 *       0 if no stutter feature is running. On PStates 3.0 systems, the
 *       pstateApi parameter is ignored, and the result is returned for the min
 *       IMP v-pstate possible.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS
 *       Only supports "get" operation. Returns information about what the possible
 *       mclk switch is.  Valid fields are:
 *         LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_POSSIBLE
 *           This field is not head-specific and indicates if mclk switch is
 *           possible with the current mode.
 *         LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_OVERRIDE_MEMPOOL
 *           This field is not head-specific and indicates if mclk switch is
 *           possible with the nominal mempool settings (_NO) or if special
 *           settings are required in order for mclk switch to be possible (_YES).
 *         LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_MID_WATERMARK
 *           Each head has its own setting for this field.  If this field is
 *           set to _YES, then the specified head will allow mclk switch to
 *           begin if mempool oclwpancy exceeds the MID_WATERMARK setting.
 *         LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_DWCF
 *           Each head has its own setting for this field.  If this field is
 *           set to _YES, then the specified head will allow mclk switch to
 *           begin if the head is in its DWCF interval, and the mempool
 *           oclwpancy is greater than or equal to the DWCF watermark.
 *       Note:  If neither _MID_WATERMARK nor _DWCF is set to _YES, then the
 *       specified head is ignored when determining when it is OK to start an
 *       mclk switch.  Mclk switch must be allowed (or ignored) by all heads
 *       before an mclk switch will actually begin.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_FORCE_MIN_MEMPOOL
 *       Should min mempool be forced.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MEMPOOL_COMPRESSION
 *       Should mempool compression be enabled.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_LWRSOR_SIZE
 *       The cursor size (in horizontal pixels) used by IMP (rather than the
 *       actual cursor size) for its computation.
 *       A maximum value is in place for what can be set. It can be queried
 *       after resetting the value - it gets reset to the maximum possible
 *       value.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_ENABLE
 *       This is to Enable/Disable ISO FB Latency Test.
 *       The test records the max ISO FB latency for all heads during the test period (excluding modeset time).
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_WC_TOTAL_LATENCY
 *       This is used to retrieve callwlated wcTotalLatency of ISO FB Latency Test.
 *       wcTotalLatency is the worst case time for a request's data to come back after the request is issued.
 *       It is the sum of IMP callwlated FbLatency and stream delay.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_MAX_LATENCY
 *       This is used to retrieve the max latency among all heads during the whole ISO FB Latency Test.
 *       The max latency can be used to compare with the wcTotalLatency we callwlated.
 *       It decides whether the ISO FB Latency Test is passed or not.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_MAX_TEST_PERIOD
 *       This is used to retrieve the max test period during the whole ISO FB Latency Test.
 *       By experimental result, the test period should be at least 10 secs to approximate the
 *       worst case Fb latency in real situation.
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_GLITCHLESS_MODESET_ENABLE
 *       This enables or disables glitchless modesets.  Modesets can be
 *       glitchless if:
 *       (1) There are no raster timing changes, and
 *       (2) The resource requirements of all bandwidth clients are either not
 *           changing, or they are all changing in the same direction (all
 *           increasing or all decreasing).
 *       If glitchless modeset is disabled, or is not possible, heads will be
 *       blanked during the modeset transition.
 *   pstateApi
 *     LW2080_CTRL_PERF_PSTATES_PXXX value.
 *     Required for LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED_PER_PSTATE,
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_MSCG_ALLOWED_PER_PSTATE,
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_STUTTER_FEATURE_PER_PSTATE and
 *     LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_STUTTER_FEATURE_PREDICTED_EFFICIENCY_PER_PSTATE
 *     on PStates 2.0 systems. For other indices must be
 *     LW2080_CTRL_PERF_PSTATES_UNDEFINED.  Not used on PStates 3.0 systems.
 *   head
 *     Head index, which is required when querying Mclk switch feature.
 *     (index = LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS)
 *   operation
 *       LW5070_CTRL_IMP_SET_GET_PARAMETER_OPERATION_GET
 *         Indicates a "get" operation.
 *       LW5070_CTRL_IMP_SET_GET_PARAMETER_OPERATION_SET
 *         Indicates a "set" operation.
 *       LW5070_CTRL_IMP_SET_GET_PARAMETER_OPERATION_RESET
 *         Indicates a "reset" operation. This operation will reset the values for
 *         all indices to their RM defaults.
 *   value
 *     Value for new setting of a "set" operation, or the returned value of a
 *     "get" operation; for enable/disable operations, "enable" is non-zero,
 *     and "disable" is zero.
 *
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_POINTER
 *   LW_ERR_ILWALID_INDEX            specified index is not supported
 *   LW_ERR_INSUFFICIENT_RESOURCES   cannot handle any more overrides
 *   LW_ERR_ILWALID_OBJECT     the struct needed to get the specified information
 *                                              is not marked as valid
 *   LW_ERR_ILWALID_STATE            the parameter has been set but resetting will
 *                                              not be possible
 */
#define LW5070_CTRL_CMD_IMP_SET_GET_PARAMETER (0x50700118) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_IMP_SET_GET_PARAMETER_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_IMP_SET_GET_PARAMETER_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW5070_CTRL_IMP_SET_GET_PARAMETER_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       index;
    LwU32                       pstateApi;
    LwU32                       head;
    LwU32                       operation;
    LwU32                       value;
} LW5070_CTRL_IMP_SET_GET_PARAMETER_PARAMS;

/* valid operation values */
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_OPERATION_GET                                                0
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_OPERATION_SET                                                1
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_OPERATION_RESET                                              2

/* valid index value */
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_NONE                                                   (0x00000000)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IMP_ENABLE                                             (0x00000001)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED                                         (0x00000002)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_ASR_ALLOWED_PER_PSTATE                              (0x00000003)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_IS_MSCG_ALLOWED_PER_PSTATE                             (0x00000004)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_STUTTER_FEATURE_PER_PSTATE                             (0x00000005)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_STUTTER_FEATURE_PREDICTED_EFFICIENCY_PER_PSTATE        (0x00000006)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS                            (0x00000007)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_FORCE_MIN_MEMPOOL                                      (0x00000008)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MEMPOOL_COMPRESSION                                    (0x00000009)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_LWRSOR_SIZE                                            (0x0000000A)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_ENABLE                               (0x0000000B)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_WC_TOTAL_LATENCY                     (0x0000000C)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_MAX_LATENCY                          (0x0000000D)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOFBLATENCY_TEST_MAX_TEST_PERIOD                      (0x0000000E)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_GLITCHLESS_MODESET_ENABLE                              (0x0000000F)

/* valid LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_ISOHUB_STUTTER_FEATURE values */
#define LW5070_CTRL_IMP_STUTTER_FEATURE_NONE                                                           0
#define LW5070_CTRL_IMP_STUTTER_FEATURE_ASR                                                            1
#define LW5070_CTRL_IMP_STUTTER_FEATURE_MSCG                                                           2

/* valid LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE values */
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_POSSIBLE                      0:0
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_POSSIBLE_NO          (0x00000000)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_POSSIBLE_YES         (0x00000001)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_OVERRIDE_MEMPOOL              1:1
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_OVERRIDE_MEMPOOL_NO  (0x00000000)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_OVERRIDE_MEMPOOL_YES (0x00000001)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_MID_WATERMARK                 2:2
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_MID_WATERMARK_NO     (0x00000000)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_MID_WATERMARK_YES    (0x00000001)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_DWCF                          3:3
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_DWCF_NO              (0x00000000)
#define LW5070_CTRL_IMP_SET_GET_PARAMETER_INDEX_MCLK_SWITCH_FEATURE_OUTPUTS_VALUE_DWCF_YES             (0x00000001)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW5070_CTRL_CMD_SET_MEMPOOL_WAR_FOR_BLIT_TEARING
 *
 * This command engages the WAR for blit tearing caused by huge mempool size and
 * mempool compression. The EVR in aero off mode uses scanline info to predict
 * where the scanline will be at a later time. Since RG scanline is used to perform
 * front buffer blits and isohub buffers large amount of display data it may have
 * fetched several lines of data ahead of where the RG is scanning out leading to
 * video tearing. The WAR for this problem is to reduce the amount of data fetched.
 *
 *   base
 *     This struct must be the first member of all 5070 control calls containing
 *     the subdeviceIndex.
 *   bEngageWAR
 *     Indicates if mempool WAR has to be engaged or disengaged.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW5070_CTRL_CMD_SET_MEMPOOL_WAR_FOR_BLIT_TEARING                                               (0x50700119) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_SET_MEMPOOL_WAR_FOR_BLIT_TEARING_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_SET_MEMPOOL_WAR_FOR_BLIT_TEARING_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW5070_CTRL_SET_MEMPOOL_WAR_FOR_BLIT_TEARING_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwBool                      bEngageWAR;
} LW5070_CTRL_SET_MEMPOOL_WAR_FOR_BLIT_TEARING_PARAMS;
typedef struct LW5070_CTRL_SET_MEMPOOL_WAR_FOR_BLIT_TEARING_PARAMS *PLW5070_CTRL_SET_MEMPOOL_WAR_FOR_BLIT_TEARING_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#define LW5070_CTRL_CMD_GET_ACTIVE_VIEWPORT_BASE (0x50700120) /* finn: Evaluated from "(FINN_LW50_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_ACTIVE_VIEWPORT_BASE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_ACTIVE_VIEWPORT_BASE_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW5070_CTRL_CMD_GET_ACTIVE_VIEWPORT_BASE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;
    LwU32                       activeViewportBase;
} LW5070_CTRL_CMD_GET_ACTIVE_VIEWPORT_BASE_PARAMS;
typedef struct LW5070_CTRL_CMD_GET_ACTIVE_VIEWPORT_BASE_PARAMS *PLW5070_CTRL_CMD_GET_ACTIVE_VIEWPORT_BASE_PARAMS;

/* _ctrl5070chnc_h_ */
