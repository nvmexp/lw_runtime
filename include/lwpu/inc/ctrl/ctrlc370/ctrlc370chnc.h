/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc370/ctrlc370chnc.finn
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
#include "ctrl/ctrl5070/ctrl5070chnc.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * The following control calls are defined in ctrl5070chnc.h, but they are
 * still supported on LWC370. We redirect these control cmds to LW5070_CTRL_CMD,
 * and keep the _PARAMS unchanged for now.
 */

#define LWC370_CTRL_CMD_GET_PINSET_COUNT                 LW5070_CTRL_CMD_GET_PINSET_COUNT
#define LWC370_CTRL_CMD_GET_PINSET_PEER                  LW5070_CTRL_CMD_GET_PINSET_PEER
#define LWC370_CTRL_CMD_SET_RMFREE_FLAGS                 LW5070_CTRL_CMD_SET_RMFREE_FLAGS
#define LWC370_CTRL_CMD_IMP_SET_GET_PARAMETER            LW5070_CTRL_CMD_IMP_SET_GET_PARAMETER
#define LWC370_CTRL_CMD_SET_MEMPOOL_WAR_FOR_BLIT_TEARING LW5070_CTRL_CMD_SET_MEMPOOL_WAR_FOR_BLIT_TEARING

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#define LWC370_CTRL_CMD_CHANNEL_STATE_IDLE                              LWBIT(0)
#define LWC370_CTRL_CMD_CHANNEL_STATE_QUIESCENT1                        LWBIT(2)
#define LWC370_CTRL_CMD_CHANNEL_STATE_QUIESCENT2                        LWBIT(3)
#define LWC370_CTRL_CMD_CHANNEL_STATE_BUSY                              LWBIT(6)
#define LWC370_CTRL_CMD_CHANNEL_STATE_DEALLOC                           LWBIT(7)
#define LWC370_CTRL_CMD_CHANNEL_STATE_DEALLOC_LIMBO                     LWBIT(8)
#define LWC370_CTRL_CMD_CHANNEL_STATE_EFI_INIT1                         LWBIT(11)
#define LWC370_CTRL_CMD_CHANNEL_STATE_EFI_INIT2                         LWBIT(12)
#define LWC370_CTRL_CMD_CHANNEL_STATE_EFI_OPERATION                     LWBIT(13)
#define LWC370_CTRL_CMD_CHANNEL_STATE_VBIOS_INIT1                       LWBIT(14)
#define LWC370_CTRL_CMD_CHANNEL_STATE_VBIOS_INIT2                       LWBIT(15)
#define LWC370_CTRL_CMD_CHANNEL_STATE_VBIOS_OPERATION                   LWBIT(16)
#define LWC370_CTRL_CMD_CHANNEL_STATE_UNCONNECTED                       LWBIT(17)
#define LWC370_CTRL_CMD_CHANNEL_STATE_INIT1                             LWBIT(18)
#define LWC370_CTRL_CMD_CHANNEL_STATE_INIT2                             LWBIT(19)
#define LWC370_CTRL_CMD_CHANNEL_STATE_SHUTDOWN1                         LWBIT(20)
#define LWC370_CTRL_CMD_CHANNEL_STATE_SHUTDOWN2                         LWBIT(21)

#define LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_CORE        1
#define LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW      32
#define LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW_IMM  32
#define LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WRITEBACK   8
#define LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_LWRSOR      8

/*
 * LWC370_CTRL_CMD_IDLE_CHANNEL
 *
 * This command tries to wait or forces the desired channel state.
 *
 *      channelClass
 *          This field indicates the hw class number (0xC378 - 0xC37E).
 *          It's defined in the h/w header (i.e. clc37d.h, etc.).
 *
 *      channelInstance
 *          This field indicates which instance of the channelClass the cmd is
 *          meant for. (zero-based)
 *
 *      desiredChannelStateMask
 *          This field indicates the desired channel states. When more than
 *          one bit is set, RM will return whenever it finds hardware on one
 *          of the states in the bistmask.
 *          Normal options are IDLE, QUIESCENT1 and QUIESCENT2.
 *          Verif only option includes BUSY as well.
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
#define LWC370_CTRL_CMD_IDLE_CHANNEL                     (0xc3700101) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LWC370_CTRL_IDLE_CHANNEL_PARAMS_MESSAGE_ID" */

#define LWC370_CTRL_IDLE_CHANNEL_MAX_INSTANCE_CORE       LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_CORE
#define LWC370_CTRL_IDLE_CHANNEL_MAX_INSTANCE_WINDOW     LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW
#define LWC370_CTRL_IDLE_CHANNEL_MAX_INSTANCE_WINDOW_IMM LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW_IMM
#define LWC370_CTRL_IDLE_CHANNEL_MAX_INSTANCE_WRITEBACK  LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WRITEBACK
#define LWC370_CTRL_IDLE_CHANNEL_MAX_INSTANCE_LWRSOR     LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_LWRSOR

#define LWC370_CTRL_IDLE_CHANNEL_STATE_IDLE                 LWC370_CTRL_CMD_CHANNEL_STATE_IDLE
#define LWC370_CTRL_IDLE_CHANNEL_STATE_QUIESCENT1           LWC370_CTRL_CMD_CHANNEL_STATE_QUIESCENT1
#define LWC370_CTRL_IDLE_CHANNEL_STATE_QUIESCENT2           LWC370_CTRL_CMD_CHANNEL_STATE_QUIESCENT2

#define LWC370_CTRL_IDLE_CHANNEL_STATE_BUSY                 LWC370_CTRL_CMD_CHANNEL_STATE_BUSY

#define LWC370_CTRL_IDLE_CHANNEL_ACCL_NONE               (0x00000000)
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_PI             (LWBIT(0))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_SKIP_NOTIF            (LWBIT(1))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_SKIP_SEMA             (LWBIT(2))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_INTERLOCK      (LWBIT(3))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_FLIPLOCK       (LWBIT(4))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_TRASH_ONLY            (LWBIT(5))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_TRASH_AND_ABORT       (LWBIT(6))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_SKIP_SYNCPOINT        (LWBIT(7))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_TIMESTAMP      (LWBIT(8))
#define LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_MGI            (LWBIT(9))

#define LWC370_CTRL_IDLE_CHANNEL_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC370_CTRL_IDLE_CHANNEL_PARAMS {
    LWC370_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;
    LwU32                       desiredChannelStateMask;
    LwU32                       accelerators;        // For future expansion. Not yet implemented
    LwU32                       timeout;             // For future expansion. Not yet implemented
    LwBool                      restoreDebugMode;
} LWC370_CTRL_IDLE_CHANNEL_PARAMS;

/*
 * LWC370_CTRL_CMD_SET_ACCL
 *
 *   This command turns accelerators on and off. The use of this command
 *   should be restricted as it may have undesirable effects. It's
 *   purpose is to provide a mechanism for clients to use the
 *   accelerator bits to get into states that are either not detectable
 *   by the RM or may take longer to reach than we think is reasonable
 *   to wait in the RM.
 *
 * LWC370_CTRL_CMD_GET_ACCL
 *
 *   This command queries the current state of the accelerators.
 *
 *      channelClass
 *          This field indicates the hw class number (0xC378 - 0xC37E).
 *          It's defined in the h/w header (i.e. clc37d.h, etc.).
 *
 *      channelInstance
 *          This field indicates which instance of the channelClass the cmd is
 *          meant for. (zero-based)
 *
 *      accelerators
 *          Accelerators to be set in the SET_ACCEL command. Returns the
 *          lwrrently set accelerators on the GET_ACCEL command.
 */
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 *          These definitions are based on the CHNCTL_X register definitions
 *          from //dev/display/lwdisplay/2.0/manuals/dev_display_fe.ref
 */
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
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

#define LWC370_CTRL_CMD_SET_ACCL                 (0xc3700102) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LWC370_CTRL_SET_ACCL_PARAMS_MESSAGE_ID" */

#define LWC370_CTRL_CMD_GET_ACCL                 (0xc3700103) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_CHNCTL_INTERFACE_ID << 8) | 0x3" */

#define LWC370_CTRL_ACCL_MAX_INSTANCE_CORE       LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_CORE
#define LWC370_CTRL_ACCL_MAX_INSTANCE_WINDOW     LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW
#define LWC370_CTRL_ACCL_MAX_INSTANCE_WINDOW_IMM LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW_IMM
#define LWC370_CTRL_ACCL_MAX_INSTANCE_WRITEBACK  LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WRITEBACK
#define LWC370_CTRL_ACCL_MAX_INSTANCE_LWRSOR     LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_LWRSOR

#define LWC370_CTRL_ACCL_NONE                    LWC370_CTRL_IDLE_CHANNEL_ACCL_NONE
#define LWC370_CTRL_ACCL_IGNORE_PI                  LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_PI
#define LWC370_CTRL_ACCL_SKIP_NOTIF                 LWC370_CTRL_IDLE_CHANNEL_ACCL_SKIP_NOTIF
#define LWC370_CTRL_ACCL_SKIP_SEMA                  LWC370_CTRL_IDLE_CHANNEL_ACCL_SKIP_SEMA
#define LWC370_CTRL_ACCL_IGNORE_INTERLOCK           LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_INTERLOCK
#define LWC370_CTRL_ACCL_IGNORE_FLIPLOCK            LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_FLIPLOCK
#define LWC370_CTRL_ACCL_TRASH_ONLY                 LWC370_CTRL_IDLE_CHANNEL_ACCL_TRASH_ONLY
#define LWC370_CTRL_ACCL_TRASH_AND_ABORT            LWC370_CTRL_IDLE_CHANNEL_ACCL_TRASH_AND_ABORT
#define LWC370_CTRL_ACCL_SKIP_SYNCPOINT             LWC370_CTRL_IDLE_CHANNEL_ACCL_SKIP_SYNCPOINT
#define LWC370_CTRL_ACCL_IGNORE_TIMESTAMP           LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_TIMESTAMP
#define LWC370_CTRL_ACCL_IGNORE_MGI                 LWC370_CTRL_IDLE_CHANNEL_ACCL_IGNORE_MGI
#define LWC370_CTRL_SET_ACCL_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC370_CTRL_SET_ACCL_PARAMS {
    LWC370_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;
    LwU32                       accelerators;
    LwU32                       accelMask;
} LWC370_CTRL_SET_ACCL_PARAMS;
typedef LWC370_CTRL_SET_ACCL_PARAMS LWC370_CTRL_GET_ACCL_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC370_CTRL_CMD_GET_ACCL_FINN_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC370_CTRL_CMD_GET_ACCL_FINN_PARAMS {
    LWC370_CTRL_GET_ACCL_PARAMS params;
} LWC370_CTRL_CMD_GET_ACCL_FINN_PARAMS;




/*
 * LWC370_CTRL_CMD_GET_CHANNEL_INFO
 *
 * This command returns the current channel state.
 *
 *      channelClass
 *          This field indicates the hw class number (0xC378 - 0xC37E).
 *          It's defined in the h/w header (i.e. clc37d.h, etc.).
 *
 *      channelInstance
 *          This field indicates which instance of the channelClass the cmd is
 *          meant for. (zero-based)
 *
 *      channelState
 *          This field indicates the desired channel state in a mask form that
 *          is compatible with LWC370_CTRL_CMD_IDLE_CHANNEL. A mask format
 *          allows clients to check for one from a group of states.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 *
 * Display driver uses this call to ensure that all it's methods have
 * propagated through hardware's internal fifo
 * (LWC370_CTRL_GET_CHANNEL_INFO_STATE_NO_METHOD_PENDING) before it calls
 * RM to check whether or not the mode it set up in Assembly State Cache will
 * be possible. Note that display driver can not use completion notifier in
 * this case because completion notifier is associated with Update and Update
 * will propagate the state from Assembly to Armed and when checking the
 * possibility of a mode, display driver wouldn't want Armed state to be
 * affected.
 */
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * These definitions are based on the state machine defined in
 * //hw/doc/gpu/display/LWDisplay/2.0/specifications/LWDisplay_FE_Method_Fetch_IAS.docx
 */
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#define LWC370_CTRL_CMD_GET_CHANNEL_INFO                     (0xc3700104) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_CHNCTL_INTERFACE_ID << 8) | LWC370_CTRL_CMD_GET_CHANNEL_INFO_PARAMS_MESSAGE_ID" */

#define LWC370_CTRL_GET_CHANNEL_INFO_MAX_INSTANCE_CORE       LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_CORE
#define LWC370_CTRL_GET_CHANNEL_INFO_MAX_INSTANCE_WINDOW     LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW
#define LWC370_CTRL_GET_CHANNEL_INFO_MAX_INSTANCE_WINDOW_IMM LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WINDOW_IMM
#define LWC370_CTRL_GET_CHANNEL_INFO_MAX_INSTANCE_WRITEBACK  LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_WRITEBACK
#define LWC370_CTRL_GET_CHANNEL_INFO_MAX_INSTANCE_LWRSOR     LWC370_CTRL_CMD_MAX_CHANNEL_INSTANCE_LWRSOR

#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_IDLE                 LWC370_CTRL_CMD_CHANNEL_STATE_IDLE
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_BUSY                 LWC370_CTRL_CMD_CHANNEL_STATE_BUSY
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_DEALLOC              LWC370_CTRL_CMD_CHANNEL_STATE_DEALLOC
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_DEALLOC_LIMBO        LWC370_CTRL_CMD_CHANNEL_STATE_DEALLOC_LIMBO
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_EFI_INIT1            LWC370_CTRL_CMD_CHANNEL_STATE_EFI_INIT1
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_EFI_INIT2            LWC370_CTRL_CMD_CHANNEL_STATE_EFI_INIT2
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_EFI_OPERATION        LWC370_CTRL_CMD_CHANNEL_STATE_EFI_OPERATION
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_VBIOS_INIT1          LWC370_CTRL_CMD_CHANNEL_STATE_VBIOS_INIT1
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_VBIOS_INIT2          LWC370_CTRL_CMD_CHANNEL_STATE_VBIOS_INIT2
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_VBIOS_OPERATION      LWC370_CTRL_CMD_CHANNEL_STATE_VBIOS_OPERATION
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_UNCONNECTED          LWC370_CTRL_CMD_CHANNEL_STATE_UNCONNECTED
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_INIT1                LWC370_CTRL_CMD_CHANNEL_STATE_INIT1
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_INIT2                LWC370_CTRL_CMD_CHANNEL_STATE_INIT2
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_SHUTDOWN1            LWC370_CTRL_CMD_CHANNEL_STATE_SHUTDOWN1
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_SHUTDOWN2            LWC370_CTRL_CMD_CHANNEL_STATE_SHUTDOWN2
#define LWC370_CTRL_GET_CHANNEL_INFO_STATE_NO_METHOD_PENDING    LWC370_CTRL_GET_CHANNEL_INFO_STATE_IDLE
#define LWC370_CTRL_CMD_GET_CHANNEL_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWC370_CTRL_CMD_GET_CHANNEL_INFO_PARAMS {
    LWC370_CTRL_CMD_BASE_PARAMS base;
    LwU32                       channelClass;
    LwU32                       channelInstance;
    LwBool                      IsChannelInDebugMode;
    LwU32                       channelState;
} LWC370_CTRL_CMD_GET_CHANNEL_INFO_PARAMS;


