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
// Source file: ctrl/ctrl9074.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW9074_CTRL_CMD(cat,idx)  LWXXXX_CTRL_CMD(0x9074, LW9074_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW9074_CTRL_RESERVED (0x00)
#define LW9074_CTRL_SEM      (0x01)

/*
 * LW9074_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LW9074_CTRL_CMD_NULL (0x90740000) /* finn: Evaluated from "(FINN_GF100_TIMED_SEMAPHORE_SW_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW9074_CTRL_CMD_FLUSH
 *
 * This command is intended to aid in idling/flushing a channel containing
 * methods ilwoked against a specific LW9074 object.
 *
 * This control manipulates an LW9074 object's  "flushing" state.
 *
 * Typical usage is:
 * LwRmControl(LW9074_CTRL_CMD_FLUSH, {TRUE, flushDelay});
 * LwRmIdleChannel(channel_containing_the_9074_object);
 * LwRmControl(LW9074_CTRL_CMD_FLUSH, {FALSE, 0});
 *
 * When an LW9074 object is placed into the flushing state, a snaphot of the
 * current timer value is taken, and "maxFlushTime" is added to this. This
 * value is the "flush limit timestamp". Any previously or newly ilwoked
 * LW9074_SEMAPHORE_SCHED methods and LW9074_CTRL_CMD_RELEASE requests that
 * specify a release timestamp at or after this "flush limit timestamp" will
 * immediately release the specified semaphore, without waiting for the
 * specified timestamp, and write a DONE_FORCED value to the specified notifier.
 */
#define LW9074_CTRL_CMD_FLUSH (0x90740101) /* finn: Evaluated from "(FINN_GF100_TIMED_SEMAPHORE_SW_SEM_INTERFACE_ID << 8) | LW9074_CTRL_CMD_FLUSH_PARAMS_MESSAGE_ID" */

#define LW9074_CTRL_CMD_FLUSH_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW9074_CTRL_CMD_FLUSH_PARAMS {
    LwU32 isFlushing;
    LW_DECLARE_ALIGNED(LwU64 maxFlushTime, 8);
} LW9074_CTRL_CMD_FLUSH_PARAMS;

/*
 * LW9074_CTRL_CMD_GET_TIME
 *
 * Retrieve the current time value.
 */
#define LW9074_CTRL_CMD_GET_TIME (0x90740102) /* finn: Evaluated from "(FINN_GF100_TIMED_SEMAPHORE_SW_SEM_INTERFACE_ID << 8) | LW9074_CTRL_CMD_GET_TIME_PARAMS_MESSAGE_ID" */

#define LW9074_CTRL_CMD_GET_TIME_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW9074_CTRL_CMD_GET_TIME_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 lwrrentTime, 8);
} LW9074_CTRL_CMD_GET_TIME_PARAMS;

/*
 * LW9074_CTRL_CMD_RELEASE
 *
 * This command adds a timed semaphore release request. When the desired time is
 * reached, the semaphore is written with the release value, the notifier is
 * filled with status and timestamp, and optionally an event is sent to all the
 * client waiting on it.
 *
 *   notifierGPUVA
 *     This parameter specifies the GPU VA of the notifier to receive the status
 *     for this particular release.
 *
 *   semaphoreGPUVA
 *     This parameter specifies the GPU VA of the semaphore to release.
 *
 *   waitTimestamp
 *     This parameter specifies the timestamp at which to release the semaphore.
 *
 *   releaseValue
 *     This parameter specifies the semaphore value to release.
 *
 *   releaseFlags
 *     This parameter specifies the flags:
 *       _NOTIFY wake client or not.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ADDRESS
 *   LW_ERR_ILWALID_EVENT
 *   LW_ERR_ILWALID_STATE
 *
 */
#define LW9074_CTRL_CMD_RELEASE (0x90740103) /* finn: Evaluated from "(FINN_GF100_TIMED_SEMAPHORE_SW_SEM_INTERFACE_ID << 8) | LW9074_CTRL_CMD_RELEASE_PARAMS_MESSAGE_ID" */

#define LW9074_CTRL_CMD_RELEASE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW9074_CTRL_CMD_RELEASE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 notifierGPUVA, 8);
    LW_DECLARE_ALIGNED(LwU64 semaphoreGPUVA, 8);
    LW_DECLARE_ALIGNED(LwU64 waitTimestamp, 8);
    LwU32 releaseValue;
    LwU32 releaseFlags;
} LW9074_CTRL_CMD_RELEASE_PARAMS;

#define LW9074_CTRL_CMD_RELEASE_FLAGS
#define LW9074_CTRL_CMD_RELEASE_FLAGS_NOTIFY                                 1:0        
#define LW9074_CTRL_CMD_RELEASE_FLAGS_NOTIFY_WRITE_ONLY        (0x00000000)
#define LW9074_CTRL_CMD_RELEASE_FLAGS_NOTIFY_WRITE_THEN_AWAKEN (0x00000001)

/* _ctrl9074.h_ */

