/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000diag.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0000/ctrl0000base.h"

#include "ctrl/ctrlxxxx.h"
/* LW01_ROOT (client) system control commands and parameters */

/*
 * LW0000_CTRL_CMD_DIAG_GET_PROFILE_DATABASE_SIZE
 *
 * This command returns the size in bytes of the profiler instrumentation
 * database. This is the first of two calls required to retrieve
 * the profiler database. The client can then allocate a profiler
 * array and retrieve the database into it.
 * 
 * The code instrumentation database is for performance profiling.
 * In order for the profiler database to exist, the code must be built
 * with the RM_PROFILER symbol defined. It is not necessary to build
 * the code with the DEBUG symbol defined. In fact, it is undesirable as
 * DEBUG may change performance characteristics, unless you're trying to
 * debug the profiler itself.
 * 
 * The client program and the driver are both built with a common header
 * file that contains the size of the database. However, it is possible
 * that the two could be built out of synchronization. So, a two-part
 * call is made to this command. One call retrieves the size
 * of the database and the other gets the actual database content.
 * 
 * This command will  return the actual size of the instrumentation database.
 * This is the number of bytes that the client program should allocate for the
 * return of the instrumentation database.
 * 
 *   sizeOfDatabase
 *       This field returns the size in bytes of the profiler database array
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW0000_CTRL_CMD_DIAG_GET_PROFILE_DATABASE_SIZE (0x401) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | 0x1" */

typedef struct LW0000_CTRL_DIAG_GET_PROFILE_DATABASE_SIZE_PARAMS {
    LwU32 sizeOfDatabase;    // size in bytes of database
} LW0000_CTRL_DIAG_GET_PROFILE_DATABASE_SIZE_PARAMS;

/*
 * LW0000_CTRL_CMD_DIAG_GET_PROFILE_DATABASE
 *
 * This command returns the content of the profiler instrumentation database.
 * database. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW0000_CTRL_CMD_DIAG_GET_PROFILE_DATABASE (0x402) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | 0x2" */

typedef struct LW0000_CTRL_DIAG_GET_PROFILE_DATABASE_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 databaseBuffer, 8);   // char buffer containing the database
} LW0000_CTRL_DIAG_GET_PROFILE_DATABASE_PARAMS;

/*
 * LW0000_CTRL_CMD_DIAG_ADD_INSTRUMENTATION_UNIT
 *
 * This command allows an RM client to ask RM to add an instrumentation unit
 * to its database.
 *
 *   category
 *     This field contains the instrumentation unit category.
 *   function
 *     This field contains the instrumentation unit function.
 *   action
 *     This field contains the instrumentation unit action.
 *   enterExit
 *     This field contains the instrumentation unit enter or exit function
 *     flag.
 *   identifier
 *     This field contains the instrumentation unit identifier.
 *   
 * Possible status values returned are:
 *   LW_OK
 */

#define LW0000_CTRL_CMD_DIAG_ADD_INSTRUMENTATION_UNIT (0x403) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | 0x3" */

typedef struct LW0000_CTRL_CMD_DIAG_ADD_INSTRUMENTATION_UNIT_PARAMS {
    LwU32 category;
    LwU32 function;
    LwU32 action;
    LwU32 enterExit;
    LwU32 identifier;
} LW0000_CTRL_CMD_DIAG_ADD_INSTRUMENTATION_UNIT_PARAMS;

/*
 * LW0000_CTRL_CMD_DIAG_GET_LOCK_METER_STATE
 *
 * This command returns the current lock meter logging state.
 *
 *   state
 *     This parameter returns the current lock meter logging state.
 *       LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_DISABLED
 *         This value indicates lock metering is disabled.
 *       LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_ENABLED
 *         This value indicates lock metering is enabled.
 *   count
 *     This parameter returns the total number of lock metering entries
 *     (LW0000_CTRL_DIAG_LOCK_METER_ENTRY) available.  This value will
 *     not exceed LW0000_CTRL_DIAG_LOCK_METER_MAX_ENTRIES.  When lock metering
 *     is enabled this parameter will return zero.
 *   missedCount
 *     This parameter returns the number of lock metering entries that had
 *     to be discarded due to a  full lock metering table.  This value will
 *     not exceed LW0000_CTRL_DIAG_LOCK_METER_MAX_TABLE_ENTRIES.  When lock
 *     metering is enabled this parameter will return zero.
 *   bCirlwlarBuffer
 *     This parameter returns type of buffer.
 *       TRUE
 *         Buffer is cirlwlar
 *       FALSE
 *         Buffer is sequential 
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0000_CTRL_CMD_DIAG_GET_LOCK_METER_STATE (0x480) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_PARAMS_MESSAGE_ID (0x80U)

typedef struct LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_PARAMS {
    LwU32  state;
    LwU32  count;
    LwU32  missedCount;
    LwBool bCirlwlarBuffer;
} LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_PARAMS;

/* valid lock metering state values */
#define LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_DISABLED (0x00000000)
#define LW0000_CTRL_DIAG_GET_LOCK_METER_STATE_ENABLED  (0x00000001)

/* maximum possible number of lock metering entries stored internally */
#define LW0000_CTRL_DIAG_LOCK_METER_MAX_TABLE_ENTRIES  (0x20000)

/*
 * LW0000_CTRL_CMD_DIAG_SET_LOCK_METER_STATE
 *
 * This command sets the current lock meter logging state.
 *
 *   state
 *     This parameter specifies the new state of the lock metering mechanism.
 *     Legal state values are:
 *       LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_DISABLE
 *         This value disables lock metering.
 *       LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_ENABLE
 *         This value enables lock metering.
 *       LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_RESET
 *         This value resets, or clears, all lock metering state.  Lock
 *         metering must be disabled prior to attempting a reset.
 *   bCirlwlarBuffer
 *     This parameter specifies type of buffer.
 *     Possible values are:
 *       TRUE
 *         For cirlwlar buffer.
 *       FALSE
 *         For sequential buffer.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW0000_CTRL_CMD_DIAG_SET_LOCK_METER_STATE      (0x481) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_PARAMS_MESSAGE_ID (0x81U)

typedef struct LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_PARAMS {
    LwU32  state;
    LwBool bCirlwlarBuffer;
} LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_PARAMS;

/* valid lock metering state values */
#define LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_DISABLE     (0x00000000)
#define LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_ENABLE      (0x00000001)
#define LW0000_CTRL_DIAG_SET_LOCK_METER_STATE_RESET       (0x00000002)

/*
 * LW0000_CTRL_DIAG_LOCK_METER_ENTRY
 *
 * This structure represents a single lock meter entry.
 *
 *   counter
 *     This field contains the number of nanonseconds elapsed since the
 *     the last system boot when the lock meter entry was generated.
 *   freq
 *     This field contains the CPU performance counter frequency in units
 *     of ticks per second.
 *   line
 *     This field contains the relevant line number.
 *   filename
 *     This field contains the relevant file name.
 *   tag
 *     This field contains a tag uniquely identifying the user of the metered
 *     lock operations.
 *   cpuNum
 *     This field contains the CPU number from which the metered operation
 *     was initiated.
 *   irql
 *     This field contains the IRQL at which the metered operation was
 *     initiated.
 *   data0
 *   data1
 *   data2
 *     These fields contain tag-specific data.
 */
#define LW0000_CTRL_DIAG_LOCK_METER_ENTRY_FILENAME_LENGTH (0xc)

typedef struct LW0000_CTRL_DIAG_LOCK_METER_ENTRY {
    LW_DECLARE_ALIGNED(LwU64 counter, 8);

    LwU32 line;
    LwU8  filename[LW0000_CTRL_DIAG_LOCK_METER_ENTRY_FILENAME_LENGTH];

    LwU16 tag;
    LwU8  cpuNum;
    LwU8  irql;

    LW_DECLARE_ALIGNED(LwU64 threadId, 8);

    LwU32 data0;
    LwU32 data1;
    LwU32 data2;
} LW0000_CTRL_DIAG_LOCK_METER_ENTRY;

/* valid lock meter entry tag values */
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ACQUIRE_SEMA        (0x00000001)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ACQUIRE_SEMA_FORCED (0x00000002)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ACQUIRE_SEMA_COND   (0x00000003)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_RELEASE_SEMA        (0x00000004)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ACQUIRE_API         (0x00000010)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_RELEASE_API         (0x00000011)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ACQUIRE_GPUS        (0x00000020)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_RELEASE_GPUS        (0x00000021)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_DATA                (0x00000100)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_RMCTRL              (0x00001000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_CFG_GET             (0x00002000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_CFG_SET             (0x00002001)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_CFG_GETEX           (0x00002002)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_CFG_SETEX           (0x00002003)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_VIDHEAP             (0x00003000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_MAPMEM              (0x00003001)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_UNMAPMEM            (0x00003002)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_MAPMEM_DMA          (0x00003003)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_UNMAPMEM_DMA        (0x00003004)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ALLOC               (0x00004000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ALLOC_MEM           (0x00004001)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_DUP_OBJECT          (0x00004010)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_CLIENT         (0x00005000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_DEVICE         (0x00005001)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_SUBDEVICE      (0x00005002)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_SUBDEVICE_DIAG (0x00005003)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_DISP           (0x00005004)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_DISP_CMN       (0x00005005)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_CHANNEL        (0x00005006)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_CHANNEL_MPEG   (0x00005007)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_CHANNEL_DISP   (0x00005008)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_MEMORY         (0x00005009)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_FBMEM          (0x0000500A)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_OBJECT         (0x0000500B)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_FREE_EVENT          (0x0000500C)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_IDLE_CHANNELS       (0x00006000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_BIND_CTXDMA         (0x00007000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ALLOC_CTXDMA        (0x00007001)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_ISR                 (0x0000F000)
#define LW0000_CTRL_DIAG_LOCK_METER_TAG_DPC                 (0x0000F00F)

/*
 * LW0000_CTRL_CMD_DIAG_GET_LOCK_METER_ENTRIES
 *
 * This command returns lock metering data in a fixed-sized array of entries.
 * Each request will return up LW0000_CTRL_CMD_DIAG_GET_LOCK_METER_MAX_ENTRIES
 * entries.
 *
 * It is up to the caller to repeat these requests to retrieve the total number
 * of entries reported by LW0000_CTRL_CMD_DIAG_GET_LOCK_METER_STATE.
 *
 *   entryCount
 *     This parameter returns the total number of valid entries returned
 *     in the entries array.  This value will not exceed
 *     LW0000_CTRL_DIAG_GET_LOCK_METER_ENTRIES_MAX but may be less.
 *     A value of zero indicates there are no more valid entries.
 *   entries
 *     This parameter contains the storage into which lock metering entry
 *     data is returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW0000_CTRL_CMD_DIAG_GET_LOCK_METER_ENTRIES         (0x485) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | LW0000_CTRL_DIAG_GET_LOCK_METER_ENTRIES_PARAMS_MESSAGE_ID" */

/* total number of entries returned */
#define LW0000_CTRL_DIAG_GET_LOCK_METER_ENTRIES_MAX         (0x40)

#define LW0000_CTRL_DIAG_GET_LOCK_METER_ENTRIES_PARAMS_MESSAGE_ID (0x85U)

typedef struct LW0000_CTRL_DIAG_GET_LOCK_METER_ENTRIES_PARAMS {
    LwU32 entryCount;
    LW_DECLARE_ALIGNED(LW0000_CTRL_DIAG_LOCK_METER_ENTRY entries[LW0000_CTRL_DIAG_GET_LOCK_METER_ENTRIES_MAX], 8);
} LW0000_CTRL_DIAG_GET_LOCK_METER_ENTRIES_PARAMS;

/*
 * LW0000_CTRL_CMD_DIAG_PROFILE_RPC
 *
 * This command returns the RPC runtime information, and
 * will only return valid when it is running inside VGX mode.
 *
 *   rpcProfileCmd:
 *      RPC profiler command issued by rpc profiler utility
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_DIAG_PROFILE_RPC (0x488) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | LW0000_CTRL_DIAG_PROFILE_RPC_PARAMS_MESSAGE_ID" */

typedef struct RPC_METER_ENTRY {
    LW_DECLARE_ALIGNED(LwU64 startTimeInNs, 8);
    LW_DECLARE_ALIGNED(LwU64 endTimeInNs, 8);
    LW_DECLARE_ALIGNED(LwU64 rpcDataTag, 8);
    LW_DECLARE_ALIGNED(LwU64 rpcExtraData, 8);
} RPC_METER_ENTRY;

#define LW0000_CTRL_DIAG_PROFILE_RPC_PARAMS_MESSAGE_ID (0x88U)

typedef struct LW0000_CTRL_DIAG_PROFILE_RPC_PARAMS {
    LwU32 rpcProfileCmd;
} LW0000_CTRL_DIAG_PROFILE_RPC_PARAMS;

#define LW0000_CTRL_PROFILE_RPC_CMD_DISABLE (0x00000000)
#define LW0000_CTRL_PROFILE_RPC_CMD_ENABLE  (0x00000001)
#define LW0000_CTRL_PROFILE_RPC_CMD_RESET   (0x00000002)

/*
 * LW0000_CTRL_CMD_DIAG_DUMP_RPC
 *
 * This command returns the RPC runtime information, which
 * will be logged by LW0000_CTRL_CMD_DIAG_PROFILE_RPC command
 * when running inside VGX mode.
 *
 * When issuing this command, the RPC profiler has to be disabled.
 *
 *   firstEntryOffset:
 *     [IN] offset for first entry.
 *
 *   outputEntryCout:
 *     [OUT] number of entries returned in rpcProfilerBuffer.
 *
 *   remainingEntryCount:
 *     [OUT] number of entries remaining. 
 *
 *   elapsedTimeInNs:
 *     [OUT] runtime for the RPC profiler tool. 
 *
 *   rpcProfilerBuffer:
 *     [OUT] buffer to store the RPC entries
 */

#define LW0000_CTRL_CMD_DIAG_DUMP_RPC       (0x489) /* finn: Evaluated from "(FINN_LW01_ROOT_DIAG_INTERFACE_ID << 8) | LW0000_CTRL_DIAG_DUMP_RPC_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_DIAG_RPC_MAX_ENTRIES    (100)

#define LW0000_CTRL_DIAG_DUMP_RPC_PARAMS_MESSAGE_ID (0x89U)

typedef struct LW0000_CTRL_DIAG_DUMP_RPC_PARAMS {
    LwU32 firstEntryOffset;
    LwU32 outputEntryCount;
    LwU32 remainingEntryCount;
    LW_DECLARE_ALIGNED(LwU64 elapsedTimeInNs, 8);
    LW_DECLARE_ALIGNED(RPC_METER_ENTRY rpcProfilerBuffer[LW0000_CTRL_DIAG_RPC_MAX_ENTRIES], 8);
} LW0000_CTRL_DIAG_DUMP_RPC_PARAMS;

/* _ctrl0000diag_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

