/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000lwd.finn
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
/* LW01_ROOT (client) lwd control commands and parameters */

/*
 * LW0080_CTRL_LWD_DUMP_COMPONENT
 *
 * The following dump components are used to describe legal ranges in
 * commands below:
 *
 *   LW0080_CTRL_CMD_LWD_DUMP_COMPONENT_SYS
 *     This is the system dump component.
 *   LW0080_CTRL_CMD_LWD_DUMP_COMPONENT_LWLOG
 *     This is the lwlog dump component.
 *   LW0080_CTRL_CMD_LWD_DUMP_COMPONENT_RESERVED
 *     This component is reserved.
 *
 * See lwdump.h for more information on dump component values.
 */
#define LW0000_CTRL_LWD_DUMP_COMPONENT_SYS      (0x400)
#define LW0000_CTRL_LWD_DUMP_COMPONENT_LWLOG    (0x800)
#define LW0000_CTRL_LWD_DUMP_COMPONENT_RESERVED (0xB00)

/*
 * LW0000_CTRL_CMD_LWD_GET_DUMP_SIZE
 *
 * This command gets the expected dump size of a particular system
 * dump component.  Note that events that occur between this command
 * and a later LW0000_CTRL_CMD_LWD_GET_DUMP command could alter the size of
 * the buffer required.
 *
 *   component
 *     This parameter specifies the system dump component for which the
 *     dump size is desired.  Legal values for this parameter must
 *     be greater than or equal to LW0000_CTRL_LWD_DUMP_COMPONENT_SYS and
 *     less than LW0000_CTRL_LWD_GET_DUMP_COMPONENT_LWLOG.
 *   size
 *     This parameter returns the expected size in bytes.  The maximum
 *     value of this call is LW0000_CTRL_LWD_MAX_DUMP_SIZE.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT if components are invalid.
 */

#define LW0000_CTRL_CMD_LWD_GET_DUMP_SIZE       (0x601) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_LWD_GET_DUMP_SIZE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_LWD_GET_DUMP_SIZE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_LWD_GET_DUMP_SIZE_PARAMS {
    LwU32 component;
    LwU32 size;
} LW0000_CTRL_LWD_GET_DUMP_SIZE_PARAMS;

/* Max size that a GET_DUMP_SIZE_PARAMS call can return */
#define LW0000_CTRL_LWD_MAX_DUMP_SIZE (1000000)

/*
 * LW0000_CTRL_CMD_LWD_GET_DUMP
 *
 * This command gets a dump of a particular system dump component. If triggers
 * is non-zero, the command waits for the trigger to occur before it returns.
 *
 *   pBuffer
 *     This parameter points to the buffer for the data.
 *   component
 *     This parameter specifies the system dump component for which the
 *     dump is to be retrieved.  Legal values for this parameter must
 *     be greater than or equal to LW0000_CTRL_LWD_DUMP_COMPONENT_SYS and
 *     less than LW0000_CTRL_LWD_GET_DUMP_COMPONENT_LWLOG.
 *   size
 *     On entry, this parameter specifies the maximum length for
 *     the returned data. On exit, it specifies the number of bytes
 *     returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_ERROR_ILWALID_ARGUMENT if components are invalid.
 *   LWOS_ERROR_ILWALID_ADDRESS if pBuffer is invalid
 *   LWOS_ERROR_ILWALID_???? if the buffer was too small
 */
#define LW0000_CTRL_CMD_LWD_GET_DUMP  (0x602) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_LWD_GET_DUMP_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_LWD_GET_DUMP_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_LWD_GET_DUMP_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pBuffer, 8);
    LwU32 component;
    LwU32 size;
} LW0000_CTRL_LWD_GET_DUMP_PARAMS;

/*
 * LW0000_CTRL_CMD_LWD_GET_TIMESTAMP
 *
 * This command returns the current value of the timestamp used
 * by the RM in LwDebug dumps. It is provided to keep the RM and LwDebug
 * clients on the same time base.
 *
 *   cpuClkId
 *     See also LW2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO
 *     This parameter specifies the source of the CPU clock. Legal values for
 *     this parameter include:
 *       LW0000_LWD_CPU_TIME_CLK_ID_DEFAULT and LW0000_LWD_CPU_TIME_CLK_ID_OSTIME
 *         This clock id will provide real time in microseconds since 00:00:00 UTC on January 1, 1970.
 *         It is callwlated as follows:
 *          (seconds * 1000000) + uSeconds
 *       LW0000_LWD_CPU_TIME_CLK_ID_PLATFORM_API
 *         This clock id will provide time stamp that is constant-rate, high
 *         precision using platform API that is also available in the user mode.
 *       LW0000_LWD_CPU_TIME_CLK_ID_TSC
 *         This clock id will provide time stamp using CPU's time stamp counter.
 *
 *   timestamp
 *     Retrieved timestamp
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_LWD_CPU_TIME_CLK_ID_DEFAULT      (0x00000000)
#define LW0000_LWD_CPU_TIME_CLK_ID_OSTIME       (0x00000001)
#define LW0000_LWD_CPU_TIME_CLK_ID_TSC          (0x00000002)
#define LW0000_LWD_CPU_TIME_CLK_ID_PLATFORM_API (0x00000003)

#define LW0000_CTRL_CMD_LWD_GET_TIMESTAMP       (0x603) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_LWD_GET_TIMESTAMP_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_LWD_GET_TIMESTAMP_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0000_CTRL_LWD_GET_TIMESTAMP_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 timestamp, 8);
    LwU8 cpuClkId;
} LW0000_CTRL_LWD_GET_TIMESTAMP_PARAMS;

/*
 * LW0000_CTRL_CMD_LWD_GET_LWLOG_INFO
 *
 * This command gets the current state of the LwLog buffer system.
 *
 *   component (in)
 *     This parameter specifies the system dump component for which the
 *     LwLog info is desired.  Legal values for this parameter must
 *     be greater than or equal to LW0000_CTRL_LWD_DUMP_COMPONENT_LWLOG and
 *     less than LW0000_CTRL_LWD_DUMP_COMPONENT_RESERVED.
 *   version (out)
 *     This parameter returns the version of the Lwlog subsystem.
 *   runtimeSizes (out)
 *     This parameter returns the array of sizes for all supported printf
 *     specifiers.  This information is necessary to know how many bytes
 *     to decode when given a certain specifier (such as %d).
 *     The following describes the contents of each array entry:
 *       LW0000_CTRL_LWD_RUNTIME_SIZE_UNUSED
 *         This array entry has special meaning and is unused in the
 *         runtimeSizes array.
 *       LW0000_CTRL_LWD_RUNTIME_SIZE_INT
 *         This array entry returns the size of integer types for use in
 *         interpreting the %d, %u, %x, %X, %i, %o specifiers.
 *       LW0000_CTRL_LWD_RUNTIME_SIZE_LONG_LONG
 *         This array entry returns the size of long long integer types for
 *         using in interpreting the %lld, %llu, %llx, %llX, %lli, %llo
 *         specifiers.
 *       LW0000_CTRL_LWD_RUNTIME_SIZE_STRING
 *         This array entry returns zero as strings are not allowed.
 *       LW0000_CTRL_LWD_RUNTIME_SIZE_PTR
 *         This array entry returns the size of the pointer type for use
 *         in interpreting the %p specifier.
 *       LW0000_CTRL_LWD_RUNTIME_SIZE_CHAR
 *         This array entry returns the size of the char type for use in
 *         intpreting the %c specifier.
 *       LW0000_CTRL_LWD_RUNTIME_SIZE_FLOAT
 *         This array entry returns the size of the float types for use in
 *         in interpreting the %f, %g, %e, %F, %G, %E specifiers.
 *     All remaining entries are reserved and return 0.
 *   printFlags (out)
 *     This parameter returns the flags of the LwLog system.
 *       LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_BUFFER_FLAGS
 *         See LW0000_CTRL_CMD_LWD_GET_LWLOG_BUF_INFO for more details.
 *       LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_BUFFER_SIZE
 *         This field returns the buffer size in KBytes.  A value of zero
 *         is returned when logging is disabled.
 *       LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_TIMESTAMP
 *         This field returns the format of the timestamp.  Legal values
 *         for this parameter include:
 *           LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_TIMESTAMP_NONE
 *             This value indicates no timestamp.
 *           LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_TIMESTAMP_32BIT
 *             This value indicates a 32-bit timestamp.
 *           LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_TIMESTAMP_64BIT
 *             This value indicates a 64-bit timestamp.
 *           LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_TIMESTAMP_32BIT_DIFF
 *             This value indicates a 32-bit differential timestamp.
 *       LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_RESERVED
 *          This field is reserved.
 *       LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_RUNTIME_LEVEL
 *          This field returns the lowest debug level for which logging
 *          is enabled by default.
 *       LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_INIT
 *          This field indicates if logging for the specified component has
 *          been initialized. Legal values for this parameter include:
 *            LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_INIT_NO
 *              This value indicates LwLog is uninitialized.
 *            LW0000_CTRL_LWD_LWLOG_PRINT_FLAGS_INIT_YES
 *              This value indicates LwLog has been initialized.
 *   signature (out)
 *     This parameter is the signature of the database
 *     required to decode these logs, autogenerated at buildtime.
 *   bufferTags (out)
 *     This parameter identifies the buffer tag used during allocation
 *     or a value of '0' if buffer is unallocated for each possible
 *     buffer.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT if components are invalid.
 */
#define LW0000_CTRL_CMD_LWD_GET_LWLOG_INFO (0x604) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_LWD_GET_LWLOG_INFO_PARAMS_MESSAGE_ID" */

/* maximum size of the runtimeSizes array */
#define LW0000_CTRL_LWD_MAX_RUNTIME_SIZES  (16)

/* size of signature parameter */
#define LW0000_CTRL_LWD_SIGNATURE_SIZE     (4)

/* Maximum number of buffers */
#define LW0000_CTRL_LWD_MAX_BUFFERS        (256)

#define LW0000_CTRL_LWD_GET_LWLOG_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0000_CTRL_LWD_GET_LWLOG_INFO_PARAMS {
    LwU32 component;
    LwU32 version;
    LwU8  runtimeSizes[LW0000_CTRL_LWD_MAX_RUNTIME_SIZES];
    LwU32 printFlags;
    LwU32 signature[LW0000_CTRL_LWD_SIGNATURE_SIZE];
    LwU32 bufferTags[LW0000_CTRL_LWD_MAX_BUFFERS];
} LW0000_CTRL_LWD_GET_LWLOG_INFO_PARAMS;

/* runtimeSize array indices */
#define LW0000_CTRL_LWD_RUNTIME_SIZE_UNUSED                       (0)
#define LW0000_CTRL_LWD_RUNTIME_SIZE_INT                          (1)
#define LW0000_CTRL_LWD_RUNTIME_SIZE_LONG_LONG                    (2)
#define LW0000_CTRL_LWD_RUNTIME_SIZE_STRING                       (3)
#define LW0000_CTRL_LWD_RUNTIME_SIZE_PTR                          (4)
#define LW0000_CTRL_LWD_RUNTIME_SIZE_CHAR                         (5)
#define LW0000_CTRL_LWD_RUNTIME_SIZE_FLOAT                        (6)

/* printFlags fields and values */
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_BUFFER_INFO          7:0
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_BUFFER_SIZE          23:8
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_BUFFER_SIZE_DISABLE (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_BUFFER_SIZE_DEFAULT (0x00000004)
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_RUNTIME_LEVEL        28:25
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_TIMESTAMP            30:29
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_TIMESTAMP_NONE      (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_TIMESTAMP_32        (0x00000001)
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_TIMESTAMP_64        (0x00000002)
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_TIMESTAMP_32_DIFF   (0x00000003)
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_INITED               31:31
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_INITED_NO           (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_INFO_PRINTFLAGS_INITED_YES          (0x00000001)

/*
 * LW0000_CTRL_CMD_LWD_GET_LWLOG_BUFFER_INFO
 *
 * This command gets the current state of a specific buffer in the LwLog
 * buffer system.
 *
 *   component (in)
 *     This parameter specifies the system dump component for which the
 *     LwLog info is desired.  Legal values for this parameter must
 *     be greater than or equal to LW0000_CTRL_LWD_DUMP_COMPONENT_LWLOG and
 *     less than LW0000_CTRL_LWD_DUMP_COMPONENT_RESERVED.
 *   buffer (in/out)
 *     This parameter specifies the buffer number from which to retrieve the
 *     buffer information. Valid values are 0 to (LW0000_CTRL_LWD_MAX_BUFFERS - 1).
 *     If the buffer is specified using the 'tag' parameter, the buffer
 *     number is returned through this one.
 *   tag (in/out)
 *     If this parameter is non-zero, it will be used to specify the buffer,
 *     instead of 'buffer' parameter. It returns the tag of the specified buffer
 *   size (out)
 *     This parameter returns the size of the specified buffer.
 *   flags (in/out)
 *     On input, this parameter sets the following behavior:
 *       LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_PAUSE
 *         This flag controls if the lwlog system should pause output
 *         to this buffer.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_PAUSE_YES
 *             The buffer should be paused until another command 
 *             unpauses this buffer.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_PAUSE_NO
 *             The buffer should not be paused.
 *     On output, this parameter returns the flags of a specified buffer:
 *       LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_DISABLED
 *         This flag indicates if logging to the specified buffer is
 *         disabled or not.
 *       LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_TYPE
 *         This flag indicates the buffer logging type:
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_TYPE_RING
 *             This type value indicates logging to the buffer wraps.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_TYPE_NOWRAP
 *             This type value indicates logging to the buffer does not wrap.
 *       LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_EXPANDABLE
 *         This flag indicates if the buffer size is expandable.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_EXPANDABLE_NO
 *             The buffer is not expandable.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_EXPANDABLE_YES
 *             The buffer is expandable.
 *       LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_NON_PAGED
 *         This flag indicates if the buffer oclwpies non-paged or pageable
 *         memory.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_NON_PAGED_NO
 *             The buffer is in pageable memory.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_NON_PAGES_YES
 *             The buffer is in non-paged memory.
 *       LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING
 *         This flag indicates the locking mode for the specified buffer.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING_NONE
 *             This locking value indicates that no locking is performed.  This
 *             locking mode is typically used for inherently single-threaded
 *             buffers.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING_STATE
 *             This locking value indicates that the buffer is locked only
 *             during state changes and that memory copying is unlocked.  This
 *             mode should not be used tiny buffers that overflow every write
 *             or two.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING_FULL
 *             This locking value indicates the buffer is locked for the full
 *             duration of the write.
 *       LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_OCA
 *         This flag indicates if the buffer is stored in OCA dumps.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_OCA_NO
 *             The buffer is not included in OCA dumps.
 *           LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_OCA_YES
 *             The buffer is included in OCA dumps.
 *   pos (out)
 *      This parameter is the current position of the tracker/cursor in the
 *      buffer.
 *   overflow (out)
 *     This parameter is the number of times the buffer has overflowed.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT if components are invalid.
 */

#define LW0000_CTRL_CMD_LWD_GET_LWLOG_BUFFER_INFO                 (0x605) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_LWD_GET_LWLOG_BUFFER_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_LWD_GET_LWLOG_BUFFER_INFO_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0000_CTRL_LWD_GET_LWLOG_BUFFER_INFO_PARAMS {
    LwU32 component;
    LwU32 buffer;
    LwU32 tag;
    LwU32 size;
    LwU32 flags;
    LwU32 pos;
    LwU32 overflow;
} LW0000_CTRL_LWD_GET_LWLOG_BUFFER_INFO_PARAMS;

/* flags fields and values */
/* input */
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_PAUSE              0:0
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_PAUSE_NO       (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_PAUSE_YES      (0x00000001)

/* output */
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_DISABLED           0:0
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_DISABLED_NO    (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_DISABLED_YES   (0x00000001)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_TYPE               1:1
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_TYPE_RING      (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_TYPE_NOWRAP    (0x00000001)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_EXPANDABLE         2:2
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_EXPANDABLE_NO  (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_EXPANDABLE_YES (0x00000001)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_NONPAGED           3:3
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_NONPAGED_NO    (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_NONPAGED_YES   (0x00000001)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING            5:4
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING_NONE   (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING_STATE  (0x00000001)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_LOCKING_FULL   (0x00000002)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_OCA                6:6
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_OCA_NO         (0x00000000)
#define LW0000_CTRL_LWD_LWLOG_BUFFER_INFO_FLAGS_OCA_YES        (0x00000001)

/*
 * LW0000_CTRL_CMD_LWD_GET_LWLOG
 *
 * This command retrieves the specified dump block from the specified
 * LwLog buffer.  To retrieve the entire buffer, the caller should start
 * with blockNum set to 0 and continue issuing calls with an incremented
 * blockNum until the returned size value is less than
 * LW0000_CTRL_LWD_LWLOG_MAX_BLOCK_SIZE.
 *
 *   component (in)
 *     This parameter specifies the system dump component for which the LwLog
 *     dump operation is to be directed.  Legal values for this parameter
 *     must be greater than or equal to LW0000_CTRL_LWD_DUMP_COMPONENT_LWLOG
 *     and less than LW0000_CTRL_LWD_DUMP_COMPONENT_RESERVED.
 *   buffer (in)
 *     This parameter specifies the LwLog buffer to dump.
 *   blockNum (in)
 *     This parameter specifies the block number for which data is to be
 *     dumped.
 *   size (in/out)
 *     On entry, this parameter specifies the maximum length in bytes for
 *     the returned data (should be set to LW0000_CTRL_LWLOG_MAX_BLOCK_SIZE).
 *     On exit, it specifies the number of bytes returned.
 *   data (out)
 *     This parameter returns the data for the specified block.  The size
 *     patameter values indicates the number of valid bytes returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_ERROR_ILWALID_ARGUMENT if components are invalid.
 */
#define LW0000_CTRL_CMD_LWD_GET_LWLOG                          (0x606) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_LWD_GET_LWLOG_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_LWLOG_MAX_BLOCK_SIZE                       (4000)

#define LW0000_CTRL_LWD_GET_LWLOG_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0000_CTRL_LWD_GET_LWLOG_PARAMS {
    LwU32 component;
    LwU32 buffer;
    LwU32 blockNum;
    LwU32 size;
    LwU8  data[LW0000_CTRL_LWLOG_MAX_BLOCK_SIZE];
} LW0000_CTRL_LWD_GET_LWLOG_PARAMS;

/*
 * LW0000_CTRL_CMD_LWD_GET_RCERR_RPT
 *
 * This command returns block of registers that were recorded at the time
 * of an RC error for the current process.
 *
 *   reqIdx:
 *      [IN] the index of the report being requested.
 *      index rolls over to 0.
 *      if the requested index is not in the cirlwlar buffer, then no data is
 *      transferred & either LW_ERR_ILWALID_INDEX (indicating the specified
 *      index is not in the table) is returned.
 *
 *   rptIdx:
 *      [OUT] the index of the report being returned.
 *      if the requested index is not in the cirlwlar buffer, then the value is
 *      undefined, no data istransferred & LW_ERR_ILWALID_INDEX is returned.
 *      if the the specified index is present, but does not meet the requested
 *      criteria (refer to the owner & processId fields). the rptIdx will be
 *      set to a value that does not match the reqIdx, and no data will be
 *      transferred. LW_ERR_INSUFFICIENT_PERMISSIONS is still returned.
 *
 *   gpuTag:
 *      [OUT] id of the GPU whose data was collected.
 *
 *   rptTimeInNs:
 *      [OUT] the timestamp for when the report was created.
 *
 *   startIdx:
 *      [OUT] the index of the oldest start record for the first report that
 *      matches the specified criteria (refer to the owner & processId
 *      fields). If no records match the specified criteria, this value is
 *      undefined, the failure code LW_ERR_MISSING_TABLE_ENTRY will
 *      be returned, and no data will be transferred.
 *
 *   endIdx:
 *      [OUT] the index of the newest end record for the most recent report that
 *      matches the specified criteria (refer to the owner & processId
 *      fields). If no records match the specified criteria, this value is
 *      undefined, the failure code LW_ERR_MISSING_TABLE_ENTRY will
 *      be returned, and no data will be transferred.
 *
 *   rptType:
 *      [OUT] indicator of what data is in the report.
 *
 *  flags
 *      [OUT] a set odf flags indicating attributes of the record
 *          LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_POS_FIRST --    indicates this is the first record of a report.
 *          LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_POS_LAST --     indicates this is the last record of the report.
 *          LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_RANGE_VALID --  indicates this is the response contains a valid
*               index range.
 *              Note, this may be set when an error is returned indicating a valid range was found, but event of
 *              the requested index was not.
 *          LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_DATA_VALID --   indicates this is the response contains valid data.
 *
 *   rptCount:
 *      [OUT] number of entries returned in report.
 *
 *   owner:
 *      [IN] Entries are only returned if they have the same owner as the specified owner or the specified 
 *      owner Id is LW0000_CTRL_CMD_LWD_RPT_ANY_OWNER_ID.
 *      if the requested index is not owned by the specified owner, the rptIdx
 *      will be set to a value that does not match the reqIdx, and no data will
 *      be transferred.  LW_ERR_INSUFFICIENT_PERMISSIONS is returned.
 *
 *   processId:
 *      [IN] Deprecated
 *   report:
 *      [OUT] array of rptCount enum/value pair entries containing the data from the report.
 *      entries beyond rptCount are undefined.
 *
 *
 * Possible status values returned are:
 *  LW_OK -- we found & transferred the requested record.
 *  LW_ERR_MISSING_TABLE_ENTRY -- we don't find any records that meet the criteria.
 *  LW_ERR_ILWALID_INDEX -- the requested index was not found in the buffer.
 *  LW_ERR_INSUFFICIENT_PERMISSIONS -- the requested record was found, but it did not meet the criteria.
 *  LW_ERR_BUSY_RETRY -- We could not access the cirlwlar buffer.
 *
 */

#define LW0000_CTRL_CMD_LWD_GET_RCERR_RPT                                          (0x607) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_CMD_LWD_GET_RCERR_RPT_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CMD_LWD_RCERR_RPT_MAX_ENTRIES                                  200

 // report types
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_TYPE_TEST                                    0
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_TYPE_GRSTATUS                                1
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_TYPE_GPCSTATUS                               2

// pseudo register enums                                                                                         attribute content
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_EMPTY                                    0x00000000
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_OVERFLOWED                               0x00000001                  // number of missed entries.
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_MAX_PSEDO_REG                            0x0000000f

// register enums
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PFB_NISO_DEBUG                           0x00000010                  // debug config
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PFB_NISO_PRI_DEBUG1                      0x00000011                  // debug config
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PFIFO_ENGINE_STATUS                      0x00000012
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY0                         0x00000020
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY1                         0x00000021
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY2                         0x00000022
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY3                         0x00000023
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY4                         0x00000024
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY5                         0x00000025
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY6                         0x00000026
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY7                         0x00000027
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY8                         0x00000028
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY9                         0x00000029
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY10                        0x0000002a
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY11                        0x0000002b
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY12                        0x0000002c
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ACTIVITY13                        0x0000002d
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_BE_CROP_STATUS1               0x00000030                  // BE #
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_BE_CROP_STATUS2               0x00000031                  // BE #
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_BE_CROP_STATUS3               0x00000032                  // BE #
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_BE_CROP_STATUS4               0x00000033                  // BE #
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_BE_ZROP_STATUS                0x00000034                  // BE #
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_BE_ZROP_STATUS2               0x00000035                  // BE #
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_STATUS                            0x00000050
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_STATUS1                           0x00000051
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_STATUS2                           0x00000052
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_EXCEPTION                         0x00000053
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_FECS_INTR                         0x00000054
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_GRFIFO_CONTROL                    0x00000055
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_GRFIFO_STATUS                     0x00000056
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_INTR                              0x00000057
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_DS_MPIPE_STATUS               0x00000058
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_GO_IDLE_CHECK              0x00000059
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_GO_IDLE_ON_STATUS          0x0000005a
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_GO_IDLE_TIMEOUT            0x0000005b
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FECS_CTXSW_STATUS_1           0x0000005c
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FECS_CTXSW_STATUS_FE_0        0x0000005e
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FECS_LWRRENT_CTX              0x0000005f
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FECS_HOST_INT_STATUS          0x00000060
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FECS_NEW_CTX                  0x00000061
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY0      0x00000100                  // gpcId
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY1      0x00000101                  // gpcId
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY2      0x00000102                  // gpcId
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY3      0x00000103                  // gpcId
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_GPCCS_GPC_ACTIVITY4      0x00000104                  // gpcId
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_ROP0_ZROP_STATUS         0x00000105                  // ROP_REG_ATTR(gpcId, ropId)
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_ROP0_ZROP_STATUS1        0x00000106                  // ROP_REG_ATTR(gpcId, ropId)
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_ROP0_CROP_STATUS1        0x00000107                  // ROP_REG_ATTR(gpcId, ropId)
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_ROP0_CROP_STATUS2        0x00000108                  // ROP_REG_ATTR(gpcId, ropId)
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_ROP0_CROP_STATUS3        0x00000109                  // ROP_REG_ATTR(gpcId, ropId)
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_ROP0_RRH_STATUS          0x0000010a                  // ROP_REG_ATTR(gpcId, ropId)

#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_TPC0_TPCCS_TPC_ACTIVITY0 0x00000200
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_TPC0_MPC_STATUS          0x00000201
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC_TPC_SM_DFD_ID             0x00000202                  // TPC_REG_ATTR(gpcId, tpcId)

#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_TPC0_SM_DFD_DATA         0x00000204                  // TPC_REG_ATTR(gpcId, tpcId)
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC_TPC_LWM_SM_DFD_DATA       0x00000205                  // count
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_TPC0_MPC_WLU_STATUS      0x00000206
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_BE_BECS_BE_ACTIVITY0          0x00000207
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_CLASS_ERROR                       0x00000208
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_ENGINE_STATUS                     0x00000209
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_PDB_STATUS                    0x0000020a
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_PDB_STATUS_SCC_IN             0x0000020b
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_PDB_STATUS_TASK_CTRL          0x0000020c
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_PDB_STATUS_PDB                0x0000020d
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE__METHOD_STATE              0x0000020e
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_LWRRENT_METHOD             0x0000020f
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_LWRRENT_OBJECT_0           0x00000210
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_SKED_ACTIVITY                 0x00000211
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_EXCEPTION1                        0x00000212
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_EXCEPTION2                        0x00000213
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_TRAPPED_ADDR                      0x00000214
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_TRAPPED_DATA_LOW                  0x00000215
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_TRAPPED_DATA_HIGH                 0x00000216
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_INTR_EN                           0x00000217
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_EXCEPTION_EN                      0x00000218
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_EXCEPTION1_EN                     0x00000219
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_EXCEPTION2_EN                     0x0000021a
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_NONSTALL_INTR_EN                  0x0000021b
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_INTR_ROUTE                        0x0000021c
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_FECS_INTR_EN                      0x0000021d
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_METHOD_STATE               0x0000021e
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_NONSTALL_INTR                     0x0000021f
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_SEMAPHORE_STATE_A          0x00000220
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_SEMAPHORE_STATE_B          0x00000221
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_SEMAPHORE_STATE_C          0x00000222
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_SEMAPHORE_STATE_D          0x00000223
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_FE_SEMAPHORE_STATE_REPORT     0x00000224
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC_TPC_LWM_SM_DFD_BASE       0x00000225
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_GPCCS_GPC_EXCEPTION      0x00000226                  // gpcId
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_PGRAPH_PRI_GPC0_TPC0_TPCCS_TPC_EXCEPTION 0x00000227                  // TPC_REG_ATTR(gpcId, tpcId)

// Flags Definitions
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_POS_FIRST                              0x00000001                  // indicates this is the first record of a report.
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_POS_LAST                               0x00000002                  // indicates this is the last record of the report.
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_RANGE_VALID                            0x00000004                  // indicates this is the response contains a valid range
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_FLAGS_DATA_VALID                             0x00000008                  // indicates this is the response contains valid data


// Attribute Definitions
#define TPC_REG_ATTR(gpcId, tpcId)                                                  ((gpcId << 8) | (tpcId))
#define ROP_REG_ATTR(gpcId, ropId)                                                  ((gpcId << 8) | (ropId))

// Process Id Pseudo values
#define LW0000_CTRL_CMD_LWD_RCERR_RPT_ANY_PROCESS_ID                               0x00000000                  // get report for any process ID

#define LW0000_CTRL_CMD_LWD_RCERR_RPT_ANY_OWNER_ID                                 0xFFFFFFFF                  // get report for any owner ID


typedef struct LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_ENTRY {
    LwU32 tag;
    LwU32 value;
    LwU32 attribute;
} LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_ENTRY;

#define LW0000_CTRL_CMD_LWD_GET_RCERR_RPT_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0000_CTRL_CMD_LWD_GET_RCERR_RPT_PARAMS {
    LwU16                                   reqIdx;
    LwU16                                   rptIdx;
    LwU32                                   GPUTag;
    LwU32                                   rptTime;     // time in seconds since 1/1/1970
    LwU16                                   startIdx;
    LwU16                                   endIdx;
    LwU16                                   rptType;
    LwU32                                   flags;
    LwU16                                   rptCount;
    LwU32                                   owner;       // indicating whose reports to get
    LwU32                                   processId;       // deprecated field

    LW0000_CTRL_CMD_LWD_RCERR_RPT_REG_ENTRY report[LW0000_CTRL_CMD_LWD_RCERR_RPT_MAX_ENTRIES];
} LW0000_CTRL_CMD_LWD_GET_RCERR_RPT_PARAMS;

/*
 * LW0000_CTRL_CMD_LWD_GET_DPC_ISR_TS
 *
 * This command returns the time stamp information that are collected from
 * the exelwtion of various DPCs/ISRs. This time stamp information is for
 * debugging purposes only and would help with analyzing regressions and
 * latencies for DPC/ISR exelwtion times.
 *
 *   tsBufferSize
 *     This field specifies the size of the buffer that the caller allocates.
 *   tsBuffer
 *     THis field specifies a pointer in the callers address space to the
 *     buffer into which the timestamp info on DPC/ISR is to be returned.
 *     This buffer must at least be as big as tsBufferSize.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW0000_CTRL_CMD_LWD_GET_DPC_ISR_TS (0x608) /* finn: Evaluated from "(FINN_LW01_ROOT_LWD_INTERFACE_ID << 8) | LW0000_CTRL_LWD_GET_DPC_ISR_TS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_LWD_GET_DPC_ISR_TS_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW0000_CTRL_LWD_GET_DPC_ISR_TS_PARAMS {
    LwU32 tsBufferSize;
    LW_DECLARE_ALIGNED(LwP64 pTSBuffer, 8);
} LW0000_CTRL_LWD_GET_DPC_ISR_TS_PARAMS;

/* _ctrl0000lwd_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

