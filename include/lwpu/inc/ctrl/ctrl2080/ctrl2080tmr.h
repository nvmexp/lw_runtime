/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080tmr.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_TIMER related control commands and parameters */

/*
 * LW2080_CTRL_CMD_TIMER_SCHEDULE
 *
 * This command schedules a GPU timer event to fire at the specified time interval.
 * Can be called without API & GPU locks if LWOS54_FLAGS_IRQL_RAISED and
 * LWOS54_FLAGS_LOCK_BYPASS are set in LWOS54_PARAMETERS.flags
 *
 *   time_nsec
 *     This parameter specifies the time in nanoseconds at which the GPU timer
 *     event is to fire.
 *   flags
 *     This parameter determines the interpretation of the value specified by
 *     the time_nsec parameter:
 *       LW2080_CTRL_TIMER_SCHEDULE_FLAGS_TIME_ABS
 *         This flag indicates that time_nsec is in absolute time.
 *       LW2080_CTRL_TIMER_SCHEDULE_FLAGS_TIME_REL
 *         This flag indicates that time_nsec is in relative time.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_TIMER_SCHEDULE (0x20800401) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_TIMER_INTERFACE_ID << 8) | LW2080_CTRL_CMD_TIMER_SCHEDULE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_TIMER_SCHEDULE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_CMD_TIMER_SCHEDULE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 time_nsec, 8);
    LwU32 flags;
} LW2080_CTRL_CMD_TIMER_SCHEDULE_PARAMS;

/* valid flag values */
#define LW2080_CTRL_TIMER_SCHEDULE_FLAGS_TIME               0:0
#define LW2080_CTRL_TIMER_SCHEDULE_FLAGS_TIME_ABS (0x00000000)
#define LW2080_CTRL_TIMER_SCHEDULE_FLAGS_TIME_REL (0x00000001)

/*
 * LW2080_CTRL_CMD_TIMER_CANCEL
 *
 * This command cancels any pending timer events initiated with the
 * LW2080_CTRL_CMD_TIMER_SCHEDULE command.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_TIMER_CANCEL              (0x20800402) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_TIMER_INTERFACE_ID << 8) | 0x2" */

/* 
 * LW2080_CTRL_CMD_TIMER_GET_TIME 
 * 
 * This command returns the current GPU timer value.  The current time is 
 * expressed in elapsed nanoseconds since 00:00 GMT, January 1, 1970 
 * (zero hour) with a resolution of 32 nanoseconds. 
 *
 * Can be called without API & GPU locks if LWOS54_FLAGS_IRQL_RAISED and
 * LWOS54_FLAGS_LOCK_BYPASS are set in LWOS54_PARAMETERS.flags
 * 
 * Possible status values returned are: 
 *   LW_OK 
 *   LW_ERR_ILWALID_ARGUMENT 
 */
#define LW2080_CTRL_CMD_TIMER_GET_TIME            (0x20800403) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_TIMER_INTERFACE_ID << 8) | LW2080_CTRL_TIMER_GET_TIME_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_TIMER_GET_TIME_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_TIMER_GET_TIME_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 time_nsec, 8);
} LW2080_CTRL_TIMER_GET_TIME_PARAMS;

/*
 * LW2080_CTRL_CMD_TIMER_GET_REGISTER_OFFSET
 *
 * The command returns the offset of the timer registers, so that clients may
 * map them directly. 
 *
 * Possible status values returned are: 
 *   LW_OK 
 */

#define LW2080_CTRL_CMD_TIMER_GET_REGISTER_OFFSET (0x20800404) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_TIMER_INTERFACE_ID << 8) | LW2080_CTRL_TIMER_GET_REGISTER_OFFSET_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_TIMER_GET_REGISTER_OFFSET_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_TIMER_GET_REGISTER_OFFSET_PARAMS {
    LwU32 tmr_offset;
} LW2080_CTRL_TIMER_GET_REGISTER_OFFSET_PARAMS;

/*
 * LW2080_CTRL_TIMER_GPU_CPU_TIME_SAMPLE
 *
 * This structure describes the information obtained with
 * LW2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO command.
 *
 *   gpuTime
 *     GPU time is the value of GPU global timer (PTIMER) with a resolution
 *     of 32 nano seconds.
 *   cpuTime
 *     CPU time. Resolution of the cpu time depends on its source. Refer to
 *     LW2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_*  for more information.

 */
typedef struct LW2080_CTRL_TIMER_GPU_CPU_TIME_SAMPLE {
    LW_DECLARE_ALIGNED(LwU64 cpuTime, 8);
    LW_DECLARE_ALIGNED(LwU64 gpuTime, 8);
} LW2080_CTRL_TIMER_GPU_CPU_TIME_SAMPLE;


/*
 * LW2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO
 *
 * This command returns correlation information between GPU time and CPU time
 * for a given CPU clock type.
 *
 *   cpuClkId
 *     This parameter specifies the source of the CPU clock. Legal values for
 *     this parameter include:
 *       LW2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_OSTIME
 *         This clock id will provide real time in microseconds since
 *         00:00:00 UTC on January 1, 1970.
 *       LW2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_PLATFORM_API
 *         This clock id will provide time stamp that is constant-rate, high
 *         precision using platform API that is also available in the user mode.
 *       LW2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_TSC
 *         This clock id will provide time stamp using CPU's time stamp counter.
 *
 *   sampleCount
 *     This field specifies the number of clock samples to be taken.
 *     This value cannot exceed LW2080_CTRL_TIMER_GPU_CPU_TIME_MAX_SAMPLES.
 *
 *   samples
 *     This field returns an array of requested samples. Refer to
 *     LW2080_CTRL_TIMER_GPU_CPU_TIME_SAMPLE to get details about each entry
 *     in the array.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO (0x20800406) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_TIMER_INTERFACE_ID << 8) | LW2080_CTRL_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_TIMER_GPU_CPU_TIME_MAX_SAMPLES              16

#define LW2080_CTRL_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO_PARAMS {
    LwU8 cpuClkId;
    LwU8 sampleCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_TIMER_GPU_CPU_TIME_SAMPLE samples[LW2080_CTRL_TIMER_GPU_CPU_TIME_MAX_SAMPLES], 8);
} LW2080_CTRL_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO_PARAMS;

/* Legal cpuClkId values */
#define LW2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_OSTIME       (0x00000001)
#define LW2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_TSC          (0x00000002)
#define LW2080_TIMER_GPU_CPU_TIME_CPU_CLK_ID_PLATFORM_API (0x00000003)
/*!
 * LW2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ
 *
 * This command changes the frequency at which Graphics Engine time stamp is
 * updated. Frequency can either be set to max or restored to default.
 * Clients can independently use this call to increase the timer frequency 
 * as a global reference count is maintained for requests to Max frequency.
 * Client is assured that the system stays in this state till the requested
 * client releases the state or is killed. Timer frequency will automatically
 * be restored to default when there is no pending request to increase.
 *
 * Note that relwrsive requests for the same state from the same client
 * are considered invalid.
 *
 * bSetMaxFreq
 *      Set to LW_TRUE if GR tick frequency needs to be set to Max.
 *
 * See @ref LW2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS for
 * documentation of parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_OPERATION
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ            (0x20800407) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_TIMER_INTERFACE_ID << 8) | LW2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS_MESSAGE_ID" */

/*!
 * This struct contains bSetMaxFreq flag.
 */
#define LW2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS {
    LwBool bSetMaxFreq;
} LW2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS;

/* _ctrl2080tmr_h_ */
