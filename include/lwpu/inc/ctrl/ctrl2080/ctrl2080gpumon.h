/*
 * SPDX-FileCopyrightText: Copyright (c) 2013-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080gpumon.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#include "ctrl/ctrl2080/ctrl2080base.h"

/*!
 *  This structure represents base class of GPU monitoring sample.
 */
typedef struct LW2080_CTRL_GPUMON_SAMPLE {
    /*!
    * Timestamps in nano-seconds. 
    */
    LW_DECLARE_ALIGNED(LwU64 timeStamp, 8);
} LW2080_CTRL_GPUMON_SAMPLE;

/*!
 *  This structure represents base GPU monitoring sample.
 */
typedef struct LW2080_CTRL_GPUMON_SAMPLES {
    /*!
    * Type of the sample, see LW2080_CTRL_GPUMON_SAMPLE_TYPE_* for reference. 
    */
    LwU8  type;
    /*!
    * Size of the buffer, this should be
    * bufSize ==  LW2080_CTRL_*_GPUMON_SAMPLE_COUNT_*
    *    sizeof(derived type of LW2080_CTRL_GPUMON_SAMPLE).
    */
    LwU32 bufSize;
    /*!
    * Number of samples in ring buffer.
    */
    LwU32 count;
    /*!
    * tracks the offset of the tail in the cirlwlar queue array pSamples.
    */
    LwU32 tracker;
    /*!
    * Pointer to a cirlwlar queue based on array of LW2080_CTRL_GPUMON_SAMPLE
    * or its derived types structs with size == bufSize.
    *
    * @note This cirlwlar queue wraps around after 10 seconds of sampling,
    * and it is clients' responsibility to query within this time frame in
    * order to avoid losing samples.
    * @note With one exception, this queue contains last 10 seconds of samples
    * with tracker poiniting to oldest entry and entry before tracker as the
    * newest entry. Exception is when queue is not full (i.e. tracker is
    * pointing to a zeroed out entry), in that case valid entries are between 0
    * and tracker.
    * @note Clients can store tracker from previous query in order to provide
    * samples since last read.
    */
    LW_DECLARE_ALIGNED(LwP64 pSamples, 8);
} LW2080_CTRL_GPUMON_SAMPLES;

/*!
 * Enumeration of GPU monitoring sample types.
 */
#define LW2080_CTRL_GPUMON_SAMPLE_TYPE_PWR_MONITOR_STATUS 0x00000001
#define LW2080_CTRL_GPUMON_SAMPLE_TYPE_PERFMON_UTIL       0x00000002

/*!
 * Macro for invalid PID.
 */
#define LW2080_GPUMON_PID_ILWALID         ((LwU32)(~0))
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

