/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080pmumon.finn
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
 *  This structure represents base class of PMUMON sample.
 */
typedef struct LW2080_CTRL_PMUMON_SAMPLE {
    /*!
    * Ptimer timestamp in nano-seconds at the time this sample was taken.
    */
    LW_DECLARE_ALIGNED(LwU64 timestamp, 8);
} LW2080_CTRL_PMUMON_SAMPLE;

/*!
 * Common meta-data to be used in control call parameters regarding sampling of
 * FB Cirlwlar Queues.
 */
typedef struct LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER {
    /*!
     * [in/out] Requested starting index for reading into the cirlwlar queue.
     *          This field will be set to (headIndex + 1) % QUEUE_SIZE
     *          automatically by the control call.
     */
    LwU32 tailIndex;

    /*!
     * [in/out] Current sequence ID of the underlying PMUMON queue. Not to be
     *          modified by the caller. Will be modified automatically by the
     *          control call.
     */
    LwU32 sequenceId;

    /*!
     * [out] Current head of the queue. The entry at this index in the queue
     *       is the most recently published.
     */
    LwU32 headIndex;

    /*!
     * [out] Number of elements copied into samples[] starting from samples[0]
     *       of parent structure. This is the number of elements between the
     *       input tailIndex and current headIndex
     */
    LwU32 numSamples;

    /*!
     * [out] Number of times the PMUMON queue in FB has been reset
     *       (i.e. pmuStateLoad). When resetCount of the RM and the client
     *       differ, the RM will correctly resume reading of queue from index 0
     *       instead of attempting to catch up from last read.
     */
    LwU32 resetCount;
} LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER;

//! Event ID type for PMUMon uStreamer queue.
typedef LwU8 LW2080_CTRL_PMUMON_USTREAMER_EVENT;

/*!
 * @defgroup PMUMON_USTRAMER_EVENT eventIds for each channel of PMUMon
 * @{
 */

#define LW2080_CTRL_PMUMON_USTREAMER_EVENT_CLK_DOMAINS        1
#define LW2080_CTRL_PMUMON_USTREAMER_EVENT_FAN_COOLER         2
#define LW2080_CTRL_PMUMON_USTREAMER_EVENT_PERF_POLICY        3
#define LW2080_CTRL_PMUMON_USTREAMER_EVENT_PERF_CF_TOPOLOGIES 4
#define LW2080_CTRL_PMUMON_USTREAMER_EVENT_PWR_CHANNELS       5
#define LW2080_CTRL_PMUMON_USTREAMER_EVENT_THERM_CHANNEL      6
#define LW2080_CTRL_PMUMON_USTREAMER_EVENT_VOLT_RAILS         7
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

