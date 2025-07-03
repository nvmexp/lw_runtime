/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _cl9074_h_
#define _cl9074_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define GF100_TIMED_SEMAPHORE_SW                                   (0x00009074)

/* LwNotification[] fields and values */
#define LW9074_NOTIFICATION_STATUS_PENDING                         (0x8000)
#define LW9074_NOTIFICATION_STATUS_DONE_FLUSHED                    (0x0001)
#define LW9074_NOTIFICATION_STATUS_DONE                            (0x0000)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* LW9074 does not support PIO access */

/* DMA method offsets, fields, and values */
#define LW9074_SET_OBJECT                                          (0x00000000)

#define LW9074_NO_OPERATION                                        (0x00000100)

/* Notifier to received status for this particular release */
#define LW9074_SET_NOTIFIER_HI                                     (0x00000140)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9074_SET_NOTIFIER_HI_V                                            7:0

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9074_SET_NOTIFIER_LO                                     (0x00000144)
#define LW9074_SET_NOTIFIER_LO_V                                           31:0

/* Semaphore to release */
#define LW9074_SET_SEMAPHORE_HI                                    (0x00000148)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9074_SET_SEMAPHORE_HI_V                                           7:0

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9074_SET_SEMAPHORE_LO                                    (0x0000014c)
#define LW9074_SET_SEMAPHORE_LO_V                                          31:0

/* GPU timestamp at which to release semaphore */
#define LW9074_SET_WAIT_TIMESTAMP_HI                               (0x00000150)
#define LW9074_SET_WAIT_TIMESTAMP_HI_V                                     31:0

#define LW9074_SET_WAIT_TIMESTAMP_LO                               (0x00000154)
#define LW9074_SET_WAIT_TIMESTAMP_LO_V                                     31:0

/* Value to release semaphore to */
#define LW9074_SET_SEMAPHORE_RELEASE_VALUE                         (0x00000158)
#define LW9074_SET_SEMAPHORE_RELEASE_VALUE_V                               31:0

/* When written, schedules semaphore release using above parameters */
#define LW9074_SCHEDULE_SEMAPHORE_RELEASE                          (0x0000015c)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9074_SCHEDULE_SEMAPHORE_RELEASE_NOTIFY                            1:0
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9074_SCHEDULE_SEMAPHORE_RELEASE_NOTIFY_WRITE_ONLY        (0x00000000)
#define LW9074_SCHEDULE_SEMAPHORE_RELEASE_NOTIFY_WRITE_THEN_AWAKEN (0x00000001)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl9074_h_ */

