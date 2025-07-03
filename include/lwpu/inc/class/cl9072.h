/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _cl9072_h_
#define _cl9072_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define GF100_DISP_SW                                                0x00009072

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9072_SET_OBJECT                                                0x0000
#define LW9072_SET_OBJECT_CLASS_ID                                         15:0
#define LW9072_SET_OBJECT_ENGINE_ID                                       20:16

#define LW9072_NO_OPERATION                                              0x0100
#define LW9072_NO_OPERATION_V                                              31:0

#define LW9072_SET_NOTIFY_HI                                             0x0104
#define LW9072_SET_NOTIFY_HI_OFFSET                                         7:0

#define LW9072_SET_NOTIFY_LO                                             0x0108
#define LW9072_SET_NOTIFY_LO_OFFSET                                        31:0

#define LW9072_NOTIFY_ON_VBLANK                                          0x010c
#define LW9072_NOTIFY_ON_VBLANK_NOTIFY                                      0:0
#define LW9072_NOTIFY_ON_VBLANK_NOTIFY_WRITE_ONLY                    0x00000000
#define LW9072_NOTIFY_ON_VBLANK_NOTIFY_WRITE_THEN_AWAKEN             0x00000001

#define LW9072_SET_SWAP_READY                                            0x0388
#define LW9072_SET_SWAP_READY_MODE                                          2:0
#define LW9072_SET_SWAP_READY_MODE_FLIP                              0x00000000
#define LW9072_SET_SWAP_READY_MODE_BLIT                              0x00000001
#define LW9072_SET_SWAP_READY_MODE_FLIP_AFR                          0x00000002
#define LW9072_SET_SWAP_READY_MODE_FLIP_SFR                          0x00000003
#define LW9072_SET_SWAP_READY_MODE_NOT_SET                           0x00000004
#define LW9072_SET_SWAP_READY_ARMED                                         3:3
#define LW9072_SET_SWAP_READY_ARMED_TRUE                             0x00000000
#define LW9072_SET_SWAP_READY_ARMED_FALSE                            0x00000001
#define LW9072_SET_SWAP_READY_SLI_MODE                                      4:4
#define LW9072_SET_SWAP_READY_SLI_MODE_DEFAULT                       0x00000000
#define LW9072_SET_SWAP_READY_SLI_MODE_MASTER_ONLY                   0x00000001
#define LW9072_SET_SWAP_READY_SURFACE                                       6:5

#define LW9072_SET_REPORT_SEMAPHORE_HI                                   0x1b00
#define LW9072_SET_REPORT_SEMAPHORE_HI_OFFSET                               7:0

#define LW9072_SET_REPORT_SEMAPHORE_LO                                   0x1b04
#define LW9072_SET_REPORT_SEMAPHORE_LO_OFFSET                              31:0

#define LW9072_SET_REPORT_SEMAPHORE_RELEASE                              0x1b08
#define LW9072_SET_REPORT_SEMAPHORE_RELEASE_V                              31:0

#define LW9072_SET_REPORT_SEMAPHORE_SCHED                                0x1b0c
#define LW9072_SET_REPORT_SEMAPHORE_SCHED_WHEN                              0:0
#define LW9072_SET_REPORT_SEMAPHORE_SCHED_WHEN_VSYNC                 0x00000000
#define LW9072_SET_REPORT_SEMAPHORE_SCHED_WHEN_SWAP_READY            0x00000001

#define LW9072_SET_PRESENT_INTERVAL                                      0x1b10


/* LwNotification[] elements
 * We skip some values for compatibility with class 07C */
#define LW9072_NOTIFIERS_NOTIFY                                             (0)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9072_NOTIFIERS_NOTIFY_ON_VBLANK                                   (9)
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9072_NOTIFIERS_MAXCOUNT                                           (10)

/* LwNotification[] fields and values */
#define LW9072_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT                  (0x2000)
#define LW9072_NOTIFICATION_STATUS_ERROR_ILWALID_STATE                 (0x1000)
#define LW9072_NOTIFICATION_STATUS_ERROR_STATE_IN_USE                  (0x0800)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LW9072_NOTIFICATION_STATUS_DONE_SUCCESS                        (0x0000)


typedef struct
{
    LwU32   logicalHeadId;
    /*
     *  0 implies use Head argument only (i.e. whatever is lwrrently setup on this head)
     */
    LwU32   displayMask;
    LwU32   caps;
} LW9072_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl9072_h_ */
