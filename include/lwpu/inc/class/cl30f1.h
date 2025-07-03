/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _cl30f1_h_
#define _cl30f1_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW30_GSYNC */
#define  LW30_GSYNC                                                 (0x000030F1)

/* 
 * A client should use LW01_EVENT_OS_EVENT as hClass and LW30F1_GSYNC_NOTIFIERS_* as 
 * notify index when allocating event, if separate event notifications are needed for 
 * separate event types.
 *
 * A client should use LW01_EVENT_KERNEL_CALLBACK as hClass and 
 * LW30F1_GSYNC_NOTIFIERS_ALL as notify index, if a single event is required. 
 * In this case RM would set event data equal to a pointer to LwNotification structure. 
 * The info32 field of LwNotification structure would be equal a bitmask of  
 * LW30F1_GSYNC_NOTIFIERS_* values.  
 */
   
/* LwNotification[] fields and values */

/* Framelock sync gain and loss events. These are connector specific events. */
#define LW30F1_GSYNC_NOTIFIERS_SYNC_LOSS(c)                               (0x00+(c))
#define LW30F1_GSYNC_NOTIFIERS_SYNC_GAIN(c)                               (0x04+(c))

/* Framelock stereo gain and loss events. These are connector specific events. */
#define LW30F1_GSYNC_NOTIFIERS_STEREO_LOSS(c)                             (0x08+(c))
#define LW30F1_GSYNC_NOTIFIERS_STEREO_GAIN(c)                             (0x0C+(c))

/* House cable gain(plug in) and loss(plug out) events. */
#define LW30F1_GSYNC_NOTIFIERS_HOUSE_GAIN                                 (0x10)
#define LW30F1_GSYNC_NOTIFIERS_HOUSE_LOSS                                 (0x11)

/* RJ45 cable gain(plug in) and loss(plug out) events. */
#define LW30F1_GSYNC_NOTIFIERS_RJ45_GAIN                                  (0x12)
#define LW30F1_GSYNC_NOTIFIERS_RJ45_LOSS                                  (0x13)

#define LW30F1_GSYNC_NOTIFIERS_MAXCOUNT                                   (0x14)

/*
 * For handling all event types.
 * Note for Windows, it only handles LW01_EVENT_KERNEL_CALLBACK_EX; as for LW01_EVENT_OS_EVENT, it can only 
 * signal an event but not handle over any information.
 */
#define LW30F1_GSYNC_NOTIFIERS_ALL                                        LW30F1_GSYNC_NOTIFIERS_MAXCOUNT


#define LW30F1_GSYNC_CONNECTOR_ONE                                        (0)
#define LW30F1_GSYNC_CONNECTOR_TWO                                        (1)
#define LW30F1_GSYNC_CONNECTOR_THREE                                      (2)
#define LW30F1_GSYNC_CONNECTOR_FOUR                                       (3)

#define LW30F1_GSYNC_CONNECTOR_PRIMARY                                    LW30F1_GSYNC_CONNECTOR_ONE
#define LW30F1_GSYNC_CONNECTOR_SECONDARY                                  LW30F1_GSYNC_CONNECTOR_TWO

#define LW30F1_GSYNC_CONNECTOR_COUNT                                      (4)


/* LwRmAlloc parameters */
#define LW30F1_MAX_GSYNCS                                            (0x0000004)
typedef struct {
    LwU32   gsyncInstance;
} LW30F1_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl30f1_h_ */
