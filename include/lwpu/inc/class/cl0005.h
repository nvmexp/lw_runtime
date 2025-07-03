/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2014 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _cl0005_h_
#define _cl0005_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW01_EVENT                                                (0x00000005)
/* LwNotification[] fields and values */
#define LW003_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl0005_tag0 {
    LwV32 Reserved00[0x7c0];
} Lw005Typedef, Lw01Event;
#define LW005_TYPEDEF                                              Lw01Event
/* obsolete stuff */
#define LW1_TIMER                                                  (0x00000004)
#define Lw1Event                                                   Lw01Event
#define lw1Event                                                   Lw01Event
#define lw01Event                                                  Lw01Event

/* LwRmAlloc() parameters */
typedef struct {
    LwHandle hParentClient;
    LwHandle hSrcResource;

    LwV32    hClass;
    LwV32    notifyIndex;
    LwP64    data LW_ALIGN_BYTES(8);
} LW0005_ALLOC_PARAMETERS;

/* LW0005_ALLOC_PARAMETERS's notifyIndex field is overloaded to contain the
 * notifyIndex value itself, plus flags, and optionally a subdevice field if
 * flags contains LW01_EVENT_SUBDEVICE_SPECIFIC. Note that LW01_EVENT_*
 * contain the full 32-bit flag value that is OR'd into notifyIndex, not the
 * contents of the FLAGS field (i.e. LW01_EVENT_* are pre-shifted into place).
 */
#define LW0005_NOTIFY_INDEX_INDEX     15:0
#define LW0005_NOTIFY_INDEX_SUBDEVICE 23:16
#define LW0005_NOTIFY_INDEX_FLAGS     31:24

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0005_h_ */
