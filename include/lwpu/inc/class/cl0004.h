/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _cl0004_h_
#define _cl0004_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW01_TIMER                                                (0x00000004)
/* LwNotification[] elements */
#define LW004_NOTIFIERS_SET_ALARM_NOTIFY                           (0)
#define LW004_NOTIFIERS_MAXCOUNT                                   (1)

/* mapped timer registers */
typedef volatile struct _Lw01TimerMapTypedef {
    LwU32 Reserved00[0x100];
    LwU32 PTimerTime0;       /* 0x00009400 */
    LwU32 Reserved01[0x3];
    LwU32 PTimerTime1;       /* 0x00009410 */
} Lw01TimerMap;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0004_h_ */
