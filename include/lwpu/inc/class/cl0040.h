/*
 * Copyright (c) 2001-2001, LWPU CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _cl0040_h_
#define _cl0040_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW01_MEMORY_LOCAL_USER                                    (0x00000040)
/* LwNotification[] fields and values */
#define LW040_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl0040_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw040Typedef, Lw01MemoryLocalUser;
#define LW040_TYPEDEF                                       Lw01MemoryLocalUser
/* obsolete stuff */
#define LW01_MEMORY_USER                                           (0x00000040)
#define LW1_MEMORY_USER                                            (0x00000040)
#define Lw01MemoryUser                                      Lw01MemoryLocalUser
#define lw01MemoryUser                                      Lw01MemoryLocalUser
#define Lw1MemoryUser                                       Lw01MemoryLocalUser
#define lw1MemoryUser                                       Lw01MemoryLocalUser
#define lw01MemoryLocalUser                                 Lw01MemoryLocalUser

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0040_h_ */
