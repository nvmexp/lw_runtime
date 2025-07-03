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

#ifndef _cl003f_h_
#define _cl003f_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW01_MEMORY_LOCAL_PRIVILEGED                              (0x0000003F)
/* LwNotification[] fields and values */
#define LW03F_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
#ifndef AMD64
typedef volatile struct _cl003f_tag0 {
#else
typedef struct {
#endif
 LwV32 Reserved00[0x7c0];
} Lw01MemoryLocalPrivileged;
#define LW03F_TYPEDEF                                 Lw01MemoryLocalPrivileged
typedef Lw01MemoryLocalPrivileged Lw03fTypedef;
/* obsolete stuff */
#define LW01_MEMORY_PRIVILEGED                                     (0x0000003F)
#define LW1_MEMORY_PRIVILEGED                                      (0x0000003F)
#define Lw01MemoryPrivileged                          Lw01MemoryLocalPrivileged
#define lw01MemoryPrivileged                          Lw01MemoryLocalPrivileged
#define Lw1MemoryPrivileged                           Lw01MemoryLocalPrivileged
#define lw1MemoryPrivileged                           Lw01MemoryLocalPrivileged
#define lw01MemoryLocalPrivileged                     Lw01MemoryLocalPrivileged

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl003f_h_ */
