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

#ifndef _cl003e_h_
#define _cl003e_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define   LW01_CONTEXT_ERROR_TO_MEMORY                             (0x0000003E)
#define   LW01_MEMORY_SYSTEM                                       (0x0000003E)
/* LwNotification[] fields and values */
#define LW03E_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl003e_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw03eTypedef, Lw01ContextErrorToMemory;
#define  LW03E_TYPEDEF                                 Lw01ContextErrorToMemory
/* obsolete stuff */
#define  LW1_CONTEXT_ERROR_TO_MEMORY                               (0x0000003E)
#define  Lw1ContextErrorToMemory                       Lw01ContextErrorToMemory
#define  lw1ContextErrorToMemory                       Lw01ContextErrorToMemory
#define  lw01ContextErrorToMemory                      Lw01ContextErrorToMemory

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl003e_h_ */
