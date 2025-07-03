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

#ifndef _cl0002_h_
#define _cl0002_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW01_CONTEXT_DMA_FROM_MEMORY                              (0x00000002)
/* LwNotification[] fields and values */
#define LW002_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl0002_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw002Typedef, Lw01ContextDmaFromMemory;
#define  LW002_TYPEDEF                                 Lw01ContextDmaFromMemory
/* obsolete stuff */
#define  LW1_CONTEXT_DMA_FROM_MEMORY                               (0x00000002)
#define  LW01_CONTEXT_DMA                                          (0x00000002)
#define  Lw1ContextDmaFromMemory                       Lw01ContextDmaFromMemory
#define  lw1ContextDmaFromMemory                       Lw01ContextDmaFromMemory
#define  lw01ContextDmaFromMemory                      Lw01ContextDmaFromMemory

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0002_h_ */
