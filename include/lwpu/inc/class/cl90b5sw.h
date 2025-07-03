/*
 * SPDX-FileCopyrightText: Copyright (c) 2008-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#include "lwtypes.h"

#ifndef _cl90b5_sw_h_
#define _cl90b5_sw_h_

#ifdef __cplusplus
extern "C" {
#endif

/* This file is *not* auto-generated. */

typedef struct
{
    LwU32    version;            // set to 0
    LwU32    engineInstance;     // CE instance, 0 = highest priority/perf CE.
} LW90B5_ALLOCATION_PARAMETERS;

#define LW90B5_SEMAPHORE_INTERRUPT                  (0)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl90b5_sw_h

