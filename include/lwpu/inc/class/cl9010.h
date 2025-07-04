/*
 * Copyright (c) 2016-2019, LWPU CORPORATION. All rights reserved.
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
#ifndef SDK_CL9010_H
#define SDK_CL9010_H

#include "lwtypes.h"

#define LW9010_VBLANK_CALLBACK 0x9010

typedef void (*OSVBLANKCALLBACKPROC)(void * pParm1, void * pParm2);

typedef struct
{
    OSVBLANKCALLBACKPROC pProc;      // Routine to call at vblank time
    LwV32    LogicalHead;            // Logical Head
    void   *pParm1;                  // pParm1
    void   *pParm2;                  // pParm2
} LW_VBLANK_CALLBACK_ALLOCATION_PARAMETERS;

#endif // SDK_CL9010_H
