/* 
 * Copyright (c) 2015-2020, LWPU CORPORATION. All rights reserved.
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

#ifndef _clc370_h_
#define _clc370_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#include "class/cl5070.h"

#define  LWC370_DISPLAY                                             (0x0000C370)

/* event values */
#define LWC370_NOTIFIERS_SW                     LW5070_NOTIFIERS_SW
#define LWC370_NOTIFIERS_BEGIN                  LW5070_NOTIFIERS_MAXCOUNT
#define LWC370_NOTIFIERS_VPR                    LWC370_NOTIFIERS_BEGIN + (0)
#define LWC370_NOTIFIERS_RG_SEM_NOTIFICATION    LWC370_NOTIFIERS_VPR + (1)
#define LWC370_NOTIFIERS_WIN_SEM_NOTIFICATION   LWC370_NOTIFIERS_RG_SEM_NOTIFICATION + (1)
#define LWC370_NOTIFIERS_MAXCOUNT               LWC370_NOTIFIERS_WIN_SEM_NOTIFICATION + (1)

typedef struct
{
    LwU32   numHeads; // Number of HEADs in this chip/display
    LwU32   numSors;  // Number of SORs in this chip/display
    LwU32   numPiors; // Number of PIORs in this chip/display
} LWC370_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clc370_h_ */
