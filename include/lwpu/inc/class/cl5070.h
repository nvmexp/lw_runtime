/*
 * Copyright (c) 1993-2021, LWPU CORPORATION. All rights reserved.
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

#ifndef _cl5070_h_
#define _cl5070_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW50_DISPLAY                                            (0x00005070)

/* event values */
#define LW5070_NOTIFIERS_SW                                      (0)
#define LW5070_NOTIFIERS_MAXCOUNT                                (1)

#define LW5070_NOTIFICATION_STATUS_IN_PROGRESS                   (0x8000)
#define LW5070_NOTIFICATION_STATUS_BAD_ARGUMENT                  (0x4000)
#define LW5070_NOTIFICATION_STATUS_ERROR_ILWALID_STATE           (0x2000)
#define LW5070_NOTIFICATION_STATUS_ERROR_STATE_IN_USE            (0x1000)
#define LW5070_NOTIFICATION_STATUS_DONE_SUCCESS                  (0x0000)

typedef struct
{
    LwU32   numHeads; // Number of HEADs in this chip/display
    LwU32   numDacs;  // Number of DACs in this chip/display
    LwU32   numSors;  // Number of SORs in this chip/display
    LwU32   numPiors; // Number of PIORs in this chip/display
} LW5070_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl5070_h_ */
