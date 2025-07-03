/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2011 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _cl0000_h_
#define _cl0000_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"
#include "lwlimits.h"

/* object LW01_NULL_OBJECT */
#define LW01_NULL_OBJECT                                           (0x00000000)
/* obsolete alises */
#define LW1_NULL_OBJECT                                            LW01_NULL_OBJECT

/*event values*/
#define LW0000_NOTIFIERS_DISPLAY_CHANGE                            (0)
#define LW0000_NOTIFIERS_EVENT_NONE_PENDING                        (1)
#define LW0000_NOTIFIERS_VM_START                                  (2)
#define LW0000_NOTIFIERS_GPU_BIND_EVENT                            (3)
#define LW0000_NOTIFIERS_LWTELEMETRY_REPORT_EVENT                  (4)
#define LW0000_NOTIFIERS_MAXCOUNT                                  (5)

/*Status definitions for LW0000_NOTIFIERS_DISPLAY_CHANGE event*/

#define LW0000_NOTIFIERS_STATUS_ACPI_DISPLAY_DEVICE_CYCLE          (0)

//---------------------------------------------------------------------------

#define LW01_ROOT                                                  (0x00000000)
/* LwNotification[] fields and values */
#define LW000_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)

/* LwAlloc parameteters */
typedef struct {
    LwHandle hClient; /* CORERM-2934: hClient must remain the first member until all allocations use these params */
    LwU32    processID;
    char     processName[LW_PROC_NAME_MAX_LENGTH];
} LW0000_ALLOC_PARAMETERS;

    /* pio method data structure */
typedef volatile struct _cl0000_tag0 {
    LwV32 Reserved00[0x7c0];
} Lw000Typedef, Lw01Root;

/* obsolete aliases */
#define LW000_TYPEDEF                                              Lw01Root
#define Lw1Root                                                    Lw01Root
#define lw1Root                                                    Lw01Root
#define lw01Root                                                   Lw01Root

/*event values*/
#define LW0000_NOTIFIERS_ENABLE_CPU_UTIL_CTRL                      (1) 

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0000_h_ */

