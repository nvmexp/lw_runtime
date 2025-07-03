/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2017 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _cl0080_h_
#define _cl0080_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwlimits.h"
#include "lwtypes.h"

#define  LW01_DEVICE_0                                             (0x00000080)
/* LwNotification[] fields and values */
#define LW080_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl0080_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw080Typedef, Lw01Device0;
#define  LW080_TYPEDEF                                             Lw01Device0

/* LwAlloc parameteters */
#define LW0080_MAX_DEVICES                                         LW_MAX_DEVICES
/**
 * @brief Alloc param 
 *
 * @param vaMode mode for virtual address space allocation
 *  Three modes:
 *  LW_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES
 *  LW_DEVICE_ALLOCATION_VAMODE_SINGLE_VASPACE
 *  LW_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES
 *  Detailed description of these modes is in lwos.h
 **/ 
typedef struct {
    LwU32       deviceId;
    LwHandle    hClientShare;
    LwHandle    hTargetClient;
    LwHandle    hTargetDevice;
    LwV32       flags;
    LwU64       vaSpaceSize LW_ALIGN_BYTES(8);
    LwU64       vaStartInternal LW_ALIGN_BYTES(8);
    LwU64       vaLimitInternal LW_ALIGN_BYTES(8);
    LwV32       vaMode;
} LW0080_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0080_h_ */
