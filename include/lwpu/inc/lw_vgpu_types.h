/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2018 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: lw_vgpu_types.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/* XAPIGEN - this file is not suitable for (nor needed by) xapigen.         */
/*           Rather than #ifdef out every such include in every sdk         */
/*           file, punt here.                                               */
#include "lwtypes.h"
 /* ! XAPIGEN */

#define VM_UUID_SIZE            16
#define ILWALID_VGPU_DEV_INST   0xFFFFFFFF
#define MAX_VGPU_DEVICES_PER_VM 16

/* This enum represents the current state of guest dependent fields */
typedef enum GUEST_VM_INFO_STATE {
    GUEST_VM_INFO_STATE_UNINITIALIZED = 0,
    GUEST_VM_INFO_STATE_INITIALIZED = 1,
} GUEST_VM_INFO_STATE;

/* This enum represents types of VM identifiers */
typedef enum VM_ID_TYPE {
    VM_ID_DOMAIN_ID = 0,
    VM_ID_UUID = 1,
} VM_ID_TYPE;

/* This structure represents VM identifier */
typedef union VM_ID {
    LwU8 vmUuid[VM_UUID_SIZE];
    LW_DECLARE_ALIGNED(LwU64 vmId, 8);
} VM_ID;
