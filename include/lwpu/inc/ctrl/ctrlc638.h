/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc638.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* AMPERE_SMC_EXEC_PARTITION_REF commands and parameters */

#define LWC638_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0xC638, LWC638_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWC638_CTRL_RESERVED       (0x00)
#define LWC638_CTRL_EXEC_PARTITION (0x01)

/*!
 * LWC638_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWC638_CTRL_CMD_NULL       (0xc6380000) /* finn: Evaluated from "(FINN_AMPERE_SMC_EXEC_PARTITION_REF_RESERVED_INTERFACE_ID << 8) | 0x0" */

/*!
 * LWC638_CTRL_CMD_GET_UUID
 *
 * This command returns SHA1 ASCII UUID string as well as the binary UUID for
 * the exelwtion partition. The ASCII string format is,
 * "MIG-%16x-%08x-%08x-%08x-%024x" (the canonical format of a UUID)
 *
 * uuid[OUT]
 *      - Raw UUID bytes
 *
 * uuidStr[OUT]
 *      - ASCII UUID string
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 */

#define LWC638_UUID_LEN            16

/* 'M' 'I' 'G' '-'(x5), '\0x0', extra = 9 */
#define LWC638_UUID_STR_LEN        (0x29) /* finn: Evaluated from "((LWC638_UUID_LEN << 1) + 9)" */





#define LWC638_CTRL_CMD_GET_UUID (0xc6380101) /* finn: Evaluated from "(FINN_AMPERE_SMC_EXEC_PARTITION_REF_EXEC_PARTITION_INTERFACE_ID << 8) | LWC638_CTRL_GET_UUID_PARAMS_MESSAGE_ID" */



#define LWC638_CTRL_GET_UUID_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC638_CTRL_GET_UUID_PARAMS {
    // C form: LwU8 uuid[LWC638_UUID_LEN];
    LwU8 uuid[LWC638_UUID_LEN];

    // C form: char uuidStr[LWC638_UUID_STR_LEN];
    char uuidStr[LWC638_UUID_STR_LEN];
} LWC638_CTRL_GET_UUID_PARAMS;//  _ctrlc638_h_
