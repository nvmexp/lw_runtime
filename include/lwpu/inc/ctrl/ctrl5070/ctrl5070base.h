/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl5070/ctrl5070base.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW5070_DISPLAY control commands and parameters */

#define LW5070_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x5070, LW5070_CTRL_##cat, idx)

/* Display command categories (6bits) */
#define LW5070_CTRL_RESERVED (0x00)
#define LW5070_CTRL_CHNCTL   (0x01)
#define LW5070_CTRL_RG       (0x02)
#define LW5070_CTRL_SEQ      (0x03)
#define LW5070_CTRL_OR       (0x04)
#define LW5070_CTRL_INST     (0x05)
#define LW5070_CTRL_VERIF    (0x06)
#define LW5070_CTRL_SYSTEM   (0x07)
#define LW5070_CTRL_EVENT    (0x09)

/*
 * LW5070_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW5070_CTRL_CMD_NULL (0x50700000) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RESERVED_INTERFACE_ID << 8) | 0x0" */


// This struct must be the first member of all
// 5070 control calls
typedef struct LW5070_CTRL_CMD_BASE_PARAMS {
    LwU32 subdeviceIndex;
} LW5070_CTRL_CMD_BASE_PARAMS;

/* _ctrl5070base_h_ */
