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
// Source file: ctrl/ctrlc370/ctrlc370base.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LWC370_DISPLAY control commands and parameters */

#define LWC370_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0XC370, LWC370_CTRL_##cat, idx)

/* LWC370_DISPLAY command categories (6bits) */
#define LWC370_CTRL_RESERVED (0x00)
#define LWC370_CTRL_CHNCTL   (0x01)
#define LWC370_CTRL_RG       (0x02)
#define LWC370_CTRL_SEQ      (0x03)
#define LWC370_CTRL_OR       (0x04)
#define LWC370_CTRL_INST     (0x05)
#define LWC370_CTRL_VERIF    (0x06)
#define LWC370_CTRL_SYSTEM   (0x07)
#define LWC370_CTRL_EVENT    (0x09)

// This struct must be the first member of all C370 control calls
typedef struct LWC370_CTRL_CMD_BASE_PARAMS {
    LwU32 subdeviceIndex;
} LWC370_CTRL_CMD_BASE_PARAMS;


/*
 * LWC370_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWC370_CTRL_CMD_NULL (0xc3700000) /* finn: Evaluated from "(FINN_LWC370_DISPLAY_RESERVED_INTERFACE_ID << 8) | 0x0" */


/* _ctrlc370base_h_ */
