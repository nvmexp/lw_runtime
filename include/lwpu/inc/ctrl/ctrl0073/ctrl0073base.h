/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0073/ctrl0073base.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW04_DISPLAY_COMMON control commands and parameters */

#define LW0073_CTRL_CMD(cat,idx)                LWXXXX_CTRL_CMD(0x0073, LW0073_CTRL_##cat, idx)

/* LW04_DISPLAY_COMMON command categories (6bits) */
#define LW0073_CTRL_RESERVED (0x00U)
#define LW0073_CTRL_SYSTEM   (0x01U)
#define LW0073_CTRL_SPECIFIC (0x02U)
#define LW0073_CTRL_EVENT    (0x03U)
#define LW0073_CTRL_INTERNAL (0x04U)
#define LW0073_CTRL_DFP      (0x11U)
#define LW0073_CTRL_DP       (0x13U)
#define LW0073_CTRL_SVP      (0x14U)
#define LW0073_CTRL_DPU      (0x15U)
#define LW0073_CTRL_PSR      (0x16U)
#define LW0073_CTRL_STEREO   (0x17U)

/*
 * LW0073_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0073_CTRL_CMD_NULL (0x730000U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl0073base_h_ */
