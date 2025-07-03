/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080base.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW01_DEVICE_XX/LW03_DEVICE control commands and parameters */

#define LW0080_CTRL_CMD(cat,idx)                LWXXXX_CTRL_CMD(0x0080, LW0080_CTRL_##cat, idx)

/* GPU device command categories (6bits) */
#define LW0080_CTRL_RESERVED (0x00)
#define LW0080_CTRL_BIF      (0x01)
#define LW0080_CTRL_GPU      (0x02)
#define LW0080_CTRL_CLK      (0x10)
#define LW0080_CTRL_GR       (0x11)
#define LW0080_CTRL_CIPHER   (0x12)
#define LW0080_CTRL_FB       (0x13)
#define LW0080_CTRL_HOST     (0x14)
#define LW0080_CTRL_VIDEO    (0x15)
#define LW0080_CTRL_FIFO     (0x17)
#define LW0080_CTRL_DMA      (0x18)
#define LW0080_CTRL_PERF     (0x19)
#define LW0080_CTRL_MSENC    (0x1B)
#define LW0080_CTRL_BSP      (0x1C)
#define LW0080_CTRL_RC       (0x1D)
#define LW0080_CTRL_OS_UNIX  (0x1E)
#define LW0080_CTRL_LWJPG    (0x1F)
#define LW0080_CTRL_INTERNAL (0x20)
#define LW0080_CTRL_LWLINK   (0x21)

/*
 * LW0080_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0080_CTRL_CMD_NULL (0x800000) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl0080base_h_ */

/* extract device cap setting from specified category-specific caps table */
#define LW0080_CTRL_GET_CAP(cat,tbl,c)    \
    LW0080_CTRL_##cat##_GET_CAP(tbl, LW0080_CTRL_##cat##_CAPS_##c)
