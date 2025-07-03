/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000base.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW01_ROOT (client) control commands and parameters */

#define LW0000_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x0000,LW0000_CTRL_##cat,idx)

/* Client command categories (6bits) */
#define LW0000_CTRL_RESERVED       (0x00)
#define LW0000_CTRL_SYSTEM         (0x01)
#define LW0000_CTRL_GPU            (0x02)
#define LW0000_CTRL_GSYNC          (0x03)
#define LW0000_CTRL_DIAG           (0x04)
#define LW0000_CTRL_EVENT          (0x05)
#define LW0000_CTRL_LWD            (0x06)
#define LW0000_CTRL_SWINSTR        (0x07)
#define LW0000_CTRL_GSPC           (0x08)
#define LW0000_CTRL_PROC           (0x09)
#define LW0000_CTRL_SYNC_GPU_BOOST (0x0A)
#define LW0000_CTRL_GPUACCT        (0x0B)
#define LW0000_CTRL_VGPU           (0x0C)
#define LW0000_CTRL_CLIENT         (0x0D)

// per-OS categories start at highest category and work backwards
#define LW0000_CTRL_OS_WINDOWS     (0x3F)
#define LW0000_CTRL_OS_MACOS       (0x3E)
#define LW0000_CTRL_OS_UNIX        (0x3D)


/*
 * LW0000_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0000_CTRL_CMD_NULL       (0x0) /* finn: Evaluated from "(FINN_LW01_ROOT_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* _ctrl0000_base_h_ */
