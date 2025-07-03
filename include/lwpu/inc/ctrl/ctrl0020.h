/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0020.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW0020_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x0020, LW0020_CTRL_##cat, idx)

/* LW0020_GPU_MANAGEMENT command categories (6bits) */
#define LW0020_CTRL_RESERVED      (0x00)
#define LW0020_CTRL_GPU_MGMT      (0x01)

/*
 * LW0020_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0020_CTRL_CMD_NULL      (0x200000) /* finn: Evaluated from "(FINN_LW0020_GPU_MANAGEMENT_RESERVED_INTERFACE_ID << 8) | 0x0" */

/* Maximum possible number of bytes of GID information */
#define LW0020_GPU_MAX_GID_LENGTH (0x00000100)

/*
 * LW0020_CTRL_CMD_GPU_MGMT_SET_SHUTDOWN_STATE
 *
 * This command modifies GPU zero power state for the desired GPU in the
 * database. This state is set by a privileged client, after the GPU is
 * completely unregistered from RM as well as PCI subsystem. On Linux,
 * clients perform this operation through pci-sysfs.
 * This control call requires admin privileges.
 *
 *   uuid (INPUT)
 *     The UUID of the gpu.
 *     Supports binary format and SHA-1 type.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */


#define LW0020_CTRL_CMD_GPU_MGMT_SET_SHUTDOWN_STATE (0x200101) /* finn: Evaluated from "(FINN_LW0020_GPU_MANAGEMENT_GPU_MGMT_INTERFACE_ID << 8) | LW0020_CTRL_GPU_MGMT_SET_SHUTDOWN_STATE_PARAMS_MESSAGE_ID" */

#define LW0020_CTRL_GPU_MGMT_SET_SHUTDOWN_STATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0020_CTRL_GPU_MGMT_SET_SHUTDOWN_STATE_PARAMS {
    LwU8 uuid[LW0020_GPU_MAX_GID_LENGTH];
} LW0020_CTRL_GPU_MGMT_SET_SHUTDOWN_STATE_PARAMS;

/* _ctrl0020_h_ */
