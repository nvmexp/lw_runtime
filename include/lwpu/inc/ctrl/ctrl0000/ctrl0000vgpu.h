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
// Source file: ctrl/ctrl0000/ctrl0000vgpu.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl0000/ctrl0000base.h"

#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrla081.h"
#include "ctrl/ctrla082.h"
#include "class/cl0000.h"
#include "lw_vgpu_types.h"
/*
 * LW0000_CTRL_CMD_VGPU_GET_START_DATA
 *
 * This command gets data associated with LW0000_NOTIFIERS_VGPU_MGR_START to
 * start VGPU process.
 *
 *   mdevUuid
 *     This parameter gives mdev device UUID for which lwpu-vgpu-mgr should
 *     init process.
 *
 *   qemuPid
 *     This parameter specifies the QEMU process ID of the VM.
 *
 *   gpuPciId
 *     This parameter provides gpuId of GPU on which vgpu device is created.
 *
 *   configParams
 *     This parameter specifies the configuration parameters for vGPU
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_EVENT
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_CLIENT
 *
 */
#define LW0000_CTRL_CMD_VGPU_GET_START_DATA (0xc01) /* finn: Evaluated from "(FINN_LW01_ROOT_VGPU_INTERFACE_ID << 8) | LW0000_CTRL_VGPU_GET_START_DATA_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_VGPU_GET_START_DATA_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_VGPU_GET_START_DATA_PARAMS {
    LwU8  mdevUuid[VM_UUID_SIZE];
    LwU8  configParams[1024];
    LwU32 qemuPid;
    LwU32 gpuPciId;
    LwU16 vgpuId;
    LwU32 gpuPciBdf;
} LW0000_CTRL_VGPU_GET_START_DATA_PARAMS;

/* _ctrl0000vgpu_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

