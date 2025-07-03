/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080dmabuf.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW2080_CTRL_CMD_DMABUF_EXPORT_OBJECTS_TO_FD
 *
 * Exports RM vidmem handles to a dma-buf fd.
 *
 * The objects in the 'handles' array are exported to the fd as range:
 * [index, index + numObjects).
 *
 * A dma-buf fd is created the first time this control call is called.
 * The fd is an input parameter for subsequent calls to attach additional handles
 * over LW2080_CTRL_DMABUF_MAX_HANDLES.
 *
 * fd
 *   A dma-buf file descriptor. If -1, a new FD will be created.
 *
 * totalObjects
 *   The total number of objects that the client wishes to export to the FD.
 *   This parameter will be honored only when the FD is getting created.
 *
 * numObjects
 *   The number of handles the user wishes to export in this call.
 *
 * index
 *   The index into the export fd at which to start exporting the handles in
 *   'handles'. This index cannot overlap a previously used index.
 *
 * totalSize
 *   The total size of memory being exported in bytes, needed to create the dma-buf.
 *   This size includes the memory that will be exported in future export calls
 *   for this dma-buf.
 *
 * handles
 *   An array of {handle, offset, size} that describes the dma-buf.
 *   The offsets and sizes must be OS page-size aligned.
 *
 * Limitations:
 *   1. This call only supports vidmem objects for now.
 *   2. All memory handles should belong to the same GPU or the same GPU MIG instance.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_NOT_SUPPORTED
 *    LW_ERR_NO_MEMORY
 *    LW_ERR_OPERATING_SYSTEM
 *    LW_ERR_IN_USE
 *    LW_ERR_ILWALID_OBJECT
 *    LW_ERR_ILWALID_OBJECT_PARENT
 */
#define LW2080_CTRL_CMD_DMABUF_EXPORT_OBJECTS_TO_FD (0x20803a01) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_DMABUF_INTERFACE_ID << 8) | LW2080_CTRL_DMABUF_EXPORT_MEM_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_DMABUF_MAX_HANDLES              128

typedef struct LW2080_CTRL_DMABUF_MEM_HANDLE_INFO {
    LwHandle hMemory;
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
} LW2080_CTRL_DMABUF_MEM_HANDLE_INFO;

#define LW2080_CTRL_DMABUF_EXPORT_MEM_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_DMABUF_EXPORT_MEM_PARAMS {
    LwS32 fd;
    LwU32 totalObjects;
    LwU32 numObjects;
    LwU32 index;
    LW_DECLARE_ALIGNED(LwU64 totalSize, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_DMABUF_MEM_HANDLE_INFO handles[LW2080_CTRL_DMABUF_MAX_HANDLES], 8);
} LW2080_CTRL_DMABUF_EXPORT_MEM_PARAMS;

// _ctrl2080dmabuf_h_
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

