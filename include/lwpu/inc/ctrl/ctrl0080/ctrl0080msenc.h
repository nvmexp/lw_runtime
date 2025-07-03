/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080msenc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE MSENC control commands and parameters */

/*
 * LW0080_CTRL_CMD_MSENC_GET_CAPS
 *
 * This command returns the set of MSENC capabilities for the device
 * in the form of an array of unsigned bytes. MSENC capabilities
 * include supported features and required workarounds for the MSENC-related
 * engine(s) within the device, each represented by a byte offset into
 * the table and a bit position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0080_CTRL_MSENC_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the MSENC caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_MSENC_GET_CAPS (0x801b01) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_MSENC_INTERFACE_ID << 8) | LW0080_CTRL_MSENC_GET_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_MSENC_GET_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_MSENC_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
} LW0080_CTRL_MSENC_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0080_CTRL_MSENC_GET_CAP(tbl,c)                (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW0080_CTRL_MSENC_CAPS_MPECREWIND_BUG_775053    0:0x01
/* This cap is used to expose fuse settings to LWENC, refer Bug 1388560 */
#define LW0080_CTRL_MSENC_CAPS_FUSE_MSENC_THROTTLE      1:0x01
/* This cap is used to expose HEVC fuse setting to LWENC, refer Bug 2010807 */
#define LW0080_CTRL_MSENC_CAPS_HEVC_DISABLED            2:0x01
/* This cap is used to expose H264 VBIOS flag to LWENC, refer Bug 2943186 */
#define LW0080_CTRL_MSENC_CAPS_H264_DISABLED            3:0x01

/* size in bytes of MSENC caps table */
#define LW0080_CTRL_MSENC_CAPS_TBL_SIZE 4

/* _ctrl0080msenc_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

