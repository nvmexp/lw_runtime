/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080bsp.finn
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

/* LW01_DEVICE_XX/LW03_DEVICE bit stream processor control commands and parameters */

/*
 * LW0080_CTRL_CMD_BSP_GET_CAPS
 *
 * This command returns the set of BSP capabilities for the device
 * in the form of an array of unsigned bytes.  BSP capabilities
 * include supported features and required workarounds for the decoder
 * within the device, each represented by a byte offset into the
 * table and a bit position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0080_CTRL_BSP_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the BSP caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *   instanceId
 *     This parameter specifies the instance Id of LWDEC for which
 *     cap bits are requested. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0080_CTRL_CMD_BSP_GET_CAPS (0x801c01) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_BSP_INTERFACE_ID << 8) | LW0080_CTRL_BSP_GET_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_BSP_GET_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_BSP_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
    LwU32 instanceId;
} LW0080_CTRL_BSP_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0080_CTRL_BSP_GET_CAP(tbl,c)                  (((LwU8)tbl[(1?c)]) & (0?c))
#define LW0080_CTRL_BSP_GET_CAP_TABLE(tbl,c)            (tbl[(1?c)])

/* caps format is byte_index:bit_mask */
#define LW0080_CTRL_BSP_CAPS_SUPPORT_VPX                0:0x01
#define LW0080_CTRL_BSP_CAPS_HEVC_DISABLED              1:0x01
#define LW0080_CTRL_BSP_CAPS_H264_DISABLED              2:0x01
#define LW0080_CTRL_BSP_CAPS_REG0                       4:0x00        /* Used only to get cap table */
#define LW0080_CTRL_BSP_CAPS_REG0_VP8_ENABLED           4:0x04
#define LW0080_CTRL_BSP_CAPS_REG0_VP9_HBD_ENABLED       4:0x10
#define LW0080_CTRL_BSP_CAPS_REG0_HS_DIS_ON             4:0x20        /* Set if LWDEC Falcon HS mode is disabled */

/*
 * Size in bytes of bsp caps table. This value should be one greater
 * than the largest byte_index value above.
 */
#define LW0080_CTRL_BSP_CAPS_TBL_SIZE   8

/*
 * LW0080_CTRL_CMD_BSP_GET_CAPS_V2
 *
 * This command returns the set of BSP capabilities for the device
 * in the form of an array of unsigned bytes.  BSP capabilities
 * include supported features and required workarounds for the decoder
 * within the device, each represented by a byte offset into the
 * table and a bit position within that byte.
 * (The V2 version flattens the capsTbl array pointer).
 *
 *   capsTbl
 *     This parameter is an array of unsigned bytes where the BSP caps bits
 *     will be transferred by the RM.
 *   instanceId
 *     This parameter specifies the instance Id of LWDEC for which
 *     cap bits are requested. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0080_CTRL_CMD_BSP_GET_CAPS_V2 (0x801c02) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_BSP_INTERFACE_ID << 8) | LW0080_CTRL_BSP_GET_CAPS_PARAMS_V2_MESSAGE_ID" */

#define LW0080_CTRL_BSP_GET_CAPS_PARAMS_V2_MESSAGE_ID (0x2U)

typedef struct LW0080_CTRL_BSP_GET_CAPS_PARAMS_V2 {
    LwU8  capsTbl[LW0080_CTRL_BSP_CAPS_TBL_SIZE];
    LwU32 instanceId;
} LW0080_CTRL_BSP_GET_CAPS_PARAMS_V2;

/* _ctrl0080bsp_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

