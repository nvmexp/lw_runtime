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

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0080/ctrl0080lwjpg.finn
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

/* LW01_DEVICE_XX/LW03_DEVICE LWJPG control commands and parameters */

/* extract cap bit setting from tbl */
#define LW0080_CTRL_LWJPG_GET_CAP(tbl,c)                (((LwU8)tbl[(1?c)]) & (0?c))
#define LW0080_CTRL_LWJPG_GET_CAP_TABLE(tbl,c)          (tbl[(1?c)])

/* caps format is byte_index:bit_mask */
/* This cap is used to expose decoding capability of LWJPG */
#define LW0080_CTRL_LWJPG_CAPS_DEC_CAPABILITY           0:0x00
/* This cap is used to expose encoding capability of LWJPG */
#define LW0080_CTRL_LWJPG_CAPS_ENC_CAPABILITY           4:0x00
/* This cap is used to expose total number of cores in LWJPG */
#define LW0080_CTRL_LWJPG_CAPS_TOTAL_CORE_NUM           8:0x00

/*
 * Size in bytes of LWJPG caps table.  This value should be one greater
 * than the largest byte_index value above.
 */
#define LW0080_CTRL_LWJPG_CAPS_TBL_SIZE   9

/*
 * LW0080_CTRL_CMD_LWJPG_GET_CAPS_V2
 *
 * This command returns the set of LWJPG capabilities for the device
 * in the form of an array of unsigned bytes. LWJPG capabilities
 * include supported features of the LWJPG engine(s) within the device,
 * each represented by a byte offset into the table and a bit position within
 * that byte.
 *
 *   [out] capsTbl
 *     This caps table array is where the LWJPG caps bits will be transferred
 *     by the RM. The caps table is an array of unsigned bytes.
 *   instanceId
 *     This parameter specifies the instance Id of LWDEC for which
 *     cap bits are requested. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_LWJPG_GET_CAPS_V2 (0x801f02) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_LWJPG_INTERFACE_ID << 8) | LW0080_CTRL_LWJPG_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_LWJPG_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0080_CTRL_LWJPG_GET_CAPS_V2_PARAMS {
    LwU8  capsTbl[LW0080_CTRL_LWJPG_CAPS_TBL_SIZE];
    LwU32 instanceId;
} LW0080_CTRL_LWJPG_GET_CAPS_V2_PARAMS;

/* _ctrl0080LWJPG_h_ */

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

