/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2018 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080clk.finn
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

/* LW01_DEVICE_XX/LW03_DEVICE clk control commands and parameters */

/*
 * LW0080_CTRL_CMD_SET_VPLL_REF
 *
 * This command sets the vpll ref that must be used for programming VPLL in
 * future. Note that VPLL won't be immediately programmed. Only future
 * programming would get affected.
 *
 *   head
 *      The head for which this cmd is intended.
 *
 *   refName
 *      The ref clk that must be used.
 *
 *   refFreq
 *      Frequency of the specified reference source. This field is relevant
 *      only when refName = EXT_REF, QUAL_EXT_REF or EXT_SPREAD.
 *      The unit of frequency is Hz.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 */
#define LW0080_CTRL_CMD_SET_VPLL_REF                       (0x801001) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CLK_INTERFACE_ID << 8) | LW0080_CTRL_CMD_SET_VPLL_REF_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_CMD_SET_VPLL_REF_REF_NAME                             31:0
#define LW0080_CTRL_CMD_SET_VPLL_REF_REF_NAME_XTAL         (0x00000000)
#define LW0080_CTRL_CMD_SET_VPLL_REF_REF_NAME_SPPLL0       (0x00000001)
#define LW0080_CTRL_CMD_SET_VPLL_REF_REF_NAME_SPPLL1       (0x00000002)
#define LW0080_CTRL_CMD_SET_VPLL_REF_REF_NAME_EXT_REF      (0x00000003)
#define LW0080_CTRL_CMD_SET_VPLL_REF_REF_NAME_QUAL_EXT_REF (0x00000004)
#define LW0080_CTRL_CMD_SET_VPLL_REF_REF_NAME_EXT_SPREAD   (0x00000005)

#define LW0080_CTRL_CMD_SET_VPLL_REF_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_CMD_SET_VPLL_REF_PARAMS {
    LwU32 head;

    LwU32 refName;
    LwU32 refFreq;
} LW0080_CTRL_CMD_SET_VPLL_REF_PARAMS;

/*
 * LW0080_CTRL_CMD_SET_VPLL_ARCH_TYPE
 *
 * This command is unsupported.
 *
 *   head
 *      The head for which this cmd is intended.
 *
 *   archType
 *      The number of stages that must be used.
 *
 * Possible status values returned are:
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_SET_VPLL_ARCH_TYPE (0x801003) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_CLK_INTERFACE_ID << 8) | 0x3" */

typedef struct LW0080_CTRL_CMD_SET_VPLL_ARCH_TYPE_PARAMS {
    LwU32 head;
    LwU32 archType;
} LW0080_CTRL_CMD_SET_VPLL_ARCH_TYPE_PARAMS;

#define LW0080_CTRL_CMD_SET_VPLL_ARCH_TYPE_SINGLE_STAGE_A (0x00000000)
#define LW0080_CTRL_CMD_SET_VPLL_ARCH_TYPE_SINGLE_STAGE_B (0x00000001)
#define LW0080_CTRL_CMD_SET_VPLL_ARCH_TYPE_DUAL_STAGE     (0x00000002)

/* _ctrl0080clk_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

