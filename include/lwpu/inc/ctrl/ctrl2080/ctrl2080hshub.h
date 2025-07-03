/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080hshub.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/*
 * LW2080_CTRL_CMD_HSHUB_GET_AVAILABLE_MASK
 *
 * This command get active HSHUB masks.
 *
 *   hshubNcisocMask
 *     NCISOC enabled active HSHUBs
 *   hshubLwlMask
 *     LWLINK capable active HSHUBs.
 */

#define LW2080_CTRL_CMD_HSHUB_GET_AVAILABLE_MASK_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_CMD_HSHUB_GET_AVAILABLE_MASK_PARAMS {
    LwU32 hshubNcisocMask;
    LwU32 hshubLwlMask;
} LW2080_CTRL_CMD_HSHUB_GET_AVAILABLE_MASK_PARAMS;

#define LW2080_CTRL_CMD_HSHUB_GET_AVAILABLE_MASK (0x20804101) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_HSHUB_INTERFACE_ID << 8) | LW2080_CTRL_CMD_HSHUB_GET_AVAILABLE_MASK_PARAMS_MESSAGE_ID" */

/* _ctrl2080hshub_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

