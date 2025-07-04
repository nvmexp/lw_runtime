/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2017 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080rc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE gpu control commands and parameters */

/*
 * LW0080_CTRL_CMD_RC_DISABLE_RESET_CHANNEL_CALLBACK
 *
 * This command prevents RM from using callbacks when resetting a channel due
 * to a page fault.
 *
 * Possible status return values are:
 *   LW_OK
 */
#define LW0080_CTRL_CMD_RC_DISABLE_RESET_CHANNEL_CALLBACK (0x801d01) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_RC_INTERFACE_ID << 8) | 0x1" */

/*
 * LW0080_CTRL_CMD_RC_ENABLE_RESET_CHANNEL_CALLBACK
 *
 * This command permits RM to use callbacks when resetting a channel due
 * to a page fault.
 *
 * Possible status return values are:
 *   LW_OK
 */
#define LW0080_CTRL_CMD_RC_ENABLE_RESET_CHANNEL_CALLBACK  (0x801d02) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_RC_INTERFACE_ID << 8) | 0x2" */

/* _ctrl0080rc_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

