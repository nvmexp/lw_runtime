/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080unix.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0080/ctrl0080base.h"

/* LW01_DEVICE_XX/LW03_DEVICE UNIX-specific control commands and parameters */

/*
 * LW0080_CTRL_CMD_OS_UNIX_VT_SWITCH
 *
 * This command notifies RM to save or restore the current console state. It is
 * intended to be called just before the display driver starts using the display
 * engine, and after it has finished using it.
 *
 *   cmd
 *    Indicates which operation should be performed.
 *
 *      SAVE_VT_STATE
 *        Records the current state of the console, to be restored later.
 *      RESTORE_VT_STATE
 *        Restores the previously-saved console state.
 *
 *   fbInfo
 *     Returns information about the system's framebuffer console, if one
 *     exists. If no console is present, all fields will be zero.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0080_CTRL_CMD_OS_UNIX_VT_SWITCH (0x801e01) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_OS_UNIX_INTERFACE_ID << 8) | LW0080_CTRL_OS_UNIX_VT_SWITCH_PARAMS_MESSAGE_ID" */

typedef struct LW0080_CTRL_OS_UNIX_VT_SWITCH_FB_INFO {
    LwU32 subDeviceInstance;

    LwU16 width;
    LwU16 height;
    LwU16 depth;
    LwU16 pitch;
} LW0080_CTRL_OS_UNIX_VT_SWITCH_FB_INFO;

#define LW0080_CTRL_OS_UNIX_VT_SWITCH_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0080_CTRL_OS_UNIX_VT_SWITCH_PARAMS {
    LwU32                                 cmd;                                    /* in */

    LW0080_CTRL_OS_UNIX_VT_SWITCH_FB_INFO fbInfo;   /* out */
} LW0080_CTRL_OS_UNIX_VT_SWITCH_PARAMS;

/* Called when the display driver needs RM to save the console data,
 * which will be used in RM based console restore */
#define LW0080_CTRL_OS_UNIX_VT_SWITCH_CMD_SAVE_VT_STATE    (0x00000001)

/* Called when the display driver needs RM to restore the console */
#define LW0080_CTRL_OS_UNIX_VT_SWITCH_CMD_RESTORE_VT_STATE (0x00000002)

/* Called when the display driver has restored the console -- RM doesn't
 * need to do anything further, but needs to be informed to avoid turning the
 * GPU off and thus destroying the console state. */
#define LW0080_CTRL_OS_UNIX_VT_SWITCH_CMD_CONSOLE_RESTORED (0x00000003)

/* _ctrl0080unix_h_ */
