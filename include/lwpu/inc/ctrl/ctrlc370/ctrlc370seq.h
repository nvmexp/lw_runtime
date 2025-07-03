/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc370/ctrlc370seq.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/* C370 is partially derived from 5070 */
#include "ctrl/ctrl5070/ctrl5070seq.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * The following control calls are defined in ctrl5070seq.h, but they are
 * still supported on LWC370. We redirect these control cmds to LW5070_CTRL_CMD,
 * and keep the _PARAMS unchanged for now.
 */

#define LWC370_CTRL_CMD_GET_SOR_SEQ_CTL     LW5070_CTRL_CMD_GET_SOR_SEQ_CTL
#define LWC370_CTRL_CMD_SET_SOR_SEQ_CTL     LW5070_CTRL_CMD_SET_SOR_SEQ_CTL
#define LWC370_CTRL_CMD_GET_PIOR_SEQ_CTL    LW5070_CTRL_CMD_GET_PIOR_SEQ_CTL
#define LWC370_CTRL_CMD_SET_PIOR_SEQ_CTL    LW5070_CTRL_CMD_SET_PIOR_SEQ_CTL
#define LWC370_CTRL_CMD_CTRL_SEQ_PROG_SPEED LW5070_CTRL_CMD_CTRL_SEQ_PROG_SPEED
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

