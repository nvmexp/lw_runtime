/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000gsync.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0000/ctrl0000base.h"

#include "ctrl/ctrlxxxx.h"
#include "class/cl30f1.h"
/* LW01_ROOT (client) system controller control commands and parameters */

/*
 * LW0000_CTRL_CMD_GSYNC_GET_ATTACHED_IDS
 *
 * This command returns a table of attached gsyncId values.
 * The table is LW0000_CTRL_GSYNC_MAX_ATTACHED_GSYNCS entries in size.
 *
 *   gsyncIds[]
 *     This parameter returns the table of attached gsync IDs.
 *     The gsync ID is an opaque platform-dependent value that
 *     can be used with the LW0000_CTRL_CMD_GSYNC_GET_ID_INFO command to
 *     retrieve additional information about the gsync device.
 *     The valid entries in gsyncIds[] are contiguous, with a value
 *     of LW0000_CTRL_GSYNC_ILWALID_ID indicating the invalid entries.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_OPERATING_SYSTEM
 */
#define LW0000_CTRL_CMD_GSYNC_GET_ATTACHED_IDS (0x301) /* finn: Evaluated from "(FINN_LW01_ROOT_GSYNC_INTERFACE_ID << 8) | LW0000_CTRL_GSYNC_GET_ATTACHED_IDS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GSYNC_GET_ATTACHED_IDS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_GSYNC_GET_ATTACHED_IDS_PARAMS {
    LwU32 gsyncIds[LW30F1_MAX_GSYNCS];
} LW0000_CTRL_GSYNC_GET_ATTACHED_IDS_PARAMS;

/* this value marks entries in gsyncIds[] as invalid */
#define LW0000_CTRL_GSYNC_ILWALID_ID      (0xffffffff)

/*
 * LW0000_CTRL_CMD_GSYNC_GET_ID_INFO
 *
 * This command returns gsync instance information for the
 * specified gsync device.
 *
 *   gsyncId
 *     This parameter should specify a valid gsync ID value.
 *     If there is no gsync present with the specified ID, a
 *     status of LW_ERR_ILWALID_ARGUMENT is returned.
 *   gsyncFlags
 *     This parameter returns the current state of the gsync device. 
 *   gsyncInstance
 *     This parameter returns the instance number associated with the
 *     specified gsync.  This value can be used to instantiate
 *     a reference to the gsync using one of the LW30_GSYNC
 *     classes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0000_CTRL_CMD_GSYNC_GET_ID_INFO (0x302) /* finn: Evaluated from "(FINN_LW01_ROOT_GSYNC_INTERFACE_ID << 8) | LW0000_CTRL_GSYNC_GET_ID_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GSYNC_GET_ID_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_GSYNC_GET_ID_INFO_PARAMS {
    LwU32 gsyncId;
    LwU32 gsyncFlags;
    LwU32 gsyncInstance;
} LW0000_CTRL_GSYNC_GET_ID_INFO_PARAMS;

/* _ctrl0000gsync_h_ */
