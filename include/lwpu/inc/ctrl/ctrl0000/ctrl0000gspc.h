/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2016 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000gspc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl0000/ctrl0000base.h"

#include "ctrl/ctrlxxxx.h"
/* LW01_ROOT (client) system control commands and parameters */

/*
 * LW0000_CTRL_CMD_GSPC_GET_INFO
 *
 * Get GSPC static info
 *
 */
#define LW0000_CTRL_CMD_GSPC_GET_INFO (0x801) /* finn: Evaluated from "(FINN_LW01_ROOT_GSPC_INTERFACE_ID << 8) | LW0000_CTRL_GSPC_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GSPC_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_GSPC_GET_INFO_PARAMS {
    LwBool isSupported;
} LW0000_CTRL_GSPC_GET_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_GSPC_GET_STATUS
 *
 * Get GSPC run-time info
 *
 */
#define LW0000_CTRL_CMD_GSPC_GET_STATUS     (0x802) /* finn: Evaluated from "(FINN_LW01_ROOT_GSPC_INTERFACE_ID << 8) | LW0000_CTRL_GSPC_GET_STATUS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_GSPC_INFO_STATE_STANDBY (0)
#define LW0000_CTRL_GSPC_INFO_STATE_ENGAGE  (1)
#define LW0000_CTRL_GSPC_INFO_STATE_LIVE    (2)
#define LW0000_CTRL_GSPC_INFO_STATE_PARKED  (3)

#define LW0000_CTRL_GSPC_GET_STATUS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_GSPC_GET_STATUS_PARAMS {
    LwU8 state;
} LW0000_CTRL_GSPC_GET_STATUS_PARAMS;

/*
 * LW0000_CTRL_CMD_GSPC_SET_INPUT
 *
 * Send input data to GSPC
 *
 */
#define LW0000_CTRL_CMD_GSPC_SET_INPUT      (0x803) /* finn: Evaluated from "(FINN_LW01_ROOT_GSPC_INTERFACE_ID << 8) | LW0000_CTRL_GSPC_SET_INPUT_PARAMS_MESSAGE_ID" */

/*
 *  Enumerate the input data types
 *  
 *  FT_ONLY:  Contains only a single frame time.
 */
#define LW0000_CTRL_GSPC_INPUT_TYPE_FT_ONLY (0)

/*
 *  Define the input data types
 */
typedef struct LW0000_CTRL_GSPC_INPUT_DATA_FT_ONLY {
    LwU32 frameTime;
} LW0000_CTRL_GSPC_INPUT_DATA_FT_ONLY;

/*
 *  Union of type-specific input data
 */


/*
 *  Define the parameters for LW0000_CTRL_CMD_GPSC_SET_INPUT
 */
#define LW0000_CTRL_GSPC_SET_INPUT_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0000_CTRL_GSPC_SET_INPUT_PARAMS {
    LwU32 type;

    union {
        LW0000_CTRL_GSPC_INPUT_DATA_FT_ONLY ftOnly;
    } data;
} LW0000_CTRL_GSPC_SET_INPUT_PARAMS;

/* _ctrl0000gspc_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

