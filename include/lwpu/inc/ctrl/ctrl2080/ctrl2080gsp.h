/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080gsp.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX GSP control commands and parameters */

/*
 * LW2080_CTRL_CMD_GSP_GET_FEATURES
 *
 * This command is used to determine which GSP features are
 * supported on this GPU.
 *
 *   gspFeatures
 *     Bit mask that specifies GSP features supported.
 *   bValid
 *     If this field is set to LW_TRUE, then above bit mask is
 *     considered valid. Otherwise, bit mask should be ignored
 *     as invalid. bValid will be set to LW_TRUE when RM is a
 *     GSP client with GPU support offloaded to GSP firmware.
 *   bDefaultGspRmGpu
 *     If this field is set to LW_TRUE, it indicates that the
 *     underlying GPU has GSP-RM enabled by default. If set to LW_FALSE,
 *     it indicates that the GPU has GSP-RM disabled by default.
 *   firmwareVersion
 *     This field contains the buffer into which the firmware build version
 *     should be returned, if GPU is offloaded. Otherwise, the buffer
 *     will remain untouched.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_GSP_GET_FEATURES    (0x20803601) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GSP_INTERFACE_ID << 8) | LW2080_CTRL_GSP_GET_FEATURES_PARAMS_MESSAGE_ID" */

#define LW2080_GSP_MAX_BUILD_VERSION_LENGTH (0x0000040)

#define LW2080_CTRL_GSP_GET_FEATURES_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_GSP_GET_FEATURES_PARAMS {
    LwU32  gspFeatures;
    LwBool bValid;
    LwBool bDefaultGspRmGpu;
    LwU8   firmwareVersion[LW2080_GSP_MAX_BUILD_VERSION_LENGTH];
} LW2080_CTRL_GSP_GET_FEATURES_PARAMS;

/* Valid feature values */
#define LW2080_CTRL_GSP_GET_FEATURES_UVM_ENABLED                    0:0
#define LW2080_CTRL_GSP_GET_FEATURES_UVM_ENABLED_FALSE (0x00000000)
#define LW2080_CTRL_GSP_GET_FEATURES_UVM_ENABLED_TRUE  (0x00000001)

// _ctrl2080gsp_h_
