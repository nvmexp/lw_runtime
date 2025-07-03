/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2019 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0080/ctrl0080bif.finn
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

/*
 * LW0080_CTRL_CMD_BIF_RESET
 *
 * This command initiates the specified reset type on the GPU.
 *
 *   flags
 *     Specifies various arguments to the reset operation.
 *
 *     Supported fields include:
 *
 *       LW0080_CTRL_BIF_RESET_FLAGS_TYPE
 *         When set to _SW_RESET, a SW (fullchip) reset is performed. When set
 *         to _SBR, a secondary-bus reset is performed. When set to
 *         _FUNDAMENTAL, a fundamental reset is performed.
 *
 *         NOTE: _FUNDAMENTAL is not yet supported.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0080_CTRL_CMD_BIF_RESET (0x800102) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_BIF_INTERFACE_ID << 8) | LW0080_CTRL_BIF_RESET_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_BIF_RESET_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0080_CTRL_BIF_RESET_PARAMS {
    LwU32 flags;
} LW0080_CTRL_BIF_RESET_PARAMS;

#define LW0080_CTRL_BIF_RESET_FLAGS_TYPE                2:0
#define LW0080_CTRL_BIF_RESET_FLAGS_TYPE_SW_RESET    (0x00000001)
#define LW0080_CTRL_BIF_RESET_FLAGS_TYPE_SBR         (0x00000002)
#define LW0080_CTRL_BIF_RESET_FLAGS_TYPE_FUNDAMENTAL (0x00000003)

/*
 * LW0080_CTRL_BIF_GET_DMA_BASE_SYSMEM_ADDR
 *
 * baseDmaSysmemAddr
 *    This parameter represents the base DMA address for sysmem which will be
 *    added to all DMA accesses issued by GPU. Lwrrently GPUs do not support 64-bit physical address,
 *    hence if sysmem is greater than max GPU supported physical address width, this address
 *    will be non-zero
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_OBJECT_PARENT
 */

#define LW0080_CTRL_CMD_BIF_GET_DMA_BASE_SYSMEM_ADDR (0x800103) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_BIF_INTERFACE_ID << 8) | LW0080_CTRL_BIF_GET_DMA_BASE_SYSMEM_ADDR_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_BIF_GET_DMA_BASE_SYSMEM_ADDR_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0080_CTRL_BIF_GET_DMA_BASE_SYSMEM_ADDR_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 baseDmaSysmemAddr, 8);
} LW0080_CTRL_BIF_GET_DMA_BASE_SYSMEM_ADDR_PARAMS;

/*
 * LW0080_CTRL_BIF_SET_ASPM_FEATURE
 *
 * aspmFeatureSupported
 *    ASPM feature override by client
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW0080_CTRL_CMD_BIF_SET_ASPM_FEATURE (0x800104) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_BIF_INTERFACE_ID << 8) | LW0080_CTRL_BIF_SET_ASPM_FEATURE_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_BIF_SET_ASPM_FEATURE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0080_CTRL_BIF_SET_ASPM_FEATURE_PARAMS {
    LwU32 aspmFeatureSupported;
} LW0080_CTRL_BIF_SET_ASPM_FEATURE_PARAMS;

#define LW0080_CTRL_BIF_ASPM_FEATURE_DT_L0S                0:0
#define LW0080_CTRL_BIF_ASPM_FEATURE_DT_L0S_ENABLED  0x000000001
#define LW0080_CTRL_BIF_ASPM_FEATURE_DT_L0S_DISABLED 0x000000000
#define LW0080_CTRL_BIF_ASPM_FEATURE_DT_L1                 1:1
#define LW0080_CTRL_BIF_ASPM_FEATURE_DT_L1_ENABLED   0x000000001
#define LW0080_CTRL_BIF_ASPM_FEATURE_DT_L1_DISABLED  0x000000000

/* _ctrl0080bif_h_ */

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

