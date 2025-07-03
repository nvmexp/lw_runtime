/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080ce.finn
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

/* LW20_SUBDEVICE_XX ce control commands and parameters */

#include "ctrl2080common.h"

/*
 * LW2080_CTRL_CMD_CE_GET_CAPS
 *
 * This command returns the set of CE capabilities for the device
 * in the form of an array of unsigned bytes.
 *
 *   ceEngineType
 *     This parameter specifies the copy engine type
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table per CE.
 *     This value should be set to LW2080_CTRL_CE_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the CE caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_CE_GET_CAPS  (0x20802a01) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_GET_CAPS_PARAMS_MESSAGE_ID" */

/*
 * Size in bytes of CE caps table.  This value should be one greater
 * than the largest byte_index value below.
 */
#define LW2080_CTRL_CE_CAPS_TBL_SIZE 2

#define LW2080_CTRL_CE_GET_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_CE_GET_CAPS_PARAMS {
    LwU32 ceEngineType;
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
} LW2080_CTRL_CE_GET_CAPS_PARAMS;

#define LW2080_CTRL_CMD_CE_GET_CAPS_V2 (0x20802a03) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CE_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_CE_GET_CAPS_V2_PARAMS {
    LwU32 ceEngineType;
    LwU8  capsTbl[LW2080_CTRL_CE_CAPS_TBL_SIZE];
} LW2080_CTRL_CE_GET_CAPS_V2_PARAMS;

/* extract cap bit setting from tbl */
#define LW2080_CTRL_CE_GET_CAP(tbl,c)               (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW2080_CTRL_CE_CAPS_CE_GRCE                          0:0x01
#define LW2080_CTRL_CE_CAPS_CE_SHARED                        0:0x02
#define LW2080_CTRL_CE_CAPS_CE_SYSMEM_READ                   0:0x04
#define LW2080_CTRL_CE_CAPS_CE_SYSMEM_WRITE                  0:0x08
#define LW2080_CTRL_CE_CAPS_CE_LWLINK_P2P                    0:0x10
#define LW2080_CTRL_CE_CAPS_CE_SYSMEM                        0:0x20
#define LW2080_CTRL_CE_CAPS_CE_P2P                           0:0x40
#define LW2080_CTRL_CE_CAPS_CE_BL_SIZE_GT_64K_SUPPORTED      0:0x80
#define LW2080_CTRL_CE_CAPS_CE_SUPPORTS_NONPIPELINED_BL      1:0x01
#define LW2080_CTRL_CE_CAPS_CE_SUPPORTS_PIPELINED_BL         1:0x02

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW2080_CTRL_CE_CAPS_CE_CC_SELWRE                     1:0x04
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 *   LW2080_CTRL_CE_CAPS_CE_GRCE
 *     Set if the CE is synchronous with GR
 *
 *   LW2080_CTRL_CE_CAPS_CE_SHARED
 *     Set if the CE shares physical CEs with any other CE
 *
 *   LW2080_CTRL_CE_CAPS_CE_SYSMEM_READ
 *     Set if the CE can give enhanced performance for SYSMEM reads over other CEs
 *
 *   LW2080_CTRL_CE_CAPS_CE_SYSMEM_WRITE
 *     Set if the CE can give enhanced performance for SYSMEM writes over other CEs
 *
 *   LW2080_CTRL_CE_CAPS_CE_LWLINK_P2P
 *     Set if the CE can be used for P2P transactions using LWLINK
 *     Once a CE is exposed for P2P over LWLINK, it will remain available for the life of RM
 *     PCE2LCE mapping may change based on the number of GPUs registered in RM however
 *
 *   LW2080_CTRL_CE_CAPS_CE_SYSMEM
 *     Set if the CE can be used for SYSMEM transactions
 *
 *   LW2080_CTRL_CE_CAPS_CE_P2P
 *     Set if the CE can be used for P2P transactions
 *
 *   LW2080_CTRL_CE_CAPS_CE_BL_SIZE_GT_64K_SUPPORTED
 *     Set if the CE supports BL copy size greater than 64K
 *
 *   LW2080_CTRL_CE_CAPS_CE_SUPPORTS_NONPIPELINED_BL
 *     Set if the CE supports non-pipelined Block linear
 *
 *   LW2080_CTRL_CE_CAPS_CE_SUPPORTS_PIPELINED_BL
 *     Set if the CE supports pipelined Block Linear
 */

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 *   LW2080_CTRL_CE_CAPS_CE_CC_SELWRE
 *     Set if the CE is capable of encryption/decryption
 */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1) && (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_CE_GET_CE_PCE_MASK
 *
 * This command returns the mapping of PCE's for the given LCE
 *
 *   ceEngineType
 *     This parameter specifies the copy engine type
 *   pceMask
 *     This parameter specifies a mask of PCEs that correspond
 *     to the LCE specified in ceEngineType
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_CE_GET_CE_PCE_MASK (0x20802a02) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS {
    LwU32 ceEngineType;
    LwU32 pceMask;
} LW2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS;

/*
 * LW2080_CTRL_CMD_CE_SET_PCE_LCE_CONFIG
 *
 * This command sets the PCE2LCE configuration
 *
 *   pceLceConfig[LW2080_CTRL_MAX_PCES]
 *     This parameter specifies the PCE-LCE mapping requested
 *   grceLceConfig[LW2080_CTRL_MAX_GRCES]
 *     This parameter specifies which LCE is the GRCE sharing with
 *     0xF -> Does not share with any LCE
 *     0-MAX_LCE -> Shares with the given LCE
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_CE_SET_PCE_LCE_CONFIG (0x20802a04) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_SET_PCE_LCE_CONFIG_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_MAX_PCES                  32
#define LW2080_CTRL_MAX_GRCES                 2

#define LW2080_CTRL_CE_SET_PCE_LCE_CONFIG_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_CE_SET_PCE_LCE_CONFIG_PARAMS {
    LwU32 ceEngineType;
    LwU32 pceLceMap[LW2080_CTRL_MAX_PCES];
    LwU32 grceSharedLceMap[LW2080_CTRL_MAX_GRCES];
} LW2080_CTRL_CE_SET_PCE_LCE_CONFIG_PARAMS;

/*
 * LW2080_CTRL_CMD_CE_UPDATE_PCE_LCE_MAPPINGS
 *
 * This command updates the PCE-LCE mappings
 *
 *   pPceLceMap [IN]
 *     This parameter tracks the array of PCE-LCE mappings.
 *
 *   pGrceConfig [IN]
 *     This parameter tracks the array of GRCE configs.
 *     0xF -> GRCE does not share with any LCE
 *     0-MAX_LCE -> GRCE shares with the given LCE
 *
 *   exposeCeMask [IN]
 *     This parameter specifies the mask of LCEs to export to the
 *     clients after the update.
 *
 *   bUpdateLwlinkPceLce [IN]
 *     Whether PCE-LCE mappings need to be updated for lwlink topology.
 *     If this is LW_FALSE, RM would ignore the above values.  However,
 *     PCE-LCE mappings will still be updated if there were any regkey
 *     overrides.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 */

#define LW2080_CTRL_CMD_CE_UPDATE_PCE_LCE_MAPPINGS (0x20802a05) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_UPDATE_PCE_LCE_MAPPINGS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CE_UPDATE_PCE_LCE_MAPPINGS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_CE_UPDATE_PCE_LCE_MAPPINGS_PARAMS {
    LwU32  pceLceMap[LW2080_CTRL_MAX_PCES];
    LwU32  grceConfig[LW2080_CTRL_MAX_GRCES];
    LwU32  exposeCeMask;
    LwBool bUpdateLwlinkPceLce;
} LW2080_CTRL_CE_UPDATE_PCE_LCE_MAPPINGS_PARAMS;

#define LW2080_CTRL_CMD_CE_UPDATE_CLASS_DB (0x20802a06) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_UPDATE_CLASS_DB_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CE_UPDATE_CLASS_DB_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_CE_UPDATE_CLASS_DB_PARAMS {
    LwU32 stubbedCeMask;
} LW2080_CTRL_CE_UPDATE_CLASS_DB_PARAMS;

/*
 * LW2080_CTRL_CMD_CE_GET_PHYSICAL_CAPS
 *
 * Query _CE_GRCE, _CE_SHARED, _CE_SUPPORTS_PIPELINED_BL, _CE_SUPPORTS_NONPIPELINED_BL bits of CE
 * capabilities.
 *
 */

#define LW2080_CTRL_CMD_CE_GET_PHYSICAL_CAPS (0x20802a07) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | 0x7" */

#define LW2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS {
    LwU32 size;
} LW2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS;

#define LW2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE (0x20802a08) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_GET_FAULT_METHOD_BUFFER_SIZE_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_CE_GET_HUB_PCE_MASKS
 *
 * Get HSHUB and FBHUB PCE masks.
 *
 *   [out] hshubPceMasks
 *     PCE mask for each HSHUB
 *   [out] fbhubPceMask
 *     FBHUB PCE mask
 */

#define LW2080_CTRL_CMD_CE_GET_HUB_PCE_MASK             (0x20802a09) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_CE_INTERFACE_ID << 8) | LW2080_CTRL_CE_GET_HUB_PCE_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CE_MAX_HSHUBS                       5

#define LW2080_CTRL_CE_GET_HUB_PCE_MASK_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_CE_GET_HUB_PCE_MASK_PARAMS {
    LwU32 hshubPceMasks[LW2080_CTRL_CE_MAX_HSHUBS];
    LwU32 fbhubPceMask;
} LW2080_CTRL_CE_GET_HUB_PCE_MASK_PARAMS;

/* _ctrl2080ce_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

