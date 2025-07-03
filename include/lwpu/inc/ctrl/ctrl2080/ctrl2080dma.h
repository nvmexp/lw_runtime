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
// Source file: ctrl/ctrl2080/ctrl2080dma.finn
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

/* LW20_SUBDEVICE_XX dma control commands and parameters */

#include "ctrl2080common.h"

/*
 * LW2080_CTRL_CMD_DMA_ILWALIDATE_TLB
 *
 * This command ilwalidates the GPU TLB. This is intended to be used
 * by RM clients that manage their own TLB consistency when updating
 * page tables on their own, or with DEFER_TLB_ILWALIDATION options
 * to other RM APIs.
 *
 *    hVASpace
 *       This parameter specifies the VASpace object whose MMU TLB entries needs to be ilwalidated.
 *       Specifying a GMMU VASpace object handle will ilwalidate the GMMU TLB for the particular VASpace.
 *       Specifying a SMMU VASpace object handle will flush the entire SMMU TLB & PTC.
 *
 * This call can be used with the LW50_DEFERRED_API_CLASS (class 0x5080).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_TIMEOUT_RETRY
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_DMA_ILWALIDATE_TLB (0x20802502) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_DMA_INTERFACE_ID << 8) | LW2080_CTRL_DMA_ILWALIDATE_TLB_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_DMA_ILWALIDATE_TLB_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_DMA_ILWALIDATE_TLB_PARAMS {
    LwHandle hClient; // Deprecated. Kept here for compactibility with chips_GB9-2-1-1
    LwHandle hDevice; // Deprecated. Kept here for compactibility with chips_GB9-2-1-1
    LwU32    engine;  // Deprecated. Kept here for compactibility with chips_GB9-2-1-1
    LwHandle hVASpace;
} LW2080_CTRL_DMA_ILWALIDATE_TLB_PARAMS;

#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_GRAPHICS             0:0
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_GRAPHICS_FALSE    (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_GRAPHICS_TRUE     (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_VIDEO                1:1
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_VIDEO_FALSE       (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_VIDEO_TRUE        (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_DISPLAY              2:2
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_DISPLAY_FALSE     (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_DISPLAY_TRUE      (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_CAPTURE              3:3
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_CAPTURE_FALSE     (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_CAPTURE_TRUE      (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_IFB                  4:4
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_IFB_FALSE         (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_IFB_TRUE          (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_MV                   5:5
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_MV_FALSE          (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_MV_TRUE           (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_MPEG                 6:6
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_MPEG_FALSE        (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_MPEG_TRUE         (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_VLD                  7:7
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_VLD_FALSE         (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_VLD_TRUE          (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_ENCRYPTION           8:8
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_ENCRYPTION_FALSE  (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_ENCRYPTION_TRUE   (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_PERFMON              9:9
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_PERFMON_FALSE     (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_PERFMON_TRUE      (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_POSTPROCESS          10:10
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_POSTPROCESS_FALSE (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_POSTPROCESS_TRUE  (0x00000001)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_BAR                  11:11
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_BAR_FALSE         (0x00000000)
#define LW2080_CTRL_DMA_ILWALIDATE_TLB_ENGINE_BAR_TRUE          (0x00000001)

/*
 * LW2080_CTRL_DMA_INFO
 *
 * This structure represents a single 32bit dma engine value.  Clients
 * request a particular DMA engine value by specifying a unique dma
 * information index.
 *
 * Legal dma information index values are:
 *   LW2080_CTRL_DMA_INFO_INDEX_SYSTEM_ADDRESS_SIZE
 *     This index can be used to request the system address size in bits.
 */
typedef struct LW2080_CTRL_DMA_INFO {
    LwU32 index;
    LwU32 data;
} LW2080_CTRL_DMA_INFO;

/* valid dma info index values */
#define LW2080_CTRL_DMA_INFO_INDEX_SYSTEM_ADDRESS_SIZE (0x000000000)

/* set INDEX_MAX to greatest possible index value */
#define LW2080_CTRL_DMA_INFO_INDEX_MAX                 LW2080_CTRL_DMA_INFO_INDEX_SYSTEM_ADDRESS_SIZE

/*
 * LW2080_CTRL_CMD_DMA_GET_INFO
 *
 * This command returns dma engine information for the associated GPU.
 * Requests to retrieve dma information use an array of one or more
 * LW2080_CTRL_DMA_INFO structures.
 *
 *   dmaInfoTblSize
 *     This field specifies the number of valid entries in the dmaInfoList
 *     array.  This value cannot exceed LW2080_CTRL_DMA_GET_INFO_MAX_ENTRIES.
 *   dmaInfoTbl
 *     This parameter contains the client's dma info table into
 *     which the dma info values will be transferred by the RM.
 *     The dma info table is an array of LW2080_CTRL_DMA_INFO structures.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_DMA_GET_INFO                   (0x20802503) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_DMA_INTERFACE_ID << 8) | LW2080_CTRL_DMA_GET_INFO_PARAMS_MESSAGE_ID" */

/* maximum number of LW2080_CTRL_DMA_INFO entries per request */
#define LW2080_CTRL_DMA_GET_INFO_MAX_ENTRIES           (256)

#define LW2080_CTRL_DMA_GET_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_DMA_GET_INFO_PARAMS {
    LwU32                dmaInfoTblSize;
    /*
     * C form:
     * LW2080_CTRL_DMA_INFO dmaInfoTbl[LW2080_CTRL_DMA_GET_INFO_MAX_ENTRIES];
     */
    LW2080_CTRL_DMA_INFO dmaInfoTbl[LW2080_CTRL_DMA_GET_INFO_MAX_ENTRIES];
} LW2080_CTRL_DMA_GET_INFO_PARAMS;

typedef struct LW2080_CTRL_DMA_UPDATE_COMPTAG_INFO_TILE_INFO {
    /*! 
     * 64KB aligned address of source 64KB tile for comptag reswizzle.
     */
    LwU32 srcAddr;

    /*! 
     * 64KB aligned address of destination 64KB tile for comptag reswizzle.
     */
    LwU32 dstAddr;

    /*!
     * Comptag index assigned to the 64K sized tile relative to
     * the compcacheline. Absolute comptag index would be:
     * startComptagIndex + relComptagIndex.
     */
    LwU16 relComptagIndex;
} LW2080_CTRL_DMA_UPDATE_COMPTAG_INFO_TILE_INFO;

// _ctrl2080dma_h_
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

