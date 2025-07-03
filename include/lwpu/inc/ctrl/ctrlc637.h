/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrlc637.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* AMPERE_SMC_PARTITION_REF commands and parameters */

#define LWC637_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0xC637, LWC637_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWC637_CTRL_RESERVED        (0x00)
#define LWC637_CTRL_EXEC_PARTITIONS (0x01)


/*!
 * LWC637_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWC637_CTRL_CMD_NULL        (0xc6370000) /* finn: Evaluated from "(FINN_AMPERE_SMC_PARTITION_REF_RESERVED_INTERFACE_ID << 8) | 0x0" */

/*!
 * LWC637_CTRL_EXEC_PARTITIONS_INFO
 *
 * This structure specifies resources in an exelwtion partition
 *
 *  GpcCount[IN/OUT]
 *      - Number of GPCs in this partition
 *
 *  VeidCount[OUT]
 *      - Number of VEIDs available in this partition.
 *
 *  ceCount[IN/OUT]
 *      - Copy Engines in this partition
 *
 *  lwEncCount[IN/OUT]
 *      - Encoder Engines in this partition
 *
 *  lwDecCount[IN/OUT]
 *      - Decoder Engines in this partition
 *
 *  lwJpgCount[IN/OUT]
 *      - Jpg Engines in this partition
 *
 *  lwOfaCount[IN/OUT]
 *      - Ofa engines in this partition
 *
 *  sharedEngFlags[IN/OUT]
 *      - Flags determining whether above engines are shared with other exelwtion partitions
 *
 *  veidStartOffset[OUT]
 *      - VEID start offset within GPU partition
 */
typedef struct LWC637_CTRL_EXEC_PARTITIONS_INFO {
    LwU32 gpcCount;
    LwU32 veidCount;
    LwU32 ceCount;
    LwU32 lwEncCount;
    LwU32 lwDecCount;
    LwU32 lwJpgCount;
    LwU32 ofaCount;
    LwU32 sharedEngFlag;
    LwU32 veidStartOffset;
} LWC637_CTRL_EXEC_PARTITIONS_INFO;

#define LWC637_CTRL_EXEC_PARTITIONS_SHARED_FLAG         31:0
#define LWC637_CTRL_EXEC_PARTITIONS_SHARED_FLAG_NONE                      0x0
#define LWC637_CTRL_EXEC_PARTITIONS_SHARED_FLAG_CE      LWBIT(0)
#define LWC637_CTRL_EXEC_PARTITIONS_SHARED_FLAG_LWDEC   LWBIT(1)
#define LWC637_CTRL_EXEC_PARTITIONS_SHARED_FLAG_LWENC   LWBIT(2)
#define LWC637_CTRL_EXEC_PARTITIONS_SHARED_FLAG_OFA     LWBIT(3)
#define LWC637_CTRL_EXEC_PARTITIONS_SHARED_FLAG_LWJPG   LWBIT(4)

#define LWC637_CTRL_MAX_EXEC_PARTITIONS                                   8
#define LWC637_CTRL_EXEC_PARTITIONS_ID_ILWALID                            0xFFFFFFFF

/*!
 * LWC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS
 *
 * This command will create requested exelwtion partitions under the subscribed
 * memory partition. The memory partition is expected to be configured before
 * exelwtion partition creation.
 *
 * bQuery[IN]
 *      - If LW_TRUE, exelwtion partitions will not be created, but return
 *      status of LW_OK will indicate that the request is valid and can
 *      lwrrently be fulfilled
 * flag [IN]
 *      REQUEST_WITH_PART_ID
 *      - If set, RM will try to assign exelwtion partition id requested by clients.
 *      This flag is only supported on vGPU enabled RM build and will be removed
 *      when vgpu plugin implements virtualized exelwtion partition ID support.
 *      (bug 2938187)
 *
 * execPartCount[IN]
 *      - Number of exelwtion partitions requested
 *
 * execPartInfo[IN]
 *      - Requested exelwtion partition resources for each requested partition
 *
 * execPartId[OUT]
 *      - ID of each requested exelwtion partition
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LWC637_CTRL_DMA_EXEC_PARTITIONS_CREATE_REQUEST_WITH_PART_ID                   0:0
#define LWC637_CTRL_DMA_EXEC_PARTITIONS_CREATE_REQUEST_WITH_PART_ID_FALSE (0x00000000)
#define LWC637_CTRL_DMA_EXEC_PARTITIONS_CREATE_REQUEST_WITH_PART_ID_TRUE  (0x00000001)





#define LWC637_CTRL_CMD_EXEC_PARTITIONS_CREATE (0xc6370101) /* finn: Evaluated from "(FINN_AMPERE_SMC_PARTITION_REF_EXEC_PARTITIONS_INTERFACE_ID << 8) | LWC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS_MESSAGE_ID" */

/*!
 * LWC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS
 *
 * This command will delete requested exelwtion partitions.
 *
 * execPartCount[IN]
 *      - Number of exelwtion partitions to delete.
 *
 * execPartId[IN]
 *      - Exelwtion partition IDs to delete
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LWC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS {
    LwBool                           bQuery;
    LwU32                            flags;
    LwU32                            execPartCount;
    // C form: LWC637_CTRL_EXEC_PARTITIONS_INFO execPartInfo[LWC637_CTRL_MAX_EXEC_PARTITIONS];
    LWC637_CTRL_EXEC_PARTITIONS_INFO execPartInfo[LWC637_CTRL_MAX_EXEC_PARTITIONS];
    // C form: LwU32 execPartId[LWC637_CTRL_MAX_EXELWTION_PARTITIONS];
    LwU32                            execPartId[LWC637_CTRL_MAX_EXEC_PARTITIONS];
} LWC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS;
#define LWC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS {
    LwU32 execPartCount;
    LwU32 execPartId[LWC637_CTRL_MAX_EXEC_PARTITIONS];
} LWC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS;

#define LWC637_CTRL_CMD_EXEC_PARTITIONS_DELETE (0xc6370102) /* finn: Evaluated from "(FINN_AMPERE_SMC_PARTITION_REF_EXEC_PARTITIONS_INTERFACE_ID << 8) | LWC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS_MESSAGE_ID" */

/*!
 * LWC637_CTRL_EXEC_PARTITIONS_GET_PARAMS
 *
 * This command will return information about exelwtion partitions which
 * lwrrently exist within the subscribed memory partition.
 *
 * execPartCount[OUT]
 *      - Number of existing exelwtion partitions
 *
 * execPartId[OUT]
 *      - ID of existing exelwtion partitions
 *
 * execPartInfo[OUT]
 *      - Resources within each existing exelwtion partition
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LWC637_CTRL_EXEC_PARTITIONS_GET_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC637_CTRL_EXEC_PARTITIONS_GET_PARAMS {
    LwU32                            execPartCount;
    // C form: LwU32 execPartId[LWC637_CTRL_MAX_EXELWTION_PARTITIONS];
    LwU32                            execPartId[LWC637_CTRL_MAX_EXEC_PARTITIONS];
    // C form: LWC637_CTRL_EXEC_PARTITIONS_INFO execPartInfo[LWC637_CTRL_MAX_EXEC_PARTITIONS];
    LWC637_CTRL_EXEC_PARTITIONS_INFO execPartInfo[LWC637_CTRL_MAX_EXEC_PARTITIONS];
} LWC637_CTRL_EXEC_PARTITIONS_GET_PARAMS;

#define LWC637_CTRL_CMD_EXEC_PARTITIONS_GET (0xc6370103) /* finn: Evaluated from "(FINN_AMPERE_SMC_PARTITION_REF_EXEC_PARTITIONS_INTERFACE_ID << 8) | LWC637_CTRL_EXEC_PARTITIONS_GET_PARAMS_MESSAGE_ID" */

/*!
 * LWC637_CTRL_EXEC_PARTITIONS_GET_ACTIVE_IDS
 *
 * This command will return IDs of all active exelwtion partitions in a memory
 * partition
 *
 * execPartCount[OUT]
 *      - Number of existing exelwtion partitions
 *
 * execPartId[OUT]
 *      - ID of existing exelwtion partitions
 *
 * execPartUuid[OUT]
 *      - ASCII UUID string of existing exelwtion partitions
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

/* 'M' 'I' 'G' '-'(x5), '\0x0', extra = 9 */
#define LWC637_UUID_LEN                     16
#define LWC637_UUID_STR_LEN                 (0x29) /* finn: Evaluated from "((LWC637_UUID_LEN << 1) + 9)" */

typedef struct LWC637_EXEC_PARTITION_UUID {
    // C form: char str[LWC638_UUID_STR_LEN];
    char str[LWC637_UUID_STR_LEN];
} LWC637_EXEC_PARTITION_UUID;

#define LWC637_CTRL_EXEC_PARTITIONS_GET_ACTIVE_IDS_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWC637_CTRL_EXEC_PARTITIONS_GET_ACTIVE_IDS_PARAMS {
    LwU32                      execPartCount;

    // C form: LwU32 execPartId[LWC637_CTRL_MAX_EXELWTION_PARTITIONS];
    LwU32                      execPartId[LWC637_CTRL_MAX_EXEC_PARTITIONS];

    // C form: LWC637_EXEC_PARTITION_UUID execPartUuid[LWC637_CTRL_MAX_EXEC_PARTITIONS];
    LWC637_EXEC_PARTITION_UUID execPartUuid[LWC637_CTRL_MAX_EXEC_PARTITIONS];
} LWC637_CTRL_EXEC_PARTITIONS_GET_ACTIVE_IDS_PARAMS;

#define LWC637_CTRL_EXEC_PARTITIONS_GET_ACTIVE_IDS               (0xc6370104) /* finn: Evaluated from "(FINN_AMPERE_SMC_PARTITION_REF_EXEC_PARTITIONS_INTERFACE_ID << 8) | LWC637_CTRL_EXEC_PARTITIONS_GET_ACTIVE_IDS_PARAMS_MESSAGE_ID" */

/*
 * LWC637_CTRL_CMD_EXEC_PARTITIONS_EXPORT 
 *
 * Export the resource and placement information about an exec partition such
 * that a similar partition can be recreated from scratch in the same position.
 */
#define LWC637_CTRL_CMD_EXEC_PARTITIONS_EXPORT                   (0xc6370105) /* finn: Evaluated from "(FINN_AMPERE_SMC_PARTITION_REF_EXEC_PARTITIONS_INTERFACE_ID << 8) | 0x5" */

/*
 * LWC637_CTRL_CMD_EXEC_PARTITIONS_IMPORT 
 *
 * Create an exec partition resembling the exported partition info. The imported
 * partition should behave identically with respect to fragmentation.
 */
#define LWC637_CTRL_CMD_EXEC_PARTITIONS_IMPORT                   (0xc6370106) /* finn: Evaluated from "(FINN_AMPERE_SMC_PARTITION_REF_EXEC_PARTITIONS_INTERFACE_ID << 8) | 0x6" */

#define LWC637_CTRL_EXEC_PARTITIONS_EXPORT_MAX_ENGINES_MASK_SIZE 4
typedef struct LWC637_CTRL_EXEC_PARTITIONS_EXPORTED_INFO {
    LW_DECLARE_ALIGNED(LwU64 enginesMask[LWC637_CTRL_EXEC_PARTITIONS_EXPORT_MAX_ENGINES_MASK_SIZE], 8);
    LwU8  uuid[LWC637_UUID_LEN];
    LwU32 sharedEngFlags;
    LwU32 gpcMask;
    LwU32 veidOffset;
    LwU32 veidCount;
} LWC637_CTRL_EXEC_PARTITIONS_EXPORTED_INFO;

typedef struct LWC637_CTRL_EXEC_PARTITIONS_IMPORT_EXPORT_PARAMS {
    LwU32 id;
    LW_DECLARE_ALIGNED(LWC637_CTRL_EXEC_PARTITIONS_EXPORTED_INFO info, 8);
} LWC637_CTRL_EXEC_PARTITIONS_IMPORT_EXPORT_PARAMS;

//  _ctrlc637_h_
