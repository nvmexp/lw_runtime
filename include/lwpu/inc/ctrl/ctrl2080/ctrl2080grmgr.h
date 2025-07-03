/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080grmgr.finn
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

/* LW20_SUBDEVICE_XX grmgr control commands and parameters */

//
// LW2080_CTRL_CMD_GRMGR_GET_GR_FS_INFO
//
// This control call works as a batched query interface where we
// have multiple different queries that can be passed in
// and RM will return the associated data and status type
// If there is any error in LW2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS,
// we will immediately fail the call.
// However, if there is an error in the query-specific calls, we will
// log the error and march on.
//
// LW2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS
//  numQueries[IN]
//      - Specifies the number of valid queries that the caller will be passing in
//
// Possible status values returned are:
//   LW_OK
//   LW_ERR_ILWALID_ARGUMENT
//   LW_ERR_ILWALID_STATE
//
#define LW2080_CTRL_CMD_GRMGR_GET_GR_FS_INFO        (0x20803801) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GRMGR_INTERFACE_ID << 8) | LW2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS_MESSAGE_ID" */

// Max number of queries that can be batched in a single call to LW2080_CTRL_CMD_GRMGR_GET_GR_FS_INFO
#define LW2080_CTRL_GRMGR_GR_FS_INFO_MAX_QUERIES    96

//
// Preference is to keep max.size of union at 24 bytes (i.e. 6 32-bit members)
// so that the size of entire query struct is maintained at 32 bytes, to ensure
// that overall params struct does not exceed 4kB
//
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_MAX_SIZE 32
#define LW2080_CTRL_GRMGR_MAX_SMC_IDS               8

/*!
 * LW2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS
 *  gpcCount[OUT]
 *      - No. of logical/local GPCs which client can use to create the
 *        logical/local mask respectively
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS {
    LwU32 gpcCount;         // param[out] - logical/local GPC mask
} LW2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS;

/*!
 * LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS
 *  gpcId[IN]
 *      - Logical/local GPC ID
 *  chipletGpcMap[OUT]
 *      - Returns chiplet GPC ID for legacy case and device monitoring client
 *      - Returns local GPC ID (== input gpcId) for SMC client
 *      - Does not support DM attribution case
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS {
    LwU32 gpcId;            // param[in] - logical/local GPC ID
    LwU32 chipletGpcMap;    // param[out] - chiplet GPC ID
} LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS;

/*!
 * LW2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS
 *  gpcId[IN]
 *      - Logical/local GPC ID
 *  tpcMask[OUT]
 *      - Returns physical TPC mask for legacy, DM client and SMC cases
 *      - Does not support DM attribution case
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS {
    LwU32 gpcId;            // param[in] - logical/local GPC ID
    LwU32 tpcMask;          // param[out] - physical TPC mask
} LW2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS;

/*!
 * LW2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS
 *  gpcId[IN]
 *      - Logical/local GPC ID
 *  ppcMask[OUT]
 *      - Returns physical PPC mask for legacy, DM client and SMC cases
 *      - Does not support DM attribution case
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS {
    LwU32 gpcId;            // param[in] - logical/local GPC ID
    LwU32 ppcMask;          // param[out] - physical PPC mask
} LW2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS;

/*!
 *  !!! DEPRECATED - This query will return LW_ERR_NOT_SUPPORTED since deleting
 *               it would break driver compatibility !!!
 *
 *  LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS
 *  swizzId[IN]
 *      - Swizz ID of partition
 *      - A DM client with an invalid swizz ID, will fail this call
 *      - This parameter is not compulsory for an SMC client; the subscription
 *        itself will do the necessary validation.
 *  gpcId[IN]
 *      - Logical/local GPC ID
 *  chipletGpcMap[OUT]
 *      - Returns chiplet GPC ID for legacy case and device monitoring client
 *      - Returns local GPC ID (== input gpcId) for SMC client
 *      - Does not support non-attribution case for DM client
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS {
    LwU32 swizzId;          // param[in]  - swizz ID of partition
    LwU32 gpcId;            // param[in]  - logical/local GPC ID
    LwU32 chipletGpcMap;    // param[out] - chiplet GPC ID
} LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS;

/*!
 * LW2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS
 *  gpcId[IN]
 *      - Logical/local GPC ID
 *  ropMask[OUT]
 *      - Returns physical ROP mask for legacy, DM client
 *      - Returns logical ROP mask for SMC
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS {
    LwU32 gpcId;            // param[in] - logical/local GPC ID
    LwU32 ropMask;          // param[out] - physical ROP mask
} LW2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS;

/*!
 *  LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS
 *  chipletSyspipeMask [OUT]
 *      - Mask of chiplet SMC-IDs for DM client attribution case
 *      - Mask of local SMC-IDs for SMC client
 *      - Legacy case returns 1 GR
 *      - Does not support attribution case for DM client
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS {
    LwU32 chipletSyspipeMask;   // param[out] - Mask of chiplet SMC IDs
} LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS;

/*!
 *  LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS
 *  swizzId[IN]
 *      - Swizz ID of partition
 *      - A DM client with an invalid swizz ID, will fail this call
 *  physSyspipeId[GRMGR_MAX_SMC_IDS] [OUT]
 *      - Physical SMC-IDs mapped to partition local idx for DM client attribution case
 *      - Does not support non-attribution case for DM client, SMC clients, legacy case
 *  physSyspipeIdCount[OUT]
 *      - Valid count of physSmcIds which has been populated in above array.
 *      - Failure case will return 0
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS {
    LwU16 swizzId;                                          // param[in]  - swizz ID of partition
    LwU16 physSyspipeIdCount;                               // param[out] - Count of physSmcIds in above array
    LwU8  physSyspipeId[LW2080_CTRL_GRMGR_MAX_SMC_IDS];      // param[out] - physical/local SMC IDs
} LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS;

/*!
 * LW2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS
 * swizzId[IN]
 *      - Swizz ID of partition
 *      - Mandatory parameter
 *      - A DM client with an invalid swizz ID, will fail this call
 * grIdx[IN]
 *      - Local grIdx for a partition
 *      - Mandatory parameter
 * gpcEnMask[OUT]
 *      - Logical enabled GPC mask associated with requested grIdx of the partition i.e swizzid->engineId->gpcMask
 *      - These Ids should be used as input further
 *      - Does not support non-attribution case for DM client, SMC clients, legacy case
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS {
    LwU32 swizzId;      // param[in]  - swizz ID of partition
    LwU32 grIdx;        // param[in]  - partition local GR ID
    LwU32 gpcEnMask;    // param[out] - logical enabled GPC mask
} LW2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS;

/*!
 * LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID
 * syspipeId[OUT]
 *      - Partition-local GR idx for client subscribed to exec partition
 *      - Does not support legacy case, DM client, or SMC client subscribed only to partition
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS {
    LwU32 syspipeId;    // param[out] - partition-local Gr idx
} LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS;

/*!
 * queryType[IN]
 *  - Use queryType defines to specify what information is being requested
 * status[OUT]
 *  - Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_ILWALID_STATE
 */
typedef struct LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS {
    LwU16 queryType;
    LwU8  reserved[2]; // To keep the struct aligned for now and available for future use (if needed)
    LwU32 status;
    union {
        LW2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS                     gpcCountData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS               chipletGpcMapData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS                      tpcMaskData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS                      ppcMaskData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS     partitionGpcMapData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS          syspipeMaskData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS partitionChipletSyspipeData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS         dmGpcMaskData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS          partitionSyspipeIdData;
        LW2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS                      ropMaskData;
    } queryData;
} LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS;

#define LW2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS {
    LwU16                                     numQueries;
    LwU8                                      reserved[6]; // To keep the struct aligned for now and available for future use (if needed)
    LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS queries[LW2080_CTRL_GRMGR_GR_FS_INFO_MAX_QUERIES];
} LW2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS;

#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_ILWALID                       0
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_GPC_COUNT                     1
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_CHIPLET_GPC_MAP               2
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_TPC_MASK                      3
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PPC_MASK                      4
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARTITION_CHIPLET_GPC_MAP     5   /* deprecated */
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_CHIPLET_SYSPIPE_MASK          6
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARTITION_CHIPLET_SYSPIPE_IDS 7
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PROFILER_MON_GPC_MASK         8
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARTITION_SYSPIPE_ID          9
#define LW2080_CTRL_GRMGR_GR_FS_INFO_QUERY_ROP_MASK                      10

/* _ctrl2080grmgr_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

