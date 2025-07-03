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

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0000/ctrl0000syncgpuboost.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0000/ctrl0000base.h"

#include "ctrl/ctrlxxxx.h"
#include "lwtypes.h"
#include "lwlimits.h"

/* --------------------------- Macros ----------------------------------------*/
// There are at least 2 GPUs in a sync group. Hence max is half of max devices.
#define LW0000_SYNC_GPU_BOOST_MAX_GROUPS       (0x10) /* finn: Evaluated from "((LW_MAX_DEVICES) >> 1)" */
#define LW0000_SYNC_GPU_BOOST_ILWALID_GROUP_ID 0xFFFFFFFF

/*-------------------------Command Prototypes---------------------------------*/

/*!
 * Query whether SYNC GPU BOOST MANAGER is enabled or disabled.
 */
#define LW0000_CTRL_CMD_SYNC_GPU_BOOST_INFO    (0xa01) /* finn: Evaluated from "(FINN_LW01_ROOT_SYNC_GPU_BOOST_INTERFACE_ID << 8) | LW0000_SYNC_GPU_BOOST_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_SYNC_GPU_BOOST_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_SYNC_GPU_BOOST_INFO_PARAMS {
    // [out] Specifies if Sync Gpu Boost Manager is enabled or not.
    LwBool bEnabled;
} LW0000_SYNC_GPU_BOOST_INFO_PARAMS;

/*!
 *  Creates a Synchronized GPU-Boost Group (SGBG)
 */
#define LW0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_CREATE (0xa02) /* finn: Evaluated from "(FINN_LW01_ROOT_SYNC_GPU_BOOST_INTERFACE_ID << 8) | LW0000_SYNC_GPU_BOOST_GROUP_CREATE_PARAMS_MESSAGE_ID" */

/*! 
 * Describes a Synchronized GPU-Boost Group configuration
 */
typedef struct LW0000_SYNC_GPU_BOOST_GROUP_CONFIG {
    // [in] Number of elements in @ref gpuIds
    LwU32  gpuCount;

    // [in] IDs of GPUs to be put in the Sync Boost Group
    LwU32  gpuIds[LW_MAX_DEVICES];

    // [out] Unique ID of the SGBG, if created
    LwU32  boostGroupId;

    // [in] If this group represents  bridgeless SLI
    LwBool bBridgeless;
} LW0000_SYNC_GPU_BOOST_GROUP_CONFIG;

#define LW0000_SYNC_GPU_BOOST_GROUP_CREATE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_SYNC_GPU_BOOST_GROUP_CREATE_PARAMS {
    LW0000_SYNC_GPU_BOOST_GROUP_CONFIG boostConfig;
} LW0000_SYNC_GPU_BOOST_GROUP_CREATE_PARAMS;

/*!
 *  Destroys a previously created Synchronized GPU-Boost Group(SGBG)
 */
#define LW0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_DESTROY (0xa03) /* finn: Evaluated from "(FINN_LW01_ROOT_SYNC_GPU_BOOST_INTERFACE_ID << 8) | LW0000_SYNC_GPU_BOOST_GROUP_DESTROY_PARAMS_MESSAGE_ID" */

#define LW0000_SYNC_GPU_BOOST_GROUP_DESTROY_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0000_SYNC_GPU_BOOST_GROUP_DESTROY_PARAMS {
    // [[in] Unique ID of the SGBG to be destroyed
    LwU32 boostGroupId;
} LW0000_SYNC_GPU_BOOST_GROUP_DESTROY_PARAMS;

/*!
 * Get configuration information for all Synchronized Boost Groups in the system.
 */
#define LW0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO (0xa04) /* finn: Evaluated from "(FINN_LW01_ROOT_SYNC_GPU_BOOST_INTERFACE_ID << 8) | LW0000_SYNC_GPU_BOOST_GROUP_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_SYNC_GPU_BOOST_GROUP_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0000_SYNC_GPU_BOOST_GROUP_INFO_PARAMS {
    // [out] Number of groups retrieved. @ref LW0000_SYNC_GPU_BOOST_GROUP_INFO_PARAMS::boostGroups
    LwU32                              groupCount;

    // [out] @ref LW0000_SYNC_GPU_BOOST_GROUP_CONFIG
    LW0000_SYNC_GPU_BOOST_GROUP_CONFIG pBoostGroups[LW0000_SYNC_GPU_BOOST_MAX_GROUPS];
} LW0000_SYNC_GPU_BOOST_GROUP_INFO_PARAMS;

/* _ctrl0000syncgpuboost_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

