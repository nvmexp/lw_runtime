/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync syncpoint primitive private structures</b>
 *
 * @b Description: This file declares core structures of syncpoint
 */

#ifndef INCLUDED_LWSCISYNC_SYNCPOINT_CORE_H
#define INCLUDED_LWSCISYNC_SYNCPOINT_CORE_H

#ifdef LW_TEGRA_MIRROR_INCLUDES
//cheetah build from perforce tree - use mobile_common.h
#include "mobile_common.h"
#else
//cheetah build from git tree - use lwrm_host1x_safe.h
#include "lwrm_host1x_safe.h"
#endif

#include "lwscisync_c2c_priv.h"

/**
 * \brief Represents LwSciSync core syncpoint structure
 */
typedef struct {
    /** Syncpoint handle */
    LwRmHost1xSyncpointHandle syncpt;
    /** Host1x handle to interact with the safety API. It's a reference
     *  to the one in the RmBackEnd. */
    LwRmHost1xHandle host1x;
    /** Array of ids if ids are known */
    uint64_t* ids;
    /** Number of ids in above array */
    size_t numIds;
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    /** C2C handle */
    LwSciC2cPcieSyncHandle syncHandle;
    /** CPU accessible memory for signaling */
    uint32_t* memShim;
    /** functions exposed by C2C */
    LwSciC2cCopyFuncs c2cCopyFuncs;
#endif
} LwSciSyncCoreSyncpointInfo;

#endif
