/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciBuf CheetAh Platform Internal Data Structures Definitions</b>
 */

#ifndef INCLUDED_LWSCIBUF_DEV_PLATFORM_TEGRA_PRIV_H
#define INCLUDED_LWSCIBUF_DEV_PLATFORM_TEGRA_PRIV_H

#include "lwscibuf_dev_platform_tegra.h"

/**
 * \brief Structure that LwSciBufDev points to.
 *        This structure is allocated and deallocated
 *        using LwSciCommon functionality.
 */
typedef struct LwSciBufDevRec {
    /** LwSciBufAllGpuContext stores LwRmGpu information of all the GPUs
     *  connected to the system. This information is needed during
     *  VidMem operations and SysMem allocation.
     */
    LwSciBufAllGpuContext allGpuContext;
} LwSciBufDevPriv;

#endif /* INCLUDED_LWSCIBUF_DEV_PLATFORM_TEGRA_PRIV_H */
