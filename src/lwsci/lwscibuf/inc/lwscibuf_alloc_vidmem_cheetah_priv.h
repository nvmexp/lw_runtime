/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_VIDMEM_TEGRA_PRIV_H
#define INCLUDED_LWSCIBUF_ALLOC_VIDMEM_TEGRA_PRIV_H

#include "lwscibuf_alloc_vidmem.h"
#include "lwrm_gpu.h"

#define LW_SCI_BUF_GPU_CONTEXT_MAGIC        0x194582AB

typedef struct {
    uint32_t magic;
    const LwSciBufAllGpuContext* allGpuContext;
} LwSciBufAllocVidMemContext;

#endif /* INCLUDED_LWSCIBUF_ALLOC_VIDMEM_TEGRA_PRIV_H */
