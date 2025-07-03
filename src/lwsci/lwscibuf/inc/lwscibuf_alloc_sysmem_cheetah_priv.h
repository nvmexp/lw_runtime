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
 * \brief <b> LwSciBuf CheetAh Sysmem Interface Internal Data Structures Definitions <b>
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_SYSMEM_TEGRA_PRIV_H
#define INCLUDED_LWSCIBUF_ALLOC_SYSMEM_TEGRA_PRIV_H

#include "lwscibuf_alloc_sysmem.h"

#define LWSCIBUF_ALLOC_SYSMEM_MAX_STR_SIZE      64
#define LWSCIBUF_SYSMEM_CONTEXT_MAGIC           0x123582ABU

typedef struct {
    /**
     * Magic ID used to detect if LwSciBufAllocSysMemContext is valid.
     *
     * The Magic ID must be initialized to the constant
     * LWSCIBUF_SYSMEM_CONTEXT_MAGIC when an LwSciBufAllocSysMemContext is
     * allocated.
     *
     * It must be deinitialized to any value within range of [0, UINT32_MAX]
     * which is not equal to LWSCIBUF_SYSMEM_CONTEXT_MAGIC constant value.
     *
     * This member must NOT be modified during the operational lifetime of
     * LwSciBufAllocSysMemContext except when it is allocated or destroyed as
     * described above.
     *
     * Whenever an LwSciBufAllocSysMemContext is received from outside the
     * CheetAh Sysmem Interface unit, this unit must validate the Magic ID.
     */
    uint32_t magic;

    /**
     * This member represents permissions, it means which GPU have access to operate the memory.
     */
    LwSciBufAllocSysGpuAccessType gpuAccessParam;

    /**
     * LwSciBufAllGpuContext stores LwRmGpu information of all the GPUs
     * connected to the system. This information is needed during
     * VidMem operations and SysMem allocation.
     */
    const LwSciBufAllGpuContext* allGpuContext;
} LwSciBufAllocSysMemContext;

typedef struct {
    /** Name of the heap. */
    char heapName[LWSCIBUF_ALLOC_SYSMEM_MAX_STR_SIZE];
    /** Specific LwRmHeap. */
    LwRmHeap allocLwRmHeap;
} LwSciBufAllocSysMemToLwRmHeapMap;

typedef struct {
    /** Name of GPU access. */
    char gpuAccessTypeName[LWSCIBUF_ALLOC_SYSMEM_MAX_STR_SIZE];
    /** Specific LwRmGpuAccess. */
    LwRmHeap allocLwRmGpuAccess;
} LwSciBufAllocSysMemToLwRmGpuAccessMap;

#endif /* INCLUDED_LWSCIBUF_ALLOC_SYSMEM_TEGRA_PRIV_H */
