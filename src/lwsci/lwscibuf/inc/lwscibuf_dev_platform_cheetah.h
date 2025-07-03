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
 * \brief <b>LwSciBuf CheetAh Device Interfaces</b>
 *
 * @b Description: This file contains LwSciBuf CheetAh Platform interfaces
 *                 and data structures definitions.
 */

#ifndef INCLUDED_LWSCIBUF_DEV_PLATFORM_TEGRA_H
#define INCLUDED_LWSCIBUF_DEV_PLATFORM_TEGRA_H

#ifdef LW_TEGRA_MIRROR_INCLUDES
#include "mobile_common.h"
#endif
#include "lwrm_gpu.h"

#include "lwscibuf.h"
#include "lwscilog.h"
#include "lwscicommon_libc.h"

/**
 * @addtogroup lwscibuf_blanket_statements
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements
 * \section lwscibuf_in_params Input parameters
 * - LwSciBufDev passed as input parameter to an API is valid input if it is
 * returned from a successful call to LwSciBufDevOpen() and has not yet been
 * deallocated using LwSciBufDevClose().
 */

/**
 * @}
 */

/**
 * \brief Defines opaque pointer to LwSciBufDevRec structure
 */
typedef struct LwSciBufDevRec* LwSciBufDev;

/**
 * \brief Structure holding information of each individual GPU
 *        in the system obtained by calling LwRmGpu interfaces.
 *
 * Synchronization: Modification to an instance of this datatype must be
 * externally synchronized
 *
 * \implements{18842133}
 */
typedef struct {
    /** LwRmGpu device handle for GPU device at LwRmGpu specific
     * index defined by deviceIndex member of gpuList array member
     * of LwSciBufAllGpuContext structure. This member is initialized
     * by calling LwRmGpuDeviceOpen() when LwSciBufAllGpuContext is
     * initialized. It should be set to NULL if LwRmGpuDeviceOpen()
     * fails. This member is deinitialized by calling LwRmGpuDeviceClose()
     * when LwSciBufAllGpuContext is deinitialized.
     */
    LwRmGpuDevice* gpuDevice;
    /** LwRmGpu specific information pertaining to GPU device handle
     * represented by gpuDevice member. This member is initialized when
     * LwSciBufAllGpuContext is initialized. It is initialized by passing
     * gpuDevice member to LwRmGpuDeviceGetInfo(). If LwRmGpuDeviceGetInfo()
     * returns NULL, gpuDevice handle must be closed by calling
     * LwRmGpuDeviceClose() and gpuDevice member must be set to NULL. This
     * member is deinitialized by setting it to NULL when LwSciBufAllGpuContext
     * is deinitialized.
     */
    const LwRmGpuDeviceInfo* gpuDeviceInfo;
} LwSciBufPerGpuContext;

/**
 * \brief Structure holding information of all the GPUs
 *        in the system which is obtained by calling
 *        LwRmGpu interfaces.
 *
 * Synchronization: Modification to an instance of this datatype must be
 * externally synchronized
 *
 * \implements{18842136}
 */
typedef struct {
    /** Pointer to LwRmGpuLib structure provided by LwRmGpu representing
     * instance of LwRmGpu library. LwRmGpu APIs should be called by using
     * this instance. This member is initialized by calling LwRmGpuLibOpen()
     * when LwSciBufDev is allocated. It must be initialized before any LwRmGpu
     * APIs (other than LwRmGpuLibOpen) can be called for this library instance.
     * This member is deinitialized by calling LwRmGpuLibClose() when LwSciBufDev
     * is deallocated. LwRmGpu APIs must not be called for the instance after it
     * is deinitialized.
     */
    LwRmGpuLib* gpuLib;

    /** Read-only array of LwRmGpuLibDeviceListEntry structure provided by LwRmGpu.
     * This structure contains LwRmGpu related data for GPUs needed to call other
     * LwRmGpu APIs. The number of members in this array is represented by gpuListSize
     * member. deviceIndex member of LwRmGpuLibDeviceListEntry structure defines
     * LwRmGpu specific deviceIndex assigned to every GPU. This member is initialized
     * by calling LwRmGpuLibListDevices() when LwSciBufDev is allocated. This member
     * must be set to NULL during initialization if gpuListSize is 0. This member is
     * deinitialized by setting it to NULL when LwSciBufDev is deallocated.
     */
    const LwRmGpuLibDeviceListEntry* gpuList;

    /** Number of GPUs in the system. This member is initialized by calling
     * LwRmGpuLibListDevices() when LwSciBufDev is allocated. This member is
     * deinitialized by setting it to 0 when LwSciBufDev is deallocated.
     */
    size_t gpuListSize;

    /** An array of LwSciBufPerGpuContext. The number of members in this array is
     * represented by gpuListSize member. This member is initialized when LwSciBufDev
     * is allocated. If gpuListSize is 0, this member should be initialized to NULL.
     * Otherwise, this member is initialized by allocating memory using LwSciCommon
     * functionality for array members corresponding to number of GPUs in the system
     * represented by gpuListSize and initializing LwSciBufPerGpuContext members for
     * each GPU. If initialization of LwSciBufPerGpuContext members for a particular
     * GPU fails, we should continue initializing this structure for other GPUs. This
     * member is deinitialized when LwSciBufDev is deallocated. It is deinitialized by
     * deinitializing LwSciBufPerGpuContext members for each GPU. If deinitialization
     * of LwSciBufPerGpuContext members for a particular GPU fails, we should continue
     * deinitializing this structure for other GPUs. Memory for this member should be
     * deallocated once deinitialization of LwSciBufPerGpuContext members is completed.
     */
    LwSciBufPerGpuContext* perGpuContext;
} LwSciBufAllGpuContext;

/**
 * \brief Allocates a new LwSciBufDev which can be used by other units
 *        for the subsequent operations and initializes LwSciBufAllGpuContext.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[out] newDev The new LwSciBufDev.
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_ResourceError if failed to open GPU device(s).
 * - panics if if newDev is NULL.
 *
 * \implements{18842874}
 */
LwSciError LwSciBufDevOpen(
    LwSciBufDev* newDev);

/**
 * \brief Frees LwSciBufDev and deinitializes LwSciBufAllGpuContext.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufDev is not used by
 *        multiple threads at the same time
 *
 * \param[in] dev The LwSciBufDev to be freed.
 *
 * \return void
 * - panics if dev is NULL.
 *
 * \implements{18842877}
 */
void LwSciBufDevClose(
    LwSciBufDev dev);

/**
 * \brief Retrieves LwSciBufAllGpuContext from LwSciBufDev.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufAllGpuContext
 *        member of the LwSciBufDev is never modified after creation (so there
 *        is no data-dependency)
 *
 * \param[in] dev LwSciBufDev from which the LwSciBufAllGpuContext
 *  should be retrieved.
 *
 * \param[out] allGpuContext LwSciBufAllGpuContext.
 *
 * \return void
 * - panics if @a dev or @a allGpuContext is NULL.
 *
 * \implements{18842880}
 */
void LwSciBufDevGetAllGpuContext(
    LwSciBufDev dev,
    const LwSciBufAllGpuContext** allGpuContext);

//TODO: Add doxygen comments
void LwSciBufDevGetGpuDeviceInfo(
    LwSciBufDev dev,
    LwSciRmGpuId gpuId,
    const LwRmGpuDeviceInfo** gpuDeviceInfo);

/**
 * \brief Checks platform version's compatibility. On cheetah, there is
 *        no need to check compatibility with lower layer APIs such as
 *        memory services and thus, this function always returns true.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[out] platformCompatibility boolean value depicting if lower
 *  level APIs for platform are compatible or not.
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if platformCompatibility is NULL
 *
 * \implements{18842883}
 */
LwSciError LwSciBufCheckPlatformVersionCompatibility(
    bool* platformCompatibility);

#endif /* INCLUDED_LWSCIBUF_DEV_PLATFORM_TEGRA_H */
