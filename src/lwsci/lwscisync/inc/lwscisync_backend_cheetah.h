/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCISYNC_BACKEND_TEGRA_H
#define INCLUDED_LWSCISYNC_BACKEND_TEGRA_H

#include "lwscisync_backend.h"
#ifdef LW_TEGRA_MIRROR_INCLUDES
//cheetah build from perforce tree - use mobile_common.h
#include "mobile_common.h"
#else
//cheetah build from git tree - use lwrm_host1x_safe.h
#include "lwrm_host1x_safe.h"
#endif

/**
 * \page lwscisync_page_unit_blanket_statements LwSciSync blanket statements
 * \section lwscisync_element_dependency Dependency on other elements
 * LwSciSync calls below LwHost interfaces:
 * - LwRmHost1xGetDefaultOpenAttrs() to get default attributes for
 * LwRmHost1xOpen().
 * - LwRmHost1xOpen() to get a new Host1x handle.
 * - LwRmHost1xClose() to close the Host1x handle.
 * - LwRmHost1xWaiterAllocate() to allocate a waiter object.
 * - LwRmHost1xWaiterFree() to free an allocated waiter.
 *
 * \implements{18844104}
 */

/**
 * \brief Retrieves the LwRmHost1xHandle held by LwSciSyncCoreRmBackEnd
 *
 * \param[in] backEnd LwSciSyncCoreRmBackEnd to retrieve LwRmHost1xHandle from
 *
 * \return LwRmHost1xHandle
 * - LwRmHost1xHandle from the @a backend
 * - Panics if @a backEnd is NULL
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - None of the access to either global or local objects requires thread
 *   synchronization.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844845}
 */
LwRmHost1xHandle LwSciSyncCoreRmGetHost1xHandle(
    LwSciSyncCoreRmBackEnd backEnd);

/**
 * \brief Retrieves LwRmHost1xWaiterHandle held by
 * LwSciSyncCoreRmWaitContextBackEnd
 *
 * \param[in] waitContextBackEnd The LwSciSyncCoreRmWaitContextBackEnd to
 * retrieve LwRmHost1xWaiterHandle from
 *
 * \return LwRmHost1xWaiterHandle
 * - LwRmHost1xWaiterHandle from the @a backend
 * - Panics if @a waitContextBackEnd is NULL
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - None of the access to either global or local objects requires thread
 *   synchronization.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844848}
 */
LwRmHost1xWaiterHandle LwSciSyncCoreRmWaitCtxGetWaiterHandle(
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd);

#endif
