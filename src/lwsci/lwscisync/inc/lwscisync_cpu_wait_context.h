/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCISYNC_CPU_WAIT_CONTEXT_H
#define INCLUDED_LWSCISYNC_CPU_WAIT_CONTEXT_H

#include "lwscisync_backend.h"

/**
 * @defgroup lwsci_sync Synchronization APIs
 * @{
 */

/**
 * LwSciSyncCpuWaitContext is allocated using LwSciCommon functionality and
 * then initialized.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Allocation of LwSciSyncCpuWaitContext requires thread synchronization;
 *   without synchronization, the function could cause a memory leak when
 *   called in parallel from multiple threads. No synchronization is done in
 *   the function. To ensure that no memory leak oclwrs, the user must ensure
 *   that the function is not called in parallel from multiple threads.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844437}
 *
 * \fn LwSciError LwSciSyncCpuWaitContextAlloc(LwSciSyncModule module,
 * LwSciSyncCpuWaitContext* newContext)
 */

/**
 * LwSciSyncCpuWaitContext is deallocated using LwSciCommon functionality.
 *
 * Conlwrrency:
 * - Thread-safe: Yes
 * - Validation of @a module value requires thread synchronization; without
 *   synchronization, the function could cause a call to LwSciCommonPanic().
 *   No synchronization is done in the function. The user must ensure that @a
 *   module value is not changed during the call to the function.
 * - Access to and releasing of the @a context->waitContextBackEnd, access to
 *   and closing of the @a context->module and releasing of the @a context all
 *   require thread synchronization; without synchronization, the function
 *   could cause system error. No synchronization is done in the function. To
 *   ensure that no system error oclwrs, the user must ensure that the @a
 *   context->waitContextBackEnd and the @a context->module values are not
 *   modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844440}
 *
 * \fn void LwSciSyncCpuWaitContextFree(LwSciSyncCpuWaitContext context)
 */

/**
 * @}
 */

/**
 * \brief Validates LwSciSyncCpuWaitContext. It validates the LwSciSyncModule
 * associated with the LwSciSyncCpuWaitContext using
 * LwSciSyncCoreModuleValidate() and LwSciSyncCoreRmWaitContextBackEnd held by
 * the LwSciSyncCpuWaitContext using LwSciSyncCoreRmWaitCtxBackEndValidate().
 *
 * \param[in] context LwSciSyncCpuWaitContext to validate.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a context is NULL, or any of the data @a context
 *   is referring to is NULL.
 * - Panics if @a context is invalid
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
 * \implements{18844446}
 */
LwSciError LwSciSyncCoreCpuWaitContextValidate(
    LwSciSyncCpuWaitContext context);

/**
 * \brief Retrieves LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 *
 * \param[in] context The LwSciSyncCpuWaitContext to retrieve
 * LwSciSyncModule from.
 *
 * \return LwSciSyncModule
 * - module associated with the provided context
 * - Panics if @a context is NULL
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
 * \implements{18844449}
 */
LwSciSyncModule LwSciSyncCoreCpuWaitContextGetModule(
    LwSciSyncCpuWaitContext context);

/**
 * \brief Retrieves LwSciSyncCoreRmWaitContextBackEnd held by the
 * LwSciSyncCpuWaitContext
 *
 * \param[in] context LwSciSyncCpuWaitContext to retrieve the
 * LwSciSyncCoreRmWaitContextBackEnd from.
 *
 * \return LwSciSyncCoreRmWaitContextBackEnd
 * - handle to platform specific resources held by @a context.
 * - Panics if @a context is NULL
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
 * \implements{18844452}
 */
LwSciSyncCoreRmWaitContextBackEnd LwSciSyncCoreCpuWaitContextGetBackEnd(
    LwSciSyncCpuWaitContext context);

#endif
