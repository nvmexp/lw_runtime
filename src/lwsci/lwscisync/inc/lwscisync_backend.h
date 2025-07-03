/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCISYNC_BACKEND_H
#define INCLUDED_LWSCISYNC_BACKEND_H

#include "lwscisync.h"

/**
 * \page lwscisync_page_unit_blanket_statements LwSciSync blanket statements
 * \section lwscisync_in_out_params Input/Output parameters
 * - LwSciSyncCoreRmBackEnd passed as input parameter to an API is valid input
 *   if it is returned from a successful call to LwSciSyncCoreRmAlloc() and has
 *   not yet been deallocated using LwSciSyncCoreRmFree().
 * - LwSciSyncCoreRmWaitContextBackEnd passed as input parameter to an API is
 *   valid input if it is returned from a successful call to
 *   LwSciSyncCoreRmWaitCtxBackEndAlloc() and has not yet been deallocated using
 *   LwSciSyncCoreRmWaitCtxBackEndFree().
 * - LwSciSyncCoreRmWaitContextBackEnd passed as input parameter to an API is
 *   validated by calling LwSciSyncCoreRmWaitCtxBackEndValidate().
 */

/**
 * \brief Pointer to LwSciSyncCoreRmBackEndRec reference structure.
 */
typedef struct LwSciSyncCoreRmBackEndRec* LwSciSyncCoreRmBackEnd;

/**
 * \brief Pointer to core RM Wait context backend structure
 */
typedef struct LwSciSyncCoreRmWaitContextBackEndRec*
        LwSciSyncCoreRmWaitContextBackEnd;

/**
 * \brief Allocates an LwSciSyncCoreRmBackEnd using LwSciCommon functionality.
 *
 * \param[out] backEnd allocated LwSciSyncCoreRmBackEnd
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if there is not enough system memory to
 *   allocate the LwSciSyncCoreRmBackEnd
 * - LwSciError_ResourceError if unable to obtain an underlying LwRmHost1xHandle
 *
 * - Panics if:
 *      - @a backEnd is NULL
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to dereferenced @a backEnd value requires thread synchronization;
 *   without synchronization, the function could cause system error. No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, the user must ensure that the dereferenced @a backEnd value is
 *   not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844830}
 */
LwSciError LwSciSyncCoreRmAlloc(
    LwSciSyncCoreRmBackEnd* backEnd);

/**
 * \brief Frees the resources allocated for LwSciSyncCoreRmBackEnd.
 *
 * \param[in] backEnd LwSciSyncCoreRmBackEnd containing the LwRmHost1xHandle
 * to be closed
 *
 * \return void
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Releasing @a backEnd value requires thread synchronization; without
 *   synchronization, the function could cause system error. No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, the user must ensure that the dereferenced @a backEnd value is
 *   not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844833}
 */
void LwSciSyncCoreRmFree(
    LwSciSyncCoreRmBackEnd backEnd);

/**
 * \brief Allocates an LwSciSyncCoreRmWaitContextBackEnd using LwSciCommon
 * functionality.
 *
 * \param[in] rmBackEnd The LwSciSyncCoreRmBackEnd
 * \param[out] waitContextBackEnd allocated LwSciSyncCoreRmWaitContextBackEnd
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if not enough memory to allocate a new
 *   LwSciSyncCoreRmWaitContextBackEnd
 * - LwSciError_ResourceError if unable to obtain an underlying
 *   LwRmHost1xWaiterHandle
 *
 * - Panics if:
 *      - @a rmBackEnd is NULL
 *      - @a waitContextBackEnd is NULL
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to dereferenced @a rmBackEnd value requires thread
 *   synchronization; without synchronization, the function could cause system
 *   error. No synchronization is done in the function. To ensure that no
 *   system error oclwrs, the user must ensure that the dereferenced @a
 *   rmBackEnd value is not modified during the call to the function.
 * - Access to dereferenced @a waitContextBackEnd value requires thread
 *   synchronization; without synchronization, the function could cause system
 *   error. No synchronization is done in the function. To ensure that no
 *   system error oclwrs, the user must ensure that the dereferenced @a
 *   waitContextBackEnd value is not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844836}
 */
LwSciError LwSciSyncCoreRmWaitCtxBackEndAlloc(
    LwSciSyncCoreRmBackEnd rmBackEnd,
    LwSciSyncCoreRmWaitContextBackEnd* waitContextBackEnd);

/**
 * \brief Frees the resources allocated for LwSciSyncCoreRmWaitContextBackEnd.
 *
 * \param[in] waitContextBackEnd LwSciSyncCoreRmWaitContextBackEnd to be freed
 *
 * \return void
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Releasing @a waitContextBackEnd value requires thread synchronization;
 *   without synchronization, the function could cause system error. No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, the user must ensure that the dereferenced @a waitContextBackEnd
 *   value is not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844839}
 */
void LwSciSyncCoreRmWaitCtxBackEndFree(
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd);

/**
 * \brief Validates the input LwSciSyncCoreRmWaitContextBackEnd by checking
 * if LwRmHost1xWaiterHandle held by LwSciSyncCoreRmWaitContextBackEnd is not
 * NULL.
 *
 * \param[in] waitContextBackEnd LwSciSyncCoreRmWaitContextBackEnd to be
 *            validated
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if the @a waiterHandle of the given
 *   @a waitContextBackEnd is NULL
 *
 * - Panics if:
 *      - @a waitContextBackEnd is NULL
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
 * \implements{18844842}
 */
LwSciError LwSciSyncCoreRmWaitCtxBackEndValidate(
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd);

#endif
