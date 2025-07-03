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
 * \brief <b>LwSciCommon os Interface</b>
 *
 * @b Description: This file contains LwSciCommon os APIs
 */

#ifndef INCLUDED_LWSCICOMMON_OS_H
#define INCLUDED_LWSCICOMMON_OS_H

#include <stdint.h>
#include "lwscierror.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(LW_QNX) || defined(LW_LINUX)
#include "lwscicommon_posix.h"
#else
#error "non-posix OS abstraction not supported"
#endif

/**
 * @defgroup lwscicommon_blanket_statements LwSciCommon blanket statements.
 * Generic statements applicable for LwSciCommon interfaces.
 * @{
 */

/**
 * \page page_blanket_statements LwSciCommon blanket statements
 * \section in_out_params Input parameters
 * - LwSciCommonMutex passed as an input parameter to an API is valid input if it is
 * initialized from successful call to LwSciCommonMutexCreate and not yet been
 * destroyed using LwSciCommonMutexDestroy.
 *
 * \section element_dependency Dependency on other elements
 * - pthread_mutex_init() to initialize thread synchronization object.
 * - pthread_mutex_lock() to lock thread synchronization object.
 * - pthread_mutex_unlock() to unlock thread synchronization object.
 * - pthread_mutex_destroy() to destroy thread synchronization object.
 * - abort() to terminate current program
 * - nanosleep() to suspend exelwtion of current thread.
 *
 */

/**
 * @}
 */


/**
 * @defgroup lwscicommon_platformutils_api LwSciCommon APIs for platform utils.
 * List of APIs exposed at Inter-Element level.
 * @{
 */

/**
 * \brief Initializes the input LwSciCommonMutex used for thread synchronization.
 *
 * \param[in,out] mutex a pointer to the LwSciCommonMutex to be initialized.
 *  Valid value: @a mutex is not NULL
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_ResourceError if system lacks resource other than memory to
 *   initialize the lock.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - Panics if:
 *      - @a mutex is NULL
 *      - failed to initialize @a mutex.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Provided there is no active operation on the @a mutex before
 *        this function call.
 *
 * \implements{18851235}
 */
LwSciError LwSciCommonMutexCreate(LwSciCommonMutex* mutex);

/**
 * \brief Acquires lock on the input LwSciCommonMutex such that the subsequent
 *  calls to LwSciCommonMutexLock will wait until the lock is released by the
 *  same exelwtion thread which acquired lock.
 *
 * \param[in,out] mutex pointer to the LwSciCommonMutex to lock.
 *  Valid value: @a mutex is not NULL and has been previously initialized using
 *  LwSciCommonMutexCreate and not destroyed by LwSciCommonMutexDestroy.
 *
 * \return void
 * - Panics if:
 *      - @a mutex is NULL
 *      - failed to acquire lock on @a mutex.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Provided that the @a mutex not destroyed by other threads at the same
 *        time. The refCount can help protect the mutex from destruction when
 *        this function is called.
 *      - Conlwrrent access to the @a mutex parameter is handled by
 *        pthread_mutex_lock(), which is thread-safe.
 *
 * \implements{18851238}
 */
void LwSciCommonMutexLock(LwSciCommonMutex* mutex);

/**
 * \brief Releases lock on the input LwSciCommonMutex which was previously
 *  locked by the same exelwtion thread.
 *
 * \param[in,out] mutex pointer to the LwSciCommonMutex to unlock.
 *  Valid value: @a mutex is not NULL and has been previously initialized using
 *  LwSciCommonMutexCreate and not destroyed by LwSciCommonMutexDestroy and
 *  locked using LwSciCommonMutexLock by the same exelwtion thread.
 *
 * \return void
 * - Panics if:
 *      - @a mutex is NULL
 *      - failed to release lock on @a mutex.
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Provided that the @a mutex not destroyed by other threads at the same
 *        time. The refCount can help protect the mutex from destruction when
 *        this function is called.
 *      - Conlwrrent access to the @a mutex parameter is handled by
 *        pthread_mutex_unlock(), which is thread-safe.
 *
 * \implements{18851241}
 */
void LwSciCommonMutexUnlock(LwSciCommonMutex* mutex);

/**
 * \brief Destroys the input LwSciCommonMutex.
 *
 * \param[in] mutex a pointer to the LwSciCommonMutex to be destroyed.
 *  Valid value: @a mutex is not NULL and has been previously initialized using
 *  LwSciCommonMutexCreate and not destroyed by LwSciCommonMutexDestroy and not
 *  lwrrently locked by LwSciCommonMutexLock.
 *
 * \return void
 * - Panics if:
 *      - @a mutex is NULL
 *      - failed to destroy @a mutex.
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the @a mutex parameter not destroyed
 *        by other threads at the same time.
 *
 * \implements{18851244}
 */
void LwSciCommonMutexDestroy(LwSciCommonMutex* mutex);

/**
 * \brief Terminates exelwtion of the program. LwSciCommonPanic is intended for
 *  unexpected error that should never occur, but could theoretically fail.
 *
 * \return void
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851247}
 */
void LwSciCommonPanic(void) __attribute__ ((noreturn));

/**
 * \brief Suspends exelwtion of a thread for at least the input time interval.
 *
 * \param[in] timeNs time interval in nanoseconds for which thread should
 * be suspended.
 *  Valid value: [0, LONG_MAX].
 *
 * \return void
 * - Panics if:
 *      - @a timeNs is not within valid range
 *      - the calling exelwtion context could not be suspended for at
 *        least the provided duration
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - No access to any shared data.
 *
 * \implements{18851250}
 */
void LwSciCommonSleepNs(uint64_t timeNs);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
