/*
 * Copyright (c) 2019-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_LWSCI_OBJECT_REFERENCE_H
#define INCLUDED_LWSCI_OBJECT_REFERENCE_H

#include <stdint.h>
#include <stdbool.h>

#include "lwscicommon_os.h"
#include "lwscierror.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup lwscicommon_blanket_statements LwSciCommon blanket statements.
 * Generic statements applicable for LwSciCommon interfaces.
 * @{
 */

/**
 * \page page_blanket_statements LwSciCommon blanket statements
 * \section in_out_params Input parameters
 * - LwSciRef passed as an input parameter to an API is valid input if it is
 * returned from successful call to LwSciCommonAllocObjWithRef or
 * LwSciCommonDuplicateRef and not yet been deallocated using
 * LwSciCommonFreeObjAndRef.
 *
 */

/**
 * @}
 */

/**
 * @defgroup lwscicommon_objref_api LwSciCommon APIs for referencing framework.
 * List of APIs exposed at Inter-Element level.
 * @{
 */

/**
 * \brief Represents LwSciRef structure.
 *  This structure is allocated, initialized by LwSciCommonAllocObjWithRef,
 *  LwSciCommonDuplicateRef and de-initialized, freed by
 *  LwSciCommonFreeObjAndRef when active number of consumers using this LwSciRef
 *  falls to zero.
 *  This structure should be first member in user specific reference container.
 *
 * \implements{21755928}
 * \implements{21751549}
 * \implements{18850770}
 */
typedef struct {
    /** Magic ID to detect if this LwSciRef is valid.
     * This member must be initialized to a particular non-zero constant.
     * It must be changed to a different value when this LwSciRef
     *  is freed.
     * This member must NOT be modified in between allocation and
     *  deallocation of the LwSciRef.
     * Whenever an object reference unit API is called with
     *  LwSciRef as input from outside the unit, the object reference unit must
     *  validate the magic ID.
     */
    uint32_t magicNumber;
    /** Size of user specific reference container.
     *  This member is initialized to the size of user specific reference structure.
     *  Valid value: [sizeof(LwSciRef), SIZE_MAX]. */
    size_t size;
    /** Number of consumers using this LwSciRef.
     *  This member is initialized to 1.
     *  Valid value: [0, INT32_MAX]. */
    int32_t refCount;
    /** Lock for atomic operation on this LwSciRef.
     *  This member is initialized by calling LwSciCommonMutexCreate and
     *  destroyed by calling LwSciCommonMutexDestroy. User needs to acquire lock
     *  using LwSciCommonRefLock before accessing user specific reference
     *  container data and release lock using LwSciCommonRefUnlock after access
     *  is complete. */
    LwSciCommonMutex refLock;
    /** Pointer to underlying LwSciObj.
     *  This member is initialized to address of newly created LwSciObj.
     *  This member is invalid if the value is NULL. */
    void* objPtr;
} LwSciRef;

/**
 * \brief Represents LwSciObj structure. This structure is allocated,
 *  initialized by LwSciCommonAllocObjWithRef and de-initialized, freed by
 *  LwSciCommonFreeObjAndRef when the number of LwSciRef associated with this
 *  LwSciObj falls to zero.
 *  This structure should be the first member in user specific container.
 *
 * \implements{21751547}
 * \implements{21755927}
 * \implements{18850767}
 */
typedef struct {
    /** Reference count of LwSciObj which indicates number of LwSciRef
     *  associated with this LwSciObj. This member is initialized to value 1.
     *  Valid value: [0, INT32_MAX]. */
    int32_t refCount;
    /** Lock for atomic operation on LwSciObj.
     *  This member is initialized by calling LwSciCommonMutexCreate and
     *  destroyed by calling LwSciCommonMutexDestroy. User needs to acquire lock
     *  using LwSciCommonObjLock before accessing user specific container data
     *  and release the lock using LwSciCommonObjUnlock when access is
     *  complete.*/
    LwSciCommonMutex objLock;
} LwSciObj;

/**
 * Macro for determining the offset of "member" in "type".
 *
 */
#if !defined(LW_OFFSETOF)
    #if defined(__GNUC__)
        #define LW_OFFSETOF(type, member)   __builtin_offsetof(type, member)
    #else
        #define LW_OFFSETOF(type, member)   ((size_t)(&(((type *)0)->member)))
    #endif
#endif

/**
 * \brief Allocates memory for user specific reference container and user
 *  specific container of requested size.
 *  It initializes LwSciRef and LwSciObj structure inside user specific
 *  reference container and user specific container.
 *
 * \param[in] objSize Size of user specific container that needs to be allocated.
 *  Valid value: [sizeof(LwSciObj), SIZE_MAX].
 * \param[in] refSize Size of user specific reference container that needs
 *  to be allocated.
 *  Valid value: [sizeof(LwSciRef), SIZE_MAX].
 * \param[out] objPtr Pointer to newly allocated LwSciObj.
 * \param[out] refPtr Pointer to newly allocated LwSciRef.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_ResourceError if system lacks resource other than memory to
 *  initialize the lock.
 * - Panics if any of the following oclwrs:
 *      - if @a objSize is less than size of LwSciObj structure
 *      - if @a refSize is less than size of LwSciRef structure
 *      - if @a objPtr is NULL
 *      - if @a refPtr is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same output @a objPtr and @a refPtr
 *        parameters are not used by multiple threads at the same time.
 *
 * \implements{18851091}
 */
LwSciError LwSciCommonAllocObjWithRef(
    size_t objSize,
    size_t refSize,
    LwSciObj** objPtr,
    LwSciRef** refPtr);

/**
 * \brief Atomically decrements the consumer count of the input
 *  LwSciRef and reference count of the LwSciObj associated with the
 *  input LwSciRef. If the consumer count of the input LwSciRef
 *  falls to zero, user specific reference container is cleaned up by calling
 *  the input refCleanupCallback function, deinitialized and freed.
 *  If the reference count of the LwSciObj associated with the input
 *  LwSciRef falls to zero, user specific container is cleaned up by calling
 *  the input objCleanupCallback, deinitialized and freed.
 *
 * \param[in] ref LwSciRef whose consumer count needs to be decremented.
 * \param[in] objCleanupCallback user clean up function which will be called if
 *  the reference count of LwSciObj associated with the input LwSciRef falls to zero.
 *  The clean up function will be called by passing the LwSciObj associated with the
 *  input LwSciRef as an argument to the function. If clean up function is NULL,
 *   clean up routine will be skipped.
 * \param[in] refCleanupCallback user clean up function which will be called if
 *  the reference count of the input LwSciRef falls to zero. The clean up function will be
 *  called by passing the input LwSciRef as an argument to the function. If the clean up
 *  function is NULL, clean up routine will be skipped.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - if @a ref is NULL
 *      - if @a ref is invalid
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - RefCount is used to guarantee that only the last user would
 *        free the resource.
 *      - Conlwrrent access to the refCount of @a ref is protected by
 *        LwSciCommonMutexLock().
 *
 * \implements{18851094}
 */
void LwSciCommonFreeObjAndRef(
    LwSciRef* ref,
    void (*objCleanupCallback)(LwSciObj* obj),
    void (*refCleanupCallback)(LwSciRef* ref));

/**
 * \brief Atomically increments consumer count of the input
 *  LwSciRef and reference count of the LwSciObj associated with
 *  the input LwSciRef by one.
 *
 * \param[in] ref LwSciRef whose reference count needs to be incremented.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_IlwalidState if any of the following oclwrs:
 *      - LwSciRef reference count is at least INT32_MAX
 *      - LwSciObj reference count is at least INT32_MAX
 * - Panics if any of the following oclwrs:
 *      - if @a ref is NULL
 *      - if @a ref is invalid
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the refCount of @a ref is protected by
 *        LwSciCommonMutexLock().
 *
 * \implements{18851097}
 */
LwSciError LwSciCommonIncrAllRefCounts(
    LwSciRef* ref);

/**
 * \brief Retrieves LwSciObj associated with the input LwSciRef.
 *
 * \param[in] ref LwSciRef from which LwSciObj needs to be retrieved.
 * \param[out] objPtr The LwSciObj associated with the input LwSciRef.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a ref is NULL
 *      - @a ref is invalid
 *      - @a objPtr is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the output @a objPtr is not modified by
 *        multiple threads at the same time.
 *
 * \implements{18851100}
 */
void LwSciCommonGetObjFromRef(
    const LwSciRef* ref,
    LwSciObj** objPtr);

/**
 * \brief Acquires thread synchronization lock on the input LwSciRef.
 *  The acquired lock can be released using LwSciCommonRefUnlock API.
 *
 * \param[in] ref LwSciRef which needs to be locked.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - if @a ref is NULL
 *      - if @a ref is invalid
 *      - failed to acquire lock on @a ref
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciCommonMutex of @a ref is handled via
 *        LwSciCommonMutexLock()
 *
 * \implements{18851103}
 */
void LwSciCommonRefLock(
    LwSciRef* ref);

/**
 * \brief Releases thread synchronization lock on the input LwSciRef that was
 *  previously held by same exelwtion thread using LwSciCommonRefLock API.
 *
 * \param[in] ref LwSciRef which needs to be unlocked.
 *  Valid value: If LwSciRef is previously locked by using
 *      LwSciCommonRefLock by the same exelwtion thread.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - if @a ref is NULL
 *      - if @a ref is invalid
 *      - failed to release lock on @a ref
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciCommonMutex of @a ref is handled via
 *        LwSciCommonMutexUnlock()
 *
 * \implements{18851106}
 */
void LwSciCommonRefUnlock(
    LwSciRef* ref);

/**
 * \brief Creates new LwSciRef structure pointed by newRef associated
 *  to the same LwSciObj as the input LwSciRef.
 *
 * \param[in] oldRef LwSciRef from which new LwSciRef instance needs to
 *  be created.
 * \param[out] newRef pointer to new LwSciRef.
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_ResourceError if system lacks resource other than memory to
 *  initialize the lock.
 * - LwSciError_IlwalidState if refcount of LwSciObj associated with the input
 *      @a oldRef is at least INT32_MAX
 * - Panics if any of the following oclwrs:
 *      - if @a oldRef is NULL
 *      - if @a oldRef is invalid
 *      - if @a newRef is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the @a oldRef is protected by
 *        LwSciCommonMutexLock()
 *
 * \implements{18851109}
 */
LwSciError LwSciCommonDuplicateRef(
    const LwSciRef* oldRef,
    LwSciRef** newRef);

/**
 * \brief Acquires thread synchronization lock on LwSciObj associated with the
 *  input LwSciRef. The acquired lock can be released using LwSciCommonObjUnlock API.
 *
 * \param[in] ref Acquires data access lock on LwSciObj associated
 *  with the input ref.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - if @a ref is NULL
 *      - if @a ref is invalid
 *      - failed to acquire lock on LwSciObj associated with input @a ref
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciCommonMutex of @a ref is handled via
 *        LwSciCommonMutexLock()
 *
 * \implements{18851112}
 */
void LwSciCommonObjLock(
    const LwSciRef* ref);

/**
 * \brief Releases thread synchronization lock on LwSciObj associated with the
 *  input LwSciRef which was previously locked by same exelwtion thread using
 *  LwSciCommonObjLock API.
 *
 * \param[in] ref LwSciRef pointer to reference structure.
 *  Valid value: if LwSciObj associated with the input ref
 *       is previously locked by using LwSciCommonObjLock.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - if @a ref is NULL
 *      - if @a ref is invalid
 *      - failed to release lock on LwSciObj associated with input @a ref
 *
 * \conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciCommonMutex of @a ref is handled via
 *        LwSciCommonMutexUnlock().
 *
 * \implements{18851115}
 */
void LwSciCommonObjUnlock(
    const LwSciRef* ref);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCI_OBJECT_REFERENCE_H */
