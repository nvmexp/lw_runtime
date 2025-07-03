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
 * \brief <b>LwSciSync Module Management Interface</b>
 *
 * @b Description: This file contains LwSciSync module management core
 * structures and interfaces.
 */

#ifndef INCLUDED_LWSCISYNC_MODULE_H
#define INCLUDED_LWSCISYNC_MODULE_H

#include "lwscicommon_objref.h"
#include "lwscisync.h"
#include "lwscisync_backend.h"
#include "lwscisync_internal.h"

/**
 * @defgroup lwsci_sync Synchronization APIs
 * @{
 */

/**
 * \page lwscisync_page_unit_blanket_statements LwSciSync blanket statements
 * \section lwscisync_element_dependency Dependency on other elements
 * LwSciSync calls below LwSciIpc interfaces:
 * - LwSciIpcGetEndpointInfo() to validate LwSciIpcEndpoint passed to LwSciSync.
 * \section lwscisync_element_dependency Dependency on other elements
 * LwSciSync calls below LwSciCommon interfaces:
 * - LwSciCommonAllocObjWithRef() to allocate reference structure along with
 * actual structure containing resource data that reference structure points to
 * - LwSciCommonFreeObjAndRef() to free the reference structure. The actual
 * structure containing resource data that reference structure points to is
 * freed when all the reference structures are freed.
 * - LwSciCommonIncrAllRefCounts() to increment the reference counts of the
 *   LwSciRef and LwSciObj together
 * - LwSciCommonDuplicateRef() to duplicate the reference.
 * - LwSciCommonGetObjFromRef() to retrieve resource structure from reference.
 * - LwSciCommonMemcpyS() to copy memory contents.
 * - LwSciCommonTransportPrepareBufferForTx() to colwert transport buffer object
 * to binary array.
 * - LwSciCommonTransportAllocTxBufferForKeys() to allocate transport buffer
 * object.
 * - LwSciCommonTransportGetNextKeyValuePair() to get key value pair from
 * transport buffer object
 * - LwSciCommonTransportAppendKeyValuePair() to append key-value pair to
 * transport buffer.
 * - LwSciCommonCalloc() to allocate and zero memory.
 * - LwSciCommonTransportGetRxBufferAndParams() to colwert binary array into
 * transport buffer object.
 * - LwSciCommonFree() to deallocate memory previously allocated with
 * LwSciCommonCalloc().
 * - LwSciCommonPanic() to "panic" and abort the process exelwtion.
 * - LwSciCommonSort() to sort an array.
 * - LwSciCommonObjLock(), LwSciCommonObjUnlock() to lock, unlock an object of
 * LwSciCommon reference framework allocated with LwSciCommonAllocObjWithRef()
 * - u32Add(), u64Add(), sizeAdd(), u64Sub(), u64Mul(), sizeMul() are used for
 * arithmetic operations.
 * - LWSCI_ERR_STR(), LWSCI_ERR_HEXUINT(), LWSCI_ERR_UINT(), LWSCI_ERR_INT(),
 * LWSCI_ERR_SLONG(), LWSCI_ERR_ULONG() to record strings in a safety log.
 * \section lwscisync_in_out_params Input/Output parameters.
 * - LwSciIpcEndpoint passed as input parameter to an API is valid if it is
 *   obtained from successful call to LwSciIpcOpenEndpoint() and has not yet
 *   been freed using LwSciIpcCloseEndpoint().
 * - LwSciSyncModule passed as input parameter to an API is valid input if it is
 *   returned from a successful call to LwSciSyncCoreModuleDup() and has not yet
 *   been deallocated using LwSciSyncModuleClose().
 * - LwSciSyncModule passed as input parameter to an API is validated by calling
 *   LwSciSyncCoreModuleValidate().
 *
 * \implements{18844104}
 */

/**
 * LwSciSyncModule is a reference to a particular module resource.
 * Module resource is a top-level container for the following set of resources:
 * LwSciSyncAttrLists, LwSciSyncObjs, LwSciSyncCpuWaitContexts, LwSciSyncFence.
 * It can be referenced by one or more LwSciSyncModules.
 */

/**
 * Allocates and initializes module resource and LwSciSyncModule using
 * LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwRm backend is handled via
 *        LwSciSyncCoreRmAlloc()
 *
 * \implements{18844596}
 *
 * \fn LwSciError LwSciSyncModuleOpen(
 *    LwSciSyncModule* newModule);
 */

/**
 * Removes reference to the module resource by destroying the LwSciSyncModule.
 * LwSciSyncModule reference is freed using LwSciCommon functionality.
 * Module resource will be freed only after all the references to it are
 * released.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the underlying module resource is handled via
 *        LwSciCommonFreeObjAndRef()
 *      - The user must ensure that the same LwSciSyncModule is not used by
 *        multiple threads in other functions other than other ilwocations of
 *        this API at the same time when calling this API
 *
 * \implements{18844599}
 *
 * \fn void LwSciSyncModuleClose(
 *   LwSciSyncModule module);
 */

/**
 * \brief Returns the current value of module counter from module resource
 * associated with input LwSciSyncModule, and then increments it.
 * The increment is done atomically by locking the reference to module resource
 * by using LwSciCommon functionality.
 *
 * For a given LwSciSyncModule, each call provides a different module
 * counter value
 * that can be used as a unique id for LwSciSyncObjs.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on the underlying module resource to serialize writes
 *      - Locks are held for the duration of all operations on the input
 *        LwSciSyncModule
 *      - Locks are released when all operations on the input LwSciSyncModule
 *        are complete
 *
 * \param[in] module LwSciSyncModule to retrieve the module counter from
 * \param[out] cntrValue module counter
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_Overflow if module count equals UINT64_MAX
 * - Panics if @a module is NULL/invalid or if @a cntrValue is NULL
 *
 * \implements{18844605}
 */
LwSciError LwSciSyncCoreModuleCntrGetNextValue(
    LwSciSyncModule module,
    uint64_t* cntrValue);

/**
 * \brief Validates the LwSciSyncModule.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the Magic ID is never
 *        modified after creation (so there is no data-dependency)
 *
 * \param[in] module LwSciSyncModule to validate
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a module is NULL
 * - Panics if @a module is invalid
 *
 * \implements{18844608}
 */
LwSciError LwSciSyncCoreModuleValidate(
    LwSciSyncModule module);

/**
 * \brief Creates a new LwSciSyncModule referencing the same module resource as
 * the input LwSciSyncModule.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the input module resource is provided via
 *        LwSciCommonDuplicateRef()
 *
 * \param[in] module LwSciSyncModule to duplicate
 * \param[out] dupModule new LwSciSyncModule
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is not enough memory to create
 *   a duplicate.
 * - LwSciError_ResourceError if system lacks resources other than memory.
 * - LwSciError_IlwalidState if the number of LwSciSyncModule referencing module
 *   resource are INT32_MAX and this API is called to create one more
 *   LwSciSyncModule reference.
 * - Panics if @a module is invalid, @a dupModule is NULL.
 *
 * \implements{18844611}
 */
LwSciError LwSciSyncCoreModuleDup(
    LwSciSyncModule module,
    LwSciSyncModule* dupModule);

/**
 * \brief Checks if the given LwSciSyncModules are referring to the same module
 * resource.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since a LwSciObj identifying
 *        a module resource on an LwSciSyncModule is never changed after
 *        creation (so there is no data-dependency)
 *
 * \param[in] module first LwSciSyncModule to compare
 * \param[in] otherModule another LwSciSyncModule to compare
 * \param[out] isDup boolean output, true if the given LwSciSyncModules are
 * referring to the same module resource else false
 *
 * \return void
 * - Panics if module, otherModule or isDup is NULL or either module or
 *   otherModule are not valid modules
 *
 * \implements{18844614}
 */
void LwSciSyncCoreModuleIsDup(
    LwSciSyncModule module,
    LwSciSyncModule otherModule,
    bool* isDup);

/**
 * \brief Callback to free the data associated with the LwSciObj representing
 * the underlying LwSciSyncCoreModule using LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciObj prior to calling this function
 *      - Conlwrrent access to the LwRm backend is handled via
 *        LwSciSyncCoreRmFree()
 *
 * \param[in] objPtr Pointer to the LwSciObj associated with the
 * LwSciSyncModule to free
 *
 * \return void
 * - Panics if objPtr is NULL or invalid
 *
 * \implements{22034751}
 */
void LwSciSyncCoreModuleFree(
    LwSciObj* objPtr);

/**
 * \brief Retrieves LwSciBufModule from LwSciSyncModule
 *
 * \param[in] module LwSciSyncModule
 * \param[out] bufModule LwSciBufModule
 *
 * \return void
 * - Panics if @a bufModule is NULL
 */
void LwSciSyncCoreModuleGetBufModule(
    LwSciSyncModule module,
    LwSciBufModule* bufModule);

/**
 * \brief Retrieves LwSciSyncCoreRmBackEnd from module resource referenced
 * by the input LwSciSyncModule.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciSyncCoreRmBackEnd
 *        set on the module resource referenced by the LwSciSyncModule is never
 *        modified after creation (so there is no data-dependency)
 *
 * \param[in] module LwSciSyncModule to retrieve the LwSciSyncCoreRmBackEnd from
 * \param[out] backEnd LwSciSyncCoreRmBackEnd
 *
 * \return void
 * - Panics if @a module is invalid or @a backEnd is NULL
 *
 * \implements{18844617}
 */
void LwSciSyncCoreModuleGetRmBackEnd(
    LwSciSyncModule module,
    LwSciSyncCoreRmBackEnd* backEnd);
 /** @} */
#endif
