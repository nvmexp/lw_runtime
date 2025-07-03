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
 * \brief <b>LwSciBuf Module Interface</b>
 *
 * @b Description: This file contains LwSciBuf Module core interfaces.
 */

#ifndef INCLUDED_LWSCIBUF_MODULE_H
#define INCLUDED_LWSCIBUF_MODULE_H

#include "lwscibuf.h"
#include "lwscibuf_dev.h"
#include "lwscicommon_objref.h"
#include "lwscibuf_alloc_interface.h"

/**
 * @defgroup lwscibuf_blanket_statements LwSciBuf blanket statements.
 * Generic statements applicable for LwSciBuf interfaces.
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements LwSciBuf blanket statements
 * \section lwscibuf_element_dependency Dependency on other elements
 * LwSciBuf calls below LwSciCommon interfaces:
 * - LwSciCommonAllocObjWithRef() to allocate reference structure along with
 * actual structure containing resource data that reference structure points to
 * - LwSciCommonFreeObjAndRef() to free the reference structure. The actual
 * structure containing resource data that reference structure points to is
 * freed when all the reference structures are freed.
 * - LwSciCommonDuplicateRef() to duplicate the reference.
 * - LwSciCommonGetObjFromRef() to retrieve resource structure from reference.
 * - LwSciCommonMemcpyS() to copy memory contents.
 * - LwSciCommonMemcmp() to compare the contents of the memory.
 * - LwSciCommonTransportPrepareBufferForTx() to colwert transport buffer
 * object to binary array.
 * - LwSciCommonTransportAllocTxBufferForKeys() to allocate transport buffer
 * object.
 * - LwSciCommonTransportGetNextKeyValuePair() to get key value pair from
 * transport buffer object.
 * - LwSciCommonTransportAppendKeyValuePair() to append key-value pair to
 * transport buffer.
 * - LwSciCommonCalloc() to allocating memory resource.
 * - LwSciCommonTransportGetRxBufferAndParams() to colwert binary array into
 * transport buffer object.
 * - LwSciCommonFree() to deallocating memory resource.
 * - LwSciCommonPanic() to abort the process exelwtion.
 * - LwSciCommonSort() to sort an array.
 * - LwSciCommonObjLock(), LwSciCommonObjUnlock() to lock, unlock
 * an object of LwSciCommon reference framework allocated with
 * LwSciCommonAllocObjWithRef.
 * - LwSciCommonRefLock(), LwSciCommonRefUnlock() to acquire, release thread
 * synchronization lock on input LwSciRef.
 * - LWSCI_ERR_STR(), LWSCI_ERR_HEXUINT(), LWSCI_ERR_UINT(), LWSCI_ERR_INT(),
 * LWSCI_ERR_SLONG(), LWSCI_ERR_ULONG() to record strings in a safety log.
 *
 * \section lwscibuf_in_params Input parameters
 * - LwSciBufModule passed as input parameter to an API is valid input
 * if it is returned from a successful call to LwSciBufModuleDupRef()
 * and has not yet been deallocated using LwSciBufModuleClose().
 *
 * \implements{18842583}
 */

/**
 * @}
 */

/**
 * LwSciBufModule is a reference to particular module resource.
 *
 * Module resource is a top-level container for the following set of resources:
 * LwSciBufAttrLists, buffers, and LwSciBufObjs. It can be referenced by one or
 * more LwSciBufModules.
 */

/**
 * @defgroup lwscibuf_init_api LwSciBuf Initialization APIs
 * List of APIs to initialize/de-initialize LwSciBuf module.
 * @{
 */

/**
 * Allocates and initializes module resource and LwSciBufModule using
 * LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent allocation of an LwSciBufDev is handled via
 *        LwSciBufDevOpen()
 *
 * \implements{18842808}
 *
 * \fn LwSciError LwSciBufModuleOpen(
 *      LwSciBufModule* newModule);
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the underlying module resource is handled via
 *        LwSciCommonFreeObjAndRef()
 *      - The user must ensure that the same LwSciBufModule is not used by
 *        multiple threads in other functions other than other ilwocations of
 *        this API at the same time when calling this API
 *
 * \implements{18842811}
 *
 * \fn void LwSciBufModuleClose(
 *      LwSciBufModule module);
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \implements{18842814}
 *
 * \fn LwSciError LwSciBufCheckVersionCompatibility(
 *      uint32_t majorVer,
 *      uint32_t minorVer,
 *      bool* isCompatible);
 */

/**
 * @}
 */

/**
 * \brief Retrieves LwSciBufDev from module resource referenced by
 * LwSciBufModule.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] module LwSciBufModule to retrieve LwSciBufDev from
 * \param[out] dev retrieved LwSciBufDev
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if:
 *       - @a module is NULL
 * - Panics if:
 *       - @a dev is NULL
 *       - @a module is invalid
 *
 * \implements{18842820}
 */
LwSciError LwSciBufModuleGetDevHandle(
    LwSciBufModule module,
    LwSciBufDev* dev);

/**
 * \brief Creates new LwSciBufModule holding the reference to the same module
 * resource to which input LwSciBufModule holds reference using
 * LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access of the underlying module resource is handled via
 *        LwSciCommonDuplicateRef()
 *
 * \param[in] oldModule LwSciBufModule from which new LwSciBufModule need to be
 *            created.
 * \param[out] newModule The new LwSciBufModule.
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if @a oldModule or @a newModule parameter is NULL.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_IlwalidState if the number of LwSciBufModules
 *   referencing module resource are INT32_MAX and this API is called to
 *   create one more LwSciBufModule reference.
 * - LwSciError_ResourceError if system lacks resource other than memory
 * - Panics if oldModule is invalid.
 *
 * \implements{18842823}
 */
LwSciError LwSciBufModuleDupRef(
    LwSciBufModule oldModule,
    LwSciBufModule* newModule);

/**
 * \brief Checks if both input LwSciBufModules reference the same module
 * resource.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufModule
 *        reference is never modified after creation (so there is no
 *        data-dependency)
 *
 * \param[in] firstModule LwSciBufModule
 * \param[in] secondModule LwSciBufModule
 * \param[out] isEqual boolean value specifying if LwSciBufModules are equal.
 *             True, if modules are equal, false otherwise.
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any parameter of following is NULL:
 *                    - @a firstModule
 *                    - @a secondModule
 *                    - @a isEqual
 * - Panics if firstModule or secondModule is invalid
 *
 * \implements{18842826}
 */
LwSciError LwSciBufModuleIsEqual(
    LwSciBufModule firstModule,
    LwSciBufModule secondModule,
    bool* isEqual);

/**
 * \brief Validates module resource referenced by LwSciBufModule by checking
 * if the Magic ID matches with predefined constant value.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the Magic ID is never
 *        modified after creation (so there is no data-dependency)
 *
 * \param[in] module LwSciBufModule to be validated
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if module is NULL
 * - Panics if module is invalid
 *
 * \implements{18842829}
 */
LwSciError LwSciBufModuleValidate(
    LwSciBufModule module);

/**
 * \brief Retrieves open context of the specified LwSciBufAllocIfaceType from
 *  module resource.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *      - Reads only occur from immutable data since the open context is never
 *        modified after creation (so there is no data-dependency)
 *
 * \param[in] module The LwSciBufModule.
 * \param[in] allocType LwSciBufAllocIfaceType. The valid value is
 *            LwSciBufAllocIfaceType_SysMem <= allocType <
              LwSciBufAllocIfaceType_Max.
 * \param[out] openContext returned open context corresponding to
 *                         provided LwSciBufAllocIfaceType.
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *     - @a module is NULL.
 *     - @a openContext is NULL.
 *     - @a allocType >= LwSciBufAllocIfaceType_Max.
 * - LwSciError_ResourceError if obtained openContext from module
 *                             resource is NULL.
 *
 * - Panics if @a module is invalid.
 *
 * \implements{18842832}
 */
LwSciError LwSciBufModuleGetAllocIfaceOpenContext(
    LwSciBufModule module,
    LwSciBufAllocIfaceType allocType,
    void** openContext);

/**
 * \brief Callback to free the data associated with the LwSciObj representing
 * the underlying LwSciBufModule using LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciObj prior to calling this function
 *
 * \param[in] obj Pointer to the LwSciObj associated with the LwSciBufModule to
 *                free
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - obj is NULL
 *      - obj is invalid
 *
 * \implements{22034419}
 */
void LwSciBufModuleCleanupObj(
    LwSciObj* obj);

#endif /* INCLUDED_LWSCIBUF_MODULE_H */
