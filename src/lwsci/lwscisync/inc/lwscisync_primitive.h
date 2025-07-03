/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync Primitve Management Interface</b>
 *
 * @b Description: This file contains LwSciSync primitve management core
 * structures and interfaces.
 */

#ifndef INCLUDED_LWSCISYNC_PRIMITIVE_H
#define INCLUDED_LWSCISYNC_PRIMITIVE_H

#include "lwscicommon_libc.h"
#include "lwscierror.h"
#include "lwscisync.h"
#include "lwscisync_internal.h"
#include "lwscisync_primitive_type.h"

#include "lwscisync_c2c_priv.h"

/**
 * \brief Size of semaphore primitive type
 *
 *  \implements{TODO}
 */
#define LWSCISYNC_CORE_PRIMITIVE_SEMAPHORE_SIZE (16U)

/**
 * \page lwscisync_page_unit_blanket_statements LwSciSync blanket statements
 * \section lwscisync_element_dependency Dependency on other elements
 * LwSciSync calls below LwHost interfaces:
 * - LwRmHost1xGetDefaultSyncpointAllocateAttrs() to get default attributes
 * for LwRmHost1xSyncpointAllocate().
 * - LwRmHost1xSyncpointAllocate() to allocate an unused syncpoint.
 * - LwRmHost1xSyncpointGetId() to retrieve the ID of the syncpoint.
 * - LwRmHost1xSyncpointRead() to read the value of a syncpoint.
 * - LwRmHost1xSyncpointFree() to free an allocated syncpoint.
 * - LwRmHost1xSyncpointIncrement() to increment the value of the syncpoint.
 * - LwRmHost1xSyncpointWait() to wait until the value of a syncpoint reaches a
 * threshold.
 * \section lwscisync_in_out_params Input/Output parameters
 * - LwSciSyncCorePrimitive passed as input parameter to an API is valid input
 *   if it is returned from a successful call to LwSciSyncCoreInitPrimitive()
 *   or LwSciSyncCorePrimitiveImport() and has not yet been deallocated using
 *   LwSciSyncCoreDeinitPrimitive().
 *
 * \implements{18844104}
 */

/**
 * \brief Allocates a new LwSciSyncCorePrimitive and initializes its members
 * with passed inputs.
 *
 * \param[in] primitiveType reconciled LwSciSyncInternalAttrValPrimitiveType
 * Valid value: LwSciSyncInternalAttrValPrimitiveType_Syncpoint for CheetAh usecases
 * \param[in] reconciledList reconciled LwSciSyncAttrList
 * \param[out] primitive LwSciSyncCorePrimitive to be initialized
 * \param[in] needsAllocation tells if needs to reserve backend primitive.
 * Valid value: true or false.
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if no memory to allocate LwSciSyncCorePrimitive
 * - LwSciError_ResourceError if failed to reserve backend primitive
 * - Panics if any of the following oclwrs:
 *      - @a primitiveType is invalid
 *      - @a reconciledList is invalid
 *      - @a primitive is NULL
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Allocation of an LwSciSyncCorePrimitiveRec object and allocation of an
 *   LwSciSyncCoreSyncpointInfo object require thread synchronization; without
 *   synchronization, the function could cause a memory leak when the function
 *   is called from more than one thread in parallel with the same value for
 *   @a primitive. No synchronization is done in the function. To ensure that
 *   there is no memory leak, the user must ensure that the function is not
 *   called from other threads in parallel with the same value for @a
 *   primitive.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 *  \implements{18844764}
 */
LwSciError LwSciSyncCoreInitPrimitive(
    LwSciSyncInternalAttrValPrimitiveType primitiveType,
    LwSciSyncAttrList reconciledList,
    LwSciSyncCorePrimitive* primitive,
    bool needsAllocation);

/**
 * \brief Frees the resources allocated for LwSciSyncCorePrimitive.
 */
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * If syncpoint's shim memory was mapped during import, this function umaps it with
 * c2cCopyFuncs.syncDeleteCpuMapping(). If the syncHandle is associated with
 * this primitive it is freed with c2cCopyFuncs.syncFreeHandle().
 */
#endif
/**
 *
 * \param[in] primitive LwSciSyncCorePrimitive to deinit
 * \return void
 * - Panics if @a primitive is invalid
 */
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 *   or there was an unexpected freeing error from C2C
 */
#endif
/**
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Releasing of @a primitive requires thread synchronization; without
 *   synchronization, the function could cause a call to LwSciCommonPanic().
 *   No synchronization is done in the function. To ensure that
 *   LwSciCommonPanic() is not called, the user must ensure that the object
 *   value is not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 *  \implements{18844767}
 *
 */
void LwSciSyncCoreDeinitPrimitive(
    LwSciSyncCorePrimitive primitive);

/**
 * \brief Export the members of the LwSciSyncCorePrimitive to export descriptor.
 *
 * Uses LwSciSyncCorePermLEq() for permissions comparisons. Uses
 * LwSciSyncCoreGetSyncTopoId() for retrieving topoId information about
 * @a ipcEndpoint and checks whether it is a C2C endpoint with
 * LwSciSyncCoreIsTopoIdC2c().
 */
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * Gets C2C function pointer table with LwSciIpcGetC2cCopyFuncSet(). Calls
 * c2cCopyFuncs.syncMapRemoteMemHandle() and
 * c2cCopyFuncs.syncGetAuthTokenFromHandle() to generate an LwSciC2cPcieSyncHandle
 * and an exportable token. Serializes ids and the token using
 * LwSciCommonTransportAllocTxBufferForKeys(), LwSciCommonTransportAppendKeyValuePair()
 * and LwSciCommonTransportPrepareBufferForTx(). Frees the temporary LwSciCommon
 * struct with LwSciCommonTransportBufferFree().
 */
#endif
/**
 *
 * \param[in] primitive LwSciSyncCorePrimitive to export
 * \param[in] permissions access permissions to be exported
 * \param[in] ipcEndpoint LwSciIpcEndpoint through which export happens
 * \param[out] data blob to write LwSciSyncCorePrimitive information
 * \param[out] length size of data blob
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a permissions includes signaling
 *   but primitive is not syncpoint or ipcEndpoint is not C2C or this primitive
 *   does not own the syncpoint
 * - LwSciError_InsufficientMemory if not enough memory to create export
 *   descriptor
 * - LwSciError_ResourceError if something went wrong with LwSciIpc
 *   or ipcEndpoint is not valid
 */
#if (LW_L4T == 1)
/**
 * - LwSciError_NotSupported if trying to export syncpoint signaling
 *   over a C2C Ipc channel.
 */
#endif
/**
 * - Panics if any of the following oclwrs:
 *    - @a primitive is NULL
 *    - @a data is NULL
 *    - @a length is NULL
 *    - fail to append data in transport buffer
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Exporting of @a primitive requires thread synchronization; without
 *   synchronization, the function could cause a memory leak when the function
 *   is called from more than one thread in parallel with the same value for
 *   @a data. No synchronization is done in the function. To ensure that there
 *   is no memory leak, the user must ensure that the function is not called
 *   from other threads in parallel with the same value for @a data.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 *  \implements{18844770}
 */
LwSciError LwSciSyncCorePrimitiveExport(
    LwSciSyncCorePrimitive primitive,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** data,
    size_t* length);

/**
 * \brief Imports the export descriptor into LwSciSyncCorePrimitive.
 *
 * Allocates memory for syncpoint specific data with LwSciCommonCalloc().
 * Retrieves LwSciSyncModule's host1x node with the sequence of
 * LwSciSyncCoreAttrListGetModule(), LwSciSyncCoreModuleGetRmBackEnd(),
 * LwSciSyncCoreRmGetHost1xHandle().
 */
#if (LW_IS_SAFETY == 0)
/**
 * Imports the tags with LwSciCommonTransportGetRxBufferAndParams(),
 * LwSciCommonTransportGetNextKeyValuePair(). Allocates memory for syncpoint ids
 * with LwSciCommonCalloc() and copies them from the export descriptor with
 * LwSciCommonMemcpyS(). Sets hasExternalPrimitiveInfo based on the result of
 * LwSciSyncAttrListGetSingleInternalAttr().
 */
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * Retreives permissions and needCpuAccess LwSciSyncAttrListGetAttrs().
 * Compares permissions usng LwSciSyncCorePermLEq() and checks whether
 * @a ipcEndpoint is C2C with LwSciSyncCoreIsTopoIdC2c(). Gets C2C function
 * table with LwSciIpcGetC2cCopyFuncSet(). Imports the C2C syncHandle from
 * the token in the export descriptor with c2cCopyFuncs.syncGetHandleFromAuthToken().
 * If the object needs cpu access, the function maps syncpoint's shim memory
 * with c2cCopyFuncs.syncCreateCpuMapping().
 */
#endif
#endif
/**
 *
 * \param[in] ipcEndpoint LwSciIpcEndpoint through which import happens
 * \param[in] reconciledList reconciled LwSciSyncAttrList
 * \param[in] data export descriptor containing LwSciSyncCorePrimitive information.
 *  Valid value: data is valid input if it not NULL.
 * \param[in] len length of the export descriptor.
 *  Valid value: len is valid input if it greater than 0.
 * \param[out] primitive LwSciSyncCorePrimitive to be initialized
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_ResourceError if something went wrong with LwSciIpc
 *   or ipcEndpoint is not valid
 * - LwSciError_BadParameter if data is not a valid primitive descriptor
 * - LwSciError_InsufficientMemory if no memory to allocate a new
 *   LwSciSyncCorePrimitive
 * - LwSciError_Overflow if data export descriptor is too big
 * - Panics if any of the following oclwrs:
 *      - @a data is NULL
 *      - @a len is 0
 *      - @a primitive is NULL
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Exporting of @a primitive requires thread synchronization; without
 *   synchronization, the function could cause a memory leak when the function
 *   is called from more than one thread in parallel with the same value for
 *   @a data. No synchronization is done in the function. To ensure that there
 *   is no memory leak, the user must ensure that the function is not called
 *   from other threads in parallel with the same value for @a data.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 *  \implements{18844773}
 */
LwSciError LwSciSyncCorePrimitiveImport(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList reconciledList,
    const void* data,
    size_t len,
    LwSciSyncCorePrimitive* primitive);

/**
 * \brief Signals the LwRmHost1xSyncpointHandle contained in
 * LwSciSyncCoreSyncpointInfo of the input LwSciSyncCorePrimitive using
 * LwRmHost1xSyncpointIncrement().
 **/
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * If the syncpoint handle is unavailable, it uses a write to the mapped
 * syncpoint shim memory instead.
 */
#endif
/**
 *
 * \param[in] primitive LwSciSyncCorePrimitive to be signaled
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_ResourceError if failed to signal host1x syncpoint
 * - Panics if @a primitive is NULL or signaling LwSciSyncCorePrimitive not
 *   supported.
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
 *  \implements{18844776}
 */
LwSciError LwSciSyncCoreSignalPrimitive(
    LwSciSyncCorePrimitive primitive);

/**
 * \brief Waits on the input syncpoint id and threshold value using
 * LwRmHost1xSyncpointWait(). The LwRmHost1xWaiterHandle needed for wait is
 * fetched from LwSciSyncCoreRmWaitContextBackEnd contained in
 * LwSciSyncCpuWaitContext.
 *
 * \param[in] primitive contains info about LwSciSyncCorePrimitive
 * \param[in] waitContext LwSciSyncCpuWaitContext that can be used to perform
 * a CPU wait
 * \param[in] id primitive identifier
 *  Valid value: [0, UINT32_MAX]
 * \param[in] value threshold value to be waited for
 *  Valid value: [0, UINT32_MAX]
 * \param[in] timeout_us timeout to wait for in micro seconds
 *  Valid value: [-1, LwSciSyncFenceMaxTimeout]
 *
 * \return LwSciError
 * - LwSciError_Success if wait is successful
 * - LwSciError_BadParameter if timeout_us is invalid
 * - LwSciError_Overflow if @a id or @a value is larger than UINT32_MAX
 * - LwSciError_ResourceError if signal operation failed
 * - LwSciError_Timeout if fence did not expire in given timeout
 * - Panics if any of the following oclwrs:
 *   - @a primitive is NULL
 *   - @a waitContext is NULL
 *   - waiting on primitive not supported
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
 *  \implements{18844779}
 */
LwSciError LwSciSyncCoreWaitOnPrimitive(
    LwSciSyncCorePrimitive primitive,
    LwSciSyncCpuWaitContext waitContext,
    uint64_t id,
    uint64_t value,
    int64_t timeout_us);

/**
 * \brief Fills the provided buffer with the list of CPU supported primitives
 * which is LwSciSyncInternalAttrValPrimitiveType_Syncpoint.
 *
 * \param[out] primitiveType buffer where supported primitives is copied
 * \param[in] len size of the output buffer in bytes
 *
 * \return void
 * - Panics if @a primitiveType is NULL.
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
 *  \implements{18844782}
 */
void LwSciSyncCoreCopyCpuPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len);

/**
 * \brief Fills the provided buffer with the list of CPU supported primitives
 * for C2C usecase.
 *
 * \param[out] primitiveType buffer where supported primitives is copied
 * \param[in] len size of the output buffer in bytes
 *
 * \return void
 * - Panics if @a primitiveType is NULL.
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
 *  \implements{}
 */
void LwSciSyncCoreCopyC2cCpuPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len);

/**
 * \brief Fills the provided buffer with the list of supported primitives
 * which is LwSciSyncInternalAttrValPrimitiveType_Syncpoint.
 *
 * \param[out] primitiveType buffer where supported primitives is copied
 * \param[in] len size of the output buffer in bytes
 *
 * \return void
 * - Panics if @a primitiveType is NULL.
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
 *  \implements{18844785}
 */
void LwSciSyncCoreGetSupportedPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len);

/**
 * \brief Fills the provided buffer with the list of deterministic primitives.
 *
 * \param[out] primitiveType buffer where deterministic primitives is copied
 * \param[in] len size of the output buffer in bytes
 *
 * \return void
 * - Panics if @a primitiveType is NULL.
 *
 *  \implements{}
 */
void LwSciSyncCoreGetDeterministicPrimitives(
    LwSciSyncInternalAttrValPrimitiveType* primitiveType,
    size_t len);

/**
 * \brief Increments and returns the lastFence value of the
 * LwSciSyncCorePrimitive, wrapping when the maximum value for the underlying
 * primitive is reached.
 *
 * \param[in] primitive contains info about LwSciSyncCorePrimitive
 *
 * \return updated snapshot of the underlying primitive
 * - Panics if @a primitive is NULL or getting new fence for
 *   LwSciSyncCorePrimitive not supported.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Incrementing the lastFence value requires thread synchronization; without
 *   synchronization, the function could cause an incorrect lastFence value to
 *   be produced if the function is called in parallel from multiple threads.
 *   No synchronization is done in the function. To ensure that lastFence is
 *   set to correct value, the user must ensure that the function is not
 *   called in parallel from multiple threads.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 *  \implements{18844788}
 */
uint64_t LwSciSyncCorePrimitiveGetNewFence(
    LwSciSyncCorePrimitive primitive);


/**
 * \brief Get id from the input LwSciSyncCorePrimitive.
 *
 * \param[in] primitive LwSciSyncCorePrimitive to get id from
 *
 * \return id of the underlying primitive
 * - Panics if primitive is NULL.
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
 *  \implements{18844791}
 */
uint64_t LwSciSyncCorePrimitiveGetId(
    LwSciSyncCorePrimitive primitive);

/**
 * \brief Return the primitive specific data of underlying primitive
 *
 * \param[in] primitive contains info about primitive
 * \param[in, out] data pointer where requested data is written
 *
 * \return LwSciError
 * - LwSciError_Success if primitive operation is supported
 * - LwSciError_BadParameter if primitive operation is not supported
 */
LwSciError LwSciSyncCorePrimitiveGetSpecificData(
    LwSciSyncCorePrimitive primitive,
    void** data);

/**
 * \brief Validate id and value for selected backend primitive.
 *
 * \param[in] primitive contains info about LwSciSyncCorePrimitive
 * \param[in] id primitive identifier
 * Valid value: [0, UINT32_MAX-1] for LwSciSyncInternalAttrValPrimitiveType_Syncpoint.
 * [0, value returned by LwSciSyncObjGetNumPrimitives()-1] for
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore and
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b.
 * \param[in] value threshold value to be waited for
 * Valid value: [0, UINT32_MAX] for LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * and LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore.
 * [0, UINT64_MAX] for
 * LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b.
 *
 * \return LwSciError
 * - LwSciError_Success if @a id or @a value are valid for selected backend primitive
 * - LwSciError_Overflow if invalid @a id or @a value
 * - Panics if any of the following oclwrs:
 *   - @a primitive is NULL
 *
 *  \implements{TODO}
 */
LwSciError LwSciSyncCoreValidatePrimitiveIdValue(
    LwSciSyncCorePrimitive primitive,
    uint64_t id,
    uint64_t value);

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * \brief Retrieve the C2c syncHandle associated with the primitive.
 *
 * \param[in] primitive contains info about LwSciSyncCorePrimitive
 * \param[out] syncHandle C2CSyncHandle written during primitive export via C2C
 *
 * \return LwSciError
 * - LwSciError_Success if successfully retrieved the syncHandle
 * - LwSciError_BadParameter if @a primitive is not backed by syncpoint
 * - LwSciError_NotInitialized if @a primitive was not transported via C2C
 * - Panics if @a primitive is NULL or invalid or @a syncHandle is NULL
 *
 *  \implements{TODO}
 */
LwSciError LwSciSyncCorePrimitiveGetC2cSyncHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncHandle* syncHandle);

/**
 * \brief Retrieve the underlying syncpointHandle
 *
 * \param[in] primitive contains info about LwSciSyncCorePrimitive
 * \param[out] syncRmHandle C2C RM handle allocated with this primitive
 *
 * \return LwSciError
 * - LwSciError_Success if successfully retrieved the syncHandle
 * - LwSciError_BadParameter if @a primitive is not backed by syncpoint
 * - LwSciError_NotInitialized if @a primitive does not own a syncpoint
 * - Panics if @a primitive is NULL or invalid or @a syncRmHandle is NULL
 *
 *  \implements{TODO}
 */
LwSciError LwSciSyncCorePrimitiveGetC2cRmHandle(
    LwSciSyncCorePrimitive primitive,
    LwSciC2cPcieSyncRmHandle* syncRmHandle);
#endif

/**
 * \brief Translate imported fence threshold to be locally-based
 *
 * \param[in] primitive contains info about LwSciSyncCorePrimitive
 * \param[inout] threshold threshold to be translated
 *
 * \return LwSciError
 * - LwSciError_Success if successfully translated @a threshold
 * - Panics if @a primitive is NULL or invalid or @a threshold is NULL
 *
 *  \implements{TODO}
 */
LwSciError LwSciSyncCorePrimitiveImportThreshold(
    LwSciSyncCorePrimitive primitive,
    uint64_t* threshold);

#endif
