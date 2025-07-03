/*
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCISYNC_OBJECT_EXTERNAL_H
#define INCLUDED_LWSCISYNC_OBJECT_EXTERNAL_H

#include "lwscierror.h"
#include "lwscisync.h"

#include "lwscisync_c2c_priv.h"

/**
 * @defgroup lwsci_sync Synchronization APIs
 * @{
 */

/**
 * \page lwscisync_page_unit_blanket_statements LwSciSync blanket statements
 * \section lwscisync_in_out_params Input/Output parameters
 * - LwSciSyncObj descriptor passed as input parameter to an API is
 *   validated by checking for the correct tags and their correct order.
 */

/******************************************************
 *                  Public functions
 ******************************************************/

/**
 * The function checks to ensure the input LwSciSyncAttrList is actually
 * reconciled using LwSciSyncAttrListIsReconciled. If so, it proceeds to
 * allocate memory for a LwSciSyncCoreObj using LwSciCommonAllocObjWithRef,
 * and initializes the object. This includes generating a header using
 * LwSciSyncCoreGenerateObjHeader for validation. A reference to the
 * reconciled LwSciSyncAttrList is taken using LwSciSyncCoreAttrListDup.
 * Subsequently, an ID is generated for the LwSciSyncObj, scoped to the
 * LwSciSyncModule associated with the LwSciSyncAttrList. This is done by
 * obtaining the module using LwSciSyncCoreAttrListGetModule, and then using a
 * monotonic counter LwSciSyncCoreModuleCntrGetNextValue. Using
 * LwSciSyncAttrListGetSingleInternalAttr, the reconciled primitive type is
 * fetched from the reconciled LwSciSyncAttrList. LwSciCommonMemcpyS is used to
 * copy this value. The function then determines whether LwSciSync is
 * responsible for allocating a primitive (or if it is the responsibility of
 * the caller) by calling LwSciSyncCoreGetSignalerUseExternalPrimitive. This
 * is passed onto LwSciSyncCoreInitPrimitive to actually perform the
 * allocation.
 * Timestamps are initialized by LwSciSyncCoreTimestampsInit.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Allocation of an underlying LwSciSync object requires thread
 *   synchronization; without synchronization, the function could cause a
 *   memory leak when the function is called from more than one thread in
 *   parallel with the same value for @a syncObj. No synchronization is done
 *   in the function. To ensure that there is no memory leak, the user must
 *   ensure that the function is not called from other threads in parallel
 *   with the same value for @a syncObj.
 * - Duplication of @a reconciledList and increment of the associated
 *   reference count requires thread synchronization. This synchronization is
 *   done using LwSciRef.refLock mutex object associated with @a
 *   reconciledList.
 * - Increment of the of the module counter associated with @a reconciledList
 *   requires thread synchronization. This synchronization is done using
 *   LwSciSyncModule.refModule mutex object associated with @a reconciledList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since:
 * - the mentioned LwSciRef.refLock is locked immediately before and released
 *   immediately after @a reconciledList reference count is incremented,
 * - the mentioned LwSciSyncModule.refModule is locked immediately before and
 *   released immediately after module counter is incremented,
 *
 * \implements{18844701}
 *
 * \fn LwSciError LwSciSyncObjAlloc(
 *     LwSciSyncAttrList reconciledList,
 *     LwSciSyncObj* syncObj);
 */

/**
 * Duplicates the reference to the underlying LwSciSyncCoreObj pointed to by
 * @a syncObj using LwSciCommonDuplicateRef.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to dereferenced @a dupObj value requires thread synchronization;
 *   without synchronization, the function could cause system error. No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, the user must ensure that the dereferenced @a dupObj value is not
 *   modified during the call to the function.
 * - Allocation of an underlying LwSciSync object for @a dupObj requires
 *   thread synchronization; without synchronization, the function could cause
 *   a memory leak when the function is called from more than one thread in
 *   parallel with the same value for @a dupObj. No synchronization is done in
 *   the function. To ensure that there is no memory leak, the user must
 *   ensure that the function is not called from other threads in parallel
 *   with the same value for @a dupObj.
 * - Increment of the of the reference counter associated with @a syncObj
 *   requires thread synchronization. This synchronization is done using
 *   LwSciRef.refLock mutex object associated with @a dupObj.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned LwSciRef.refLock is locked immediately before and
 *   released immediately after @a dupObj reference count is incremented.
 *
 * \implements{18844704}
 *
 * \fn LwSciError LwSciSyncObjDup(
 *     LwSciSyncObj syncObj,
 *     LwSciSyncObj* dupObj);
 */

/**
 * Frees the backing structure using LwSciSyncObjFreeObjAndRef.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Acces to the object pointed to by @a syncObj value requires thread
 *   synchronization; without synchronization, the function could cause a
 *   system error when called in parallel with a function which modifies the
 *   mentioned object. No synchronization is done in the function. To ensure
 *   that no system error oclwrs, the user must ensure that the object is not
 *   modified during the call to the function.
 * - Releasing of LwSciRef associated with the @a syncObj requires thread
 *   synchronization. This synchronization is done using LwSciRef.refLock
 *   mutex object associated with @a syncObj.
 * - Releasing of the object pointed to by @a syncObj value requires thread
 *   synchronization. This synchronization is done using LwSciObj.objLock
 *   mutex object associated with @a syncObj.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since:
 * - the mentioned LwSciRef.refLock is locked immediately before and destroyed
 *   immediately after LwSciRef object is released,
 * - the mentioned LwSciObj.objLock is locked immediately before and destroyed
 *   immediately after LwSciObj object is released.
 *
 * \implements{18844707}
 *
 * \fn void LwSciSyncObjFree(
 *     LwSciSyncObj syncObj);
 */

/**
 * The backing primitive associated with the LwSciSyncObj is first serialized
 * for transport via LwSciSyncCorePrimitiveExport. Space for the rest of the
 * Transport Keys to be serialized is allocated by a call to
 * LwSciCommonTransportAllocTxBufferForKeys. The expected IPC export
 * permissions are then callwlated via LwSciSyncCoreAttrListGetIpcExportPerm,
 * and compared with LwSciSyncCoreAttrListPermLessThan to determine
 * whether the expected permissions match the requested permissions. If the
 * requested permission is LwSciSyncAccessPerm_Auto, then the previously
 * computed value from LwSciSyncCoreAttrListGetIpcExportPerm is taken, which is
 * the expected access permission.
 *
 * Then all the Transport Keys are appended to the Transport Buffer via
 * LwSciCommonTransportAppendKeyValuePair. This exports the following
 * Transport Keys: LwSciSyncCoreObjKey_AccessPerm,
 * LwSciSyncCoreObjKey_ModuleCnt, LwSciSyncCoreObjKey_IpcEndpoint,
 * LwSciSyncCoreObjKey_CorePrimitive.
 *
 * This buffer is prepared by calling LwSciCommonTransportPrepareBufferForTx,
 * and the temporary buffers allocated in LwSciSyncCorePrimitiveExport
 * and LwSciCommonTransportAllocTxBufferForKeys  are subsequently freed using
 * LwSciCommonTransportBufferFree and LwSciCommonFree. A separate Transport
 * Buffer is also allocated to store the serialized LwSciSyncObj, which holds
 * the previously allocated Transport Buffer. This is done via a call to
 * LwSciCommonTransportAllocTxBufferForKeys and the key is added via
 * LwSciCommonTransportAppendKeyValuePair. This new buffer is finally
 * serialized again via LwSciCommonTransportPrepareBufferForTx, and then
 * copied to the output parameter via LwSciCommonMemcpyS.
 *
 * At the end, temporary buffers from the latest calls to
 * LwSciCommonTransportAllocTxBufferForKeys and
 * LwSciCommonTransportPrepareBufferForTx are freed using LwSciCommonFree and
 * LwSciCommonTransportBufferFree.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to dereferenced @a desc value requires thread synchronization;
 *   without synchronization, the function could cause a memory leak when the
 *   function is called from more than one thread in parallel with the same
 *   value for @a desc. No synchronization is done in the function. To ensure
 *   that there is no memory leak, the user must ensure that the function is
 *   not called from other threads in parallel with the same value for @a
 *   desc.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844710}
 *
 * \fn LwSciError LwSciSyncObjIpcExport(
 *     LwSciSyncObj syncObj,
 *     LwSciSyncAccessPerm permissions,
 *     LwSciIpcEndpoint ipcEndpoint,
 *     LwSciSyncObjIpcExportDescriptor* desc);
 */

/**
 * The function checks to ensure the input LwSciSyncAttrList is actually
 * reconciled using LwSciSyncAttrListIsReconciled. If so, it proceeds to make
 * a copy of the LwSciSyncObjIpcExportDescriptor using LwSciCommonCalloc and
 * LwSciCommonMemcpyS. A new LwSciSyncCoreObj is allocated using the
 * functionality provided LwSciCommonAllocObjWithRef. The object's header is
 * computed and set using LwSciSyncCoreGenerateObjHeader. Then, a reference on
 * the reconciled LwSciSync LwSciSyncAttrList is taken using
 * LwSciSyncAttrListClone. LwSciCommonTransportGetRxBufferAndParams is used to
 * read the serialized data from the binary descriptor. The compatibility
 * between the LwSciSync version and the received serialized descriptor is
 * checked with a help of LwSciSyncCoreGetLibVersion.
 * Then the LwSciSyncObj is imported by fetching the next key value
 * using LwSciCommonTransportGetNextKeyValuePair. If this is the Transport key
 * denoting a LwSciSyncObj, the function fetches the serialized data for the
 * LwSciSyncObj using another call to LwSciCommonTransportGetRxBufferAndParams,
 * and then iterates over the Transport keys using the functionality provided
 * by LwSciCommonTransportGetNextKeyValuePair. The following Transport Keys
 * defined in LwSciSyncCoreObjKey are expected to be received:
 * LwSciSyncCoreObjKey_AccessPerm, LwSciSyncCoreObjKey_ModuleCnt,
 * LwSciSyncCoreObjKey_IpcEndpoint, LwSciSyncCoreObjKey_CorePrimitive.
 *
 * When importing LwSciSyncCoreObjKey_AccessPerm, if the expected permissions
 * is set to LwSciSyncAccessPerm_Auto, the function uses the computed expected
 * permissions from LwSciSyncAttrListGetAttr. Using
 * LwSciSyncCoreAttrListPermLessThan, the function checks to ensure that any
 * permissions required were actually granted. If all this is successful, the
 * permission is set via LwSciSyncCoreAttrListSetActualPerm.
 *
 * When importing LwSciSyncCoreObjKey_ModuleCnt, the value is set on the
 * @a moduleCntr of the LwSciSyncCoreObjId associated with the LwSciSyncObj.
 * This is used in conjunction with the value imported from the
 * LwSciSyncCoreObjKey_IpcEndpoint Transport Key to create an identifier for
 * the imported LwSciSyncObj.
 *
 * When importing LwSciSyncCoreObjKey_IpcEndpoint, the value is set on the
 * @a ipcEndpoint of the LwSciSyncCoreObjId associated with the LwSciSyncObj.
 * This is used in conjunction with the value imported from the
 * LwSciSyncCoreObjKey_ModuleCnt Transport Key to create an identifier for
 * the imported LwSciSyncObj.
 *
 * When importing LwSciSyncCoreObjKey_CorePrimitive, the value is set using
 * LwSciSyncCorePrimitiveImport.
 *
 * The transport buffers are then all freed using LwSciCommonTransportBufferFree.
 * The copy of the whole descriptor is freed with LwSciCommonFree.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to dereferenced @a syncObj value requires thread synchronization;
 *   without synchronization, the function could cause a memory leak when the
 *   function is called from more than one thread in parallel with the same
 *   value for @a syncObj. No synchronization is done in the function. To
 *   ensure that there is no memory leak, the user must ensure that the
 *   function is not called from other threads in parallel with the same value
 *   for @a syncObj.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844713}
 *
 * \fn LwSciError LwSciSyncObjIpcImport(
 *     LwSciIpcEndpoint ipcEndpoint,
 *     const LwSciSyncObjIpcExportDescriptor* desc,
 *     LwSciSyncAttrList inputAttrList,
 *     LwSciSyncAccessPerm permissions,
 *     int64_t timeoutUs,
 *     LwSciSyncObj* syncObj);
 */

/**
 * The reconciled LwSciSyncAttrList associated with the input LwSciSyncObj is
 * retrieved with LwSciSyncObjGetAttrList(), in order to ensure that the
 * reconciled LwSciSyncAttrList is for a CPU signaler or C2C signaler  by calling
 * LwSciSyncCoreAttrListTypeIsCpuSignaler() and
 * LwSciSyncCoreAttrListTypeIsC2cSignaler(). Validation is also performed to
 * ensure that LwSciSync is managing the allocated primitive, and not the
 * caller, by calling LwSciSyncCoreGetSignalerUseExternalPrimitive(). The
 * function generates the next snapshot of the primitive using
 * LwSciSyncCorePrimitiveGetNewFence(). Then, an identifier for the fence is
 * obtained using LwSciSyncCorePrimitiveGetId. The output Fence is finally
 * updated with the snapshot using LwSciSyncFenceUpdateFence().
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - Generating next point on sync timeline of @a syncObj requires thread
 *   synchronization. This synchronization is done using refObj mutex object
 *   associated with @a syncObj.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned refObj is locked immediately before and released
 *   immediately after generation of the timeline point.
 *
 * \implements{18844716}
 *
 * \fn LwSciError LwSciSyncObjGenerateFence(
 *     LwSciSyncObj syncObj,
 *     LwSciSyncFence* syncFence);
 */

/**
 * The reconciled LwSciSyncAttrList associated with the input LwSciSyncObj is
 * retrieved with LwSciSyncObjGetAttrList, in order to ensure that the
 * reconciled LwSciSyncAttrList is for a CPU signaler using
 * LwSciSyncCoreAttrListTypeIsCpuSignaler. Validation is also performed to
 * ensure that LwSciSync is managing the allocated primitive, and not the
 * caller, by calling LwSciSyncCoreGetSignalerUseExternalPrimitive. Finally,
 * the primitive is signaled using LwSciSyncCoreSignalPrimitive.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - Signaling the @a syncObj requires thread synchronization. This
 *   synchronization is done using refObj mutex object associated with @a
 *   syncObj.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned refObj is locked immediately before and released
 *   immediately after signaling the @a syncObj.
 *
 * \implements{18844719}
 *
 * \fn LwSciError LwSciSyncObjSignal(
 *     LwSciSyncObj syncObj);
 */

/**
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to @a inputArray array requires thread synchronization; without
 *   the synchronization, the function could access invalid LwSciSyncAttrList
 *   objects, in turn causing a system error or calling LwSciCommonPanic(). No
 *   synchronization is done in the function. To ensure that no errors oclwrs,
 *   and that correct objects are accessed, the user mush ensure that the the
 *   elements are not modified during the call to the function.
 * - Access to elements of the @a inputArray array requires thread
 *   synchronization while the elements are being reconciled. This
 *   synchronization is done using LwSciObj.objLock mutex object associated
 *   with each of the elements.
 * - Locking of elements of the @a inputArray array requires thread
 *   synchronization; without synchronization, the function could cause a
 *   deadlock when called in parallel from two or more threads. This
 *   synchronization is done by locking the elements in a specific order, and
 *   this order is the same during every call to the function from every
 *   thread.
 * - Allocation of an underlying LwSciSync object requires thread
 *   synchronization; without synchronization, the function could cause a
 *   memory leak when the function is called from more than one thread in
 *   parallel with the same value for @a syncObj. No synchronization is done
 *   in the function. To ensure that there is no memory leak, the user must
 *   ensure that the function is not called from other threads in parallel
 *   with the same value for @a syncObj.
 * - Duplication of the reconciled list and increment of the associated
 *   reference count requires thread synchronization. This synchronization is
 *   done using LwSciRef.refLock mutex object associated with @a
 *   reconciledList.
 * - Increment of the of the module counter associated with the reconciled
 *   list requires thread synchronization. This synchronization is done using
 *   LwSciSyncModule.refModule mutex object associated with the reconciled
 *   list.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since:
 * - the mentioned LwSciObj.objLock objects are locked immediately before and
 *   released immediately after the elements are reconciled,
 * - the mentioned LwSciObj.objLock objects are locked in a specific order,
 *   and this order is the same during every call to the function from every
 *   thread,
 * - the mentioned LwSciRef.refLock is locked immediately before and released
 *   immediately after the reconciled list reference count is incremented,
 * - the mentioned LwSciSyncModule.refModule is locked immediately before and
 *   released immediately after module counter is incremented.
 *
 * \implements{18844722}
 *
 * \fn LwSciError LwSciSyncAttrListReconcileAndObjAlloc(
 *     const LwSciSyncAttrList inputArray[],
 *     size_t inputCount,
 *     LwSciSyncObj* syncObj,
 *     LwSciSyncAttrList* newConflictList);
 */

/**
 * The reconciled attribute list is retrieved with LwSciSyncObjGetAttrList.
 * The resulting binary descriptor representing the LwSciSyncObj and reconciled
 * LwSciSync LwSciSyncAttrList is copied to a newly allocated object using
 * LwSciCommonCalloc and LwSciCommonMemcpyS before a reference on the
 * LwSciSync reconciled LwSciSyncAttrList is released.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to dereferenced @a attrListAndObjDesc value requires thread
 *   synchronization; without synchronization, the function could cause system
 *   error. No synchronization is done in the function. To ensure that no
 *   system error oclwrs, the user must ensure that the dereferenced @a
 *   attrListAndObjDesc value is not modified during the call to the function.
 * - This access could also cause a memory leak when the function is called
 *   from more than one thread in parallel with the same value for @a
 *   attrListAndObjDesc. No synchronization is done in the function. To ensure
 *   that there is no memory leak, the user must ensure that the function is
 *   not called from other threads in parallel with the same value for @a
 *   attrListAndObjDesc.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844725}
 *
 * \fn LwSciError LwSciSyncIpcExportAttrListAndObj(
 *     LwSciSyncObj syncObj,
 *     LwSciSyncAccessPerm permissions,
 *     LwSciIpcEndpoint ipcEndpoint,
 *     void** attrListAndObjDesc,
 *     size_t* attrListAndObjDescSize);
 */

/**
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the dereferenced @a attrListAndObjDescBuf value requires thread
 *   synchronization; without synchronization, the function could cause a call
 *   to LwSciCommonPanic() when called in parallel with another function,
 *   which modifies memory starting sizeof(LwSciCommonAllocHeader) bytes
 *   before the dereferenced @a attrListAndObjDescBuf value. No
 *   synchronization is done in the function. To ensure that
 *   LwSciCommonPanic() is not called, the user must ensure that the mentioned
 *   memory is not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844728}
 *
 * \fn void LwSciSyncAttrListAndObjFreeDesc(
 *     void* attrListAndObjDescBuf);
 */

/**
 * Conlwrrency:
 * - Thread-safe: no
 * - Incrementing the reference count of the @a module requires thread
 *   synchronization. This synchronization is done by locking LwSciObj.ObjLock
 *   mutex object associated with @a module.
 * - Access to memory starting from value of @a attrListAndObjDesc requires
 *   thread synchronization; without synchronization, the function could cause
 *   system error when called in parallel with a function which modifies the
 *   mentioned memory. No synchronization is done in the function. To ensure
 *   that no system error oclwrs, the user must ensure that the mentioned
 *   memory is not modified during the call to the function.
 * - Access to dereferenced @a attrListAndObjDesc value requires thread
 *   synchronization; without synchronization, the function could cause a
 *   memory leak when the function is called from more than one thread in
 *   parallel with the same value for @a syncObj. No synchronization is done
 *   in the function. To ensure that there is no memory leak, the user must
 *   ensure that the function is not called from other threads in parallel
 *   with the same value for @a syncObj.
 * - Access to the @a attrList array requires thread synchronization; without
 *   the synchronization, the function could access invalid LwSciSyncAttrList
 *   objects, in turn causing a system error or calling LwSciCommonPanic(). No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, and that correct objects are accessed, the user must ensure that
 *   the elements are not modified during the call to the function.
 * - Access to elements of the @a attrList array requires thread
 *   synchronization; without the synchronization, the function could read
 *   incorrect or invalid attribute values from the elements. This
 *   synchronization is done using LwSciObj.objLock mutex associated with each
 *   of the elements.
 * - Locking of elements of the @a attrList array requires thread
 *   synchronization; without synchronization, the function could cause a
 *   deadlock when called in parallel from two or more threads. This
 *   synchronization is done by locking the elements in a specific order, and
 *   this order is the same during every call to the function from every
 *   thread.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since:
 * - the mentioned LwSciObj.ObjLock mutex object associated with @a module is
 *   locked immediately before and released immediately after the reference
 *   count is incremented,
 * - the mentioned @a attrList array element mutex object is locked
 *   immediately before and released immediately after the attribute values
 *   are set,
 * - the mentioned @a attrList array elements mutex objects are locked in a
 *   specific order, and this order is the same during every call to the
 *   function from every thread.
 *
 * \implements{18844731}
 *
 * \fn LwSciError LwSciSyncIpcImportAttrListAndObj(
 *     LwSciSyncModule module,
 *     LwSciIpcEndpoint ipcEndpoint,
 *     const void* attrListAndObjDesc,
 *     size_t attrListAndObjDescSize,
 *     LwSciSyncAttrList const attrList[],
 *     size_t attrListCount,
 *     LwSciSyncAccessPerm minPermissions,
 *     int64_t timeoutUs,
 *     LwSciSyncObj* syncObj);
 */


#if (LW_IS_SAFETY == 0)  && (LW_L4T == 0)
/**
 * \brief Retrieves the C2c handle associated with the syncObj.
 *
 * Validates the @a syncObj with LwSciSyncCoreObjValidate(). Retrieves
 * the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef().
 * Retrieves the syncHandle with LwSciSyncCorePrimitiveGetC2cSyncHandle().
 *
 * @param[in] syncObj input LwSciSyncObj to retrieve syncHandle from
 * @param[out] syncHandle C2c handle associated with @a syncObj.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a syncObj or @a syncHandle is NULL,
 *       - @a syncObj is not associated with syncpoints,
 * - LwSciError_NotInitialized if @a syncObj has no associated syncHandle
 * - Panics if:
 *       - @a syncObj is invalid.
 */
LwSciError LwSciSyncObjGetC2cSyncHandle(
    LwSciSyncObj syncObj,
    LwSciC2cInterfaceSyncHandle* syncHandle);

/**
 * \brief Retrieves the syncpoint handle associated with the syncObj.
 *
 * Validates the @a syncObj with LwSciSyncCoreObjValidate(). Retrieves
 * the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef().
 * Retrieves the syncpointHandle with
 * LwSciSyncCorePrimitiveGetC2cRmHandle().
 *
 * @param[in] syncObj input LwSciSyncObj to retrieve syncHandle from
 * @param[out] syncRmHandle C2C RM handle associated with @a syncObj.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *       - @a syncObj or @a syncRmHandle is NULL,
 *       - @a syncObj is not associated with syncpoints,
 * - LwSciError_NotInitialized if @a syncObj has no associated syncpoint handle
 * - Panics if:
 *       - @a syncObj is invalid.
 */
LwSciError LwSciSyncCoreObjGetC2cRmHandle(
    LwSciSyncObj syncObj,
    LwSciC2cPcieSyncRmHandle* syncRmHandle);
#endif

/******************************************************
 *                 Internal functions
 ******************************************************/

/**
 * First it retrieves the LwSciSyncAttrList with LwSciSyncObjGetAttrList.
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
 * \implements{18844734}
 *
 * \fn LwSciError LwSciSyncObjGetNumPrimitives(
 *     LwSciSyncObj syncObj,
 *     uint32_t* numPrimitives);
 */

/**
 * First it retrieves the LwSciSyncAttrList with LwSciSyncObjGetAttrList.
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
 * \implements{18844737}
 *
 * \fn LwSciError LwSciSyncObjGetPrimitiveType(
 *     LwSciSyncObj syncObj,
 *     LwSciSyncInternalAttrValPrimitiveType* primitiveType);
 */

 /** @} */

#endif
