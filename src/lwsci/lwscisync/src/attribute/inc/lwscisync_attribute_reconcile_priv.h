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
 * \brief <b>LwSciSync attribute reconcile definitions</b>
 *
 * @b Description: This file declares items exposed by attribute reconcile unit
 */

#ifndef INCLUDED_LWSCISYNC_ATTRIBUTE_RECONCILE_PRIV_H
#define INCLUDED_LWSCISYNC_ATTRIBUTE_RECONCILE_PRIV_H

/**
 * @defgroup lwsci_sync Synchronization APIs
 *
 * @ingroup lwsci_group_stream
 * @{
 */

#include "lwscisync_attribute_core.h"
#include "lwscisync_attribute_core_cluster.h"

/**
 * \brief Create and fill LwSciBufAttrList for the timestamp buffer
 *
 * \param[in,out] objAttrList LwSciSyncAttrList to hold the timestamp buffer
 *  attributes
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if there is not enough memory to create
 *   LwSciBufAttrList
 */
LwSciError LwSciSyncCoreFillTimestampBufAttrList(
    const LwSciSyncCoreAttrListObj* objAttrList);

/**
 * \brief Create and fill LwSciBufAttrList for the semaphore buffer
 *
 * \param[in,out] objAttrList LwSciSyncAttrList to hold the semaphore buffer attributes
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if there is not enough memory to create
 *   LwSciBuf LwSciSyncAttrList
 */
LwSciError LwSciSyncCoreFillSemaAttrList(
    LwSciSyncCoreAttrListObj* objAttrList);


/**
 * \brief Acts the same as LwSciSyncAttrListValidateReconciled but allows
 * explicit requiring of locking.
 *
 * \param[in] reconciledAttrList Reconciled LwSciSyncAttrList to be validated.
 * \param[in] inputUnreconciledAttrListArray Array containing the unreconciled
 * LwSciSyncAttrLists used for validation.
 * Valid value: Array of valid unreconciled LwSciSyncAttrLists.
 * \param[in] inputUnreconciledAttrListCount number of elements/indices in
 * @a inputUnreconciledAttrListArray.
 * Valid value: [1, SIZE_MAX]
 * \param[in] acquireLocks indicates if all LwSciSyncAttrList in @a
 * inputUnreconciledAttrListArray should be locked before performing any operation.
 * Valid value: acquireLocks is valid if it is either true or false.
 * \param[out] isReconciledListValid A pointer to a boolean to store whether the
 * @a reconciled LwSciSyncAttrLists satisfies the parameters of set of
 * unreconciled LwSciSyncAttrList or not.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a inputUnreconciledAttrListArray is NULL
 *         - @a inputUnreconciledAttrListCount is 0
 *         - @a isReconciledListValid is NULL
 *         - any of the input LwSciSyncAttrLists are reconciled
 *         - @a reconciledAttrList is NULL or not reconciled
 *         - not all the LwSciSyncAttrLists in @a inputUnreconciledAttrListArray
 *           and the @a reconciledAttrList are bound to the same LwSciSyncModule
 *           instance.
 *         - reconciled LwSciSyncAttrList does not satisfy the unreconciled
 *           LwSciSyncAttrLists requirements.
 * - ::LwSciError_InsufficientMemory if there is insufficient system memory
 *   to create temporary data structures
 * - ::LwSciError_Overflow if internal integer overflow oclwrs.
 * - Panics if @a reconciledAttrList or any of the input unreconciled
 *   LwSciSyncAttrList are not valid.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to @a reconciledAttrList requires thread synchronization; without
 *   synchronization, the function could cause a system error or a call to
 *   LwSciCommonPanic() if the function is called in parallel with another
 *   function which modifies @a reconciledAttrList. No synchronization is done
 *   in the function. To ensure that no errors oclwrs and that
 *   LwSciCommonPanic() is not called, the user must ensure that @a
 *   reconciledAttrList is not modified during the call to the function.
 * - Access to @a inputUnreconciledAttrListArray array requires thread
 *   synchronization; without the synchronization, the function could access
 *   invalid LwSciSyncAttrList objects, in turn causing a system error or
 *   calling LwSciCommonPanic(). No synchronization is done in the function.
 *   To ensure that no errors oclwrs, and that correct objects are accessed,
 *   the user must ensure that the the values are not modified during the call
 *   to the function.
 * - Access to elements of the @a inputUnreconciledAttrListArray array
 *   requires thread synchronization while the elements are being validated;
 *   without synchronization, the function could cause incorrect result to be
 *   returned when called in parallel with another function which modifies any
 *   of the elements in another thread.
 * - If @a acquireLocks value is true, the mentioned thread synchronization is
 *   done using LwSciObj.objLock mutex object associated with each of the
 *   elements.
 * - If @a acquireLocks value is false, no synchronization is done in the
 *   function. To ensure the correct value is returned by the function, the
 *   user must ensure that none of the elements is modified during the call to
 *   the function.
 * - Locking of elements of the @a inputUnreconciledAttrListArray array
 *   requires thread synchronization; without synchronization, the function
 *   could cause a deadlock when called in parallel from two or more threads.
 *   This synchronization is done by locking the elements in a specific order,
 *   and this order is the same during every call to the function from every
 *   thread.
 * - The operations are not expected to cause nor contribute to a deadlock
 *   since:
 * - when @a acquireLocks value is true, the mentioned LwSciObj.objLock
 *   objects are locked immediately before and released immediately after the
 *   validation, and the locking is done in a specific order, and this order
 *   is the same during every call to the function from every thread,
 * - when @a acquireLocks value is false, there is no locking nor unlocking of
 *   any thread synchronization objects.
 *
 * \implements{18844338}
 */
LwSciError LwSciSyncCoreAttrListValidateReconciledWithLocks(
    LwSciSyncAttrList reconciledAttrList,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool acquireLocks,
    bool* isReconciledListValid);

/**
 * Reconciliation is done as mentioned in the following steps:
 *
 * 1) Creates an appended LwSciSyncAttrList from the list of LwSciSyncAttrLists
 * from inputArray[].
 *
 * 2) Iterates over attribute keys in appended LwSciSyncAttrList and reconciles
 *   as per policy described for each attribute.
 *
 * \implements{18844329}
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommon() would be called.
 * - Access to @a inputArray array requires thread synchronization; without
 *   the synchronization, the function could access invalid LwSciSyncAttrList
 *   objects, in turn causing a system error or calling LwSciCommonPanic(). No
 *   synchronization is done in the function. To ensure that no errors oclwrs,
 *   and that correct objects are accessed, the user must ensure that the the
 *   elements are not modified during the call to the function.
 * - Access to elements of the @a inputArray array requires thread
 *   synchronization while the elements are being reconciled. This
 *   synchronization is done using LwSciObj.objLock mutex object associated
 *   with each of the elements.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned LwSciObj.objLock objects are locked immediately
 *   before and released immediately after the elements are reconciled, and
 *   the locking is done in a specific order, and this order is the same
 *   during every call to the function from every thread.
 *
 * \fn LwSciError LwSciSyncAttrListReconcile(
 *   const LwSciSyncAttrList inputArray[],
 *   size_t inputCount,
 *   LwSciSyncAttrList* newReconciledList,
 *   LwSciSyncAttrList* newConflictList);
 */

/**
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
 * \implements{18844326}
 *
 * \fn LwSciError LwSciSyncAttrListIsReconciled(
 *   LwSciSyncAttrList attrList,
 *   bool* isReconciled);
 */

/**
 * Creates an appended unreconciled LwSciSyncAttrList from the list of input
 * inputUnreconciledAttrListArray[].
 * The compatibility check is performed between the values corresponding to the
 * same keys between the appended unreconciled LwSciSyncAttrList and
 * reconciled LwSciSyncAttrList.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommon() would be called.
 * - Access to @a inputUnreconciledAttrListArray array requires thread
 *   synchronization; without the synchronization, the function could access
 *   invalid LwSciSyncAttrList objects, in turn causing a system error or
 *   calling LwSciCommonPanic(). No synchronization is done in the function.
 *   To ensure that no errors oclwrs, and that correct objects are accessed,
 *   the user must ensure that the the elements are not modified during the
 *   call to the function.
 * - Access to elements of the @a inputUnreconciledAttrListArray array
 *   requires thread synchronization while the elements are being validated;
 *   without synchronization, the function could cause incorrect result to be
 *   returned when called in parallel with another function which modifies any
 *   of the elements in another thread. This synchronization is done using
 *   LwSciObj.objLock mutex object associated with each of the elements.
 * - The operations are not expected to cause nor contribute to a deadlock
 *   since the mentioned LwSciObj.objLock objects are locked immediately
 *   before and released immediately after the validation, and the locking is
 *   done in a specific order, and this order is the same during every call to
 *   the function from every thread.
 *
 * \implements{18844332}
 *
 * \fn LwSciError LwSciSyncAttrListValidateReconciled(
 *   LwSciSyncAttrList reconciledAttrList,
 *   const LwSciSyncAttrList inputUnreconciledAttrListArray[],
 *   size_t inputUnreconciledAttrListCount,
 *   bool* isReconciledListValid);
 */

/**
 * \brief Copies the entries from LwSciSyncCoreSupportedPrimitives array to
 * LwSciSyncInternalAttrKey_SignalerPrimitiveInfo and
 * LwSciSyncInternalAttrKey_WaiterPrimitiveInfo attributes in all
 * LwSciSyncCoreAttrLists contained in input LwSciSyncCoreAttrListObj
 * if needCpuAccess is true and actualPerm/requiredPerm contains Signaler and
 * Waiter permission bit set and those attributes haven't been set.
 * Additionally sets signalerPrimitiveCount to 1 under the same condition, and
 * also updates the valSize accordingly.
 *
 * In case of CPU signaler or waiter, there is no UMD who could provide
 * information on which primitives are available. In that case LwSciSync
 * provides this information in all slots if the user did not.
 *
 * \param[in,out] objAttrList LwSciSyncCoreAttrListObj to get the primitive info
 * \param[in] hasC2C flag indicating C2C case
 *
 * \return void
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Getting valuess of needCpuAccess, signalerPrimitiveInfo and
 *   waiterPrimitiveInfo from LwSciSyncCoreAttrLists objects requires thread
 *   synchronization; without synchronization, the incorrect values could be
 *   obtained when the function is called in parallel with another function
 *   which modifies these values. No synchronization is done in the function.
 *   To ensure the correct values are obtained by the function, the user must
 *   ensure that no other function which can modify the values is called
 *   during the call to LwSciSyncCoreFillCpuPrimitiveInfo().
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844341}
 */
void LwSciSyncCoreFillCpuPrimitiveInfo(
    const LwSciSyncCoreAttrListObj* objAttrList,
    bool hasC2C);

/**
 * \brief Finds the index of the given LwSciSyncInternalAttrValPrimitiveType in
 * the LwSciSyncInternalAttrKey_SignalerPrimitiveInfo array.
 *
 * \param[in] objAttrList LwSciSyncCoreAttrListObj to get the primitive info
 * \param[in] reconciledPrimitive LwSciSyncInternalAttrValPrimitiveType to
 * search for
 * \param[out] primitiveIndex Index within the
 * LwSciSyncInternalAttrValPrimitiveType array provided by the
 * LwSciSyncInternalAttrKey_SignalerPrimitiveInfo key
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if the LwSciSyncInternalAttrValPrimitiveType could
 *   not be found
 * - Panis if any of the following oclwrs:
 *   - coreAttrList is NULL
 *   - primitiveIndex is NULL
 *
 * \implements{}
 */
LwSciError LwSciSyncGetSignalerPrimitiveInfoIndex(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncInternalAttrValPrimitiveType reconciledPrimitive,
    size_t* primitiveIndex);

/**
 * \brief Validates an array of LwSciSyncAttrValTimestampInfo to ensure that
 * each LwSciSyncAttrValTimestampInfo is valid.
 *
 * \param[in] timestampInfo array of LwSciSyncAttrValTimestampInfo to validate
 * \param[in] timestampInfoLen number of LwSciSyncAttrValTimestampInfo in the
 *  array
 *
 * \return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the LwSciSyncAttrValTimestampInfo(s) in
 *   timestampInfo is invalid.
 */
LwSciError LwSciSyncValidateTimestampInfo(
    const LwSciSyncAttrValTimestampInfo* timestampInfo,
    size_t timestampInfoLen);

 /** @} */
#endif
