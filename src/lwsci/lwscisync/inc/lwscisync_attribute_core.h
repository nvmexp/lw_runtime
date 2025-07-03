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
 * \brief <b>LwSciSync core attribute definitions</b>
 *
 * @b Description: This file declares items common to attribute unit cluster
 */

#ifndef INCLUDED_LWSCISYNC_ATTRIBUTE_CORE_H
#define INCLUDED_LWSCISYNC_ATTRIBUTE_CORE_H

/**
 * @defgroup lwsci_sync Synchronization APIs
 *
 * @ingroup lwsci_group_stream
 * @{
 */

#include <stdbool.h>
#include "lwscibuf.h"
#include "lwscicommon_objref.h"
#include "lwscilog.h"
#include "lwscisync_internal.h"

/**
 * \brief Gets LwSciSyncModule from LwSciSyncCoreAttrListObj referenced by the
 * input LwSciSyncAttrList.
 *
 * \param[in] attrList LwSciSyncAttrList from which the LwSciSyncModule should be
 * retrieved
 * \param[out] module The LwSciSyncModule
 * \return void
 * - Panics if @a attrList is invalid or @a module is NULL
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
 * \implements{18844245}
 */
void LwSciSyncCoreAttrListGetModule(
    LwSciSyncAttrList attrList,
    LwSciSyncModule* module);

/**
 * \brief Creates a new LwSciSyncAttrList referencing the same
 * LwSciSyncCoreAttrListObj as the input LwSciSyncAttrList.
 *
 * The resulting *dupAttrList is a new reference that needs to be
 * freed separately but the underlying LwSciSyncCoreAttrListObj is the same
 * one. Attribute changes in one are going to be visible in the other.
 *
 * \param[in] attrList LwSciSyncAttrList to duplicate
 * \param[out] dupAttrList new LwSciSyncAttrList
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if failed to create new reference.
 * - LwSciError_ResourceError if system lacks resource other than memory.
 * - LwSciError_IlwalidState if the number of LwSciSyncAttrList referencing
 *   LwSciSyncCoreAttrListObj are INT32_MAX and this API is called to create
 *   one more LwSciSyncAttrList reference.
 * - Panics if @a attrList is invalid or if @a attrList or @a dupAttrList is NULL.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommon() would be called.
 * - Access to LwSciObj.refCount object associated with @a attrList requires
 *   thread synchronization while the object is being incremented. This
 *   synchronization is done using LwSciObj.objLock mutex object associated
 *   with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned LwSciObj.objLock is locked immediately before and
 *   released immediately after LwSciObj.refCount is incremented.
 *
 * \implements{18844248}
 */
LwSciError LwSciSyncCoreAttrListDup(
    LwSciSyncAttrList attrList,
    LwSciSyncAttrList* dupAttrList);

/**
 * \brief Retrieves the reconciled semaphore LwSciBufAttrList from
 * LwSciSyncAttrList
 *
 * \param[in] syncAttrList list being queried
 * \param[out] semaAttrList semaphore related LwSciSyncAttrList in syncAttrList
 *
 * \return void
 */
void LwSciSyncCoreAttrListGetSemaAttrList(
    LwSciSyncAttrList syncAttrList,
    LwSciBufAttrList* semaAttrList);

/**
 * \brief Retrieves the reconciled timestampBuf LwSciBufAttrList from
 * reconciled LwSciSyncAttrList
 *
 * \param[in] syncAttrList object of type LwSciSyncAttrList
 * \param[out] timestampBufAttrList pointer to LwSciBufAttrList
 *
 * \return void
 */
void LwSciSyncCoreAttrListGetTimestampBufAttrList(
    LwSciSyncAttrList syncAttrList,
    LwSciBufAttrList* timestampBufAttrList);

/**
 * \brief Sets the value of LwSciSyncAttrKey_ActualPerm in the slot 0 of the
 * LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the
 * input LwSciSyncAttrList.
 *
 * \param[in,out] attrList The LwSciSyncAttrList
 * \param[in] actualPerm LwSciSyncAccessPerm to set
 * Valid value: Any of the following: LwSciSyncAccessPerm_WaitOnly,
 * LwSciSyncAccessPerm_SignalOnly, LwSciSyncAccessPerm_WaitSignal
 *
 * \return void
 * - Panics if @a attrList is invalid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Setting the LwSciSyncAttrKey_ActualPerm value requires thread
 *   synchronization; without synchronization, the function could attempt to
 *   set the value while the value is being accessed by another function in
 *   parallel. No synchronization is done in the function. The user must
 *   ensure that objAttrList->coreAttrList->attrs.actualPerm is not accessed
 *   during the call to LwSciSyncCoreAttrListSetActualPerm().
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844251}
 */
void LwSciSyncCoreAttrListSetActualPerm(
    LwSciSyncAttrList attrList,
    LwSciSyncAccessPerm actualPerm);

/**
 * \brief Checks if slot 0 of the LwSciSyncCoreAttrList contained in
 * LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has
 * needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_SignalOnly
 * set.
 *
 * \param[in] attrList The LwSciSyncAttrList
 * \param[out] isCpuSignaler Boolean value to indicate whether the
 * LwSciSyncAttrList is a CPU signaler or not
 * \return void
 * - Panics if @a isCpuSignaler is NULL or @a attrList is invalid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could return incorrect value when called in
 *   parallel with another function which modifies the value of needCpuAccess
 *   or LwSciSyncAccessPerm_SignalOnly attribute. No synchronization is done
 *   in the function. To ensure the correct value is returned, the user must
 *   ensure that values of needCpuAccess and LwSciSyncAccessPerm_SignalOnly
 *   attribute are not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844254}
 */
void LwSciSyncCoreAttrListTypeIsCpuSignaler(
    LwSciSyncAttrList attrList,
    bool* isCpuSignaler);

/**
 * \brief Checks if slot 0 of the LwSciSyncCoreAttrList contained in
 * LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has
 * needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly
 * set.
 *
 * \param[in] attrList The LwSciSyncAttrList
 * \param[out] isCpuWaiter Boolean value to indicate whether the
 * LwSciSyncAttrList is a CPU waiter or not
 * \return void
 * - Panics if @a isCpuWaiter is NULL or @a attrList is invalid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could return incorrect value when called in
 *   parallel with another function which modifies the value of needCpuAccess
 *   or LwSciSyncAccessPerm_WaitOnly attribute. No synchronization is done in
 *   the function. To ensure the correct value is returned, the user must
 *   ensure that values of needCpuAccess and LwSciSyncAccessPerm_WaitOnly
 *   attribute are not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844257}
 */
void LwSciSyncCoreAttrListTypeIsCpuWaiter(
    LwSciSyncAttrList attrList,
    bool* isCpuWaiter);

/**
 * \brief Checks if slot 0 of the LwSciSyncCoreAttrList contained in
 * LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has
 * LwSciSyncHwEngName_PCIe included in EngineArray
 * and actualPerm has LwSciSyncAccessPerm_SignalOnly set.
 *
 * Uses LwSciSyncHwEngGetNameFromId() for engine callwlations.
 *
 * \param[in] attrList The LwSciSyncAttrList
 * \param[out] isC2cSignaler Boolean value to indicate whether the
 * LwSciSyncAttrList is a C2c signaler or not
 * \return void
 * - Panics if @a isC2cSignaler is NULL or @a attrList is invalid
 *
 * \implements{18844254}
 */
void LwSciSyncCoreAttrListTypeIsC2cSignaler(
    LwSciSyncAttrList attrList,
    bool* isC2cSignaler);

/**
 * \brief Validates the input LwSciSyncAttrList and its associated
 * LwSciSyncCoreAttrListObj.
 *
 * \param[in] attrList LwSciSyncAttrList to validate
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a attrList is NULL
 * - Panics if LwSciSyncCoreAttrListObj referenced by the @a attrList is invalid.
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
 * \implements{18844281}
 *
 */
LwSciError LwSciSyncCoreAttrListValidate(
    LwSciSyncAttrList attrList);

/**
 * \brief Gets the value of signalerUseExternalPrimitive member from the
 * slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj
 * referenced by the input LwSciSyncAttrList.
 *
 * \param[in] attrList The LwSciSyncAttrList to get value from
 * \param[out] signalerUseExternalPrimitive pointer where bool value is written
 * \return void
 * - Panics if any of the following oclwrs:
 *   - @a attrList is NULL
 *   - @a attrList is not a valid LwSciSyncAttrList
 *   - @a signalerUseExternalPrimitive is NULL
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
 * \implements{18844287}
 */
 void LwSciSyncCoreGetSignalerUseExternalPrimitive(
    LwSciSyncAttrList attrList,
    bool* signalerUseExternalPrimitive);

/**
 * \brief Gets the value of lastExport member from the
 * slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj
 * referenced by the input LwSciSyncAttrList.
 *
 * \param[in] attrList The LwSciSyncAttrList to get value from
 * \param[out] ipcEndpoint pointer where lastExport value is written
 * \return void
 * - Panics if any of the following oclwrs:
 *   - @a attrList is NULL
 *   - @a attrList is not a valid LwSciSyncAttrList
 *   - @a ipcEndpoint is NULL
 *
 * \implements{TBD}
 */
 void LwSciSyncCoreGetLastExport(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint* ipcEndpoint);

/**
 * \brief Sanity check for input LwSciSyncInternalAttrValPrimitiveType values
 *
 * \param[in] primitiveInfo array of LwSciSyncInternalAttrValPrimitiveType
 * Valid value: Entries in the array having value in
 * (LwSciSyncInternalAttrValPrimitiveType_LowerBound,
 * LwSciSyncInternalAttrValPrimitiveType_UpperBound).
 * \param[in] size size of @a primitiveInfo array
 * Valid value: [0, SIZE_MAX]
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if invalid primitive info
 * - Panics if @a primitiveInfo is NULL.
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
 * \implements{18844284}
 */
static inline LwSciError LwSciSyncCoreCheckPrimitiveValues(
    const LwSciSyncInternalAttrValPrimitiveType* primitiveInfo,
    size_t size)
{
    LwSciError error = LwSciError_Success;
    size_t i;

    if (primitiveInfo == NULL) {
        LWSCI_ERR_STR("primitiveInfo is NULL\n");
        LwSciCommonPanic();
    }

    for (i = 0U; i < size; i++) {
        if ((primitiveInfo[i] <=
                LwSciSyncInternalAttrValPrimitiveType_LowerBound) ||
                (primitiveInfo[i] >=
                LwSciSyncInternalAttrValPrimitiveType_UpperBound)) {
            LWSCI_ERR_INT("Invalid value for PrimitiveInfo: \n",
                    primitiveInfo[i]);
            error = LwSciError_BadParameter;
            break;
        }
    }
    return error;
}

/**
 * \brief Attempts to acquire locks for all the LwSciSyncAttrLists in an
 * all-or-nothing fashion.
 *
 * \param[in] inputAttrListArr Array of LwSciSyncAttrLists
 * Valid value: Array of valid LwSciSyncAttrLists
 * \param[in] attrListCount number of LwSciSyncAttrLists in @a inputAttrListArr
 * Valid value: [0, SIZE_MAX]
 *
 * \return LwSciError
 * - LwSciError_BadParameter if @a inputAttrListArr is NULL and
 *     @a attrListCount indicates that there were LwSciSyncAttrList
 * - LwSciError_Overflow if the required space to allocate a temporary
 *     array exceeds SIZE_MAX
 * - LwSciError_InsufficientMemory if not enough memory to allocate a
 *     temporary array
 * - Panics if:
 *   - any of the LwSciSyncAttrLists is invalid
 *   - unable to sort the temporary array
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to the @a inputAttrListArr array requires thread synchronization;
 *   without the synchronization, the function could access invalid
 *   LwSciSyncAttrList objects, in turn causing a system error or calling
 *   LwSciCommonPanic(). No synchronization is done in the function. To ensure
 *   that no errors oclwrs, and that correct objects are accessed, the user
 *   must ensure that the the elements are not modified during the call to the
 *   function.
 * - Locking of elements of the @a inputAttrListArr array requires thread
 *   synchronization; without synchronization, the function could cause a
 *   deadlock when called in parallel from two or more threads. This
 *   synchronization is done by locking the elements in a specific order, and
 *   this order is the same during every call to the function from every
 *   thread.
 * - The operations are not expected to cause nor contribute to a deadlock due
 *   to the mentioned locking of the elements in a specific order, and this
 *   order is the same during every call to the function from every thread.
 *
 * \implements{18844296}
 */
LwSciError LwSciSyncCoreAttrListsLock(
    const LwSciSyncAttrList inputAttrListArr[],
    size_t attrListCount);

/**
 * \brief Release locks on all the LwSciSyncAttrLists held by the caller.
 *
 * Note: This function assumes that all LwSciSyncAttrLists were successfully
 * locked.
 *
 * \param[in] inputAttrListArr list of LwSciSyncAttrLists to release locks on.
 * Valid value: Array of valid LwSciSyncAttrLists
 * \param[in] attrListCount number of LwSciSyncAttrLists in @a inputAttrListArr.
 * Valid value: [0, SIZE_MAX]
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a inputAttrListAttr is NULL and
 *     @a attrListCount indicates that there were LwSciSyncAttrList
 * - Panics if:
 *   - any of the LwSciSyncAttrLists is invalid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to the @a inputAttrListArr array requires thread synchronization;
 *   without the synchronization, the function could access invalid
 *   LwSciSyncAttrList objects, in turn causing a system error or calling
 *   LwSciCommonPanic(). No synchronization is done in the function. To ensure
 *   that no errors oclwrs, and that correct objects are accessed, the user
 *   mush ensure that the the elements are not modified during the call to the
 *   function.
 * - The operations can cause a call to LwSciCommonPanic() when any of the
 *   elements of the @a inputAttrListArr array have been locked by a call to
 *   LwSciSyncCoreAttrListsLock() from a thread different than the thread from
 *   which LwSciSyncCoreAttrListsUnlock() is called.
 *
 * \implements{18844299}
 */
LwSciError LwSciSyncCoreAttrListsUnlock(
    const LwSciSyncAttrList inputAttrListArr[],
    size_t attrListCount);

/**
 * \brief Appends unreconciled LwSciSyncAttrList(s), allowing the caller to
 * control whether acquiring locks on the LwSciSyncAttrList(s) is necessary.
 * Creates a new unreconciled LwSciSyncAttrList using
 * LwSciSyncAttrListCreateMultiSlot() with the summed up slot
 * count and copies the input unreconciled LwSciSyncAttrList(s)
 * to the new unreconciled LwSciSyncAttrList slot by slot.
 *
 * \param[in] inputUnreconciledAttrListArray an array of unreconciled
 * LwSciSyncAttrLists.
 * Valid value: Array of valid LwSciSyncAttrLists where the array size is at least 1
 * \param[in] inputUnreconciledAttrListCount the number of unreconciled
 * LwSciSyncAttrLists in @a inputUnreconciledAttrListArray.
 * Valid value: inputUnreconciledAttrListCount is valid input if is non-zero.
 * \param[in] acquireLocks Whether locks on all the LwSciSyncAttrLists are
 * acquired. If this is set to false, then it is the caller's responsibility to
 * manage the locking.
 * \param[out] newUnreconciledAttrList the appended LwSciSyncAttrList.
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *         - @a inputUnreconciledAttrListArray is NULL
 *         - @a inputUnreconciledAttrListCount is 0
 *         - @a newUnreconciledAttrList is NULL
 *         - any of the input LwSciSyncAttrLists are not unreconciled
 *         - not all the LwSciSyncAttrLists in @a inputUnreconciledAttrListArray
 *           are bound to the same LwSciSyncModule instance.
 * - LwSciError_InsufficientMemory if there is insufficient system memory to
 *   create the new unreconciled LwSciSyncAttrList.
 * - LwSciError_Overflow if the combined slot counts of all the input
 *   LwSciSyncAttrLists exceeds UINT64_MAX
 * - LwSciError_IlwalidState if no more references can be taken for
 *   LwSciSyncModule associated with the LwSciSyncAttrList in @a
 *   inputUnreconciledAttrListArray to create the new LwSciSyncAttrList.
 * - Panics if any of the input LwSciSyncAttrLists are not valid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to the @a inputUnreconciledAttrListArray array requires thread
 *   synchronization; without the synchronization, the function could access
 *   invalid LwSciSyncAttrList objects, in turn causing a system error or
 *   calling LwSciCommonPanic(). No synchronization is done in the function.
 *   To ensure that no errors oclwrs, and that correct objects are accessed,
 *   the user mush ensure that the the elements are not modified during the
 *   call to the function.
 * - Access to elements of the @a inputUnreconciledAttrListArray array
 *   requires thread synchronization; without the synchronization, the
 *   function could read incorrect or invalid attribute values from the
 *   elements, and consequently set those attribute values in the @a
 *   newUnreconciledAttrList.
 * - If @a acquireLocks value is true, the mentioned thread synchronization is
 *   done using LwSciObj.objLock mutex associated with each of the elements.
 * - If @a acquireLocks value is false, no synchronization is done in the
 *   function. To ensure correct values are set in @a newUnreconciledAttrList,
 *   the user must ensure that none of the @a inputUnreconciledAttrListArray
 *   elements are modified during the call to the function.
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
 *   validation,
 * - when @a acquireLocks value is false, there is no locking nor unlocking of
 *   any thread synchronization objects.
 *
 * \implements{18844302}
 */
LwSciError LwSciSyncCoreAttrListAppendUnreconciledWithLocks(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool acquireLocks,
    LwSciSyncAttrList* newUnreconciledAttrList);

/**
 * \brief Validates the given LwSciSyncAttrLists
 *
 * This function checks all of the given LwSciSyncAttrLists to ensure that they
 * are valid.
 *
 * \param[in] attrListArray Array of LwSciSyncAttrList to be checked.
 * Valid value: Array of valid LwSciSyncAttrLists
 * \param[in] attrListCount Number of entries/elements in attrListArray.
 * Valid value: [0, SIZE_MAX] but also >1 if allowEmpty is false.
 * \param[in] allowEmpty whether to consider empty arrays valid or not.
 *
 * \return LwSciError
 * - LwSciError_Success if all the given LwSciSyncAttrLists are valid
 * - LwSciError_BadParameter if:
 *      - @a attrListArray is NULL and @a allowEmpty is false
 *      - @a attrListCount is 0 and @a allowEmpty is false
 *      - @a attrListCount is 0 but @a attrListArray is not NULL
 *      - one or more of the given LwSciSyncAttrLists is NULL
 * - Panics if one or more LwSciSyncAttrLists are invalid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the @a attrListArray array requires thread synchronization;
 *   without the synchronization, the function could access invalid
 *   LwSciSyncAttrList objects, in turn causing a system error or calling
 *   LwSciCommonPanic(). No synchronization is done in the function. To ensure
 *   that no errors oclwrs, and that correct objects are accessed, the user
 *   mush ensure that the the elements are not modified during the call to the
 *   function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844293}
 */
LwSciError LwSciSyncCoreValidateAttrListArray(
    const LwSciSyncAttrList attrListArray[],
    size_t attrListCount,
    bool allowEmpty);

/**
 * \brief Callback to free the data associated with the LwSciObj representing
 * the underlying LwSciSyncCoreAttrListObj using LwSciCommon functionality.
 * This function assumes that objPtr is non-NULL.
 *
 * \param[in] objPtr Pointer to the LwSciObj associated with the
 * LwSciSyncAttrList to free
 *
 * \return void
 * - Panics if objPtr is invalid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Closing the module asscociated with @a ObjPtr requires thread
 *   synchronization; without synchronization, the function could cause a call
 *   to LwSciCommonPanic(). No synchronization is done in the function. To
 *   ensure that LwSciCommonPanic() is not called, the user must ensure that
 *   the module value is not modified during the call to the function.
 * - Releasing @a ObjPtr requires thread synchronization; without
 *   synchronization, the function could cause a call to LwSciCommonPanic().
 *   No synchronization is done in the function. To ensure that LwSciCommon is
 *   not called, the user must ensure that the object values is not modified
 *   during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{22034745}
 */
void LwSciSyncCoreAttrListFree(
    LwSciObj* objPtr);

/**
 * \brief Callback used to compare two LwSciSyncAttrList handles to determine
 * an ordering when sorting. This function assumes that elem1 and elem2 are
 * non-NULL.
 *
 * \param[in] elem1 The first LwSciSyncAttrList handle to compare
 * \param[in] elem2 The second LwSciSyncAttrList handle to compare
 *
 * \return int32_t
 *  - 1 if the LwSciSyncAttrList handle represented by elem2 should be placed
 *    before elem1 in the sorted list
 *  - 0 if the LwSciSyncAttrList handles are identical
 *  - -1 otherwise
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
 * \implements{22034754}
 */
int32_t LwSciSyncAttrListCompare(
    const void* elem1,
    const void* elem2);

/**
 * Allocates an LwSciSyncAttrListRec and its associated
 * LwSciSyncCoreAttrListObj using LwSciCommon functionality.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - Duplication of the @a module, needed for creation of an attribute list,
 *   requires thread synchronization; without synchronization, the function
 *   could cause a call to LwSciCommonPanic(). No synchronization is done in
 *   the function. To ensure that LwSciCommonPanic() is not called, the user
 *   must ensure that the module value is not modified during the call to the
 *   function.
 * - Incrementing the reference count of the @a module also requires thread
 *   synchronization. This synchronization is done by locking LwSciObj.ObjLock
 *   mutex object associated with @a module.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute values are set.
 *
 * \implements{18844206}
 *
 * \fn LwSciError LwSciSyncAttrListCreate(
 *   LwSciSyncModule module,
 *   LwSciSyncAttrList* attrList);
 */

/**
 * Deallocates LwSciSyncAttrListRec using LwSciCommon functionality.
 * LwSciSyncCoreAttrListObj is deallocated using LwSciCommon functionality
 * when all the LwSciSyncAttrListRec references are destroyed.
 *
 * Conlwrrency:
 * - Thread-safe: Yes
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could cause a system error or a call to the
 *   LwSciCommonPanic(). No synchronization is done in the function. To ensure
 *   that there is no system error and no call to the LwSciCommonPani(), the
 *   user must ensure that @a attrList value is not modified during the call
 *   to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844209}
 *
 * \fn void LwSciSyncAttrListFree(
 *   LwSciSyncAttrList attrList);
 */

/**
 * Value(s) for LwSciSyncAttrKey(s) are set in slot 0 of
 * LwSciSyncCoreAttrList contained in the LwSciSyncCoreAttrListObj referenced
 * by the input LwSciSyncAttrList, to the provided value(s).
 * Additionally sets the valSize to LwSciSyncAttrKeyValuePair's len and keyState
 * to LwSciSyncCoreAttrKeyState_SetLocked.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Setting attribute values in @a attrList requires thread synchronization;
 *   without synchronization, the function could cause invalid values to be
 *   set when called in parallel from two or more threads and passed the same
 *   value for @a attrList. This synchronization is done using
 *   LwSciObj.objLock mutex object associated with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute values are set.
 *
 * \implements{18844212}
 *
 * \fn LwSciError LwSciSyncAttrListSetAttrs(
 *   LwSciSyncAttrList attrList,
 *   const LwSciSyncAttrKeyValuePair* pairArray,
 *   size_t pairCount);
 */

/**
 * Value(s) of requested LwSciSyncAttrKey(s) are fetched from slot 0 of
 * LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by
 * the input LwSciSyncAttrList.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Getting attribute values from @a attrList requires thread
 *   synchronization; without synchronization, the function could cause
 *   incorrect values to be returned when called in parallel with the another
 *   function which modifies @a attrList in another thread. This
 *   synchronization is done using LwSciObj.objLock mutex object associated
 *   with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute values are obtained from @a
 *   attrList.
 *
 * \implements{18844215}
 *
 * \fn LwSciError LwSciSyncAttrListGetAttrs(
 *   LwSciSyncAttrList attrList,
 *   LwSciSyncAttrKeyValuePair* pairArray,
 *   size_t pairCount);
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
 * \implements{18844218}
 *
 * \fn size_t LwSciSyncAttrListGetSlotCount(
 *    LwSciSyncAttrList attrList);
 */

/**
 * This function internally calls LwSciSyncCoreAttrListAppendUnreconciledWithLocks()
 * to create a new appended unreconciled LwSciSyncAttrList with acquiring locks
 * on the LwSciSyncAttrLists during entire operation.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to the @a inputUnreconciledAttrListArray array requires thread
 *   synchronization; without the synchronization, the function could access
 *   invalid LwSciSyncAttrList objects, in turn causing a system error or
 *   calling LwSciCommonPanic(). No synchronization is done in the function.
 *   To ensure that no errors oclwrs, and that correct objects are accessed,
 *   the user must ensure that the the elements are not modified during the
 *   call to the function.
 * - Locking of elements of the @a inputUnreconciledAttrListArray array
 *   requires thread synchronization; without synchronization, the function
 *   could cause a deadlock when called in parallel from two or more threads.
 *   This synchronization is done by locking the elements in a specific order,
 *   and this order is the same during every call to the function from every
 *   thread.
 * - The operations are not expected to cause nor contribute to a deadlock due
 *   to the mentioned locking of the elements in a specific order, and this
 *   order is the same during every call to the function from every thread.
 *
 * \implements{18844221}
 *
 * \fn LwSciError LwSciSyncAttrListAppendUnreconciled(
 *   const LwSciSyncAttrList inputUnreconciledAttrListArray[],
 *   size_t inputUnreconciledAttrListCount,
 *   LwSciSyncAttrList* newUnreconciledAttrList);
 */

/**
 * Creates a new LwSciSyncAttrList using LwSciSyncCoreAttrListCreateMultiSlot()
 * with equal number of slots as input LwSciSyncAttrList. Then copies all the
 * data contained in the input LwSciSyncAttrList. If the LwSciSyncAttrList being
 * cloned is unreconciled, then updates the keyState in cloned LwSciSyncAttrList
 * of already set keys to LwSciSyncCoreAttrKeyState_SetUnlocked.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Copying of data contained in @a origAttrList to @a newAttrList requires
 *   thread synchronization; without synchronization, the function could cause
 *   invalid or incorrect attribute values to be copied when the function is
 *   called in parallel with another function which modifies @a origAttrList
 *   in another thread. This synchronization is done using LwSciObj.objLock
 *   mutex object associated with @a origAttrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the data is copied.
 *
 * \implements{18844224}
 *
 * \fn LwSciError LwSciSyncAttrListClone(
 *   LwSciSyncAttrList origAttrList,
 *   LwSciSyncAttrList* newAttrList);
 */

/**
 * The value(s) of requested LwSciSyncAttrKey(s) keys are fetched
 * from the LwSciSyncCoreAttrList for slotIndex which is retrieved from
 * LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Getting attribute values from @a attrList requires thread
 *   synchronization; without synchronization, the function could cause
 *   incorrect values to be returned when called in parallel with the another
 *   function which modifies @a attrList in another thread. This
 *   synchronization is done using LwSciObj.objLock mutex object associated
 *   with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute values are obtained from @a
 *   attrList.
 *
 * \implements{18844227}
 *
 * \fn LwSciError LwSciSyncAttrListSlotGetAttrs(
 *   LwSciSyncAttrList attrList,
 *   size_t slotIndex,
 *   LwSciSyncAttrKeyValuePair* pairArray,
 *   size_t pairCount);
 */

/**
 * The value(s) of LwSciSyncInternalAttrKey(s) are set in slot 0 of
 * LwSciSyncCoreAttrList retrieved from LwSciSyncCoreAttrListObj referenced by
 * the input LwSciSyncAttrList.
 * Additionally sets the valSize to attribute length and keyState to locked.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Setting attribute values in @a attrList requires thread synchronization;
 *   without synchronization, the function could cause invalid values to be
 *   set when called in parallel from two or more threads and passed the same
 *   value for @a attrList. This synchronization is done using
 *   LwSciObj.objLock mutex object associated with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute values are set.
 *
 * \implements{18844233}
 *
 * \fn LwSciError LwSciSyncAttrListSetInternalAttrs(
 *   LwSciSyncAttrList attrList,
 *   const LwSciSyncInternalAttrKeyValuePair* pairArray,
 *   size_t pairCount);
 */

/**
 * The values of LwSciSyncInternalAttrKey(s) are fetched from
 * the LwSciSyncCoreAttrList for slot index 0 which is retrieved from
 * LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Getting attribute values from @a attrList requires thread
 *   synchronization; without synchronization, the function could cause
 *   incorrect values to be returned when called in parallel with the another
 *   function which modifies @a attrList in another thread. This
 *   synchronization is done using LwSciObj.objLock mutex object associated
 *   with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute values are obtained from @a
 *   attrList.
 *
 * \implements{18844236}
 *
 * \fn LwSciError LwSciSyncAttrListGetInternalAttrs(
 *   LwSciSyncAttrList attrList,
 *   LwSciSyncInternalAttrKeyValuePair* pairArray,
 *   size_t pairCount);
 */

/**
 * The value of requested single LwSciSyncInternalAttrKey is
 * fetched from the LwSciSyncCoreAttrList for slot index 0 which is retrieved from
 * LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Getting attribute value from @a attrList requires thread synchronization;
 *   without synchronization, the function could cause incorrect values to be
 *   returned when called in parallel with the another function which modifies
 *   @a attrList in another thread. This synchronization is done using
 *   LwSciObj.objLock mutex object associated with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute value is obtained from @a
 *   attrList.
 *
 * \implements{18844239}
 *
 * \fn LwSciError LwSciSyncAttrListGetSingleInternalAttr(
 *   LwSciSyncAttrList attrList,
 *   LwSciSyncInternalAttrKey key,
 *   const void** value,
 *   size_t* len);
 */

/**
 * The value of requested LwSciSyncAttrKey(s) are fetched from the
 * LwSciSyncCoreAttrList for slot index 0 which is retrieved from
 * LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Getting attribute value from @a attrList requires thread synchronization;
 *   without synchronization, the function could cause incorrect values to be
 *   returned when called in parallel with the another function which modifies
 *   @a attrList in another thread. This synchronization is done using
 *   LwSciObj.objLock mutex object associated with @a attrList.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute value is obtained from @a
 *   attrList.
 *
 * \implements{18844230}
 *
 * \fn LwSciError LwSciSyncAttrListGetAttr(
 *   LwSciSyncAttrList attrList,
 *   LwSciSyncAttrKey key,
 *   const void** value,
 *   size_t* len);
 */

 /** @} */
#endif
