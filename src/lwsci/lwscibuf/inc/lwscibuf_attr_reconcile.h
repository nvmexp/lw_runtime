/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_RECONCILE_H
#define INCLUDED_LWSCIBUF_ATTR_RECONCILE_H

#include "lwscibuf_attr_constraint.h"

/**
 * @defgroup lwscibuf_attr_list_api LwSciBuf Attribute List APIs
 * Methods to perform operations on LwSciBuf attribute lists.
 * @{
 */

/**
 * Reconciliation is done as mentioned in the following steps:
 *
 * 1) Creates an appended LwSciBufAttrList from the list of inputArray[]
 *   LwSciBufAttrList(s).
 *
 * 2) Iterate for each of the key in appended LwSciBufAttrList and obtain the
 *   reconciliation policy to be used for that particular attribute key from
 *   LwSciBufAttrKeyGetPolicy().
 *
 * 3) The values corresponding to attribute keys in the appended
 *   LwSciBufAttrList are merged according to a reconciliation policy.
 *
 * 4) The dependencies are validated for the attribute keys.
 *
 * 5) Hardware buffer constraints are applied using LwSciBufApplyConstrain().
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on each LwSciBufAttrList in inputArray to serialize
 *        reads using LwSciBufAttrListsLock()
 *      - Locks are held for the duration of any reads from all of the
 *        LwSciBufAttrList(s) in inputArray
 *      - Locks are released when all operations on the LwSciBufAttrList(s) in
 *        inputArray are complete
 *      - Conlwrrent access to the LwSciBufModule associated with the
 *        LwSciBufAttrList(s) is provided via LwSciBufAttrListCreateMultiSlot()
 *
 * \implements{17827251}
 *
 * \fn LwSciError LwSciBufAttrListReconcile(
 *    const LwSciBufAttrList inputArray[],
 *    size_t inputCount,
 *    LwSciBufAttrList* newReconciledAttrList,
 *    LwSciBufAttrList* newConflictList);
 */


/**
 * Creates an appended unreconciled LwSciBufAttrList from the list of input
 * LwSciBufAttrList(s) in unreconciledAttrListArray[].
 *
 * The compatibility check is performed for the values corresponding to the
 * same attribute keys between the appended unreconciled LwSciBufAttrList and
 * reconciledAttrList, according to a reconciliation policy.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufAttrList(s) in
 *        unreconciledAttrListArray is provided via
 *        LwSciBufAttrListAppendUnreconciled()
 *      - Conlwrrent access to the LwSciBufModule associated with the
 *        LwSciBufAttrList(s) in unreconciledAttrListArray is provided via
 *        LwSciBufAttrListAppendUnreconciled()
 *
 * \implements{18843138}
 *
 * \fn LwSciError LwSciBufAttrListValidateReconciled(
 *    LwSciBufAttrList reconciledAttrList,
 *    const LwSciBufAttrList unreconciledAttrListArray[],
 *    size_t unreconciledAttrListCount, bool* isReconcileListValid);
 */

/**
 * @brief Reconciles the given unreconciled LwSciBufAttrList(s) into a new
 * reconciled LwSciBufAttrList if the unreconciled LwSciBufAttrList(s) are
 * supplied as input and @a ignoreUnreconciledLists is false otherwise
 * reconciles the output attributes in the provided reconciled LwSciBufAttrList
 * (@a newAttrList) which are dependent on other output attributes.
 * In the latter case, it is expected that the output attributes which are
 * dependent on input attributes be present in the input reconciled
 * LwSciBufAttrList for reconciliation to succeed.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on each
 *        LwSciBufAttrList in inputArray prior to calling this function
 *      - Conlwrrent access must be serialized on the LwSciBufAttrList
 *        newAttrList prior to calling this function
 *      - Conlwrrent access to the LwSciBufModule associated with the
 *        LwSciBufAttrList(s) is provided via LwSciBufAttrListCreateMultiSlot()
 *
 * @param[in] inputArray Array containing unreconciled LwSciBufAttrList(s) to be
 *            reconciled. @a inputArray is valid if @a ignoreUnreconciledLists
 *            is false and @a inputArray is non-NULL where every member of an
 *            array is a valid unreconciled LwSciBufAttrList.
 *            If @a ignoreUnreconciledLists is true then @a inputArray is
 *            ignored.
 * @param[in] inputCount The number of unreconciled LwSciBufAttrList(s) in
 *            @a inputArray. @a inputCount is valid if
 *            @a ignoreUnreconciledLists is false and it is non-zero. If
 *            @a ignoreUnreconciledLists is true then @a inputCount is ignored.
 * @param[in,out] newAttrList Reconciled LwSciBufAttrList. This field
 *             is populated only if the reconciliation succeeded.
 *             valid @a newAttrList.
 */
#if (LW_IS_SAFETY == 0)
/**
 * @param[out] newConflictList Unreconciled LwSciBufAttrList consisting of the
 * key/value pairs which caused the reconciliation failure. This field is
 * populated only if the reconciliation failed.
 */
#else
/**
 * @param[out] newConflictList unused.
 */
#endif
/**
 * @param[in] ignoreUnreconciledLists A boolean value indicating if unreconciled
 *            LwSciBufAttrList(s) passed via @a inputArray should be ignored
 *            during reconciliation.
 *
 * @return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a inputArray[] is NULL provided @a ignoreUnreconciledLists is false.
 *      - @a inputCount is 0 provided @a ignoreUnreconciledLists is false.
 *      - Any of the LwSciBufAttrLists in @a inputArray is reconciled provided
 *        @a ignoreUnreconciledLists is false.
 *      - Not all the LwSciBufAttrLists in @a inputArray are bound to the
 *        same LwSciBufModule provided @a ignoreUnreconciledLists is false.
 *      - @a newAttrList is NULL.
 *      - An attribute key necessary for reconciling against the given data
 *        type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation is
 *        unset provided @a ignoreUnreconciledLists is false.
 *      - An attribute key is set to an unsupported value considering the data
 *        type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation.
 */
#if (LW_IS_SAFETY == 0)
/**      - @a newConflictList is NULL
 */
#endif
/**
 * - ::LwSciError_InsufficientMemory if not enough system memory provided
 *   @a ignoreUnreconciledLists is false.
 * - ::LwSciError_IlwalidState if an intermediate appended LwSciBufAttrList
 *   cannot be associated with the LwSciBufModule associated with the
 *   LwSciBufAttrList(s) in the given @a inputArray provided
 *   @a ignoreUnreconciledLists is false.
 * - ::LwSciError_NotSupported if an attribute key is set resulting in a
 *   combination of given constraints that are not supported.
 * - ::LwSciError_Overflow if internal integer overflow is detected.
 * - ::LwSciError_ReconciliationFailed if reconciliation failed.
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 *   provided @a ignoreUnreconciledLists is false.
 * - Panic if:
 *      - @a unreconciled LwSciBufAttrList(s) in @a inputArray is not valid
 *        provided @a ignoreUnreconciledLists is false.
 *      - @a newAttrList is invalid.
 *
 * @implements{22034348}
 */
LwSciError LwSciBufAttrListReconcileInternal(
    const LwSciBufAttrList inputArray[],
    size_t inputCount,
    LwSciBufAttrList newAttrList,
    LwSciBufAttrList* newConflictList,
    bool ignoreUnreconciledLists);

/*
 * @brief Updates LwSciBufAttrList based on input LwSciIpcEndpoint.
 * If input @a LwSciIpcEndpoint is a inter SoC endpoint, then LwSciBufHwEngName_PCIe
 * engine will be appended to input Unreconciled LwSciBufAttrList.
 *
 * @param[in,out] attrList: LwSciBufAttrList whose values needs to be updated based on
 * input LwSciIpcEndpoint before exporting it.
 * @param[in] ipcEndPoint: LwSciIpcEndpoint.
 * Valid value: 0 is a valid value. If it is not 0, it should have been obtained
 * from a successful call to LwSciIpcOpenEndpoint() and has not yet been freed
 * using LwSciIpcCloseEndpoint().
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL.
 *      - @a ipcEndPoint is not valid
 * - LwSciError_InsufficientResource if any of the following oclwrs:
 *      - the API is unable to implicitly append an additional
 *        LwSciBufHwEngName_PCIe hardware engine
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid.
 *
 * @implements{}
 */
LwSciError LwSciBufAttrListUpdateBeforeExport(
    LwSciBufAttrList attrList,
    const LwSciIpcEndpoint ipcEndPoint);

/**
 * @brief Reconciles @a key by retrieving unreconciled values for the @a key
 * from LwSciBufIpcTable* in @a attrList and by applying LwSciBuf_ReconcilePolicy
 * to the unreconciled values. The reconciled value of the @a key is set in
 * the @a attrList.
 *
 * @param[in] attrList: reconciled LwSciBufAttrList from which unreconciled
 * values of the @a key present in the LwSciBufIpcTable* need to be fetched.
 * @param[in] key: attribute key.
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey, or LwSciBufPrivateAttrKey enums.
 * @param[in] ipcEndPoint: LwSciIpcEndpoint.
 * Valid value: 0 is a valid value. If it is not 0, it should have been obtained
 * from a successful call to LwSciIpcOpenEndpoint() and has not yet been freed
 * using LwSciIpcCloseEndpoint().
 * @param[in] localPeer boolean value indicating if the reconciled value for the
 * @a key is callwlated for local peer or remote peer.
 * @param[in] overrideKeyAffinity boolean value indicating if the
 * LwSciBufIpcRouteAffinity should be considered from the @a key or from the
 * @a routeAffinity.
 * @param[in] routeAffinity Affinity using which the values of the @a key need
 * to be found in the LwSciBufIpcTable*. If LwSciBufIpcRoute_AffinityNone is
 * passed then the default value associated with @a key is considered otherwise
 * @a routeAffinity value is considered.
 * Valid value: LwSciBufIpcRoute_AffinityNone <= routeAffinity <
 *               LwSciBufIpcRoute_Max if overrideKeyAffinity is true, ignored
 *              otherwise.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL.
 *      - @a attrList is not reconciled.
 *      - @a key is not a valid enumeration value defined by the LwSciBufAttrKey,
 *        LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums, or is not
 *        a UMD-specific LwSciBufInternalAttrKey
 *      - @a ipcEndPoint is not valid provided the input @a ipcEndPoint is
 *        non-zero.
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid.
 *      - @a routeAffinity is invalid.
 *
 * @implements{}
 */
LwSciError LwSciBufAttrListReconcileFromIpcTable(
    LwSciBufAttrList attrList,
    uint32_t key,
    LwSciIpcEndpoint ipcEndPoint,
    bool localPeer,
    bool overrideKeyAffinity,
    LwSciBufIpcRouteAffinity routeAffinity);

/**
 * @}
 */

#endif /* INCLUDED_LWSCIBUF_ATTR_RECONCILE_H */
