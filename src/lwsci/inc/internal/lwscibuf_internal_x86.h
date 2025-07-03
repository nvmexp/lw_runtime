/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_INTERNAL_X86_H
#define INCLUDED_LWSCIBUF_INTERNAL_X86_H

#include "stdint.h"

#include "lwscibuf.h"
#include "lwtypes.h"

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @defgroup lwscibuf_obj_datatype_int LwSciBuf object internal datatypes
 * List of all LwSciBuf object internal data types
 * @{
 */

/**
 * @brief structures holding hClient hDevice and hMem handle
 */
typedef struct {
    LwHandle hClient;
    LwHandle hDevice;
    LwHandle hMemory;
} LwSciBufRmHandle;

/**
 * @}
 */

/* This is forward declaration of LwSciBufInternalAttrKeyValuePairRec defined
 * in lwscibuf_internal.h
 */
struct LwSciBufInternalAttrKeyValuePairRec;

/**
 * @brief Structure providing
 * LwSciBufAttrKeyValuePair/LwSciBufInternalAttrKeyValuePair
 * for output only attributes to be set in the reconciled LwSciBufAttrList.
 */
typedef struct {
    /** Pointer to array of output only public attribute (Key,Value) pairs.*/
    LwSciBufAttrKeyValuePair*  publicAttrs;
    /** Number of entries in publicAttrs Array.*/
    size_t publicAttrsCount;
    /** Pointer to array of output only internal attribute (key, value) pairs.*/
    struct LwSciBufInternalAttrKeyValuePairRec* internalAttrs;
    /** Number of entries in internalAttrs Array.*/
    size_t internalAttrsCount;
} LwSciBufOutputAttrKeyValuePair;

/**
 * @brief This function has to be used as an alternative for
 * LwSciBufAttrListReconcile() function for cases where the buffer is already
 * allocated and user provided output attributes have to be used instead of
 * computing the attributes internally. The API will succeed only if all the
 * output-only attributes of the datatype(s) used in the provided unreconciled
 * attribute lists are given through the outputAttributes input param. The API
 * will check if the provided size & alignment are consistent with the other
 * provided output parameters. If the check fails, the API returns failure.
 *
 * @param[in] inputUnreconciledAttrLists one or more unreconciled
 * LwSciBufAttrList(s) that need to be reconciled.
 * @param[in] attrListCount number of unreconciled attribute lists in the
 * inputUnreconciledAttrLists array.
 * @param[in] outputAttributes LwSciBufOutputAttrKeyValuePair specifying output
 * only attributes to be set in reconciled LwSciBufAttrList.
 * @param[out] outputReconciledAttrList output reconciled attribute list
 * containing values from @a outputAttributes. Value will be NULL if
 * reconciliation fails.
 * @param[out] conflictAttrList conflict attribute list that will have a
 * valid pointer only if reconciliation fails
 * and can only be used with LwSciBufAttrListDebugDump function. Value will be
 * NULL if reconciliation succeeds.
 *
 * @return ::LwSciError, the completion status of the operation:
 * - ::LwSciError_Success, if reconciliation succeeds
 * - ::LwSciError_BadParameter, if any of the following oclwrs:
 *      - @a inputUnreconciledAttrlists is NULL
 *      - @a attrListCount is 0
 *      - not all the LwSciBufAttrLists in @a inputUnreconciledAttrlists are bound to the
 *        same LwSciBufModule.
 *      - an attribute key necessary for reconciling against the given data
 *        type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation is
 *        unset
 *      - an attribute key is set to an unsupported value considering the data
 *        type(s) of the LwSciBufAttrList(s) ilwolved in reconciliation
 *      - all the necessary output attributes belonging to the datatype(s)
 *          specified in @a inputUnreconciledAttrLists are not provided in
 *          @a outputAttributes
 *      - @a outputReconciledAttrList is NULL
 *      - any of the attribute in @a outputAttributes is not a output-only
 *          attribute
 *      - any of the attribute is repeated more than once in @a outputAttributes
 *      - any of the attribute in @a outputAttributes is set to an
 *          unsupported value considering the datatype(s) of LwSciBufAttrList(s)
 *          ilwolved in reconciliation.
 * - ::LwSciError_InsufficientMemory, if unable to allocate memory for
 *      reconciled LwSciBufAttrList
 * - ::LwSciError_ReconciliationFailed, if
 *      - reconciliation errors occur
 *      - size, alignment consistency check fails
 * - ::LwSciError_Overflow if internal integer overflow is detected
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - ::LwSciError_NotSupported if any of the following occur:
 *      - an attribute key is set in inputUnreconciledAttrLists resulting in a
 *          combination of given constraints that are not supported.
 *      - buffer type set in inputUnreconciledAttrLists is not supported
 * - ::LwSciError_IlwalidState if any of the following occur:
 *      - a new LwSciBufAttrList cannot be associated with the LwSciBufModule
 *          associated with the LwSciBufAttrList(s) in the given
 *          @a inputUnreconciledAttrLists to create a new reconciled
 *          LwSciBufAttrList
 * - Panic if:
 *      - unreconciled LwSciBufAttrList(s) in @a inputUnreconciledAttrLists
 *          is not valid.
*/
LwSciError LwSciBufAttrListReconcileWithOutputAttrs(
    LwSciBufAttrList inputUnreconciledAttrLists[],
    size_t attrListCount,
    LwSciBufOutputAttrKeyValuePair outputAttributes,
    LwSciBufAttrList* outputReconciledAttrList,
    LwSciBufAttrList* conflictAttrList);

#if defined(__cplusplus)
}
#endif

#endif /* INCLUDED_LWSCIBUF_INTERNAL_X86_H */
