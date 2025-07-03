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
 * \brief <b>LwSciSync core attribute definitions</b>
 *
 * @b Description: This file declares items common to attribute unit cluster
 */

#ifndef INCLUDED_LWSCISYNC_ATTRIBUTE_TRANSPORT_H
#define INCLUDED_LWSCISYNC_ATTRIBUTE_TRANSPORT_H

#include <stdbool.h>
#include <stddef.h>
#include "lwscierror.h"
#include "lwscisync.h"

/**
 * \page lwscisync_page_unit_blanket_statements LwSciSync blanket statements
 * \section lwscisync_in_out_params Input/Output parameters
 * - LwSciSyncAttrList descriptor passed as input parameter to an API is
 *   validated by checking for the correct tags and their correct order.
 */

/**
 * \brief Callwlates the proper export LwSciSyncAccessPerm for a given
 * LwSciIpcEndpoint.
 *
 * Retrieves LwSciSyncCoreAttrListObj with LwSciSyncCoreAttrListGetObjFromRef.
 * Calls LwSciSyncCoreIpcTableGetPermAtSubTree() to get the permisions data.
 * Returns final permissions via *actualPerm.
 *
 * \param[in] attrList The LwSciSyncAttrList
 * \param[in] ipcEndpoint LwSciIpcEndpoint for which LwSciSyncAccessPerm is needed
 * \param[out] actualPerm pointer where value of actual LwSciSyncAccessPerm is
 * written
 *
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter in case of failure.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could cause system error. No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, the user must ensure that @a attrList is not modified during the
 *   call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844377}
 */
LwSciError LwSciSyncCoreAttrListGetIpcExportPerm(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAccessPerm* actualPerm);

/**
 * \brief Callwlate the proper export waiterRequireTimestamps value for a given
 * LwSciIpcEndpoint.
 *
 * Finds all ipc routes in slot 0's ipcPerm table that end with LwSciIpcEndpoint.
 * Aggregates all waiterRequireTimestamps corresponding to those routes.
 * Returns final value via *waiterRequireTimestamps.
 *
 * \param[in] attrList The LwSciSyncAttrList
 * \param[in] ipcEndpoint LwSciIpcEndpoint for which waiterRequireTimestamps
 * value is needed
 * \param[out] waiterRequireTimestamps pointer where value of
 * waiterRequireTimestamps is written
 *
 * \return void
 * - Panics on internal failure
 */
void LwSciSyncCoreAttrListGetIpcExportRequireTimestamps(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint ipcEndpoint,
    bool* waiterRequireTimestamps);

/**
 * Iterates through each of the LwSciSyncAttrList in
 * unreconciledAttrListArray[] to identify the total size of the attribute
 * values and key-value pairs. Allocates the memory for the transport buffers
 * of type LwSciCommonTransportBuf* based on the total size identified, including
 * the metadata necessary for the export descriptor, using
 * LwSciCommonTransportAllocTxBufferForKeys() and copies all the attribute keys
 * and values to it using LwSciCommonTransportAppendKeyValuePair().
 *
 * Finally export the LwSciCommonTransportBuf* using
 * LwSciCommonTransportPrepareBufferForTx().
 *
 * The descBuf can be freed using LwSciSyncAttrListFreeDesc().
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the @a unreconciledAttrListArray array requires thread
 *   synchronization; without the synchronization, the function could access
 *   invalid LwSciSyncAttrList objects, in turn causing a system error or
 *   calling LwSciCommonPanic(). No synchronization is done in the function.
 *   To ensure that no system error oclwrs, and that correct objects are
 *   accessed, the user must ensure that the elements are not modified during
 *   the call to the function.
 * - Access to elements of the @a unreconciledAttrListArray array requires
 *   thread synchronization; without the synchronization, the function could
 *   read incorrect or invalid attribute values from the elements. This
 *   synchronization is done using LwSciObj.objLock mutex object associated
 *   with each of the elements.
 * - Locking of elements of the @a unreconciledAttrListArray array requires
 *   thread synchronization; without synchronization, the function could cause
 *   a deadlock when called in parallel from two or more threads. This
 *   synchronization is done by locking the elements in a specific order, and
 *   this order is the same during every call to the function from every
 *   thread.
 * - The operations are not expected to cause nor contribute to a deadlock
 *   since:
 * - the mentioned @a unreconciledAttrListArray array element mutex object is
 *   locked immediately before and released immediately after the attribute
 *   values are set,
 * - the mentioned @a unreconciledAttrListArray array elements mutex objects
 *   are locked in a specific order, and this order is the same during every
 *   call to the function from every thread.
 *
 * \implements{18844359}
 *
 * \fn LwSciError LwSciSyncAttrListIpcExportUnreconciled(
 *   const LwSciSyncAttrList unreconciledAttrListArray[],
 *   size_t unreconciledAttrListCount,
 *   LwSciIpcEndpoint ipcEndpoint,
 *   void** descBuf,
 *   size_t* descLen);
 */

/**
 * Iterates through the reconciled LwSciSyncAttrList to
 * identify the total size of the attribute values and key-value pairs.
 * Allocates the memory for the transport buffers
 * of type LwSciCommonTransportBuf* based on the total size identified, including
 * the metadata necessary for the export descriptor, using
 * LwSciCommonTransportAllocTxBufferForKeys() and copies all the attribute keys
 * and values to it using LwSciCommonTransportAppendKeyValuePair().
 *
 * Finally export the LwSciCommonTransportBuf* using
 * LwSciCommonTransportPrepareBufferForTx().
 *
 * The descBuf can be freed using LwSciSyncAttrListFreeDesc().
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to dereferenced descBuf value requires thread synchronization;
 *   without synchronization, the function could cause system error. No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, the user must ensure that the dereferenced descBuf value is not
 *   modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844362}
 *
 * \fn LwSciError LwSciSyncAttrListIpcExportReconciled(
 *   const LwSciSyncAttrList reconciledAttrList,
 *   LwSciIpcEndpoint ipcEndpoint,
 *   void** descBuf,
 *   size_t* descLen);
 */

/**
 * Gets the LwSciCommonTransportBuf* from the descBuf using
 * LwSciCommonTransportGetRxBufferAndParams(). Iterates through the received
 * TransportBuf and reads its attributes' key-values using
 * LwSciCommonTransportGetNextKeyValuePair() and puts those attributes in the
 * new unreconciled LwSciSyncAttrList.
 * The new unreconciled LwSciSyncAttrList has the following properties:
 * - its number of slots equals to the sum of numbers of slots of the
 *   unreconciled LwSciSyncAttrList used in creation of the input
 *   LwSciSyncAttrList export descriptor,
 * - there is a 1 - 1 correspondence between slots in the new LwSciSyncAttrList
 *   and slots in the unreconciled LwSciSyncAttrLists used in creation of the
 *   input LwSciSyncAttrList export descriptor, such that all attributes except
 *   signalerPrimitiveInfo and waiterPrimitiveInfo in the corresponding pairs
 *   have the same values,
 * - SignalerPrimitiveInfo in all the slots in the output LwSciSyncAttrList have
 *   values set in the input LwSciSyncAttrList export descriptor accordingly
 * - WaiterPrimitiveInfo in all the slots in the output LwSciSyncAttrList have
 *   values set in the input LwSciSyncAttrList export descriptor
 * - the LwSciSyncAttrList is bound to the input LwSciSyncModule.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the @a module, needed for creation of the attribute list,
 *   requires thread synchronization; without synchronization, the function
 *   could cause a call to LwSciCommonPanic(). No synchronization is done in
 *   the function. To ensure that LwSciCommonPanic() is not called, the user
 *   must ensure that the @a module value is not modified during the call to
 *   the function.
 * - Incrementing the reference count of the @a module during duplication of
 *   the @a module also requires thread synchronization. This synchronization
 *   is done by locking LwSciObj.ObjLock mutex object associated with @a
 *   module.
 * - Access to memory starting from value of @a descBuf requires thread
 *   synchronization; without synchronization, the function could cause system
 *   error when called in parallel with a function which modifies the
 *   mentioned memory, or the function could create incorrect or invalid value
 *   of @a importedUnreconciledAttrList. No synchronization is done in the
 *   function. To ensure that no system error oclwrs, and that the @a
 *   importedUnreconciledAttrList value is valid and correct, the user must
 *   ensure that the mentioned memory is not modified during the call to the
 *   function.
 * - The operations are not expected to cause nor contribute to a deadlock
 *   since the mentioned LwSciObj.ObjLock mutex object associated with @a
 *   module is locked immediately before and released immediately after the
 *   reference count is incremented.
 *
 * \implements{18844365}
 *
 * \fn LwSciError LwSciSyncAttrListIpcImportUnreconciled(
 *   LwSciSyncModule module,
 *   LwSciIpcEndpoint ipcEndpoint,
 *   const void* descBuf,
 *   size_t descLen,
 *   LwSciSyncAttrList* importedUnreconciledAttrList);
 */

/**
 * Gets the LwSciCommonTransportBuf* from the descBuf using
 * LwSciCommonTransportGetRxBufferAndParams(). Iterates through the received
 * TransportBuf and reads its attributes' key-values using
 * LwSciCommonTransportGetNextKeyValuePair() and puts those attributes in the
 * new reconciled LwSciSyncAttrList.
 *
 * The received reconciled LwSciSyncAttrList is also validated against
 * un-reconciled inputUnreconciledAttrListArray[] by
 * LwSciSyncAttrListValidateReconciled().
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the @a module value, needed for creation of the attribute list,
 *   requires thread synchronization; without synchronization, the function
 *   could cause a call to LwSciCommonPanic(). No synchronization is done in
 *   the function. To ensure that LwSciCommonPanic() is not called, the user
 *   must ensure that the @a module value is not modified during the call to
 *   the function.
 * - Incrementing the reference count of the @a module during duplication of
 *   the @a module also requires thread synchronization. This synchronization
 *   is done by locking LwSciObj.ObjLock mutex object associated with @a
 *   module.
 * - Access to memory starting from value of @a descBuf requires thread
 *   synchronization; without synchronization, the function could cause system
 *   error when called in parallel with a function which modifies the
 *   mentioned memory, or the function could create incorrect or invalid value
 *   of @a importedReconciledAttrList. No synchronization is done in the
 *   function. To ensure that no system error oclwrs, and that the @a
 *   importedReconciledAttrList value is valid and correct, the user must
 *   ensure that the mentioned memory is not modified during the call to the
 *   function.
 * - Access to the @a inputUnreconciledAttrListArray array requires thread
 *   synchronization; without the synchronization, the function could access
 *   invalid LwSciSyncAttrList objects, in turn causing a system error or
 *   calling LwSciCommonPanic(). No synchronization is done in the function.
 *   To ensure that no system error oclwrs, and that correct objects are
 *   accessed, the user must ensure that the elements are not modified during
 *   the call to the function.
 * - Access to elements of the @a inputUnreconciledAttrListArray array
 *   requires thread synchronization; without the synchronization, the
 *   function could read incorrect or invalid attribute values from the
 *   elements. This synchronization is done using LwSciObj.objLock mutex
 *   associated with each of the elements.
 * - Locking of elements of the @a inputUnreconciledAttrListArray array
 *   requires thread synchronization; without synchronization, the function
 *   could cause a deadlock when called in parallel from two or more threads.
 *   This synchronization is done by locking the elements in a specific order,
 *   and this order is the same during every call to the function from every
 *   thread.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since:
 * - the mentioned LwSciObj.ObjLock mutex object associated with @a module is
 *   locked immediately before and released immediately after the reference
 *   count is incremented,
 * - the mentioned @a inputUnreconciledAttrListArray array element mutex
 *   object is locked immediately before and released immediately after the
 *   attribute values are set,
 * - the mentioned @a inputUnreconciledAttrListArray array elements mutex
 *   objects are locked in a specific order, and this order is the same during
 *   every call to the function from every thread.
 *
 * \implements{18844368}
 *
 * \fn LwSciError LwSciSyncAttrListIpcImportReconciled(
 *   LwSciSyncModule module,
 *   LwSciIpcEndpoint ipcEndpoint,
 *   const void* descBuf,
 *   size_t descLen,
 *   const LwSciSyncAttrList inputUnreconciledAttrListArray[],
 *   size_t inputUnreconciledAttrListCount,
 *   LwSciSyncAttrList* importedReconciledAttrList);
 */

/**
 * Uses LwSciCommonFree() to free the descBuf.
 *
 * Conlwrrency:
 * - Thread-safe: Yes
 * - Access to the dereferenced @a descBuf value requires thread
 *   synchronization; without synchronization, the function could cause a call
 *   to LwSciCommonPanic() when called in parallel with another function,
 *   which modifies memory starting sizeof(LwSciCommonAllocHeader) bytes
 *   before the dereferenced @a descBuf value. No synchronization is done in
 *   the function. To ensure that LwSciCommonPanic() is not called, the user
 *   must ensure that the mentioned memory is not modified during the call to
 *   the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844371}
 *
 * \fn void LwSciSyncAttrListFreeDesc(
 *   void* descBuf);
 */
#endif
