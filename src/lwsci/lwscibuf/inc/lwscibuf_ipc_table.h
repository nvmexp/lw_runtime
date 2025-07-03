/*
 * lwscibuf_ipc_table.h
 *
 * Header file for LwSciBuf IPC Table handling.
 *
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_IPC_TABLE_H
#define INCLUDED_LWSCIBUF_IPC_TABLE_H

#include "lwsciipc_internal.h"
#include "lwscilist.h"
#include "lwscierror.h"

/**
 * @addtogroup lwscibuf_blanket_statements
 * @{
 */

/**
 * \page lwscibuf_page_blanket_statements
 * \section lwscibuf_in_params Input parameters
 *
 *  - LwSciBufIpcTable* passed as input parameter to an API is valid input if
 *  it is returned from a successful call to LwSciBufCreateIpcTable()/
 *  LwSciBufIpcTableImport()/LwSciBufIpcTableClone() and not yet freed by
 *  LwSciBufFreeIpcTable().
 *
 *  - LwSciBufIpcRoute* passed as input parameter to an API is valid input if
 *  it is returned from a successful call to LwSciBufIpcRouteImport()/
 *  LwSciBufIpcRouteClone() or the value corresponding to
 *  LwSciBufPrivateAttrKey_SciIpcRoute key of a Unreconciled LwSciBufAttrList
 *  or LwSciBufIpcRoute* in a LwSciBufIpcTable* and not yet freed
 *  by LwSciBufFreeIpcRoute().
 *
 *  - LwSciBufIpcTableIter* passed as input parameter to an API is valid input if
 *  it is returned from a successful call to LwSciBufInitIpcTableIter() and
 *  not yet freed by LwSciBufFreeIpcIter().
 *
 * \implements{18842583}
 */

/**
 * @}
 */

typedef struct LwSciBufIpcRouteRec LwSciBufIpcRoute;

/**
 * \brief Defines opaque pointer to IPC Table structure
 */
typedef struct LwSciBufIpcTableRec LwSciBufIpcTable;

typedef struct LwSciBufIpcTableIterRec LwSciBufIpcTableIter;

/**
 * \brief enum defining the affinity based on which LwSciBufIpcRoute* can be
 * searched in the LwSciBufIpcTable*.
 *
 * \implements{}
 */
typedef enum {
    LwSciBufIpcRoute_AffinityNone = 0U,
    LwSciBufIpcRoute_OwnerAffinity = 1U,
    LwSciBufIpcRoute_SocAffinity = 2U,
    LwSciBufIpcRoute_RouteAffinity = 3U,
    LwSciBufIpcRoute_Max
} LwSciBufIpcRouteAffinity;

/**
 * \brief Creates an LwSciBufIpcTable* with space to store entryCount entries.
 * Each entry in the array holds a list of (key, value) pairs of various
 * attributes corresponding to an LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] entryCount Indicates the number of entries the LwSciBufIpcTable*
 * to be created can store. Valid range: [1, SIZE_MAX].
 *
 * \param[out] outIpcTable pointer at which the new LwSciBufIpcTable* address
 * is stored.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - panics if any one of the following:
 *         - @a entryCount is 0
 *         - @a outIpcTable is NULL
 *
 * \implements{18843048}
 */
LwSciError LwSciBufCreateIpcTable(
    size_t entryCount,
    LwSciBufIpcTable** outIpcTable);

/**
 * \brief Adds input LwSciBufIpcRoute* to LwSciBufIpcTable* at given index.
 *
 * \param[in] ipcTable LwSciBufIpcTable* to which input LwSciBufIpcRoute* is
 * added.
 * \param[in] ipcRoute LwSciBufIpcRoute* to be added to the LwSciBufIpcTable*.
 * NULL LwSciBufIpcRoute* is acceptable.
 * \param[in] index index at which the LwSciBufIpcRoute* needs to be added in the
 * LwSciBufIpcTable*.
 * Valid value: 0 <= index < number of entries with which LwSciBufIpcTable* is
 * created using LwSciBufCreateIpcTable().
 *
 * \return
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - panics if any one of the following:
 *      - ipcTable is NULL.
 *      - index >= number of entries with which LwSciBufIpcTable* is created
 *        using LwSciBufCreateIpcTable().
 *
 * \implements{}
 */
LwSciError LwSciBufIpcAddRouteToTable(
    LwSciBufIpcTable* ipcTable,
    const LwSciBufIpcRoute* ipcRoute,
    uint64_t index);

/**
 * \brief Adds input attribute of given length and value in the LwSciBufIpcTable*
 * corresponding to the LwSciBufIpcRoute* at input index.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTable* is not used
 *        by multiple threads at the same time
 *
 * \param[in] ipcTable LwSciBufIpcTable* in which the entry has to be added.
 * \param[in] index The index of the LwSciBufIpcRoute* in the LwSciBufIpcTable*
 * for which the attribute needs to be added in the LwSciBufIpcTable*.
 * Valid values: 0 <= index < number of entries with which
 * LwSciBufIpcTable* was created using LwSciBufCreateIpcTable().
 * \param[in] attrKey Attribue key which needs to be added to the list of
 * attributes corresponding LwSciBufIpcTable*.
 *  Valid values: The values of the following enums
 *    - All values of LwSciBufAttrKey type except lower and upper bound values.
 *    - All values of LwSciBufInternalAttrKey type except lower bound.
 * \param[in] len size of the value provided by the user
 *   Valid range: 0 < len <= UINT64_MAX
 * \param[in] value address at which the value is stored.
 *   Valid value: any valid non-NULL pointer
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_Overflow if computing operation
 *   (including add, subtract, multiply) exceed data type MAX.
 * - panics if any one of the following:
 *         - @a len is 0
 *         - @a value is NULL
 *         - @a ipcTable is NULL
 *         - index >= number of entries with which LwSciBufIpcTable* is created
 *           using LwSciBufCreateIpcTable().
 *         - The attribute is already present in the LwSciBufIpcRoute* and caller
 *           tries to add the same attribute via @a attrKey again.
 *
 * \implements{18843051}
 */
LwSciError LwSciBufIpcAddAttributeToTable(
    LwSciBufIpcTable* ipcTable,
    uint64_t index,
    uint32_t attrKey,
    uint64_t len,
    const void* value);

/**
 * \brief Creates an iterator to iterate through the input LwSciBufIpcTable*.
 * The created iterator by this API can iterate only through the entries that
 * meet the following criteria:
 *   1) If @a routeAffinity is LwSciBufIpcRoute_OwnerAffinity, then the
 *   LwSciBufIpcTableIter* is returned such that the second LwSciIpcEndpoint in
 *   the LwSciBufIpcRoute* of an entry in the LwSciBufIpcTable* matches with
 *   @a ipcEndpoint. The match implies that the current peer imported the
 *   unreconciled LwSciBufAttrList which originated at the exporting peer.
 *   2) If @a routeAffinity is LwSciBufIpcRoute_RouteAffinity, then the
 *   LwSciBufIpcTableIter* is returned such that any of the LwSciIpcEndpoint in
 *   the LwSciBufIpcRoute* of an entry in the LwSciBufIpcTable* matches with
 *   @a ipcEndpoint.
 *   3) If @a routeAffinity is LwSciBufIpcRoute_SocAffinity, then the
 *   LwSciBufIpcTableIter* is returned such that the second LwSciIpcEndpoint in
 *   the LwSciBufIpcRoute* of an entry in the LwSciBufIpcTable* matches with
 *   @a ipcEndpoint and the SocId associated with the first LwSciIpcEndpoint in
 *   the LwSciBufIpcRoute* is different than the SocId of the @a ipcEndpoint
 *   implying that the unreconciled LwSciBufAttrList associated with this
 *   LwSciBufIpcRoute* originated from different Soc than the current Soc.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTable* is not used
 *        by multiple threads at the same time
 *
 * \param[in] inputIpcTable LwSciBufIpcTable* to which the iterator is created.
 * \param[in] ipcEndpoint The LwSciIpcEndpoint. Note that 0 is also a valid
 *            value.
 * \param[in] routeAffinity Indicates the affinity based on which
 *            the LwSciBufIpcRoute* in the LwSciBufIpcTable* should be found.
 *            valid value: LwSciBufIpcRoute_OwnerAffinity <= routeAffinity <
 *                          LwSciBufIpcRoute_Max
 * \param[out] outIpcIter Pointer that contains the address of the LwSciBufIpcTable*
 * iterator.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - panics if any one of the following:
 *         - @a inputIpcTable is NULL
 *         - @a outIpcIter is NULL
 *
 * \implements{18843093}
 */
LwSciError LwSciBufInitIpcTableIter(
    const LwSciBufIpcTable* inputIpcTable,
    LwSciBufIpcTableIter** outIpcIter);

/**
 * \brief Gets the length and value pointer of the user specified attribute
 * key from the LwSciBufIpcTable* entry to which the iterator has lwrrently
 * matched to the LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTableIter* is not
 *        used by multiple threads at the same time
 *      - Calls to this function for a particular LwSciBufIpcTableIter* instance
 *        must be externally synchronized if modification of the associated
 *        LwSciBufIpcTable* may potentially occur
 *
 * \param[in] ipcIter The LwSciBufIpcTableIter*.
 * \param[in] attrKey Attribute Key whose value has to be returned from the
 * current LwSciBufIpcRoute* entry.
 *  Valid values: The values of the following enums
 *    - All values of LwSciBufAttrKey type except lower and upper bound values.
 *    - All values of LwSciBufInternalAttrKey type except lower bound.
 *
 * \param[out] len Address at which the length of the value to be stored.
 * \param[out] value Address at which pointer to the value has to be stored.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_IlwalidOperation if this function is called before
 *   LwSciBufIpcTableIterNext function is ilwoked at least once with the
 *   @a ipcIter value.
 * - panics if any one of the following:
 *         - @a ipcIter is NULL
 *         - @a len is NULL
 *         - @a value is NULL
 *
 * \implements{18843090}
 */
LwSciError LwSciBufIpcIterLwrrGetAttrKey(
    const LwSciBufIpcTableIter* ipcIter,
    uint32_t attrKey,
    size_t* len,
    const void** value);

/**
 * \brief Iterates to the next entry in the LwSciBufIpcTable* that matches the
 * LwSciBufIpcRoute* containing the LwSciIpcEndpoint of the iterator. Note that
 * this function iterates over all the entries if the LwSciBufIpcTableIter* is
 * initialized with LwSciIpcEndpoint 0.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTableIter* is not
 *        used by multiple threads at the same time
 *      - Calls to this function for a particular LwSciBufIpcTableIter* instance
 *        must be externally synchronized if modification of the associated
 *        LwSciBufIpcTable* may potentially occur
 *
 * \param[in] ipcIter The LwSciBufIpcTableIter*.
 *
 * \returns bool
 * - True if successfully able to iterate to an entry that matches to the
 *   LwSciIpcEndpoint in the iterator.
 * - False otherwise.
 * - Panics if @a ipcIter is NULL.
 *
 * \implements{18843087}
 */
bool LwSciBufIpcTableIterNext(
    LwSciBufIpcTableIter* ipcIter);

#if (LW_IS_SAFETY == 0)
void LwSciBufPrintIpcTable(
    const LwSciBufIpcTable* ipcTable);
#endif

/**
 * \brief Gets the size required to export a LwSciBufIpcRoute*.
 * If @a ipcEndpoint is non-zero then sizeof(LwSciIpcEndpoint) is added to
 * the total size of LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufIpcRoute* is
 *        never modified after creation (so there is no data-dependency)
 *
 * \param[in] ipcRoute LwSciBufIpcRoute* for which size has to be computed.
 * \param[in] ipcEndpoint LwSciIpcEndpoint.
 * Valid value: Valid LwSciIpcEndpoint or the zero value.
 *
 * \returns size_t
 *  - 0 if there is any overflow error during size computation
 *  - [1, SIZE_MAX], otherwise
 *
 * \implements{18843084}
 */
size_t LwSciBufIpcRouteExportSize(
    const LwSciBufIpcRoute* ipcRoute,
    LwSciIpcEndpoint ipcEndpoint);

/**
 * \brief Gets the serialized export descriptor and its size for the
 * user specified LwSciBufIpcRoute*. If @a ipcEndpoint is non-zero then it is
 * added at the end of serialized LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufIpcRoute* is
 *        never modified after creation (so there is no data-dependency)
 *
 * \param[in] ipcRoute The LwSciBufIpcRoute* for which the export descriptor
 * needs to be computed.
 * \param[out] desc Address at which the pointer to the export descriptor
 * has to be stored.
 * \param[out] len Address at which size of the export descriptor has to be
 * stored.
 * \param[in] ipcEndpoint LwSciIpcEndpoint
 * Valid value: Valid LwSciIpcEndpoint or the zero value.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - panics if any one of the following:
 *         - @a desc is NULL
 *         - @a len is NULL
 *
 * \implements{18843081}
 */
LwSciError LwSciBufIpcRouteExport(
    const LwSciBufIpcRoute* ipcRoute,
    void** desc,
    size_t* len,
    LwSciIpcEndpoint ipcEndpoint);

/**
 * \brief Gets the size required to export the entries of a LwSciBufIpcTable*
 * that match the user specified LwSciIpcEndpoint as their outer endpoint.
 * If the entire LwSciBufIpcTable* need to be exported, then user can specify the
 * @a ipcEndpoint as NULL and in such cases, user can choose to
 * exclude/include NULL LwSciBufIpcRoute*(s).
 * (Note: A NULL LwSciBufIpcRoute* is a route whose endpoint count is 0.)
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTable* is not used
 *        by multiple threads at the same time
 *
 * \param[in] ipcTable LwSciBufIpcTable* for which size has to be computed.
 * \param[in] ipcEndpoint IPC LwSciIpcEndpoint which needs to be matched against
 * the outer LwSciIpcEndpoint of the LwSciBufIpcRoute* of each entry
 * in the LwSciBufIpcTable*.
 *
 * \returns size_t
 *  - 0 if there is any overflow error during size computation
 *  - [1, SIZE_MAX], otherwise
 *
 * \implements{18843078}
 */
size_t LwSciBufIpcTableExportSize(
    const LwSciBufIpcTable* const ipcTable,
    LwSciIpcEndpoint ipcEndpoint);

/**
 * \brief Gets the serialized export descriptor and descriptor size for the
 * entries in the user specified LwSciBufIpcTable* that match the user specified
 * LwSciIpcEndpoint as their outer endpoint.
 * If the entire LwSciBufIpcTable* needs to be exported,
 * then the user can specify the
 * @a ipcEndpoint as NULL and in such a case, user can choose to export by
 * excluding/including NULL LwSciBufIpcRoute(s).
 * (Note: A NULL LwSciBufIpcRoute* is a route whose endpoint count is 0.)
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTable* is not used
 *        by multiple threads at the same time
 *
 * \param[in] ipcTable The LwSciBufIpcTable* for which size has to be computed.
 * \param[in] ipcEndpoint IPC LwSciIpcEndpoint which needs to be matched against
 * the outer LwSciIpcEndpoint of the LwSciBufIpcRoute* of each entry
 * in the LwSciBufIpcTable*.
 * \param[out] desc Address at which the pointer to the export descriptor
 *  has to be stored.
 * \param[out] len Address at which size of the export descriptor has to be
 *  stored.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_Overflow if computing operation
 *   (including add, subtract, multiply) exceed data type MAX.
 * - panics if any one of the following:
 *         - @a desc is NULL
 *         - @a len is NULL
 *         - @a ipcTable is NULL
 *         - Valid entry count in given LwSciBufIpcTable* is 0
 *
 * \implements{18843075}
 */
LwSciError LwSciBufIpcTableExport(
    const LwSciBufIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    void** desc,
    size_t* len);

/**
 * \brief Imports a LwSciBufIpcRoute* from the descriptor which was created by
 * LwSciIpcIpcRouteExport function.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] ipcEndpoint The LwSciIpcEndpoint.
 * \param[in] desc Pointer to the export descriptor.
 *    Valid value: descriptor returned by LwSciBufIpcRouteExport function.
 * \param[in] len Size of the export descriptor.
 *    Valid value: len returned by LwSciBufIpcRouteExport function for the
 *    above descriptor. Range: [8, SIZE_MAX]
 * \param[out] ipcRoute Address at which the imported LwSciBufIpcRoute* has to be
 * stored.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - ipcRoute is NULL
 *      - desc is NULL
 *      - desc is invalid
 *      - len is invalid
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_Overflow if computing operation
 *   (including add, subtract, multiply) exceed data type MAX.
 *
 * \implements{18843072}
 */
LwSciError LwSciBufIpcRouteImport(
    LwSciIpcEndpoint ipcEndpoint,
    const void* desc,
    size_t len,
    LwSciBufIpcRoute** ipcRoute);

/**
 * \brief Imports a LwSciBufIpcTable* from the descriptor which was created by
 * LwSciIpcIpcTableExport function. If @a ipcEndpoint is non-zero then it is
 * added at the end of every deserialized LwSciBufIpcRoute* in the imported
 * LwSciBufIpcTable*.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \param[in] desc Pointer to the imported descriptor
 *    Valid value: descriptor returned by LwSciBufIpcTableExport function.
 * \param[in] len Size of the imported descriptor.
 *    Valid value: len returned by LwSciBufIpcTableExport function for the
 *    above descriptor. Range: [8, SIZE_MAX]
 * \param[out] ipcTable Address at which the imported LwSciBufIpcTable* has to be
 * stored.
 * \param[in] ipcEndpoint LwSciIpcEndpoint
 * Valid value: Valid LwSciIpcEndpoint or the zero value.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - desc is NULL
 *      - desc is invalid
 *      - len is invalid
 *      - ipcTable is NULL
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_Overflow if computing operation
 *   (including add, subtract, multiply) exceed data type MAX.
 * - LwSciError_NotSupported if locally computed LwSciIpcEndpoint size is
 *   less than the size of a LwSciIpcEndpoint in the descriptor header. This
 *   indicates that exporter and importer are using different versions of
 *   SCI library and size of the LwSciIpcEndpoint is matching in both of them
 *   and hence the import descriptor cannot be supported on this version.
 *
 * \implements{18843069}
 */
LwSciError LwSciBufIpcTableImport(
    const void* desc,
    size_t len,
    LwSciBufIpcTable** ipcTable,
    LwSciIpcEndpoint ipcEndpoint);

/**
 * \brief Creates a new LwSciBufIpcRoute* and copies the content of
 * source LwSciBufIpcRoute* to the new LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufIpcRoute* is
 *        never modified after creation (so there is no data-dependency)
 *
 * \param[in] srcIpcRouteAddr Address at which the source LwSciBufIpcRoute* is stored.
 *   Valid value: any non NULL pointer that is the pointer to a LwSciBufIpcRoute*
 *   returned by LwSciBufIpcRouteImport or this API.
 * \param[out] dstIpcRouteAddr Address at which the cloned LwSciBufIpcRoute*
 * has to be stored.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - panics if any one of the following:
 *         - @a srcIpcRouteAddr is NULL
 *         - @a dstIpcRouteAddr is NULL
 *
 * \implements{18843066}
 */
LwSciError LwSciBufIpcRouteClone(
    const LwSciBufIpcRoute* const * srcIpcRouteAddr,
    LwSciBufIpcRoute** dstIpcRouteAddr);

/**
 * \brief Creates a new LwSciBufIpcTable* and copies the content of
 * source LwSciBufIpcTable* to the new LwSciBufIpcTable2*.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTable* is not used
 *        by multiple threads at the same time
 *
 * \param[in] srcIpcTableAddr Address at which the source LwSciBufIpcTable* is stored.
 *   Valid value: any non NULL pointer that is the pointer to a LwSciBufIpcTable*
 *   returned by LwSciBufIpcRouteImport or this API.
 * \param[out] dstIpcTableAddr Address at which the cloned LwSciBufIpcTable* has to be
 * stored.
 *
 * \returns LwSciError, the completion code of this operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is insufficient memory to complete
 *   the operation.
 * - LwSciError_Overflow if computing operation
 *   (including add, subtract, multiply) exceed data type MAX.
 * - panics if any one of the following:
 *         - @a srcIpcTableAddr is NULL
 *         - @a dstIpcTableAddr is NULL
 *
 * \implements{18843063}
 */
LwSciError LwSciBufIpcTableClone(
    const LwSciBufIpcTable* const * srcIpcTableAddr,
    LwSciBufIpcTable** dstIpcTableAddr);

/**
 * \brief Free the memory oclwpied by the LwSciBufIpcTable*.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTable* is not used
 *        by multiple threads at the same time
 *
 * \param[in] valPtr pointer to the LwSciBufIpcTable*.
 *
 * \returns void
 *
 * \implements{18843060}
 */
void LwSciBufFreeIpcTable(
    LwSciBufIpcTable* const * valPtr);

/**
 * \brief Free the memory oclwpied by the LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcRoute* is not used
 *        by multiple threads at the same time
 *
 * \param[in] valPtr pointer to the LwSciBufIpcRoute*.
 *
 * \returns void
 *
 * \implements{18843057}
 */
void LwSciBufFreeIpcRoute(
    LwSciBufIpcRoute* const * valPtr);

/**
 * \brief Free the memory oclwpied by the LwSciBufIpcTableIter*.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufIpcTableIter* is not
 *        used by multiple threads at the same time
 *
 * \param[in] ipcIter The LwSciBufIpcTableIter*.
 *
 * \returns void
 *
 * \implements{18843054}
 */
void LwSciBufFreeIpcIter(
    LwSciBufIpcTableIter* ipcIter);

/**
 * \brief Checks if given LwSciBufIpcRoute* has the same affinity as the
 * @a routeAffinity.
 *
 * \param[in] ipcRoute LwSciBufIpcRoute* whose affinity is checked against
 * @a routeAffinity.
 * \param[in] routeAffinity LwSciBufIpcRouteAffinity to be checked for the
 * @a ipcRoute.
 * Valid value: LwSciBufIpcRoute_AffinityNone <= routeAffinity <
 *                LwSciBufIpcRoute_Max
 * \param[in] ipcEndpoint LwSciIpcEndpoint.
 * Valid value: Valid LwSciIpcEndpoint if @a localPeer is true, ignored
 * otherwise.
 * \param[in] localPeer boolean value indicating if the LwSciBufIpcRouteAffinity
 * is checked for local peer or remote peer.
 * \param[out] isMatch boolean value indicating whether the @a ipcRoute has
 * same affinity as that represented by @a routeAffinity. True implies that
 * @a ipcRoute and @a routeAffinity has the same affinity. False implies
 * otherwise.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a isMatch is NULL
 *      - @a routeAffinity >= LwSciBufIpcRoute_Max
 *      - @a localPeer is false AND @a ipcEndpoint is 0.
 *
 * \implements{}
 */
void LwSciBufIpcRouteMatchAffinity(
    const LwSciBufIpcRoute* ipcRoute,
    LwSciBufIpcRouteAffinity routeAffinity,
    LwSciIpcEndpoint ipcEndpoint,
    bool localPeer,
    bool* isMatch);

/**
 * \brief Gets LwSciBufIpcRoute* from LwSciBufIpcTableIter*.
 * Note that LwSciBufIpcTableIter* must be initialized with
 * LwSciBufInitIpcTableIter() before this function can be called.
 *
 * \param[in] iter LwSciBufIpcTableIter*.
 * Valid value: Valid LwSciBufIpcTableIter* obtained by calling
 * LwSciBufInitIpcTableIter() or by iterating over LwSciBufIpcTable* by calling
 * LwSciBufIpcTableIterNext().
 * \param[out] ipcRoute LwSciBufIpcRoute*
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a iter is NULL.
 *      - @a ipcRoute is NULL.
 *
 * \implements{}
 */
void LwSciBufIpcGetRouteFromIter(
    const LwSciBufIpcTableIter* iter,
    const LwSciBufIpcRoute** ipcRoute);

#endif //INCLUDED_LWSCIBUF_IPC_TABLE_H
