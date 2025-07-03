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
 * \brief <b>Ipc Table Management declarations</b>
 *
 * @b Description: This file declares IPC Table interface
 */

#ifndef INCLUDED_LWSCISYNC_IPC_TABLE_H
#define INCLUDED_LWSCISYNC_IPC_TABLE_H

/**
 * @defgroup lwsci_sync Synchronization APIs
 *
 * @ingroup lwsci_group_stream
 * @{
 */

#include "lwscicommon_transportutils.h"
#include "lwscierror.h"
#include "lwsciipc.h"
#include "lwscisync.h"
#include "lwscisync_core.h"
#include "lwscisync_internal.h"

/**
 * \page lwscisync_page_unit_blanket_statements LwSciSync blanket statements
 * \section lwscisync_in_out_params Input/Output parameters
 * - LwSciSyncCoreIpcTable passed as input parameter to an API is valid input
 *   if it is returned from a successful call to LwSciSyncCoreIpcTableTreeAlloc()
 *   and has not yet been deallocated using LwSciSyncCoreIpcTableFree().
 */

/**
 * \brief Contains the CPU access and required permissions of the original
 * LwSciSyncAttrList and the import path traversed by the LwSciSyncAttrList
 * starting from the node where the LwSciSyncAttrList was created.
 *
 * An Unreconciled LwSciSyncAttrList that is imported/exported keeps track
 * of LwSciIpcEndpoint(s) and their corresponding LwSciSyncTopoIds that
 * it was imported through. Those paths are then
 * provided to a Reconciled LwSciSyncAttrList so during its travel back the
 * correct permissions can be callwlated.
 *
 * \implements{18845835}
 */
typedef struct {
    /**
     * Path traversed by the LwSciSyncAttrList starting from its creator node.
     * This contains the LwSciIpcEndpoints of the previous importers of the
     * Unreconciled LwSciSyncAttrList.
     *
     * When Importing an LwSciSyncAttrList, this is set to the value of the
     * IPC Table Transport Key, and Exported using the same Transport Key.
     *
     * When Reconciling, the IPC Table branches from each of the Unreconciled
     * LwSciSyncAttrList are copied onto the new Reconciled LwSciSyncAttrList's
     * LwSciSyncCoreIpcTable via LwSciSyncCoreIpcTableAddBranch().
     *
     * When Cloning an LwSciSyncAttrList, this is dynamically allocated and the
     * LwSciIpcEndpoint(s) are copied from the source LwSciSyncAttrList to the
     * destination LwSciSyncAttrList.
     */
    LwSciIpcEndpoint* ipcRoute;
    /**
     * Additional information associated with LwSciIpcEndpoints. Entries
     * in this array correspond to entries in ipcRoute. It is created,
     * modified and copied together with ipcRoute.
     */
    LwSciSyncIpcTopoId* topoIds;
    /**
     * Number of entries in the ipcRoute array and topoIds array.
     *
     * When Importing an LwSciSyncAttrList, this is set in ImportIpcPermEntry
     * to the value obtained by the IPC Table Transport Key corresponding to
     * this Key Value, and Exported using the same Transport Key.
     *
     * When Cloning an LwSciSyncAttrList, this is set to the value specified
     * in the source LwSciSyncAttrList's member.
     *
     * When Appending Unreconciled LwSciSyncAttrLists, this is set in the
     * multi-slot LwSciSyncAttrList to the value specified in the corresponding
     * Unreconciled LwSciSyncAttrList's member.
     *
     * When Reconciling, the IPC Table branches from each of the Unreconciled
     * LwSciSyncAttrList are copied onto the new Reconciled LwSciSyncAttrList's
     * LwSciSyncCoreIpcTable via LwSciSyncCoreIpcTableAddBranch().
     *
     * This is updated by LwSciSyncCoreIpcTableLwtSubTree() to reduce the number
     * of entries being considered as part of the IPC Route tree that goes
     * through a particular LwSciIpcEndpoint.
     */
    size_t ipcRouteEntries;
    /**
     * Value for LwSciSyncAttrKey_NeedCpuAccess public key of the original
     * LwSciSyncAttrList.
     */
    bool needCpuAccess;
    /**
     * Value for LwSciSyncAttrKey_RequiredPerm public key of the original
     * LwSciSyncAttrList.
     */
    LwSciSyncAccessPerm requiredPerm;
    /**
     * Value for LwSciSyncAttrKey_WaiterRequireTimestamps public key of the
     * original LwSciSyncAttrList.
     */
    bool waiterRequireTimestamps;
    /**
     * Value of LwSciSyncInternalAttrKey_EngineArray internal key of the
     * original LwSciSyncAttrList.
     */
    LwSciSyncHwEngine engineArray[MAX_HW_ENGINE_TYPE];
    /**
     * Number of LwSciSyncHwEngine elements lwrrently stored in engineArray.
     */
    size_t engineArrayLen;
} LwSciSyncCoreAttrIpcPerm;


/**
 * \brief Contains information about the LwSciSyncObj tree topology.
 *
 * \implements{18845838}
 */
typedef struct {
    /**
     * Path traversed by the LwSciSyncAttrList starting from its creator node.
     * This contains the LwSciIpcEndpoints of the importer of unreconciled
     * LwSciSyncAttrList.
     *
     * During Import, this is dynamically allocated to an array with the
     * capacity to hold the previous path, plus an additional entry for the
     * current importer's LwSciIpcEndpoint by the handler for the
     * LwSciSyncCoreIpcTableKey_NumIpcEndpoint Transport Key during
     * LwSciSyncCoreImportIpcTable().
     *
     * When an LwSciSyncAttrList is Cloned or when Unreconciled
     * LwSciSyncAttrLists are appended, this is dynamically allocated to an
     * identical copy of the source LwSciSyncAttrList's member and set on
     * the destination LwSciSyncAttrList in the corresponding slot attribute
     * list.
     *
     * This is freed during LwSciSyncCoreIpcTableFree().
     */
    LwSciIpcEndpoint* ipcRoute;
    /**
     * Additional information associated with LwSciIpcEndpoints. Entries
     * in this array correspond to entries in ipcRoute. It is created,
     * modified and copied together with ipcRoute.
     * Entries are created from the corresponding LwSciIpcEndpoints in ipcRoute
     * with LwSciIpcEndpointGetTopoId() and LwSciIpcEndpointGetVuid().
     */
    LwSciSyncIpcTopoId* topoIds;
    /**
     * Number of entries in the ipcRoute array and topoIds array.
     *
     * This is initialized to 0 when the Unreconciled LwSciSyncAttrList has
     * not yet been exported/imported.
     *
     * During Import, this is set to the number of entries in the received IPC
     * Route, plus an additional entry for the current importer's
     * LwSciIpcEndpoint.
     *
     * When an LwSciSyncAttrList is Cloned or when Unreconciled
     * LwSciSyncAttrLists are appended, this is set to an identical copy of the
     * source LwSciSyncAttrList's member on the destination LwSciSyncAttrList
     * in the corresponding slot.
     */
    size_t ipcRouteEntries;

    /*** tree description used on the way back upstream */
    /**
     * Holds the IPC Route, topoIds, CPU access, requested permissions
     * and hardware engines
     * of Unreconciled LwSciSyncAttrList(s) used for reconciliation.
     *
     * This is dynamically allocated during LwSciSyncCoreIpcTableTreeAlloc()
     * and entries are added when LwSciSyncCoreIpcTableAddBranch() is called.
     *
     * This is freed during LwSciSyncCoreIpcTableFree().
     */
    LwSciSyncCoreAttrIpcPerm* ipcPerm;
    /**
     * Number of entries in the ipcPerm array.
     *
     * This is initialized when the IPC Table is allocated via
     * LwSciSyncCoreIpcTableTreeAlloc().
     */
    size_t ipcPermEntries;
} LwSciSyncCoreIpcTable;

/**
 * \brief Allocates the requested number of branches in input
 * LwSciSyncCoreIpcTable's LwSciSyncCoreAttrIpcPerm array. No allocation is
 * needed if the requested size is 0.
 *
 * Needs to be called before any LwSciSyncCoreIpcTableAddBranch() calls are made
 * on this input LwSciSyncCoreIpcTable.
 *
 * This function should be called only once for the same input
 * LwSciSyncCoreIpcTable.
 *
 * \param[in,out] ipcTable pointer to LwSciSyncCoreIpcTable with all members
 * zero/NULL initialized.
 * \param[in] size number of branches to allocate
 * Valid value: [0, SIZE_MAX]
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if not enough memory
 * - Panics if @a ipcTable is NULL or ipcTable is not zero/NULL initialized.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to @a ipcTable->ipcRoute requires thread synchronization; without
 *   synchronization, the function could cause system a call to
 *   LwSciCommonPanic() when the function is called from more than one thread
 *   in parallel. No synchronization is done in the function. To ensure that
 *   there is no call to LwSciCommonPanic(), the user must ensure that the
 *   function is not called from from other threads in parallel.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844545}
 */
LwSciError LwSciSyncCoreIpcTableTreeAlloc(
    LwSciSyncCoreIpcTable* ipcTable,
    size_t size);

/**
 * \brief Appends the LwSciIpcEndpoint to ipcRoute and its additional info
 * to topoIds in input LwSciSyncCoreIpcTable.
 *
 * \param[in,out] ipcTable LwSciSyncCoreIpcTable to add a new IPC Route node to
 * \param[in] ipcEndpoint value in the new node in IPC Route
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if ipcEndpoint is not a valid LwSciSyncIpcEndpoint
 * - LwSciError_InsufficientMemory if not enough memory
 * - Panics if @a ipcTable is NULL or if trying to Append in empty ipcRoute
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to @a ipcTable->ipcRouteEntries requires thread synchronization;
 *   without synchronization, the function could cause a call to
 *   LwSciCommonPanic() or memory leak when the function is called from more
 *   than one thread in parallel. No synchronization is done in the function.
 *   To ensure that there is no call to LwSciCommonPanic() nor memory leak,
 *   the user must ensure that the function is not called from other threads
 *   in parallel.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844548}
 */
LwSciError LwSciSyncCoreIpcTableAppend(
    LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint);

/**
 * \brief Adds a new branch to the LwSciSyncCoreAttrIpcPerm array stored in
 * the LwSciSyncCoreIpcTable.
 *
 * Allocates a new ipcRoute and topoIds in provided slot in ipcTable's
 * LwSciSyncCoreAttrIpcPerm array and copies ipcRoute from ipcTableWithRoute in it.
 * The other members in this LwSciSyncCoreAttrIpcPerm array's slot get
 * associated with the provided input values describing permissions, whether
 * the waiter requires timestamps, and the LwSciSyncHwEngine access.
 *
 * Needs to be called only after LwSciSyncCoreIpcTableTreeAlloc() call is made
 * on this input LwSciSyncCoreIpcTable.
 *
 * \param[in,out] ipcTable LwSciSyncCoreIpcTable to add a new IPC Route node to
 * \param[in] slot Index of the newly created branch
 * Valid value: [0, number of branches - 1]
 * \param[in] ipcTableWithRoute Contains ipcRoute to copy
 * Valid value: ipcTableWithRoute is valid if it is non-NULL.
 * \param[in] needCpuAccess value associated with the new branch.
 * Valid value: needCpuAccess is valid if it is either true or false.
 * \param[in] waiterRequireTimestamps value associated with the new branch.
 * Valid value: waiterRequireTimestamps is valid if it is either true or false.
 * \param[in] requiredPerm LwSciSyncAccessPerm associated with the new branch.
 * Valid value: requiredPerm is valid for all LwSciSyncAccessPerm except for
 * LwSciSyncAccessPerm_Auto.
 * \param[in] engineArray LwSciSyncHw engine array associated with the new branch
 * Valid value: engineArray is valid if if each member is a valid LwSciSyncHwEngine
 * \return LwSciError
 * \param[in] engineArrayLen Number of entries in the engineArray array
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if not enough memory
 * - Panics if any of the following oclwrs:
 *      - slot is not a valid index of an empty branch
 *      - ipcRoute for the branch is NULL
 *      - topoIds for the branch is NULL
 *      - ipcTable is NULL
 *      - ipcTableWithRoute is NULL
 *      - engineArray is NULL
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Allocation of @a ipcTable->ipcPerm[].ipcRoute array requires thread
 *   synchronization; without synchronization, the function could cause memory
 *   leak when the function is called from more than one thread in parallel.
 *   No synchronization is done in the function. To ensure that there is no
 *   memory leak, the user must ensure that the function is not called from
 *   from other threads in parallel.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844551}
 */
LwSciError LwSciSyncCoreIpcTableAddBranch(
    const LwSciSyncCoreIpcTable* ipcTable,
    size_t slot,
    const LwSciSyncCoreIpcTable* ipcTableWithRoute,
    bool needCpuAccess,
    bool waiterRequireTimestamps,
    LwSciSyncAccessPerm requiredPerm,
    LwSciSyncHwEngine* engineArray,
    size_t engineArrayLen);

/**
 * \brief Creates a new LwSciSyncCoreIpcTable containing copy of all entries of
 * input LwSciSyncCoreIpcTable.
 *
 * The newIpcTable should be 0/NULL initialized.
 *
 * \param[in] ipcTable LwSciSyncCoreIpcTable to copy
 * \param[in,out] newIpcTable new LwSciSyncCoreIpcTable
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if not enough memory to create a new
 *   LwSciSyncCoreIpcTable
 * - Panics if @a ipcTable is NULL or @a newIpcTable is NULL or is not 0/NULL
 *   initialized.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to @a ipcTable->ipcRouteEntries, @a newIpcTable->ipcRoute, @a
 *   newIpcTable->ipcPermEntries and @a newIpcTable->ipcPerm all require
 *   thread synchronization; without synchronization, the function could cause
 *   a call to LwSciCommonPanic() when the function is called from more than
 *   one thread in parallel. No synchronization is done in the function. To
 *   ensure that there is no call to LwSciCommonPanic(), the user must ensure
 *   that the function is not called from from other threads in parallel.
 * - Allocation of @a newIpcTable->ipcRoute, @a newIpcTable->ipcPerm and @a
 *   newIpcTable->ipcPerm[].ipcRoute arrays all require thread
 *   synchronization; without synchronization, the function could cause memory
 *   leak when the function is called from more than one thread in parallel.
 *   No synchronization is done in the function. To ensure that there is no
 *   memory leak, the user must ensure that the function is not called from
 *   from other threads in parallel.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844554}
 */
LwSciError LwSciSyncCoreCopyIpcTable(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciSyncCoreIpcTable* newIpcTable);

/**
 * \brief Free memory resources associated with the LwSciSyncCoreIpcTable
 * and 0/NULL initializes the same.
 *
 * \param[in,out] ipcTable LwSciSyncCoreIpcTable to free
 * \return void
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Releasing of the @a ipcTable->ipcRoute, @a ipcTable->ipcPerm[].ipcRoute
 *   and @a ipcTable->ipcPerm all require thread synchronization; without
 *   synchronization, the function could cause system error. No
 *   synchronization is done in the function. To ensure that no system error
 *   oclwrs, the user must ensure that the @a context->waitContextBackEnd, @a
 *   ipcTable->ipcRoute, @a ipcTable->ipcPerm[].ipcRoute and @a
 *   ipcTable->ipcPerm values are not modified during the call to the
 *   function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844557}
 */
void LwSciSyncCoreIpcTableFree(
    LwSciSyncCoreIpcTable* ipcTable);

/**
 * \brief Modifies IPC peer tree to describe a subtree
 *
 * Fetches the sum of permissions from input LwSciSyncCoreIpcTable's
 * LwSciSyncCoreAttrIpcPerm by matching ipcPerm's ipcRoute to the input
 * LwSciIpcEndpoint.
 * The output permissions are set to 0 if input LwSciIpcEndpoint
 * does not match any entry in ipcPerm's ipcRoute.
 * Removes the last entry from ipcPerm's ipcRoute if it matches input
 * LwSciIpcEndpoint, else removes the entire ipcRoute from corresponding ipcPerm.
 * This call should be made only once per input LwSciSyncCoreIpcTable.
 *
 * \param[in,out] ipcTable LwSciSyncCoreIpcTable which holds the tree to be
 * modified
 * \param[in] ipcEndpoint determines the subtree
 * \param[out] needCpuAccess holds sum of cpu access permission of the new tree
 * \param[out] waiterRequireTimestamps holds sum of waiterRequireTimestamps of the
 * node corresponding to the ipcEndpoint.
 * \param[out] requiredPerm holds sum of LwSciSyncAccessPerm of the new tree
 * \return void
 * \param[in] engineArray holds the LwSciSyncHwEngine array of the new tree
 * \param[in] bufLen Number of LwSciSyncHwEngine entries the provided engineArray
 * buffer can hold
 * \param[out] engineArrayLen length of the LwSciSyncHwEngine array of the new
 * tree
 * - Panics if any of the following oclwrs:
 *      - @a ipcTable is NULL
 *      - @a needCpuAccess is NULL
 *      - @a requiredPerm is NULL
 *      - called on same input ipcTable more than once.
 *      - @a engineArray is NULL
 *      - @a engineArrayLen is NULL
 *      - @a bufLen is not sufficient to store the array of LwSciSyncHwEngine(s)
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Decreasing of the @a ipcTable->ipcPerm[].ipcRouteEntries value requires
 *   thread synchronization; without synchronization, the function could cause
 *   a call to LwSciCommonPanic() when called in parallel from other threads.
 *   No synchronization is done in the function. To ensure that there is no
 *   call to LwSciCommonPanic(), the user must ensure that the the function is
 *   not called from other threads in parallel.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844560}
 */
void LwSciSyncCoreIpcTableLwtSubTree(
    LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    bool* needCpuAccess,
    bool* waiterRequireTimestamps,
    LwSciSyncAccessPerm* requiredPerm,
    LwSciSyncHwEngine* engineArray,
    size_t bufLen,
    size_t* engineArrayLen);

/**
 * \brief Get the aggregate of waiterRequireTimestamps of a subtree in the input
 * IPC peer tree
 *
 * The subtree is rooted in the node corresponding to
 * the provided LwSciIpcEndpoint which is a child of the current node.
 *
 * \param[in] ipcTable LwSciSyncCoreIpcTable which contains the subtree
 * \param[in] ipcEndpoint determines the subtree
 * \param[out] waiterRequireTimestamps aggregate of waiterRequireTimestamps of
 * all the nodes in the subtree.
 * \return void
 * - Panics if @a ipcTable is NULL or @a waiterRequireTimestamps is NULL.
 */
void LwSciSyncCoreIpcTableGetRequireTimestampsAtSubTree(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    bool* waiterRequireTimestamps);

/**
 * \brief Get the aggregate of waiterRequireTimestamps in the IPC table
 *
 * \param[in] ipcTable LwSciSyncCoreIpcTable which contains the subtree
 * \param[out] waiterRequireTimestamps aggregate of waiterRequireTimestamps of
 * all the nodes in the subtree.
 * \return void
 * - Panics if @a ipcTable is NULL or @a waiterRequireTimestamps is NULL.
 */
void LwSciSyncCoreIpcTableGetRequireTimestampsSum(
    const LwSciSyncCoreIpcTable* ipcTable,
    bool* waiterRequireTimestamps);

/**
 * \brief Gets sum of LwSciSyncAccessPerm from input ipcTable's ipcPerm for which
 * its ipcRoute's last entry matches input ipcEndpoint.
 *
 * \param[in] ipcTable input LwSciSyncCoreIpcTable
 * \param[in] ipcEndpoint LwSciIpcEndpoint for which sum of permissions is
 * required.
 * \param[out] perm Sum of the LwSciSyncAccessPerm of all the nodes matching input
 * ipcEndpoint.
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if the tree has no valid permissions for input
 *   LwSciIpcEndpoint.
 * - Panics if @a ipcTable is NULL or @a perm is NULL or empty ipc perm branch
 *   in ipcTable.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the @a ipcTable->ipcPerm[].ipcRouteEntries values requires
 *   thread synchronization; without synchronization, the function could cause
 *   a call to LwSciCommonPanic() when called in parallel with a function
 *   which modifies the mentioned value. No synchronization is done in the
 *   function. To ensure that there is no call to LwSciCommonPanic(), the user
 *   must ensure that the the function is not called in parallel with a
 *   function which modifies the mentioned value.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844563}
 */
LwSciError LwSciSyncCoreIpcTableGetPermAtSubTree(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAccessPerm* perm);

/**
 * \brief Gets LwSciSyncHwEngine array from all ipcPerm in the input ipcTable
 * for which the ipcRoute's last entries matches the input ipcEndpoint.
 *
 * \param[in] ipcTable input LwSciSyncCoreIpcTable
 * \param[in] ipcEndpoint LwSciIpcEndpoint for which sum of permissions is
 * required.
 * \param[out] engineArray LwSciSyncHwEngine array at the subtree identified by
 * the input LwSciIpcEndpoint
 * \param[in] bufLen Number of LwSciSyncHwEngine entries engineArray is capable
 * of holding
 * \param[out] engineArrayLen Number of LwSciSyncHwEngine entries written to
 * engineArray
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - Panics if any of the following oclwrs:
 *      - @a ipcTable is NULL
 *      - @a engineArray is NULL
 *      - @a engineArrayLen is NULL
 *      - if there is not enough space in @a engineArray to store the
 *        LwSciSyncHwEngine entries
 *
 * \implements{22837637}
 */
LwSciError LwSciSyncCoreIpcTableGetEngineArrayAtSubTree(
    const LwSciSyncCoreIpcTable* ipcTable,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncHwEngine* engineArray,
    size_t bufLen,
    size_t* engineArrayLen);

/**
 * \brief Checks if the ipcRoute in the input LwSciSyncCoreIpcTable is empty
 *
 * \param[in] ipcTable LwSciSyncCoreIpcTable to check
 * \return bool
 * - true if the IPC Route is empty
 * - false if the IPC Route in not empty
 * - Panics if @a ipcTable is NULL.
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
 * \implements{18844566}
 */
bool LwSciSyncCoreIpcTableRouteIsEmpty(
    const LwSciSyncCoreIpcTable* ipcTable);

/**
 * \brief Imports an LwSciSyncCoreIpcTable from a descriptor buffer.
 * The descriptor buffer is read using LwSciCommonTransportGetRxBufferAndParams().
 * Individual LwSciSyncCoreIpcTable Transport Key-Value Pairs are obtained by
 * calling LwSciCommonTransportGetNextKeyValuePair(). The number of times each
 * Transport Key is processed is also tracked, and if this doesn't match the
 * expected number of times the Transport Key should be seen, then an error is
 * returned.
 *
 * \param[in,out] ipcTable Imported LwSciSyncCoreIpcTable
 * \param[in] desc LwSciSyncCoreIpcTable descriptor
 * Valid value: desc is valid if it is not NULL
 * \param[in] size size of the descriptor
 * Valid value: [1, SIZE_MAX]
 * \param[in] importReconciled Whether this is importing a Reconciled or
 * Unreconciled LwSciSyncAttrList.
 * Valid value: importReconciled is valid if it is either true or false.
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is not enough memory
 * - LwSciError_BadParameter if the descriptor is invalid or size is 0
 * - LwSciError_Overflow if too many IPC paths are imported into the IPC Table
 *   or a path is so long that its size or size of accompanying topoIds
 *   cannot be stored in a uint64_t
 * - Panics if @a ipcTable is NULL or @a desc is NULL.
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the dereferenced @a desc values requires thread synchronization;
 *   without synchronization, the function could cause a system error when
 *   called in parallel with a function which modifies the mentioned value. No
 *   synchronization is done in the function. To ensure that there no system
 *   error oclwrs, the user must ensure that the the function is not called in
 *   parallel with a function which modifies the mentioned value.
 * - Access to the dereferenced @a ipcTable value requires thread
 *   synchronization; without synchronization, the function could cause
 *   incorrect or invalid LwSciSyncCoreIpcTable to be imported, when called in
 *   parallel with a function which modifies the mentioned value. No
 *   synchronization is done in the function. To ensure that correct
 *   LwSciSyncCoreIpcTable is imported, the user must ensure that the the
 *   function is not called in parallel with a function which modifies the
 *   mentioned value.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844569}
 */
LwSciError LwSciSyncCoreImportIpcTable(
    LwSciSyncCoreIpcTable* ipcTable,
    const void* desc,
    size_t size,
    bool importReconciled);

/**
 * \brief Export LwSciSyncCoreIpcTable to a transport descriptor.
 * A transport buffer is created using LwSciCommonTransportAllocTxBufferForKeys().
 * The individual LwSciSyncCoreIpcTable Transport Keys are appended using
 * LwSciCommonTransportAppendKeyValuePair() and the final
 * buffer is prepared using LwSciCommonTransportPrepareBufferForTx().
 *
 * \param[in] ipcTable LwSciSyncCoreIpcTable to export
 * \param[out] txbufPtr The Export Descriptor containing the serialized
 * LwSciSyncCoreIpcTable
 * \param[out] txbufSize Size of the serialized LwSciSyncCoreIpcTable Export Descriptor
 * \return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if there is not enough memory
 * - LwSciError_NoSpace if no space is left in transport buffer to append the
 *      key-value pair.
 * - LwSciError_Overflow if too many IPC paths are exported to the export descriptor
 * - Panics if @a ipcTable is NULL or @a txbufPtr is NULL or @a txbufSize is NULL
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Access to the dereferenced @a ipcTable values requires thread
 *   synchronization; without synchronization, the function could cause a call
 *   to LwSciCommonPanic() when called in parallel with a function which
 *   modifies the mentioned value. No synchronization is done in the function.
 *   To ensure that no system error oclwrs, the user must ensure that the the
 *   function is not called in parallel with a function which modifies the
 *   mentioned value.
 * - Access to the dereferenced @a txbufPtr value requires thread
 *   synchronization; without synchronization, the function could cause a
 *   system error, when called in parallel with a function which modifies the
 *   mentioned value. No synchronization is done in the function. To ensure
 *   that no system error oclwrs, the user must ensure that the the function
 *   is not called in parallel with a function which modifies the mentioned
 *   value.
 *
 * \implements{18844572}
 */
LwSciError LwSciSyncCoreExportIpcTable(
    const LwSciSyncCoreIpcTable* ipcTable,
    void** txbufPtr,
    size_t* txbufSize);

/**
 * \brief Check whether the input LwSciSyncCoreIpcTable's
 * ipcRoute and topoIds  or any of the ipcPerm paths contain
 * a C2C LwSciIpcEndpoint.
 *
 * \param[in] ipcTable LwSciSyncCoreIpcTable
 * \return bool
 * - true if the ipcRoute and topoIds contain a C2C LwSciIpcEndpoint
 *   or the ipcRoute and topoIds of any entry in ipcPerm contain
 *   a C2C LwSciIpcEndpoint
 * - false otherwise
 * - Panics if @a ipcTable is NULL
 *
 * \implements{TBD}
 */
bool LwSciSyncCoreIpcTableHasC2C(
    const LwSciSyncCoreIpcTable* ipcTable);
 /** @} */
#endif
