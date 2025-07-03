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
 * \brief <b>LwSciSync Fence Management Interface</b>
 *
 * @b Description: This file contains LwSciSync fence management core
 * structures and interfaces.
 */

#ifndef INCLUDED_LWSCISYNC_FENCE_H
#define INCLUDED_LWSCISYNC_FENCE_H

/**
 * @defgroup lwsci_sync Synchronization APIs
 * @{
 */

/**
 * If the input LwSciSyncFence was not cleared, this function
 * validates the LwSciSyncObj associated with it
 * and drops a reference on it using LwSciSyncObjFreeObjAndRef().
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciSyncObj associated with the
 *        LwSciSyncFence is provided via LwSciSyncObjFreeObjAndRef()
 *      - The user must ensure that the same LwSciSyncFence is not modified by
 *        multiple threads at the same time
 *
 * \implements{18844485}
 *
 * \fn void LwSciSyncFenceClear(
 *    LwSciSyncFence* syncFence);
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciSyncObj associated with the input
 *        @a srcSyncFence is provided via LwSciSyncObjRef()
 *      - The user must ensure that the same LwSciSyncFence to be updated is
 *        not used by multiple threads at the same time, since calling
 *        LwSciSyncFenceClear() from multiple threads on the same
 *        LwSciSyncFence is unsafe
 *      - The user must ensure that the same output @a dstSyncFence parameter
 *        is not used by multiple threads at the same time
 *
 * \implements{18844488}
 *
 * \fn LwSciError LwSciSyncFenceDup(
 *    const LwSciSyncFence* srcSyncFence,
 *    LwSciSyncFence* dstSyncFence);
 */

/**
 * The function retrieves LwSciSyncModule from the LwSciSyncObj associated
 * with the input LwSciSyncFence and from the input LwSciSyncCpuWaitContext
 * and verifies it is the same module using LwSciSyncCoreModuleIsDup().
 * Checks if the LwSciSyncObj associated with input LwSciSyncFence has
 * cpu waiting permissions using LwSciSyncCoreAttrListTypeIsCpuWaiter() on
 * the LwSciSyncAttrList retrieved with LwSciSyncObjGetAttrList() from the
 * LwSciSyncObj.
 * Performs the wait by calling LwSciSyncCoreWaitOnPrimitive()
 * on the LwSciSyncCorePrimitive associated
 * with the LwSciSyncObj retrieved with
 * LwSciSyncCoreObjGetPrimitive() using the primitive ID and value extracted
 * from the LwSciSyncFence.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciSyncFence is
 *        never modified after an LwSciSyncObj was associated (so there is no
 *        data-dependency)
 *      - Reads only occur from immutable data since the
 *        LwSciSyncCpuWaitContext is never modified after creation (so there is
 *        no data-dependency)
 *
 * \implements{18844491}
 *
 * \fn LwSciError LwSciSyncFenceWait(
 *    const LwSciSyncFence* syncFence,
 *    LwSciSyncCpuWaitContext context,
 *    int64_t timeoutUs);
 */

/**
 * The function validates the input ipcEndpoint with
 * LwSciSyncCoreValidateIpcEndpoint.
 * If the input LwSciSyncFence is cleared, it fills desc with zeroes
 * and returns.
 * Otherwise, it retrieves objId from the LwSciSyncObj
 * using LwSciSyncCoreObjGetId.
 * Figures out the first LwSciIpcEndpoint the LwSciSyncFence
 * was transported through (potentially this is the first transport).
 * Writes this objId updated with the first LwSciIpcEndpoint,
 * fence's id, and value
 * to the descriptor accordingly to the LwSciSyncFenceIpcExportDescriptor's
 * layout.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciSyncFence is
 *        never modified after it was generated or updated (so there is no
 *        data-dependency)
 *
 * \implements{18844494}
 *
 * \fn LwSciError LwSciSyncIpcExportFence(
 *    const LwSciSyncFence* syncFence,
 *    LwSciIpcEndpoint ipcEndpoint,
 *    LwSciSyncFenceIpcExportDescriptor* desc);
 */

/**
 * Copies fence's id and value from
 * the descriptor to the LwSciSyncFence according to the LwSciSyncFence's
 * and LwSciSyncFenceIpcExportDescriptor's layouts. Updates the value
 * in the imported fence by adding the initial known value of the primitive.
 * If the LwSciSyncFenceIpcExportDescriptor represents a not cleared fence,
 * the function verifies that the input
 * LwSciSyncObj has the same objId as the input descriptor using
 * LwSciSyncCoreObjMatchId.
 * The fence's value is imported with LwSciSyncCoreObjImportThreshold().
 * The association of the LwSciSyncFence with the input LwSciSyncObj
 * is done with LwSciSyncObjRef.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciSyncObj is handled via
 *        LwSciSyncObjRef()
 *      - The user must ensure that the same LwSciSyncFence to be updated is
 *        not used by multiple threads at the same time, since calling
 *        LwSciSyncFenceClear() from multiple threads on the same
 *        LwSciSyncFence is unsafe
 *
 * \implements{18844497}
 *
 * \fn LwSciError LwSciSyncIpcImportFence(
 *    LwSciSyncObj syncObj,
 *    const LwSciSyncFenceIpcExportDescriptor* desc,
 *    LwSciSyncFence* syncFence);
 */

/**
 * This function directly writes id and value to output parameters based
 * on the LwSciSyncFence's memory layout.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the fence ID is never
 *        modified after it was generated or updated (so there is no
 *        data-dependency)
 *      - Reads only occur from immutable data since the fence value is never
 *        modified after it was generated or updated (so there is no
 *        data-dependency)
 *
 * \implements{18844503}
 *
 * \fn LwSciError LwSciSyncFenceExtractFence(
 *    const LwSciSyncFence* syncFence,
 *    uint64_t* id,
 *    uint64_t* value);
 */

/**
 * The function first clears the LwSciSyncFence.
 * Takes a reference on the input syncObj with LwSciSyncObjRef and
 * updates LwSciSyncFence's id, value and syncObj fields according to the layout.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access to the LwSciSyncObj is provided via
 *        LwSciSyncObjRef()
 *      - The user must ensure that the same LwSciSyncFence to be updated is
 *        not used by multiple threads at the same time, since calling
 *        LwSciSyncFenceClear() from multiple threads on the same
 *        LwSciSyncFence is unsafe
 *
 * \implements{18844500}
 *
 * \fn LwSciError LwSciSyncFenceUpdateFence(
 *    LwSciSyncObj syncObj,
 *    uint64_t id,
 *    uint64_t value,
 *    LwSciSyncFence* syncFence);
 */

/**
 * This function returns the pointer from the underlying LwSciSyncFence's structure
 * according to LwSciSyncFence's layout.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciSyncObj
 *        associated with the LwSciSyncFence is never modified after it was
 *        initially associated (so there is no data-dependency)
 *
 * \implements{18844506}
 *
 * \fn LwSciError LwSciSyncFenceGetSyncObj(
 *    const LwSciSyncFence* syncFence,
 *    LwSciSyncObj* syncObj);
 */

 /** @} */
#endif
