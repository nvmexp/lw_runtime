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
 * \brief <b>LwSciSync Object Management Interface</b>
 *
 * @b Description: This file contains LwSciSync object management core
 * structures and interfaces.
 */

#ifndef INCLUDED_LWSCISYNC_OBJECT_CORE_H
#define INCLUDED_LWSCISYNC_OBJECT_CORE_H

#include <stdint.h>

#include "lwscicommon_objref.h"
#include "lwsciipc.h"
#include "lwscisync.h"
#include "lwscisync_internal.h"
#include "lwscisync_primitive.h"
#include "lwscisync_timestamps.h"

/**
 * \brief A unique identifier used to identify an LwSciSyncObj
 *
 * \implements{18845814}
 */
typedef struct {
    /** Unique identifier received from the associated LwSciSyncModule.
     * This value is initialized by LwSciSyncCoreModuleCntrGetNextValue
     * during initialization of the LwSciSyncCoreObj in a monotonically
     * increasing fashion.
     * Valid Value: [0, UINT64_MAX]
     */
    uint64_t moduleCntr;
    /** The IPC Endpoint through which the LwSciSyncObj was first exported
     * through.
     *
     * This value is initialized to 0 if the LwSciSyncObj has not been
     * exported. After the LwSciSyncObj has been exported for the first time,
     * this represents the original endpoint the LwSciSyncObj was exported
     * through.
     */
    LwSciIpcEndpoint ipcEndpoint;
} LwSciSyncCoreObjId;

/**
 * \brief LwSciSync object reference structure.
 *
 * A LwSciSyncObjRec passed as an
 * input parameter to an API is valid input if it is returned from a successful
 * call to: LwSciSyncIpcImportAttrListAndObj, LwSciSyncObjAlloc,
 * LwSciSyncObjDup, LwSciSyncObjIpcImport or
 * LwSciSyncAttrListReconcileAndObjAlloc
 * and has not yet been deallocated using LwSciSyncObjFree.
 *
 * This structure is allocated and deallocated using LwSciCommon functionality.
 * An LwSciSyncObjRec holds a reference to an LwSciRef structure, which
 * contains the actual object resource data.
 *
 * \implements{18845817}
 */
struct LwSciSyncObjRec {
    /** Reference to the core object */
    LwSciRef refObj;
};

/**
 * \brief Validates an LwSciSyncObj by comparing a newly computed header
 * value of the given LwSciSyncObj against the previously computed header.
 *
 * Retrieves the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef.
 *
 * \param[in] syncObj LwSciSyncObj to validate.
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @p syncObj is NULL
 * - Panics if @a syncObj is invalid
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844662}
 */
LwSciError LwSciSyncCoreObjValidate(
    LwSciSyncObj syncObj);

/**
 * \brief Gets the LwSciSyncCoreObjId identifying the given LwSciSyncObj.
 *
 * Retrieves the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef.
 *
 * \param[in] syncObj The LwSciSyncObj to get the Object ID of.
 * \param[out] objId The LwSciSyncObj ID.
 *
 * \return void
 * - Panics if:
 *      - any argument is NULL
 *      - @a syncObj is invalid
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844665}
 */
void LwSciSyncCoreObjGetId(
    LwSciSyncObj syncObj,
    LwSciSyncCoreObjId* objId);

/**
 * \brief Compares the given LwSciSyncCoreObjId against the LwSciSyncCoreObjId
 * of the given LwSciSyncObj to check whether they refer to the same
 * underlying Synchronization Object. Two LwSciSyncCoreObjId are considered
 * equal if they point to the same IPC Endpoint and have the same
 * LwSciSyncModule Counter value.
 *
 * Retrieves the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef.
 *
 * \param[in] syncObj LwSciSyncObj to compare against.
 * \param[in] objId The given LwSciSyncObj ID.
 * \param[out] isEqual Boolean result of the equality check.
 *
 * \return void
 * - Panics if:
 *      - any argument is NULL
 *      - @a syncObj is invalid
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844668}
 */
void LwSciSyncCoreObjMatchId(
    LwSciSyncObj syncObj,
    const LwSciSyncCoreObjId* objId,
    bool* isEqual);

/**
 * \brief Retrieves the LwSciSyncModule associated with the given LwSciSyncObj
 * using LwSciSyncCoreAttrListGetModule().
 *
 * Retrieves the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef.
 *
 * \param[in] syncObj LwSciSyncObj to retrieve the associated Module from.
 * \param[out] module The LwSciSyncModule associated with the given LwSciSyncObj.
 *
 * \return void
 * - Panics if:
 *      - any argument is NULL
 *      - @a syncObj is invalid
 *      - Attribute List associated with the LwSciSyncObj is invalid
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844671}
 */
void LwSciSyncCoreObjGetModule(
    LwSciSyncObj syncObj,
    LwSciSyncModule* module);

/**
 * \brief Retrieves timestamps structure owned by this object
 *
 * \param[in] syncObj object of type LwSciSyncObj
 * \param[out] timestamps structure owned by syncObj
 *
 * \return void
 * - Panics if @a syncObj is invalid
 * - Panics if @a timestamps is NULL
 *
 */
void LwSciSyncCoreObjGetTimestamps(
    LwSciSyncObj syncObj,
    LwSciSyncCoreTimestamps* timestamps);

/**
 * \brief Retrieves the underlying LwSciSyncCorePrimitive that the given
 * LwSciSyncObj is associated with.
 *
 * Retrieves the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef.
 *
 * \param[in] syncObj LwSciSyncObj to fetch the associated primitive from.
 * \param[out] primitive Primitive owned by @a syncObj
 *
 * \return void
 * - Panics if:
 *      - any argument is NULL
 *      - @a syncObj is invalid
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844674}
 */
void LwSciSyncCoreObjGetPrimitive(
    LwSciSyncObj syncObj,
    LwSciSyncCorePrimitive* primitive);

/**
 * \brief Frees the underlying object and reference to the LwSciSyncObj
 * using LwSciCommonFreeObjAndRef(), passing LwSciSyncCoreObjClose to it.
 * The associated LwSciSyncAttrList is freed
 * once all references to the underlying LwSciSyncCoreObj are released. The
 * underlying primitive will also be freed if it is managed by LwSciSync.
 *
 * \param[in] syncObj LwSciSyncObj to be freed.
 *
 * \return void
 * - Panics if @a syncObj is invalid or NULL
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
 * \implements{18844677}
 */
void LwSciSyncObjFreeObjAndRef(
    LwSciSyncObj syncObj);

/**
 * \brief Callback to free the data associated with the LwSciObj representing
 * the underlying LwSciSyncCoreObj using LwSciCommon functionality.
 *
 * \param[in] objPtr Pointer to the LwSciObj associated with the LwSciSyncObj
 * to free
 *
 * \return void
 * - Panics if objPtr is NULL or invalid
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - Acces to the object pointed to by @a objPtr value requires thread
 *   synchronization; without synchronization, the function could cause a
 *   system error when called in parallel with a function which modifies the
 *   mentioned object. No synchronization is done in the function. To ensure
 *   that no system error oclwrs, the user must ensure that the object is not
 *   modified.
 * - Deinitialization of the primitive and the attribute list associated with
 *   the object pointed to by @a objPtr value requires thread synchronization;
 *   without synchronization, the function could cause a system error when
 *   called in parallel with a function which modifies the mentioned object.
 *   No synchronization is done in the function. To ensure that no system
 *   error oclwrs, the user must ensure that the object is not modified.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{22034748}
 */
void LwSciSyncCoreObjClose(
    LwSciObj* objPtr);

/**
 * \brief Translates imported fence threshold to be locally-based
 *
 * Retreives the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef().
 * Imports the threshold using LwSciSyncCorePrimitiveImportThreshold().
 *
 * \param[in] syncObj contains info about the local primitive
 * \param[inout] threshold threshold to be translated
 *
 * \return LwSciError
 * - LwSciError_Success if successfully retrieved the syncHandle
 * - Panics if @a threshold is NULL or syncObj is NULL or invalid
 *
 *  \implements{TODO}
 */
LwSciError LwSciSyncCoreObjImportThreshold(
    LwSciSyncObj syncObj,
    uint64_t* threshold);

/**
 * Retrieves the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef.
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844656}
 *
 * \fn LwSciError LwSciSyncObjGetAttrList(
 *     LwSciSyncObj syncObj,
 *     LwSciSyncAttrList* syncAttrList);
 */

/**
 * Retrieves the underlying LwSciSyncCoreObj with LwSciCommonGetObjFromRef.
 * The additional references on both the underlying reference and object are
 * taken using LwSciCommonIncrAllRefCounts().
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - Acces to the object pointed to by @a syncObj value requires thread
 *   synchronization; without synchronization, the function could cause a
 *   system error when called in parallel with a function which modifies the
 *   mentioned value. No synchronization is done in the function. To ensure
 *   that no system error oclwrs, the user must ensure that the object is not
 *   modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned LwSciObj.objLock is locked immediately before and
 *   released immediately after LwSciObj.refCount is incremented.
 *
 * \implements{18844653}
 *
 * \fn LwSciError LwSciSyncObjRef(
 *     LwSciSyncObj syncObj);
 */
#endif
