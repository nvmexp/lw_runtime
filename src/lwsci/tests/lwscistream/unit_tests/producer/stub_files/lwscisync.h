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
 * @file
 *
 * @brief <b> LWPU Software Communications Interface (SCI) : LwSciSync </b>
 *
 * The LwSciSync library allows applications to manage synchronization
 * objects which coordinate when sequences of operations begin and end.
 */
#ifndef INCLUDED_LWSCISYNC_H
#define INCLUDED_LWSCISYNC_H

/**
 * @defgroup lwsci_sync Synchronization APIs
 *
 * @ingroup lwsci_group_stream
 * @{
 *
 * The LwSciSync library allows applications to manage synchronization
 * objects which coordinate when sequences of operations begin and end.
 *
 * The following constants are defined and have type @c unsigned @c int:
 * * @c LwSciSyncMajorVersion
 * * @c LwSciSyncMinorVersion
 *
 * In C and C++ these constants are guaranteed to be defined as global
 * const variables.

 * Upon each new release of LwSciSync:
 * * If the new release changes behavior of LwSciSync that could in any way
 *   prevent an application compliant with previous versions of this
 *   specification from functioning correctly, or that could prevent
 *   interoperability with earlier versions of LwSciSync, the major version is
 *   increased and the minor version is reset to zero.
 *   @note This is typically done only after a deprecation period.
 * * Otherwise, if the new release changes LwSciSync in any application
 *   observable manner (e.g., new features, or bug fixes), the major version is
 *   kept the same and the minor version is increased.
 * * Otherwise, the major and minor version are both kept the same.
 *
 * This version of this specification corresponds to LwSciSync version 1.0
 * (major version 1, minor version 0).
 *
 * Different processes using the LwSciSync inter-process APIs may use different
 * minor versions of LwSciSync within the same major version, provided that if
 * a process uses a feature newer than another processor's LwSciSync version,
 * the latter process does not import an unreconciled LwSciSyncAttrList (directly
 * or indirectly) from the former process.
 *
 * In general, an LwSciSync API call will not return an error code if it has
 * caused any side effects other than allocating resources and subsequently
 * freeing those resources.
 *
 * In general, unless specified otherwise, if a NULL pointer is passed to an
 * LwSciSync API call, the API call will either return ::LwSciError_BadParameter
 * or (if there are other applicable error conditions as well) an error code
 * corresponding to another error.
 *
 * Each LwSciSyncAttrList is either unreconciled or reconciled.
 * It is unreconciled if it was:
 * - Created by LwSciSyncAttrListCreate(),
 * - Created by a failed LwSciSyncAttrListReconcile() call,
 * - Created by a failed LwSciSyncAttrListReconcileAndObjAlloc() call, or
 * - An import of an export of one or more unreconciled LwSciSyncAttrLists.
 *
 * It is reconciled if it was:
 * - Created by a successful LwSciSyncAttrListReconcile() call,
 * - Provided by LwSciSyncObjGetAttrList(), or
 * - An import of an export of another reconciled LwSciSyncAttrList.
 */

/**
 * \page lwscisync_page_blanket_statements LwSciSync blanket statements
 * \section lwscisync_in_out_params Input/Output parameters
 * - LwSciSyncFence passed as input parameter to an API is valid input if
 * it was first initialized to all zeroes
 * or modified by any successful LwSciSync API accepting LwSciSyncFence.
 * - LwSciSyncObj passed as an input parameter to an API is valid input if it
 * is returned from a successful call to LwSciSyncObjAlloc(), LwSciSyncObjDup(),
 * LwSciSyncObjIpcImport(), LwSciSyncIpcImportAttrListAndObj(),
 * LwSciSyncAttrListReconcileAndObjAlloc(), or LwSciSyncFenceGetSyncObj() and
 * has not yet been deallocated using LwSciSyncObjFree().
 * - LwSciSyncCpuWaitContext passed as an input parameter to an API is valid
 * input if it is returned from a successful call to
 * LwSciSyncCpuWaitContextAlloc() and has not been deallocated using
 * LwSciSyncCpuWaitContextFree().
 * - LwSciSyncModule passed as input parameter to an API is valid input if it is
 * returned from a successful call to LwSciSyncModuleOpen() and has not yet
 * been deallocated using LwSciSyncModuleClose().
 * - LwSciIpcEndpoint passed as input parameter to an API is valid if it is
 * obtained from successful call to LwSciIpcOpenEndpoint() and has not yet been
 * freed using LwSciIpcCloseEndpoint().
 * - Unreconciled LwSciSyncAttrList is valid if it is obtained from successful
 * call to LwSciSyncAttrListCreate() or if it is obtained from successful call to
 * LwSciSyncAttrListClone() where input to LwSciSyncAttrListClone() is valid
 * unreconciled LwSciSyncAttrList or if it is obtained from successful call to
 * LwSciSyncAttrListIpcImportUnreconciled() and has not been deallocated using
 * LwSciSyncAttrListFree().
 * - Reconciled LwSciSyncAttrList is valid if it is obtained from successful call
 * to LwSciSyncAttrListReconcile() or if it is obtained from successful call to
 * LwSciSyncAttrListClone() where input to LwSciSyncAttrListClone() is valid
 * reconciled LwSciSyncAttrList or if it is obtained from successful call to
 * LwSciSyncAttrListIpcImportReconciled() and has not been deallocated using
 * LwSciSyncAttrListFree() or has been obtained from a successful call to
 * LwSciSyncObjGetAttrList() and the input LwSciSyncObj to this call has not
 * yet been deallocated using LwSciSyncObjFree().
 * - If the valid range for the input parameter is not explicitly mentioned in
 * the API specification or in the blanket statements then it is considered that
 * the input parameter takes any value from the entire range corresponding to
 * its datatype as the valid value. Please note that this also applies to the
 * members of a structure if the structure is taken as an input parameter.
 *
 * \section lwscisync_out_params Output parameters
 * - In general, output parameters are passed by reference through pointers.
 * Also, since a null pointer cannot be used to convey an output parameter, API
 * functions typically return an error code if a null pointer is supplied for a
 * required output parameter unless otherwise stated explicitly. Output
 * parameter is valid only if error code returned by an API is
 * LwSciError_Success unless otherwise stated explicitly.
 *
 * \section lwscisync_conlwrrency Conlwrrency
 * - Every individual function can be called conlwrrently with itself
 * without any side-effects unless otherwise stated explicitly in
 * the interface specifications.
 * - The conditions for combinations of functions that cannot be called
 * conlwrrently or calling them conlwrrently leads to side effects are
 * explicitly stated in the interface specifications.
 *
 * \section lwscisync_fence_states Fence states
 * - A zero initialized LwSciSyncFence or one fed to LwSciSyncFenceClear()
 *   becomes cleared.
 * - LwSciSyncFence becomes not cleared if it is modified by a successful
 * LwSciSyncObjGenerateFence() or LwSciSyncFenceDup().
 * - LwSciSyncFence filled by successful LwSciSyncIpcImportFence() is cleared
 * if and only if the input fence descriptor was created from a cleared
 * LwSciSyncFence.
 *
 * \implements{18839709}
 */

#if !defined (__cplusplus)
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#endif

#include <lwscierror.h>
#include <lwsciipc.h>

#if defined (__cplusplus)
extern "C"
{
#endif

/**
 * \brief LwSciSync major version number.
 *
 * \implements{18840177}
 */
static const uint32_t LwSciSyncMajorVersion = 2U;

/**
 * \brief LwSciSync minor version number.
 *
 * \implements{18840180}
 */
static const uint32_t LwSciSyncMinorVersion = 2U;

/**
 * Maximum supported timeout value.
 * LwSciSyncFenceWait() can wait for at most this many microseconds.
 * This value also corresponds to infinite timeout.
 */
static const int64_t LwSciSyncFenceMaxTimeout = (0x7fffffffffffffff / 1000);

/**
 * \brief Represents an instance of the LwSciSyncModule. Any LwSciSyncAttrList
 * created or imported using a particular LwSciSyncModule is bound to that
 * module instance, along with any LwSciSyncObjs created or imported using those
 * LwSciSyncAttrLists and any LwSciSyncFences created or imported using those
 * LwSciSyncObjs.
 *
 * @note For any LwSciSync API call that has more than one input of type
 * LwSciSyncModule, LwSciSyncAttrList, LwSciSyncObj, and/or LwSciSyncFence, all
 * such inputs must agree on the LwSciSyncModule instance.
 */
typedef struct LwSciSyncModuleRec* LwSciSyncModule;

/**
 * \brief Represents the right to perform a CPU wait on an LwSciSyncFence. It
 * holds resources necessary to perform a CPU wait using LwSciSyncFenceWait().
 * It can be used to wait on LwSciSyncFence(s) associated with the same
 * LwSciSyncModule that this LwSciSyncCpuWaitContext is associated with.
 * An LwSciSyncCpuWaitContext can be used to wait on only one
 * LwSciSyncFence at a time. However, a single LwSciSyncCpuWaitContext
 * can be used to wait on different LwSciSyncFences at different
 * instants of time.
 */
typedef struct LwSciSyncCpuWaitContextRec* LwSciSyncCpuWaitContext;

/**
 * \brief Defines the opaque LwSciSyncFence.
 *
 * This structure describes a synchronization primitive snapshot
 * that must be reached to release the waiters.
 *
 * Unlike LwSciSyncAttrList and LwSciSyncObj objects, applications are
 * responsible for the memory management of LwSciSyncFences.
 *
 * Every LwSciSyncFence must be initialized to all zeros after its storage
 * is allocated before the first time it is passed to any LwSciSync API
 * calls. LwSciSyncFenceInitializer can be used for this, or
 * memset(), or any other mechanism for zeroing memory.
 *
 * Every LwSciSyncFence not in a cleared state holds a reference to
 * the LwSciSyncObj it is related to, preventing that LwSciSyncObj
 * from being deleted. It also contains
 * id and value of the synchronization primitive corresponding
 * to the desired snapshot.
 *
 * In case if the corresponding LwSciSyncObj supports timestamps,
 * this structure also contains information about the memory location
 * of the timestamp of the event unblocking the LwSciSyncFence.
 */
/**
 * \implements{18840156}
 */
typedef struct {
    uint64_t payload[6];
} LwSciSyncFence;

/**
 * \brief Defines the value used to zero-initialize the LwSciSyncFence object.
 *  An LwSciSyncFence that is all zeroes is in a cleared state.
 *
 * \implements{18840183}
 */
static const LwSciSyncFence LwSciSyncFenceInitializer = {{0U}};

/**
 * \brief Defines the exported form of LwSciSyncFence intended to be shared
 * across an LwSciIpc channel.
 *
 * \implements{18840195}
 */
typedef struct {
    uint64_t payload[7];
} LwSciSyncFenceIpcExportDescriptor;

/**
 * \brief Defines the exported form of LwSciSyncObj intended to be shared
 * across an LwSciIpc channel.
 *
 * \implements{18840198}
 */
typedef struct {
    /** Exported data (blob) for LwSciSyncObj */
    uint64_t payload[128];
} LwSciSyncObjIpcExportDescriptor;

/**
 * A Synchronization Object is a container holding the reconciled
 * LwSciSyncAttrList defining constraints of the Fence and the handle of the
 * actual Primitive, with access permissions being enforced by the
 * LwSciSyncAttrKey_RequiredPerm and LwSciSyncAttrKey_NeedCpuAccess Attribute
 * Keys.
 *
 * If Timestamps have been requested prior to Reconciliation via the
 * LwSciSyncAttrKey_WaiterRequireTimestamps key, this will also hold the
 * Timestamp Buffer allocated by LwSciBuf.
 */

/**
 * \brief A reference to a particular Synchronization Object.
 *
 * @note Every LwSciSyncObj that has been created but not freed
 * holds a reference to the LwSciSyncModule, preventing the module
 * from being de-initialized.
 */
typedef struct LwSciSyncObjRec* LwSciSyncObj;

/**
 * \brief A reference, that is not modifiable,
 *  to a particular Synchronization Object.
 */
typedef const struct LwSciSyncObjRec* LwSciSyncObjConst;

/**
 * \brief A container constituting an LwSciSyncAttrList which contains:
 * - set of LwSciSyncAttrKey attributes defining synchronization object
 *   constraints
 * - slot count defining number of slots in an LwSciSyncAttrList
 * - flag specifying if LwSciSyncAttrList is reconciled or unreconciled.
 *
 * @note Every LwSciSyncAttrList that has been created but not freed
 * holds a reference to the LwSciSyncModule, preventing the module
 * from being de-initialized.
 */
typedef struct LwSciSyncAttrListRec* LwSciSyncAttrList;

/**
 * \brief Describes LwSciSyncObj access permissions.
 *
 * \implements{18840171}
 */
typedef enum {
    /**
     * This represents the capability to wait on an LwSciSyncObj as it
     * progresses through points on its sync timeline.
     */
    LwSciSyncAccessPerm_WaitOnly = (uint64_t)1U << 0U,
    /**
     * This represents the capability to advance an LwSciSyncObj to its
     * next point on its sync timeline.
     */
    LwSciSyncAccessPerm_SignalOnly = (uint64_t)1U << 1U,
     /**
      * This represents the capability to advance an LwSciSyncObj to its
      * next point on its sync timeline and also wait until that next point is
      * reached.
      */
    LwSciSyncAccessPerm_WaitSignal = LwSciSyncAccessPerm_WaitOnly | LwSciSyncAccessPerm_SignalOnly,
    /**
     * Usage of Auto permissions is restricted only for export/import APIs and
     * shouldn't be used as valid value for LwSciSyncAttrKey_RequiredPerm
     * Attribute.
     */
    LwSciSyncAccessPerm_Auto = (uint64_t)1U << 63U,
} LwSciSyncAccessPerm;

/**
 * \brief Describes the LwSciSync public attribute keys holding the
 * corresponding values specifying synchronization object constraints. Input
 * attribute keys specify desired synchronization object constraints and can be
 * set/retrieved from the unreconciled LwSciSyncAttrList using
 * LwSciSyncAttrListSetAttrs()/LwSciSyncAttrListGetAttrs() respectively. Output
 * attribute keys specify actual constraints computed by LwSciSync if
 * reconciliation succeeds. Output attribute keys can be retrieved from a
 * reconciled LwSciSyncAttrList using LwSciSyncAttrListGetAttrs().
 *
 * \implements{18840165}
 */
typedef enum {
    /** Specifies the lower bound - for LwSciSync internal use only. */
    LwSciSyncAttrKey_LowerBound,
    /** (bool, inout) Specifies if CPU access is required.
     *
     * During reconciliation, reconciler sets value of this key to true in the
     * reconciled LwSciSyncAttrList if any of the unreconciled
     * LwSciSyncAttrList(s) ilwolved in reconciliation that is owned by the
     * reconciler has this key set to true, otherwise it is set to false in
     * reconciled LwSciSyncAttrList.
     *
     * When importing the reconciled LwSciSyncAttrList LwSciSync will set the key
     * to OR of values of this key in unreconciled LwSciSyncAttrList(s) relayed by
     * the peer.
     *
     * During validation of reconciled LwSciSyncAttrList against input
     * unreconciled LwSciSyncAttrList(s), validation succeeds if value of this
     * attribute in the reconciled LwSciSyncAttrList is true provided any of the
     * input unreconciled LwSciSyncAttrList(s) owned by the peer set it to
     * true OR if value of this attribute in the reconciled LwSciSyncAttrList is
     * false provided all of the input unreconciled LwSciSyncAttrList(s) owned by
     * the peer set it to false.
     */
    LwSciSyncAttrKey_NeedCpuAccess,
    /**
     * (LwSciSyncAccessPerm, inout) Specifies the required access permissions.
     * If @ref LwSciSyncAttrKey_NeedCpuAccess is true, the CPU will be offered
     * at least these permissions.
     * Any hardware accelerators that contribute to this LwSciSyncAttrList will be
     * offered at least these permissions.
     */
    LwSciSyncAttrKey_RequiredPerm,
    /**
     * (LwSciSyncAccessPerm, out) Actual permission granted after reconciliation.
     * @note This key is read-only.
     *
     * It represents the cumulative permissions of the
     * LwSciSyncAttrKey_RequiredPerm in all LwSciSyncAttrLists being reconciled.
     * The reconciliation fails if any of the following conditions are met:
     * - no LwSciSyncAttrList with LwSciSyncAttrKey_RequiredPerm being set to
     * LwSciSyncAccessPerm_SignalOnly,
     * - more than one LwSciSyncAttrList with LwSciSyncAttrKey_RequiredPerm
     * being set to LwSciSyncAccessPerm_SignalOnly,
     * - no LwSciSyncAttrList with LwSciSyncAttrKey_RequiredPerm
     * being set to LwSciSyncAccessPerm_WaitOnly.
     *
     * If LwSciSyncObj is obtained by calling LwSciSyncObjAlloc(),
     * LwSciSyncAttrKey_ActualPerm is set to LwSciSyncAccessPerm_WaitSignal
     * in the reconciled LwSciSyncAttrList corresponding to it since allocated
     * LwSciSyncObj gets wait-signal permissions by default.
     *
     * For any peer importing the LwSciSyncObj, this key is set in the reconciled
     * LwSciSyncAttrList to the sum of LwSciSyncAttrKey_RequiredPerm requested
     * by the peer and all peers relaying their LwSciSyncAttrList export
     * descriptors via it.
     *
     * During validation of reconciled LwSciSyncAttrList against input
     * unreconciled LwSciSyncAttrList(s), validation succeeds only if
     * LwSciSyncAttrKey_ActualPerm in reconciled is bigger or equal than
     * LwSciSyncAttrKey_RequiredPerm of all the input unreconciled
     * LwSciSyncAttrLists.
     */
    LwSciSyncAttrKey_ActualPerm,
    /**
     * (bool, inout) Importing and then exporting an
     * LwSciSyncFenceIpcExportDescriptor has no side effects and yields an
     * identical LwSciSyncFenceIpcExportDescriptor even if the
     * LwSciIpcEndpoint(s) used for import and export are different from ones
     * used for exporting/importing LwSciSyncAttrList(s).
     *
     * If this attribute key is set to false, this indicates that the
     * LwSciSyncFenceIpcExportDescriptor must be exported through the same IPC
     * path as the LwSciSyncObj. Otherwise if set to true, this indicates that
     * the LwSciSyncFenceIpcExportDescriptor must be exported via LwSciIpc
     * through the first peer that was part of the IPC path travelled through
     * by the LwSciSyncObj (but not necessarily an identical path).
     *
     * During reconciliation, this key is set to true in reconciled
     * LwSciSyncAttrList if any one of the input LwSciSyncAttrList has this set
     * to true.
     */
    LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
    /** (bool, inout) Specifies if timestamps are required. If the waiter
     * wishes to read timestamps then the LwSciSyncAttrKey_NeedCpuAccess key
     * should be set to true.
     *
     * Only valid for CPU waiters. */
    LwSciSyncAttrKey_WaiterRequireTimestamps,
    /**
     * (bool, inout) Specifies if deterministic primitives are required.
     * This allows for the possibility of generating fences on the waiter's
     * side without needing to import them. This means that the Signaler will
     * increment the instance 0 of the primitive in LwSciSyncObj by 1 at each
     * submission.
     *
     * During reconciliation, this key is set to true in the reconciled
     * LwSciSyncAttrList if any one of the input LwSciSyncAttrList(s) has this
     * set to true.
     */
    LwSciSyncAttrKey_RequireDeterministicFences,
    /** Specifies the upper bound - for LwSciSync internal use only. */
    LwSciSyncAttrKey_UpperBound,
} LwSciSyncAttrKey;

/**
 * \brief This structure defines a key/value pair used to get or set
 * the LwSciSyncAttrKey(s) and their corresponding values from or to
 * LwSciSyncAttrList.
 *
 * \implements{18840168}
 */
typedef struct {
    /** LwSciSyncAttrKey for which value needs to be set/retrieved. This member
     * is initialized to any defined value of the LwSciSyncAttrKey other than
     * LwSciSyncAttrKey_LowerBound and LwSciSyncAttrKey_UpperBound */
    LwSciSyncAttrKey attrKey;
    /** Memory which contains the value corresponding to the key */
    const void* value;
    /** Length of the value in bytes */
    size_t len;
} LwSciSyncAttrKeyValuePair;

/**
 * \brief Initializes and returns a new LwSciSyncModule with no
 * LwSciSyncAttrLists, LwSciSyncCpuWaitContexts, LwSciSyncObjs or
 * LwSciSyncFences bound to it.
 *
 * @note A process may call this function multiple times. Each successful
 * invocation will yield a new LwSciSyncModule instance.
 *
 * \param[out] newModule The new LwSciSyncModule.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a newModule is NULL.
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_ResourceError if system drivers are not available or
 *    resources other then memory are unavailable.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncModuleOpen(
    LwSciSyncModule* newModule);

/**
 * \brief Closes an instance of the LwSciSyncModule that was
 * obtained through an earlier call to LwSciSyncModuleOpen(). Once an
 * LwSciSyncModule is closed and all LwSciSyncAttrLists, LwSciSyncObjs,
 * LwSciSyncCpuWaitContexts, LwSciSyncFences bound to that module instance are
 * freed, the LwSciSyncModule instance will be de-initialized in the calling
 * process. Until then the LwSciSyncModule will still be accessible from those
 * objects still referencing it.
 *
 * \note Every owner of the LwSciSyncModule must call LwSciSyncModuleClose()
 * only after all the functions ilwoked by the owner with LwSciSyncModule as an
 * input are completed.
 *
 * \param[in] module The LwSciSyncModule instance to close. The calling process
 * must not pass this module to another LwSciSync API call.
 *
 * \return void
 * - Panics if @a module is invalid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the input
 *        LwSciSyncModule @a module
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciSyncModuleClose(
    LwSciSyncModule module);

/**
 * \brief Allocates a new LwSciSyncCpuWaitContext.
 *
 * \param[in] module LwSciSyncModule instance with which to associate
 *            the new LwSciSyncCpuWaitContext.
 * \param[out] newContext The new LwSciSyncCpuWaitContext.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a module is NULL or @a newContext is NULL.
 * - ::LwSciError_IlwalidState if failed to associate @a module with @a
 *   newContext.
 * - ::LwSciError_InsufficientMemory if not enough system memory to create a
 *   new context.
 * - ::LwSciError_ResourceError if not enough system resources.
 * - Panics if @a module is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
*/
LwSciError LwSciSyncCpuWaitContextAlloc(
    LwSciSyncModule module,
    LwSciSyncCpuWaitContext* newContext);

/**
 * \brief Releases the LwSciSyncCpuWaitContext.
 *
 * \param[in] context LwSciSyncCpuWaitContext to be freed.
 *
 * \return void
 * - Panics if LwSciSyncModule associated with @a context is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the input
 *        LwSciSyncCpuWaitContext @a context
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
*/
void LwSciSyncCpuWaitContextFree(
    LwSciSyncCpuWaitContext context);

/**
 * \brief Creates a new, single-slot unreconciled LwSciSyncAttrList
 * associated with the input LwSciSyncModule with empty LwSciSyncAttrKeys.
 *
 * \param[in] module The LwSciSyncModule instance with which to associate the
 * new LwSciSyncAttrList.
 * \param[out] attrList The new LwSciSyncAttrList.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any argument is NULL.
 * - ::LwSciError_InsufficientMemory if there is insufficient system memory
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 *   to create a LwSciSyncAttrList.
 * - ::LwSciError_IlwalidState if no more references can be taken for
 *     input LwSciSyncModule to create the new LwSciSyncAttrList.
 * - Panics if @a module is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListCreate(
    LwSciSyncModule module,
    LwSciSyncAttrList* attrList);

/**
 * \brief Frees the LwSciSyncAttrList and removes its association with the
 * LwSciSyncModule with which it was created.
 *
 * \param[in] attrList The LwSciSyncAttrList to be freed.
 * \return void
 * - Panics if @a attrList is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the input
 *        LwSciSyncAttrList @a attrList
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciSyncAttrListFree(
    LwSciSyncAttrList attrList);

/**
 * \brief Checks whether the LwSciSyncAttrList is reconciled
 *
 * \param[in] attrList LwSciSyncAttrList to check.
 * \param[out] isReconciled A pointer to a boolean to store whether the
 * @a attrList is reconciled or not.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any argument is NULL.
 * - Panics if @a attrList is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListIsReconciled(
    LwSciSyncAttrList attrList,
    bool* isReconciled);

/**
 * \brief Validates a reconciled LwSciSyncAttrList against a set of input
 * unreconciled LwSciSyncAttrLists.
 *
 * \param[in] reconciledAttrList Reconciled LwSciSyncAttrList to be validated.
 * \param[in] inputUnreconciledAttrListArray Array containing the unreconciled
 * LwSciSyncAttrLists used for validation.
 * Valid value: Array of valid unreconciled LwSciSyncAttrLists
 * \param[in] inputUnreconciledAttrListCount number of elements/indices in
 * @a inputUnreconciledAttrListArray
 * Valid value: [1, SIZE_MAX]
 * \param[out] isReconciledListValid A pointer to a boolean to store whether
 * the reconciled LwSciSyncAttrList satisfies the parameters of set of
 * unreconciled LwSciSyncAttrList(s) or not.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a inputUnreconciledAttrListArray is NULL
 *         - @a inputUnreconciledAttrListCount is 0
 *         - @a isReconciledListValid is NULL
 *         - any of the input LwSciSyncAttrLists in
 *           inputUnreconciledAttrListArray are not unreconciled
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
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListValidateReconciled(
    LwSciSyncAttrList reconciledAttrList,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool* isReconciledListValid);

/**
 * \brief Sets the values for LwSciSyncAttrKey(s) in slot 0 of the
 * input LwSciSyncAttrList.
 *
 * Reads values only during the call, saving copies. Only allows writing
 * attributes once, making them non writable in the LwSciSyncAttrList.
 *
 * \param[in] attrList An unreconciled LwSciSyncAttrList containing the attribute
 * key and value to set.
 * \param[in] pairArray Array of LwSciSyncAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL and
 * key member of every LwSciSyncAttrKeyValuePair in the array is an input or
 * input/output attribute and it is > LwSciSyncAttrKey_LowerBound and <
 * LwSciSyncAttrKey_UpperBound and value member of every LwSciSyncAttrKeyValuePair
 * in the array is not NULL.
 * \param[in] pairCount The number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a pairArray is NULL
 *         - @a attrList is NULL
 *         - @a attrList is not unreconciled and/or not writable,
 *         - @a pairCount is 0
 *         - @a pairArray has duplicate keys
 *         - any of the keys in @a pairArray is not a supported public key
 *         - any of the values in @a pairArray is NULL
 *         - any of the len(s) in @a pairArray is invalid for a given attribute
 *         - any of the attributes to be written is non-writable in attrList
 * - Panics if @a attrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListSetAttrs(
    LwSciSyncAttrList attrList,
    const LwSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Gets the value of LwSciSyncAttrKey from slot 0 of the input
 * LwSciSyncAttrList.
 *
 * The return values, stored in LwSciSyncAttrKeyValuePair, consist of
 * const void* pointers to the attribute values from LwSciSyncAttrList.
 * The application must not write to this data. If an attribute was never set,
 * the corresponding value will be set to NULL and length to 0.
 *
 * \param[in] attrList LwSciSyncAttrList to retrieve the value for given
 * LwSciSyncAttrKey(s) from
 * \param[in,out] pairArray A pointer to the array of LwSciSyncAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every LwSciSyncAttrKeyValuePair in the array > LwSciSyncAttrKey_LowerBound
 * and < LwSciSyncAttrKey_UpperBound.
 * \param[in] pairCount The number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 *         - @a pairCount is 0
 *         - any of the keys in @a pairArray is not a supported LwSciSyncAttrKey
 * - Panics if @a attrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListGetAttrs(
    LwSciSyncAttrList attrList,
    LwSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Gets the slot count of the given LwSciSyncAttrList.
 *
 * \param[in] attrList LwSciSyncAttrList to get the slot count from.
 * \return Number of slots or 0 if attrList is NULL or panic if attrList
 *  is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
size_t LwSciSyncAttrListGetSlotCount(
    LwSciSyncAttrList attrList);

/**
 * \brief Appends multiple unreconciled LwSciSyncAttrLists together, forming a
 * single new unreconciled LwSciSyncAttrList with a slot count equal to the sum
 * of all the slot counts of LwSciSyncAttrList(s) in the input array which is
 * no longer writable.
 *
 * \param[in] inputUnreconciledAttrListArray Array containing the unreconciled
 * LwSciSyncAttrList(s) to be appended together.
 * Valid value: Array of unreconciled LwSciSyncAttrList(s) where the array
 * size is at least 1.
 * \param[in] inputUnreconciledAttrListCount Number of unreconciled
 * LwSciSyncAttrList(s) in @a inputUnreconciledAttrListArray.
 * Valid value: inputUnreconciledAttrListCount is valid input if it
 * is non-zero.
 * \param[out] newUnreconciledAttrList Appended LwSciSyncAttrList created out of
 * the input unreconciled LwSciSyncAttrList(s). The output LwSciSyncAttrList is
 * non-writable.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a inputUnreconciledAttrListArray is NULL
 *         - @a inputUnreconciledAttrListCount is 0
 *         - @a newUnreconciledAttrList is NULL
 *         - any of the input LwSciSyncAttrLists in
 *           inputUnreconciledAttrListArray are not unreconciled
 *         - not all the LwSciSyncAttrLists in @a inputUnreconciledAttrListArray
 *           are bound to the same LwSciSyncModule instance.
 * - ::LwSciError_InsufficientMemory if there is insufficient system memory to
 *   create the new unreconciled LwSciSyncAttrList.
 * - ::LwSciError_Overflow if the combined slot counts of all the input
 *   LwSciSyncAttrLists exceeds UINT64_MAX
 * - ::LwSciError_IlwalidState if no more references can be taken for
 *   LwSciSyncModule associated with the LwSciSyncAttrList in @a
 *   inputUnreconciledAttrListArray to create the new LwSciSyncAttrList.
 * - Panics if any of the input LwSciSyncAttrLists are not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListAppendUnreconciled(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncAttrList* newUnreconciledAttrList);

/**
 * \brief Clones an LwSciSyncAttrList. The cloned LwSciSyncAttrList will
 * contain slot count, reconciliation type and all the attribute values of the
 * original LwSciSyncAttrList.
 * If the original LwSciSyncAttrList is unreconciled, then modification will be
 * allowed on the cloned LwSciSyncAttrList using set attributes APIs even if the
 * attributes had been set in the original LwSciSyncAttrList, but the calls to
 * set attributes in either LwSciSyncAttrList will not affect the other.
 *
 * \param[in] origAttrList LwSciSyncAttrList to be cloned.
 * \param[out] newAttrList The new LwSciSyncAttrList.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a origAttrList or @a newAttrList is NULL.
 * - ::LwSciError_InsufficientMemory if there is insufficient system memory
 * - ::LwSciError_IlwalidState if no more references can be taken for
 *     LwSciSyncModule associated with @a origAttrList to create the new
 *     LwSciSyncAttrList.
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 *   to create a LwSciSyncAttrList.
 * - Panic if @a origAttrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListClone(
    LwSciSyncAttrList origAttrList,
    LwSciSyncAttrList* newAttrList);

/**
 * \brief Gets the value(s) of LwSciSyncAttrKey(s) from an LwSciSyncAttrList at
 * given slot index in a multi-slot unreconciled LwSciSyncAttrList.
 *
 * The returned pairArray consist of const void* pointers to the actual attribute
 * values from LwSciSyncAttrList. The application must not overwrite this data.
 * If an attribute was never set, the corresponding value will be set to NULL
 * and length to 0.
 *
 * \param[in] attrList LwSciSyncAttrList to retrieve the LwSciSyncAttrKey and value
 * from.
 * \param[in] slotIndex Index in the LwSciSyncAttrList.
 * Valid value: 0 to slot count of LwSciSyncAttrList - 1.
 * \param[in,out] pairArray Array of LwSciSyncAttrKeyValuePair. Holds the
 * LwSciSyncAttrKey(s) passed into the function and returns an array of
 * LwSciSyncAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every LwSciSyncAttrKeyValuePair in the array > LwSciSyncAttrKey_LowerBound
 * and < LwSciSyncAttrKey_UpperBound.
 * \param[in] pairCount Indicates the number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a slotIndex is not a valid slot in @a attrList
 *         - @a attrList is NULL
 *         - @a pairArray is NULL
 *         - @a pairCount is 0
 *         - any of the keys in @a pairArray is not a LwSciSyncAttrKey
 * - Panics if @a attrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListSlotGetAttrs(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    LwSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Reconciles the input unreconciled LwSciSyncAttrLists into a new
 * reconciled LwSciSyncAttrList.
 *
 * On success, this API call allocates memory for the reconciled LwSciSyncAttrList
 * which has to be freed by the caller using LwSciSyncAttrListFree().
 *
 */
#if (LW_IS_SAFETY == 0)
/**
 * On reconciliation failure, this API call allocates memory for the conflicting
 * LwSciSyncAttrList which has to be freed by the caller using
 * LwSciSyncAttrListFree().
 *
 */
#endif
/**
 * \param[in] inputArray Array containing unreconciled LwSciSyncAttrLists to be
 * reconciled.
 * Valid value: Array of valid LwSciSyncAttrLists where the array size is at least 1
 * \param[in] inputCount The number of unreconciled LwSciSyncAttrLists in
 * @a inputArray.
 * Valid value: inputCount is valid input if is non-zero.
 * \param[out] newReconciledList Reconciled LwSciSyncAttrList. This field is populated
 * only if the reconciliation succeeded.
 */
#if (LW_IS_SAFETY == 0)
/**
 * \param[out] newConflictList Unreconciled LwSciSyncAttrList consisting of the
 * key/value pairs which caused the reconciliation failure. This field is
 * populated only if the reconciliation failed.
 */
#else
/**
 * \param[out] newConflictList Unused.
 */
#endif
/**
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - inputCount is 0
 *         - any of the input LwSciSyncAttrLists are not unreconciled
 *         - not all the LwSciSyncAttrLists in inputArray are bound to the same
 *           LwSciSyncModule instance.
 *         - any of the attributes in any of the input LwSciSyncAttrLists has
 *           an invalid value for that attribute
 *         - inputArray is NULL
 *         - newReconciledList is NULL
 */
#if (LW_IS_SAFETY == 0)
/**        - newConflictList is NULL
 */
#endif
/**
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if a new LwSciSyncAttrList cannot be associated
 *   with the LwSciSyncModule associated with the LwSciSyncAttrList(s) in the
 *   given @a inputArray to create a new reconciled LwSciSyncAttrList
 * - ::LwSciError_Overflow if internal integer overflow is detected.
 * - ::LwSciError_ReconciliationFailed if reconciliation failed because
 *     of conflicting attributes
 * - ::LwSciError_UnsupportedConfig if there is an attribute mismatch between
 *   signaler and waiters.
 * - Panics if any of the input LwSciSyncAttrLists is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListReconcile(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    LwSciSyncAttrList* newReconciledList,
    LwSciSyncAttrList* newConflictList);

#if (LW_IS_SAFETY == 0)
/**
 * \brief Dumps the LwSciSyncAttrList into a binary descriptor.
 *
 * @note This API can be used for debugging purpose.
 *
 * \param[in] attrList LwSciSyncAttrList to create the blob from.
 * \param[out] buf A pointer to binary descriptor buffer.
 * \param[out] len The length of the binary descriptor buffer created.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 * - ::LwSciError_InsufficientMemory if memory allocation failed
 * - Panics if attrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListDebugDump(
    LwSciSyncAttrList attrList,
    void** buf,
    size_t* len);
#endif

/**
 * \brief Transforms the input unreconciled LwSciSyncAttrList(s) to an exportable
 * unreconciled LwSciSyncAttrList descriptor that can be transported by the
 * application to any remote process as a serialized set of bytes over an
 * LwSciIpc channel.
 *
 * @note When exporting an array containing multiple unreconciled
 * LwSciSyncAttrLists, the importing endpoint still imports just one unreconciled
 * LwSciSyncAttrList. This unreconciled LwSciSyncAttrList is referred to as a
 * multi-slot LwSciSyncAttrList. It logically represents an array of
 * LwSciSyncAttrLists, where each key has an array of values, one per slot.
 *
 * \param[in] unreconciledAttrListArray LwSciSyncAttrList(s) to be exported.
 * Valid value: Array of valid LwSciSyncAttrList(s) where the array
 * size is at least 1.
 * \param[in] unreconciledAttrListCount Number of LwSciSyncAttrList(s) in
 * @a unreconciledAttrListArray.
 * Valid value: unreconciledAttrListCount is valid input if it
 * is non-zero.
 * \param[in] ipcEndpoint The LwSciIpcEndpoint through which the caller may
 * send the exported unreconciled LwSciSyncAttrList descriptor.
 * \param[out] descBuf A pointer to the new unreconciled LwSciSyncAttrList
 * descriptor, which the caller can deallocate later using
 * LwSciSyncAttrListFreeDesc().
 * \param[out] descLen The size of the new unreconciled LwSciSyncAttrList
 * descriptor.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a unreconciledAttrListCount is 0
 *      - @a unreconciledAttrListArray is NULL
 *      - @a ipcEndpoint is not a valid LwSciIpcEndpoint
 *      - @a descBuf is NULL
 *      - @a descLen is NULL
 *      - any of the input LwSciSyncAttrLists is not unreconciled
 *      - Not all of the LwSciSyncAttrLists in @a unreconciledAttrListArray
 *        are bound to the same LwSciSyncModule instance.
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if no more references can be taken for the
 *     LwSciSyncModule associated with @a unreconciledAttrListArray.
 * - ::LwSciError_Overflow if the combined slot count of all the unreconciled
 *   LwSciSyncAttrList exceeds UINT64_MAX
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 */
#if (LW_IS_SAFETY == 0)
/**
 *   or there was a problem with LwSciIpc
*/
#endif
/**
 * - ::LwSciError_NoSpace if no space is left in transport buffer to append the
 *        key-value pair.
 * - Panic if any of the input LwSciSyncAttrLists is not valid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListIpcExportUnreconciled(
    const LwSciSyncAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * \brief Transforms the reconciled LwSciSyncAttrList to an exportable reconciled
 * LwSciSyncAttrList descriptor that can be transported by the application to any
 * remote process as a serialized set of bytes over an LwSciIpc channel.
 *
 * \param[in] reconciledAttrList The LwSciSyncAttrList to be exported.
 * \param[in] ipcEndpoint The LwSciIpcEndpoint through which the caller may
 * send the exported reconciled LwSciSyncAttrList descriptor.
 * \param[out] descBuf A pointer to the new reconciled LwSciSyncAttrList
 * descriptor, which the caller can deallocate later using
 * LwSciSyncAttrListFreeDesc().
 * \param[out] descLen The size of the new reconciled LwSciSyncAttrList
 * descriptor.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 *         - @a ipcEndpoint is not a valid LwSciIpcEndpoint
 *         - @a reconciledAttrList does not correspond to a waiter or a signaler
 *         - @a reconciledAttrList is not a reconciled LwSciSyncAttrList
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - ::LwSciError_IlwalidState if no more references can be taken for
 *     LwSciSyncModule associated with @a reconciledAttrList to create the new
 *     LwSciSyncAttrList.
 * - Panic if @a reconciledAttrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListIpcExportReconciled(
    const LwSciSyncAttrList reconciledAttrList,
    LwSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * \brief Transforms an exported unreconciled LwSciSyncAttrList descriptor
 * (potentially received from any process) into an unreconciled
 * LwSciSyncAttrList which is no longer writable.
 *
 * \param[in] module The LwSciSyncModule instance with which to associate the
 * imported LwSciSyncAttrList.
 * \param[in] ipcEndpoint The LwSciIpcEndpoint through which the caller receives
 * the exported unreconciled LwSciSyncAttrList descriptor.
 * \param[in] descBuf The unreconciled LwSciSyncAttrList descriptor to be
 * translated into an unreconciled LwSciSyncAttrList. It should be the result of
 * LwSciSyncAttrListIpcExportUnreconciled
 * Valid value: descBuf is valid input if it is non-NULL.
 * \param[in] descLen The size of the unreconciled LwSciSyncAttrList descriptor.
 * Valid value: descLen is valid input if it is not 0.
 * \param[out] importedUnreconciledAttrList Imported unreconciled LwSciSyncAttrList.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 *         - @a descLen is 0
 *         - @a ipcEndpoint is not a valid LwSciIpcEndpoint
 *         - array of bytes indicated by @a descBuf and @a descLen
 *           do not constitute a valid exported LwSciSyncAttrList descriptor
 *           for an unreconciled LwSciSyncAttrList
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_Overflow if internal integer overflow is detected.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a module is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListIpcImportUnreconciled(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    LwSciSyncAttrList* importedUnreconciledAttrList);

/**
 * \brief Translates an exported reconciled LwSciSyncAttrList descriptor
 * (potentially received from any process) into a reconciled LwSciSyncAttrList.
 *
 * It also validates that the reconciled LwSciSyncAttrList to be imported will
 * be a reconciled LwSciSyncAttrList that is consistent with the constraints in
 * an array of input unreconciled LwSciSyncAttrList(s). This is recommended
 * while importing what is expected to be a reconciled LwSciSyncAttrList to
 * cause LwSciSync to validate the reconciled LwSciSyncAttrList against the input
 * unreconciled LwSciSyncAttrList(s), so that the importing process can be sure
 * that an LwSciSyncObj will satisfy the input constraints.
 *
 * \param[in] module The LwSciSyncModule instance with which to associate the
 * imported LwSciSyncAttrList.
 * \param[in] ipcEndpoint The LwSciIpcEndpoint through which the caller
 *            receives the exported reconciled LwSciSyncAttrList descriptor.
 * \param[in] descBuf The reconciled LwSciSyncAttrList descriptor to be
 * transformed into a reconciled LwSciSyncAttrList.
 * Valid value: descBuf is valid if it is non-NULL.
 * \param[in] descLen The size of the reconciled LwSciSyncAttrList descriptor.
 * Valid value: descLen is valid if it is not 0.
 * \param[in] inputUnreconciledAttrListArray The array of LwSciSyncAttrLists against
 * which the new LwSciSyncAttrList is to be validated.
 * Valid value: Array of valid LwSciSyncAttrList(s)
 * \param[in] inputUnreconciledAttrListCount The number of LwSciSyncAttrLists in
 * inputUnreconciledAttrListArray. If inputUnreconciledAttrListCount is
 * non-zero, then this operation will fail with an error unless all the
 * constraints of all the LwSciSyncAttrLists in inputUnreconciledAttrListArray are
 * met by the imported LwSciSyncAttrList.
 * Valid value: [0, SIZE_MAX]
 * \param[out] importedReconciledAttrList Imported LwSciSyncAttrList.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is not a valid LwSciIpcEndpoint
 *      - the array of bytes indicated by @a descBuf and @a descLen do not
 *        constitute a valid exported LwSciSyncAttrList descriptor for a
 *        reconciled LwSciSyncAttrList
 *      - @a inputUnreconciledAttrListArray is NULL but
 *        @a inputUnreconciledAttrListCount is not 0
 *      - any of the LwSciSyncAttrLists in inputUnreconciledAttrListArray are
 *        not unreconciled
 *      - any of the LwSciSyncAttrLists in @a inputUnreconciledAttrListArray
 *        is not bound to @a module.
 *      - @a importedReconciledAttrList is NULL
 * - ::LwSciError_AttrListValidationFailed if the LwSciSyncAttrList to be
 *   imported either would not be a reconciled LwSciSyncAttrList or would not meet
 *   at least one of constraints in one of the input unreconciled
 *   LwSciSyncAttrLists.
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if no more references can be taken on the
 *     LwSciSyncModule.
 * - ::LwSciError_Overflow if internal integer overflow is detected.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a module or any of the input LwSciSyncAttrLists are not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListIpcImportReconciled(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncAttrList* importedReconciledAttrList);

/**
 * \brief Frees an exported LwSciSyncAttrList descriptor previously returned by
 * any LwSciSyncAttrList exporting function.
 *
 * \param[in] descBuf The exported LwSciSyncAttrList descriptor to be freed.
 * The valid value is non-NULL.
 * \return void
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the input @a descBuf
 *        to be freed
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciSyncAttrListFreeDesc(
    void* descBuf);

/**
 * \brief Frees any resources allocated for the LwSciSyncFence.
 *
 * Upon return, the memory pointed to by the LwSciSyncFence is guaranteed to be all
 * zeros and thus the LwSciSyncFence is returned to the cleared state.
 *
 * \param[in,out] syncFence A pointer to LwSciSyncFence.
 * \return void
 * - Panics if the LwSciSyncObj associated with @a syncFence is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the @a syncFence to
 *        be cleared
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
void LwSciSyncFenceClear(
    LwSciSyncFence* syncFence);

/**
 * \brief Duplicates the given LwSciSyncFence, such that any wait on duplicated
 * LwSciSyncFence will complete at the same time as a wait on given
 * LwSciSyncFence. If the given LwSciSyncFence is in a cleared state, then so
 * also will be the duplicated LwSciSyncFence. The given LwSciSyncFence will be
 * cleared before the duplication. If the given LwSciSyncFence holds any
 * reference on a LwSciSyncObj, then the duplicated LwSciSyncFence will create
 * an additional reference on it.
 *
 * \param[in] srcSyncFence LwSciSyncFence to duplicate.
 * \param[out] dstSyncFence duplicated LwSciSyncFence.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any argument is NULL or both parameters
     point to the same LwSciSyncFence.
 * - ::LwSciError_IlwalidState if no more references can be taken
 *      for LwSciSyncObj associated with @a srcSyncFence
 * - Panics if the LwSciSyncObj associated with @a srcSyncFence
 *   or @a dstSyncFence are not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the @a dstSyncFence
 *        if it had previously been associated with an LwSciSyncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncFenceDup(
    const LwSciSyncFence* srcSyncFence,
    LwSciSyncFence* dstSyncFence);

/**
 * \brief Allocates and initializes a @ref LwSciSyncObj that meets all the
 * constraints specified in the given reconciled LwSciSyncAttrList.
 *
 * @note This function does not take ownership of the reconciled LwSciSyncAttrList.
 * The caller remains responsible for freeing the reconciled LwSciSyncAttrList.
 * The caller may free the reconciled LwSciSyncAttrList any time after this
 * function is called.
 *
 * \param[in] reconciledList A reconciled LwSciSyncAttrList.
 * \param[out] syncObj The allocated @ref LwSciSyncObj.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_IlwalidState if not enough references remain on the
 *      reconciled LwSciSyncAttrList
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 *         - @a reconciledList is not a reconciled LwSciSyncAttrList
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a reconciledList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncObjAlloc(
    LwSciSyncAttrList reconciledList,
    LwSciSyncObj* syncObj);

/**
 * \brief Creates a new @ref LwSciSyncObj holding a reference to the original
 * resources to which the input @ref LwSciSyncObj holds reference to.
 *
 * The duplicated LwSciSyncObj is not a completely new LwSciSyncObj. Therefore,
 * signaling and generating LwSciSyncFences from one affects the state of the
 * other, because it is the same underlying LwSciSyncObj.
 *
 * The resulting LwSciSyncObj must be freed separately by the user.
 *
 * \param[in] syncObj LwSciSyncObj to duplicate.
 * \param[out] dupObj Duplicated LwSciSyncObj.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any argument is NULL.
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if the number of references to the
 *   synchronization object of the input LwSciSyncObj is INT32_MAX and the
 *   newly duplicated LwSciSyncObj tries to take one more reference using this
 *   API.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a syncObj is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncObjDup(
    LwSciSyncObj syncObj,
    LwSciSyncObj* dupObj);

/**
 * \brief Retrieves the reconciled LwSciSyncAttrList associated with an
 * input LwSciSyncObj.
 *
 * The retrieved reconciled LwSciSyncAttrList is always read-only and the
 * Attribute Key values in the LwSciSyncAttrList cannot be changed using the
 * set attribute APIs. In addition, the returned LwSciSyncAttrList must not be
 * freed.
 *
 * \param[in] syncObj Handle corresponding to LwSciSyncObj from which the
 * LwSciSyncAttrList has to be retrieved.
 * \param[out] syncAttrList pointer to the retrieved LwSciSyncAttrList.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if @a syncObj or @a syncAttrList is NULL.
 * - Panics if @a syncObj is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncObjGetAttrList(
    LwSciSyncObj syncObj,
    LwSciSyncAttrList* syncAttrList);

/**
 * \brief Destroys a valid @ref LwSciSyncObj and frees any resources that were
 * allocated for it.
 *
 * \param[in] syncObj LwSciSyncObj to be freed.
 *
 * \return void
 * - Panics if @a syncObj is not a valid @ref LwSciSyncObj
#if 0
 *   or there was an unexpected freeing error from C2C
#endif
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the
 *        LwSciSyncAttrList obtained from LwSciSyncObjGetAttrList() to be
 *        freed, since the lifetime of that reconciled LwSciSyncAttrList is
 *        tied to the associated LwSciSyncObj
 *      - Provided there is no active operation ilwolving the input
 *        LwSciSyncObj @a syncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \implements{11273441}
 */
void LwSciSyncObjFree(
    LwSciSyncObj syncObj);

/**
 * \brief Exports an LwSciSyncObj into an LwSciIpc-transferable object
 * binary descriptor.
 *
 * The binary descriptor can be transferred to a Waiter to create a matching
 * LwSciSyncObj.
 *
 * \param[in] syncObj A LwSciSyncObj to export.
 * \param[in] permissions Flag indicating the expected LwSciSyncAccessPerm.
 * Valid value: any value of LwSciSyncAccessPerm
 * \param[in] ipcEndpoint The LwSciIpcEndpoint through which the caller
 *            intends to transfer the exported LwSciSyncObj descriptor.
 * \param[out] desc LwSciSync fills in this caller-supplied descriptor with
 *             the exported form of LwSciSyncObj that is to be shared across
 *             an LwSciIpc channel.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_ResourceError if something went wrong with LwSciIpc
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a desc is NULL
 *      - @a syncObj is NULL
 *      - @a permissions is invalid
 *      - @a permissions contains larger permissions than those set on
 *        LwSciSyncAttrKey_ActualPerm on the reconciled LwSciSyncAttrList
 *        associated with the LwSciSyncObj granted to this peer
 *      - @a permissions contains smaller permissions than the expected
 *        permissions requested by the receiving peer
 *      - @a ipcEndpoint is invalid
 *      - @a ipcEndpoint does not lead to a peer in the topology tree
 *        of this LwSciSyncObj
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 */
#if (LW_L4T == 1)
/**
 * - ::LwSciError_NotSupported if trying to export syncpoint signaling
 *   over a C2C Ipc channel.
 */
#endif
/**
 * - Panics if @a syncObj is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncObjIpcExport(
    LwSciSyncObj syncObj,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncObjIpcExportDescriptor* desc);

/**
 * \brief Creates and returns an @ref LwSciSyncObj based on the supplied binary
 * descriptor describing an exported @ref LwSciSyncObj.
 *
 * This function is called from the waiter after it receives the binary
 * descriptor from the signaler who has created and exported the binary
 * descriptor.
 *
 * @note This function does not take ownership of input LwSciSyncAttrList. The
 * caller remains responsible for freeing input LwSciSyncAttrList. The caller
 * may free the input LwSciSyncAttrList any time after this function is called.
 *
 * \param[in] ipcEndpoint The @ref LwSciIpcEndpoint through which the caller
 *            received the exported LwSciSyncObj descriptor.
 * \param[in] desc The exported form of @ref LwSciSyncObj received through the
 *            LwSciIpc channel.
 * Valid value: desc is valid if it is non-NULL
 * \param[in] inputAttrList The reconciled LwSciSyncAttrList returned by
 *            @ref LwSciSyncAttrListIpcImportReconciled.
 * \param[in] permissions LwSciSyncAccessPerm indicating the expected access
 * permissions.
 * Valid value: any value of LwSciSyncAccessPerm
 * \param[in] timeoutUs Unused
 * \param[out] syncObj The Waiter's LwSciSyncObj.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_ResourceError if something went wrong with LwSciIpc
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a ipcEndpoint is invalid
 *      - @a desc is invalid
 *      - @a inputAttrList is NULL
 *      - @a permissions is invalid
 *      - @a syncObj is NULL
 *      - @a permissions is LwSciSyncAccessPerm_Auto but permissions granted
 *        in @a desc are not enough to satisfy expected permissions stored in
 *        @a inputAttrList
 *      - the LwSciSyncObjIpcExportDescriptor export descriptor corresponds to
 *        a descriptor generated by an incompatible LwSciSync library version
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if the imported LwSciSyncObj cannot be associated
 *   with the LwSciSyncModule associated with the reconciled input
 *   LwSciSyncAttrList.
 * - ::LwSciError_Overflow if @a desc is too big to be imported.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panics if @a inputAttrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncObjIpcImport(
    LwSciIpcEndpoint ipcEndpoint,
    const LwSciSyncObjIpcExportDescriptor* desc,
    LwSciSyncAttrList inputAttrList,
    LwSciSyncAccessPerm permissions,
    int64_t timeoutUs,
    LwSciSyncObj* syncObj);

/**
 * \brief Exports the input LwSciSyncFence into a binary descriptor shareable
 * across the LwSciIpc channel.
 *
 * The resulting descriptor of a non-cleared LwSciSyncFence is associated with
 * LwSciSyncObj associated with the LwSciSyncFence. After transporting
 * the descriptor via an Ipc path, LwSciSync will be able to recognize
 * that the LwSciSyncFence is associated with this LwSciSyncObj if LwSciSyncObj
 * traversed the same Ipc path.
 *
 * \param[in] syncFence A pointer to LwSciSyncFence object to be exported.
 * \param[in] ipcEndpoint The LwSciIpcEndpoint through which the caller may
 *            send the exported fence descriptor.
 * \param[out] desc The exported form of LwSciSyncFence shared across
 *             an LwSciIpc channel.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - any argument is NULL
 *      - ipcEndpoint is not a valid LwSciIpcEndpoint
 * - Panics if @a syncFence is associated an invalid LwSciSyncObj
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncIpcExportFence(
    const LwSciSyncFence* syncFence,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncFenceIpcExportDescriptor* desc);

/**
 * \brief Fills in the LwSciSyncFence based on the input
 * binary descriptor. If the LwSciSyncFence descriptor does not describe a
 * cleared LwSciSyncFence, then LwSciSync will validate if it corresponds to the
 * LwSciSyncObj and it will associate the out LwSciSyncFence with the
 * LwSciSyncObj.
 *
 * The LwSciSyncFence will be cleared first, removing any previous
 * reference to a LwSciSyncObj.
 *
 * \param[in] syncObj The LwSciSyncObj.
 * \param[in] desc The exported form of LwSciSyncFence.
 *  Valid value: A binary descriptor produced by LwSciSyncIpcExportFence.
 * \param[out] syncFence A pointer to LwSciSyncFence object.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - any argument is NULL
 *      - fence descriptor does not describe a cleared LwSciSyncFence but
 *        it is associated with an LwSciSyncObj different from @a syncObj
 *      - fence descriptor's value exceeds allowed range for syncObj's primitive
 * - ::LwSciError_IlwalidState if @a syncObj cannot take more references.
 * - Panics if @a syncObj is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the @a syncFence if
 *        it had previously been associated with an LwSciSyncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncIpcImportFence(
    LwSciSyncObj syncObj,
    const LwSciSyncFenceIpcExportDescriptor* desc,
    LwSciSyncFence* syncFence);

/**
 * \brief Generates next point on sync timeline of an LwSciSyncObj and fills
 * in the supplied LwSciSyncFence object.
 *
 * This function can be used when the CPU is the Signaler.
 *
 * \param[in] syncObj A valid LwSciSyncObj.
 * \param[out] syncFence LwSciSyncFence to be filled
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 *         - @a syncObj is not a CPU signaler
 */
#if (LW_IS_SAFETY == 0)
/**
 *              and not a C2C signaler
 */
#endif
/**         - @a syncObj does not own the backing primitive
 * - ::LwSciError_IlwalidState if the newly created LwSciSyncFence cannot be
 *   associated with the synchronization object of the given LwSciSyncObj.
 * - Panics if @a syncObj is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *      - Provided there is no active operation ilwolving the @a syncFence if
 *        it had previously been associated with an LwSciSyncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncObjGenerateFence(
    LwSciSyncObj syncObj,
    LwSciSyncFence* syncFence);

/**
 * \brief Signals the @ref LwSciSyncObj using the reconciled primitive that
 * was allocated along with the LwSciSyncObj.
 */
#if (LW_IS_SAFETY == 0)
/**
 * If the signal operation fails, then the timestamp value is undefined.
 */
#endif
/**
 * This function is called when the CPU is the Signaler.
 *
 * \param[in] syncObj A valid LwSciSyncObj to signal.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a syncObj is NULL
 *         - @a syncObj is not a CPU Signaler
 *         - @a syncObj does not own the backing primitive
 * - ::LwSciError_ResourceError if the signal operation fails.
 * - Panics if @a syncObj is not valid
 * 
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncObjSignal(
    LwSciSyncObj syncObj);

/**
 * \brief Performs a synchronous wait on the @ref LwSciSyncFence object until the
 * LwSciSyncFence has been signaled or the timeout expires. Any
 * LwSciSyncCpuWaitContext may be used for waiting on any LwSciSyncFence provided
 * they were created in the same LwSciSyncModule context.
 * One LwSciSyncCpuWaitContext can be used to wait on only one LwSciSyncFence at a time
 * but it can be used to wait on a different LwSciSyncFence at a different time.
 *
 * Waiting on a cleared and expired LwSciSyncFence is always not blocking.
 *
 * \param[in] syncFence The LwSciSyncFence to wait on.
 * \param[in] context LwSciSyncCpuWaitContext holding resources needed
 *            to perform waiting.
 * \param[in] timeoutUs Timeout to wait for in micro seconds, -1 for infinite
 *            wait.
 *  Valid value: [-1, LwSciSyncFenceMaxTimeout]
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if any of the following oclwrs:
 *         - @a syncFence is cleared
 *         - @a syncFence is expired
 *         - @a syncFence has been signaled within the given timeout
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a syncFence is NULL
 *         - @a context is NULL
 *         - @a syncFence and @a context are associated with different
 *           LwSciSyncModule
 *         - @a timeoutUs is invalid
 *         - if caller doesn't have CPU wait permissions in LwSciSyncObj
 *           associated with @a syncFence
 *         - the module reference associated with @a context
 *           is NULL
 * - ::LwSciError_ResourceError if wait operation did not complete
 *   successfully.
 * - ::LwSciError_Timeout if wait did not complete in the given timeout.
 * - ::LwSciError_Overflow if the LwSciSyncFence's id or value are not in range
 *   supported by the primitive this LwSciSyncFence corresponds to.
 * - Panics if any LwSciSyncObj associated with @a syncFence or @a context
 *   is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncFenceWait(
    const LwSciSyncFence* syncFence,
    LwSciSyncCpuWaitContext context,
    int64_t timeoutUs);

/**
 * \brief Read the timestamp associated with the LwSciSyncFence
 *
 * This function can be used when the CPU is the waiter.
 *
 * \param[in] syncFence object of type LwSciSyncFence
 * \param[out] timestampUS time (in microseconds) when the LwSciSyncFence expired.
 *
 * \return ::LwSciError
 * - ::LwSciError_Success if successful
 * - ::LwSciError_ClearedFence if @a syncFence is cleared
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - any argument is NULL
 *         - @a syncFence is NULL or has no non NULL LwSciSyncObj associated
 *         - timestamps are not supported by LwSciSyncObj associated with
 *         - @a syncFence
 *         - @a syncFence does not support timestamps
 *         - the LwSciSyncAttrList associated with the @a syncFence has not
 *           requested CPU access
 * - Panics if LwSciSyncObj associated with @a syncFence is not valid.
 * 
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
LwSciError LwSciSyncFenceGetTimestamp(
    const LwSciSyncFence* syncFence,
    uint64_t* timestampUS);

/*
 * LwSciSync Utility functions
 */

/**
 * \brief Gets the attribute value from the slot 0 of the passed LwSciSyncAttrList
 * with the given LwSciSyncAttrKey.
 * If an LwSciSyncAttrKey was not set, this function will set *value to NULL
 * and *len to 0.
 *
 * \param[in] attrList LwSciSyncAttrList to retrieve the LwSciSyncAttrKey and
 * value from.
 * \param[in] key LwSciSyncAttrKey for which value to retrieve.
 * Valid value: key is a valid input if it is an input or input/output attribute
 * and it is > LwSciSyncAttrKey_LowerBound and < LwSciSyncAttrKey_UpperBound
 * \param[out] value A pointer to the location where the attribute value is written.
 * \param[out] len Length of the value.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *         - @a attrList is NULL
 *         - @a value is NULL
 *         - @a len is NULL
 *         - @a key is not a supported public key
 * - Panics if @a attrList is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListGetAttr(
    LwSciSyncAttrList attrList,
    LwSciSyncAttrKey key,
    const void** value,
    size_t* len);

/**
 * \brief Reconciles the input unreconciled LwSciSyncAttrList(s) into a new
 * reconciled LwSciSyncAttrList. If successful, a reconciled LwSciSyncAttrList will
 * be associated with a newly-allocated @ref LwSciSyncObj that satisfies all
 * the constraints specified in the reconciled LwSciSyncAttrList.
 *
 * Note: This function serves as a colwenience function that combines calls to
 * LwSciSyncAttrListReconcile and LwSciSyncObjAlloc.
 *
 * \param[in] inputArray Array containing the unreconciled LwSciSyncAttrList(s)
 *            to reconcile.
 * Valid value: Array of valid LwSciSyncAttrLists where the array size is at least 1
 * \param[in] inputCount Number of unreconciled LwSciSyncAttrLists in
 *            @a inputArray.
 * Valid value: inputCount is valid input if is non-zero.
 * \param[out] syncObj The new LwSciSyncObj.
 */
#if (LW_IS_SAFETY == 0)
/**
 * \param[out] newConflictList unreconciled LwSciSyncAttrList consisting of the
 *             key-value pairs which caused the reconciliation failure.
 * Valid value: This parameter is a valid output parameter only if the return
 *     code is ::LwSciError_ReconciliationFailed
 */
#else
/**
 * \param[out] newConflictList Unused
 */
#endif
/**
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a inputCount is 0
 *      - @a inputArray is NULL
 *      - @a syncObj is NULL
 *      - not all the LwSciSyncAttrLists in inputArray are bound to the same
 *        LwSciSyncModule instance
 *      - any of the attributes in any of the input LwSciSyncAttrLists has an
 *        invalid value for that attribute
 *      - if any of the LwSciSyncAttrList in attrList are not unreconciled
 */
#if (LW_IS_SAFETY == 0)
/**      - @a newConflictList is NULL
 */
#endif
/**
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_IlwalidState if the newly created LwSciSyncObj cannot be
 *   associated with the LwSciSyncModule with which the LwSciSyncAttrList(s) in
 *   @a inputArray are associated.
 * - ::LwSciError_ReconciliationFailed if reconciliation failed.
 * - ::LwSciError_ResourceError if system lacks resource other than memory.
 * - ::LwSciError_UnsupportedConfig if there is an LwSciSyncAttrList mismatch between
 *   Signaler and Waiters.
 * - Panics if any of the input LwSciSyncAttrLists are not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncAttrListReconcileAndObjAlloc(
    const LwSciSyncAttrList inputArray[],
    size_t inputCount,
    LwSciSyncObj* syncObj,
    LwSciSyncAttrList* newConflictList);

/**
 * \brief Exports an LwSciSyncAttrList and LwSciSyncObj into an
 * LwSciIpc-transferable object binary descriptor pointed to by @a data.
 *
 * The binary descriptor can subsequently be transferred to Waiters to create
 * a matching LwSciSyncObj.
 *
 * Note: This function serves as a colwenience function that combines calls to
 * LwSciSyncAttrListIpcExportReconciled and LwSciSyncObjIpcExport.
 *
 * \param[in] syncObj LwSciSyncObj to export.
 * \param[in] permissions Flag indicating the expected LwSciSyncAccessPerm.
 * Valid value: permissions is valid if it is set to LwSciSyncAccessPerm_WaitOnly
 * or LwSciSyncAccessPerm_Auto.
 * \param[in] ipcEndpoint The LwSciIpcEndpoint through which the caller may send
 *            the exported LwSciSyncAttrList and LwSciSyncObj descriptor.
 * \param[out] attrListAndObjDesc Exported form of LwSciSyncAttrList and
 * LwSciSyncObj shareable across an LwSciIpc channel.
 * \param[out] attrListAndObjDescSize Size of the exported blob.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrListAndObjDesc is NULL
 *      - @a attrListAndObjDescSize is NULL
 *      - @a syncObj is NULL
 *      - @a permissions flag contains signaling rights
 *      - @a ipcEndpoint is invalid.
 *      - @a ipcEndpoint does not lead to a peer in the topology tree
 *        of this LwSciSyncObj
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 *     or something went wrong with LwSciIpc
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panic if @a syncObj is not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncIpcExportAttrListAndObj(
    LwSciSyncObj syncObj,
    LwSciSyncAccessPerm permissions,
    LwSciIpcEndpoint ipcEndpoint,
    void** attrListAndObjDesc,
    size_t* attrListAndObjDescSize);

/**
 * \brief Frees an @ref LwSciSyncIpcExportAttrListAndObj descriptor
 * returned by a successful call to @ref LwSciSyncIpcExportAttrListAndObj.
 *
 * Does nothing for NULL.
 *
 * \param[in] attrListAndObjDescBuf Exported @ref LwSciSyncIpcExportAttrListAndObj
 * descriptor to be freed.
 * \return void
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation ilwolving the input
 *        @a attrListAndObjDescBuf to be freed
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void LwSciSyncAttrListAndObjFreeDesc(
    void* attrListAndObjDescBuf);

/**
 * \brief Creates an LwSciSyncObj based on the supplied binary descriptor
 * returned from a successful call to @ref LwSciSyncIpcExportAttrListAndObj
 * that has not yet been freed via @ref LwSciSyncAttrListAndObjFreeDesc. It
 * also validates reconciled LwSciSyncAttrList against input
 * unreconciled LwSciSyncAttrLists to ensure that the reconciled
 * LwSciSyncAttrList satisfies the constraints of all the given unreconciled
 * LwSciSyncAttrLists.
 *
 * This function is called from the Waiter after it receives the binary
 * descriptor from the Signaler who has created the binary descriptor.
 * Waiter will create its own LwSciSyncObj and return as output.
 *
 * Note: This function serves as a colwenience function that combines calls to
 * LwSciSyncAttrListIpcImportReconciled and LwSciSyncObjIpcImport.
 *
 * \param[in] module A @ref LwSciSyncModule to associate the imported
 *            @ref LwSciSyncAttrList with.
 * \param[in] ipcEndpoint The @ref LwSciIpcEndpoint through which the caller
 *            receives the exported LwSciSyncAttrList and LwSciSyncObj descriptor.
 * \param[in] attrListAndObjDesc Exported form of LwSciSyncAttrList and
 *            LwSciSyncObj received through LwSciIpc channel.
 * Valid value: attrListAndObjDesc is valid if it is non-NULL.
 * \param[in] attrListAndObjDescSize Size of the exported blob.
 * Valid value: attrListAndObjDescSize is valid if it is bigger or equal
 *              sizeof(LwSciSyncObjIpcExportDescriptor).
 * \param[in] attrList The array of unreconciled LwSciSyncAttrLists
 *            against which the new LwSciSyncAttrList is to be validated.
 * Valid value: Array of valid LwSciSyncAttrList(s)
 * \param[in] attrListCount Number of unreconciled LwSciSyncAttrLists in the
 *            @a attrList array.
 * Valid value: [0, SIZE_MAX]
 * \param[in] minPermissions Flag indicating the expected LwSciSyncAccessPerm.
 * Valid value: LwSciSyncAccessPerm_WaitOnly and LwSciSyncAccessPerm_Auto
 * \param[in] timeoutUs Unused
 * \param[out] syncObj Waiter's LwSciSyncObj.
 *
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a ipcEndpoint is invalid
 *      - @a attrListAndObjDesc is invalid
 *      - @a attrListAndObjDescSize is invalid
 *      - @a minPermissions is invalid
 *      - @a syncObj is NULL
 *      - input unreconciled LwSciSyncAttrLists' constraints
 *        are not satisfied by attributes of imported LwSciSyncObj.
 *      - @a attrList is NULL but @a attrListCount is not 0
 *      - if any of the LwSciSyncAttrList in attrList are not unreconciled
 *      - @a minPermissions is LwSciSyncAccessPerm_Auto but permissions granted
 *        in the object part of @a attrListAndObjDesc are not enough to satisfy
 *        expected permissions stored in the attribute list part
 * - ::LwSciError_InsufficientMemory if memory allocation failed.
 * - ::LwSciError_ResourceError if system lacks resource other than memory
 *     or something went wrong with LwSciIpc
 * - ::LwSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panics if any of the following oclwrs:
 *      - @a module is not valid
 *      - any of the input unreconciled LwSciSyncAttrLists are not valid
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
LwSciError LwSciSyncIpcImportAttrListAndObj(
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    const void* attrListAndObjDesc,
    size_t attrListAndObjDescSize,
    LwSciSyncAttrList const attrList[],
    size_t attrListCount,
    LwSciSyncAccessPerm minPermissions,
    int64_t timeoutUs,
    LwSciSyncObj* syncObj);

/**
 * \brief Checks if the loaded library version is compatible with the version
 * the application was compiled against.
 *
 * This function checks the version of all dependent libraries and sets the
 * output variable to true if all libraries are compatible and all in parameters
 * valid, else sets output to false.
 *
 * \param[in] majorVer build major version.
 * Valid value: valid if set to LwSciSyncMajorVersion
 * \param[in] minorVer build minor version.
 * Valid value: valid if set to <= LwSciSyncMinorVersion
 * \param[out] isCompatible pointer to the bool holding the result.
 * \return ::LwSciError, the completion code of the operation:
 * - ::LwSciError_Success if successful.
 * - ::LwSciError_BadParameter if any of the following oclwrs:
 *      - @a isCompatible is NULL
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
#if (LW_IS_SAFETY == 0)
/**
 *      - @a failed to check dependent library versions.
 */
#endif
LwSciError LwSciSyncCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible);

#if defined(__cplusplus)
}
#endif // __cplusplus
 /** @} */
#endif // INCLUDED_LWSCISYNC_H
