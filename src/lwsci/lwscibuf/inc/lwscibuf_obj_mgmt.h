/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_OBJ_MGMT_H
#define INCLUDED_LWSCIBUF_OBJ_MGMT_H

#include <stdbool.h>
#include <stdint.h>

#include "lwscibuf.h"
#include "lwscibuf_internal.h"
#include "lwscicommon_objref.h"
#include "lwscicommon_utils.h"
#include "lwscierror.h"
#include "lwsciipc_internal.h"
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
#include "lwscibuf_c2c_internal.h"

/**
 * @brief Abstraction for C2c interface specific target handle.
 *
 * @implements{}
 */
typedef union {
    /* Target handle for PCIe interface. */
    LwSciC2cPcieBufTargetHandle pcieTargetHandle;
} LwSciC2cInterfaceTargetHandle;
#endif

/**
 * @defgroup lwscibuf_obj_api LwSciBuf Object APIs
 * List of APIs to create/operate on LwSciBufObj.
 * @{
 */

/**
 * New LwSciBuObj reference is created from input LwSciBufObj
 * using LwSciCommon functionality to duplicate the reference.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the input LwSciBufObj reference is provided via
 *        LwSciCommonDuplicateRef()
 *
 * @implements{18843231}
 *
 * @fn LwSciError LwSciBufObjDup(LwSciBufObj bufObj, LwSciBufObj* dupObj)
 */

/**
 * Calls LwSciBufAttrListReconcile() followed by LwSciBufObjAlloc() and
 * frees reconciled LwSciBufAttrList obtained from successful call to
 * LwSciBufAttrListReconcile() after successfully calling LwSciBufObjAlloc()
 * since LwSciBufObjAlloc() takes a reference to reconciled LwSciBufAttrList
 * in its call.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the input LwSciBufAttrList(s) and their
 *        associated LwSciBufModule is handled via LwSciBufAttrListReconcile()
 *
 * @implements{17827350}
 *
 * @fn LwSciError LwSciBufAttrListReconcileAndObjAlloc(
 * const LwSciBufAttrList attrListArray[], size_t attrListCount,
 * LwSciBufObj* bufObj, LwSciBufAttrList* newConflictList)
 */

/**
 * LwSciBufObj reference is freed using LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The user must ensure that the same LwSciBufObj is not used by
 *        multiple threads in other functions other than other ilwocations of
 *        this API at the same time when calling this API
 *
 * @implements{18843252}
 *
 * @fn void LwSciBufObjFree(LwSciBufObj bufObj)
 */

/**
 * LwSciBufAttrList is retrieved from the memory object to which
 * the input LwSciBufObj holds reference.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufAttrList
 *        associated with the LwSciBufObj is never modified after creation
 *        (so there is no data-dependency)
 *
 * @implements{18843237}
 *
 * @fn LwSciError LwSciBufObjGetAttrList(LwSciBufObj bufObj,
 * LwSciBufAttrList* bufAttrList)
 *
 */

/**
 * CPU virtual address is retrieved from memory object containing it.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the CPU pointer associated
 *        with the LwSciBufObj is never modified after creation when it is
 *        mapped (so there is no data-dependency)
 *
 * @implements{18843240}
 *
 * @fn LwSciError LwSciBufObjGetCpuPtr(LwSciBufObj bufObj, void**  ptr)
 */

/**
 * CPU virtual address is retrieved from memory object containing it.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the CPU pointer associated
 *        with the LwSciBufObj is never modified after creation when it is
 *        mapped (so there is no data-dependency)
 *
 * @implements{18843243}
 *
 * @fn LwSciError LwSciBufObjGetConstCpuPtr(LwSciBufObj bufObj,
 * const void**  ptr)
 */

/**
 * CPU virtual address of the buffer is retrieved from the memory object
 * to which the input LwSciBufObj holds reference and flushes it for the
 * given len bytes starting at offset using LwSciBufAllocIfaceCpuCacheFlush().
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent operations on the underlying buffer is handled via
 *        LwSciBufAllocIfaceCpuCacheFlush()
 *
 * @implements{18843246}
 *
 * @fn LwSciError LwSciBufObjFlushCpuCacheRange( LwSciBufObj bufObj,
 * uint64_t offset, uint64_t len)
 *
 */

/**
 * LwSciBufObj and memory object are allocated together using LwSciCommon
 * functionality. The buffer is allocated by supplying allocation parameters
 * to LwSciBufAllocIfaceAlloc() which are obtained from values of attributes
 * specifying buffer allocation parameters in reconciled LwSciBufAttrList.
 * The allocated buffer conforms to the following buffer properties defined in
 * the reconciled LwSciBufAttrList:
 * 1) buffer size as specified in LwSciBufPrivateAttrKey_Size in the reconciled
 *    LwSciBufAttrList.
 * 2) Memory domain of the buffer as specified in
 *    LwSciBufInternalGeneralAttrKey_MemDomainArray in the reconciled
 *    LwSciBufAttrList.
 * 3) Whether GPUs can access the buffer based on the values specified in
 *    LwSciBufGeneralAttrKey_GpuId in the reconciled LwSciBufAttrList.
 * 4) Heap from which the buffer is allocated based on the value specified in
 *    LwSciBufPrivateAttrKey_HeapType in the reconciled LwSciBufAttrList as well
 *    as the contiguity associated with the heap.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization:
 *      - Conlwrrent access to the input LwSciBufAttrList is provided via
 *        LwSciBufAttrListDupRef()
 *      - Conlwrrent reads of attributes in the input LwSciBufAttrList is
 *        provided via LwSciBufAttrListCommonGetAttrs()
 *      - Conlwrrent modification of the LwSciBufAttrList to set the
 *        LwSciBufGeneralAttrKey_ActualPerm is handled via
 *        LwSciBufAttrListCommonSetAttrs()
 *      - Note: Since many APIs lwrrently read from reconciled LwSciBufAttrList
 *        without synchronization claiming that the
 *        LwSciBufGeneralAttrKey_ActualPerm attribute is not modified, we claim
 *        that this API is non-thread-safe to ensure that is never run
 *        conlwrrently with any other APIs that use the input LwSciBufAttrList
 *        so that the synchronization in those APIs is not needed. In practice
 *        due to ARR LWSCIBUF_RES_004 this is not a permitted use case.
 *
 * @implements{18843228}
 *
 * @fn LwSciError LwSciBufObjAlloc(LwSciBufAttrList reconciledAttrList,
 * LwSciBufObj* bufObj)
 */

/**
 * If the given LwSciBufAttrValAccessPerm is less than actual
 * LwSciBufAttrValAccessPerm of the input LwSciBufObj, it retrieves
 * the reconciled LwSciBufAttrList and LwSciBufRmHandle from memory
 * object referenced by the input LwSciBufObj, creates a clone of the
 * reconciled LwSciBufAttrList and sets the LwSciBufGeneralAttrKey_ActualPerm
 * key for the cloned LwSciBufAttrList to the given LwSciBufAttrValAccessPerm,
 * and creates a new memory object and LwSciBufObj referencing it by calling
 * LwSciBufObjCreateFromMemHandlePriv by passing true value to one of the
 * parameters colweying the function to duplicate the LwSciBufRmHandle.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - If the permissions represented by the buffer handle in the memory
 *        object referenced by the input LwSciBufObj is the same as the input
 *        LwSciBufAttrValAccessPerm
 *          - Conlwrrent access to the LwSciBufObj is provided via
 *            LwSciBufObjDup()
 *      - Otherwise
 *          - Conlwrrent access to the reconciled LwSciBufAttrList associated
 *            with the input LwSciBufObj is provided via
 *            LwSciBufAttrListClone()
 *          - Conlwrrent reads of attributes in the LwSciBufAttrList associated
 *            with the input LwSciBufObj is provided via
 *            LwSciBufAttrListCommonGetAttrs()
 *          - Conlwrrent access to the underlying LwSciBufRmHandle is provided
 *            via LwSciBufObjCreateFromMemHandlePriv()
 *
 * @implements{18843249}
 *
 * @fn LwSciError LwSciBufObjDupWithReducePerm(LwSciBufObj bufObj,
 * LwSciBufAttrValAccessPerm reducedPerm, LwSciBufObj* newBufObj)
 */

/**
 * @}
 */

/**
 * @defgroup lwscibuf_umd_api LwSciBuf APIs
 * List of APIs specific to LwMedia specific UMDs
 * @{
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on the LwSciBufObj reference to serialize access to
 *        the LwMedia flags via LwSciCommonRefLock()
 *      - Locks are held for the duration of any operations on the LwMedia
 *        flags
 *      - Locks are released when all operations on the LwMedia flags are
 *        complete via LwSciCommonRefUnlock()
 *
 * @implements{18843255}
 *
 * @fn bool LwSciBufObjAtomicGetAndSetLwMediaFlag(LwSciBufObj bufObj,
 * uint32_t flagIndex, bool newValue)
 */

/**
 * refcount for LwSciBufObj reference is incremented using
 * LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufObj reference is provided via
 *        LwSciCommonIncrAllRefCounts()
 *
 * @implements{18843258}
 *
 * @fn LwSciError LwSciBufObjRef(LwSciBufObj bufObj)
 */

/**
 * @}
 */

/**
 * @defgroup lwscibuf_obj_api_int LwSciBuf internal object APIs
 * List of internal APIs to operate on LwSciBuf object
 * @{
 */

/**
 * LwSciBufRmHandle, offset and length are retrieved from the memory
 * object to which the input LwSciBufObj holds reference.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufRmHandle is
 *        never modified after the LwSciBufObj creation (so there is no
 *        data-dependency)
 *      - Reads only occur from immutable data since the offset is never
 *        modified after the LwSciBufObj creation (so there is no
 *        data-dependency)
 *      - Reads only occur from immutable data since the len is never modified
 *        after the LwSciBufObj creation (so there is no data-dependency)
 *
 * @implements{18843261}
 *
 * @fn LwSciError LwSciBufObjGetMemHandle(LwSciBufObj bufObj,
 * LwSciBufRmHandle* memHandle, uint64_t* offset, uint64_t* len)
 */

/**
 * This interface just calls helper LwSciBufObjCreateFromMemHandlePriv()
 * passing it true parameter for duplicating LwSciBufRmHandle.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access is handled via LwSciBufObjCreateFromMemHandlePriv()
 *
 * @implements{18843264}
 *
 * @fn LwSciError LwSciBufObjCreateFromMemHandle(
 * const LwSciBufRmHandle memHandle, uint64_t offset, uint64_t len,
 * LwSciBufAttrList reconciledAttrList, LwSciBufObj* bufObj)
 */

/**
 * @}
 */

/**
 * @brief Allocates and initializes a memory object with the buffer
 * represented by the given LwSciBufRmHandle at the given @a offset
 * for given @a len, and outputs a new LwSciBufObj referencing the
 * memory object. LwSciBufObj and memory object are allocated together
 * using LwSciCommon functionality. Buffer constraints are considered
 * from the given reconciled LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufAttrList reference is handled via
 *        LwSciBufAttrListDupRef()
 *      - Conlwrrent access to the LwSciBufRmHandle is handled via
 *        LwSciBufAllocIfaceDupHandle() if dupHandle is true
 *      - The LwSciBufGeneralAttrKey_ActualPerm key is only ever modified after
 *        reconciliation in LwSciBufObjAlloc(). But LwSciBufObjAlloc() is not
 *        thread-safe if multiple APIs are using the same reconciled
 *        LwSciBufAttrList, so conlwrrent modification and reads leading to a
 *        non-thread-safe behavior is not possible.
 *
 * @param[in] memHandle LwSciBufRmHandle.
 * Valid value: memHandle is valid input if the RM memory handle represented
 * by it is received from a successful call to LwRmMemHandleAllocAttr() and
 * has not been deallocated by using LwRmMemHandleFree().
 * @param[in] offset The offset within the buffer represented by memHandle.
 * Valid value: 0 to size of the buffer represented by LwSciBufRmHandle - 1.
 * @param[in] len The length of the buffer to be represented by the new
 * LwSciBufObj. The size of the buffer represented by memHandle must be at
 * least @a offset + @a len.
 * Valid value: 1 to size of the buffer represented by LwSciBufRmHandle -
 * @a offset.
 * @param[in] reconciledAttrList The reconciled LwSciBufAttrList.
 * @param[in] dupHandle Flag to indicate whether the buffer handle represented
 * by LwSciBufRmHandle should be duplicated instead of being reused.
 * Valid value: true or false.
 * @param[in] isRemoteObject boolean flag indicating whether LwSciBufObj being
 *            created is remote or local. True implies that LwSciBufObj is
 *            remote (meaning it is imported from the remote peer for which
 *            there is no backing LwSciBufRmHandle. This can is set to true
 *            only in C2c case when LwSciBufObj allocated by remote Soc peer
 *            is imported), false implies otherwise.
 * @param[in] dupC2cTargetHandle boolean flag indicating if
 *            LwSciC2cInterfaceTargetHandle is duplicated.
 * @param[in] copyFuncs LwSciC2cCopyFuncs.
 * @param[in] c2cTargetHandle LwSciC2cInterfaceTargetHandle.
 * @param[out] bufObj new LwSciBufObj.
 *
 * @return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a bufObj is NULL.
 *    - @a reconciledAttrList is NULL.
 *    - @a reconciledAttrList is unreconciled.
 *    - @a offset + @a len > buffer size represented by LwSciBufRmHandle. The
 *      size represented by LwSciBufRmHandle is obtained by calling
 *      LwSciBufAllocIfaceGetSize() interface.
 *    - @a len > buffer size represented by output attributes for respective
 *      LwSciBufType in @a reconciledAttrList.
 *    - buffer size represented by output attributes for respective
 *      LwSciBufType in @a reconciledAttrList > buffer size represented by
 *      LwSciBufRmHandle.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_ResourceError if any of the following oclwrs:
 *      - LWPU driver stack failed
 *      - system lacks resource other than memory
 * - LwSciError_Overflow if @a len + @a offset exceeds UINT64_MAX.
 * - LwSciError_IlwalidState if new LwSciBufAttrList cannot be associated with
 *   the LwSciBufModule associated with the given LwSciBufAttrList to create a
 *   new LwSciBufObj
 * - LwSciError_NotSupported if LwSciBufPrivateAttrKey_MemDomain on the
 *   LwSciBufAttrList is not supported.
 * - Panics if:
 *        - @a bufObj is NULL.
 *        - @a reconciledAttrList is NULL.
 *        - @a reconciledAttrList is invalid.
 *        - @a len is 0.
 */
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 *        - @a dupHandle is true and @a isRemoteObject is true.
 *        - @a dupC2cTargetHandle is true and @a c2cTargetHandle is non-NULL and
 *          function pointer for duplicating @a c2cTargetHandle in @a copyFuncs
 *          is NULL.
 */
#endif
/**
 * @implements{18843270}
 */
LwSciError LwSciBufObjCreateFromMemHandlePriv(
    const LwSciBufRmHandle memHandle,
    uint64_t offset,
    uint64_t len,
    LwSciBufAttrList reconciledAttrList,
    bool dupHandle,
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    bool isRemoteObject,
    bool dupC2cTargetHandle,
    LwSciC2cCopyFuncs copyFuncs,
    LwSciC2cInterfaceTargetHandle c2cTargetHandle,
#endif
    LwSciBufObj* bufObj);

/**
 * @brief Callback to free the data associated with the LwSciObj representing
 * the underlying memory object using LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciObj prior to calling this function
 *
 * @param[in] obj Pointer to the LwSciObj associated with the LwSciBufObj to
 * free
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - obj is NULL
 *      - obj is invalid
 *
 * @implements{22034427}
 */
void LwSciBufObjCleanup(
    LwSciObj* obj);

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
/**
 * @brief Sets LwSciC2cPcieBufTargetHandle into memory object referenced by
 * LwSciBufObj.
 *
 * @param[in] bufObj Reference to memory object.
 * @param[in] targetHandle LwSciC2cPcieBufTargetHandle to be set in the memory
 * object.
 *
 * @return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if @a bufObj is NULL.
 * - Panics if any of the following oclwrs:
 *      - @a bufObj is invalid.
 *      - @a targetHandle is NULL.
 *
 */
LwSciError LwSciBufObjSetC2cTargetHandle(
    LwSciBufObj bufObj,
    LwSciC2cInterfaceTargetHandle targetHandle);

/**
 * @brief Sets LwSciC2cCopyFuncs into memory object referenced by
 * LwSciBufObj.
 *
 * @param[in] bufObj Reference to memory object.
 * @param[in] c2cCopyFuncs LwSciC2cCopyFuncs to be set in the memory
 * object.
 *
 * @return LwSciError
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if @a bufObj is NULL.
 * - Panics if any of the following oclwrs:
 *      - @a bufObj is invalid.
 *
 */
LwSciError LwSciBufObjSetC2cCopyFunctions(
    LwSciBufObj bufObj,
    LwSciC2cCopyFuncs c2cCopyFuncs);

/**
 * @brief Retrives C2c interface specific LwSciC2cInterfaceTargetHandle from
 * LwSciBufObj.
 * @note LwSciC2cInterfaceTargetHandle is owned by LwSciBuf and thus user
 * retriving LwSciC2cInterfaceTargetHandle via this API must not free it
 * directly.
 *
 * @param[in] bufObj LwSciBufObj from which LwSciC2cInterfaceTargetHandle needs
 * to be retrieved.
 * @param[out] c2cInterfaceTargetHandle LwSciC2cInterfaceTargetHandle to be
 * retrieved.
 *
 * @return LwSciError
 * - LwSciError_Success if successful.
 * LwSciError_BadParameter if any of the following oclwrs:
 *      - @a bufObj is NULL.
 *      - @a c2cInterfaceTargetHandle is NULL.
 * - Panics of bufObj is invalid.
 */
LwSciError LwSciBufObjGetC2cInterfaceTargetHandle(
    LwSciBufObj bufObj,
    LwSciC2cInterfaceTargetHandle* c2cInterfaceTargetHandle);
#endif

#endif /* INCLUDED_LWSCIBUF_OBJ_MGMT_H */
