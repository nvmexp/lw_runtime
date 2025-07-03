/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_ATTR_H
#define INCLUDED_LWSCIBUF_ATTR_H

#include "lwscibuf_module.h"
#include "lwscibuf_attr_desc.h"

/**
 * @brief Enumeration of different states of LwSciBufAttrList.
 * Note: Cloned LwSciBufAttrList will inherit state of input
 * LwSciBufAttrList.
 *
 * @implements{18842196}
 */
typedef enum {
    LwSciBufAttrListState_Unreconciled = 0x0U,
    LwSciBufAttrListState_Reconciled,
    LwSciBufAttrListState_Appended,
    LwSciBufAttrListState_UpperBound = 0xFF,
} LwSciBufAttrListState;

/**
 * @brief Global constant to specify maximum attribute keys per LwSciBufType.
 * There can be total 64k keys for the combination of LwSciBufType and
 * LwSciBufAttrKeyType. However, we only need maximum 32 keys as of
 * now. Thus, maximum value is set to 32 for optimization.
 *
 * @implements{18842220}
 */
#define LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE  32U

/**
 * @brief Represents a node of a LWList in which each node represents
 *  a UMD specific LwSciBufInternalAttrKey/value pair.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * @implements{18842223}
 */
typedef struct {
    /** UMD specific LwSciBufInternalAttrKey. This member is initialized to
     * the given UMD specific LwSciBufInternalAttrKey when a new
     * LwSciBufUmdAttrValData is allocated.
     */
    uint32_t key;

    /** Length of value to be stored for the key. This member is initialized
     * to given length in bytes when a new LwSciBufUmdAttrValData is allocated.
     */
    uint64_t len;

    /** Pointer to value of the key. This member is initialized by allocating
     * the memory of len bytes and the value is copied from the given pointer
     * when a new LwSciBufUmdAttrValData is allocated.
     */
    void* value;

    /** Write Lock Enable flag. This member is initialized to
     * LwSciBufAttrStatus_SetLocked when a new LwSciBufUmdAttrValData is
     * allocated.
     */
    LwSciBufAttrStatus privateAttrStatus;

    /** LWList node entry. This member is initialized by a successful call to
     * lwListInit() when a new LwSciBufUmdAttrValData is allocated. This member
     * is deinitialized by calling lwListDel() when LwSciBufUmdAttrValData
     * is deallocated.
     */
    LWListRec listEntry;
} LwSciBufUmdAttrValData;

/**
 * @brief Per slot storage for all class of keys
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * @implements{18842184}
 */
typedef struct {
    /** Data storage for keys having datatype LwSciBufType_General and
     * keytype LwSciBufAttrKeyType_Public/LwSciBufAttrKeyType_Internal. This
     * member is zero-initialized when the LwSciBufPerSlotAttrList is created
     * and initialized to a given LwSciBufGeneralAttrObjPriv when it is
     * cloned. This member is deinitialized when the LwSciBufPerSlotAttrList is
     * freed. */
    LwSciBufGeneralAttrObjPriv genAttr;

    /** Data storage for keys having datatype other than LwSciBufType_General
     * and keytype LwSciBufAttrKeyType_Public/LwSciBufAttrKeyType_Internal.
     * The data storage for relevant indices are allocated by calling
     * LwSciBufAttrListMallocBufferType() depending on
     * the value of the LwSciBufGeneralAttrKey_Types key. The data storage for
     * each data type is freed when the LwSciBufPerSlotAttrList is freed. */
    char *dataTypeAttr[LwSciBufType_MaxValid];

    /** Data storage for keys having keytype LwSciBufAttrKeyType_UMDPrivate.
     * This member is zero-initialized when the LwSciBufPerSlotAttrList is
     * created and initialized to a deep copy of a given LwSciBufUmdAttrObjPriv
     * when it is cloned. This member is deinitialized and any allocated UMD
     * keys are freed when the LwSciBufPerSlotAttrList is freed.*/
    LwSciBufUmdAttrObjPriv umdAttr;

    /** Data storage for keys having keytype LwSciBufAttrKeyType_Private.
     * This member is zero-initialized when the LwSciBufPerSlotAttrList is
     * created and initialized to a given LwSciBufPrivateAttrObjPriv when it is
     * cloned. This member is deinitialized when the LwSciBufPerSlotAttrList is
     * freed. */
    LwSciBufPrivateAttrObjPriv privAttr;
} LwSciBufPerSlotAttrList;

/**
 * @brief Structure that LwSciBufAttrList points to. An LwSciBufAttrListRec
 * holds a reference to a LwSciBufAttrListObjPriv structure which contains the
 * actual data. Multiple LwSciBufAttrListRec can reference a particular
 * LwSciBufAttrListObjPriv. This structure is allocated along with
 * LwSciBufAttrListObjPriv using LwSciCommon functionality. Subsequent
 * LwSciBufAttrListRec references can be created to LwSciBufAttrListObjPriv
 * using duplicate referencing functionality provided by LwSciCommon. This
 * structure is deallocated using LwSciCommon functionality, closing access to
 * LwSciBufAttrListObjPriv via this reference.
 *
 * @implements{18842169}
 */
struct LwSciBufAttrListRec {
    /** LwSciCommon referencing header. This should be a first member in the
     * structure. */
    LwSciRef refHeader;
};

typedef struct LwSciBufAttrListRec LwSciBufAttrListRefPriv;

static inline LwSciBufAttrListRefPriv*
    LwSciCastRefToLwSciBufAttrListRefPriv(LwSciRef* arg)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    return (LwSciBufAttrListRefPriv*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciBufAttrListRefPriv, refHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}
/**
 * @brief Actual structure that contains data corresponding to the
 * LwSciBufAttrList. This structure is allocated using LwSciCommon functionality
 * as part of allocating LwSciBufAttrListRec. This structure is deallocated
 * using LwSciCommon functionality when all the LwSciBufAttrListRec holding
 * reference to it are deallocated.
 *
 * @implements{18842172}
 */
struct LwSciBufAttrListObjPriv {
    /** LwSciCommon object. This should be a first member in the structure. */
    LwSciObj objHeader;

    /** Magic ID to detect if this LwSciBufAttrListObjPriv is valid. This member
     * must be initialized to a particular constant value when
     * the LwSciBufAttrListObjPriv is allocated. The constant value chosen
     * to initialize this member must be non-zero. It must be changed to a
     * different value when this LwSciBufAttrListObjPriv is deallocated.
     * This member must NOT be modified in between allocation and
     * deallocation of the LwSciBufAttrListObjPriv. Whenever an
     * LwSciBufAttrListObjPriv is retrieved using LwSciCommon from an
     * LwSciBufAttrListRec received (via an LwSciBufAttrList) from
     * outside the attribute core unit (including, but not limited to
     * LwSciBufAttrListValidate()), the attribute core unit must validate
     * the Magic ID.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. If it does, this indicates corruption. As such, there is
     *        no data-dependency and no locking is necessary.
     * */
    uint32_t magic;

    /** Number of slots. It is initialized to 1 by LwSciBufAttrListCreate(),
     * initialized to number of specified slots by
     * LwSciBufAttrListCreateMultiSlot(), initialized to sum of slots in
     * unreconciled LwSciBufAttrLists by LwSciBufAttrListAppendUnreconciled(),
     * initialized to number of slots in the original LwSciBufAttrList from
     * which LwSciBufAttrList is cloned by LwSciBufAttrListClone(). This member
     * must NOT be modified in between allocation and deallocation of the
     * LwSciBufAttrListObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. This means conlwrrent reads will always read a constant
     *        value once the LwSciBufAttrList is returned to the caller of any
     *        element-level API. As such, there is no data-dependency and no
     *        locking is necessary.
     */
    uint64_t slotCount;

    /** Array of data storage for number of slots specified by @a slotCount. This
     * member is initialized by allocating a dynamic array of
     * LwSciBufPerSlotAttrList for the size of slotCount during
     * LwSciBufAttrListCreate() or LwSciBufAttrListCreateMultiSlot(). This
     * member is deinitialized by deallocating the array of
     * LwSciBufPerSlotAttrList during LwSciBufAttrListFree() call.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is protected by the LwSciObj lock
     *      - Any conlwrrent access must be serialized by holding the LwSciObj
     *        lock
     */
    LwSciBufPerSlotAttrList* slotAttrList;

    /** LwSciBufModule. Initialized by duplicating the LwSciBufModule reference.
     * Deinitialized by destroying LwSciBufModule reference. This member must
     * NOT be modified in between allocation and deallocation of the
     * LwSciBufAttrListObjPriv.
     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. This means conlwrrent reads will always read a constant
     *        value once the LwSciBufAttrList is returned to the caller of any
     *        element-level API. As such, there is no data-dependency and no
     *        locking is necessary.
     */
    LwSciBufModule module;

    /** State of LwSciBufAttrList. Initialized to
     * LwSciBufAttrListState_Unreconciled when LwSciBufAttrList is created. It
     * is set to LwSciBufAttrListState_Appended when unreconciled
     * LwSciBufAttrList(s) are appended or when they are imported over LwSciIpc
     * channel. It is set to LwSciBufAttrListState_Reconciled when
     * LwSciBufAttrList is reconciled. This member must NOT be modified in
     * between allocation and deallocation of the LwSciBufAttrListObjPriv.

     *
     * Conlwrrency:
     *  - Synchronization
     *      - This is not protected by locking the LwSciObj
     *      - Once returned from an element-level API, this value does not
     *        change. This means conlwrrent reads will always read a constant
     *        value once the LwSciBufAttrList is returned to the caller of any
     *        element-level API. As such, there is no data-dependency and no
     *        locking is necessary.
     */
    LwSciBufAttrListState state;
};

typedef struct LwSciBufAttrListObjPriv LwSciBufAttrListObjPriv;

static inline LwSciBufAttrListObjPriv*
    LwSciCastObjToLwSciBufAttrListObjPriv(LwSciObj* arg)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciBuf-ADV-MISRAC2012-017")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciBuf-ADV-MISRAC2012-014")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciBuf-ADV-MISRAC2012-001")
    return (LwSciBufAttrListObjPriv*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciBufAttrListObjPriv, objHeader));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}

/**
 * @brief This structure is used to store state of iterator which is
 * used to iterate UMD specific LwSciBufInternalAttrKey(s) of a
 * LwSciBufAttrList.
 *
 * Synchronization: Access to an instance of this datatype must be
 * externally synchronized
 *
 * @implements{18842226}
 */
typedef struct {
    /** Head node of LWList. This member is initialized with the value
     * of the LwSciBufInternalAttrKey_LwMediaPrivateFirst attribute key
     * for the given slotIndex in the given LwSciBufAttrList when this
     * iterator is initialized.
     */
    LWListRec* headAddr;

    /** Current position in LWList. This member is initialized to
     * the first node in LWList when this iterator is initialized.
     * On every iteration using LwSciBufUmdAttrKeyIterNext() this member
     * will be updated to point to the next node in LWList.
     */
    LWListRec* iterAddr;
} LwSciBufUmdAttrKeyIterator;

/**
 * @brief Iterator for LwSciBufAttrKeys and LwSciBufInternalAttrKeys.
 *
 * Synchronization: Access to an instance of this datatype must be externally
 * synchronized
 *
 * @implements{18842229}
 */
typedef struct {
    /** Max number of LwSciBufAttrKeyType. This member is initialized to
     * LwSciBufAttrKeyType_MaxValid.
     */
    uint32_t keyTypeMax;

    /** Max number of LwSciBufType. This member is initialized to
     * LwSciBufType_MaxValid.
     */
    uint32_t dataTypeMax;

    /** Max Number of keys per LwSciBufAttrKeyType and LwSciBufType. This
     * member is initialized to LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE.
     */
    uint32_t keyMax;

    /** Current value of LwSciBufAttrKeyType. This member is initialized to 0
     * or to the given offset value.
     */
    uint32_t keyType;

    /** Current value of LwSciBufType. This member is initialized to 0
     * or to the given offset value.
     */
    uint32_t dataType;

    /** Current value of attribute key identifier. This member is initialized
     * to 0 or to the given offset value. On every iteration using
     * LwSciBufAttrKeyIterNext(), this member is incremented, when it
     * reaches keyMax it will be reset to zero and dataType member is
     * incremented. If dataType member reaches dataTypeMax, it will be reset to
     * zero and keyType member is incremented. When keyType member reaches
     * keyTypeMax, the iteration is completed.
     */
    uint32_t key;
} LwSciBufAttrKeyIterator;


/**
 * @defgroup lwscibuf_attr_list_api LwSciBuf Attribute List APIs
 * Methods to perform operations on LwSciBuf attribute lists.
 * @{
 */

/**
 * Allocates LwSciBufAttrListRec and LwSciBufAttrListObjPriv using
 * LwSciCommon functionality and initializes them to values defining
 * new single slot unreconciled LwSciBufAttrList. The single slot unreconciled
 * LwSciBufAttrList is created by calling LwSciBufAttrListCreateMultiSlot()
 * by passing @a emptyIpcRoute parameter to true in order to create the
 * empty LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access is provided via LwSciBufAttrListCreateMultiSlot()
 *
 * @implements{18843435}
 *
 * @fn LwSciError LwSciBufAttrListCreate(LwSciBufModule module,
 * LwSciBufAttrList* newAttrList)
 */

/**
 * Deallocates LwSciBufAttrListRec using LwSciCommon functionality.
 * LwSciBufAttrListObjPriv is deallocated using LwSciCommon functionality when
 * all the LwSciBufAttrListRec references are destroyed.
 *
 * @note Every owner of the LwSciBufAttrList shall call LwSciBufAttrListFree()
 * only after all the functions ilwoked by the owner with LwSciBufAttrList
 * as an input are completed.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufAttrListRec and
 *        LwSciBufAttrListObjPriv is handled via LwSciCommonFreeObjAndRef()
 *      - The user must ensure that the same LwSciBufAttrList is not used by
 *        multiple threads in other functions other than other ilwocations of
 *        this API at the same time when calling this API
 *
 * @implements{18843438}
 *
 * @fn void LwSciBufAttrListFree(LwSciBufAttrList attrList)
 */

/**
 * Retrieves LwSciBufType from every key in the LwSciBufAttrKeyValuePair, gets
 * data storage corresponding to LwSciBufType and LwSciBufAttrKeyType_Public
 * corresponding to slot 0 from LwSciBufPerSlotAttrList contained in specified
 * LwSciBufAttrListObjPriv and returns value of the key by filling
 * LwSciBufAttrKeyValuePair structure provided LwSciBufAttrStatus of the key
 * is not LwSciBufAttrStatus_Empty or LwSciBufAttrStatus_SetUnlocked.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access is handled via LwSciBufAttrListSlotGetAttrs()
 *
 * @implements{18843447}
 *
 * @fn LwSciError LwSciBufAttrListGetAttrs(LwSciBufAttrList attrList,
 * LwSciBufAttrKeyValuePair* pairArray, size_t pairCount)
 */

/**
 * Checks for the LwSciBufGeneralAttrKey_Types key in the input @a pairArray
 * and determines if data storage needs to be allocated. If so, data storage
 * for each LwSciBufType specified in LwSciBufGeneralAttrKey_Types is
 * allocated. Then retrieves LwSciBufType from every key in the
 * LwSciBufAttrKeyValuePair, gets data storage corresponding to LwSciBufType
 * and LwSciBufAttrKeyType_Public corresponding to slot 0 from
 * LwSciBufPerSlotAttrList contained in specified LwSciBufAttrListObjPriv,
 * copies value of the key specified in LwSciBufAttrKeyValuePair structure to
 * the corresponding data storage and sets LwSciBufAttrStatus of key to
 * LwSciBufAttrStatus_SetLocked indicating that the key is set.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access is handled via LwSciBufAttrListCommonSetAttrs()
 *
 * @implements{17827563}
 *
 * @fn LwSciError LwSciBufAttrListSetAttrs(LwSciBufAttrList attrList,
 * LwSciBufAttrKeyValuePair* pairArray, size_t pairCount)
 */

/**
 * Slot count is retrieved from the LwSciBufAttrListObjPriv for which the
 * input LwSciBufAttrList holds reference.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the slot count is never
 *        modified after creation (so there is no data-dependency)
 *
 * @implements{18843444}
 *
 * @fn size_t LwSciBufAttrListGetSlotCount(LwSciBufAttrList attrList)
 */

/**
 * Retrieves LwSciBufType from every key in the LwSciBufAttrKeyValuePair, gets
 * data storage corresponding to LwSciBufType and LwSciBufAttrKeyType_Public
 * corresponding to input @a slotIndex from LwSciBufPerSlotAttrList contained
 * in specified LwSciBufAttrListObjPriv and returns value of the key by filling
 * LwSciBufAttrKeyValuePair structure provided LwSciBufAttrStatus of the key is
 * not LwSciBufAttrStatus_Empty or LwSciBufAttrStatus_SetUnlocked.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on the input LwSciBufAttrList to serialize reads
 *      - Locks are held for the duration of any reads from the input
 *        LwSciBufAttrList
 *      - Locks are released when all operations on the input LwSciBufAttrList
 *        are complete
 *
 * @implements{18843450}
 *
 * @fn LwSciError LwSciBufAttrListSlotGetAttrs(LwSciBufAttrList attrList,
 * size_t slotIndex, LwSciBufAttrKeyValuePair* pairArray, size_t pairCount)
 */

/**
 * Allocates new LwSciBufAttrListObjPriv  and copies all the data contained in
 * input LwSciBufAttrListObjPriv to new LwSciBufAttrListObjPriv. If
 * LwSciBufAttrList being cloned is unreconciled, LwSciBufAttrStatus of all the
 * previously set attributes is set to LwSciBufAttrStatus_SetUnlocked indicating
 * that key can be set using set APIs for LwSciBufAttrList. The @a newAttrList
 * is created using LwSciBufAttrListCreateMultiSlot() by passing
 * @a emptyIpcRoute as false.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on the input LwSciBufAttrList to serialize reads
 *      - Locks are held for the duration of any reads from the input
 *        LwSciBufAttrList
 *      - Locks are released when all operations on the input LwSciBufAttrList
 *        are complete
 *      - Conlwrrent access to the LwSciBufModule associated with origAttrList
 *        is provided via LwSciBufAttrListCreateMultiSlot()
 *
 * @implements{18843453}
 *
 * @fn LwSciError LwSciBufAttrListClone(LwSciBufAttrList origAttrList,
 * LwSciBufAttrList* newAttrList)
 */

/**
 * Creates a new unreconciled LwSciBufAttrList using
 * LwSciBufAttrListAppendWithLocksUnreconciled() by passing @a acquireLocks to
 * true.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the input LwSciBufAttrList(s) and their
 *        associated LwSciBufModule is handled via
 *        LwSciBufAttrListAppendWithLocksUnreconciled(..., acquireLocks=true)
 *
 * @implements{18843456}
 *
 * @fn LwSciError LwSciBufAttrListAppendUnreconciled(
 * const LwSciBufAttrList inputUnreconciledAttrListArray[],
 * size_t inputUnreconciledAttrListCount,
 * LwSciBufAttrList* newUnreconciledAttrList)
 */

/**
 * isReconciled flag is retrieved from the LwSciBufAttrListObjPriv for
 * which the input LwSciBufAttrList holds reference.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the reconciliation state
 *        is never modified after creation (so there is no data-dependency)
 *
 * @implements{18843459}
 *
 * @fn LwSciError LwSciBufAttrListIsReconciled(LwSciBufAttrList attrList,
 * bool* isReconciled)
 */

/**
 * @}
 */

/**
 * @defgroup lwscibuf_attr_list_api_int LwSciBuf internal Attribute list APIs
 * Attribute list APIs exposed internally
 * @{
 */

/**
 * Retrieves LwSciBufType from every key in the
 * LwSciBufInternalAttrKeyValuePair, gets data storage corresponding to
 * LwSciBufType and LwSciBufAttrKeyType_Internal/LwSciBufAttrKeyType_UMDPrivate
 * corresponding to slot 0 from LwSciBufPerSlotAttrList contained in specified
 * LwSciBufAttrListObjPriv, copies value of the key specified in
 * LwSciBufInternalAttrKeyValuePair structure to the corresponding data storage
 * and sets LwSciBufAttrStatus of key to LwSciBufAttrStatus_SetLocked indicating
 * that key is set.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access is handled via LwSciBufAttrListCommonSetAttrs()
 *
 * @implements{17827584}
 *
 * @fn LwSciError LwSciBufAttrListSetInternalAttrs(LwSciBufAttrList attrList,
 * LwSciBufInternalAttrKeyValuePair* pairArray, size_t pairCount)
 */

/**
 * Retrieves LwSciBufType from every key in the
 * LwSciBufInternalAttrKeyValuePair, gets data storage corresponding to
 * LwSciBufType and LwSciBufAttrKeyType_Internal/LwSciBufAttrKeyType_UMDPrivate
 * corresponding to slot 0 from LwSciBufPerSlotAttrList contained in specified
 * LwSciBufAttrListObjPriv and returns value of the key by filling
 * LwSciBufInternalAttrKeyValuePair structure provided LwSciBufAttrStatus of the
 * key is not LwSciBufAttrStatus_Empty or LwSciBufAttrStatus_SetUnlocked.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on the input LwSciBufAttrList to serialize reads
 *      - Locks are held for the duration of any reads from the input
 *        LwSciBufAttrList
 *      - Locks are released when all operations on the input LwSciBufAttrList
 *        are complete
 *
 * @implements{18843465}
 *
 * @fn LwSciError LwSciBufAttrListGetInternalAttrs(LwSciBufAttrList attrList,
 * LwSciBufInternalAttrKeyValuePair* pairArray, size_t pairCount)
 */

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @implements{18843468}
 *
 * @fn LwSciError LwSciBufGetUMDPrivateKeyWithOffset(
 * LwSciBufInternalAttrKey key,
 * uint32_t offset,
 * LwSciBufInternalAttrKey* offsettedKey)
 */

/**
 * @}
 */

/**
 * @brief Initializes the given LwSciBufAttrKeyIterator.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @param[in] keyTypeOffset: LwSciBufAttrKeyType offset to start the
 * iteration with. Valid value: an enumeration value as defined by
 * LwSciBufAttrKeyType not including LwSciBufAttrKeyType_UMDPrivate.
 * @param[in] dataTypeOffset: LwSciBufType offset to start the iteration with.
 * Valid value: LwSciBufType_General <= dataTypeOffset < LwSciBufType_MaxValid.
 * @param[in] keyOffset: attribute key identifier offset to start the iteration
 * with. Valid value: 0 <= keyOffset < LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE.
 *
 * @param[out] iter: Initialized LwSciBufAttrKeyIterator.
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *    - @a iter is NULL
 *    - @a keyTypeOffset is not valid
 *    - @a dataTypeOffset is not valid
 *    - @a keyOffset is not valid
 *
 * @implements{18843474}
 */
void LwSciBufAttrKeyIterInit(
    uint32_t keyTypeOffset,
    uint32_t dataTypeOffset,
    uint32_t keyOffset,
    LwSciBufAttrKeyIterator* iter);

/**
 * @brief Encodes and returns the LwSciBufAttrKey or LwSciBufInternalAttrKey or
 * LwSciBufPrivateAttrKey from given LwSciBufAttrKeyIterator and updates the
 * LwSciBufAttrKeyIterator to represent next LwSciBufAttrKey or
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey. UMD keys are not iterated
 * over.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the same input LwSciBufAttrKeyIter is
 *        not used by multiple threads at the same time
 *
 * @param[in] iter: LwSciBufAttrKeyIterator. Valid value: iter is valid input
 * if it is not NULL and it was initialized using successful call to
 * LwSciBufAttrKeyIterInit().
 *
 * @param[out] keyTypeEnd: Flag to indicate whether the keyType member of
 * LwSciBufAttrKeyIterator reaches LwSciBufAttrKeyType_MaxValid, true means the
 * iteration is completed.
 * @param[out] dataTypeEnd: Flag to indicate whether the dataType member of
 * LwSciBufAttrKeyIterator reaches LwSciBufType_MaxValid, true means iteration
 * for all LwSciBufType(s) for a LwSciBufAttrKeyType is completed.
 * @param[out] keyEnd: Flag to indicate the key member of
 * LwSciBufAttrKeyIterator reaches LW_SCI_BUF_MAX_DEFINED_KEYS_PER_TYPE, true
 * means iteration of LwSciBufAttrKey or LwSciBufInternalAttrKey or
 * LwSciBufPrivateAttrKey for a LwSciBufType is over.
 * @param[out] key: Output LwSciBufAttrKey or LwSciBufInternalAttrKey or
 * LwSciBufPrivateAttrKey.
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - @a iter is NULL
 *      - @a keyTypeEnd is NULL
 *      - @a dataTypeEnd is NULL
 *      - @a keyEnd is NULL
 *      - @a key is NULL
 *
 * @implements{18843477}
 */
void LwSciBufAttrKeyIterNext(
    LwSciBufAttrKeyIterator* iter,
    bool* keyTypeEnd,
    bool* dataTypeEnd,
    bool* keyEnd,
    uint32_t* key);

/**
 * @brief Allocates a new LwSciBufUmdAttrValData.
 * This function only reads from the @a value and saves copies.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * @param[in] key: UMD specific LwSciBufInternalAttrKey.
 * Valid value: key is valid input if it is within the range
 * of UMD private keys.
 * @param[in] len: Length of value in bytes.
 * Valid value: len is valid if it is non-zero.
 * @param[in] value: Pointer to value for the key.
 * Valid value: value is valid input if it not NULL.
 *
 * @param[out] privateKeyNode: Allocated LwSciBufUmdAttrValData.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_InsufficientMemory if insufficient system memory
 *   to allocate LwSciBufUmdAttrValData.
 * - LwSciError_BadParameter if any one of the following oclwrs:
 *    - @a key is not in the range of UMD private keys
 *    - @a len is 0
 *    - @a value is NULL
 *    - @a privateKeyNode is NULL
 *
 * @implements{18843480}
 */
LwSciError LwSciBufCreatePrivateKeyNode(
    uint32_t key,
    uint64_t len,
    const void* value,
    LwSciBufUmdAttrValData** privateKeyNode);


/** @brief Checks if the magic member of the LwSciBufAttrListObjPriv matches
 * with the Magic ID.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the Magic ID is never
 *        modified after creation (so there is no data-dependency)
 *
 * @param[in] attrList: LwSciBufAttrList to validate.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if @a attrList is valid.
 * - LwSciError_BadParameter if @a attrList is NULL.
 * - panics if @a attrList is not valid
 *
 * @implements{18843483}
 */
LwSciError LwSciBufAttrListValidate(
    LwSciBufAttrList attrList);


/**
 * @brief Allocates LwSciBufAttrListRec and LwSciBufAttrListObjPriv using
 * LwSciCommon functionality and initializes them to values defining
 * new multi-slot (specified by @a slotCount) unreconciled LwSciBufAttrList.
 * If @a emptyIpcRoute is true then LwSciBufPrivateAttrKey_SciIpcRoute is
 * set with NULL LwSciBufIpcRoute*.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access to the LwSciBufModule is provided via
 *        LwSciBufModuleDupRef()
 *
 * @param[in] module: LwSciBufModule to associate with the newly
 * created LwSciBufAttrList.
 * @param[in] slotCount: Number of slots required in LwSciBufAttrList.
 * Valid value: slotCount is valid input if it is non-zero.
 * @param[out] newAttrList: The new LwSciBufAttrList.
 * @param[in] emptyIpcRoute: boolean value indicating if the empty IPC route
 * needs to be created in the LwSciBufAttrList.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a module is NULL
 *      - @a newAttrList is NULL
 *      - @a slotCount is 0.
 * - LwSciError_InsufficientMemory if insufficient system memory to create
 *   a LwSciBufAttrList.
 * - LwSciError_IlwalidState if a new LwSciBufAttrList cannot be associated
 *   with the given LwSciBufModule
 * - LwSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a module is invalid
 *
 * @implements{18843486}
 */
LwSciError LwSciBufAttrListCreateMultiSlot(
    LwSciBufModule module,
    size_t slotCount,
    LwSciBufAttrList* newAttrList,
    bool emptyIpcRoute);

/**
 * @brief Calls LwSciBufAttrListCommonGetAttrsWithLock() by passing
 * 'acquireLock' flags as true.
 *
 * @param[in] attrList: LwSciBufAttrList to fetch the
 * LwSciBufAttrKeyValuePair(s) or LwSciBufInternalAttrKeyValuePair(s) or
 * LwSciBufPrivateAttrKeyValuePair(s) from.
 * @param[in] slotIndex: Index in the LwSciBufAttrList.
 * Valid value: 0 to slot count of LwSciBufAttrList - 1.
 * @param[in,out] pairArray: Array of LwSciBufAttrKeyValuePair or
 * LwSciBufInternalAttrKeyValuePair or LwSciBufPrivateAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL.
 * @param[in] pairCount: Number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * @param[in] keyType: LwSciBufAttrKeyType to indicate type of @a pairArray
 * (LwSciBufAttrKeyValuePair or LwSciBufInternalAttrKeyValuePair or
 * LwSciBufPrivateAttrKeyValuePair).
 * Valid value: keyType is valid input if it is LwSciBufAttrKeyType_Public or
 * LwSciBufAttrKeyType_Internal or LwSciBufAttrKeyType_Private.
 * @param[in] override: If true, value of the key will be read even if the key
 * is not readable. Valid value: True or false.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - @a override is false, @a attrList is reconciled and any of the
 *        attribute key(s) specified in @a pairArray is input only
 *      - @a override is false, @a attrList is unreconciled and any of the
 *        attribute key(s) specified in @a pairArray is output only
 *      - @a keyType is invalid
 *      - any of the keys in @a pairArray does not match the provided @a keyType
 *      - any of the keys specified in @a pairArray is not a valid enumeration
 *        value defined in the LwSciBufAttrKey, LwSciBufInternalAttrKey,
 *        LwSciBufPrivateAttrKey enums
 *      - @a slotIndex >= slot count of LwSciBufAttrList
 *      - @a attrList is an imported unreconciled LwSciBufAttrList
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid
 *
 * @implements{18843489}
 */
LwSciError LwSciBufAttrListCommonGetAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    void* pairArray,
    size_t pairCount,
    LwSciBufAttrKeyType keyType,
    bool override);

/**
 * @brief Retrieves LwSciBufType from every key in the @a pairArray, gets
 * data storage corresponding to LwSciBufType and LwSciBufAttrKeyType
 * corresponding to specified @a slotIndex from LwSciBufPerSlotAttrList
 * contained in specified LwSciBufAttrListObjPriv and returns value of
 * the key by filling @a pairArray structure provided
 * LwSciBufAttrStatus of the key is not LwSciBufAttrStatus_Empty or
 * LwSciBufAttrStatus_SetUnlocked.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - This API manages its own synchronization
 *      - Locks are taken on the input LwSciBufAttrList to serialize reads
 *      - Locks are held for the duration of any reads from the input
 *        LwSciBufAttrList
 *      - Locks are released when all operations on the input LwSciBufAttrList
 *        are complete
 *
 * @param[in] attrList: LwSciBufAttrList to fetch the
 * LwSciBufAttrKeyValuePair(s) or LwSciBufInternalAttrKeyValuePair(s) or
 * LwSciBufPrivateAttrKeyValuePair(s) from.
 * @param[in] slotIndex: Index in the LwSciBufAttrList.
 * Valid value: 0 to slot count of LwSciBufAttrList - 1.
 * @param[in,out] pairArray: Array of LwSciBufAttrKeyValuePair or
 * LwSciBufInternalAttrKeyValuePair or LwSciBufPrivateAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL.
 * @param[in] pairCount: Number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * @param[in] keyType: LwSciBufAttrKeyType to indicate type of @a pairArray
 * (LwSciBufAttrKeyValuePair or LwSciBufInternalAttrKeyValuePair or
 * LwSciBufPrivateAttrKeyValuePair).
 * Valid value: keyType is valid input if it is LwSciBufAttrKeyType_Public or
 * LwSciBufAttrKeyType_Internal or LwSciBufAttrKeyType_Private or
 * LwSciBufAttrKeyType_UMDPrivate. LwSciBufAttrKeyType_Internal and
 * LwSciBufAttrKeyType_UMDPrivate are interchangeable.
 * @param[in] override: If true, value of the key will be read even if the key
 * is not readable. This is ignored when the LwSciBufAttrKeyType of the
 * attribute key in @a pairArray is an UMD key. Valid value: True or false.
 * @param[in] acquireLock boolean value indicating if @a attrList should be
 * locked before performing get operartion on it.
 * Valid value: True or False.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - @a override is false, @a attrList is reconciled and any of the
 *        attribute key(s) specified in @a pairArray is input only
 *      - @a override is false, @a attrList is unreconciled and any of the
 *        attribute key(s) specified in @a pairArray is output only
 *      - @a keyType is invalid
 *      - any of the attribute key(s) specified in @a pairArray belong to a
 *        different LwSciBufAttrKeyType than specified in @a keyType (other
 *        than LwSciBufAttrKeyType_Internal/LwSciBufAttrKeyType_UMDPrivate)
 *      - any of the attribute key(s) specified in @a pairArray is not a valid
 *        enumeration value defined in the LwSciBufAttrKey,
 *        LwSciBufInternalAttrKey, LwSciBufPrivateAttrKey enums, or is not a
 *        UMD-specific LwSciBufInternalAttrKey
 *      - @a slotIndex >= slot count of LwSciBufAttrList
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid
 *
 * @implements{}
 */
LwSciError LwSciBufAttrListCommonGetAttrsWithLock(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    void* pairArray,
    size_t pairCount,
    LwSciBufAttrKeyType keyType,
    bool override,
    bool acquireLock);

/**
 * @brief Checks for the LwSciBufGeneralAttrKey_Types key in the input
 * @a pairArray and determines if data storage needs to be allocated. If so,
 * data storage for each LwSciBufType specified in LwSciBufGeneralAttrKey_Types
 * is allocated. Then retrieves LwSciBufType from every key in the @a pairArray,
 * gets data storage corresponding to LwSciBufType and LwSciBufAttrKeyType
 * corresponding to @a slotIndex from LwSciBufPerSlotAttrList contained in the
 * specified LwSciBufAttrListObjPriv, copies value of the key specified in
 * @a pairArray structure to the corresponding data storage and sets
 * LwSciBufAttrStatus of the key to LwSciBufAttrStatus_SetLocked indicating
 * that the key is set.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on the input LwSciBufAttrList to serialize writes
 *      - Locks are held for the duration of any writes from the input
 *        LwSciBufAttrList
 *      - Locks are released when all operations on the input LwSciBufAttrList
 *        are complete
 *
 * @param[in] attrList Unreconciled LwSciBufAttrList where the function
 * will set the values for LwSciBufAttrKey(s) or LwSciBufInternalAttrKey(s) or
 * LwSciBufPrivateAttrKey(s).
 * @param[in] slotIndex: Index in the LwSciBufAttrList.
 * Valid value: 0 to slot count of LwSciBufAttrList - 1.
 * @param[in] pairArray: Array of LwSciBufAttrKeyValuePair or
 * LwSciBufInternalAttrKeyValuePair or LwSciBufPrivateAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL.
 * @param[in] pairCount Number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * @param[in] keyType: LwSciBufAttrKeyType to indicate type of @a pairArray
 * (LwSciBufAttrKeyValuePair or LwSciBufInternalAttrKeyValuePair or
 * LwSciBufPrivateAttrKeyValuePair).
 * Valid value: keyType is valid input if it is LwSciBufAttrKeyType_Public or
 * LwSciBufAttrKeyType_Internal or LwSciBufAttrKeyType_Private or
 * LwSciBufAttrKeyType_UMDPrivate. LwSciBufAttrKeyType_Internal and
 * LwSciBufAttrKeyType_UMDPrivate are interchangeable.
 * @param[in] override: Flag identifying if the value is written by LwSciBuf
 * driver or by the user of LwSciBuf. If set to true, the value is written even
 * if the key is in LwSciBufAttrStatus_SetLocked state or even it is a
 * read-only key. Also, if this flag is set to true then valid value of the key
 * is checked against valid values that are allowed to be set by LwSciBuf for
 * the key. If it is false then valid value of the key is checked against valid
 * values that are allowed to be set by the user for the key. This is ignored
 * when the LwSciBufAttrKeyType of the attribute key in @a pairArray is an UMD
 * key.
 * Valid value: True or false.
 * @param[in] skipValidation Flag indicating if the valid value check for the
 * value being set can be skipped. Validation is skipped if the flag is true
 * otherwise validation is performed.
 * Valid value: True or false.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a attrList is a reconciled LwSciBufAttrList
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - @a keyType is invalid
 *      - @a slotIndex >= slot count of LwSciBufAttrList
 *      - @a override is false and @a attrList is a reconciled LwSciBufAttrList
 *      - @a override if false, @a attrList is an unreconciled LwSciBufAttrList,
 *        and any of the attribute key(s) specified in @a pairArray are output
 *        only
 *      - @a override is false and any of the attribute key(s) specified in
 *        @a pairArray has already been set
 *      - any of the attribute key(s) specified in @a pairArray is not a valid
 *        enumeration value defined in the LwSciBufAttrKey,
 *        LwSciBufInternalAttrKey, LwSciBufPrivateAttrKey enums, or is not a
 *        UMD-specific LwSciBufInternalAttrKey
 *      - any of the attribute key(s) specified in @a pairArray belong to a
 *        different LwSciBufAttrKeyType than specified in @a keyType (other
 *        than LwSciBufAttrKeyType_Internal/LwSciBufAttrKeyType_UMDPrivate)
 *      - any of the attribute key(s) specified in @a pairArray oclwrs more
 *        than once
 *      - the LwSciBufGeneralAttrKey_Types key set (or lwrrently being set) on
 *        @a attrList does not contain the LwSciBufType of the datatype-specific
 *        attribute key(s)
 *      - length(s) set for attribute key(s) in @a pairArray are invalid
 *      - value(s) set for attribute key(s) in @a pairArray are invalid
 * - LwSciError_InsufficientMemory if not enough system memory
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid
 *
 * @implements{18843492}
 */
LwSciError LwSciBufAttrListCommonSetAttrs(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    const void* pairArray,
    size_t pairCount,
    LwSciBufAttrKeyType keyType,
    bool override,
    bool skipValidation);

/** @brief Checks if the magic member of the LwSciBufAttrListObjPriv matches
 * with the LwSciBufAttrList Magic ID for every unreconciled LwSciBufAttrList
 * of @a inputArray associated with the same LwSciBufModule.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the Magic ID is never
 *        modified after creation (so there is no data-dependency)
 *
 * @param[in] inputArray: Array of unreconciled LwSciBufAttrList(s) to validate.
 * Valid value: inputArray is valid input if it is not NULL.
 * @param[in] inputCount: Number of elements/entries in @a inputArray.
 * Valid value: inputCount is valid input if it is non-zero.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - any LwSciBufAttrList(s) in @a inputArray is NULL
 *      - not all the LwSciBufAttrList(s) in @a inputAttrListArr are bound to
 *        the same LwSciBufModule instance
 *      - any of the LwSciBufAttrList(s) in @a inputAttrListArr is reconciled
 * - panics if any of the LwSciBufAttrList is invalid.
 *
 * @implements{18843495}
 */
LwSciError LwSciBufValidateAttrListArray(
    const LwSciBufAttrList inputArray[],
    size_t inputCount);

/** @brief Gets details of a LwSciBufAttrKey or LwSciBufInternalAttrKey
 * or LwSciBufPrivateAttrKey from Attribute key descriptor. This function
 * assumes that validation has already oclwrred to verify that a corresponding
 * Attribute Key Descriptor exists for the given key.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from memory which is never modified (so there is no
 *        data-dependency)
 *
 * @param[in] key: attribute key.
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums.
 * @param[out] dataSize: size of each element in attribute key's value.
 * @param[out] dataMaxInstance: number of elements in attribute key's value.
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - dataSize is NULL
 *      - dataMaxInstance is NULL
 *      - key is not a valid enumeration value defined in the LwSciBufAttrKey,
 *        LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums
 *
 * @implements{18843498}
 */
void LwSciBufAttrGetDataDetails(
    uint32_t key,
    size_t* dataSize,
    uint32_t* dataMaxInstance);


/** @brief Gets LwSciBufKeyAccess details of a LwSciBufAttrKey or
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey from Attribute key
 * descriptor. This function assumes that validation has already oclwrred to
 * verify that a corresponding Attribute Key Descriptor exists for the given key.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from memory which is never modified (so there is no
 *        data-dependency)
 *
 * @param[in] key: attribute key.
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums.
 * @param[out] keyAccess: LwSciBufKeyAccess.
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - keyAccess is NULL
 *      - key is not a valid enumeration value defined in the LwSciBufAttrKey,
 *        LwSciBufInternalAttrKey, LwSciBufPrivateAttrKey enums
 *
 * @implements{18843552}
*/
void LwSciBufAttrGetKeyAccessDetails(
     uint32_t key,
     LwSciBufKeyAccess* keyAccess);

/** @brief Get details of LwSciBufAttrKey or LwSciBufInternalAttrKey
 * or LwSciBufPrivateAttrKey from corresponding data store of
 * LwSciBufPerSlotAttrList for slotIndex which is retrieved from the buffer
 * LwSciBufAttrListObjPriv for which the input LwSciBufAttrList holds reference.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciBufAttrList prior to calling this function. This must be held
 *        at least until the output parameters are dereferenced and read.
 *
 * @param[in] attrList: LwSciBufAttrList from which the details of
 * LwSciBufAttrKey or LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey
 * should be retrieved.
 * @param[in] slotIndex: Index in the LwSciBufAttrList.
 * Valid value: 0 to slot count of LwSciBufAttrList - 1.
 * @param[in] key: attribute key.
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums.
 * @param[out] baseAddr: Base address of attribute key's value in corresponding
 * data store of LwSciBufPerSlotAttrLIst for slotIndex.
 * @param[out] status: LwSciBufAttrStatus of attribute key retrieved from static
 * storage for slotIndex.
 * @param[out] setLen: Size of attribute  key's value in bytes.
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - attrList is invalid
 *      - baseAddr is NULL
 *      - status is NULL
 *      - setLen is NULL
 *      - @a key is not a valid enumeration value defined by the LwSciBufAttrKey,
 *        LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums
 *
 * @implements{18843501}
 */
void LwSciBufAttrGetKeyDetail(
    LwSciBufAttrList attrList,
    size_t slotIndex,
    uint32_t key,
    void** baseAddr,
    LwSciBufAttrStatus** status,
    uint64_t** setLen);

/** @brief Creates a new LwSciBufAttrList referencing the same
 *  LwSciBufAttrListObjPriv as the input LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent access is provided via LwSciCommonDuplicateRef()
 *
 * @param[in] oldAttrList: LwSciBufAttrList for which new LwSciBufAttrList need
 * to be created.
 * @param[out] newAttrList: The new LwSciBufAttrList.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any the following oclwrs:
 *      - @a oldAttrList is NULL
 *      - @a newAttrList is NULL
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_IlwalidState if the number of LwSciBufAttrList
 *   referencing LwSciBufAttrListObjPriv are INT32_MAX and this API is
 *   called to create one more LwSciBufAttrList reference.
 * - Panics if oldAttrList is not valid.
 *
 * @implements{18843504}
 */
LwSciError LwSciBufAttrListDupRef(
    LwSciBufAttrList oldAttrList,
    LwSciBufAttrList* newAttrList);


/** @brief Gets LwSciBufModule associated with the LwSciBufAttrList from
 *  LwSciBufAttrListObjPriv referenced by the LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the LwSciBufModule
 *        associated with the LwSciBufAttrList is never modified after creation
 *        (so there is no data-dependency)
 *
 * @param[in] attrList: LwSciBufAttrList from which the LwSciBufModule
 * to be retrieved.
 * @param[out] module: Retrieved LwSciBufModule.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a module is NULL
 * - panics if @a attrList is not valid
 *
 * @implements{18843507}
 */
LwSciError LwSciBufAttrListGetModule(
    LwSciBufAttrList attrList,
    LwSciBufModule* module);

/**
 * @brief Initializes the given LwSciBufUmdAttrKeyIterator.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciBufAttrList prior to calling this function
 *
 * @param[in] attrList: LwsciBufAttrList from which the UMD specific
 * keys should be iterated.
 * @param[in] slotNum: Index in the LwSciBufAttrList.
 * Valid value: 0 to slot count of LwSciBufAttrList - 1.
 * @param[out] iter: Initialized LwSciBufUmdAttrKeyIterator.
 *
 * @return LwSciError, the completion status of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a slotNum >= slot count of LwSciBufAttrList.
 * - Panics if any of the following oclwrs:
 *      - @a attrList is NULL
 *      - @a attrList is not valid
 *      - @a iter is NULL
 *
 * @implements{18843516}
 */
LwSciError LwSciBufUmdAttrKeyIterInit(
    LwSciBufAttrList attrList,
    uint64_t slotNum,
    LwSciBufUmdAttrKeyIterator* iter);

/**
 * @brief Retrieves the UMD specific LwSciBufInternalAttrKey
 * lwrrently pointed by LwSciBufUmdAttrKeyIterator and update
 * the state of LwSciBufUmdAttrKeyIterator to point to the
 * next UMD specific LwSciBufInternalAttrKey.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciBufAttrList for which the LwSciBufUmdAttrKeyIterator was created
 *        for prior to calling this function
 *      - The user must ensure that the same input LwSciBufUmdAttrKeyIterator
 *        is not used by multiple threads at the same time
 *
 * @param[in] iter: LwSciBufUmdAttrKeyIterator. Valid value: iter is valid
 * input if it is not NULL and it was initialized using successful call to
 * LwSciBufUmdAttrKeyIterInit().
 * @param[out] keyEnd: Flag to indicate the iteration of UMD specific
 * LwSciBufInternalAttrKey(s) is completed.
 * @param[out] key: Output UMD specific LwSciBufInternalAttrKey.
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - @a iter is NULL
 *      - @a keyEnd is NULL
 *      - @a key is NULL
 *
 * @implements{18843519}
 */
void LwSciBufUmdAttrKeyIterNext(
    LwSciBufUmdAttrKeyIterator* iter,
    bool* keyEnd,
    uint32_t* key);

/**
 * @brief Compares reconciliation status of a LwSciBufAttrList
 * with the given @a isReconciled flag. Reconciliation status
 * is retrieved from the LwSciBufAttrListObjPriv for which
 * the given LwSciBufAttrList holds a reference.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Conlwrrent acces is provided via LwSciBufAttrListIsReconciled()
 *
 * @param[in] attrList LwSciBufAttrList for which the reconciliation
 * status to be checked.
 * @param[in] isReconciled Flag value to compare against the
 * reconciliation status of given LwSciBufAttrList.
 * Valid value: True or false.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success reconciliation status of @a attrList matches
 *   with @a isReconciled.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a attrList is NULL.
 *    - reconciliation status of @a attrList doesn't match with
 *      @a isReconciled.
 * - panics if @a attrList is not valid.
 *
 * @implements{18843522}
 */
LwSciError LwSciBufAttrListCompareReconcileStatus(
    LwSciBufAttrList attrList,
    bool isReconciled);

/**
 * @brief Decodes LwSciBufAttrKeyType, LwSciBufType and attribute key
 * identifier from the given LwSciBufAttrKey or LwSciBufInternalAttrKey
 * or LwSciBufPrivateAttrKey.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data provided via the input key
 *        parameter
 *
 * @param[in] key: Attribute key.
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey, or LwSciBufPrivateAttrKey enums.
 * @param[out] decodedKey: Decoded LwSciBufAttrKeyType.
 * @param[out] decodedDataType: Decoded LwSciBufType.
 * @param[out] decodedKeyType: Decoded key identifier.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a decodedKey is NULL
 *      - @a decodedDataType is NULL
 *      - @a decodedKeyType is NULL
 *      - key is not a valid enumeration value defined by the LwSciBufAttrKey,
 *        LwSciBufInternalAttrKey, LwSciBufPrivateAttrKey enums or is not a
 *        UMD-specific LwSciBufInternalAttrKey
 *
 * @implements{18843525}
 */
LwSciError LwSciBufAttrKeyDecode(
    uint32_t key,
    uint32_t* decodedKey,
    uint32_t* decodedDataType,
    uint32_t* decodedKeyType);

/**
 * @brief Allocates memory for datatype member of LwSciBufPerSlotAttrList
 * corresponding to @a slotIndex which is retrieved from the
 * LwSciBufAttrListObjPriv referenced by the input LwSciBufAttrList. This memory
 * is used to store the values and LwSciBufAttrStatus of LwSciBufAttrKey(s) and
 * LwSciBufInternalAttrKey(s) for required LwSciBufType(s) by the
 * LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciBufAttrList prior to calling this function
 *
 * @param[in] attrList: LwSciBufAttrList for which the memory should be
 * allocated.
 * @param[in] slotIndex: Index of LwSciBufAttrList.
 * Valid value: 0 to slot count of LwSciBufAttrList - 1.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a attrList is NULL
 * - LwSciError_InsufficientMemory is memory allocation is failed
 * - Panics if any of the following oclwrs:
 *      - @a attrList is not valid.
 *      - @a slotIndex >= slot count of LwSciBufAttrList.
 *
 * @implements{18843528}
 */
LwSciError LwSciBufAttrListMallocBufferType(
    LwSciBufAttrList attrList,
    size_t slotIndex);

/**
 * @brief Retrieves the LwSciBufType(s) set to the LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on the input LwSciBufAttrList to serialize reads
 *      - Locks are held for the duration of any operations on the input
 *        LwSciBufAttrList
 *      - Locks are released when all operations on the input LwSciBufAttrList
 *        are complete
 *
 * @param[in] attrList: LwSciBufAttrList from which the LwSciBufType(s)
 * should be retrieved.
 * @param[out] bufType: The LwSciBufType(s).
 * @param[out] numDataTypes: Number of elements/entries in @a bufType.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *    - @a attrList is NULL
 *    - @a bufType is NULL
 *    - @a numDataTypes is NULL
 * - panics if @a attrList is not valid.
 *
 * @implements{18843531}
 */
LwSciError LwSciBufAttrListGetDataTypes(
    LwSciBufAttrList attrList,
    const LwSciBufType** bufType,
    size_t* numDataTypes);

/**
 * @brief Gets LwSciBufIpcRouteAffinity from attribute
 * key descriptor of the given LwSciBufAttrKey or
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey.
 *
 * @param[in] key attribute key.
 * Valid value LwSciBufAttrKey_LowerBound < key < LwSciBufAttrKey_UpperBound or
 * LwSciBufInternalAttrKey_LowerBound < key <=
 * LwSciBufInternalAttrKey_LwMediaPrivateFirst or
 * LwSciBufPrivateAttrKey_LowerBound < key <=
 * LwSciBufPrivateAttrKey_ConflictKey.
 * @param[in] localPeer boolean value indicating if the LwSciBufIpcRouteAffinity
 * is obtained for the local peer or the remote peer.
 * @param[out] routeAffinty LwSciBufIpcRouteAffinity of the @a key.
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - @a routeAffinity is NULL.
 *      - @a key could not be decoded to get the attribute key descriptor.
 *
 * @implements{}
 */
void LwSciBufAttrKeyGetIpcRouteAffinity(
    uint32_t key,
    bool localPeer,
    LwSciBufIpcRouteAffinity* routeAffinity);

/**
 * @brief Gets LwSciBuf_ReconcilePolicy from attribute
 * key descriptor of the given LwSciBufAttrKey or
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from memory which is never modified (so there is no
 *        data-dependency)
 *
 * @param[in] key: attribute key.
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums.
 * @param[out] policy: LwSciBuf_ReconcilePolicy of the @a key.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - Panics if any of the following oclwrs:
 *    - @a key is invalid
 *    - @a policy is NULL
 *
 * @implements{18843534}
 */
void LwSciBufAttrKeyGetPolicy(
    uint32_t key,
    LwSciBuf_ReconcilePolicy* policy);

/**
 * @brief Sorts the given unreconciled LwSciBufAttrList(s) by their addresses
 * using LwSciCommonSort() and locks the LwSciBufAttrListRec(s) to
 * the LwSciBufAttrListObjPriv held by the LwSciBufAttrList(s) using
 * LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Locks are taken on each LwSciBufAttrList
 *      - Locks are held by the caller until explicitly unlocked after calling
 *        this function, ideally via LwSciBufAttrListsUnlock()
 *
 * @param[in] inputAttrListArr: Array of unreconciled LwSciBufAttrList(s).
 * Valid value: NULL is valid value provided @a attrListCount is 0.
 * @param[in] attrListCount: Number of elements/entries in @a inputAttrListArr.
 * Valid value: attrListCount is valid input if it is >= 0.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - any LwSciBufAttrList(s) in @a inputArray is NULL
 *      - not all the LwSciBufAttrList(s) in @a inputAttrListArr are bound to
 *        the same LwSciBufModule instance
 *      - any of the LwSciBufAttrList(s) in @a inputAttrListArr is reconciled
 * - LwSciError_Overflow if size of the memory to be allocated
 *   exceeds SIZE_MAX.
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - panics if any of the following oclwrs:
 *      - inputAttrListArr is NULL.
 *      - attrListCount is 0.
 *      - any of the LwSciBufAttrList(s) in inputAttrListArr is invalid.
 *
 * @implements{18843540}
 */
LwSciError LwSciBufAttrListsLock(
    const LwSciBufAttrList inputAttrListArr[],
    size_t attrListCount);

/**
 * @brief Unlocks the LwSciBufAttrListRec(s) to the LwSciBufAttrListObjPriv
 * held by the unreconciled LwSciBufAttrList(s) using LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on each
 *        LwSciBufAttrList prior to calling this function, ideally via
 *        LwSciBufAttrListsLock()
 *      - Locks are released on each LwSciBufAttrList
 *
 * @param[in] inputAttrListArr: Array of unreconciled LwSciBufAttrList(s).
 * Valid value: NULL is valid value provided @a attrListCount is 0.
 * @param[in] attrListCount: Number of elements/entries in @a inputAttrListArr.
 * Valid value: attrListCount is valid input if it is >= 0.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - any LwSciBufAttrList(s) in @a inputArray is NULL
 *      - not all the LwSciBufAttrList(s) in @a inputAttrListArr are bound to
 *        the same LwSciBufModule instance
 *      - any of the LwSciBufAttrList(s) in @a inputAttrListArr is reconciled
 * - panics if any of the following oclwrs:
 *      - inputAttrListArr is NULL.
 *      - attrListCount is 0.
 *      - any of the LwSciBufAttrList(s) in inputAttrListArr is invalid.
 *
 * @implements{18843543}
 */
LwSciError LwSciBufAttrListsUnlock(
    const LwSciBufAttrList inputAttrListArr[],
    size_t attrListCount);

/**
 * @brief Creates a new unreconciled LwSciBufAttrList using
 * LwSciBufAttrListCreateMultiSlot() by passing @a emptyIpcRoute as false and
 * with the summed up slot count and clones the input unreconciled
 * LwSciBufAttrList(s) to the new unreconciled LwSciBufAttrList slot by slot.
 * Additionally it provides an option to lock and unlock the
 * LwSciBufAttrListRec(s) to LwSciBufAttrListObjPriv(s) held by the input
 * unreconciled LwSciBufAttrList(s) before and after appending them to a new
 * unreconciled LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes, provided @a acquireLocks is true
 *  - Synchronization
 *      - Locks are taken on each LwSciBufAttrList (if @a acquireLocks is true)
 *      - Locks are held for the duration of any operations on each
 *        LwSciBufAttrList (if @a acquireLocks is true)
 *      - Locks are released when all operations on the LwSciBufAttrList are
 *        complete (if @a acquireLocks is true)
 *      - Conlwrrent access to the LwSciBufModule associated with the
 *        LwSciBufAttrList(s) is provided via LwSciBufAttrListCreateMultiSlot()
 *      - Conlwrrent access must be serialized by taking the lock on each
 *        LwSciBufAttrList prior to calling this function if @a acquireLocks is
 *        false
 *
 * @param[in] inputUnreconciledAttrListArray: Array of LwSciBufAttrList(s).
 * Valid value: inputUnreconciledAttrListArray is valid input if it is not NULL.
 * @param[in] inputUnreconciledAttrListCount: Number of elements/entries in @a
 * inputUnreconciledAttrListArray.
 * Valid value: inputUnreconciledAttrListCount is valid input if it is non-zero.
 * @param[in] acquireLocks: Flag to indicate whether the LwSciBufAttrList(s)
 * from @a inputUnreconciledAttrListArray should be locked or not before
 * appending them to a new unreconciled LwSciBufAttrList.
 * Valid value: True or false.
 *
 * @param[out] newUnreconciledAttrList: Appended unreconciled LwSciBufAttrList.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful.
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a inputUnreconciledAttrListArray is NULL
 *      - @a inputUnreconciledAttrListCount is 0
 *      - @a newUnreconciledAttrList is NULL
 *      - all of the LwSciBufAttrLists do not belong to same LwSciBufModule
 *      - if any of the LwSciBufAttrList is reconciled
 *      - the LwSciBufGeneralAttrKey_Types key is not set on any of the
 *        LwSciBufAttrList(s) in @a inputUnreconciledAttrListArray
 * - LwSciError_InsufficientMemory if memory allocation failed.
 * - LwSciError_IlwalidState if a new LwSciBufAttrList cannot be associated
 *   with the LwSciBufModule associated with the LwSciBufAttrList(s) in the
 *   given @a inputUnreconciledAttrListArray to create the new LwSciBufAttrList.
 * - LwSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a any LwSciBufAttrList in the @a
 *   inputUnreconciledAttrListArray is invalid
 *
 * @implements{18843546}
 */
LwSciError LwSciBufAttrListAppendWithLocksUnreconciled(
    const LwSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool acquireLocks,
    LwSciBufAttrList* newUnreconciledAttrList);

/**
 * @brief Sets LwSciBufAttrListState of the LwSciBufAttrList.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciBufAttrList prior to calling this function
 *
 * @param[in] attrList: The LwSciBufAttrList whose state is being set.
 * @param[in] state: The LwSciBufAttrListState we assign to @a attrList
 * Valid value:  state is valid input if it is equal to
 * LwSciBufAttrListState_Unreconciled or  LwSciBufAttrListState_Reconciled or
 * LwSciBufAttrListState_Appended.
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any of the following oclwrs:
 *      - @a attrList is NULL
 * - panics if @a state is greater than or equal to
 *   LwSciBufAttrListState_UpperBound.
 * - panics if @a attrList is invalid.
 *
 * @implements{18843549}
 */
LwSciError LwSciBufAttrListSetState(
    LwSciBufAttrList attrList,
    LwSciBufAttrListState state);

/**
 * @brief Determine whether a key needs to be present during import of a
 * Reconciled LwSciBufAttrList. This function assumes that validation has
 * already oclwrred to verify that a corresponding Attribute Key Descriptor
 * exists for the given key.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciBufAttrList prior to calling this function
 *
 * @param[in] reconciledList: The Reconciled LwSciBufAttrList to check
 * @param[in] key: The key to determine whether Import Checking is necessary for
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums.
 * @param[out] result: Whether checking is needed
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any one of the following oclwrs:
 *      - @a reconciledList is NULL
 * - Panics if any of the following oclwrs:
 *      - @a key is not a valid enumeration value defined by the
 *        LwSciBufAttrKey, LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey
 *        enums
 *      - @a reconciledList is invalid
 *      - @a result is NULL
 *
 * @implements{19731888}
 */
LwSciError LwSciBufImportCheckingNeeded(
    LwSciBufAttrList reconciledList,
    uint32_t key,
    bool *result);

/**
 * @brief Determine whether a key needs to be present during reconciliation of
 * Unreconciled LwSciBufAttrList(s). This function assumes that validation has
 * already oclwrred to verify that a corresponding Attribute Key Descriptor
 * exists for the given key.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciBufAttrList prior to calling this function
 *
 * @param[in] reconciledList: The Reconciled LwSciBufAttrList to check
 * @param[in] key: The key to determine whether Import Checking is necessary for
 * Valid value: key is a valid enumeration value defined by the LwSciBufAttrKey,
 * LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey enums.
 * @param[out] result: Whether checking is needed
 *
 * @return LwSciError, the completion code of the operation:
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if any one of the following oclwrs:
 *      - @a reconciledList is NULL
 * - Panics if any of the following oclwrs:
 *      - @a key is not a valid enumeration value defined by the
 *        LwSciBufAttrKey, LwSciBufInternalAttrKey or LwSciBufPrivateAttrKey
 *        enums
 *      - @a reconciledList is invalid
 *      - @a result is NULL
 *
 * @implements{19731891}
 */
LwSciError LwSciBufReconcileCheckingNeeded(
    LwSciBufAttrList reconciledList,
    uint32_t key,
    bool *result);

//TODO: Add doxygen comments for JAMA SWAD/SWUD. Also note that, this function
//adds dependency of attribute core unit on constraints library. This dependency
//needs to be reflected in section 4.4.1 in SWAD
LwSciError LwSciBufAttrListIsIsoEngine(
    LwSciBufAttrList attrList,
    bool* isIsoEngine);

/**
 * @brief Callback to free the data associated with the LwSciObj representing
 * the underlying LwSciBufAttrListObjPriv using LwSciCommon functionality.
 *
 * Conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - Conlwrrent access must be serialized by taking the lock on the
 *        LwSciObj prior to calling this function
 *      - Conlwrrent access to the LwSciBufModule is handled via
 *        LwSciBufModuleClose()
 *
 * @param[in] attrListPtr Pointer to the LwSciObj associated with the
 * LwSciBufAttrList to free
 *
 * @return void
 * - Panics if any of the following oclwrs:
 *      - attrListPtr is NULL
 *      - attrListPtr is invalid
 *
 * @implements{22034423}
 */
void LwSciBufAttrCleanupCallback(
    LwSciObj* attrListPtr);

/**
 * @brief Callback used to compare two pointers to LwSciBufAttrLists to
 * determine a total ordering when sorting.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - Reads only occur from immutable data since the location of LwSciRef
 *        identifying the LwSciBufAttrList is never modified after creation
 *        (so there is no data-dependency)
 *
 * @param[in] elem1 The first LwSciBufAttrList to compare
 * @param[in] elem2 The second LwSciBufAttrList to compare
 *
 * @return int32_t
 *  - 1 if the LwSciBufAttrList represented by elem1 is greater than elem2 in
 *  the total ordering
 *  - 0 if the LwSciBufAttrLists are identical
 *  - -1 otherwise
 *  - Panics if any of the following oclwrs:
 *      - elem1 is NULL
 *      - elem2 is NULL
 *      - elem1 is invalid
 *      - elem2 is invalid
 *
 * @implements{22034432}
 */
int32_t LwSciBufAttrListCompare(
    const void* elem1,
    const void* elem2);

/**
 * @brief Callback used to compare two pointers to uint32_ts representing
 * potential LwSciBufAttrKeys, LwSciBufInternalAttrKeys, LwSciBufPrivateAttrKeys
 * or UMD-specific LwSciBufInternalAttrKeys. This does not perform any
 * additional validation that the uint32_ts represented by elem1 and elem2
 * actually correspond to valid attribute keys and simply treats them as
 * uint32_t values.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - This is a pure function with no side-effects
 *
 * @param[in] elem1 The first uint32_t representing an attribute key to compare
 * @param[in] elem2 The second uint32_t representing an attribute key to compare
 *
 * @return int32_t
 *  - 1 if the uint32_t represented by elem1 is greater than elem2 in
 *  the total ordering
 *  - 0 if the uint32_ts are identical
 *  - -1 otherwise
 *  - Panics if any of the following oclwrs:
 *      - elem1 is NULL
 *      - elem2 is NULL
 *
 * \implements{22034437}
 */
int32_t LwSciBufAttrKeyCompare(
    const void* elem1,
    const void* elem2);

#endif /* INCLUDED_LWSCIBUF_ATTR_H */
