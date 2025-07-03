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
 * \brief <b>LwSciSync private core attribute definitions</b>
 *
 * @b Description: This file declares basic core attribute structures and
 * interfaces to be used by attribute units
 */

#ifndef INCLUDED_LWSCISYNC_ATTRIBUTE_CORE_PRIVATE_H
#define INCLUDED_LWSCISYNC_ATTRIBUTE_CORE_PRIVATE_H

/**
 * @defgroup lwsci_sync Synchronization APIs
 *
 * @ingroup lwsci_group_stream
 * @{
 */

#include "lwscicommon_objref.h"
#include "lwscicommon_covanalysis.h"
#include "lwscisync_core.h"
#include "lwscisync_ipc_table.h"
#include "lwscisync_internal.h"

/* Helper macros */

/**
 * \brief Number of all supported LwSciSyncAttrKeys
 *
 *  \implements{18845796}
 */
#define PUBLIC_KEYS_COUNT ((size_t)LwSciSyncAttrKey_UpperBound - \
        (size_t)LwSciSyncAttrKey_LowerBound - 1U)

/**
 * \brief Number of all supported LwSciSyncInternalAttrKeys
 *
 *  \implements{18845799}
 */
#define INTERNAL_KEYS_COUNT ((size_t)LwSciSyncInternalAttrKey_UpperBound - \
        (size_t)LwSciSyncInternalAttrKey_LowerBound - 1U)

/**
 * \brief Number of supported LwSciSyncAttrKeys and LwSciSyncInternalAttrKeys
 *
 *  \implements{18845805}
 */
#define KEYS_COUNT (PUBLIC_KEYS_COUNT + INTERNAL_KEYS_COUNT)

/**
 * \brief Translates an internal/public key enum to a unified index.
 *  Unified indices start with 0, go over all public keys,
 *  and then over internal keys. x is assumed to be a valid key.
 **/
#define KEY_TO_INDEX(x)                                            \
    ((((size_t)x) < (size_t)LwSciSyncInternalAttrKey_LowerBound) ? \
     (((size_t)x) - (size_t)LwSciSyncAttrKey_LowerBound - 1U) :    \
     (((size_t)x) - (size_t)LwSciSyncInternalAttrKey_LowerBound +  \
      (size_t)LwSciSyncAttrKey_UpperBound - 2U))

/**
 * \brief Translates a unified index to int32_t whose value equals to
 *  an internal/public key enum.
 *  This is a reverse operation of KEY_TO_INDEX(). x is assumed to be
 *  a valid unified index.
 **/
#define INDEX_TO_KEY(x)                                         \
    ((((size_t)x) < ((size_t)LwSciSyncAttrKey_UpperBound - 1U)) ?           \
     (uint32_t)(((uint32_t)x) + (uint32_t)LwSciSyncAttrKey_LowerBound + 1U) :  \
     (uint32_t)(((uint32_t)x) + (uint32_t)LwSciSyncInternalAttrKey_LowerBound \
      - (uint32_t)LwSciSyncAttrKey_UpperBound + 2U))


/**
 * \brief Translates an LwSciSyncAttrKey/LwSciSyncInternalAttrKey enum to a
 * unified index.
 * Unified indices start with 0, go over all LwSciSyncAttrKey keys, and then
 * over LwSciSyncInternalAttrKey. Key is assumed to be within bounds.
 * \param[in] key LwSciSyncAttrKey/LwSciSyncInternalAttrKey to be translated.
 * Valid value: key is valid input if it is a valid
 * LwSciSyncAttrKey/LwSciSyncInternalAttrKey enum
 * \return size_t
 * Unified index
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
 * \implements{18844260}
 **/
static inline size_t LwSciSyncCoreKeyToIndex(uint32_t key) {
    if(0U == key) {
        LwSciCommonPanic();
    }
    return KEY_TO_INDEX(key);
}

/**
 * \brief Translates a unified index to uint32_t whose value equals to
 *  an LwSciSyncAttrKey/LwSciSyncInternalAttrKey enum.
 *  Index is assumed to be within bounds.
 * \param[in] index value to be translated
 * Valid value: [0, KEYS_COUNT-1]
 * \return size_t
 * Enum's value corresponding to the input index
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
 * \implements{18844263}
 **/
static inline uint32_t LwSciSyncCoreIndexToKey(size_t index) {
    if(KEYS_COUNT <= index) {
        LwSciCommonPanic();
    }
    return INDEX_TO_KEY(index);
}

/**
 * \brief States of LwSciSyncAttrList.
 *
 *  State of LwSciSyncAttrList is immutable and depends on how
 *  it was created. Each function returning a new LwSciSyncAttrList
 *  specifies what is the state of the new LwSciSyncAttrList.
 *
 *  Some functions require that a provided LwSciSyncAttrList is of a certain
 *  state.
 *
 *  \implements{18845766}
 */
typedef enum {
    /** Represents unreconciled LwSciSyncAttrList */
    LwSciSyncCoreAttrListState_Unreconciled = 1,
    /** Represents reconciled LwSciSyncAttrList */
    LwSciSyncCoreAttrListState_Reconciled,
#if (LW_IS_SAFETY == 0)
    /** Represents conflict LwSciSyncAttrList */
    LwSciSyncCoreAttrListState_Conflict,
#endif
} LwSciSyncCoreAttrListState;

/**
 * \brief States of attributes in an LwSciSyncAttrList slot.
 *
 * \implements{18845769}
 */
typedef enum {
    /** Attribute not set. This means that the attribute value is undefined. */
    LwSciSyncCoreAttrKeyState_Empty,
    /** Cannot update the attribute. This is legal only when
     * LwScisyncCoreAttrListState is unreconciled. */
    LwSciSyncCoreAttrKeyState_SetLocked,
    /** Can update the attribute. This is legal only when
     * LwScisyncCoreAttrListState is unreconciled. */
    LwSciSyncCoreAttrKeyState_SetUnlocked,
    /** Attribute is reconciled. This is legal only when
     * LwScisyncCoreAttrListState is reconciled. */
    LwSciSyncCoreAttrKeyState_Reconciled,
    /** Attribute conflicts in the Conflict LwSciSyncAttrList. This is legal
     * only when LwScisyncCoreAttrListState is conflict. */
    LwSciSyncCoreAttrKeyState_Conflict,
} LwSciSyncCoreAttrKeyState;

/***
 * Core structures declaration.
 */

typedef struct {
    size_t index;
    const void* value;
    size_t len;
} CoreAttribute;

typedef struct {
    /** SoC Id */
    uint32_t socId;
    /** Virtual Machine Id */
    uint32_t vmId;
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 2_3), "LwSciSync-ADV-MISRAC2012-002")
} LwSciSyncCoreAttrValTopoId;

#ifdef LWSCISYNC_EMU_SUPPORT
/**
 * @brief Union of various supported primitive info.
 */
typedef union {
    /** Describes a simple external primitive. */
    LwSciSyncSimplePrimitiveInfo simplePrimitiveInfo;
    /** Describes external semaphore primitive. */
    LwSciSyncSemaphorePrimitiveInfo semaphorePrimitiveInfo;
} LwSciSyncPrimitiveInfo;
#endif

/**
 * \brief Structure gathering all supported attributes and their
 * LwSciSyncCoreAttrKeyStates.
 *
 * \implements{18845772}
 */
typedef struct {
    /*** Public attributes: */
    /**
     * Value for LwSciSyncAttrKey_NeedCpuAccess public key.
     * Valid value for this member is true/false.
     */
    bool needCpuAccess;
    /**
     * Value for LwSciSyncAttrKey_RequiredPerm public key.
     * This member is not set in the reconciled LwSciSyncAttrList.
     * Valid value for this member is all the values defined by
     * LwSciSyncAccessPerm enum except for LwSciSyncAccessPerm_Auto.
     */
    LwSciSyncAccessPerm requiredPerm;
    /**
     * Value for LwSciSyncAttrKey_ActualPerm public key.
     * This member is set only in reconciled LwSciSyncAttrList.
     * Valid value for this member is all the values defined by
     * LwSciSyncAccessPerm enum except for LwSciSyncAccessPerm_Auto.
     * This member cannot be modified by application directly, however apps
     * can read the value.
     */
    LwSciSyncAccessPerm actualPerm;
    /**
     * Value for LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports public
     * key.
     * Valid value for this member is true/false.
     */
    bool waiterContextInsensitiveFenceExports;
    /** Value for LwSciSyncAttrKey_WaiterRequireTimestamps public key */
    bool waiterRequireTimestamps;
    /**
     * Value for LwSciSyncAttrKey_RequireDeterministicFences public key. This
     * is set to true in the reconciled LwSciSyncAttrList if any one of the
     * input LwSciSyncAttrList has this set to true.
     * Valid value for this member is true/false.
     */
    bool requireDeterministicFences;

    /*** Internal attributes: */
    /**
     * Value for LwSciSyncInternalAttrKey_SignalerPrimitiveInfo internal
     * key.
     * Only valid entries are those having value greater than
     * LwSciSyncInternalAttrValPrimitiveType_LowerBound and less than
     * LwSciSyncInternalAttrValPrimitiveType_UpperBound.
     */
    LwSciSyncInternalAttrValPrimitiveType signalerPrimitiveInfo[
            MAX_PRIMITIVE_TYPE];
    /**
     * Value for LwSciSyncInternalAttrKey_WaiterPrimitiveInfo internal
     * key. Value of this member in reconciled LwSciSyncAttrList will be same
     * as that of signalerPrimitiveInfo.
     * Only valid entries are those having value greater than
     * LwSciSyncInternalAttrValPrimitiveType_LowerBound and less than
     * LwSciSyncInternalAttrValPrimitiveType_UpperBound.
     */
    LwSciSyncInternalAttrValPrimitiveType waiterPrimitiveInfo[
            MAX_PRIMITIVE_TYPE];
    /**
     * Value for LwSciSyncInternalAttrKey_SignalerPrimitiveCount internal
     * key.
     * Value value range for this member is [0, UINT32_MAX]
     */
    uint32_t signalerPrimitiveCount;
    /** Value for LwSciSyncInternalAttrKey_GpuId internal key. */
    LwSciRmGpuId gpuId;
    /** Value for LwSciSyncInternalAttrKey_SignalerTimestampInfo internal key.*/
    LwSciSyncAttrValTimestampInfo signalerTimestampInfo;

    /** Value for LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti internal key.*/
    LwSciSyncAttrValTimestampInfo signalerTimestampInfoMulti[MAX_PRIMITIVE_TYPE];
#ifdef LWSCISYNC_EMU_SUPPORT
    /** Value for LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo key.*/
    LwSciSyncPrimitiveInfo* signalerExternalPrimitiveInfo[MAX_PRIMITIVE_TYPE];
#endif

    /** Value for LwSciSyncInternalAttrKey_EngineArray internal key */
    LwSciSyncHwEngine engineArray[MAX_HW_ENGINE_TYPE];

    /*** Meta data for attributes: */
    /** Attributes' states in this slot. Each entry in this array is mapped to
     * members of this structure using LwSciSyncCoreKeyInfo variable. All the
     * members of this array are initialized to LwSciSyncCoreAttrKeyState_Empty */
    LwSciSyncCoreAttrKeyState keyState[KEYS_COUNT];
    /** Length of the attribute value for attributes whose keyState is not
     * empty. This is undefined if keyState is empty. Each entry in this array
     * is mapped to members of this structure using LwSciSyncCoreKeyInfo
     * variable. */
    uint64_t valSize[KEYS_COUNT];
} LwSciSyncCoreAttrs;

/**
 * \brief Represents a slot of LwSciSyncAttrList.
 *
 * \implements{18845775}
 */
typedef struct {
    /** Specifies whether the Signaler will use an externally provided
     * Primitive or allocate one upon LwSciSyncObj allocation.
     * It is initialized to false when a new LwSciSyncAttrList is created.
     */
    bool signalerUseExternalPrimitive;
    /** LwSciIpcEndpoint this list was exported through most recently if it
     * was a C2c ipcEndpoint. Reconciled to lastExport of the signaler.
     */
    LwSciIpcEndpoint lastExport;
    /** Structure gathering all supported attributes and their states.
     * Attributes and its state is initialized to unset  when a new
     * LwSciSyncAttrList is created. */
    LwSciSyncCoreAttrs attrs;
    /** LwSciSync object's tree topology info
     * ipcTable is NULL to represent an unreconciled LwSciSyncAttrList that has
     * not been received over an LwSciIpc channel. A reconciled LwSciSyncAttrList
     * that has not been exported yet has this non-NULL. This member is populated upon
     * reconciliation using function calls like LwSciSyncCoreIpcTableTreeAlloc(),
     * LwSciSyncCoreIpcTableAddBranch() etc. This is freed using
     * LwSciSyncCoreIpcTableFree(). This is exported in export descriptor using
     * LwSciSyncCoreExportIpcTable() and imported in new LwSciSyncCoreIpcTable
     * using LwSciSyncCoreImportIpcTable() */
    LwSciSyncCoreIpcTable ipcTable;
    /** LwSciBuf attr list needed for semaphore allocation */
    LwSciBufAttrList semaAttrList;
    /** LwSciBuf attr list for the timestamp buffer */
    LwSciBufAttrList timestampBufAttrList;
} LwSciSyncCoreAttrList;

/**
 * \brief Actual structure that contains data corresponding to the LwSciSyncAttrList.
 * More than one LwSciSyncAttrList referencing to
 * the same LwSciSyncCoreAttrListObj is possible only when the LwSciSyncAttrList is
 * reconciled.
 *
 * \implements{18845778}
 */
typedef struct {
    /** Reference target for this LwSciSyncCoreAttrListObj. */
    LwSciObj objAttrList;
    /** Magic ID that is used to detect cases where pointer is
     * not of the right type. This is initialized
     * with a bit manipulated value generated from address of LwSciSyncAttrList,
     * and explicitly set to 0 when this structure is freed. */
    uint64_t header;
    /** LwSciSyncModule which LwSciSyncAttrList belongs to.
     * This module is duplicated during creation of LwSciSyncAttrList, and has
     * to be explicitly freed when this structure is destroyed.
     * This maps to the module, to which the LwSciSyncAttrListRec associated with
     * this LwSciSyncCoreAttrListObj is bound to. */
    LwSciSyncModule module;
    /** Pointer to array of private LwSciSyncCoreAttrList structures.
     * The number of elements in this array equals numCoreAttrList.
     * Memory is allocated during creation and freed upon destroying.
     * If the LwSciSyncAttrList associated with this
     * LwSciSyncCoreAttrListObj is empty,
     * then all the LwSciSyncCoreAttrList structs have no keys set.
     * Each entry in this array represents a single slot of the
     * LwSciSyncAttrList. */
    LwSciSyncCoreAttrList* coreAttrList;
    /** Number of LwSciSyncCoreAttrList in coreAttrList.
     * This is set at least to 1 or whatever is requested during creation of
     * LwSciSyncAttrList associated with this LwSciSyncCoreAttrListObj.
     * This is the number of slots in the LwSciSyncAttrList associated with this
     * LwSciSyncCoreAttrListObj (For example: numCoreAttrList is 1 for
     * single-slot LwSciSyncAttrList). */
    size_t numCoreAttrList;
    /** State of the LwSciSyncAttrList associated with this
     * LwSciSyncCoreAttrListObj. Initialized to
     * LwSciSyncCoreAttrListState_Unreconciled during creation and
     * LwSciSyncCoreAttrListState_Reconciled during reconciliation. */
    LwSciSyncCoreAttrListState state;
    /** Flag indicating whether or not the values in the LwSciSyncCoreAttrList
     * structures pointed to by coreAttrList may be modified.
     * True indicates writable and false otherwise.
     * This maps to whether LwSciSyncAttrList associated with this
     * LwSciSyncCoreAttrListObj is writable.
     * Initialized to true during local creation and to false during import
     * and reconciliation */
    bool writable;
} LwSciSyncCoreAttrListObj;

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciSync-ADV-MISRAC2012-001")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
static inline LwSciSyncCoreAttrListObj*
    LwSciCastObjToBufAttrListObjPriv(LwSciObj* arg)
{
    return (LwSciSyncCoreAttrListObj*)(void*)((char*)(void*)arg
        - LW_OFFSETOF(LwSciSyncCoreAttrListObj, objAttrList));
}
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

/**
 * \brief Structure that LwSciSyncAttrList actually points to.
 * It is a referencing framework wrapper around LwSciSyncCoreAttrListObj
 * which holds actual data.
 */
struct LwSciSyncAttrListRec {
    /** Holds reference to the LwSciSyncCoreAttrListObj structure.
     * This structure is allocated and freed using LwSciCommon object
     * referencing framework. */
    LwSciRef refAttrList;
};

/**
 * \brief Structure containing information about LwSciSyncAttrKeys and
 * LwSciSyncInternalAttrKeys.
 * This information is needed for  Set/Get of public and internal keys.
 *
 * \implements{18845781}
 */
typedef struct {
    /** Offset of LwSciSyncAttrKey or LwSciSyncAttrInternalKey's value
     * in LwSciSyncCoreAttrList structure */
    size_t offset;
    /** Size of the datatype corresponding to LwSciSyncAttrKey or
     * LwSciSyncAttrInternalKey's value */
    size_t elemSize;
    /** Max number of elements in the value corresponding to
     * LwSciSyncAttrKey or LwSciSyncAttrInternalKey */
    size_t maxElements;
    /** Indicates if LwSciSyncAttrKey or LwSciSyncAttrInternalKey
     * is read only or read/write */
    bool writable;
} LwSciSyncCoreAttrKeyInfo;

#define SET_KEY_INFO(ELEMENT, NUM, WRITABLE)                            \
    {((size_t)&(((STRUCT*)0)->ELEMENT)),                                \
     (sizeof(((STRUCT*)0)->ELEMENT) / (NUM)), (NUM), (bool)(WRITABLE)}

#define STRUCT LwSciSyncCoreAttrs

/**
 * \brief Metadata of attributes. It helps manage serialization of attributes,
 * sanity checking and similar properties.
 * The array is indexed by unified indices obtained by passing LwSciSyncAttrKey
 * or LwSciSyncInternalAttrKey value to LwSciSyncCoreKeyToIndex().
 *
 * \implements{18845784}
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 5_9), "LwSciSync-ADV-MISRAC2012-008")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_9), "LwSciSync-ADV-MISRAC2012-010")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_4), "LwSciSync-ADV-MISRAC2012-012")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
static const
LwSciSyncCoreAttrKeyInfo LwSciSyncCoreKeyInfo[KEYS_COUNT] =
{
   [KEY_TO_INDEX(LwSciSyncAttrKey_NeedCpuAccess)] =
           SET_KEY_INFO(needCpuAccess, 1U, true),
   [KEY_TO_INDEX(LwSciSyncAttrKey_RequiredPerm)] =
           SET_KEY_INFO(requiredPerm, 1U, true),
   [KEY_TO_INDEX(LwSciSyncAttrKey_ActualPerm)] =
           SET_KEY_INFO(actualPerm, 1U, false),
   [KEY_TO_INDEX(LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports)]=
           SET_KEY_INFO(waiterContextInsensitiveFenceExports, 1U, true),
   [KEY_TO_INDEX(LwSciSyncAttrKey_WaiterRequireTimestamps)]=
           SET_KEY_INFO(waiterRequireTimestamps, 1U, true),
   [KEY_TO_INDEX(LwSciSyncAttrKey_RequireDeterministicFences)] =
           SET_KEY_INFO(requireDeterministicFences, 1U, true),
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_SignalerPrimitiveInfo)] =
           SET_KEY_INFO(signalerPrimitiveInfo, MAX_PRIMITIVE_TYPE, true),
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_WaiterPrimitiveInfo)] =
           SET_KEY_INFO(waiterPrimitiveInfo, MAX_PRIMITIVE_TYPE, true),
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_SignalerPrimitiveCount)] =
           SET_KEY_INFO(signalerPrimitiveCount, 1U, true),
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_GpuId)] =
           SET_KEY_INFO(gpuId, 1U, true),
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_SignalerTimestampInfo)] =
           SET_KEY_INFO(signalerTimestampInfo, 1U, true),
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti)] =
           SET_KEY_INFO(signalerTimestampInfoMulti, MAX_PRIMITIVE_TYPE, true),
#ifdef LWSCISYNC_EMU_SUPPORT
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo)] =
           SET_KEY_INFO(signalerExternalPrimitiveInfo, MAX_PRIMITIVE_TYPE, true),
#endif
   [KEY_TO_INDEX(LwSciSyncInternalAttrKey_EngineArray)] =
           SET_KEY_INFO(engineArray, MAX_HW_ENGINE_TYPE, true),
};
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 5_9))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_9))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_4))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

/**
 * \brief Retrieves LwSciSyncCoreAttrListObj referenced
 * by the input LwSciSyncAttrList.
 *
 * It is a wrapper function on LwSciCommonGetObjFromRef
 * to avoid type-casting to void* at multiple places.
 *
 * \param[in] attrList LwSciSyncAttrList from which the LwSciSyncCoreAttrListObj
 * should be retrieved
 * \param[out] objAttrList LwSciSyncCoreAttrListObj
 * \return void
 * - Panics if @a attrList is not a valid LwSciSyncAttrList
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
 * \implements{18844266}
 */
static inline void LwSciSyncCoreAttrListGetObjFromRef(
    LwSciSyncAttrList attrList,
    LwSciSyncCoreAttrListObj** objAttrList)
{
    LwSciObj* objAttrListParam = NULL;
    LwSciCommonGetObjFromRef(&attrList->refAttrList, &objAttrListParam);
    *objAttrList = LwSciCastObjToBufAttrListObjPriv(objAttrListParam);
}

/**
 * \brief Returns a value of LwSciSyncAttrKey or LwSciSyncAttrInternalKey
 * with a unified index keyIdx in the input LwSciSyncCoreAttrList.
 *
 * If the attribute was not set, the resulting pointer will point to the
 * zero-initialized memory prepared for the attribute's value.
 * The function does no validation and assumes the keyIdx is a valid
 * unified key index.
 *
 * \param[in] coreAttrList LwSciSyncCoreAttrList.
 * Valid value: coreAttrList is valid if it is non-NULL
 * \param[in] keyIdx unified key index.
 * Valid value: [0, KEYS_COUNT-1]
 * \return void*
 * - pointer to the place where the attribute value is stored
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could set incorrect value of @a keyIdx when
 *   called in parallel with another function which modifies the value of the
 *   underlying attribute-list object of @a attrList. To ensure the correct
 *   value is set, the user must ensure that the underlying attribute-list
 *   object of the @a attrList is not modified during the call to the
 *   function.
 * - None of the access to either global or local objects requires thread
 *   synchronization.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844269}
 */
static inline void* LwSciSyncCoreAttrListGetValForKey(
    LwSciSyncCoreAttrList* coreAttrList,
    size_t keyIdx)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    return (void*)((uint8_t*)&coreAttrList->attrs +
            LwSciSyncCoreKeyInfo[keyIdx].offset);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}

/**
 * \brief Returns a constant value of LwSciSyncAttrKey or
 * LwSciSyncAttrInternalKey with a unified index keyIdx in the input
 * LwSciSyncCoreAttrList.
 *
 * If the attribute was not set, the resulting pointer will point to the
 * zero-initialized memory prepared for the attribute's value.
 * The function does no validation and assumes the keyIdx is a valid
 * unified key index.
 *
 * \param[in] coreAttrList LwSciSyncCoreAttrList
 * Valid value: coreAttrList is valid if it is non-NULL
 * \param[in] keyIdx unified key index
 * Valid value: [0, KEYS_COUNT)
 * \return const void*
 * - constant pointer to the place where the attribute value is stored
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could set incorrect value of @a keyIdx when
 *   called in parallel with another function which modifies the value of the
 *   underlying attribute-list object of @a attrList. To ensure the correct
 *   value is set, the user must ensure that the underlying attribute-list
 *   object of the @a attrList is not modified during the call to the
 *   function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{22034757}
 */
static inline const void* LwSciSyncCoreAttrListGetConstValForKey(
    const LwSciSyncCoreAttrList* coreAttrList,
    size_t keyIdx)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    return (const void*)((const uint8_t*)&coreAttrList->attrs +
            LwSciSyncCoreKeyInfo[keyIdx].offset);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
}

/**
 * \brief Checks if requiredPerm/actualPerm values in LwSciSyncCoreAttrList
 * has signaler permission bit set.
 *
 * This function does no validation and assumes the provided parameter(s)
 * are valid.
 *
 * \param[in] coreAttrList LwSciSyncCoreAttrList
 * Valid value: coreAttrList is valid if it is non-NULL
 *
 * \return bool
 * - true if the value corresponding to LwSciSyncAttrKey_RequiredPerm
 *   or LwSciSyncAttrKey_ActualPerm contains SignalOnly bit set.
 * - false otherwise
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could return incorrect value when called in
 *   parallel with another function which modifies the value of the underlying
 *   attribute-list object of @a attrList. To ensure the correct value is set,
 *   the user must ensure that the underlying attribute-list object of the @a
 *   attrList is not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844272}
 */
static inline bool LwSciSyncCoreAttrListHasSignalerPerm(
    const LwSciSyncCoreAttrList* coreAttrList)
{
    return (((uint64_t) coreAttrList->attrs.requiredPerm &
                 (uint64_t) LwSciSyncAccessPerm_SignalOnly) != 0U) ||
            (((uint64_t) coreAttrList->attrs.actualPerm &
                 (uint64_t) LwSciSyncAccessPerm_SignalOnly) != 0U);
}

/**
 * \brief Checks if requiredPerm/actualPerm values in LwSciSyncCoreAttrList
 * has waiter permission bit set.
 *
 * This function does no validation and assumes the provided parameter(s)
 * are valid.
 *
 * \param[in] coreAttrList LwSciSyncCoreAttrList
 * Valid value: coreAttrList is valid if it is non-NULL
 *
 * \return bool
 * - true if the value corresponding to LwSciSyncAttrKey_RequiredPerm
 *   or LwSciSyncAttrKey_ActualPerm contains WaitOnly bit set.
 * - false otherwise
 *
 * Conlwrrency:
 * - Thread-safe: no
 * - There are no operations in the function which depend on the order of
 *   access to either global or local objects, such that a system error would
 *   be caused, or that LwSciCommonPanic() would be called.
 * - Access to @a attrList requires thread synchronization; without
 *   synchronization, the function could return incorrect value when called in
 *   parallel with another function which modifies the value of the underlying
 *   attribute-list object of @a attrList. To ensure the correct value is set,
 *   the user must ensure that the underlying attribute-list object of the @a
 *   attrList is not modified during the call to the function.
 * - The operations are not expected to cause nor contribute to a deadlock, as
 *   there is no locking nor unlocking of any thread synchronization objects.
 *
 * \implements{18844275}
 */
static inline bool LwSciSyncCoreAttrListHasWaiterPerm(
    const LwSciSyncCoreAttrList* coreAttrList)
{
    return (((uint64_t) coreAttrList->attrs.requiredPerm &
                 (uint64_t) LwSciSyncAccessPerm_WaitOnly) != 0U) ||
            (((uint64_t) coreAttrList->attrs.actualPerm &
                 (uint64_t) LwSciSyncAccessPerm_WaitOnly) != 0U);
}

/**
 * \brief Allocates a new LwSciSyncAttrList and its associated
 * LwSciSyncCoreAttrListObj using LwSciCommon referencing framework and
 * initializes LwSciSyncCoreAttrListObj's members to represent a new, empty,
 * unreconciled and writable LwSciSyncAttrList, that has not been received over an LwSciIpc
 * channel. Also validates the input LwSciSyncModule and
 * binds a reference of the input LwSciSyncModule to the LwSciSyncModule member
 * in the LwSciSyncCoreAttrListObj.
 * Additionally, allocates valueCount number of LwSciSyncCoreAttrList slots in
 * LwSciSyncCoreAttrListObj.
 *
 * \param[in] module LwSciSyncModule the new LwSciSyncAttrList will be associated with
 * \param[in] valueCount number of slots
 *  Valid value: [1, SIZE_MAX]
 * \param[in] allocatePerSlotMembers Whether allocation of per-slot members is needed
 *  Valid value: true/false
 * \param[out] attrList the resulting LwSciSyncAttrList
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if @a attrList is NULL
 * - LwSciError_InsufficientMemory if there is no memory to create a new LwSciSyncAttrList
 * - LwSciError_IlwalidState if no more references can be taken for
 *   input LwSciSyncModule to create the new LwSciSyncAttrList.
 * - LwSciError_ResourceError if system lacks resource other than memory
 *   to create a LwSciSyncAttrList.
 * - Panics if @a module is invalid or @a valueCount is 0
 *
 * Conlwrrency:
 * - Thread-safe: yes
 * - Duplication of the @a module, needed for creation of an attribute list,
 *   requires thread synchronization; without synchronization, the function
 *   could cause a call to LwSciCommonPanic(). No synchronization is done in
 *   the function. To ensure that LwSciCommonPanic() is not called, the user
 *   must ensure that the module value is not modified during the call to the
 *   function.
 * - Incrementing the reference count of the @a module also requires thread
 *   synchronization. This synchronization is done by locking LwSciObj.ObjLock
 *   mutex object associated with @a module.
 * - The operations are not expected to cause nor contribute to a deadlock,
 *   since the mentioned mutex object is locked immediately before and
 *   released immediately after the attribute values are set.
 *
 * \implements{18844278}
 */
LwSciError LwSciSyncCoreAttrListCreateMultiSlot(
    LwSciSyncModule module,
    size_t valueCount,
    bool allocatePerSlotMembers,
    LwSciSyncAttrList* attrList);

/**
 * \brief Gets the value of signalerTimestampInfo or signalerTimestampInfoMulti
 * member from the reconciled attributes of an LwSciSyncAttrList.
 *
 * \param[in] reconciledAttrs The reconciled attributes corresponding to a
 * reconciled LwSciSyncAttrList
 * \param[out] timestampInfo pointer to where LwSciSyncAttrValTimestampInfo is
 * written
 * \return void
 * - Panics if any of the following oclwrs:
 *   - @a reconciledAttrs is NULL
 *   - @a timestampInfo is NULL
 *
 * \implements{}
 */
void LwSciSyncCoreGetTimestampInfo(
    const LwSciSyncCoreAttrs* reconciledAttrs,
    const LwSciSyncAttrValTimestampInfo** timestampInfo);

#ifdef LWSCISYNC_EMU_SUPPORT
 /**
 * \brief Copies ExternalPrimitiveInfo attribute
 *
 * \param[in,out] dest Destination memory where primitive info is copied.
 * \param[in] src Source memory containing primitive info to be copied.
 * \param[in] cnt Number of primitive info entries in src memory.
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if not enough memory to allocate a
 *     primitive info
 */
LwSciError LwSciSyncCoreCopySignalerExternalPrimitiveInfo(
    LwSciSyncPrimitiveInfo** dest,
    LwSciSyncPrimitiveInfo* const* src,
    size_t cnt);

/** Allocates memory for storing external primitive attribute */
LwSciError LwSciSyncCoreSignalerExternalPrimitiveAttrAlloc(
    size_t valueCount,
    LwSciSyncCoreAttrList* coreAttrList);

/** Frees memory for storing external primitive attribute */
void LwSciSyncCoreSignalerExternalPrimitiveAttrFree(
    LwSciSyncCoreAttrList* coreAttrList);

/** Handles copying of external primitive info attribute */
LwSciError LwSciSyncCoreCopyAttrVal(
    void* val,
    CoreAttribute* attribute,
    size_t maxSize);
#endif

/** @} */
#endif
