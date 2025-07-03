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
 * \brief <b>LwSciSync Core Attribute Management Implementation</b>
 *
 * @b Description: This file implements LwSciSync core attribute management APIs
 *
 * The code in this file is organised as below:
 * -Core structures declaration.
 * -Core interfaces declaration.
 * -Public interfaces definition.
 * -Core interfaces definition.
 */
#include "lwscisync_attribute_core.h"

#include "lwscicommon_libc.h"
#include "lwscicommon_objref.h"
#include "lwscicommon_os.h"
#include "lwscicommon_utils.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core_cluster.h"
#include "lwscisync_attribute_core_semaphore.h"
#include "lwscisync_ipc_table.h"
#include "lwscisync_module.h"

#define MAX_ELEMENTS (1UL<<32U)
#define MAX_ELEM_SIZE (1UL<<32U)

typedef struct {
    void (* toCoreAttr)(const void* pair, CoreAttribute* coreAttr);
    void (* toKeyValuePair)(const CoreAttribute* coreAttr, void* pair);
    void*(* getPairAt)(void* pairArray, size_t index);
    const void*(* getConstPairAt)(const void* pairArray, size_t index);
    bool(* isAttrKeyValid)(const void* pair);
} KeyValuePairTranslationOps;

/** copies public attribute to Core attribute type */
static void PublicAttrToCoreAttr(
    const void* pair,
    CoreAttribute* coreAttr);

/** copies internal attribute to Core attribute type */
static void InternalAttrToCoreAttr(
    const void* pair,
    CoreAttribute* coreAttr);

static void CoreAttrToPublicAttr(
    const CoreAttribute* coreAttr,
    void* pair);

static void CoreAttrToInternalAttr(
    const CoreAttribute* coreAttr,
    void* pair);

static const void* GetPublicConstPairAt(
    const void* pairArray,
    size_t index);

static const void* GetInternalConstPairAt(
    const void* pairArray,
    size_t index);

static void* GetPublicPairAt(
    void* pairArray,
    size_t index);

static void* GetInternalPairAt(
    void* pairArray,
    size_t index);

static LwSciError LwSciSyncAttrListSlotSetAttrs(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    const void* pairArray,
    size_t pairCount,
    const KeyValuePairTranslationOps* kvpOps);

static LwSciError AttrListSlotGetAttrs(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    void* pairArray,
    size_t pairCount,
    const KeyValuePairTranslationOps* kvpOps);

/** Generate a header value for input obj attr list */
static inline uint64_t AttrListGenerateHeader(
    const LwSciSyncCoreAttrListObj* objAttrList)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_4), "LwSciSync-ADV-MISRAC2012-012")
    return (((uint64_t)objAttrList & (~0xFFFFULL)) | 0xCDULL);
}

/** Allocate memory for core attr list structure */
static LwSciError CoreAttrListAlloc(
    size_t valueCount,
    bool allocatePerSlotMembers,
    LwSciSyncCoreAttrList** coreAttrList);


/** Wrapper function to avoid type-casting to void* at multiple places */
static inline void FreeObjAndRef(
    LwSciSyncAttrList attrList)
{
    LwSciCommonFreeObjAndRef(&attrList->refAttrList, LwSciSyncCoreAttrListFree, NULL);
}

/** Returns true if attr list is empty or unreconciled */
static inline bool IsAttrListWritable(
    const LwSciSyncCoreAttrListObj* list)
{

    return (list->state == LwSciSyncCoreAttrListState_Unreconciled) &&
        (list->writable == true);
}

/** Ensure public key is within bounds */
static inline bool PublicAttrKeyIsValid(
    const void* pair)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    const LwSciSyncAttrKeyValuePair* publicPair =
            (const LwSciSyncAttrKeyValuePair*) pair;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LwSciSyncAttrKey attrKey = publicPair->attrKey;
    return (attrKey > LwSciSyncAttrKey_LowerBound) &&
            (attrKey < LwSciSyncAttrKey_UpperBound);
}

/** Ensure internal key is within bounds */
static inline bool InternalAttrKeyIsValid(
    const void* pair)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    const LwSciSyncInternalAttrKeyValuePair* internalPair =
            (const LwSciSyncInternalAttrKeyValuePair*) pair;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    LwSciSyncInternalAttrKey attrKey = internalPair->attrKey;
    return (attrKey > LwSciSyncInternalAttrKey_LowerBound) &&
            (attrKey < LwSciSyncInternalAttrKey_UpperBound);
}

/** Sanity check for key value pair */
static LwSciError ValidateKeyValuePair(
    const void* pair,
    const KeyValuePairTranslationOps* kvpOps);

/** Sanity check for an attribute */
static LwSciError ValidateCoreAttribute(
    const CoreAttribute* attr);

/** Check if key can be set */
static inline bool IsAttributeWritable(
    const LwSciSyncCoreAttrList* coreAttrList,
    size_t keyIdx)
{
    return (((coreAttrList->attrs.keyState[keyIdx] ==
                LwSciSyncCoreAttrKeyState_Empty) ||
           (coreAttrList->attrs.keyState[keyIdx] ==
                LwSciSyncCoreAttrKeyState_SetUnlocked)) &&
           (LwSciSyncCoreKeyInfo[keyIdx].writable));
}

/** Wrapper for validating attr list array */
static LwSciError ValidateInputsAndRetrieveObj(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    const void* pairArray,
    size_t pairCount,
    LwSciSyncCoreAttrListObj** objAttrList);

/**
 * @brief Copy unreconciled attr lists
 *
 * Note: The caller should hold locks for the Attribute Lists being operated
 * on. No relwrsive locking is attempted here.
 */
static LwSciError CopySlots(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncCoreAttrListObj* newObjAttrList,
    size_t slotCnt);

/** Unpack attribute lists */
static LwSciError UnpackAttrLists(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncModule* module,
    size_t* slotCnt);

/** Change state for locked keys to unlocked */
static void UnlockKeyState(
    LwSciSyncCoreAttrList* coreAttrList);

/** Sanity check for length input */
static inline bool IsValueLenSane(size_t len,
    size_t elemSize,
    size_t maxElements)
{
    return ((0U != len) && (0U != elemSize) && (0U == (len % elemSize))
            && (maxElements < MAX_ELEMENTS)
            && (elemSize < MAX_ELEM_SIZE)
            && (len <= (maxElements * elemSize)));
}

static const KeyValuePairTranslationOps PublicKeyValueOps = {
    .toCoreAttr = PublicAttrToCoreAttr,
    .toKeyValuePair = CoreAttrToPublicAttr,
    .getPairAt = GetPublicPairAt,
    .getConstPairAt = GetPublicConstPairAt,
    .isAttrKeyValid = PublicAttrKeyIsValid,
};

static const KeyValuePairTranslationOps InternalKeyValueOps = {
    .toCoreAttr = InternalAttrToCoreAttr,
    .toKeyValuePair = CoreAttrToInternalAttr,
    .getPairAt = GetInternalPairAt,
    .getConstPairAt = GetInternalConstPairAt,
    .isAttrKeyValid = InternalAttrKeyIsValid,
};

/******************************************************
 *            Public interfaces definition
 ******************************************************/

/** Copy LwSciSyncCoreAttrList member */
static LwSciError LwSciSyncCoreAttrListCopy(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrList* newCoreAttrList);

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListCreate(
    LwSciSyncModule module,
    LwSciSyncAttrList* attrList)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreAttrListCreateMultiSlot(module, 1U, true, attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

void LwSciSyncAttrListFree(
    LwSciSyncAttrList attrList)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("attrList : %p\n", attrList);

    FreeObjAndRef(attrList);

fn_exit:

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListSetAttrs(
    LwSciSyncAttrList attrList,
    const LwSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncAttrListSlotSetAttrs(attrList, 0U, pairArray, pairCount,
            &PublicKeyValueOps);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListGetAttrs(
    LwSciSyncAttrList attrList,
    LwSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = AttrListSlotGetAttrs(attrList, 0U, pairArray, pairCount,
            &PublicKeyValueOps);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
size_t LwSciSyncAttrListGetSlotCount(
    LwSciSyncAttrList attrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    size_t slotCount = 0U;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("attrList: %p\n", attrList);

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    slotCount = objAttrList->numCoreAttrList;

    LWSCI_INFO("*slotCount : %d\n", slotCount);

fn_exit:

    LWSCI_FNEXIT("");

    return slotCount;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
int32_t LwSciSyncAttrListCompare(
    const void* elem1,
    const void* elem2)
{
    int32_t ret = 0;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_4), "LwSciSync-ADV-MISRAC2012-012")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    uint64_t attrListAddr1 = (uint64_t)*(const LwSciSyncAttrList*)elem1;
    uint64_t attrListAddr2 = (uint64_t)*(const LwSciSyncAttrList*)elem2;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_4))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

    LWSCI_FNENTRY("");

    if (attrListAddr1 > attrListAddr2) {
        ret = 1;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (attrListAddr1 < attrListAddr2) {
        ret = -1;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return ret;
}

LwSciError LwSciSyncCoreAttrListsLock(
    const LwSciSyncAttrList inputAttrListArr[],
    size_t attrListCount)
{
    LwSciError error = LwSciError_Success;

    LwSciSyncAttrList* sortedAttrListArr = NULL;
    size_t lwrrAttrList = 0U;
    size_t attrListArrSize = 0U;
    uint8_t overflow = OP_FAIL;

    LWSCI_FNENTRY("");

    if (0U == attrListCount) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreValidateAttrListArray(inputAttrListArr, attrListCount,
            false);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to validate Attribute List array\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    sizeMul(sizeof(LwSciSyncAttrList), attrListCount, &attrListArrSize, &overflow);
    if (OP_FAIL == overflow) {
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    sortedAttrListArr = (LwSciSyncAttrList*)LwSciCommonCalloc(
            attrListCount, sizeof(LwSciSyncAttrList));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == sortedAttrListArr) {
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonMemcpyS(sortedAttrListArr, attrListArrSize,
            inputAttrListArr, attrListArrSize);

    /*
     * Sort the Attribute Lists to enforce a Resource Ordering, so we don't run
     * into a situation where we can deadlock.
     */
    LwSciCommonSort(sortedAttrListArr, attrListCount,
            sizeof(LwSciSyncAttrList), LwSciSyncAttrListCompare);

    for (lwrrAttrList = 0U; lwrrAttrList < attrListCount; ++lwrrAttrList) {
        LwSciSyncAttrList attrList = sortedAttrListArr[lwrrAttrList];

        LwSciCommonObjLock(&attrList->refAttrList);
    }

    LwSciCommonFree(sortedAttrListArr);

fn_exit:
    return error;
}

LwSciError LwSciSyncCoreAttrListsUnlock(
    const LwSciSyncAttrList inputAttrListArr[],
    size_t attrListCount)
{
    LwSciError error = LwSciError_Success;

    size_t lwrrAttrList = 0U;

    LWSCI_FNENTRY("");

    if (0U == attrListCount) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreValidateAttrListArray(inputAttrListArr,
            attrListCount, false);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to validate Attribute List array\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    for (lwrrAttrList = 0U; lwrrAttrList < attrListCount; ++lwrrAttrList) {
        LwSciSyncAttrList attrList = inputAttrListArr[lwrrAttrList];
        LwSciCommonObjUnlock(&attrList->refAttrList);
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncCoreAttrListAppendUnreconciledWithLocks(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool acquireLocks,
    LwSciSyncAttrList* newUnreconciledAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* newObjAttrList = NULL;
    LwSciSyncModule module = NULL;
    size_t slotCnt = 0U;

    LWSCI_FNENTRY("");

    /** Acquire all locks for all the Attribute Lists */
    if (acquireLocks) {
        error = LwSciSyncCoreAttrListsLock(inputUnreconciledAttrListArray,
                inputUnreconciledAttrListCount);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    error = UnpackAttrLists(
            inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
            &module, &slotCnt);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto release_locks;
    }

    LWSCI_INFO("inputUnreconciledAttrListCount: %zu\n",
            inputUnreconciledAttrListCount);
    LWSCI_INFO("newUnreconciledAttrList: %p\n", newUnreconciledAttrList);

    error = LwSciSyncCoreAttrListCreateMultiSlot(module, slotCnt,
            false, newUnreconciledAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto release_locks;
    }
    LwSciSyncCoreAttrListGetObjFromRef(*newUnreconciledAttrList,
            &newObjAttrList);

    error = CopySlots(
            inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
            newObjAttrList, slotCnt);
    if (LwSciError_Success != error) {
        LwSciSyncAttrListFree(*newUnreconciledAttrList);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto release_locks;
    }

    // Make appended list read-only
    newObjAttrList->writable = false;

    LWSCI_INFO("*newUnreconciledAttrList: %p\n", *newUnreconciledAttrList);

release_locks:
    if (acquireLocks) {
        LwSciError err = LwSciError_Success;
        err = LwSciSyncCoreAttrListsUnlock(inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount);
        if (LwSciError_Success != err) {
            LWSCI_ERR_STR("Could not unlock Attribute Lists\n");
            LwSciCommonPanic();
        }
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LwSciError LwSciSyncAttrListAppendUnreconciled(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncAttrList* newUnreconciledAttrList)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreAttrListAppendUnreconciledWithLocks(
            inputUnreconciledAttrListArray, inputUnreconciledAttrListCount,
            true, newUnreconciledAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LwSciError LwSciSyncAttrListClone(
    LwSciSyncAttrList origAttrList,
    LwSciSyncAttrList* newAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciSyncCoreAttrListObj* newObjAttrList = NULL;
    LwSciSyncCoreAttrList* newCoreAttrList = NULL;
    const LwSciSyncCoreAttrList* coreAttrList = NULL;
    size_t i = 0U;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(origAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("origAttrList: %p\n", origAttrList);
    LWSCI_INFO("newAttrList: %p\n", newAttrList);

    LwSciSyncCoreAttrListGetObjFromRef(origAttrList, &objAttrList);

    coreAttrList = objAttrList->coreAttrList;
    error = LwSciSyncCoreAttrListCreateMultiSlot(objAttrList->module,
            objAttrList->numCoreAttrList, false, newAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncCoreAttrListGetObjFromRef(*newAttrList, &newObjAttrList);

    newCoreAttrList = newObjAttrList->coreAttrList;

    LwSciCommonObjLock(&origAttrList->refAttrList);

    /** Copy core attr list */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        error = LwSciSyncCoreAttrListCopy(&coreAttrList[i],
                &newCoreAttrList[i]);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto unlock_attr_list;
        }
    }
    newObjAttrList->state = objAttrList->state;
    newObjAttrList->writable = objAttrList->writable;

    /** Update the key state */
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        UnlockKeyState(&newCoreAttrList[i]);
    }

    LWSCI_INFO("*newAttrList: %p\n", *newAttrList);

unlock_attr_list:
    LwSciCommonObjUnlock(&origAttrList->refAttrList);

    if (LwSciError_Success != error) {
        LwSciSyncAttrListFree(*newAttrList);
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListSlotGetAttrs(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    LwSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError error = LwSciError_Success;
    error = AttrListSlotGetAttrs(attrList, slotIndex, (void*) pairArray,
            pairCount, &PublicKeyValueOps);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Call AttrListSlotGetAttrs failed.\n");
    }
    return error;
}

/******************************************************
 *           Internal interfaces definition
 ******************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListSetInternalAttrs(
    LwSciSyncAttrList attrList,
    const LwSciSyncInternalAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncAttrListSlotSetAttrs(attrList, 0U, pairArray,
            pairCount, &InternalKeyValueOps);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListGetInternalAttrs(
    LwSciSyncAttrList attrList,
    LwSciSyncInternalAttrKeyValuePair* pairArray,
    size_t pairCount)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = AttrListSlotGetAttrs(attrList, 0U, pairArray,
            pairCount, &InternalKeyValueOps);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListGetSingleInternalAttr(
    LwSciSyncAttrList attrList,
    LwSciSyncInternalAttrKey key,
    const void** value,
    size_t* len)
{
    LwSciError error;
    LwSciSyncInternalAttrKeyValuePair pairArray = {key, NULL, 0};

    LWSCI_FNENTRY("");

    if (NULL == value) {
        LWSCI_ERR_STR("Invalid value: null pointer passed\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == len) {
        LWSCI_ERR_STR("Invalid len pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncAttrListGetInternalAttrs(attrList, &pairArray, 1);
    if (LwSciError_Success == error) {
        *value = pairArray.value;
        *len = pairArray.len;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

/******************************************************
 *             Core interfaces definition
 ******************************************************/

LwSciError LwSciSyncCoreAttrListValidate(
    LwSciSyncAttrList attrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("attrList : %p\n", attrList);

    if (NULL == attrList) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    if (AttrListGenerateHeader(objAttrList) != objAttrList->header) {
        LWSCI_ERR_STR("Invalid LwSciSyncAttrList\n");
        LwSciCommonPanic();
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

void LwSciSyncCoreAttrListGetModule(
    LwSciSyncAttrList attrList,
    LwSciSyncModule* module)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed.\n");
        LwSciCommonPanic();
    }
    if (NULL == module) {
        LWSCI_ERR_STR("Invalid module.\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("attrList : %p\n", attrList);
    LWSCI_INFO("module : %p\n", module);

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    *module = objAttrList->module;

    LWSCI_INFO("*module : %p\n", *module);

}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciSync-ADV-MISRAC2012-001")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
static inline LwSciSyncAttrList CastRefToSyncAttrList(LwSciRef* arg) {
    return (LwSciSyncAttrList)(void*)((char*)(void*)arg
        - LW_OFFSETOF(struct LwSciSyncAttrListRec, refAttrList));
}
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 18_4))
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCoreAttrListDup(
    LwSciSyncAttrList attrList,
    LwSciSyncAttrList* dupAttrList)
{
    LwSciRef* dupAttrListParam = NULL;
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed.\n");
        LwSciCommonPanic();
    }
    if (NULL == dupAttrList) {
        LWSCI_ERR_STR("Invalid arguments: dupAttrList is NULL\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("attrList: %p\n", attrList);
    LWSCI_INFO("dupAttrList: %p\n", dupAttrList);

    error = LwSciCommonDuplicateRef(&attrList->refAttrList, &dupAttrListParam);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *dupAttrList = CastRefToSyncAttrList(dupAttrListParam);

fn_exit:

    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncCoreAttrListCreateMultiSlot(
    LwSciSyncModule module,
    size_t valueCount,
    bool allocatePerSlotMembers,
    LwSciSyncAttrList* attrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciObj* objAttrListParam = NULL;
    LwSciRef* attrListParam = NULL;

    LWSCI_FNENTRY("");

    /** validate all input args */
    if (NULL == attrList) {
        LWSCI_ERR_STR("Invalid argument: attrList: NULL\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *attrList = NULL;
    if (0U == valueCount) {
        LWSCI_ERR_ULONG("Invalid argument: valueCount: \n", valueCount);
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Allocate new attribute list */
    error = LwSciCommonAllocObjWithRef(sizeof(LwSciSyncCoreAttrListObj),
            sizeof(struct LwSciSyncAttrListRec), &objAttrListParam,
            &attrListParam);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to create attr list\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    objAttrList = LwSciCastObjToBufAttrListObjPriv(objAttrListParam);
    *attrList = CastRefToSyncAttrList(attrListParam);

    error = CoreAttrListAlloc(valueCount, allocatePerSlotMembers, &(objAttrList->coreAttrList));
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreModuleDup(module, &(objAttrList->module));
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Set attr list header for future validation */
    objAttrList->header = AttrListGenerateHeader(objAttrList);
    /** Set number of core attr lists */
    objAttrList->numCoreAttrList = valueCount;
    /** Set attr list state to empty */
    objAttrList->state = LwSciSyncCoreAttrListState_Unreconciled;
    objAttrList->writable = true;

fn_exit:
    if (LwSciError_Success != error) {
        if (NULL != objAttrList) {
            FreeObjAndRef(*attrList);
        }
    }

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError LwSciSyncAttrListSlotSetAttrs(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    const void* pairArray,
    size_t pairCount,
    const KeyValuePairTranslationOps* kvpOps)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrList* coreAttrList = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    size_t i = 0U;
    void* val = NULL;
    size_t keyCntInPairArray[KEYS_COUNT] = {0U};

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = ValidateInputsAndRetrieveObj(attrList, slotIndex,
            (const void*)pairArray, pairCount, &objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    coreAttrList = &objAttrList->coreAttrList[slotIndex];

    if (!IsAttrListWritable(objAttrList)){
        LWSCI_ERR_STR("Attribute list is not writable\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonObjLock(&attrList->refAttrList);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    for (i = 0U; i < pairCount; i++) {
        uint8_t addStatus = OP_FAIL;
        CoreAttribute attribute;
        const void* pair = kvpOps->getConstPairAt(pairArray, i);
        /** validate all input args */
        error = ValidateKeyValuePair(pair, kvpOps);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto unlock;
        }
        kvpOps->toCoreAttr(pair, &attribute);
        if (!IsAttributeWritable(coreAttrList, attribute.index)) {
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto unlock;
        }
        u64Add(keyCntInPairArray[attribute.index], 1U,
                   &keyCntInPairArray[attribute.index], &addStatus);
        if (OP_SUCCESS != addStatus) {
            LwSciCommonPanic();
        }
        if (keyCntInPairArray[attribute.index] > 1U) {
            LWSCI_ERR_UINT("Duplicate key found in pairArray: \n",
                    LwSciSyncCoreIndexToKey(attribute.index));
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto unlock;
        }
    }

    /** Update the values, len and state */
    for (i = 0U; i < pairCount; i++) {
        CoreAttribute attribute;
        const void* pair = kvpOps->getConstPairAt(pairArray, i);
        size_t maxSize = 0U;
        uint8_t opStatus = OP_FAIL;

        kvpOps->toCoreAttr(pair, &attribute);
        sizeMul(LwSciSyncCoreKeyInfo[attribute.index].elemSize,
                LwSciSyncCoreKeyInfo[attribute.index].maxElements,
                &maxSize, &opStatus);
        if (OP_SUCCESS != opStatus) {
            LWSCI_ERR_STR("Arithmetic failure although the check was done before");
            LwSciCommonPanic();
        }

        val = LwSciSyncCoreAttrListGetValForKey(coreAttrList, attribute.index);
#ifdef LWSCISYNC_EMU_SUPPORT
        error = LwSciSyncCoreCopyAttrVal(val, &attribute, maxSize);
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
#else
        LwSciCommonMemcpyS(val, maxSize, attribute.value, attribute.len);
#endif
        coreAttrList->attrs.valSize[attribute.index] = attribute.len;
        coreAttrList->attrs.keyState[attribute.index] =
                LwSciSyncCoreAttrKeyState_SetLocked;
    }

unlock:
    {
        LwSciCommonObjUnlock(&attrList->refAttrList);
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError AttrListSlotGetAttrs(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    void* pairArray,
    size_t pairCount,
    const KeyValuePairTranslationOps* kvpOps)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrList* coreAttrList = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    size_t i = 0U;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = ValidateInputsAndRetrieveObj(attrList, slotIndex,
            pairArray, pairCount, &objAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    coreAttrList = &objAttrList->coreAttrList[slotIndex];

    for (i = 0U; i < pairCount; i++) {
        const void* pair = kvpOps->getPairAt(pairArray, i);
        if (!kvpOps->isAttrKeyValid(pair)) {
            LWSCI_ERR_STR("Invalid attribute key\n");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
    LwSciCommonObjLock(&attrList->refAttrList);

    /** Get the values */
    for (i = 0U; i < pairCount; i++) {
        CoreAttribute attribute;
        void* pair = kvpOps->getPairAt(pairArray, i);
        kvpOps->toCoreAttr(pair, &attribute);

        attribute.len = coreAttrList->attrs.valSize[attribute.index];
        if (0U != attribute.len) {
            attribute.value = LwSciSyncCoreAttrListGetValForKey(coreAttrList,
                    attribute.index);
        } else {
            attribute.value = NULL;
        }
        kvpOps->toKeyValuePair(&attribute, pair);
    }
    LwSciCommonObjUnlock(&attrList->refAttrList);

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreAttrListTypeIsCpuSignaler(
    LwSciSyncAttrList attrList,
    bool* isCpuSignaler)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed.\n");
        LwSciCommonPanic();
    }

    if (NULL == isCpuSignaler) {
        LWSCI_ERR_STR("Invalid isCpuSignaler\n");
        LwSciCommonPanic();
    }

    /* This call won't fail since the attrList has been validated */
    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    *isCpuSignaler = objAttrList->coreAttrList->attrs.needCpuAccess &&
            (((uint64_t)(objAttrList->coreAttrList->attrs.actualPerm) &
            ((uint64_t)LwSciSyncAccessPerm_SignalOnly)) != 0U);

}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreAttrListTypeIsCpuWaiter(
    LwSciSyncAttrList attrList,
    bool* isCpuWaiter)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed.\n");
        LwSciCommonPanic();
    }

    if (NULL == isCpuWaiter) {
        LWSCI_ERR_STR("Invalid isCpuWaiter\n");
        LwSciCommonPanic();
    }

    /* This call won't fail since the attrList has been validated */
    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    *isCpuWaiter = objAttrList->coreAttrList->attrs.needCpuAccess &&
            (((uint64_t)(objAttrList->coreAttrList->attrs.actualPerm) &
            ((uint64_t)LwSciSyncAccessPerm_WaitOnly)) != 0U);

}

void LwSciSyncCoreAttrListTypeIsC2cSignaler(
    LwSciSyncAttrList attrList,
    bool* isC2cSignaler)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciSyncHwEngName engName = LwSciSyncHwEngName_LowerBound;
    size_t len = 0U;
    const void* value = NULL;

    /** validate all input args */
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed");
        LwSciCommonPanic();
    }

    if (NULL == isC2cSignaler) {
        LWSCI_ERR_STR("Invalid isC2cSignaler");
        LwSciCommonPanic();
    }

    /* This call won't fail since the attrList has been validated */
    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    error = LwSciSyncAttrListGetSingleInternalAttr(
        attrList,
        LwSciSyncInternalAttrKey_EngineArray,
        &value, &len);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("unexpected get attribute failure");
        LwSciCommonPanic();
    }
    if (sizeof(LwSciSyncHwEngine) != len) {
        *isC2cSignaler = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncHwEngGetNameFromId(
        objAttrList->coreAttrList->attrs.engineArray[0].rmModuleID,
        &engName);
    if (LwSciError_Success != error) {
        *isC2cSignaler = false;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    *isC2cSignaler =
        (((uint64_t)(objAttrList->coreAttrList->attrs.actualPerm) &
          ((uint64_t)LwSciSyncAccessPerm_SignalOnly)) != 0U) &&
        (LwSciSyncHwEngName_PCIe == engName);
fn_exit:
    return;
}

static LwSciError CoreAttrListAlloc(
    size_t valueCount,
    bool allocatePerSlotMembers,
    LwSciSyncCoreAttrList** coreAttrList)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrList* temp = NULL;

    LWSCI_FNENTRY("");

    LWSCI_INFO("valueCount : %zu\n", valueCount);
    LWSCI_INFO("coreAttrList : %p\n", coreAttrList);

    /** Allocate memory for core attr list */
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    temp = (LwSciSyncCoreAttrList*)LwSciCommonCalloc(valueCount,
            sizeof(LwSciSyncCoreAttrList));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (NULL == temp) {
        LWSCI_ERR_STR("failed to allocate memory.\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

#ifdef LWSCISYNC_EMU_SUPPORT
    if (allocatePerSlotMembers) {
        error = LwSciSyncCoreSignalerExternalPrimitiveAttrAlloc(valueCount, temp);
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
    }
#else
    (void)allocatePerSlotMembers;
#endif

fn_exit:
    if (LwSciError_Success == error) {
        *coreAttrList = temp;
        LWSCI_INFO("*coreAttrList : %p\n", *coreAttrList);
    }
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreAttrListFree(
    LwSciObj* objPtr)
{
    size_t i = 0U;
    const LwSciSyncCoreAttrListObj* objAttrList =
        LwSciCastObjToBufAttrListObjPriv(objPtr);
    LwSciSyncCoreAttrList* coreAttrList = NULL;

    LWSCI_FNENTRY("");

    LwSciSyncModuleClose(objAttrList->module);
    for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
        coreAttrList = &objAttrList->coreAttrList[i];
        LwSciSyncCoreIpcTableFree(&coreAttrList->ipcTable);

        /** Free LwSciBuf attr lists */
        if (NULL != coreAttrList->semaAttrList) {
            LwSciBufAttrListFree(coreAttrList->semaAttrList);
        }
        /** Free LwSciBuf timestampBufAttrList */
        if (NULL != coreAttrList->timestampBufAttrList) {
            LwSciBufAttrListFree(coreAttrList->timestampBufAttrList);
        }
#ifdef LWSCISYNC_EMU_SUPPORT
        LwSciSyncCoreSignalerExternalPrimitiveAttrFree(coreAttrList);
#endif
    }
    LwSciCommonFree(objAttrList->coreAttrList);
    LWSCI_FNEXIT("");
}

static LwSciError ValidateKeyValuePair(
    const void* pair,
    const KeyValuePairTranslationOps* kvpOps)
{
    LwSciError error = LwSciError_Success;
    CoreAttribute attr;

    LWSCI_FNENTRY("");

    if (!kvpOps->isAttrKeyValid(pair)) {
        LWSCI_ERR_STR("Invalid attribute key\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    kvpOps->toCoreAttr(pair, &attr);

    error = ValidateCoreAttribute(&attr);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError ValidateInputsAndRetrieveObj(
    LwSciSyncAttrList attrList,
    size_t slotIndex,
    const void* pairArray,
    size_t pairCount,
    LwSciSyncCoreAttrListObj** objAttrList)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (NULL == pairArray) {
        LWSCI_ERR_STR("Invalid argument: pairArray: NULL\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (0U == pairCount) {
        LWSCI_ERR_ULONG("Invalid argument: pairCount: \n", pairCount);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    LwSciSyncCoreAttrListGetObjFromRef(attrList, objAttrList);

    if (slotIndex >= (*objAttrList)->numCoreAttrList) {
        LWSCI_ERR_ULONG("Invalid argument: slotIndex: \n", slotIndex);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

static LwSciError CopySlots(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncCoreAttrListObj* newObjAttrList,
    size_t slotCnt
)
{
    LwSciError error = LwSciError_Success;
    size_t inputListIdx = 0U;
    LwSciSyncAttrList attrList = NULL;
    size_t coreListIdx = 0U;
    size_t newCoreListIdx = 0U;
    uint8_t addStatus = OP_FAIL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    for (inputListIdx = 0U; inputListIdx < inputUnreconciledAttrListCount;
            inputListIdx++) {
        attrList = inputUnreconciledAttrListArray[inputListIdx];
        LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
        for (coreListIdx = 0U; coreListIdx < objAttrList->numCoreAttrList;
                coreListIdx++) {
            error = LwSciSyncCoreAttrListCopy(
                    &objAttrList->coreAttrList[coreListIdx],
                    &newObjAttrList->coreAttrList[newCoreListIdx]);
            if (LwSciError_Success != error) {
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
            u64Add(newCoreListIdx, 1U, &newCoreListIdx, &addStatus);
            if (OP_SUCCESS != addStatus) {
                LWSCI_ERR_STR("newCoreListIdx value is out of range.\n");
                error = LwSciError_Overflow;
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
        }
        newObjAttrList->state = objAttrList->state;
        newObjAttrList->writable = objAttrList->writable;
    }

    /** assert boundary check */
    if (newCoreListIdx > slotCnt) {
        LWSCI_ERR_STR("Writing beyond limits\n");
    }

fn_exit:

    return error;
}

static LwSciError UnpackAttrLists(
    const LwSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    LwSciSyncModule* module,
    size_t* slotCnt)
{
    LwSciError error = LwSciError_Success;
    size_t inputListIdx = 0U;
    LwSciSyncAttrList attrList = NULL;
    bool isDup = false;
    uint8_t addStatus = OP_FAIL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    error = LwSciSyncCoreValidateAttrListArray(inputUnreconciledAttrListArray,
            inputUnreconciledAttrListCount, false);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Invalid argument: inputUnreconciledAttrListArray: NULL\n");
        LWSCI_ERR_ULONG("Invalid argument: inputUnreconciledAttrListCount: \n",
            inputUnreconciledAttrListCount);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_4), "LwSciSync-ADV-MISRAC2012-015")
    for (inputListIdx = 0U; inputListIdx < inputUnreconciledAttrListCount;
            inputListIdx++) {
        attrList = inputUnreconciledAttrListArray[inputListIdx];

        LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

        /** Validate modules */
        if (0U == inputListIdx) {
            *module = objAttrList->module;
        }
        LwSciSyncCoreModuleIsDup(*module, objAttrList->module,
                &isDup);

        if (false == isDup) {
            LWSCI_ERR_ULONG("Incompatible modules in list 0 and \n", inputListIdx);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        /** Callwlate the number of slots needed */
        u64Add((*slotCnt), (objAttrList->numCoreAttrList),
               slotCnt, &addStatus);
        if (OP_SUCCESS != addStatus) {
            LWSCI_ERR_STR("*slotCnt value is out of range.\n");
            error = LwSciError_Overflow;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
        /** Ensure input attr list is unreconciled */
        if (LwSciSyncCoreAttrListState_Unreconciled != objAttrList->state) {
            LWSCI_ERR_ULONG("Input attrList index is unreconciled: \n",
                    inputListIdx);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        LWSCI_INFO("inputUnreconciledAttrListArray[%zu]: %p\n", inputListIdx,
                attrList);
    }

fn_exit:

    return error;
}

LwSciError LwSciSyncCoreValidateAttrListArray(
    const LwSciSyncAttrList attrListArray[],
    size_t attrListCount,
    bool allowEmpty)
{
    LwSciError error = LwSciError_Success;
    size_t inputListIdx;

    if (0U == attrListCount) {
        if (!allowEmpty || (NULL != attrListArray)) {
            error = LwSciError_BadParameter;

            LWSCI_ERR_STR("Unexpected empty LwSciSyncAttrList array given");
        }
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == attrListArray) {
        error = LwSciError_BadParameter;

        LWSCI_ERR_STR("NULL LwSciSyncAttrList array given");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    for (inputListIdx = 0U; inputListIdx < attrListCount;
            inputListIdx++) {
        LwSciSyncAttrList attrList = attrListArray[inputListIdx];

        error = LwSciSyncCoreAttrListValidate(attrList);
        if (LwSciError_Success != error) {
            LWSCI_ERR_STR("Invalid attr list: NULL");
            LWSCI_ERR_ULONG("attrListArray", inputListIdx);
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }
fn_exit:
    return error;
}

static void UnlockKeyState(
    LwSciSyncCoreAttrList* coreAttrList)
{
    size_t i = 0U;
    for (i = 0U; i < KEYS_COUNT; i++) {
        if (LwSciSyncCoreAttrKeyState_SetLocked ==
                coreAttrList->attrs.keyState[i]) {
            coreAttrList->attrs.keyState[i] =
                    LwSciSyncCoreAttrKeyState_SetUnlocked;
        }
    }
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncAttrListGetAttr(
    LwSciSyncAttrList attrList,
    LwSciSyncAttrKey key,
    const void** value,
    size_t* len)
{
    LwSciError error;
    LwSciSyncAttrKeyValuePair pairArray = {key, NULL, 0};

    LWSCI_FNENTRY("");

    if (NULL == value) {
        LWSCI_ERR_STR("Invalid value: NULL\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == len) {
        LWSCI_ERR_STR("Invalid len (NULL pointer)\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncAttrListGetAttrs(attrList, &pairArray, 1U);
    if (LwSciError_Success == error) {
        *value = pairArray.value;
        *len = pairArray.len;
    }

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

void LwSciSyncCoreAttrListGetTimestampBufAttrList(
    LwSciSyncAttrList syncAttrList,
    LwSciBufAttrList* timestampBufAttrList)
{
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    const LwSciSyncCoreAttrs* attrs = NULL;

    const LwSciSyncAttrValTimestampInfo* timestampInfo = NULL;

    LwSciSyncCoreAttrListGetObjFromRef(syncAttrList, &objAttrList);

    attrs = &objAttrList->coreAttrList->attrs;

    LwSciSyncCoreGetTimestampInfo(attrs, &timestampInfo);
    if ((timestampInfo != NULL) &&
            (timestampInfo->format == LwSciSyncTimestampFormat_EmbeddedInPrimitive)) {
        *timestampBufAttrList = objAttrList->coreAttrList->semaAttrList;
    }
    else {
        *timestampBufAttrList = objAttrList->coreAttrList->timestampBufAttrList;
    }

    return;
}

static LwSciError LwSciSyncCoreAttrListCopy(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrList* newCoreAttrList)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    /** Copy direct members */
    /* Note: The LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo
     * attribute key needs to be deep-copied. This happens below in the
     * LwSciSyncCoreSignalerExternalPrimitiveAttrAlloc() call.
     *
     * We've ensured that the LwSciSyncCoreAttrList passed into this function
     * hasn't allocated any LwSciCommon-allocated memory, so this shallow-copy
     * here is temporary (so that all the other attributes are copied) and
     * will not leak.
     */
    newCoreAttrList->attrs = coreAttrList->attrs;
    newCoreAttrList->lastExport = coreAttrList->lastExport;

#ifdef LWSCISYNC_EMU_SUPPORT
    error = LwSciSyncCoreSignalerExternalPrimitiveAttrAlloc(1U, newCoreAttrList);
    if (LwSciError_Success != error) {
        goto fn_exit;
    }

    error = LwSciSyncCoreCopySignalerExternalPrimitiveInfo(
                newCoreAttrList->attrs.signalerExternalPrimitiveInfo,
                coreAttrList->attrs.signalerExternalPrimitiveInfo,
                coreAttrList->attrs.valSize[LwSciSyncCoreKeyToIndex((uint32_t)
                        LwSciSyncInternalAttrKey_SignalerExternalPrimitiveInfo)]/
                        sizeof(void*));
    if (LwSciError_Success != error) {
        goto fn_exit;
    }
#endif

    error = CopySemaAttrList(coreAttrList,
            newCoreAttrList);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Copy timestampBufAttrList */
    if (NULL != coreAttrList->timestampBufAttrList) {
        error = LwSciBufAttrListClone(coreAttrList->timestampBufAttrList,
                &newCoreAttrList->timestampBufAttrList);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    /** Copy Ipc table */
    error = LwSciSyncCoreCopyIpcTable(&coreAttrList->ipcTable,
            &newCoreAttrList->ipcTable);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreAttrListSetActualPerm(
    LwSciSyncAttrList attrList,
    LwSciSyncAccessPerm actualPerm)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed.\n");
        LwSciCommonPanic();
    }

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    objAttrList->coreAttrList->attrs.actualPerm = actualPerm;
}

/** copies public attribute to Core attribute type */
static void PublicAttrToCoreAttr(
    const void* pair,
    CoreAttribute* coreAttr)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    const LwSciSyncAttrKeyValuePair* publicAttr =
            (const LwSciSyncAttrKeyValuePair*) pair;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    coreAttr->index = LwSciSyncCoreKeyToIndex((uint32_t)publicAttr->attrKey);
    coreAttr->value = publicAttr->value;
    coreAttr->len = publicAttr->len;
}

/** copies internal attribute to Core attribute type */
static void InternalAttrToCoreAttr(
    const void* pair,
    CoreAttribute* coreAttr)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    const LwSciSyncInternalAttrKeyValuePair* internalAttr =
            (const LwSciSyncInternalAttrKeyValuePair*) pair;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    coreAttr->index = LwSciSyncCoreKeyToIndex((uint32_t)internalAttr->attrKey);
    coreAttr->value = internalAttr->value;
    coreAttr->len = internalAttr->len;
}

static void CoreAttrToPublicAttr(
    const CoreAttribute* coreAttr,
    void* pair)
{
    uint32_t attrKey;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    LwSciSyncAttrKeyValuePair* publicAttr =
        (LwSciSyncAttrKeyValuePair*) pair;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    attrKey = LwSciSyncCoreIndexToKey(coreAttr->index);
    LwSciCommonMemcpyS(&publicAttr->attrKey, sizeof(publicAttr->attrKey),
                                   &attrKey, sizeof(attrKey));
    publicAttr->value = coreAttr->value;
    publicAttr->len = coreAttr->len;
}

static void CoreAttrToInternalAttr(
    const CoreAttribute* coreAttr,
    void* pair)
{
    uint32_t attrKey;
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    LwSciSyncInternalAttrKeyValuePair* internalAttr =
        (LwSciSyncInternalAttrKeyValuePair*) pair;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    attrKey = LwSciSyncCoreIndexToKey(coreAttr->index);
    LwSciCommonMemcpyS(&internalAttr->attrKey, sizeof(internalAttr->attrKey),
                                     &attrKey, sizeof(attrKey));
    internalAttr->value = coreAttr->value;
    internalAttr->len = coreAttr->len;
}

static LwSciError ValidateCoreAttribute(
    const CoreAttribute* attr)
{
    LwSciError error = LwSciError_Success;
    size_t keyIdx = 0;
    size_t elemSize;
    size_t maxElements;

    keyIdx = attr->index;
    elemSize = LwSciSyncCoreKeyInfo[keyIdx].elemSize;
    maxElements = LwSciSyncCoreKeyInfo[keyIdx].maxElements;
    if (!IsValueLenSane(attr->len, elemSize, maxElements)) {
        LWSCI_ERR_ULONG("Invalid argument: len: \n", attr->len);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == attr->value) {
        LWSCI_ERR_STR("Invalid argument: value: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    return error;
}

static const void* GetPublicConstPairAt(
    const void* pairArray,
    size_t index)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    const LwSciSyncAttrKeyValuePair* publicPairArray =
            (const LwSciSyncAttrKeyValuePair*) pairArray;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    return (const void*) &publicPairArray[index];
}

static const void* GetInternalConstPairAt(
    const void* pairArray,
    size_t index)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    const LwSciSyncInternalAttrKeyValuePair* internalPairArray =
            (const LwSciSyncInternalAttrKeyValuePair*) pairArray;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    return (const void*) &internalPairArray[index];
}

static void* GetPublicPairAt(
    void* pairArray,
    size_t index)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    LwSciSyncAttrKeyValuePair* publicPairArray =
            (LwSciSyncAttrKeyValuePair*) pairArray;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    return (void*) &publicPairArray[index];
}

static void* GetInternalPairAt(
    void* pairArray,
    size_t index)
{
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    LwSciSyncInternalAttrKeyValuePair* internalPairArray =
            (LwSciSyncInternalAttrKeyValuePair*) pairArray;
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    return (void*) &internalPairArray[index];
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreGetSignalerUseExternalPrimitive(
    LwSciSyncAttrList attrList,
    bool* signalerUseExternalPrimitive)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    /** validate all input args */
    if (NULL == signalerUseExternalPrimitive) {
        LWSCI_ERR_STR("Invalid signalerUseExternalPrimitive: NULL pointer\n");
        LwSciCommonPanic();
    }

    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed.\n");
        LwSciCommonPanic();
    }

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    *signalerUseExternalPrimitive =
            objAttrList->coreAttrList->signalerUseExternalPrimitive;
}

void LwSciSyncCoreGetLastExport(
    LwSciSyncAttrList attrList,
    LwSciIpcEndpoint* ipcEndpoint)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    /** validate all input args */
    if (NULL == ipcEndpoint) {
        LWSCI_ERR_STR("Invalid signalerUseExternalPrimitive: NULL pointer\n");
        LwSciCommonPanic();
    }

    error = LwSciSyncCoreAttrListValidate(attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreAttrListValidate failed.\n");
        LwSciCommonPanic();
    }

    LwSciSyncCoreAttrListGetObjFromRef(attrList, &objAttrList);

    *ipcEndpoint = objAttrList->coreAttrList->lastExport;

}

void LwSciSyncCoreGetTimestampInfo(
    const LwSciSyncCoreAttrs* reconciledAttrs,
    const LwSciSyncAttrValTimestampInfo** timestampInfo)
{
    const size_t index = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfo);
    const size_t indexMulti = LwSciSyncCoreKeyToIndex(
        (uint32_t)LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti);
    LwSciSyncCoreAttrKeyState keyState = LwSciSyncCoreAttrKeyState_Empty;
    LwSciSyncCoreAttrKeyState keyStateMulti = LwSciSyncCoreAttrKeyState_Empty;

    if ((reconciledAttrs == NULL) || (timestampInfo == NULL)) {
        LwSciCommonPanic();
    }

    keyState = reconciledAttrs->keyState[index];
    keyStateMulti = reconciledAttrs->keyState[indexMulti];

    if (keyStateMulti != LwSciSyncCoreAttrKeyState_Empty) {
        *timestampInfo =
            &reconciledAttrs->signalerTimestampInfoMulti[0];
        goto fn_exit;
    }

    if (keyState != LwSciSyncCoreAttrKeyState_Empty) {
        *timestampInfo = &reconciledAttrs->signalerTimestampInfo;
        goto fn_exit;
    }

fn_exit:
    return;
}
