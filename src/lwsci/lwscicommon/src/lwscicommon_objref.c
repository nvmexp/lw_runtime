/*
 * Copyright (c) 2019-2022, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdlib.h>
#include <stdint.h>

#include "lwscilog.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_objref.h"
#include "lwscicommon_covanalysis.h"

#define LW_SCI_COMMON_REF_MAGIC 0xC001C0DEU

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciCommon-ADV-MISRAC2012-008")
static void LwSciCommonRefValidate(
    const LwSciRef* ref)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LWSCI_FNENTRY("");

    /* Check input parameters */
    if (NULL == ref) {
        LWSCI_ERR_STR("NULL input parameter\n");
        LwSciCommonPanic();
    }

    if (LW_SCI_COMMON_REF_MAGIC != ref->magicNumber) {
        LWSCI_ERR_STR("Invalid object as input parameter\n");
        LwSciCommonPanic();
    }

    LWSCI_FNEXIT("");
}

/*----------------------------------------------------------------------------
 * Public Interface definition
 *----------------------------------------------------------------------------*/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
LwSciError LwSciCommonAllocObjWithRef(
    size_t objSize,
    size_t refSize,
    LwSciObj** objPtr,
    LwSciRef** refPtr)
{
    LwSciRef* ref = NULL;
    LwSciObj* objRef = NULL;
    LwSciError sciErr = LwSciError_Success;

    LWSCI_FNENTRY("");

    /* Check function argument & set to default */
    if ((NULL == refPtr) || (NULL == objPtr) || (sizeof(LwSciRef) > refSize) ||
        (sizeof(LwSciObj) > objSize)) {
        LWSCI_ERR_STR("Invalid input parameters\n");
        LwSciCommonPanic();
    }
    *refPtr = NULL;
    *objPtr = NULL;

    /* Print incoming values */
    LWSCI_INFO("refsize: %.10u, objsize: %.10u, refPtr: %p, objPtr: %p\n",
        refSize, objSize, refPtr, objPtr);

    /* Allocate memory for reference object */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    ref = (LwSciRef* )LwSciCommonCalloc(1, refSize);
    if (NULL == ref) {
        LWSCI_ERR_STR("Failed to allocate memory for reference\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    /* Initialize lock */
    sciErr = LwSciCommonMutexCreate(&ref->refLock);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to create mutex lock\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto free_ref;
    }

    /* Set necessary values in newly created objects */
    ref->magicNumber = LW_SCI_COMMON_REF_MAGIC;
    ref->refCount = 0;
    ref->size = refSize;

    /* Allocate memory for underlying object */
    ref->objPtr = LwSciCommonCalloc(1, objSize);
    if (NULL == ref->objPtr) {
        LWSCI_ERR_STR("Failed to allocate memory for object\n");
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto free_refLock;
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    objRef = (LwSciObj*)ref->objPtr;
    objRef->refCount = 0;

    /* Initialize lock */
    sciErr = LwSciCommonMutexCreate(&objRef->objLock);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to create mutex lock\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto free_obj;
    }

    /* Set refCount of reference object & underlying object as 1 since we have
     *  a consumer now */
    ref->refCount = 1;
    objRef->refCount = 1;

    /* Set output variable */
    *refPtr = ref;
    *objPtr = objRef;

    /* Print outgoing values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u, "
        "objRefCount: %u\n", ref, ref->objPtr, ref->magicNumber,
        ref->refCount, objRef->refCount);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
    goto ret;

free_obj:
    LwSciCommonFree(ref->objPtr);
free_refLock:
    LwSciCommonMutexDestroy(&ref->refLock);
free_ref:
    LwSciCommonFree(ref);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

static void LwSciCommonFreeRef(
    LwSciRef* ref,
    void (*refCleanupCallback)(LwSciRef* ref))
{
    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(ref);

    /* Acquire lock */
    LwSciCommonMutexLock(&ref->refLock);
    if (0 < ref->refCount) {
        ref->refCount--;

        if (0 == ref->refCount) {
            if (NULL != refCleanupCallback) {
                refCleanupCallback(ref);
            }
            /* Release & destroy lock */
            LwSciCommonMutexUnlock(&ref->refLock);

            LwSciCommonMutexDestroy(&ref->refLock);

            /* free entire reference structure so lock is also freed in same call*/
            LwSciCommonFree(ref);
        } else {
            /* Release lock */
            LwSciCommonMutexUnlock(&ref->refLock);
        }
    } else {
        LWSCI_ERR_STR("Reference count cannot go negative\n");

        LwSciCommonMutexUnlock(&ref->refLock);

        /* This indicates that somehow LwSci either ignored the mutex elsewhere
         * or the LwSciRef was corrupted. */
        LwSciCommonPanic();
    }

    LWSCI_FNEXIT("");
}

static void LwSciCommonFreeObj(
    LwSciObj* obj,
    void (*objCleanupCallback)(LwSciObj* obj))
{
    LWSCI_FNENTRY("");

    /* Acquire lock */
    LwSciCommonMutexLock(&obj->objLock);

    if (0 < obj->refCount) {

        obj->refCount--;

        if (0 == obj->refCount) {
            if (NULL != objCleanupCallback) {
                objCleanupCallback(obj);
            }

            /* Release & destroy lock */
            LwSciCommonMutexUnlock(&obj->objLock);

            LwSciCommonMutexDestroy(&obj->objLock);

            /* free entire reference structure so lock is also freed in same call*/
            LwSciCommonFree(obj);
        } else {
            /* Release lock */
            LwSciCommonMutexUnlock(&obj->objLock);
        }
    } else {
        LWSCI_ERR_STR("Reference count cannot go negative\n");

        LwSciCommonMutexUnlock(&obj->objLock);

        /* This indicates that somehow LwSci either ignored the mutex elsewhere
         * or the LwSciObj was corrupted. */
        LwSciCommonPanic();
    }

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonFreeObjAndRef(
    LwSciRef* ref,
    void (*objCleanupCallback)(LwSciObj* obj),
    void (*refCleanupCallback)(LwSciRef* ref))
{
    LwSciObj* objRef = NULL;

    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(ref);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    objRef = (LwSciObj*)ref->objPtr;

    if (NULL == objRef) {
        LwSciCommonPanic();
    }

    /* Print incoming values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u, "
        "objRefCount: %u\n", ref, ref->objPtr, ref->magicNumber,
        ref->refCount, objRef->refCount);

    LwSciCommonFreeRef(ref, refCleanupCallback);

    LwSciCommonFreeObj(objRef, objCleanupCallback);

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
LwSciError LwSciCommonIncrAllRefCounts(
    LwSciRef* ref)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciObj* objRef = NULL;

    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(ref);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    objRef = (LwSciObj*)ref->objPtr;

    /* Print incoming values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u, "
        "objRefCount: %u\n", ref, ref->objPtr, ref->magicNumber,
        ref->refCount, objRef->refCount);

    /* Aquire lock */
    LwSciCommonMutexLock(&ref->refLock);

    /* Aquire lock */
    LwSciCommonMutexLock(&objRef->objLock);

    if (((INT32_MAX - 1) < ref->refCount) ||
        ((INT32_MAX - 1) < objRef->refCount)) {
        sciErr = LwSciError_IlwalidState;
        LWSCI_ERR_STR("Reference Count overflow\n");
    } else {
        ref->refCount++;
        objRef->refCount++;
    }

    /* Release lock */
    LwSciCommonMutexUnlock(&objRef->objLock);

    /* Release lock */
    LwSciCommonMutexUnlock(&ref->refLock);

    /* Print outgoing values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u, "
        "objRefCount: %u\n", ref, ref->objPtr, ref->magicNumber,
        ref->refCount, objRef->refCount);

    LWSCI_FNEXIT("");
    return sciErr;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonGetObjFromRef(
    const LwSciRef* ref,
    LwSciObj** objPtr)
{
    LWSCI_FNENTRY("");

    if (NULL == objPtr) {
        LwSciCommonPanic();
    }

    LwSciCommonRefValidate(ref);

    /* Print incoming values */
    LWSCI_INFO("refptr: %p, magic: 0x%.8x, objptr: %p, outptr: %p\n", ref,
        ref->magicNumber, ref->objPtr, objPtr);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    *objPtr = (LwSciObj*)ref->objPtr;

    /* Print outgoing values */
    LWSCI_INFO("refptr: %p, objPtr: %p, ouput: %p\n", ref, ref->objPtr,
    *objPtr);

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonRefLock(
    LwSciRef* ref)
{
    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(ref);

    /* Print incoming values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    /* Aquire lock */
    LwSciCommonMutexLock(&ref->refLock);

    /* Print outgoing values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonObjLock(
    const LwSciRef* ref)
{
    LwSciObj* obj = NULL;

    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(ref);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    obj = (LwSciObj*)ref->objPtr;

    /* Print incoming values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    /* Aquire lock */
    LwSciCommonMutexLock(&obj->objLock);

    /* Print outgoing values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonRefUnlock(
    LwSciRef* ref)
{
    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(ref);

    /* Print incoming values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    /* Release lock */
    LwSciCommonMutexUnlock(&ref->refLock);

    /* Print outgoing values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonObjUnlock(
    const LwSciRef* ref)
{
    LwSciObj* obj = NULL;

    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(ref);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    obj = (LwSciObj*)ref->objPtr;

    /* Print incoming values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    /* Release lock */
    LwSciCommonMutexUnlock(&obj->objLock);

    /* Print outgoing values */
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u\n",
        ref, ref->objPtr, ref->magicNumber, ref->refCount);

    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
LwSciError LwSciCommonDuplicateRef(
    const LwSciRef* oldRef,
    LwSciRef** newRef)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciObj* objRef = NULL;
    LwSciRef* inputRef = NULL;

    LWSCI_FNENTRY("");

    LwSciCommonRefValidate(oldRef);

    /* Check function arguments */
    if (NULL == newRef) {
        LWSCI_ERR_STR("NULL input parameter\n");
        LwSciCommonPanic();
    }

    *newRef = NULL;

    /* Print incoming values */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    objRef = (LwSciObj*)oldRef->objPtr;
    LWSCI_INFO("refptr: %p, objPtr: %p, magic : 0x%.8x, refCount: %u, "
        "objRefCount: %u\n", oldRef, oldRef->objPtr,
        oldRef->magicNumber, oldRef->refCount, objRef->refCount);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciCommon-ADV-MISRAC2012-005")
    inputRef = (LwSciRef*)LwSciCommonCalloc(1, oldRef->size);
    if (NULL == inputRef) {
        sciErr = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    /* Initialize lock */
    sciErr = LwSciCommonMutexCreate(&inputRef->refLock);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to create mutex lock\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto free_ref;
    }

    /* Set necessary values in newly created objects */
    inputRef->magicNumber = oldRef->magicNumber;
    inputRef->size = oldRef->size;
    inputRef->refCount = 0;
    inputRef->objPtr = objRef;

    /* Increment refCount of reference object & underlying object since we have
     *  a consumer now */
    sciErr = LwSciCommonIncrAllRefCounts(inputRef);
    if (LwSciError_Success != sciErr) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto free_refLock;
    }

    *newRef = inputRef;

    LWSCI_INFO("oldrefptr: %p, oldobjPtr: %p, oldmagic : 0x%.8x, "
        "oldrefCount: %u, objRefCount: %u, newrefptr: %p, newobjPtr: %p, "
        "newmagic : 0x%.8x, newrefCount: %u\n", oldRef,
        oldRef->objPtr, oldRef->magicNumber, oldRef->refCount,
        objRef->refCount, inputRef, inputRef->objPtr, inputRef->magicNumber,
        inputRef->refCount);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
    goto ret;

free_refLock:
    LwSciCommonMutexDestroy(&inputRef->refLock);
free_ref:
    LwSciCommonFree(inputRef);
ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

