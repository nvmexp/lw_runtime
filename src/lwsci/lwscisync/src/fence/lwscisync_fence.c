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
 * \brief <b>LwSciSync Fence Management Implementation</b>
 *
 * @b Description: This file implements LwSciSync fence management APIs
 *
 * The code in this file is organised as below:
 * -Core structures declaration.
 * -Core interfaces declaration.
 * -Public interfaces definition.
 * -Internal interfaces definition.
 * -Core interfaces definition.
 */

#include "lwscisync_fence.h"

#include <string.h>
#include "lwscilog.h"
#include "lwscisync.h"
#include "lwscisync_attribute_core.h"
#include "lwscisync_cpu_wait_context.h"
#include "lwscisync_object_core.h"
#include "lwscisync_primitive.h"
#include "lwscisync_timestamps.h"
#include "lwscisync_core.h"
#include "lwscisync_module.h"
#include "lwscicommon_covanalysis.h"

/******************************************************
 *            Core structures declaration
 ******************************************************/

/** Represents the LwSciSync core synchronization object */
typedef struct {
    /**
     * LwSciSync object creating a context for this fence. This member must NOT
     * be modified in between update and clearing of the LwSciSyncCoreFence.
     */
    LwSciSyncObj syncObj;
    /**
     * Primitive's id. This member must NOT be modified in between update and
     * clearing of the LwSciSyncCoreFence.
     */
    uint64_t id;
    /**
     * Threshold value. This member must NOT be modified in between update and
     * clearing of the LwSciSyncCoreFence.
     */
    uint64_t value;
    /**
     * slot index with the timestamp value. This member must NOT be modified
     * in between update and clearing of the LwSciSyncCoreFence.
     */
    uint32_t timestampSlot;
    /**
     * padding to match LwSciSyncFence size.
     */
    uint32_t padding[5];
} LwSciSyncCoreFence;

/** break the build on condition */
#define BUILD_BUG_ON(                               \
    condition)                                      \
    typedef char p__LINE__[ (condition) ? -1 : 1]

/** make sure those sizes match */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Directive, 4_9), "LwSciSync-ADV-MISRAC2012-007")
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 2_3), "LwSciSync-ADV-MISRAC2012-002")
BUILD_BUG_ON( sizeof(LwSciSyncCoreFence) != sizeof(LwSciSyncFence) );
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Directive, 4_9))

/******************************************************
 *             Core interfaces declaration
 ******************************************************/

/** check if syncFence is all zeros */
static bool isSyncFenceCleared(
    const LwSciSyncFence* syncFence);

/** check if desc is all zeros */
static bool isFenceDescEmpty(
    const LwSciSyncFenceIpcExportDescriptor* desc);

/** fence colwerter uses double casting to avoid
 * misra violations */
static inline LwSciSyncCoreFence* toCoreFence(
    LwSciSyncFence* syncFence)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    return (LwSciSyncCoreFence*)(void*) syncFence;
}

static inline const LwSciSyncCoreFence* toConstCoreFence(
    const LwSciSyncFence* syncFence)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    return (const LwSciSyncCoreFence*)(const void*) syncFence;
}

/** update the fence with verified parameters */
static LwSciError updateFence(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value,
    uint32_t slotIndex,
    LwSciSyncFence* syncFence);

static LwSciError ValidateFenceIdValue(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value);

/******************************************************
 *            Public interfaces definition
 ******************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncFenceClear(
    LwSciSyncFence* syncFence)
{
    LwSciSyncCoreFence* coreFence = toCoreFence(syncFence);
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (NULL == syncFence) {
        LWSCI_ERR_STR("invalid syncFence: NULL pointer\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncFence: %p\n", syncFence);

    /** allow cleared fence many clearings but don't actually do anything */
    if (false == isSyncFenceCleared(syncFence)) {
        /** Check for invalid sync object */
        error = LwSciSyncCoreObjValidate(coreFence->syncObj);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        LwSciSyncObjFreeObjAndRef(coreFence->syncObj);

        /** Clear the LwSciSyncFence */
        (void)memset(coreFence, 0, sizeof(LwSciSyncCoreFence));
    }

fn_exit:
    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncFenceDup(
    const LwSciSyncFence* srcSyncFence,
    LwSciSyncFence* dstSyncFence)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreFence* srcFence = toConstCoreFence(srcSyncFence);
    LwSciSyncCoreFence* dstFence = toCoreFence(dstSyncFence);
    bool cleared = false;

    LWSCI_FNENTRY("");

    if (NULL == srcSyncFence) {
        LWSCI_ERR_STR("invalid srcSyncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == dstSyncFence) {
        LWSCI_ERR_STR("invalid dstSyncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (srcSyncFence == dstSyncFence) {
        LWSCI_ERR_STR("src fence the same as dst fence\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("srcSyncFence: %p\n", srcSyncFence);
    LWSCI_INFO("dstSyncFence: %p\n", dstSyncFence);

    cleared = isSyncFenceCleared(srcSyncFence);
    if (true == cleared) {
        /* clearing dstSyncFence is enough to dup an empty fence */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto clear_fence;
    }

    /** Check for invalid sync object */
    error = LwSciSyncCoreObjValidate(srcFence->syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncObjRef(srcFence->syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

clear_fence:
    LwSciSyncFenceClear(dstSyncFence);

    *dstFence = *srcFence;

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncIpcExportFence(
    const LwSciSyncFence* syncFence,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncFenceIpcExportDescriptor* desc)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreFence* coreFence = toConstCoreFence(syncFence);
    LwSciSyncCoreObjId objId = {0};
    bool cleared = false;
    uint8_t* byteDesc = (uint8_t*) desc;
    const uint8_t* fieldOffset = NULL;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == syncFence) {
        LWSCI_ERR_STR("invalid syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == desc) {
        LWSCI_ERR_STR("invalid desc: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != error) {
        LWSCI_ERR_ULONG("invalid ipcEndpoint \n", ipcEndpoint);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** validate syncObj only if fence isn't cleared */
    cleared = isSyncFenceCleared(syncFence);
    if (cleared == false) {
        error = LwSciSyncCoreObjValidate(coreFence->syncObj);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    LWSCI_INFO("syncFence: %p\n", syncFence);
    LWSCI_INFO("ipcEndpoint: %" PRIu64 "\n", ipcEndpoint);
    LWSCI_INFO("desc: %p\n", desc);

    /** start from a clean state */
    (void)memset(desc, 0, sizeof(LwSciSyncFenceIpcExportDescriptor));

    /** clearing desc is enough to export an empty fence */
    if (true == cleared) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Serialize fence data */
    LwSciSyncCoreObjGetId(coreFence->syncObj, &objId);

    /** Forward origin IpcEndpoint in case of hop, else use current */
    if (0U == objId.ipcEndpoint ) {
        objId.ipcEndpoint = ipcEndpoint;
    }

    /* copy the fields according to the descriptor layout */
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    LwSciCommonMemcpyS(byteDesc + 0U, 16U, (uint8_t*) &objId, sizeof(objId));

    fieldOffset = (const uint8_t*) &coreFence->id;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    LwSciCommonMemcpyS(byteDesc + 16U, 8U, fieldOffset, sizeof(uint64_t));

    fieldOffset = (const uint8_t*) &coreFence->value;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    LwSciCommonMemcpyS(byteDesc + 24U, 8U, fieldOffset, sizeof(uint64_t));

    fieldOffset = (const uint8_t*) &coreFence->timestampSlot;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    LwSciCommonMemcpyS(byteDesc + 32U, 4U, fieldOffset, sizeof(uint32_t));

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncIpcImportFence(
    LwSciSyncObj syncObj,
    const LwSciSyncFenceIpcExportDescriptor* desc,
    LwSciSyncFence* syncFence)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreFence* coreFence = toCoreFence(syncFence);
    bool emptyFence = false;
    LwSciSyncCoreObjId objId = {0};
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP36_C), "LwSciSync-ADV-CERTC-001")
    const uint8_t* byteDesc = (const uint8_t*) desc;
    uint8_t* fieldOffset = NULL;
    bool isEqual = false;
    uint64_t threshold = 0U;

    LWSCI_FNENTRY("");

    /** validate all input args */
    if (NULL == desc) {
        LWSCI_ERR_STR("invalid desc: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == syncFence) {
        LWSCI_ERR_STR("invalid syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    emptyFence = isFenceDescEmpty(desc);

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("desc: %p\n", desc);
    LWSCI_INFO("syncFence: %p\n", syncFence);

    if (emptyFence) {
        /* Sufficient to clear fence for empty descriptor */
        LwSciSyncFenceClear(syncFence);
    } else {
        /** validate the unique ID of a non-empty fence */
        /* copy the fields according to the descriptor layout */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
        LwSciCommonMemcpyS((uint8_t*)&objId, sizeof(objId), byteDesc + 0U, 16U);

        LwSciSyncCoreObjMatchId(syncObj,
              (const LwSciSyncCoreObjId*)&objId, &isEqual);
        if (!isEqual) {
            LWSCI_ERR_STR("Fence descriptor does not match syncObj\n");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        threshold = *(const uint64_t*)(byteDesc + 24U);
        /* translate value to be locally-based */
        error = LwSciSyncCoreObjImportThreshold(
            syncObj, &threshold);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        /** associate non-empty fence with the syncObj*/
        error = LwSciSyncObjRef(syncObj);
        if (LwSciError_Success != error) {
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        LwSciSyncFenceClear(syncFence);

        coreFence->syncObj = syncObj;
        fieldOffset = (uint8_t*)&coreFence->id;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
        LwSciCommonMemcpyS(fieldOffset, sizeof(uint64_t), byteDesc + 16U, 8U);

        coreFence->value = threshold;

        fieldOffset = (uint8_t*)&coreFence->timestampSlot;
        LwSciCommonMemcpyS(fieldOffset, sizeof(uint32_t),
                byteDesc + 32U, 4U);
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncFenceWait(
    const LwSciSyncFence* syncFence,
    LwSciSyncCpuWaitContext context,
    int64_t timeoutUs)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreFence* coreFence = toConstCoreFence(syncFence);
    bool isWaiter = false;
    LwSciSyncAttrList attrList = NULL;
    bool cleared = false;
    LwSciSyncModule fenceModule = NULL;
    LwSciSyncModule contextModule = NULL;
    bool isDup = false;
    LwSciSyncCorePrimitive primitive;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == syncFence) {
        LWSCI_ERR_STR("Invalid syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == context) {
        LWSCI_ERR_STR("Invalid context: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if ((timeoutUs < -1) || (timeoutUs > LwSciSyncFenceMaxTimeout)) {
        LWSCI_ERR_SLONG("Invalid timeoutUs: \n", timeoutUs);
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** cleared fence counts as expired */
    cleared = isSyncFenceCleared(syncFence);
    if (true == cleared) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** This API must be called only from waiter */
    error = LwSciSyncObjGetAttrList(coreFence->syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncObjGetAttrList failed.\n");
        LwSciCommonPanic();
    }

    LwSciSyncCoreAttrListTypeIsCpuWaiter(attrList, &isWaiter);

    if (!isWaiter) {
        LWSCI_ERR_STR("Invalid operation\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreCpuWaitContextValidate(context);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    contextModule = LwSciSyncCoreCpuWaitContextGetModule(context);

    LwSciSyncCoreObjGetModule(coreFence->syncObj, &fenceModule);

    LwSciSyncCoreModuleIsDup(contextModule, fenceModule, &isDup);

    if (false == isDup) {
        LWSCI_ERR_STR("Incompatible modules in syncFence and waitContext \n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncFence: %p\n", syncFence);
    LWSCI_INFO("timeoutUs: %" PRId64 "\n", timeoutUs);

    LwSciSyncCoreObjGetPrimitive(coreFence->syncObj, &primitive);

    /** Make the wait call */
    error = LwSciSyncCoreWaitOnPrimitive(primitive,
            context, coreFence->id, coreFence->value, timeoutUs);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LwSciError LwSciSyncFenceGetTimestamp(
    const LwSciSyncFence* syncFence,
    uint64_t* timestampUS)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreFence* coreFence = toConstCoreFence(syncFence);
    LwSciSyncCoreTimestamps timestamps = NULL;
    LwSciSyncCorePrimitive primitive = NULL;
    LwSciSyncAttrList attrList = NULL;
    const void *value = NULL;
    size_t len = 0U;
    bool isCpuWaiter = false;
    bool supportsTimestamps = false;

    LWSCI_FNENTRY("");

    if (NULL == syncFence) {
        LWSCI_ERR_STR("Invalid output syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (true == isSyncFenceCleared(syncFence)) {
        LWSCI_INFO("Cleared fence %p has no timestamp\n", syncFence);
        error = LwSciError_ClearedFence;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == coreFence->syncObj) {
        LWSCI_ERR_STR("invalid syncFence: coreFence->syncObj: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreObjValidate(coreFence->syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncObjGetAttrList(coreFence->syncObj, &attrList);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncObjGetAttrList failed.\n");
        LwSciCommonPanic();
    }

    /** This API must be called only from a CPU waiter */
    LwSciSyncCoreAttrListTypeIsCpuWaiter(attrList, &isCpuWaiter);
    if (!isCpuWaiter) {
        LWSCI_ERR_STR("Invalid operation");
        error = LwSciError_BadParameter;
        goto fn_exit;
    }

    error = LwSciSyncAttrListGetAttr(attrList,
            LwSciSyncAttrKey_WaiterRequireTimestamps, &value, &len);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncAttrListGetAttr failed.\n");
        LwSciCommonPanic();
    }
    supportsTimestamps = *(const bool*)value;

    if (false == supportsTimestamps) {
        LWSCI_ERR_STR("Timestamps are not suppported\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == timestampUS) {
        LWSCI_ERR_STR("Invalid timestampUS: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (TIMESTAMPS_ILWALID_SLOT == coreFence->timestampSlot) {
        LWSCI_ERR_STR("syncFence does not use timestamps\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncFence: %p\n", syncFence);
    LWSCI_INFO("timestampUS: %p\n", timestampUS);

    LwSciCommonObjLock(&coreFence->syncObj->refObj);

    LwSciSyncCoreObjGetTimestamps(coreFence->syncObj, &timestamps);
    if (timestamps == NULL) {
        LwSciCommonObjUnlock(&coreFence->syncObj->refObj);
        LWSCI_ERR_STR("timestamps is NULL\n");
        LwSciCommonPanic();
    }

    LwSciSyncCoreObjGetPrimitive(coreFence->syncObj, &primitive);
    error = LwSciSyncCoreValidatePrimitiveIdValue(primitive,
            coreFence->id, coreFence->value);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    error = LwSciSyncCoreTimestampsGetTimestamp(timestamps,
            coreFence->timestampSlot, primitive, coreFence->id, timestampUS);

    LwSciCommonObjUnlock(&coreFence->syncObj->refObj);

    if (error != LwSciError_Success) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("*timestampUS: %" PRIu64 "\n", *timestampUS);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

/******************************************************
 *           Internal interfaces definition
 ******************************************************/

LwSciError LwSciSyncFenceUpdateFenceWithTimestamp(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value,
    uint32_t slotIndex,
    LwSciSyncFence* syncFence)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreTimestamps timestamps = NULL;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = ValidateFenceIdValue(syncObj, id, value);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    if (NULL == syncFence) {
        LWSCI_ERR_STR("NULL syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (TIMESTAMPS_ILWALID_SLOT != slotIndex) {
        LwSciSyncCoreObjGetTimestamps(syncObj, &timestamps);

        if (NULL == timestamps) {
            LWSCI_ERR_STR("timestamps not supported in this syncObj: NULL pointer\n");
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }

        if (false == LwSciSyncCoreTimestampsIsSlotValid(timestamps, slotIndex)) {
            LWSCI_ERR_UINT("Invalid slotIndex: \n", slotIndex);
            error = LwSciError_BadParameter;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("id: %" PRIu64 "\n", id);
    LWSCI_INFO("value: %" PRIu64 "\n", value);
    LWSCI_INFO("slotIndex: %" PRIu32 "\n", slotIndex);

    error = updateFence(syncObj, id, value, slotIndex, syncFence);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LwSciError LwSciSyncFenceUpdateFence(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value,
    LwSciSyncFence* syncFence)
{
    LwSciError error = LwSciError_Success;
    uint32_t slotIndex = UINT32_MAX;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = ValidateFenceIdValue(syncObj, id, value);
    if (error != LwSciError_Success) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == syncFence) {
        LWSCI_ERR_STR("NULL syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("id: %" PRIu64 "\n", id);
    LWSCI_INFO("value: %" PRIu64 "\n", value);

    error = updateFence(syncObj, id, value, slotIndex, syncFence);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncFenceExtractFence(
    const LwSciSyncFence* syncFence,
    uint64_t* id,
    uint64_t* value)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreFence* coreFence = toConstCoreFence(syncFence);
    bool isCleared = false;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == syncFence) {
        LWSCI_ERR_STR("invalid syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    isCleared = isSyncFenceCleared(syncFence);
    if (true == isCleared) {
        LWSCI_INFO("cleared syncFence: %p\n", syncFence);
        error = LwSciError_ClearedFence;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == id) {
        LWSCI_ERR_STR("invalid id: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == value) {
        LWSCI_ERR_STR("invalid value: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreObjValidate(coreFence->syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncFence: %p\n", syncFence);
    LWSCI_INFO("id: %p\n", id);
    LWSCI_INFO("value: %p\n", value);

    *id = coreFence->id;
    *value = coreFence->value;

    LWSCI_INFO("*id: %" PRIu64 "\n", *id);
    LWSCI_INFO("*value: %" PRIu64 "\n", *value);

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncFenceGetSyncObj(
    const LwSciSyncFence* syncFence,
    LwSciSyncObj* syncObj)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreFence* coreFence = toConstCoreFence(syncFence);

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == syncObj) {
        LWSCI_ERR_STR("Invalid output syncObj: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    if (NULL == syncFence) {
        LWSCI_ERR_STR("Invalid input syncFence: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (true == isSyncFenceCleared(syncFence)) {
        LWSCI_INFO("Cleared fence %p has no syncObj\n", syncFence);
        error = LwSciError_ClearedFence;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** Validate and return the sync object */
    error = LwSciSyncCoreObjValidate(coreFence->syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncFence: %p\n", syncFence);
    LWSCI_INFO("syncObj: %p\n", syncObj);

    *syncObj = coreFence->syncObj;

fn_exit:
    LWSCI_FNEXIT("");
    return error;
}

/******************************************************
 *             Core interfaces definition
 ******************************************************/

static bool isSyncFenceCleared(
    const LwSciSyncFence* syncFence)
{
    size_t i;
    size_t payloadSize = sizeof(syncFence->payload)/
            sizeof(syncFence->payload[0]);
    bool result = true;

    LWSCI_FNENTRY("");

    for (i = 0U; i < payloadSize; ++i) {
        if (0U != syncFence->payload[i]) {
            result = false;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    LWSCI_FNEXIT("");
    return result;
}

static bool isFenceDescEmpty(
    const LwSciSyncFenceIpcExportDescriptor* desc)
{
    size_t i;
    bool result = true;
    const uint8_t* offset = (const uint8_t*) desc;

    LWSCI_FNENTRY("");

    for (i = 0U; i < sizeof(*desc); ++i) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
        if (0U != *(offset + i)) {
            result = false;
            LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
            goto fn_exit;
        }
    }

fn_exit:
    LWSCI_FNEXIT("");
    return result;
}

static LwSciError updateFence(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value,
    uint32_t slotIndex,
    LwSciSyncFence* syncFence)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreFence* coreFence = toCoreFence(syncFence);

    /** relate this fence to syncObj */
    error = LwSciSyncObjRef(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    /** clear the previous data */
    LwSciSyncFenceClear(syncFence);
    /** Set values in the core fence structure */
    coreFence->syncObj = syncObj;
    coreFence->id = id;
    coreFence->value = value;
    coreFence->timestampSlot = slotIndex;

fn_exit:
    return error;
}

static LwSciError ValidateFenceIdValue(
    LwSciSyncObj syncObj,
    uint64_t id,
    uint64_t value)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCorePrimitive primitive = NULL;
    LwSciSyncCoreObjGetPrimitive(syncObj, &primitive);
    error = LwSciSyncCoreValidatePrimitiveIdValue(primitive,
            id, value);
    if (LwSciError_Success != error) {
        goto fn_exit;
    }

fn_exit:
    LWSCI_FNEXIT("");

    return error;
}
