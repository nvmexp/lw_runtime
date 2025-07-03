/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscisync_core.h"
#include "lwscicommon_covanalysis.h"

#include "lwscibuf.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscilog.h"
#include "lwscisync.h"
#include "lwscisync_internal.h"

#include <unistd.h>

#define LW_SCI_SYNC_ENG_NAME_BIT_COUNT   16U
#define LW_SCI_SYNC_ENG_NAME_BIT_MASK \
        (((uint32_t)1U << LW_SCI_SYNC_ENG_NAME_BIT_COUNT) - 1U)
#define LW_SCI_SYNC_ENG_NAME_BIT_START   0U

#define LW_SCI_SYNC_ENG_INSTANCE_BIT_COUNT   8U
#define LW_SCI_SYNC_ENG_INSTANCE_BIT_MASK \
        (((uint32_t)1U << LW_SCI_SYNC_ENG_INSTANCE_BIT_COUNT) - 1U)
#define LW_SCI_SYNC_ENG_INSTANCE_BIT_START   LW_SCI_SYNC_ENG_NAME_BIT_COUNT

static LwSciError BufCheckVersionCompatibility(
    bool* isCompatible)
{
    LwSciError error = LwSciError_Success;
    bool isBufLibCompatible = false;

    error = LwSciBufCheckVersionCompatibility(LwSciBufMajorVersion,
            LwSciBufMinorVersion, &isBufLibCompatible);
    if (error != LwSciError_Success) {
        goto fn_exit;
    }

    *isCompatible = *isCompatible && isBufLibCompatible;

fn_exit:

    return error;
}

static void LwSciSyncHwEngCreateIdHelper(
    LwSciSyncHwEngName engName,
    uint32_t instance,
    int64_t* engId)
{
    uint32_t id = (((uint32_t)engName & LW_SCI_SYNC_ENG_NAME_BIT_MASK)
            << LW_SCI_SYNC_ENG_NAME_BIT_START)
        | ((instance & LW_SCI_SYNC_ENG_INSTANCE_BIT_MASK)
            << LW_SCI_SYNC_ENG_INSTANCE_BIT_START);
    *engId = (int64_t) id;
}

static LwSciError LwSciSyncHwEngNameValidate(
    LwSciSyncHwEngName engName)
{
    LwSciError err = LwSciError_Success;

    if (engName != LwSciSyncHwEngName_PCIe) {
        err = LwSciError_BadParameter;
    }

    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (isCompatible == NULL) {
        LWSCI_ERR_STR("Invalid arguments: isCompatible is null\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *isCompatible = false;

    LWSCI_INFO("majorVer: %" PRIu32 "\n", majorVer);
    LWSCI_INFO("minorVer: %" PRIu32 "\n", minorVer);
    LWSCI_INFO("isCompatible: %p\n", isCompatible);

    if ((majorVer == LwSciSyncMajorVersion) &&
        (minorVer <= LwSciSyncMinorVersion)) {
        *isCompatible = true;
    }
    error = BufCheckVersionCompatibility(isCompatible);
    if (error != LwSciError_Success) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

bool LwSciSyncCorePermLEq(
    LwSciSyncAccessPerm permA,
    LwSciSyncAccessPerm permB)
{
    return (((uint64_t)permA) | ((uint64_t)permB)) == ((uint64_t)permB);
}

bool LwSciSyncCorePermLessThan(
    LwSciSyncAccessPerm permA,
    LwSciSyncAccessPerm permB)
{
    return  LwSciSyncCorePermLEq(permA, permB) &&
            (((uint64_t)permA) != ((uint64_t)permB));
}

bool LwSciSyncCorePermValid(
    LwSciSyncAccessPerm perm)
{
    return
        (LwSciSyncCorePermLEq(perm, LwSciSyncAccessPerm_WaitSignal)) &&
        (0U != (uint64_t)perm);
}

LwSciError LwSciSyncHwEngCreateIdWithoutInstance(
    LwSciSyncHwEngName engName,
    int64_t* engId)
{
    LwSciError err = LwSciError_Success;

    LWSCI_FNENTRY("");

    err = LwSciSyncHwEngNameValidate(engName);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }

    if (NULL == engId) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciSyncHwEngCreateIdWithoutInstance");
        LWSCI_ERR_UINT("engName: \n", (uint32_t)engName);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }

    LWSCI_INFO("Input: engName: %u, engId: %p", engName, engId);

    LwSciSyncHwEngCreateIdHelper(engName, 0, engId);

    LWSCI_INFO("output: engId: 0x%x", *engId);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciSyncHwEngGetNameFromId(
    int64_t engId,
    LwSciSyncHwEngName* engName)
{
    LwSciError err = LwSciError_Success;

    uint32_t tmpEngName;
    LwSciSyncHwEngName name = LwSciSyncHwEngName_LowerBound;

    LWSCI_FNENTRY("");

    if (NULL == engName) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciSyncHwEngGetNameFromId");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }

    LWSCI_INFO("Input: endId: 0x%x, engName: %p", engId, engName);

    if (engId < 0L) {
        LWSCI_ERR_INT("engId id negative: ", (int32_t)engId);
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }
    if (engId > INT32_MAX) {
        /* We only use the lower 32 bits of the engine id */
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }

    tmpEngName = (((uint32_t)engId >> LW_SCI_SYNC_ENG_NAME_BIT_START) &
                    LW_SCI_SYNC_ENG_NAME_BIT_MASK);

    LwSciCommonMemcpyS(&name, sizeof(name), &tmpEngName, sizeof(tmpEngName));

    err = LwSciSyncHwEngNameValidate(name);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }
    *engName = name;

    LWSCI_INFO("Output: engName: %u", *engName);

ret:
    LWSCI_FNEXIT("");
    return err;
}

LwSciError LwSciSyncHwEngGetInstanceFromId(
    int64_t engId,
    uint32_t* instance)
{
    LwSciError err = LwSciError_Success;

    LwSciSyncHwEngName engName = LwSciSyncHwEngName_LowerBound;

    LWSCI_FNENTRY("");

    if (NULL == instance) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciSyncHwEngGetInstanceFromId");
        err = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }

    LWSCI_INFO("Input: endId: 0x%x, instance: %p", engId, instance);

    /* Validate that the engId is valid. Since instances aren't validated, we
     * check that we can extract an LwSciSyncHwEngName. */
    err = LwSciSyncHwEngGetNameFromId(engId, &engName);
    if (LwSciError_Success != err) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto ret;
    }

    *instance = (((uint32_t)engId >> LW_SCI_SYNC_ENG_INSTANCE_BIT_START) &
                    LW_SCI_SYNC_ENG_INSTANCE_BIT_MASK);

    LWSCI_INFO("Output: engine instance: %u", *instance);

ret:
    LWSCI_FNEXIT("");
    return err;
}

bool LwSciSyncHwEngineEqual(
    const LwSciSyncHwEngine* engineA,
    const LwSciSyncHwEngine* engineB)
{
    LwSciError error = LwSciError_Success;

    if ((NULL == engineA) || (NULL == engineB)) {
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreCheckHwEngineValues(engineA, 1);
    if (LwSciError_Success != error) {
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreCheckHwEngineValues(engineB, 1);
    if (LwSciError_Success != error) {
        LwSciCommonPanic();
    }

    /* We can't use memcmp on the LwSciSyncHwEngine structures since we don't
     * guarantee that padding is zero-initialized.
     *
     * However, for revisions, since we consider the entire union as the
     * revision, we can use memcmp. */
    return ((engineA->engNamespace == engineB->engNamespace) &&
            (engineA->rmModuleID == engineB->rmModuleID) &&
            (engineA->subEngineID == engineB->subEngineID) &&
            (memcmp(&engineA->rev, &engineB->rev, sizeof(engineA->rev)) == 0));
}

void LwSciSyncAppendHwEngineToArrayUnion(
    LwSciSyncHwEngine* dstEngineArray,
    size_t dstEngineArrayMaxLen,
    const LwSciSyncHwEngine* srcEngineArray,
    size_t srcEngineArrayLen,
    size_t* dstEngineArrayLen)
{
    if ((NULL == dstEngineArray) || (NULL == srcEngineArray) ||
            (NULL == dstEngineArrayLen)) {
        LwSciCommonPanic();
    }

    for (size_t i = 0U; i < srcEngineArrayLen; i++) {
        bool containsEngine = false;
        for (size_t j = 0U; j < *dstEngineArrayLen; j++) {
            containsEngine = LwSciSyncHwEngineEqual(
                    &srcEngineArray[i], &dstEngineArray[j]);
            if (containsEngine) {
                break;
            }
        }
        if (false == containsEngine) {
            if (*dstEngineArrayLen == dstEngineArrayMaxLen) {
                LwSciCommonPanic();
            }
            LwSciCommonMemcpyS(
                &dstEngineArray[*dstEngineArrayLen],
                dstEngineArrayMaxLen * sizeof(*dstEngineArray) - (*dstEngineArrayLen * sizeof(*dstEngineArray)),
                &srcEngineArray[i],
                sizeof(srcEngineArray[i]));
            *dstEngineArrayLen += 1U;
        }
    }
}

bool LwSciSyncCoreIsTopoIdC2c(
    LwSciIpcTopoId topoId)
{
    return (topoId.SocId != LWSCIIPC_SELF_SOCID);
}

LwSciError LwSciSyncCoreGetSyncTopoId(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncIpcTopoId* syncTopoId)
{
    LwSciError error = LwSciError_Success;

    error = LwSciSyncCoreValidateIpcEndpoint(ipcEndpoint);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (syncTopoId == NULL) {
        LWSCI_ERR_STR("NULL input parameters");
        LwSciCommonPanic();
    }

#if (LW_IS_SAFETY == 0)
    error = LwSciIpcEndpointGetTopoId(ipcEndpoint, &syncTopoId->topoId);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciIpcEndpointGetTopoId failed");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
#else
    syncTopoId->topoId.SocId = LWSCIIPC_SELF_SOCID;
    syncTopoId->topoId.VmId = LWSCIIPC_SELF_VMID;
#endif

#if !defined(__x86_64__)
    if (false == LwSciSyncCoreIsTopoIdC2c(syncTopoId->topoId)) {
            error = LwSciIpcEndpointGetVuid(ipcEndpoint, &syncTopoId->vuId);
            if (LwSciError_Success != error) {
                LWSCI_ERR_STR("LwSciIpcEndpointGetVuid failed");
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
                goto fn_exit;
            }
    } else { /* C2C case */
        /* LwSciIpcEndpointGetVuid() returns _NotSupported
           for C2C based ipcEndpoints, so replacing with a mock */
        syncTopoId->vuId = ((ipcEndpoint << 32) & 0xFFFFFFFF00000000) |
            (getpid() & 0x00000000FFFFFFFF);
    }
#else
    syncTopoId->vuId = ((ipcEndpoint << 32) & 0xFFFFFFFF00000000) |
        (getpid() & 0x00000000FFFFFFFF);
#endif

fn_exit:
    return error;
}

LwSciError LwSciSyncCoreIsIpcEndpointC2c(
    LwSciIpcEndpoint ipcEndpoint,
    bool* hasC2C)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncIpcTopoId syncTopoId = {0};
    *hasC2C = false;

    error = LwSciSyncCoreGetSyncTopoId(ipcEndpoint, &syncTopoId);
    if (LwSciError_Success != error) {
        LWSCI_ERR_INT("Something went wrong with LwSciIpc ", error);
        error = LwSciError_ResourceError;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *hasC2C = LwSciSyncCoreIsTopoIdC2c(syncTopoId.topoId);

fn_exit:
    return error;
}
