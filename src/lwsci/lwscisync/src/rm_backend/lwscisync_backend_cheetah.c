/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscisync_backend_tegra.h"

#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"

/**
 * \brief Represents RM backend
 */
struct LwSciSyncCoreRmBackEndRec {
    /** Host1x handle for syncpoint operations */
    LwRmHost1xHandle host1x;
};

/**
 * \brief Represents RM wait context backend
 */
struct LwSciSyncCoreRmWaitContextBackEndRec {
    /** Wait context handle for host1x */
    LwRmHost1xWaiterHandle waiterHandle;
};

LwSciError LwSciSyncCoreRmAlloc(
    LwSciSyncCoreRmBackEnd* backEnd)
{
    LwSciError error = LwSciError_Success;
    LwError lwErr = LwError_Success;
    LwRmHost1xOpenAttrs host1xOpenAttrs;

    if (NULL == backEnd) {
        LWSCI_ERR_STR("Null pointer to RM backend. Panicking!!\n");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    *backEnd = (LwSciSyncCoreRmBackEnd) LwSciCommonCalloc(1U,
            sizeof(struct LwSciSyncCoreRmBackEndRec));
    if (NULL == *backEnd) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    host1xOpenAttrs = LwRmHost1xGetDefaultOpenAttrs();
    lwErr = LwRmHost1xOpen(&(*backEnd)->host1x, host1xOpenAttrs);
    if (LwError_Success != lwErr) {
        error = LwSciError_ResourceError;
        LWSCI_ERR_INT("Failed to open LwRmHost1xHandle. LwError: \n", lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    if (LwSciError_Success != error) {
       LwSciSyncCoreRmFree(*backEnd);
       *backEnd = NULL;
    }

    return error;
}

void LwSciSyncCoreRmFree(
    LwSciSyncCoreRmBackEnd backEnd)
{
    if (NULL == backEnd) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL != backEnd->host1x) {
        LwRmHost1xClose(backEnd->host1x);
    }

    LwSciCommonFree(backEnd);

fn_exit:
    return;
}

LwRmHost1xHandle LwSciSyncCoreRmGetHost1xHandle(
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
    LwSciSyncCoreRmBackEnd backEnd)
{
    if (NULL == backEnd) {
        LWSCI_ERR_STR("Null RM backend. Panicking!!\n");
        LwSciCommonPanic();
    }
    return backEnd->host1x;
}

LwSciError LwSciSyncCoreRmWaitCtxBackEndAlloc(
    LwSciSyncCoreRmBackEnd rmBackEnd,
    LwSciSyncCoreRmWaitContextBackEnd* waitContextBackEnd)
{
    LwSciError error = LwSciError_Success;
    LwError lwErr = LwError_Success;
    LwRmHost1xHandle host1x = NULL;

    if (NULL == rmBackEnd) {
        LWSCI_ERR_STR("Null RM backend. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == waitContextBackEnd) {
        LWSCI_ERR_STR("Null pointer to ContextBackEnd. Panicking!!\n");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    *waitContextBackEnd = (LwSciSyncCoreRmWaitContextBackEnd)
            LwSciCommonCalloc(1U,
            sizeof(struct LwSciSyncCoreRmWaitContextBackEndRec));
    if (NULL == *waitContextBackEnd) {
        LWSCI_ERR_STR("Failed to allocate memory\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    host1x = LwSciSyncCoreRmGetHost1xHandle(rmBackEnd);

    lwErr = LwRmHost1xWaiterAllocate(&(*waitContextBackEnd)->waiterHandle,
            host1x);
    if (LwError_Success != lwErr) {
        error = LwSciError_ResourceError;
        LWSCI_ERR_INT("failed to allocate host1x waiter. LwError: \n", lwErr);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:
    if (LwSciError_Success != error) {
        LwSciSyncCoreRmWaitCtxBackEndFree(*waitContextBackEnd);
        *waitContextBackEnd = NULL;
    }

    return error;
}

void LwSciSyncCoreRmWaitCtxBackEndFree(
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd)
{
    if (NULL == waitContextBackEnd) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL != waitContextBackEnd->waiterHandle) {
        LwRmHost1xWaiterFree(waitContextBackEnd->waiterHandle);
    }

    LwSciCommonFree(waitContextBackEnd);

fn_exit:
    return;
}

LwSciError LwSciSyncCoreRmWaitCtxBackEndValidate(
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd)
{
    LwSciError error = LwSciError_Success;

    if (NULL == waitContextBackEnd) {
        LWSCI_ERR_STR("Null WaitContextBackEnd. Panicking!!\n");
        LwSciCommonPanic();
    }
    if (NULL == waitContextBackEnd->waiterHandle) {
        LWSCI_ERR_STR("Invalid host1x waiter handle: NULL pointer\n");
        error = LwSciError_BadParameter;
    }

    return error;
}

LwRmHost1xWaiterHandle LwSciSyncCoreRmWaitCtxGetWaiterHandle(
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd)
{
    if (NULL == waitContextBackEnd) {
        LWSCI_ERR_STR("Null WaitContextBackEnd. Panicking!!\n");
        LwSciCommonPanic();
    }
    return waitContextBackEnd->waiterHandle;
}
