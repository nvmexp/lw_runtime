/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscisync_cpu_wait_context.h"

#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscisync_module.h"

/** Generate a header value for input cpu wait context */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
static inline uint64_t CpuWaitCtxGenerateHeader(
    LwSciSyncCpuWaitContext context)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_4), "LwSciSync-ADV-MISRAC2012-012")
    return (((uint64_t)context & (0xFFFF00000000FFFFULL)) |
            0x123456780000ULL);
}

/**
 * \brief LwSciSync CPU wait context structure
 */
struct LwSciSyncCpuWaitContextRec {
    /** Magic ID to ensure this is a valid cpu wait context */
    uint64_t header;
    /** Reference to LwSciSync module */
    LwSciSyncModule module;
    /** Platform-specific resources of the wait context */
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd;
};

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCpuWaitContextAlloc(
    LwSciSyncModule module,
    LwSciSyncCpuWaitContext* newContext)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCpuWaitContext context = NULL;
    LwSciSyncCoreRmBackEnd rmBackEnd = NULL;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (NULL == newContext) {
        LWSCI_ERR_STR("Invalid argument: newContext: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("newContext: %p\n", newContext);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    context = LwSciCommonCalloc(1U, sizeof(struct LwSciSyncCpuWaitContextRec));
    if (NULL == context) {
        LWSCI_ERR_STR("failed to alloc cpuWaitContextRec\n");
        error = LwSciError_InsufficientMemory;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciSyncCoreModuleGetRmBackEnd(module, &rmBackEnd);

    error = LwSciSyncCoreRmWaitCtxBackEndAlloc(rmBackEnd,
            &context->waitContextBackEnd);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("failed to alloc RmWaitContextBackEnd\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreModuleDup(module, &context->module);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    context->header = CpuWaitCtxGenerateHeader(context);

    *newContext = context;

fn_exit:
    if ((LwSciError_Success != error) && (NULL != context)) {
        LwSciSyncCoreRmWaitCtxBackEndFree(context->waitContextBackEnd);
        LwSciCommonFree(context);
    }

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCpuWaitContextFree(
    LwSciSyncCpuWaitContext context)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (NULL == context) {
        LWSCI_ERR_STR("Invalid arguments: context: NULL pointer\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreModuleValidate(context->module);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("context: %p\n", context);

    LwSciSyncCoreRmWaitCtxBackEndFree(context->waitContextBackEnd);
    LwSciSyncModuleClose(context->module);
    LwSciCommonFree(context);

    LWSCI_FNEXIT("");

fn_exit:
    return;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCoreCpuWaitContextValidate(
    LwSciSyncCpuWaitContext context)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    if (NULL == context) {
        LWSCI_ERR_STR("Invalid argument: context: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    if (context->header != CpuWaitCtxGenerateHeader(context)) {
        LWSCI_ERR_STR("Invalid LwSciSyncCpuWaitContext\n");
        LwSciCommonPanic();
    }

    error = LwSciSyncCoreModuleValidate(context->module);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    error = LwSciSyncCoreRmWaitCtxBackEndValidate(
        context->waitContextBackEnd);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciSyncModule LwSciSyncCoreCpuWaitContextGetModule(
    LwSciSyncCpuWaitContext context)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    if (NULL == context) {
        LWSCI_ERR_STR("Invalid argument: context: NULL pointer\n");
        LwSciCommonPanic();
    }
    return context->module;
}

LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 8_13), "LwSciSync-ADV-MISRAC2012-011")
LwSciSyncCoreRmWaitContextBackEnd LwSciSyncCoreCpuWaitContextGetBackEnd(
    LwSciSyncCpuWaitContext context)
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 8_13))
{
    if (NULL == context) {
        LWSCI_ERR_STR("Invalid argument: context: NULL pointer\n");
        LwSciCommonPanic();
    }
    return context->waitContextBackEnd;
}
