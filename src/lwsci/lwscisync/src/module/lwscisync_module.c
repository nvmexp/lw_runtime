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
 * \brief <b>LwSciSync Module Management Implementation</b>
 *
 * @b Description: This file implements LwSciSync module management APIs
 *
 * The code in this file is organised as below:
 * -Core interfaces declaration.
 * -Public interfaces definition.
 * -Core interfaces definition.
 */
#include "lwscisync_module.h"

#include "lwscicommon_objref.h"
#include "lwscicommon_libc.h"
#include "lwscicommon_os.h"
#include "lwscicommon_utils.h"
#include "lwscilog.h"
#include "lwscisync_module_priv.h"

/******************************************************
 *             Core interfaces declaration
 ******************************************************/

static uint64_t GenerateModuleHeader(
    const LwSciSyncCoreModule* coreModule);

static inline LwSciSyncModule CastObjToModule(LwSciRef* arg) {
  LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),"LwSciSync-ADV-MISRAC2012-013")
  LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciSync-ADV-MISRAC2012-001")
  LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
  return (LwSciSyncModule)(void*)((char*)(void*)arg
        - LW_OFFSETOF(struct LwSciSyncModuleRec, refModule));
  LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
  LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
}

/******************************************************
 *            Public interfaces definition
 ******************************************************/

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncModuleOpen(
    LwSciSyncModule* newModule)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreModule* coreModule = NULL;
    LwSciObj* coreModuleParam = NULL;
    LwSciRef* newModuleParam = NULL;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == newModule) {
        LWSCI_ERR_STR("Invalid arguments: newModule: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *newModule = NULL;

    LWSCI_INFO("newModule: %p\n", newModule);

    /** Allocate memory for new module */
    error = LwSciCommonAllocObjWithRef(sizeof(LwSciSyncCoreModule),
            sizeof(struct LwSciSyncModuleRec), &coreModuleParam,
            &newModuleParam);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to create module\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    coreModule = LwSciCastObjToSyncCoreModule(coreModuleParam);
    *newModule = CastObjToModule(newModuleParam);

    /** Set module header for future validation */
    coreModule->header = GenerateModuleHeader(coreModule);

    error = LwSciBufModuleOpen(&coreModule->bufModule);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to open LwSciBufModule\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto cleanup;
    }

    error = LwSciSyncCoreRmAlloc(&coreModule->backEnd);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to open LwRmHost1xHandle\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto cleanup;
    }

    LWSCI_INFO("*newModule: %p\n", *newModule);

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
    goto fn_exit;

cleanup:
    LwSciCommonFreeObjAndRef(&(*newModule)->refModule, LwSciSyncCoreModuleFree, NULL);

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

void LwSciSyncModuleClose(
    LwSciSyncModule module)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("module: %p\n", module);

    LwSciCommonFreeObjAndRef(&module->refModule, LwSciSyncCoreModuleFree, NULL);

fn_exit:

    LWSCI_FNEXIT("");
}

/******************************************************
 *             Core interfaces definition
 ******************************************************/

LwSciError LwSciSyncCoreModuleValidate(
    LwSciSyncModule module)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreModule* coreModule = NULL;
    LwSciObj* coreModuleParam = NULL;

    LWSCI_FNENTRY("");

    if (NULL == module) {
        LWSCI_ERR_STR("Invalid argument: module: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("module: %p\n", module);

    LwSciCommonGetObjFromRef(&module->refModule, &coreModuleParam);

    coreModule = LwSciCastObjToSyncCoreModule(coreModuleParam);

    if (GenerateModuleHeader(coreModule) != coreModule->header) {
        LWSCI_ERR_STR("Invalid LwSciSyncModule\n");
        LwSciCommonPanic();
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
LwSciError LwSciSyncCoreModuleCntrGetNextValue(
    LwSciSyncModule module,
    uint64_t* cntrValue)
{
    LwSciError error = LwSciError_Success;
    LwSciSyncCoreModule* coreModule = NULL;
    LwSciObj* coreModuleParam = NULL;
    uint8_t addStatus = OP_FAIL;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreModuleValidate failed.\n");
        LwSciCommonPanic();
    }
    if (NULL == cntrValue) {
        LWSCI_ERR_STR("Invalid argument: cntrValue: NULL pointer\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("module: %p\n", module);
    LWSCI_INFO("cntrValue: %p\n", cntrValue);

    LwSciCommonGetObjFromRef(&module->refModule, &coreModuleParam);

    coreModule = LwSciCastObjToSyncCoreModule(coreModuleParam);

    LwSciCommonObjLock(&module->refModule);

    *cntrValue = coreModule->moduleCounter;
    u64Add(coreModule->moduleCounter, 1U,
           &(coreModule->moduleCounter), &addStatus);
    if (OP_SUCCESS != addStatus) {
        LWSCI_ERR_STR("Not enough unique identifiers left in module \n");
        error = LwSciError_Overflow;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LwSciCommonObjUnlock(&module->refModule);

fn_exit:

    LWSCI_FNEXIT("");
    return error;
}

static inline LwSciSyncModule CastRefToSyncModule(LwSciRef* arg) {
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5),"LwSciSync-ADV-MISRAC2012-013")
    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 1_2), "LwSciSync-ADV-MISRAC2012-001")
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 18_4), "LwSciSync-ADV-MISRAC2012-016")
    return (LwSciSyncModule)(void*)((char*)(void*)arg
        - LW_OFFSETOF(struct LwSciSyncModuleRec, refModule));
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 1_2))
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
}

LwSciError LwSciSyncCoreModuleDup(
    LwSciSyncModule module,
    LwSciSyncModule* dupModule)
{
    LwSciError error = LwSciError_Success;
    LwSciRef* dupModuleParam = NULL;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreModuleValidate failed.\n");
        LwSciCommonPanic();
    }
    if (NULL == dupModule) {
        LWSCI_ERR_STR("Invalid arguments: dupModule: NULL pointer\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("module: %p\n", module);
    LWSCI_INFO("dupModule: %p\n", dupModule);

    error = LwSciCommonDuplicateRef(&module->refModule, &dupModuleParam);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    *dupModule = CastRefToSyncModule(dupModuleParam);

fn_exit:

    LWSCI_FNEXIT("");
    return error;
}

void LwSciSyncCoreModuleIsDup(
    LwSciSyncModule module,
    LwSciSyncModule otherModule,
    bool* isDup)
{
    LwSciError error = LwSciError_Success;
    LwSciObj* coreModule = NULL;
    LwSciObj* otherCoreModule = NULL;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreModuleValidate failed.\n");
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreModuleValidate(otherModule);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreModuleValidate failed.\n");
        LwSciCommonPanic();
    }
    if (NULL == isDup) {
        LWSCI_ERR_STR("Invalid arguments: isDup: NULL pointer\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("module: %p\n", module);
    LWSCI_INFO("otherModule: %p\n", otherModule);
    LWSCI_INFO("isDup: %p\n", isDup);

    LwSciCommonGetObjFromRef(&module->refModule, &coreModule);

    LwSciCommonGetObjFromRef(&otherModule->refModule,
        &otherCoreModule);

    *isDup = (coreModule == otherCoreModule);

    LWSCI_INFO("*isDup: %d\n", *isDup);

}

void LwSciSyncCoreModuleGetBufModule(
    LwSciSyncModule module,
    LwSciBufModule* bufModule)
{
    const LwSciSyncCoreModule* coreModule = NULL;
    LwSciObj* coreModuleParam = NULL;

    if (NULL == bufModule) {
        LWSCI_ERR_STR("Invalid argument: bufModule is NULL\n");
        LwSciCommonPanic();
    }

    LwSciCommonGetObjFromRef(&module->refModule, &coreModuleParam);

    coreModule = LwSciCastObjToSyncCoreModule(coreModuleParam);

    *bufModule = coreModule->bufModule;

}

void LwSciSyncCoreModuleGetRmBackEnd(
    LwSciSyncModule module,
    LwSciSyncCoreRmBackEnd* backEnd)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreModule* coreModule = NULL;
    LwSciObj* coreModuleParam = NULL;

    /* Check for invalid arguments */
    error = LwSciSyncCoreModuleValidate(module);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Failed to validate module\n");
        LwSciCommonPanic();
    }

    if (NULL == backEnd) {
        LWSCI_ERR_STR("Invalid argument: backEnd is NULL\n");
        LwSciCommonPanic();
    }

    LwSciCommonGetObjFromRef(&module->refModule, &coreModuleParam);

    coreModule = LwSciCastObjToSyncCoreModule(coreModuleParam);

    *backEnd = coreModule->backEnd;

}

static uint64_t GenerateModuleHeader(
    const LwSciSyncCoreModule* coreModule)
{
    LWSCI_INFO("coreModule: %p\n", coreModule);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 11_4), "LwSciSync-ADV-MISRAC2012-012")
    return (((uint64_t)coreModule & (~0xFFFFULL)) | 0xABULL);
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreModuleFree(
    LwSciObj* objPtr)
{
    const LwSciSyncCoreModule* coreModule = NULL;

    if (NULL == objPtr) {
        LWSCI_ERR_STR("Invalid LwSciSyncModule: NULL pointer");
        LwSciCommonPanic();
    }

    coreModule = LwSciCastObjToSyncCoreModule(objPtr);

    if (GenerateModuleHeader(coreModule) != coreModule->header) {
        LWSCI_ERR_STR("Invalid LwSciSyncModule");
        LwSciCommonPanic();
    }

    LwSciBufModuleClose(coreModule->bufModule);
    LwSciSyncCoreRmFree(coreModule->backEnd);
}
