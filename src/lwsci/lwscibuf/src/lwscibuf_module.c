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
 * \brief <b>LwSciBuf Module Implementation</b>
 *
 * @b Description: This file implements LwSciBuf Module APIs
 *
 * The code in this file is organised as below:
 * -Public interfaces definition.
 */

#include "lwscibuf_module_priv.h"
#include "lwscicommon_objref.h"

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
void LwSciBufModuleCleanupObj(
    LwSciObj* obj)
{
    const LwSciBufModuleObjPriv* moduleObj = NULL;
    uint32_t iFaceType = (uint32_t)LwSciBufAllocIfaceType_SysMem;

    LWSCI_FNENTRY("");
    if (NULL == obj) {
        LwSciCommonPanic();
    }

    moduleObj = LwSciCastObjToBufModuleObjPriv(obj);
    /* print input parameters */
    LWSCI_INFO("Input: moduleObj: %p\n", moduleObj);

    if (LW_SCI_BUF_MODULE_MAGIC != moduleObj->magic) {
        LwSciCommonPanic();
    }

    /* close all alloc interfaces */
    for (iFaceType = (uint32_t)LwSciBufAllocIfaceType_SysMem;
            iFaceType < (uint32_t)LwSciBufAllocIfaceType_Max; iFaceType++) {
        if (NULL != moduleObj->iFaceOpenContext[iFaceType]) {
            LwSciBufAllocIfaceType allocType;
            LwSciCommonMemcpyS(&allocType, sizeof(allocType),
                                               &iFaceType, sizeof(iFaceType));
            /* close allocation interface only if it was initialized */
            LwSciBufAllocIfaceClose(allocType,
                moduleObj->iFaceOpenContext[iFaceType]);
        }
    }

    LwSciBufDevClose(moduleObj->dev);

    LWSCI_FNEXIT("");
}

/**
 * @brief public function definitions
 */
void LwSciBufModuleClose(
    LwSciBufModule module)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciObj* moduleObjParam = NULL;

    LWSCI_FNENTRY("");

    sciErr = LwSciBufModuleValidate(module);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to validate LwSciBufModule reference.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input variables */
    LWSCI_INFO("Input: module: %p\n", module);

    LwSciCommonGetObjFromRef(&module->refHeader, &moduleObjParam);
    LwSciCommonFreeObjAndRef(&module->refHeader, LwSciBufModuleCleanupObj, NULL);

    /* print output variables */
    LWSCI_INFO("Output: Module closed successfully\n");

  ret:
    LWSCI_FNEXIT("");
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufModuleOpen(
    LwSciBufModule* newModule)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciBufModuleRefPriv* moduleRef = NULL;
    LwSciRef* moduleRefParam = NULL;
    LwSciBufModuleObjPriv* moduleObj = NULL;
    LwSciObj* moduleObjParam = NULL;
    LwSciBufAllocIfaceType iFaceTypeEnum = LwSciBufAllocIfaceType_SysMem;
    uint32_t iFaceType = (uint32_t)LwSciBufAllocIfaceType_SysMem;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == newModule) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameters supplied to LwSciBufModuleOpen\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *newModule = NULL;

    /* print input variables */
    LWSCI_INFO("Input: LwSciBufModule* newModule: %p\n", newModule);

    sciErr = LwSciCommonAllocObjWithRef(sizeof(LwSciBufModuleObjPriv),
                sizeof(LwSciBufModuleRefPriv), &moduleObjParam,
                &moduleRefParam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Could not allocate memory for LwSciBufModulePriv\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    moduleObj = LwSciCastObjToBufModuleObjPriv(moduleObjParam);
    moduleRef = LwSciCastRefToBufModuleRefPriv(moduleRefParam);

    moduleObj->magic = LW_SCI_BUF_MODULE_MAGIC;

    sciErr = LwSciBufDevOpen(&moduleObj->dev);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("LwSciBufDevOpen failed\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto free_module;
    }

    for (iFaceType = (uint32_t)LwSciBufAllocIfaceType_SysMem;
            iFaceType < (uint32_t)LwSciBufAllocIfaceType_Max; iFaceType++) {

        LwSciCommonMemcpyS(&iFaceTypeEnum, sizeof(iFaceTypeEnum), &iFaceType, sizeof(iFaceType));
        sciErr = LwSciBufAllocIfaceOpen(iFaceTypeEnum, moduleObj->dev,
            &moduleObj->iFaceOpenContext[iFaceType]);
        if (LwSciError_Success != sciErr) {
            moduleObj->iFaceOpenContext[iFaceType] = NULL;
            if (LwSciError_NotSupported != sciErr) {
                LWSCI_ERR_UINT("Failed opening allocation interface for alloc type \n",
                    iFaceType);
                LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
                goto free_alloc_interface;
            } else {
                LWSCI_WARN("alloc inteface for alloc type %u not supported\n",
                    iFaceType);
                /* Ignore unsupported interface and continue opening remaining
                 * interfaces
                 */
                sciErr = LwSciError_Success;
            }
        }
    }

    *newModule = moduleRef;
    /* print output variables */
    LWSCI_INFO("Output: *newModule: %p\n", *newModule);
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

free_alloc_interface:
    {
        // Close any allocation interfaces that have been opened
        uint32_t interface = 0U;
        LwSciBufAllocIfaceType allocType;

        for (interface = (uint32_t)LwSciBufAllocIfaceType_SysMem;
                interface < iFaceType; ++interface) {
            LwSciCommonMemcpyS(&allocType, sizeof(allocType),
                                               &interface, sizeof(interface));
            LwSciBufAllocIfaceClose(allocType,
                    &moduleObj->iFaceOpenContext[interface]);
        }

        LwSciBufDevClose(moduleObj->dev);
    }

free_module:
    LwSciCommonFreeObjAndRef(&moduleRef->refHeader, NULL, NULL);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufModuleGetDevHandle(
    LwSciBufModule module,
    LwSciBufDev* dev)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufModuleObjPriv* moduleObj = NULL;
    LwSciObj* moduleObjParam = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == module) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufModuleGetDevHandle");
        sciErr = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    if (NULL == dev) {
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufModuleGetDevHandle");
        LwSciCommonPanic();
    }

    /* print input variables */
    LWSCI_INFO("Input: module: %p, LwSciBufDev* dev: %p\n", module, dev);

    LwSciCommonGetObjFromRef(&module->refHeader, &moduleObjParam);
    moduleObj = LwSciCastObjToBufModuleObjPriv(moduleObjParam);

    if ((NULL == moduleObj) || (LW_SCI_BUF_MODULE_MAGIC != moduleObj->magic)) {
        LWSCI_ERR_STR("Invalid LwSciBufModule. Panicking!!");
        LwSciCommonPanic();
        /* This goto is no-op and is used here to avoid cert-c violations. */
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    *dev = moduleObj->dev;

    /* print output variables */
    LWSCI_INFO("Output: LwSciBufDev dev: %p\n", *dev);

  ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

/**
 * @brief private function definitions
 */
LwSciError LwSciBufModuleDupRef(
    LwSciBufModule oldModule,
    LwSciBufModule* newModule)
{
    LwSciError sciErr = LwSciError_Success;
    LwSciRef* newModuleParam = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == newModule) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufModuleDupRef\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    sciErr = LwSciBufModuleValidate(oldModule);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to validate LwSciBufModule reference.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input variables */
    LWSCI_INFO("Input: oldmodule: %p, newmoduleAddr: %p\n", oldModule,
    newModule);

    sciErr = LwSciCommonDuplicateRef(&oldModule->refHeader, &newModuleParam);
    if (LwSciError_Success != sciErr) {
        LWSCI_ERR_STR("Failed to duplicate module instance\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *newModule = LwSciCastRefToBufModuleRefPriv(newModuleParam);

    /* print output variables */
    LWSCI_INFO("Output: newmodule: %p\n", *newModule);

  ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufModuleIsEqual(
    LwSciBufModule firstModule,
    LwSciBufModule secondModule,
    bool* isEqual)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufModuleObjPriv* firstModuleObj = NULL;
    const LwSciBufModuleObjPriv* secondModuleObj = NULL;
    LwSciObj* moduleObjParam = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((NULL == firstModule) || (NULL == secondModule) || (NULL == isEqual)) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufModuleDupRef\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *isEqual = false;

    /* print input variables */
    LWSCI_INFO("Input: firstmodule: %p, secondmodule: %p, isEqualAddr: %p\n",
    firstModule, secondModule, isEqual);

    LwSciCommonGetObjFromRef(&firstModule->refHeader, &moduleObjParam);
    firstModuleObj = LwSciCastObjToBufModuleObjPriv(moduleObjParam);

    LwSciCommonGetObjFromRef(&secondModule->refHeader, &moduleObjParam);
    secondModuleObj = LwSciCastObjToBufModuleObjPriv(moduleObjParam);

    if ((LW_SCI_BUF_MODULE_MAGIC != firstModuleObj->magic) ||
        (LW_SCI_BUF_MODULE_MAGIC != secondModuleObj->magic)) {
        LWSCI_ERR_STR("Invalid LwSciBufModule. Panicking!!\n");
        LwSciCommonPanic();
    }

    if (firstModuleObj == secondModuleObj) {
        *isEqual = true;
    }

    /* print output variables */
    LWSCI_INFO("Output: isEqual: %d\n", *isEqual);

  ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufModuleValidate(
    LwSciBufModule module)
{
    LwSciError sciErr = LwSciError_Success;
    const LwSciBufModuleObjPriv* moduleObj = NULL;
    LwSciObj* moduleObjParam = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if (NULL == module) {
        sciErr = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufModuleValidate\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    /* print input variables */
    LWSCI_INFO("Input: module: %p\n", module);

    LwSciCommonGetObjFromRef(&module->refHeader, &moduleObjParam);
    moduleObj = LwSciCastObjToBufModuleObjPriv(moduleObjParam);

    if (LW_SCI_BUF_MODULE_MAGIC != moduleObj->magic) {
        LWSCI_ERR_STR("Invalid LwSciBufModule. Panicking!!\n");
        LwSciCommonPanic();
    }

    /* print output variables */
    LWSCI_INFO("Output: module: %p\n", module);

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}

LwSciError LwSciBufModuleGetAllocIfaceOpenContext(
    LwSciBufModule module,
    LwSciBufAllocIfaceType allocType,
    void** openContext)
{
    LwSciError err = LwSciError_Success;
    const LwSciBufModuleObjPriv* moduleObj = NULL;
    LwSciObj* moduleObjParam = NULL;

    LWSCI_FNENTRY("");

    /* verify input parameters */
    if ((LwSciBufAllocIfaceType_Max <= allocType) || (NULL == openContext)) {
        err = LwSciError_BadParameter;
        LWSCI_ERR_STR("Bad parameter supplied to LwSciBufModuleGetAllocIfaceOpenContext");
        LWSCI_ERR_UINT("allocType: ", allocType);
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }

    err = LwSciBufModuleValidate(module);
    if (LwSciError_Success != err) {
        LWSCI_ERR_STR("Failed to validate LwSciBufModule reference.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    /* print input variables */
    LWSCI_INFO("Input: module: %p, allocType: %u, openContext: %p\n", module,
        allocType, openContext);

    *openContext = NULL;

    LwSciCommonGetObjFromRef(&module->refHeader, &moduleObjParam);
    moduleObj = LwSciCastObjToBufModuleObjPriv(moduleObjParam);

    if (NULL != moduleObj->iFaceOpenContext[(uint32_t)allocType]) {
        *openContext = moduleObj->iFaceOpenContext[(uint32_t)allocType];
    } else {
        err = LwSciError_ResourceError;
        LWSCI_ERR_UINT("Could not obtain openContext for allocType \n", allocType);
        LWSCI_ERR_STR("openContext might not be available because:\n");
        LWSCI_ERR_STR("1. Failed to open alloc interface during LwSciBufModuleOpen OR\n");
        LWSCI_ERR_UINT("2. alloc interface is not supported on the system: \n",
            (uint32_t)allocType);
    }

ret:
    LWSCI_FNEXIT("");
    return err;
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciBuf-ADV-MISRAC2012-010")
LwSciError LwSciBufCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible)
{
    LwSciError error = LwSciError_Success;
    bool platformCompatibility = false;

    (void)minorVer;

    LWSCI_FNENTRY("");

    /** Check for invalid arguments */
    if (NULL == isCompatible) {
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto ret;
    }
    *isCompatible = false;

    LWSCI_INFO("majorVer: %" PRIu32 "\n", majorVer);
    LWSCI_INFO("minorVer: %" PRIu32 "\n", minorVer);
    LWSCI_INFO("isCompatible: %p\n", isCompatible);

    if ((majorVer == LwSciBufMajorVersion) &&
        (minorVer <= LwSciBufMinorVersion)) {
        *isCompatible = true;
    }

    error = LwSciBufCheckPlatformVersionCompatibility(&platformCompatibility);
#if (LW_IS_SAFETY == 0)
    if (LwSciError_Success != error) {
        error = LwSciError_IlwalidState;
        LWSCI_ERR_STR("Failed to check platform compatibility\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
        goto failed;
    }
#else
    (void)error;
#endif

    *isCompatible = (*isCompatible && platformCompatibility);

    LWSCI_INFO("LwSciBufVersion compatible: %s\n",
        (*isCompatible) ? "True" : "False");

    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciBuf-ADV-MISRAC2012-015")
    goto ret;

#if (LW_IS_SAFETY == 0)
failed:
    *isCompatible = false;
#endif
ret:
    LWSCI_FNEXIT("");
    return error;
}
