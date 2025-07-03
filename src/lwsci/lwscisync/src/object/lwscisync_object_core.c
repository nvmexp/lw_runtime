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
 * \brief <b>LwSciSync Object Core Implementation</b>
 *
 * @b Description: This file implements LwSciSync core object management APIs
 *
 * The code in this file is organised as below:
 * -Core interfaces declaration.
 * -Core interfaces definition.
 */
#include "lwscisync_object_core.h"

#include "lwscicommon_objref.h"
#include "lwscicommon_os.h"
#include "lwscicommon_covanalysis.h"
#include "lwscilog.h"
#include "lwscisync_attribute_core.h"
#include "lwscisync_object_core_cluster.h"

/******************************************************
 *             Core interfaces definition
 ******************************************************/

LwSciError LwSciSyncCoreObjValidate(
    LwSciSyncObj syncObj)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;

    LWSCI_FNENTRY("");

    if (NULL == syncObj) {
        LWSCI_ERR_STR("Invalid argument: syncObj: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    if (LwSciSyncCoreGenerateObjHeader(coreObj) != coreObj->header) {
        LWSCI_ERR_STR("Invalid LwSciSyncObj\n");
        LwSciCommonPanic();
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

void LwSciSyncCoreObjGetId(
    LwSciSyncObj syncObj,
    LwSciSyncCoreObjId* objId)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;

    LWSCI_FNENTRY("");

    if (NULL == objId) {
        LWSCI_ERR_STR("Invalid argument: objId: NULL pointer\n");
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreObjValidate failed.\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("objId: %p\n", objId);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    *objId = coreObj->objId;

}

void LwSciSyncCoreObjMatchId(
    LwSciSyncObj syncObj,
    const LwSciSyncCoreObjId* objId,
    bool* isEqual)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;

    LWSCI_FNENTRY("");

    if (NULL == objId) {
        LWSCI_ERR_STR("Invalid argument: objId: NULL pointer\n");
        LwSciCommonPanic();
    }
    if (NULL == isEqual) {
        LWSCI_ERR_STR("Invalid argument: isEqual: NULL pointer\n");
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreObjValidate failed.\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("objId: %p\n", objId);
    LWSCI_INFO("isEqual: %p\n", isEqual);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    *isEqual = ((objId->moduleCntr == coreObj->objId.moduleCntr) &&
            (objId->ipcEndpoint == coreObj->objId.ipcEndpoint));

}

void LwSciSyncCoreObjGetModule(
    LwSciSyncObj syncObj,
    LwSciSyncModule* module)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;

    LWSCI_FNENTRY("");

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreObjValidate failed.\n");
        LwSciCommonPanic();
    }

    if (NULL == module) {
        LWSCI_ERR_STR("Invalid module: NULL pointer\n");
        LwSciCommonPanic();
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("module: %p\n", module);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    LwSciSyncCoreAttrListGetModule(coreObj->attrList, module);

}

void LwSciSyncCoreObjGetTimestamps(
    LwSciSyncObj syncObj,
    LwSciSyncCoreTimestamps* timestamps)
{
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciError error = LwSciError_Success;

    if (NULL == timestamps) {
        LWSCI_ERR_STR("Invalid argument: timestamps: NULL pointer\n");
        LwSciCommonPanic();
    }

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreObjValidate failed.\n");
        LwSciCommonPanic();
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    *timestamps = coreObj->timestamps;
}

void LwSciSyncCoreObjGetPrimitive(
    LwSciSyncObj syncObj,
    LwSciSyncCorePrimitive* primitive)
{
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;
    LwSciError error = LwSciError_Success;

    /** validate all input args */
    if (NULL == primitive) {
        LWSCI_ERR_STR("Invalid arguments: primitive is NULL\n");
        LwSciCommonPanic();
    }
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("LwSciSyncCoreObjValidate failed.\n");
        LwSciCommonPanic();
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    *primitive = coreObj->primitive;
}

LwSciError LwSciSyncObjGetAttrList(
    LwSciSyncObj syncObj,
    LwSciSyncAttrList* syncAttrList)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;

    LWSCI_FNENTRY("");

    /** validate all input args */
    if (NULL == syncAttrList) {
        LWSCI_ERR_STR("Invalid arguments: syncAttrList: NULL pointer\n");
        error = LwSciError_BadParameter;
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);
    LWSCI_INFO("syncAttrList: %p\n", syncAttrList);

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);

    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    *syncAttrList = coreObj->attrList;

    LWSCI_INFO("*syncAttrList: %d\n", *syncAttrList);

fn_exit:

    LWSCI_FNEXIT("");
    return error;
}

LwSciError LwSciSyncObjRef(
    LwSciSyncObj syncObj)
{
    LwSciError error = LwSciError_Success;

    LWSCI_FNENTRY("");

    /** validate all input args */
    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

    LWSCI_INFO("syncObj: %p\n", syncObj);

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP37_C), "LwSciSync-ADV-CERTC-002")
    error = LwSciCommonIncrAllRefCounts((void*)syncObj);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
    if (LwSciError_Success != error) {
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciSync-ADV-MISRAC2012-014")
        goto fn_exit;
    }

fn_exit:

    LWSCI_FNEXIT("");

    return error;
}

void LwSciSyncObjFreeObjAndRef(
    LwSciSyncObj syncObj)
{
    LwSciError error = LwSciError_Success;

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("Invalid or NULL LwSciSyncObj");
        LwSciCommonPanic();
    }

    LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 11_5), "LwSciSync-ADV-MISRAC2012-013")
    LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(EXP37_C), "LwSciSync-ADV-CERTC-002")
    LwSciCommonFreeObjAndRef((void*)syncObj, LwSciSyncCoreObjClose, NULL);
    LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 11_5))
}

LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciSync-ADV-MISRAC2012-009")
void LwSciSyncCoreObjClose(
    LwSciObj* objPtr)
{
    const LwSciSyncCoreObj* coreObj = NULL;

    if (NULL == objPtr) {
        LWSCI_ERR_STR("Invalid LwSciObj: NULL pointer");
        LwSciCommonPanic();
    }

    coreObj = LwSciCastObjToSyncCoreObj(objPtr);

    if (LwSciSyncCoreGenerateObjHeader(coreObj) != coreObj->header) {
        LWSCI_ERR_STR("Invalid LwSciObj");
        LwSciCommonPanic();
    }

    /** Deinitialize the backend primitive */
    LwSciSyncCoreDeinitPrimitive(coreObj->primitive);

    LwSciSyncCoreTimestampsDeinit(coreObj->timestamps);

    /** Free attr list */
    LwSciSyncAttrListFree(coreObj->attrList);

    LWSCI_FNEXIT("");
}

LwSciError LwSciSyncCoreObjImportThreshold(
    LwSciSyncObj syncObj,
    uint64_t* threshold)
{
    LwSciError error = LwSciError_Success;
    const LwSciSyncCoreObj* coreObj = NULL;
    LwSciObj* coreObjParam = NULL;

    if (NULL == threshold) {
        LWSCI_ERR_STR("NULL threshold!");
        LwSciCommonPanic();
    }

    error = LwSciSyncCoreObjValidate(syncObj);
    if (LwSciError_Success != error) {
        LWSCI_ERR_STR("syncObj is NULL");
        LwSciCommonPanic();
    }

    LwSciCommonGetObjFromRef(&syncObj->refObj, &coreObjParam);
    coreObj = LwSciCastObjToSyncCoreObj(coreObjParam);

    error = LwSciSyncCorePrimitiveImportThreshold(
        coreObj->primitive,
        threshold);
    return error;
}
