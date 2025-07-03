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
 * \brief <b>LwSciSync semaphore transport implementation</b>
 *
 * @b Description: This file implements semaphore attribute transport logic
 */
#include "lwscisync_attribute_transport_semaphore.h"

#include "lwscicommon_libc.h"
#include "lwscilog.h"
#include "lwscisync_module.h"

LwSciError ExportSemaAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    void** txbufPtr,
    size_t* txbufSize)
{
    LwSciError error = LwSciError_Success;

    /** Return if nothing to export */
    if (coreAttrList->semaAttrList == NULL) {
        goto fn_exit;
    }

    /** WAR to ilwoke debug dump function */
    if (ipcEndpoint == 0U) {
#if (LW_IS_SAFETY == 0)
        error = LwSciBufAttrListDebugDump(coreAttrList->semaAttrList, txbufPtr,
                txbufSize);
#endif
        goto fn_exit;
    }

    /** Ilwoke appropriate export function based on sema attr list state */
    if (state == LwSciSyncCoreAttrListState_Unreconciled) {
        error = LwSciBufAttrListIpcExportUnreconciled(
                &coreAttrList->semaAttrList, 1U, ipcEndpoint, txbufPtr,
                txbufSize);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    } else if (state == LwSciSyncCoreAttrListState_Reconciled) {
        error = LwSciBufAttrListIpcExportReconciled(coreAttrList->semaAttrList,
                ipcEndpoint, txbufPtr, txbufSize);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    } else {
        LWSCI_ERR_STR("Exporting invalid buf attr list\n");
    }

fn_exit:
    return error;
}

LwSciError ImportSemaAttrList(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    bool importReconciled,
    const void* inputValue,
    size_t length)
{
    LwSciError error = LwSciError_Success;
    LwSciBufModule bufModule = NULL;
    LwSciSyncCoreAttrListObj* objAttrList = NULL;
    LwSciBufAttrList* semaAttrListArray = NULL;
    size_t semaAttrListCount = 0U;
    size_t i = 0U;

    LwSciSyncCoreModuleGetBufModule(module, &bufModule);

    if (!importReconciled) {
        error = LwSciBufAttrListIpcImportUnreconciled(bufModule, ipcEndpoint,
                inputValue, length, &coreAttrList->semaAttrList);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    } else {
        if (NULL != inputAttrList) {
            LwSciSyncCoreAttrListGetObjFromRef(inputAttrList,
                    &objAttrList);
            semaAttrListArray = (LwSciBufAttrList*) LwSciCommonCalloc(
                    objAttrList->numCoreAttrList,
                    sizeof(LwSciBufAttrList));
            if (semaAttrListArray == NULL) {
                LWSCI_ERR_STR("failed to allocate memory.\n");
                error = LwSciError_InsufficientMemory;
                goto fn_exit;
            }
            for (i = 0U; i < objAttrList->numCoreAttrList; i++) {
                if (objAttrList->coreAttrList[i].semaAttrList != NULL) {
                        semaAttrListArray[semaAttrListCount] =
                            objAttrList->coreAttrList[i].semaAttrList;
                    semaAttrListCount++;
                }
            }
            if (semaAttrListCount == 0U) {
                LwSciCommonFree(semaAttrListArray);
                semaAttrListArray = NULL;
            }
        }
        error = LwSciBufAttrListIpcImportReconciled(bufModule, ipcEndpoint,
                inputValue, length, semaAttrListArray, semaAttrListCount,
                &coreAttrList->semaAttrList);
        if (error != LwSciError_Success) {
            goto fn_exit;
        }
    }
fn_exit:
    if (semaAttrListArray != NULL) {
        LwSciCommonFree(semaAttrListArray);
    }
    return error;
}

