/*
 * Copyright (c) 2019-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <lwscisync.h>
#include <lwscisync_test_signaler.h>
#include <umd.h>

LwSciError cpuSignalStream(
    struct ThreadConf* conf,
    struct StreamResources* resources)
{
    size_t submitSize = LWSCISYNC_TEST_STANDARD_SUBMIT_SIZE;
    size_t j;
    LwSciError err = LwSciError_Success;

    for (j = 0; j < submitSize; ++j) {
        size_t i;
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        LwSciSyncObj syncObj = resources->syncObj;
        err = LwSciSyncObjGenerateFence(syncObj, &syncFence);
        if (err != LwSciError_Success) {
            return err;
        }

        for (i = 0; i < resources->upstreamSize; ++i) {
            IpcWrapperOld ipcWrapper = resources->upstreamIpcs[i];
            LwSciSyncFenceIpcExportDescriptor fenceDesc = {0};
            err = LwSciSyncIpcExportFence(&syncFence,
                    ipcWrapperGetEndpoint(ipcWrapper), &fenceDesc);
            if (err != LwSciError_Success) {
                return err;
            }

            err = ipcSend(ipcWrapper, &fenceDesc,
                    sizeof(LwSciSyncFenceIpcExportDescriptor));
            if (err != LwSciError_Success) {
                return err;
            }
        }
        LwSciSyncFenceClear(&syncFence);

        err = LwSciSyncObjSignal(syncObj);
        if (err != LwSciError_Success) {
            return err;
        }
    }

    return err;
}

LwSciSyncTestStatus standardSignaler(void* args)
{
    struct ThreadConf* conf = (struct ThreadConf*) args;
    size_t i;
    LwSciError err;
    LwSciSyncAttrList* unreconciledList = {NULL};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncAttrList signalerAttrList = NULL;
    LwSciSyncModule module = NULL;
    LwSciSyncObj syncObj = NULL;
    void** objAndListDescs = NULL;
    size_t* objAndListSizes = 0U;
    IpcWrapperOld* ipcWrappers = NULL;
    size_t* waiterAttrListSizes = NULL;
    void** waiterAttrListDescs = NULL;
    size_t upstreamSize = conf->upstreamSize;
    size_t* upstreamMsg = NULL;

    /* Initialize LwSciIpc */
    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        goto fail;
    }

    ipcWrappers = (IpcWrapperOld*) malloc(upstreamSize * sizeof(IpcWrapperOld));
    if (ipcWrappers == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    for (i = 0; i < conf->upstreamSize; ++i) {
        err = ipcInit(conf->upstream[i], &ipcWrappers[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    /* Signaler Setup/Init phase */
    /* Initialize the LwSciSync module */
    err = LwSciSyncModuleOpen(&module);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListCreate(module, &signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = conf->fillAttrList(signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* import waiters' attribute lists */
    waiterAttrListSizes = (size_t*) malloc(upstreamSize * sizeof(size_t));
    if (waiterAttrListSizes == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }
    waiterAttrListDescs =  (void**)malloc(upstreamSize * sizeof(void*));
    if (waiterAttrListDescs == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    /* one more for the local list */
    unreconciledList = (LwSciSyncAttrList*) malloc((upstreamSize + 1) * sizeof(LwSciSyncAttrList));
    if (unreconciledList == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    for (i = 0; i < upstreamSize; ++i) {
        err = ipcRecvFill(ipcWrappers[i], &waiterAttrListSizes[i],
                sizeof(size_t));
        if (err != LwSciError_Success) {
            goto fail;
        }
        waiterAttrListDescs[i] = malloc(waiterAttrListSizes[i]);
        if (waiterAttrListDescs[i] == NULL) {
            err = LwSciError_InsufficientMemory;
            goto fail;
        }

        err = ipcRecvFill(ipcWrappers[i], waiterAttrListDescs[i],
                waiterAttrListSizes[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }

        err = LwSciSyncAttrListIpcImportUnreconciled(module,
                ipcWrapperGetEndpoint(ipcWrappers[i]),
                waiterAttrListDescs[i], waiterAttrListSizes[i],
                &unreconciledList[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    unreconciledList[upstreamSize] = signalerAttrList;

    /* Reconcile Signaler and Waiter LwSciSyncAttrList */
    err = LwSciSyncAttrListReconcile(unreconciledList, upstreamSize + 1,
            &reconciledList, &newConflictList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Create LwSciSync object and get the syncObj */
    err = LwSciSyncObjAlloc(reconciledList, &syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    objAndListSizes = (size_t*) malloc(upstreamSize * sizeof(size_t));
    if (objAndListSizes == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    objAndListDescs = (void**) malloc(upstreamSize * sizeof(void*));
    if (objAndListDescs == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    /* Export attr list and obj */
    for (i = 0; i < upstreamSize; ++i) {
        err = LwSciSyncIpcExportAttrListAndObj(syncObj,
                conf->objExportPerm, ipcWrapperGetEndpoint(ipcWrappers[i]),
                &objAndListDescs[i], &objAndListSizes[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }

        err = ipcSend(ipcWrappers[i], &objAndListSizes[i], sizeof(size_t));
        if (err != LwSciError_Success) {
            goto fail;
        }
        err = ipcSend(ipcWrappers[i], objAndListDescs[i], objAndListSizes[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    if (conf->stream) {
        struct StreamResources resources;
        memset(&resources, 0, sizeof(resources));
        resources.upstreamIpcs = ipcWrappers;
        resources.upstreamSize = upstreamSize;
        resources.syncObj = syncObj;
        err = conf->stream(conf, &resources);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    /* Receive clean-up msg from upstream */
    upstreamMsg = (size_t*) calloc(upstreamSize, sizeof(size_t));
    if (upstreamMsg == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    for (i = 0; i < upstreamSize; ++i) {
        err = ipcRecvFill(ipcWrappers[i], &upstreamMsg[i],
                sizeof(size_t));
        if (err != LwSciError_Success) {
            printf("Failed to receive clean-up msg from upstream[%d],"
                    " err = %d\n", i, err);
            goto fail;
        }
    }

fail:
    for (i = 0; i < upstreamSize; ++i) {
        LwSciSyncAttrListAndObjFreeDesc(objAndListDescs[i]);
    }
    free(objAndListDescs);
    free(objAndListSizes);
    free(upstreamMsg);

    /* Free LwSciSyncObj */
    LwSciSyncObjFree(syncObj);

    /* Free Attribute list objects */
    LwSciSyncAttrListFree(reconciledList);
    LwSciSyncAttrListFree(newConflictList);
    LwSciSyncAttrListFree(signalerAttrList);
    for (i = 0; i < conf->upstreamSize; ++i) {
        LwSciSyncAttrListFree(unreconciledList[i]);
    }
    free(unreconciledList);

    for (i = 0; i < conf->upstreamSize; ++i) {
        free(waiterAttrListDescs[i]);
    }
    free(waiterAttrListDescs);
    free(waiterAttrListSizes);

    /* Deinitialize the LwSciSync module */
    LwSciSyncModuleClose(module);

    /* Deinitialize LwSciIpc */
    for (i = 0; i < conf->upstreamSize; ++i) {
        ipcDeinit(ipcWrappers[i]);
    }
    free(ipcWrappers);
    LwSciIpcDeinit();

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return LwSciSyncTestStatus::Failure;
    }

    return LwSciSyncTestStatus::Success;
}
