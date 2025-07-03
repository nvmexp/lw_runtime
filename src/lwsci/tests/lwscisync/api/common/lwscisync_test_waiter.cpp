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
#include <lwscisync_test_waiter.h>
#include <umd.h>

LwSciError cpuWaitStream(
    struct ThreadConf* conf,
    struct StreamResources* resources)
{
    size_t submitSize = LWSCISYNC_TEST_STANDARD_SUBMIT_SIZE;
    size_t j;
    LwSciError err = LwSciError_Success;

    for (j = 0; j < submitSize; ++j) {
        size_t i;
        LwSciSyncFenceIpcExportDescriptor fenceDesc = {0};
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        LwSciSyncObj syncObj = resources->syncObj;

        err = ipcRecvFill(resources->downstreamIpc,
                &fenceDesc, sizeof(fenceDesc));
        if (err != LwSciError_Success) {
            return err;
        }

        err = LwSciSyncIpcImportFence(syncObj,
                &fenceDesc, &syncFence);
        if (err != LwSciError_Success) {
            return err;
        }

        for (i = 0; i < resources->upstreamSize; ++i) {
            IpcWrapperOld ipcWrapper = resources->upstreamIpcs[i];
            LwSciSyncFenceIpcExportDescriptor upstreamFenceDesc = {0};
            err = LwSciSyncIpcExportFence(&syncFence,
                    ipcWrapperGetEndpoint(ipcWrapper), &upstreamFenceDesc);
            if (err != LwSciError_Success) {
                return err;
            }

            err = ipcSend(ipcWrapper, &upstreamFenceDesc,
                    sizeof(LwSciSyncFenceIpcExportDescriptor));
            if (err != LwSciError_Success) {
                return err;
            }
        }

        err = LwSciSyncFenceWait(&syncFence,
                resources->waitContext, -1);
        if (err != LwSciError_Success) {
            return err;
        }

        LwSciSyncFenceClear(&syncFence);
    }

    return err;
}

LwSciSyncTestStatus standardWaiter(void* args)
{
    struct ThreadConf* conf = (struct ThreadConf*) args;
    size_t i;
    LwSciError err;
    LwSciSyncModule module = NULL;
    LwSciSyncAttrList appendedList = NULL;
    LwSciSyncAttrList waiterAttrList = NULL;
    LwSciSyncAttrList* unreconciledList = {NULL};
    void** upstreamAttrListDescs;
    size_t* upstreamAttrListSizes;
    void* appendedDesc;
    size_t appendedSize;
    LwSciSyncObj syncObj = NULL;
    IpcWrapperOld downstreamIpc = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    void* objAndListDesc = NULL;
    size_t objAndListSize = 0U;
    size_t upstreamSize = conf->upstreamSize;
    IpcWrapperOld* upstreamIpcs = NULL;
    void** upstreamOALDescs = NULL;
    size_t* upstreamOALSizes = NULL;
    size_t* upstreamMsg = NULL;
    size_t downstreamMsg = 0U;

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcInit(conf->downstream, &downstreamIpc);
    if (err != LwSciError_Success) {
        goto fail;
    }

    upstreamIpcs = (IpcWrapperOld*) malloc(upstreamSize * sizeof(IpcWrapperOld));
    if (upstreamIpcs == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    for (i = 0; i < conf->upstreamSize; ++i) {
        err = ipcInit(conf->upstream[i], &upstreamIpcs[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    /* Waiter Setup/Init phase */
    /* Initialize the LwSciSync module */
    err = LwSciSyncModuleOpen(&module);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncCpuWaitContextAlloc(module, &waitContext);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Get waiter's LwSciSyncAttrList from LwSciSync for CPU waiter */
    err = LwSciSyncAttrListCreate(module, &waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = conf->fillAttrList(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* import from upstream peers */
    upstreamAttrListSizes = (size_t*) malloc(upstreamSize * sizeof(size_t));
    if (upstreamAttrListSizes == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }
    upstreamAttrListDescs =  (void**) malloc(upstreamSize * sizeof(void*));
    if (upstreamAttrListDescs == NULL) {
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
        err = ipcRecvFill(upstreamIpcs[i], &upstreamAttrListSizes[i],
                sizeof(size_t));
        if (err != LwSciError_Success) {
            goto fail;
        }
        upstreamAttrListDescs[i] = malloc(upstreamAttrListSizes[i]);
        if (upstreamAttrListDescs[i] == NULL) {
            err = LwSciError_InsufficientMemory;
            goto fail;
        }

        err = ipcRecvFill(upstreamIpcs[i], upstreamAttrListDescs[i],
                upstreamAttrListSizes[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }

        err = LwSciSyncAttrListIpcImportUnreconciled(module,
                ipcWrapperGetEndpoint(upstreamIpcs[i]),
                upstreamAttrListDescs[i], upstreamAttrListSizes[i],
                &unreconciledList[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    unreconciledList[upstreamSize] = waiterAttrList;

    err = LwSciSyncAttrListAppendUnreconciled(
            unreconciledList, upstreamSize + 1, &appendedList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Export waiter's LwSciSyncAttrList */
    err = LwSciSyncAttrListIpcExportUnreconciled(&appendedList, 1,
            ipcWrapperGetEndpoint(downstreamIpc),
            &appendedDesc, &appendedSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcSend(downstreamIpc, &appendedSize, sizeof(size_t));
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcSend(downstreamIpc, appendedDesc, appendedSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcRecvFill(downstreamIpc, &objAndListSize, sizeof(size_t));
    if (err != LwSciError_Success) {
        goto fail;
    }
    objAndListDesc = malloc(objAndListSize);
    if (objAndListDesc == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    err = ipcRecvFill(downstreamIpc, objAndListDesc, objAndListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncIpcImportAttrListAndObj(module,
            ipcWrapperGetEndpoint(downstreamIpc),
            objAndListDesc, objAndListSize,
            &waiterAttrList, 1,
            conf->objImportPerm,
            ipcWrapperGetEndpoint(downstreamIpc), &syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Pass the object further upstream */
    upstreamOALSizes = (size_t*) malloc(upstreamSize * sizeof(size_t));
    if (upstreamOALSizes == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }
    upstreamOALDescs = (void**) malloc(upstreamSize * sizeof(void*));
    if (upstreamOALDescs == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    for (i = 0; i < upstreamSize; ++i) {
        err = LwSciSyncIpcExportAttrListAndObj(syncObj,
                conf->objExportPerm, ipcWrapperGetEndpoint(upstreamIpcs[i]),
                &upstreamOALDescs[i], &upstreamOALSizes[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }

        err = ipcSend(upstreamIpcs[i], &upstreamOALSizes[i], sizeof(size_t));
        if (err != LwSciError_Success) {
            goto fail;
        }
        err = ipcSend(upstreamIpcs[i], upstreamOALDescs[i], upstreamOALSizes[i]);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    if (conf->stream) {
        struct StreamResources resources = {
            .upstreamIpcs = upstreamIpcs,
            .upstreamSize = upstreamSize,
            .downstreamIpc = downstreamIpc,
            .syncObj = syncObj,
            .waitContext = waitContext,
        };
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
        err = ipcRecvFill(upstreamIpcs[i], &upstreamMsg[i],
                sizeof(size_t));
        if (err != LwSciError_Success) {
            printf("Failed to receive clean-up msg from upstream[%d],"
                    " err = %d\n", i, err);
            goto fail;
        }
    }
    /* Send clean-up msg to downstream */
    err = ipcSend(downstreamIpc, &downstreamMsg, sizeof(size_t));
    if (err != LwSciError_Success) {
        printf("Failed to send clean-up msg to downstream, err = %d\n", err);
    }

fail:
    for (i = 0; i < upstreamSize; ++i) {
        LwSciSyncAttrListAndObjFreeDesc(upstreamOALDescs[i]);
    }
    free(upstreamOALDescs);
    free(upstreamOALSizes);
    free(upstreamMsg);

    free(objAndListDesc);

    LwSciSyncObjFree(syncObj);

    LwSciSyncAttrListFree(appendedList);
    LwSciSyncAttrListFreeDesc(appendedDesc);

    LwSciSyncCpuWaitContextFree(waitContext);
    /* Deinitialize the LwSciSync module */
    LwSciSyncModuleClose(module);

    LwSciSyncAttrListFree(waiterAttrList);
    for (i = 0; i < upstreamSize; ++i) {
        LwSciSyncAttrListFree(unreconciledList[i]);
        free(upstreamAttrListDescs[i]);
    }
    free(unreconciledList);
    free(upstreamAttrListDescs);
    free(upstreamAttrListSizes);

    for (i = 0; i < conf->upstreamSize; ++i) {
        ipcDeinit(upstreamIpcs[i]);
    }
    free(upstreamIpcs);
    /* Deinitialize LwSciIpc */
    ipcDeinit(downstreamIpc);
    LwSciIpcDeinit();

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return LwSciSyncTestStatus::Failure;
    }

    return LwSciSyncTestStatus::Success;
}
