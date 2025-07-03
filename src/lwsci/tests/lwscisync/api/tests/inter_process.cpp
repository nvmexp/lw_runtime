/*
 * Copyright (c) 2019-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

/*
 * This test illustrates inter-thread intra-process use case,
 * where within same process
 * two threads are spawned and uses LwSciSync public APIs for synchronization.
 * The shared memory illustrated here should be managed by LwSciBuf APIs.
 * The test uses semaphore signalling for data transfers between signaler and
 * waiter processes, but in real application it should be done by LwSciIpc.
 */

#include <semaphore.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <lwscisync_test_common.h>
#include <lwscisync_test_signaler.h>
#include <lwscisync_test_waiter.h>
#include <lwscisync.h>
#include <umd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ipc_wrapper.h>
#include <cinttypes>

#define PEER_DESTROY_SYNC_OBJ (0x55F0)

typedef struct ThreadArgsRec
{
    uint32_t submitSize;
    uint32_t index;
    uint32_t reportProgress;
} ThreadArgs;

/* data prepared by the test initialization */
struct LocalThreadArgs {
    ThreadArgs targs;

    LwSciError(* fillSignalerAttrList)(LwSciSyncAttrList list);
#ifdef LWSCISYNC_EMU_SUPPORT
    LwSciError(* fillExternalPrimitiveInfo)(LwSciSyncAttrList list, TestResources res);
#endif
    LwSciError(* fillWaiterAttrList)(LwSciSyncAttrList list);
    LwSciError(* rmTestSetup)(TestResources* res);
    LwSciError(* rmTestMapSemaphore)(TestResources res, LwSciSyncObj syncObj);
    void(* rmTestTeardown)(TestResources res);
    LwSciError(* signalStreamFrame)(
        LwSciSyncObj syncObj,
        TestResources resource,
        LwSciSyncFenceIpcExportDescriptor* fenceDesc,
        IpcWrapperOld ipcWrapper);
    LwSciError(* waitStreamFrame)(
        LwSciSyncObj syncObj,
        LwSciSyncCpuWaitContext waitContext,
        TestResources resource,
        LwSciSyncFenceIpcExportDescriptor* fenceDesc,
        IpcWrapperOld ipcWrapper);

    LwSciSyncAttrValTimestampInfo* expectedTimestampInfo;
    LwSciSyncAttrValTimestampInfo* expectedTimestampInfoMulti;
};

LwSciError interProcessCpuSignalStreamFrame(
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFenceIpcExportDescriptor* fenceDesc,
    IpcWrapperOld ipcWrapper)
{
    LwSciError err;
    LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
    LwSciSyncFenceIpcExportDescriptor localDesc = {0};

    if (fenceDesc != NULL) {
        printf("fenceDesc should be NULL\n");
        err = LwSciError_BadParameter;
        return err;
    }

    err = LwSciSyncObjGenerateFence(syncObj, &syncFence);
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncIpcExportFence(&syncFence,
            ipcWrapperGetEndpoint(ipcWrapper), &localDesc);
    if (err != LwSciError_Success) {
        return err;
    }
    LwSciSyncFenceClear(&syncFence);

    err = ipcSend(ipcWrapper, &localDesc,
            sizeof(LwSciSyncFenceIpcExportDescriptor));
    if (err != LwSciError_Success) {
        return err;
    }

    return LwSciSyncObjSignal(syncObj);
}

LwSciError interProcessUmdSignalStreamFrame(
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFenceIpcExportDescriptor* fenceDesc,
    IpcWrapperOld ipcWrapper)
{
    LwSciError err;
    LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
    LwSciSyncFenceIpcExportDescriptor localDesc = {0};

    if (fenceDesc != NULL) {
        printf("fenceDesc should be NULL\n");
        err = LwSciError_BadParameter;
        return err;
    }

    err = umdGetPostLwSciSyncFence(syncObj, resource, &syncFence);
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncIpcExportFence(&syncFence,
            ipcWrapperGetEndpoint(ipcWrapper), &localDesc);
    if (err != LwSciError_Success) {
        return err;
    }
    LwSciSyncFenceClear(&syncFence);

    err = ipcSend(ipcWrapper, &localDesc,
            sizeof(LwSciSyncFenceIpcExportDescriptor));
    if (err != LwSciError_Success) {
        return err;
    }

    return LwSciError_Success;
}

LwSciError interProcessCpuWaitStreamFrame(
    LwSciSyncObj syncObj,
    LwSciSyncCpuWaitContext waitContext,
    TestResources resource,
    LwSciSyncFenceIpcExportDescriptor* fenceDesc,
    IpcWrapperOld ipcWrapper)
{
    LwSciError err;
    LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
    LwSciSyncFenceIpcExportDescriptor localFenceDesc = {0};

    if (fenceDesc != NULL) {
        printf("fenceDesc should be NULL\n");
    }

    err = ipcRecvFill(ipcWrapper, &localFenceDesc, sizeof(localFenceDesc));
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncIpcImportFence(syncObj,
                                  &localFenceDesc,
                                  &syncFence);
    if (err != LwSciError_Success) {
        return err;
    }

    /* Perform Wait on LwSciSyncFence */
    err = LwSciSyncFenceWait(&syncFence,
            waitContext, -1);
    if (err != LwSciError_Success) {
        return err;
    }
    LwSciSyncFenceClear(&syncFence);

    return LwSciError_Success;
}

static LwSciError InterProcessTestAssertSignalerTimestampInfo(
    LwSciSyncAttrList reconciledList,
    LwSciSyncAttrValTimestampInfo* timestampInfo,
    LwSciSyncAttrValTimestampInfo* timestampInfoMulti)
{
    LwSciSyncAttrValTimestampInfo actualSignalerTimestampInfo;
    LwSciSyncAttrValTimestampInfo expectedTimestampInfo;
    const void* signalerTimestampInfo = NULL;
    size_t signalerTimestampInfoLen = 0U;

    LwSciError err = LwSciError_Success;

    if ((timestampInfo == NULL) && (timestampInfoMulti == NULL)) {
        goto ret;
    }

    if (timestampInfo != nullptr) {
        err = LwSciSyncAttrListGetSingleInternalAttr(
            reconciledList, LwSciSyncInternalAttrKey_SignalerTimestampInfo,
            &signalerTimestampInfo, &signalerTimestampInfoLen);
        expectedTimestampInfo = *timestampInfo;
    }
    if (timestampInfoMulti != nullptr) {
        err = LwSciSyncAttrListGetSingleInternalAttr(
            reconciledList, LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti,
            &signalerTimestampInfo, &signalerTimestampInfoLen);
        expectedTimestampInfo = *timestampInfoMulti;
    }
    EXPECT_EQ(err, LwSciError_Success);

    EXPECT_NE(signalerTimestampInfo, nullptr);
    EXPECT_NE(signalerTimestampInfoLen, 0U);

    actualSignalerTimestampInfo =
            *(const LwSciSyncAttrValTimestampInfo*)signalerTimestampInfo;

    EXPECT_EQ(expectedTimestampInfo.format,
            actualSignalerTimestampInfo.format);
    EXPECT_EQ(expectedTimestampInfo.scaling.scalingFactorNumerator,
            actualSignalerTimestampInfo.scaling.scalingFactorNumerator);
    EXPECT_EQ(expectedTimestampInfo.scaling.scalingFactorDenominator,
            actualSignalerTimestampInfo.scaling.scalingFactorDenominator);
    EXPECT_EQ(expectedTimestampInfo.scaling.sourceOffset,
            actualSignalerTimestampInfo.scaling.sourceOffset);

ret:
    return err;
}

LwSciError interProcessCpuWaitWithTimestampsStreamFrame(
    LwSciSyncObj syncObj,
    LwSciSyncCpuWaitContext waitContext,
    TestResources resource,
    LwSciSyncFenceIpcExportDescriptor* fenceDesc,
    IpcWrapperOld ipcWrapper)
{
    LwSciError err;
    LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
    LwSciSyncFenceIpcExportDescriptor localFenceDesc = {0};
    uint64_t timestamp = 0U;

    if (fenceDesc != NULL) {
        printf("fenceDesc should be NULL\n");
    }

    err = ipcRecvFill(ipcWrapper, &localFenceDesc, sizeof(localFenceDesc));
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncIpcImportFence(syncObj,
                                  &localFenceDesc,
                                  &syncFence);
    if (err != LwSciError_Success) {
        return err;
    }

    /* Perform Wait on LwSciSyncFence */
    err = LwSciSyncFenceWait(&syncFence,
            waitContext, -1);
    if (err != LwSciError_Success) {
        return err;
    }
    err = LwSciSyncFenceGetTimestamp(&syncFence, &timestamp);
    if (err != LwSciError_Success) {
        return err;
    }

    LwSciSyncFenceClear(&syncFence);

    return LwSciError_Success;
}

LwSciError interProcessUmdWaitStreamFrame(
    LwSciSyncObj syncObj,
    LwSciSyncCpuWaitContext waitContext,
    TestResources resource,
    LwSciSyncFenceIpcExportDescriptor* fenceDesc,
    IpcWrapperOld ipcWrapper)
{
    LwSciError err;
    LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
    LwSciSyncFenceIpcExportDescriptor localFenceDesc = {0};

    if (fenceDesc != NULL) {
        printf("fenceDesc should be NULL\n");
    }

    err = ipcRecvFill(ipcWrapper, &localFenceDesc,
                sizeof(localFenceDesc));
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncIpcImportFence(syncObj,
                                  &localFenceDesc,
                                  &syncFence);
    if (err != LwSciError_Success) {
        return err;
    }

    err = umdWaitOnPreLwSciSyncFence(resource, &syncFence);
    if (err != LwSciError_Success) {
        return err;
    }

    LwSciSyncFenceClear(&syncFence);

    return LwSciError_Success;
}

static LwSciSyncTestStatus signaler(struct LocalThreadArgs* largs)
{
    size_t i;
    LwSciError err;
    TestResources resource = NULL;
    ThreadArgs* v = &largs->targs;
    uint32_t submitSize = v->submitSize;
    LwSciSyncAttrList unreconciledList[2] = {NULL};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncAttrList signalerAttrList = NULL;
    LwSciSyncModule module = NULL;
    LwSciSyncObj syncObj = NULL;
    LwSciSyncAttrList importedUnreconciledAttrList = NULL;
    void* objAndListDesc = NULL;
    size_t objAndListSize = 0U;
    LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
    IpcWrapperOld ipcWrapper = NULL;
    LwSciSyncFenceIpcExportDescriptor* fenceDesc = NULL;
    size_t waiterAttrListSize = 0U;
    void* waiterAttrListDesc = NULL;
    uint64_t destroySyncObj = 0;
    bool isValid = false;

    /* Initialize LwSciIpc */
    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcInit("lwscisync_a_0", &ipcWrapper);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Signaler Setup/Init phase */
    /* Initialize the LwSciSync module */
    err = LwSciSyncModuleOpen(&module);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Get signaler's LwSciSyncAttrList from UMD */
    err = LwSciSyncAttrListCreate(module, &signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    if (largs->rmTestSetup != NULL) {
        err = largs->rmTestSetup(&resource);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    err = largs->fillSignalerAttrList(signalerAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

#ifdef LWSCISYNC_EMU_SUPPORT
    if (largs->fillExternalPrimitiveInfo != NULL) {
        err = largs->fillExternalPrimitiveInfo(signalerAttrList, resource);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }
#endif

    err = ipcRecvFill(ipcWrapper, &waiterAttrListSize,
            sizeof(waiterAttrListSize));
    if (err != LwSciError_Success) {
        goto fail;
    }

    waiterAttrListDesc = malloc(waiterAttrListSize);
    if (waiterAttrListDesc == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    err = ipcRecvFill(ipcWrapper, waiterAttrListDesc, waiterAttrListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListIpcImportUnreconciled(module,
            ipcWrapperGetEndpoint(ipcWrapper),
            waiterAttrListDesc, waiterAttrListSize,
            &importedUnreconciledAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    unreconciledList[0] = signalerAttrList;
    unreconciledList[1] = importedUnreconciledAttrList;

    /* Reconcile Signaler and Waiter LwSciSyncAttrList */
    err = LwSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList,
            &newConflictList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncAttrListValidateReconciled(reconciledList,
            unreconciledList, 2, &isValid);
    if (err != LwSciError_Success || !isValid) {
        goto fail;
    }

    /* empty validation to verify it fails */
    NegativeTestPrint();
    err = LwSciSyncAttrListValidateReconciled(reconciledList,
            NULL, 0, &isValid);
    if (err != LwSciError_BadParameter) {
        goto fail;
    }

    /* Assert on the LwSciSyncInternalAttrKey_SignalerTimestampInfo and
     * LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti keys */
    err = InterProcessTestAssertSignalerTimestampInfo(
        reconciledList, largs->expectedTimestampInfo,
        largs->expectedTimestampInfoMulti);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Create LwSciSync object and get the syncObj */
    err = LwSciSyncObjAlloc(reconciledList, &syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Export attr list and obj and signal waiter*/
    err = LwSciSyncIpcExportAttrListAndObj(syncObj,
        LwSciSyncAccessPerm_WaitOnly, ipcWrapperGetEndpoint(ipcWrapper),
        &objAndListDesc, &objAndListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcSend(ipcWrapper, &objAndListSize, sizeof(size_t));
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = ipcSend(ipcWrapper, objAndListDesc, objAndListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    if (largs->rmTestMapSemaphore != NULL) {
        err = largs->rmTestMapSemaphore(resource, syncObj);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    /* Signaler streaming phase */
    for (i = 0; i < submitSize; ++i) {
        LwSciSyncFenceIpcExportDescriptor* descriptor = NULL;

        err = largs->signalStreamFrame(
                syncObj,
                resource,
                descriptor,
                ipcWrapper);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    err = ipcRecvFill(ipcWrapper, &destroySyncObj, sizeof(destroySyncObj));
    if (err == LwSciError_Success && destroySyncObj != PEER_DESTROY_SYNC_OBJ) {
        printf("Received message is not PEER_DESTROY_SYNC_OBJ\n");
    }

fail:
    if (largs->rmTestTeardown != NULL) {
        largs->rmTestTeardown(resource);
    }

    if (fenceDesc != NULL) {
        free(fenceDesc);
    }

    LwSciSyncAttrListAndObjFreeDesc(objAndListDesc);
    free(waiterAttrListDesc);

    /* Free LwSciSyncObj */
    LwSciSyncObjFree(syncObj);

    /* Free Attribute list objects */
    LwSciSyncAttrListFree(reconciledList);
    LwSciSyncAttrListFree(newConflictList);
    LwSciSyncAttrListFree(signalerAttrList);
    LwSciSyncAttrListFree(importedUnreconciledAttrList);

    /* Deinitialize the LwSciSync module */
    LwSciSyncModuleClose(module);

    /* Deinitialize LwSciIpc */
    ipcDeinit(ipcWrapper);
    LwSciIpcDeinit();

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return LwSciSyncTestStatus::Failure;
    }

    return LwSciSyncTestStatus::Success;
}

static LwSciSyncTestStatus waiter(struct LocalThreadArgs* largs)
{
    size_t i;
    LwSciError err;
    TestResources resource = NULL;
    ThreadArgs* v = &largs->targs;
    bool reportProgress = v->reportProgress;
    uint32_t submitSize = v->submitSize;
    LwSciSyncModule module = NULL;
    LwSciSyncAttrList waiterAttrList = NULL;
    size_t waiterAttrListSize = 0U;
    void* waiterListDesc;
    LwSciSyncObj syncObj = NULL;
    IpcWrapperOld ipcWrapper = NULL;
    LwSciSyncCpuWaitContext waitContext = NULL;
    void* objAndListDesc = NULL;
    size_t objAndListSize = 0U;
    LwSciSyncFenceIpcExportDescriptor* fenceDesc = NULL;
    struct timespec beginTimespec;
    struct timespec endTimespec;
    const size_t progressMask = 0xfffff;
    uint64_t destroySyncObj = PEER_DESTROY_SYNC_OBJ;

    err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = ipcInit("lwscisync_a_1", &ipcWrapper);
    if (err != LwSciError_Success) {
        goto fail;
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
    err = largs->fillWaiterAttrList(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Export waiter's LwSciSyncAttrList */
    err = LwSciSyncAttrListIpcExportUnreconciled(&waiterAttrList, 1,
            ipcWrapperGetEndpoint(ipcWrapper),
            &waiterListDesc, &waiterAttrListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcSend(ipcWrapper,
            &waiterAttrListSize,
            sizeof(waiterAttrListSize));
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcSend(ipcWrapper, waiterListDesc, waiterAttrListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = ipcRecvFill(ipcWrapper, &objAndListSize, sizeof(size_t));
    if (err != LwSciError_Success) {
        goto fail;
    }
    objAndListDesc = malloc(objAndListSize);
    if (objAndListDesc == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    err = ipcRecvFill(ipcWrapper, objAndListDesc, objAndListSize);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = LwSciSyncIpcImportAttrListAndObj(module,
            ipcWrapperGetEndpoint(ipcWrapper),
            objAndListDesc, objAndListSize,
            &waiterAttrList, 1,
            LwSciSyncAccessPerm_WaitOnly,
            ipcWrapperGetEndpoint(ipcWrapper), &syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }

    if (largs->rmTestSetup != NULL) {
        err = largs->rmTestSetup(&resource);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    if (largs->rmTestMapSemaphore != NULL) {
        err = largs->rmTestMapSemaphore(resource, syncObj);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    if (reportProgress) {
        clock_gettime(CLOCK_REALTIME, &beginTimespec);
    }

    /* Waiter streaming phase */
    for (i = 0; i < submitSize; ++i) {
        LwSciSyncFenceIpcExportDescriptor* descriptor = NULL;

        err = largs->waitStreamFrame(
                syncObj,
                waitContext,
                resource,
                descriptor,
                ipcWrapper);
        if (err != LwSciError_Success) {
            goto fail;
        }

        if (reportProgress &&
                (i & progressMask) == progressMask) {
            printTimestampDiff(&beginTimespec, &endTimespec);
            printf("Finished loop %zu\n", i);
        }
    }

    if (reportProgress) {
        printTimestampDiff(&beginTimespec, &endTimespec);
        printf("The entire fence waiting loop finished\n");
    }

    err = ipcSend(ipcWrapper, &destroySyncObj, sizeof(destroySyncObj));

fail:

    if (largs->rmTestTeardown != NULL) {
        largs->rmTestTeardown(resource);
    }

    free(objAndListDesc);

    LwSciSyncAttrListFree(waiterAttrList);
    LwSciSyncAttrListFreeDesc(waiterListDesc);

    LwSciSyncObjFree(syncObj);

    LwSciSyncCpuWaitContextFree(waitContext);
    /* Deinitialize the LwSciSync module */
    LwSciSyncModuleClose(module);

    /* Deinitialize LwSciIpc */
    ipcDeinit(ipcWrapper);
    LwSciIpcDeinit();

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        return LwSciSyncTestStatus::Failure;
    }

    return LwSciSyncTestStatus::Success;
}

static void
inter_process(struct LocalThreadArgs* ltargs)
{
    size_t i;
    LwSciSyncTestStatus waiterStatus, signalerStatus;
    ThreadArgs t_args;

    if (fork() == 0) {
        if (waiter(ltargs) == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        }
        else {
            exit(EXIT_FAILURE);
        }
    } else {
        int status = 0;
        signalerStatus = signaler(ltargs);
        wait(&status);
        if (!WIFEXITED(status)) {
            printf("waiter did not exit\n");
        }
        waiterStatus = WEXITSTATUS(status) == EXIT_SUCCESS ?
            LwSciSyncTestStatus::Success : LwSciSyncTestStatus::Failure;
    }

    ASSERT_EQ(waiterStatus, LwSciSyncTestStatus::Success);
    ASSERT_EQ(signalerStatus, LwSciSyncTestStatus::Success);
}

/** @jama{7198777} Inter-process CPU signaler and CPU waiter - V1
 * -cpu_signaler_cpu_waiter_inter_process
 *   This test represents the use case of CPU Signaler and CPU waiter.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling and shm for data transfers between signaler and
 *   waiter processes, but in real application it should be done
 *   by their own communication mechanism.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterProcessCpuSignalerCpuWaiter, 7198777)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterAttrList;

    ltargs.signalStreamFrame = interProcessCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessCpuWaitStreamFrame;

    inter_process(&ltargs);
}

/** @jama{7198790} Intra-process UMD signaler and UMD waiter - V1
 * -umd_signaler_umd_waiter_inter_process
 *   This test represents the use case of UMD Signaler and UMD waiter.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling and shm for data transfers between signaler and
 *   waiter processes, but in real application it should be done
 *   by their own communication mechanism.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterProcessUmdSignalerUmdWaiter, 7198790)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillUmdWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interProcessUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessUmdWaitStreamFrame;

    inter_process(&ltargs);
}

/** @jama{9405353} Inter-process UMD signaler and CPU waiter - V1
 * -umd_signaler_cpu_waiter_inter_process
 *   This test represents the use case of UMD Signaler and CPU waiter.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterProcessUmdSignalerCpuWaiter, 9405353)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interProcessUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessCpuWaitStreamFrame;

    inter_process(&ltargs);
}

/** @jama{9405350} Inter-process CPU signaler and UMD waiter - V1
 * -cpu_signaler_umd_waiter_inter_process
 *  This test represents the use case of CPU Signaler and UMD waiter.
 *  It illustrates inter-process use case, where 2 processes
 *  are spawned and use LwSciSync public APIs for synchronization.
 *  causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterProcessCpuSignalerUmdWaiter, 9405350)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillUmdWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interProcessCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessUmdWaitStreamFrame;

    inter_process(&ltargs);
}

/** @jama{9829289} Inter-process CPU signaler and CPU waiter using sysmem semaphore - V1
 * -cpu_signaler_cpu_waiter_inter_process_sysmemsema
 *   This test represents the use case of CPU Signaler and CPU waiter
 *   with sysmem semaphore as backend primitive.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling and shm for data transfers between signaler and
 *   waiter processes, but in real application it should be done
 *   by their own communication mechanism.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestSysmemSema, InterProcessCpuSignalerCpuWaiterSysmemSema, 9829289)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerSysmemSemaAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterSysmemSemaAttrList;

    ltargs.signalStreamFrame = interProcessCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessCpuWaitStreamFrame;

    inter_process(&ltargs);
}

/* TODO: Add JAMA id  */
/** @jama{} Inter-process CPU signaler and CPU waiter using sysmem semaphore
 *   with 64b payload - V1
 * -cpu_signaler_cpu_waiter_inter_process_sysmemsema
 *   This test represents the use case of CPU Signaler and CPU waiter
 *   with sysmem semaphore with 64b payload as backend primitive.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling and shm for data transfers between signaler and
 *   waiter processes, but in real application it should be done
 *   by their own communication mechanism.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestSysmemSema, InterProcessCpuSignalerCpuWaiterSysmemSemaPayload64b, 0)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerSysmemSemaPayload64bAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterSysmemSemaPayload64bAttrList;

    ltargs.signalStreamFrame = interProcessCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessCpuWaitStreamFrame;

    inter_process(&ltargs);
}

#ifdef LWSCISYNC_EMU_SUPPORT
/** @jama{TODO} Intra-process UMD signaler and UMD waiter using external primitive - V1
 * -umd_signaler_umd_waiter_inter_process_external_primitive
 *   This test represents the use case of UMD Signaler and UMD waiter where
 *   signaler uses an externally allocated primitive.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling and shm for data transfers between signaler and
 *   waiter processes, but in real application it should be done
 *   by their own communication mechanism.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterProcessUmdSignalerUmdWaiterExternalPrimitive, 0)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillExternalPrimitiveInfo = LwSciSyncTest_FillUmdExternalPrimitiveInfo;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillUmdWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interProcessUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessUmdWaitStreamFrame;

    inter_process(&ltargs);
}

/** @jama{TODO} Inter-process UMD signaler and CPU waiter using external primitive - V1
 * -umd_signaler_cpu_waiter_inter_process_external_primitive
 *   This test represents the use case of UMD Signaler and CPU waiter where
 *   signaler uses an externally allocated primitive.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterProcessUmdSignalerCpuWaiterExternalPrimitive, 1)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillExternalPrimitiveInfo = LwSciSyncTest_FillUmdExternalPrimitiveInfo;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interProcessUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessCpuWaitStreamFrame;

    inter_process(&ltargs);
}
#endif

/** @jama{9938069} Inter-process CPU signaler and CPU waiter with timestamps
 * -cpu_signaler_cpu_waiter_inter_process_with_timestamps
 *   This test represents the use case of CPU Signaler and CPU waiter.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization.
 *   The syncObj signals leave timestamps that can be read from expired fences.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestTimestampSupport, InterProcessCpuSignalerCpuWaiterTimestamps, 9938069)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillTimestampsSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillTimestampsWaiterAttrList;

    ltargs.signalStreamFrame = interProcessCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessCpuWaitWithTimestampsStreamFrame;

    inter_process(&ltargs);
}

/** @jama{13493795} Inter-process CPU signaler and CPU waiter with timestamps and implicit SignalerTimestampInfo
 * -cpu_signaler_implicit_cpu_waiter_inter_process_with_timestamps
 *   This test represents the use case of CPU Signaler and CPU waiter with
 *   implicit SignalerTimestampInfo.
 *   It illustrates inter-process use case, where 2 processes
 *   are spawned and use LwSciSync public APIs for synchronization, and
 *   LwSciSync provides the SignalerTimestampInfo.
 *   The syncObj signals leave timestamps that can be read from expired fences.
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestTimestampSupport, InterProcessCpuSignalerImplicitCpuWaiterTimestamps, 13493795)
{
    struct LocalThreadArgs ltargs = {0};
    ThreadArgs t_args;

    LwSciSyncAttrValTimestampInfo tinfo = {
        .format = LwSciSyncTimestampFormat_8Byte,
        .scaling = {
            .scalingFactorNumerator = 1U,
            .scalingFactorDenominator = 1U,
            .sourceOffset = 0U,
        },
    };
    // ltargs.expectedTimestampInfo = nullptr;
    ltargs.expectedTimestampInfoMulti = &tinfo;

    t_args.submitSize = info->submitSize;
    t_args.reportProgress = info->reportProgress;
    ltargs.targs = t_args;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillTimestampsSignalerImplicitAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillTimestampsWaiterAttrList;

    ltargs.signalStreamFrame = interProcessCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interProcessCpuWaitWithTimestampsStreamFrame;

    inter_process(&ltargs);
}

LwSciError cpuPreinitSignalStream(
    struct ThreadConf* conf,
    struct StreamResources* resources)
{
    size_t submitSize = 1024;
    LwSciSyncFence* fences = (LwSciSyncFence*)malloc(submitSize * sizeof(LwSciSyncFence));
    size_t j;
    size_t i;
    LwSciError err = LwSciError_Success;

    /* create all the fences first */
    for (j = 0; j < submitSize; ++j) {
        LwSciSyncObj syncObj = resources->syncObj;
        fences[j] = LwSciSyncFenceInitializer;
        err = LwSciSyncObjGenerateFence(syncObj, &fences[j]);
        if (err != LwSciError_Success) {
            return err;
        }
    }

    /* pass the fences upstream */
    for (i = 0; i < resources->upstreamSize; ++i) {
        IpcWrapperOld ipcWrapper = resources->upstreamIpcs[i];
        for(j = 0; j < submitSize; ++j) {
            LwSciSyncFenceIpcExportDescriptor fenceDesc = {0};
            err = LwSciSyncIpcExportFence(&fences[i],
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
    }

    /* clear no longer needed fences */
    for (j = 0; j < submitSize; ++j) {
        LwSciSyncFenceClear(&fences[j]);
    }
    free(fences);

    /* signal */
    for (j = 0; j < submitSize; ++j) {
        err = LwSciSyncObjSignal(resources->syncObj);
        if (err != LwSciError_Success) {
            return err;
        }
    }

    return err;
}

LwSciError cpuPreinitWaitStream(
    struct ThreadConf* conf,
    struct StreamResources* resources)
{
    size_t submitSize = 1024;
    size_t j;
    size_t i;
    LwSciError err = LwSciError_Success;
    LwSciSyncFence* fences = (LwSciSyncFence*)malloc(submitSize * sizeof(LwSciSyncFence));

    /* import all the fences first */
    for (j = 0; j < submitSize; ++j) {
        LwSciSyncFenceIpcExportDescriptor fenceDesc = {0};
        fences[j] = LwSciSyncFenceInitializer;
        err = ipcRecvFill(resources->downstreamIpc,
                &fenceDesc, sizeof(fenceDesc));
        if (err != LwSciError_Success) {
            return err;
        }

        err = LwSciSyncIpcImportFence(resources->syncObj,
                &fenceDesc, &fences[j]);
        if (err != LwSciError_Success) {
            return err;
        }
    }

    /* pass the fences upstream */
    for (i = 0; i < resources->upstreamSize; ++i) {
        IpcWrapperOld ipcWrapper = resources->upstreamIpcs[i];
        for(j = 0; j < submitSize; ++j) {
            LwSciSyncFenceIpcExportDescriptor fenceDesc = {0};
            err = LwSciSyncIpcExportFence(&fences[i],
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
    }

    /* perform all waits */
    for (j = 0; j < submitSize; ++j) {
        err = LwSciSyncFenceWait(&fences[i],
                resources->waitContext, -1);
        if (err != LwSciError_Success) {
            return err;
        }
    }

    /* clear fences */
    for (j = 0; j < submitSize; ++j) {
        LwSciSyncFenceClear(&fences[j]);
    }
    free(fences);
    return err;
}

/** @jama{10182422} Inter-process CPU signaler and CPU waiter with preinitialized fences - V1
 * -preinitialized_fences
 *   This test is similar to cpu_signaler_cpu_waiter
 *   but instead of preparing a new fence
 *   and immediately signaling it, the test prepares
 *   all the fences first and then signals them
 *   causes an expected error: Empty input unreconciled list array
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterProcessPreinitializedFences, 10182422)
{
    LwSciSyncTestStatus status = LwSciSyncTestStatus::Success;
    struct ThreadConf conf = {0};
    pid_t peers[2] = {0};

    conf.info = info;

    if ((peers[0] = fork()) == 0) {
        const char* upstream[] = {
            "lwscisync_a_0",
        };
        conf.downstream = NULL;
        conf.upstream = upstream;
        conf.upstreamSize = sizeof(upstream) / sizeof(char*);
        conf.fillAttrList = LwSciSyncTest_FillCpuSignalerAttrList;
        conf.stream = cpuPreinitSignalStream;
        conf.objExportPerm = LwSciSyncAccessPerm_WaitOnly;

        status = standardSignaler(&conf);
        if (status == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        } else {
            exit(EXIT_FAILURE);
        }
    } else if ((peers[1] = fork()) == 0) {
        const char* downstream = "lwscisync_a_1";
        conf.downstream = downstream;
        conf.upstream = NULL;
        conf.upstreamSize = 0U;
        conf.fillAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
        conf.stream = cpuPreinitWaitStream;
        conf.objImportPerm = LwSciSyncAccessPerm_WaitOnly;

        status = standardWaiter(&conf);
        if (status == LwSciSyncTestStatus::Success) {
            exit(EXIT_SUCCESS);
        } else {
            exit(EXIT_FAILURE);
        }
    } else {
        int procExit = 0;

        for (auto const& peerPid : peers) {
            waitpid(peerPid, &procExit, 0);
            if (!WIFEXITED(procExit)) {
                printf("a peer did not exit\n");
                status = LwSciSyncTestStatus::Failure;
            }
            status = WEXITSTATUS(procExit) == EXIT_SUCCESS ?
                    status : LwSciSyncTestStatus::Failure;
        }
    }
    ASSERT_EQ(status, LwSciSyncTestStatus::Success);
}
