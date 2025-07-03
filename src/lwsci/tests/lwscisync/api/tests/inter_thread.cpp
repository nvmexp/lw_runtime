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
 * waiter threads, but in real application it should be done by their own
 * communication mechanism.
 */

#include <semaphore.h>
#include <pthread.h>
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
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cinttypes>

/* shared fence buffer size */
#define FENCE_BUFFER_SIZE 1024

#define SEMA_PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)

/* This test case makes use of named semaphores for sync between processes */
#define EMPTY_SLOTS_SEMA  "/lwscisync_sema_1"
#define FILLED_SLOTS_SEMA "/lwscisync_sema_2"
#define OBJECT_ALLOC_SEMA         "/lwscisync_sema_3"
#define OBJECT_DESTROY_SEMA         "/lwscisync_sema_4"
#define UMD_ATTR_SEMA     "/lwscisync_sema_5"
#define SHARED_FENCE_SEMA "/lwscisync_sema_6"

struct sharing
{
    LwSciSyncAttrList waiterAttrList;

    /* array of pointers of shared LwSciSyncFence buffer */
    LwSciSyncFence fence[FENCE_BUFFER_SIZE];
    LwSciSyncModule module;
    LwSciSyncObj syncObj;
};

struct semaphores
{
    /* Semaphore for managing shared fence buffer */
    sem_t* empty_slots;
    sem_t* filled_slots;

    /* Semaphore for sharing sync object */
    sem_t* syncObj_alloc_sema;
    sem_t* syncObj_destroy_sema;

    /* Semaphore for sharing waiter's umd attributes */
    sem_t* umd_attr_sema;

    /* Semaphore for sharing fence buffer at once if flagged by user */
    sem_t* share_fence_buf;
};

struct LocalThreadArgs {
    TestInfo* info;
    struct semaphores sems;
    struct sharing* shared;
    LwSciSyncTestStatus waiterStatus;
    LwSciSyncTestStatus signalerStatus;

    LwSciError(* fillSignalerAttrList)(LwSciSyncAttrList list);
    LwSciError(* fillWaiterAttrList)(LwSciSyncAttrList list);
#ifdef LWSCISYNC_EMU_SUPPORT
    LwSciError(* fillExternalPrimitiveInfo)(LwSciSyncAttrList list, TestResources res);
#endif
    LwSciError(* rmTestSetup)(TestResources* res);
    LwSciError(* rmTestMapSemaphore)(TestResources res, LwSciSyncObj syncObj);
    void(* rmTestTeardown)(TestResources res);
    LwSciError(* signalStreamFrame)(
        sem_t* prefence,
        sem_t* postfence,
        LwSciSyncObj syncObj,
        TestResources resource,
        LwSciSyncFence* shared_fence);
    LwSciError(* waitStreamFrame)(
        sem_t* prefence,
        sem_t* postfence,
        LwSciSyncCpuWaitContext waitContext,
        TestResources resource,
        LwSciSyncFence* shared_fence);
    LwSciSyncAttrValTimestampInfo* expectedTimestampInfo;
    LwSciSyncAttrValTimestampInfo* expectedTimestampInfoMulti;
};

LwSciError interThreadCpuSignalStreamFrame(
    sem_t* prefence,
    sem_t* postfence,
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFence* shared_fence)
{
    LwSciError err;
    LwSciSyncFence localFence = LwSciSyncFenceInitializer;

    sem_wait(prefence);
    err = LwSciSyncObjGenerateFence(syncObj, &localFence);
    if (err != LwSciError_Success) {
        return err;
    }

    /* duplicate fence before sharing */
    LwSciSyncFenceDup(&localFence, shared_fence);
    /* local copy no longer necessary, so dispose of it */
    LwSciSyncFenceClear(&localFence);
    sem_post(postfence);

    return LwSciSyncObjSignal(syncObj);
}

LwSciError interThreadUmdSignalStreamFrame(
    sem_t* prefence,
    sem_t* postfence,
    LwSciSyncObj syncObj,
    TestResources resource,
    LwSciSyncFence* shared_fence)
{
    LwSciError err;
    LwSciSyncFence localFence = LwSciSyncFenceInitializer;

    sem_wait(prefence);

    err = umdGetPostLwSciSyncFence(syncObj, resource, &localFence);
    if (err != LwSciError_Success) {
        return err;
    }

    /* duplicate fence before sharing */
    LwSciSyncFenceDup(&localFence, shared_fence);
    /* local copy no longer necessary, so dispose of it */
    LwSciSyncFenceClear(&localFence);

    sem_post(postfence);
    return LwSciError_Success;
}

LwSciError interThreadCpuWaitStreamFrame(
    sem_t* prefence,
    sem_t* postfence,
    LwSciSyncCpuWaitContext waitContext,
    TestResources resource,
    LwSciSyncFence* shared_fence)
{
    LwSciError err;

    sem_wait(prefence);
    /* Perform Wait on LwSciSyncFence */
    err = LwSciSyncFenceWait(shared_fence,
            waitContext, -1);
    if (err != LwSciError_Success) {
        return err;
    }

    LwSciSyncFenceClear(shared_fence);
    sem_post(postfence);

    return LwSciError_Success;
}

static LwSciError InterThreadTestAssertSignalerTimestampInfo(
    LwSciSyncAttrList reconciledList,
    LwSciSyncAttrValTimestampInfo* timestampInfo,
    LwSciSyncAttrValTimestampInfo* timestampInfoMulti)
{
    LwSciSyncAttrValTimestampInfo actualSignalerTimestampInfo;
    LwSciSyncAttrValTimestampInfo expectedTimestampInfo;
    const void* signalerTimestampInfo = NULL;
    size_t signalerTimestampInfoLen = 0U;

    LwSciError err = LwSciError_Success;

    if ((timestampInfo == nullptr) && (timestampInfoMulti == nullptr)) {
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

LwSciError interThreadCpuWaitWithTimestampsStreamFrame(
    sem_t* prefence,
    sem_t* postfence,
    LwSciSyncCpuWaitContext waitContext,
    TestResources resource,
    LwSciSyncFence* shared_fence)
{
    LwSciError err;
    uint64_t timestamp = 0U;

    sem_wait(prefence);
    /* Perform Wait on LwSciSyncFence */
    err = LwSciSyncFenceWait(shared_fence,
            waitContext, -1);
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncFenceGetTimestamp(shared_fence, &timestamp);
    if (err != LwSciError_Success) {
        return err;
    }

    LwSciSyncFenceClear(shared_fence);
    sem_post(postfence);

    return LwSciError_Success;
}

LwSciError interThreadCpuWaitWithTimestampsStreamFrameWithFencesInited(
    sem_t* prefence,
    sem_t* postfence,
    LwSciSyncCpuWaitContext waitContext,
    TestResources resource,
    LwSciSyncFence* shared_fence)
{
    LwSciError err;
    uint64_t timestamp = 0U;

    /* Perform Wait on LwSciSyncFence */
    err = LwSciSyncFenceWait(shared_fence,
            waitContext, -1);
    if (err != LwSciError_Success) {
        return err;
    }

    err = LwSciSyncFenceGetTimestamp(shared_fence, &timestamp);
    if (err != LwSciError_Success) {
        return err;
    }

    LwSciSyncFenceClear(shared_fence);
    sem_post(postfence);

    return LwSciError_Success;
}

LwSciError interThreadUmdWaitStreamFrame(
    sem_t* prefence,
    sem_t* postfence,
    LwSciSyncCpuWaitContext waitContext,
    TestResources resource,
    LwSciSyncFence* shared_fence)
{
    LwSciError err = LwSciError_Success;

    sem_wait(prefence);

    err = umdWaitOnPreLwSciSyncFence(resource, shared_fence);
    if (err != LwSciError_Success) {
        return err;
    }

    LwSciSyncFenceClear(shared_fence);
    sem_post(postfence);

    return LwSciError_Success;
}

static void* signaler(void* args)
{
    size_t i;
    LwSciError err;
    struct LocalThreadArgs* largs = (struct LocalThreadArgs*) args;
    TestResources resource = NULL;
    uint32_t submitSize = largs->info->submitSize;
    LwSciSyncAttrList unreconciledList[2] = {NULL};
    LwSciSyncAttrList reconciledList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncAttrList signalerAttrList = NULL;
    LwSciSyncModule module = NULL;
    LwSciSyncObj syncObj;
    struct semaphores* sems = &largs->sems;
    void* tempMapping;
    struct sharing* shared;
    const void* signalerTimestampInfo = NULL;
    size_t signalerTimestampInfoLen = 0U;

    shared = largs->shared;

    /* Reserve memory for shared LwSciSyncFence buffer */
    for (i = 0; i < FENCE_BUFFER_SIZE; ++i) {
        memset(&shared->fence[i], 0, sizeof(LwSciSyncFence));
    }

    sem_wait(sems->umd_attr_sema);
    module = largs->shared->module;

    /* Create signaler's AttrList */
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

    /* Wait for waiter's LwSciSyncAttrList */
    unreconciledList[0] = signalerAttrList;
    unreconciledList[1] = shared->waiterAttrList;

    /* Reconcile Signaler and Waiter LwSciSyncAttrList */
    err = LwSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList,
            &newConflictList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Assert on the LwSciSyncInternalAttrKey_SignalerTimestampInfo and
     * LwSciSyncInternalAttrKey_SignalerTimestampInfoMulti keys */
    err = InterThreadTestAssertSignalerTimestampInfo(
        reconciledList, largs->expectedTimestampInfo,
        largs->expectedTimestampInfoMulti);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Create LwSciSync object and get the syncObj */
    err = LwSciSyncObjAlloc(reconciledList, &largs->shared->syncObj);
    if (err != LwSciError_Success) {
        goto fail;
    }
    syncObj = largs->shared->syncObj;

    /* Signal the waiter after creation of sync object */
    sem_post(sems->syncObj_alloc_sema);

    if (largs->rmTestMapSemaphore != NULL) {
        err = largs->rmTestMapSemaphore(resource, syncObj);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    /* Signaler streaming phase */
    for (i = 0; i < submitSize; ++i) {
        uint32_t putIndex = i % FENCE_BUFFER_SIZE;
        err = largs->signalStreamFrame(sems->empty_slots,
                                       sems->filled_slots,
                                       syncObj,
                                       resource,
                                       &shared->fence[putIndex]);
        if (err != LwSciError_Success) {
            goto fail;
        }
    }

    /* Wait till its safe to destroy LwSciSyncObj */
    sem_wait(sems->syncObj_destroy_sema);

    /* Free LwSciSyncObj */
    LwSciSyncObjFree(largs->shared->syncObj);

fail:
    if (largs->rmTestTeardown != NULL) {
        largs->rmTestTeardown(resource);
    }

    /* Free Attribute list objects */
    if (reconciledList != NULL) {
        LwSciSyncAttrListFree(reconciledList);
    }
    if (newConflictList != NULL) {
        LwSciSyncAttrListFree(newConflictList);
    }
    if (signalerAttrList != NULL) {
        LwSciSyncAttrListFree(signalerAttrList);
    }
    if (shared->waiterAttrList != NULL) {
        LwSciSyncAttrListFree(shared->waiterAttrList);
    }

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        largs->signalerStatus = LwSciSyncTestStatus::Failure;
    } else {
        largs->signalerStatus = LwSciSyncTestStatus::Success;
    }
    return NULL;
}

static void* waiter(void* args)
{
    size_t i;
    LwSciError err;
    struct LocalThreadArgs* largs = (struct LocalThreadArgs*) args;
    TestResources resource = NULL;
    bool reportProgress = largs->info->reportProgress;
    uint32_t submitSize = largs->info->submitSize;
    LwSciSyncModule module = NULL;
    LwSciSyncObj syncObj;
    struct semaphores* sems = &largs->sems;
    void* tempMapping;
    struct sharing* shared;
    LwSciSyncCpuWaitContext waitContext = NULL;
    struct timespec beginTimespec;
    struct timespec endTimespec;
    const uint32_t progressMask = 0xfffff;

    shared = largs->shared;

    /* Waiter Setup/Init phase */
    /* Initialize the LwSciSync module */
    err = LwSciSyncModuleOpen(&largs->shared->module);
    if (err != LwSciError_Success) {
        goto fail;
    }
    module = largs->shared->module;

    err = LwSciSyncCpuWaitContextAlloc(module, &waitContext);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Create waiter's AttrList */
    err = LwSciSyncAttrListCreate(module, &shared->waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }
    err = largs->fillWaiterAttrList(shared->waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    /* Send waiter's attr to signaler */
    sem_post(sems->umd_attr_sema);

    /* Wait for signaler to create LwSciSyncObj */
    sem_wait(sems->syncObj_alloc_sema);
    syncObj = largs->shared->syncObj;

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
        uint32_t getIndex = i % FENCE_BUFFER_SIZE;
        err = largs->waitStreamFrame(sems->filled_slots,
                                     sems->empty_slots,
                                     waitContext,
                                     resource,
                                     &shared->fence[getIndex]);
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

    /* Inform signaler that it's safe to destroy LwSciSyncObj */
    sem_post(sems->syncObj_destroy_sema);


    LwSciSyncCpuWaitContextFree(waitContext);
    /* Deinitialize the LwSciSync module */
    LwSciSyncModuleClose(largs->shared->module);

fail:
    if (largs->rmTestTeardown != NULL) {
        largs->rmTestTeardown(resource);
    }

    if (err != LwSciError_Success) {
        printf("err = %d\n", err);
        largs->waiterStatus = LwSciSyncTestStatus::Failure;
    } else {
        largs->waiterStatus = LwSciSyncTestStatus::Success;
    }
    return NULL;
}

static void
inter_thread(struct LocalThreadArgs* ltargs)
{
    size_t i;
    void* status;
    LwSciSyncTestStatus waiterStatus, signalerStatus;
    pthread_t signaler_tid;
    pthread_t waiter_tid;
    struct semaphores* sems = &ltargs->sems;
    int sharedMemFd;

    sem_unlink(EMPTY_SLOTS_SEMA);
    sem_unlink(FILLED_SLOTS_SEMA);
    sem_unlink(OBJECT_ALLOC_SEMA);
    sem_unlink(OBJECT_DESTROY_SEMA);
    sem_unlink(UMD_ATTR_SEMA);
    sem_unlink(SHARED_FENCE_SEMA);

    shm_unlink("test_lwscisync_api_shared");

    sharedMemFd = shm_open("test_lwscisync_api_shared", O_CREAT | O_RDWR,
                           0777);
    ftruncate(sharedMemFd, sizeof(struct sharing));
    ltargs->shared =
        (struct sharing*) mmap(NULL, sizeof(struct sharing),
                               PROT_READ | PROT_WRITE, MAP_SHARED, sharedMemFd, 0);
    close(sharedMemFd);

    /* initialize semaphores */
    sems->empty_slots = sem_open(EMPTY_SLOTS_SEMA,
        O_CREAT, SEMA_PERMS, FENCE_BUFFER_SIZE);
    sems->filled_slots = sem_open(FILLED_SLOTS_SEMA,
        O_CREAT, SEMA_PERMS, 0);
    sems->syncObj_alloc_sema = sem_open(OBJECT_ALLOC_SEMA,
        O_CREAT, SEMA_PERMS, 0);
    sems->syncObj_destroy_sema =
        sem_open(OBJECT_DESTROY_SEMA,
                 O_CREAT, SEMA_PERMS, 0);
    sems->umd_attr_sema = sem_open(UMD_ATTR_SEMA,
        O_CREAT, SEMA_PERMS, 0);
    sems->share_fence_buf = sem_open(SHARED_FENCE_SEMA,
        O_CREAT, SEMA_PERMS, 0);

    (void)pthread_create(&signaler_tid, NULL, signaler, ltargs);
    (void)pthread_create(&waiter_tid, NULL, waiter, ltargs);
    (void)pthread_join(signaler_tid, NULL);
    (void)pthread_join(waiter_tid, NULL);

    /* destroy semaphores */
    sem_close(sems->empty_slots);
    sem_close(sems->filled_slots);
    sem_close(sems->syncObj_alloc_sema);
    sem_close(sems->syncObj_destroy_sema);
    sem_close(sems->umd_attr_sema);
    sem_close(sems->share_fence_buf);

    sem_unlink(EMPTY_SLOTS_SEMA);
    sem_unlink(FILLED_SLOTS_SEMA);
    sem_unlink(OBJECT_ALLOC_SEMA);
    sem_unlink(OBJECT_DESTROY_SEMA);
    sem_unlink(UMD_ATTR_SEMA);
    sem_unlink(SHARED_FENCE_SEMA);

    munmap(ltargs->shared, sizeof(struct sharing));

    shm_unlink("test_lwscisync_api_shared");

    ASSERT_EQ(ltargs->waiterStatus, LwSciSyncTestStatus::Success);
    ASSERT_EQ(ltargs->signalerStatus, LwSciSyncTestStatus::Success);
}

/** @jama{7097243} Intra-process CPU signaler and CPU waiter - V1
 * -cpu_signaler_cpu_waiter_inter_thread
 *   This test represents the use case of CPU Signaler and CPU waiter.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling for data transfers between signaler and
 *   waiter threads, but in real application it should be done by their own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterThreadCpuSignalerCpuWaiter, 7097243)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterAttrList;

    ltargs.signalStreamFrame = interThreadCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadCpuWaitStreamFrame;

    inter_thread(&ltargs);
}

/** @jama{7112771} Intra-process UMD signaler and UMD waiter - V1
 * -umd_signaler_umd_waiter_inter_thread
 *   This test represents the use case of UMD Signaler and UMD waiter.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs along with UMD APIs for
 *   synchronization. The shared memory illustrated here should be managed by
 *   LwSciBuf APIs. The test uses semaphore signalling for data transfers between
 *   signaler and waiter threads, but in real application it should be done by their
 *   own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterThreadUmdSignalerUmdWaiter, 7112771)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillUmdWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interThreadUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadUmdWaitStreamFrame;

    inter_thread(&ltargs);
}

/** @jama{9405347} Intra-process UMD signaler and CPU waiter - V1
 * -umd_signaler_cpu_waiter_inter_thread
 *   This test represents the use case of UMD Signaler and CPU waiter.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs along with UMD APIs for
 *   synchronization. The shared memory illustrated here should be managed by
 *   LwSciBuf APIs. The test uses semaphore signalling for data transfers between
 *   signaler and waiter threads, but in real application it should be done by their
 *   own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterThreadUmdSignalerCpuWaiter, 9405347)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interThreadUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadCpuWaitStreamFrame;

    inter_thread(&ltargs);
}

/** @jama{9405341} Intra-process CPU signaler and UMD waiter - V1
 * -cpu_signaler_umd_waiter_inter_thread
 *   This test represents the use case of CPU Signaler and UMD waiter.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs along with UMD APIs for
 *   synchronization. The shared memory illustrated here should be managed by
 *   LwSciBuf APIs. The test uses semaphore signalling for data transfers between
 *   signaler and waiter threads, but in real application it should be done by their
 *   own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterThreadCpuSignalerUmdWaiter, 9405341)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillUmdWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interThreadCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadUmdWaitStreamFrame;

    inter_thread(&ltargs);
}

/** @jama{9938042} Intra-process CPU signaler and CPU waiter with timestamps
 * -cpu_signaler_cpu_waiter_inter_thread_with_timestamps
 *   This test represents the use case of CPU Signaler and CPU waiter
 *   with writing and reading timestamps.
 */
LWSCISYNC_DECLARE_TEST(TestTimestampSupport, InterThreadCpuSignalerCpuWaiterWithTimestamps, 9938042)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillTimestampsSignalerAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillTimestampsWaiterAttrList;

    ltargs.signalStreamFrame = interThreadCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadCpuWaitWithTimestampsStreamFrame;

    inter_thread(&ltargs);
}

/** @jama{13493801} Inter-thread CPU signaler and CPU waiter with timestamps and implicit SignalerTimestampInfo
 * -cpu_signaler_implicit_cpu_waiter_inter_thread_with_timestamps
 *   This test represents the use case of CPU Signaler and CPU waiter
 *   with writing and reading timestamps, when LwSciSync fills the
 *   SignalerTimestampInfo key.
 */
LWSCISYNC_DECLARE_TEST(TestTimestampSupport, InterThreadCpuSignalerImplicitCpuWaiterWithTimestamps, 13493801)
{
    struct LocalThreadArgs ltargs = {0};
    LwSciSyncAttrValTimestampInfo tinfo = {
        .format = LwSciSyncTimestampFormat_8Byte,
        .scaling = {
            .scalingFactorNumerator = 1U,
            .scalingFactorDenominator = 1U,
            .sourceOffset = 0U,
        },
    };
    ltargs.expectedTimestampInfoMulti = &tinfo;

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillTimestampsSignalerImplicitAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillTimestampsWaiterAttrList;

    ltargs.signalStreamFrame = interThreadCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadCpuWaitWithTimestampsStreamFrame;

    inter_thread(&ltargs);
}

/** @jama{9829283} Intra-process CPU signaler and CPU waiter using sysmem semaphore - V1
 * -cpu_signaler_cpu_waiter_inter_thread
 *   This test represents the use case of CPU Signaler and CPU waiter
 *   with sysmem semaphore as backend primitive.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling for data transfers between signaler and
 *   waiter threads, but in real application it should be done by their own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestSysmemSema, InterThreadCpuSignalerCpuWaiterSysmemSema, 9829283)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerSysmemSemaAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterSysmemSemaAttrList;

    ltargs.signalStreamFrame = interThreadCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadCpuWaitStreamFrame;

    inter_thread(&ltargs);
}

/* TODO: Add JAMA id  */
/** @jama{} Intra-process CPU signaler and CPU waiter using sysmem semaphore
 *   with 64b payload - V1
 * -cpu_signaler_cpu_waiter_inter_thread
 *   This test represents the use case of CPU Signaler and CPU waiter
 *   with sysmem semaphore with 64b payload as backend primitive.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs for synchronization.
 *   The shared memory illustrated here should be managed by LwSciBuf APIs.
 *   The test uses semaphore signalling for data transfers between signaler and
 *   waiter threads, but in real application it should be done by their own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestSysmemSema, InterThreadCpuSignalerCpuWaiterSysmemSemaPayload64b, 0)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillCpuSignalerSysmemSemaPayload64bAttrList;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterSysmemSemaPayload64bAttrList;

    ltargs.signalStreamFrame = interThreadCpuSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadCpuWaitStreamFrame;

    inter_thread(&ltargs);
}

#ifdef LWSCISYNC_EMU_SUPPORT
/** @jama{TODO} Intra-process UMD signaler and UMD waiter using external primitive - V1
 * -umd_signaler_umd_waiter_inter_thread_external_primitive
 *   This test represents the use case of UMD Signaler and UMD waiter where
 *   signaler uses an externally allocated primitive.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs along with UMD APIs for
 *   synchronization. The shared memory illustrated here should be managed by
 *   LwSciBuf APIs. The test uses semaphore signalling for data transfers between
 *   signaler and waiter threads, but in real application it should be done by their
 *   own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterThreadUmdSignalerUmdWaiterExternalPrimitive, 0)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillExternalPrimitiveInfo = LwSciSyncTest_FillUmdExternalPrimitiveInfo;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillUmdWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interThreadUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadUmdWaitStreamFrame;

    inter_thread(&ltargs);
}

/** @jama{TODO} Intra-process UMD signaler and CPU waiter using external primitive - V1
 * -umd_signaler_cpu_waiter_inter_thread_external_primitive
 *   This test represents the use case of UMD Signaler and CPU waiter where
 *   signaler uses an externally allocated primitive.
 *   It illustrates inter-thread intra-process use case, where within same process
 *   two threads are spawned and use LwSciSync public APIs along with UMD APIs for
 *   synchronization. The shared memory illustrated here should be managed by
 *   LwSciBuf APIs. The test uses semaphore signalling for data transfers between
 *   signaler and waiter threads, but in real application it should be done by their
 *   own communication mechanism.
 */
LWSCISYNC_DECLARE_TEST(TestBaseSupport, InterThreadUmdSignalerCpuWaiterExternalPrimitive, 1)
{
    struct LocalThreadArgs ltargs = {0};

    ltargs.info = info;

    ltargs.fillSignalerAttrList = LwSciSyncTest_FillUmdSignalerAttrList;
    ltargs.fillExternalPrimitiveInfo = LwSciSyncTest_FillUmdExternalPrimitiveInfo;
    ltargs.fillWaiterAttrList = LwSciSyncTest_FillCpuWaiterAttrList;
    ltargs.rmTestSetup = LwRmGpu_TestSetup;
    ltargs.rmTestMapSemaphore = LwRmGpu_TestMapSemaphore;
    ltargs.rmTestTeardown = LwRmGpu_TestTeardown;
    ltargs.signalStreamFrame = interThreadUmdSignalStreamFrame;
    ltargs.waitStreamFrame = interThreadCpuWaitStreamFrame;

    inter_thread(&ltargs);
}
#endif
