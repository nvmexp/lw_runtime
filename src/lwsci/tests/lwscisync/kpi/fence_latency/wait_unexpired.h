//! \file
//! \brief LwSciSync fence latency test.
//!
//! \copyright
//! Copyright (c) 2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef WAIT_UNEXPIRED_H
#define WAIT_UNEXPIRED_H

#include "common.h"

void* signalerUnexpired(void* args)
{
    TestArgs* testArgs = (TestArgs*) args;
    if (testArgs->pinSignaler) {
        pin_thread_to_core(testArgs->signalerCore);
    }

#ifdef __QNX__
    struct timespec ns_to_spin = {
        .tv_sec = 0,
        .tv_nsec = WAIT_NS,
    };
#endif

    for (uint32_t i = 0U; i < testArgs->N; i++) {
        checkSem(sem_wait(&waiterSem), "ERROR: signaler semwait failed\n");

        /* Busy wait to let waiter thread reach FenceWait() call first */
#ifdef __QNX__
        nanospin(&ns_to_spin);
#else
        for (uint32_t j = 0U; j < BUSY_WAIT_ITERATIONS; j++);
#endif

        startBuf[i] = get_timestamp();
        LwSciSyncObjSignal(syncObj);

        checkSem(sem_post(&signalerSem), "ERROR: signaler sempost failed\n");
    }

    return (void*) 0;
}

void* waiterUnexpired(void* args)
{
    TestArgs* testArgs = (TestArgs*) args;
    if (testArgs->pinWaiter) {
        pin_thread_to_core(testArgs->waiterCore);
    }

    uint64_t startTime, nextTime;
    uint64_t fenceOpPeriod;
    if (testArgs->rate > 0U) {
        fenceOpPeriod = NSEC_IN_SEC / testArgs->rate;
    }

    LwSciSyncFence fenceWait = LwSciSyncFenceInitializer;

    startTime = get_timestamp();
    for (uint32_t i = 0U; i < testArgs->N; i++) {
        LwSciSyncObjGenerateFence(syncObj, &fenceWait);
        checkSem(sem_post(&waiterSem), "ERROR: waiter sempost failed\n");

        LwSciSyncFenceWait(&fenceWait, waitContext, -1);
        endBuf[i] = get_timestamp();

        checkSem(sem_wait(&signalerSem), "ERROR: waiter semwait failed\n");
        LwSciSyncFenceClear(&fenceWait);

        if (testArgs->rate > 0U) {
            /* Simulate the operation rate */
            nextTime = startTime + fenceOpPeriod * (i + 1U);
            waitForNextOp(nextTime);
        }
    }

    return (void*) 0;
}

void unexpiredBenchmark(TestArgs* testArgs)
{
    pthread_t signalerThread, waiterThread;

    sem_init(&waiterSem, 0, 0);
    sem_init(&signalerSem, 0, 0);

    /* Used to store timestamps */
    startBuf = (uint64_t*)malloc(sizeof(uint64_t) * testArgs->N);
    endBuf = (uint64_t*)malloc(sizeof(uint64_t) * testArgs->N);

    pthread_create(&signalerThread, NULL, signalerUnexpired, testArgs);
    pthread_create(&waiterThread, NULL, waiterUnexpired, testArgs);

    pthread_join(signalerThread, NULL);
    pthread_join(waiterThread, NULL);

    sem_destroy(&waiterSem);
    sem_destroy(&signalerSem);

    fprintf(stderr, "INFO: The latency of signalling a fence is included "
                    "in these (unexpired fence wait) measurements\n");
    fprintf(stderr, "INFO: Subtract the latency to signal from the values "
                    "obtained to obtain the latency taken to wake up "
                    "after a signal\n");

    perfAnalyze(testArgs->N, startBuf, endBuf);

    free(startBuf);
    free(endBuf);
}

#endif // !WAIT_UNEXPIRED_H
