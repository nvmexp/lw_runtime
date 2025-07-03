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

#ifndef SIGNAL_H
#define SIGNAL_H

#include "common.h"

void signalBenchmark(TestArgs* testArgs)
{
    if (testArgs->pinSignaler) {
        pin_thread_to_core(testArgs->signalerCore);
    }

    LwSciSyncFence fenceWait = LwSciSyncFenceInitializer;

    uint64_t startTime, nextTime;
    uint64_t fenceOpPeriod;
    if (testArgs->rate > 0U) {
        fenceOpPeriod = NSEC_IN_SEC / testArgs->rate;
    }

    /* Used to store timestamps */
    startBuf = (uint64_t*)malloc(sizeof(uint64_t) * testArgs->N);
    endBuf = (uint64_t*)malloc(sizeof(uint64_t) * testArgs->N);

    startTime = get_timestamp();
    for (uint32_t i = 0U; i < testArgs->N; i++) {
        LwSciSyncObjGenerateFence(syncObj, &fenceWait);

        startBuf[i] = get_timestamp();
        LwSciSyncObjSignal(syncObj);
        endBuf[i] = get_timestamp();

        LwSciSyncFenceClear(&fenceWait);

        if (testArgs->rate > 0U) {
            /* Simulate the operation rate */
            nextTime = startTime + fenceOpPeriod * (i + 1U);
            waitForNextOp(nextTime);
        }
    }

    perfAnalyze(testArgs->N, startBuf, endBuf);

    free(startBuf);
    free(endBuf);
}

#endif // !SIGNAL_H
