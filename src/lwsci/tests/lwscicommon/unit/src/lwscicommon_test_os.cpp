/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <lwscicommon_os.h>
#ifdef LW_QNX
#include <sys/neutrino.h>
#include <sys/syspage.h>
#endif

//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include "gtest/gtest.h"

TEST(TestPlatformUtilities, TestMutexSingleThread) {
    LwSciCommonMutex mutex;
    /* These functions will cause assert errors
       upon errors.
    */
    LwSciCommonMutexCreate(&mutex);
    LwSciCommonMutexLock(&mutex);
    LwSciCommonMutexUnlock(&mutex);
    LwSciCommonMutexDestroy(&mutex);
}

static inline uint64_t GetTimeNsNow()
{
    uint64_t timens = 0;
    struct timespec ts;

#ifdef LW_QNX
    uint64_t freq = 0;
    asm volatile("mrs %0, CNTFRQ_EL0" : "=r"(freq) :: );
    asm volatile("mrs %0, CNTVCT_EL0" : "=r"(timens) :: );
    // colwert CPU ticks to nano seconds
    timens = timens*1000000000/freq;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
    // Colwert time to nano seconds
    timens = ts.tv_sec*1000000000 + ts.tv_nsec;
#endif
    return timens;
}


TEST(TestPlatformUtilities, TestSleepNs) {
    uint64_t sleepTimeNs = 0;
    uint64_t lowerTimeLimitNs = 10000;
    uint64_t upperTimeLimitNs = 100000;
    uint64_t startTime;
    uint64_t delta;

    for (int i = 0; i < 100; i++) {
        sleepTimeNs =  (rand() % (upperTimeLimitNs - lowerTimeLimitNs + 1))
                        + lowerTimeLimitNs;

        startTime = GetTimeNsNow();
        LwSciCommonSleepNs(sleepTimeNs);
        delta = GetTimeNsNow() - startTime;

        ASSERT_GT(delta, sleepTimeNs);
    }
}
