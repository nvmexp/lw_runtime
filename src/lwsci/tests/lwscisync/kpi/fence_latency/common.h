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

#ifndef COMMON_H
#define COMMON_H

#ifdef __QNX__
#else
#define _GNU_SOURCE
#endif

#define NSEC_IN_SEC 1000000000

#include "lwscisync.h"
#include "lwscisync_internal.h"

#include <stdio.h>
#include <semaphore.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>

#include "utils.h"

/* Semaphores to ensure ordering between threads in Wait benchmarks */
sem_t waiterSem;
sem_t signalerSem;

/* Array to store start and end timestamps */
uint64_t* startBuf;
uint64_t* endBuf;

/* Test arguments */
typedef struct {
    /* Type of test */
    char testType;

    /* Type of test */
    char primitiveType;

    /* Number of iterations */
    uint32_t N;

    /* Number of operations per second */
    int rate;

    /* Flag indicating whether to pin threads to specific cores */
    bool pinSignaler;
    bool pinWaiter;

    /* Cores to pin threads to */
    int signalerCore;
    int waiterCore;
} TestArgs;

/* LwSciSync variables */
LwSciSyncModule module = NULL;
LwSciSyncCpuWaitContext waitContext = NULL;
LwSciSyncObj syncObj = NULL;

void help(void)
{
    fprintf(stderr, "usage: ./fence_latency -t <e|u|s> -n <# of iterations> -s <core#> -w <core#>\n");
    fprintf(stderr, "i.e. './fence_latency -t e -n 10000' runs 10000 iterations of waiting on an expired fence\n");
    fprintf(stderr, "-t <s|e|u>: test type\n");
    fprintf(stderr, "   <s>: signal fence\n");
    fprintf(stderr, "   <e>: wait on expired fence\n");
    fprintf(stderr, "   <u>: wait on unexpired fence\n");
    fprintf(stderr, "-p <p|s>: primitive type\n");
    fprintf(stderr, "   <p>: syncpoint, by default\n");
    fprintf(stderr, "   <s>: sysmem semaphore\n");
    fprintf(stderr, "-n <N>: number of iterations\n");
    fprintf(stderr, "-s <N>: pin signaler thread to core N\n");
    fprintf(stderr, "-w <N>: pin waiter thread to core N\n");
    fprintf(stderr, "-r <N>: rate of fence operations per second\n");
}

bool parseArgs(int argc, char** argv, TestArgs* testArgs)
{
    if (argc < 5) {
        fprintf(stderr, "ERROR: Invalid number of arguments.\n");
        return false;
    }

    int opt;
    while ((opt = getopt(argc, argv, "n:t:s:w:r:p:")) != -1) {
        switch (opt) {
        case 'n':
            if (atoi(optarg) >= 0) {
                testArgs->N = strtoul(optarg, NULL, 0);
                testArgs->N = (testArgs->N > 0) ? testArgs->N : 1;
            } else {
                fprintf(stderr, "ERROR: Invalid number of iterations.\n");
                return false;
            }
            break;
        case 't':
            if (*optarg != 's' && *optarg != 'u' && *optarg != 'e') {
                fprintf(stderr, "ERROR: Invalid Test Type.\n");
                return false;
            }
            testArgs->testType = *optarg;
            break;
        case 's':
            testArgs->signalerCore = atoi(optarg);
            testArgs->pinSignaler = true;
            break;
        case 'w':
            testArgs->waiterCore = atoi(optarg);
            testArgs->pinWaiter = true;
            break;
        case 'r':
            testArgs->rate = atoi(optarg);
            if (testArgs->rate < 0) {
                fprintf(stderr, "ERROR: Invalid rate.\n");
                return false;
            }
            break;
        case 'p':
            if (*optarg != 'p' && *optarg != 's') {
                fprintf(stderr, "ERROR: Invalid Primitve Type.\n");
                return false;
            }
            testArgs->primitiveType = *optarg;
            break;
        default:
            return false;
        }
    }

    return true;
}

static inline void checkErr(LwSciError e)
{
    if (e != LwSciError_Success) {
        fprintf(stderr, "%s, %s:%d, LwSci error %0x\n",
                __FILE__, __func__, __LINE__, e);
        return;
    }
}

static inline void checkSem(int sem_call, char const *msg)
{
    if (sem_call != 0) {
        printf(msg);
    }
}

void init(LwSciSyncInternalAttrValPrimitiveType primitiveType)
{
#ifdef __QNX__
    nsPerCycles = NSEC_IN_SEC / SYSPAGE_ENTRY(qtime)->cycles_per_sec;
#endif

    LwSciSyncAttrList signalerList = NULL;
    LwSciSyncAttrList waiterList = NULL;
    LwSciSyncAttrList newConflictList = NULL;
    LwSciSyncAttrList reconciledList = NULL;

    checkErr(LwSciSyncModuleOpen(&module));
    checkErr(LwSciSyncAttrListCreate(module, &signalerList));
    checkErr(LwSciSyncAttrListCreate(module, &waiterList));

    /* Set Signaler Attributes */
    LwSciSyncAttrKeyValuePair publicSignaler[2];

    bool cpuSignaler = true;
    publicSignaler[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    publicSignaler[0].value = (void*) &cpuSignaler;
    publicSignaler[0].len = sizeof(cpuSignaler);

    LwSciSyncAccessPerm accessPermSignaler = LwSciSyncAccessPerm_SignalOnly;
    publicSignaler[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    publicSignaler[1].value = (void*) &accessPermSignaler;
    publicSignaler[1].len = sizeof(accessPermSignaler);

    checkErr(LwSciSyncAttrListSetAttrs(signalerList, publicSignaler, 2));

    LwSciSyncInternalAttrKeyValuePair internalSignaler[2];

    internalSignaler[0].attrKey =
        LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
    internalSignaler[0].value = (void*)&primitiveType;
    internalSignaler[0].len = sizeof(primitiveType);

    uint32_t primitiveCnt = 1U;
    internalSignaler[1].attrKey =
        LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
    internalSignaler[1].value = (void*)&primitiveCnt;
    internalSignaler[1].len = sizeof(primitiveCnt);

    checkErr(LwSciSyncAttrListSetInternalAttrs(signalerList,
                                               internalSignaler, 2));

    /* Set Waiter Attributes */
    LwSciSyncAttrKeyValuePair publicWaiter[2];

    bool cpuWaiter = true;
    publicWaiter[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    publicWaiter[0].value = (void*) &cpuWaiter;
    publicWaiter[0].len = sizeof(cpuWaiter);

    LwSciSyncAccessPerm accessPermWaiter = LwSciSyncAccessPerm_WaitOnly;
    publicWaiter[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    publicWaiter[1].value = (void*) &accessPermWaiter;
    publicWaiter[1].len = sizeof(accessPermWaiter);

    checkErr(LwSciSyncAttrListSetAttrs(waiterList, publicWaiter, 2));

    LwSciSyncInternalAttrKeyValuePair internalWaiter[2];

    internalWaiter[0].attrKey =
        LwSciSyncInternalAttrKey_SignalerPrimitiveInfo;
    internalWaiter[0].value = (void*)&primitiveType;
    internalWaiter[0].len = sizeof(primitiveType);

    internalWaiter[1].attrKey =
        LwSciSyncInternalAttrKey_SignalerPrimitiveCount;
    internalWaiter[1].value = (void*)&primitiveCnt;
    internalWaiter[1].len = sizeof(primitiveCnt);

    checkErr(LwSciSyncAttrListSetInternalAttrs(waiterList,
                                               internalWaiter, 2));

    /* Reconcile attributes and allocate sync object */
    LwSciSyncAttrList unreconciledList[2] = { signalerList,  waiterList };

    checkErr(LwSciSyncAttrListReconcile(unreconciledList, 2U,
                                        &reconciledList, &newConflictList));
    checkErr(LwSciSyncObjAlloc(reconciledList, &syncObj));
    checkErr(LwSciSyncCpuWaitContextAlloc(module, &waitContext));

    /* Clean up */
    LwSciSyncAttrListFree(signalerList);
    LwSciSyncAttrListFree(waiterList);
    LwSciSyncAttrListFree(newConflictList);
    LwSciSyncAttrListFree(reconciledList);
}

void deinit(void)
{
    if (waitContext != NULL) {
        LwSciSyncCpuWaitContextFree(waitContext);
    }
    if (syncObj != NULL) {
        LwSciSyncObjFree(syncObj);
    }
    if (module != NULL) {
        LwSciSyncModuleClose(module);
    }
}

/* Comparator function, sorts in increasing order */
int compare(const void* a, const void* b)
{
    uint64_t va = *(const uint64_t*)a;
    uint64_t vb = *(const uint64_t*)b;

    if (va == vb) {
        return 0;
    } else if (va > vb) {
        return 1;
    } else {
        return -1;
    }
}

void perfAnalyze(uint32_t N, uint64_t* start, uint64_t* end)
{
    uint64_t total = 0U;
    uint64_t* diff = (uint64_t*)malloc(sizeof(uint64_t) * N);

    /* Callwlate average value */
    for (uint32_t i = 0U; i < N; i++) {
        if (end[i] <= start[i]) {
            fprintf(stderr, "WARNING: Sample %d end time <= start time: "
                    "%lu <= %lu\n", i, end[i], start[i]);
            diff[i] = 0U;
        } else {
            diff[i] = end[i] - start[i];
        }
        total += diff[i];

        printf("%lu, %lu\n", start[i], diff[i]);
    }

    fprintf(stderr, "RESULT: Average Latency: %lu ns\n", total / N);


    /* Get the 99.99 percentile value */
    qsort(diff, N, sizeof(uint64_t), compare);

    double param = (9999.0 / 10000.0) * N;
    uint64_t idx = (uint64_t)floor(param);

    fprintf(stderr, "RESULT: 99.99th percentile: %lu ns\n", diff[idx]);

    free(diff);
}

/* Simulate the operation rate */
static inline void waitForNextOp(uint64_t const nextTime)
{
    struct timespec sleepTime = {
        .tv_sec = 0,
        .tv_nsec = 0,
    };

    uint64_t lwrTime = get_timestamp();

    /* Simulate the operation rate */
    if (nextTime > lwrTime) {
        sleepTime.tv_nsec = nextTime - lwrTime;
        nanosleep(&sleepTime, NULL);
    }
}

#endif // !COMMON_H
