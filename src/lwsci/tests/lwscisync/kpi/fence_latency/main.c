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

#include "common.h"

#include "wait_expired.h"
#include "wait_unexpired.h"
#include "signal.h"


int main(int argc, char** argv)
{
    TestArgs testArgs = {
        .testType = ' ',
        .primitiveType = 'p',
        .N = 1,
        .rate = 0,
        .pinSignaler = false,
        .pinWaiter = false,
        .signalerCore = 5,
        .waiterCore = 6,
    };

    if (!parseArgs(argc, argv, &testArgs)) {
        help();
        exit(1);
    }

    fprintf(stderr, "INFO: Test Type: %c\n", testArgs.testType);
    fprintf(stderr, "INFO: Sync Primitive Type: %s\n",
            testArgs.primitiveType == 'p' ? "syncpoint" : "sysmem semaphore");
    fprintf(stderr, "INFO: Iterations: %u\n", testArgs.N);

    if (testArgs.pinWaiter) {
        fprintf(stderr, "INFO: Pin waiter to core: %d\n",
                testArgs.waiterCore);
    }

    if (testArgs.pinSignaler) {
        fprintf(stderr, "INFO: Pin signaler to core: %d\n",
                testArgs.signalerCore);
    }

    if (testArgs.rate > 0) {
        fprintf(stderr, "INFO: Rate (fence ops per second): %d \n",
                testArgs.rate);
    } else {
        fprintf(stderr, "INFO: Rate (fence ops per second): Unlimited\n");
    }

    printf("timestamp, latency\n");
    printf("nsec, nsec\n");

    if (testArgs.primitiveType == 'p') {
        init(LwSciSyncInternalAttrValPrimitiveType_Syncpoint);
    } else {
        init(LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore);
    }

    if (testArgs.testType == 's') {
        signalBenchmark(&testArgs);
    } else if (testArgs.testType == 'e') {
        expiredBenchmark(&testArgs);
    } else if (testArgs.testType == 'u') {
        unexpiredBenchmark(&testArgs);
    } else {
        fprintf(stderr, "ERROR: Invalid Test Type.\n");
        help();
    }

    deinit();

    return 0;
}
