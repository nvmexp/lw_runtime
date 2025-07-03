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

#ifndef UTIL_H
#define UTIL_H

#include <linux/sched.h>
#include <time.h>

#define BUSY_WAIT_ITERATIONS 50000

static inline void pin_thread_to_core(int core)
{
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (core < 0 || core >= num_cores) {
        fprintf(stderr, "ERROR: Cannot pin thread to core %d.\n", core);
        exit(1);
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    pthread_t lwrrent_thread = pthread_self();
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

static inline uint64_t get_timestamp(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)NSEC_IN_SEC * ts.tv_sec + ts.tv_nsec);
}

#endif // !UTIL_H
