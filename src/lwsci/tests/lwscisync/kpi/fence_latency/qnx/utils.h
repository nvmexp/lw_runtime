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

#include <sys/trace.h>
#include <sys/syspage.h>
#include <sys/neutrino.h>

uint64_t nsPerCycles;

#define WAIT_NS 50000

static inline void pin_thread_to_core(int core)
{
    int runmask = 1 << core;
    if (ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET, (void*) &runmask) == -1) {
        fprintf(stderr, "WARNING: Unable to pin thread to core %d,"
                        "runmask 0x%x (%d).\n", core, runmask, runmask);
    }
}

static inline uint64_t get_timestamp(void)
{
    return ClockCycles() * nsPerCycles;
}

#endif // !UTIL_H
