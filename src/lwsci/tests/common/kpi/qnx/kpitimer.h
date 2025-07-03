//! \file
//! \brief LwSci perf test timer.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef KPITIMER_H
#define KPITIMER_H
#include <sys/neutrino.h>
#include <sys/syspage.h>
#include <iostream>

#define BILLION 1000000000U
#define USEC2NSEC 1000U
#define SEC2USEC  1000000U

typedef struct {
    uint64_t s_Start;
    uint64_t s_End;
    uint64_t sec, nsec;
    bool isSet = false;
} KPItimer;

void inline KPIStart(KPItimer* obj)
{
    obj->s_Start = ClockCycles();
}

void inline KPIDiffTime(KPItimer* obj)
{
    uint64_t nCycles = obj->s_End - obj->s_Start;
    uint64_t cycles_per_sec = SYSPAGE_ENTRY(qtime)->cycles_per_sec;

    obj->sec = nCycles / cycles_per_sec;
    obj->nsec = ((nCycles % cycles_per_sec) * BILLION) / cycles_per_sec;

    if (obj->nsec >= BILLION) {
        obj->nsec -= BILLION;
        obj->sec += 1;
    }

    // Return in micro-second (us)
    double delta = obj->sec * SEC2USEC + obj->nsec / (double)USEC2NSEC;
    std::cout << delta << std::endl;
}

void inline KPIEnd(KPItimer* obj, bool diffTime = true)
{
    obj->s_End = ClockCycles();
    obj->isSet = true;

    if (diffTime) {
        KPIDiffTime(obj);
    }
}

#endif // KPITIMER_H