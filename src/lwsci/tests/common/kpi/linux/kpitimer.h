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

#include <time.h>
#include <iostream>

#define USEC2NSEC 1000U
#define SEC2USEC  1000000U

typedef struct {
    timespec s_Start;
    timespec s_End;
    uint64_t sec, nsec;
    bool isSet = false;
} KPItimer;

void inline KPIStart(KPItimer* obj)
{
    clock_gettime(CLOCK_MONOTONIC, &obj->s_Start);
}

void inline KPIDiffTime(KPItimer* obj)
{
    obj->sec = obj->s_End.tv_sec - obj->s_Start.tv_sec;
    obj->nsec = obj->s_End.tv_nsec - obj->s_Start.tv_nsec;

    // Return in micro-second (us)
    double delta = obj->sec * SEC2USEC + obj->nsec / (double)USEC2NSEC;
    std::cout << delta << std::endl;
}

void inline KPIEnd(KPItimer* obj, bool diffTime = true)
{
    clock_gettime(CLOCK_MONOTONIC, &obj->s_End);
    obj->isSet = true;

    if (diffTime) {
        KPIDiffTime(obj);
    }
}

#endif // KPITIMER_H