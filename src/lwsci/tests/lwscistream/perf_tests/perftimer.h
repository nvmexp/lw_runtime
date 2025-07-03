//! \file
//! \brief Perf Timer declaration.
//!
//! \copyright
//! Copyright (c) 2019 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PERFTIMER_H
#define PERFTIMER_H

#include <time.h>
#include "util.h"

class PerfTimer
{
public:
    PerfTimer(void) = default;

    void setStart(void)
    {
        clock_gettime(CLOCK_REALTIME, &start);
        hasStarted = true;
    };

    void setStart(struct timespec t)
    {
        start = t;
        hasStarted = true;
    };

    void setStop(void)
    {
        clock_gettime(CLOCK_REALTIME, &stop);
    }

    void setStop(struct timespec t)
    {
        stop = t;
    }

    double checkInterval(void)
    {
        double delta = double(stop.tv_sec - start.tv_sec);
        delta += double(stop.tv_nsec - start.tv_nsec) / 1000000000.0;
        // Return in micro-second (us)
        return delta * 1000000;
    };

    bool isSet(void)
    {
        return hasStarted;
    };

private:
    bool hasStarted{ false };
    struct timespec start;
    struct timespec stop;
} ;

#endif // PERFTIMER_H
