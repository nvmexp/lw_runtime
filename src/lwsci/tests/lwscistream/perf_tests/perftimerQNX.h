//! \file
//! \brief QNX Perf Timer declaration.
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

#include <sys/neutrino.h>
#include <sys/syspage.h>
#include "util.h"

class PerfTimer
{
public:
    PerfTimer(void) = default;

    void setStart(void)
    {
        start = ClockCycles();
        hasStarted = true;
    };

    void setStart(uint64_t t)
    {
        start = t;
        hasStarted = true;
    };

    void setStop(void)
    {
        stop = ClockCycles();
    }

    void setStop(uint64_t t)
    {
        stop = t;
    }

    double checkInterval(void)
    {
        uint64_t nCycles = stop - start;
        double delta = (double)nCycles / cyclesPerSec;
        // Return in micro-second (us)
        return delta * 1000000;
    };

    bool isSet(void)
    {
        return hasStarted;
    };

private:
    static uint64_t cyclesPerSec;

    bool hasStarted{ false };

    uint64_t start{ 0U };
    uint64_t stop{ 0U };
} ;

#endif // PERFTIMER_H
