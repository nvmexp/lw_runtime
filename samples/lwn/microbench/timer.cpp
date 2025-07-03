/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "timer.hpp"
#include <time.h>
#include <assert.h>

#include <stdio.h>

#if defined(LW_HOS)
  #include <nn/os.h>
#elif defined(_WIN32)
  #include <windows.h>
#endif

using namespace LwnUtil;

double Timer::ticksToSecs(uint64_t t)
{
    return (double)t / m_frequency;
}

Timer::Timer()
{
#if defined(LW_HOS)
    m_frequency = nn::os::GetSystemTickFrequency();
#elif defined(_WIN32)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    assert(freq.QuadPart < (1ULL << 32));
    m_frequency = (int)freq.QuadPart;
#elif defined(LW_LINUX)
    m_frequency = 1000000000;       // second -> nanosecond
#else
    m_frequency = 1000000;
#endif
}

Timer::~Timer()
{
}

Timer* Timer::instance()
{
    static Timer timer;
    return &timer;
}

uint64_t Timer::getTicks()
{
#if defined(LW_HOS)
    return nn::os::GetSystemTick().GetInt64Value();
#elif defined(_WIN32)
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
#elif defined(LW_LINUX)
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_nsec + m_frequency * now.tv_sec;
#else
#error getTicks not implemented for an unknown platform
    return 0;
#endif
}
