#pragma once

#include <Windows.h>
#include <stdint.h>

// TODO: Windows-specific code
using TIMER_TYPE = __int64;

class Timer
{
public:

    Timer();

    void start();
    double time();
    double timeFromDeltaCount(const TIMER_TYPE & deltaCount);
    TIMER_TYPE deltaCount();

private:

    TIMER_TYPE count();

    static inline TIMER_TYPE callwlateFrequency();
    static TIMER_TYPE getFrequency();

    TIMER_TYPE m_startTime;
    double m_elapsedTime;
    TIMER_TYPE m_timerFrequency;
};
