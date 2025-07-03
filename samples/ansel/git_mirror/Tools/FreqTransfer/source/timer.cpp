#include "timer.h"

Timer::Timer()
{
    m_timerFrequency = getFrequency();
}

void Timer::start()
{
    LARGE_INTEGER s;
    QueryPerformanceCounter(&s);
    m_startTime = s.QuadPart;
}

double Timer::time()
{
    m_elapsedTime = ((count() - m_startTime) * 1000 / (double)m_timerFrequency);
    return m_elapsedTime;
}

double Timer::timeFromDeltaCount(const TIMER_TYPE & deltaCount)
{
    m_elapsedTime = (deltaCount * 1000 / (double)m_timerFrequency);
    return m_elapsedTime;
}

TIMER_TYPE Timer::deltaCount()
{
    return count() - m_startTime;
}

TIMER_TYPE Timer::count()
{
    LARGE_INTEGER s;
    QueryPerformanceCounter(&s);
    return s.QuadPart;
}

inline TIMER_TYPE Timer::callwlateFrequency()
{
    LARGE_INTEGER tFreq;
    QueryPerformanceFrequency(&tFreq);
    return tFreq.QuadPart;
}

TIMER_TYPE Timer::getFrequency()
{
    static TIMER_TYPE Freq = callwlateFrequency();
    return Freq;
}
