#include <Windows.h>

#include "Timer.h"
#include "Log.h"

namespace shadermod
{
    Timer::Timer()
    {
        m_timerFrequency = GetFrequency();
    }

    void Timer::Start()
    {
        LARGE_INTEGER s;
        QueryPerformanceCounter(&s);
        m_startTime = s.QuadPart;
    }

    double Timer::Time()
    {
        m_elapsedTime = ((Count() - m_startTime) * 1000 / (double)m_timerFrequency);
        return m_elapsedTime;
    }

    double Timer::TimeFromDeltaCount(const TIMER_TYPE & deltaCount)
    {
        m_elapsedTime = (deltaCount * 1000 / (double)m_timerFrequency);
        return m_elapsedTime;
    }

    TIMER_TYPE Timer::DeltaCount()
    {
        return Count() - m_startTime;
    }

    TIMER_TYPE Timer::Count()
    {
        LARGE_INTEGER s;
        QueryPerformanceCounter(&s);
        return s.QuadPart;
    }

    inline TIMER_TYPE Timer::CallwlateFrequency()
    {
        LARGE_INTEGER tFreq;
        QueryPerformanceFrequency(&tFreq);
        return tFreq.QuadPart;
    }

    TIMER_TYPE Timer::GetFrequency()
    {
        static TIMER_TYPE Freq = CallwlateFrequency();
        return Freq;
    }

    void Timer::PrintTimeSinceLast(const std::string& label)
    {
        LOG_DEBUG("Timer %s: %f", label.c_str(), Time());
        
        Start();
    }
}
