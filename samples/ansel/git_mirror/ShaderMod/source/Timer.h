#pragma once

#include <stdint.h>
#include <string>

namespace shadermod
{

    // TODO: Windows-specific code
    using TIMER_TYPE = __int64;

    class Timer
    {
    public:

        Timer();

        void Start();
        double Time();
        double TimeFromDeltaCount(const TIMER_TYPE & deltaCount);
        TIMER_TYPE DeltaCount();
        void PrintTimeSinceLast(const std::string& label);

    private:

        TIMER_TYPE Count();

        static inline TIMER_TYPE CallwlateFrequency();
        static TIMER_TYPE GetFrequency();

        TIMER_TYPE m_startTime;
        double m_elapsedTime;
        TIMER_TYPE m_timerFrequency;
    };

}