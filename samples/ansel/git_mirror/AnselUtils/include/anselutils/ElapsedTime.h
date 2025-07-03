#pragma once
#include <chrono>

namespace anselutils
{
    class ElapsedTime
    {
    public:
        // by default ElapsedTime will not cap the time callwlated between two time points.
        // Setting capElapsedTime to a value >= 0.0f, will cap the real elapsed time with that value.
        // Example of where this is needed would be a camera controller which could be created at one time, but
        // the updated much later. In this case the first update would cause the camera controller to move by a large value.
        ElapsedTime(const float capElapsedTime = -1.0f);
        float elapsed();
    private:
        float m_cap;
        std::chrono::time_point<std::chrono::system_clock> m_lastTime;
    };
}
