#include <algorithm>
#include <anselutils/ElapsedTime.h>

namespace anselutils
{
    ElapsedTime::ElapsedTime(const float capElapsedTime) : m_cap(capElapsedTime), m_lastTime(std::chrono::system_clock::now())
    {
    }

    float ElapsedTime::elapsed()
    {
        const auto now = std::chrono::system_clock::now();
        const std::chrono::duration<float> elapsed = now - m_lastTime;
        m_lastTime = now;
        return std::min(elapsed.count(), m_cap);
    }
}
