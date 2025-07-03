#pragma once
#include <stdint.h>

namespace anselutils
{
    double colwertVerticalToHorizontalFov(double fov, uint32_t viewportWidth, uint32_t viewportHeight);
    double colwertHorizontalToVerticalFov(double fov, uint32_t viewportWidth, uint32_t viewportHeight);

    template<typename T>
    bool areAlmostEqual(T x, T y)
    {
        return std::abs(x - y) <= std::numeric_limits<T>::epsilon();
    }
}
