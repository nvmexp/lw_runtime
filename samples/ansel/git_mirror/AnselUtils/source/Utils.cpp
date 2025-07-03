#pragma once
#include "anselutils/Utils.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace anselutils
{
    double colwertVerticalToHorizontalFov(double fov, uint32_t viewportWidth, uint32_t viewportHeight)
    {
        return 2.0 * atan(tan(fov * M_PI / 360.0f) * double(viewportWidth) / double(viewportHeight)) * 180.0 / M_PI; //in degrees
    }

    double colwertHorizontalToVerticalFov(double fov, uint32_t viewportWidth, uint32_t viewportHeight)
    {
        return 2.0 * atan(tan(fov * M_PI / 360.0f) * double(viewportHeight) / double(viewportWidth)) * 180.0 / M_PI; //in degrees
    }
}
