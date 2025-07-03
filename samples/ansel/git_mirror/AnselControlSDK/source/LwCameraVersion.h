#pragma once
#include <stdint.h>

namespace internal
{
    struct Version
    {
        uint16_t major;
        uint16_t minor;
        uint16_t build;
        uint16_t revision;
    };

    bool getLwCameraVersion(Version& version);
}
