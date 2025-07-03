#pragma once
#include <stdint.h>

namespace
{
    unsigned int year()
    {
        return (__DATE__[7] - '0') * 1000 + (__DATE__[8] - '0') * 100 + (__DATE__[9] - '0') * 10 + (__DATE__[10] - '0');
    }

    unsigned int month()
    {
        return __DATE__[0] == 'J' && __DATE__[1] == 'a' && __DATE__[2] == 'n' ? 1 :
            (__DATE__[0] == 'F' && __DATE__[1] == 'e' && __DATE__[2] == 'b') ? 2 :
            (__DATE__[0] == 'M' && __DATE__[1] == 'a' && __DATE__[2] == 'r') ? 3 :
            (__DATE__[0] == 'A' && __DATE__[1] == 'p' && __DATE__[2] == 'b') ? 4 :
            (__DATE__[0] == 'M' && __DATE__[1] == 'a' && __DATE__[2] == 'y') ? 5 :
            (__DATE__[0] == 'J' && __DATE__[1] == 'u' && __DATE__[2] == 'n') ? 6 :
            (__DATE__[0] == 'J' && __DATE__[1] == 'u' && __DATE__[2] == 'l') ? 7 :
            (__DATE__[0] == 'A' && __DATE__[1] == 'u' && __DATE__[2] == 'g') ? 8 :
            (__DATE__[0] == 'S' && __DATE__[1] == 'e' && __DATE__[2] == 'p') ? 9 :
            (__DATE__[0] == 'O' && __DATE__[1] == 'c' && __DATE__[2] == 't') ? 10 :
            (__DATE__[0] == 'N' && __DATE__[1] == 'o' && __DATE__[2] == 'v') ? 11 :
            (__DATE__[0] == 'D' && __DATE__[1] == 'e' && __DATE__[2] == 'c') ? 12 : 0;
    }

    unsigned int day()
    {
        return (__DATE__[4] - '0') * 10 + (__DATE__[5] - '0');
    }

    unsigned int hour()
    {
        return (__TIME__[0] - '0') * 10 + (__TIME__[1] - '0');
    }

    unsigned int minutes()
    {
        return (__TIME__[3] - '0') * 10 + (__TIME__[4] - '0');
    }

    unsigned int seconds()
    {
        return (__TIME__[6] - '0') * 10 + (__TIME__[7] - '0');
    }

    uint64_t getVersion()
    {
        const uint64_t major = 1;
        const uint64_t minor = 0;
        // Version format is XXX.YYY.TIMESTAMP, where TIMESTAMP is:
        // YYYY.MM.DD.HH.MM.SS
        const uint64_t version = major * 1000000000000000000ull +
            minor * 1000000000000000ull +
            year() * 1000000000000ull +
            month() * 100000000 +
            day() * 1000000 +
            hour() * 10000 +
            minutes() * 100 +
            seconds();
        return version;
    }
}
