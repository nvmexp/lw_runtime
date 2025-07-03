#define ANSEL_SDK_EXPORTS
#include <ansel/Defines.h>
#include <ansel/Version.h>
#include <stdint.h>

namespace ansel
{
    ANSEL_SDK_INTERNAL_API uint64_t getVersion()
    {
        return ANSEL_SDK_VERSION;
    }
}
