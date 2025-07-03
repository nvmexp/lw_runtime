#define ANSEL_CONTROL_SDK_EXPORTS
#include <anselcontrol/Defines.h>
#include <anselcontrol/Version.h>
#include <stdint.h>

namespace anselcontrol
{
    ANSEL_CONTROL_SDK_INTERNAL_API uint64_t getVersion()
    {
        return ANSEL_CONTROL_SDK_VERSION;
    }
}
