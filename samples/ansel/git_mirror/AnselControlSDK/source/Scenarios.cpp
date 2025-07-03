#define ANSEL_CONTROL_SDK_EXPORTS
#include <anselcontrol/Defines.h>
#include <anselcontrol/Common.h>
#include <anselcontrol/Scenarios.h>
#include <anselcontrol/Interface.h>
#include <windows.h>
#include <stdio.h>

namespace anselcontrol
{
    // State
    static CaptureShotState s_scenarioCaptureShotState;
    extern bool s_anselServerIsAvailable;

    // Public
    Status captureShot(const ShotDescription & shotDesc)
    {
        if (!s_anselServerIsAvailable)
        {
            return kControlNotInitialized;
        }

        s_scenarioCaptureShotState.shotDesc = shotDesc;
        s_scenarioCaptureShotState.isValid = true;
        return kControlSuccess;
    }

    // Internal

    // Function that allows to get the "captureShot" scenario state
    ANSEL_CONTROL_SDK_INTERNAL_API void getCaptureShotScenarioState(CaptureShotState & captureShotState)
    {
        captureShotState = s_scenarioCaptureShotState;
    }
    // Function that resets the "captureShot" scenario state
    ANSEL_CONTROL_SDK_INTERNAL_API bool ilwalidateCaptureShotScenarioState()
    {
        s_scenarioCaptureShotState.isValid = false;
        return true;
    }
}
