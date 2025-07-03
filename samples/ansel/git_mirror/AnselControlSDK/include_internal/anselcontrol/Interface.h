#pragma once

#include "anselcontrol/Configuration.h"

namespace anselcontrol
{
    struct CaptureShotState
    {
        ShotDescription shotDesc;
        bool isValid;
        
        CaptureShotState():
            isValid(false)
        {
        }
    };

    ANSEL_CONTROL_SDK_INTERNAL_API bool ilwalidateCaptureShotScenarioState();
}
