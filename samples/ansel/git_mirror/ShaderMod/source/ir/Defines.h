#pragma once

#define IR_REFLECTION_NOT_FOUND (UINT)-1

#define IR_FILENAME_MAX         FILENAME_MAX
#define IR_EFFECT_BASEPATH      L"shaders"

#define IR_RESOURCENAME_MAX     64

namespace shadermod
{
    struct ResourceName
    {
        char name[IR_RESOURCENAME_MAX];
    };
}
