#pragma once

namespace ui
{

    static int getPrecisionFromRange(float absRange)
    {
        if (absRange > 50.0f)
        {
            return 0;
        }
        else if (absRange > 5.0f)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }

}
