
#include <errno.h>
#include <stdlib.h>
#include <vector>

#include "ParsingUtility.h"
#include "../../common/DcgmStringTokenize.h"
#include "dcgm_structs.h"
#include "dcgm_fields.h"


unsigned long long GetThrottleIgnoreReasonMaskFromString(std::string reasonStr) 
{
    // Parse given reasonStr to determine throttle reasons ignore mask
    
    // Early exit check
    if (reasonStr.size() == 0)
    {
        // empty string is equivalent to having no value set
        return DCGM_INT64_BLANK;
    }
    
    // Check if reasonStr contains the integer value of the mask
    if (isdigit(reasonStr[0])) 
    {
        // Colwert from str to ull
        const char *s = reasonStr.c_str();
        char *end;
        uint64_t mask = strtoull(s, &end, 10);
        
        // mask colwerted successfully and is valid
        if (end != s && errno != ERANGE && mask != 0 && mask <= MAX_THROTTLE_IGNORE_MASK_VALUE)
        {
            return mask;
        }
        // Colwersion not successful or invalid value for mask or value was set to 0
        return 0;
    }

    // Input string could be a CSV list of reason names    
    std::vector<std::string> reasons;
    tokenizeString(reasonStr, ",", reasons);
    uint64_t mask = 0;

    for (size_t i = 0; i < reasons.size(); i++) 
    {
        if (reasons[i] == "hw_slowdown")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN;
        }
        else if (reasons[i] == "sw_thermal")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL;
        }
        else if (reasons[i] == "hw_thermal")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL;
        }
        else if (reasons[i] == "hw_power_brake")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE;
        }
    }

    if (mask == 0)
    {
        // Invalid csv list of reasons - treat as no value set
        return DCGM_INT64_BLANK;
    }
    return mask;
}
