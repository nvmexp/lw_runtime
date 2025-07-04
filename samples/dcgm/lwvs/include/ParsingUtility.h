#ifndef PARSING_UTILITY_H
#define PARSING_UTILITY_H

/* 
    Helpers and constants for command line + config file parsing.
*/

#include <string>

/* Helpers for parsing throttle ignore reasons */

// Maximum valid value of throttle ignore reason mask (i.e value of failureMask in DcgmRecorder::CheckForThrottling)
#define MAX_THROTTLE_IGNORE_MASK_VALUE 0x00000000000000E8LL

/* 
Parses the given reasonStr to determine the throttle ignore mask. 

Parameters: 
    reasonStr: 
        A comma separated list of throttle reasons OR the integer value of the ignore mask.
        Lwrrently, only the following reasons are parsed (the corresponding reason mask is in brackets):
        - 'hw_slowdown' (DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN)
        - 'sw_thermal' (DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL)
        - 'hw_thermal' (DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL)
        - 'hw_power_brake' (DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE)
        Ensue that reasonStr is lowercase before calling this method (unless the string is just the integer value of 
        the ignore mask).
Return: 
    The parsed throttle ignore mask, or 0 if reasonStr could not be parsed. Any unrecognized tokens are ignored during 
    parsing.
*/
unsigned long long GetThrottleIgnoreReasonMaskFromString(std::string reasonStr);

#endif // PARSING_UTILITY_H
