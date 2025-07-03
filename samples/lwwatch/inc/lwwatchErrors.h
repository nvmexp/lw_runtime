/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2013 by LWPU Corporation. All rights reserved.  All 
 * information contained herein is proprietary and confidential to LWPU 
 * Corporation.  Any use, reproduction, or disclosure without the written 
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - July 2004 
// Macros used to print error messages. These will
// mostly go away when all functions in osWin.c are
// implemented for lwwatchMods and lwwatchHw
// 
//*****************************************************

//
// Macros used to print error messages for commands *not* implemented
//
#define PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN() do { \
    dprintf("lw: %s:%d not implemented.\n", __FILE__, __LINE__);\
    return;\
} while (0)

#define PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0() do { \
    dprintf("lw: %s:%d not implemented.\n", __FILE__, __LINE__);\
    return 0;\
} while (0)

//
// Macros used to print error messages for commands *not* implemented for
// lwwatchMods
//
#define LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE() do \
    { \
        if (usingMods) \
            dprintf("lw: " __FILE__ ": " __FUNCTION__ " not implemented for "\
                "lwwatchMods.\n");\
    } while (0)

#define PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE_AND_RETURN() \
    do \
    { \
        if (usingMods) \
        {   dprintf("lw: " __FILE__ ": " __FUNCTION__ " not implemented for "\
                "lwwatchMods.\n");\
            return;\
        } \
    } while (0)

#define PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0() \
    do \
    { \
        if (usingMods) \
        {   dprintf("lw: " __FILE__ ": " __FUNCTION__ " not implemented for "\
                "lwwatchMods.\n");\
            return 0;\
        } \
    } while (0)

#define PRINT_LWWATCHMODS_NOT_IMPLEMENTED_MESSAGE2() \
    do { \
        if (usingMods) \
        { \
            dprintf("lw: " __FILE__ ": " __FUNCTION__ " not implemented for "\
                "lwwatchMods.\n");\
            dprintf("lw: Replace the call to " __FUNCTION__" or write it.\n"); \
        } \
    } while(0)


//
// Macros used to print error messages for commands implemented *only* for
// lwwatchMods
//
#define LWWATCHMODS_IMPLEMENTED_ONLY_MESSAGE() do \
    { \
        if (!usingMods) \
            dprintf("lw: " __FILE__ ": " __FUNCTION__ " only implemented for "\
                "lwwatchMods");\
    } while (0)

#define LWWATCHMODS_IMPLEMENTED_ONLY_MESSAGE_AND_RETURN_VAL(value) do \
    { \
        if (!usingMods) \
        { \
            dprintf("lw: " __FILE__ ": " __FUNCTION__ " only implemented for "\
                "lwwatchMods");\
            return (value); \
        } \
    } while (0)

#define LWWATCHMODS_IMPLEMENTED_ONLY_MESSAGE_AND_RETURN() do \
    { \
        if (!usingMods) \
        { \
            dprintf("lw: " __FILE__ ": " __FUNCTION__ " only implemented for "\
                "lwwatchMods");\
            return; \
        } \
    } while (0)
