/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   UTILITY.h
 * @brief  declarations to Logging related functions
 */

#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <stdio.h>
#include "lwtypes.h"

// opens the file for logging
void initLogFile();

// writes the log into the file
void unitPrintf(char* file, int line);

// closes the file after logging
void closeLogFile();

// checks if log file has been created (ie wheter UNIT_ASSERT was hit)
int checkIfAssertLogFile();

// Unittest infra implementation of DBG_PRINTF
void utDbg_Printf(const char* file, int line, const char *function, int debuglevel, const char* s, ...);

// Stub or enable unittest infra specific DBG_PRINTF functionality
void utApiEnableDbgPrintf(LwBool flag);

#ifndef UNIT_DEBUG_BUILD

// only logging here
#define UNIT_ASSERT(cond)                                                     \
    do                                                                        \
    {                                                                         \
        if ((cond) == 0x0)                                                    \
            unitPrintf(__FILE__, __LINE__);                                    \
    }while(0)

#else

#ifndef UNIT_WINDOWS

    #define UNIT_ASSERT(cond)                                                     \
    do                                                                            \
    {                                                                             \
        if ((cond) == 0x0)                                                        \
        {                                                                         \
             printf("Error::Assertion Failed at File:%s Line:%d\n",               \
                    __FILE__, __LINE__);                                          \
            __asm__("int $3");                                                    \
        }                                                                         \
    }while(0)

#else

    #define UNIT_ASSERT(cond)                                                     \
    do                                                                            \
    {                                                                             \
        if ((cond) == 0x0)                                                        \
        {                                                                         \
             printf("Error::Assertion Failed at File:%s Line:%d\n",               \
             __FILE__, __LINE__);                                                 \
             __debugbreak();                                                      \
        }                                                                         \
    }while(0)

#endif // UNIT_WINDOWS

#endif // UNIT_DEBUG_BUILD

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef NULL
#define NULL 0
#endif

#endif // _UTILITY_H_
