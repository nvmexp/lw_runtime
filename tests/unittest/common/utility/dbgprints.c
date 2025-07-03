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
 * @file   dbgprints.c
 * @brief  utility functions for overridden debug printfs
 */

#include <stdio.h>
#include "lwtypes.h"
#include "stdarg.h"

//
// Flag to either stub DBG_PRINTF or redirect to
// unittest infra specific function
//
static LwBool enableDbgPrintf = LW_FALSE;

/*
 * @brief Unittest infra implementation of DBG_PRINTF
 *
 * @param[in] file       : Filename from where it got called
 * @param[in] line       : Line number
 * @param[in] function   : function from where it got called
 * @param]in] debuglevel : just a place holder as per actual call
 * @param[in] s          : Sting which wants to print
 */
void utDbg_Printf(const char* file, int line, const char *function, int debuglevel, const char* s, ...)
{
    va_list args;
    va_start( args, s );

    if (enableDbgPrintf)
    {
        printf("File : %s, Line : %d, Function : %s, MSG : ", file, line, function);
        vprintf(s, args);
        printf("\n");
    }
    va_end( args );
    return;
}

/*
 * @brief Stub or enable unittest infra specific DBG_PRINTF functionality
 *
 * @param[in] flag : Specify if stub the prints or enable it.
 */
void utApiEnableDbgPrintf(LwBool flag)
{
    enableDbgPrintf = flag;
}
