 /* Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifdef UNIT_WINDOWS
#include <windows.h>
#endif

#include "lwtest.h"

/*
 * @brief Print the string in red color, used to indicate failure
 *
 * @param[in]    str     String to be colored
 */
void colorMeRed(const char* str)
{
#ifdef UNIT_WINDOWS
    HANDLE h = GetStdHandle ( STD_OUTPUT_HANDLE );
    WORD wOldColorAttrs;
    CONSOLE_SCREEN_BUFFER_INFO csbiInfo;

    // First save the current color information
    GetConsoleScreenBufferInfo(h, &csbiInfo);
    wOldColorAttrs = csbiInfo.wAttributes;

    //Set the new color information
    SetConsoleTextAttribute ( h, FOREGROUND_RED | FOREGROUND_INTENSITY );

    printf ( "%s", str);

    //Restore the original colors
    SetConsoleTextAttribute ( h, wOldColorAttrs);
#else
    printf("\e[31m%s\e[0m", str);
#endif
}

#ifdef UNIT_WINDOWS

extern jmp_buf jmpbuf;

/*
 * @brief run the test case with exceptions handled
 *        Windows version.
 *
 * @param[in] tc  Pointer to test case to be run
 *
 */
void runWithExceptionHandledWindows(LwTest* tc)
{
    __try
    {
        (tc->function)(tc);

        // if there is a verify function, then execute it
        if (tc->verify != NULL)
        {
            (tc->verify)(tc);
        }

    }

    __except(GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ?
              EXCEPTION_EXELWTE_HANDLER :EXCEPTION_CONTINUE_SEARCH)
    {
            longjmp(jmpbuf, RETURN_FROM_ACCESS_VIOLATION_EXCEPTION);
    }
}
#endif
