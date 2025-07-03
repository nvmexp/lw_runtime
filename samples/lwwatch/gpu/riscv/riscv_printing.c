/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdio.h>
#include <stdlib.h>

#include "riscv_printing.h"

LwBool debugPrints = LW_TRUE;

int lvprintf(enum PRINT_LEVEL level, const char *format, va_list va)
{
    char sBuffer[4096];
    vsprintf(sBuffer, format, va);
    va_end(va);
#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
    switch (level) {
    case PL_ERROR:
        return MessageBox(NULL, sBuffer, "Lwwatch: Error", MB_OK | MB_ICONERROR);
        break;
    case PL_DEBUG:
        if (!debugPrints) break;
    case PL_INFO:
        return dprintf("%s", sBuffer);
    }

#elif LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
    switch (level) {
    case PL_DEBUG:
        if (!debugPrints) break;
    case PL_ERROR:
    case PL_INFO:
        return dprintf("%s", sBuffer);
    }
#endif
    return -1;
}

int lprintf(enum PRINT_LEVEL level, const char *format, ...)
{
    va_list va;
    va_start(va, format);
    return lvprintf(level, format, va);
}
