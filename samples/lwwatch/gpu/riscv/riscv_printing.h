/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef PRINTING_H
#define PRINTING_H

#include <lwtypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <print.h>
#include "lwwatch.h"

enum PRINT_LEVEL
{
    PL_ERROR = 0,
    PL_INFO,
    PL_DEBUG,
};

extern LwBool debugPrints;

int lprintf(enum PRINT_LEVEL level, const char *format, ...);
int lvprintf(enum PRINT_LEVEL level, const char *format, va_list va);

#endif // PRINTING_H
