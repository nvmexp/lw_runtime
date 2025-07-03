 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <windows.h>

#include <lw32.h>
#include <lwos.h>
#include <lwcm.h>
#include <lwrmapi.h>

#include "lwtypes.h"
#include "Lwcm.h"

#include "CycleStatsConsole.h"
#include "../../../../drivers/common/cyclestats/gpu/lwPerfMonHW.h"

#define TRUE            1
#define FALSE           0

// Q: why isn't this a build-in define ?
#undef max
#define max(a,b)    __GL_MAX(a, b)

//---------------------------------------------------------------------------

// arghh, d3d has its own set of debug printfs
static void PF(const char *szFormat, ...)
{
    char tmp[8192];
    _vsnprintf(tmp, sizeof(tmp), szFormat, (va_list)(&szFormat+1));
    log("%s\n", tmp);
}

static void DPF(const char *szFormat, ...)
{
#ifdef DEBUG
    char tmp[8192];
    _vsnprintf(tmp, sizeof(tmp), szFormat, (va_list)(&szFormat+1));
    log("%s\n", tmp);
#endif
}

#define RPRINTF(ARGS) PF ARGS
#define TPRINTF(ARGS) PF ARGS

#define lwStrCaseCmp(X, Y) _stricmp(X, Y)

// ugly, but allows us to share almost all the code...
#include "../../../../drivers/common/inc/generics/lwStringClassGlue.h"
#include "../../../../drivers/common/src/generics/lwStringClass.cpp"
#include "../../../../drivers/common/cyclestats/gpu/lwPerfMonHW.cpp"
