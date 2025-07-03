/*
 * Copyright 2003-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// exts.c
//
//*****************************************************

//
// includes
//
#include "lwwatch.h"
#include "vgpu.h"

#if !defined(LW_MAC_KEXT)
#include <stdio.h>
#include <string.h>
#endif

#include "os.h"
#include "hal.h"

#if defined(USERMODE) && defined(LW_WINDOWS)
#include <usermode.h>
#include <wdbgexts.h>
#include <dbgeng.h>
#endif

#ifndef CLIENT_SIDE_RESMAN
#if defined(USERMODE)

#elif defined(LW_WINDOWS)
#include "lwwatch.h"

#elif LWWATCHCFG_IS_PLATFORM(UNIX) && !LWWATCHCFG_FEATURE_ENABLED(UNIX_USERMODE)
#include "lwstring.h"
#endif
#endif

#include "exts.h"

//
// lw includes
//
#include "exts.h"
#include "falcon.h"
#include "pmu.h"
#include "dpu.h"
#include "tegrasys.h"
#include "socbrdg.h"

#if defined(LWDEBUG_SUPPORTED)
#include "prbdec.h"
#include "g_lwdebug_pb.h"
#include "g_regs_pb.h"
#endif

#if !defined(MINIRM)
#include "br04.h"
#endif

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_IS_PLATFORM(OSX) || defined(CLIENT_SIDE_RESMAN)
typedef char* PCSTR;
#endif

//
// FIXME: redef bug
// note that in windows usermode, "args" has to be overwritten by "userargs":
//  #if defined(USERMODE) && defined(LW_WINDOWS)
//       args = userargs;
//  #endif
// this is due to a redef problem that causes args to be overwritten
//

#if defined(EFI_APP)
extern int nargs, argstaken;
extern char *argv[MAX_ARGS];    // argv
extern char  argstring[128];    // copy of full cmd line
extern char *args;              // ptr to arg's portion of cmd line
#endif // EFI_APP || USERMODE

//windows GetExpression crashes on usermode
//and disallows compilation in lwwatch_usermode.cpp
#if defined(WIN32) && defined(USERMODE)
ULONG64 WinGetExpression(const char *str)
{
    char *eptr;
    ULONG64 ret;

    if( argstaken == nargs )
    {
        dprintf("flcndbg: No Argument supplied. Defaulting to 0.\n");
        return 0;
    }

    ret = strtoul(argv[argstaken],&eptr,16);
     if( *eptr != '\0' )
    {
        dprintf("flcndbg: Unknown or Invalid Argument '%s'. Defaulting to 0.\n", argv[argstaken]);
        ret = 0;
    }

    argstaken++;
    return ret;
}

#undef GetExpression
#define GetExpression WinGetExpression

BOOL WinGetExpressionEx( const char *expr, ULONG64 *val, const char **ignored )
{

    if( argstaken <= nargs -1 )                   // at least one arg can be taken.
    {
        *val = (ULONG64) GetExpression(expr);  // Get the first one.
        if( argstaken <= nargs - 1)   // one more arg left.
            return TRUE;
        return FALSE; // else
    }

    *val = 0;        // Zero arguments remain.
    return FALSE;
}

#undef GetExpressionEx
#define GetExpressionEx WinGetExpressionEx

#endif //USERMODE && WIN32
//
// Checks for NULL ptr's and empty before calling GetExpression
// prevents Windbg access violations where we call GetExpression without
// checking these conditions
//
ULONG64 GetSafeExpression(PCSTR lpExpression)
{
    if (!lpExpression)
        return 0;

    if (*lpExpression == '\0')
        return 0;

    return GetExpression(lpExpression);
}

BOOL GetSafeExpressionEx
(
    PCSTR Expression,
    ULONG64* Value,
    PCSTR* Remainder
)
{
    if (!Expression)
        return FALSE;

    if (*Expression == '\0')
        return FALSE;

    return GetExpressionEx(Expression, Value, Remainder) &&
           (Remainder != NULL && (*Remainder)[0] != '\0');
}

void SaveStateBeforeExtensionExec()
{
    if(IsSocBrdg())
        pSocbrdg[indexGpu].socbrdgSaveState();
}

void RestoreStateAfterExtensionExec()
{
    if(IsSocBrdg())
        pSocbrdg[indexGpu].socbrdgRestoreState();
}

extern LwU64 multiGpuBar0[8];
extern LwU64 multiGpuBar1[8];

//-----------------------------------------------------
// init [lwBar0] [lwBar1]
// - If lwBar0 is 0 or not specified, we will search
//   for LW devices.
//-----------------------------------------------------
DECLARE_API( init )
{
    LwU64 bar0;
    usingMods = 0;
    lwMode = MODE_LIVE;

#if defined(USERMODE) && defined(LW_WINDOWS)
    args = userargs;
#endif

#ifdef CLIENT_SIDE_RESMAN
    if(lwBar0 != 0)
    {
        dprintf("initLwWatch() was already called once.\n");
        return;
    }
#endif

    memset(&multiGpuBar0, 0x0, sizeof(multiGpuBar0));
    memset(&multiGpuBar1, 0x0, sizeof(multiGpuBar1));

#if LWWATCHCFG_IS_PLATFORM(OSX) && !defined(USERMODE)

    LwU64 bar0Size, bar0Virt, bar1, bar1Size, bar1Virt;

    // set the lwaddr - lwBar0 is global
    lwBar0 = GetSafeExpression(args);

    //
    // We can take either an index into the lw_devices array, or specify the address/size
    // of BAR0 and BAR1 manually.
    //
    if (GetSafeExpressionEx(args, &bar0, &args))
    {
        lwBar0 = bar0;

        if (!GetSafeExpressionEx(args, &bar0Size, &args))
            goto init_bad_usage;
        if (!GetSafeExpressionEx(args, &bar0Virt, &args))
            goto init_bad_usage;
        if (!GetSafeExpressionEx(args, &bar1, &args))
            goto init_bad_usage;
        if (!GetSafeExpressionEx(args, &bar1Size, &args))
            goto init_bad_usage;
        bar1Virt = GetSafeExpression(args);

        InitBarAddresses(lwBar0, bar0Size, bar0Virt, bar1, bar1Size, bar1Virt);
    }

    initLwWatch();
    dprintf("\n");

    return;

init_bad_usage:
    dprintf("lw: Usage: lw_init <GPU index> OR <Bar0Phys> <Bar0Size> <Bar0Virt> <Bar1Phys> <Bar1Size> <Bar1Virt>\n");
    return;

#elif !defined(EFI_APP)

    // set the lwaddr - lwBar0 is global
    lwBar0 = (PhysAddr) GetSafeExpression(args);

    if (GetSafeExpressionEx(args, &bar0, &args))
    {
        lwBar1 = (PhysAddr)GetSafeExpression(args);
        lwBar0 = (PhysAddr)bar0;
    }

    if (!LWWATCHCFG_IS_PLATFORM(OSX))
    {
        if (lwBar0 <= 0xff)
            lwBar0 <<= 24;

        if (lwBar1 <= 0xff)
            lwBar1 <<= 24;
    }

    initLwWatch();
    dprintf("\n");

#else // !defined(EFI_APP)

    initLwWatch();

    // set the lwaddr - lwBar0 is global
    lwBar0 = GetBar0Address();
#endif
}

//-----------------------------------------------------
// dispParseArgs
// - Yet another arguments packer
// Rationale: display commands take arguments in a different
//            way from the old commands, and the existing
//            functions do not nicely fit into our needs.
//
// XXX: move this up out of MAC_OS ifdef once it's used
//      by other functions.. it's within this ifdef
//      to remove "defined-but-not-used" warnings..
//-----------------------------------------------------
#define DISP_MAXARGV (44)
#define DISP_MAXTOK (512)

static int dArgc;
static char dArgv[DISP_MAXARGV][DISP_MAXTOK];

static void dispParseArgs(char *line);
#define PA_DBG (0)
static void dispParseArgs(char *line)
{
    char c, buf[DISP_MAXTOK], *pBuf;

    dArgc = 0;
    if (!line || *line == '\0') {
        return;
    }

    if (PA_DBG) {
        dprintf("::%s::\n", line);
    }

    // skip whitespaces
    while (*line == ' ' || *line == '\t')
        line++;
    // copy arguments until it hits the end of string
    pBuf = buf;
    while (dArgc < DISP_MAXARGV) {
        c = *line;
        // assume these are the only delimeters
        if (c != ' ' && c != '\t' && c != '\0') {
            *pBuf++ = c;
            line++;
        }else {
            *pBuf = '\0';
            strncpy(dArgv[dArgc], buf, DISP_MAXTOK);
            dArgv[dArgc][DISP_MAXTOK-1] = '\0';
            dArgc++;
            while (*line == ' ' || *line == '\t')
                line++;

            if (*line == '\0')
                break;
            pBuf = buf;
        }
    }

    // DEBUG
    if (PA_DBG) {
        int i;
        dprintf("dArgc = %d" , dArgc);
        for (i = 0; i < dArgc ; i++)
            dprintf(", dArgv[%d] = %s",i,dArgv[i]);
        dprintf("\n");
    }
}

//-----------------------------------------------------
// Falcon Debugger
//-----------------------------------------------------
DECLARE_API( loadsym )
{
    char* sessionName;
    char* exePath;

#if LWWATCHCFG_IS_PLATFORM(UNIX_MMAP) || LWWATCHCFG_IS_PLATFORM(OSX)

    dprintf("Error: FLCNDBG not supported on this platform\n");

#else

    // Lwrrently flcndbg supported only on lwwatch on mods and WinDbg
    dispParseArgs((char *) args);

    if (dArgc < 2)
    {
        dprintf("flcndbg: Usage: !flcndbg.loadsym <sessionID> [exelwtable full file path\n");
        dprintf("\t-symbol file full path: no whilespace should be used\n");
        dprintf("\t-A new session will be created for new sessionIDs\n");
        return ;
    }

    sessionName = dArgv[0];
    exePath = dArgv[1];

    flcngdbMenu(sessionName, exePath);

#endif
}
