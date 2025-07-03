//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// lwwatch.h
//
//*****************************************************

#ifndef _LWWATCH_H_
#define _LWWATCH_H_

//
// Define the LwWatch version (Must be at least 2.0 as non-version extension is considered 1.0)
//
#define LWWATCH_MAJOR_VERSION   2
#define LWWATCH_MINOR_VERSION   0

//
// includes
//

#include "lwwatch-config.h"

#if defined(LW_MODS)
// Avoid clashes between dprintf function from stdio.h and the macro
// defined by lwwatch by including stdio.h early.
// Otherwise clang detects dprintf redefinition and exits with an error.
#include <stdio.h>
#endif

#if LWWATCHCFG_IS_PLATFORM(WINDOWS)
#include <windows.h>

//
// Comment copied from sample extension:
// Define KDEXT_64BIT to make all wdbgexts APIs recognize 64 bit addresses. It
// is recommended for extensions to use 64 bit headers from wdbgexts so the
// extensions could support 64 bit targets.
//
#if !LWWATCHCFG_FEATURE_ENABLED(WINDOWS_STANDALONE)
#define KDEXT_64BIT
#include <wdbgexts.h>
#include <dbgeng.h>
#endif // WINDOWS_STANDALONE
#undef dprintf
#ifdef __cplusplus
extern "C" {
#endif
int dprintf(const char* pszFormat, ...);
void dbgprintf(const char* pszFormat, ...);
int outputFlush(int OutputFlush, ULONG timeoutMs);
#ifdef __cplusplus
}
#endif
#endif // LWWATCHCFG_IS_PLATFORM(WINDOWS)

#endif // _LWWATCH_H_

