/* This file was procedurally generated!  Do not modify this file by hand.  */

/*
* Copyright 2009-2016  LWPU Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to LWPU ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to LWPU and is being provided under the terms and conditions
* of a form of LWPU software license agreement.
*
* LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

#ifndef LWTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by lwToolsExt.h (except when LWTX_NO_IMPL is defined).
#endif

/* ---- Platform-independent helper definitions and functions ---- */

/* Prefer macros over inline functions to reduce symbol resolution at link time */

#if defined(_WIN32) 
#define LWTX_PATHCHAR   wchar_t
#define LWTX_STR(x)     L##x
#define LWTX_GETELW     _wgetelw
#define LWTX_BUFSIZE    MAX_PATH
#define LWTX_DLLHANDLE  HMODULE
#define LWTX_DLLOPEN(x) LoadLibraryW(x)
#define LWTX_DLLFUNC    GetProcAddress
#define LWTX_DLLCLOSE   FreeLibrary
#define LWTX_YIELD()    SwitchToThread()
#define LWTX_MEMBAR()   MemoryBarrier()
#define LWTX_ATOMIC_WRITE_32(address, value)                        InterlockedExchange((volatile LONG*)address, value)
#define LWTX_ATOMIC_CAS_32(old, address, exchange, comparand) old = InterlockedCompareExchange((volatile LONG*)address, exchange, comparand)
#elif defined(__GNUC__)
#define LWTX_PATHCHAR   char
#define LWTX_STR(x)     x
#define LWTX_GETELW     getelw
#define LWTX_BUFSIZE    PATH_MAX
#define LWTX_DLLHANDLE  void*
#define LWTX_DLLOPEN(x) dlopen(x, RTLD_LAZY)
#define LWTX_DLLFUNC    dlsym
#define LWTX_DLLCLOSE   dlclose
#define LWTX_YIELD()    sched_yield()
#define LWTX_MEMBAR()   __sync_synchronize()
/* Ensure full memory barrier for atomics, to match Windows functions */
#define LWTX_ATOMIC_WRITE_32(address, value)                  __sync_synchronize();       __sync_lock_test_and_set(address, value)
#define LWTX_ATOMIC_CAS_32(old, address, exchange, comparand) __sync_synchronize(); old = __sync_val_compare_and_swap(address, exchange, comparand)
#else
#error The library does not support your configuration!
#endif

/* Define this to 1 for platforms that where pre-injected libraries can be discovered. */
#if defined(_WIN32)
/* TODO */
#define LWTX_SUPPORT_ALREADY_INJECTED_LIBRARY 0
#else
#define LWTX_SUPPORT_ALREADY_INJECTED_LIBRARY 0
#endif

/* Define this to 1 for platforms that support environment variables */
/* TODO: Detect UWP, a.k.a. Windows Store app, and set this to 0. */
/* Try:  #if defined(WINAPI_FAMILY_PARTITION) && WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) */
#define LWTX_SUPPORT_ELW_VARS 1

/* Define this to 1 for platforms that support dynamic/shared libraries */
#define LWTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY 1

/* Injection libraries implementing InitializeInjectionLwtx2 may be statically linked,
*  and this will override any dynamic injection.  Useful for platforms where dynamic
*  injection is not available.  Since weak symbols not explicitly marked extern are
*  guaranteed to be initialized to zero if no definitions are found by the linker, the
*  dynamic injection process proceeds normally if pfnInitializeInjectionLwtx2 is 0. */
#if defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN__)
#define LWTX_SUPPORT_STATIC_INJECTION_LIBRARY 1
/* To statically inject an LWTX library, define InitializeInjectionLwtx2_fnptr as a normal
*  symbol (not weak) pointing to the implementation of InitializeInjectionLwtx2 (which
*  does not need to be named "InitializeInjectionLwtx2" as is necessary in a dynamic
*  injection library. */
__attribute__((weak)) LwtxInitializeInjectionLwtxFunc_t InitializeInjectionLwtx2_fnptr;
#else
#define LWTX_SUPPORT_STATIC_INJECTION_LIBRARY 0
#endif

/* This function tries to find or load an LWTX injection library and get the
*  address of its InitializeInjection2 function.  If such a function pointer
*  is found, it is called, and passed the address of this LWTX instance's
*  lwtxGetExportTable function, so the injection can attach to this instance.
*  If the initialization fails for any reason, any dynamic library loaded will
*  be freed, and all LWTX implementation functions will be set to no-ops.  If
*  initialization succeeds, LWTX functions not attached to the tool will be set
*  to no-ops.  This is implemented as one function instead of several small
*  functions to minimize the number of weak symbols the linker must resolve.
*  Order of search is:
*  - Pre-injected library exporting InitializeInjectionLwtx2
*  - Loadable library exporting InitializeInjectionLwtx2
*      - Path specified by elw var LWTX_INJECTION??_PATH (?? is 32 or 64)
*      - On Android, libLwtxInjection??.so within the package (?? is 32 or 64)
*  - Statically-linked injection library defining InitializeInjectionLwtx2_fnptr
*/
LWTX_LINKONCE_FWDDECL_FUNCTION int LWTX_VERSIONED_IDENTIFIER(lwtxInitializeInjectionLibrary)(void);
LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_VERSIONED_IDENTIFIER(lwtxInitializeInjectionLibrary)(void)
{
    const char* const initFuncName = "InitializeInjectionLwtx2";
    LwtxInitializeInjectionLwtxFunc_t init_fnptr = (LwtxInitializeInjectionLwtxFunc_t)0;
    LWTX_DLLHANDLE injectionLibraryHandle = (LWTX_DLLHANDLE)0;
    int entryPointStatus = 0;

#if LWTX_SUPPORT_ALREADY_INJECTED_LIBRARY
    /* Use POSIX global symbol chain to query for init function from any module */
    init_fnptr = (LwtxInitializeInjectionLwtxFunc_t)LWTX_DLLFUNC(0, initFuncName);
#endif

#if LWTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY
    /* Try discovering dynamic injection library to load */
    if (!init_fnptr)
    {
#if LWTX_SUPPORT_ELW_VARS
        /* If elw var LWTX_INJECTION64_PATH is set, it should contain the path
        *  to a 64-bit dynamic LWTX injection library (and similar for 32-bit). */
        const LWTX_PATHCHAR* const lwtxElwVarName = (sizeof(void*) == 4)
            ? LWTX_STR("LWTX_INJECTION32_PATH")
            : LWTX_STR("LWTX_INJECTION64_PATH");
#endif /* LWTX_SUPPORT_ELW_VARS */
        LWTX_PATHCHAR injectionLibraryPathBuf[LWTX_BUFSIZE];
        const LWTX_PATHCHAR* injectionLibraryPath = (const LWTX_PATHCHAR*)0;

        /* Refer to this variable explicitly in case all references to it are #if'ed out */
        (void)injectionLibraryPathBuf;

#if LWTX_SUPPORT_ELW_VARS
        /* Disable the warning for getelw & _wgetelw -- this usage is safe because
        *  these functions are not called again before using the returned value. */
#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif
        injectionLibraryPath = LWTX_GETELW(lwtxElwVarName);
#if defined(_MSC_VER)
#pragma warning( pop )
#endif
#endif

#if defined(__ANDROID__)
        if (!injectionLibraryPath)
        {
            const char *bits = (sizeof(void*) == 4) ? "32" : "64";
            char cmdlineBuf[32];
            char pkgName[PATH_MAX];
            int count;
            int pid;
            FILE *fp;
            size_t bytesRead;
            size_t pos;

            pid = (int)getpid();
            count = snprintf(cmdlineBuf, sizeof(cmdlineBuf), "/proc/%d/cmdline", pid);
            if (count <= 0 || count >= (int)sizeof(cmdlineBuf))
            {
                LWTX_ERR("Path buffer too small for: /proc/%d/cmdline\n", pid);
                return LWTX_ERR_INIT_ACCESS_LIBRARY;
            }

            fp = fopen(cmdlineBuf, "r");
            if (!fp)
            {
                LWTX_ERR("File couldn't be opened: %s\n", cmdlineBuf);
                return LWTX_ERR_INIT_ACCESS_LIBRARY;
            }

            bytesRead = fread(pkgName, 1, sizeof(pkgName) - 1, fp);
            fclose(fp);
            if (bytesRead == 0)
            {
                LWTX_ERR("Package name couldn't be read from file: %s\n", cmdlineBuf);
                return LWTX_ERR_INIT_ACCESS_LIBRARY;
            }

            pkgName[bytesRead] = 0;

            /* String can contain colon as a process separator. In this case the package name is before the colon. */
            pos = 0;
            while (pos < bytesRead && pkgName[pos] != ':' && pkgName[pos] != '\0')
            {
                ++pos;
            }
            pkgName[pos] = 0;

            count = snprintf(injectionLibraryPathBuf, LWTX_BUFSIZE, "/data/data/%s/files/libLwtxInjection%s.so", pkgName, bits);
            if (count <= 0 || count >= LWTX_BUFSIZE)
            {
                LWTX_ERR("Path buffer too small for: /data/data/%s/files/libLwtxInjection%s.so\n", pkgName, bits);
                return LWTX_ERR_INIT_ACCESS_LIBRARY;
            }

            /* On Android, verify path is accessible due to aggressive file access restrictions. */
            /* For dlopen, if the filename contains a leading slash, then it is interpreted as a */
            /* relative or absolute pathname; otherwise it will follow the rules in ld.so. */
            if (injectionLibraryPathBuf[0] == '/')
            {
#if (__ANDROID_API__ < 21)
                int access_err = access(injectionLibraryPathBuf, F_OK | R_OK);
#else
                int access_err = faccessat(AT_FDCWD, injectionLibraryPathBuf, F_OK | R_OK, 0);
#endif
                if (access_err != 0)
                {
                    LWTX_ERR("Injection library path wasn't accessible [code=%s] [path=%s]\n", strerror(errno), injectionLibraryPathBuf);
                    return LWTX_ERR_INIT_ACCESS_LIBRARY;
                }
            }
            injectionLibraryPath = injectionLibraryPathBuf;
        }
#endif

        /* At this point, injectionLibraryPath is specified if a dynamic
        *  injection library was specified by a tool. */
        if (injectionLibraryPath)
        {
            /* Load the injection library */
            injectionLibraryHandle = LWTX_DLLOPEN(injectionLibraryPath);
            if (!injectionLibraryHandle)
            {
                LWTX_ERR("Failed to load injection library\n");
                return LWTX_ERR_INIT_LOAD_LIBRARY;
            }
            else
            {
                /* Attempt to get the injection library's entry-point */
                init_fnptr = (LwtxInitializeInjectionLwtxFunc_t)LWTX_DLLFUNC(injectionLibraryHandle, initFuncName);
                if (!init_fnptr)
                {
                    LWTX_DLLCLOSE(injectionLibraryHandle);
                    LWTX_ERR("Failed to get address of function InitializeInjectionLwtx2 from injection library\n");
                    return LWTX_ERR_INIT_MISSING_LIBRARY_ENTRY_POINT;
                }
            }
        }
    }
#endif

#if LWTX_SUPPORT_STATIC_INJECTION_LIBRARY
    if (!init_fnptr)
    {
        /* Check weakly-defined function pointer.  A statically-linked injection can define this as
        *  a normal symbol and it will take precedence over a dynamic injection. */
        if (InitializeInjectionLwtx2_fnptr)
        {
            init_fnptr = InitializeInjectionLwtx2_fnptr;
        }
    }
#endif

    /* At this point, if init_fnptr is not set, then no tool has specified
    *  an LWTX injection library -- return non-success result so all LWTX
    *  API functions will be set to no-ops. */
    if (!init_fnptr)
    {
        return LWTX_ERR_NO_INJECTION_LIBRARY_AVAILABLE;
    }

    /* Ilwoke injection library's initialization function.  If it returns
    *  0 (failure) and a dynamic injection was loaded, unload it. */
    entryPointStatus = init_fnptr(LWTX_VERSIONED_IDENTIFIER(lwtxGetExportTable));
    if (entryPointStatus == 0)
    {
        LWTX_ERR("Failed to initialize injection library -- initialization function returned 0\n");
        if (injectionLibraryHandle)
        {
            LWTX_DLLCLOSE(injectionLibraryHandle);
        }
        return LWTX_ERR_INIT_FAILED_LIBRARY_ENTRY_POINT;
    }

    return LWTX_SUCCESS;
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)(void)
{
    unsigned int old;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).initState == LWTX_INIT_STATE_COMPLETE)
    {
        return;
    }

    LWTX_ATOMIC_CAS_32(
        old,
        &LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).initState,
        LWTX_INIT_STATE_STARTED,
        LWTX_INIT_STATE_FRESH);
    if (old == LWTX_INIT_STATE_FRESH)
    {
        int result;
        int forceAllToNoops;

        /* Load & initialize injection library -- it will assign the function pointers */
        result = LWTX_VERSIONED_IDENTIFIER(lwtxInitializeInjectionLibrary)();

        /* Set all pointers not assigned by the injection to null */
        forceAllToNoops = result != LWTX_SUCCESS; /* Set all to null if injection init failed */
        LWTX_VERSIONED_IDENTIFIER(lwtxSetInitFunctionsToNoops)(forceAllToNoops);

        /* Signal that initialization has finished, so now the assigned function pointers will be used */
        LWTX_ATOMIC_WRITE_32(
            &LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).initState,
            LWTX_INIT_STATE_COMPLETE);
    }
    else /* Spin-wait until initialization has finished */
    {
        LWTX_MEMBAR();
        while (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).initState != LWTX_INIT_STATE_COMPLETE)
        {
            LWTX_YIELD();
            LWTX_MEMBAR();
        }
    }
}
