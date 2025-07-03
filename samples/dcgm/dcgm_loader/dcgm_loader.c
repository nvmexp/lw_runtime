#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"
#include "dcgm_loader_hook.h"
#include "lwos.h"
#include "lwml_lwos_wrapper.h"
#include "spinlock.h"
#include "dcgm_structs.h"

static LWOSLibrary dcgmLib = 0;
static volatile unsigned int dcgmLibLock = 0;
static volatile unsigned int dcgmStaticLibResetHooksCount = 0;

// Not sure if this comment applies
//
//
// The following defines the hooking mechanism that calls the hooked function
// set by a user of this library. Insert this macro to enable an API to be hooked 
// if the hooked function is set. This was done because dcgmInit() is a special
// case which must be hookable before it attempts to dynamically load a library.
#define DCGM_API_HOOK(libFunctionName, ...)                                                      \
do                                                                                               \
{                                                                                                \
    if ((NULL != libFunctionName ## HookedFunc) &&                                               \
        (libFunctionName ## HookResetCount == dcgmStaticLibResetHooksCount))                     \
    {                                                                                            \
        dcgmReturn_t hookedFuncResult;                                                           \
        /* The number of times this hook was reset equals the number of times */                 \
        /* the static library was reset. So call the hooked function */                          \
        hookedFuncResult = (* libFunctionName ## HookedFunc)(__VA_ARGS__);                       \
        return hookedFuncResult;                                                                 \
    }                                                                                            \
} while(0)

#define DCGM_DYNAMIC_WRAP(newName, libFunctionName, argtypes, ...)                               \
static libFunctionName ## _loader_t libFunctionName ## DefaultFunc = NULL;                       \
static libFunctionName ## _loader_t libFunctionName ## HookedFunc = NULL;                        \
static volatile unsigned int libFunctionName ## HookResetCount = 0;                              \
void set_ ## libFunctionName ## Hook (libFunctionName ## _loader_t dcgmFuncHook)                 \
{                                                                                                \
    libFunctionName ## HookResetCount = dcgmStaticLibResetHooksCount;                            \
    libFunctionName ## HookedFunc = dcgmFuncHook;                                                \
}                                                                                                \
void reset_ ## libFunctionName ## Hook (void)                                                    \
{                                                                                                \
    libFunctionName ## HookedFunc = NULL;                                                        \
}                                                                                                \
dcgmReturn_t newName argtypes                                                                    \
{                                                                                                \
    static volatile int isLookupDone = 0;                                                        \
    DCGM_API_HOOK(libFunctionName, ##__VA_ARGS__);                                               \
                                                                                                 \
    if (!dcgmLib)                                                                                \
        return DCGM_ST_UNINITIALIZED;                                                         \
                                                                                                 \
    if (!isLookupDone)                                                                           \
    {                                                                                            \
        static volatile unsigned int initLock = 0;                                               \
        lwmlSpinLock(&initLock);                                                                 \
        if (!isLookupDone)                                                                       \
        {                                                                                        \
            libFunctionName ## DefaultFunc = (libFunctionName ## _loader_t)lwosGetProcAddress(   \
                dcgmLib, #libFunctionName);                                                      \
            isLookupDone = 1;                                                                    \
        }                                                                                        \
        lwmlUnlock(&initLock);                                                                   \
    }                                                                                            \
                                                                                                 \
    if (!libFunctionName ## DefaultFunc)                                                         \
        return DCGM_ST_GENERIC_ERROR;                                                            \
                                                                                                 \
    return (*libFunctionName ## DefaultFunc)(__VA_ARGS__);                                       \
}

#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)                        \
    DCGM_DYNAMIC_WRAP(dcgmFuncname, dcgmFuncname, argtypes, ##__VA_ARGS__)
#define DCGM_INT_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)
#include "entry_point.h"
#undef DCGM_INT_ENTRY_POINT
#undef DCGM_ENTRY_POINT

dcgmReturn_t dcgmLoadDefaultSharedLibrary(void)
{
#ifdef _WINDOWS
    dcgmReturn_t dcgmRet;
    char path[1024];
#endif

    if (dcgmLib)
        return DCGM_ST_GENERIC_ERROR;

    lwmlSpinLock(&dcgmLibLock);

#ifdef _UNIX
    dcgmLib = lwosLoadLibrary("libdcgm.so.1");
    if (dcgmLib)
        goto great_success;
    
    //
    // for Ubuntu we support /usr/lib{,32,64}/lwpu-current/...
    // However, we add these paths to ldconfig so this is not needed
    // If user messes with ldconfig after the installer sets things up it's up to them to fix apps
    //
    
    // For x64 .run and other installs
    dcgmLib = lwosLoadLibrary("/usr/lib64/libdcgm.so.1");
    if (dcgmLib)
        goto great_success;
        
    // For RPM Fusion (64 bit)
    dcgmLib = lwosLoadLibrary("/usr/lib64/lwpu/libdcgm.so.1");
    if (dcgmLib)
        goto great_success;

    // For some 32 bit and some 64 bit .run and other installs
    dcgmLib = lwosLoadLibrary("/usr/lib/libdcgm.so.1");
    if (dcgmLib)
        goto great_success;

    // For some 32 bit .run and other installs
    dcgmLib = lwosLoadLibrary("/usr/lib32/libdcgm.so.1");
    if (dcgmLib)
        goto great_success;
        
    // For RPM Fusion (32 bit)
    dcgmLib = lwosLoadLibrary("/usr/lib/lwpu/libdcgm.so.1");
    if (dcgmLib)
        goto great_success;

    //For Ubuntu on PPC and x64
#if defined(__powerpc64__)
    dcgmLib = lwosLoadLibrary("/usr/lib/powerpc64le-linux-gnu/libdcgm.so.1");
    if (dcgmLib)
        goto great_success;
#else //X64
    dcgmLib = lwosLoadLibrary("/usr/lib/x86_64-linux-gnu/libdcgm.so.1");
    if (dcgmLib)
        goto great_success;
#endif

    lwmlUnlock(&dcgmLibLock);
    return DCGM_ST_GENERIC_ERROR;

great_success:
    lwmlUnlock(&dcgmLibLock);
    return DCGM_ST_OK;
#endif // _UNIX

#ifdef _WINDOWS
    // it's possible to maliciously change this elwironmental variable
    // but then the load library wrapper will reject the path
    if (0 != lwosGetElw("ProgramW6432", path, sizeof(path)))
    {
        return DCGM_ST_GENERIC_ERROR;
    }
    // This is the only location we should look on Windows
    // todo: change this to the actual DCGM path on Windows when the time comes
    strncat(path, "\\LWPU Corporation\\LWSMI\\dcgm.dll", sizeof(path));
    dcgmRet = lwmlLwosLoadLibraryTrusted(path, &dcgmLib);
    lwmlUnlock(&dcgmLibLock);
    return dcgmRet;
#endif // _WINDOWS
}

static dcgmReturn_t localLwcmEngineInit(dcgmOperationMode_t mode);
DCGM_DYNAMIC_WRAP(localLwcmEngineInit, dcgmEngineInit, (dcgmOperationMode_t mode), mode)
dcgmReturn_t dcgmEngineInit(dcgmOperationMode_t mode)
{
    DCGM_API_HOOK(dcgmEngineInit, mode);

    if (!dcgmLib)
    {
        dcgmReturn_t result = dcgmLoadDefaultSharedLibrary();
        if (DCGM_ST_OK != result)
            return result;
    }

    return localLwcmEngineInit(mode);
}

DCGM_DYNAMIC_WRAP(dcgmEngineShutdown, dcgmEngineShutdown, (void))

DCGM_DYNAMIC_WRAP(dcgmInternalGetExportTable, dcgmInternalGetExportTable,
        (const void **ppExportTable, const dcgmUuid_t *pExportTableId),
        ppExportTable, pExportTableId)

typedef const char* (*dcgmErrorString_loader_t)(dcgmReturn_t result);
static const char* localLwcmErrorString(dcgmReturn_t result)
{
    static dcgmErrorString_loader_t func = NULL;
    static volatile int isLookupDone = 0;

    if (!dcgmLib)
        return NULL;

    if (!isLookupDone)
    {
        static volatile unsigned int initLock = 0;
        lwmlSpinLock(&initLock);
        if (!isLookupDone)
        {
            func = (dcgmErrorString_loader_t) lwosGetProcAddress(dcgmLib, "dcgmErrorString");
            isLookupDone = 1;
        }
        lwmlUnlock(&initLock);
    }

    if (!func)
        return NULL;

    return func(result);
}

const char* dcgmErrorString(dcgmReturn_t result)
{
    const char* str = errorString(result);

    if (!str)
        str = localLwcmErrorString(result);

    if (!str)
        str = "Unknown Error";

    return str;
}

/**
 * Notifies all hooks that they are to reset the next time their associated DCGM API is 
 * ilwoked using a global reset counter. API's compare their local counter to the global
 * count and reset their hooks if a mismatch is detected.
 */
void resetAllLwcmHooks (void)
{
    ++dcgmStaticLibResetHooksCount;
}
