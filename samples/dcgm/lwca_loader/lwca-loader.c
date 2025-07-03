#include <assert.h>
#include <lwca.h>
#include <lwos.h>
#include <string.h>
#include "lwca-loader.h"
#include "lwca-hook.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Import Table
// ----------------------------------------
// The following declares and defines an import table for all dynamically loaded LWCA API's. 

#define BEGIN_ENTRYPOINTS                                                               \
    typedef struct itblLwdaLibary_st                                                    \
    {
#define END_ENTRYPOINTS                                                                 \
    } itblLwdaLibrary;
#define LWDA_API_ENTRYPOINT(lwdaFuncname, entrypointApiFuncname, argtypes, fmt, ...)    \
    LWresult (LWDAAPI *lwdaFuncname) argtypes;

#include "lwca-entrypoints.h"

#undef LWDA_API_ENTRYPOINT
#undef END_ENTRYPOINTS
#undef BEGIN_ENTRYPOINTS

// The import table of the dynamically loaded LWCA library
static itblLwdaLibrary g_itblDefaultLwdaLibrary;

// The import table consisting of hooks that are to be called instead of the API's in the LWCA library
static itblLwdaLibrary g_itblLwdaLibrary;
static LWOSLibrary g_lwdaLibrary = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Stubs
// ----------------------------------------
// The following defines function stubs for each of the LWCA API's that can be dynamically loaded
// at runtime. These stubs allow programs to use a provided unmodified <lwca.h> within their programs
// and link against this static library to dynamically link with the LWCA library at runtime. Each 
// function stub forwards the call to their LWCA API equivalent.

#define BEGIN_ENTRYPOINTS
#define END_ENTRYPOINTS
#define LWDA_API_ENTRYPOINT(lwdaFuncname, entrypointApiFuncname, argtypes, fmt, ...)                \
    LWresult LWDAAPI lwdaFuncname argtypes                                                          \
    {                                                                                               \
        if (NULL != g_itblLwdaLibrary.lwdaFuncname)                                                 \
            return (*(g_itblLwdaLibrary.lwdaFuncname))(__VA_ARGS__);                                \
        return (*(g_itblDefaultLwdaLibrary.lwdaFuncname))(__VA_ARGS__);                             \
    }                                                                                               \
    void set_ ## lwdaFuncname ## Hook (lwdaFuncname ## _loader_t lwdaFuncHook)                      \
    {                                                                                               \
        g_itblLwdaLibrary.lwdaFuncname = lwdaFuncHook;                                              \
    }                                                                                               \
    void reset_ ## lwdaFuncname ## Hook (void)                                                      \
    {                                                                                               \
        g_itblLwdaLibrary.lwdaFuncname = NULL;                                                      \
    }

#include "lwca-entrypoints.h"

#undef LWDA_API_ENTRYPOINT
#undef END_ENTRYPOINTS
#undef BEGIN_ENTRYPOINTS

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pre-emptive API Loading
// ----------------------------------------
// The following defines a function which Dynamically loads all LWCA API's into the import table.

#define BEGIN_ENTRYPOINTS                                                                           \
    static lwdaLibraryLoadResult_t loadLwdaLibraryApis (void)                                       \
    {                                                                                               \
        assert(g_lwdaLibrary);
#define LWDA_API_ENTRYPOINT(lwdaFuncname, entrypointApiFuncname, argtypes, fmt, ...)                \
        assert(NULL == g_itblDefaultLwdaLibrary.lwdaFuncname);                                      \
        g_itblDefaultLwdaLibrary.lwdaFuncname = (lwdaFuncname ## _loader_t)lwosGetProcAddress(      \
            g_lwdaLibrary, #entrypointApiFuncname);                                                 \
        if (NULL == g_itblDefaultLwdaLibrary.lwdaFuncname)                                          \
        {                                                                                           \
            return LWDA_LIBRARY_ERROR_API_NOT_FOUND;                                                \
        } 
#define END_ENTRYPOINTS                                                                             \
        return LWDA_LIBRARY_LOAD_SUCCESS;                                                           \
    }

#include "lwca-entrypoints.h"

#undef END_ENTRYPOINTS
#undef LWDA_API_ENTRYPOINT
#undef BEGIN_ENTRYPOINTS

/**
 * Attempts to dynamically load the LWCA library from typical locations where the LWCA
 * library is found.
 * 
 * @return
 *         - LWDA_LIBRARY_LOAD_SUCCESS      If the LWCA library was found and successfully loaded
 *         - LWDA_ERROR_LIBRARY_NOT_FOUND   If the LWCA library was not found or could not be loaded
 */
lwdaLibraryLoadResult_t loadDefaultLwdaLibrary (void)
{
    if(g_lwdaLibrary)
        return LWDA_LIBRARY_LOAD_SUCCESS;

    assert(sizeof(g_itblDefaultLwdaLibrary) == sizeof(g_itblLwdaLibrary));
    assert(sizeof(g_itblLwdaLibrary) == sizeof(itblLwdaLibrary));
    memset(&g_itblDefaultLwdaLibrary, 0, sizeof(g_itblDefaultLwdaLibrary));
    memset(&g_itblLwdaLibrary, 0, sizeof(g_itblLwdaLibrary));
#ifdef _UNIX

    g_lwdaLibrary = lwosLoadLibrary("liblwda.so.1");
    if(g_lwdaLibrary)
        return loadLwdaLibraryApis();

    //
    // for Ubuntu we support /usr/lib{,32,64}/lwpu-current/...
    // However, we add these paths to ldconfig so this is not needed
    // If user messes with ldconfig after the installer sets things up it's up to them to fix apps
    //

    // For x64 .run installs
    g_lwdaLibrary = lwosLoadLibrary("/usr/lib64/liblwda.so.1");
    if(g_lwdaLibrary)
        return loadLwdaLibraryApis();

    // For RPM fusion x64
    g_lwdaLibrary = lwosLoadLibrary("/usr/lib64/lwpu/liblwda.so.1");
    if(g_lwdaLibrary)
        return loadLwdaLibraryApis();

    // For some 32 and 64 bit installs
    g_lwdaLibrary = lwosLoadLibrary("/usr/lib/liblwda.so.1");
    if(g_lwdaLibrary)
        return loadLwdaLibraryApis();

    // For some 32 bit installs
    g_lwdaLibrary = lwosLoadLibrary("/usr/lib32/liblwda.so.1");
    if(g_lwdaLibrary)
        return loadLwdaLibraryApis();

    // For RPM Fusion 32 bit
    g_lwdaLibrary = lwosLoadLibrary("/usr/lib/lwpu/liblwda.so.1");
    if(g_lwdaLibrary)
        return loadLwdaLibraryApis();

#endif /* _UNIX */
#ifdef _WINDOWS
    // Load from the system directory (a trusted location)
    g_lwdaLibrary = lwosLoadLibrary("lwlwda.dll");
    if(g_lwdaLibrary)
        return loadLwdaLibraryApis();
#endif /* _WINDOWS */

    return LWDA_LIBRARY_ERROR_NOT_FOUND;
}

/**
 * Attempts to dynamically unload the LWCA library.
 *
 * @return
 *         - LWDA_LIBRARY_LOAD_SUCCESS          If the LWCA library was successfully unloaded
 *         - LWDA_ERROR_LIBRARY_UNLOAD_FAILED   If the LWCA library could not be unloaded
 */
lwdaLibraryLoadResult_t unloadLwdaLibrary (void)
{
    memset(&g_itblDefaultLwdaLibrary, 0, sizeof(itblLwdaLibrary));
    if(!lwosFreeLibrary(g_lwdaLibrary))
        return LWDA_LIBRARY_ERROR_UNLOAD_FAILED;
    g_lwdaLibrary = 0;
    return LWDA_LIBRARY_LOAD_SUCCESS;
}

void resetAllLwdaHooks (void)
{
    memset(&g_itblLwdaLibrary, 0, sizeof(itblLwdaLibrary));
}
