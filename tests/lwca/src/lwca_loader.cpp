#include <g_lwconfig.h>
#include <lwos.h>
#include <lwos_platform_os.h>

#if lwosOsIsWindows()
#include <Windows.h>
#include <d3d11.h>
#include <d3d10.h>
#include <d3d9.h>
#define LOADERAPI __stdcall
#else
#include <dlfcn.h>
#define LOADERAPI
#endif

#define __LWDA_API_VERSION_INTERNAL
#include <lwca.h>
#include <lwdaProfiler.h>

#if lwosOsIsWindows()
#include <lwdaD3D11.h>
#include <lwdaD3D10.h>
#include <lwdaD3D9.h>
#endif

// To work around a bug in parsing non-lwca.h generated headers
#undef LWDAAPI
#define LWDAAPI

// TODO: Add more headers here for GL, EGL, VDPAU, etc when the corresponding
// loaders are implemented

#define __lwda_fun__(name, ret_type, arg_types, args) \
typedef ret_type (LOADERAPI *name##_t)arg_types;      \
static name##_t __fun_##name = 0;                     \
ret_type LOADERAPI name arg_types                     \
{                                                     \
    if ( __fun_##name) {                              \
        return (*__fun_##name)args;                   \
    }                                                 \
    return LWDA_ERROR_UNKNOWN;                        \
}                                                     \

#define SKIP_LW_INIT
#include <generated_types_lwda.h>
#include <generated_types_lwdaProfiler.h>

#if lwosOsIsWindows()
#include <generated_types_lwdaD3D11.h>
#include <generated_types_lwdaD3D10.h>
#include <generated_types_lwdaD3D9.h>
#endif

#undef SKIP_LW_INIT

#undef __lwda_fun__

typedef LWresult (LOADERAPI *lwInit_t)(int a);
static lwInit_t __fun_lwInit = 0;

#if lwosOsIsApple()
#define LWDA_NAME "LWCA.framework/LWCA"
#elif lwosOsIsWindows()
#define LWDA_NAME "lwlwda.dll"
#else
#define LWDA_NAME "liblwda.so.1"
#endif

static void load_lwda(void)
{
    LWOSLibrary lib = lwosLoadLibraryUnsafe(LWDA_NAME);
// TODO: Eventually augment with lwGetProcAddress
#define __lwda_fun__(name, ret_type, arg_types, args) __fun_##name = (name##_t)lwosGetProcAddress(lib, #name);
#include <generated_types_lwda.h>
#include <generated_types_lwdaProfiler.h>
#if lwosOsIsWindows()
#include <generated_types_lwdaD3D11.h>
#include <generated_types_lwdaD3D10.h>
#include <generated_types_lwdaD3D9.h>
#endif
#undef __lwda_fun__
}

static lwosOnceControl load_lwda_control = LWOS_ONCE_INIT;

LWresult LOADERAPI lwInit(unsigned int Flags)
{
    lwosOnce(&load_lwda_control, load_lwda);
    if (__fun_lwInit) {
        return (*__fun_lwInit)(Flags);
    }
    return LWDA_ERROR_UNKNOWN;
}
