/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#if !defined(__LWDA_INTERNAL_COMPILATION__)

#define __LWDA_INTERNAL_COMPILATION__
#define __text__
#define __surf__
#define __name__shadow_var(c, cpp) \
        #c
#define __name__text_var(c, cpp) \
        #cpp
#define __host__shadow_var(c, cpp) \
        cpp
#define __text_var(c, cpp) \
        cpp
#define __device_fun(fun) \
        #fun
#define __device_var(var) \
        #var
#define __device__text_var(c, cpp) \
        #c
#define __device__shadow_var(c, cpp) \
        #c

#if defined(_WIN32) && !defined(_WIN64)

#define __pad__(f) \
        f

#else /* _WIN32 && !_WIN64 */

#define __pad__(f)

#endif /* _WIN32 && !_WIN64 */

#include "builtin_types.h"
#include "storage_class.h"

#else /* !__LWDA_INTERNAL_COMPILATION__ */

template <typename T>
static inline T *__lwdaAddressOf(T &val) 
{
    return (T *)((void *)(&(const_cast<char &>(reinterpret_cast<const volatile char &>(val)))));
}

#define __lwdaRegisterBinary(X)                                                   \
        __lwdaFatLwbinHandle = __lwdaRegisterFatBinary((void*)&__fatDeviceText); \
        { void (*callback_fp)(void **) =  (void (*)(void **))(X); (*callback_fp)(__lwdaFatLwbinHandle); }\
        atexit(__lwdaUnregisterBinaryUtil)
        
#define __lwdaRegisterVariable(handle, var, ext, size, constant, global) \
        __lwdaRegisterVar(handle, (char*)&__host##var, (char*)__device##var, __name##var, ext, size, constant, global)
#define __lwdaRegisterManagedVariable(handle, var, ext, size, constant, global) \
        __lwdaRegisterManagedVar(handle, (void **)&__host##var, (char*)__device##var, __name##var, ext, size, constant, global)

#define __lwdaRegisterGlobalTexture(handle, tex, dim, norm, ext) \
        __lwdaRegisterTexture(handle, (const struct textureReference*)&tex, (const void**)(void*)__device##tex, __name##tex, dim, norm, ext)
#define __lwdaRegisterGlobalSurface(handle, surf, dim, ext) \
        __lwdaRegisterSurface(handle, (const struct surfaceReference*)&surf, (const void**)(void*)__device##surf, __name##surf, dim, ext)
#define __lwdaRegisterEntry(handle, funptr, fun, thread_limit) \
        __lwdaRegisterFunction(handle, (const char*)funptr, (char*)__device_fun(fun), #fun, -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0)
          
#define __lwdaSetupArg(arg, offset) \
        if (lwdaSetupArgument((void *)(__lwdaAddressOf(arg)), sizeof(arg), (size_t)offset) != lwdaSuccess) \
          return
          
#define __lwdaSetupArgSimple(arg, offset) \
        if (lwdaSetupArgument((void *)(char *)&arg, sizeof(arg), (size_t)offset) != lwdaSuccess) \
          return

#if defined(__GNUC__)
#define __lwdaLaunch(fun) \
        { volatile static char *__f __attribute__((unused)); __f = fun; (void)lwdaLaunch(fun); }
#else /* __GNUC__ */
#define __lwdaLaunch(fun) \
        { volatile static char *__f; __f = fun; (void)lwdaLaunch(fun); }
#endif /* __GNUC__ */

#if defined(__GNUC__)
#define __lw_dummy_param_ref(param) \
        { volatile static void **__ref __attribute__((unused)); __ref = (volatile void **)param; }
#else /* __GNUC__ */
#define __lw_dummy_param_ref(param) \
        { volatile static void **__ref; __ref = (volatile void **)param; }
#endif /* __GNUC__ */

static void ____lw_dummy_param_ref(void *param) __lw_dummy_param_ref(param)

#define __REGISTERFUNCNAME_CORE(X) __lwdaRegisterLinkedBinary##X
#define __REGISTERFUNCNAME(X) __REGISTERFUNCNAME_CORE(X)

extern "C" {
void __REGISTERFUNCNAME( __LW_MODULE_ID ) ( void (*)(void **), void *, void *, void (*)(void *));
}

#define __TO_STRING_CORE(X) #X
#define __TO_STRING(X) __TO_STRING_CORE(X)

extern "C" {
#if defined(_WIN32)
#pragma data_seg("__lw_module_id")
  static const __declspec(allocate("__lw_module_id")) unsigned char __module_id_str[] = __TO_STRING(__LW_MODULE_ID);
#pragma data_seg()
#elif defined(__APPLE__)
  static const unsigned char __module_id_str[] __attribute__((section ("__LW_LWDA,__lw_module_id"))) = __TO_STRING(__LW_MODULE_ID);
#else
  static const unsigned char __module_id_str[] __attribute__((section ("__lw_module_id"))) = __TO_STRING(__LW_MODULE_ID);
#endif

#undef __FATIDNAME_CORE
#undef __FATIDNAME
#define __FATIDNAME_CORE(X) __fatbinwrap##X
#define __FATIDNAME(X) __FATIDNAME_CORE(X)

#define  ____lwdaRegisterLinkedBinary(X) \
{ __REGISTERFUNCNAME(__LW_MODULE_ID) (( void (*)(void **))(X), (void *)&__FATIDNAME(__LW_MODULE_ID), (void *)&__module_id_str, (void (*)(void *))&____lw_dummy_param_ref); }

}

extern "C" {
extern void** LWDARTAPI __lwdaRegisterFatBinary(
  void *fatLwbin
);

extern void LWDARTAPI __lwdaUnregisterFatBinary(
  void **fatLwbinHandle
);

extern void LWDARTAPI __lwdaRegisterVar(
        void **fatLwbinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
);

extern void LWDARTAPI __lwdaRegisterManagedVar(
        void **fatLwbinHandle,
        void **hostVarPtrAddress,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
);

extern char LWDARTAPI __lwdaInitModule(
        void **fatLwbinHandle
);

extern void LWDARTAPI __lwdaRegisterTexture(
        void                    **fatLwbinHandle,
  const struct textureReference  *hostVar,
  const void                    **deviceAddress,
  const char                     *deviceName,
        int                       dim,       
        int                       norm,      
        int                        ext        
);

extern void LWDARTAPI __lwdaRegisterSurface(
        void                    **fatLwbinHandle,
  const struct surfaceReference  *hostVar,
  const void                    **deviceAddress,
  const char                     *deviceName,
        int                       dim,       
        int                       ext        
);

extern void LWDARTAPI __lwdaRegisterFunction(
        void   **fatLwbinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
);

#if defined(__APPLE__)
extern "C" int atexit(void (*)(void));

#elif  defined(__GNUC__) && !defined(__ANDROID__) && !defined(__HORIZON__)
extern int atexit(void(*)(void)) throw();

#elif defined(__HORIZON__)

// __TEMP_WAR__ 200132570 HOS : Disable atexit call until it works
#define atexit(p)

#else /* __GNUC__ && !__ANDROID__ */
extern int __cdecl atexit(void(__cdecl *)(void));
#endif

}

static void **__lwdaFatLwbinHandle;

static void __cdecl __lwdaUnregisterBinaryUtil(void)
{
  ____lw_dummy_param_ref((void *)&__lwdaFatLwbinHandle);
  __lwdaUnregisterFatBinary(__lwdaFatLwbinHandle);
}

static char __lw_init_managed_rt_with_module(void **handle)
{
  return __lwdaInitModule(handle);
}

#include "common_functions.h"

#if defined(__APPLE__)

#pragma options align=natural

#else /* __APPLE__ */

#pragma pack()

#if defined(_WIN32)

#pragma warning(disable: 4099)

#if !defined(_WIN64)

#pragma warning(disable: 4408)

#endif /* !_WIN64 */

#endif /* _WIN32 */

#endif /* __APPLE__ */

#endif /* !__LWDA_INTERNAL_COMPILATION__ */
