/*
 * Copyright 1993-2014 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * LWPU Corporation is strictly prohibited.
 *
 * Please refer to the applicable LWPU end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this LWPU software.
 *
 */

// With these flags defined, this source file will dynamically
// load the corresponding functions.  Disabled by default.
//#define LWDA_INIT_D3D9
//#define LWDA_INIT_D3D10
//#define LWDA_INIT_D3D11
//#define LWDA_INIT_OPENGL

#include <stdio.h>
#include "lwda_drvapi_dynlink.h"

tlwInit                               *_lwInit;
tlwDriverGetVersion                   *lwDriverGetVersion;
tlwDeviceGet                          *lwDeviceGet;
tlwDeviceGetCount                     *lwDeviceGetCount;
tlwDeviceGetName                      *lwDeviceGetName;
tlwDeviceComputeCapability            *lwDeviceComputeCapability;
tlwDeviceTotalMem                     *lwDeviceTotalMem;
tlwDeviceGetProperties                *lwDeviceGetProperties;
tlwDeviceGetAttribute                 *lwDeviceGetAttribute;
tlwGetErrorString                     *lwGetErrorString;
tlwCtxCreate                          *lwCtxCreate;
tlwCtxDestroy                         *lwCtxDestroy;
tlwCtxAttach                          *lwCtxAttach;
tlwCtxDetach                          *lwCtxDetach;
tlwCtxPushLwrrent                     *lwCtxPushLwrrent;
tlwCtxPopLwrrent                      *lwCtxPopLwrrent;
tlwCtxGetLwrrent                      *lwCtxGetLwrrent;
tlwCtxSetLwrrent                      *lwCtxSetLwrrent;
tlwCtxGetDevice                       *lwCtxGetDevice;
tlwCtxSynchronize                     *lwCtxSynchronize;
tlwModuleLoad                         *lwModuleLoad;
tlwModuleLoadData                     *lwModuleLoadData;
tlwModuleLoadDataEx                   *lwModuleLoadDataEx;
tlwModuleLoadFatBinary                *lwModuleLoadFatBinary;
tlwModuleUnload                       *lwModuleUnload;
tlwModuleGetFunction                  *lwModuleGetFunction;
tlwModuleGetGlobal                    *lwModuleGetGlobal;
tlwModuleGetTexRef                    *lwModuleGetTexRef;
tlwModuleGetSurfRef                   *lwModuleGetSurfRef;
tlwMemGetInfo                         *lwMemGetInfo;
tlwMemAlloc                           *lwMemAlloc;
tlwMemAllocPitch                      *lwMemAllocPitch;
tlwMemFree                            *lwMemFree;
tlwMemGetAddressRange                 *lwMemGetAddressRange;
tlwMemAllocHost                       *lwMemAllocHost;
tlwMemFreeHost                        *lwMemFreeHost;
tlwMemHostAlloc                       *lwMemHostAlloc;
tlwMemHostGetFlags                    *lwMemHostGetFlags;

tlwMemHostGetDevicePointer            *lwMemHostGetDevicePointer;
tlwDeviceGetByPCIBusId                *lwDeviceGetByPCIBusId;
tlwDeviceGetPCIBusId                  *lwDeviceGetPCIBusId;
tlwIpcGetEventHandle                  *lwIpcGetEventHandle;
tlwIpcOpenEventHandle                 *lwIpcOpenEventHandle;
tlwIpcGetMemHandle                    *lwIpcGetMemHandle;
tlwIpcOpenMemHandle                   *lwIpcOpenMemHandle;
tlwIpcCloseMemHandle                  *lwIpcCloseMemHandle;

tlwMemHostRegister                    *lwMemHostRegister;
tlwMemHostUnregister                  *lwMemHostUnregister;
tlwMemcpyHtoD                         *lwMemcpyHtoD;
tlwMemcpyDtoH                         *lwMemcpyDtoH;
tlwMemcpyDtoD                         *lwMemcpyDtoD;
tlwMemcpyDtoA                         *lwMemcpyDtoA;
tlwMemcpyAtoD                         *lwMemcpyAtoD;
tlwMemcpyHtoA                         *lwMemcpyHtoA;
tlwMemcpyAtoH                         *lwMemcpyAtoH;
tlwMemcpyAtoA                         *lwMemcpyAtoA;
tlwMemcpy2D                           *lwMemcpy2D;
tlwMemcpy2DUnaligned                  *lwMemcpy2DUnaligned;
tlwMemcpy3D                           *lwMemcpy3D;
tlwMemcpyHtoDAsync                    *lwMemcpyHtoDAsync;
tlwMemcpyDtoHAsync                    *lwMemcpyDtoHAsync;
tlwMemcpyDtoDAsync                    *lwMemcpyDtoDAsync;
tlwMemcpyHtoAAsync                    *lwMemcpyHtoAAsync;
tlwMemcpyAtoHAsync                    *lwMemcpyAtoHAsync;
tlwMemcpy2DAsync                      *lwMemcpy2DAsync;
tlwMemcpy3DAsync                      *lwMemcpy3DAsync;
tlwMemcpy                             *lwMemcpy;
tlwMemcpyPeer                         *lwMemcpyPeer;
tlwMemsetD8                           *lwMemsetD8;
tlwMemsetD16                          *lwMemsetD16;
tlwMemsetD32                          *lwMemsetD32;
tlwMemsetD2D8                         *lwMemsetD2D8;
tlwMemsetD2D16                        *lwMemsetD2D16;
tlwMemsetD2D32                        *lwMemsetD2D32;
tlwFuncSetBlockShape                  *lwFuncSetBlockShape;
tlwFuncSetSharedSize                  *lwFuncSetSharedSize;
tlwFuncGetAttribute                   *lwFuncGetAttribute;
tlwFuncSetCacheConfig                 *lwFuncSetCacheConfig;
tlwFuncSetSharedMemConfig             *lwFuncSetSharedMemConfig;
tlwLaunchKernel                       *lwLaunchKernel;
tlwArrayCreate                        *lwArrayCreate;
tlwArrayGetDescriptor                 *lwArrayGetDescriptor;
tlwArrayDestroy                       *lwArrayDestroy;
tlwArray3DCreate                      *lwArray3DCreate;
tlwArray3DGetDescriptor               *lwArray3DGetDescriptor;
tlwTexRefCreate                       *lwTexRefCreate;
tlwTexRefDestroy                      *lwTexRefDestroy;
tlwTexRefSetArray                     *lwTexRefSetArray;
tlwTexRefSetAddress                   *lwTexRefSetAddress;
tlwTexRefSetAddress2D                 *lwTexRefSetAddress2D;
tlwTexRefSetFormat                    *lwTexRefSetFormat;
tlwTexRefSetAddressMode               *lwTexRefSetAddressMode;
tlwTexRefSetFilterMode                *lwTexRefSetFilterMode;
tlwTexRefSetFlags                     *lwTexRefSetFlags;
tlwTexRefGetAddress                   *lwTexRefGetAddress;
tlwTexRefGetArray                     *lwTexRefGetArray;
tlwTexRefGetAddressMode               *lwTexRefGetAddressMode;
tlwTexRefGetFilterMode                *lwTexRefGetFilterMode;
tlwTexRefGetFormat                    *lwTexRefGetFormat;
tlwTexRefGetFlags                     *lwTexRefGetFlags;
tlwSurfRefSetArray                    *lwSurfRefSetArray;
tlwSurfRefGetArray                    *lwSurfRefGetArray;
tlwParamSetSize                       *lwParamSetSize;
tlwParamSeti                          *lwParamSeti;
tlwParamSetf                          *lwParamSetf;
tlwParamSetv                          *lwParamSetv;
tlwParamSetTexRef                     *lwParamSetTexRef;
tlwLaunch                             *lwLaunch;
tlwLaunchGrid                         *lwLaunchGrid;
tlwLaunchGridAsync                    *lwLaunchGridAsync;
tlwEventCreate                        *lwEventCreate;
tlwEventRecord                        *lwEventRecord;
tlwEventQuery                         *lwEventQuery;
tlwEventSynchronize                   *lwEventSynchronize;
tlwEventDestroy                       *lwEventDestroy;
tlwEventElapsedTime                   *lwEventElapsedTime;
tlwStreamCreate                       *lwStreamCreate;
tlwStreamWaitEvent                    *lwStreamWaitEvent;
tlwStreamAddCallback                  *lwStreamAddCallback;
tlwStreamQuery                        *lwStreamQuery;
tlwStreamSynchronize                  *lwStreamSynchronize;
tlwStreamDestroy                      *lwStreamDestroy;
tlwGraphicsUnregisterResource         *lwGraphicsUnregisterResource;
tlwGraphicsSubResourceGetMappedArray  *lwGraphicsSubResourceGetMappedArray;
tlwGraphicsResourceGetMappedPointer   *lwGraphicsResourceGetMappedPointer;
tlwGraphicsResourceSetMapFlags        *lwGraphicsResourceSetMapFlags;
tlwGraphicsMapResources               *lwGraphicsMapResources;
tlwGraphicsUnmapResources             *lwGraphicsUnmapResources;
tlwGetExportTable                     *lwGetExportTable;
tlwCtxSetLimit                        *lwCtxSetLimit;
tlwCtxGetLimit                        *lwCtxGetLimit;
tlwCtxGetCacheConfig                  *lwCtxGetCacheConfig;
tlwCtxSetCacheConfig                  *lwCtxSetCacheConfig;
tlwCtxGetSharedMemConfig              *lwCtxGetSharedMemConfig;
tlwCtxSetSharedMemConfig              *lwCtxSetSharedMemConfig;
tlwCtxGetApiVersion                   *lwCtxGetApiVersion;

tlwMipmappedArrayCreate               *lwMipmappedArrayCreate;
tlwMipmappedArrayGetLevel             *lwMipmappedArrayGetLevel;
tlwMipmappedArrayDestroy              *lwMipmappedArrayDestroy;

tlwProfilerStop                       *lwProfilerStop;

#ifdef LWDA_INIT_D3D9
// D3D9/LWCA interop (LWCA 1.x compatible API). These functions
// are deprecated; please use the ones below
tlwD3D9Begin                          *lwD3D9Begin;
tlwD3D9End                            *lwD3DEnd;
tlwD3D9RegisterVertexBuffer           *lwD3D9RegisterVertexBuffer;
tlwD3D9MapVertexBuffer                *lwD3D9MapVertexBuffer;
tlwD3D9UnmapVertexBuffer              *lwD3D9UnmapVertexBuffer;
tlwD3D9UnregisterVertexBuffer         *lwD3D9UnregisterVertexBuffer;

// D3D9/LWCA interop (LWCA 2.x compatible)
tlwD3D9GetDirect3DDevice              *lwD3D9GetDirect3DDevice;
tlwD3D9RegisterResource               *lwD3D9RegisterResource;
tlwD3D9UnregisterResource             *lwD3D9UnregisterResource;
tlwD3D9MapResources                   *lwD3D9MapResources;
tlwD3D9UnmapResources                 *lwD3D9UnmapResources;
tlwD3D9ResourceSetMapFlags            *lwD3D9ResourceSetMapFlags;
tlwD3D9ResourceGetSurfaceDimensions   *lwD3D9ResourceGetSurfaceDimensions;
tlwD3D9ResourceGetMappedArray         *lwD3D9ResourceGetMappedArray;
tlwD3D9ResourceGetMappedPointer       *lwD3D9ResourceGetMappedPointer;
tlwD3D9ResourceGetMappedSize          *lwD3D9ResourceGetMappedSize;
tlwD3D9ResourceGetMappedPitch         *lwD3D9ResourceGetMappedPitch;

// D3D9/LWCA interop (LWCA 2.0+)
tlwD3D9GetDevice                      *lwD3D9GetDevice;
tlwD3D9CtxCreate                      *lwD3D9CtxCreate;
tlwGraphicsD3D9RegisterResource       *lwGraphicsD3D9RegisterResource;
#endif

#ifdef LWDA_INIT_D3D10
// D3D10/LWCA interop (LWCA 3.0+)
tlwD3D10GetDevice                     *lwD3D10GetDevice;
tlwD3D10CtxCreate                     *lwD3D10CtxCreate;
tlwGraphicsD3D10RegisterResource      *lwGraphicsD3D10RegisterResource;
#endif


#ifdef LWDA_INIT_D3D11
// D3D11/LWCA interop (LWCA 3.0+)
tlwD3D11GetDevice                     *lwD3D11GetDevice;
tlwD3D11CtxCreate                     *lwD3D11CtxCreate;
tlwGraphicsD3D11RegisterResource      *lwGraphicsD3D11RegisterResource;
#endif

// GL/LWCA interop
#ifdef LWDA_INIT_OPENGL
tlwGLCtxCreate                        *lwGLCtxCreate;
tlwGraphicsGLRegisterBuffer           *lwGraphicsGLRegisterBuffer;
tlwGraphicsGLRegisterImage            *lwGraphicsGLRegisterImage;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
tlwWGLGetDevice                       *lwWGLGetDevice;
#endif
#endif

#define STRINGIFY(X) #X

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <Windows.h>

#ifdef UNICODE
static LPCWSTR __LwdaLibName = L"lwlwda.dll";
#else
static LPCSTR __LwdaLibName = "lwlwda.dll";
#endif

typedef HMODULE LWDADRIVER;

static LWresult LOAD_LIBRARY(LWDADRIVER *pInstance)
{
    *pInstance = LoadLibrary(__LwdaLibName);

    if (*pInstance == NULL)
    {
        printf("LoadLibrary \"%s\" failed!\n", __LwdaLibName);
        return LWDA_ERROR_UNKNOWN;
    }

    return LWDA_SUCCESS;
}

#define GET_PROC_EX(name, alias, required)                     \
    alias = (t##name *)GetProcAddress(LwdaDrvLib, #name);               \
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               #name, __LwdaLibName);                                  \
        return LWDA_ERROR_UNKNOWN;                                      \
    }

#define GET_PROC_EX_V2(name, alias, required)                           \
    alias = (t##name *)GetProcAddress(LwdaDrvLib, STRINGIFY(name##_v2));\
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               STRINGIFY(name##_v2), __LwdaLibName);                       \
        return LWDA_ERROR_UNKNOWN;                                      \
    }

#define GET_PROC_EX_V3(name, alias, required)                           \
    alias = (t##name *)GetProcAddress(LwdaDrvLib, STRINGIFY(name##_v3));\
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               STRINGIFY(name##_v3), __LwdaLibName);                       \
        return LWDA_ERROR_UNKNOWN;                                      \
    }

#elif defined(__unix__) || defined (__QNX__) || defined(__APPLE__) || defined(__MACOSX)

#include <dlfcn.h>

#if defined(__APPLE__) || defined(__MACOSX)
static char __LwdaLibName[] = "/usr/local/lwca/lib/liblwda.dylib";
#elif defined(__ANDROID__)
#if defined (__aarch64__)
static char __LwdaLibName[] = "/system/vendor/lib64/liblwda.so";
#elif defined(__arm__)
static char __LwdaLibName[] = "/system/vendor/lib/liblwda.so";
#endif
#else
static char __LwdaLibName[] = "liblwda.so.1";
#endif

typedef void *LWDADRIVER;

static LWresult LOAD_LIBRARY(LWDADRIVER *pInstance)
{
    *pInstance = dlopen(__LwdaLibName, RTLD_NOW);

    if (*pInstance == NULL)
    {
        printf("dlopen \"%s\" failed!\n", __LwdaLibName);
        return LWDA_ERROR_UNKNOWN;
    }

    return LWDA_SUCCESS;
}

#define GET_PROC_EX(name, alias, required)                              \
    alias = (t##name *)dlsym(LwdaDrvLib, #name);                        \
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               #name, __LwdaLibName);                                  \
        return LWDA_ERROR_UNKNOWN;                                      \
    }

#define GET_PROC_EX_V2(name, alias, required)                           \
    alias = (t##name *)dlsym(LwdaDrvLib, STRINGIFY(name##_v2));         \
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               STRINGIFY(name##_v2), __LwdaLibName);                    \
        return LWDA_ERROR_UNKNOWN;                                      \
    }

#define GET_PROC_EX_V3(name, alias, required)                           \
    alias = (t##name *)dlsym(LwdaDrvLib, STRINGIFY(name##_v3));         \
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               STRINGIFY(name##_v3), __LwdaLibName);                    \
        return LWDA_ERROR_UNKNOWN;                                      \
    }

#else
#error unsupported platform
#endif

#define CHECKED_CALL(call)              \
    do {                                \
        LWresult result = (call);       \
        if (LWDA_SUCCESS != result) {   \
            return result;              \
        }                               \
    } while(0)

#define GET_PROC_REQUIRED(name) GET_PROC_EX(name,name,1)
#define GET_PROC_OPTIONAL(name) GET_PROC_EX(name,name,0)
#define GET_PROC(name)          GET_PROC_REQUIRED(name)
#define GET_PROC_V2(name)       GET_PROC_EX_V2(name,name,1)
#define GET_PROC_V3(name)       GET_PROC_EX_V3(name,name,1)

LWresult LWDAAPI lwInit(unsigned int Flags, int lwdaVersion)
{
    LWDADRIVER LwdaDrvLib;
    int driverVer = 1000;

    CHECKED_CALL(LOAD_LIBRARY(&LwdaDrvLib));

    // lwInit is required; alias it to _lwInit
    GET_PROC_EX(lwInit, _lwInit, 1);
    CHECKED_CALL(_lwInit(Flags));

    // available since 2.2. if not present, version 1.0 is assumed
    GET_PROC_OPTIONAL(lwDriverGetVersion);

    if (lwDriverGetVersion)
    {
        CHECKED_CALL(lwDriverGetVersion(&driverVer));
    }

    // fetch all function pointers
    GET_PROC(lwDeviceGet);
    GET_PROC(lwDeviceGetCount);
    GET_PROC(lwDeviceGetName);
    GET_PROC(lwDeviceComputeCapability);
    GET_PROC(lwDeviceGetProperties);
    GET_PROC(lwDeviceGetAttribute);
    GET_PROC(lwGetErrorString);
    GET_PROC(lwCtxDestroy);
    GET_PROC(lwCtxAttach);
    GET_PROC(lwCtxDetach);
    GET_PROC(lwCtxPushLwrrent);
    GET_PROC(lwCtxPopLwrrent);
    GET_PROC(lwCtxGetDevice);
    GET_PROC(lwCtxSynchronize);
    GET_PROC(lwCtxSetLimit);
    GET_PROC(lwCtxGetCacheConfig);
    GET_PROC(lwCtxSetCacheConfig);
    GET_PROC(lwCtxGetApiVersion);
    GET_PROC(lwModuleLoad);
    GET_PROC(lwModuleLoadData);
    GET_PROC(lwModuleUnload);
    GET_PROC(lwModuleGetFunction);
    GET_PROC(lwModuleGetTexRef);
    GET_PROC(lwMemFreeHost);
    GET_PROC(lwMemHostAlloc);
    GET_PROC(lwFuncSetBlockShape);
    GET_PROC(lwFuncSetSharedSize);
    GET_PROC(lwFuncGetAttribute);
    GET_PROC(lwArrayDestroy);
    GET_PROC(lwTexRefCreate);
    GET_PROC(lwTexRefDestroy);
    GET_PROC(lwTexRefSetArray);
    GET_PROC(lwTexRefSetFormat);
    GET_PROC(lwTexRefSetAddressMode);
    GET_PROC(lwTexRefSetFilterMode);
    GET_PROC(lwTexRefSetFlags);
    GET_PROC(lwTexRefGetArray);
    GET_PROC(lwTexRefGetAddressMode);
    GET_PROC(lwTexRefGetFilterMode);
    GET_PROC(lwTexRefGetFormat);
    GET_PROC(lwTexRefGetFlags);
    GET_PROC(lwParamSetSize);
    GET_PROC(lwParamSeti);
    GET_PROC(lwParamSetf);
    GET_PROC(lwParamSetv);
    GET_PROC(lwParamSetTexRef);
    GET_PROC(lwLaunch);
    GET_PROC(lwLaunchGrid);
    GET_PROC(lwLaunchGridAsync);
    GET_PROC(lwEventCreate);
    GET_PROC(lwEventRecord);
    GET_PROC(lwEventQuery);
    GET_PROC(lwEventSynchronize);
    GET_PROC(lwEventDestroy);
    GET_PROC(lwEventElapsedTime);
    GET_PROC(lwStreamCreate);
    GET_PROC(lwStreamWaitEvent);
    GET_PROC(lwStreamAddCallback);
    GET_PROC(lwStreamQuery);
    GET_PROC(lwStreamSynchronize);
    GET_PROC(lwStreamDestroy);

    // These are LWCA 5.0 new functions
    if (driverVer >= 5000)
    {
        GET_PROC(lwMipmappedArrayCreate);
        GET_PROC(lwMipmappedArrayDestroy);
        GET_PROC(lwMipmappedArrayGetLevel);
    }

    // These are LWCA 4.2 new functions
    if (driverVer >= 4020)
    {
        GET_PROC(lwFuncSetSharedMemConfig);
        GET_PROC(lwCtxGetSharedMemConfig);
        GET_PROC(lwCtxSetSharedMemConfig);
    }

    // These are LWCA 4.1 new functions
    if (lwdaVersion >= 4010 && __LWDA_API_VERSION >= 4010)
    {
        GET_PROC(lwDeviceGetByPCIBusId);
        GET_PROC(lwDeviceGetPCIBusId);
        GET_PROC(lwIpcGetEventHandle);
        GET_PROC(lwIpcOpenEventHandle);
        GET_PROC(lwIpcGetMemHandle);
        GET_PROC(lwIpcOpenMemHandle);
        GET_PROC(lwIpcCloseMemHandle);
    }

    // These could be _v2 interfaces
    if (lwdaVersion >= 4000 && __LWDA_API_VERSION >= 4000)
    {
        GET_PROC_V2(lwCtxDestroy);
        GET_PROC_V2(lwCtxPopLwrrent);
        GET_PROC_V2(lwCtxPushLwrrent);
        GET_PROC_V2(lwStreamDestroy);
        GET_PROC_V2(lwEventDestroy);
    }

    if (lwdaVersion >= 3020 && __LWDA_API_VERSION >= 3020)
    {
        GET_PROC_V2(lwDeviceTotalMem);
        GET_PROC_V2(lwCtxCreate);
        GET_PROC_V2(lwModuleGetGlobal);
        GET_PROC_V2(lwMemGetInfo);
        GET_PROC_V2(lwMemAlloc);
        GET_PROC_V2(lwMemAllocPitch);
        GET_PROC_V2(lwMemFree);
        GET_PROC_V2(lwMemGetAddressRange);
        GET_PROC_V2(lwMemAllocHost);
        GET_PROC_V2(lwMemHostGetDevicePointer);
        GET_PROC_V2(lwMemcpyHtoD);
        GET_PROC_V2(lwMemcpyDtoH);
        GET_PROC_V2(lwMemcpyDtoD);
        GET_PROC_V2(lwMemcpyDtoA);
        GET_PROC_V2(lwMemcpyAtoD);
        GET_PROC_V2(lwMemcpyHtoA);
        GET_PROC_V2(lwMemcpyAtoH);
        GET_PROC_V2(lwMemcpyAtoA);
        GET_PROC_V2(lwMemcpy2D);
        GET_PROC_V2(lwMemcpy2DUnaligned);
        GET_PROC_V2(lwMemcpy3D);
        GET_PROC_V2(lwMemcpyHtoDAsync);
        GET_PROC_V2(lwMemcpyDtoHAsync);
        GET_PROC_V2(lwMemcpyHtoAAsync);
        GET_PROC_V2(lwMemcpyAtoHAsync);
        GET_PROC_V2(lwMemcpy2DAsync);
        GET_PROC_V2(lwMemcpy3DAsync);
        GET_PROC_V2(lwMemsetD8);
        GET_PROC_V2(lwMemsetD16);
        GET_PROC_V2(lwMemsetD32);
        GET_PROC_V2(lwMemsetD2D8);
        GET_PROC_V2(lwMemsetD2D16);
        GET_PROC_V2(lwMemsetD2D32);
        GET_PROC_V2(lwArrayCreate);
        GET_PROC_V2(lwArrayGetDescriptor);
        GET_PROC_V2(lwArray3DCreate);
        GET_PROC_V2(lwArray3DGetDescriptor);
        GET_PROC_V2(lwTexRefSetAddress);
        GET_PROC_V2(lwTexRefGetAddress);

        if (lwdaVersion >= 4010 && __LWDA_API_VERSION >= 4010)
        {
            GET_PROC_V3(lwTexRefSetAddress2D);
        }
        else
        {
            GET_PROC_V2(lwTexRefSetAddress2D);
        }
    }
    else
    {
        // versions earlier than 3020
        GET_PROC(lwDeviceTotalMem);
        GET_PROC(lwCtxCreate);
        GET_PROC(lwModuleGetGlobal);
        GET_PROC(lwMemGetInfo);
        GET_PROC(lwMemAlloc);
        GET_PROC(lwMemAllocPitch);
        GET_PROC(lwMemFree);
        GET_PROC(lwMemGetAddressRange);
        GET_PROC(lwMemAllocHost);
        GET_PROC(lwMemHostGetDevicePointer);
        GET_PROC(lwMemcpyHtoD);
        GET_PROC(lwMemcpyDtoH);
        GET_PROC(lwMemcpyDtoD);
        GET_PROC(lwMemcpyDtoA);
        GET_PROC(lwMemcpyAtoD);
        GET_PROC(lwMemcpyHtoA);
        GET_PROC(lwMemcpyAtoH);
        GET_PROC(lwMemcpyAtoA);
        GET_PROC(lwMemcpy2D);
        GET_PROC(lwMemcpy2DUnaligned);
        GET_PROC(lwMemcpy3D);
        GET_PROC(lwMemcpyHtoDAsync);
        GET_PROC(lwMemcpyDtoHAsync);
        GET_PROC(lwMemcpyHtoAAsync);
        GET_PROC(lwMemcpyAtoHAsync);
        GET_PROC(lwMemcpy2DAsync);
        GET_PROC(lwMemcpy3DAsync);
        GET_PROC(lwMemsetD8);
        GET_PROC(lwMemsetD16);
        GET_PROC(lwMemsetD32);
        GET_PROC(lwMemsetD2D8);
        GET_PROC(lwMemsetD2D16);
        GET_PROC(lwMemsetD2D32);
        GET_PROC(lwArrayCreate);
        GET_PROC(lwArrayGetDescriptor);
        GET_PROC(lwArray3DCreate);
        GET_PROC(lwArray3DGetDescriptor);
        GET_PROC(lwTexRefSetAddress);
        GET_PROC(lwTexRefSetAddress2D);
        GET_PROC(lwTexRefGetAddress);
    }

    // The following functions are specific to LWCA versions
    if (driverVer >= 4000)
    {
        GET_PROC(lwCtxSetLwrrent);
        GET_PROC(lwCtxGetLwrrent);
        GET_PROC(lwMemHostRegister);
        GET_PROC(lwMemHostUnregister);
        GET_PROC(lwMemcpy);
        GET_PROC(lwMemcpyPeer);
        GET_PROC(lwLaunchKernel);
        GET_PROC(lwProfilerStop);
    }

    if (driverVer >= 3010)
    {
        GET_PROC(lwModuleGetSurfRef);
        GET_PROC(lwSurfRefSetArray);
        GET_PROC(lwSurfRefGetArray);
        GET_PROC(lwCtxSetLimit);
        GET_PROC(lwCtxGetLimit);
    }

    if (driverVer >= 3000)
    {
        GET_PROC(lwMemcpyDtoDAsync);
        GET_PROC(lwFuncSetCacheConfig);
#ifdef LWDA_INIT_D3D11
        GET_PROC(lwD3D11GetDevice);
        GET_PROC(lwD3D11CtxCreate);
        GET_PROC(lwGraphicsD3D11RegisterResource);
#endif
        GET_PROC(lwGraphicsUnregisterResource);
        GET_PROC(lwGraphicsSubResourceGetMappedArray);

        if (lwdaVersion >= 3020 && __LWDA_API_VERSION >= 3020)
        {
            GET_PROC_V2(lwGraphicsResourceGetMappedPointer);
        }
        else
        {
            GET_PROC(lwGraphicsResourceGetMappedPointer);
        }

        GET_PROC(lwGraphicsResourceSetMapFlags);
        GET_PROC(lwGraphicsMapResources);
        GET_PROC(lwGraphicsUnmapResources);
        GET_PROC(lwGetExportTable);
    }

    if (driverVer >= 2030)
    {
        GET_PROC(lwMemHostGetFlags);
#ifdef LWDA_INIT_D3D10
        GET_PROC(lwD3D10GetDevice);
        GET_PROC(lwD3D10CtxCreate);
        GET_PROC(lwGraphicsD3D10RegisterResource);
#endif
#ifdef LWDA_INIT_OPENGL
        GET_PROC(lwGraphicsGLRegisterBuffer);
        GET_PROC(lwGraphicsGLRegisterImage);
#endif
    }

    if (driverVer >= 2010)
    {
        GET_PROC(lwModuleLoadDataEx);
        GET_PROC(lwModuleLoadFatBinary);
#ifdef LWDA_INIT_OPENGL
        GET_PROC(lwGLCtxCreate);
        GET_PROC(lwGraphicsGLRegisterBuffer);
        GET_PROC(lwGraphicsGLRegisterImage);
#  ifdef WIN32
        GET_PROC(lwWGLGetDevice);
#  endif
#endif
#ifdef LWDA_INIT_D3D9
        GET_PROC(lwD3D9GetDevice);
        GET_PROC(lwD3D9CtxCreate);
        GET_PROC(lwGraphicsD3D9RegisterResource);
#endif
    }

    return LWDA_SUCCESS;
}
