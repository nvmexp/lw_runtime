#ifndef LWDAD3D11TYPEDEFS_H
#define LWDAD3D11TYPEDEFS_H

// Dependent includes for lwdaD3D11.h
#include <rpcsal.h>
#include <D3D11_1.h>

#include <lwdaD3D11.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwdaD3D11.h
 */
#define PFN_lwD3D11GetDevice  PFN_lwD3D11GetDevice_v3000
#define PFN_lwD3D11GetDevices  PFN_lwD3D11GetDevices_v3020
#define PFN_lwGraphicsD3D11RegisterResource  PFN_lwGraphicsD3D11RegisterResource_v3000
#define PFN_lwD3D11CtxCreate  PFN_lwD3D11CtxCreate_v3020
#define PFN_lwD3D11CtxCreateOnDevice  PFN_lwD3D11CtxCreateOnDevice_v3020
#define PFN_lwD3D11GetDirect3DDevice  PFN_lwD3D11GetDirect3DDevice_v3020


/**
 * Type definitions for functions defined in lwdaD3D11.h
 */
typedef LWresult (LWDAAPI *PFN_lwD3D11GetDevice_v3000)(LWdevice_v1 *pLwdaDevice, IDXGIAdapter *pAdapter);
typedef LWresult (LWDAAPI *PFN_lwD3D11GetDevices_v3020)(unsigned int *pLwdaDeviceCount, LWdevice_v1 *pLwdaDevices, unsigned int lwdaDeviceCount, ID3D11Device *pD3D11Device, LWd3d11DeviceList deviceList);
typedef LWresult (LWDAAPI *PFN_lwGraphicsD3D11RegisterResource_v3000)(LWgraphicsResource *pLwdaResource, ID3D11Resource *pD3DResource, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwD3D11CtxCreate_v3020)(LWcontext *pCtx, LWdevice_v1 *pLwdaDevice, unsigned int Flags, ID3D11Device *pD3DDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D11CtxCreateOnDevice_v3020)(LWcontext *pCtx, unsigned int flags, ID3D11Device *pD3DDevice, LWdevice_v1 lwdaDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D11GetDirect3DDevice_v3020)(ID3D11Device **ppD3DDevice);

#if defined(__LWDA_API_VERSION_INTERNAL)
    typedef LWresult (LWDAAPI *PFN_lwD3D11CtxCreate_v3000)(LWcontext *pCtx, LWdevice_v1 *pLwdaDevice, unsigned int Flags, ID3D11Device *pD3DDevice);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
