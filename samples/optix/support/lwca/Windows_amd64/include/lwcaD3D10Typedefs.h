#ifndef LWDAD3D10TYPEDEFS_H
#define LWDAD3D10TYPEDEFS_H

// Dependent includes for lwdaD3D10.h
#include <rpcsal.h>
#include <D3D10_1.h>

#include <lwdaD3D10.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwdaD3D10.h
 */
#define PFN_lwD3D10GetDevice  PFN_lwD3D10GetDevice_v2010
#define PFN_lwD3D10GetDevices  PFN_lwD3D10GetDevices_v3020
#define PFN_lwGraphicsD3D10RegisterResource  PFN_lwGraphicsD3D10RegisterResource_v3000
#define PFN_lwD3D10CtxCreate  PFN_lwD3D10CtxCreate_v3020
#define PFN_lwD3D10CtxCreateOnDevice  PFN_lwD3D10CtxCreateOnDevice_v3020
#define PFN_lwD3D10GetDirect3DDevice  PFN_lwD3D10GetDirect3DDevice_v3020
#define PFN_lwD3D10RegisterResource  PFN_lwD3D10RegisterResource_v2010
#define PFN_lwD3D10UnregisterResource  PFN_lwD3D10UnregisterResource_v2010
#define PFN_lwD3D10MapResources  PFN_lwD3D10MapResources_v2010
#define PFN_lwD3D10UnmapResources  PFN_lwD3D10UnmapResources_v2010
#define PFN_lwD3D10ResourceSetMapFlags  PFN_lwD3D10ResourceSetMapFlags_v2010
#define PFN_lwD3D10ResourceGetMappedArray  PFN_lwD3D10ResourceGetMappedArray_v2010
#define PFN_lwD3D10ResourceGetMappedPointer  PFN_lwD3D10ResourceGetMappedPointer_v3020
#define PFN_lwD3D10ResourceGetMappedSize  PFN_lwD3D10ResourceGetMappedSize_v3020
#define PFN_lwD3D10ResourceGetMappedPitch  PFN_lwD3D10ResourceGetMappedPitch_v3020
#define PFN_lwD3D10ResourceGetSurfaceDimensions  PFN_lwD3D10ResourceGetSurfaceDimensions_v3020


/**
 * Type definitions for functions defined in lwdaD3D10.h
 */
typedef LWresult (LWDAAPI *PFN_lwD3D10GetDevice_v2010)(LWdevice_v1 *pLwdaDevice, IDXGIAdapter *pAdapter);
typedef LWresult (LWDAAPI *PFN_lwD3D10GetDevices_v3020)(unsigned int *pLwdaDeviceCount, LWdevice_v1 *pLwdaDevices, unsigned int lwdaDeviceCount, ID3D10Device *pD3D10Device, LWd3d10DeviceList deviceList);
typedef LWresult (LWDAAPI *PFN_lwGraphicsD3D10RegisterResource_v3000)(LWgraphicsResource *pLwdaResource, ID3D10Resource *pD3DResource, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwD3D10CtxCreate_v3020)(LWcontext *pCtx, LWdevice_v1 *pLwdaDevice, unsigned int Flags, ID3D10Device *pD3DDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D10CtxCreateOnDevice_v3020)(LWcontext *pCtx, unsigned int flags, ID3D10Device *pD3DDevice, LWdevice_v1 lwdaDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D10GetDirect3DDevice_v3020)(ID3D10Device **ppD3DDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D10RegisterResource_v2010)(ID3D10Resource *pResource, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwD3D10UnregisterResource_v2010)(ID3D10Resource *pResource);
typedef LWresult (LWDAAPI *PFN_lwD3D10MapResources_v2010)(unsigned int count, ID3D10Resource **ppResources);
typedef LWresult (LWDAAPI *PFN_lwD3D10UnmapResources_v2010)(unsigned int count, ID3D10Resource **ppResources);
typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceSetMapFlags_v2010)(ID3D10Resource *pResource, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetMappedArray_v2010)(LWarray *pArray, ID3D10Resource *pResource, unsigned int SubResource);
typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetMappedPointer_v3020)(LWdeviceptr_v2 *pDevPtr, ID3D10Resource *pResource, unsigned int SubResource);
typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetMappedSize_v3020)(size_t *pSize, ID3D10Resource *pResource, unsigned int SubResource);
typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetMappedPitch_v3020)(size_t *pPitch, size_t *pPitchSlice, ID3D10Resource *pResource, unsigned int SubResource);
typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetSurfaceDimensions_v3020)(size_t *pWidth, size_t *pHeight, size_t *pDepth, ID3D10Resource *pResource, unsigned int SubResource);

/*
 * Type definitions for older versioned functions in lwdaD3D10.h
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
    typedef LWresult (LWDAAPI *PFN_lwD3D10CtxCreate_v2010)(LWcontext *pCtx, LWdevice_v1 *pLwdaDevice, unsigned int Flags, ID3D10Device *pD3DDevice);
    typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetMappedPitch_v2010)(unsigned int *pPitch, unsigned int *pPitchSlice, ID3D10Resource *pResource, unsigned int SubResource);
    typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetMappedPointer_v2010)(LWdeviceptr_v1 *pDevPtr, ID3D10Resource *pResource, unsigned int SubResource);
    typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetMappedSize_v2010)(unsigned int *pSize, ID3D10Resource *pResource, unsigned int SubResource);
    typedef LWresult (LWDAAPI *PFN_lwD3D10ResourceGetSurfaceDimensions_v2010)(unsigned int *pWidth, unsigned int *pHeight, unsigned int *pDepth, ID3D10Resource *pResource, unsigned int SubResource);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
