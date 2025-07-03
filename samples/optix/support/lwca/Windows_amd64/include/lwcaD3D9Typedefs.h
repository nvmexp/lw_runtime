#ifndef LWDAD3D9TYPEDEFS_H
#define LWDAD3D9TYPEDEFS_H

// Dependent includes for lwdaD3D11.h
#include <d3d9.h>

#include <lwdaD3D9.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwdaD3D9.h
 */
#define PFN_lwD3D9GetDevice  PFN_lwD3D9GetDevice_v2000
#define PFN_lwD3D9GetDevices  PFN_lwD3D9GetDevices_v3020
#define PFN_lwD3D9CtxCreate  PFN_lwD3D9CtxCreate_v3020
#define PFN_lwD3D9CtxCreateOnDevice  PFN_lwD3D9CtxCreateOnDevice_v3020
#define PFN_lwD3D9GetDirect3DDevice  PFN_lwD3D9GetDirect3DDevice_v2000
#define PFN_lwGraphicsD3D9RegisterResource  PFN_lwGraphicsD3D9RegisterResource_v3000
#define PFN_lwD3D9RegisterResource  PFN_lwD3D9RegisterResource_v2000
#define PFN_lwD3D9UnregisterResource  PFN_lwD3D9UnregisterResource_v2000
#define PFN_lwD3D9MapResources  PFN_lwD3D9MapResources_v2000
#define PFN_lwD3D9UnmapResources  PFN_lwD3D9UnmapResources_v2000
#define PFN_lwD3D9ResourceSetMapFlags  PFN_lwD3D9ResourceSetMapFlags_v2000
#define PFN_lwD3D9ResourceGetSurfaceDimensions  PFN_lwD3D9ResourceGetSurfaceDimensions_v3020
#define PFN_lwD3D9ResourceGetMappedArray  PFN_lwD3D9ResourceGetMappedArray_v2010
#define PFN_lwD3D9ResourceGetMappedPointer  PFN_lwD3D9ResourceGetMappedPointer_v3020
#define PFN_lwD3D9ResourceGetMappedSize  PFN_lwD3D9ResourceGetMappedSize_v3020
#define PFN_lwD3D9ResourceGetMappedPitch  PFN_lwD3D9ResourceGetMappedPitch_v3020
#define PFN_lwD3D9Begin  PFN_lwD3D9Begin_v2000
#define PFN_lwD3D9End  PFN_lwD3D9End_v2000
#define PFN_lwD3D9RegisterVertexBuffer  PFN_lwD3D9RegisterVertexBuffer_v2000
#define PFN_lwD3D9MapVertexBuffer  PFN_lwD3D9MapVertexBuffer_v3020
#define PFN_lwD3D9UnmapVertexBuffer  PFN_lwD3D9UnmapVertexBuffer_v2000
#define PFN_lwD3D9UnregisterVertexBuffer  PFN_lwD3D9UnregisterVertexBuffer_v2000


/**
 * Type definitions for functions defined in lwdaD3D9.h
 */
typedef LWresult (LWDAAPI *PFN_lwD3D9GetDevice_v2000)(LWdevice_v1 *pLwdaDevice, const char *pszAdapterName);
typedef LWresult (LWDAAPI *PFN_lwD3D9GetDevices_v3020)(unsigned int *pLwdaDeviceCount, LWdevice_v1 *pLwdaDevices, unsigned int lwdaDeviceCount, IDirect3DDevice9 *pD3D9Device, LWd3d9DeviceList deviceList);
typedef LWresult (LWDAAPI *PFN_lwD3D9CtxCreate_v3020)(LWcontext *pCtx, LWdevice_v1 *pLwdaDevice, unsigned int Flags, IDirect3DDevice9 *pD3DDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D9CtxCreateOnDevice_v3020)(LWcontext *pCtx, unsigned int flags, IDirect3DDevice9 *pD3DDevice, LWdevice_v1 lwdaDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D9GetDirect3DDevice_v2000)(IDirect3DDevice9 **ppD3DDevice);
typedef LWresult (LWDAAPI *PFN_lwGraphicsD3D9RegisterResource_v3000)(LWgraphicsResource *pLwdaResource, IDirect3DResource9 *pD3DResource, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwD3D9RegisterResource_v2000)(IDirect3DResource9 *pResource, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwD3D9UnregisterResource_v2000)(IDirect3DResource9 *pResource);
typedef LWresult (LWDAAPI *PFN_lwD3D9MapResources_v2000)(unsigned int count, IDirect3DResource9 **ppResource);
typedef LWresult (LWDAAPI *PFN_lwD3D9UnmapResources_v2000)(unsigned int count, IDirect3DResource9 **ppResource);
typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceSetMapFlags_v2000)(IDirect3DResource9 *pResource, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetSurfaceDimensions_v3020)(size_t *pWidth, size_t *pHeight, size_t *pDepth, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetMappedArray_v2010)(LWarray *pArray, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetMappedPointer_v3020)(LWdeviceptr_v2 *pDevPtr, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetMappedSize_v3020)(size_t *pSize, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetMappedPitch_v3020)(size_t *pPitch, size_t *pPitchSlice, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
typedef LWresult (LWDAAPI *PFN_lwD3D9Begin_v2000)(IDirect3DDevice9 *pDevice);
typedef LWresult (LWDAAPI *PFN_lwD3D9End_v2000)(void);
typedef LWresult (LWDAAPI *PFN_lwD3D9RegisterVertexBuffer_v2000)(IDirect3DVertexBuffer9 *pVB);
typedef LWresult (LWDAAPI *PFN_lwD3D9MapVertexBuffer_v3020)(LWdeviceptr_v2 *pDevPtr, size_t *pSize, IDirect3DVertexBuffer9 *pVB);
typedef LWresult (LWDAAPI *PFN_lwD3D9UnmapVertexBuffer_v2000)(IDirect3DVertexBuffer9 *pVB);
typedef LWresult (LWDAAPI *PFN_lwD3D9UnregisterVertexBuffer_v2000)(IDirect3DVertexBuffer9 *pVB);

/*
 * Type definitions for older versioned functions in lwdaD3D9.h
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
    typedef LWresult (LWDAAPI *PFN_lwD3D9CtxCreate_v2000)(LWcontext *pCtx, LWdevice_v1 *pLwdaDevice, unsigned int Flags, IDirect3DDevice9 *pD3DDevice);
    typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetSurfaceDimensions_v2000)(unsigned int *pWidth, unsigned int *pHeight, unsigned int *pDepth, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetMappedPointer_v2000)(LWdeviceptr_v1 *pDevPtr, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetMappedSize_v2000)(unsigned int *pSize, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    typedef LWresult (LWDAAPI *PFN_lwD3D9ResourceGetMappedPitch_v2000)(unsigned int *pPitch, unsigned int *pPitchSlice, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    typedef LWresult (LWDAAPI *PFN_lwD3D9MapVertexBuffer_v2000)(LWdeviceptr_v1 *pDevPtr, unsigned int *pSize, IDirect3DVertexBuffer9 *pVB);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
