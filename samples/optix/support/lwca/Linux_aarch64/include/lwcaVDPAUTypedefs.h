#ifndef LWDAVDPAUTYPEDEFS_H
#define LWDAVDPAUTYPEDEFS_H

// Dependent includes for lwdavdpau.h
#include <vdpau/vdpau.h>

#include <lwdaVDPAU.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwdaVDPAU.h
 */
#define PFN_lwVDPAUGetDevice  PFN_lwVDPAUGetDevice_v3010
#define PFN_lwVDPAUCtxCreate  PFN_lwVDPAUCtxCreate_v3020
#define PFN_lwGraphicsVDPAURegisterVideoSurface  PFN_lwGraphicsVDPAURegisterVideoSurface_v3010
#define PFN_lwGraphicsVDPAURegisterOutputSurface  PFN_lwGraphicsVDPAURegisterOutputSurface_v3010


/**
 * Type definitions for functions defined in lwdaVDPAU.h
 */
typedef LWresult (LWDAAPI *PFN_lwVDPAUGetDevice_v3010)(LWdevice_v1 *pDevice, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
typedef LWresult (LWDAAPI *PFN_lwVDPAUCtxCreate_v3020)(LWcontext *pCtx, unsigned int flags, LWdevice_v1 device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
typedef LWresult (LWDAAPI *PFN_lwGraphicsVDPAURegisterVideoSurface_v3010)(LWgraphicsResource *pLwdaResource, VdpVideoSurface vdpSurface, unsigned int flags);
typedef LWresult (LWDAAPI *PFN_lwGraphicsVDPAURegisterOutputSurface_v3010)(LWgraphicsResource *pLwdaResource, VdpOutputSurface vdpSurface, unsigned int flags);

/*
 * Type definitions for older versioned functions in lwdaVDPAU.h
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
typedef LWresult (LWDAAPI *PFN_lwVDPAUCtxCreate_v3010)(LWcontext *pCtx, unsigned int flags, LWdevice_v1 device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
