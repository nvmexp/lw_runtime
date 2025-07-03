// This file is generated.  Any changes you make will be lost during the next clean build.

// Dependent includes
#include <vdpau/vdpau.h>

// LWCA public interface, for type definitions and lw* function prototypes
#include "lwdaVDPAU.h"


// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct lwVDPAUGetDevice_params_st {
    LWdevice *pDevice;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} lwVDPAUGetDevice_params;

typedef struct lwVDPAUCtxCreate_v2_params_st {
    LWcontext *pCtx;
    unsigned int flags;
    LWdevice device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} lwVDPAUCtxCreate_v2_params;

typedef struct lwGraphicsVDPAURegisterVideoSurface_params_st {
    LWgraphicsResource *pLwdaResource;
    VdpVideoSurface vdpSurface;
    unsigned int flags;
} lwGraphicsVDPAURegisterVideoSurface_params;

typedef struct lwGraphicsVDPAURegisterOutputSurface_params_st {
    LWgraphicsResource *pLwdaResource;
    VdpOutputSurface vdpSurface;
    unsigned int flags;
} lwGraphicsVDPAURegisterOutputSurface_params;

typedef struct lwVDPAUCtxCreate_params_st {
    LWcontext *pCtx;
    unsigned int flags;
    LWdevice device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} lwVDPAUCtxCreate_params;
