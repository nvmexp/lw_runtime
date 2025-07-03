// This file is generated.  Any changes you make will be lost during the next clean build.

// LWCA public interface, for type definitions and api function prototypes
#include "lwda_vdpau_interop.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Lwrrently used parameter trace structures 
typedef struct lwdaVDPAUGetDevice_v3020_params_st {
    int *device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} lwdaVDPAUGetDevice_v3020_params;

typedef struct lwdaVDPAUSetVDPAUDevice_v3020_params_st {
    int device;
    VdpDevice vdpDevice;
    VdpGetProcAddress *vdpGetProcAddress;
} lwdaVDPAUSetVDPAUDevice_v3020_params;

typedef struct lwdaGraphicsVDPAURegisterVideoSurface_v3020_params_st {
    struct lwdaGraphicsResource **resource;
    VdpVideoSurface vdpSurface;
    unsigned int flags;
} lwdaGraphicsVDPAURegisterVideoSurface_v3020_params;

typedef struct lwdaGraphicsVDPAURegisterOutputSurface_v3020_params_st {
    struct lwdaGraphicsResource **resource;
    VdpOutputSurface vdpSurface;
    unsigned int flags;
} lwdaGraphicsVDPAURegisterOutputSurface_v3020_params;

// Parameter trace structures for removed functions 


// End of parameter trace structures
