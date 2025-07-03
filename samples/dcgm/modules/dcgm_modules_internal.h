#ifndef DCGM_MODULES_INTERNAL_H
#define DCGM_MODULES_INTERNAL_H

#ifdef DCGM_BUILD_LWSWITCH_MODULE
    #include "dcgm_lwswitch_internal.h"
#endif
#ifdef DCGM_BUILD_VGPU_MODULE
    #include "dcgm_vgpu_internal.h"
#endif

#endif //DCGM_MODULES_INTERNAL_H
