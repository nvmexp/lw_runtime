/* LWPU-Generated. Include config.h based on the target arch */
#if defined(__powerpc64__)
#include "config_ppc64le_linux.h"
#elif LWOS_IS_VMWARE
#include "config_x64_vmware.h"
#else
#include "config_x64_linux.h"
#endif
