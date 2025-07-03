#ifndef __LWDA_HOOK_H__
#define __LWDA_HOOK_H__

#include <lwca.h>
#include "lwca-loader.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hook API's
#define BEGIN_ENTRYPOINTS
#define END_ENTRYPOINTS
#define LWDA_API_ENTRYPOINT(lwdaFuncname, entrypointApiFuncname, argtypes, fmt, ...)                \
    typedef LWresult (LWDAAPI *lwdaFuncname ## _loader_t) argtypes;                                 \
    void set_ ## lwdaFuncname ## Hook (lwdaFuncname ## _loader_t lwdaFuncHook);                     \
    void reset_ ## lwdaFuncname ## Hook (void);

#include "lwca-entrypoints.h"

#undef LWDA_API_ENTRYPOINT
#undef END_ENTRYPOINTS
#undef BEGIN_ENTRYPOINTS

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hook Managment

void resetAllLwdaHooks (void);

#endif /* __LWDA_HOOK_H__ */
