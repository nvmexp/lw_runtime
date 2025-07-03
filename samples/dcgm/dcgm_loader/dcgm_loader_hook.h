#ifndef __DCGM_LOADER_HOOK_H__
#define __DCGM_LOADER_HOOK_H__

#include "dcgm_agent.h"
#include "dcgm_agent_internal.h"
#include "dcgm_client_internal.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Declare API Hooking Mechanisms
#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)           \
    typedef dcgmReturn_t (*dcgmFuncname ## _loader_t) argtypes;                     \
    void set_ ## dcgmFuncname ## Hook (dcgmFuncname ## _loader_t dcgmFuncHook);     \
    void reset_ ## dcgmFuncname ## Hook (void)
// Ignore internal API's since they have no public-facing API to hook. To hook 
// internal API's use set_dcgmInternalGetExportTableHook and return your custom
// export table for DCGM internal API's. 
#define DCGM_INT_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)

#include "entry_point.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Special Manually declared API Hooking Mechanisms 

DCGM_ENTRY_POINT(dcgmEngineInit, dcgmEngineInit, 
    (dcgmOperationMode_t mode),
    "(%d)",
    mode);

DCGM_ENTRY_POINT(dcgmEngineShutdown, dcgmEngineShutdown,
    (void),
    "()");

DCGM_ENTRY_POINT(dcgmInternalGetExportTable, dcgmInternalGetExportTable,
    (const void **ppExportTable, const dcgmUuid_t *pExportTableId),
    "(%p, %p)",
    ppExportTable, pExportTableId);

#undef DCGM_INT_ENTRY_POINT
#undef DCGM_ENTRY_POINT

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hooking Management API's
void resetAllLwmlHooks (void);

#endif /* __DCGM_LOADER_HOOK_H__ */
