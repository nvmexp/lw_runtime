#if !defined(_LWPTI_TARGET_H_)
#define _LWPTI_TARGET_H_

/*
LWPTI profiler target API's
This file contains the LWPTI profiling API's.
*/
#include <lwpti_result.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility push(default)
#endif

#ifndef LWPTI_PROFILER_STRUCT_SIZE
#define LWPTI_PROFILER_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

typedef struct LWpti_Device_GetChipName_Params
{
    size_t structSize;                                      //!< [in]
    void* pPriv;                                            //!< [in] assign to NULL

    size_t deviceIndex;                                     //!< [in]
    const char* pChipName;                                  //!< [out]
} LWpti_Device_GetChipName_Params;

#define LWpti_Device_GetChipName_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Device_GetChipName_Params, pChipName)
LWptiResult LWPTIAPI lwptiDeviceGetChipName(LWpti_Device_GetChipName_Params *pParams);

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif
