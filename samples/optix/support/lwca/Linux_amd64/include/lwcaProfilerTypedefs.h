#ifndef LWDAPROFILERTYPEDEFS_H
#define LWDAPROFILERTYPEDEFS_H

#include <lwdaProfiler.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwdaProfiler.h
 */
#define PFN_lwProfilerInitialize  PFN_lwProfilerInitialize_v4000
#define PFN_lwProfilerStart  PFN_lwProfilerStart_v4000
#define PFN_lwProfilerStop  PFN_lwProfilerStop_v4000


/**
 * Type definitions for functions defined in lwdaProfiler.h
 */
typedef LWresult (LWDAAPI *PFN_lwProfilerInitialize_v4000)(const char *configFile, const char *outputFile, LWoutput_mode outputMode);
typedef LWresult (LWDAAPI *PFN_lwProfilerStart_v4000)(void);
typedef LWresult (LWDAAPI *PFN_lwProfilerStop_v4000)(void);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
