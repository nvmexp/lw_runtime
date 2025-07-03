#ifndef __LWDA_LOADER_H__
#define __LWDA_LOADER_H__

#include <lwca.h>

typedef enum lwdaLibraryLoadResult
{
    LWDA_LIBRARY_LOAD_SUCCESS = 0,
    LWDA_LIBRARY_ERROR_NOT_FOUND,
    LWDA_LIBRARY_ERROR_API_NOT_FOUND,
    LWDA_LIBRARY_ERROR_UNLOAD_FAILED,
    LWDA_LIBRARY_ERROR_OUT_OF_MEMORY
} lwdaLibraryLoadResult_t;

#if defined(__cplusplus)
extern "C" {
#endif

lwdaLibraryLoadResult_t loadDefaultLwdaLibrary (void);
lwdaLibraryLoadResult_t unloadLwdaLibrary (void);

#ifdef __cplusplus
}
#endif

#endif

