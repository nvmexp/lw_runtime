#ifndef LWDAGLTYPEDEFS_H
#define LWDAGLTYPEDEFS_H

// Dependent includes for lwdagl.h
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <lwdaGL.h>

#if defined(LWDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptds
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptsz
#else
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## default_version
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## default_version
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in lwdaGL.h
 */
#define PFN_lwGraphicsGLRegisterBuffer  PFN_lwGraphicsGLRegisterBuffer_v3000
#define PFN_lwGraphicsGLRegisterImage  PFN_lwGraphicsGLRegisterImage_v3000
#define PFN_lwWGLGetDevice  PFN_lwWGLGetDevice_v2020
#define PFN_lwGLGetDevices  PFN_lwGLGetDevices_v6050
#define PFN_lwGLCtxCreate  PFN_lwGLCtxCreate_v3020
#define PFN_lwGLInit  PFN_lwGLInit_v2000
#define PFN_lwGLRegisterBufferObject  PFN_lwGLRegisterBufferObject_v2000
#define PFN_lwGLMapBufferObject  __API_TYPEDEF_PTDS(PFN_lwGLMapBufferObject, 3020, 7000)
#define PFN_lwGLUnmapBufferObject  PFN_lwGLUnmapBufferObject_v2000
#define PFN_lwGLUnregisterBufferObject  PFN_lwGLUnregisterBufferObject_v2000
#define PFN_lwGLSetBufferObjectMapFlags  PFN_lwGLSetBufferObjectMapFlags_v2030
#define PFN_lwGLMapBufferObjectAsync  __API_TYPEDEF_PTSZ(PFN_lwGLMapBufferObjectAsync, 3020, 7000)
#define PFN_lwGLUnmapBufferObjectAsync  PFN_lwGLUnmapBufferObjectAsync_v2030


/**
 * Type definitions for functions defined in lwdaGL.h
 */
typedef LWresult (LWDAAPI *PFN_lwGraphicsGLRegisterBuffer_v3000)(LWgraphicsResource *pLwdaResource, GLuint buffer, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwGraphicsGLRegisterImage_v3000)(LWgraphicsResource *pLwdaResource, GLuint image, GLenum target, unsigned int Flags);
#ifdef _WIN32
typedef LWresult (LWDAAPI *PFN_lwWGLGetDevice_v2020)(LWdevice_v1 *pDevice, HGPULW hGpu);
#endif
typedef LWresult (LWDAAPI *PFN_lwGLGetDevices_v6050)(unsigned int *pLwdaDeviceCount, LWdevice_v1 *pLwdaDevices, unsigned int lwdaDeviceCount, LWGLDeviceList deviceList);
typedef LWresult (LWDAAPI *PFN_lwGLCtxCreate_v3020)(LWcontext *pCtx, unsigned int Flags, LWdevice_v1 device);
typedef LWresult (LWDAAPI *PFN_lwGLInit_v2000)(void);
typedef LWresult (LWDAAPI *PFN_lwGLRegisterBufferObject_v2000)(GLuint buffer);
typedef LWresult (LWDAAPI *PFN_lwGLMapBufferObject_v7000_ptds)(LWdeviceptr_v2 *dptr, size_t *size, GLuint buffer);
typedef LWresult (LWDAAPI *PFN_lwGLUnmapBufferObject_v2000)(GLuint buffer);
typedef LWresult (LWDAAPI *PFN_lwGLUnregisterBufferObject_v2000)(GLuint buffer);
typedef LWresult (LWDAAPI *PFN_lwGLSetBufferObjectMapFlags_v2030)(GLuint buffer, unsigned int Flags);
typedef LWresult (LWDAAPI *PFN_lwGLMapBufferObjectAsync_v7000_ptsz)(LWdeviceptr_v2 *dptr, size_t *size, GLuint buffer, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGLUnmapBufferObjectAsync_v2030)(GLuint buffer, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGLMapBufferObject_v3020)(LWdeviceptr_v2 *dptr, size_t *size, GLuint buffer);
typedef LWresult (LWDAAPI *PFN_lwGLMapBufferObjectAsync_v3020)(LWdeviceptr_v2 *dptr, size_t *size, GLuint buffer, LWstream hStream);

/*
 * Type definitions for older versioned functions in lwca.h
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
typedef LWresult (LWDAAPI *PFN_lwGLGetDevices_v4010)(unsigned int *pLwdaDeviceCount, LWdevice_v1 *pLwdaDevices, unsigned int lwdaDeviceCount, LWGLDeviceList deviceList);
typedef LWresult (LWDAAPI *PFN_lwGLMapBufferObject_v2000)(LWdeviceptr_v1 *dptr, unsigned int *size, GLuint buffer);
typedef LWresult (LWDAAPI *PFN_lwGLMapBufferObjectAsync_v2030)(LWdeviceptr_v1 *dptr, unsigned int *size, GLuint buffer, LWstream hStream);
typedef LWresult (LWDAAPI *PFN_lwGLCtxCreate_v2000)(LWcontext *pCtx, unsigned int Flags, LWdevice_v1 device);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
