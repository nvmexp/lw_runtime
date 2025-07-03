// This file is generated.  Any changes you make will be lost during the next clean build.

// Dependent includes
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// LWCA public interface, for type definitions and lw* function prototypes
#include "lwdaGL.h"


// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct lwGraphicsGLRegisterBuffer_params_st {
    LWgraphicsResource *pLwdaResource;
    GLuint buffer;
    unsigned int Flags;
} lwGraphicsGLRegisterBuffer_params;

typedef struct lwGraphicsGLRegisterImage_params_st {
    LWgraphicsResource *pLwdaResource;
    GLuint image;
    GLenum target;
    unsigned int Flags;
} lwGraphicsGLRegisterImage_params;

typedef struct lwGLGetDevices_v2_params_st {
    unsigned int *pLwdaDeviceCount;
    LWdevice *pLwdaDevices;
    unsigned int lwdaDeviceCount;
    LWGLDeviceList deviceList;
} lwGLGetDevices_v2_params;

typedef struct lwGLCtxCreate_v2_params_st {
    LWcontext *pCtx;
    unsigned int Flags;
    LWdevice device;
} lwGLCtxCreate_v2_params;

typedef struct lwGLRegisterBufferObject_params_st {
    GLuint buffer;
} lwGLRegisterBufferObject_params;

typedef struct lwGLMapBufferObject_v2_ptds_params_st {
    LWdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
} lwGLMapBufferObject_v2_ptds_params;

typedef struct lwGLUnmapBufferObject_params_st {
    GLuint buffer;
} lwGLUnmapBufferObject_params;

typedef struct lwGLUnregisterBufferObject_params_st {
    GLuint buffer;
} lwGLUnregisterBufferObject_params;

typedef struct lwGLSetBufferObjectMapFlags_params_st {
    GLuint buffer;
    unsigned int Flags;
} lwGLSetBufferObjectMapFlags_params;

typedef struct lwGLMapBufferObjectAsync_v2_ptsz_params_st {
    LWdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
    LWstream hStream;
} lwGLMapBufferObjectAsync_v2_ptsz_params;

typedef struct lwGLUnmapBufferObjectAsync_params_st {
    GLuint buffer;
    LWstream hStream;
} lwGLUnmapBufferObjectAsync_params;

typedef struct lwGLGetDevices_params_st {
    unsigned int *pLwdaDeviceCount;
    LWdevice *pLwdaDevices;
    unsigned int lwdaDeviceCount;
    LWGLDeviceList deviceList;
} lwGLGetDevices_params;

typedef struct lwGLMapBufferObject_v2_params_st {
    LWdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
} lwGLMapBufferObject_v2_params;

typedef struct lwGLMapBufferObjectAsync_v2_params_st {
    LWdeviceptr *dptr;
    size_t *size;
    GLuint buffer;
    LWstream hStream;
} lwGLMapBufferObjectAsync_v2_params;

typedef struct lwGLCtxCreate_params_st {
    LWcontext *pCtx;
    unsigned int Flags;
    LWdevice device;
} lwGLCtxCreate_params;

typedef struct lwGLMapBufferObject_params_st {
    LWdeviceptr_v1 *dptr;
    unsigned int *size;
    GLuint buffer;
} lwGLMapBufferObject_params;

typedef struct lwGLMapBufferObjectAsync_params_st {
    LWdeviceptr_v1 *dptr;
    unsigned int *size;
    GLuint buffer;
    LWstream hStream;
} lwGLMapBufferObjectAsync_params;
