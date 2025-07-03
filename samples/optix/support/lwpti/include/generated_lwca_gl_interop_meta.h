// This file is generated.  Any changes you make will be lost during the next clean build.

// LWCA public interface, for type definitions and api function prototypes
#include "lwda_gl_interop.h"

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

// Lwrrently used parameter trace structures 
typedef struct lwdaGLGetDevices_v4010_params_st {
    unsigned int *pLwdaDeviceCount;
    int *pLwdaDevices;
    unsigned int lwdaDeviceCount;
    enum lwdaGLDeviceList deviceList;
} lwdaGLGetDevices_v4010_params;

typedef struct lwdaGraphicsGLRegisterImage_v3020_params_st {
    struct lwdaGraphicsResource **resource;
    GLuint image;
    GLenum target;
    unsigned int flags;
} lwdaGraphicsGLRegisterImage_v3020_params;

typedef struct lwdaGraphicsGLRegisterBuffer_v3020_params_st {
    struct lwdaGraphicsResource **resource;
    GLuint buffer;
    unsigned int flags;
} lwdaGraphicsGLRegisterBuffer_v3020_params;

typedef struct lwdaGLSetGLDevice_v3020_params_st {
    int device;
} lwdaGLSetGLDevice_v3020_params;

typedef struct lwdaGLRegisterBufferObject_v3020_params_st {
    GLuint bufObj;
} lwdaGLRegisterBufferObject_v3020_params;

typedef struct lwdaGLMapBufferObject_v3020_params_st {
    void **devPtr;
    GLuint bufObj;
} lwdaGLMapBufferObject_v3020_params;

typedef struct lwdaGLUnmapBufferObject_v3020_params_st {
    GLuint bufObj;
} lwdaGLUnmapBufferObject_v3020_params;

typedef struct lwdaGLUnregisterBufferObject_v3020_params_st {
    GLuint bufObj;
} lwdaGLUnregisterBufferObject_v3020_params;

typedef struct lwdaGLSetBufferObjectMapFlags_v3020_params_st {
    GLuint bufObj;
    unsigned int flags;
} lwdaGLSetBufferObjectMapFlags_v3020_params;

typedef struct lwdaGLMapBufferObjectAsync_v3020_params_st {
    void **devPtr;
    GLuint bufObj;
    lwdaStream_t stream;
} lwdaGLMapBufferObjectAsync_v3020_params;

typedef struct lwdaGLUnmapBufferObjectAsync_v3020_params_st {
    GLuint bufObj;
    lwdaStream_t stream;
} lwdaGLUnmapBufferObjectAsync_v3020_params;

// Parameter trace structures for removed functions 


// End of parameter trace structures
