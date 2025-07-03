/*
 * Copyright (c) 2019, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <eglapiinterface.h>
#include <stdio.h>
#include <string.h>
#define GET_FUNC(name, type) \
    do { \
        name = (type)eglGetProcAddress(#name); \
        if (!name) { \
            fprintf(stderr, "Failed to find EGL EXT Function %s\n", #name); \
        } \
    } while(0)

#define CHECK_EGL_ERROR(msg) checkErrors(__FILE__, __LINE__, 0, msg)
#define CHECK_LW_ERROR(func, msg) checkErrors(__FILE__, __LINE__, 1, msg, (func))
#define CHECK_GL_ERROR(msg) checkErrors(__FILE__, __LINE__, 2, msg)

#define MAX_SURFACE_PLANES 3
static const char vtxShaderSource[] = {
    "//#extension GL_ARB_explicit_uniform_location: enable \n"
    "//layout (location = 0) in vec4 Vertex;\n"
    "attribute vec4 Vertex;\n"

    "uniform vec4 TransRot;\n"
    "uniform mat4 ModelViewProjectionMatrix;\n"

    "varying vec2 TexCoord;\n"
    "varying float Opacity;\n"

    "void main (void)\n"
    "{\n"
    "    vec4 transformedVert = Vertex;\n"
    "    float rotation = radians(TransRot.w);\n"
    "    mat4 rotMat = mat4(1.0);\n"
    "    mat4 transMat = mat4(1.0);\n"

    "    // Construct Rotation Matrix\n"
    "    rotMat[0][0] = cos(rotation);\n"
    "    rotMat[1][0] = -sin(rotation);\n"
    "    rotMat[0][1] = sin(rotation);\n"
    "    rotMat[1][1] = cos(rotation);\n"

    "    // Construct Translation Matrix\n"
    "    transMat[3][0] = TransRot.x;\n"
    "    transMat[3][1] = TransRot.y * 1.0;\n"

    "    //All our quads are screen aligned and layered (z = TransRot.z)\n"
    "    transformedVert.z = TransRot.z;\n"
    "    transformedVert.w = 1.0;\n"

    "    //Get our texture coordinates\n"
    "    TexCoord.s = Vertex.z;\n"
    "    TexCoord.t = Vertex.w;\n"

    "    //Pass Through the Opacity\n"
    "    Opacity = 100.0;\n"

    "    gl_Position = ModelViewProjectionMatrix * transMat * rotMat * transformedVert;\n"
    "}\n"
};
static const char fragShaderSource[] ={
    "#extension GL_LW_EGL_stream_consumer_external: enable\n"
    "#extension GL_OES_EGL_image_external : enable\n"

    "uniform samplerExternalOES Texture0;\n"

    "varying lowp vec2 TexCoord;\n"
    "varying lowp float Opacity;\n"

    "void main (void)\n"
    "{\n"
    "    gl_FragColor = texture2D(Texture0, TexCoord);\n"
    "    gl_FragColor.a *= Opacity;\n"
    "}\n"
};

struct eglResources{
    EGLDisplay display;
    EGLenum platform;
    EGLStreamKHR stream;
    EGLStreamKHR pvtStream;
    EGLContext context;
    EGLSurface surface;
    EGLConfig config;
    LwEglApiAccessFuncs apiProcs;
    LwEglApiStream2ProducerCaps producerCaps;
    EGLNativeWindowType nativeWindow;
    EGLNativeDisplayType nativeDisplay;
    LwGlsiEglImageHandle glsiImage;
    LwEglApiStream2Frame streamFrame;
    LwEglApiClientBuffer clientBuffer;
};

typedef struct {
    struct LwRmSyncRec *sync;
    LwS8 eglIndex;
} StreamBuffers;

struct bufferResources{
    LwRmSurface yuvSurf[MAX_SURFACE_PLANES];
    LwRmSurface testSurf;
    LwRmDeviceHandle hRm;
    StreamBuffers streamBuffer;
    char testSurfCrc[LWRM_SURFACE_MD5_BUFFER_SIZE];
};

struct glResources {
    GLuint texId;
    GLuint shaderId;
    GLuint quadVboId;
};

static bool checkErrors(const char *file, int line, int idx, const char *msg, int retVal = 0)
{
    switch(idx)
    {
        case 0:
            {
                EGLint eglError = eglGetError();

                if (eglError != EGL_SUCCESS) {
                    fprintf(stderr, "EGL Error 0x%x at %s:%i\n%s\n", (int)eglError, file, line, msg);
                    return true;
                } else {
                    return false;
                }
            }
        case 1:
                if (retVal) {
                    fprintf(stderr, "LwError 0x%x at %s:%i\n%s\n", retVal, file, line, msg);
                    return true;
                } else {
                    return false;
                }
        case 2:
            {
                GLenum glError = glGetError();

                if (glError) {
                    fprintf(stderr, "GL Error 0x%x at %s:%i\n%s\n", glError, file, line, msg);
                    return true;
                } else {
                    return false;
                }
            }
    }
    return false;
}

PFNEGLCREATESTREAMKHRPROC eglCreateStreamKHR;
PFNEGLDESTROYSTREAMKHRPROC eglDestroyStreamKHR;
PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC eglStreamConsumerGLTextureExternalKHR;
PFNEGLSTREAMCONSUMERACQUIREKHRPROC eglStreamConsumerAcquireKHR;

static void initializeEglExtFunctions(void)
{
    GET_FUNC(eglCreateStreamKHR, PFNEGLCREATESTREAMKHRPROC);
    GET_FUNC(eglDestroyStreamKHR, PFNEGLDESTROYSTREAMKHRPROC);
    GET_FUNC(eglStreamConsumerGLTextureExternalKHR, PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC);
    GET_FUNC(eglStreamConsumerAcquireKHR, PFNEGLSTREAMCONSUMERACQUIREKHRPROC);
}
