/*
 * Copyright (c) 2016-2018, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/*
 * gltri
 *
 * Demonstrates OpenGL ES rendering, setting up a window using EGL and nn::vi.
 */

#ifndef _GLTRI_H_
#define _GLTRI_H_

#if !defined(WIN_INTERFACE_LWSTOM)
// Platform specifier used by EGL
#define WIN_INTERFACE_LWSTOM
#endif

#include <EGL/egl.h>
#include <nn/gll.h>
#include <nn/nn_Assert.h>
#include <nn/vi.h>

// Initializes and cleans up nn::vi resources
class WindowMgr
{
public:
    WindowMgr() :
        mDisplay(0),
        mLayer(0)
    {
        nn::vi::Initialize();
        if (!nn::vi::OpenDefaultDisplay(&mDisplay).IsSuccess()) {
            NN_ASSERT(!"OpenDisplay failed.");
        }
        if (!nn::vi::CreateLayer(&mLayer, mDisplay, 640, 480).IsSuccess()) {
            NN_ASSERT(!"CreateLayer failed.");
        }
    }

    ~WindowMgr()
    {
        nn::vi::DestroyLayer(mLayer);
        nn::vi::CloseDisplay(mDisplay);
        nn::vi::Finalize();
    }

    NativeWindowType GetNativeWindowHandle()
    {
        nn::vi::NativeWindowHandle result;
        if (!nn::vi::GetNativeWindow(&result, mLayer).IsSuccess()) {
            NN_ASSERT(!"GetNativeWindow failed.");
        }
        return static_cast<NativeWindowType>(result);
    }

private:
    nn::vi::Display* mDisplay;
    nn::vi::Layer* mLayer;
};

// Initializes and cleans up EGL resources. Makes an OpenGL ES 2.0 context current in the
// constructor, using the provided native window handle as the default framebuffer.
class EglMgr
{
public:
    explicit EglMgr(NativeWindowType nativeWindow) :
        mDisplay(EGL_NO_DISPLAY),
        mSurface(EGL_NO_SURFACE)
    {
        mDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (!mDisplay) {
            NN_ASSERT(!"eglGetDisplay failed.");
        }
        if (!eglInitialize(mDisplay, 0, 0)) {
            NN_ASSERT(!"eglInitialize failed.");
        }
        EGLint configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_NONE
        };
        eglBindAPI(EGL_OPENGL_API);
        EGLint numConfigs = 0;
        EGLConfig config;
        if (!eglChooseConfig(mDisplay, configAttribs, &config, 1, &numConfigs) ||
                numConfigs != 1) {
            NN_ASSERT(!"eglChooseConfig failed.");
        }
        mSurface = eglCreateWindowSurface(mDisplay, config, nativeWindow, 0);
        if (mSurface == EGL_NO_SURFACE) {
            NN_ASSERT(!"eglCreateWindowSurface failed.");
        }
        EGLint contextAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 5,
            EGL_NONE
        };
        EGLContext context = eglCreateContext(mDisplay, config, EGL_NO_CONTEXT, contextAttribs);
        if (context == EGL_NO_CONTEXT) {
            NN_ASSERT(!"eglCreateContext failed.");
        }
        if (!eglMakeLwrrent(mDisplay, mSurface, mSurface, context)) {
            NN_ASSERT(!"eglMakeLwrrent failed.");
        }
        if (nngllInitializeGl() != nngllResult_Succeeded) {
            NN_ASSERT(!"nngllInitializeGl failed.");
        }
    }

    ~EglMgr()
    {
        eglMakeLwrrent(mDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglTerminate(mDisplay);
    }
    void SwapBuffers()
    {
        eglSwapBuffers(mDisplay, mSurface);
    }

    EGLDisplay mDisplay;
    EGLSurface mSurface;
};

// Renders animated frames using a context compatible with OpenGL. Sets up the
// state in the constructor, and draws frames based on a floating-point time code.
// The animation consists of a green triangle rotating against a dark blue background.
class GlMgr
{
public:
    GlMgr(const char* vtxSrc = nullptr, const char* frgSrc = nullptr) : mPosLocation(0), mTimeLocation(0)
    {
        InitializeProgram(vtxSrc, frgSrc);
        InitializeVertexData();
        glClearColor(0.1, 0.1, 0.3, 1.0);
    }

    void DrawFrame(GLfloat time)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform1f(mTimeLocation, time);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

protected:
    void InitializeProgram(const char* vtxSrc, const char* frgSrc)
    {
        static const char *VERTEX_SOURCE =
            "#version 100\n"
            "precision highp float;\n"
            "attribute vec2 a_position;\n"
            "uniform float u_time;\n"
            "void main() {\n"
            "    mat2 xform = mat2(cos(u_time), sin(u_time),\n"
            "                      -sin(u_time), cos(u_time));\n"
            "    gl_Position = vec4(xform * a_position, 0.0, 1.0);\n"
            "}\n";
        static const char *FRAGMENT_SOURCE =
            "#version 100\n"
            "void main() {\n"
            "    gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            "}\n";
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, (vtxSrc ? &vtxSrc : &VERTEX_SOURCE), 0);
        glCompileShader(vs);
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, (frgSrc ? &frgSrc : &FRAGMENT_SOURCE), 0);
        glCompileShader(fs);
        GLuint program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        glUseProgram(program);
        mPosLocation = glGetAttribLocation(program, "a_position");
        mTimeLocation = glGetUniformLocation(program, "u_time");
    }

    void InitializeVertexData()
    {
        static const GLfloat VERTEX_DATA[] = {
            -0.5, -0.5,
            0.5, -0.5,
            0.0, 1.0,
        };
        GLuint vertexArray;
        glGelwertexArrays(1, &vertexArray);
        glBindVertexArray(vertexArray);
        GLuint vertexBuffer;
        glGenBuffers(1, &vertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(VERTEX_DATA), VERTEX_DATA, GL_STATIC_DRAW);
        glEnableVertexAttribArray(mPosLocation);
        glVertexAttribPointer(mPosLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    }

    GLuint mPosLocation;
    GLuint mTimeLocation;
};

#endif
