/*
 * Copyright (c) 2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

class BenchmarkContextLwWinsysLWN : public BenchmarkContextLWN
{
private:
public:
    BenchmarkContextLwWinsysLWN(LWNdevice* dev,
                                LWNnativeWindow nativeWindow,
                                int w,
                                int h) :
        BenchmarkContextLWN(dev, nativeWindow, w, h)
    {
    }

    ~BenchmarkContextLwWinsysLWN()
    {
    }

    void flip()
    {
#if defined(_WIN32)
        lwogSwapBuffers();
#endif
    }
};

#if !defined(LW_LINUX)

class BenchmarkContextLwWinsysOGL : public BenchmarkContextOGL
{
public:
    struct InitParams {
#if defined(_WIN32)
        int foo;
#else
        EGLDisplay eglDisplay;
        EGLSurface eglSurface;
#endif
        InitParams() {
            memset(this, 0, sizeof(*this));
        }
    };

    BenchmarkContextLwWinsysOGL(const InitParams& params,
                                int w,
                                int h) :
        BenchmarkContextOGL(w, h),
        m_params(params)
    {
    }

    ~BenchmarkContextLwWinsysOGL()
    {
    }

    void flip()
    {
#if !defined(_WIN32)
        eglSwapBuffers(m_params.eglDisplay, m_params.eglSurface);
#else
        lwogSwapBuffers();
#endif
    }

private:
    // TODO argh, this whole EGL business here is just so messy.
    // Should probably just get rid of this whole class altogether.
    InitParams m_params;
};

#endif /* !LW_LINUX */
