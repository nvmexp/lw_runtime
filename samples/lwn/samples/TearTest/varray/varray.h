#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <cstdio>
#include <cassert>

#include "renderer.h"
#include "vertex.h"
#include "../bitmap/glutfont.h"

// The main purpose of this class is to colwert OpenGL immediate mode rendering
// into vertex array renering. Additionally, instead of immediate mode texture
// enable/disable this code also enables the use of the shader in the derived class.

class RendererVarray : public Renderer
{
public:
    void Init(int winWidth, int winHeight);
    void Reshape(int width, int height);
    void DrawStripe(float x1, float x2, const float *colorTop, const float *colorBottom);
    void DrawTexStripe(float x1, float x2, const float *colorTop, const float *colorBottom,
                                                    unsigned int texId, float repeatV);
    void DrawGradient(const float *colorMiddle);
    void DrawQuad(float x1, float y1, float x2, float y2);
    void DrawWedge(float x1, float y1, float x2, float y2, float arrowHeight);
    void DrawTexQuad(float x1, float y1, float x2, float y2, int tex);
    void LoadGPU(float amt);
    void SetColor(const float* color);
    void RenderString(float x, float y, const char* c);
    void RenderPlot(float *data, int N, int lwrIdx, int color, bool bgnd = true, const char* name = 0,
                        int nameOffset = 0, float horizLimitMin = 0, float horizLimitMax = 0);
    void RenderQAState(QA_STATE state);
    void DeleteTex(unsigned int id) {}      // Automatic cleanup at the end

    // API specific - passthrough to derived class

    virtual const char *Name() = 0;
    virtual void Clear() = 0;
    virtual void SetClearColor(const float* color) = 0;
    virtual unsigned int CreateTex(TexFormat texFormat, unsigned int width, unsigned int height, void *data) = 0;
    virtual void InsertQuery(int index) = 0;
    virtual unsigned __int64 GetQuery(int index) = 0;
    virtual void Screenshot(unsigned int width, unsigned int height, void *data, bool readAlpha = false) = 0;
    virtual void Finish() = 0;
    virtual void SetVSync(bool v) = 0;
    virtual void BlitFrontToBackBuffer() = 0;

protected:
    enum ShaderProgram {PGM_COL = 0, PGM_TEX, PGM_BITMAP, PGM_MAX};      // Used as indices
    enum PrimMode {LINES = 0x1, LINE_STRIP = 0x3, TRIANGLES = 0x4, TRIANGLE_STRIP = 0x5, TRIANGLE_FAN = 0x6};
    enum RenderCap {BLEND = 0x0BE2};

    int winWidth;
    int winHeight;

    virtual void *Init(int winWidth, int WinHeight, int maxVertexCount) = 0;
    virtual void Viewport(int x, int y, uint32_t width, uint32_t height) = 0;
    virtual void UseProgram(ShaderProgram pgm) = 0;
    virtual void DrawPrimitive(PrimMode mode, int32_t first, uint32_t count) = 0;
    virtual void DrawPrimitiveInstanced(PrimMode mode, int32_t first, uint32_t count, int instanceCount) = 0;
    virtual void BindTexture(unsigned int index) = 0;
    virtual void Enable(RenderCap cap) = 0;
    virtual void Disable(RenderCap cap) = 0;

private:
    BitmapText *m_bitmap;
    unsigned int m_bitmapTexture;

    // Request large vertex buffer so that we can rollover and
    // reuse from the beginning without synchronization

    static const int m_maxVertices = 50000;

    VertexVCT *m_vertexVCT;
    int m_nextIndex = 0;     // Next free vertex

    // GL immediate mode type functions

    PrimMode m_mode = LINES; // Primitive mode
    int m_first = 0;         // Index of 1st vertex in a primitive
    int m_count = 0;         // Number of vertices in the primitive

    Color4 m_color;          // Cached Color state

    void glBegin(PrimMode mode, int count)
    {
        if (m_nextIndex + count > m_maxVertices)
        {
            m_nextIndex = 0;    // Rolled over
        }
        m_mode  = mode;
        m_first = m_nextIndex;
        m_count = count;
    }

    // Begin primitive and provide direct access to vertex buffer pointer

    void *glBeginDirect(PrimMode mode, int count)
    {
        glBegin(mode, count);
        m_nextIndex = m_first + m_count;
        return (void *) &m_vertexVCT[m_first].vertex;
    }

    void glEnd()
    {
        // Ensure that correct number of vertex calls have been made
        assert(m_nextIndex - m_first == m_count);
        DrawPrimitive(m_mode, m_first, m_count);
    }
    void glEndInstanced(int instanceCount)
    {
        // Ensure that correct number of vertex calls have been made
        assert(m_nextIndex - m_first == m_count);
        DrawPrimitiveInstanced(m_mode, m_first, m_count, instanceCount);
    }

    void glColor4fv(const float *color)
    {
        m_color = *(Color4 *) color;
    }

    void glColor4f(float r, float g, float b, float a)
    {
        m_color = {r, g, b, a};
    }

    void glTexCoord2f(float u, float v)
    {
        assert(m_count != 0);
        m_vertexVCT[m_nextIndex].tex = {u, v};
    }

    void glVertex3f(float x, float y, float z)
    {
        assert(m_count != 0);

        y = +1.0f - 2.0f * (y / winHeight);      // To account for ilwerted ortho
        x = -1.0f + 2.0f * (x / winWidth);

        m_vertexVCT[m_nextIndex].vertex = {x, y, z};
        m_vertexVCT[m_nextIndex].color = m_color;
        m_nextIndex++;
    }

    void glVertex2f(float x, float y)
    {
        glVertex3f(x, y, 0.0);
    }
};
