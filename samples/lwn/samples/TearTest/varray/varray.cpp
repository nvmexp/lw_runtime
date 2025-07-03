/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#define NOMINMAX
#include "varray.h"
#include <algorithm>

// Few #defines so that the code mostly matches the original
// rendering code. Makes it easier to merge.

#define RendererGL          RendererVarray

#define GL_LINES            LINES
#define GL_LINE_STRIP       LINE_STRIP
#define GL_TRIANGLES        TRIANGLES
#define GL_TRIANGLE_STRIP   TRIANGLE_STRIP
#define GL_TRIANGLE_FAN     TRIANGLE_FAN
#define GL_BLEND            BLEND

#define glBindTexture(a,b)  BindTexture(b)
#define glEnable(a)         Enable(a)
#define glDisable(a)        Disable(a)

#define glViewport(a,b,c,d) Viewport(a,b,c,d)

void RendererGL::Init(int winWidth, int winHeight)
{
    // Initialize the API layer before defining the bitmap
    m_vertexVCT = (VertexVCT *) Init(winWidth, winHeight, m_maxVertices);

    // Create bitmap texture

    int texWidth, texHeight;
    m_bitmap = new BitmapText(BitmapText::BITMAP_9X15);
    m_bitmap->GetBitmapTextureSize(&texWidth, &texHeight);

    uint8_t *texData = new uint8_t[texWidth * texHeight]();
    m_bitmap->GetBitmapTexture(texData);
    m_bitmap->Viewport(winWidth, winHeight);

    m_bitmapTexture = CreateTex(TEX_R8, texWidth, texHeight, (void *) texData);
    delete[] texData;
}

void RendererGL::Reshape(int width, int height)
{
    glViewport(0, 0, width, height);

    winWidth  = width;
    winHeight = height;

    m_bitmap->Viewport(width, height);
}

void RendererGL::DrawStripe(float x1, float x2, const float *colorTop, const float* colorBottom)
{
    glBegin(GL_TRIANGLE_FAN, 4);
    glColor4fv(colorTop);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(x2, 0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(x1, 0);
    glColor4fv(colorBottom);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(x1, winHeight);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(x2, winHeight);
    glEnd();
}

void RendererGL::DrawTexStripe(float x1, float x2, const float *colorTop, const float *colorBottom, unsigned int texId, float repeatV)
{
    glEnable(GL_BLEND);
    glBindTexture(GL_TEXTURE_2D, texId);
    UseProgram(PGM_TEX);

    glBegin(GL_TRIANGLE_FAN, 4);
    glColor4fv(colorTop);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(x1, 0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(x2, 0);
    glColor4fv(colorBottom);
    glTexCoord2f(1.0f, repeatV); glVertex2f(x2, winHeight);
    glTexCoord2f(0.0f, repeatV); glVertex2f(x1, winHeight);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);
    UseProgram(PGM_COL);
}

void RendererGL::DrawGradient(const float *colorMiddle)
{
    glBegin(GL_TRIANGLE_STRIP, 6);
    glColor4fv(colors[black]);
    glVertex2f(0, 0);
    glVertex2f(0, winHeight);
    glColor4fv(colorMiddle);
    glVertex2f(winWidth / 2, 0);
    glVertex2f(winWidth / 2, winHeight);
    glColor4f(2.0f * colorMiddle[0], 2.0f * colorMiddle[1], 2.0f * colorMiddle[2], 1.0f);
    glVertex2f(winWidth, 0);
    glVertex2f(winWidth, winHeight);
    glEnd();
}

void RendererGL::DrawQuad(float x1, float y1, float x2, float y2)
{
    glBegin(GL_TRIANGLE_FAN, 4);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(x1, y1);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(x1, y2);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(x2, y2);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(x2, y1);
    glEnd();
}

void RendererGL::DrawWedge(float x1, float y1, float x2, float y2, float arrowHeight)
{
    glBegin(GL_TRIANGLE_STRIP, 5);
    glVertex2f(x1, y1);
    glVertex2f(x2, y1);
    glVertex2f(x1, y2 - arrowHeight);
    glVertex2f(x2, y2 - arrowHeight);
    glVertex2f(0.5f * (x2 + x1), y2);
    glEnd();
}

void RendererGL::DrawTexQuad(float x1, float y1, float x2, float y2, int tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);
    UseProgram(PGM_TEX);

    DrawQuad(x1, y1, x2, y2);

    glBindTexture(GL_TEXTURE_2D, 0);
    UseProgram(PGM_COL);
}

void RendererGL::LoadGPU(float amt)
{
    int lim = amt;

    float w1 = 0;
    float w2 = winWidth;
    float h1 = winHeight;
    float h2 = winHeight - 16;

    glColor4fv(colors[green]);

    glBegin(GL_TRIANGLE_STRIP, 4);
    glVertex2f(w1, h1);
    glVertex2f(w2, h1);
    glVertex2f(w1, h2);
    glVertex2f(w2, h2);
    glEndInstanced(lim);

    glColor4fv(colors[white]);
}

void RendererGL::SetColor(const float* color)
{
    glColor4fv(color);
}

void RendererGL::RenderString(float x, float y, const char* s)
{
    UseProgram(PGM_BITMAP);
    glBindTexture(GL_TEXTURE_2D, m_bitmapTexture);

    while (s != NULL)
    {
        const char *sdraw = s;  // Start of string to draw
        const char *split;

        while (true)
        {
            if (*s == '\n')
            {
                split = s;
                s += 1;         // Skip 1 char for next string
                break;
            }
            else if ((s[0] == '\\') && (s[1] == 'n'))
            {
                split = s;
                s += 2;         // Skip 2 chars for next string
                break;
            }
            else if (*s == '\0')
            {
                split = NULL;
                s = NULL;       // Last sub-string
                break;
            }
            s++;
        }

        char buf[256];

        if (split != NULL)
        {
            int count = split - sdraw;
            assert (count < sizeof(buf));

            strncpy(buf, sdraw, count);
            buf[count] = '\0';
            sdraw = buf;
        }

        void *v = glBeginDirect(GL_TRIANGLES, m_bitmap->GetVertexCount(sdraw));
        m_bitmap->FillVertices(x, y, &m_color.r, sdraw, v);
        glEnd();

        y += TEXT_ROW_OFFSET;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    UseProgram(PGM_COL);
}

void RendererGL::RenderPlot(float *data, int N, int lwrIdx, int color, bool bgnd, const char* name, int nameOffset, float horizLimitMin, float horizLimitMax)
{
    const float SCR_EDGE_OFFSET = 32;
    const float SCALE_Y = 3.0f;
    const float SCALE_X = 3.0f;
    const float MAX_Y = 160.0f;
    float x = winWidth - SCALE_X*N - SCR_EDGE_OFFSET;
    float y = winHeight - SCR_EDGE_OFFSET;

    if (bgnd)
    {
        glEnable(GL_BLEND);

        if (horizLimitMin)
        {
            glColor4f(0.2f, 0, 0, 0.6f);
            DrawQuad(x, y, x + SCALE_X*(N - 1), y - SCALE_Y * horizLimitMin);
        }

        if (horizLimitMax < MAX_Y)
        {
            glColor4fv(colors[black]);
            DrawQuad(x, y - SCALE_Y * horizLimitMin, x + SCALE_X*(N - 1), y - SCALE_Y * horizLimitMax);
            glColor4f(0.2f, 0, 0, 0.6f);
            DrawQuad(x, y - SCALE_Y * horizLimitMax, x + SCALE_X*(N - 1), y - SCALE_Y * MAX_Y);
        }
        else
        {
            glColor4fv(colors[black]);
            DrawQuad(x, y - SCALE_Y * horizLimitMin, x + SCALE_X*(N - 1), y - SCALE_Y * MAX_Y);
        }

        glColor4fv(colors[white]);

        for (float i = 0; i <= 8; i++)
        {
            float v = MAX_Y / 8 * i;
            char str[128];
            sprintf(str, "%.0f", v);
            RenderString(x + SCALE_X*N + 3, y - SCALE_Y * v + 5, str);
            glBegin(GL_LINES, 2);
            glVertex2f(x, y - SCALE_Y * v);
            glVertex2f(x + SCALE_X*N, y - SCALE_Y * v);
            glEnd();
        }
        glDisable(GL_BLEND);
    }

    glColor4fv(colors[color]);
    glBegin(GL_LINE_STRIP, N);
    for (int i = 0; i < N; i++)
    {
        int idx = (lwrIdx + N + i) % N;
        glVertex2f(x + SCALE_X*i, y - SCALE_Y * std::min(MAX_Y, data[idx]));
    }
    glEnd();

    if (name)
    {
        RenderString(x, y - SCALE_Y * MAX_Y - nameOffset, name);
    }
    glColor4fv(colors[white]);
}

void RendererGL::RenderQAState(QA_STATE state)
{
    if (state == NONE)
    {
        return;
    }

    float xcen = winWidth / 2;
    float ycen = winHeight / 2 + 50;

    int w, h;

    w = 30;// min(winWidth, winHeight) / 4;
    h = 15;

    glEnable(GL_BLEND);
    glColor4fv(colors[black]);
    DrawQuad(xcen - w, ycen - h, xcen + w, ycen + h);
    glDisable(GL_BLEND);

    glColor4fv(colors[white]);
    RenderString(xcen - w/2, ycen - h + 20, state == YES ? "Yes!" : "No!");
}
