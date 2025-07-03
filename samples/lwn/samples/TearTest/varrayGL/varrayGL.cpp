/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "../varray/varray.h"
#include "gl_util.h"
#include "shaders.glsl"

#include <vector>

class RendererVarrayGL : public RendererVarray
{
public:
    void *Init(int winWidth, int winHeight, int maxVertexCount)
    {
        // Init routine needs to do the following tasks:
        // (Note: Textures are created dynamically)

        // 1. Initialize the API

        InitGLPointers();

        // 2. Set up default BlendFunc

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // 3. Create 3 shaders corresponding to PGM_COL, PGM_TEX, PGM_BITMAP

        GLuint vs_ColTex = LoadShader(GL_VERTEX_SHADER,   1, &vert_ColTex);
        GLuint ps_Col    = LoadShader(GL_FRAGMENT_SHADER, 1, &frag_Col);
        GLuint ps_ColTex = LoadShader(GL_FRAGMENT_SHADER, 1, &frag_ColTex);
        GLuint ps_Bitmap = LoadShader(GL_FRAGMENT_SHADER, 1, &frag_Bitmap);

        m_programIds[PGM_COL] = CreateProgram(vs_ColTex, ps_Col);
        m_programIds[PGM_TEX] = CreateProgram(vs_ColTex, ps_ColTex);
        m_programIds[PGM_BITMAP] = CreateProgram(vs_ColTex, ps_Bitmap);

        // 4. Create a query object for timestamp. Also, queue a query.
        //    After this, we always obtain this query and queue a new one.

        glGenQueries(m_maxQueries, m_queryId);
        for (int i = 0; i < m_maxQueries; i++)
        {
            glQueryCounter(m_queryId[i], GL_TIMESTAMP);
        }

        // 5. Allocate a large vertex buffer and setup vertex state.
        // Right now, this code is using immediate mode arrays but should be switched to
        // hardware buffers.

        VertexVCT *vertexVCT = new VertexVCT[maxVertexCount];

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexVCT), (void *) &vertexVCT[0].vertex);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexVCT), (void *) &vertexVCT[0].color);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexVCT), (void *) &vertexVCT[0].tex);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);

        return (void *) vertexVCT;
    }

    const char *Name()
    {
        return "OpenGL";
    }

    void Viewport(int x, int y, uint32_t width, uint32_t height)
    {
        glViewport(x, y, width, height);
    }

    void Clear()
    {
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void SetClearColor(const float *color)
    {
        glClearColor(color[0], color[1], color[2], color[3]);
    }

    void UseProgram(ShaderProgram pgm)
    {
        glUseProgram(m_programIds[pgm]);
    }

    void DrawPrimitive(PrimMode mode, int32_t first, uint32_t count)
    {
        glDrawArrays(mode, first, count);
    }

    void DrawPrimitiveInstanced(PrimMode mode, int32_t first, uint32_t count, int instanceCount)
    {
        glDrawArraysInstanced(mode, first, count, instanceCount);
    }

    unsigned int CreateTex(TexFormat texFormat, unsigned int width, unsigned int height, void *data)
    {
        GLuint texId;
        glGenTextures(1, &texId);

        if (texId == 0)
        {
            Error("Unable to create texture object");
        }

        GLenum texFormatGL = (texFormat == TEX_R8) ? GL_RED : GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, texId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, texFormatGL, width, height, 0,
                                texFormatGL, GL_UNSIGNED_BYTE, data);
        m_texIds.push_back(texId);

        return m_texIds.size() - 1;         // Return "index" to the texId.
    }

    void BindTexture(unsigned int index)
    {
        glBindTexture(GL_TEXTURE_2D, m_texIds[index]);

        if (index == 0)
        {
            glDisable(GL_TEXTURE_2D);
        }
        else
        {
            glEnable(GL_TEXTURE_2D);
        }
    }

    void Enable(RenderCap cap)
    {
        glEnable(cap);
    }
    void Disable(RenderCap cap)
    {
        glDisable(cap);
    }

    void BeginFrame()
    {
        m_queryIndex = 0;
    }

    void EndFrame()
    {
        SwapBuffers(wglGetLwrrentDC());
    }

    void InsertQuery(int index)
    {
        glQueryCounter(m_queryId[index], GL_TIMESTAMP);
    };

    unsigned __int64 GetQuery(int index)
    {
        GLuint queryId = m_queryId[index];
        while (true)
        {
            GLuint64 available = 0;
            glGetQueryObjectui64v(queryId, GL_QUERY_RESULT_AVAILABLE, &available);

            if (available) {
                break;
            }
            Sleep(0);
        }

        unsigned __int64 t;
        glGetQueryObjectui64v(queryId, GL_QUERY_RESULT, &t);

        return t;
    }

    void Screenshot(unsigned int width, unsigned int height, void *data, bool readAlpha)
    {
        glReadPixels(0, 0, width, height, readAlpha ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, data);
    }

    void SetVSync(bool v)
    {
        wglSwapIntervalEXT(v);
    }

    void Finish()
    {
        glFinish();
    }

    void BlitFrontToBackBuffer()
    {
        glReadBuffer(GL_FRONT);
        glBlitFramebuffer(0, 0, winWidth, winHeight,
            0, 0, winWidth, winHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

private:
    static const int m_maxQueries = 6;

    GLuint m_programIds[PGM_MAX];

    int    m_queryIndex = 0;
    GLuint m_queryId[m_maxQueries];
    std::vector<GLuint> m_texIds = {0};         // Array of texture ID's. Index = 0 corresponds to "No" texture
};

Renderer *CreateVarrayGLRenderer()
{
    return new RendererVarrayGL;
}
