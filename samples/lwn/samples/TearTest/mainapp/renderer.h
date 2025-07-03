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

#include <string>
#include <cstdlib>
#include "error.h"

// Hack: Renderer's depend on this stuff

#include "colors.h"
const float TEXT_ROW_OFFSET = 15.0f;

enum QA_STATE
{
    NONE,
    YES,
    NO
};

class Renderer
{
public:
    enum TexFormat {TEX_RGBA8, TEX_R8};

    virtual void BeginFrame() {};
    virtual void EndFrame() = 0;        // Signal to renderer to finish frame and Swap

    virtual void Init(int winWidth, int winHeight) = 0;
    virtual const char *Name() = 0;
    virtual void Reshape(int width, int height) = 0;
    virtual void Finish() = 0;
    virtual void DrawStripe(float x1, float x2, const float *colorTop, const float *colorBottom) = 0;
    virtual void DrawTexStripe(float x1, float x2, const float *colorTop, const float *colorBottom, unsigned int texId, float repeatV) = 0;
    virtual void DrawGradient(const float *colorMiddle) = 0;
    virtual void DrawQuad(float x1, float y1, float x2, float y2) = 0;
    virtual void DrawWedge(float x1, float y1, float x2, float y2, float arrowHeight) = 0;
    virtual void DrawTexQuad(float x1, float y1, float x2, float y2, int tex) = 0;
    virtual void LoadGPU(float amt) = 0;
    virtual void Clear() = 0;
    virtual void SetColor(const float* color) = 0;
    virtual void SetClearColor(const float* color) = 0;
    virtual void SetVSync(bool v) = 0;
    virtual void RenderString(float x, float y, const char* c) = 0;
    virtual unsigned int CreateTex(TexFormat texFormat, unsigned int width, unsigned int height, void *data) = 0;
    virtual void Screenshot(unsigned int width, unsigned int height, void *data, bool readAlpha = false) = 0;
    virtual void DeleteTex(unsigned int id) = 0;
    virtual void RenderPlot(float *data, int N, int lwrIdx, int color, bool bgnd = true, const char* name = 0, int nameOffset = 0, float horizLimitMin = 0, float horizLimitMax = 0) = 0;

    virtual void InsertQuery(int index) = 0;
    virtual unsigned __int64 GetQuery(int index) = 0;

    virtual void RenderQAState(QA_STATE state) = 0;
    virtual void BlitFrontToBackBuffer() = 0;

    virtual ~Renderer() {}
};

extern Renderer *renderer;

extern Renderer *CreateVarrayGLRenderer();
extern Renderer *CreateVarrayLWNRenderer();

inline Renderer *CreateRenderer(const char *apiString)
{
    if (strcmp(apiString, "OpenGL") == 0)
    {
        renderer = CreateVarrayGLRenderer();
    }
#ifdef BUILD_LWN
    else if (strcmp(apiString, "LWN") == 0)
    {
        renderer = CreateVarrayLWNRenderer();
    }
#endif
    else
    {
        Error("Unknown api(%s) selection", apiString);
    }
    return renderer;
}
