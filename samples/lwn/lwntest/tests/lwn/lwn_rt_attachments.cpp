/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#ifndef CEIL
#define CEIL(a,b)        (((a)+(b)-1)/(b))
#endif

#ifndef ROUND_UP
#define ROUND_UP(N, S) (CEIL((N),(S)) * (S))
#endif

using namespace lwn;

class LWNRenderTargetAttachmentsTest
{
    static const int texDim = 2;
    void BindClearRT(Texture* rt0, int* color0, Texture* rt1, int* color1) const;
    void DisplayTex(TextureHandle handle, int x, int y, int w, int h) const;
public:
    LWNTEST_CppMethods();
};

lwString LWNRenderTargetAttachmentsTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Test checks for behavior of bound render targets when NULLs are passed.\n"
          "This test uses two integer render targets and alternates their binding.\n"
          "Output is considered good if no red color is visible in the output image.";
    return sb.str();
}

int LWNRenderTargetAttachmentsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 19);
}

void LWNRenderTargetAttachmentsTest::BindClearRT(Texture* rt0, int* color0, Texture* rt1, int* color1) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    Texture* targets[2];
    targets[0] = rt0;
    targets[1] = rt1;
    queueCB.SetRenderTargets(2, targets, NULL, NULL, NULL);
    queueCB.SetViewportScissor(0,0, texDim,texDim);
    queueCB.ClearColori(0, color0, ClearColorMask::RGBA);
    queueCB.ClearColori(1, color1, ClearColorMask::RGBA);
    queueCB.submit();
    queue->Finish();
}
void LWNRenderTargetAttachmentsTest::DisplayTex(TextureHandle handle, int x, int y, int w, int h) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    g_lwnWindowFramebuffer.bind();
    queueCB.SetViewportScissor(x, y, w, h);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, handle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    queueCB.submit();
    queue->Finish();
}

void LWNRenderTargetAttachmentsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    // sampler
    Sampler* smp;
    LWNuint smpID;
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    smp = sb.CreateSampler();
    smpID = smp->GetRegisteredID();

    // create and fill texture
    Texture* tex[2];
    TextureHandle texHandle[2];
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::COMPRESSIBLE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(texDim, texDim);
    tb.SetFormat(Format::RGBA8I);
    size_t texSize = tb.GetStorageSize();
    size_t texAlignment = tb.GetStorageAlignment();
    texSize = ROUND_UP(texSize, texAlignment);
    MemoryPool* texGpuMemPool = device->CreateMemoryPool(NULL, 2*texSize, MemoryPoolType::GPU_ONLY);
    tex[0] = tb.CreateTextureFromPool(texGpuMemPool, 0);
    tex[1] = tb.CreateTextureFromPool(texGpuMemPool, texSize);
    texHandle[0] = device->GetTextureHandle(tex[0]->GetRegisteredTextureID(), smpID);
    texHandle[1] = device->GetTextureHandle(tex[1]->GetRegisteredTextureID(), smpID);

    // program
    Program *pgm;
    VertexShader vs(440);
    FragmentShader fs(440);
    vs <<
        "out vec2 uv;\n"
        "void main() {\n"
        "  vec2 pos = vec2(0.0);\n"
        "  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);\n"
        "  if (gl_VertexID == 1) pos = vec2(1.0, -1.0);\n"
        "  if (gl_VertexID == 2) pos = vec2(1.0, 1.0);\n"
        "  if (gl_VertexID == 3) pos = vec2(-1.0, 1.0);\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  uv = pos * 0.5 + 0.5;\n"
        "}\n";
    fs <<
        "layout(binding=0) uniform isampler2D tex;\n"
        "in vec2 uv;\n"
        "out vec4 fcolor;"
        "void main() {\n"
        "  fcolor = vec4(texture(tex,uv) / 0x7F);\n"
        "}\n";
    pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // draw
    int cellW = lwrrentWindowWidth / 4;
    int cellH = lwrrentWindowHeight / 2;
    int iGreen[4] = {0, 0x7F, 0, 0x7F};
    int iBlue[4] = {0, 0, 0x7F, 0x7F};
    int iRed[4] = {0x7F, 0, 0, 0x7F};
    int iWhite[4] = {0x7F, 0x7F, 0x7F, 0x7F};

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
    queueCB.submit();
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    // 1. both attached (green, blue)
    BindClearRT(tex[0], iGreen, tex[1], iBlue);
    DisplayTex(texHandle[0], 0, 0, cellW, cellH);
    DisplayTex(texHandle[1], cellW, 0, cellW, cellH);

    // 2. only first attached (white, stays blue)
    BindClearRT(tex[0], iWhite, NULL, iRed);
    DisplayTex(texHandle[0], 2*cellW, 0, cellW, cellH);
    DisplayTex(texHandle[1], 3*cellW, 0, cellW, cellH);

    // 3. only second attached (stays white, green)
    BindClearRT(NULL, iRed, tex[1], iGreen);
    DisplayTex(texHandle[0], 0, cellH, cellW, cellH);
    DisplayTex(texHandle[1], cellW, cellH, cellW, cellH);

    // 4. neither (stays white, stays green)
    BindClearRT(NULL, iRed, NULL, iRed);
    DisplayTex(texHandle[0], 2*cellW, cellH, cellW, cellH);
    DisplayTex(texHandle[1], 3*cellW, cellH, cellW, cellH);

    // cleanup
    tex[0]->Free();
    tex[1]->Free();
    smp->Free();
    texGpuMemPool->Free();
    pgm->Free();
}

OGTEST_CppTest(LWNRenderTargetAttachmentsTest, lwn_rt_attachments, );
