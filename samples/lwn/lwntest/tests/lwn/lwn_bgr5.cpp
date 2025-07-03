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

using namespace lwn;

class LWNColorBGRTest
{
    // Draw 192x192 textures in 200x200 cells.
    static const int texSize = 192;
    static const int cellSize = 200;
    static const int cellMargin = 4;

public:
    LWNTEST_CppMethods();
};

lwString LWNColorBGRTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic test of the BGR565, BGR5, and BGR5A1 formats in LWN.  This test "
        "draws six cells.  The bottom rows render a triangle into a BGR565, "
        "BGR5, and BGR5A1 texture (left to right).  The top rows make a copy "
        "of the contents of the bottom rows.  The background of the test is "
        "dark blue.  The texture colors are adjusted using alpha values, which makes "
        "the background of the BGR5A1 column darker because the background is "
        "cleared with alpha==0.";
    return sb.str();
}

int LWNColorBGRTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 10);
}

void LWNColorBGRTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Simple program to render shaded primitives.
    VertexShader vsColor(440);
    vsColor <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fsColor(440);
    fsColor <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    Program *colorPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(colorPgm, vsColor, fsColor)) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
    }

    // Simple program to display texture mapped primitives.  We scale the
    // colors by (alpha+1)/2 to make "transparent" parts of RGB5A1 darker.
    VertexShader vsTex(440);
    vsTex <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec2 tc;\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  otc = tc;\n"
        "}\n";
    FragmentShader fsTex(440);
    fsTex <<
        "layout(binding = 0) uniform sampler2D smp;\n"
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec4 texel = texture(smp, otc);\n"
        "  fcolor = texel * (0.5 + 0.5 * texel.w);\n"
        "}\n";

    Program *texPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(texPgm, vsTex, fsTex)) {
        printf("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog());
    }


    struct ColorVertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const ColorVertex colorVertexData[] = {
        { dt::vec3(-0.8, -0.8, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.8, +0.8, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.8, -0.8, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    struct TextureVertex {
        dt::vec3 position;
        dt::vec2 tc;
    };
    static const TextureVertex texVertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec2(0.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec2(0.0, 1.0) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec2(1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec2(1.0, 1.0) },
    };

    MemoryPoolAllocator vboAllocator(device, NULL, 64 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream colorStream(sizeof(ColorVertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(colorStream, ColorVertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(colorStream, ColorVertex, color);
    VertexArrayState colorVertex = colorStream.CreateVertexArrayState();
    Buffer *colorVBO = colorStream.AllocateVertexBuffer(device, 3, vboAllocator, colorVertexData);
    BufferAddress colorVBOAddr = colorVBO->GetAddress();

    VertexStream textureStream(sizeof(TextureVertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(textureStream, TextureVertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(textureStream, TextureVertex, tc);
    VertexArrayState textureVertex = textureStream.CreateVertexArrayState();
    Buffer *textureVBO = textureStream.AllocateVertexBuffer(device, 4, vboAllocator, texVertexData);
    BufferAddress textureVBOAddr = textureVBO->GetAddress();

    // Simple sampler used for all textures.
    SamplerBuilder sb;
    sb.SetDefaults().SetDevice(device);
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    Sampler *smp = sb.CreateSampler();

    MemoryPoolAllocator texAllocator(device, NULL, 16 * 1024 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    TextureBuilder tb;
    tb.SetDefaults().SetDevice(device);
    tb.SetLevels(1);
    tb.SetSize2D(texSize, texSize);
    tb.SetTarget(TextureTarget::TARGET_2D);

    // Initialize two textures for each of the BGR5 formats.
    Format formats[3] = { Format::BGR565, Format::BGR5, Format::BGR5A1 };
    Texture *textures[3][2];
    TextureHandle texHandles[3][2];
    for (int i = 0; i < 3; i++) {
        tb.SetFormat(formats[i]);
        for (int j = 0; j < 2; j++) {
            textures[i][j] = texAllocator.allocTexture(&tb);
            texHandles[i][j] = device->GetTextureHandle(textures[i][j]->GetRegisteredTextureID(),
                                                        smp->GetRegisteredID());
        }
    }

    queueCB.Barrier(BarrierBits::ILWALIDATE_TEXTURE_DESCRIPTOR);

    CopyRegion cr = { 0, 0, 0, texSize, texSize, 1 };

    // Render into one texture of each format and then copy into the second
    // texture of the sample format.
    queueCB.BindProgram(colorPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(colorVertex);
    queueCB.BindVertexBuffer(0, colorVBOAddr, sizeof(colorVertexData));
    queueCB.SetViewportScissor(0, 0, texSize, texSize);
    for (int i = 0; i < 3; i++) {
        queueCB.SetRenderTargets(1, &textures[i][0], NULL, NULL, NULL);
        queueCB.ClearColor(0, 0.0, 0.0, 0.6, 0.0);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        queueCB.CopyTextureToTexture(textures[i][0], NULL, &cr, textures[i][1], NULL, &cr, CopyFlags::NONE);
    }

    // Wait for the texture rendering to complete.
    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

    // Clear the window.
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    // Display each texture in its own cell on-screen.
    queueCB.BindProgram(texPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(textureVertex);
    queueCB.BindVertexBuffer(0, textureVBOAddr, sizeof(texVertexData));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandles[i][j]);
            queueCB.SetViewportScissor(i * cellSize + cellMargin, j * cellSize + cellMargin,
                                       cellSize - 2 * cellMargin, cellSize - 2 * cellMargin);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        }
    }

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNColorBGRTest, lwn_bgr5, );
