/*
* Copyright (c) 2015 LWPU Corporation.  All rights reserved.
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

class LWNDrawTextureBasicTest {
public:
    enum Variant { DefaultTest, ShaderZTest };
private:
    Variant m_variant;
public:
    LWNDrawTextureBasicTest(Variant variant) : m_variant(variant) {}
    LWNTEST_CppMethods();
};


static Texture* createTexture(int w, int h, MemoryPoolAllocator& alloc)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    TextureBuilder textureBuilder;

    textureBuilder.SetDevice(device).SetDefaults()
        .SetTarget(TextureTarget::TARGET_2D)
        .SetFormat(Format::RGBA8)
        .SetSize2D(w, h);

    Texture *tex = alloc.allocTexture(&textureBuilder);

    const size_t texSize = textureBuilder.GetStorageSize();

    MemoryPoolAllocator pboAllocator(device, NULL, texSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *pbo = pboAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texSize);
    BufferAddress pboAddr = pbo->GetAddress();

    uint8_t *ptr = (uint8_t *)pbo->Map();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x){
            *ptr++ = 255 * (float)y / (float)h;
            *ptr++ = 255 * (float)x / (float)w;
            *ptr++ = 255 * (1 - (float)y / (float)h);
            *ptr++ = 255 * (1 - (float)x / (float)w);
        }
    }

    CopyRegion copyRegion = { 0, 0, 0, w, h, 1 };
    queueCB.CopyBufferToTexture(pboAddr, tex, NULL, &copyRegion, CopyFlags::NONE);

    queueCB.submit();
    queue->Finish();

    return tex;
}

lwString LWNDrawTextureBasicTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "This tests creates a texture and uses DrawTexture to copy it into\n"
        "the render target. The following tests are exelwted:\n"
        "1: Draw the entire texture\n"
        "2: Draw the upper right quarter of the texture\n"
        "3: Draw the texture blended on top of a white background with transparency going from 0 to 1 from left to right.\n"
        "4: Draw the texture with a scissor rectangle. Only the center part of the texture is drawn.\n"
        "5: Draw texture with clamp to border. The source region is extended beyond the upper right corner of the texture.\n"
        "6: Draw texture with clamp to edge. The source region is extended beyond the upper right corner of the texture.\n"
        "7: Draw texture with source coordinates reversed to mirror the texture horizontally and vertically.\n"
        "8: Verifies that the texture can be drawn with a fragment shader bound and verifies that regular 3D rendering works after DrawTexture.\n"
        "   The result is a green quad\n"
        "Tests 1-4 are rendered on the bottom half of the screen; tests 5-8 are rendered on the top half.\n";
    if (m_variant == ShaderZTest) {
        sb <<
            "\nThis test uses a shader that writes to gl_FragDepth to verify that the driver properly "
            "handles this combination of states.";
    }
    return sb.str();
}

int LWNDrawTextureBasicTest::isSupported() const
{
    return g_lwnDeviceCaps.supportsDrawTexture && lwogCheckLWNAPIVersion(40, 18);
}

void LWNDrawTextureBasicTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "out vec4 col;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  col = vec4(color.r, color.g, color.b, 1.0f);\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "in vec4 col;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = col;\n";
    if (m_variant == ShaderZTest) {
        fs << "  gl_FragDepth = 0.75;\n";
    }
    fs << "}\n";

    const LWNuint vertexCount = 4;

    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };

    const Vertex vertices[] = {
        { dt::vec3(-1.0f, -1.0f, 0.0f), dt::vec3(0.0f, 1.0f, 0.0f) },
        { dt::vec3(1.0f, -1.0f, 0.0f), dt::vec3(0.0f, 1.0f, 0.0f) },
        { dt::vec3(-1.0f, 1.0f, 0.0f), dt::vec3(0.0f, 1.0f, 0.0f) },
        { dt::vec3(1.0f, 1.0f, 0.0f), dt::vec3(0.0f, 1.0f, 0.0f) }
    };

    MemoryPoolAllocator bufferAllocator(device, NULL, vertexCount * sizeof(vertices), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, color);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, vertexCount, bufferAllocator, vertices);
    BufferAddress vboAddr = vbo->GetAddress();

    MemoryPoolAllocator texAllocator(device, NULL, 0, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    const LWNuint texWidth = 128;
    const LWNuint texHeight = 128;

    DepthStencilState dss;
    dss.SetDefaults()
        .SetDepthTestEnable(false)
        .SetStencilTestEnable(false);
    queueCB.BindDepthStencilState(&dss);

    BlendState bs;
    bs.SetDefaults();

    ColorState cs;
    cs.SetDefaults();

    Texture *tex = createTexture(texWidth, texHeight, texAllocator);

    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults()
        .SetWrapMode(WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER);

    Sampler *smp = sb.CreateSampler();
    LWNuint smp1ID = smp->GetRegisteredID();

    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);

    Sampler *smp2 = sb.CreateSampler();
    LWNuint smp2ID = smp2->GetRegisteredID();

    LWNuint texID = tex->GetRegisteredTextureID();

    TextureHandle hTex1 = device->GetTextureHandle(texID, smp1ID);
    TextureHandle hTex2 = device->GetTextureHandle(texID, smp2ID);

    static const LWNuint NUM_ROWS = 2;
    static const LWNuint NUM_TESTS_PER_ROW = 4;
    static const LWNuint WIN_RECT_WIDTH = lwrrentWindowWidth / NUM_TESTS_PER_ROW;
    static const LWNuint WIN_RECT_HEIGHT = lwrrentWindowHeight / NUM_ROWS;

    DrawTextureRegion winRect = { 0.0f, 0.0f, (float)WIN_RECT_WIDTH, (float)WIN_RECT_HEIGHT };
    DrawTextureRegion texRect = { 0.0f, 0.0f, (float)texWidth, (float)texHeight };

    queueCB.SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 1.0f, 1.0f, 1.0f, 1.0f);

    // Draw entire texture
    queueCB.DrawTexture(hTex1, &winRect, &texRect);

    // Draw upper right quarter of the texture
    winRect.x0 += WIN_RECT_WIDTH;
    winRect.x1 += WIN_RECT_WIDTH;

    texRect.x0 = texWidth * 0.5f;
    texRect.y0 = texHeight * 0.5f;
    texRect.x1 = texWidth;
    texRect.y1 = texHeight;
    queueCB.DrawTexture(hTex1, &winRect, &texRect);

    texRect.x0 = 0.0f;
    texRect.y0 = 0.0f;
    texRect.x1 = texWidth;
    texRect.y1 = texHeight;

    // Draw entire texture with blending enabled
    winRect.x0 += WIN_RECT_WIDTH;
    winRect.x1 += WIN_RECT_WIDTH;

    cs.SetBlendEnable(0, true);
    queueCB.BindColorState(&cs);

    bs.SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA, BlendFunc::ONE, BlendFunc::ONE);
    bs.SetBlendEquation(BlendEquation::ADD, BlendEquation::ADD);
    queueCB.BindBlendState(&bs);

    queueCB.DrawTexture(hTex1, &winRect, &texRect);

    cs.SetBlendEnable(0, false);
    queueCB.BindColorState(&cs);

    // Set viewport and scissor rectangle. The viewport should not affect DrawTexture
    // but the scissor rectangle should affect DrawTexture
    winRect.x0 += WIN_RECT_WIDTH;
    winRect.x1 += WIN_RECT_WIDTH;

    queueCB.SetViewport(0, 0, 0, 0);
    queueCB.SetScissor(winRect.x0 + WIN_RECT_WIDTH / 4, winRect.y0 + WIN_RECT_HEIGHT / 4, WIN_RECT_WIDTH / 2, WIN_RECT_HEIGHT / 2);

    queueCB.DrawTexture(hTex1, &winRect, &texRect);

    queueCB.SetScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);

    // Draw texture and test clamp to border
    winRect.x0 = 0;
    winRect.y0 = WIN_RECT_HEIGHT;
    winRect.x1 = WIN_RECT_WIDTH;
    winRect.y1 += WIN_RECT_HEIGHT;

    texRect.x1 = texWidth * 1.5;
    texRect.y1 = texWidth * 1.5;

    queueCB.DrawTexture(hTex1, &winRect, &texRect);

    // Draw texture and test clamp to edge
    winRect.x0 += WIN_RECT_WIDTH;
    winRect.x1 += WIN_RECT_WIDTH;

    queueCB.DrawTexture(hTex2, &winRect, &texRect);

    texRect.x1 = texWidth;
    texRect.y1 = texHeight;

    // Test restore of the shaders
    winRect.x0 += WIN_RECT_WIDTH;
    winRect.x1 += WIN_RECT_WIDTH;

    Program *program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        LWNFailTest();
    }
    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);

    texRect.x0 = texWidth;
    texRect.y0 = texHeight;
    texRect.x1 = 0;
    texRect.y1 = 0;

    queueCB.DrawTexture(hTex1, &winRect, &texRect);

    texRect.x0 = 0.0f;
    texRect.y0 = 0.0f;
    texRect.x1 = texWidth;
    texRect.y1 = texHeight;

    winRect.x0 += WIN_RECT_WIDTH;
    winRect.x1 += WIN_RECT_WIDTH;

    queueCB.SetViewport(winRect.x0, winRect.y0, WIN_RECT_WIDTH, WIN_RECT_HEIGHT);
    queueCB.SetScissor(winRect.x0, winRect.y0, WIN_RECT_WIDTH, WIN_RECT_HEIGHT);

    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertices));

    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, vertexCount);

    queueCB.submit();

    queue->Finish();

    program->Free();
}

OGTEST_CppTest(LWNDrawTextureBasicTest, lwn_draw_texture, (LWNDrawTextureBasicTest::DefaultTest));
OGTEST_CppTest(LWNDrawTextureBasicTest, lwn_draw_texture_shdz, (LWNDrawTextureBasicTest::ShaderZTest));
