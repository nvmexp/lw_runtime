/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

using namespace lwn;

class LWNTestZlwllStencil
{
public:
    LWNTEST_CppMethods();

private:
    static const int    CELL_WIDTH = 80;
    static const int    CELL_HEIGHT = 80;

    void drawResults(QueueCommandBuffer &queueCB, Texture *tex, int x, int y, int w, int h, bool success) const;
};

int LWNTestZlwllStencil::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 207);
}

lwString LWNTestZlwllStencil::getDescription() const
{
    return "Test to verify the lwnCommandBufferSetStencilLwllCriteria function. "
           "This test first renders a small quad into the stencil buffer, then a larger "
           "quad is drawn with the stencil test enabled. Fragments that cover the area of "
           "the small quad will fail the test. This is done twice for the different stencil "
           "formats and functions. In the first run stencil lwlling is disabled and no pixels "
           "should be discarded by ZLwll. In the second run stencil lwlling is enabled and the "
           "ZLwll criteria is set to match the depth stencil state. In this run pixels should "
           "be lwlled by ZLwll. After each run the performance counter for ZLwll are checked to "
           "verify that the function worked as expected.\n";
}

void LWNTestZlwllStencil::drawResults(QueueCommandBuffer &queueCB, Texture *tex, int x, int y, int w, int h, bool success) const
{
    if (success) {
        CopyRegion srcReg = {0, 0, 0, w, h, 1};
        CopyRegion dstReg = {x, y, 0, CELL_WIDTH, CELL_HEIGHT, 1};

        queueCB.CopyTextureToTexture(tex, NULL, &srcReg, g_lwnWindowFramebuffer.getAcquiredTexture(), NULL, &dstReg, CopyFlags::LINEAR_FILTER);
    } else {
        const float failColor[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
        queueCB.SetViewportScissor(x, y, CELL_WIDTH, CELL_HEIGHT);
        queueCB.ClearColor(0, failColor, ClearColorMask::RGBA);
    }
}


void LWNTestZlwllStencil::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec4 position;\n"
        "layout(binding = 0) uniform Block {\n"
        "    vec4 scale;\n"
        "};\n"
        "void main() {\n"
        "  gl_Position = position * vec4(scale.xyz, 1.0f);\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(0.0f, 1.0f, 0.0f, 1.0f);\n"
        "}\n";

    Program *program = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        DEBUG_PRINT("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog());

        LWNFailTest();
        return;
    }

    MemoryPoolAllocator bufferAllocator(device, NULL, 512, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    MemoryPoolAllocator texAllocator(device, NULL, (64<<10), LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    const int vertexCount = 4;

    struct Vertex {
        dt::vec3 position;
    };

    const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertexState = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, vertexCount, bufferAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    BufferBuilder bb;
    bb.SetDefaults().SetDevice(device);

    struct Uniforms {
        dt::vec4 scale;
    };

    Buffer *ubo = bufferAllocator.allocBuffer(&bb, BufferAlignBits::BUFFER_ALIGN_UNIFORM_BIT, sizeof(Uniforms));
    BufferAddress uboGpuAddr = ubo->GetAddress();
    Uniforms *uboCpuAddr = static_cast<Uniforms*>(ubo->Map());

    uboCpuAddr->scale = dt::vec4(1.0f);

    struct ZLwllCounter
    {
        uint32_t zlwll0;    // Nubers of tiles processed by ZLwll
        uint32_t zlwll1;    // Number of 4x2 pixel blocks lwlled due to failing the depth test
        uint32_t zlwll2;    // Number of 8x8 pixel blocks lwlled because they were in front of previous primitives
        uint32_t zlwll3;    // Number of 4x4 pixel blocks that were lwlled due to failing the stencil test
    };

    Buffer *countBuffer = bufferAllocator.allocBuffer(&bb, BufferAlignBits::BUFFER_ALIGN_COUNTER_BIT, sizeof(ZLwllCounter));
    BufferAddress counterGpuAddr = countBuffer->GetAddress();
    ZLwllCounter* counterCpuAddr = static_cast<ZLwllCounter*>(countBuffer->Map());

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.2f, 0.2f, 0.2f, 1.0f, ClearColorMask::RGBA);

    DepthStencilState ds;

    ds.SetDefaults().
       SetDepthWriteEnable(LWN_FALSE).
       SetStencilTestEnable(LWN_TRUE);

    queueCB.BindDepthStencilState(&ds);

    ChannelMaskState cs;
    cs.SetDefaults();

    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboGpuAddr, sizeof(Uniforms));

    const int fboWidth = 640;
    const int fboHeight = 480;

    TextureBuilder tb;

    tb.SetDefaults().SetDevice(device).
       SetTarget(TextureTarget::TARGET_2D).
       SetFormat(Format::RGBA8).
       SetFlags(TextureFlags::COMPRESSIBLE).
       SetSize2D(fboWidth, fboHeight);

    Texture *colorTex = texAllocator.allocTexture(&tb);

    int cellX = 0;
    int cellY = 0;

    static const Format stencilFormats[] = {
        Format::STENCIL8,
        Format::DEPTH24_STENCIL8,
        Format::DEPTH32F_STENCIL8
    };

    struct StencilParam {
        int             ref;        // Reference value used for stencil test
        int             clear;      // Value used to clear the stencil buffer
        int             fail;       // Value used to make the stencil test fail.
        int             mask;       // Mask used for the stencil test
        StencilFunc     func;       // Function used for the stencil test.
    };

    // Stencil parameters used for the tests.
    StencilParam sp[] = {
        { 0x80, 0xFF, 0x10, 0xFF, StencilFunc::LESS     },
        { 0x80, 0x80, 0xF0, 0xFF, StencilFunc::EQUAL    },
        { 0x80, 0x80, 0x10, 0xFF, StencilFunc::LEQUAL   },
        { 0xF0, 0x80, 0xFF, 0xFF, StencilFunc::GREATER  },
        { 0x80, 0xFF, 0x80, 0xFF, StencilFunc::NOTEQUAL },
        { 0xF0, 0x80, 0xFF, 0xFF, StencilFunc::GEQUAL   },
    };

    for (size_t i = 0; i < __GL_ARRAYSIZE(stencilFormats); ++i) {

        if ((stencilFormats[i] == Format::STENCIL8) && !g_lwnDeviceCaps.supportsStencil8) {
            continue;
        }

        tb.SetFormat(stencilFormats[i]).
           SetFlags(TextureFlags::COMPRESSIBLE);

        for (size_t j = 0; j < __GL_ARRAYSIZE(sp); ++j) {
            for (int k = 0; k < 2; ++k) {
                if (k & 1) {
                    // Enable stencil lwlling in ZLwll
                    tb.SetFlags(tb.GetFlags() | TextureFlags::ZLWLL_SUPPORT_STENCIL);
                    // Set Lwll criteria that matches stencil state that will be used.
                    queueCB.SetStencilLwllCriteria(sp[j].func, sp[j].ref, sp[j].mask);
                }

                Texture *depthTex = texAllocator.allocTexture(&tb);

                queueCB.SetRenderTargets(1, &colorTex, NULL, depthTex, NULL);
                queueCB.SetViewportScissor(0, 0, fboWidth, fboHeight);

                const float clearColor[4] = { 0.0f, 0.0f, 0.8f, 1.0f };
                queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
                queueCB.ClearDepthStencil(1.0f, LWN_TRUE, sp[j].clear, sp[j].mask);

                // Render small quad to stencil buffer. Fragments coverd by this quad
                // will have a stencil value defined by sp[j].fail.
                dt::vec4 scale(0.5f);
                queueCB.UpdateUniformBuffer(uboGpuAddr, sizeof(Uniforms), 0, sizeof(Uniforms), &scale);

                queueCB.SetStencilMask(Face::FRONT_AND_BACK, sp[j].mask);
                queueCB.SetStencilRef(Face::FRONT_AND_BACK, sp[j].fail);

                ds.SetStencilFunc(Face::FRONT_AND_BACK, StencilFunc::ALWAYS).
                   SetStencilOp(Face::FRONT_AND_BACK, StencilOp::KEEP, StencilOp::KEEP, StencilOp::REPLACE);

                queueCB.BindDepthStencilState(&ds);

                cs.SetChannelMask(0, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE);
                queueCB.BindChannelMaskState(&cs);

                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, vertexCount);

                // Render larger quad to color buffer with stencil test enabled. The region
                // covered by the smaller quad will not be drawn, since these fragments should
                // fail the stencil test.
                scale = dt::vec4(0.8f);
                queueCB.UpdateUniformBuffer(uboGpuAddr, sizeof(Uniforms), 0, sizeof(Uniforms), &scale);

                queueCB.SetStencilRef(Face::FRONT_AND_BACK, sp[j].ref);

                ds.SetStencilFunc(Face::FRONT_AND_BACK, sp[j].func).
                   SetStencilOp(Face::FRONT_AND_BACK, StencilOp::KEEP, StencilOp::KEEP, StencilOp::KEEP);

                queueCB.BindDepthStencilState(&ds);

                cs.SetChannelMask(0, LWN_TRUE, LWN_TRUE, LWN_TRUE, LWN_TRUE);
                queueCB.BindChannelMaskState(&cs);

                // Make sure stencil buffer is entirely updated by the previous draw.
                // Note: ORDER_FRAGMENTS does not give this guarantee since it allows
                // the GPU to do stencil tests even if previous commands did not
                // complete.
                queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES);

                queueCB.ResetCounter(CounterType::ZLWLL_STATS);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, vertexCount);
                queueCB.ReportCounter(CounterType::ZLWLL_STATS, counterGpuAddr);

                queueCB.submit();
                queue->Finish();

                bool success = (k & 1)? (counterCpuAddr->zlwll3 > 0) : (counterCpuAddr->zlwll3 == 0);

                g_lwnWindowFramebuffer.bind();
                drawResults(queueCB, colorTex, cellX, cellY, fboWidth, fboHeight, success);

                cellX += CELL_WIDTH;
                if (cellX >= lwrrentWindowWidth) {
                    cellX = 0;
                    cellY += CELL_HEIGHT;
                }

                texAllocator.freeTexture(depthTex);
            } // for (int k = 0; k < 2; ++k)
        } // for (size_t j = 0; j < __GL_ARRAYSIZE(sp); ++j)
    }  // for (size_t i = 0; i < __GL_ARRAYSIZE(stencilFormats); ++i)

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    ds.SetDefaults();
    queueCB.BindDepthStencilState(&ds);

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNTestZlwllStencil, lwn_zlwll_stencil, );