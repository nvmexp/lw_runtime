/*
 * Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

class LWNTestZlwllZF32
{
    void drawResult(QueueCommandBuffer &queueCB, lwn::Texture* tex, const CopyRegion& dstRegion) const;
    void runTest() const;

    void updateCellCoords(int& x, int& y) const;

    static const int CELL_WIDTH  = 160;
    static const int CELL_HEIGHT = 120;

public:
    LWNTEST_CppMethods();
};

namespace
{
    class ScopedFBO : public Framebuffer
    {
    public:
        ScopedFBO(lwn::Device& dev, int w, int h, lwn::Format cf = lwn::Format::RGBA8, lwn::Format df = lwn::Format::DEPTH32F) : Framebuffer(w, h)
        {
            setColorFormat(lwn::Format::RGBA8);
            setDepthStencilFormat(lwn::Format::DEPTH32F);
            alloc(&dev);
        }

        ~ScopedFBO()
        {
            destroy();
        }
    };
}

int LWNTestZlwllZF32::isSupported() const
{
    return lwogCheckLWNAPIVersion(55, 11);
}

lwString LWNTestZlwllZF32::getDescription() const
{
    return "Basic test to verify the SetZLwllZF32CompressionEnable function for DEPTH32F "
        "depth buffer. The test first renders a smaller green quad, then it renders a "
        "larger blue quad behind the green one. If ZLwll works as expected it will lwll "
        "blocks of the blue quad that are covered by the green one. The z distance "
        "between the two quads is very small and and ZLwll will fail if a "
        "non optimal combination of depth function, clear value and ZF32 compression "
        "is selected.";
}

void LWNTestZlwllZF32::updateCellCoords(int& x, int& y) const
{
    x += CELL_WIDTH;

    if (x >= lwrrentWindowWidth) {
        x = 0;
        y += CELL_HEIGHT;
    }
}

void LWNTestZlwllZF32::drawResult(QueueCommandBuffer &queueCB, lwn::Texture* tex, const CopyRegion& dstRegion) const
{
    lwn::CopyRegion srcRegion = { 0, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight, 1 };

    queueCB.CopyTextureToTexture(tex, NULL, &srcRegion,
                                 g_lwnWindowFramebuffer.getAcquiredTexture(), NULL, &dstRegion,
                                 CopyFlags::NONE);
}

void LWNTestZlwllZF32::runTest() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(460);
    vs <<
        "layout(location = 0) in  vec4 position;\n"
        "layout(location = 0) out vec4 frag_color;\n"
        "\n"
        "const vec4 color[2] = {vec4(0.0f, 1.0f, 0.0f, 1.0f),\n"
        "                       vec4(0.0f, 0.0f, 1.0f, 1.0f)};\n"
        "\n"
        "const float scale[2] = { 0.8f, 1.0f };\n"
        "\n"
        "layout(binding = 0) uniform Block {\n"
        "    float zvalues[2];\n"
        "};\n"
        "\n"
        "void main() {\n"
        "  int idx = gl_BaseInstance + gl_InstanceID;\n"
        "  gl_Position = vec4(position.xy * scale[idx], zvalues[idx], 1.0f);\n"
        "  frag_color = color[idx];\n"
        "}\n";

    FragmentShader fs(460);
    fs <<
        "layout(location = 0) in  vec4 frag_color;\n"
        "layout(location = 0) out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = frag_color;\n"
        "}\n";

    Program *program = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        DEBUG_PRINT("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog());

        LWNFailTest();
        return;
    }

    MemoryPoolAllocator bufferAllocator(device, NULL, 512, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    const int vertexCount = 4;

    struct Vertex {
        dt::vec2 position;
    };
    const Vertex vertexData[] = {
        { dt::vec2(-1.0, -1.0) },
        { dt::vec2(+1.0, -1.0) },
        { dt::vec2(-1.0, +1.0) },
        { dt::vec2(+1.0, +1.0) },
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertexState = stream.CreateVertexArrayState();

    Buffer *vbo = stream.AllocateVertexBuffer(device, vertexCount, bufferAllocator, vertexData);
    BufferAddress vboGpuAddr = vbo->GetAddress();

    struct Uniforms {
        float zvalues[2];
    };

    BufferBuilder bb;
    bb.SetDefaults().SetDevice(device);
    Buffer *ubo = bufferAllocator.allocBuffer(&bb, BufferAlignBits::BUFFER_ALIGN_UNIFORM_BIT, sizeof(Uniforms));
    BufferAddress uboGpuAddr = ubo->GetAddress();
    Uniforms *uboCpuAddr = static_cast<Uniforms*>(ubo->Map());

    uboCpuAddr->zvalues[0] = -2.0f;
    uboCpuAddr->zvalues[1] = -2.0f;

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

    // Chose two z values that are close to each other. If the precision of the ZLwll format
    // is not good enough, ZLwll will fail and no blocks will be lwlled. If the right format
    // is selected, precision is good enough and ZLwll will succeed.
    const float z0 = 0.001f, z1 = 0.0005f;

    g_lwnWindowFramebuffer.bind();
    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);
    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboGpuAddr, sizeof(vertexData));
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboGpuAddr, sizeof(Uniforms));

    int cellX = 0, cellY = 0;

    struct TestCase
    {
        enum ZF32Mode {ZF32_AS_Z16_ON = 0, ZF32_AS_Z16_OFF = 1};

        float           zValues[2];
        float           depthClearValue;
        DepthFunc       depthFunction;
        ZF32Mode        expectedToBeBetter;
    };

    // The test cases will verify the ZLwll performance as descrived in http://lwbugs/2757650/112
    const TestCase cases[] = {
        // ZF32, LESS: Best near 1.0 with ZF32_AS_Z16 on, otherwise worse near 1.0
        {{1.0f - z0, 1.0f - z1},     1.0f,   DepthFunc::LESS,    TestCase::ZF32Mode::ZF32_AS_Z16_ON},
        // MSB, LESS: moderate precision 0.25-1 with ZF32_AS_Z16 on, otherwise excellent near 0.0
        {{0.9f - z0, 0.9f - z1},     0.99f,  DepthFunc::LESS,    TestCase::ZF32Mode::ZF32_AS_Z16_ON},
        // ZF32, GREATER: Bad < 0.25 with ZF32_AS_Z16 on, otherwise excellent near 0.0
        {{z0, z1},                   0.0f,   DepthFunc::GREATER, TestCase::ZF32Mode::ZF32_AS_Z16_OFF},
        // MSB, GREATER: Bad < 0.25 with ZF32_AS_Z16 on, otherwise excellent near 0.0
        {{0.001f + z0, 0.001f + z1}, 0.001f, DepthFunc::GREATER, TestCase::ZF32Mode::ZF32_AS_Z16_OFF},
    };

    const int numCases = sizeof(cases) / sizeof(TestCase);

    uint32_t counter[2] = {};

    DepthStencilState ds;

    ds.SetDefaults().
       SetDepthWriteEnable(LWN_TRUE).
       SetDepthTestEnable(LWN_TRUE);

    ScopedFBO fbo(*device, lwrrentWindowWidth, lwrrentWindowHeight);
    fbo.bind(queueCB);

    for (int i = 0; i < numCases; ++i) {
        uboCpuAddr->zvalues[0] = cases[i].zValues[0];
        uboCpuAddr->zvalues[1] = cases[i].zValues[1];

        ds.SetDepthFunc(cases[i].depthFunction);
        queueCB.BindDepthStencilState(&ds);

        const int lwrrentCellX = cellX, lwrrentCellY = cellY;

        for (int n = 0; n < 2; ++n) {
            queueCB.SetZLwllZF32CompressionEnable((n != 1));

            queueCB.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);
            queueCB.ClearDepthStencil(cases[i].depthClearValue, LWN_TRUE, 0, 0);

            queueCB.ResetCounter(CounterType::ZLWLL_STATS);

            queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 0, 1);
            queueCB.Barrier(LWN_BARRIER_ORDER_PRIMITIVES_BIT);
            queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 1, 1);
            queueCB.Barrier(LWN_BARRIER_ORDER_PRIMITIVES_BIT);

            queueCB.ReportCounter(CounterType::ZLWLL_STATS, counterGpuAddr);

            queueCB.Barrier(lwn::BarrierBits::ILWALIDATE_ZLWLL);

            drawResult(queueCB, fbo.getColorTexture(0), { cellX, cellY, 0, CELL_WIDTH, CELL_HEIGHT, 1 });
            updateCellCoords(cellX, cellY);

            queueCB.submit();
            queue->Finish();

            counter[n] = counterCpuAddr->zlwll1;
        }

        if (counter[cases[i].expectedToBeBetter] <= counter[1 - cases[i].expectedToBeBetter]) {
            g_lwnWindowFramebuffer.bind();

            queueCB.SetViewportScissor(lwrrentCellX, lwrrentCellY, CELL_WIDTH, CELL_HEIGHT);
            queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
            queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);

            DEBUG_PRINT("\nTest failed. i=%d comparing ZLWLL stats n=0.Word1 with n=1.Word1: Counter 0: %d, Counter 1: %d, expected to be better: %d\t z0=%.32f z1=%.32f\n",
                        i, counter[0], counter[1], cases[i].expectedToBeBetter, z0, z1);

            fbo.bind(queueCB);
        } else {
            DEBUG_PRINT("\nTest succeed. i=%d comparing ZLWLL stats n=0.Word1 with n=1.Word1: Counter 0: %d, Counter 1: %d, expected to be better: %d\t z0=%.32f z1=%.32f\n",
                        i, counter[0], counter[1], cases[i].expectedToBeBetter, z0, z1);
        }
    }

    // Save/Restore test
    {
        ScopedFBO fbo[2] = { ScopedFBO(*device, lwrrentWindowWidth, lwrrentWindowHeight),
                             ScopedFBO(*device, lwrrentWindowWidth, lwrrentWindowHeight) };

        uboCpuAddr->zvalues[0] = z0;
        uboCpuAddr->zvalues[1] = z1;

        // Setup ZLwll storage buffer
        LWNint zlwllBufAlignment = 0;
        device->GetInteger(DeviceInfo::ZLWLL_SAVE_RESTORE_ALIGNMENT, &zlwllBufAlignment);

        const size_t zlwllBufferSize = fbo[0].getDepthTexture()->GetZLwllStorageSize();
        Buffer *zlwllBuffer = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_ZLWLL_SAVE_BIT, zlwllBufferSize + zlwllBufAlignment);
        memset(zlwllBuffer->Map(), 0, zlwllBufferSize);

        // Bind first FBO and draw smaller green quad.
        fbo[0].bind(queueCB);
        queueCB.SetZLwllZF32CompressionEnable(false);

        ds.SetDepthFunc(DepthFunc::GREATER);
        queueCB.BindDepthStencilState(&ds);

        queueCB.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);
        queueCB.ClearDepthStencil(0.0f, LWN_TRUE, 0, 0);

        // Draw green quad and fill ZLwll buffer.
        queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 0, 1);

        // Stote ZLwll buffer with depth values of first quad
        queueCB.SaveZLwllData(zlwllBuffer->GetAddress(), zlwllBufferSize);

        // Bind second FBO this will clear the ZLwll buffer
        fbo[1].bind(queueCB);

        ds.SetDepthFunc(DepthFunc::LESS);
        queueCB.BindDepthStencilState(&ds);

        queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0, 0);

        uboCpuAddr->zvalues[0] = 1.0f - z0;
        uboCpuAddr->zvalues[1] = 1.0f - z1;

        queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 0, 2);

        // Bind first FBO again.
        fbo[0].bind(queueCB);

        uboCpuAddr->zvalues[0] = z0;
        uboCpuAddr->zvalues[1] = z1;

        ds.SetDepthFunc(DepthFunc::GREATER);
        queueCB.BindDepthStencilState(&ds);

        // Draw second blue quad behind the initial green quad. Since the ZLwll
        // buffer was not restored, it is expected that Zlwll does not work.
        queueCB.ResetCounter(CounterType::ZLWLL_STATS);

        queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 1, 1);
        queueCB.Barrier(LWN_BARRIER_ORDER_PRIMITIVES_BIT);

        queueCB.ReportCounter(CounterType::ZLWLL_STATS, counterGpuAddr);

        queueCB.submit();
        queue->Finish();

        counter[0] = counterCpuAddr->zlwll1;

        // Restore ZLwll data and draw the second quad again. Since data was
        // restored it is expected thet ZLwll works.
        queueCB.RestoreZLwllData(zlwllBuffer->GetAddress(), zlwllBufferSize);

        queueCB.ResetCounter(CounterType::ZLWLL_STATS);

        queueCB.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 1, 1);
        queueCB.Barrier(LWN_BARRIER_ORDER_PRIMITIVES_BIT);

        queueCB.ReportCounter(CounterType::ZLWLL_STATS, counterGpuAddr);

        drawResult(queueCB, fbo[0].getColorTexture(0), { cellX, cellY, 0, CELL_WIDTH, CELL_HEIGHT, 1 });

        queueCB.submit();
        queue->Finish();

        counter[1] = counterCpuAddr->zlwll1;

        if (counter[0] >= counter[1]) {
            g_lwnWindowFramebuffer.bind();

            queueCB.SetViewportScissor(cellX, cellY, CELL_WIDTH, CELL_HEIGHT);
            queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
            queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);

            DEBUG_PRINT("\nTest failed save/restore. Counter 0: %d, Counter 1: %d, expected to be better: 1 \n",
                        counter[0], counter[1]);
        } else {
            DEBUG_PRINT("\nTest succeed save/restore. Counter 0: %d, Counter 1: %d, expected to be better: 1 \n",
                        counter[0], counter[1]);
        }
    }

    g_lwnWindowFramebuffer.bind();

    ds.SetDefaults();
    queueCB.BindDepthStencilState(&ds);

    g_lwnWindowFramebuffer.setViewportScissor();

    queueCB.submit();
    queue->Finish();
}

void LWNTestZlwllZF32::doGraphics() const
{
    DisableLWNObjectTracking();
    {
        DeviceState deviceStateNearIsZero(LWNdeviceFlagBits(0),
                                          LWN_WINDOW_ORIGIN_MODE_LOWER_LEFT,
                                          LWN_DEPTH_MODE_NEAR_IS_ZERO,
                                          LWNqueueFlags(0));

        deviceStateNearIsZero.SetActive();
        runTest();
    }
    EnableLWNObjectTracking();

    DeviceState::SetDefaultActive();
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNTestZlwllZF32, lwn_zlwll_zf32, );

