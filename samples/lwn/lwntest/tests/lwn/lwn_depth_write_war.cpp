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

//Test for driver workaround for bug 1703533

class LWNDepthWriteWAR
{
public:
    LWNTEST_CppMethods();

    static const int fboWidth  = 128;
    static const int fboHeight = 128;

    void showFbo(QueueCommandBuffer &queueCB, Texture *tex, int srcW, int srcH, int dstIdx) const;
};

lwString LWNDepthWriteWAR::getDescription() const
{
    lwStringBuf sb;
    sb << "Repro and test for bug 1703533 (depth write enable doesn't get updated after certain methods).\n"
          "\n"
          "The left subtest renders two triangles. First a smaller red one with closer depth values, then a\n"
          "larger green triangle with farther depth values. If depth write state is updated correctly,\n"
          "the red triangle appears on top of the green triangle, otherwise only the green triangle\n"
          "is visible.\n"
          "The right subtest renders the same triangles using S8 format depth stencil buffer. The first"
          "smaller triangle sets stencil values to 1, and the second larger triangle is visible only with"
          "pixels with stencil value 1. Thus the correct image has a small green triangle.";
    return sb.str();
}

int LWNDepthWriteWAR::isSupported() const
{
    return lwogCheckLWNAPIVersion(26, 1) && g_lwnDeviceCaps.supportsStencil8;
}

void LWNDepthWriteWAR::showFbo(QueueCommandBuffer &queueCB, Texture *tex, int srcW, int srcH, int dstIdx) const
{
    CopyRegion srcRegion = { 0, 0, 0, srcW, srcH, 1 };

    assert(dstIdx < 4);

    int dstX = (dstIdx & 1) * lwrrentWindowWidth / 2;
    int dstY = ((dstIdx>>1) & 1) * lwrrentWindowHeight / 2;
    CopyRegion dstRegion = { dstX, dstY, 0, lwrrentWindowWidth / 2, lwrrentWindowHeight / 2, 1 };

    queueCB.CopyTextureToTexture(tex, NULL, &srcRegion,
                                 g_lwnWindowFramebuffer.getAcquiredTexture(), NULL, &dstRegion,
                                 CopyFlags::NONE);
}

void LWNDepthWriteWAR::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(binding = 0) uniform segmentBlock {\n"
        "  vec4 scale;\n"
        "  vec4 offset;\n"
        "  vec4 color;\n"
        "};\n"
        "out vec4 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position*scale.xyz, 1.0) + offset;\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec4 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = ocolor;\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    // Compile and call lwnProgramSetShaders.
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };

    struct Uniforms {
        dt::vec4 scale;
        dt::vec4 offset;
        dt::vec4 color;
    };

    static const Vertex vertexData[] = {
        { dt::vec3(-0.5, -0.5, 0.5) },
        { dt::vec3(-0.5, +0.5, 0.5) },
        { dt::vec3(+0.5, -0.5, 0.5) },
    };

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Create a vertex buffer and fill it with data
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();
    Buffer *ubo = allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_UNIFORM_BIT, sizeof(Uniforms));
    // Get a handle to be used for setting the buffer as a uniform buffer
    BufferAddress uboAddr = ubo->GetAddress();
    void *uboMem = (void *)ubo->Map();
    memset(uboMem, 0, sizeof(Uniforms));
    Uniforms *uboCpuVa = (Uniforms *)uboMem;

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(Uniforms));

    Framebuffer fbos[] = {
        Framebuffer(fboWidth, fboHeight),
        Framebuffer(fboWidth, fboHeight)
    };
    const int NUM_FBOS = sizeof(fbos)/sizeof(fbos[0]);

    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].setColorFormat(0, Format::RGBA8);
        if (i == 0)
            fbos[i].setDepthStencilFormat(Format::STENCIL8);
        else
            fbos[i].setDepthStencilFormat(Format::DEPTH24);

        fbos[i].alloc(device);
    }

    // Repro for Case 1 (buggy SetZtFormat)
    // Test that depth writes get enabled after a switch from
    // S8 to a depth format.

    DepthStencilState depthState;
    depthState.SetDefaults()
        .SetDepthTestEnable(LWN_TRUE)
        .SetDepthWriteEnable(LWN_TRUE)
        .SetDepthFunc(DepthFunc::LEQUAL);
    queueCB.BindDepthStencilState(&depthState);

    Texture *color = fbos[0].getColorTexture(0);
    queueCB.SetRenderTargets(1, &color, NULL, NULL, NULL);

    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].bind(queueCB);
        queueCB.SetViewport(0, 0, fboWidth, fboHeight);
        queueCB.SetScissor(0, 0, fboWidth, fboHeight);
        queueCB.ClearColor(0, 0.1f, 0.2f, 0.4f, 0);
        queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0x00, 0xff);

        // Render 2 triangles into an offscreen buffer.  The second,
        // larger green triangle will be partially occluded by the first
        // red triangle. If depth write doesn't get enabled when switching
        // fbos, only the green triangle will be visible.
        uboCpuVa->scale  = dt::vec4(1, 1, 1, 1);
        uboCpuVa->offset = dt::vec4(0, 0, 0, 0);
        uboCpuVa->color  = dt::vec4(1, 0, 0, 1);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        queueCB.submit();
        queue->Finish();
        uboCpuVa->scale  = dt::vec4(1.5, 1.5, 1.0, 1.0);
        uboCpuVa->offset = dt::vec4(0.1, 0.1, 0.1f, 0);
        uboCpuVa->color  = dt::vec4(0, 1, 0, 1);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
        queueCB.submit();
        queue->Finish();
    }

    g_lwnWindowFramebuffer.bind();
    showFbo(queueCB, fbos[1].getColorTexture(0), fboWidth, fboHeight, 0);
    queueCB.submit();
    queue->Finish();

    // Repro for Case 2 (buggy SetDepthWrite)
    // Test that drawing to S8 with depth writes enabled
    // doesn't cause any issues.
    
    depthState.SetDefaults()
        .SetDepthTestEnable(LWN_TRUE)
        .SetDepthWriteEnable(LWN_FALSE)
        .SetDepthFunc(DepthFunc::LEQUAL)
        .SetStencilFunc(Face::FRONT_AND_BACK, StencilFunc::ALWAYS)
        .SetStencilOp(Face::FRONT_AND_BACK, StencilOp::INCR, StencilOp::INCR, StencilOp::INCR)
        .SetStencilTestEnable(LWN_TRUE);
    queueCB.BindDepthStencilState(&depthState);
    queueCB.SetStencilRef(Face::FRONT_AND_BACK, 1);
    queueCB.SetStencilMask(Face::FRONT_AND_BACK, 0xff);
    queueCB.SetStencilValueMask(Face::FRONT_AND_BACK, 0xff);

    color = fbos[0].getColorTexture(0);
    queueCB.SetRenderTargets(1, &color, NULL, NULL, NULL);

    fbos[0].bind(queueCB);
    depthState.SetDepthWriteEnable(LWN_TRUE);
    queueCB.BindDepthStencilState(&depthState);

    queueCB.SetViewport(0, 0, fboWidth, fboHeight);
    queueCB.SetScissor(0, 0, fboWidth, fboHeight);
    queueCB.ClearColor(0, 0.1f, 0.2f, 0.4f, 0);
    queueCB.ClearDepthStencil(1.0f, LWN_TRUE, 0x00, 0xff);

    // Render 2 triangles into an offscreen buffer. The first, smaller red
    // triangle writes 1 to stencil buffer. The second, larger green triangle
    // will be visible where pixel's stencil value is 1.
    uboCpuVa->scale  = dt::vec4(1, 1, 1, 1);
    uboCpuVa->offset = dt::vec4(0, 0, 0, 0);
    uboCpuVa->color  = dt::vec4(1, 0, 0, 1);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
    queueCB.submit();
    queue->Finish();

    depthState.SetStencilFunc(Face::FRONT_AND_BACK, StencilFunc::EQUAL)
              .SetStencilOp(Face::FRONT_AND_BACK, StencilOp::KEEP, StencilOp::KEEP, StencilOp::KEEP)
              .SetStencilTestEnable(LWN_TRUE);
    queueCB.BindDepthStencilState(&depthState);

    uboCpuVa->scale  = dt::vec4(1.5, 1.5, 1.0, 1.0);
    uboCpuVa->offset = dt::vec4(0.1, 0.1, 0.1f, 0);
    uboCpuVa->color  = dt::vec4(0, 1, 0, 1);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
    queueCB.submit();
    queue->Finish();

    g_lwnWindowFramebuffer.bind();
    showFbo(queueCB, fbos[0].getColorTexture(0), fboWidth, fboHeight, 1);
    queueCB.submit();
    queue->Finish();
    
    for (int i = 0; i < NUM_FBOS; i++) {
        fbos[i].destroy();
    }
}

OGTEST_CppTest(LWNDepthWriteWAR, lwn_depth_write_war, );
