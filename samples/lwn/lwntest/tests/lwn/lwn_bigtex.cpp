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

class LWNBigTextureTest
{
    static const int texWidth = 4096;
    static const int texHeight = 3072;
    static const int checkerSize = 64;
public:
    LWNTEST_CppMethods();
};

lwString LWNBigTextureTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple render-to-texture test using a large (4K x 3K) texture.  "
        "Draws a gradient black with concentric black rings around the middle.  "
        "Exercises bug 1667928 (corruption with textures in memory pools).";
    return sb.str();    
}

int LWNBigTextureTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(21, 0);
}

void LWNBigTextureTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "  otc = 0.5 * position.xy + 0.5;\n"
        "}\n";
    FragmentShader drawFS(440);
    drawFS <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec2 center = vec2(" << texWidth / 2 << ", " << texHeight / 2 << ");\n"
        "  float dist = length(gl_FragCoord.xy - center);\n"
        "  if ((int(dist) % 64) < 40) {\n"
        "    fcolor = vec4(ocolor, 1.0);\n"
        "  } else {\n"
        "    fcolor = vec4(0.0);\n"
        "  }\n"
        "}\n";
    FragmentShader copyFS(440);
    copyFS <<
        "layout(binding=0) uniform sampler2D tex;\n"
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc);\n"
        "}\n";

    Program *drawProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(drawProgram, vs, drawFS)) {
        LWNFailTest();
        return;
    }

    Program *copyProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(copyProgram, vs, copyFS)) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec3(1.0, 1.0, 1.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 4 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    TextureBuilder tb;
    tb.SetDefaults().SetDevice(device);
    tb.SetFlags(TextureFlags::COMPRESSIBLE).
        SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(texWidth, texHeight).
        SetFormat(Format::RGBA8);
    LWNsizeiptr size = tb.GetStorageSize();
    LWNsizeiptr alignment = tb.GetStorageAlignment();
    size = alignment * ((size + alignment - 1) / alignment);
    MemoryPoolAllocator texAllocator(device, NULL, size, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *tex = texAllocator.allocTexture(&tb);

    SamplerBuilder sb;
    sb.SetDefaults().SetDevice(device);
    Sampler *smp = sb.CreateSampler();

    TextureHandle texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID());

    queueCB.SetRenderTargets(1, &tex, NULL, NULL, NULL);
    queueCB.SetViewportScissor(0, 0, texWidth, texHeight);

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB.BindProgram(drawProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    // Need a barrier between the render and texture passes to ensure proper ordering.
    queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    queueCB.BindProgram(copyProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNBigTextureTest, lwn_bigtex, );
