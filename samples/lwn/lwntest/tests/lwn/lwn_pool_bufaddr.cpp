/*
 * Copyright (c) 2015-2016 LWPU Corporation.  All rights reserved.
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

class LWNPoolBufferAddressTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNPoolBufferAddressTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Simple single-triangle 'hello world' test for LWN.\n\n"
        "Uses lwnMemoryPoolGetBufferAddress to determine GPU addresses for "
        "various buffer data without creating LWN buffer objects.";
    return sb.str();    
}

int LWNPoolBufferAddressTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 7);
}

void LWNPoolBufferAddressTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "layout(binding=0) uniform Block {\n"
        "  vec4 scale;\n"
        "  vec4 bias;\n"
        "};\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0) * scale + bias;\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    MemoryPool *bufdata = device->CreateMemoryPool(NULL, 4096, MemoryPoolType::CPU_COHERENT);
    BufferAddress bufaddr = bufdata->GetBufferAddress();
    char *bufmem = (char *) bufdata->Map();

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();

    Vertex *vertices = (Vertex *) bufmem;
    vertices[0].position = dt::vec3(-0.75, -0.5, 0.0);
    vertices[0].color = dt::vec3(0.0, 0.0, 1.0);
    vertices[1].position = dt::vec3(-0.75, +1.5, 0.0);
    vertices[1].color= dt::vec3(0.0, 1.0, 0.0);
    vertices[2].position = dt::vec3(+0.75, -0.5, 0.0);
    vertices[2].color = dt::vec3(1.0, 0.0, 0.0);

    // Set up a uniform buffer that scales/biases our coordinates back to
    // fit in the same +/-0.375 range as "01tri".
    struct UBO {
        dt::vec4 scale;
        dt::vec4 bias;
    };
    UBO *ubodata = (UBO *) (bufmem + 1024);
    ubodata->scale = dt::vec4(0.5, 0.5, 1.0, 1.0);
    ubodata->bias = dt::vec4(0.0, -0.25, 0.0, 0.0);

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, bufaddr, 3 * sizeof(Vertex));
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, bufaddr + 1024, sizeof(UBO));
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNPoolBufferAddressTest, lwn_pool_bufaddr, );
