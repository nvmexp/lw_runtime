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

class LWNTriangleTestCpp
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTriangleTestCpp::getDescription() const
{
    lwStringBuf sb;
    sb << "Simple single-triangle 'hello world' test for LWN.";
    return sb.str();    
}

int LWNTriangleTestCpp::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

void LWNTriangleTestCpp::doGraphics() const
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
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
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
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.375, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.375, +0.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.375, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);

    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNTriangleTestCpp, lwn_01tri_cpp, );
