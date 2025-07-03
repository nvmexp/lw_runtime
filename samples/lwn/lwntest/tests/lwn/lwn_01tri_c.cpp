/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_c.h"
#include "lwn_utils.h"

class LWNTriangleTestC
{
public:
    LWNTEST_CppMethods();
};

lwString LWNTriangleTestC::getDescription() const
{
    lwStringBuf sb;
    sb << "Simple single-triangle 'hello world' test for LWN.";
    return sb.str();    
}

int LWNTriangleTestC::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

void LWNTriangleTestC::doGraphics() const
{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;
    LWNdevice *device = g_lwnDevice;
    LWNcommandBuffer *cmdBuf = queueCB;
    LWNqueue *queue = g_lwnQueue;

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

    LWNprogram *pgm = lwnDeviceCreateProgram(device);

    // Compile and call lwnProgramSetShaders.
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // Set up the vertex format and buffer.
    struct Vertex {
        lwn::dt::vec3 position;
        lwn::dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { lwn::dt::vec3(-0.375, -0.5, 0.0), lwn::dt::vec3(0.0, 0.0, 1.0) },
        { lwn::dt::vec3(-0.375, +0.5, 0.0), lwn::dt::vec3(0.0, 1.0, 0.0) },
        { lwn::dt::vec3(+0.375, -0.5, 0.0), lwn::dt::vec3(1.0, 0.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    lwnTest::VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    lwnTest::VertexArrayState vertex = stream.CreateVertexArrayState();
    LWNbuffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    LWNbufferAddress vboAddr = lwnBufferGetAddress(vbo);

    LWNfloat clearColor[] = { 0.4, 0.0, 0.0, 0.0 };
    lwnCommandBufferClearColor(cmdBuf, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferBindProgram(cmdBuf, pgm, LWN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    vertex.bind(cmdBuf);
    lwnCommandBufferBindVertexBuffer(cmdBuf, 0, vboAddr, sizeof(vertexData));
    lwnCommandBufferDrawArrays(cmdBuf, LWN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    lwnQueueFinish(queue);
}

OGTEST_CppTest(LWNTriangleTestC, lwn_01tri, );
