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

class LWNDepthRangeTest
{
    static const int cellsX = 10;
    static const int cellsY = 8;
public:
    OGTEST_CppMethods();
};

lwString LWNDepthRangeTest::getDescription()
{
    lwStringBuf sb;
    sb <<
        "Basic LWN test of depth range and depth clamping.  In each cell, we render "
        "a shaded quad on top of a dark red backdrop.  The clip Z value varies from "
        "-1.0 (near plane, bottom) to 1.0 (far plane, top) for the backdrop and -1.5 "
        "(left) to +1.5 (right) for the shaded quad.  The columns vary the depth "
        "range from [0.0,1.0] (left) to [0.0,0.1] (right).  Rows 0 and 1 render "
        "with a depth test of LESS (below) and GREATER.  Rows 2 and 3 repeat those "
        "rows with the depth range reversed.  Rows 4-7 repeat rows 0-3 with depth "
        "clamping enabled, where the left- and right-most 1/6th of the quad are "
        "not clipped away.";
    return sb.str();    
}

int LWNDepthRangeTest::isSupported()
{
    return lwogCheckLWNAPIVersion(12, 2);
}

void LWNDepthRangeTest::initGraphics()
{
    cellTestInit(cellsX, cellsY);
    lwnDefaultInitGraphics();
}

void LWNDepthRangeTest::exitGraphics()
{
    lwnDefaultExitGraphics();
}

void LWNDepthRangeTest::doGraphics()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;

    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Shaders for basic smooth shading based on provided input colors.
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

    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    if (!compiled) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        // Triangle strip to draw a background (with default state).  The
        // window Z value (with normal depth ranges) has a gradient of 0.0
        // (bottom) to 1.0 (top).
        { dt::vec3(-1.0, -1.0, -1.0), dt::vec3(0.2, 0.0, 0.0) },
        { dt::vec3(-1.0, +1.0, +1.0), dt::vec3(0.2, 0.0, 0.0) },
        { dt::vec3(+1.0, -1.0, -1.0), dt::vec3(0.2, 0.0, 0.0) },
        { dt::vec3(+1.0, +1.0, +1.0), dt::vec3(0.2, 0.0, 0.0) },

        // Triangle strip to draw a test polygon.  Z has a gradient of less
        // than zero (left) to greater than one (right), before the depth
        // range.  Without depth clamping, the left 1/6th and right 1/6th of
        // the primitive will be clipped.
        { dt::vec3(-1.0, -1.0, -1.5), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-1.0, +1.0, -1.5), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, +1.5), dt::vec3(1.0, 0.0, 1.0) },
        { dt::vec3(+1.0, +1.0, +1.5), dt::vec3(1.0, 1.0, 0.0) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 8, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Create depth state object, and enable depth test
    DepthStencilState depth;
    depth.SetDefaults();
    depth.SetDepthTestEnable(LWN_TRUE);

    queueCB.BindDepthStencilState(&depth);
    queueCB.BindVertexArrayState(vertex);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 10; col++) {
            if (!cellAllowed(col, row)) {
                continue;
            }
            SetCellViewportScissorPadded(queueCB, col, row, 2);

            // Render the backdrop with normal state vectors and a depth
            // function of ALWAYS to ensure the depth buffer is updated.
            depth.SetDepthFunc(DepthFunc::ALWAYS);
            queueCB.BindDepthStencilState(&depth);
            queueCB.SetDepthRange(0.0, 1.0);
            queueCB.SetDepthClamp(LWN_FALSE);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

            // Set up the depth test, range, and clamping enables for the row.
            depth.SetDepthFunc((row & 1) ? DepthFunc::GREATER : DepthFunc::LESS);
            queueCB.BindDepthStencilState(&depth);
            if (row & 2) {
                queueCB.SetDepthRange(1.0 - col * 0.1, 0.0);
            } else {
                queueCB.SetDepthRange(0.0, 1.0 - col * 0.1);
            }
            queueCB.SetDepthClamp((row & 4) ? LWN_TRUE : LWN_FALSE);

            // Render the shaded quad.
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 4, 4);
        }
    }
    depth.SetDefaults();
    queueCB.BindDepthStencilState(&depth);
    queueCB.SetDepthRange(0.0, 1.0);
    queueCB.SetDepthClamp(LWN_FALSE);

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNDepthRangeTest, lwn_depth_range, );
