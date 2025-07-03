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

class LWNAlphaTest
{
    static const int cellSize = 60;
public:
    LWNTEST_CppMethods();
};

lwString LWNAlphaTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic function test for the LWN alpha test.  This test draws an "
        "array of cells, where each cell draws a square with cirlwlar "
        "rings with alpha varying from 0.0 to 1.0 in increments of 0.25. "
        "The rings are alternately colored yellow and blue.\n\n"
        "Each row has a different alpha test (bottom to top:  NEVER, LESS, "
        "EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS).  Each column has "
        "a different reference value (left = 0.0, right = 1.0, steps of 0.125). "
        "Each cell should show a subset of the yellow and blue rings based on "
        "the test and reference value.";
    return sb.str();    
}

int LWNAlphaTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 6);
}

void LWNAlphaTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  otc = position.xy;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"

        // Compute a ring number based by computing the distance of the
        // screen-space (x,y) in texture coordinate <otc> from the origin,
        // scaling and rounding to an integer, dividing by 8, and clamping to
        // 1.0.  Use that value as alpha, and to choose a blue or yellow color.
        "  float f = min(floor(length(otc) * 8.0) / 8.0, 1.0);\n"
        "  int cidx = int(8.0 * f);\n"
        "  vec3 color = (0 == (int(8.0 * f) & 1)) ? vec3(1,1,0) : vec3(0,0,1);\n"
        "  fcolor = vec4(color, f);\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.9, -0.9, 0.0) },
        { dt::vec3(-0.9, +0.9, 0.0) },
        { dt::vec3(+0.9, -0.9, 0.0) },
        { dt::vec3(+0.9, +0.9, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 4 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    AlphaFunc alphaFuncs[8] = {
        AlphaFunc::NEVER,
        AlphaFunc::LESS,
        AlphaFunc::EQUAL,
        AlphaFunc::LEQUAL,
        AlphaFunc::GREATER,
        AlphaFunc::NOTEQUAL,
        AlphaFunc::GEQUAL,
        AlphaFunc::ALWAYS,
    };
    ColorState *cstate = device->CreateColorState();
    for (int row = 0; row < 8; row++) {
        cstate->SetAlphaTest(alphaFuncs[row]);
        queueCB.BindColorState(cstate);
        for (int col = 0; col < 9; col++) {
            queueCB.SetAlphaRef(col / 8.0);
            queueCB.SetViewportScissor(col * cellSize, row * cellSize, cellSize, cellSize);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        }
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNAlphaTest, lwn_alphatest, );
