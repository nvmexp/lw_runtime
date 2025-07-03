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

class LWNDepthBoundsTest
{
    static const int cellSize = 48;
    static const int cellMargin = 2;
public:
    LWNTEST_CppMethods();
};

lwString LWNDepthBoundsTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Functional test for LWN depth bounds testing.  This test draws an "
        "array of cells, where near values range from 0.0 to 1.0 vertically "
        "and far values range from 0.0 to 1.0 horizontally.  Cells where "
        "near >= far are skipped and drawn in gray.  Other cells set up a "
        "gradient in the depth buffer with 0.0 on the left and 1.0 on the "
        "right.  The background is cleared to blue and pixels that pass the "
        "depth bounds test are rendered in white.";
    return sb.str();    
}

int LWNDepthBoundsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 12);
}

void LWNDepthBoundsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(1.0);\n"
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
        { dt::vec3(-1.0, -1.0, -1.0) },
        { dt::vec3(-1.0, +1.0, -1.0) },
        { dt::vec3(+1.0, -1.0, +1.0) },
        { dt::vec3(+1.0, +1.0, +1.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 4 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // The first pass in each cell draws with depth testing enabled (ALWAYS)
    // to update depth buffer, with color updates masked off.
    DepthStencilState dssUpdate;
    ChannelMaskState maskNoColor;
    dssUpdate.SetDefaults();
    dssUpdate.SetDepthTestEnable(LWN_TRUE);
    dssUpdate.SetDepthFunc(DepthFunc::ALWAYS);
    maskNoColor.SetDefaults();
    maskNoColor.SetChannelMask(0, LWN_FALSE, LWN_FALSE, LWN_FALSE, LWN_FALSE);

    // The second pass draws with depth testing disabled but with depth bounds
    // test enabled.  Color updates are enabled.
    DepthStencilState dssDefault;
    ChannelMaskState maskDefault;
    dssDefault.SetDefaults();
    maskDefault.SetDefaults();

    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < 10; col++) {
            queueCB.SetViewportScissor(col * cellSize + cellMargin, row * cellSize + cellMargin,
                                       cellSize - 2 * cellMargin, cellSize - 2 * cellMargin);
            if (row >= col) {
                queueCB.ClearColor(0, 0.2, 0.2, 0.2, 1.0);
                continue;  // don't render anything else
            } else {
                queueCB.ClearColor(0, 0.0, 0.0, 0.5, 1.0);
            }
            queueCB.BindDepthStencilState(&dssUpdate);
            queueCB.BindChannelMaskState(&maskNoColor);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            queueCB.SetDepthBounds(LWN_TRUE, float(row) / 9.0, float(col) / 9.0);
            queueCB.BindDepthStencilState(&dssDefault);
            queueCB.BindChannelMaskState(&maskDefault);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            queueCB.SetDepthBounds(LWN_FALSE, 0.0, 1.0);
        }
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNDepthBoundsTest, lwn_dbounds, );
