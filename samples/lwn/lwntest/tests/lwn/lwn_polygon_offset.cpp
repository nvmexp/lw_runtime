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
#include "lwn_basic.h"

using namespace lwn;

class LWNPolygonOffsetTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNPolygonOffsetTest::getDescription() const
{
    return "Testing polygon offset.\n"
            "First row checks depth testing and poffset's unit working.\n"
            "Second row checks that only one mode is affected by poffset.\n"
            "Third row checks for offsets applied for different slopes and\n"
            "tests if the getter is correct.";
}

int LWNPolygonOffsetTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 9);
}

class ColoredDrawer
{
    struct UBOBlock_t
    {
        float uColor[4];
        float uVerts[4][4];
    };
    Program* pgm;
    MemoryPool* uboMem;
    size_t uboSize;
    BufferAddress uboAddress;
    UBOBlock_t* uboPtr;
    QueueCommandBuffer &cmd;
    Queue* queue;

public:
    ColoredDrawer()
        : cmd(DeviceState::GetActive()->getQueueCB())
    {
        Device *device = DeviceState::GetActive()->getDevice();
        queue = DeviceState::GetActive()->getQueue();

        VertexShader vs(440);
        vs <<
              "layout(binding=0, std140) uniform Block {\n"
              "  vec4 color;\n"
              "  vec4 vertices[4];\n"
              "};\n"
              "void main() {\n"
              "  gl_Position = vertices[gl_VertexID];\n"
              "}\n";
        FragmentShader fs(440);
        fs <<
              "layout(binding=0, std140) uniform Block {\n"
              "  vec4 color;\n"
              "  vec4 vertices[4];\n"
              "};\n"
              "out vec4 fcolor;\n"
              "void main() {\n"
              "  fcolor = color;\n"
              "}\n";
        pgm = device->CreateProgram();
        g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

        LWNint uboAlignment;
        device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
        uboSize = ((sizeof(UBOBlock_t) + uboAlignment - 1) / uboAlignment) * uboAlignment;

        uboMem = device->CreateMemoryPool(NULL, uboSize, MemoryPoolType::CPU_COHERENT);
        uboAddress = uboMem->GetBufferAddress();
        uboPtr = static_cast<UBOBlock_t*>(uboMem->Map());

    }
    void Draw(dt::vec3* positions, dt::vec4 color)
    {
        uboPtr->uColor[0] = color.x();
        uboPtr->uColor[1] = color.y();
        uboPtr->uColor[2] = color.z();
        uboPtr->uColor[3] = color.w();
        for(int i=0; i<4; i++)
        {
            uboPtr->uVerts[i][0] = positions[i].x();
            uboPtr->uVerts[i][1] = positions[i].y();
            uboPtr->uVerts[i][2] = positions[i].z();
            uboPtr->uVerts[i][3] = 1.0;
        }
        cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        cmd.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddress, uboSize);
        cmd.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddress, uboSize);
        cmd.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
        cmd.submit();
        queue->Finish();
    }
    ~ColoredDrawer()
    {
        pgm->Free();
        uboMem->Free();
    }
};

void LWNPolygonOffsetTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    dt::vec4 redCol(1.0, 0.0, 0.0, 1.0);
    dt::vec4 greenCol(0.0, 1.0, 0.0, 1.0);
    dt::vec4 blueCol(0.0, 0.0, 1.0, 1.0);
    dt::vec4 whiteCol(1.0, 1.0, 1.0, 1.0);

    dt::vec3 fannedPos[4] = {
        dt::vec3(-1.0, -1.0, 0.0),
        dt::vec3(+1.0, -1.0, 0.0),
        dt::vec3(+1.0, +1.0, 0.0),
        dt::vec3(-1.0, +1.0, 0.0)
    };
    dt::vec3 crossPos[4] = {
        dt::vec3(-1.0, -1.0, 0.0),
        dt::vec3(+1.0, +1.0, 0.0),
        dt::vec3(+1.0, -1.0, 0.0),
        dt::vec3(-1.0, +1.0, 0.0)
    };

    ColoredDrawer cd;
    PolygonState ps;
    DepthStencilState ds;

    int widthThird = lwrrentWindowWidth / 3;
    int heightThird = lwrrentWindowHeight / 3;

    ps.SetDefaults();
    ds.SetDefaults().SetDepthTestEnable(true);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0);
    queueCB.ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
    queueCB.SetPointSize(50.0);

    // -------------------------------------------------------------------------

    // LESS, red quad, white quad
    // result: red (no z-fight)
    queueCB.SetViewportScissor(0, heightThird*2, widthThird, heightThird);
    ds.SetDepthFunc(DepthFunc::LESS);
    queueCB.BindDepthStencilState(&ds);
    cd.Draw(fannedPos, redCol);
    cd.Draw(fannedPos, whiteCol);

    // LEQUAL, red quad, white quad
    // result: white (z-fight)
    queueCB.SetViewportScissor(widthThird, heightThird*2, widthThird, heightThird);
    ds.SetDepthFunc(DepthFunc::LEQUAL);
    queueCB.BindDepthStencilState(&ds);
    cd.Draw(fannedPos, redCol);
    cd.Draw(fannedPos, whiteCol);

    // LEQUAL, red quad, poffset(0,1, FILL), white quad
    // result: red (no z-fight thanks to poffset)
    queueCB.SetViewportScissor(widthThird*2, heightThird*2, widthThird, heightThird);
    cd.Draw(fannedPos, redCol);
    ps.SetPolygonOffsetEnables(PolygonOffsetEnable::FILL);
    queueCB.BindPolygonState(&ps);
    queueCB.SetPolygonOffsetClamp(0.0, 1.0, 1.0);
    cd.Draw(fannedPos, whiteCol);

    // -------------------------------------------------------------------------

    ds.SetDepthFunc(DepthFunc::LESS);
    queueCB.BindDepthStencilState(&ds);
    queueCB.SetPolygonOffsetClamp(0.0, -1.0, 1.0);

    // LESS, poffset(0, -1, LINE), blue points, red quad, white cross
    // result: white cross on top of red quad with blue corners
    queueCB.SetViewportScissor(0, heightThird, widthThird, heightThird);
    ps.SetPolygonOffsetEnables(PolygonOffsetEnable::LINE);
    ps.SetPolygonMode(PolygonMode::POINT);
    queueCB.BindPolygonState(&ps);
    cd.Draw(fannedPos, blueCol);
    ps.SetPolygonMode(PolygonMode::FILL);
    queueCB.BindPolygonState(&ps);
    cd.Draw(fannedPos, redCol);
    ps.SetPolygonMode(PolygonMode::LINE);
    queueCB.BindPolygonState(&ps);
    cd.Draw(crossPos, whiteCol);

    // LESS, poffset(0, -1, POINT), white cross, red quad, blue points
    // result: blue corners on top of white crossed red quad
    queueCB.SetViewportScissor(widthThird, heightThird, widthThird, heightThird);
    ps.SetPolygonOffsetEnables(PolygonOffsetEnable::POINT);
    ps.SetPolygonMode(PolygonMode::LINE);
    queueCB.BindPolygonState(&ps);
    cd.Draw(crossPos, whiteCol);
    ps.SetPolygonMode(PolygonMode::FILL);
    queueCB.BindPolygonState(&ps);
    cd.Draw(fannedPos, redCol);
    ps.SetPolygonMode(PolygonMode::POINT);
    queueCB.BindPolygonState(&ps);
    cd.Draw(fannedPos, blueCol);

    // LESS, poffset(0, -1, FILL), white cross, blue points, red quad
    // result: red
    queueCB.SetViewportScissor(widthThird*2, heightThird, widthThird, heightThird);
    ps.SetPolygonOffsetEnables(PolygonOffsetEnable::FILL);
    ps.SetPolygonMode(PolygonMode::LINE);
    queueCB.BindPolygonState(&ps);
    cd.Draw(crossPos, whiteCol);
    ps.SetPolygonMode(PolygonMode::POINT);
    queueCB.BindPolygonState(&ps);
    cd.Draw(fannedPos, blueCol);
    ps.SetPolygonMode(PolygonMode::FILL);
    queueCB.BindPolygonState(&ps);
    cd.Draw(fannedPos, redCol);

    // -------------------------------------------------------------------------

    ps.SetPolygonOffsetEnables(PolygonOffsetEnable::FILL);
    ps.SetPolygonMode(PolygonMode::FILL);
    queueCB.BindPolygonState(&ps);
    ds.SetDepthFunc(DepthFunc::LESS);
    queueCB.BindDepthStencilState(&ds);

    dt::vec3 tiltedPos[4] = {
        dt::vec3(-1.0, -1.0, -1.0),
        dt::vec3(+1.0, -1.0, -1.0),
        dt::vec3(+1.0, +1.0,  1.0),
        dt::vec3(-1.0, +1.0,  1.0)
    };
    dt::vec3 tiltedPos2[4] = {
        dt::vec3(-1.0, -1.0, -0.5),
        dt::vec3(+1.0, -1.0, -0.5),
        dt::vec3(+1.0, +1.0,  0.5),
        dt::vec3(-1.0, +1.0,  0.5)
    };

    // depth slope = 0.0 -> total offset=0.0
    // result: red
    queueCB.SetViewportScissor(0, 0, widthThird, heightThird);
    queueCB.SetPolygonOffsetClamp(0.0,0.0,0.0);
    cd.Draw(fannedPos, redCol);
    queueCB.SetPolygonOffsetClamp(-10000.0,0.0,0.0);
    cd.Draw(fannedPos, whiteCol);

    // draw: red, slope=1.0, poffset(-5,0) -> total offset=-5.0
    // draw: white, slope=0.5, poffset(-10,0) -> total offset =-5.0
    // result: visible white-red (50%-50%)
    queueCB.SetViewportScissor(widthThird, 0, widthThird, heightThird);
    queueCB.SetPolygonOffsetClamp(-5.0,0.0,0.0);
    cd.Draw(tiltedPos, redCol);
    queueCB.SetPolygonOffsetClamp(-10.0,0.0,0.0);
    cd.Draw(tiltedPos2, whiteCol);

    // test got GetPolygonOffsetEnables value
    // result: green if passed, red otherwise
    queueCB.SetViewportScissor(widthThird*2, 0, widthThird, heightThird);
    dt::vec4 testCol = ((PolygonOffsetEnable::FILL==ps.GetPolygonOffsetEnables()) ? greenCol : redCol);
    queueCB.ClearColor(0,testCol.x(),testCol.y(),testCol.z());
    queueCB.submit();
}

OGTEST_CppTest(LWNPolygonOffsetTest, lwn_polygon_offset, );
