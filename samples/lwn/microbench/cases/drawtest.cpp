
/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Draw test benchmark exercising multiple draw APIs.
//

#include "drawtest.hpp"
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define TEST_DESC_ENTRIES(segmentCount) \
    { segmentCount, DrawElements }, \
    { segmentCount, DrawArrays }, \
    { segmentCount, DrawElementsBaseVertex }, \
    { segmentCount, DrawElementsIndirect }, \
    { segmentCount, DrawArraysIndirect }, \
    { segmentCount, DrawArraysInstanced }, \
    { segmentCount, DrawElementsInstanced },

const BenchmarkDrawTestLWN::TestDescr BenchmarkDrawTestLWN::subtests[] = {
    TEST_DESC_ENTRIES(6)
    TEST_DESC_ENTRIES(32)

#if 0
    // Some more tests that can be run locally if required.
    TEST_DESC_ENTRIES(7)
    TEST_DESC_ENTRIES(8)
    TEST_DESC_ENTRIES(10)
    TEST_DESC_ENTRIES(16)
    TEST_DESC_ENTRIES(64)
    TEST_DESC_ENTRIES(128)
#endif
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

const int OBJS_X = 128;
const int OBJS_Y = 128;

// How many instances to draw
const int N_DRAWS = OBJS_X * OBJS_Y;

static const char *VS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(binding = 0) uniform segmentBlock {\n"
    "  vec4 offset;\n"
    "  vec4 color;\n"
    "};\n"
    "layout(location = 0) in vec3 position;\n"
    "out IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  gl_Position = vec4(position, 1.0) + offset;\n"
    "  vtxcol = color;\n"
    "}\n";

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  color = vtxcol;\n"
    "}\n";

BenchmarkDrawTestLWN::BenchmarkDrawTestLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkDrawTestLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

const char * BenchmarkDrawTestLWN::DrawFuncToString(DrawFunc drawFunc)
{
    switch (drawFunc) {
    case DrawArraysIndirect:
        return "DrawArraysIndirect";
    case DrawElementsIndirect:
        return "DrawElementsIndirect";
    case DrawArraysInstanced:
        return "DrawArraysInstanced";
    case DrawElementsInstanced:
        return "DrawElementsInstanced";
    case DrawArrays:
        return "DrawArrays";
    case DrawElements:
        return "DrawElements";
    case DrawElementsBaseVertex:
        return "DrawElementsBaseVertex";
    default:
        break;
    }

    return NULL;
}

BenchmarkCase::Description BenchmarkDrawTestLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];
    const char *drawFuncName = DrawFuncToString(t.drawFunc);

    snprintf(testName, sizeof(testName),
             "draw_test.numSegments=%d.drawFunc=%s",
             t.numSegments, drawFuncName);

    Description d;
    d.name  = testName;
    d.units = "draw/s";
    return d;
}

LWNcommandHandle BenchmarkDrawTestLWN::renderCommands(LWNcommandBuffer* cmdBuffer,
        DrawFunc drawFunc)
{
    lwnCommandBufferBeginRecording(cmdBuffer);
    lwnCommandBufferBindProgram(cmdBuffer, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    m_vertex->bind(cmdBuffer);
    lwnCommandBufferBindVertexBuffer(cmdBuffer, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));

    LWNbufferAddress cbAddress      = m_cb->address();
    LWNbufferAddress iboAddress     = m_mesh->iboAddress();
    const int        numPrimitives  = m_mesh->numTriangles()*3;

    LWNdrawElementsIndirectData deiData;
    deiData.baseVertex    = 0;
    deiData.baseInstance  = 0;
    deiData.instanceCount = 1;
    deiData.firstIndex    = 0;
    deiData.count         = numPrimitives;
    m_dei->set(0, deiData);

    LWNdrawArraysIndirectData daiData;
    daiData.baseInstance  = 0;
    daiData.count         = numPrimitives;
    daiData.first         = 0;
    daiData.instanceCount = 1;
    m_dai->set(0, daiData);

    for (int drawIdx = 0; drawIdx < N_DRAWS; drawIdx++) {
        lwnCommandBufferBindUniformBuffer(cmdBuffer, LWN_SHADER_STAGE_VERTEX,
                                          0,
                                          cbAddress + m_cb->offset(drawIdx),
                                          sizeof(SegAttrs));

        switch (drawFunc) {
        case DrawArrays:
            lwnCommandBufferDrawArrays(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLE_FAN, 0, numPrimitives);
            break;
        case DrawElements:
            lwnCommandBufferDrawElements(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                         numPrimitives, iboAddress);
            break;
        case DrawElementsBaseVertex:
            lwnCommandBufferDrawElementsBaseVertex(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                         numPrimitives, iboAddress, 0);
            break;
        case DrawArraysInstanced:
            lwnCommandBufferDrawArraysInstanced(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLE_FAN, 0, numPrimitives, 0, 1);
            break;
        case DrawElementsInstanced:
            lwnCommandBufferDrawElementsInstanced(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                                  numPrimitives, iboAddress, 0, 0, 1);
            break;
        case DrawArraysIndirect:
            lwnCommandBufferDrawArraysIndirect(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLE_FAN, m_dai->address());
            break;
        case DrawElementsIndirect:
            lwnCommandBufferDrawElementsIndirect(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES,
                                                 LWN_INDEX_TYPE_UNSIGNED_INT, iboAddress, m_dei->address());
            break;
        default:
            assert(!"Invalid draw function.");
            break;
        }
    }
    return lwnCommandBufferEndRecording(cmdBuffer);
}

void BenchmarkDrawTestLWN::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];

    m_numInstancesRendered = 0;

    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 1, 2*1024*1024, 1024*1024);
    LWNcommandBuffer* cmd = m_cmdBuf->cmd();

    lwnCommandBufferBeginRecording(cmd);

    // Create programs from the device, provide them shader code and compile/link them
    m_pgm = new LWNprogram;
    lwnProgramInitialize(m_pgm, device());

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2]   = { VS_STRING, FS_STRING };
    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(m_pgm, stages, nSources, sources))
    {
        assert(0); // TBD
    }

    m_vertex = new LwnUtil::VertexState;
    m_vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_vertex->setStream(0, 12);

    int numSegments = testDescr.numSegments;
    m_mesh = LwnUtil::Mesh::createCircle(device(), coherentPool(), numSegments, Vec2f(1.f/(float)OBJS_X, 1.f/(float)OBJS_Y), 1.f);

    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0, LWN_TRUE, 0, 0);

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_cmdBuf->submit(1, &cmdHandle);

    m_cb = new LwnUtil::UboArr<SegAttrs>(device(), coherentPool(), N_DRAWS);

    for (int row = 0; row < OBJS_Y; row++) {
        for (int col = 0; col < OBJS_X; col++) {
            SegAttrs a;
            float u = ((float)col/(float)OBJS_X);
            float v = ((float)row/(float)OBJS_Y);
            float x = (u - 0.5f) * 2.f + (1.f/(float)OBJS_X);
            float y = (v - 0.5f) * 2.f + (1.f/(float)OBJS_Y);

            a.offset = Vec4f(x, y, 0.f, 0.f);
            a.color  = Vec4f(u, v, 1.f, 0.f);
            m_cb->set(row*OBJS_X + col, a);
        }
    }

    m_dei = new LwnUtil::IndirectArr<LWNdrawElementsIndirectData>(device(), coherentPool(), 1);
    m_dai = new LwnUtil::IndirectArr<LWNdrawArraysIndirectData>(device(), coherentPool(), 1);

    m_prebuiltCmdHandle = renderCommands(cmd, testDescr.drawFunc);
#if 0
    PRINTF("cmd mem used = %d free = %d, ctrl mem used = %d free = %d\n",
        (int)lwnCommandBufferGetCommandMemoryUsed(cmd), (int)lwnCommandBufferGetCommandMemoryFree(cmd),
        (int)lwnCommandBufferGetControlMemoryUsed(cmd), (int)lwnCommandBufferGetControlMemoryFree(cmd));
#endif
}

void BenchmarkDrawTestLWN::draw(const DrawParams* params)
{
    m_numInstancesRendered += N_DRAWS;

    m_cmdBuf->submit(1, &m_prebuiltCmdHandle);
}

double BenchmarkDrawTestLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numInstancesRendered / elapsedTime;
}

void BenchmarkDrawTestLWN::deinit(int subtest)
{
    delete m_vertex;
    delete m_mesh;
    delete m_cb;
    delete m_dei;
    delete m_dai;

    lwnProgramFinalize(m_pgm);
    delete m_pgm;

    delete m_cmdBuf;
}

BenchmarkDrawTestLWN::~BenchmarkDrawTestLWN()
{
}
