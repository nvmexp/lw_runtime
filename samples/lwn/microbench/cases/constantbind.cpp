/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Constant buffer bind benchmark.
//

#include "constantbind.hpp"
#include <string.h>
#include <stdio.h>
#include <assert.h>

struct TestDescr
{
    int  numSegments;
};

static const TestDescr subtests[] = {
    { 6  },
    { 7  },
    { 8  },
    { 10 },
    { 16 },
    { 32 }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

const int OBJS_X = 64*2;
const int OBJS_Y = 64*2;

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

BenchmarkConstantBindLWN::BenchmarkConstantBindLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkConstantBindLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkConstantBindLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    snprintf(testName, sizeof(testName), "constant_bind.numSegments=%d", t.numSegments);

    Description d;
    d.name  = testName;
    d.units = "draw/s";
    return d;
}

LWNcommandHandle BenchmarkConstantBindLWN::renderCommands(LWNcommandBuffer* cmdBuffer)
{
    lwnCommandBufferBeginRecording(cmdBuffer);
    lwnCommandBufferBindProgram(cmdBuffer, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    m_vertex->bind(cmdBuffer);
    lwnCommandBufferBindVertexBuffer(cmdBuffer, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));

    LWNbufferAddress cbAddress      = m_cb->address();
    LWNbufferAddress iboAddress     = m_mesh->iboAddress();
    const int       numPrimitives = m_mesh->numTriangles()*3;

    for (int drawIdx = 0; drawIdx < N_DRAWS; drawIdx++) {
        lwnCommandBufferBindUniformBuffer(cmdBuffer, LWN_SHADER_STAGE_VERTEX,
                                          0,
                                          cbAddress + m_cb->offset(drawIdx),
                                          sizeof(SegAttrs));

        lwnCommandBufferDrawElements(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     numPrimitives, iboAddress);
    }
    return lwnCommandBufferEndRecording(cmdBuffer);
}

void BenchmarkConstantBindLWN::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];

    m_numInstancesRendered = 0;

    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 1, 2*1024*1024, 1024);
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

    m_prebuiltCmdHandle = renderCommands(cmd);
#if 0
    PRINTF("cmd mem used = %d free = %d, ctrl mem used = %d free = %d\n",
        (int)lwnCommandBufferGetCommandMemoryUsed(cmd), (int)lwnCommandBufferGetCommandMemoryFree(cmd),
        (int)lwnCommandBufferGetControlMemoryUsed(cmd), (int)lwnCommandBufferGetControlMemoryFree(cmd));
#endif
}

void BenchmarkConstantBindLWN::draw(const DrawParams* params)
{
    m_numInstancesRendered += N_DRAWS;

    m_cmdBuf->submit(1, &m_prebuiltCmdHandle);
}

double BenchmarkConstantBindLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numInstancesRendered / elapsedTime;
}

void BenchmarkConstantBindLWN::deinit(int subtest)
{
    delete m_vertex;
    delete m_mesh;
    delete m_cb;

    lwnProgramFinalize(m_pgm);
    delete m_pgm;

    delete m_cmdBuf;
}

BenchmarkConstantBindLWN::~BenchmarkConstantBindLWN()
{
}
