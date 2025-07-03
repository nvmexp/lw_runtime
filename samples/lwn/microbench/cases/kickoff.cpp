/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Kickoff and sync benchmark latency benchmark.

#include "kickoff.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkCase::Description Description;
typedef BenchmarkKickoffLWN::SegAttrs SegAttrs;
typedef BenchmarkKickoffLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { BenchmarkKickoffLWN::CMDBUF_SINGLE, false },
    { BenchmarkKickoffLWN::CMDBUF_MULTI, false },
    { BenchmarkKickoffLWN::CMDBUF_MULTI, true }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;

const int OBJS_X = 64;
const int OBJS_Y = 64;

// How many instances to draw
const int N_DRAWS = OBJS_X * OBJS_Y;

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)


#define UNIFORM_INPUT_LWN \
    "layout(binding = 0) uniform segmentBlock {\n" \
    "  vec4 offset;\n" \
    "  vec4 color;\n" \
    "};\n"

#define PROLOG \
    "#version 440 core\n" \
    "#extension GL_LW_gpu_shader5:require\n"

#define VS(ofsinput, color) \
    "layout(location = 0) in vec3 position;\n" \
    "out IO { vec4 vtxcol; };\n" \
    "void main() {\n" \
    "  gl_Position = vec4(position, 1.0) + " ofsinput ";\n" \
    "  vtxcol = " color ";\n" \
    "}\n";

static const char *VS_STRING_LWN =
    PROLOG
    UNIFORM_INPUT_LWN
    VS("offset", "color")

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  color = vtxcol;\n" // vtxcol
    "}\n";

BenchmarkKickoffLWN::BenchmarkKickoffLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkKickoffLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

Description BenchmarkKickoffLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    assert(t.cmdBufMode == CMDBUF_SINGLE || t.cmdBufMode == CMDBUF_MULTI);
    snprintf(testName, sizeof(testName), "kickoff.cmdbuf=%s.bufmem=%s",
             t.cmdBufMode == CMDBUF_SINGLE ? "single" : "multi",
             t.useCachedCmdBufMemory ? "cached" : "uncached");

    Description d;
    d.name  = testName;
    d.units = "bufs/s";
    return d;
}

void BenchmarkKickoffLWN::renderCommands(LWNcommandBuffer* cmdBuffer, int drawIdx)
{
    LWNbufferAddress cbAddress     = m_cb->address();
    LWNbufferAddress iboAddress    = m_mesh->iboAddress();
    const int        numPrimitives = m_mesh->numTriangles()*3;

    lwnCommandBufferBindUniformBuffer(cmdBuffer, LWN_SHADER_STAGE_VERTEX,
                                      0,
                                      cbAddress + m_cb->offset(drawIdx),
                                      sizeof(SegAttrs));

    lwnCommandBufferDrawElements(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                 numPrimitives, iboAddress);
}

static void initObjectAttrs(SegAttrs* dst)
{
    for (int row = 0; row < OBJS_Y; row++) {
        for (int col = 0; col < OBJS_X; col++) {
            SegAttrs a;
            float u = ((float)col/(float)OBJS_X);
            float v = ((float)row/(float)OBJS_Y);
            float x = (u - 0.5f) * 2.f + (1.f/(float)OBJS_X);
            float y = (v - 0.5f) * 2.f + (1.f/(float)OBJS_Y);

            a.offset = Vec4f(x, y, 0.f, 0.f);
            a.color  = Vec4f(u, v, 1.f, 0.f);
            dst[row*OBJS_X + col] = a;
        }
    }
}

void BenchmarkKickoffLWN::init(int subtest)
{
    m_subtestIdx = subtest;
    const TestDescr& testDescr = subtests[subtest];

    m_numCmdbufsSubmitted = 0;

    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, OBJS_X*OBJS_Y*128, 64*1024);
    LWNcommandBuffer* cmd = m_cmdBuf->cmd();

    lwnCommandBufferBeginRecording(cmd);

    // Create programs from the device, provide them shader code and compile/link them
    m_pgm = new LWNprogram;
    lwnProgramInitialize(m_pgm, device());

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2]   = { VS_STRING_LWN, FS_STRING };
    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(m_pgm, stages, nSources, sources))
    {
        assert(0);
    }

    m_vertex = new LwnUtil::VertexState;
    m_vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_vertex->setStream(0, 12);

    int numSegments = 10;
    m_mesh = LwnUtil::Mesh::createCircle(device(), coherentPool(), numSegments, Vec2f(0.5f/(float)OBJS_X, 0.5f/(float)OBJS_Y), 1.f);

    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0, LWN_TRUE, 0, 0);

    lwnCommandBufferBindProgram(cmd, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    m_vertex->bind(cmd);
    lwnCommandBufferBindVertexBuffer(cmd, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_cmdBuf->submit(1, &cmdHandle);

    m_cb = new LwnUtil::UboArr<SegAttrs>(device(), coherentPool(), N_DRAWS);

    std::vector<SegAttrs> objs(N_DRAWS);
    initObjectAttrs(objs.data());
    for (int i = 0; i < N_DRAWS; i++) {
        m_cb->set(i, objs[i]);
    }


    LwnUtil::BufferPool* cmdbufMemory = (testDescr.useCachedCmdBufMemory ?
                                         static_cast<LwnUtil::BufferPool*>(cpuCachedPool()) :
                                         static_cast<LwnUtil::BufferPool*>(coherentPool()));

    uintptr_t poolLwrAllocOffset = cmdbufMemory->used();

    // Use a single command buffer containing all rendering, or split
    // to N command buffers
    if (testDescr.cmdBufMode == CMDBUF_MULTI) {
        m_numCommandBuffers = N_DRAWS;
        m_commandBuffers = new LwnUtil::CompiledCmdBuf*[m_numCommandBuffers];
        m_commandHandles = new LWNcommandHandle[m_numCommandBuffers];

        for (int drawIdx = 0; drawIdx < N_DRAWS; drawIdx++) {
            LwnUtil::CompiledCmdBuf* buf = new LwnUtil::CompiledCmdBuf(device(), cmdbufMemory, 512, 128);
            m_commandBuffers[drawIdx] = buf;

            buf->begin();
            renderCommands(buf->cmd(), drawIdx);
            buf->end();
            m_commandHandles[drawIdx] = buf->handle();
        }
    } else {
        assert(testDescr.cmdBufMode == CMDBUF_SINGLE);

        m_numCommandBuffers = 1;
        m_commandBuffers = new LwnUtil::CompiledCmdBuf*[m_numCommandBuffers];
        m_commandHandles = new LWNcommandHandle[m_numCommandBuffers];

        LwnUtil::CompiledCmdBuf* buf = new LwnUtil::CompiledCmdBuf(device(), cmdbufMemory,
                                                                   512*N_DRAWS, 128*N_DRAWS);
        m_commandBuffers[0] = buf;

        buf->begin();
        for (int drawIdx = 0; drawIdx < N_DRAWS; drawIdx++) {
            renderCommands(buf->cmd(), drawIdx);
        }
        buf->end();
        m_commandHandles[0] = buf->handle();
    }

    if (testDescr.useCachedCmdBufMemory) {
        size_t cmdBufAllocated = cmdbufMemory->used() - poolLwrAllocOffset;
        lwnMemoryPoolFlushMappedRange(cmdbufMemory->pool(), poolLwrAllocOffset, cmdBufAllocated);
    }

}

void BenchmarkKickoffLWN::draw(const DrawParams* params)
{
    m_numCmdbufsSubmitted += N_DRAWS;

    lwnQueueSubmitCommands(queue(), m_numCommandBuffers, m_commandHandles);
    lwnQueueFlush(queue());
}

double BenchmarkKickoffLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numCmdbufsSubmitted / elapsedTime;
}

void BenchmarkKickoffLWN::deinit(int subtest)
{
    for (int i = 0; i < m_numCommandBuffers; i++) {
        delete m_commandBuffers[i];
    }
    delete[] m_commandBuffers;
    delete[] m_commandHandles;

    delete m_vertex;
    delete m_mesh;
    delete m_cb;

    lwnProgramFinalize(m_pgm);
    delete m_pgm;

    delete m_cmdBuf;
}

BenchmarkKickoffLWN::~BenchmarkKickoffLWN()
{
}
