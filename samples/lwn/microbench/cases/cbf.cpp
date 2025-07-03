/*
* Copyright (c) 2016 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

// Test to compare performance of Lwll-Before-Fetch (CBF) by drawing a grid
// offscreen and onscreen, with and without CBF enabled

// Lwll-Before-Fetch:
// 1) Generates two vertex shaders (vertexA and vertexB)
// 2) vertexA is a simplified version of the vertex shader which only fetches the position attribute, while vertexB is the regular shader.
// 3) If vertexA determines that the vertex is outside of the frustum we don't bother running vertexB shader.
// 4) Thus CBF reduces VAF b/w in high lwlling cases that use lots of attributes and are outside the frustum, and helps eliminate work for PES by removing prims early.
// 5) CBF can have a significant negative perf impact if vertices are not outside the frustum, since we essentially end up running the vertex shader twice

#include "cbf.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

struct TestDescr
{
    bool offscreen;
    bool cbf;
};

// offscreen, cbf
static const TestDescr subtests[] = {
    { false, false },
    { false,  true },
    {  true, false },
    {  true,  true },
};

struct GpuCounters
{
    LWNcounterData gpuTime0;
    LWNcounterData gpuTime1;
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

// Vertex shader which has a lot of vertex attributes
// With CBF we expect VAF traffic to be reduced
#define NUM_ATTRIBS 6
static const char *VS_STRING =
"#version 440 core\n"
"#extension GL_LW_gpu_shader5:require\n"
"layout(location = 0) in vec3 position;\n"
"layout(location = 1) in vec4 vtxInput[6];\n"
"out IO { vec4 vtxOutput; };\n"
"void main() {\n"
"  gl_Position = vec4(position, 1.0);\n"
"  vtxOutput = vec4(1.3, 0.2, 0.07, 1.9) * vtxInput[0] + vtxInput[1] + vtxInput[2] + vtxInput[3] + vtxInput[4] + vtxInput[5];\n"
"}\n";

// Fragment shader which uses the input from the VS to produce the final color
static const char *FS_STRING =
"#version 440 core\n"
"#extension GL_LW_gpu_shader5:require\n"
"layout(location = 0) out vec4 color;\n"
"in IO { vec4 vtxOutput; };\n"
"void main() {\n"
"  color = vtxOutput;\n"
"}\n";

BenchmarkCBFLWN::BenchmarkCBFLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkCBFLWN::numSubtests() const
{
    return sizeof(subtests) / sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkCBFLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    snprintf(testName, sizeof(testName),
             "cbf%s%s",
             t.offscreen ? ".offscreen=1" : ".offscreen=0",
             t.cbf ? ".cbf=on" : ".cbf=off");

    Description d;
    d.name = testName;
    d.units = "ms";
    return d;
}

void BenchmarkCBFLWN::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];

    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, 64 * 1024, 64 * 1024);
    m_cmd = m_cmdBuf->cmd();

    // create buffer for counters
    const int counterBufSize = sizeof(GpuCounters);
    m_counters = new LwnUtil::Buffer(device(), coherentPool(), nullptr, counterBufSize,
        BUFFER_ALIGN_COUNTER_BIT);

    // start recording commands
    lwnCommandBufferBeginRecording(m_cmd);

    lwnCommandBufferSetTiledCacheAction(m_cmd, LWN_TILED_CACHE_ACTION_DISABLE);
    LwnUtil::RenderTarget::setColorDepthMode(m_cmd, RenderTarget::DEST_WRITE_DEPTH_BIT | RenderTarget::DEST_WRITE_COLOR_BIT, true);

    float clearColor[] = { 0.0, 0.0, 0.0, 1 };
    lwnCommandBufferClearColor(m_cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(m_cmd, 1.0, LWN_TRUE, 0, 0xFF);

    // Create programs from the device, provide them shader code and compile/link them
    m_pgm = new LWNprogram;
    lwnProgramInitialize(m_pgm, device());

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2] = { VS_STRING, FS_STRING };
    int32_t nSources = 2;

    // enable/disable cbf
    LwnUtil::enableCBF(testDescr.cbf);

    // compile shaders
    if (!LwnUtil::compileAndSetShaders(m_pgm, stages, nSources, sources))
    {
        assert(0);
    }

    // Setup grid
    // If offscreen, set offset to Vec2f(-0.5, 0.0)
    m_vertex = new LwnUtil::VertexState;
    m_vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_vertex->setStream(0, 12);
    m_vertex->setAttribute(1, LWN_FORMAT_RGBA32F, 0, 0);
    m_vertex->setStream(1, 16 * NUM_ATTRIBS); // vec4s * NUM_ATTRIBS

    m_mesh = LwnUtil::Mesh::createGrid(device(), coherentPool(), GRIDX, GRIDY, testDescr.offscreen? Vec2f(-5.0f, 0.0f) : Vec2f(0.0f, 0.0f) , Vec2f(1.f, 1.f / (float)Y_SEGMENTS), 0.0f);

    // generate more vertex attributes
    m_vertexAttributes = new LwnUtil::Buffer(device(), coherentPool(), nullptr, m_mesh->numVertices()*sizeof(Vec4f)*NUM_ATTRIBS,
        BUFFER_ALIGN_VERTEX_BIT);
    memset(m_vertexAttributes->ptr(), 1, m_mesh->numVertices()*sizeof(Vec4f)*NUM_ATTRIBS);

    lwnCommandBufferBindProgram(m_cmd, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);

    m_vertex->bind(m_cmd);

    lwnCommandBufferBindVertexBuffer(m_cmd, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));
    lwnCommandBufferBindVertexBuffer(m_cmd, 1, m_vertexAttributes->address(), m_mesh->numVertices()*sizeof(Vec4f)*NUM_ATTRIBS);

    // start timing
    lwnCommandBufferReportCounter(m_cmd, LWN_COUNTER_TYPE_TIMESTAMP, m_counters->address() + offsetof(GpuCounters, gpuTime0));
}

void BenchmarkCBFLWN::draw(const DrawParams* params)
{
    if (!lwnCommandBufferIsRecording(m_cmd))
    {
        lwnCommandBufferBeginRecording(m_cmd);
        lwnCommandBufferBindProgram(m_cmd, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);

        m_vertex->bind(m_cmd);

        lwnCommandBufferBindVertexBuffer(m_cmd, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));
        lwnCommandBufferBindVertexBuffer(m_cmd, 1, m_vertexAttributes->address(), m_mesh->numVertices() * 4 * sizeof(Vec4f));
    }

    // draw the grid
    for (int y = 0; y < Y_SEGMENTS; y++) {
        lwnCommandBufferDrawElements(m_cmd, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
            m_mesh->numTriangles() * 3, m_mesh->iboAddress());
    }

    // record final timestamp
    lwnCommandBufferReportCounter(m_cmd, LWN_COUNTER_TYPE_TIMESTAMP, m_counters->address() + offsetof(GpuCounters, gpuTime1));

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(m_cmd);
    m_cmdBuf->submit(1, &cmdHandle);
}

double BenchmarkCBFLWN::measuredValue(int subtest, double elapsedTime)
{
    // get elapsed time (in ms)
    const GpuCounters *counterVA = (const GpuCounters *)m_counters->ptr();
    uint64_t gpuTimeNs0 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime0);
    uint64_t gpuTimeNs1 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime1);
    uint64_t gpuTimeNs = gpuTimeNs1 - gpuTimeNs0;
    double m_gpuTime = (double)gpuTimeNs / 1000000.0f;

    return (double)m_gpuTime;
}

void BenchmarkCBFLWN::deinit(int subtest)
{
    LwnUtil::enableCBF(false);
    delete m_mesh;
    delete m_vertex;
    delete m_vertexAttributes;
    delete m_counters;

    lwnProgramFinalize(m_pgm);
    delete m_pgm;

    delete m_cmdBuf;
}

BenchmarkCBFLWN::~BenchmarkCBFLWN()
{
}
