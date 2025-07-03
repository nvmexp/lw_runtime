/*
* Copyright (c) 2016 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

// Test to compare performance of warp lwlling (vtxAB) by drawing a grid
// offscreen and onscreen, with and without warp lwlling enabled

// warp lwlling:
// 1) helps lwll prims and prevents them from going to PES thereby reducing PES workload.
// 2) allows VS to run in shared ISBE mode(subject to code scheduling - all ALDs must be scheduled ahead of all ASTs)
// 3) saves SM work(attribute stores) by exiting early(if lwll condition is true) and avoiding ASTs that occur later in the program.

#include "warp_lwlling.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

struct TestDescr
{
    bool offscreen;
    bool warplwll;
};

// offscreen, warplwll
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
using LwnUtil::RenderTarget;

// Vertex shader which outputs a lot of attributes.
// With warp lwlling the vtxOutput* attributes will be
// discarded and not callwlated, resulting in better performance
static const char *VS_STRING =
"#version 440 core\n"
"#extension GL_LW_gpu_shader5:require\n"
"layout(location = 0) in vec3 position;\n"
"out IO { vec4 vtxOutput[4]; };\n"
"void main() {\n"
"  gl_Position = vec4(position, 1.0);\n"
"  vtxOutput[0] = vec4(position, 1.0) * 1.3;\n"
"  vtxOutput[1] = vec4(0.4, 0.3, 0.2, 1.0);\n"
"  vtxOutput[2] = vec4(position, 0.5) * vec4(0.3, 0.1, 2.3, 0.8);\n"
"  vtxOutput[3] = vec4(position, 0.7) * vec4(0.14, 5.5, 0.15, 1.0) + vec4(-0.5, 0.4, 0.12, 0.45);\n"
"}\n";

// Fragment shader which uses all the inputs from the VS to produce the final color
static const char *FS_STRING =
"#version 440 core\n"
"#extension GL_LW_gpu_shader5:require\n"
"layout(location = 0) out vec4 color;\n"
"in IO { vec4 vtxOutput[4]; };\n"
"void main() {\n"
"  color = vtxOutput[0] * vtxOutput[1] * vtxOutput[2] + vtxOutput[3];\n"
"}\n";

BenchmarkWarpLwllingLWN::BenchmarkWarpLwllingLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkWarpLwllingLWN::numSubtests() const
{
    return sizeof(subtests) / sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkWarpLwllingLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    snprintf(testName, sizeof(testName), "warp_lwlling%s%s",
             t.offscreen ? ".offscreen=1" : ".offscreen=0",
             t.warplwll ? ".warplwll=on" : ".warplwll=off");

    Description d;
    d.name = testName;
    d.units = "ms";
    return d;
}

void BenchmarkWarpLwllingLWN::init(int subtest)
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

    // enable/disable warp lwlling
    LwnUtil::enableWarpLwlling(testDescr.warplwll);

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
    m_mesh = LwnUtil::Mesh::createGrid(device(), coherentPool(), GRIDX, GRIDY, testDescr.offscreen? Vec2f(-5.0f, 0.0f) : Vec2f(0.0f, 0.0f) , Vec2f(1.f, 1.f / (float)Y_SEGMENTS), 0.0f);

    lwnCommandBufferBindProgram(m_cmd, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);

    m_vertex->bind(m_cmd);

    lwnCommandBufferBindVertexBuffer(m_cmd, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));

    // start timing
    lwnCommandBufferReportCounter(m_cmd, LWN_COUNTER_TYPE_TIMESTAMP, m_counters->address() + offsetof(GpuCounters, gpuTime0));
}

void BenchmarkWarpLwllingLWN::draw(const DrawParams* params)
{
    if (!lwnCommandBufferIsRecording(m_cmd))
    {
        lwnCommandBufferBeginRecording(m_cmd);
        lwnCommandBufferBindProgram(m_cmd, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);

        m_vertex->bind(m_cmd);

        lwnCommandBufferBindVertexBuffer(m_cmd, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));
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

double BenchmarkWarpLwllingLWN::measuredValue(int subtest, double elapsedTime)
{
    // get elapsed time (in ms)
    const GpuCounters *counterVA = (const GpuCounters *)m_counters->ptr();
    uint64_t gpuTimeNs0 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime0);
    uint64_t gpuTimeNs1 = lwnDeviceGetTimestampInNanoseconds(device(), &counterVA->gpuTime1);
    uint64_t gpuTimeNs = gpuTimeNs1 - gpuTimeNs0;
    double m_gpuTime = (double)gpuTimeNs / 1000000.0f;

    return (double)m_gpuTime;
}

void BenchmarkWarpLwllingLWN::deinit(int subtest)
{
    delete m_mesh;
    delete m_vertex;
    delete m_counters;

    lwnProgramFinalize(m_pgm);
    delete m_pgm;

    delete m_cmdBuf;
}

BenchmarkWarpLwllingLWN::~BenchmarkWarpLwllingLWN()
{
}
