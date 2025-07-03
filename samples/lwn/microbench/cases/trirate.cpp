/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Simple speed-of-light (SOL) tests for things triangles/sec.
//
// Speed-of-light (expected theoretical max) for tris/sec for
// depth-only draw should be around 0.7 triangles/clock.

#include "trirate.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

struct TestDescr
{
    bool colorWrite;
    bool depthWrite;
    bool tiledCache;
};

static const TestDescr subtests[] = {
    { false, true, false },
    { true,  true, false, },
    { false, true, true },
    { true,  true, true, }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

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

BenchmarkTrirateLWN::BenchmarkTrirateLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkTrirateLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkTrirateLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    snprintf(testName, sizeof(testName), "trirate%s%s%s",
             t.colorWrite ? ".color=1" : ".color=0",
             t.depthWrite ? ".depth=1" : ".depth=0",
             t.tiledCache ? ".tc=on" : "");

    Description d;
    d.name  = testName;
    d.units = "tris/s";
    return d;
}

#include <stdio.h>

void BenchmarkTrirateLWN::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];

    m_numTrisRendered = 0;

    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, 64*1024, 64*1024);
    m_cmd = m_cmdBuf->cmd();

    lwnCommandBufferBeginRecording(m_cmd);

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

    // We will create a pipeline, if we aren't using raw data structures.
    uint32_t depthMask = testDescr.depthWrite ? RenderTarget::DEST_WRITE_DEPTH_BIT : 0;
    uint32_t colorMask = testDescr.colorWrite ? RenderTarget::DEST_WRITE_COLOR_BIT : 0;
    LwnUtil::RenderTarget::setColorDepthMode(m_cmd, depthMask | colorMask, false);

    m_vertex = new LwnUtil::VertexState;
    m_vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_vertex->setStream(0, 12);

    m_mesh = LwnUtil::Mesh::createGrid(device(), coherentPool(), GRIDX, GRIDY, Vec2f(-0.5f, 0.f), Vec2f(2.f, 2.f/(float)Y_SEGMENTS), 1.f);

    lwnCommandBufferSetTiledCacheAction(m_cmd, testDescr.tiledCache ? LWN_TILED_CACHE_ACTION_ENABLE : LWN_TILED_CACHE_ACTION_DISABLE);

    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(m_cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(m_cmd, 1.0, LWN_TRUE, 0, 0);

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(m_cmd);
    m_cmdBuf->submit(1, &cmdHandle);

    m_cb = new LwnUtil::UboArr<SegAttrs>(device(), coherentPool(), Y_SEGMENTS);

    for (int y = 0; y < Y_SEGMENTS; y++) {
        SegAttrs a;
        float t = (float)y / Y_SEGMENTS;
        a.offset = Vec4f(0.f, (t - 0.5f)*2.f, 0.f, 0.f);
        a.color  = Vec4f(t, 0.f, 1.f - t, 1.f);
        m_cb->set(y, a);
    }
}

void BenchmarkTrirateLWN::draw(const DrawParams* params)
{
    lwnCommandBufferBeginRecording(m_cmd);
    lwnCommandBufferBindProgram(m_cmd, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    m_vertex->bind(m_cmd);

    lwnCommandBufferBindVertexBuffer(m_cmd, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));

    for (int y = 0; y < Y_SEGMENTS; y++) {
        lwnCommandBufferBindUniformBuffer(m_cmd, LWN_SHADER_STAGE_VERTEX,
                                          0,
                                          m_cb->address() + m_cb->offset(y),
                                          sizeof(SegAttrs));

        lwnCommandBufferDrawElements(m_cmd, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     m_mesh->numTriangles()*3, m_mesh->iboAddress());
        m_numTrisRendered += numTris;
    }

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(m_cmd);
    m_cmdBuf->submit(1, &cmdHandle);
}

double BenchmarkTrirateLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numTrisRendered / elapsedTime;
}

void BenchmarkTrirateLWN::deinit(int subtest)
{
    delete m_mesh;
    delete m_cb;

    delete m_vertex;
    lwnProgramFinalize(m_pgm);
    delete m_pgm;

    delete m_cmdBuf;
}

BenchmarkTrirateLWN::~BenchmarkTrirateLWN()
{
}
