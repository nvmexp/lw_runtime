/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Tiled caching test

#include "tiled_cache.hpp"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "lwnUtil/lwnUtil_TiledCacheState.h"
#include "lwnUtil/lwnUtil_TiledCacheStateImpl.h"

static const BenchmarkTiledCacheLWN::TestDescr subtests[] = {
    { false, false /* No MSAA */, false },
    { true,  false /* No MSAA */, false },
    { false, false /* No MSAA */, true },
    { true,  false /* No MSAA */, true },
    { false, true  /* 4x MSAA */, false },
    { true,  true  /* 4x MSAA */, false },
    { false, true  /* 4x MSAA */, true },
    { true,  true  /* 4x MSAA */, true },
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

const int OVERDRAW = 30; // render the object grid this many times
const int OBJS_X = 7;
const int OBJS_Y = 7;

// How many instances to draw
const int N_DRAWS = OBJS_X * OBJS_Y * OVERDRAW;

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

BenchmarkTiledCacheLWN::BenchmarkTiledCacheLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkTiledCacheLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkTiledCacheLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    snprintf(testName, sizeof(testName), "tiled_cache.tc=%s.msaa=%s.mode=%s",
             t.tiledCache ? "on" : "off",
             t.msaa ? "4x" : "off",
             t.autoMode ? "auto" : "default");

    Description d;
    d.name  = testName;
    d.units = "draw/s";
    return d;
}

LWNcommandHandle BenchmarkTiledCacheLWN::renderCommands(LWNcommandBuffer* cmdBuffer)
{
    lwnCommandBufferBeginRecording(cmdBuffer);

    float clearColor[] = { 0, 0, 0, 1 };

    lwnCommandBufferClearColor(cmdBuffer, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmdBuffer, 1.0, LWN_TRUE, 0, 0);

    // Disable depth test
    LWNdepthStencilState depth;
    // Create new state vectors
    lwnDepthStencilStateSetDefaults(&depth);
    lwnDepthStencilStateSetDepthTestEnable(&depth, false);
    lwnDepthStencilStateSetDepthWriteEnable(&depth, true);
    lwnCommandBufferBindDepthStencilState(cmdBuffer, &depth);

    LWNcolorState colorState;
    lwnColorStateSetDefaults(&colorState);
    lwnColorStateSetBlendEnable(&colorState, 0, LWN_TRUE);
    lwnCommandBufferBindColorState(cmdBuffer, &colorState);

    // Additive blend
    LWNblendState blend;
    lwnBlendStateSetDefaults(&blend);
    lwnBlendStateSetBlendFunc(&blend, LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE,
                              LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE);
    lwnBlendStateSetBlendEquation(&blend, LWN_BLEND_EQUATION_ADD, LWN_BLEND_EQUATION_ADD);
    lwnCommandBufferBindBlendState(cmdBuffer, &blend);

    lwnCommandBufferBindProgram(cmdBuffer, &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    m_res->vertex->bind(cmdBuffer);
    lwnCommandBufferBindVertexBuffer(cmdBuffer, 0, m_res->mesh->vboAddress(), m_res->mesh->numVertices()*sizeof(Vec3f));

    LWNbufferAddress cbAddress      = m_res->cb->address();
    LWNbufferAddress iboAddress     = m_res->mesh->iboAddress();
    const int        numPrimitives = m_res->mesh->numTriangles()*3;

    for (int drawIdx = 0; drawIdx < N_DRAWS; drawIdx++) {
        lwnCommandBufferBindUniformBuffer(cmdBuffer, LWN_SHADER_STAGE_VERTEX,
                                          0,
                                          cbAddress + m_res->cb->offset(drawIdx),
                                          sizeof(SegAttrs));

        lwnCommandBufferDrawElements(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     numPrimitives, iboAddress);
    }
    return lwnCommandBufferEndRecording(cmdBuffer);
}

void BenchmarkTiledCacheLWN::init(int subtest)
{
    m_res = new Resources();

    const TestDescr& testDescr = subtests[subtest];
    m_testDescr = &testDescr;

    m_numInstancesRendered = 0;

    RenderTarget::CreationFlags rtFlags = RenderTarget::CreationFlags(0);
    if (testDescr.msaa) {
        rtFlags = RenderTarget::MSAA_4X;
    }

    m_res->rt.reset(new LwnUtil::RenderTarget(device(), pools(), width(), height(), rtFlags));

    m_res->cmdBuf.reset(new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 32, 65536*4, 16384));

    LWNcommandBuffer* cmd = m_res->cmdBuf->cmd();
    lwnCommandBufferBeginRecording(cmd);

    // Create programs from the device, provide them shader code and compile/link them
    lwnProgramInitialize(&m_pgm, device());

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2]   = { VS_STRING, FS_STRING };
    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(&m_pgm, stages, nSources, sources))
    {
        assert(0); // TBD
    }

    m_res->vertex.reset(new LwnUtil::VertexState);
    m_res->vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_res->vertex->setStream(0, 12);

    int numSegments = 5;
    m_res->mesh.reset(LwnUtil::Mesh::createCircle(device(), coherentPool(), numSegments, Vec2f(1.f/(float)OBJS_X, 1.f/(float)OBJS_Y), 1.f));

    lwnCommandBufferSetTiledCacheAction(cmd, testDescr.tiledCache ? LWN_TILED_CACHE_ACTION_ENABLE : LWN_TILED_CACHE_ACTION_DISABLE);

    // Perform tiled cache "auto heuristics" to setup appropriate TC
    // parameters to the GPU.
    if (testDescr.tiledCache && testDescr.autoMode) {
        lwn::util::TiledCacheState tcState(device());
        tcState.SetStrategy(lwn::util::TiledCacheState::SKIP_STENCIL_COMPONENT_BIT);

        LWNtexture* colors[] = { m_res->rt->colorBuffers()[0] };
        tcState.UpdateTileState(cmd, 1, m_res->rt->getNumSamples(), colors, nullptr);
    }


    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_res->cmdBuf->submit(1, &cmdHandle);

    m_res->cb.reset(new LwnUtil::UboArr<SegAttrs>(device(), coherentPool(), N_DRAWS));

    srand(0xdeadbeef);

    int dstIdx = 0;
    for (int pass = 0; pass < OVERDRAW; pass++) {
        for (int row = 0; row < OBJS_Y; row++) {
            for (int col = 0; col < OBJS_X; col++) {
                SegAttrs a;
                float u = ((float)col/(float)OBJS_X);
                float v = ((float)row/(float)OBJS_Y);
                float x = (u - 0.5f) * 2.f + (1.f/(float)OBJS_X);
                float y = (v - 0.5f) * 2.f + (1.f/(float)OBJS_Y);

                float jitx = 0.2f * (((rand() & 255) / 255.f) - 0.5f);
                float jity = 0.2f * (((rand() & 255) / 255.f) - 0.5f);

                a.offset = Vec4f(x+jitx, y+jity, 0.f, 0.f);
                a.color  = Vec4f(u*0.2f, v*0.2f, 0.2f, 0.f);

                m_res->cb->set(dstIdx++, a);
            }
        }
    }

    m_prebuiltCmdHandle = renderCommands(cmd);
}

void BenchmarkTiledCacheLWN::draw(const DrawParams* drawParams)
{
    LWNcommandBuffer* cmd = m_res->cmdBuf->cmd();
    lwnCommandBufferBeginRecording(cmd);

    BenchmarkCaseLWN::DrawParamsLWN* params = (BenchmarkCaseLWN::DrawParamsLWN*)drawParams;

    m_numInstancesRendered += N_DRAWS;

    // Bind MSAA target
    if (m_res->rt->getNumSamples() != 1) {
        m_res->rt->setTargets(cmd, 0);
    }

    lwnCommandBufferCallCommands(cmd, 1, &m_prebuiltCmdHandle);

    if (m_res->rt->getNumSamples() != 1) {
        // NOTE:
        //
        // The downsample operation in the current version of LWN does
        // the MSAA resolve in TWOD.  According to matthewj, this
        // operation is NOT binned.  So instead of using downsample()
        // here, idelly we should use 3D to perform the downsample so
        // that the whole operation is binned.
        //
        // Something like this:
        //
        // 1. a tiled barrier for writing to source MSAA from prev work
        // 2. a 3D engine binned downsample from source MSAA -> dest
        //    1x AA (the app can do this)
        // 3. discard source MSAA color (then depth)
        // 4. present


        m_res->rt->downsample(cmd, params->dstColor);

        if (m_testDescr->tiledCache) {
            lwnCommandBufferSetTiledCacheAction(cmd, LWN_TILED_CACHE_ACTION_FLUSH);
        }
    }

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_res->cmdBuf->submit(1, &cmdHandle);
}

double BenchmarkTiledCacheLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numInstancesRendered / elapsedTime;
}

void BenchmarkTiledCacheLWN::deinit(int subtest)
{
    delete m_res;
    lwnProgramFinalize(&m_pgm);
}

BenchmarkTiledCacheLWN::~BenchmarkTiledCacheLWN()
{
}
