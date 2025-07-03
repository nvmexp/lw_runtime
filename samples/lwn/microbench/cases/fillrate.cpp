/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Simple speed-of-light (SOL) tests for fillrate.
//
// Speed-of-light (expected theoretical max) for pixels/sec for
// depth-only draw should be around 64 pixels/clock, and for
// depth+color around 13.6 pixels/clock.
//
// SOL for ZLwll reject rate should be 256 pix/clock when we can fully
// cover the screen.  At 1080p with D24S8 we actually cannot cover the
// whole screen but we're still running at roughly 190 pix/clock.
//
// ZLwll tests (fillrate.color=1.depth=1.depthTest=fail*) also tests
// different combinations of ZLwll ilwalidation.  The LWN driver has a
// few "redundant RT change" optimizations which we test for.  These
// cases are:
//
// 1. a) SetRenderTargets(colorA, depthA)
//    b) SetRenderTargets(colorB, null)
//    c) SetRenderTargets(colorA, depthA)
//
// Depth didn't change so we shouldn't ilwalidate or conservative
// clear ZLwll.
//
// 2. a) SetRenderTargets(colorA, depthA)
//    a) SetRenderTargets(colorA, depthA)
//
// Again there was no depth change, so no need for ZLwll ilwalidate.
//
// The cases #1 and #2 above should be running at roughly SOL.  If
// ZLwll RT change filtering is broken, these should be running at
// roughly 0.5 - 3/5'th of SOL.
//
// On 2016-06-09, these were tested to work correctly by intentionally
// breaking the ZLwll MME filter code and observing the drop in
// performance.
//
// We also test for ZLwll ilwalidation when using ZLwll subregions
// (ADAPTIVE_ZLWLL).  In this case, performance will severely drop in
// case render target changes in between rendering.  This is because
// an RT change with subregions puts ZLwll in invalid change, allowing
// for zero ZLwll lwlling.  Sub-cases which use
// "SaveZLwllData/RestoreZLwllData" should however run at close to SOL
// speed because ZLwll is kept valid using save & restore of the ZLwll
// RAM.

#include "fillrate.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkFillrateLWN::TestDescr TestDescr;

const TestDescr BenchmarkFillrateLWN::subtests[] = {
    { false, true, DEPTHTEST_OFF,  false, NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_OFF,  false, NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_PASS, false, NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_FAIL, false, NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { false, true, DEPTHTEST_OFF,  true,  NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_OFF,  true,  NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_PASS, true,  NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_FAIL, true,  NO_RT_CHANGES,        NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_FAIL, false, REDUNDANT_RT_CHANGE,  NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_FAIL, false, COLOR_ONLY_RT_CHANGE, NO_SUBREGIONS,           true,  false },
    { true,  true, DEPTHTEST_FAIL, false, NO_RT_CHANGES,        SUBREGIONS,              true,  false },
    // SOL 256pix/clock case with subregions.  Subregions has a wider
    // variety of internal formats to pick and thus it can cover the
    // whole screen with ZLwll when we use DEPTH24 (sans stencil).
    { true,  true, DEPTHTEST_FAIL, false, NO_RT_CHANGES,        SUBREGIONS,              false, false },
    { true,  true, DEPTHTEST_FAIL, false, COLOR_ONLY_RT_CHANGE, SUBREGIONS,              true,  false },
    { true,  true, DEPTHTEST_FAIL, false, NO_RT_CHANGES,        SUBREGIONS_SAVE_RESTORE, true,  false },
    { true,  true, DEPTHTEST_FAIL, false, COLOR_ONLY_RT_CHANGE, SUBREGIONS_SAVE_RESTORE, true,  false },
    // DEPTH24 case without stencil
    { true,  true, DEPTHTEST_FAIL, false, NO_RT_CHANGES,        NO_SUBREGIONS,           false, false },
    // Normal zlwll discard but start with another RT with subregions
    // enabled, switch to normal depth.  This tests that zlwll gets
    // correctly initialized on first quad render even if we don't
    // explicitly clear the depth buffer.
    { true,  true, DEPTHTEST_FAIL, false,  REDUNDANT_RT_CHANGE, NO_SUBREGIONS,           false, true },
    // Performance in this case should be less than 64 pix/clock
    // because we switch to a subregions enabled depth buffer without
    // clearing it.  This means zlwll should be invalid on first draw,
    // and draws will not validate zlwll, so no zlwll discard should
    // happen.
    { true,  true, DEPTHTEST_FAIL, false,  REDUNDANT_RT_CHANGE, SUBREGIONS,              false, true },
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

static const char *VS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) in vec3 position;\n"
    "out IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  gl_Position = vec4(position, 1.0);\n"
    "  vtxcol = vec4(0.2, 0.4, 0.8, 1.);\n"
    "}\n";

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  color = vec4(0.2, 0.3, 0.5, 1.0);\n"
    "}\n";

BenchmarkFillrateLWN::BenchmarkFillrateLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkFillrateLWN::getNumSubTests()
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

int BenchmarkFillrateLWN::numSubtests() const
{
    return BenchmarkFillrateLWN::getNumSubTests();
}

BenchmarkCase::Description BenchmarkFillrateLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];
    const char* depthTest = nullptr;
    const char* rtMode = nullptr;
    const char* subRegions = nullptr;

    switch (t.depthTest) {
    case DEPTHTEST_OFF:  depthTest = ".depthTest=off"; break;
    case DEPTHTEST_PASS: depthTest = ".depthTest=pass"; break;
    case DEPTHTEST_FAIL: depthTest = ".depthTest=fail"; break;
    default: assert(0);
    }

    if (t.rtMode != NO_RT_CHANGES) {
        switch (t.rtMode) {
        case REDUNDANT_RT_CHANGE:
            rtMode = ".rt=sameColorDepth";
            break;
        case COLOR_ONLY_RT_CHANGE:
            rtMode = ".rt=sameColorNullDepth";
            break;
        default:
            assert(0);
        }
    }

    switch(t.subregions) {
    case NO_SUBREGIONS:
        // nada
        break;
    case SUBREGIONS:
        subRegions = ".subregions=true";
        break;
    case SUBREGIONS_SAVE_RESTORE:
        subRegions = ".subregions=true.saveRestore=true";
        break;
    default:
        assert(0);
    }

    snprintf(testName, sizeof(testName), "fillrate%s%s%s%s%s%s%s%s", 
        t.colorWrite ? ".color=1" : ".color=0",
        t.depthWrite ? ".depth=1" : ".depth=0",
        depthTest,
        t.tiledCache ? ".tc=on" : "",
        rtMode,
        subRegions,
        t.stencil ? "" : ".format=d24",
        t.transitionToFromSubregions ? ".transitionFromSubregions=true" : "");

    Description d;
    d.name  = testName;
    d.units = "pix/s";
    return d;
}

void BenchmarkFillrateLWN::clearBuffers(LWNcommandBuffer *cmd)
{
    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, (m_testDescr->depthTest == DEPTHTEST_FAIL) ? 0.6f : 1.0f, LWN_TRUE, 0, 0);
}

void BenchmarkFillrateLWN::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];
    m_testDescr = &testDescr;

    m_numPixRendered = 0;

    uint32_t texFlags = 0;
    if (!testDescr.stencil) {
        texFlags |= (uint32_t)LwnUtil::RenderTarget::DEPTH_FORMAT_D24;
    }

    m_rtNormal = new LwnUtil::RenderTarget(device(), pools(), width(), height(),
                                           LwnUtil::RenderTarget::CreationFlags(texFlags));

    m_rtSubregions = new LwnUtil::RenderTarget(device(), pools(), width(), height(),
                                               LwnUtil::RenderTarget::CreationFlags(texFlags | LwnUtil::RenderTarget::ADAPTIVE_ZLWLL));

    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, 64*1024, 64*1024);    //TODO use larger chunks, more chunks?
    m_cmd = m_cmdBuf->cmd();

    lwnCommandBufferBeginRecording(m_cmd);

    LwnUtil::RenderTarget *rt1 = m_rtSubregions;
    LwnUtil::RenderTarget *rt2 = m_rtNormal;

    // If subregions is enabled, leave m_rtSubregions active,
    // otherwise bind m_rtNormal last to use non-subregions zlwll.
    if (testDescr.subregions != NO_SUBREGIONS) {
        rt1 = m_rtNormal;
        rt2 = m_rtSubregions;
    }
    rt1->setTargets(m_cmd, 0);
    clearBuffers(m_cmd);
    rt2->setTargets(m_cmd, 0);
    clearBuffers(m_cmd);

    // Create programs from the device, provide them shader code and
    // compile/link them
    m_pgm = new LWNprogram;
    lwnProgramInitialize(m_pgm, device());

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2]   = { VS_STRING, FS_STRING };
    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(m_pgm, stages, nSources, sources))
    {
        assert(0);
    }

    uint32_t depthMask = testDescr.depthWrite ? RenderTarget::DEST_WRITE_DEPTH_BIT : 0;
    uint32_t colorMask = testDescr.colorWrite ? RenderTarget::DEST_WRITE_COLOR_BIT : 0;
    LwnUtil::RenderTarget::setColorDepthMode(m_cmd, depthMask | colorMask, testDescr.depthTest != DEPTHTEST_OFF);

    m_vertex = new LwnUtil::VertexState;
    m_vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_vertex->setStream(0, 12);

    m_mesh = LwnUtil::Mesh::createFullscreenTriangle(device(), coherentPool(), 0.75f);

    PRINTF("  %s: Num pixels  %d (%d / frame)\n", description(subtest).name, width()*height(), 16*width()*height());

    lwnCommandBufferSetTiledCacheAction(m_cmd, testDescr.tiledCache ? LWN_TILED_CACHE_ACTION_ENABLE : LWN_TILED_CACHE_ACTION_DISABLE);

    lwnCommandBufferBindProgram(m_cmd, m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    m_vertex->bind(m_cmd);
    lwnCommandBufferBindVertexBuffer(m_cmd, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(m_cmd);
    m_cmdBuf->submit(1, &cmdHandle);
}

void BenchmarkFillrateLWN::draw(const DrawParams* drawParams)
{
    // TODO this test will not produce any output with --flip --
    // should either blit to drawParams.color or bypass use of m_rt
    // when flip is enabled.

    lwnCommandBufferBeginRecording(m_cmd);

    // Test that we can switch from subregions enabled depth buffer to
    // normal zlwll without needing to clear the depth buffer before
    // rendering into it.  IOW, after a RT switch to normal depth we
    // must be able to clear the depth buffer & validate zlwll by just
    // rendering a quad to it without needing to use an explicit
    // ClearDepthStencil on it.
    if (m_testDescr->transitionToFromSubregions) {
        LWNtexture *color = m_rtSubregions->colorBuffers()[0];;
        LWNtexture *depth = m_rtSubregions->depthBuffer();

        lwnCommandBufferSetRenderTargets(m_cmd, 1, &color, nullptr, depth, nullptr);
    }

    LwnUtil::RenderTarget* rt = m_testDescr->subregions == NO_SUBREGIONS ? m_rtNormal : m_rtSubregions;

    LWNtexture *color = rt->colorBuffers()[0];
    LWNtexture *depth = rt->depthBuffer();

    uint64_t zlwllSaveAddr;
    size_t   zlwllSaveSize;
    rt->getZLwllSaveRestoreBuffer(&zlwllSaveAddr, &zlwllSaveSize);

    for (int i = 0; i < NUM_OVERDRAWS; i++) {
        if ((i % 4) == 0) {
            if (m_testDescr->subregions == SUBREGIONS_SAVE_RESTORE) {
                lwnCommandBufferSaveZLwllData(m_cmd, zlwllSaveAddr, zlwllSaveSize);
            }

            switch (m_testDescr->rtMode) {
            case REDUNDANT_RT_CHANGE:
                lwnCommandBufferSetRenderTargets(m_cmd, 1, &color, nullptr, depth, nullptr);
                lwnCommandBufferSetRenderTargets(m_cmd, 1, &color, nullptr, depth, nullptr);
                break;
            case COLOR_ONLY_RT_CHANGE:
                // Switch to a color only RT, then switch back to original
                lwnCommandBufferSetRenderTargets(m_cmd, 1, &color, nullptr, nullptr, nullptr);
                lwnCommandBufferSetRenderTargets(m_cmd, 1, &color, nullptr, depth, nullptr);
                break;
            case NO_RT_CHANGES:
                // nada
                break;
            default:
                assert(0);
            }

            if (m_testDescr->subregions == SUBREGIONS_SAVE_RESTORE) {
                lwnCommandBufferRestoreZLwllData(m_cmd, zlwllSaveAddr, zlwllSaveSize);
            }
        }

        lwnCommandBufferDrawElements(m_cmd,
                                     LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     m_mesh->numTriangles()*3, m_mesh->iboAddress());

        m_numPixRendered += width()*height();
    }

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(m_cmd);
    m_cmdBuf->submit(1, &cmdHandle);
}

double BenchmarkFillrateLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numPixRendered / elapsedTime;
}

void BenchmarkFillrateLWN::deinit(int subtest)
{
    delete m_mesh;

    delete m_vertex;
    lwnProgramFinalize(m_pgm);
    delete m_pgm;
    delete m_rtNormal;
    delete m_rtSubregions;

    delete m_cmdBuf;
}

BenchmarkFillrateLWN::~BenchmarkFillrateLWN()
{
}
