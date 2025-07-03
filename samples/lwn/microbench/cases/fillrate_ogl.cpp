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
// WARNING NOTE WARNING!! must keep rendering logic in sync with fillrate.cpp!!

#include "fillrate_ogl.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkFillrateLWN::TestDescr TestDescr;

static const BenchmarkFillrateLWN::SubregionMode subreg = BenchmarkFillrateLWN::NO_SUBREGIONS;

#define ENTRY(colorWrite, depthWrite, depthTest) { colorWrite, depthWrite, depthTest, true, BenchmarkFillrateLWN::NO_RT_CHANGES, subreg, true, false }

static const TestDescr subtests[] = {
    ENTRY(false, true, BenchmarkFillrateLWN::DEPTHTEST_OFF),
    ENTRY(true,  true, BenchmarkFillrateLWN::DEPTHTEST_OFF),
    ENTRY(true,  true, BenchmarkFillrateLWN::DEPTHTEST_PASS),
    ENTRY(true,  true, BenchmarkFillrateLWN::DEPTHTEST_FAIL)
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

// TODO share with trirate.cpp
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
    "  color = vec4(1.2, 0.3, 0.5, 1.0);\n"
    "}\n";

BenchmarkFillrateOGL::BenchmarkFillrateOGL(int w, int h) :
    BenchmarkCaseOGL(w, h)
{
}

int BenchmarkFillrateOGL::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkFillrateOGL::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];
    const char* depthTest = nullptr;

    switch (t.depthTest) {
    case BenchmarkFillrateLWN::DEPTHTEST_OFF:  depthTest = ".depthTest=off"; break;
    case BenchmarkFillrateLWN::DEPTHTEST_PASS: depthTest = ".depthTest=pass"; break;
    case BenchmarkFillrateLWN::DEPTHTEST_FAIL: depthTest = ".depthTest=fail"; break;
    default: assert(0);
    }

    snprintf(testName, sizeof(testName), "fillrate%s%s%s.ogl",
             t.colorWrite ? ".color=1" : ".color=0",
             t.depthWrite ? ".depth=1" : ".depth=0",
             depthTest);

    Description d;
    d.name  = testName;
    d.units = "pix/s";
    return d;
}

#include <stdio.h>

void BenchmarkFillrateOGL::init(int subtest)
{
    const TestDescr& testDescr = BenchmarkFillrateLWN::subtests[subtest];

    m_numPixRendered = 0;

    m_program = new GlProgram();
    std::string vs(VS_STRING);
    std::string fs(FS_STRING);

    m_program->shaderSource(GlProgram::VertexShader, vs);
    m_program->shaderSource(GlProgram::FragmentShader, fs);
    m_program->useProgram();

    LwnUtil::g_glDepthMask(testDescr.depthWrite ? GL_TRUE : GL_FALSE);
    GLboolean col = testDescr.colorWrite ? GL_TRUE : GL_FALSE;
    LwnUtil::g_glColorMask(col, col, col, col);

    switch (testDescr.depthTest) {
    case BenchmarkFillrateLWN::DEPTHTEST_OFF:
        LwnUtil::g_glDepthFunc(GL_ALWAYS);
        LwnUtil::g_glDisable(GL_DEPTH_TEST);
        break;
    case BenchmarkFillrateLWN::DEPTHTEST_PASS: /* FALL-THRU */
    case BenchmarkFillrateLWN::DEPTHTEST_FAIL:
        LwnUtil::g_glDepthFunc(GL_LEQUAL);
        LwnUtil::g_glEnable(GL_DEPTH_TEST);
        break;
    }

    float clearDepth = (testDescr.depthTest == BenchmarkFillrateLWN::DEPTHTEST_FAIL) ? 0.6f : 1.0f;
    LwnUtil::g_glClearDepthf(clearDepth);
    LwnUtil::g_glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_mesh = LwnUtil::OGLMesh::createFullscreenTriangle(0.75f);
}

void BenchmarkFillrateOGL::draw(const DrawParams* params)
{
    m_mesh->bindGeometryGL(0);

    for (int i = 0; i < BenchmarkFillrateLWN::NUM_OVERDRAWS; i++)
    {
        LwnUtil::g_glDrawElements(GL_TRIANGLES, m_mesh->numTriangles()*3, GL_UNSIGNED_INT, 0);
        m_numPixRendered += width()*height();
    }
}

double BenchmarkFillrateOGL::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numPixRendered / elapsedTime;
}

void BenchmarkFillrateOGL::deinit(int subtest)
{
    delete m_mesh;
    delete m_program;
}

BenchmarkFillrateOGL::~BenchmarkFillrateOGL()
{
}
