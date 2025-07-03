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
//
// MUST BE KEEP IN SYNC WITH trirate.cpp OR ELSE!!

#include "trirate_ogl.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

struct TestDescr
{
    bool colorWrite;
    bool depthWrite;
};

static const TestDescr subtests[] = {
    { false, true },
    { true,  true }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

const int Y_SEGMENTS = BenchmarkTrirateLWN::Y_SEGMENTS; // draw the grid this many times
const int GRIDX = BenchmarkTrirateLWN::GRIDX;
const int GRIDY = BenchmarkTrirateLWN::GRIDY;

static const char *VS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "uniform vec4 offset;\n"
    "uniform vec4 color;\n"
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

BenchmarkTrirateOGL::BenchmarkTrirateOGL(int w, int h) :
    BenchmarkCaseOGL(w, h)
{
}

int BenchmarkTrirateOGL::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkTrirateOGL::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    snprintf(testName, sizeof(testName), "trirate%s%s.ogl",
             t.colorWrite ? ".color=1" : ".color=0",
             t.depthWrite ? ".depth=1" : ".depth=0");

    Description d;
    d.name  = testName;
    d.units = "tris/s";
    return d;
}

#include <stdio.h>

void BenchmarkTrirateOGL::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];

    m_numTrisRendered = 0;

    m_mesh = LwnUtil::OGLMesh::createGrid(GRIDX, GRIDY, Vec2f(-0.5f, 0.f), Vec2f(2.f, 2.f/(float)Y_SEGMENTS), 1.f);

    m_program = new GlProgram();
    std::string vs(VS_STRING);
    std::string fs(FS_STRING);

    m_program->shaderSource(GlProgram::VertexShader, vs);
    m_program->shaderSource(GlProgram::FragmentShader, fs);
    m_program->useProgram();

    LwnUtil::g_glDepthMask(testDescr.depthWrite ? GL_TRUE : GL_FALSE);
    GLboolean col = testDescr.colorWrite ? GL_TRUE : GL_FALSE;
    LwnUtil::g_glColorMask(col, col, col, col);

    LwnUtil::g_glDepthFunc(GL_ALWAYS);
    LwnUtil::g_glDisable(GL_DEPTH_TEST);
}

void BenchmarkTrirateOGL::draw(const DrawParams* params)
{
    m_mesh->bindGeometryGL(0);

    const int offsetLoc = m_program->uniformLocation("offset");
    const int colorLoc  = m_program->uniformLocation("color");

    for (int y = 0; y < Y_SEGMENTS; y++) {
        float t = (float)y / Y_SEGMENTS;
        Vec4f offset = Vec4f(0.f, (t - 0.5f)*2.f, 0.f, 0.f);
        Vec4f color  = Vec4f(t, 0.f, 1.f - t, 1.f);
        LwnUtil::g_glUniform4fv(offsetLoc, 1, (const GLfloat*)&offset);
        LwnUtil::g_glUniform4fv(colorLoc,  1, (const GLfloat*)&color);
        LwnUtil::g_glDrawElements(GL_TRIANGLES, m_mesh->numTriangles()*3, GL_UNSIGNED_INT, 0);

        m_numTrisRendered += m_mesh->numTriangles();
    }
}

double BenchmarkTrirateOGL::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numTrisRendered / elapsedTime;
}

void BenchmarkTrirateOGL::deinit(int subtest)
{
    delete m_mesh;
    delete m_program;
}

BenchmarkTrirateOGL::~BenchmarkTrirateOGL()
{
}
