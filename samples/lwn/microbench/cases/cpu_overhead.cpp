/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Draw call overhead with uniform and shader switches at varying draw
// call frequencies.  Intended to show case simple CPU overhead
// difference between typical LWN and OGL usage.  OGL version can
// obviously be sped up by using various AZDO techniques.

#include "cpu_overhead.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkCase::Description Description;
typedef BenchmarkCpuOverheadLWN::SegAttrs SegAttrs;
typedef BenchmarkCpuOverheadLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { 10, true,  BenchmarkCpuOverheadLWN::UNIFORM_UBO },
    { 10, false, BenchmarkCpuOverheadLWN::UNIFORM_UBO },
    { 16, true,  BenchmarkCpuOverheadLWN::UNIFORM_UBO }
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

#define UNIFORM_INPUT_LWN \
    "layout(binding = 0) uniform segmentBlock {\n" \
    "  vec4 offset;\n" \
    "  vec4 color;\n" \
    "};\n"

#define N_UBO_ELEMS 256

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define UNIFORM_INPUT_OGL_UBO \
    "struct ObjData {\n" \
    "  vec4 offset;\n" \
    "  vec4 color;\n" \
    "  vec4 padding[14];\n" \
    "};\n" \
    "layout (binding=0, std140) uniform Objs { ObjData objs[" STR(N_UBO_ELEMS) "]; };\n"

#define UNIFORM_INPUT_OGL \
    "uniform vec4 offset;\n" \
    "uniform vec4 color;\n"

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

static const char *VS_STRING_OGL =
    PROLOG
    UNIFORM_INPUT_OGL
    VS("offset", "color")

static const char *VS_STRING_OGL_UBO =
    PROLOG
    UNIFORM_INPUT_OGL_UBO
    VS("objs[gl_InstanceID].offset", "objs[gl_InstanceID].color")

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  color = vtxcol;\n" // vtxcol
    "}\n";

BenchmarkCpuOverheadLWN::BenchmarkCpuOverheadLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkCpuOverheadLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

static Description testDescription(const TestDescr& t, const char* postfix)
{
    static char testName[256];
    const char *ubo = nullptr;

    switch (t.uniformMode) {
    case BenchmarkCpuOverheadLWN::UNIFORM_INLINE:
        ubo = "inline";
        break;
    case BenchmarkCpuOverheadLWN::UNIFORM_UBO:
        ubo = "ubo";
        break;
    }

    snprintf(testName, sizeof(testName),
            "cpu_overhead.numSegments=%d.precomp=%d.uniform=%s%s",
            t.numSegments, (int)t.precompiled, ubo, postfix);

    Description d;
    d.name  = testName;
    d.units = "draw/s";
    return d;
}

Description BenchmarkCpuOverheadLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    return testDescription(t, "");
}

LWNcommandHandle BenchmarkCpuOverheadLWN::renderCommands(LWNcommandBuffer* cmdBuffer)
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

void BenchmarkCpuOverheadLWN::init(int subtest)
{
    m_subtestIdx = subtest;
    const TestDescr& testDescr = subtests[subtest];

    m_numInstancesRendered = 0;

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

    int numSegments = testDescr.numSegments;
    m_mesh = LwnUtil::Mesh::createCircle(device(), coherentPool(), numSegments, Vec2f(1.f/(float)OBJS_X, 1.f/(float)OBJS_Y), 1.f);

    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0, LWN_TRUE, 0, 0);

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_cmdBuf->submit(1, &cmdHandle);

    m_cb = new LwnUtil::UboArr<SegAttrs>(device(), coherentPool(), N_DRAWS);

    std::vector<SegAttrs> objs(N_DRAWS);
    initObjectAttrs(objs.data());
    for (int i = 0; i < N_DRAWS; i++) {
        m_cb->set(i, objs[i]);
    }

    if (testDescr.precompiled) {
        m_prebuiltCmdHandle = renderCommands(cmd);
    }
}

void BenchmarkCpuOverheadLWN::draw(const DrawParams* params)
{
    const TestDescr& testDescr = subtests[m_subtestIdx];

    m_numInstancesRendered += N_DRAWS;

    if (testDescr.precompiled) {
        m_cmdBuf->submit(1, &m_prebuiltCmdHandle);
    } else {
        LWNcommandBuffer* cmd = m_cmdBuf->cmd();
        LWNcommandHandle  cmdHandle = renderCommands(cmd);
        m_cmdBuf->submit(1, &cmdHandle);
    }
}

double BenchmarkCpuOverheadLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numInstancesRendered / elapsedTime;
}

void BenchmarkCpuOverheadLWN::deinit(int subtest)
{
    delete m_vertex;
    delete m_mesh;
    delete m_cb;

    lwnProgramFinalize(m_pgm);
    delete m_pgm;

    delete m_cmdBuf;
}

BenchmarkCpuOverheadLWN::~BenchmarkCpuOverheadLWN()
{
}

//--------------------------------------------------------------------
// OGL version
//--------------------------------------------------------------------

static const TestDescr subtestsOGL[] = {
    { 10, false, BenchmarkCpuOverheadLWN::UNIFORM_INLINE },
    { 16, false, BenchmarkCpuOverheadLWN::UNIFORM_INLINE },
    { 10, false, BenchmarkCpuOverheadLWN::UNIFORM_UBO },
    { 16, false, BenchmarkCpuOverheadLWN::UNIFORM_UBO }
};

BenchmarkCpuOverheadOGL::BenchmarkCpuOverheadOGL(int w, int h) :
    BenchmarkCaseOGL(w, h)
{
}

int BenchmarkCpuOverheadOGL::numSubtests() const
{
    return sizeof(subtestsOGL)/sizeof(subtestsOGL[0]);
}

Description BenchmarkCpuOverheadOGL::description(int subtest) const
{
    const TestDescr& t = subtestsOGL[subtest];
    return testDescription(t, ".ogl");
}

void BenchmarkCpuOverheadOGL::init(int subtest)
{
    const TestDescr& testDescr = subtestsOGL[subtest];

    m_testDescr = &testDescr;
    m_numInstancesRendered = 0;

    m_program = new GlProgram();

    std::string fs(FS_STRING);

    switch (testDescr.uniformMode) {
    case BenchmarkCpuOverheadLWN::UNIFORM_INLINE:
        {
            std::string vs(VS_STRING_OGL);
            m_program->shaderSource(GlProgram::VertexShader, vs);
        }
        break;
    case BenchmarkCpuOverheadLWN::UNIFORM_UBO:
        {
            std::string vs(VS_STRING_OGL_UBO);
            m_program->shaderSource(GlProgram::VertexShader, vs);
        }
        break;
    }

    m_program->shaderSource(GlProgram::FragmentShader, fs);
    m_program->useProgram();

    m_mesh = LwnUtil::OGLMesh::createCircle(testDescr.numSegments,
                                            Vec2f(1.f/(float)OBJS_X, 1.f/(float)OBJS_Y), 1.f);

    m_objectAttrs = new SegAttrs[N_DRAWS];
    initObjectAttrs(m_objectAttrs);

    if (testDescr.uniformMode == BenchmarkCpuOverheadLWN::UNIFORM_UBO) {
        LwnUtil::g_glGenBuffers(1, &m_ubo);
        LwnUtil::g_glBindBuffer(GL_UNIFORM_BUFFER, m_ubo);

        // Note: buffer data size here exceeds the HW 64K limit.  But
        // the rendering loop in ::draw() will never use more than 64K
        // at a time, so that should be ok.
        LwnUtil::g_glBufferData(GL_UNIFORM_BUFFER, sizeof(BenchmarkCpuOverheadLWN::SegAttrs)*N_DRAWS,
                                m_objectAttrs, GL_STATIC_DRAW);
    }
}

void BenchmarkCpuOverheadOGL::draw(const DrawParams* params)
{
    m_numInstancesRendered += N_DRAWS;

    m_mesh->bindGeometryGL(0);

    if (params->flags & DISPLAY_PRESENT_BIT) {
        LwnUtil::g_glClearColor(0, 0, 0, 1);
        LwnUtil::g_glClear(GL_COLOR_BUFFER_BIT);
    }

    if (m_testDescr->uniformMode == BenchmarkCpuOverheadLWN::UNIFORM_INLINE) {
        const int offsetLoc = m_program->uniformLocation("offset");
        const int colorLoc  = m_program->uniformLocation("color");

        for (int i = 0; i < N_DRAWS; i++) {
            LwnUtil::g_glUniform4fv(offsetLoc, 1, (const GLfloat*)&m_objectAttrs[i].offset);
            LwnUtil::g_glUniform4fv(colorLoc,  1, (const GLfloat*)&m_objectAttrs[i].color);
            LwnUtil::g_glDrawElements(GL_TRIANGLES, m_mesh->numTriangles()*3, GL_UNSIGNED_INT, 0);
        }
    } else {
        // TODO for loop to get over the 64K CB limit

        GLuint program = m_program->programHandle();
        GLuint blockIdx = LwnUtil::g_glGetUniformBlockIndex(program, "Objs");
        LwnUtil::g_glUniformBlockBinding(program, blockIdx, 0/*binding_point_index*/);

        assert((N_DRAWS % N_UBO_ELEMS) == 0);

        size_t uboOffs = 0;
        const int N = N_DRAWS / N_UBO_ELEMS;
        for (int i = 0; i < N; i++, uboOffs += sizeof(SegAttrs)*N_UBO_ELEMS) {
            LwnUtil::g_glBindBufferRange(GL_UNIFORM_BUFFER, 0/*binding point idx*/, m_ubo, uboOffs,
                                         N_UBO_ELEMS * sizeof(SegAttrs));

            LwnUtil::g_glDrawElementsInstanced(GL_TRIANGLES, m_mesh->numTriangles()*3,
                                               GL_UNSIGNED_INT, 0, N_UBO_ELEMS);
        }
    }
}

double BenchmarkCpuOverheadOGL::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numInstancesRendered / elapsedTime;
}

void BenchmarkCpuOverheadOGL::deinit(int subtest)
{
    LwnUtil::g_glBindBuffer(GL_UNIFORM_BUFFER, 0);

    delete m_mesh;
    delete m_program;
    delete[] m_objectAttrs;
}

BenchmarkCpuOverheadOGL::~BenchmarkCpuOverheadOGL()
{
}
