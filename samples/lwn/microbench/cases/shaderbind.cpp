/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Shader buffer bind benchmark.

#include "shaderbind.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>


typedef BenchmarkShaderBindLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { 7, 1, 1 },
    { 7, 4, 1 },
    { 7, 1, 4 },
    { 7, 4, 4 }
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

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  color = vtxcol;\n"
    "}\n";

static char* vertexShaderVariant(Vec4f color)
{
    char* shd = new char[1024];
    static const char* a =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(binding = 0) uniform segmentBlock {\n"
    "  vec4 offset;\n"
    "  vec4 color;\n"
    "};\n"
    "layout(location = 0) in vec3 position;\n"
    "out IO { vec4 vtxcol; };\n"
    "void main() {\n"
    "  gl_Position = vec4(position, 1.0) + offset;\n";

    static const char* b =
    "}\n";

    sprintf(shd, "%s  vtxcol = vec4(%f, %f, %f, %f);\n%s", a, color.x, color.y, color.z, color.w, b);
    return shd;
}

BenchmarkShaderBindLWN::BenchmarkShaderBindLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkShaderBindLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkShaderBindLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    snprintf(testName, sizeof(testName),
             "shader_bind.numSegments=%d.numVtx=%d.numFrg=%d",
             t.numSegments, t.numVtxProgs, t.numFrgProgs);

    Description d;
    d.name  = testName;
    d.units = "draw/s";
    return d;
}

// Generate N different vtx/frg shader variants
void BenchmarkShaderBindLWN::generateShaders(const TestDescr& testDescr)
{
    // Random constant colors used for vertex shaders
    static const Vec4f vtxCols[8] = {
        Vec4f(1.3f, 0.0f, 0.1f, 1.f),
        Vec4f(0.0f, 1.2f, 0.0f, 1.f),
        Vec4f(0.3f, 1.1f, 0.7f, 1.f),
        Vec4f(0.3f, 0.5f, 0.9f, 1.f),
        Vec4f(0.6f, 1.3f, 0.1f, 1.f),
        Vec4f(0.9f, 0.1f, 1.0f, 1.f),
        Vec4f(0.1f, 0.5f, 0.4f, 1.f),
        Vec4f(0.2f, 0.1f, 0.7f, 1.f),
    };

    int numVtxProgs = testDescr.numVtxProgs;
    int numFrgProgs = testDescr.numFrgProgs;

    m_numPrograms =  numVtxProgs * numFrgProgs;
    m_programs = new LWNprogram*[m_numPrograms];

    int shaderIdx = 0;

    for (int frgIdx = 0; frgIdx < numFrgProgs; frgIdx++) {
        for (int vtxIdx = 0; vtxIdx < numVtxProgs; vtxIdx++, shaderIdx++) {
            assert((unsigned int)vtxIdx < sizeof(vtxCols)/sizeof(vtxCols[0]));

            m_programs[shaderIdx] = new LWNprogram;
            lwnProgramInitialize(m_programs[shaderIdx], device());

            Vec4f color = vtxCols[vtxIdx];
            char* vtxShader = vertexShaderVariant(color);

            LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
            const char *sources[2]   = { vtxShader, FS_STRING };

            if (!LwnUtil::compileAndSetShaders(m_programs[shaderIdx], stages, 2, sources))
            {
                assert(0);
            }
            delete vtxShader;
        }
    }
}

LWNcommandHandle BenchmarkShaderBindLWN::renderCommands(LWNcommandBuffer* cmdBuffer)
{
    lwnCommandBufferBeginRecording(cmdBuffer);
    lwnCommandBufferBindVertexBuffer(cmdBuffer, 0, m_mesh->vboAddress(), m_mesh->numVertices()*sizeof(Vec3f));

    LWNbufferAddress cbAddress      = m_cb->address();
    LWNbufferAddress iboAddress     = m_mesh->iboAddress();
    const int       numPrimitives = m_mesh->numTriangles()*3;

    LWNprogram* prevProgram = NULL;

    m_vertex->bind(cmdBuffer);

    for (int drawIdx = 0, shaderIdx = 0; drawIdx < N_DRAWS; drawIdx++) {
        LWNprogram* newProgram = m_programs[shaderIdx];

        if (newProgram != prevProgram) {
            lwnCommandBufferBindProgram(cmdBuffer, newProgram,
                                        LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
            prevProgram = newProgram;
        }


        lwnCommandBufferBindUniformBuffer(cmdBuffer, LWN_SHADER_STAGE_VERTEX,
                                          0,
                                          cbAddress + m_cb->offset(drawIdx),
                                          sizeof(SegAttrs));

        lwnCommandBufferDrawElements(cmdBuffer, LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     numPrimitives, iboAddress);

        shaderIdx++;
        if (shaderIdx >= m_numPrograms)
            shaderIdx = 0;
    }

    return lwnCommandBufferEndRecording(cmdBuffer);
}

void BenchmarkShaderBindLWN::init(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];

    m_numInstancesRendered = 0;

    m_cmdBuf = new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 1, 4*1024*1024, 1024);
    LWNcommandBuffer* cmd = m_cmdBuf->cmd();

    lwnCommandBufferBeginRecording(cmd);

    generateShaders(testDescr);

    m_vertex = new LwnUtil::VertexState;
    m_vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_vertex->setStream(0, 12);

    int numSegments = testDescr.numSegments;
    m_mesh = LwnUtil::Mesh::createCircle(device(), coherentPool(), numSegments, Vec2f(1.f/(float)OBJS_X, 1.f/(float)OBJS_Y), 1.f);

    float clearColor[] = { 0.1f, 0.1f, 0.1f, 0.1f };
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

void BenchmarkShaderBindLWN::draw(const DrawParams* params)
{
    m_numInstancesRendered += N_DRAWS;

    m_cmdBuf->submit(1, &m_prebuiltCmdHandle);
}

double BenchmarkShaderBindLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numInstancesRendered / elapsedTime;
}

void BenchmarkShaderBindLWN::deinit(int subtest)
{
    delete m_vertex;
    delete m_mesh;
    delete m_cb;

    for (int i = 0; i < m_numPrograms; i++) {
        lwnProgramFinalize(m_programs[i]);
        delete m_programs[i];
    }
    delete[] m_programs;

    delete m_cmdBuf;
}

BenchmarkShaderBindLWN::~BenchmarkShaderBindLWN()
{
}
