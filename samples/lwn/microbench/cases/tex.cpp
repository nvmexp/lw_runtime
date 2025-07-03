/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
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

#include "tex.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkCase::Description Description;
typedef BenchmarkTextureLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { 256,  256 },
    { 512,  512 },
    { 1024, 1024 },
    { 2048, 2048 }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

// How many instances to draw
const int N_DRAWS = 16;

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define PROLOG \
    "#version 440 core\n" \
    "#extension GL_LW_gpu_shader5:require\n"

#define VS() \
    "layout(location = 0) in vec3 position;\n" \
    "layout(location = 1) in vec2 texcoord;\n" \
    "out IO { vec2 fTexCoord; };\n" \
    "void main() {\n" \
    "  gl_Position = vec4(position, 1.0);\n" \
    "  fTexCoord   = texcoord;\n" \
    "}\n";

static const char *VS_STRING_LWN =
    PROLOG
    VS()

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(binding=4) uniform sampler2D tex;\n"
    "in IO { vec2 fTexCoord; };\n"
    "void main() {\n"
    "  color = texture(tex, fTexCoord);\n"
    "}\n";

BenchmarkTextureLWN::BenchmarkTextureLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h),
    m_res(nullptr)
{
}

int BenchmarkTextureLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

static Description testDescription(const TestDescr& t, const char* postfix)
{
    static char testName[256];

    sprintf(testName, "tex.texWidth=%d.texHeight=%d", t.texWidth, t.texHeight);

    Description d;
    d.name  = testName;
    d.units = "pix/s";
    return d;
}

Description BenchmarkTextureLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    return testDescription(t, "");
}

void BenchmarkTextureLWN::setupTexture(LWNcommandBuffer* cmd, int texWidth, int texHeight)
{
    //////////////////////////////////////////
    // Generate a 2d texture
    LWNtextureBuilder textureBuilder;
    lwnTextureBuilderSetDevice(&textureBuilder, device());
    lwnTextureBuilderSetDefaults(&textureBuilder);
    lwnTextureBuilderSetTarget(&textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(&textureBuilder, LWN_FORMAT_RGBA8);
    lwnTextureBuilderSetSize2D(&textureBuilder, texWidth, texHeight);

    uintptr_t texSize = lwnTextureBuilderGetStorageSize(&textureBuilder);
    uintptr_t texAlign = lwnTextureBuilderGetStorageAlignment(&textureBuilder);
    uintptr_t poolOffset = gpuPool()->alloc(texSize, texAlign);

    lwnTextureBuilderSetStorage(&textureBuilder, gpuPool()->pool(), poolOffset);

    lwnTextureInitialize(&m_texture, &textureBuilder);
    uint32_t textureID = descriptorPool()->allocTextureID();
    descriptorPool()->registerTexture(textureID, &m_texture);

    m_res->texPbo.reset(new LwnUtil::Buffer(device(), coherentPool(), NULL, texWidth * texHeight * 4,
                                            BUFFER_ALIGN_COPY_READ_BIT));

    uint32_t* texels = (uint32_t*)m_res->texPbo->ptr();
    for (int y = 0; y < texHeight; y++) {
        for (int x = 0; x < texWidth; x++) {
            uint32_t c = (x>>1) | ((y>>1) << 8) | (((x^y)&1)<<(16+7));
            texels[x + y*texWidth] = c;
        }
    }

    // Download the texture data
    LWNcopyRegion copyRegion = { 0, 0, 0, texWidth, texHeight, 1 };
    lwnCommandBufferCopyBufferToTexture(cmd, m_res->texPbo->address(), &m_texture, NULL, &copyRegion, LWN_COPY_FLAGS_NONE);

    // Sampler + texture handle business
    LWNsamplerBuilder sb;
    lwnSamplerBuilderSetDevice(&sb, device());
    lwnSamplerBuilderSetDefaults(&sb);
    lwnSamplerInitialize(&m_sampler, &sb);

    uint32_t samplerID = descriptorPool()->allocSamplerID();
    descriptorPool()->registerSampler(samplerID, &m_sampler);
    m_textureHandle = lwnDeviceGetTextureHandle(device(), textureID, samplerID);
}

void BenchmarkTextureLWN::init(int subtest)
{
    m_subtestIdx = subtest;
    const TestDescr& testDescr = subtests[subtest];

    m_numPixelsRendered = 0;

    m_res = new Resources();
    m_res->cmdBuf.reset(new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, 64*1024, 64*1024));

    LWNcommandBuffer* cmd = m_res->cmdBuf->cmd();
    lwnCommandBufferBeginRecording(cmd);

    // Create programs from the device, provide them shader code and compile/link them
    lwnProgramInitialize(&m_pgm, device());

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2]   = { VS_STRING_LWN, FS_STRING };
    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(&m_pgm, stages, nSources, sources))
    {
        assert(0);
    }

    m_vertex = new LwnUtil::VertexState;
    m_vertex->setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    m_vertex->setStream(0, 12);
    m_vertex->setAttribute(1, LWN_FORMAT_RG32F, 0, 1);
    m_vertex->setStream(1, 8);

    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0, LWN_TRUE, 0, 0);

    setupTexture(cmd, testDescr.texWidth, testDescr.texHeight);

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_res->cmdBuf->submit(1, &cmdHandle);

    /////////////////////////////////////////////
    m_res->mesh.reset(LwnUtil::Mesh::createFullscreenTriangle(device(), coherentPool(), 0.75f));
    m_res->texDraws.reset(new LwnUtil::CompiledCmdBuf(device(), coherentPool(), N_DRAWS*128, 2048));

    m_res->texDraws->begin();
    m_vertex->bind(m_res->texDraws->cmd());

    lwnCommandBufferBindVertexBuffer(m_res->texDraws->cmd(), 0,
                                     m_res->mesh->vboAddress(),
                                     m_res->mesh->numVertices()*sizeof(Vec3f));

    lwnCommandBufferBindVertexBuffer(m_res->texDraws->cmd(), 1,
                                     m_res->mesh->texVboAddress(),
                                     m_res->mesh->numVertices()*sizeof(Vec2f));

    lwnCommandBufferBindTexture(m_res->texDraws->cmd(), LWN_SHADER_STAGE_FRAGMENT, 4, m_textureHandle);

    lwnCommandBufferBindProgram(m_res->texDraws->cmd(), &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    for (int i = 0; i < N_DRAWS; i++) {
        lwnCommandBufferDrawElements(m_res->texDraws->cmd(),
                                     LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     m_res->mesh->numTriangles()*3, m_res->mesh->iboAddress());
    }
    m_res->texDraws->end();
}

void BenchmarkTextureLWN::draw(const DrawParams* params)
{
    m_numPixelsRendered += N_DRAWS * width()*height();

    m_res->texDraws->submit(queue());
}

double BenchmarkTextureLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numPixelsRendered / elapsedTime;
}

void BenchmarkTextureLWN::deinit(int subtest)
{
    delete m_res;
    delete m_vertex;

    lwnSamplerFinalize(&m_sampler);
    lwnTextureFinalize(&m_texture);
    lwnProgramFinalize(&m_pgm);
}

BenchmarkTextureLWN::~BenchmarkTextureLWN()
{
}

#if 0
//--------------------------------------------------------------------
// OGL version
//--------------------------------------------------------------------

static const TestDescr subtestsOGL[] = {
    { 10, false, BenchmarkTextureLWN::UNIFORM_INLINE },
    { 16, false, BenchmarkTextureLWN::UNIFORM_INLINE },
    { 10, false, BenchmarkTextureLWN::UNIFORM_UBO },
    { 16, false, BenchmarkTextureLWN::UNIFORM_UBO }
};

BenchmarkTextureOGL::BenchmarkTextureOGL(int w, int h) :
    BenchmarkCaseOGL(w, h)
{
}

int BenchmarkTextureOGL::numSubtests() const
{
    return sizeof(subtestsOGL)/sizeof(subtestsOGL[0]);
}

Description BenchmarkTextureOGL::description(int subtest) const
{
    const TestDescr& t = subtestsOGL[subtest];
    return testDescription(t, ".ogl");
}

void BenchmarkTextureOGL::init(int subtest)
{
    const TestDescr& testDescr = subtestsOGL[subtest];

    m_testDescr = &testDescr;
    m_numInstancesRendered = 0;

    m_program = new GlProgram();

    std::string fs(FS_STRING);

    switch (testDescr.uniformMode) {
    case BenchmarkTextureLWN::UNIFORM_INLINE:
        {
            std::string vs(VS_STRING_OGL);
            m_program->shaderSource(GlProgram::VertexShader, vs);
        }
        break;
    case BenchmarkTextureLWN::UNIFORM_UBO:
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

    if (testDescr.uniformMode == BenchmarkTextureLWN::UNIFORM_UBO) {
        glGenBuffers(1, &m_ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, m_ubo);

        // Note: buffer data size here exceeds the HW 64K limit.  But
        // the rendering loop in ::draw() will never use more than 64K
        // at a time, so that should be ok.
        glBufferData(GL_UNIFORM_BUFFER, sizeof(BenchmarkTextureLWN::SegAttrs)*N_DRAWS,
                     m_objectAttrs, GL_STATIC_DRAW);
    }
}

void BenchmarkTextureOGL::draw(const DrawParams* params)
{
    m_numInstancesRendered += N_DRAWS;

    m_mesh->bindGeometryGL(0);

    if (params->flags & DISPLAY_PRESENT_BIT) {
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    if (m_testDescr->uniformMode == BenchmarkTextureLWN::UNIFORM_INLINE) {
        const int offsetLoc = m_program->uniformLocation("offset");
        const int colorLoc  = m_program->uniformLocation("color");

        for (int i = 0; i < N_DRAWS; i++) {
            glUniform4fv(offsetLoc, 1, (const GLfloat*)&m_objectAttrs[i].offset);
            glUniform4fv(colorLoc,  1, (const GLfloat*)&m_objectAttrs[i].color);
            glDrawElements(GL_TRIANGLES, m_mesh->numTriangles()*3, GL_UNSIGNED_INT, 0);
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

double BenchmarkTextureOGL::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numInstancesRendered / elapsedTime;
}

void BenchmarkTextureOGL::deinit(int subtest)
{
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    delete m_mesh;
    delete m_program;
    delete[] m_objectAttrs;
}

BenchmarkTextureOGL::~BenchmarkTextureOGL()
{
}
#endif
