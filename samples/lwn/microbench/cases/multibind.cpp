/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Benchmark BindTexture vs BindTextures. The test draws a grid of
// quads, and for each quad binds 32 textures.

#include "multibind.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkCase::Description Description;
typedef BenchmarkMultiBindLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { false },
    { true }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

const int GRIDX = 32;
const int GRIDY = 32;
const int N_DRAWS = (GRIDX-1) * (GRIDY-1);

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define PROLOG \
    "#version 440 core\n" \
    "#extension GL_LW_gpu_shader5:require\n"

#define VS() \
    "layout(location = 0) in vec3 position;\n" \
    "void main() {\n" \
    "  gl_Position = vec4(position, 1.0);\n" \
    "}\n";

static const char *VS_STRING_LWN =
    PROLOG
    VS()

// Version of the fragment shader using bound textures.
static const char *FS_STRING =
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(binding=0) uniform sampler2D tex[32];\n"
    "void main() {\n"
    "  int t = int(floor(gl_FragCoord.x)) & 31;\n"
    "  color = texture(tex[t], vec2(0));\n"
    "}\n";

BenchmarkMultiBindLWN::BenchmarkMultiBindLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h),
    m_res(nullptr)
{
}

int BenchmarkMultiBindLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

static Description testDescription(const TestDescr& t, const char* postfix)
{
    static char testName[256];

    sprintf(testName, "multibind.useMultiBind=%d", t.useMultiBind);

    Description d;
    d.name  = testName;
    d.units = "binds/s";
    return d;
}

Description BenchmarkMultiBindLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    return testDescription(t, "");
}

void BenchmarkMultiBindLWN::setupTextures(LWNcommandBuffer* cmd)
{
    const int texWidth = 4;
    const int texHeight = 4;

    // Sampler + texture handle business
    LWNsamplerBuilder sb;
    lwnSamplerBuilderSetDevice(&sb, device());
    lwnSamplerBuilderSetDefaults(&sb);
    lwnSamplerBuilderSetWrapMode(&sb, LWN_WRAP_MODE_CLAMP_TO_EDGE, LWN_WRAP_MODE_CLAMP_TO_EDGE, LWN_WRAP_MODE_CLAMP_TO_EDGE);
    lwnSamplerBuilderSetMinMagFilter(&sb, LWN_MIN_FILTER_NEAREST, LWN_MAG_FILTER_NEAREST);
    lwnSamplerInitialize(&m_sampler, &sb);

    uint32_t samplerID = descriptorPool()->allocSamplerID();
    descriptorPool()->registerSampler(samplerID, &m_sampler);

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
    for (int i = 0; i < TEXTURE_BINDINGS_PER_STAGE; ++i) {
        uintptr_t poolOffset = gpuPool()->alloc(texSize, texAlign);

        lwnTextureBuilderSetStorage(&textureBuilder, gpuPool()->pool(), poolOffset);

        lwnTextureInitialize(&m_textures[i], &textureBuilder);
        uint32_t textureID = descriptorPool()->allocTextureID();
        descriptorPool()->registerTexture(textureID, &m_textures[i]);

        m_res->texPbo.reset(new LwnUtil::Buffer(device(), coherentPool(), NULL, texWidth * texHeight * 4,
                                                BUFFER_ALIGN_COPY_READ_BIT));

        uint32_t* texels = (uint32_t*)m_res->texPbo->ptr();
        for (int y = 0; y < texHeight; y++) {
            for (int x = 0; x < texWidth; x++) {
                uint32_t c = i << 3;
                texels[x + y*texWidth] = (c<<24)|(c<<16)|(c<<8)|c;
            }
        }

        // Download the texture data
        LWNcopyRegion copyRegion = { 0, 0, 0, texWidth, texHeight, 1 };
        lwnCommandBufferCopyBufferToTexture(cmd, m_res->texPbo->address(), &m_textures[i], NULL, &copyRegion, LWN_COPY_FLAGS_NONE);

        m_textureHandles[i] = lwnDeviceGetTextureHandle(device(), textureID, samplerID);
    }
}

void BenchmarkMultiBindLWN::init(int subtest)
{
    m_subtestIdx = subtest;
    const TestDescr& testDescr = subtests[subtest];

    m_numBinds = 0;

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

    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0, LWN_TRUE, 0, 0);

    setupTextures(cmd);

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_res->cmdBuf->submit(1, &cmdHandle);

    /////////////////////////////////////////////
    m_res->mesh.reset(LwnUtil::Mesh::createGrid(device(), coherentPool(), GRIDX, GRIDY, Vec2f(-0.5f, -0.5f), Vec2f(2.f, 2.f), 1.f));

    assert(m_res->mesh->numTriangles() / 2 == N_DRAWS);

    m_res->texDraws.reset(new LwnUtil::CompiledCmdBuf(device(), coherentPool(), N_DRAWS*1024, 2048));

    m_res->texDraws->begin();
    m_vertex->bind(m_res->texDraws->cmd());

    lwnCommandBufferBindVertexBuffer(m_res->texDraws->cmd(), 0,
                                     m_res->mesh->vboAddress(),
                                     m_res->mesh->numVertices()*sizeof(Vec3f));

    lwnCommandBufferBindProgram(m_res->texDraws->cmd(), &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    for (int i = 0; i < N_DRAWS; i++) {

        if (testDescr.useMultiBind) {
            lwnCommandBufferBindTextures(m_res->texDraws->cmd(), LWN_SHADER_STAGE_FRAGMENT, 0, TEXTURE_BINDINGS_PER_STAGE, m_textureHandles);
        } else {
            for (int i = 0; i < TEXTURE_BINDINGS_PER_STAGE; ++i) {
                lwnCommandBufferBindTexture(m_res->texDraws->cmd(), LWN_SHADER_STAGE_FRAGMENT, i, m_textureHandles[i]);
            }
        }

        LWNbufferAddress indexAddr = m_res->mesh->iboAddress() + i * 6 * sizeof(uint32_t);
        lwnCommandBufferDrawElements(m_res->texDraws->cmd(),
                                     LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                     6, indexAddr);
    }
    m_res->texDraws->end();
}

void BenchmarkMultiBindLWN::draw(const DrawParams* params)
{
    m_numBinds += N_DRAWS * TEXTURE_BINDINGS_PER_STAGE;

    m_res->texDraws->submit(queue());
}

double BenchmarkMultiBindLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numBinds / elapsedTime;
}

void BenchmarkMultiBindLWN::deinit(int subtest)
{
    delete m_res;

    lwnSamplerFinalize(&m_sampler);
    for (int i = 0; i < TEXTURE_BINDINGS_PER_STAGE; ++i) {
        lwnTextureFinalize(&m_textures[i]);
    }
    lwnProgramFinalize(&m_pgm);
}

BenchmarkMultiBindLWN::~BenchmarkMultiBindLWN()
{
}
