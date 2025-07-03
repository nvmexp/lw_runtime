/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// BenchmarkShaderPerfLWN is an "umbrella test" that contains a number of
// separate shader perf tests.  The shader perf tests are implemented by
// deriving from the ShaderTest class.  The source code for these tests can be
// found under the microbench/cases/shaderperf/ directory.
//
// The BenchmarkShaderPerfLWN will initialize things like command buffer,
// appropriate render state, and perform drawing of a fullscreen quad and
// measuring the time it takes to render them.  The subtest's responsibility
// is to setup a shader and any inputs required for using it as a pixel
// shader.  This may be extended later for vertex shaders.
//
// Adding a new test:
//
// See class implementation in cases/shaderperf/dce.cpp.  To add a new test,
// you also need to add your subtests in cases/shadeperf/tests.hpp.  This
// pattern is borrowed from lwntest/lwogtest.

#include "shaderperf.hpp"
#include <assert.h>
#include <string.h>
#include <string> // Use C++ strings for replace
#include <stdio.h>

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::RenderTarget;

struct Subtest
{
    const char*          name;
    CreateShaderTestFunc create;
};

static Subtest s_subtests[] = {
#define DECLARE_SHADERPERF_TEST(name) { #name, createShaderTest_##name },
#include "cases/shaderperf/tests.hpp"
};

ShaderTest::ShaderTest(Context *context) :
    m_context(context),
    texPbo(nullptr)
{
}

void ShaderTest::setupSampler2DTexture(int texWidth, int texHeight, int samplerIndex)
{
    //////////////////////////////////////////
    // Generate a 2d texture
    // copied from tex.cpp
    m_texWidth = texWidth;
    m_texHeight = texHeight;
    assert(m_texWidth > 0);
    assert(m_texHeight > 0);
    assert(samplerIndex >= 0);

    LWNcommandBuffer *cmd = cmdBuf()->cmd();

    LWNtextureBuilder textureBuilder;
    lwnTextureBuilderSetDevice(&textureBuilder, device());
    lwnTextureBuilderSetDefaults(&textureBuilder);
    lwnTextureBuilderSetTarget(&textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(&textureBuilder, LWN_FORMAT_RGBA8);
    lwnTextureBuilderSetSize2D(&textureBuilder, m_texWidth, m_texHeight);

    uintptr_t texSize = lwnTextureBuilderGetStorageSize(&textureBuilder);
    uintptr_t texAlign = lwnTextureBuilderGetStorageAlignment(&textureBuilder);
    uintptr_t poolOffset = gpuPool()->alloc(texSize, texAlign);

    lwnTextureBuilderSetStorage(&textureBuilder, gpuPool()->pool(), poolOffset);

    lwnTextureInitialize(&m_texture, &textureBuilder);
    uint32_t textureID = descriptorPool()->allocTextureID();
    descriptorPool()->registerTexture(textureID, &m_texture);

    texPbo.reset(new LwnUtil::Buffer(device(), coherentPool(), NULL, m_texWidth * m_texHeight * 4,
        BUFFER_ALIGN_COPY_READ_BIT));

    uint32_t* texels = (uint32_t*)texPbo->ptr();
    for (int y = 0; y < m_texHeight; y++) {
        for (int x = 0; x < m_texWidth; x++) {
            uint32_t c = (x >> 1) | ((y >> 1) << 8) | (((x^y) & 1) << (16 + 7));
            texels[x + y*m_texWidth] = c;
        }
    }

    // Download the texture data
    LWNcopyRegion copyRegion = { 0, 0, 0, m_texWidth, m_texHeight, 1 };
    lwnCommandBufferCopyBufferToTexture(cmd, texPbo->address(), &m_texture, NULL, &copyRegion, LWN_COPY_FLAGS_NONE);

    // Sampler + texture handle business
    LWNsamplerBuilder sb;
    lwnSamplerBuilderSetDevice(&sb, device());
    lwnSamplerBuilderSetDefaults(&sb);

    lwnSamplerInitialize(&m_sampler, &sb);

    uint32_t samplerID = descriptorPool()->allocSamplerID();
    descriptorPool()->registerSampler(samplerID, &m_sampler);
    m_textureHandle = lwnDeviceGetTextureHandle(device(), textureID, samplerID);

    lwnCommandBufferBindTexture(cmd, LWN_SHADER_STAGE_FRAGMENT, samplerIndex, m_textureHandle);
}

ShaderTest::~ShaderTest()
{
    if (texPbo != nullptr) {
        LwnUtil::Buffer *tPbo = texPbo.release();
        assert(texPbo == nullptr);
        delete tPbo;

        lwnSamplerFinalize(&m_sampler);
        lwnTextureFinalize(&m_texture);
    }
}

BenchmarkShaderPerfLWN::BenchmarkShaderPerfLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkShaderPerfLWN::numSubtests() const
{
    return sizeof(s_subtests)/sizeof(s_subtests[0]);
}

BenchmarkCase::Description BenchmarkShaderPerfLWN::description(int subtest) const
{
    Description d;
    d.name = s_subtests[subtest].name;
    d.units = "pix/s";
    return d;
}

void BenchmarkShaderPerfLWN::init(int subtest)
{
    m_numPixRendered = 0;
    m_context = new ShaderTest::Context(device(), coherentPool(), gpuPool(), descriptorPool(),
        new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, 64 * 1024, 64 * 1024),
        LwnUtil::Mesh::createFullscreenTriangle(device(), coherentPool(), 0.75f));

    LWNcommandBuffer *cmd = m_context->cmdBuf->cmd();

    lwnCommandBufferBeginRecording(cmd);
    const float clearColor[] = { 0, 0.1f, 0.1f, 1 };

    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0f, LWN_TRUE, 0, 0);

    m_subtest = s_subtests[subtest].create(m_context);

    uint32_t writeMask =  RenderTarget::DEST_WRITE_DEPTH_BIT | RenderTarget::DEST_WRITE_COLOR_BIT;
    LwnUtil::RenderTarget::setColorDepthMode(cmd, writeMask, false);

    LwnUtil::VertexState vertex;
    vertex.setAttribute(0, LWN_FORMAT_RGB32F, 0, 0);
    vertex.setStream(0, 12);
    vertex.setAttribute(1, LWN_FORMAT_RG32F, 0, 1);
    vertex.setStream(1, 8);
    vertex.bind(cmd);

    const LwnUtil::Mesh *mesh = m_context->mesh.get();
    lwnCommandBufferBindVertexBuffer(cmd, 0, mesh->vboAddress(), mesh->numVertices()*sizeof(Vec3f));
    lwnCommandBufferBindVertexBuffer(cmd, 1, mesh->texVboAddress(), mesh->numVertices()*sizeof(Vec2f));

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_context->cmdBuf->submit(1, &cmdHandle);
}

void BenchmarkShaderPerfLWN::draw(const DrawParams* drawParams)
{
    LWNcommandBuffer *cmd = m_context->cmdBuf->cmd();

    // TODO this does not need to create command buffers here.  Could create
    // one static command buffer at ::init() time and then just use
    // lwnQueueSubmitCommands here.
    lwnCommandBufferBeginRecording(cmd);
    int w = width() / 4;
    int h = height() / 4;
    lwnCommandBufferSetScissor(cmd, 0, 0, w, h);
    lwnCommandBufferDrawElements(cmd,
                                 LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                 m_context->mesh->numTriangles()*3, m_context->mesh->iboAddress());

    m_numPixRendered += w * h;

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_context->cmdBuf->submit(1, &cmdHandle);
}

double BenchmarkShaderPerfLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numPixRendered / elapsedTime;
}

void BenchmarkShaderPerfLWN::deinit(int subtest)
{
    delete m_subtest;
    delete m_context;
}

BenchmarkShaderPerfLWN::~BenchmarkShaderPerfLWN()
{
}
