/*
 * Copyright (c) 2016-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Measure time taken by lwnTextureInitialize.
// Allocate one memory pool per texture in init, and lwnTextureInitialize
// each texture in the benchmark loop.
// There's an option for leaking textures and memory pools which can be
// used for checking whether the number of allocated textures/pools has
// any effect on lwnTextureInitialize timing.

#include "tex_init.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkCase::Description Description;
typedef BenchmarkTexInitLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { 100, false },
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

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
    "  gl_Position = vec4(position*0.01, 1.0);\n" \
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
    "in IO { vec2 fTexCoord; };\n"
    "void main() {\n"
    "  color = vec4(fTexCoord.x, fTexCoord.y, 0, 1);\n"
    "}\n";

BenchmarkTexInitLWN::BenchmarkTexInitLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h),
    m_res(nullptr)
{
}

int BenchmarkTexInitLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

Description BenchmarkTexInitLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];

    static char testName[256];

    sprintf(testName, "tex_init.numTextures=%d.leak=%d", t.numTextures, t.leak?1:0);

    Description d;
    d.name  = testName;
    d.units = "lwnTextureInitializes/s";
    return d;
}

void BenchmarkTexInitLWN::init(int subtest)
{
    m_subtestIdx = subtest;
    const TestDescr& testDescr = subtests[subtest];

    m_numTexInits = 0;

    m_res = new Resources();
    m_res->cmdBuf.reset(new LwnUtil::CmdBuf(device(), queue(), coherentPool(), 8, 64*1024, 64*1024));

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

    LWNcommandBuffer* cmd = m_res->cmdBuf->cmd();
    lwnCommandBufferBeginRecording(cmd);

    float clearColor[] = { 0, 1, 0, 1 };
    lwnCommandBufferClearColor(cmd, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    lwnCommandBufferClearDepthStencil(cmd, 1.0, LWN_TRUE, 0, 0);

    LWNcommandHandle cmdHandle = lwnCommandBufferEndRecording(cmd);
    m_res->cmdBuf->submit(1, &cmdHandle);

    /////////////////////////////////////////////
    m_res->mesh.reset(LwnUtil::Mesh::createFullscreenTriangle(device(), coherentPool(), 0.75f));
    m_res->perFrameCmd.reset(new LwnUtil::CompiledCmdBuf(device(), coherentPool(), 1024, 2048));

    m_res->perFrameCmd->begin();
    m_vertex->bind(m_res->perFrameCmd->cmd());

    lwnCommandBufferBindVertexBuffer(m_res->perFrameCmd->cmd(), 0,
                                     m_res->mesh->vboAddress(),
                                     m_res->mesh->numVertices()*sizeof(Vec3f));

    lwnCommandBufferBindVertexBuffer(m_res->perFrameCmd->cmd(), 1,
                                     m_res->mesh->texVboAddress(),
                                     m_res->mesh->numVertices()*sizeof(Vec2f));

    lwnCommandBufferBindProgram(m_res->perFrameCmd->cmd(), &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
    lwnCommandBufferDrawElements(m_res->perFrameCmd->cmd(),
                                 LWN_DRAW_PRIMITIVE_TRIANGLES, LWN_INDEX_TYPE_UNSIGNED_INT,
                                 m_res->mesh->numTriangles()*3, m_res->mesh->iboAddress());

    m_res->perFrameCmd->end();

    // Create a collection of dummy pools backed by a single chunk of memory.
    // See bug 1713805.
    m_res->poolMemory.reset(new uint8_t[LwnUtil::alignSize(testDescr.numTextures * DUMMY_POOL_SIZE,
                                                           LWN_MEMORY_POOL_STORAGE_ALIGNMENT)]);
    LWNmemoryPoolBuilder poolBuilder;
    lwnMemoryPoolBuilderSetDevice(&poolBuilder, device());
    lwnMemoryPoolBuilderSetDefaults(&poolBuilder);
    m_res->dummyPools.resize(testDescr.numTextures);
    for (int i = 0; i < testDescr.numTextures; i++) {
        uint8_t *aligned = (LwnUtil::align(m_res->poolMemory.get(), LWN_MEMORY_POOL_STORAGE_ALIGNMENT) +
                            i * DUMMY_POOL_SIZE);
        lwnMemoryPoolBuilderSetStorage(&poolBuilder, aligned, DUMMY_POOL_SIZE);
        lwnMemoryPoolInitialize(&m_res->dummyPools[i], &poolBuilder);
    }

    // Generate a 2d texture
    lwnTextureBuilderSetDevice(&m_textureBuilder, device());
    lwnTextureBuilderSetDefaults(&m_textureBuilder);
    lwnTextureBuilderSetTarget(&m_textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(&m_textureBuilder, LWN_FORMAT_RGBA8);
    lwnTextureBuilderSetSize2D(&m_textureBuilder, 256, 256);

/*
    uint32_t texSize = lwnTextureBuilderGetStorageSize(&m_textureBuilder);
    uint32_t texAlign = lwnTextureBuilderGetStorageAlignment(&m_textureBuilder);
    uintptr_t poolOffset = gpuPool()->allocTexture(texSize, texAlign);

    lwnTextureBuilderSetStorage(&m_textureBuilder, gpuPool()->pool(), poolOffset);
*/
}

void BenchmarkTexInitLWN::draw(const DrawParams* params)
{
    std::vector<LWNmemoryPool>::iterator it = m_res->dummyPools.begin();
    for (;it != m_res->dummyPools.end(); ++it) {
        LWNmemoryPool* p = &(*it);
        LWNtexture t;
        lwnTextureBuilderSetStorage(&m_textureBuilder, p, 0);
        lwnTextureInitialize(&t, &m_textureBuilder);
        lwnTextureFinalize(&t);
        m_numTexInits++;
    }
}

double BenchmarkTexInitLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numTexInits / elapsedTime;
}

void BenchmarkTexInitLWN::deinit(int subtest)
{
    const TestDescr& testDescr = subtests[subtest];
    if (!testDescr.leak)
        delete m_res;

    lwnProgramFinalize(&m_pgm);
}

BenchmarkTexInitLWN::~BenchmarkTexInitLWN()
{
}
