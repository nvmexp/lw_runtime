/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include "bench.hpp"
#include "glprogram.hpp"

class BenchmarkTextureLWN : public BenchmarkCaseLWN
{
public:
    enum UniformMode
    {
        UNIFORM_INLINE,
        UNIFORM_UBO
    };

    struct TestDescr
    {
        int texWidth;
        int texHeight;
    };

private:
    struct Resources
    {
        Resources() : cmdBuf(nullptr), texDraws(nullptr), mesh(nullptr), texPbo(nullptr) { }

        std::unique_ptr<LwnUtil::CmdBuf>         cmdBuf;
        std::unique_ptr<LwnUtil::CompiledCmdBuf> texDraws;
        std::unique_ptr<LwnUtil::Mesh>           mesh;
        std::unique_ptr<LwnUtil::Buffer>         texPbo;
    };

    Resources*                 m_res;
    LWNprogram                 m_pgm;
    LwnUtil::VertexState*      m_vertex;
    LWNtexture                 m_texture;
    LWNsampler                 m_sampler;
    LWNtextureHandle           m_textureHandle;

    int                        m_subtestIdx;
    uint64_t                   m_numPixelsRendered;

    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdBuf);
    void setupTexture(LWNcommandBuffer* cmd, int texWidth, int texHeight);

public:
    BenchmarkTextureLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkTextureLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};

class BenchmarkTextureOGL : public BenchmarkCaseOGL
{
private:
    LwnUtil::OGLMesh* m_mesh;
    GlProgram*        m_program;
    GLuint            m_ubo;

    uint64_t m_numPixelsRendered;

    const BenchmarkTextureLWN::TestDescr* m_testDescr;
public:
    BenchmarkTextureOGL(int w, int h);
    ~BenchmarkTextureOGL();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams *params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
