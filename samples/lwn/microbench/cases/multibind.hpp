/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include "bench.hpp"

#define TEXTURE_BINDINGS_PER_STAGE 32

class BenchmarkMultiBindLWN : public BenchmarkCaseLWN
{
public:
    struct TestDescr
    {
        bool useMultiBind;
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
    LWNtexture                 m_textures[TEXTURE_BINDINGS_PER_STAGE];
    LWNsampler                 m_sampler;
    LWNtextureHandle           m_textureHandles[TEXTURE_BINDINGS_PER_STAGE];

    int                        m_subtestIdx;
    uint64_t                   m_numBinds;

    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdBuf);
    void setupTextures(LWNcommandBuffer* cmd);

public:
    BenchmarkMultiBindLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkMultiBindLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
