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
#include "glprogram.hpp"

class BenchmarkTexInitLWN : public BenchmarkCaseLWN
{
public:
    static const int DUMMY_POOL_SIZE = 5 * 65536;

    struct TestDescr
    {
        int numTextures;
        bool leak;
    };

private:
    struct Resources
    {
        Resources() : cmdBuf(nullptr), perFrameCmd(nullptr), mesh(nullptr) { }
        ~Resources()
        {
            for (std::vector<LWNmemoryPool>::iterator it = dummyPools.begin();
                 it != dummyPools.end(); ++it) {
                lwnMemoryPoolFinalize(&(*it));
            }
        }

        std::unique_ptr<LwnUtil::CmdBuf>         cmdBuf;
        std::unique_ptr<LwnUtil::CompiledCmdBuf> perFrameCmd;
        std::unique_ptr<LwnUtil::Mesh>           mesh;
        std::vector<LWNmemoryPool>               dummyPools;
        std::unique_ptr<uint8_t[]>               poolMemory;
    };

    Resources*                 m_res;
    LWNprogram                 m_pgm;
    LwnUtil::VertexState*      m_vertex;
    LWNtextureBuilder          m_textureBuilder;

    int                        m_subtestIdx;
    uint64_t                   m_numTexInits;

    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdBuf);
    void setupTexture(LWNcommandBuffer* cmd, int texWidth, int texHeight);

public:
    BenchmarkTexInitLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkTexInitLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
