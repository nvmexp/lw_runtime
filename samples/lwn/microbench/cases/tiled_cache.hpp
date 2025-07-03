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

class BenchmarkTiledCacheLWN : public BenchmarkCaseLWN
{
public:
    struct TestDescr
    {
        bool tiledCache;
        bool msaa;
        bool autoMode;
    };

    BenchmarkTiledCacheLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkTiledCacheLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);

private:
    struct SegAttrs
    {
        LwnUtil::Vec4f offset;
        LwnUtil::Vec4f color;
    };

    struct Resources
    {
        Resources() :
            vertex(nullptr),
            mesh(nullptr),
            cmdBuf(nullptr),
            cb(nullptr)
        {
        }

        ~Resources() {
        }

        std::unique_ptr<LwnUtil::VertexState>      vertex;
        std::unique_ptr<LwnUtil::Mesh>             mesh;
        std::unique_ptr<LwnUtil::CmdBuf>           cmdBuf;
        std::unique_ptr<LwnUtil::UboArr<SegAttrs>> cb;
        std::unique_ptr<LwnUtil::RenderTarget>     rt;
    };


    LWNprogram       m_pgm;
    LWNcommandHandle m_prebuiltCmdHandle;
    Resources*       m_res;
    uint64_t         m_numInstancesRendered;

    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdBuf);

    const TestDescr* m_testDescr;
};
