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

class BenchmarkGpuTimeLWN : public BenchmarkCaseLWN
{
public:
    struct TestDescr
    {
        int dummy;
    };

    BenchmarkGpuTimeLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkGpuTimeLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);

    uint32_t testProps() const;

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
            counters(nullptr)
        {
        }

        ~Resources() {
        }

        std::unique_ptr<LwnUtil::VertexState>      vertex;
        std::unique_ptr<LwnUtil::Mesh>             mesh;
        std::unique_ptr<LwnUtil::CmdBuf>           cmdBuf;
        std::unique_ptr<LwnUtil::RenderTarget>     rt;
        std::unique_ptr<LwnUtil::Buffer>           counters;
    };


    LWNprogram       m_pgm;
    Resources*       m_res;
    uint64_t         m_numInstancesRendered;
    float            m_minGpuTime;
    LWNsync          m_textureAvailableSync;

    const TestDescr* m_testDescr;

    void renderAndFlip(const DrawParamsLWN* params, int* renderTargetIdx, int numTriangles);;
    void timedRender(const BenchmarkCaseLWN::DrawParamsLWN *params, int *renderTargetIdx, int numTriangles, float *cpuTime, float *gpuTime);

};
