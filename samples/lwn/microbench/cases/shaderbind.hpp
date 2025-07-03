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

class BenchmarkShaderBindLWN : public BenchmarkCaseLWN
{
public:
    struct TestDescr
    {
        int numSegments;
        int numVtxProgs;
        int numFrgProgs;
    };

private:
    struct SegAttrs
    {
        LwnUtil::Vec4f offset;
        LwnUtil::Vec4f color;
    };

    int                        m_numPrograms;
    LWNprogram**               m_programs;
    LwnUtil::VertexState*      m_vertex;
    LwnUtil::CmdBuf*           m_cmdBuf;
    LWNcommandHandle           m_prebuiltCmdHandle;
    LwnUtil::Mesh*             m_mesh;
    LwnUtil::UboArr<SegAttrs>* m_cb;

    uint64_t m_numInstancesRendered;

    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdbuf);
    void generateShaders(const TestDescr& testDescr);

public:
    BenchmarkShaderBindLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkShaderBindLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
