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

class BenchmarkTrirateLWN : public BenchmarkCaseLWN
{
private:
    struct SegAttrs
    {
        LwnUtil::Vec4f offset;
        LwnUtil::Vec4f color;
    };

    LWNprogram*                m_pgm;
    LwnUtil::VertexState*      m_vertex;
    LWNcommandBuffer*          m_cmd;
    LwnUtil::CmdBuf*           m_cmdBuf;
    LwnUtil::Mesh*             m_mesh;
    LwnUtil::UboArr<SegAttrs>* m_cb;

    uint64_t m_numTrisRendered;

public:
    static const int Y_SEGMENTS = 4; // draw the grid this many times
    static const int GRIDX = 1920;
    static const int GRIDY = 1200 / Y_SEGMENTS;
    static const int numVtx = GRIDX * GRIDY;
    static const int numTris = (GRIDX-1)*(GRIDY-1)*2;

    BenchmarkTrirateLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkTrirateLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
