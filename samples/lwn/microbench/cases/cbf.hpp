/*
 * Copyright (c) 2016
 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include "bench.hpp"

class BenchmarkCBFLWN : public BenchmarkCaseLWN
{
private:
    LWNprogram*                m_pgm;
    LWNcommandBuffer*          m_cmd;
    LwnUtil::CmdBuf*           m_cmdBuf;
    LwnUtil::VertexState*      m_vertex;
    LwnUtil::Mesh*             m_mesh;
    LwnUtil::Buffer*           m_vertexAttributes;
    LwnUtil::Buffer*           m_counters;

public:
    static const int Y_SEGMENTS = 8; // draw the grid this many times
    static const int GRIDX = 1280;
    static const int GRIDY =  720 / Y_SEGMENTS;
    static const int numVtx = GRIDX * GRIDY;
    static const int numTris = (GRIDX-1)*(GRIDY-1)*2;

    BenchmarkCBFLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkCBFLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
