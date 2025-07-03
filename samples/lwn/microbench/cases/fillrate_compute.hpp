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

class BenchmarkFillrateComputeLWN : public BenchmarkCaseLWN
{
private:
    LWNprogram*             m_pgm;
    LWNcommandBuffer*       m_cmd;
    LwnUtil::CmdBuf*        m_cmdBuf;
    LwnUtil::Buffer*        m_counters;
    LWNtexture              m_image;
    LWNimageHandle          m_imageHandle;

    double                  m_gpuTime;
    uint64_t                m_numDispatches;

    void setupTexture(LWNcommandBuffer* cmd, int texWidth, int texHeight);

public:
    struct TestDescr
    {
        int texWidth;
        int texHeight;
        int csLocalSizeX;
        int csLocalSizeY;
    };

    static int getNumSubTests();
    static const TestDescr subtests[];

    BenchmarkFillrateComputeLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkFillrateComputeLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);

private:
    const TestDescr* m_testDescr;
};
