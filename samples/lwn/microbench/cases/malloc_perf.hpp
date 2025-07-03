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

class BenchmarkMallocPerfLWN : public BenchmarkCaseLWN
{
public:
    struct TestDescr
    {
        int numInitMallocs;
    };

private:
    int                        m_mallocSizeIndex;

    std::vector<void*>         m_initMallocs;
    std::vector<void*>         m_runtimeMallocs;

    int                        m_subtestIdx;
    uint64_t                   m_numMallocs;

public:
    BenchmarkMallocPerfLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkMallocPerfLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* extras);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
