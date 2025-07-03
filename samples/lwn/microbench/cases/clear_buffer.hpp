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

class BenchmarkClearBufferLWN : public BenchmarkCaseLWN
{
public:
    struct TestDescr
    {
        int length;
    };

    BenchmarkClearBufferLWN(LWNdevice *dev, LWNqueue *q, LwnUtil::Pools *pools, int w, int h);
    ~BenchmarkClearBufferLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);

private:
    struct Resources
    {
        Resources() :
            commands(nullptr),
            buffer(nullptr)
        {
        }

        ~Resources() {
        }

        std::unique_ptr<LwnUtil::CompiledCmdBuf> commands;
        std::unique_ptr<LwnUtil::Buffer>         buffer;
    };


    Resources       *m_res;
    uint64_t         m_numDwordsCleared;
    const TestDescr *m_testDescr;
};
