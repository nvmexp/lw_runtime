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

class BenchmarkGpfifoLWN : public BenchmarkCaseLWN
{
public:
    struct TestDescr
    {
        int  numWorkItems;
        bool useGpfifoPerWorkItem;
        bool useSubmitCommands;
    };

private:
    struct Resources
    {
        Resources() :
            perFrameCmd(nullptr),
            workItemCmd(nullptr) {
        }

        ~Resources() {
        }

        std::unique_ptr<LwnUtil::CompiledCmdBuf> perFrameCmd;
        std::unique_ptr<LwnUtil::CompiledCmdBuf> workItemCmd;
        std::vector<LWNcommandHandle>            cmdbufs;
    };

    Resources*       m_res;
    int              m_subtestIdx;
    uint64_t         m_numFlushes;
    const TestDescr* m_testDescr;

public:
    BenchmarkGpfifoLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkGpfifoLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
