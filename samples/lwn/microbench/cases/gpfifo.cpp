/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// A benchmark case that measures the effect of the number of gpfifo
// entries.The test exelwtes batches of dummy GPU commands either directly,
// or by using lwnCommandBufferCallCommands.The latter creates one gpfifo
// entry per call, while the former uses just a few gpfifo entries.

#include "gpfifo.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>

typedef BenchmarkCase::Description Description;
typedef BenchmarkGpfifoLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { 1,    false, false },
    { 50,   false, false },
    { 100,  false, false },
    { 150,  false, false },
    { 200,  false, false },
    { 1,    true,  false },
    { 50,   true,  false },
    { 100,  true,  false },
    { 150,  true,  false },
    { 200,  true,  false },
    { 1,    true,  true  },
    { 50,   true,  true  },
    { 100,  true,  true  },
    { 150,  true,  true  },
    { 200,  true,  true  },
    { 200,  true,  true  },
    { 2000, true,  false },
    { 2000, true,  true  }
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

BenchmarkGpfifoLWN::BenchmarkGpfifoLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h),
    m_res(nullptr),
    m_testDescr(nullptr)
{
}

int BenchmarkGpfifoLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

Description BenchmarkGpfifoLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];

    static char testName[256];
    char subteststr[64];

    if (t.useSubmitCommands) {
        strcpy(subteststr, ".submitCommands=1");
        assert(t.useGpfifoPerWorkItem == 1);
    } else {
        sprintf(subteststr, ".useGpfifoPerWorkItem=%d", (int)t.useGpfifoPerWorkItem);
    }
    sprintf(testName, "gpfifo.numWorkItems=%d%s", t.numWorkItems, subteststr);

    Description d;
    d.name  = testName;
    d.units = "flushes/s";
    return d;
}

static void insertDummyCommands(LWNcommandBuffer* cmd)
{
    lwnCommandBufferSetRenderEnable(cmd, LWN_TRUE);
}

void BenchmarkGpfifoLWN::init(int subtest)
{
    m_subtestIdx = subtest;
    const TestDescr& testDescr = subtests[subtest];
    m_testDescr = &testDescr;

    m_numFlushes = 0;

    m_res = new Resources();

    //-----------------------------------------------------
    m_res->perFrameCmd.reset(new LwnUtil::CompiledCmdBuf(device(), coherentPool(), 2048 * 128, 4096 * 16));

    if (testDescr.useGpfifoPerWorkItem) {
        m_res->workItemCmd.reset(new LwnUtil::CompiledCmdBuf(device(), coherentPool(), 1024, 2048));
        LWNcommandBuffer* dummyCommands = m_res->workItemCmd->cmd();
        lwnCommandBufferBeginRecording(dummyCommands);

        // Dummy commands in order to hit the large command buffer
        // code path (that will create one gpfifo entry per
        // CallCommands)
        insertDummyCommands(dummyCommands);

        LWNcommandHandle workItemHandle = lwnCommandBufferEndRecording(dummyCommands);

        LWNcommandBuffer* cmd = m_res->perFrameCmd->cmd();
        m_res->perFrameCmd->begin();
        m_res->cmdbufs.resize(testDescr.numWorkItems);

        for(int i = 0; i < testDescr.numWorkItems; i++) {
            // CallCommands will create one gpfifo entry per call
            lwnCommandBufferCallCommands(cmd, 1, &workItemHandle);
            m_res->cmdbufs[i] = workItemHandle;
        }
    } else {
        LWNcommandBuffer* cmd = m_res->perFrameCmd->cmd();
        m_res->perFrameCmd->begin();

        for (int i = 0; i < testDescr.numWorkItems; i++) {
            insertDummyCommands(cmd);
        }
    }
    m_res->perFrameCmd->end();
}

void BenchmarkGpfifoLWN::draw(const DrawParams* params)
{
    m_numFlushes++;

    if (m_testDescr->useSubmitCommands) {
        lwnQueueSubmitCommands(queue(), (int) m_res->cmdbufs.size(), &m_res->cmdbufs[0]);
    } else {
        m_res->perFrameCmd->submit(queue());
    }
    lwnQueueFlush(queue());
}

double BenchmarkGpfifoLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numFlushes / elapsedTime;
}

void BenchmarkGpfifoLWN::deinit(int subtest)
{
    delete m_res;
}

BenchmarkGpfifoLWN::~BenchmarkGpfifoLWN()
{
}
