/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// ClearBuffer test.
//
// CopyEngine clears are expected to run at about 80% of 16B/clock.

#include "clear_buffer.hpp"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static const BenchmarkClearBufferLWN::TestDescr subtests[] = {
    { 1024    },
    { 8192    },
    { 65536   },
    { 65536*4 },
};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

const int N_DRAWS = 10;

BenchmarkClearBufferLWN::BenchmarkClearBufferLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkClearBufferLWN::numSubtests() const
{
    return sizeof(subtests) / sizeof(subtests[0]);
}

BenchmarkCase::Description BenchmarkClearBufferLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];
    static char testName[256];

    sprintf(testName, "clear_buffer.numDwords=%d", t.length);

    Description d;
    d.name  = testName;
    d.units = "dwords/s";
    return d;
}

void BenchmarkClearBufferLWN::init(int subtest)
{
    m_res = new Resources();

    m_testDescr = &subtests[subtest];
    m_numDwordsCleared = 0;

    m_res->buffer.reset(new LwnUtil::Buffer(device(), coherentPool(), nullptr, m_testDescr->length*4, BUFFER_ALIGN_COPY_WRITE_BIT));
    m_res->commands.reset(new LwnUtil::CompiledCmdBuf(device(), coherentPool(), 128*N_DRAWS, 256));
    m_res->commands->begin();
    for (int i = 0; i < N_DRAWS; i++) {
        lwnCommandBufferClearBuffer(m_res->commands->cmd(), m_res->buffer->address(), m_testDescr->length*4, 0xc001ca75);
    }
    m_res->commands->end();
}

void BenchmarkClearBufferLWN::draw(const DrawParams *drawParams)
{
    LWNcommandHandle cmdHandle = m_res->commands->handle();
    lwnQueueSubmitCommands(queue(), 1, &cmdHandle);
    m_numDwordsCleared += N_DRAWS * m_testDescr->length;
}

double BenchmarkClearBufferLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numDwordsCleared / elapsedTime;
}

void BenchmarkClearBufferLWN::deinit(int subtest)
{
    delete m_res;
}

BenchmarkClearBufferLWN::~BenchmarkClearBufferLWN()
{
}
