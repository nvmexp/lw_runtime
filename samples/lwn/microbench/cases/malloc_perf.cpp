/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// Benchmark malloc+free: allocate numInitMallocs memory blocks during
// benchmark init, and measure whether the number of initial allocs
// has any effect on the time taken by runtime mallocs/frees.

#include "malloc_perf.hpp"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>

typedef BenchmarkCase::Description Description;
typedef BenchmarkMallocPerfLWN::TestDescr TestDescr;

static const TestDescr subtests[] = {
    { 0 },
};

static const size_t mallocSizes[] = {16,24,32,64};

using LwnUtil::Vec2f;
using LwnUtil::Vec3f;
using LwnUtil::Vec3i;
using LwnUtil::Vec4f;
using LwnUtil::RenderTarget;

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

BenchmarkMallocPerfLWN::BenchmarkMallocPerfLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h) :
    BenchmarkCaseLWN(dev, q, pools, w, h)
{
}

int BenchmarkMallocPerfLWN::numSubtests() const
{
    return sizeof(subtests)/sizeof(subtests[0]);
}

Description BenchmarkMallocPerfLWN::description(int subtest) const
{
    const TestDescr& t = subtests[subtest];

    static char testName[256];

    sprintf(testName, "malloc_perf.numInitMallocs=%d", t.numInitMallocs);

    Description d;
    d.name  = testName;
    d.units = "mallocs/s";
    return d;
}

void BenchmarkMallocPerfLWN::init(int subtest)
{
    m_subtestIdx = subtest;
    const TestDescr& testDescr = subtests[subtest];

    m_numMallocs = 0;

    m_mallocSizeIndex = 0;

    m_initMallocs.resize(testDescr.numInitMallocs);
    for (std::vector<void*>::iterator it = m_initMallocs.begin(); it != m_initMallocs.end(); ++it) {
        *it = malloc(mallocSizes[m_mallocSizeIndex++ % (sizeof(mallocSizes)/sizeof(int))]);
    }

    m_runtimeMallocs.resize(1000);
}

void BenchmarkMallocPerfLWN::draw(const DrawParams* params)
{
    for (std::vector<void*>::iterator it = m_runtimeMallocs.begin(); it != m_runtimeMallocs.end(); ++it) {
        *it = malloc(mallocSizes[m_mallocSizeIndex++ % (sizeof(mallocSizes)/sizeof(int))]);
        m_numMallocs++;
    }
    for (std::vector<void*>::iterator it = m_runtimeMallocs.begin(); it != m_runtimeMallocs.end(); ++it) {
        free(*it);
    }
}

double BenchmarkMallocPerfLWN::measuredValue(int subtest, double elapsedTime)
{
    return (double)m_numMallocs / elapsedTime;
}

void BenchmarkMallocPerfLWN::deinit(int subtest)
{
    for (std::vector<void*>::iterator it = m_initMallocs.begin(); it != m_initMallocs.end(); ++it) {
        free(*it);
    }
}

BenchmarkMallocPerfLWN::~BenchmarkMallocPerfLWN()
{
}
