/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "bench_ogl.hpp"
#include "options.hpp"
#include <assert.h>

#include "cases/cpu_overhead.hpp"
#include "cases/trirate_ogl.hpp"
#include "cases/fillrate_ogl.hpp"

#include <GL/gl.h>
#include <stdio.h>

using namespace LwnUtil;

BenchmarkContextOGL::BenchmarkContextOGL(int w, int h) :
    m_width(w),
    m_height(h)
{
    m_timer = Timer::instance();
}

BenchmarkContextOGL::~BenchmarkContextOGL()
{
}

void BenchmarkContextOGL::setupDefaultState()
{
    // TODO some generic reset GL state function
    LwnUtil::g_glDepthFunc(GL_LEQUAL);
    LwnUtil::g_glDisable(GL_DEPTH_TEST);
    LwnUtil::g_glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    LwnUtil::g_glDepthMask(GL_TRUE);
}

void BenchmarkContextOGL::runAll(ResultCollector* collector)
{
    {
        BenchmarkTrirateOGL test(m_width, m_height);
        run(collector, &test);
    }

    {
        BenchmarkFillrateOGL test(m_width, m_height);
        run(collector, &test);
    }

    {
        BenchmarkCpuOverheadOGL test(m_width, m_height);
        run(collector, &test);
    }
}

void BenchmarkContextOGL::run(ResultCollector* collector, BenchmarkCaseOGL* bench)
{
    int      nSubtests = bench->numSubtests();

    for (int subtestIdx = 0; subtestIdx < nSubtests; subtestIdx++) {
        auto descr = bench->description(subtestIdx);

        if (!g_options.matchTestName(descr.name))
            continue;

        PRINTF("Initialize benchmark case %s (%d/%d)\n", descr.name, subtestIdx+1, nSubtests);
        FLUSH_STDOUT();

        collector->begin(descr.name, descr.units);
        setupDefaultState();
        bench->init(subtestIdx);

        LwnUtil::g_glFinish();

        uint64_t startTime = m_timer->getTicks();

        const int numFrames = g_options.numFrames();
        uint32_t drawFlags  = 0;

        if (g_options.flags() & Options::FLIP_BIT) {
            drawFlags |= BenchmarkCase::DISPLAY_PRESENT_BIT;
        }

        if (g_options.flags() & Options::COLLECT_GPU_COUNTERS_BIT) {
            assert(0);   //unimplemented
        }

        for (int i = 0; i < numFrames; i++) {
            DrawParams params;
            params.flags = drawFlags;

            bench->draw(&params);

            if (drawFlags & BenchmarkCase::DISPLAY_PRESENT_BIT) {
                flip();
            }

#if LWOS_IS_HOS
                // On HOS need to flush due to tough (1 sec) GPU queue timeout.
                if ((i & 7) == 0) {
                    LwnUtil::g_glFlush();
                }
#endif

        }

        LwnUtil::g_glFinish();

        uint64_t endTime = m_timer->getTicks();
        double   value   = bench->measuredValue(subtestIdx, m_timer->ticksToSecs(endTime - startTime));

        collector->end(value);

        bench->deinit(subtestIdx);
    }
}

BenchmarkCaseOGL::BenchmarkCaseOGL(int w, int h) :
    BenchmarkCase(w, h)
{
}

BenchmarkCaseOGL::~BenchmarkCaseOGL()
{
}
