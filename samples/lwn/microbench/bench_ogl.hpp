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

// Context class for running a benchmark case.  Collection of some
// platform dependent utilities (like flipping the display) plus basic
// benchmark infrastructure functionality like timing, reporting
// results.
class BenchmarkContextOGL
{
private:
protected:
    LwnUtil::Timer* m_timer;
    int             m_width;
    int             m_height;

    void setupDefaultState();
    void run(ResultCollector* results, BenchmarkCaseOGL* test);
public:
    BenchmarkContextOGL(int w, int h);
    virtual ~BenchmarkContextOGL();

    // Run all BenchmarkCases
    void runAll(ResultCollector* results);

    virtual void flip() = 0;
};
