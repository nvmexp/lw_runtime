/*
 * Copyright (c) 2015-2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include <cstdint>

class Options
{
private:
    int         m_numFrames;
    int         m_numLoops;
    uint32_t    m_cpuRateMHz;
    uint32_t    m_gpuRateMHz;
    uint32_t    m_emcRateMHz;
    uint32_t    m_flags;
    const char* m_testName;
    const char* m_lwnglslcDllPath;

public:
    enum {
        // present FB to display
        FLIP_BIT                 = 1 << 0,
        COLLECT_GPU_COUNTERS_BIT = 1 << 1,
        VERBOSE_BIT              = 1 << 2,
        DEBUG_LAYER_BIT          = 1 << 3,
        OPENGL_TESTS_BIT         = 1 << 4,
        INTERACTIVE_MODE_BIT     = 1 << 5,
        // Force run tests on all platforms even if the test marks
        // itself as runnable by default only on HOS.
        NO_PLATFORM_SKIP_BIT     = 1 << 6,
        REST_OUTPUT_BIT          = 1 << 7       // Use REST parser compatible result output
    };

    Options();

    void init(int argc, const char** argv);

    uint32_t flags() const { return m_flags; }

    uint32_t numFrames() const { return m_numFrames; }
    uint32_t numLoops() const { return m_numLoops; }

    uint32_t cpuRateMHz() const { return m_cpuRateMHz; }
    uint32_t gpuRateMHz() const { return m_gpuRateMHz; }
    uint32_t emcRateMHz() const { return m_emcRateMHz; }

    bool matchTestName(const char* testName) const;

    const char* lwnglslcDllPath() const { return m_lwnglslcDllPath; }

};

extern Options g_options;

