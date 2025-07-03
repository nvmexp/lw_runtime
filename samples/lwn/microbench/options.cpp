/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "options.hpp"
#include "utils.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
// Running a large number of frames takes a long time and there's no
// progress indicator.  Use a low number for the default "number of
// frames" to make microbench complete quickly.
#define DEFAULT_NUM_FRAMES 1
#else
#define DEFAULT_NUM_FRAMES 1000
#endif

Options g_options;

Options::Options() :
    m_numFrames(DEFAULT_NUM_FRAMES),
    m_numLoops(1),
    m_cpuRateMHz(918),
    m_gpuRateMHz(384),
    m_emcRateMHz(800),
    m_flags(OPENGL_TESTS_BIT),
    m_testName(nullptr),
    m_lwnglslcDllPath(nullptr)
{
}

void Options::init(int argc, const char** argv)
{
    const char *helpMsg =
        "Supported options:\n"
#if defined(_WIN32)
        "-i                     : Wait for esc before exiting program\n"
#endif
        "--debug                : Add debug layers\n"
        "--enable-gpu-counters  : Collect GPU counters\n"
        "--flip                 : Offscreen and presentation control\n"
        "--rest                 : Output in REST format\n"
        "--skip-opengl          : Skip OpenGL tests\n"
        "--verbose              : Ignored\n"
        "--frames <count>       : Number of frames to run\n"
        "--loop <count>         : Number of times to repeat the test\n"
        "--lwnglslc-dll <path>  : Load path for glslc-dll\n"
        "--test <string>        : Exact test name or \"sub*\" for subexpression match\n"
#if defined(LW_HOS)
        "--cpu <value>          : CPU frequency\n"
        "--gpu <value>          : GPU frequency\n"
        "--emc <value>          : memory requency\n"
#else
        "--no-platform-skip     : Force HOS tests to run on non-HOS platforms\n"
#endif
        "\n";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--flip")) {
            m_flags |= FLIP_BIT;
        } else if (!strcmp(argv[i], "--enable-gpu-counters")) {
            m_flags |= COLLECT_GPU_COUNTERS_BIT;
        } else if (!strcmp(argv[i], "--verbose")) {
            m_flags |= VERBOSE_BIT;
        } else if (!strcmp(argv[i], "--debug")) {
            m_flags |= DEBUG_LAYER_BIT;
        } else if (!strcmp(argv[i], "--skip-opengl")) {
            m_flags &= ~OPENGL_TESTS_BIT;
        } else if (!strcmp(argv[i], "--rest")) {
            m_flags |= REST_OUTPUT_BIT;
        } else if (!strcmp(argv[i], "--frames")) {
            if (i + 1 >= argc) {
                PRINTF("must specify integer argument for --frames\n");
                exit(1);
            }
            m_numFrames = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--loop")) {
            if (i + 1 >= argc) {
                PRINTF("must specify integer argument for --loop\n");
                exit(1);
            }
            m_numLoops = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--test")) {
            if (i + 1 >= argc) {
                PRINTF("must specify a string argument for --test\n");
                exit(1);
            }
            m_testName = argv[++i];
        } else if (!strcmp(argv[i], "--lwnglslc-dll")) {
            if (i + 1 >= argc) {
                PRINTF("must specify a string argument for --lwnglslc-dll\n");
                exit(1);
            }
            m_lwnglslcDllPath = argv[++i];
#if defined(LW_HOS)
        } else if (!strcmp(argv[i], "--cpu")) {
            if (i + 1 >= argc) {
                PRINTF("must specify integer argument for --cpu\n");
                exit(1);
            }
            m_cpuRateMHz = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--gpu")) {
            if (i + 1 >= argc) {
                PRINTF("must specify integer argument for --gpu\n");
                exit(1);
            }
            m_gpuRateMHz = atol(argv[++i]);
       } else if (!strcmp(argv[i], "--emc")) {
            if (i + 1 >= argc) {
                PRINTF("must specify integer argument for --emc\n");
                exit(1);
            }
            m_emcRateMHz = atol(argv[++i]);
#else
  #if defined(_WIN32)
        } else if (!strcmp(argv[i], "-i")) {
            m_flags |= INTERACTIVE_MODE_BIT;
  #endif
        } else if (!strcmp(argv[i], "--no-platform-skip")) {
            m_flags |= NO_PLATFORM_SKIP_BIT;
#endif
        } else {
            PRINTF(helpMsg);

            if ((strcmp(argv[i], "--help") == 0) ||
                (strcmp(argv[i], "-h") == 0)) {
                exit(0);
            } else {
                PRINTF("error: unknown option '%s'\n\n", argv[i]);
                exit(1);
            }
        }
    }
}

bool Options::matchTestName(const char* name) const
{
    if (!m_testName)
        return true;

    size_t len = strlen(m_testName);
    if (len > 0) {
        if (m_testName[len - 1] == '*') {
            // Check if the first 'len' chars of the full testname (including
            // subtest name) match the "--test <test>" filter.
            return strncmp(m_testName, name, len-1) == 0;
        }
        return !strcmp(m_testName, name);
    }
    return false;
}
