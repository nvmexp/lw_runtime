/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <lwndevtools_bootstrap.h>

#if defined(LW_LINUX)
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

static const LWNdevtoolsBootstrapFunctions* devtools;

static bool TestZbcGet(size_t tableSize, const uint32_t expectedClearColor[4], const uint32_t expectedClearDepthUI)
{
    bool success;
    bool got_color = false;
    bool got_depth = false;

    for (size_t i = 0; i < tableSize; i++)
    {
        uint32_t fb[4] = {};
        uint32_t ds[4] = {};
        uint32_t fmt; // We dont actually test for formats, as it seems they change...

        success = devtools->ZbcGetColorTableEntry(g_device, i, fb, ds, &fmt);
        TEST_FMT(success, "get fast clear color failed!");

#define __SAME_COLORS(x,y) (x[0] == y[0]) && (x[1] == y[1]) && (x[2] == y[2]) && (x[3] == y[3])
        got_color += __SAME_COLORS(fb, expectedClearColor);
        got_color += __SAME_COLORS(ds, expectedClearColor);
#undef __SAME_COLORS

        success = devtools->ZbcGetDepthTableEntry(g_device, i, fb, &fmt);
        TEST_FMT(success, "get fast clear depth failed!");
        got_depth += fb[0] == expectedClearDepthUI;
    }

    TEST_FMT(got_color, "get expected fast clear color failed!");
    TEST_FMT(got_depth, "get expected fast clear depth failed!");

    return true;
}

inline bool TestFastColor()
{
    llgd_lwn::QueueHolder qh;

    bool success;
    size_t size = 0;

    qh.Initialize(g_device);

    // Let's register one of each.
    const static float clearDepth = 3.14159;
    const static uint32_t clearDepthUI = *((uint32_t*)(&clearDepth));
    success = g_device->RegisterFastClearDepth(clearDepth);
    TEST_FMT(success, "register fast clear depth failed!");

    const static uint32_t clearColor[4] = {299, 792, 458, 6674};
    success = g_device->RegisterFastClearColorui(clearColor, Format::R32UI);
    TEST_FMT(success, "register fast clear color failed!");

    devtools = lwnDevtoolsBootstrap();

    // And check that we can get it back.
    success = devtools->ZbcGetTableSize(g_device, &size);
    TEST_FMT(success, "get fast clear table size failed!");
    TEST_FMT(size > 1, "fast clear table size too small!");

    // Test if getting registered Zbc values are correct or not
    TEST_FMT(TestZbcGet(size, clearColor, clearDepthUI), "get registered fast clear color and depth by ZbcGet functions failed.");

    // Change fast clear color & depth by bootstrap functions
    const static float newClearDepth = 2.71828;
    const static uint32_t newClearDepthUI = *((uint32_t*)(&newClearDepth));
    const static uint32_t newClearColor[4] = {7609, 14, 465, 68};

    for (size_t i=0; i < size; i++)
    {
        // Get format for color
        uint32_t fb[4] = {};
        uint32_t ds[4] = {};
        uint32_t fmt;
        success = devtools->ZbcGetColorTableEntry(g_device, i, fb, ds, &fmt);
        TEST_FMT(success, "get fast clear color and its format failed!");

        // Change Zbc color
        success = devtools->ZbcSetColorTableEntry(g_device, i, newClearColor, newClearColor, fmt);
        TEST_FMT(success, "set fast clear color failed!");

        // Get format for depth
        success = devtools->ZbcGetDepthTableEntry(g_device, i, fb, &fmt);
        TEST_FMT(success, "get fast clear depth and its format failed!");

        // Change Zbc depth
        success = devtools->ZbcSetDepthTableEntry(g_device, i, newClearDepthUI, fmt);
        TEST_FMT(success, "set fast clear depth failed!");
    }

    // Test if colors and depth are correctly set by ZbcSet*** functions or not
    TEST_FMT(TestZbcGet(size, newClearColor, newClearDepthUI), "get fast clear color and depth set by ZbcSet*** failed.");

    return true;
}

LLGD_DEFINE_TEST(FastColor, UNIT, LwError Execute() { return TestFastColor() ? LwSuccess : LwError_IlwalidState; });
