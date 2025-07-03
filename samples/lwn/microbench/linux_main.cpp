/*
 * Copyright (c) 2020, LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>

#include "lwnWin/lwn_win.h"

#include "bench.hpp"
#include "bench_ctx.hpp"
#include "options.hpp"

static const int32_t DEFAULT_WIDTH   = 1280, DEFAULT_HEIGHT   = 720;
static const int32_t OFFSCREEN_WIDTH = 1920, OFFSCREEN_HEIGHT = 1200;

int main(int argc, const char *argv[])
{
    // Parse command line options
    g_options.init(argc, argv);

    bool renderOffscreen = !(g_options.flags() & Options::FLIP_BIT);

    int width  = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;

    LwnWin *lwnWin       = nullptr;
    void   *nativeWindow = nullptr;

    if (renderOffscreen) {
        width  = OFFSCREEN_WIDTH;
        height = OFFSCREEN_HEIGHT;
    } else {
        lwnWin = LwnWin::GetInstance();
        if (lwnWin == NULL) {
            printf("Cannot obtain window interface.\n");
            return false;
        }

        nativeWindow = lwnWin->CreateWindow("LWN", width, height);
        if (!nativeWindow) {
            printf("Cannot create window.\n");
            return false;
        }
    }

    LWNdevice *device = LwnUtil::init((g_options.flags() & Options::DEBUG_LAYER_BIT) ?
                                      LwnUtil::LWN_INIT_DEBUG_LAYER_BIT : 0, nullptr);
    if (!device) {
        printf("Cannot create lwn device.\n");
        return EXIT_FAILURE;
    }

    ResultCollector collector;
    ResultPrinterStdout printer;

    {
        BenchmarkContextLwWinsysLWN ctx(device, nativeWindow, width, height);
        ctx.runAll(&collector);
    }

    collector.print(printer);

    delete device;

    if (nativeWindow) {
        lwnWin->DestroyWindow(nativeWindow);
    }

    return EXIT_SUCCESS;
}
