/*
 * Copyright (c) 2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>

#include "lwnWin/lwn_win.h"

#include <lwn/lwn.h>
#include <demo.h>

#include "lwwinsys_app.h"

static const int32_t DEFAULT_WIDTH = 1280, DEFAULT_HEIGHT = 720;

int main(int argc, char** argv)
{
    bool continueRender = true;

    int width  = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;

    LwnWin *lwnWin = LwnWin::GetInstance();
    if (lwnWin == NULL) {
        printf("Cannot obtain window interface.\n");
        return false;
    }

    void *nativeWindow = lwnWin->CreateWindow("LWN", width, height);
    if (!nativeWindow) {
        printf("Cannot create window.\n");
        return false;
    }

    // Set DEMO framebuffer size to match.
    DEMOGfxOffscreenWidth = width;
    DEMOGfxOffscreenHeight = height;

    appInit(argc, argv, nativeWindow);

    while (continueRender)
    {
        continueRender = appDisplay();
    }

    appShutdown();

    lwnWin->DestroyWindow(nativeWindow);

    return 0;
}
