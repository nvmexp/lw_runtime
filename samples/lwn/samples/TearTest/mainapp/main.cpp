/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "stdafx.h"
#include "utils.h"
#include "renderer.h"
#include "TearTestApp.h"
#include "window.h"

#define VERSION "1.19"

int winWidth = 0;
int winHeight = 0;
bool fullscreen = 1;
char *resString = NULL;

float fps = START_FPS;
float targetFrameTimeMs = 1000.0f / fps;
float measuredFrameTime;
float monMinRefresh = 0;
float monMaxRefresh = 160;
float fpsGPU = 0;
int vsync = 1;
bool cpuTimer = 1;

Renderer *renderer;
TearTestApp tt;
Window *window;

void AtExitFunc()
{
    tt.DeInit();
    flushall();
}

int main(int argc, char **argv)
{
    atexit(AtExitFunc);

    puts("LWPU Tear, Stutter and Overdrive Test v" VERSION ".\n");

    // Argument parsing should happen first belwase selection of
    // Window/Renderer objects depends upon inputs

    tt.ParseCmdLine(argc, argv);

    window = CreateWindowClass("GLUT");
    renderer = CreateRenderer(apiString);

    window->Create("TearTest", winWidth, winHeight, fullscreen);
    tt.Init();
    window->Run();

    delete renderer;
    delete window;

    return 0;
}
