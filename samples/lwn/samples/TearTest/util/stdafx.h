#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#define _USE_MATH_DEFINES

#include <windows.h>
#include <winnt.h>
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <intrin.h>

//#define IMAGE_SUPPORT
#ifdef IMAGE_SUPPORT
 #include <IL/il.h>
#endif

#include "error.h"
#include "renderer.h"
#include "utils.h"

const float MIN_FPS = 1.0f;
const float START_FPS = 60.0f;

using namespace std;

#ifdef LW_INTERNAL
void SetFmin(float fps);
void ToggleLwrsorState(bool forceOn = false);
void GetVRRParams();
#endif // LW_INTERNAL

void RunSettings();

extern int winWidth;
extern int winHeight;
extern float fps;
extern float fpsGPU;
extern float targetFrameTimeMs;
extern float measuredFrameTime;
extern int vsync;
extern bool cpuTimer;
extern float monMinRefresh;
extern float monMaxRefresh;
extern bool fullscreen;
extern char *resString;
extern char *apiString;
