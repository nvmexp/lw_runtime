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

#include "primitives.h"

class TearTestApp
{
public:

    void KeyPress(int key);
    void KeyPressSpecial(int key);
    void Mouse(int x, int y, bool btnDown);
    void Render();
    void ToggleHelp();
    void ParseCmdLine(int argc, char **argv);
    void Init();
    void DeInit();
    Scene &GetScene() { return s; };

private:
    void SetQAState(QA_STATE s);
    void SettingsToClipboard();

    float exitTime;
    const float HELP_TIME_DURATION = 3.0f;
    float helpTime = HELP_TIME_DURATION;

    HelpTex helpTex;
    WorkloadTimer wt;
    int noRender = 0;
    float noRenderTimeout = 2.5f;
    Alarm a;
    Timer t, totalTime;
    Scene s;
    unsigned __int64 gpuTimestampLastFrame = 0;

    float userMonMinRefresh = 0;
    bool noLwrsor = 0;
};
