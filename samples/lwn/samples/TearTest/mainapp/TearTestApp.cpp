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
#include "renderer.h"
#include "TearTestApp.h"
#include "window.h"

QA_STATE    qaState = NONE;
char       *qaFileName = 0;

#ifdef BUILD_LWN
  char     *apiString = "LWN";
#else
  char     *apiString = "OpenGL";
#endif

ofstream    qaFile;
bool        logEdid;

extern Renderer *renderer;
extern TearTestApp tt;
extern Window *window;

void TearTestApp::SetQAState(QA_STATE s)
{
    if (qaFile.is_open())
    {
        qaState = s;

        qaFile << qaFileName << (s == YES ? ": Yes\n" : ": No\n");

        qaFile.flush();

        exitTime = totalTime.getTime() + 2;

        if (noRender)
        {
            renderer->BeginFrame();
            renderer->BlitFrontToBackBuffer();
            renderer->RenderQAState(qaState);
            renderer->EndFrame();
            window->Redisplay();
        }
    }
}

void TearTestApp::ToggleHelp()
{
    float timeNow = totalTime.getTime();
    if (helpTime > timeNow)
    {
        helpTime = 0;
    }
    else
    {
        helpTime = timeNow + HELP_TIME_DURATION;
    }
}

void TearTestApp::SettingsToClipboard()
{
    string str = "teartest.exe -fps ";
    str += to_string(fps);
    if (resString)
    {
        str += " -res " + to_string(winWidth) + "x" + to_string(winHeight);
    }
    str += " -vsync ";
    str += to_string(vsync);
    if (!fullscreen)
    {
        str += " -w";
    }
    if (noRender)
    {
        str += " -noRender";
    }
    s.SaveCmdLine(str);
    CopyToClipboard(str);
    if (!fullscreen)
    {
        MessageBox(NULL, "Copied to Clipboard.", "Tear Test", MB_OK);
    }
}

void TearTestApp::Mouse(int x, int y, bool btnDown)
{
    s.Mouse(x, y, btnDown);
}

void TearTestApp::KeyPress(int key)
{
    switch (key)
    {
    case 27: exit(0); break;
    case '0':
        key = '9' + 1;
    case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
        fps = 10.0f * (key - '0'); targetFrameTimeMs = 1000.0f / fps; break;
    case '<': case ',': s.OscAmtUp(-0.5f); break;
    case '>': case '.': s.OscAmtUp(0.5f); break;
    case '[': s.OscFreqUp(-M_PI_4 * 0.5f); break;
    case ']': s.OscFreqUp(M_PI_4 * 0.5f); break;
    case 'V': case 'v': vsync = !vsync;  renderer->SetVSync(vsync); break;
    case 'G': case 'g': s.bgnd.gradient = !s.bgnd.gradient; break;
    case 'C': case 'c': SettingsToClipboard(); break;
    case 'H': case 'h': ToggleHelp(); break;
    case '+': case '=': s.SpeedUp(1.1f); break;
    case '-': case '_': s.SpeedUp(1.0f / 1.1f); break;
    case 'S': case 's': s.SaveScreenshot(); break;
    case 'T': case 't': s.ToggleTexStripe(); break;
    case 'P': case 'p': s.TogglePlots(); break;
    case 'R': case 'r': noRender = !noRender; break;
    case ' ': cpuTimer = !cpuTimer; break;
    case '~': s.SwitchOscillator(OSC_SINE); break;
    case '`': s.SwitchOscillator(OSC_SQUARE); break;
    case 'B': case 'b': s.ToggleBars(); break;

#ifdef LW_INTERNAL
    case 'Y': case 'y': SetQAState(YES); break;
    case 'N': case 'n': SetQAState(NO); break;
    case 'Z': SetFmin(1.0f); userMonMinRefresh = 1; break;
    case 'M': case 'm': ToggleLwrsorState(); noLwrsor = 1; break;
#endif // LW_INTERNAL
    }
    s.RefreshSettings();
    window->Redisplay();
}

void TearTestApp::Render()
{
    static unsigned __int64 timestamp[3];
    static int queryStartIndex = 0;
    static int countersInserted = 0;

    float timeNow = totalTime.getTime();

    if ((exitTime) && (exitTime < timeNow))
    {
        PostQuitMessage(0);
    }

    if (noRender)
    {
        if (noRenderTimeout > timeNow)
        {
            renderer->BeginFrame();
            renderer->Clear();
            char msg[256];
            sprintf(msg, "Stopping render in %.2f s.", noRenderTimeout - timeNow);
            renderer->RenderString(winWidth / 2 - 60, winHeight / 2, msg);
            renderer->EndFrame();
            window->Redisplay();
            Sleep(1);
            return;
        }
        if (noRender == 2)
        {
            window->Reschedule();
            Sleep(1);
            return;
        }
        renderer->BeginFrame();
        renderer->Clear();
        renderer->RenderString(winWidth / 2 - 40, winHeight / 2, s.msg ? s.msg : "Refresh Stopped.");
        renderer->EndFrame();
        window->Redisplay();
        noRender++;
        return;
    }

    renderer->BeginFrame();
    renderer->InsertQuery(queryStartIndex + 0);

    s.Render();
    if (helpTime > timeNow)
    {
        helpTex.Render();
    }
    renderer->RenderQAState(qaState);

    if (cpuTimer)
    {
        float late = a.Wait();                    // Ensure that we met the sw frame time
        float waitTime = max(2, targetFrameTimeMs - late);
        a.Set(waitTime);

        countersInserted = 0;                   // No longer inserting load counters
    }
    else
    {
        renderer->InsertQuery(queryStartIndex + 1);
        wt.Render();
        renderer->InsertQuery(queryStartIndex + 2);
        countersInserted++;
    }
    queryStartIndex = (queryStartIndex == 0) ? 3 : 0;   // For next frame and queries

    // Obtain queries for hw frame and data for loading. There is a sync here - this
    // frame is waiting for queries from previous frame to be completed but this is
    // fine - the frame is all composed on the CPU and only thing remaining is kickoff.
    // So the CPU side is running too fast, it will ensure that we don't dispatch work
    // until GPU is done with previous frame (and it still may have swap pending).

    unsigned __int64 startFrame = renderer->GetQuery(queryStartIndex + 0);
    unsigned __int64 delta      = startFrame - timestamp[0];
    fpsGPU = (float)(1.0e09 / delta);

    timestamp[0] = startFrame;
    if ((!cpuTimer) && (countersInserted >= 2))
    {
        timestamp[1] = renderer->GetQuery(queryStartIndex + 1);
        timestamp[2] = renderer->GetQuery(queryStartIndex + 2);

        wt.Correct((timestamp[1] - timestamp[0]) * 1.0e-06,     // Colwerted from nano to milli
                   (timestamp[2] - timestamp[1]) * 1.0e-06);
    }

    renderer->EndFrame();
    // Obtain SW frame measurement time
    measuredFrameTime = t.getTimeAndReset();

    window->Redisplay();
}

void TearTestApp::ParseCmdLine(int argc, char **argv)
{
#if _WIN32
    bool skipGUI = (GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000;
    if (!skipGUI && argc == 1)
    {
        puts("* Provide any command-line argument or hold CTRL to skip the settings dialog on startup. *\n\n");
    }
#endif

    puts("Hotkeys:\n\n"
        "Arrow Keys\t- Change Refresh Rate\n"
        "< and >\t\t- Change Oscillation\n"
        "[ and ]\t\t- Change Oscillation Speed\n"
        "1...9\t\t- Set FPS to 10...90\n"
        "+,-\t\t- Stripe Speed\n"
        "V\t\t- Toggle VSync\n"
        "G\t\t- Toggle Gradient Background\n"
        "C\t\t- Copy Current Settings to Clipboard\n"
        "P\t\t- Toggle Timing Plots\n"
        "R\t\t- Toggle Rendering\n"
        "S\t\t- Save Screenshot\n"
        "T\t\t- Toggle Texture on Stripe\n"
        "M\t\t- Measure Frame Submit / Present Return\n"
        "B\t\t- Toggle the Moving Bars\n"
        "~ and `\t\t- Switch Oscillator Between Sine and Square wave\n"
        "Y and N\t\t- Pass/Fail the Test\n"
        "SPACE\t\t- CPU / GPU Time Pacing\n\n");
    puts("Command-line Arguments:\n\n"
        "-w\t\t- Start Windowed\n"
        "-g\t\t- Gradient Background\n"
        "-fps\t\t- Set Center FPS\n"
        "-osc\t\t- Set Oscillation Amount (in FPS)\n"
        "-oscF\t\t- Set Oscillation Frequency (in Hz)\n"
        "-ostT\t\t- Set Oscillator Type (sine / square)\n"
        "-vsync 0/1\t- Set VSync\n"
        "-res WxH\t- Set Resolution\n"
        "-color R G B\t- Set Background Color [0.0, 1.0]\n"
        "-api <OpenGL, LWN> - choose api\n"
#ifdef LW_INTERNAL
        "-qaName FILE\t- Set QA File Name\n"
        "-logEdid\t- Log the monitor EDID to the QA log file\n"
        "-fmin\t\t- Override the monitor min refresh rate\n"
#endif // LW_INTERNAL
        "-noBars\t\t- Don't render the bars\n"
        "-noPlot\t\t- Don't render the plot\n"
        "-message\t- Show the user a message. Use '\\n' for new line.\n"
        "-timeout s\t- Exit after predefined seconds\n"
        "-noLwrsor\t- Turn off the cursor on startup\n"
        "-noRender\t- Render only 1 frame\n\n");
#ifdef LW_INTERNAL
    puts("Send Questions/Requests to: rdimitrov@lwpu.com.\n");
#endif // LW_INTERNAL

#if _WIN32
    if ((argc == 1) && (skipGUI == false))
    {
        RunSettings();
    }
    else
#endif
    {
        // cmdline parsing
        for (int i = 1; i < argc; i++)
        {
            if (!strcmp(argv[i], "-w"))
            {
                fullscreen = 0;
            }
            else if (!strcmp(argv[i], "-g"))
            {
                s.bgnd.gradient = 1;
            }
            else if (!strcmp(argv[i], "-fps"))
            {
                if (++i < argc)
                {
                    float f = atof(argv[i]);
                    if (f > 1.0f)
                    {
                        fps = f;
                        targetFrameTimeMs = 1000.0f / fps;
                    }
                }
            }
            else if (!strcmp(argv[i], "-osc"))
            {
                if (++i < argc)
                {
                    s.oscAmt = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "-oscT"))
            {
                if (++i < argc)
                {
                    if (!strcmp(argv[i], "square"))
                    {
                        s.SwitchOscillator(OSC_SQUARE);
                    }
                }
            }
            else if (!strcmp(argv[i], "-oscF"))
            {
                if (++i < argc)
                {
                    s.osc->freq = atof(argv[i]) * M_PI * 2.0f;
                }
            }
            else if (!strcmp(argv[i], "-vsync"))
            {
                if (++i < argc)
                {
                    vsync = atoi(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "-res"))
            {
                if (++i < argc)
                {
                    resString = argv[i];
                }
            }
            else if (!strcmp(argv[i], "-color"))
            {
                if (i + 3 < argc)
                {
                    s.bgnd.color[0] = atof(argv[++i]);
                    s.bgnd.color[1] = atof(argv[++i]);
                    s.bgnd.color[2] = atof(argv[++i]);
                }
                else
                {
                    Error("Couldn't parse color.");
                }
            }
            else if (!strcmp(argv[i], "-qaName"))
            {
                if (++i < argc)
                {
                    qaFileName = argv[i];
                }
            }
            else if (!strcmp(argv[i], "-noPlot"))
            {
                s.TogglePlots();
            }
            else if (!strcmp(argv[i], "-noBars"))
            {
                s.ToggleBars();
            }
            else if (!strcmp(argv[i], "-message"))
            {
                if (++i < argc)
                {
                    s.msg = argv[i];
                }
            }
            else if (!strcmp(argv[i], "-fmin"))
            {
                if (++i < argc)
                {
                    userMonMinRefresh = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "-noRender"))
            {
                noRender = 1;
            }
            else if (!strcmp(argv[i], "-noLwrsor"))
            {
                noLwrsor = true;
            }
            else if (!strcmp(argv[i], "-logEdid"))
            {
                logEdid = true;
            }
            else if (!strcmp(argv[i], "-runtime"))
            {
                if (++i < argc)
                {
                    exitTime = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "-api"))
            {
                if (++i < argc)
                {
                    apiString = argv[i];
                }
            }
            else
            {
                fprintf(stderr, "Unknown cmdline argument: %s\n", argv[i]);
                Sleep(1000);
                exit(1);
            }
        }
        s.RefreshSettings();
    }

    if (resString)
    {
        sscanf(resString, "%dx%d", &winWidth, &winHeight);
    }

    if (s.oscAmt > fps)
    {
        Error("The oscillation amount should be less than the FPS.");
    }
    // Initialized so that initial time correction is zero.
    measuredFrameTime = targetFrameTimeMs / 1000.0;
}

void TearTestApp::Init()
{
    if (qaFileName)
    {
        qaFile.open("TearTest.log", ofstream::out | ofstream::app);
    }

#ifdef LW_INTERNAL
    GetVRRParams();

    if (userMonMinRefresh)
    {
        if (monMinRefresh < 0)
        {
            Error("Couldn't query the EDID monitor min refresh.");
        }
        SetFmin(userMonMinRefresh);
    }

    if (noLwrsor)
    {
        ToggleLwrsorState();
    }
#endif // LW_INTERNAL

    renderer->Init(winWidth, winHeight);
    s.Init();
    helpTex.Init();
    a.Set(5);

    if (vsync == 0)
    {
        renderer->SetVSync(vsync);
    }
    totalTime.start();
}

void TearTestApp::KeyPressSpecial(int key)
{
    switch (key)
    {
        case Window::KEY_LEFT:  fps -= 0.25f; break;
        case Window::KEY_RIGHT: fps += 0.25f; break;
        case Window::KEY_UP:    fps += 5.0f; break;
        case Window::KEY_DOWN:  fps -= 5.0f; break;
    }
    if (fps < 5.0f)
    {
        fps = 5.0f;
    }
    targetFrameTimeMs = 1000.0f / fps;
    s.RefreshSettings();
    window->Redisplay();
}

void TearTestApp::DeInit()
{
#ifdef LW_INTERNAL
    if (userMonMinRefresh)
    {
        SetFmin(monMinRefresh);
    }

    ToggleLwrsorState(true);

    qaFile.close();
#endif
}

void display()
{
    tt.Render();
}

bool mouseBtnDown;

void mouseButton(int button, int state, int x, int y)
{
    if (button == Window::MOUSE_LEFT_BUTTON)
    {
        mouseBtnDown = (state == Window::MOUSE_DOWN);
        tt.Mouse(x, y, mouseBtnDown);
    }
}

void mouseMove(int x, int y)
{
    tt.Mouse(x, y, mouseBtnDown);
}

void keyboard(unsigned char key, int x, int y)
{
    tt.KeyPress(key);
}

void reshape(int width, int height)
{
    winWidth = width;
    winHeight = height;

    renderer->Reshape(width, height);
}

void keyboard2(int key, int x, int y)
{
    tt.KeyPressSpecial(key);
}

