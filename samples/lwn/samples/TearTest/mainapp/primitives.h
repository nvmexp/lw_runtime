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

class Primitive
{
    virtual void Init() { };
    virtual void Render() = 0;
};

class ScrollBar : public Primitive
{
public:
    ScrollBar(float x, float y, float min, float max, char* name, float initialV = 0, const char* units = 0);
    void Render();
    void Mouse(int x, int y, bool btnDown);
    void Set(float v);

    float x, y;
    float w = 200, h = 5;
    float value;
    char valueStr[32];
    const char* name;
    const char* units = 0;
    float milw, maxV;
    int color = blue;
    int lwrColor = darkgray;
    bool mouseInput = false;
    float quantization = 0.25f;

    const float BOX = 5;
    const float TEXT_OFFSET = 120.0f;
};

template <int T>
class Plot : public Primitive
{
public:
    void Render();
    void Push(float v);
    float data[T] = { 0 };
    int lwrIdx = 0;
    int color = white;
    const char *name = 0;
    int nameOffset = 0;
};

class Stripe : public Primitive
{
public:
    virtual void Init() { };
    virtual void Render();
    void SetColor(const float* c);
    virtual void DrawInternal(float x1, float x2);
    void SpeedUp(float amt);

    float x = 0;
    float w = 50;
    float speed = 0.25;  // Cover width of window in 4 secs
    float color[4];
    bool gradient = false;
};

class TexStripe : public Stripe
{
public:
    void Init();
    void DrawInternal(float x1, float x2);

    bool texEnabled = 0;
    unsigned int tex = 0;
    unsigned int texW;
    unsigned int texH;
};

class Background : public Primitive
{
public:
    void Render();
    void Init();

    float color[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
    bool gradient = false;
};

class HelpTex : public Primitive
{
public:
    ~HelpTex();
    void Init();
    void Render();

    unsigned int width;
    unsigned int height;
    unsigned int tex = 0;
};

class WorkloadTimer : public Primitive
{
public:
    void Render();
    void Correct(float workLoadTimeMs, float loadTimeMs);

    float m_workLoadTimeMs     = 0.0;       // Seed time for rendering the actual frame
    float m_timePerIterationMs = 0.5e-03;   // Seed time for rendering one GPU load iteration
    int   m_numIterations      = 1000;
    float m_decayFactor        = 0.04;      // Smaller decay factor dampens the change more

    int   m_numIterationsPrev;
};

enum OSC_TYPE
{
    OSC_SINE,
    OSC_SQUARE
};

class Oscillator
{
public:
    virtual float Get(float dt) = 0;
    virtual const char* GetTypeStr() = 0;
    virtual OSC_TYPE GetType() = 0;
    float period  = 0;
    float freq = M_PI_4;
};

class SineOscillator : public Oscillator
{
public:
    float Get(float dt)
    {
        period += dt * freq;
        if (period > M_PI)
        {
            period -= 2.0f * M_PI;
        }
        return sin(period);
    }

    const char* GetTypeStr() { return "Sine Osc"; }

    OSC_TYPE GetType() { return OSC_SINE; }
};

class SquareOscillator : public Oscillator
{
public:
    float Get(float dt)
    {
        period += dt * freq;
        if (period > M_PI)
        {
            period -= 2.0f * M_PI;
        }
        return period > 0 ? 1.0f : -1.0f;
    }

    const char* GetTypeStr() { return "Square Osc"; }
    OSC_TYPE GetType() { return OSC_SQUARE; }
};

class Scene
{
public:
    Scene();
    ~Scene();
    void Init();
    void UpdateState();
    void Render();
    void Mouse(int x, int y, bool btnDown);
    void SaveCmdLine(string &str);
    void ToggleHelp();
    void SpeedUp(float amt);
    void OscFreqUp(float amt);
    void OscAmtUp(float amt);
    void RefreshSettings();
    void SaveScreenshot();
    void ToggleTexStripe();
    void TogglePlots();
    void ToggleBars();
    void SwitchOscillator(OSC_TYPE type);

    // primitives
    Oscillator* osc;

    TexStripe s1;
    Stripe s2;
    HelpTex helpTex;
    Background bgnd;
    ScrollBar sbFps;
    ScrollBar sbOsc;
    ScrollBar sbRate;

    Plot<256> cpuPlot;
    Plot<256> gpuPlot;
    bool      showPlots = 1;
    bool      showBars = 1;

    // state
    float oscAmt = 0;
    float oscFps;
    float smoothFps;

    char status[256] = { "Tear Test" };
    char *msg = 0;

    int frameIdx;
    int frameIdxHelp;

    const int HELP_DURATION = 3; // 2 sec
};
