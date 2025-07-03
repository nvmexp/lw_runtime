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
#include "primitives.h"

#ifndef IMAGE_SUPPORT
  #include "stripe.txt"
  #include "help.txt"
#endif

static float timeErrorMs = 0.0;

ScrollBar::ScrollBar(float x, float y, float min, float max, char* name, float initialV, const char* units)
{
    this->x = x + TEXT_OFFSET;
    this->y = y + TEXT_ROW_OFFSET;
    this->milw = min;
    this->maxV = max;
    this->name = name;
    this->units = units;
    Set(initialV);
}

void ScrollBar::Set(float v)
{
    v = quantization * round(v / quantization);
    value = max(milw, min(maxV, v));
    if (units)
    {
        sprintf(valueStr, "%5.2f %s", value, units);
    }
    else
    {
        sprintf(valueStr, "%5.2f", value);
    }
}

void ScrollBar::Render()
{
    renderer->RenderString(x - TEXT_OFFSET, y, name);

    renderer->SetColor(colors[white]);
    renderer->DrawQuad(x, y - 3, x + w, y - 1);

    float boxX = x + (value - milw) / (maxV - milw) * w;

    renderer->SetColor(colors[lwrColor]);
    renderer->DrawWedge(boxX - BOX, y - 2.1f * BOX, boxX + BOX, y + BOX, 0.5f * BOX);

    renderer->SetColor(colors[white]);
    renderer->RenderString(x + w + 15, y, valueStr);
}

void ScrollBar::Mouse(int mx, int my, bool btnDown)
{
    if (mouseInput || (mx > x - BOX && mx < x + w + BOX && my > y - 2 * BOX && my < y + 2 * BOX))
    {
        lwrColor = color;
        if (btnDown)
        {
            value = ((float)(mx - x) / w) * (maxV - milw) + milw;
            Set(value);
            mouseInput = true;
        }
    }
    else
    {
        lwrColor = darkgray;
    }

    if (!btnDown)
    {
        mouseInput = false;
    }
}

template <int T>
void Plot<T>::Render()
{
    renderer->RenderPlot(data, T, lwrIdx, color, nameOffset == 0, name, nameOffset + 1, monMinRefresh, monMaxRefresh);
}

template <int T>
void Plot<T>::Push(float v)
{
    data[lwrIdx] = v;
    lwrIdx = (lwrIdx + 1) % T;
}

void Stripe::SetColor(const float* c)
{
    for (int i = 0; i < 4; i++)
    {
        color[i] = c[i];
    }
}

void Stripe::Render()
{
    // The speed represents the real world movement of a bar and
    // should be independent of fps and resolution. It is also
    // independent of the display size - whether a big display or a small
    // display, the bar should traverse from left to right in the same
    // duration.
    //
    // Define speed in terms of "window crossing / sec" (WC/sec). If speed is
    // 1.0, it would take the bar 1.0 second to start from the left-end and
    // exit at the right end.
    //
    // Given a fps, in a second, we want the bar to move "speed" amount in
    // WC units. Hence, for each frame, the expected speed movement is
    // speed/fps (WC). Then for a given resolution, the movement in X is
    // winWidth * speed/fps.

    x += speed * ((timeErrorMs + targetFrameTimeMs) * 0.001f) * winWidth;

    float hww = winWidth / 2;
    while (x > hww)
    {
        x -= hww;
    }
    float hw = w / 2;
    float drawX = round(x + hw);

    DrawInternal(drawX - hw - hww, drawX + hw - hww);
    DrawInternal(drawX - hw + hww, drawX + hw + hww);
    DrawInternal(drawX - hw,       drawX + hw);
}

void Stripe::DrawInternal(float x1, float x2)
{
    renderer->DrawStripe(x1, x2, color, gradient ? colors[black] : color);
}

void Stripe::SpeedUp(float amt)
{
    speed *= amt;
}

void TexStripe::Init()
{
    void *data;
#ifdef IMAGE_SUPPORT
    uint32_t ilTex = CreateImage("stripe.png", &texW, &texH, &data);

    if (ilTex)
    {
        tex = renderer->CreateTex(Renderer::TEX_RGBA8, texW, texH, data);
        DestroyImage(ilTex);
    }
#else
    texW = stripeImage.width;
    texH = stripeImage.height;
    data = (void *) stripeImage.data;
    tex = renderer->CreateTex(Renderer::TEX_RGBA8, texW, texH, data);
#endif
}

void TexStripe::DrawInternal(float x1, float x2)
{
    if (texEnabled && tex)
    {
        renderer->DrawTexStripe(x1, x1 + texW, color, gradient ? colors[black] : color, tex, winHeight / texH);
    }
    else
    {
        Stripe::DrawInternal(x1, x2);
    }
}

void Background::Render()
{
    if (gradient)
    {
        renderer->DrawGradient(color);
    }
    else
    {
        renderer->Clear();
    }
}

void Background::Init()
{
    renderer->SetClearColor(color);
}

HelpTex::~HelpTex()
{
    renderer->DeleteTex(tex);
}
void HelpTex::Init()
{
    void *data;
#ifdef IMAGE_SUPPORT
    uint32_t ilTex = CreateImage("help.png", &width, &height, &data);

    if (ilTex)
    {
        tex = renderer->CreateTex(Renderer::TEX_RGBA8, width, height, data);
        DestroyImage(ilTex);
    }
#else
    width = helpImage.width;
    height = helpImage.height;
    data = (void *) helpImage.data;
    tex = renderer->CreateTex(Renderer::TEX_RGBA8, width, height, data);
#endif
}

void HelpTex::Render()
{
    if (tex)
    {
        int x = (winWidth - width) / 2;
        int y = (winHeight - height) / 2;

        renderer->DrawTexQuad(x, y, x + width, y + height, tex);
    }
}

float IncrementValue(float newValue, float oldValue, float decayFactor)
{
    return oldValue + decayFactor * (newValue - oldValue);
}

void WorkloadTimer::Correct(float workLoadTimeMs, float loadTimeMs)
{
    float timePerIterationMs = loadTimeMs / m_numIterationsPrev;

    m_workLoadTimeMs     = IncrementValue(workLoadTimeMs,     m_workLoadTimeMs,     m_decayFactor);
    m_timePerIterationMs = IncrementValue(timePerIterationMs, m_timePerIterationMs, m_decayFactor);
}

void WorkloadTimer::Render()
{
    m_numIterationsPrev = m_numIterations;

    float loadTime = targetFrameTimeMs - m_workLoadTimeMs;
    m_numIterations = loadTime / m_timePerIterationMs + 0.5;       // 0.5 for rounding

    if (m_numIterations < 100)      // Don't want this to zero so that we can recover
    {                               // from bad events.
        m_numIterations = 100;
    }

    renderer->LoadGPU(m_numIterations);
}

Scene::Scene() : sbFps(10, 80, 10, 150, "Center FPS", START_FPS),
                 sbOsc(10, 80 + 2.0f * TEXT_ROW_OFFSET, 0, 50, "Osc Amount", 0),
                 sbRate(10, 80 + 4.0f * TEXT_ROW_OFFSET, 0.125f, 5, "Osc Rate", 0, "Hz")
{
    osc = new SineOscillator;
}

void Scene::Init()
{
    float whiteColor[] = { 1,1,1,1 };

    s1.SetColor(whiteColor);
    s2.SetColor(whiteColor);
    s2.gradient = true;
    s2.speed *= 1.5f;
    oscFps = fps;
    smoothFps = fps;
    sbRate.quantization = 1.0f / 16.0f;
    bgnd.Init();
    cpuPlot.name = "CPU (FPS)";
    gpuPlot.color = green;
    gpuPlot.name = "GPU";
    gpuPlot.nameOffset = TEXT_ROW_OFFSET;

    s1.Init();
    s2.Init();
}

void Scene::SwitchOscillator(OSC_TYPE type)
{
    Oscillator *newOsc;

    if (type == OSC_SINE)
    {
        newOsc = new SineOscillator;
    }
    else
    {
        newOsc = new SquareOscillator;
    }

    *newOsc = *osc;

    delete osc;
    osc = newOsc;
}

Scene::~Scene()
{
    delete osc;
}

void Scene::UpdateState()
{
    // Compute a correction by how much we are off from our targetFrameTime
    // Worst error is when we drop from 120fps to 60fps or 60fps to 30fps.
    // Cap the error so that the bad frames don't mess up things.

    timeErrorMs = measuredFrameTime * 1000.0 - targetFrameTimeMs;
    if (timeErrorMs > targetFrameTimeMs)
    {
        timeErrorMs = targetFrameTimeMs;
    }

    // update state
    float measuredFps = 1.0f / measuredFrameTime;
    const float SMOOTH_K = 0.95f;
    smoothFps = SMOOTH_K * smoothFps + (1.0f - SMOOTH_K) * measuredFps;

    int l = sprintf(status, "Measured FPS:   %7.2f (%s)\nSmoothened FPS: %7.2f\nTarget:         %7.2f", measuredFps, "Frame Submit Time", smoothFps, max(MIN_FPS, fps + oscFps));

    if (oscAmt)
    {
        sprintf(status + l, " (%.2f - %.2f)", max(MIN_FPS, fps - oscAmt), fps + oscAmt);

        oscFps = oscAmt * osc->Get(measuredFrameTime);
    }
    else
    {
        oscFps = 0;
    }
    targetFrameTimeMs = 1000.0f / (fps + oscFps); // new frame time

    float maxWait = 1000.0f / MIN_FPS;
    if (targetFrameTimeMs < 0 || targetFrameTimeMs > maxWait)
    {
        targetFrameTimeMs = maxWait;
    }

    cpuPlot.Push(measuredFps);
    gpuPlot.Push(fpsGPU);
}

void Scene::Render()
{
    UpdateState();

    // render
    bgnd.Render();

    if (showBars)
    {
        s2.Render();
        s1.Render();
    }

    float textY = 20;

    renderer->SetColor(colors[white]);
    renderer->RenderString(10, textY, status);

    if (msg)
    {
        textY += 100.0f;
        renderer->RenderString(10, textY, msg);
    }
    else
    {
        textY += 170.0f;
        if (vsync == 0)
        {
            renderer->RenderString(10, textY, "VSync Off");
        }
        else if (vsync == 1)
        {
            renderer->RenderString(10, textY, "VSync On");
        }

        sbFps.Render();
        sbOsc.Render();
        if (oscAmt)
        {
            sbRate.Render();
        }
        else
        {
            renderer->RenderString(10, sbRate.y, "No Osc");
        }
        textY += TEXT_ROW_OFFSET;

        renderer->RenderString(10, textY, osc->GetTypeStr());

        textY += TEXT_ROW_OFFSET;

        if (cpuTimer)
        {
            renderer->RenderString(10, textY, "CPU Timer");
        }
        else
        {
            renderer->SetColor(colors[green]);
            renderer->RenderString(10, textY, "GPU Timer");
            renderer->SetColor(colors[white]);
        }
        textY += TEXT_ROW_OFFSET;
        char buf[256];
        sprintf(buf, "%s : %dx%d", renderer->Name(), winWidth, winHeight);
        renderer->RenderString(10, textY, buf);
    }

    if (showPlots)
    {
        cpuPlot.Render();
        gpuPlot.Render();
    }
}

void Scene::Mouse(int x, int y, bool btnDown)
{
    sbFps.Mouse(x, y, btnDown);
    sbOsc.Mouse(x, y, btnDown);
    sbRate.Mouse(x, y, btnDown);

    if (btnDown)
    {
        fps = sbFps.value;
        oscAmt = sbOsc.value;
        osc->freq = 2.0f * M_PI * sbRate.value;
    }
}

void Scene::SaveCmdLine(string &str)
{
    if (bgnd.gradient)
    {
        str += " -g";
    }

    if (oscAmt > 0)
    {
        str += " -osc " + to_string(oscAmt);
        str += " -oscF " + to_string(osc->freq / M_PI / 2.0f);
    }
    if (bgnd.color[0] != 0.5f || bgnd.color[1] != 0.5f || bgnd.color[2] != 0.5f)
    {
        str += " -color ";
        str += to_string(bgnd.color[0]) + " " + to_string(bgnd.color[1]) + " " + to_string(bgnd.color[2]);
    }

    if (osc->GetType() == OSC_SQUARE)
    {
        str += " -oscT square";
    }
}

void Scene::SpeedUp(float amt)
{
    s1.SpeedUp(amt);
    s2.SpeedUp(amt);
}

void Scene::OscFreqUp(float amt)
{
    osc->freq += amt;
    osc->freq = max(M_PI_4 * 0.5f, osc->freq);
}

void Scene::OscAmtUp(float amt)
{
    oscAmt += amt;
    oscAmt = max(0, oscAmt);
}

void Scene::RefreshSettings()
{
    sbFps.Set(fps);
    sbOsc.Set(oscAmt);
    sbRate.Set(osc->freq / M_PI / 2.0f);
}

void Scene::SaveScreenshot()
{
    static int index = 0;
    char filename[256];
    sprintf(filename, "screenshot%02d.png", index++);

    unsigned int width  = winWidth & ~3;
    unsigned int height = winHeight;

    uint32_t *data = new uint32_t[width * height];

    renderer->Screenshot(width, height, (void *) data);
    SaveImage(filename, width, height, (void *) data);

    delete[] data;
}

void Scene::ToggleTexStripe()
{
    s1.texEnabled = !s1.texEnabled;
}

void Scene::TogglePlots()
{
    showPlots = !showPlots;
}

void Scene::ToggleBars()
{
    showBars = !showBars;
}
