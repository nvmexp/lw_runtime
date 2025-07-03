#pragma once

/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

#include "error.h"
#include "wayland_input.h"
#include "args.h"
#include "timing.h"
#include "lwnGraphics.h"

using namespace lwn;

class LwnRenderBase : public LwnGraphics
{
protected:
    int m_winWidth;
    int m_winHeight;

    int m_textureIndex;

    BlendState          m_blendState;
    ColorState          m_colorState;
    ChannelMaskState    m_cmask;
    DepthStencilState   m_depth;
    MultisampleState    m_multisample;
    PolygonState        m_polygon;

    TimingStats *m_presentTimer;

    virtual CommandHandle RecordCommands(EventInfo *eventInfo, const Texture *windowTexture) = 0;

public:
    virtual ~LwnRenderBase() {}

    virtual void InitGraphics(void *window, int width, int height)
    {
        m_winWidth = width;
        m_winHeight = height;

        InitLwnGraphics(window, width, height, Format::RGBA8);

        // Setup default state
        m_colorState.SetDefaults();
        m_blendState.SetDefaults();
        m_blendState.SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA,
                                  BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
        m_cmask.SetDefaults();
        m_depth.SetDefaults();
        m_depth.SetDepthWriteEnable(LWN_FALSE);    // Not using Depth buffer
        m_multisample.SetDefaults();
        m_polygon.SetDefaults();

        m_presentTimer = NULL;
        if (g_args.m_measureFrameRate) {
            m_presentTimer = new TimingStats("Present");
        }
    }

    void RenderFrame(EventInfo *eventInfo)
    {
        // Acquire next texture from the window.
        mWindow.AcquireTexture(&mWindowTextureAvailableSync, &m_textureIndex);
        const Texture* windowTexture = mWindowTextures + m_textureIndex;

        // Ensure that subsequent rendering commands are not processed by the
        // GPU until the acquired texture is ready for use.
        mQueue.WaitSync(&mWindowTextureAvailableSync);

        CommandHandle cmd = RecordCommands(eventInfo, windowTexture);
        mQueue.SubmitCommands(1, &cmd);

        // Present the texture to the window.  This will also flush the queue
        // so that the GPU will see the commands submitted above.  Due to the
        // WaitSync call in BeginFrame(), the GPU will remain blocked until
        // "textureAvailableSync" has signaled.
        mQueue.PresentTexture(&mWindow, m_textureIndex);  // Implicit flush

        // Wait for the texture to become available to limit the rendering
        // rate to the display refresh rate (vsync).
        mWindowTextureAvailableSync.Wait(LWN_WAIT_TIMEOUT_MAXIMUM);

        if (m_presentTimer) {
            m_presentTimer->AddTimingRecord();
        }
    }

    // Finish is required before resource cleanup in the class destructor. Since
    // derived class destructor is called first, do a Finish() here to make
    // it safer for derived class destructor to do the cleanup.
    virtual void TerminateGraphics()
    {
        mQueue.Finish();
        delete m_presentTimer;
    }
};
