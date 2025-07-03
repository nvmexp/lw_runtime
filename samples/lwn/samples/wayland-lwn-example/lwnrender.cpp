/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

#include "lwnrender.h"
#include "lwnrenderbase.h"

static constexpr int SQ_SIZE   = 8;
static constexpr int LWR_SIZE  = 8;
static constexpr int LWR_WIDTH = 2;

class LwnRender : public LwnRenderBase
{
private:
    CommandHandle RecordCommands(EventInfo *eventInfo, const Texture *windowTexture);
};

CommandHandle LwnRender::RecordCommands(EventInfo *eventInfo, const Texture *windowTexture)
{
    mCommandBuffer.BeginRecording();

    mCommandBuffer.SetRenderTargets(1, &windowTexture, nullptr, nullptr, nullptr);
    mCommandBuffer.SetViewport(0, 0, m_winWidth, m_winHeight);

    // Monotonically keep increasing/decreasing the background color
    {
        static int inc = 1;
        static int count = 0;
        static int numClearColors = 255*2;

        float clearValue = (float) count / (float) numClearColors;
        float clearColor[] = {0.0f, clearValue, clearValue, 0.5f};

        mCommandBuffer.SetScissor(0, 0, m_winWidth, m_winHeight);
        mCommandBuffer.ClearColor(0, clearColor, ClearColorMask::RGBA);

        // Setup next frame
        count += inc;
        if (count == numClearColors) {
            count = 0;
        }
    }

    // Animating square
    {
        static int loc = 0;
        static int maxOffset = 10;
        static int inc = 1;

        int x = (int) (eventInfo->leftMouseX);
        int y = m_winHeight - (int) (eventInfo->leftMouseY);
        x += loc;

        float clearColor[] = {1.0f, 0.0f, 0.0f, 0.5f};

        mCommandBuffer.SetScissor(x - SQ_SIZE, y - SQ_SIZE, 2 * SQ_SIZE, 2 * SQ_SIZE);
        mCommandBuffer.ClearColor(0, clearColor, ClearColorMask::RGBA);

        // Setup next frame
        if ((loc == maxOffset) || (loc == -maxOffset)) {
            inc *= -1;
        }
        loc += inc;
    }

    // Cursor: Display only if window is active
    if (eventInfo->active) {
        int x = (int) (eventInfo->mouseX);
        int y = m_winHeight - (int) (eventInfo->mouseY);

        float clearColor[] = {0.0f, 1.0f, 0.0f, 0.5f};

        mCommandBuffer.SetScissor(x - LWR_SIZE, y - LWR_WIDTH, 2 * LWR_SIZE, 2 * LWR_WIDTH);
        mCommandBuffer.ClearColor(0, clearColor, ClearColorMask::RGBA);
        mCommandBuffer.SetScissor(x - LWR_WIDTH, y - LWR_SIZE, 2 * LWR_WIDTH, 2 * LWR_SIZE);
        mCommandBuffer.ClearColor(0, clearColor, ClearColorMask::RGBA);
    }

    return mCommandBuffer.EndRecording();       // Return commandHandle
}

static LwnRender *s_lwnRender;

void InitGraphics(void *window, int width, int height)
{
    s_lwnRender = new LwnRender;
    s_lwnRender->InitGraphics(window, width, height);
}

void RenderFrame(EventInfo *eventInfo)
{
    if (s_lwnRender) {
        s_lwnRender->RenderFrame(eventInfo);
    }
}

// TerminateGraphics should ensure all rendering operations have completed and
// clean up all LWN resources (including Window textures) such that when
// Wayland window is destroyed, it should not have any pending LWN internal resources
// associated with the window.
void TerminateGraphics()
{
    LwnRender *lwnRender = s_lwnRender;
    s_lwnRender = nullptr;      // Prevents frames from being rendered in the middle of cleanup

    lwnRender->TerminateGraphics();
    delete lwnRender;
}
