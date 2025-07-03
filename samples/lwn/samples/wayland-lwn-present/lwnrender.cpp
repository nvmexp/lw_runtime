/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

// This sample is for verifying parallel loading of CPU and GPU for
// LWN workloads. In an ideal situation, the following is expected:
//
//     Display: Scanning Frame N-2   [Texture A]
//     GPU:     Working on Frame N-1 [Texture B]
//     CPU:     Working on Frame N   [Texture A]
//
// A complex application could be maxing out both the CPU and the GPU -
// i.e. the time spent on the CPU to compose the frame might be
// 16.6 msec and same might be true for the time spent on the GPU. If
// an application follows the code sample in section "19.1 Presentation
// sample code" of the programming guide, it should be able to achieve
// 60 fps rate even with both CPU and GPU load being 16.6 msec.
//
// While the maximum loading can be 16.6 msec, due to interaction with
// Wayland, the CPU/GPU loads in this sample are initialized to 12 msec.
// If the implementation is incorrect and CPU/GPU are serialized, the
// combined CPU/GPU time with the defaults is well over 16.6 msec such
// that 60 fps would not be possible.
//
// One of the important side factors in achieving in CPU-GPU overlap is
// Weston / Wayland compositor configuration. The default for Weston to
// start compositing its frame is 8 msec (defined in Weston.ini). With this
// default, it is not possible to exceed a CPU load of 8 msec and still
// achieve 60 fps. It is recommended to push set value to 14 msec (Weston
// internally only works with integer values). This value should permit
// this sample to work at 60 fps.
//
// This sample simulates the GPU frame time by rendering an instanced
// primitive and scaling the number of instances to achieve the target
// GPU duration. The CPU frame time is simulated through cpu sleep.

#include <stdint.h>
#include <unistd.h>
#include "../wayland-lwn-example/lwnrender.h"
#include "../wayland-lwn-example/lwnrenderbase.h"
#include "lwnGlslc.h"
#include "timing.h"

static const char *vert_Col =
R"glsl_str(
#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;

layout(location = 0) out vec4 vColor;

void main()
{
    gl_Position = vec4(aPosition,1.0f);
    vColor = aColor;
}
)glsl_str";

static const char *frag_Col =
R"glsl_str(
#version 450 core
layout(location = 0) in vec4 vColor;

layout(location = 0) out vec4 oFrag;

void main()
{
    oFrag = vColor;
}
)glsl_str";

static constexpr int SQ_SIZE = 16;

class LwnRender : public LwnRenderBase
{
public:
    void InitGraphics(void *window, int width, int height) override;

    ~LwnRender()
    {
        delete m_vertexBuffer;
        delete m_counterBuffer;
        delete m_bufferMemoryPool;
    }

private:
    // HW and SW frame times in msec
    float m_swFrameTimeMsec;
    float m_hwFrameTimeMsec;

    CommandHandle RecordCommands(EventInfo *eventInfo, const Texture *windowTexture) override;
    void InsertQuery(int index);
    uint64_t GetQuery(int index);

    GlslcProgram m_glslc;

    LwnMemoryPool *m_bufferMemoryPool;      // For queries and vertices

    // Query related variables
    LwnBuffer            *m_counterBuffer;
    static constexpr int m_maxQueries = 4;   // Exceeds triple buffer count
    int                  m_queryIndex = 0;
    lwn::Sync            *m_timestampCompleteSync[2 * m_maxQueries] = {NULL};
    int                  m_gpuLoad[m_maxQueries];    // Gpu load associated with a query

    // Rendering state for GPU loading
    lwn::Program         *m_pgmCol;
    LwnBuffer            *m_vertexBuffer;

    VertexAttribState    m_vertexAttribs[2]; // Position and Color
    VertexStreamState    m_vertexStreams[2];
};

void LwnRender::InsertQuery(int index)
{
    assert(index < 2 * m_maxQueries);

    lwn::Sync *psync = m_timestampCompleteSync[index];

    mCommandBuffer.ReportCounter(lwn::CounterType::TIMESTAMP,
                   m_counterBuffer->m_gpuAddr + index * sizeof(CounterData));
    mCommandBuffer.FenceSync(psync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE,
                             lwn::SyncFlagBits::FLUSH_FOR_CPU);
}

uint64_t LwnRender::GetQuery(int index)
{
    uint64_t time = 0;
    assert(index < 2 * m_maxQueries);

    lwn::Sync *psync = m_timestampCompleteSync[index];

    // Wait for previously submitted query to be finished and get the time
    psync->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);
    CounterData *counterData = (CounterData *) m_counterBuffer->m_cpuAddr;
    time = mDevice.GetTimestampInNanoseconds(&counterData[index]);

    return time;
}

void LwnRender::InitGraphics(void *window, int width, int height)
{
    LwnRenderBase::InitGraphics(window, width, height);

    m_swFrameTimeMsec = g_args.m_cpuLoad;
    m_hwFrameTimeMsec = g_args.m_gpuLoad;

    // Override blending state
    m_blendState.SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ZERO,
                              BlendFunc::SRC_ALPHA, BlendFunc::ZERO);
    m_colorState.SetBlendEnable(0, true);

    m_pgmCol = m_glslc.CreateProgram(&mDevice, vert_Col, frag_Col);

    m_bufferMemoryPool = new LwnMemoryPool(&mDevice, 1 * 1024 * 1024, MemoryPoolFlags::CPU_UNCACHED |
                                                                      MemoryPoolFlags::GPU_CACHED);

    m_counterBuffer = new LwnBuffer(&mDevice, m_bufferMemoryPool, 1024);  // More than sufficient for
                                                                          // maxQueries
    for (int i = 0; i < 2 * m_maxQueries; i++) {
        m_timestampCompleteSync[i] = new lwn::Sync;
        m_timestampCompleteSync[i]->Initialize(&mDevice);
    }

    // Rendering state
    Format format[] = {Format::RGB32F, Format::RGBA32F};     // Vertex + Color
    int sizes[] = {3 * sizeof(float), 4 * sizeof(float)};    // Sizes of RGB32F and RGBA32F
    int vertexSize = sizes[0] + sizes[1];

    for (int i = 0, offset = 0; i < 2; i++)
    {
        m_vertexStreams[i].SetDefaults();
        m_vertexStreams[i].SetStride(vertexSize);

        m_vertexAttribs[i].SetDefaults();
        m_vertexAttribs[i].SetStreamIndex(i);
        m_vertexAttribs[i].SetFormat(format[i], offset);
        offset += sizes[i];
    }

    // A constant color rectangular bar at the bottom of the window
    m_vertexBuffer = new LwnBuffer(&mDevice, m_bufferMemoryPool, vertexSize * 4);   // Only 4 vertices

    float rectHeight = 64.0f / height;  // 64-pixel high rectangle at the bottom
    float y1 = -1.0f;
    float y2 = y1 + rectHeight;
    float alpha = 64.0f/255.0f;

    struct Vertex {
        float x, y, z;
        float r, g, b, a;
    };
    Vertex *v = (Vertex *) m_vertexBuffer->m_cpuAddr;
    *v++ = {-1.0f, y1, 0.0f, 0.0f, 1.0f, 0.0f, alpha};
    *v++ = { 1.0f, y1, 0.0f, 0.0f, 1.0f, 0.0f, alpha};
    *v++ = {-1.0f, y2, 0.0f, 0.0f, 1.0f, 0.0f, alpha};
    *v++ = { 1.0f, y2, 0.0f, 0.0f, 1.0f, 0.0f, alpha};
}

// Note: Both gpuTime and desiredTime are in nanoSeconds.
inline int GpuLoad(int64_t gpuTime, int loadFactor, int64_t desiredTime)
{
    // GPU load is "K0 + K1 * loadFactor". Given that most of the load comes from gpu loading,
    // ignore K0 for ease of computation and numerically more stable scheme.
    return 0.5f + loadFactor * ((double) desiredTime / (double) gpuTime); // 0.5 for rounding
}

CommandHandle LwnRender::RecordCommands(EventInfo *eventInfo, const Texture *windowTexture)
{
    // Initial time for sw loading
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Until we have been through with round of queries, gpuTime is incorrect.
    static int skip = 0;
    int64_t gpuTimeNs;
    if (skip < m_maxQueries) {
        skip++;
        m_gpuLoad[m_queryIndex] = 50;   // Some dummy load factor
    } else {
        // Before reinserting the query, get elapsed time for a previous frame and obtain gpu load factor
        gpuTimeNs = GetQuery(m_queryIndex + m_maxQueries) - GetQuery(m_queryIndex);
        m_gpuLoad[m_queryIndex] = GpuLoad(gpuTimeNs, m_gpuLoad[m_queryIndex], m_hwFrameTimeMsec * 1.0e06 /* nsec */);
    }

    mCommandBuffer.BeginRecording();

    InsertQuery(m_queryIndex);

    mCommandBuffer.SetRenderTargets(1, &windowTexture, nullptr, nullptr, nullptr);
    mCommandBuffer.SetViewport(0, 0, m_winWidth, m_winHeight);

    mCommandBuffer.BindBlendState(&m_blendState);
    mCommandBuffer.BindColorState(&m_colorState);
    mCommandBuffer.BindChannelMaskState(&m_cmask);
    mCommandBuffer.BindDepthStencilState(&m_depth);
    mCommandBuffer.BindMultisampleState(&m_multisample);
    mCommandBuffer.BindPolygonState(&m_polygon);
    mCommandBuffer.SetSampleMask(~0);

    // Clear frame
    {
        float clearValue = 0.4f;
        float clearColor[] = {0.0f, clearValue, clearValue, 0.5f};
        mCommandBuffer.SetScissor(0.0f, 0.0f, m_winWidth, m_winHeight);
        mCommandBuffer.ClearColor(0, clearColor, ClearColorMask::RGBA);
    }

    // GPU loading bar
    {
        mCommandBuffer.BindProgram(m_pgmCol, ShaderStageBits::ALL_GRAPHICS_BITS);

        mCommandBuffer.BindVertexAttribState(2, m_vertexAttribs);
        mCommandBuffer.BindVertexStreamState(2, m_vertexStreams);

        mCommandBuffer.BindVertexBuffer(0, m_vertexBuffer->m_gpuAddr, m_vertexBuffer->m_size);
        mCommandBuffer.BindVertexBuffer(1, m_vertexBuffer->m_gpuAddr, m_vertexBuffer->m_size);

        int instanceCount = m_gpuLoad[m_queryIndex];
        mCommandBuffer.DrawArraysInstanced(DrawPrimitive::TRIANGLE_STRIP, 0, 4, 0, instanceCount);
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

        // Restore Scissor
        mCommandBuffer.SetScissor(0, 0, m_winWidth, m_winHeight);
    }

    InsertQuery(m_queryIndex + m_maxQueries);

    // Simulate time spent in application by sleeping
    clock_gettime(CLOCK_MONOTONIC, &end);
    float sleepTimeMsec = m_swFrameTimeMsec - timespec_sub(end, start);
    if (sleepTimeMsec > 0.0f) {
        usleep(sleepTimeMsec * 1000.0f /* usec */);
    }

    //printf("GpuTime = %f, SleepTime = %f, LoadFactor = %d\n", gpuTimeNs / 1.0e06, sleepTimeMsec, m_gpuLoad[m_queryIndex]);

    if (++m_queryIndex == m_maxQueries) {
        m_queryIndex = 0;
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
