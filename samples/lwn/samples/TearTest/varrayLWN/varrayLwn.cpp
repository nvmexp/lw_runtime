/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "../varray/varray.h"
#include "../varrayGL/shaders.glsl"

#include <vector>
#include "lwnGraphics.h"
#include "lwnGlslc.h"

using namespace lwn;

class RendererVarrayLWN : public RendererVarray, public LwnGraphics
{
public:
    const char *Name()
    {
        return "LWN";
    }

    void *Init(int winWidth, int winHeight, int maxVertexCount)
    {
        m_init = true;
        m_winWidth = winWidth;
        m_winHeight = winHeight;

        // Init routine needs to do the following tasks:
        // (Note: Textures are created dynamically)

        // 1. Initialize the API

        InitLwnGraphics(winWidth, winHeight, Format::RGBA8);

        // Create a pool for texture even though we are not allocating textures right now
        // Note: Seems like there is a catch-22 - CPU_NO_ACCESS triggers a debug layer error
        // but cannot create the pool with CPU access.
        m_textureMemoryPool = new LwnMemoryPool(&mDevice, 32 * 1024 * 1024,
                                  MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED);

        m_textureDescPool = new LwnDescriptorPool<lwn::TexturePool>(&mDevice, TEXTURE, NUM_DESC);
        m_samplerDescPool = new LwnDescriptorPool<lwn::SamplerPool>(&mDevice, SAMPLER, NUM_DESC);

        // 2. Set up default BlendFunc. In case of LWN setup all other default state too.

        m_colorState.SetDefaults();
        m_blendState.SetDefaults();
        m_blendState.SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA,
                                  BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);

        m_cmask.SetDefaults();
        m_depth.SetDefaults();
        m_depth.SetDepthWriteEnable(LWN_FALSE);    // Not using Depth buffer
        m_multisample.SetDefaults();
        m_polygon.SetDefaults();

        // 3. Create 3 shaders corresponding to PGM_COL, PGM_TEX, PGM_BITMAP

        m_programIds[PGM_COL] = m_glslc.CreateProgram(&mDevice, vert_ColTex, frag_Col);
        m_programIds[PGM_TEX] = m_glslc.CreateProgram(&mDevice, vert_ColTex, frag_ColTex);
        m_programIds[PGM_BITMAP] = m_glslc.CreateProgram(&mDevice, vert_ColTex, frag_Bitmap);

        // 4. Create query objects for timestamp. Also, queue a query.
        //    After this, we always obtain this query and queue a new one.

        // These are setup as needed. Also see setup of counterBuffer later in this routine

        // 5. Allocate a large vertex buffer and setup vertex state.

        Format format[] = {Format::RGB32F, Format::RGBA32F, Format::RG32F};     // VCT
        int sizes[] = {sizeof(Vertex3), sizeof(Color4), sizeof(Tex2)};

        for (int i = 0, offset = 0; i < 3; i++)
        {
            m_vertexStreams[i].SetDefaults();
            m_vertexStreams[i].SetStride(sizeof(VertexVCT));

            m_vertexAttribs[i].SetDefaults();
            m_vertexAttribs[i].SetStreamIndex(i);
            m_vertexAttribs[i].SetFormat(format[i], offset);
            offset += sizes[i];
        }

        // Memory pool for vertex
        m_bufferMemoryPool = new LwnMemoryPool(&mDevice, 32 * 1024 * 1024, MemoryPoolFlags::CPU_UNCACHED |
                                                                           MemoryPoolFlags::GPU_CACHED);
        m_vertexBuffer = new LwnBuffer(&mDevice, m_bufferMemoryPool, sizeof(VertexVCT) * maxVertexCount);

        // Use the same pool for counter buffer
        m_counterBuffer = new LwnBuffer(&mDevice, m_bufferMemoryPool, 1024);  // More than sufficient for
                                                                              // maxQueries

        return m_vertexBuffer->m_cpuAddr;
    }

    void BeginFrame()
    {
        assert(!m_inFrame);
        m_inFrame = true;

        // It is assumed that there are several time stamp queries in a frame and their
        // id is inferred from sequence in the frame.
        m_queryIndex = 0;

        // Acquire next texture from the window.
        mWindow.AcquireTexture(&mWindowTextureAvailableSync, &m_textureIndex);

        // At this point, GPU has completed rendering the previous frame to
        // the acquired texture, but the acquired texture may still be used
        // by the display.

        // Ensure that subsequent rendering commands are not processed by the
        // GPU until the acquired texture is ready for use.

        mQueue.WaitSync(&mWindowTextureAvailableSync);
        mCommandBuffer.BeginRecording();
        const Texture* windowTexture = mWindowTextures + m_textureIndex;
        mCommandBuffer.SetRenderTargets(1, &windowTexture, nullptr, nullptr, nullptr);

        mCommandBuffer.SetScissor(0, 0, m_winWidth, m_winHeight);
        mCommandBuffer.SetViewport(0, 0, m_winWidth, m_winHeight);

        mCommandBuffer.BindProgram(m_programIds[PGM_COL], ShaderStageBits::ALL_GRAPHICS_BITS);
        mCommandBuffer.BindBlendState(&m_blendState);
        mCommandBuffer.BindColorState(&m_colorState);   // Has blendEnable setup correctly

        mCommandBuffer.BindChannelMaskState(&m_cmask);
        mCommandBuffer.BindDepthStencilState(&m_depth);
        mCommandBuffer.BindMultisampleState(&m_multisample);
        mCommandBuffer.BindPolygonState(&m_polygon);
        mCommandBuffer.SetSampleMask(~0);

        mCommandBuffer.BindVertexAttribState(3, m_vertexAttribs);
        mCommandBuffer.BindVertexStreamState(3, m_vertexStreams);

        mCommandBuffer.BindVertexBuffer(0, m_vertexBuffer->m_gpuAddr, m_vertexBuffer->m_size);
        mCommandBuffer.BindVertexBuffer(1, m_vertexBuffer->m_gpuAddr, m_vertexBuffer->m_size);
        mCommandBuffer.BindVertexBuffer(2, m_vertexBuffer->m_gpuAddr, m_vertexBuffer->m_size);

        mCommandBuffer.SetTexturePool(&m_textureDescPool->m_apiDescPool);
        mCommandBuffer.SetSamplerPool(&m_samplerDescPool->m_apiDescPool);
    }

    void EndFrame()
    {
        assert(m_inFrame);
        m_inFrame = false;

        CommandHandle commandHandle = mCommandBuffer.EndRecording();
        mQueue.SubmitCommands(1, &commandHandle);

        // Present the texture to the window.  This will also flush the queue
        // so that the GPU will see the commands submitted above.  Due to the
        // WaitSync call in BeginFrame(), the GPU will remain blocked until
        // "textureAvailableSync" has signaled.
        mQueue.PresentTexture(&mWindow, m_textureIndex);  // Implicit flush

        // Wait for the texture to become available to limit the rendering
        // rate to the display refresh rate (vsync).
        mWindowTextureAvailableSync.Wait(LWN_WAIT_TIMEOUT_MAXIMUM);
    }

    void Viewport(int x, int y, uint32_t width, uint32_t height)
    {
        // No need to do anything. BeginFrame() functions sets up
        // Viewport and Scissor based on window size and that is
        // good enough for this application.
    }

    void Clear()
    {
        // Ignore clear calls when not within a frame
        if (m_inFrame)
        {
            mCommandBuffer.ClearColor(0, &m_clearColor.r, ClearColorMask::RGBA);
        }
    }

    void SetClearColor(const float *color)
    {
        m_clearColor = *(Color4 *) color;
    }

    void UseProgram(ShaderProgram pgm)
    {
        assert(m_inFrame);
        mCommandBuffer.BindProgram(m_programIds[pgm], ShaderStageBits::ALL_GRAPHICS_BITS);
    }

    void DrawPrimitive(PrimMode mode, int32_t first, uint32_t count)
    {
        assert(m_inFrame);
        // Fortunately the "mode" values match - we can just cast
        lwn::DrawPrimitive drawPrimitive = static_cast<DrawPrimitive::Enum> (mode);
        mCommandBuffer.DrawArrays(drawPrimitive, first, count);
    }

    void DrawPrimitiveInstanced(PrimMode mode, int32_t first, uint32_t count, int instanceCount)
    {
        assert(m_inFrame);
        lwn::DrawPrimitive drawPrimitive = static_cast<DrawPrimitive::Enum> (mode);
        mCommandBuffer.DrawArraysInstanced(drawPrimitive, first, count, 0, instanceCount);
    }

    unsigned int CreateTex(TexFormat texFormat, unsigned int width, unsigned int height, void *data)
    {
        Format::Enum texFormatLwn = (texFormat == TEX_R8) ? lwn::Format::R8 : lwn::Format::RGBA8;

        LwnTexture *lwnTexture = new LwnTexture(&mDevice, m_textureMemoryPool, texFormatLwn,
                                                width, height, data);
        m_lwnTextures.push_back(lwnTexture);

        // Allocate a sampler and corresponding entry in descrPool
        lwn::SamplerBuilder samplerBuilder;
        samplerBuilder.SetDefaults();
        samplerBuilder.SetDevice(&mDevice);
        samplerBuilder.SetMinMagFilter(lwn::MinFilter::LINEAR, lwn::MagFilter::LINEAR);

        int samplerIndex = m_samplerDescPool->Alloc();
        m_samplerDescPool->m_apiDescPool.RegisterSamplerBuilder(samplerIndex, &samplerBuilder);

        // Allocate a textureView and corresponding entry in descrPool

        int textureIndex = m_textureDescPool->Alloc();
        m_textureDescPool->m_apiDescPool.RegisterTexture(textureIndex, &lwnTexture->m_apiTexture, NULL);

        // Obtain a texture handle
        TextureHandle textureHandle = mDevice.GetTextureHandle(textureIndex, samplerIndex);
        m_textureHandles.push_back(textureHandle);

        return m_lwnTextures.size() - 1;         // Return "index" to the lwnTextures
    }

    void BindTexture(unsigned int index)
    {
        assert(m_inFrame);

        // Unbinding a texture handle is not supported except by creating yet
        // another texture handle. Just skip the binding to "0" and rely on
        // the shader program to not make texture accesses.
        if (index) {
            mCommandBuffer.BindTexture(ShaderStage::FRAGMENT, 0, m_textureHandles[index]);
        }
    }

    // For Enable/Disable functions, RenderCap enum typedefinition only has BLEND. When
    // it is extended, this code needs to be modified.
    void Enable(RenderCap cap)
    {
        assert(m_inFrame);
        assert(cap == BLEND);

        if (cap == BLEND) {
            SetBlendEnable(true);
        }
    }

    void Disable(RenderCap cap)
    {
        assert(m_inFrame);
        assert(cap == BLEND);

        if (cap == BLEND) {
            SetBlendEnable(false);
        }
    }

    void InsertQuery(int index)
    {
        assert(m_inFrame);
        assert(index < m_maxQueries);

        lwn::Sync **psync = &m_timestampCompleteSync[index];

        if (psync[0] == NULL)
        {
            // Create a sync object for the query. Since first time,
            // return the current time.
            psync[0] = new lwn::Sync;
            psync[0]->Initialize(&mDevice);
        }

        mCommandBuffer.ReportCounter(lwn::CounterType::TIMESTAMP,
                       m_counterBuffer->m_gpuAddr + index * sizeof(CounterData));
        mCommandBuffer.FenceSync(psync[0], SyncCondition::ALL_GPU_COMMANDS_COMPLETE,
                                 lwn::SyncFlagBits::FLUSH_FOR_CPU);
    }

    unsigned __int64 GetQuery(int index)
    {
        unsigned __int64 time = 0;
        assert(index < m_maxQueries);

        lwn::Sync **psync = &m_timestampCompleteSync[index];

        // If no query has been created, return the current timestamp

        if (psync[0] == NULL) {
            time = mDevice.GetLwrrentTimestampInNanoseconds();
        } else {
            // Wait for previously submitted query to be finished and get the time
            psync[0]->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);
            CounterData *counterData = (CounterData *) m_counterBuffer->m_cpuAddr;
            time = mDevice.GetTimestampInNanoseconds(&counterData[index]);
        }
        return time;
    }

    void Screenshot(unsigned int width, unsigned int height, void *data, bool readAlpha)
    {
        // Not supported for LWN port
    }

    void SetVSync(bool v)
    {
        mWindow.SetPresentInterval(v);
    }

    void Finish()
    {
        // Unlike GL, this probably does not handle finish commands that in the command
        // buffer but not yet submitted. But this is good enough for this applciation.
        assert(!m_inFrame);
        mQueue.Finish();
    }

    void BlitFrontToBackBuffer()
    {
        // Not needed for LWN port - SQA functionality that is not enabled.
    }

    ~RendererVarrayLWN()
    {
        if (m_init)
        {
            delete m_vertexBuffer;
            delete m_counterBuffer;
            delete m_bufferMemoryPool;
            delete m_textureMemoryPool;
            delete m_textureDescPool;
            delete m_samplerDescPool;
        }
    }

private:
    bool m_init = false;
    bool m_inFrame = false;

    int m_winWidth;
    int m_winHeight;
    Color4 m_clearColor;

    int m_textureIndex;

    BlendState          m_blendState;
    ColorState          m_colorState;
    ChannelMaskState    m_cmask;
    DepthStencilState   m_depth;
    MultisampleState    m_multisample;
    PolygonState        m_polygon;

    // Query related variables
    LwnBuffer           *m_counterBuffer;
    static const int    m_maxQueries = 6;
    int                 m_queryIndex = 0;
    lwn::Sync           *m_timestampCompleteSync[m_maxQueries] = {NULL};

    LwnMemoryPool       *m_bufferMemoryPool;
    LwnBuffer           *m_vertexBuffer;

    VertexAttribState   m_vertexAttribs[3];     // Vertex of type VCT with 3 components
    VertexStreamState   m_vertexStreams[3];

    lwn::Program *m_programIds[PGM_MAX];

    LwnMemoryPool       *m_textureMemoryPool;
    std::vector<LwnTexture *> m_lwnTextures = {NULL};  // Array of textures.  Index = 0 corresponds to "No" texture
    std::vector<TextureHandle> m_textureHandles = {NULL};

    static constexpr int NUM_DESC = 128;
    LwnDescriptorPool<lwn::TexturePool>  *m_textureDescPool;
    LwnDescriptorPool<lwn::SamplerPool>  *m_samplerDescPool;

    GlslcProgram m_glslc;

    void SetBlendEnable(bool flag)
    {
        m_colorState.SetBlendEnable(0, flag);
        mCommandBuffer.BindColorState(&m_colorState);
    }
};

Renderer *CreateVarrayLWNRenderer()
{
    return new RendererVarrayLWN;
}
